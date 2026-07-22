from __future__ import annotations

from typing import Any

import torch

from .plugins import RecoveryRequestContext
from .radix_backend import (
    DeviceKVRef,
    RadixKVTransferBackend,
    RoPEConfig,
)
from .raw_rope import (
    RAW_ROPE_PLUGIN_NAME,
    RawRoPERecoveryRequest,
    RawRoPERecoveryUnavailable,
    select_contiguous_segments,
)
from .request import ApproxKVRequestOperation
from .types import (
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
    token_ids_hash,
)


class ApproxKVRegistrationError(RuntimeError):
    pass


def _allocator(tree_cache: Any) -> Any:
    allocator = tree_cache.token_to_kv_pool_allocator
    if allocator is None:
        raise RuntimeError("approximate KV requires a token allocator")
    return allocator


def allocate_recovery_slots(tree_cache: Any, num_tokens: int):
    """Allocate approximate-recovery slots after evicting exact Radix victims.

    The naive ``allocator.alloc(num_tokens)`` call bypasses SGLang's standard
    ``evict_from_tree_cache -> allocator.alloc`` ordering, which is safe only
    when the pool has slack. Under the high-pressure R0 benchmark contract
    (multi-object working sets that push actual reusable rho above 1x) this
    causes allocator exhaustion/OOM instead of evicting exact Radix victims
    first, so both the source-registration and the target-restore call sites
    below must route through this shared helper.
    """
    allocator = _allocator(tree_cache)
    if (
        hasattr(tree_cache, "evict")
        and hasattr(tree_cache, "is_chunk_cache")
        and hasattr(allocator, "available_size")
    ):
        # Local import avoids the common.py -> approx_kv.runtime import cycle.
        from sglang.srt.mem_cache.common import evict_from_tree_cache

        evict_from_tree_cache(tree_cache, num_tokens)
    return allocator.alloc(num_tokens)


def _release_device_ref(allocator: Any):
    def release(backend_ref: object, residency: ResidencyTier) -> None:
        if residency != ResidencyTier.DEVICE or not isinstance(
            backend_ref,
            DeviceKVRef,
        ):
            raise TypeError("invalid approximate KV device reference")
        allocator.free(backend_ref.indices)

    return release


def _segment_key(
    *,
    tokens: tuple[int, ...],
    content_hash: str,
    model_fingerprint: str,
    cache_dtype: str,
) -> KVSegmentKey:
    return KVSegmentKey(
        content_hash=content_hash,
        token_hash=token_ids_hash(tokens),
        token_count=len(tokens),
        model_fingerprint=model_fingerprint,
        cache_dtype=cache_dtype,
        kind=SegmentKind.ARTIFACT,
    )


def register_request_segments(tree_cache: Any, req: Any) -> int:
    try:
        return _register_request_segments(tree_cache, req)
    except (KeyError, MemoryError, RuntimeError, TypeError, ValueError) as exc:
        manager = getattr(tree_cache, "approx_kv", None)
        if manager is not None:
            manager.record_request("register", "error")
        raise ApproxKVRegistrationError(
            "failed to register approximate KV source segments"
        ) from exc


def _register_request_segments(tree_cache: Any, req: Any) -> int:
    metadata = getattr(req, "approx_kv_metadata", None)
    manager = getattr(tree_cache, "approx_kv", None)
    if (
        metadata is None
        or metadata.operation != ApproxKVRequestOperation.REGISTER
        or manager is None
        or not manager.config.core_enabled
    ):
        return 0
    if req.req_pool_idx is None or req.kv is None:
        raise RuntimeError("request KV must exist before source registration")

    allocator = _allocator(tree_cache)
    registered = 0
    for segment in metadata.segments:
        target_end = segment.target_end
        if target_end > req.effective_kv_committed_len():
            raise ValueError("source segment exceeds committed request KV")
        tokens = tuple(
            int(token)
            for token in req.full_untruncated_fill_ids[
                segment.target_start : target_end
            ]
        )
        key = _segment_key(
            tokens=tokens,
            content_hash=segment.content_hash,
            model_fingerprint=metadata.model_fingerprint,
            cache_dtype=metadata.cache_dtype,
        )
        source_indices = tree_cache.req_to_token_pool.req_to_token[
            req.req_pool_idx,
            segment.target_start : target_end,
        ].clone()

        if manager.config.host_residency_enabled:
            load_result = manager.export_to_host(DeviceKVRef(source_indices))
            handle = manager.register_segment(
                key=key,
                token_ids=tokens,
                source_start=segment.target_start,
                residency=ResidencyTier.HOST,
                backend_ref=load_result.backend_ref,
                release_backend=load_result.release_backend,
            )
            if handle is None:
                if load_result.release_backend is not None:
                    load_result.release_backend(
                        load_result.backend_ref,
                        ResidencyTier.HOST,
                    )
                raise RuntimeError("approximate KV manager rejected source segment")
            manager.record_host_export(
                load_result.num_tokens,
                load_result.bytes_transferred,
            )
        else:
            target_indices = allocate_recovery_slots(tree_cache, segment.length)
            if target_indices is None or len(target_indices) != segment.length:
                if target_indices is not None:
                    allocator.free(target_indices)
                raise MemoryError(
                    "unable to allocate device slots for approximate KV"
                )
            try:
                allocator.get_kvcache().move_kv_cache(
                    target_indices,
                    source_indices,
                )
            except Exception:
                allocator.free(target_indices)
                raise
            handle = manager.register_segment(
                key=key,
                token_ids=tokens,
                source_start=segment.target_start,
                residency=ResidencyTier.DEVICE,
                backend_ref=DeviceKVRef(target_indices),
                release_backend=_release_device_ref(allocator),
            )
            if handle is None:
                allocator.free(target_indices)
                raise RuntimeError("approximate KV manager rejected source segment")
        registered += segment.length

    manager.record_request("register", "success")
    req.approx_kv_registered_tokens = registered
    return registered


def restore_request_prefix(tree_cache: Any, req: Any) -> bool:
    metadata = getattr(req, "approx_kv_metadata", None)
    manager = getattr(tree_cache, "approx_kv", None)
    if (
        metadata is None
        or metadata.operation != ApproxKVRequestOperation.REUSE
        or manager is None
        or not manager.config.core_enabled
    ):
        return False
    if req.needs_host_load_back():
        manager.record_request("reuse", "exact_host_preferred")
        return False

    # Explicit plugin gate (Phase 4 R0): the raw+RoPE recovery algorithm is
    # only exercised on the request path when it has been registered under
    # RAW_ROPE_PLUGIN_NAME, which manager.py only does when
    # config.raw_rope_plugin_enabled is set. This keeps common-core's
    # generic store/transfer machinery free of any paper-specific policy
    # by default, matching the other Phase 4 research branches that will
    # register their own plugin under the same manager/store instead.
    if RAW_ROPE_PLUGIN_NAME not in manager.plugins.names():
        manager.record_request("reuse", "recovery_plugin_disabled")
        return False
    plugin = manager.plugins.get(RAW_ROPE_PLUGIN_NAME)

    reusable_limit = len(req.full_untruncated_fill_ids) - 1
    exact_length = len(req.prefix_indices)

    # Best-effort promote to device residency exactly the segments the
    # plugin could plausibly use (same contiguous-run selection the plugin
    # itself performs -- see select_contiguous_segments docstring). This is
    # the only step that needs manager/backend I/O access, which the
    # RecoveryPlugin protocol intentionally does not expose to build_plan.
    active_segments = select_contiguous_segments(
        metadata.segments,
        exact_length,
        reusable_limit,
    )
    if not active_segments or active_segments[0].target_start > exact_length:
        manager.record_request("reuse", "exact")
        return False
    restore_end = min(active_segments[-1].target_end, reusable_limit)
    restore_length = restore_end - exact_length
    if restore_length <= 0:
        manager.record_request("reuse", "exact")
        return False

    for segment in active_segments:
        overlap_start = max(segment.target_start, exact_length)
        overlap_end = min(segment.target_end, restore_end)
        if overlap_end <= overlap_start:
            continue
        tokens = tuple(
            int(token)
            for token in req.full_untruncated_fill_ids[
                segment.target_start : segment.target_end
            ]
        )
        key = _segment_key(
            tokens=tokens,
            content_hash=segment.content_hash,
            model_fingerprint=metadata.model_fingerprint,
            cache_dtype=metadata.cache_dtype,
        )
        handle = manager.store.lookup(key)
        if handle is None:
            # Missing coverage: leave it to the plugin's build_plan to
            # raise RawRoPERecoveryUnavailable for this segment.
            continue
        try:
            manager.ensure_device(handle)
        except Exception:
            manager.record_fallback("residency_load_failed", restore_length)
            manager.record_request("reuse", "dense_fallback")
            return False

    context = RecoveryRequestContext(
        request_id=str(getattr(req, "rid", None) or "unknown"),
        target_token_ids=tuple(
            int(token) for token in req.full_untruncated_fill_ids
        ),
        exact_prefix_length=exact_length,
        custom_metadata={
            RawRoPERecoveryRequest.KEY: RawRoPERecoveryRequest(
                segments=active_segments,
                model_fingerprint=metadata.model_fingerprint,
                cache_dtype=metadata.cache_dtype,
            ),
        },
    )
    try:
        plan = plugin.build_plan(context, manager.store)
    except RawRoPERecoveryUnavailable as exc:
        manager.record_fallback(str(exc) or "raw_rope_unavailable", restore_length)
        manager.record_request("reuse", "dense_fallback")
        return False
    # The plugin is the authority on the final plan shape; re-derive
    # restore_length from it rather than trusting the orchestration-side
    # estimate above, in case a plugin narrows the span further.
    restore_length = len(plan.target_token_ids)
    if restore_length <= 0:
        manager.record_request("reuse", "exact")
        return False

    rope_config = manager.rope_config or RoPEConfig(
        rotary_dim=0,
        base=10000.0,
        is_neox_style=True,
    )
    if rope_config.rotary_dim == 0 and any(
        span.rope_delta != 0 for span in plan.copied_spans
    ):
        manager.record_fallback("rope_config_unavailable", restore_length)
        manager.record_request("reuse", "dense_fallback")
        return False

    allocator = _allocator(tree_cache)
    restored_indices = allocate_recovery_slots(tree_cache, restore_length)
    if restored_indices is None or len(restored_indices) != restore_length:
        if restored_indices is not None:
            allocator.free(restored_indices)
        manager.record_fallback("device_allocation_failed", restore_length)
        manager.record_request("reuse", "dense_fallback")
        return False

    fallback_reasons: list[str] = []
    backend = RadixKVTransferBackend(
        allocator=allocator,
        target_indices=lambda start, length: restored_indices[
            start : start + length
        ],
        dense_prefill=lambda start, length, reason: fallback_reasons.append(reason),
        rope=rope_config,
    )
    try:
        stats = manager.execute(plan, backend)
    except Exception:
        allocator.free(restored_indices)
        raise
    if fallback_reasons or stats.recomputed_tokens:
        allocator.free(restored_indices)
        manager.record_request("reuse", "dense_fallback")
        return False

    req.prefix_indices = torch.cat(
        (
            req.prefix_indices,
            restored_indices.to(
                device=req.prefix_indices.device,
                dtype=req.prefix_indices.dtype,
            ),
        )
    )
    req.approx_kv_restored_len = restore_length
    req.approx_kv_stats = stats
    manager.record_request("reuse", "success")
    return True
