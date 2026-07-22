from __future__ import annotations

import logging
from typing import Any

import torch

from .kvcomm import (
    KVCOMMAction,
    KVCOMMCapabilityError,
    KVCOMMInvariantError,
    KVCOMMObservedSegment,
    KVCOMMReconstructionPlan,
    KVCOMMRecoveryPlugin,
    KVCOMMRequestSpec,
    KVCOMMRuntimeCapabilities,
    execute_kvcomm_reconstruction,
    make_kvcomm_segment_key,
)
from .plugins import RecoveryRequestContext
from .radix_backend import (
    DeviceKVRef,
    RadixKVTransferBackend,
    RoPEConfig,
)
from .request import ApproxKVRequestOperation
from .types import (
    KVReusePlan,
    KVSegmentKey,
    RecoveryMode,
    ResidencyTier,
    SegmentKind,
    TransferSpan,
    token_ids_hash,
)

logger = logging.getLogger(__name__)


class ApproxKVRegistrationError(RuntimeError):
    pass


def _allocator(tree_cache: Any) -> Any:
    allocator = tree_cache.token_to_kv_pool_allocator
    if allocator is None:
        raise RuntimeError("approximate KV requires a token allocator")
    return allocator


def allocate_recovery_slots(tree_cache: Any, num_tokens: int):
    """Allocate recovery slots after evicting exact Radix victims."""
    allocator = _allocator(tree_cache)
    if (
        hasattr(tree_cache, "evict")
        and hasattr(tree_cache, "is_chunk_cache")
        and hasattr(allocator, "available_size")
    ):
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
    kind: SegmentKind = SegmentKind.ARTIFACT,
) -> KVSegmentKey:
    return KVSegmentKey(
        content_hash=content_hash,
        token_hash=token_ids_hash(tokens),
        token_count=len(tokens),
        model_fingerprint=model_fingerprint,
        cache_dtype=cache_dtype,
        kind=kind,
    )


def register_request_segments(tree_cache: Any, req: Any) -> int:
    try:
        metadata = getattr(req, "approx_kv_metadata", None)
        if metadata is not None and metadata.plugin == "kvcomm":
            return _register_kvcomm_request(tree_cache, req)
        return _register_request_segments(tree_cache, req)
    except (KeyError, MemoryError, RuntimeError, TypeError, ValueError) as exc:
        manager = getattr(tree_cache, "approx_kv", None)
        if manager is not None:
            metadata = getattr(req, "approx_kv_metadata", None)
            operation = (
                str(getattr(metadata.operation, "value", metadata.operation))
                if metadata is not None
                else "register"
            )
            manager.record_request(operation, "error")
        raise ApproxKVRegistrationError(
            "failed to register approximate KV source segments"
        ) from exc


def _copy_segment_to_store(
    *,
    tree_cache: Any,
    req: Any,
    segment: Any,
    key: KVSegmentKey,
    tokens: tuple[int, ...],
    force_device: bool = False,
) -> Any:
    manager = tree_cache.approx_kv
    allocator = _allocator(tree_cache)
    target_end = segment.target_end
    source_indices = tree_cache.req_to_token_pool.req_to_token[
        req.req_pool_idx,
        segment.target_start : target_end,
    ].clone()

    if manager.config.host_residency_enabled and not force_device:
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
        return handle

    target_indices = allocator.alloc(segment.length)
    if target_indices is None or len(target_indices) != segment.length:
        if target_indices is not None:
            allocator.free(target_indices)
        raise MemoryError("unable to allocate device slots for approximate KV")
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
    return handle


def _observed_kvcomm_segments(
    tree_cache: Any,
    req: Any,
    metadata: Any,
    spec: KVCOMMRequestSpec,
) -> tuple[KVCOMMObservedSegment, ...]:
    descriptors = {descriptor.segment_index: descriptor for descriptor in spec.segments}
    observed = []
    for index, segment in enumerate(metadata.segments):
        target_end = segment.target_end
        if target_end > req.effective_kv_committed_len():
            raise ValueError("KVCOMM segment exceeds committed request KV")
        tokens = tuple(
            int(token)
            for token in req.full_untruncated_fill_ids[
                segment.target_start : target_end
            ]
        )
        key = make_kvcomm_segment_key(
            tokens=tokens,
            content_hash=segment.content_hash,
            model_fingerprint=metadata.model_fingerprint,
            cache_dtype=metadata.cache_dtype,
            kind=SegmentKind.KVCOMM_BASE,
        )
        indices = tree_cache.req_to_token_pool.req_to_token[
            req.req_pool_idx,
            segment.target_start : target_end,
        ].clone()
        observed.append(
            KVCOMMObservedSegment(
                descriptor=descriptors[index],
                key=key,
                token_ids=tokens,
                positions=tuple(range(segment.target_start, target_end)),
                indices=indices,
            )
        )
    return tuple(observed)


def _register_kvcomm_request(tree_cache: Any, req: Any) -> int:
    metadata = getattr(req, "approx_kv_metadata", None)
    manager = getattr(tree_cache, "approx_kv", None)
    if (
        metadata is None
        or metadata.plugin != "kvcomm"
        or manager is None
        or not manager.config.core_enabled
    ):
        return 0
    if req.req_pool_idx is None or req.kv is None:
        raise RuntimeError("request KV must exist before KVCOMM update")
    if manager.config.host_residency_enabled or manager.config.async_prefetch_enabled:
        raise KVCOMMCapabilityError("kvcomm_requires_gpu_only")
    eviction_policy = getattr(tree_cache, "eviction_policy", "lru")
    eviction_policy = getattr(eviction_policy, "value", eviction_policy)
    if str(eviction_policy).lower() != "lru":
        raise KVCOMMCapabilityError("kvcomm_requires_lru")
    if getattr(req, "kvcomm_reconstructed", False) or getattr(
        req,
        "approx_kv_exact_preferred",
        False,
    ):
        return 0

    plugin = manager.plugins.get("kvcomm")
    if not isinstance(plugin, KVCOMMRecoveryPlugin):
        raise TypeError("registered kvcomm plugin has an invalid type")
    spec = KVCOMMRequestSpec.from_metadata(metadata)
    observed = _observed_kvcomm_segments(
        tree_cache,
        req,
        metadata,
        spec,
    )
    allocator = _allocator(tree_cache)
    capabilities = manager.runtime_capabilities
    if not isinstance(capabilities, KVCOMMRuntimeCapabilities):
        raise KVCOMMCapabilityError("capability_unavailable")
    guard_reason = capabilities.guard_kvcache(allocator.get_kvcache())
    if guard_reason is not None:
        raise KVCOMMCapabilityError(guard_reason)
    dtype_reason = capabilities.guard_declared_dtype(
        allocator.get_kvcache(),
        metadata.cache_dtype,
    )
    if dtype_reason is not None:
        raise KVCOMMCapabilityError(dtype_reason)

    if spec.action == KVCOMMAction.BASE:
        registered_handles = []
        completed = []
        try:
            for item in observed:
                segment = metadata.segments[item.descriptor.segment_index]
                handle = _copy_segment_to_store(
                    tree_cache=tree_cache,
                    req=req,
                    segment=segment,
                    key=item.key,
                    tokens=item.token_ids,
                    force_device=True,
                )
                registered_handles.append(handle)
                completed.append(
                    KVCOMMObservedSegment(
                        descriptor=item.descriptor,
                        key=item.key,
                        token_ids=item.token_ids,
                        positions=item.positions,
                        indices=item.indices,
                        handle=handle,
                    )
                )
            token_count = plugin.register_base_segments(
                metadata=metadata,
                spec=spec,
                observed=tuple(completed),
                store=manager.store,
                allocator=allocator,
            )
        except Exception:
            for handle in registered_handles:
                manager.store.release(handle)
            raise
        manager.record_request("kvcomm_base", "success")
        req.approx_kv_registered_tokens = token_count
        return token_count

    if spec.action not in (KVCOMMAction.ANCHOR, KVCOMMAction.REUSE):
        return 0
    token_count = plugin.update_from_dense(
        metadata=metadata,
        spec=spec,
        observed=observed,
        store=manager.store,
        allocator=allocator,
    )
    manager.record_request("kvcomm_update", "success")
    req.approx_kv_registered_tokens = token_count
    return token_count


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
        _copy_segment_to_store(
            tree_cache=tree_cache,
            req=req,
            segment=segment,
            key=key,
            tokens=tokens,
        )
        registered += segment.length

    manager.record_request("register", "success")
    req.approx_kv_registered_tokens = registered
    return registered


def _record_kvcomm_fallback(
    manager: Any,
    req: Any,
    reason: str,
    num_tokens: int,
) -> bool:
    req.kvcomm_fallback_reason = reason
    manager.record_fallback(reason, max(0, num_tokens))
    manager.record_request("reuse", "dense_fallback")
    return False


def _restore_kvcomm_prefix(
    tree_cache: Any,
    req: Any,
    metadata: Any,
    manager: Any,
) -> bool:
    reusable_limit = len(req.full_untruncated_fill_ids) - 1
    exact_length = len(req.prefix_indices)
    remaining = max(0, reusable_limit - exact_length)
    if remaining == 0:
        req.approx_kv_exact_preferred = True
        manager.record_request("reuse", "exact")
        return False
    if manager.config.host_residency_enabled or manager.config.async_prefetch_enabled:
        return _record_kvcomm_fallback(
            manager,
            req,
            "kvcomm_requires_gpu_only",
            remaining,
        )
    eviction_policy = getattr(tree_cache, "eviction_policy", "lru")
    eviction_policy = getattr(eviction_policy, "value", eviction_policy)
    if str(eviction_policy).lower() != "lru":
        return _record_kvcomm_fallback(
            manager,
            req,
            "kvcomm_requires_lru",
            remaining,
        )

    try:
        plugin = manager.plugins.get("kvcomm")
    except KeyError:
        return _record_kvcomm_fallback(
            manager,
            req,
            "kvcomm_plugin_missing",
            remaining,
        )
    if not isinstance(plugin, KVCOMMRecoveryPlugin):
        return _record_kvcomm_fallback(
            manager,
            req,
            "kvcomm_plugin_invalid",
            remaining,
        )

    context = RecoveryRequestContext(
        request_id=str(getattr(req, "rid", "unknown")),
        target_token_ids=tuple(int(token) for token in req.full_untruncated_fill_ids),
        exact_prefix_length=exact_length,
        custom_metadata={"approx_kv_metadata": metadata},
    )
    try:
        plan = plugin.build_plan(context, manager.store)
    except Exception:
        return _record_kvcomm_fallback(
            manager,
            req,
            "kvcomm_plan_failed",
            remaining,
        )
    if plan.recovery_mode != RecoveryMode.KVCOMM:
        reason = (
            plan.dense_ranges[0].reason
            if plan.dense_ranges
            else "kvcomm_dense_fallback"
        )
        return _record_kvcomm_fallback(
            manager,
            req,
            reason,
            len(plan.target_token_ids) or remaining,
        )
    reconstruction_plan = plan.plugin_data
    if not isinstance(reconstruction_plan, KVCOMMReconstructionPlan):
        return _record_kvcomm_fallback(
            manager,
            req,
            "kvcomm_plan_invalid",
            remaining,
        )
    try:
        validation_reason = plugin.validate_plan(
            reconstruction_plan,
            manager.store,
        )
    except Exception:
        logger.exception(
            "KVCOMM plan validation failed for request %s",
            getattr(req, "rid", "<unknown>"),
        )
        return _record_kvcomm_fallback(
            manager,
            req,
            "kvcomm_plan_validation_failed",
            reconstruction_plan.restore_length,
        )
    if validation_reason is not None:
        return _record_kvcomm_fallback(
            manager,
            req,
            validation_reason,
            reconstruction_plan.restore_length,
        )

    capabilities = manager.runtime_capabilities
    if capabilities is None:
        return _record_kvcomm_fallback(
            manager,
            req,
            "capability_unavailable",
            reconstruction_plan.restore_length,
        )
    allocator = _allocator(tree_cache)
    restored_indices = allocate_recovery_slots(
        tree_cache,
        reconstruction_plan.restore_length,
    )
    if (
        restored_indices is None
        or len(restored_indices) != reconstruction_plan.restore_length
    ):
        if restored_indices is not None:
            allocator.free(restored_indices)
        return _record_kvcomm_fallback(
            manager,
            req,
            "device_allocation_failed",
            reconstruction_plan.restore_length,
        )

    try:
        stats = execute_kvcomm_reconstruction(
            plan=reconstruction_plan,
            store=manager.store,
            allocator=allocator,
            target_indices=restored_indices,
            capabilities=capabilities,
        )
    except Exception as exc:
        allocator.free(restored_indices)
        logger.exception(
            "KVCOMM reconstruction failed for request %s",
            getattr(req, "rid", "<unknown>"),
        )
        reason = (
            exc.reason
            if isinstance(exc, KVCOMMCapabilityError)
            else "kvcomm_execution_failed"
        )
        return _record_kvcomm_fallback(
            manager,
            req,
            reason,
            reconstruction_plan.restore_length,
        )

    if exact_length + reconstruction_plan.restore_length >= len(
        req.full_untruncated_fill_ids
    ):
        allocator.free(restored_indices)
        return _record_kvcomm_fallback(
            manager,
            req,
            "kvcomm_final_token_violation",
            reconstruction_plan.restore_length,
        )
    req.prefix_indices = torch.cat(
        (
            req.prefix_indices,
            restored_indices.to(
                device=req.prefix_indices.device,
                dtype=req.prefix_indices.dtype,
            ),
        )
    )
    req.approx_kv_restored_len = reconstruction_plan.restore_length
    req.approx_kv_stats = stats
    req.kvcomm_reconstructed = True
    manager.record_transfer_stats(stats)
    manager.record_request("reuse", "success")
    return True


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
        req.approx_kv_exact_preferred = True
        manager.record_request("reuse", "exact_host_preferred")
        return False
    if metadata.plugin is not None:
        if metadata.plugin == "kvcomm":
            return _restore_kvcomm_prefix(
                tree_cache,
                req,
                metadata,
                manager,
            )
        manager.record_fallback(
            "unsupported_recovery_plugin",
            max(
                0,
                len(req.full_untruncated_fill_ids) - 1 - len(req.prefix_indices),
            ),
        )
        manager.record_request("reuse", "dense_fallback")
        return False

    reusable_limit = len(req.full_untruncated_fill_ids) - 1
    exact_length = len(req.prefix_indices)
    ordered_segments = sorted(
        metadata.segments,
        key=lambda segment: segment.target_start,
    )
    active_segments = []
    next_target = exact_length
    for segment in ordered_segments:
        if segment.target_end <= exact_length:
            continue
        if segment.target_start > next_target:
            break
        active_segments.append(segment)
        next_target = max(next_target, segment.target_end)
        if next_target >= reusable_limit:
            break
    restore_end = min(next_target, reusable_limit)
    restore_length = restore_end - exact_length
    if restore_length <= 0:
        req.approx_kv_exact_preferred = True
        manager.record_request("reuse", "exact")
        return False

    handles = []
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
            manager.record_fallback("store_miss", restore_length)
            manager.record_request("reuse", "dense_fallback")
            return False
        try:
            handle = manager.ensure_device(handle)
        except Exception:
            manager.record_fallback("residency_load_failed", restore_length)
            manager.record_request("reuse", "dense_fallback")
            return False
        handles.append((segment, handle, overlap_start, overlap_end))

    if not handles or handles[0][2] != exact_length:
        manager.record_fallback("prefix_gap", restore_length)
        manager.record_request("reuse", "dense_fallback")
        return False
    for previous, current in zip(handles, handles[1:]):
        if previous[3] != current[2]:
            manager.record_fallback("prefix_gap", restore_length)
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
    rope_config = manager.rope_config or RoPEConfig(
        rotary_dim=0,
        base=10000.0,
        is_neox_style=True,
    )
    spans = []
    for segment, handle, overlap_start, overlap_end in handles:
        source_offset = segment.source_offset + (overlap_start - segment.target_start)
        source_position = handle.source_start + source_offset
        rope_delta = overlap_start - source_position
        if rope_delta != 0 and rope_config.rotary_dim == 0:
            allocator.free(restored_indices)
            manager.record_fallback("rope_config_unavailable", restore_length)
            manager.record_request("reuse", "dense_fallback")
            return False
        spans.append(
            TransferSpan(
                source=handle,
                source_offset=source_offset,
                target_start=overlap_start - exact_length,
                length=overlap_end - overlap_start,
                rope_delta=rope_delta,
                chunk_start=0,
                chunk_length=restore_length,
            )
        )

    backend = RadixKVTransferBackend(
        allocator=allocator,
        target_indices=lambda start, length: restored_indices[start : start + length],
        dense_prefill=lambda start, length, reason: fallback_reasons.append(reason),
        rope=rope_config,
    )
    target_tokens = tuple(
        int(token) for token in req.full_untruncated_fill_ids[exact_length:restore_end]
    )
    try:
        stats = manager.execute(
            KVReusePlan(
                target_token_ids=target_tokens,
                recovery_mode=RecoveryMode.COPY,
                copied_spans=tuple(spans),
                require_full_coverage=True,
            ),
            backend,
        )
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
