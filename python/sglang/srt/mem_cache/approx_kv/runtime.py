from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

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


class ApproxKVRegistrationError(RuntimeError):
    pass


def _allocator(tree_cache: Any) -> Any:
    allocator = tree_cache.token_to_kv_pool_allocator
    if allocator is None:
        raise RuntimeError("approximate KV requires a token allocator")
    return allocator


def release_provisional_recovery_slots(tree_cache: Any, req: Any) -> int:
    """Free recovery slots that were never committed to a batch.

    Recovery attaches device slots from inside ``init_next_round_input``,
    which runs before ``schedule_policy.add_one_req`` decides whether to
    admit the request. If the request is not admitted, ``prepare_for_extend``
    never copies those slots into ``req_to_token``, so the normal
    end-of-request release cannot see them, and the next ``match_prefix``
    overwrites ``req.prefix_indices`` and drops the only reference to them.

    Ownership transfers to the request in ``prepare_for_extend``; until then
    the slots are provisional and this is the only thing that can reclaim
    them.
    """
    indices = getattr(req, "approx_kv_provisional_indices", None)
    if indices is None:
        return 0
    allocator = getattr(tree_cache, "token_to_kv_pool_allocator", None)
    if allocator is None:
        # Keep the reference: dropping it here would lose the only handle on
        # these slots and leak them permanently.
        return 0
    allocator.free(indices)
    # Clear only after the free succeeded, for the same reason.
    req.approx_kv_provisional_indices = None
    req.approx_kv_restored_len = 0
    manager = getattr(tree_cache, "approx_kv", None)
    if manager is not None:
        manager.remove_provisional_tokens(len(indices))
    return len(indices)


def commit_provisional_recovery_slots(tree_cache: Any, req: Any) -> int:
    """Transfer provisional recovery slots to req_to_token ownership."""

    indices = getattr(req, "approx_kv_provisional_indices", None)
    if indices is None:
        return 0
    req.approx_kv_provisional_indices = None
    manager = getattr(tree_cache, "approx_kv", None)
    if manager is not None:
        manager.remove_provisional_tokens(len(indices))
    return len(indices)


@contextmanager
def protect_request_prefix(tree_cache: Any, req: Any):
    """Hold the request's own prefix lock for the whole recovery window.

    ``Req.init_next_round_input`` runs recovery *before* the scheduler takes
    the request's prefix lock in ``schedule_policy.add_one_req``. In that
    window ``req.last_node`` still has ``lock_ref == 0``, so the exact nodes
    backing ``req.prefix_indices`` are legal cross-store eviction victims.
    Recovering under device pressure could therefore free the request's own
    prefix and hand the very same slots back as the recovery destination,
    silently overwriting the KV the request is about to attend over.

    ``inc_lock_ref`` walks to the root, so this protects the whole matched
    chain and removes it from ``evictable_leaves`` for the duration.
    """
    node = getattr(req, "last_node", None)
    inc_lock_ref = getattr(tree_cache, "inc_lock_ref", None)
    dec_lock_ref = getattr(tree_cache, "dec_lock_ref", None)
    if node is None or inc_lock_ref is None or dec_lock_ref is None:
        yield
        return
    result = inc_lock_ref(node)
    # SWA and Unified caches return the acquired window plus the nodes they
    # skipped; releasing without it can walk past that window and decrement an
    # ancestor another request still holds.
    params = getattr(result, "to_dec_params", None)
    try:
        yield
    finally:
        if params is None:
            dec_lock_ref(node)
        else:
            dec_lock_ref(node, params())


def allocate_recovery_slots(tree_cache: Any, num_tokens: int):
    """Allocate approximate-recovery slots after evicting exact Radix victims."""
    allocator = _allocator(tree_cache)
    manager = getattr(tree_cache, "approx_kv", None)
    if (
        manager is not None
        and manager.config.cross_store_enabled
        and hasattr(tree_cache, "cross_store_resources")
    ):
        try:
            result = manager.cross_store_coordinator(tree_cache).allocate_tokens(
                num_tokens
            )
        except (
            AttributeError,
            KeyError,
            MemoryError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            manager.record_fallback("cross_store_error", num_tokens)
            return None
        if result.committed:
            return result.allocation
        manager.record_fallback("cross_store_reservation_failed", num_tokens)
        return None
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


def _pin_registered_segment(manager: Any, metadata: Any, handle: Any) -> None:
    """Hold an opt-in persistent lease on a just-registered source segment."""
    if not getattr(metadata, "pin_until_reset", False) or handle is None:
        return
    try:
        manager.pin_registration(handle)
    except Exception:
        manager.release_segment(handle)
        raise


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
    if metadata.pin_until_reset:
        manager.validate_persistent_registration_request(len(metadata.segments))
    if req.req_pool_idx is None or req.kv is None:
        raise RuntimeError("request KV must exist before source registration")

    allocator = _allocator(tree_cache)
    registered = 0
    requested_tokens = sum(segment.length for segment in metadata.segments)
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

        register_to_host = (
            manager.config.host_residency_enabled
            and segment.residency != ResidencyTier.DEVICE
        )
        dependency_leases = []
        try:
            dependency_missing = False
            for dependency in sorted(segment.dependencies):
                try:
                    dependency_leases.append(
                        manager.store.pin(
                            manager.store.handle_for_object_id(dependency),
                            ttl_s=3600,
                        )
                    )
                except KeyError:
                    manager.record_fallback(
                        "registration_dependency_missing",
                        segment.length,
                    )
                    dependency_missing = True
                    break
            if dependency_missing:
                continue
            if register_to_host:
                load_result = manager.export_to_host(DeviceKVRef(source_indices))
                expected_bytes = (
                    segment.length * manager.config.cross_store_bytes_per_token
                )
                if manager.config.cross_store_enabled:
                    try:
                        transfer_bytes = manager.validate_transfer_bytes(
                            load_result.bytes_transferred,
                            expected_bytes=expected_bytes,
                        )
                    except ValueError:
                        if load_result.release_backend is not None:
                            load_result.release_backend(
                                load_result.backend_ref,
                                ResidencyTier.HOST,
                            )
                        raise
                    resident_bytes = expected_bytes
                else:
                    transfer_bytes = (
                        load_result.bytes_transferred
                        if load_result.bytes_transferred > 0
                        else expected_bytes
                    )
                    resident_bytes = transfer_bytes
                handle = manager.register_segment(
                    key=key,
                    token_ids=tokens,
                    source_start=segment.target_start,
                    residency=ResidencyTier.HOST,
                    backend_ref=load_result.backend_ref,
                    release_backend=load_result.release_backend,
                    resident_bytes=resident_bytes,
                    object_id=segment.object_id or f"approx:{segment.content_hash}",
                    object_kind=segment.object_kind,
                    dependencies=segment.dependencies,
                    dense_cost_ms=segment.dense_cost_ms,
                    recovery_cost_ms=segment.recovery_cost_ms,
                    next_use_ordinal=segment.next_use_ordinal,
                    retired=segment.retired,
                )
                if handle is None:
                    continue
                _pin_registered_segment(manager, metadata, handle)
                manager.record_host_export(
                    load_result.num_tokens or segment.length,
                    transfer_bytes,
                    duration_ms=load_result.duration_ms,
                )
            else:
                target_indices = (
                    allocate_recovery_slots(tree_cache, segment.length)
                    if manager.config.cross_store_registration_evicts_exact
                    else allocator.alloc(segment.length)
                )
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
                    resident_bytes=(
                        segment.length * manager.config.cross_store_bytes_per_token
                    ),
                    object_id=segment.object_id or f"approx:{segment.content_hash}",
                    object_kind=segment.object_kind,
                    dependencies=segment.dependencies,
                    dense_cost_ms=segment.dense_cost_ms,
                    recovery_cost_ms=segment.recovery_cost_ms,
                    next_use_ordinal=segment.next_use_ordinal,
                    retired=segment.retired,
                )
                if handle is None:
                    continue
                _pin_registered_segment(manager, metadata, handle)
        finally:
            for lease in reversed(dependency_leases):
                manager.store.unpin(lease)
        registered += segment.length

    manager.record_request(
        "register",
        (
            "success"
            if registered == requested_tokens
            else ("dense_only" if registered == 0 else "partial")
        ),
    )
    req.approx_kv_registered_tokens = registered
    return registered


@dataclass(frozen=True)
class ResolvedReuseSpans:
    """Validated, device-resident, contiguous body spans for reuse.

    Shared between the raw R0 copy path (``restore_request_prefix``) and
    any recovery plugin path (e.g. EPIC's ``epic_runtime.py``) so the
    segment-matching, staleness, residency and RoPE-availability checks are
    implemented exactly once.
    """

    exact_length: int
    restore_end: int
    restore_length: int
    spans: tuple[TransferSpan, ...]
    rope_config: RoPEConfig
    leases: tuple[Any, ...]


@contextmanager
def pin_reuse_sources(manager: Any, resolved: ResolvedReuseSpans):
    leases = list(resolved.leases)
    try:
        if not leases:
            manager.record_fallback(
                "source_pin_stale",
                resolved.restore_length,
            )
            manager.record_request("reuse", "dense_fallback")
            yield False
            return
        yield True
    finally:
        for lease in reversed(leases):
            manager.store.unpin(lease)


def resolve_reuse_spans(
    tree_cache: Any,
    req: Any,
    metadata: Any,
    manager: Any,
) -> ResolvedReuseSpans | None:
    """Resolve the reusable body for ``req`` into contiguous transfer spans.

    Returns ``None`` (after recording the appropriate ``exact`` or
    ``dense_fallback`` telemetry, exactly as the previous inline
    implementation did) when the request cannot be serviced from
    approximate KV, e.g. because there is no reusable gap, a source segment
    is missing/stale, or RoPE relocation would be required without a bound
    RoPE config.
    """
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
        pending_end = min(
            max(
                (
                    segment.target_end
                    for segment in ordered_segments
                    if segment.target_end > exact_length
                ),
                default=exact_length,
            ),
            reusable_limit,
        )
        pending_length = pending_end - exact_length
        if pending_length > 0:
            manager.record_fallback("prefix_gap", pending_length)
            manager.record_request("reuse", "dense_fallback")
        else:
            manager.record_request("reuse", "exact")
        return None

    handles = []
    leases = []
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
            for lease in reversed(leases):
                manager.store.unpin(lease)
            manager.record_fallback("store_miss", restore_length)
            manager.record_request("reuse", "dense_fallback")
            return None
        try:
            lease = manager.store.pin(handle, ttl_s=3600)
        except KeyError:
            for acquired in reversed(leases):
                manager.store.unpin(acquired)
            manager.record_fallback("source_pin_stale", restore_length)
            manager.record_request("reuse", "dense_fallback")
            return None
        leases.append(lease)
        try:
            handle = manager.ensure_device(handle)
        except (KeyError, MemoryError, RuntimeError, TypeError, ValueError):
            for acquired in reversed(leases):
                manager.store.unpin(acquired)
            manager.record_fallback("residency_load_failed", restore_length)
            manager.record_request("reuse", "dense_fallback")
            return None
        handles.append((segment, handle, overlap_start, overlap_end))

    if not handles or handles[0][2] != exact_length:
        for lease in reversed(leases):
            manager.store.unpin(lease)
        manager.record_fallback("prefix_gap", restore_length)
        manager.record_request("reuse", "dense_fallback")
        return None
    for previous, current in zip(handles, handles[1:]):
        if previous[3] != current[2]:
            for lease in reversed(leases):
                manager.store.unpin(lease)
            manager.record_fallback("prefix_gap", restore_length)
            manager.record_request("reuse", "dense_fallback")
            return None

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
            for lease in reversed(leases):
                manager.store.unpin(lease)
            manager.record_fallback("rope_config_unavailable", restore_length)
            manager.record_request("reuse", "dense_fallback")
            return None
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

    return ResolvedReuseSpans(
        exact_length=exact_length,
        restore_end=restore_end,
        restore_length=restore_length,
        spans=tuple(spans),
        rope_config=rope_config,
        leases=tuple(leases),
    )


def finalize_copy_reuse(
    tree_cache: Any,
    req: Any,
    manager: Any,
    resolved: ResolvedReuseSpans,
) -> bool:
    """Physically copy+RoPE-correct ``resolved.spans`` and commit the prefix.

    This is the raw (R0-equivalent) whole-span copy execution, shared by
    the plain reuse path and EPIC's k=0 degenerate case (no leading-k
    repair requested, pure copy).
    """
    with pin_reuse_sources(manager, resolved) as pinned:
        if not pinned:
            return False
        allocator = _allocator(tree_cache)
        restored_indices = allocate_recovery_slots(
            tree_cache,
            resolved.restore_length,
        )
        if restored_indices is None or len(restored_indices) != resolved.restore_length:
            if restored_indices is not None:
                allocator.free(restored_indices)
            # The cross-store allocator already records the specific terminal
            # reason (cross_store_error or cross_store_reservation_failed).
            # Recording device_allocation_failed as well double-counts the
            # same tokens in one metric family.
            if not manager.config.cross_store_enabled:
                manager.record_fallback(
                    "device_allocation_failed",
                    resolved.restore_length,
                )
            manager.record_request("reuse", "dense_fallback")
            return False

        fallback_reasons: list[str] = []
        backend = RadixKVTransferBackend(
            allocator=allocator,
            target_indices=lambda start, length: restored_indices[
                start : start + length
            ],
            dense_prefill=lambda start, length, reason: fallback_reasons.append(reason),
            rope=resolved.rope_config,
        )
        target_tokens = tuple(
            int(token)
            for token in req.full_untruncated_fill_ids[
                resolved.exact_length : resolved.restore_end
            ]
        )
        try:
            stats = manager.execute(
                KVReusePlan(
                    target_token_ids=target_tokens,
                    recovery_mode=RecoveryMode.COPY,
                    copied_spans=resolved.spans,
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
        req.approx_kv_restored_len = resolved.restore_length
        # Provisional until prepare_for_extend copies them into req_to_token.
        req.approx_kv_provisional_indices = restored_indices
        manager.add_provisional_tokens(len(restored_indices))
        req.approx_kv_stats = stats
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
        manager.record_request("reuse", "exact_host_preferred")
        return False

    resolved = resolve_reuse_spans(tree_cache, req, metadata, manager)
    if resolved is None:
        return False
    return finalize_copy_reuse(tree_cache, req, manager, resolved)
