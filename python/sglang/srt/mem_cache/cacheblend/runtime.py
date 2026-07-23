from __future__ import annotations

"""Truthful CacheBlend server request path.

Wires the actual per-request flow:

1. Exact cache first -- unchanged: ``tree_cache.match_prefix`` (called by
   the scheduler before this function runs) already fills
   ``req.prefix_indices`` with any exact-hit prefix. This module only ever
   *extends* that prefix; it never overwrites or duplicates the exact
   match, and (like every other approximate-KV path) never writes into the
   exact Radix tree -- ``schedule_batch.Req.skip_radix_cache_insert`` is
   already forced True for any request carrying ``approx_kv_metadata``,
   so a CacheBlend-served request is never inserted as if it were an
   exact match.
2. Registered source segments are looked up and made resident (with
   optional load/recompute overlap across multiple segments), then
   raw-copied + RoPE-corrected into the destination KV slots -- this is
   the common-core baseline reuse mechanism (`approx_kv.transfer`),
   reused as-is.
3. Real HKVD measurement + gradual filtering select the token positions
   whose reused KV deviates most from what the model would actually
   compute for them.
4. Selected-token repair: every layer from the probe layer onward is
   really recomputed, in one batched call per layer, for exactly the
   selected positions (see ``recompute.py`` for why per-token recompute
   is rejected).
5. The request's final prompt token is *never* included in the restored
   range (``reusable_limit = prompt_length - 1``), so it always gets a
   real forward pass through the normal scheduler path.
6. Any unsupported layout or violated invariant (gap in segment
   coverage, stale handle, residency failure, missing capability, mid-
   flight mismatch) aborts the *entire* restore and frees the
   provisional allocation, falling back to the normal dense/exact path --
   never a partial, silently-degraded write.
7. This function is called directly from ``Req.init_next_round_input``,
   a scheduler-critical path with no wrapping try/except at the call
   site: every failure mode this module can hit -- including a
   misregistered plugin of the wrong type and a KV-transfer execution
   exception -- must be caught, logged, recorded as an honest
   ``dense_fallback`` via ``manager.record_fallback`` /
   ``manager.record_request``, and returned as ``False`` rather than
   raised or re-raised, or it would escape this call site and kill the
   scheduler process.

The restore buffer itself is allocated via the shared common-core
``approx_kv.runtime.allocate_recovery_slots`` (ported from the R1
EPIC/LegoLink fork) rather than a bare ``allocator.alloc``, so that under
real GPU pressure this path evicts exact Radix victims first instead of
bypassing SGLang's standard ``evict_from_tree_cache -> allocator.alloc``
ordering.
"""

import logging
from typing import Any

import torch

from sglang.srt.mem_cache.approx_kv.radix_backend import (
    RadixKVTransferBackend,
    RoPEConfig,
)
from sglang.srt.mem_cache.approx_kv.request import ApproxKVRequestOperation
from sglang.srt.mem_cache.approx_kv.runtime import allocate_recovery_slots
from sglang.srt.mem_cache.approx_kv.types import (
    KVReusePlan,
    KVSegmentKey,
    KVTransferStats,
    RecoveryMode,
    ResidencyTier,
    SegmentKind,
    TransferSpan,
    token_ids_hash,
)

from .hkvd import compute_token_deviation, select_hkvd_tokens
from .plugin import CACHEBLEND_PLUGIN_NAME, CacheBlendRecoveryPlugin
from .precomputed import FreshKVSpan, PrecomputedCacheBlendBackend
from .recompute import LayerRecomputeCoordinator

logger = logging.getLogger(__name__)


def _allocator(tree_cache: Any) -> Any:
    allocator = tree_cache.token_to_kv_pool_allocator
    if allocator is None:
        raise RuntimeError("approximate KV requires a token allocator")
    return allocator


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


def _merge_stats(all_stats: list[KVTransferStats]) -> KVTransferStats:
    merged = KVTransferStats(
        recovery_mode=RecoveryMode.COPY,
        target_tokens=sum(s.target_tokens for s in all_stats),
    )
    for stats in all_stats:
        merged.copied_k_tokens += stats.copied_k_tokens
        merged.rotated_k_tokens += stats.rotated_k_tokens
        merged.copied_v_tokens += stats.copied_v_tokens
        merged.recomputed_tokens += stats.recomputed_tokens
        merged.h2d_tokens += stats.h2d_tokens
        merged.h2d_bytes += stats.h2d_bytes
        merged.h2d_ms += stats.h2d_ms
        merged.copy_ms += stats.copy_ms
        merged.rope_ms += stats.rope_ms
        merged.fallback_reasons.extend(stats.fallback_reasons)
    return merged


def restore_request_prefix_cacheblend(tree_cache: Any, req: Any) -> bool:
    """Real server-path entry point for the CacheBlend recovery plugin."""
    metadata = getattr(req, "approx_kv_metadata", None)
    manager = getattr(tree_cache, "approx_kv", None)
    if (
        metadata is None
        or metadata.operation != ApproxKVRequestOperation.REUSE
        or metadata.plugin != CACHEBLEND_PLUGIN_NAME
        or manager is None
        or not manager.config.core_enabled
    ):
        return False

    try:
        plugin = manager.plugins.get(CACHEBLEND_PLUGIN_NAME)
    except KeyError:
        manager.record_fallback("cacheblend_plugin_missing", 0)
        manager.record_request("reuse", "dense_fallback")
        return False
    if not isinstance(plugin, CacheBlendRecoveryPlugin):
        logger.error(
            "CacheBlend plugin registration has unexpected type %s "
            "(expected CacheBlendRecoveryPlugin) for request %s; falling "
            "back to dense instead of raising out of the scheduler's "
            "init_next_round_input path",
            type(plugin).__name__,
            getattr(req, "rid", "<unknown>"),
        )
        manager.record_fallback("cacheblend_plugin_wrong_type", 0)
        manager.record_request("reuse", "dense_fallback")
        return False

    if req.needs_host_load_back():
        manager.record_request("reuse", "exact_host_preferred")
        return False

    reusable_limit = len(req.full_untruncated_fill_ids) - 1
    exact_length = len(req.prefix_indices)
    ordered_segments = sorted(metadata.segments, key=lambda s: s.target_start)
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
        manager.record_request("reuse", "exact")
        return False

    precomputed_requested = all(
        segment.content_hash.startswith("cacheblend-raw:")
        for segment in active_segments
    )
    if not plugin.capable and not precomputed_requested:
        manager.record_fallback("cacheblend_capability_unavailable", restore_length)
        manager.record_request("reuse", "dense_fallback")
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

    probe_backend = plugin.probe_backend
    recompute_backend = plugin.recompute_backend
    precomputed_backend = None
    if not plugin.capable:
        fresh_spans = []
        for segment, _, overlap_start, overlap_end in handles:
            fresh_hash = segment.content_hash.replace(
                "cacheblend-raw:",
                "cacheblend-fresh:",
                1,
            )
            tokens = tuple(
                int(token)
                for token in req.full_untruncated_fill_ids[
                    segment.target_start : segment.target_end
                ]
            )
            fresh_key = _segment_key(
                tokens=tokens,
                content_hash=fresh_hash,
                model_fingerprint=metadata.model_fingerprint,
                cache_dtype=metadata.cache_dtype,
            )
            fresh_handle = manager.store.lookup(fresh_key)
            if fresh_handle is None:
                manager.record_fallback(
                    "cacheblend_fresh_store_miss",
                    restore_length,
                )
                manager.record_request("reuse", "dense_fallback")
                return False
            try:
                fresh_handle = manager.ensure_device(fresh_handle)
            except Exception:
                manager.record_fallback(
                    "cacheblend_fresh_residency_load_failed",
                    restore_length,
                )
                manager.record_request("reuse", "dense_fallback")
                return False
            fresh_spans.append(
                FreshKVSpan(
                    target_start=overlap_start,
                    length=overlap_end - overlap_start,
                    source=fresh_handle,
                    source_offset=overlap_start - segment.target_start,
                )
            )
        precomputed_backend = PrecomputedCacheBlendBackend(
            kvcache=_allocator(tree_cache).get_kvcache(),
            spans=fresh_spans,
        )
        probe_backend = precomputed_backend
        recompute_backend = precomputed_backend

    allocator = _allocator(tree_cache)
    restored_indices = allocate_recovery_slots(tree_cache, restore_length)
    if restored_indices is None or len(restored_indices) != restore_length:
        if restored_indices is not None:
            allocator.free(restored_indices)
        manager.record_fallback("device_allocation_failed", restore_length)
        manager.record_request("reuse", "dense_fallback")
        return False

    # --- Load/recompute overlap: kick off every segment's residency
    # ticket up front (when async prefetch is configured) so a later
    # segment's host->device transfer runs concurrently with an earlier
    # segment's baseline copy + HKVD + selective recompute below. When
    # async prefetch is disabled (the Phase 4 R2 default: scheduler stays
    # S0 LRU / GPU-only / prefetch off) this degrades to a plain
    # sequential `ensure_device` per segment, identical to the R0 path.
    tickets: dict[int, Any] = {}
    if manager.config.async_prefetch_enabled:
        for _, handle, _, _ in handles:
            if handle.residency != ResidencyTier.DEVICE and id(handle) not in tickets:
                tickets[id(handle)] = manager.begin_prefetch(handle)

    rope_config = manager.rope_config or RoPEConfig(
        rotary_dim=0,
        base=10000.0,
        is_neox_style=True,
    )

    segment_stats: list[KVTransferStats] = []
    for segment, handle, overlap_start, overlap_end in handles:
        ticket = tickets.get(id(handle))
        try:
            resolved = ticket.wait() if ticket is not None else manager.ensure_device(
                handle
            )
        except Exception:
            allocator.free(restored_indices)
            manager.record_fallback("residency_load_failed", restore_length)
            manager.record_request("reuse", "dense_fallback")
            return False

        base_offset = overlap_start - exact_length
        length = overlap_end - overlap_start
        source_offset = segment.source_offset + (overlap_start - segment.target_start)
        source_position = resolved.source_start + source_offset
        rope_delta = overlap_start - source_position
        if rope_delta != 0 and rope_config.rotary_dim == 0:
            allocator.free(restored_indices)
            manager.record_fallback("rope_config_unavailable", restore_length)
            manager.record_request("reuse", "dense_fallback")
            return False

        fallback_reasons: list[str] = []
        backend = RadixKVTransferBackend(
            allocator=allocator,
            target_indices=lambda start, ln, base=base_offset: restored_indices[
                base + start : base + start + ln
            ],
            dense_prefill=lambda start, ln, reason: fallback_reasons.append(reason),
            rope=rope_config,
        )
        target_tokens = tuple(
            int(token)
            for token in req.full_untruncated_fill_ids[overlap_start:overlap_end]
        )
        plan = KVReusePlan(
            target_token_ids=target_tokens,
            recovery_mode=RecoveryMode.COPY,
            copied_spans=(
                TransferSpan(
                    source=resolved,
                    source_offset=source_offset,
                    target_start=0,
                    length=length,
                    rope_delta=rope_delta,
                    chunk_start=0,
                    chunk_length=length,
                ),
            ),
            require_full_coverage=True,
        )
        try:
            stats = manager.execute(plan, backend)
        except Exception:
            allocator.free(restored_indices)
            logger.exception(
                "CacheBlend KV transfer execution failed for request %s",
                getattr(req, "rid", "<unknown>"),
            )
            manager.record_fallback(
                "cacheblend_transfer_execution_failed", restore_length
            )
            manager.record_request("reuse", "dense_fallback")
            return False
        if fallback_reasons or stats.recomputed_tokens:
            allocator.free(restored_indices)
            manager.record_request("reuse", "dense_fallback")
            return False
        segment_stats.append(stats)

    # --- Real HKVD measurement + gradual filtering over the whole
    # restored span (all segments combined), driven only by genuine
    # freshly-computed K from `plugin.probe_backend` -- never from any
    # static/structural proxy (see hkvd.py module docstring for why that
    # approach was falsified historically).
    kvcache = allocator.get_kvcache()

    def deviation_fn(probe_layer_id: int, local_positions: torch.Tensor) -> torch.Tensor:
        slot_indices = restored_indices[local_positions]
        token_positions = local_positions + exact_length
        reused_keys = kvcache.get_key_buffer(probe_layer_id)[slot_indices]
        fresh_keys = probe_backend.probe_layer(
            layer_id=probe_layer_id,
            slot_indices=slot_indices,
            token_positions=token_positions,
        )
        return compute_token_deviation(fresh_keys, reused_keys)

    try:
        selection = select_hkvd_tokens(
            list(range(restore_length)),
            stages=plugin.config.probe_stages,
            final_ratio=plugin.config.ratio,
            deviation_fn=deviation_fn,
        )
    except Exception:
        allocator.free(restored_indices)
        logger.exception(
            "CacheBlend HKVD measurement failed for request %s",
            getattr(req, "rid", "<unknown>"),
        )
        manager.record_fallback("cacheblend_hkvd_measurement_failed", restore_length)
        manager.record_request("reuse", "dense_fallback")
        return False

    selected_local = selection.selected_positions
    if selected_local:
        selected_slots = [int(restored_indices[p]) for p in selected_local]
        selected_positions = [p + exact_length for p in selected_local]
        coordinator = LayerRecomputeCoordinator(
            recompute_backend,
            first_recompute_layer=plugin.config.first_recompute_layer,
            layer_num=kvcache.layer_num,
        )
        try:
            recompute_results = coordinator.recompute_selected(
                slot_indices=selected_slots,
                token_positions=selected_positions,
            )
        except Exception:
            allocator.free(restored_indices)
            logger.exception(
                "CacheBlend selective recompute failed for request %s",
                getattr(req, "rid", "<unknown>"),
            )
            manager.record_fallback(
                "cacheblend_selective_recompute_failed", restore_length
            )
            manager.record_request("reuse", "dense_fallback")
            return False
    else:
        recompute_results = ()

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
    req.approx_kv_stats = _merge_stats(segment_stats)
    req.cacheblend_selected_tokens = len(selected_local)
    req.cacheblend_candidate_tokens = restore_length
    req.cacheblend_ratio = plugin.config.ratio
    req.cacheblend_recomputed_layers = tuple(
        result.layer_id for result in recompute_results
    )
    req.cacheblend_precomputed = precomputed_backend is not None
    manager.record_cacheblend_repair(
        selected_tokens=len(selected_local),
        recomputed_layers=len(recompute_results),
        precomputed=precomputed_backend is not None,
    )
    manager.record_request("reuse", "success")
    return True
