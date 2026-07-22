from __future__ import annotations

"""Real CacheTune server request path.

Wires the actual per-request flow. Steps 1-2 and 5-9 below mirror
`research/cacheblend`'s `runtime.py` almost exactly (same common-core
baseline reuse mechanism, same dense-fallback-safe error handling
conventions); steps 3-4 are what makes this CacheTune rather than a copy
of CacheBlend: the repair-token *count* for step 8 is not a fixed sweep
parameter but the live output of the hardware-aware roofline controller.

1. Exact cache first -- unchanged: ``tree_cache.match_prefix`` (called by
   the scheduler before this function runs) already fills
   ``req.prefix_indices`` with any exact-hit prefix. This module only ever
   *extends* that prefix; it never overwrites or duplicates the exact
   match, and never writes into the exact Radix tree --
   ``schedule_batch.Req.skip_radix_cache_insert`` is already forced True
   for any request carrying ``approx_kv_metadata``.
2. Registered source segments are looked up and made resident, then
   raw-copied + RoPE-corrected into the destination KV slots -- the
   common-core baseline reuse mechanism (`approx_kv.transfer`), reused
   as-is. This is CacheTune's "transfer" critical path and always covers
   the *entire* restored span, regardless of the selected ratio.
3. The live ``CacheTuneController`` resolves a ``HardwareProfileKey`` for
   this request's ``(hardware_tier, model_fingerprint,
   chunk_length_bucket(restore_length))`` and returns a
   ``CacheTuneDecision`` -- the controller's roofline- or
   calibration-derived repair ratio, deterministically quantized to an
   executable integer ``repair_tokens`` count for this exact restore
   length (see `hardware_profile.py`/`controller.py`). If no hardware
   measurement is available for this profile and no deployment-wide
   measurement was configured at server startup, the request honestly
   dense-falls-back (``cachetune_measurement_unavailable``) rather than
   fabricate a ratio.
4. Repair-token selection (`token_selection.select_repair_tokens`) is
   given the controller's ``repair_tokens`` count directly (not a ratio
   it would have to re-round itself), and its own result dataclass
   self-validates that it produced exactly that many positions -- see
   ``token_selection.TokenSelection.__post_init__``. This is what
   guarantees "实际 repair token count 与选择一致" end to end.
5. Real per-token deviation measurement (reusing
   `research/cacheblend`'s scoring primitive, ported into
   `token_selection.py`) drives *which* positions within the restored
   span are selected for repair -- never a static/structural proxy.
6. Selected-token repair: every layer from ``first_recompute_layer``
   onward is really recomputed, in one batched call per layer, for
   exactly the selected positions (see ``recompute.py`` for why
   per-token recompute is rejected). Because SGLang's ``ModelRunner``
   has no hook for a genuine inline per-layer forward on an arbitrary
   token subset, the actual K/V written here comes from a real,
   separate dense preparation request registered under a
   ``cachetune-fresh:`` content-hash prefix (`precomputed.py`) -- ported
   unchanged in mechanism from CacheBlend's ``cacheblend-fresh:``
   adapter. This means the "recompute" and "transfer" critical paths
   this controller optimizes over are NOT executed with genuine
   wall-clock overlap in this backend (that would require a ModelRunner
   hook this fork does not have); the roofline model is used faithfully
   to *choose* the ratio, but the *execution* of the chosen ratio is
   this project's honest, available-hardware adaptation -- see
   `cachetune/__init__.py` for the explicit scope statement.
7. The request's final prompt token is *never* included in the restored
   range (``reusable_limit = prompt_length - 1``), so it always gets a
   real forward pass through the normal scheduler path.
8. Any unsupported layout or violated invariant (gap in segment
   coverage, stale handle, residency failure, missing capability, mid-
   flight mismatch) aborts the *entire* restore and frees the
   provisional allocation, falling back to the normal dense/exact path --
   never a partial, silently-degraded write.
"""

import logging
from typing import Any

import torch

from sglang.srt.mem_cache.approx_kv.radix_backend import (
    RadixKVTransferBackend,
    RoPEConfig,
)
from sglang.srt.mem_cache.approx_kv.request import ApproxKVRequestOperation
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

from .hardware_profile import HardwareProfileKey, chunk_length_bucket
from .plugin import CACHETUNE_PLUGIN_NAME, CacheTuneRecoveryPlugin
from .precomputed import FreshKVSpan, PrecomputedCacheTuneBackend
from .recompute import LayerRecomputeCoordinator
from .token_selection import (
    TokenSelection,
    compute_token_deviation,
    select_repair_tokens,
)

logger = logging.getLogger(__name__)

_RAW_PREFIX = "cachetune-raw:"
_FRESH_PREFIX = "cachetune-fresh:"


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


def restore_request_prefix_cachetune(tree_cache: Any, req: Any) -> bool:
    """Real server-path entry point for the CacheTune recovery plugin."""
    metadata = getattr(req, "approx_kv_metadata", None)
    manager = getattr(tree_cache, "approx_kv", None)
    if (
        metadata is None
        or metadata.operation != ApproxKVRequestOperation.REUSE
        or metadata.plugin != CACHETUNE_PLUGIN_NAME
        or manager is None
        or not manager.config.core_enabled
    ):
        return False

    try:
        plugin = manager.plugins.get(CACHETUNE_PLUGIN_NAME)
    except KeyError:
        manager.record_fallback("cachetune_plugin_missing", 0)
        manager.record_request("reuse", "dense_fallback")
        return False
    if not isinstance(plugin, CacheTuneRecoveryPlugin):
        raise TypeError(
            "the 'cachetune' plugin registration must be a "
            "CacheTuneRecoveryPlugin instance"
        )

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

    # --- CacheTune controller decision. Resolved as early as possible
    # (before any segment store lookups) so the "no hardware measurement
    # configured" gap is reported cheaply and honestly, without doing
    # unnecessary store/residency work first.
    allocator = _allocator(tree_cache)
    kvcache = allocator.get_kvcache()
    profile_key = HardwareProfileKey(
        hardware_tier=plugin.config.hardware_tier,
        model_fingerprint=metadata.model_fingerprint,
        chunk_length_bucket=chunk_length_bucket(restore_length),
    )
    controller = plugin.controller
    if not controller.has_measurement(profile_key):
        deployment_measurement = plugin.config.deployment_measurement
        if deployment_measurement is None:
            manager.record_fallback("cachetune_measurement_unavailable", restore_length)
            manager.record_request("reuse", "dense_fallback")
            return False
        # Explicit, telemetry-visible policy (not a silent fallback):
        # seed this newly-seen chunk-length bucket from the single
        # deployment-wide measurement, matching the paper's "one
        # deployment profiling pass" calibration cadence.
        controller.record_measurement(profile_key, deployment_measurement)
    decision = controller.select_ratio(profile_key, restore_length, kvcache.layer_num)

    if decision.repair_tokens > 0:
        precomputed_requested = all(
            segment.content_hash.startswith(_RAW_PREFIX) for segment in active_segments
        )
        if not plugin.capable and not precomputed_requested:
            manager.record_fallback("cachetune_capability_unavailable", restore_length)
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
    if decision.repair_tokens > 0 and not plugin.capable:
        fresh_spans = []
        for segment, _, overlap_start, overlap_end in handles:
            fresh_hash = segment.content_hash.replace(_RAW_PREFIX, _FRESH_PREFIX, 1)
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
                    "cachetune_fresh_store_miss",
                    restore_length,
                )
                manager.record_request("reuse", "dense_fallback")
                return False
            try:
                fresh_handle = manager.ensure_device(fresh_handle)
            except Exception:
                manager.record_fallback(
                    "cachetune_fresh_residency_load_failed",
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
        precomputed_backend = PrecomputedCacheTuneBackend(
            kvcache=kvcache,
            spans=fresh_spans,
        )
        probe_backend = precomputed_backend
        recompute_backend = precomputed_backend

    restored_indices = allocator.alloc(restore_length)
    if restored_indices is None or len(restored_indices) != restore_length:
        if restored_indices is not None:
            allocator.free(restored_indices)
        manager.record_fallback("device_allocation_failed", restore_length)
        manager.record_request("reuse", "dense_fallback")
        return False

    # --- Load/recompute overlap: kick off every segment's residency
    # ticket up front (when async prefetch is configured) so a later
    # segment's host->device transfer runs concurrently with an earlier
    # segment's baseline copy. When async prefetch is disabled (the
    # Phase 4 R5 default: scheduler stays S0 LRU / GPU-only / prefetch
    # off) this degrades to a plain sequential `ensure_device` per
    # segment, identical to the R0 path.
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
            resolved = (
                ticket.wait() if ticket is not None else manager.ensure_device(handle)
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
            raise
        if fallback_reasons or stats.recomputed_tokens:
            allocator.free(restored_indices)
            manager.record_request("reuse", "dense_fallback")
            return False
        segment_stats.append(stats)

    # --- Controller-selected repair-token selection over the whole
    # restored span (all segments combined). `decision.repair_tokens` is
    # the sole source of truth for how many positions get selected; see
    # `token_selection.TokenSelection.__post_init__` for the
    # self-validating invariant that enforces this exactly.
    if decision.repair_tokens == 0:
        selection = TokenSelection(
            candidate_positions=tuple(range(restore_length)),
            requested_count=0,
            selected_positions=(),
            stage_scores=(),
        )
    else:

        def deviation_fn(
            probe_layer_id: int, local_positions: torch.Tensor
        ) -> torch.Tensor:
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
            selection = select_repair_tokens(
                list(range(restore_length)),
                stages=plugin.config.probe_stages,
                final_count=decision.repair_tokens,
                deviation_fn=deviation_fn,
            )
        except Exception:
            allocator.free(restored_indices)
            logger.exception(
                "CacheTune repair-token selection failed for request %s",
                getattr(req, "rid", "<unknown>"),
            )
            manager.record_fallback("cachetune_token_selection_failed", restore_length)
            manager.record_request("reuse", "dense_fallback")
            return False

    selected_local = selection.selected_positions
    # Hard invariant (redundant with, but explicit alongside,
    # `TokenSelection.__post_init__`): the number of tokens actually
    # selected for repair must equal the controller's decision exactly.
    # This is the concrete guarantee the task requires -- fail loudly
    # (never dense-fallback) if it is ever violated, since a mismatch
    # here means a bug in this module, not an expected runtime condition.
    if len(selected_local) != decision.repair_tokens:
        allocator.free(restored_indices)
        raise RuntimeError(
            "CacheTune repair token count mismatch: controller selected "
            f"{decision.repair_tokens} but {len(selected_local)} were "
            "actually chosen for repair"
        )

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
                "CacheTune selective repair failed for request %s",
                getattr(req, "rid", "<unknown>"),
            )
            manager.record_fallback("cachetune_selective_repair_failed", restore_length)
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
    req.cachetune_selected_tokens = len(selected_local)
    req.cachetune_candidate_tokens = restore_length
    req.cachetune_ratio = decision.executable_ratio
    req.cachetune_ratio_source = decision.source
    req.cachetune_roofline_ratio = decision.roofline_ratio
    req.cachetune_mode = decision.mode.value
    req.cachetune_predicted_ttft_ms = decision.predicted_ttft_ms
    req.cachetune_recomputed_layers = tuple(
        result.layer_id for result in recompute_results
    )
    req.cachetune_precomputed = precomputed_backend is not None
    manager.record_cachetune_repair(
        selected_tokens=len(selected_local),
        recomputed_layers=len(recompute_results),
        precomputed=precomputed_backend is not None,
        ratio_source=decision.source,
    )
    manager.record_request("reuse", "success")
    return True
