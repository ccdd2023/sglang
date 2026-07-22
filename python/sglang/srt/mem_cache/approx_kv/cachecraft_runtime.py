"""Cache-Craft server-request-path integration.

Mirrors the structure of the common-core `runtime.restore_request_prefix`
(exact cache preferred first, real last-token forward always required,
dense fallback on anything unsupported) but drives execution through
`CacheCraftPlugin` and `CacheCraftRecomputeBackend` so that a partial-repair
decision invokes a real selected-token recompute hook rather than only
recording a fallback reason.

Known blockers / capability gates (see module docstrings in
`cachecraft_recompute.py` and `cachecraft_attention.py` for the mechanism-
level detail):

1. No production `ChunkRecomputeHook` is wired to a real model runner in
   this worktree: SGLang's `ForwardMode.TARGET_VERIFY` (the closest existing
   "recompute selected tokens against an existing KV context" mechanism) is
   only reachable from inside the speculative-decoding worker pipeline
   (`sglang.srt.speculative.eagle_worker_v2`, `spec_utils.py`), not as a
   standalone API. Until such a hook (or an eager-attention equivalent) is
   exposed, `recompute_hook` will be `None` on any real GPU server run, and
   `restore_request_via_cachecraft` always safely falls back to a real
   dense prefill for the whole chunk (returns `False`) whenever Cache-Craft
   would otherwise attempt a partial repair.
2. The wire-level request schema (`request.parse_request_metadata`, frozen
   common core) has no field for the new prompt's chunk order or per-chunk
   attention profile; those are Cache-Craft-specific data that this module
   reads from request-local attributes
   (`req.approx_kv_new_prefix_order`) populated out-of-band by the caller,
   not from `custom_params.approx_kv` itself. Extending the wire schema
   would require editing frozen common-core `request.py`, which this
   worktree intentionally does not do.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from .cachecraft_metrics import CacheCraftDecision
from .cachecraft_plugin import CacheCraftPlugin
from .cachecraft_recompute import (
    CacheCraftRecomputeBackend,
    CacheCraftUnsupportedError,
    ChunkRecomputeHook,
)
from .plugins import RecoveryRequestContext
from .radix_backend import RadixKVTransferBackend, RoPEConfig
from .request import ApproxKVRequestOperation
from .types import KVReusePlan, KVSegmentKey, SegmentKind, token_ids_hash


def _segment_key(tokens: tuple[int, ...], segment: Any, metadata: Any) -> KVSegmentKey:
    return KVSegmentKey(
        content_hash=segment.content_hash,
        token_hash=token_ids_hash(tokens),
        token_count=len(tokens),
        model_fingerprint=metadata.model_fingerprint,
        cache_dtype=metadata.cache_dtype,
        kind=SegmentKind.ARTIFACT,
    )


def _with_rope_delta(plan: KVReusePlan, rope_delta: int) -> KVReusePlan:
    if rope_delta == 0:
        return plan
    updated_spans = tuple(
        dataclasses.replace(span, rope_delta=rope_delta) for span in plan.copied_spans
    )
    return dataclasses.replace(plan, copied_spans=updated_spans)


def restore_request_via_cachecraft(
    tree_cache: Any,
    req: Any,
    *,
    plugin: CacheCraftPlugin,
    recompute_hook: ChunkRecomputeHook | None,
) -> bool:
    """Attempt Cache-Craft chunk-cache reuse for `req`.

    Returns `True` only when the chunk was fully restored (direct reuse or
    a partial repair with a real recompute hook actually invoked) and
    `req.prefix_indices` was extended accordingly. Returns `False` in every
    other case (exact cache preferred, store miss, full-recompute decision,
    unavailable recompute hook, or any mechanical invariant violation), in
    which case the caller must run a real dense prefill for the chunk --
    exactly the same contract as `runtime.restore_request_prefix`.
    """
    metadata = getattr(req, "approx_kv_metadata", None)
    manager = getattr(tree_cache, "approx_kv", None)
    if (
        metadata is None
        or metadata.operation != ApproxKVRequestOperation.REUSE
        or metadata.plugin != plugin.name
        or manager is None
        or not manager.config.core_enabled
    ):
        return False
    if req.needs_host_load_back():
        manager.record_request("reuse", "exact_host_preferred")
        return False
    if len(metadata.segments) != 1:
        raise ValueError("cachecraft currently manages exactly one chunk per request")
    segment = metadata.segments[0]

    exact_length = len(req.prefix_indices)
    reusable_limit = len(req.full_untruncated_fill_ids) - 1
    chunk_start = segment.target_start
    chunk_end = min(segment.target_end, reusable_limit)
    chunk_length = chunk_end - chunk_start
    if chunk_length <= 0 or chunk_start != exact_length:
        manager.record_request("reuse", "exact")
        return False

    target_tokens = tuple(
        int(token) for token in req.full_untruncated_fill_ids[chunk_start:chunk_end]
    )
    chunk_key = _segment_key(target_tokens, segment, metadata)
    context = RecoveryRequestContext(
        request_id=str(getattr(req, "rid", "cachecraft")),
        target_token_ids=target_tokens,
        exact_prefix_length=0,
        custom_metadata={
            "chunk_id": segment.content_hash,
            "chunk_key": chunk_key,
            "chunk_start": 0,
            "chunk_length": chunk_length,
            "new_prefix_order": getattr(req, "approx_kv_new_prefix_order", ()),
        },
    )
    plan = plugin.build_plan(context, manager.store)
    trace = plugin.last_trace
    if trace is None:
        raise RuntimeError("cachecraft plugin did not record a decision trace")

    if trace.decision is CacheCraftDecision.FULL_RECOMPUTE:
        manager.record_fallback("cachecraft_full_recompute", chunk_length)
        manager.record_request("reuse", "dense_fallback")
        return False

    handle = manager.store.lookup(chunk_key)
    if handle is None:
        manager.record_fallback("store_miss", chunk_length)
        manager.record_request("reuse", "dense_fallback")
        return False
    try:
        handle = manager.ensure_device(handle)
    except Exception:
        manager.record_fallback("residency_load_failed", chunk_length)
        manager.record_request("reuse", "dense_fallback")
        return False

    rope_delta = chunk_start - handle.source_start
    plan = _with_rope_delta(plan, rope_delta)

    allocator = tree_cache.token_to_kv_pool_allocator
    target_indices = allocator.alloc(chunk_length)
    if target_indices is None or len(target_indices) != chunk_length:
        if target_indices is not None:
            allocator.free(target_indices)
        manager.record_fallback("device_allocation_failed", chunk_length)
        manager.record_request("reuse", "dense_fallback")
        return False

    rope_config = manager.rope_config or RoPEConfig(
        rotary_dim=0,
        base=10000.0,
        is_neox_style=True,
    )
    inner_backend = RadixKVTransferBackend(
        allocator=allocator,
        target_indices=lambda start, length: target_indices[start : start + length],
        dense_prefill=lambda *args, **kwargs: None,
        rope=rope_config,
    )
    backend = CacheCraftRecomputeBackend(
        inner=inner_backend,
        kvcache=allocator.get_kvcache(),
        target_indices=lambda start, length: target_indices[start : start + length],
        token_ids=lambda start, length: target_tokens[start : start + length],
        recompute_hook=recompute_hook,
    )

    try:
        stats = manager.execute(plan, backend)
    except CacheCraftUnsupportedError:
        allocator.free(target_indices)
        return False
    except Exception:
        allocator.free(target_indices)
        raise

    if backend.unsupported_reasons or not stats.mechanically_valid:
        allocator.free(target_indices)
        reason = (
            "cachecraft_recompute_hook_unavailable"
            if backend.unsupported_reasons
            else "cachecraft_invariant_violation"
        )
        manager.record_fallback(reason, chunk_length)
        manager.record_request("reuse", "dense_fallback")
        return False

    req.prefix_indices = torch.cat(
        (
            req.prefix_indices,
            target_indices.to(
                device=req.prefix_indices.device,
                dtype=req.prefix_indices.dtype,
            ),
        )
    )
    req.approx_kv_restored_len = chunk_length
    req.approx_kv_stats = stats
    req.cachecraft_trace = trace
    req.cachecraft_recompute_invocations = tuple(backend.invocations)
    outcome = (
        "success_direct_reuse"
        if trace.decision is CacheCraftDecision.DIRECT_REUSE
        else "success_partial_repair"
    )
    manager.record_request("reuse", outcome)
    return True
