from __future__ import annotations

"""R0 Raw+RoPE recovery plugin.

This is the Phase 4 "R0" research artifact: a speed-only upper bound that
answers "how fast could cross-context KV reuse be if we skip faithful
reconstruction and just copy raw K/V pages with a whole-span RoPE position
correction". It is explicitly **not** a faithful reproduction of KVCOMM
(``2510.12872``); KVCOMM's base/delta/anchor reconstruction is out of scope
here and belongs to the separate R3 research branch.

The plugin reuses the frozen common-core primitives verbatim:

- identity/equality: :class:`~sglang.srt.mem_cache.approx_kv.types.KVSegmentKey`
  and :func:`~sglang.srt.mem_cache.approx_kv.types.token_ids_hash`.
- the segment store/lease mechanism:
  :class:`~sglang.srt.mem_cache.approx_kv.store.ApproxKVSegmentStore`.
- the generic transfer/validation engine:
  :func:`~sglang.srt.mem_cache.approx_kv.transfer.execute_reuse_plan`.
- the recovery plugin protocol:
  :class:`~sglang.srt.mem_cache.approx_kv.plugins.RecoveryPlugin`.

It does not reimplement or fork the store; it only decides *which* handles
to request and how to shape the resulting :class:`KVReusePlan`.

Required behavior contract (see task spec for Phase 4 R0):

1. Exact match is always attempted first. This plugin never sees tokens
   already covered by the caller's exact Radix prefix match; it only
   receives the remaining suffix past ``exact_prefix_length``.
2. Any tokens it *does* reuse are a raw K/V copy plus a full-key RoPE
   position relocation (:class:`TransferSpan.rope_delta` may be zero,
   positive, or negative -- all three are handled identically by the
   shared ``RadixKVTransferBackend`` since RoPE rotation is just a signed
   angle).
3. Coverage is all-or-nothing *within the leading contiguous run this
   plugin decides to attempt* (see the non-contiguous-coverage note
   below for how that run is chosen): any stale handle, residency miss,
   token mismatch, or missing segment inside that run causes
   :func:`execute_reuse_plan` (invalid/stale coverage) or this module
   (missing coverage) to abort the *entire* recovery attempt for that
   request -- there is no partial/best-effort splicing inside a run.
4. The final prompt token is never included in a copied or dense range --
   callers must always run a real forward pass for it. This module never
   requests coverage past ``len(target_token_ids) - 1``.
5. Approximate reuse never touches the exact Radix trie: this plugin
   works purely in terms of already-registered segments and appended,
   request-local device slots; ownership of the exact prefix tree is
   untouched.

Known hard limitation (documented rather than silently ignored): this
plugin only ever reuses a single *contiguous* run of segments anchored
exactly at the exact-prefix boundary (covering interior segments that
immediately follow a dense/exact head is supported). Arbitrary
non-contiguous selective prefill -- reusing multiple disjoint spans
separated by gaps that must be recomputed in between -- would require
per-gap recompute scheduling inside the model's forward pass, which is
out of scope for this raw-speed upper bound and is deferred to the R2
(selective repair) and R3 (KVCOMM) research branches.

Concretely, when the declared segments are non-contiguous,
:func:`select_contiguous_segments` (invoked by the request-path caller in
``runtime.py`` *before* this plugin is ever asked to build a plan) trims
the segment list down to only the leading contiguous run anchored at
``exact_prefix_length``; everything at or past the first gap is left
completely unattempted by this call -- it is not silently repaired, and
it is not forced through this module's dense-fallback bookkeeping either.
It simply falls outside the returned plan's coverage, so the caller's own
scheduling naturally treats it as an ordinary (non-approximate) prefill.
:class:`RawRoPERecoveryUnavailable` is reserved for cases *within* the
already-narrowed leading run: a segment declared in that run whose source
is missing from the store (never registered, evicted, or otherwise
absent), or a run that, when re-validated inside :func:`build_raw_rope_plan`,
still turns out to contain an internal gap. Both are treated identically
by the caller: abort the whole recovery attempt for this call (dense
fallback), never guess at a partial repair.
"""

from dataclasses import dataclass
from typing import Sequence

from .plugins import RecoveryRequestContext
from .request import ApproxKVRequestSegment
from .store import ApproxKVSegmentStore
from .types import (
    KVReusePlan,
    KVSegmentKey,
    RecoveryMode,
    SchedulerMetadata,
    SegmentKind,
    TransferSpan,
    token_ids_hash,
)
from .radix_backend import RoPEConfig

RAW_ROPE_PLUGIN_NAME = "raw_rope"


def resolve_model_rope_config(model_config) -> RoPEConfig | None:
    hf_config = model_config.hf_config
    model_type = str(getattr(hf_config, "model_type", "")).lower()
    if model_type not in {"qwen2", "qwen3"}:
        return None
    if getattr(hf_config, "rope_scaling", None):
        return None
    head_dim = getattr(hf_config, "head_dim", None)
    if head_dim is None:
        hidden_size = int(hf_config.hidden_size)
        num_heads = int(hf_config.num_attention_heads)
        if hidden_size % num_heads:
            return None
        head_dim = hidden_size // num_heads
    partial_factor = float(getattr(hf_config, "partial_rotary_factor", 1.0))
    rotary_dim = int(int(head_dim) * partial_factor)
    if rotary_dim <= 0 or rotary_dim % 2:
        return None
    return RoPEConfig(
        rotary_dim=rotary_dim,
        base=float(getattr(hf_config, "rope_theta", 10000.0)),
        is_neox_style=True,
    )


class RawRoPERecoveryUnavailable(RuntimeError):
    """Raised when no valid raw+RoPE plan can cover the pending gap.

    Callers must treat this exactly like any other "invalid/stale/missing
    coverage" outcome: fall back to a dense prefill for the whole pending
    span rather than attempting a partial repair.
    """


@dataclass(frozen=True)
class RawRoPERecoveryRequest:
    """Plugin-specific payload carried in ``RecoveryRequestContext.custom_metadata``.

    ``segments`` may include entries that do not end up contributing to the
    final plan (segments before the exact prefix, non-contiguous segments
    past the first gap, etc); :func:`select_contiguous_segments` and
    :meth:`RawRoPERecoveryPlugin.build_plan` are responsible for narrowing
    them down to the actually-usable contiguous run.
    """

    segments: tuple[ApproxKVRequestSegment, ...]
    model_fingerprint: str
    cache_dtype: str

    KEY = "raw_rope_request"


def select_contiguous_segments(
    segments: Sequence[ApproxKVRequestSegment],
    exact_length: int,
    reusable_limit: int,
) -> tuple[ApproxKVRequestSegment, ...]:
    """Return the contiguous run of segments starting at ``exact_length``.

    This is shared, pure metadata logic (no store/manager access) so both
    the request-path orchestration (deciding which handles to promote to
    device residency) and :meth:`RawRoPERecoveryPlugin.build_plan` (deciding
    the actual plan shape) agree on exactly the same candidate set -- there
    is a single implementation, not two independently-maintained ones.
    """
    ordered = sorted(segments, key=lambda segment: segment.target_start)
    active: list[ApproxKVRequestSegment] = []
    next_target = exact_length
    for segment in ordered:
        if segment.target_end <= exact_length:
            # Fully covered by the exact prefix already; irrelevant here.
            continue
        if segment.target_start > next_target:
            # A gap: R0 stops extending the contiguous run here. An
            # interior segment that starts *exactly* at the current
            # boundary (e.g. right after a dense/exact head) is still
            # accepted by the `<=` implied above via `next_target`.
            break
        active.append(segment)
        next_target = max(next_target, segment.target_end)
        if next_target >= reusable_limit:
            break
    return tuple(active)


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


def build_raw_rope_plan(
    *,
    target_token_ids: Sequence[int],
    exact_prefix_length: int,
    segments: Sequence[ApproxKVRequestSegment],
    model_fingerprint: str,
    cache_dtype: str,
    store: ApproxKVSegmentStore,
) -> KVReusePlan:
    """Build a raw-copy + full-key-RoPE :class:`KVReusePlan`.

    Raises :class:`RawRoPERecoveryUnavailable` when the declared segments do
    not form a single contiguous, currently-registered run starting exactly
    at ``exact_prefix_length``. Zero, positive, and negative RoPE position
    deltas are all represented identically via ``TransferSpan.rope_delta``;
    the sign only affects the rotation angle applied downstream.
    """
    reusable_limit = len(target_token_ids) - 1
    if reusable_limit <= exact_prefix_length:
        raise RawRoPERecoveryUnavailable(
            "no room to recover before the final prompt token"
        )

    active = select_contiguous_segments(segments, exact_prefix_length, reusable_limit)
    if not active or active[0].target_start > exact_prefix_length:
        raise RawRoPERecoveryUnavailable(
            "no contiguous raw+RoPE coverage at the exact-prefix boundary"
        )

    restore_end = min(active[-1].target_end, reusable_limit)
    if restore_end <= exact_prefix_length:
        raise RawRoPERecoveryUnavailable("contiguous coverage does not extend past the exact prefix")

    spans: list[TransferSpan] = []
    cursor = exact_prefix_length
    for segment in active:
        overlap_start = max(segment.target_start, exact_prefix_length)
        overlap_end = min(segment.target_end, restore_end)
        if overlap_end <= overlap_start:
            continue
        if overlap_start != cursor:
            raise RawRoPERecoveryUnavailable(
                "raw+RoPE coverage has an internal gap"
            )
        tokens = tuple(
            int(token)
            for token in target_token_ids[segment.target_start : segment.target_end]
        )
        key = _segment_key(
            tokens=tokens,
            content_hash=segment.content_hash,
            model_fingerprint=model_fingerprint,
            cache_dtype=cache_dtype,
        )
        handle = store.lookup(key)
        if handle is None:
            raise RawRoPERecoveryUnavailable(
                f"missing source segment for content_hash={segment.content_hash!r}"
            )
        source_offset = segment.source_offset + (overlap_start - segment.target_start)
        source_position = handle.source_start + source_offset
        # Zero, positive, or negative -- the shared backend applies this as
        # a signed rotation angle regardless of sign.
        rope_delta = overlap_start - source_position
        spans.append(
            TransferSpan(
                source=handle,
                source_offset=source_offset,
                target_start=overlap_start - exact_prefix_length,
                length=overlap_end - overlap_start,
                rope_delta=rope_delta,
                chunk_start=0,
                chunk_length=restore_end - exact_prefix_length,
            )
        )
        cursor = overlap_end

    if cursor != restore_end:
        raise RawRoPERecoveryUnavailable(
            "raw+RoPE coverage does not reach the requested restore end"
        )

    return KVReusePlan(
        target_token_ids=tuple(
            int(token) for token in target_token_ids[exact_prefix_length:restore_end]
        ),
        recovery_mode=RecoveryMode.COPY,
        copied_spans=tuple(spans),
        require_full_coverage=True,
    )


class RawRoPERecoveryPlugin:
    """Formal :class:`RecoveryPlugin` wrapper around :func:`build_raw_rope_plan`.

    Registered under :data:`RAW_ROPE_PLUGIN_NAME` only when the explicit
    ``SGLANG_APPROX_KV_RAW_ROPE`` gate is enabled (see
    :class:`~sglang.srt.mem_cache.approx_kv.config.ApproxKVFeatureConfig`).
    """

    @property
    def name(self) -> str:
        return RAW_ROPE_PLUGIN_NAME

    def build_plan(
        self,
        context: RecoveryRequestContext,
        store: ApproxKVSegmentStore,
    ) -> KVReusePlan:
        payload = context.custom_metadata.get(RawRoPERecoveryRequest.KEY)
        if not isinstance(payload, RawRoPERecoveryRequest):
            raise RawRoPERecoveryUnavailable(
                "raw_rope plugin requires a RawRoPERecoveryRequest payload "
                "under custom_metadata['raw_rope_request']"
            )
        return build_raw_rope_plan(
            target_token_ids=context.target_token_ids,
            exact_prefix_length=context.exact_prefix_length,
            segments=payload.segments,
            model_fingerprint=payload.model_fingerprint,
            cache_dtype=payload.cache_dtype,
            store=store,
        )

    def scheduler_metadata(
        self,
        context: RecoveryRequestContext,
    ) -> tuple[SchedulerMetadata, ...]:
        # R0 is speed-only and carries no scheduling policy or priority
        # opinion; Phase 5 scheduler axes are explicitly out of scope here.
        return ()
