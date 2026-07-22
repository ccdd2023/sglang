"""EPIC-style fixed leading-k attention-sink repair plugin.

EPIC (and the broader attention-sink literature) observes that the first
few tokens of a reused/appended context window receive disproportionate
attention weight, so recomputing just a small leading window of the newly
reused region -- immediately adjacent to the real, exact-matched prefix --
recovers most of the accuracy lost by reusing the remaining body verbatim.

This plugin only decides *policy*: given a resolved, contiguous set of body
copy spans covering the reusable region (built by ``epic_runtime.py`` using
the same store/segment lookup as the raw R0 path), it carves out the
leading-k tokens into an explicit ``DenseRange`` so the common-core
transfer engine (``transfer.execute_reuse_plan``) treats them as a
first-class part of the plan rather than a failure signal. The actual
per-layer genuine recompute for that carved-out range is performed by
``epic_runtime.py`` via ``epic_recompute.LayerwiseEpicExecutor``.

Supported leading-k values: 0, 2, 4, 8, 16, 32 (see
``config.SUPPORTED_EPIC_K_VALUES``). k=0 degenerates to a pure raw-copy
plan identical in shape to the R0 path.
"""

from __future__ import annotations

from dataclasses import dataclass

from .plugins import RecoveryPlugin, RecoveryRequestContext
from .store import ApproxKVSegmentStore
from .types import (
    DenseRange,
    KVReusePlan,
    RecoveryMode,
    SchedulerMetadata,
    TransferSpan,
)

EPIC_LEADING_K_REPAIR_REASON = "epic_leading_k_repair"


def _split_span_at(span: TransferSpan, offset: int) -> TransferSpan | None:
    """Return the tail of ``span`` starting at absolute position ``offset``.

    ``offset`` must fall within ``[span.target_start, span.target_start +
    span.length]``. Returns ``None`` if the whole span is consumed by the
    leading-k window.
    """
    if offset <= span.target_start:
        return span
    if offset >= span.target_start + span.length:
        return None
    consumed = offset - span.target_start
    return TransferSpan(
        source=span.source,
        source_offset=span.source_offset + consumed,
        target_start=offset,
        length=span.length - consumed,
        rope_delta=span.rope_delta,
        chunk_start=span.chunk_start,
        chunk_length=span.chunk_length,
    )


def carve_leading_k(
    spans: tuple[TransferSpan, ...],
    k: int,
) -> tuple[TransferSpan, ...]:
    """Drop/trim the first ``k`` target positions from ``spans``."""
    if k <= 0:
        return spans
    carved: list[TransferSpan] = []
    for span in spans:
        tail = _split_span_at(span, k)
        if tail is not None:
            carved.append(tail)
    return tuple(carved)


@dataclass(frozen=True)
class EPICLeadingKPlugin(RecoveryPlugin):
    """Fixed leading-k attention-sink repair recovery plugin."""

    k: int
    attention_sink: bool = True

    @property
    def name(self) -> str:
        return "epic"

    def leading_k_window(self, restore_length: int) -> int:
        """Clamp the configured k to what the restorable region can hold.

        At least one token must remain for the raw/body copy invariant
        checks to make sense; if the whole restorable region is smaller
        than k, the entire region becomes the leading-k repair window.
        """
        if restore_length <= 0:
            return 0
        return min(self.k, restore_length)

    def build_plan(
        self,
        context: RecoveryRequestContext,
        store: ApproxKVSegmentStore,
    ) -> KVReusePlan:
        resolved_spans = tuple(context.custom_metadata.get("resolved_spans", ()))
        restore_length = len(context.target_token_ids)
        k = self.leading_k_window(restore_length)
        remaining_spans = carve_leading_k(resolved_spans, k)
        dense_ranges = (
            (DenseRange(0, k, reason=EPIC_LEADING_K_REPAIR_REASON),) if k > 0 else ()
        )
        return KVReusePlan(
            target_token_ids=context.target_token_ids,
            recovery_mode=RecoveryMode.COPY,
            copied_spans=remaining_spans,
            dense_ranges=dense_ranges,
            require_full_coverage=True,
        )

    def scheduler_metadata(
        self,
        context: RecoveryRequestContext,
    ) -> tuple[SchedulerMetadata, ...]:
        restore_length = len(context.target_token_ids)
        k = self.leading_k_window(restore_length)
        return (
            SchedulerMetadata(
                object_id=context.request_id,
                resident_bytes=0,
                workflow_stage=f"epic_leading_k={k}",
            ),
        )
