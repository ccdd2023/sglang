from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from sglang.srt.mem_cache.kvcomm.types import (
    DenseRange,
    KVReusePlan,
    KVSegmentHandle,
    TransferSpan,
)


class CodingRisk(str, Enum):
    CRITICAL = "critical"
    STABLE = "stable"


@dataclass(frozen=True)
class CodingSegment:
    """A policy decision for one byte/token-identical code segment."""

    slot_id: str
    target_start: int
    token_ids: tuple[int, ...]
    risk: CodingRisk
    source: KVSegmentHandle | None
    head_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.slot_id:
            raise ValueError("slot_id must be non-empty")
        if self.target_start < 0 or not self.token_ids:
            raise ValueError("coding segment must have valid target bounds")
        if self.head_tokens < 0 or self.head_tokens > len(self.token_ids):
            raise ValueError("head_tokens must lie within the segment")


def _contiguous_ranges(positions: list[int]) -> list[tuple[int, int]]:
    if not positions:
        return []
    result: list[tuple[int, int]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            result.append((start, previous - start + 1))
            start = position
        previous = position
    result.append((start, previous - start + 1))
    return result


def build_coding_reuse_plan(
    *,
    target_token_ids: Sequence[int],
    segments: Sequence[CodingSegment],
) -> KVReusePlan:
    """Translate coding risk decisions into a complete KVCOMM reuse plan.

    This function has no scheduler, eviction, residency-loading, or prefetch
    behavior. Non-resident handles are left to the shared transfer safety gate.
    """

    target = tuple(int(token) for token in target_token_ids)
    occupied: set[int] = set()
    dense: list[DenseRange] = []
    copied: list[TransferSpan] = []

    for segment in sorted(segments, key=lambda item: item.target_start):
        start = segment.target_start
        length = len(segment.token_ids)
        end = start + length
        if end > len(target):
            raise ValueError(f"segment {segment.slot_id} exceeds target prompt")
        positions = set(range(start, end))
        if occupied & positions:
            raise ValueError("coding segments must not overlap")
        occupied |= positions

        target_slice = target[start:end]
        source = segment.source
        force_dense_reason = None
        if target_slice != segment.token_ids:
            force_dense_reason = "target_manifest_mismatch"
        elif source is None:
            force_dense_reason = "missing_source"
        elif source.token_ids != segment.token_ids:
            force_dense_reason = "source_token_mismatch"
        elif segment.risk == CodingRisk.CRITICAL:
            force_dense_reason = "coding_critical"
        elif segment.head_tokens >= length:
            force_dense_reason = "full_head_budget"

        if force_dense_reason is not None:
            dense.append(DenseRange(start, length, force_dense_reason))
            continue

        head = segment.head_tokens
        if head:
            dense.append(DenseRange(start, head, "coding_head_budget"))
        body_start = start + head
        body_length = length - head
        copied.append(
            TransferSpan(
                source=source,
                source_offset=head,
                target_start=body_start,
                length=body_length,
                rope_delta=body_start - (source.source_start + head),
                chunk_start=start,
                chunk_length=length,
            )
        )

    uncovered = sorted(set(range(len(target))) - occupied)
    dense.extend(
        DenseRange(start, length, "outside_coding_segments")
        for start, length in _contiguous_ranges(uncovered)
    )
    return KVReusePlan(
        target_token_ids=target,
        copied_spans=tuple(copied),
        dense_ranges=tuple(dense),
        require_full_coverage=True,
    )
