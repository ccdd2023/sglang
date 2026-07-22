from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..types import DenseRange, KVSegmentHandle


@dataclass(frozen=True)
class ReusableSegment:
    segment_id: str
    target_start: int
    token_ids: tuple[int, ...]
    source: KVSegmentHandle | None

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("segment_id must be non-empty")
        if self.target_start < 0:
            raise ValueError("target_start must be non-negative")
        if not self.token_ids:
            raise ValueError("token_ids must be non-empty")

    @property
    def target_end(self) -> int:
        return self.target_start + len(self.token_ids)


def contiguous_ranges(
    positions: Sequence[int],
) -> list[tuple[int, int]]:
    if not positions:
        return []
    ordered = sorted(set(positions))
    result: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for position in ordered[1:]:
        if position != previous + 1:
            result.append((start, previous - start + 1))
            start = position
        previous = position
    result.append((start, previous - start + 1))
    return result


def validate_segments(
    *,
    target_token_ids: tuple[int, ...],
    segments: Sequence[ReusableSegment],
) -> set[int]:
    occupied: set[int] = set()
    for segment in sorted(segments, key=lambda item: item.target_start):
        if segment.target_end > len(target_token_ids):
            raise ValueError(f"segment {segment.segment_id} exceeds target prompt")
        positions = set(range(segment.target_start, segment.target_end))
        if occupied & positions:
            raise ValueError("reusable segments must not overlap")
        occupied |= positions
    return occupied


def dense_uncovered_ranges(
    *,
    target_token_ids: tuple[int, ...],
    occupied: set[int],
) -> list[DenseRange]:
    uncovered = sorted(set(range(len(target_token_ids))) - occupied)
    return [
        DenseRange(start, length, "outside_reusable_segments")
        for start, length in contiguous_ranges(uncovered)
    ]


def force_dense_reason(
    *,
    target_token_ids: tuple[int, ...],
    segment: ReusableSegment,
) -> str | None:
    target_slice = target_token_ids[segment.target_start : segment.target_end]
    if target_slice != segment.token_ids:
        return "target_manifest_mismatch"
    if segment.source is None:
        return "missing_source"
    if segment.source.token_ids != segment.token_ids:
        return "source_token_mismatch"
    return None
