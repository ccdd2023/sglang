from __future__ import annotations

import math
from typing import Mapping, Sequence

from ..types import (
    DenseRange,
    KVReusePlan,
    RecoveryMode,
    TransferSpan,
)
from .common import (
    ReusableSegment,
    contiguous_ranges,
    dense_uncovered_ranges,
    force_dense_reason,
    validate_segments,
)


def _repair_count(length: int, ratio: float) -> int:
    if length <= 0:
        raise ValueError("length must be positive")
    if ratio < 0 or ratio > 1:
        raise ValueError("ratio must lie in [0, 1]")
    return min(length, math.ceil(length * ratio))


def repair_offsets_from_fraction(
    length: int,
    ratio: float,
) -> tuple[int, ...]:
    return tuple(range(_repair_count(length, ratio)))


def repair_offsets_from_scores(
    scores: Sequence[float],
    ratio: float,
) -> tuple[int, ...]:
    count = _repair_count(len(scores), ratio)
    ranked = sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), index),
    )
    return tuple(sorted(ranked[:count]))


def build_selective_repair_plan(
    *,
    target_token_ids: Sequence[int],
    segments: Sequence[ReusableSegment],
    repair_offsets: Mapping[str, Sequence[int]],
) -> KVReusePlan:
    target = tuple(int(token) for token in target_token_ids)
    occupied = validate_segments(
        target_token_ids=target,
        segments=segments,
    )
    dense: list[DenseRange] = []
    copied: list[TransferSpan] = []

    for segment in sorted(segments, key=lambda item: item.target_start):
        length = len(segment.token_ids)
        reason = force_dense_reason(
            target_token_ids=target,
            segment=segment,
        )
        if reason is not None:
            dense.append(DenseRange(segment.target_start, length, reason))
            continue

        selected = sorted(
            set(
                int(offset)
                for offset in repair_offsets.get(
                    segment.segment_id,
                    (),
                )
            )
        )
        if any(offset < 0 or offset >= length for offset in selected):
            raise ValueError(f"repair offset exceeds segment {segment.segment_id}")

        repair_set = set(selected)
        for local_start, run_length in contiguous_ranges(selected):
            dense.append(
                DenseRange(
                    segment.target_start + local_start,
                    run_length,
                    "selective_repair",
                )
            )

        copy_positions = [
            offset for offset in range(length) if offset not in repair_set
        ]
        source = segment.source
        assert source is not None
        for local_start, run_length in contiguous_ranges(copy_positions):
            target_start = segment.target_start + local_start
            copied.append(
                TransferSpan(
                    source=source,
                    source_offset=local_start,
                    target_start=target_start,
                    length=run_length,
                    rope_delta=target_start - (source.source_start + local_start),
                    chunk_start=segment.target_start,
                    chunk_length=length,
                )
            )

    dense.extend(
        dense_uncovered_ranges(
            target_token_ids=target,
            occupied=occupied,
        )
    )
    return KVReusePlan(
        target_token_ids=target,
        recovery_mode=RecoveryMode.SELECTIVE_REPAIR,
        copied_spans=tuple(copied),
        dense_ranges=tuple(dense),
        require_full_coverage=True,
    )
