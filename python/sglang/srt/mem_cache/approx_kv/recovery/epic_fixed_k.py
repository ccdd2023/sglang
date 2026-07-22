from __future__ import annotations

from typing import Sequence

from ..types import (
    DenseRange,
    KVReusePlan,
    RecoveryMode,
    TransferSpan,
)
from .common import (
    ReusableSegment,
    dense_uncovered_ranges,
    force_dense_reason,
    validate_segments,
)


def build_epic_fixed_k_plan(
    *,
    target_token_ids: Sequence[int],
    segments: Sequence[ReusableSegment],
    repair_tokens: int,
) -> KVReusePlan:
    if repair_tokens < 0:
        raise ValueError("repair_tokens must be non-negative")

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

        head = min(repair_tokens, length)
        if head == length:
            dense.append(
                DenseRange(
                    segment.target_start,
                    length,
                    "full_fixed_k_budget",
                )
            )
            continue
        if head:
            dense.append(
                DenseRange(
                    segment.target_start,
                    head,
                    "epic_fixed_k",
                )
            )

        source = segment.source
        assert source is not None
        target_start = segment.target_start + head
        copied.append(
            TransferSpan(
                source=source,
                source_offset=head,
                target_start=target_start,
                length=length - head,
                rope_delta=target_start - (source.source_start + head),
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
        recovery_mode=RecoveryMode.EPIC_FIXED_K,
        copied_spans=tuple(copied),
        dense_ranges=tuple(dense),
        require_full_coverage=True,
    )
