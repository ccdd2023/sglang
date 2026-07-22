from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..types import (
    AnchorReconstructionSpan,
    DenseRange,
    KVReusePlan,
    KVSegmentHandle,
    RecoveryMode,
)
from .common import contiguous_ranges


@dataclass(frozen=True)
class AnchorCandidate:
    handle: KVSegmentHandle
    embedding: tuple[float, ...]
    placeholder_length: int
    use_count: int = 0
    created_order: int = 0

    def __post_init__(self) -> None:
        if not self.embedding:
            raise ValueError("anchor embedding must be non-empty")
        if self.placeholder_length <= 0:
            raise ValueError("placeholder_length must be positive")
        if self.use_count < 0 or self.created_order < 0:
            raise ValueError("anchor counters must be non-negative")


@dataclass(frozen=True)
class AnchorMatch:
    anchors: tuple[AnchorCandidate, ...]
    weights: tuple[float, ...]
    normalized_entropy: float
    shareable: bool
    reason: str


@dataclass(frozen=True)
class AnchorSegment:
    segment_id: str
    target_start: int
    token_ids: tuple[int, ...]
    base: KVSegmentHandle | None
    match: AnchorMatch

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("segment_id must be non-empty")
        if self.target_start < 0 or not self.token_ids:
            raise ValueError("anchor segment has invalid target bounds")


def _squared_l2(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    return sum(
        (float(left_value) - float(right_value)) ** 2
        for left_value, right_value in zip(left, right)
    )


def match_anchors(
    *,
    target_embedding: Sequence[float],
    target_length: int,
    anchors: Sequence[AnchorCandidate],
    max_anchors: int = 4,
    temperature: float = 1.0,
    entropy_threshold: float = 0.8,
    speed_only: bool = False,
) -> AnchorMatch:
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if max_anchors <= 0:
        raise ValueError("max_anchors must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if entropy_threshold < 0 or entropy_threshold > 1:
        raise ValueError("entropy_threshold must lie in [0, 1]")

    eligible = [
        anchor for anchor in anchors if anchor.placeholder_length >= target_length
    ]
    if not eligible:
        return AnchorMatch(
            anchors=(),
            weights=(),
            normalized_entropy=1.0,
            shareable=False,
            reason="no_length_compatible_anchor",
        )

    ranked = sorted(
        (
            (
                _squared_l2(target_embedding, anchor.embedding),
                -anchor.use_count,
                anchor.created_order,
                anchor,
            )
            for anchor in eligible
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )[:max_anchors]
    logits = [-item[0] / temperature for item in ranked]
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    normalizer = sum(exponentials)
    weights = tuple(value / normalizer for value in exponentials)
    if len(weights) == 1:
        entropy = 0.0
    else:
        entropy = -sum(
            weight * math.log(weight) for weight in weights if weight > 0
        ) / math.log(len(weights))
    shareable = speed_only or entropy <= entropy_threshold
    return AnchorMatch(
        anchors=tuple(item[3] for item in ranked),
        weights=weights,
        normalized_entropy=entropy,
        shareable=shareable,
        reason="speed_only"
        if speed_only
        else ("entropy_pass" if shareable else "entropy_reject"),
    )


def interpolate_delta(
    *,
    base: Sequence[float],
    anchor_deltas: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> tuple[float, ...]:
    if len(anchor_deltas) != len(weights) or not anchor_deltas:
        raise ValueError("anchor delta and weight counts must match")
    if abs(sum(float(weight) for weight in weights) - 1.0) > 1e-6:
        raise ValueError("anchor weights must sum to one")
    if any(len(delta) != len(base) for delta in anchor_deltas):
        raise ValueError("base and delta dimensions must match")
    return tuple(
        float(base[index])
        + sum(
            float(weight) * float(delta[index])
            for weight, delta in zip(weights, anchor_deltas)
        )
        for index in range(len(base))
    )


def build_kvcomm_anchor_plan(
    *,
    target_token_ids: Sequence[int],
    segments: Sequence[AnchorSegment],
) -> KVReusePlan:
    target = tuple(int(token) for token in target_token_ids)
    occupied: set[int] = set()
    dense: list[DenseRange] = []
    reconstructed: list[AnchorReconstructionSpan] = []

    for segment in sorted(segments, key=lambda item: item.target_start):
        length = len(segment.token_ids)
        end = segment.target_start + length
        if end > len(target):
            raise ValueError(
                f"anchor segment {segment.segment_id} exceeds target prompt"
            )
        positions = set(range(segment.target_start, end))
        if occupied & positions:
            raise ValueError("anchor segments must not overlap")
        occupied |= positions

        reason = None
        if target[segment.target_start : end] != segment.token_ids:
            reason = "target_manifest_mismatch"
        elif segment.base is None:
            reason = "missing_canonical_base"
        elif segment.base.token_ids != segment.token_ids:
            reason = "base_token_mismatch"
        elif not segment.match.shareable:
            reason = segment.match.reason
        elif any(
            anchor.handle.token_ids != segment.token_ids
            for anchor in segment.match.anchors
        ):
            reason = "anchor_token_mismatch"

        if reason is not None:
            dense.append(DenseRange(segment.target_start, length, reason))
            continue

        base = segment.base
        assert base is not None
        reconstructed.append(
            AnchorReconstructionSpan(
                base=base,
                anchors=tuple(anchor.handle for anchor in segment.match.anchors),
                weights=segment.match.weights,
                source_offset=0,
                target_start=segment.target_start,
                length=length,
                rope_delta=segment.target_start - base.source_start,
                chunk_start=segment.target_start,
                chunk_length=length,
            )
        )

    uncovered = sorted(set(range(len(target))) - occupied)
    dense.extend(
        DenseRange(start, length, "outside_anchor_segments")
        for start, length in contiguous_ranges(uncovered)
    )
    return KVReusePlan(
        target_token_ids=target,
        recovery_mode=RecoveryMode.KVCOMM_ANCHOR,
        reconstructed_spans=tuple(reconstructed),
        dense_ranges=tuple(dense),
        require_full_coverage=True,
    )
