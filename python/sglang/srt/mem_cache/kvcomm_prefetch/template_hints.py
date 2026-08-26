"""Compile template-admitted file islands into KVPrefetchHint objects.

The template answers *which* spans may be copied. This module only emits
*when/where* those same keys should become device-resident. It does not
estimate Attention, does not change the admit set, and does not rotate K.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from sglang.srt.mem_cache.kvcomm.types import (
    KVPrefetchHint,
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
)


@dataclass(frozen=True)
class TemplatePrefetchIsland:
    """One compiler-admitted file-module island, independent of copy enable."""

    source_id: str
    key: KVSegmentKey
    remaining_uses: int
    next_group_index: int | None = None
    eligible: bool = True
    token_ids_match: bool = True
    version_valid: bool = True
    delta_nonzero: bool = True
    single_file_repository_code: bool = True


@dataclass(frozen=True)
class TemplatePrefetchPlan:
    hints: tuple[KVPrefetchHint, ...]
    skipped_source_ids: tuple[str, ...]
    skip_reasons: tuple[str, ...]


@dataclass(frozen=True)
class NextIslandObservation:
    """A materialized prefix or file-island span scored from protocol signals.

    ``later_roles_in_protocol`` is the number of downstream roles in the
    rolling-6 coding protocol that typically re-attach the same file (e.g.
    3 after a planner read). It is not future ``target_uses``.

    ``SegmentKind.PREFIX`` is the lossless radix prefix (Δ=0).
    ``SegmentKind.MIDDLE`` is the shifted file island (Δ≠0).
    """

    source_id: str
    key: KVSegmentKey
    later_roles_in_protocol: int
    eligible: bool = True
    token_ids_match: bool = True
    version_valid: bool = True
    delta_nonzero: bool = True
    single_file_repository_code: bool = True
    residency: ResidencyTier | None = None
    sequential_next_use: bool = False
    span_kind: SegmentKind = SegmentKind.MIDDLE
    priority_override: int | None = None


# PREFIX always ranks above MIDDLE: later-roles (≤3) + class bonus (≤4) < 8.
PREFIX_PRIORITY_FLOOR = 8


def protocol_later_roles(
    policy_label: str, explicit: int | None = None
) -> int:
    """Later roles from the coding protocol, not remaining target_uses."""
    if explicit is not None:
        return int(explicit)
    if "coding" in (policy_label or ""):
        return 3
    return 0


def _skip_reason(island: TemplatePrefetchIsland) -> str | None:
    if not island.eligible:
        return "not_eligible"
    if not island.single_file_repository_code:
        return "not_single_file_repository_code"
    if not island.token_ids_match:
        return "token_ids_mismatch"
    if not island.version_valid:
        return "version_invalid"
    if not island.delta_nonzero:
        return "zero_shift"
    if island.remaining_uses <= 0:
        return "no_remaining_uses"
    return None


def compile_template_prefetch_hints(
    islands: Sequence[TemplatePrefetchIsland],
    *,
    group_eta_s: Mapping[int, float] | None = None,
    now_s: float = 0.0,
    default_deadline_s: float | None = None,
    target_tier: ResidencyTier = ResidencyTier.DEVICE,
) -> TemplatePrefetchPlan:
    """Whitelist-only hints. Priority is remaining declared uses.

    ``deadline_s`` is the scheduler's relative queue-and-load budget. When
    ``group_eta_s`` maps a next group index to an absolute time, the budget is
    ``max(0, eta - now_s)``. Earlier groups therefore rank first via deadline,
    then higher remaining uses via priority.
    """

    hints: list[KVPrefetchHint] = []
    skipped: list[str] = []
    reasons: list[str] = []
    # Dedup by key: keep the tighter deadline and the higher priority.
    best: dict[KVSegmentKey, KVPrefetchHint] = {}

    for island in islands:
        reason = _skip_reason(island)
        if reason is not None:
            skipped.append(island.source_id)
            reasons.append(f"{island.source_id}:{reason}")
            continue
        deadline_s = default_deadline_s
        if (
            group_eta_s is not None
            and island.next_group_index is not None
            and island.next_group_index in group_eta_s
        ):
            deadline_s = max(0.0, float(group_eta_s[island.next_group_index]) - now_s)
        hint = KVPrefetchHint(
            key=island.key,
            target_tier=target_tier,
            deadline_s=deadline_s,
            priority=int(island.remaining_uses),
        )
        existing = best.get(island.key)
        if existing is None:
            best[island.key] = hint
            continue
        existing_deadline = (
            float("inf") if existing.deadline_s is None else existing.deadline_s
        )
        new_deadline = float("inf") if hint.deadline_s is None else hint.deadline_s
        if (new_deadline, -hint.priority) < (existing_deadline, -existing.priority):
            best[island.key] = hint

    ordered = sorted(
        best.values(),
        key=lambda hint: (
            float("inf") if hint.deadline_s is None else hint.deadline_s,
            -hint.priority,
            hint.key,
        ),
    )
    return TemplatePrefetchPlan(
        hints=tuple(ordered),
        skipped_source_ids=tuple(skipped),
        skip_reasons=tuple(reasons),
    )


def _next_island_skip_reason(obs: NextIslandObservation) -> str | None:
    if obs.key is not None and obs.key.kind == SegmentKind.PREFIX:
        kind = SegmentKind.PREFIX
    else:
        kind = obs.span_kind
    if not obs.eligible:
        return "not_eligible"
    if not obs.token_ids_match:
        return "token_ids_mismatch"
    if not obs.version_valid:
        return "version_invalid"
    if (
        obs.residency == ResidencyTier.DEVICE
        and obs.sequential_next_use
    ):
        return "no_overlap_window"
    if kind == SegmentKind.PREFIX:
        # Exact Δ=0 prefix is reused by radix; it is not a shifted island.
        return None
    if not obs.single_file_repository_code:
        return "not_single_file_repository_code"
    if not obs.delta_nonzero:
        return "zero_shift"
    if obs.later_roles_in_protocol <= 0:
        return "no_protocol_reread"
    return None


def compile_next_island_prefetch_hints(
    observations: Sequence[NextIslandObservation],
    *,
    default_deadline_s: float | None = None,
    target_tier: ResidencyTier = ResidencyTier.DEVICE,
) -> TemplatePrefetchPlan:
    """Prefetch hints for lossless prefix and shifted islands.

    PREFIX spans are Δ=0 radix reuse. MIDDLE spans are template-admitted
    file islands. Emitted keys do not grow the admit/copy set. K is not
    rotated.
    """

    skipped: list[str] = []
    reasons: list[str] = []
    best: dict[KVSegmentKey, KVPrefetchHint] = {}
    for obs in observations:
        reason = _next_island_skip_reason(obs)
        if reason is not None:
            skipped.append(obs.source_id)
            reasons.append(f"{obs.source_id}:{reason}")
            continue
        kind = (
            SegmentKind.PREFIX
            if obs.key is not None and obs.key.kind == SegmentKind.PREFIX
            else obs.span_kind
        )
        base_priority = (
            int(obs.priority_override)
            if obs.priority_override is not None
            else int(obs.later_roles_in_protocol)
        )
        if kind == SegmentKind.PREFIX:
            base_priority = PREFIX_PRIORITY_FLOOR + max(0, base_priority)
        hint = KVPrefetchHint(
            key=obs.key,
            target_tier=target_tier,
            deadline_s=default_deadline_s,
            priority=base_priority,
        )
        existing = best.get(obs.key)
        if existing is None or hint.priority > existing.priority:
            best[obs.key] = hint
    ordered = sorted(
        best.values(),
        key=lambda hint: (-hint.priority, hint.key),
    )
    return TemplatePrefetchPlan(
        hints=tuple(ordered),
        skipped_source_ids=tuple(skipped),
        skip_reasons=tuple(reasons),
    )
