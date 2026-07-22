from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .policy import (
    CacheCandidate,
    EvictionPolicy,
    select_victims,
)


class PrefetchMode(str, Enum):
    OFF = "off"
    FREE_SPACE_ONLY = "free_space_only"
    RETIRED_ONLY = "retired_only"
    ORACLE = "oracle"


@dataclass(frozen=True)
class PrefetchRequest:
    object_id: str
    resident_bytes: int
    miss_cost_ms: float
    oracle_next_use_step: int

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must be non-empty")
        if self.resident_bytes <= 0:
            raise ValueError("resident_bytes must be positive")
        if self.miss_cost_ms < 0 or self.oracle_next_use_step < 0:
            raise ValueError("prefetch cost and next-use step must be valid")


@dataclass(frozen=True)
class PrefetchDecision:
    admitted: bool
    victims: tuple[CacheCandidate, ...] = ()
    expected_gain_ms: float = 0.0
    reason: str = ""


def admit_prefetch(
    request: PrefetchRequest,
    *,
    mode: PrefetchMode,
    free_bytes: int,
    candidates: Sequence[CacheCandidate],
    current_step: int,
) -> PrefetchDecision:
    if free_bytes < 0:
        raise ValueError("free_bytes must be non-negative")
    if mode == PrefetchMode.OFF:
        return PrefetchDecision(False, reason="prefetch_disabled")
    if free_bytes >= request.resident_bytes:
        return PrefetchDecision(
            True,
            expected_gain_ms=request.miss_cost_ms,
            reason="free_space",
        )
    if mode == PrefetchMode.FREE_SPACE_ONLY:
        return PrefetchDecision(False, reason="insufficient_free_space")

    need = request.resident_bytes - free_bytes
    if mode == PrefetchMode.RETIRED_ONLY:
        retired = [candidate for candidate in candidates if candidate.retired]
        try:
            victims = select_victims(
                retired,
                bytes_to_free=need,
                policy=EvictionPolicy.LRU,
                current_step=current_step,
            )
        except MemoryError:
            return PrefetchDecision(
                False,
                reason="insufficient_retired_capacity",
            )
    elif mode == PrefetchMode.ORACLE:
        try:
            victims = select_victims(
                candidates,
                bytes_to_free=need,
                policy=EvictionPolicy.BELADY_ORACLE,
                current_step=current_step,
            )
        except MemoryError:
            return PrefetchDecision(
                False,
                reason="insufficient_evictable_capacity",
            )
        if any(
            victim.oracle_next_use_step is not None
            and victim.oracle_next_use_step <= request.oracle_next_use_step
            for victim in victims
        ):
            return PrefetchDecision(
                False,
                victims=victims,
                reason="victim_needed_no_later_than_target",
            )
    else:
        raise ValueError(f"unsupported prefetch mode: {mode}")

    victim_cost = sum(victim.recovery_cost_ms for victim in victims)
    expected_gain = request.miss_cost_ms - victim_cost
    if expected_gain <= 0:
        return PrefetchDecision(
            False,
            victims=victims,
            expected_gain_ms=expected_gain,
            reason="non_positive_expected_gain",
        )
    return PrefetchDecision(
        True,
        victims=victims,
        expected_gain_ms=expected_gain,
        reason="admitted",
    )
