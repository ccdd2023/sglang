from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class CacheObjectKind(str, Enum):
    EXACT_VARIANT = "exact_variant"
    CANONICAL_BASE = "canonical_base"
    CONTEXT_ANCHOR = "context_anchor"
    REPAIR_METADATA = "repair_metadata"


class EvictionPolicy(str, Enum):
    LRU = "lru"
    STEPS_ONLY = "steps_only"
    BELADY_ORACLE = "belady_oracle"
    VALUE_DENSITY = "value_density"
    HIERARCHICAL = "hierarchical"


@dataclass(frozen=True)
class CacheCandidate:
    object_id: str
    resident_bytes: int
    last_access_step: int
    dense_cost_ms: float
    recovery_cost_ms: float
    kind: CacheObjectKind
    steps_to_execution: int | None = None
    oracle_next_use_step: int | None = None
    reuse_frequency: float = 1.0
    retired: bool = False
    pinned: bool = False

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must be non-empty")
        if self.resident_bytes <= 0:
            raise ValueError("resident_bytes must be positive")
        if self.last_access_step < 0:
            raise ValueError("last_access_step must be non-negative")
        if (
            min(
                self.dense_cost_ms,
                self.recovery_cost_ms,
                self.reuse_frequency,
            )
            < 0
        ):
            raise ValueError("costs and frequency must be non-negative")
        if self.steps_to_execution is not None and self.steps_to_execution < 0:
            raise ValueError("steps_to_execution must be non-negative")

    @property
    def saved_ms(self) -> float:
        return max(0.0, self.dense_cost_ms - self.recovery_cost_ms)

    def value_density(self) -> float:
        distance = (
            math.inf if self.steps_to_execution is None else self.steps_to_execution
        )
        urgency = 0.0 if math.isinf(distance) else 1.0 / (1.0 + distance)
        return self.saved_ms * self.reuse_frequency * urgency / self.resident_bytes


def _future_first_key(value: int | None) -> tuple[int, float]:
    if value is None:
        return (0, 0.0)
    return (1, -float(value))


def rank_for_eviction(
    candidates: Sequence[CacheCandidate],
    *,
    policy: EvictionPolicy,
    current_step: int,
) -> list[CacheCandidate]:
    if current_step < 0:
        raise ValueError("current_step must be non-negative")
    evictable = [candidate for candidate in candidates if not candidate.pinned]

    if policy == EvictionPolicy.LRU:
        return sorted(
            evictable,
            key=lambda item: (item.last_access_step, item.object_id),
        )
    if policy == EvictionPolicy.STEPS_ONLY:
        return sorted(
            evictable,
            key=lambda item: (
                *_future_first_key(item.steps_to_execution),
                item.last_access_step,
                item.object_id,
            ),
        )
    if policy == EvictionPolicy.BELADY_ORACLE:
        return sorted(
            evictable,
            key=lambda item: (
                *_future_first_key(
                    None
                    if item.oracle_next_use_step is None
                    else item.oracle_next_use_step - current_step
                ),
                item.last_access_step,
                item.object_id,
            ),
        )
    if policy == EvictionPolicy.VALUE_DENSITY:
        return sorted(
            evictable,
            key=lambda item: (
                item.value_density(),
                item.last_access_step,
                item.object_id,
            ),
        )
    if policy == EvictionPolicy.HIERARCHICAL:
        kind_order = {
            CacheObjectKind.EXACT_VARIANT: 1,
            CacheObjectKind.REPAIR_METADATA: 2,
            CacheObjectKind.CONTEXT_ANCHOR: 3,
            CacheObjectKind.CANONICAL_BASE: 4,
        }
        return sorted(
            evictable,
            key=lambda item: (
                0 if item.retired else kind_order[item.kind],
                item.value_density(),
                item.last_access_step,
                item.object_id,
            ),
        )
    raise ValueError(f"unsupported eviction policy: {policy}")


def select_victims(
    candidates: Sequence[CacheCandidate],
    *,
    bytes_to_free: int,
    policy: EvictionPolicy,
    current_step: int,
) -> tuple[CacheCandidate, ...]:
    if bytes_to_free <= 0:
        return ()
    victims = []
    freed = 0
    for candidate in rank_for_eviction(
        candidates,
        policy=policy,
        current_step=current_step,
    ):
        victims.append(candidate)
        freed += candidate.resident_bytes
        if freed >= bytes_to_free:
            return tuple(victims)
    raise MemoryError("insufficient evictable cache capacity")
