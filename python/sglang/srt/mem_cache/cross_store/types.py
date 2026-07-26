from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum


class CrossStoreTier(str, Enum):
    DEVICE = "device"
    HOST = "host"


class ObjectProvenance(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"


class CrossStoreKind(str, Enum):
    EXACT_VARIANT = "exact_variant"
    CANONICAL_BASE = "canonical_base"
    REPAIR_STATE = "repair_state"
    PRECOMPUTED_ADAPTER = "precomputed_adapter"
    ANCHOR = "anchor"
    DELTA = "delta"
    HOST_COPY = "host_copy"
    MATERIALIZATION_SCRATCH = "materialization_scratch"
    FILLER = "filler"


@dataclass(frozen=True)
class CrossStoreObject:
    object_id: str
    kind: CrossStoreKind
    tier: CrossStoreTier
    provenance: ObjectProvenance
    token_count: int
    resident_bytes: int
    event_ordinal: int
    generation: int = 1
    dependencies: frozenset[str] = field(default_factory=frozenset)
    dense_cost_ms: float | None = None
    recovery_cost_ms: float | None = None
    next_use_ordinal: int | None = None
    retired: bool = False
    recoverable_from_lower_tier: bool = False
    pinned: bool = False
    leased: bool = False
    in_flight: bool = False
    reserved: bool = False
    evictable: bool = True
    demotable: bool = False

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must be non-empty")
        if self.token_count <= 0 or self.resident_bytes <= 0:
            raise ValueError("token_count and resident_bytes must be positive")
        if self.event_ordinal < 0 or self.generation <= 0:
            raise ValueError("invalid event ordinal or generation")
        if self.object_id in self.dependencies:
            raise ValueError("an object cannot depend on itself")
        if self.dense_cost_ms is not None and self.dense_cost_ms < 0:
            raise ValueError("dense_cost_ms must be non-negative")
        if self.recovery_cost_ms is not None and self.recovery_cost_ms < 0:
            raise ValueError("recovery_cost_ms must be non-negative")

    @property
    def protected(self) -> bool:
        return self.pinned or self.leased or self.in_flight or self.reserved

    @property
    def saved_ms(self) -> float:
        if self.dense_cost_ms is None:
            return 0.0
        return max(0.0, self.dense_cost_ms - (self.recovery_cost_ms or 0.0))

    @property
    def value_density(self) -> float:
        return self.saved_ms / self.resident_bytes

    def touched(self, event_ordinal: int) -> CrossStoreObject:
        return replace(self, event_ordinal=event_ordinal)
