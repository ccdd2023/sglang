from __future__ import annotations

from enum import Enum

from .class_order import s4_class, s4_next_use_key
from .types import CrossStoreObject


class PolicyKind(str, Enum):
    S0_LRU = "s0_lru"
    S4_HIERARCHICAL = "s4_hierarchical"


class CrossStorePolicy:
    def __init__(self, kind: PolicyKind) -> None:
        self.kind = kind

    def victim_key(self, item: CrossStoreObject) -> tuple:
        if self.kind == PolicyKind.S0_LRU:
            return (item.event_ordinal, item.object_id)
        if self.kind != PolicyKind.S4_HIERARCHICAL:
            raise ValueError(f"unsupported policy: {self.kind}")
        effective_class = s4_class(
            item.kind.value,
            retired=item.retired,
            recoverable_from_lower_tier=item.recoverable_from_lower_tier,
        )
        return (
            effective_class,
            item.value_density,
            s4_next_use_key(item.next_use_ordinal),
            item.event_ordinal,
            item.object_id,
        )
