from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode

# Token-type role constants (mirrored from radix_cache.py to avoid circular imports)
# These must match the values in sglang.srt.mem_cache.radix_cache
_ROLE_TYPE_SYSTEM = 1   # Tier-0: universal system prompt
_ROLE_TYPE_ROLE   = 2   # Tier-1: role-based imports/signatures
_ROLE_TYPE_TASK   = 3   # Tier-2: workflow-specific task context
# 0 = unknown / Tier-3 dynamic suffix


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction with multi-signal priority scoring.

    Priority values represent the absolute step number at which a prefix will
    next be needed (higher value = further in the future = less urgent).

    Eviction order: nodes with the LOWEST sort key are evicted first.
    The key is designed so that:
      - Tier: TASK/unknown evicted before ROLE, ROLE before SYSTEM
      - Within Tier: larger critical_path_distance evicted first
      - Within same distance: larger priority evicted first
      - Final tiebreaker: LRU (older last_access_time evicted first)

    Two complementary signals:

    1. DAG critical-path proximity (PRIMARY):
       - crit_distance = "how far from leaf execution"
       - smaller crit_distance = closer to execution = MORE urgent to protect
       - Formula: -crit_distance × CRIT_WEIGHT (subtractive, so late = lower score)
       - CRIT_WEIGHT=20: crit_dist range 1-5 gives 5-25 points separation within each tier.
         This is enough to rank "late stage" nodes correctly (TASK late stage evicted
         before TASK early stage) without crossing Tier boundaries.

    2. Role-type retention boost (SECONDARY - Tier separator):
       - Prevents crit_distance from crossing Tier boundaries
       - SYSTEM=10000, ROLE=5000, TASK=0
       - Tier gap of 5000 >> crit_dist range of 25 → role_type is stable

    Key:
      (role_rank, -crit_distance, -priority, last_access_time)

    Where role_rank: TASK/unknown=0, ROLE=1, SYSTEM=2 (larger = more protected).
    """

    CRIT_WEIGHT = 20
    ROLE_TYPE_BOOST = {
        _ROLE_TYPE_SYSTEM: 10000,
        _ROLE_TYPE_ROLE: 5000,
        _ROLE_TYPE_TASK: 0,
    }

    def get_priority(self, node: "TreeNode") -> Tuple[int, int, int, float]:
        if getattr(node, "lock_ref", 0) > 0:
            return (float("inf"), 0, 0, 0)

        crit_distance = max(1, getattr(node, "critical_path_distance", 1))
        if getattr(node, "reuse_mode", "") == "lossy":
            crit_distance = max(1, crit_distance - 1)
            if float(getattr(node, "reuse_confidence", 0.0) or 0.0) >= 0.75:
                crit_distance = max(1, crit_distance - 1)
        if node.role_type == _ROLE_TYPE_SYSTEM:
            role_rank = 2
        elif node.role_type == _ROLE_TYPE_ROLE:
            role_rank = 1
        else:
            role_rank = 0

        return (role_rank, -crit_distance, -node.priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Priority Logic:
        # Smaller value = Evicted earlier.
        #
        # Segment 0 (Probationary): hit_count < threshold
        # Segment 1 (Protected): hit_count >= threshold
        #
        # Tuple comparison: (segment, last_access_time)
        # Nodes in segment 0 will always be evicted before segment 1.
        # Inside the same segment, older nodes (smaller time) are evicted first.

        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)


class TieredPriorityStrategy(EvictionStrategy):
    """Tiered Priority eviction (v5: Priority-Compatible with Tier Metadata).

    Design (v5: Priority-Compatible):
    This version adopts the SAME eviction formula as PriorityStrategy to ensure
    identical protection for shared prefix nodes. The tier concept is preserved
    for OTHER optimizations (e.g., prefetch hints) but does NOT affect eviction.

    Key difference from v4 (INF Protection):
    - v4: float("inf") as first dimension for shared tier
           → Problem: shared tier internal eviction can still disrupt shared prefix
           → Result: Phase 3 TTFT = 568ms (partial shared prefix loss)
    - v5: Same formula as Priority (role_rank with large gap)
           → Result: Should match Priority Phase 3 TTFT (~6ms)

    The eviction formula is IDENTICAL to PriorityStrategy:
      (role_rank, -crit_distance, -priority, last_access_time)

    Where role_rank: SYSTEM=2, ROLE=1, TASK=0 (larger = more protected).
    The gap of 5000 between tiers ensures role_type dominates over crit_distance.

    NOTE: This v5 essentially equals Priority in eviction behavior.
          The "tiered" distinction now only applies to metadata tracking
          (e.g., for prefetch hints), not eviction itself.
    """

    CRIT_WEIGHT = 20
    SHARED_RANK_SYSTEM = 2
    SHARED_RANK_ROLE = 1

    def __init__(
        self,
        shared_layer: str = "priority_compatible",
        private_layer: str = "priority_compatible",
        crit_weight: int = 20,
    ):
        """
        Args:
            shared_layer: Eviction strategy for shared prefixes (Tier-0/1).
            private_layer: Eviction strategy for private prefixes (Tier-2).
            crit_weight: Weight for critical_path_distance.
                         Set to 20 to match PriorityStrategy.
        """
        self.shared_layer = shared_layer
        self.private_layer = private_layer
        self.crit_weight = crit_weight

    def _role_rank_from_type(self, role_type: int) -> int:
        if role_type == _ROLE_TYPE_SYSTEM:
            return self.SHARED_RANK_SYSTEM
        elif role_type == _ROLE_TYPE_ROLE:
            return self.SHARED_RANK_ROLE
        else:
            return 0

    def get_priority(self, node: "TreeNode") -> Tuple:
        """Get eviction priority. Formula matches PriorityStrategy exactly."""
        if getattr(node, "lock_ref", 0) > 0:
            return (float("inf"), 0, 0, 0)

        crit_distance = max(1, getattr(node, "critical_path_distance", 1))
        if getattr(node, "reuse_mode", "") == "lossy":
            crit_distance = max(1, crit_distance - 1)
            if float(getattr(node, "reuse_confidence", 0.0) or 0.0) >= 0.75:
                crit_distance = max(1, crit_distance - 1)
        role_rank = self._role_rank_from_type(node.role_type)

        return (role_rank, -crit_distance, -node.priority, node.last_access_time)
