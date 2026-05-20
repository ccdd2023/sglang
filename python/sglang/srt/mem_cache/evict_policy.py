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
       - CRIT_WEIGHT=5: crit_dist range 1-5 gives 5-25 points separation within each tier.
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
    """Tiered Priority eviction to avoid Priority x Prefetch negative interaction.

    Design:
    - Layer 1 (Shared prefixes: Tier-0, Tier-1): LRU-based eviction
      - Use last_access_time as primary factor
      - Prefetch can target this layer
      - Natural retention of recently used shared prefixes

    - Layer 2 (Private prefixes: Tier-2): Priority-based eviction
      - Use DAG-aware priority for workflow-specific prefixes
      - Convergence protection for DAG nodes

    Why this fixes the negative interaction:
    - Original Priority x Prefetch: Both compete for the same cache space
      Prefetch preloads Tier-1 prefixes, Priority evicts them thinking "not urgent"
      Result: Prefetch wasted, Priority confused, cache fragmented

    - Tiered approach: Clear separation
      Shared prefixes (Tier-0/1) use LRU - natural temporal locality
      Private prefixes (Tier-2) use Priority - DAG-aware scheduling
      No interference between the two eviction domains

    Effective priority formula:
    - Tier-0/Tier-1: (-last_access_time, -node.key_len)
    - Tier-2: (priority - crit_distance × CRIT_WEIGHT, -last_access_time)

    Returns (priority_value, tiebreaker) where lower = evicted first.
    """

    CRIT_WEIGHT = 5

    def __init__(self, shared_layer: str = "lru", private_layer: str = "priority"):
        """
        Args:
            shared_layer: Eviction strategy for shared prefixes (Tier-0/1).
                        Options: "lru", "lfu", "lifespan"
            private_layer: Eviction strategy for private prefixes (Tier-2).
                        Options: "priority", "dags_aware"
        """
        self.shared_layer = shared_layer
        self.private_layer = private_layer

    def get_priority(self, node: "TreeNode") -> Tuple:
        """Get eviction priority for a node based on its tier."""
        # Skip locked nodes (Prefetch is actively using them)
        if getattr(node, "lock_ref", 0) > 0:
            return (float("inf"), 0)  # Never evict locked nodes

        # Determine which layer this node belongs to
        role_type = node.role_type

        if role_type in (_ROLE_TYPE_SYSTEM, _ROLE_TYPE_ROLE):
            # Layer 1: Shared prefixes (Tier-0, Tier-1) - Use LRU
            return self._get_shared_priority(node)
        else:
            # Layer 2: Private prefixes (Tier-2, Tier-3) - Use Priority
            return self._get_private_priority(node)

    def _get_shared_priority(self, node: "TreeNode") -> Tuple:
        """LRU-based priority for shared prefixes (Tier-0, Tier-1).

        Key insight: For shared prefixes, LRU naturally prefers recently used ones.
        This aligns with the goal: if a Tier-1 prefix was used recently by one
        workflow, it's likely to be used again by another workflow soon.

        Returns (lower = evicted first):
        - Primary: last_access_time (older = evicted first)
        - Tiebreaker: key_len (shorter = evicted first, prefer evicting larger nodes)
        """
        return (node.last_access_time, -len(node.key))

    def _get_private_priority(self, node: "TreeNode") -> Tuple:
        """Priority-based eviction for private prefixes (Tier-2, Tier-3).

        Uses the same additive formula as PriorityStrategy so results are
        consistent across both eviction strategies.

        Returns (lower = evicted first):
        - Primary: effective_priority (lower = closer to execution = evicted first)
        - Tiebreaker: last_access_time
        """
        crit_distance = max(1, getattr(node, "critical_path_distance", 1))
        return (-crit_distance, -node.priority, node.last_access_time)
