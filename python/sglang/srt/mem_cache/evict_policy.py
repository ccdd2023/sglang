from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, Tuple, Union

from sglang.srt.mem_cache.cache_policy import (
    CacheProtectionMetadata,
    belady_eviction_key,
    hierarchical_eviction_key,
    steps_eviction_key,
    value_density_eviction_key,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


def _event_ordinal(node: TreeNode) -> int:
    ordinal = getattr(node, "event_ordinal", None)
    return 0 if ordinal is None else int(ordinal)


class EvictionStrategy(ABC):
    def __init__(self) -> None:
        self.current_step = 0

    def observe(
        self,
        metadata: Iterable[CacheProtectionMetadata],
    ) -> None:
        steps = [
            item.current_step for item in metadata if item.current_step is not None
        ]
        if steps:
            self.current_step = max(self.current_step, max(steps))

    def reset(self) -> None:
        self.current_step = 0

    @abstractmethod
    def get_priority(self, node: TreeNode) -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return _event_ordinal(node)


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class WorkflowStepsStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> Tuple[int, int, float]:
        return steps_eviction_key(node.cache_protection, _event_ordinal(node))


class BeladyStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> Tuple[int, int, float]:
        return belady_eviction_key(node.cache_protection, _event_ordinal(node))


class RecoveryValueStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> Tuple[int, float, int, float]:
        return value_density_eviction_key(
            node.cache_protection,
            _event_ordinal(node),
            self.current_step,
        )


class HierarchicalObjectStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> Tuple[int, float, int, float]:
        return hierarchical_eviction_key(node.cache_protection, _event_ordinal(node))


class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        super().__init__()
        self.protected_threshold = protected_threshold

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
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
