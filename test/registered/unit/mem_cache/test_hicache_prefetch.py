"""Unit tests for HiCache workflow-aware async prefetch and step encoding.

Tests cover:
- ServerArgs flag parsing for enable_hicache_prefetch
- Positive absolute step encoding correctness (PriorityStrategy)
- prefetch_next_agent selection logic (candidate picking, skip rules, no-free-memory)
- Sequential cycle prefetch futility proof

Usage:
    python -m pytest test_hicache_prefetch.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.server_args import ServerArgs


class TestHiCachePrefetchFlags(unittest.TestCase):
    """ServerArgs flag parsing for HiCache prefetch."""

    def test_hicache_prefetch_flags_default(self):
        """--enable-hicache-prefetch defaults to False."""
        args = ServerArgs(model_path="dummy")
        self.assertFalse(args.enable_hicache_prefetch)


class TestAbsoluteStepEncoding(unittest.TestCase):
    """Tests for the positive absolute step encoding scheme.

    Priority values represent the absolute step at which a prefix will next
    be needed.  PriorityStrategy evicts nodes with the HIGHEST priority value
    first (furthest from next use).
    """

    def setUp(self):
        from sglang.srt.mem_cache.evict_policy import PriorityStrategy

        self.strategy = PriorityStrategy()

    def _make_node(self, priority, last_access_time=0.0):
        node = MagicMock()
        node.priority = priority
        node.last_access_time = last_access_time
        return node

    def test_absolute_step_encoding_eviction_order(self):
        """3 agents with positive absolute step numbers: A=3, B=4, C=5.
        Eviction should be C(5) first, then B(4), then A(3) last."""
        node_a = self._make_node(priority=3)
        node_b = self._make_node(priority=4)
        node_c = self._make_node(priority=5)

        nodes = [node_a, node_b, node_c]
        eviction_order = sorted(nodes, key=self.strategy.get_priority)

        # C (priority=5) evicted first, then B (4), then A (3)
        self.assertIs(eviction_order[0], node_c)
        self.assertIs(eviction_order[1], node_b)
        self.assertIs(eviction_order[2], node_a)

    def test_absolute_step_encoding_with_max(self):
        """After agent A re-executes, max(old=3, new=6) = 6.
        A should now be evicted before B(4) because A is further from next use."""
        # After max() propagation: A has priority 6, B has priority 4
        node_a = self._make_node(priority=max(3, 6))  # = 6
        node_b = self._make_node(priority=4)

        # A (6) should be evicted before B (4)
        self.assertLess(
            self.strategy.get_priority(node_a),
            self.strategy.get_priority(node_b),
        )

    def test_absolute_step_encoding_cycle_correctness(self):
        """Full 3-agent 2-round cycle verification.

        Round 0:
          Step 0: A executes, priority = 0 + 3 = 3  (next use at step 3)
          Step 1: B executes, priority = 1 + 3 = 4  (next use at step 4)
          Step 2: C executes, priority = 2 + 3 = 5  (next use at step 5)
          Eviction order: C(5) first, B(4) second, A(3) last  [correct]

        Round 1:
          Step 3: A re-executes, priority = max(3, 3+3) = max(3, 6) = 6
          Eviction order: A(6) first, C(5) second, B(4) last  [B protected, correct]
        """
        num_agents = 3

        # Round 0
        step_counter = 0
        prio_a = step_counter + num_agents  # 0 + 3 = 3
        step_counter += 1
        prio_b = step_counter + num_agents  # 1 + 3 = 4
        step_counter += 1
        prio_c = step_counter + num_agents  # 2 + 3 = 5
        step_counter += 1

        node_a = self._make_node(priority=prio_a)  # 3
        node_b = self._make_node(priority=prio_b)  # 4
        node_c = self._make_node(priority=prio_c)  # 5

        eviction_round0 = sorted(
            [node_a, node_b, node_c], key=self.strategy.get_priority
        )
        # C first, B second, A last
        self.assertIs(eviction_round0[0], node_c)
        self.assertIs(eviction_round0[1], node_b)
        self.assertIs(eviction_round0[2], node_a)

        # Round 1, Step 3: A re-executes
        new_prio_a = step_counter + num_agents  # 3 + 3 = 6
        # max() propagation in _insert_helper
        node_a.priority = max(prio_a, new_prio_a)  # max(3, 6) = 6
        step_counter += 1

        eviction_round1 = sorted(
            [node_a, node_b, node_c], key=self.strategy.get_priority
        )
        # A(6) first, C(5) second, B(4) last — B is protected (nearest to next use)
        self.assertIs(eviction_round1[0], node_a)
        self.assertIs(eviction_round1[1], node_c)
        self.assertIs(eviction_round1[2], node_b)


class _MockNode:
    """Lightweight mock for TreeNode used in prefetch selection tests."""

    _counter = 0

    def __init__(
        self,
        priority: int,
        evicted: bool = True,
        backuped: bool = True,
        lock_ref: int = 0,
        host_value=None,
        last_access_time: float = 0.0,
    ):
        self.priority = priority
        self.evicted = evicted
        self.backuped = backuped
        self.lock_ref = lock_ref
        self.host_value = host_value if host_value is not None else torch.tensor([1])
        self.value = None if evicted else torch.tensor([1])
        self.last_access_time = last_access_time
        self.parent = None  # root sentinel
        self.children = {}
        _MockNode._counter += 1
        self.id = _MockNode._counter


class TestPrefetchNextAgent(unittest.TestCase):
    """Tests for the prefetch_next_agent selection logic in HiRadixCache.

    These tests mock out just enough of the HiRadixCache to exercise the
    prefetch_next_agent method in isolation, verifying candidate selection,
    skip rules, and no-free-memory behaviour.
    """

    def setUp(self):
        from sglang.srt.mem_cache.evict_policy import PriorityStrategy

        self.strategy = PriorityStrategy()

    def _make_cache(self, nodes, ongoing_load_back=None, gpu_leaves=None):
        """Create a minimal mock that quacks like HiRadixCache for prefetch."""
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        cache = MagicMock(spec=HiRadixCache)
        cache.evictable_host_leaves = set(nodes)
        cache.evictable_leaves = set(gpu_leaves) if gpu_leaves is not None else set()
        cache.ongoing_load_back = ongoing_load_back or {}
        cache.eviction_strategy = self.strategy
        cache.evictable_size_ = 0
        # Bind the real methods to our mock
        cache.prefetch_next_agent = HiRadixCache.prefetch_next_agent.__get__(
            cache, HiRadixCache
        )
        cache._find_eviction_victim = HiRadixCache._find_eviction_victim.__get__(
            cache, HiRadixCache
        )
        cache.inc_lock_ref = MagicMock()
        return cache

    # ------------------------------------------------------------------
    # Test 1
    # ------------------------------------------------------------------
    def test_prefetch_picks_lowest_priority_node(self):
        """prefetch_next_agent should select the evicted node with lowest
        priority (= smallest node.priority = soonest needed in positive
        step encoding)."""
        # Three evicted nodes with different priorities.
        # node_a has the lowest node.priority → soonest needed → should be picked.
        node_a = _MockNode(priority=10, evicted=True, backuped=True)
        node_b = _MockNode(priority=20, evicted=True, backuped=True)
        node_c = _MockNode(priority=30, evicted=True, backuped=True)

        # Set up a root parent that is NOT evicted (chain terminator)
        root = _MockNode(priority=0, evicted=False, backuped=False)
        for n in (node_a, node_b, node_c):
            n.parent = root

        cache = self._make_cache([node_a, node_b, node_c])
        # cache_controller.load returns device indices on success
        cache.cache_controller = MagicMock()
        cache.cache_controller.load.return_value = torch.tensor([42])

        result = cache.prefetch_next_agent()

        self.assertTrue(result)
        # Should have picked node_a (priority=10, the soonest needed)
        self.assertIn(node_a.id, cache.ongoing_load_back)
        # Others should NOT be in ongoing_load_back
        self.assertNotIn(node_b.id, cache.ongoing_load_back)
        self.assertNotIn(node_c.id, cache.ongoing_load_back)

    # ------------------------------------------------------------------
    # Test 2
    # ------------------------------------------------------------------
    def test_prefetch_skips_already_loading(self):
        """Nodes already in ongoing_load_back should be skipped."""
        node_a = _MockNode(priority=10, evicted=True, backuped=True)
        node_b = _MockNode(priority=20, evicted=True, backuped=True)

        root = _MockNode(priority=0, evicted=False, backuped=False)
        node_a.parent = root
        node_b.parent = root

        # node_a is already being loaded back
        cache = self._make_cache(
            [node_a, node_b], ongoing_load_back={node_a.id: node_a}
        )
        cache.cache_controller = MagicMock()
        cache.cache_controller.load.return_value = torch.tensor([42])

        result = cache.prefetch_next_agent()

        self.assertTrue(result)
        # Should have fallen back to node_b since node_a was skipped
        self.assertIn(node_b.id, cache.ongoing_load_back)

    # ------------------------------------------------------------------
    # Test 3: no free memory AND no evictable GPU leaves → skip
    # ------------------------------------------------------------------
    def test_prefetch_skips_when_no_free_memory_no_victims(self):
        """When cache_controller.load() returns None and there are no
        evictable GPU leaves, prefetch should return False."""
        node_a = _MockNode(priority=10, evicted=True, backuped=True)

        root = _MockNode(priority=0, evicted=False, backuped=False)
        node_a.parent = root

        # No GPU leaves → _find_eviction_victim returns None
        cache = self._make_cache([node_a], gpu_leaves=[])
        cache.cache_controller = MagicMock()
        cache.cache_controller.load.return_value = None  # no free memory

        result = cache.prefetch_next_agent()

        self.assertFalse(result)
        self.assertEqual(len(cache.ongoing_load_back), 0)

    # ------------------------------------------------------------------
    # Test 3b: no free memory, victim is less urgent → evict and prefetch
    # ------------------------------------------------------------------
    def test_prefetch_evicts_when_upgrade(self):
        """When GPU victim has higher priority (less urgent) than prefetch
        target, eviction is an upgrade → prefetch should proceed."""
        # CPU node: priority=10 (soonest needed)
        node_cpu = _MockNode(priority=10, evicted=True, backuped=True)
        root = _MockNode(priority=0, evicted=False, backuped=False)
        node_cpu.parent = root

        # GPU victim: priority=50 (far from next use, less urgent)
        node_gpu = _MockNode(priority=50, evicted=False, backuped=False)

        cache = self._make_cache([node_cpu], gpu_leaves=[node_gpu])
        cache.cache_controller = MagicMock()
        # First call returns None (no free), second call (after evict) succeeds
        cache.cache_controller.load.side_effect = [None, torch.tensor([42])]

        result = cache.prefetch_next_agent()

        self.assertTrue(result)
        self.assertIn(node_cpu.id, cache.ongoing_load_back)
        # evict() should have been called
        cache.evict.assert_called_once()

    # ------------------------------------------------------------------
    # Test 3c: no free memory, victim is equally/more urgent → skip
    # ------------------------------------------------------------------
    def test_prefetch_skips_when_downgrade(self):
        """When GPU victim has lower or equal priority (more urgent) than
        prefetch target, eviction is a downgrade → prefetch should skip."""
        # CPU node: priority=50 (far from next use)
        node_cpu = _MockNode(priority=50, evicted=True, backuped=True)
        root = _MockNode(priority=0, evicted=False, backuped=False)
        node_cpu.parent = root

        # GPU victim: priority=10 (soonest needed, more urgent)
        node_gpu = _MockNode(priority=10, evicted=False, backuped=False)

        cache = self._make_cache([node_cpu], gpu_leaves=[node_gpu])
        cache.cache_controller = MagicMock()
        cache.cache_controller.load.return_value = None

        result = cache.prefetch_next_agent()

        self.assertFalse(result)
        self.assertEqual(len(cache.ongoing_load_back), 0)
        # evict() should NOT have been called
        cache.evict.assert_not_called()

    # ------------------------------------------------------------------
    # Test 4
    # ------------------------------------------------------------------
    def test_prefetch_returns_false_no_candidates(self):
        """When no evicted+backuped nodes exist, returns False."""
        # Case 1: empty evictable_host_leaves
        cache = self._make_cache([])
        cache.cache_controller = MagicMock()

        result = cache.prefetch_next_agent()
        self.assertFalse(result)

        # Case 2: all nodes have lock_ref > 0 (in use, not prefetchable)
        node_locked = _MockNode(priority=10, evicted=True, backuped=True, lock_ref=1)
        cache2 = self._make_cache([node_locked])
        cache2.cache_controller = MagicMock()

        result2 = cache2.prefetch_next_agent()
        self.assertFalse(result2)

    # ------------------------------------------------------------------
    # Test 5
    # ------------------------------------------------------------------
    def test_sequential_cycle_prefetch_never_beneficial(self):
        """In a sequential cycle with N agents and cache holding N-K,
        the K agents in CPU always have HIGHER priority than the N-K in GPU.
        Demonstrate that prefetch target priority > any GPU agent priority,
        meaning eviction for prefetch would always be a downgrade.

        Setup: 10 agents, priorities 30-39 (next-use step numbers).
        GPU holds agents 0-6 (priority 30-36), CPU holds agents 7-9 (37-39).

        Prefetch would target agent-7 (priority 37, the soonest-needed on CPU).
        To make room, we'd evict the HIGHEST-priority GPU node = agent-6 (36).

        But 37 > 36, so the prefetched agent is FURTHER from next use than
        the evicted one. This proves prefetch is never beneficial in a
        sequential cycle — which is exactly why we gate on free memory only.
        """
        num_agents = 10
        num_gpu = 7  # cache capacity
        num_cpu = num_agents - num_gpu  # 3 agents on CPU

        # Simulate step encoding: step_i + num_agents for agent i
        # Round 0, steps 0..9, each agent's next-use = step + 10
        # After round 0: agent-0 priority=10, agent-1=11, ..., agent-9=19
        # After eviction of agents 7-9: GPU has 0-6, CPU has 7-9
        # Actually, for this test the specific numbers don't matter,
        # what matters is the RELATIVE ordering.

        # Use priorities 30..39 for clarity
        gpu_nodes = []
        for i in range(num_gpu):
            n = _MockNode(priority=30 + i, evicted=False, backuped=False)
            gpu_nodes.append(n)

        cpu_nodes = []
        for i in range(num_cpu):
            n = _MockNode(priority=30 + num_gpu + i, evicted=True, backuped=True)
            cpu_nodes.append(n)

        # Prefetch target: the CPU node with LOWEST priority (soonest needed)
        prefetch_target = min(cpu_nodes, key=lambda n: n.priority)
        self.assertEqual(prefetch_target.priority, 37)  # agent-7

        # Eviction victim: the GPU node with HIGHEST priority (furthest from use)
        # In PriorityStrategy, get_priority returns (-priority, ...), so
        # eviction picks min(get_priority) = highest node.priority
        eviction_victim = min(gpu_nodes, key=lambda n: self.strategy.get_priority(n))
        self.assertEqual(eviction_victim.priority, 36)  # agent-6

        # The key assertion: prefetch target priority > eviction victim priority
        # This means the prefetched agent is needed LATER than the one we'd
        # have to evict — a strict downgrade. Prefetch is never beneficial.
        self.assertGreater(
            prefetch_target.priority,
            eviction_victim.priority,
            "Prefetch target (CPU) should have HIGHER priority (= further from "
            "next use) than any GPU eviction victim in a sequential cycle. "
            "This proves prefetch-with-eviction is never beneficial.",
        )

        # Corollary: in a sequential cycle, the conditional eviction gate
        # in prefetch_next_agent will ALWAYS reject eviction because the
        # victim is more urgent than the target (victim.priority <= target.priority).
        # Prefetch only proceeds when there is free GPU memory (zero-cost)
        # or when the victim is strictly less urgent (non-sequential workflows).


if __name__ == "__main__":
    unittest.main()
