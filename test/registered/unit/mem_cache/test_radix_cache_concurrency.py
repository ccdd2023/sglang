"""
Regression tests for the `_delete_leaf` race that previously allowed
``inc_lock_ref`` / ``dec_lock_ref`` to mutate ``lock_ref``,
``evictable_size_``, ``protected_size_``, and ``evictable_leaves``
without holding any lock.

Bug (2026-06-27): in ``python/sglang/srt/mem_cache/radix_cache.py``,
``inc_lock_ref`` (line ~3492) and ``dec_lock_ref`` (line ~3578) mutated
shared state outside any lock. A concurrent eviction could observe a
half-applied lock_ref transition between the ``if cur.lock_ref == 0``
check and the ``cur.lock_ref += 1`` write, producing
``RuntimeError: dictionary changed size during iteration`` or
``AssertionError`` at the eviction site's ``len(x.parent.children) == 0``
assert.

Fix: wrap each method's body in
``with self.anchor_kv_store_lock:``. The lock is an ``RLock``, so the
existing eviction path that already holds it (via
``_decrement_anchor_refs``) re-enters harmlessly.

These tests drive concurrent ``inc_lock_ref`` / ``dec_lock_ref`` /
``evict`` traffic and assert the cache invariants hold end-to-end.

Run:
    python -m pytest test/registered/unit/mem_cache/test_radix_cache_concurrency.py -v
"""

from __future__ import annotations

import threading
import time
import unittest
import unittest.mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode, get_child_key
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=15, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=15, suite="stage-b-test-small-1-gpu-amd")


def _make_mock_allocator():
    """Allocator stub that records free() but doesn't actually free.

    Eviction needs ``token_to_kv_pool_allocator.free(value)`` and
    ``.alloc(n)`` to work; we provide no-op mocks with the methods
    called by ``RadixCache.evict``. We don't actually free anything
    (the cache evicts whole nodes at once); the test only asserts that
    the lock_ref accounting invariants hold.
    """
    mock = unittest.mock.Mock()
    mock.device = torch.device("cpu")
    mock.available_size.return_value = 10_000
    mock.alloc.return_value = torch.arange(0, 0, dtype=torch.int64)
    mock.get_kvcache.return_value = unittest.mock.Mock()
    return mock


def _make_simulated_cache():
    """Build a CPU RadixCache with realistic locks + a small tree."""
    from sglang.srt.mem_cache import radix_cache as rc

    TreeNode.counter = 0
    mock_allocator = _make_mock_allocator()
    cache = rc.RadixCache.create_simulated(
        disable=False, page_size=1, mock_allocator=mock_allocator
    )

    # Populate with a small tree: root -> [n1, n2, n3, n4, n5, n6].
    # Each leaf is 4 tokens so eviction has room to chew.
    nodes = []
    for i in range(6):
        tokens = list(range(100 + i * 4, 100 + (i + 1) * 4))
        node = rc.TreeNode(priority=0)
        node.key = rc.RadixKey(tokens)
        node.value = torch.arange(i * 4, (i + 1) * 4, dtype=torch.int64)
        node.parent = cache.root_node
        cache.root_node.children[cache.get_child_key_fn(node.key)] = node
        cache.evictable_size_ += len(tokens)
        cache.evictable_leaves.add(node)
        nodes.append(node)

    return cache, nodes


class TestRadixCacheConcurrency(unittest.TestCase):
    """Concurrent lock_ref + evict should not corrupt the cache."""

    def _assert_invariants(self, cache, label):
        total = cache.total_size()
        evictable = cache.evictable_size()
        protected = cache.protected_size()
        self.assertGreaterEqual(
            total,
            0,
            msg=f"[{label}] total_size went negative: {total}",
        )
        self.assertGreaterEqual(
            evictable,
            0,
            msg=f"[{label}] evictable_size went negative: {evictable}",
        )
        self.assertGreaterEqual(
            protected,
            0,
            msg=f"[{label}] protected_size went negative: {protected}",
        )
        # The accounting identity: evictable + protected should equal total.
        # Allow a 0-token tolerance for the root_node which holds no tokens.
        self.assertEqual(
            evictable + protected,
            total,
            msg=f"[{label}] accounting drift: "
            f"evictable={evictable} + protected={protected} != total={total}",
        )

    def test_concurrent_inc_dec_lock_ref_and_evict(self):
        """4 churn threads + 1 eviction thread, run for 3 seconds.

        Pre-fix: this surfaces either a RuntimeError on set iteration
        or an AssertionError at the eviction call site within seconds.
        Post-fix: invariants hold and no exception escapes.
        """
        cache, nodes = _make_simulated_cache()
        stop = threading.Event()
        errors: list[BaseException] = []
        err_lock = threading.Lock()

        def churn(idx: int):
            try:
                while not stop.is_set():
                    node = nodes[idx % len(nodes)]
                    cache.inc_lock_ref(node)
                    # Tiny hold so eviction actually has something to race with.
                    time.sleep(0)
                    cache.dec_lock_ref(node)
            except BaseException as e:  # noqa: BLE001
                with err_lock:
                    errors.append(e)

        def evict():
            try:
                while not stop.is_set():
                    cache.evict(EvictParams(num_tokens=4))
                    time.sleep(0)
            except BaseException as e:  # noqa: BLE001
                with err_lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=churn, args=(i,), name=f"churn-{i}")
            for i in range(4)
        ]
        threads.append(threading.Thread(target=evict, name="evict"))
        for t in threads:
            t.start()

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            time.sleep(0.25)
            self._assert_invariants(cache, "running")
            if errors:
                break

        stop.set()
        for t in threads:
            t.join(timeout=5.0)
            self.assertFalse(t.is_alive(), msg=f"thread {t.name} hung")

        self.assertFalse(
            errors,
            msg=f"threads raised: {[type(e).__name__ + ': ' + str(e) for e in errors[:3]]}",
        )
        self._assert_invariants(cache, "post")

    def test_lock_ref_methods_take_anchor_kv_store_lock(self):
        """The fix MUST hold anchor_kv_store_lock around the lock_ref mutations.

        We patch ``anchor_kv_store_lock`` to record every acquire/release
        pair and assert that ``inc_lock_ref`` / ``dec_lock_ref`` go through
        it. If a future refactor accidentally drops the ``with`` block, this
        catches it.
        """
        from sglang.srt.mem_cache import radix_cache as rc

        mock_allocator = _make_mock_allocator()
        cache = rc.RadixCache.create_simulated(
            disable=False, page_size=1, mock_allocator=mock_allocator
        )

        original_lock = cache.anchor_kv_store_lock
        recorded: list[str] = []
        held = threading.local()

        class _TracingLock:
            def __init__(self, real):
                self._real = real

            def __enter__(self):
                self._real.__enter__()
                recorded.append("acquire")
                held.count = getattr(held, "count", 0) + 1
                return self

            def __exit__(self, *exc):
                held.count = held.count - 1
                recorded.append("release")
                return self._real.__exit__(*exc)

        cache.anchor_kv_store_lock = _TracingLock(original_lock)

        node = rc.TreeNode(priority=0)
        node.key = rc.RadixKey([1, 2, 3, 4])
        node.value = torch.arange(4, dtype=torch.int64)
        node.parent = cache.root_node
        cache.root_node.children[cache.get_child_key_fn(node.key)] = node
        cache.evictable_size_ += 4

        recorded.clear()
        cache.inc_lock_ref(node)
        self.assertIn(
            "acquire",
            recorded,
            msg="inc_lock_ref did NOT acquire anchor_kv_store_lock",
        )
        self.assertGreaterEqual(
            recorded.count("acquire"),
            1,
            msg="inc_lock_ref did not acquire the lock at least once",
        )

        recorded.clear()
        cache.dec_lock_ref(node)
        self.assertIn(
            "acquire",
            recorded,
            msg="dec_lock_ref did NOT acquire anchor_kv_store_lock",
        )


if __name__ == "__main__":
    unittest.main()