from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.approx_kv.config import ApproxKVFeatureConfig
from sglang.srt.mem_cache.approx_kv.manager import ApproxKVManager
from sglang.srt.mem_cache.approx_kv.radix_backend import (
    AllocatorCPUResidencyBackend,
)
from sglang.srt.mem_cache.approx_kv.request import (
    ApproxKVRequestMetadata,
    ApproxKVRequestOperation,
    ApproxKVRequestSegment,
)
from sglang.srt.mem_cache.approx_kv.runtime import (
    ApproxKVRegistrationError,
    allocate_recovery_slots,
    register_request_segments,
    restore_request_prefix,
)
from sglang.srt.mem_cache.approx_kv.types import ResidencyTier
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class FakeKVCache:
    def __init__(self):
        self.layer_num = 2
        shape = (64, 2, 8)
        self.k_buffer = [
            torch.arange(torch.tensor(shape).prod()).reshape(shape).float() + layer
            for layer in range(self.layer_num)
        ]
        self.v_buffer = [buffer + 1000 for buffer in self.k_buffer]

    def move_kv_cache(self, target, source):
        for layer in range(self.layer_num):
            self.k_buffer[layer][target] = self.k_buffer[layer][source]
            self.v_buffer[layer][target] = self.v_buffer[layer][source]

    def get_key_buffer(self, layer_id):
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id):
        return self.v_buffer[layer_id]

    def get_kv_size_bytes(self):
        key_bytes = sum(
            buffer.numel() * buffer.element_size() for buffer in self.k_buffer
        )
        value_bytes = sum(
            buffer.numel() * buffer.element_size() for buffer in self.v_buffer
        )
        return key_bytes, value_bytes


class FakeAllocator:
    device = "cpu"
    size_full = 64

    def __init__(self, kvcache):
        self.kvcache = kvcache
        self.next_index = 16
        self.freed = []

    def alloc(self, size):
        result = torch.arange(
            self.next_index,
            self.next_index + size,
            dtype=torch.int64,
        )
        self.next_index += size
        return result

    def free(self, indices):
        self.freed.extend(int(index) for index in indices)

    def get_kvcache(self):
        return self.kvcache

    def get_cpu_copy(self, indices, mamba_indices=None):
        del mamba_indices
        return (
            [buffer[indices].clone() for buffer in self.kvcache.k_buffer],
            [buffer[indices].clone() for buffer in self.kvcache.v_buffer],
        )

    def load_cpu_copy(self, payload, indices, mamba_indices=None):
        del mamba_indices
        keys, values = payload
        for layer in range(self.kvcache.layer_num):
            self.kvcache.k_buffer[layer][indices] = keys[layer]
            self.kvcache.v_buffer[layer][indices] = values[layer]


class PressureAllocator(FakeAllocator):
    """A capacity-limited allocator: `alloc()` only succeeds while
    `capacity` covers the requested size. `evict()` (driven by the real
    `evict_from_tree_cache`, via `FakeEvictingTree.evict`) reclaims
    `recovered_on_evict` slots of capacity -- 0 by default in the
    "no-leak" tests, honestly simulating an eviction that found no
    exact-Radix victims to reclaim. A passing "evicts before
    allocating" test therefore *proves* the evict-then-alloc ordering:
    with `recovered_on_evict > 0`, `alloc()` can only succeed after
    `evict()` actually raised `capacity` first."""

    def __init__(self, kvcache, *, capacity=0, recovered_on_evict=1_000_000):
        super().__init__(kvcache)
        self.capacity = capacity
        self.recovered_on_evict = recovered_on_evict

    def available_size(self):
        return self.capacity

    def alloc(self, size):
        if size > self.capacity:
            return None
        self.capacity -= size
        return super().alloc(size)


class FakeEvictingTree:
    """Tree-cache double exposing the real eviction protocol
    (`evict`/`is_chunk_cache`) so `allocate_recovery_slots` takes its
    evict-before-alloc branch, matching the shape production
    `RadixCache`/`ChunkCache` trees provide (plus the ordinary
    `req_to_token_pool`/`approx_kv` attributes other fixtures need)."""

    def __init__(
        self,
        allocator,
        req_to_token_pool=None,
        approx_kv_manager=None,
        *,
        is_chunk_cache=False,
    ):
        self.token_to_kv_pool_allocator = allocator
        self.req_to_token_pool = req_to_token_pool
        self.approx_kv = approx_kv_manager
        self.evict_params = []
        self._is_chunk_cache = is_chunk_cache

    def is_chunk_cache(self):
        return self._is_chunk_cache

    def evict(self, params):
        self.evict_params.append(params)
        self.token_to_kv_pool_allocator.capacity += (
            self.token_to_kv_pool_allocator.recovered_on_evict
        )


class FakeReqToTokenPool:
    def __init__(self):
        self.req_to_token = torch.full((2, 64), -1, dtype=torch.int64)


class FakeReq:
    def __init__(self, metadata, tokens, needs_host_load_back=False):
        self.approx_kv_metadata = metadata
        self.req_pool_idx = 0
        self.kv = SimpleNamespace(kv_allocated_len=len(tokens))
        self.full_untruncated_fill_ids = list(tokens)
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.rid = "req"
        self._needs_host_load_back = needs_host_load_back

    def effective_kv_committed_len(self):
        return len(self.full_untruncated_fill_ids)

    def needs_host_load_back(self):
        return self._needs_host_load_back


class TestApproxKVRuntime(unittest.TestCase):
    def setUp(self):
        self.kvcache = FakeKVCache()
        self.allocator = FakeAllocator(self.kvcache)
        self.req_pool = FakeReqToTokenPool()
        self.req_pool.req_to_token[0, :4] = torch.tensor([0, 1, 2, 3])
        config = ApproxKVFeatureConfig(
            core_enabled=True,
            host_residency_enabled=True,
        )
        self.manager = ApproxKVManager(config)
        self.manager.bind_residency_backend(
            AllocatorCPUResidencyBackend(self.allocator)
        )
        self.tree = SimpleNamespace(
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=self.req_pool,
            approx_kv=self.manager,
        )
        self.segment = ApproxKVRequestSegment(
            content_hash="artifact",
            target_start=0,
            length=3,
        )

    def metadata(self, operation):
        return ApproxKVRequestMetadata(
            operation=operation,
            segments=(self.segment,),
            model_fingerprint="model",
            cache_dtype="fp32",
        )

    def test_host_register_load_copy_and_last_token_forward(self):
        source_keys = [
            buffer[torch.tensor([0, 1, 2])].clone() for buffer in self.kvcache.k_buffer
        ]
        source_values = [
            buffer[torch.tensor([0, 1, 2])].clone() for buffer in self.kvcache.v_buffer
        ]
        source = FakeReq(
            self.metadata(ApproxKVRequestOperation.REGISTER),
            (10, 11, 12, 13),
        )
        self.assertEqual(register_request_segments(self.tree, source), 3)
        handle = self.manager.store.handles()[0]
        self.assertEqual(handle.residency, ResidencyTier.HOST)

        reuse = FakeReq(
            self.metadata(ApproxKVRequestOperation.REUSE),
            (10, 11, 12, 99),
        )
        self.assertTrue(restore_request_prefix(self.tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), 3)
        self.assertLess(len(reuse.prefix_indices), len(reuse.full_untruncated_fill_ids))
        for layer in range(self.kvcache.layer_num):
            torch.testing.assert_close(
                self.kvcache.k_buffer[layer][reuse.prefix_indices],
                source_keys[layer],
            )
            torch.testing.assert_close(
                self.kvcache.v_buffer[layer][reuse.prefix_indices],
                source_values[layer],
            )
        self.assertTrue(reuse.approx_kv_stats.mechanically_valid)

    def test_token_mismatch_uses_dense_fallback_without_allocating(self):
        source = FakeReq(
            self.metadata(ApproxKVRequestOperation.REGISTER),
            (10, 11, 12, 13),
        )
        register_request_segments(self.tree, source)
        next_index = self.allocator.next_index
        mismatch = FakeReq(
            self.metadata(ApproxKVRequestOperation.REUSE),
            (10, 77, 12, 99),
        )
        self.assertFalse(restore_request_prefix(self.tree, mismatch))
        self.assertEqual(self.allocator.next_index, next_index)
        self.assertEqual(len(mismatch.prefix_indices), 0)

    def test_registration_error_releases_request_before_reraising(self):
        req = SimpleNamespace(
            req_pool_idx=0,
            kv=SimpleNamespace(kv_allocated_len=4),
            mamba_pool_idx=None,
            rid="registration-error",
            effective_kv_committed_len=lambda: 4,
        )
        calls = []

        def cache_finished_req(request, **kwargs):
            calls.append(kwargs)
            request.req_pool_idx = None
            request.kv = None

        tree = SimpleNamespace(
            supports_mamba=lambda: False,
            cache_finished_req=cache_finished_req,
        )
        error = ApproxKVRegistrationError("registration failed")
        with patch(
            "sglang.srt.mem_cache.common.register_request_segments",
            side_effect=error,
        ):
            with self.assertLogs(
                "sglang.srt.mem_cache.common",
                level="ERROR",
            ):
                release_kv_cache(req, tree)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(req.req_pool_idx)
        self.assertIsNone(req.kv)
        self.assertEqual(
            req.approx_kv_registration_error,
            "registration failed",
        )

    def test_h2d_failure_falls_back_without_restored_slots(self):
        source = FakeReq(
            self.metadata(ApproxKVRequestOperation.REGISTER),
            (10, 11, 12, 13),
        )
        register_request_segments(self.tree, source)

        class FailingBackend:
            def load(self, handle, target_tier):
                del handle, target_tier
                raise RuntimeError("injected H2D failure")

        self.manager.bind_residency_backend(FailingBackend())
        next_index = self.allocator.next_index
        reuse = FakeReq(
            self.metadata(ApproxKVRequestOperation.REUSE),
            (10, 11, 12, 99),
        )
        self.assertFalse(restore_request_prefix(self.tree, reuse))
        self.assertEqual(self.allocator.next_index, next_index)
        self.assertEqual(len(reuse.prefix_indices), 0)

    def test_exact_host_hit_preempts_approximate_restore(self):
        source = FakeReq(
            self.metadata(ApproxKVRequestOperation.REGISTER),
            (10, 11, 12, 13),
        )
        register_request_segments(self.tree, source)
        next_index = self.allocator.next_index
        reuse = FakeReq(
            self.metadata(ApproxKVRequestOperation.REUSE),
            (10, 11, 12, 99),
            needs_host_load_back=True,
        )
        self.assertFalse(restore_request_prefix(self.tree, reuse))
        self.assertEqual(self.allocator.next_index, next_index)
        self.assertEqual(len(reuse.prefix_indices), 0)

    # ------------------------------------------------------------------
    # Recovery-slot allocation must evict exact-Radix victims first (via
    # `allocate_recovery_slots`), and must never leak an allocator slot
    # when the pool is still short even after eviction was attempted.
    # These use a dedicated `PressureAllocator`/`FakeEvictingTree` pair
    # instead of `setUp`'s plain fixtures: the shared `FakeAllocator`
    # always succeeds and the shared tree exposes no eviction protocol
    # at all, so neither can distinguish "evicted then allocated" from
    # "just allocated".
    # ------------------------------------------------------------------
    def _build_pressure_fixture(self, *, recovers_after_eviction=True):
        kvcache = FakeKVCache()
        allocator = PressureAllocator(
            kvcache,
            capacity=1_000_000,  # plenty while registering the source segment
            recovered_on_evict=1_000_000 if recovers_after_eviction else 0,
        )
        # `host_residency_enabled=False` keeps the source segment
        # DEVICE-resident, so the later reuse's `ensure_device` is a
        # no-op (no allocator call) and the *only* allocator.alloc call
        # left in the reuse path is the one this test targets: the
        # destination `allocate_recovery_slots` call.
        req_pool = FakeReqToTokenPool()
        req_pool.req_to_token[0, :4] = torch.tensor([0, 1, 2, 3])
        config = ApproxKVFeatureConfig(
            core_enabled=True,
            host_residency_enabled=False,
        )
        manager = ApproxKVManager(config)
        tree = FakeEvictingTree(allocator, req_pool, manager)
        segment = ApproxKVRequestSegment(
            content_hash="artifact",
            target_start=0,
            length=3,
        )

        def metadata(operation):
            return ApproxKVRequestMetadata(
                operation=operation,
                segments=(segment,),
                model_fingerprint="model",
                cache_dtype="fp32",
            )

        source = FakeReq(metadata(ApproxKVRequestOperation.REGISTER), (10, 11, 12, 13))
        self.assertEqual(register_request_segments(tree, source), 3)

        # Simulate capacity pressure arising *after* the source segment
        # was registered but *before* this reuse request's destination
        # allocation, so the destination allocation must evict
        # exact-Radix victims to succeed.
        allocator.capacity = 0
        return allocator, tree, metadata

    def test_restore_evicts_exact_radix_before_allocating_recovery_slots(self):
        allocator, tree, metadata = self._build_pressure_fixture()
        reuse = FakeReq(metadata(ApproxKVRequestOperation.REUSE), (10, 11, 12, 99))

        self.assertTrue(restore_request_prefix(tree, reuse))

        self.assertEqual(len(tree.evict_params), 1)
        self.assertEqual(tree.evict_params[0].num_tokens, 3)
        self.assertEqual(len(reuse.prefix_indices), 3)

    def test_restore_no_leak_when_allocation_still_fails_after_eviction(self):
        allocator, tree, metadata = self._build_pressure_fixture(
            recovers_after_eviction=False
        )
        reuse = FakeReq(metadata(ApproxKVRequestOperation.REUSE), (10, 11, 12, 99))

        self.assertFalse(restore_request_prefix(tree, reuse))

        # Eviction was genuinely attempted (not skipped)...
        self.assertEqual(len(tree.evict_params), 1)
        self.assertEqual(tree.evict_params[0].num_tokens, 3)
        # ...but the allocator honestly still returned None, so there is
        # nothing to leak: no slots were ever handed out, `free()` was
        # never even called (there was nothing to free), and the
        # request falls back to dense cleanly.
        self.assertEqual(allocator.freed, [])
        self.assertEqual(len(reuse.prefix_indices), 0)


class TestAllocateRecoverySlots(unittest.TestCase):
    """Direct unit tests of `allocate_recovery_slots` in isolation, with
    no manager/plugin wiring, proving its evict-before-alloc contract
    and its safe no-op behavior on trees/allocators that do not support
    the eviction protocol at all (e.g. every other fixture in this
    file)."""

    def setUp(self):
        self.kvcache = FakeKVCache()

    def test_evicts_exact_radix_before_allocating_when_capacity_is_short(self):
        allocator = PressureAllocator(self.kvcache)
        tree = FakeEvictingTree(allocator)

        slots = allocate_recovery_slots(tree, 8)

        self.assertEqual(len(slots), 8)
        self.assertEqual(len(tree.evict_params), 1)
        self.assertEqual(tree.evict_params[0].num_tokens, 8)

    def test_skips_eviction_when_capacity_already_sufficient(self):
        allocator = PressureAllocator(self.kvcache, capacity=1_000_000)
        tree = FakeEvictingTree(allocator)

        slots = allocate_recovery_slots(tree, 8)

        self.assertEqual(len(slots), 8)
        # Eviction must not be attempted when it is not needed.
        self.assertEqual(tree.evict_params, [])

    def test_skips_eviction_for_chunk_cache(self):
        allocator = PressureAllocator(self.kvcache)
        tree = FakeEvictingTree(allocator, is_chunk_cache=True)

        result = allocate_recovery_slots(tree, 8)

        # Chunk caches are never evicted from -- the allocator (still
        # genuinely short on capacity) honestly fails instead of a
        # silent fabricated success.
        self.assertIsNone(result)
        self.assertEqual(tree.evict_params, [])

    def test_skips_eviction_when_tree_lacks_eviction_protocol(self):
        # Matches every `SimpleNamespace`-based tree fixture used
        # elsewhere in this suite: no `evict`/`is_chunk_cache`. Must not
        # raise `AttributeError`; must fall straight through to a
        # direct `allocator.alloc` call, unchanged from before this
        # function existed.
        allocator = FakeAllocator(self.kvcache)
        tree = SimpleNamespace(token_to_kv_pool_allocator=allocator)

        slots = allocate_recovery_slots(tree, 5)

        self.assertEqual(len(slots), 5)

    def test_no_leak_when_allocation_still_fails_after_eviction(self):
        allocator = PressureAllocator(self.kvcache, recovered_on_evict=0)
        tree = FakeEvictingTree(allocator)

        result = allocate_recovery_slots(tree, 8)

        self.assertIsNone(result)
        self.assertEqual(len(tree.evict_params), 1)  # eviction was attempted
        self.assertEqual(allocator.freed, [])  # nothing allocated, nothing to leak


if __name__ == "__main__":
    unittest.main()
