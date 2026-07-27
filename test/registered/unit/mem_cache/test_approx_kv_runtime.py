from __future__ import annotations

import unittest
from pathlib import Path
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
    protect_request_prefix,
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
            cross_store_bytes_per_token=256,
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
        expected_host_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*source_keys, *source_values)
        )
        self.assertEqual(
            self.manager.store.host_owned_bytes,
            expected_host_bytes,
        )

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

    def test_missing_registration_dependency_keeps_request_dense(self):
        metadata = ApproxKVRequestMetadata(
            operation=ApproxKVRequestOperation.REGISTER,
            segments=(
                ApproxKVRequestSegment(
                    content_hash="dependent",
                    target_start=0,
                    length=3,
                    object_id="dependent",
                    dependencies=frozenset({"missing"}),
                ),
            ),
            model_fingerprint="model",
            cache_dtype="fp32",
        )
        source = FakeReq(metadata, (10, 11, 12, 13))
        self.assertEqual(register_request_segments(self.tree, source), 0)
        self.assertEqual(self.manager.store.record_count, 0)

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

    def test_prefix_gap_is_a_counted_dense_fallback_not_an_exact_hit(self):
        """A gap before the first segment must not be reported as ``exact``.

        When the exact prefix is shorter than the first approximate segment's
        ``target_start`` the request is fully dense-prefilled. Reporting it as
        ``exact`` hides a real dense fallback and leaves the explicit fallback
        counter at zero, which the evidence contract forbids.
        """
        gapped_segment = ApproxKVRequestSegment(
            content_hash="artifact",
            target_start=1,
            length=2,
        )
        metadata = ApproxKVRequestMetadata(
            operation=ApproxKVRequestOperation.REUSE,
            segments=(gapped_segment,),
            model_fingerprint="model",
            cache_dtype="fp32",
        )
        recorded_requests = []
        recorded_fallbacks = []
        self.manager.metrics_collector = SimpleNamespace(
            increment_approx_kv_request=(
                lambda operation, outcome: recorded_requests.append(
                    (operation, outcome)
                )
            ),
            increment_approx_kv_fallback=(
                lambda reason, num_tokens: recorded_fallbacks.append(
                    (reason, num_tokens)
                )
            ),
        )
        reuse = FakeReq(metadata, (10, 11, 12, 99))
        next_index = self.allocator.next_index

        self.assertFalse(restore_request_prefix(self.tree, reuse))

        self.assertEqual(self.allocator.next_index, next_index)
        self.assertEqual(len(reuse.prefix_indices), 0)
        self.assertIn(("reuse", "dense_fallback"), recorded_requests)
        self.assertNotIn(("reuse", "exact"), recorded_requests)
        self.assertEqual(recorded_fallbacks, [("prefix_gap", 3)])

    def test_fully_covered_prefix_is_still_reported_as_exact(self):
        covered_segment = ApproxKVRequestSegment(
            content_hash="artifact",
            target_start=0,
            length=2,
        )
        metadata = ApproxKVRequestMetadata(
            operation=ApproxKVRequestOperation.REUSE,
            segments=(covered_segment,),
            model_fingerprint="model",
            cache_dtype="fp32",
        )
        recorded_requests = []
        recorded_fallbacks = []
        self.manager.metrics_collector = SimpleNamespace(
            increment_approx_kv_request=(
                lambda operation, outcome: recorded_requests.append(
                    (operation, outcome)
                )
            ),
            increment_approx_kv_fallback=(
                lambda reason, num_tokens: recorded_fallbacks.append(
                    (reason, num_tokens)
                )
            ),
        )
        reuse = FakeReq(metadata, (10, 11, 12, 99))
        reuse.prefix_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        self.assertFalse(restore_request_prefix(self.tree, reuse))

        self.assertIn(("reuse", "exact"), recorded_requests)
        self.assertNotIn(("reuse", "dense_fallback"), recorded_requests)
        self.assertEqual(recorded_fallbacks, [])


class TestRecoveryPrefixProtection(unittest.TestCase):
    """The request's own matched prefix must be locked during recovery.

    Req.init_next_round_input runs recovery before schedule_policy takes the
    prefix lock, so without an explicit guard the exact nodes backing
    req.prefix_indices are unlocked, are legal cross-store eviction victims,
    and can be freed and handed straight back as the recovery destination.
    """

    def setUp(self):
        self.locked = []
        self.unlocked = []
        self.tree = SimpleNamespace(
            inc_lock_ref=self.locked.append,
            dec_lock_ref=self.unlocked.append,
        )
        self.node = object()
        self.req = SimpleNamespace(last_node=self.node)

    def test_prefix_is_locked_for_the_whole_recovery_window(self):
        with protect_request_prefix(self.tree, self.req):
            self.assertEqual(self.locked, [self.node])
            self.assertEqual(self.unlocked, [])
        self.assertEqual(self.unlocked, [self.node])

    def test_prefix_lock_is_released_even_if_recovery_raises(self):
        with self.assertRaises(RuntimeError):
            with protect_request_prefix(self.tree, self.req):
                raise RuntimeError("recovery blew up")
        self.assertEqual(self.locked, [self.node])
        self.assertEqual(self.unlocked, [self.node])

    def test_missing_node_or_lock_api_is_a_no_op(self):
        with protect_request_prefix(self.tree, SimpleNamespace(last_node=None)):
            pass
        self.assertEqual(self.locked, [])
        with protect_request_prefix(SimpleNamespace(), self.req):
            pass
        self.assertEqual(self.locked, [])

    def test_recovery_call_site_is_wrapped_in_the_guard(self):
        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/managers/schedule_batch.py"
        ).read_text()
        self.assertIn("with protect_request_prefix(tree_cache, self):", source)
        guard_at = source.index("with protect_request_prefix(tree_cache, self):")
        for call in (
            "restore_request_prefix(tree_cache, self)",
            "restore_request_prefix_epic(tree_cache, self)",
        ):
            self.assertGreater(source.index(call), guard_at)


if __name__ == "__main__":
    unittest.main()
