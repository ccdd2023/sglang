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
    commit_provisional_recovery_slots,
    protect_request_prefix,
    register_request_segments,
    release_provisional_recovery_slots,
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

        def inc(node):
            self.locked.append(node)
            return None

        def dec(node, params=None):
            self.unlocked.append(node)

        self.tree = SimpleNamespace(inc_lock_ref=inc, dec_lock_ref=dec)
        self.node = object()
        self.req = SimpleNamespace(last_node=self.node)

    def test_swa_release_metadata_is_passed_back_to_dec_lock_ref(self):
        """SWA and Unified caches need the acquired-window metadata back.

        Releasing without it can walk past the window that was actually
        acquired and decrement an ancestor another request still holds.
        """
        released = []
        params = object()
        tree = SimpleNamespace(
            inc_lock_ref=lambda node: SimpleNamespace(to_dec_params=lambda: params),
            dec_lock_ref=lambda node, got=None: released.append((node, got)),
        )
        with protect_request_prefix(tree, self.req):
            pass
        self.assertEqual(released, [(self.node, params)])

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


class TestProvisionalRecoverySlots(unittest.TestCase):
    """Recovery slots must not leak when the scheduler rejects the request.

    Recovery attaches device slots inside init_next_round_input, which runs
    before add_one_req decides whether to admit the request. Ownership only
    transfers in prepare_for_extend, so an unadmitted request must give the
    slots back or they are lost when the next match_prefix rebuilds
    prefix_indices.
    """

    def setUp(self):
        self.freed = []
        self.tree = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(free=self.freed.append)
        )

    def test_uncommitted_slots_are_reclaimed_once(self):
        indices = torch.tensor([4, 5, 6], dtype=torch.int64)
        req = SimpleNamespace(
            approx_kv_provisional_indices=indices,
            approx_kv_restored_len=3,
        )

        self.assertEqual(release_provisional_recovery_slots(self.tree, req), 3)

        self.assertEqual(len(self.freed), 1)
        torch.testing.assert_close(self.freed[0], indices)
        self.assertIsNone(req.approx_kv_provisional_indices)
        self.assertEqual(req.approx_kv_restored_len, 0)

        # A second release must be a no-op, never a double free.
        self.assertEqual(release_provisional_recovery_slots(self.tree, req), 0)
        self.assertEqual(len(self.freed), 1)

    def test_request_without_recovery_is_untouched(self):
        req = SimpleNamespace()
        self.assertEqual(release_provisional_recovery_slots(self.tree, req), 0)
        self.assertEqual(self.freed, [])

    def test_manager_tracks_release_and_ownership_transfer(self):
        indices = torch.tensor([4, 5, 6], dtype=torch.int64)
        manager = ApproxKVManager(ApproxKVFeatureConfig(core_enabled=True))
        tree = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(free=self.freed.append),
            approx_kv=manager,
        )

        manager.add_provisional_tokens(3)
        rejected = SimpleNamespace(
            approx_kv_provisional_indices=indices,
            approx_kv_restored_len=3,
        )
        self.assertEqual(release_provisional_recovery_slots(tree, rejected), 3)
        self.assertEqual(manager.provisional_tokens, 0)

        manager.add_provisional_tokens(3)
        admitted = SimpleNamespace(
            approx_kv_provisional_indices=indices,
            approx_kv_restored_len=3,
        )
        self.assertEqual(commit_provisional_recovery_slots(tree, admitted), 3)
        self.assertEqual(manager.provisional_tokens, 0)
        self.assertEqual(len(self.freed), 1)

    def test_ownership_transfer_and_release_points_are_wired(self):
        root = Path(__file__).resolve().parents[4]
        batch = (root / "python/sglang/srt/managers/schedule_batch.py").read_text()
        common = (root / "python/sglang/srt/mem_cache/common.py").read_text()

        # released before match_prefix rebuilds the prefix
        self.assertLess(
            batch.index("release_provisional_recovery_slots(tree_cache, self)"),
            batch.index("match_result = tree_cache.match_prefix("),
        )
        # ownership transfers once the batch allocation happened
        self.assertLess(
            batch.index("out_cache_loc, req_pool_indices_tensor"),
            batch.index("commit_provisional_recovery_slots(self.tree_cache, req)"),
        )
        # and teardown reclaims anything still provisional
        self.assertIn("release_provisional_recovery_slots(tree_cache, req)", common)

    def test_reference_is_kept_if_the_free_fails(self):
        """A failed free must not drop the only handle on the slots."""

        class Boom(SimpleNamespace):
            def free(self, indices):
                raise RuntimeError("allocator refused")

        indices = torch.tensor([1, 2], dtype=torch.int64)
        req = SimpleNamespace(
            approx_kv_provisional_indices=indices, approx_kv_restored_len=2
        )
        tree = SimpleNamespace(token_to_kv_pool_allocator=Boom())
        with self.assertRaises(RuntimeError):
            release_provisional_recovery_slots(tree, req)
        self.assertIsNotNone(req.approx_kv_provisional_indices)

    def test_missing_allocator_keeps_the_reference(self):
        indices = torch.tensor([1, 2], dtype=torch.int64)
        req = SimpleNamespace(
            approx_kv_provisional_indices=indices, approx_kv_restored_len=2
        )
        self.assertEqual(release_provisional_recovery_slots(SimpleNamespace(), req), 0)
        self.assertIsNotNone(req.approx_kv_provisional_indices)

    def test_scheduler_releases_on_rejection_and_on_abort(self):
        """The two paths that previously leaked must both release.

        Waiting for the next scheduling round is not enough: a request that is
        rejected and then aborted never gets that round.
        """
        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/managers/scheduler.py"
        ).read_text()
        self.assertEqual(
            source.count("release_provisional_recovery_slots(self.tree_cache, req)"),
            2,
        )
        rejection = source.split("added = len(adder.can_run_list) > 0", 1)[1][:600]
        self.assertIn("release_provisional_recovery_slots", rejection)
        abort = source.split("req = self.waiting_queue.pop(i)", 1)[1][:400]
        self.assertIn("release_provisional_recovery_slots", abort)


if __name__ == "__main__":
    unittest.main()
