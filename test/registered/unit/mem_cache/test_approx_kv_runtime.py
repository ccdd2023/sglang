from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.approx_kv.config import ApproxKVFeatureConfig
from sglang.srt.mem_cache.approx_kv.manager import ApproxKVManager
from sglang.srt.mem_cache.approx_kv.radix_backend import (
    AllocatorCPUResidencyBackend,
    RoPEConfig,
    resolve_model_rope_config,
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
    `evict()` actually raised `capacity` first. `free()` also returns
    its slots' capacity, like a real allocator -- required for any test
    that asserts `available_size()` is fully restored after a fallback
    that frees a provisional allocation (the exact "no slot leak"
    condition; a fixture whose `free()` did not return capacity could
    not tell a genuine leak from a merely-cosmetic one)."""

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

    def free(self, indices):
        self.capacity += len(indices)
        super().free(indices)


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


class TestMultiSegmentRestoreRopeFallback(unittest.TestCase):
    """Regression coverage for a real-GPU bug report (multi-chunk /
    multi-segment restores with a nonzero RoPE delta on a *later*
    segment): with no RoPE config bound (`ApproxKVManager.
    bind_rope_config` was dead code in production before this fix),
    the second of two segments in a single restore forces a
    `rope_config_unavailable` dense fallback, and the whole restore's
    provisional destination allocation must be freed back in full --
    not leaked -- or repeated occurrences under eviction pressure would
    eventually starve the allocator (`available_size() == 0` OOM, as
    observed on real hardware).

    Builds a genuinely two-segment restore where the *first* segment's
    source/target token positions align exactly (`rope_delta == 0`,
    always safe) and the *second* segment's do not (`rope_delta != 0`,
    the actual multi-chunk scenario reported): segment A is registered
    and reused at the same offset (0); segment B is registered at
    offset 10 in its *source* request but reused at offset 3 in the
    *target* request, an intentional cross-context relocation, exactly
    the case this module's RoPE binding exists to serve.
    """

    def _build_fixture(self, *, capacity=6):
        kvcache = FakeKVCache()
        allocator = PressureAllocator(
            kvcache,
            capacity=1_000_000,  # plenty while registering both source segments
            recovered_on_evict=1_000_000,
        )
        req_pool = FakeReqToTokenPool()
        # Segment A's source content lives at pool offset [0:3]; segment
        # B's lives at [10:13] -- distinct, non-adjacent source
        # locations, so the fixture cannot accidentally pass by both
        # segments aliasing the same physical slots.
        req_pool.req_to_token[0, :3] = torch.tensor([0, 1, 2])
        req_pool.req_to_token[0, 10:13] = torch.tensor([10, 11, 12])
        config = ApproxKVFeatureConfig(core_enabled=True, host_residency_enabled=False)
        manager = ApproxKVManager(config)
        tree = FakeEvictingTree(allocator, req_pool, manager)

        segment_a = ApproxKVRequestSegment(
            content_hash="segment-a", target_start=0, length=3
        )
        segment_b = ApproxKVRequestSegment(
            content_hash="segment-b", target_start=10, length=3
        )
        # Source tokens: [0:3] is segment A's content, [10:13] is
        # segment B's -- must be >= 13 tokens long for both segments'
        # `effective_kv_committed_len` check to pass at registration.
        source_tokens = (
            (10, 11, 12) + tuple(range(100, 107)) + (30, 31, 32)  # 13 tokens
        )
        self.assertEqual(source_tokens[10:13], (30, 31, 32))
        source = FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REGISTER,
                segments=(segment_a, segment_b),
                model_fingerprint="model",
                cache_dtype="fp32",
            ),
            source_tokens,
        )
        self.assertEqual(register_request_segments(tree, source), 6)

        # Reuse: the *same* two content hashes, but segment B is now
        # placed at target offset 3 (right after segment A's 3 tokens)
        # instead of its original registration-time offset of 10 --
        # this offset mismatch is exactly what produces a nonzero
        # `rope_delta` for segment B while segment A (registered and
        # reused at the same offset, 0) stays at delta 0.
        reuse_segment_a = ApproxKVRequestSegment(
            content_hash="segment-a", target_start=0, length=3
        )
        reuse_segment_b = ApproxKVRequestSegment(
            content_hash="segment-b", target_start=3, length=3
        )
        reuse_metadata = ApproxKVRequestMetadata(
            operation=ApproxKVRequestOperation.REUSE,
            segments=(reuse_segment_a, reuse_segment_b),
            model_fingerprint="model",
            cache_dtype="fp32",
        )
        # Target tokens: segment A's content (10,11,12) then segment
        # B's content (30,31,32) then one real forward token (99).
        reuse_tokens = (10, 11, 12, 30, 31, 32, 99)
        reuse = FakeReq(reuse_metadata, reuse_tokens)

        # Capacity pressure arises only for *this* reuse's own
        # destination allocation, exactly like `_build_pressure_fixture`
        # elsewhere in this file.
        allocator.capacity = capacity
        return allocator, tree, manager, reuse

    def test_second_segment_rope_delta_frees_full_allocation_with_no_rope_config(self):
        # Reproduces the real-GPU report: with no RoPE config ever
        # bound (`manager.rope_config` stays `None`, the historical
        # production default), segment B's nonzero `rope_delta` is
        # unresolvable and the whole restore must cleanly dense-
        # fallback -- freeing the *entire* 6-token provisional
        # allocation (both segments' slots), not just segment B's.
        allocator, tree, manager, reuse = self._build_fixture()
        available_before = allocator.available_size()

        self.assertFalse(restore_request_prefix(tree, reuse))

        self.assertEqual(len(reuse.prefix_indices), 0)
        # No slot leak: all capacity temporarily consumed by this
        # restore attempt (including eviction headroom) is back to
        # exactly its pre-call value -- the exact condition whose
        # violation would eventually manifest as the reported
        # `available_size() == 0` OOM under repeated occurrences.
        self.assertEqual(allocator.available_size(), available_before)

    def test_second_segment_rope_delta_records_honest_fallback_telemetry(self):
        allocator, tree, manager, reuse = self._build_fixture()
        metrics = _RecordingMetricsCollector()
        manager.metrics_collector = metrics

        self.assertFalse(restore_request_prefix(tree, reuse))

        self.assertIn(("rope_config_unavailable", 6), metrics.fallbacks)
        self.assertIn(("reuse", "dense_fallback"), metrics.requests)

    def test_binding_a_resolved_rope_config_lets_the_same_restore_succeed(self):
        # Same exact two-segment, nonzero-second-delta scenario, but
        # with a real RoPE config bound first via
        # `resolve_model_rope_config` (as `build_kv_cache` now does at
        # startup) -- this is the fix: segment B's nonzero delta is
        # actually relocated instead of forcing a fallback.
        allocator, tree, manager, reuse = self._build_fixture()
        fake_qwen_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
            hf_text_config=SimpleNamespace(
                rope_theta=1000000.0,
                rope_scaling=None,
            ),
            head_dim=8,  # matches FakeKVCache's last dimension exactly
        )
        manager.bind_rope_config(resolve_model_rope_config(fake_qwen_config))
        self.assertEqual(manager.rope_config.rotary_dim, 8)

        self.assertTrue(restore_request_prefix(tree, reuse))

        self.assertEqual(len(reuse.prefix_indices), 6)
        self.assertTrue(reuse.approx_kv_stats.mechanically_valid)


class _RecordingMetricsCollector:
    def __init__(self):
        self.requests: list[tuple[str, str]] = []
        self.fallbacks: list[tuple[str, int]] = []

    def increment_approx_kv_request(self, operation, outcome):
        self.requests.append((operation, outcome))

    def increment_approx_kv_fallback(self, reason, num_tokens):
        self.fallbacks.append((reason, num_tokens))


class TestResolveModelRopeConfig(unittest.TestCase):
    """Unit tests for `resolve_model_rope_config` in isolation: no
    manager/tree/allocator wiring, just the pure config -> `RoPEConfig`
    resolution this module's cross-context relocation depends on."""

    @staticmethod
    def _model_config(
        *,
        architectures,
        head_dim=128,
        rope_theta=1000000.0,
        rope_scaling=None,
        rope_parameters=None,
        dual_chunk_attention_config=None,
    ):
        if rope_parameters is not None:
            hf_text_config = SimpleNamespace(rope_parameters=rope_parameters)
        else:
            hf_text_config = SimpleNamespace(
                rope_theta=rope_theta, rope_scaling=rope_scaling
            )
        if dual_chunk_attention_config is not None:
            hf_text_config.dual_chunk_attention_config = dual_chunk_attention_config
        return SimpleNamespace(
            hf_config=SimpleNamespace(architectures=list(architectures)),
            hf_text_config=hf_text_config,
            head_dim=head_dim,
        )

    def test_qwen3_v4_style_config_resolves_real_rotary_dim(self):
        config = self._model_config(
            architectures=["Qwen3ForCausalLM"],
            head_dim=128,
            rope_theta=1000000.0,
            rope_scaling=None,
        )
        resolved = resolve_model_rope_config(config)
        self.assertEqual(
            resolved, RoPEConfig(rotary_dim=128, base=1000000.0, is_neox_style=True)
        )

    def test_qwen2_v4_style_config_resolves_real_rotary_dim(self):
        config = self._model_config(
            architectures=["Qwen2ForCausalLM"],
            head_dim=64,
            rope_theta=1000000.0,
            rope_scaling=None,
        )
        resolved = resolve_model_rope_config(config)
        self.assertEqual(
            resolved, RoPEConfig(rotary_dim=64, base=1000000.0, is_neox_style=True)
        )

    def test_v5_rope_parameters_default_type_resolves_real_rotary_dim(self):
        # transformers v5's unified `rope_parameters` may be present
        # even for a plain, unscaled model -- `rope_type: "default"`
        # with no `mrope_section`/`use_fope` must NOT be mistaken for
        # genuine scaling.
        config = self._model_config(
            architectures=["Qwen3ForCausalLM"],
            head_dim=128,
            rope_parameters={"rope_type": "default", "rope_theta": 5000000.0},
        )
        resolved = resolve_model_rope_config(config)
        self.assertEqual(
            resolved, RoPEConfig(rotary_dim=128, base=5000000.0, is_neox_style=True)
        )

    def test_yarn_scaling_conservatively_disables_relocation(self):
        config = self._model_config(
            architectures=["Qwen3ForCausalLM"],
            head_dim=128,
            rope_scaling={"type": "yarn", "factor": 4.0},
        )
        resolved = resolve_model_rope_config(config)
        self.assertEqual(resolved.rotary_dim, 0)
        self.assertEqual(resolved.base, 1000000.0)

    def test_llama3_scaling_conservatively_disables_relocation(self):
        config = self._model_config(
            architectures=["Qwen2ForCausalLM"],
            rope_scaling={"rope_type": "llama3", "factor": 8.0},
        )
        resolved = resolve_model_rope_config(config)
        self.assertEqual(resolved.rotary_dim, 0)

    def test_dual_chunk_attention_config_conservatively_disables_relocation(self):
        # Real Qwen2.5-1M-style long-context checkpoints set a non-empty
        # `dual_chunk_attention_config` on the same hf_text_config this
        # function reads rope_theta/rope_scaling from (see qwen2.py's
        # own `getattr(config, "dual_chunk_attention_config", None)`
        # plumbing into `get_rope`). Such models route through
        # `DualChunkRotaryEmbedding`'s chunk-aware, clamped-position
        # scheme, not this module's plain neox absolute-delta rotation
        # -- even though rope_scaling itself may look perfectly plain
        # ("default"/None), applying the simple rotation here would
        # silently compute a WRONG relocated key, not merely skip an
        # optimization.
        config = self._model_config(
            architectures=["Qwen2ForCausalLM"],
            head_dim=128,
            rope_theta=1000000.0,
            rope_scaling=None,
            dual_chunk_attention_config={
                "chunk_size": 8192,
                "local_size": 1024,
                "sparse_attention_config": None,
            },
        )
        resolved = resolve_model_rope_config(config)
        self.assertEqual(resolved.rotary_dim, 0)
        self.assertEqual(resolved.base, 1000000.0)

    def test_empty_dual_chunk_attention_config_does_not_disable_relocation(self):
        # An empty dict (falsy) must NOT be treated the same as a real,
        # populated dual_chunk_attention_config -- some configs may
        # carry an empty placeholder rather than omitting the attribute
        # entirely.
        config = self._model_config(
            architectures=["Qwen2ForCausalLM"],
            head_dim=64,
            rope_theta=1000000.0,
            rope_scaling=None,
            dual_chunk_attention_config={},
        )
        resolved = resolve_model_rope_config(config)
        self.assertEqual(
            resolved, RoPEConfig(rotary_dim=64, base=1000000.0, is_neox_style=True)
        )

    def test_default_type_with_mrope_section_conservatively_disables_relocation(self):
        # `rope_type == "default"` alone is not sufficient: `get_rope`
        # still special-cases `mrope_section` (M-RoPE, e.g. Qwen2-VL)
        # even under a "default" label.
        config = self._model_config(
            architectures=["Qwen3ForCausalLM"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 1000000.0,
                "mrope_section": [16, 24, 24],
            },
        )
        resolved = resolve_model_rope_config(config)
        self.assertEqual(resolved.rotary_dim, 0)

    def test_unrecognized_architecture_conservatively_disables_relocation(self):
        config = self._model_config(
            architectures=["GptOssForCausalLM"],
            head_dim=128,
            rope_theta=1000000.0,
            rope_scaling=None,
        )
        resolved = resolve_model_rope_config(config)
        self.assertEqual(resolved.rotary_dim, 0)

    def test_non_neox_style_architecture_is_not_in_the_verified_allowlist(self):
        # ChatGLM/GLM4/CommandR/GPT-J/... pass is_neox_style=False to
        # get_rope; this module's relocation formula has only been
        # verified for the neox-style Qwen2/Qwen3 family, so any of
        # these must resolve conservatively even with an otherwise
        # plain, unscaled rope_scaling.
        config = self._model_config(
            architectures=["ChatGLMForCausalLM"],
            head_dim=128,
            rope_theta=10000.0,
            rope_scaling=None,
        )
        resolved = resolve_model_rope_config(config)
        self.assertEqual(resolved.rotary_dim, 0)

    def test_bind_rope_config_updates_manager_rope_config(self):
        # Exercises the exact call sequence `build_kv_cache` performs:
        # `manager.bind_rope_config(resolve_model_rope_config(model_config))`.
        manager = ApproxKVManager(ApproxKVFeatureConfig(core_enabled=True))
        self.assertIsNone(manager.rope_config)
        config = self._model_config(architectures=["Qwen3ForCausalLM"], head_dim=128)

        manager.bind_rope_config(resolve_model_rope_config(config))

        self.assertEqual(
            manager.rope_config,
            RoPEConfig(rotary_dim=128, base=1000000.0, is_neox_style=True),
        )


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
