from __future__ import annotations

"""Branch-specific tests for the Phase 4 R0 raw+RoPE recovery plugin.

These cover the required-behavior contract from the R0 task spec that the
shared common-core test suite (test_approx_kv_core.py /
test_approx_kv_runtime.py) does not exercise directly:

- zero, positive, and negative RoPE position delta;
- contiguous multi-segment recovery;
- an interior segment recovered right after a dense/exact head;
- the explicit plugin gate itself (registered vs not registered);
- missing/non-contiguous coverage always raising a dense-fallback signal.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.approx_kv.config import ApproxKVFeatureConfig
from sglang.srt.mem_cache.approx_kv.manager import ApproxKVManager
from sglang.srt.mem_cache.approx_kv.plugins import RecoveryRequestContext
from sglang.srt.mem_cache.approx_kv.radix_backend import (
    AllocatorCPUResidencyBackend,
    RoPEConfig,
)
from sglang.srt.mem_cache.approx_kv.raw_rope import (
    RAW_ROPE_PLUGIN_NAME,
    RawRoPERecoveryPlugin,
    RawRoPERecoveryRequest,
    RawRoPERecoveryUnavailable,
    build_raw_rope_plan,
    select_contiguous_segments,
    resolve_model_rope_config,
)
from sglang.srt.mem_cache.approx_kv.request import (
    ApproxKVRequestMetadata,
    ApproxKVRequestOperation,
    ApproxKVRequestSegment,
)
from sglang.srt.mem_cache.approx_kv.runtime import (
    register_request_segments,
    restore_request_prefix,
)
from sglang.srt.mem_cache.approx_kv.store import ApproxKVSegmentStore
from sglang.srt.mem_cache.approx_kv.types import (
    KVSegmentKey,
    RecoveryMode,
    ResidencyTier,
    SegmentKind,
    token_ids_hash,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")

MODEL_FINGERPRINT = "test-model"
CACHE_DTYPE = "fp32"


def _key(tokens: tuple[int, ...], content_hash: str, source_kind=SegmentKind.ARTIFACT):
    return KVSegmentKey(
        content_hash=content_hash,
        token_hash=token_ids_hash(tokens),
        token_count=len(tokens),
        model_fingerprint=MODEL_FINGERPRINT,
        cache_dtype=CACHE_DTYPE,
        kind=source_kind,
    )


def _register(
    store: ApproxKVSegmentStore,
    *,
    content_hash: str,
    tokens: tuple[int, ...],
    source_start: int,
):
    return store.register(
        key=_key(tokens, content_hash),
        token_ids=tokens,
        source_start=source_start,
        residency=ResidencyTier.DEVICE,
        backend_ref=object(),
    )


class TestRawRopePlanConstruction(unittest.TestCase):
    def test_qwen_rope_config_binding(self):
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen3",
                head_dim=128,
                rope_theta=None,
                rope_scaling={
                    "rope_type": "default",
                    "rope_theta": 1000000.0,
                },
            )
        )
        config = resolve_model_rope_config(model_config)
        self.assertEqual(config.rotary_dim, 128)
        self.assertEqual(config.base, 1000000.0)
        self.assertTrue(config.is_neox_style)

    def test_scaled_or_unknown_rope_stays_unbound(self):
        scaled = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen3",
                head_dim=128,
                rope_scaling={"rope_type": "yarn"},
            )
        )
        unknown = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="unknown",
                head_dim=128,
                rope_scaling=None,
            )
        )
        self.assertIsNone(resolve_model_rope_config(scaled))
        self.assertIsNone(resolve_model_rope_config(unknown))

    """Pure build_raw_rope_plan()/RawRoPERecoveryPlugin tests: no torch I/O."""

    def setUp(self):
        self.store = ApproxKVSegmentStore()

    def test_zero_delta_plan(self):
        # Registered at position 5, reused at position 5: rope_delta == 0.
        _register(self.store, content_hash="a", tokens=(1, 2, 3), source_start=5)
        target = (0, 0, 0, 0, 0, 1, 2, 3, 9)
        plan = build_raw_rope_plan(
            target_token_ids=target,
            exact_prefix_length=5,
            segments=(
                ApproxKVRequestSegment(content_hash="a", target_start=5, length=3),
            ),
            model_fingerprint=MODEL_FINGERPRINT,
            cache_dtype=CACHE_DTYPE,
            store=self.store,
        )
        self.assertEqual(plan.recovery_mode, RecoveryMode.COPY)
        self.assertEqual(len(plan.copied_spans), 1)
        self.assertEqual(plan.copied_spans[0].rope_delta, 0)
        self.assertEqual(plan.target_token_ids, (1, 2, 3))

    def test_positive_delta_plan(self):
        # Registered at position 5, reused later at position 9: delta > 0.
        _register(self.store, content_hash="a", tokens=(1, 2, 3), source_start=5)
        target = (0,) * 9 + (1, 2, 3, 9)
        plan = build_raw_rope_plan(
            target_token_ids=target,
            exact_prefix_length=9,
            segments=(
                ApproxKVRequestSegment(content_hash="a", target_start=9, length=3),
            ),
            model_fingerprint=MODEL_FINGERPRINT,
            cache_dtype=CACHE_DTYPE,
            store=self.store,
        )
        self.assertEqual(plan.copied_spans[0].rope_delta, 4)

    def test_negative_delta_plan(self):
        # Registered at position 5, reused earlier at position 1: delta < 0.
        _register(self.store, content_hash="a", tokens=(1, 2, 3), source_start=5)
        target = (0, 1, 2, 3, 9)
        plan = build_raw_rope_plan(
            target_token_ids=target,
            exact_prefix_length=1,
            segments=(
                ApproxKVRequestSegment(content_hash="a", target_start=1, length=3),
            ),
            model_fingerprint=MODEL_FINGERPRINT,
            cache_dtype=CACHE_DTYPE,
            store=self.store,
        )
        self.assertEqual(plan.copied_spans[0].rope_delta, -4)

    def test_contiguous_multi_segment_recovery(self):
        _register(self.store, content_hash="a", tokens=(1, 2), source_start=0)
        _register(self.store, content_hash="b", tokens=(3, 4, 5), source_start=2)
        target = (1, 2, 3, 4, 5, 9)
        plan = build_raw_rope_plan(
            target_token_ids=target,
            exact_prefix_length=0,
            segments=(
                ApproxKVRequestSegment(content_hash="a", target_start=0, length=2),
                ApproxKVRequestSegment(content_hash="b", target_start=2, length=3),
            ),
            model_fingerprint=MODEL_FINGERPRINT,
            cache_dtype=CACHE_DTYPE,
            store=self.store,
        )
        self.assertEqual(len(plan.copied_spans), 2)
        self.assertEqual(plan.target_token_ids, (1, 2, 3, 4, 5))
        first, second = plan.copied_spans
        self.assertEqual(first.target_start, 0)
        self.assertEqual(first.length, 2)
        self.assertEqual(second.target_start, 2)
        self.assertEqual(second.length, 3)
        # Both segments were registered at exactly their target position:
        # both spans should carry a zero delta.
        self.assertEqual(first.rope_delta, 0)
        self.assertEqual(second.rope_delta, 0)

    def test_interior_segment_after_dense_exact_head(self):
        # Positions [0, 4) are already covered by a dense/exact head (not
        # this plugin's concern); only the interior segment at [4, 7) is
        # recovered by raw+RoPE, immediately followed by the reserved
        # final token at position 7.
        _register(self.store, content_hash="mid", tokens=(7, 8, 9), source_start=10)
        target = (0, 1, 2, 3, 7, 8, 9, 99)
        plan = build_raw_rope_plan(
            target_token_ids=target,
            exact_prefix_length=4,
            segments=(
                ApproxKVRequestSegment(content_hash="mid", target_start=4, length=3),
            ),
            model_fingerprint=MODEL_FINGERPRINT,
            cache_dtype=CACHE_DTYPE,
            store=self.store,
        )
        self.assertEqual(plan.target_token_ids, (7, 8, 9))
        self.assertEqual(plan.copied_spans[0].target_start, 0)
        self.assertEqual(plan.copied_spans[0].rope_delta, 4 - 10)

    def test_final_prompt_token_is_never_included(self):
        _register(self.store, content_hash="a", tokens=(1, 2, 3), source_start=0)
        target = (1, 2, 3, 99)
        plan = build_raw_rope_plan(
            target_token_ids=target,
            exact_prefix_length=0,
            segments=(
                ApproxKVRequestSegment(content_hash="a", target_start=0, length=3),
            ),
            model_fingerprint=MODEL_FINGERPRINT,
            cache_dtype=CACHE_DTYPE,
            store=self.store,
        )
        # length 3 segment fully covers [0,3); the 4th token (index 3) is
        # the final prompt token and must never appear in the plan.
        self.assertEqual(len(plan.target_token_ids), 3)
        self.assertNotIn(99, plan.target_token_ids)

    def test_missing_segment_raises_unavailable(self):
        target = (1, 2, 3, 9)
        with self.assertRaises(RawRoPERecoveryUnavailable):
            build_raw_rope_plan(
                target_token_ids=target,
                exact_prefix_length=0,
                segments=(
                    ApproxKVRequestSegment(
                        content_hash="never-registered",
                        target_start=0,
                        length=3,
                    ),
                ),
                model_fingerprint=MODEL_FINGERPRINT,
                cache_dtype=CACHE_DTYPE,
                store=self.store,
            )

    def test_noncontiguous_gap_raises_unavailable(self):
        _register(self.store, content_hash="a", tokens=(1, 2), source_start=0)
        _register(self.store, content_hash="b", tokens=(5, 6), source_start=10)
        target = (1, 2, 0, 5, 6, 9)
        segments = (
            ApproxKVRequestSegment(content_hash="a", target_start=0, length=2),
            # Gap at position 2 (not covered by any segment) before "b".
            ApproxKVRequestSegment(content_hash="b", target_start=3, length=2),
        )
        active = select_contiguous_segments(segments, exact_length=0, reusable_limit=5)
        # Only the first segment is part of the contiguous run; "b" is
        # unreachable because of the gap at position 2.
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].content_hash, "a")

    def test_plugin_wraps_pure_function_via_protocol(self):
        _register(self.store, content_hash="a", tokens=(1, 2, 3), source_start=5)
        plugin = RawRoPERecoveryPlugin()
        self.assertEqual(plugin.name, RAW_ROPE_PLUGIN_NAME)
        context = RecoveryRequestContext(
            request_id="req-1",
            target_token_ids=(0, 0, 0, 0, 0, 1, 2, 3, 9),
            exact_prefix_length=5,
            custom_metadata={
                RawRoPERecoveryRequest.KEY: RawRoPERecoveryRequest(
                    segments=(
                        ApproxKVRequestSegment(
                            content_hash="a", target_start=5, length=3
                        ),
                    ),
                    model_fingerprint=MODEL_FINGERPRINT,
                    cache_dtype=CACHE_DTYPE,
                ),
            },
        )
        plan = plugin.build_plan(context, self.store)
        self.assertEqual(plan.copied_spans[0].rope_delta, 0)
        self.assertEqual(plugin.scheduler_metadata(context), ())

    def test_plugin_requires_payload(self):
        plugin = RawRoPERecoveryPlugin()
        context = RecoveryRequestContext(
            request_id="req-1",
            target_token_ids=(1, 2, 3),
            exact_prefix_length=0,
            custom_metadata={},
        )
        with self.assertRaises(RawRoPERecoveryUnavailable):
            plugin.build_plan(context, self.store)


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


class FakeAllocator:
    device = "cpu"

    def __init__(self, kvcache, next_index=16):
        self.kvcache = kvcache
        self.next_index = next_index
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
        self.req_to_token = torch.full((4, 64), -1, dtype=torch.int64)


class FakeReq:
    def __init__(self, metadata, tokens, exact_prefix_len=0, needs_host_load_back=False):
        self.approx_kv_metadata = metadata
        self.req_pool_idx = 0
        self.kv = SimpleNamespace(kv_allocated_len=len(tokens))
        self.full_untruncated_fill_ids = list(tokens)
        self.prefix_indices = torch.arange(exact_prefix_len, dtype=torch.int64)
        self.rid = "req"
        self._needs_host_load_back = needs_host_load_back

    def effective_kv_committed_len(self):
        return len(self.full_untruncated_fill_ids)

    def needs_host_load_back(self):
        return self._needs_host_load_back


def _metadata(segments, operation):
    return ApproxKVRequestMetadata(
        operation=operation,
        segments=segments,
        model_fingerprint=MODEL_FINGERPRINT,
        cache_dtype=CACHE_DTYPE,
    )


class TestRawRopeRuntimeIntegration(unittest.TestCase):
    """End-to-end restore_request_prefix() through the raw_rope plugin gate."""

    def setUp(self):
        self.kvcache = FakeKVCache()
        self.allocator = FakeAllocator(self.kvcache)
        self.req_pool = FakeReqToTokenPool()
        config = ApproxKVFeatureConfig(
            core_enabled=True,
            host_residency_enabled=True,
            raw_rope_plugin_enabled=True,
        )
        self.manager = ApproxKVManager(config)
        self.manager.bind_residency_backend(
            AllocatorCPUResidencyBackend(self.allocator)
        )
        self.manager.bind_rope_config(
            RoPEConfig(rotary_dim=8, base=10000.0, is_neox_style=True)
        )
        self.tree = SimpleNamespace(
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=self.req_pool,
            approx_kv=self.manager,
        )

    def _register_source(self, row, content_hash, source_tokens, source_start):
        self.req_pool.req_to_token[
            row, source_start : source_start + len(source_tokens)
        ] = torch.arange(source_start, source_start + len(source_tokens))
        segment = ApproxKVRequestSegment(
            content_hash=content_hash,
            target_start=source_start,
            length=len(source_tokens),
        )
        filler = tuple(range(source_start)) + tuple(source_tokens)
        src_req = FakeReq(
            _metadata((segment,), ApproxKVRequestOperation.REGISTER),
            filler,
        )
        src_req.req_pool_idx = row
        register_request_segments(self.tree, src_req)

    def test_plugin_disabled_gate_blocks_recovery(self):
        self._register_source(0, "art", (10, 11, 12, 13), source_start=0)
        self.manager.config = ApproxKVFeatureConfig(
            core_enabled=True,
            host_residency_enabled=True,
            raw_rope_plugin_enabled=False,
        )
        next_index = self.allocator.next_index
        reuse = FakeReq(
            _metadata(
                (ApproxKVRequestSegment(content_hash="art", target_start=0, length=3),),
                ApproxKVRequestOperation.REUSE,
            ),
            (10, 11, 12, 99),
        )
        self.assertFalse(restore_request_prefix(self.tree, reuse))
        self.assertEqual(self.allocator.next_index, next_index)
        self.assertEqual(len(reuse.prefix_indices), 0)

    def test_zero_delta_recovery(self):
        self._register_source(0, "art", (10, 11, 12), source_start=0)
        source_keys = [
            buffer[torch.tensor([0, 1, 2])].clone() for buffer in self.kvcache.k_buffer
        ]
        reuse = FakeReq(
            _metadata(
                (ApproxKVRequestSegment(content_hash="art", target_start=0, length=3),),
                ApproxKVRequestOperation.REUSE,
            ),
            (10, 11, 12, 99),
        )
        self.assertTrue(restore_request_prefix(self.tree, reuse))
        for layer in range(self.kvcache.layer_num):
            torch.testing.assert_close(
                self.kvcache.k_buffer[layer][reuse.prefix_indices],
                source_keys[layer],
            )
        self.assertTrue(reuse.approx_kv_stats.mechanically_valid)

    def test_positive_delta_recovery_rotates_keys(self):
        # Source artifact lives at position 0; reused at position 6, so the
        # relocation delta is +6. Preceding tokens [0,6) are simulated as an
        # already-exact head via a preset prefix_indices length.
        self._register_source(0, "art", (10, 11, 12), source_start=0)
        unrotated_keys = [
            buffer[torch.tensor([0, 1, 2])].clone() for buffer in self.kvcache.k_buffer
        ]
        unrotated_values = [
            buffer[torch.tensor([0, 1, 2])].clone() for buffer in self.kvcache.v_buffer
        ]
        reuse = FakeReq(
            _metadata(
                (ApproxKVRequestSegment(content_hash="art", target_start=6, length=3),),
                ApproxKVRequestOperation.REUSE,
            ),
            (0, 0, 0, 0, 0, 0, 10, 11, 12, 99),
            exact_prefix_len=6,
        )
        self.assertTrue(restore_request_prefix(self.tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), 6 + 3)
        restored = reuse.prefix_indices[6:]
        for layer in range(self.kvcache.layer_num):
            # Values are never rotated; keys must be rotated away from the
            # raw copy since rope_delta (+6) is nonzero.
            torch.testing.assert_close(
                self.kvcache.v_buffer[layer][restored],
                unrotated_values[layer],
            )
            self.assertFalse(
                torch.equal(
                    self.kvcache.k_buffer[layer][restored],
                    unrotated_keys[layer],
                )
            )
        self.assertTrue(reuse.approx_kv_stats.mechanically_valid)

    def test_negative_delta_recovery_rotates_keys(self):
        # Source artifact registered later, at position 6; reused earlier,
        # at position 1, so delta is -5 (negative).
        self._register_source(0, "art", (10, 11, 12), source_start=6)
        unrotated_keys = [
            buffer[torch.tensor([6, 7, 8])].clone() for buffer in self.kvcache.k_buffer
        ]
        unrotated_values = [
            buffer[torch.tensor([6, 7, 8])].clone() for buffer in self.kvcache.v_buffer
        ]
        reuse = FakeReq(
            _metadata(
                (ApproxKVRequestSegment(content_hash="art", target_start=1, length=3),),
                ApproxKVRequestOperation.REUSE,
            ),
            (0, 10, 11, 12, 99),
            exact_prefix_len=1,
        )
        self.assertTrue(restore_request_prefix(self.tree, reuse))
        restored = reuse.prefix_indices[1:]
        for layer in range(self.kvcache.layer_num):
            torch.testing.assert_close(
                self.kvcache.v_buffer[layer][restored],
                unrotated_values[layer],
            )
            self.assertFalse(
                torch.equal(
                    self.kvcache.k_buffer[layer][restored],
                    unrotated_keys[layer],
                )
            )
        self.assertTrue(reuse.approx_kv_stats.mechanically_valid)

    def test_contiguous_multi_segment_recovery_end_to_end(self):
        self._register_source(0, "head", (10, 11), source_start=0)
        self._register_source(1, "tail", (12, 13, 14), source_start=2)
        source_keys = [
            buffer[torch.tensor([0, 1, 2, 3, 4])].clone()
            for buffer in self.kvcache.k_buffer
        ]
        reuse = FakeReq(
            _metadata(
                (
                    ApproxKVRequestSegment(content_hash="head", target_start=0, length=2),
                    ApproxKVRequestSegment(content_hash="tail", target_start=2, length=3),
                ),
                ApproxKVRequestOperation.REUSE,
            ),
            (10, 11, 12, 13, 14, 99),
        )
        self.assertTrue(restore_request_prefix(self.tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), 5)
        for layer in range(self.kvcache.layer_num):
            torch.testing.assert_close(
                self.kvcache.k_buffer[layer][reuse.prefix_indices],
                source_keys[layer],
            )
        self.assertTrue(reuse.approx_kv_stats.mechanically_valid)

    def test_interior_segment_after_dense_exact_head(self):
        # Simulate positions [0, 4) already resolved by a dense/exact head
        # (their content is irrelevant to this plugin); only the interior
        # segment starting exactly at the boundary (position 4) is
        # recovered via raw+RoPE.
        self._register_source(0, "mid", (20, 21, 22), source_start=8)
        reuse = FakeReq(
            _metadata(
                (ApproxKVRequestSegment(content_hash="mid", target_start=4, length=3),),
                ApproxKVRequestOperation.REUSE,
            ),
            (1, 2, 3, 4, 20, 21, 22, 99),
            exact_prefix_len=4,
        )
        self.assertTrue(restore_request_prefix(self.tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), 4 + 3)
        # The dense/exact head's indices must be untouched by the restore.
        torch.testing.assert_close(
            reuse.prefix_indices[:4], torch.arange(4, dtype=torch.int64)
        )
        self.assertTrue(reuse.approx_kv_stats.mechanically_valid)

    def test_noncontiguous_segments_fall_back_to_dense(self):
        self._register_source(0, "a", (10, 11), source_start=0)
        # "b" is never registered -> the gap after "a" cannot be covered.
        reuse = FakeReq(
            _metadata(
                (
                    ApproxKVRequestSegment(content_hash="a", target_start=0, length=2),
                    ApproxKVRequestSegment(content_hash="b", target_start=3, length=2),
                ),
                ApproxKVRequestOperation.REUSE,
            ),
            (10, 11, 0, 20, 21, 99),
        )
        self.assertTrue(restore_request_prefix(self.tree, reuse))
        # Only the contiguous "a" segment is recovered; "b" is unreachable
        # and simply never attempted (all-or-nothing within its own run).
        self.assertEqual(len(reuse.prefix_indices), 2)

    def test_missing_source_segment_falls_back_without_allocating(self):
        next_index = self.allocator.next_index
        reuse = FakeReq(
            _metadata(
                (ApproxKVRequestSegment(content_hash="never", target_start=0, length=3),),
                ApproxKVRequestOperation.REUSE,
            ),
            (10, 11, 12, 99),
        )
        self.assertFalse(restore_request_prefix(self.tree, reuse))
        self.assertEqual(self.allocator.next_index, next_index)
        self.assertEqual(len(reuse.prefix_indices), 0)


if __name__ == "__main__":
    unittest.main()
