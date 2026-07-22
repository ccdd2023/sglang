from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.approx_kv.config import ApproxKVFeatureConfig
from sglang.srt.mem_cache.approx_kv.kvcomm import (
    KVCOMMInvariantError,
    KVCOMMReconstructionPlan,
    KVCOMMRecoveryPlugin,
    KVCOMMRequestSpec,
    KVCOMMRuntimeCapabilities,
    compute_interpolation_weights,
    normalize_key_positions,
    relocate_key_positions,
    rotate_key_positions,
    validate_interpolation_weights,
)
from sglang.srt.mem_cache.approx_kv.manager import ApproxKVManager
from sglang.srt.mem_cache.approx_kv.plugins import RecoveryRequestContext
from sglang.srt.mem_cache.approx_kv.radix_backend import RoPEConfig
from sglang.srt.mem_cache.approx_kv.request import (
    ApproxKVRequestMetadata,
    ApproxKVRequestOperation,
    ApproxKVRequestSegment,
)
from sglang.srt.mem_cache.approx_kv.runtime import (
    allocate_recovery_slots,
    register_request_segments,
    restore_request_prefix,
)
from sglang.srt.mem_cache.approx_kv.types import RecoveryMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-c-test-cpu")


class FakeKVCache:
    def __init__(
        self,
        *,
        capacity: int = 512,
        layers: int = 3,
        heads: int = 2,
        key_dim: int = 8,
        value_dim: int = 8,
    ) -> None:
        self.layer_num = layers
        self.k_buffer = [torch.zeros(capacity, heads, key_dim) for _ in range(layers)]
        self.v_buffer = [torch.zeros(capacity, heads, value_dim) for _ in range(layers)]

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
    size_full = 512

    def __init__(self, kvcache: FakeKVCache) -> None:
        self.kvcache = kvcache
        self.next_index = 128
        self.freed: list[int] = []

    def alloc(self, size):
        if self.next_index + size > self.size_full:
            return None
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


class PressureAllocator(FakeAllocator):
    def __init__(self, kvcache: FakeKVCache) -> None:
        super().__init__(kvcache)
        self.evicted = False

    def available_size(self):
        return 0 if not self.evicted else 64

    def alloc(self, size):
        if not self.evicted:
            return None
        return super().alloc(size)


class FakeEvictingTree:
    def __init__(self, allocator) -> None:
        self.token_to_kv_pool_allocator = allocator
        self.evict_params = []

    def is_chunk_cache(self):
        return False

    def evict(self, params):
        self.evict_params.append(params)
        self.token_to_kv_pool_allocator.evicted = True


class FakeReqToTokenPool:
    def __init__(self) -> None:
        self.req_to_token = torch.full((2, 128), -1, dtype=torch.int64)


class FakeReq:
    def __init__(
        self,
        metadata,
        tokens,
        *,
        prefix_indices=None,
    ) -> None:
        self.approx_kv_metadata = metadata
        self.req_pool_idx = 0
        self.kv = SimpleNamespace(kv_allocated_len=len(tokens))
        self.full_untruncated_fill_ids = list(tokens)
        self.prefix_indices = (
            torch.empty(0, dtype=torch.int64)
            if prefix_indices is None
            else prefix_indices.clone()
        )
        self.rid = "kvcomm-request"

    def effective_kv_committed_len(self):
        return len(self.full_untruncated_fill_ids)

    def needs_host_load_back(self):
        return False


def make_capabilities(
    *,
    supported: bool = True,
    reason: str | None = None,
    attention_arch: str = "MHA",
    model_type: str = "qwen3",
) -> KVCOMMRuntimeCapabilities:
    return KVCOMMRuntimeCapabilities(
        supported=supported,
        reason=reason,
        model_type=model_type,
        attention_arch=attention_arch,
        cache_layout="separate_kv_token_major",
        layer_count=3,
        kv_head_count=2,
        key_head_dim=8,
        value_head_dim=8,
        rope=RoPEConfig(
            rotary_dim=8,
            base=10000.0,
            is_neox_style=True,
        ),
    )


def make_metadata(
    *,
    operation: ApproxKVRequestOperation,
    action: str,
    segments,
    descriptors,
    context: str,
    tokenizer: str = "tokenizer-v1",
    max_anchors: int = 20,
    prune_window: int = 5,
) -> ApproxKVRequestMetadata:
    return ApproxKVRequestMetadata(
        operation=operation,
        segments=tuple(segments),
        model_fingerprint="model-v1",
        cache_dtype="fp32",
        plugin="kvcomm",
        plugin_params={
            "action": action,
            "agent_id": "coder",
            "tokenizer_fingerprint": tokenizer,
            "template_fingerprint": "coder-template-v1",
            "context_fingerprint": context,
            "segments": list(descriptors),
            "entropy_threshold": 0.3,
            "temperature": 0.05,
            "max_anchors": max_anchors,
            "prune_window": prune_window,
            "min_anchors": 2,
        },
    )


class KVCOMMFixture:
    def __init__(self) -> None:
        self.kvcache = FakeKVCache()
        self.allocator = FakeAllocator(self.kvcache)
        self.req_pool = FakeReqToTokenPool()
        self.manager = ApproxKVManager(ApproxKVFeatureConfig(core_enabled=True))
        self.capabilities = make_capabilities()
        self.manager.bind_runtime_capabilities(self.capabilities)
        self.tree = SimpleNamespace(
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=self.req_pool,
            approx_kv=self.manager,
        )
        self.plugin = self.manager.plugins.get("kvcomm")
        assert isinstance(self.plugin, KVCOMMRecoveryPlugin)
        self._source_cursor = 0

    def payload(
        self,
        length: int,
        *,
        value_direction: float,
        seed: int,
    ):
        generator = torch.Generator().manual_seed(seed)
        keys = []
        values = []
        direction = torch.zeros(8)
        direction[0] = value_direction
        direction[1] = 1.0
        for layer in range(self.kvcache.layer_num):
            keys.append(
                torch.randn(
                    length,
                    2,
                    8,
                    generator=generator,
                )
                * 0.05
                + layer * 0.01
            )
            values.append(direction.reshape(1, 1, 8).expand(length, 2, 8).clone())
        return keys, values

    def write_request(
        self,
        req: FakeReq,
        payloads,
    ) -> None:
        prompt_length = len(req.full_untruncated_fill_ids)
        source = torch.arange(
            self._source_cursor,
            self._source_cursor + prompt_length,
            dtype=torch.int64,
        )
        self._source_cursor += prompt_length
        self.req_pool.req_to_token[0, :prompt_length] = source
        for layer in range(self.kvcache.layer_num):
            self.kvcache.k_buffer[layer][source] = 0
            self.kvcache.v_buffer[layer][source] = 0
        for segment_index, (keys, values) in payloads.items():
            segment = req.approx_kv_metadata.segments[segment_index]
            indices = source[segment.target_start : segment.target_end]
            positions = range(segment.target_start, segment.target_end)
            for layer in range(self.kvcache.layer_num):
                self.kvcache.k_buffer[layer][indices] = rotate_key_positions(
                    keys[layer],
                    positions,
                    self.capabilities.rope,
                )
                self.kvcache.v_buffer[layer][indices] = values[layer]

    def register_base(
        self,
        *,
        content: str,
        tokens,
        source: str,
        payload,
        neighbor_tokens=None,
        neighbor_payload=None,
    ) -> None:
        segments = [
            ApproxKVRequestSegment(
                content_hash=content,
                target_start=0,
                length=len(tokens),
            )
        ]
        descriptors = [
            {
                "segment_index": 0,
                "placeholder_id": "shared",
                "role": "placeholder",
                "source_fingerprint": source,
            }
        ]
        prompt_tokens = list(tokens)
        payloads = {0: payload}
        if neighbor_tokens is not None:
            segments.append(
                ApproxKVRequestSegment(
                    content_hash="fixed-neighbor",
                    target_start=len(tokens),
                    length=len(neighbor_tokens),
                )
            )
            descriptors.append(
                {
                    "segment_index": 1,
                    "placeholder_id": "shared",
                    "role": "neighbor",
                    "source_fingerprint": "fixed-neighbor-source",
                }
            )
            prompt_tokens.extend(neighbor_tokens)
            payloads[1] = neighbor_payload
        prompt_tokens.append(999)
        metadata = make_metadata(
            operation=ApproxKVRequestOperation.REGISTER,
            action="base",
            segments=segments,
            descriptors=descriptors,
            context=f"canonical-{content}",
        )
        req = FakeReq(metadata, prompt_tokens)
        self.write_request(req, payloads)
        expected = sum(segment.length for segment in segments)
        assert register_request_segments(self.tree, req) == expected

    def add_anchor(
        self,
        *,
        content: str,
        tokens,
        source: str,
        base_payload,
        placeholder_delta: float,
        neighbor_tokens,
        neighbor_base_payload,
        neighbor_delta: float,
        prefix_length: int,
        context: str,
        max_anchors: int = 20,
    ) -> None:
        segments = [
            ApproxKVRequestSegment(
                content_hash=content,
                target_start=prefix_length,
                length=len(tokens),
            ),
            ApproxKVRequestSegment(
                content_hash="fixed-neighbor",
                target_start=prefix_length + len(tokens),
                length=len(neighbor_tokens),
            ),
        ]
        descriptors = [
            {
                "segment_index": 0,
                "placeholder_id": "shared",
                "role": "placeholder",
                "source_fingerprint": source,
            },
            {
                "segment_index": 1,
                "placeholder_id": "shared",
                "role": "neighbor",
                "source_fingerprint": "fixed-neighbor-source",
            },
        ]
        metadata = make_metadata(
            operation=ApproxKVRequestOperation.REGISTER,
            action="anchor",
            segments=segments,
            descriptors=descriptors,
            context=context,
            max_anchors=max_anchors,
            prune_window=2,
        )
        prompt = [700 + i for i in range(prefix_length)]
        prompt.extend(tokens)
        prompt.extend(neighbor_tokens)
        prompt.append(999)
        req = FakeReq(metadata, prompt)
        placeholder_actual = (
            [
                keys if layer_id == 0 else keys + placeholder_delta
                for layer_id, keys in enumerate(base_payload[0])
            ],
            [
                values if layer_id == 0 else values + placeholder_delta
                for layer_id, values in enumerate(base_payload[1])
            ],
        )
        neighbor_actual = (
            [
                keys if layer_id == 0 else keys + neighbor_delta
                for layer_id, keys in enumerate(neighbor_base_payload[0])
            ],
            [
                values if layer_id == 0 else values + neighbor_delta
                for layer_id, values in enumerate(neighbor_base_payload[1])
            ],
        )
        self.write_request(
            req,
            {
                0: placeholder_actual,
                1: neighbor_actual,
            },
        )
        assert register_request_segments(self.tree, req) == (
            len(tokens) + len(neighbor_tokens)
        )

    def prepare(self):
        target_tokens = (31, 32)
        anchor_one_tokens = (41, 42)
        anchor_two_tokens = (51, 52)
        neighbor_tokens = (61,)
        target_payload = self.payload(
            2,
            value_direction=1.0,
            seed=1,
        )
        anchor_one_payload = self.payload(
            2,
            value_direction=1.0,
            seed=2,
        )
        anchor_two_payload = self.payload(
            2,
            value_direction=-1.0,
            seed=3,
        )
        neighbor_payload = self.payload(
            1,
            value_direction=0.5,
            seed=4,
        )
        self.register_base(
            content="target",
            tokens=target_tokens,
            source="target-source",
            payload=target_payload,
            neighbor_tokens=neighbor_tokens,
            neighbor_payload=neighbor_payload,
        )
        self.register_base(
            content="anchor-one",
            tokens=anchor_one_tokens,
            source="anchor-one-source",
            payload=anchor_one_payload,
        )
        self.register_base(
            content="anchor-two",
            tokens=anchor_two_tokens,
            source="anchor-two-source",
            payload=anchor_two_payload,
        )
        self.add_anchor(
            content="anchor-one",
            tokens=anchor_one_tokens,
            source="anchor-one-source",
            base_payload=anchor_one_payload,
            placeholder_delta=0.5,
            neighbor_tokens=neighbor_tokens,
            neighbor_base_payload=neighbor_payload,
            neighbor_delta=0.25,
            prefix_length=2,
            context="anchor-context-one",
        )
        self.add_anchor(
            content="anchor-two",
            tokens=anchor_two_tokens,
            source="anchor-two-source",
            base_payload=anchor_two_payload,
            placeholder_delta=2.0,
            neighbor_tokens=neighbor_tokens,
            neighbor_base_payload=neighbor_payload,
            neighbor_delta=1.5,
            prefix_length=3,
            context="anchor-context-two",
        )
        return {
            "target_tokens": target_tokens,
            "neighbor_tokens": neighbor_tokens,
            "target_payload": target_payload,
            "neighbor_payload": neighbor_payload,
            "anchor_one_payload": anchor_one_payload,
            "anchor_two_payload": anchor_two_payload,
        }

    def reuse_metadata(self, *, tokenizer="tokenizer-v1"):
        segments = (
            ApproxKVRequestSegment(
                content_hash="target",
                target_start=1,
                length=2,
            ),
            ApproxKVRequestSegment(
                content_hash="fixed-neighbor",
                target_start=3,
                length=1,
            ),
        )
        descriptors = (
            {
                "segment_index": 0,
                "placeholder_id": "shared",
                "role": "placeholder",
                "source_fingerprint": "target-source",
            },
            {
                "segment_index": 1,
                "placeholder_id": "shared",
                "role": "neighbor",
                "source_fingerprint": "fixed-neighbor-source",
            },
        )
        return make_metadata(
            operation=ApproxKVRequestOperation.REUSE,
            action="reuse",
            segments=segments,
            descriptors=descriptors,
            context="target-context",
            tokenizer=tokenizer,
        )

    def build_plan(self, metadata):
        context = RecoveryRequestContext(
            request_id="plan",
            target_token_ids=(800, 31, 32, 61, 999),
            exact_prefix_length=1,
            custom_metadata={"approx_kv_metadata": metadata},
        )
        return self.plugin.build_plan(context, self.manager.store)


class TestKVCOMMCore(unittest.TestCase):
    def test_recovery_allocation_evicts_before_allocating(self):
        allocator = PressureAllocator(FakeKVCache())
        tree = FakeEvictingTree(allocator)
        slots = allocate_recovery_slots(tree, 8)
        self.assertEqual(len(slots), 8)
        self.assertEqual(len(tree.evict_params), 1)
        self.assertEqual(tree.evict_params[0].num_tokens, 8)

    def test_zero_positive_and_negative_relocation(self):
        rope = RoPEConfig(
            rotary_dim=8,
            base=10000.0,
            is_neox_style=True,
        )
        normalized = torch.randn(4, 2, 8)
        source_positions = torch.tensor([4, 5, 6, 7])
        stored = rotate_key_positions(
            normalized,
            source_positions,
            rope,
        )
        zero = relocate_key_positions(
            stored,
            source_positions,
            source_positions,
            rope,
        )
        positive = relocate_key_positions(
            stored,
            source_positions,
            source_positions + 5,
            rope,
        )
        negative = relocate_key_positions(
            positive,
            source_positions + 5,
            source_positions - 3,
            rope,
        )
        torch.testing.assert_close(zero, stored, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            normalize_key_positions(positive, source_positions + 5, rope),
            normalized,
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            normalize_key_positions(negative, source_positions - 3, rope),
            normalized,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_anchor_weights_are_validated(self):
        target = torch.tensor([1.0, 0.0])
        result = compute_interpolation_weights(
            target,
            (
                torch.tensor([1.0, 0.0]),
                torch.tensor([-1.0, 0.0]),
            ),
            temperature=0.05,
        )
        validated = validate_interpolation_weights(result.weights, 2)
        self.assertAlmostEqual(float(validated.sum()), 1.0, places=6)
        self.assertGreater(validated[0], 0.99)
        self.assertLess(result.entropy, 0.3 * torch.log(torch.tensor(2.0)))
        with self.assertRaisesRegex(
            KVCOMMInvariantError,
            "sum to one",
        ):
            validate_interpolation_weights((0.2, 0.2), 2)
        with self.assertRaisesRegex(KVCOMMInvariantError, "finite"):
            validate_interpolation_weights((float("nan"), 1.0), 2)

    def test_server_path_reconstructs_neighbor_and_leaves_last_token(self):
        fixture = KVCOMMFixture()
        data = fixture.prepare()
        metadata = fixture.reuse_metadata()
        req = FakeReq(
            metadata,
            (
                800,
                *data["target_tokens"],
                *data["neighbor_tokens"],
                999,
            ),
            prefix_indices=torch.tensor([20], dtype=torch.int64),
        )
        self.assertTrue(restore_request_prefix(fixture.tree, req))
        self.assertEqual(len(req.prefix_indices), 4)
        self.assertLess(
            len(req.prefix_indices),
            len(req.full_untruncated_fill_ids),
        )
        self.assertTrue(req.kvcomm_reconstructed)
        self.assertEqual(req.approx_kv_stats.layer_count, 3)
        self.assertEqual(req.approx_kv_stats.copied_k_layer_tokens, 9)
        self.assertEqual(req.approx_kv_stats.rotated_k_layer_tokens, 9)
        self.assertEqual(req.approx_kv_stats.copied_v_layer_tokens, 9)
        self.assertTrue(req.approx_kv_stats.mechanically_valid)

        restored = req.prefix_indices[1:]
        placeholder_indices = restored[:2]
        neighbor_indices = restored[2:]
        for layer in range(fixture.kvcache.layer_num):
            placeholder_delta = 0.0 if layer == 0 else 0.5
            neighbor_delta = 0.0 if layer == 0 else 0.25
            placeholder_normalized = normalize_key_positions(
                fixture.kvcache.k_buffer[layer][placeholder_indices],
                (1, 2),
                fixture.capabilities.rope,
            )
            torch.testing.assert_close(
                placeholder_normalized,
                data["target_payload"][0][layer] + placeholder_delta,
                atol=2e-4,
                rtol=2e-4,
            )
            torch.testing.assert_close(
                fixture.kvcache.v_buffer[layer][placeholder_indices],
                data["target_payload"][1][layer] + placeholder_delta,
                atol=2e-4,
                rtol=2e-4,
            )
            neighbor_normalized = normalize_key_positions(
                fixture.kvcache.k_buffer[layer][neighbor_indices],
                (3,),
                fixture.capabilities.rope,
            )
            torch.testing.assert_close(
                neighbor_normalized,
                data["neighbor_payload"][0][layer] + neighbor_delta,
                atol=2e-4,
                rtol=2e-4,
            )
            torch.testing.assert_close(
                fixture.kvcache.v_buffer[layer][neighbor_indices],
                data["neighbor_payload"][1][layer] + neighbor_delta,
                atol=2e-4,
                rtol=2e-4,
            )
        self.assertEqual(register_request_segments(fixture.tree, req), 0)

    def test_exact_prefix_preempts_kvcomm(self):
        fixture = KVCOMMFixture()
        data = fixture.prepare()
        req = FakeReq(
            fixture.reuse_metadata(),
            (
                800,
                *data["target_tokens"],
                *data["neighbor_tokens"],
                999,
            ),
            prefix_indices=torch.tensor([20, 21, 22, 23], dtype=torch.int64),
        )
        next_index = fixture.allocator.next_index
        self.assertFalse(restore_request_prefix(fixture.tree, req))
        self.assertTrue(req.approx_kv_exact_preferred)
        self.assertEqual(fixture.allocator.next_index, next_index)
        self.assertFalse(hasattr(req, "kvcomm_reconstructed"))

    def test_non_lru_runtime_uses_dense_fallback(self):
        fixture = KVCOMMFixture()
        fixture.tree.eviction_policy = "lfu"
        req = FakeReq(
            fixture.reuse_metadata(),
            (800, 31, 32, 61, 999),
        )
        self.assertFalse(restore_request_prefix(fixture.tree, req))
        self.assertEqual(req.kvcomm_fallback_reason, "kvcomm_requires_lru")

    def test_plan_validation_exception_dense_falls_back(self):
        fixture = KVCOMMFixture()
        data = fixture.prepare()
        req = FakeReq(
            fixture.reuse_metadata(),
            (800, *data["target_tokens"], *data["neighbor_tokens"], 999),
            prefix_indices=torch.tensor([20], dtype=torch.int64),
        )
        with patch.object(
            fixture.plugin,
            "validate_plan",
            side_effect=IndexError("injected validation failure"),
        ):
            self.assertFalse(restore_request_prefix(fixture.tree, req))
        self.assertEqual(
            req.kvcomm_fallback_reason,
            "kvcomm_plan_validation_failed",
        )

    def test_execution_index_error_frees_slots_and_dense_falls_back(self):
        fixture = KVCOMMFixture()
        data = fixture.prepare()
        req = FakeReq(
            fixture.reuse_metadata(),
            (800, *data["target_tokens"], *data["neighbor_tokens"], 999),
            prefix_indices=torch.tensor([20], dtype=torch.int64),
        )
        with patch(
            "sglang.srt.mem_cache.approx_kv.runtime.execute_kvcomm_reconstruction",
            side_effect=IndexError("injected execution failure"),
        ):
            self.assertFalse(restore_request_prefix(fixture.tree, req))
        self.assertEqual(req.kvcomm_fallback_reason, "kvcomm_execution_failed")
        self.assertTrue(fixture.allocator.freed)

    def test_neighbor_mismatch_filters_anchor_pool(self):
        fixture = KVCOMMFixture()
        data = fixture.prepare()
        alternate_neighbor = (62,)
        alternate_payload = fixture.payload(
            1,
            value_direction=0.75,
            seed=12,
        )
        fixture.register_base(
            content="target",
            tokens=data["target_tokens"],
            source="target-source",
            payload=data["target_payload"],
            neighbor_tokens=alternate_neighbor,
            neighbor_payload=alternate_payload,
        )
        metadata = fixture.reuse_metadata()
        plan = fixture.plugin.build_plan(
            RecoveryRequestContext(
                request_id="neighbor-mismatch",
                target_token_ids=(
                    800,
                    *data["target_tokens"],
                    *alternate_neighbor,
                    999,
                ),
                exact_prefix_length=1,
                custom_metadata={"approx_kv_metadata": metadata},
            ),
            fixture.manager.store,
        )
        self.assertEqual(plan.recovery_mode, RecoveryMode.DENSE)
        self.assertEqual(
            plan.dense_ranges[0].reason,
            "insufficient_compatible_anchors",
        )

    def test_stale_generation_and_provenance_use_dense_fallback(self):
        fixture = KVCOMMFixture()
        fixture.prepare()
        metadata = fixture.reuse_metadata()
        plan = fixture.build_plan(metadata)
        self.assertEqual(plan.recovery_mode, RecoveryMode.KVCOMM)
        self.assertIsInstance(plan.plugin_data, KVCOMMReconstructionPlan)
        stale_delta = plan.plugin_data.slices[0].deltas[0].handle
        self.assertTrue(fixture.manager.store.release(stale_delta))
        self.assertEqual(
            fixture.plugin.validate_plan(
                plan.plugin_data,
                fixture.manager.store,
            ),
            "stale_delta_generation",
        )

        mismatch = fixture.build_plan(fixture.reuse_metadata(tokenizer="tokenizer-v2"))
        self.assertEqual(mismatch.recovery_mode, RecoveryMode.DENSE)
        self.assertEqual(
            mismatch.dense_ranges[0].reason,
            "base_provenance_mismatch",
        )
        source_params = dict(metadata.plugin_params)
        source_segments = [dict(descriptor) for descriptor in source_params["segments"]]
        source_segments[0]["source_fingerprint"] = "different-target-source"
        source_params["segments"] = source_segments
        source_mismatch = fixture.build_plan(
            replace(metadata, plugin_params=source_params)
        )
        self.assertEqual(source_mismatch.recovery_mode, RecoveryMode.DENSE)
        self.assertEqual(
            source_mismatch.dense_ranges[0].reason,
            "base_source_mismatch",
        )

        fixture = KVCOMMFixture()
        fixture.prepare()
        plan = fixture.build_plan(fixture.reuse_metadata())
        anchor_base = fixture.manager.store.lookup(
            plan.plugin_data.slices[0].deltas[0].base_key
        )
        self.assertIsNotNone(anchor_base)
        self.assertTrue(fixture.manager.store.release(anchor_base))
        self.assertEqual(
            fixture.plugin.validate_plan(
                plan.plugin_data,
                fixture.manager.store,
            ),
            "stale_anchor_base_generation",
        )

    def test_online_updates_advance_generations_and_prune(self):
        fixture = KVCOMMFixture()
        data = fixture.prepare()
        snapshot = fixture.plugin.pool_snapshot()
        pool_key, anchors = next(iter(snapshot.items()))
        first_generation = fixture.plugin.pool_generation(pool_key)
        first_anchor = anchors[0]
        first_delta = first_anchor.placeholder_delta.handle

        fixture.add_anchor(
            content="anchor-one",
            tokens=(41, 42),
            source="anchor-one-source",
            base_payload=data["anchor_one_payload"],
            placeholder_delta=0.75,
            neighbor_tokens=data["neighbor_tokens"],
            neighbor_base_payload=data["neighbor_payload"],
            neighbor_delta=0.5,
            prefix_length=2,
            context="anchor-context-one",
        )
        self.assertGreater(
            fixture.plugin.pool_generation(pool_key),
            first_generation,
        )
        self.assertFalse(fixture.manager.store.is_current(first_delta))

        third_payload = fixture.payload(
            2,
            value_direction=0.5,
            seed=9,
        )
        fixture.register_base(
            content="anchor-three",
            tokens=(71, 72),
            source="anchor-three-source",
            payload=third_payload,
        )
        fixture.add_anchor(
            content="anchor-three",
            tokens=(71, 72),
            source="anchor-three-source",
            base_payload=third_payload,
            placeholder_delta=1.0,
            neighbor_tokens=data["neighbor_tokens"],
            neighbor_base_payload=data["neighbor_payload"],
            neighbor_delta=0.75,
            prefix_length=2,
            context="anchor-context-three",
            max_anchors=2,
        )
        pool = fixture.plugin.pool_snapshot()[pool_key]
        self.assertEqual(len(pool), 2)
        self.assertEqual(len({anchor.generation for anchor in pool}), 2)

    def test_capability_guards_mla_scaled_rope_and_layout(self):
        unsupported = make_capabilities(
            supported=False,
            reason="unsupported_attention_arch",
            attention_arch="MLA",
        )
        fixture = KVCOMMFixture()
        fixture.manager.bind_runtime_capabilities(unsupported)
        plan = fixture.build_plan(fixture.reuse_metadata())
        self.assertEqual(plan.recovery_mode, RecoveryMode.DENSE)
        self.assertEqual(
            plan.dense_ranges[0].reason,
            "unsupported_attention_arch",
        )

        text_config = SimpleNamespace(
            model_type="llama",
            num_hidden_layers=3,
            partial_rotary_factor=1.0,
            rope_theta=500000.0,
            rope_is_neox_style=True,
            rope_scaling={"rope_type": "llama3"},
        )
        model_config = SimpleNamespace(
            hf_text_config=text_config,
            hf_config=text_config,
            attention_arch=SimpleNamespace(name="MHA"),
            head_dim=8,
            v_head_dim=8,
            num_key_value_heads=2,
        )
        scaled = KVCOMMRuntimeCapabilities.from_model_config(
            model_config,
            tp_size=1,
            is_hybrid_swa=False,
            is_hybrid_ssm=False,
            is_multimodal=False,
            is_speculative=False,
        )
        self.assertFalse(scaled.supported)
        self.assertEqual(scaled.reason, "unsupported_rope_scaling")

        bad_layout = FakeKVCache()
        bad_layout.v_buffer[0] = bad_layout.k_buffer[0]
        self.assertEqual(
            make_capabilities().guard_kvcache(bad_layout),
            "unsupported_cache_layout",
        )
        full_layer_range = FakeKVCache()
        full_layer_range.start_layer = 0
        full_layer_range.end_layer = full_layer_range.layer_num
        self.assertIsNone(make_capabilities().guard_kvcache(full_layer_range))
        self.assertIsNone(
            make_capabilities().guard_declared_dtype(FakeKVCache(), "fp32")
        )
        self.assertEqual(
            make_capabilities().guard_declared_dtype(FakeKVCache(), "bf16"),
            "cache_dtype_mismatch",
        )
        self.assertEqual(
            make_capabilities().guard_declared_dtype(FakeKVCache(), "auto"),
            "cache_dtype_unspecified",
        )
        quantized = FakeKVCache()
        quantized.is_quantized_kv_cache = True
        self.assertEqual(
            make_capabilities().guard_kvcache(quantized),
            "unsupported_cache_dtype",
        )

        speculative = KVCOMMRuntimeCapabilities.from_model_config(
            model_config,
            tp_size=1,
            is_hybrid_swa=False,
            is_hybrid_ssm=False,
            is_multimodal=False,
            is_speculative=True,
        )
        self.assertFalse(speculative.supported)
        self.assertEqual(
            speculative.reason,
            "unsupported_speculative_cache",
        )

        metadata = KVCOMMFixture().reuse_metadata()
        params = dict(metadata.plugin_params)
        params["max_anchors"] = 1
        with self.assertRaisesRegex(
            ValueError,
            "min_anchors cannot exceed max_anchors",
        ):
            KVCOMMRequestSpec.from_metadata(replace(metadata, plugin_params=params))


if __name__ == "__main__":
    unittest.main()
