from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.approx_kv.config import ApproxKVFeatureConfig
from sglang.srt.mem_cache.approx_kv.epic_capability import (
    LayerwiseCapability,
    inspect_layerwise_recompute_capability,
    inspect_source_layer_forward_params,
)
from sglang.srt.mem_cache.approx_kv.epic_plugin import (
    EPIC_LEADING_K_REPAIR_REASON,
    EPICLeadingKPlugin,
    carve_leading_k,
)
from sglang.srt.mem_cache.approx_kv.epic_recompute import (
    EpicRecomputeStats,
    LayerwiseEpicExecutor,
    LayerwiseLeadingKRepairError,
)
from sglang.srt.mem_cache.approx_kv.epic_runtime import (
    EpicForwardBatchBundle,
    restore_request_prefix_epic,
)
from sglang.srt.mem_cache.approx_kv.manager import ApproxKVManager
from sglang.srt.mem_cache.approx_kv.plugins import RecoveryRequestContext
from sglang.srt.mem_cache.approx_kv.radix_backend import (
    AllocatorCPUResidencyBackend,
)
from sglang.srt.mem_cache.approx_kv.request import (
    ApproxKVRequestMetadata,
    ApproxKVRequestOperation,
    ApproxKVRequestSegment,
)
from sglang.srt.mem_cache.approx_kv.runtime import register_request_segments
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------
# Shared fakes (extend the FakeKVCache/FakeAllocator pattern already used by
# test_approx_kv_runtime.py) -- a real allocator + a real, multi-layer KV
# buffer, plus a fake but genuinely tensor-computing decoder-layer stack that
# proves ``LayerwiseEpicExecutor``/``ModelRunnerLeadingKRecomputeBackend``
# drive real per-layer forward calls rather than a copy-shaped stub.
# ---------------------------------------------------------------------------


class FakeKVCache:
    def __init__(self, layer_num=3, pool_size=128, num_heads=2, head_dim=8):
        self.layer_num = layer_num
        shape = (pool_size, num_heads, head_dim)
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

    def __init__(self, kvcache, start=32):
        self.kvcache = kvcache
        self.next_index = start
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
    def __init__(self, pool_size=128):
        self.req_to_token = torch.full((2, pool_size), -1, dtype=torch.int64)


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


class FakeDecoderLayer:
    """A genuinely tensor-computing decoder layer used to *prove* that
    ``LayerwiseEpicExecutor``/``ModelRunnerLeadingKRecomputeBackend`` drive
    a real, chained forward pass rather than a success-shaped stub.

    ``forward`` derives its outputs from the actual ``positions`` and
    ``hidden_states`` it is given (a per-layer affine transform, standing
    in for real attention+MLP math), and -- mirroring how a real decoder
    layer's ``self.attn(..., save_kv_cache=True)`` call writes K/V into the
    pool as a side effect of a real forward -- writes newly *derived*
    (not copied) K/V values into ``forward_batch.leading_k_target_indices``
    for this layer only.
    """

    def __init__(self, layer_id: int, kvcache: FakeKVCache) -> None:
        self.layer_id = layer_id
        self.kvcache = kvcache
        self.forward_calls: list[tuple[int, torch.Tensor]] = []

    def forward(self, *, positions, hidden_states, forward_batch, residual):
        self.forward_calls.append((self.layer_id, positions.clone()))
        layer_scale = float(self.layer_id + 1)
        new_hidden = hidden_states * layer_scale + positions.unsqueeze(-1).float()
        new_residual = new_hidden if residual is None else residual + new_hidden

        target_indices = forward_batch.leading_k_target_indices
        key_buffer = self.kvcache.get_key_buffer(self.layer_id)
        value_buffer = self.kvcache.get_value_buffer(self.layer_id)
        derived = new_hidden.sum(dim=-1)
        k_shape = key_buffer.shape[1:]
        k_values = (
            derived.view(-1, *([1] * len(k_shape)))
            .expand(len(target_indices), *k_shape)
            .clone()
        )
        v_values = k_values + 5000.0
        key_buffer[target_indices] = k_values
        value_buffer[target_indices] = v_values
        return new_hidden, new_residual


class FakeModel:
    def __init__(self, layers):
        self.layers = layers


class FakeModelWrapper:
    def __init__(self, layers):
        self.model = FakeModel(layers)


class FakeModelRunner:
    def __init__(self, kvcache: FakeKVCache, num_layers: int = 3):
        self.layers = [FakeDecoderLayer(i, kvcache) for i in range(num_layers)]
        self.model = FakeModelWrapper(self.layers)


class NonConformingLayer:
    def forward(self, x):
        return x


def _fake_forward_batch_factory(tree_cache, req, resolved, k, leading_k_target_indices):
    del tree_cache, req
    positions = torch.arange(
        resolved.exact_length,
        resolved.exact_length + k,
        dtype=torch.int64,
    )
    hidden_states = torch.zeros((k, 4))
    forward_batch = SimpleNamespace(leading_k_target_indices=leading_k_target_indices)
    return EpicForwardBatchBundle(
        positions=positions,
        hidden_states=hidden_states,
        residual=None,
        forward_batch=forward_batch,
    )


# ---------------------------------------------------------------------------
# Capability guard tests
# ---------------------------------------------------------------------------


class TestEpicCapability(unittest.TestCase):
    def test_conforming_fake_model_is_supported(self):
        model_runner = FakeModelRunner(FakeKVCache(), num_layers=4)
        capability = inspect_layerwise_recompute_capability(model_runner)
        self.assertTrue(capability.supported)
        self.assertEqual(capability.num_layers, 4)
        self.assertTrue(bool(capability))

    def test_missing_layers_is_unsupported(self):
        model_runner = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace()))
        capability = inspect_layerwise_recompute_capability(model_runner)
        self.assertFalse(capability.supported)
        self.assertIn("layers", capability.reason)

    def test_missing_model_attribute_is_unsupported(self):
        capability = inspect_layerwise_recompute_capability(SimpleNamespace())
        self.assertFalse(capability.supported)

    def test_non_conforming_layer_forward_signature_is_unsupported(self):
        model_runner = SimpleNamespace(
            model=SimpleNamespace(model=SimpleNamespace(layers=[NonConformingLayer()]))
        )
        capability = inspect_layerwise_recompute_capability(model_runner)
        self.assertFalse(capability.supported)
        self.assertIn("forward()", capability.reason)

    def test_real_qwen3_decoder_layer_matches_contract(self):
        # Proves the capability guard's contract against real upstream
        # source, without importing the full (CUDA-heavy) sglang package.
        qwen3_path = REPO_ROOT / "python/sglang/srt/models/qwen3.py"
        params = inspect_source_layer_forward_params(qwen3_path, "Qwen3DecoderLayer")
        for required in ("positions", "hidden_states", "forward_batch"):
            self.assertIn(required, params)

    def test_layerwise_capability_bool_and_repr(self):
        capability = LayerwiseCapability(False, "no reason given")
        self.assertFalse(capability)
        self.assertFalse(capability.supported)


# ---------------------------------------------------------------------------
# LayerwiseEpicExecutor: mechanical proof of genuine layer-by-layer
# interleaving (not a success-shaped stub).
# ---------------------------------------------------------------------------


class RecordingRecomputeBackend:
    """A minimal recompute backend that genuinely chains hidden_states."""

    def __init__(self, num_layers: int) -> None:
        self._num_layers = num_layers
        self.calls: list[int] = []

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def recompute_layer(
        self, *, layer_id, positions, hidden_states, residual, forward_batch
    ):
        del positions, forward_batch
        self.calls.append(layer_id)
        new_hidden = hidden_states + 1
        new_residual = new_hidden if residual is None else residual + new_hidden
        return new_hidden, new_residual


class RecordingBodyCopyBackend:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def copy_layer(self, *, layer_id):
        self.calls.append(layer_id)


class TestLayerwiseEpicExecutor(unittest.TestCase):
    def test_genuine_interleave_order_recompute_then_copy_per_layer(self):
        recompute = RecordingRecomputeBackend(num_layers=4)
        copy_backend = RecordingBodyCopyBackend()
        executor = LayerwiseEpicExecutor(
            recompute_backend=recompute, body_copy_backend=copy_backend
        )
        stats, hidden_states, _ = executor.run(
            positions=torch.arange(2),
            hidden_states=torch.zeros(2, 3),
            residual=None,
            forward_batch=None,
            leading_k_tokens=2,
            body_tokens=5,
        )
        self.assertEqual(
            stats.layer_order,
            [
                "recompute:0",
                "copy:0",
                "recompute:1",
                "copy:1",
                "recompute:2",
                "copy:2",
                "recompute:3",
                "copy:3",
            ],
        )
        self.assertTrue(stats.genuinely_layerwise)
        self.assertEqual(stats.layers_invoked, 4)
        self.assertEqual(recompute.calls, [0, 1, 2, 3])
        self.assertEqual(copy_backend.calls, [0, 1, 2, 3])
        # hidden_states was genuinely chained across all 4 layers (each
        # layer adds 1): starting from zeros, ending at 4.
        torch.testing.assert_close(hidden_states, torch.full((2, 3), 4.0))

    def test_body_tokens_zero_skips_copy_but_still_recomputes_every_layer(self):
        recompute = RecordingRecomputeBackend(num_layers=3)
        executor = LayerwiseEpicExecutor(
            recompute_backend=recompute, body_copy_backend=None
        )
        stats, _, _ = executor.run(
            positions=torch.arange(2),
            hidden_states=torch.zeros(2, 3),
            residual=None,
            forward_batch=None,
            leading_k_tokens=2,
            body_tokens=0,
        )
        self.assertEqual(
            stats.layer_order, ["recompute:0", "recompute:1", "recompute:2"]
        )
        self.assertTrue(stats.genuinely_layerwise)

    def test_body_tokens_positive_requires_body_copy_backend(self):
        recompute = RecordingRecomputeBackend(num_layers=2)
        executor = LayerwiseEpicExecutor(
            recompute_backend=recompute, body_copy_backend=None
        )
        with self.assertRaises(LayerwiseLeadingKRepairError):
            executor.run(
                positions=torch.arange(2),
                hidden_states=torch.zeros(2, 3),
                residual=None,
                forward_batch=None,
                leading_k_tokens=2,
                body_tokens=3,
            )

    def test_leading_k_tokens_must_be_positive(self):
        recompute = RecordingRecomputeBackend(num_layers=2)
        executor = LayerwiseEpicExecutor(
            recompute_backend=recompute, body_copy_backend=None
        )
        with self.assertRaises(ValueError):
            executor.run(
                positions=torch.arange(0),
                hidden_states=torch.zeros(0, 3),
                residual=None,
                forward_batch=None,
                leading_k_tokens=0,
                body_tokens=0,
            )

    def test_genuinely_layerwise_detects_reordered_stub(self):
        # A hand-crafted, non-interleaved order must be rejected by the
        # mechanical proof -- this is the regression test guarding against
        # a "plan k but never actually interleave" style stub.
        stats = EpicRecomputeStats(
            layers_invoked=2, leading_k_tokens=2, body_tokens_copied=4
        )
        stats.layer_order = ["recompute:0", "recompute:1", "copy:0", "copy:1"]
        self.assertFalse(stats.genuinely_layerwise)


# ---------------------------------------------------------------------------
# EPICLeadingKPlugin: policy/planning tests across every supported k.
# ---------------------------------------------------------------------------


class TestEPICLeadingKPlugin(unittest.TestCase):
    def test_supported_k_values_match_config(self):
        self.assertEqual(
            ApproxKVFeatureConfig().epic_k, 0
        )  # sanity: default is the degenerate case
        for k in (0, 2, 4, 8, 16, 32):
            ApproxKVFeatureConfig(core_enabled=True, epic_enabled=True, epic_k=k)

    def test_leading_k_window_clamps_to_restore_length(self):
        plugin = EPICLeadingKPlugin(k=32)
        self.assertEqual(plugin.leading_k_window(10), 10)
        self.assertEqual(plugin.leading_k_window(0), 0)
        self.assertEqual(plugin.leading_k_window(40), 32)

    def test_build_plan_carves_leading_k_dense_range_for_every_k(self):
        for k in (0, 2, 4, 8, 16, 32):
            with self.subTest(k=k):
                plugin = EPICLeadingKPlugin(k=k)
                restore_length = 40
                context = RecoveryRequestContext(
                    request_id="req",
                    target_token_ids=tuple(range(restore_length)),
                    exact_prefix_length=0,
                    custom_metadata={"resolved_spans": ()},
                )
                plan = plugin.build_plan(context, store=None)
                if k == 0:
                    self.assertEqual(plan.dense_ranges, ())
                else:
                    self.assertEqual(len(plan.dense_ranges), 1)
                    dense = plan.dense_ranges[0]
                    self.assertEqual(dense.target_start, 0)
                    self.assertEqual(dense.length, k)
                    self.assertEqual(dense.reason, EPIC_LEADING_K_REPAIR_REASON)

    def test_scheduler_metadata_reports_leading_k(self):
        plugin = EPICLeadingKPlugin(k=8)
        context = RecoveryRequestContext(
            request_id="req-1",
            target_token_ids=tuple(range(20)),
            exact_prefix_length=0,
            custom_metadata={},
        )
        (metadata,) = plugin.scheduler_metadata(context)
        self.assertEqual(metadata.workflow_stage, "epic_leading_k=8")

    def test_carve_leading_k_splits_overlapping_span(self):
        from sglang.srt.mem_cache.approx_kv.types import (
            KVSegmentHandle,
            KVSegmentKey,
            ResidencyTier,
            SegmentKind,
            TransferSpan,
        )

        key = KVSegmentKey(
            content_hash="artifact",
            token_hash="hash",
            token_count=10,
            model_fingerprint="model",
            cache_dtype="fp32",
            kind=SegmentKind.ARTIFACT,
        )
        handle = KVSegmentHandle(
            key=key,
            generation=1,
            residency=ResidencyTier.DEVICE,
            source_start=0,
            token_ids=tuple(range(10)),
            backend_ref=None,
        )
        span = TransferSpan(
            source=handle,
            source_offset=0,
            target_start=0,
            length=10,
            rope_delta=0,
            chunk_start=0,
            chunk_length=10,
        )
        carved = carve_leading_k((span,), 4)
        self.assertEqual(len(carved), 1)
        self.assertEqual(carved[0].target_start, 4)
        self.assertEqual(carved[0].length, 6)
        self.assertEqual(carved[0].source_offset, 4)

        # k == whole span consumes it entirely.
        self.assertEqual(carve_leading_k((span,), 10), ())
        # k == 0 leaves spans untouched.
        self.assertEqual(carve_leading_k((span,), 0), (span,))


# ---------------------------------------------------------------------------
# ApproxKVFeatureConfig: EPIC-specific validation.
# ---------------------------------------------------------------------------


class TestApproxKVFeatureConfigEpic(unittest.TestCase):
    def test_unsupported_k_value_rejected(self):
        with self.assertRaises(ValueError):
            ApproxKVFeatureConfig(core_enabled=True, epic_enabled=True, epic_k=3)

    def test_epic_enabled_requires_core_enabled(self):
        with self.assertRaises(ValueError):
            ApproxKVFeatureConfig(core_enabled=False, epic_enabled=True)

    def test_from_env_reads_epic_settings(self):
        env = {
            "SGLANG_APPROX_KV_CORE": "1",
            "SGLANG_APPROX_KV_EPIC": "1",
            "SGLANG_APPROX_KV_EPIC_K": "16",
            "SGLANG_APPROX_KV_EPIC_ATTENTION_SINK": "0",
        }
        config = ApproxKVFeatureConfig.from_env(env)
        self.assertTrue(config.epic_enabled)
        self.assertEqual(config.epic_k, 16)
        self.assertFalse(config.epic_attention_sink)

    def test_from_env_rejects_unsupported_k(self):
        env = {
            "SGLANG_APPROX_KV_CORE": "1",
            "SGLANG_APPROX_KV_EPIC": "1",
            "SGLANG_APPROX_KV_EPIC_K": "3",
        }
        with self.assertRaises(ValueError):
            ApproxKVFeatureConfig.from_env(env)


# ---------------------------------------------------------------------------
# Full epic_runtime.restore_request_prefix_epic integration tests: proves
# real per-layer recompute + per-layer body copy against a genuine
# allocator/KV-buffer, for every supported k value.
# ---------------------------------------------------------------------------


class TestEpicRuntimeIntegration(unittest.TestCase):
    NUM_LAYERS = 3
    SOURCE_LEN = 40

    def setUp(self):
        self.kvcache = FakeKVCache(layer_num=self.NUM_LAYERS, pool_size=256)
        self.allocator = FakeAllocator(self.kvcache, start=64)
        self.req_pool = FakeReqToTokenPool(pool_size=256)
        self.req_pool.req_to_token[0, : self.SOURCE_LEN] = torch.arange(self.SOURCE_LEN)
        config = ApproxKVFeatureConfig(
            core_enabled=True,
            host_residency_enabled=True,
            epic_enabled=True,
            epic_k=0,
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
            length=self.SOURCE_LEN,
        )
        source_tokens = tuple(range(1000, 1000 + self.SOURCE_LEN))
        source = FakeReq(
            self._metadata(ApproxKVRequestOperation.REGISTER),
            source_tokens,
        )
        register_request_segments(self.tree, source)
        self.source_k = [
            buffer[torch.arange(self.SOURCE_LEN)].clone()
            for buffer in self.kvcache.k_buffer
        ]
        self.source_v = [
            buffer[torch.arange(self.SOURCE_LEN)].clone()
            for buffer in self.kvcache.v_buffer
        ]
        self.source_tokens = source_tokens

    def _metadata(self, operation):
        return ApproxKVRequestMetadata(
            operation=operation,
            segments=(self.segment,),
            model_fingerprint="model",
            cache_dtype="fp32",
        )

    def _make_reuse_req(self):
        # reuse tokens: same body as source, plus one distinct real final
        # prompt token (never approximated -- matches the "last prompt
        # token real forward" invariant).
        tokens = self.source_tokens + (99999,)
        return FakeReq(self._metadata(ApproxKVRequestOperation.REUSE), tokens)

    def _set_epic_plugin(self, k, attention_sink=True):
        self.manager.plugins._plugins.pop("epic", None)
        self.manager.register_plugin(
            EPICLeadingKPlugin(k=k, attention_sink=attention_sink)
        )
        object.__setattr__(self.manager.config, "epic_k", k)

    def test_k0_delegates_to_identical_raw_copy_mechanism(self):
        self._set_epic_plugin(0)
        reuse = self._make_reuse_req()
        self.assertTrue(restore_request_prefix_epic(self.tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), self.SOURCE_LEN)
        for layer in range(self.NUM_LAYERS):
            torch.testing.assert_close(
                self.kvcache.k_buffer[layer][reuse.prefix_indices], self.source_k[layer]
            )
            torch.testing.assert_close(
                self.kvcache.v_buffer[layer][reuse.prefix_indices], self.source_v[layer]
            )

    def test_missing_model_runner_dense_falls_back(self):
        self._set_epic_plugin(4)
        reuse = self._make_reuse_req()
        self.assertFalse(restore_request_prefix_epic(self.tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), 0)

    def test_precomputed_leading_k_uses_dense_repair_provenance(self):
        self._set_epic_plugin(4)
        repair_segment = ApproxKVRequestSegment(
            content_hash="epic-repair:artifact",
            target_start=0,
            length=4,
        )
        body_segment = ApproxKVRequestSegment(
            content_hash="epic-body:artifact",
            target_start=4,
            length=self.SOURCE_LEN - 4,
        )
        source = FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REGISTER,
                segments=(repair_segment, body_segment),
                model_fingerprint="model",
                cache_dtype="fp32",
            ),
            self.source_tokens,
        )
        register_request_segments(self.tree, source)
        reuse = FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REUSE,
                segments=(repair_segment, body_segment),
                model_fingerprint="model",
                cache_dtype="fp32",
                plugin="epic_precomputed",
            ),
            self.source_tokens + (99999,),
        )
        self.assertTrue(restore_request_prefix_epic(self.tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), self.SOURCE_LEN)

    def test_missing_forward_batch_factory_dense_falls_back(self):
        self._set_epic_plugin(4)
        self.manager.bind_model_runner(FakeModelRunner(self.kvcache, self.NUM_LAYERS))
        reuse = self._make_reuse_req()
        self.assertFalse(restore_request_prefix_epic(self.tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), 0)

    def test_capability_rejected_model_dense_falls_back(self):
        self._set_epic_plugin(4)
        self.manager.bind_model_runner(
            SimpleNamespace(
                model=SimpleNamespace(
                    model=SimpleNamespace(layers=[NonConformingLayer()])
                )
            )
        )
        self.manager.epic_forward_batch_factory = _fake_forward_batch_factory
        reuse = self._make_reuse_req()
        self.assertFalse(restore_request_prefix_epic(self.tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), 0)

    def test_leading_k_repair_recomputes_and_copies_per_layer_genuinely(self):
        k = 4
        self._set_epic_plugin(k)
        model_runner = FakeModelRunner(self.kvcache, self.NUM_LAYERS)
        self.manager.bind_model_runner(model_runner)
        self.manager.epic_forward_batch_factory = _fake_forward_batch_factory

        reuse = self._make_reuse_req()
        self.assertTrue(restore_request_prefix_epic(self.tree, reuse))

        self.assertEqual(len(reuse.prefix_indices), self.SOURCE_LEN)
        exec_stats = reuse.approx_kv_epic_stats
        self.assertTrue(exec_stats.genuinely_layerwise)
        self.assertEqual(exec_stats.layers_invoked, self.NUM_LAYERS)
        self.assertEqual(exec_stats.leading_k_tokens, k)
        self.assertEqual(exec_stats.body_tokens_copied, self.SOURCE_LEN - k)

        # Every fake decoder layer was genuinely invoked exactly once
        # with a real, distinct positions tensor -- not skipped/stubbed.
        for layer in model_runner.layers:
            self.assertEqual(len(layer.forward_calls), 1)
            called_layer_id, positions = layer.forward_calls[0]
            self.assertEqual(called_layer_id, layer.layer_id)
            self.assertEqual(len(positions), k)

        leading_indices = reuse.prefix_indices[:k]
        body_indices = reuse.prefix_indices[k:]

        # The leading-k slots must NOT equal the raw source values (proof
        # that real recompute -- not a copy -- produced them).
        for layer in range(self.NUM_LAYERS):
            self.assertFalse(
                torch.equal(
                    self.kvcache.k_buffer[layer][leading_indices],
                    self.source_k[layer][:k],
                )
            )
            # The body slots (after the leading-k window) must exactly
            # match the raw registered source (genuine per-layer copy).
            torch.testing.assert_close(
                self.kvcache.k_buffer[layer][body_indices],
                self.source_k[layer][k:],
            )
            torch.testing.assert_close(
                self.kvcache.v_buffer[layer][body_indices],
                self.source_v[layer][k:],
            )

    def test_all_supported_k_values_end_to_end(self):
        for k in (0, 2, 4, 8, 16, 32):
            with self.subTest(k=k):
                self.setUp()
                self._set_epic_plugin(k)
                model_runner = FakeModelRunner(self.kvcache, self.NUM_LAYERS)
                self.manager.bind_model_runner(model_runner)
                self.manager.epic_forward_batch_factory = _fake_forward_batch_factory

                reuse = self._make_reuse_req()
                self.assertTrue(restore_request_prefix_epic(self.tree, reuse))
                self.assertEqual(len(reuse.prefix_indices), self.SOURCE_LEN)
                if k > 0:
                    exec_stats = reuse.approx_kv_epic_stats
                    self.assertTrue(exec_stats.genuinely_layerwise)
                    self.assertEqual(exec_stats.leading_k_tokens, k)

    def test_last_prompt_token_is_never_part_of_reusable_region(self):
        self._set_epic_plugin(4)
        model_runner = FakeModelRunner(self.kvcache, self.NUM_LAYERS)
        self.manager.bind_model_runner(model_runner)
        self.manager.epic_forward_batch_factory = _fake_forward_batch_factory
        reuse = self._make_reuse_req()
        restore_request_prefix_epic(self.tree, reuse)
        # SOURCE_LEN + 1 tokens total; the reusable region must stop one
        # token short of the full prompt (the real, always-forwarded last
        # prompt token is never approximated).
        self.assertEqual(
            len(reuse.prefix_indices), len(reuse.full_untruncated_fill_ids) - 1
        )

    def test_no_reusable_region_never_touches_capability_or_factory(self):
        # A request with no matching segments must short-circuit via
        # resolve_reuse_spans before any EPIC-specific machinery runs,
        # exactly like the raw R0 path -- no exact-Radix-adjacent
        # bookkeeping is disturbed.
        self._set_epic_plugin(4)
        mismatched = FakeReq(
            self._metadata(ApproxKVRequestOperation.REUSE),
            tuple(range(2000, 2000 + self.SOURCE_LEN)) + (99999,),
        )
        self.assertFalse(restore_request_prefix_epic(self.tree, mismatched))
        self.assertEqual(len(mismatched.prefix_indices), 0)
        self.assertEqual(self.manager.model_runner, None)


if __name__ == "__main__":
    unittest.main()
