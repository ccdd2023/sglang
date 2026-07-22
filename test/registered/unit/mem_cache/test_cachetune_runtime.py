from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.approx_kv.config import ApproxKVFeatureConfig
from sglang.srt.mem_cache.approx_kv.manager import ApproxKVManager
from sglang.srt.mem_cache.approx_kv.radix_backend import AllocatorCPUResidencyBackend
from sglang.srt.mem_cache.approx_kv.request import (
    ApproxKVRequestMetadata,
    ApproxKVRequestOperation,
    ApproxKVRequestSegment,
)
from sglang.srt.mem_cache.approx_kv.runtime import register_request_segments
from sglang.srt.mem_cache.cachetune.controller import CacheTuneController
from sglang.srt.mem_cache.cachetune.hardware_profile import (
    CacheTuneMode,
    HardwareMeasurement,
    HardwareProfileKey,
    chunk_length_bucket,
)
from sglang.srt.mem_cache.cachetune.plugin import (
    CACHETUNE_PLUGIN_NAME,
    CacheTuneConfig,
    CacheTuneRecoveryPlugin,
)
from sglang.srt.mem_cache.cachetune.recompute import LayerRecomputeResult
from sglang.srt.mem_cache.cachetune.runtime import restore_request_prefix_cachetune
from sglang.srt.mem_cache.cachetune.token_selection import GradualFilterStage
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")

LAYER_NUM = 3
SLOT_SHAPE = (2, 4)
BUFFER_SLOTS = 320
SOURCE_BASE = 200
RESTORE_LENGTH = 100
HOT_LOCAL_POSITIONS = (3, 27, 51, 68, 91)  # exactly 5 tokens
HARDWARE_TIER = "test-tier"
MODEL_FINGERPRINT = "model"

# roofline_ratio = t_i / (t_c + t_i) == 1.0 / (19.0 + 1.0) == 0.05, which
# quantizes to exactly round_half_up(0.05 * 100) == 5 repair tokens over
# the 100-token restored span -- matching len(HOT_LOCAL_POSITIONS).
FIVE_PERCENT_MEASUREMENT = HardwareMeasurement(t_c_ms=19.0, t_i_ms=1.0, t_o_ms=0.0)
# t_c wildly larger than t_i: roofline collapses to ~0, quantizing to
# exactly 0 repair tokens under the speed-only 0% floor.
ZERO_REPAIR_MEASUREMENT = HardwareMeasurement(t_c_ms=1.0e6, t_i_ms=1.0e-3, t_o_ms=0.0)


def _profile_key(restore_length: int = RESTORE_LENGTH) -> HardwareProfileKey:
    return HardwareProfileKey(
        hardware_tier=HARDWARE_TIER,
        model_fingerprint=MODEL_FINGERPRINT,
        chunk_length_bucket=chunk_length_bucket(restore_length),
    )


class FakeKVCache:
    def __init__(self):
        self.layer_num = LAYER_NUM
        shape = (BUFFER_SLOTS,) + SLOT_SHAPE
        self.k_buffer = [
            torch.arange(torch.tensor(shape).prod()).reshape(shape).float()
            + layer * 100000.0
            for layer in range(self.layer_num)
        ]
        self.v_buffer = [buffer + 1_000_000.0 for buffer in self.k_buffer]

    def move_kv_cache(self, target, source):
        for layer in range(self.layer_num):
            self.k_buffer[layer][target] = self.k_buffer[layer][source]
            self.v_buffer[layer][target] = self.v_buffer[layer][source]

    def get_key_buffer(self, layer_id):
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id):
        return self.v_buffer[layer_id]

    def get_kv_size_bytes(self):
        key_bytes = sum(b.numel() * b.element_size() for b in self.k_buffer)
        value_bytes = sum(b.numel() * b.element_size() for b in self.v_buffer)
        return key_bytes, value_bytes


class FakeAllocator:
    device = "cpu"
    size_full = BUFFER_SLOTS

    def __init__(self, kvcache):
        self.kvcache = kvcache
        self.next_index = 16
        self.freed = []

    def alloc(self, size):
        result = torch.arange(
            self.next_index, self.next_index + size, dtype=torch.int64
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
        self.req_to_token = torch.full((2, BUFFER_SLOTS), -1, dtype=torch.int64)


class FakeReq:
    def __init__(self, metadata, tokens, prefix_len=0, needs_host_load_back=False):
        self.approx_kv_metadata = metadata
        self.req_pool_idx = 0
        self.kv = SimpleNamespace(kv_allocated_len=len(tokens))
        self.full_untruncated_fill_ids = list(tokens)
        self.prefix_indices = torch.arange(prefix_len, dtype=torch.int64)
        self.rid = "req"
        self._needs_host_load_back = needs_host_load_back

    def effective_kv_committed_len(self):
        return len(self.full_untruncated_fill_ids)

    def needs_host_load_back(self):
        return self._needs_host_load_back


class FakeMetricsCollector:
    def __init__(self):
        self.requests: list[tuple[str, str]] = []
        self.fallbacks: list[tuple[str, int]] = []
        self.cachetune_repairs: list[tuple[int, int, bool, str]] = []

    def increment_approx_kv_request(self, operation, outcome):
        self.requests.append((operation, outcome))

    def increment_approx_kv_fallback(self, reason, num_tokens):
        self.fallbacks.append((reason, num_tokens))

    def record_approx_kv_transfer(self, stats):
        del stats

    def increment_approx_kv_h2d_tokens(self, num_tokens):
        del num_tokens

    def increment_approx_kv_host_export(self, num_tokens, num_bytes):
        del num_tokens, num_bytes

    def observe_approx_kv_h2d(self, num_tokens, num_bytes, duration_ms):
        del num_tokens, num_bytes, duration_ms

    def record_approx_kv_cachetune_repair(
        self, selected_tokens, recomputed_layers, precomputed, ratio_source
    ):
        self.cachetune_repairs.append(
            (selected_tokens, recomputed_layers, precomputed, ratio_source)
        )


class HotSpotProbeBackend:
    """Real (fake) shallow-layer probe hook. Returns the model's genuine
    fresh K for every slot except a pre-arranged 'hot' subset of local
    restore positions, whose fresh K is deliberately perturbed to be far
    from what is sitting in the KV buffer -- i.e. these are the only
    tokens with a real, non-fabricated high KV deviation."""

    def __init__(self, kvcache, base_offset, hot_local_positions, delta=500.0):
        self.kvcache = kvcache
        self.base_offset = base_offset
        self.hot_local_positions = set(hot_local_positions)
        self.delta = delta
        self.calls: list[tuple[int, tuple[int, ...]]] = []

    def probe_layer(self, *, layer_id, slot_indices, token_positions):
        del token_positions
        self.calls.append((layer_id, tuple(int(s) for s in slot_indices.tolist())))
        fresh = self.kvcache.get_key_buffer(layer_id)[slot_indices].clone()
        locals_ = (slot_indices - self.base_offset).tolist()
        for i, local in enumerate(locals_):
            if local in self.hot_local_positions:
                fresh[i] = fresh[i] + self.delta
        return fresh


class RecordingRecomputeBackend:
    """Real (fake) per-layer selective-recompute hook. Writes a
    deterministic marker into the KV buffer for every requested slot in
    one batched call per layer, and records every call it receives."""

    def __init__(self, kvcache):
        self.kvcache = kvcache
        self.calls: list[tuple[int, tuple[int, ...]]] = []

    def recompute_layer(self, *, layer_id, slot_indices, token_positions):
        self.calls.append((layer_id, tuple(int(s) for s in slot_indices.tolist())))
        marker = 777_000.0 + layer_id * 10.0
        self.kvcache.k_buffer[layer_id][
            slot_indices
        ] = marker + token_positions.float().unsqueeze(-1).unsqueeze(-1)
        return LayerRecomputeResult(
            layer_id=layer_id,
            recomputed_slot_indices=tuple(int(s) for s in slot_indices.tolist()),
        )


class TestCacheTuneRuntime(unittest.TestCase):
    def setUp(self):
        self.kvcache = FakeKVCache()
        self.allocator = FakeAllocator(self.kvcache)
        self.req_pool = FakeReqToTokenPool()
        self.req_pool.req_to_token[0, :RESTORE_LENGTH] = torch.arange(
            SOURCE_BASE, SOURCE_BASE + RESTORE_LENGTH
        )
        self.metrics = FakeMetricsCollector()
        self.config = ApproxKVFeatureConfig(
            core_enabled=True,
            host_residency_enabled=True,
        )
        self.manager = ApproxKVManager(self.config, metrics_collector=self.metrics)
        self.manager.bind_residency_backend(
            AllocatorCPUResidencyBackend(self.allocator)
        )
        self.tree = SimpleNamespace(
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=self.req_pool,
            approx_kv=self.manager,
        )
        self.segment = ApproxKVRequestSegment(
            content_hash="artifact", target_start=0, length=RESTORE_LENGTH
        )
        self.tokens = tuple(range(1000, 1000 + RESTORE_LENGTH))

        source = FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REGISTER,
                segments=(self.segment,),
                model_fingerprint=MODEL_FINGERPRINT,
                cache_dtype="fp32",
            ),
            self.tokens,
        )
        register_request_segments(self.tree, source)
        self.restore_base = self.allocator.next_index

    def _cachetune_config(
        self,
        *,
        mode: CacheTuneMode = CacheTuneMode.SPEED_ONLY,
        first_recompute_layer: int = 1,
        deployment_measurement=None,
    ) -> CacheTuneConfig:
        return CacheTuneConfig(
            mode=mode,
            hardware_tier=HARDWARE_TIER,
            probe_stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),),
            first_recompute_layer=first_recompute_layer,
            deployment_measurement=deployment_measurement,
        )

    def _reuse_metadata(self):
        return ApproxKVRequestMetadata(
            operation=ApproxKVRequestOperation.REUSE,
            segments=(self.segment,),
            model_fingerprint=MODEL_FINGERPRINT,
            cache_dtype="fp32",
            plugin=CACHETUNE_PLUGIN_NAME,
        )

    def _reuse_req(self, tokens=None):
        tokens = tokens if tokens is not None else self.tokens + (9999,)
        return FakeReq(self._reuse_metadata(), tokens)

    def _register_plugin(
        self,
        *,
        controller: CacheTuneController,
        probe_backend=None,
        recompute_backend=None,
        first_recompute_layer: int = 1,
        mode: CacheTuneMode = CacheTuneMode.SPEED_ONLY,
        deployment_measurement=None,
    ) -> CacheTuneRecoveryPlugin:
        plugin = CacheTuneRecoveryPlugin(
            config=self._cachetune_config(
                mode=mode,
                first_recompute_layer=first_recompute_layer,
                deployment_measurement=deployment_measurement,
            ),
            controller=controller,
            probe_backend=probe_backend,
            recompute_backend=recompute_backend,
        )
        self.manager.register_plugin(plugin)
        return plugin

    # ------------------------------------------------------------------
    # Controller-driven repair with real (fake) probe/recompute backends.
    # ------------------------------------------------------------------
    def test_controller_selected_ratio_drives_recompute_of_exactly_that_many_tokens(
        self,
    ):
        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        controller.record_measurement(_profile_key(), FIVE_PERCENT_MEASUREMENT)
        probe = HotSpotProbeBackend(
            self.kvcache, self.restore_base, HOT_LOCAL_POSITIONS
        )
        recompute = RecordingRecomputeBackend(self.kvcache)
        self._register_plugin(
            controller=controller, probe_backend=probe, recompute_backend=recompute
        )

        source_keys = [
            buffer[SOURCE_BASE : SOURCE_BASE + RESTORE_LENGTH].clone()
            for buffer in self.kvcache.k_buffer
        ]

        reuse = self._reuse_req()
        self.assertTrue(restore_request_prefix_cachetune(self.tree, reuse))

        self.assertEqual(len(reuse.prefix_indices), RESTORE_LENGTH)
        self.assertEqual(reuse.cachetune_candidate_tokens, RESTORE_LENGTH)
        self.assertAlmostEqual(reuse.cachetune_ratio, 0.05)
        self.assertEqual(reuse.cachetune_ratio_source, "roofline")
        self.assertEqual(reuse.cachetune_mode, "speed_only")
        # The number of tokens actually selected/recomputed must equal
        # the controller's decision exactly -- the core guarantee.
        self.assertEqual(reuse.cachetune_selected_tokens, len(HOT_LOCAL_POSITIONS))
        self.assertEqual(reuse.cachetune_recomputed_layers, (1, 2))
        self.assertFalse(reuse.cachetune_precomputed)

        expected_slots = sorted(self.restore_base + p for p in HOT_LOCAL_POSITIONS)
        self.assertEqual([c[0] for c in recompute.calls], [1, 2])
        for layer_id, called_slots in recompute.calls:
            self.assertEqual(sorted(called_slots), expected_slots)

        self.assertEqual(len(probe.calls), 1)
        self.assertEqual(probe.calls[0][0], 0)
        self.assertEqual(len(probe.calls[0][1]), RESTORE_LENGTH)

        hot_slots = {self.restore_base + p for p in HOT_LOCAL_POSITIONS}
        for layer in range(LAYER_NUM):
            for local in range(RESTORE_LENGTH):
                slot = self.restore_base + local
                if slot in hot_slots and layer >= 1:
                    continue
                torch.testing.assert_close(
                    self.kvcache.k_buffer[layer][slot],
                    source_keys[layer][local],
                )

        # Telemetry: the manager's record_cachetune_repair callback must
        # have fired with the exact same numbers.
        self.assertEqual(len(self.metrics.cachetune_repairs), 1)
        selected, layers, precomputed, source = self.metrics.cachetune_repairs[0]
        self.assertEqual(selected, len(HOT_LOCAL_POSITIONS))
        self.assertEqual(layers, 2)
        self.assertFalse(precomputed)
        self.assertEqual(source, "roofline")

    # ------------------------------------------------------------------
    # speed_only mode's 0% floor: no capability required at all.
    # ------------------------------------------------------------------
    def test_zero_repair_tokens_serves_via_baseline_copy_with_no_backends(self):
        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        controller.record_measurement(_profile_key(), ZERO_REPAIR_MEASUREMENT)
        # No probe_backend / recompute_backend bound at all: plugin is
        # not `capable`, yet this must still succeed because the
        # controller decided on 0 repair tokens.
        plugin = self._register_plugin(controller=controller)
        self.assertFalse(plugin.capable)

        reuse = self._reuse_req()
        self.assertTrue(restore_request_prefix_cachetune(self.tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), RESTORE_LENGTH)
        self.assertEqual(reuse.cachetune_selected_tokens, 0)
        self.assertEqual(reuse.cachetune_recomputed_layers, ())
        self.assertFalse(reuse.cachetune_precomputed)
        self.assertIn(("reuse", "success"), self.metrics.requests)

    def test_paper_mechanism_mode_never_selects_zero_repair_tokens(self):
        # Under paper_mechanism the 15% floor forces repair_tokens > 0
        # even for the most extreme compute-bound hardware, so a
        # capability gap *does* block the request.
        controller = CacheTuneController(CacheTuneMode.PAPER_MECHANISM)
        controller.record_measurement(_profile_key(), ZERO_REPAIR_MEASUREMENT)
        self._register_plugin(controller=controller, mode=CacheTuneMode.PAPER_MECHANISM)

        reuse = self._reuse_req()
        self.assertFalse(restore_request_prefix_cachetune(self.tree, reuse))
        self.assertIn(
            ("cachetune_capability_unavailable", RESTORE_LENGTH),
            self.metrics.fallbacks,
        )

    # ------------------------------------------------------------------
    # Honest "no measurement configured" dense-fallback + auto-seeding.
    # ------------------------------------------------------------------
    def test_measurement_unavailable_dense_falls_back_without_allocating(self):
        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        # No measurement recorded, and no deployment_measurement
        # configured either.
        self._register_plugin(controller=controller, deployment_measurement=None)
        next_index = self.allocator.next_index

        reuse = self._reuse_req()
        self.assertFalse(restore_request_prefix_cachetune(self.tree, reuse))
        self.assertEqual(self.allocator.next_index, next_index)
        self.assertEqual(len(reuse.prefix_indices), 0)
        self.assertIn(
            ("cachetune_measurement_unavailable", RESTORE_LENGTH),
            self.metrics.fallbacks,
        )
        self.assertIn(("reuse", "dense_fallback"), self.metrics.requests)
        self.assertFalse(controller.has_measurement(_profile_key()))

    def test_new_profile_key_is_auto_seeded_from_deployment_measurement(self):
        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        self.assertFalse(controller.has_measurement(_profile_key()))
        # The seeded measurement yields a nonzero repair-token count, so
        # real probe/recompute backends must be bound for this request
        # to actually succeed -- this test isolates the auto-seeding
        # behavior itself, not the capability gate (already covered by
        # `test_capability_guard_dense_falls_back_...`).
        probe = HotSpotProbeBackend(
            self.kvcache, self.restore_base, HOT_LOCAL_POSITIONS
        )
        recompute = RecordingRecomputeBackend(self.kvcache)
        self._register_plugin(
            controller=controller,
            probe_backend=probe,
            recompute_backend=recompute,
            deployment_measurement=FIVE_PERCENT_MEASUREMENT,
        )

        reuse = self._reuse_req()
        self.assertTrue(restore_request_prefix_cachetune(self.tree, reuse))
        # The controller must now have learned this exact profile key
        # from the deployment-wide measurement.
        self.assertTrue(controller.has_measurement(_profile_key()))
        self.assertIs(controller.measurement(_profile_key()), FIVE_PERCENT_MEASUREMENT)
        self.assertAlmostEqual(reuse.cachetune_ratio, 0.05)

    # ------------------------------------------------------------------
    # Capability gate / dense-fallback safety, mirroring CacheBlend.
    # ------------------------------------------------------------------
    def test_capability_guard_dense_falls_back_without_probe_or_recompute_backend(
        self,
    ):
        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        controller.record_measurement(_profile_key(), FIVE_PERCENT_MEASUREMENT)
        plugin = self._register_plugin(controller=controller)
        self.assertFalse(plugin.capable)

        next_index = self.allocator.next_index
        reuse = self._reuse_req()
        self.assertFalse(restore_request_prefix_cachetune(self.tree, reuse))
        self.assertEqual(self.allocator.next_index, next_index)
        self.assertEqual(len(reuse.prefix_indices), 0)
        self.assertIn(
            ("cachetune_capability_unavailable", RESTORE_LENGTH),
            self.metrics.fallbacks,
        )
        self.assertIn(("reuse", "dense_fallback"), self.metrics.requests)

    def test_missing_plugin_dense_falls_back_without_exception(self):
        reuse = self._reuse_req()
        self.assertFalse(restore_request_prefix_cachetune(self.tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), 0)
        self.assertIn(("reuse", "dense_fallback"), self.metrics.requests)

    def test_wrong_plugin_type_dense_falls_back_without_exception(self):
        # A plugin object registered under the "cachetune" name that is
        # *not* a CacheTuneRecoveryPlugin instance (e.g. a misconfigured
        # deployment that registered a different plugin under this
        # name) is a genuine server misconfiguration -- but must still
        # dense-fallback rather than raise, since this function runs
        # synchronously on the scheduler's hot path and a single
        # misrouted request must never take the whole scheduler down.
        self.manager.register_plugin(SimpleNamespace(name=CACHETUNE_PLUGIN_NAME))

        next_index = self.allocator.next_index
        reuse = self._reuse_req()
        self.assertFalse(restore_request_prefix_cachetune(self.tree, reuse))
        # The type check happens before any allocation, so nothing was
        # ever allocated and there is nothing to free -- the strongest
        # possible "no slot leak" guarantee for this path.
        self.assertEqual(self.allocator.next_index, next_index)
        self.assertEqual(self.allocator.freed, [])
        self.assertEqual(len(reuse.prefix_indices), 0)
        self.assertIn(
            ("cachetune_plugin_type_mismatch", 0),
            self.metrics.fallbacks,
        )
        self.assertIn(("reuse", "dense_fallback"), self.metrics.requests)

    def test_probe_exception_dense_falls_back_without_exception(self):
        class FailingProbe:
            def probe_layer(self, **kwargs):
                del kwargs
                raise RuntimeError("injected probe failure")

        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        controller.record_measurement(_profile_key(), FIVE_PERCENT_MEASUREMENT)
        self._register_plugin(
            controller=controller,
            probe_backend=FailingProbe(),
            recompute_backend=RecordingRecomputeBackend(self.kvcache),
        )
        reuse = self._reuse_req()
        self.assertFalse(restore_request_prefix_cachetune(self.tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), 0)
        self.assertTrue(self.allocator.freed)
        self.assertIn(
            ("cachetune_token_selection_failed", RESTORE_LENGTH),
            self.metrics.fallbacks,
        )

    def test_backend_execute_exception_dense_falls_back_without_exception(self):
        # `manager.execute` performs the real KV transfer copy for every
        # segment; a failure there (e.g. a residency/copy backend bug)
        # must dense-fallback exactly like a probe/recompute failure --
        # never re-raise into the scheduler's synchronous hot path.
        # Zero repair tokens means no probe/recompute backend is needed:
        # manager.execute is called unconditionally for the baseline
        # copy before any repair-token selection happens.
        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        controller.record_measurement(_profile_key(), ZERO_REPAIR_MEASUREMENT)
        self._register_plugin(controller=controller)

        next_index = self.allocator.next_index
        reuse = self._reuse_req()
        with patch.object(
            self.manager,
            "execute",
            side_effect=RuntimeError("injected execute failure"),
        ):
            self.assertFalse(restore_request_prefix_cachetune(self.tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), 0)
        # No slot leak: `restored_indices` (the destination scratch
        # allocation this function is responsible for) is the *first*
        # `RESTORE_LENGTH`-sized block allocated after `next_index`, and
        # it must be freed exactly once -- no leak (it does appear) and
        # no double-free (it appears exactly once, with no duplicate
        # indices). A second, later allocation may legitimately follow
        # (the source segment's own device-residency load via
        # `ensure_device`, a lasting cache resource independent of this
        # one request -- not this function's responsibility to free).
        self.assertEqual(
            sorted(self.allocator.freed),
            list(range(next_index, next_index + RESTORE_LENGTH)),
        )
        self.assertIn(
            ("cachetune_transfer_execution_failed", RESTORE_LENGTH),
            self.metrics.fallbacks,
        )
        self.assertIn(("reuse", "dense_fallback"), self.metrics.requests)

    def test_token_mismatch_uses_dense_fallback_without_allocating(self):
        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        controller.record_measurement(_profile_key(), FIVE_PERCENT_MEASUREMENT)
        probe = HotSpotProbeBackend(
            self.kvcache, self.restore_base, HOT_LOCAL_POSITIONS
        )
        recompute = RecordingRecomputeBackend(self.kvcache)
        self._register_plugin(
            controller=controller, probe_backend=probe, recompute_backend=recompute
        )

        next_index = self.allocator.next_index
        mismatched_tokens = list(self.tokens)
        mismatched_tokens[5] = 424242
        reuse = self._reuse_req(tokens=tuple(mismatched_tokens) + (9999,))
        self.assertFalse(restore_request_prefix_cachetune(self.tree, reuse))
        self.assertEqual(self.allocator.next_index, next_index)
        self.assertEqual(len(reuse.prefix_indices), 0)
        self.assertEqual(recompute.calls, [])

    def test_no_write_into_exact_radix_tree(self):
        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        controller.record_measurement(_profile_key(), FIVE_PERCENT_MEASUREMENT)
        probe = HotSpotProbeBackend(
            self.kvcache, self.restore_base, HOT_LOCAL_POSITIONS
        )
        recompute = RecordingRecomputeBackend(self.kvcache)
        self._register_plugin(
            controller=controller, probe_backend=probe, recompute_backend=recompute
        )
        self.assertFalse(hasattr(self.tree, "insert"))
        reuse = self._reuse_req()
        self.assertTrue(restore_request_prefix_cachetune(self.tree, reuse))
        self.assertFalse(hasattr(self.tree, "insert"))

    # ------------------------------------------------------------------
    # Precomputed fresh-KV adapter (the CacheBlend-ported mechanism).
    # ------------------------------------------------------------------
    def test_precomputed_fresh_kv_drives_repair_when_not_capable(self):
        raw_segment = ApproxKVRequestSegment(
            content_hash="cachetune-raw:artifact",
            target_start=0,
            length=RESTORE_LENGTH,
        )
        fresh_segment = ApproxKVRequestSegment(
            content_hash="cachetune-fresh:artifact",
            target_start=0,
            length=RESTORE_LENGTH,
        )
        raw_source = FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REGISTER,
                segments=(raw_segment,),
                model_fingerprint=MODEL_FINGERPRINT,
                cache_dtype="fp32",
            ),
            self.tokens,
        )
        register_request_segments(self.tree, raw_source)

        for layer in range(LAYER_NUM):
            for local in HOT_LOCAL_POSITIONS:
                self.kvcache.k_buffer[layer][SOURCE_BASE + local] += 500.0
                self.kvcache.v_buffer[layer][SOURCE_BASE + local] += 700.0
        fresh_source = FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REGISTER,
                segments=(fresh_segment,),
                model_fingerprint=MODEL_FINGERPRINT,
                cache_dtype="fp32",
            ),
            self.tokens,
        )
        register_request_segments(self.tree, fresh_source)

        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        controller.record_measurement(_profile_key(), FIVE_PERCENT_MEASUREMENT)
        # No real probe/recompute backend bound -- plugin.capable is
        # False, but the precomputed cachetune-raw:/cachetune-fresh:
        # segments make repair possible anyway.
        plugin = self._register_plugin(controller=controller)
        self.assertFalse(plugin.capable)

        reuse = FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REUSE,
                segments=(raw_segment,),
                model_fingerprint=MODEL_FINGERPRINT,
                cache_dtype="fp32",
                plugin=CACHETUNE_PLUGIN_NAME,
            ),
            self.tokens + (9999,),
        )
        self.assertTrue(restore_request_prefix_cachetune(self.tree, reuse))
        self.assertTrue(reuse.cachetune_precomputed)
        self.assertEqual(reuse.cachetune_selected_tokens, len(HOT_LOCAL_POSITIONS))

    def test_missing_fresh_counterpart_dense_falls_back_honestly(self):
        # The active segment *is* "cachetune-raw:"-prefixed (so the
        # capability gate's `precomputed_requested` escape hatch is
        # satisfied and the request is allowed past it), but no matching
        # "cachetune-fresh:" segment was ever registered. This must
        # honestly dense-fallback (`cachetune_fresh_store_miss`) rather
        # than crash or silently skip the precomputed repair.
        raw_segment = ApproxKVRequestSegment(
            content_hash="cachetune-raw:artifact",
            target_start=0,
            length=RESTORE_LENGTH,
        )
        raw_source = FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REGISTER,
                segments=(raw_segment,),
                model_fingerprint=MODEL_FINGERPRINT,
                cache_dtype="fp32",
            ),
            self.tokens,
        )
        register_request_segments(self.tree, raw_source)

        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        controller.record_measurement(_profile_key(), FIVE_PERCENT_MEASUREMENT)
        plugin = self._register_plugin(controller=controller)
        self.assertFalse(plugin.capable)

        next_index = self.allocator.next_index
        reuse = FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REUSE,
                segments=(raw_segment,),
                model_fingerprint=MODEL_FINGERPRINT,
                cache_dtype="fp32",
                plugin=CACHETUNE_PLUGIN_NAME,
            ),
            self.tokens + (9999,),
        )
        self.assertFalse(restore_request_prefix_cachetune(self.tree, reuse))
        self.assertEqual(self.allocator.next_index, next_index)
        self.assertIn(
            ("cachetune_fresh_store_miss", RESTORE_LENGTH),
            self.metrics.fallbacks,
        )

    # ------------------------------------------------------------------
    # Defense-in-depth: the explicit runtime-level hard invariant check.
    # ------------------------------------------------------------------
    def test_runtime_hard_invariant_dense_falls_back_without_exception(self):
        # `TokenSelection.__post_init__` already structurally guarantees
        # `len(selected_positions) == requested_count`, making runtime.py's
        # own redundant check unreachable in practice. This test proves
        # the explicit safety net in `runtime.py` really would catch a
        # regression if `select_repair_tokens` ever returned a selection
        # with the wrong count through some future code path, by
        # monkeypatching it to return exactly that broken object. Even
        # though this indicates an internal bug, it must dense-fallback
        # rather than raise: this function runs synchronously on the
        # scheduler's Req.init_next_round_input hot path, so a single
        # request hitting an internal invariant violation must never be
        # allowed to take the whole scheduler down.
        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        controller.record_measurement(_profile_key(), FIVE_PERCENT_MEASUREMENT)
        probe = HotSpotProbeBackend(
            self.kvcache, self.restore_base, HOT_LOCAL_POSITIONS
        )
        recompute = RecordingRecomputeBackend(self.kvcache)
        self._register_plugin(
            controller=controller, probe_backend=probe, recompute_backend=recompute
        )

        broken_selection = SimpleNamespace(selected_positions=(0, 1, 2, 3))
        next_index = self.allocator.next_index
        reuse = self._reuse_req()
        with patch(
            "sglang.srt.mem_cache.cachetune.runtime.select_repair_tokens",
            return_value=broken_selection,
        ):
            self.assertFalse(restore_request_prefix_cachetune(self.tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), 0)
        self.assertEqual(recompute.calls, [])
        # No slot leak: `restored_indices` (the destination scratch
        # allocation this function is responsible for) is the *first*
        # `RESTORE_LENGTH`-sized block allocated after `next_index`, and
        # it must be freed exactly once -- no leak (it does appear) and
        # no double-free (it appears exactly once, with no duplicate
        # indices). A second, later allocation may legitimately follow
        # (the source segment's own device-residency load via
        # `ensure_device`, a lasting cache resource independent of this
        # one request -- not this function's responsibility to free).
        self.assertEqual(
            sorted(self.allocator.freed),
            list(range(next_index, next_index + RESTORE_LENGTH)),
        )
        self.assertIn(
            ("cachetune_repair_count_invariant_violated", RESTORE_LENGTH),
            self.metrics.fallbacks,
        )
        self.assertIn(("reuse", "dense_fallback"), self.metrics.requests)


if __name__ == "__main__":
    unittest.main()
