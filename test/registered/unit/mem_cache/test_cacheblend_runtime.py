from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.approx_kv.config import ApproxKVFeatureConfig
from sglang.srt.mem_cache.approx_kv.manager import ApproxKVManager
from sglang.srt.mem_cache.approx_kv.radix_backend import (
    AllocatorCPUResidencyBackend,
    RoPEConfig,
)
from sglang.srt.mem_cache.approx_kv.request import (
    ApproxKVRequestMetadata,
    ApproxKVRequestOperation,
    ApproxKVRequestSegment,
)
from sglang.srt.mem_cache.approx_kv.runtime import register_request_segments
from sglang.srt.mem_cache.cacheblend.hkvd import GradualFilterStage
from sglang.srt.mem_cache.cacheblend.plugin import (
    CACHEBLEND_PLUGIN_NAME,
    CacheBlendConfig,
    CacheBlendRecoveryPlugin,
)
from sglang.srt.mem_cache.cacheblend.recompute import LayerRecomputeResult
from sglang.srt.mem_cache.cacheblend.runtime import restore_request_prefix_cacheblend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")

LAYER_NUM = 3
SLOT_SHAPE = (2, 4)
BUFFER_SLOTS = 320
SOURCE_BASE = 200
RESTORE_LENGTH = 100
HOT_LOCAL_POSITIONS = (3, 27, 51, 68, 91)  # exactly 5 => 5% of 100 candidates


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


class HotSpotProbeBackend:
    """Real (fake) shallow-layer probe hook. Returns the model's genuine
    fresh K for every slot except a pre-arranged 'hot' subset of local
    restore positions, whose fresh K is deliberately perturbed to be far
    from what is sitting in the KV buffer -- i.e. these are the only
    tokens with a real, non-fabricated high KV deviation. This directly
    proves HKVD scores (not any static/structural proxy) drive selection:
    if selection did not depend on the real scores, it would not reliably
    single out exactly this engineered hot set."""

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
    one batched call per layer, and records every call it receives so
    tests can assert exactly one batched call per layer covering exactly
    the HKVD-selected slots -- never a per-token loop."""

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


class TestCacheBlendRuntime(unittest.TestCase):
    def setUp(self):
        self.kvcache = FakeKVCache()
        self.allocator = FakeAllocator(self.kvcache)
        self.req_pool = FakeReqToTokenPool()
        # Source segment lives at physical slots disjoint from future
        # allocator.alloc() calls (which start at 16) so copies cannot
        # collide with the source's own resident slots.
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
                model_fingerprint="model",
                cache_dtype="fp32",
            ),
            self.tokens,
        )
        register_request_segments(self.tree, source)
        # `restored_indices` will be allocated *after* registration, so
        # its base is exactly the allocator's current next_index.
        self.restore_base = self.allocator.next_index

    def _reuse_config(self, first_recompute_layer=1, ratio=0.05):
        return CacheBlendConfig(
            ratio=ratio,
            probe_stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),),
            first_recompute_layer=first_recompute_layer,
        )

    def _reuse_metadata(self):
        return ApproxKVRequestMetadata(
            operation=ApproxKVRequestOperation.REUSE,
            segments=(self.segment,),
            model_fingerprint="model",
            cache_dtype="fp32",
            plugin=CACHEBLEND_PLUGIN_NAME,
        )

    def _reuse_req(self, tokens=None):
        tokens = tokens if tokens is not None else self.tokens + (9999,)
        return FakeReq(self._reuse_metadata(), tokens)

    def test_hkvd_selection_drives_recompute_and_leaves_last_token_untouched(self):
        probe = HotSpotProbeBackend(
            self.kvcache, self.restore_base, HOT_LOCAL_POSITIONS
        )
        recompute = RecordingRecomputeBackend(self.kvcache)
        plugin = CacheBlendRecoveryPlugin(
            config=self._reuse_config(),
            probe_backend=probe,
            recompute_backend=recompute,
        )
        self.manager.register_plugin(plugin)

        source_keys = [
            buffer[SOURCE_BASE : SOURCE_BASE + RESTORE_LENGTH].clone()
            for buffer in self.kvcache.k_buffer
        ]

        reuse = self._reuse_req()
        self.assertTrue(restore_request_prefix_cacheblend(self.tree, reuse))

        # The final prompt token is never restored -- always left for a
        # real forward pass.
        self.assertEqual(len(reuse.prefix_indices), RESTORE_LENGTH)
        self.assertEqual(reuse.cacheblend_candidate_tokens, RESTORE_LENGTH)
        self.assertAlmostEqual(reuse.cacheblend_ratio, 0.05)
        self.assertEqual(reuse.cacheblend_selected_tokens, len(HOT_LOCAL_POSITIONS))
        self.assertEqual(reuse.cacheblend_recomputed_layers, (1, 2))

        expected_slots = sorted(self.restore_base + p for p in HOT_LOCAL_POSITIONS)
        # Exactly one batched call per recomputed layer, covering exactly
        # the HKVD-selected slots -- proves "selected tokens invoke real
        # recompute hooks per layer" without any per-token loop.
        self.assertEqual([c[0] for c in recompute.calls], [1, 2])
        for layer_id, called_slots in recompute.calls:
            self.assertEqual(sorted(called_slots), expected_slots)

        # The probe backend was consulted (real HKVD measurement, not a
        # fabricated score) for every candidate at the (only) probe layer.
        self.assertEqual(len(probe.calls), 1)
        self.assertEqual(probe.calls[0][0], 0)
        self.assertEqual(len(probe.calls[0][1]), RESTORE_LENGTH)

        hot_slots = {self.restore_base + p for p in HOT_LOCAL_POSITIONS}
        for layer in range(LAYER_NUM):
            for local in range(RESTORE_LENGTH):
                slot = self.restore_base + local
                if slot in hot_slots and layer >= 1:
                    # Selected + recomputed: overwritten by the real
                    # per-layer recompute marker, not the raw copy.
                    continue
                # Everyone else (non-selected at any layer, and selected
                # positions at the probe-only layer 0) must retain the
                # raw copied+RoPE value, byte-for-byte.
                torch.testing.assert_close(
                    self.kvcache.k_buffer[layer][slot],
                    source_keys[layer][local],
                )

    def test_capability_guard_dense_falls_back_without_probe_or_recompute_backend(self):
        plugin = CacheBlendRecoveryPlugin(config=self._reuse_config())
        self.manager.register_plugin(plugin)
        self.assertFalse(plugin.capable)

        next_index = self.allocator.next_index
        reuse = self._reuse_req()
        self.assertFalse(restore_request_prefix_cacheblend(self.tree, reuse))
        self.assertEqual(self.allocator.next_index, next_index)
        self.assertEqual(len(reuse.prefix_indices), 0)
        self.assertIn(
            ("cacheblend_capability_unavailable", RESTORE_LENGTH),
            self.metrics.fallbacks,
        )
        self.assertIn(("reuse", "dense_fallback"), self.metrics.requests)

    def test_missing_plugin_dense_falls_back_without_exception(self):
        reuse = self._reuse_req()
        self.assertFalse(restore_request_prefix_cacheblend(self.tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), 0)
        self.assertIn(("reuse", "dense_fallback"), self.metrics.requests)

    def test_probe_exception_dense_falls_back_without_exception(self):
        class FailingProbe:
            def probe_layer(self, **kwargs):
                del kwargs
                raise RuntimeError("injected probe failure")

        plugin = CacheBlendRecoveryPlugin(
            config=self._reuse_config(),
            probe_backend=FailingProbe(),
            recompute_backend=RecordingRecomputeBackend(self.kvcache),
        )
        self.manager.register_plugin(plugin)
        reuse = self._reuse_req()
        self.assertFalse(restore_request_prefix_cacheblend(self.tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), 0)
        self.assertTrue(self.allocator.freed)

    def test_partial_capability_still_dense_falls_back(self):
        plugin = CacheBlendRecoveryPlugin(
            config=self._reuse_config(),
            probe_backend=HotSpotProbeBackend(self.kvcache, self.restore_base, ()),
        )
        self.manager.register_plugin(plugin)
        self.assertFalse(plugin.capable)

        reuse = self._reuse_req()
        self.assertFalse(restore_request_prefix_cacheblend(self.tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), 0)

    def test_token_mismatch_uses_dense_fallback_without_allocating(self):
        probe = HotSpotProbeBackend(
            self.kvcache, self.restore_base, HOT_LOCAL_POSITIONS
        )
        recompute = RecordingRecomputeBackend(self.kvcache)
        plugin = CacheBlendRecoveryPlugin(
            config=self._reuse_config(),
            probe_backend=probe,
            recompute_backend=recompute,
        )
        self.manager.register_plugin(plugin)

        next_index = self.allocator.next_index
        mismatched_tokens = list(self.tokens)
        mismatched_tokens[5] = 424242
        reuse = self._reuse_req(tokens=tuple(mismatched_tokens) + (9999,))
        self.assertFalse(restore_request_prefix_cacheblend(self.tree, reuse))
        self.assertEqual(self.allocator.next_index, next_index)
        self.assertEqual(len(reuse.prefix_indices), 0)
        self.assertEqual(recompute.calls, [])

    def test_no_write_into_exact_radix_tree(self):
        # A CacheBlend-served request must never be inserted into the
        # exact Radix tree as if it were an exact match: the caller
        # (schedule_batch.Req) forces skip_radix_cache_insert True for
        # any request carrying approx_kv_metadata. This runtime function
        # itself must never touch anything resembling a Radix insertion
        # API -- it only ever appends to req.prefix_indices.
        probe = HotSpotProbeBackend(
            self.kvcache, self.restore_base, HOT_LOCAL_POSITIONS
        )
        recompute = RecordingRecomputeBackend(self.kvcache)
        plugin = CacheBlendRecoveryPlugin(
            config=self._reuse_config(),
            probe_backend=probe,
            recompute_backend=recompute,
        )
        self.manager.register_plugin(plugin)
        self.assertFalse(hasattr(self.tree, "insert"))
        reuse = self._reuse_req()
        self.assertTrue(restore_request_prefix_cacheblend(self.tree, reuse))
        self.assertFalse(hasattr(self.tree, "insert"))

    def test_ratio_sweep_selects_expected_token_counts(self):
        for ratio in (0.01, 0.05, 0.15, 0.30):
            with self.subTest(ratio=ratio):
                kvcache = FakeKVCache()
                allocator = FakeAllocator(kvcache)
                req_pool = FakeReqToTokenPool()
                req_pool.req_to_token[0, :RESTORE_LENGTH] = torch.arange(
                    SOURCE_BASE, SOURCE_BASE + RESTORE_LENGTH
                )
                manager = ApproxKVManager(
                    ApproxKVFeatureConfig(
                        core_enabled=True, host_residency_enabled=True
                    )
                )
                manager.bind_residency_backend(AllocatorCPUResidencyBackend(allocator))
                tree = SimpleNamespace(
                    token_to_kv_pool_allocator=allocator,
                    req_to_token_pool=req_pool,
                    approx_kv=manager,
                )
                segment = ApproxKVRequestSegment(
                    content_hash="artifact", target_start=0, length=RESTORE_LENGTH
                )
                tokens = tuple(range(2000, 2000 + RESTORE_LENGTH))
                source = FakeReq(
                    ApproxKVRequestMetadata(
                        operation=ApproxKVRequestOperation.REGISTER,
                        segments=(segment,),
                        model_fingerprint="model",
                        cache_dtype="fp32",
                    ),
                    tokens,
                )
                register_request_segments(tree, source)
                restore_base = allocator.next_index

                # All 100 candidates get a distinct, monotonically
                # increasing real deviation score -- the funnel/ratio
                # must select exactly round(100 * ratio) of them.
                class RankedProbeBackend:
                    def probe_layer(self, *, layer_id, slot_indices, token_positions):
                        del layer_id, token_positions
                        fresh = kvcache.get_key_buffer(0)[slot_indices].clone()
                        locals_ = (slot_indices - restore_base).float()
                        fresh += locals_.unsqueeze(-1).unsqueeze(-1)
                        return fresh

                recompute = RecordingRecomputeBackend(kvcache)
                plugin = CacheBlendRecoveryPlugin(
                    config=CacheBlendConfig(
                        ratio=ratio,
                        probe_stages=(
                            GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),
                        ),
                        first_recompute_layer=1,
                    ),
                    probe_backend=RankedProbeBackend(),
                    recompute_backend=recompute,
                )
                manager.register_plugin(plugin)

                reuse = FakeReq(
                    ApproxKVRequestMetadata(
                        operation=ApproxKVRequestOperation.REUSE,
                        segments=(segment,),
                        model_fingerprint="model",
                        cache_dtype="fp32",
                        plugin=CACHEBLEND_PLUGIN_NAME,
                    ),
                    tokens + (9999,),
                )
                self.assertTrue(restore_request_prefix_cacheblend(tree, reuse))
                expected_count = max(1, round(RESTORE_LENGTH * ratio))
                self.assertEqual(reuse.cacheblend_selected_tokens, expected_count)
                # Highest-local-index candidates have the highest
                # engineered deviation, so they must be exactly what got
                # selected and recomputed.
                expected_slots = sorted(
                    restore_base + p
                    for p in range(RESTORE_LENGTH - expected_count, RESTORE_LENGTH)
                )
                for layer_id, called_slots in recompute.calls:
                    self.assertEqual(sorted(called_slots), expected_slots)

    def test_precomputed_fresh_kv_drives_hkvd_and_selected_repair(self):
        raw_segment = ApproxKVRequestSegment(
            content_hash="cacheblend-raw:artifact",
            target_start=0,
            length=RESTORE_LENGTH,
        )
        fresh_segment = ApproxKVRequestSegment(
            content_hash="cacheblend-fresh:artifact",
            target_start=0,
            length=RESTORE_LENGTH,
        )
        raw_source = FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REGISTER,
                segments=(raw_segment,),
                model_fingerprint="model",
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
                model_fingerprint="model",
                cache_dtype="fp32",
            ),
            self.tokens,
        )
        register_request_segments(self.tree, fresh_source)

        plugin = CacheBlendRecoveryPlugin(config=self._reuse_config())
        self.manager.register_plugin(plugin)
        reuse = FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REUSE,
                segments=(raw_segment,),
                model_fingerprint="model",
                cache_dtype="fp32",
                plugin=CACHEBLEND_PLUGIN_NAME,
            ),
            self.tokens + (9999,),
        )
        self.assertTrue(restore_request_prefix_cacheblend(self.tree, reuse))
        self.assertTrue(reuse.cacheblend_precomputed)
        self.assertEqual(reuse.cacheblend_selected_tokens, len(HOT_LOCAL_POSITIONS))


class RecordingAsyncResidencyBackend:
    """Combined sync+async residency backend. Exposes `begin_load` (so
    `ApproxKVManager.bind_residency_backend` auto-binds it as the async
    loader) and records the order of `begin_load`/`wait` calls, so a test
    can prove multiple segments' host->device loads are *all* kicked off
    up front (overlapping with each other / with earlier segments'
    compute) rather than one strictly-sequential load-then-compute chain
    per segment."""

    def __init__(self, allocator):
        self._inner = AllocatorCPUResidencyBackend(allocator)
        self.events: list[tuple[str, int]] = []

    def export_to_host(self, device_ref):
        return self._inner.export_to_host(device_ref)

    def load(self, handle, target_tier):
        return self._inner.load(handle, target_tier)

    def begin_load(self, handle, target_tier):
        self.events.append(("begin_load", handle.source_start))
        return _RecordingTransfer(self, handle, target_tier)


class _RecordingTransfer:
    def __init__(self, backend, handle, target_tier):
        self._backend = backend
        self._handle = handle
        self._target_tier = target_tier
        self._done = False

    @property
    def done(self):
        return self._done

    def wait(self, timeout_s=None):
        del timeout_s
        self._backend.events.append(("wait", self._handle.source_start))
        result = self._backend.load(self._handle, self._target_tier)
        self._done = True
        return result

    def cancel(self):
        self._done = True


class TestCacheBlendLoadRecomputeOverlap(unittest.TestCase):
    def test_all_segment_loads_are_kicked_off_before_any_is_waited_on(self):
        kvcache = FakeKVCache()
        allocator = FakeAllocator(kvcache)
        req_pool = FakeReqToTokenPool()
        req_pool.req_to_token[0, :RESTORE_LENGTH] = torch.arange(
            SOURCE_BASE, SOURCE_BASE + RESTORE_LENGTH
        )
        backend = RecordingAsyncResidencyBackend(allocator)
        manager = ApproxKVManager(
            ApproxKVFeatureConfig(
                core_enabled=True,
                host_residency_enabled=True,
                async_prefetch_enabled=True,
            )
        )
        manager.bind_residency_backend(backend)
        tree = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=req_pool,
            approx_kv=manager,
        )
        seg_a = ApproxKVRequestSegment(
            content_hash="artifact-a", target_start=0, length=50
        )
        seg_b = ApproxKVRequestSegment(
            content_hash="artifact-b", target_start=50, length=50
        )
        tokens = tuple(range(3000, 3000 + RESTORE_LENGTH))
        source = FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REGISTER,
                segments=(seg_a, seg_b),
                model_fingerprint="model",
                cache_dtype="fp32",
            ),
            tokens,
        )
        register_request_segments(tree, source)
        restore_base = allocator.next_index

        probe = HotSpotProbeBackend(kvcache, restore_base, (5,))
        recompute = RecordingRecomputeBackend(kvcache)
        plugin = CacheBlendRecoveryPlugin(
            config=CacheBlendConfig(
                ratio=0.01,
                probe_stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),),
                first_recompute_layer=1,
            ),
            probe_backend=probe,
            recompute_backend=recompute,
        )
        manager.register_plugin(plugin)

        reuse = FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REUSE,
                segments=(seg_a, seg_b),
                model_fingerprint="model",
                cache_dtype="fp32",
                plugin=CACHEBLEND_PLUGIN_NAME,
            ),
            tokens + (9999,),
        )
        self.assertTrue(restore_request_prefix_cacheblend(tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), RESTORE_LENGTH)

        # Both segments' loads must be kicked off (begin_load) before
        # either one is waited on -- proving the two host->device
        # transfers overlap rather than running strictly sequentially.
        begin_events = [e for e in backend.events if e[0] == "begin_load"]
        wait_events = [e for e in backend.events if e[0] == "wait"]
        self.assertEqual(len(begin_events), 2)
        self.assertEqual(len(wait_events), 2)
        first_wait_index = backend.events.index(wait_events[0])
        last_begin_index = max(backend.events.index(event) for event in begin_events)
        self.assertLess(
            last_begin_index,
            first_wait_index,
            "all segment loads must be kicked off before any is waited on",
        )


class PressureAllocator(FakeAllocator):
    """A `FakeAllocator` that only starts returning slots once the tree
    cache has actually evicted -- proves a caller went through the real
    `evict -> alloc` sequence rather than allocating directly."""

    def __init__(self, kvcache):
        super().__init__(kvcache)
        self.evicted = False

    def available_size(self):
        return 0 if not self.evicted else BUFFER_SLOTS

    def alloc(self, size):
        if not self.evicted:
            return None
        return super().alloc(size)


class FakeEvictingTree:
    def __init__(self, allocator, req_pool):
        self.token_to_kv_pool_allocator = allocator
        self.req_to_token_pool = req_pool
        self.evict_params = []

    def is_chunk_cache(self):
        return False

    def evict(self, params):
        self.evict_params.append(params)
        self.token_to_kv_pool_allocator.evicted = True


class TestCacheBlendRecoveryAllocationEviction(unittest.TestCase):
    """`restore_request_prefix_cacheblend`'s recovery-buffer allocation
    must go through the shared common-core `allocate_recovery_slots`
    (ported from the R1 EPIC/LegoLink fork), not a bare
    `allocator.alloc`, so it evicts exact Radix victims under pressure --
    and must never leak the buffer it allocates when a later step in the
    same restore attempt fails."""

    def _build_tree(self, allocator, req_pool, *, host_residency_enabled=False):
        manager = ApproxKVManager(
            ApproxKVFeatureConfig(
                core_enabled=True,
                host_residency_enabled=host_residency_enabled,
            )
        )
        tree = FakeEvictingTree(allocator, req_pool)
        tree.approx_kv = manager
        return tree, manager

    def test_evicts_before_allocating_recovery_buffer(self):
        # Device-only registration (`host_residency_enabled=False`) keeps
        # both the raw and fresh segments already DEVICE-resident, so
        # every `ensure_device` call along this path is a no-op and the
        # *only* allocator.alloc call this restore exercises is the
        # recovery-buffer allocation itself.
        kvcache = FakeKVCache()
        allocator = PressureAllocator(kvcache)
        req_pool = FakeReqToTokenPool()
        req_pool.req_to_token[0, :3] = torch.tensor([200, 201, 202])
        tree, manager = self._build_tree(allocator, req_pool)

        raw_segment = ApproxKVRequestSegment(
            content_hash="cacheblend-raw:pressure", target_start=0, length=3
        )
        fresh_segment = ApproxKVRequestSegment(
            content_hash="cacheblend-fresh:pressure", target_start=0, length=3
        )
        tokens = (10, 11, 12, 13)

        # Registration happens while the pool has room.
        allocator.evicted = True
        register_request_segments(
            tree,
            FakeReq(
                ApproxKVRequestMetadata(
                    operation=ApproxKVRequestOperation.REGISTER,
                    segments=(raw_segment,),
                    model_fingerprint="model",
                    cache_dtype="fp32",
                ),
                tokens,
            ),
        )
        register_request_segments(
            tree,
            FakeReq(
                ApproxKVRequestMetadata(
                    operation=ApproxKVRequestOperation.REGISTER,
                    segments=(fresh_segment,),
                    model_fingerprint="model",
                    cache_dtype="fp32",
                ),
                tokens,
            ),
        )

        plugin = CacheBlendRecoveryPlugin(
            config=CacheBlendConfig(
                ratio=0.05,
                probe_stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),),
                first_recompute_layer=1,
            )
        )
        manager.register_plugin(plugin)
        self.assertFalse(plugin.capable)

        # Pool has since filled up -- the target's recovery-buffer
        # allocation must now block until `allocate_recovery_slots`
        # triggers a real eviction.
        allocator.evicted = False
        reuse = FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REUSE,
                segments=(raw_segment,),
                model_fingerprint="model",
                cache_dtype="fp32",
                plugin=CACHEBLEND_PLUGIN_NAME,
            ),
            tokens + (9999,),
        )
        self.assertTrue(restore_request_prefix_cacheblend(tree, reuse))
        self.assertEqual(len(tree.evict_params), 1)
        self.assertEqual(tree.evict_params[0].num_tokens, 3)
        self.assertTrue(reuse.cacheblend_precomputed)

    def test_no_leak_when_rope_config_missing_after_recovery_allocation(self):
        # Register a segment at target_start=0 but reuse it at a
        # *different* position (target_start=2, same token content and
        # content_hash) -- the same non-prefix-context shape every
        # paper-specific pressure runner relies on -- which forces a
        # non-zero `rope_delta`. With no RoPE config bound, the restore
        # must dense-fallback, and the recovery buffer that
        # `allocate_recovery_slots` already handed out must be freed, not
        # leaked, even though the failure is discovered strictly after
        # that allocation succeeded.
        kvcache = FakeKVCache()
        allocator = FakeAllocator(kvcache)
        req_pool = FakeReqToTokenPool()
        req_pool.req_to_token[0, :3] = torch.tensor([200, 201, 202])
        tree, manager = self._build_tree(allocator, req_pool)

        raw_register_segment = ApproxKVRequestSegment(
            content_hash="cacheblend-raw:pressure", target_start=0, length=3
        )
        fresh_register_segment = ApproxKVRequestSegment(
            content_hash="cacheblend-fresh:pressure", target_start=0, length=3
        )
        tokens = (10, 11, 12, 13)
        register_request_segments(
            tree,
            FakeReq(
                ApproxKVRequestMetadata(
                    operation=ApproxKVRequestOperation.REGISTER,
                    segments=(raw_register_segment,),
                    model_fingerprint="model",
                    cache_dtype="fp32",
                ),
                tokens,
            ),
        )
        register_request_segments(
            tree,
            FakeReq(
                ApproxKVRequestMetadata(
                    operation=ApproxKVRequestOperation.REGISTER,
                    segments=(fresh_register_segment,),
                    model_fingerprint="model",
                    cache_dtype="fp32",
                ),
                tokens,
            ),
        )

        plugin = CacheBlendRecoveryPlugin(
            config=CacheBlendConfig(
                ratio=0.05,
                probe_stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),),
                first_recompute_layer=1,
            )
        )
        manager.register_plugin(plugin)

        next_index = allocator.next_index
        reuse_segment = ApproxKVRequestSegment(
            content_hash="cacheblend-raw:pressure", target_start=2, length=3
        )
        reuse = FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REUSE,
                segments=(reuse_segment,),
                model_fingerprint="model",
                cache_dtype="fp32",
                plugin=CACHEBLEND_PLUGIN_NAME,
            ),
            (77, 77, 10, 11, 12, 999),
            prefix_len=2,
        )
        self.assertFalse(restore_request_prefix_cacheblend(tree, reuse))
        self.assertEqual(len(reuse.prefix_indices), 2)
        self.assertGreater(allocator.next_index, next_index)
        newly_allocated = set(range(next_index, allocator.next_index))
        self.assertTrue(newly_allocated <= set(allocator.freed))


class TestCacheBlendMultiSegmentRopeCorrection(unittest.TestCase):
    """Reproduces the real production shape a body > 512 tokens takes:
    the raw and fresh sources are each registered as two independent
    <=512-token chunks (see run_phase4_cacheblend_pressure.py's
    segment_chunks/register_source_segments -- "raw+fresh each 2x512"
    for body=1024), and the *second* chunk is always restored at a
    different absolute position than the position it was registered
    under (registration always uses a per-call-constant target_start =
    len(header); restore uses the cumulative target position) -- so it
    always needs a non-zero rope_delta correction. Without a bound
    RoPEConfig this deterministically dense-falls-back on every such
    multi-chunk restore (never leaking the recovery buffer, just always
    failing); with a real RoPEConfig bound, both chunks actually restore.

    Chunk lengths here (3 + 3 tokens) are scaled down from the real
    512-token contract chunks purely for test speed -- the
    position-mismatch mechanic is identical regardless of chunk size.
    """

    def _build_tree(self, allocator, req_pool, metrics=None):
        manager = ApproxKVManager(
            ApproxKVFeatureConfig(core_enabled=True),
            metrics_collector=metrics,
        )
        tree = FakeEvictingTree(allocator, req_pool)
        tree.approx_kv = manager
        return tree, manager

    def _register_two_chunk_raw_and_fresh(self, tree, chunk0_tokens, chunk1_tokens):
        for hash_prefix in ("cacheblend-raw:", "cacheblend-fresh:"):
            for index, tokens in enumerate((chunk0_tokens, chunk1_tokens)):
                register_request_segments(
                    tree,
                    FakeReq(
                        ApproxKVRequestMetadata(
                            operation=ApproxKVRequestOperation.REGISTER,
                            segments=(
                                ApproxKVRequestSegment(
                                    content_hash=f"{hash_prefix}multi-chunk{index}",
                                    target_start=0,
                                    length=len(tokens),
                                ),
                            ),
                            model_fingerprint="model",
                            cache_dtype="fp32",
                        ),
                        tokens,
                    ),
                )

    def _build_reuse_req(self, chunk0_tokens, chunk1_tokens):
        return FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REUSE,
                segments=(
                    ApproxKVRequestSegment(
                        content_hash="cacheblend-raw:multi-chunk0",
                        target_start=0,
                        length=len(chunk0_tokens),
                    ),
                    ApproxKVRequestSegment(
                        content_hash="cacheblend-raw:multi-chunk1",
                        target_start=len(chunk0_tokens),
                        length=len(chunk1_tokens),
                    ),
                ),
                model_fingerprint="model",
                cache_dtype="fp32",
                plugin=CACHEBLEND_PLUGIN_NAME,
            ),
            chunk0_tokens + chunk1_tokens + (999,),
        )

    def _build_plugin(self):
        return CacheBlendRecoveryPlugin(
            config=CacheBlendConfig(
                ratio=0.05,
                probe_stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),),
                first_recompute_layer=1,
            )
        )

    def test_second_chunk_dense_falls_back_and_frees_once_without_rope_config(self):
        kvcache = FakeKVCache()
        allocator = FakeAllocator(kvcache)
        req_pool = FakeReqToTokenPool()
        req_pool.req_to_token[0, :3] = torch.tensor([200, 201, 202])
        metrics = FakeMetricsCollector()
        tree, manager = self._build_tree(allocator, req_pool, metrics)

        chunk0_tokens = (10, 11, 12)
        chunk1_tokens = (20, 21, 22)
        self._register_two_chunk_raw_and_fresh(tree, chunk0_tokens, chunk1_tokens)
        manager.register_plugin(self._build_plugin())
        self.assertIsNone(manager.rope_config)

        next_index = allocator.next_index
        reuse = self._build_reuse_req(chunk0_tokens, chunk1_tokens)

        result = restore_request_prefix_cacheblend(tree, reuse)

        self.assertFalse(result)
        # The *entire* 6-token recovery buffer (both chunks) is freed
        # exactly once -- not leaked, and not double-freed -- even
        # though only the second chunk's rope check actually failed.
        newly_allocated = set(range(next_index, allocator.next_index))
        self.assertGreater(len(newly_allocated), 0)
        self.assertEqual(sorted(allocator.freed), sorted(newly_allocated))
        self.assertEqual(metrics.fallbacks, [("rope_config_unavailable", 6)])
        self.assertEqual(metrics.requests[-1], ("reuse", "dense_fallback"))

    def test_both_chunks_restore_successfully_with_bound_rope_config(self):
        kvcache = FakeKVCache()
        allocator = FakeAllocator(kvcache)
        req_pool = FakeReqToTokenPool()
        req_pool.req_to_token[0, :3] = torch.tensor([200, 201, 202])
        metrics = FakeMetricsCollector()
        tree, manager = self._build_tree(allocator, req_pool, metrics)

        chunk0_tokens = (10, 11, 12)
        chunk1_tokens = (20, 21, 22)
        self._register_two_chunk_raw_and_fresh(tree, chunk0_tokens, chunk1_tokens)
        manager.register_plugin(self._build_plugin())
        # The default-Qwen RoPE layout resolve_model_rope_config would
        # bind in production: a small, even rotary_dim that fits this
        # FakeKVCache's (2, 4)-shaped slots.
        manager.bind_rope_config(
            RoPEConfig(rotary_dim=4, base=10000.0, is_neox_style=True)
        )

        reuse = self._build_reuse_req(chunk0_tokens, chunk1_tokens)

        result = restore_request_prefix_cacheblend(tree, reuse)

        self.assertTrue(result)
        self.assertEqual(metrics.fallbacks, [])
        self.assertEqual(metrics.requests[-1], ("reuse", "success"))
        self.assertTrue(reuse.cacheblend_precomputed)


class _WrongTypeCacheBlendPlugin:
    """Satisfies the `RecoveryPlugin` protocol shape (`name` /
    `build_plan` / `scheduler_metadata`) well enough for
    `RecoveryPluginRegistry.register` to accept it under the CacheBlend
    plugin name, but is deliberately *not* a `CacheBlendRecoveryPlugin`
    instance -- simulating a scheduler-side registration bug/mismatch
    that `restore_request_prefix_cacheblend` must reject defensively
    instead of trusting."""

    name = CACHEBLEND_PLUGIN_NAME

    def build_plan(self, context, store):
        raise NotImplementedError

    def scheduler_metadata(self, context):
        return ()


class TestCacheBlendSchedulerSafetyFallbacks(unittest.TestCase):
    """`restore_request_prefix_cacheblend` is called directly from
    `Req.init_next_round_input`, a scheduler-critical path with no
    wrapping try/except at the call site. Every failure mode -- including
    a misregistered plugin of the wrong type and a KV-transfer execution
    exception -- must be caught, logged, and turned into an honest
    dense_fallback (`return False`) rather than escaping and killing the
    scheduler process."""

    def _build_tree(self, allocator, req_pool, metrics=None):
        manager = ApproxKVManager(
            ApproxKVFeatureConfig(core_enabled=True),
            metrics_collector=metrics,
        )
        tree = FakeEvictingTree(allocator, req_pool)
        tree.approx_kv = manager
        return tree, manager

    def test_wrong_plugin_type_falls_back_instead_of_raising(self):
        kvcache = FakeKVCache()
        allocator = FakeAllocator(kvcache)
        req_pool = FakeReqToTokenPool()
        metrics = FakeMetricsCollector()
        tree, manager = self._build_tree(allocator, req_pool, metrics)
        manager.register_plugin(_WrongTypeCacheBlendPlugin())

        reuse = FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REUSE,
                segments=(
                    ApproxKVRequestSegment(
                        content_hash="cacheblend-raw:pressure",
                        target_start=0,
                        length=3,
                    ),
                ),
                model_fingerprint="model",
                cache_dtype="fp32",
                plugin=CACHEBLEND_PLUGIN_NAME,
            ),
            (10, 11, 12, 13),
        )

        # Must not raise TypeError (or anything else) out of this call --
        # that would escape init_next_round_input and kill the scheduler.
        result = restore_request_prefix_cacheblend(tree, reuse)

        self.assertFalse(result)
        self.assertEqual(metrics.fallbacks, [("cacheblend_plugin_wrong_type", 0)])
        self.assertEqual(metrics.requests, [("reuse", "dense_fallback")])
        # No recovery buffer was ever allocated for this rejection path,
        # so nothing should have been freed either.
        self.assertEqual(allocator.freed, [])

    def test_transfer_execution_failure_frees_slot_once_and_falls_back(self):
        kvcache = FakeKVCache()
        allocator = FakeAllocator(kvcache)
        req_pool = FakeReqToTokenPool()
        req_pool.req_to_token[0, :3] = torch.tensor([200, 201, 202])
        metrics = FakeMetricsCollector()
        tree, manager = self._build_tree(allocator, req_pool, metrics)

        raw_segment = ApproxKVRequestSegment(
            content_hash="cacheblend-raw:pressure", target_start=0, length=3
        )
        fresh_segment = ApproxKVRequestSegment(
            content_hash="cacheblend-fresh:pressure", target_start=0, length=3
        )
        tokens = (10, 11, 12, 13)
        register_request_segments(
            tree,
            FakeReq(
                ApproxKVRequestMetadata(
                    operation=ApproxKVRequestOperation.REGISTER,
                    segments=(raw_segment,),
                    model_fingerprint="model",
                    cache_dtype="fp32",
                ),
                tokens,
            ),
        )
        register_request_segments(
            tree,
            FakeReq(
                ApproxKVRequestMetadata(
                    operation=ApproxKVRequestOperation.REGISTER,
                    segments=(fresh_segment,),
                    model_fingerprint="model",
                    cache_dtype="fp32",
                ),
                tokens,
            ),
        )

        plugin = CacheBlendRecoveryPlugin(
            config=CacheBlendConfig(
                ratio=0.05,
                probe_stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),),
                first_recompute_layer=1,
            )
        )
        manager.register_plugin(plugin)
        self.assertFalse(plugin.capable)

        def _boom(plan, backend):
            del plan, backend
            raise RuntimeError("simulated CacheBlend transfer execution failure")

        manager.execute = _boom

        next_index = allocator.next_index
        reuse = FakeReq(
            ApproxKVRequestMetadata(
                operation=ApproxKVRequestOperation.REUSE,
                segments=(raw_segment,),
                model_fingerprint="model",
                cache_dtype="fp32",
                plugin=CACHEBLEND_PLUGIN_NAME,
            ),
            tokens + (9999,),
        )

        # Must not raise out of this call -- the exception from
        # `manager.execute` must be caught and turned into a dense
        # fallback instead of escaping init_next_round_input.
        result = restore_request_prefix_cacheblend(tree, reuse)

        self.assertFalse(result)
        self.assertGreater(allocator.next_index, next_index)
        newly_allocated = set(range(next_index, allocator.next_index))
        # The recovery buffer allocated for this restore attempt must be
        # freed exactly once -- no leak, and no double-free of the same
        # indices.
        self.assertEqual(sorted(allocator.freed), sorted(newly_allocated))
        self.assertEqual(
            metrics.fallbacks, [("cacheblend_transfer_execution_failed", 3)]
        )
        self.assertEqual(metrics.requests[-1], ("reuse", "dense_fallback"))


if __name__ == "__main__":
    unittest.main()
