from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.srt.mem_cache.approx_kv.cachecraft_metrics import (
    CacheCraftDecision,
    ChunkContextProfile,
)
from sglang.srt.mem_cache.approx_kv.cachecraft_plugin import (
    CacheCraftPlugin,
    CacheCraftProfileStore,
)
from sglang.srt.mem_cache.approx_kv.cachecraft_runtime import (
    restore_request_via_cachecraft,
)
from sglang.srt.mem_cache.approx_kv.config import ApproxKVFeatureConfig
from sglang.srt.mem_cache.approx_kv.manager import ApproxKVManager
from sglang.srt.mem_cache.approx_kv.radix_backend import DeviceKVRef
from sglang.srt.mem_cache.approx_kv.request import (
    ApproxKVRequestMetadata,
    ApproxKVRequestOperation,
    ApproxKVRequestSegment,
)
from sglang.srt.mem_cache.approx_kv.types import (
    KVLayerTransferResult,
    KVSegmentKey,
    ResidencyTier,
    token_ids_hash,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class FakeKVCache:
    """Same conventions as `test_approx_kv_runtime.FakeKVCache`: a real,
    layered device KV buffer that `move_kv_cache` genuinely mutates."""

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
    size_full = 64

    def __init__(self, kvcache):
        self.kvcache = kvcache
        self.next_index = 16
        self.freed: list[int] = []

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


class FakeReq:
    def __init__(self, metadata, tokens, new_prefix_order=()):
        self.approx_kv_metadata = metadata
        self.full_untruncated_fill_ids = list(tokens)
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.rid = "req"
        self.approx_kv_new_prefix_order = new_prefix_order

    def needs_host_load_back(self):
        return False


@dataclass
class RealMarkerRecomputeHook:
    """A genuine per-token/per-layer recompute hook (mirrors
    `test_cachecraft_recompute.RealMarkerRecomputeHook`): it writes real,
    token/position-derived values, not a no-op, into exactly the requested
    physical KV indices."""

    calls: list[dict] = field(default_factory=list)

    def recompute(self, *, kvcache, target_indices, token_ids, reason):
        self.calls.append(
            {
                "target_indices": target_indices.clone(),
                "token_ids": token_ids,
                "reason": reason,
            }
        )
        for slot, position, token_id in zip(
            target_indices.tolist(), range(len(token_ids)), token_ids
        ):
            for layer in range(kvcache.layer_num):
                marker = float(token_id) * 100.0 + float(position) + layer * 10.0
                kvcache.k_buffer[layer][slot] = torch.full(
                    (kvcache.k_buffer[layer].shape[1:]), marker
                )
                kvcache.v_buffer[layer][slot] = torch.full(
                    (kvcache.v_buffer[layer].shape[1:]), marker + 0.5
                )
        return KVLayerTransferResult(
            copied_k_tokens=len(token_ids),
            rotated_k_tokens=len(token_ids),
            copied_v_tokens=len(token_ids),
        )


CHUNK_TOKENS = (10, 11, 12)


def make_chunk_key() -> KVSegmentKey:
    return KVSegmentKey(
        content_hash="chunk-C",
        token_hash=token_ids_hash(CHUNK_TOKENS),
        token_count=len(CHUNK_TOKENS),
        model_fingerprint="model",
        cache_dtype="fp32",
    )


def make_fixture():
    """Builds a fresh (kvcache, allocator, manager, tree) tuple and
    registers the chunk's source K/V at physical buffer rows [0, 1, 2]."""
    kvcache = FakeKVCache()
    allocator = FakeAllocator(kvcache)
    manager = ApproxKVManager(ApproxKVFeatureConfig(core_enabled=True))
    tree = type(
        "Tree", (), {"token_to_kv_pool_allocator": allocator, "approx_kv": manager}
    )()
    manager.store.register(
        key=make_chunk_key(),
        token_ids=CHUNK_TOKENS,
        source_start=0,
        residency=ResidencyTier.DEVICE,
        backend_ref=DeviceKVRef(indices=torch.tensor([0, 1, 2], dtype=torch.int64)),
    )
    return kvcache, allocator, manager, tree


def make_req(new_prefix_order=()):
    metadata = ApproxKVRequestMetadata(
        operation=ApproxKVRequestOperation.REUSE,
        segments=(
            ApproxKVRequestSegment(content_hash="chunk-C", target_start=0, length=3),
        ),
        model_fingerprint="model",
        cache_dtype="fp32",
        plugin="cachecraft",
    )
    # 4 tokens total; chunk covers [0:3), leaving token index 3 for a
    # real last-token forward (the common-core invariant).
    return FakeReq(metadata, (10, 11, 12, 99), new_prefix_order=new_prefix_order)


class TestCacheCraftRuntimeDirectReuse(unittest.TestCase):
    def test_direct_reuse_extends_prefix_via_real_device_copy(self):
        kvcache, allocator, manager, tree = make_fixture()
        profiles = CacheCraftProfileStore()
        # No recorded old-prefix attention mass at all -> beta defaults to
        # 1.0 (Eq. (6) zero-denominator edge case) -> beta' = 1.0 -> CFO = 0
        # regardless of CCI -> DIRECT_REUSE (Case 1 in Fig. 11).
        profiles.register(
            ChunkContextProfile(
                chunk_id="chunk-C",
                length=3,
                old_prefix_order=(),
                prefix_chunk_lengths={},
                inter_attention_by_layer={},
                intra_attention_by_layer=(2.0,),
                token_inter_scores=(0.0, 0.0, 0.0),
            )
        )
        plugin = CacheCraftPlugin(profiles)
        req = make_req(new_prefix_order=())

        hook = RealMarkerRecomputeHook()
        source_k_before = [buffer[[0, 1, 2]].clone() for buffer in kvcache.k_buffer]
        source_v_before = [buffer[[0, 1, 2]].clone() for buffer in kvcache.v_buffer]

        ok = restore_request_via_cachecraft(
            tree, req, plugin=plugin, recompute_hook=hook
        )

        self.assertTrue(ok)
        self.assertEqual(req.cachecraft_trace.decision, CacheCraftDecision.DIRECT_REUSE)
        self.assertEqual(len(req.prefix_indices), 3)
        # A real device copy happened (no recompute needed at all): the hook
        # must never have been invoked for a pure direct-reuse decision.
        self.assertEqual(hook.calls, [])
        self.assertEqual(req.cachecraft_recompute_invocations, ())
        target = req.prefix_indices
        for layer in range(kvcache.layer_num):
            torch.testing.assert_close(
                kvcache.k_buffer[layer][target], source_k_before[layer]
            )
            torch.testing.assert_close(
                kvcache.v_buffer[layer][target], source_v_before[layer]
            )
        self.assertTrue(req.approx_kv_stats.mechanically_valid)


class TestCacheCraftRuntimePartialRepair(unittest.TestCase):
    def test_partial_repair_invokes_real_hook_for_selected_tokens_and_copies_rest(self):
        kvcache, allocator, manager, tree = make_fixture()
        profiles = CacheCraftProfileStore()
        # a = 0.6/(3*2) = 0.1, b = 9.0/(3*3) = 1.0 -> CCI = sigmoid(0.1)
        # ~= 0.52498. new_prefix_order=() means beta = 0 (no overlap with
        # the recorded old prefix "P") -> beta' = 0 -> CFO = CCI ~= 0.52498,
        # strictly between 0 and the default full_recompute_threshold=1.0
        # -> PARTIAL_REPAIR, selecting N=ceil(0.52498*3)=2 tokens (the two
        # highest external-attention-score positions: 0 and 1).
        profiles.register(
            ChunkContextProfile(
                chunk_id="chunk-C",
                length=3,
                old_prefix_order=("P",),
                prefix_chunk_lengths={"P": 2},
                inter_attention_by_layer={"P": (0.6,)},
                intra_attention_by_layer=(9.0,),
                token_inter_scores=(5.0, 1.0, 1.0),
            )
        )
        plugin = CacheCraftPlugin(profiles)
        req = make_req(new_prefix_order=())
        hook = RealMarkerRecomputeHook()

        source_k_pos2_before = [buffer[2].clone() for buffer in kvcache.k_buffer]
        source_v_pos2_before = [buffer[2].clone() for buffer in kvcache.v_buffer]

        ok = restore_request_via_cachecraft(
            tree, req, plugin=plugin, recompute_hook=hook
        )

        self.assertTrue(ok)
        trace = req.cachecraft_trace
        self.assertEqual(trace.decision, CacheCraftDecision.PARTIAL_REPAIR)
        self.assertAlmostEqual(trace.cfo, 0.5249791875454877, places=6)
        self.assertEqual(trace.recompute_positions, (0, 1))

        # The real hook was genuinely invoked for exactly the two selected
        # positions, with correct absolute physical target indices and the
        # real token ids at those positions.
        self.assertEqual(len(hook.calls), 1)
        call = hook.calls[0]
        self.assertEqual(call["token_ids"], (10, 11))
        target = req.prefix_indices
        self.assertEqual(len(target), 3)
        torch.testing.assert_close(call["target_indices"], target[0:2].to(torch.int64))

        # Positions 0 and 1 hold real recompute markers (not the original
        # source K/V and not each other's constant), position 2 holds a
        # genuine copy of the original source K/V (unmodified content).
        for layer in range(kvcache.layer_num):
            expected_pos0 = torch.full(
                kvcache.k_buffer[layer].shape[1:], 10.0 * 100.0 + 0.0 + layer * 10.0
            )
            expected_pos1 = torch.full(
                kvcache.k_buffer[layer].shape[1:], 11.0 * 100.0 + 1.0 + layer * 10.0
            )
            torch.testing.assert_close(
                kvcache.k_buffer[layer][target[0]], expected_pos0
            )
            torch.testing.assert_close(
                kvcache.k_buffer[layer][target[1]], expected_pos1
            )
            torch.testing.assert_close(
                kvcache.k_buffer[layer][target[2]], source_k_pos2_before[layer]
            )
            torch.testing.assert_close(
                kvcache.v_buffer[layer][target[2]], source_v_pos2_before[layer]
            )

        self.assertEqual(len(req.cachecraft_recompute_invocations), 1)
        self.assertEqual(req.cachecraft_recompute_invocations[0].length, 2)
        self.assertTrue(req.approx_kv_stats.mechanically_valid)


class TestCacheCraftRuntimeCapabilityGate(unittest.TestCase):
    def test_missing_recompute_hook_dense_falls_back_and_frees_allocation(self):
        # Same PARTIAL_REPAIR-triggering profile as above, but with no real
        # recompute hook available (the documented production blocker):
        # the runtime must safely dense-fallback (return False) rather than
        # silently skip the recompute or corrupt the KV cache.
        kvcache, allocator, manager, tree = make_fixture()
        profiles = CacheCraftProfileStore()
        profiles.register(
            ChunkContextProfile(
                chunk_id="chunk-C",
                length=3,
                old_prefix_order=("P",),
                prefix_chunk_lengths={"P": 2},
                inter_attention_by_layer={"P": (0.6,)},
                intra_attention_by_layer=(9.0,),
                token_inter_scores=(5.0, 1.0, 1.0),
            )
        )
        plugin = CacheCraftPlugin(profiles)
        req = make_req(new_prefix_order=())
        next_index_before = allocator.next_index

        ok = restore_request_via_cachecraft(
            tree, req, plugin=plugin, recompute_hook=None
        )

        self.assertFalse(ok)
        self.assertEqual(len(req.prefix_indices), 0)
        # The allocated device slots must be released back, not leaked.
        self.assertEqual(allocator.freed, list(range(16, 16 + 3)))
        self.assertEqual(allocator.next_index, next_index_before + 3)


class TestCacheCraftRuntimeFullRecompute(unittest.TestCase):
    def test_store_miss_forces_full_recompute_and_dense_fallback(self):
        # A profile is registered, but the store handle for this chunk is
        # not (simulating an evicted/never-cached chunk) -> FULL_RECOMPUTE
        # -> the runtime must dense-fallback without touching the store.
        kvcache = FakeKVCache()
        allocator = FakeAllocator(kvcache)
        manager = ApproxKVManager(ApproxKVFeatureConfig(core_enabled=True))
        tree = type(
            "Tree", (), {"token_to_kv_pool_allocator": allocator, "approx_kv": manager}
        )()
        profiles = CacheCraftProfileStore()
        profiles.register(
            ChunkContextProfile(
                chunk_id="chunk-C",
                length=3,
                old_prefix_order=(),
                prefix_chunk_lengths={},
                inter_attention_by_layer={},
                intra_attention_by_layer=(1.0,),
                token_inter_scores=(0.0, 0.0, 0.0),
            )
        )
        plugin = CacheCraftPlugin(profiles)
        metadata = ApproxKVRequestMetadata(
            operation=ApproxKVRequestOperation.REUSE,
            segments=(
                ApproxKVRequestSegment(
                    content_hash="chunk-C", target_start=0, length=3
                ),
            ),
            model_fingerprint="model",
            cache_dtype="fp32",
            plugin="cachecraft",
        )
        req = FakeReq(metadata, (10, 11, 12, 99))

        ok = restore_request_via_cachecraft(
            tree, req, plugin=plugin, recompute_hook=RealMarkerRecomputeHook()
        )
        self.assertFalse(ok)
        self.assertEqual(len(req.prefix_indices), 0)
        self.assertEqual(allocator.next_index, 16)  # no allocation attempted


# ---------------------------------------------------------------------------
# High-pressure recovery allocation for the partial-repair path: proves
# ``restore_request_via_cachecraft`` allocates its target chunk buffer via
# the shared ``allocate_recovery_slots`` helper, which evicts exact Radix
# victims *before* allocating -- matching SGLang's standard
# ``evict_from_tree_cache -> allocator.alloc`` ordering -- instead of the
# prior direct ``allocator.alloc`` call that would fail under real device
# KV pressure even when eviction could free enough slots.
# ---------------------------------------------------------------------------


class PressureAllocator(FakeAllocator):
    def __init__(self, kvcache):
        super().__init__(kvcache)
        self.pressured = False
        self.evicted = False

    def available_size(self):
        return 0 if (self.pressured and not self.evicted) else self.size_full

    def alloc(self, size):
        if self.pressured and not self.evicted:
            return None
        return super().alloc(size)


class PressureTree:
    def __init__(self, allocator, manager):
        self.token_to_kv_pool_allocator = allocator
        self.approx_kv = manager
        self.evict_params = []

    def is_chunk_cache(self):
        return False

    def evict(self, params):
        self.evict_params.append(params)
        self.token_to_kv_pool_allocator.evicted = True


def make_pressure_fixture():
    kvcache = FakeKVCache()
    allocator = PressureAllocator(kvcache)
    manager = ApproxKVManager(ApproxKVFeatureConfig(core_enabled=True))
    tree = PressureTree(allocator, manager)
    manager.store.register(
        key=make_chunk_key(),
        token_ids=CHUNK_TOKENS,
        source_start=0,
        residency=ResidencyTier.DEVICE,
        backend_ref=DeviceKVRef(indices=torch.tensor([0, 1, 2], dtype=torch.int64)),
    )
    return kvcache, allocator, manager, tree


class TestCacheCraftRuntimeHighPressureAllocation(unittest.TestCase):
    def test_partial_repair_succeeds_under_pressure_via_eviction(self):
        kvcache, allocator, manager, tree = make_pressure_fixture()
        profiles = CacheCraftProfileStore()
        # Same PARTIAL_REPAIR-triggering profile as
        # TestCacheCraftRuntimePartialRepair (CFO ~= 0.52498, selecting
        # positions 0 and 1).
        profiles.register(
            ChunkContextProfile(
                chunk_id="chunk-C",
                length=3,
                old_prefix_order=("P",),
                prefix_chunk_lengths={"P": 2},
                inter_attention_by_layer={"P": (0.6,)},
                intra_attention_by_layer=(9.0,),
                token_inter_scores=(5.0, 1.0, 1.0),
            )
        )
        plugin = CacheCraftPlugin(profiles)
        req = make_req(new_prefix_order=())
        hook = RealMarkerRecomputeHook()

        # Simulate real device KV pressure only for the chunk target-buffer
        # allocation itself; registration/store setup already happened.
        allocator.pressured = True

        ok = restore_request_via_cachecraft(
            tree, req, plugin=plugin, recompute_hook=hook
        )

        self.assertTrue(ok)
        self.assertEqual(len(tree.evict_params), 1)
        self.assertEqual(tree.evict_params[0].num_tokens, 3)
        trace = req.cachecraft_trace
        self.assertEqual(trace.decision, CacheCraftDecision.PARTIAL_REPAIR)
        self.assertEqual(len(hook.calls), 1)
        self.assertEqual(len(req.prefix_indices), 3)
        self.assertTrue(req.approx_kv_stats.mechanically_valid)

    def test_partial_repair_dense_falls_back_without_leak_when_oom_persists(self):
        kvcache, allocator, manager, tree = make_pressure_fixture()
        profiles = CacheCraftProfileStore()
        profiles.register(
            ChunkContextProfile(
                chunk_id="chunk-C",
                length=3,
                old_prefix_order=("P",),
                prefix_chunk_lengths={"P": 2},
                inter_attention_by_layer={"P": (0.6,)},
                intra_attention_by_layer=(9.0,),
                token_inter_scores=(5.0, 1.0, 1.0),
            )
        )
        plugin = CacheCraftPlugin(profiles)
        req = make_req(new_prefix_order=())
        hook = RealMarkerRecomputeHook()

        # Eviction runs but genuinely cannot free enough slots (real OOM):
        # ``evict`` never flips ``evicted`` to True, so allocation keeps
        # failing even after the eviction attempt.
        def evict_without_freeing(params):
            tree.evict_params.append(params)

        tree.evict = evict_without_freeing
        allocator.pressured = True
        next_index = allocator.next_index

        ok = restore_request_via_cachecraft(
            tree, req, plugin=plugin, recompute_hook=hook
        )

        self.assertFalse(ok)
        self.assertEqual(len(tree.evict_params), 1)
        self.assertEqual(allocator.next_index, next_index)
        self.assertEqual(allocator.freed, [])
        self.assertEqual(hook.calls, [])
        self.assertEqual(len(req.prefix_indices), 0)


if __name__ == "__main__":
    unittest.main()
