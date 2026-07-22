from __future__ import annotations

import unittest

from sglang.srt.mem_cache.approx_kv.cachecraft_metrics import (
    CacheCraftDecision,
    ChunkContextProfile,
)
from sglang.srt.mem_cache.approx_kv.cachecraft_plugin import (
    CacheCraftDecisionTrace,
    CacheCraftPlugin,
    CacheCraftProfileStore,
)
from sglang.srt.mem_cache.approx_kv.plugins import RecoveryRequestContext
from sglang.srt.mem_cache.approx_kv.store import ApproxKVSegmentStore
from sglang.srt.mem_cache.approx_kv.types import (
    KVSegmentKey,
    RecoveryMode,
    ResidencyTier,
    token_ids_hash,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")

CHUNK_TOKENS = (11, 12, 13, 14)


def make_key(chunk_id: str) -> KVSegmentKey:
    return KVSegmentKey(
        content_hash=f"content-{chunk_id}",
        token_hash=token_ids_hash(CHUNK_TOKENS),
        token_count=len(CHUNK_TOKENS),
        model_fingerprint="fp-test",
        cache_dtype="fp16",
    )


def register_chunk(store: ApproxKVSegmentStore, chunk_id: str):
    return store.register(
        key=make_key(chunk_id),
        token_ids=CHUNK_TOKENS,
        source_start=0,
        residency=ResidencyTier.DEVICE,
        backend_ref=object(),
    )


def make_context(
    chunk_id: str, new_prefix_order: tuple[str, ...]
) -> RecoveryRequestContext:
    return RecoveryRequestContext(
        request_id="req-1",
        target_token_ids=CHUNK_TOKENS,
        exact_prefix_length=0,
        custom_metadata={
            "chunk_id": chunk_id,
            "chunk_key": make_key(chunk_id),
            "chunk_start": 0,
            "chunk_length": len(CHUNK_TOKENS),
            "new_prefix_order": new_prefix_order,
        },
    )


class TestCacheCraftPluginStoreMiss(unittest.TestCase):
    def test_missing_profile_forces_full_recompute(self):
        store = ApproxKVSegmentStore()
        register_chunk(store, "C")
        profiles = CacheCraftProfileStore()  # no profile registered
        plugin = CacheCraftPlugin(profiles)
        plan = plugin.build_plan(make_context("C", ()), store)
        self.assertEqual(plan.recovery_mode, RecoveryMode.DENSE)
        self.assertEqual(len(plan.dense_ranges), 1)
        self.assertEqual(plan.dense_ranges[0].reason, "cachecraft_no_profile_or_handle")
        self.assertEqual(plugin.last_trace.decision, CacheCraftDecision.FULL_RECOMPUTE)

    def test_missing_handle_forces_full_recompute(self):
        store = ApproxKVSegmentStore()  # nothing registered
        profiles = CacheCraftProfileStore()
        profiles.register(
            ChunkContextProfile(
                chunk_id="C",
                length=4,
                old_prefix_order=(),
                prefix_chunk_lengths={},
                inter_attention_by_layer={},
                intra_attention_by_layer=(1.0,),
                token_inter_scores=(0.0, 0.0, 0.0, 0.0),
            )
        )
        plugin = CacheCraftPlugin(profiles)
        plan = plugin.build_plan(make_context("C", ()), store)
        self.assertEqual(plan.recovery_mode, RecoveryMode.DENSE)
        self.assertEqual(plugin.last_trace.decision, CacheCraftDecision.FULL_RECOMPUTE)


class TestCacheCraftPluginDirectReuse(unittest.TestCase):
    def test_stale_profile_generation_forces_full_recompute(self):
        store = ApproxKVSegmentStore()
        first = register_chunk(store, "C")
        profiles = CacheCraftProfileStore()
        profiles.register(
            ChunkContextProfile(
                chunk_id="C",
                length=4,
                old_prefix_order=("A",),
                prefix_chunk_lengths={"A": 3},
                inter_attention_by_layer={"A": (1.0,)},
                intra_attention_by_layer=(0.1,),
                token_inter_scores=(1.0, 1.0, 1.0, 1.0),
            ),
            generation=first.generation,
        )
        replacement = register_chunk(store, "C")
        self.assertGreater(replacement.generation, first.generation)
        plugin = CacheCraftPlugin(profiles)
        plan = plugin.build_plan(make_context("C", ("A",)), store)
        self.assertEqual(plan.recovery_mode, RecoveryMode.DENSE)
        self.assertEqual(
            plugin.last_trace.decision,
            CacheCraftDecision.FULL_RECOMPUTE,
        )

    def test_same_prefix_same_order_is_direct_reuse(self):
        store = ApproxKVSegmentStore()
        handle = register_chunk(store, "C")
        profiles = CacheCraftProfileStore()
        profiles.register(
            ChunkContextProfile(
                chunk_id="C",
                length=4,
                old_prefix_order=("A", "B"),
                prefix_chunk_lengths={"A": 3, "B": 3},
                inter_attention_by_layer={"A": (2.0,), "B": (2.0,)},
                intra_attention_by_layer=(0.1,),
                token_inter_scores=(1.0, 1.0, 1.0, 1.0),
            )
        )
        plugin = CacheCraftPlugin(profiles)
        plan = plugin.build_plan(make_context("C", ("A", "B")), store)

        trace = plugin.last_trace
        self.assertIsInstance(trace, CacheCraftDecisionTrace)
        self.assertAlmostEqual(trace.beta, 1.0, places=6)
        self.assertAlmostEqual(trace.gamma, 0.0, places=6)
        self.assertAlmostEqual(trace.cfo, 0.0, places=6)
        self.assertEqual(trace.decision, CacheCraftDecision.DIRECT_REUSE)

        self.assertEqual(plan.recovery_mode, RecoveryMode.COPY)
        self.assertEqual(len(plan.copied_spans), 1)
        self.assertEqual(plan.dense_ranges, ())
        span = plan.copied_spans[0]
        self.assertEqual(span.length, 4)
        self.assertEqual(span.source, handle)


class TestCacheCraftPluginFullRecomputeFromCFO(unittest.TestCase):
    def test_no_prefix_overlap_and_high_cci_triggers_full_recompute(self):
        store = ApproxKVSegmentStore()
        register_chunk(store, "C")
        profiles = CacheCraftProfileStore()
        profiles.register(
            ChunkContextProfile(
                chunk_id="C",
                length=4,
                old_prefix_order=("A",),
                prefix_chunk_lengths={"A": 3},
                # Heavy external attention (a is large) vs tiny self
                # attention (b is small) -> CCI close to 1.
                inter_attention_by_layer={"A": (24.0,)},
                intra_attention_by_layer=(0.01,),
                token_inter_scores=(6.0, 6.0, 6.0, 6.0),
            )
        )
        # A low full_recompute_threshold makes a high-but-not-exactly-1 CFO
        # cross into FULL_RECOMPUTE, mirroring a deployment tuned to avoid
        # partial repair once the fix overhead approaches the chunk size.
        plugin = CacheCraftPlugin(profiles, full_recompute_threshold=0.5)
        # New prefix has zero overlap with the chunk's old prefix ("A" is
        # gone) -> beta = 0 -> beta' = 0 -> cfo = cci.
        plan = plugin.build_plan(make_context("C", ("Z",)), store)

        trace = plugin.last_trace
        self.assertAlmostEqual(trace.beta, 0.0, places=6)
        self.assertGreaterEqual(trace.cfo, 0.5)
        self.assertEqual(trace.decision, CacheCraftDecision.FULL_RECOMPUTE)
        self.assertEqual(plan.recovery_mode, RecoveryMode.DENSE)
        self.assertEqual(len(plan.dense_ranges), 1)
        self.assertEqual(plan.dense_ranges[0].reason, "cachecraft_cfo_full")
        self.assertEqual(plan.dense_ranges[0].length, 4)
        self.assertEqual(plan.copied_spans, ())


class TestCacheCraftPluginPartialRepair(unittest.TestCase):
    def test_partial_overlap_selects_a_strict_subset_for_recompute(self):
        store = ApproxKVSegmentStore()
        handle = register_chunk(store, "C")
        profiles = CacheCraftProfileStore()
        profiles.register(
            ChunkContextProfile(
                chunk_id="C",
                length=4,
                old_prefix_order=("A", "B"),
                prefix_chunk_lengths={"A": 3, "B": 3},
                inter_attention_by_layer={"A": (3.0,), "B": (3.0,)},
                intra_attention_by_layer=(1.0,),
                # Token 3 has by far the highest external attention score,
                # so with a partial CFO it must be the (or among the)
                # selected recompute position(s).
                token_inter_scores=(0.5, 0.5, 0.5, 5.0),
            )
        )
        plugin = CacheCraftPlugin(profiles, full_recompute_threshold=1.0)
        # Keep "A" in the new prefix but drop "B": partial overlap, and
        # (with only one common chunk) order penalty gamma = 0 by definition.
        plan = plugin.build_plan(make_context("C", ("A",)), store)

        trace = plugin.last_trace
        self.assertGreater(trace.beta, 0.0)
        self.assertLess(trace.beta, 1.0)
        self.assertEqual(trace.decision, CacheCraftDecision.PARTIAL_REPAIR)
        self.assertGreater(trace.cfo, 0.0)
        self.assertLess(trace.cfo, 1.0)

        self.assertEqual(plan.recovery_mode, RecoveryMode.COPY)
        self.assertGreater(len(plan.dense_ranges), 0)
        self.assertGreater(len(plan.copied_spans), 0)
        recompute_positions = set(trace.recompute_positions)
        self.assertIn(3, recompute_positions)  # highest-score token selected
        self.assertLess(len(recompute_positions), 4)  # strict subset

        dense_covered = sum(dr.length for dr in plan.dense_ranges)
        copy_covered = sum(span.length for span in plan.copied_spans)
        self.assertEqual(dense_covered, len(recompute_positions))
        self.assertEqual(copy_covered, 4 - len(recompute_positions))
        for span in plan.copied_spans:
            self.assertEqual(span.source, handle)

    def test_order_change_alone_can_flip_the_decision(self):
        # Same profile, same prefix-chunk *membership* (both "A" and "B"
        # are common to old/new, so beta is unaffected); only the *order*
        # of those shared prefix chunks differs between old and new. That
        # alone drives gamma from 0 to 1, beta' from 1.0 to 0.0, and CFO
        # from 0 to ~cci -- flipping the decision from DIRECT_REUSE to
        # FULL_RECOMPUTE with identical attention statistics.
        store = ApproxKVSegmentStore()
        register_chunk(store, "C")
        profiles = CacheCraftProfileStore()
        profiles.register(
            ChunkContextProfile(
                chunk_id="C",
                length=4,
                old_prefix_order=("A", "B"),
                prefix_chunk_lengths={"A": 3, "B": 3},
                inter_attention_by_layer={"A": (3.0,), "B": (3.0,)},
                intra_attention_by_layer=(1.0,),
                token_inter_scores=(0.5, 0.5, 0.5, 5.0),
            )
        )
        plugin_same_order = CacheCraftPlugin(profiles, full_recompute_threshold=0.42)
        plan_same_order = plugin_same_order.build_plan(
            make_context("C", ("A", "B")), store
        )
        trace_same_order = plugin_same_order.last_trace

        plugin_reversed = CacheCraftPlugin(profiles, full_recompute_threshold=0.42)
        plan_reversed = plugin_reversed.build_plan(make_context("C", ("B", "A")), store)
        trace_reversed = plugin_reversed.last_trace

        self.assertAlmostEqual(trace_same_order.beta, trace_reversed.beta, places=6)
        self.assertAlmostEqual(trace_same_order.beta, 1.0, places=6)
        self.assertAlmostEqual(trace_same_order.gamma, 0.0, places=6)
        self.assertAlmostEqual(trace_reversed.gamma, 1.0, places=6)
        self.assertLess(trace_same_order.gamma, trace_reversed.gamma)
        self.assertLess(trace_same_order.cfo, trace_reversed.cfo)
        self.assertEqual(trace_same_order.decision, CacheCraftDecision.DIRECT_REUSE)
        self.assertEqual(trace_reversed.decision, CacheCraftDecision.FULL_RECOMPUTE)
        self.assertNotEqual(trace_same_order.decision, trace_reversed.decision)
        self.assertEqual(plan_same_order.recovery_mode, RecoveryMode.COPY)
        self.assertEqual(plan_reversed.recovery_mode, RecoveryMode.DENSE)


if __name__ == "__main__":
    unittest.main()
