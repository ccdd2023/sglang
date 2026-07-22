from __future__ import annotations

import importlib
import sys
import types as python_types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_DIR = REPO_ROOT / "python/sglang/srt/mem_cache/approx_kv"
PACKAGE_NAME = "approx_kv_recovery_under_test"

package = python_types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_DIR)]
sys.modules[PACKAGE_NAME] = package

config_module = importlib.import_module(f"{PACKAGE_NAME}.config")
manager_module = importlib.import_module(f"{PACKAGE_NAME}.manager")
recovery_module = importlib.import_module(f"{PACKAGE_NAME}.recovery")
types_module = importlib.import_module(f"{PACKAGE_NAME}.types")

ApproxKVFeatureConfig = config_module.ApproxKVFeatureConfig
ApproxKVManager = manager_module.ApproxKVManager
ReusableSegment = recovery_module.ReusableSegment
ResidencyTier = types_module.ResidencyTier
SegmentKind = types_module.SegmentKind
KVSegmentKey = types_module.KVSegmentKey
RecoveryMode = types_module.RecoveryMode
token_ids_hash = types_module.token_ids_hash


def make_key(
    tokens,
    name="segment",
    kind=SegmentKind.CANONICAL_BASE,
):
    return KVSegmentKey(
        content_hash=name,
        token_hash=token_ids_hash(tokens),
        token_count=len(tokens),
        model_id="test-model",
        cache_dtype="bf16",
        kind=kind,
    )


def make_manager():
    return ApproxKVManager(
        ApproxKVFeatureConfig(
            core_enabled=True,
            lossy_recovery_enabled=True,
        )
    )


class RecordingAnchorBackend:
    def __init__(self):
        self.dense = []
        self.reconstructions = []

    def dense_prefill(self, *, target_start, length, reason):
        self.dense.append((target_start, length, reason))

    def copy_and_rotate(self, **kwargs):
        raise AssertionError("anchor test must not use raw copy")

    def reconstruct_and_rotate(
        self,
        *,
        base_ref,
        anchor_refs,
        weights,
        source_offset,
        target_start,
        length,
        rope_delta,
    ):
        self.reconstructions.append(
            (
                base_ref,
                anchor_refs,
                weights,
                source_offset,
                target_start,
                length,
                rope_delta,
            )
        )
        return length, length, length


class TestApproxKVRecovery(unittest.TestCase):
    def test_raw_rope_copies_complete_segment(self):
        tokens = tuple(range(6))
        manager = make_manager()
        source = manager.register_segment(
            key=make_key(tokens),
            token_ids=tokens,
            source_start=3,
            residency=ResidencyTier.DEVICE,
            backend_ref="kv",
        )
        plan = recovery_module.build_raw_rope_plan(
            target_token_ids=(90, 91) + tokens + (92,),
            segments=(
                ReusableSegment(
                    segment_id="code",
                    target_start=2,
                    token_ids=tokens,
                    source=source,
                ),
            ),
        )
        self.assertEqual(plan.recovery_mode, RecoveryMode.RAW_ROPE)
        self.assertEqual(
            [(item.target_start, item.length) for item in plan.dense_ranges],
            [(0, 2), (8, 1)],
        )
        self.assertEqual(len(plan.copied_spans), 1)
        span = plan.copied_spans[0]
        self.assertEqual(span.source_offset, 0)
        self.assertEqual(span.target_start, 2)
        self.assertEqual(span.length, 6)
        self.assertEqual(span.rope_delta, -1)

    def test_epic_fixed_k_repairs_leading_tokens(self):
        tokens = tuple(range(8))
        manager = make_manager()
        source = manager.register_segment(
            key=make_key(tokens),
            token_ids=tokens,
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="kv",
        )
        plan = recovery_module.build_epic_fixed_k_plan(
            target_token_ids=tokens,
            segments=(ReusableSegment("code", 0, tokens, source),),
            repair_tokens=2,
        )
        self.assertEqual(
            plan.recovery_mode,
            RecoveryMode.EPIC_FIXED_K,
        )
        self.assertEqual(
            [(item.target_start, item.length) for item in plan.dense_ranges],
            [(0, 2)],
        )
        self.assertEqual(
            (
                plan.copied_spans[0].source_offset,
                plan.copied_spans[0].target_start,
                plan.copied_spans[0].length,
            ),
            (2, 2, 6),
        )

    def test_selective_repair_partitions_dense_and_copy_ranges(self):
        tokens = tuple(range(10))
        manager = make_manager()
        source = manager.register_segment(
            key=make_key(tokens),
            token_ids=tokens,
            source_start=4,
            residency=ResidencyTier.DEVICE,
            backend_ref="kv",
        )
        plan = recovery_module.build_selective_repair_plan(
            target_token_ids=tokens,
            segments=(ReusableSegment("code", 0, tokens, source),),
            repair_offsets={"code": (1, 2, 7)},
        )
        self.assertEqual(
            [(item.target_start, item.length) for item in plan.dense_ranges],
            [(1, 2), (7, 1)],
        )
        self.assertEqual(
            [
                (item.source_offset, item.target_start, item.length)
                for item in plan.copied_spans
            ],
            [(0, 0, 1), (3, 3, 4), (8, 8, 2)],
        )
        self.assertTrue(all(item.rope_delta == -4 for item in plan.copied_spans))

    def test_fraction_and_score_selection(self):
        self.assertEqual(
            recovery_module.repair_offsets_from_fraction(10, 0.15),
            (0, 1),
        )
        self.assertEqual(
            recovery_module.repair_offsets_from_scores(
                (0.1, 0.9, 0.8, 0.2),
                0.5,
            ),
            (1, 2),
        )

    def test_missing_or_mismatched_source_falls_back_dense(self):
        tokens = tuple(range(4))
        missing_plan = recovery_module.build_raw_rope_plan(
            target_token_ids=tokens,
            segments=(ReusableSegment("code", 0, tokens, None),),
        )
        self.assertEqual(
            [(item.length, item.reason) for item in missing_plan.dense_ranges],
            [(4, "missing_source")],
        )
        self.assertEqual(missing_plan.copied_spans, ())

        manager = make_manager()
        source = manager.register_segment(
            key=make_key((9, 8, 7, 6)),
            token_ids=(9, 8, 7, 6),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="kv",
        )
        mismatch_plan = recovery_module.build_raw_rope_plan(
            target_token_ids=tokens,
            segments=(ReusableSegment("code", 0, tokens, source),),
        )
        self.assertEqual(
            [(item.length, item.reason) for item in mismatch_plan.dense_ranges],
            [(4, "source_token_mismatch")],
        )

    def test_kvcomm_anchor_matching_interpolation_and_execution(self):
        tokens = tuple(range(4))
        manager = make_manager()
        base = manager.register_segment(
            key=make_key(tokens, "base"),
            token_ids=tokens,
            source_start=2,
            residency=ResidencyTier.DEVICE,
            backend_ref="base-kv",
        )
        near = manager.register_segment(
            key=make_key(
                tokens,
                "near",
                SegmentKind.CONTEXT_ANCHOR,
            ),
            token_ids=tokens,
            source_start=2,
            residency=ResidencyTier.DEVICE,
            backend_ref="near-delta",
        )
        far = manager.register_segment(
            key=make_key(
                tokens,
                "far",
                SegmentKind.CONTEXT_ANCHOR,
            ),
            token_ids=tokens,
            source_start=2,
            residency=ResidencyTier.DEVICE,
            backend_ref="far-delta",
        )
        match = recovery_module.match_anchors(
            target_embedding=(0.0, 0.0),
            target_length=4,
            anchors=(
                recovery_module.AnchorCandidate(
                    near,
                    (0.0, 0.1),
                    4,
                    use_count=5,
                ),
                recovery_module.AnchorCandidate(
                    far,
                    (5.0, 5.0),
                    4,
                ),
            ),
            max_anchors=2,
            temperature=0.1,
            entropy_threshold=0.5,
        )
        self.assertTrue(match.shareable)
        self.assertLess(match.normalized_entropy, 0.5)
        self.assertGreater(match.weights[0], match.weights[1])
        interpolated = recovery_module.interpolate_delta(
            base=(10.0, 20.0),
            anchor_deltas=((1.0, 2.0), (9.0, 10.0)),
            weights=(0.75, 0.25),
        )
        self.assertEqual(interpolated, (13.0, 24.0))

        plan = recovery_module.build_kvcomm_anchor_plan(
            target_token_ids=(99,) + tokens,
            segments=(
                recovery_module.AnchorSegment(
                    segment_id="code",
                    target_start=1,
                    token_ids=tokens,
                    base=base,
                    match=match,
                ),
            ),
        )
        backend = RecordingAnchorBackend()
        stats = manager.execute(plan, backend)
        self.assertEqual(
            backend.dense,
            [(0, 1, "outside_anchor_segments")],
        )
        self.assertEqual(len(backend.reconstructions), 1)
        self.assertEqual(stats.reconstructed_k_tokens, 4)
        self.assertTrue(stats.mechanically_valid)

    def test_kvcomm_gate_rejects_without_compatible_anchor(self):
        tokens = tuple(range(3))
        manager = make_manager()
        base = manager.register_segment(
            key=make_key(tokens, "base"),
            token_ids=tokens,
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="base",
        )
        short = manager.register_segment(
            key=make_key(
                tokens,
                "short",
                SegmentKind.CONTEXT_ANCHOR,
            ),
            token_ids=tokens,
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="delta",
        )
        match = recovery_module.match_anchors(
            target_embedding=(0.0,),
            target_length=4,
            anchors=(
                recovery_module.AnchorCandidate(
                    short,
                    (0.0,),
                    placeholder_length=3,
                ),
            ),
        )
        self.assertFalse(match.shareable)
        plan = recovery_module.build_kvcomm_anchor_plan(
            target_token_ids=tokens,
            segments=(
                recovery_module.AnchorSegment(
                    "code",
                    0,
                    tokens,
                    base,
                    match,
                ),
            ),
        )
        self.assertEqual(
            [(item.length, item.reason) for item in plan.dense_ranges],
            [(3, "no_length_compatible_anchor")],
        )


if __name__ == "__main__":
    unittest.main()
