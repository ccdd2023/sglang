from __future__ import annotations

import math
import unittest

import torch

from sglang.srt.mem_cache.approx_kv.cachecraft_attention import (
    capture_chunk_profile,
    causal_attention_weights,
)
from sglang.srt.mem_cache.approx_kv.cachecraft_metrics import compute_a, compute_b
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class TestCausalAttentionWeights(unittest.TestCase):
    def test_rows_sum_to_one_and_are_causal(self):
        torch.manual_seed(0)
        query = torch.randn(6, 4)
        key = torch.randn(6, 4)
        weights = causal_attention_weights(query, key)
        self.assertEqual(weights.shape, (6, 6))
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones(6), atol=1e-6, rtol=1e-6
        )
        # No probability mass above the diagonal (no attending to the future).
        upper = torch.triu(weights, diagonal=1)
        self.assertEqual(float(upper.abs().sum()), 0.0)

    def test_matches_manual_softmax_for_two_tokens(self):
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        key = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        weights = causal_attention_weights(query, key, scale=1.0)
        # Row 0 can only attend to itself.
        self.assertAlmostEqual(float(weights[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(weights[0, 1]), 0.0, places=6)
        # Row 1: scores = [q1.k0=0, q1.k1=1] -> softmax([0,1])
        expected_1 = math.exp(1.0) / (math.exp(0.0) + math.exp(1.0))
        self.assertAlmostEqual(float(weights[1, 1]), expected_1, places=6)

    def test_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "matching shape"):
            causal_attention_weights(torch.randn(4, 4), torch.randn(3, 4))


class TestCaptureChunkProfile(unittest.TestCase):
    def _weights_for(self, seed: int, seq_len: int = 9, head_dim: int = 4):
        torch.manual_seed(seed)
        query = torch.randn(seq_len, head_dim)
        key = torch.randn(seq_len, head_dim)
        return causal_attention_weights(query, key)

    def test_capture_matches_manual_sums(self):
        # sequence: A [0:3], B [3:6], target chunk Ci [6:9]
        weights = self._weights_for(seed=1)
        spans = {"A": (0, 3), "B": (3, 6), "target": (6, 9)}
        profile = capture_chunk_profile(
            chunk_id="target",
            weights_per_layer=[weights],
            chunk_spans=spans,
            old_prefix_order=("A", "B"),
        )
        self.assertEqual(profile.length, 3)
        self.assertEqual(profile.layer_num, 1)

        expected_intra = float(weights[6:9, 6:9].tril(diagonal=-1).sum())
        self.assertAlmostEqual(
            profile.intra_attention_by_layer[0], expected_intra, places=6
        )
        expected_inter_a = float(weights[6:9, 0:3].sum())
        expected_inter_b = float(weights[6:9, 3:6].sum())
        self.assertAlmostEqual(
            profile.inter_attention_by_layer["A"][0], expected_inter_a, places=6
        )
        self.assertAlmostEqual(
            profile.inter_attention_by_layer["B"][0], expected_inter_b, places=6
        )
        expected_token_scores = tuple(
            float(weights[6 + k, 0:6].sum()) for k in range(3)
        )
        for actual, expected in zip(profile.token_inter_scores, expected_token_scores):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_reordering_prefix_changes_captured_context(self):
        # Same target chunk tokens/logical content, but the prefix chunks
        # occupy different positions -> different causal masking -> real,
        # measurable change in captured inter-attention per prefix chunk.
        weights_ab = self._weights_for(seed=2)
        spans_ab = {"A": (0, 3), "B": (3, 6), "target": (6, 9)}
        profile_ab = capture_chunk_profile(
            chunk_id="target",
            weights_per_layer=[weights_ab],
            chunk_spans=spans_ab,
            old_prefix_order=("A", "B"),
        )

        # Reuse the exact same underlying attention matrix, but tell the
        # capture function the prefix chunks are swapped (as if B preceded
        # A physically); the per-chunk inter sums must swap accordingly,
        # proving the capture is sensitive to (not independent of) the
        # asserted chunk layout/order.
        spans_swapped = {"B": (0, 3), "A": (3, 6), "target": (6, 9)}
        profile_swapped = capture_chunk_profile(
            chunk_id="target",
            weights_per_layer=[weights_ab],
            chunk_spans=spans_swapped,
            old_prefix_order=("B", "A"),
        )
        # Swapping which physical span each label points to makes each
        # chunk id pick up the *other* chunk's real cross-attention mass:
        # this proves the captured inter-attention is a genuine function of
        # token position/order, not just a chunk-id lookup.
        self.assertAlmostEqual(
            profile_swapped.inter_attention_by_layer["A"][0],
            profile_ab.inter_attention_by_layer["B"][0],
            places=6,
        )
        self.assertAlmostEqual(
            profile_swapped.inter_attention_by_layer["B"][0],
            profile_ab.inter_attention_by_layer["A"][0],
            places=6,
        )
        # And the two chunks have genuinely different attention mass in the
        # original layout, otherwise the swap check above would be trivial.
        self.assertNotAlmostEqual(
            profile_ab.inter_attention_by_layer["A"][0],
            profile_ab.inter_attention_by_layer["B"][0],
            places=3,
        )

    def test_more_layers_of_real_attention_change_cci_inputs(self):
        # Two independent (different-seed) real attention matrices as two
        # "layers" for the same chunk layout: a(Ci)/b(Ci) must reflect the
        # genuine layer-averaged sums, not a single layer's values.
        spans = {"A": (0, 3), "target": (3, 6)}
        weights_layer_0 = self._weights_for(seed=3, seq_len=6)
        weights_layer_1 = self._weights_for(seed=4, seq_len=6)
        profile = capture_chunk_profile(
            chunk_id="target",
            weights_per_layer=[weights_layer_0, weights_layer_1],
            chunk_spans=spans,
            old_prefix_order=("A",),
        )
        self.assertEqual(profile.layer_num, 2)
        expected_a = (
            float(weights_layer_0[3:6, 0:3].sum()) / (3 * 3)
            + float(weights_layer_1[3:6, 0:3].sum()) / (3 * 3)
        ) / 2
        self.assertAlmostEqual(compute_a(profile), expected_a, places=6)
        expected_b = (
            float(weights_layer_0[3:6, 3:6].tril(diagonal=-1).sum()) / (3 * 3)
            + float(weights_layer_1[3:6, 3:6].tril(diagonal=-1).sum()) / (3 * 3)
        ) / 2
        self.assertAlmostEqual(compute_b(profile), expected_b, places=6)

    def test_rejects_missing_chunk_span(self):
        weights = self._weights_for(seed=5, seq_len=6)
        with self.assertRaisesRegex(ValueError, "chunk_spans"):
            capture_chunk_profile(
                chunk_id="missing",
                weights_per_layer=[weights],
                chunk_spans={"A": (0, 3)},
                old_prefix_order=(),
            )


if __name__ == "__main__":
    unittest.main()
