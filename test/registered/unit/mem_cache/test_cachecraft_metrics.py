from __future__ import annotations

import math
import unittest

from sglang.srt.mem_cache.approx_kv.cachecraft_metrics import (
    CacheCraftDecision,
    ChunkContextProfile,
    adjusted_beta,
    compute_a,
    compute_b,
    compute_beta,
    compute_cci,
    compute_cfo,
    decide,
    kendall_tau_order_penalty,
    layer_average,
    select_recompute_positions,
    total_inter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def make_profile(
    *,
    chunk_id: str = "chunk",
    length: int = 4,
    old_prefix_order: tuple[str, ...] = ("A", "B"),
    prefix_lengths: dict[str, int] | None = None,
    inter_by_layer: dict[str, tuple[float, ...]] | None = None,
    intra_by_layer: tuple[float, ...] = (1.0, 1.0),
    token_inter_scores: tuple[float, ...] | None = None,
) -> ChunkContextProfile:
    prefix_lengths = prefix_lengths or {"A": 4, "B": 4}
    inter_by_layer = inter_by_layer or {
        "A": (2.0, 2.0),
        "B": (1.0, 1.0),
    }
    token_inter_scores = token_inter_scores or tuple(float(i) for i in range(length))
    return ChunkContextProfile(
        chunk_id=chunk_id,
        length=length,
        old_prefix_order=old_prefix_order,
        prefix_chunk_lengths=prefix_lengths,
        inter_attention_by_layer=inter_by_layer,
        intra_attention_by_layer=intra_by_layer,
        token_inter_scores=token_inter_scores,
    )


class TestChunkContextProfileValidation(unittest.TestCase):
    def test_rejects_empty_chunk_id_and_bad_length(self):
        with self.assertRaisesRegex(ValueError, "chunk_id"):
            make_profile(chunk_id="")
        with self.assertRaisesRegex(ValueError, "length must be positive"):
            make_profile(length=0)

    def test_rejects_missing_prefix_entries(self):
        with self.assertRaisesRegex(ValueError, "prefix_chunk_lengths"):
            make_profile(prefix_lengths={"A": 4})
        with self.assertRaisesRegex(ValueError, "inter_attention_by_layer"):
            make_profile(inter_by_layer={"A": (2.0, 2.0)})

    def test_rejects_token_score_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "token_inter_scores"):
            make_profile(token_inter_scores=(1.0, 2.0))

    def test_rejects_duplicate_prefix_chunk(self):
        with self.assertRaisesRegex(ValueError, "not repeat"):
            make_profile(old_prefix_order=("A", "A"))


class TestContextualizationChangesCCI(unittest.TestCase):
    """Proves that changing the *context* (the attention a chunk received
    from its prefix when cached) changes a(Ci)/CCI, holding order fixed."""

    def test_more_external_attention_increases_a_and_cci(self):
        self_contextualized = make_profile(
            inter_by_layer={"A": (0.1, 0.1), "B": (0.1, 0.1)},
            intra_by_layer=(5.0, 5.0),
        )
        heavily_contextualized = make_profile(
            inter_by_layer={"A": (8.0, 8.0), "B": (8.0, 8.0)},
            intra_by_layer=(5.0, 5.0),
        )
        a_low = compute_a(self_contextualized)
        a_high = compute_a(heavily_contextualized)
        self.assertLess(a_low, a_high)

        cci_low = compute_cci(self_contextualized)
        cci_high = compute_cci(heavily_contextualized)
        self.assertLess(cci_low, cci_high)
        # Case 1 (self-contextualized) vs Case 2 (heavily contextualized)
        # from Fig. 11: CCI must clearly separate the two regimes.
        self.assertLess(cci_low, 0.6)
        self.assertGreater(cci_high, 0.9)

    def test_cci_changes_propagate_to_cfo_and_decision(self):
        self_contextualized = make_profile(
            inter_by_layer={"A": (0.05, 0.05), "B": (0.05, 0.05)},
            intra_by_layer=(5.0, 5.0),
        )
        heavily_contextualized = make_profile(
            inter_by_layer={"A": (9.0, 9.0), "B": (9.0, 9.0)},
            intra_by_layer=(5.0, 5.0),
        )
        # Same beta'/context overlap for both; only CCI differs.
        beta_prime = 0.5
        cfo_low = compute_cfo(compute_cci(self_contextualized), beta_prime)
        cfo_high = compute_cfo(compute_cci(heavily_contextualized), beta_prime)
        self.assertLess(cfo_low, cfo_high)

        # A deployment can tune the full-recompute threshold (mirrors the
        # paper's alpha-tuning discussion, Eq. (13)); with a threshold of
        # 0.4 the CCI-driven CFO gap alone flips the decision category.
        decision_low = decide(cfo_low, cache_hit=True, full_recompute_threshold=0.4)
        decision_high = decide(cfo_high, cache_hit=True, full_recompute_threshold=0.4)
        self.assertEqual(decision_low, CacheCraftDecision.PARTIAL_REPAIR)
        self.assertEqual(decision_high, CacheCraftDecision.FULL_RECOMPUTE)
        self.assertNotEqual(decision_low, decision_high)


class TestOrderPenaltyChangesDecision(unittest.TestCase):
    """Proves that changing chunk *order* alone (same set of prefix chunks,
    same attention statistics) changes gamma/beta'/CFO/decision, even though
    beta (order-invariant subset overlap) stays identical."""

    def test_kendall_tau_distance_matches_hand_computation(self):
        # old order: A, B, C ; new order: C, B, A -> fully reversed -> all
        # 3 pairs discordant -> gamma = 1.0
        gamma_reversed = kendall_tau_order_penalty(("A", "B", "C"), ("C", "B", "A"))
        self.assertAlmostEqual(gamma_reversed, 1.0)

        # identical order -> no discordant pairs -> gamma = 0
        gamma_identical = kendall_tau_order_penalty(("A", "B", "C"), ("A", "B", "C"))
        self.assertAlmostEqual(gamma_identical, 0.0)

        # single swap out of 3 pairs discordant
        gamma_single_swap = kendall_tau_order_penalty(("A", "B", "C"), ("B", "A", "C"))
        self.assertAlmostEqual(gamma_single_swap, 1.0 / 3.0)

    def test_beta_is_order_invariant_but_gamma_is_not(self):
        profile = make_profile(
            old_prefix_order=("A", "B"),
            prefix_lengths={"A": 4, "B": 4},
            inter_by_layer={"A": (3.0, 3.0), "B": (3.0, 3.0)},
        )
        beta_order_1 = compute_beta(profile, ("A", "B"))
        beta_order_2 = compute_beta(profile, ("B", "A"))
        self.assertAlmostEqual(beta_order_1, beta_order_2)

        gamma_order_1 = kendall_tau_order_penalty(("A", "B"), ("A", "B"))
        gamma_order_2 = kendall_tau_order_penalty(("A", "B"), ("B", "A"))
        self.assertNotEqual(gamma_order_1, gamma_order_2)
        self.assertAlmostEqual(gamma_order_1, 0.0)
        self.assertAlmostEqual(gamma_order_2, 1.0)

        beta_prime_1 = adjusted_beta(beta_order_1, gamma_order_1)
        beta_prime_2 = adjusted_beta(beta_order_2, gamma_order_2)
        self.assertGreater(beta_prime_1, beta_prime_2)

    def test_order_change_alone_flips_decision(self):
        # High CCI chunk: reusable only when order also matches closely.
        profile = make_profile(
            old_prefix_order=("A", "B", "C"),
            prefix_lengths={"A": 4, "B": 4, "C": 4},
            inter_by_layer={
                "A": (6.0, 6.0),
                "B": (6.0, 6.0),
                "C": (6.0, 6.0),
            },
            intra_by_layer=(2.0, 2.0),
        )
        cci = compute_cci(profile)
        new_prefix = ("A", "B", "C")
        reversed_prefix = ("C", "B", "A")

        beta_same_order = compute_beta(profile, new_prefix)
        gamma_same_order = kendall_tau_order_penalty(
            profile.old_prefix_order, new_prefix
        )
        beta_prime_same = adjusted_beta(beta_same_order, gamma_same_order)
        cfo_same_order = compute_cfo(cci, beta_prime_same)

        beta_reversed = compute_beta(profile, reversed_prefix)
        gamma_reversed = kendall_tau_order_penalty(
            profile.old_prefix_order, reversed_prefix
        )
        beta_prime_reversed = adjusted_beta(beta_reversed, gamma_reversed)
        cfo_reversed = compute_cfo(cci, beta_prime_reversed)

        # Same chunk set (beta identical) but different order -> strictly
        # higher CFO (more recomputation needed) when reversed.
        self.assertAlmostEqual(beta_same_order, beta_reversed)
        self.assertLess(cfo_same_order, cfo_reversed)

        decision_same_order = decide(cfo_same_order, cache_hit=True)
        decision_reversed = decide(cfo_reversed, cache_hit=True)
        self.assertEqual(decision_same_order, CacheCraftDecision.DIRECT_REUSE)
        self.assertNotEqual(decision_same_order, decision_reversed)


class TestBetaEdgeCases(unittest.TestCase):
    def test_beta_zero_denominator_treated_as_fully_reusable(self):
        profile = make_profile(
            old_prefix_order=(),
            prefix_lengths={},
            inter_by_layer={},
        )
        self.assertEqual(compute_beta(profile, ("X",)), 1.0)

    def test_total_inter_missing_chunk_is_zero(self):
        profile = make_profile()
        self.assertEqual(total_inter(profile, "does-not-exist"), 0.0)

    def test_kendall_tau_length_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            kendall_tau_order_penalty(("A", "B"), ("A",), common_ids=("A", "B"))


class TestCFOAndDecision(unittest.TestCase):
    def test_cfo_clamped_to_unit_interval(self):
        self.assertEqual(compute_cfo(1.0, -1.0, alpha=5.0), 1.0)
        self.assertEqual(compute_cfo(0.0, 1.0, alpha=5.0), 0.0)

    def test_cfo_requires_positive_alpha(self):
        with self.assertRaisesRegex(ValueError, "alpha"):
            compute_cfo(0.5, 0.5, alpha=0.0)

    def test_decide_store_miss_forces_full_recompute(self):
        self.assertEqual(
            decide(0.0, cache_hit=False),
            CacheCraftDecision.FULL_RECOMPUTE,
        )

    def test_decide_boundaries(self):
        self.assertEqual(decide(0.0, cache_hit=True), CacheCraftDecision.DIRECT_REUSE)
        self.assertEqual(decide(0.5, cache_hit=True), CacheCraftDecision.PARTIAL_REPAIR)
        self.assertEqual(decide(1.0, cache_hit=True), CacheCraftDecision.FULL_RECOMPUTE)
        self.assertEqual(
            decide(0.8, cache_hit=True, full_recompute_threshold=0.7),
            CacheCraftDecision.FULL_RECOMPUTE,
        )

    def test_decide_rejects_bad_threshold(self):
        with self.assertRaisesRegex(ValueError, "full_recompute_threshold"):
            decide(0.5, cache_hit=True, full_recompute_threshold=0.0)


class TestTopNTokenSelection(unittest.TestCase):
    def test_selects_highest_scoring_tokens(self):
        profile = make_profile(
            length=5,
            token_inter_scores=(0.1, 0.9, 0.4, 0.8, 0.2),
        )
        positions = select_recompute_positions(profile, cfo=0.4)
        # ceil(0.4*5) = 2 -> tokens 1 (0.9) and 3 (0.8), sorted ascending.
        self.assertEqual(positions, (1, 3))

    def test_zero_cfo_selects_nothing_full_cfo_selects_everything(self):
        profile = make_profile(length=4, token_inter_scores=(1, 2, 3, 4))
        self.assertEqual(select_recompute_positions(profile, cfo=0.0), ())
        self.assertEqual(select_recompute_positions(profile, cfo=1.0), (0, 1, 2, 3))

    def test_context_change_alters_top_n_selection(self):
        # Same CFO, but a change in *which* tokens were externally attended
        # (context) changes which positions get selected for recompute.
        profile_a = make_profile(length=4, token_inter_scores=(9.0, 1.0, 1.0, 1.0))
        profile_b = make_profile(length=4, token_inter_scores=(1.0, 1.0, 1.0, 9.0))
        positions_a = select_recompute_positions(profile_a, cfo=0.25)
        positions_b = select_recompute_positions(profile_b, cfo=0.25)
        self.assertEqual(positions_a, (0,))
        self.assertEqual(positions_b, (3,))
        self.assertNotEqual(positions_a, positions_b)

    def test_rejects_out_of_range_cfo(self):
        profile = make_profile()
        with self.assertRaisesRegex(ValueError, "cfo"):
            select_recompute_positions(profile, cfo=1.5)


class TestLayerAverage(unittest.TestCase):
    def test_layer_average_basic(self):
        self.assertAlmostEqual(layer_average([1.0, 2.0, 3.0]), 2.0)

    def test_layer_average_rejects_empty(self):
        with self.assertRaisesRegex(ValueError, "values"):
            layer_average([])

    def test_compute_b_matches_hand_computation(self):
        profile = make_profile(length=2, intra_by_layer=(4.0, 8.0))
        # b_l = intra_l / length^2 = 4/4=1.0, 8/4=2.0 -> mean = 1.5
        self.assertAlmostEqual(compute_b(profile), 1.5)

    def test_compute_cci_matches_sigmoid_of_ratio(self):
        profile = make_profile(
            length=2,
            old_prefix_order=("A",),
            prefix_lengths={"A": 2},
            inter_by_layer={"A": (2.0, 2.0)},
            intra_by_layer=(4.0, 4.0),
        )
        # a_l = 2/(2*2) = 0.5 -> a=0.5 ; b_l = 4/4 = 1.0 -> b=1.0
        # ratio = 0.5 -> sigmoid(0.5)
        expected = 1.0 / (1.0 + math.exp(-0.5))
        self.assertAlmostEqual(compute_cci(profile), expected)


if __name__ == "__main__":
    unittest.main()
