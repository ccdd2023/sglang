from __future__ import annotations

import unittest

import torch

from sglang.srt.mem_cache.cachetune.token_selection import (
    GradualFilterStage,
    TokenSelection,
    compute_token_deviation,
    select_repair_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class TestComputeTokenDeviation(unittest.TestCase):
    def test_identical_tensors_have_zero_deviation(self):
        reused = torch.randn(6, 4, 8)
        fresh = reused.clone()
        deviation = compute_token_deviation(fresh, reused)
        torch.testing.assert_close(deviation, torch.zeros(6))

    def test_perturbed_tokens_rank_above_unperturbed(self):
        reused = torch.ones(5, 2, 4)
        fresh = reused.clone()
        fresh[1] += 5.0
        fresh[3] += 5.0
        deviation = compute_token_deviation(fresh, reused)
        ranked = torch.argsort(deviation, descending=True)[:2].tolist()
        self.assertEqual(sorted(ranked), [1, 3])

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            compute_token_deviation(torch.randn(3, 2), torch.randn(4, 2))

    def test_empty_tensor_rejected(self):
        with self.assertRaises(ValueError):
            compute_token_deviation(torch.randn(0, 2), torch.randn(0, 2))

    def test_value_weight_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            compute_token_deviation(
                torch.randn(3, 2), torch.randn(3, 2), value_weight=-0.1
            )
        with self.assertRaises(ValueError):
            compute_token_deviation(
                torch.randn(3, 2), torch.randn(3, 2), value_weight=1.1
            )

    def test_value_weight_requires_values(self):
        with self.assertRaises(ValueError):
            compute_token_deviation(
                torch.randn(3, 2),
                torch.randn(3, 2),
                value_weight=0.5,
            )

    def test_value_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            compute_token_deviation(
                torch.randn(3, 2),
                torch.randn(3, 2),
                fresh_values=torch.randn(3, 2),
                reused_values=torch.randn(4, 2),
                value_weight=0.5,
            )

    def test_value_weight_blends_key_and_value_deviation(self):
        reused_k = torch.ones(4, 2)
        fresh_k = reused_k.clone()
        reused_v = torch.ones(4, 2)
        fresh_v = reused_v.clone()
        fresh_v[2] += 10.0
        deviation = compute_token_deviation(
            fresh_k,
            reused_k,
            fresh_values=fresh_v,
            reused_values=reused_v,
            value_weight=1.0,
        )
        self.assertGreater(float(deviation[2]), float(deviation[0]))

    def test_value_weight_zero_ignores_value_tensors_entirely(self):
        reused_k = torch.ones(4, 2)
        fresh_k = reused_k.clone()
        deviation_without_values = compute_token_deviation(fresh_k, reused_k)
        deviation_with_ignored_values = compute_token_deviation(
            fresh_k,
            reused_k,
            fresh_values=torch.randn(4, 2) * 1000.0,
            reused_values=torch.zeros(4, 2),
            value_weight=0.0,
        )
        torch.testing.assert_close(
            deviation_without_values, deviation_with_ignored_values
        )


class TestGradualFilterStageValidation(unittest.TestCase):
    def test_negative_probe_layer_rejected(self):
        with self.assertRaises(ValueError):
            GradualFilterStage(probe_layer_id=-1, keep_ratio=0.5)

    def test_keep_ratio_zero_rejected(self):
        with self.assertRaises(ValueError):
            GradualFilterStage(probe_layer_id=0, keep_ratio=0.0)

    def test_keep_ratio_above_one_rejected(self):
        with self.assertRaises(ValueError):
            GradualFilterStage(probe_layer_id=0, keep_ratio=1.5)

    def test_keep_ratio_exactly_one_allowed(self):
        GradualFilterStage(probe_layer_id=0, keep_ratio=1.0)


class TestTokenSelectionSelfValidation(unittest.TestCase):
    def test_matching_count_constructs_cleanly(self):
        TokenSelection(
            candidate_positions=(0, 1, 2),
            requested_count=2,
            selected_positions=(0, 1),
            stage_scores=(),
        )

    def test_mismatched_count_raises_runtime_error(self):
        with self.assertRaisesRegex(RuntimeError, "requested exactly"):
            TokenSelection(
                candidate_positions=(0, 1, 2),
                requested_count=2,
                selected_positions=(0,),
                stage_scores=(),
            )

    def test_selected_positions_must_be_subset_of_candidates(self):
        with self.assertRaises(ValueError):
            TokenSelection(
                candidate_positions=(0, 1, 2),
                requested_count=1,
                selected_positions=(99,),
                stage_scores=(),
            )

    def test_duplicate_selected_positions_rejected(self):
        with self.assertRaises(ValueError):
            TokenSelection(
                candidate_positions=(0, 1, 2),
                requested_count=2,
                selected_positions=(0, 0),
                stage_scores=(),
            )


class TestSelectRepairTokens(unittest.TestCase):
    def test_final_count_zero_short_circuits_without_scoring(self):
        calls = []

        def deviation_fn(probe_layer_id, positions_tensor):
            calls.append((probe_layer_id, positions_tensor))
            raise AssertionError("deviation_fn must not be called when final_count=0")

        selection = select_repair_tokens(
            list(range(50)),
            stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=0.5),),
            final_count=0,
            deviation_fn=deviation_fn,
        )
        self.assertEqual(selection.selected_positions, ())
        self.assertEqual(selection.requested_count, 0)
        self.assertEqual(calls, [])

    def test_final_count_equals_total_short_circuits_without_scoring(self):
        def deviation_fn(probe_layer_id, positions_tensor):
            raise AssertionError(
                "deviation_fn must not be called when final_count == total"
            )

        selection = select_repair_tokens(
            list(range(30)),
            stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=0.5),),
            final_count=30,
            deviation_fn=deviation_fn,
        )
        self.assertEqual(selection.selected_positions, tuple(range(30)))
        self.assertEqual(selection.requested_count, 30)

    def test_single_stage_selects_top_scores_by_real_scores(self):
        hot = {3, 17, 42, 71}

        def deviation_fn(probe_layer_id, positions_tensor):
            del probe_layer_id
            local = positions_tensor.tolist()
            return torch.tensor([5.0 if p in hot else 0.001 for p in local])

        selection = select_repair_tokens(
            list(range(100)),
            stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),),
            final_count=4,
            deviation_fn=deviation_fn,
        )
        self.assertEqual(set(selection.selected_positions), hot)
        self.assertEqual(selection.requested_count, 4)

    def test_count_sweep_produces_exact_requested_counts(self):
        total = 200

        def deviation_fn(probe_layer_id, positions_tensor):
            del probe_layer_id
            return positions_tensor.float()

        for final_count in (2, 10, 30, 60, 100, 199):
            with self.subTest(final_count=final_count):
                selection = select_repair_tokens(
                    list(range(total)),
                    stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),),
                    final_count=final_count,
                    deviation_fn=deviation_fn,
                )
                self.assertEqual(len(selection.selected_positions), final_count)
                self.assertEqual(
                    set(selection.selected_positions),
                    set(range(total - final_count, total)),
                )

    def test_gradual_multi_stage_narrows_candidate_pool_before_final_score(self):
        # Stage 0 (shallow probe layer) ranks by a *misleading* signal
        # that only weakly correlates with the true repair-worthy
        # tokens, but is cheap and keeps a generous 50% pool. Stage 1 (a
        # deeper probe layer) re-scores only that surviving pool with
        # the *true* signal, proving later/deeper probe layers only ever
        # see an already-narrowed candidate set.
        true_hot = {6, 40}
        calls = []

        def deviation_fn(probe_layer_id, positions_tensor):
            calls.append((probe_layer_id, tuple(positions_tensor.tolist())))
            local = positions_tensor.tolist()
            if probe_layer_id == 0:
                return torch.tensor([1.0 if p % 2 == 0 else 0.5 for p in local])
            return torch.tensor([5.0 if p in true_hot else 0.01 for p in local])

        selection = select_repair_tokens(
            list(range(50)),
            stages=(
                GradualFilterStage(probe_layer_id=0, keep_ratio=0.5),
                GradualFilterStage(probe_layer_id=3, keep_ratio=0.5),
            ),
            final_count=2,
            deviation_fn=deviation_fn,
        )
        self.assertEqual(len(calls[0][1]), 50)
        self.assertLess(len(calls[1][1]), 50)
        self.assertEqual(set(selection.selected_positions), true_hot)

    def test_funnel_never_shrinks_pool_below_final_count(self):
        # A stage whose keep_ratio would otherwise shrink the pool
        # *below* final_count must instead retain at least final_count
        # candidates, so the final selection always has enough
        # candidates to choose exactly final_count from.
        calls = []

        def deviation_fn(probe_layer_id, positions_tensor):
            calls.append(len(positions_tensor))
            return positions_tensor.float()

        selection = select_repair_tokens(
            list(range(20)),
            stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=0.1),),
            final_count=5,
            deviation_fn=deviation_fn,
        )
        self.assertEqual(len(selection.selected_positions), 5)
        # keep_ratio=0.1 over 20 candidates would ask for 2, but the
        # funnel must clamp up to final_count=5.
        self.assertNotIn(2, calls)

    def test_empty_candidates_with_nonzero_final_count_rejected(self):
        with self.assertRaises(ValueError):
            select_repair_tokens(
                [],
                stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),),
                final_count=1,
                deviation_fn=lambda layer, positions: torch.zeros(len(positions)),
            )

    def test_empty_candidates_with_final_count_zero_is_allowed(self):
        selection = select_repair_tokens(
            [],
            stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),),
            final_count=0,
            deviation_fn=lambda layer, positions: torch.zeros(len(positions)),
        )
        self.assertEqual(selection.selected_positions, ())

    def test_final_count_negative_rejected(self):
        with self.assertRaises(ValueError):
            select_repair_tokens(
                list(range(10)),
                stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),),
                final_count=-1,
                deviation_fn=lambda layer, positions: torch.zeros(len(positions)),
            )

    def test_final_count_exceeding_total_rejected(self):
        with self.assertRaises(ValueError):
            select_repair_tokens(
                list(range(10)),
                stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),),
                final_count=11,
                deviation_fn=lambda layer, positions: torch.zeros(len(positions)),
            )

    def test_no_stages_still_scores_full_pool_with_probe_layer_zero(self):
        calls = []

        def deviation_fn(probe_layer_id, positions_tensor):
            calls.append(probe_layer_id)
            return positions_tensor.float()

        selection = select_repair_tokens(
            list(range(10)),
            stages=(),
            final_count=3,
            deviation_fn=deviation_fn,
        )
        self.assertEqual(calls, [0])
        self.assertEqual(set(selection.selected_positions), {7, 8, 9})

    def test_deviation_fn_returning_wrong_shape_rejected(self):
        def deviation_fn(probe_layer_id, positions_tensor):
            del probe_layer_id
            return torch.zeros(len(positions_tensor) - 1)

        with self.assertRaises(ValueError):
            select_repair_tokens(
                list(range(10)),
                stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),),
                final_count=3,
                deviation_fn=deviation_fn,
            )

    def test_selection_is_deterministic_for_a_deterministic_deviation_fn(self):
        def deviation_fn(probe_layer_id, positions_tensor):
            del probe_layer_id
            # Deterministic pseudo-scores from position values.
            return (positions_tensor.float() * 37.0) % 23.0

        results = set()
        for _ in range(10):
            selection = select_repair_tokens(
                list(range(64)),
                stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=0.5),),
                final_count=7,
                deviation_fn=deviation_fn,
            )
            results.add(selection.selected_positions)
        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
