from __future__ import annotations

import unittest

import torch

from sglang.srt.mem_cache.cacheblend.hkvd import (
    GradualFilterStage,
    compute_token_deviation,
    select_hkvd_tokens,
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

    def test_value_weight_requires_values(self):
        with self.assertRaises(ValueError):
            compute_token_deviation(
                torch.randn(3, 2),
                torch.randn(3, 2),
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


class TestGradualFilter(unittest.TestCase):
    def _make_scores(self, size: int, hot_positions: set[int], magnitude=5.0):
        base = torch.rand(size) * 0.01
        for position in hot_positions:
            base[position] = magnitude
        return base

    def test_single_stage_selects_top_ratio_by_real_scores(self):
        hot = {3, 17, 42, 71}

        def deviation_fn(probe_layer_id, positions_tensor):
            del probe_layer_id
            local = positions_tensor.tolist()
            return torch.tensor(
                [5.0 if p in hot else 0.001 for p in local]
            )

        selection = select_hkvd_tokens(
            list(range(100)),
            stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),),
            final_ratio=0.04,
            deviation_fn=deviation_fn,
        )
        self.assertEqual(set(selection.selected_positions), hot)

    def test_ratio_sweep_produces_expected_counts(self):
        total = 200

        def deviation_fn(probe_layer_id, positions_tensor):
            del probe_layer_id
            # Deterministic but non-degenerate scores: distinct per position.
            return positions_tensor.float()

        for ratio, expected_count in (
            (0.01, 2),
            (0.05, 10),
            (0.15, 30),
            (0.30, 60),
        ):
            with self.subTest(ratio=ratio):
                selection = select_hkvd_tokens(
                    list(range(total)),
                    stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),),
                    final_ratio=ratio,
                    deviation_fn=deviation_fn,
                )
                self.assertEqual(len(selection.selected_positions), expected_count)
                # Highest-score (highest local index) positions must win.
                self.assertEqual(
                    set(selection.selected_positions),
                    set(range(total - expected_count, total)),
                )

    def test_gradual_multi_stage_narrows_candidate_pool_before_final_score(self):
        # Stage 0 (shallow probe layer) ranks by a *misleading* signal that
        # only weakly correlates with the true HKVD-relevant tokens, but is
        # cheap and keeps a generous top-50% pool. Stage 1 (a deeper probe
        # layer) re-scores only that surviving pool with the *true* signal.
        # This proves later, more expensive/precise layers only ever see an
        # already-narrowed candidate set (the "gradual" part of gradual
        # filtering), while the true HKVD deviation still drives the final
        # pick.
        true_hot = {6, 40}
        calls = []

        def deviation_fn(probe_layer_id, positions_tensor):
            calls.append((probe_layer_id, tuple(positions_tensor.tolist())))
            local = positions_tensor.tolist()
            if probe_layer_id == 0:
                # Cheap, imprecise: mildly favors even positions but does
                # not single out the true hot set.
                return torch.tensor([1.0 if p % 2 == 0 else 0.5 for p in local])
            # Deep probe layer: only reachable after stage-0 filtering; the
            # true signal.
            return torch.tensor([5.0 if p in true_hot else 0.01 for p in local])

        selection = select_hkvd_tokens(
            list(range(50)),
            stages=(
                GradualFilterStage(probe_layer_id=0, keep_ratio=0.5),
                GradualFilterStage(probe_layer_id=3, keep_ratio=0.5),
            ),
            final_ratio=0.04,
            deviation_fn=deviation_fn,
        )
        # Stage 0 was called against the full 50-position candidate pool.
        self.assertEqual(len(calls[0][1]), 50)
        # Stage 1 (the deep probe) only ever saw the narrowed, shrunk pool.
        self.assertLess(len(calls[1][1]), 50)
        self.assertEqual(set(selection.selected_positions), true_hot)

    def test_empty_candidates_rejected(self):
        with self.assertRaises(ValueError):
            select_hkvd_tokens(
                [],
                stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=1.0),),
                final_ratio=0.05,
                deviation_fn=lambda layer, positions: torch.zeros(len(positions)),
            )

    def test_invalid_keep_ratio_rejected(self):
        with self.assertRaises(ValueError):
            GradualFilterStage(probe_layer_id=0, keep_ratio=0.0)
        with self.assertRaises(ValueError):
            GradualFilterStage(probe_layer_id=0, keep_ratio=1.5)


if __name__ == "__main__":
    unittest.main()
