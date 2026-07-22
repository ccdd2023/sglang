from __future__ import annotations

import math
import unittest

from sglang.srt.mem_cache.cachetune.golden_section import (
    golden_section_search_minimize,
    warm_start_bracket,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class TestGoldenSectionSearchMinimize(unittest.TestCase):
    def test_finds_minimum_of_simple_parabola(self):
        # f(x) = (x - 0.37)^2, true minimizer at x=0.37.
        minimizer = golden_section_search_minimize(
            lambda x: (x - 0.37) ** 2, 0.0, 1.0, tol=1e-6
        )
        self.assertAlmostEqual(minimizer, 0.37, places=4)

    def test_finds_minimum_at_lower_bound(self):
        minimizer = golden_section_search_minimize(lambda x: x, 0.0, 1.0, tol=1e-6)
        self.assertAlmostEqual(minimizer, 0.0, places=4)

    def test_finds_minimum_at_upper_bound(self):
        minimizer = golden_section_search_minimize(lambda x: -x, 0.0, 1.0, tol=1e-6)
        self.assertAlmostEqual(minimizer, 1.0, places=4)

    def test_finds_minimum_of_roofline_style_max_function(self):
        # Mimics T_layer(r) = max(r * a, (1 - r) * b) + c: convex,
        # unimodal, minimized where the two linear pieces cross.
        a, b, c = 3.0, 1.0, 10.0

        def f(r: float) -> float:
            return max(r * a, (1.0 - r) * b) + c

        expected_r = b / (a + b)
        minimizer = golden_section_search_minimize(f, 0.0, 1.0, tol=1e-6)
        self.assertAlmostEqual(minimizer, expected_r, places=4)

    def test_flat_function_tie_break_prefers_lower_half(self):
        # A perfectly flat function anywhere in the bracket must
        # deterministically converge toward the *lower* half (smaller
        # ratio => less recompute work), never toward the upper half and
        # never depend on evaluation order.
        calls: list[float] = []

        def f(x: float) -> float:
            calls.append(x)
            return 1.0  # perfectly flat everywhere

        minimizer = golden_section_search_minimize(f, 0.0, 1.0, tol=1e-3)
        self.assertLess(minimizer, 0.5)

    def test_converges_within_requested_tolerance(self):
        tol = 1e-5
        minimizer = golden_section_search_minimize(
            lambda x: (x - 0.6) ** 2, 0.0, 1.0, tol=tol, max_iterations=200
        )
        self.assertLess(abs(minimizer - 0.6), tol * 10)

    def test_lo_greater_than_hi_rejected(self):
        with self.assertRaises(ValueError):
            golden_section_search_minimize(lambda x: x, 1.0, 0.0)

    def test_non_positive_tol_rejected(self):
        with self.assertRaises(ValueError):
            golden_section_search_minimize(lambda x: x, 0.0, 1.0, tol=0.0)
        with self.assertRaises(ValueError):
            golden_section_search_minimize(lambda x: x, 0.0, 1.0, tol=-1e-4)

    def test_non_positive_max_iterations_rejected(self):
        with self.assertRaises(ValueError):
            golden_section_search_minimize(lambda x: x, 0.0, 1.0, max_iterations=0)

    def test_degenerate_bracket_returns_midpoint_immediately(self):
        minimizer = golden_section_search_minimize(
            lambda x: (x - 0.5) ** 2, 0.5, 0.5, tol=1e-4
        )
        self.assertAlmostEqual(minimizer, 0.5)

    def test_does_not_exceed_max_iterations_evaluations(self):
        calls = {"count": 0}

        def f(x: float) -> float:
            calls["count"] += 1
            return (x - 0.42) ** 2

        golden_section_search_minimize(f, 0.0, 1.0, tol=1e-9, max_iterations=5)
        # Two initial evaluations plus at most one new evaluation per
        # iteration.
        self.assertLessEqual(calls["count"], 2 + 5)


class TestWarmStartBracket(unittest.TestCase):
    def test_centers_window_on_r0(self):
        lo, hi = warm_start_bracket(0.5, 0.0, 1.0, span=0.4)
        self.assertAlmostEqual((lo + hi) / 2.0, 0.5, places=6)
        self.assertAlmostEqual(hi - lo, 0.4, places=6)

    def test_window_is_clamped_when_r0_near_lower_bound(self):
        lo, hi = warm_start_bracket(0.02, 0.0, 1.0, span=0.4)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        self.assertLess(lo, 0.02)

    def test_window_is_clamped_when_r0_near_upper_bound(self):
        lo, hi = warm_start_bracket(0.98, 0.0, 1.0, span=0.4)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        self.assertGreater(hi, 0.98)

    def test_r0_outside_bounds_is_clamped_into_range(self):
        lo, hi = warm_start_bracket(1.5, 0.15, 0.9, span=0.4)
        self.assertGreaterEqual(lo, 0.15)
        self.assertLessEqual(hi, 0.9)

    def test_full_bounds_used_when_window_degenerates(self):
        # span so small relative to a near-boundary r0 that the centered
        # window would collapse to empty; must fall back to the full
        # bounds rather than return an empty/invalid bracket.
        lo, hi = warm_start_bracket(0.15, 0.15, 0.15, span=0.01)
        self.assertEqual((lo, hi), (0.15, 0.15))

    def test_r_min_greater_than_r_max_rejected(self):
        with self.assertRaises(ValueError):
            warm_start_bracket(0.5, 0.9, 0.1)

    def test_span_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            warm_start_bracket(0.5, 0.0, 1.0, span=0.0)
        with self.assertRaises(ValueError):
            warm_start_bracket(0.5, 0.0, 1.0, span=1.5)

    def test_non_finite_r0_rejected(self):
        with self.assertRaises(ValueError):
            warm_start_bracket(math.nan, 0.0, 1.0)
        with self.assertRaises(ValueError):
            warm_start_bracket(math.inf, 0.0, 1.0)

    def test_bracket_is_always_within_full_bounds(self):
        for r0 in (-1.0, 0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0, 2.0):
            with self.subTest(r0=r0):
                lo, hi = warm_start_bracket(r0, 0.15, 0.9, span=0.6)
                self.assertGreaterEqual(lo, 0.15)
                self.assertLessEqual(hi, 0.9)
                self.assertLessEqual(lo, hi)


if __name__ == "__main__":
    unittest.main()
