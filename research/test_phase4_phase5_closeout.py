from __future__ import annotations

import unittest

from phase4_phase5_closeout import (
    ScratchNamespaceTracker,
    build_cost_ledger,
    nearest_rank,
    offline_variable_size_optimum,
    pressure_filler_count,
    summarize_requests,
)


class TestCloseoutMetrics(unittest.TestCase):
    def test_nearest_rank_matches_runner_definition(self):
        self.assertEqual(nearest_rank(list(range(20)), 0.95), 18)

    def test_hit_fraction_clamps_per_request(self):
        rows = [
            {
                "ttft_ms": 1.0,
                "elapsed_ms": 1.1,
                "cached_tokens": 11,
                "expected_reusable_prefix_tokens": 10,
            },
            {
                "ttft_ms": 2.0,
                "elapsed_ms": 2.1,
                "cached_tokens": 5,
                "expected_reusable_prefix_tokens": 10,
            },
        ]
        summary = summarize_requests(rows)
        self.assertEqual(summary["per_request_clamped_hit_fraction"], 0.75)
        self.assertEqual(summary["misses"], 1)

    def test_variable_size_optimum_respects_capacity(self):
        rows = [
            {
                "step": 0,
                "object_id": "a",
                "expected_reusable_prefix_tokens": 6,
            },
            {
                "step": 1,
                "object_id": "b",
                "expected_reusable_prefix_tokens": 6,
            },
            {
                "step": 2,
                "object_id": "a",
                "expected_reusable_prefix_tokens": 6,
            },
            {
                "step": 3,
                "object_id": "b",
                "expected_reusable_prefix_tokens": 6,
            },
        ]
        result = offline_variable_size_optimum(rows, capacity_tokens=6)
        self.assertEqual(result["reuse_requests"], 2)
        self.assertEqual(result["optimal_hits"], 1)
        self.assertEqual(result["optimal_misses"], 1)

    def test_cost_ledger_identities(self):
        ledger = build_cost_ledger(
            source_preparation_ms=10,
            target_adapter_preparation_ms=20,
            seed_head_ms=1,
            post_pressure_reseed_ms=2,
            transfer_ms=3,
            target_only_ms=4,
        )
        self.assertEqual(ledger["request_path_ms"], 30)
        self.assertEqual(ledger["recovery_object_lifecycle_ms"], 40)

    def test_pressure_budget_subtracts_used_and_evictable_setup(self):
        fillers = pressure_filler_count(
            capacity_tokens=100,
            rho_logical_demand=2,
            setup_used_tokens=20,
            setup_evictable_tokens=30,
            tokens_per_filler=25,
        )
        self.assertEqual(fillers, 6)

    def test_scratch_namespace_gc_without_global_flush(self):
        tracker = ScratchNamespaceTracker()
        tracker.acquire("round-a")
        tracker.acquire("round-b")
        tracker.release("round-a")
        self.assertEqual(tracker.active(), ("round-b",))
        tracker.release("round-b")
        self.assertEqual(tracker.active(), ())


if __name__ == "__main__":
    unittest.main()
