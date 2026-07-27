from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")

from benchmark.approx_kv import run_cl3_phase5_recompute as cl3
from benchmark.approx_kv.phase6.runner import append_jsonl, execution_status
from benchmark.approx_kv.run_cl1_qualification import (
    candidate_k,
    chunks,
    percentile,
    promotion,
    target_segments,
)
from benchmark.approx_kv.run_cl2_chunk_gate import summarize


class TestCloseoutRunners(unittest.TestCase):
    def test_causal_segments_cover_body_without_exceeding_limit(self):
        body = list(range(2048))
        parts = chunks(body, 512)
        self.assertEqual([len(part) for part in parts], [512, 512, 512, 512])
        segments = target_segments(
            body_tokens=2048,
            header_tokens=64,
            segment_tokens=512,
            setting_id="setting",
        )
        self.assertEqual(
            [segment["target_start"] for segment in segments],
            [64, 576, 1088, 1600],
        )
        self.assertEqual(sum(segment["length"] for segment in segments), 2048)

    def test_candidate_parser(self):
        self.assertIsNone(candidate_k("r0"))
        self.assertEqual(candidate_k("r1_k32"), 32)

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)

    def test_promotion_prefers_more_repair_inside_two_percent_tie(self):
        summaries = {
            "r1_k8": {
                "2048": {
                    "median_request_path_speedup": 1.20,
                    "paired_target_p95_ratio": 1.0,
                    "amortized_speedup": {"8": 1.1},
                    "all_guardrails_passed": True,
                    "per_restart_median_speedup": [1.1, 1.2, 1.3],
                }
            },
            "r1_k16": {
                "2048": {
                    "median_request_path_speedup": 1.19,
                    "paired_target_p95_ratio": 0.99,
                    "amortized_speedup": {"8": 1.1},
                    "all_guardrails_passed": True,
                    "per_restart_median_speedup": [1.1, 1.2, 1.3],
                }
            },
        }
        result = promotion(summaries, restarts=3)
        self.assertEqual(result["winner"], "r1_k16")

    def test_cl2_summary_uses_all_paired_samples(self):
        rows = [
            {
                "formal": [
                    {
                        "request_path_speedup": 2.0,
                        "passed": True,
                        "approx": {
                            "target": {"ttft_ms": 10.0},
                            "ledger": {
                                "setup_ms": 5.0,
                                "request_path_ms": 10.0,
                            },
                        },
                        "dense": {
                            "target": {"ttft_ms": 20.0},
                            "ledger": {"request_path_ms": 20.0},
                        },
                    },
                    {
                        "request_path_speedup": 1.5,
                        "passed": True,
                        "dense": {
                            "target": {"ttft_ms": 18.0},
                            "ledger": {"request_path_ms": 18.0},
                        },
                        "approx": {
                            "target": {"ttft_ms": 12.0},
                            "ledger": {
                                "setup_ms": 5.0,
                                "request_path_ms": 12.0,
                            },
                        },
                    },
                ]
            }
        ]
        result = summarize(rows)
        self.assertEqual(result["median_request_path_speedup"], 1.75)
        self.assertTrue(result["all_guardrails_passed"])

    def test_central_log_records_and_classifies_driver_block(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.jsonl"
            append_jsonl(path, {"run_id": "run", "status": "running"})
            self.assertEqual(
                json.loads(path.read_text().strip()),
                {"run_id": "run", "status": "running"},
            )
        self.assertEqual(
            execution_status(
                RuntimeError(
                    "Failed to initialize NVML: driver/library version mismatch"
                )
            ),
            "blocked",
        )


if __name__ == "__main__":
    unittest.main()


class TestCL3Recalculation(unittest.TestCase):
    def _record(self, **overrides):
        record = {
            "sample_kind": "measured",
            "phase": "workflow",
            "role": "coder",
            "repeat": 0,
            "ttft_ms": 100.0,
            "elapsed_ms": 110.0,
            "cached_tokens": 512,
            "expected_reusable_prefix_tokens": 1024,
        }
        record.update(overrides)
        return record

    def test_hit_fraction_is_clamped_per_request(self):
        over = self._record(cached_tokens=2048)
        self.assertEqual(cl3.clamped_hit_fraction(over), 1.0)
        self.assertEqual(cl3.clamped_hit_fraction(self._record()), 0.5)
        self.assertIsNone(
            cl3.clamped_hit_fraction(self._record(expected_reusable_prefix_tokens=0))
        )
        stats = cl3.ttft_stats([over, self._record()])
        self.assertAlmostEqual(stats["clamped_hit_fraction_mean"], 0.75)
        self.assertEqual(stats["partial_or_full_miss_requests"], 1)

    def test_denominators_select_different_request_sets(self):
        records = [
            self._record(phase="workflow"),
            self._record(phase="pressure_replay"),
            self._record(phase="fill", expected_reusable_prefix_tokens=0),
        ]
        self.assertEqual(len(cl3.select(records, "workflow_only")), 1)
        self.assertEqual(len(cl3.select(records, "all_reusable")), 2)

    def test_s2_is_labelled_belady_style(self):
        self.assertIn("Belady-style", cl3.POLICY_LABELS["belady"])
        self.assertNotIn("optimum", cl3.POLICY_LABELS["belady"])


class TestP64OutcomeLabelling(unittest.TestCase):
    """An exact-cache miss must not be reported as a dense fallback.

    A profile with no approximate metadata never enters the recovery path,
    so a short prefix there is an ordinary cache miss. Labelling it
    dense_fallback previously caused exact-only misses to be cited as
    evidence that the approximate fallback path had executed.
    """

    def test_runner_distinguishes_cache_miss_from_dense_fallback(self):
        source = (
            Path(__file__).resolve().parents[4]
            / "benchmark/approx_kv/run_p6_4_capacity_pilot.py"
        ).read_text()

        self.assertIn('else "exact_cache_miss"', source)
        self.assertIn('"exact_cache_miss",', source)

        # fallback_reachable must still demand a real reservation failure.
        self.assertIn(
            'row["outcome"] == "dense_fallback" and row["reservation_failures"] > 0',
            source,
        )

        # the no-metadata outcome assignment must not resurrect the old label
        tail = source.split("request_outcomes = {}", 1)[1]
        assignment = tail.split("outcome = (", 1)[1].split(")", 1)[0]
        self.assertIn("exact_cache_miss", assignment)
        self.assertNotIn("dense_fallback", assignment)


if __name__ == "__main__":
    unittest.main()
