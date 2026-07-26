from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")

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
