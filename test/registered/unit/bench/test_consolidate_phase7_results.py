from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-c-test-cpu")

from benchmark.approx_kv.consolidate_phase7_results import (
    AUTHORIZED_DESIGN_SHA256,
    AUTHORIZED_MANIFEST_SHA256,
    ConsolidationError,
    aggregate_a8,
    aggregate_w_cross_policy,
    attach_self_hash,
    canonical_sha256,
    central_run_durations,
    expected_execution_plan,
    file_sha256,
    validate_authorized_manifest,
    validate_r4_contract,
    validate_raw_artifact,
    validate_reset_invariants,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = (
    REPO_ROOT / "benchmark/approx_kv/results/phase7/phase7-primary-manifest.json"
)


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def fake_a8(setting_id: str, body: int, rho: float, speedup: float) -> dict:
    canary = {
        "complete_8_tokens": True,
        "matched": True,
        "engineering_status": "valid",
    }
    arms = {
        "D0": {"same_context_canary": None},
        "E0": {"same_context_canary": None},
        "R0": {"same_context_canary": canary},
    }
    return {
        "setting_id": setting_id,
        "setting": {
            "body_tokens": body,
            "rho_logical_demand": rho,
            "chunked_prefill_size": 4096,
        },
        "restart_index": 0,
        "formal": [
            {
                "arms": copy.deepcopy(arms),
                "amortization": {
                    "full_setup_break_even_observed_N": ">8/not_observed",
                    "incremental_setup_break_even_observed_N": ">8/not_observed",
                },
            },
            {
                "arms": copy.deepcopy(arms),
                "amortization": {
                    "full_setup_break_even_observed_N": ">8/not_observed",
                    "incremental_setup_break_even_observed_N": ">8/not_observed",
                },
            },
        ],
        "summary": {
            "paired_target_request_path_median_speedup": speedup,
            "amortization_median_speedup": {
                str(n): {
                    "full_setup": speedup / 2,
                    "incremental_setup": speedup / 1.5,
                }
                for n in (1, 2, 4, 8)
            },
        },
        "outcome": {
            "counts": {
                "dense_no_reuse_baseline": 16,
                "exact_gpu_hit": 16,
                "approximate_gpu_recovery": 16,
                "ordinary_exact_cache_miss": 0,
                "host_demand_load": 0,
                "approximate_recovery_failed_dense": 0,
            }
        },
    }


def fake_w_restart(
    *,
    rho: float,
    policy: str,
    restart: int,
    all_mean: float,
    workflow_mean: float,
    all_wall: float,
    workflow_wall: float,
    all_p95: float,
    workflow_p95: float,
    misses: int,
    peak: float,
) -> dict:
    return {
        "policy": policy,
        "rho_logical_demand": rho,
        "restart_index": restart,
        "aggregate_two_formals": {
            "denominators": {
                "all_reusable": {
                    "r0": {
                        "ttft_mean_ms": all_mean,
                        "wall_clock_ms": all_wall,
                        "ttft_p95_ms": all_p95,
                        "partial_or_full_miss_requests": misses,
                    }
                },
                "workflow_only": {
                    "r0": {
                        "ttft_mean_ms": workflow_mean,
                        "wall_clock_ms": workflow_wall,
                        "ttft_p95_ms": workflow_p95,
                        "partial_or_full_miss_requests": 0,
                    }
                },
            },
            "r0_peak_device_bytes": peak,
        },
    }


class TestPhase7ConsolidatorPureFunctions(unittest.TestCase):
    def test_expected_execution_set(self):
        manifest = load_manifest()
        validate_authorized_manifest(manifest)
        plan = expected_execution_plan(manifest)
        self.assertEqual(
            manifest["preregistered_manifest_sha256"], AUTHORIZED_MANIFEST_SHA256
        )
        self.assertEqual(manifest["design_payload_sha256"], AUTHORIZED_DESIGN_SHA256)
        self.assertEqual(len(plan["executed"]), 22)
        self.assertEqual(len(plan["wave0_required"]), 2)
        self.assertEqual(len(plan["a8_primary_restart0"]), 4)
        self.assertEqual(len(plan["a8_primary_supplements_skipped_es_r0_mde"]), 8)
        self.assertEqual(len(plan["chunk1024_sensitivity"]), 2)
        self.assertEqual(len(plan["w_main"]), 12)
        self.assertEqual(len(plan["r4_diagnostic"]), 2)
        self.assertEqual(
            plan["rho3_conditional_disabled"],
            [("p6delta-s4-rho3-chunk4096", 0)],
        )

    def test_a8_aggregation_is_negative(self):
        raws = [
            fake_a8("a", 1024, 1.5, 0.77),
            fake_a8("b", 1024, 2.0, 0.78),
            fake_a8("c", 2048, 1.5, 0.93),
            fake_a8("d", 2048, 2.0, 0.94),
        ]
        result = aggregate_a8(raws)
        self.assertEqual(result["mechanism_status"], "NEGATIVE")
        self.assertFalse(result["headline_speedup_allowed"])
        self.assertEqual(result["primary_supplement_disposition"]["skipped_starts"], 8)
        self.assertEqual(
            [row["request_path_median_speedup"] for row in result["table"]],
            [0.77, 0.78, 0.93, 0.94],
        )

    def test_w_cross_policy_uses_paired_restart_median(self):
        rows = []
        for rho in (1.5, 2.0):
            for restart, ratio in enumerate((1.0, 1.2, 1.1)):
                rows.append(
                    fake_w_restart(
                        rho=rho,
                        policy="lru",
                        restart=restart,
                        all_mean=100 * ratio,
                        workflow_mean=200 * ratio,
                        all_wall=1000 * ratio,
                        workflow_wall=2000 * ratio,
                        all_p95=500,
                        workflow_p95=600,
                        misses=20,
                        peak=1000,
                    )
                )
                rows.append(
                    fake_w_restart(
                        rho=rho,
                        policy="hierarchical",
                        restart=restart,
                        all_mean=100,
                        workflow_mean=200,
                        all_wall=1000,
                        workflow_wall=2000,
                        all_p95=450,
                        workflow_p95=540,
                        misses=15,
                        peak=1010,
                    )
                )
        result = aggregate_w_cross_policy(rows)
        for rho in ("1.5", "2.0"):
            median = result[rho]["median_across_restarts"]
            self.assertAlmostEqual(
                median["all_reusable"]["mean_speedup_s0_over_s4"], 1.1
            )
            self.assertAlmostEqual(
                median["workflow_only"]["mean_speedup_s0_over_s4"], 1.1
            )
            self.assertEqual(median["miss_delta_s4_minus_s0"], -5)
            self.assertAlmostEqual(median["peak_ratio_s4_over_s0"], 1.01)

    def test_central_duration_sums_run_intervals(self):
        expected = {("p7-a8-test", 0), ("p7-w-test", 1)}
        events = [
            {
                "run_id": "run-a",
                "setting_id": "p7-a8-test",
                "restart_index": 0,
                "manifest_sha256": AUTHORIZED_MANIFEST_SHA256,
                "phase": "Phase7-ceiling",
                "status": "running",
                "timestamp": "2026-07-28T10:00:00+00:00",
            },
            {
                "run_id": "run-a",
                "setting_id": "p7-a8-test",
                "restart_index": 0,
                "raw_sha256": "a" * 64,
                "phase": "Phase7-ceiling",
                "status": "completed",
                "timestamp": "2026-07-28T10:30:00+00:00",
            },
            {
                "run_id": "run-b",
                "setting_id": "p7-w-test",
                "restart_index": 1,
                "manifest_sha256": AUTHORIZED_MANIFEST_SHA256,
                "phase": "Phase7-scheduler",
                "status": "running",
                "timestamp": "2026-07-28T11:00:00+00:00",
            },
            {
                "run_id": "run-b",
                "setting_id": "p7-w-test",
                "restart_index": 1,
                "raw_sha256": "b" * 64,
                "phase": "Phase7-scheduler",
                "status": "completed",
                "timestamp": "2026-07-28T11:15:00+00:00",
            },
        ]
        result = central_run_durations(events, expected)
        self.assertEqual(result["total_elapsed_seconds"], 2700)
        self.assertEqual(result["total_elapsed_gpu_equivalent_hours"], 0.75)

    def test_invalid_and_reset_failure_are_rejected(self):
        manifest = load_manifest()
        settings = {row["setting_id"]: row for row in manifest["settings"]}
        manifest_file_hash = file_sha256(MANIFEST_PATH)
        staging = REPO_ROOT / ".phase7-consolidator-test-staging"
        raw_path = staging / "raw/p7-a8-r0-body1024-rho1.5-r0.json"
        raw = {
            "raw_sha256": "",
            "manifest_revision": 12,
            "preregistered_manifest_sha256": AUTHORIZED_MANIFEST_SHA256,
            "manifest_file_sha256": manifest_file_hash,
            "phase": "Phase7-ceiling",
            "setting_id": "p7-a8-r0-body1024-rho1.5",
            "restart_index": 0,
            "setting": settings["p7-a8-r0-body1024-rho1.5"],
            "runner": {
                "module": "benchmark.approx_kv.run_p7_ceiling",
                "path": "benchmark/approx_kv/run_p7_ceiling.py",
                "sha256": manifest["runners"]["ceiling"]["sha256"],
            },
            "status": "invalid",
        }
        raw["raw_sha256"] = canonical_sha256(raw, "raw_sha256")
        with self.assertRaisesRegex(ConsolidationError, "forbidden status"):
            validate_raw_artifact(
                raw,
                path=raw_path,
                staging_dir=staging,
                manifest=manifest,
                manifest_file_hash=manifest_file_hash,
                settings=settings,
            )

        with self.assertRaisesRegex(ConsolidationError, "startup reset"):
            validate_reset_invariants(
                {
                    "phase": "Phase7-ceiling",
                    "reset": {"startup": {"passed": False}},
                    "formal": [],
                }
            )

    def test_r4_semantics_are_synthetic_and_policy_specific(self):
        contract = {
            "arm_label": "R4-like-5x",
            "claim": "synthetic_footprint_and_victim_diagnostic_only_not_kvcomm",
            "performance_ranking_enabled": False,
        }

        def raw(policy: str) -> dict:
            available = policy == "lru"
            return {
                "performance_contract": contract,
                "setting": {"policy": policy},
                "status": "valid" if available else "inconclusive",
                "formal": [
                    {
                        "arms": {
                            "R4-like-5x": {
                                "setup": {
                                    "profile": "r4_like",
                                    "representation_multiplicity": 5,
                                    "representation_kinds": [
                                        "canonical_base",
                                        "anchor",
                                        "delta",
                                        "anchor",
                                        "delta",
                                    ],
                                    "registration_failed": not available,
                                },
                                "diagnostic_status": (
                                    "available"
                                    if available
                                    else "diagnostic_unavailable"
                                ),
                                "records": [None] * 61 if available else [],
                            }
                        }
                    }
                    for _ in range(2)
                ],
            }

        validate_r4_contract(raw("lru"))
        validate_r4_contract(raw("hierarchical"))
        drifted = raw("hierarchical")
        drifted["formal"][0]["arms"]["R4-like-5x"]["diagnostic_status"] = "available"
        with self.assertRaisesRegex(ConsolidationError, "diagnostic semantics"):
            validate_r4_contract(drifted)

    def test_self_hash_is_deterministic(self):
        left = {"z": [3, 2, 1], "a": {"b": 2, "a": 1}}
        right = {"a": {"a": 1, "b": 2}, "z": [3, 2, 1]}
        self.assertEqual(
            canonical_sha256(left, "artifact_sha256"),
            canonical_sha256(right, "artifact_sha256"),
        )
        hashed = attach_self_hash(copy.deepcopy(left), "artifact_sha256")
        self.assertEqual(
            hashed["artifact_sha256"],
            canonical_sha256(hashed, "artifact_sha256"),
        )


if __name__ == "__main__":
    unittest.main()
