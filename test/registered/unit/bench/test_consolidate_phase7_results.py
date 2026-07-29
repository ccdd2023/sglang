from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-c-test-cpu")

from benchmark.approx_kv import consolidate_phase7_results as consolidator
from benchmark.approx_kv.consolidate_phase7_results import (
    AUTHORIZED_DESIGN_SHA256,
    AUTHORIZED_MANIFEST_SHA256,
    ConsolidationError,
    _validate_correction_execution_provenance,
    aggregate_a8,
    aggregate_w_cross_policy,
    attach_self_hash,
    build_compact_artifact,
    canonical_sha256,
    capacity_terminal_reason_correction_required,
    central_run_durations,
    expected_execution_plan,
    file_sha256,
    load_evidence_correction,
    require_capacity_terminal_reason_correction,
    serialized_manifest_sha256,
    summarize_r4,
    summarize_w_victim_footprint,
    summarize_wave0,
    validate_authorized_manifest,
    validate_capacity_terminal_reason_contract,
    validate_capacity_terminal_reason_correction,
    validate_correction_manifest,
    validate_r4_contract,
    validate_raw_artifact,
    validate_reset_invariants,
)
from benchmark.approx_kv.phase7.correction import (
    BASE_MANIFEST_PATH,
    CAPACITY_CORRECTION_CPU_EVIDENCE_PATH,
    CAPACITY_CORRECTION_MANIFEST_PATH,
    CAPACITY_CORRECTION_REVIEW_PATH,
    CAPACITY_RUNNER_PATH,
    build_authorized_capacity_correction_manifest,
    build_pinned_capacity_correction_manifest,
    correction_manifest_payload_sha256,
)
from benchmark.approx_kv.phase7.correction_review import build_correction_review

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = (
    REPO_ROOT / "benchmark/approx_kv/results/phase7/phase7-primary-manifest.json"
)


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def correction_provenance_fixture() -> tuple[dict, dict, dict, str]:
    manifest_path = CAPACITY_CORRECTION_MANIFEST_PATH
    runner_path = CAPACITY_RUNNER_PATH
    runner_entry = {"path": runner_path, "sha256": "3" * 64}
    manifest_file_hash = "7" * 64
    review_file_hash = "4" * 64
    cpu_file_hash = "5" * 64
    allowlist = [
        "benchmark/approx_kv/results/phase7/RESULT_MANIFEST.json",
        manifest_path,
        CAPACITY_CORRECTION_REVIEW_PATH,
        CAPACITY_CORRECTION_CPU_EVIDENCE_PATH,
    ]
    manifest = {
        "correction_pinned_implementation_sha": "1" * 40,
        "correction_pinned_tree_sha": "2" * 40,
        "capacity_runner_sha256": runner_entry["sha256"],
        "post_pin_allowlist": allowlist,
        "review_evidence": {
            "artifact_path": CAPACITY_CORRECTION_REVIEW_PATH,
            "file_sha256": review_file_hash,
        },
        "capacity_cpu_evidence": {
            "path": CAPACITY_CORRECTION_CPU_EVIDENCE_PATH,
            "file_sha256": cpu_file_hash,
        },
    }
    envelope = {
        "evidence_correction_scope": "capacity_terminal_reason",
        "execution_kind": "capacity_correction",
        "primary_execution_envelope": False,
        "pinned_is_ancestor_of_execution_head": True,
        "worktree_clean": True,
        "worktree_status_entries": [],
        "pinned_source_git_sha": manifest["correction_pinned_implementation_sha"],
        "pinned_source_tree_sha": manifest["correction_pinned_tree_sha"],
        "execution_head_git_sha": "8" * 40,
        "execution_head_tree_sha": "9" * 40,
        "post_pin_envelope_allowlist": allowlist,
        "post_pin_changed_paths": [
            manifest_path,
            CAPACITY_CORRECTION_REVIEW_PATH,
            CAPACITY_CORRECTION_CPU_EVIDENCE_PATH,
        ],
        "post_pin_envelope_sha256": {
            allowlist[0]: "6" * 64,
            manifest_path: manifest_file_hash,
            CAPACITY_CORRECTION_REVIEW_PATH: review_file_hash,
            CAPACITY_CORRECTION_CPU_EVIDENCE_PATH: cpu_file_hash,
        },
        "correction_runner_path": runner_path,
        "correction_runner_sha256": runner_entry["sha256"],
        "pinned_runner_sha256": runner_entry["sha256"],
    }
    source = {
        "source_git_sha": manifest["correction_pinned_implementation_sha"],
        "source_tree_sha": manifest["correction_pinned_tree_sha"],
        "execution_head_git_sha": envelope["execution_head_git_sha"],
        "execution_head_tree_sha": envelope["execution_head_tree_sha"],
        "source_binding": "dedicated_capacity_correction_pin",
    }
    implementation = {
        "correction_pinned_implementation_sha": manifest[
            "correction_pinned_implementation_sha"
        ],
        "correction_pinned_tree_sha": manifest["correction_pinned_tree_sha"],
        "capacity_runner_sha256": manifest["capacity_runner_sha256"],
        "post_pin_allowlist": allowlist,
    }
    raw = {
        "execution_envelope": envelope,
        "runner": {
            "path": runner_path,
            "sha256": runner_entry["sha256"],
        },
        "provenance": {
            "manifest_path": f"/repo/{manifest_path}",
            "manifest_file_sha256": manifest_file_hash,
            "runner_sha256": runner_entry["sha256"],
            "implementation": implementation,
            "source": source,
        },
        "source_git_sha": manifest["correction_pinned_implementation_sha"],
        "source_tree_sha": manifest["correction_pinned_tree_sha"],
        "manifest_file_sha256": manifest_file_hash,
        "correction_manifest_file_sha256": manifest_file_hash,
    }
    return manifest, runner_entry, raw, manifest_file_hash


def correction_manifest_fixture() -> tuple[dict, dict]:
    primary = load_manifest()
    runner_sha = "3" * 64
    cpu_summary = {
        "status": "passed",
        "path": CAPACITY_CORRECTION_CPU_EVIDENCE_PATH,
        "file_sha256": "4" * 64,
        "artifact_sha256": "5" * 64,
        "runner_sha256": runner_sha,
        "image_digest": primary["environment"]["image_digest"],
        "command": primary["runners"]["capacity_pilot"]["required_cpu_test"],
        "exit_code": 0,
        "summary_line": "42 passed in 1.00s",
        "passed_count": 42,
        "subtests": {"passed_count": 0, "names": []},
        "timestamp": "2026-07-28T16:00:00-07:00",
    }
    pinned = build_pinned_capacity_correction_manifest(
        base_manifest=primary,
        base_manifest_path=Path(BASE_MANIFEST_PATH),
        base_manifest_file_sha256="6" * 64,
        original_raw_file_sha256="7" * 64,
        correction_manifest_revision=1,
        correction_pinned_implementation_sha="8" * 40,
        correction_pinned_tree_sha="9" * 40,
        capacity_runner_sha256=runner_sha,
        capacity_cpu_evidence=cpu_summary,
        manifest_generation_sha="a" * 40,
        manifest_generation_tree_sha="b" * 40,
    )
    review = build_correction_review(
        reviewer="Claude Opus 5",
        model="claude-opus-5",
        verdict="PASS",
        open_p0=0,
        open_p1=0,
        reviewed_correction_manifest_revision=1,
        reviewed_correction_manifest_sha256=pinned["correction_manifest_sha256"],
        base_manifest_revision=pinned["base_manifest_revision"],
        base_manifest_self_sha256=pinned["base_manifest_self_sha256"],
        base_manifest_design_sha256=pinned["base_manifest_design_sha256"],
        base_manifest_path=pinned["base_manifest_path"],
        reviewed_correction_pinned_implementation_sha=pinned[
            "correction_pinned_implementation_sha"
        ],
        reviewed_correction_pinned_tree_sha=pinned["correction_pinned_tree_sha"],
        capacity_runner_sha256=runner_sha,
        original_raw_sha256=pinned["original_raw_sha256"],
        scope=pinned["scope"],
        allowed_setting=pinned["allowed_setting"],
        restart=pinned["restart"],
        findings=[],
        disposition="all closed",
        timestamp="2026-07-28T16:05:00-07:00",
    )
    with tempfile.TemporaryDirectory(prefix="phase7-correction-review-") as raw:
        root = Path(raw)
        review_path = root / CAPACITY_CORRECTION_REVIEW_PATH
        review_path.parent.mkdir(parents=True)
        review_path.write_text(json.dumps(review, indent=2) + "\n")
        correction = build_authorized_capacity_correction_manifest(
            reviewed_manifest=pinned,
            review=review,
            review_path=review_path,
            repo_root=root,
            correction_manifest_revision=2,
            manifest_generation_sha="c" * 40,
            manifest_generation_tree_sha="d" * 40,
        )
    return primary, correction


def fake_a8(setting_id: str, body: int, rho: float, speedup: float) -> dict:
    canary = {
        "complete_8_tokens": True,
        "matched": True,
        "engineering_status": "valid",
        "dense_output_ids": [198] * 8,
        "recovery_output_ids": [198] * 8,
    }

    def arm(name: str) -> dict:
        cached = {"D0": 0, "E0": 64, "R0": body + 64}[name]
        target_ms = {"D0": 100.0, "E0": 90.0, "R0": 120.0}[name]
        request_ms = {"D0": 100.0, "E0": 95.0, "R0": 130.0}[name]
        return {
            "same_context_canary": canary if name == "R0" else None,
            "targets": [
                {
                    "cached_tokens": cached,
                    "target_only_ms": target_ms,
                    "request_path_ms": request_ms,
                }
                for _ in range(8)
            ],
            "ledger": {
                "full_lifecycle_ms": {
                    "D0": 1000.0,
                    "E0": 900.0,
                    "R0": 1200.0,
                }[name]
            },
        }

    arms = {name: arm(name) for name in ("D0", "E0", "R0")}
    return {
        "setting_id": setting_id,
        "setting": {
            "body_tokens": body,
            "rho_logical_demand": rho,
            "chunked_prefill_size": 4096,
        },
        "restart_index": 0,
        "ledger": {"setup": {"server_cold_start_ms": 49000.0}},
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
                        "ttft_p50_ms": all_mean,
                        "ttft_p95_ms": all_p95,
                        "partial_or_full_miss_requests": misses,
                    }
                },
                "workflow_only": {
                    "r0": {
                        "ttft_mean_ms": workflow_mean,
                        "wall_clock_ms": workflow_wall,
                        "ttft_p50_ms": workflow_mean,
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

    def test_correction_manifest_pins_new_code_and_strict_allowlist(self):
        primary, correction = correction_manifest_fixture()
        validate_correction_manifest(
            correction,
            primary_manifest=primary,
        )
        self.assertEqual(
            serialized_manifest_sha256(correction),
            hashlib.sha256(
                (json.dumps(correction, indent=2) + "\n").encode("utf-8")
            ).hexdigest(),
        )

        stale_pin = copy.deepcopy(correction)
        stale_pin["correction_pinned_implementation_sha"] = primary["implementation"][
            "phase7_pinned_implementation_sha"
        ]
        stale_pin["correction_manifest_sha256"] = correction_manifest_payload_sha256(
            stale_pin
        )
        with self.assertRaisesRegex(ConsolidationError, "new implementation"):
            validate_correction_manifest(
                stale_pin,
                primary_manifest=primary,
            )

        broad_allowlist = copy.deepcopy(correction)
        broad_allowlist["post_pin_allowlist"].append("python/sglang/srt/unsafe.py")
        broad_allowlist["correction_manifest_sha256"] = (
            correction_manifest_payload_sha256(broad_allowlist)
        )
        with self.assertRaisesRegex(ConsolidationError, "allowlist"):
            validate_correction_manifest(
                broad_allowlist,
                primary_manifest=primary,
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
        self.assertEqual(result["independent_replicate_unit"], "server_restart")
        self.assertEqual(result["n_per_setting"], 1)
        self.assertFalse(result["three_restart_range"]["available"])
        self.assertIn("target_only", result["table"][0]["primary_views"])
        self.assertIn("request_path", result["table"][0]["primary_views"])
        self.assertTrue(
            result["table"][0]["primary_views"]["request_path"][
                "is_preregistered_mde_metric"
            ]
        )
        self.assertFalse(
            result["table"][0]["primary_views"]["target_only"][
                "is_preregistered_mde_metric"
            ]
        )
        self.assertEqual(
            result["table"][0]["canary"]["repeat_values"][0]["distinct_output_tokens"],
            1,
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
            self.assertEqual(median["miss_delta_s4_minus_s0"]["all_reusable"], -5)
            self.assertEqual(median["miss_delta_s4_minus_s0"]["workflow_only"], 0)
            self.assertAlmostEqual(median["peak_ratio_s4_over_s0"], 1.01)
            self.assertEqual(
                result[rho]["comparison_design"],
                "seed-matched_non_adjacent_restart_comparison",
            )
            self.assertEqual(
                median["all_reusable"]["ratio_of_marginal_p95s_direction"],
                "s4_over_s0",
            )

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
        self.assertEqual(result["wall_clock_span_hours"], 1.25)
        self.assertIn("inter-run gaps", result["sum_of_run_intervals_exclusion_note"])
        correction = central_run_durations(
            events,
            expected,
            excluded_run_classes=("primary execution runs",),
        )
        note = correction["sum_of_run_intervals_exclusion_note"]
        self.assertIn("primary execution runs", note)
        self.assertNotIn("evidence-correction runs", note)

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

    def test_capacity_correction_requires_40_direct_exclusive_reasons(self):
        primary, manifest = correction_manifest_fixture()
        original = {
            "raw_sha256": manifest["original_raw_sha256"],
            "setting_id": "p6delta-s0-rho2-chunk4096",
            "restart_index": 0,
            "setting": next(
                setting
                for setting in primary["settings"]
                if setting["setting_id"] == "p6delta-s0-rho2-chunk4096"
            ),
            "outcome": {
                "counts": {"approximate_recovery_failed_dense": 40},
                "terminal_reason_counts": {},
            },
        }
        self.assertTrue(capacity_terminal_reason_correction_required(original))
        with self.assertRaisesRegex(ConsolidationError, "--correction-dir"):
            require_capacity_terminal_reason_correction(
                original=original,
                correction_dir=None,
            )
        require_capacity_terminal_reason_correction(
            original=original,
            correction_dir=Path("corrections"),
        )
        observation = {
            "verification": "direct",
            "value_unit": "tokens",
            "raw": {"store_miss": 1024.0},
            "mapped": {
                "cross_store_reservation_failed": None,
                "device_allocation_failed": None,
                "unsupported": 1024.0,
                "registration_failed": None,
                "prefix_gap": None,
            },
            "mapped_from": {
                "cross_store_reservation_failed": [],
                "device_allocation_failed": [],
                "unsupported": ["store_miss"],
                "registration_failed": [],
                "prefix_gap": [],
            },
            "unmapped_raw_reasons": {},
        }
        replay = {
            "outcome": "dense_fallback",
            "terminal_reason": "unsupported",
            "terminal_reason_verification": "direct",
            "terminal_reason_valid": True,
            "terminal_reason_observations": observation,
        }
        correction = {
            "correction": {
                "scope": "capacity_terminal_reason",
                "original_raw_sha256": original["raw_sha256"],
                "setting_id": original["setting_id"],
                "restart_index": 0,
            },
            "phase": "Phase7-capacity",
            "setting_id": original["setting_id"],
            "restart_index": 0,
            "setting": original["setting"],
            "base_manifest_revision": manifest["base_manifest_revision"],
            "base_manifest_self_sha256": manifest["base_manifest_self_sha256"],
            "base_manifest_design_sha256": manifest["base_manifest_design_sha256"],
            "correction_manifest_revision": manifest["correction_manifest_revision"],
            "correction_manifest_sha256": manifest["correction_manifest_sha256"],
            "design_payload_sha256": manifest["base_manifest_design_sha256"],
            "plan": manifest["plan"],
            "status": "inconclusive",
            "outcome": {
                "taxonomy": manifest["outcome_taxonomy"],
                "counts": {
                    "dense_no_reuse_baseline": 0,
                    "exact_gpu_hit": 1,
                    "approximate_gpu_recovery": 0,
                    "approximate_recovery_failed_dense": 40,
                    "ordinary_exact_cache_miss": 9,
                    "host_demand_load": 0,
                },
                "exclusive_terminal_reasons": manifest["exclusive_terminal_reasons"],
                "terminal_reason_counts": {
                    "cross_store_reservation_failed": 0,
                    "device_allocation_failed": 0,
                    "unsupported": 40,
                    "registration_failed": 0,
                    "prefix_gap": 0,
                },
            },
            "cells": [
                {
                    "profiles": [
                        {
                            "profile": "exact_only",
                            "formal": [
                                {
                                    "replay": [
                                        {"outcome": "exact_gpu_hit"},
                                        *[
                                            {"outcome": "exact_cache_miss"}
                                            for _ in range(4)
                                        ],
                                    ]
                                },
                                {
                                    "replay": [
                                        {"outcome": "exact_cache_miss"}
                                        for _ in range(5)
                                    ]
                                },
                            ],
                        },
                        *[
                            {
                                "profile": name,
                                "formal": [
                                    {
                                        "replay": [
                                            copy.deepcopy(replay) for _ in range(5)
                                        ]
                                    },
                                    {
                                        "replay": [
                                            copy.deepcopy(replay) for _ in range(5)
                                        ]
                                    },
                                ],
                            }
                            for name in (
                                "r0_like",
                                "r1_like_k32",
                                "r2_like",
                                "r4_like",
                            )
                        ],
                    ]
                }
            ],
        }
        with patch.object(
            consolidator,
            "validate_capacity_terminal_reason_contract",
            wraps=validate_capacity_terminal_reason_contract,
        ) as contract:
            reason_counts = validate_capacity_terminal_reason_correction(
                correction,
                original=original,
                manifest=manifest,
            )
        contract.assert_called_once_with(correction)
        self.assertEqual(reason_counts["unsupported"], 40)
        validate_capacity_terminal_reason_contract(correction)
        wrong_binding = copy.deepcopy(correction)
        wrong_binding["correction"]["original_raw_sha256"] = "b" * 64
        with self.assertRaisesRegex(ConsolidationError, "binding mismatch"):
            validate_capacity_terminal_reason_correction(
                wrong_binding,
                original=original,
                manifest=manifest,
            )
        wrong_setting = copy.deepcopy(correction)
        wrong_setting["correction"]["setting_id"] = "p6delta-s4-rho2-chunk4096"
        with self.assertRaisesRegex(ConsolidationError, "binding mismatch"):
            validate_capacity_terminal_reason_correction(
                wrong_setting,
                original=original,
                manifest=manifest,
            )
        wrong_design = copy.deepcopy(correction)
        wrong_design["design_payload_sha256"] = "b" * 64
        with self.assertRaisesRegex(ConsolidationError, "design"):
            validate_capacity_terminal_reason_correction(
                wrong_design,
                original=original,
                manifest=manifest,
            )
        missing = copy.deepcopy(correction)
        missing["cells"][0]["profiles"][1]["formal"][0]["replay"][0][
            "terminal_reason_observations"
        ]["mapped"]["unsupported"] = None
        with self.assertRaisesRegex(ConsolidationError, "one direct exclusive"):
            validate_capacity_terminal_reason_correction(
                missing,
                original=original,
                manifest=manifest,
            )
        with self.assertRaisesRegex(ConsolidationError, "mapped terminal reason"):
            validate_capacity_terminal_reason_contract(missing)
        multiple = copy.deepcopy(correction)
        multiple["cells"][0]["profiles"][1]["formal"][0]["replay"][0][
            "terminal_reason_observations"
        ]["mapped"]["prefix_gap"] = 1.0
        with self.assertRaisesRegex(ConsolidationError, "one direct exclusive"):
            validate_capacity_terminal_reason_correction(
                multiple,
                original=original,
                manifest=manifest,
            )

        with self.assertRaisesRegex(ConsolidationError, "startup reset"):
            validate_reset_invariants(
                {
                    "phase": "Phase7-ceiling",
                    "reset": {"startup": {"passed": False}},
                    "formal": [],
                }
            )

    def test_correction_loader_rejects_more_than_one_supplementary_raw(self):
        with tempfile.TemporaryDirectory(prefix="phase7-correction-dir-") as raw:
            correction_dir = Path(raw)
            raw_dir = correction_dir / "raw"
            raw_dir.mkdir()
            (raw_dir / "first.json").write_text("{}\n")
            (raw_dir / "second.json").write_text("{}\n")
            with self.assertRaisesRegex(
                ConsolidationError,
                "exactly one S0 supplementary",
            ):
                load_evidence_correction(
                    correction_dir=correction_dir,
                    correction_manifest_path=correction_dir / "manifest.json",
                    original={},
                    original_file_sha256=None,
                    manifest=load_manifest(),
                    verify_git=False,
                )

    def test_correction_loader_accepts_one_s0_supplementary_artifact(self):
        with tempfile.TemporaryDirectory(prefix="phase7-correction-dir-") as raw:
            root = Path(raw)
            correction_dir = root / "correction"
            raw_dir = correction_dir / "raw"
            raw_dir.mkdir(parents=True)
            binding = {
                "scope": "capacity_terminal_reason",
                "original_raw_sha256": "0" * 64,
                "setting_id": "p6delta-s0-rho2-chunk4096",
                "restart_index": 0,
            }
            correction = {
                "correction": binding,
                "setting_id": binding["setting_id"],
                "restart_index": 0,
                "raw_sha256": "1" * 64,
                "run_id": "correction-run",
                "phase": "Phase7-capacity",
            }
            correction_path = raw_dir / "correction.json"
            correction_path.write_text(json.dumps(correction))
            manifest_path = root / CAPACITY_CORRECTION_MANIFEST_PATH
            manifest_path.parent.mkdir(parents=True)
            review_path = root / CAPACITY_CORRECTION_REVIEW_PATH
            review_path.write_text("{}\n")
            correction_manifest = {
                "correction_manifest_sha256": "2" * 64,
                "review_evidence": {
                    "artifact_path": CAPACITY_CORRECTION_REVIEW_PATH,
                    "file_sha256": file_sha256(review_path),
                },
                "capacity_cpu_evidence": {
                    "path": CAPACITY_CORRECTION_CPU_EVIDENCE_PATH,
                    "file_sha256": "3" * 64,
                },
            }
            manifest_path.write_text(json.dumps(correction_manifest))
            events = [
                {
                    "run_id": "correction-run",
                    "phase": "Phase7-capacity",
                    "status": "running",
                    "setting_id": binding["setting_id"],
                    "restart_index": 0,
                    "manifest_sha256": "2" * 64,
                    "correction": binding,
                    "timestamp": "2026-07-28T16:00:00+00:00",
                },
                {
                    "run_id": "correction-run",
                    "phase": "Phase7-capacity",
                    "status": "completed",
                    "setting_id": binding["setting_id"],
                    "restart_index": 0,
                    "raw_sha256": "1" * 64,
                    "correction": binding,
                    "timestamp": "2026-07-28T16:01:00+00:00",
                },
            ]
            (correction_dir / "phase7-runs.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in events)
            )
            with (
                patch.object(consolidator, "validate_correction_manifest"),
                patch.object(
                    consolidator,
                    "validate_correction_review_binding",
                ),
                patch.object(
                    consolidator,
                    "validate_correction_artifact",
                    return_value={
                        "raw_sha256": "1" * 64,
                        "file_sha256": file_sha256(correction_path),
                        "log_path": "logs/correction.log",
                        "log_sha256": "4" * 64,
                        "terminal_reason_counts": {"unsupported": 40},
                    },
                ),
            ):
                loaded = load_evidence_correction(
                    correction_dir=correction_dir,
                    correction_manifest_path=manifest_path,
                    original={},
                    manifest=load_manifest(),
                    verify_git=False,
                )
        self.assertEqual(loaded["artifact"]["setting_id"], binding["setting_id"])
        self.assertEqual(loaded["elapsed"]["total_elapsed_seconds"], 60.0)

    def test_capacity_raw_with_fallback_always_checks_terminal_reasons(self):
        manifest = load_manifest()
        settings = {setting["setting_id"]: setting for setting in manifest["settings"]}
        staging = REPO_ROOT / "benchmark/approx_kv/results/phase7"
        raw_path = staging / "raw/p6delta-s0-rho2-chunk4096-r0.json"
        raw = json.loads(raw_path.read_text())
        manifest_file_hash = file_sha256(MANIFEST_PATH)
        with self.assertRaisesRegex(ConsolidationError, "mapped terminal reason"):
            validate_raw_artifact(
                raw,
                path=raw_path,
                staging_dir=staging,
                manifest=manifest,
                manifest_file_hash=manifest_file_hash,
                settings=settings,
            )
        validate_raw_artifact(
            raw,
            path=raw_path,
            staging_dir=staging,
            manifest=manifest,
            manifest_file_hash=manifest_file_hash,
            settings=settings,
            allow_missing_capacity_terminal_reason_correction=True,
        )

    def test_correction_provenance_rejects_dirty_unlisted_or_wrong_runner(self):
        manifest, runner_entry, raw, manifest_file_hash = (
            correction_provenance_fixture()
        )
        _validate_correction_execution_provenance(
            raw,
            manifest,
            manifest_file_hash,
            runner_entry,
            "correction.json",
        )

        dirty = copy.deepcopy(raw)
        dirty["execution_envelope"]["worktree_clean"] = False
        dirty["execution_envelope"]["worktree_status_entries"] = [
            " M benchmark/approx_kv/phase7/common.py"
        ]
        with self.assertRaisesRegex(ConsolidationError, "must be clean"):
            _validate_correction_execution_provenance(
                dirty,
                manifest,
                manifest_file_hash,
                runner_entry,
                "correction.json",
            )

        unlisted = copy.deepcopy(raw)
        unlisted["execution_envelope"]["post_pin_changed_paths"].append(
            "benchmark/approx_kv/unlisted.py"
        )
        with self.assertRaisesRegex(ConsolidationError, "unlisted post-pin"):
            _validate_correction_execution_provenance(
                unlisted,
                manifest,
                manifest_file_hash,
                runner_entry,
                "correction.json",
            )

        core_drift = copy.deepcopy(raw)
        core_drift["execution_envelope"]["post_pin_changed_paths"].append(
            "python/sglang/srt/unsafe.py"
        )
        with self.assertRaisesRegex(ConsolidationError, "unlisted post-pin"):
            _validate_correction_execution_provenance(
                core_drift,
                manifest,
                manifest_file_hash,
                runner_entry,
                "correction.json",
            )

        wrong_runner = copy.deepcopy(raw)
        wrong_runner["execution_envelope"]["pinned_runner_sha256"] = "0" * 64
        with self.assertRaisesRegex(ConsolidationError, "runner binding mismatch"):
            _validate_correction_execution_provenance(
                wrong_runner,
                manifest,
                manifest_file_hash,
                runner_entry,
                "correction.json",
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

    def test_real_wave0_requires_correction_and_separates_status_axes(self):
        raw_dir = REPO_ROOT / "benchmark/approx_kv/results/phase7/raw"
        s0 = json.loads((raw_dir / "p6delta-s0-rho2-chunk4096-r0.json").read_text())
        s4 = json.loads((raw_dir / "p6delta-s4-rho2-chunk4096-r0.json").read_text())
        with self.assertRaisesRegex(ConsolidationError, "requires terminal-reason"):
            summarize_wave0([s0, s4])
        corrected = copy.deepcopy(s0)
        corrected["outcome"]["terminal_reason_counts"] = {
            "cross_store_reservation_failed": 0,
            "device_allocation_failed": 0,
            "unsupported": 40,
            "registration_failed": 0,
            "prefix_gap": 0,
        }
        summary = summarize_wave0([s0, s4], correction=corrected)
        self.assertEqual(
            summary["S0"]["formal_outcomes"]["approximate_gpu_recovery"], 0
        )
        self.assertEqual(summary["S0"]["formal_outcomes"]["dense_fallback"], 40)
        self.assertEqual(
            summary["S4"]["formal_outcomes"]["approximate_gpu_recovery"], 40
        )
        self.assertEqual(
            summary["S4"]["profile_registration_reachability"]["r4_like"][
                "reachability"
            ],
            "diagnostic_unavailable",
        )
        self.assertNotEqual(
            summary["S0"]["raw_status"],
            summary["S0"]["artifact_status"],
        )

    def test_real_r4_summary_retains_outcomes_and_accounting(self):
        raw_dir = REPO_ROOT / "benchmark/approx_kv/results/phase7/raw"
        raws = [
            json.loads((raw_dir / "p7-w-r4like-lru-rho2-r0.json").read_text()),
            json.loads((raw_dir / "p7-w-r4like-hierarchical-rho2-r0.json").read_text()),
        ]
        summary = summarize_r4(raws)
        self.assertFalse(summary["performance_ranking_enabled"])
        self.assertEqual(
            summary["policies"]["S0"]["recovery_success_fraction"]["count"], 12
        )
        self.assertEqual(summary["policies"]["S0"]["fallback_fraction"]["count"], 110)
        self.assertEqual(
            summary["policies"]["S4"]["diagnostic_statuses"],
            ["diagnostic_unavailable", "diagnostic_unavailable"],
        )
        self.assertTrue(summary["policies"]["S0"]["victim_sequence"])
        self.assertTrue(
            summary["policies"]["S0"]["victim_class_accounting"][
                "victim_evict_bytes_by_requester_provenance_object_kind"
            ]
        )

    def test_real_w_compact_and_victim_footprint_retain_primary_axes(self):
        raw_dir = REPO_ROOT / "benchmark/approx_kv/results/phase7/raw"
        path = raw_dir / "p7-w-r0-lru-rho2.0-r0.json"
        raw = json.loads(path.read_text())
        compact = build_compact_artifact(
            raw,
            raw_relative_path=f"raw/{path.name}",
            log_relative_path=f"logs/{path.stem}.log",
            raw_file_sha256=file_sha256(path),
            log_sha256=raw["server_log_sha256"],
        )
        formal = compact["key_metrics"]["formal"][0]["arms"]["R0"]
        self.assertIn("victim_sequence", formal)
        self.assertIn("memory_footprint_after", formal)
        self.assertIn("outcomes", formal)
        self.assertEqual(
            compact["key_metrics"]["paired_E0_R0"]["denominators"]["all_reusable"][
                "ratio_of_marginal_p95s_direction"
            ],
            "r0_over_e0",
        )
        footprint = summarize_w_victim_footprint([raw])
        self.assertTrue(footprint["primary_axis_after_a8_negative"])
        self.assertTrue(
            footprint["rows"][0]["arms"]["R0"][
                "victim_evict_bytes_by_requester_provenance_object_kind"
            ]
        )

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
