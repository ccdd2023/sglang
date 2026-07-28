from __future__ import annotations

import copy
import json
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")

from benchmark.approx_kv import run_p7_scheduler
from benchmark.approx_kv.build_phase7_manifest import (
    build_a8_workload,
    build_artifact_templates,
    build_inactive_counter_pins,
    build_settings,
    build_w_workload,
    design_payload_sha256,
)
from benchmark.approx_kv.phase6.schema import payload_sha256
from benchmark.approx_kv.phase7.common import (
    SCHEDULER_RUNNER,
    Phase7ContractError,
    arm_inactive_observations,
    build_arm_inactive_counter_assertion,
    cross_store_metrics,
    manifest_self_sha256,
    select_setting,
    w_workload,
)
from benchmark.approx_kv.phase7.statistics import (
    pair_scheduler_arms,
    performance_ranking_enabled,
    summarize_workflow_records,
)
from benchmark.approx_kv.run_p7_scheduler import (
    formal_arm_order,
    scheduler_performance_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = (
    REPO_ROOT / "benchmark/approx_kv/results/phase7/phase7-primary-manifest.json"
)
FINAL_REVIEW_REL = "benchmark/approx_kv/results/phase7/phase7-final-opus-review.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def revised_manifest() -> dict:
    manifest = load_manifest()
    manifest["manifest_revision"] = 6
    manifest["plan"]["version"] = "V7"
    manifest["settings"] = build_settings()
    p6_contract = json.loads(
        (
            REPO_ROOT / "benchmark/approx_kv/results/phase6/p6-0-contract.json"
        ).read_text()
    )
    manifest["workloads"]["A8"] = build_a8_workload()
    manifest["workloads"]["W"] = build_w_workload(p6_contract)
    manifest["server_template"]["plugin_env"].update(
        {
            "SGLANG_APPROX_KV_ALLOW_PERSISTENT_PINS": "1",
            "SGLANG_APPROX_KV_MAX_PERSISTENT_PINS": "16",
        }
    )
    manifest["r2_strategy"] = "disabled_not_comparable"
    manifest["conditional_resolution"]["CR-R2-ADAPTER"] = "disabled_not_comparable"
    manifest["conditional_user_authorization_recorded"] = True
    manifest["review_contract"] = {
        "final_opus_required": True,
        "reviewer": "Claude Opus 5 / Max Thinking / long context",
        "scope": "test",
        "pass_condition": "no open P0/P1 after accepted-feedback closure",
        "artifact_path": FINAL_REVIEW_REL,
        "authorization_activation": "test",
    }
    manifest["review_evidence"] = {
        "status": "pending",
        "artifact_path": FINAL_REVIEW_REL,
        "artifact_sha256": None,
        "round_summary": "synthetic pending V7 review",
    }
    manifest["artifact_templates"] = build_artifact_templates()
    manifest["inactive_counter_pins"] = build_inactive_counter_pins()
    manifest["design_payload_sha256"] = design_payload_sha256(manifest)
    manifest["preregistered_manifest_sha256"] = manifest_self_sha256(manifest)
    return manifest


def record(
    request: dict,
    *,
    repeat: int,
    ttft_ms: float,
    cached_tokens: int | None = None,
) -> dict:
    expected = 1088
    return {
        "sample_kind": "measured",
        "repeat": repeat,
        "request_index": request["request_index"],
        "phase": request["phase"],
        "role": request["role"],
        "object_id": request["object_id"],
        "ttft_ms": ttft_ms,
        "elapsed_ms": ttft_ms + 2.0,
        "cached_tokens": expected if cached_tokens is None else cached_tokens,
        "expected_reusable_prefix_tokens": expected,
    }


class TestPhase7WContract(unittest.TestCase):
    def test_authorization_is_loaded_before_staging_validation(self):
        args = SimpleNamespace(
            manifest=MANIFEST_PATH,
            setting_id="p7-w-r0-lru-rho1.5",
            restart_index=0,
            output=Path("/results/phase7/raw/test.json"),
            log=Path("/results/phase7/logs/test.log"),
            central_log=Path("/results/phase7/central.jsonl"),
        )
        with (
            patch.object(run_p7_scheduler, "parse_args", return_value=args),
            patch.object(
                run_p7_scheduler,
                "load_execution_context",
                side_effect=Phase7ContractError("unauthorized"),
            ) as load_context,
            patch.object(
                run_p7_scheduler,
                "ensure_artifact_path_layout",
            ) as ensure_layout,
        ):
            with self.assertRaisesRegex(Phase7ContractError, "unauthorized"):
                run_p7_scheduler.main()
        load_context.assert_called_once()
        ensure_layout.assert_not_called()

    def test_request_order_hash_count_and_phases_are_frozen(self):
        workload = w_workload(revised_manifest())
        self.assertEqual(workload["workload_id"], "W-fixed40-v1")
        self.assertEqual(workload["segment_tokens_max"], 512)
        self.assertEqual(len(workload["objects"]), 40)
        self.assertEqual(len(workload["request_order"]), 61)
        self.assertEqual(
            workload["request_order_sha256"],
            payload_sha256(workload["request_order"]),
        )
        self.assertEqual(
            Counter(row["phase"] for row in workload["request_order"]),
            Counter({"workflow": 5, "replay": 28, "replay-2": 28}),
        )

        drifted = copy.deepcopy(revised_manifest())
        drifted["workloads"]["W"]["request_order"][0]["object_id"] = "drift"
        with self.assertRaisesRegex(Phase7ContractError, "differs from frozen"):
            w_workload(drifted)

    def test_scheduler_setting_selection_and_arm_order(self):
        manifest = revised_manifest()
        setting = select_setting(
            manifest,
            setting_id="p7-w-r0-hierarchical-rho2.0",
            restart_index=1,
            runner_module=SCHEDULER_RUNNER,
        )
        self.assertEqual(setting["policy"], "hierarchical")
        self.assertEqual(formal_arm_order(setting, 0), ("E0", "R0"))
        self.assertEqual(formal_arm_order(setting, 1), ("R0", "E0"))

    def test_cl3_compatible_stats_include_all_views_and_roles(self):
        workload = w_workload(revised_manifest())
        records = [
            record(
                request,
                repeat=0,
                ttft_ms=10.0 + request["request_index"],
                cached_tokens=(0 if request["request_index"] in {1, 8} else None),
            )
            for request in workload["request_order"]
        ]
        summary = summarize_workflow_records(records)
        self.assertTrue(summary["cl3_compatible"])
        self.assertEqual(summary["denominators"]["workflow_only"]["requests"], 5)
        self.assertEqual(summary["denominators"]["all_reusable"]["requests"], 61)
        self.assertEqual(
            summary["denominators"]["all_reusable"]["partial_or_full_miss_requests"],
            2,
        )
        self.assertIn(
            "architect",
            summary["denominators"]["all_reusable"]["per_role"],
        )
        self.assertIn(
            "live_filler",
            summary["denominators"]["all_reusable"]["per_role"],
        )
        self.assertEqual(
            summary["per_repeat"]["0"]["full_trace_wall_clock_ms"],
            sum(row["elapsed_ms"] for row in records),
        )


class TestPhase7SchedulerPairing(unittest.TestCase):
    def test_e0_r0_pairing_is_request_exact_and_per_role(self):
        workload = w_workload(revised_manifest())
        requests = workload["request_order"][:8]
        e0 = [
            record(request, repeat=0, ttft_ms=20.0 + index)
            for index, request in enumerate(requests)
        ]
        r0 = [
            record(request, repeat=0, ttft_ms=10.0 + index)
            for index, request in enumerate(requests)
        ]
        paired = pair_scheduler_arms(e0, r0)
        self.assertEqual(paired["pair_count"], 8)
        self.assertTrue(paired["cl3_compatible"])
        self.assertGreater(paired["denominators"]["all_reusable"]["mean_speedup"], 1.0)
        self.assertIn(
            "architect",
            paired["denominators"]["all_reusable"]["per_role"],
        )

        broken = list(r0)
        broken[0] = {**broken[0], "object_id": "different"}
        with self.assertRaisesRegex(ValueError, "do not pair"):
            pair_scheduler_arms(e0, broken)

    def test_r4_label_disables_performance_ranking_and_mde_stop(self):
        contract = scheduler_performance_contract(["R4-like-5x"])
        self.assertEqual(contract["arm_label"], "R4-like-5x")
        self.assertFalse(contract["performance_ranking_enabled"])
        self.assertFalse(contract["r0_mde_applies"])
        self.assertEqual(contract["early_stop"], "ES-ENGINEERING-only")
        self.assertIn("not_kvcomm", contract["claim"])
        self.assertFalse(performance_ranking_enabled(["R4-like-5x"]))
        self.assertTrue(performance_ranking_enabled(["E0", "R0"]))


class TestPhase7SchedulerInactiveCounters(unittest.TestCase):
    def arm(self, host_load_value: float) -> dict:
        return {
            "metrics": {
                "inactive_tracks": {
                    counter: {
                        "verification": "direct",
                        "value": (host_load_value if counter == "host_load" else 0.0),
                        "metric": metric,
                    }
                    for counter, metric in (
                        ("host_load", "sglang:approx_kv_h2d_tokens_total"),
                        (
                            "prefetch_request",
                            "sglang:workflow_prefetch_requests_total",
                        ),
                        (
                            "prefetch_loaded_tokens",
                            "sglang:workflow_prefetch_loaded_tokens_total",
                        ),
                        (
                            "async_load",
                            "sglang:approx_kv_h2d_duration_seconds_count",
                        ),
                    )
                }
            }
        }

    def test_scheduler_assertion_covers_warmup_and_formal_arms(self):
        manifest = revised_manifest()
        warmup = [{"E0": self.arm(0.0), "R0": self.arm(0.0)}]
        formal = [
            {
                "repeat_index": 0,
                "arm_order": ["E0", "R0"],
                "arms": {"E0": self.arm(0.0), "R0": self.arm(0.0)},
            },
            {
                "repeat_index": 1,
                "arm_order": ["R0", "E0"],
                "arms": {"R0": self.arm(0.0), "E0": self.arm(0.0)},
            },
        ]
        self.assertEqual(
            len(arm_inactive_observations(warmup=warmup, formal=formal)),
            6,
        )
        assertion = build_arm_inactive_counter_assertion(
            manifest,
            warmup=warmup,
            formal=formal,
        )
        self.assertTrue(assertion["passed"])
        self.assertEqual(
            assertion["assertions"]["host_load"]["source_observation_count"],
            6,
        )
        leaked = build_arm_inactive_counter_assertion(
            manifest,
            warmup=[{"E0": self.arm(7.0), "R0": self.arm(0.0)}],
            formal=formal,
        )
        self.assertFalse(leaked["passed"])
        self.assertEqual(leaked["assertions"]["host_load"]["value"], 7.0)

    def test_scheduler_runner_uses_the_shared_arm_assertion(self):
        self.assertIs(
            run_p7_scheduler.build_arm_inactive_counter_assertion,
            build_arm_inactive_counter_assertion,
        )


class TestPhase7PhysicalMetrics(unittest.TestCase):
    def test_labeled_victim_demotion_and_missing_inactive_counters(self):
        before_text = (
            "sglang:cross_store_evicted_bytes_total"
            '{requester="approximate",provenance="exact",'
            'object_kind="filler"} 10\n'
            "sglang:cross_store_demoted_bytes_total"
            '{requester="exact",provenance="approximate",'
            'object_kind="canonical_base"} 4\n'
        )
        after_text = (
            "sglang:cross_store_evicted_bytes_total"
            '{requester="approximate",provenance="exact",'
            'object_kind="filler"} 30\n'
            "sglang:cross_store_demoted_bytes_total"
            '{requester="exact",provenance="approximate",'
            'object_kind="canonical_base"} 9\n'
        )
        before = {"sglang:cross_store_wasted_bytes_total": 2.0}
        after = {
            "sglang:cross_store_wasted_bytes_total": 5.0,
            "sglang:cross_store_peak_device_bytes": 4096.0,
        }
        metrics = cross_store_metrics(
            before_text=before_text,
            after_text=after_text,
            before_snapshot=before,
            after_snapshot=after,
        )
        self.assertEqual(metrics["victim_evict_bytes"]["rows"][0]["bytes_or_count"], 20)
        self.assertEqual(metrics["demote_bytes"]["rows"][0]["bytes_or_count"], 5)
        self.assertEqual(metrics["wasted_bytes"]["value"], 3)
        self.assertTrue(metrics["wasted_bytes"]["wasted_is_subset_of_evicted"])
        self.assertEqual(metrics["churn_bytes"]["value"], 25)
        self.assertEqual(
            metrics["churn_bytes"]["definition"],
            "evicted_bytes + demoted_bytes (wasted is excluded)",
        )
        self.assertEqual(metrics["arm_interval_peak_device_bytes"], 4096)
        self.assertEqual(
            metrics["peak_semantics"],
            "arm_high_water_since_last_full_reset",
        )
        self.assertNotIn("physical_peak_device_bytes", metrics)
        self.assertEqual(
            metrics["inactive_tracks"]["host_load"]["verification"],
            "indirectly_verified",
        )
        self.assertIsNone(metrics["inactive_tracks"]["host_load"]["value"])

    def test_churn_stays_direct_when_only_wasted_is_unobserved(self):
        before_text = (
            "sglang:cross_store_evicted_bytes_total"
            '{requester="approximate",provenance="exact",'
            'object_kind="filler"} 10\n'
        )
        after_text = (
            "sglang:cross_store_evicted_bytes_total"
            '{requester="approximate",provenance="exact",'
            'object_kind="filler"} 40\n'
        )
        metrics = cross_store_metrics(
            before_text=before_text,
            after_text=after_text,
            before_snapshot={},
            after_snapshot={},
        )
        self.assertEqual(metrics["wasted_bytes"]["verification"], "unknown")
        self.assertIsNone(metrics["wasted_bytes"]["value"])
        self.assertEqual(metrics["churn_bytes"]["verification"], "partially_direct")
        self.assertEqual(
            metrics["churn_bytes"]["unobserved_components"], ["demote_bytes"]
        )
        self.assertEqual(metrics["churn_bytes"]["value"], 30)
        self.assertIsNone(metrics["arm_interval_peak_device_bytes"])

    def test_churn_is_unknown_when_neither_component_is_observed(self):
        metrics = cross_store_metrics(
            before_text="",
            after_text="",
            before_snapshot={},
            after_snapshot={},
        )
        self.assertEqual(metrics["churn_bytes"]["verification"], "unknown")
        self.assertIsNone(metrics["churn_bytes"]["value"])
        self.assertEqual(
            metrics["churn_bytes"]["unobserved_components"],
            ["victim_evict_bytes", "demote_bytes"],
        )


class TestPhase7SchedulerDiagnosticsAndLedger(unittest.TestCase):
    def test_request_http_error_preserves_setup_partial_records_and_ledger(self):
        setup = {
            "arm": "R4-like-5x",
            "profile": "r4_like",
            "representation_kinds": ["canonical_base"],
            "representation_multiplicity": 5,
            "rows": [{"object_id": "source"}],
            "materialize_ms": 10.0,
            "register_ms": 20.0,
            "setup_ms": 30.0,
            "registration_failed": False,
            "registration_failed_by_object": {},
            "victim_sequence": [],
        }
        completed = {
            "request_index": 0,
            "phase": "workflow",
            "object_id": "object-0",
            "seed_head_ms": 1.0,
            "target_ttft_ms": 3.0,
            "ttft_ms": 4.0,
            "elapsed_ms": 9.0,
            "metrics": {"victim_evict_bytes": {"rows": []}},
        }
        workload = {
            "segment_tokens_max": 512,
            "request_order": [
                {
                    "request_index": 0,
                    "phase": "workflow",
                    "object_id": "object-0",
                },
                {
                    "request_index": 1,
                    "phase": "workflow",
                    "object_id": "object-1",
                },
            ],
        }
        metrics = {
            "victim_evict_bytes": {"rows": []},
            "arm_interval_peak_device_bytes": 0,
        }
        with (
            patch.object(
                run_p7_scheduler,
                "full_reset",
                return_value=({}, {"passed": True}),
            ),
            patch.object(run_p7_scheduler, "metric_text", return_value=""),
            patch.object(run_p7_scheduler, "metric_snapshot", return_value={}),
            patch.object(run_p7_scheduler, "setup_arm", return_value=setup),
            patch.object(
                run_p7_scheduler,
                "run_request",
                side_effect=[completed, requests.HTTPError("request failed")],
            ),
            patch.object(
                run_p7_scheduler,
                "cross_store_metrics",
                return_value=metrics,
            ),
            patch.object(run_p7_scheduler, "memory_footprint", return_value={}),
        ):
            result = run_p7_scheduler.run_arm(
                args=SimpleNamespace(port=30000),
                context=SimpleNamespace(),
                workload=workload,
                clean_baseline={},
                arm="R4-like-5x",
                repeat_index=0,
                measured=True,
                bytes_per_token=1,
            )

        self.assertIs(result["setup"], setup)
        self.assertFalse(result["setup"]["registration_failed"])
        self.assertEqual(result["records"], [completed])
        self.assertEqual(result["diagnostic_status"], "diagnostic_unavailable")
        self.assertEqual(result["diagnostic_error_stage"], "request")
        self.assertEqual(result["request_diagnostic"]["failed_request_index"], 1)
        ledger = result["ledger"]
        self.assertEqual(ledger["full_trace_wall_clock_ms"], 9.0)
        self.assertEqual(ledger["full_lifecycle_ms"], 34.0)
        self.assertEqual(ledger["non_overlapping_total_ms"], 34.0)


if __name__ == "__main__":
    unittest.main()
