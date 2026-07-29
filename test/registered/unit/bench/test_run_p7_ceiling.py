from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-c-test-cpu")

from benchmark.approx_kv import build_phase7_manifest as phase7_manifest_builder
from benchmark.approx_kv import run_p7_ceiling
from benchmark.approx_kv.build_phase7_manifest import (
    RUNNER_SPECS,
    build_a8_workload,
    build_artifact_templates,
    build_inactive_counter_pins,
    build_settings,
    build_w_workload,
    design_payload_sha256,
    load_versioned_runner_test_evidence,
)
from benchmark.approx_kv.phase7 import common as phase7_common
from benchmark.approx_kv.phase7.common import (
    CAPACITY_RUNNER,
    CEILING_RUNNER,
    Phase7ContractError,
    a8_tokens,
    arm_inactive_observations,
    build_arm_inactive_counter_assertion,
    build_inactive_counter_assertion,
    classify_request_outcome,
    ensure_artifact_path_layout,
    execution_envelope,
    filler_pool_tokens,
    finalize_artifact_hash,
    labeled_counter_observation,
    manifest_self_sha256,
    memory_footprint,
    pending_result_provenance,
    phase7_reset_invariant,
    post_pin_envelope_allowlist,
    require_envelope_path,
    require_read_only_implementation_worktree,
    select_filler_prefix,
    select_setting,
    terminal_reason_observations,
    validate_evidence_artifact_binding,
    validate_final_review_binding,
    validate_manifest_envelope,
    validate_outcome_record,
    validate_phase7_artifact,
    validate_runner_binding,
)
from benchmark.approx_kv.phase7.evidence import (
    build_runner_test_evidence,
    validate_runner_test_evidence,
)
from benchmark.approx_kv.phase7.review import (
    build_final_review,
    validate_final_review,
    validate_review_binding,
)
from benchmark.approx_kv.phase7.statistics import (
    compute_amortization,
    same_context_canary,
)
from benchmark.approx_kv.run_p7_ceiling import (
    ceiling_early_stop_contract,
    formal_arm_order,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = (
    REPO_ROOT / "benchmark/approx_kv/results/phase7/phase7-primary-manifest.json"
)
PRIMARY_MANIFEST_REL = "benchmark/approx_kv/results/phase7/phase7-primary-manifest.json"
RESULT_MANIFEST_REL = "benchmark/approx_kv/results/phase7/RESULT_MANIFEST.json"
FINAL_REVIEW_REL = "benchmark/approx_kv/results/phase7/phase7-final-opus-review.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def synthetic_cpu_evidence(runner_sha256: str, runner_key: str = "ceiling") -> dict:
    return {
        "path": "benchmark/approx_kv/results/phase7/evidence/cpu.json",
        "file_sha256": "e" * 64,
        "artifact_sha256": "f" * 64,
        "image_digest": (
            "sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781"
        ),
        "command": RUNNER_SPECS[runner_key]["required_cpu_test"],
        "exit_code": 0,
        "summary_line": "152 passed, 10 subtests passed in 12.34s",
        "passed_count": 1,
        "subtests": {"passed_count": 0, "names": []},
        "timestamp": "2026-07-28T05:00:00+00:00",
        "runner_sha256": runner_sha256,
    }


def synthetic_review_evidence() -> dict:
    return {
        "status": "passed",
        "artifact_path": FINAL_REVIEW_REL,
        "artifact_sha256": "d" * 64,
        "round_summary": "synthetic final V7 review",
    }


def synthetic_final_review(**overrides) -> dict:
    payload = {
        "reviewer": "Claude Opus 5 / Max Thinking / long context",
        "model": "claude-opus-5",
        "verdict": "PASS",
        "open_p0": 0,
        "open_p1": 0,
        "reviewed_manifest_revision": 9,
        "reviewed_manifest_sha256": "1" * 64,
        "design_payload_sha256": "2" * 64,
        "reviewed_pinned_implementation_sha": "a" * 40,
        "reviewed_pinned_tree_sha": "b" * 40,
        "runner_sha256": {
            "ceiling": "c" * 64,
            "scheduler": "5" * 64,
            "capacity_pilot": "9" * 64,
        },
        "findings": [
            {
                "finding_id": "P1-CAPACITY-ARM-ORDER",
                "severity": "P1",
                "summary": "capacity arms now run repeat-major",
                "disposition": "closed",
            }
        ],
        "disposition": "all accepted feedback closed",
        "timestamp": "2026-07-28T06:00:00-07:00",
    }
    payload.update(overrides)
    return build_final_review(**payload)


def review_evidence_summary(review: dict, path: str, artifact_sha256: str) -> dict:
    return {
        "status": "passed",
        "artifact_path": path,
        "artifact_sha256": artifact_sha256,
        "verdict": review["verdict"],
        "open_p0": review["open_p0"],
        "open_p1": review["open_p1"],
        "reviewed_manifest_revision": review["reviewed_manifest_revision"],
        "reviewed_manifest_sha256": review["reviewed_manifest_sha256"],
        "reviewed_design_payload_sha256": review["design_payload_sha256"],
        "reviewed_pinned_implementation_sha": review[
            "reviewed_pinned_implementation_sha"
        ],
        "round_summary": "synthetic final V7 review",
    }


def arm_metrics(host_load_value: float) -> dict:
    return {
        "metrics": {
            "inactive_tracks": {
                counter: {
                    "verification": "direct",
                    "value": host_load_value if counter == "host_load" else 0.0,
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
                    ("async_load", "sglang:approx_kv_h2d_duration_seconds_count"),
                )
            }
        }
    }


def authorized_manifest() -> dict:
    manifest = revised_manifest()
    manifest["status"] = "authorized"
    manifest["phase7_execution_authorized"] = True
    manifest["execution_blockers"] = []
    manifest["review_evidence"] = synthetic_review_evidence()
    manifest["implementation"]["phase7_pinned_implementation_sha"] = "a" * 40
    manifest["implementation"]["phase7_pinned_tree_sha"] = "b" * 40
    manifest["implementation"]["post_pin_envelope_allowlist"] = [
        RESULT_MANIFEST_REL,
        PRIMARY_MANIFEST_REL,
        FINAL_REVIEW_REL,
        "benchmark/approx_kv/results/phase7/evidence/cpu.json",
    ]
    manifest["runners"]["ceiling"] = {
        "module": CEILING_RUNNER,
        "path": "benchmark/approx_kv/run_p7_ceiling.py",
        "exists": True,
        "sha256": "c" * 64,
        "required_cpu_test": RUNNER_SPECS["ceiling"]["required_cpu_test"],
        "cpu_test_status": "passed",
        "cpu_test_evidence": synthetic_cpu_evidence("c" * 64),
        "review_status": "reviewed",
        "review_evidence": synthetic_review_evidence(),
    }
    manifest["preregistered_manifest_sha256"] = manifest_self_sha256(manifest)
    return manifest


def revised_manifest() -> dict:
    """Synthetic manifest carrying the post-review builder semantics."""
    manifest = load_manifest()
    manifest["manifest_revision"] = 6
    manifest["status"] = "pinned_blocked"
    manifest["phase7_execution_authorized"] = False
    manifest["execution_blockers"] = ["final_opus_review_pending"]
    manifest["plan"]["version"] = "V7"
    manifest["workloads"]["A8"] = build_a8_workload()
    manifest["settings"] = build_settings()
    p6_contract = json.loads(
        (
            REPO_ROOT / "benchmark/approx_kv/results/phase6/p6-0-contract.json"
        ).read_text()
    )
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


def target_rows(
    *,
    request_path_ms: list[float],
    invalid_index: int | None = None,
) -> list[dict]:
    return [
        {
            "target_id": f"target-{index}",
            "request_path_ms": value,
            "expected_outcome": index != invalid_index,
        }
        for index, value in enumerate(request_path_ms)
    ]


def minimal_phase7_artifact(
    manifest: dict,
    *,
    inactive_counter_assertion: dict,
    status: str,
) -> dict:
    payload = {
        "schema_version": 1,
        "run_id": "synthetic-phase7-artifact",
        "phase": "Phase7-test",
        "source_git_sha": "a" * 40,
        "source_tree_sha": "b" * 40,
        **pending_result_provenance(),
        "raw_sha256": "",
        "server_argv": [],
        "plugin_env": manifest["server_template"]["plugin_env"],
        "machine": {},
        "image_digest": manifest["environment"]["image_digest"],
        "requested_capacity": {"tokens": None, "pages": None, "bytes": None},
        "observed_capacity": {"tokens": None, "pages": None, "bytes": None},
        "crosses_chunk_boundary": False,
        "segment_count": 0,
        "warmup_repeats": 0,
        "formal_repeats": 1,
        "restarts": 1,
        "ledger": {
            "setup": {},
            "materialization": {},
            "recovery": {},
            "scheduler": {},
            "transfer": {},
            "temporary_peak": {},
        },
        "rho": {},
        "status": status,
        "manifest_revision": manifest["manifest_revision"],
        "preregistered_manifest_sha256": manifest["preregistered_manifest_sha256"],
        "manifest_file_sha256": "c" * 64,
        "plan": manifest["plan"],
        "setting_id": "synthetic",
        "restart_index": 0,
        "runner": {},
        "outcome": {},
        "reset": {},
        "provenance": {},
        "server_log_path": "/results/phase7/logs/synthetic.log",
        "server_log_sha256": None,
        "inactive_counter_assertion": inactive_counter_assertion,
    }
    finalize_artifact_hash(payload)
    return payload


class TestPhase7CeilingGuards(unittest.TestCase):
    def test_manifest_selection_and_authorization_guard(self):
        revision_five = revised_manifest()
        revision_five["manifest_revision"] = 5
        revision_five["preregistered_manifest_sha256"] = manifest_self_sha256(
            revision_five
        )
        with self.assertRaisesRegex(Phase7ContractError, "at least 6"):
            validate_manifest_envelope(
                revision_five,
                require_authorized=False,
            )

        manifest = revised_manifest()
        validate_manifest_envelope(manifest, require_authorized=False)
        with self.assertRaisesRegex(
            Phase7ContractError, "requires an authorized manifest"
        ):
            validate_manifest_envelope(manifest, require_authorized=True)

        authorized = authorized_manifest()
        validate_manifest_envelope(authorized, require_authorized=True)
        setting = select_setting(
            authorized,
            setting_id="p7-a8-r0-body1024-rho1.5",
            restart_index=2,
            runner_module=CEILING_RUNNER,
        )
        self.assertEqual(setting["body_tokens"], 1024)

    def test_self_hash_runner_restart_and_source_guards(self):
        manifest = authorized_manifest()
        drifted = copy.deepcopy(manifest)
        drifted["settings"][3]["body_tokens"] = 999
        with self.assertRaisesRegex(Phase7ContractError, "self-hash mismatch"):
            validate_manifest_envelope(drifted, require_authorized=True)

        with self.assertRaisesRegex(Phase7ContractError, "restart 9"):
            select_setting(
                manifest,
                setting_id="p7-a8-r0-body1024-rho1.5",
                restart_index=9,
                runner_module=CEILING_RUNNER,
            )
        with self.assertRaisesRegex(Phase7ContractError, "expected runner"):
            select_setting(
                manifest,
                setting_id="p7-w-r0-lru-rho1.5",
                restart_index=0,
                runner_module=CEILING_RUNNER,
            )

        validate_runner_binding(
            manifest,
            runner_key="ceiling",
            runner_module=CEILING_RUNNER,
            runner_path="benchmark/approx_kv/run_p7_ceiling.py",
            current_runner_sha256="c" * 64,
            pinned_runner_sha256="c" * 64,
            observed_pinned_sha="a" * 40,
            observed_pinned_tree="b" * 40,
        )
        with self.assertRaisesRegex(Phase7ContractError, "blob hash mismatch"):
            validate_runner_binding(
                manifest,
                runner_key="ceiling",
                runner_module=CEILING_RUNNER,
                runner_path="benchmark/approx_kv/run_p7_ceiling.py",
                current_runner_sha256="d" * 64,
                pinned_runner_sha256="c" * 64,
                observed_pinned_sha="a" * 40,
                observed_pinned_tree="b" * 40,
            )
        with self.assertRaisesRegex(Phase7ContractError, "source tree mismatch"):
            validate_runner_binding(
                manifest,
                runner_key="ceiling",
                runner_module=CEILING_RUNNER,
                runner_path="benchmark/approx_kv/run_p7_ceiling.py",
                current_runner_sha256="c" * 64,
                pinned_runner_sha256="c" * 64,
                observed_pinned_sha="a" * 40,
                observed_pinned_tree="d" * 40,
            )
        evidence_free = copy.deepcopy(manifest)
        evidence_free["runners"]["ceiling"]["cpu_test_evidence"] = None
        with self.assertRaisesRegex(Phase7ContractError, "evidence is missing"):
            validate_runner_binding(
                evidence_free,
                runner_key="ceiling",
                runner_module=CEILING_RUNNER,
                runner_path="benchmark/approx_kv/run_p7_ceiling.py",
                current_runner_sha256="c" * 64,
                pinned_runner_sha256="c" * 64,
                observed_pinned_sha="a" * 40,
                observed_pinned_tree="b" * 40,
            )

    def test_r2_disabled_not_comparable_is_not_executable(self):
        manifest = authorized_manifest()
        validate_manifest_envelope(manifest, require_authorized=True)
        self.assertEqual(manifest["r2_strategy"], "disabled_not_comparable")
        self.assertFalse(any("R2" in row["arms"] for row in manifest["settings"]))
        with self.assertRaisesRegex(Phase7ContractError, "found 0"):
            select_setting(
                manifest,
                setting_id="p7-a8-r2-rho1.5",
                restart_index=0,
                runner_module=CEILING_RUNNER,
            )

    def test_capacity_setting_selection_and_disabled_rho3(self):
        manifest = authorized_manifest()
        setting = select_setting(
            manifest,
            setting_id="p6delta-s0-rho2-chunk4096",
            restart_index=0,
            runner_module=CAPACITY_RUNNER,
        )
        self.assertEqual(setting["policy"], "lru")
        self.assertEqual(setting["rho_logical_demand"], 2.0)
        with self.assertRaisesRegex(
            Phase7ContractError,
            "rho3 conditional resolution",
        ):
            select_setting(
                manifest,
                setting_id="p6delta-s4-rho3-chunk4096",
                restart_index=0,
                runner_module=CAPACITY_RUNNER,
            )

    def test_capacity_runner_blob_guard(self):
        manifest = authorized_manifest()
        manifest["runners"]["capacity_pilot"] = {
            "module": CAPACITY_RUNNER,
            "path": "benchmark/approx_kv/run_p6_4_capacity_pilot.py",
            "exists": True,
            "sha256": "9" * 64,
            "required_cpu_test": RUNNER_SPECS["capacity_pilot"]["required_cpu_test"],
            "cpu_test_status": "passed",
            "cpu_test_evidence": synthetic_cpu_evidence("9" * 64, "capacity_pilot"),
            "review_status": "reviewed",
            "review_evidence": synthetic_review_evidence(),
        }
        validate_runner_binding(
            manifest,
            runner_key="capacity_pilot",
            runner_module=CAPACITY_RUNNER,
            runner_path="benchmark/approx_kv/run_p6_4_capacity_pilot.py",
            current_runner_sha256="9" * 64,
            pinned_runner_sha256="9" * 64,
            observed_pinned_sha="a" * 40,
            observed_pinned_tree="b" * 40,
        )
        with self.assertRaisesRegex(Phase7ContractError, "pinned runner blob"):
            validate_runner_binding(
                manifest,
                runner_key="capacity_pilot",
                runner_module=CAPACITY_RUNNER,
                runner_path="benchmark/approx_kv/run_p6_4_capacity_pilot.py",
                current_runner_sha256="9" * 64,
                pinned_runner_sha256="8" * 64,
                observed_pinned_sha="a" * 40,
                observed_pinned_tree="b" * 40,
            )

    def test_artifact_paths_must_be_distinct(self):
        path = Path("/results/phase7/phase7-path-placeholder")
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            ensure_artifact_path_layout(
                output=path,
                log=path,
                central_log=path.with_name("central.jsonl"),
                staging_root="/results/phase7",
            )

    def test_artifact_paths_must_remain_in_runtime_staging(self):
        templates = build_artifact_templates()
        self.assertEqual(templates["runtime_staging_root"], "/results/phase7")
        self.assertNotIn("raw_json", templates)
        self.assertIn("versioned_destination_raw_json", templates)
        ensure_artifact_path_layout(
            output=Path("/results/phase7/raw/allowed.json"),
            log=Path("/results/phase7/logs/allowed.log"),
            central_log=Path("/results/phase7/central.jsonl"),
            staging_root="/results/phase7",
        )
        with self.assertRaisesRegex(ValueError, "staging root"):
            ensure_artifact_path_layout(
                output=REPO_ROOT
                / "benchmark/approx_kv/results/phase7/raw/rejected.json",
                log=Path("/results/phase7/logs/rejected.log"),
                central_log=Path("/results/phase7/central.jsonl"),
                staging_root="/results/phase7",
            )
        with self.assertRaisesRegex(ValueError, "staging root"):
            ensure_artifact_path_layout(
                output=Path("/results/phase7/../escaped.json"),
                log=Path("/results/phase7/logs/rejected.log"),
                central_log=Path("/results/phase7/central.jsonl"),
                staging_root="/results/phase7",
            )

    def test_authorization_is_loaded_before_staging_validation(self):
        args = Namespace(
            manifest=MANIFEST_PATH,
            setting_id="p7-a8-r0-body1024-rho1.5",
            restart_index=0,
            output=Path("/results/phase7/raw/test.json"),
            log=Path("/results/phase7/logs/test.log"),
            central_log=Path("/results/phase7/central.jsonl"),
        )
        with (
            patch.object(run_p7_ceiling, "parse_args", return_value=args),
            patch.object(
                run_p7_ceiling,
                "load_execution_context",
                side_effect=Phase7ContractError("unauthorized"),
            ) as load_context,
            patch.object(
                run_p7_ceiling,
                "ensure_artifact_path_layout",
            ) as ensure_layout,
        ):
            with self.assertRaisesRegex(Phase7ContractError, "unauthorized"):
                run_p7_ceiling.main()
        load_context.assert_called_once()
        ensure_layout.assert_not_called()

    def test_phase7_requires_read_only_implementation_mount(self):
        with patch.object(
            phase7_common.os,
            "statvfs",
            return_value=Namespace(f_flag=0),
        ):
            with self.assertRaisesRegex(Phase7ContractError, "mounted read-only"):
                require_read_only_implementation_worktree()
        with patch.object(
            phase7_common.os,
            "statvfs",
            return_value=Namespace(f_flag=phase7_common.os.ST_RDONLY),
        ):
            require_read_only_implementation_worktree()


class TestPhase7RunnerEvidence(unittest.TestCase):
    def test_cpu_test_evidence_records_reproducible_inputs(self):
        payload = build_runner_test_evidence(
            runner_key="ceiling",
            runner_module=CEILING_RUNNER,
            runner_path="benchmark/approx_kv/run_p7_ceiling.py",
            runner_sha256="a" * 64,
            image_digest="sha256:" + "b" * 64,
            command=RUNNER_SPECS["ceiling"]["required_cpu_test"],
            exit_code=0,
            summary_line="42 passed, 10 subtests passed in 9.99s",
            passed_count=42,
            subtests_passed_count=10,
            subtests=["reset invariants", "artifact guards"],
            timestamp="2026-07-28T05:30:00-07:00",
        )
        validate_runner_test_evidence(payload)
        self.assertEqual(payload["passed_count"], 42)
        self.assertEqual(payload["exit_code"], 0)
        self.assertEqual(
            payload["summary_line"],
            "42 passed, 10 subtests passed in 9.99s",
        )
        self.assertEqual(payload["subtests"]["passed_count"], 10)
        self.assertEqual(
            payload["subtests"]["names"],
            ["reset invariants", "artifact guards"],
        )

    def test_cpu_test_evidence_rejects_unpinned_image(self):
        with self.assertRaisesRegex(ValueError, "image_digest"):
            build_runner_test_evidence(
                runner_key="ceiling",
                runner_module=CEILING_RUNNER,
                runner_path="benchmark/approx_kv/run_p7_ceiling.py",
                runner_sha256="a" * 64,
                image_digest="latest",
                command=RUNNER_SPECS["ceiling"]["required_cpu_test"],
                exit_code=0,
                summary_line="1 passed",
                passed_count=1,
                subtests_passed_count=0,
                subtests=[],
                timestamp="2026-07-28T05:30:00-07:00",
            )

    def test_cpu_test_evidence_requires_zero_exit_and_a_summary_line(self):
        def build(**overrides):
            payload = {
                "runner_key": "ceiling",
                "runner_module": CEILING_RUNNER,
                "runner_path": "benchmark/approx_kv/run_p7_ceiling.py",
                "runner_sha256": "a" * 64,
                "image_digest": "sha256:" + "b" * 64,
                "command": RUNNER_SPECS["ceiling"]["required_cpu_test"],
                "exit_code": 0,
                "summary_line": "1 passed in 0.10s",
                "passed_count": 1,
                "subtests_passed_count": 0,
                "subtests": [],
                "timestamp": "2026-07-28T05:30:00-07:00",
            }
            payload.update(overrides)
            return build_runner_test_evidence(**payload)

        with self.assertRaisesRegex(ValueError, "exit_code must be 0"):
            build(exit_code=1)
        with self.assertRaisesRegex(ValueError, "summary_line"):
            build(summary_line="   ")
        with self.assertRaisesRegex(ValueError, "passed_count must be positive"):
            build(passed_count=0)

    def test_manifest_and_runtime_require_the_frozen_cpu_test_command(self):
        manifest = authorized_manifest()
        validate_runner_binding(
            manifest,
            runner_key="ceiling",
            runner_module=CEILING_RUNNER,
            runner_path="benchmark/approx_kv/run_p7_ceiling.py",
            current_runner_sha256="c" * 64,
            pinned_runner_sha256="c" * 64,
            observed_pinned_sha="a" * 40,
            observed_pinned_tree="b" * 40,
        )
        for field, value, message in (
            ("command", "python3 -m pytest -q other_test.py", "command mismatch"),
            ("exit_code", 1, "exit code is not 0"),
            ("summary_line", "  ", "lacks a summary"),
            ("passed_count", 0, "no passing tests"),
        ):
            drifted = copy.deepcopy(manifest)
            drifted["runners"]["ceiling"]["cpu_test_evidence"][field] = value
            with self.assertRaisesRegex(Phase7ContractError, message):
                validate_runner_binding(
                    drifted,
                    runner_key="ceiling",
                    runner_module=CEILING_RUNNER,
                    runner_path="benchmark/approx_kv/run_p7_ceiling.py",
                    current_runner_sha256="c" * 64,
                    pinned_runner_sha256="c" * 64,
                    observed_pinned_sha="a" * 40,
                    observed_pinned_tree="b" * 40,
                )
        drifted = copy.deepcopy(manifest)
        drifted["runners"]["ceiling"]["required_cpu_test"] = "pytest"
        with self.assertRaisesRegex(Phase7ContractError, "required CPU test drift"):
            validate_runner_binding(
                drifted,
                runner_key="ceiling",
                runner_module=CEILING_RUNNER,
                runner_path="benchmark/approx_kv/run_p7_ceiling.py",
                current_runner_sha256="c" * 64,
                pinned_runner_sha256="c" * 64,
                observed_pinned_sha="a" * 40,
                observed_pinned_tree="b" * 40,
            )


class TestPhase7FinalReviewArtifact(unittest.TestCase):
    def test_review_artifact_freezes_the_reviewed_identity(self):
        review = synthetic_final_review()
        validate_final_review(review)
        self.assertEqual(review["artifact"], "phase7-final-opus-review")
        self.assertEqual(review["verdict"], "PASS")
        self.assertEqual(review["open_p0"], 0)
        self.assertEqual(review["open_p1"], 0)
        self.assertEqual(
            sorted(review["runner_sha256"]),
            ["capacity_pilot", "ceiling", "scheduler"],
        )
        tampered = copy.deepcopy(review)
        tampered["verdict"] = "PASS_WITH_CAVEATS"
        with self.assertRaisesRegex(ValueError, "self-hash mismatch"):
            validate_final_review(tampered)

    def test_review_rejects_failing_verdicts_and_open_severe_findings(self):
        with self.assertRaises(ValueError):
            synthetic_final_review(verdict="FAIL")
        with self.assertRaisesRegex(ValueError, "zero open P0"):
            synthetic_final_review(open_p0=1)
        with self.assertRaisesRegex(ValueError, "zero open P1"):
            synthetic_final_review(open_p1=2)
        with self.assertRaisesRegex(ValueError, "P0/P1"):
            synthetic_final_review(
                findings=[
                    {
                        "finding_id": "P1-OPEN",
                        "severity": "P1",
                        "summary": "still open",
                        "disposition": "deferred",
                    }
                ]
            )
        with self.assertRaisesRegex(ValueError, "lacks runner hashes"):
            synthetic_final_review(runner_sha256={"ceiling": "c" * 64})
        review = synthetic_final_review(
            verdict="PASS_WITH_CAVEATS",
            findings=[
                {
                    "finding_id": "P2-DOC",
                    "severity": "P2",
                    "summary": "documentation caveat",
                    "disposition": "deferred",
                }
            ],
        )
        validate_final_review(review)

    def test_review_binding_requires_design_supersede_revision_and_runners(self):
        review = synthetic_final_review()
        runner_sha256 = dict(review["runner_sha256"])
        validate_review_binding(
            review,
            design_payload_sha256="2" * 64,
            supersedes_manifest_sha256="1" * 64,
            manifest_revision=10,
            pinned_implementation_sha="a" * 40,
            pinned_tree_sha="b" * 40,
            runner_sha256=runner_sha256,
        )
        with self.assertRaisesRegex(ValueError, "pinned implementation SHA"):
            validate_review_binding(
                review,
                design_payload_sha256="2" * 64,
                supersedes_manifest_sha256="1" * 64,
                manifest_revision=10,
                pinned_implementation_sha="f" * 40,
                pinned_tree_sha="b" * 40,
                runner_sha256=runner_sha256,
            )
        with self.assertRaisesRegex(ValueError, "pinned implementation tree"):
            validate_review_binding(
                review,
                design_payload_sha256="2" * 64,
                supersedes_manifest_sha256="1" * 64,
                manifest_revision=10,
                pinned_implementation_sha="a" * 40,
                pinned_tree_sha="e" * 40,
                runner_sha256=runner_sha256,
            )
        with self.assertRaisesRegex(ValueError, "design payload hash mismatch"):
            validate_review_binding(
                review,
                design_payload_sha256="3" * 64,
                pinned_implementation_sha="a" * 40,
                pinned_tree_sha="b" * 40,
                supersedes_manifest_sha256="1" * 64,
                manifest_revision=10,
                runner_sha256=runner_sha256,
            )
        with self.assertRaisesRegex(ValueError, "superseded manifest revision"):
            validate_review_binding(
                review,
                design_payload_sha256="2" * 64,
                pinned_implementation_sha="a" * 40,
                pinned_tree_sha="b" * 40,
                supersedes_manifest_sha256="4" * 64,
                manifest_revision=10,
                runner_sha256=runner_sha256,
            )
        with self.assertRaisesRegex(ValueError, "does not record a superseded hash"):
            validate_review_binding(
                review,
                design_payload_sha256="2" * 64,
                pinned_implementation_sha="a" * 40,
                pinned_tree_sha="b" * 40,
                supersedes_manifest_sha256=None,
                manifest_revision=10,
                runner_sha256=runner_sha256,
            )
        with self.assertRaisesRegex(ValueError, "must be greater"):
            validate_review_binding(
                review,
                design_payload_sha256="2" * 64,
                pinned_implementation_sha="a" * 40,
                pinned_tree_sha="b" * 40,
                supersedes_manifest_sha256="1" * 64,
                manifest_revision=9,
                runner_sha256=runner_sha256,
            )
        with self.assertRaisesRegex(ValueError, "runner hash mismatch"):
            validate_review_binding(
                review,
                design_payload_sha256="2" * 64,
                pinned_implementation_sha="a" * 40,
                pinned_tree_sha="b" * 40,
                supersedes_manifest_sha256="1" * 64,
                manifest_revision=10,
                runner_sha256={**runner_sha256, "scheduler": "6" * 64},
            )

    def test_review_of_its_own_manifest_hash_is_not_self_containing(self):
        """A review may only review the superseded revision, never itself."""
        review = synthetic_final_review()
        with self.assertRaisesRegex(ValueError, "superseded manifest revision"):
            validate_review_binding(
                review,
                design_payload_sha256="2" * 64,
                pinned_implementation_sha="a" * 40,
                pinned_tree_sha="b" * 40,
                # the activating manifest's own self-hash, not what it supersedes
                supersedes_manifest_sha256="7" * 64,
                manifest_revision=10,
                runner_sha256=dict(review["runner_sha256"]),
            )


class TestPhase7InactiveCounterAssertion(unittest.TestCase):
    def test_missing_series_is_indirect_not_fabricated_zero(self):
        manifest = authorized_manifest()
        assertion = build_inactive_counter_assertion(manifest, [])
        self.assertTrue(assertion["passed"])
        for row in assertion["assertions"].values():
            self.assertEqual(row["verification"], "indirectly_verified")
            self.assertIsNone(row["value"])
        artifact = minimal_phase7_artifact(
            manifest,
            inactive_counter_assertion=assertion,
            status="valid",
        )
        validate_phase7_artifact(artifact, manifest=manifest)

    def test_direct_nonzero_requires_invalid_artifact(self):
        manifest = authorized_manifest()
        observations = {
            counter: {
                "verification": "direct",
                "value": 1.0 if counter == "host_load" else 0.0,
            }
            for counter in manifest["required_inactive_counters"]
        }
        assertion = build_inactive_counter_assertion(manifest, [observations])
        self.assertFalse(assertion["passed"])
        artifact = minimal_phase7_artifact(
            manifest,
            inactive_counter_assertion=assertion,
            status="valid",
        )
        with self.assertRaisesRegex(Phase7ContractError, "requires invalid"):
            validate_phase7_artifact(artifact, manifest=manifest)
        artifact["status"] = "invalid"
        finalize_artifact_hash(artifact)
        validate_phase7_artifact(artifact, manifest=manifest)

    def test_warmup_and_formal_arms_both_reach_the_assertion(self):
        manifest = authorized_manifest()
        warmup = [{"D0": arm_metrics(0.0), "E0": arm_metrics(0.0)}]
        formal = [
            {
                "repeat_index": 0,
                "arm_order": ["D0", "E0"],
                "arms": {"D0": arm_metrics(0.0), "E0": arm_metrics(0.0)},
            }
        ]
        self.assertEqual(
            len(arm_inactive_observations(warmup=warmup, formal=formal)),
            4,
        )
        assertion = build_arm_inactive_counter_assertion(
            manifest,
            warmup=warmup,
            formal=formal,
        )
        self.assertTrue(assertion["passed"])
        self.assertEqual(
            assertion["assertions"]["host_load"]["source_observation_count"],
            4,
        )

        warmup_only_leak = [{"D0": arm_metrics(3.0), "E0": arm_metrics(0.0)}]
        leaked = build_arm_inactive_counter_assertion(
            manifest,
            warmup=warmup_only_leak,
            formal=formal,
        )
        self.assertFalse(leaked["passed"])
        self.assertEqual(leaked["assertions"]["host_load"]["value"], 3.0)
        artifact = minimal_phase7_artifact(
            manifest,
            inactive_counter_assertion=leaked,
            status="valid",
        )
        with self.assertRaisesRegex(Phase7ContractError, "requires invalid"):
            validate_phase7_artifact(artifact, manifest=manifest)


class TestPhase7A8Contract(unittest.TestCase):
    def test_a8_tokens_hashes_order_and_arm_order(self):
        manifest = revised_manifest()
        for body_tokens in (1024, 2048):
            workload = a8_tokens(manifest, body_tokens=body_tokens)
            self.assertEqual(len(workload["source_header"]), 64)
            self.assertEqual(len(workload["body"]), body_tokens)
            self.assertEqual(len(workload["targets"]), 8)
            self.assertEqual(
                [row["spec"]["order"] for row in workload["targets"]],
                list(range(8)),
            )

        setting = next(
            row
            for row in manifest["settings"]
            if row["setting_id"] == "p7-a8-r0-body1024-rho1.5"
        )
        self.assertEqual(formal_arm_order(setting, 0), ("D0", "E0", "R0"))
        self.assertEqual(formal_arm_order(setting, 1), ("R0", "E0", "D0"))

    def test_mde_only_gates_primary_a8_supplements(self):
        manifest = load_manifest()
        primary = next(
            row
            for row in manifest["settings"]
            if row["setting_id"] == "p7-a8-r0-body1024-rho1.5"
        )
        with self.assertRaisesRegex(Phase7ContractError, "mde-gate-passed"):
            ceiling_early_stop_contract(
                primary,
                restart_index=1,
                mde_gate_passed=False,
            )
        sensitivity = next(
            row
            for row in manifest["settings"]
            if row["setting_id"] == "p7-a8-r0-body2048-rho2-chunk1024-sensitivity"
        )
        contract = ceiling_early_stop_contract(
            sensitivity,
            restart_index=1,
            mde_gate_passed=False,
        )
        self.assertFalse(contract["r0_mde_applies"])
        self.assertFalse(contract["mde_gate_required"])

    def test_filler_selection_subtracts_setup_and_is_deterministic(self):
        pool = filler_pool_tokens(load_manifest())
        selection = select_filler_prefix(
            pool,
            capacity_tokens=1000,
            rho_logical_demand=1.5,
            setup_resident_tokens=100,
        )
        self.assertEqual(
            selection["selected_filler_ids"],
            ["p7-filler-00", "p7-filler-01", "p7-filler-02"],
        )
        self.assertEqual(selection["needed_filler_tokens"], 1400)
        self.assertEqual(selection["selected_filler_tokens"], 1536)

    def test_real_n_accumulation_and_break_even(self):
        dense = target_rows(request_path_ms=[10.0] * 8)
        recovery = target_rows(request_path_ms=[4.0] * 8)
        result = compute_amortization(
            dense,
            recovery,
            dense_source_materialization_ms=3.0,
            recovery_source_preparation_ms=8.0,
        )
        self.assertAlmostEqual(result["n"]["1"]["speedup_full_setup"], 10.0 / 12.0)
        self.assertAlmostEqual(result["n"]["2"]["speedup_full_setup"], 20.0 / 16.0)
        self.assertEqual(result["full_setup_break_even_observed_N"], 2)
        self.assertEqual(result["incremental_setup_break_even_observed_N"], 1)

    def test_invalid_prefix_invalidates_all_larger_n(self):
        dense = target_rows(request_path_ms=[10.0] * 8)
        recovery = target_rows(
            request_path_ms=[4.0] * 8,
            invalid_index=1,
        )
        result = compute_amortization(
            dense,
            recovery,
            dense_source_materialization_ms=3.0,
            recovery_source_preparation_ms=8.0,
        )
        self.assertTrue(result["n"]["1"]["valid"])
        for n in ("2", "4", "8"):
            self.assertFalse(result["n"][n]["valid"])
            self.assertEqual(result["n"][n]["reason"], "invalid_prefix_outcome")

    def test_same_context_mismatch_is_invalid(self):
        matched = same_context_canary(list(range(8)), list(range(8)))
        self.assertEqual(matched["engineering_status"], "valid")
        mismatch = same_context_canary(
            list(range(8)),
            [0, 1, 2, 99, 4, 5, 6, 7],
        )
        self.assertEqual(mismatch["engineering_status"], "invalid")
        self.assertFalse(mismatch["matched"])


class TestPhase7OutcomeAndReset(unittest.TestCase):
    @staticmethod
    def request_observations(**values):
        names = ("success", "dense_fallback", "exact", "exact_host_preferred")
        return {
            "outcomes": {
                name: {
                    "verification": "direct",
                    "value": float(values.get(name, 0)),
                }
                for name in names
            }
        }

    def test_outcome_and_terminal_reason_are_exclusive(self):
        success = classify_request_outcome(
            arm="R0",
            cached_tokens=1088,
            expected_cached_tokens=1088,
            request_observations=self.request_observations(success=1),
            terminal_observations={
                "mapped": {
                    "cross_store_reservation_failed": 0,
                    "device_allocation_failed": 0,
                    "unsupported": 0,
                    "registration_failed": 0,
                    "prefix_gap": 0,
                }
            },
        )
        self.assertEqual(success["outcome"], "approximate_gpu_recovery")
        self.assertIsNone(success["terminal_reason"])
        self.assertTrue(success["taxonomy_valid"])

        fallback = classify_request_outcome(
            arm="R0",
            cached_tokens=64,
            expected_cached_tokens=1088,
            request_observations=self.request_observations(dense_fallback=1),
            terminal_observations={
                "mapped": {
                    "cross_store_reservation_failed": 1024,
                    "device_allocation_failed": 0,
                    "unsupported": 0,
                    "registration_failed": 0,
                    "prefix_gap": 0,
                }
            },
        )
        self.assertEqual(fallback["outcome"], "approximate_recovery_failed_dense")
        self.assertEqual(fallback["terminal_reason"], "cross_store_reservation_failed")
        self.assertTrue(fallback["taxonomy_valid"])

        exact_after_failed_registration = classify_request_outcome(
            arm="R0",
            cached_tokens=1088,
            expected_cached_tokens=1088,
            request_observations=self.request_observations(exact=1),
            terminal_observations={"mapped": {}},
            registration_failed=True,
            expected_outcomes=("exact_gpu_hit",),
        )
        self.assertEqual(exact_after_failed_registration["outcome"], "exact_gpu_hit")
        self.assertIsNone(exact_after_failed_registration["terminal_reason"])

        unknown_reason = classify_request_outcome(
            arm="R0",
            cached_tokens=64,
            expected_cached_tokens=1088,
            request_observations=self.request_observations(dense_fallback=1),
            terminal_observations={
                "mapped": {
                    "cross_store_reservation_failed": None,
                    "device_allocation_failed": None,
                    "unsupported": None,
                    "registration_failed": None,
                    "prefix_gap": None,
                }
            },
        )
        self.assertFalse(unknown_reason["taxonomy_valid"])
        self.assertIsNone(unknown_reason["terminal_reason"])
        self.assertFalse(
            validate_outcome_record(
                {
                    "outcome": "exact_gpu_hit",
                    "terminal_reason": "prefix_gap",
                    "ambiguity": None,
                }
            )
        )

    def test_missing_label_counter_is_not_fabricated_as_zero(self):
        observation = labeled_counter_observation(
            "",
            "",
            name="sglang:approx_kv_requests_total",
            required_labels={"operation": "reuse", "outcome": "success"},
            indirect_evidence="cached-token path matched",
        )
        self.assertEqual(observation["verification"], "indirectly_verified")
        self.assertIsNone(observation["value"])

        terminal = terminal_reason_observations(
            'sglang:approx_kv_dense_fallback_total{reason="prefix_gap"} 4\n',
            'sglang:approx_kv_dense_fallback_total{reason="prefix_gap"} 8\n',
        )
        self.assertEqual(terminal["mapped"]["prefix_gap"], 4)
        self.assertIsNone(terminal["mapped"]["cross_store_reservation_failed"])

        transfer_fallback = terminal_reason_observations(
            "",
            (
                'sglang:approx_kv_dense_fallback_total{reason="stale_handle"} 2\n'
                'sglang:approx_kv_dense_fallback_total{reason="residency_miss"} 3\n'
                "sglang:approx_kv_dense_fallback_total"
                '{reason="source_slice_mismatch"} 5\n'
            ),
        )
        self.assertEqual(transfer_fallback["mapped"]["unsupported"], 10)
        self.assertEqual(transfer_fallback["unmapped_raw_reasons"], {})

    def test_unmapped_and_excluded_raw_reasons_are_not_forced_into_taxonomy(self):
        before = (
            'sglang:approx_kv_dense_fallback_total{reason="prefix_gap"} 4\n'
            'sglang:approx_kv_dense_fallback_total{reason="brand_new_reason"} 0\n'
            "sglang:approx_kv_dense_fallback_total"
            '{reason="cross_store_exact_pressure_failed"} 0\n'
        )
        after = (
            'sglang:approx_kv_dense_fallback_total{reason="prefix_gap"} 4\n'
            'sglang:approx_kv_dense_fallback_total{reason="brand_new_reason"} 7\n'
            "sglang:approx_kv_dense_fallback_total"
            '{reason="cross_store_exact_pressure_failed"} 9\n'
        )
        observations = terminal_reason_observations(before, after)
        self.assertEqual(observations["value_unit"], "tokens")
        self.assertEqual(observations["unmapped_raw_reasons"], {"brand_new_reason": 7})
        self.assertEqual(
            observations["excluded_raw_reasons"],
            {"cross_store_exact_pressure_failed": 9},
        )
        self.assertIsNone(observations["mapped"]["unsupported"])
        self.assertIsNone(observations["mapped"]["prefix_gap"])

        record = classify_request_outcome(
            arm="R0",
            cached_tokens=64,
            expected_cached_tokens=1088,
            request_observations=self.request_observations(dense_fallback=1),
            terminal_observations=observations,
        )
        self.assertEqual(record["outcome"], "approximate_recovery_failed_dense")
        self.assertIsNone(record["terminal_reason"])
        self.assertIn("unmapped terminal reasons", record["ambiguity"])
        self.assertFalse(record["taxonomy_valid"])

    def test_memory_footprint_does_not_double_count_the_two_stores(self):
        footprint = memory_footprint(
            {
                "sglang:kv_used_tokens": 100.0,
                "sglang:kv_evictable_tokens": 20.0,
                "sglang:approx_kv_store_device_bytes": 400.0,
                "sglang:approx_kv_store_host_bytes": 0.0,
            },
            bytes_per_token=10,
        )
        self.assertEqual(footprint["nonfree_resident_tokens"], 120.0)
        self.assertEqual(footprint["nonfree_resident_bytes"], 1200.0)
        self.assertEqual(footprint["approx_device_bytes"], 400.0)
        self.assertEqual(footprint["exact_only_estimated_bytes"], 800.0)
        self.assertIn("already contains", footprint["overlap_note"])
        self.assertNotIn("exact_resident_tokens", footprint)

        clamped = memory_footprint(
            {
                "sglang:kv_used_tokens": 1.0,
                "sglang:kv_evictable_tokens": 0.0,
                "sglang:approx_kv_store_device_bytes": 999.0,
            },
            bytes_per_token=10,
        )
        self.assertEqual(clamped["exact_only_estimated_bytes"], 0.0)

        unknown = memory_footprint({}, bytes_per_token=10)
        self.assertIsNone(unknown["nonfree_resident_tokens"])
        self.assertIsNone(unknown["exact_only_estimated_bytes"])

    def test_reset_covers_all_phase7_stores_and_gauges(self):
        clean = {
            "sglang:max_total_num_tokens": 1000.0,
            "sglang:kv_available_tokens": 1000.0,
            "sglang:kv_evictable_tokens": 0.0,
            "sglang:kv_used_tokens": 0.0,
            "sglang:approx_kv_store_records": 0.0,
            "sglang:approx_kv_store_device_bytes": 0.0,
            "sglang:approx_kv_store_host_bytes": 0.0,
            "sglang:approx_kv_store_leases": 0.0,
            "sglang:approx_kv_store_orphans": 0.0,
            "sglang:approx_kv_provisional_tokens": 0.0,
            "sglang:cross_store_reserved_device_bytes": 0.0,
        }
        result = phase7_reset_invariant(clean, strict=True, clean_baseline=clean)
        self.assertTrue(result["passed"])
        self.assertTrue(result["strict"])
        self.assertEqual(result["not_yet_exported"], [])
        self.assertEqual(
            set(result["components"]),
            {
                "exact",
                "approximate",
                "metadata",
                "reserved",
                "provisional",
                "leases",
                "orphans",
            },
        )

        dirty = dict(clean)
        dirty["sglang:approx_kv_provisional_tokens"] = 1.0
        result = phase7_reset_invariant(dirty, strict=True, clean_baseline=clean)
        self.assertFalse(result["passed"])
        self.assertFalse(result["components"]["provisional"])

    def test_startup_missing_reserved_series_is_not_an_explicit_zero(self):
        clean = {
            "sglang:max_total_num_tokens": 1000.0,
            "sglang:kv_available_tokens": 1000.0,
            "sglang:kv_evictable_tokens": 0.0,
            "sglang:kv_used_tokens": 0.0,
            "sglang:approx_kv_store_records": 0.0,
            "sglang:approx_kv_store_device_bytes": 0.0,
            "sglang:approx_kv_store_host_bytes": 0.0,
            "sglang:approx_kv_store_leases": 0.0,
            "sglang:approx_kv_store_orphans": 0.0,
            "sglang:approx_kv_provisional_tokens": 0.0,
        }
        startup = phase7_reset_invariant(clean, strict=False, clean_baseline=None)
        self.assertTrue(startup["passed"])
        self.assertEqual(
            startup["not_yet_exported"],
            ["sglang:cross_store_reserved_device_bytes"],
        )
        self.assertEqual(
            startup["gauge_states"]["sglang:cross_store_reserved_device_bytes"],
            "not_yet_exported",
        )
        self.assertEqual(startup["missing_gauges"], [])
        self.assertIsNone(
            startup["store_gauges"]["sglang:cross_store_reserved_device_bytes"]
        )

        strict = phase7_reset_invariant(clean, strict=True, clean_baseline=None)
        self.assertFalse(strict["passed"])
        self.assertEqual(strict["not_yet_exported"], [])
        self.assertEqual(
            strict["missing_gauges"],
            ["sglang:cross_store_reserved_device_bytes"],
        )
        self.assertFalse(strict["components"]["reserved"])

        leaked = dict(clean)
        leaked["sglang:cross_store_reserved_device_bytes"] = 4096.0
        after_r0 = phase7_reset_invariant(leaked, strict=True, clean_baseline=None)
        self.assertFalse(after_r0["passed"])
        self.assertFalse(after_r0["components"]["reserved"])

    def test_missing_store_series_never_passes_even_when_not_strict(self):
        partial = {
            "sglang:max_total_num_tokens": 1000.0,
            "sglang:kv_available_tokens": 1000.0,
            "sglang:kv_evictable_tokens": 0.0,
            "sglang:kv_used_tokens": 0.0,
            "sglang:approx_kv_store_records": 0.0,
            "sglang:approx_kv_store_device_bytes": 0.0,
            "sglang:approx_kv_store_host_bytes": 0.0,
            "sglang:approx_kv_store_orphans": 0.0,
            "sglang:approx_kv_provisional_tokens": 0.0,
        }
        result = phase7_reset_invariant(partial, strict=False, clean_baseline=None)
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["missing_gauges"],
            ["sglang:approx_kv_store_leases"],
        )
        self.assertEqual(
            result["gauge_states"]["sglang:approx_kv_store_leases"],
            "missing",
        )
        self.assertFalse(result["components"]["leases"])


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _commit(cwd: Path, message: str) -> str:
    _git(cwd, "add", "--all")
    _git(
        cwd,
        "-c",
        "user.email=phase7@example.invalid",
        "-c",
        "user.name=phase7",
        "commit",
        "--quiet",
        "-m",
        message,
    )
    return _git(cwd, "rev-parse", "HEAD")


class TestPhase7ExecutionEnvelope(unittest.TestCase):
    """The pinned code SHA no longer has to equal the execution HEAD."""

    def setUp(self):
        self._workspace = tempfile.TemporaryDirectory(prefix="p7-envelope-")
        self.root = Path(self._workspace.name)
        self.addCleanup(self._workspace.cleanup)
        _git(self.root, "init", "--quiet", "-b", "main")
        (self.root / "benchmark/approx_kv/results/phase7").mkdir(parents=True)
        (self.root / "benchmark/approx_kv/run_p7_ceiling.py").write_text("runner\n")
        (self.root / PRIMARY_MANIFEST_REL).write_text('{"revision": 5}\n')
        (self.root / RESULT_MANIFEST_REL).write_text('{"runs": []}\n')
        self.pinned_sha = _commit(self.root, "pin")
        self.pinned_tree = _git(self.root, "rev-parse", "HEAD^{tree}")
        self.manifest = {
            "implementation": {
                "post_pin_envelope_allowlist": [
                    RESULT_MANIFEST_REL,
                    PRIMARY_MANIFEST_REL,
                ]
            }
        }

    def envelope(self):
        with patch.object(phase7_common, "REPO_ROOT", self.root):
            return execution_envelope(
                self.manifest,
                pinned_sha=self.pinned_sha,
                pinned_tree=self.pinned_tree,
            )

    def test_pinned_sha_may_be_an_ancestor_of_the_execution_head(self):
        (self.root / RESULT_MANIFEST_REL).write_text('{"runs": ["r0"]}\n')
        head_sha = _commit(self.root, "post-pin result envelope")
        envelope = self.envelope()
        self.assertEqual(envelope["pinned_source_git_sha"], self.pinned_sha)
        self.assertEqual(envelope["pinned_source_tree_sha"], self.pinned_tree)
        self.assertEqual(envelope["execution_head_git_sha"], head_sha)
        self.assertNotEqual(head_sha, self.pinned_sha)
        self.assertTrue(envelope["pinned_is_ancestor_of_execution_head"])
        self.assertEqual(envelope["post_pin_changed_paths"], [RESULT_MANIFEST_REL])
        self.assertEqual(
            sorted(envelope["post_pin_envelope_sha256"]),
            sorted([PRIMARY_MANIFEST_REL, RESULT_MANIFEST_REL]),
        )

    def test_head_equal_to_the_pin_is_still_accepted(self):
        envelope = self.envelope()
        self.assertEqual(
            envelope["execution_head_git_sha"],
            envelope["pinned_source_git_sha"],
        )
        self.assertEqual(envelope["post_pin_changed_paths"], [])

    def test_post_pin_source_change_is_rejected(self):
        (self.root / "benchmark/approx_kv/run_p7_ceiling.py").write_text("drift\n")
        _commit(self.root, "post-pin source drift")
        with self.assertRaisesRegex(Phase7ContractError, "outside the envelope"):
            self.envelope()

    def test_dirty_worktree_is_rejected(self):
        (self.root / RESULT_MANIFEST_REL).write_text('{"runs": ["dirty"]}\n')
        with self.assertRaisesRegex(Phase7ContractError, "worktree must be clean"):
            self.envelope()

    def test_non_ancestor_pin_is_rejected(self):
        _git(self.root, "checkout", "--quiet", "-b", "sibling", self.pinned_sha)
        (self.root / RESULT_MANIFEST_REL).write_text('{"runs": ["sibling"]}\n')
        sibling_sha = _commit(self.root, "sibling envelope")
        _git(self.root, "checkout", "--quiet", "main")
        (self.root / RESULT_MANIFEST_REL).write_text('{"runs": ["main"]}\n')
        _commit(self.root, "main envelope")
        self.pinned_sha = sibling_sha
        self.pinned_tree = _git(self.root, "rev-parse", f"{sibling_sha}^{{tree}}")
        with self.assertRaisesRegex(Phase7ContractError, "not an ancestor"):
            self.envelope()

    def test_allowlisted_path_must_match_the_execution_head_blob(self):
        (self.root / RESULT_MANIFEST_REL).write_text('{"runs": ["r0"]}\n')
        _commit(self.root, "post-pin result envelope")
        _git(self.root, "update-index", "--assume-unchanged", RESULT_MANIFEST_REL)
        (self.root / RESULT_MANIFEST_REL).write_text('{"runs": ["tampered"]}\n')
        try:
            with self.assertRaisesRegex(
                Phase7ContractError, "differs from the execution HEAD blob"
            ):
                self.envelope()
        finally:
            _git(
                self.root, "update-index", "--no-assume-unchanged", RESULT_MANIFEST_REL
            )

    def test_missing_allowlisted_path_is_rejected(self):
        (self.root / RESULT_MANIFEST_REL).unlink()
        _commit(self.root, "drop result manifest")
        self.assertFalse((self.root / RESULT_MANIFEST_REL).exists())
        with self.assertRaisesRegex(Phase7ContractError, "is missing"):
            self.envelope()

    def test_manifest_must_be_an_allowlisted_in_repo_envelope_path(self):
        with patch.object(phase7_common, "REPO_ROOT", self.root):
            self.assertEqual(
                require_envelope_path(
                    self.root / PRIMARY_MANIFEST_REL,
                    manifest=self.manifest,
                    field="manifest",
                ),
                PRIMARY_MANIFEST_REL,
            )
            with self.assertRaisesRegex(Phase7ContractError, "outside REPO_ROOT"):
                require_envelope_path(
                    MANIFEST_PATH,
                    manifest=self.manifest,
                    field="manifest",
                )
            with self.assertRaisesRegex(
                Phase7ContractError, "not a declared post-pin envelope path"
            ):
                require_envelope_path(
                    self.root / "benchmark/approx_kv/run_p7_ceiling.py",
                    manifest=self.manifest,
                    field="manifest",
                )

    def test_review_artifact_must_match_manifest_and_head_blob_hash(self):
        review_path = self.root / FINAL_REVIEW_REL
        review_path.write_text('{"verdict": "PASS"}\n')
        _commit(self.root, "add final review")
        self.manifest["implementation"]["post_pin_envelope_allowlist"].append(
            FINAL_REVIEW_REL
        )
        with patch.object(phase7_common, "REPO_ROOT", self.root):
            envelope = execution_envelope(
                self.manifest,
                pinned_sha=self.pinned_sha,
                pinned_tree=self.pinned_tree,
            )
            expected_sha = hashlib.sha256(review_path.read_bytes()).hexdigest()
            self.assertEqual(
                validate_evidence_artifact_binding(
                    manifest=self.manifest,
                    envelope=envelope,
                    artifact_path=FINAL_REVIEW_REL,
                    artifact_sha256=expected_sha,
                    field="review_contract",
                ),
                FINAL_REVIEW_REL,
            )
            with self.assertRaisesRegex(Phase7ContractError, "file SHA-256 mismatch"):
                validate_evidence_artifact_binding(
                    manifest=self.manifest,
                    envelope=envelope,
                    artifact_path=FINAL_REVIEW_REL,
                    artifact_sha256="0" * 64,
                    field="review_contract",
                )

    def test_authorized_runtime_binds_the_final_review_content(self):
        review = synthetic_final_review()
        review_path = self.root / FINAL_REVIEW_REL
        review_path.write_text(json.dumps(review, indent=2) + "\n")
        _commit(self.root, "add final review")
        self.manifest["implementation"]["post_pin_envelope_allowlist"].append(
            FINAL_REVIEW_REL
        )
        review_sha = hashlib.sha256(review_path.read_bytes()).hexdigest()
        self.manifest["implementation"].update(
            {
                "phase7_pinned_implementation_sha": review[
                    "reviewed_pinned_implementation_sha"
                ],
                "phase7_pinned_tree_sha": review["reviewed_pinned_tree_sha"],
            }
        )
        self.manifest.update(
            {
                "manifest_revision": 10,
                "supersedes_manifest_sha256": review["reviewed_manifest_sha256"],
                "design_payload_sha256": review["design_payload_sha256"],
                "runners": {
                    name: {"sha256": value}
                    for name, value in review["runner_sha256"].items()
                },
                "review_contract": {"artifact_path": FINAL_REVIEW_REL},
                "review_evidence": review_evidence_summary(
                    review,
                    FINAL_REVIEW_REL,
                    review_sha,
                ),
            }
        )
        with patch.object(phase7_common, "REPO_ROOT", self.root):
            envelope = execution_envelope(
                self.manifest,
                pinned_sha=self.pinned_sha,
                pinned_tree=self.pinned_tree,
            )
            bound = validate_final_review_binding(
                manifest=self.manifest,
                envelope=envelope,
            )
            self.assertEqual(bound["artifact_sha256"], review["artifact_sha256"])

            for mutation, message in (
                ({"design_payload_sha256": "3" * 64}, "design payload hash mismatch"),
                (
                    {"supersedes_manifest_sha256": "4" * 64},
                    "superseded manifest revision",
                ),
                ({"manifest_revision": 9}, "must be greater"),
                (
                    {
                        "implementation": {
                            **self.manifest["implementation"],
                            "phase7_pinned_implementation_sha": "f" * 40,
                        }
                    },
                    "pinned implementation SHA",
                ),
                (
                    {
                        "runners": {
                            **{
                                name: {"sha256": value}
                                for name, value in review["runner_sha256"].items()
                            },
                            "scheduler": {"sha256": "6" * 64},
                        }
                    },
                    "runner hash mismatch",
                ),
            ):
                drifted = copy.deepcopy(self.manifest)
                drifted.update(mutation)
                with self.assertRaisesRegex(Phase7ContractError, message):
                    validate_final_review_binding(
                        manifest=drifted,
                        envelope=envelope,
                    )

            pending = copy.deepcopy(self.manifest)
            pending["review_evidence"]["status"] = "pending"
            with self.assertRaisesRegex(Phase7ContractError, "has not passed"):
                validate_final_review_binding(manifest=pending, envelope=envelope)

            summary_drift = copy.deepcopy(self.manifest)
            summary_drift["review_evidence"]["verdict"] = "PASS_WITH_CAVEATS"
            with self.assertRaisesRegex(Phase7ContractError, "summary mismatch"):
                validate_final_review_binding(
                    manifest=summary_drift,
                    envelope=envelope,
                )

    def test_tampered_final_review_blob_is_rejected_at_runtime(self):
        review = synthetic_final_review()
        review_path = self.root / FINAL_REVIEW_REL
        review_path.write_text(json.dumps(review, indent=2) + "\n")
        _commit(self.root, "add final review")
        self.manifest["implementation"]["post_pin_envelope_allowlist"].append(
            FINAL_REVIEW_REL
        )
        review_sha = hashlib.sha256(review_path.read_bytes()).hexdigest()
        self.manifest["implementation"].update(
            {
                "phase7_pinned_implementation_sha": review[
                    "reviewed_pinned_implementation_sha"
                ],
                "phase7_pinned_tree_sha": review["reviewed_pinned_tree_sha"],
            }
        )
        self.manifest.update(
            {
                "manifest_revision": 10,
                "supersedes_manifest_sha256": review["reviewed_manifest_sha256"],
                "design_payload_sha256": review["design_payload_sha256"],
                "runners": {
                    name: {"sha256": value}
                    for name, value in review["runner_sha256"].items()
                },
                "review_contract": {"artifact_path": FINAL_REVIEW_REL},
                "review_evidence": review_evidence_summary(
                    review,
                    FINAL_REVIEW_REL,
                    review_sha,
                ),
            }
        )
        with patch.object(phase7_common, "REPO_ROOT", self.root):
            envelope = execution_envelope(
                self.manifest,
                pinned_sha=self.pinned_sha,
                pinned_tree=self.pinned_tree,
            )
            tampered = dict(review)
            tampered["open_p1"] = 3
            _git(self.root, "update-index", "--assume-unchanged", FINAL_REVIEW_REL)
            review_path.write_text(json.dumps(tampered, indent=2) + "\n")
            try:
                with self.assertRaisesRegex(
                    Phase7ContractError, "file SHA-256 mismatch"
                ):
                    validate_final_review_binding(
                        manifest=self.manifest,
                        envelope=envelope,
                    )
            finally:
                _git(
                    self.root,
                    "update-index",
                    "--no-assume-unchanged",
                    FINAL_REVIEW_REL,
                )

    def test_runner_test_evidence_must_be_a_versioned_head_blob(self):
        evidence_rel = (
            "benchmark/approx_kv/results/phase7/evidence/ceiling-cpu-tests.json"
        )
        evidence_path = self.root / evidence_rel
        evidence_path.parent.mkdir(parents=True)
        runner_path = self.root / "benchmark/approx_kv/run_p7_ceiling.py"
        image_digest = "sha256:" + "7" * 64
        payload = build_runner_test_evidence(
            runner_key="ceiling",
            runner_module=CEILING_RUNNER,
            runner_path="benchmark/approx_kv/run_p7_ceiling.py",
            runner_sha256=hashlib.sha256(runner_path.read_bytes()).hexdigest(),
            image_digest=image_digest,
            command=RUNNER_SPECS["ceiling"]["required_cpu_test"],
            exit_code=0,
            summary_line="12 passed, 2 subtests passed in 3.21s",
            passed_count=12,
            subtests_passed_count=2,
            subtests=["a", "b"],
            timestamp="2026-07-28T05:30:00+00:00",
        )
        evidence_path.write_text(json.dumps(payload) + "\n")
        _commit(self.root, "add CPU test evidence")
        with patch.object(phase7_manifest_builder, "REPO_ROOT", self.root):
            summary = load_versioned_runner_test_evidence(
                runner_key="ceiling",
                evidence_path=evidence_path,
                image_digest=image_digest,
            )
        self.assertEqual(summary["path"], evidence_rel)
        self.assertEqual(summary["passed_count"], 12)
        self.assertEqual(summary["exit_code"], 0)
        self.assertEqual(
            summary["summary_line"],
            "12 passed, 2 subtests passed in 3.21s",
        )
        self.assertEqual(
            summary["command"],
            RUNNER_SPECS["ceiling"]["required_cpu_test"],
        )

    def test_manifest_builder_rejects_a_foreign_cpu_test_command(self):
        evidence_rel = (
            "benchmark/approx_kv/results/phase7/evidence/ceiling-cpu-tests.json"
        )
        evidence_path = self.root / evidence_rel
        evidence_path.parent.mkdir(parents=True)
        runner_path = self.root / "benchmark/approx_kv/run_p7_ceiling.py"
        image_digest = "sha256:" + "7" * 64
        payload = build_runner_test_evidence(
            runner_key="ceiling",
            runner_module=CEILING_RUNNER,
            runner_path="benchmark/approx_kv/run_p7_ceiling.py",
            runner_sha256=hashlib.sha256(runner_path.read_bytes()).hexdigest(),
            image_digest=image_digest,
            command="python3 -m pytest -q some_other_test.py",
            exit_code=0,
            summary_line="12 passed",
            passed_count=12,
            subtests_passed_count=0,
            subtests=[],
            timestamp="2026-07-28T05:30:00+00:00",
        )
        evidence_path.write_text(json.dumps(payload) + "\n")
        _commit(self.root, "add foreign CPU test evidence")
        with patch.object(phase7_manifest_builder, "REPO_ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                load_versioned_runner_test_evidence(
                    runner_key="ceiling",
                    evidence_path=evidence_path,
                    image_digest=image_digest,
                )

    def test_allowlist_shape_is_validated(self):
        with self.assertRaisesRegex(Phase7ContractError, "does not declare"):
            post_pin_envelope_allowlist({"implementation": {}})
        with self.assertRaisesRegex(Phase7ContractError, "escapes the repository"):
            post_pin_envelope_allowlist(
                {
                    "implementation": {
                        "post_pin_envelope_allowlist": [
                            "benchmark/approx_kv/results/phase7/../../../etc/passwd"
                        ]
                    }
                }
            )
        with self.assertRaisesRegex(Phase7ContractError, "outside the result envelope"):
            post_pin_envelope_allowlist(
                {
                    "implementation": {
                        "post_pin_envelope_allowlist": ["python/sglang/srt/server.py"]
                    }
                }
            )
        with self.assertRaisesRegex(Phase7ContractError, "duplicates"):
            post_pin_envelope_allowlist(
                {
                    "implementation": {
                        "post_pin_envelope_allowlist": [
                            RESULT_MANIFEST_REL,
                            RESULT_MANIFEST_REL,
                        ]
                    }
                }
            )
        self.assertEqual(
            post_pin_envelope_allowlist(authorized_manifest()),
            (
                RESULT_MANIFEST_REL,
                PRIMARY_MANIFEST_REL,
                FINAL_REVIEW_REL,
                "benchmark/approx_kv/results/phase7/evidence/cpu.json",
            ),
        )


class TestPhase7ManifestReviewBinding(unittest.TestCase):
    """The manifest builder must bind the review to what it supersedes."""

    def setUp(self):
        self._workspace = tempfile.TemporaryDirectory(prefix="p7-review-")
        self.root = Path(self._workspace.name)
        self.addCleanup(self._workspace.cleanup)
        self.review_path = self.root / "phase7-final-opus-review.json"
        self.manifest = load_manifest()
        self.manifest["settings"] = build_settings()
        self.manifest["inactive_counter_pins"] = build_inactive_counter_pins()
        self.manifest["runners"]["capacity_pilot"] = dict(
            self.manifest["runners"]["ceiling"],
            module=CAPACITY_RUNNER,
            path="benchmark/approx_kv/run_p6_4_capacity_pilot.py",
        )

    def bind(self, **review_overrides):
        manifest = copy.deepcopy(self.manifest)
        implementation = manifest["implementation"]
        overrides = {
            "reviewed_manifest_revision": int(manifest["manifest_revision"]) - 1,
            "reviewed_manifest_sha256": manifest["supersedes_manifest_sha256"],
            "design_payload_sha256": manifest["design_payload_sha256"],
            "reviewed_pinned_implementation_sha": implementation[
                "phase7_pinned_implementation_sha"
            ],
            "reviewed_pinned_tree_sha": implementation["phase7_pinned_tree_sha"],
            "runner_sha256": {
                name: manifest["runners"][name]["sha256"] for name in RUNNER_SPECS
            },
        }
        overrides.update(review_overrides)
        review = synthetic_final_review(**overrides)
        self.review_path.write_text(json.dumps(review, indent=2) + "\n")
        artifact_sha = phase7_manifest_builder.sha256_file(self.review_path)
        manifest["review_contract"]["artifact_path"] = str(self.review_path)
        manifest["review_evidence"] = review_evidence_summary(
            review,
            str(self.review_path),
            artifact_sha,
        )
        return manifest, review

    def problems(self, manifest):
        with patch.object(
            phase7_manifest_builder,
            "FINAL_OPUS_REVIEW",
            self.review_path,
        ):
            return phase7_manifest_builder.validate(
                manifest,
                Namespace(check_plan=False),
            )

    def test_review_of_the_superseded_revision_is_accepted(self):
        manifest, _ = self.bind()
        problems = self.problems(manifest)
        self.assertFalse(
            [row for row in problems if "final Opus review" in row],
            problems,
        )

    def test_review_must_review_the_superseded_manifest_and_design(self):
        manifest, _ = self.bind(reviewed_manifest_sha256="4" * 64)
        self.assertTrue(
            [
                row
                for row in self.problems(manifest)
                if "not bound" in row and "superseded manifest revision" in row
            ]
        )
        manifest, _ = self.bind(design_payload_sha256="3" * 64)
        self.assertTrue(
            [
                row
                for row in self.problems(manifest)
                if "not bound" in row and "design payload hash mismatch" in row
            ]
        )
        manifest, _ = self.bind(
            runner_sha256={
                name: ("7" * 64 if name == "scheduler" else value)
                for name, value in {
                    key: self.manifest["runners"][key]["sha256"] for key in RUNNER_SPECS
                }.items()
            }
        )
        self.assertTrue(
            [
                row
                for row in self.problems(manifest)
                if "not bound" in row and "runner hash mismatch" in row
            ]
        )

    def test_review_may_not_review_the_revision_it_activates(self):
        manifest, _ = self.bind()
        manifest["manifest_revision"] = int(
            manifest["review_evidence"]["reviewed_manifest_revision"]
        )
        self.assertTrue(
            [
                row
                for row in self.problems(manifest)
                if "not bound" in row and "must be greater" in row
            ]
        )

    def test_recorded_review_summary_must_match_the_artifact(self):
        manifest, _ = self.bind()
        manifest["review_evidence"]["open_p1"] = 2
        self.assertTrue(
            [
                row
                for row in self.problems(manifest)
                if "final Opus review summary mismatch" in row
            ]
        )

    def test_missing_review_artifact_is_reported(self):
        manifest, _ = self.bind()
        self.review_path.unlink()
        self.assertTrue(
            [
                row
                for row in self.problems(manifest)
                if "final Opus review artifact is missing" in row
            ]
        )


class TestPhase7FrozenSegmentContract(unittest.TestCase):
    def test_segment_tokens_and_source_pin_are_frozen_in_the_manifest(self):
        workload = a8_tokens(revised_manifest(), body_tokens=1024)
        self.assertEqual(workload["segment_tokens_max"], 512)
        self.assertTrue(workload["source_pin_until_reset"])

        stale = revised_manifest()
        del stale["workloads"]["A8"]["segment_tokens_max"]
        with self.assertRaisesRegex(Phase7ContractError, "differs from frozen builder"):
            a8_tokens(stale, body_tokens=1024)

    def test_ceiling_runner_has_no_segment_tokens_override(self):
        argv = [
            "run_p7_ceiling.py",
            "--manifest",
            str(MANIFEST_PATH),
            "--setting-id",
            "p7-a8-r0-body1024-rho1.5",
            "--restart-index",
            "0",
            "--output",
            "out.json",
            "--central-log",
            "central.jsonl",
            "--log",
            "server.log",
            "--segment-tokens",
            "256",
        ]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                run_p7_ceiling.parse_args()
        with patch.object(sys, "argv", argv[:-2]):
            args = run_p7_ceiling.parse_args()
        self.assertFalse(hasattr(args, "segment_tokens"))

    def test_phase7_persistent_pin_env_and_pending_result_provenance(self):
        manifest = revised_manifest()
        plugin_env = manifest["server_template"]["plugin_env"]
        self.assertEqual(
            plugin_env["SGLANG_APPROX_KV_ALLOW_PERSISTENT_PINS"],
            "1",
        )
        self.assertEqual(
            plugin_env["SGLANG_APPROX_KV_MAX_PERSISTENT_PINS"],
            "16",
        )
        self.assertEqual(
            pending_result_provenance(),
            {
                "result_git_sha": None,
                "result_commit_status": "pending_result_commit",
            },
        )


if __name__ == "__main__":
    unittest.main()
