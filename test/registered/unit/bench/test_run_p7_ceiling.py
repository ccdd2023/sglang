from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-c-test-cpu")

from benchmark.approx_kv import run_p7_ceiling
from benchmark.approx_kv.build_phase7_manifest import (
    build_a8_workload,
    build_settings,
    build_w_workload,
    design_payload_sha256,
)
from benchmark.approx_kv.phase7 import common as phase7_common
from benchmark.approx_kv.phase7.common import (
    CEILING_RUNNER,
    Phase7ContractError,
    a8_tokens,
    classify_request_outcome,
    ensure_artifact_path_layout,
    execution_envelope,
    filler_pool_tokens,
    labeled_counter_observation,
    manifest_self_sha256,
    memory_footprint,
    pending_result_provenance,
    phase7_reset_invariant,
    post_pin_envelope_allowlist,
    require_envelope_path,
    select_filler_prefix,
    select_setting,
    terminal_reason_observations,
    validate_manifest_envelope,
    validate_outcome_record,
    validate_runner_binding,
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


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def authorized_manifest() -> dict:
    manifest = revised_manifest()
    manifest["status"] = "authorized"
    manifest["phase7_execution_authorized"] = True
    manifest["execution_blockers"] = []
    manifest["implementation"]["phase7_pinned_implementation_sha"] = "a" * 40
    manifest["implementation"]["phase7_pinned_tree_sha"] = "b" * 40
    manifest["implementation"]["post_pin_envelope_allowlist"] = [
        RESULT_MANIFEST_REL,
        PRIMARY_MANIFEST_REL,
    ]
    manifest["runners"]["ceiling"] = {
        "path": "benchmark/approx_kv/run_p7_ceiling.py",
        "exists": True,
        "sha256": "c" * 64,
        "required_cpu_test": "pytest",
        "cpu_test_status": "passed",
        "review_status": "reviewed",
    }
    manifest["preregistered_manifest_sha256"] = manifest_self_sha256(manifest)
    return manifest


def revised_manifest() -> dict:
    """Synthetic rev6 envelope carrying the post-review builder semantics.

    The committed rev5 artifact is intentionally left untouched; the main
    session will generate the real V6/rev6 manifest.
    """
    manifest = load_manifest()
    manifest["manifest_revision"] = 6
    manifest["plan"]["version"] = "V6"
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


class TestPhase7CeilingGuards(unittest.TestCase):
    def test_manifest_selection_and_authorization_guard(self):
        with self.assertRaisesRegex(Phase7ContractError, "at least 6"):
            validate_manifest_envelope(load_manifest(), require_authorized=False)

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

    def test_artifact_paths_must_be_distinct(self):
        path = REPO_ROOT / "phase7-path-placeholder"
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            ensure_artifact_path_layout(
                output=path,
                log=path,
                central_log=path.with_name("central.jsonl"),
            )


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
                'sglang:approx_kv_dense_fallback_total'
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
            (RESULT_MANIFEST_REL, PRIMARY_MANIFEST_REL),
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
