from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")

from benchmark.approx_kv import build_phase7_manifest as phase7_manifest_builder
from benchmark.approx_kv import run_p6_4_capacity_pilot
from benchmark.approx_kv.build_phase7_manifest import (
    RUNNER_SPECS,
    build_inactive_counter_pins,
    build_settings,
)
from benchmark.approx_kv.build_result_manifest import build_entries
from benchmark.approx_kv.phase6.manifest import build_fixed40_manifest
from benchmark.approx_kv.phase6.schema import payload_sha256
from benchmark.approx_kv.phase7 import common as phase7_common
from benchmark.approx_kv.phase7.common import (
    CAPACITY_RUNNER,
    Phase7ContractError,
    ensure_artifact_path_layout,
    execution_envelope,
    load_execution_context,
    terminal_reason_observations,
)
from benchmark.approx_kv.phase7.correction import (
    BASE_MANIFEST_PATH,
    CAPACITY_CORRECTION_CPU_EVIDENCE_PATH,
    CAPACITY_CORRECTION_MANIFEST_PATH,
    CAPACITY_CORRECTION_REVIEW_PATH,
    CAPACITY_CORRECTION_SETTING_ID,
    CAPACITY_RUNNER_PATH,
    ORIGINAL_RAW_PATH,
    build_authorized_capacity_correction_manifest,
    build_pinned_capacity_correction_manifest,
    correction_manifest_payload_sha256,
    design_key_value_bytes,
)
from benchmark.approx_kv.phase7.correction import file_sha256 as correction_file_sha256
from benchmark.approx_kv.phase7.correction import (
    load_capacity_cpu_evidence,
    pending_review_evidence,
    validate_capacity_correction_manifest,
)
from benchmark.approx_kv.phase7.correction_review import build_correction_review
from benchmark.approx_kv.phase7.evidence import build_runner_test_evidence
from benchmark.approx_kv.run_p6_0_contract import build_contract, verify_contract
from benchmark.approx_kv.run_p6_4_capacity_pilot import (
    configure_phase7_args,
    execution_cells,
    labeled_metric_delta,
    launch_cells,
    phase7_failure_artifact,
    phase7_mode_requested,
    phase7_terminal_reason_result,
    representation_metadata,
    run_phase7_repeat_major,
    run_profile,
)
from benchmark.approx_kv.run_p6_h_host_roundtrip import metadata as host_metadata

REPO_ROOT = Path(__file__).resolve().parents[4]
PHASE7_MANIFEST_PATH = (
    REPO_ROOT / "benchmark/approx_kv/results/phase7/phase7-primary-manifest.json"
)
CAPACITY_SETTING_ID = "p6delta-s4-rho2-chunk4096"


def capacity_setting(setting_id: str = CAPACITY_SETTING_ID) -> dict:
    return next(row for row in build_settings() if row["setting_id"] == setting_id)


def fake_round(profile: str, round_index: int) -> dict:
    dense_fallback = int(profile == "r0_like")
    return {
        "round_index": round_index,
        "profile": profile,
        "registered_segments": 4,
        "metrics": {
            "exact_evicted_bytes": 1.0,
            "approx_evicted_bytes": 1.0,
            "exact_requester_approx_victim_bytes": 1.0,
            "approx_requester_exact_victim_bytes": 1.0,
            "peak_device_bytes": 8.0,
        },
        "cache_outcomes": {
            "exact_gpu_hit": 1,
            "approximate_gpu_recovery": 1 - dense_fallback,
            "host_demand_load": 0,
            "dense_fallback": dense_fallback,
            "exact_cache_miss": 0,
            "unknown": 0,
        },
        "fallback_reachable": True,
        "replay": (
            [
                {
                    "outcome": "dense_fallback",
                    "terminal_reason": "unsupported",
                }
            ]
            if dense_fallback
            else []
        ),
        "reachability": "reachable",
        "reset_invariant": {"passed": True},
        "store_reset_gauges": {},
        "valid": True,
    }


def recording_run_round(calls: list) -> callable:
    def run_round(args, manifest, *, profile, representation_kinds, round_index):
        calls.append((profile, round_index))
        return fake_round(profile, round_index)

    return run_round


def diagnostic_recording_run_round(calls: list, token: str) -> callable:
    def run_round(args, manifest, *, profile, representation_kinds, round_index):
        calls.append((profile, round_index))
        row = fake_round(profile, round_index)
        row["reachability"] = token
        return row

    return run_round


def phase7_capacity_manifest() -> dict:
    plugin_env = {
        "SGLANG_APPROX_KV_BYTES_PER_TOKEN": "114688",
        "SGLANG_APPROX_KV_HOST_BUDGET_BYTES": "0",
    }
    return {
        "manifest_revision": 10,
        "preregistered_manifest_sha256": "a" * 64,
        "design_payload_sha256": "f" * 64,
        "plan": {"version": "V7"},
        "implementation": {},
        "environment": {
            "model": "manifest-model",
            "model_revision": "manifest-revision",
            "image_digest": "sha256:" + "1" * 64,
        },
        "server_template": {
            "restart_seeds": [17, 18, 19],
            "attention_backend": "torch_native",
            "sampling_backend": "pytorch",
            "plugin_env": plugin_env,
        },
        "required_inactive_counters": [
            "host_load",
            "prefetch_request",
            "prefetch_loaded_tokens",
            "async_load",
        ],
        "inactive_counter_pins": build_inactive_counter_pins(),
        "skipped_tracks": [
            "host_matrix",
            "prefetch_functionality",
            "prefetch_performance",
            "async_h2d_performance",
        ],
        "outcome_taxonomy": [
            "dense_no_reuse_baseline",
            "exact_gpu_hit",
            "ordinary_exact_cache_miss",
            "approximate_gpu_recovery",
            "host_demand_load",
            "approximate_recovery_failed_dense",
        ],
        "exclusive_terminal_reasons": [
            "cross_store_reservation_failed",
            "device_allocation_failed",
            "unsupported",
            "registration_failed",
            "prefix_gap",
        ],
    }


def phase7_capacity_context(setting: dict) -> SimpleNamespace:
    return SimpleNamespace(
        manifest=phase7_capacity_manifest(),
        setting=setting,
        restart_index=0,
        source={"source_git_sha": "b" * 40, "source_tree_sha": "c" * 40},
        envelope={},
        manifest_file_sha256="d" * 64,
        manifest_path=Path("/results/phase7/manifest.json"),
        runner_module="benchmark.approx_kv.run_p6_4_capacity_pilot",
        runner_path="benchmark/approx_kv/run_p6_4_capacity_pilot.py",
        runner_sha256="e" * 64,
        is_correction=False,
    )


def correction_cpu_summary(runner_sha256: str) -> dict:
    base = json.loads(PHASE7_MANIFEST_PATH.read_text())
    return {
        "status": "passed",
        "path": CAPACITY_CORRECTION_CPU_EVIDENCE_PATH,
        "file_sha256": "6" * 64,
        "artifact_sha256": "7" * 64,
        "runner_sha256": runner_sha256,
        "image_digest": base["environment"]["image_digest"],
        "command": RUNNER_SPECS["capacity_pilot"]["required_cpu_test"],
        "exit_code": 0,
        "summary_line": "42 passed in 1.00s",
        "passed_count": 42,
        "subtests": {"passed_count": 0, "names": []},
        "timestamp": "2026-07-28T16:00:00-07:00",
    }


def pinned_correction_manifest(runner_sha256: str = "e" * 64) -> dict:
    base = json.loads(PHASE7_MANIFEST_PATH.read_text())
    return build_pinned_capacity_correction_manifest(
        base_manifest=base,
        base_manifest_path=Path(BASE_MANIFEST_PATH),
        base_manifest_file_sha256="4" * 64,
        original_raw_file_sha256="5" * 64,
        correction_manifest_revision=1,
        correction_pinned_implementation_sha="8" * 40,
        correction_pinned_tree_sha="9" * 40,
        capacity_runner_sha256=runner_sha256,
        capacity_cpu_evidence=correction_cpu_summary(runner_sha256),
        manifest_generation_sha="a" * 40,
        manifest_generation_tree_sha="b" * 40,
    )


def correction_review(pinned: dict) -> dict:
    return build_correction_review(
        reviewer="Claude Opus 5 / Max Thinking / long context",
        model="claude-opus-5",
        verdict="PASS",
        open_p0=0,
        open_p1=0,
        reviewed_correction_manifest_revision=pinned["correction_manifest_revision"],
        reviewed_correction_manifest_sha256=pinned["correction_manifest_sha256"],
        base_manifest_revision=pinned["base_manifest_revision"],
        base_manifest_self_sha256=pinned["base_manifest_self_sha256"],
        base_manifest_design_sha256=pinned["base_manifest_design_sha256"],
        base_manifest_path=pinned["base_manifest_path"],
        reviewed_correction_pinned_implementation_sha=pinned[
            "correction_pinned_implementation_sha"
        ],
        reviewed_correction_pinned_tree_sha=pinned["correction_pinned_tree_sha"],
        capacity_runner_sha256=pinned["capacity_runner_sha256"],
        original_raw_sha256=pinned["original_raw_sha256"],
        scope=pinned["scope"],
        allowed_setting=pinned["allowed_setting"],
        restart=pinned["restart"],
        findings=[],
        disposition="all P0/P1 closed",
        timestamp="2026-07-28T16:05:00-07:00",
    )


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def commit(root: Path, message: str) -> str:
    git(root, "add", "--all")
    git(
        root,
        "-c",
        "user.name=phase7-correction-test",
        "-c",
        "user.email=phase7-correction@example.invalid",
        "commit",
        "--quiet",
        "-m",
        message,
    )
    return git(root, "rev-parse", "HEAD")


def write_json(root: Path, relative: str, payload: dict) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def build_runtime_correction_repo(root: Path) -> dict:
    for relative in (BASE_MANIFEST_PATH, ORIGINAL_RAW_PATH, CAPACITY_RUNNER_PATH):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO_ROOT / relative).read_bytes())
    write_json(
        root,
        "benchmark/approx_kv/results/phase7/RESULT_MANIFEST.json",
        {},
    )
    git(root, "init", "--quiet", "-b", "main")
    pin = commit(root, "correction code pin")
    tree = git(root, "rev-parse", "HEAD^{tree}")
    base = json.loads((root / BASE_MANIFEST_PATH).read_text())
    runner_sha = correction_file_sha256(root / CAPACITY_RUNNER_PATH)
    evidence = build_runner_test_evidence(
        runner_key="capacity_pilot",
        runner_module=CAPACITY_RUNNER,
        runner_path=CAPACITY_RUNNER_PATH,
        runner_sha256=runner_sha,
        image_digest=base["environment"]["image_digest"],
        command=RUNNER_SPECS["capacity_pilot"]["required_cpu_test"],
        exit_code=0,
        summary_line="49 passed in 4.00s",
        passed_count=49,
        subtests_passed_count=0,
        subtests=[],
        timestamp="2026-07-28T16:00:00-07:00",
    )
    evidence_path = write_json(
        root,
        CAPACITY_CORRECTION_CPU_EVIDENCE_PATH,
        evidence,
    )
    _, evidence_summary = load_capacity_cpu_evidence(
        evidence_path,
        runner_sha256=runner_sha,
        image_digest=base["environment"]["image_digest"],
        repo_root=root,
    )
    pinned = build_pinned_capacity_correction_manifest(
        base_manifest=base,
        base_manifest_path=Path(BASE_MANIFEST_PATH),
        base_manifest_file_sha256=correction_file_sha256(root / BASE_MANIFEST_PATH),
        original_raw_file_sha256=correction_file_sha256(root / ORIGINAL_RAW_PATH),
        correction_manifest_revision=1,
        correction_pinned_implementation_sha=pin,
        correction_pinned_tree_sha=tree,
        capacity_runner_sha256=runner_sha,
        capacity_cpu_evidence=evidence_summary,
        manifest_generation_sha=pin,
        manifest_generation_tree_sha=tree,
    )
    write_json(root, CAPACITY_CORRECTION_MANIFEST_PATH, pinned)
    commit(root, "pinned correction manifest")
    review = correction_review(pinned)
    review_path = write_json(root, CAPACITY_CORRECTION_REVIEW_PATH, review)
    generation_sha = commit(root, "correction review")
    generation_tree = git(root, "rev-parse", "HEAD^{tree}")
    authorized = build_authorized_capacity_correction_manifest(
        reviewed_manifest=pinned,
        review=review,
        review_path=review_path,
        repo_root=root,
        correction_manifest_revision=2,
        manifest_generation_sha=generation_sha,
        manifest_generation_tree_sha=generation_tree,
    )
    write_json(root, CAPACITY_CORRECTION_MANIFEST_PATH, authorized)
    commit(root, "authorized correction manifest")
    return authorized


def phase7_capacity_args(
    context: SimpleNamespace,
    *,
    capacity_tolerance: float = 1.0,
) -> SimpleNamespace:
    args = SimpleNamespace(
        log=Path("/results/phase7/logs/capacity.log"),
        log_dir=None,
        capacity_tolerance=capacity_tolerance,
        port=30011,
        server_start_timeout_s=1.0,
    )
    configure_phase7_args(args, context)
    return args


class TestPhase6Manifest(unittest.TestCase):
    def test_result_manifest_recurses_into_phase7_evidence(self):
        root = Path(__file__).resolve().parents[4]
        results = root / "benchmark/approx_kv/results/phase7"
        entries = build_entries(results, results / "RESULT_MANIFEST.json")
        files = {Path(entry["file"]).as_posix() for entry in entries}
        for name in (
            "ceiling-cpu.json",
            "scheduler-cpu.json",
            "capacity-pilot-cpu.json",
        ):
            self.assertTrue(
                any(path.endswith(f"/phase7/evidence/{name}") for path in files)
            )

    def test_phase7_result_manifest_has_versioned_gpu_artifacts(self):
        root = Path(__file__).resolve().parents[4]
        results = root / "benchmark/approx_kv/results/phase7"
        self.assertEqual(len(list((results / "raw").glob("*.json"))), 22)
        self.assertEqual(len(list((results / "compact").glob("*.json"))), 22)
        self.assertEqual(len(list((results / "logs").glob("*.log"))), 22)

    def test_phase7_final_disposition_is_self_hashed(self):
        root = Path(__file__).resolve().parents[4]
        path = root / "benchmark/approx_kv/results/phase7/PHASE7_FINAL_DISPOSITION.json"
        payload = json.loads(path.read_text())
        stored = payload.pop("disposition_sha256")
        observed = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(stored, observed)
        self.assertEqual(payload["open_findings"], {"P0": 0, "P1": 0})
        self.assertEqual(payload["publication_disposition"], "READY_WITH_CAVEATS")

    def test_manifest_is_deterministic_and_fixed(self):
        first = build_fixed40_manifest()
        second = build_fixed40_manifest()
        self.assertEqual(first, second)
        self.assertEqual(first["object_count"], 40)
        self.assertEqual(len({item["object_id"] for item in first["objects"]}), 40)
        lengths = {item["logical_tokens"] for item in first["objects"]}
        self.assertGreaterEqual(len(lengths), 2)
        workflow_lengths = {
            item["logical_tokens"]
            for item in first["objects"]
            if item["object_id"].startswith("workflow-")
        }
        self.assertEqual(workflow_lengths, {1024, 2048})
        self.assertTrue(all(item["token_ids_sha256"] for item in first["objects"]))

    def test_dead_live_identity_is_frozen(self):
        manifest = build_fixed40_manifest()
        dead = [item for item in manifest["objects"] if item["retired"]]
        live = [item for item in manifest["objects"] if item["active"]]
        self.assertEqual(len(dead), 12)
        self.assertEqual(len(live), 28)

    def test_contract_records_provisional_chunk_and_profiles(self):
        contract = build_contract(
            Namespace(
                source_git_sha="abc",
                source_tree_sha="tree",
                image_digest="sha256:image",
                model="model",
                model_revision="revision",
                chunked_prefill_size=1024,
                chunk_source="provisional_worst_case",
            )
        )
        self.assertEqual(contract["settings"]["chunk_source"], "provisional_worst_case")
        self.assertEqual(
            contract["representation_profiles"]["r4_like"]["resident_multiplicity"],
            5,
        )
        self.assertFalse(contract["performance_ranking_enabled"])
        self.assertIn("exact_only", contract["representation_profiles"])
        self.assertEqual(contract["workload"]["chunked_prefill_size"], 1024)
        contract["run_id"] = "test-run"
        verify_contract(contract)

    def test_contract_verification_rejects_drift(self):
        contract = build_contract(
            Namespace(
                source_git_sha="abc",
                source_tree_sha="tree",
                image_digest="sha256:image",
                model="model",
                model_revision="revision",
                chunked_prefill_size=1024,
                chunk_source="provisional_worst_case",
            )
        )
        contract["workload"]["objects"][0]["logical_tokens"] += 1
        with self.assertRaises(ValueError):
            verify_contract(contract)

    def test_capacity_runner_uses_segment_bounded_representations(self):
        item = build_fixed40_manifest()["objects"][1]
        metadata = representation_metadata(
            item,
            profile="r4_like",
            representation_index=4,
            object_kind="delta",
            round_index=0,
            segment_tokens_max=512,
        )
        self.assertEqual(sum(row["length"] for row in metadata["segments"]), 2048)
        self.assertTrue(all(row["length"] <= 512 for row in metadata["segments"]))
        self.assertEqual(
            len({row["object_id"] for row in metadata["segments"]}),
            len(metadata["segments"]),
        )
        self.assertTrue(
            all("rep3:" in row["dependencies"][0] for row in metadata["segments"])
        )

    def test_host_roundtrip_uses_actual_bounded_segments(self):
        payload = host_metadata(
            operation="register",
            content_hash="host",
            object_id="host",
            header_tokens=64,
            body_tokens=1024,
            object_kind="materialization_scratch",
            residency="device",
            segment_tokens=512,
        )
        self.assertEqual(len(payload["segments"]), 2)
        self.assertEqual(
            [segment["target_start"] for segment in payload["segments"]],
            [64, 576],
        )
        self.assertTrue(
            all(segment["length"] <= 512 for segment in payload["segments"])
        )

    def test_capacity_launch_order_pairs_s0_s4_at_rho_two(self):
        cells = launch_cells((1.1, 1.5, 2.0, 3.0))
        self.assertEqual(
            cells,
            [
                ("hierarchical", 1.1),
                ("hierarchical", 1.5),
                ("lru", 2.0),
                ("hierarchical", 2.0),
                ("hierarchical", 3.0),
            ],
        )

    def test_phase7_capacity_uses_one_manifest_selected_cell(self):
        setting = next(
            row
            for row in build_settings()
            if row["setting_id"] == "p6delta-s4-rho2-chunk4096"
        )
        manifest = {
            "environment": {
                "model": "manifest-model",
                "model_revision": "manifest-revision",
                "image_digest": "sha256:" + "1" * 64,
            },
            "server_template": {
                "restart_seeds": [17, 18, 19],
                "attention_backend": "torch_native",
                "sampling_backend": "pytorch",
                "plugin_env": {
                    "SGLANG_APPROX_KV_BYTES_PER_TOKEN": "114688",
                    "SGLANG_APPROX_KV_HOST_BUDGET_BYTES": "0",
                },
            },
        }
        context = SimpleNamespace(
            manifest=manifest,
            setting=setting,
            restart_index=0,
            source={"source_git_sha": "a" * 40},
        )
        args = SimpleNamespace(
            log=Path("/results/phase7/logs/capacity.log"),
            log_dir=None,
        )
        configure_phase7_args(args, context)
        self.assertEqual(execution_cells(args), [("hierarchical", 2.0)])
        self.assertEqual(args.model, "manifest-model")
        self.assertEqual(args.chunked_prefill_size, 4096)
        self.assertEqual(args.chunk_source, "cl2")
        self.assertEqual(args.phase7_max_total_tokens, 11392)
        self.assertEqual(args.mem_fraction_static, 0.65)
        self.assertEqual(args.profiles, ",".join(setting["arms"]))
        self.assertEqual(args.formal_repeats, 2)
        self.assertEqual(args.phase7_server_seed, 17)

    def test_historical_capacity_matrix_is_unchanged(self):
        args = SimpleNamespace(rhos="1.1,1.5,2.0,3.0")
        self.assertEqual(
            execution_cells(args),
            launch_cells((1.1, 1.5, 2.0, 3.0)),
        )

    def test_phase7_cli_group_is_all_or_nothing(self):
        args = SimpleNamespace(
            phase7_manifest=Path("manifest.json"),
            phase7_setting_id=None,
            phase7_restart_index=None,
        )
        with self.assertRaisesRegex(Phase7ContractError, "provided together"):
            phase7_mode_requested(args)

    def test_phase7_correction_sha_is_phase7_only_and_validated(self):
        historical = SimpleNamespace(
            phase7_manifest=None,
            phase7_setting_id=None,
            phase7_restart_index=None,
            phase7_correction_of_raw_sha256="a" * 64,
        )
        with self.assertRaisesRegex(Phase7ContractError, "only valid in Phase7"):
            phase7_mode_requested(historical)
        invalid = SimpleNamespace(
            phase7_manifest=Path("manifest.json"),
            phase7_setting_id=CAPACITY_SETTING_ID,
            phase7_restart_index=0,
            phase7_correction_of_raw_sha256="not-a-sha",
        )
        with self.assertRaisesRegex(Phase7ContractError, "lowercase SHA-256"):
            phase7_mode_requested(invalid)
        valid = copy.copy(invalid)
        valid.phase7_correction_of_raw_sha256 = "a" * 64
        self.assertTrue(phase7_mode_requested(valid))

    def test_capacity_correction_pinned_review_authorized_chain(self):
        base = json.loads(PHASE7_MANIFEST_PATH.read_text())
        pinned = pinned_correction_manifest()
        validate_capacity_correction_manifest(
            pinned,
            base_manifest=base,
            require_authorized=False,
        )
        self.assertEqual(pinned["review_evidence"], pending_review_evidence())
        review = correction_review(pinned)
        with tempfile.TemporaryDirectory(prefix="phase7-correction-review-") as raw:
            root = Path(raw)
            review_path = root / CAPACITY_CORRECTION_REVIEW_PATH
            review_path.parent.mkdir(parents=True)
            review_path.write_text(json.dumps(review, indent=2) + "\n")
            authorized = build_authorized_capacity_correction_manifest(
                reviewed_manifest=pinned,
                review=review,
                review_path=review_path,
                repo_root=root,
                correction_manifest_revision=2,
                manifest_generation_sha="c" * 40,
                manifest_generation_tree_sha="d" * 40,
            )
        validate_capacity_correction_manifest(
            authorized,
            base_manifest=base,
            require_authorized=True,
            review=review,
        )
        self.assertEqual(authorized["status"], "authorized_correction")
        self.assertTrue(authorized["phase7_execution_authorized"])
        self.assertEqual(
            authorized["supersedes_correction_manifest_sha256"],
            pinned["correction_manifest_sha256"],
        )
        self.assertEqual(
            review["reviewed_correction_manifest_sha256"],
            pinned["correction_manifest_sha256"],
        )
        self.assertNotEqual(
            review["reviewed_correction_manifest_sha256"],
            authorized["correction_manifest_sha256"],
        )

    def test_capacity_correction_keeps_base_design_values_byte_identical(self):
        base = json.loads(PHASE7_MANIFEST_PATH.read_text())
        pinned = pinned_correction_manifest()
        for key in base["design_keys"]:
            self.assertEqual(
                design_key_value_bytes(pinned, key),
                design_key_value_bytes(base, key),
            )
        drifted = copy.deepcopy(pinned)
        drifted["statistics"] = dict(reversed(drifted["statistics"].items()))
        drifted["correction_manifest_sha256"] = correction_manifest_payload_sha256(
            drifted
        )
        with self.assertRaisesRegex(ValueError, "not byte-identical"):
            validate_capacity_correction_manifest(
                drifted,
                base_manifest=base,
                require_authorized=False,
            )

    def test_capacity_correction_rejects_wrong_scope_pin_and_review(self):
        base = json.loads(PHASE7_MANIFEST_PATH.read_text())
        pinned = pinned_correction_manifest()
        for field, value, message in (
            ("allowed_setting", CAPACITY_SETTING_ID, "scope binding"),
            ("original_raw_sha256", "0" * 64, "scope binding"),
            (
                "correction_pinned_implementation_sha",
                base["implementation"]["phase7_pinned_implementation_sha"],
                "new implementation pin",
            ),
        ):
            drifted = copy.deepcopy(pinned)
            drifted[field] = value
            drifted["correction_manifest_sha256"] = correction_manifest_payload_sha256(
                drifted
            )
            with self.assertRaisesRegex(ValueError, message):
                validate_capacity_correction_manifest(
                    drifted,
                    base_manifest=base,
                    require_authorized=False,
                )

        review = correction_review(pinned)
        bad_review = copy.deepcopy(review)
        bad_review["capacity_runner_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory(prefix="phase7-correction-review-") as raw:
            root = Path(raw)
            review_path = root / CAPACITY_CORRECTION_REVIEW_PATH
            review_path.parent.mkdir(parents=True)
            review_path.write_text(json.dumps(bad_review, indent=2) + "\n")
            with self.assertRaisesRegex(ValueError, "self-hash|binding"):
                build_authorized_capacity_correction_manifest(
                    reviewed_manifest=pinned,
                    review=bad_review,
                    review_path=review_path,
                    repo_root=root,
                    correction_manifest_revision=2,
                    manifest_generation_sha="c" * 40,
                    manifest_generation_tree_sha="d" * 40,
                )

    def test_runtime_routes_correction_only_through_dedicated_manifest(self):
        with self.assertRaisesRegex(
            Phase7ContractError,
            "dedicated correction manifest",
        ):
            load_execution_context(
                manifest_path=PHASE7_MANIFEST_PATH,
                setting_id=CAPACITY_CORRECTION_SETTING_ID,
                restart_index=0,
                runner_key="capacity_pilot",
                runner_module=CAPACITY_RUNNER,
                runner_file=REPO_ROOT / CAPACITY_RUNNER_PATH,
                correction_of_raw_sha256="a" * 64,
            )
        pinned = pinned_correction_manifest()
        with tempfile.TemporaryDirectory(prefix="phase7-correction-runtime-") as raw:
            path = Path(raw) / "correction.json"
            path.write_text(json.dumps(pinned))
            with self.assertRaisesRegex(Phase7ContractError, "original raw"):
                load_execution_context(
                    manifest_path=path,
                    setting_id=CAPACITY_CORRECTION_SETTING_ID,
                    restart_index=0,
                    runner_key="capacity_pilot",
                    runner_module=CAPACITY_RUNNER,
                    runner_file=REPO_ROOT / CAPACITY_RUNNER_PATH,
                    correction_of_raw_sha256="a" * 64,
                )
            with self.assertRaisesRegex(Phase7ContractError, "S0 setting/restart"):
                load_execution_context(
                    manifest_path=path,
                    setting_id=CAPACITY_SETTING_ID,
                    restart_index=0,
                    runner_key="capacity_pilot",
                    runner_module=CAPACITY_RUNNER,
                    runner_file=REPO_ROOT / CAPACITY_RUNNER_PATH,
                    correction_of_raw_sha256=pinned["original_raw_sha256"],
                )
            with self.assertRaisesRegex(Phase7ContractError, "requires"):
                load_execution_context(
                    manifest_path=path,
                    setting_id=CAPACITY_CORRECTION_SETTING_ID,
                    restart_index=0,
                    runner_key="capacity_pilot",
                    runner_module=CAPACITY_RUNNER,
                    runner_file=REPO_ROOT / CAPACITY_RUNNER_PATH,
                )

    def test_runtime_accepts_authorized_correction_with_dedicated_review(self):
        with tempfile.TemporaryDirectory(prefix="phase7-correction-runtime-") as raw:
            root = Path(raw)
            manifest = build_runtime_correction_repo(root)
            with (
                patch.object(phase7_common, "REPO_ROOT", root),
                patch.object(
                    phase7_common,
                    "require_read_only_implementation_worktree",
                ),
            ):
                context = load_execution_context(
                    manifest_path=root / CAPACITY_CORRECTION_MANIFEST_PATH,
                    setting_id=CAPACITY_CORRECTION_SETTING_ID,
                    restart_index=0,
                    runner_key="capacity_pilot",
                    runner_module=CAPACITY_RUNNER,
                    runner_file=root / CAPACITY_RUNNER_PATH,
                    correction_of_raw_sha256=manifest["original_raw_sha256"],
                )
        self.assertTrue(context.is_correction)
        self.assertEqual(
            context.manifest_self_sha256,
            manifest["correction_manifest_sha256"],
        )
        self.assertEqual(
            context.runtime_staging_root,
            "/results/phase7-capacity-correction",
        )
        ensure_artifact_path_layout(
            output=Path("/results/phase7-capacity-correction/raw/correction.json"),
            log=Path("/results/phase7-capacity-correction/logs/correction.log"),
            central_log=Path("/results/phase7-capacity-correction/phase7-runs.jsonl"),
            staging_root=context.runtime_staging_root,
        )

    def test_phase7_correction_envelope_allows_only_post_pin_artifacts(self):
        manifest_path = CAPACITY_CORRECTION_MANIFEST_PATH
        manifest = {
            "artifact": "phase7-capacity-correction-manifest",
            "post_pin_allowlist": [manifest_path],
        }
        pinned_sha = "1" * 40
        pinned_tree = "2" * 40
        head_sha = "3" * 40
        head_tree = "4" * 40

        def git_text(*args, status="", changed=manifest_path):
            responses = {
                ("status", "--porcelain", "--untracked-files=all"): status,
                ("rev-parse", "HEAD"): head_sha,
                ("rev-parse", "HEAD^{tree}"): head_tree,
                ("rev-parse", f"{pinned_sha}^{{commit}}"): pinned_sha,
                ("rev-parse", f"{pinned_sha}^{{tree}}"): pinned_tree,
                ("diff", "--name-only", f"{pinned_sha}..{head_sha}"): changed,
            }
            return responses[args]

        with (
            patch.object(
                phase7_common.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0),
            ),
            patch.object(
                phase7_common,
                "_verify_envelope_path",
                return_value="a" * 64,
            ),
        ):
            with (
                patch.object(
                    phase7_common,
                    "_git_text",
                    side_effect=lambda *args: git_text(
                        *args,
                        status=f" M {manifest_path}",
                    ),
                ),
                self.assertRaisesRegex(
                    Phase7ContractError,
                    "worktree must be clean",
                ),
            ):
                execution_envelope(
                    manifest,
                    pinned_sha=pinned_sha,
                    pinned_tree=pinned_tree,
                    execution_kind="capacity_correction",
                )
            with patch.object(phase7_common, "_git_text", side_effect=git_text):
                correction = execution_envelope(
                    manifest,
                    pinned_sha=pinned_sha,
                    pinned_tree=pinned_tree,
                    execution_kind="capacity_correction",
                )
            with (
                patch.object(
                    phase7_common,
                    "_git_text",
                    side_effect=lambda *args: git_text(
                        *args,
                        changed="benchmark/approx_kv/unlisted.py",
                    ),
                ),
                self.assertRaisesRegex(
                    Phase7ContractError,
                    "outside the envelope allowlist",
                ),
            ):
                execution_envelope(
                    manifest,
                    pinned_sha=pinned_sha,
                    pinned_tree=pinned_tree,
                    execution_kind="capacity_correction",
                )
        self.assertTrue(correction["worktree_clean"])
        self.assertEqual(correction["worktree_status_entries"], [])
        self.assertEqual(
            correction["post_pin_envelope_allowlist"],
            [manifest_path],
        )

    def test_phase7_terminal_reason_requires_one_direct_positive_reason(self):
        mapped = terminal_reason_observations(
            "",
            'sglang:approx_kv_dense_fallback_total{reason="store_miss"} 1024\n',
        )
        self.assertEqual(mapped["mapped"]["unsupported"], 1024)
        self.assertEqual(mapped["mapped_from"]["unsupported"], ["store_miss"])
        store_miss = {
            "verification": "direct",
            "mapped": {
                "cross_store_reservation_failed": None,
                "device_allocation_failed": None,
                "unsupported": 1024.0,
                "registration_failed": None,
                "prefix_gap": None,
            },
            "unmapped_raw_reasons": {},
        }
        result = phase7_terminal_reason_result("dense_fallback", store_miss)
        self.assertEqual(result["terminal_reason"], "unsupported")
        self.assertTrue(result["terminal_reason_valid"])

        missing = copy.deepcopy(store_miss)
        missing["mapped"]["unsupported"] = None
        self.assertFalse(
            phase7_terminal_reason_result("dense_fallback", missing)[
                "terminal_reason_valid"
            ]
        )
        multiple = copy.deepcopy(store_miss)
        multiple["mapped"]["prefix_gap"] = 1.0
        self.assertFalse(
            phase7_terminal_reason_result("dense_fallback", multiple)[
                "terminal_reason_valid"
            ]
        )

    def test_unauthorized_phase7_capacity_stops_before_server_launch(self):
        args = SimpleNamespace(
            phase7_manifest=Path("manifest.json"),
            phase7_setting_id="p6delta-s0-rho2-chunk4096",
            phase7_restart_index=0,
            output=Path("/results/phase7/raw/capacity.json"),
            central_log=Path("/results/phase7/central.jsonl"),
            log=Path("/results/phase7/logs/capacity.log"),
            log_dir=None,
        )
        with (
            patch.object(
                run_p6_4_capacity_pilot,
                "parse_args",
                return_value=args,
            ),
            patch.object(
                run_p6_4_capacity_pilot,
                "load_execution_context",
                side_effect=Phase7ContractError("unauthorized"),
            ),
            patch.object(run_p6_4_capacity_pilot, "launch_server") as launch_server,
        ):
            with self.assertRaisesRegex(Phase7ContractError, "unauthorized"):
                run_p6_4_capacity_pilot.main()
        launch_server.assert_not_called()

    def test_phase7_capacity_failure_artifact_has_phase7_binding(self):
        setting = next(
            row
            for row in build_settings()
            if row["setting_id"] == "p6delta-s0-rho2-chunk4096"
        )
        plugin_env = {
            "SGLANG_APPROX_KV_BYTES_PER_TOKEN": "114688",
            "SGLANG_APPROX_KV_HOST_BUDGET_BYTES": "0",
        }
        manifest = {
            "environment": {"image_digest": "sha256:" + "1" * 64},
            "server_template": {"plugin_env": plugin_env},
            "required_inactive_counters": [
                "host_load",
                "prefetch_request",
                "prefetch_loaded_tokens",
                "async_load",
            ],
            "inactive_counter_pins": build_inactive_counter_pins(),
            "skipped_tracks": [
                "host_matrix",
                "prefetch_functionality",
                "prefetch_performance",
                "async_h2d_performance",
            ],
            "outcome_taxonomy": [
                "dense_no_reuse_baseline",
                "exact_gpu_hit",
                "ordinary_exact_cache_miss",
                "approximate_gpu_recovery",
                "host_demand_load",
                "approximate_recovery_failed_dense",
            ],
            "exclusive_terminal_reasons": [
                "cross_store_reservation_failed",
                "device_allocation_failed",
                "unsupported",
                "registration_failed",
                "prefix_gap",
            ],
            "implementation": {},
            "plan": {"version": "V7"},
            "manifest_revision": 10,
            "preregistered_manifest_sha256": "a" * 64,
            "design_payload_sha256": "f" * 64,
        }
        context = SimpleNamespace(
            manifest=manifest,
            setting=setting,
            restart_index=0,
            source={"source_git_sha": "b" * 40, "source_tree_sha": "c" * 40},
            envelope={},
            manifest_file_sha256="d" * 64,
            manifest_path=Path("manifest.json"),
            runner_module="benchmark.approx_kv.run_p6_4_capacity_pilot",
            runner_path="benchmark/approx_kv/run_p6_4_capacity_pilot.py",
            runner_sha256="e" * 64,
        )
        args = SimpleNamespace(
            log=Path("/results/phase7/logs/not-created.log"),
            phase7_plugin_env=plugin_env,
            kv_bytes_per_token=114688,
            phase7_policy="lru",
            phase7_rho=2.0,
            chunked_prefill_size=4096,
            phase7_max_total_tokens=11392,
            mem_fraction_static=0.65,
            formal_repeats=2,
            phase7_server_seed=17,
        )
        artifact = phase7_failure_artifact(
            args=args,
            context=context,
            run_id="phase7-capacity-test",
            error=RuntimeError("synthetic"),
        )
        self.assertTrue(artifact["phase7_mode"])
        self.assertEqual(artifact["setting_id"], setting["setting_id"])
        self.assertIn("inactive_counter_assertion", artifact)
        self.assertEqual(
            artifact["provenance"]["runner_sha256"],
            "e" * 64,
        )

    def test_phase7_capacity_correction_artifact_binds_original_raw(self):
        setting = capacity_setting(CAPACITY_CORRECTION_SETTING_ID)
        manifest = pinned_correction_manifest()
        context = SimpleNamespace(
            manifest=manifest,
            setting=setting,
            restart_index=0,
            source={"source_git_sha": "8" * 40, "source_tree_sha": "9" * 40},
            envelope={},
            manifest_file_sha256="d" * 64,
            manifest_path=Path(CAPACITY_CORRECTION_MANIFEST_PATH),
            runner_module=CAPACITY_RUNNER,
            runner_path=CAPACITY_RUNNER_PATH,
            runner_sha256="e" * 64,
            is_correction=True,
        )
        args = phase7_capacity_args(context)
        args.phase7_correction_of_raw_sha256 = manifest["original_raw_sha256"]
        artifact = phase7_failure_artifact(
            args=args,
            context=context,
            run_id="phase7-capacity-correction-test",
            error=RuntimeError("synthetic"),
        )
        self.assertEqual(
            artifact["correction"],
            {
                "scope": "capacity_terminal_reason",
                "original_raw_sha256": manifest["original_raw_sha256"],
                "setting_id": setting["setting_id"],
                "restart_index": 0,
            },
        )
        self.assertEqual(
            artifact["design_payload_sha256"],
            manifest["base_manifest_design_sha256"],
        )
        self.assertEqual(
            artifact["correction_manifest_sha256"],
            manifest["correction_manifest_sha256"],
        )
        self.assertNotIn("correction_manifest", artifact)

    def test_phase7_capacity_formal_repeats_are_repeat_major(self):
        setting = capacity_setting()
        arms = list(setting["arms"])
        calls: list = []
        args = SimpleNamespace(formal_repeats=int(setting["formal_repeats"]))
        with patch.object(
            run_p6_4_capacity_pilot,
            "run_round",
            side_effect=recording_run_round(calls),
        ):
            profiles, formal_repeats = run_phase7_repeat_major(
                args,
                {},
                profiles=tuple(arms),
                setting=setting,
            )

        warmup_calls = [row for row in calls if row[1] < 0]
        formal_calls = [row for row in calls if row[1] >= 0]
        self.assertEqual([row[0] for row in warmup_calls], arms)
        self.assertEqual({row[1] for row in warmup_calls}, {-1})
        self.assertEqual(
            formal_calls,
            [(arm, 0) for arm in arms] + [(arm, 1) for arm in reversed(arms)],
        )
        self.assertEqual([row["repeat_index"] for row in formal_repeats], [0, 1])
        self.assertEqual(formal_repeats[0]["arm_order"], arms)
        self.assertEqual(formal_repeats[1]["arm_order"], list(reversed(arms)))
        self.assertEqual(
            [row["profile"] for row in formal_repeats[0]["profiles"]],
            arms,
        )
        self.assertEqual(
            [row["profile"] for row in formal_repeats[1]["profiles"]],
            list(reversed(arms)),
        )
        self.assertEqual(
            [row["execution_index"] for row in formal_repeats[1]["profiles"]],
            list(range(len(arms))),
        )

        rounds = {
            (row["profile"], round_row["round_index"]): round_row
            for row in profiles
            for round_row in row["formal"]
        }
        for repeat in formal_repeats:
            for entry in repeat["profiles"]:
                self.assertEqual(
                    entry["round_sha256"],
                    payload_sha256(rounds[(entry["profile"], entry["round_index"])]),
                )
        self.assertEqual([row["profile"] for row in profiles], arms)
        for row in profiles:
            self.assertEqual([r["round_index"] for r in row["formal"]], [0, 1])
            self.assertEqual(len(row["warmup"]), 1)

    def test_phase7_capacity_rejects_an_incomplete_arm_order(self):
        setting = dict(capacity_setting())
        setting["arm_order_by_repeat"] = {
            "0": list(setting["arms"]),
            "1": list(setting["arms"])[:-1],
        }
        args = SimpleNamespace(formal_repeats=int(setting["formal_repeats"]))
        with patch.object(
            run_p6_4_capacity_pilot,
            "run_round",
            side_effect=recording_run_round([]),
        ):
            with self.assertRaisesRegex(Phase7ContractError, "invalid arm order"):
                run_phase7_repeat_major(
                    args,
                    {},
                    profiles=tuple(setting["arms"]),
                    setting=setting,
                )

    def test_phase7_capacity_rejects_profiles_outside_the_frozen_arms(self):
        setting = capacity_setting()
        args = SimpleNamespace(formal_repeats=int(setting["formal_repeats"]))
        with patch.object(
            run_p6_4_capacity_pilot,
            "run_round",
            side_effect=recording_run_round([]),
        ):
            with self.assertRaisesRegex(
                Phase7ContractError, "differ from the preregistered arms"
            ):
                run_phase7_repeat_major(
                    args,
                    {},
                    profiles=tuple(setting["arms"])[:-1],
                    setting=setting,
                )

    def test_phase7_capacity_warmup_repeats_follow_the_setting(self):
        setting = dict(capacity_setting())
        setting["warmup_repeats"] = 2
        arms = list(setting["arms"])
        calls: list = []
        args = SimpleNamespace(formal_repeats=int(setting["formal_repeats"]))
        with patch.object(
            run_p6_4_capacity_pilot,
            "run_round",
            side_effect=recording_run_round(calls),
        ):
            profiles, _ = run_phase7_repeat_major(
                args,
                {},
                profiles=tuple(arms),
                setting=setting,
            )
        warmup_calls = [row for row in calls if row[1] < 0]
        self.assertEqual([row[0] for row in warmup_calls], arms * 2)
        self.assertEqual([row[1] for row in warmup_calls], [-1] * 5 + [-2] * 5)
        for row in profiles:
            self.assertEqual(len(row["warmup"]), 2)
            self.assertEqual(row["warmup_repeats"], 2)

    def test_historical_capacity_execution_stays_profile_major(self):
        calls: list = []
        args = SimpleNamespace(formal_repeats=2)
        with patch.object(
            run_p6_4_capacity_pilot,
            "run_round",
            side_effect=recording_run_round(calls),
        ):
            result = run_profile(
                args,
                {},
                profile="exact_only",
                representation_kinds=(),
            )
        self.assertEqual(
            calls, [("exact_only", -1), ("exact_only", 0), ("exact_only", 1)]
        )
        self.assertIsInstance(result["warmup"], dict)
        self.assertNotIn("warmup_repeats", result)
        self.assertEqual([row["round_index"] for row in result["formal"]], [0, 1])

    def test_phase7_capacity_tolerance_overrides_the_cli(self):
        setting = capacity_setting()
        self.assertEqual(setting["capacity_relative_error_tolerance"], 0.05)
        context = phase7_capacity_context(setting)
        args = phase7_capacity_args(context, capacity_tolerance=1.0)
        self.assertEqual(args.capacity_tolerance, 0.05)
        self.assertEqual(args.warmup_repeats, int(setting["warmup_repeats"]))

    def test_phase7_capacity_tolerance_must_be_frozen_in_the_setting(self):
        setting = dict(capacity_setting())
        setting["capacity_relative_error_tolerance"] = None
        context = phase7_capacity_context(setting)
        args = SimpleNamespace(
            log=Path("/results/phase7/logs/capacity.log"),
            log_dir=None,
            capacity_tolerance=1.0,
        )
        with self.assertRaisesRegex(Phase7ContractError, "capacity tolerance"):
            configure_phase7_args(args, context)

    def test_builder_requires_a_bounded_capacity_tolerance(self):
        manifest = json.loads(PHASE7_MANIFEST_PATH.read_text())
        manifest["settings"] = build_settings()
        manifest["inactive_counter_pins"] = build_inactive_counter_pins()
        manifest["runners"]["capacity_pilot"] = dict(
            manifest["runners"]["ceiling"],
            module=RUNNER_SPECS["capacity_pilot"]["module"],
            path=RUNNER_SPECS["capacity_pilot"]["path"],
        )
        check = Namespace(check_plan=False)
        clean = phase7_manifest_builder.validate(copy.deepcopy(manifest), check)
        self.assertFalse([row for row in clean if "capacity tolerance" in row])
        self.assertFalse([row for row in clean if "capacity_relative_error" in row])

        for setting_id, value, expected in (
            (CAPACITY_SETTING_ID, None, "lacks a frozen"),
            (CAPACITY_SETTING_ID, 0.2, "must be within"),
            ("p6delta-s0-rho2-chunk4096", -0.01, "must be within"),
        ):
            drifted = copy.deepcopy(manifest)
            for row in drifted["settings"]:
                if row["setting_id"] == setting_id:
                    row["capacity_relative_error_tolerance"] = value
            problems = phase7_manifest_builder.validate(drifted, check)
            self.assertTrue(
                [
                    row
                    for row in problems
                    if row.startswith(setting_id) and expected in row
                ],
                problems,
            )

    def test_phase7_capacity_artifact_records_repeat_major_execution(self):
        setting = capacity_setting()
        arms = list(setting["arms"])
        context = phase7_capacity_context(setting)
        args = phase7_capacity_args(context)
        server = SimpleNamespace(
            command=["python3", "-m", "sglang.launch_server"], plugin_env={}
        )
        calls: list = []
        with (
            patch.object(run_p6_4_capacity_pilot, "launch_server", return_value=server),
            patch.object(run_p6_4_capacity_pilot, "wait_ready"),
            patch.object(run_p6_4_capacity_pilot, "stop_server"),
            patch.object(
                run_p6_4_capacity_pilot,
                "metric_snapshot",
                return_value={"sglang:max_total_num_tokens": 11392},
            ),
            patch.object(run_p6_4_capacity_pilot, "metric_text", return_value=""),
            patch.object(run_p6_4_capacity_pilot, "machine_manifest", return_value={}),
            patch.object(
                run_p6_4_capacity_pilot,
                "run_round",
                side_effect=recording_run_round(calls),
            ),
        ):
            payload = run_p6_4_capacity_pilot.execute(args, "p7-capacity-test")

        self.assertEqual(payload["phase"], "Phase7-capacity")
        self.assertEqual(payload["warmup_repeats"], int(setting["warmup_repeats"]))
        self.assertEqual(payload["formal_repeats"], int(setting["formal_repeats"]))
        cell = payload["cells"][0]
        self.assertEqual(cell["execution_order"], "repeat_major")
        self.assertEqual(
            [row["arm_order"] for row in cell["formal_repeats"]],
            [arms, list(reversed(arms))],
        )
        self.assertEqual(cell["capacity_relative_error_tolerance"], 0.05)
        self.assertEqual([row["profile"] for row in cell["profiles"]], arms)
        parameters = payload["phase7_parameters"]
        self.assertEqual(parameters["capacity_relative_error_tolerance"], 0.05)
        self.assertEqual(parameters["warmup_repeats"], int(setting["warmup_repeats"]))
        self.assertEqual(parameters["execution_order"], "repeat_major")
        self.assertEqual(
            parameters["arm_order_by_repeat"],
            setting["arm_order_by_repeat"],
        )
        self.assertEqual(
            payload["outcome"]["terminal_reason_counts"]["unsupported"],
            2,
        )
        formal_calls = [row for row in calls if row[1] >= 0]
        self.assertEqual(
            formal_calls,
            [(arm, 0) for arm in arms] + [(arm, 1) for arm in reversed(arms)],
        )

    def test_historical_capacity_artifact_schema_is_unchanged(self):
        args = SimpleNamespace(
            model="Qwen/Qwen3-0.6B",
            model_revision="revision",
            source_git_sha="a" * 40,
            image_digest="sha256:" + "1" * 64,
            log_dir=Path("/results/historical"),
            port=30011,
            mem_fraction_static=0.65,
            chunked_prefill_size=1024,
            chunk_source="provisional_worst_case",
            rhos="1.5",
            profiles="exact_only",
            formal_repeats=2,
            server_start_timeout_s=1.0,
            kv_bytes_per_token=114688,
            capacity_tolerance=0.05,
        )
        server = SimpleNamespace(command=["python3"], plugin_env={})
        calls: list = []
        with (
            patch.object(
                run_p6_4_capacity_pilot,
                "source_provenance",
                return_value={
                    "source_git_sha": "a" * 40,
                    "source_tree_sha": "b" * 40,
                },
            ),
            patch.object(run_p6_4_capacity_pilot, "launch_server", return_value=server),
            patch.object(run_p6_4_capacity_pilot, "wait_ready"),
            patch.object(run_p6_4_capacity_pilot, "stop_server"),
            patch.object(
                run_p6_4_capacity_pilot,
                "metric_snapshot",
                return_value={"sglang:max_total_num_tokens": 40000},
            ),
            patch.object(run_p6_4_capacity_pilot, "metric_text", return_value=""),
            patch.object(run_p6_4_capacity_pilot, "machine_manifest", return_value={}),
            patch.object(
                run_p6_4_capacity_pilot,
                "run_round",
                side_effect=recording_run_round(calls),
            ),
        ):
            payload = run_p6_4_capacity_pilot.execute(args, "p6-4-test")

        self.assertEqual(payload["phase"], "P6-4")
        self.assertEqual(payload["warmup_repeats"], 1)
        self.assertNotIn("phase7_mode", payload)
        self.assertNotIn("phase7_parameters", payload)
        cell = payload["cells"][0]
        for field in (
            "execution_order",
            "formal_repeats",
            "capacity_relative_error_tolerance",
            "inactive_counter_observations",
        ):
            self.assertNotIn(field, cell)
        profile = cell["profiles"][0]
        self.assertIsInstance(profile["warmup"], dict)
        self.assertNotIn("warmup_repeats", profile)
        self.assertEqual(len(payload["cells"]), 3)
        self.assertEqual(
            calls,
            [("exact_only", -1), ("exact_only", 0), ("exact_only", 1)] * 3,
        )

    def test_historical_diagnostic_profile_keeps_hyphenated_status(self):
        args = SimpleNamespace(
            model="Qwen/Qwen3-0.6B",
            model_revision="revision",
            source_git_sha="a" * 40,
            image_digest="sha256:" + "1" * 64,
            log_dir=Path("/results/historical"),
            port=30011,
            mem_fraction_static=0.65,
            chunked_prefill_size=1024,
            chunk_source="provisional_worst_case",
            rhos="1.5",
            profiles="exact_only",
            formal_repeats=2,
            server_start_timeout_s=1.0,
            kv_bytes_per_token=114688,
            capacity_tolerance=10.0,
        )
        server = SimpleNamespace(command=["python3"], plugin_env={})
        calls: list = []
        with (
            patch.object(
                run_p6_4_capacity_pilot,
                "source_provenance",
                return_value={
                    "source_git_sha": "a" * 40,
                    "source_tree_sha": "b" * 40,
                },
            ),
            patch.object(run_p6_4_capacity_pilot, "launch_server", return_value=server),
            patch.object(run_p6_4_capacity_pilot, "wait_ready"),
            patch.object(run_p6_4_capacity_pilot, "stop_server"),
            patch.object(
                run_p6_4_capacity_pilot,
                "metric_snapshot",
                return_value={"sglang:max_total_num_tokens": 40000},
            ),
            patch.object(run_p6_4_capacity_pilot, "metric_text", return_value=""),
            patch.object(run_p6_4_capacity_pilot, "machine_manifest", return_value={}),
            patch.object(
                run_p6_4_capacity_pilot,
                "run_round",
                side_effect=diagnostic_recording_run_round(
                    calls,
                    "diagnostic-unavailable",
                ),
            ),
        ):
            payload = run_p6_4_capacity_pilot.execute(args, "p6-4-diagnostic-test")

        self.assertEqual(payload["status"], "inconclusive")
        self.assertEqual(
            {cell["status"] for cell in payload["cells"]},
            {"diagnostic-unavailable"},
        )
        self.assertNotIn("diagnostic_unavailable", str(payload["cells"]))

    def test_phase7_diagnostic_profile_keeps_underscored_status(self):
        setting = capacity_setting()
        context = phase7_capacity_context(setting)
        args = phase7_capacity_args(context)
        server = SimpleNamespace(
            command=["python3", "-m", "sglang.launch_server"],
            plugin_env={},
        )
        calls: list = []
        with (
            patch.object(run_p6_4_capacity_pilot, "launch_server", return_value=server),
            patch.object(run_p6_4_capacity_pilot, "wait_ready"),
            patch.object(run_p6_4_capacity_pilot, "stop_server"),
            patch.object(
                run_p6_4_capacity_pilot,
                "metric_snapshot",
                return_value={"sglang:max_total_num_tokens": 11392},
            ),
            patch.object(run_p6_4_capacity_pilot, "metric_text", return_value=""),
            patch.object(run_p6_4_capacity_pilot, "machine_manifest", return_value={}),
            patch.object(
                run_p6_4_capacity_pilot,
                "run_round",
                side_effect=diagnostic_recording_run_round(
                    calls,
                    "diagnostic_unavailable",
                ),
            ),
        ):
            payload = run_p6_4_capacity_pilot.execute(
                args,
                "p7-capacity-diagnostic-test",
            )

        self.assertEqual(payload["status"], "inconclusive")
        self.assertEqual(payload["cells"][0]["status"], "diagnostic_unavailable")
        self.assertNotIn("diagnostic-unavailable", str(payload["cells"]))

    def test_labeled_metric_delta_preserves_provenance(self):
        before = (
            "sglang:cross_store_evicted_bytes_total"
            '{requester="approximate",provenance="exact",'
            'object_kind="filler"} 10\n'
        )
        after = (
            "sglang:cross_store_evicted_bytes_total"
            '{requester="approximate",provenance="exact",'
            'object_kind="filler"} 30\n'
            "sglang:cross_store_evicted_bytes_total"
            '{requester="exact",provenance="approximate",'
            'object_kind="delta"} 40\n'
        )
        self.assertEqual(
            labeled_metric_delta(
                before,
                after,
                "sglang:cross_store_evicted_bytes_total",
                {"provenance": "exact"},
            ),
            20,
        )
        self.assertEqual(
            labeled_metric_delta(
                before,
                after,
                "sglang:cross_store_evicted_bytes_total",
                {"requester": "exact", "provenance": "approximate"},
            ),
            40,
        )


if __name__ == "__main__":
    unittest.main()
