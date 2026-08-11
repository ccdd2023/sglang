#!/usr/bin/env python3
"""Recover the interrupted expanded24 campaign without overwriting evidence.

The original Dense run completed 20 trajectories before its parent process
disappeared during task 21, leaving the SGLang child orphaned.  This recovery
keeps those 20 artifacts immutable, reruns the four instances lacking a final
trajectory in a separate run directory, evaluates both Dense shards, then
runs the originally registered 24-task policy arm and combines only official
task outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from benchmark.multi_workflow import run_bridge_reuse_agent_experiment as bridge
from benchmark.multi_workflow.run_natural_code_cost_expanded_accuracy_campaign import (
    FRESH9,
    MINI_PYTHON,
    PROJECT,
    _paired_summary,
)


ROOT = Path("/home/gfy/CodeMAS_Project")
ARTIFACTS = ROOT / "kvflow-artifacts"
CAMPAIGN = (
    ARTIFACTS / "impactkv_natural_code_cost_agent_expanded24_20260808"
)
ORIGINAL_REGISTRATION = CAMPAIGN / "CAMPAIGN_REGISTRATION.json"
ORIGINAL_BRIDGE_REGISTRATION = (
    CAMPAIGN / "BRIDGE_EXPANDED24_REGISTRATION.json"
)
ORIGINAL_SNAPSHOT = CAMPAIGN / "FROZEN_EXPANDED24.json"
ORIGINAL_DENSE = CAMPAIGN / "online/dense/full_24"
RECOVERY = CAMPAIGN / "recovery_dense4"
RECOVERY_REGISTRATION = CAMPAIGN / "RECOVERY_REGISTRATION.json"
RECOVERY_BRIDGE_REGISTRATION = (
    RECOVERY / "BRIDGE_RECOVERY4_REGISTRATION.json"
)
EXPANDED_RUNNER = (
    PROJECT
    / "benchmark/multi_workflow/"
    "run_natural_code_cost_expanded_accuracy_campaign.py"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registered_ids() -> list[str]:
    value = read_json(ORIGINAL_REGISTRATION)
    return [row["instance_id"] for row in value["selection"]["instances"]]


def _completed_dense_ids() -> list[str]:
    completed = {
        path.name.removesuffix(".traj.json")
        for path in ORIGINAL_DENSE.rglob("*.traj.json")
    }
    return [value for value in _registered_ids() if value in completed]


def _remaining_dense_ids() -> list[str]:
    completed = set(_completed_dense_ids())
    return [value for value in _registered_ids() if value not in completed]


def prepare() -> dict[str, Any]:
    if RECOVERY_REGISTRATION.exists():
        return read_json(RECOVERY_REGISTRATION)
    if RECOVERY.exists() and any(RECOVERY.iterdir()):
        raise FileExistsError(RECOVERY)

    completed = _completed_dense_ids()
    remaining = _remaining_dense_ids()
    if len(completed) != 20 or len(remaining) != 4:
        raise AssertionError(
            f"expected 20 completed and 4 remaining, got {len(completed)}/{len(remaining)}"
        )
    raw_predictions = read_json(ORIGINAL_DENSE / "preds.json")
    if set(raw_predictions) != set(completed):
        raise AssertionError("original Dense predictions do not match completed trajectories")

    snapshot = read_json(ORIGINAL_SNAPSHOT)
    by_id = {row["instance_id"]: row for row in snapshot}
    remaining_rows = [by_id[value] for value in remaining]
    RECOVERY.mkdir(parents=True)
    recovery_snapshot = RECOVERY / "FROZEN_REMAINING4.json"
    recovery_dataset = RECOVERY / "dataset/test.jsonl"
    write_json(recovery_snapshot, remaining_rows)
    write_jsonl(recovery_dataset, remaining_rows)
    write_json(
        RECOVERY_BRIDGE_REGISTRATION,
        {
            "schema_version": 1,
            "registration_id": "impactkv-natural-code-cost-expanded24-dense-recovery4",
            "registered_at_utc": utc_now(),
            "dataset": {
                "name": "princeton-nlp/SWE-bench_Verified",
                "split": "test",
            },
            "instances": [{"instance_id": value} for value in remaining],
        },
    )

    trajectories = sorted(ORIGINAL_DENSE.rglob("*.traj.json"))
    value = {
        "status": "REGISTERED_BEFORE_DENSE_RECOVERY4_GPU",
        "registered_at_utc": utc_now(),
        "incident": {
            "original_status": read_json(
                ORIGINAL_DENSE / "PIPELINE_STATUS.json"
            ),
            "last_completed_trajectory_utc": "2026-08-08T03:03:45Z",
            "parent_process_disappeared_before_task_21_completed": True,
            "orphaned_sglang_process_released_at_utc": (
                "2026-08-09T05:24:20Z"
            ),
            "algorithm_error_observed": False,
            "gpu_error_observed": False,
        },
        "recovery_rule": {
            "preserve_original_completed20": True,
            "rerun_only_instances_without_final_trajectory": True,
            "combine_only_official_task_outcomes": True,
            "do_not_combine_interrupted_free_running_latency": True,
            "completed_ids": completed,
            "remaining_ids": remaining,
        },
        "protocol_unchanged": {
            "task_selection": True,
            "prompt_and_mas": True,
            "model": True,
            "temperature": True,
            "step_limit": True,
            "dense_algorithm": True,
            "policy_algorithm": True,
            "official_evaluator": True,
            "prefetch": False,
        },
        "inputs": {
            "original_registration_sha256": sha256(ORIGINAL_REGISTRATION),
            "original_bridge_registration_sha256": sha256(
                ORIGINAL_BRIDGE_REGISTRATION
            ),
            "original_snapshot_sha256": sha256(ORIGINAL_SNAPSHOT),
            "original_preds_sha256": sha256(ORIGINAL_DENSE / "preds.json"),
            "original_client_ledger_sha256": sha256(
                ORIGINAL_DENSE / "CLIENT_LEDGER.jsonl"
            ),
            "completed_trajectory_sha256": {
                str(path.relative_to(ORIGINAL_DENSE)): sha256(path)
                for path in trajectories
            },
            "recovery_snapshot_sha256": sha256(recovery_snapshot),
            "recovery_dataset_sha256": sha256(recovery_dataset),
            "recovery_bridge_registration_sha256": sha256(
                RECOVERY_BRIDGE_REGISTRATION
            ),
            "recovery_source_sha256": sha256(Path(__file__).resolve()),
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "original_artifacts_overwritten": False,
            "preregistration_thresholds_modified": False,
            "paper_modified": False,
            "prefetch": False,
        },
    }
    write_json(RECOVERY_REGISTRATION, value)
    return value


def evaluate_original_dense20() -> dict[str, Any]:
    prepare()
    if (ORIGINAL_DENSE / "OFFICIAL_RESULT.json").exists():
        return read_json(ORIGINAL_DENSE / "OFFICIAL_RESULT.json")
    return bridge.run_official_evaluation(
        output=CAMPAIGN,
        run_dir=ORIGINAL_DENSE,
        arm="dense-recovered20",
        instance_ids=_completed_dense_ids(),
        registration=ORIGINAL_BRIDGE_REGISTRATION,
        snapshot=ORIGINAL_SNAPSHOT,
    )


def run_dense_recovery4(port: int) -> None:
    prepare()
    env = os.environ.copy()
    env.update(
        {
            "IMPACTKV_DATASET_ROOT": str(RECOVERY / "dataset"),
            "IMPACTKV_EVAL_SNAPSHOT": str(
                RECOVERY / "FROZEN_REMAINING4.json"
            ),
            "IMPACTKV_EVAL_REGISTRATION": str(
                RECOVERY_BRIDGE_REGISTRATION
            ),
            "IMPACTKV_AGENT_STEP_LIMIT": "32",
            "PYTHONPATH": (
                str(PROJECT)
                + (
                    os.pathsep + env["PYTHONPATH"]
                    if env.get("PYTHONPATH")
                    else ""
                )
            ),
        }
    )
    command = [
        str(MINI_PYTHON),
        str(Path(bridge.__file__).resolve()),
        "--output",
        str(RECOVERY / "online"),
        "run-arm",
        "--arm",
        "dense",
        "--scope",
        "full",
        "--port",
        str(port),
        "--official",
    ]
    subprocess.run(command, cwd=PROJECT, env=env, check=True)


def run_policy24(port: int) -> None:
    prepare()
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(PROJECT)
        + (
            os.pathsep + env["PYTHONPATH"]
            if env.get("PYTHONPATH")
            else ""
        )
    )
    command = [
        str(MINI_PYTHON),
        str(EXPANDED_RUNNER),
        "run-arm",
        "--arm",
        "coding_natural_code_cost",
        "--port",
        str(port),
    ]
    subprocess.run(command, cwd=PROJECT, env=env, check=True)


def _official_report(path: Path) -> dict[str, Any]:
    value = read_json(path)
    report = value.get("report")
    if report is None:
        raise ValueError(f"official report missing: {path}")
    return dict(report)


def _combine_dense_reports(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    list_fields = (
        "completed_ids",
        "incomplete_ids",
        "empty_patch_ids",
        "submitted_ids",
        "resolved_ids",
        "unresolved_ids",
        "error_ids",
    )
    combined = {
        field: sorted(set(left.get(field, ())) | set(right.get(field, ())))
        for field in list_fields
    }
    combined.update(
        {
            "total_instances": 24,
            "submitted_instances": len(combined["submitted_ids"]),
            "completed_instances": len(combined["completed_ids"]),
            "resolved_instances": len(combined["resolved_ids"]),
            "unresolved_instances": len(combined["unresolved_ids"]),
            "empty_patch_instances": len(combined["empty_patch_ids"]),
            "error_instances": len(combined["error_ids"]),
            "schema_version": 2,
            "recovered_from_shards": [20, 4],
        }
    )
    if set(combined["submitted_ids"]) != set(_registered_ids()):
        raise AssertionError("combined Dense evaluator does not cover all 24 tasks")
    write_json(CAMPAIGN / "DENSE_RECOVERED_OFFICIAL_RESULT.json", combined)
    return combined


def summarize() -> dict[str, Any]:
    prepare()
    dense20 = _official_report(ORIGINAL_DENSE / "OFFICIAL_RESULT.json")
    dense4 = _official_report(
        RECOVERY / "online/dense/full_4/OFFICIAL_RESULT.json"
    )
    dense = _combine_dense_reports(dense20, dense4)
    policy_run = CAMPAIGN / "online/coding_natural_code_cost/full_24"
    policy = _official_report(policy_run / "OFFICIAL_RESULT.json")
    ids = _registered_ids()
    expanded = _paired_summary(
        ids, set(dense["resolved_ids"]), set(policy["resolved_ids"])
    )

    prior = read_json(FRESH9 / "RESULT.json")["accuracy"]
    prior_ids = [row["instance_id"] for row in prior["per_task"]]
    if set(prior_ids) & set(ids):
        raise AssertionError("fresh9 and expanded24 overlap")
    prior_dense = {
        row["instance_id"]
        for row in prior["per_task"]
        if row["dense_resolved"]
    }
    prior_policy = {
        row["instance_id"]
        for row in prior["per_task"]
        if row["policy_resolved"]
    }
    aggregate = _paired_summary(
        prior_ids + ids,
        prior_dense | set(dense["resolved_ids"]),
        prior_policy | set(policy["resolved_ids"]),
    )
    runtime = read_json(policy_run / "RUNTIME_SUMMARY.json")
    value = {
        "status": "COMPLETE",
        "classification": (
            "expanded24 official accuracy with registered Dense 20+4 execution recovery"
        ),
        "expanded24": expanded,
        "fresh33_transparent_aggregate": aggregate,
        "official_evaluator": {
            "dense_recovered20_plus4": dense,
            "coding_natural_code_cost": policy,
        },
        "physical_reuse": {
            "source_materialized_events": runtime[
                "source_materialized_events"
            ],
            "target_copy_events": runtime["target_copy_events"],
            "copied_tokens": runtime["copied_tokens"],
            "target_fallback_events": runtime[
                "target_fallback_events"
            ],
            "prefetch": False,
        },
        "speed_reporting": {
            "expanded24_free_running_dense_latency_reported": False,
            "reason": (
                "Dense execution was interrupted between tasks and resumed; "
                "task accuracy remains composable but arm wall-time/latency does not"
            ),
            "prior_exact_prompt_reference": read_json(
                FRESH9 / "exact_prompt_speed/RESULT.json"
            ),
        },
        "recovery_registration": str(RECOVERY_REGISTRATION),
        "external_baseline_ranking_allowed": False,
    }
    write_json(CAMPAIGN / "RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("evaluate-dense20")
    dense = sub.add_parser("run-dense4")
    dense.add_argument("--port", type=int, default=30000)
    policy = sub.add_parser("run-policy24")
    policy.add_argument("--port", type=int, default=30000)
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare()
    elif args.command == "evaluate-dense20":
        value = evaluate_original_dense20()
    elif args.command == "run-dense4":
        run_dense_recovery4(args.port)
        value = {"status": "COMPLETE", "arm": "dense-recovery4"}
    elif args.command == "run-policy24":
        run_policy24(args.port)
        value = {"status": "COMPLETE", "arm": "coding_natural_code_cost"}
    else:
        value = summarize()
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
