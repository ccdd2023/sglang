#!/usr/bin/env python3
"""Audited completion of V27C after the frozen Pylint timeout."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_bridge_reuse_agent_experiment import DATASET
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
PROJECT = Path(__file__).resolve().parents[2]
PRIOR = ARTIFACTS / "impactkv_v27c_dense_pass_triple_20260727"
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v27d_dense_pass_audited_completion_20260727"
)
RUNNER = PROJECT / "benchmark/multi_workflow/run_v25_paired_agent_canary.py"
PYTHON = Path("/home/gfy/.venvs/mini-swe-agent-v2.3.0/bin/python")
V23 = "coding_post_mutation_target_prefix_v23"
GENERAL = "general"
DENSE = "dense"
ARMS = (V23, GENERAL, DENSE)
REUSE_ARMS = (V23, GENERAL)
PRIOR_COMPLETE = (
    "astropy__astropy-7336",
    "django__django-14855",
    "django__django-16899",
)
TIMEOUT_TASK = "pylint-dev__pylint-7277"
REMAINING = (
    "pytest-dev__pytest-7982",
    "sympy__sympy-24539",
)
CACHEBLEND_DAMAGE_RATE = 9 / 167


def task_dir(output: Path, instance_id: str) -> Path:
    return output / "tasks" / instance_id


def _command(output: Path, instance_id: str, stage: str) -> list[str]:
    return [
        str(PYTHON),
        str(RUNNER),
        stage,
        "--output",
        str(task_dir(output, instance_id)),
    ]


def _environment(instance_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "ALL_PROXY": "",
            "HTTPS_PROXY": "",
            "HTTP_PROXY": "",
            "all_proxy": "",
            "https_proxy": "",
            "http_proxy": "",
            "NO_PROXY": "localhost,127.0.0.1",
            "no_proxy": "localhost,127.0.0.1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": f"{PROJECT / 'python'}:{PROJECT}",
            "IMPACTKV_PAIRED_INSTANCE_ID": instance_id,
            "IMPACTKV_REQUEST_TIMEOUT_SECONDS": "180",
            "IMPACTKV_PAIRED_DENSE_CONTROL": "1",
            "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "1",
        }
    )
    return env


def _prior_hashes(instance_id: str) -> dict[str, str]:
    child = PRIOR / "tasks" / instance_id
    paths = (
        child / "V25_REGISTRATION.json",
        child / "V25_RESULT.json",
        child / "V25_OFFICIAL_RESULT.json",
    )
    if any(not path.exists() for path in paths):
        raise FileNotFoundError(f"incomplete frozen prior task: {instance_id}")
    return {path.name: sha256(path) for path in paths}


def register(output: Path) -> dict[str, Any]:
    path = output / "V27D_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    timeout_root = PRIOR / "tasks" / TIMEOUT_TASK
    timeout_paths = {
        "child_registration": timeout_root / "V25_REGISTRATION.json",
        "server_ledger": timeout_root / "run/SERVER_LEDGER.jsonl",
        "server_log": timeout_root / "run/sglang_server.log",
        "orchestration_log": (
            PRIOR
            / "orchestration_logs"
            / TIMEOUT_TASK
            / "V27C_run.log"
        ),
    }
    if any(not item.exists() for item in timeout_paths.values()):
        raise FileNotFoundError("V27C timeout evidence is incomplete")
    dataset = {
        json.loads(line)["instance_id"]
        for line in (DATASET / "test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    if any(instance_id not in dataset for instance_id in REMAINING):
        raise ValueError("remaining task absent from frozen dataset")
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_ANY_V27D_GPU_RUN",
        "experiment": "V27D audited completion of the V27C frozen task order",
        "motivation": (
            "V27C completed three official triple-control tasks. Its fourth "
            "task, Pylint-7277, returned HTTP 200 only after the frozen "
            "180-second client deadline on three identical attempts and "
            "never reached a target-copy event. Stop repeated non-evidence, "
            "freeze that task as an infrastructure timeout, and run only the "
            "two untouched tasks remaining in the original order."
        ),
        "frozen_prior_results": {
            instance_id: _prior_hashes(instance_id)
            for instance_id in PRIOR_COMPLETE
        },
        "declared_timeout": {
            "instance_id": TIMEOUT_TASK,
            "classification": (
                "INCOMPLETE_INFRASTRUCTURE_TIMEOUT; excluded from accuracy "
                "and damage denominators, counted against completion."
            ),
            "attempts_observed": 3,
            "request_timeout_seconds": 180,
            "evidence_sha256": {
                label: sha256(item)
                for label, item in timeout_paths.items()
            },
            "do_not_retry": True,
        },
        "remaining_selection": {
            "rule": (
                "Continue the immutable V27C all-Dense-pass order after the "
                "declared timeout; run Pytest-7982 then SymPy-24539. No arm "
                "outcome is used for this choice."
            ),
            "instance_ids": list(REMAINING),
            "reuse_outcomes_used": False,
        },
        "protocol": {
            "arms": list(ARMS),
            "same_runner_and_triple_control": True,
            "request_timeout_seconds": 180,
            "model_retry_attempts": 1,
            "official_swebench_each_arm": True,
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "frozen_gates": {
            "remaining_official_tasks_completed": len(REMAINING),
            "remaining_runtime_failures": 0,
            "remaining_target_fallbacks": 0,
            "combined_v23_resolved_not_below_general": True,
            "combined_v23_resolved_not_below_dense": True,
            "combined_v23_damage_not_above_general": True,
            "combined_v23_damage_rate_below_cacheblend": (
                CACHEBLEND_DAMAGE_RATE
            ),
            "campaign_incomplete_due_declared_timeout": True,
            "promotion_forbidden": True,
        },
        "commands": {
            instance_id: {
                stage: _command(output, instance_id, stage)
                for stage in ("register", "run", "evaluate")
            }
            for instance_id in REMAINING
        },
        "inputs": {
            "prior_v27c_registration_sha256": sha256(
                PRIOR / "V27C_REGISTRATION.json"
            ),
            "dataset_sha256": sha256(DATASET / "test.jsonl"),
            "runner_sha256": sha256(RUNNER),
            "completion_script_sha256": sha256(Path(__file__)),
        },
        "protected": {
            "prefetch": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "old_dirty_checkout_modified": False,
        },
    }
    write_json(path, value)
    return value


def _run_stage(
    output: Path, instance_id: str, stage: str
) -> dict[str, Any]:
    log_path = (
        output / "orchestration_logs" / instance_id / f"V27D_{stage}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            _command(output, instance_id, stage),
            cwd=PROJECT,
            env=_environment(instance_id),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    value = {
        "instance_id": instance_id,
        "stage": stage,
        "returncode": result.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "log_path": str(log_path),
    }
    write_json(
        output
        / "orchestration_status"
        / instance_id
        / f"V27D_{stage}_STATUS.json",
        value,
    )
    return value


def preregister_tasks(output: Path) -> list[dict[str, Any]]:
    register(output)
    rows = []
    for instance_id in REMAINING:
        child = task_dir(output, instance_id) / "V25_REGISTRATION.json"
        rows.append(
            {
                "instance_id": instance_id,
                "stage": "register",
                "returncode": 0,
                "resumed": True,
            }
            if child.exists()
            else _run_stage(output, instance_id, "register")
        )
    write_json(output / "V27D_CHILD_REGISTRATIONS.json", rows)
    if any(row["returncode"] != 0 for row in rows):
        raise RuntimeError("V27D child preregistration failed")
    return rows


def _run_remaining(output: Path) -> list[dict[str, Any]]:
    preregister_tasks(output)
    stages = []
    for instance_id in REMAINING:
        child = task_dir(output, instance_id)
        if not (child / "V25_RESULT.json").exists():
            stages.append(_run_stage(output, instance_id, "run"))
        if not (child / "V25_RESULT.json").exists():
            continue
        if not (child / "V25_OFFICIAL_RESULT.json").exists():
            stages.append(_run_stage(output, instance_id, "evaluate"))
    write_json(output / "V27D_STAGE_STATUS.json", stages)
    return stages


def _row(root: Path, instance_id: str) -> dict[str, Any]:
    runtime = read_json(root / "tasks" / instance_id / "V25_RESULT.json")
    official = read_json(
        root / "tasks" / instance_id / "V25_OFFICIAL_RESULT.json"
    )
    resolved = {
        arm: int(official["arms"][arm]["resolved"]) for arm in ARMS
    }
    return {
        "instance_id": instance_id,
        "source_campaign": str(root),
        "runtime_status": runtime["status"],
        "treated": runtime["branch"] is not None,
        "resolved": resolved,
        "v23_damage": resolved[DENSE] == 1 and resolved[V23] == 0,
        "general_damage": resolved[DENSE] == 1
        and resolved[GENERAL] == 0,
        "target_fallbacks": runtime["server"]["target_fallbacks"],
        "branch": runtime["branch"],
        "copy_counts": runtime["server"]["copy_counts"],
        "dense_control_requests": runtime["server"].get(
            "dense_control_requests", 0
        ),
        "branched_agent_elapsed_seconds": runtime.get(
            "branched_agent_elapsed_seconds"
        ),
    }


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    rows = []
    for instance_id in PRIOR_COMPLETE:
        rows.append(_row(PRIOR, instance_id))
    remaining_complete = []
    for instance_id in REMAINING:
        child = task_dir(output, instance_id)
        if (child / "V25_RESULT.json").exists() and (
            child / "V25_OFFICIAL_RESULT.json"
        ).exists():
            rows.append(_row(output, instance_id))
            remaining_complete.append(instance_id)
    resolved = {
        arm: sum(row["resolved"][arm] for row in rows) for arm in ARMS
    }
    dense_passes = resolved[DENSE]
    damages = {
        V23: sum(row["v23_damage"] for row in rows),
        GENERAL: sum(row["general_damage"] for row in rows),
    }
    damage_rates = {
        arm: damages[arm] / dense_passes if dense_passes else None
        for arm in REUSE_ARMS
    }
    remaining_rows = [
        row for row in rows if row["instance_id"] in REMAINING
    ]
    remaining_runtime_failures = sum(
        row["runtime_status"] != "PASS" for row in remaining_rows
    )
    remaining_fallbacks = sum(
        row["target_fallbacks"] for row in remaining_rows
    )
    gates = {
        "remaining_official_tasks_completed": (
            len(remaining_complete) == len(REMAINING)
        ),
        "remaining_runtime_failures": remaining_runtime_failures == 0,
        "remaining_target_fallbacks": remaining_fallbacks == 0,
        "combined_v23_resolved_not_below_general": (
            resolved[V23] >= resolved[GENERAL]
        ),
        "combined_v23_resolved_not_below_dense": (
            resolved[V23] >= resolved[DENSE]
        ),
        "combined_v23_damage_not_above_general": (
            damages[V23] <= damages[GENERAL]
        ),
        "combined_v23_damage_rate_below_cacheblend": (
            damage_rates[V23] is not None
            and damage_rates[V23] < CACHEBLEND_DAMAGE_RATE
        ),
        "campaign_incomplete_due_declared_timeout": True,
        "promotion_forbidden": True,
    }
    value = {
        "summarized_at_utc": utc_now(),
        "status": (
            "AUDITED_COMPLETION_WITH_DECLARED_TIMEOUT"
            if len(remaining_complete) == len(REMAINING)
            else "INCOMPLETE"
        ),
        "valid_official_tasks": rows,
        "excluded_task": registration["declared_timeout"],
        "aggregate": {
            "valid_official_tasks": len(rows),
            "selected_tasks": 6,
            "declared_infrastructure_timeouts": 1,
            "treated_valid_tasks": sum(row["treated"] for row in rows),
            "runtime_passed_valid_tasks": sum(
                row["runtime_status"] == "PASS" for row in rows
            ),
            "resolved": resolved,
            "accuracy_over_valid_tasks": {
                arm: resolved[arm] / len(rows) if rows else None
                for arm in ARMS
            },
            "concurrent_dense_passes": dense_passes,
            "damage_count_given_concurrent_dense_pass": damages,
            "damage_rate_given_concurrent_dense_pass": damage_rates,
            "cacheblend_damage_rate_reference": CACHEBLEND_DAMAGE_RATE,
            "mean_branched_agent_elapsed_seconds": {
                arm: statistics.fmean(
                    row["branched_agent_elapsed_seconds"][arm]
                    for row in rows
                    if row["treated"]
                )
                for arm in ARMS
            },
        },
        "gate_outcomes": gates,
        "decision": (
            "Never promote V27C/V27D: one selected task is an infrastructure "
            "timeout and a prior valid task failed its child submission gate. "
            "Use the five valid task outcomes only as selector-development "
            "evidence."
        ),
    }
    write_json(output / "V27D_RESULT.json", value)
    return value


def run(output: Path) -> dict[str, Any]:
    _run_remaining(output)
    return summarize(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("register", "preregister", "run", "summarize"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "register":
        value = register(args.output)
    elif args.command == "preregister":
        value = {"children": preregister_tasks(args.output)}
    elif args.command == "run":
        value = run(args.output)
    else:
        value = summarize(args.output)
    print(
        {
            "status": value.get("status"),
            "output": str(args.output),
            "gate_outcomes": value.get("gate_outcomes"),
        }
    )


if __name__ == "__main__":
    main()
