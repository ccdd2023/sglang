#!/usr/bin/env python3
"""Repeat the frozen Fresh8 protocol with limit-time patch capture enabled.

The original outcomes are already open, so this is strictly a protocol
recovery audit.  It reuses the exact eight tasks and compares rolling Dense
with the unchanged dependency-cold policy; it must not be cited as an
independent accuracy experiment for the new graph selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import (
    run_natural_code_cost_expanded_accuracy_campaign as paired,
)


PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path("/home/gfy/CodeMAS_Project")
SOURCE = ROOT / "kvflow-artifacts/impactkv_dependency_cold_fresh8_20260810"
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_dependency_cold_fresh8_patch_capture_20260811"
)
BRIDGE_RUNNER = (
    PROJECT / "benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py"
)
MINI_PYTHON = Path("/home/gfy/.venvs/mini-swe-agent-v2.3.0/bin/python")
POLICY_ARM = "coding_dependency_cold_cost"
ARMS = ("dense", POLICY_ARM)
TASKS = 8


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output: Path) -> dict[str, Any]:
    path = output / "RECOVERY_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    source_registration = read_json(SOURCE / "CAMPAIGN_REGISTRATION.json")
    identifiers = [
        str(row["instance_id"])
        for row in source_registration["selection"]["instances"]
    ]
    value = {
        "status": "REGISTERED_AFTER_ORIGINAL_OUTCOMES_PROTOCOL_AUDIT_ONLY",
        "registered_at_utc": utc_now(),
        "classification": "Fresh8 limit-time patch-capture recovery audit",
        "instances": identifiers,
        "protocol": {
            "arms": list(ARMS),
            "model": "Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
            "rolling_history_groups": 6,
            "step_limit": 32,
            "capture_git_diff_on_limits_exceeded": True,
            "capture_git_diff_on_empty_submitted": True,
            "official_metric": "SWE-bench resolved",
            "prefetch": False,
        },
        "inputs": {
            "source_campaign": str(SOURCE),
            "source_registration_sha256": sha256(
                SOURCE / "CAMPAIGN_REGISTRATION.json"
            ),
            "dataset_sha256": sha256(SOURCE / "dataset/test.jsonl"),
            "snapshot_sha256": sha256(SOURCE / "FROZEN_FRESH8.json"),
            "bridge_registration_sha256": sha256(
                SOURCE / "BRIDGE_FRESH8_REGISTRATION.json"
            ),
        },
        "claim_limit": (
            "The original task outcomes are open. This repeat diagnoses the "
            "submission protocol and is not independent algorithm evidence."
        ),
        "protected": {
            "paper_modified": False,
            "prefetch": False,
            "old_preregistration_thresholds_modified": False,
        },
    }
    write_json(path, value)
    return value


def run_arm(output: Path, arm: str, port: int) -> None:
    if arm not in ARMS:
        raise ValueError(arm)
    prepare(output)
    env = os.environ.copy()
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(key, None)
    env.update(
        {
            "IMPACTKV_DATASET_ROOT": str(SOURCE / "dataset"),
            "IMPACTKV_EVAL_SNAPSHOT": str(SOURCE / "FROZEN_FRESH8.json"),
            "IMPACTKV_EVAL_REGISTRATION": str(
                SOURCE / "BRIDGE_FRESH8_REGISTRATION.json"
            ),
            "IMPACTKV_AGENT_STEP_LIMIT": "32",
            "IMPACTKV_CAPTURE_LIMIT_PATCH": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "PYTHONPATH": str(PROJECT),
        }
    )
    command = [
        str(MINI_PYTHON),
        str(BRIDGE_RUNNER),
        "--output",
        str(output / "online"),
        "run-arm",
        "--arm",
        arm,
        "--scope",
        "full",
        "--port",
        str(port),
        "--official",
    ]
    subprocess.run(command, cwd=PROJECT, env=env, check=True)


def trajectory_summary(root: Path, instance_id: str) -> dict[str, Any]:
    paths = list((root / instance_id).glob("*.traj.json"))
    if len(paths) != 1:
        raise ValueError(f"expected one trajectory for {instance_id}: {paths}")
    trajectory = read_json(paths[0])
    info = trajectory.get("info") or {}
    submission = str(info.get("submission") or trajectory.get("submission") or "")
    return {
        "exit_status": info.get("exit_status") or trajectory.get("exit_status"),
        "submission_characters": len(submission),
        "nonempty_submission": bool(submission.strip()),
    }


def summarize(output: Path) -> dict[str, Any]:
    registration = prepare(output)
    identifiers = list(registration["instances"])
    official: dict[str, dict[str, Any]] = {}
    runtime: dict[str, dict[str, Any]] = {}
    trajectories: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        root = output / f"online/{arm}/full_{TASKS}"
        official_value = read_json(root / "OFFICIAL_RESULT.json")
        report = official_value.get("report")
        if report is None:
            raise ValueError(f"official report absent for {arm}")
        official[arm] = dict(report)
        runtime[arm] = read_json(root / "RUNTIME_SUMMARY.json")
        trajectories[arm] = {
            instance_id: trajectory_summary(root, instance_id)
            for instance_id in identifiers
        }

    pair = paired._paired_summary(
        identifiers,
        set(official["dense"].get("resolved_ids") or ()),
        set(official[POLICY_ARM].get("resolved_ids") or ()),
    )
    result = {
        "status": "COMPLETE",
        "classification": "open-outcome protocol recovery audit",
        "paired_official_accuracy": pair,
        "official": official,
        "submissions": {
            arm: {
                "nonempty": sum(
                    row["nonempty_submission"]
                    for row in trajectories[arm].values()
                ),
                "tasks": trajectories[arm],
            }
            for arm in ARMS
        },
        "runtime_descriptive_only": {
            arm: {
                "requests": runtime[arm]["requests"],
                "median_ttft_ms": runtime[arm]["median_ttft_ms"],
                "target_copy_events": runtime[arm]["target_copy_events"],
                "copied_tokens": runtime[arm]["copied_tokens"],
                "target_fallback_events": runtime[arm][
                    "target_fallback_events"
                ],
            }
            for arm in ARMS
        },
        "interpretation": (
            "Use this result only to decide whether patch capture repairs the "
            "measurement protocol before running new task-disjoint evidence."
        ),
    }
    write_json(output / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run-arm")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--port", type=int, default=30000)
    sub.add_parser("summarize")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        value = prepare(output)
    elif args.command == "run-arm":
        run_arm(output, args.arm, args.port)
        value = {"arm": args.arm, "status": "COMPLETE"}
    else:
        value = summarize(output)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
