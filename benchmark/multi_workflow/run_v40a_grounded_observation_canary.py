#!/usr/bin/env python3
"""Run the preregistered V40 grounded-observation mechanism canary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v40a_grounded_observation_canary_20260728"
)
PROJECT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT / "benchmark/multi_workflow/run_v25_paired_agent_canary.py"
PYTHON = Path("/home/gfy/.venvs/mini-swe-agent-v2.3.0/bin/python")
MOTIVATION = (
    ARTIFACTS
    / "impactkv_v40_grounded_observation_motivation_20260728"
    / "V40_MOTIVATION_RESULT.json"
)
AUDIT = (
    ARTIFACTS
    / "impactkv_v39_v38_equivalence_audit_20260728"
    / "V39_V38_EQUIVALENCE_AUDIT.json"
)
INSTANCE_ID = "astropy__astropy-7336"
V40 = "coding_grounded_observation_island_v40"
ARMS = (V40, "general", "dense")


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "all_proxy",
        "https_proxy",
        "http_proxy",
    ):
        env.pop(key, None)
    env.update(
        {
            "NO_PROXY": "*",
            "no_proxy": "*",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": (
                f"{PROJECT}:{PROJECT / 'python'}:"
                "/home/gfy/.venvs/mini-swe-agent-v2.3.0/"
                "lib/python3.12/site-packages"
            ),
            "IMPACTKV_PAIRED_CANDIDATE_ARM": V40,
            "IMPACTKV_PAIRED_DENSE_CONTROL": "1",
            "IMPACTKV_ALLOW_EMPTY_SUBMISSION_OUTCOME": "1",
            "IMPACTKV_REQUIRE_BRANCH": "1",
            "IMPACTKV_PAIRED_INSTANCE_ID": INSTANCE_ID,
            "IMPACTKV_REQUEST_TIMEOUT_SECONDS": "180",
            "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "1",
        }
    )
    return env


def register(output: Path) -> dict[str, Any]:
    path = output / "V40A_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    motivation = read_json(MOTIVATION)
    if motivation["status"] != "PASS_V40_MOTIVATION":
        raise AssertionError("V40 motivation did not pass")
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_V40A_TREATMENT",
        "experiment": "V40 grounded-observation paired mechanism canary",
        "motivation": (
            "V40 passed its outcome-independent capacity gates. Run the "
            "previously exposed Astropy-7336 damage task only as a mechanism "
            "and preservation canary: branch from one shared Dense prefix, "
            "compare a tool-observation island with General's whole retained "
            "history, and include a zero-copy Dense arm."
        ),
        "selection": {
            "instance_id": INSTANCE_ID,
            "classification": (
                "TUNED_EXPOSED_POSITIVE_CONTROL_NOT_GENERALIZATION"
            ),
            "reason": (
                "General and V35B both damaged this Dense-pass task in V36; "
                "later reruns showed outcome variability, so this task cannot "
                "establish superiority."
            ),
            "replacement_on_failure": False,
        },
        "protocol": {
            "arms": list(ARMS),
            "shared_dense_history_before_first_source-plan_difference": True,
            "container_snapshot_before_branch": True,
            "official_swebench_container_each_arm": True,
            "step_limit": 20,
            "temperature": 0,
            "request_timeout_seconds": 180,
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "frozen_gates": {
            "runtime_status": "PASS",
            "branch_reached": True,
            "branch_kind": "future_source_plan",
            "candidate_copy_requests_min": 1,
            "general_copy_requests_min": 1,
            "candidate_copied_tokens_strictly_below_general": True,
            "candidate_assistant_tokens_selected": 0,
            "target_fallbacks": 0,
            "official_arms_completed": 3,
            "candidate_resolved_not_below_general": True,
            "candidate_resolved_not_below_dense": True,
        },
        "inputs": {
            "motivation": str(MOTIVATION),
            "motivation_sha256": sha256(MOTIVATION),
            "v39_audit": str(AUDIT),
            "v39_audit_sha256": sha256(AUDIT),
            "runner_sha256": sha256(RUNNER),
            "script_sha256": sha256(Path(__file__)),
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
    }
    write_json(path, value)
    return value


def _run_stage(output: Path, stage: str) -> dict[str, Any]:
    log_path = output / "orchestration_logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        str(RUNNER),
        stage,
        "--output",
        str(output / "task"),
    ]
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=PROJECT,
            env=_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    value = {
        "stage": stage,
        "returncode": process.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "log_path": str(log_path),
    }
    write_json(output / "orchestration_status" / f"{stage}.json", value)
    if process.returncode:
        raise RuntimeError(f"V40A {stage} failed; see {log_path}")
    return value


def _client(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    task = output / "task"
    runtime = read_json(task / "V25_RESULT.json")
    official = read_json(task / "V25_OFFICIAL_RESULT.json")
    clients = {
        arm: _client(task / arm / "CLIENT_LEDGER.jsonl") for arm in ARMS
    }
    copy_requests = {
        arm: sum(
            int(row["copied_tokens_planned"]) > 0 for row in clients[arm]
        )
        for arm in ARMS
    }
    copied_tokens = {
        arm: sum(
            int(row["copied_tokens_planned"]) for row in clients[arm]
        )
        for arm in ARMS
    }
    candidate_decisions = [
        row.get("reuse_policy_decision", {})
        for row in clients[V40]
        if row.get("reuse_policy_decision", {}).get("mode")
        == "grounded_version_valid_observation_island"
    ]
    resolved = {
        arm: int(official["arms"][arm]["resolved"]) for arm in ARMS
    }
    frozen = registration["frozen_gates"]
    gates = {
        "runtime_status": runtime["status"] == frozen["runtime_status"],
        "branch_reached": (
            (runtime["branch"] is not None)
            == frozen["branch_reached"]
        ),
        "branch_kind": (
            runtime["branch"]["kind"] == frozen["branch_kind"]
            if runtime["branch"] is not None
            else False
        ),
        "candidate_copy_requests_min": (
            copy_requests[V40] >= frozen["candidate_copy_requests_min"]
        ),
        "general_copy_requests_min": (
            copy_requests["general"]
            >= frozen["general_copy_requests_min"]
        ),
        "candidate_copied_tokens_strictly_below_general": (
            copied_tokens[V40] < copied_tokens["general"]
        ),
        "candidate_assistant_tokens_selected": (
            bool(candidate_decisions)
            and all(
                int(decision["assistant_tokens_selected"])
                == frozen["candidate_assistant_tokens_selected"]
                for decision in candidate_decisions
            )
        ),
        "target_fallbacks": (
            int(runtime["server"]["target_fallbacks"])
            == frozen["target_fallbacks"]
        ),
        "official_arms_completed": (
            len(official["arms"]) == frozen["official_arms_completed"]
        ),
        "candidate_resolved_not_below_general": (
            resolved[V40] >= resolved["general"]
        ),
        "candidate_resolved_not_below_dense": (
            resolved[V40] >= resolved["dense"]
        ),
    }
    value = {
        "completed_at_utc": utc_now(),
        "status": (
            "PASS_V40A_MECHANISM_CANARY"
            if all(gates.values())
            else "FAIL_V40A_MECHANISM_CANARY"
        ),
        "registration_sha256": sha256(
            output / "V40A_REGISTRATION.json"
        ),
        "runtime_status": runtime["status"],
        "branch": runtime["branch"],
        "copy_requests": copy_requests,
        "copied_tokens": copied_tokens,
        "candidate_grounded_decisions": len(candidate_decisions),
        "resolved": resolved,
        "gate_outcomes": gates,
        "interpretation": (
            "A pass permits a new frozen multi-task development sample; it "
            "does not establish population accuracy or SOTA superiority."
        ),
    }
    write_json(output / "V40A_RESULT.json", value)
    return value


def run(output: Path) -> dict[str, Any]:
    register(output)
    _run_stage(output, "register")
    _run_stage(output, "run")
    _run_stage(output, "evaluate")
    return summarize(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("register", "run", "summarize"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = (
        register(args.output)
        if args.command == "register"
        else summarize(args.output)
        if args.command == "summarize"
        else run(args.output)
    )
    print(
        {
            "status": value["status"],
            "resolved": value.get("resolved"),
            "gate_outcomes": value.get("gate_outcomes"),
        }
    )


if __name__ == "__main__":
    main()
