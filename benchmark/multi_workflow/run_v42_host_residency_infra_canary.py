#!/usr/bin/env python3
"""Validate the paired host-residency fix on a known capacity blocker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import (
    run_v39_v38_independent_campaign as orchestration,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v42_host_residency_canary_20260728"
V41_AUDIT = (
    ARTIFACTS
    / "impactkv_v41_v40_independent_20260728"
    / "V41_CAPACITY_DEADLOCK_AUDIT.json"
)
INSTANCE_ID = "astropy__astropy-14995"
V40 = "coding_grounded_observation_island_v40"
GENERAL = "general"
DENSE = "dense"
ARMS = (V40, GENERAL, DENSE)


def _configure() -> None:
    orchestration.V38 = V40


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def register(output: Path) -> dict[str, Any]:
    path = output / "V42_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    audit = read_json(V41_AUDIT)
    if (
        audit["status"]
        != "V41_INFRA_FAILURE_PAIRED_SOURCE_CAPACITY_DEADLOCK"
    ):
        raise AssertionError("V41 capacity audit changed")
    astropy = next(
        row
        for row in audit["failed_tasks"]
        if row["instance_id"] == INSTANCE_ID
    )
    if astropy["pending_policy"] != V40:
        raise AssertionError("known Astropy blocker changed")
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_V42_INFRA_TREATMENT",
        "experiment": "V42 paired host-residency infrastructure canary",
        "motivation": (
            "V41 Astropy timed out before its first V40 target staging event "
            "because two paired-arm device sources left fewer slots than the "
            "conservative target admission lower bound. Re-run that known "
            "capacity blocker only to test the host-residency mechanism."
        ),
        "selection": {
            "instance_id": INSTANCE_ID,
            "rule": (
                "Use the first V41 capacity-deadlocked task, whose pending "
                "target was V40. This is failure-directed infrastructure "
                "validation, not an accuracy or generalization sample."
            ),
            "official_outcomes_used": False,
            "classification": "EXPOSED_INFRA_CANARY_DO_NOT_SCORE",
            "replacement_for_v41": False,
        },
        "protocol": {
            "arms": list(ARMS),
            "shared_server_for_accuracy_mechanics_only": True,
            "prefer_host_sources": True,
            "host_snapshot_created_only_after_source_completion": True,
            "target_load_is_synchronous": True,
            "prefetch": False,
            "step_limit": 20,
            "temperature": 0,
            "request_timeout_seconds": 180,
            "official_evaluation": False,
        },
        "frozen_gates": {
            "run_returncode": 0,
            "runtime_status": "PASS",
            "branch_reached": True,
            "v40_host_sources_min": 1,
            "general_host_sources_min": 1,
            "device_sources": 0,
            "v40_target_copies_min": 1,
            "general_target_copies_min": 1,
            "target_fallbacks": 0,
            "midstream_timeout": False,
            "official_accuracy_claim": False,
        },
        "inputs": {
            "v41_audit": str(V41_AUDIT),
            "v41_audit_sha256": sha256(V41_AUDIT),
            "runner_sha256": sha256(orchestration.RUNNER),
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


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    child = orchestration.task_dir(output, INSTANCE_ID)
    stage_path = (
        output / "orchestration_status" / INSTANCE_ID / "run.json"
    )
    stage = read_json(stage_path) if stage_path.exists() else {}
    runtime_path = child / "V25_RESULT.json"
    runtime = read_json(runtime_path) if runtime_path.exists() else None
    server = _jsonl(child / "run" / "SERVER_LEDGER.jsonl")
    log_path = (
        output / "orchestration_logs" / INSTANCE_ID / "run.log"
    )
    log = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    host_sources = {
        arm: sum(
            row.get("event") == "source_materialized_host"
            and row.get("reason") == "preferred_host_residency"
            and row.get("policy_label") == arm
            for row in server
        )
        for arm in (V40, GENERAL)
    }
    device_sources = sum(
        row.get("event") == "source_materialized" for row in server
    )
    copies = {
        arm: sum(
            row.get("event") == "target_copied"
            and row.get("policy_label") == arm
            for row in server
        )
        for arm in (V40, GENERAL)
    }
    fallbacks = sum(
        row.get("event") == "target_fallback" for row in server
    )
    frozen = registration["frozen_gates"]
    gates = {
        "run_returncode": (
            stage.get("returncode") == frozen["run_returncode"]
        ),
        "runtime_status": (
            runtime is not None
            and runtime["status"] == frozen["runtime_status"]
        ),
        "branch_reached": (
            runtime is not None
            and (runtime["branch"] is not None)
            == frozen["branch_reached"]
        ),
        "v40_host_sources_min": (
            host_sources[V40] >= frozen["v40_host_sources_min"]
        ),
        "general_host_sources_min": (
            host_sources[GENERAL] >= frozen["general_host_sources_min"]
        ),
        "device_sources": device_sources == frozen["device_sources"],
        "v40_target_copies_min": (
            copies[V40] >= frozen["v40_target_copies_min"]
        ),
        "general_target_copies_min": (
            copies[GENERAL] >= frozen["general_target_copies_min"]
        ),
        "target_fallbacks": fallbacks == frozen["target_fallbacks"],
        "midstream_timeout": (
            ("httpcore.ReadTimeout: timed out" in log)
            == frozen["midstream_timeout"]
        ),
        "official_accuracy_claim": (
            False == frozen["official_accuracy_claim"]
        ),
    }
    value = {
        "completed_at_utc": utc_now(),
        "status": (
            "PASS_V42_HOST_RESIDENCY_INFRA_CANARY"
            if all(gates.values())
            else "FAIL_V42_HOST_RESIDENCY_INFRA_CANARY"
        ),
        "registration_sha256": sha256(
            output / "V42_REGISTRATION.json"
        ),
        "run_returncode": stage.get("returncode"),
        "runtime_status": runtime["status"] if runtime else None,
        "branch": runtime["branch"] if runtime else None,
        "host_sources": host_sources,
        "device_sources": device_sources,
        "target_copies": copies,
        "target_fallbacks": fallbacks,
        "midstream_timeout": "httpcore.ReadTimeout: timed out" in log,
        "official_evaluation_run": False,
        "official_accuracy_result": None,
        "gate_outcomes": gates,
        "interpretation": (
            "A pass validates only that paired host residency removes the "
            "known capacity deadlock while executing real V40 and General KV "
            "copies. It is not accuracy, speed, population, or SOTA evidence."
        ),
    }
    write_json(output / "V42_RESULT.json", value)
    return value


def run(output: Path) -> dict[str, Any]:
    _configure()
    register(output)
    child = orchestration.task_dir(output, INSTANCE_ID)
    if not (child / "V25_REGISTRATION.json").exists():
        orchestration._run_stage(output, INSTANCE_ID, "register")
    if not (child / "V25_RESULT.json").exists():
        orchestration._run_stage(output, INSTANCE_ID, "run")
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
            "gate_outcomes": value.get("gate_outcomes"),
        }
    )


if __name__ == "__main__":
    main()
