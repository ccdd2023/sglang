#!/usr/bin/env python3
"""Run V40 on an outcome-independent short, source-rich canary."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import (
    run_v40a_grounded_observation_canary as base,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v40a3_short_grounded_canary_20260728"
)
MOTIVATION = (
    ARTIFACTS
    / "impactkv_v40_grounded_observation_motivation_20260728"
    / "V40_MOTIVATION_RESULT.json"
)
V40A2_FAILURE = (
    ARTIFACTS
    / "impactkv_v40a2_grounded_observation_canary_20260728"
    / "V40A2_INFRA_FAILURE.json"
)
INSTANCE_ID = "pytest-dev__pytest-7982"
V40 = base.V40
ARMS = base.ARMS


def _selected_task() -> tuple[str, int, int]:
    rows = read_json(MOTIVATION)["cohorts"]["full18"]
    short = [row for row in rows if int(row["calls"]) <= 15]
    selected = sorted(
        short,
        key=lambda row: (
            -int(row["requests_with_source"]),
            str(row["instance_id"]),
        ),
    )[0]
    return (
        str(selected["instance_id"]),
        int(selected["calls"]),
        int(selected["requests_with_source"]),
    )


def register(output: Path) -> dict[str, Any]:
    path = output / "V40A3_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    selected = _selected_task()
    if selected != (INSTANCE_ID, 13, 6):
        raise AssertionError("V40A3 outcome-independent selection changed")
    failure = read_json(V40A2_FAILURE)
    if failure["status"] != "V40A2_INFRA_FAILURE_NO_ACCURACY_RESULT":
        raise AssertionError("V40A2 failure audit changed")
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_V40A3_TREATMENT",
        "experiment": "V40 short source-rich paired mechanism canary",
        "motivation": (
            "V40A2 verified 13 V40 and 12 General copies but its final "
            "General request timed out on a 20-call source-rich trace. Avoid "
            "posthoc timeout extension: select the full18 task with the most "
            "candidate-source requests among traces of at most 15 calls, "
            "without reading task outcomes."
        ),
        "selection": {
            "instance_id": INSTANCE_ID,
            "rule": (
                "Filter frozen V40 full18 motivation rows to calls <= 15; "
                "maximize requests_with_source; lexicographic instance_id "
                "tie-break."
            ),
            "calls": selected[1],
            "requests_with_source": selected[2],
            "official_outcomes_used": False,
            "classification": (
                "EXPOSED_MECHANISM_CANARY_NOT_GENERALIZATION"
            ),
            "replacement_for_prior_canaries": False,
        },
        "protocol": {
            "arms": list(ARMS),
            "shared_dense_history_before_first_source_plan_difference": True,
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
            "v40a2_failure": str(V40A2_FAILURE),
            "v40a2_failure_sha256": sha256(V40A2_FAILURE),
            "runner_sha256": sha256(base.RUNNER),
            "base_orchestrator_sha256": sha256(Path(base.__file__)),
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
    task = output / "task"
    runtime = read_json(task / "V25_RESULT.json")
    official = read_json(task / "V25_OFFICIAL_RESULT.json")
    clients = {
        arm: base._client(task / arm / "CLIENT_LEDGER.jsonl")
        for arm in ARMS
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
            "PASS_V40A3_MECHANISM_CANARY"
            if all(gates.values())
            else "FAIL_V40A3_MECHANISM_CANARY"
        ),
        "registration_sha256": sha256(
            output / "V40A3_REGISTRATION.json"
        ),
        "runtime_status": runtime["status"],
        "branch": runtime["branch"],
        "copy_requests": copy_requests,
        "copied_tokens": copied_tokens,
        "candidate_grounded_decisions": len(candidate_decisions),
        "resolved": resolved,
        "gate_outcomes": gates,
        "interpretation": (
            "A pass permits a preregistered multi-task development sample. "
            "This exposed canary is not population or SOTA evidence."
        ),
    }
    write_json(output / "V40A3_RESULT.json", value)
    return value


def run(output: Path) -> dict[str, Any]:
    register(output)
    base.INSTANCE_ID = INSTANCE_ID
    base._run_stage(output, "register")
    base._run_stage(output, "run")
    base._run_stage(output, "evaluate")
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
