#!/usr/bin/env python3
"""Freeze why V43 cannot distinguish V40 from General or Dense."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v43_new_verified_v40_20260728"
V40 = "coding_grounded_observation_island_v40"
GENERAL = "general"
DENSE = "dense"
ARMS = (V40, GENERAL, DENSE)
ARM_DIRS = {
    V40: V40,
    GENERAL: GENERAL,
    DENSE: DENSE,
}
TASKS = (
    "sphinx-doc__sphinx-9461",
    "pydata__xarray-2905",
    "sympy__sympy-21930",
    "django__django-16263",
    "mwaskom__seaborn-3187",
    "pytest-dev__pytest-5840",
)
STEP_LIMIT = 20
SHARED_CALLS = 7
BRANCH_REQUEST_INDEX = 8
_SOURCE_WRITE = re.compile(
    r"(?:write_text\s*\(|open\s*\([^)]*,\s*['\"]w['\"]\s*\)|"
    r"apply_patch|git\s+apply|sed\s+-i|perl\s+[^;\n]*-i)"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trajectory_path(task: Path, arm: str, instance_id: str) -> Path:
    path = task / ARM_DIRS[arm] / instance_id / f"{instance_id}.traj.json"
    if not path.exists():
        raise AssertionError(f"missing trajectory: {path}")
    return path


def _command(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    if not calls:
        return ""
    arguments = calls[0]["function"]["arguments"]
    return str(json.loads(arguments).get("command", ""))


def _trajectory_audit(path: Path) -> dict[str, Any]:
    trajectory = read_json(path)
    messages = trajectory["messages"]
    assistants = [
        (index, message)
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]
    writes: list[dict[str, Any]] = []
    submissions: list[int] = []
    for request_index, (message_index, message) in enumerate(assistants, 1):
        command = _command(message)
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command:
            submissions.append(request_index)
        if not _SOURCE_WRITE.search(command):
            continue
        next_message = (
            messages[message_index + 1]
            if message_index + 1 < len(messages)
            else {}
        )
        returncode = (
            next_message.get("extra", {}).get("returncode")
            if next_message.get("role") == "tool"
            else None
        )
        writes.append(
            {
                "request_index": request_index,
                "shell_returncode": returncode,
            }
        )
    return {
        "api_calls": trajectory["info"]["model_stats"]["api_calls"],
        "exit_status": trajectory["info"]["exit_status"],
        "submission_bytes": len(trajectory["info"]["submission"].encode()),
        "source_write_commands": writes,
        "successful_source_write_command_indices": [
            row["request_index"]
            for row in writes
            if row["shell_returncode"] == 0
        ],
        "submission_command_indices": submissions,
        "trajectory_sha256": sha256(path),
    }


def _task_audit(output: Path, instance_id: str) -> dict[str, Any]:
    task = output / "tasks" / instance_id
    runtime_path = task / "V25_RESULT.json"
    official_path = task / "V25_OFFICIAL_RESULT.json"
    runtime = read_json(runtime_path)
    official = read_json(official_path)
    branch = runtime["branch"]
    if branch["branch_request_index"] != BRANCH_REQUEST_INDEX:
        raise AssertionError(f"{instance_id}: branch request changed")
    if branch["shared_calls"] != SHARED_CALLS:
        raise AssertionError(f"{instance_id}: shared-call count changed")
    if len(set(branch["source_lengths"].values())) != 2:
        raise AssertionError(f"{instance_id}: source plans no longer differ")
    arms: dict[str, Any] = {}
    for arm in ARMS:
        path = _trajectory_path(task, arm, instance_id)
        row = _trajectory_audit(path)
        if row["api_calls"] != STEP_LIMIT:
            raise AssertionError(f"{instance_id}/{arm}: call limit not reached")
        if row["exit_status"] != "LimitsExceeded":
            raise AssertionError(f"{instance_id}/{arm}: unexpected exit")
        if row["submission_bytes"] != 0:
            raise AssertionError(f"{instance_id}/{arm}: unexpected submission")
        if row["submission_command_indices"]:
            raise AssertionError(f"{instance_id}/{arm}: submission ran")
        official_arm = official["arms"][arm]
        if official_arm["resolved"] != 0 or official_arm["empty_patch"] != 1:
            raise AssertionError(
                f"{instance_id}/{arm}: official empty failure changed"
            )
        if runtime["calls"][arm] != STEP_LIMIT:
            raise AssertionError(f"{instance_id}/{arm}: runtime calls changed")
        if runtime["submission_bytes"][arm] != 0:
            raise AssertionError(
                f"{instance_id}/{arm}: runtime submission changed"
            )
        arms[arm] = row
    return {
        "instance_id": instance_id,
        "branch_request_index": branch["branch_request_index"],
        "shared_calls": branch["shared_calls"],
        "source_lengths": branch["source_lengths"],
        "arms": arms,
        "inputs": {
            "runtime_sha256": sha256(runtime_path),
            "official_sha256": sha256(official_path),
        },
    }


def run(output: Path) -> dict[str, Any]:
    result_path = output / "V43_RESULT.json"
    registration_path = output / "V43_REGISTRATION.json"
    result = read_json(result_path)
    if result["status"] != "FAIL_V43_DEVELOPMENT":
        raise AssertionError("V43 result is no longer a completed failure")
    if result["aggregate"]["complete_tasks"] != len(TASKS):
        raise AssertionError("V43 is no longer complete")
    if any(result["aggregate"]["resolved"].values()):
        raise AssertionError("V43 now contains an official pass")
    rows = [_task_audit(output, instance_id) for instance_id in TASKS]
    all_arms = [
        arm_row
        for task_row in rows
        for arm_row in task_row["arms"].values()
    ]
    successful_writes = [
        {
            "instance_id": task_row["instance_id"],
            "arm": arm,
            "request_index": request_index,
        }
        for task_row in rows
        for arm, arm_row in task_row["arms"].items()
        for request_index in arm_row[
            "successful_source_write_command_indices"
        ]
    ]
    if successful_writes != [
        {
            "instance_id": "sympy__sympy-21930",
            "arm": V40,
            "request_index": 20,
        },
        {
            "instance_id": "pytest-dev__pytest-5840",
            "arm": DENSE,
            "request_index": 19,
        },
    ]:
        raise AssertionError("late-write evidence changed")
    value = {
        "completed_at_utc": utc_now(),
        "status": "V43_BENCHMARK_SENSITIVITY_FAILURE_CALL_BUDGET_COLLAPSE",
        "classification": (
            "COMPLETE_ITT_ZERO_ACCURACY; NOT_AN_ALGORITHM_SEPARATION_RESULT"
        ),
        "tasks": rows,
        "aggregate": {
            "tasks": len(rows),
            "arms": len(all_arms),
            "limits_exceeded": sum(
                row["exit_status"] == "LimitsExceeded" for row in all_arms
            ),
            "empty_submissions": sum(
                row["submission_bytes"] == 0 for row in all_arms
            ),
            "official_empty_patches": len(all_arms),
            "successful_source_write_commands": successful_writes,
            "earliest_successful_source_write_request": min(
                row["request_index"] for row in successful_writes
            ),
            "submission_commands": sum(
                len(row["submission_command_indices"]) for row in all_arms
            ),
        },
        "causal_interpretation": [
            "All 18 arms reached the identical frozen 20-call limit and "
            "exited LimitsExceeded without invoking the submission command.",
            "Only two shell-successful source-write commands occurred, at "
            "requests 19 and 20. Even a useful edit at that point could not "
            "complete the required create/inspect/submit sequence.",
            "The six tasks reached the online treatment branch at request 8 "
            "with distinct V40 and General source plans and executed real KV "
            "copies, so the treatment mechanics ran.",
            "Because Dense also scored 0/6 with six empty patches, this cohort "
            "contains no Dense-pass task on which reuse damage can be "
            "observed. The 0/6 ties do not establish V40 preservation, "
            "General equivalence, or coding-aware failure.",
        ],
        "next_experiment_requirement": {
            "primary_accuracy_cohort": (
                "Select only with a treatment-blind Dense qualification run; "
                "freeze all Dense-pass tasks before revealing reuse outcomes."
            ),
            "secondary_rescue_cohort": (
                "Report Dense-fail tasks separately; never mix rescue with "
                "Dense-preservation claims."
            ),
            "call_budget": (
                "Increase the common all-arm call budget enough to leave the "
                "three required submission actions after a source edit; "
                "freeze it before any reuse treatment."
            ),
            "v43_rerun_or_replacement": False,
        },
        "inputs": {
            "v43_result_sha256": sha256(result_path),
            "v43_registration_sha256": sha256(registration_path),
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
    }
    write_json(output / "V43_CALL_BUDGET_COLLAPSE_AUDIT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = run(args.output)
    print(
        {
            "status": value["status"],
            "aggregate": value["aggregate"],
        }
    )


if __name__ == "__main__":
    main()
