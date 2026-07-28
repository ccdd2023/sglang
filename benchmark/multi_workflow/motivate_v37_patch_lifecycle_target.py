#!/usr/bin/env python3
"""Preregister V37 patch-lifecycle target-veto reach."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from benchmark.multi_workflow import (
    motivate_v33_state_transition_target as v33a,
)
from benchmark.multi_workflow.coding_reuse_policy import (
    _tool_command,
    critical_coding_event_reasons,
    is_high_value_executable_failure,
    is_successful_focused_validation,
)
from benchmark.multi_workflow.context_bounded_litellm_model import (
    ContextBoundedLitellmModel,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v37_patch_lifecycle_motivation_20260728"
AUDIT = (
    ARTIFACTS
    / "impactkv_v36_failure_audit_20260728"
    / "V36_FAILURE_AUDIT.json"
)
POLICY = Path(__file__).with_name("coding_reuse_policy.py")
ROLLING_GROUPS = 6
_SHELL_SOURCE_WRITE = re.compile(
    r"\b(?:cat|printf|echo)\b[^\n]*(?:>>|>)\s*"
    r"(?:/testbed/|\./)?[^\s;&|]+"
    r"\.(?:py|pyi|toml|yaml|yml|json|cfg|ini)\b",
    re.I,
)


def _shell_source_write(
    group: Sequence[dict[str, Any]],
) -> bool:
    commands = "\n".join(
        command
        for message in group
        if (command := _tool_command(message))
    )
    return bool(_SHELL_SOURCE_WRITE.search(commands))


def _state_change_reasons(
    group: Sequence[dict[str, Any]],
) -> list[str]:
    reasons = critical_coding_event_reasons(group)
    if _shell_source_write(group):
        reasons = [*reasons, "shell_source_write"]
    return list(dict.fromkeys(reasons))


def coding_patch_lifecycle_reasons(
    groups: Sequence[Sequence[dict[str, Any]]],
) -> list[str]:
    """Protect repair, first validation, and patch-review decisions."""

    if not groups:
        return []
    latest = groups[-1]
    if is_high_value_executable_failure(latest):
        return ["executable_failure_before_repair"]
    if "repository_diff_observed" in _state_change_reasons(latest):
        return ["patch_diff_before_submission_decision"]
    if not is_successful_focused_validation(latest):
        return []
    state_changes = [
        index
        for index, group in enumerate(groups[:-1])
        if any(
            reason
            in {
                "repository_mutation_command",
                "repository_diff_observed",
                "shell_source_write",
            }
            for reason in _state_change_reasons(group)
        )
    ]
    if not state_changes:
        return []
    latest_change = state_changes[-1]
    if any(
        is_successful_focused_validation(group)
        for group in groups[latest_change + 1 : -1]
    ):
        return []
    return ["first_validation_of_version_before_submit"]


def register(output: Path) -> dict[str, Any]:
    path = output / "V37_MOTIVATION_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    trajectories = v33a._trajectories()
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_V37_MOTIVATION_ANALYSIS",
        "motivation": (
            "V36 tied General at 3/6 and damaged one of four Dense-pass "
            "tasks.  Its only damage copied 986 tokens on the request after "
            "an online-visible git diff, while the paired Dense control "
            "continued inspection and passed.  V37 adds the missing "
            "patch-review/submission boundary and recognizes shell writes to "
            "source files, while retaining V35B's sparse failure and "
            "first-validation guards."
        ),
        "candidate": {
            "name": "coding_patch_lifecycle_target_v37",
            "current_target_veto": True,
            "future_source_policy": "general_contiguous_4096",
            "events": {
                "executable_failure_before_repair": True,
                "first_validation_of_version_before_submit": True,
                "patch_diff_before_submission_decision": True,
            },
            "recognize_shell_source_write": True,
            "mutation_alone_vetoes": False,
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "analysis": {
            "only_completed_groups_before_each_candidate_request": True,
            "same_21_frozen_dense_trajectories_as_v33": True,
            "task_outcomes_read": False,
            "rolling_groups": ROLLING_GROUPS,
        },
        "frozen_gates": {
            "v32r_tasks_reached_min": 2,
            "full18_tasks_reached_min": 13,
            "guarded_request_rate_min": 0.15,
            "guarded_request_rate_max": 0.28,
            "executable_failure_events_min": 5,
            "first_validation_events_min": 3,
            "patch_diff_events_min": 10,
            "shell_source_write_tasks_min": 1,
        },
        "inputs": {
            "v36_failure_audit": str(AUDIT),
            "v36_failure_audit_sha256": sha256(AUDIT),
            "policy_sha256": sha256(POLICY),
            "script_sha256": sha256(Path(__file__)),
            "trajectory_sha256": {
                str(path): sha256(path)
                for paths in trajectories.values()
                for path in paths
            },
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


def _measure(path: Path) -> dict[str, Any]:
    trajectory = read_json(path)
    calls = int(trajectory["info"]["model_stats"]["api_calls"])
    groups = ContextBoundedLitellmModel._turn_groups(
        trajectory["messages"][2:]
    )
    eligible = 0
    guarded: list[int] = []
    reasons: Counter[str] = Counter()
    shell_write_seen = False
    for completed_index, group in enumerate(groups, start=1):
        if completed_index >= calls:
            break
        if not any(message.get("role") == "tool" for message in group):
            continue
        if completed_index < ROLLING_GROUPS:
            continue
        eligible += 1
        rolling = groups[
            max(0, completed_index - ROLLING_GROUPS) : completed_index
        ]
        shell_write_seen |= any(_shell_source_write(item) for item in rolling)
        current = coding_patch_lifecycle_reasons(rolling)
        if current:
            guarded.append(completed_index + 1)
            reasons.update(current)
    return {
        "instance_id": trajectory["instance_id"],
        "calls": calls,
        "eligible_target_requests": eligible,
        "guarded_requests": len(guarded),
        "guarded_request_indices": guarded,
        "reason_counts": dict(sorted(reasons.items())),
        "shell_source_write_seen": shell_write_seen,
        "reached": bool(guarded),
    }


def run(output: Path) -> dict[str, Any]:
    registration = register(output)
    cohorts = {
        name: [_measure(path) for path in paths]
        for name, paths in v33a._trajectories().items()
    }
    rows = [row for cohort in cohorts.values() for row in cohort]
    eligible = sum(row["eligible_target_requests"] for row in rows)
    guarded = sum(row["guarded_requests"] for row in rows)
    rate = guarded / eligible
    reasons: Counter[str] = Counter()
    for row in rows:
        reasons.update(row["reason_counts"])
    frozen = registration["frozen_gates"]
    gates = {
        "v32r_tasks_reached_min": sum(
            row["reached"] for row in cohorts["v32r"]
        )
        >= frozen["v32r_tasks_reached_min"],
        "full18_tasks_reached_min": sum(
            row["reached"] for row in cohorts["full18"]
        )
        >= frozen["full18_tasks_reached_min"],
        "guarded_request_rate_min": (
            rate >= frozen["guarded_request_rate_min"]
        ),
        "guarded_request_rate_max": (
            rate <= frozen["guarded_request_rate_max"]
        ),
        "executable_failure_events_min": (
            reasons["executable_failure_before_repair"]
            >= frozen["executable_failure_events_min"]
        ),
        "first_validation_events_min": (
            reasons["first_validation_of_version_before_submit"]
            >= frozen["first_validation_events_min"]
        ),
        "patch_diff_events_min": (
            reasons["patch_diff_before_submission_decision"]
            >= frozen["patch_diff_events_min"]
        ),
        "shell_source_write_tasks_min": sum(
            row["shell_source_write_seen"] for row in rows
        )
        >= frozen["shell_source_write_tasks_min"],
    }
    value = {
        "completed_at_utc": utc_now(),
        "status": (
            "PASS_V37_MOTIVATION"
            if all(gates.values())
            else "FAIL_V37_MOTIVATION"
        ),
        "registration_sha256": sha256(
            output / "V37_MOTIVATION_REGISTRATION.json"
        ),
        "cohorts": cohorts,
        "aggregate": {
            "tasks": len(rows),
            "tasks_reached": sum(row["reached"] for row in rows),
            "eligible_target_requests": eligible,
            "guarded_requests": guarded,
            "guarded_request_rate": rate,
            "reason_counts": dict(sorted(reasons.items())),
            "shell_source_write_tasks": sum(
                row["shell_source_write_seen"] for row in rows
            ),
        },
        "gates": gates,
        "decision": (
            "Implement serving V37 and verify exact motivation parity."
            if all(gates.values())
            else "Reject V37 before GPU work."
        ),
    }
    write_json(output / "V37_MOTIVATION_RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("register", "run"), nargs="?", default="run"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = (
        register(args.output)
        if args.command == "register"
        else run(args.output)
    )
    print(
        {
            "status": value["status"],
            "aggregate": value.get("aggregate"),
            "gates": value.get("gates"),
        }
    )


if __name__ == "__main__":
    main()
