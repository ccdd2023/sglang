#!/usr/bin/env python3
"""Preregister V34 reach using only direct critical coding events."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import (
    motivate_v33_state_transition_target as v33a,
)
from benchmark.multi_workflow.coding_reuse_policy import (
    critical_coding_event_reasons,
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
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v34_critical_target_motivation_20260727"
)
POLICY = Path(__file__).with_name("coding_reuse_policy.py")
ROLLING_GROUPS = 6


def register(output: Path) -> dict[str, Any]:
    path = output / "V34_MOTIVATION_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    trajectories = v33a._trajectories()
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_V34_MOTIVATION_ANALYSIS",
        "motivation": (
            "V33A reached all tasks but vetoed 55.51% of eligible requests. "
            "V33B reduced this with a two-interaction cooldown, but its "
            "offline reach failed to reproduce online on Astropy-7671 and a "
            "late veto did not reproduce the earlier V31 Pylint-7277 rescue. "
            "Remove phase-transition events and cooldown; retain only direct "
            "repository mutation, observed diff, or executable-failure "
            "events so the current target is protected immediately after an "
            "online-visible coding state change."
        ),
        "candidate": {
            "name": "coding_critical_current_target_v34",
            "current_target_veto": True,
            "future_source_policy": "general_contiguous_4096",
            "events": [
                "repository_mutation_command",
                "repository_diff_observed",
                "executable_failure",
            ],
            "phase_transition_events": False,
            "cooldown": False,
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
            "full18_tasks_reached_min": 14,
            "guarded_request_rate_min": 0.20,
            "guarded_request_rate_max": 0.40,
            "relative_veto_reduction_vs_v33a_min": 0.35,
            "pylint_7277_first_guarded_request_max": 8,
            "astropy_7671_guarded_request_8": True,
        },
        "inputs": {
            "v33a_registration": str(
                v33a.DEFAULT_OUTPUT / "V33_MOTIVATION_REGISTRATION.json"
            ),
            "v33a_registration_sha256": sha256(
                v33a.DEFAULT_OUTPUT / "V33_MOTIVATION_REGISTRATION.json"
            ),
            "v33a_result": str(
                v33a.DEFAULT_OUTPUT / "V33_MOTIVATION_RESULT.json"
            ),
            "v33a_result_sha256": sha256(
                v33a.DEFAULT_OUTPUT / "V33_MOTIVATION_RESULT.json"
            ),
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
    for completed_index, group in enumerate(groups, start=1):
        if completed_index >= calls:
            break
        if not any(message.get("role") == "tool" for message in group):
            continue
        if completed_index < ROLLING_GROUPS:
            continue
        eligible += 1
        current_reasons = critical_coding_event_reasons(group)
        if current_reasons:
            guarded.append(completed_index + 1)
            reasons.update(current_reasons)
    return {
        "instance_id": trajectory["instance_id"],
        "calls": calls,
        "eligible_target_requests": eligible,
        "guarded_requests": len(guarded),
        "guarded_request_indices": guarded,
        "reason_counts": dict(sorted(reasons.items())),
        "reached": bool(guarded),
    }


def _row(
    cohorts: dict[str, list[dict[str, Any]]],
    instance_id: str,
) -> dict[str, Any]:
    return next(
        row
        for rows in cohorts.values()
        for row in rows
        if row["instance_id"] == instance_id
    )


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
    v33a_result = read_json(
        v33a.DEFAULT_OUTPUT / "V33_MOTIVATION_RESULT.json"
    )
    old_guarded = int(v33a_result["aggregate"]["guarded_requests"])
    reduction = (old_guarded - guarded) / old_guarded
    pylint = _row(cohorts, "pylint-dev__pylint-7277")
    astropy = _row(cohorts, "astropy__astropy-7671")
    gates = {
        "v32r_tasks_reached_min": sum(
            row["reached"] for row in cohorts["v32r"]
        )
        >= registration["frozen_gates"]["v32r_tasks_reached_min"],
        "full18_tasks_reached_min": sum(
            row["reached"] for row in cohorts["full18"]
        )
        >= registration["frozen_gates"]["full18_tasks_reached_min"],
        "guarded_request_rate_min": (
            rate
            >= registration["frozen_gates"]["guarded_request_rate_min"]
        ),
        "guarded_request_rate_max": (
            rate
            <= registration["frozen_gates"]["guarded_request_rate_max"]
        ),
        "relative_veto_reduction_vs_v33a_min": (
            reduction
            >= registration["frozen_gates"][
                "relative_veto_reduction_vs_v33a_min"
            ]
        ),
        "pylint_7277_first_guarded_request_max": (
            bool(pylint["guarded_request_indices"])
            and min(pylint["guarded_request_indices"])
            <= registration["frozen_gates"][
                "pylint_7277_first_guarded_request_max"
            ]
        ),
        "astropy_7671_guarded_request_8": (
            8 in astropy["guarded_request_indices"]
        ),
    }
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reason_counts.update(row["reason_counts"])
    value = {
        "completed_at_utc": utc_now(),
        "status": (
            "PASS_V34_MOTIVATION"
            if all(gates.values())
            else "FAIL_V34_MOTIVATION"
        ),
        "registration_sha256": sha256(
            output / "V34_MOTIVATION_REGISTRATION.json"
        ),
        "cohorts": cohorts,
        "aggregate": {
            "tasks": len(rows),
            "tasks_reached": sum(row["reached"] for row in rows),
            "eligible_target_requests": eligible,
            "guarded_requests": guarded,
            "guarded_request_rate": rate,
            "relative_veto_reduction_vs_v33a": reduction,
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "gates": gates,
        "decision": (
            "Implement V34 current-target veto in serving and paired runner."
            if all(gates.values())
            else "Reject or revise V34 before GPU work."
        ),
    }
    write_json(output / "V34_MOTIVATION_RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("register", "run"),
        nargs="?",
        default="run",
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
