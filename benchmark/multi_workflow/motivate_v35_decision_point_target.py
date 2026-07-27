#!/usr/bin/env python3
"""Preregister V35 target-veto reach at coding decision points."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from benchmark.multi_workflow import (
    motivate_v33_state_transition_target as v33a,
)
from benchmark.multi_workflow.coding_reuse_policy import (
    critical_coding_event_reasons,
    is_concrete_source_read,
    is_high_value_executable_failure,
    is_successful_executable_evidence,
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
    ARTIFACTS / "impactkv_v35_decision_point_motivation_20260727"
)
POLICY = Path(__file__).with_name("coding_reuse_policy.py")
ROLLING_GROUPS = 6


def coding_decision_point_reasons(
    groups: Sequence[Sequence[dict[str, Any]]],
) -> list[str]:
    """Classify online-visible observations preceding high-stakes decisions."""

    if not groups:
        return []
    latest = groups[-1]
    if is_concrete_source_read(latest):
        return ["concrete_source_read_before_edit"]
    if is_high_value_executable_failure(latest):
        return ["executable_failure_before_repair"]
    prior_state_change = any(
        reason
        in {"repository_mutation_command", "repository_diff_observed"}
        for group in groups[:-1]
        for reason in critical_coding_event_reasons(group)
    )
    if (
        prior_state_change
        and is_successful_executable_evidence(latest)
    ):
        return ["validated_change_before_submit"]
    return []


def register(output: Path) -> dict[str, Any]:
    path = output / "V35_MOTIVATION_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    trajectories = v33a._trajectories()
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_V35_MOTIVATION_ANALYSIS",
        "motivation": (
            "V34 vetoed after any mutation, diff, or executable failure and "
            "causally changed a Pylint trajectory, but neither reproduced a "
            "rescue nor met its early-reach gate. Whole-request protection "
            "should be reserved for observations that directly precede an "
            "edit, repair, or submission decision, rather than every coding "
            "state change."
        ),
        "candidate": {
            "name": "coding_decision_point_target_v35",
            "current_target_veto": True,
            "future_source_policy": "general_contiguous_4096",
            "events": {
                "concrete_source_read_before_edit": (
                    "substantial successful non-test Python source read"
                ),
                "executable_failure_before_repair": (
                    "high-value executable or focused-test failure"
                ),
                "validated_change_before_submit": (
                    "successful execution after a mutation or observed diff "
                    "within the online rolling window"
                ),
            },
            "mutation_or_diff_alone_vetoes": False,
            "generic_read_phase_vetoes": False,
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
            "v32r_tasks_reached_min": 1,
            "full18_tasks_reached_min": 12,
            "guarded_request_rate_min": 0.10,
            "guarded_request_rate_max": 0.35,
            "relative_veto_reduction_vs_v33a_min": 0.45,
            "concrete_source_read_events_min": 5,
            "executable_failure_events_min": 5,
            "validated_change_events_min": 3,
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
        rolling = groups[
            max(0, completed_index - ROLLING_GROUPS) : completed_index
        ]
        current_reasons = coding_decision_point_reasons(rolling)
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
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reason_counts.update(row["reason_counts"])
    v33a_result = read_json(
        v33a.DEFAULT_OUTPUT / "V33_MOTIVATION_RESULT.json"
    )
    old_guarded = int(v33a_result["aggregate"]["guarded_requests"])
    reduction = (old_guarded - guarded) / old_guarded
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
        "relative_veto_reduction_vs_v33a_min": (
            reduction >= frozen["relative_veto_reduction_vs_v33a_min"]
        ),
        "concrete_source_read_events_min": (
            reason_counts["concrete_source_read_before_edit"]
            >= frozen["concrete_source_read_events_min"]
        ),
        "executable_failure_events_min": (
            reason_counts["executable_failure_before_repair"]
            >= frozen["executable_failure_events_min"]
        ),
        "validated_change_events_min": (
            reason_counts["validated_change_before_submit"]
            >= frozen["validated_change_events_min"]
        ),
    }
    value = {
        "completed_at_utc": utc_now(),
        "status": (
            "PASS_V35_MOTIVATION"
            if all(gates.values())
            else "FAIL_V35_MOTIVATION"
        ),
        "registration_sha256": sha256(
            output / "V35_MOTIVATION_REGISTRATION.json"
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
            "Implement V35 in serving and paired runner."
            if all(gates.values())
            else "Reject or revise V35 before GPU work."
        ),
    }
    write_json(output / "V35_MOTIVATION_RESULT.json", value)
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
