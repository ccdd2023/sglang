#!/usr/bin/env python3
"""Preregister and measure V33 target-veto reach on frozen Dense traces."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.coding_reuse_policy import (
    coding_state_transition_target_reasons,
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
    ARTIFACTS / "impactkv_v33_state_transition_motivation_20260727"
)
V32R_TASKS = (
    ARTIFACTS
    / "impactkv_v32r_stream_close_replication_20260727/tasks"
)
FULL18 = (
    ARTIFACTS
    / "swebench_verified_bridge_v1_20260724"
    / "agent_dense_contextbound_v1/full_18"
)
POLICY = Path(__file__).with_name("coding_reuse_policy.py")
ROLLING_GROUPS = 6


def _trajectories() -> dict[str, list[Path]]:
    return {
        "v32r": sorted(V32R_TASKS.glob("*/dense/*/*.traj.json")),
        "full18": sorted(FULL18.glob("*/*.traj.json")),
    }


def register(output: Path) -> dict[str, Any]:
    path = output / "V33_MOTIVATION_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    trajectories = _trajectories()
    if len(trajectories["v32r"]) != 3 or len(trajectories["full18"]) != 18:
        raise AssertionError("unexpected V33 motivation cohort size")
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_V33_MOTIVATION_ANALYSIS",
        "motivation": (
            "V32R reached a policy branch on only 1/3 tasks and changed zero "
            "final patches. Test whether current-request target vetoes at "
            "coding state transitions reach earlier decisions without "
            "degenerating to Dense."
        ),
        "candidate": {
            "name": "coding_state_transition_target_v33",
            "current_target_veto": True,
            "future_source_policy": "general_contiguous_4096",
            "events": [
                "repository mutation or observed diff",
                "executable failure",
                "entry into substantial successful read-only evidence",
                "entry into successful execution/validation evidence",
            ],
            "same_evidence_phase_cooldown": (
                "Do not veto for consecutive observations in the same "
                "read-only or successful-execution phase."
            ),
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "analysis": {
            "only_completed_groups_before_each_candidate_request": True,
            "rolling_groups": ROLLING_GROUPS,
            "target_possible_after_completed_group": 6,
            "task_outcomes_read": False,
        },
        "frozen_gates": {
            "v32r_tasks_reached_min": 3,
            "full18_tasks_reached_min": 12,
            "guarded_request_rate_min": 0.10,
            "guarded_request_rate_max": 0.45,
            "critical_event_requests_min": 2,
        },
        "inputs": {
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
    instance_id = trajectory["instance_id"]
    calls = int(trajectory["info"]["model_stats"]["api_calls"])
    groups = ContextBoundedLitellmModel._turn_groups(
        trajectory["messages"][2:]
    )
    eligible = 0
    guarded = []
    reasons = Counter()
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
        current_reasons = coding_state_transition_target_reasons(rolling)
        if current_reasons:
            guarded.append(completed_index + 1)
            reasons.update(current_reasons)
    return {
        "instance_id": instance_id,
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
        for name, paths in _trajectories().items()
    }
    all_rows = [row for rows in cohorts.values() for row in rows]
    eligible = sum(row["eligible_target_requests"] for row in all_rows)
    guarded = sum(row["guarded_requests"] for row in all_rows)
    reason_counts = Counter(
        {
            reason: sum(
                row["reason_counts"].get(reason, 0) for row in all_rows
            )
            for reason in {
                reason
                for row in all_rows
                for reason in row["reason_counts"]
            }
        }
    )
    rate = guarded / eligible
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
        "critical_event_requests_min": (
            sum(
                count
                for reason, count in reason_counts.items()
                if reason
                in {
                    "repository_mutation_command",
                    "repository_diff_observed",
                    "executable_failure",
                }
            )
            >= registration["frozen_gates"]["critical_event_requests_min"]
        ),
    }
    value = {
        "completed_at_utc": utc_now(),
        "status": (
            "PASS_V33_MOTIVATION"
            if all(gates.values())
            else "FAIL_V33_MOTIVATION"
        ),
        "registration_sha256": sha256(
            output / "V33_MOTIVATION_REGISTRATION.json"
        ),
        "cohorts": cohorts,
        "aggregate": {
            "tasks": len(all_rows),
            "tasks_reached": sum(row["reached"] for row in all_rows),
            "eligible_target_requests": eligible,
            "guarded_requests": guarded,
            "guarded_request_rate": rate,
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "gates": gates,
        "decision": (
            "Implement V33 current-target veto and paired runner."
            if all(gates.values())
            else "Reject or narrow/widen V33 before GPU work."
        ),
    }
    write_json(output / "V33_MOTIVATION_RESULT.json", value)
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
    value = register(args.output) if args.command == "register" else run(args.output)
    print(
        {
            "status": value["status"],
            "aggregate": value.get("aggregate"),
            "gates": value.get("gates"),
        }
    )


if __name__ == "__main__":
    main()
