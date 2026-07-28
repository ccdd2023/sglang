#!/usr/bin/env python3
"""Preregister exploration-reuse/commit-Dense phase reach for V38."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from benchmark.multi_workflow import (
    motivate_v33_state_transition_target as v33a,
)
from benchmark.multi_workflow.coding_reuse_policy import (
    critical_coding_event_reasons,
    is_shell_source_write,
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
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v38_commit_phase_motivation_20260728"
AUDIT = (
    ARTIFACTS
    / "impactkv_v37a2_failure_audit_20260728"
    / "V37A2_FAILURE_AUDIT.json"
)
POLICY = Path(__file__).with_name("coding_reuse_policy.py")
ROLLING_GROUPS = 6


def is_repository_mutation(
    group: Sequence[dict[str, Any]],
) -> bool:
    return is_shell_source_write(group) or (
        "repository_mutation_command"
        in critical_coding_event_reasons(group)
    )


def register(output: Path) -> dict[str, Any]:
    path = output / "V38_MOTIVATION_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    trajectories = v33a._trajectories()
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_V38_MOTIVATION_ANALYSIS",
        "motivation": (
            "V37 fired three intended single-request guards but its four "
            "intermediate lossy copies still led to a General-only official "
            "resolution. Treat coding as two online phases: use General KV "
            "reuse while exploring, then latch to Dense immediately after "
            "the first observed repository source/config mutation and remain "
            "Dense for the rest of that task."
        ),
        "candidate": {
            "name": "coding_commit_phase_dense_v38",
            "exploration_phase": "general_contiguous_4096",
            "commit_phase": "no lossy target and no future source",
            "latch_event": (
                "first completed repository mutation command or shell "
                "redirection to a source/configuration file"
            ),
            "latch_reset": "new benchmark task only",
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "analysis": {
            "only_completed_groups_before_each_candidate_request": True,
            "same_21_frozen_dense_trajectories_as_v33": True,
            "task_outcomes_read": False,
            "rolling_groups_for_reuse_eligibility": ROLLING_GROUPS,
            "mutation_latch_reads_full_completed_online_history": True,
        },
        "frozen_gates": {
            "v32r_tasks_latched_min": 2,
            "full18_tasks_latched_min": 13,
            "eligible_requests_min": 200,
            "exploration_reuse_requests_min": 35,
            "exploration_reuse_rate_min": 0.15,
            "commit_dense_rate_min": 0.35,
            "commit_dense_rate_max": 0.85,
            "shell_source_write_tasks_min": 1,
        },
        "inputs": {
            "v37a2_failure_audit": str(AUDIT),
            "v37a2_failure_audit_sha256": sha256(AUDIT),
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
    latched = False
    latch_after_group: int | None = None
    eligible = 0
    exploration = 0
    commit = 0
    shell_write_seen = False
    for completed_index, group in enumerate(groups, start=1):
        if completed_index >= calls:
            break
        mutation = is_repository_mutation(group)
        shell_write_seen |= is_shell_source_write(group)
        if mutation and not latched:
            latched = True
            latch_after_group = completed_index
        if not any(message.get("role") == "tool" for message in group):
            continue
        if completed_index < ROLLING_GROUPS:
            continue
        eligible += 1
        if latched:
            commit += 1
        else:
            exploration += 1
    return {
        "instance_id": trajectory["instance_id"],
        "calls": calls,
        "eligible_target_requests": eligible,
        "exploration_reuse_requests": exploration,
        "commit_dense_requests": commit,
        "latch_after_completed_group": latch_after_group,
        "latched": latch_after_group is not None,
        "shell_source_write_seen": shell_write_seen,
    }


def run(output: Path) -> dict[str, Any]:
    registration = register(output)
    cohorts = {
        name: [_measure(path) for path in paths]
        for name, paths in v33a._trajectories().items()
    }
    rows = [row for cohort in cohorts.values() for row in cohort]
    eligible = sum(row["eligible_target_requests"] for row in rows)
    exploration = sum(row["exploration_reuse_requests"] for row in rows)
    commit = sum(row["commit_dense_requests"] for row in rows)
    frozen = registration["frozen_gates"]
    gates = {
        "v32r_tasks_latched_min": sum(
            row["latched"] for row in cohorts["v32r"]
        )
        >= frozen["v32r_tasks_latched_min"],
        "full18_tasks_latched_min": sum(
            row["latched"] for row in cohorts["full18"]
        )
        >= frozen["full18_tasks_latched_min"],
        "eligible_requests_min": eligible
        >= frozen["eligible_requests_min"],
        "exploration_reuse_requests_min": exploration
        >= frozen["exploration_reuse_requests_min"],
        "exploration_reuse_rate_min": exploration / eligible
        >= frozen["exploration_reuse_rate_min"],
        "commit_dense_rate_min": commit / eligible
        >= frozen["commit_dense_rate_min"],
        "commit_dense_rate_max": commit / eligible
        <= frozen["commit_dense_rate_max"],
        "shell_source_write_tasks_min": sum(
            row["shell_source_write_seen"] for row in rows
        )
        >= frozen["shell_source_write_tasks_min"],
    }
    value = {
        "completed_at_utc": utc_now(),
        "status": (
            "PASS_V38_MOTIVATION"
            if all(gates.values())
            else "FAIL_V38_MOTIVATION"
        ),
        "registration_sha256": sha256(
            output / "V38_MOTIVATION_REGISTRATION.json"
        ),
        "cohorts": cohorts,
        "aggregate": {
            "tasks": len(rows),
            "tasks_latched": sum(row["latched"] for row in rows),
            "eligible_target_requests": eligible,
            "exploration_reuse_requests": exploration,
            "commit_dense_requests": commit,
            "exploration_reuse_rate": exploration / eligible,
            "commit_dense_rate": commit / eligible,
            "shell_source_write_tasks": sum(
                row["shell_source_write_seen"] for row in rows
            ),
        },
        "gates": gates,
        "decision": (
            "Implement a persistent online V38 phase latch."
            if all(gates.values())
            else "Reject V38 before GPU work."
        ),
    }
    write_json(output / "V38_MOTIVATION_RESULT.json", value)
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
