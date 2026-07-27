#!/usr/bin/env python3
"""Preregister V33B reach after adding a two-interaction veto cooldown."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import (
    motivate_v33_state_transition_target as v33a,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v33b_cooldown_target_motivation_20260727"
)
POLICY = Path(__file__).with_name("coding_reuse_policy.py")


def register(output: Path) -> dict[str, Any]:
    path = output / "V33B_MOTIVATION_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    trajectories = v33a._trajectories()
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_V33B_MOTIVATION_ANALYSIS",
        "motivation": (
            "V33A reached all 21 tasks but vetoed 55.51% of eligible target "
            "requests, failing its 45% speed-budget proxy."
        ),
        "candidate": {
            "name": "coding_state_transition_target_v33b",
            "inherits_v33a_raw_events": True,
            "cooldown_completed_interactions": 2,
            "maximum_veto_frequency": "one in any three interactions",
            "current_target_veto": True,
            "future_source_policy": "general_contiguous_4096",
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "analysis": {
            "only_completed_groups_before_each_candidate_request": True,
            "task_outcomes_read": False,
            "same_21_frozen_dense_trajectories_as_v33a": True,
        },
        "frozen_gates": {
            "v32r_tasks_reached_min": 3,
            "full18_tasks_reached_min": 12,
            "guarded_request_rate_min": 0.10,
            "guarded_request_rate_max": 0.40,
            "relative_veto_reduction_vs_v33a_min": 0.25,
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


def run(output: Path) -> dict[str, Any]:
    registration = register(output)
    cohorts = {
        name: [v33a._measure(path) for path in paths]
        for name, paths in v33a._trajectories().items()
    }
    rows = [row for cohort in cohorts.values() for row in cohort]
    eligible = sum(row["eligible_target_requests"] for row in rows)
    guarded = sum(row["guarded_requests"] for row in rows)
    rate = guarded / eligible
    v33a_result = read_json(
        v33a.DEFAULT_OUTPUT / "V33_MOTIVATION_RESULT.json"
    )
    old_guarded = v33a_result["aggregate"]["guarded_requests"]
    reduction = (old_guarded - guarded) / old_guarded
    reason_counts = Counter(
        reason
        for row in rows
        for reason, count in row["reason_counts"].items()
        for _ in range(count)
    )
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
    }
    value = {
        "completed_at_utc": utc_now(),
        "status": (
            "PASS_V33B_MOTIVATION"
            if all(gates.values())
            else "FAIL_V33B_MOTIVATION"
        ),
        "registration_sha256": sha256(
            output / "V33B_MOTIVATION_REGISTRATION.json"
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
            "Implement V33B in serving and paired runner."
            if all(gates.values())
            else "Reject or revise V33B before GPU work."
        ),
    }
    write_json(output / "V33B_MOTIVATION_RESULT.json", value)
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
