#!/usr/bin/env python3
"""Audit why V38 did not separate from General in frozen V39."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
RUN = ARTIFACTS / "impactkv_v39_v38_independent_20260728"
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v39_v38_equivalence_audit_20260728"
V38 = "coding_commit_phase_dense_v38"
ARMS = (V38, "general", "dense")


def _trajectory(task: Path, arm: str) -> Path:
    paths = list((task / arm).glob("*/*.traj.json"))
    if len(paths) != 1:
        raise AssertionError(f"expected one {arm} trajectory under {task}")
    return paths[0]


def _actions(path: Path) -> list[dict[str, Any]]:
    """Return assistant decisions, excluding timestamps and tool observations."""

    value = read_json(path)
    return [
        {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls") or [],
        }
        for message in value["messages"]
        if message.get("role") == "assistant"
    ]


def _common_prefix(left: list[Any], right: list[Any]) -> int:
    count = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        count += 1
    return count


def _client(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run(output: Path) -> dict[str, Any]:
    result_path = RUN / "V39_RESULT.json"
    result = read_json(result_path)
    if result["status"] != "FAIL_V39_DEVELOPMENT":
        raise AssertionError("V39 outcome changed")
    if result["aggregate"]["resolved"] != {
        V38: 3,
        "dense": 2,
        "general": 3,
    }:
        raise AssertionError("unexpected V39 resolved counts")

    rows: list[dict[str, Any]] = []
    for summary in result["tasks"]:
        instance_id = summary["instance_id"]
        task = RUN / "tasks" / instance_id
        official_path = task / "V25_OFFICIAL_RESULT.json"
        official = read_json(official_path)
        resolved = {
            arm: int(official["arms"][arm]["resolved"]) for arm in ARMS
        }
        trajectories = {arm: _trajectory(task, arm) for arm in ARMS}
        hashes = {arm: sha256(path) for arm, path in trajectories.items()}
        actions = {arm: _actions(path) for arm, path in trajectories.items()}
        patches = {
            arm: json.loads(
                (task / arm / "predictions.jsonl").read_text(
                    encoding="utf-8"
                )
            )["model_patch"]
            for arm in ARMS
        }
        clients = {
            arm: _client(task / arm / "CLIENT_LEDGER.jsonl")
            for arm in ARMS
        }
        copied = {
            arm: sum(
                int(record["copied_tokens_planned"]) > 0
                for record in records
            )
            for arm, records in clients.items()
        }
        rows.append(
            {
                "instance_id": instance_id,
                "branch_kind": summary["branch_kind"],
                "branch_reached": summary["branch_reached"],
                "resolved": resolved,
                "trajectory_sha256": hashes,
                "all_trajectories_byte_identical": len(
                    set(hashes.values())
                )
                == 1,
                "candidate_general_patch_identical": (
                    patches[V38] == patches["general"]
                ),
                "candidate_dense_patch_identical": (
                    patches[V38] == patches["dense"]
                ),
                "candidate_general_action_common_prefix": _common_prefix(
                    actions[V38], actions["general"]
                ),
                "candidate_dense_action_common_prefix": _common_prefix(
                    actions[V38], actions["dense"]
                ),
                "candidate_actions": len(actions[V38]),
                "general_actions": len(actions["general"]),
                "dense_actions": len(actions["dense"]),
                "requests_with_lossy_copy": copied,
                "candidate_and_dense_are_zero_copy": (
                    copied[V38] == 0 and copied["dense"] == 0
                ),
            }
        )

    branch_rows = [row for row in rows if row["branch_reached"]]
    no_branch_rows = [row for row in rows if not row["branch_reached"]]
    candidate_general_outcome_equal = all(
        row["resolved"][V38] == row["resolved"]["general"] for row in rows
    )
    candidate_general_patch_equal = all(
        row["candidate_general_patch_identical"] for row in rows
    )
    zero_copy_dense_disagreement = [
        row["instance_id"]
        for row in rows
        if row["candidate_and_dense_are_zero_copy"]
        and row["resolved"][V38] != row["resolved"]["dense"]
    ]
    value = {
        "completed_at_utc": utc_now(),
        "status": "PASS_V39_V38_EQUIVALENCE_AUDIT",
        "inputs": {
            "v39_result": str(result_path),
            "v39_result_sha256": sha256(result_path),
            "official_result_sha256": {
                row["instance_id"]: sha256(
                    RUN
                    / "tasks"
                    / row["instance_id"]
                    / "V25_OFFICIAL_RESULT.json"
                )
                for row in rows
            },
        },
        "aggregate": {
            "tasks": len(rows),
            "branches_reached": len(branch_rows),
            "no_branch_tasks": len(no_branch_rows),
            "no_branch_trajectories_byte_identical": sum(
                row["all_trajectories_byte_identical"]
                for row in no_branch_rows
            ),
            "candidate_general_official_outcome_equal_on_all_tasks": (
                candidate_general_outcome_equal
            ),
            "candidate_general_final_patch_equal_on_all_tasks": (
                candidate_general_patch_equal
            ),
            "candidate_only_vs_general": sum(
                row["resolved"][V38] > row["resolved"]["general"]
                for row in rows
            ),
            "general_only_vs_candidate": sum(
                row["resolved"]["general"] > row["resolved"][V38]
                for row in rows
            ),
            "zero_copy_candidate_dense_outcome_disagreement_tasks": (
                zero_copy_dense_disagreement
            ),
        },
        "tasks": rows,
        "conclusions": [
            (
                "V38 is empirically equivalent to General on this frozen "
                "sample: every official outcome and every final patch is "
                "identical, including all four tasks where an online branch "
                "was reached."
            ),
            (
                "The two no-branch tasks are not treatment evidence: all "
                "three saved trajectories are byte-identical and inherit the "
                "shared Dense outcome."
            ),
            (
                "The SymPy candidate-minus-Dense rescue is not attributable "
                "to KV reuse. Both V38 and Dense made zero lossy copies after "
                "the same shared prefix, yet their trajectories eventually "
                "diverged. It is evidence of non-bitwise-deterministic decode "
                "noise under the fixed dispatch protocol."
            ),
            (
                "A later or broader binary Dense latch is therefore not "
                "motivated. The next candidate must change which KV content "
                "is reused while retaining a measurable intervention, and "
                "must be tested with counterbalanced/repeated zero-copy "
                "controls before treating a one-task accuracy flip as causal."
            ),
        ],
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
    }
    if not candidate_general_outcome_equal:
        raise AssertionError("V38/General outcome equivalence no longer holds")
    if not candidate_general_patch_equal:
        raise AssertionError("V38/General patch equivalence no longer holds")
    if len(no_branch_rows) != 2 or not all(
        row["all_trajectories_byte_identical"] for row in no_branch_rows
    ):
        raise AssertionError("unexpected no-branch behavior")
    if zero_copy_dense_disagreement != ["sympy__sympy-24539"]:
        raise AssertionError("unexpected zero-copy Dense disagreement")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "V39_V38_EQUIVALENCE_AUDIT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = run(args.output)
    print({"status": value["status"], "aggregate": value["aggregate"]})


if __name__ == "__main__":
    main()
