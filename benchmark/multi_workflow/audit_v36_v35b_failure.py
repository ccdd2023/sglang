#!/usr/bin/env python3
"""Freeze the request-level explanation of the V36 V35B failure."""

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
V36 = ARTIFACTS / "impactkv_v36_v35b_task_level_campaign_20260727"
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v36_failure_audit_20260728"
V35B = "coding_version_validation_target_v35b"


def _commands(path: Path) -> list[str]:
    trajectory = read_json(path)
    commands: list[str] = []
    for message in trajectory["messages"]:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or ()
        if not calls:
            continue
        call = calls[0].get("function", calls[0])
        arguments: Any = call.get("arguments") or {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        commands.append(str(arguments.get("command") or ""))
    return commands


def _client_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    result_path = V36 / "V36_RESULT.json"
    result = read_json(result_path)
    damages = [
        row for row in result["tasks"] if row["candidate_damage"]
    ]
    if len(damages) != 1:
        raise AssertionError("V36 must contain exactly one V35B damage")
    damage = damages[0]
    if damage["instance_id"] != "astropy__astropy-7336":
        raise AssertionError("unexpected V36 damage task")

    task = V36 / "tasks" / damage["instance_id"]
    candidate_traj = next((task / V35B).glob("*/*.traj.json"))
    dense_traj = next((task / "dense").glob("*/*.traj.json"))
    candidate_commands = _commands(candidate_traj)
    dense_commands = _commands(dense_traj)
    candidate_rows = _client_rows(task / V35B / "CLIENT_LEDGER.jsonl")
    q12 = next(
        row for row in candidate_rows if row["request_index"] == 12
    )
    if q12["copied_tokens_planned"] != 986:
        raise AssertionError("V36 damage copy changed")
    if "git diff" not in candidate_commands[10]:
        raise AssertionError("q11 must expose the patch diff")
    if "COMPLETE_TASK" not in candidate_commands[11]:
        raise AssertionError("q12 must be the malformed submit decision")
    # Dense request indices restart at the branch.  Its fourth request is
    # global q12, and the preceding commands/observations remain paired.
    if "find /testbed" not in dense_commands[11]:
        raise AssertionError("Dense q12 counterfactual changed")

    value = {
        "completed_at_utc": utc_now(),
        "status": "PASS_V36_FAILURE_AUDIT",
        "inputs": {
            "v36_result": str(result_path),
            "v36_result_sha256": sha256(result_path),
            "candidate_trajectory_sha256": sha256(candidate_traj),
            "dense_trajectory_sha256": sha256(dense_traj),
        },
        "aggregate": result["aggregate"],
        "failure_chain": {
            "instance_id": damage["instance_id"],
            "official_outcome": damage["resolved"],
            "candidate_damage": True,
            "candidate_minus_general": 0,
            "first_v35b_veto_request": damage["branch_request_index"],
            "online_visible_patch_diff_request": 11,
            "missed_patch_decision_request": 12,
            "missed_request_copied_tokens": q12[
                "copied_tokens_planned"
            ],
            "candidate_q12_command": candidate_commands[11],
            "dense_q12_command": dense_commands[11],
        },
        "conclusion": (
            "V35B protected one first-validation decision but did not protect "
            "the later online-visible patch-review/submission boundary.  On "
            "the only V36 Dense-pass damage, q12 copied 986 stale-position KV "
            "tokens after q11 displayed git diff; the paired Dense trajectory "
            "continued inspection and passed.  This motivates a prospective "
            "patch-lifecycle guard, but this tuned task can only be a positive "
            "control and cannot establish generalization."
        ),
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
    }
    write_json(output / "V36_FAILURE_AUDIT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = run(args.output)
    print(
        {
            "status": value["status"],
            "failure_chain": value["failure_chain"],
        }
    )


if __name__ == "__main__":
    main()
