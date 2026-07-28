#!/usr/bin/env python3
"""Freeze why V37 single-request guards failed their positive control."""

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
RUN = ARTIFACTS / "impactkv_v37a2_positive_control_astropy7336_20260728"
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v37a2_failure_audit_20260728"
V37 = "coding_patch_lifecycle_target_v37"


def _commands(path: Path) -> list[str]:
    trajectory = read_json(path)
    values: list[str] = []
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
        values.append(str(arguments.get("command") or ""))
    return values


def _client(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    official_path = RUN / "V25_OFFICIAL_RESULT.json"
    runtime_path = RUN / "V25_RESULT.json"
    official = read_json(official_path)
    runtime = read_json(runtime_path)
    resolved = {
        arm: row["resolved"] for arm, row in official["arms"].items()
    }
    if resolved != {V37: 0, "dense": 1, "general": 1}:
        raise AssertionError("unexpected V37A2 official outcome")
    if runtime["server"]["candidate_target_vetoes"] != 3:
        raise AssertionError("unexpected V37 veto count")
    if runtime["server"]["copy_counts"][V37] != 4:
        raise AssertionError("unexpected V37 copy count")

    candidate_traj = next((RUN / V37).glob("*/*.traj.json"))
    dense_traj = next((RUN / "dense").glob("*/*.traj.json"))
    candidate_commands = _commands(candidate_traj)
    dense_commands = _commands(dense_traj)
    common_prefix = 0
    for candidate, dense in zip(candidate_commands, dense_commands):
        if candidate != dense:
            break
        common_prefix += 1
    if common_prefix != 12:
        raise AssertionError("V37/Dense command prefix changed")
    candidate_client = _client(RUN / V37 / "CLIENT_LEDGER.jsonl")
    copied = [
        row["request_index"]
        for row in candidate_client
        if row["copied_tokens_planned"] > 0
    ]
    vetoed = [
        row["request_index"]
        for row in candidate_client
        if row["reuse_policy_decision"].get("target_vetoed")
    ]

    value = {
        "completed_at_utc": utc_now(),
        "status": "PASS_V37A2_FAILURE_AUDIT",
        "inputs": {
            "official_result_sha256": sha256(official_path),
            "runtime_result_sha256": sha256(runtime_path),
            "candidate_trajectory_sha256": sha256(candidate_traj),
            "dense_trajectory_sha256": sha256(dense_traj),
        },
        "official_resolved": resolved,
        "mechanism": {
            "branch_request_index": runtime["branch"][
                "branch_request_index"
            ],
            "candidate_target_vetoes": runtime["server"][
                "candidate_target_vetoes"
            ],
            "candidate_target_copies": runtime["server"]["copy_counts"][V37],
            "vetoed_request_indices": vetoed,
            "copied_request_indices": copied,
            "candidate_dense_identical_tool_command_prefix": common_prefix,
            "candidate_final_command": candidate_commands[-1],
            "dense_final_command": dense_commands[-1],
        },
        "conclusion": (
            "V37 fired at the intended diff/validation boundaries but still "
            "performed four lossy copies inside the same patch-finalization "
            "episode. Candidate and Dense issued the same first 12 tool "
            "commands, yet the candidate's final submission omitted the "
            "patch. This falsifies isolated one-request protection on this "
            "positive control and motivates testing a persistent online "
            "commit-phase Dense latch. The result does not prove that every "
            "intermediate copy caused the failure."
        ),
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
    }
    write_json(output / "V37A2_FAILURE_AUDIT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = run(args.output)
    print({"status": value["status"], "mechanism": value["mechanism"]})


if __name__ == "__main__":
    main()
