#!/usr/bin/env python3
"""Audit whether V23 full18 task transitions began before KV treatment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_INPUT = ARTIFACTS / "impactkv_v23_full18_accuracy_20260727"
REGISTRATION = (
    Path(__file__).resolve().parent / "swebench_verified_bridge_v1.json"
)
GENERAL = "general"
CANDIDATE = "coding_post_mutation_target_prefix_v23"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assistant_key(message: dict[str, Any]) -> tuple[Any, ...]:
    calls = message.get("tool_calls") or ()
    commands = tuple(
        (
            call["function"]["name"],
            call["function"]["arguments"],
        )
        for call in calls
    )
    return message.get("content"), commands


def assistant_messages(root: Path, arm: str, instance_id: str):
    path = root / arm / "full_18" / instance_id / f"{instance_id}.traj.json"
    value = read_json(path)
    return (
        [row for row in value["messages"] if row["role"] == "assistant"],
        value,
        path,
    )


def client_rows(root: Path, arm: str) -> list[dict[str, Any]]:
    path = root / arm / "full_18" / "CLIENT_LEDGER.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit(root: Path) -> dict[str, Any]:
    result_path = root / "V23_FULL18_RESULT.json"
    result = read_json(result_path)
    transitions = result["paired_transition"]
    instances = [
        row["instance_id"] for row in read_json(REGISTRATION)["instances"]
    ]
    ledgers = {
        arm: client_rows(root, arm) for arm in (GENERAL, CANDIDATE)
    }
    rows = []
    for transition, ids in (
        ("damage", transitions["damage"]),
        ("rescue", transitions["rescue"]),
    ):
        for instance_id in ids:
            model_number = instances.index(instance_id) + 1
            per_arm_ledger = {
                arm: [
                    row
                    for row in ledgers[arm]
                    if row["model_instance_nonce"].endswith(
                        f"-m{model_number}"
                    )
                ]
                for arm in (GENERAL, CANDIDATE)
            }
            messages = {}
            trajectory_values = {}
            trajectory_paths = {}
            for arm in (GENERAL, CANDIDATE):
                (
                    messages[arm],
                    trajectory_values[arm],
                    trajectory_paths[arm],
                ) = assistant_messages(root, arm, instance_id)
            first_divergence = next(
                (
                    index
                    for index, (general, candidate) in enumerate(
                        zip(messages[GENERAL], messages[CANDIDATE]), start=1
                    )
                    if assistant_key(general) != assistant_key(candidate)
                ),
                min(len(messages[GENERAL]), len(messages[CANDIDATE])) + 1,
            )
            treatment_at_divergence = {
                arm: (
                    per_arm_ledger[arm][first_divergence - 1]
                    if first_divergence <= len(per_arm_ledger[arm])
                    else None
                )
                for arm in (GENERAL, CANDIDATE)
            }
            first_copy = {
                arm: next(
                    (
                        row["request_index"]
                        for row in per_arm_ledger[arm]
                        if row["target_registered"]
                        and row["copied_tokens_planned"] > 0
                    ),
                    None,
                )
                for arm in (GENERAL, CANDIDATE)
            }
            rows.append(
                {
                    "instance_id": instance_id,
                    "transition": transition,
                    "first_assistant_divergence_request": first_divergence,
                    "first_physical_copy_request": first_copy,
                    "divergence_treatment": {
                        arm: (
                            {
                                "target_registered": value[
                                    "target_registered"
                                ],
                                "copied_tokens_planned": value[
                                    "copied_tokens_planned"
                                ],
                                "mode": value["reuse_policy_decision"]["mode"],
                            }
                            if value is not None
                            else None
                        )
                        for arm, value in treatment_at_divergence.items()
                    },
                    "diverged_before_any_copy": all(
                        value is not None
                        and not value["target_registered"]
                        and value["copied_tokens_planned"] == 0
                        for value in treatment_at_divergence.values()
                    ),
                    "general_exit": trajectory_values[GENERAL]["info"][
                        "exit_status"
                    ],
                    "candidate_exit": trajectory_values[CANDIDATE]["info"][
                        "exit_status"
                    ],
                    "trajectory_sha256": {
                        arm: sha256(trajectory_paths[arm])
                        for arm in (GENERAL, CANDIDATE)
                    },
                }
            )
    all_pre_copy = all(row["diverged_before_any_copy"] for row in rows)
    return {
        "status": "V23_FULL18_PRE_COPY_DIVERGENCE_AUDIT_COMPLETE",
        "completed_at_utc": utc_now(),
        "classification": "retrospective_causal_localization",
        "input": {
            "path": str(result_path),
            "sha256": sha256(result_path),
        },
        "transition_tasks": rows,
        "all_transition_tasks_diverged_before_any_copy": all_pre_copy,
        "finding": (
            "Every damage/rescue task diverged while both arms were still on "
            "an unregistered dense request with zero planned copy. The "
            "observed full18 score transition therefore cannot be attributed "
            "to the V23 KV treatment, although the frozen promotion gate "
            "correctly remains failed."
        ),
        "next_protocol": (
            "Share one agent/repository prefix, clone conversation and "
            "container state immediately before the first eligible target, "
            "and branch General versus V23 only at KV treatment. Evaluate "
            "both final patches with the official harness."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = audit(args.input.resolve())
    output = args.output or args.input / "V23_PRE_COPY_DIVERGENCE_AUDIT.json"
    write_json(output, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
