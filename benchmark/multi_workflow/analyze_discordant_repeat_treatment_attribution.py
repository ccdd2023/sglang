#!/usr/bin/env python3
"""Audit whether discordant7 outcomes can be attributed to KV treatment.

For each task, this script locates the first physically registered target in
the policy ledger and compares the generated assistant/tool interaction prefix
against the matched Dense trajectory.  A rescue is called causally clean only
when treatment occurred and every completed interaction before the first
treatment is byte-identical after removing request-local tool-call IDs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_ROOT = (
    ARTIFACTS / "impactkv_natural_code_cost_discordant7_repeat_20260809"
)
POLICY = "coding_natural_code_cost"
NONCE_RE = re.compile(r"-m(\d+)$")
CASE_NONCE_RE = re.compile(r"^(p\d+-m\d+)-")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def canonical_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for call in message.get("tool_calls") or ():
        function = call.get("function") or {}
        raw = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            arguments = raw
        output.append(
            {"name": function.get("name"), "arguments": arguments}
        )
    return output


def trajectory_interactions(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return generated interaction groups without request-local IDs."""

    output: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") != "assistant":
            index += 1
            continue
        group: dict[str, Any] = {
            "assistant_content": message.get("content") or "",
            "tool_calls": canonical_tool_calls(message),
            "tool_results": [],
        }
        index += 1
        while index < len(messages) and messages[index].get("role") == "tool":
            group["tool_results"].append(messages[index].get("content") or "")
            index += 1
        output.append(group)
    return output


def action_signature(interaction: dict[str, Any]) -> Any:
    calls = interaction["tool_calls"]
    if calls:
        return calls
    return {"final": interaction["assistant_content"]}


def common_prefix_length(left: list[Any], right: list[Any]) -> int:
    count = 0
    for first, second in zip(left, right):
        if first != second:
            break
        count += 1
    return count


def _ordered_nonce_groups(
    rows: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model_instance_nonce"])].append(row)

    def nonce_index(item: tuple[str, list[dict[str, Any]]]) -> int:
        match = NONCE_RE.search(item[0])
        if match is None:
            raise ValueError(f"malformed model nonce: {item[0]}")
        return int(match.group(1))

    return sorted(grouped.items(), key=nonce_index)


def analyze(root: Path) -> dict[str, Any]:
    registration = read_json(root / "CAMPAIGN_REGISTRATION.json")
    result = read_json(root / "RESULT.json")
    instances = [
        str(row["instance_id"])
        for row in registration["selection"]["instances"]
    ]
    policy_dir = root / f"online/{POLICY}/full_{len(instances)}"
    dense_dir = root / f"online/dense/full_{len(instances)}"
    client_groups = _ordered_nonce_groups(
        read_jsonl(policy_dir / "CLIENT_LEDGER.jsonl")
    )
    if len(client_groups) != len(instances):
        raise AssertionError("policy ledger/task count mismatch")

    copied_by_nonce: dict[str, dict[str, int]] = defaultdict(
        lambda: {"copy_events": 0, "copied_tokens": 0}
    )
    for row in read_jsonl(policy_dir / "SERVER_LEDGER.jsonl"):
        if row.get("event") != "target_copied":
            continue
        match = CASE_NONCE_RE.match(str(row.get("case_id") or ""))
        if match is None:
            raise ValueError("target copy lacks model nonce")
        nonce = match.group(1)
        copied_by_nonce[nonce]["copy_events"] += 1
        copied_by_nonce[nonce]["copied_tokens"] += int(
            row.get("copied_k_tokens", 0)
        )

    repeat_rows = {
        row["instance_id"]: row
        for row in result["stability"]["per_task"]
    }
    per_task = []
    for instance_id, (nonce, client_rows) in zip(
        instances, client_groups, strict=True
    ):
        policy_traj = read_json(
            policy_dir / instance_id / f"{instance_id}.traj.json"
        )
        dense_traj = read_json(
            dense_dir / instance_id / f"{instance_id}.traj.json"
        )
        policy_messages = policy_traj["messages"]
        dense_messages = dense_traj["messages"]
        if policy_messages[:2] != dense_messages[:2]:
            raise AssertionError(f"initial prompt differs: {instance_id}")
        policy_interactions = trajectory_interactions(policy_messages)
        dense_interactions = trajectory_interactions(dense_messages)
        common_interactions = common_prefix_length(
            policy_interactions, dense_interactions
        )
        common_actions = common_prefix_length(
            [action_signature(row) for row in policy_interactions],
            [action_signature(row) for row in dense_interactions],
        )
        first_target = next(
            (
                int(row["request_index"])
                for row in client_rows
                if row.get("target_registered")
            ),
            None,
        )
        physical = copied_by_nonce.get(
            nonce, {"copy_events": 0, "copied_tokens": 0}
        )
        treated = physical["copy_events"] > 0
        if treated != (first_target is not None):
            raise AssertionError(f"client/server treatment mismatch: {instance_id}")
        pre_treatment_interactions = (
            first_target - 1 if first_target is not None else None
        )
        exact_before_treatment = bool(
            treated
            and common_interactions >= int(pre_treatment_interactions)
        )
        action_same_before_treatment = bool(
            treated and common_actions >= int(pre_treatment_interactions)
        )
        repeat_label = repeat_rows[instance_id]["repeat"]
        stable_rescue = (
            repeat_rows[instance_id]["original"] == "rescue"
            and repeat_label == "rescue"
        )
        causally_clean_rescue = bool(stable_rescue and exact_before_treatment)
        per_task.append(
            {
                "instance_id": instance_id,
                "original_label": repeat_rows[instance_id]["original"],
                "repeat_label": repeat_label,
                "stable_rescue": stable_rescue,
                "causally_clean_rescue": causally_clean_rescue,
                "model_instance_nonce": nonce,
                "policy_requests": len(policy_interactions),
                "dense_requests": len(dense_interactions),
                "first_target_request": first_target,
                "copy_events": physical["copy_events"],
                "copied_tokens": physical["copied_tokens"],
                "common_exact_interaction_prefix": common_interactions,
                "common_tool_action_prefix": common_actions,
                "exact_interactions_before_treatment": exact_before_treatment,
                "tool_actions_same_before_treatment": action_same_before_treatment,
                "first_exact_interaction_divergence_request": (
                    common_interactions + 1
                    if common_interactions
                    < min(len(policy_interactions), len(dense_interactions))
                    else None
                ),
                "first_tool_action_divergence_request": (
                    common_actions + 1
                    if common_actions
                    < min(len(policy_interactions), len(dense_interactions))
                    else None
                ),
            }
        )

    stable = [row for row in per_task if row["stable_rescue"]]
    value = {
        "status": "COMPLETE",
        "classification": "discordant7 first-treatment attribution audit",
        "definition": {
            "causally_clean_rescue": (
                "repeat rescue with at least one physical KV copy and exact "
                "assistant/tool interaction identity through every completed "
                "request before the first copied target"
            ),
            "tool_call_ids_ignored": True,
            "initial_system_user_prompt_required_equal": True,
        },
        "summary": {
            "tasks": len(per_task),
            "treated_tasks": sum(row["copy_events"] > 0 for row in per_task),
            "untreated_tasks": sum(row["copy_events"] == 0 for row in per_task),
            "treated_with_exact_pre_treatment_history": sum(
                row["exact_interactions_before_treatment"] for row in per_task
            ),
            "treated_with_same_pre_treatment_tool_actions": sum(
                row["tool_actions_same_before_treatment"] for row in per_task
            ),
            "stable_rescues": len(stable),
            "stable_rescues_without_treatment": sum(
                row["copy_events"] == 0 for row in stable
            ),
            "stable_rescues_diverged_before_treatment": sum(
                row["copy_events"] > 0
                and not row["exact_interactions_before_treatment"]
                for row in stable
            ),
            "causally_clean_rescues": sum(
                row["causally_clean_rescue"] for row in stable
            ),
            "accuracy_rescue_attribution_allowed": False,
        },
        "interpretation": (
            "The reversed-order repeat supports arm-level directional "
            "stability, but none of its stable rescues isolates lossy KV as "
            "the cause. Untreated rescues measure agent repeat variance; "
            "treated rescues already had different histories before copy."
        ),
        "per_task": per_task,
        "inputs": {
            "campaign_registration": str(root / "CAMPAIGN_REGISTRATION.json"),
            "campaign_result": str(root / "RESULT.json"),
            "policy_client_ledger": str(policy_dir / "CLIENT_LEDGER.jsonl"),
            "policy_server_ledger": str(policy_dir / "SERVER_LEDGER.jsonl"),
        },
    }
    write_json(root / "TREATMENT_ATTRIBUTION.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
