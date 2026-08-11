#!/usr/bin/env python3
"""Post-hoc validity and selector audit for the frozen dependency-cold fresh8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CAMPAIGN = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_dependency_cold_fresh8_20260810"
)
POLICY = "coding_dependency_cold_cost"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def trajectory_row(root: Path, instance_id: str) -> dict[str, Any]:
    paths = list((root / instance_id).glob("*.traj.json"))
    if len(paths) != 1:
        raise ValueError(f"expected one trajectory for {instance_id}: {paths}")
    value = read_json(paths[0])
    info = value.get("info") or {}
    submission = str(info.get("submission") or value.get("submission") or "")
    return {
        "exit_status": info.get("exit_status") or value.get("exit_status"),
        "submission_characters": len(submission),
        "trajectory_messages": len(value.get("messages") or []),
    }


def audit(campaign: Path) -> dict[str, Any]:
    registration = read_json(campaign / "CAMPAIGN_REGISTRATION.json")
    result = read_json(campaign / "RESULT.json")
    identifiers = [
        row["instance_id"] for row in registration["selection"]["instances"]
    ]
    dense_official = result["official_evaluator"]["dense"]
    policy_official = result["official_evaluator"][POLICY]
    dense_resolved = set(dense_official["resolved_ids"])
    policy_resolved = set(policy_official["resolved_ids"])
    dense_completed = set(dense_official["completed_ids"])
    policy_completed = set(policy_official["completed_ids"])

    roots = {
        arm: campaign / "online" / arm / "full_8"
        for arm in ("dense", POLICY)
    }
    per_task = []
    for instance_id in identifiers:
        per_task.append(
            {
                "instance_id": instance_id,
                "dense": {
                    **trajectory_row(roots["dense"], instance_id),
                    "official_completed": instance_id in dense_completed,
                    "official_resolved": instance_id in dense_resolved,
                },
                POLICY: {
                    **trajectory_row(roots[POLICY], instance_id),
                    "official_completed": instance_id in policy_completed,
                    "official_resolved": instance_id in policy_resolved,
                },
            }
        )

    rows = [
        json.loads(line)
        for line in (roots[POLICY] / "CLIENT_LEDGER.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    decisions = [row.get("reuse_policy_decision") or {} for row in rows]
    dense_ttft = float(
        result["free_running_latency_descriptive_only"]["dense"][
            "median_ttft_ms"
        ]
    )
    policy_ttft = float(
        result["free_running_latency_descriptive_only"][POLICY][
            "median_ttft_ms"
        ]
    )
    value = {
        "classification": "post-hoc validity and selector audit",
        "frozen_direction_decision": result["decision"],
        "posthoc_interpretation": "PROMISING_DIRECTION_PROTOCOL_LIMITED",
        "accuracy": {
            "denominator": len(identifiers),
            "dense_resolved": len(dense_resolved),
            "policy_resolved": len(policy_resolved),
            "dense_nonempty_completed": len(dense_completed),
            "policy_nonempty_completed": len(policy_completed),
            "rescues": sorted(policy_resolved - dense_resolved),
            "damages": sorted(dense_resolved - policy_resolved),
            "paired_accuracy_preservation_estimable": bool(dense_completed),
            "reason": (
                "Dense produced no non-empty patch, so fresh8 has a positive "
                "directional rescue but cannot estimate preservation among "
                "Dense-correct or even Dense-completed tasks."
            ),
        },
        "physical_selector": {
            "requests": len(rows),
            "target_registered_requests": sum(
                bool(row.get("target_registered")) for row in rows
            ),
            "dependency_cold_observation_decisions": sum(
                int(row.get("dependency_cold_observations") or 0)
                for row in decisions
            ),
            "dependency_hot_observation_protections": sum(
                int(row.get("dependency_hot_observations_protected") or 0)
                for row in decisions
            ),
            "eligible_before_dependency_guard": sum(
                int(row.get("eligible_observations_before_dependency_guard") or 0)
                for row in decisions
            ),
            "planned_copied_tokens": sum(
                int(row.get("copied_tokens_planned") or 0) for row in rows
            ),
            **result["physical_reuse"],
        },
        "latency": {
            "dense_free_running_median_ttft_ms": dense_ttft,
            "policy_free_running_median_ttft_ms": policy_ttft,
            "descriptive_median_ttft_saving_percent": (
                100.0 * (dense_ttft - policy_ttft) / dense_ttft
            ),
            "causal_claim_allowed": False,
            "reason": (
                "the two free-running agent arms issued different request "
                "counts and prompt trajectories"
            ),
        },
        "per_task": per_task,
        "claim_boundary": [
            "Fresh8 supports a promising accuracy direction (1/8 versus 0/8).",
            "It does not prove accuracy non-inferiority because Dense had no completed patch.",
            "Its free-running TTFT reduction is descriptive, not a paired causal speedup.",
            "Causal speed evidence must come from equal-prompt same-history forks.",
        ],
    }
    write_json(campaign / "POSTHOC_AUDIT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    args = parser.parse_args()
    print(json.dumps(audit(args.campaign.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
