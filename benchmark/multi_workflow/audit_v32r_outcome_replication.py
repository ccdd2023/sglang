#!/usr/bin/env python3
"""Audit V32R task accuracy, Dense preservation, reach, and latency."""

from __future__ import annotations

import argparse
import hashlib
import statistics
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_ROOT = (
    ARTIFACTS / "impactkv_v32r_stream_close_replication_20260727"
)
ARMS = ("coding_critical_event_abstain_v31", "general", "dense")


def _patch_hash(path: Path, instance_id: str) -> str:
    patch = read_json(path)[instance_id]["model_patch"]
    return hashlib.sha256(patch.encode()).hexdigest()


def audit(root: Path) -> dict[str, Any]:
    registration = read_json(root / "V32R_REGISTRATION.json")
    rows = []
    for selected in registration["selection"]["selected"]:
        instance_id = selected["instance_id"]
        task = root / "tasks" / instance_id
        runtime = read_json(task / "V25_RESULT.json")
        official = read_json(task / "V25_OFFICIAL_RESULT.json")
        resolved = {
            arm: int(official["arms"][arm]["resolved"]) for arm in ARMS
        }
        patch_hashes = {
            arm: _patch_hash(task / arm / "preds.json", instance_id)
            for arm in ARMS
        }
        rows.append(
            {
                "instance_id": instance_id,
                "runtime_status": runtime["status"],
                "branched": runtime["branch"] is not None,
                "shared_calls": (
                    runtime["branch"]["shared_calls"]
                    if runtime["branch"] is not None
                    else runtime["calls"]["dense"]
                ),
                "resolved": resolved,
                "patch_hashes": patch_hashes,
                "distinct_final_patches": len(set(patch_hashes.values())),
                "candidate_critical_abstentions": runtime["server"][
                    "candidate_critical_abstentions"
                ],
                "copy_counts": runtime["server"]["copy_counts"],
                "target_fallbacks": runtime["server"]["target_fallbacks"],
                "median_ttft_ms": {
                    arm: official["arms"][arm]["median_ttft_ms"]
                    for arm in ARMS
                },
            }
        )

    accuracy = {
        arm: {
            "resolved": sum(row["resolved"][arm] for row in rows),
            "total": len(rows),
            "rate": sum(row["resolved"][arm] for row in rows) / len(rows),
        }
        for arm in ARMS
    }
    dense_pass = [row for row in rows if row["resolved"]["dense"]]
    dense_fail = [row for row in rows if not row["resolved"]["dense"]]
    preservation = {
        arm: {
            "dense_pass_denominator": len(dense_pass),
            "damage": sum(not row["resolved"][arm] for row in dense_pass),
            "dense_fail_denominator": len(dense_fail),
            "rescue": sum(row["resolved"][arm] for row in dense_fail),
        }
        for arm in ARMS[:-1]
    }
    branched = [row for row in rows if row["branched"]]
    changed = [
        row
        for row in branched
        if len(set(row["patch_hashes"].values())) > 1
    ]
    ttft_ratios = [
        row["median_ttft_ms"][ARMS[0]]
        / row["median_ttft_ms"]["general"]
        for row in branched
        if row["median_ttft_ms"][ARMS[0]] is not None
        and row["median_ttft_ms"]["general"] is not None
    ]
    gates = {
        "all_children_completed": len(rows) == 3
        and all(row["runtime_status"] == "PASS" for row in rows),
        "target_fallbacks_zero": all(
            row["target_fallbacks"] == 0 for row in rows
        ),
        "v31_resolved_not_below_general": (
            accuracy[ARMS[0]]["resolved"]
            >= accuracy["general"]["resolved"]
        ),
        "v31_damage_not_above_general": (
            preservation[ARMS[0]]["damage"]
            <= preservation["general"]["damage"]
        ),
        "strategy_reach_at_least_two_tasks": len(branched) >= 2,
        "strategy_changes_at_least_one_final_patch": len(changed) >= 1,
    }
    value = {
        "audited_at_utc": utc_now(),
        "status": (
            "FAIL_NO_ACCURACY_SEPARATION_STRATEGY_REACH_LOW"
            if not all(gates.values())
            else "PASS_DIAGNOSTIC_REPLICATION"
        ),
        "registration": str(root / "V32R_REGISTRATION.json"),
        "registration_sha256": sha256(root / "V32R_REGISTRATION.json"),
        "tasks": rows,
        "overall_accuracy": accuracy,
        "dense_preservation_and_rescue": preservation,
        "strategy_reach": {
            "branched_tasks": len(branched),
            "total_tasks": len(rows),
            "rate": len(branched) / len(rows),
            "final_patch_changed_tasks": len(changed),
        },
        "fixed_order_latency_diagnostic": {
            "v31_over_general_ttft_ratios": ttft_ratios,
            "median_ratio": (
                statistics.median(ttft_ratios) if ttft_ratios else None
            ),
            "unbiased_speed_claim_allowed": False,
        },
        "gates": gates,
        "interpretation": (
            "V31, General, and Dense all resolve 1/3. V31 causes no Dense "
            "damage but also no rescue. Only one task reaches a V31 branch, "
            "and all three arms reconverge to the same final patch. This "
            "sample validates infrastructure and shows insufficient policy "
            "reach; it does not validate an accuracy advantage."
        ),
        "next_motivation": (
            "Move the coding-aware decision earlier than post-mutation/test "
            "events and preregister a reach gate before another accuracy "
            "campaign. The next candidate must affect requests while the "
            "agent is still choosing localization and edit strategy."
        ),
    }
    write_json(root / "V32R_AUDIT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    value = audit(args.root)
    print(
        {
            "status": value["status"],
            "overall_accuracy": value["overall_accuracy"],
            "strategy_reach": value["strategy_reach"],
            "gates": value["gates"],
        }
    )


if __name__ == "__main__":
    main()
