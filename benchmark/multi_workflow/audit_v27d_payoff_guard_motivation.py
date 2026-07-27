#!/usr/bin/env python3
"""Retrospective V27D audit motivating the pre-registered V28 selector."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_bridge_reuse_agent_experiment import (
    load_jsonl,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
V27C = ARTIFACTS / "impactkv_v27c_dense_pass_triple_20260727"
V27D = ARTIFACTS / "impactkv_v27d_dense_pass_audited_completion_20260727"
DEFAULT_OUTPUT = V27D / "V28_MOTIVATION_AUDIT.json"
V23 = "coding_post_mutation_target_prefix_v23"
GENERAL = "general"
DENSE = "dense"
PAYOFF_RATIO = 0.60
MIN_FUTURE_TARGETS = 4
STEP_LIMIT = 20


def _task_root(row: dict[str, Any]) -> Path:
    return Path(row["source_campaign"]) / "tasks" / row["instance_id"]


def _arm_metrics(
    root: Path, arm: str, runtime: dict[str, Any]
) -> dict[str, Any]:
    server = load_jsonl(root / "run/SERVER_LEDGER.jsonl")
    copies = [
        row
        for row in server
        if row.get("event") == "target_copied"
        and row.get("policy_label") == arm
    ]
    client = [
        row
        for row in load_jsonl(root / arm / "CLIENT_LEDGER.jsonl")
        if row.get("event") == "request_complete"
    ]
    return {
        "requests": len(client),
        "copies": len(copies),
        "copied_tokens": sum(
            int(row["copied_k_tokens"]) for row in copies
        ),
        "ordinary_prefix_tokens": sum(
            int(row["ordinary_prefix_tokens"]) for row in copies
        ),
        "effective_dense_tokens": sum(
            int(row["effective_dense_tokens"]) for row in copies
        ),
        "first_ordinary_prefix_tokens": (
            int(copies[0]["ordinary_prefix_tokens"]) if copies else 0
        ),
        "prompt_tokens": sum(
            int(row.get("prompt_tokens", 0)) for row in client
        ),
        "model_elapsed_seconds": sum(
            float(row.get("request_elapsed_seconds", 0.0))
            for row in client
        ),
        "agent_elapsed_seconds": float(
            runtime["branched_agent_elapsed_seconds"][arm]
        ),
    }


def build() -> dict[str, Any]:
    result = read_json(V27D / "V27D_RESULT.json")
    rows = []
    for task in result["valid_official_tasks"]:
        root = _task_root(task)
        runtime = read_json(root / "V25_RESULT.json")
        metrics = {
            arm: _arm_metrics(root, arm, runtime)
            for arm in (V23, GENERAL)
        }
        branch_call = int(runtime["branch"]["shared_calls"])
        future_targets = STEP_LIMIT - branch_call
        source_lengths = runtime["branch"]["source_lengths"]
        v23_source = int(source_lengths[V23])
        general_source = int(source_lengths[GENERAL])
        first_prefix = metrics[V23]["first_ordinary_prefix_tokens"]
        payoff_ratio = (v23_source + first_prefix) / general_source
        if future_targets < MIN_FUTURE_TARGETS:
            proposed_mode = "dense_abstain_late_branch"
        elif payoff_ratio >= PAYOFF_RATIO:
            proposed_mode = "coding_post_mutation_protected"
        else:
            proposed_mode = "general_middle_plus_exact_target_prefix"
        rows.append(
            {
                "instance_id": task["instance_id"],
                "resolved": task["resolved"],
                "runtime_status": task["runtime_status"],
                "branch_shared_calls": branch_call,
                "future_target_upper_bound": future_targets,
                "initial_source_lengths": source_lengths,
                "first_v23_ordinary_prefix_tokens": first_prefix,
                "initial_payoff_ratio": payoff_ratio,
                "v28_proposed_mode": proposed_mode,
                "metrics": metrics,
                "v23_minus_general_agent_seconds": (
                    metrics[V23]["agent_elapsed_seconds"]
                    - metrics[GENERAL]["agent_elapsed_seconds"]
                ),
            }
        )

    valid = len(rows)
    v23_resolved = sum(row["resolved"][V23] for row in rows)
    general_resolved = sum(row["resolved"][GENERAL] for row in rows)
    dense_resolved = sum(row["resolved"][DENSE] for row in rows)
    modes = {
        mode: sum(row["v28_proposed_mode"] == mode for row in rows)
        for mode in (
            "coding_post_mutation_protected",
            "general_middle_plus_exact_target_prefix",
            "dense_abstain_late_branch",
        )
    }
    v23_agent = [
        row["metrics"][V23]["agent_elapsed_seconds"] for row in rows
    ]
    general_agent = [
        row["metrics"][GENERAL]["agent_elapsed_seconds"] for row in rows
    ]
    value = {
        "created_at_utc": utc_now(),
        "classification": (
            "RETROSPECTIVE_MOTIVATION_ONLY_NOT_PREREGISTERED"
        ),
        "question": (
            "Why did V23 fail to beat General despite using code-version "
            "information, and what minimal selector change is justified?"
        ),
        "inputs": {
            "v27d_result": str(V27D / "V27D_RESULT.json"),
            "v27d_result_sha256": sha256(V27D / "V27D_RESULT.json"),
            "valid_task_artifacts": {
                row["instance_id"]: {
                    "runtime_sha256": sha256(
                        _task_root(row) / "V25_RESULT.json"
                    ),
                    "official_sha256": sha256(
                        _task_root(row) / "V25_OFFICIAL_RESULT.json"
                    ),
                    "server_ledger_sha256": sha256(
                        _task_root(row) / "run/SERVER_LEDGER.jsonl"
                    ),
                }
                for row in result["valid_official_tasks"]
            },
        },
        "task_rows": rows,
        "aggregate": {
            "valid_tasks": valid,
            "resolved": {
                V23: v23_resolved,
                GENERAL: general_resolved,
                DENSE: dense_resolved,
            },
            "accuracy_difference_v23_minus_general": (
                (v23_resolved - general_resolved) / valid
            ),
            "mean_agent_elapsed_seconds": {
                V23: statistics.fmean(v23_agent),
                GENERAL: statistics.fmean(general_agent),
            },
            "mean_agent_elapsed_ratio_v23_over_general": (
                statistics.fmean(v23_agent)
                / statistics.fmean(general_agent)
            ),
            "proposed_mode_counts": modes,
        },
        "findings": [
            (
                "V23, General, and concurrent Dense resolve the same 3/5 "
                "valid tasks; V23 has no measured accuracy advantage here."
            ),
            (
                "V23 mean branched agent time is higher than General. The "
                "fixed order prevents a fair speed claim, but it provides no "
                "motivation to keep low-payoff protection unconditionally."
            ),
            (
                "On low-payoff tasks V23 recomputes substantially more dense "
                "tokens than General without an observed correctness gain."
            ),
            (
                "A branch with fewer than four possible future targets has "
                "too little amortization opportunity and should abstain."
            ),
        ],
        "v28_frozen_proposal_for_separate_preregistration": {
            "name": "coding_post_mutation_payoff_guard_v28",
            "step_limit": STEP_LIMIT,
            "minimum_future_target_upper_bound": MIN_FUTURE_TARGETS,
            "payoff_ratio_threshold": PAYOFF_RATIO,
            "payoff_ratio": (
                "(coding middle tokens + exact target-prefix tokens) / "
                "General middle tokens, evaluated at the first target"
            ),
            "rules": [
                (
                    "If fewer than four future targets remain, perform Dense "
                    "and register no lossy source."
                ),
                (
                    "If payoff ratio is at least 0.60, keep V23's "
                    "post-mutation code protection."
                ),
                (
                    "Otherwise use General's middle span plus the exact "
                    "target-only ordinary prefix."
                ),
            ],
            "predicted_task_modes_on_motivation_set": modes,
            "no_prefetch": True,
            "no_reference_patch_or_future_output": True,
            "required_next_evidence": [
                "mechanical replay with zero fallback",
                "cache-ready and N=4 speed against General",
                "same-history triple-control official accuracy",
                "no promotion from this retrospective audit",
            ],
        },
    }
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    write_json(args.output, value)
    print(
        {
            "classification": value["classification"],
            "output": str(args.output),
            "aggregate": value["aggregate"],
        }
    )


if __name__ == "__main__":
    main()
