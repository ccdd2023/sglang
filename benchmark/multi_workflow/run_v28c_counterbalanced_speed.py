#!/usr/bin/env python3
"""Counterbalanced V28/General frozen-prompt speed replication."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import run_frozen_trajectory_replay_v18 as replay
from benchmark.multi_workflow.run_bridge_reuse_agent_experiment import (
    load_jsonl,
)
from benchmark.multi_workflow.run_v28b_order_replicated_speed import (
    _gpu_processes,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v28c_counterbalanced_speed_20260727"
)
PROJECT = Path(__file__).resolve().parents[2]
V28 = "coding_post_mutation_payoff_guard_v28"
GENERAL = "general"
ARMS = (V28, GENERAL)
ROUNDS = (
    ("round_1_general_then_v28", (GENERAL, V28)),
    ("round_2_v28_then_general", (V28, GENERAL)),
)
PRIOR_A = (
    ARTIFACTS
    / "impactkv_v28_payoff_guard_replay_20260727"
    / "V28_REPLAY_RESULT.json"
)
PRIOR_B = (
    ARTIFACTS
    / "impactkv_v28b_order_replicated_speed_20260727"
    / "V28B_RESULT.json"
)


def _configure() -> None:
    replay.ARMS = ARMS


def register(output: Path) -> dict[str, Any]:
    path = output / "V28C_REGISTRATION.json"
    if path.exists():
        return replay.read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _configure()
    plans = {arm: replay.simulate_arm(arm) for arm in ARMS}
    identities = {
        arm: [
            (
                row["instance_id"],
                row["request_index"],
                row["prompt_hash"],
            )
            for row in plans[arm]
        ]
        for arm in ARMS
    }
    if identities[V28] != identities[GENERAL]:
        raise AssertionError("V28C prompt identities differ")
    value = {
        "registered_at_utc": replay.utc_now(),
        "status": "REGISTERED_BEFORE_ANY_V28C_GPU_RUN",
        "experiment": "V28C two-round counterbalanced speed replication",
        "motivation": (
            "V28A common-key TTFT ratio was 1.594 and V28B reverse-order "
            "ratio was 0.984. Run one fresh General->V28 pair and one fresh "
            "V28->General pair under a single frozen registration. Require "
            "new-order consistency and a robust median across all four "
            "observed order runs before any official accuracy canary."
        ),
        "rounds": [
            {"round": name, "arm_order": list(order)}
            for name, order in ROUNDS
        ],
        "prior_results": {
            "v28a": {
                "path": str(PRIOR_A),
                "sha256": replay.sha256(PRIOR_A),
                "common_key_ratio": 1.5935730580813203,
            },
            "v28b": {
                "path": str(PRIOR_B),
                "sha256": replay.sha256(PRIOR_B),
                "common_key_ratio": float(
                    replay.read_json(PRIOR_B)[
                        "common_median_ttft_ratio_v28_over_general"
                    ]
                ),
            },
        },
        "protocol": {
            "same_frozen_prompt_ids": True,
            "diagnostic_new_tokens": 1,
            "diagnostic_output_never_enters_future_prompt": True,
            "common_key_definition": "V28 registered-target prompt keys",
            "fresh_server_each_arm": True,
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "frozen_gates": {
            "new_rounds_complete": 2,
            "prompt_hashes_identical_each_round": True,
            "target_fallbacks_each_arm": 0,
            "new_round_geomean_ratio_not_above_one": True,
            "each_new_round_ratio_max": 1.05,
            "all_four_median_ratio_not_above_one": True,
            "all_four_v28_faster_rounds_min": 3,
            "do_not_override_v28a_failure": True,
            "do_not_claim_task_accuracy": True,
        },
        "offline_plans": plans,
        "gpu_processes_at_registration": _gpu_processes(),
        "inputs": {
            "source_sha256": {
                str(source.relative_to(PROJECT)): replay.sha256(source)
                for source in (
                    PROJECT
                    / "benchmark/multi_workflow/coding_reuse_policy.py",
                    PROJECT
                    / "benchmark/multi_workflow/bridge_reuse_litellm_model.py",
                    PROJECT
                    / "benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py",
                    PROJECT
                    / "python/sglang/srt/mem_cache/kvcomm_exact.py",
                    Path(__file__),
                )
            }
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
    }
    replay.write_json(path, value)
    return value


def preregister_rounds(output: Path) -> None:
    parent = register(output)
    parent_hash = replay.sha256(output / "V28C_REGISTRATION.json")
    for name, order in ROUNDS:
        child = output / name
        path = child / "REPLAY_REGISTRATION.json"
        if path.exists():
            continue
        child.mkdir(parents=True)
        replay.write_json(
            path,
            {
                "registered_at_utc": replay.utc_now(),
                "status": "CHILD_REGISTERED_BEFORE_ANY_V28C_GPU_RUN",
                "parent_registration_sha256": parent_hash,
                "round": name,
                "arm_order": list(order),
                "instances": list(replay.INSTANCE_IDS),
                "protected": parent["protected"],
            },
        )


def _round_summary(root: Path) -> dict[str, Any]:
    rows = {
        arm: replay.read_json(
            root / arm / "REPLAY_RESULTS.json"
        )["rows"]
        for arm in ARMS
    }
    indexed = {
        arm: {
            (row["instance_id"], int(row["request_index"])): row
            for row in rows[arm]
        }
        for arm in ARMS
    }
    keys = [
        key for key, row in indexed[V28].items() if row["target_registered"]
    ]
    prompt_identity = all(
        indexed[V28][key]["prompt_hash"]
        == indexed[GENERAL][key]["prompt_hash"]
        for key in indexed[V28]
    )
    values = {
        arm: [float(indexed[arm][key]["ttft_ms"]) for key in keys]
        for arm in ARMS
    }
    ledgers = {
        arm: load_jsonl(root / arm / "SERVER_LEDGER.jsonl")
        for arm in ARMS
    }
    fallbacks = {
        arm: sum(row.get("event") == "target_fallback" for row in ledger)
        for arm, ledger in ledgers.items()
    }
    medians = {
        arm: statistics.median(values[arm]) for arm in ARMS
    }
    pair_ratios = [
        indexed[V28][key]["ttft_ms"] / indexed[GENERAL][key]["ttft_ms"]
        for key in keys
    ]
    return {
        "common_keys": len(keys),
        "prompt_hashes_identical": prompt_identity,
        "target_fallbacks": fallbacks,
        "median_common_ttft_ms": medians,
        "common_median_ratio_v28_over_general": (
            medians[V28] / medians[GENERAL]
        ),
        "median_paired_ratio_v28_over_general": (
            statistics.median(pair_ratios)
        ),
        "v28_faster_requests": sum(ratio < 1 for ratio in pair_ratios),
    }


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    rounds = {
        name: _round_summary(output / name) for name, _ in ROUNDS
    }
    new_ratios = [
        rounds[name]["common_median_ratio_v28_over_general"]
        for name, _ in ROUNDS
    ]
    all_ratios = [
        float(
            registration["prior_results"]["v28a"]["common_key_ratio"]
        ),
        float(
            registration["prior_results"]["v28b"]["common_key_ratio"]
        ),
        *new_ratios,
    ]
    gates = {
        "new_rounds_complete": len(rounds) == 2,
        "prompt_hashes_identical_each_round": all(
            row["prompt_hashes_identical"] for row in rounds.values()
        ),
        "target_fallbacks_each_arm": all(
            all(value == 0 for value in row["target_fallbacks"].values())
            for row in rounds.values()
        ),
        "new_round_geomean_ratio_not_above_one": (
            math.sqrt(math.prod(new_ratios)) <= 1
        ),
        "each_new_round_ratio_max": max(new_ratios) <= 1.05,
        "all_four_median_ratio_not_above_one": (
            statistics.median(all_ratios) <= 1
        ),
        "all_four_v28_faster_rounds_min": (
            sum(ratio < 1 for ratio in all_ratios) >= 3
        ),
        "do_not_override_v28a_failure": True,
        "do_not_claim_task_accuracy": True,
    }
    value = {
        "completed_at_utc": replay.utc_now(),
        "status": (
            "PASS_V28C_COUNTERBALANCED_SPEED"
            if all(gates.values())
            else "FAIL_V28C_COUNTERBALANCED_SPEED"
        ),
        "rounds": rounds,
        "new_round_ratios": new_ratios,
        "new_round_geomean_ratio": math.sqrt(math.prod(new_ratios)),
        "all_four_order_ratios": all_ratios,
        "all_four_median_ratio": statistics.median(all_ratios),
        "all_four_v28_faster_rounds": sum(
            ratio < 1 for ratio in all_ratios
        ),
        "gpu_processes_at_completion": _gpu_processes(),
        "gate_outcomes": gates,
        "decision": (
            "Eligible for a separately preregistered small same-history "
            "official accuracy canary; V28A remains a recorded speed failure."
            if all(gates.values())
            else "Reject V28 speed robustness; do not run official accuracy."
        ),
    }
    replay.write_json(output / "V28C_RESULT.json", value)
    return value


def run(output: Path) -> dict[str, Any]:
    preregister_rounds(output)
    _configure()
    port = 33025
    for name, order in ROUNDS:
        child = output / name
        for arm in order:
            replay.run_arm(child, arm, port)
    return summarize(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("register", "preregister", "run", "summarize"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "register":
        value = register(args.output)
    elif args.command == "preregister":
        preregister_rounds(args.output)
        value = {"status": "CHILD_ROUNDS_REGISTERED"}
    elif args.command == "run":
        value = run(args.output)
    else:
        value = summarize(args.output)
    print(
        {
            "status": value.get("status"),
            "output": str(args.output),
            "gate_outcomes": value.get("gate_outcomes"),
        }
    )


if __name__ == "__main__":
    main()
