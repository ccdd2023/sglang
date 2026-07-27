#!/usr/bin/env python3
"""Validate target-scoped ordinary-prefix reuse plus the coding KV island."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import benchmark.multi_workflow.run_v21_robust_dual_replay as replay
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    INSTANCE_IDS,
    read_json,
    sha256,
    simulate_arm,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v23_target_prefix_replay_20260727"
V22_RESULT = (
    ARTIFACTS
    / "impactkv_v22_seam32_replay_20260727"
    / "V22_REPLAY_RESULT.json"
)
V22_AUDIT = (
    ARTIFACTS
    / "impactkv_v22_seam32_replay_20260727"
    / "V22_FAILURE_AUDIT.json"
)
PROJECT = Path(__file__).resolve().parents[2]
CANDIDATE = "coding_post_mutation_target_prefix_v23"
ARMS = ("dense", "general", CANDIDATE)
ORDERS = {
    1: ARMS,
    2: (ARMS[1], ARMS[2], ARMS[0]),
    3: (ARMS[2], ARMS[0], ARMS[1]),
}
PORTS = {
    (repeat, arm): 32600 + 10 * repeat + index
    for repeat, order in ORDERS.items()
    for index, arm in enumerate(order)
}


def configure_shared_runner() -> None:
    replay.ARMS = ARMS
    replay.CANDIDATE = CANDIDATE
    replay.ORDERS = ORDERS
    replay.PORTS = PORTS
    replay.register = register


def register(output: Path) -> dict[str, Any]:
    path = output / "V23_REPLAY_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    plans = {arm: simulate_arm(arm) for arm in ARMS}
    identities = [
        (row["instance_id"], row["request_index"], row["prompt_hash"])
        for row in plans["dense"]
    ]
    for arm in ARMS[1:]:
        if identities != [
            (row["instance_id"], row["request_index"], row["prompt_hash"])
            for row in plans[arm]
        ]:
            raise ValueError(f"{arm}: prompt identities differ")
    value = {
        "registered_at_utc": utc_now(),
        "registered_before_gpu": True,
        "experiment": "V23 target-scoped ordinary-prefix reuse",
        "status": "REGISTERED_BEFORE_V23_GPU",
        "motivation": (
            "V22 changed first tokens on an unregistered request and failed "
            "to alter its purported seam. Global ordinary Radix reuse was "
            "therefore a confound: it affected unregistered requests and "
            "source-building requests outside the coding selector."
        ),
        "candidate": {
            "arm": CANDIDATE,
            "selection": "V20 post-mutation island; no latest-risk guard",
            "ordinary_prefix_scope": "registered targets only",
            "source_building_requests": "dense",
            "unregistered_requests": "dense",
            "prefix_repair_tokens": 0,
            "sweep_performed": False,
        },
        "protocol": {
            "arms": list(ARMS),
            "orders": ORDERS,
            "ports": {
                f"{repeat}:{arm}": port
                for (repeat, arm), port in PORTS.items()
            },
            "repeats": 3,
            "requests_per_arm_per_repeat": len(identities),
            "instances": list(INSTANCE_IDS),
            "temperature": 0,
            "diagnostic_new_tokens": 1,
            "diagnostic_output_fed_forward": False,
            "copy_cap_tokens": 4096,
            "prefetch": False,
            "bootstrap_seed": replay.BOOTSTRAP_SEED,
            "bootstrap_iterations": replay.BOOTSTRAPS,
        },
        "frozen_gates": {
            "candidate_copy_events_each_repeat": 39,
            "candidate_fallbacks_each_repeat_max": 0,
            "candidate_positive_target_prefix_hits_each_repeat": 39,
            "candidate_effective_dense_tokens_below_raw_recomputed": True,
            "pooled_first_token_agreement_not_below_general": True,
            "candidate_js_below_general_each_repeat": True,
            "median_run_cache_ready_ratio_vs_general_max": 1.0,
            "median_run_n4_ratio_vs_general_max": 1.0,
            "changed_span_js_reduction_each_repeat_min": 0.20,
        },
        "plans": plans,
        "inputs": {
            "v22_result_path": str(V22_RESULT),
            "v22_result_sha256": sha256(V22_RESULT),
            "v22_audit_path": str(V22_AUDIT),
            "v22_audit_sha256": sha256(V22_AUDIT),
            "source_sha256": {
                str(source.relative_to(PROJECT)): sha256(source)
                for source in (
                    PROJECT / "benchmark/multi_workflow/coding_reuse_policy.py",
                    PROJECT
                    / "benchmark/multi_workflow/bridge_reuse_litellm_model.py",
                    PROJECT
                    / "benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py",
                    PROJECT / "python/sglang/srt/mem_cache/kvcomm_exact.py",
                    Path(__file__),
                )
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


def summarize(output: Path) -> dict[str, Any]:
    configure_shared_runner()
    registration = register(output)
    repeats = [replay.summarize_repeat(output, repeat) for repeat in ORDERS]
    gates = registration["frozen_gates"]
    cache_ratios = [
        row["candidate_over_general_cache_ready"] for row in repeats
    ]
    n4_ratios = [row["candidate_over_general_n4"] for row in repeats]
    pooled_matches = {
        arm: sum(
            row["fidelity"][arm]["first_token_matches"] for row in repeats
        )
        for arm in ARMS[1:]
    }
    outcomes = {
        "mechanism": all(
            row["mechanism"][CANDIDATE]["copy_events"]
            == gates["candidate_copy_events_each_repeat"]
            and row["mechanism"][CANDIDATE]["fallbacks"]
            <= gates["candidate_fallbacks_each_repeat_max"]
            for row in repeats
        ),
        "target_prefix_mechanism": all(
            row["candidate_prefix_telemetry"]["positive_prefix_hits"]
            == gates["candidate_positive_target_prefix_hits_each_repeat"]
            and row["candidate_prefix_telemetry"][
                "median_effective_dense_tokens"
            ]
            < row["candidate_prefix_telemetry"][
                "median_raw_recomputed_tokens"
            ]
            for row in repeats
        ),
        "pooled_first_token": (
            pooled_matches[CANDIDATE] >= pooled_matches["general"]
        ),
        "js": all(
            row["fidelity"][CANDIDATE]["mean_top20_plus_residual_js"]
            < row["fidelity"]["general"]["mean_top20_plus_residual_js"]
            for row in repeats
        ),
        "cache_ready": statistics.median(cache_ratios)
        <= gates["median_run_cache_ready_ratio_vs_general_max"],
        "n4": statistics.median(n4_ratios)
        <= gates["median_run_n4_ratio_vs_general_max"],
        "changed_span_js": all(
            row["changed_span_js_reduction_vs_general"]
            >= gates["changed_span_js_reduction_each_repeat_min"]
            for row in repeats
        ),
    }
    value = {
        "status": "V23_REPLAY_COMPLETE",
        "completed_at_utc": utc_now(),
        "candidate": CANDIDATE,
        "repeats": repeats,
        "aggregate": {
            "cache_ready_ratios": cache_ratios,
            "median_cache_ready_ratio": statistics.median(cache_ratios),
            "mean_cache_ready_ratio_bootstrap95": replay.bootstrap_mean_ci(
                cache_ratios
            ),
            "n4_ratios": n4_ratios,
            "median_n4_ratio": statistics.median(n4_ratios),
            "mean_n4_ratio_bootstrap95": replay.bootstrap_mean_ci(n4_ratios),
            "pooled_first_token_matches": pooled_matches,
            "pooled_requests_per_reuse_arm": 180,
        },
        "gate_outcomes": outcomes,
        "promoted_to_development_accuracy": all(outcomes.values()),
        "scope": (
            "same-prompt mechanism/speed/first-token fidelity only; no task "
            "accuracy claim"
        ),
        "prefetch": False,
    }
    write_json(output / "V23_REPLAY_RESULT.json", value)
    return value


def main() -> None:
    configure_shared_runner()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--repeat", type=int, choices=ORDERS)
    run_parser.add_argument("--arm", choices=ARMS)
    sub.add_parser("summarize")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "register":
        value = register(output)
    elif args.command == "run":
        repeats = (args.repeat,) if args.repeat else tuple(ORDERS)
        completed = {}
        for repeat in repeats:
            arms = (args.arm,) if args.arm else ORDERS[repeat]
            for arm in arms:
                result = replay.run_arm(output, repeat, arm)
                completed[f"{repeat}:{arm}"] = {
                    key: result[key] for key in ("arm", "repeat", "requests")
                }
        if all(
            (
                output / f"repeat_{repeat}" / arm / "REPLAY_RESULTS.json"
            ).exists()
            for repeat in ORDERS
            for arm in ARMS
        ):
            completed["summary"] = summarize(output)
        value = completed
    else:
        value = summarize(output)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
