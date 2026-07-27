#!/usr/bin/env python3
"""Robust three-run replay of Dense, General, and the V20 dual candidate."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_bridge_reuse_agent_experiment import (
    init_manifest,
    launch_server,
    load_jsonl,
    stop_server,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    INSTANCE_IDS,
    assistant_request_prefixes,
    coarse_js,
    generate_one,
    make_planner,
    plan_request,
    read_json,
    reset_planner_session,
    sha256,
    simulate_arm,
    token_id,
    top_distribution,
    trajectory_path,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v21_robust_dual_replay_20260727"
V20_RESULT = (
    ARTIFACTS
    / "impactkv_v20_dual_island_replay_20260727"
    / "V20_REPLAY_RESULT.json"
)
PROJECT = Path(__file__).resolve().parents[2]
ARMS = ("dense", "general", "coding_post_mutation_dual_v20")
CANDIDATE = ARMS[-1]
ORDERS = {
    1: ARMS,
    2: (ARMS[1], ARMS[2], ARMS[0]),
    3: (ARMS[2], ARMS[0], ARMS[1]),
}
PORTS = {
    (repeat, arm): 32400 + 10 * repeat + index
    for repeat, order in ORDERS.items()
    for index, arm in enumerate(order)
}
BOOTSTRAP_SEED = 20260727
BOOTSTRAPS = 20_000


def register(output: Path) -> dict[str, Any]:
    path = output / "V21_REPLAY_REGISTRATION.json"
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
        "experiment": "V21 robust V20 dual-island replication",
        "status": "REGISTERED_BEFORE_V21_GPU",
        "prior_result": (
            "V20 remains not promoted under its original per-repeat gates. "
            "V21 is a new robust replication and cannot retroactively change "
            "that decision."
        ),
        "motivation": (
            "V20 improved JS and cache-ready TTFT in both runs, but missed one "
            "N=4 gate by 0.98pp and one first-token gate by one request. Run "
            "three fresh Dense/General/candidate triplets with Latin ordering "
            "and corrected ordinary-prefix telemetry."
        ),
        "protocol": {
            "arms": list(ARMS),
            "orders": ORDERS,
            "ports": {
                f"{repeat}:{arm}": port
                for (repeat, arm), port in PORTS.items()
            },
            "requests_per_arm_per_repeat": len(identities),
            "repeats": 3,
            "temperature": 0,
            "diagnostic_new_tokens": 1,
            "diagnostic_output_fed_forward": False,
            "copy_cap_tokens": 4096,
            "ordinary_prefix_source": "preceding real request only",
            "prefetch": False,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_iterations": BOOTSTRAPS,
        },
        "frozen_gates": {
            "candidate_copy_events_each_repeat": 39,
            "candidate_fallbacks_each_repeat_max": 0,
            "candidate_ordinary_prefix_tokens_median_min": 1,
            "candidate_effective_dense_tokens_below_raw_recomputed": True,
            "pooled_first_token_agreement_not_below_general": True,
            "candidate_js_below_general_each_repeat": True,
            "median_run_cache_ready_ratio_vs_general_max": 1.0,
            "median_run_n4_ratio_vs_general_max": 1.0,
            "changed_span_js_reduction_each_repeat_min": 0.20,
        },
        "plans": plans,
        "inputs": {
            "v20_result_path": str(V20_RESULT),
            "v20_result_sha256": sha256(V20_RESULT),
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
            "trajectory_sha256": {
                instance_id: sha256(trajectory_path(instance_id))
                for instance_id in INSTANCE_IDS
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


def run_arm(output: Path, repeat: int, arm: str) -> dict[str, Any]:
    register(output)
    if (repeat, arm) not in PORTS:
        raise ValueError(f"unsupported repeat/arm: {repeat}/{arm}")
    run_dir = output / f"repeat_{repeat}" / arm
    result_path = run_dir / "REPLAY_RESULTS.json"
    if result_path.exists():
        return read_json(result_path)
    run_dir.mkdir(parents=True)
    manifest = init_manifest(run_dir, arm)
    planner = make_planner(
        arm=arm,
        manifest_path=manifest if arm != "dense" else None,
        client_ledger_path=(
            run_dir / "PLANNER_LEDGER.jsonl" if arm != "dense" else None
        ),
        instance_nonce=f"runtime-v21-{arm}-r{repeat}",
    )
    process, log = launch_server(
        run_dir=run_dir,
        arm=arm,
        manifest=manifest,
        port=PORTS[(repeat, arm)],
    )
    rows: list[dict[str, Any]] = []
    try:
        base_url = f"http://127.0.0.1:{PORTS[(repeat, arm)]}"
        for instance_id in INSTANCE_IDS:
            reset_planner_session(planner, instance_id=instance_id)
            messages = read_json(trajectory_path(instance_id))["messages"]
            for request_index, prefix in enumerate(
                assistant_request_prefixes(messages), start=1
            ):
                planned = plan_request(planner, prefix)
                target = planned["target"]
                key = (
                    str(target["case_id"])
                    if target
                    else f"v21-{arm}-r{repeat}-{instance_id}-q{request_index}"
                )
                generated = generate_one(
                    base_url=base_url,
                    input_ids=planned["prompt_ids"],
                    key=key,
                )
                rows.append(
                    {
                        "arm": arm,
                        "repeat": repeat,
                        "instance_id": instance_id,
                        "request_index": request_index,
                        "request_key": key,
                        "prompt_hash": planned["prompt_hash"],
                        "prompt_tokens": planned["prompt_tokens"],
                        "target_registered": target is not None,
                        "target_source_id": (
                            str(target["source_id"]) if target else None
                        ),
                        "target_length": (
                            int(target["length"]) if target else 0
                        ),
                        "source_registered": planned["source"] is not None,
                        "source_id": (
                            str(planned["source"]["source_id"])
                            if planned["source"]
                            else None
                        ),
                        "source_length": (
                            int(planned["source"]["length"])
                            if planned["source"]
                            else 0
                        ),
                        "decision": planned["decision"],
                        **generated,
                    }
                )
        if planner._pending_source is not None and arm != "dense":
            planner._atomic_sidecar_update(
                release_source_ids=[
                    str(planner._pending_source["source_id"])
                ]
            )
            planner._pending_source = None
    finally:
        stop_server(process, log)
    value = {
        "arm": arm,
        "repeat": repeat,
        "completed_at_utc": utc_now(),
        "requests": len(rows),
        "rows": rows,
    }
    write_json(result_path, value)
    return value


def _index(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row["instance_id"]), int(row["request_index"])): row
        for row in rows
    }


def _fidelity(
    keys: list[tuple[str, int]],
    dense: dict,
    rows: dict,
) -> dict[str, Any]:
    agreements = [token_id(rows[key]) == token_id(dense[key]) for key in keys]
    values = [
        coarse_js(top_distribution(dense[key]), top_distribution(rows[key]))
        for key in keys
    ]
    valid = [value for value in values if value is not None]
    return {
        "requests": len(keys),
        "first_token_matches": sum(agreements),
        "first_token_agreement": sum(agreements) / len(agreements),
        "mean_top20_plus_residual_js": statistics.fmean(valid),
    }


def summarize_repeat(output: Path, repeat: int) -> dict[str, Any]:
    rows = {
        arm: _index(
            read_json(
                output / f"repeat_{repeat}" / arm / "REPLAY_RESULTS.json"
            )["rows"]
        )
        for arm in ARMS
    }
    keys = list(rows["dense"])
    if any(set(rows[arm]) != set(keys) for arm in ARMS):
        raise ValueError("request identities differ")
    if any(
        rows[arm][key]["prompt_hash"] != rows["dense"][key]["prompt_hash"]
        for arm in ARMS
        for key in keys
    ):
        raise ValueError("prompt hashes differ")
    target_keys = [
        key for key in keys if rows[CANDIDATE][key]["target_registered"]
    ]
    changed_keys = [
        key
        for key in target_keys
        if (
            rows[CANDIDATE][key]["target_length"],
            rows[CANDIDATE][key]["source_length"],
        )
        != (
            rows["general"][key]["target_length"],
            rows["general"][key]["source_length"],
        )
    ]
    ledgers = {
        arm: load_jsonl(
            output / f"repeat_{repeat}" / arm / "SERVER_LEDGER.jsonl"
        )
        for arm in ARMS[1:]
    }
    copies = {
        arm: [
            row for row in ledgers[arm] if row.get("event") == "target_copied"
        ]
        for arm in ARMS[1:]
    }
    builds = {
        arm: {
            str(row["source_id"]): float(row["materialize_ms"])
            for row in ledgers[arm]
            if row.get("event")
            in ("source_materialized", "source_materialized_host")
        }
        for arm in ARMS[1:]
    }
    cache_ttft = {
        arm: statistics.median(
            float(rows[arm][key]["ttft_ms"]) for key in target_keys
        )
        for arm in ARMS
    }
    n4 = {}
    for arm in ARMS[1:]:
        n4[arm] = statistics.median(
            float(rows[arm][key]["ttft_ms"])
            + builds[arm][str(rows[arm][key]["target_source_id"])] / 4
            for key in target_keys
        )
    fidelity = {
        arm: _fidelity(keys, rows["dense"], rows[arm])
        for arm in ARMS[1:]
    }
    changed = {
        arm: _fidelity(changed_keys, rows["dense"], rows[arm])
        for arm in ARMS[1:]
    }
    candidate_ordinary = [
        int(row.get("ordinary_prefix_tokens", 0))
        for row in copies[CANDIDATE]
    ]
    candidate_effective = [
        int(row.get("effective_dense_tokens", row["recomputed_tokens"]))
        for row in copies[CANDIDATE]
    ]
    candidate_raw = [
        int(row["recomputed_tokens"]) for row in copies[CANDIDATE]
    ]
    return {
        "repeat": repeat,
        "target_requests": len(target_keys),
        "changed_target_requests": len(changed_keys),
        "mechanism": {
            arm: {
                "copy_events": len(copies[arm]),
                "fallbacks": sum(
                    row.get("event") == "target_fallback"
                    for row in ledgers[arm]
                ),
            }
            for arm in ARMS[1:]
        },
        "candidate_prefix_telemetry": {
            "median_ordinary_prefix_tokens": statistics.median(
                candidate_ordinary
            ),
            "positive_prefix_hits": sum(value > 0 for value in candidate_ordinary),
            "median_raw_recomputed_tokens": statistics.median(candidate_raw),
            "median_effective_dense_tokens": statistics.median(
                candidate_effective
            ),
        },
        "cache_ready_ttft_ms": cache_ttft,
        "n4_including_build_ms": n4,
        "candidate_over_general_cache_ready": (
            cache_ttft[CANDIDATE] / cache_ttft["general"]
        ),
        "candidate_over_general_n4": n4[CANDIDATE] / n4["general"],
        "fidelity": fidelity,
        "changed_span_fidelity": changed,
        "changed_span_js_reduction_vs_general": 1.0
        - changed[CANDIDATE]["mean_top20_plus_residual_js"]
        / changed["general"]["mean_top20_plus_residual_js"],
    }


def bootstrap_mean_ci(values: list[float]) -> list[float]:
    rng = random.Random(BOOTSTRAP_SEED)
    samples = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(BOOTSTRAPS)
    )
    return [
        samples[int(0.025 * BOOTSTRAPS)],
        samples[int(0.975 * BOOTSTRAPS)],
    ]


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    repeats = [summarize_repeat(output, repeat) for repeat in ORDERS]
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
        "ordinary_prefix_proven": all(
            row["candidate_prefix_telemetry"][
                "median_ordinary_prefix_tokens"
            ]
            >= gates["candidate_ordinary_prefix_tokens_median_min"]
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
        "cache_ready": (
            statistics.median(cache_ratios)
            <= gates["median_run_cache_ready_ratio_vs_general_max"]
        ),
        "n4": (
            statistics.median(n4_ratios)
            <= gates["median_run_n4_ratio_vs_general_max"]
        ),
        "changed_span_js": all(
            row["changed_span_js_reduction_vs_general"]
            >= gates["changed_span_js_reduction_each_repeat_min"]
            for row in repeats
        ),
    }
    value = {
        "status": "V21_REPLAY_COMPLETE",
        "completed_at_utc": utc_now(),
        "candidate": CANDIDATE,
        "repeats": repeats,
        "aggregate": {
            "cache_ready_ratios": cache_ratios,
            "median_cache_ready_ratio": statistics.median(cache_ratios),
            "mean_cache_ready_ratio_bootstrap95": bootstrap_mean_ci(
                cache_ratios
            ),
            "n4_ratios": n4_ratios,
            "median_n4_ratio": statistics.median(n4_ratios),
            "mean_n4_ratio_bootstrap95": bootstrap_mean_ci(n4_ratios),
            "pooled_first_token_matches": pooled_matches,
            "pooled_requests_per_reuse_arm": 3 * 60,
        },
        "gate_outcomes": outcomes,
        "promoted_to_development_accuracy": all(outcomes.values()),
        "prior_v20_decision_unchanged": "not_promoted",
        "scope": (
            "same-prompt mechanism/speed/first-token fidelity only; no task "
            "accuracy claim"
        ),
        "prefetch": False,
    }
    write_json(output / "V21_REPLAY_RESULT.json", value)
    return value


def main() -> None:
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
                result = run_arm(output, repeat, arm)
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
