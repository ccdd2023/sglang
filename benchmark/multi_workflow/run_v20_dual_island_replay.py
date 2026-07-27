#!/usr/bin/env python3
"""Test exact-prefix + post-mutation shifted KV reuse on identical prompts."""

from __future__ import annotations

import argparse
import json
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
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v20_dual_island_replay_20260727"
BASELINE_REPLAYS = (
    ARTIFACTS / "impactkv_v18_frozen_replay_20260727",
    ARTIFACTS / "impactkv_v18r_frozen_replay_replication_20260727",
)
V19_RESULT = (
    ARTIFACTS
    / "impactkv_v19_post_mutation_replay_20260727"
    / "V19_REPLAY_RESULT.json"
)
V8_COLD_RESULT = (
    ARTIFACTS
    / "impactkv_coding_dual_island_v8_cold_20260727"
    / "COLD_RESULT.json"
)
PROJECT = Path(__file__).resolve().parents[2]
CONTROL = "general_dual_4k"
CANDIDATE = "coding_post_mutation_dual_v20"
NEW_ARMS = (CONTROL, CANDIDATE)
PORTS = {
    (1, CONTROL): 32320,
    (1, CANDIDATE): 32321,
    (2, CANDIDATE): 32322,
    (2, CONTROL): 32323,
}
ORDER = {1: NEW_ARMS, 2: tuple(reversed(NEW_ARMS))}


def _identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["instance_id"], row["request_index"], row["prompt_hash"])


def _span(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["target_registered"],
        row["copied_tokens_planned"],
        row["source_registered"],
        row["source_tokens_planned"],
    )


def register(output: Path) -> dict[str, Any]:
    path = output / "V20_REPLAY_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    plans = {
        arm: simulate_arm(arm)
        for arm in ("general", CONTROL, CANDIDATE)
    }
    identities = [_identity(row) for row in plans["general"]]
    if any(
        [_identity(row) for row in plans[arm]] != identities
        for arm in NEW_ARMS
    ):
        raise ValueError("V20 changes prompt identity")
    if any(
        _span(left) != _span(right)
        for left, right in zip(plans["general"], plans[CONTROL])
    ):
        raise ValueError("dual General changes shifted span selection")
    changed = [
        {
            "instance_id": left["instance_id"],
            "request_index": left["request_index"],
            "general_span": _span(left),
            "candidate_span": _span(right),
            "candidate_decision": right["decision"],
        }
        for left, right in zip(plans["general"], plans[CANDIDATE])
        if _span(left) != _span(right)
    ]
    value = {
        "registered_at_utc": utc_now(),
        "registered_before_candidate_gpu": True,
        "experiment": "V20 exact-prefix plus post-mutation shifted island replay",
        "motivation": (
            "V19 consistently lowers JS but misses cache-ready and N=4 speed "
            "gates by 9.6-13.3%. Reuse the exact ordinary Radix prefix already "
            "created by each preceding real request to offset V19's extra "
            "dense recomputation; no synthetic build or prefetch is allowed."
        ),
        "arms": {
            CONTROL: (
                "General shifted 4K island plus lossless ordinary Radix prefix"
            ),
            CANDIDATE: (
                "V19 post-mutation shifted island plus the same lossless "
                "ordinary Radix prefix"
            ),
        },
        "protocol": {
            "repeats": 2,
            "arm_order": ORDER,
            "ports": {
                f"repeat_{repeat}_{arm}": port
                for (repeat, arm), port in PORTS.items()
            },
            "requests_per_arm": len(identities),
            "instances": list(INSTANCE_IDS),
            "temperature": 0,
            "diagnostic_new_tokens": 1,
            "diagnostic_output_fed_forward": False,
            "copy_cap_tokens": 4096,
            "ordinary_prefix_source": "preceding real request only",
            "prefetch": False,
        },
        "frozen_gates": {
            "physical_copy_events_each_arm": 39,
            "target_fallbacks_each_arm_max": 0,
            "candidate_first_token_agreement_not_below_general": True,
            "candidate_mean_js_not_above_general": True,
            "candidate_cache_ready_ttft_over_general_max": 1.0,
            "candidate_n4_including_build_over_general_max": 1.0,
            "candidate_changed_span_js_reduction_vs_general_min": 0.20,
            "general_dual_median_recomputed_tokens_below_general": True,
        },
        "plans": {
            **plans,
            "candidate_changed_requests": changed,
            "candidate_changed_request_count": len(changed),
        },
        "inputs": {
            "v19_result_path": str(V19_RESULT),
            "v19_result_sha256": sha256(V19_RESULT),
            "v8_cold_result_path": str(V8_COLD_RESULT),
            "v8_cold_result_sha256": sha256(V8_COLD_RESULT),
            "candidate_source_sha256": {
                str(source.relative_to(PROJECT)): sha256(source)
                for source in (
                    PROJECT / "benchmark/multi_workflow/coding_reuse_policy.py",
                    PROJECT
                    / "benchmark/multi_workflow/bridge_reuse_litellm_model.py",
                    PROJECT
                    / "benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py",
                    Path(__file__),
                )
            },
            "baseline_replays": [
                {
                    "path": str(path),
                    "summary_sha256": sha256(path / "REPLAY_SUMMARY.json"),
                    "dense_results_sha256": sha256(
                        path / "dense" / "REPLAY_RESULTS.json"
                    ),
                    "general_results_sha256": sha256(
                        path / "general" / "REPLAY_RESULTS.json"
                    ),
                    "general_server_ledger_sha256": sha256(
                        path / "general" / "SERVER_LEDGER.jsonl"
                    ),
                }
                for path in BASELINE_REPLAYS
            ],
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
        "status": "REGISTERED_BEFORE_V20_GPU",
    }
    write_json(path, value)
    return value


def run_arm(output: Path, repeat: int, arm: str) -> dict[str, Any]:
    register(output)
    if (repeat, arm) not in PORTS:
        raise ValueError(f"unsupported run: repeat={repeat}, arm={arm}")
    run_dir = output / f"repeat_{repeat}" / arm
    result_path = run_dir / "REPLAY_RESULTS.json"
    if result_path.exists():
        return read_json(result_path)
    run_dir.mkdir(parents=True)
    manifest = init_manifest(run_dir, arm)
    planner = make_planner(
        arm=arm,
        manifest_path=manifest,
        client_ledger_path=run_dir / "PLANNER_LEDGER.jsonl",
        instance_nonce=f"runtime-{arm}-r{repeat}",
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
                    else f"{arm}-r{repeat}-{instance_id}-q{request_index}"
                )
                generated = generate_one(
                    base_url=base_url,
                    input_ids=planned["prompt_ids"],
                    key=key,
                )
                rows.append(
                    {
                        "arm": arm,
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
        if planner._pending_source is not None:
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
        "completed_at_utc": utc_now(),
        "repeat": repeat,
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


def _copies(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["case_id"]): row
        for row in load_jsonl(path)
        if row.get("event") == "target_copied"
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
        "first_token_agreement": sum(agreements) / len(agreements),
        "mean_top20_plus_residual_js": statistics.fmean(valid),
    }


def summarize_repeat(output: Path, repeat: int) -> dict[str, Any]:
    baseline = BASELINE_REPLAYS[repeat - 1]
    rows = {
        "dense": _index(
            read_json(baseline / "dense" / "REPLAY_RESULTS.json")["rows"]
        ),
        "general": _index(
            read_json(baseline / "general" / "REPLAY_RESULTS.json")["rows"]
        ),
        **{
            arm: _index(
                read_json(
                    output / f"repeat_{repeat}" / arm / "REPLAY_RESULTS.json"
                )["rows"]
            )
            for arm in NEW_ARMS
        },
    }
    keys = set(rows["dense"])
    if any(set(value) != keys for value in rows.values()):
        raise ValueError("request identities differ")
    if any(
        rows[arm][key]["prompt_hash"] != rows["dense"][key]["prompt_hash"]
        for arm in rows
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
        "general": load_jsonl(baseline / "general" / "SERVER_LEDGER.jsonl"),
        **{
            arm: load_jsonl(
                output
                / f"repeat_{repeat}"
                / arm
                / "SERVER_LEDGER.jsonl"
            )
            for arm in NEW_ARMS
        },
    }
    copies = {
        arm: {
            str(row["case_id"]): row
            for row in ledger
            if row.get("event") == "target_copied"
        }
        for arm, ledger in ledgers.items()
    }
    builds = {
        arm: {
            str(row["source_id"]): float(row["materialize_ms"])
            for row in ledgers[arm]
            if row.get("event")
            in ("source_materialized", "source_materialized_host")
        }
        for arm in NEW_ARMS
    }
    cache_ttft = {
        arm: statistics.median(
            float(rows[arm][key]["ttft_ms"]) for key in target_keys
        )
        for arm in ("general", *NEW_ARMS)
    }
    baseline_summary = read_json(baseline / "REPLAY_SUMMARY.json")
    n4 = {
        "general": baseline_summary["arm_summaries"]["general"][
            "median_n4_including_build_ms"
        ]
    }
    for arm in NEW_ARMS:
        values = [
            float(rows[arm][key]["ttft_ms"])
            + builds[arm][str(rows[arm][key]["target_source_id"])] / 4
            for key in target_keys
        ]
        n4[arm] = statistics.median(values)
    fidelity = {
        arm: _fidelity(list(keys), rows["dense"], rows[arm])
        for arm in ("general", *NEW_ARMS)
    }
    changed_fidelity = {
        arm: _fidelity(changed_keys, rows["dense"], rows[arm])
        for arm in ("general", CANDIDATE)
    }
    recomputed = {
        arm: statistics.median(
            int(copies[arm][str(rows[arm][key]["request_key"])]["recomputed_tokens"])
            for key in target_keys
        )
        for arm in ("general", *NEW_ARMS)
    }
    return {
        "repeat": repeat,
        "target_requests": len(target_keys),
        "changed_target_requests": len(changed_keys),
        "mechanism": {
            arm: {
                "physical_copy_events": len(copies[arm]),
                "target_fallbacks": sum(
                    row.get("event") == "target_fallback"
                    for row in ledgers[arm]
                ),
                "median_recomputed_tokens": recomputed[arm],
            }
            for arm in ("general", *NEW_ARMS)
        },
        "cache_ready_ttft_ms": cache_ttft,
        "n4_including_build_ms": n4,
        "candidate_over_general_cache_ready_ttft": (
            cache_ttft[CANDIDATE] / cache_ttft["general"]
        ),
        "candidate_over_general_n4": n4[CANDIDATE] / n4["general"],
        "fidelity": fidelity,
        "changed_span_fidelity": changed_fidelity,
        "changed_span_js_reduction_vs_general": 1.0
        - changed_fidelity[CANDIDATE]["mean_top20_plus_residual_js"]
        / changed_fidelity["general"]["mean_top20_plus_residual_js"],
    }


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    repeats = [summarize_repeat(output, repeat) for repeat in ORDER]
    gates = registration["frozen_gates"]
    outcomes = []
    for row in repeats:
        mechanism = row["mechanism"]
        outcomes.append(
            {
                "repeat": row["repeat"],
                "mechanism": all(
                    mechanism[arm]["physical_copy_events"]
                    == gates["physical_copy_events_each_arm"]
                    and mechanism[arm]["target_fallbacks"]
                    <= gates["target_fallbacks_each_arm_max"]
                    for arm in NEW_ARMS
                ),
                "candidate_first_token": (
                    row["fidelity"][CANDIDATE]["first_token_agreement"]
                    >= row["fidelity"]["general"]["first_token_agreement"]
                ),
                "candidate_js": (
                    row["fidelity"][CANDIDATE][
                        "mean_top20_plus_residual_js"
                    ]
                    <= row["fidelity"]["general"][
                        "mean_top20_plus_residual_js"
                    ]
                ),
                "candidate_cache_ready_ttft": (
                    row["candidate_over_general_cache_ready_ttft"]
                    <= gates["candidate_cache_ready_ttft_over_general_max"]
                ),
                "candidate_n4": (
                    row["candidate_over_general_n4"]
                    <= gates["candidate_n4_including_build_over_general_max"]
                ),
                "changed_span_js": (
                    row["changed_span_js_reduction_vs_general"]
                    >= gates[
                        "candidate_changed_span_js_reduction_vs_general_min"
                    ]
                ),
                "ordinary_prefix_recompute": (
                    mechanism[CONTROL]["median_recomputed_tokens"]
                    < mechanism["general"]["median_recomputed_tokens"]
                ),
            }
        )
    promoted = all(
        all(value for key, value in row.items() if key != "repeat")
        for row in outcomes
    )
    value = {
        "status": "V20_REPLAY_COMPLETE",
        "completed_at_utc": utc_now(),
        "candidate": CANDIDATE,
        "repeats": repeats,
        "gate_outcomes": outcomes,
        "promoted_to_development_accuracy": promoted,
        "scope": (
            "same-prompt mechanism/speed/first-token fidelity only; no task "
            "accuracy claim"
        ),
        "prefetch": False,
    }
    write_json(output / "V20_REPLAY_RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--repeat", type=int, choices=ORDER)
    run_parser.add_argument("--arm", choices=NEW_ARMS)
    sub.add_parser("summarize")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "register":
        value = register(output)
    elif args.command == "run":
        repeats = (args.repeat,) if args.repeat else tuple(ORDER)
        completed = {}
        for repeat in repeats:
            arms = (args.arm,) if args.arm else ORDER[repeat]
            for arm in arms:
                result = run_arm(output, repeat, arm)
                completed[f"{repeat}:{arm}"] = {
                    key: result[key] for key in ("arm", "repeat", "requests")
                }
        if all(
            (
                output / f"repeat_{repeat}" / arm / "REPLAY_RESULTS.json"
            ).exists()
            for repeat in ORDER
            for arm in NEW_ARMS
        ):
            completed["summary"] = summarize(output)
        value = completed
    else:
        value = summarize(output)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
