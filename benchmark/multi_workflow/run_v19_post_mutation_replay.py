#!/usr/bin/env python3
"""Validate the V19 post-mutation island on frozen identical agent prompts."""

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
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v19_post_mutation_replay_20260727"
BASELINE_REPLAYS = (
    ARTIFACTS / "impactkv_v18_frozen_replay_20260727",
    ARTIFACTS / "impactkv_v18r_frozen_replay_replication_20260727",
)
SELECTOR_AUDIT = (
    ARTIFACTS
    / "impactkv_v19_selector_cost_audit_20260727"
    / "V19_SELECTOR_COST_RESULT.json"
)
HOST_CONTROL = (
    ARTIFACTS
    / "impactkv_v19_host_overflow_control_20260727"
    / "V19_HOST_CONTROL_RESULT.json"
)
PROJECT = Path(__file__).resolve().parents[2]
CANDIDATE = "coding_post_mutation_v19"
PORTS = {1: 32310, 2: 32311}


def _plan_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["instance_id"],
        row["request_index"],
        row["prompt_hash"],
    )


def _span_plan(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["target_registered"],
        row["copied_tokens_planned"],
        row["source_registered"],
        row["source_tokens_planned"],
    )


def register(output: Path) -> dict[str, Any]:
    path = output / "V19_REPLAY_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    general = simulate_arm("general")
    candidate = simulate_arm(CANDIDATE)
    if [_plan_identity(row) for row in general] != [
        _plan_identity(row) for row in candidate
    ]:
        raise ValueError("candidate changes frozen prompt identity")
    changed = [
        {
            "instance_id": left["instance_id"],
            "request_index": left["request_index"],
            "general_span": _span_plan(left),
            "candidate_span": _span_plan(right),
            "candidate_decision": right["decision"],
        }
        for left, right in zip(general, candidate)
        if _span_plan(left) != _span_plan(right)
    ]
    value = {
        "registered_at_utc": utc_now(),
        "registered_before_candidate_gpu": True,
        "experiment": "V19 post-mutation island identical-prompt replay",
        "candidate": CANDIDATE,
        "motivation": (
            "V17's exact-General spans have no measurable process overhead. "
            "Changed spans reduce JS but add a median 1249.5 recomputed tokens. "
            "The latest-risk-only guard worsens JS; V19 removes that guard and "
            "uses file-version boundaries only to select a later contiguous "
            "island after a mutation."
        ),
        "causal_scope": (
            "V19 does not claim recomputed earlier tokens can attend to a later "
            "mutation. The online file-version event is only a selector for a "
            "more recent reusable contiguous island."
        ),
        "protocol": {
            "instances": list(INSTANCE_IDS),
            "requests_per_arm": len(candidate),
            "repeats": 2,
            "ports": PORTS,
            "temperature": 0,
            "diagnostic_new_tokens": 1,
            "diagnostic_output_fed_forward": False,
            "copy_cap_tokens": 4096,
            "host_overflow": True,
            "prefetch": False,
        },
        "frozen_gates": {
            "prompt_hashes_identical": True,
            "physical_copy_events": 39,
            "target_fallbacks_max": 0,
            "first_token_agreement_with_dense_each_repeat_min": 0.95,
            "mean_js_not_above_general_each_repeat": True,
            "cache_ready_ttft_over_general_each_repeat_max": 1.10,
            "n4_including_build_over_general_each_repeat_max": 1.10,
            "changed_span_js_reduction_vs_general_each_repeat_min": 0.20,
        },
        "plans": {
            "general": general,
            CANDIDATE: candidate,
            "changed_requests": changed,
            "changed_request_count": len(changed),
        },
        "inputs": {
            "selector_audit_path": str(SELECTOR_AUDIT),
            "selector_audit_sha256": sha256(SELECTOR_AUDIT),
            "host_control_path": str(HOST_CONTROL),
            "host_control_sha256": sha256(HOST_CONTROL),
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
        "status": "REGISTERED_BEFORE_CANDIDATE_GPU",
    }
    write_json(path, value)
    return value


def run_candidate(output: Path, repeat: int) -> dict[str, Any]:
    register(output)
    if repeat not in PORTS:
        raise ValueError(f"unsupported repeat: {repeat}")
    run_dir = output / f"repeat_{repeat}" / CANDIDATE
    result_path = run_dir / "REPLAY_RESULTS.json"
    if result_path.exists():
        return read_json(result_path)
    run_dir.mkdir(parents=True)
    manifest = init_manifest(run_dir, CANDIDATE)
    planner = make_planner(
        arm=CANDIDATE,
        manifest_path=manifest,
        client_ledger_path=run_dir / "PLANNER_LEDGER.jsonl",
        instance_nonce=f"runtime-{CANDIDATE}-r{repeat}",
    )
    process, log = launch_server(
        run_dir=run_dir,
        arm=CANDIDATE,
        manifest=manifest,
        port=PORTS[repeat],
    )
    rows: list[dict[str, Any]] = []
    try:
        base_url = f"http://127.0.0.1:{PORTS[repeat]}"
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
                    else f"{CANDIDATE}-r{repeat}-{instance_id}-q{request_index}"
                )
                generated = generate_one(
                    base_url=base_url,
                    input_ids=planned["prompt_ids"],
                    key=key,
                )
                rows.append(
                    {
                        "arm": CANDIDATE,
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
        "arm": CANDIDATE,
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


def _summarize_repeat(output: Path, repeat: int) -> dict[str, Any]:
    run_dir = output / f"repeat_{repeat}" / CANDIDATE
    candidate = _index(read_json(run_dir / "REPLAY_RESULTS.json")["rows"])
    baseline = BASELINE_REPLAYS[repeat - 1]
    dense = _index(
        read_json(baseline / "dense" / "REPLAY_RESULTS.json")["rows"]
    )
    general = _index(
        read_json(baseline / "general" / "REPLAY_RESULTS.json")["rows"]
    )
    if not (set(candidate) == set(dense) == set(general)):
        raise ValueError("request identities differ")
    if not all(
        candidate[key]["prompt_hash"] == dense[key]["prompt_hash"]
        == general[key]["prompt_hash"]
        for key in candidate
    ):
        raise ValueError("prompt hashes differ")
    target_keys = [
        key for key, row in candidate.items() if row["target_registered"]
    ]
    changed_keys = [
        key
        for key in target_keys
        if (
            candidate[key]["target_length"],
            candidate[key]["source_length"],
        )
        != (
            general[key]["target_length"],
            general[key]["source_length"],
        )
    ]
    ledger = load_jsonl(run_dir / "SERVER_LEDGER.jsonl")
    copies = [row for row in ledger if row.get("event") == "target_copied"]
    fallbacks = [row for row in ledger if row.get("event") == "target_fallback"]
    builds = {
        str(row["source_id"]): float(row["materialize_ms"])
        for row in ledger
        if row.get("event")
        in ("source_materialized", "source_materialized_host")
    }

    def fidelity(keys: list[tuple[str, int]], arm_rows: dict) -> dict[str, Any]:
        agreements = [
            token_id(arm_rows[key]) == token_id(dense[key]) for key in keys
        ]
        divergences = [
            coarse_js(
                top_distribution(dense[key]),
                top_distribution(arm_rows[key]),
            )
            for key in keys
        ]
        valid = [value for value in divergences if value is not None]
        return {
            "requests": len(keys),
            "first_token_agreement": sum(agreements) / len(agreements),
            "mean_top20_plus_residual_js": statistics.fmean(valid),
        }

    cache_ttft = {
        "general": statistics.median(
            float(general[key]["ttft_ms"]) for key in target_keys
        ),
        CANDIDATE: statistics.median(
            float(candidate[key]["ttft_ms"]) for key in target_keys
        ),
    }
    n4_candidate = [
        float(candidate[key]["ttft_ms"])
        + builds[str(candidate[key]["target_source_id"])] / 4
        for key in target_keys
    ]
    baseline_summary = read_json(baseline / "REPLAY_SUMMARY.json")
    n4_general = baseline_summary["arm_summaries"]["general"][
        "median_n4_including_build_ms"
    ]
    full_fidelity = {
        "general": fidelity(list(candidate), general),
        CANDIDATE: fidelity(list(candidate), candidate),
    }
    changed_fidelity = {
        "general": fidelity(changed_keys, general),
        CANDIDATE: fidelity(changed_keys, candidate),
    }
    return {
        "repeat": repeat,
        "requests": len(candidate),
        "target_requests": len(target_keys),
        "changed_target_requests": len(changed_keys),
        "physical_copy_events": len(copies),
        "target_fallbacks": len(fallbacks),
        "host_target_copies": sum(
            row.get("source_residency") == "host" for row in copies
        ),
        "cache_ready_ttft_ms": cache_ttft,
        "candidate_over_general_cache_ready_ttft": (
            cache_ttft[CANDIDATE] / cache_ttft["general"]
        ),
        "n4_including_build_ms": {
            "general": n4_general,
            CANDIDATE: statistics.median(n4_candidate),
        },
        "candidate_over_general_n4": (
            statistics.median(n4_candidate) / n4_general
        ),
        "full_fidelity": full_fidelity,
        "changed_span_fidelity": changed_fidelity,
        "changed_span_js_reduction_vs_general": (
            1.0
            - changed_fidelity[CANDIDATE]["mean_top20_plus_residual_js"]
            / changed_fidelity["general"]["mean_top20_plus_residual_js"]
        ),
    }


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    repeats = [_summarize_repeat(output, repeat) for repeat in PORTS]
    gates = registration["frozen_gates"]
    gate_rows = []
    for row in repeats:
        candidate_fidelity = row["full_fidelity"][CANDIDATE]
        general_fidelity = row["full_fidelity"]["general"]
        gate_rows.append(
            {
                "repeat": row["repeat"],
                "mechanism": (
                    row["physical_copy_events"]
                    == gates["physical_copy_events"]
                    and row["target_fallbacks"]
                    <= gates["target_fallbacks_max"]
                ),
                "first_token": (
                    candidate_fidelity["first_token_agreement"]
                    >= gates[
                        "first_token_agreement_with_dense_each_repeat_min"
                    ]
                ),
                "full_js": (
                    candidate_fidelity["mean_top20_plus_residual_js"]
                    <= general_fidelity["mean_top20_plus_residual_js"]
                ),
                "cache_ready_ttft": (
                    row["candidate_over_general_cache_ready_ttft"]
                    <= gates[
                        "cache_ready_ttft_over_general_each_repeat_max"
                    ]
                ),
                "n4": (
                    row["candidate_over_general_n4"]
                    <= gates[
                        "n4_including_build_over_general_each_repeat_max"
                    ]
                ),
                "changed_span_js": (
                    row["changed_span_js_reduction_vs_general"]
                    >= gates[
                        "changed_span_js_reduction_vs_general_each_repeat_min"
                    ]
                ),
            }
        )
    promoted = all(
        all(value for key, value in row.items() if key != "repeat")
        for row in gate_rows
    )
    value = {
        "status": "V19_REPLAY_COMPLETE",
        "completed_at_utc": utc_now(),
        "candidate": CANDIDATE,
        "repeats": repeats,
        "gate_outcomes": gate_rows,
        "promoted_to_development_accuracy": promoted,
        "scope": (
            "same-prompt mechanism/speed/first-token fidelity only; no task "
            "accuracy claim"
        ),
    }
    write_json(output / "V19_REPLAY_RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--repeat", type=int, choices=PORTS)
    sub.add_parser("summarize")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "register":
        value = register(output)
    elif args.command == "run":
        repeats = (args.repeat,) if args.repeat else tuple(PORTS)
        value = {
            str(repeat): {
                key: result[key]
                for key in ("arm", "repeat", "requests")
            }
            for repeat in repeats
            for result in (run_candidate(output, repeat),)
        }
        if all(
            (
                output
                / f"repeat_{repeat}"
                / CANDIDATE
                / "REPLAY_RESULTS.json"
            ).exists()
            for repeat in PORTS
        ):
            value["summary"] = summarize(output)
    else:
        value = summarize(output)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
