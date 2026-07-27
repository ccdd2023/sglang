#!/usr/bin/env python3
"""Run a General-4K + host-overflow control on frozen identical prompts.

V17 used host overflow while General did not.  This control keeps General's
planner, copied span, prompt IDs, and 4K cap unchanged and toggles only the
on-demand host-overflow mechanism.  It therefore diagnoses whether V17's
latency gap comes from repository-version selection or KV residency.
"""

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
    ARMS,
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
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v19_host_overflow_control_20260727"
BASELINE_REPLAYS = (
    ARTIFACTS / "impactkv_v18_frozen_replay_20260727",
    ARTIFACTS / "impactkv_v18r_frozen_replay_replication_20260727",
)
CONTROL = "general_host4k_v19_control"
PORTS = {1: 32300, 2: 32301}


def registration(output: Path) -> dict[str, Any]:
    path = output / "V19_HOST_CONTROL_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    plans = simulate_arm("general")
    value = {
        "registration_id": output.name,
        "registered_at_utc": utc_now(),
        "registered_before_control_gpu": True,
        "experiment": "V19 General-4K host-overflow matched control",
        "motivation": (
            "V17 enabled on-demand host overflow and was about 20% slower than "
            "General on identical target prompts, while the official General "
            "run recorded source skips and target fallbacks. Isolate storage "
            "tiering before changing the coding selector."
        ),
        "hypotheses": {
            "host_overflow_explains_gap": (
                "If General+host is at least 15% slower than General-device "
                "and approaches V17 latency, storage tiering explains most of "
                "the observed gap."
            ),
            "selector_explains_gap": (
                "If General+host stays within 10% of General-device, the V17 "
                "selector/span shape or an interaction with host residency "
                "requires direct profiling."
            ),
        },
        "protocol": {
            "control_policy": "General contiguous retained groups, 4096 cap",
            "only_intended_toggle": "host_overflow_enabled=true",
            "instances": list(INSTANCE_IDS),
            "requests": len(plans),
            "repeats": 2,
            "ports": PORTS,
            "diagnostic_new_tokens": 1,
            "diagnostic_output_fed_forward": False,
            "temperature": 0,
            "prefetch": False,
        },
        "frozen_gates": {
            "prompt_hash_identity_with_general": True,
            "planned_span_identity_with_general": True,
            "physical_copy_events_min": 1,
            "target_fallbacks_max": 0,
            "host_explanation_slowdown_vs_general_min_percent": 15.0,
            "selector_explanation_abs_slowdown_vs_general_max_percent": 10.0,
        },
        "inputs": {
            "baseline_replays": [
                {
                    "path": str(path),
                    "registration_sha256": sha256(
                        path / "REPLAY_REGISTRATION.json"
                    ),
                    "summary_sha256": sha256(path / "REPLAY_SUMMARY.json"),
                    "general_results_sha256": sha256(
                        path / "general" / "REPLAY_RESULTS.json"
                    ),
                    "dense_results_sha256": sha256(
                        path / "dense" / "REPLAY_RESULTS.json"
                    ),
                    "v17_results_sha256": sha256(
                        path
                        / "coding_version_graph_v17"
                        / "REPLAY_RESULTS.json"
                    ),
                }
                for path in BASELINE_REPLAYS
            ],
            "trajectory_sha256": {
                instance_id: sha256(trajectory_path(instance_id))
                for instance_id in INSTANCE_IDS
            },
        },
        "plans": plans,
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
        "status": "REGISTERED_BEFORE_CONTROL_GPU",
    }
    write_json(path, value)
    return value


def run_control(output: Path, repeat: int) -> dict[str, Any]:
    registration(output)
    if repeat not in PORTS:
        raise ValueError(f"unsupported repeat: {repeat}")
    run_dir = output / f"repeat_{repeat}" / CONTROL
    result_path = run_dir / "REPLAY_RESULTS.json"
    if result_path.exists():
        return read_json(result_path)
    run_dir.mkdir(parents=True)

    # Use V17 only as the server-launch capability label.  The planner remains
    # exactly General, and the manifest is relabeled for audit readability.
    manifest = init_manifest(run_dir, "coding_version_graph_v17")
    manifest_value = read_json(manifest)
    if manifest_value["host_overflow_enabled"] is not True:
        raise ValueError("control manifest did not enable host overflow")
    manifest_value["arm"] = CONTROL
    write_json(manifest, manifest_value)
    planner = make_planner(
        arm="general",
        manifest_path=manifest,
        client_ledger_path=run_dir / "PLANNER_LEDGER.jsonl",
        instance_nonce=f"runtime-{CONTROL}-r{repeat}",
    )
    process, log = launch_server(
        run_dir=run_dir,
        arm="coding_version_graph_v17",
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
                    else f"{CONTROL}-r{repeat}-{instance_id}-q{request_index}"
                )
                generated = generate_one(
                    base_url=base_url,
                    input_ids=planned["prompt_ids"],
                    key=key,
                )
                rows.append(
                    {
                        "arm": CONTROL,
                        "planner_arm": "general",
                        "host_overflow_enabled": True,
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
        "arm": CONTROL,
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


def summarize_repeat(output: Path, repeat: int) -> dict[str, Any]:
    run_dir = output / f"repeat_{repeat}" / CONTROL
    control_rows = read_json(run_dir / "REPLAY_RESULTS.json")["rows"]
    baseline = BASELINE_REPLAYS[repeat - 1]
    general_rows = read_json(
        baseline / "general" / "REPLAY_RESULTS.json"
    )["rows"]
    dense_rows = read_json(baseline / "dense" / "REPLAY_RESULTS.json")["rows"]
    v17_rows = read_json(
        baseline / "coding_version_graph_v17" / "REPLAY_RESULTS.json"
    )["rows"]
    control = _index(control_rows)
    general = _index(general_rows)
    dense = _index(dense_rows)
    v17 = _index(v17_rows)
    if not (set(control) == set(general) == set(dense) == set(v17)):
        raise ValueError("repeat request identities differ")
    prompt_identity = all(
        control[key]["prompt_hash"] == general[key]["prompt_hash"]
        for key in control
    )
    span_identity = all(
        (
            control[key]["target_registered"],
            control[key]["target_length"],
            control[key]["source_registered"],
            control[key]["source_length"],
        )
        == (
            general[key]["target_registered"],
            general[key]["target_length"],
            general[key]["source_registered"],
            general[key]["source_length"],
        )
        for key in control
    )
    target_keys = [key for key, row in control.items() if row["target_registered"]]
    ledgers = load_jsonl(run_dir / "SERVER_LEDGER.jsonl")
    copies = [row for row in ledgers if row.get("event") == "target_copied"]
    fallbacks = [row for row in ledgers if row.get("event") == "target_fallback"]
    host_builds = [
        row for row in ledgers if row.get("event") == "source_materialized_host"
    ]
    host_copies = [
        row
        for row in copies
        if row.get("source_residency") == "host"
    ]
    ttft = {
        "general_device": statistics.median(
            float(general[key]["ttft_ms"]) for key in target_keys
        ),
        CONTROL: statistics.median(
            float(control[key]["ttft_ms"]) for key in target_keys
        ),
        "coding_version_graph_v17_host": statistics.median(
            float(v17[key]["ttft_ms"]) for key in target_keys
        ),
    }
    control_slowdown = (
        100.0 * (ttft[CONTROL] / ttft["general_device"] - 1.0)
    )
    agreements = [
        token_id(control[key]) == token_id(dense[key]) for key in control
    ]
    js_values = [
        coarse_js(
            top_distribution(control[key]),
            top_distribution(dense[key]),
        )
        for key in control
    ]
    valid_js = [value for value in js_values if value is not None]
    return {
        "repeat": repeat,
        "prompt_hash_identity_with_general": prompt_identity,
        "planned_span_identity_with_general": span_identity,
        "requests": len(control),
        "target_requests": len(target_keys),
        "physical_copy_events": len(copies),
        "target_fallbacks": len(fallbacks),
        "host_source_builds": len(host_builds),
        "host_target_copies": len(host_copies),
        "median_cache_ready_target_ttft_ms": ttft,
        "control_slowdown_vs_general_percent": control_slowdown,
        "control_first_token_agreement_with_dense": (
            sum(agreements) / len(agreements)
        ),
        "control_mean_top20_plus_residual_js_vs_dense": statistics.fmean(
            valid_js
        ),
    }


def summarize(output: Path) -> dict[str, Any]:
    registered = registration(output)
    repeats = [summarize_repeat(output, repeat) for repeat in PORTS]
    threshold = registered["frozen_gates"]
    slowdowns = [
        row["control_slowdown_vs_general_percent"] for row in repeats
    ]
    host_events = sum(row["host_target_copies"] for row in repeats)
    value = {
        "status": "V19_HOST_CONTROL_COMPLETE",
        "completed_at_utc": utc_now(),
        "repeats": repeats,
        "mean_control_slowdown_vs_general_percent": statistics.fmean(slowdowns),
        "host_target_copies": host_events,
        "verdict": {
            "identity_passed": all(
                row["prompt_hash_identity_with_general"]
                and row["planned_span_identity_with_general"]
                for row in repeats
            ),
            "mechanism_passed": all(
                row["physical_copy_events"]
                >= threshold["physical_copy_events_min"]
                and row["target_fallbacks"]
                <= threshold["target_fallbacks_max"]
                for row in repeats
            ),
            "host_path_exercised": host_events > 0,
            "host_overflow_explains_gap": all(
                slowdown
                >= threshold[
                    "host_explanation_slowdown_vs_general_min_percent"
                ]
                for slowdown in slowdowns
            ),
            "selector_or_interaction_explains_gap": all(
                abs(slowdown)
                <= threshold[
                    "selector_explanation_abs_slowdown_vs_general_max_percent"
                ]
                for slowdown in slowdowns
            ),
        },
        "interpretation_rule": (
            "If no host source is actually copied, this experiment can reject "
            "a global flag overhead but cannot estimate host-transfer cost."
        ),
    }
    write_json(output / "V19_HOST_CONTROL_RESULT.json", value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--repeat", type=int, choices=PORTS)
    sub.add_parser("summarize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if args.command == "register":
        value = registration(output)
    elif args.command == "run":
        repeats = (args.repeat,) if args.repeat else tuple(PORTS)
        value = {}
        for repeat in repeats:
            value[str(repeat)] = run_control(output, repeat)
        if all(
            (
                output
                / f"repeat_{repeat}"
                / CONTROL
                / "REPLAY_RESULTS.json"
            ).exists()
            for repeat in PORTS
        ):
            value["summary"] = summarize(output)
    else:
        value = summarize(output)
    if args.command == "run":
        compact = {
            key: (
                {
                    "arm": row["arm"],
                    "requests": row["requests"],
                    "repeat": row["repeat"],
                }
                if key != "summary"
                else row
            )
            for key, row in value.items()
        }
        print(json.dumps(compact, indent=2))
    else:
        print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
