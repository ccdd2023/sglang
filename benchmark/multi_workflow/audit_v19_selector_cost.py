#!/usr/bin/env python3
"""Attribute V17's identical-prompt latency/fidelity trade-off by selector event."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_bridge_reuse_agent_experiment import load_jsonl
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    coarse_js,
    read_json,
    sha256,
    token_id,
    top_distribution,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v19_selector_cost_audit_20260727"
REPLAYS = (
    ARTIFACTS / "impactkv_v18_frozen_replay_20260727",
    ARTIFACTS / "impactkv_v18r_frozen_replay_replication_20260727",
)
ARMS = ("dense", "general", "coding_version_graph_v17")


def register(output: Path) -> dict[str, Any]:
    path = output / "V19_SELECTOR_COST_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    value = {
        "registered_at_utc": utc_now(),
        "registered_before_row_level_analysis": True,
        "experiment": "V19 V17 selector cost/fidelity attribution",
        "question": (
            "With host residency already ruled out in the 60-request replay, "
            "which online version-graph event causes V17's latency cost, and "
            "does that event deliver enough Dense-reference fidelity gain?"
        ),
        "cohorts_frozen_before_analysis": {
            "exact_general_span": (
                "target_start, length, and segment token hash all match General"
            ),
            "stale_only": "V17 changes span with stale_groups>0 and no latest guard",
            "latest_guard_only": (
                "V17 changes span with latest_group_protected and no stale group"
            ),
            "stale_and_latest_guard": (
                "V17 changes span with both stale removal and latest guard"
            ),
            "other_changed_span": "changed span not covered above",
        },
        "metrics": [
            "median cache-ready TTFT and paired per-request TTFT delta",
            "median recomputed-token delta",
            "mean top20+residual JS to Dense",
            "first-token agreement to Dense",
            "fraction of requests where V17 JS is lower than General",
        ],
        "frozen_interpretation_gates": {
            "exact_span_abs_median_ttft_delta_percent_max": 5.0,
            "changed_span_median_ttft_slowdown_percent_min": 15.0,
            "changed_span_median_recomputed_token_increase_min": 1,
            "changed_span_v17_lower_js_fraction_min": 0.50,
        },
        "inputs": [
            {
                "path": str(path),
                "summary_sha256": sha256(path / "REPLAY_SUMMARY.json"),
                **{
                    f"{arm}_results_sha256": sha256(
                        path / arm / "REPLAY_RESULTS.json"
                    )
                    for arm in ARMS
                },
                **{
                    f"{arm}_manifest_sha256": sha256(
                        path / arm / "DYNAMIC_MANIFEST.json"
                    )
                    for arm in ARMS[1:]
                },
                **{
                    f"{arm}_server_ledger_sha256": sha256(
                        path / arm / "SERVER_LEDGER.jsonl"
                    )
                    for arm in ARMS[1:]
                },
            }
            for path in REPLAYS
        ],
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
        "status": "REGISTERED_BEFORE_ROW_LEVEL_ANALYSIS",
    }
    write_json(path, value)
    return value


def index_rows(path: Path, arm: str) -> dict[tuple[str, int], dict[str, Any]]:
    rows = read_json(path / arm / "REPLAY_RESULTS.json")["rows"]
    return {
        (str(row["instance_id"]), int(row["request_index"])): row
        for row in rows
    }


def cases_by_prompt(path: Path, arm: str) -> dict[str, dict[str, Any]]:
    rows = read_json(path / arm / "DYNAMIC_MANIFEST.json")["cases"]
    return {str(row["target_prompt_hash"]): row for row in rows}


def copies_by_case(path: Path, arm: str) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path / arm / "SERVER_LEDGER.jsonl")
    return {
        str(row["case_id"]): row
        for row in rows
        if row.get("event") == "target_copied"
    }


def source_decisions(
    rows: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row["source_id"]): row["decision"]
        for row in rows.values()
        if row.get("source_id")
    }


def cohort_name(
    *,
    same_span: bool,
    decision: dict[str, Any],
) -> str:
    if same_span:
        return "exact_general_span"
    stale = int(decision.get("stale_groups", 0)) > 0
    guarded = bool(decision.get("latest_group_protected", False))
    if stale and guarded:
        return "stale_and_latest_guard"
    if stale:
        return "stale_only"
    if guarded:
        return "latest_guard_only"
    return "other_changed_span"


def request_rows(path: Path, repeat: int) -> list[dict[str, Any]]:
    indexed = {arm: index_rows(path, arm) for arm in ARMS}
    if not (
        set(indexed["dense"])
        == set(indexed["general"])
        == set(indexed["coding_version_graph_v17"])
    ):
        raise ValueError("request identities differ across arms")
    manifests = {
        arm: cases_by_prompt(path, arm) for arm in ARMS[1:]
    }
    copies = {arm: copies_by_case(path, arm) for arm in ARMS[1:]}
    decisions = source_decisions(indexed["coding_version_graph_v17"])
    output = []
    for key, v17 in indexed["coding_version_graph_v17"].items():
        if not v17["target_registered"]:
            continue
        dense = indexed["dense"][key]
        general = indexed["general"][key]
        general_case = manifests["general"][general["prompt_hash"]]
        v17_case = manifests["coding_version_graph_v17"][v17["prompt_hash"]]
        same_span = all(
            general_case[field] == v17_case[field]
            for field in ("target_start", "length", "segment_token_hash")
        )
        decision = decisions[str(v17["target_source_id"])]
        general_copy = copies["general"][str(general["request_key"])]
        v17_copy = copies["coding_version_graph_v17"][
            str(v17["request_key"])
        ]
        general_js = coarse_js(
            top_distribution(dense), top_distribution(general)
        )
        v17_js = coarse_js(top_distribution(dense), top_distribution(v17))
        output.append(
            {
                "repeat": repeat,
                "instance_id": key[0],
                "request_index": key[1],
                "cohort": cohort_name(
                    same_span=same_span, decision=decision
                ),
                "same_span": same_span,
                "general_ttft_ms": float(general["ttft_ms"]),
                "v17_ttft_ms": float(v17["ttft_ms"]),
                "v17_minus_general_ttft_ms": (
                    float(v17["ttft_ms"]) - float(general["ttft_ms"])
                ),
                "v17_slowdown_vs_general_percent": 100.0
                * (
                    float(v17["ttft_ms"]) / float(general["ttft_ms"])
                    - 1.0
                ),
                "general_recomputed_tokens": int(
                    general_copy["recomputed_tokens"]
                ),
                "v17_recomputed_tokens": int(v17_copy["recomputed_tokens"]),
                "v17_minus_general_recomputed_tokens": int(
                    v17_copy["recomputed_tokens"]
                )
                - int(general_copy["recomputed_tokens"]),
                "general_copied_tokens": int(general_copy["copied_k_tokens"]),
                "v17_copied_tokens": int(v17_copy["copied_k_tokens"]),
                "general_js_to_dense": general_js,
                "v17_js_to_dense": v17_js,
                "v17_lower_js": (
                    v17_js < general_js
                    if v17_js is not None and general_js is not None
                    else None
                ),
                "general_first_token_matches_dense": (
                    token_id(general) == token_id(dense)
                ),
                "v17_first_token_matches_dense": (
                    token_id(v17) == token_id(dense)
                ),
                "decision": decision,
            }
        )
    return output


def summarize_cohort(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"requests": 0}
    valid_js = [
        row for row in rows if row["v17_lower_js"] is not None
    ]
    return {
        "requests": len(rows),
        "median_general_ttft_ms": statistics.median(
            row["general_ttft_ms"] for row in rows
        ),
        "median_v17_ttft_ms": statistics.median(
            row["v17_ttft_ms"] for row in rows
        ),
        "median_paired_ttft_delta_ms": statistics.median(
            row["v17_minus_general_ttft_ms"] for row in rows
        ),
        "median_paired_slowdown_percent": statistics.median(
            row["v17_slowdown_vs_general_percent"] for row in rows
        ),
        "median_recomputed_token_increase": statistics.median(
            row["v17_minus_general_recomputed_tokens"] for row in rows
        ),
        "mean_general_js_to_dense": statistics.fmean(
            row["general_js_to_dense"]
            for row in rows
            if row["general_js_to_dense"] is not None
        ),
        "mean_v17_js_to_dense": statistics.fmean(
            row["v17_js_to_dense"]
            for row in rows
            if row["v17_js_to_dense"] is not None
        ),
        "v17_lower_js_fraction": (
            sum(row["v17_lower_js"] for row in valid_js) / len(valid_js)
        ),
        "general_first_token_agreement": sum(
            row["general_first_token_matches_dense"] for row in rows
        )
        / len(rows),
        "v17_first_token_agreement": sum(
            row["v17_first_token_matches_dense"] for row in rows
        )
        / len(rows),
    }


def run(output: Path) -> dict[str, Any]:
    registration = register(output)
    rows = [
        row
        for repeat, path in enumerate(REPLAYS, start=1)
        for row in request_rows(path, repeat)
    ]
    cohorts = {
        name: summarize_cohort(
            [row for row in rows if row["cohort"] == name]
        )
        for name in registration["cohorts_frozen_before_analysis"]
    }
    changed = [row for row in rows if not row["same_span"]]
    exact = [row for row in rows if row["same_span"]]
    changed_summary = summarize_cohort(changed)
    exact_summary = summarize_cohort(exact)
    gates = registration["frozen_interpretation_gates"]
    result = {
        "status": "V19_SELECTOR_COST_COMPLETE",
        "completed_at_utc": utc_now(),
        "requests": len(rows),
        "target_requests_per_repeat": len(rows) // len(REPLAYS),
        "cohorts": cohorts,
        "exact_span": exact_summary,
        "changed_span": changed_summary,
        "verdict": {
            "exact_span_process_baseline_passed": abs(
                exact_summary["median_paired_slowdown_percent"]
            )
            <= gates["exact_span_abs_median_ttft_delta_percent_max"],
            "changed_span_latency_cost_passed": (
                changed_summary["median_paired_slowdown_percent"]
                >= gates["changed_span_median_ttft_slowdown_percent_min"]
            ),
            "changed_span_recompute_cost_passed": (
                changed_summary["median_recomputed_token_increase"]
                >= gates[
                    "changed_span_median_recomputed_token_increase_min"
                ]
            ),
            "changed_span_fidelity_utility_passed": (
                changed_summary["v17_lower_js_fraction"]
                >= gates["changed_span_v17_lower_js_fraction_min"]
            ),
        },
        "rows": rows,
    }
    write_json(output / "V19_SELECTOR_COST_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("command", choices=("register", "run"))
    args = parser.parse_args()
    value = (
        register(args.output.resolve())
        if args.command == "register"
        else run(args.output.resolve())
    )
    if args.command == "run":
        value = {
            key: value[key]
            for key in (
                "status",
                "requests",
                "cohorts",
                "exact_span",
                "changed_span",
                "verdict",
            )
        }
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
