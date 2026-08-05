#!/usr/bin/env python3
"""Request-disjoint holdout for M52's reverse path-dependency finding.

M52 unexpectedly found that path-relevant observations receive more target
attention while showing less K/V drift and less physical-splice harm.  M53
freezes that direction on unused request IDs and keeps at most one request per
candidate-pair identity.  Tasks and some individual observations may overlap
M52, so this is transition holdout evidence, not task/observation independence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from benchmark.multi_workflow.motivate_v50_coding_provenance import (
    _balanced_select,
    _sha256,
)
from benchmark.multi_workflow import motivate_v52_path_dependency as m52
from benchmark.multi_workflow.run_bridge_reuse_pilot import write_json


ROOT = Path("/home/gfy/CodeMAS_Project")
DEVELOPMENT_DESIGN = (
    ROOT
    / "kvflow-artifacts/impactkv_m52_path_dependency_20260805/"
    "matched20/DESIGN.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_m53_path_dependency_holdout_20260805/"
    "request_disjoint19"
)
RANDOM_SEED = 202608053


def _candidate_pair_key(case: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(str(candidate["segment_token_hash"]) for candidate in case["candidates"])
    )


def _holdout_pool(tokenizer: Any) -> list[dict[str, Any]]:
    development = json.loads(DEVELOPMENT_DESIGN.read_text(encoding="utf-8"))[
        "cases"
    ]
    development_ids = {str(row["case_id"]) for row in development}
    remaining = [
        row
        for row in m52._candidate_pool(tokenizer)
        if str(row["case_id"]) not in development_ids
    ]
    ranked = sorted(
        remaining,
        key=lambda row: hashlib.sha256(
            f"{RANDOM_SEED}:{row['case_id']}".encode()
        ).hexdigest(),
    )
    selected = []
    used_pairs = set()
    for row in ranked:
        key = _candidate_pair_key(row)
        if key in used_pairs:
            continue
        selected.append(row)
        used_pairs.add(key)
    return selected


def prepare(output: Path, limit: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(m52.MODEL, local_files_only=True)
    eligible = _holdout_pool(tokenizer)
    selected = _balanced_select(eligible, limit)
    if len(selected) < limit:
        raise ValueError(f"only {len(selected)} eligible cases for requested {limit}")
    output.mkdir(parents=True)
    design_path = output / "DESIGN.json"
    write_json(
        design_path,
        {
            "cases": selected,
            "eligible_unique_candidate_pairs": len(eligible),
            "model": str(m52.MODEL),
        },
    )
    registration = {
        "status": "REGISTERED_BEFORE_HOLDOUT_GPU",
        "purpose": (
            "validate M52's post-discovery direction: path-relevant blocks "
            "receive more attention but have lower drift and splice harm"
        ),
        "design_sha256": _sha256(design_path),
        "development_design": str(DEVELOPMENT_DESIGN),
        "development_design_sha256": _sha256(DEVELOPMENT_DESIGN),
        "cases": len(selected),
        "tasks": len({row["instance_id"] for row in selected}),
        "eligible_unique_candidate_pairs": len(eligible),
        "model": str(m52.MODEL),
        "holdout_contract": {
            "development_request_id_overlap": 0,
            "one_request_per_candidate_pair_identity": True,
            "task_disjoint": False,
            "observation_disjoint": False,
            "same_measurement_as_M52": True,
        },
        "frozen_replication_rule": {
            "minimum_complete_cases": 16,
            "minimum_tasks": 8,
            "relevant_higher_attention_pair_fraction_min": 0.60,
            "position_adjusted_attention_ratio_min": 1.25,
            "relevant_lower_drift_pair_fraction_min": 0.60,
            "position_adjusted_drift_ratio_max": 0.70,
            "relevant_lower_JS_pair_fraction_min": 0.60,
            "position_adjusted_JS_ratio_max": 0.80,
        },
        "interpretation_limits": [
            "post-M52 directional hypothesis, tested once on unused requests",
            "some tasks and observations overlap development",
            "Dense target attention remains an oracle diagnostic",
            "passing does not establish task accuracy or latency",
        ],
    }
    write_json(output / "REGISTRATION.json", registration)
    return registration


def analyze(output: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (output / "OBSERVATIONS.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not rows or any(not m52._complete(row) for row in rows):
        raise ValueError("observations are missing or incomplete")
    metrics = {
        metric: m52._metric_summary(rows, metric)
        for metric in (
            "attention_mean",
            "kv_cosine_drift_mean",
            "risk_product_mean",
            "causal_splice_logit_js",
        )
    }
    registration = json.loads((output / "REGISTRATION.json").read_text())
    gates = registration["frozen_replication_rule"]
    attention = metrics["attention_mean"]
    drift = metrics["kv_cosine_drift_mean"]
    js = metrics["causal_splice_logit_js"]
    gate_results = {
        "minimum_complete_cases": len(rows) >= gates["minimum_complete_cases"],
        "minimum_tasks": len({row["instance_id"] for row in rows})
        >= gates["minimum_tasks"],
        "relevant_higher_attention_pair_fraction": attention[
            "path_relevant_higher_pair_fraction"
        ]
        >= gates["relevant_higher_attention_pair_fraction_min"],
        "position_adjusted_attention_ratio": attention[
            "position_adjusted_geometric_ratio"
        ]
        >= gates["position_adjusted_attention_ratio_min"],
        "relevant_lower_drift_pair_fraction": (
            1 - drift["path_relevant_higher_pair_fraction"]
        )
        >= gates["relevant_lower_drift_pair_fraction_min"],
        "position_adjusted_drift_ratio": drift[
            "position_adjusted_geometric_ratio"
        ]
        <= gates["position_adjusted_drift_ratio_max"],
        "relevant_lower_JS_pair_fraction": (
            1 - js["path_relevant_higher_pair_fraction"]
        )
        >= gates["relevant_lower_JS_pair_fraction_min"],
        "position_adjusted_JS_ratio": js["position_adjusted_geometric_ratio"]
        <= gates["position_adjusted_JS_ratio_max"],
    }
    decision = "REPLICATED" if all(gate_results.values()) else "NOT_REPLICATED"
    value = {
        "status": "COMPLETE",
        "decision": decision,
        "cases": len(rows),
        "tasks": len({row["instance_id"] for row in rows}),
        "metrics": metrics,
        "frozen_gate_results": gate_results,
        "scope": (
            "request-disjoint, candidate-pair-deduplicated transition holdout; "
            "not task- or observation-disjoint"
        ),
        "next_step": (
            "build an online path-relevant-first V54 selector and validate on new tasks"
            if decision == "REPLICATED"
            else "keep M52 as exploratory and collect new task-disjoint trajectories"
        ),
    }
    write_json(output / "RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--limit", type=int, default=19)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    measure_parser.add_argument("--max-cases", type=int, default=0)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.output, args.limit)
    elif args.command == "measure":
        value = m52.measure(args.output, args.max_cases)
    else:
        value = analyze(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
