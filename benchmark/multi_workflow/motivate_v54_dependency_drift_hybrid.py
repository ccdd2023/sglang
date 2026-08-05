#!/usr/bin/env python3
"""Test a coding-dependency x 16-token K/V-drift hybrid risk score.

M54 uses the 14 M52-eligible requests not opened by either M52 or M53.  The
path-relevance weight is frozen from M52's position-adjusted attention ratio;
the layer/head configuration is frozen from M49.  No M54 causal label is read
before registration.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.multi_workflow.motivate_v48_attention_kv_risk import (
    _compose_splice,
    _dense_source,
    _model_theta,
    _spearman,
)
from benchmark.multi_workflow.motivate_v49_probe_proxy import _probe_score
from benchmark.multi_workflow import motivate_v52_path_dependency as m52
from benchmark.multi_workflow.motivate_v50_coding_provenance import _sha256
from benchmark.multi_workflow.run_bridge_reuse_pilot import write_json


ROOT = Path("/home/gfy/CodeMAS_Project")
M52_DESIGN = (
    ROOT
    / "kvflow-artifacts/impactkv_m52_path_dependency_20260805/"
    "matched20/DESIGN.json"
)
M52_RESULT = M52_DESIGN.with_name("RESULT.json")
M53_DESIGN = (
    ROOT
    / "kvflow-artifacts/impactkv_m53_path_dependency_holdout_20260805/"
    "request_disjoint19/DESIGN.json"
)
M49_LOCK = (
    ROOT
    / "kvflow-artifacts/impactkv_m49_probe_proxy_20260805/PROXY_LOCK.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_m54_dependency_drift_hybrid_20260805/"
    "untouched14"
)


def _unopened_cases(tokenizer: Any) -> list[dict[str, Any]]:
    used = set()
    for path in (M52_DESIGN, M53_DESIGN):
        used.update(
            str(row["case_id"])
            for row in json.loads(path.read_text(encoding="utf-8"))["cases"]
        )
    return [
        row
        for row in m52._candidate_pool(tokenizer)
        if str(row["case_id"]) not in used
    ]


def prepare(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    m52_result = json.loads(M52_RESULT.read_text(encoding="utf-8"))
    path_weight = float(
        m52_result["metrics"]["attention_mean"][
            "position_adjusted_geometric_ratio"
        ]
    )
    m49_lock = json.loads(M49_LOCK.read_text(encoding="utf-8"))
    layer = int(m49_lock["chosen"]["layer"])
    head_tokens = int(m49_lock["chosen"]["head_tokens"])
    tokenizer = AutoTokenizer.from_pretrained(m52.MODEL, local_files_only=True)
    cases = _unopened_cases(tokenizer)
    if len(cases) != 14:
        raise ValueError(f"expected 14 unopened cases, found {len(cases)}")
    output.mkdir(parents=True)
    design_path = output / "DESIGN.json"
    write_json(design_path, {"cases": cases, "model": str(m52.MODEL)})
    registration = {
        "status": "REGISTERED_BEFORE_GPU_AND_CAUSAL_LABELS",
        "purpose": (
            "test whether path dependency multiplied by a 16-token K/V "
            "drift probe ranks single-island splice harm better than the "
            "probe alone"
        ),
        "design_sha256": _sha256(design_path),
        "cases": len(cases),
        "tasks": len({row["instance_id"] for row in cases}),
        "model": str(m52.MODEL),
        "frozen_score": {
            "probe_layer_zero_based": layer,
            "probe_head_tokens": head_tokens,
            "path_relevant_weight": path_weight,
            "path_disjoint_weight": 1.0,
            "hybrid": "probe_score * path_weight",
        },
        "frozen_inputs": {
            "m49_lock": str(M49_LOCK),
            "m49_lock_sha256": _sha256(M49_LOCK),
            "m52_result": str(M52_RESULT),
            "m52_result_sha256": _sha256(M52_RESULT),
            "m52_design_sha256": _sha256(M52_DESIGN),
            "m53_design_sha256": _sha256(M53_DESIGN),
            "m52_m53_request_overlap": 0,
        },
        "frozen_support_rule": {
            "minimum_cases": 12,
            "minimum_tasks": 5,
            "hybrid_global_JS_spearman_min": 0.30,
            "hybrid_minus_probe_global_spearman_min": 0.05,
            "hybrid_pair_ranking_accuracy_min": 0.60,
            "hybrid_minus_probe_pair_accuracy_min": 0.05,
        },
        "interpretation_limits": [
            "only 14 requests from six tasks remain unopened",
            "requests can reuse observations seen in earlier experiments",
            "the measurement computes Dense target K/V to label the proxy",
            "single-island causal ranking is not multi-island task accuracy",
        ],
    }
    write_json(output / "REGISTRATION.json", registration)
    return registration


def _complete(row: dict[str, Any]) -> bool:
    if row.get("status") != "ok" or len(row.get("candidates", [])) != 2:
        return False
    return all(
        math.isfinite(float(candidate[key]))
        for candidate in row["candidates"]
        for key in ("probe_score", "hybrid_score", "causal_splice_logit_js")
    )


def measure(output: Path) -> dict[str, Any]:
    design_path = output / "DESIGN.json"
    registration = json.loads((output / "REGISTRATION.json").read_text())
    if registration["design_sha256"] != _sha256(design_path):
        raise ValueError("design changed after registration")
    frozen = registration["frozen_score"]
    cases = json.loads(design_path.read_text())["cases"]
    destination = output / "OBSERVATIONS.jsonl"
    if destination.exists():
        raise FileExistsError(destination)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU substitution is forbidden")
    model = AutoModelForCausalLM.from_pretrained(
        m52.MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda").eval()
    theta = _model_theta(model.config)
    with destination.open("w", encoding="utf-8") as stream:
        for index, case in enumerate(cases, 1):
            source_cache, _ = _dense_source(model, case["source_input_ids"])
            target_cache, dense_logits = _dense_source(
                model, case["target_input_ids"]
            )
            measured = []
            for candidate in case["candidates"]:
                proxy = _probe_score(
                    source_cache=source_cache,
                    target_cache=target_cache,
                    candidate=candidate,
                    layer=int(frozen["probe_layer_zero_based"]),
                    head_tokens=int(frozen["probe_head_tokens"]),
                    theta=theta,
                )
                path_weight = (
                    float(frozen["path_relevant_weight"])
                    if candidate["candidate_id"] == "path_relevant"
                    else float(frozen["path_disjoint_weight"])
                )
                splice_logits = _compose_splice(
                    model=model,
                    target_ids=case["target_input_ids"],
                    target_cache=target_cache,
                    source_cache=source_cache,
                    candidates=[candidate],
                    theta=theta,
                )
                measured.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "probe_score": proxy["score"],
                        "path_weight": path_weight,
                        "hybrid_score": proxy["score"] * path_weight,
                        "position_fraction": candidate["target_start"]
                        / len(case["target_input_ids"]),
                        "causal_splice_logit_js": m52._js(
                            dense_logits, splice_logits
                        ),
                    }
                )
                del splice_logits
            row = {
                "status": "ok",
                "case_id": case["case_id"],
                "instance_id": case["instance_id"],
                "candidates": measured,
            }
            if not _complete(row):
                raise RuntimeError("case produced incomplete metrics")
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                json.dumps({"case": index, "case_id": case["case_id"]}),
                flush=True,
            )
            del source_cache, target_cache, dense_logits
            gc.collect()
            torch.cuda.empty_cache()
    status = {"status": "COMPLETE", "cases": len(cases)}
    write_json(output / "MEASUREMENT_STATUS.json", status)
    return status


def _pair_accuracy(rows: list[dict[str, Any]], score: str) -> float:
    correct = 0.0
    for row in rows:
        left, right = row["candidates"]
        score_delta = float(left[score]) - float(right[score])
        label_delta = float(left["causal_splice_logit_js"]) - float(
            right["causal_splice_logit_js"]
        )
        if score_delta == 0 or label_delta == 0:
            correct += 0.5
        elif (score_delta > 0) == (label_delta > 0):
            correct += 1
    return correct / len(rows)


def analyze(output: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (output / "OBSERVATIONS.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not rows or any(not _complete(row) for row in rows):
        raise ValueError("observations are missing or incomplete")
    labels = [
        float(candidate["causal_splice_logit_js"])
        for row in rows
        for candidate in row["candidates"]
    ]
    metrics = {}
    for score in ("probe_score", "hybrid_score"):
        scores = [
            float(candidate[score])
            for row in rows
            for candidate in row["candidates"]
        ]
        metrics[score] = {
            "global_JS_spearman": _spearman(scores, labels),
            "pair_ranking_accuracy": _pair_accuracy(rows, score),
        }
    metrics["improvement"] = {
        "global_JS_spearman": metrics["hybrid_score"]["global_JS_spearman"]
        - metrics["probe_score"]["global_JS_spearman"],
        "pair_ranking_accuracy": metrics["hybrid_score"]["pair_ranking_accuracy"]
        - metrics["probe_score"]["pair_ranking_accuracy"],
    }
    registration = json.loads((output / "REGISTRATION.json").read_text())
    gates = registration["frozen_support_rule"]
    gate_results = {
        "minimum_cases": len(rows) >= gates["minimum_cases"],
        "minimum_tasks": len({row["instance_id"] for row in rows})
        >= gates["minimum_tasks"],
        "hybrid_global_JS_spearman": metrics["hybrid_score"][
            "global_JS_spearman"
        ]
        >= gates["hybrid_global_JS_spearman_min"],
        "hybrid_minus_probe_global_spearman": metrics["improvement"][
            "global_JS_spearman"
        ]
        >= gates["hybrid_minus_probe_global_spearman_min"],
        "hybrid_pair_ranking_accuracy": metrics["hybrid_score"][
            "pair_ranking_accuracy"
        ]
        >= gates["hybrid_pair_ranking_accuracy_min"],
        "hybrid_minus_probe_pair_accuracy": metrics["improvement"][
            "pair_ranking_accuracy"
        ]
        >= gates["hybrid_minus_probe_pair_accuracy_min"],
    }
    decision = "SUPPORTED" if all(gate_results.values()) else "NOT_SUPPORTED"
    value = {
        "status": "COMPLETE",
        "decision": decision,
        "cases": len(rows),
        "tasks": len({row["instance_id"] for row in rows}),
        "metrics": metrics,
        "frozen_gate_results": gate_results,
        "next_step": (
            "validate per-island hybrid admission on new task-disjoint trajectories"
            if decision == "SUPPORTED"
            else "do not implement this multiplicative hybrid in SGLang"
        ),
        "scope": "unopened 14-request causal proxy audit, not runtime accuracy",
    }
    write_json(output / "RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.output)
    elif args.command == "measure":
        value = measure(args.output)
    else:
        value = analyze(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
