#!/usr/bin/env python3
"""Calibrate and independently validate a cheap K/V probe for M48 risk.

M48 showed that full-target attention x K/V drift is an informative oracle,
but it cannot run before reuse without computing the target.  M49 tests a
deployable approximation: recompute only the first H tokens of each otherwise
eligible V46 island, compare their K/V to the cached source at one layer, and
use the maximum mean K/V cosine deviation as a request-level abstention score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.multi_workflow.motivate_v48_attention_kv_risk import (
    MODEL,
    _compose_splice,
    _cosine_deviation_by_head,
    _dense_source,
    _first_token_nll,
    _js,
    _measurement_complete,
    _model_theta,
    _rope_shift,
    _spearman,
    prepare_case,
)
from benchmark.multi_workflow.run_bridge_reuse_pilot import write_json


ROOT = Path("/home/gfy/CodeMAS_Project")
DEFAULT_OUTPUT = (
    ROOT / "kvflow-artifacts/impactkv_m49_probe_proxy_20260805"
)
DEV_M48 = (
    ROOT
    / "kvflow-artifacts/impactkv_m48_attention_kv_risk_20260805/full50"
)
HOLDOUT_WORKLOAD = (
    ROOT
    / "kvflow-artifacts/impactkv_codemas_v2_controlled_sota_20260729"
    / "v66_final_repobench_holdout100/WORKLOAD.json"
)
LAYERS = (8, 17, 26)
HEAD_TOKENS = (8, 16, 32, 64)
RANDOM_SEED = 20260805


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def prepare(
    *,
    output: Path,
    dev_m48: Path,
    holdout_workload: Path,
) -> dict[str, Any]:
    dev_design = dev_m48 / "DESIGN.json"
    dev_observations = dev_m48 / "OBSERVATIONS.jsonl"
    if not dev_design.exists() or not dev_observations.exists():
        raise FileNotFoundError("completed M48 development artifacts are required")
    observations = _read_jsonl(dev_observations)
    if len(observations) != 50 or any(
        not _measurement_complete(row) for row in observations
    ):
        raise ValueError("M48 development observations are incomplete")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    workload = json.loads(holdout_workload.read_text())
    holdout_cases = []
    skipped_cases = []
    for case in workload["cases"]:
        try:
            prepared = prepare_case(tokenizer, case)
        except ValueError as error:
            skipped_cases.append(
                {"case_id": str(case["case_id"]), "reason": str(error)}
            )
            continue
        holdout_cases.append(prepared)
        if len(holdout_cases) == 50:
            break
    if len(holdout_cases) != 50:
        raise ValueError("independent workload has fewer than 50 eligible cases")
    output.mkdir(parents=True, exist_ok=True)
    holdout_design = output / "HOLDOUT_DESIGN.json"
    write_json(
        holdout_design,
        {
            "cases": holdout_cases,
            "dataset": "RepoBench-P independent holdout50",
            "model": MODEL,
        },
    )
    registration = {
        "status": "REGISTERED_BEFORE_M49_GPU",
        "purpose": (
            "find a small Dense probe-head K/V drift proxy for the M48 oracle "
            "and validate it on 50 disjoint RepoBench-P cases"
        ),
        "development": {
            "cases": 50,
            "m48_design": str(dev_design),
            "m48_design_sha256": _sha(dev_design),
            "m48_observations": str(dev_observations),
            "m48_observations_sha256": _sha(dev_observations),
        },
        "holdout": {
            "cases": len(holdout_cases),
            "workload": str(holdout_workload),
            "workload_sha256": _sha(holdout_workload),
            "design": str(holdout_design),
            "design_sha256": _sha(holdout_design),
            "overlap_with_development": 0,
            "selection": (
                "first 50 cases in source order satisfying the unchanged "
                "three-by-512-token V46 candidate contract"
            ),
            "skipped_before_frozen_50": skipped_cases,
        },
        "grid": {
            "layers_zero_based": list(LAYERS),
            "probe_head_tokens": list(HEAD_TOKENS),
            "score": "max(mean RoPE-corrected K cosine drift, mean V cosine drift)",
        },
        "configuration_selection": (
            "maximize development request-level Spearman between the maximum "
            "probe score over V46's three islands and composed-logit JS; all "
            "configurations within 0.02 of the best use the smallest H, then "
            "the shallower layer"
        ),
        "threshold": (
            "freeze the development 90th percentile of the selected request "
            "score before holdout GPU measurement"
        ),
        "holdout_gates": {
            "single_island_mean_within_case_js_spearman_min": 0.30,
            "request_composed_js_spearman_min": 0.30,
            "high_risk_to_low_risk_mean_js_ratio_min": 1.50,
        },
        "scope": (
            "motivation for an online V46 guard; no SGLang speed or functional "
            "accuracy claim until a later runtime experiment"
        ),
    }
    write_json(output / "REGISTRATION.json", registration)
    return registration


def _probe_score(
    *,
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    candidate: Mapping[str, Any],
    layer: int,
    head_tokens: int,
    theta: float,
) -> dict[str, float]:
    head = min(head_tokens, int(candidate["length"]))
    source_start = int(candidate["source_start"])
    target_start = int(candidate["target_start"])
    delta = target_start - source_start
    source_key, source_value = source_cache[layer]
    target_key, target_value = target_cache[layer]
    source_key = _rope_shift(
        source_key[:, source_start : source_start + head], delta, theta
    )
    target_key = target_key[:, target_start : target_start + head]
    source_value = source_value[:, source_start : source_start + head]
    target_value = target_value[:, target_start : target_start + head]
    key = float(_cosine_deviation_by_head(source_key, target_key).mean())
    value = float(_cosine_deviation_by_head(source_value, target_value).mean())
    return {"k_deviation": key, "v_deviation": value, "score": max(key, value)}


def _load_model(local_files_only: bool) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU substitution is forbidden")
    return AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
        local_files_only=local_files_only,
    ).eval()


def measure_development(
    *,
    output: Path,
    dev_m48: Path,
    local_files_only: bool,
) -> dict[str, Any]:
    registration = json.loads((output / "REGISTRATION.json").read_text())
    design_path = dev_m48 / "DESIGN.json"
    observations_path = dev_m48 / "OBSERVATIONS.jsonl"
    if (
        _sha(design_path)
        != registration["development"]["m48_design_sha256"]
        or _sha(observations_path)
        != registration["development"]["m48_observations_sha256"]
    ):
        raise ValueError("M48 development input changed after M49 registration")
    cases = json.loads(design_path.read_text())["cases"]
    labels = {row["case_id"]: row for row in _read_jsonl(observations_path)}
    destination = output / "DEV_PROXIES.jsonl"
    if destination.exists():
        raise FileExistsError(destination)
    model = _load_model(local_files_only)
    theta = _model_theta(model.config)
    with destination.open("w", encoding="utf-8") as stream:
        for index, case in enumerate(cases, 1):
            source_cache, _ = _dense_source(model, case["source_input_ids"])
            target_cache, _ = _dense_source(model, case["target_input_ids"])
            label_by_id = {
                row["candidate_id"]: row
                for row in labels[case["case_id"]]["candidates"]
            }
            candidates = []
            for candidate in case["candidates"]:
                configurations = []
                for layer in LAYERS:
                    for head in HEAD_TOKENS:
                        configurations.append(
                            {
                                "layer": layer,
                                "head_tokens": head,
                                **_probe_score(
                                    source_cache=source_cache,
                                    target_cache=target_cache,
                                    candidate=candidate,
                                    layer=layer,
                                    head_tokens=head,
                                    theta=theta,
                                ),
                            }
                        )
                causal = label_by_id[candidate["candidate_id"]]
                candidates.append(
                    {
                        **candidate,
                        "configurations": configurations,
                        "causal_splice_logit_js": causal[
                            "causal_splice_logit_js"
                        ],
                        "answer_first_token_nll_delta": causal[
                            "answer_first_token_nll_delta"
                        ],
                    }
                )
            row = {
                "case_id": case["case_id"],
                "v46_candidate_ids": case["v46_candidate_ids"],
                "v46_composed": labels[case["case_id"]]["v46_composed"],
                "candidates": candidates,
            }
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps({"case": index, "case_id": case["case_id"]}), flush=True)
            del source_cache, target_cache
            torch.cuda.empty_cache()
    status = {"status": "COMPLETE", "cases": len(cases), "output": str(destination)}
    write_json(output / "DEV_MEASUREMENT_STATUS.json", status)
    return status


def _configuration(
    candidate: Mapping[str, Any], layer: int, head_tokens: int
) -> Mapping[str, Any]:
    return next(
        row
        for row in candidate["configurations"]
        if int(row["layer"]) == layer and int(row["head_tokens"]) == head_tokens
    )


def _finite_mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else math.nan


def _config_metrics(
    rows: Sequence[Mapping[str, Any]], layer: int, head_tokens: int
) -> dict[str, Any]:
    single_scores, single_js = [], []
    within_case = []
    request_scores, request_js, request_nll = [], [], []
    for row in rows:
        scores = [
            float(_configuration(candidate, layer, head_tokens)["score"])
            for candidate in row["candidates"]
        ]
        js = [float(candidate["causal_splice_logit_js"]) for candidate in row["candidates"]]
        single_scores.extend(scores)
        single_js.extend(js)
        within_case.append(_spearman(scores, js))
        selected = [
            candidate
            for candidate in row["candidates"]
            if candidate["candidate_id"] in set(row["v46_candidate_ids"])
        ]
        request_scores.append(
            max(
                float(_configuration(candidate, layer, head_tokens)["score"])
                for candidate in selected
            )
        )
        request_js.append(float(row["v46_composed"]["causal_splice_logit_js"]))
        request_nll.append(
            float(row["v46_composed"]["answer_first_token_nll_delta"])
        )
    return {
        "layer": layer,
        "head_tokens": head_tokens,
        "probe_fraction": head_tokens / 512,
        "single_global_js_spearman": _spearman(single_scores, single_js),
        "single_mean_within_case_js_spearman": _finite_mean(within_case),
        "request_composed_js_spearman": _spearman(request_scores, request_js),
        "request_composed_nll_spearman": _spearman(request_scores, request_nll),
        "request_scores": request_scores,
    }


def lock(output: Path) -> dict[str, Any]:
    registration = json.loads((output / "REGISTRATION.json").read_text())
    rows = _read_jsonl(output / "DEV_PROXIES.jsonl")
    if len(rows) != registration["development"]["cases"]:
        raise ValueError("development proxy coverage is incomplete")
    evaluated = [
        _config_metrics(rows, layer, head)
        for layer in LAYERS
        for head in HEAD_TOKENS
    ]
    best_correlation = max(row["request_composed_js_spearman"] for row in evaluated)
    near_best = [
        row
        for row in evaluated
        if row["request_composed_js_spearman"] >= best_correlation - 0.02
    ]
    chosen = min(near_best, key=lambda row: (row["head_tokens"], row["layer"]))
    threshold = float(np.quantile(chosen["request_scores"], 0.90))
    chosen = {key: value for key, value in chosen.items() if key != "request_scores"}
    value = {
        "status": "LOCKED_BEFORE_HOLDOUT_GPU",
        "chosen": chosen,
        "request_risk_threshold": threshold,
        "evaluated_configurations": [
            {key: value for key, value in row.items() if key != "request_scores"}
            for row in evaluated
        ],
        "registration_sha256": _sha(output / "REGISTRATION.json"),
        "development_proxies_sha256": _sha(output / "DEV_PROXIES.jsonl"),
        "holdout_design_sha256": registration["holdout"]["design_sha256"],
    }
    write_json(output / "PROXY_LOCK.json", value)
    return value


def measure_holdout(output: Path, local_files_only: bool) -> dict[str, Any]:
    lock_value = json.loads((output / "PROXY_LOCK.json").read_text())
    registration = json.loads((output / "REGISTRATION.json").read_text())
    if lock_value["status"] != "LOCKED_BEFORE_HOLDOUT_GPU":
        raise ValueError("proxy is not locked")
    design_path = output / "HOLDOUT_DESIGN.json"
    if _sha(design_path) != lock_value["holdout_design_sha256"]:
        raise ValueError("holdout design changed after lock")
    destination = output / "HOLDOUT_OBSERVATIONS.jsonl"
    if destination.exists():
        raise FileExistsError(destination)
    cases = json.loads(design_path.read_text())["cases"]
    layer = int(lock_value["chosen"]["layer"])
    head = int(lock_value["chosen"]["head_tokens"])
    model = _load_model(local_files_only)
    theta = _model_theta(model.config)
    with destination.open("w", encoding="utf-8") as stream:
        for index, case in enumerate(cases, 1):
            source_cache, _ = _dense_source(model, case["source_input_ids"])
            target_cache, dense_logits = _dense_source(model, case["target_input_ids"])
            dense_nll = _first_token_nll(
                dense_logits, int(case["answer_first_token_id"])
            )
            candidates = []
            by_id = {}
            for candidate in case["candidates"]:
                proxy = _probe_score(
                    source_cache=source_cache,
                    target_cache=target_cache,
                    candidate=candidate,
                    layer=layer,
                    head_tokens=head,
                    theta=theta,
                )
                splice_logits = _compose_splice(
                    model=model,
                    target_ids=case["target_input_ids"],
                    target_cache=target_cache,
                    source_cache=source_cache,
                    candidates=[candidate],
                    theta=theta,
                )
                splice_nll = _first_token_nll(
                    splice_logits, int(case["answer_first_token_id"])
                )
                measured = {
                    **candidate,
                    **proxy,
                    "causal_splice_logit_js": _js(dense_logits, splice_logits),
                    "answer_first_token_nll_delta": splice_nll - dense_nll,
                }
                candidates.append(measured)
                by_id[candidate["candidate_id"]] = candidate
                del splice_logits
            selected = [by_id[value] for value in case["v46_candidate_ids"]]
            v46_logits = _compose_splice(
                model=model,
                target_ids=case["target_input_ids"],
                target_cache=target_cache,
                source_cache=source_cache,
                candidates=selected,
                theta=theta,
            )
            v46_nll = _first_token_nll(
                v46_logits, int(case["answer_first_token_id"])
            )
            row = {
                "case_id": case["case_id"],
                "v46_candidate_ids": case["v46_candidate_ids"],
                "candidates": candidates,
                "v46_composed": {
                    "causal_splice_logit_js": _js(dense_logits, v46_logits),
                    "answer_first_token_nll_delta": v46_nll - dense_nll,
                },
            }
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps({"case": index, "case_id": case["case_id"]}), flush=True)
            del source_cache, target_cache, dense_logits, v46_logits
            torch.cuda.empty_cache()
    status = {
        "status": "COMPLETE",
        "cases": len(cases),
        "layer": layer,
        "head_tokens": head,
        "output": str(destination),
    }
    write_json(output / "HOLDOUT_MEASUREMENT_STATUS.json", status)
    return status


def evaluate(output: Path) -> dict[str, Any]:
    registration = json.loads((output / "REGISTRATION.json").read_text())
    lock_value = json.loads((output / "PROXY_LOCK.json").read_text())
    rows = _read_jsonl(output / "HOLDOUT_OBSERVATIONS.jsonl")
    if len(rows) != registration["holdout"]["cases"]:
        raise ValueError("holdout coverage is incomplete")
    single_scores, single_js = [], []
    within_case = []
    request_scores, request_js, request_nll = [], [], []
    for row in rows:
        scores = [float(candidate["score"]) for candidate in row["candidates"]]
        js = [float(candidate["causal_splice_logit_js"]) for candidate in row["candidates"]]
        single_scores.extend(scores)
        single_js.extend(js)
        within_case.append(_spearman(scores, js))
        selected = [
            candidate
            for candidate in row["candidates"]
            if candidate["candidate_id"] in set(row["v46_candidate_ids"])
        ]
        request_scores.append(max(float(candidate["score"]) for candidate in selected))
        request_js.append(float(row["v46_composed"]["causal_splice_logit_js"]))
        request_nll.append(
            float(row["v46_composed"]["answer_first_token_nll_delta"])
        )
    threshold = float(lock_value["request_risk_threshold"])
    high_js = [js for score, js in zip(request_scores, request_js) if score >= threshold]
    low_js = [js for score, js in zip(request_scores, request_js) if score < threshold]
    metrics = {
        "single_island_global_js_spearman": _spearman(single_scores, single_js),
        "single_island_mean_within_case_js_spearman": _finite_mean(within_case),
        "request_composed_js_spearman": _spearman(request_scores, request_js),
        "request_composed_nll_spearman": _spearman(request_scores, request_nll),
        "high_risk_requests": len(high_js),
        "low_risk_requests": len(low_js),
        "high_risk_mean_composed_js": statistics.fmean(high_js) if high_js else math.nan,
        "low_risk_mean_composed_js": statistics.fmean(low_js) if low_js else math.nan,
        "high_to_low_mean_js_ratio": (
            statistics.fmean(high_js) / statistics.fmean(low_js)
            if high_js and low_js and statistics.fmean(low_js) > 0
            else math.nan
        ),
    }
    gates = registration["holdout_gates"]
    passed = (
        metrics["single_island_mean_within_case_js_spearman"]
        >= gates["single_island_mean_within_case_js_spearman_min"]
        and metrics["request_composed_js_spearman"]
        >= gates["request_composed_js_spearman_min"]
        and metrics["high_to_low_mean_js_ratio"]
        >= gates["high_risk_to_low_risk_mean_js_ratio_min"]
    )
    value = {
        "status": "PASSED_PROXY_MOTIVATION" if passed else "FALSIFIED_PROXY",
        "passed": passed,
        "chosen": lock_value["chosen"],
        "request_risk_threshold": threshold,
        "metrics": metrics,
        "gates": gates,
        "next_step": (
            "implement SGLang canary with this fixed head/layer/threshold"
            if passed
            else "do not implement this probe in SGLang"
        ),
        "scope": "independent causal-proxy validation, not functional accuracy",
    }
    write_json(output / "FINAL_RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--dev-m48", type=Path, default=DEV_M48)
    prepare_parser.add_argument(
        "--holdout-workload", type=Path, default=HOLDOUT_WORKLOAD
    )
    dev_parser = subparsers.add_parser("measure-development")
    dev_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    dev_parser.add_argument("--dev-m48", type=Path, default=DEV_M48)
    dev_parser.add_argument("--local-files-only", action="store_true")
    lock_parser = subparsers.add_parser("lock")
    lock_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    holdout_parser = subparsers.add_parser("measure-holdout")
    holdout_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    holdout_parser.add_argument("--local-files-only", action="store_true")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(
            output=args.output,
            dev_m48=args.dev_m48,
            holdout_workload=args.holdout_workload,
        )
    elif args.command == "measure-development":
        value = measure_development(
            output=args.output,
            dev_m48=args.dev_m48,
            local_files_only=args.local_files_only,
        )
    elif args.command == "lock":
        value = lock(args.output)
    elif args.command == "measure-holdout":
        value = measure_holdout(args.output, args.local_files_only)
    else:
        value = evaluate(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
