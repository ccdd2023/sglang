#!/usr/bin/env python3
"""Test path dependency as a coding-specific lossy-reuse risk signal.

For each frozen Dense coding-agent target prompt, M52 pairs two exact 128-token
grounded observations from the same source/target transition:

* path-relevant: its repository path is referenced by the latest completed
  coding interaction;
* path-disjoint: its path is not referenced by that interaction.

The experiment measures Dense target-query attention, K/V drift, and physical
splice harm.  It separates evidence that path overlap predicts *dependency*
from evidence that it is strong enough to drive a Dense-protection guard.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.multi_workflow.coding_reuse_policy import (
    is_successful_readonly_evidence,
    repository_paths,
)
from benchmark.multi_workflow.measure_sessiongraph_atlas import _js
from benchmark.multi_workflow.motivate_v48_attention_kv_risk import (
    _candidate_internal_metrics,
    _compose_splice,
    _dense_source,
    _first_token_nll,
    _model_theta,
    _target_forward,
)
from benchmark.multi_workflow.motivate_v50_coding_provenance import (
    MODEL,
    _balanced_select,
    _message_candidate,
    _render_rolling,
    _sha256,
    _token_ids_hash,
    _trajectory_paths,
    _turn_groups,
)
from benchmark.multi_workflow.run_bridge_reuse_pilot import write_json


ROOT = Path("/home/gfy/CodeMAS_Project")
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_m52_path_dependency_20260805"
    / "matched20"
)
MAX_POSITION_DISTANCE_FRACTION = 0.25
RANDOM_SEED = 202608052


def _candidate_pool(tokenizer: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for instance_id, path in _trajectory_paths().items():
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        messages = trajectory["messages"]
        base = messages[:2]
        groups = _turn_groups(messages[2:])
        for target_completed in range(7, len(groups)):
            latest_paths = repository_paths(groups[target_completed - 1])
            if not latest_paths:
                continue
            source_ids, source_spans = _render_rolling(
                tokenizer, base, groups[: target_completed - 1]
            )
            target_ids, target_spans = _render_rolling(
                tokenizer, base, groups[:target_completed]
            )
            if len(source_ids) > 30_000 or len(target_ids) > 30_000:
                continue
            relevant: list[dict[str, Any]] = []
            disjoint: list[dict[str, Any]] = []
            group_indices = sorted(
                {key[0] for key in source_spans}
                & {key[0] for key in target_spans}
            )
            for group_index in group_indices:
                group = groups[group_index]
                paths = repository_paths(group)
                if not paths or not is_successful_readonly_evidence(group):
                    continue
                category = (
                    "path_relevant" if paths & latest_paths else "path_disjoint"
                )
                for message_index, message in enumerate(group):
                    if message.get("role") != "tool":
                        continue
                    candidate = _message_candidate(
                        category=category,
                        group_index=group_index,
                        message_index=message_index,
                        source_ids=source_ids,
                        source_spans=source_spans,
                        target_ids=target_ids,
                        target_spans=target_spans,
                    )
                    if candidate:
                        candidate["candidate_id"] = category
                        candidate["repository_paths"] = sorted(paths)
                        (relevant if category == "path_relevant" else disjoint).append(
                            candidate
                        )
            if not relevant or not disjoint:
                continue
            distance, selected_relevant, selected_disjoint = min(
                (
                    abs(left["target_start"] - right["target_start"]),
                    left,
                    right,
                )
                for left in relevant
                for right in disjoint
            )
            distance_fraction = distance / len(target_ids)
            if distance_fraction > MAX_POSITION_DISTANCE_FRACTION:
                continue
            answer = str(groups[target_completed][0].get("content") or "")
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            if not answer_ids:
                continue
            case_id = f"{instance_id}-path-q{target_completed + 1}"
            rows.append(
                {
                    "answer_first_token_id": int(answer_ids[0]),
                    "case_id": case_id,
                    "candidates": [selected_relevant, selected_disjoint],
                    "instance_id": instance_id,
                    "latest_paths": sorted(latest_paths),
                    "position_distance_fraction": distance_fraction,
                    "position_distance_tokens": distance,
                    "source_input_ids": source_ids,
                    "source_prompt_hash": _token_ids_hash(source_ids),
                    "target_input_ids": target_ids,
                    "target_prompt_hash": _token_ids_hash(target_ids),
                    "target_request_index": target_completed + 1,
                    "trajectory_path": str(path),
                    "trajectory_sha256": _sha256(path),
                }
            )
    return rows


def prepare(output: Path, limit: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    eligible = _candidate_pool(tokenizer)
    selected = _balanced_select(eligible, limit)
    if len(selected) < limit:
        raise ValueError(f"only {len(selected)} eligible cases for requested {limit}")
    output.mkdir(parents=True)
    design_path = output / "DESIGN.json"
    write_json(
        design_path,
        {
            "cases": selected,
            "eligible_cases_before_sampling": len(eligible),
            "model": str(MODEL),
        },
    )
    registration = {
        "status": "REGISTERED_BEFORE_GPU",
        "purpose": (
            "test whether online path overlap with the latest coding "
            "interaction predicts model dependency and causal reuse harm"
        ),
        "design_sha256": _sha256(design_path),
        "cases": len(selected),
        "tasks": len({row["instance_id"] for row in selected}),
        "eligible_cases_before_sampling": len(eligible),
        "model": str(MODEL),
        "matching_contract": {
            "same_source_target_prompt_within_pair": True,
            "candidate_tokens_each": 128,
            "candidate_token_identical_source_target": True,
            "latest_completed_group_only": True,
            "maximum_position_distance_fraction": MAX_POSITION_DISTANCE_FRACTION,
            "one_case_per_task_before_second_case": True,
        },
        "frozen_dependency_support_rule": {
            "path_relevant_higher_attention_pair_fraction_min": 0.60,
            "position_adjusted_attention_ratio_min": 1.25,
            "minimum_complete_cases": 16,
            "minimum_tasks": 8,
        },
        "frozen_guard_support_rule": {
            "path_relevant_higher_JS_pair_fraction_min": 0.60,
            "position_adjusted_JS_ratio_min": 1.10,
        },
        "interpretation_limits": [
            "Dense target attention is oracle motivation, not controller overhead",
            "path extraction is mechanical and may miss aliases/symbol dependencies",
            "position is regression-adjusted because real path blocks are not co-located",
            "offline logit harm is not SWE-bench accuracy or TTFT",
        ],
    }
    write_json(output / "REGISTRATION.json", registration)
    return registration


def _complete(row: Mapping[str, Any]) -> bool:
    if row.get("status") != "ok" or len(row.get("candidates", [])) != 2:
        return False
    values = []
    for candidate in row["candidates"]:
        for metric in (
            "attention_mean",
            "kv_cosine_drift_mean",
            "risk_product_mean",
            "causal_splice_logit_js",
        ):
            values.append(float(candidate[metric]))
    return all(math.isfinite(value) for value in values)


def measure(output: Path, max_cases: int) -> dict[str, Any]:
    design_path = output / "DESIGN.json"
    registration = json.loads((output / "REGISTRATION.json").read_text())
    if registration["design_sha256"] != _sha256(design_path):
        raise ValueError("design changed after registration")
    design = json.loads(design_path.read_text())
    cases = design["cases"][:max_cases] if max_cases > 0 else design["cases"]
    observations_path = output / "OBSERVATIONS.jsonl"
    completed: set[str] = set()
    if observations_path.exists():
        for line in observations_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if _complete(row):
                    completed.add(str(row["case_id"]))
    pending = [row for row in cases if row["case_id"] not in completed]
    if not pending:
        return {"status": "COMPLETE", "cases": len(cases), "new_cases": 0}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU substitution is forbidden")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda").eval()
    theta = _model_theta(model.config)
    written = 0
    errors = []
    for index, case in enumerate(pending, 1):
        try:
            source_cache, _ = _dense_source(model, case["source_input_ids"])
            target_cache, dense_logits, attention = _target_forward(
                model=model,
                target_ids=case["target_input_ids"],
                candidates=case["candidates"],
            )
            dense_nll = _first_token_nll(
                dense_logits, int(case["answer_first_token_id"])
            )
            measured = []
            for candidate in case["candidates"]:
                candidate_id = str(candidate["candidate_id"])
                internal = _candidate_internal_metrics(
                    candidate=candidate,
                    attention=attention[candidate_id],
                    source_cache=source_cache,
                    target_cache=target_cache,
                    theta=theta,
                    num_attention_heads=int(model.config.num_attention_heads),
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
                measured.append(
                    {
                        **candidate,
                        **internal,
                        "position_fraction": candidate["target_start"]
                        / len(case["target_input_ids"]),
                        "prefix_shift_tokens": candidate["target_start"]
                        - candidate["source_start"],
                        "causal_splice_logit_js": _js(
                            dense_logits, splice_logits
                        ),
                        "causal_splice_top1_changed": int(dense_logits.argmax())
                        != int(splice_logits.argmax()),
                        "answer_first_token_nll_delta": splice_nll - dense_nll,
                    }
                )
                del splice_logits
            row = {
                "status": "ok",
                "case_id": case["case_id"],
                "instance_id": case["instance_id"],
                "source_tokens": len(case["source_input_ids"]),
                "target_tokens": len(case["target_input_ids"]),
                "candidates": measured,
            }
            if not _complete(row):
                raise RuntimeError("case produced incomplete/non-finite metrics")
            with observations_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
            by_id = {candidate["candidate_id"]: candidate for candidate in measured}
            print(
                json.dumps(
                    {
                        "case": index,
                        "case_id": case["case_id"],
                        "pending": len(pending),
                        "relevant_attention": by_id["path_relevant"]["attention_mean"],
                        "disjoint_attention": by_id["path_disjoint"]["attention_mean"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del source_cache, target_cache, dense_logits, attention
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as error:
            errors.append(
                {
                    "case_id": case["case_id"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(json.dumps(errors[-1], sort_keys=True), flush=True)
            break
    status = {
        "status": "COMPLETE" if not errors and written == len(pending) else "PARTIAL",
        "selected_cases": len(cases),
        "previously_completed_cases": len(completed),
        "new_cases": written,
        "errors": errors,
    }
    write_json(output / "MEASUREMENT_STATUS.json", status)
    return status


def _adjusted_ratio(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    relevant = [
        next(value for value in row["candidates"] if value["candidate_id"] == "path_relevant")
        for row in rows
    ]
    disjoint = [
        next(value for value in row["candidates"] if value["candidate_id"] == "path_disjoint")
        for row in rows
    ]
    y = np.asarray(
        [
            math.log(float(left[metric]) + 1e-10)
            - math.log(float(right[metric]) + 1e-10)
            for left, right in zip(relevant, disjoint, strict=True)
        ]
    )
    x = np.asarray(
        [
            [
                float(left["position_fraction"]) - float(right["position_fraction"]),
                (
                    float(left["prefix_shift_tokens"])
                    - float(right["prefix_shift_tokens"])
                )
                / 1000,
            ]
            for left, right in zip(relevant, disjoint, strict=True)
        ]
    )
    design = np.column_stack((np.ones(len(rows)), x))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return {
        "position_adjusted_geometric_ratio": math.exp(float(coefficients[0])),
        "position_difference_coefficient": float(coefficients[1]),
        "prefix_shift_difference_per_1000_coefficient": float(coefficients[2]),
    }


def _metric_summary(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    relevant = [
        next(value for value in row["candidates"] if value["candidate_id"] == "path_relevant")
        for row in rows
    ]
    disjoint = [
        next(value for value in row["candidates"] if value["candidate_id"] == "path_disjoint")
        for row in rows
    ]
    left = [float(row[metric]) for row in relevant]
    right = [float(row[metric]) for row in disjoint]
    wins = sum(a > b for a, b in zip(left, right, strict=True))
    return {
        "path_relevant_mean": statistics.fmean(left),
        "path_relevant_median": statistics.median(left),
        "path_disjoint_mean": statistics.fmean(right),
        "path_disjoint_median": statistics.median(right),
        "path_relevant_higher_pair_fraction": wins / len(rows),
        "one_sided_sign_probability": sum(
            math.comb(len(rows), value) for value in range(wins, len(rows) + 1)
        )
        / (2 ** len(rows)),
        **_adjusted_ratio(rows, metric),
    }


def analyze(output: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (output / "OBSERVATIONS.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not rows or any(not _complete(row) for row in rows):
        raise ValueError("observations are missing or incomplete")
    metrics = {
        metric: _metric_summary(rows, metric)
        for metric in (
            "attention_mean",
            "kv_cosine_drift_mean",
            "risk_product_mean",
            "causal_splice_logit_js",
        )
    }
    registration = json.loads((output / "REGISTRATION.json").read_text())
    dependency = registration["frozen_dependency_support_rule"]
    guard = registration["frozen_guard_support_rule"]
    attention = metrics["attention_mean"]
    js = metrics["causal_splice_logit_js"]
    dependency_gates = {
        "minimum_complete_cases": len(rows) >= dependency["minimum_complete_cases"],
        "minimum_tasks": len({row["instance_id"] for row in rows})
        >= dependency["minimum_tasks"],
        "higher_attention_pair_fraction": attention[
            "path_relevant_higher_pair_fraction"
        ]
        >= dependency["path_relevant_higher_attention_pair_fraction_min"],
        "position_adjusted_attention_ratio": attention[
            "position_adjusted_geometric_ratio"
        ]
        >= dependency["position_adjusted_attention_ratio_min"],
    }
    guard_gates = {
        "higher_JS_pair_fraction": js["path_relevant_higher_pair_fraction"]
        >= guard["path_relevant_higher_JS_pair_fraction_min"],
        "position_adjusted_JS_ratio": js["position_adjusted_geometric_ratio"]
        >= guard["position_adjusted_JS_ratio_min"],
    }
    dependency_decision = "SUPPORTED" if all(dependency_gates.values()) else "NOT_SUPPORTED"
    guard_decision = (
        "SUPPORTED"
        if dependency_decision == "SUPPORTED" and all(guard_gates.values())
        else "NOT_SUPPORTED"
    )
    value = {
        "status": "COMPLETE",
        "dependency_decision": dependency_decision,
        "dense_protection_guard_decision": guard_decision,
        "cases": len(rows),
        "tasks": len({row["instance_id"] for row in rows}),
        "metrics": metrics,
        "dependency_gate_results": dependency_gates,
        "guard_gate_results": guard_gates,
        "scope": (
            "path-dependency oracle motivation; not a runtime selector, "
            "functional accuracy result, or latency result"
        ),
        "next_step": (
            "validate an online path/symbol dependency guard on disjoint tasks"
            if guard_decision == "SUPPORTED"
            else "do not promote path overlap as a Dense-protection guard"
        ),
    }
    write_json(output / "RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--limit", type=int, default=20)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    measure_parser.add_argument("--max-cases", type=int, default=0)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.output, args.limit)
    elif args.command == "measure":
        value = measure(args.output, args.max_cases)
    else:
        value = analyze(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
