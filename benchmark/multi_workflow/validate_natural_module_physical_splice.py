#!/usr/bin/env python3
"""Gate natural modules with causal K/V splices after Attention passes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from benchmark.multi_workflow import motivate_module_conditioned_attention_kv as old
from benchmark.multi_workflow.motivate_attention_kv_perturbation_bound import (
    _head_query_bound_metrics,
    _js,
    _model_theta,
    _physical_splice_logits,
    _rope_shift,
)
from benchmark.multi_workflow.motivate_natural_module_attention import (
    BOOTSTRAP_SAMPLES,
    PRIMARY_TYPES,
    PROBE_LAYERS,
)


ROOT = Path("/home/gfy/CodeMAS_Project")
DEFAULT_ATTENTION = (
    ROOT
    / "kvflow-artifacts/impactkv_natural_module_attention_20260808/"
    "attention_initial20"
)
DEFAULT_OUTPUT = DEFAULT_ATTENTION / "physical_splice"
STRONG_GATE_POLICY = "original-strong"
MINIMAL_RELIABLE_GATE_POLICY = "minimal-reliable"
GATE_POLICIES = (STRONG_GATE_POLICY, MINIMAL_RELIABLE_GATE_POLICY)
MIN_PAIRS = 32
MIN_TASKS = 8
MAX_PER_TYPE = 32
QUERY_CHUNK = 64
SELECTION_SALT = "natural-module-physical-splice-20260808-v1"
BOOTSTRAP_SEED = 2026080805


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable(value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{value}".encode()).hexdigest()


def select_balanced(
    candidates: Sequence[Mapping[str, Any]], limit: int = MAX_PER_TYPE
) -> list[str]:
    """Round-robin tasks without consulting physical-splice outcomes."""

    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_task[str(candidate["instance_id"])].append(candidate)
    for task in by_task:
        by_task[task].sort(key=lambda row: _stable(str(row["candidate_key"])))
    selected: list[str] = []
    cursor = 0
    while len(selected) < limit:
        added = False
        for task in sorted(by_task, key=_stable):
            if cursor < len(by_task[task]):
                selected.append(str(by_task[task][cursor]["candidate_key"]))
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        cursor += 1
    return selected


def attention_admission_gates(
    attention_result: Mapping[str, Any], gate_policy: str
) -> dict[str, bool]:
    """Decide whether Attention may open a separately registered physical stage."""

    if gate_policy == STRONG_GATE_POLICY:
        return {
            "original_attention_status_pass": attention_result.get("status") == "PASS",
            "all_original_attention_gates_pass": all(
                bool(value) for value in attention_result.get("gates", {}).values()
            ),
        }
    if gate_policy != MINIMAL_RELIABLE_GATE_POLICY:
        raise ValueError(f"unknown gate policy: {gate_policy}")
    gates: dict[str, bool] = {}
    for module_type in PRIMARY_TYPES:
        result = attention_result["type_results"][module_type]
        gates[f"{module_type}_raw_paired_direction_above_chance"] = (
            float(result["raw_natural_to_boundary_paired_direction"]) > 0.5
        )
        gates[f"{module_type}_task_bootstrap_q025_above_1"] = (
            float(result["task_bootstrap_adjusted_ratio_q025_q50_q975"][0]) > 1.0
        )
    gates["enhanced_prediction_bootstrap_q025_above_0"] = (
        float(
            attention_result["prediction"][
                "task_bootstrap_improvement_q025_q50_q975"
            ][0]
        )
        > 0.0
    )
    return gates


def freeze(
    attention: Path,
    output: Path,
    gate_policy: str = STRONG_GATE_POLICY,
) -> dict[str, Any]:
    destination = output / "REGISTRATION.json"
    if destination.exists():
        value = _read_json(destination)
        if value.get("gate_policy", STRONG_GATE_POLICY) != gate_policy:
            raise ValueError("existing registration uses a different gate policy")
        return value
    if (output / "OBSERVATIONS.jsonl").exists() or (output / "RESULT.json").exists():
        raise RuntimeError("physical outcomes exist without a frozen registration")
    attention_result = _read_json(attention / "RESULT.json")
    admission_gates = attention_admission_gates(attention_result, gate_policy)
    if not all(admission_gates.values()):
        raise RuntimeError(
            f"Attention did not open physical splice under {gate_policy}: "
            f"{admission_gates}"
        )
    design = _read_json(attention / "DESIGN.json")
    cases = {str(row["case_id"]): row for row in design["cases"]}
    available: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases.values():
        for candidate in case["candidates"]:
            if candidate.get("relation_control") is None:
                continue
            module_type = str(candidate["module_type"])
            available[module_type].append(
                {
                    "candidate_key": f"{case['case_id']}::{candidate['candidate_id']}",
                    "case_id": case["case_id"],
                    "candidate_id": candidate["candidate_id"],
                    "instance_id": case["instance_id"],
                }
            )
    selected: list[str] = []
    capacity: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    for module_type in PRIMARY_TYPES:
        typed = available[module_type]
        chosen = select_balanced(typed)
        tasks = {
            str(row["instance_id"])
            for row in typed
            if row["candidate_key"] in set(chosen)
        }
        capacity[module_type] = {
            "available_pairs": len(typed),
            "available_tasks": len({row["instance_id"] for row in typed}),
            "selected_pairs": len(chosen),
            "selected_tasks": len(tasks),
        }
        gates[f"{module_type}_32_pairs_8_tasks"] = (
            len(chosen) >= MIN_PAIRS and len(tasks) >= MIN_TASKS
        )
        selected.extend(chosen)
    output.mkdir(parents=True)
    value = {
        "status": "REGISTERED_BEFORE_PHYSICAL_OUTCOMES" if all(gates.values()) else "CAPACITY_SHORTFALL",
        "policy_status": (
            "REGISTERED_AFTER_ATTENTION_BEFORE_PHYSICAL_OUTCOMES"
            if all(gates.values())
            else "CAPACITY_SHORTFALL"
        ),
        "gate_policy": gate_policy,
        "gate_policy_reason": (
            "User accepts any task-reproducible advantage; effect-size gates are not required."
            if gate_policy == MINIMAL_RELIABLE_GATE_POLICY
            else "Original strong effect-size policy."
        ),
        "attention_admission_gates": admission_gates,
        "attention_selection_used_to_open_stage": True,
        "old_strong_gate_result_preserved": True,
        "attention_result": str(attention / "RESULT.json"),
        "attention_result_sha256": _sha256(attention / "RESULT.json"),
        "attention_observations_sha256": _sha256(attention / "ATTENTION.jsonl"),
        "design": str(attention / "DESIGN.json"),
        "design_sha256": _sha256(attention / "DESIGN.json"),
        "selected_candidate_keys": selected,
        "capacity": capacity,
        "capacity_gates": gates,
        "physical_outcome_used_for_selection": False,
        "conditions": [
            "whole_natural_module",
            "same_parent_same_length_boundary",
            "same_type_same_length_recency",
            "fixed_128_tail_diagnostic_when_available",
        ],
        "confirmatory_gates": (
            {
                "attention_density_natural_boundary_ratio": 1.25,
                "attention_paired_direction": 0.65,
                "attention_task_bootstrap_q025": 1.0,
                "local_output_natural_boundary_ratio": 0.90,
                "local_output_paired_win": 0.60,
                "minimum_pairs_per_type": MIN_PAIRS,
                "minimum_tasks_per_type": MIN_TASKS,
            }
            if gate_policy == STRONG_GATE_POLICY
            else {
                "attention_admission": "task-bootstrap lower bound above null",
                "local_output_natural_boundary_ratio": "below 1.0",
                "local_output_paired_win": "above 0.5",
                "local_output_task_bootstrap_q975": "below 1.0",
                "minimum_pairs_per_type": MIN_PAIRS,
                "minimum_tasks_per_type": MIN_TASKS,
            }
        ),
    }
    _write_json(destination, value)
    return value


def _local_output_perturbation(
    *,
    model: Any,
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    hidden: Mapping[int, torch.Tensor],
    source_start: int,
    target_start: int,
    length: int,
    query_start: int,
    query_end: int,
    theta: float,
) -> float:
    target_tokens = int(target_cache[0][0].shape[-2])
    key_positions = torch.arange(target_tokens, device="cuda")
    total = 0.0
    points = 0
    for layer in PROBE_LAYERS:
        attention = model.model.layers[layer].self_attn
        num_heads = int(model.config.num_attention_heads)
        num_kv_heads = int(model.config.num_key_value_heads)
        groups = num_heads // num_kv_heads
        head_dim = int(getattr(model.config, "head_dim", 0)) or int(model.config.hidden_size) // num_heads
        dense_key = target_cache[layer][0].to("cuda")
        dense_value = target_cache[layer][1].to("cuda")
        stale_key = dense_key.clone()
        stale_value = dense_value.clone()
        stale_key[:, target_start : target_start + length] = _rope_shift(
            source_cache[layer][0][:, source_start : source_start + length].to("cuda"),
            target_start - source_start,
            theta,
        )
        stale_value[:, target_start : target_start + length] = source_cache[layer][1][
            :, source_start : source_start + length
        ].to("cuda")
        dense_key_h = dense_key.repeat_interleave(groups, dim=0)
        stale_key_h = stale_key.repeat_interleave(groups, dim=0)
        dense_value_h = dense_value.repeat_interleave(groups, dim=0)
        stale_value_h = stale_value.repeat_interleave(groups, dim=0)
        for left in range(query_start, query_end, QUERY_CHUNK):
            right = min(query_end, left + QUERY_CHUNK)
            query, positions = old._query_tensor(
                model=model,
                layer_index=layer,
                hidden=hidden[layer],
                query_left=left,
                query_right=right,
            )
            dense_scores = torch.matmul(
                query[0].float(), dense_key_h.float().transpose(-1, -2)
            ) / math.sqrt(head_dim)
            stale_scores = torch.matmul(
                query[0].float(), stale_key_h.float().transpose(-1, -2)
            ) / math.sqrt(head_dim)
            causal = key_positions.view(1, 1, -1) > positions.view(1, -1, 1)
            metrics = _head_query_bound_metrics(
                dense_scores=dense_scores.masked_fill(causal, -torch.inf),
                stale_scores=stale_scores.masked_fill(causal, -torch.inf),
                dense_values=dense_value_h,
                stale_values=stale_value_h,
                island_start=target_start,
                island_end=target_start + length,
            )
            total += float(metrics["actual_kv_output_relative"].sum())
            points += int(metrics["actual_kv_output_relative"].numel())
            del query, positions, dense_scores, stale_scores, causal, metrics
        del dense_key, dense_value, stale_key, stale_value
        del dense_key_h, stale_key_h, dense_value_h, stale_value_h
        torch.cuda.empty_cache()
    return total / points


def _condition(
    *,
    model: Any,
    case: Mapping[str, Any],
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    hidden: Mapping[int, torch.Tensor],
    dense_logits: torch.Tensor,
    theta: float,
    source_start: int,
    target_start: int,
    length: int,
    query_start: int,
    query_end: int,
) -> dict[str, Any]:
    physical_case = {
        "source_input_ids": case["source_input_ids"],
        "target_input_ids": case["target_input_ids"],
        "source_start": source_start,
        "target_start": target_start,
        "length": length,
    }
    logits = _physical_splice_logits(
        model=model,
        case=physical_case,
        source_cache=source_cache,
        target_cache=target_cache,
        theta=theta,
        mode="kv",
    )
    local = _local_output_perturbation(
        model=model,
        source_cache=source_cache,
        target_cache=target_cache,
        hidden=hidden,
        source_start=source_start,
        target_start=target_start,
        length=length,
        query_start=query_start,
        query_end=query_end,
        theta=theta,
    )
    value = {
        "source_start": source_start,
        "target_start": target_start,
        "length": length,
        "conditional_local_output_relative": local,
        "final_logit_js": _js(dense_logits, logits),
        "top1_changed": int(dense_logits.argmax()) != int(logits.argmax()),
    }
    del logits
    return value


@torch.inference_mode()
def measure(
    attention: Path,
    output: Path,
    gate_policy: str = STRONG_GATE_POLICY,
) -> dict[str, Any]:
    registration = freeze(attention, output, gate_policy)
    if registration["status"] != "REGISTERED_BEFORE_PHYSICAL_OUTCOMES":
        raise RuntimeError("physical capacity gates did not pass")
    if registration["design_sha256"] != _sha256(attention / "DESIGN.json"):
        raise ValueError("physical design changed")
    selected = set(registration["selected_candidate_keys"])
    design = _read_json(attention / "DESIGN.json")
    cases = [
        case
        for case in design["cases"]
        if any(f"{case['case_id']}::{candidate['candidate_id']}" in selected for candidate in case["candidates"])
    ]
    observations = output / "OBSERVATIONS.jsonl"
    completed = set()
    if observations.exists():
        for line in observations.read_text().splitlines():
            if line.strip():
                completed.add(str(json.loads(line)["case_id"]))
    pending = [case for case in cases if str(case["case_id"]) not in completed]
    if not pending:
        return {"status": "COMPLETE", "cases": len(cases), "new_cases": 0}
    model = AutoModelForCausalLM.from_pretrained(
        old.MODEL,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda").eval()
    theta = _model_theta(model.config)
    written = 0
    errors = []
    for index, case in enumerate(pending, 1):
        try:
            source_cache, _, _ = old._dense_full(model, case["source_input_ids"], False)
            target_cache, dense_logits, hidden = old._dense_full(model, case["target_input_ids"], True)
            measured = []
            for candidate in case["candidates"]:
                key = f"{case['case_id']}::{candidate['candidate_id']}"
                if key not in selected:
                    continue
                relation = candidate["relation_control"]
                query_start = int(relation["query_start"])
                query_end = int(relation["query_end"])
                length = int(candidate["natural_length"])
                conditions = {
                    "natural": _condition(
                        model=model, case=case, source_cache=source_cache,
                        target_cache=target_cache, hidden=hidden, dense_logits=dense_logits,
                        theta=theta, source_start=int(candidate["source_start"]),
                        target_start=int(candidate["target_start"]), length=length,
                        query_start=query_start, query_end=query_end,
                    ),
                    "boundary": _condition(
                        model=model, case=case, source_cache=source_cache,
                        target_cache=target_cache, hidden=hidden, dense_logits=dense_logits,
                        theta=theta, source_start=int(candidate["source_boundary_start"]),
                        target_start=int(candidate["target_boundary_start"]), length=length,
                        query_start=query_start, query_end=query_end,
                    ),
                    "recency": _condition(
                        model=model, case=case, source_cache=source_cache,
                        target_cache=target_cache, hidden=hidden, dense_logits=dense_logits,
                        theta=theta, source_start=int(relation["source_control_start"]),
                        target_start=int(relation["control_start"]), length=length,
                        query_start=query_start, query_end=query_end,
                    ),
                }
                if length >= 128:
                    conditions["fixed_128_tail_diagnostic"] = _condition(
                        model=model, case=case, source_cache=source_cache,
                        target_cache=target_cache, hidden=hidden, dense_logits=dense_logits,
                        theta=theta, source_start=int(candidate["source_start"]) + length - 128,
                        target_start=int(candidate["target_start"]) + length - 128,
                        length=128, query_start=query_start, query_end=query_end,
                    )
                measured.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "module_type": candidate["module_type"],
                        "natural_length": length,
                        "conditions": conditions,
                    }
                )
            with observations.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"status": "ok", "case_id": case["case_id"], "instance_id": case["instance_id"], "candidates": measured}, sort_keys=True) + "\n")
            written += 1
            print(json.dumps({"case": index, "pending": len(pending), "case_id": case["case_id"]}), flush=True)
            del source_cache, target_cache, dense_logits, hidden, measured
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as error:
            errors.append({"case_id": case["case_id"], "error": f"{type(error).__name__}: {error}"})
            break
    del model
    gc.collect()
    torch.cuda.empty_cache()
    status = {
        "status": "COMPLETE" if not errors and written == len(pending) else "PARTIAL",
        "cases": len(cases),
        "previously_completed": len(completed),
        "new_cases": written,
        "errors": errors,
    }
    _write_json(output / "STATUS.json", status)
    return status


def _bootstrap(task_values: Mapping[str, Sequence[float]]) -> tuple[float, float, float]:
    tasks = sorted(task_values)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(tasks, len(tasks), replace=True)
        pooled = [value for task in sampled for value in task_values[str(task)]]
        values.append(statistics.median(pooled))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.5, 0.975]))


def summarize(attention: Path, output: Path) -> dict[str, Any]:
    if _read_json(output / "STATUS.json")["status"] != "COMPLETE":
        raise RuntimeError("physical measurements incomplete")
    registration = _read_json(output / "REGISTRATION.json")
    if registration["attention_result_sha256"] != _sha256(attention / "RESULT.json"):
        raise ValueError("Attention result changed after physical registration")
    gate_policy = str(registration.get("gate_policy", STRONG_GATE_POLICY))
    observations = [
        json.loads(line)
        for line in (output / "OBSERVATIONS.jsonl").read_text().splitlines()
        if line.strip()
    ]
    attention_rows = [
        point
        for line in (attention / "ATTENTION.jsonl").read_text().splitlines()
        if line.strip()
        for point in json.loads(line)["points"]
    ]
    attention_index: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in attention_rows:
        if row["kind"] in ("intra_natural", "intra_boundary"):
            attention_index[(str(row["case_id"]), str(row["candidate_id"]), str(row["kind"]))].append(float(row["attention_density"]))
    results: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    for module_type in PRIMARY_TYPES:
        attention_task: dict[str, list[float]] = defaultdict(list)
        local_task: dict[str, list[float]] = defaultdict(list)
        recency_task: dict[str, list[float]] = defaultdict(list)
        absolute_by_condition: dict[str, list[float]] = defaultdict(list)
        for row in observations:
            for candidate in row["candidates"]:
                if candidate["module_type"] != module_type:
                    continue
                key = (str(row["case_id"]), str(candidate["candidate_id"]))
                natural_attention = statistics.fmean(attention_index[(*key, "intra_natural")])
                boundary_attention = statistics.fmean(attention_index[(*key, "intra_boundary")])
                attention_task[str(row["instance_id"])].append(
                    natural_attention / max(boundary_attention, 1e-12)
                )
                natural_local = float(candidate["conditions"]["natural"]["conditional_local_output_relative"])
                boundary_local = float(candidate["conditions"]["boundary"]["conditional_local_output_relative"])
                local_task[str(row["instance_id"])].append(
                    natural_local / max(boundary_local, 1e-12)
                )
                recency_local = float(candidate["conditions"]["recency"]["conditional_local_output_relative"])
                recency_task[str(row["instance_id"])].append(
                    natural_local / max(recency_local, 1e-12)
                )
                for condition_name, condition in candidate["conditions"].items():
                    absolute_by_condition[str(condition_name)].append(
                        float(condition["conditional_local_output_relative"])
                    )
        attention_ratios = [value for values in attention_task.values() for value in values]
        local_ratios = [value for values in local_task.values() for value in values]
        attention_ci = _bootstrap(attention_task)
        local_ci = _bootstrap(local_task)
        recency_ratios = [value for values in recency_task.values() for value in values]
        recency_ci = _bootstrap(recency_task)
        result = {
            "pairs": len(local_ratios),
            "tasks": len(local_task),
            "attention_density_natural_boundary_median_ratio": statistics.median(attention_ratios),
            "attention_paired_direction": sum(value > 1 for value in attention_ratios) / len(attention_ratios),
            "attention_task_bootstrap_q025_q50_q975": attention_ci,
            "local_output_natural_boundary_median_ratio": statistics.median(local_ratios),
            "local_output_natural_wins": sum(value < 1 for value in local_ratios) / len(local_ratios),
            "local_output_natural_boundary_task_bootstrap_q025_q50_q975": local_ci,
            "local_output_natural_recency_median_ratio": statistics.median(recency_ratios),
            "local_output_natural_recency_wins": sum(value < 1 for value in recency_ratios) / len(recency_ratios),
            "local_output_natural_recency_task_bootstrap_q025_q50_q975": recency_ci,
            "conditional_local_output_relative_median_by_condition": {
                condition: statistics.median(values)
                for condition, values in sorted(absolute_by_condition.items())
            },
        }
        results[module_type] = result
        gates[f"{module_type}_32_pairs_8_tasks"] = result["pairs"] >= MIN_PAIRS and result["tasks"] >= MIN_TASKS
        if gate_policy == STRONG_GATE_POLICY:
            gates[f"{module_type}_attention_ratio_at_least_1_25"] = result["attention_density_natural_boundary_median_ratio"] >= 1.25
            gates[f"{module_type}_attention_direction_at_least_65pct"] = result["attention_paired_direction"] >= 0.65
            gates[f"{module_type}_attention_bootstrap_q025_above_1"] = attention_ci[0] > 1
            gates[f"{module_type}_local_ratio_at_most_0_90"] = result["local_output_natural_boundary_median_ratio"] <= 0.90
            gates[f"{module_type}_local_win_at_least_60pct"] = result["local_output_natural_wins"] >= 0.60
        elif gate_policy == MINIMAL_RELIABLE_GATE_POLICY:
            gates[f"{module_type}_attention_admission_passed"] = all(
                registration["attention_admission_gates"].values()
            )
            gates[f"{module_type}_local_ratio_below_1"] = result["local_output_natural_boundary_median_ratio"] < 1.0
            gates[f"{module_type}_local_win_above_chance"] = result["local_output_natural_wins"] > 0.5
            gates[f"{module_type}_local_bootstrap_q975_below_1"] = local_ci[2] < 1.0
        else:
            raise ValueError(f"unknown gate policy: {gate_policy}")
    value = {
        "status": "PASS" if all(gates.values()) else "STOP_BEFORE_STAGE_OVERHEAD",
        "gate_policy": gate_policy,
        "module_results": results,
        "gates": gates,
        "next_action": "measure_variable_length_stage_overhead" if all(gates.values()) else "stop_stage_overhead_and_runtime",
    }
    _write_json(output / "RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "measure", "summarize"))
    parser.add_argument("--attention", type=Path, default=DEFAULT_ATTENTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gate-policy", choices=GATE_POLICIES, default=STRONG_GATE_POLICY)
    args = parser.parse_args()
    if args.command == "freeze":
        value = freeze(args.attention, args.output, args.gate_policy)
    elif args.command == "measure":
        value = measure(args.attention, args.output, args.gate_policy)
    else:
        value = summarize(args.attention, args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
