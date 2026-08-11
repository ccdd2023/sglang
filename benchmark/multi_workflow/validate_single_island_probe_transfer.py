#!/usr/bin/env python3
"""Transfer the frozen M49 single-island probe to unopened agent candidates.

The experiment deliberately excludes every candidate whose physical splice
outcome was opened by the module-conditioned Attention/KV study.  It then:

1. freezes the remaining capacity without reading physical outcomes;
2. measures the already-frozen M49 layer-17 / 16-token probe;
3. freezes equal-128-token recency, probe, module-oracle and random arms;
4. opens physical K+V splice outcomes once; and
5. stops before runtime work unless both the oracle opportunity and cheap
   probe selection gates pass.

The module oracle uses Dense target internals and is not deployable.  It is an
upper bound that tells us whether the newly established local-risk mechanism
can improve single-island final-logit fidelity at all.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from benchmark.multi_workflow.motivate_attention_kv_perturbation_bound import (
    _js,
    _model_theta,
    _physical_splice_logits,
    _rope_shift,
)
from benchmark.multi_workflow.motivate_module_conditioned_attention_kv import (
    CANDIDATE_TOKENS,
    MODEL,
    _cosine_drift,
    _crossfit_candidate_risk,
    _dense_full,
    _physical_training_points,
    _spearman,
)


ROOT = Path("/home/gfy/CodeMAS_Project")
BASE = (
    ROOT
    / "kvflow-artifacts/impactkv_module_conditioned_attention_kv_20260807/"
    "task_disjoint20"
)
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_single_island_probe_transfer_20260807/"
    "unopened82"
)
PROBE_LAYER = 17
PROBE_TOKENS = 16
M49_THRESHOLD = 0.011477339267730712
SEED = 2026080708
MIN_CANDIDATES = 80
MIN_CASES = 48
MIN_TASKS = 12
MIN_SELECTION_CASES = 16
MIN_SELECTION_TASKS = 8
MIN_DISAGREEMENT_CASES = 8
MIN_DISAGREEMENT_TASKS = 6


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o644)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _candidate_key(case_id: str, candidate_id: str) -> str:
    return f"{case_id}::{candidate_id}"


def prepare(output: Path, base: Path) -> dict[str, Any]:
    """Freeze only candidates whose physical outcomes have never been opened."""

    if output.exists():
        raise FileExistsError(output)
    required = (
        base / "DESIGN.json",
        base / "REGISTRATION.json",
        base / "INTERNALS.jsonl",
        base / "CELL_REGISTRATION.json",
        base / "RESULT.json",
    )
    if any(not path.exists() for path in required):
        raise FileNotFoundError("completed module-conditioned study is required")
    design = json.loads((base / "DESIGN.json").read_text())
    cells = json.loads((base / "CELL_REGISTRATION.json").read_text())
    opened = set(str(value) for value in cells["selected_candidate_keys"])
    cases = []
    for case in design["cases"]:
        candidates = [
            copy.deepcopy(candidate)
            for candidate in case["candidates"]
            if _candidate_key(str(case["case_id"]), str(candidate["candidate_id"]))
            not in opened
        ]
        if candidates:
            cases.append({**copy.deepcopy(case), "candidates": candidates})
    candidate_count = sum(len(case["candidates"]) for case in cases)
    tasks = len({str(case["instance_id"]) for case in cases})
    selection_cases = [case for case in cases if len(case["candidates"]) >= 2]
    selection_tasks = len({str(case["instance_id"]) for case in selection_cases})
    capacity = {
        "unopened_candidates": candidate_count,
        "cases": len(cases),
        "tasks": tasks,
        "selection_cases_with_at_least_two_candidates": len(selection_cases),
        "selection_tasks": selection_tasks,
        "minimum_candidates": MIN_CANDIDATES,
        "minimum_cases": MIN_CASES,
        "minimum_tasks": MIN_TASKS,
        "minimum_selection_cases": MIN_SELECTION_CASES,
        "minimum_selection_tasks": MIN_SELECTION_TASKS,
    }
    gates = {
        "candidates": candidate_count >= MIN_CANDIDATES,
        "cases": len(cases) >= MIN_CASES,
        "tasks": tasks >= MIN_TASKS,
        "selection_cases": len(selection_cases) >= MIN_SELECTION_CASES,
        "selection_tasks": selection_tasks >= MIN_SELECTION_TASKS,
        "all_exact_128_tokens": all(
            int(candidate["length"]) == CANDIDATE_TOKENS
            and case["source_input_ids"][
                int(candidate["source_start"]): int(candidate["source_start"])
                + CANDIDATE_TOKENS
            ]
            == case["target_input_ids"][
                int(candidate["target_start"]): int(candidate["target_start"])
                + CANDIDATE_TOKENS
            ]
            for case in cases
            for candidate in case["candidates"]
        ),
        "all_version_valid": all(
            bool(candidate["version_valid_at_target"])
            for case in cases
            for candidate in case["candidates"]
        ),
    }
    output.mkdir(parents=True)
    design_path = output / "DESIGN.json"
    _write_json(
        design_path,
        {
            "cases": cases,
            "model": str(MODEL),
            "candidate_tokens": CANDIDATE_TOKENS,
            "excluded_opened_candidate_keys": sorted(opened),
        },
    )
    registration = {
        "status": "REGISTERED_BEFORE_PROBE_AND_NEW_OUTCOMES"
        if all(gates.values())
        else "STOPPED_BEFORE_PROBE_AND_NEW_OUTCOMES",
        "purpose": "fixed M49 single-island probe transfer on unopened coding-agent candidates",
        "design_sha256": _sha(design_path),
        "script_sha256": _sha(Path(__file__)),
        "capacity": capacity,
        "capacity_gates": gates,
        "frozen_probe": {
            "source": "M49 independent single-island validation",
            "layer_zero_based": PROBE_LAYER,
            "head_tokens": PROBE_TOKENS,
            "score": "max(RoPE-corrected K cosine drift, V cosine drift)",
            "old_m49_threshold_descriptive_only": M49_THRESHOLD,
            "configuration_tuned_on_current_outcomes": False,
        },
        "frozen_arms": {
            "current_recency": "first candidate in unchanged recency order",
            "fixed_probe_min": "minimum frozen M49 probe score",
            "module_attention_oracle": (
                "minimum leave-one-task-out module-conditioned risk trained only "
                "on the prior 55 opened candidates"
            ),
            "seeded_random": f"sha256 order with seed {SEED}",
            "tokens_per_arm": CANDIDATE_TOKENS,
        },
        "outcome_gates": {
            "probe_global_js_spearman_min": 0.30,
            "probe_mean_within_case_js_spearman_min": 0.30,
            "module_oracle_vs_recency_median_js_ratio_max": 0.90,
            "module_oracle_vs_recency_disagreement_win_fraction_min": 0.60,
            "probe_vs_recency_median_js_ratio_max": 0.90,
            "probe_vs_recency_disagreement_win_fraction_min": 0.60,
        },
        "inputs": {
            path.name: {"path": str(path), "sha256": _sha(path)}
            for path in required
        },
        "physical_outcomes_read_by_prepare": False,
        "protected": {
            "paper_modified": False,
            "prefetch": False,
            "old_dirty_checkout_modified": False,
            "old_preregistration_thresholds_modified": False,
        },
    }
    _write_json(output / "CAPACITY.json", capacity)
    _write_json(output / "REGISTRATION.json", registration)
    return registration


def _signals_complete(row: Mapping[str, Any], expected: int) -> bool:
    return (
        row.get("status") == "ok"
        and len(row.get("candidates", [])) == expected
        and all(math.isfinite(float(value["probe_score"])) for value in row["candidates"])
    )


@torch.inference_mode()
def measure_signals(output: Path, max_cases: int) -> dict[str, Any]:
    registration = json.loads((output / "REGISTRATION.json").read_text())
    design_path = output / "DESIGN.json"
    if registration["status"] != "REGISTERED_BEFORE_PROBE_AND_NEW_OUTCOMES":
        raise RuntimeError("capacity gate did not pass")
    if _sha(design_path) != registration["design_sha256"]:
        raise ValueError("design changed after registration")
    cases = json.loads(design_path.read_text())["cases"]
    if max_cases > 0:
        cases = cases[:max_cases]
    destination = output / "SIGNALS.jsonl"
    completed = set()
    if destination.exists():
        expected = {str(case["case_id"]): len(case["candidates"]) for case in cases}
        for row in _jsonl(destination):
            case_id = str(row["case_id"])
            if case_id in expected and _signals_complete(row, expected[case_id]):
                completed.add(case_id)
    pending = [case for case in cases if str(case["case_id"]) not in completed]
    if not pending:
        return {"status": "COMPLETE", "selected_cases": len(cases), "new_cases": 0}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    theta = _model_theta(model.config)
    errors = []
    written = 0
    for index, case in enumerate(pending, 1):
        try:
            source_cache, _, _ = _dense_full(model, case["source_input_ids"], False)
            target_cache, _, _ = _dense_full(model, case["target_input_ids"], False)
            measured = []
            source_key, source_value = source_cache[PROBE_LAYER]
            target_key, target_value = target_cache[PROBE_LAYER]
            for candidate in case["candidates"]:
                source_start = int(candidate["source_start"])
                target_start = int(candidate["target_start"])
                stale_key = _rope_shift(
                    source_key[:, source_start: source_start + PROBE_TOKENS].to("cuda"),
                    target_start - source_start,
                    theta,
                ).cpu()
                dense_key = target_key[:, target_start: target_start + PROBE_TOKENS]
                stale_value = source_value[:, source_start: source_start + PROBE_TOKENS]
                dense_value = target_value[:, target_start: target_start + PROBE_TOKENS]
                key_drift = _cosine_drift(stale_key, dense_key)
                value_drift = _cosine_drift(stale_value, dense_value)
                measured.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "key_drift": key_drift,
                        "value_drift": value_drift,
                        "probe_score": max(key_drift, value_drift),
                    }
                )
            row = {
                "status": "ok",
                "case_id": case["case_id"],
                "instance_id": case["instance_id"],
                "candidates": measured,
            }
            if not _signals_complete(row, len(case["candidates"])):
                raise RuntimeError("incomplete probe signals")
            with destination.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
            print(json.dumps({"case": index, "pending": len(pending), "case_id": case["case_id"]}), flush=True)
            del source_cache, target_cache
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
        "selected_cases": len(cases),
        "previously_completed": len(completed),
        "new_cases": written,
        "errors": errors,
        "probe_layer_zero_based": PROBE_LAYER,
        "probe_tokens": PROBE_TOKENS,
    }
    _write_json(output / "SIGNAL_MEASUREMENT_STATUS.json", status)
    return status


def select_arms(
    case: Mapping[str, Any],
    probe_by_id: Mapping[str, float],
    oracle_by_id: Mapping[str, float],
) -> dict[str, str]:
    candidates = list(case["candidates"])
    if len(candidates) < 2:
        raise ValueError("arm selection requires at least two candidates")
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    if any(candidate_id not in probe_by_id or candidate_id not in oracle_by_id for candidate_id in candidate_ids):
        raise ValueError("candidate signal is missing")
    return {
        "current_recency": candidate_ids[0],
        "fixed_probe_min": min(candidate_ids, key=lambda value: (probe_by_id[value], value)),
        "module_attention_oracle": min(candidate_ids, key=lambda value: (oracle_by_id[value], value)),
        "seeded_random": min(candidate_ids, key=lambda value: _stable(f"{case['case_id']}::{value}")),
    }


def freeze_arms(output: Path, base: Path) -> dict[str, Any]:
    destination = output / "ARM_REGISTRATION.json"
    if destination.exists():
        return json.loads(destination.read_text())
    design = json.loads((output / "DESIGN.json").read_text())
    signals = _jsonl(output / "SIGNALS.jsonl")
    if len(signals) != len(design["cases"]):
        raise ValueError("all probe signals must finish before arm freeze")
    probe = {
        _candidate_key(str(row["case_id"]), str(candidate["candidate_id"])): float(candidate["probe_score"])
        for row in signals
        for candidate in row["candidates"]
    }
    cells = json.loads((base / "CELL_REGISTRATION.json").read_text())
    modules = list(cells["qualifying_modules"])
    training = _physical_training_points(base, cells)
    all_points = [point for point in cells["cell_points"] if point["module"] in modules]
    oracle, _ = _crossfit_candidate_risk(
        training_points=training,
        all_points=all_points,
        modules=modules,
    )
    cases = []
    for case in design["cases"]:
        if len(case["candidates"]) < 2:
            continue
        probe_by_id = {
            str(candidate["candidate_id"]): probe[
                _candidate_key(str(case["case_id"]), str(candidate["candidate_id"]))
            ]
            for candidate in case["candidates"]
        }
        oracle_by_id = {
            str(candidate["candidate_id"]): oracle[
                _candidate_key(str(case["case_id"]), str(candidate["candidate_id"]))
            ]
            for candidate in case["candidates"]
            if _candidate_key(str(case["case_id"]), str(candidate["candidate_id"])) in oracle
        }
        if len(oracle_by_id) != len(case["candidates"]):
            continue
        arms = select_arms(case, probe_by_id, oracle_by_id)
        cases.append(
            {
                "case_id": case["case_id"],
                "instance_id": case["instance_id"],
                "request_index": case["request_index"],
                "arms": arms,
                "probe_score": probe_by_id,
                "module_oracle_risk": oracle_by_id,
            }
        )
    probe_disagreements = [row for row in cases if row["arms"]["fixed_probe_min"] != row["arms"]["current_recency"]]
    oracle_disagreements = [row for row in cases if row["arms"]["module_attention_oracle"] != row["arms"]["current_recency"]]
    tasks = len({str(row["instance_id"]) for row in cases})
    probe_tasks = len({str(row["instance_id"]) for row in probe_disagreements})
    oracle_tasks = len({str(row["instance_id"]) for row in oracle_disagreements})
    gates = {
        "selection_cases": len(cases) >= MIN_SELECTION_CASES,
        "selection_tasks": tasks >= MIN_SELECTION_TASKS,
        "probe_disagreement_cases": len(probe_disagreements) >= MIN_DISAGREEMENT_CASES,
        "probe_disagreement_tasks": probe_tasks >= MIN_DISAGREEMENT_TASKS,
        "oracle_disagreement_cases": len(oracle_disagreements) >= MIN_DISAGREEMENT_CASES,
        "oracle_disagreement_tasks": oracle_tasks >= MIN_DISAGREEMENT_TASKS,
    }
    value = {
        "status": "REGISTERED_BEFORE_NEW_PHYSICAL_OUTCOMES"
        if all(gates.values())
        else "STOPPED_BEFORE_NEW_PHYSICAL_OUTCOMES",
        "cases": cases,
        "selected_cases": len(cases),
        "selected_tasks": tasks,
        "probe_disagreement_cases": len(probe_disagreements),
        "probe_disagreement_tasks": probe_tasks,
        "oracle_disagreement_cases": len(oracle_disagreements),
        "oracle_disagreement_tasks": oracle_tasks,
        "gates": gates,
        "signals_sha256": _sha(output / "SIGNALS.jsonl"),
        "prior_training_splices_sha256": _sha(base / "SPLICE_OBSERVATIONS.jsonl"),
        "new_physical_outcomes_unopened": True,
    }
    _write_json(destination, value)
    return value


def _outcome_complete(row: Mapping[str, Any], expected: int) -> bool:
    return (
        row.get("status") == "ok"
        and len(row.get("candidates", [])) == expected
        and all(math.isfinite(float(value["final_logit_js"])) for value in row["candidates"])
    )


@torch.inference_mode()
def measure_outcomes(output: Path, max_cases: int) -> dict[str, Any]:
    arms = json.loads((output / "ARM_REGISTRATION.json").read_text())
    if arms["status"] != "REGISTERED_BEFORE_NEW_PHYSICAL_OUTCOMES":
        raise RuntimeError("arm capacity gate did not pass")
    design = json.loads((output / "DESIGN.json").read_text())
    cases = design["cases"]
    if max_cases > 0:
        cases = cases[:max_cases]
    destination = output / "OUTCOMES.jsonl"
    completed = set()
    expected = {str(case["case_id"]): len(case["candidates"]) for case in cases}
    if destination.exists():
        for row in _jsonl(destination):
            case_id = str(row["case_id"])
            if case_id in expected and _outcome_complete(row, expected[case_id]):
                completed.add(case_id)
    pending = [case for case in cases if str(case["case_id"]) not in completed]
    if not pending:
        return {"status": "COMPLETE", "selected_cases": len(cases), "new_cases": 0}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    theta = _model_theta(model.config)
    errors = []
    written = 0
    for index, case in enumerate(pending, 1):
        try:
            source_cache, _, _ = _dense_full(model, case["source_input_ids"], False)
            target_cache, dense_logits, _ = _dense_full(model, case["target_input_ids"], False)
            measured = []
            for candidate in case["candidates"]:
                physical_case = {
                    **case,
                    "source_start": int(candidate["source_start"]),
                    "target_start": int(candidate["target_start"]),
                    "length": CANDIDATE_TOKENS,
                }
                logits = _physical_splice_logits(
                    model=model,
                    case=physical_case,
                    source_cache=source_cache,
                    target_cache=target_cache,
                    theta=theta,
                    mode="kv",
                )
                measured.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "final_logit_js": _js(dense_logits, logits),
                        "top1_changed": int(dense_logits.argmax()) != int(logits.argmax()),
                    }
                )
                del logits
            row = {
                "status": "ok",
                "case_id": case["case_id"],
                "instance_id": case["instance_id"],
                "candidates": measured,
            }
            if not _outcome_complete(row, len(case["candidates"])):
                raise RuntimeError("incomplete physical outcomes")
            with destination.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
            print(json.dumps({"case": index, "pending": len(pending), "case_id": case["case_id"], "candidates": len(measured)}), flush=True)
            del source_cache, target_cache, dense_logits
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
        "selected_cases": len(cases),
        "previously_completed": len(completed),
        "new_cases": written,
        "errors": errors,
    }
    _write_json(output / "OUTCOME_MEASUREMENT_STATUS.json", status)
    return status


def mean_within_case_spearman(
    scores: Mapping[str, Mapping[str, float]],
    outcomes: Mapping[str, Mapping[str, float]],
) -> float:
    values = []
    for case_id in sorted(scores):
        common = sorted(set(scores[case_id]) & set(outcomes.get(case_id, {})))
        if len(common) < 2:
            continue
        value = _spearman(
            [float(scores[case_id][candidate]) for candidate in common],
            [float(outcomes[case_id][candidate]) for candidate in common],
        )
        if math.isfinite(value):
            values.append(value)
    return statistics.fmean(values) if values else math.nan


def _arm_summary(
    registration: Mapping[str, Any],
    outcomes: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    names = ("current_recency", "fixed_probe_min", "module_attention_oracle", "seeded_random")
    result = {}
    for name in names:
        values = [
            float(outcomes[str(row["case_id"])][str(row["arms"][name])])
            for row in registration["cases"]
        ]
        result[name] = {"median_final_logit_js": statistics.median(values), "mean_final_logit_js": statistics.fmean(values)}
    recency = "current_recency"
    for name in ("fixed_probe_min", "module_attention_oracle", "seeded_random"):
        disagreements = [row for row in registration["cases"] if row["arms"][name] != row["arms"][recency]]
        treatment = [float(outcomes[str(row["case_id"])][str(row["arms"][name])]) for row in disagreements]
        control = [float(outcomes[str(row["case_id"])][str(row["arms"][recency])]) for row in disagreements]
        result[name]["vs_recency"] = {
            "disagreement_cases": len(disagreements),
            "disagreement_tasks": len({str(row["instance_id"]) for row in disagreements}),
            "win_fraction": statistics.fmean(float(left < right) for left, right in zip(treatment, control, strict=True)),
            "tie_fraction": statistics.fmean(float(left == right) for left, right in zip(treatment, control, strict=True)),
            "median_js_ratio_all_cases": result[name]["median_final_logit_js"]
            / max(result[recency]["median_final_logit_js"], 1e-20),
            "mean_paired_js_delta": statistics.fmean(left - right for left, right in zip(treatment, control, strict=True)),
        }
    return result


def summarize(output: Path) -> dict[str, Any]:
    signals_rows = _jsonl(output / "SIGNALS.jsonl")
    outcome_rows = _jsonl(output / "OUTCOMES.jsonl")
    arms = json.loads((output / "ARM_REGISTRATION.json").read_text())
    design = json.loads((output / "DESIGN.json").read_text())
    if len(outcome_rows) != len(design["cases"]):
        raise ValueError("all unopened outcomes must finish")
    signals = {
        str(row["case_id"]): {
            str(candidate["candidate_id"]): float(candidate["probe_score"])
            for candidate in row["candidates"]
        }
        for row in signals_rows
    }
    outcomes = {
        str(row["case_id"]): {
            str(candidate["candidate_id"]): float(candidate["final_logit_js"])
            for candidate in row["candidates"]
        }
        for row in outcome_rows
    }
    candidate_probe = []
    candidate_js = []
    for case_id in sorted(signals):
        for candidate_id in sorted(signals[case_id]):
            candidate_probe.append(signals[case_id][candidate_id])
            candidate_js.append(outcomes[case_id][candidate_id])
    global_spearman = _spearman(candidate_probe, candidate_js)
    within = mean_within_case_spearman(signals, outcomes)
    arm_summary = _arm_summary(arms, outcomes)
    probe_vs = arm_summary["fixed_probe_min"]["vs_recency"]
    oracle_vs = arm_summary["module_attention_oracle"]["vs_recency"]
    gates = {
        "probe_global_js_spearman_min_0_30": global_spearman >= 0.30,
        "probe_mean_within_case_js_spearman_min_0_30": within >= 0.30,
        "module_oracle_median_js_ratio_max_0_90": oracle_vs["median_js_ratio_all_cases"] <= 0.90,
        "module_oracle_win_fraction_min_0_60": oracle_vs["win_fraction"] >= 0.60,
        "probe_median_js_ratio_max_0_90": probe_vs["median_js_ratio_all_cases"] <= 0.90,
        "probe_win_fraction_min_0_60": probe_vs["win_fraction"] >= 0.60,
    }
    result = {
        "status": "COMPLETE",
        "decision": "SUPPORTED_FIXED_SINGLE_ISLAND_PROBE_CANARY"
        if all(gates.values())
        else "NOT_SUPPORTED_FOR_RUNTIME_CANARY",
        "unopened_candidates": len(candidate_js),
        "cases": len(outcome_rows),
        "tasks": len({str(row["instance_id"]) for row in outcome_rows}),
        "fixed_probe": {
            "layer_zero_based": PROBE_LAYER,
            "head_tokens": PROBE_TOKENS,
            "candidate_fraction": PROBE_TOKENS / CANDIDATE_TOKENS,
            "global_final_js_spearman": global_spearman,
            "mean_within_case_final_js_spearman": within,
        },
        "arms": arm_summary,
        "gate_results": gates,
        "scope": (
            "Previously unopened task-disjoint physical K+V splice outcomes. "
            "Final-logit fidelity only; not functional accuracy or SGLang TTFT."
        ),
    }
    _write_json(output / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--base", type=Path, default=BASE)
    signal_parser = sub.add_parser("measure-signals")
    signal_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    signal_parser.add_argument("--max-cases", type=int, default=0)
    freeze_parser = sub.add_parser("freeze-arms")
    freeze_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    freeze_parser.add_argument("--base", type=Path, default=BASE)
    outcome_parser = sub.add_parser("measure-outcomes")
    outcome_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    outcome_parser.add_argument("--max-cases", type=int, default=0)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.output, args.base)
    elif args.command == "measure-signals":
        value = measure_signals(args.output, args.max_cases)
    elif args.command == "freeze-arms":
        value = freeze_arms(args.output, args.base)
    elif args.command == "measure-outcomes":
        value = measure_outcomes(args.output, args.max_cases)
    else:
        value = summarize(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
