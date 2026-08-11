#!/usr/bin/env python3
"""Measure 64-token action divergence for frozen equal-budget KV selectors.

The prior single-island study found only one immediate top-1 flip among 82
physical K+V splices.  That label is too sparse to tell whether a candidate
changes the agent's next action.  This follow-up freezes the 19 multi-candidate
cases and their already-registered 128-token arms, then generates a 64-token
greedy continuation from Dense and from each uniquely selected physical splice.

This is a behavioral-resolution experiment, not task accuracy and not TTFT.
It must not trigger a runtime implementation unless the action target both has
enough within-case variation and the frozen selectors win paired comparisons.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.multi_workflow.motivate_attention_kv_perturbation_bound import (
    FORWARD_CHUNK,
    _append_component_island,
    _cache_from_prefix,
    _model_theta,
)
from benchmark.multi_workflow.motivate_module_conditioned_attention_kv import (
    MODEL,
    _dense_full,
    _spearman,
)


ROOT = Path("/home/gfy/CodeMAS_Project")
SOURCE_STUDY = (
    ROOT
    / "kvflow-artifacts/impactkv_single_island_probe_transfer_20260807/"
    "unopened82"
)
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_single_island_action_divergence_20260807/"
    "frozen19"
)
CONTINUATION_TOKENS = 64
MIN_CASES = 16
MIN_TASKS = 8
MIN_UNIQUE_SPLICES = 30
MIN_VARIATION_CASES = 8
MIN_VARIATION_TASKS = 6


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o644)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prepare(source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    source_design_path = source / "DESIGN.json"
    arm_path = source / "ARM_REGISTRATION.json"
    result_path = source / "RESULT.json"
    source_design = json.loads(source_design_path.read_text(encoding="utf-8"))
    source_cases = {str(row["case_id"]): row for row in source_design["cases"]}
    arm_registration = json.loads(arm_path.read_text(encoding="utf-8"))
    cases = []
    for arm_case in arm_registration["cases"]:
        case_id = str(arm_case["case_id"])
        source_case = source_cases[case_id]
        selected_ids = sorted(set(str(value) for value in arm_case["arms"].values()))
        candidates = [
            copy.deepcopy(candidate)
            for candidate in source_case["candidates"]
            if str(candidate["candidate_id"]) in selected_ids
        ]
        if {str(row["candidate_id"]) for row in candidates} != set(selected_ids):
            raise ValueError(f"arm candidate missing from source design: {case_id}")
        cases.append(
            {
                **copy.deepcopy(source_case),
                "candidates": candidates,
                "arms": copy.deepcopy(arm_case["arms"]),
                "probe_score": copy.deepcopy(arm_case["probe_score"]),
                "module_oracle_risk": copy.deepcopy(arm_case["module_oracle_risk"]),
            }
        )
    tasks = len({str(row["instance_id"]) for row in cases})
    splice_count = sum(len(row["candidates"]) for row in cases)
    gates = {
        "cases": len(cases) >= MIN_CASES,
        "tasks": tasks >= MIN_TASKS,
        "unique_splices": splice_count >= MIN_UNIQUE_SPLICES,
        "all_equal_128_token_budget": all(
            int(candidate["length"]) == 128
            for case in cases
            for candidate in case["candidates"]
        ),
        "all_arms_frozen_before_prior_physical_outcomes": (
            arm_registration["status"] == "REGISTERED_BEFORE_NEW_PHYSICAL_OUTCOMES"
            and bool(arm_registration["new_physical_outcomes_unopened"])
        ),
    }
    output.mkdir(parents=True)
    design_path = output / "DESIGN.json"
    _write(
        design_path,
        {
            "model": str(MODEL),
            "continuation_tokens": CONTINUATION_TOKENS,
            "cases": cases,
        },
    )
    registration = {
        "status": "REGISTERED_BEFORE_ACTION_OUTCOMES"
        if all(gates.values())
        else "STOPPED_BEFORE_ACTION_OUTCOMES",
        "purpose": (
            "test whether 64-token next-action divergence provides a more "
            "resolved behavioral target than immediate top-1 or final-logit JS"
        ),
        "script_sha256": _sha(Path(__file__)),
        "design_sha256": _sha(design_path),
        "source_files": {
            str(path): _sha(path)
            for path in (source_design_path, arm_path, result_path)
        },
        "capacity": {
            "cases": len(cases),
            "tasks": tasks,
            "unique_selected_splices": splice_count,
        },
        "capacity_gates": gates,
        "frozen_decode": {
            "greedy": True,
            "maximum_new_tokens": CONTINUATION_TOKENS,
            "stop_at_eos": True,
            "same_model_and_prompt": True,
        },
        "frozen_outcome_gates": {
            "candidate_divergence_fraction_min": 0.20,
            "within_case_candidate_variation_cases_min": MIN_VARIATION_CASES,
            "within_case_candidate_variation_tasks_min": MIN_VARIATION_TASKS,
            "probe_vs_recency_paired_win_fraction_min": 0.60,
            "oracle_vs_recency_paired_win_fraction_min": 0.60,
        },
        "metric_scope": (
            "64-token next-action token divergence is a resolution diagnostic; "
            "it is not official task accuracy, semantic equivalence, or TTFT"
        ),
        "prior_final_js_used_to_tune_design": False,
        "new_action_outcomes_opened_by_prepare": False,
        "protected": {
            "paper_modified": False,
            "prefetch": False,
            "old_dirty_checkout_modified": False,
            "old_preregistration_thresholds_modified": False,
        },
    }
    _write(output / "REGISTRATION.json", registration)
    return registration


def common_prefix_tokens(left: Sequence[int], right: Sequence[int]) -> int:
    cursor = 0
    for left_token, right_token in zip(left, right):
        if int(left_token) != int(right_token):
            break
        cursor += 1
    return cursor


def levenshtein(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row_index, right_value in enumerate(right, 1):
        current = [row_index]
        for column_index, left_value in enumerate(left, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1]
                    + int(int(left_value) != int(right_value)),
                )
            )
        previous = current
    return previous[-1]


def sequence_metrics(dense: Sequence[int], splice: Sequence[int]) -> dict[str, Any]:
    distance = levenshtein(dense, splice)
    denominator = max(len(dense), len(splice), 1)
    prefix = common_prefix_tokens(dense, splice)
    return {
        "exact_match": list(dense) == list(splice),
        "common_prefix_tokens": prefix,
        "common_prefix_fraction": prefix / denominator,
        "token_edit_distance": distance,
        "normalized_token_edit_distance": distance / denominator,
    }


def _eos_ids(model: Any) -> set[int]:
    value = getattr(model.generation_config, "eos_token_id", None)
    if value is None:
        value = getattr(model.config, "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, int):
        return {int(value)}
    return {int(item) for item in value}


@torch.inference_mode()
def _greedy_continuation(
    *, model: Any, cache: Any, final_logits: torch.Tensor
) -> list[int]:
    eos = _eos_ids(model)
    logits = final_logits.to("cuda")
    tokens = []
    for index in range(CONTINUATION_TOKENS):
        token = int(logits.argmax().item())
        tokens.append(token)
        if token in eos or index + 1 == CONTINUATION_TOKENS:
            break
        output = model(
            input_ids=torch.tensor([[token]], device="cuda", dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        logits = output.logits[0, -1]
        del output
    del cache, logits
    gc.collect()
    torch.cuda.empty_cache()
    return tokens


@torch.inference_mode()
def _spliced_cache_and_logits(
    *,
    model: Any,
    case: Mapping[str, Any],
    candidate: Mapping[str, Any],
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    theta: float,
) -> tuple[Any, torch.Tensor]:
    start = int(candidate["target_start"])
    end = start + int(candidate["length"])
    cache = _cache_from_prefix(model, target_cache, start)
    cache = _append_component_island(
        model=model,
        cache=cache,
        source_cache=source_cache,
        target_cache=target_cache,
        source_start=int(candidate["source_start"]),
        target_start=start,
        length=int(candidate["length"]),
        theta=theta,
        mode="kv",
    )
    logits = None
    for offset in range(end, len(case["target_input_ids"]), FORWARD_CHUNK):
        output = model(
            input_ids=torch.tensor(
                [case["target_input_ids"][offset : offset + FORWARD_CHUNK]],
                device="cuda",
                dtype=torch.long,
            ),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        logits = output.logits[0, -1].detach()
        del output
    if logits is None:
        raise RuntimeError("physical splice suffix produced no logits")
    return cache, logits


def _complete(row: Mapping[str, Any], expected: int) -> bool:
    return (
        row.get("status") == "ok"
        and bool(row.get("dense_continuation_tokens"))
        and len(row.get("candidates", [])) == expected
        and all(bool(candidate.get("continuation_tokens")) for candidate in row["candidates"])
    )


@torch.inference_mode()
def measure(output: Path, max_cases: int) -> dict[str, Any]:
    registration = json.loads((output / "REGISTRATION.json").read_text())
    design_path = output / "DESIGN.json"
    if registration["status"] != "REGISTERED_BEFORE_ACTION_OUTCOMES":
        raise RuntimeError("capacity registration did not pass")
    if registration["design_sha256"] != _sha(design_path):
        raise ValueError("design changed after registration")
    cases = json.loads(design_path.read_text())["cases"]
    if max_cases > 0:
        cases = cases[:max_cases]
    destination = output / "ACTION_OUTCOMES.jsonl"
    expected = {str(row["case_id"]): len(row["candidates"]) for row in cases}
    completed = set()
    if destination.exists():
        for row in _jsonl(destination):
            case_id = str(row["case_id"])
            if case_id in expected and _complete(row, expected[case_id]):
                completed.add(case_id)
    pending = [row for row in cases if str(row["case_id"]) not in completed]
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    theta = _model_theta(model.config)
    errors = []
    written = 0
    for index, case in enumerate(pending, 1):
        try:
            source_cache, _, _ = _dense_full(model, case["source_input_ids"], False)
            target_cache, dense_logits, _ = _dense_full(
                model, case["target_input_ids"], False
            )
            dense_cache = _cache_from_prefix(
                model, target_cache, len(case["target_input_ids"])
            )
            dense_tokens = _greedy_continuation(
                model=model, cache=dense_cache, final_logits=dense_logits
            )
            candidates = []
            for candidate in case["candidates"]:
                cache, logits = _spliced_cache_and_logits(
                    model=model,
                    case=case,
                    candidate=candidate,
                    source_cache=source_cache,
                    target_cache=target_cache,
                    theta=theta,
                )
                tokens = _greedy_continuation(
                    model=model, cache=cache, final_logits=logits
                )
                candidates.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "continuation_tokens": tokens,
                        "continuation_text": tokenizer.decode(
                            tokens, skip_special_tokens=False
                        ),
                        **sequence_metrics(dense_tokens, tokens),
                    }
                )
                del logits
            row = {
                "status": "ok",
                "case_id": case["case_id"],
                "instance_id": case["instance_id"],
                "dense_continuation_tokens": dense_tokens,
                "dense_continuation_text": tokenizer.decode(
                    dense_tokens, skip_special_tokens=False
                ),
                "candidates": candidates,
            }
            if not _complete(row, len(case["candidates"])):
                raise RuntimeError("incomplete action outcome")
            with destination.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
            print(
                json.dumps(
                    {
                        "case": index,
                        "pending": len(pending),
                        "case_id": case["case_id"],
                        "splices": len(candidates),
                        "diverged": sum(not row["exact_match"] for row in candidates),
                    }
                ),
                flush=True,
            )
            del source_cache, target_cache, dense_logits
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as error:
            errors.append(
                {
                    "case_id": case["case_id"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
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
    _write(output / "MEASUREMENT_STATUS.json", status)
    return status


def _paired(
    cases: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, Mapping[str, float]],
    treatment: str,
) -> dict[str, Any]:
    disagreements = [
        case
        for case in cases
        if case["arms"][treatment] != case["arms"]["current_recency"]
    ]
    left = [
        float(outcomes[str(case["case_id"])][str(case["arms"][treatment])])
        for case in disagreements
    ]
    right = [
        float(outcomes[str(case["case_id"])][str(case["arms"]["current_recency"])])
        for case in disagreements
    ]
    return {
        "disagreement_cases": len(disagreements),
        "disagreement_tasks": len(
            {str(case["instance_id"]) for case in disagreements}
        ),
        "win_fraction": statistics.fmean(a < b for a, b in zip(left, right, strict=True)),
        "tie_fraction": statistics.fmean(a == b for a, b in zip(left, right, strict=True)),
        "mean_paired_edit_delta": statistics.fmean(
            a - b for a, b in zip(left, right, strict=True)
        ),
    }


def summarize(output: Path) -> dict[str, Any]:
    registration = json.loads((output / "REGISTRATION.json").read_text())
    design = json.loads((output / "DESIGN.json").read_text())
    rows = _jsonl(output / "ACTION_OUTCOMES.jsonl")
    if len(rows) != len(design["cases"]):
        raise ValueError("all action outcomes must complete")
    outcome_rows = {str(row["case_id"]): row for row in rows}
    distance = {
        case_id: {
            str(candidate["candidate_id"]): float(
                candidate["normalized_token_edit_distance"]
            )
            for candidate in row["candidates"]
        }
        for case_id, row in outcome_rows.items()
    }
    all_candidates = [candidate for row in rows for candidate in row["candidates"]]
    variation_cases = []
    for case in design["cases"]:
        case_id = str(case["case_id"])
        sequences = {
            tuple(candidate["continuation_tokens"])
            for candidate in outcome_rows[case_id]["candidates"]
        }
        if len(sequences) >= 2:
            variation_cases.append(case)

    arm_names = (
        "current_recency",
        "fixed_probe_min",
        "module_attention_oracle",
        "seeded_random",
    )
    arms = {}
    for name in arm_names:
        values = [
            distance[str(case["case_id"])][str(case["arms"][name])]
            for case in design["cases"]
        ]
        arms[name] = {
            "mean_normalized_token_edit_distance": statistics.fmean(values),
            "median_normalized_token_edit_distance": statistics.median(values),
            "exact_dense_match_fraction": statistics.fmean(value == 0 for value in values),
        }
    for name in ("fixed_probe_min", "module_attention_oracle", "seeded_random"):
        arms[name]["vs_recency"] = _paired(design["cases"], distance, name)

    probes = []
    oracles = []
    distances = []
    for case in design["cases"]:
        case_id = str(case["case_id"])
        for candidate in case["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            probes.append(float(case["probe_score"][candidate_id]))
            oracles.append(float(case["module_oracle_risk"][candidate_id]))
            distances.append(distance[case_id][candidate_id])
    resolution_gates = {
        "candidate_divergence_fraction_min_0_20": statistics.fmean(
            not bool(row["exact_match"]) for row in all_candidates
        )
        >= 0.20,
        "within_case_variation_cases_min_8": len(variation_cases)
        >= MIN_VARIATION_CASES,
        "within_case_variation_tasks_min_6": len(
            {str(case["instance_id"]) for case in variation_cases}
        )
        >= MIN_VARIATION_TASKS,
    }
    selector_gates = {
        "probe_paired_win_min_0_60": arms["fixed_probe_min"]["vs_recency"][
            "win_fraction"
        ]
        >= 0.60,
        "oracle_paired_win_min_0_60": arms["module_attention_oracle"][
            "vs_recency"
        ]["win_fraction"]
        >= 0.60,
    }
    resolution_passed = all(resolution_gates.values())
    selector_passed = resolution_passed and all(selector_gates.values())
    decision = (
        "SUPPORTED_ACTION_TARGET_AND_FROZEN_SELECTORS"
        if selector_passed
        else "SUPPORTED_ACTION_TARGET_SELECTORS_FAILED"
        if resolution_passed
        else "ACTION_TARGET_TOO_SPARSE"
    )
    result = {
        "status": "COMPLETE",
        "decision": decision,
        "cases": len(rows),
        "tasks": len({str(row["instance_id"]) for row in rows}),
        "unique_selected_splices": len(all_candidates),
        "candidate_divergence_fraction": statistics.fmean(
            not bool(row["exact_match"]) for row in all_candidates
        ),
        "within_case_candidate_variation": {
            "cases": len(variation_cases),
            "tasks": len({str(case["instance_id"]) for case in variation_cases}),
        },
        "signal_to_action_distance_spearman": {
            "fixed_probe": _spearman(probes, distances),
            "module_attention_oracle": _spearman(oracles, distances),
        },
        "arms": arms,
        "resolution_gates": resolution_gates,
        "selector_gates": selector_gates,
        "scope": registration["metric_scope"],
        "next_step": (
            "run an execution-level canary only if action target and selectors pass"
            if selector_passed
            else "do not implement the frozen KV-distance selector"
        ),
    }
    _write(output / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--source", type=Path, default=SOURCE_STUDY)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    measure_parser = sub.add_parser("measure")
    measure_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    measure_parser.add_argument("--max-cases", type=int, default=0)
    summarize_parser = sub.add_parser("summarize")
    summarize_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.source, args.output)
    elif args.command == "measure":
        value = measure(args.output, args.max_cases)
    else:
        value = summarize(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
