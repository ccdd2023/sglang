#!/usr/bin/env python3
"""Measure whether same-file mutation raises old-observation reuse risk.

Each M51 pair follows the *same exact grounded observation* at two real
rolling-history transitions from a frozen Dense coding-agent trajectory:

* treatment: the newly completed group mutates a path read by the observation;
* control: the newly completed group references the same path without any
  mutation, diff, executable failure, or other critical event.

The 128-token observation tail is physically spliced in both contexts.  This
tests V45/V46's file-version motivation, not a deployable policy or task-level
accuracy.
"""

from __future__ import annotations

import argparse
import gc
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

from benchmark.multi_workflow.coding_reuse_policy import (
    critical_coding_event_reasons,
    is_successful_readonly_evidence,
    repository_paths,
)
from benchmark.multi_workflow.measure_sessiongraph_atlas import _js
from benchmark.multi_workflow.motivate_v48_attention_kv_risk import (
    _cache_from_dense_prefix,
    _dense_source,
    _model_theta,
)
from benchmark.multi_workflow.motivate_v50_coding_provenance import (
    ANSWER_TOKENS,
    CANDIDATE_TOKENS,
    MODEL,
    _balanced_select,
    _continuation_nll,
    _kv_metrics,
    _message_candidate,
    _render_rolling,
    _sha256,
    _splice_with_cache,
    _token_ids_hash,
    _trajectory_paths,
    _turn_groups,
)
from benchmark.multi_workflow.run_bridge_reuse_pilot import write_json


ROOT = Path("/home/gfy/CodeMAS_Project")
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_m51_file_version_risk_20260805"
    / "matched18_v2"
)
RANDOM_SEED = 202608051


def _context_rows(tokenizer: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for instance_id, path in _trajectory_paths().items():
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        messages = trajectory["messages"]
        base = messages[:2]
        groups = _turn_groups(messages[2:])
        for target_completed in range(7, len(groups)):
            new_group = groups[target_completed - 1]
            reasons = critical_coding_event_reasons(new_group)
            if reasons == ["repository_mutation_command"]:
                category = "same_path_mutation"
            elif not reasons:
                category = "same_path_noncritical"
            else:
                continue
            new_paths = repository_paths(new_group)
            if not new_paths:
                continue
            source_ids, source_spans = _render_rolling(
                tokenizer, base, groups[: target_completed - 1]
            )
            target_ids, target_spans = _render_rolling(
                tokenizer, base, groups[:target_completed]
            )
            if len(source_ids) > 30_000 or len(target_ids) > 30_000:
                continue
            group_indices = sorted(
                {key[0] for key in source_spans}
                & {key[0] for key in target_spans}
            )
            for group_index in group_indices:
                group = groups[group_index]
                source_paths = repository_paths(group)
                if (
                    not source_paths
                    or source_paths.isdisjoint(new_paths)
                    or not is_successful_readonly_evidence(group)
                ):
                    continue
                for message_index, message in enumerate(group):
                    if message.get("role") != "tool":
                        continue
                    candidate = _message_candidate(
                        category="grounded_readonly_tool",
                        group_index=group_index,
                        message_index=message_index,
                        source_ids=source_ids,
                        source_spans=source_spans,
                        target_ids=target_ids,
                        target_spans=target_spans,
                    )
                    if candidate is None:
                        continue
                    answer = str(groups[target_completed][0].get("content") or "")
                    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                    if not answer_ids:
                        continue
                    rows.append(
                        {
                            "answer_ids": answer_ids[:ANSWER_TOKENS],
                            "candidate": candidate,
                            "category": category,
                            "instance_id": instance_id,
                            "new_paths": sorted(new_paths),
                            "position_fraction": candidate["target_start"]
                            / len(target_ids),
                            "prefix_shift_tokens": candidate["target_start"]
                            - candidate["source_start"],
                            "source_input_ids": source_ids,
                            "source_observation_key": (
                                f"{instance_id}:g{group_index}:m{message_index}"
                            ),
                            "source_paths": sorted(source_paths),
                            "source_prompt_hash": _token_ids_hash(source_ids),
                            "target_input_ids": target_ids,
                            "target_prompt_hash": _token_ids_hash(target_ids),
                            "target_request_index": target_completed + 1,
                            "trajectory_path": str(path),
                            "trajectory_sha256": _sha256(path),
                        }
                    )
    return rows


def _match_score(treatment: Mapping[str, Any], control: Mapping[str, Any]) -> float:
    return (
        abs(treatment["position_fraction"] - control["position_fraction"])
        + abs(treatment["prefix_shift_tokens"] - control["prefix_shift_tokens"])
        / 5000
        + abs(len(treatment["target_input_ids"]) - len(control["target_input_ids"]))
        / 20_000
    )


def _matched_pairs(tokenizer: Any) -> list[dict[str, Any]]:
    by_observation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _context_rows(tokenizer):
        by_observation[str(row["source_observation_key"])].append(row)
    pairs = []
    for observation_key, rows in by_observation.items():
        treatments = [row for row in rows if row["category"] == "same_path_mutation"]
        controls = [row for row in rows if row["category"] == "same_path_noncritical"]
        possible = [
            (_match_score(treatment, control), treatment, control)
            for treatment in treatments
            for control in controls
        ]
        if not possible:
            continue
        score, treatment, control = min(
            possible,
            key=lambda item: (
                item[0],
                item[1]["target_request_index"],
                item[2]["target_request_index"],
            ),
        )
        if treatment["candidate"]["segment_token_hash"] != control["candidate"][
            "segment_token_hash"
        ]:
            raise ValueError("matched contexts do not reuse the same observation")
        pair_id = hashlib.sha256(
            f"{RANDOM_SEED}:{observation_key}".encode()
        ).hexdigest()[:16]
        pairs.append(
            {
                "case_id": f"version-{pair_id}",
                "instance_id": treatment["instance_id"],
                "match_score": score,
                "source_observation_key": observation_key,
                "treatment": treatment,
                "control": control,
            }
        )
    return pairs


def prepare(output: Path, limit: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    eligible = _matched_pairs(tokenizer)
    selected = _balanced_select(eligible, limit)
    if len(selected) < limit:
        raise ValueError(f"only {len(selected)} eligible pairs for requested {limit}")
    output.mkdir(parents=True)
    design_path = output / "DESIGN.json"
    write_json(
        design_path,
        {
            "cases": selected,
            "eligible_pairs_before_sampling": len(eligible),
            "model": str(MODEL),
        },
    )
    registration = {
        "status": "REGISTERED_BEFORE_GPU",
        "purpose": (
            "test whether a same-file repository mutation increases causal "
            "lossy-reuse harm for an earlier grounded observation"
        ),
        "design_sha256": _sha256(design_path),
        "cases": len(selected),
        "tasks": len({row["instance_id"] for row in selected}),
        "eligible_pairs_before_sampling": len(eligible),
        "model": str(MODEL),
        "invalidated_predecessor": {
            "path": str(DEFAULT_OUTPUT.parent / "matched18"),
            "reason": "balanced sampler repeated case IDs after round one",
            "thresholds_changed": False,
        },
        "matching_contract": {
            "same_exact_observation_within_pair": True,
            "candidate_tokens": CANDIDATE_TOKENS,
            "treatment": "new pure mutation overlaps observation path",
            "control": "new noncritical group references the same path",
            "real_dense_agent_transitions": True,
            "one_pair_per_task_before_second_pair": True,
        },
        "frozen_support_rule": {
            "mutation_higher_JS_pair_fraction_min": 0.65,
            "covariate_adjusted_geometric_JS_ratio_min": 1.25,
            "minimum_complete_pairs": 16,
            "minimum_tasks": 8,
        },
        "interpretation_limits": [
            "matched transitions are different requests, not synthetic twins",
            "feature adjustment cannot remove all target-semantic differences",
            "offline diagnostic is not SWE-bench accuracy or TTFT",
            "support motivates a version guard but does not tune its threshold",
        ],
    }
    write_json(output / "REGISTRATION.json", registration)
    return registration


def _measure_context(
    *, model: Any, theta: float, context: Mapping[str, Any]
) -> dict[str, Any]:
    source_cache, _ = _dense_source(model, context["source_input_ids"])
    target_cache, dense_logits = _dense_source(model, context["target_input_ids"])
    dense_answer_cache = _cache_from_dense_prefix(
        model=model,
        target_cache=target_cache,
        prefix_tokens=len(context["target_input_ids"]),
    )
    dense_nll = _continuation_nll(
        model,
        dense_answer_cache,
        dense_logits,
        context["answer_ids"],
    )
    candidate = context["candidate"]
    kv = _kv_metrics(
        candidate=candidate,
        source_cache=source_cache,
        target_cache=target_cache,
        theta=theta,
    )
    splice_cache, splice_logits = _splice_with_cache(
        model=model,
        target_ids=context["target_input_ids"],
        target_cache=target_cache,
        source_cache=source_cache,
        candidate=candidate,
        theta=theta,
    )
    splice_nll = _continuation_nll(
        model, splice_cache, splice_logits, context["answer_ids"]
    )
    value = {
        "category": context["category"],
        "target_request_index": context["target_request_index"],
        "position_fraction": context["position_fraction"],
        "prefix_shift_tokens": context["prefix_shift_tokens"],
        "source_tokens": len(context["source_input_ids"]),
        "target_tokens": len(context["target_input_ids"]),
        **kv,
        "causal_splice_logit_js": _js(dense_logits, splice_logits),
        "causal_splice_top1_changed": int(dense_logits.argmax())
        != int(splice_logits.argmax()),
        "dense_next_action_nll": dense_nll,
        "next_action_nll_delta": splice_nll - dense_nll,
    }
    del (
        source_cache,
        target_cache,
        dense_logits,
        dense_answer_cache,
        splice_cache,
        splice_logits,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return value


def _complete(row: Mapping[str, Any]) -> bool:
    if row.get("status") != "ok":
        return False
    values = []
    for name in ("treatment", "control"):
        for metric in (
            "causal_splice_logit_js",
            "kv_cosine_drift_mean",
            "next_action_nll_delta",
        ):
            values.append(float(row[name][metric]))
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
            treatment = _measure_context(
                model=model, theta=theta, context=case["treatment"]
            )
            control = _measure_context(
                model=model, theta=theta, context=case["control"]
            )
            row = {
                "status": "ok",
                "case_id": case["case_id"],
                "instance_id": case["instance_id"],
                "source_observation_key": case["source_observation_key"],
                "match_score": case["match_score"],
                "treatment": treatment,
                "control": control,
            }
            if not _complete(row):
                raise RuntimeError("pair produced incomplete/non-finite metrics")
            with observations_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
            print(
                json.dumps(
                    {
                        "case": index,
                        "case_id": case["case_id"],
                        "pending": len(pending),
                        "mutation_js": treatment["causal_splice_logit_js"],
                        "control_js": control["causal_splice_logit_js"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
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
    y = np.asarray(
        [
            math.log(float(row["treatment"][metric]) + 1e-8)
            - math.log(float(row["control"][metric]) + 1e-8)
            for row in rows
        ],
        dtype=np.float64,
    )
    covariates = np.asarray(
        [
            [
                float(row["treatment"]["position_fraction"])
                - float(row["control"]["position_fraction"]),
                (
                    float(row["treatment"]["prefix_shift_tokens"])
                    - float(row["control"]["prefix_shift_tokens"])
                )
                / 1000,
                (
                    float(row["treatment"]["target_tokens"])
                    - float(row["control"]["target_tokens"])
                )
                / 1000,
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    design = np.column_stack((np.ones(len(rows)), covariates))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return {
        "covariate_adjusted_geometric_ratio": math.exp(float(coefficients[0])),
        "coefficients": {
            "position_fraction_difference": float(coefficients[1]),
            "prefix_shift_difference_per_1000": float(coefficients[2]),
            "target_length_difference_per_1000": float(coefficients[3]),
        },
    }


def _sign_probability(wins: int, trials: int) -> float:
    return sum(math.comb(trials, value) for value in range(wins, trials + 1)) / (
        2**trials
    )


def analyze(output: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (output / "OBSERVATIONS.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not rows or any(not _complete(row) for row in rows):
        raise ValueError("observations are missing or incomplete")
    metrics = {}
    for metric in ("causal_splice_logit_js", "kv_cosine_drift_mean"):
        treatment = [float(row["treatment"][metric]) for row in rows]
        control = [float(row["control"][metric]) for row in rows]
        wins = sum(a > b for a, b in zip(treatment, control, strict=True))
        metrics[metric] = {
            "mutation_mean": statistics.fmean(treatment),
            "mutation_median": statistics.median(treatment),
            "control_mean": statistics.fmean(control),
            "control_median": statistics.median(control),
            "mutation_higher_pair_fraction": wins / len(rows),
            "one_sided_sign_probability": _sign_probability(wins, len(rows)),
            **_adjusted_ratio(rows, metric),
        }
    mutation_nll = [float(row["treatment"]["next_action_nll_delta"]) for row in rows]
    control_nll = [float(row["control"]["next_action_nll_delta"]) for row in rows]
    registration = json.loads((output / "REGISTRATION.json").read_text())
    gates = registration["frozen_support_rule"]
    js = metrics["causal_splice_logit_js"]
    gate_results = {
        "minimum_complete_pairs": len(rows) >= gates["minimum_complete_pairs"],
        "minimum_tasks": len({row["instance_id"] for row in rows})
        >= gates["minimum_tasks"],
        "mutation_higher_JS_pair_fraction": js["mutation_higher_pair_fraction"]
        >= gates["mutation_higher_JS_pair_fraction_min"],
        "covariate_adjusted_geometric_JS_ratio": js[
            "covariate_adjusted_geometric_ratio"
        ]
        >= gates["covariate_adjusted_geometric_JS_ratio_min"],
    }
    decision = "SUPPORTED" if all(gate_results.values()) else "NOT_SUPPORTED"
    value = {
        "status": "COMPLETE",
        "decision": decision,
        "pairs": len(rows),
        "tasks": len({row["instance_id"] for row in rows}),
        "metrics": metrics,
        "next_action_nll_delta": {
            "mutation_mean": statistics.fmean(mutation_nll),
            "mutation_median": statistics.median(mutation_nll),
            "control_mean": statistics.fmean(control_nll),
            "control_median": statistics.median(control_nll),
            "mutation_higher_pair_fraction": sum(
                a > b for a, b in zip(mutation_nll, control_nll, strict=True)
            )
            / len(rows),
        },
        "frozen_gate_results": gate_results,
        "scope": (
            "file-version motivation under physical lossy-KV splice; not "
            "functional accuracy, latency, or a tuned guard"
        ),
        "next_step": (
            "retain version invalidation as coding-aware mechanism evidence"
            if decision == "SUPPORTED"
            else "do not claim same-path mutation is a proven causal guard"
        ),
    }
    write_json(output / "RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--limit", type=int, default=18)
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
