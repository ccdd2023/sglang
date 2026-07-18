#!/usr/bin/env python3
"""Calibrate ProbeHead V12 and gate sequential development/holdout composition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from benchmark.multi_workflow.probehead_v12 import (
    BOOTSTRAP_ITERATIONS,
    HEAD_CANDIDATES,
    JS_LIMIT,
    MAX_PROBE_P95_MS,
    MIN_HARM_REDUCTION,
    MIN_PROMPT_COPY_FRACTION,
    PROFILE,
    ProbeCandidate,
    decide_probe_candidates,
)
from benchmark.multi_workflow.sessiongraph_v11 import CostModel, read_jsonl


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _p95(values: Sequence[float]) -> float:
    return float(np.quantile(values, 0.95)) if values else math.inf


def _registration(path: Path, design_path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("policy") != PROFILE:
        raise ValueError("registration is not ProbeHead V12")
    if value.get("design_sha256") != _sha(design_path):
        raise ValueError("design does not match the frozen registration")
    return value


def _executor_amendment(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if {
        "accepted": value.get("accepted"),
        "probe_warmup_iterations": value.get("probe_warmup_iterations"),
        "timed_scope": value.get("timed_scope"),
        "thresholds_changed": value.get("thresholds_changed"),
        "holdout_opened": value.get("holdout_opened"),
    } != {
        "accepted": True,
        "probe_warmup_iterations": 3,
        "timed_scope": "vectorized KV comparison only",
        "thresholds_changed": False,
        "holdout_opened": False,
    }:
        raise ValueError("invalid ProbeHead reference executor amendment")
    return value


def _candidate(row: Mapping[str, Any]) -> ProbeCandidate:
    return ProbeCandidate(
        session_id=str(row["session_id"]),
        turn_id=int(row["turn_id"]),
        module_id=str(row["module_id"]),
        source_start=int(row["source_start"]),
        target_start=int(row["target_start"]),
        length=int(row["token_count"]),
        prompt_tokens=int(row["target_prompt_tokens"]),
    )


def _observation_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["case_kind"]),
        str(row["cohort"]),
        str(row["session_id"]),
        int(row["turn_id"]),
        str(row["module_id"]),
        str(row["disturbance"]),
        int(row["head_tokens"]),
    )


def _configuration_metrics(
    *,
    rows: Sequence[Mapping[str, Any]],
    head_tokens: int,
    threshold: float,
    cost_model: CostModel,
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["head_tokens"]) == head_tokens:
            grouped[(str(row["session_id"]), int(row["turn_id"]))].append(row)
    fractions, selected_js, top1_changes = [], [], 0
    reductions_by_session: dict[str, list[float]] = defaultdict(list)
    for (session_id, _), request_rows in grouped.items():
        scores = {
            str(row["module_id"]): float(row["probe_score"])
            for row in request_rows
        }
        probe_p95_us = _p95(
            [float(row["probe_ms"]) for row in request_rows]
        ) * 1000
        decisions = decide_probe_candidates(
            candidates=[_candidate(row) for row in request_rows],
            scores=scores,
            head_tokens=head_tokens,
            threshold=threshold,
            cost_model=cost_model,
            probe_compare_us=probe_p95_us,
        )
        by_module = {str(row["module_id"]): row for row in request_rows}
        fractions.append(
            sum(value.copied_tokens for value in decisions)
            / int(request_rows[0]["target_prompt_tokens"])
        )
        accepted = [
            by_module[value.candidate.module_id]
            for value in decisions
            if value.accepted
        ]
        selected_js.extend(
            float(row["causal_splice_logit_js"]) for row in accepted
        )
        top1_changes += sum(
            int(row["splice_top1_changed"]) for row in accepted
        )
        baseline = mean(
            float(row["causal_splice_logit_js"]) for row in request_rows
        )
        selected = (
            mean(float(row["causal_splice_logit_js"]) for row in accepted)
            if accepted
            else 0.0
        )
        if baseline > 0:
            reductions_by_session[session_id].append(
                (baseline - selected) / baseline
            )
    reductions = [
        mean(values) for values in reductions_by_session.values() if values
    ]
    return {
        "head_tokens": head_tokens,
        "threshold": threshold,
        "requests": len(grouped),
        "median_cost_positive_copy_fraction": median(fractions)
        if fractions
        else 0.0,
        "selected_modules": len(selected_js),
        "selected_splice_p95_js": _p95(selected_js),
        "selected_splice_top1_changes": top1_changes,
        "copy_all_harm_reduction": mean(reductions) if reductions else math.nan,
    }


def calibrate(
    *,
    observations_path: Path,
    design_path: Path,
    registration_path: Path,
    executor_amendment_path: Path,
    cost_gate_path: Path,
    lock_output: Path,
    report_output: Path,
) -> dict[str, Any]:
    registration = _registration(registration_path, design_path)
    _executor_amendment(executor_amendment_path)
    rows = [
        row
        for row in read_jsonl(observations_path)
        if row.get("case_kind") == "workflow"
        and row.get("cohort") == "development"
    ]
    if not rows or any(row.get("status") != "ok" for row in rows):
        raise ValueError("development probe observations are absent or invalid")
    expected_rows = [
        row
        for row in read_jsonl(design_path)
        if row.get("case_kind") == "workflow"
        and row.get("cohort") == "development"
    ]
    expected_keys = {_observation_key(row) for row in expected_rows}
    observed_keys = [_observation_key(row) for row in rows]
    duplicates = [
        key for key, count in Counter(observed_keys).items() if count != 1
    ]
    if (
        duplicates
        or set(observed_keys) != expected_keys
        or len(observed_keys) != len(expected_rows)
    ):
        raise ValueError(
            "development observations do not exactly cover the frozen "
            "workflow design"
        )
    counts: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    for row in rows:
        counts[
            (
                str(row["session_id"]),
                int(row["turn_id"]),
                str(row["module_id"]),
            )
        ].add(int(row["head_tokens"]))
    incomplete = [key for key, heads in counts.items() if heads != set(HEAD_CANDIDATES)]
    if incomplete:
        raise ValueError(
            f"development observations have {len(incomplete)} incomplete head grids"
        )
    cost_gate = json.loads(cost_gate_path.read_text(encoding="utf-8"))
    cost_model = CostModel(**cost_gate["cost_model"])
    evaluated = []
    for head in HEAD_CANDIDATES:
        thresholds = sorted(
            {
                float(row["probe_score"])
                for row in rows
                if int(row["head_tokens"]) == head
            }
        )
        for threshold in thresholds:
            metrics = _configuration_metrics(
                rows=rows,
                head_tokens=head,
                threshold=threshold,
                cost_model=cost_model,
            )
            metrics["feasible"] = (
                metrics["median_cost_positive_copy_fraction"]
                >= MIN_PROMPT_COPY_FRACTION
                and metrics["selected_splice_p95_js"] <= JS_LIMIT
                and metrics["selected_splice_top1_changes"] == 0
                and metrics["copy_all_harm_reduction"] >= MIN_HARM_REDUCTION
            )
            evaluated.append(metrics)
    feasible = [row for row in evaluated if row["feasible"]]
    chosen = (
        sorted(
            feasible,
            key=lambda row: (
                -row["median_cost_positive_copy_fraction"],
                row["selected_splice_p95_js"],
                row["head_tokens"],
                row["threshold"],
            ),
        )[0]
        if feasible
        else None
    )
    report = {
        "passed": chosen is not None,
        "status": "CALIBRATED" if chosen else "FALSIFIED",
        "observations": len(rows),
        "candidate_modules": len(counts),
        "configurations_evaluated": len(evaluated),
        "feasible_configurations": len(feasible),
        "chosen": chosen,
        "gates": registration["gates"],
        "development_observations_sha256": _sha(observations_path),
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lock = {
        "status": "LOCKED" if chosen else "FALSIFIED",
        "policy": PROFILE,
        "head_tokens": chosen["head_tokens"] if chosen else None,
        "threshold": chosen["threshold"] if chosen else None,
        "selection_rule": registration["calibration_rule"],
        "registration_sha256": _sha(registration_path),
        "design_sha256": _sha(design_path),
        "development_observations_sha256": _sha(observations_path),
        "cost_gate_sha256": _sha(cost_gate_path),
        "executor_amendment_sha256": _sha(executor_amendment_path),
        "holdout_measurements_read": False,
    }
    lock_output.parent.mkdir(parents=True, exist_ok=True)
    lock_output.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _cluster_reduction(
    rows: Sequence[Mapping[str, Any]],
    baseline_key: str,
    iterations: int,
) -> tuple[float, float]:
    by_session: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        baseline = float(row[baseline_key])
        probe = float(row["probe_composed_js"])
        if baseline > 0:
            by_session[str(row["session_id"])].append(
                (baseline - probe) / baseline
            )
    reductions = [mean(values) for values in by_session.values() if values]
    if not reductions:
        return math.nan, math.nan
    rng = random.Random(20260717 if baseline_key.startswith("copy") else 1729)
    draws = sorted(
        mean(rng.choice(reductions) for _ in reductions)
        for _ in range(iterations)
    )
    return mean(reductions), draws[int(0.025 * len(draws))]


def gate_composition(
    *,
    stage: str,
    module_observations_path: Path,
    request_observations_path: Path,
    design_path: Path,
    registration_path: Path,
    calibration_lock_path: Path,
    executor_amendment_path: Path,
    output_path: Path,
    verdict_path: Path,
    iterations: int,
) -> dict[str, Any]:
    registration = _registration(registration_path, design_path)
    _executor_amendment(executor_amendment_path)
    lock = json.loads(calibration_lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED":
        raise ValueError("composition gate requires a locked calibration")
    if lock.get("registration_sha256") != _sha(registration_path):
        raise ValueError("calibration lock registration mismatch")
    if lock.get("executor_amendment_sha256") != _sha(executor_amendment_path):
        raise ValueError("calibration lock executor amendment mismatch")
    cohort = "development" if stage == "development-compose" else "holdout"
    module_rows = [
        row
        for row in read_jsonl(module_observations_path)
        if row.get("cohort") == cohort
        and int(row["head_tokens"]) == int(lock["head_tokens"])
    ]
    request_rows = [
        row
        for row in read_jsonl(request_observations_path)
        if row.get("cohort") == cohort
    ]
    reasons = []
    expected_module_rows = [
        row
        for row in read_jsonl(design_path)
        if row.get("cohort") == cohort
        and int(row["head_tokens"]) == int(lock["head_tokens"])
    ]
    expected_module_keys = {
        _observation_key(row) for row in expected_module_rows
    }
    module_keys = [_observation_key(row) for row in module_rows]
    duplicate_module_keys = sum(
        count - 1 for count in Counter(module_keys).values() if count > 1
    )
    missing_module_keys = len(expected_module_keys - set(module_keys))
    extra_module_keys = len(set(module_keys) - expected_module_keys)
    if (
        duplicate_module_keys
        or missing_module_keys
        or extra_module_keys
        or len(module_rows) != len(expected_module_rows)
    ):
        reasons.append("module observations do not exactly cover frozen design")
    invalid = sum(row.get("status") != "ok" for row in module_rows) + sum(
        row.get("status") != "ok" for row in request_rows
    )
    request_keys = [
        (str(row["session_id"]), int(row["turn_id"])) for row in request_rows
    ]
    duplicate_requests = len(request_keys) - len(set(request_keys))
    sessions = {key[0] for key in request_keys}
    if len(request_rows) != 96 or len(sessions) != 32 or duplicate_requests:
        reasons.append("request coverage is not exactly 32 sessions / 96 requests")
    if invalid:
        reasons.append("measurement rows are invalid")
    controls = [
        float(row["causal_splice_logit_js"])
        for row in module_rows
        if row.get("case_kind") == "stress"
        and row.get("disturbance") in {"identity", "change_after"}
    ]
    negative_control_max = max(controls) if controls else math.inf
    if negative_control_max > JS_LIMIT:
        reasons.append("negative-control max JS exceeds 1e-3")
    fractions = [
        float(row["cost_positive_copy_fraction"]) for row in request_rows
    ]
    median_fraction = median(fractions) if fractions else 0.0
    if median_fraction < MIN_PROMPT_COPY_FRACTION:
        reasons.append("median cost-positive copy fraction is below 15%")
    composed_p95 = _p95(
        [float(row["probe_composed_js"]) for row in request_rows]
    )
    if composed_p95 > JS_LIMIT:
        reasons.append("composed-splice p95 JS exceeds 1e-3")
    top1_changes = sum(int(row["probe_top1_changed"]) for row in request_rows)
    if top1_changes:
        reasons.append("composed splice changes teacher top-1")
    copy_reduction, copy_low = _cluster_reduction(
        request_rows, "copy_all_composed_js", iterations
    )
    shuffled_reduction, shuffled_low = _cluster_reduction(
        request_rows, "shuffled_composed_js", iterations
    )
    if not (
        copy_reduction >= MIN_HARM_REDUCTION
        and copy_low > 0
    ):
        reasons.append("copy-all harm-reduction gate failed")
    if not (
        shuffled_reduction >= MIN_HARM_REDUCTION
        and shuffled_low > 0
    ):
        reasons.append("shuffled harm-reduction gate failed")
    probe_p95 = _p95(
        [float(row["probe_p95_ms"]) for row in request_rows]
    )
    if probe_p95 >= MAX_PROBE_P95_MS:
        reasons.append("probe comparison p95 is not below 2ms")
    result = {
        "passed": not reasons,
        "status": "PASS" if not reasons else "FALSIFIED",
        "stage": stage,
        "reasons": reasons,
        "sessions": len(sessions),
        "requests": len(request_rows),
        "duplicate_request_keys": duplicate_requests,
        "duplicate_module_keys": duplicate_module_keys,
        "missing_module_keys": missing_module_keys,
        "extra_module_keys": extra_module_keys,
        "invalid_rows": invalid,
        "negative_control_max_js": negative_control_max,
        "median_cost_positive_copy_fraction": median_fraction,
        "composed_splice_p95_js": composed_p95,
        "composed_splice_top1_changes": top1_changes,
        "copy_all_harm_reduction": copy_reduction,
        "copy_all_harm_reduction_ci_low": copy_low,
        "shuffled_harm_reduction": shuffled_reduction,
        "shuffled_harm_reduction_ci_low": shuffled_low,
        "probe_p95_ms": probe_p95,
        "bootstrap_iterations": iterations,
        "head_tokens": int(lock["head_tokens"]),
        "threshold": float(lock["threshold"]),
        "inputs": {
            "registration": _sha(registration_path),
            "design": _sha(design_path),
            "calibration_lock": _sha(calibration_lock_path),
            "executor_amendment": _sha(executor_amendment_path),
            "module_observations": _sha(module_observations_path),
            "request_observations": _sha(request_observations_path),
        },
        "claim_scope": (
            "ProbeHead P0 signal only; no workflow accuracy or TTFT claim"
        ),
        "v11_thresholds_changed": False,
        "paper_modified": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verdict_path.write_text(
        "\n".join(
            [
                f"# ProbeHead StateSensitivityKV V12 — {stage}",
                "",
                f"**{result['status']}**",
                "",
                f"- Sessions / requests: {len(sessions)} / {len(request_rows)}",
                f"- Median cost-positive copy fraction: {median_fraction:.4%}",
                f"- Composed splice p95 JS: {composed_p95:.8g}",
                f"- Copy-all harm reduction: {copy_reduction:.4f} "
                f"(CI low {copy_low:.4f})",
                f"- Shuffled harm reduction: {shuffled_reduction:.4f} "
                f"(CI low {shuffled_low:.4f})",
                f"- Probe p95: {probe_p95:.4f} ms",
                "",
                "This verdict does not open workflow accuracy, TTFT, or P1.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--observations", type=Path, required=True)
    calibration.add_argument("--design", type=Path, required=True)
    calibration.add_argument("--registration", type=Path, required=True)
    calibration.add_argument("--executor-amendment", type=Path, required=True)
    calibration.add_argument("--cost-gate", type=Path, required=True)
    calibration.add_argument("--lock-output", type=Path, required=True)
    calibration.add_argument("--report-output", type=Path, required=True)

    gate = subparsers.add_parser("gate")
    gate.add_argument(
        "--stage",
        choices=("development-compose", "holdout"),
        required=True,
    )
    gate.add_argument("--module-observations", type=Path, required=True)
    gate.add_argument("--request-observations", type=Path, required=True)
    gate.add_argument("--design", type=Path, required=True)
    gate.add_argument("--registration", type=Path, required=True)
    gate.add_argument("--calibration-lock", type=Path, required=True)
    gate.add_argument("--executor-amendment", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--verdict", type=Path, required=True)
    gate.add_argument("--bootstrap", type=int, default=BOOTSTRAP_ITERATIONS)
    args = parser.parse_args()
    if args.command == "calibrate":
        result = calibrate(
            observations_path=args.observations,
            design_path=args.design,
            registration_path=args.registration,
            executor_amendment_path=args.executor_amendment,
            cost_gate_path=args.cost_gate,
            lock_output=args.lock_output,
            report_output=args.report_output,
        )
    else:
        result = gate_composition(
            stage=args.stage,
            module_observations_path=args.module_observations,
            request_observations_path=args.request_observations,
            design_path=args.design,
            registration_path=args.registration,
            calibration_lock_path=args.calibration_lock,
            executor_amendment_path=args.executor_amendment,
            output_path=args.output,
            verdict_path=args.verdict,
            iterations=args.bootstrap,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
