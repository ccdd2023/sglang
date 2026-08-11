#!/usr/bin/env python3
"""Freeze a task-grouped lower bound for cache-ready reuse TTFT saving.

This calibration consumes the already completed 56-target exact-prompt replay.
It never reads agent answers or SWE-bench outcomes.  The predictor is total
copied-island attention work and the response is paired median Dense TTFT minus
reuse TTFT.  Tasks, rather than target requests, are assigned to folds so that
nearby requests from one trajectory cannot leak across validation folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
PATHS = RuntimePaths.from_project(PROJECT)
DEFAULT_INPUT = (
    PATHS.artifacts
    / "impactkv_natural_code_cost_agent_20260808/"
    "exact_prompt_speed"
)
DEFAULT_OUTPUT = (
    PATHS.artifacts
    / "impactkv_dependency_graph_lcb_20260811/"
    "CALIBRATION.json"
)
TASK_PATTERN = re.compile(r"-m(\d+)-")
FOLDS = 5
RESIDUAL_QUANTILE = 0.10


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def linear_fit(rows: list[dict[str, Any]]) -> tuple[float, float]:
    x_mean = statistics.mean(float(row["attention_work_10k"]) for row in rows)
    y_mean = statistics.mean(float(row["observed_saving_ms"]) for row in rows)
    denominator = sum(
        (float(row["attention_work_10k"]) - x_mean) ** 2 for row in rows
    )
    if denominator <= 0:
        raise ValueError("calibration predictor has zero variance")
    slope = sum(
        (float(row["attention_work_10k"]) - x_mean)
        * (float(row["observed_saving_ms"]) - y_mean)
        for row in rows
    ) / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires observations")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def balanced_task_folds(task_counts: Counter[str]) -> list[list[str]]:
    """Assign whole tasks greedily to five deterministic, balanced folds."""

    folds: list[list[str]] = [[] for _ in range(FOLDS)]
    loads = [0] * FOLDS
    for task, count in sorted(task_counts.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(FOLDS), key=lambda index: (loads[index], index))
        folds[fold].append(task)
        loads[fold] += count
    return folds


def build_rows(input_dir: Path) -> list[dict[str, Any]]:
    plan = read_json(input_dir / "PLAN.json")["groups"]
    dense = read_json(input_dir / "dense.json")["targets"]
    reuse = read_json(input_dir / "reuse.json")["targets"]

    def measured_median(rows: list[dict[str, Any]], group_index: int) -> float:
        values = [
            float(row["ttft_ms"])
            for row in rows
            if int(row["group_index"]) == group_index and not row["warmup"]
        ]
        if len(values) != 3:
            raise ValueError(
                f"group {group_index} has {len(values)} measured rounds"
            )
        return statistics.median(values)

    rows: list[dict[str, Any]] = []
    for group in plan:
        group_index = int(group["group_index"])
        match = TASK_PATTERN.search(str(group["original_target_group_id"]))
        if match is None:
            raise ValueError("target group does not expose its frozen task key")
        prompt_tokens = len(group["target_input_ids"])
        copied_tokens = sum(int(case["length"]) for case in group["cases"])
        dense_ttft = measured_median(dense, group_index)
        reuse_ttft = measured_median(reuse, group_index)
        rows.append(
            {
                "group_index": group_index,
                "task_key": f"m{match.group(1)}",
                "target_prompt_tokens": prompt_tokens,
                "copied_tokens": copied_tokens,
                "islands": len(group["cases"]),
                "attention_work_10k": copied_tokens * prompt_tokens / 10_000,
                "dense_median_ttft_ms": dense_ttft,
                "reuse_median_ttft_ms": reuse_ttft,
                "observed_saving_ms": dense_ttft - reuse_ttft,
            }
        )
    return rows


def calibrate(input_dir: Path) -> dict[str, Any]:
    rows = build_rows(input_dir)
    slope, intercept = linear_fit(rows)
    response_mean = statistics.mean(
        float(row["observed_saving_ms"]) for row in rows
    )
    residual_sum = sum(
        (
            float(row["observed_saving_ms"])
            - (
                slope * float(row["attention_work_10k"])
                + intercept
            )
        )
        ** 2
        for row in rows
    )
    total_sum = sum(
        (float(row["observed_saving_ms"]) - response_mean) ** 2
        for row in rows
    )

    counts = Counter(str(row["task_key"]) for row in rows)
    folds = balanced_task_folds(counts)
    cross_validated_residuals: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_index, held_out_tasks in enumerate(folds):
        held_out = set(held_out_tasks)
        training = [row for row in rows if row["task_key"] not in held_out]
        validation = [row for row in rows if row["task_key"] in held_out]
        fold_slope, fold_intercept = linear_fit(training)
        residuals = [
            float(row["observed_saving_ms"])
            - (
                fold_slope * float(row["attention_work_10k"])
                + fold_intercept
            )
            for row in validation
        ]
        cross_validated_residuals.extend(residuals)
        fold_rows.append(
            {
                "fold": fold_index,
                "held_out_tasks": held_out_tasks,
                "validation_targets": len(validation),
                "training_slope_ms_per_work_10k": fold_slope,
                "training_intercept_ms": fold_intercept,
                "validation_residual_median_ms": statistics.median(residuals),
            }
        )

    residual_q10 = quantile(cross_validated_residuals, RESIDUAL_QUANTILE)
    return {
        "status": "FROZEN_BEFORE_NEW_ACCURACY_OUTCOMES",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": (
            "engineering TTFT admission calibration; not an accuracy proxy"
        ),
        "input": {
            "directory": str(input_dir),
            "plan_sha256": sha256(input_dir / "PLAN.json"),
            "dense_sha256": sha256(input_dir / "dense.json"),
            "reuse_sha256": sha256(input_dir / "reuse.json"),
        },
        "coverage": {
            "target_groups": len(rows),
            "tasks": len(counts),
            "task_target_counts": dict(sorted(counts.items())),
            "folds": FOLDS,
            "measured_rounds_per_arm": 3,
        },
        "model": {
            "formula": (
                "predicted_ms = slope * "
                "(copied_tokens * target_prompt_tokens / 10000) + intercept"
            ),
            "slope_ms_per_work_10k": slope,
            "intercept_ms": intercept,
            "r2_in_sample": 1 - residual_sum / total_sum,
            "cross_validated_residual_quantile": RESIDUAL_QUANTILE,
            "cross_validated_residual_q10_ms": residual_q10,
            "online_formula": "lower_bound_ms = predicted_ms + residual_q10_ms",
            "online_admission": "lower_bound_ms > 0",
            "source_build_included": False,
        },
        "folds": fold_rows,
        "claim_limits": [
            "The replay contains 56 targets from seven tasks and is imbalanced.",
            "The lower bound is an empirical engineering guard, not a confidence interval.",
            "New accuracy outcomes were not used to fit or choose the coefficients.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = calibrate(args.input.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
