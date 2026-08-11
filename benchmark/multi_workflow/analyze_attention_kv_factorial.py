#!/usr/bin/env python3
"""Exploratory Attention x K/V-deviation factorial analysis.

This script only re-analyzes the frozen M48 single-island observations.  It is
the development half of the module-conditioned motivation experiment: the
cell rule and reporting code are exercised here before a task-disjoint coding
agent cohort is opened.  The analysis is explicitly post-hoc and must not be
reported as functional accuracy evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/gfy/CodeMAS_Project")
DEFAULT_INPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_m48_attention_kv_risk_20260805/full50/"
    "OBSERVATIONS.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_attention_kv_factorial_20260807/"
    "exploratory_m48"
)
BOOTSTRAP_SEED = 2026080701
BOOTSTRAP_DRAWS = 4000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o644)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _median(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return float(statistics.median(values))


def assign_cell(
    *, attention: float, drift: float, attention_median: float, drift_median: float
) -> str:
    """Assign the frozen inclusive-median 2x2 cell."""

    attention_band = "high_attention" if attention >= attention_median else "low_attention"
    drift_band = "high_drift" if drift >= drift_median else "low_drift"
    return f"{attention_band}__{drift_band}"


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2
        for index in order[cursor:end]:
            result[index] = rank
        cursor = end
    return result


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_centered = np.asarray(left, dtype=np.float64) - statistics.fmean(left)
    right_centered = np.asarray(right, dtype=np.float64) - statistics.fmean(right)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0:
        return math.nan
    return float(np.dot(left_centered, right_centered) / denominator)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_ranks(left), _ranks(right))


def _cluster_bootstrap_cell_medians(
    rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int
) -> dict[str, dict[str, float]]:
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    cases = sorted(by_case)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(draws):
        selected = [rng.choice(cases) for _ in cases]
        boot_rows = [row for case_id in selected for row in by_case[case_id]]
        cells: dict[str, list[float]] = defaultdict(list)
        for row in boot_rows:
            cells[str(row["cell"])].append(float(row["causal_splice_logit_js"]))
        for cell, values in cells.items():
            samples[cell].append(_median(values))
    return {
        cell: {
            "q025": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "q975": float(np.quantile(values, 0.975)),
        }
        for cell, values in sorted(samples.items())
        if values
    }


def _interaction_regression(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Fit a transparent log-scale A, D, A*D model for exploration only."""

    attention = np.asarray([float(row["attention_mean"]) for row in rows])
    drift = np.asarray([float(row["kv_cosine_drift_mean"]) for row in rows])
    target = np.log1p(
        np.asarray([float(row["causal_splice_logit_js"]) for row in rows]) * 1e6
    )
    log_attention = np.log(np.maximum(attention, 1e-12))
    log_drift = np.log(np.maximum(drift, 1e-12))
    a = (log_attention - log_attention.mean()) / max(log_attention.std(), 1e-12)
    d = (log_drift - log_drift.mean()) / max(log_drift.std(), 1e-12)
    design = np.column_stack((np.ones_like(a), a, d, a * d))
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coefficients
    return {
        "intercept": float(coefficients[0]),
        "standardized_log_attention": float(coefficients[1]),
        "standardized_log_drift": float(coefficients[2]),
        "attention_drift_interaction": float(coefficients[3]),
        "prediction_spearman": _spearman(prediction.tolist(), target.tolist()),
    }


def _plot_cells(result: Mapping[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    order = (
        "low_attention__low_drift",
        "high_attention__low_drift",
        "low_attention__high_drift",
        "high_attention__high_drift",
    )
    labels = ("Low A\nLow D", "High A\nLow D", "Low A\nHigh D", "High A\nHigh D")
    colors = ("#9ecae1", "#6baed6", "#fdae6b", "#e6550d")
    medians = [result["cells"][cell]["median_logit_js"] for cell in order]
    intervals = result["cluster_bootstrap"]
    lower = [medians[i] - intervals[cell]["q025"] for i, cell in enumerate(order)]
    upper = [intervals[cell]["q975"] - medians[i] for i, cell in enumerate(order)]
    fig, axis = plt.subplots(figsize=(8.6, 5.6))
    axis.bar(labels, medians, color=colors, edgecolor="#333333", linewidth=0.7)
    axis.errorbar(
        range(4), medians, yerr=[lower, upper], fmt="none", color="#222222", capsize=4
    )
    axis.set_yscale("log")
    axis.set_ylabel("Single-island final-logit JS (log scale)")
    axis.set_title("Exploratory Attention × KV-deviation factorial (294 candidates)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(input_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    source = _load_rows(input_path)
    candidates = [
        {"case_id": str(row["case_id"]), **candidate}
        for row in source
        if row.get("status") == "ok"
        for candidate in row.get("candidates", [])
    ]
    required = ("attention_mean", "kv_cosine_drift_mean", "causal_splice_logit_js")
    candidates = [
        row
        for row in candidates
        if all(math.isfinite(float(row[key])) for key in required)
    ]
    if not candidates:
        raise ValueError("no complete M48 candidate observations")
    attention_median = _median([float(row["attention_mean"]) for row in candidates])
    drift_median = _median([float(row["kv_cosine_drift_mean"]) for row in candidates])
    for row in candidates:
        row["cell"] = assign_cell(
            attention=float(row["attention_mean"]),
            drift=float(row["kv_cosine_drift_mean"]),
            attention_median=attention_median,
            drift_median=drift_median,
        )
    output.mkdir(parents=True)
    registration = {
        "status": "REGISTERED_POST_HOC_EXPLORATORY",
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "scope": "design-set mechanism exploration; not independent confirmation or accuracy",
        "cell_rule": {
            "attention": "global inclusive median of Dense target attention_mean",
            "drift": "global inclusive median of RoPE-corrected kv_cosine_drift_mean",
            "outcome_not_used_for_cell_assignment": True,
        },
        "bootstrap": {
            "cluster": "case_id",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
        },
    }
    _write_json(output / "REGISTRATION.json", registration)
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_cell[str(row["cell"])].append(row)
    cells = {}
    for cell, rows in sorted(by_cell.items()):
        js = [float(row["causal_splice_logit_js"]) for row in rows]
        cells[cell] = {
            "candidates": len(rows),
            "cases": len({str(row["case_id"]) for row in rows}),
            "median_logit_js": _median(js),
            "mean_logit_js": float(statistics.fmean(js)),
            "top1_changed": sum(bool(row.get("causal_splice_top1_changed")) for row in rows),
        }
    bootstrap = _cluster_bootstrap_cell_medians(
        candidates, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    high_high = cells["high_attention__high_drift"]["median_logit_js"]
    low_high = cells["low_attention__high_drift"]["median_logit_js"]
    result = {
        "status": "COMPLETE",
        "cases": len({str(row["case_id"]) for row in candidates}),
        "candidate_observations": len(candidates),
        "thresholds": {
            "attention_median": attention_median,
            "kv_drift_median": drift_median,
        },
        "cell_counts": dict(Counter(str(row["cell"]) for row in candidates)),
        "cells": cells,
        "cluster_bootstrap": bootstrap,
        "high_attention_vs_low_attention_at_high_drift_js_ratio": high_high / low_high,
        "correlations": {
            "attention_to_js_spearman": _spearman(
                [float(row["attention_mean"]) for row in candidates],
                [float(row["causal_splice_logit_js"]) for row in candidates],
            ),
            "drift_to_js_spearman": _spearman(
                [float(row["kv_cosine_drift_mean"]) for row in candidates],
                [float(row["causal_splice_logit_js"]) for row in candidates],
            ),
            "attention_times_drift_to_js_spearman": _spearman(
                [
                    float(row["attention_mean"]) * float(row["kv_cosine_drift_mean"])
                    for row in candidates
                ],
                [float(row["causal_splice_logit_js"]) for row in candidates],
            ),
        },
        "continuous_interaction_regression": _interaction_regression(candidates),
        "interpretation": (
            "Feasibility-only factorial evidence. Cell thresholds and all outcomes come "
            "from the existing M48 development cohort; a task-disjoint module-conditioned "
            "experiment is required before policy use."
        ),
    }
    _write_json(output / "RESULT.json", result)
    with (output / "CANDIDATES.csv").open("w", newline="", encoding="utf-8") as stream:
        keys = (
            "case_id",
            "candidate_id",
            "context_index",
            "attention_mean",
            "kv_cosine_drift_mean",
            "causal_splice_logit_js",
            "causal_splice_top1_changed",
            "cell",
        )
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)
    _plot_cells(result, output / "01_attention_kv_factorial.png")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
