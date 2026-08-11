#!/usr/bin/env python3
"""Explain why the frozen single-island KV probe did not transfer.

This is a post-hoc diagnostic over an already completed and registered study.
It never changes the registered selector, gates, candidates, or physical
outcomes.  The analysis separates two links that were previously conflated:

    cheap 16-token probe -> full 128-token KV drift -> final-logit change

and reports equal-budget selector medians alongside paired disagreement wins.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/gfy/CodeMAS_Project")
DEFAULT_BASE = (
    ROOT
    / "kvflow-artifacts/impactkv_module_conditioned_attention_kv_20260807/"
    "task_disjoint20"
)
DEFAULT_STUDY = (
    ROOT
    / "kvflow-artifacts/impactkv_single_island_probe_transfer_20260807/"
    "unopened82"
)
DEFAULT_FIGURES = (
    ROOT
    / "sglang-kvflow-worktrees/coding-aware/docs/kvflow/assets/"
    "module_conditioned_attention_kv_20260807"
)
BOOTSTRAP_SEED = 2026080711
BOOTSTRAP_SAMPLES = 2000


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            result[order[position]] = rank
        cursor = end
    return result


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    x = np.asarray(_ranks(left), dtype=np.float64)
    y = np.asarray(_ranks(right), dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator else math.nan


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o644)


def _candidate_key(case_id: str, candidate_id: str) -> str:
    return f"{case_id}::{candidate_id}"


def collect_rows(base: Path, study: Path) -> list[dict[str, Any]]:
    """Join frozen probes, Dense internals, and newly opened outcomes."""

    qualifying = set(_read(base / "CELL_REGISTRATION.json")["qualifying_modules"])
    internals = {}
    for row in _jsonl(base / "INTERNALS.jsonl"):
        for candidate in row["candidates"]:
            internals[
                _candidate_key(str(row["case_id"]), str(candidate["candidate_id"]))
            ] = (row, candidate)

    signals = {
        _candidate_key(str(row["case_id"]), str(candidate["candidate_id"])): candidate
        for row in _jsonl(study / "SIGNALS.jsonl")
        for candidate in row["candidates"]
    }
    outcomes = {
        _candidate_key(str(row["case_id"]), str(candidate["candidate_id"])): (
            row,
            candidate,
        )
        for row in _jsonl(study / "OUTCOMES.jsonl")
        for candidate in row["candidates"]
    }
    if set(signals) != set(outcomes):
        raise ValueError("probe and outcome candidate sets differ")

    arm_registration = _read(study / "ARM_REGISTRATION.json")
    oracle = {
        _candidate_key(str(row["case_id"]), str(candidate_id)): float(value)
        for row in arm_registration["cases"]
        for candidate_id, value in row["module_oracle_risk"].items()
    }

    joined = []
    for key in sorted(signals):
        if key not in internals:
            raise KeyError(f"missing Dense internal: {key}")
        internal_row, candidate = internals[key]
        outcome_row, outcome = outcomes[key]
        full_drift = statistics.fmean(
            float(layer["raw_kv_drift"]) for layer in candidate["layers"]
        )
        attention_values = [
            float(value["attention_mass"])
            for module, value in candidate["module_attention"].items()
            if module in qualifying
        ]
        if not attention_values:
            raise ValueError(f"no qualifying module attention: {key}")
        max_attention = max(attention_values)
        joined.append(
            {
                "candidate_key": key,
                "case_id": str(outcome_row["case_id"]),
                "instance_id": str(outcome_row["instance_id"]),
                "candidate_id": str(outcome["candidate_id"]),
                "probe_score": float(signals[key]["probe_score"]),
                "full_128_token_kv_drift": full_drift,
                "max_qualifying_module_attention": max_attention,
                "max_module_attention_x_full_drift": max_attention * full_drift,
                "module_oracle_risk": oracle.get(key),
                "final_logit_js": float(outcome["final_logit_js"]),
                "top1_changed": bool(outcome["top1_changed"]),
            }
        )
    return joined


def _rho(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> float:
    selected = [
        row
        for row in rows
        if row.get(left) is not None
        and row.get(right) is not None
        and math.isfinite(float(row[left]))
        and math.isfinite(float(row[right]))
    ]
    return _spearman(
        [float(row[left]) for row in selected],
        [float(row[right]) for row in selected],
    )


def task_bootstrap_rho(
    rows: Sequence[Mapping[str, Any]], left: str, right: str
) -> dict[str, float | int]:
    """Task-cluster bootstrap without pretending candidates are independent."""

    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(left) is not None and row.get(right) is not None:
            by_task[str(row["instance_id"])].append(row)
    tasks = sorted(by_task)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(tasks, size=len(tasks), replace=True)
        current = [row for task in sampled for row in by_task[str(task)]]
        value = _rho(current, left, right)
        if math.isfinite(value):
            values.append(value)
    return {
        "tasks": len(tasks),
        "samples": len(values),
        "q025": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "q975": float(np.quantile(values, 0.975)),
    }


def drift_quartiles(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["full_128_token_kv_drift"]))
    groups = np.array_split(np.asarray(ordered, dtype=object), 4)
    result = []
    for index, group in enumerate(groups, 1):
        values = list(group)
        result.append(
            {
                "quartile": index,
                "candidates": len(values),
                "median_full_kv_drift": statistics.median(
                    float(row["full_128_token_kv_drift"]) for row in values
                ),
                "median_final_logit_js": statistics.median(
                    float(row["final_logit_js"]) for row in values
                ),
                "mean_final_logit_js": statistics.fmean(
                    float(row["final_logit_js"]) for row in values
                ),
            }
        )
    return result


def analyze(rows: Sequence[Mapping[str, Any]], registered_result: Mapping[str, Any]) -> dict[str, Any]:
    links = {
        "probe_to_full_kv_drift": _rho(
            rows, "probe_score", "full_128_token_kv_drift"
        ),
        "full_kv_drift_to_final_js": _rho(
            rows, "full_128_token_kv_drift", "final_logit_js"
        ),
        "max_module_attention_to_final_js": _rho(
            rows, "max_qualifying_module_attention", "final_logit_js"
        ),
        "max_module_attention_x_drift_to_final_js": _rho(
            rows, "max_module_attention_x_full_drift", "final_logit_js"
        ),
        "probe_to_final_js": _rho(rows, "probe_score", "final_logit_js"),
        "crossfit_module_oracle_to_final_js": _rho(
            rows, "module_oracle_risk", "final_logit_js"
        ),
    }
    bootstrap = {
        name: task_bootstrap_rho(rows, left, right)
        for name, left, right in (
            (
                "probe_to_full_kv_drift",
                "probe_score",
                "full_128_token_kv_drift",
            ),
            (
                "full_kv_drift_to_final_js",
                "full_128_token_kv_drift",
                "final_logit_js",
            ),
            (
                "max_module_attention_x_drift_to_final_js",
                "max_module_attention_x_full_drift",
                "final_logit_js",
            ),
            ("probe_to_final_js", "probe_score", "final_logit_js"),
        )
    }
    return {
        "status": "COMPLETE",
        "analysis_type": "POSTHOC_FAILURE_ATTRIBUTION_NO_GATE_TUNING",
        "candidates": len(rows),
        "cases": len({str(row["case_id"]) for row in rows}),
        "tasks": len({str(row["instance_id"]) for row in rows}),
        "correlations": links,
        "task_cluster_bootstrap": bootstrap,
        "drift_quartiles": drift_quartiles(rows),
        "immediate_behavior_resolution": {
            "top1_changes": sum(bool(row["top1_changed"]) for row in rows),
            "fraction": statistics.fmean(bool(row["top1_changed"]) for row in rows),
            "interpretation": (
                "one-token top-1 change is too sparse to train or validate a guard"
            ),
        },
        "equal_budget_128_token_arms": registered_result["arms"],
        "registered_decision": registered_result["decision"],
        "diagnosis": (
            "The frozen cheap probe transfers to full KV drift, but neither full "
            "drift nor module-conditioned Attention x drift transfers to final-logit "
            "fidelity on the unopened candidates. The failed link is the target "
            "definition, not the 16-token approximation."
        ),
        "development_decision": (
            "STOP_PROBE_TUNING_AND_SKIP_RUNTIME_CANARY; require a multi-token "
            "action or execution-level behavioral target before another selector"
        ),
    }


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    path.chmod(0o644)


def build_figures(result: Mapping[str, Any], output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    correlations = result["correlations"]
    labels = (
        "16-token probe\n→ full KV drift",
        "Full KV drift\n→ final JS",
        "Module Attention × drift\n→ final JS",
        "16-token probe\n→ final JS",
    )
    names = (
        "probe_to_full_kv_drift",
        "full_kv_drift_to_final_js",
        "max_module_attention_x_drift_to_final_js",
        "probe_to_final_js",
    )
    values = [float(correlations[name]) for name in names]
    intervals = result["task_cluster_bootstrap"]
    errors = np.asarray(
        [
            [values[i] - float(intervals[name]["q025"]) for i, name in enumerate(names)],
            [float(intervals[name]["q975"]) - values[i] for i, name in enumerate(names)],
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.7))
    axes[0].bar(labels, values, color=("#3182bd", "#e6550d", "#31a354", "#756bb1"))
    axes[0].errorbar(range(len(values)), values, yerr=errors, fmt="none", color="#222222", capsize=4)
    axes[0].axhline(0, color="#444444", linewidth=0.9)
    axes[0].set_ylabel("Spearman correlation")
    axes[0].set_title("The proxy tracks drift; drift does not track the final output")
    axes[0].tick_params(axis="x", labelsize=9)
    axes[0].grid(axis="y", alpha=0.25)
    for index, value in enumerate(values):
        axes[0].text(index, value + (0.035 if value >= 0 else -0.055), f"{value:.3f}", ha="center", fontsize=10)

    arm_names = ("fixed_probe_min", "module_attention_oracle", "seeded_random")
    arm_labels = ("Fixed probe", "Module oracle", "Seeded random")
    wins = [
        float(result["equal_budget_128_token_arms"][name]["vs_recency"]["win_fraction"])
        for name in arm_names
    ]
    ratios = [
        float(result["equal_budget_128_token_arms"][name]["vs_recency"]["median_js_ratio_all_cases"])
        for name in arm_names
    ]
    axes[1].bar(arm_labels, wins, color=("#3182bd", "#31a354", "#bdbdbd"))
    axes[1].axhline(0.60, color="#cb181d", linestyle="--", label="Frozen 60% win gate")
    axes[1].set_ylim(0, 0.72)
    axes[1].set_ylabel("Win fraction on disagreements vs recency")
    axes[1].set_title("Lower aggregate medians did not yield reliable paired wins")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(axis="y", alpha=0.25)
    for index, (win, ratio) in enumerate(zip(wins, ratios, strict=True)):
        axes[1].text(index, win + 0.025, f"wins {win:.3f}\nmedian ratio {ratio:.3f}", ha="center", fontsize=9)
    _save(fig, output / "06_single_island_probe_transfer_failure.png")

    quartiles = result["drift_quartiles"]
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    x = [f"Q{row['quartile']}" for row in quartiles]
    medians = [float(row["median_final_logit_js"]) for row in quartiles]
    means = [float(row["mean_final_logit_js"]) for row in quartiles]
    axis.plot(x, medians, marker="o", linewidth=2.2, label="Median final-logit JS")
    axis.plot(x, means, marker="s", linewidth=2.0, label="Mean final-logit JS")
    axis.set_xlabel("Full 128-token KV-drift quartile (low → high)")
    axis.set_ylabel("Final-logit JS")
    axis.set_title("More KV drift did not produce a monotonic final-output penalty")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    _save(fig, output / "07_drift_quartile_final_js.png")
    return (
        "06_single_island_probe_transfer_failure.png",
        "07_drift_quartile_final_js.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    rows = collect_rows(args.base, args.study)
    result = analyze(rows, _read(args.study / "RESULT.json"))
    result["figures"] = build_figures(result, args.figures)
    _write(args.study / "POSTHOC_DIAGNOSTIC.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
