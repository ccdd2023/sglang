#!/usr/bin/env python3
"""Build paper-readable figures for the module-conditioned motivation study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/gfy/CodeMAS_Project")
EXPLORATORY = (
    ROOT
    / "kvflow-artifacts/impactkv_attention_kv_factorial_20260807/"
    "exploratory_m48/RESULT.json"
)
CONFIRMATORY = (
    ROOT
    / "kvflow-artifacts/impactkv_module_conditioned_attention_kv_20260807/"
    "task_disjoint20/RESULT.json"
)
MULTI = CONFIRMATORY.parent / "MULTI_RESULT.json"
DEFAULT_OUTPUT = (
    ROOT
    / "sglang-kvflow-worktrees/coding-aware/docs/kvflow/assets/"
    "module_conditioned_attention_kv_20260807"
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    path.chmod(0o644)


def _exploratory(result: dict[str, Any], output: Path) -> None:
    order = (
        "low_attention__low_drift",
        "high_attention__low_drift",
        "low_attention__high_drift",
        "high_attention__high_drift",
    )
    labels = ("Low A\nLow D", "High A\nLow D", "Low A\nHigh D", "High A\nHigh D")
    values = [result["cells"][cell]["median_logit_js"] for cell in order]
    intervals = result["cluster_bootstrap"]
    errors = np.asarray(
        [
            [values[index] - intervals[cell]["q025"] for index, cell in enumerate(order)],
            [intervals[cell]["q975"] - values[index] for index, cell in enumerate(order)],
        ]
    )
    fig, axis = plt.subplots(figsize=(8.8, 5.8))
    axis.bar(labels, values, color=("#9ecae1", "#6baed6", "#fdae6b", "#e6550d"))
    axis.errorbar(range(4), values, yerr=errors, fmt="none", color="#222222", capsize=4)
    axis.set_yscale("log")
    axis.set_ylabel("Final-logit JS (log scale)")
    axis.set_title("Development cohort: high KV drift is most harmful when attention is high")
    axis.grid(axis="y", alpha=0.25)
    _save(fig, output / "01_exploratory_factorial.png")


def _confirmatory(result: dict[str, Any], output: Path) -> None:
    modules = result["qualifying_modules"]
    cells = (
        "low_attention__low_drift",
        "high_attention__low_drift",
        "low_attention__high_drift",
        "high_attention__high_drift",
    )
    labels = ("Low A / Low D", "High A / Low D", "Low A / High D", "High A / High D")
    x = np.arange(len(modules))
    width = 0.19
    fig, axis = plt.subplots(figsize=(12.8, 6.4))
    for index, (cell, label) in enumerate(zip(cells, labels, strict=True)):
        values = [result["factorial"][module][cell]["median_local_output_change"] for module in modules]
        axis.bar(x + (index - 1.5) * width, values, width, label=label)
    axis.set_xticks(x, [module.replace("read_observation_", "read: ").replace("_", "\n") for module in modules])
    axis.set_yscale("log")
    axis.set_ylabel("Local attention-output relative change (log scale)")
    axis.set_title("Task-disjoint confirmation by downstream prompt module")
    axis.legend(ncol=2, frameon=False)
    axis.grid(axis="y", alpha=0.25)
    _save(fig, output / "02_confirmatory_module_factorial.png")

    matrix = np.asarray(
        [
            [result["factorial"][module][cell]["median_local_output_change"] for cell in cells]
            for module in modules
        ],
        dtype=float,
    )
    log_matrix = np.log10(np.maximum(matrix, 1e-12))
    fig, axis = plt.subplots(figsize=(11.2, max(4.8, 1.0 * len(modules) + 2.0)))
    image = axis.imshow(log_matrix, cmap="YlOrRd", aspect="auto")
    axis.set_xticks(range(len(cells)), labels)
    axis.set_yticks(
        range(len(modules)),
        [module.replace("read_observation_", "read: ").replace("_", " ") for module in modules],
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2e}",
                ha="center",
                va="center",
                color="white" if log_matrix[row_index, column_index] > np.nanmedian(log_matrix) else "#222222",
                fontsize=9,
            )
    axis.set_title("Where does stale KV enter the computation? (module × Attention/KV cell)")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("log10 local attention-output relative change")
    _save(fig, output / "02b_confirmatory_module_heatmap.png")

    held = result["leave_one_task_out"]
    fig, axis = plt.subplots(figsize=(7.8, 5.6))
    values = (
        held["baseline_drift_module_spearman"],
        held["module_attention_interaction_spearman"],
    )
    axis.bar(("KV drift + module", "Module Attention × KV"), values, color=("#9e9ac8", "#31a354"))
    axis.set_ylim(min(0, min(values) - 0.05), max(values) + 0.08)
    axis.set_ylabel("Leave-one-task-out Spearman")
    axis.set_title("Does module-conditioned attention improve held-out local-risk ranking?")
    axis.grid(axis="y", alpha=0.25)
    _save(fig, output / "03_held_out_risk_prediction.png")

    components = result["physical_component_js"]
    fig, axis = plt.subplots(figsize=(7.8, 5.6))
    names = ("key_only", "value_only", "kv")
    axis.bar(("K only", "V only", "K + V"), [components[name]["median"] for name in names], color=("#3182bd", "#756bb1", "#e6550d"))
    axis.set_yscale("log")
    axis.set_ylabel("Median final-logit JS (log scale)")
    axis.set_title("Physical component ablation on the frozen intervention subset")
    axis.grid(axis="y", alpha=0.25)
    _save(fig, output / "04_kv_component_ablation.png")


def _multi(result: dict[str, Any], output: Path) -> None:
    arms = ("current_recency", "module_risk_then_path_utility", "seeded_random")
    labels = ("Current recency", "Risk → path utility", "Seeded random")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    for axis, metric, title in (
        (axes[0], "final_logit_js", "Three-island final-logit JS"),
        (axes[1], "attention_row_tv_mean", "Three-island local attention-row TV"),
    ):
        values = [result["arms"][arm][metric] for arm in arms]
        axis.bar(labels, values, color=("#9e9ac8", "#31a354", "#bdbdbd"))
        axis.set_yscale("log")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    _save(fig, output / "05_multi_island_comparison.png")


def build(exploratory: Path, confirmatory: Path, multi: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    _exploratory(_read(exploratory), output)
    generated.append("01_exploratory_factorial.png")
    if confirmatory.exists():
        _confirmatory(_read(confirmatory), output)
        generated.extend(
            (
                "02_confirmatory_module_factorial.png",
                "02b_confirmatory_module_heatmap.png",
                "03_held_out_risk_prediction.png",
                "04_kv_component_ablation.png",
            )
        )
    if multi.exists():
        _multi(_read(multi), output)
        generated.append("05_multi_island_comparison.png")
    value = {"status": "COMPLETE", "generated": generated}
    manifest = output / "FIGURE_MANIFEST.json"
    manifest.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest.chmod(0o644)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exploratory", type=Path, default=EXPLORATORY)
    parser.add_argument("--confirmatory", type=Path, default=CONFIRMATORY)
    parser.add_argument("--multi", type=Path, default=MULTI)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.exploratory, args.confirmatory, args.multi, args.output), indent=2))


if __name__ == "__main__":
    main()
