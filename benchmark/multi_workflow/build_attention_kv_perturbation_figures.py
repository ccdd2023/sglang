#!/usr/bin/env python3
"""Build figures for the attention-weighted K/V perturbation motivation audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT.parents[1] / "kvflow-artifacts"
DEFAULT_BOUND = (
    ARTIFACT_ROOT
    / "impactkv_attention_kv_bound_20260806/frozen26_mass_aware"
)
DEFAULT_GLOBAL = (
    ARTIFACT_ROOT
    / "impactkv_global_block_attention_20260806/frozen26_r2"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/kvflow/assets/attention_kv_theory_20260806"

BLUE = "#4C78A8"
ORANGE = "#F28E2B"
GREEN = "#59A14F"
RED = "#E15759"
GRAY = "#A0A7B0"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def configure() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    path.chmod(0o644)


def plot_local_and_endpoint(
    observations: list[dict[str, Any]], result: dict[str, Any], output: Path
) -> None:
    layer_rows = [layer for row in observations for layer in row["layers"]]
    case_rows = result["case_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))

    axes[0].scatter(
        [row["attention_times_drift_mean"] for row in layer_rows],
        [row["actual_kv_output_relative_mean"] for row in layer_rows],
        color=BLUE,
        alpha=0.62,
        edgecolor="none",
        s=34,
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("attention mass × K/V cosine drift")
    axes[0].set_ylabel("local attention-output change (relative)")
    axes[0].set_title("Local mechanism: strong association")
    axes[0].text(
        0.04,
        0.94,
        "Spearman = 0.833\nn = 130 case-layer points",
        transform=axes[0].transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )

    axes[1].scatter(
        [row["local_kv_output_relative_mean"] for row in case_rows],
        [row["kv_js"] for row in case_rows],
        color=RED,
        alpha=0.72,
        edgecolor="none",
        s=46,
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("mean local attention-output change")
    axes[1].set_ylabel("final-logit JS after physical K/V splice")
    axes[1].set_title("End-to-end propagation: weak association")
    axes[1].text(
        0.04,
        0.94,
        "Spearman = 0.220\nn = 26 physical reuse cases",
        transform=axes[1].transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    fig.suptitle(
        "Attention-weighted K/V drift explains a local perturbation, not final quality",
        fontsize=16,
        y=1.02,
    )
    save(fig, output / "01_local_mechanism_vs_endpoint.png")


def plot_layer_correlations(result: dict[str, Any], output: Path) -> None:
    rows = result["local_correlations_by_layer"]
    layers = [0, 8, 17, 26, 35]
    raw = [rows[str(layer)]["raw_kv_drift"] for layer in layers]
    weighted = [rows[str(layer)]["attention_times_drift"] for layer in layers]
    first_order = [rows[str(layer)]["first_order_score"] for layer in layers]
    fig, axis = plt.subplots(figsize=(10.5, 5.5))
    axis.plot(layers, raw, "o-", color=GRAY, linewidth=2.2, label="raw K/V drift")
    axis.plot(
        layers,
        weighted,
        "o-",
        color=BLUE,
        linewidth=2.5,
        label="attention × drift",
    )
    axis.plot(
        layers,
        first_order,
        "o-",
        color=GREEN,
        linewidth=2.2,
        label="K/V first-order score",
    )
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_ylim(-0.3, 1.0)
    axis.set_xticks(layers, ["1", "9", "18", "27", "36"])
    axis.set_xlabel("Transformer layer (one-based)")
    axis.set_ylabel("Spearman with local attention-output change")
    axis.set_title("Attention weighting adds information at every probed layer")
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.2)
    save(fig, output / "02_layerwise_local_correlations.png")


def plot_components(observations: list[dict[str, Any]], output: Path) -> None:
    modes = ["key_only", "value_only", "kv"]
    values = [
        [row["physical_splice"][mode]["final_logit_js"] for row in observations]
        for mode in modes
    ]
    fig, axis = plt.subplots(figsize=(8.8, 5.5))
    boxes = axis.boxplot(
        values,
        tick_labels=["K only", "V only", "K + V"],
        patch_artist=True,
        showfliers=True,
    )
    for patch, color in zip(boxes["boxes"], [BLUE, ORANGE, RED], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    axis.set_yscale("log")
    axis.set_ylabel("final-logit JS after physical splice (log scale)")
    axis.set_title("Both K and V can carry contextual loss")
    medians = [float(np.median(row)) for row in values]
    for index, median in enumerate(medians, 1):
        axis.text(index, median * 1.35, f"median {median:.2e}", ha="center")
    axis.text(
        0.03,
        0.94,
        "V-only > K-only in 16/26 cases; K-only > V-only in 10/26",
        transform=axis.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )
    axis.grid(axis="y", alpha=0.2, which="both")
    save(fig, output / "03_key_value_component_js.png")


def plot_global_attention(global_result: dict[str, Any], output: Path) -> None:
    aggregate = global_result["aggregate"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
    tv = [
        aggregate["generation_tv"]["median"],
        aggregate["suffix_tv"]["median"],
    ]
    bars = axes[0].bar(
        ["Generation query", "Dense suffix queries"],
        tv,
        color=[BLUE, GREEN],
        width=0.6,
    )
    axes[0].bar_label(bars, labels=[f"{value:.4f}" for value in tv], padding=3)
    axes[0].set_ylim(0, max(tv) * 1.35)
    axes[0].set_ylabel("median block-attention total variation")
    axes[0].set_title("Global routing changes are small")

    agreements = [
        aggregate["generation_top_block_agreement_fraction"],
        aggregate["suffix_top_block_agreement_fraction"],
    ]
    bars = axes[1].bar(
        ["Generation query", "Dense suffix queries"],
        [100 * value for value in agreements],
        color=[BLUE, GREEN],
        width=0.6,
    )
    axes[1].bar_label(
        bars, labels=[f"{100 * value:.2f}%" for value in agreements], padding=3
    )
    axes[1].set_ylim(90, 100)
    axes[1].set_ylabel("same highest-attended structural block")
    axes[1].set_title("The dominant block is almost always preserved")
    fig.suptitle("Runtime-faithful reuse on 26 current-method islands", fontsize=16)
    save(fig, output / "04_global_block_attention_preservation.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound", type=Path, default=DEFAULT_BOUND)
    parser.add_argument("--global-attention", type=Path, default=DEFAULT_GLOBAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure()
    observations = load_jsonl(args.bound / "OBSERVATIONS.jsonl")
    result = load_json(args.bound / "RESULT.json")
    global_result = load_json(args.global_attention / "RESULT.json")
    if len(observations) != 26 or result["cases"] != 26:
        raise ValueError("expected the frozen 26-case perturbation audit")
    plot_local_and_endpoint(observations, result, args.output)
    plot_layer_correlations(result, args.output)
    plot_components(observations, args.output)
    plot_global_attention(global_result, args.output)
    manifest = {
        "bound_result": str(args.bound / "RESULT.json"),
        "global_attention_result": str(args.global_attention / "RESULT.json"),
        "figures": sorted(path.name for path in args.output.glob("*.png")),
    }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
