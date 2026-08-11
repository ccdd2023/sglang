#!/usr/bin/env python3
"""Build report figures for the natural-module Attention experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/gfy/CodeMAS_Project")
RESULT = (
    ROOT
    / "kvflow-artifacts/impactkv_natural_module_attention_20260808/"
    "attention_initial20_r1/RESULT.json"
)
PHYSICAL_RESULT = RESULT.parent / "physical_splice_minimal_reliable/RESULT.json"
STAGE_RESULT = PHYSICAL_RESULT.parent / "stage_overhead_code_only_r2/RESULT.json"
AGENT_RESULT = (
    ROOT
    / "kvflow-artifacts/impactkv_natural_code_cost_agent_20260808/RESULT.json"
)
EXACT_SPEED_RESULT = AGENT_RESULT.parent / "exact_prompt_speed/RESULT.json"
DEFAULT_OUTPUT = (
    ROOT
    / "sglang-kvflow-worktrees/coding-aware/docs/kvflow/assets/"
    "natural_module_attention_20260808"
)
LABELS = {
    "repository_code": "Repository code",
    "assistant_interpretation": "Assistant interpretation",
}
COLORS = {
    "raw": "#4C78A8",
    "adjusted": "#F58518",
    "baseline": "#9D9DA1",
    "enhanced": "#54A24B",
    "relation": "#B279A2",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: Any, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def boundaries(result: dict[str, Any], output: Path) -> None:
    order = ("repository_code", "assistant_interpretation")
    raw = [
        result["type_results"][key]["raw_natural_to_boundary_median_density_ratio"]
        for key in order
    ]
    adjusted = [
        result["type_results"][key][
            "median_baseline_adjusted_natural_to_boundary_density_ratio"
        ]
        for key in order
    ]
    ci = [
        result["type_results"][key][
            "task_bootstrap_adjusted_ratio_q025_q50_q975"
        ]
        for key in order
    ]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9.4, 5.5))
    width = 0.34
    ax.bar(x - width / 2, raw, width, label="Raw matched ratio", color=COLORS["raw"])
    errors = np.asarray(
        [[value - bounds[0] for value, bounds in zip(adjusted, ci)],
         [bounds[2] - value for value, bounds in zip(adjusted, ci)]]
    )
    ax.bar(
        x + width / 2,
        adjusted,
        width,
        yerr=errors,
        capsize=5,
        label="Length/position/distance adjusted",
        color=COLORS["adjusted"],
    )
    ax.axhline(1.0, color="#555", linewidth=1, linestyle=":", label="No advantage")
    ax.axhline(1.20, color="#C44E52", linewidth=1.5, linestyle="--", label="Frozen gate = 1.20")
    for offset, values in ((-width / 2, raw), (width / 2, adjusted)):
        for position, value in zip(x + offset, values):
            ax.text(position, value + 0.012, f"{value:.3f}×", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x, [LABELS[key] for key in order])
    ax.set_ylabel("Natural-module / boundary attention density")
    ax.set_ylim(0.96, 1.25)
    ax.set_title("Natural boundaries are real, but incremental cohesion misses the frozen gate")
    ax.legend(loc="upper left", ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, output, "01_natural_boundary_attention.png")


def prediction(result: dict[str, Any], output: Path) -> None:
    values = result["prediction"]
    baseline = float(values["baseline_task_leave_one_out_spearman"])
    enhanced = float(values["enhanced_task_leave_one_out_spearman"])
    improvement = float(values["improvement"])
    ci = values["task_bootstrap_improvement_q025_q50_q975"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.3), gridspec_kw={"width_ratios": [1.4, 1]})
    axes[0].bar(
        [0, 1],
        [baseline, enhanced],
        color=[COLORS["baseline"], COLORS["enhanced"]],
        width=0.62,
    )
    axes[0].set_xticks([0, 1], ["Length/position\nbaseline", "+ module/relation\nfeatures"])
    axes[0].set_ylim(0.82, 0.98)
    axes[0].set_ylabel("Task-LOO Spearman")
    for index, value in enumerate((baseline, enhanced)):
        axes[0].text(index, value + 0.006, f"{value:.3f}", ha="center", fontsize=12)
    axes[0].set_title("Prediction improves")
    axes[0].grid(axis="y", alpha=0.2)

    error = np.asarray([[improvement - ci[0]], [ci[2] - improvement]])
    axes[1].bar([0], [improvement], color=COLORS["enhanced"], width=0.55, yerr=error, capsize=6)
    axes[1].axhline(0, color="#555", linewidth=1)
    axes[1].axhline(0.10, color="#C44E52", linestyle="--", linewidth=1.5, label="Frozen gate = +0.10")
    axes[1].set_xticks([0], ["Observed gain"])
    axes[1].set_ylim(0, 0.12)
    axes[1].set_ylabel("Spearman improvement")
    axes[1].text(0, improvement + 0.008, f"+{improvement:.3f}\n95% task CI [{ci[0]:.3f}, {ci[2]:.3f}]", ha="center", fontsize=11)
    axes[1].set_title("Gain is positive, but too small")
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Coding structure adds signal beyond geometry, but not the preregistered amount", fontsize=14)
    fig.tight_layout()
    _save(fig, output, "02_crossfit_prediction_gain.png")


def consumers(result: dict[str, Any], output: Path) -> None:
    order = ("repository_code", "assistant_interpretation")
    rows = [
        result["type_results"][key]["source_to_consumer_vs_recency_control"]
        for key in order
    ]
    values = [row["median_density_ratio"] for row in rows]
    ci = [row["task_bootstrap_ratio_q025_q50_q975"] for row in rows]
    errors = np.asarray(
        [[value - bounds[0] for value, bounds in zip(values, ci)],
         [bounds[2] - value for value, bounds in zip(values, ci)]]
    )
    fig, ax = plt.subplots(figsize=(9.4, 5.5))
    bars = ax.bar(
        np.arange(2), values, yerr=errors, capsize=6, color=COLORS["relation"], width=0.58
    )
    ax.axhline(1, color="#555", linestyle=":", linewidth=1)
    ax.set_xticks(np.arange(2), [LABELS[key] for key in order])
    ax.set_ylabel("Source→consumer / matched recency density")
    ax.set_title("Path/symbol-linked consumers attend strongly to their actual source")
    for bar, value, row in zip(bars, values, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            f"{value:.2f}×\n{100 * row['paired_direction']:.1f}% pairs",
            ha="center",
            fontsize=11,
        )
    ax.text(
        0.01,
        0.98,
        "Descriptive follow-up; not a physical-safety gate",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#8C2D2D",
        fontsize=11,
    )
    ax.set_ylim(0, max(bounds[2] for bounds in ci) * 1.25)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, output, "03_source_consumer_attention.png")


def physical_and_stage(
    physical_result: dict[str, Any],
    stage_result: dict[str, Any],
    output: Path,
) -> None:
    order = ("repository_code", "assistant_interpretation")
    physical_rows = [physical_result["module_results"][key] for key in order]
    ratios = [row["local_output_natural_boundary_median_ratio"] for row in physical_rows]
    cis = [row["local_output_natural_boundary_task_bootstrap_q025_q50_q975"] for row in physical_rows]
    errors = np.asarray(
        [
            [value - bounds[0] for value, bounds in zip(ratios, cis)],
            [bounds[2] - value for value, bounds in zip(ratios, cis)],
        ]
    )
    buckets = stage_result["latency"]["posthoc_by_island_length"]
    bucket_order = ("lt_128", "128_255", "256_511", "ge_512")
    bucket_labels = ("<128", "128–255", "256–511", "≥512")
    savings = [buckets[key]["mean_ttft_saving_percent"] for key in bucket_order]
    counts = [buckets[key]["cases"] for key in bucket_order]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), constrained_layout=True)
    axes[0].bar(
        np.arange(2),
        ratios,
        yerr=errors,
        capsize=6,
        color=[COLORS["raw"], COLORS["adjusted"]],
        width=0.62,
    )
    axes[0].axhline(1, color="#555", linestyle=":", linewidth=1.2)
    axes[0].set_xticks(np.arange(2), [LABELS[key] for key in order])
    axes[0].set_ylabel("Natural / cross-boundary local perturbation")
    axes[0].set_title("Only repository code has a directional safety gain")
    for index, (value, row) in enumerate(zip(ratios, physical_rows)):
        axes[0].text(
            index,
            max(value, cis[index][2]) + 0.04,
            f"{value:.3f}×\n{100 * row['local_output_natural_wins']:.1f}% wins",
            ha="center",
            fontsize=11,
        )
    axes[0].set_ylim(0.6, 1.42)
    axes[0].grid(axis="y", alpha=0.2)

    colors = ["#C44E52" if value < 0 else COLORS["enhanced"] for value in savings]
    bars = axes[1].bar(np.arange(4), savings, color=colors, width=0.68)
    axes[1].axhline(0, color="#555", linewidth=1)
    axes[1].set_xticks(np.arange(4), bucket_labels)
    axes[1].set_xlabel("Natural code-island tokens")
    axes[1].set_ylabel("Mean paired TTFT saving (%)")
    axes[1].set_title("SGLang benefit appears only for long code modules")
    for bar, value, count in zip(bars, savings, counts):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value / 2,
            f"{value:+.1f}%\nn={count}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10,
        )
    axes[1].set_ylim(-7.2, 13.8)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Physical K/V splice narrows the policy to code; runtime narrows it to long code",
        fontsize=14,
    )
    _save(fig, output, "04_physical_splice_and_stage_ttft_r1.png")


def fresh_agent_accuracy_and_speed(
    agent_result: dict[str, Any],
    exact_speed_result: dict[str, Any],
    output: Path,
) -> None:
    accuracy = agent_result["accuracy"]
    speed = exact_speed_result["latency"]
    denominator = int(accuracy["denominator"])
    resolved = [
        int(accuracy["dense_resolved"]),
        int(accuracy["policy_resolved"]),
    ]
    speedups = [
        float(speed["cache_ready_speedup_ratio_of_means"]),
        float(speed["n4_including_one_source_build_speedup"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), constrained_layout=True)
    bars = axes[0].bar(
        np.arange(2),
        resolved,
        color=[COLORS["baseline"], COLORS["enhanced"]],
        width=0.62,
    )
    axes[0].set_xticks(
        np.arange(2), ["Dense", "Natural code\n+ cost gate"]
    )
    axes[0].set_ylabel("Official SWE-bench tasks resolved")
    axes[0].set_ylim(0, denominator)
    axes[0].set_title("Fresh tasks: directional accuracy gain")
    for bar, value in zip(bars, resolved):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            f"{value}/{denominator}",
            ha="center",
            fontsize=13,
            fontweight="bold",
        )
    axes[0].text(
        0.02,
        0.96,
        "2 rescues, 1 damage; n=9 is exploratory",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="#555",
    )
    axes[0].grid(axis="y", alpha=0.2)

    colors = [COLORS["enhanced"], "#C44E52"]
    bars = axes[1].bar(np.arange(2), speedups, color=colors, width=0.62)
    axes[1].axhline(1.0, color="#555", linestyle=":", linewidth=1.3)
    axes[1].set_xticks(
        np.arange(2),
        ["Cache-ready\nexact prompts", "N=4 + repeated\nsource replay"],
    )
    axes[1].set_ylabel("Speedup over exact-prompt Dense")
    axes[1].set_ylim(0.8, 1.45)
    axes[1].set_title("Target compute wins; pessimistic rebuild does not")
    for bar, value in zip(bars, speedups):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.035,
            f"{value:.3f}×",
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="white",
        )
    axes[1].text(
        0.02,
        0.96,
        "56 exact target prompts, 168 measured pairs\n316/316 copies; zero fallback",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="#555",
    )
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Fresh online accuracy and exact-token latency answer different questions",
        fontsize=15,
    )
    _save(fig, output, "05_fresh_agent_accuracy_and_exact_speed.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--physical-result", type=Path, default=PHYSICAL_RESULT)
    parser.add_argument("--stage-result", type=Path, default=STAGE_RESULT)
    parser.add_argument("--agent-result", type=Path, default=AGENT_RESULT)
    parser.add_argument(
        "--exact-speed-result", type=Path, default=EXACT_SPEED_RESULT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = _load(args.result)
    boundaries(value, args.output)
    prediction(value, args.output)
    consumers(value, args.output)
    physical_and_stage(
        _load(args.physical_result), _load(args.stage_result), args.output
    )
    fresh_agent_accuracy_and_speed(
        _load(args.agent_result), _load(args.exact_speed_result), args.output
    )
    print(json.dumps({"output": str(args.output), "figures": 5}, indent=2))


if __name__ == "__main__":
    main()
