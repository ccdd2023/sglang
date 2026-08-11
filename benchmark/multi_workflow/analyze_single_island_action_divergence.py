#!/usr/bin/env python3
"""Build an audit figure for the frozen 64-token action-divergence study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/gfy/CodeMAS_Project")
DEFAULT_STUDY = (
    ROOT
    / "kvflow-artifacts/impactkv_single_island_action_divergence_20260807/"
    "frozen19"
)
DEFAULT_OUTPUT = (
    ROOT
    / "sglang-kvflow-worktrees/coding-aware/docs/kvflow/assets/"
    "module_conditioned_attention_kv_20260807"
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(result: dict[str, Any]) -> dict[str, Any]:
    names = ("fixed_probe_min", "module_attention_oracle", "seeded_random")
    paired = {}
    for name in names:
        value = result["arms"][name]["vs_recency"]
        paired[name] = {
            "wins": float(value["win_fraction"]),
            "ties": float(value["tie_fraction"]),
            "losses": 1.0
            - float(value["win_fraction"])
            - float(value["tie_fraction"]),
        }
    return {
        "status": "COMPLETE",
        "decision": result["decision"],
        "action_divergence": {
            "splices": result["unique_selected_splices"],
            "fraction": result["candidate_divergence_fraction"],
        },
        "within_case_variation": result["within_case_candidate_variation"],
        "arm_exact_dense_match_fraction": {
            name: float(value["exact_dense_match_fraction"])
            for name, value in result["arms"].items()
        },
        "paired_vs_recency": paired,
        "signal_to_action_distance_spearman": result[
            "signal_to_action_distance_spearman"
        ],
        "interpretation": (
            "The 64-token target is more resolved than immediate top-1, but "
            "candidate choice changes the continuation in only 7/19 cases and "
            "neither frozen selector wins paired comparisons. Exact token "
            "divergence also includes semantically similar paraphrases."
        ),
    }


def build_figure(result: dict[str, Any], output: Path) -> str:
    output.mkdir(parents=True, exist_ok=True)
    arm_names = (
        "current_recency",
        "fixed_probe_min",
        "module_attention_oracle",
        "seeded_random",
    )
    labels = ("Recency", "Fixed probe", "Module oracle", "Seeded random")
    colors = ("#9e9ac8", "#3182bd", "#31a354", "#bdbdbd")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.4))

    exact = [
        float(result["arms"][name]["exact_dense_match_fraction"])
        for name in arm_names
    ]
    axes[0].bar(labels, exact, color=colors)
    axes[0].set_ylim(0, 0.75)
    axes[0].set_ylabel("Exact Dense match fraction")
    axes[0].set_title("Aggregate match rate\n(random ties probe/oracle)")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", alpha=0.25)
    for index, value in enumerate(exact):
        axes[0].text(index, value + 0.025, f"{value:.3f}", ha="center")

    comparison_names = ("fixed_probe_min", "module_attention_oracle", "seeded_random")
    comparison_labels = ("Fixed probe", "Module oracle", "Seeded random")
    wins = np.asarray(
        [result["arms"][name]["vs_recency"]["win_fraction"] for name in comparison_names]
    )
    ties = np.asarray(
        [result["arms"][name]["vs_recency"]["tie_fraction"] for name in comparison_names]
    )
    losses = 1.0 - wins - ties
    axes[1].bar(comparison_labels, wins, color="#31a354", label="Win")
    axes[1].bar(comparison_labels, ties, bottom=wins, color="#bdbdbd", label="Tie")
    axes[1].bar(
        comparison_labels,
        losses,
        bottom=wins + ties,
        color="#de2d26",
        label="Loss",
    )
    axes[1].axhline(0.60, color="#222222", linestyle="--", linewidth=1.0)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Fraction of disagreement cases")
    axes[1].set_title("Paired outcomes vs recency\n(60% frozen win gate)")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].legend(frameon=False, ncol=3, fontsize=8, loc="upper center")

    funnel_labels = (
        "Immediate top-1\nchanges",
        "64-token splice\ndivergence",
        "Within-case candidate\nvariation",
    )
    funnel = (1 / 82, result["candidate_divergence_fraction"], 7 / 19)
    axes[2].bar(funnel_labels, funnel, color=("#fdae6b", "#3182bd", "#756bb1"))
    axes[2].set_ylim(0, 0.62)
    axes[2].set_ylabel("Observed fraction")
    axes[2].set_title("Longer action labels add resolution,\nbut not enough selector evidence")
    axes[2].tick_params(axis="x", rotation=12)
    axes[2].grid(axis="y", alpha=0.25)
    annotations = ("1/82", "18/36", "7/19")
    for index, (value, text) in enumerate(zip(funnel, annotations, strict=True)):
        axes[2].text(index, value + 0.025, text, ha="center")

    fig.tight_layout()
    destination = output / "08_action_divergence_resolution.png"
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)
    destination.chmod(0o644)
    return destination.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = _read(args.study / "RESULT.json")
    value = audit(result)
    value["figure"] = build_figure(result, args.output)
    destination = args.study / "POSTHOC_ACTION_AUDIT.json"
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    destination.chmod(0o644)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
