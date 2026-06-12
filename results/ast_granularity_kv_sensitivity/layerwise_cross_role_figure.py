#!/usr/bin/env python3
"""Per-layer cross-role K drift figure for the AST-granularity study.

Plots 6 lines (one per AST granularity) of mean planner-vs-coder K drift
(per-element RMS of K diff) across the 28 transformer layers of
Qwen2.5-Coder-7B-Instruct. The point: cross-role K drift is near-zero
in early layers, rises monotonically through mid-1 / mid-2, and
plateaus in the late layers; the last-4-layers cut is the smallest
layer set that preserves the late-bin signal.

Output: figures/fig_ast_granularity_layerwise.pdf (paper)
        figures/fig_layerwise_ast_cross_role.png (sglang-kvflow)
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results/ast_granularity_kv_sensitivity/data/layerwise_ast_granularity_comparison.csv"
PAPER_OUT = Path("/home/gfy/Paper_CodeMAS/CodeAgent_UCM_HKBU/figures/fig_ast_granularity_layerwise.pdf")
LOCAL_OUT = ROOT / "results/ast_granularity_kv_sensitivity/figures/fig_layerwise_ast_cross_role.png"

GRANULARITIES = ["function", "method", "class", "control_block", "statement_window", "file_prefix"]
COLORS = {
    "function": "#2c7fb8",
    "method": "#7fcdbb",
    "class": "#d7301f",
    "control_block": "#fd8d3c",
    "statement_window": "#feb24c",
    "file_prefix": "#8856a7",
}


def main() -> int:
    if not CSV.exists():
        print(f"missing {CSV}", flush=True)
        return 1
    series: dict[str, dict[int, list[float]]] = {g: defaultdict(list) for g in GRANULARITIES}
    with CSV.open() as f:
        for r in csv.DictReader(f):
            if r["variant"] != "correct_delta":
                continue
            g = r["granularity"]
            if g not in series:
                continue
            try:
                layer = int(r["layer"])
                l2 = float(r["k_l2_norm"])
            except (KeyError, ValueError):
                continue
            series[g][layer].append(l2)
    layers = sorted({L for g in series.values() for L in g})
    means = {g: [np.mean(series[g][L]) if series[g][L] else np.nan for L in layers] for g in GRANULARITIES}
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for g in GRANULARITIES:
        ax.plot(layers, means[g], label=g, color=COLORS[g], linewidth=1.6, marker="o", markersize=3.2)
    ax.axvspan(24, 27, color="grey", alpha=0.13, label="last-4 cut")
    ax.set_xlabel("Transformer layer (0-indexed, 28 layers total)")
    ax.set_ylabel("Mean $\|K_\\mathrm{planner} - K_\\mathrm{coder}\\|_2 / \\sqrt{|K|}$")
    ax.set_title("Cross-role K drift by AST granularity and layer\n(Qwen2.5-Coder-7B-Instruct, RoPE-aligned)")
    ax.set_xticks([0, 7, 14, 21, 27])
    ax.set_xticklabels(["0", "7", "14", "21", "27"])
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.85)
    fig.tight_layout()
    PAPER_OUT.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_OUT, dpi=300, bbox_inches="tight")
    fig.savefig(LOCAL_OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {PAPER_OUT} and {LOCAL_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
