#!/usr/bin/env python3
"""Per-layer cross-role K drift for the AST-granularity study.

The headline cross-role table (Table \\ref{tab:ast-granularity-cross-role})
uses the L2-based d_norm metric, computed on the concat of the last 4
layers. A natural reviewer question is: why last 4 layers, not all 28?

This script answers that question directly by aggregating the per-layer
planner-vs-coder K drift from
layerwise_ast_granularity_comparison.csv (sglang-kvflow fork, 2026-06-10)
across 5 layer bins:

  - early    [0,  6]   (7 layers)
  - mid-1    [7, 13]   (7 layers)
  - mid-2    [14,20]   (7 layers)
  - late     [21,27]   (7 layers; the "late layers" superset)
  - last-4   [24,27]   (4 layers; the headline cross-role cut)

We report two metrics per (granularity, layer_bin):
  - mean k_cosine   (1.0 = identical direction; 0.0 = orthogonal)
  - mean k_l2_norm  (per-element RMS of the K diff; higher = more drift)

The point: cross-role K direction is near-perfect at every layer, but the
*contrast* between roles is concentrated in the late layers. Early layers
collapse to cos ~ 1.000 regardless of granularity, so they cannot
differentiate planner K from coder K. The last-4-layers cut is the
smallest layer set that still preserves the cross-role signal.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results/ast_granularity_kv_sensitivity/data/layerwise_ast_granularity_comparison.csv"
OUT_JSON = ROOT / "results/ast_granularity_kv_sensitivity/data/layerwise_cross_role.json"
OUT_MD = ROOT / "results/ast_granularity_kv_sensitivity/layerwise_cross_role.md"

LAYER_BINS = [
    ("early [0-6]", 0, 6),
    ("mid-1 [7-13]", 7, 13),
    ("mid-2 [14-20]", 14, 20),
    ("late [21-27]", 21, 27),
    ("last-4 [24-27]", 24, 27),
]
GRANULARITIES = ["function", "method", "class", "control_block", "statement_window", "file_prefix"]


def main() -> int:
    if not CSV.exists():
        print(f"missing {CSV}", flush=True)
        return 1
    cos_by: dict[tuple[str, int], list[float]] = defaultdict(list)
    l2_by: dict[tuple[str, int], list[float]] = defaultdict(list)
    with CSV.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["variant"] != "correct_delta":
                continue
            try:
                layer = int(r["layer"])
                cos = float(r["k_cosine"])
                l2 = float(r["k_l2_norm"])
            except (KeyError, ValueError):
                continue
            cos_by[(r["granularity"], layer)].append(cos)
            l2_by[(r["granularity"], layer)].append(l2)
    summary: dict[str, dict[str, dict]] = {}
    for g in GRANULARITIES:
        summary[g] = {}
        for label, lo, hi in LAYER_BINS:
            cos_vals: list[float] = []
            l2_vals: list[float] = []
            for layer in range(lo, hi + 1):
                cos_vals.extend(cos_by.get((g, layer), []))
                l2_vals.extend(l2_by.get((g, layer), []))
            if not cos_vals or not l2_vals:
                continue
            summary[g][label] = {
                "n": len(cos_vals),
                "mean_k_cosine": round(statistics.fmean(cos_vals), 6),
                "min_k_cosine": round(min(cos_vals), 6),
                "mean_k_l2_norm": round(statistics.fmean(l2_vals), 6),
                "max_k_l2_norm": round(max(l2_vals), 6),
            }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Per-layer cross-role K drift (planner-vs-coder, RoPE-aligned)\n"]
    lines.append("Source: layerwise_ast_granularity_comparison.csv, correct_delta variant.\n")
    lines.append("Qwen2.5-Coder-7B-Instruct, 28 layers aggregated into 5 layer bins.\n")
    lines.append("Two metrics: mean k_cosine (1.0 = identical direction) and mean k_l2_norm (per-element RMS of K diff).\n")
    lines.append("Hypothesis: cross-role K direction is near-perfect at every layer, but the *contrast* between roles is concentrated in the late layers; the last-4-layers cut is the smallest layer set that preserves the cross-role signal.\n")
    lines.append("\n| Granularity | bin | n | mean k_cos | min k_cos | mean k_l2 | max k_l2 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for g in GRANULARITIES:
        for label, _, _ in LAYER_BINS:
            s = summary[g].get(label)
            if not s:
                continue
            lines.append(
                f"| {g} | {label} | {s['n']} | {s['mean_k_cosine']:.4f} | {s['min_k_cosine']:.4f} | "
                f"{s['mean_k_l2_norm']:.4f} | {s['max_k_l2_norm']:.4f} |"
            )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
