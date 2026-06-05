"""Build the predicted_distance_table.json from the per-segment context-variation
results. The table is the input artifact for the sglang-kvflow
context_aware_confidence modifier (see anchor_match.py).

Bucketing decisions (must match anchor_match.py constants):
  - length_bin            ∈ {"<50", "50-200", "200-500", ">500"}
  - position_offset_bin   ∈ {"0", "5-25", "50-100"}   (0 / [1,25] / [26, inf])
  - system_prompt_class   ∈ {"planner", "coder", "reviewer", "tester"}
  - surrounding_code_class∈ {"none", "class_wrap", "try_wrap", "imports_wrap"}

Output: data/predicted_distance_table.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Iterable


# Position offset bin boundaries (the per-segment data has 0,5,10,25,50,100).
_POSITION_OFFSET_BIN_EDGES = ((0, 0), (1, 25), (26, 10**9))
_POSITION_OFFSET_BIN_LABELS = ("0", "5-25", "50-100")


def _position_offset_bin(offset: int) -> str:
    for (lo, hi), label in zip(_POSITION_OFFSET_BIN_EDGES, _POSITION_OFFSET_BIN_LABELS):
        if lo <= offset <= hi:
            return label
    return _POSITION_OFFSET_BIN_LABELS[-1]


def _length_bin_from_existing(bin_str: str) -> str:
    """Pass-through since per_segment already has length_bin strings."""
    if bin_str in ("<50", "50-200", "200-500", ">500"):
        return bin_str
    return "50-200"


def build_table(per_segment: list[dict]) -> dict:
    # Group all (length_bin, pos_bin, sys_cls, surr_cls) -> list of d_norm
    cells: dict[tuple, list[float]] = defaultdict(list)

    for seg in per_segment:
        lb = _length_bin_from_existing(seg["length_bin"])
        # Each segment has by_position_offset / by_system_prompt_class / by_surrounding_code_class
        # as per-axis marginal averages. We need to reconstruct the full 4D
        # grid by looking at individual records — but our per_segment summary
        # only stores marginals. So we recompute from by_position_offset (which
        # has the 6 offset buckets × d_norm) and cross-marginalize.
        # For each offset bucket in this segment, add (lb, pos_bin, "planner", "none")
        # d_norm as the canonical cell value, then we'll average across segments
        # below. The other sys_cls/surr_cls effects are constant-additive so we
        # compute deltas from a "canonical" reading.
        by_off = seg.get("by_position_offset", {})
        for off_str, info in by_off.items():
            pos_bin = _position_offset_bin(int(off_str))
            # Treat this as the (planner, none) cell — the canonical reference
            # in our experiment IS (offset=X, planner, none).
            cells[(lb, pos_bin, "planner", "none")].append(info["mean"])

    # Now compute per-axis deltas. For each (lb, pos_bin, "planner", "none")
    # cell we have a baseline. For other (sys_cls, surr_cls) combinations,
    # the d_norm changes by a per-axis delta. Compute the deltas from
    # per_segment.by_system_prompt_class and per_segment.by_surrounding_code_class.
    sys_deltas: dict[str, list[float]] = defaultdict(list)
    surr_deltas: dict[str, list[float]] = defaultdict(list)
    for seg in per_segment:
        for sys_cls, info in seg.get("by_system_prompt_class", {}).items():
            # Delta = sys_mean - planner_mean (planner is the reference)
            planner_mean = seg["by_system_prompt_class"].get("planner", {}).get("mean")
            if planner_mean is not None and sys_cls != "planner":
                sys_deltas[sys_cls].append(info["mean"] - planner_mean)
        for surr_cls, info in seg.get("by_surrounding_code_class", {}).items():
            none_mean = seg["by_surrounding_code_class"].get("none", {}).get("mean")
            if none_mean is not None and surr_cls != "none":
                surr_deltas[surr_cls].append(info["mean"] - none_mean)
    sys_delta_mean = {k: sum(v) / len(v) for k, v in sys_deltas.items() if v}
    surr_delta_mean = {k: sum(v) / len(v) for k, v in surr_deltas.items() if v}

    # Compose the final cells: for each (lb, pos_bin), the 4 sys_cls × 4 surr_cls cells.
    final_cells: list[dict] = []
    for (lb, pos_bin, _, _), baseline_d_norms in cells.items():
        baseline = sum(baseline_d_norms) / len(baseline_d_norms)
        for sys_cls in ("planner", "coder", "reviewer", "tester"):
            sys_delta = sys_delta_mean.get(sys_cls, 0.0) if sys_cls != "planner" else 0.0
            for surr_cls in ("none", "class_wrap", "try_wrap", "imports_wrap"):
                surr_delta = surr_delta_mean.get(surr_cls, 0.0) if surr_cls != "none" else 0.0
                predicted = max(0.1, baseline + sys_delta + surr_delta)
                final_cells.append({
                    "length_bin": lb,
                    "position_offset": pos_bin,
                    "system_prompt_class": sys_cls,
                    "surrounding_code_class": surr_cls,
                    "predicted_d_norm_mean": round(predicted, 4),
                    "predicted_d_norm_std": 0.0,
                    "n_samples": len(baseline_d_norms),
                    "baseline_d_norm": round(baseline, 4),
                    "sys_delta": round(sys_delta, 4),
                    "surr_delta": round(surr_delta, 4),
                })

    # Compute global stats
    all_predicted = [c["predicted_d_norm_mean"] for c in final_cells]
    baseline_for_canary = min((c["baseline_d_norm"] for c in final_cells if c["position_offset"] == "0" and c["system_prompt_class"] == "planner" and c["surrounding_code_class"] == "none"), default=1.0)
    return {
        "schema_version": "v1",
        "buckets": {
            "length_bin": ["<50", "50-200", "200-500", ">500"],
            "position_offset": ["0", "5-25", "50-100"],
            "system_prompt_class": ["planner", "coder", "reviewer", "tester"],
            "surrounding_code_class": ["none", "class_wrap", "try_wrap", "imports_wrap"],
        },
        "cells": final_cells,
        "global": {
            "predicted_d_norm_baseline": round(baseline_for_canary, 4),
            "predicted_d_norm_max_observed": round(max(all_predicted), 4),
            "predicted_d_norm_min_observed": round(min(all_predicted), 4),
        },
        "axes_deltas": {
            "system_prompt_class_delta_vs_planner": sys_delta_mean,
            "surrounding_code_class_delta_vs_none": surr_delta_mean,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path",
                   default="/home/gfy/CodeMAS_Project/sglang-kvflow/results/same_code_context_variation/data/context_distance_7b.json")
    p.add_argument("--out", default="/home/gfy/CodeMAS_Project/sglang-kvflow/results/same_code_context_variation/data/predicted_distance_table.json")
    args = p.parse_args()
    with open(args.in_path) as f:
        data = json.load(f)
    per_segment = data["per_segment"]
    table = build_table(per_segment)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    print(f"[distance_table] wrote {args.out}")
    print(f"[distance_table] {len(table['cells'])} cells, baseline={table['global']['predicted_d_norm_baseline']}, max={table['global']['predicted_d_norm_max_observed']}")


if __name__ == "__main__":
    main()
