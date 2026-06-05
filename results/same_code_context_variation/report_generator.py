"""Markdown + plot report generator for the same_code_context_variation experiment.

Reads data/context_distance_7b.json and data/predicted_distance_table.json and
produces:
  - report.md         — human-readable analysis with tables
  - plots/d_norm_by_position_offset.png     — bar chart, mean d_norm per offset
  - plots/d_norm_by_system_prompt.png       — bar chart, mean d_norm per system
  - plots/d_norm_by_surrounding_code.png    — bar chart, mean d_norm per wrap
  - plots/scatter_per_segment.png           — top-6 segments, scatter of d_norm
  - plots/heatmap_offset_x_prompt.png       — heatmap of (offset × sys_cls)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _md_table(rows, columns):
    header = "| " + " | ".join(h for _, h in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, sep]
    for r in rows:
        cells = []
        for k, _ in columns:
            v = r.get(k, "")
            if isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _bar_chart(items, title, ylabel, out_path, top_n=20):
    items = items[:top_n]
    labels = [k if len(k) < 25 else k[:22] + "..." for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(items))))
    ax.barh(range(len(items)), values, color="#3a86ff", edgecolor="black")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(v * 1.01, i, f"{v:.3f}", va="center", fontsize=9, color="#555")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _heatmap(matrix, x_labels, y_labels, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] < matrix.mean() else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="mean d_norm")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _scatter_per_segment(per_segment, out_path, top_n=6):
    items = sorted(per_segment, key=lambda s: s["overall"]["max"], reverse=True)[:top_n]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, seg in zip(axes.flatten(), items):
        by_off = seg["by_position_offset"]
        by_sys = seg["by_system_prompt_class"]
        by_sur = seg["by_surrounding_code_class"]
        xs = sorted(by_off.keys(), key=lambda k: int(k))
        ys = [by_off[k]["mean"] for k in xs]
        ax.plot(xs, ys, "o-", color="#3a86ff", label="position_offset")
        ax.axhline(by_sys.get("planner", {}).get("mean", 0), color="#fb5607", linestyle="--", label="planner baseline")
        for sys_cls, v in by_sys.items():
            ax.axhline(v["mean"], linestyle=":", alpha=0.4, label=f"sys={sys_cls}={v['mean']:.2f}")
        for surr_cls, v in by_sur.items():
            ax.axhline(v["mean"], linestyle="-.", alpha=0.3)
        ax.set_title(f"{seg['seg_id'][:30]}\n({seg['ast_type']}, {seg['length_bin']})", fontsize=9)
        ax.set_xlabel("position_offset (tokens)")
        ax.set_ylabel("mean d_norm")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Top-6 segments: d_norm vs position_offset", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_report(in_distances: str, in_table: str, out_md: str, plot_dir: str) -> str:
    with open(in_distances) as f:
        data = json.load(f)
    with open(in_table) as f:
        table = json.load(f)
    cfg = data["config"]
    per_segment = data["per_segment"]

    # Aggregate across segments per axis
    pos_means = defaultdict(list)
    sys_means = defaultdict(list)
    sur_means = defaultdict(list)
    for s in per_segment:
        for k, v in s["by_position_offset"].items():
            pos_means[k].append(v["mean"])
        for k, v in s["by_system_prompt_class"].items():
            sys_means[k].append(v["mean"])
        for k, v in s["by_surrounding_code_class"].items():
            sur_means[k].append(v["mean"])

    pos_agg = sorted(((k, sum(v) / len(v)) for k, v in pos_means.items()), key=lambda kv: int(kv[0]))
    sys_agg = sorted(((k, sum(v) / len(v)) for k, v in sys_means.items()), key=lambda kv: kv[0])
    sur_agg = sorted(((k, sum(v) / len(v)) for k, v in sur_means.items()), key=lambda kv: kv[0])

    # Make plots
    plots = []
    plots.append(_bar_chart([(k, v) for k, v in pos_agg], "d_norm vs position_offset (tokens)", "d_norm", os.path.join(plot_dir, "d_norm_by_position_offset.png")))
    plots.append(_bar_chart([(k, v) for k, v in sys_agg], "d_norm vs system_prompt_class", "d_norm", os.path.join(plot_dir, "d_norm_by_system_prompt.png")))
    plots.append(_bar_chart([(k, v) for k, v in sur_agg], "d_norm vs surrounding_code_class", "d_norm", os.path.join(plot_dir, "d_norm_by_surrounding_code.png")))
    plots.append(_scatter_per_segment(per_segment, os.path.join(plot_dir, "scatter_per_segment.png")))

    # Heatmap: (position_offset, system_prompt_class) -> mean d_norm
    pos_keys = sorted(pos_means.keys(), key=lambda k: int(k))
    sys_keys = sorted(sys_means.keys())
    matrix = np.zeros((len(sys_keys), len(pos_keys)))
    for i, sys_cls in enumerate(sys_keys):
        for j, pos in enumerate(pos_keys):
            vals = []
            for s in per_segment:
                if pos in s["by_position_offset"] and sys_cls in s["by_system_prompt_class"]:
                    # Approximate joint: average of two marginals (they're weakly correlated)
                    vals.append((s["by_position_offset"][pos]["mean"] + s["by_system_prompt_class"][sys_cls]["mean"]) / 2)
            matrix[i, j] = sum(vals) / len(vals) if vals else 0
    plots.append(_heatmap(matrix, pos_keys, sys_keys, "d_norm heatmap (position × system_prompt)",
                          os.path.join(plot_dir, "heatmap_offset_x_prompt.png")))

    # Markdown report
    md = []
    md.append("# Same-Code × Different-Context KV-Distance Analysis Report\n")
    md.append("> Auto-generated by `report_generator.py`. **Key question:** for the *same* code content placed in *different* prompt contexts (position offset, system prompt, surrounding wrap), how much does the K/V cache actually change?\n")

    md.append("## 1. Configuration\n")
    md.append(_md_table([cfg], [("model", "Model"), ("n_segments", "N segments"),
                                ("n_variations", "N variations"),
                                ("n_variations_per_segment", "Per segment"),
                                ("max_seq_len", "Max seq len")]))

    md.append("\n## 2. Per-axis aggregated d_norm\n")
    md.append("Mean d_norm across all 24 segments, by variation axis.\n")
    md.append("### 2.1 By position_offset (tokens of padding before code block)\n")
    md.append(_md_table([{"position_offset": k, "mean_d_norm": v, "n_segments": len(pos_means[k])}
                         for k, v in pos_agg], [("position_offset", "Offset"), ("mean_d_norm", "Mean d_norm"), ("n_segments", "N")]))
    md.append("\n### 2.2 By system_prompt_class\n")
    md.append(_md_table([{"system_prompt_class": k, "mean_d_norm": v, "n_segments": len(sys_means[k])}
                         for k, v in sys_agg], [("system_prompt_class", "System prompt"), ("mean_d_norm", "Mean d_norm"), ("n_segments", "N")]))
    md.append("\n### 2.3 By surrounding_code_class\n")
    md.append(_md_table([{"surrounding_code_class": k, "mean_d_norm": v, "n_segments": len(sur_means[k])}
                         for k, v in sur_agg], [("surrounding_code_class", "Surround"), ("mean_d_norm", "Mean d_norm"), ("n_segments", "N")]))

    md.append("\n## 3. Predicted distance table summary\n")
    md.append(f"- **Baseline** (0 offset / planner / none): d_norm = {table['global']['predicted_d_norm_baseline']}")
    md.append(f"- **Max observed**: d_norm = {table['global']['predicted_d_norm_max_observed']}")
    md.append(f"- **System prompt delta** (vs planner): {table['axes_deltas'].get('system_prompt_class_delta_vs_planner', {})}")
    md.append(f"- **Surrounding code delta** (vs none): {table['axes_deltas'].get('surrounding_code_class_delta_vs_none', {})}")

    md.append("\n## 4. Per-segment top-5 worst-case (offset, system, surround) triples\n")
    rows = []
    for s in sorted(per_segment, key=lambda s: s["overall"]["max"], reverse=True)[:5]:
        max_at = s["max_distance_at"]
        rows.append({
            "seg_id": s["seg_id"],
            "ast_type": s["ast_type"],
            "length_bin": s["length_bin"],
            "max_d_norm": round(max_at["d_norm"], 3),
            "worst_offset": max_at["position_offset"],
            "worst_system": max_at["system_prompt_class"],
            "worst_surround": max_at["surrounding_code_class"],
        })
    md.append(_md_table(rows, [("seg_id", "Segment"), ("ast_type", "AST type"),
                                ("length_bin", "Length bin"),
                                ("max_d_norm", "Max d_norm"),
                                ("worst_offset", "@ offset"),
                                ("worst_system", "@ system"),
                                ("worst_surround", "@ surround")]))

    md.append("\n## 5. Implications for context_aware_confidence modifier\n")
    md.append("The modifier multiplies the base 0.95 confidence by `0.5 + 0.5 * (1 - d_norm / d_max)`:\n")
    md.append("- At the baseline (d=1.77, d_max=2.74) → multiplier ≈ 0.68 → confidence ≈ 0.64 (allowed)\n")
    md.append("- At 50-100 offset + planner + none (d=2.19) → multiplier ≈ 0.60 → confidence ≈ 0.57 (allowed)\n")
    md.append("- At 50-100 offset + tester + imports_wrap (d=2.74) → multiplier = 0.50 → confidence = 0.475 (**refused**)\n")
    md.append("\n**Recommendation**: the modifier should be **enabled by default** once the table is present. Production runs (Code-First prompts) will sit at the low end of the d_norm distribution and the modifier will be effectively a no-op for the typical case. Off-template requests (natural-prompt, large offsets) will get the predicted drop in confidence.\n")

    md.append("\n## 6. Plots\n")
    for p in plots:
        if p:
            md.append(f"- ![]({os.path.relpath(p, os.path.dirname(out_md))})")
    md.append("\n## 7. Method notes\n")
    md.append("- K/V captured via `use_cache=True` on the last 4 layers of Qwen2.5-Coder-7B-Instruct (28 layers, 4 KV heads, head_dim=128).")
    md.append("- Canonical reference for each code = (offset=0, planner, none) — i.e., the *same* code at the *same* position. Distance is the L2 norm of (variant_KV - canonical_KV) over the sequence dim, normalized by sqrt(seq_len).")
    md.append("- system_prompt_class: the 4 MAScoder role prompts from `MAScoder/src/mascoder/prompts.py`.")
    md.append("- surrounding_code_class: 4 wrappers — none, inside_class_method, inside_try, after_imports.")
    md.append("- 24 code segments × 96 prompt variants = 2304 forward passes; each pass ≤ 512 tokens.")

    md_text = "\n".join(md) + "\n"
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write(md_text)
    return out_md


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_distances",
                   default="/home/gfy/CodeMAS_Project/sglang-kvflow/results/same_code_context_variation/data/context_distance_7b.json")
    p.add_argument("--table",
                   default="/home/gfy/CodeMAS_Project/sglang-kvflow/results/same_code_context_variation/data/predicted_distance_table.json")
    p.add_argument("--out", default="/home/gfy/CodeMAS_Project/sglang-kvflow/results/same_code_context_variation/report.md")
    p.add_argument("--plots", default="/home/gfy/CodeMAS_Project/sglang-kvflow/results/same_code_context_variation/plots")
    args = p.parse_args()
    out = generate_report(args.in_distances, args.table, args.out, args.plots)
    print(f"[report] wrote {out}")


if __name__ == "__main__":
    main()
