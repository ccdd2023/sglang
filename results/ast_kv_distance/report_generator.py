"""Markdown + plot report generator for the AST × KV-distance experiment.

Reads data/distance_7b.json and produces:
  - report.md         — human-readable summary with tables and key findings
  - plots/d_norm_by_ast_type.png    — bar chart of normalised d by AST type
  - plots/d_norm_by_template.png    — bar chart of normalised d by template
  - plots/d_norm_by_length.png      — bar chart of normalised d by length bin
  - plots/entropy_by_dimension.png  — KVCOMM entropy across the three dimensions
  - plots/within_vs_cross_ast.png   — within vs cross AST-type distance
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _sort_by_key(d: dict, key: str = "d_norm_avg", reverse: bool = False) -> list[tuple[str, dict]]:
    items = [(k, v) for k, v in d.items() if isinstance(v, dict) and key in v]
    items.sort(key=lambda kv: kv[1][key], reverse=reverse)
    return items


def _plot_bar(items: list[tuple[str, dict]], metric: str, title: str, ylabel: str, out_path: str, top_n: int = 12):
    items = items[:top_n]
    labels = [k if len(k) < 30 else k[:27] + "..." for k, _ in items]
    values = [v[metric] for _, v in items]
    counts = [v.get("count", 0) for _, v in items]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(items))))
    bars = ax.barh(range(len(items)), values, color="#3a86ff", edgecolor="black")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    for i, (b, c) in enumerate(zip(bars, counts)):
        ax.text(b.get_width() * 1.01, b.get_y() + b.get_height() / 2,
                f" n={c}", va="center", fontsize=9, color="#555")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _plot_within_vs_cross(stats: dict, out_path: str):
    w = stats.get("within_ast_type", {})
    c = stats.get("cross_ast_type", {})
    if not w or not c:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = ["d_norm_avg", "d_norm_p50"]
    wv = [w.get(m, 0) for m in metrics]
    cv = [c.get(m, 0) for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, wv, width, label="within-AST-type", color="#3a86ff")
    ax.bar(x + width / 2, cv, width, label="cross-AST-type", color="#fb5607")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("L2 distance (normalised by sqrt(seq_len))")
    ax.set_title("Within vs cross AST-type KV distance")
    ax.legend()
    for i, (a, b) in enumerate(zip(wv, cv)):
        ax.text(i - width / 2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _plot_entropy_bars(pool: dict, out_path: str):
    """Three side-by-side bars: entropy_avg by AST type, by template, by length bin."""
    dims = [("by_ast_type", "AST type"), ("by_template", "Template"), ("by_length_bin", "Length bin")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, title) in zip(axes, dims):
        d = pool.get(key, {})
        items = [(k, v) for k, v in d.items() if isinstance(v, dict) and "entropy_avg" in v]
        items.sort(key=lambda kv: kv[1]["entropy_avg"])
        if not items:
            ax.set_title(f"{title} (no data)")
            continue
        labels = [k if len(k) < 18 else k[:15] + "..." for k, _ in items]
        ents = [v["entropy_avg"] for _, v in items]
        rates = [v["gate_pass_rate"] for _, v in items]
        bars = ax.bar(range(len(items)), ents, color="#8338ec", edgecolor="black")
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"{title} — KVCOMM entropy")
        ax.set_ylabel("Shannon entropy (bits)")
        for i, (b, r) in enumerate(zip(bars, rates)):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"gate={r:.0%}", ha="center", fontsize=8, color="#555")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _md_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    """rows: list of dicts; columns: list of (key, header)"""
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


def generate_report(in_path: str, out_path: str, plot_dir: str) -> str:
    with open(in_path) as f:
        data = json.load(f)
    cfg = data.get("config", {})
    seg = data.get("segment_summary", {})
    pairs = data.get("pair_aggregations", {})
    pool = data.get("pool_entropy_stats", {})
    struct = data.get("structural_gate_stats", {})

    os.makedirs(plot_dir, exist_ok=True)
    plots_made = []
    # Plot 1: d_norm_avg by ast_type pair
    if pairs.get("by_ast_type_pair"):
        items = _sort_by_key(pairs["by_ast_type_pair"], "d_norm_avg")
        plots_made.append(_plot_bar(
            items, "d_norm_avg",
            "d_norm_avg by AST-type pair (top 12, sorted)",
            "d_norm", os.path.join(plot_dir, "d_norm_by_ast_type.png"),
        ))
    # Plot 2: d_norm_avg by template pair
    if pairs.get("by_template_pair"):
        items = _sort_by_key(pairs["by_template_pair"], "d_norm_avg")
        plots_made.append(_plot_bar(
            items, "d_norm_avg",
            "d_norm_avg by template pair",
            "d_norm", os.path.join(plot_dir, "d_norm_by_template.png"),
        ))
    # Plot 3: d_norm_avg by length pair
    if pairs.get("by_length_pair"):
        items = _sort_by_key(pairs["by_length_pair"], "d_norm_avg")
        plots_made.append(_plot_bar(
            items, "d_norm_avg",
            "d_norm_avg by length-bin pair",
            "d_norm", os.path.join(plot_dir, "d_norm_by_length.png"),
        ))
    # Plot 4: entropy by dimension
    if pool:
        plots_made.append(_plot_entropy_bars(pool, os.path.join(plot_dir, "entropy_by_dimension.png")))
    # Plot 5: within vs cross
    if struct:
        plots_made.append(_plot_within_vs_cross(struct, os.path.join(plot_dir, "within_vs_cross_ast.png")))

    md = []
    md.append("# AST × KV-Distance Analysis Report\n")
    md.append("> Auto-generated by `report_generator.py` from `data/distance_7b.json`\n")
    md.append("## 1. Configuration\n")
    md.append(_md_table(
        [cfg],
        [("model", "Model"), ("max_segments", "Max segments"), ("max_seq_len", "Max seq len"),
         ("kvcomm_threshold", "KVCOMM threshold (γ)"), ("selected_layers", "Selected layers")],
    ))
    md.append("\n## 2. Segment summary\n")
    md.append(f"- Total segments captured: **{seg.get('n_segments')}**")
    md.append(f"- by AST type: `{seg.get('by_ast_type')}`")
    md.append(f"- by template: `{seg.get('by_template')}`")
    md.append(f"- by length bin: `{seg.get('by_length_bin')}`\n")

    md.append("## 3. Pairwise L2 distance — top findings\n")
    md.append("`d_norm = d_mean / sqrt(seq_len)` — length-normalised, comparable across bins.\n")
    if pairs.get("by_ast_type_pair"):
        md.append("### 3.1 By AST-type pair (top 8 closest)\n")
        closest = _sort_by_key(pairs["by_ast_type_pair"], "d_norm_avg")[:8]
        rows = [{"pair": k, "count": v["count"], "d_mean_p50": v["d_mean_p50"],
                 "d_norm_avg": v["d_norm_avg"], "d_norm_p50": v["d_norm_p50"]}
                for k, v in closest]
        md.append(_md_table(rows, [("pair", "AST-type pair"), ("count", "n"),
                                    ("d_mean_p50", "d_mean p50"),
                                    ("d_norm_avg", "d_norm avg"), ("d_norm_p50", "d_norm p50")]))
        md.append("\n### 3.2 By AST-type pair (top 8 farthest)\n")
        farthest = _sort_by_key(pairs["by_ast_type_pair"], "d_norm_avg", reverse=True)[:8]
        rows = [{"pair": k, "count": v["count"], "d_mean_p50": v["d_mean_p50"],
                 "d_norm_avg": v["d_norm_avg"], "d_norm_p50": v["d_norm_p50"]}
                for k, v in farthest]
        md.append(_md_table(rows, [("pair", "AST-type pair"), ("count", "n"),
                                    ("d_mean_p50", "d_mean p50"),
                                    ("d_norm_avg", "d_norm avg"), ("d_norm_p50", "d_norm p50")]))

    if pairs.get("by_template_pair"):
        md.append("\n### 3.3 By template pair\n")
        items = _sort_by_key(pairs["by_template_pair"], "d_norm_avg")
        rows = [{"pair": k, "count": v["count"], "d_norm_avg": v["d_norm_avg"]}
                for k, v in items]
        md.append(_md_table(rows, [("pair", "Template pair"), ("count", "n"),
                                    ("d_norm_avg", "d_norm avg")]))

    if pairs.get("by_length_pair"):
        md.append("\n### 3.4 By length-bin pair\n")
        items = _sort_by_key(pairs["by_length_pair"], "d_norm_avg")
        rows = [{"pair": k, "count": v["count"], "d_norm_avg": v["d_norm_avg"]}
                for k, v in items]
        md.append(_md_table(rows, [("pair", "Length pair"), ("count", "n"),
                                    ("d_norm_avg", "d_norm avg")]))

    md.append("\n## 4. KVCOMM pool entropy\n")
    md.append("Per the KVCOMM paper, the entropy of the softmax distribution over a pool of anchor KVs "
              "indicates whether a query has a close match (low entropy) or is unlike all candidates "
              "(high entropy). A pool is 'coherent' if its members all have low entropy to each other.\n")
    for dim in ["by_ast_type", "by_template", "by_length_bin"]:
        if dim not in pool:
            continue
        md.append(f"### 4.{ {'by_ast_type':1,'by_template':2,'by_length_bin':3}[dim] } {dim}\n")
        items = pool[dim]
        rows = []
        for k, v in items.items():
            if "skipped" in v:
                continue
            rows.append({
                "key": k, "count": v["count"],
                "entropy_avg": v["entropy_avg"],
                "entropy_min": v["entropy_min"],
                "entropy_max": v["entropy_max"],
                "gate_pass_rate": v["gate_pass_rate"],
            })
        rows.sort(key=lambda r: r["entropy_avg"])
        md.append(_md_table(rows, [("key", dim), ("count", "n"),
                                    ("entropy_avg", "entropy avg"),
                                    ("entropy_min", "entropy min"),
                                    ("entropy_max", "entropy max"),
                                    ("gate_pass_rate", "KVCOMM gate pass rate")]))

    md.append("\n## 5. Within vs cross AST-type — the central question\n")
    if struct:
        w = struct.get("within_ast_type", {})
        c = struct.get("cross_ast_type", {})
        ratio = struct.get("ratio_within_to_cross_d_norm")
        verdict = (
            "**Within-type pairs are CLOSER per token** → AST structure IS a useful reuse signal"
            if ratio is not None and ratio < 1
            else "**Cross-type pairs are closer per token** → AST structure is NOT a useful reuse signal"
        )
        md.append(f"- within-AST-type: n={w.get('count')}, d_norm_avg={w.get('d_norm_avg', 0):.4f}")
        md.append(f"- cross-AST-type:  n={c.get('count')}, d_norm_avg={c.get('d_norm_avg', 0):.4f}")
        md.append(f"- ratio (within / cross) = **{ratio:.4f}**")
        md.append(f"- {verdict}\n")
        md.append("Interpretation: " + struct.get("interpretation", ""))
    md.append("\n## 6. Plots\n")
    for p in plots_made:
        if p:
            md.append(f"- ![]({os.path.relpath(p, os.path.dirname(out_path))})")
    md.append("\n## 7. Implications for sglang-kvflow anchor gate\n")
    md.append("Based on the ratio in §5:\n")
    if struct and struct.get("ratio_within_to_cross_d_norm") is not None:
        r = struct["ratio_within_to_cross_d_norm"]
        if r < 0.85:
            md.append("- Strong evidence: add a `structural_distance_gate` tier to `anchor_match.py` "
                      "that allows reuse when AST-type-equal even if `content_signature` differs.")
        elif r < 1.0:
            md.append("- Weak evidence: structure is mildly informative. Add as a *secondary* hint "
                      "(reduces confidence of a structural match but does not enable it on its own).")
        else:
            md.append("- No evidence that structure helps. KEEP the content-signature gate as the "
                      "primary and only gate; AST remains a locator-only metadata.")
    md.append("\n## 8. Method notes\n")
    md.append("- K/V captured via `use_cache=True` on the last 4 layers of "
              "Qwen2.5-Coder-7B-Instruct (28 layers total, 4 KV heads, head_dim=128).")
    md.append("- Distance metric: L2 norm over the sequence dim, mean over "
              "(layer, head, dim) — same as KVCOMM's `_compute_anchor_weight_entry` prefix K branch.")
    md.append("- Pool entropy: softmax over L2 distances against the pool, then "
              "Shannon entropy in bits — same as KVCOMM's `predict_as_anchor` gate.")
    md.append("- KVCOMM gate: pass if `entropy <= γ × log2(pool_size)` with γ = 0.3.")
    md.append("- Each segment was prompted as a 'Summarise this code' task; K/V comes "
              "from the model processing the code under a fixed instruction prefix.")

    md_text = "\n".join(md) + "\n"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(md_text)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default="/home/gfy/CodeMAS_Project/sglang-kvflow/results/ast_kv_distance/data/distance_7b.json")
    p.add_argument("--out", default="/home/gfy/CodeMAS_Project/sglang-kvflow/results/ast_kv_distance/report.md")
    p.add_argument("--plots", default="/home/gfy/CodeMAS_Project/sglang-kvflow/results/ast_kv_distance/plots")
    args = p.parse_args()
    out = generate_report(args.in_path, args.out, args.plots)
    print(f"[report] wrote {out}")


if __name__ == "__main__":
    main()
