"""Cross-model transferability report for the predicted_distance_table.

Reads results/lookup_table_transferability/data/predicted_distance_table_<slug>.json
for each of the 4 studied models, and produces:
  - report.md
  - plots/d_norm_per_axis_<model>.png  (one per model)
  - plots/cross_model_d_norm_heatmap.png  (4x4 |d_norm_A - d_norm_B|)
  - cross_model_comparison.json  (machine-readable)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
DATA_DIR = PROJECT_ROOT / "results" / "lookup_table_transferability" / "data"
PLOT_DIR = PROJECT_ROOT / "results" / "lookup_table_transferability" / "plots"

MODELS = [
    ("Qwen/Qwen2.5-Coder-7B-Instruct", "qwen-qwen2.5-coder-7b-instruct"),
    ("Qwen/Qwen2.5-Coder-3B-Instruct", "qwen-qwen2.5-coder-3b-instruct"),
    ("Qwen/Qwen2.5-7B-Instruct", "qwen-qwen2.5-7b-instruct"),
    ("Qwen/Qwen3-8B", "qwen-qwen3-8b"),
]


def _load_table(slug: str) -> dict | None:
    path = DATA_DIR / f"predicted_distance_table_{slug}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _per_axis_per_model(tables: dict[str, dict]) -> dict:
    """For each model, compute the per-axis aggregated d_norm: by position,
    by system, by surrounding. Aggregated across all length bins."""
    out: dict = {}
    for model, table in tables.items():
        if not table or not table.get("cells"):
            out[model] = None
            continue
        cells = table["cells"]
        by_pos = defaultdict(list)
        by_sys = defaultdict(list)
        by_sur = defaultdict(list)
        for c in cells:
            by_pos[c["position_offset"]].append(c["predicted_d_norm_mean"])
            by_sys[c["system_prompt_class"]].append(c["predicted_d_norm_mean"])
            by_sur[c["surrounding_code_class"]].append(c["predicted_d_norm_mean"])
        out[model] = {
            "by_position_offset": {k: sum(v) / len(v) for k, v in by_pos.items()},
            "by_system_prompt_class": {k: sum(v) / len(v) for k, v in by_sys.items()},
            "by_surrounding_code_class": {k: sum(v) / len(v) for k, v in by_sur.items()},
            "global_baseline": table.get("global", {}).get("predicted_d_norm_baseline"),
            "global_max": table.get("global", {}).get("predicted_d_norm_max_observed"),
        }
    return out


def _pairwise_diff_matrix(per_model: dict) -> tuple[list[str], np.ndarray]:
    """For each pair of models, compute the mean |d_norm_A - d_norm_B| across all cells."""
    models = [m for m, v in per_model.items() if v is not None]
    n = len(models)
    matrix = np.zeros((n, n))
    for i, ma in enumerate(models):
        for j, mb in enumerate(models):
            ta = _load_table(_slug_for(ma))
            tb = _load_table(_slug_for(mb))
            if not ta or not tb:
                continue
            a_cells = {(c["length_bin"], c["position_offset"],
                        c["system_prompt_class"], c["surrounding_code_class"]):
                       c["predicted_d_norm_mean"] for c in ta.get("cells", [])}
            b_cells = {(c["length_bin"], c["position_offset"],
                        c["system_prompt_class"], c["surrounding_code_class"]):
                       c["predicted_d_norm_mean"] for c in tb.get("cells", [])}
            common = set(a_cells) & set(b_cells)
            if not common:
                continue
            matrix[i, j] = sum(abs(a_cells[k] - b_cells[k]) for k in common) / len(common)
    return models, matrix


def _slug_for(model_name: str) -> str:
    s = model_name.lower().replace("/", "-")
    out = []
    for ch in s:
        if ch.isalnum() or ch == ".":
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-")


def _heatmap(models: list[str], matrix: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.split("/")[-1].replace("-Instruct", "") for m in models],
                       rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.split("/")[-1].replace("-Instruct", "") for m in models],
                       fontsize=9)
    ax.set_title("Mean |d_norm_A - d_norm_B| across all 144 cells\n(0 = tables identical, larger = more divergent)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.mean() else "black", fontsize=10)
    plt.colorbar(im, ax=ax, label="mean |Δd_norm|")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _per_axis_bar(per_model: dict, axis: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    color_cycle = ["#3a86ff", "#fb5607", "#8338ec", "#06d6a0"]
    for i, (model, axis_data) in enumerate(per_model.items()):
        if axis_data is None:
            continue
        data = axis_data[axis]
        xs = list(data.keys())
        ys = [data[k] for k in xs]
        ax.plot(xs, ys, "o-", label=model.split("/")[-1].replace("-Instruct", ""),
                color=color_cycle[i % len(color_cycle)])
    ax.set_xlabel(axis)
    ax.set_ylabel("mean d_norm")
    ax.set_title(f"d_norm vs {axis} (one line per model)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _verdict(matrix: np.ndarray, models: list[str]) -> str:
    if matrix.size == 0:
        return "no data"
    off_diag = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j:
                off_diag.append(matrix[i, j])
    if not off_diag:
        return "no data"
    mean_diff = sum(off_diag) / len(off_diag)
    if mean_diff < 0.15:
        return "**Strong portable**: tables agree within ±0.15 d_norm on average → 7-8B model-agnostic"
    if mean_diff < 0.30:
        return "**Medium portable**: tables diverge moderately → family-specific but no per-model calibration needed"
    return "**Weak portable**: tables diverge significantly → per-model calibration required"


def main() -> None:
    tables = {model: _load_table(slug) for model, slug in MODELS}
    available = {m: t for m, t in tables.items() if t is not None}
    print(f"[cross_model] loaded {len(available)} / {len(MODELS)} models")
    if not available:
        print("[cross_model] no data; run run_all.sh first")
        return

    per_model = _per_axis_per_model(available)
    models_with_data, diff_matrix = _pairwise_diff_matrix(per_model)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _heatmap(models_with_data, diff_matrix, PLOT_DIR / "cross_model_d_norm_heatmap.png")
    for axis in ("by_position_offset", "by_system_prompt_class", "by_surrounding_code_class"):
        _per_axis_bar(per_model, axis, PLOT_DIR / f"d_norm_{axis}.png")

    # machine-readable summary
    summary = {
        "models": list(available.keys()),
        "per_model": per_model,
        "pairwise_mean_abs_diff": {
            f"{models_with_data[i]} vs {models_with_data[j]}": float(diff_matrix[i, j])
            for i in range(len(models_with_data))
            for j in range(len(models_with_data))
        },
        "verdict": _verdict(diff_matrix, models_with_data),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "cross_model_comparison.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # markdown report
    lines = [
        "# Cross-Model Transferability of predicted_distance_table\n",
        "> Per-model re-run of `results/same_code_context_variation/` on 4 models.\n",
        "## 1. Models studied\n",
    ]
    for m, _ in MODELS:
        lines.append(f"- {m}  {'(loaded)' if m in available else '(missing)'}")
    lines.append("\n## 2. Per-axis d_norm per model\n")
    for axis in ("by_position_offset", "by_system_prompt_class", "by_surrounding_code_class"):
        lines.append(f"### 2.{ {'by_position_offset':1,'by_system_prompt_class':2,'by_surrounding_code_class':3}[axis] } {axis}\n")
        for model, axis_data in per_model.items():
            if axis_data is None:
                continue
            kv = axis_data[axis]
            pretty = model.split("/")[-1].replace("-Instruct", "")
            pretty_kv = ", ".join(f"{k}={v:.3f}" for k, v in sorted(kv.items(),
                                                                 key=lambda kv: (kv[0] if axis != "by_position_offset" else str(kv[0]))))
            lines.append(f"- **{pretty}**: {pretty_kv}")
        lines.append("")
    lines.append("## 3. Pairwise mean |Δd_norm|\n")
    lines.append("See `plots/cross_model_d_norm_heatmap.png` for the 4×4 matrix.\n")
    lines.append("| pair | mean |Δd_norm| |")
    lines.append("|---|---|")
    for i in range(len(models_with_data)):
        for j in range(len(models_with_data)):
            if i < j:
                lines.append(f"| {models_with_data[i].split('/')[-1]}  vs  {models_with_data[j].split('/')[-1]} | {diff_matrix[i, j]:.4f} |")
    lines.append("\n## 4. Verdict\n")
    lines.append(summary["verdict"])
    lines.append("\n## 5. Plots\n")
    for f in sorted(PLOT_DIR.glob("*.png")):
        lines.append(f"- ![]({f.name})")
    out = DATA_DIR.parent / "report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[cross_model] wrote {out}")


if __name__ == "__main__":
    main()
