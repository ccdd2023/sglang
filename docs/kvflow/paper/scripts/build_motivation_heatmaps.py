#!/usr/bin/env python3
"""Column-width motivation lines/areas/heatmaps from frozen JSON. No GPU."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from derive_7b_copier_motivation import group_extra_series
from derive_7b_motivation import DEFAULT_PLAN, group_coverage_series

ATTN = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_global_block_attention_20260806/frozen26_r2/RESULT.json"
)
SPARSE = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_attention_sparsity_20260806/frozen20/RESULT.json"
)
FOUR = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_common_prompt_attention_kv_mechanism_20260813/"
    "FOUR_ARM_RESULT.json"
)
FOUR_DIR = FOUR.parent
COPIER = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_swebench_7b_sota_copiers_20260824"
)
MOTIVATION = DEFAULT_PLAN.with_name("MOTIVATION.json")
FIG = Path(__file__).resolve().parents[1] / "figures"

BLUE = "#2f6db3"
ORANGE = "#d96b2b"
TEAL = "#5bb3a7"
NAVY = "#1b4f8a"
GRAY = "#8a8a8a"
LIME = "#c5e0a5"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(plt) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def _save(plt, fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def _binned_means(xs: list[int], ys: list[float], bins: int = 8) -> tuple[list[float], list[float]]:
    if len(xs) != len(ys) or not xs:
        raise ValueError("coverage bin lengths")
    lo, hi = min(xs), max(xs)
    width = (hi - lo) / bins
    centres: list[float] = []
    means: list[float] = []
    for i in range(bins):
        left = lo + i * width
        right = hi if i == bins - 1 else lo + (i + 1) * width
        bucket = [
            y
            for x, y in zip(xs, ys)
            if (left <= x <= right if i == bins - 1 else left <= x < right)
        ]
        if bucket:
            centres.append((left + right) / 2)
            means.append(statistics.fmean(bucket))
    return centres, means


def extra_token_series() -> dict[str, list[int]]:
    coding = json.loads((COPIER / "PLAN.coding.json").read_text(encoding="utf-8"))
    kvcomm = json.loads((COPIER / "PLAN.kvcomm.json").read_text(encoding="utf-8"))
    series = group_extra_series(coding, kvcomm)
    if len(series["extra"]) != 235:
        raise ValueError(f"copier group mismatch {len(series['extra'])}")
    if min(series["extra"]) < 0:
        raise ValueError("negative extra tokens")
    return series


def _jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def module_tv_by_case() -> dict[str, list[float]]:
    arms = {
        "coding": FOUR_DIR / "OBSERVATIONS_SGLANG.jsonl",
        "cacheblend": FOUR_DIR / "OBSERVATIONS_CACHEBLEND.jsonl",
        "kvcomm": FOUR_DIR / "OBSERVATIONS_KVCOMM.jsonl",
    }
    per_arm: dict[str, dict[str, float]] = {}
    for name, path in arms.items():
        by_id: dict[str, float] = {}
        for rec in _jsonl(path):
            if rec.get("physical_reuse") is False:
                by_id[str(rec["case_id"])] = float("nan")
                continue
            tvs = [float(layer["module_attention_tv"]) for layer in rec["layers"]]
            by_id[str(rec["case_id"])] = statistics.median(tvs)
        per_arm[name] = by_id
    ids = sorted(per_arm["coding"].keys() & per_arm["cacheblend"].keys() & per_arm["kvcomm"].keys())
    if len(ids) != 8:
        raise ValueError(f"four-arm case mismatch {len(ids)}")
    ids.sort(key=lambda case: per_arm["cacheblend"][case])
    return {
        "coding": [per_arm["coding"][i] for i in ids],
        "cacheblend": [per_arm["cacheblend"][i] for i in ids],
        "kvcomm": [per_arm["kvcomm"][i] for i in ids],
    }


def main() -> None:
    plt = _pyplot()
    _style(plt)

    attn = json.loads(ATTN.read_text(encoding="utf-8"))
    four = json.loads(FOUR.read_text(encoding="utf-8"))
    sparse = json.loads(SPARSE.read_text(encoding="utf-8"))
    if attn.get("aggregate", {}).get("cases") != 26:
        raise ValueError("frozen26 case count drifted")
    if four.get("status") != "COMPLETE":
        raise ValueError("four-arm probe not COMPLETE")
    if int(sparse.get("aggregate", {}).get("cases", 0)) != 20:
        raise ValueError("frozen20 case count drifted")

    suffix_med = float(attn["aggregate"]["suffix_tv"]["median"])
    nxt_med = float(attn["aggregate"]["generation_tv"]["median"])
    form_med = float(attn["aggregate"]["formation_tv"]["median"])
    cases = sorted(
        attn["case_summaries"],
        key=lambda row: float(row["median_formation_tv_over_layers"]),
    )
    xs = list(range(1, len(cases) + 1))
    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    ax.plot(
        xs,
        [float(c["median_suffix_tv_over_layers"]) for c in cases],
        color=TEAL,
        linewidth=1.4,
        marker="o",
        markersize=2.4,
        label="Suffix TV",
    )
    ax.plot(
        xs,
        [float(c["median_generation_tv_over_layers"]) for c in cases],
        color=BLUE,
        linewidth=1.4,
        marker="o",
        markersize=2.4,
        label="Next-action TV",
    )
    ax.plot(
        xs,
        [float(c["median_formation_tv_over_layers"]) for c in cases],
        color=ORANGE,
        linewidth=1.6,
        marker="o",
        markersize=2.4,
        label="Formation TV",
    )
    ax.axhline(form_med, color=ORANGE, linewidth=0.7, linestyle="--")
    ax.axhline(suffix_med, color=TEAL, linewidth=0.7, linestyle=":")
    ax.set_xlabel("Islands sorted by formation TV")
    ax.set_ylabel("TV (Dense vs splice)")
    ax.set_title("Where the loss sits (3B, 26 islands)")
    ax.set_xlim(1, 26)
    ax.set_ylim(0, 0.25)
    ax.legend(frameon=False, loc="upper left", ncol=1)
    _save(plt, fig, "fig_tv_locus")

    per_prompt = module_tv_by_case()
    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    px = list(range(1, 9))
    ax.plot(px, per_prompt["coding"], color=BLUE, linewidth=1.6, marker="o", markersize=4, label="File-module")
    ax.plot(px, per_prompt["cacheblend"], color=TEAL, linewidth=1.6, marker="o", markersize=4, label="CacheBlend")
    ax.plot(px, per_prompt["kvcomm"], color=ORANGE, linewidth=1.6, marker="o", markersize=4, label="KVCOMM")
    ax.axhline(
        float(four["summaries"]["coding_aware"]["attention_tv_median"]),
        color=BLUE,
        linewidth=0.7,
        linestyle=":",
    )
    ax.set_xlabel("Same 8 prompts")
    ax.set_ylabel("Module attention TV")
    ax.set_title("Same 8 prompts, 3B (not Accuracy)")
    ax.set_xticks(px)
    ax.set_ylim(0, 0.13)
    ax.legend(frameon=False, loc="upper right")
    _save(plt, fig, "fig_module_tv")

    cov = group_coverage_series(DEFAULT_PLAN)
    n = len(cov["copied_frac"])
    gx = list(range(1, n + 1))
    extra = extra_token_series()
    ex = list(range(1, len(extra["extra"]) + 1))
    mot = json.loads(MOTIVATION.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.45))
    axes[0].scatter(gx, cov["lcp_frac"], s=6, alpha=0.22, color=TEAL, linewidths=0, zorder=2)
    axes[0].scatter(gx, cov["copied_frac"], s=6, alpha=0.22, color=BLUE, linewidths=0, zorder=2)
    bx, blcp = _binned_means(gx, cov["lcp_frac"])
    _, bfile = _binned_means(gx, cov["copied_frac"])
    axes[0].plot(bx, blcp, color=TEAL, linewidth=2.0, marker="o", markersize=3.2, label="Radix LCP")
    axes[0].plot(bx, bfile, color=BLUE, linewidth=2.0, marker="o", markersize=3.2, label="File-island")
    axes[0].axhline(float(mot["mean_radix_fraction"]), color=TEAL, linewidth=0.7, linestyle=":")
    axes[0].axhline(float(mot["mean_lossy_fraction"]), color=BLUE, linewidth=0.7, linestyle=":")
    axes[0].set_ylabel("Share of prompt")
    axes[0].set_title("Complementary reuse (7B PLAN)")
    axes[0].set_xlim(1, n)
    axes[0].set_ylim(0, 0.85)
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].plot(ex, extra["file"], color=BLUE, linewidth=1.15, label="File-module")
    axes[1].plot(ex, extra["kvcomm"], color=ORANGE, linewidth=1.15, label="KVCOMM-style")
    axes[1].plot(ex, extra["extra"], color=GRAY, linewidth=1.0, linestyle="--", label="Extra (refused)")
    axes[1].set_xlabel("Groups (top: by length; bottom: by extra tokens)")
    axes[1].set_ylabel("Copied tokens")
    axes[1].set_title("File gate vs unconstrained copy")
    axes[1].set_xlim(1, len(ex))
    axes[1].legend(frameon=False, loc="upper left", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_motivation_coverage.pdf")
    fig.savefig(FIG / "fig_motivation_coverage.png")
    fig.savefig(FIG / "fig_motivation_extra.pdf")
    fig.savefig(FIG / "fig_motivation_extra.png")
    plt.close(fig)

    layers = sparse["representative_profile"]["layers"]
    frac_keys = ["0.01", "0.05", "0.1", "0.2", "0.5"]
    frac_x = [1, 5, 10, 20, 50]
    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    curves = []
    for layer in layers:
        ys = [100 * float(layer["global_attention_mass_by_token_fraction"][k]) for k in frac_keys]
        curves.append(ys)
        ax.plot(frac_x, ys, color=LIME, linewidth=1.0, alpha=0.75)
    median_curve = [statistics.median(col) for col in zip(*curves)]
    ax.plot(
        frac_x,
        median_curve,
        color=NAVY,
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="Median over layers",
    )
    ax.axhline(80.1, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Top token fraction (%)")
    ax.set_ylabel("Attention mass (%)")
    ax.set_title("Dense attention sparsity (3B)")
    ax.set_xlim(1, 50)
    ax.set_ylim(30, 105)
    ax.legend(frameon=False, loc="lower right")
    _save(plt, fig, "fig_attn_proxy")

    print(
        "wrote motivation figures",
        {
            "suffix": suffix_med,
            "formation": form_med,
            "next_action": nxt_med,
            "coverage_groups": n,
            "extra_sum": int(sum(extra["extra"])),
            "coding_tv": float(four["summaries"]["coding_aware"]["attention_tv_median"]),
        },
    )


if __name__ == "__main__":
    main()
