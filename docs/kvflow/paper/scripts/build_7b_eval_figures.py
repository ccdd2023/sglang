#!/usr/bin/env python3
"""Rebuild 7B eval figures from frozen 137185 / 139839 / 137400. No GPU."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from derive_7b_137185_slices import DEFAULT_ART
from derive_96092_slices import pair_rows, read_json

FIG = Path(__file__).resolve().parents[1] / "figures"
from impactkv_paths import artifact_root

_ART = artifact_root()
PREFIX_ON = _ART / "impactkv_swebench_7b_prefix_on_20260825"
COPIERS = _ART / "impactkv_swebench_7b_sota_copiers_20260824"
GRAY = "#8a8a8a"
BLUE = "#2f6db3"
ORANGE = "#d96b2b"
TEAL = "#5bb3a7"
NAVY = "#1b4f8a"
GREEN = "#3f9d5d"
LIME = "#c5e0a5"


def figure_values(art: Path = DEFAULT_ART) -> dict[str, object]:
    result = read_json(art / "RESULT.json")
    slices = read_json(art / "SLICES.json")
    if result.get("status") != "COMPLETE":
        raise ValueError("137185 RESULT is not COMPLETE")
    n_use = slices["n_use_including_one_source_build"]
    length = slices["length_buckets"]
    islands = slices["island_count_slices"]
    delta = slices["abs_delta_slices"]
    frac = slices["copied_fraction_quartiles"]["slices"]
    mech = result["mechanism"]
    ttft = slices["ttft_ms"]
    prefix = read_json(PREFIX_ON / "RESULT.json")
    copiers = read_json(COPIERS / "RESULT.json")
    return {
        "cache_ready": float(result["latency"]["cache_ready_speedup_ratio_of_means"]),
        "n4": float(result["latency"]["n4_including_one_source_build_speedup"]),
        "n_use": {int(k): float(v) for k, v in n_use.items()},
        "length": {k: float(length[k]["cache_ready_speedup"]) for k in length},
        "island_groups": {int(k): int(islands[k]["groups"]) for k in islands},
        "island_speedup": {
            int(k): float(islands[k]["cache_ready_speedup"]) for k in islands
        },
        "delta_speedup": {
            k: float(delta[k]["cache_ready_speedup"]) for k in delta
        },
        "frac_speedup": {
            k: float(frac[k]["cache_ready_speedup"]) for k in frac
        },
        "ttft": {k: float(v) if isinstance(v, (int, float)) else v for k, v in ttft.items()},
        "planned": int(mech["expected_copy_events"]),
        "copied": int(mech["copy_events"]),
        "fallback": int(mech["fallback_events"]),
        "prerotate": int(mech["source_prerotation_events"]),
        "prefix_on": {
            "prefix_only": float(
                prefix["latency"]["prefix_only"]["cache_ready_speedup_ratio_of_means"]
            ),
            "lossy_only": float(
                prefix["latency"]["lossy_only"]["cache_ready_speedup_ratio_of_means"]
            ),
            "dual": float(
                prefix["latency"]["dual"]["cache_ready_speedup_ratio_of_means"]
            ),
            "copy_on_prefix": float(prefix["algorithm_bars"]["copy_on_prefix"]),
        },
        "copiers": {
            "file_module": float(
                copiers["coding"]["latency"]["cache_ready_speedup_ratio_of_means"]
            ),
            "kvcomm": float(
                copiers["kvcomm_style"]["latency"]["cache_ready_speedup_ratio_of_means"]
            ),
            "cacheblend": float(
                copiers["cacheblend_style"]["latency"][
                    "cache_ready_speedup_ratio_of_means"
                ]
            ),
            "file_agree": float(
                copiers["coding"]["one_token_output_agreement"]["fraction"]
            ),
            "kvcomm_agree": float(
                copiers["kvcomm_style"]["one_token_output_agreement"]["fraction"]
            ),
            "cacheblend_agree": float(
                copiers["cacheblend_style"]["one_token_output_agreement"]["fraction"]
            ),
        },
        "scatter_frac": [float(x) for x in slices["group_scatter"]["copied_fraction"]],
        "scatter_speedup": [
            float(x) for x in slices["group_scatter"]["cache_ready_speedup"]
        ],
    }


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
            "legend.fontsize": 7.5,
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


def _annotate_bars(ax, bars, fmt: str = "{:.2f}x") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.03,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=7,
        )


def _empirical_cdf(values: list[float]) -> tuple[list[float], list[float]]:
    ordered = sorted(values)
    n = len(ordered)
    ys = [(i + 1) / n for i in range(n)]
    return ordered, ys


def _binned_means(
    xs: list[float], ys: list[float], bins: int = 8
) -> tuple[list[float], list[float]]:
    if len(xs) != len(ys) or not xs:
        raise ValueError("scatter lengths")
    lo, hi = min(xs), max(xs)
    width = (hi - lo) / bins
    centres: list[float] = []
    means: list[float] = []
    for i in range(bins):
        left = lo + i * width
        right = hi if i == bins - 1 else lo + (i + 1) * width
        bucket = [y for x, y in zip(xs, ys) if left <= x <= right] if i == bins - 1 else [
            y for x, y in zip(xs, ys) if left <= x < right
        ]
        if bucket:
            centres.append((left + right) / 2)
            means.append(statistics.fmean(bucket))
    return centres, means


def main() -> None:
    plt = _pyplot()
    _style(plt)
    values = figure_values()
    dense = read_json(DEFAULT_ART / "dense.json")
    reuse = read_json(DEFAULT_ART / "reuse.json")
    pairs = pair_rows(dense, reuse)
    dense_ttft = [float(a["ttft_ms"]) for a, _ in pairs.values()]
    reuse_ttft = [float(b["ttft_ms"]) for _, b in pairs.values()]

    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    dx, dy = _empirical_cdf(dense_ttft)
    rx, ry = _empirical_cdf(reuse_ttft)
    ax.plot(dx, dy, color=GRAY, linewidth=1.8, label="Dense")
    ax.plot(rx, ry, color=BLUE, linewidth=1.8, label="ImpactKV")
    ax.axvline(values["ttft"]["dense_p50"], color=GRAY, linewidth=0.7, linestyle=":")
    ax.axvline(values["ttft"]["reuse_p50"], color=BLUE, linewidth=0.7, linestyle=":")
    ax.set_xlabel("Cache-ready TTFT (ms)")
    ax.set_ylabel("CDF")
    ax.set_xlim(150, 1350)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Paired TTFT CDF (705 pairs)")
    _save(plt, fig, "fig_ttft_cdf")

    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    ax.scatter(
        values["scatter_frac"],
        values["scatter_speedup"],
        s=12,
        alpha=1.0,
        color=BLUE,
        edgecolors="white",
        linewidths=0.25,
        zorder=2,
    )
    bx, by = _binned_means(values["scatter_frac"], values["scatter_speedup"])
    ax.plot(bx, by, color=ORANGE, linewidth=2.0, marker="o", markersize=4, label="Binned mean")
    ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Copied-token fraction")
    ax.set_ylabel("Cache-ready speedup")
    ax.set_ylim(0.7, 3.6)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Speedup vs copied fraction")
    _save(plt, fig, "fig_copied_speedup")

    fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.35))
    length_order = ["<3K", "3-5K", "5-7K", ">=7K"]
    length_labels = ["<3K", "3–5K", "5–7K", "≥7K"]
    bars = axes[0, 0].bar(
        length_labels,
        [values["length"][k] for k in length_order],
        color=[LIME, TEAL, BLUE, NAVY],
        width=0.7,
    )
    axes[0, 0].axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[0, 0].set_title("Prompt length")
    axes[0, 0].set_ylabel("Speedup vs Dense")
    axes[0, 0].set_ylim(0, 2.4)
    _annotate_bars(axes[0, 0], bars)

    island_order = [1, 2, 3]
    bars = axes[0, 1].bar(
        ["1", "2", "3"],
        [values["island_speedup"][k] for k in island_order],
        color=[LIME, TEAL, BLUE],
        width=0.7,
    )
    axes[0, 1].axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[0, 1].set_title("Islands per group")
    axes[0, 1].set_ylim(0, 2.4)
    _annotate_bars(axes[0, 1], bars)

    frac_order = ["Q1", "Q2", "Q3", "Q4"]
    bars = axes[1, 0].bar(
        ["Q1", "Q2", "Q3", "Q4"],
        [values["frac_speedup"][k] for k in frac_order],
        color=[LIME, TEAL, BLUE, NAVY],
        width=0.7,
    )
    axes[1, 0].axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[1, 0].set_title("Copied-token quartile")
    axes[1, 0].set_ylabel("Speedup vs Dense")
    axes[1, 0].set_ylim(0, 2.6)
    _annotate_bars(axes[1, 0], bars)

    delta_order = ["<500", "500-1500", "1500-3000", ">=3000"]
    delta_labels = ["<0.5k", "0.5–1.5k", "1.5–3k", "≥3k"]
    bars = axes[1, 1].bar(
        delta_labels,
        [values["delta_speedup"][k] for k in delta_order],
        color=[LIME, TEAL, BLUE, NAVY],
        width=0.7,
    )
    axes[1, 1].axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[1, 1].set_title(r"Absolute shift $|\Delta|$")
    axes[1, 1].set_ylim(0, 2.4)
    _annotate_bars(axes[1, 1], bars)
    for ax in axes.flat:
        ax.tick_params(axis="x", labelsize=6.5, labelrotation=18)
    fig.tight_layout()
    _save(plt, fig, "fig_slices")

    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    labels = ["Dense", "prefix-only", "lossy-only", "dual"]
    heights = [
        1.0,
        values["prefix_on"]["prefix_only"],
        values["prefix_on"]["lossy_only"],
        values["prefix_on"]["dual"],
    ]
    bars = ax.bar(labels, heights, color=[GRAY, TEAL, BLUE, NAVY], width=0.7)
    ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Speedup vs this Dense")
    ax.set_title("Prefix vs lossy isolation")
    ax.set_ylim(0, 2.55)
    _annotate_bars(ax, bars)
    ax.tick_params(axis="x", labelsize=7)
    _save(plt, fig, "fig_prefix_on")

    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.65))
    copier_labels = ["File-module", "KVCOMM-style", "CacheBlend-style"]
    copier_speed = [
        values["copiers"]["file_module"],
        values["copiers"]["kvcomm"],
        values["copiers"]["cacheblend"],
    ]
    copier_agree = [
        100 * values["copiers"]["file_agree"],
        100 * values["copiers"]["kvcomm_agree"],
        100 * values["copiers"]["cacheblend_agree"],
    ]
    bars = axes[0].bar(copier_labels, copier_speed, color=[BLUE, ORANGE, TEAL], width=0.65)
    axes[0].axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Cache-ready speedup")
    axes[0].set_title("Same-engine TTFT")
    axes[0].set_ylim(0, 2.5)
    _annotate_bars(axes[0], bars)
    bars = axes[1].bar(copier_labels, copier_agree, color=[BLUE, ORANGE, TEAL], width=0.65)
    axes[1].set_ylabel("One-token agreement (%)")
    axes[1].set_title("Agreement vs Dense")
    axes[1].set_ylim(80, 100)
    for bar, height in zip(bars, copier_agree):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.4,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    _save(plt, fig, "fig_admit")

    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    labels = ["Planned\ncopies", "Observed\ncopies", "Fallbacks", "Source\npre-rotations"]
    heights = [
        values["planned"],
        values["copied"],
        values["fallback"],
        values["prerotate"],
    ]
    bars = ax.bar(labels, heights, color=["#a6c8e6", BLUE, GRAY, GREEN], width=0.65)
    ax.set_ylabel("Event count")
    ax.set_title("Fail-closed copy mechanics")
    ax.set_ylim(0, 2000)
    for bar, height in zip(bars, heights):
        y = height + 40 if height > 0 else 40
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            str(int(height)),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    _save(plt, fig, "fig_mechanism")

    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    labels = ["1 island", "2 islands", "3 islands"]
    counts = [values["island_groups"][1], values["island_groups"][2], values["island_groups"][3]]
    bars = ax.bar(labels, counts, color=[LIME, TEAL, BLUE], width=0.65)
    ax.set_ylabel("Target groups (n = 235)")
    ax.set_title("True-lossy islands per group")
    ax.set_ylim(0, 130)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 2,
            str(count),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    _save(plt, fig, "fig_coverage")

    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    bars = ax.bar(
        ["<3K", "3–5K", "5–7K", "≥7K"],
        [values["length"][k] for k in length_order],
        color=[LIME, TEAL, BLUE, NAVY],
        width=0.65,
    )
    ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Target prompt length")
    ax.set_ylabel("Cache-ready speedup vs Dense")
    ax.set_title("Cache-ready TTFT by prompt length")
    ax.set_ylim(0, 2.4)
    _annotate_bars(ax, bars)
    _save(plt, fig, "fig_length")

    print(
        "wrote 7B figures from frozen JSON",
        {
            "cache_ready": values["cache_ready"],
            "dense_p50": values["ttft"]["dense_p50"],
            "reuse_p50": values["ttft"]["reuse_p50"],
            "prefix_dual": values["prefix_on"]["dual"],
        },
    )


if __name__ == "__main__":
    main()
