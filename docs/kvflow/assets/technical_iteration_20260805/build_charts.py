#!/usr/bin/env python3
"""Build the evidence figures embedded by the 2026-08-05 iteration review."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "evidence_data.json").read_text())

BLUE = "#2563eb"
ORANGE = "#ea580c"
GREEN = "#15803d"
RED = "#b91c1c"
GRAY = "#6b7280"
LIGHT_GRAY = "#d1d5db"


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(HERE / name, bbox_inches="tight")
    plt.close(fig)


def label_bars(ax: plt.Axes, bars, fmt: str = "{:.1f}") -> None:
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def capacity_signal() -> None:
    rows = DATA["capacity_route"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))

    cap_rows = rows[:4]
    labels = [r["version"] for r in cap_rows]
    values = [r["eligible_capacity_pct"] for r in cap_rows]
    bars = axes[0].bar(labels, values, color=[RED, RED, GREEN, ORANGE])
    axes[0].axhline(20, color=GRAY, linestyle="--", linewidth=1.5, label="V9–V11 gate: 20%")
    axes[0].axhline(15, color=LIGHT_GRAY, linestyle=":", linewidth=2, label="V12 gate: 15%")
    axes[0].set_ylabel("Legally reusable prompt capacity (%)")
    axes[0].set_title("Capacity became feasible only after file-versioning")
    axes[0].legend(loc="upper left")
    axes[0].tick_params(axis="x", rotation=15)
    label_bars(axes[0], bars, "{:.2f}%")

    v12 = rows[3:]
    x = np.arange(len(v12))
    width = 0.34
    cap = axes[1].bar(x - width / 2, [r["eligible_capacity_pct"] for r in v12], width, label="Capacity", color=BLUE)
    harm = axes[1].bar(x + width / 2, [r["harm_reduction_pct"] for r in v12], width, label="Harm reduction", color=ORANGE)
    axes[1].axhline(15, color=BLUE, linestyle=":", linewidth=1.5)
    axes[1].axhline(30, color=ORANGE, linestyle="--", linewidth=1.5)
    axes[1].set_xticks(x, ["Near-capacity\nconfiguration", "Harm-safe\nconfiguration"])
    axes[1].set_ylabel("Percent (%)")
    axes[1].set_title("V12 had no joint feasible operating point")
    axes[1].legend(loc="upper left")
    label_bars(axes[1], cap, "{:.2f}%")
    label_bars(axes[1], harm, "{:.2f}%")
    save(fig, "01_capacity_and_signal_gates.png")


def p_series() -> None:
    rows = DATA["p_series_tail_delta"]
    labels = [r["version"].split()[0] for r in rows]
    values = [r["mean_nll_advantage_vs_tail"] for r in rows]
    colors = [GREEN if v > 0 else RED for v in values]
    fig, ax = plt.subplots(figsize=(12.8, 5.1))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0, color=GRAY, linewidth=1.2)
    ax.axhline(0.005, color=GRAY, linestyle="--", linewidth=1.2, label="P23 oracle ceiling gate (+0.005)")
    ax.set_ylabel("Mean NLL advantage over equal-cost tail\n(positive is better)")
    ax.set_title("Static AST/symbol repair repeatedly failed to establish a robust tail advantage")
    ax.legend(loc="lower left")
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:+.4f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5 if value >= 0 else -14),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    ax.set_ylim(min(values) - 0.0012, 0.006)
    save(fig, "02_pseries_vs_tail_nll.png")


def p27_reversal() -> None:
    rows = DATA["p27_generalization"]
    labels = ["Development\n8 cases", "Independent\n17 cases"]
    x = np.arange(2)
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))

    pipeline = [r["pipeline_vs_full_tail_nll"] for r in rows]
    context = [r["capsule_dense_vs_full_dense_nll"] for r in rows]
    b1 = axes[0].bar(x - width / 2, pipeline, width, color=BLUE, label="Final pipeline vs full tail")
    b2 = axes[0].bar(x + width / 2, context, width, color=ORANGE, label="Capsule Dense vs Full Dense")
    axes[0].axhline(0, color=GRAY, linewidth=1.2)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Mean NLL advantage (positive is better)")
    axes[0].set_title("P27's development win reversed out of sample")
    axes[0].legend(loc="lower left")
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            axes[0].annotate(f"{v:+.5f}", (bar.get_x()+bar.get_width()/2, v), xytext=(0, 4 if v >= 0 else -14), textcoords="offset points", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)

    stale = [r["stale_kv_loss"] for r in rows]
    speed = [r["prefill_reduction_pct"] for r in rows]
    b3 = axes[1].bar(x - width / 2, [v * 1000 for v in stale], width, color=ORANGE, label="Stale-KV loss ×1000")
    ax_speed = axes[1].twinx()
    b4 = ax_speed.bar(x + width / 2, speed, width, color=GREEN, alpha=0.76, label="Prefill reduction (%)")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Stale-KV loss ×1000")
    ax_speed.set_ylabel("Prefill reduction (%)")
    axes[1].set_ylim(0, 5.5)
    ax_speed.set_ylim(0, 30)
    axes[1].set_title("Mechanism stayed efficient; context selection failed")
    axes[1].legend([b3, b4], [b3.get_label(), b4.get_label()], loc="upper left")
    label_bars(axes[1], b3, "{:.2f}")
    for bar in b4:
        ax_speed.annotate(f"{bar.get_height():.2f}%", (bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    save(fig, "03_p27_generalization_reversal.png")


def observation_route() -> None:
    bridge = DATA["bridge_formal_18"]
    v44 = DATA["v44_official_12"]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.4), sharex="col")

    labels = [r["method"] for r in bridge]
    resolved = [r["resolved"] for r in bridge]
    ttft_speed = [bridge[0]["median_ttft_ms"] / r["median_ttft_ms"] for r in bridge]
    x = np.arange(len(labels))
    b1 = axes[0, 0].bar(x, resolved, color=BLUE)
    axes[0, 0].set_ylabel("Official resolved / 18")
    axes[0, 0].set_title("First matched bridge: no coding accuracy win")
    label_bars(axes[0, 0], b1, "{:.0f}")
    b2 = axes[1, 0].bar(x, ttft_speed, color=GREEN)
    axes[1, 0].axhline(1.0, color=GRAY, linewidth=1.2)
    axes[1, 0].set_ylabel("Median TTFT speedup")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylim(0, 1.62)
    label_bars(axes[1, 0], b2, "{:.2f}x")

    labels2 = [r["method"] for r in v44]
    resolved2 = [r["resolved"] for r in v44]
    copied = [r["copied_tokens"] / 1000 for r in v44]
    x2 = np.arange(len(labels2))
    b3 = axes[0, 1].bar(x2, resolved2, color=BLUE)
    axes[0, 1].set_ylabel("Official resolved / 12")
    axes[0, 1].set_title("V40: one positive task-level signal")
    label_bars(axes[0, 1], b3, "{:.0f}")
    b4 = axes[1, 1].bar(x2, copied, color=ORANGE)
    axes[1, 1].set_ylabel("Copied tokens (thousand)")
    axes[1, 1].set_xticks(x2, labels2)
    for bar in b4:
        axes[1, 1].annotate(f"{bar.get_height():.0f}k", (bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Grounded-observation reuse reduced exposure, but evidence stayed small", fontsize=16)
    save(fig, "04_observation_route_evidence.png")


def trajectory_progression() -> None:
    rows = DATA["trajectory_official_progression"]
    labels = [r["cohort"] for r in rows]
    x = np.arange(len(rows))
    width = 0.25
    dense = [np.nan if r["dense"] is None else 100 * r["dense"] / r["cases"] for r in rows]
    general = [100 * r["general"] / r["cases"] for r in rows]
    candidate = [100 * r["candidate"] / r["cases"] for r in rows]
    fig, ax = plt.subplots(figsize=(12.8, 5.1))
    b1 = ax.bar(x - width, dense, width, color=GRAY, label="Dense")
    b2 = ax.bar(x, general, width, color=ORANGE, label="General")
    b3 = ax.bar(x + width, candidate, width, color=BLUE, label="Coding candidate")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Official resolved rate (%)")
    ax.set_title("Trajectory guards did not produce a stable advantage until V40's small development signal")
    ax.legend(loc="upper left")
    for bars, key in ((b1, "dense"), (b2, "general"), (b3, "candidate")):
        for bar, row in zip(bars, rows):
            value = row[key]
            if value is None:
                continue
            ax.annotate(f"{value}/{row['cases']}", (bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax.text(0.99, 0.96, "Cohorts differ and are small; compare decisions within each group,\nnot bar heights across groups.", transform=ax.transAxes, ha="right", va="top", fontsize=10, color=GRAY)
    save(fig, "09_trajectory_guard_progression.png")


def controlled_dev() -> None:
    rows = DATA["controlled_dev50"]
    labels = [r["method"].replace(" ", "\n", 1) for r in rows]
    passes = [r["passes"] for r in rows]
    speed = [r["speedup"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.05, 1]})
    colors = [GRAY, ORANGE, RED, BLUE, BLUE, ORANGE, GREEN]
    bars = axes[0].bar(x, passes, color=colors)
    axes[0].axhline(12, color=GRAY, linestyle="--", linewidth=1.2, label="Dense / CacheBlend: 12")
    axes[0].set_ylabel("Official DS-1000 passes / 50")
    axes[0].set_title("CacheBlend-derived route: quality rose only after task-conditioned layer routing")
    axes[0].legend(loc="upper left")
    label_bars(axes[0], bars, "{:.0f}")
    bars2 = axes[1].bar(x, speed, color=colors)
    axes[1].axhline(1.0, color=GRAY, linewidth=1.2)
    axes[1].set_ylabel("Cache-ready speedup vs same-engine Dense")
    axes[1].set_xticks(x, labels)
    axes[1].tick_params(axis="x", labelsize=9)
    axes[1].set_ylim(0.95, 1.16)
    label_bars(axes[1], bars2, "{:.3f}x")
    save(fig, "05_controlled_route_development.png")


def sealed_results() -> None:
    v90 = DATA["sealed_v90_100"]
    v92 = DATA["fresh_v92_100"]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0))

    def panel(rows, ax_acc, ax_speed, title):
        labels = [r["method"] for r in rows]
        x = np.arange(len(rows))
        colors = [GRAY if r["method"] == "Dense" else ORANGE if "CacheBlend" in r["method"] or "KVCOMM" in r["method"] else BLUE for r in rows]
        b1 = ax_acc.bar(x, [r["passes"] for r in rows], color=colors)
        ax_acc.set_xticks(x, labels, rotation=15)
        ax_acc.set_ylabel("Official passes / 100")
        ax_acc.set_title(title + ": execution accuracy")
        label_bars(ax_acc, b1, "{:.0f}")
        b2 = ax_speed.bar(x, [r["normalized_cache_ready_speedup"] for r in rows], color=colors)
        ax_speed.axhline(1.0, color=GRAY, linewidth=1.2)
        ax_speed.set_xticks(x, labels, rotation=15)
        ax_speed.set_ylabel("Cache-ready speedup vs own Dense")
        ax_speed.set_title(title + ": normalized speed")
        label_bars(ax_speed, b2, "{:.3f}x")

    panel(v90, axes[0, 0], axes[0, 1], "V90 sealed split")
    panel(v92, axes[1, 0], axes[1, 1], "V92 fresh split")
    save(fig, "06_fair_prompt_accuracy_speed.png")


def build_break_even() -> None:
    rows = DATA["build_break_even"]
    labels = [r["method"] for r in rows]
    values = [r["minimum_reuses"] for r in rows]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bars = ax.bar(labels, values, color=[ORANGE, BLUE, BLUE, GREEN])
    ax.set_ylabel("Minimum target reuses needed to beat Dense")
    ax.set_title("Cache-ready wins did not solve sequential source-build cost")
    label_bars(ax, bars, "{:.0f}")
    ax.text(0.99, 0.96, "Values come from different frozen DS-1000 splits;\nuse as deployment diagnostics, not method ranking.", transform=ax.transAxes, ha="right", va="top", fontsize=10, color=GRAY)
    save(fig, "07_source_build_break_even.png")


def current_sglang() -> None:
    rows = DATA["sglang_static50"]
    official = DATA["v46_official_preservation"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    labels = [r["method"] for r in rows]
    x = np.arange(len(rows))
    width = 0.34
    b1 = axes[0].bar(x - width / 2, [r["cache_ready_speedup"] for r in rows], width, color=BLUE, label="Cache-ready")
    b2 = axes[0].bar(x + width / 2, [r["n4_build_speedup"] for r in rows], width, color=ORANGE, label="N=4 incl. build")
    axes[0].axhline(1.0, color=GRAY, linewidth=1.2)
    axes[0].set_xticks(x, labels, rotation=12)
    axes[0].set_ylabel("Speedup vs native Dense")
    axes[0].set_title("V46 increased SGLang copy opportunity and speed")
    axes[0].legend(loc="upper left")
    label_bars(axes[0], b1, "{:.3f}x")
    label_bars(axes[0], b2, "{:.3f}x")

    labels2 = [r["method"] for r in official]
    bars = axes[1].bar(labels2, [r["resolved"] for r in official], color=[GRAY, BLUE, ORANGE, RED])
    axes[1].set_ylim(0, 3.7)
    axes[1].set_ylabel("Official SWE-bench resolved / 3")
    axes[1].set_title("But V46 failed the known-pass preservation gate")
    label_bars(axes[1], bars, "{:.0f}")
    axes[0].text(0.01, -0.26, "* CacheBlend uses its native Dense normalization; next-line quality is not task accuracy.", transform=axes[0].transAxes, fontsize=9, color=GRAY)
    save(fig, "08_current_sglang_v46_tradeoff.png")


def coding_motivation_gates() -> None:
    rows = DATA["coding_motivation_gates"]
    labels = [row["experiment"] for row in rows]
    values = [row["observed_pct"] for row in rows]
    gates = [row["gate_pct"] for row in rows]
    colors = [GREEN if row["passed"] else RED for row in rows]
    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    x = np.arange(len(rows))
    bars = ax.bar(x, values, color=colors)
    ax.scatter(x, gates, marker="_", s=650, linewidths=3, color=GRAY, label="Frozen gate")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Pair-direction consistency / accuracy (%)")
    ax.set_title("Only coding path dependency passed the frozen consistency gates")
    ax.legend(loc="upper left")
    label_bars(ax, bars, "{:.1f}%")
    ax.text(
        0.99,
        0.96,
        "Green means the preregistered direction gate passed;\nM50/M51/M54 must not be promoted as positive motivation.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=GRAY,
    )
    save(fig, "10_coding_motivation_gates.png")


def path_dependency() -> None:
    rows = DATA["path_dependency"]
    labels = [row["cohort"].replace(" ", "\n", 1) for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.1))

    relevant = axes[0].bar(
        x - width / 2,
        [row["relevant_attention_mean"] for row in rows],
        width,
        color=BLUE,
        label="Path relevant",
    )
    disjoint = axes[0].bar(
        x + width / 2,
        [row["disjoint_attention_mean"] for row in rows],
        width,
        color=GRAY,
        label="Path disjoint",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Mean target-query attention")
    axes[0].set_title("Path overlap repeatedly predicted model dependency")
    axes[0].legend(loc="upper right")
    label_bars(axes[0], relevant, "{:.3f}")
    label_bars(axes[0], disjoint, "{:.3f}")

    ratio_width = 0.24
    attention = axes[1].bar(
        x - ratio_width,
        [row["attention_adjusted_ratio"] for row in rows],
        ratio_width,
        color=GREEN,
        label="Attention ratio",
    )
    drift = axes[1].bar(
        x,
        [row["drift_adjusted_ratio"] for row in rows],
        ratio_width,
        color=ORANGE,
        label="K/V drift ratio",
    )
    js = axes[1].bar(
        x + ratio_width,
        [row["js_adjusted_ratio"] for row in rows],
        ratio_width,
        color=BLUE,
        label="Splice-JS ratio",
    )
    axes[1].axhline(1.0, color=GRAY, linewidth=1.2)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Path-relevant / path-disjoint adjusted ratio")
    axes[1].set_title("Dependency replicated; safety did not replicate consistently")
    axes[1].legend(loc="upper right")
    for bars in (attention, drift, js):
        label_bars(axes[1], bars, "{:.2f}x")
    save(fig, "11_path_dependency_evidence.png")


def fresh_task_accuracy_and_exposure() -> None:
    rows = DATA["m55_fresh13"]
    labels = [row["method"] for row in rows]
    colors = [GRAY, ORANGE, BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))

    resolved = axes[0].bar(
        labels, [row["resolved"] for row in rows], color=colors
    )
    axes[0].set_ylim(0, max(1.6, max(row["cases"] for row in rows) * 0.18))
    axes[0].set_ylabel("Official SWE-bench resolved")
    axes[0].set_title("Fresh-13 accuracy had no identifying power")
    for bar, row in zip(resolved, rows, strict=True):
        axes[0].annotate(
            f"{row['resolved']}/{row['cases']}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[0].text(
        0.5,
        0.73,
        "All arms scored zero; equality is not evidence\nof equal population accuracy.",
        transform=axes[0].transAxes,
        ha="center",
        va="center",
        color=RED,
        fontsize=11,
    )

    copied = axes[1].bar(
        labels,
        [row["copied_tokens"] / 1000 for row in rows],
        color=colors,
    )
    axes[1].set_ylabel("Physically copied tokens (thousand)")
    axes[1].set_title("The run still verifies different reuse exposure")
    axes[1].set_ylim(
        0, max(row["copied_tokens"] / 1000 for row in rows) * 1.14
    )
    for bar, row in zip(copied, rows, strict=True):
        axes[1].annotate(
            f"{bar.get_height():.1f}k\n{row['copy_requests']} requests",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save(fig, "12_m55_fresh_accuracy_exposure.png")


def fresh_selector_capacity() -> None:
    row = DATA["m55_selector_capacity"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))

    labels = ["Path pairs\nbefore guard", "Version-valid\npairs"]
    bars = axes[0].bar(
        labels,
        [row["pairs_before_version_guard"], row["version_valid_pairs"]],
        color=[ORANGE, BLUE],
    )
    axes[0].set_ylabel("Candidate request pairs")
    axes[0].set_title("Version validity was not the capacity bottleneck")
    label_bars(axes[0], bars, "{:.0f}")

    labels2 = ["Cases", "Tasks"]
    observed = [row["cases"], row["tasks"]]
    gates = [row["minimum_cases"], row["minimum_tasks"]]
    x = np.arange(2)
    bars2 = axes[1].bar(x, observed, color=[GREEN, RED], label="Observed")
    axes[1].scatter(
        x, gates, marker="_", s=750, linewidths=3, color=GRAY, label="Frozen minimum"
    )
    axes[1].set_xticks(x, labels2)
    axes[1].set_ylabel("Count")
    axes[1].set_title("M55 stopped before opening GPU causal labels")
    axes[1].legend(loc="upper right")
    label_bars(axes[1], bars2, "{:.0f}")
    for index, gate in enumerate(gates):
        axes[1].annotate(
            f"gate {gate}",
            (index, gate),
            xytext=(9, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=GRAY,
            fontsize=9,
        )
    save(fig, "13_m55_selector_capacity.png")


def same_prompt_speed() -> None:
    row = DATA["m56_same_prompt"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))

    labels = ["Dense", "V40\ncache-ready", "V40 N=4\nincl. build"]
    speeds = [1.0, row["cache_ready_speedup"], row["n4_speedup"]]
    bars = axes[0].bar(labels, speeds, color=[GRAY, BLUE, ORANGE])
    axes[0].axhline(1.0, color=GRAY, linewidth=1.2)
    axes[0].axhline(
        row["cache_ready_gate"],
        color=LIGHT_GRAY,
        linestyle="--",
        linewidth=1.5,
        label="Frozen cache-ready gate",
    )
    axes[0].set_ylabel("Median TTFT speedup vs Dense")
    axes[0].set_title("Exact-same-prompt V40 speed mechanism")
    axes[0].legend(loc="lower right")
    label_bars(axes[0], bars, "{:.3f}x")

    agreement = 100 * row["first_token_agreement"]
    agreement_gate = 100 * row["first_token_agreement_gate"]
    bar = axes[1].bar(["V40 target requests"], [agreement], color=BLUE)
    axes[1].axhline(
        agreement_gate,
        color=GRAY,
        linestyle="--",
        linewidth=1.5,
        label="Frozen agreement gate",
    )
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("First-token agreement with Dense (%)")
    axes[1].set_title(
        f"{row['target_requests']} copied requests / {row['target_tasks']} tasks"
    )
    axes[1].legend(loc="lower right")
    label_bars(axes[1], bar, "{:.1f}%")
    save(fig, "14_m56_same_prompt_speed.png")


def main() -> None:
    setup()
    capacity_signal()
    p_series()
    p27_reversal()
    observation_route()
    trajectory_progression()
    controlled_dev()
    sealed_results()
    build_break_even()
    current_sglang()
    coding_motivation_gates()
    path_dependency()
    fresh_task_accuracy_and_exposure()
    fresh_selector_capacity()
    same_prompt_speed()


if __name__ == "__main__":
    main()
