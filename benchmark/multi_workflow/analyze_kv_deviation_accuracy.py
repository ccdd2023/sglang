#!/usr/bin/env python3
"""Audit whether K/V-deviation proxies predict coding-task accuracy.

This script is intentionally read-only with respect to frozen experiment
artifacts.  It joins the registered proxy measurements with their official
execution outcomes and writes a compact, reproducible evidence bundle for the
corresponding Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import binomtest, fisher_exact, rankdata, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT.parents[1] / "kvflow-artifacts"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "docs/kvflow/assets/kv_deviation_accuracy_20260806"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [math.nan, math.nan]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [center - half_width, center + half_width]


def binary_auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    label_array = np.asarray(list(labels), dtype=np.int64)
    score_array = np.asarray(list(scores), dtype=np.float64)
    positives = int(label_array.sum())
    negatives = int(label_array.size - positives)
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = rankdata(score_array)
    rank_sum = float(ranks[label_array == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def bootstrap_auc(
    labels: list[int],
    scores: list[float],
    *,
    samples: int = 20_000,
    seed: int = 20_260_806,
) -> list[float]:
    rng = np.random.default_rng(seed)
    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    estimates: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(labels), len(labels))
        estimate = binary_auc(label_array[indices], score_array[indices])
        if math.isfinite(estimate):
            estimates.append(estimate)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def evaluation_map(path: Path) -> dict[str, int]:
    return {
        row["case_id"]: int(row["execution_passed"])
        for row in load_json(path)["records"]
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o644)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 180,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def plot_current_method_pipeline(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(14.0, 6.4))
    axis.set_xlim(0, 14.0)
    axis.set_ylim(0, 6.4)
    axis.axis("off")

    steps = [
        (0.30, "Agent history", "Repeated read/search\ntool observations", "#D9EAF7"),
        (3.05, "Admit candidates", "Successful + read-only\n+ path-localized", "#D9EAF7"),
        (5.80, "Validity gate", "Reject changed files\nor ambiguous scope", "#FDE2A7"),
        (8.55, "Current selector", "One conservative island\nor ≤3 longest/recent", "#FDE2A7"),
        (11.30, "SGLang executor", "Dense gaps + copied V\n+ RoPE-shifted K", "#D8EAD1"),
    ]
    box_width = 2.25
    box_height = 1.55
    for x, title, detail, color in steps:
        patch = FancyBboxPatch(
            (x, 3.65),
            box_width,
            box_height,
            boxstyle="round,pad=0.025,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#555555",
            facecolor=color,
        )
        axis.add_patch(patch)
        axis.text(
            x + box_width / 2,
            4.75,
            title,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#222222",
        )
        axis.text(
            x + box_width / 2,
            4.18,
            detail,
            ha="center",
            va="center",
            fontsize=8.8,
            color="#333333",
        )
    for index in range(len(steps) - 1):
        start = steps[index][0] + box_width
        end = steps[index + 1][0]
        axis.add_patch(
            FancyArrowPatch(
                (start + 0.05, 4.42),
                (end - 0.05, 4.42),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4,
                color="#666666",
            )
        )

    axis.text(
        0.30,
        5.92,
        "Current coding-aware lossy KV reuse: provenance decides eligibility; SGLang performs the physical copy",
        ha="left",
        va="center",
        fontsize=14.5,
        fontweight="bold",
    )

    lossy = FancyBboxPatch(
        (0.30, 0.45),
        6.45,
        1.85,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.2,
        edgecolor="#E45756",
        facecolor="#FCE5E4",
    )
    axis.add_patch(lossy)
    axis.text(
        0.62,
        1.86,
        "Why it is lossy",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    axis.text(
        0.62,
        1.20,
        "The token span x is identical, but its cache came from an older prefix.\n"
        "KV(source prefix, x) ≠ KV(target prefix, x).\n"
        "RoPE fixes absolute position—not the missing target-prefix context.",
        ha="left",
        va="center",
        fontsize=9.2,
        color="#333333",
    )

    next_box = FancyBboxPatch(
        (7.15, 0.45),
        6.50,
        1.85,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.3,
        linestyle="--",
        edgecolor="#4C78A8",
        facecolor="#E5EEF8",
    )
    axis.add_patch(next_box)
    axis.text(
        7.47,
        1.86,
        "Research target—not yet the deployed selector",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    axis.text(
        7.47,
        1.20,
        "Validity gate → reject high contextual-risk candidates →\n"
        "rank survivors by path/action utility under a fixed copy budget.\n"
        "This two-stage selector is a research target, not a deployed claim.",
        ha="left",
        va="center",
        fontsize=9.2,
        color="#333333",
    )

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    output.chmod(0o644)


def plot_quartiles(rows: list[dict[str, Any]], output: Path) -> None:
    arm_names = [
        ("fixed_route", "Fixed route"),
        ("stale_guard", "Stale-mass guard"),
        ("dense", "Dense"),
    ]
    x = np.arange(4)
    width = 0.24
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    colors = ["#4C78A8", "#F58518", "#79706E"]
    for index, (key, label) in enumerate(arm_names):
        values = [
            100.0 * row[f"{key}_passes"] / row["cases"] for row in rows
        ]
        positions = x + (index - 1) * width
        bars = axis.bar(positions, values, width, label=label, color=colors[index])
        axis.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    axis.set_xticks(x, ["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"])
    axis.set_xlabel("Residual V-difference mass quartile (18 requests each)")
    axis.set_ylabel("Official DS-1000 execution pass rate")
    axis.set_ylim(0, 60)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=3, loc="upper left")
    axis.set_title("Lower residual V-difference mass did not predict higher accuracy")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    output.chmod(0o644)


def plot_density_sweep(rows: list[dict[str, Any]], output: Path) -> None:
    ratios = [100.0 * row["recompute_ratio"] for row in rows]
    stale = [row["minimum_stale_tokens"] for row in rows]
    exact = [row["exact_line_count"] for row in rows]
    fig, (upper, lower) = plt.subplots(
        2,
        1,
        figsize=(8.4, 6.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"hspace": 0.12},
    )
    upper.plot(ratios, stale, marker="o", linewidth=2.0, color="#4C78A8")
    for x, y in zip(ratios, stale):
        upper.annotate(str(y), (x, y), xytext=(0, 7), textcoords="offset points", ha="center")
    upper.set_ylabel("Minimum stale tokens")
    upper.set_title("Nested top-V-difference repair leaves less stale KV")
    upper.grid(axis="y", alpha=0.22)

    lower.plot(ratios, exact, marker="o", linewidth=2.0, color="#E45756")
    for x, y in zip(ratios, exact):
        lower.annotate(str(y), (x, y), xytext=(0, 7), textcoords="offset points", ha="center")
    lower.set_xlabel("Recomputed token ratio (%)")
    lower.set_ylabel("Exact next-line hits / 100")
    lower.set_xticks(ratios)
    lower.set_ylim(min(exact) - 2, max(exact) + 2)
    lower.grid(axis="y", alpha=0.22)
    lower.set_title("But next-line accuracy is non-monotonic")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    output.chmod(0o644)


def plot_functional_counterexample(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    for row, color in zip(rows, ["#4C78A8", "#E45756"]):
        axis.scatter(
            100.0 * row["mean_stale_fraction"],
            row["execution_passes"],
            s=120,
            color=color,
            zorder=3,
        )
        axis.annotate(
            f"{row['label']}\n{row['execution_passes']}/50 pass\n"
            f"{row['same_as_dense_outputs']}/50 Dense-identical",
            (100.0 * row["mean_stale_fraction"], row["execution_passes"]),
            xytext=(10, 7),
            textcoords="offset points",
        )
    axis.annotate(
        "",
        xy=(100.0 * rows[1]["mean_stale_fraction"], rows[1]["execution_passes"]),
        xytext=(100.0 * rows[0]["mean_stale_fraction"], rows[0]["execution_passes"]),
        arrowprops={"arrowstyle": "->", "linewidth": 1.8, "color": "#79706E"},
    )
    axis.axhline(12, linestyle="--", linewidth=1.3, color="#79706E", label="Dense: 12/50")
    axis.set_xlabel("Mean stale-token fraction (%)")
    axis.set_ylabel("Official DS-1000 execution passes / 50")
    axis.set_ylim(8.5, 12.8)
    axis.invert_xaxis()
    axis.grid(alpha=0.22)
    axis.legend(loc="lower left")
    axis.set_title("2.5x less stale KV, but one fewer functional pass")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    output.chmod(0o644)


def plot_triggered_changes(counts: dict[str, int], output: Path) -> None:
    categories = ["Rescued", "Damaged", "Unchanged"]
    values = [counts["rescues"], counts["damages"], counts["unchanged"]]
    colors = ["#54A24B", "#E45756", "#BAB0AC"]
    fig, axis = plt.subplots(figsize=(8.4, 2.8))
    left = 0
    for category, value, color in zip(categories, values, colors):
        axis.barh([0], [value], left=left, color=color, label=category)
        axis.text(left + value / 2, 0, str(value), ha="center", va="center", fontsize=11)
        left += value
    axis.set_xlim(0, sum(values))
    axis.set_yticks([])
    axis.set_xlabel("High-risk requests routed to stronger repair")
    axis.set_title("Stronger repair changed official outcomes in both directions")
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    output.chmod(0o644)


def plot_failure_auc(predictor_audit: dict[str, Any], output: Path) -> None:
    rows = [
        ("Fixed route", predictor_audit["fixed_route"]),
        ("Stale-mass guard", predictor_audit["stale_guard"]),
        ("Dense", predictor_audit["dense"]),
    ]
    fig, axis = plt.subplots(figsize=(8.6, 4.2))
    y = np.arange(len(rows))[::-1]
    colors = ["#4C78A8", "#F58518", "#79706E"]
    for position, (label, row), color in zip(y, rows, colors):
        point = row["failure_auc_using_stale_mass"]
        low, high = row["failure_auc_bootstrap95"]
        axis.errorbar(
            point,
            position,
            xerr=[[point - low], [high - point]],
            fmt="o",
            markersize=8,
            capsize=5,
            linewidth=2,
            color=color,
        )
        axis.text(high + 0.012, position, f"{point:.3f}  [{low:.3f}, {high:.3f}]", va="center")
    axis.axvline(0.5, linestyle="--", linewidth=1.5, color="#79706E")
    axis.text(0.505, 2.28, "random classifier", color="#79706E", va="center")
    axis.set_yticks(y, [label for label, _ in rows])
    axis.set_ylim(-0.5, 2.5)
    axis.set_xlim(0.10, 0.67)
    axis.set_xlabel("AUROC when higher residual V-mass predicts task failure")
    axis.set_title("Residual V-mass is not a calibrated failure score")
    axis.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    output.chmod(0o644)


def plot_functional_transition(matrix: list[list[int]], output: Path) -> None:
    values = np.asarray(matrix, dtype=np.int64)
    fig, axis = plt.subplots(figsize=(6.6, 5.2))
    image = axis.imshow(values, cmap="Blues", vmin=0, vmax=int(values.max()))
    for row in range(2):
        for column in range(2):
            value = int(values[row, column])
            color = "white" if value > values.max() / 2 else "black"
            label = ""
            if row == 0 and column == 1:
                label = "\nrescue"
            elif row == 1 and column == 0:
                label = "\ndamage"
            axis.text(column, row, f"{value}{label}", ha="center", va="center", color=color, fontsize=13)
    axis.set_xticks([0, 1], ["Fail", "Pass"])
    axis.set_yticks([0, 1], ["Fail", "Pass"])
    axis.set_xlabel("Repair 90% official outcome")
    axis.set_ylabel("Repair 75% official outcome")
    axis.set_title("Paired DS-1000 transitions: 1 rescue, 2 damages")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.78)
    colorbar.set_label("Cases")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    output.chmod(0o644)


def plot_proxy_scope(proxy_scope: dict[str, Any], output: Path) -> None:
    rows = [
        (
            "Single-island KV drift → causal logit JS",
            proxy_scope["single_island_kv_drift_vs_causal_js_spearman"],
            "#4C78A8",
        ),
        (
            "Request KV drift → code-similarity change",
            proxy_scope["request_kv_drift_vs_code_similarity_change_spearman"],
            "#F58518",
        ),
        (
            "Request KV drift → composed NLL",
            proxy_scope["request_kv_drift_vs_composed_nll_spearman"],
            "#F58518",
        ),
        (
            "16-token probe → composed NLL",
            proxy_scope["short_probe_request_composed_nll_spearman"],
            "#BAB0AC",
        ),
        (
            "Dense-target drift → NLL repair utility",
            proxy_scope["dense_target_drift_vs_nll_repair_utility_spearman"],
            "#E45756",
        ),
    ]
    y = np.arange(len(rows))[::-1]
    fig, axis = plt.subplots(figsize=(10.0, 5.1))
    for position, (label, value, color) in zip(y, rows):
        axis.hlines(position, 0, value, color=color, linewidth=3)
        axis.scatter(value, position, s=90, color=color, zorder=3)
        axis.text(value + 0.012, position, f"{value:.3f}", va="center")
    axis.set_yticks(y, [label for label, _, _ in rows])
    axis.set_xlim(0, 0.61)
    axis.set_xlabel("Spearman correlation")
    axis.set_title("K/V drift is strongest for local perturbation, weak for request-level outcomes")
    axis.grid(axis="x", alpha=0.22)
    fig.text(
        0.99,
        0.01,
        "Different frozen cohorts and outcomes; descriptive scope audit, not a pooled estimator.",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#79706E",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    output.chmod(0o644)


def plot_nll_generalization(capsule_nll: dict[str, Any], output: Path) -> None:
    labels = ["Development\n8 cases", "Independent\n17 cases"]
    rows = [capsule_nll["development"], capsule_nll["independent"]]
    advantage = [row["pipeline_nll_improvement"] for row in rows]
    win_rate = [100.0 * row["wins"] / row["cases"] for row in rows]
    colors = ["#54A24B", "#E45756"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.7))

    bars = left.bar(labels, advantage, color=colors)
    left.axhline(0, linewidth=1.2, color="#79706E")
    left.set_ylabel("Pipeline NLL advantage (positive is better)")
    left.set_ylim(-0.0087, 0.0278)
    left.set_title("Mean NLL advantage reverses")
    left.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, advantage):
        left.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.0007 if value >= 0 else -0.0007),
            f"{value:+.5f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )

    bars = right.bar(labels, win_rate, color=colors)
    right.set_ylabel("Wins versus full-tail (%)")
    right.set_ylim(0, 100)
    right.set_title("Win rate does not generalize")
    right.grid(axis="y", alpha=0.22)
    for bar, value, row in zip(bars, win_rate, rows):
        right.text(
            bar.get_x() + bar.get_width() / 2,
            value + 3,
            f"{value:.1f}%\n{row['severe_losses']} severe loss(es)",
            ha="center",
            va="bottom",
        )
    fig.suptitle(
        "Low mean stale-KV NLL loss did not guarantee independent quality",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    output.chmod(0o644)


def plot_distance_selector_counterexample(
    distance_selector: dict[str, Any], output: Path
) -> None:
    accuracy = distance_selector["independent_exact_line"]
    agreement = distance_selector["transition_agreement_probe"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.8))

    labels = ["Dense", "Generic\nV-difference", "Semantic +\ndistance gate"]
    values = [accuracy["dense"], accuracy["generic_value_diff"], accuracy["distance_consensus"]]
    bars = left.bar(labels, values, color=["#79706E", "#4C78A8", "#F58518"])
    left.bar_label(bars, labels=[f"{value}/200" for value in values], padding=3)
    left.set_ylim(0, 67)
    left.set_ylabel("Exact next-line matches / 200")
    left.set_title("Independent fixed-budget comparison")
    left.grid(axis="y", alpha=0.22)
    left.text(
        1.5,
        64.0,
        f"{accuracy['rescues']} rescues / {accuracy['damages']} damages",
        ha="center",
        va="center",
        fontsize=10,
        color="#79706E",
    )

    features = ["K top-k agreement", "V top-k agreement"]
    deltas = [
        100.0 * agreement["key"]["rescue_minus_damage"],
        100.0 * agreement["value"]["rescue_minus_damage"],
    ]
    aucs = [agreement["key"]["auc"], agreement["value"]["auc"]]
    y = np.arange(2)[::-1]
    bars = right.barh(y, deltas, color=["#E45756", "#E45756"])
    right.axvline(0, linewidth=1.3, color="#79706E")
    right.set_yticks(
        y,
        [f"{feature}\nrescue AUC {auc:.3f}" for feature, auc in zip(features, aucs)],
    )
    right.set_xlim(-1.65, 0.35)
    right.set_xlabel("Rescue − damage mean agreement (pp)")
    right.set_title("Agreement was slightly higher for damages")
    right.grid(axis="x", alpha=0.22)
    for bar, value in zip(bars, deltas):
        right.text(
            value / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f} pp",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
        )

    fig.suptitle("Distance-aware consensus did not improve exact-line accuracy", y=0.99)
    fig.text(
        0.5,
        0.01,
        "RepoBench-P; same prompt/token IDs and 75% recomputation budget. Exact-line is not functional execution accuracy.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#79706E",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    output.chmod(0o644)


def plot_related_work_evidence_matrix(
    rows: list[dict[str, Any]], output: Path
) -> None:
    columns = [
        ("internal_proxy", "Internal KV /\nattention proxy"),
        ("nll_ppl", "NLL / PPL"),
        ("output_metric", "F1 / ROUGE /\ntext metric"),
        ("task_success", "Accuracy / Pass@1 /\nexecution"),
        ("human_check", "Human\ncorrectness"),
        ("systems", "TTFT / reuse /\ntransfer cost"),
    ]
    values = np.asarray(
        [[int(row[key]) for key, _ in columns] for row in rows],
        dtype=np.int64,
    )
    colors = plt.matplotlib.colors.ListedColormap(["#F1F1F1", "#F2CF5B", "#59A14F"])
    fig, axis = plt.subplots(figsize=(11.0, 5.3))
    axis.imshow(values, cmap=colors, vmin=-0.5, vmax=2.5, aspect="auto")
    axis.set_xticks(np.arange(len(columns)), [label for _, label in columns])
    axis.set_yticks(np.arange(len(rows)), [row["work"] for row in rows])
    axis.tick_params(axis="x", rotation=0, labelsize=9.5)
    axis.tick_params(axis="y", labelsize=10.5)
    labels = {0: "—", 1: "Proxy", 2: "Downstream"}
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = int(values[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                labels[value],
                ha="center",
                va="center",
                fontsize=9.2,
                color="white" if value == 2 else "#333333",
                fontweight="bold" if value else "normal",
            )
    axis.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_title(
        "Lossy-KV papers use internal distance as a proxy, then validate downstream quality"
    )
    fig.text(
        0.5,
        0.015,
        "Proxy = mechanism, selection, or language-model diagnostic; Downstream = reported output/task/user outcome. "
        "Coverage only—scores across papers are not comparable.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    output.chmod(0o644)


def plot_kvcomm_distance_ablation(
    kvcomm: dict[str, Any], output: Path
) -> None:
    offset_rows = kvcomm["offset_approximation_humaneval_4agent"]
    matching_rows = kvcomm["matching_criterion_mmlu_4agent"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 5.0))

    offset_labels = [
        "Nearest\nanchor",
        "Cosine-weighted\noffsets",
        "L2-weighted\noffsets",
        "Dense /\nOriginal",
    ]
    offset_values = [float(row["accuracy_percent"]) for row in offset_rows]
    offset_colors = ["#E45756", "#4C78A8", "#59A14F", "#79706E"]
    bars = left.bar(offset_labels, offset_values, color=offset_colors)
    left.bar_label(bars, labels=[f"{value:.2f}%" for value in offset_values], padding=3)
    left.set_ylim(0, 94)
    left.set_ylabel("HumanEval Pass@1 / accuracy")
    left.set_title("Nearest anchor was not the safest approximation")
    left.grid(axis="y", alpha=0.22)
    for index, row in enumerate(offset_rows[:-1]):
        left.text(
            index,
            4,
            f"reuse {row['reuse_percent']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="white",
            rotation=90,
        )

    labels = ["Length only", "Length +\nembedding distance"]
    x = np.arange(len(labels))
    width = 0.34
    accuracy = [float(row["accuracy_percent"]) for row in matching_rows]
    reuse = [float(row["reuse_percent"]) for row in matching_rows]
    accuracy_bars = right.bar(
        x - width / 2, accuracy, width, label="Task accuracy", color="#4C78A8"
    )
    reuse_bars = right.bar(
        x + width / 2, reuse, width, label="KV reuse rate", color="#F58518"
    )
    right.bar_label(accuracy_bars, labels=[f"{value:.1f}%" for value in accuracy], padding=3)
    right.bar_label(reuse_bars, labels=[f"{value:.1f}%" for value in reuse], padding=3)
    right.set_xticks(x, labels)
    right.set_ylim(0, 103)
    right.set_ylabel("Percent")
    right.set_title("A stricter distance gate traded reuse for accuracy")
    right.grid(axis="y", alpha=0.22)
    right.legend(ncol=2, loc="lower center")
    right.annotate(
        "accuracy +5.9 pp\nreuse −23.2 pp",
        xy=(1, 68.0),
        xytext=(0.63, 41),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=9.5,
        ha="center",
    )

    fig.suptitle("KVComm validates distance heuristics with downstream task outcomes", y=0.99)
    fig.text(
        0.5,
        0.012,
        "Paper-native ablations (different datasets/protocols from this project): HumanEval and MMLU, four-agent setting.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    output.chmod(0o644)


def analyze(artifact_root: Path, output_dir: Path) -> dict[str, Any]:
    controlled = artifact_root / "impactkv_codemas_v2_controlled_sota_20260729"
    online_dir = controlled / "v90_online_kv_risk"
    functional_dir = controlled / "v80_ds1000_functional_validation"
    dense_repair_dir = controlled / "v81_high_density_late_lossy"
    sweep_dir = controlled / "v75_lcc_lossy_density_frontier"
    p33_dir = artifact_root / "impactkv_code_drift_oracle_p33_20260724"
    m48_dir = artifact_root / "impactkv_m48_attention_kv_risk_20260805/full50"
    m49_dir = artifact_root / "impactkv_m49_probe_proxy_20260805"
    p27c_dir = artifact_root / "impactkv_task_capsule_p27c_budget_grid_20260722"
    p27e_dir = artifact_root / "impactkv_task_capsule_p27e_confirmatory_20260722"
    distance_controlled = artifact_root / "impactkv_codemas_v2_controlled_sota_20260729"
    probehead_dir = artifact_root / "impactkv_probehead_v12_20260717"

    sources = {
        "online_metrics": online_dir / "FINAL_V90_100.jsonl",
        "fixed_route_eval": online_dir / "FINAL_V88_100_EVAL.json",
        "stale_guard_eval": online_dir / "FINAL_V90_100_EVAL.json",
        "dense_eval": online_dir / "FINAL_DENSE_100_EVAL.json",
        "online_workload": online_dir / "SEALED_VALIDATION_WORKLOAD.json",
        "functional_result": functional_dir / "RESULT.json",
        "high_density_result": dense_repair_dir / "RESULT.json",
        "functional_dense_metrics": functional_dir / "DENSE_DEVELOPMENT_50.jsonl",
        "functional_r075_metrics": functional_dir / "V80_LAYER24_DEVELOPMENT_50.jsonl",
        "functional_r090_metrics": dense_repair_dir / "V81_DEVELOPMENT_50.jsonl",
        "density_sweep": sweep_dir / "RESULT.json",
        "drift_oracle_attribution": p33_dir / "P33_ATTRIBUTION.json",
        "attention_kv_risk": m48_dir / "RESULT.json",
        "probe_proxy": m49_dir / "FINAL_RESULT.json",
        "capsule_development": p27c_dir / "P27C_DEVELOPMENT_RESULT.json",
        "capsule_independent": p27e_dir / "P27E_CONFIRMATORY_RESULT.json",
        "distance_consensus_independent": distance_controlled
        / "v65_independent_repobench_holdout200/RESULT.json",
        "dual_kv_transition_probe": distance_controlled
        / "v67_dual_kv_consensus_probe/RESULT.json",
        "probehead_calibration": probehead_dir / "DEVELOPMENT_CALIBRATION_REPORT.json",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen artifact(s):\n" + "\n".join(missing))

    online_rows = load_jsonl(sources["online_metrics"])
    fixed_pass = evaluation_map(sources["fixed_route_eval"])
    guard_pass = evaluation_map(sources["stale_guard_eval"])
    dense_pass = evaluation_map(sources["dense_eval"])
    workload = load_json(sources["online_workload"])
    libraries = {
        row["case_id"]: row["metadata"]["library"] for row in workload["cases"]
    }
    adaptive = [
        row for row in online_rows if row.get("online_value_diff_stale_mass") is not None
    ]
    if len(adaptive) != 72:
        raise AssertionError(f"expected 72 adaptive rows, found {len(adaptive)}")
    adaptive.sort(key=lambda row: float(row["online_value_diff_stale_mass"]))

    quartile_rows: list[dict[str, Any]] = []
    for quartile in range(4):
        bucket = adaptive[quartile * 18 : (quartile + 1) * 18]
        record: dict[str, Any] = {
            "quartile": quartile + 1,
            "cases": len(bucket),
            "stale_mass_min": min(float(row["online_value_diff_stale_mass"]) for row in bucket),
            "stale_mass_mean": float(
                np.mean([float(row["online_value_diff_stale_mass"]) for row in bucket])
            ),
            "stale_mass_max": max(float(row["online_value_diff_stale_mass"]) for row in bucket),
            "fixed_route_passes": sum(fixed_pass[row["case_id"]] for row in bucket),
            "stale_guard_passes": sum(guard_pass[row["case_id"]] for row in bucket),
            "dense_passes": sum(dense_pass[row["case_id"]] for row in bucket),
            "library_counts": dict(Counter(libraries[row["case_id"]] for row in bucket)),
        }
        for key in ("fixed_route", "stale_guard", "dense"):
            record[f"{key}_wilson95"] = wilson_interval(
                int(record[f"{key}_passes"]), len(bucket)
            )
        quartile_rows.append(record)

    masses = [float(row["online_value_diff_stale_mass"]) for row in adaptive]
    outcome_maps = {
        "fixed_route": fixed_pass,
        "stale_guard": guard_pass,
        "dense": dense_pass,
    }
    predictor_audit: dict[str, Any] = {}
    for index, (name, outcomes) in enumerate(outcome_maps.items()):
        passes = [outcomes[row["case_id"]] for row in adaptive]
        statistic, p_value = spearmanr(masses, passes)
        failure_labels = [1 - value for value in passes]
        predictor_audit[name] = {
            "cases": len(passes),
            "passes": sum(passes),
            "mean_stale_mass_pass": float(
                np.mean([mass for mass, passed in zip(masses, passes) if passed])
            ),
            "mean_stale_mass_fail": float(
                np.mean([mass for mass, passed in zip(masses, passes) if not passed])
            ),
            "spearman_stale_mass_vs_pass": float(statistic),
            "spearman_two_sided_p": float(p_value),
            "failure_auc_using_stale_mass": binary_auc(failure_labels, masses),
            "failure_auc_bootstrap95": bootstrap_auc(
                failure_labels, masses, seed=20_260_806 + index
            ),
        }

    low_fixed = int(quartile_rows[0]["fixed_route_passes"])
    high_fixed = int(quartile_rows[-1]["fixed_route_passes"])
    fixed_fisher = fisher_exact(
        [[low_fixed, 18 - low_fixed], [high_fixed, 18 - high_fixed]],
        alternative="two-sided",
    )
    predictor_audit["fixed_route_low_vs_high_quartile"] = {
        "low": [low_fixed, 18 - low_fixed],
        "high": [high_fixed, 18 - high_fixed],
        "odds_ratio": float(fixed_fisher.statistic),
        "fisher_exact_two_sided_p": float(fixed_fisher.pvalue),
    }

    triggered = [row for row in adaptive if row["online_kv_risk_triggered"]]
    triggered_changes = {
        "cases": len(triggered),
        "rescues": sum(
            not fixed_pass[row["case_id"]] and guard_pass[row["case_id"]]
            for row in triggered
        ),
        "damages": sum(
            fixed_pass[row["case_id"]] and not guard_pass[row["case_id"]]
            for row in triggered
        ),
    }
    triggered_changes["unchanged"] = (
        triggered_changes["cases"]
        - triggered_changes["rescues"]
        - triggered_changes["damages"]
    )
    discordant = triggered_changes["rescues"] + triggered_changes["damages"]
    triggered_changes["exact_mcnemar_two_sided_p"] = float(
        binomtest(
            min(triggered_changes["rescues"], triggered_changes["damages"]),
            discordant,
            0.5,
            alternative="two-sided",
        ).pvalue
        if discordant
        else 1.0
    )
    triggered_case_rows = [
        {
            "case_id": row["case_id"],
            "stale_mass": float(row["online_value_diff_stale_mass"]),
            "fixed_route_pass": fixed_pass[row["case_id"]],
            "stale_guard_pass": guard_pass[row["case_id"]],
            "dense_pass": dense_pass[row["case_id"]],
            "outcome_change": (
                "rescue"
                if not fixed_pass[row["case_id"]] and guard_pass[row["case_id"]]
                else "damage"
                if fixed_pass[row["case_id"]] and not guard_pass[row["case_id"]]
                else "unchanged"
            ),
        }
        for row in triggered
    ]

    functional_result = load_json(sources["functional_result"])
    high_density_result = load_json(sources["high_density_result"])
    dense_metrics = {row["case_id"]: row for row in load_jsonl(sources["functional_dense_metrics"])}
    r075_metrics = {row["case_id"]: row for row in load_jsonl(sources["functional_r075_metrics"])}
    r090_metrics = {row["case_id"]: row for row in load_jsonl(sources["functional_r090_metrics"])}
    if set(dense_metrics) != set(r075_metrics) or set(dense_metrics) != set(r090_metrics):
        raise AssertionError("functional comparison arms do not have identical case IDs")
    same_as_dense = {
        "r075": sum(
            r075_metrics[case_id]["output_sha256"] == dense_metrics[case_id]["output_sha256"]
            for case_id in dense_metrics
        ),
        "r090": sum(
            r090_metrics[case_id]["output_sha256"] == dense_metrics[case_id]["output_sha256"]
            for case_id in dense_metrics
        ),
    }
    functional_counterexample = [
        {
            "label": "Layer 24 / repair 75%",
            "recompute_ratio": 0.75,
            "mean_stale_fraction": functional_result["identity_and_mechanism"][
                "V80_mean_stale_fraction_of_target_tokens"
            ],
            "execution_passes": functional_result["development"]["arms"][
                "V80_layer24_r075"
            ]["execution_passes"],
            "same_as_dense_outputs": same_as_dense["r075"],
        },
        {
            "label": "Layer 24 / repair 90%",
            "recompute_ratio": 0.90,
            "mean_stale_fraction": high_density_result["V81"][
                "mean_stale_fraction_of_target_tokens"
            ],
            "execution_passes": high_density_result["execution_passes"][
                "V81_layer24_r090"
            ],
            "same_as_dense_outputs": same_as_dense["r090"],
        },
    ]
    functional_paired = {
        "r090_vs_r075_rescues": high_density_result["paired_accuracy"]["V81_vs_V80"][
            "rescues"
        ],
        "r090_vs_r075_damages": high_density_result["paired_accuracy"]["V81_vs_V80"][
            "damages"
        ],
    }
    r075_passes = int(functional_counterexample[0]["execution_passes"])
    rescues = len(functional_paired["r090_vs_r075_rescues"])
    damages = len(functional_paired["r090_vs_r075_damages"])
    functional_transition_matrix = [
        [len(dense_metrics) - r075_passes - rescues, rescues],
        [damages, r075_passes - damages],
    ]
    functional_paired[
        "transition_matrix_rows_r075_fail_pass_cols_r090_fail_pass"
    ] = functional_transition_matrix

    sweep_result = load_json(sources["density_sweep"])
    density_sweep_rows: list[dict[str, Any]] = []
    for ratio in (75, 80, 85, 90):
        key = f"cacheblend_r0{ratio}"
        row = sweep_result["results"][key]
        density_sweep_rows.append(
            {
                "recompute_ratio": ratio / 100.0,
                "minimum_stale_tokens": row["minimum_stale_k_tokens"],
                "exact_line_count": row["exact_line_count"],
                "mean_code_similarity": row["mean_code_similarity"],
                "speedup_vs_dense": row["speedup_vs_dense"],
            }
        )

    p33 = load_json(sources["drift_oracle_attribution"])
    m48 = load_json(sources["attention_kv_risk"])
    m49 = load_json(sources["probe_proxy"])
    p27c = load_json(sources["capsule_development"])
    p27e = load_json(sources["capsule_independent"])
    proxy_scope = {
        "single_island_kv_drift_vs_causal_js_spearman": m48["signals"][
            "kv_cosine_drift_mean"
        ]["causal_splice_logit_js"]["global_spearman"],
        "request_kv_drift_vs_composed_nll_spearman": m48["v46_request_correlations"][
            "kv_drift_q90_max"
        ]["composed_nll_delta_spearman"],
        "request_kv_drift_vs_code_similarity_change_spearman": m48[
            "v46_request_correlations"
        ]["kv_drift_q90_max"]["sglang_abs_code_sim_change_spearman"],
        "dense_target_drift_vs_nll_repair_utility_spearman": p33["correlations"][
            "dense_target_state_drift_vs_repair_utility"
        ]["spearman"],
        "short_probe_request_composed_js_spearman": m49["metrics"][
            "request_composed_js_spearman"
        ],
        "short_probe_request_composed_nll_spearman": m49["metrics"][
            "request_composed_nll_spearman"
        ],
        "short_probe_registered_gate_passed": m49["passed"],
    }
    capsule_nll = {
        "development": {
            "cases": 8,
            "mean_stale_kv_nll_loss": p27c["summaries"][
                "p27c-complete-capsule-tail-floor-r20"
            ]["mean_nll_loss_vs_capsule_dense"],
            "pipeline_nll_improvement": p27c["summaries"][
                "p27c-complete-capsule-tail-floor-r20"
            ]["mean_nll_improvement_vs_historical_full_tail"],
            "wins": p27c["summaries"]["p27c-complete-capsule-tail-floor-r20"][
                "wins_vs_historical_full_tail"
            ],
            "severe_losses": p27c["summaries"][
                "p27c-complete-capsule-tail-floor-r20"
            ]["severe_losses"],
        },
        "independent": {
            "cases": 17,
            "eligible_cases": p27e["eligible_cases"],
            "mean_stale_kv_nll_loss": p27e["mean_policy_loss_vs_capsule_dense"],
            "pipeline_nll_improvement": p27e["mean_policy_improvement_vs_full_tail"],
            "wins": p27e["wins_vs_full_tail"],
            "severe_losses": p27e["severe_losses"],
        },
    }
    distance_consensus = load_json(sources["distance_consensus_independent"])
    dual_kv_probe = load_json(sources["dual_kv_transition_probe"])
    probehead_calibration = load_json(sources["probehead_calibration"])
    distance_rows = distance_consensus["rows"]
    distance_rescues = sum(
        bool(row["candidate_exact_line"]) and not bool(row["cacheblend_exact_line"])
        for row in distance_rows
    )
    distance_damages = sum(
        bool(row["cacheblend_exact_line"]) and not bool(row["candidate_exact_line"])
        for row in distance_rows
    )
    distance_selector = {
        "method_semantics": (
            "At the same 75-percent recomputation budget, activate the semantic "
            "mask only when at least 75 percent of its positions overlap the "
            "layer-1 current-versus-cached V-difference top-k; otherwise retain "
            "the generic value-difference selector."
        ),
        "independent_exact_line": {
            "dataset": "RepoBench-P independent holdout",
            "cases": distance_consensus["samples"],
            "dense": distance_consensus["quality"]["exact_line"]["dense"],
            "generic_value_diff": distance_consensus["quality"]["exact_line"][
                "cacheblend"
            ],
            "distance_consensus": distance_consensus["quality"]["exact_line"][
                "candidate"
            ],
            "rescues": distance_rescues,
            "damages": distance_damages,
            "same_recompute_ratio": 0.75,
            "same_prompt_and_token_ids": True,
            "metric_limit": "exact next-line, not functional execution accuracy",
        },
        "transition_agreement_probe": {
            "cases": dual_kv_probe["population"]["cases"],
            "rescues": dual_kv_probe["population"]["rescues"],
            "damages": dual_kv_probe["population"]["damages"],
            "key": {
                "rescue_mean": dual_kv_probe["primary_feature"]["rescue_mean"],
                "damage_mean": dual_kv_probe["primary_feature"]["damage_mean"],
                "rescue_minus_damage": dual_kv_probe["primary_feature"][
                    "rescue_minus_damage"
                ],
                "auc": dual_kv_probe["primary_feature"][
                    "roc_auc_higher_predicts_rescue"
                ],
            },
            "value": {
                "rescue_mean": dual_kv_probe["secondary_features"][
                    "full_candidate_value_topk_overlap"
                ]["rescue_mean"],
                "damage_mean": dual_kv_probe["secondary_features"][
                    "full_candidate_value_topk_overlap"
                ]["damage_mean"],
                "rescue_minus_damage": dual_kv_probe["secondary_features"][
                    "full_candidate_value_topk_overlap"
                ]["rescue_minus_damage"],
                "auc": dual_kv_probe["secondary_features"][
                    "full_candidate_value_topk_overlap"
                ]["roc_auc_higher_predicts_rescue"],
            },
        },
        "earlier_low_deviation_gate": {
            "definition": (
                "Dense-recompute H head tokens under the target prefix and copy "
                "the remaining body only when max shifted-K/V cosine deviation "
                "is below a threshold."
            ),
            "configurations_evaluated": probehead_calibration[
                "configurations_evaluated"
            ],
            "feasible_configurations": probehead_calibration[
                "feasible_configurations"
            ],
            "metric_limit": (
                "capacity and causal-JS harm gates only; task accuracy stage was closed"
            ),
        },
    }

    related_work = {
        "evidence_scale": {
            "0": "not used in the scoped evidence",
            "1": "internal mechanism, selector, or language-model proxy",
            "2": "downstream output, task, user, or system outcome",
        },
        "coverage": [
            {
                "work": "CacheBlend",
                "internal_proxy": 1,
                "nll_ppl": 0,
                "output_metric": 2,
                "task_success": 0,
                "human_check": 0,
                "systems": 2,
                "quality_summary": "KV/attention deviation for selection; F1 or ROUGE-L for final quality",
                "source": "https://arxiv.org/html/2405.16444",
            },
            {
                "work": "KVComm",
                "internal_proxy": 1,
                "nll_ppl": 0,
                "output_metric": 0,
                "task_success": 2,
                "human_check": 0,
                "systems": 2,
                "quality_summary": "embedding/KV proximity for matching; Accuracy and HumanEval Pass@1 for final quality",
                "source": "https://arxiv.org/html/2510.12872",
            },
            {
                "work": "CacheGen",
                "internal_proxy": 0,
                "nll_ppl": 1,
                "output_metric": 2,
                "task_success": 2,
                "human_check": 0,
                "systems": 2,
                "quality_summary": "perplexity proxy plus LongChat accuracy and QA F1",
                "source": "https://arxiv.org/html/2310.07240",
            },
            {
                "work": "Cache-Craft",
                "internal_proxy": 1,
                "nll_ppl": 0,
                "output_metric": 2,
                "task_success": 2,
                "human_check": 2,
                "systems": 2,
                "quality_summary": "attention-derived CCI/CFO for recomputation; ROUGE/Jaccard/accuracy and a user study",
                "source": "https://arxiv.org/html/2502.15734",
            },
            {
                "work": "This project audit",
                "internal_proxy": 1,
                "nll_ppl": 1,
                "output_metric": 1,
                "task_success": 2,
                "human_check": 0,
                "systems": 2,
                "quality_summary": "KV/NLL/exact diagnostics; official coding execution and TTFT for decisions",
                "source": "local frozen artifacts listed in audit_data.json",
            },
        ],
        "kvcomm": {
            "source": "https://arxiv.org/html/2510.12872",
            "offset_approximation_humaneval_4agent": [
                {"method": "nearest", "accuracy_percent": 47.20, "reuse_percent": 78.9},
                {"method": "cosine", "accuracy_percent": 83.23, "reuse_percent": 82.5},
                {"method": "l2_weighted", "accuracy_percent": 83.23, "reuse_percent": 81.1},
                {"method": "dense_original", "accuracy_percent": 84.45, "reuse_percent": None},
            ],
            "matching_criterion_mmlu_4agent": [
                {"method": "length_only", "accuracy_percent": 62.1, "reuse_percent": 93.3},
                {"method": "length_and_embedding_distance", "accuracy_percent": 68.0, "reuse_percent": 70.1},
            ],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_quartiles = [
        {
            key: (json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value)
            for key, value in row.items()
        }
        for row in quartile_rows
    ]
    write_csv(output_dir / "v_stale_mass_accuracy_quartiles.csv", csv_quartiles)
    write_csv(output_dir / "triggered_request_outcomes.csv", triggered_case_rows)
    write_csv(output_dir / "nested_value_diff_density_sweep.csv", density_sweep_rows)
    write_csv(output_dir / "functional_stale_fraction_counterexample.csv", functional_counterexample)
    write_csv(
        output_dir / "functional_accuracy_transition.csv",
        [
            {
                "repair_75_outcome": row_label,
                "repair_90_fail": functional_transition_matrix[row_index][0],
                "repair_90_pass": functional_transition_matrix[row_index][1],
            }
            for row_index, row_label in enumerate(("fail", "pass"))
        ],
    )
    write_csv(
        output_dir / "proxy_scope_correlations.csv",
        [
            {
                "comparison": label,
                "spearman": value,
                "evidence_scope": scope,
            }
            for label, value, scope in (
                (
                    "single-island KV drift vs causal logit JS",
                    proxy_scope["single_island_kv_drift_vs_causal_js_spearman"],
                    "local perturbation",
                ),
                (
                    "request KV drift vs code-similarity change",
                    proxy_scope[
                        "request_kv_drift_vs_code_similarity_change_spearman"
                    ],
                    "request-level text proxy",
                ),
                (
                    "request KV drift vs composed NLL",
                    proxy_scope["request_kv_drift_vs_composed_nll_spearman"],
                    "request-level NLL proxy",
                ),
                (
                    "16-token probe vs composed NLL",
                    proxy_scope[
                        "short_probe_request_composed_nll_spearman"
                    ],
                    "short-probe to request proxy",
                ),
                (
                    "dense-target drift vs NLL repair utility",
                    proxy_scope[
                        "dense_target_drift_vs_nll_repair_utility_spearman"
                    ],
                    "repair utility",
                ),
            )
        ],
    )
    write_csv(
        output_dir / "nll_generalization.csv",
        [
            {
                "split": split,
                "cases": row["cases"],
                "eligible_cases": row.get("eligible_cases", row["cases"]),
                "mean_stale_kv_nll_loss": row["mean_stale_kv_nll_loss"],
                "pipeline_nll_improvement": row["pipeline_nll_improvement"],
                "wins": row["wins"],
                "severe_losses": row["severe_losses"],
            }
            for split, row in capsule_nll.items()
        ],
    )
    write_csv(
        output_dir / "distance_consensus_accuracy.csv",
        [
            {
                "method": method,
                "exact_line_matches": value,
                "cases": distance_selector["independent_exact_line"]["cases"],
                "recompute_ratio": (
                    1.0
                    if method == "dense"
                    else distance_selector["independent_exact_line"][
                        "same_recompute_ratio"
                    ]
                ),
            }
            for method, value in (
                (
                    "dense",
                    distance_selector["independent_exact_line"]["dense"],
                ),
                (
                    "generic_value_difference",
                    distance_selector["independent_exact_line"][
                        "generic_value_diff"
                    ],
                ),
                (
                    "semantic_value_distance_consensus",
                    distance_selector["independent_exact_line"][
                        "distance_consensus"
                    ],
                ),
            )
        ],
    )
    write_csv(
        output_dir / "kv_agreement_transition_probe.csv",
        [
            {
                "signal": signal,
                "rescue_mean": row["rescue_mean"],
                "damage_mean": row["damage_mean"],
                "rescue_minus_damage": row["rescue_minus_damage"],
                "rescue_auc": row["auc"],
            }
            for signal, row in (
                ("key_topk_agreement", distance_selector["transition_agreement_probe"]["key"]),
                ("value_topk_agreement", distance_selector["transition_agreement_probe"]["value"]),
            )
        ],
    )
    write_csv(output_dir / "related_work_evidence_coverage.csv", related_work["coverage"])
    write_csv(
        output_dir / "kvcomm_distance_ablations.csv",
        [
            {
                "experiment": experiment,
                "method": row["method"],
                "accuracy_percent": row["accuracy_percent"],
                "reuse_percent": row["reuse_percent"],
                "source": related_work["kvcomm"]["source"],
            }
            for experiment, rows in (
                (
                    "offset_approximation_humaneval_4agent",
                    related_work["kvcomm"]["offset_approximation_humaneval_4agent"],
                ),
                (
                    "matching_criterion_mmlu_4agent",
                    related_work["kvcomm"]["matching_criterion_mmlu_4agent"],
                ),
            )
            for row in rows
        ],
    )

    configure_plotting()
    plot_current_method_pipeline(
        output_dir / "00_current_method_pipeline.png"
    )
    plot_quartiles(quartile_rows, output_dir / "01_v_stale_mass_accuracy_quartiles.png")
    plot_density_sweep(density_sweep_rows, output_dir / "02_nested_value_diff_density_sweep.png")
    plot_functional_counterexample(
        functional_counterexample,
        output_dir / "03_functional_stale_fraction_counterexample.png",
    )
    plot_triggered_changes(
        triggered_changes,
        output_dir / "04_stronger_repair_paired_outcomes.png",
    )
    plot_failure_auc(
        predictor_audit,
        output_dir / "05_failure_auc_with_ci.png",
    )
    plot_functional_transition(
        functional_transition_matrix,
        output_dir / "06_functional_paired_transition.png",
    )
    plot_proxy_scope(
        proxy_scope,
        output_dir / "07_proxy_scope_correlation_decay.png",
    )
    plot_nll_generalization(
        capsule_nll,
        output_dir / "08_nll_generalization_reversal.png",
    )
    plot_distance_selector_counterexample(
        distance_selector,
        output_dir / "09_distance_selector_counterexample.png",
    )
    plot_related_work_evidence_matrix(
        related_work["coverage"],
        output_dir / "10_related_work_evidence_matrix.png",
    )
    plot_kvcomm_distance_ablation(
        related_work["kvcomm"],
        output_dir / "11_kvcomm_distance_ablations.png",
    )

    report = {
        "status": "COMPLETE_EXISTING_ARTIFACT_AUDIT",
        "claim": (
            "Existing controlled evidence falsifies the claim that lower K/V-deviation "
            "or staleness proxies are sufficient, monotonic predictors of higher coding-task accuracy."
        ),
        "claim_limit": (
            "The audit does not prove zero population association and does not contain a saved "
            "all-layer final-KV tensor norm. The strongest intervention guarantees lower residual "
            "check-layer V-difference by nested top-k construction and directly measures fewer stale tokens."
        ),
        "online_stale_mass_predictor": {
            "definition": (
                "Fraction of layer-1 squared current-versus-cached V-difference mass left "
                "outside the top-60-percent selected tokens."
            ),
            "adaptive_cases": len(adaptive),
            "quartiles": quartile_rows,
            "predictor_audit": predictor_audit,
            "triggered_paired_changes": triggered_changes,
        },
        "functional_density_counterexample": {
            "dataset": "DS-1000 development50 with official execution evaluator",
            "same_prompt_token_hashes": "50/50",
            "arms": functional_counterexample,
            "paired": functional_paired,
            "dense_execution_passes": functional_result["development"]["arms"]["Dense"][
                "execution_passes"
            ],
            "selector_fact": (
                "Both arms use the same layer-24 value-difference ranking. Top-90% is a "
                "superset of top-75%, so residual check-layer V-difference mass is non-increasing."
            ),
        },
        "next_line_density_sweep": density_sweep_rows,
        "proxy_scope_decay": proxy_scope,
        "nll_generalization_counterexample": capsule_nll,
        "earlier_distance_aware_selector_counterexample": distance_selector,
        "related_work_evidence": related_work,
        "source_artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in sources.items()
        },
    }
    result_path = output_dir / "audit_data.json"
    result_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result_path.chmod(0o644)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args.artifact_root.resolve(), args.output_dir.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(args.output_dir.resolve()),
                "adaptive_cases": result["online_stale_mass_predictor"]["adaptive_cases"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
