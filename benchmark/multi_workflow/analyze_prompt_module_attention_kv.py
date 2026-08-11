#!/usr/bin/env python3
"""Compare prompt-module attention under Dense and lossy current-method reuse.

This is a read-only analysis over the frozen 26-case global block-attention
artifact and the matching attention/KV perturbation artifact.  It does not run
the model, alter frozen observations, or treat attention/KV metrics as task
accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


plt: Any = None
mcolors: Any = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT.parents[1] / "kvflow-artifacts"
DEFAULT_GLOBAL = (
    ARTIFACT_ROOT
    / "impactkv_global_block_attention_20260806/frozen26_r2"
)
DEFAULT_KV = (
    ARTIFACT_ROOT
    / "impactkv_attention_kv_bound_20260806/frozen26_mass_aware"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "docs/kvflow/assets/prompt_module_attention_kv_20260806"
)
LAYERS = (0, 8, 17, 26, 35)

CATEGORY_ORDER = (
    "system_instruction",
    "user_task",
    "compaction_notice",
    "assistant_action",
    "copied_observation_island",
    "read_observation_path_relevant",
    "read_observation_path_disjoint",
    "other_tool_result",
    "generation_marker",
)
QUERY_ORDER = tuple(
    category
    for category in CATEGORY_ORDER
    if category != "copied_observation_island"
)
SUFFIX_QUERY_ORDER = (
    "assistant_action",
    "read_observation_path_relevant",
    "read_observation_path_disjoint",
    "other_tool_result",
    "generation_marker",
)
LABELS = {
    "system_instruction": "System prompt",
    "user_task": "Coding task",
    "compaction_notice": "Context control",
    "assistant_action": "Agent action",
    "copied_observation_island": "Copied repo evidence",
    "read_observation_path_relevant": "Path-relevant repo evidence",
    "read_observation_path_disjoint": "Other repo evidence",
    "other_tool_result": "Tool / runtime feedback",
    "generation_marker": "Next action",
}
SHORT_LABELS = {
    "system_instruction": "System",
    "user_task": "Task",
    "compaction_notice": "Context",
    "assistant_action": "Agent action",
    "copied_observation_island": "Copied evidence",
    "read_observation_path_relevant": "Relevant evidence",
    "read_observation_path_disjoint": "Other evidence",
    "other_tool_result": "Tool feedback",
    "generation_marker": "Next action",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o644)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o644)


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return math.nan
    return float(spearmanr(left, right).statistic)


def category_distribution(
    distribution: Mapping[str, float], blocks: Mapping[str, Mapping[str, Any]]
) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    for block_id, mass in distribution.items():
        result[str(blocks[block_id]["category"])] += float(mass)
    return dict(result)


def aggregate_matrix(
    *,
    observations: Sequence[Mapping[str, Any]],
    design_by_id: Mapping[str, Mapping[str, Any]],
    arm: str,
    suffix_only: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    weights: dict[str, float] = defaultdict(float)
    for observation in observations:
        case = design_by_id[str(observation["case_id"])]
        blocks = {str(row["block_id"]): row for row in case["target_blocks"]}
        copied_end = int(case["target_start"]) + int(case["length"])
        for layer in LAYERS:
            matrix = observation[f"{arm}_matrix"][str(layer)]
            for query_id, distribution in matrix.items():
                if distribution is None:
                    continue
                query = blocks[query_id]
                query_category = str(query["category"])
                if query_category == "copied_observation_island":
                    continue
                if suffix_only and int(query["start"]) < copied_end:
                    continue
                weight = float(query["tokens"])
                weights[query_category] += weight
                category_mass = category_distribution(distribution, blocks)
                for key_category, mass in category_mass.items():
                    sums[query_category][key_category] += weight * mass
    result = {
        query: {
            key: value / weights[query]
            for key, value in key_values.items()
        }
        for query, key_values in sums.items()
    }
    return result, dict(weights)


def matrix_array(
    matrix: Mapping[str, Mapping[str, float]], query_order: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [
            [float(matrix.get(query, {}).get(key, 0.0)) for key in CATEGORY_ORDER]
            for query in query_order
        ],
        dtype=np.float64,
    )


def largest_matrix_changes(
    dense: Mapping[str, Mapping[str, float]],
    reuse: Mapping[str, Mapping[str, float]],
    query_order: Sequence[str],
    limit: int = 12,
) -> list[dict[str, Any]]:
    rows = []
    for query in query_order:
        for key in CATEGORY_ORDER:
            dense_value = float(dense.get(query, {}).get(key, 0.0))
            reuse_value = float(reuse.get(query, {}).get(key, 0.0))
            rows.append(
                {
                    "query_module": query,
                    "key_module": key,
                    "dense_attention_mass": dense_value,
                    "reuse_attention_mass": reuse_value,
                    "delta_percentage_points": 100.0 * (reuse_value - dense_value),
                }
            )
    return sorted(
        rows,
        key=lambda row: abs(float(row["delta_percentage_points"])),
        reverse=True,
    )[:limit]


def module_rows(
    *,
    global_observations: Sequence[Mapping[str, Any]],
    kv_observations: Mapping[str, Mapping[str, Any]],
    design_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for observation in global_observations:
        case_id = str(observation["case_id"])
        case = design_by_id[case_id]
        blocks = {str(row["block_id"]): row for row in case["target_blocks"]}
        copied_end = int(case["target_start"]) + int(case["length"])
        kv_by_layer = {
            int(row["layer"]): row
            for row in kv_observations[case_id]["layers"]
        }
        for layer in LAYERS:
            dense_matrix = observation["dense_matrix"][str(layer)]
            reuse_matrix = observation["reuse_matrix"][str(layer)]
            raw_drift = float(kv_by_layer[layer]["raw_kv_drift_mean"])
            key_drift = float(kv_by_layer[layer]["key_cosine_drift_mean"])
            value_drift = float(kv_by_layer[layer]["value_cosine_drift_mean"])
            for query_id, dense_distribution in dense_matrix.items():
                reuse_distribution = reuse_matrix[query_id]
                query = blocks[query_id]
                if (
                    dense_distribution is None
                    or reuse_distribution is None
                    or int(query["start"]) < copied_end
                ):
                    continue
                dense_categories = category_distribution(dense_distribution, blocks)
                reuse_categories = category_distribution(reuse_distribution, blocks)
                categories = set(dense_categories) | set(reuse_categories)
                row_tv = 0.5 * sum(
                    abs(
                        reuse_categories.get(category, 0.0)
                        - dense_categories.get(category, 0.0)
                    )
                    for category in categories
                )
                dense_copied_mass = dense_categories.get(
                    "copied_observation_island", 0.0
                )
                reuse_copied_mass = reuse_categories.get(
                    "copied_observation_island", 0.0
                )
                rows.append(
                    {
                        "case_id": case_id,
                        "instance_id": observation["instance_id"],
                        "layer": layer,
                        "query_block_id": query_id,
                        "query_module": str(query["category"]),
                        "query_tokens": int(query["tokens"]),
                        "row_tv": row_tv,
                        "dense_copied_mass": dense_copied_mass,
                        "reuse_copied_mass": reuse_copied_mass,
                        "copied_mass_delta": reuse_copied_mass - dense_copied_mass,
                        "raw_kv_drift": raw_drift,
                        "key_cosine_drift": key_drift,
                        "value_cosine_drift": value_drift,
                        "attention_times_drift": dense_copied_mass * raw_drift,
                    }
                )
    return rows


def aggregate_case_module_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["case_id"]), str(row["query_module"]))].append(row)
    result = []
    metrics = (
        "row_tv",
        "dense_copied_mass",
        "reuse_copied_mass",
        "copied_mass_delta",
        "raw_kv_drift",
        "key_cosine_drift",
        "value_cosine_drift",
        "attention_times_drift",
    )
    for (case_id, module), selected in grouped.items():
        weight = sum(float(row["query_tokens"]) for row in selected)
        aggregate = {
            "case_id": case_id,
            "query_module": module,
            "points": len(selected),
            "weighted_query_tokens": weight,
        }
        for metric in metrics:
            aggregate[metric] = sum(
                float(row["query_tokens"]) * float(row[metric])
                for row in selected
            ) / weight
        result.append(aggregate)
    return result


def module_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for module in SUFFIX_QUERY_ORDER:
        selected = [row for row in rows if row["query_module"] == module]
        if not selected:
            continue
        target = [float(row["row_tv"]) for row in selected]
        result[module] = {
            "cases": len(selected),
            "median_row_tv": statistics.median(target),
            "median_dense_copied_mass": statistics.median(
                float(row["dense_copied_mass"]) for row in selected
            ),
            "median_abs_copied_mass_delta": statistics.median(
                abs(float(row["copied_mass_delta"])) for row in selected
            ),
            "raw_drift_vs_row_tv_spearman": safe_spearman(
                [float(row["raw_kv_drift"]) for row in selected], target
            ),
            "attention_times_drift_vs_row_tv_spearman": safe_spearman(
                [float(row["attention_times_drift"]) for row in selected], target
            ),
        }
    return result


def configure_plotting() -> None:
    global plt, mcolors
    import matplotlib.colors as matplotlib_colors
    import matplotlib.pyplot as pyplot

    plt = pyplot
    mcolors = matplotlib_colors
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, path: Path, *, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    path.chmod(0o644)


def annotate_mass(axis: plt.Axes, values: np.ndarray) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if value >= 0.015:
                color = "white" if value >= 0.35 else "black"
                axis.text(
                    column,
                    row,
                    f"{100 * value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.7,
                    color=color,
                )


def annotate_delta(axis: plt.Axes, values_pp: np.ndarray) -> None:
    threshold = max(float(np.abs(values_pp).max()) * 0.08, 0.005)
    for row in range(values_pp.shape[0]):
        for column in range(values_pp.shape[1]):
            value = values_pp[row, column]
            if abs(value) >= threshold:
                axis.text(
                    column,
                    row,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.7,
                    color="black",
                )


def plot_heatmaps(
    *,
    dense: Mapping[str, Mapping[str, float]],
    reuse: Mapping[str, Mapping[str, float]],
    query_order: Sequence[str],
    title: str,
    path: Path,
) -> None:
    dense_values = matrix_array(dense, query_order)
    reuse_values = matrix_array(reuse, query_order)
    delta_pp = 100.0 * (reuse_values - dense_values)
    maximum_delta = max(float(np.abs(delta_pp).max()), 1e-4)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(22.0, max(5.6, 0.65 * len(query_order) + 2.2)),
        gridspec_kw={"width_ratios": [1, 1, 1.10]},
        constrained_layout=True,
    )
    norm = mcolors.PowerNorm(gamma=0.55, vmin=0.0, vmax=1.0)
    for axis, values, subtitle in zip(
        axes[:2],
        (dense_values, reuse_values),
        ("Dense full computation", "Lossy K/V reuse"),
        strict=True,
    ):
        image = axis.imshow(values, cmap="Blues", norm=norm, aspect="auto")
        annotate_mass(axis, values)
        axis.set_title(subtitle)
        axis.set_xticks(range(len(CATEGORY_ORDER)), [SHORT_LABELS[x] for x in CATEGORY_ORDER], rotation=42, ha="right")
        axis.set_yticks(range(len(query_order)), [SHORT_LABELS[x] for x in query_order])
        axis.set_xlabel("Key module receiving attention")
    axes[0].set_ylabel("Query module")
    axes[1].set_yticklabels([])
    delta_image = axes[2].imshow(
        delta_pp,
        cmap="RdBu_r",
        vmin=-maximum_delta,
        vmax=maximum_delta,
        aspect="auto",
    )
    annotate_delta(axes[2], delta_pp)
    axes[2].set_title("Lossy − Dense (percentage points)")
    axes[2].set_xticks(range(len(CATEGORY_ORDER)), [SHORT_LABELS[x] for x in CATEGORY_ORDER], rotation=42, ha="right")
    axes[2].set_yticks(range(len(query_order)), [SHORT_LABELS[x] for x in query_order])
    axes[2].set_xlabel("Key module receiving attention")
    fig.colorbar(
        delta_image,
        ax=axes[2],
        fraction=0.045,
        pad=0.025,
        label="attention-mass change (percentage points)",
    )
    fig.suptitle(title, fontsize=16, y=1.075)
    save(fig, path, tight=False)


def plot_module_kv_correlations(stats: Mapping[str, Any], path: Path) -> None:
    modules = [module for module in SUFFIX_QUERY_ORDER if module in stats]
    raw = [stats[module]["raw_drift_vs_row_tv_spearman"] for module in modules]
    weighted = [
        stats[module]["attention_times_drift_vs_row_tv_spearman"]
        for module in modules
    ]
    positions = np.arange(len(modules))
    width = 0.36
    fig, axis = plt.subplots(figsize=(12.2, 5.8))
    raw_bars = axis.bar(
        positions - width / 2,
        raw,
        width,
        color="#A0A7B0",
        label="raw K/V drift",
    )
    weighted_bars = axis.bar(
        positions + width / 2,
        weighted,
        width,
        color="#4C78A8",
        label="copied-island attention × drift",
    )
    axis.bar_label(raw_bars, fmt="%.2f", padding=3, fontsize=9)
    axis.bar_label(weighted_bars, fmt="%.2f", padding=3, fontsize=9)
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_ylim(-0.1, 1.08)
    axis.set_xticks(
        positions,
        [f"{SHORT_LABELS[module]}\n(n={stats[module]['cases']} cases)" for module in modules],
    )
    axis.set_ylabel("Spearman with module-row attention TV")
    axis.set_title("K/V drift explains routing change differently across prompt modules")
    axis.legend(loc="upper left")
    axis.grid(axis="y", alpha=0.2)
    save(fig, path)


def plot_module_sensitivity(stats: Mapping[str, Any], path: Path) -> None:
    modules = [module for module in SUFFIX_QUERY_ORDER if module in stats]
    tv = [100 * stats[module]["median_row_tv"] for module in modules]
    mass = [100 * stats[module]["median_dense_copied_mass"] for module in modules]
    positions = np.arange(len(modules))
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.5))
    bars = axes[0].barh(positions, mass, color="#59A14F", alpha=0.82)
    axes[0].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[0].set_yticks(positions, [SHORT_LABELS[module] for module in modules])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("median Dense attention to copied evidence")
    axes[0].set_title("Which suffix modules read the copied island?")

    bars = axes[1].barh(positions, tv, color="#E15759", alpha=0.8)
    axes[1].bar_label(bars, fmt="%.3f pp", padding=3)
    axes[1].set_yticks(positions, [SHORT_LABELS[module] for module in modules])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("median Dense↔Lossy row TV (percentage points)")
    axes[1].set_title("Whose attention routing changes most?")
    fig.suptitle("Module dependency and observed lossy perturbation", fontsize=16)
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-attention", type=Path, default=DEFAULT_GLOBAL)
    parser.add_argument("--kv", type=Path, default=DEFAULT_KV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    design = load_json(args.global_attention / "DESIGN.json")
    design_by_id = {str(row["case_id"]): row for row in design["cases"]}
    global_observations = load_jsonl(args.global_attention / "OBSERVATIONS.jsonl")
    kv_observations = {
        str(row["case_id"]): row
        for row in load_jsonl(args.kv / "OBSERVATIONS.jsonl")
    }
    case_ids = {str(row["case_id"]) for row in global_observations}
    if len(case_ids) != 26 or case_ids != set(kv_observations):
        raise ValueError("expected identical frozen 26-case cohorts")

    dense_full, dense_full_weights = aggregate_matrix(
        observations=global_observations,
        design_by_id=design_by_id,
        arm="dense",
        suffix_only=False,
    )
    reuse_full, reuse_full_weights = aggregate_matrix(
        observations=global_observations,
        design_by_id=design_by_id,
        arm="reuse",
        suffix_only=False,
    )
    dense_suffix, dense_suffix_weights = aggregate_matrix(
        observations=global_observations,
        design_by_id=design_by_id,
        arm="dense",
        suffix_only=True,
    )
    reuse_suffix, reuse_suffix_weights = aggregate_matrix(
        observations=global_observations,
        design_by_id=design_by_id,
        arm="reuse",
        suffix_only=True,
    )
    if dense_full_weights != reuse_full_weights or dense_suffix_weights != reuse_suffix_weights:
        raise ValueError("Dense and reuse query weights differ")

    raw_rows = module_rows(
        global_observations=global_observations,
        kv_observations=kv_observations,
        design_by_id=design_by_id,
    )
    case_module_rows = aggregate_case_module_rows(raw_rows)
    stats = module_statistics(case_module_rows)
    pooled_target = [float(row["row_tv"]) for row in case_module_rows]
    pooled = {
        "case_module_points": len(case_module_rows),
        "raw_drift_vs_row_tv_spearman": safe_spearman(
            [float(row["raw_kv_drift"]) for row in case_module_rows],
            pooled_target,
        ),
        "attention_times_drift_vs_row_tv_spearman": safe_spearman(
            [float(row["attention_times_drift"]) for row in case_module_rows],
            pooled_target,
        ),
        "warning": (
            "Pooled correlations mix module identities; module-conditional values "
            "are the primary interpretation."
        ),
    }
    matrices = (dense_full, reuse_full, dense_suffix, reuse_suffix)
    row_sum_error_max = max(
        abs(sum(row.values()) - 1.0)
        for matrix in matrices
        for row in matrix.values()
    )
    prefix_negative_control_categories = (
        "system_instruction",
        "user_task",
        "compaction_notice",
    )
    prefix_negative_control_delta_max = max(
        abs(
            float(reuse_full.get(query, {}).get(key, 0.0))
            - float(dense_full.get(query, {}).get(key, 0.0))
        )
        for query in prefix_negative_control_categories
        if query in dense_full and query in reuse_full
        for key in CATEGORY_ORDER
    )
    if row_sum_error_max > 1e-6:
        raise ValueError(f"category row normalization error: {row_sum_error_max}")
    if prefix_negative_control_delta_max > 1e-7:
        raise ValueError(
            "prefix negative control changed: "
            f"{prefix_negative_control_delta_max}"
        )

    result = {
        "status": "COMPLETE",
        "analysis_type": "post_hoc_read_only_module_diagnostic",
        "cases": 26,
        "tasks": len({row["instance_id"] for row in global_observations}),
        "layers_zero_based": list(LAYERS),
        "module_labels": LABELS,
        "comparison_contract": {
            "dense": "full current-prompt computation",
            "reuse": "runtime-faithful single-island lossy K/V splice",
            "full_heatmap_query_rows": (
                "all query modules executed in both arms; copied-island target rows "
                "are excluded because reuse does not execute them"
            ),
            "suffix_heatmap_query_rows": (
                "only query blocks at or after the copied island end; these can be "
                "causally affected by the lossy cache"
            ),
            "key_columns": "all prompt modules, including copied repository evidence",
        },
        "dense_full_matrix": dense_full,
        "reuse_full_matrix": reuse_full,
        "dense_suffix_matrix": dense_suffix,
        "reuse_suffix_matrix": reuse_suffix,
        "largest_full_matrix_changes": largest_matrix_changes(
            dense_full, reuse_full, QUERY_ORDER
        ),
        "largest_suffix_matrix_changes": largest_matrix_changes(
            dense_suffix, reuse_suffix, SUFFIX_QUERY_ORDER
        ),
        "module_statistics": stats,
        "pooled_statistics": pooled,
        "mechanical_checks": {
            "matrix_row_sum_abs_error_max": row_sum_error_max,
            "prefix_negative_control_abs_delta_max": (
                prefix_negative_control_delta_max
            ),
            "post_copy_block_layer_rows": len(raw_rows),
            "case_module_rows": len(case_module_rows),
            "matching_case_ids": True,
            "copied_query_rows_compared": False,
        },
        "scope": (
            "Qwen2.5-Coder-3B BF16 mechanism proxy over unchanged real current-method "
            "prompts; attention is token-weighted over five layers. K/V deviation uses "
            "the matching copied island. This is not task accuracy, native-30B attention, "
            "an online selector, or an independent cohort."
        ),
    }
    write_json(args.output / "RESULT.json", result)
    write_csv(args.output / "CASE_MODULE_ROWS.csv", case_module_rows)

    configure_plotting()
    full_queries = [query for query in QUERY_ORDER if query in dense_full and query in reuse_full]
    suffix_queries = [query for query in SUFFIX_QUERY_ORDER if query in dense_suffix and query in reuse_suffix]
    plot_heatmaps(
        dense=dense_full,
        reuse=reuse_full,
        query_order=full_queries,
        title="Full prompt modules: Dense computation versus lossy K/V reuse",
        path=args.output / "01_full_prompt_module_heatmaps.png",
    )
    plot_heatmaps(
        dense=dense_suffix,
        reuse=reuse_suffix,
        query_order=suffix_queries,
        title="Post-copy query modules: the causally affected attention region",
        path=args.output / "02_post_copy_module_heatmaps.png",
    )
    plot_module_sensitivity(stats, args.output / "03_module_dependency_and_tv.png")
    plot_module_kv_correlations(stats, args.output / "04_module_kv_correlations.png")
    manifest = {
        "result": str(args.output / "RESULT.json"),
        "case_module_rows": str(args.output / "CASE_MODULE_ROWS.csv"),
        "figures": sorted(path.name for path in args.output.glob("*.png")),
    }
    write_json(args.output / "MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
