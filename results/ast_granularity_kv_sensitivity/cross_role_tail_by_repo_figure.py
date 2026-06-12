#!/usr/bin/env python3
"""Plot cross-role tail concentration by repository.

The path-level table shows that class and statement-window tail cells are
localized. This script aggregates the same cross-role cells to the repository
level and writes the paper figure plus a small JSON/Markdown summary.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/ast_granularity_kv_sensitivity/data/ast_granularity_distance_7b.json"
OUT_JSON = ROOT / "results/ast_granularity_kv_sensitivity/data/cross_role_tail_by_repo.json"
OUT_MD = ROOT / "results/ast_granularity_kv_sensitivity/cross_role_tail_by_repo.md"
LOCAL_OUT = ROOT / "results/ast_granularity_kv_sensitivity/figures/fig_cross_role_tail_by_repo.png"
PAPER_OUT = Path("/home/gfy/Paper_CodeMAS/CodeAgent_UCM_HKBU/figures/fig_cross_role_tail_by_repo.pdf")

REPOS = [
    "astropy",
    "django",
    "flask",
    "matplotlib",
    "pylint",
    "pytest",
    "requests",
    "scikit-learn",
    "seaborn",
    "xarray",
]
GRANULARITIES = ["class", "statement_window"]
THRESHOLD = 0.5


def repo_for_path(path: str) -> str:
    if path.startswith("astropy/"):
        return "astropy"
    if path.startswith("django/"):
        return "django"
    if path.startswith("src/flask/"):
        return "flask"
    if path.startswith("lib/matplotlib/"):
        return "matplotlib"
    if path.startswith("pylint/"):
        return "pylint"
    if path.startswith("src/_pytest/") or path.startswith("testing/python/"):
        return "pytest"
    if path.startswith("requests/"):
        return "requests"
    if path.startswith("sklearn/"):
        return "scikit-learn"
    if path.startswith("seaborn/"):
        return "seaborn"
    if path.startswith("xarray/"):
        return "xarray"
    return "unknown"


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}", flush=True)
        return 1
    records = json.loads(SRC.read_text(encoding="utf-8"))["records"]
    tail_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cell_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    unknown_paths: set[str] = set()

    for row in records:
        granularity = row.get("granularity", "")
        if row.get("agent_role") == "planner" or granularity not in GRANULARITIES:
            continue
        repo = repo_for_path(row["path"])
        if repo == "unknown":
            unknown_paths.add(row["path"])
            continue
        cell_counts[repo][granularity] += 1
        if float(row["d_norm"]) > THRESHOLD:
            tail_counts[repo][granularity] += 1

    summary = {
        repo: {
            "class_tail_cells": tail_counts[repo]["class"],
            "statement_window_tail_cells": tail_counts[repo]["statement_window"],
            "class_cross_role_cells": cell_counts[repo]["class"],
            "statement_window_cross_role_cells": cell_counts[repo]["statement_window"],
        }
        for repo in REPOS
    }
    zero_class = sum(1 for repo in REPOS if summary[repo]["class_tail_cells"] == 0)
    zero_statement = sum(1 for repo in REPOS if summary[repo]["statement_window_tail_cells"] == 0)
    zero_combined = sum(
        1
        for repo in REPOS
        if summary[repo]["class_tail_cells"] + summary[repo]["statement_window_tail_cells"] == 0
    )
    out = {
        "threshold": THRESHOLD,
        "granularities": GRANULARITIES,
        "repos": summary,
        "zero_tail_repo_counts": {
            "class": zero_class,
            "statement_window": zero_statement,
            "combined_class_or_statement_window": zero_combined,
        },
        "unknown_paths": sorted(unknown_paths),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = ["# Cross-role tail cells by repository\n"]
    lines.append(f"Threshold: d_norm > {THRESHOLD}; planner self-comparisons excluded.\n")
    lines.append(f"Zero-tail repos: class {zero_class}/10, statement_window {zero_statement}/10, combined {zero_combined}/10.\n")
    lines.append("\n| Repo | class tail | statement_window tail | class cells | statement_window cells |")
    lines.append("|---|---:|---:|---:|---:|")
    for repo in REPOS:
        s = summary[repo]
        lines.append(
            f"| {repo} | {s['class_tail_cells']} | {s['statement_window_tail_cells']} | "
            f"{s['class_cross_role_cells']} | {s['statement_window_cross_role_cells']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    x = np.arange(len(REPOS))
    class_vals = np.array([summary[repo]["class_tail_cells"] for repo in REPOS])
    stmt_vals = np.array([summary[repo]["statement_window_tail_cells"] for repo in REPOS])

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.bar(x, class_vals, width=0.72, color="#3568a8", label="class")
    ax.bar(x, stmt_vals, width=0.72, bottom=class_vals, color="#d8802f", label="statement window")
    ax.set_ylabel("Cross-role tail cells")
    ax.set_xlabel("Repository")
    ax.set_xticks(x)
    ax.set_xticklabels(REPOS, rotation=35, ha="right")
    ax.set_ylim(0, max((class_vals + stmt_vals).max() + 2, 4))
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.set_title("Cross-role AST tail cells are repository-localized")
    for idx, total in enumerate(class_vals + stmt_vals):
        if total == 0:
            ax.text(idx, 0.12, "0", ha="center", va="bottom", fontsize=8, color="#555555")
        else:
            ax.text(idx, total + 0.2, str(int(total)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    PAPER_OUT.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_OUT, dpi=300, bbox_inches="tight")
    fig.savefig(LOCAL_OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {PAPER_OUT}")
    print(f"wrote {LOCAL_OUT}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
