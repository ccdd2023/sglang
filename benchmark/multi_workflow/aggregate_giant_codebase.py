"""Aggregator for the giant-codebase benchmark.

Reads one or more ``rows.csv`` outputs (one row per task × agent) and
emits a markdown report with:
    - Per-task cached_ratio trend (the headline reuse metric).
    - TTFT speedup vs `prefix_cache_only` baseline (if provided).
    - Pool growth curve (placeholder anchor pool size, hit count).
    - Summary F1 vs baseline.

Usage:
    python -m benchmark.multi_workflow.aggregate_giant_codebase \\
        --runs results/ttft_agenttemplatekv/giant_pandas_pilot5_v4_20260626 \\
              results/ttft_agenttemplatekv/giant_pandas_50_20260626 \\
              results/ttft_agenttemplatekv/giant_pandas_500_20260626 \\
        --baseline results/ttft_agenttemplatekv/giant_pandas_baseline_20260626 \\
        --output results/ttft_agenttemplatekv/giant_pandas_REPORT.md
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Read a rows.csv file written by bench_giant_codebase_reuse.py."""
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def summarize_run(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Compute aggregate metrics over all rows in a single run."""
    if not rows:
        return {"label": label, "rows": 0}

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        # Use task_index if available (driver-written), else fall back to case_id grouping.
        key = r.get("task_index", r.get("case_id", "?"))
        by_task[str(key)].append(r)

    n_tasks = len(by_task)
    n_rows = len(rows)

    # Aggregate per-task.
    task_summaries: list[dict[str, Any]] = []
    for task_key in sorted(by_task.keys(), key=lambda k: safe_int(k, 0)):
        trs = by_task[task_key]
        avg_cached = sum(safe_int(r.get("cached_tokens")) for r in trs) / max(len(trs), 1)
        avg_ratio = sum(safe_float(r.get("cached_ratio")) for r in trs) / max(len(trs), 1)
        sum_prompt = sum(safe_int(r.get("prompt_tokens")) for r in trs)
        sum_cached = sum(safe_int(r.get("cached_tokens")) for r in trs)
        reuse_ratio = sum_cached / max(sum_prompt, 1)
        avg_ttft = sum(safe_float(r.get("ttft_ms")) for r in trs) / max(len(trs), 1)
        sum_ttft = sum(safe_float(r.get("ttft_ms")) for r in trs)
        sum_hits = sum(safe_int(r.get("placeholder_anchor_pool_hit_count")) for r in trs)
        sum_misses = sum(safe_int(r.get("placeholder_anchor_pool_miss_count")) for r in trs)
        pool_size = max(safe_int(r.get("placeholder_anchor_store_entry_count")) for r in trs)
        matched = sum(safe_int(r.get("placeholder_kv_prefill_matched_slots")) for r in trs)
        task_summaries.append(
            {
                "task_index": safe_int(task_key, 0),
                "case_id": trs[0].get("case_id", "?"),
                "avg_cached_tokens": round(avg_cached, 1),
                "avg_cached_ratio": round(avg_ratio, 4),
                "reuse_ratio": round(reuse_ratio, 4),
                "avg_ttft_ms": round(avg_ttft, 2),
                "sum_ttft_ms": round(sum_ttft, 2),
                "sum_prompt_tokens": sum_prompt,
                "sum_cached_tokens": sum_cached,
                "pool_size_at_end": pool_size,
                "anchor_pool_hits": sum_hits,
                "anchor_pool_misses": sum_misses,
                "matched_slots": matched,
                "n_agents": len(trs),
            }
        )

    overall = {
        "label": label,
        "n_tasks": n_tasks,
        "n_rows": n_rows,
        "total_prompt_tokens": sum(t["sum_prompt_tokens"] for t in task_summaries),
        "total_cached_tokens": sum(t["sum_cached_tokens"] for t in task_summaries),
        "total_reuse_ratio": (
            sum(t["sum_cached_tokens"] for t in task_summaries)
            / max(sum(t["sum_prompt_tokens"] for t in task_summaries), 1)
        ),
        "total_workflow_ttft_ms": sum(t["sum_ttft_ms"] for t in task_summaries),
        "avg_workflow_ttft_ms": (
            sum(t["sum_ttft_ms"] for t in task_summaries) / max(n_tasks, 1)
        ),
        "total_anchor_pool_hits": sum(t["anchor_pool_hits"] for t in task_summaries),
        "total_matched_slots": sum(t["matched_slots"] for t in task_summaries),
        "max_pool_size": max((t["pool_size_at_end"] for t in task_summaries), default=0),
        "per_task": task_summaries,
    }
    return overall


def emit_markdown(runs: list[dict[str, Any]], baseline: dict[str, Any] | None, output: Path) -> None:
    """Render a markdown report with tables for each run."""
    lines: list[str] = []
    lines.append("# Giant-Codebase Reuse Benchmark — Report")
    lines.append("")
    lines.append(
        "Headline reuse metrics across the persistent-server multi-agent "
        "runs. Each run loaded N SWE-Smith pandas tasks × 5 agents from a "
        "single sglang server (chunked at `chunk-size` tasks per chunk "
        "to dodge the `_delete_leaf` race)."
    )
    lines.append("")

    # Per-run overview table.
    lines.append("## Per-Run Overview")
    lines.append("")
    lines.append(
        "| Run | Tasks | Rows | Total Prompt Tok | Total Cached Tok | **Reuse Ratio** | "
        "Avg Workflow TTFT (ms) | Anchor Hits | Matched Slots | Max Pool Size |"
    )
    lines.append(
        "|-----|------:|-----:|-----------------:|-----------------:|----------------:|"
        "-----------------------:|------------:|--------------:|--------------:|"
    )
    for r in runs:
        lines.append(
            f"| `{r['label']}` | {r['n_tasks']} | {r['n_rows']} | "
            f"{r['total_prompt_tokens']:,} | {r['total_cached_tokens']:,} | "
            f"**{r['total_reuse_ratio']:.4f}** | {r['avg_workflow_ttft_ms']:.0f} | "
            f"{r['total_anchor_pool_hits']} | {r['total_matched_slots']} | "
            f"{r['max_pool_size']} |"
        )
    lines.append("")

    # Per-task trend for the largest run.
    if runs:
        biggest = max(runs, key=lambda r: r["n_tasks"])
        lines.append(f"## Per-Task Reuse Trend — `{biggest['label']}` (N={biggest['n_tasks']})")
        lines.append("")
        lines.append(
            "| Task Idx | Case ID | Avg Cached Ratio | Reuse Ratio | "
            "Avg Agent TTFT (ms) | Sum Workflow TTFT (ms) | Pool Size | Anchor Hits | Matched |"
        )
        lines.append(
            "|---------:|---------|-----------------:|------------:|"
            "---------------------:|------------------------:|----------:|------------:|--------:|"
        )
        for t in biggest["per_task"]:
            case_short = t["case_id"].split(".")[-1] if "." in t["case_id"] else t["case_id"]
            lines.append(
                f"| {t['task_index']} | `{case_short[:30]}` | "
                f"{t['avg_cached_ratio']:.3f} | {t['reuse_ratio']:.3f} | "
                f"{t['avg_ttft_ms']:.0f} | {t['sum_ttft_ms']:.0f} | "
                f"{t['pool_size_at_end']} | {t['anchor_pool_hits']} | {t['matched_slots']} |"
            )
        lines.append("")

    # Baseline comparison.
    if baseline is not None and runs:
        lines.append("## Baseline Comparison (vs `prefix_cache_only`)")
        lines.append("")
        b_avg = baseline.get("avg_workflow_ttft_ms", 0.0)
        lines.append(
            f"**Baseline** (`prefix_cache_only` mode): "
            f"{baseline['n_tasks']} tasks, {baseline['n_rows']} rows, "
            f"avg workflow TTFT = **{b_avg:.0f} ms/task**, "
            f"reuse = {baseline['total_reuse_ratio']:.4f}."
        )
        lines.append("")
        lines.append(
            "Per-task speedup vs baseline (lower per-task TTFT = more reuse):"
        )
        lines.append("")
        lines.append(
            "| Run | Tasks | Per-Task Avg TTFT (ms) | Baseline (ms/task) | Speedup | "
            "Reuse Ratio (Run) | Reuse Ratio (Baseline) |"
        )
        lines.append(
            "|-----|------:|-----------------------:|-------------------:|--------:|"
            "---------------------:|------------------------:|"
        )
        for r in runs:
            speedup = b_avg / max(r["avg_workflow_ttft_ms"], 1.0) if b_avg > 0 else 0.0
            lines.append(
                f"| `{r['label']}` | {r['n_tasks']} | "
                f"{r['avg_workflow_ttft_ms']:.0f} | {b_avg:.0f} | "
                f"**{speedup:.2f}×** | {r['total_reuse_ratio']:.4f} | "
                f"{baseline['total_reuse_ratio']:.4f} |"
            )
        lines.append("")

    # Pool growth interpretation.
    lines.append("## Pool Growth Interpretation")
    lines.append("")
    lines.append(
        "- `placeholder_anchor_store_entry_count` is the cumulative pool size at the end of each task's last agent. "
        "Monotonic non-decreasing growth within a chunk indicates the placeholder k-NN body is writing new anchors."
    )
    lines.append(
        "- `placeholder_anchor_pool_hit_count` is the per-task sum of k-NN body matches. A non-zero value means "
        "downstream agents found a similar (cos ≥ min_cosine) anchor in the pool from a prior request."
    )
    lines.append(
        "- `placeholder_kv_prefill_matched_slots` is the count of slots whose KV was successfully copied from a pool "
        "entry instead of dense-prefilled. This is the operation that produces real TTFT savings."
    )
    lines.append("")
    lines.append(
        "**Note**: A run may show non-zero `reuse_ratio` from prefix-cache reuse alone even when "
        "anchor hits = 0. This is the **cache-ordering** contribution described in the v44 plan §3.1 — "
        "the KNN body adds additional speedup on top."
    )
    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"[aggregator] wrote report -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories containing rows.csv (each is a single run).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline rows.csv (prefix_cache_only) for speedup comparison.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Markdown report output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_summary: list[dict[str, Any]] = []
    for run_dir in args.runs:
        csv_path = run_dir / "rows.csv"
        if not csv_path.is_file():
            print(f"[aggregator] WARN: {csv_path} not found, skipping")
            continue
        rows = load_rows(csv_path)
        summary = summarize_run(rows, label=run_dir.name)
        runs_summary.append(summary)
        print(
            f"[aggregator] {run_dir.name}: {summary['n_tasks']} tasks, "
            f"{summary['n_rows']} rows, reuse={summary['total_reuse_ratio']:.4f}, "
            f"avg_workflow_ttft={summary['avg_workflow_ttft_ms']:.0f}ms"
        )

    baseline_summary: dict[str, Any] | None = None
    if args.baseline is not None:
        b_path = args.baseline / "rows.csv"
        if b_path.is_file():
            b_rows = load_rows(b_path)
            baseline_summary = summarize_run(b_rows, label=args.baseline.name)
            print(
                f"[aggregator] baseline {args.baseline.name}: {baseline_summary['n_tasks']} tasks, "
                f"avg_workflow_ttft={baseline_summary['avg_workflow_ttft_ms']:.0f}ms"
            )
        else:
            print(f"[aggregator] WARN: baseline {b_path} not found, skipping")

    emit_markdown(runs_summary, baseline_summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
