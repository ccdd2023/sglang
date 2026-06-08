#!/usr/bin/env python3
"""Generate a Markdown report for AST-granularity KV sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
DATA = ROOT / "results" / "ast_granularity_kv_sensitivity" / "data"


ORDER = ["file_prefix", "class", "function", "method", "control_block", "statement_window"]


def fmt(x) -> str:
    if isinstance(x, (int, float)):
        return f"{x:.3f}"
    return str(x)


def table_for(bucket: dict, ordered: bool = False) -> list[str]:
    lines = [
        "| Bucket | spans | n | mean toks | retention toks | mean d_norm | p90 d_norm | max d_norm | weighted d_norm | reuse score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    keys = [key for key in ORDER if key in bucket] if ordered else sorted(bucket)
    for name in keys:
        stats = bucket[name]
        lines.append(
            f"| {name} | {stats.get('unique_spans', 0)} | {stats.get('count', 0)} | "
            f"{fmt(stats.get('mean_span_tokens', 0))} | {fmt(stats.get('device_retention_cost_tokens', 0))} | "
            f"{fmt(stats.get('mean', 0))} | "
            f"{fmt(stats.get('p90', 0))} | {fmt(stats.get('max', 0))} | "
            f"{fmt(stats.get('token_weighted_d_norm', 0))} | {fmt(stats.get('reuse_score', 0))} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA / "ast_granularity_distance_7b.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "ast_granularity_kv_sensitivity" / "report.md")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary = payload["summary"]
    cfg = payload["config"]

    lines = [
        "# AST-Granularity KV Sensitivity",
        "",
        "This experiment keeps every code object byte-identical and varies the AST granularity used as the reuse object. The canonical cache source is the planner view of the same exact span; coder and reviewer prompts measure whether that span remains a stable and useful K/V anchor.",
        "",
        "## Setup",
        "",
        f"- Model: `{cfg['model']}`",
        f"- Canonical cell: `{cfg['canonical']['agent_role']}` on the same exact code object",
        f"- Selected layers: `{cfg['selected_layers']}`",
        f"- Spans: `{cfg['n_spans']}`",
        f"- Variations: `{cfg['n_variations']}`",
        "",
        "## Overall",
        "",
        f"- n = {summary['overall'].get('count', 0)}",
        f"- mean d_norm = {fmt(summary['overall'].get('mean', 0))}",
        f"- p90 d_norm = {fmt(summary['overall'].get('p90', 0))}",
        f"- max d_norm = {fmt(summary['overall'].get('max', 0))}",
        "",
        "## By AST Granularity",
        "",
        *table_for(summary["by_granularity"], ordered=True),
        "",
        "## By Token Bin",
        "",
        *table_for(summary["by_token_bin"]),
        "",
        "## By Agent Role",
        "",
        *table_for(summary["by_agent_role"]),
        "",
        "## Worst Cases",
        "",
        "| span_id | granularity | role | path | lines | d_norm | span_tokens | target_start |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in summary.get("worst_cases", []):
        lines.append(
            f"| {row['span_id']} | {row['granularity']} | {row['agent_role']} | {row['path']} | "
            f"{row['lines']} | {fmt(row['d_norm'])} | {row['span_tokens']} | {row['target_start']} |"
        )

    lines.extend(
        [
            "",
            "## Regularities",
            "",
            "- Exact content remains the non-negotiable reuse gate; AST granularity only chooses which exact byte span becomes the reusable object.",
            "- Function and method spans form the best default policy unit: low mean/p90 distance, useful token payload, bounded retention cost, and natural alignment with coding-agent edits.",
            "- Statement windows can be stable on average, but their semantic boundary is weak and their tail risk is high; use them as fallback exact spans, not as the primary template object.",
            "- Class spans are useful when downstream agents repeatedly inspect related methods, but the higher p90 distance means they should require DAG evidence and TTL protection.",
            "- File prefixes offer the largest theoretical saving, but retention cost is an order of magnitude larger; protect them only for stable codebase-front blocks with strong future-use evidence.",
        ],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] wrote {args.out}")


if __name__ == "__main__":
    main()
