#!/usr/bin/env python3
"""Generate a Markdown report for coding-structure KV sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
DATA = ROOT / "results" / "coding_structure_kv_sensitivity" / "data"


def fmt(x) -> str:
    if isinstance(x, (int, float)):
        return f"{x:.3f}"
    return str(x)


def table_for(bucket: dict) -> list[str]:
    lines = ["| Bucket | n | mean | p50 | p90 | max |", "|---|---:|---:|---:|---:|---:|"]
    for name, stats in bucket.items():
        lines.append(
            f"| {name} | {stats.get('count', 0)} | {fmt(stats.get('mean', 0))} | "
            f"{fmt(stats.get('p50', 0))} | {fmt(stats.get('p90', 0))} | {fmt(stats.get('max', 0))} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA / "coding_structure_distance_7b.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "coding_structure_kv_sensitivity" / "report.md")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary = payload["summary"]
    cfg = payload["config"]

    lines = [
        "# Coding-Structure KV Sensitivity",
        "",
        "This experiment keeps target code bytes identical and varies only the coding-agent prompt structure around that code. It measures K/V distance on the target code span, not on the whole prompt.",
        "",
        "## Setup",
        "",
        f"- Model: `{cfg['model']}`",
        f"- Canonical cell: `{cfg['canonical']['agent_role']} / {cfg['canonical']['coding_structure']}`",
        f"- Selected layers: `{cfg['selected_layers']}`",
        f"- Variations: `{cfg['n_variations']}`",
        "",
        "## Overall",
        "",
        f"- n = {summary['overall'].get('count', 0)}",
        f"- mean d_norm = {fmt(summary['overall'].get('mean', 0))}",
        f"- p90 d_norm = {fmt(summary['overall'].get('p90', 0))}",
        f"- max d_norm = {fmt(summary['overall'].get('max', 0))}",
        "",
        "## By Coding Structure",
        "",
        *table_for(summary["by_coding_structure"]),
        "",
        "## By Agent Role",
        "",
        *table_for(summary["by_agent_role"]),
        "",
        "## By AST Type",
        "",
        *table_for(summary["by_ast_type"]),
        "",
        "## Worst Cases",
        "",
        "| seg_id | role | structure | d_norm | span_tokens | target_start |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary.get("worst_cases", []):
        lines.append(
            f"| {row['seg_id']} | {row['agent_role']} | {row['coding_structure']} | "
            f"{fmt(row['d_norm'])} | {row['span_tokens']} | {row['target_start']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Hook",
            "",
            "Use this report to justify AgentTemplateKV policy choices: exact-content remains the safety gate, while coding structure controls whether reuse is low-risk, should be prefetched/protected, or should be refused because the target code span is structurally far from the canonical code-first template.",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] wrote {args.out}")


if __name__ == "__main__":
    main()
