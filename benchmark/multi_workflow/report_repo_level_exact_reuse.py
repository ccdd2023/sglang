#!/usr/bin/env python3
"""Generate a Markdown report for repo-level exact KV reuse experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_INPUT = Path("results/real_codebase_exact_reuse/repo_dataset_combined_summary.json")
DEFAULT_OUTPUT = Path("results/real_codebase_exact_reuse/repo_dataset_report.md")


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def get_layer_metric(item: dict[str, Any], layer_id: int, key: str) -> float | None:
    for layer in item.get("layers", []):
        if layer.get("layer") == layer_id:
            value = layer.get(key)
            return float(value) if value is not None else None
    return None


def flatten_pairs(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case in data.get("sglang_exact_reuse", {}).get("cases", []):
        for pair in case.get("pairs", []):
            rows.append(
                {
                    "case_id": case.get("case_id", ""),
                    "repo_key": case.get("repo_key", ""),
                    **pair,
                }
            )
    return rows


def summarize_sglang(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        "avg_speedup": mean(float(row.get("speedup_vs_lossless", 0.0)) for row in rows),
        "avg_lossy_cached_tokens": mean(float(row.get("lossy_cached_tokens", 0.0)) for row in rows),
        "avg_lossless_cached_tokens": mean(float(row.get("lossless_cached_tokens", 0.0)) for row in rows),
        "avg_token_f1": mean(float(row.get("token_f1", 0.0)) for row in rows),
        "exact_match_rate": mean(1.0 if row.get("exact_output_match") else 0.0 for row in rows),
    }


def summarize_hf(items: list[dict[str, Any]]) -> dict[str, float]:
    if not items:
        return {}
    token_counts = [float(item.get("tokens", 0.0)) for item in items]
    layer_ids = sorted(
        {
            int(layer["layer"])
            for item in items
            for layer in item.get("layers", [])
            if layer.get("layer") is not None
        }
    )
    summary_layer = 24 if 24 in layer_ids else (layer_ids[-1] if layer_ids else 24)
    layer_k = [v for item in items if (v := get_layer_metric(item, summary_layer, "k_cosine")) is not None]
    layer_v = [v for item in items if (v := get_layer_metric(item, summary_layer, "v_cosine")) is not None]
    return {
        "avg_tokens": mean(token_counts),
        "summary_layer": summary_layer,
        "avg_layer_k_cosine": mean(layer_k) if layer_k else 0.0,
        "min_layer_k_cosine": min(layer_k) if layer_k else 0.0,
        "avg_layer_v_cosine": mean(layer_v) if layer_v else 0.0,
        "min_layer_v_cosine": min(layer_v) if layer_v else 0.0,
    }


def render_report(data: dict[str, Any], source_path: Path) -> str:
    sglang_rows = flatten_pairs(data)
    hf_items = data.get("hf_kv_delta", {}).get("results", [])
    sglang = summarize_sglang(sglang_rows)
    hf = summarize_hf(hf_items)

    lines = [
        "# Repo-Level Exact Code-Base KV Reuse Report",
        "",
        f"Model: `{data.get('model', '')}`",
        f"Generated from: `{source_path}`",
        "",
        "Experiment order: cold lossless baseline without anchor metadata, then planner warmup with anchor metadata, then lossy KVCOMM reuse.",
        "",
        "## Dataset",
        "",
        "| Case | Repo | Files | Total lines |",
        "|---|---|---:|---:|",
    ]
    for case in data.get("cases", []):
        segments = case.get("segments", [])
        total_lines = sum(int(seg.get("lines", 0)) for seg in segments)
        lines.append(
            f"| `{case.get('case_id', '')}` | `{case.get('repo_key', '')}` | {len(segments)} | {total_lines} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Results",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| HF avg reusable segment length | {fmt(hf.get('avg_tokens', 0.0), 1)} tokens |",
            f"| HF layer-{int(hf.get('summary_layer', 24))} key cosine avg/min | {fmt(hf.get('avg_layer_k_cosine', 0.0), 6)} / {fmt(hf.get('min_layer_k_cosine', 0.0), 6)} |",
            f"| HF layer-{int(hf.get('summary_layer', 24))} value cosine avg/min | {fmt(hf.get('avg_layer_v_cosine', 0.0), 6)} / {fmt(hf.get('min_layer_v_cosine', 0.0), 6)} |",
            f"| sglang avg speedup | {fmt(sglang.get('avg_speedup', 0.0), 3)}x |",
            f"| sglang cached tokens, lossless -> lossy | {fmt(sglang.get('avg_lossless_cached_tokens', 0.0), 1)} -> {fmt(sglang.get('avg_lossy_cached_tokens', 0.0), 1)} |",
            f"| Output exact-match rate | {pct(sglang.get('exact_match_rate', 0.0))} |",
            f"| Output token F1 avg | {fmt(sglang.get('avg_token_f1', 0.0), 4)} |",
            "",
            "## sglang Exact-Reuse Runs",
            "",
            "| Case | Agent | cached lossless | cached lossy | speedup | token F1 | Match reason | Matched content |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in sglang_rows:
        meta = row.get("lossy_meta", {})
        matched = str(meta.get("lossy_first_matched_content_signature", ""))
        if matched:
            matched = matched[:12]
        lines.append(
            "| "
            f"`{row.get('case_id', '')}` | "
            f"{row.get('agent', '')} | "
            f"{row.get('lossless_cached_tokens', 0)} | "
            f"{row.get('lossy_cached_tokens', 0)} | "
            f"{fmt(float(row.get('speedup_vs_lossless', 0.0)), 3)}x | "
            f"{fmt(float(row.get('token_f1', 0.0)), 4)} | "
            f"`{meta.get('lossy_first_match_reason', '')}` | "
            f"`{matched}` |"
        )

    risky = [row for row in sglang_rows if float(row.get("token_f1", 0.0)) < 0.6]
    lines.extend(
        [
            "",
            "## Accuracy Risk Cases",
            "",
        ]
    )
    if not risky:
        lines.append("No output pairs had token F1 below 0.6.")
    else:
        lines.extend(["| Case | Agent | token F1 | speedup | Notes |", "|---|---|---:|---:|---|"])
        for row in risky:
            lines.append(
                f"| `{row.get('case_id', '')}` | {row.get('agent', '')} | "
                f"{fmt(float(row.get('token_f1', 0.0)), 4)} | "
                f"{fmt(float(row.get('speedup_vs_lossless', 0.0)), 3)}x | "
                "Lossy output diverged under deterministic decoding; needs task-level pass/fail validation. |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Exact code-content signatures are the reuse gate; AST/anchor fields only locate code-base segments.",
            "- RoPE delta gives high key cosine on real repo files, but values and later-layer keys still reflect upstream-context differences.",
            "- Cached-token gains are large on multi-file repo prompts; output stability varies, so final accuracy claims should use SWE-bench-style pass/fail or patch-level validation rather than token overlap alone.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    report = render_report(data, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
