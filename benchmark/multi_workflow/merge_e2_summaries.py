#!/usr/bin/env python3
"""Merge E2 partial summary.json files into a single final summary.

Usage:
    python merge_e2_summaries.py \
        results/coding_kvflow_prefetch/qwen2_5_7b_100/summary.json \
        results/coding_kvflow_prefetch/qwen2_5_7b_100_part2/summary.json \
        results/coding_kvflow_prefetch/qwen2_5_7b_100_part3/summary.json \
        --out results/coding_kvflow_prefetch/qwen2_5_7b_100/summary.json

Deduplicates by instance_id (keeps last occurrence) and recomputes mode_summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

MODES = [
    "baseline_prefix_cache_only",
    "kvflow_prefix_only",
    "kvflow_prefix_plus_codebase_prefetch",
    "kvcomm_lossy_plus_codebase_prefetch",
]


def compute_mode_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for case in results:
        for row in case["modes"]:
            by_mode[row["mode"]].append(row)
    stats = {}
    for mode, rows in by_mode.items():
        if not rows:
            continue
        latencies = [float(r["elapsed_ms"]) for r in rows]
        cached = [float(r["cached_tokens"]) for r in rows]
        f1s = [float(r["output_token_f1_vs_baseline"]) for r in rows]
        stats[mode] = {
            "n": len(rows),
            "avg_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p50_latency_ms": statistics.median(latencies),
            "p90_latency_ms": sorted(latencies)[int(len(latencies) * 0.9)] if len(latencies) >= 10 else max(latencies),
            "avg_cached_tokens": statistics.mean(cached),
            "avg_prefetch_queued_tokens": statistics.mean(float(r["codebase_prefetch_queued_tokens"]) for r in rows),
            "avg_prefetch_matched_tokens": statistics.mean(float(r["codebase_prefetch_matched_tokens"]) for r in rows),
            "avg_prefetch_hints": statistics.mean(float(r["codebase_prefetch_hint_count"]) for r in rows),
            "prefetch_success_rate": statistics.mean(
                1.0 if int(r["codebase_prefetch_success_count"] or 0) > 0 else 0.0
                for r in rows
            ),
            "prefetch_device_hit_rate": statistics.mean(
                1.0 if int(r["codebase_prefetch_device_hit_count"] or 0) > 0 else 0.0
                for r in rows
            ),
            "exact_content_hit_rate": statistics.mean(
                1.0 if r.get("lossy_match_reason") == "exact_code_content_signature" else 0.0
                for r in rows
            ),
            "avg_token_f1_vs_baseline": statistics.mean(f1s),
        }
        ttft_rows = [r for r in rows if r.get("ttft_ms") is not None]
        if ttft_rows:
            ttft_values = [float(r["ttft_ms"]) for r in ttft_rows]
            stats[mode]["avg_ttft_ms"] = statistics.mean(ttft_values)
            stats[mode]["median_ttft_ms"] = statistics.median(ttft_values)
            stats[mode]["p90_ttft_ms"] = (
                sorted(ttft_values)[int(len(ttft_values) * 0.9)]
                if len(ttft_values) >= 10
                else max(ttft_values)
            )
    return stats


def merge_summaries(paths: list[Path], out_path: Path) -> None:
    all_results: list[dict[str, Any]] = []
    base_meta: dict[str, Any] = {}

    for p in paths:
        if not p.exists():
            print(f"  [skip] {p} does not exist", file=sys.stderr)
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if not base_meta:
            base_meta = {
                "model": data.get("model", ""),
                "dataset": data.get("dataset", ""),
                "manifest": data.get("manifest", ""),
                "hicache_storage_backend": data.get("hicache_storage_backend", "disabled"),
                "hierarchical_cache": data.get("hierarchical_cache", True),
                "modes": data.get("modes", MODES),
            }
        all_results.extend(data.get("results", []))
        print(f"  [loaded] {p}: {len(data.get('results', []))} cases")

    # Deduplicate by instance_id, keeping last occurrence
    seen: dict[str, int] = {}
    for i, r in enumerate(all_results):
        iid = r["instance_id"]
        seen[iid] = i
    deduped = [all_results[i] for i in sorted(seen.values())]
    print(f"  [dedup] {len(all_results)} -> {len(deduped)} unique cases")

    mode_summary = compute_mode_summary(deduped)

    merged = {
        **base_meta,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "results": deduped,
        "failed_reason": None,
        "mode_summary": mode_summary,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  [saved] {out_path}")

    # Also write prefetch_summary.json alongside
    ps_path = out_path.parent / "prefetch_summary.json"
    with open(ps_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  [saved] {ps_path}")

    # Write prefetch_table.csv (consumed by paper/scripts/generate_paper_figures.py)
    csv_rows = []
    for case in deduped:
        for row in case["modes"]:
            csv_rows.append(
                {
                    "instance_id": case["instance_id"],
                    "repo": case["repo"],
                    **{k: v for k, v in row.items() if k != "raw_metadata"},
                }
            )
    csv_path = out_path.parent / "prefetch_table.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"  [saved] {csv_path} ({len(csv_rows)} rows)")

    # Print summary table
    print("\n  === Merged Mode Summary ===")
    print(f"  {'mode':<45} {'n':>5} {'avg_lat':>9} {'p50':>9} {'p90':>9} {'avg_cache':>10} {'exact_hit':>10} {'avg_f1':>8}")
    for mode in MODES:
        if mode not in mode_summary:
            continue
        s = mode_summary[mode]
        print(
            f"  {mode:<45} {s['n']:>5} {s['avg_latency_ms']:>9.1f} "
            f"{s.get('p50_latency_ms', s['median_latency_ms']):>9.1f} "
            f"{s.get('p90_latency_ms', 0):>9.1f} "
            f"{s['avg_cached_tokens']:>10.1f} "
            f"{s['exact_content_hit_rate']:>10.2f} "
            f"{s['avg_token_f1_vs_baseline']:>8.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge E2 partial summaries")
    ap.add_argument("inputs", nargs="+", type=Path, help="Partial summary.json paths")
    ap.add_argument("--out", type=Path, required=True, help="Output summary.json path")
    args = ap.parse_args()
    merge_summaries(args.inputs, args.out)


if __name__ == "__main__":
    main()
