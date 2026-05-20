#!/usr/bin/env python3
"""
KVFlow Optimal Benchmark Results Analyzer

Compares hicache (LRU) vs kvflow (Priority) results and generates summary report.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def compare_configs(result_dir: str) -> Dict:
    """Compare all configurations in the result directory."""

    result_path = Path(result_dir)
    hicache_files = sorted(result_path.glob("kvflow_opt_hicache_*.json"))
    kvflow_files = sorted(result_path.glob("kvflow_opt_kvflow_*.json"))

    results = {
        "hicache": [],
        "kvflow": [],
        "comparisons": []
    }

    # Load hicache results
    for f in hicache_files:
        data = load_json(f)
        results["hicache"].append({
            "file": f.name,
            "config": data["config"],
            "aggregate": data["aggregate"],
            "round_summaries": data["round_summaries"],
        })

    # Load kvflow results
    for f in kvflow_files:
        data = load_json(f)
        results["kvflow"].append({
            "file": f.name,
            "config": data["config"],
            "aggregate": data["aggregate"],
            "round_summaries": data["round_summaries"],
        })

    # Match and compare
    for h in results["hicache"]:
        h_config = h["config"]
        h_wf = h_config["num_workflows"]
        h_ag = h_config["agents_per_workflow"]

        # Find matching kvflow
        for k in results["kvflow"]:
            k_config = k["config"]
            if k_config["num_workflows"] == h_wf and k_config["agents_per_workflow"] == h_ag:
                comparison = compare_two(h, k)
                results["comparisons"].append(comparison)

    return results


def compare_two(hicache: dict, kvflow: dict) -> dict:
    """Compare two specific configurations."""

    h_agg = hicache["aggregate"]
    k_agg = kvflow["aggregate"]

    # Calculate speedups
    ttft_speedup = h_agg["ttft_avg_ms"] / k_agg["ttft_avg_ms"] if k_agg["ttft_avg_ms"] > 0 else 0
    e2e_speedup = h_agg["e2e_avg_ms"] / k_agg["e2e_avg_ms"] if k_agg["e2e_avg_ms"] > 0 else 0
    round_e2e_speedup = h_agg["round_e2e_avg_ms"] / k_agg["round_e2e_avg_ms"] if k_agg["round_e2e_avg_ms"] > 0 else 0

    return {
        "scenario": f"{hicache['config']['num_workflows']}wf × {hicache['config']['agents_per_workflow']}ag",
        "hicache": {
            "ttft_avg_ms": h_agg["ttft_avg_ms"],
            "e2e_avg_ms": h_agg["e2e_avg_ms"],
            "round_e2e_avg_ms": h_agg["round_e2e_avg_ms"],
            "warmup_ttft_avg_ms": h_agg["warmup_ttft_avg_ms"],
        },
        "kvflow": {
            "ttft_avg_ms": k_agg["ttft_avg_ms"],
            "e2e_avg_ms": k_agg["e2e_avg_ms"],
            "round_e2e_avg_ms": k_agg["round_e2e_avg_ms"],
            "warmup_ttft_avg_ms": k_agg["warmup_ttft_avg_ms"],
        },
        "speedup": {
            "ttft": ttft_speedup,
            "e2e": e2e_speedup,
            "round_e2e": round_e2e_speedup,
        },
        "round_details": compare_round_details(hicache, kvflow),
    }


def compare_round_details(hicache: dict, kvflow: dict) -> List[dict]:
    """Compare round-by-round details."""

    h_rounds = hicache["round_summaries"]
    k_rounds = kvflow["round_summaries"]

    details = []
    for round_name in sorted(h_rounds.keys(), key=lambda x: int(x.split('_')[1]))[1:]:  # Skip warmup
        h_data = h_rounds.get(round_name, {})
        k_data = k_rounds.get(round_name, {})

        h_ttft = h_data.get("avg_ttft", 0)
        k_ttft = k_data.get("avg_ttft", 0)
        ttft_ratio = k_ttft / h_ttft if h_ttft > 0 else 1.0

        details.append({
            "round": round_name,
            "hicache_ttft": h_ttft,
            "kvflow_ttft": k_ttft,
            "kvflow_vs_hicache": ttft_ratio,
        })

    return details


def print_report(results: Dict):
    """Print formatted report."""

    print("=" * 80)
    print("KVFlow Optimal Scenario Benchmark Report")
    print("=" * 80)

    if not results["comparisons"]:
        print("\nNo comparison data available yet.")
        print(f"Hicache results: {len(results['hicache'])}")
        print(f"KVFlow results: {len(results['kvflow'])}")
        return

    for comp in results["comparisons"]:
        print(f"\n{'='*80}")
        print(f"Scenario: {comp['scenario']}")
        print(f"{'='*80}")

        print("\n| Metric | Hicache (LRU) | KVFlow (Priority) | Speedup |")
        print("|--------|---------------|------------------|---------|")
        print(f"| TTFT (avg) | {comp['hicache']['ttft_avg_ms']:.2f} ms | {comp['kvflow']['ttft_avg_ms']:.2f} ms | {comp['speedup']['ttft']:.3f}x |")
        print(f"| E2E (avg) | {comp['hicache']['e2e_avg_ms']:.2f} ms | {comp['kvflow']['e2e_avg_ms']:.2f} ms | {comp['speedup']['e2e']:.3f}x |")
        print(f"| Round E2E | {comp['hicache']['round_e2e_avg_ms']:.2f} ms | {comp['kvflow']['round_e2e_avg_ms']:.2f} ms | {comp['speedup']['round_e2e']:.3f}x |")
        print(f"| Warmup TTFT | {comp['hicache']['warmup_ttft_avg_ms']:.2f} ms | {comp['kvflow']['warmup_ttft_avg_ms']:.2f} ms | - |")

        # Cache speedup (vs warmup)
        h_speedup = comp['hicache']['warmup_ttft_avg_ms'] / comp['hicache']['ttft_avg_ms'] if comp['hicache']['ttft_avg_ms'] > 0 else 0
        k_speedup = comp['kvflow']['warmup_ttft_avg_ms'] / comp['kvflow']['ttft_avg_ms'] if comp['kvflow']['ttft_avg_ms'] > 0 else 0
        print(f"| Cache Speedup | {h_speedup:.3f}x | {k_speedup:.3f}x | - |")

        print("\nRound-by-Round TTFT Comparison:")
        print("| Round | Hicache TTFT | KVFlow TTFT | KVFlow/Hicache |")
        print("|-------|-------------|-------------|----------------|")
        for rd in comp["round_details"]:
            ratio = rd["kvflow_vs_hicache"]
            indicator = " " if ratio > 0.95 else "!"
            print(f"| {rd['round']} | {rd['hicache_ttft']:.2f}ms | {rd['kvflow_ttft']:.2f}ms | {ratio:.3f}{indicator} |")

        # Interpretation
        print("\nInterpretation:")
        if comp['speedup']['ttft'] > 1.05:
            print("  KVFlow shows BETTER TTFT performance (Priority is working)")
        elif comp['speedup']['ttft'] < 0.95:
            print("  KVFlow shows WORSE TTFT performance (Priority overhead)")
        else:
            print("  No significant TTFT difference between LRU and Priority")

        if comp['speedup']['round_e2e'] > 1.05:
            print("  KVFlow shows BETTER Round E2E performance")
        elif comp['speedup']['round_e2e'] < 0.95:
            print("  KVFlow shows WORSE Round E2E performance")
        else:
            print("  No significant Round E2E difference")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze KVFlow benchmark results")
    parser.add_argument(
        "--result-dir",
        type=str,
        default="/home/comp/25480812/logs/kvflow-8b/results",
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for report (default: print to stdout)"
    )
    args = parser.parse_args()

    results = compare_configs(args.result_dir)
    report = generate_text_report(results)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


def generate_text_report(results: Dict) -> str:
    """Generate a text report from results."""
    import io
    import sys

    # Capture print output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    print_report(results)

    report = sys.stdout.getvalue()
    sys.stdout = old_stdout

    return report


if __name__ == "__main__":
    main()
