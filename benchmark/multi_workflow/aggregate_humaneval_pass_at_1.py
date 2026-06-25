#!/usr/bin/env python3
"""Aggregate HumanEval-lite pass@1 baseline vs v44.

Usage:
    python -m benchmark.multi_workflow.aggregate_humaneval_pass_at_1 \\
        --baseline results/humaneval_baseline_lite_<date>/summary.json \\
        --v44 results/humaneval_v44_lite_<date>/summary.json
"""
import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--v44", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    base = load_summary(args.baseline)
    v44 = load_summary(args.v44)

    base_results = {r["task_id"]: r for r in base.get("results", [])}
    v44_results = {r["task_id"]: r for r in v44.get("results", [])}
    common = sorted(set(base_results) & set(v44_results))

    print(f"# HumanEval-lite pass@1 — baseline vs v44")
    print(f"\nbaseline: {args.baseline} (n={len(base_results)})")
    print(f"v44:      {args.v44} (n={len(v44_results)})")
    print(f"common task_ids: {len(common)}")

    # Per-task diff
    diff_rows = []
    for tid in common:
        b = base_results[tid]
        v = v44_results[tid]
        diff_rows.append({
            "task_id": tid,
            "baseline_passed": b.get("passed", False),
            "v44_passed": v.get("passed", False),
            "delta": ("regress" if b.get("passed") and not v.get("passed")
                      else "improve" if not b.get("passed") and v.get("passed")
                      else "same"),
            "baseline_sha": b.get("completion_sha", "?"),
            "v44_sha": v.get("completion_sha", "?"),
        })

    n_pass_base = sum(1 for r in diff_rows if r["baseline_passed"])
    n_pass_v44 = sum(1 for r in diff_rows if r["v44_passed"])
    pass1_base = n_pass_base / len(diff_rows) if diff_rows else 0.0
    pass1_v44 = n_pass_v44 / len(diff_rows) if diff_rows else 0.0
    delta = pass1_v44 - pass1_base

    # Same SHA per task (model determinism check)
    same_sha = sum(1 for r in diff_rows if r["baseline_sha"] == r["v44_sha"])

    print(f"\n## pass@1 summary")
    print(f"\n| run | pass@1 | n_pass | n_total |")
    print(f"|---|---|---|---|")
    print(f"| baseline | {pass1_base:.2%} | {n_pass_base} | {len(diff_rows)} |")
    print(f"| v44 | {pass1_v44:.2%} | {n_pass_v44} | {len(diff_rows)} |")
    print(f"\n**Regression (v44 - baseline) = {delta:+.2%}**")
    print(f"§6.6 gate: regression ≤ 3 pp → {'✅ PASS' if delta >= -0.03 else '❌ FAIL'}")

    print(f"\n## Model determinism (same SHA baseline vs v44)")
    print(f"{same_sha}/{len(diff_rows)} tasks produced byte-identical completion across modes.")

    print(f"\n## Per-task table")
    print(f"| task_id | baseline | v44 | delta | baseline_sha | v44_sha | same_sha |")
    print(f"|---|---|---|---|---|---|---|")
    for r in diff_rows:
        print(f"| {r['task_id']} | {'✅' if r['baseline_passed'] else '❌'} | "
              f"{'✅' if r['v44_passed'] else '❌'} | {r['delta']} | "
              f"{r['baseline_sha']} | {r['v44_sha']} | "
              f"{'✅' if r['baseline_sha'] == r['v44_sha'] else '❌'} |")

    summary = {
        "baseline_path": str(args.baseline),
        "v44_path": str(args.v44),
        "n_common": len(common),
        "n_pass_baseline": n_pass_base,
        "n_pass_v44": n_pass_v44,
        "pass1_baseline": pass1_base,
        "pass1_v44": pass1_v44,
        "regression_pp": delta * 100,
        "pass6_6_gate": "PASS" if delta >= -0.03 else "FAIL",
        "n_same_sha": same_sha,
        "diff_rows": diff_rows,
    }
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())