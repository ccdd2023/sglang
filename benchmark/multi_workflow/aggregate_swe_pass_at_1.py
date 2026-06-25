#!/usr/bin/env python3
"""
Phase 2.1c: SWE-bench pass@1 aggregator.

Reads one or more bench_swe_generated_patch_kvcomm summary.json files and computes
per-mode pass@1. A mode passes for a case iff `candidate_test.returncode == 0`.

Usage:
    python -m benchmark.multi_workflow.aggregate_swe_pass_at_1 \
        --summary results/swe_correctness_baseline_10_*/summary.json \
        --summary results/swe_correctness_v44_10_*/summary.json \
        --out-md results/pass_at_1_compare_<date>.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_summaries(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        if not p.exists():
            print(f"WARN: missing summary {p}", file=sys.stderr)
            continue
        out.append({"path": str(p), "data": json.loads(p.read_text())})
    return out


def summarize_one(summary: dict) -> dict:
    """Return per-mode pass/fail counts from one summary.json."""
    by_mode: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0, "no_patch": 0, "no_test": 0})
    for case in summary["results"]:
        for mode_result in case.get("modes", []):
            mode = mode_result["mode"]
            ct = mode_result.get("candidate_test", {})
            rc = ct.get("returncode") if isinstance(ct, dict) else None
            if not mode_result.get("diff_extracted"):
                by_mode[mode]["no_patch"] += 1
            elif rc == 0:
                by_mode[mode]["pass"] += 1
            elif rc is None:
                # usually "skipped by --skip-candidate-tests" or "no diff extracted"
                by_mode[mode]["no_test"] += 1
            else:
                by_mode[mode]["fail"] += 1
    return dict(by_mode)


def render_table(rows: list[tuple[str, dict, int]]) -> str:
    """rows = [(label, by_mode, total_cases)]"""
    out = []
    headers = ["run", "total_cases"]
    # Discover modes from first non-empty row
    all_modes = sorted({m for _, by_mode, _ in rows for m in by_mode})
    for m in all_modes:
        headers += [f"{m}.pass", f"{m}.fail", f"{m}.no_patch", f"{m}.no_test", f"{m}.pass@1"]
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for label, by_mode, total in rows:
        cells = [label, str(total)]
        for m in all_modes:
            d = by_mode.get(m, {"pass": 0, "fail": 0, "no_patch": 0, "no_test": 0})
            denom = sum(d.values()) or 1
            cells += [str(d["pass"]), str(d["fail"]), str(d["no_patch"]), str(d["no_test"]), f"{d['pass']/denom:.0%}"]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def render_regression(rows: list[tuple[str, dict, int]], baseline_label: str) -> str:
    by_label = {label: by_mode for label, by_mode, _ in rows}
    if baseline_label not in by_label:
        return f"(baseline '{baseline_label}' not found)"
    base = by_label[baseline_label]
    out = [f"Regression vs `{baseline_label}` (pp = percentage points):", ""]
    headers = ["run", "mode", "baseline", "current", "Δ pass"]
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for label, by_mode, total in rows:
        if label == baseline_label:
            continue
        for mode in sorted(set(base) | set(by_mode)):
            b = base.get(mode, {"pass": 0, "fail": 0, "no_patch": 0, "no_test": 0})
            c = by_mode.get(mode, {"pass": 0, "fail": 0, "no_patch": 0, "no_test": 0})
            b_pass = b["pass"] / max(sum(b.values()), 1)
            c_pass = c["pass"] / max(sum(c.values()), 1)
            delta = c_pass - b_pass
            flag = "🚨" if delta < -0.02 else ("✅" if delta >= 0 else "⚠️")
            out.append(f"| {label} | {mode} | {b_pass:.0%} | {c_pass:.0%} | {delta:+.1%} {flag} |")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, action="append", required=True,
                   help="One or more bench_swe summary.json paths (can pass multiple times).")
    p.add_argument("--baseline-label", default=None,
                   help="Label of the baseline run; used for regression computation. Defaults to first.")
    p.add_argument("--out-md", type=Path, default=None, help="Optional Markdown output path.")
    p.add_argument("--out-json", type=Path, default=None, help="Optional JSON output path.")
    args = p.parse_args()

    summaries = load_summaries(args.summary)
    if not summaries:
        print("No valid summaries loaded", file=sys.stderr)
        sys.exit(2)

    rows = []
    for s in summaries:
        # derive label = parent dir basename (e.g. swe_correctness_baseline_10_DATE)
        label = Path(s["path"]).parent.name
        by_mode = summarize_one(s["data"])
        total = sum(v["pass"] + v["fail"] + v["no_patch"] + v["no_test"] for v in by_mode.values()) // max(len(by_mode), 1)
        rows.append((label, by_mode, total))

    baseline = args.baseline_label or rows[0][0]

    md_lines = [
        "# Phase 2.1 SWE-bench pass@1 summary",
        "",
        f"Baseline: **{baseline}**",
        "",
        "## Per-mode pass@1",
        "",
        render_table(rows),
        "",
        "## Regression vs baseline",
        "",
        render_regression(rows, baseline),
        "",
        "Legend: ✅ ≥0pp  ⚠️ 0~-2pp  🚨 < -2pp (handoff §6.5 strict gate is -2pp).",
    ]
    md = "\n".join(md_lines)
    print(md)
    if args.out_md:
        args.out_md.write_text(md)
        print(f"\nWrote Markdown: {args.out_md}", file=sys.stderr)
    if args.out_json:
        args.out_json.write_text(json.dumps(
            [{"label": l, "by_mode": m, "total": t} for l, m, t in rows], indent=2
        ))
        print(f"Wrote JSON: {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()