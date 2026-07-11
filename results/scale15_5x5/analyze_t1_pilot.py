#!/usr/bin/env python3
"""Phase T1 (True CacheBlend) overhead gate analyzer.

Compares True CacheBlend per-token minipre cost vs R32 baseline at matched setup.
PASS if per-request minipre overhead p95 <= SGLANG_TRUE_CACHEBLEND_OVERHEAD_GATE_P95_MS (8ms).
FAIL => Path A infeasible => write ABLATION_TRUE_CACHEBLEND.md "Phase T1 overhead fail"
      => update CLAUDE.md §6 P3' to FALSIFIED (4th falsification)
      => memory pointer.

Inputs:
  - T1 rows.csv (with SGLANG_TRUE_CACHEBLEND=1)
  - baseline rows.csv (R32 with SGLANG_TRUE_CACHEBLEND=0)
  - gate from env SGLANG_TRUE_CACHEBLEND_OVERHEAD_GATE_P95_MS (default 8ms)

Per-request overhead is computed as:
  overhead_ms = ttft_t1 - ttft_baseline (paired by case_id + agent_id)
  minipre_launches = placeholder_chunk_pool_true_cacheblend_positions_count (per request)
  per_minipre_ms = overhead_ms / max(minipre_launches, 1)

Decision metric: per_minipre_ms p95 across all requests where minipre_launches > 0.
"""
import csv, json, os, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
BASE = ROOT / "results/scale15_5x5"

GATE_MS = float(os.environ.get("SGLANG_TRUE_CACHEBLEND_OVERHEAD_GATE_P95_MS", "8"))


def load_rows(path: Path):
    """Load rows.csv; return dict (case_id, agent_id) -> row dict."""
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            key = (row["case_id"], row["agent_id"])
            out[key] = row
    return out


def safe_float(s, default=0.0):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def safe_int(s, default=0):
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def analyze(t1_csv: Path, baseline_csv: Path, out_md: Path):
    t1 = load_rows(t1_csv)
    baseline = load_rows(baseline_csv)

    # Paired by (case_id, agent_id)
    paired = []
    for key, t1_row in t1.items():
        b_row = baseline.get(key)
        if b_row is None:
            continue
        ttft_t1 = safe_float(t1_row.get("ttft_ms"))
        ttft_b = safe_float(b_row.get("ttft_ms"))
        positions = safe_int(t1_row.get("placeholder_chunk_pool_true_cacheblend_positions_count"))
        paired.append({
            "key": key,
            "ttft_baseline": ttft_b,
            "ttft_t1": ttft_t1,
            "ttft_delta": ttft_t1 - ttft_b,
            "minipre_launches": positions,
            "per_minipre_ms": (ttft_t1 - ttft_b) / max(positions, 1),
        })

    if not paired:
        print("ERROR: no paired rows between T1 and baseline")
        return False

    n_paired = len(paired)
    n_with_positions = sum(1 for p in paired if p["minipre_launches"] > 0)
    total_minipre = sum(p["minipre_launches"] for p in paired)

    ttft_deltas = [p["ttft_delta"] for p in paired]
    per_minipre_ms = [p["per_minipre_ms"] for p in paired if p["minipre_launches"] > 0]

    def percentile(xs, q):
        if not xs:
            return 0.0
        xs = sorted(xs)
        idx = max(0, min(len(xs) - 1, int(q * (len(xs) - 1))))
        return xs[idx]

    p50 = percentile(per_minipre_ms, 0.5) if per_minipre_ms else 0.0
    p95 = percentile(per_minipre_ms, 0.95) if per_minipre_ms else 0.0
    p99 = percentile(per_minipre_ms, 0.99) if per_minipre_ms else 0.0

    # Decision
    passed = p95 <= GATE_MS and n_with_positions > 0

    print(f"T1 pilot overhead report")
    print(f"========================")
    print(f"Gate: p95 per-minipre ms <= {GATE_MS:.1f}")
    print(f"Paired rows:           {n_paired}")
    print(f"Rows w/ positions > 0: {n_with_positions}")
    print(f"Total minipre launches: {total_minipre}")
    print(f"Avg launches/req:      {total_minipre / max(n_paired, 1):.1f}")
    print(f"TTFT delta (p50/p95):  {percentile(ttft_deltas, 0.5):.1f} / {percentile(ttft_deltas, 0.95):.1f} ms")
    if per_minipre_ms:
        print(f"per-minipre ms (p50/p95/p99): {p50:.3f} / {p95:.3f} / {p99:.3f}")
    print()
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        if n_with_positions == 0:
            print(f"  Reason: no rows emitted positions (producer not firing).")
            print(f"  Likely cause: SGLANG_TRUE_CACHEBLEND=1 not set or chunk pool not active.")
        else:
            print(f"  Reason: per-minipre p95 ({p95:.3f} ms) > gate ({GATE_MS:.1f} ms).")
            print(f"  Path A infeasible. STOP. Decide Path B or retire.")

    # Write markdown report
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# T1 Pilot Overhead Report",
        f"",
        f"**Generated**: {Path(__file__).name}",
        f"**Gate**: per-minipre p95 <= {GATE_MS:.1f} ms (`SGLANG_TRUE_CACHEBLEND_OVERHEAD_GATE_P95_MS`)",
        f"",
        f"## Inputs",
        f"",
        f"- T1 rows: `{t1_csv}`",
        f"- Baseline rows: `{baseline_csv}`",
        f"",
        f"## Metrics",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Paired rows | {n_paired} |",
        f"| Rows with positions > 0 | {n_with_positions} |",
        f"| Total minipre launches | {total_minipre} |",
        f"| Avg launches/req | {total_minipre / max(n_paired, 1):.1f} |",
        f"| TTFT delta p50 (ms) | {percentile(ttft_deltas, 0.5):.1f} |",
        f"| TTFT delta p95 (ms) | {percentile(ttft_deltas, 0.95):.1f} |",
    ]
    if per_minipre_ms:
        lines += [
            f"| per-minipre ms p50 | {p50:.3f} |",
            f"| per-minipre ms p95 | {p95:.3f} |",
            f"| per-minipre ms p99 | {p99:.3f} |",
        ]
    lines += [
        f"",
        f"## Verdict",
        f"",
        f"**{'PASS' if passed else 'FAIL'}**",
        f"",
    ]
    if not passed:
        lines += [
            f"### Action",
            f"",
            f"- Write ABLATION_TRUE_CACHEBLEND.md with full NEGATIVE report",
            f"- Update CLAUDE.md §6 P3' to `FALSIFIED at policy layer (4th falsification)`",
            f"- Add memory pointer",
            f"- Decision: Path B (5-8 days) OR retire P3' entirely",
        ]
    else:
        lines += [
            f"### Action",
            f"",
            f"- Proceed to Phase T2 (HKVD signal wiring)",
            f"- Use control_flow vs data_flow HKVD labels (Phase 4 multi-signal POSITIVE)",
        ]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Report written: {out_md}")
    return passed


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: analyze_t1_pilot.py <t1_rows.csv> <baseline_rows.csv> [out.md]")
        sys.exit(1)
    t1_csv = Path(sys.argv[1])
    baseline_csv = Path(sys.argv[2])
    out_md = Path(sys.argv[3]) if len(sys.argv) > 3 else BASE / "t1_pilot_report.md"
    ok = analyze(t1_csv, baseline_csv, out_md)
    sys.exit(0 if ok else 2)