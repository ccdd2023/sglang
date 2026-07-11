#!/usr/bin/env python3
"""Phase T1 (True CacheBlend) overhead gate analyzer.

Compares True CacheBlend per-token minipre cost vs R32 baseline at matched setup.
FAIL if per-request minipre overhead p95 > SGLANG_TRUE_CACHEBLEND_OVERHEAD_GATE_P95_MS
(8ms) OR TTFT delta p50 > 30 ms (practical regression gate).
FAIL => Path A infeasible => update CLAUDE.md §6 P3' to FALSIFIED (4th falsification).

Inputs:
  - T1 rows.csv (with SGLANG_TRUE_CACHEBLEND=1)
  - baseline rows.csv (R32 with SGLANG_TRUE_CACHEBLEND=0)

Per-request overhead is computed as:
  overhead_ms = ttft_t1 - ttft_baseline (paired by case_id + agent_id)
  minipre_launches = placeholder_chunk_pool_true_cacheblend_unique_positions delta
                    (per-request, cached once after cache-and-lock, NOT inflated by
                    per-round re-emit)
  per_minipre_ms = overhead_ms / max(minipre_launches, 1)

Decision metrics:
  (a) per_minipre_ms p95 <= 8ms (formal T1 gate)
  (b) TTFT delta p50 <= 30 ms (practical: T1 must not regress requests)

Both must pass.
"""
import csv, json, os, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
BASE = ROOT / "results/scale15_5x5"

GATE_MS = float(os.environ.get("SGLANG_TRUE_CACHEBLEND_OVERHEAD_GATE_P95_MS", "8"))
PRACTICAL_TTFT_GATE_MS = float(os.environ.get("SGLANG_TRUE_CACHEBLEND_PRACTICAL_GATE_MS", "30"))


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


def per_req_minipre_launches(t1_rows):
    """Estimate per-req TRUE minipre launch count.

    The `placeholder_chunk_pool_true_cacheblend_unique_positions` counter is
    process-global. We diff consecutive rows in CSV order: each T1-active
    request contributes (cached_list_size) to the counter. Per-request
    contribution = curr_total - prev_total.

    Caveat: rows are sorted by (case_id, agent_id) which may not match
    completion order; if so, the diffs are noisy. For the T1 pilot we
    have small N (<=9) and identical completion order vs sort order.
    """
    rows = sorted(t1_rows.values(), key=lambda r: (
        r.get("case_id", ""), r.get("agent_id", "")
    ))
    per_req = {}
    prev_total = 0
    for r in rows:
        curr_total = safe_int(r.get("placeholder_chunk_pool_true_cacheblend_unique_positions"))
        delta = curr_total - prev_total
        per_req[(r["case_id"], r["agent_id"])] = max(0, delta)
        prev_total = curr_total
    return per_req


def analyze(t1_csv: Path, baseline_csv: Path, out_md: Path):
    t1 = load_rows(t1_csv)
    baseline = load_rows(baseline_csv)

    per_req_launches = per_req_minipre_launches(t1)

    # Paired by (case_id, agent_id)
    paired = []
    for key, t1_row in t1.items():
        b_row = baseline.get(key)
        if b_row is None:
            continue
        ttft_t1 = safe_float(t1_row.get("ttft_ms"))
        ttft_b = safe_float(b_row.get("ttft_ms"))
        launches = per_req_launches.get(key, 0)
        positions_emitted = safe_int(t1_row.get("placeholder_chunk_pool_true_cacheblend_positions_count"))
        paired.append({
            "key": key,
            "ttft_baseline": ttft_b,
            "ttft_t1": ttft_t1,
            "ttft_delta": ttft_t1 - ttft_b,
            "minipre_launches": launches,
            "positions_emitted_total": positions_emitted,
            "per_minipre_ms": (ttft_t1 - ttft_b) / max(launches, 1),
        })

    if not paired:
        print("ERROR: no paired rows between T1 and baseline")
        return False

    n_paired = len(paired)
    n_with_launches = sum(1 for p in paired if p["minipre_launches"] > 0)
    total_launches = sum(p["minipre_launches"] for p in paired)
    total_emitted = sum(p["positions_emitted_total"] for p in paired)

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

    # TTFT delta for T1-active rows only (where minipre actually launched)
    ttft_deltas_t1 = [p["ttft_delta"] for p in paired if p["minipre_launches"] > 0]
    ttft_p50_t1 = percentile(ttft_deltas_t1, 0.5) if ttft_deltas_t1 else 0.0
    ttft_p95_t1 = percentile(ttft_deltas_t1, 0.95) if ttft_deltas_t1 else 0.0

    # Two-gate decision
    passed_gate = p95 <= GATE_MS and n_with_launches > 0
    passed_practical = ttft_p50_t1 <= PRACTICAL_TTFT_GATE_MS
    passed = passed_gate and passed_practical

    print(f"T1 pilot overhead report")
    print(f"========================")
    print(f"Gate (per-minipre p95): p95 <= {GATE_MS:.1f} ms")
    print(f"Practical gate: TTFT delta p50 <= {PRACTICAL_TTFT_GATE_MS:.1f} ms (T1-active rows)")
    print(f"Paired rows:           {n_paired}")
    print(f"Rows w/ minipre > 0:   {n_with_launches}")
    print(f"Total minipre launches (unique): {total_launches}")
    print(f"Total positions emitted (raw, inflated): {total_emitted}")
    print(f"Avg launches/req:      {total_launches / max(n_with_launches, 1):.1f}")
    print(f"TTFT delta (p50/p95) [all rows]:   {percentile(ttft_deltas, 0.5):.1f} / {percentile(ttft_deltas, 0.95):.1f} ms")
    print(f"TTFT delta (p50/p95) [T1-active]:  {ttft_p50_t1:.1f} / {ttft_p95_t1:.1f} ms")
    if per_minipre_ms:
        print(f"per-minipre ms (p50/p95/p99): {p50:.2f} / {p95:.2f} / {p99:.2f}")
    print()
    print(f"  Gate PASS:       {passed_gate}")
    print(f"  Practical PASS:  {passed_practical}")
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        reasons = []
        if not passed_gate:
            if n_with_launches == 0:
                reasons.append("no rows emitted positions (producer not firing).")
            else:
                reasons.append(f"per-minipre p95 ({p95:.2f} ms) > gate ({GATE_MS:.1f} ms).")
        if not passed_practical:
            reasons.append(f"TTFT p50 regressed {ttft_p50_t1:.0f} ms (> {PRACTICAL_TTFT_GATE_MS:.0f} ms practical gate).")
        print(f"  Reason(s): {', '.join(reasons)}")
        print(f"  Path A infeasible. STOP. Decide Path B or retire.")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# T1 Pilot Overhead Report",
        f"",
        f"**Generated**: {Path(__file__).name}",
        f"**Gate (formal)**: per-minipre p95 <= {GATE_MS:.1f} ms",
        f"**Gate (practical)**: TTFT delta p50 <= {PRACTICAL_TTFT_GATE_MS:.1f} ms (T1-active rows)",
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
        f"| Rows with minipre launches > 0 | {n_with_launches} |",
        f"| Total minipre launches (unique, cached) | {total_launches} |",
        f"| Total positions emitted (raw, inflated) | {total_emitted} |",
        f"| Avg launches/req | {total_launches / max(n_with_launches, 1):.1f} |",
        f"| TTFT delta p50 (all) (ms) | {percentile(ttft_deltas, 0.5):.1f} |",
        f"| TTFT delta p95 (all) (ms) | {percentile(ttft_deltas, 0.95):.1f} |",
        f"| TTFT delta p50 (T1-active) (ms) | {ttft_p50_t1:.1f} |",
        f"| TTFT delta p95 (T1-active) (ms) | {ttft_p95_t1:.1f} |",
    ]
    if per_minipre_ms:
        lines += [
            f"| per-minipre ms p50 | {p50:.2f} |",
            f"| per-minipre ms p95 | {p95:.2f} |",
            f"| per-minipre ms p99 | {p99:.2f} |",
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
            f"### Reason",
            f"",
            f"- {', '.join(reasons) if reasons else 'unknown'}",
            f"",
            f"### Action",
            f"",
            f"- Write ABLATION_TRUE_CACHEBLEND.md with full NEGATIVE report",
            f"- Update CLAUDE.md §6 P3' to `FALSIFIED at policy layer (4th falsification)`",
            f"- Add memory pointer: `true-cacheblend-phase-t1-fail-2026-07-11.md`",
            f"- Decision: Path B (5-8 days) OR retire P3' entirely",
        ]
    else:
        lines += [
            f"### Action",
            f"",
            f"- Proceed to Phase T2 (HKVD signal wiring)",
            f"- Use control_flow vs data_flow axis (Phase 4 multi-signal POSITIVE)",
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