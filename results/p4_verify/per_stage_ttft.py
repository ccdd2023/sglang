#!/usr/bin/env python3
"""Per-stage TTFT breakdown: R32_f015 vs lossless (post-P4-fix, 3-case each).

Uses the 6 newly-unblocked fields (P4 fix) to decompose where R32's TTFT
goes vs lossless. This is the direct value demonstration of the P4 fix.

Stages (from get_ttft_breakdown_ms):
  tokenize_ms              - API-server tokenization (both configs)
  radix_prefix_ms          - radix L1 prefix match (both)
  chunk_plan_ms            - AST chunk plan (R32 only; lossless=0)
  copy_ms                  - chunk KV copy GPU->slot (R32 only)
  gap_prefill_ms           - gap-zero prefill (R32 only)
  head_recompute_early_ms  - head RoPE recompute, early chunks (R32 only)
  head_recompute_late_ms   - head RoPE recompute, late chunks (R32 only)

Lossless path has no chunk_plan/copy/gap/head_recompute (it copies the whole
slot losslessly), so those stages are 0 - the contrast shows exactly what R32
spends its time on.
"""
import csv
import sys
from pathlib import Path

ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")

CONFIGS = {
    "lossless": ROOT / "results/p4_verify/lossless_3case/rows.csv",
    "R32_f015": ROOT / "results/p4_verify/r32_f015_3case/rows.csv",
}

STAGES = [
    "ttft_tokenize_ms", "ttft_radix_prefix_ms", "ttft_chunk_plan_ms",
    "ttft_copy_ms", "ttft_gap_prefill_ms", "ttft_head_recompute_early_ms",
    "ttft_head_recompute_late_ms",
]


def load(path):
    if not path.exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def stage_stats(rows):
    """Return {stage: (mean, nonzero/total)}."""
    out = {}
    for s in STAGES:
        vals = [float(r[s]) for r in rows if r.get(s)]
        nz = sum(1 for v in vals if v != 0)
        mean = sum(vals) / len(vals) if vals else 0.0
        out[s] = (mean, nz, len(vals))
    return out


def main():
    data = {}
    for label, path in CONFIGS.items():
        rows = load(path)
        if rows is None:
            print(f"{label}: NOT FOUND ({path})")
            continue
        data[label] = stage_stats(rows)
        print(f"{label}: {len(rows)} rows loaded")

    if len(data) < 2:
        print("\nneed both configs; aborting")
        return

    # Per-stage table
    print(f"\n{'stage':<32} {'lossless(ms)':>14} {'R32_f015(ms)':>14} {'R32-only?':>10}")
    print("-" * 75)
    totals = {label: 0.0 for label in data}
    for s in STAGES:
        ls = data["lossless"][s][0]
        r32 = data["R32_f015"][s][0]
        r32_only = "R32-only" if ls == 0 and r32 > 0 else ""
        print(f"{s:<32} {ls:>14.3f} {r32:>14.3f} {r32_only:>10}")
        totals["lossless"] += ls
        totals["R32_f015"] += r32
    print("-" * 75)
    print(f"{'SUM (breakdown stages)':<32} {totals['lossless']:>14.3f} {totals['R32_f015']:>14.3f}")

    # Also show actual TTFT (ttft_ms column) for the speedup number
    print(f"\n{'metric':<32} {'lossless':>14} {'R32_f015':>14} {'speedup':>10}")
    print("-" * 75)
    for label, path in CONFIGS.items():
        rows = load(path)
        ttfts = [float(r["ttft_ms"]) for r in rows if r.get("ttft_ms")]
        data[label + "_ttft"] = sum(ttfts) / len(ttfts) if ttfts else 0
    ls_ttft = data["lossless_ttft"]
    r32_ttft = data["R32_f015_ttft"]
    speedup = ls_ttft / r32_ttft if r32_ttft > 0 else 0
    print(f"{'ttft_ms (actual, reusers)':<32} {ls_ttft:>14.1f} {r32_ttft:>14.1f} {speedup:>9.2f}x")

    # R32 time decomposition (where does R32 spend its time?)
    print(f"\n=== R32_f015 TTFT decomposition (where the time goes) ===")
    r32_total = totals["R32_f015"]
    for s in STAGES:
        mean, nz, n = data["R32_f015"][s]
        pct = mean / r32_total * 100 if r32_total > 0 else 0
        print(f"  {s:<32} {mean:>8.3f}ms  ({pct:>5.1f}% of breakdown sum)")

    # JSON for deck
    out = {
        "lossless": {s: {"mean": data["lossless"][s][0],
                         "nonzero": data["lossless"][s][1],
                         "total": data["lossless"][s][2]} for s in STAGES},
        "R32_f015": {s: {"mean": data["R32_f015"][s][0],
                         "nonzero": data["R32_f015"][s][1],
                         "total": data["R32_f015"][s][2]} for s in STAGES},
        "ttft_ms": {"lossless": ls_ttft, "R32_f015": r32_ttft,
                     "speedup": speedup},
        "breakdown_sum": totals,
    }
    out_path = ROOT / "results/p4_verify/per_stage_ttft_breakdown.json"
    out_path.write_text(__import__("json").dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()