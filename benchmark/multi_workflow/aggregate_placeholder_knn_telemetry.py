#!/usr/bin/env python3
"""
Phase 6.2 / handoff §6.8 aggregator.

Aggregates the F1-skip-rate and other v44 placeholder_knn_reuse telemetry
fields that bench_kvcomm_ttft_stress.py emits per request row.

Outputs a per-(mode, agent_count) table to stdout. Compares against the
handoff §6.8 gate (F1-skip-rate < 5%) and prints PASS/FAIL.

Usage:
    python -m benchmark.multi_workflow.aggregate_placeholder_knn_telemetry \
        --ttft-table results/ttft_agenttemplatekv/multi_agent_placeholder_v44_KNNFIRST_*/ttft_table.csv
    # glob expansion is done by the shell; pass explicit paths or quote the glob
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


# handoff §6.8 gate: F1-skip-rate (placeholder_anchor_store_skipped_low_f1_count /
# placeholder_anchor_store_entry_count) must be < 5%.
F1_SKIP_GATE = 0.05

# Fields aggregated per (mode, agent_count)
SKIP_FIELDS = [
    "placeholder_anchor_store_skipped_low_f1_count",
    "placeholder_anchor_pool_skipped_cost_count",
    "placeholder_knn_skipped_high_overlap_count",
    "placeholder_knn_skipped_short_new_tokens_count",
    "placeholder_knn_skipped_high_span_overlap_count",
    "placeholder_knn_skipped_high_new_token_ratio_count",
]
COUNTER_FIELDS = [
    "placeholder_anchor_store_entry_count",
    "placeholder_anchor_pool_hit_count",
    "placeholder_anchor_pool_miss_count",
    "placeholder_kv_prefill_matched_slots",
    "placeholder_kv_prefill_skipped_tokens",
    "placeholder_kv_prefill_overlap_tokens",
    "placeholder_knn_pre_rotated_hit_count",
    "placeholder_knn_pre_rotated_miss_count",
    "placeholder_knn_head_rotation_total_ops",
    "placeholder_anchor_pool_copy_error_count",
]
COPY_METHOD_FIELD = "placeholder_knn_copy_method"
SIM_MEAN_FIELD = "placeholder_knn_topk_similarity_mean"


def _to_int(v):
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _to_float(v):
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def aggregate(csv_path: Path) -> dict:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {}

    # Per (mode, agent_count) bucket
    bucket = defaultdict(lambda: {
        "rows": 0,
        "counters": defaultdict(int),
        "copy_methods": defaultdict(int),
        "sim_sum": 0.0,
        "sim_n": 0,
    })
    for r in rows:
        mode = r.get("mode", "?")
        ac = r.get("agent_count", "?")
        key = (mode, ac)
        b = bucket[key]
        b["rows"] += 1
        for f in SKIP_FIELDS + COUNTER_FIELDS:
            b["counters"][f] += _to_int(r.get(f))
        cm = (r.get(COPY_METHOD_FIELD) or "").strip() or "<empty>"
        b["copy_methods"][cm] += 1
        sim = _to_float(r.get(SIM_MEAN_FIELD))
        if sim > 0:
            b["sim_sum"] += sim
            b["sim_n"] += 1

    # Materialize
    out_rows = []
    for (mode, ac), b in sorted(bucket.items()):
        entries = b["counters"]["placeholder_anchor_store_entry_count"]
        skipped_low_f1 = b["counters"]["placeholder_anchor_store_skipped_low_f1_count"]
        skip_rate = (skipped_low_f1 / entries) if entries else 0.0
        out_rows.append({
            "mode": mode,
            "agent_count": ac,
            "rows": b["rows"],
            "anchor_entries": entries,
            "skipped_low_f1": skipped_low_f1,
            "f1_skip_rate": skip_rate,
            "copy_methods": dict(b["copy_methods"]),
            "sim_mean_avg": (b["sim_sum"] / b["sim_n"]) if b["sim_n"] else 0.0,
            "counters": dict(b["counters"]),
        })
    return {"source": str(csv_path), "rows": out_rows}


def print_report(report: dict, gate: float = F1_SKIP_GATE) -> None:
    print(f"Source: {report['source']}")
    print()
    print(
        f"{'mode':<35} {'ac':<3} {'rows':<5} {'entries':<8} "
        f"{'skip_LF1':<10} {'rate':<8} {'gate':<6} {'sim_mean':<9} {'copy_methods'}"
    )
    print("-" * 110)
    fails = []
    for r in report["rows"]:
        rate_str = f"{r['f1_skip_rate']:.2%}"
        gate_label = "PASS" if r["f1_skip_rate"] < gate else "FAIL"
        if gate_label == "FAIL":
            fails.append((r["mode"], r["agent_count"], r["f1_skip_rate"]))
        cm = ", ".join(f"{k}={v}" for k, v in sorted(r["copy_methods"].items()))
        sim = f"{r['sim_mean_avg']:.4f}" if r["sim_mean_avg"] else "—"
        print(
            f"{r['mode']:<35} {r['agent_count']:<3} {r['rows']:<5} "
            f"{r['anchor_entries']:<8} {r['skipped_low_f1']:<10} "
            f"{rate_str:<8} {gate_label:<6} {sim:<9} {cm}"
        )
    print()
    if fails:
        print(f"GATE §6.8 FAIL ({len(fails)} cells above {gate:.0%}):")
        for mode, ac, rate in fails:
            print(f"  - {mode} ac={ac}: {rate:.2%}")
        sys.exit(2)
    else:
        print(f"GATE §6.8 PASS: every (mode, agent_count) cell has F1-skip < {gate:.0%}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ttft-table",
        type=Path,
        action="append",
        required=True,
        help="One or more ttft_table.csv paths (can be passed multiple times).",
    )
    p.add_argument("--json-out", type=Path, default=None,
                   help="Optional path to dump the full aggregated JSON.")
    p.add_argument("--gate", type=float, default=F1_SKIP_GATE,
                   help=f"F1-skip-rate gate (default {F1_SKIP_GATE}).")
    args = p.parse_args()

    all_reports = []
    for path in args.ttft_table:
        if not path.exists():
            print(f"WARN: skipping missing file {path}", file=sys.stderr)
            continue
        r = aggregate(path)
        if r:
            all_reports.append(r)
            print_report(r, gate=args.gate)
            print()

    if args.json_out:
        args.json_out.write_text(json.dumps(all_reports, indent=2))
        print(f"Wrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()