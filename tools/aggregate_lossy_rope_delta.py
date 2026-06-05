"""Aggregate lossy-anchor telemetry across one or more request log files.

Reads JSON log files (each line is a JSON object), extracts the
`lossy_anchor_rope_delta`, `lossy_predicted_distance`, `lossy_context_aware_*`,
and `lossy_final_reuse_confidence` fields, and prints:

  - rope_delta distribution (count, mean, std, p50, p99, max, %-equal-zero)
  - predicted_distance distribution (count, mean, std, p50, p99, max)
  - % of requests where the modifier demoted (predicted_distance >= d_max)
  - confidence reduction: avg(lossy_final_reuse_confidence - lossy_context_aware_confidence)
    for requests where they differ

Usage:
    python tools/aggregate_lossy_rope_delta.py results/ma_ttft/sglang.log \\
        --table results/same_code_context_variation/data/predicted_distance_table.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter


def _percentile(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = int(len(xs) * p)
    idx = max(0, min(idx, len(xs) - 1))
    return xs[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_files", nargs="+", help="one or more JSON-lines log files")
    p.add_argument("--table", help="predicted_distance_table.json (for d_max reference)")
    args = p.parse_args()

    d_max = None
    if args.table and os.path.exists(args.table):
        with open(args.table) as f:
            t = json.load(f)
        d_max = t.get("global", {}).get("predicted_d_norm_max_observed")

    rope_deltas: list[int] = []
    predicted: list[float] = []
    multipliers: list[float] = []
    confidence_diffs: list[float] = []
    base_confidences: list[float] = []
    modified_confidences: list[float] = []
    demoted_count = 0
    total = 0
    match_reason_counter: Counter = Counter()

    for path in args.log_files:
        if not os.path.exists(path):
            print(f"[warn] file not found: {path}")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Accept either top-level fields or nested under "response" / "meta_info"
                rec = obj.get("meta_info", obj) if isinstance(obj, dict) else {}
                if not isinstance(rec, dict):
                    continue
                total += 1
                rd = rec.get("lossy_anchor_rope_delta")
                if rd is not None:
                    rope_deltas.append(int(rd))
                pd_val = rec.get("lossy_predicted_distance")
                if pd_val is not None:
                    predicted.append(float(pd_val))
                mult = rec.get("lossy_context_aware_multiplier")
                if mult is not None:
                    multipliers.append(float(mult))
                # Compare base (0.95) vs modified confidence
                final_conf = rec.get("lossy_final_reuse_confidence")
                modified_conf = rec.get("lossy_context_aware_confidence")
                if final_conf is not None and modified_conf is not None and abs(final_conf - modified_conf) > 0.001:
                    confidence_diffs.append(final_conf - modified_conf)
                if final_conf is not None:
                    base_confidences.append(float(final_conf))
                if modified_conf is not None:
                    modified_confidences.append(float(modified_conf))
                if d_max is not None and pd_val is not None and pd_val >= d_max * 0.99:
                    demoted_count += 1
                reason = rec.get("lossy_final_match_reason")
                if reason:
                    match_reason_counter[reason] += 1

    print(f"=== AGGREGATED LOSSY TELEMETRY ({total} records) ===")
    print()
    print(f"lossy_anchor_rope_delta: count={len(rope_deltas)}, mean={sum(rope_deltas)/max(1,len(rope_deltas)):.2f}, "
          f"std={(sum((x - sum(rope_deltas)/max(1,len(rope_deltas)))**2 for x in rope_deltas)/max(1,len(rope_deltas)-1))**0.5:.2f}, "
          f"p50={_percentile(rope_deltas, 0.5)}, p99={_percentile(rope_deltas, 0.99)}, "
          f"max={max(rope_deltas) if rope_deltas else 0}, "
          f"%_equal_zero={sum(1 for x in rope_deltas if x == 0)/max(1,len(rope_deltas)):.1%}")
    print()
    print(f"lossy_predicted_distance: count={len(predicted)}, "
          f"mean={sum(predicted)/max(1,len(predicted)):.3f}, "
          f"std={(sum((x - sum(predicted)/max(1,len(predicted)))**2 for x in predicted)/max(1,len(predicted)-1))**0.5:.3f}, "
          f"p50={_percentile(predicted, 0.5):.3f}, p99={_percentile(predicted, 0.99):.3f}, "
          f"max={max(predicted) if predicted else 0:.3f}")
    if d_max is not None:
        print(f"  d_max (from table) = {d_max:.3f}")
        print(f"  requests within 1% of d_max (= potential demotions): {demoted_count}")
    print()
    print(f"context_aware_multiplier: count={len(multipliers)}, "
          f"mean={sum(multipliers)/max(1,len(multipliers)):.3f}, "
          f"min={min(multipliers) if multipliers else 1.0:.3f}")
    print()
    print(f"confidence delta (final - context_aware): count={len(confidence_diffs)}, "
          f"mean={sum(confidence_diffs)/max(1,len(confidence_diffs)):.4f}")
    print()
    print("match_reason distribution:")
    for reason, n in match_reason_counter.most_common():
        print(f"  {reason:>40}: {n}")


if __name__ == "__main__":
    main()
