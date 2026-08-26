#!/usr/bin/env python3
"""PLAN-only motivation: radix LCP vs shifted file-module copy are disjoint."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from impactkv_paths import artifact_root

DEFAULT_PLAN = (
    artifact_root()
    / "impactkv_swebench_7b_file_modules_prefixkey_20260824/PLAN.json"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def lcp_len(left: list[int], right: list[int]) -> int:
    n = 0
    for a, b in zip(left, right):
        if a != b:
            break
        n += 1
    return n


def best_source_lcp(group: dict[str, Any]) -> int:
    target = [int(v) for v in group["target_input_ids"]]
    return max(
        (lcp_len([int(v) for v in source], target) for source in group["source_input_ids"]),
        default=0,
    )


def group_coverage_series(plan_path: Path = DEFAULT_PLAN) -> dict[str, list]:
    """Per-group LCP / file-island / remainder. No GPU. Sorted by prompt length."""
    groups = read_json(plan_path)["groups"]
    if not groups:
        raise ValueError("empty PLAN")
    rows: list[tuple[int, int, int, int]] = []
    for group in groups:
        target_len = len(group["target_input_ids"])
        lcp = best_source_lcp(group)
        copied = int(group["copied_tokens"])
        rows.append((target_len, lcp, copied, int(group["group_index"])))
    rows.sort(key=lambda row: (row[0], row[3]))
    target_len = [row[0] for row in rows]
    lcp = [row[1] for row in rows]
    copied = [row[2] for row in rows]
    return {
        "target_tokens": target_len,
        "lcp_tokens": lcp,
        "copied_tokens": copied,
        "lcp_frac": [a / b for a, b in zip(lcp, target_len)],
        "copied_frac": [a / b for a, b in zip(copied, target_len)],
        "rest_frac": [max(0.0, 1.0 - a / c - b / c) for a, b, c in zip(lcp, copied, target_len)],
    }


def analyze_plan(plan_path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    doc = read_json(plan_path)
    groups = doc["groups"]
    if not groups:
        raise ValueError("empty PLAN")
    radix = []
    lossy = []
    overlap_tokens = 0
    lcp_past_island = 0
    both = 0
    for group in groups:
        target_len = len(group["target_input_ids"])
        lcp = best_source_lcp(group)
        copied = int(group["copied_tokens"])
        first_island = min(int(case["target_start"]) for case in group["cases"])
        if lcp > first_island:
            lcp_past_island += 1
        if lcp > 0 and copied > 0:
            both += 1
        radix.append(lcp / target_len)
        lossy.append(copied / target_len)
        for case in group["cases"]:
            start = int(case["target_start"])
            length = int(case["length"])
            lo = max(start, 0)
            hi = min(start + length, lcp)
            if hi > lo:
                overlap_tokens += hi - lo
    hashes = defaultdict(set)
    prefixes = set()
    for group in groups:
        for case in group["cases"]:
            hashes[str(case["content_hash"])].add(int(group["group_index"]))
            prefixes.add(str(case["source_prefix_token_hash"]))
    shared = sum(1 for groups_for_hash in hashes.values() if len(groups_for_hash) > 1)
    return {
        "schema_version": 1,
        "status": "DERIVED_FROM_7B_PLAN",
        "not_a_new_gpu_arm": True,
        "model": doc.get("model"),
        "groups": len(groups),
        "islands": sum(int(group["islands"]) for group in groups),
        "mean_target_tokens": statistics.fmean(len(g["target_input_ids"]) for g in groups),
        "mean_radix_lcp_tokens": statistics.fmean(
            best_source_lcp(group) for group in groups
        ),
        "mean_radix_fraction": statistics.fmean(radix),
        "mean_lossy_fraction": statistics.fmean(lossy),
        "mean_copied_tokens": statistics.fmean(int(g["copied_tokens"]) for g in groups),
        "lcp_island_overlap_tokens": overlap_tokens,
        "lcp_past_first_island_groups": lcp_past_island,
        "groups_with_both_radix_and_lossy": both,
        "unique_content_hashes": len(hashes),
        "content_hashes_in_multiple_groups": shared,
        "unique_source_prefixes": len(prefixes),
        "disjoint_radix_and_file_islands": overlap_tokens == 0 and lcp_past_island == 0,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PLAN.with_name("MOTIVATION.json"),
    )
    args = parser.parse_args()
    value = analyze_plan(args.plan)
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
