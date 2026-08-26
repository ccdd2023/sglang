#!/usr/bin/env python3
"""Derive 7B ablation slices from frozen job 137185. No GPU."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from derive_96092_slices import (
    DEFAULT_TRAJ,
    cache_ready_speedup,
    delta_bucket,
    group_instance_id,
    length_bucket,
    nonce_to_instance,
    pair_rows,
    read_json,
    repo_of,
    write_json,
)

DEFAULT_ART = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_swebench_7b_file_modules_prefixkey_20260824"
)


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def analyze(art: Path, traj: Path = DEFAULT_TRAJ) -> dict[str, Any]:
    plan = read_json(art / "PLAN.json")["groups"]
    dense = read_json(art / "dense.json")
    reuse = read_json(art / "reuse.json")
    result = read_json(art / "RESULT.json")
    if result.get("status") != "COMPLETE" or result.get("qwen25_rope_ok") is not True:
        raise ValueError("parent 7B RESULT is not COMPLETE / rope-ok")
    pairs = pair_rows(dense, reuse)
    source_build = {
        index: sum(
            float(row["elapsed_ms"])
            for row in reuse["sources"]
            if int(row["group_index"]) == index
        )
        for index in range(len(plan))
    }
    n_use: dict[str, float] = {}
    for n in (1, 2, 4, 8):
        dense_sum = 0.0
        reuse_sum = 0.0
        for group in plan:
            index = int(group["group_index"])
            group_pairs = [pair for key, pair in pairs.items() if key[0] == index]
            d = statistics.fmean(float(a["ttft_ms"]) for a, _ in group_pairs)
            r = statistics.fmean(float(b["ttft_ms"]) for _, b in group_pairs)
            dense_sum += d * n
            reuse_sum += r * n + source_build[index]
        n_use[str(n)] = dense_sum / reuse_sum
    n4 = float(result["latency"]["n4_including_one_source_build_speedup"])
    if abs(n_use["4"] - n4) > 1e-6:
        raise ValueError(f"derived N=4 {n_use['4']} drifted from RESULT {n4}")

    nonce_map = nonce_to_instance(
        traj / "CLIENT_LEDGER.jsonl", traj / "TELEMETRY.json"
    )
    group_meta: list[dict[str, Any]] = []
    for group in plan:
        instance_id, _digest = group_instance_id(
            str(group["original_target_group_id"]), nonce_map
        )
        prompt_tokens = len(group["target_input_ids"])
        copied_tokens = int(group["copied_tokens"])
        group_meta.append(
            {
                "group_index": int(group["group_index"]),
                "instance_id": instance_id,
                "repo": repo_of(instance_id),
                "islands": int(group["islands"]),
                "abs_delta": abs(int(group["pre_rotate_delta"])),
                "prompt_tokens": prompt_tokens,
                "copied_tokens": copied_tokens,
                "copied_fraction": copied_tokens / prompt_tokens,
            }
        )
    if len(group_meta) != 235:
        raise ValueError("expected 235 groups")

    def slice_speedup(key_fn) -> dict[str, Any]:
        buckets: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
        group_counts: dict[str, set[int]] = defaultdict(set)
        for meta in group_meta:
            label = str(key_fn(meta))
            group_counts[label].add(meta["group_index"])
            for pair_key, pair in pairs.items():
                if pair_key[0] == meta["group_index"]:
                    buckets[label].append(pair)
        return {
            label: {
                "groups": len(group_counts[label]),
                "pairs": len(buckets[label]),
                "cache_ready_speedup": cache_ready_speedup(buckets[label]),
            }
            for label in sorted(buckets, key=lambda item: (len(item), item))
        }

    fractions = [meta["copied_fraction"] for meta in group_meta]
    cuts = statistics.quantiles(fractions, n=4)

    def frac_quartile(meta: dict[str, Any]) -> str:
        frac = meta["copied_fraction"]
        if frac <= cuts[0]:
            return "Q1"
        if frac <= cuts[1]:
            return "Q2"
        if frac <= cuts[2]:
            return "Q3"
        return "Q4"

    sources_by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in reuse["sources"]:
        sources_by_group[int(row["group_index"])].append(row)
    hash_elapsed: dict[str, list[float]] = defaultdict(list)
    for group in plan:
        index = int(group["group_index"])
        rows = sorted(
            sources_by_group[index], key=lambda item: int(item["source_index"])
        )
        hashes = list(group["source_prompt_hashes"])
        if len(rows) != len(hashes):
            raise ValueError(f"source rows/hashes mismatch group {index}")
        for digest, row in zip(hashes, rows, strict=True):
            hash_elapsed[str(digest)].append(float(row["elapsed_ms"]))
    shared_build = sum(statistics.fmean(values) for values in hash_elapsed.values())
    dense_n4 = 0.0
    reuse_ttft_n4 = 0.0
    for group in plan:
        index = int(group["group_index"])
        group_pairs = [pair for key, pair in pairs.items() if key[0] == index]
        d = statistics.fmean(float(a["ttft_ms"]) for a, _ in group_pairs)
        r = statistics.fmean(float(b["ttft_ms"]) for _, b in group_pairs)
        dense_n4 += d * 4
        reuse_ttft_n4 += r * 4
    n4_shared = dense_n4 / (reuse_ttft_n4 + shared_build)

    repo_counts = Counter(row["repo"] for row in group_meta)
    instance_counts = Counter(row["instance_id"] for row in group_meta)
    universe = set(nonce_map.values())
    used = {row["instance_id"] for row in group_meta}

    bucket_dense: dict[str, list[float]] = defaultdict(list)
    bucket_reuse: dict[str, list[float]] = defaultdict(list)
    prompt_len = {row["group_index"]: row["prompt_tokens"] for row in group_meta}
    for key, (a, b) in pairs.items():
        name = length_bucket(prompt_len[key[0]])
        bucket_dense[name].append(float(a["ttft_ms"]))
        bucket_reuse[name].append(float(b["ttft_ms"]))

    dense_ttft = [float(a["ttft_ms"]) for a, _ in pairs.values()]
    reuse_ttft = [float(b["ttft_ms"]) for _, b in pairs.values()]
    pair_saving = [
        (dense - reuse) / dense for dense, reuse in zip(dense_ttft, reuse_ttft)
    ]
    group_speedup: list[float] = []
    group_frac: list[float] = []
    by_group: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for key, pair in pairs.items():
        by_group[key[0]].append(pair)
    meta_by_index = {row["group_index"]: row for row in group_meta}
    for index, group_pairs in by_group.items():
        group_speedup.append(cache_ready_speedup(group_pairs))
        group_frac.append(float(meta_by_index[index]["copied_fraction"]))

    return {
        "schema_version": 1,
        "status": "DERIVED_FROM_FROZEN_137185",
        "not_a_new_gpu_arm": True,
        "prefetch": False,
        "parent_job": "137185",
        "parent_result_status": result["status"],
        "dataset": {
            "name": "princeton-nlp/SWE-bench_Verified",
            "source_run": "expanded24",
            "tasks": 24,
            "instances_with_eligible_groups": len(instance_counts),
            "instances_without_eligible_groups": sorted(universe - used),
            "groups_are_rolling6_turns_not_tasks": True,
            "target_groups": 235,
            "repos": dict(sorted(repo_counts.items())),
            "n_repos": len(repo_counts),
            "convenience_sample": True,
            "retokenized_7b": True,
        },
        "n_use_including_one_source_build": n_use,
        "length_buckets": {
            name: {
                "pairs": len(bucket_dense[name]),
                "cache_ready_speedup": (
                    statistics.fmean(bucket_dense[name])
                    / statistics.fmean(bucket_reuse[name])
                ),
            }
            for name in ("<3K", "3-5K", "5-7K", ">=7K")
            if bucket_dense[name]
        },
        "island_count_slices": slice_speedup(lambda meta: meta["islands"]),
        "abs_delta_slices": slice_speedup(
            lambda meta: delta_bucket(meta["abs_delta"])
        ),
        "repo_slices": slice_speedup(lambda meta: meta["repo"]),
        "copied_fraction_quartiles": {
            "cuts": cuts,
            "slices": slice_speedup(frac_quartile),
        },
        "ttft_ms": {
            "pairs": len(dense_ttft),
            "dense_mean": statistics.fmean(dense_ttft),
            "reuse_mean": statistics.fmean(reuse_ttft),
            "dense_p50": _percentile(dense_ttft, 50),
            "reuse_p50": _percentile(reuse_ttft, 50),
            "dense_p90": _percentile(dense_ttft, 90),
            "reuse_p90": _percentile(reuse_ttft, 90),
            "dense_p99": _percentile(dense_ttft, 99),
            "reuse_p99": _percentile(reuse_ttft, 99),
            "paired_saving_mean": statistics.fmean(pair_saving),
            "paired_saving_median": statistics.median(pair_saving),
            "group_speedup_median": statistics.median(group_speedup),
            "group_speedup_min": min(group_speedup),
            "group_speedup_max": max(group_speedup),
        },
        "group_scatter": {
            "copied_fraction": group_frac,
            "cache_ready_speedup": group_speedup,
        },
        "cross_group_source_amortization": {
            "n4_per_group_build": n_use["4"],
            "n4_unique_source_build": n4_shared,
            "not_headline": True,
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ART)
    args = parser.parse_args()
    slices = analyze(args.artifact)
    write_json(args.artifact / "SLICES.json", slices)
    print(
        {
            "n4": slices["n_use_including_one_source_build"]["4"],
            "islands": slices["island_count_slices"],
            "status": slices["status"],
        }
    )


if __name__ == "__main__":
    main()
