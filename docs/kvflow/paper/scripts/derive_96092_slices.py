#!/usr/bin/env python3
"""Derive ASPLOS dataset/ablation slices from frozen job 96092 JSON.

No GPU. Reads PLAN.json, dense.json, reuse.json, RESULT.json,
REGISTRATION.json plus the expanded24 CLIENT_LEDGER and TELEMETRY.
Writes ANALYSIS.json (backward compatible) and SLICES.json.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

GROUP_ID = re.compile(
    r"p(?P<pid>\d+)-m(?P<m>\d+)-s(?P<s>\d+)-q(?P<q>\d+)-v\d+-(?P<h>[0-9a-f]+)$"
)

from impactkv_paths import artifact_root

_ART = artifact_root()
DEFAULT_ART = _ART / "impactkv_swebench_prerotated_file_modules_20260818"
DEFAULT_TRAJ = (
    _ART
    / "impactkv_natural_code_cost_agent_expanded24_20260808/online/"
    / "coding_natural_code_cost/full_24"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def length_bucket(n: int) -> str:
    if n < 3000:
        return "<3K"
    if n < 5000:
        return "3-5K"
    if n < 7000:
        return "5-7K"
    return ">=7K"


def delta_bucket(abs_delta: int) -> str:
    if abs_delta < 500:
        return "<500"
    if abs_delta < 1500:
        return "500-1500"
    if abs_delta < 3000:
        return "1500-3000"
    return ">=3000"


def repo_of(instance_id: str) -> str:
    return instance_id.split("__")[0]


def pair_rows(dense: dict, reuse: dict) -> dict[tuple[int, int], tuple[dict, dict]]:
    dense_rows = {
        (int(row["group_index"]), int(row["round_index"])): row
        for row in dense["targets"]
        if not row["warmup"]
    }
    reuse_rows = {
        (int(row["group_index"]), int(row["round_index"])): row
        for row in reuse["targets"]
        if not row["warmup"]
    }
    if set(dense_rows) != set(reuse_rows):
        raise ValueError("paired targets differ")
    return {key: (dense_rows[key], reuse_rows[key]) for key in dense_rows}


def cache_ready_speedup(pairs: list[tuple[dict, dict]]) -> float:
    dense_mean = statistics.fmean(float(a["ttft_ms"]) for a, _ in pairs)
    reuse_mean = statistics.fmean(float(b["ttft_ms"]) for _, b in pairs)
    if reuse_mean <= 0:
        raise ValueError("reuse mean ttft must be positive")
    return dense_mean / reuse_mean


def nonce_to_instance(ledger_path: Path, telemetry_path: Path) -> dict[str, str]:
    seen: list[str] = []
    ncalls: dict[str, int] = {}
    with ledger_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            nonce = str(row["model_instance_nonce"])
            ncalls[nonce] = ncalls.get(nonce, 0) + 1
            if nonce not in seen:
                seen.append(nonce)
    telemetry = read_json(telemetry_path)
    instance_ids = list(telemetry["instances"])
    if len(seen) != len(instance_ids):
        raise ValueError(
            f"nonce/instance count mismatch: {len(seen)} vs {len(instance_ids)}"
        )
    mapping: dict[str, str] = {}
    for nonce, instance_id in zip(seen, instance_ids, strict=True):
        expected = len(telemetry["instances"][instance_id]["calls"])
        if ncalls[nonce] != expected:
            raise ValueError(
                f"call-count mismatch {nonce} {instance_id}: "
                f"{ncalls[nonce]} vs {expected}"
            )
        mapping[nonce] = instance_id
    return mapping


def group_instance_id(
    original_id: str, nonce_map: dict[str, str]
) -> tuple[str, str]:
    match = GROUP_ID.match(original_id)
    if match is None:
        raise ValueError(f"unparseable group id {original_id}")
    nonce = f"p{match.group('pid')}-m{match.group('m')}"
    if nonce not in nonce_map:
        raise ValueError(f"no instance for nonce {nonce}")
    return nonce_map[nonce], match.group("h")


def zero_shift_token_share(manifest_path: Path, plan: list[dict]) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    dropped = [
        case
        for case in manifest["cases"]
        if int(case["target_start"]) - int(case["source_start"]) == 0
    ]
    dropped_tokens = sum(int(case["length"]) for case in dropped)
    target_tokens = sum(len(group["target_input_ids"]) for group in plan)
    copied = sum(int(group["copied_tokens"]) for group in plan)
    return {
        "dropped_islands": len(dropped),
        "dropped_tokens": dropped_tokens,
        "plan_copied_tokens": copied,
        "plan_target_tokens": target_tokens,
        "dropped_over_target": dropped_tokens / target_tokens if target_tokens else 0.0,
        "not_a_radix_ttft": True,
    }


def analyze(
    art: Path,
    traj: Path = DEFAULT_TRAJ,
) -> dict[str, Any]:
    plan = read_json(art / "PLAN.json")["groups"]
    dense = read_json(art / "dense.json")
    reuse = read_json(art / "reuse.json")
    result = read_json(art / "RESULT.json")
    if result.get("status") != "COMPLETE":
        raise ValueError("parent RESULT is not COMPLETE")
    registration = read_json(art / "REGISTRATION.json")
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
    unmatched = 0
    for group in plan:
        original = str(group["original_target_group_id"])
        instance_id, digest = group_instance_id(original, nonce_map)
        if not str(group["target_prompt_hash"]).startswith(digest):
            unmatched += 1
            raise ValueError(f"hash prefix mismatch {original}")
        abs_delta = abs(int(group["pre_rotate_delta"]))
        prompt_tokens = len(group["target_input_ids"])
        copied_tokens = int(group["copied_tokens"])
        group_meta.append(
            {
                "group_index": int(group["group_index"]),
                "instance_id": instance_id,
                "repo": repo_of(instance_id),
                "islands": int(group["islands"]),
                "abs_delta": abs_delta,
                "prompt_tokens": prompt_tokens,
                "copied_tokens": copied_tokens,
                "copied_fraction": copied_tokens / prompt_tokens,
            }
        )
    if unmatched:
        raise ValueError("unmatched PLAN hashes")
    if len(group_meta) != 235:
        raise ValueError("expected 235 groups")
    universe = set(nonce_map.values())
    if len(universe) != 24:
        raise ValueError(f"telemetry is not 24 instances: {len(universe)}")
    used = {row["instance_id"] for row in group_meta}
    extra = used - universe
    if extra:
        raise ValueError(f"PLAN instances outside expanded24: {sorted(extra)[:3]}")

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

    island_slices = slice_speedup(lambda meta: meta["islands"])
    if sum(row["groups"] for row in island_slices.values()) != 235:
        raise ValueError("island slices must cover 235 groups")

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

    frac_slices = slice_speedup(frac_quartile)
    if sum(row["groups"] for row in frac_slices.values()) != 235:
        raise ValueError("copied-fraction quartiles must cover 235 groups")

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
    if n4_shared <= n_use["4"]:
        raise ValueError("unique-source amortization should not be worse than per-group")

    dense_ttft = [float(a["ttft_ms"]) for a, _ in pairs.values()]
    reuse_ttft = [float(b["ttft_ms"]) for _, b in pairs.values()]
    pair_speedup = [
        float(a["ttft_ms"]) / float(b["ttft_ms"]) for a, b in pairs.values()
    ]
    bucket_dense: dict[str, list[float]] = defaultdict(list)
    bucket_reuse: dict[str, list[float]] = defaultdict(list)
    prompt_len = {row["group_index"]: row["prompt_tokens"] for row in group_meta}
    for key, (a, b) in pairs.items():
        name = length_bucket(prompt_len[key[0]])
        bucket_dense[name].append(float(a["ttft_ms"]))
        bucket_reuse[name].append(float(b["ttft_ms"]))

    hash_groups: dict[str, set[int]] = defaultdict(set)
    for group in plan:
        for digest in group["source_prompt_hashes"]:
            hash_groups[digest].add(int(group["group_index"]))
    reuse_hist = {
        str(k): sum(1 for v in hash_groups.values() if len(v) == k)
        for k in sorted({len(v) for v in hash_groups.values()})
    }

    repo_counts = Counter(row["repo"] for row in group_meta)
    instance_counts = Counter(row["instance_id"] for row in group_meta)
    manifest = traj / "DYNAMIC_MANIFEST.json"
    zero = zero_shift_token_share(manifest, plan)
    if zero["dropped_islands"] != int(
        registration["dataset"]["skipped"]["zero_shift_islands"]
    ):
        raise ValueError("zero-shift island count drifted from REGISTRATION")

    slices = {
        "schema_version": 2,
        "status": "DERIVED_FROM_FROZEN_96092",
        "not_a_new_gpu_arm": True,
        "prefetch": False,
        "parent_result_status": result["status"],
        "dataset": {
            "name": "princeton-nlp/SWE-bench_Verified",
            "source_run": "expanded24",
            "tasks": 24,
            "instances_with_eligible_groups": len(instance_counts),
            "instances_without_eligible_groups": sorted(universe - used),
            "groups_are_rolling6_turns_not_tasks": True,
            "target_groups": 235,
            "instances": sorted(instance_counts),
            "instance_group_counts": dict(sorted(instance_counts.items())),
            "repos": dict(sorted(repo_counts.items())),
            "n_repos": len(repo_counts),
            "convenience_sample": True,
        },
        "ttft_ms": {
            "dense_mean": statistics.fmean(dense_ttft),
            "reuse_mean": statistics.fmean(reuse_ttft),
            "dense_p50": statistics.median(dense_ttft),
            "reuse_p50": statistics.median(reuse_ttft),
            "dense_p90": sorted(dense_ttft)[int(0.9 * (len(dense_ttft) - 1))],
            "reuse_p90": sorted(reuse_ttft)[int(0.9 * (len(reuse_ttft) - 1))],
            "paired_speedup_p50": statistics.median(pair_speedup),
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
        "island_count_slices": island_slices,
        "abs_delta_slices": slice_speedup(lambda meta: delta_bucket(meta["abs_delta"])),
        "repo_slices": slice_speedup(lambda meta: meta["repo"]),
        "copied_fraction_quartiles": {
            "cuts": cuts,
            "slices": frac_slices,
        },
        "cross_group_source_amortization": {
            "n4_per_group_build": n_use["4"],
            "n4_unique_source_build_once": n4_shared,
            "unique_source_hashes": len(hash_elapsed),
            "shared_source_build_ms": shared_build,
            "per_group_source_build_ms": sum(source_build.values()),
            "not_headline": True,
        },
        "source_sharing": {
            "unique_source_prompt_hashes": len(hash_groups),
            "target_groups": len(plan),
            "sources_used_in_more_than_one_group": sum(
                1 for v in hash_groups.values() if len(v) > 1
            ),
            "groups_per_source_hash": reuse_hist,
        },
        "skipped_zero_shift": zero,
        "skipped_zero_shift_islands": zero["dropped_islands"],
    }
    # Keep ANALYSIS.json loadable by anything that expected schema 1 keys.
    analysis = {
        "schema_version": 1,
        "status": slices["status"],
        "not_a_new_gpu_arm": True,
        "prefetch": False,
        "parent_result_status": result["status"],
        "ttft_ms": slices["ttft_ms"],
        "n_use_including_one_source_build": n_use,
        "length_buckets": slices["length_buckets"],
        "source_sharing": slices["source_sharing"],
        "skipped_zero_shift_islands": zero["dropped_islands"],
    }
    write_json(art / "ANALYSIS.json", analysis)
    write_json(art / "SLICES.json", slices)
    return slices


def main() -> None:
    slices = analyze(DEFAULT_ART, DEFAULT_TRAJ)
    print(
        json.dumps(
            {
                "status": slices["status"],
                "tasks": slices["dataset"]["tasks"],
                "n_repos": slices["dataset"]["n_repos"],
                "n4": slices["n_use_including_one_source_build"]["4"],
                "island_count_slices": slices["island_count_slices"],
                "copied_fraction_quartiles": slices["copied_fraction_quartiles"][
                    "slices"
                ],
                "cross_group_n4": slices["cross_group_source_amortization"][
                    "n4_unique_source_build_once"
                ],
                "dropped_zero_shift": slices["skipped_zero_shift"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
