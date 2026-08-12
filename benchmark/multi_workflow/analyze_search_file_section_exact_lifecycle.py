#!/usr/bin/env python3
"""Compute the frozen persistent-source lifecycle speed metric."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.prepare_search_file_section_lifecycle_metric import (
    CAMPAIGN,
    read,
    source_usage,
    write,
)


def measured(rows: list[dict[str, Any]], group_index: int) -> list[float]:
    return [
        float(row["ttft_ms"])
        for row in rows
        if not row["warmup"] and int(row["group_index"]) == group_index
    ]


def summarize_lifecycle(
    plan: dict[str, Any], passes: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    groups = plan["groups"]
    usage = source_usage(plan)
    group_source: dict[int, str] = {}
    dense_by_group: dict[int, float] = {}
    reuse_by_group: dict[int, float] = {}
    builds: dict[str, list[float]] = {source_id: [] for source_id in usage}
    for group in groups:
        index = int(group["group_index"])
        source_id = str(group["cases"][0]["source_id"])
        group_source[index] = source_id
        dense_samples = []
        reuse_samples = []
        for value in passes.values():
            dense_samples.extend(measured(value["dense"]["targets"], index))
            reuse_samples.extend(measured(value["reuse"]["targets"], index))
            per_group_build = sum(
                float(row["elapsed_ms"])
                for row in value["reuse"]["sources"]
                if int(row["group_index"]) == index
            )
            builds[source_id].append(per_group_build)
        if not dense_samples or len(dense_samples) != len(reuse_samples):
            raise ValueError(f"group {index}: paired measured TTFT absent")
        dense_by_group[index] = statistics.fmean(dense_samples)
        reuse_by_group[index] = statistics.fmean(reuse_samples)
    build_once = {
        source_id: statistics.median(samples)
        for source_id, samples in builds.items()
    }
    dense_total = sum(dense_by_group.values())
    reuse_targets_total = sum(reuse_by_group.values())
    build_total = sum(build_once.values())

    def speedup(uses: int) -> float:
        return uses * dense_total / (uses * reuse_targets_total + build_total)

    per_source = []
    for source_id, target_uses in sorted(usage.items()):
        indices = [
            index for index, value in group_source.items() if value == source_id
        ]
        dense = sum(dense_by_group[index] for index in indices)
        reuse = sum(reuse_by_group[index] for index in indices)
        per_source.append(
            {
                "source_id": source_id,
                "target_groups": target_uses,
                "build_once_ms": build_once[source_id],
                "cache_ready_speedup": dense / reuse,
                "observed_lifecycle_n1_speedup": (
                    dense / (reuse + build_once[source_id])
                ),
            }
        )
    return {
        "target_groups": len(groups),
        "distinct_targeted_sources": len(usage),
        "target_uses_per_source": sorted(usage.values()),
        "dense_target_ttft_sum_ms": dense_total,
        "reuse_target_ttft_sum_ms": reuse_targets_total,
        "distinct_source_build_sum_ms": build_total,
        "cache_ready_speedup_ratio_of_sums": dense_total / reuse_targets_total,
        "observed_lifecycle_n1_speedup": speedup(1),
        "observed_lifecycle_n4_speedup": speedup(4),
        "observed_lifecycle_n16_speedup": speedup(16),
        "macro_source_lifecycle_speedup_median": statistics.median(
            row["observed_lifecycle_n1_speedup"] for row in per_source
        ),
        "per_source": per_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("canary4", "fresh24"), default="canary4")
    args = parser.parse_args()
    root = CAMPAIGN / f"exact_prompt_replay/{args.label}/sglang_coding"
    registration = read(
        CAMPAIGN / "SEARCH_FILE_SECTION_LIFECYCLE_METRIC_REGISTRATION.json"
    )
    passes = {
        sequence: {
            arm: read(root / sequence / f"{arm}.json")
            for arm in ("dense", "reuse")
        }
        for sequence in ("ab", "ba")
    }
    value = {
        "schema_version": 1,
        "status": "COMPLETE",
        "classification": "secondary preregistered persistent-source lifecycle metric",
        "label": args.label,
        "summary": summarize_lifecycle(read(root / "PLAN.json"), passes),
        "registration": registration,
    }
    output = root / "LIFECYCLE_RESULT.json"
    write(output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
