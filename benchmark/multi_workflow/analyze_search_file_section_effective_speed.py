#!/usr/bin/env python3
"""Partition frozen exact speed by whether online decoding produced an action.

This is a post-result descriptive audit, not an admission rule.  It prevents
the many FormatError requests in the weak common-agent run from being mistaken
for useful coding-decision acceleration.  It also reports actual online K/V
snapshot materialization overhead separately from synthetic source-prompt
replay, because online sources arise during normal required agent requests.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
CAMPAIGN = (
    RuntimePaths.from_project(PROJECT).artifacts
    / "impactkv_common_agent_search_file_section_20260812"
)
ARM = "coding_search_file_section_mean"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def target_hashes_with_saved_action(run: Path) -> set[str]:
    hashes = set()
    for trajectory in sorted(run.glob("*/*.traj.json")):
        for message in read(trajectory).get("messages") or ():
            treatment = (message.get("extra") or {}).get("reuse_treatment") or {}
            if treatment.get("target_registered"):
                hashes.add(str(treatment["input_ids_sha256"]))
    return hashes


def partition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"targets": 0}
    dense_sum = sum(float(row["median_dense_ttft_ms"]) for row in rows)
    reuse_sum = sum(float(row["median_reuse_ttft_ms"]) for row in rows)
    return {
        "targets": len(rows),
        "cache_ready_speedup_median": statistics.median(
            float(row["cache_ready_speedup"]) for row in rows
        ),
        "cache_ready_speedup_ratio_of_sums": dense_sum / reuse_sum,
        "targets_cache_ready_faster": sum(
            float(row["cache_ready_speedup"]) > 1 for row in rows
        ),
        "reusable_tokens_median": statistics.median(
            int(row["reusable_tokens"]) for row in rows
        ),
    }


def actual_materialization(
    plan: dict[str, Any], online_server: list[dict[str, Any]], exact_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    targeted_source_ids = {
        str(case["source_id"])
        for group in plan.get("groups") or ()
        for case in group.get("cases") or ()
    }
    events = [
        row
        for row in online_server
        if row.get("event") == "source_materialized"
        and str(row.get("source_id")) in targeted_source_ids
    ]
    observed_ids = {str(row["source_id"]) for row in events}
    if observed_ids != targeted_source_ids:
        raise ValueError("targeted online source materialization coverage mismatch")
    dense = sum(float(row["median_dense_ttft_ms"]) for row in exact_rows)
    reuse = sum(float(row["median_reuse_ttft_ms"]) for row in exact_rows)
    overhead = sum(float(row["materialize_ms"]) for row in events)
    return {
        "distinct_targeted_sources": len(targeted_source_ids),
        "source_materialization_events": len(events),
        "incremental_materialization_ms": overhead,
        "cache_ready_speedup_ratio_of_sums": dense / reuse,
        "observed_online_lifecycle_speedup": dense / (reuse + overhead),
        "source_prompt_replay_excluded": True,
        "reason": (
            "source prompts were required online agent requests in both Dense and "
            "reuse; only snapshot materialize_ms is incremental reuse overhead"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("canary4", "fresh24"), default="canary4")
    args = parser.parse_args()
    online_scope = "sglang_canary" if args.label == "canary4" else "sglang_formal"
    online_count = 3 if args.label == "canary4" else 24
    online = CAMPAIGN / f"runs/{online_scope}/{ARM}/full_{online_count}"
    exact = CAMPAIGN / f"exact_prompt_replay/{args.label}/sglang_coding"
    plan = read(exact / "PLAN.json")
    result = read(exact / "RESULT.json")
    targets_by_group = {
        int(row["group_index"]): row for row in result.get("targets") or ()
    }
    action_hashes = target_hashes_with_saved_action(online)
    action_rows = []
    format_rows = []
    for group in plan.get("groups") or ():
        row = targets_by_group[int(group["group_index"])]
        target = (
            action_rows
            if str(group["target_prompt_hash"]) in action_hashes
            else format_rows
        )
        target.append(row)
    all_rows = [*action_rows, *format_rows]
    value = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "post-result descriptive partition; not policy tuning",
        "label": args.label,
        "saved_action_targets": partition_summary(action_rows),
        "format_error_targets_without_saved_action": partition_summary(format_rows),
        "all_targets": partition_summary(all_rows),
        "actual_online_materialization": actual_materialization(
            plan, read_jsonl(online / "SERVER_LEDGER.jsonl"), all_rows
        ),
        "interpretation_limit": (
            "A saved action means the online model response was executable, not that "
            "the task was correct. Official resolved accuracy remains the quality metric."
        ),
    }
    output = exact / "EFFECTIVE_REQUEST_SPEED_AUDIT.json"
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
