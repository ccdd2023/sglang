#!/usr/bin/env python3
"""Attribute Fresh24 outcomes to actual copy exposure and audit exact speed.

The agent arms can diverge before a target is eligible for reuse.  A raw
resolved-count delta therefore cannot by itself be attributed to lossy KV
copying.  This audit joins trajectory process nonces, the online copy plan,
official task outcomes, and the counterbalanced exact-prompt replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/gfy/CodeMAS_Project")
DEFAULT_CAMPAIGN = (
    ROOT / "kvflow-artifacts/impactkv_dependency_graph_fresh24_20260811"
)
NEW_ARM = "coding_dependency_graph_cold_lcb"
CURRENT_ARM = "coding_dependency_cold_cost"
NONCE_PATTERN = re.compile(r"(?:call_|\b)(p\d+-m\d+)(?:_|-)")

EXTERNAL_RESULTS = {
    "cacheblend": (
        ROOT
        / "kvflow-artifacts/impactkv_three_method_coding_benchmark_20260728"
        / "repobench-p/cacheblend/RESULT.json"
    ),
    "kvcomm": (
        ROOT
        / "kvflow-artifacts/impactkv_three_method_coding_benchmark_20260728"
        / "repobench-p/kvcomm/RESULT.json"
    ),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trajectory_nonce_map(online: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(online.glob("*/*.traj.json")):
        match = NONCE_PATTERN.search(path.read_text(encoding="utf-8"))
        if match is None:
            raise ValueError(f"trajectory has no model nonce: {path}")
        nonce = match.group(1)
        task = path.stem.removesuffix(".traj")
        previous = result.setdefault(nonce, task)
        if previous != task:
            raise ValueError(f"nonce {nonce} maps to both {previous} and {task}")
    return result


def plan_nonce(group: dict[str, Any]) -> str:
    match = NONCE_PATTERN.search(str(group["original_target_group_id"]))
    if match is None:
        raise ValueError(f"target group has no model nonce: {group}")
    return match.group(1)


def paired_exact_rows(
    exact_root: Path,
) -> tuple[list[dict[str, Any]], dict[int, list[float]], dict[int, list[float]]]:
    paired: list[dict[str, Any]] = []
    builds: dict[int, list[float]] = defaultdict(list)
    savings: dict[int, list[float]] = defaultdict(list)
    for sequence in ("ab", "ba"):
        dense = read_json(exact_root / sequence / "dense.json")
        reuse = read_json(exact_root / sequence / "reuse.json")
        dense_rows = {
            (int(row["group_index"]), int(row["round_index"])): row
            for row in dense["targets"]
            if not bool(row["warmup"])
        }
        reuse_rows = {
            (int(row["group_index"]), int(row["round_index"])): row
            for row in reuse["targets"]
            if not bool(row["warmup"])
        }
        if set(dense_rows) != set(reuse_rows):
            raise ValueError(f"{sequence}: exact-prompt pairs differ")
        for key in sorted(dense_rows):
            dense_ms = float(dense_rows[key]["ttft_ms"])
            reuse_ms = float(reuse_rows[key]["ttft_ms"])
            row = {
                "sequence": sequence,
                "group_index": key[0],
                "round_index": key[1],
                "dense_ttft_ms": dense_ms,
                "reuse_ttft_ms": reuse_ms,
                "saving_fraction": 1 - reuse_ms / dense_ms,
            }
            paired.append(row)
            savings[key[0]].append(row["saving_fraction"])
        for row in reuse["sources"]:
            builds[int(row["group_index"])].append(float(row["elapsed_ms"]))
    return paired, builds, savings


def outcome_label(left: bool, right: bool) -> str:
    if left and right:
        return "both_resolved"
    if left and not right:
        return "new_damage"
    if not left and right:
        return "new_rescue"
    return "both_unresolved"


def layer_summary(rows: list[dict[str, Any]], label_key: str) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    tasks: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        label = str(row[label_key])
        counts[label] += 1
        tasks[label].append(str(row["task"]))
    return {
        "tasks": len(rows),
        "counts": dict(sorted(counts.items())),
        "task_ids": {key: sorted(value) for key, value in sorted(tasks.items())},
    }


def external_reference() -> dict[str, Any]:
    rows = {}
    provenance = {}
    for label, path in EXTERNAL_RESULTS.items():
        value = read_json(path)
        quality = value["quality"]
        latency = value["latency"]
        rows[label] = {
            "method": value["method"],
            "model": value["model"],
            "dataset": value["dataset"],
            "samples": int(value["samples"]),
            "dense_exact_line": int(quality["dense_exact_line"]),
            "reuse_exact_line": int(quality["reuse_exact_line"]),
            "exact_line_delta_pp": float(quality["exact_line_delta_pp"]),
            "cache_ready_speedup_vs_native_dense": float(
                latency["cache_ready_speedup_vs_native_dense"]
            ),
            "n4_including_build_speedup_vs_native_dense": float(
                latency["build_amortized"]["4"]["speedup_vs_native_dense"]
            ),
        }
        provenance[str(path)] = sha256(path)
    return {
        "rows": rows,
        "rankable_against_fresh24": False,
        "reason": (
            "The native references use Qwen2.5-Coder-3B on a frozen "
            "RepoBench-P next-line workload. Fresh24 uses a Qwen3-Coder-30B "
            "rolling tool agent and official SWE-bench resolution. Only each "
            "method's delta/speedup versus its own native Dense is valid; the "
            "rows are targets for the next shared-adapter experiment, not a "
            "cross-protocol SOTA ranking."
        ),
        "provenance_sha256": provenance,
    }


def analyze(campaign: Path) -> dict[str, Any]:
    result_path = campaign / "RESULT.json"
    fresh = read_json(result_path)
    online = campaign / f"online/{NEW_ARM}/full_24"
    exact_root = campaign / "exact_prompt_speed_abba"
    exact_result_path = exact_root / "RESULT.json"
    exact_result = read_json(exact_result_path)
    plan_path = exact_root / "PLAN.json"
    groups = read_json(plan_path)["groups"]
    nonce_to_task = trajectory_nonce_map(online)

    group_task: dict[int, str] = {}
    for index, group in enumerate(groups):
        nonce = plan_nonce(group)
        if nonce not in nonce_to_task:
            raise ValueError(f"plan nonce absent from trajectories: {nonce}")
        group_task[index] = nonce_to_task[nonce]

    client_rows = [
        row
        for row in read_jsonl(online / "CLIENT_LEDGER.jsonl")
        if row.get("event") == "request_complete"
    ]
    requests_by_task: dict[str, int] = defaultdict(int)
    for row in client_rows:
        nonce = str(row["model_instance_nonce"])
        if nonce not in nonce_to_task:
            raise ValueError(f"client nonce absent from trajectories: {nonce}")
        requests_by_task[nonce_to_task[nonce]] += 1

    pairs, builds, group_savings = paired_exact_rows(exact_root)
    dense_resolved = set(fresh["official"]["dense"]["resolved_ids"])
    current_resolved = set(
        fresh["official"][CURRENT_ARM]["resolved_ids"]
    )
    new_resolved = set(fresh["official"][NEW_ARM]["resolved_ids"])
    registered_tasks = [
        str(row["instance_id"])
        for row in fresh["registration"]["selection"]["instances"]
    ] if "registration" in fresh else [
        str(row["instance_id"])
        for row in read_json(campaign / "CAMPAIGN_REGISTRATION.json")[
            "selection"
        ]["instances"]
    ]

    groups_by_task: dict[str, list[int]] = defaultdict(list)
    for group_index, task in group_task.items():
        groups_by_task[task].append(group_index)
    task_rows = []
    for task in registered_tasks:
        task_group_indices = groups_by_task.get(task, [])
        task_savings = [
            value
            for index in task_group_indices
            for value in group_savings[index]
        ]
        task_rows.append(
            {
                "task": task,
                "agent_requests": requests_by_task.get(task, 0),
                "copy_exposed": bool(task_group_indices),
                "copy_events": len(task_group_indices),
                "copied_tokens": sum(
                    int(groups[index]["copied_tokens"])
                    for index in task_group_indices
                ),
                "dense_resolved": task in dense_resolved,
                "current_resolved": task in current_resolved,
                "new_resolved": task in new_resolved,
                "new_vs_dense": outcome_label(
                    task in dense_resolved, task in new_resolved
                ),
                "new_vs_current": outcome_label(
                    task in current_resolved, task in new_resolved
                ),
                "exact_pairs": len(task_savings),
                "exact_ttft_saving_median": (
                    statistics.median(task_savings) if task_savings else None
                ),
            }
        )

    treated = [row for row in task_rows if row["copy_exposed"]]
    untreated = [row for row in task_rows if not row["copy_exposed"]]
    dense_mean = statistics.fmean(row["dense_ttft_ms"] for row in pairs)
    reuse_mean = statistics.fmean(row["reuse_ttft_ms"] for row in pairs)
    build_mean = statistics.fmean(
        value for values in builds.values() for value in values
    )
    amortized = {}
    for reuse_count in (1, 4, 16):
        amortized[str(reuse_count)] = {
            "target_plus_build_share_ms": reuse_mean + build_mean / reuse_count,
            "speedup_vs_dense": dense_mean
            / (reuse_mean + build_mean / reuse_count),
        }
    break_even = []
    for index in sorted(group_task):
        group_pairs = [row for row in pairs if row["group_index"] == index]
        dense_group = statistics.fmean(
            row["dense_ttft_ms"] for row in group_pairs
        )
        reuse_group = statistics.fmean(
            row["reuse_ttft_ms"] for row in group_pairs
        )
        build_group = statistics.fmean(builds[index])
        delta = dense_group - reuse_group
        break_even.append(
            math.ceil(build_group / delta) if delta > 0 else None
        )

    copied_tokens = int(
        fresh["runtime_descriptive_only"][NEW_ARM]["copied_tokens"]
    )
    copy_events = int(
        fresh["runtime_descriptive_only"][NEW_ARM]["target_copy_events"]
    )
    output = {
        "status": "COMPLETE",
        "classification": (
            "post-outcome treatment-attribution and exact-speed lifecycle audit"
        ),
        "fresh24": {
            "resolved": {
                "dense": len(dense_resolved),
                "current_flat_cold": len(current_resolved),
                "graph_lcb_single_island": len(new_resolved),
            },
            "copy_exposure": {
                "tasks": len(treated),
                "tasks_percent": 100 * len(treated) / len(task_rows),
                "requests": copy_events,
                "requests_percent": 100 * copy_events / len(client_rows),
                "copied_tokens": copied_tokens,
                "fallback_events": int(
                    fresh["runtime_descriptive_only"][NEW_ARM][
                        "target_fallback_events"
                    ]
                ),
            },
            "new_vs_dense_all": layer_summary(task_rows, "new_vs_dense"),
            "new_vs_dense_copy_exposed": layer_summary(
                treated, "new_vs_dense"
            ),
            "new_vs_dense_unexposed": layer_summary(
                untreated, "new_vs_dense"
            ),
            "new_vs_current_copy_exposed": layer_summary(
                treated, "new_vs_current"
            ),
            "new_vs_current_unexposed": layer_summary(
                untreated, "new_vs_current"
            ),
            "interpretation": (
                "The all-task +1 resolved point estimate is not fully "
                "attributable to lossy KV reuse. Rescues or damages in tasks "
                "with zero copy exposure are protocol/repeat variance. The "
                "copy-exposed stratum is the relevant descriptive treatment "
                "audit, but it is post-treatment and not a randomized "
                "accuracy estimate."
            ),
            "task_rows": task_rows,
        },
        "exact_prompt_speed": {
            "measured_pairs": len(pairs),
            "dense_mean_ttft_ms": dense_mean,
            "reuse_mean_ttft_ms": reuse_mean,
            "ratio_of_means_speedup": dense_mean / reuse_mean,
            "paired_saving_median": statistics.median(
                row["saving_fraction"] for row in pairs
            ),
            "paired_win_rate": sum(
                row["saving_fraction"] > 0 for row in pairs
            )
            / len(pairs),
            "mean_source_materialization_ms": build_mean,
            "build_amortized": amortized,
            "median_break_even_reuses": statistics.median(
                value for value in break_even if value is not None
            ),
            "copy_events": int(exact_result["mechanism"]["copy_events"]),
            "expected_copy_events": int(
                exact_result["mechanism"]["expected_copy_events"]
            ),
            "fallback_events": int(
                exact_result["mechanism"]["fallback_events"]
            ),
            "claim_limit": (
                "Cache-ready replay is causal for the copied target requests. "
                "Build-amortized rows are lifecycle scenarios, not measured "
                "rolling-agent end-to-end speedups."
            ),
        },
        "external_native_reference": external_reference(),
        "provenance_sha256": {
            str(result_path): sha256(result_path),
            str(plan_path): sha256(plan_path),
            str(exact_result_path): sha256(exact_result_path),
            str(online / "CLIENT_LEDGER.jsonl"): sha256(
                online / "CLIENT_LEDGER.jsonl"
            ),
        },
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    output = args.output or (campaign / "ATTRIBUTION_AUDIT.json")
    value = analyze(campaign)
    write_json(output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
