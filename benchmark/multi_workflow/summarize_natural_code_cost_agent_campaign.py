#!/usr/bin/env python3
"""Summarize official accuracy and bounded latency evidence for fresh9."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_CAMPAIGN = ARTIFACTS / "impactkv_natural_code_cost_agent_20260808"
POLICY = "coding_natural_code_cost"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(fraction * len(ordered)) - 1))]


def official(run_dir: Path) -> dict[str, Any]:
    value = read_json(run_dir / "OFFICIAL_RESULT.json")
    if value.get("report") is None:
        raise ValueError(f"official report is absent: {run_dir}")
    return dict(value["report"])


def request_rows(run_dir: Path) -> list[dict[str, Any]]:
    return [
        row
        for row in read_jsonl(run_dir / "CLIENT_LEDGER.jsonl")
        if row.get("event") == "request_complete"
    ]


def latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ttft = [1_000 * float(row["ttft_seconds"]) for row in rows]
    prompt = [int(row["prompt_tokens"]) for row in rows]
    return {
        "requests": len(rows),
        "prompt_tokens_median": median([float(value) for value in prompt]),
        "prompt_tokens_p95": quantile([float(value) for value in prompt], 0.95),
        "ttft_ms_median": median(ttft),
        "ttft_ms_p95": quantile(ttft, 0.95),
        "ttft_ms_sum": sum(ttft),
    }


def nearest_length_diagnostic(
    policy_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare copy requests to nearest-length Dense requests in one run.

    Unregistered requests in the policy arm execute Dense on the same server.
    This controls prompt length more tightly than cross-trajectory aggregates,
    but is not an exact-prompt pair and remains descriptive.
    """

    targets = [row for row in policy_rows if row.get("target_registered")]
    controls = [
        row for row in policy_rows if not row.get("target_registered")
    ]
    pairs = []
    for target in targets:
        target_tokens = int(target["prompt_tokens"])
        control = min(
            controls,
            key=lambda row: abs(int(row["prompt_tokens"]) - target_tokens),
        )
        control_tokens = int(control["prompt_tokens"])
        target_ttft = 1_000 * float(target["ttft_seconds"])
        control_ttft = 1_000 * float(control["ttft_seconds"])
        relative_length_gap = abs(control_tokens - target_tokens) / target_tokens
        pairs.append(
            {
                "target_prompt_tokens": target_tokens,
                "control_prompt_tokens": control_tokens,
                "relative_prompt_length_gap": relative_length_gap,
                "target_ttft_ms": target_ttft,
                "control_ttft_ms": control_ttft,
                "ttft_saving_fraction": 1 - target_ttft / control_ttft,
            }
        )
    close = [row for row in pairs if row["relative_prompt_length_gap"] <= 0.05]
    return {
        "classification": "same-server nearest-prompt-length descriptive diagnostic",
        "not_exact_prompt_pair": True,
        "all_pairs": len(pairs),
        "within_5pct_prompt_length_pairs": len(close),
        "within_5pct_median_ttft_saving_fraction": median(
            [row["ttft_saving_fraction"] for row in close]
        ),
        "within_5pct_win_rate": (
            sum(row["ttft_saving_fraction"] > 0 for row in close) / len(close)
            if close
            else None
        ),
        "pairs": pairs,
    }


def summarize(campaign: Path) -> dict[str, Any]:
    dense_dir = campaign / "online/dense/full_9"
    policy_dir = campaign / f"online/{POLICY}/full_9"
    dense_official = official(dense_dir)
    policy_official = official(policy_dir)
    dense_resolved = set(dense_official["resolved_ids"])
    policy_resolved = set(policy_official["resolved_ids"])
    registration = read_json(campaign / "CAMPAIGN_REGISTRATION.json")
    all_ids = list(registration["selection"]["instance_ids"])
    per_task = [
        {
            "instance_id": instance_id,
            "dense_resolved": instance_id in dense_resolved,
            "policy_resolved": instance_id in policy_resolved,
            "paired_outcome": (
                "rescue"
                if instance_id in policy_resolved - dense_resolved
                else "damage"
                if instance_id in dense_resolved - policy_resolved
                else "both_resolved"
                if instance_id in dense_resolved
                else "both_unresolved"
            ),
        }
        for instance_id in all_ids
    ]
    policy_rows = request_rows(policy_dir)
    dense_rows = request_rows(dense_dir)
    runtime = read_json(policy_dir / "RUNTIME_SUMMARY.json")
    exact_path = campaign / "exact_prompt_speed/RESULT.json"
    exact_speed = read_json(exact_path) if exact_path.exists() else None
    result = {
        "status": "COMPLETE",
        "classification": (
            "fresh9 exploratory official-accuracy campaign after a no-copy canary"
        ),
        "accuracy": {
            "denominator": len(all_ids),
            "dense_resolved": len(dense_resolved),
            "policy_resolved": len(policy_resolved),
            "rescue": len(policy_resolved - dense_resolved),
            "damage": len(dense_resolved - policy_resolved),
            "both_resolved": len(dense_resolved & policy_resolved),
            "both_unresolved": len(
                set(all_ids) - (dense_resolved | policy_resolved)
            ),
            "per_task": per_task,
        },
        "physical_reuse": {
            "source_materialized_events": runtime[
                "source_materialized_events"
            ],
            "host_source_materialized_events": runtime[
                "source_materialized_host_events"
            ],
            "target_copy_events": runtime["target_copy_events"],
            "copied_tokens": runtime["copied_tokens"],
            "target_fallback_events": runtime["target_fallback_events"],
            "prefetch": False,
        },
        "latency": {
            "dense_free_running": latency_summary(dense_rows),
            "policy_free_running": latency_summary(policy_rows),
            "cross_trajectory_warning": (
                "request counts and prompt distributions differ; raw arm TTFT "
                "is not a causal speed comparison"
            ),
            "policy_same_server_nearest_length": nearest_length_diagnostic(
                policy_rows
            ),
            "exact_same_prompt_replay_complete": exact_speed is not None,
            "exact_same_prompt_replay": exact_speed,
        },
        "external_baselines": {
            "CacheBlend": "not run as rolling agent; fixed-prompt native reference only",
            "KVCOMM": "not run as rolling agent; two-message fixed-prompt native reference only",
            "ranking_allowed": False,
        },
    }
    output = campaign / "RESULT.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    args = parser.parse_args()
    print(json.dumps(summarize(args.campaign), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
