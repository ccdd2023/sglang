#!/usr/bin/env python3
"""Audit V45's version-evidence motivation on completed V40 trajectories.

This is a policy-only, answer-blind audit.  It does not replay a model, claim
task accuracy, estimate tokens, or issue GPU requests.  It asks two narrower
questions before V45 can be promoted:

1. Did V40 expose an eligible observation to a write completed before the
   next target request (the cross-request invalidation window)?
2. Do explicit, disjoint Python symbols recover any otherwise file-stale
   observation without relaxing an ambiguous mutation?
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from benchmark.multi_workflow.coding_reuse_policy import (
    grounded_observation_candidates,
    repository_observation_symbols,
    repository_paths,
    tool_observation_sha256,
    versioned_evidence_target_guard,
    versioned_symbol_observation_candidates,
)


def turn_groups(messages: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Match the bridge's assistant-led completed-interaction grouping."""

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "assistant" and current:
            groups.append(current)
            current = []
        current.append(message)
    if current:
        groups.append(current)
    return groups


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pending_evidence(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_observation_sha256": tool_observation_sha256(group),
        "source_paths": sorted(repository_paths(group)),
        "source_symbols": sorted(repository_observation_symbols(group)),
    }


def audit_trajectory(path: Path, *, rolling_groups: int) -> dict[str, Any]:
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    groups = turn_groups(trajectory["messages"][2:])
    counts: Counter[str] = Counter()
    target_invalidation_reasons: Counter[str] = Counter()
    source_invalidation_reasons: Counter[str] = Counter()

    for completed_count in range(rolling_groups, len(groups) + 1):
        selected = groups[completed_count - rolling_groups : completed_count]
        retained = selected[1:]
        v40_candidates, v40 = grounded_observation_candidates(retained)
        v45_candidates, v45 = versioned_symbol_observation_candidates(retained)
        counts["source_windows"] += 1
        counts["v40_candidate_instances"] += len(v40_candidates)
        counts["v45_candidate_instances"] += len(v45_candidates)
        counts["v45_source_invalidations"] += v45[
            "version_invalidated_observations"
        ]
        counts["symbol_disjoint_source_preservations"] += v45[
            "symbol_disjoint_preservations"
        ]
        source_invalidation_reasons.update(
            v45["version_invalidation_reasons"]
        )

        v40_indices = set(v40["candidate_group_indices"])
        v45_indices = set(v45["candidate_group_indices"])
        added = v45_indices - v40_indices
        counts["v45_added_candidate_instances"] += len(added)
        evidence_by_index = {
            item["group_index"]: item for item in v45["candidate_evidence"]
        }
        counts["ambiguous_relaxation_violations"] += sum(
            not evidence_by_index[index]["symbol_disjoint_mutations"]
            for index in added
        )

        if completed_count >= len(groups):
            continue
        target_selected = groups[
            completed_count + 1 - rolling_groups : completed_count + 1
        ]
        for index in v40["candidate_group_indices"]:
            guard = versioned_evidence_target_guard(
                _pending_evidence(retained[index]), target_selected
            )
            counts["v40_candidate_next_target_checks"] += 1
            if not guard["target_evidence_valid"]:
                counts["v40_candidate_next_target_invalidations"] += 1
                target_invalidation_reasons[str(guard["reason"])] += 1
        for item in v45["candidate_evidence"]:
            guard = versioned_evidence_target_guard(
                {
                    "source_observation_sha256": item["observation_sha256"],
                    "source_paths": item["paths"],
                    "source_symbols": item["symbols"],
                },
                target_selected,
            )
            counts["v45_candidate_next_target_checks"] += 1
            if not guard["target_evidence_valid"]:
                counts["v45_target_guard_invalidations"] += 1

    return {
        "instance_id": trajectory.get("instance_id"),
        "trajectory": str(path),
        "trajectory_sha256": file_sha256(path),
        "groups": len(groups),
        "counts": dict(sorted(counts.items())),
        "source_invalidation_reasons": dict(
            sorted(source_invalidation_reasons.items())
        ),
        "target_invalidation_reasons": dict(
            sorted(target_invalidation_reasons.items())
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rolling-groups", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(
        path
        for path in args.trajectory_root.rglob("*.traj.json")
        if "coding_grounded_observation_island_v40" in path.parts
    )
    if not paths:
        raise SystemExit("no V40 trajectories found")
    rows = [
        audit_trajectory(path, rolling_groups=args.rolling_groups)
        for path in paths
    ]
    totals: Counter[str] = Counter()
    source_reasons: Counter[str] = Counter()
    target_reasons: Counter[str] = Counter()
    for row in rows:
        totals.update(row["counts"])
        source_reasons.update(row["source_invalidation_reasons"])
        target_reasons.update(row["target_invalidation_reasons"])
    interpretation = {
        "cross_request_gap_observed": (
            totals["v40_candidate_next_target_invalidations"] > 0
        ),
        "symbol_relaxation_opportunity_observed": (
            totals["v45_added_candidate_instances"] > 0
        ),
        "ambiguous_mutation_was_relaxed": (
            totals["ambiguous_relaxation_violations"] > 0
        ),
    }
    interpretation["gpu_promotion_eligible"] = bool(
        interpretation["cross_request_gap_observed"]
        and interpretation["symbol_relaxation_opportunity_observed"]
        and not interpretation["ambiguous_mutation_was_relaxed"]
    )
    result = {
        "schema": "impactkv-v45-versioned-evidence-motivation-v1",
        "scope": {
            "answer_blind": True,
            "gpu_requests": 0,
            "task_accuracy_claimed": False,
            "token_or_latency_claimed": False,
            "rolling_history_groups": args.rolling_groups,
            "trajectory_count": len(rows),
        },
        "totals": dict(sorted(totals.items())),
        "source_invalidation_reasons": dict(sorted(source_reasons.items())),
        "target_invalidation_reasons": dict(sorted(target_reasons.items())),
        "interpretation": interpretation,
        "trajectories": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**interpretation, **result["scope"]}, sort_keys=True))


if __name__ == "__main__":
    main()
