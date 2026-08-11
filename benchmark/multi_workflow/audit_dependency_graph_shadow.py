#!/usr/bin/env python3
"""Shadow the new dependency graph on frozen rolling agent histories.

This audit is answer blind: it compares only which already visible code
observations the flat and graph selectors mark cold.  It does not use task
resolution, generated continuations, Attention, or hidden repository files.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.coding_reuse_policy import (
    cold_natural_repository_code_candidates,
    dependency_graph_cold_repository_code_candidates,
)


DEFAULT_INPUT = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_dependency_cold_fresh8_20260810/online/"
    "coding_dependency_cold_cost/full_8"
)
DEFAULT_OUTPUT = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_dependency_graph_lcb_20260811/SHADOW_AUDIT.json"
)
ROLLING_GROUPS = 6


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def completed_groups(prefix: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    for message in prefix[2:]:
        if message.get("role") == "assistant":
            if current:
                groups.append(current)
            current = [message]
        elif current is not None:
            current.append(message)
    if current and any(message.get("role") == "tool" for message in current):
        groups.append(current)
    return groups


def assistant_prefixes(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    return [
        messages[:index]
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]


def evidence_map(decision: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["group_sha256"]): item
        for item in decision.get("candidate_evidence") or ()
    }


def audit(input_root: Path) -> dict[str, Any]:
    trajectories = sorted(input_root.glob("*/*.traj.json"))
    if not trajectories:
        raise FileNotFoundError(input_root)
    disagreements: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    parse_statuses: Counter[str] = Counter()
    relation_kinds: Counter[str] = Counter()
    for path in trajectories:
        trajectory = read_json(path)
        instance_id = path.parent.name
        for request_index, prefix in enumerate(
            assistant_prefixes(trajectory["messages"]), start=1
        ):
            groups = completed_groups(prefix)
            selected = groups[-ROLLING_GROUPS:]
            retained = selected if len(selected) < ROLLING_GROUPS else selected[1:]
            _, flat = cold_natural_repository_code_candidates(retained)
            _, graph = dependency_graph_cold_repository_code_candidates(retained)
            flat_map = evidence_map(flat)
            graph_map = evidence_map(graph)
            flat_keys = set(flat_map)
            graph_keys = set(graph_map)
            for status, count in (
                graph.get("source_parse_status_counts") or {}
            ).items():
                parse_statuses[str(status)] += int(count)
            for kind, count in (
                graph.get("dependency_relation_kind_counts") or {}
            ).items():
                relation_kinds[str(kind)] += int(count)
            request_rows.append(
                {
                    "instance_id": instance_id,
                    "request_index": request_index,
                    "flat_cold": len(flat_keys),
                    "graph_cold": len(graph_keys),
                    "flat_only": len(flat_keys - graph_keys),
                    "graph_only": len(graph_keys - flat_keys),
                }
            )
            for direction, keys, rows in (
                ("flat_only_graph_hot", flat_keys - graph_keys, flat_map),
                ("graph_only_flat_hot", graph_keys - flat_keys, graph_map),
            ):
                for key in sorted(keys):
                    item = rows[key]
                    disagreements.append(
                        {
                            "instance_id": instance_id,
                            "request_index": request_index,
                            "direction": direction,
                            "source_group_sha256": key,
                            "source_group_index": item["group_index"],
                            "paths": item["paths"],
                            "symbols": item["symbols"],
                            "graph": item.get("dependency_graph"),
                        }
                    )

    unique_disagreements = {
        (
            row["instance_id"],
            row["direction"],
            row["source_group_sha256"],
        )
        for row in disagreements
    }
    return {
        "status": "COMPLETE",
        "classification": "answer-blind frozen-history selector shadow audit",
        "coverage": {
            "tasks": len(trajectories),
            "requests": len(request_rows),
            "request_observation_rows": sum(
                row["flat_cold"] + row["graph_cold"] for row in request_rows
            ),
        },
        "selector": {
            "requests_with_any_disagreement": sum(
                bool(row["flat_only"] or row["graph_only"])
                for row in request_rows
            ),
            "repeated_disagreement_rows": len(disagreements),
            "unique_task_direction_source_disagreements": len(
                unique_disagreements
            ),
            "flat_only_graph_hot_rows": sum(
                row["direction"] == "flat_only_graph_hot"
                for row in disagreements
            ),
            "graph_only_flat_hot_rows": sum(
                row["direction"] == "graph_only_flat_hot"
                for row in disagreements
            ),
            "parse_status_counts": dict(sorted(parse_statuses.items())),
            "relation_kind_counts": dict(sorted(relation_kinds.items())),
        },
        "disagreements": disagreements,
        "claim_limit": (
            "Selection disagreement motivates causal forks; it is not an "
            "accuracy or TTFT result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.input.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
