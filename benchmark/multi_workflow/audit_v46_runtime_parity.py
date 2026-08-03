#!/usr/bin/env python3
"""Verify V46 bridge coverage and source lifetime safety offline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template
from tokenizers import Tokenizer

from benchmark.multi_workflow.audit_v451_multi_observation_pool import (
    file_sha256,
)
from benchmark.multi_workflow.audit_v45_selected_target_guard import (
    V45,
    _groups,
    _planner,
)
from benchmark.multi_workflow.bridge_reuse_litellm_model import token_ids_hash


V46 = "coding_observed_path_pool_v46"


def replay_trajectory(
    path: Path,
    *,
    tokenizer: Tokenizer,
    chat_template: Template,
    frozen_prompt_hashes: dict[int, str],
) -> dict[str, Any]:
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    base = trajectory["messages"][:2]
    groups = _groups(trajectory["messages"][2:])
    calls = int(trajectory["info"]["model_stats"]["api_calls"])
    request_count = min(calls, len(groups) + 1)
    model = _planner(V46, tokenizer=tokenizer, chat_template=chat_template)
    model._pending_sources = {}
    counts: Counter[str] = Counter()
    rows = []
    for completed_count in range(request_count):
        messages = list(base)
        for group in groups[:completed_count]:
            messages.extend(group)
        prepared = model.prepare_reuse_query(messages, write_sidecar=False)
        request_index = completed_count + 1
        prompt_hash = token_ids_hash(prepared["prompt_ids"])
        if frozen_prompt_hashes.get(request_index) != prompt_hash:
            counts["prompt_hash_mismatches"] += 1
        sources = prepared["sources"]
        targets = prepared["targets"]
        target_source_ids = {
            str(case["source_id"]) for case in targets
        }
        release_source_ids = {
            str(source_id) for source_id in prepared["releases"]
        }
        release_conflicts = sorted(
            target_source_ids & release_source_ids
        )
        counts["requests"] += 1
        counts["prompt_tokens"] += len(prepared["prompt_ids"])
        counts["source_registrations"] += len(sources)
        counts["target_islands"] += len(targets)
        counts["copied_tokens"] += sum(
            int(case["length"]) for case in targets
        )
        if targets:
            counts["target_requests_with_copy"] += 1
        if len(targets) > 1:
            counts["multi_island_target_requests"] += 1
        counts["target_source_release_conflicts"] += len(
            release_conflicts
        )
        counts["max_islands_on_one_target"] = max(
            counts["max_islands_on_one_target"], len(targets)
        )
        counts["max_pool_size"] = max(
            counts["max_pool_size"], len(model._pending_sources)
        )
        rows.append(
            {
                "request_index": request_index,
                "prompt_hash": prompt_hash,
                "prompt_tokens": len(prepared["prompt_ids"]),
                "sources_registered": len(sources),
                "target_islands": len(targets),
                "copied_tokens": sum(
                    int(case["length"]) for case in targets
                ),
                "pool_size": len(model._pending_sources),
                "releases": prepared["releases"],
                "target_source_release_conflicts": release_conflicts,
            }
        )
    return {
        "instance_id": trajectory["instance_id"],
        "trajectory": str(path),
        "trajectory_sha256": file_sha256(path),
        "counts": dict(sorted(counts.items())),
        "requests": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--frozen-v45-audit", type=Path, required=True)
    parser.add_argument("--v453-result", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(
        path
        for path in args.trajectory_root.rglob("*.traj.json")
        if "coding_grounded_observation_island_v40" in path.parts
    )
    frozen = json.loads(args.frozen_v45_audit.read_text(encoding="utf-8"))
    frozen_hashes = {
        str(row["instance_id"]): {
            int(request["request_index"]): str(request["prompt_hashes"][V45])
            for request in row["requests"]
        }
        for row in frozen["trajectories"]
    }
    expected = json.loads(args.v453_result.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    chat_template = Template(
        args.chat_template.read_text(encoding="utf-8"),
        undefined=StrictUndefined,
    )
    trajectories = []
    for path in paths:
        instance_id = json.loads(path.read_text(encoding="utf-8"))["instance_id"]
        trajectories.append(
            replay_trajectory(
                path,
                tokenizer=tokenizer,
                chat_template=chat_template,
                frozen_prompt_hashes=frozen_hashes[instance_id],
            )
        )
    totals: Counter[str] = Counter()
    max_islands = 0
    max_pool = 0
    for row in trajectories:
        counts = row["counts"]
        max_islands = max(max_islands, int(counts.get("max_islands_on_one_target", 0)))
        max_pool = max(max_pool, int(counts.get("max_pool_size", 0)))
        totals.update(
            {
                key: value
                for key, value in counts.items()
                if key not in {"max_islands_on_one_target", "max_pool_size"}
            }
        )
    totals["max_islands_on_one_target"] = max_islands
    totals["max_pool_size"] = max_pool

    parity_fields = (
        "requests",
        "prompt_tokens",
        "source_registrations",
        "target_requests_with_copy",
        "target_islands",
        "multi_island_target_requests",
        "copied_tokens",
        "max_islands_on_one_target",
        "max_pool_size",
    )
    expected_totals = expected["totals"]
    target_request_rate = (
        totals["target_requests_with_copy"] / totals["requests"]
    )
    copied_prompt_fraction = (
        totals["copied_tokens"] / totals["prompt_tokens"]
    )
    lifecycle_gates = {
        "prompt_hash_mismatches_zero": totals["prompt_hash_mismatches"] == 0,
        "target_source_release_conflicts_zero": totals[
            "target_source_release_conflicts"
        ]
        == 0,
        "target_request_rate_at_least_0_65": target_request_rate >= 0.65,
        "copied_prompt_fraction_at_least_0_25": (
            copied_prompt_fraction >= 0.25
        ),
        "max_islands_at_most_3": totals["max_islands_on_one_target"] <= 3,
        "max_pool_size_at_most_3": totals["max_pool_size"] <= 3,
    }
    v453_parity = {
        f"{field}_matches_v453": int(totals[field])
        == int(expected_totals[field])
        for field in parity_fields
    }
    result = {
        "schema": "impactkv-v46-lifecycle-safe-parity-v2",
        "scope": {
            "trajectory_count": len(trajectories),
            "model_requests": 0,
            "gpu_requests": 0,
            "production_bridge_planner": True,
            "prefetch": False,
            "task_accuracy_claimed": False,
            "latency_claimed": False,
        },
        "totals": dict(sorted(totals.items())),
        "rates": {
            "target_request_rate": target_request_rate,
            "copied_prompt_fraction": copied_prompt_fraction,
        },
        "v453_expected_totals": {
            field: expected_totals[field] for field in parity_fields
        },
        "frozen_lifecycle_gates": lifecycle_gates,
        "informational_v453_parity": v453_parity,
        "gpu_canary_eligible": all(lifecycle_gates.values()),
        "inputs": {
            "frozen_v45_audit_sha256": file_sha256(args.frozen_v45_audit),
            "v453_result_sha256": file_sha256(args.v453_result),
            "tokenizer_sha256": file_sha256(args.tokenizer),
            "chat_template_sha256": file_sha256(args.chat_template),
        },
        "trajectories": trajectories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gpu_canary_eligible": result["gpu_canary_eligible"],
                "totals": result["totals"],
                "rates": result["rates"],
                "frozen_lifecycle_gates": lifecycle_gates,
                "informational_v453_parity": v453_parity,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
