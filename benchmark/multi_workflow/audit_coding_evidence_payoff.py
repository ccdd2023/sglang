#!/usr/bin/env python3
"""Audit natural large-span opportunities for coding-evidence payoff reuse.

The input must be a completed General-8K bridge-agent run.  This is a post-hoc
motivation audit: it measures how often the V7 feature would change the
physical copy budget, but it does not estimate accuracy or causal speedup.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.coding_reuse_policy import (
    is_successful_readonly_evidence,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def turn_groups(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages[2:]:
        if message.get("role") == "assistant" and current:
            groups.append(current)
            current = []
        current.append(message)
    if current:
        groups.append(current)
    return groups


def audit(run_dir: Path, registration: Path) -> dict[str, Any]:
    frozen = read_json(registration)
    instance_ids = [row["instance_id"] for row in frozen["instances"]]
    trajectories = {
        instance_id: turn_groups(
            read_json(
                run_dir
                / instance_id
                / f"{instance_id}.traj.json"
            )["messages"]
        )
        for instance_id in instance_ids
    }
    opportunities: list[dict[str, Any]] = []
    request_rows = [
        row
        for row in read_jsonl(run_dir / "CLIENT_LEDGER.jsonl")
        if row.get("event") == "request_complete"
    ]
    for row in request_rows:
        match = re.search(
            r"-m(\d+)$", str(row["model_instance_nonce"])
        )
        if match is None:
            raise ValueError("model instance nonce does not end in -mN")
        instance_index = int(match.group(1)) - 1
        if not 0 <= instance_index < len(instance_ids):
            raise ValueError("model instance nonce exceeds registration")
        instance_id = instance_ids[instance_index]
        request_index = int(row["request_index"])
        groups = trajectories[instance_id]
        group_index = request_index - 2
        if not 0 <= group_index < len(groups):
            continue
        decision = row.get("reuse_policy_decision", {})
        candidate_tokens = int(decision.get("selected_tokens", 0) or 0)
        if candidate_tokens < 5120:
            continue
        if not is_successful_readonly_evidence(groups[group_index]):
            continue
        v7_tokens = min(candidate_tokens, 6144)
        opportunities.append(
            {
                "instance_id": instance_id,
                "source_request_index": request_index,
                "target_request_index": request_index + 1,
                "candidate_tokens": candidate_tokens,
                "general_4k_tokens": 4096,
                "v7_tokens": v7_tokens,
                "marginal_tokens": v7_tokens - 4096,
            }
        )
    by_task = Counter(
        row["instance_id"] for row in opportunities
    )
    return {
        "classification": (
            "post-hoc natural-trajectory opportunity audit; not a causal "
            "speed or accuracy result"
        ),
        "input_run": str(run_dir),
        "requests_audited": len(request_rows),
        "opportunities": len(opportunities),
        "tasks_with_opportunity": len(by_task),
        "marginal_copy_tokens": sum(
            row["marginal_tokens"] for row in opportunities
        ),
        "opportunities_by_task": dict(sorted(by_task.items())),
        "rows": opportunities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir, args.registration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
