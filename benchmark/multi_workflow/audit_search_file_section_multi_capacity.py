#!/usr/bin/env python3
"""Audit whether one-island search targets hide safe multi-island capacity.

The audit reads only already-emitted policy guards and source geometry.  It
does not read official outcomes, generated actions, raw TTFT, NLL, KV distance,
or attention.  It is a mechanism-capacity input to the next preregistration.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = RuntimePaths.from_project(PROJECT).artifacts
SOURCE = ARTIFACTS / "impactkv_common_agent_search_file_section_20260812"
TARGET = ARTIFACTS / "impactkv_search_file_section_multi_capacity_20260813"
ARM = "coding_search_file_section_mean"


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


def main() -> None:
    result_path = TARGET / "RESULT.json"
    if result_path.is_file():
        print(result_path)
        return
    run = SOURCE / f"runs/sglang_canary/{ARM}/full_3"
    rows = read_jsonl(run / "CLIENT_LEDGER.jsonl")
    manifest = read_json(run / "DYNAMIC_MANIFEST.json")
    source_lengths = {
        str(row["source_id"]): int(row["length"])
        for row in manifest.get("sources") or ()
    }
    opportunities = []
    for row in rows:
        decision = row.get("reuse_policy_decision") or {}
        guards = decision.get("target_evidence_guards") or []
        valid = [guard for guard in guards if guard.get("target_evidence_valid")]
        if len(valid) < 2:
            continue
        lengths = [source_lengths[str(guard["source_id"])] for guard in valid]
        opportunities.append(
            {
                "model_instance_nonce": str(row["model_instance_nonce"]),
                "request_index": int(row["request_index"]),
                "target_prompt_tokens": int(row["prompt_tokens"]),
                "currently_selected_islands": int(row.get("target_islands") or 0),
                "currently_selected_tokens": int(
                    row.get("copied_tokens_planned") or 0
                ),
                "version_and_graph_valid_islands": len(valid),
                "available_tokens": sum(lengths),
                "island_lengths": sorted(lengths, reverse=True),
                "all_mean_cost_admitted": all(
                    guard.get("reuse_admitted") for guard in valid
                ),
            }
        )
    task_nonces = sorted(
        {row["model_instance_nonce"] for row in opportunities}
    )
    gate = len(opportunities) >= 4 and len(task_nonces) >= 2
    value = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "outcome-blind mechanism-capacity audit",
        "source_arm": ARM,
        "counts": {
            "requests_with_two_or_more_valid_islands": len(opportunities),
            "task_wrappers_with_capacity": len(task_nonces),
            "currently_selected_islands": sum(
                row["currently_selected_islands"] for row in opportunities
            ),
            "available_valid_islands": sum(
                row["version_and_graph_valid_islands"] for row in opportunities
            ),
            "currently_selected_tokens": sum(
                row["currently_selected_tokens"] for row in opportunities
            ),
            "available_tokens": sum(
                row["available_tokens"] for row in opportunities
            ),
        },
        "opportunities": opportunities,
        "preregistered_capacity_gate": {
            "minimum_requests": 4,
            "minimum_task_wrappers": 2,
            "passed": gate,
        },
        "excluded_fields": [
            "official accuracy",
            "generated action text",
            "raw or summarized TTFT",
            "NLL",
            "KV deviation",
            "attention",
        ],
    }
    write_json(
        TARGET / "REGISTRATION.json",
        {
            "schema_version": 1,
            "status": "FROZEN_CAPACITY_RULE_BEFORE_AUDIT",
            "rule": (
                "keep only requests with at least two target-time guards that "
                "are version-valid, dependency-graph-cold, mean-cost-admitted; "
                "pass at >=4 requests across >=2 task wrappers"
            ),
        },
    )
    write_json(result_path, value)
    print(result_path)


if __name__ == "__main__":
    main()
