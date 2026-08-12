#!/usr/bin/env python3
"""Describe search-file-section canary behavior against frozen Dense.

This post-run audit never selects tasks or changes policy.  It compares every
successfully parsed model request by request index, records the subset exposed
to physical lossy reuse, and separately accounts for copied requests whose
model output became a FormatError and therefore has no assistant action in the
saved trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.analyze_graph_mean_canary_counterfactual import (
    official_instance,
    requests,
)
from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = RuntimePaths.from_project(PROJECT).artifacts
BASE = ARTIFACTS / "impactkv_common_agent_baselines_fresh24_20260812"
SEARCH = ARTIFACTS / "impactkv_common_agent_search_file_section_20260812"
ARM = "coding_search_file_section_mean"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def compare_task(dense_path: Path, search_path: Path) -> dict[str, Any]:
    dense_value = read_json(dense_path)
    search_value = read_json(search_path)
    dense_rows = requests(dense_path)
    search_rows = requests(search_path)
    dense_by_request = {row["request_index"]: row for row in dense_rows}
    common = [
        (row, dense_by_request[row["request_index"]])
        for row in search_rows
        if row["request_index"] in dense_by_request
    ]
    treated = [row for row in search_rows if row["target_registered"]]
    dense_submission = str((dense_value.get("info") or {}).get("submission") or "")
    search_submission = str((search_value.get("info") or {}).get("submission") or "")
    return {
        "instance_id": search_path.parent.name,
        "dense_successful_requests": len(dense_rows),
        "search_successful_requests": len(search_rows),
        "common_request_indices": len(common),
        "common_inputs_identical": sum(
            search["input_ids_sha256"] == dense["input_ids_sha256"]
            for search, dense in common
        ),
        "common_actions_identical": sum(
            search["command"] == dense["command"] for search, dense in common
        ),
        "successful_copy_exposed_requests": len(treated),
        "successful_copy_exposed_tokens": sum(
            int(row["copied_tokens_planned"]) for row in treated
        ),
        "all_successful_copy_actions_match_dense": all(
            row["request_index"] in dense_by_request
            and row["input_ids_sha256"]
            == dense_by_request[row["request_index"]]["input_ids_sha256"]
            and row["command"]
            == dense_by_request[row["request_index"]]["command"]
            for row in treated
        ),
        "dense_exit_status": (dense_value.get("info") or {}).get("exit_status"),
        "search_exit_status": (search_value.get("info") or {}).get("exit_status"),
        "dense_submission_sha256": sha256_text(dense_submission),
        "search_submission_sha256": sha256_text(search_submission),
        "final_submission_identical": dense_submission == search_submission,
        "submission_characters": len(search_submission),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=SEARCH / "SEARCH_FILE_SECTION_CANARY_COUNTERFACTUAL.json",
    )
    args = parser.parse_args()
    dense = BASE / "runs/sglang_formal/dense/full_24"
    search = SEARCH / f"runs/sglang_canary/{ARM}/full_3"
    tasks = []
    for search_path in sorted(search.glob("*/*.traj.json")):
        instance_id = search_path.parent.name
        row = compare_task(
            dense / instance_id / f"{instance_id}.traj.json", search_path
        )
        row["official"] = {
            "dense": official_instance(dense, instance_id),
            "search": official_instance(search, instance_id),
        }
        tasks.append(row)
    if not tasks:
        raise RuntimeError("search-file-section canary trajectories absent")
    runtime = read_json(search / "RUNTIME_SUMMARY.json")
    physical = int(runtime.get("target_copy_events") or 0)
    successful_exposed = sum(
        int(row["successful_copy_exposed_requests"]) for row in tasks
    )
    value = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "post-run descriptive attribution; not policy tuning",
        "physical_copy_exposed_requests": physical,
        "successful_copy_exposed_requests": successful_exposed,
        "format_error_copy_exposures_without_assistant_action": (
            physical - successful_exposed
        ),
        "all_common_successful_inputs_identical": all(
            row["common_inputs_identical"] == row["common_request_indices"]
            for row in tasks
        ),
        "all_common_successful_actions_identical": all(
            row["common_actions_identical"] == row["common_request_indices"]
            for row in tasks
        ),
        "all_successful_copy_actions_match_dense": all(
            row["all_successful_copy_actions_match_dense"] for row in tasks
        ),
        "all_final_submissions_identical": all(
            row["final_submission_identical"] for row in tasks
        ),
        "tasks": tasks,
        "interpretation_limit": (
            "Identical parsed actions and final submissions on three tasks show no "
            "observed canary behavior damage. FormatError responses contain no "
            "assistant action to compare, and 0/3 resolved cannot establish quality."
        ),
    }
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
