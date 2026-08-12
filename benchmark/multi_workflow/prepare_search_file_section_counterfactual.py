#!/usr/bin/env python3
"""Preregister the file-bounded search-result lossy-reuse counterfactual."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = RuntimePaths.from_project(PROJECT).artifacts
SOURCE = ARTIFACTS / "impactkv_common_agent_baselines_fresh24_20260812"
AUDIT = ARTIFACTS / "impactkv_search_file_module_audit_20260812"
TARGET = ARTIFACTS / "impactkv_common_agent_search_file_section_20260812"
ARM = "coding_search_file_section_mean"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    registration_path = TARGET / "SEARCH_FILE_SECTION_PREREGISTRATION.json"
    if registration_path.is_file():
        print(registration_path)
        return
    if TARGET.exists() and any(TARGET.iterdir()):
        raise FileExistsError(f"nonempty unregistered campaign: {TARGET}")
    audit_registration = AUDIT / "REGISTRATION.json"
    audit_result_path = AUDIT / "RESULT.json"
    result = read_json(audit_result_path)
    opportunities = result.get("opportunities") or []
    task_ids = sorted({str(row["instance_id"]) for row in opportunities})
    if len(opportunities) < 4 or len(task_ids) < 2:
        raise RuntimeError("pre-registered search-module capacity gate did not pass")
    # Every capacity task is used; no task outcome or TTFT ranks them.
    canary_ids = task_ids
    formal_path = SOURCE / "FROZEN_FRESH24.json"
    formal = read_json(formal_path)
    by_id = {str(row["instance_id"]): row for row in formal}
    missing = [value for value in canary_ids if value not in by_id]
    if missing:
        raise ValueError(f"audit tasks absent from Fresh24: {missing}")
    canary = [by_id[value] for value in canary_ids]
    source_registration = read_json(SOURCE / "BRIDGE_FRESH24_REGISTRATION.json")
    dataset_rows = [
        json.loads(line)
        for line in (SOURCE / "formal_dataset/test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    dataset_by_id = {str(row["instance_id"]): row for row in dataset_rows}

    TARGET.mkdir(parents=True)
    write_json(TARGET / "FROZEN_FRESH24.json", formal)
    write_json(TARGET / "CANARY4.json", canary)
    write_json(
        TARGET / "BRIDGE_FRESH24_REGISTRATION.json",
        {
            **source_registration,
            "registration_id": "impactkv-search-file-section-fresh24-20260812",
        },
    )
    write_json(
        TARGET / "BRIDGE_CANARY4_REGISTRATION.json",
        {
            **source_registration,
            "registration_id": "impactkv-search-file-section-canary4-20260812",
            "instances": [{"instance_id": value} for value in canary_ids],
        },
    )
    (TARGET / "formal_dataset").mkdir()
    shutil.copy2(
        SOURCE / "formal_dataset/test.jsonl", TARGET / "formal_dataset/test.jsonl"
    )
    (TARGET / "canary_dataset").mkdir()
    with (TARGET / "canary_dataset/test.jsonl").open("w", encoding="utf-8") as stream:
        for instance_id in canary_ids:
            stream.write(json.dumps(dataset_by_id[instance_id], sort_keys=True) + "\n")

    write_json(
        registration_path,
        {
            "schema_version": 1,
            "status": "FROZEN_BEFORE_SEARCH_FILE_SECTION_MODEL_REQUESTS",
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "arm": ARM,
            "motivation": {
                "capacity_opportunities": len(opportunities),
                "capacity_tasks": len(task_ids),
                "audit_outcome_blind": True,
                "audit_result": str(audit_result_path),
            },
            "single_variable_intervention": (
                "Compared with dependency-graph-cold mean reuse, expand source "
                "extraction from direct single-file reads to literal contiguous "
                "file-prefixed sections inside successful search output. Keep "
                "per-file version validation, graph-hot recomputation, positive "
                "frozen mean cost admission, one target island, no ordinary "
                "prefix reuse, and no prefetch."
            ),
            "canary_ids": canary_ids,
            "formal_ids": [str(row["instance_id"]) for row in formal],
            "gates": {
                "canary_physical_target_copy_events_min": 1,
                "fallback_events_max": 0,
                "first_prompt_identity_against_frozen_dense": True,
                "exact_prompt_input_identity": True,
                "cache_ready_speedup_min": 1.0,
                "official_accuracy_no_degradation_against_frozen_dense": True,
            },
            "inputs": {
                "capacity_registration_sha256": sha256(audit_registration),
                "capacity_result_sha256": sha256(audit_result_path),
                "fresh24_snapshot_sha256": sha256(formal_path),
                "formal_dataset_sha256": sha256(
                    SOURCE / "formal_dataset/test.jsonl"
                ),
            },
            "protected": {
                "prefetch": False,
                "ordinary_radix_prefix_reuse": False,
                "fixed_token_islands": False,
                "paper_modified": False,
                "old_dirty_checkout_modified": False,
                "old_preregistration_thresholds_modified": False,
            },
        },
    )
    write_json(
        TARGET / "AUTOMATED_SEARCH_FILE_SECTION_STATUS.json",
        {
            "schema_version": 1,
            "state": "registered",
            "model_requests_issued": 0,
            "jobs": {},
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(registration_path)


if __name__ == "__main__":
    main()
