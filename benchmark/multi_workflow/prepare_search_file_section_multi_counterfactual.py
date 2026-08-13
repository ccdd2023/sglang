#!/usr/bin/env python3
"""Preregister the three-island natural search-file counterfactual."""

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
SOURCE = ARTIFACTS / "impactkv_common_agent_search_file_section_20260812"
CAPACITY = ARTIFACTS / "impactkv_search_file_section_multi_capacity_20260813"
TARGET = ARTIFACTS / "impactkv_common_agent_search_file_section_multi_20260813"
ARM = "coding_search_file_section_multi_mean"
SOURCE_ARM = "coding_search_file_section_mean"
SINGLE_ACTION_SPEED = 1.1732503727293053
BASELINE_N1_MAX = 1.053785


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
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
    registration = TARGET / "SEARCH_FILE_MULTI_PREREGISTRATION.json"
    if registration.is_file():
        print(registration)
        return
    if TARGET.exists() and any(TARGET.iterdir()):
        raise FileExistsError(f"nonempty unregistered campaign: {TARGET}")
    capacity_path = CAPACITY / "RESULT.json"
    capacity = read(capacity_path)
    if not (capacity.get("preregistered_capacity_gate") or {}).get("passed"):
        raise RuntimeError("multi-island mechanism capacity gate failed")
    canary = read(SOURCE / "CANARY4.json")
    formal = read(SOURCE / "FROZEN_FRESH24.json")
    canary_ids = [str(row["instance_id"]) for row in canary]
    source_canary_registration = read(SOURCE / "BRIDGE_CANARY4_REGISTRATION.json")
    source_formal_registration = read(SOURCE / "BRIDGE_FRESH24_REGISTRATION.json")

    TARGET.mkdir(parents=True)
    write(TARGET / "CANARY4.json", canary)
    write(TARGET / "FROZEN_FRESH24.json", formal)
    write(
        TARGET / "BRIDGE_CANARY4_REGISTRATION.json",
        {
            **source_canary_registration,
            "registration_id": "impactkv-search-file-multi-canary3-20260813",
        },
    )
    write(
        TARGET / "BRIDGE_FRESH24_REGISTRATION.json",
        {
            **source_formal_registration,
            "registration_id": "impactkv-search-file-multi-fresh24-20260813",
        },
    )
    for scope in ("canary_dataset", "formal_dataset"):
        (TARGET / scope).mkdir()
        shutil.copy2(SOURCE / scope / "test.jsonl", TARGET / scope / "test.jsonl")
    write(
        registration,
        {
            "schema_version": 1,
            "status": "FROZEN_BEFORE_MULTI_ISLAND_MODEL_REQUESTS",
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "arm": ARM,
            "source_arm": SOURCE_ARM,
            "motivation": {
                "capacity_result": str(capacity_path),
                "requests_with_two_or_more_valid_islands": capacity["counts"][
                    "requests_with_two_or_more_valid_islands"
                ],
                "task_wrappers_with_capacity": capacity["counts"][
                    "task_wrappers_with_capacity"
                ],
                "available_valid_islands": capacity["counts"][
                    "available_valid_islands"
                ],
                "currently_selected_islands": capacity["counts"][
                    "currently_selected_islands"
                ],
            },
            "single_variable_intervention": (
                "Increase only the maximum target islands from one to three "
                "when the already-frozen per-file version guard, dependency-graph-"
                "cold guard, positive mean cost gate, and non-overlap check all pass."
            ),
            "unchanged": [
                "literal contiguous path-prefixed file boundaries",
                "per-file version validation",
                "dependency-graph-hot Dense recomputation",
                "positive frozen mean cost admission per island",
                "rolling-6 prompt construction",
                "4096 token per-island cap",
                "no ordinary radix prefix reuse",
                "no prefetch",
            ],
            "canary_ids": canary_ids,
            "formal_ids": [str(row["instance_id"]) for row in formal],
            "gates": {
                "canary_manifest_islands_per_target_min": 2,
                "target_fallback_events_max": 0,
                "first_prompt_identity_against_frozen_dense": True,
                "exact_prompt_input_identity": True,
                "official_accuracy_no_degradation_against_frozen_dense": True,
                "canary_saved_action_ratio_of_sums_speedup_gt_single_island": (
                    SINGLE_ACTION_SPEED
                ),
                "formal_observed_online_lifecycle_speedup_gt_native_baseline_n1": (
                    BASELINE_N1_MAX
                ),
            },
            "inputs": {
                "capacity_sha256": sha256(capacity_path),
                "canary_snapshot_sha256": sha256(SOURCE / "CANARY4.json"),
                "formal_snapshot_sha256": sha256(SOURCE / "FROZEN_FRESH24.json"),
                "canary_dataset_sha256": sha256(SOURCE / "canary_dataset/test.jsonl"),
                "formal_dataset_sha256": sha256(SOURCE / "formal_dataset/test.jsonl"),
            },
            "protected": {
                "prefetch": False,
                "ordinary_radix_prefix_reuse": False,
                "exact_only_reuse": False,
                "paper_modified": False,
                "old_dirty_checkout_modified": False,
                "old_preregistration_thresholds_modified": False,
            },
        },
    )
    write(
        TARGET / "AUTOMATED_SEARCH_FILE_MULTI_STATUS.json",
        {
            "schema_version": 1,
            "state": "registered",
            "model_requests_issued": 0,
            "jobs": {},
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(registration)


if __name__ == "__main__":
    main()
