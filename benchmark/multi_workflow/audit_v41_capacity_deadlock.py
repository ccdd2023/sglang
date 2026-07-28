#!/usr/bin/env python3
"""Freeze the V41 paired-source capacity deadlock diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v41_v40_independent_20260728"
FAILED = (
    "astropy__astropy-14995",
    "psf__requests-1142",
)
V40 = "coding_grounded_observation_island_v40"
GENERAL = "general"
MAX_NEW_TOKENS = 2048
KV_CAPACITY_TOKENS = 14482


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _task_audit(output: Path, instance_id: str) -> dict[str, Any]:
    task = output / "tasks" / instance_id
    manifest_path = task / "run" / "DYNAMIC_MANIFEST.json"
    server_path = task / "run" / "SERVER_LEDGER.jsonl"
    log_path = output / "orchestration_logs" / instance_id / "run.log"
    manifest = read_json(manifest_path)
    server = _jsonl(server_path)
    log = log_path.read_text(encoding="utf-8")
    active_device_sources = {
        str(row["source_id"]): int(row["tokens"])
        for row in server
        if row.get("event") == "source_materialized"
    }
    cases_by_id = {
        str(row["case_id"]): row for row in manifest["cases"]
    }
    for row in server:
        if (
            row.get("event") == "target_complete"
            and row.get("source_released")
        ):
            case = cases_by_id[str(row["case_id"])]
            active_device_sources.pop(
                str(case.get("source_id") or case["case_id"]),
                None,
            )
    resident_source_tokens = sum(active_device_sources.values())
    targets_started = [
        row
        for row in server
        if row.get("event") == "target_ordinary_prefix_matched"
    ]
    cases = manifest["cases"]
    pending_cases = [
        row
        for row in cases
        if row.get("reuse_enabled", True)
        and row["case_id"]
        not in {
            started["case_id"] for started in targets_started
        }
    ]
    if not pending_cases:
        raise AssertionError(f"{instance_id}: no pending reuse target")
    pending = pending_cases[0]
    minimum_prompt_tokens = (
        int(pending["target_start"]) + int(pending["length"]) + 1
    )
    minimum_admission_tokens = minimum_prompt_tokens + MAX_NEW_TOKENS
    remaining_device_tokens = KV_CAPACITY_TOKENS - resident_source_tokens
    if "httpcore.ReadTimeout: timed out" not in log:
        raise AssertionError(f"{instance_id}: timeout evidence missing")
    if minimum_admission_tokens <= remaining_device_tokens:
        raise AssertionError(
            f"{instance_id}: frozen lower bound does not prove capacity block"
        )
    if instance_id == FAILED[0] and targets_started:
        raise AssertionError("Astropy unexpectedly entered target staging")
    if instance_id == FAILED[1]:
        completed_v40 = [
            row
            for row in server
            if row.get("event") == "target_complete"
            and row.get("policy_label") == V40
        ]
        if not completed_v40:
            raise AssertionError("Requests V40 control target did not complete")
        if pending.get("policy_label") != GENERAL:
            raise AssertionError("Requests pending target is not General")
    return {
        "instance_id": instance_id,
        "classification": "PAIRED_SOURCE_DEVICE_CAPACITY_DEADLOCK",
        "resident_device_source_tokens": resident_source_tokens,
        "device_kv_capacity_tokens": KV_CAPACITY_TOKENS,
        "remaining_device_tokens": remaining_device_tokens,
        "pending_policy": pending["policy_label"],
        "pending_case_id": pending["case_id"],
        "pending_target_start": int(pending["target_start"]),
        "pending_copy_tokens": int(pending["length"]),
        "minimum_prompt_tokens": minimum_prompt_tokens,
        "frozen_max_new_tokens": MAX_NEW_TOKENS,
        "minimum_admission_tokens": minimum_admission_tokens,
        "capacity_deficit_lower_bound_tokens": (
            minimum_admission_tokens - remaining_device_tokens
        ),
        "pending_target_staging_events_before_timeout": sum(
            row.get("case_id") == pending["case_id"]
            for row in targets_started
        ),
        "active_device_source_ids": sorted(active_device_sources),
        "midstream_read_timeout": True,
        "inputs": {
            "manifest_sha256": sha256(manifest_path),
            "server_ledger_sha256": sha256(server_path),
            "run_log_sha256": sha256(log_path),
        },
    }


def run(output: Path) -> dict[str, Any]:
    result_path = output / "V41_RESULT.json"
    result = read_json(result_path)
    if result["status"] != "INCOMPLETE_V41":
        raise AssertionError("V41 result is no longer incomplete")
    rows = [_task_audit(output, instance_id) for instance_id in FAILED]
    value = {
        "completed_at_utc": utc_now(),
        "status": "V41_INFRA_FAILURE_PAIRED_SOURCE_CAPACITY_DEADLOCK",
        "classification": "INCOMPLETE_ITT_DO_NOT_SCORE_FAILED_TASKS",
        "failed_tasks": rows,
        "causal_chain": [
            "The paired runner materialized both V40 and General source spans "
            "in one shared device KV pool.",
            "SGLang admission counted the complete target prompt plus the "
            "frozen maximum output against the remaining device slots.",
            "For both failed requests, a conservative lower bound on target "
            "admission exceeded the remaining slots before target staging.",
            "Sources are released only after their target requests finish, so "
            "waiting cannot increase capacity and the client eventually hits "
            "the frozen 180-second streaming timeout.",
        ],
        "arm_specific_failure_rejected": (
            "The first blocked target is V40 on Astropy and General on "
            "Requests-1142. The failure follows target size versus shared "
            "device capacity, not one policy label."
        ),
        "fix_scope": {
            "accuracy_protocol": (
                "Store paired-arm source snapshots in host memory so the two "
                "experimental arms do not artificially reserve one shared "
                "device KV pool. Host loading is exact KV reuse, not prefetch."
            ),
            "speed_protocol": (
                "Do not use paired host residency for formal speed claims. "
                "Run each arm independently with its native device-resident "
                "source lifecycle and counterbalanced order."
            ),
            "v41_mutated_or_rescored": False,
        },
        "inputs": {
            "v41_result_sha256": sha256(result_path),
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
    }
    write_json(output / "V41_CAPACITY_DEADLOCK_AUDIT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = run(args.output)
    print(
        {
            "status": value["status"],
            "failed_tasks": value["failed_tasks"],
        }
    )


if __name__ == "__main__":
    main()
