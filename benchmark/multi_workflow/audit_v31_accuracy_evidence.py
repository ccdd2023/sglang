#!/usr/bin/env python3
"""Conservatively audit valid V31 paired-agent accuracy evidence."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = (
    ARTIFACTS
    / "impactkv_v31_accuracy_evidence_audit_20260727"
    / "V31_ACCURACY_EVIDENCE_AUDIT.json"
)
V31 = "coding_critical_event_abstain_v31"
GENERAL = "general"
DENSE = "dense"
ARMS = (V31, GENERAL, DENSE)
CACHEBLEND_DAMAGE_RATE = 9 / 167
MAIN_RUNS = (
    (
        "scikit-learn__scikit-learn-12585",
        ARTIFACTS
        / "impactkv_v31b_paired_agent_canary_sklearn12585_20260727",
        "outcome-independent V31 replay selection",
    ),
    (
        "astropy__astropy-7336",
        ARTIFACTS
        / "impactkv_v31f_paired_accuracy_20260727/tasks/"
        "astropy__astropy-7336",
        "outcome-exposed Dense-preservation diagnostic",
    ),
    (
        "django__django-14855",
        ARTIFACTS
        / "impactkv_v31f_paired_accuracy_20260727/tasks/"
        "django__django-14855",
        "outcome-exposed Dense-preservation diagnostic",
    ),
    (
        "pytest-dev__pytest-7982",
        ARTIFACTS
        / "impactkv_v31f_paired_accuracy_20260727/tasks/"
        "pytest-dev__pytest-7982",
        "outcome-exposed Dense-preservation diagnostic",
    ),
    (
        "pylint-dev__pylint-7277",
        ARTIFACTS
        / "impactkv_v31f_paired_accuracy_20260727/tasks/"
        "pylint-dev__pylint-7277",
        "outcome-exposed General-only challenge diagnostic",
    ),
)
SUPPORTING_REPEAT = (
    "astropy__astropy-7336",
    ARTIFACTS
    / "impactkv_v31e_single_use_canary_astropy7336_20260727",
)
INCOMPLETE = (
    "scikit-learn__scikit-learn-13779",
    ARTIFACTS
    / "impactkv_v31g_terminal_outcome_sklearn13779_20260727",
)


def _task_row(
    instance_id: str,
    root: Path,
    selection_class: str,
) -> dict[str, Any]:
    runtime_path = root / "V25_RESULT.json"
    official_path = root / "V25_OFFICIAL_RESULT.json"
    runtime = read_json(runtime_path)
    official = read_json(official_path)
    if official["instance_id"] != instance_id:
        raise ValueError(f"{instance_id}: official identity mismatch")
    resolved = {
        arm: int(official["arms"][arm]["resolved"]) for arm in ARMS
    }
    fallback_count = int(runtime["server"]["target_fallbacks"])
    if fallback_count:
        raise ValueError(f"{instance_id}: invalid target fallback")
    first_hashes = runtime["first_branch_prompt_hashes"]
    first_hash_equal = not first_hashes or len(set(first_hashes.values())) == 1
    if not first_hash_equal:
        raise ValueError(f"{instance_id}: first branch prompt mismatch")
    exit_status = runtime["exit_status"]
    if any(
        status not in {"Submitted", "LimitsExceeded"}
        for status in exit_status.values()
    ):
        raise ValueError(f"{instance_id}: non-terminal arm")
    return {
        "instance_id": instance_id,
        "root": str(root),
        "selection_class": selection_class,
        "resolved": resolved,
        "empty_patch": {
            arm: int(official["arms"][arm]["empty_patch"]) for arm in ARMS
        },
        "exit_status": exit_status,
        "branch": runtime["branch"],
        "copy_counts": runtime["server"]["copy_counts"],
        "critical_abstentions": runtime["server"][
            "candidate_critical_abstentions"
        ],
        "target_fallbacks": fallback_count,
        "median_ttft_ms": {
            arm: official["arms"][arm]["median_ttft_ms"] for arm in ARMS
        },
        "runtime_registration_sha256": sha256(
            root / "V25_REGISTRATION.json"
        ),
        "runtime_result_sha256": sha256(runtime_path),
        "official_result_sha256": sha256(official_path),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = {
        arm: sum(row["resolved"][arm] for row in rows) for arm in ARMS
    }
    dense_passes = sum(row["resolved"][DENSE] for row in rows)
    dense_fails = len(rows) - dense_passes
    damage = {
        arm: sum(
            row["resolved"][DENSE] == 1 and row["resolved"][arm] == 0
            for row in rows
        )
        for arm in (V31, GENERAL)
    }
    rescue = {
        arm: sum(
            row["resolved"][DENSE] == 0 and row["resolved"][arm] == 1
            for row in rows
        )
        for arm in (V31, GENERAL)
    }
    v31_only = sum(
        row["resolved"][V31] == 1 and row["resolved"][GENERAL] == 0
        for row in rows
    )
    general_only = sum(
        row["resolved"][V31] == 0 and row["resolved"][GENERAL] == 1
        for row in rows
    )
    ratios = []
    for row in rows:
        left = row["median_ttft_ms"][V31]
        right = row["median_ttft_ms"][GENERAL]
        if left is not None and right not in (None, 0):
            ratios.append(float(left) / float(right))
    return {
        "task_runs": len(rows),
        "resolved": resolved,
        "accuracy": {arm: resolved[arm] / len(rows) for arm in ARMS},
        "v31_minus_general_pp": (
            100 * (resolved[V31] - resolved[GENERAL]) / len(rows)
        ),
        "v31_minus_dense_pp": (
            100 * (resolved[V31] - resolved[DENSE]) / len(rows)
        ),
        "paired_v31_only": v31_only,
        "paired_general_only": general_only,
        "dense_passes": dense_passes,
        "dense_fails": dense_fails,
        "damage_count_given_dense_pass": damage,
        "damage_rate_given_dense_pass": {
            arm: damage[arm] / dense_passes if dense_passes else None
            for arm in damage
        },
        "rescue_count_given_dense_fail": rescue,
        "rescue_rate_given_dense_fail": {
            arm: rescue[arm] / dense_fails if dense_fails else None
            for arm in rescue
        },
        "cacheblend_damage_rate_reference": CACHEBLEND_DAMAGE_RATE,
        "paired_ttft_ratio_v31_over_general": {
            "comparable_task_runs": len(ratios),
            "median": statistics.median(ratios) if ratios else None,
            "values": ratios,
        },
    }


def audit(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    rows = [
        _task_row(instance_id, root, selection_class)
        for instance_id, root, selection_class in MAIN_RUNS
    ]
    repeat = _task_row(
        SUPPORTING_REPEAT[0],
        SUPPORTING_REPEAT[1],
        "supporting repeated task; excluded from main denominator",
    )
    incomplete_root = INCOMPLETE[1]
    incomplete = {
        "instance_id": INCOMPLETE[0],
        "root": str(incomplete_root),
        "classification": (
            "INCOMPLETE_INFRASTRUCTURE_MIDSTREAM_TIMEOUT; excluded from all "
            "accuracy, damage, rescue, and speed denominators"
        ),
        "registration_sha256": sha256(
            incomplete_root / "V25_REGISTRATION.json"
        ),
        "server_ledger_sha256": sha256(
            incomplete_root / "run/SERVER_LEDGER.jsonl"
        ),
        "result_exists": (incomplete_root / "V25_RESULT.json").exists(),
    }
    value = {
        "audited_at_utc": utc_now(),
        "status": "RETROSPECTIVE_DEVELOPMENT_SIGNAL_NOT_PROMOTIONAL",
        "classification": (
            "Conservative unique-task audit. Outcome-exposed tasks prevent "
            "held-out or SOTA claims; repeated Astropy evidence is separate."
        ),
        "main_rows": rows,
        "main_aggregate": _aggregate(rows),
        "supporting_repeat": repeat,
        "incomplete_excluded": incomplete,
        "interpretation": {
            "task_correctness": (
                "V31 has a positive paired development signal versus both "
                "General and Dense on the five unique valid task runs."
            ),
            "dense_preservation": (
                "No V31 damage is observed among concurrent Dense passes, "
                "but the denominator is too small and outcome-exposed."
            ),
            "speed": (
                "Fixed candidate-first order and path-dependent request counts "
                "make these TTFT ratios diagnostic only. Use counterbalanced "
                "frozen replay for the speed gate."
            ),
            "next": (
                "Freeze an outcome-independent critical-event-covered task "
                "sample, replicate V31/General/Dense with the repaired runner, "
                "then compare native KVCOMM and CacheBlend on the same task "
                "identities before any promotion claim."
            ),
        },
    }
    write_json(output, value)
    return value


if __name__ == "__main__":
    result = audit()
    print(
        {
            "status": result["status"],
            "output": str(DEFAULT_OUTPUT),
            "main_aggregate": result["main_aggregate"],
        }
    )
