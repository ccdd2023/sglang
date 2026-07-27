#!/usr/bin/env python3
"""Retrospectively test whether V30 fidelity signals separate task damage.

This audit is intentionally non-promotional: the CacheBlend damage labels and
all V30 measurements already exist.  Its only purpose is to prevent a fidelity
metric from being silently reused as an accuracy oracle when the two cohorts
are not cleanly separable.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    sha256_file,
    write_json,
)
from benchmark.multi_workflow.run_coding_native_workload_v10 import read_json


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v30_kv_component_replay_20260727"
MEASUREMENTS = DEFAULT_OUTPUT / "V30_MEASUREMENTS.json"
RESULT = DEFAULT_OUTPUT / "V30_RESULT.json"
DIAGNOSTIC_AUC_FLOOR = 0.75


def _auc_high_means_damage(values: list[tuple[float, bool]]) -> float:
    positive = [value for value, label in values if label]
    negative = [value for value, label in values if not label]
    if not positive or not negative:
        raise ValueError("both damage and safe cases are required")
    wins = sum(
        float(left > right) + 0.5 * float(left == right)
        for left in positive
        for right in negative
    )
    return wins / (len(positive) * len(negative))


def _features(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for row in rows:
        case = cases.setdefault(
            str(row["case_id"]),
            {"damage": row["cohort"] == "damage"},
        )
        case[str(row["variant"])] = row
    result = {}
    for case_id, case in cases.items():
        full = float(case["full_copy"]["kl_mean"])

        def reduction(variant: str) -> float:
            value = float(case[variant]["kl_mean"])
            return (full - value) / max(full, 1e-12)

        result[case_id] = {
            "damage": bool(case["damage"]),
            "full_copy_kl": full,
            "k_repair_reduction": reduction("target_k_source_v"),
            "v_repair_reduction": reduction("source_k_target_v"),
            "early12_reduction": reduction("repair_early12"),
            "middle12_reduction": reduction("repair_middle12"),
            "late12_reduction": reduction("repair_late12"),
            "k_vs_v_kl": (
                float(case["source_k_target_v"]["kl_mean"])
                - float(case["target_k_source_v"]["kl_mean"])
            ),
        }
    return result


def audit(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = output / "V30_DAMAGE_SEPARABILITY_AUDIT.json"
    if destination.exists():
        return read_json(destination)
    rows = read_json(MEASUREMENTS)["rows"]
    cases = _features(rows)
    feature_names = [
        name for name in next(iter(cases.values())) if name != "damage"
    ]
    summaries = {}
    for feature in feature_names:
        values = [
            (float(case[feature]), bool(case["damage"]))
            for case in cases.values()
        ]
        auc = _auc_high_means_damage(values)
        summaries[feature] = {
            "damage_mean": statistics.mean(
                value for value, label in values if label
            ),
            "matched_safe_mean": statistics.mean(
                value for value, label in values if not label
            ),
            "auc_high_means_damage": auc,
            "best_oriented_auc_posthoc": max(auc, 1.0 - auc),
        }
    best_feature = max(
        summaries,
        key=lambda name: summaries[name]["best_oriented_auc_posthoc"],
    )
    best_auc = summaries[best_feature]["best_oriented_auc_posthoc"]
    value = {
        "classification": "RETROSPECTIVE_DIAGNOSTIC_ONLY_NOT_PROMOTIONAL",
        "cases": {
            "damage": sum(case["damage"] for case in cases.values()),
            "matched_safe": sum(
                not case["damage"] for case in cases.values()
            ),
        },
        "decision": (
            "Do not use V30 KL/component signals as an accuracy-risk oracle; "
            "develop online coding-event abstention and validate task damage "
            "directly."
        ),
        "diagnostic_auc_floor": DIAGNOSTIC_AUC_FLOOR,
        "feature_summaries": summaries,
        "highest_posthoc_oriented_auc": best_auc,
        "highest_posthoc_oriented_feature": best_feature,
        "inputs": {
            "measurements_sha256": sha256_file(MEASUREMENTS),
            "result_sha256": sha256_file(RESULT),
            "source_sha256": sha256_file(Path(__file__)),
        },
        "outcome": (
            "NO_CLEAN_DAMAGE_SEPARATION"
            if best_auc < DIAGNOSTIC_AUC_FLOOR
            else "POSTHOC_SIGNAL_REQUIRES_NEW_HOLDOUT"
        ),
        "per_case": cases,
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
    }
    write_json(destination, value)
    return value


def main() -> None:
    print(json.dumps(audit(), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
