#!/usr/bin/env python3
"""Preregister the positive-mean cost-gate counterfactual without inference.

The completed/active lower-confidence-bound arm already records target-time
cost estimates before generation.  This preparation freezes cases where the
dependency/version guard passed and the same calibrated regression predicted
a positive mean saving, but the residual-Q10 lower bound vetoed reuse.  The
counterfactual changes only that admission inequality.
"""

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
TARGET = ARTIFACTS / "impactkv_common_agent_graph_mean_20260812"
SOURCE_ARM = "coding_dependency_graph_cold_lcb"
ARM = "coding_dependency_graph_cold_mean"
CANARY_IDS = (
    "django__django-15957",
    "pydata__xarray-6744",
    "django__django-12325",
    "sympy__sympy-18763",
)


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    registration_path = TARGET / "GRAPH_MEAN_PREREGISTRATION.json"
    if registration_path.is_file():
        print(registration_path)
        return
    if TARGET.exists() and any(TARGET.iterdir()):
        raise FileExistsError(f"nonempty unregistered campaign: {TARGET}")

    formal_snapshot_path = SOURCE / "FROZEN_FRESH24.json"
    formal_registration_path = SOURCE / "BRIDGE_FRESH24_REGISTRATION.json"
    ledger_path = (
        SOURCE
        / "runs/sglang_formal"
        / SOURCE_ARM
        / "full_24/CLIENT_LEDGER.jsonl"
    )
    required = (
        formal_snapshot_path,
        formal_registration_path,
        SOURCE / "formal_dataset/test.jsonl",
        ledger_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")

    motivating_targets: list[dict[str, Any]] = []
    for row in load_jsonl(ledger_path):
        decision = row.get("reuse_policy_decision") or {}
        for guard in decision.get("target_evidence_guards") or ():
            cost = guard.get("cost_estimate") or {}
            predicted = cost.get("predicted_cache_ready_saving_ms")
            lower = cost.get("lower_bound_cache_ready_saving_ms")
            if (
                guard.get("target_evidence_valid") is True
                and predicted is not None
                and float(predicted) > 0
                and lower is not None
                and float(lower) <= 0
            ):
                motivating_targets.append(
                    {
                        "model_instance_nonce": row.get("model_instance_nonce"),
                        "request_index": row.get("request_index"),
                        "source_id": guard.get("source_id"),
                        "source_paths": guard.get("source_paths"),
                        "island_tokens": cost.get("island_tokens"),
                        "target_prompt_tokens": cost.get("target_prompt_tokens"),
                        "predicted_cache_ready_saving_ms": predicted,
                        "lower_bound_cache_ready_saving_ms": lower,
                        "observed_before_official_outcome": True,
                    }
                )
    if not motivating_targets:
        raise RuntimeError(
            "no version-valid dependency-cold target was vetoed only by Q10"
        )

    formal = read_json(formal_snapshot_path)
    by_id = {str(row["instance_id"]): row for row in formal}
    absent = [value for value in CANARY_IDS if value not in by_id]
    if absent:
        raise ValueError(f"canary tasks absent from Fresh24: {absent}")
    canary = [by_id[value] for value in CANARY_IDS]

    TARGET.mkdir(parents=True)
    write_json(TARGET / "FROZEN_FRESH24.json", formal)
    write_json(TARGET / "CANARY4.json", canary)
    source_registration = read_json(formal_registration_path)
    formal_registration = {
        **source_registration,
        "registration_id": "impactkv-common-agent-graph-mean-fresh24-20260812",
    }
    canary_registration = {
        **source_registration,
        "registration_id": "impactkv-common-agent-graph-mean-canary4-20260812",
        "instances": [{"instance_id": value} for value in CANARY_IDS],
    }
    write_json(TARGET / "BRIDGE_FRESH24_REGISTRATION.json", formal_registration)
    write_json(TARGET / "BRIDGE_CANARY4_REGISTRATION.json", canary_registration)

    (TARGET / "formal_dataset").mkdir()
    shutil.copy2(
        SOURCE / "formal_dataset/test.jsonl",
        TARGET / "formal_dataset/test.jsonl",
    )
    dataset_rows = load_jsonl(SOURCE / "formal_dataset/test.jsonl")
    dataset_by_id = {str(row["instance_id"]): row for row in dataset_rows}
    (TARGET / "canary_dataset").mkdir()
    with (TARGET / "canary_dataset/test.jsonl").open("w", encoding="utf-8") as stream:
        for instance_id in CANARY_IDS:
            stream.write(json.dumps(dataset_by_id[instance_id], sort_keys=True) + "\n")

    write_json(
        registration_path,
        {
            "schema_version": 1,
            "status": "FROZEN_BEFORE_GRAPH_MEAN_MODEL_REQUESTS",
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_arm": SOURCE_ARM,
            "treatment_arm": ARM,
            "motivation": {
                "question": (
                    "Does residual-Q10 conservatism suppress physically useful "
                    "coding-aware targets even after version and dependency guards pass?"
                ),
                "online_pre_outcome_targets": motivating_targets,
            },
            "single_variable_intervention": (
                "Keep natural single-file extraction, version validation, "
                "dependency-graph-hot recomputation, one-island limit, frozen "
                "regression coefficients, no ordinary prefix and no prefetch; "
                "change admission from predicted+residual_q10>0 to predicted>0."
            ),
            "canary_ids": list(CANARY_IDS),
            "formal_ids": [str(row["instance_id"]) for row in formal],
            "gates": {
                "canary_physical_target_copy_events_min": 1,
                "target_fallback_events_max": 0,
                "formal_official_instances": 24,
                "exact_prompt_input_identity": True,
                "report_cache_ready_and_n1_n4_n16": True,
                "keep": (
                    "physical copy passes and median cache-ready speedup >1; "
                    "official accuracy must have no unexplained degradation"
                ),
            },
            "inputs": {
                "source_client_ledger": str(ledger_path),
                "source_client_ledger_sha256": sha256(ledger_path),
                "fresh24_snapshot_sha256": sha256(formal_snapshot_path),
                "formal_dataset_sha256": sha256(
                    SOURCE / "formal_dataset/test.jsonl"
                ),
            },
            "protected": {
                "prefetch": False,
                "ordinary_radix_prefix_reuse": False,
                "old_dirty_checkout_modified": False,
                "paper_modified": False,
                "old_preregistration_thresholds_modified": False,
            },
        },
    )
    write_json(
        TARGET / "AUTOMATED_GRAPH_MEAN_STATUS.json",
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
