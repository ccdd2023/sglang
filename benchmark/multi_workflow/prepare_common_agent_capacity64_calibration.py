#!/usr/bin/env python3
"""Register the minimal 32-to-64-call Dense competence counterfactual.

This is deliberately a diagnostic cohort, not a final method-ranking cohort.
It selects every Fresh24 task whose common Dense trajectory exhausted the
pre-registered 32-call budget.  The selection therefore tests whether agent
capacity caused the observed accuracy floor without selecting on a reuse
method's wins or losses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
PATHS = RuntimePaths.from_project(PROJECT)
SOURCE = PATHS.artifacts / "impactkv_common_agent_baselines_fresh24_20260812"
DEFAULT_OUTPUT = PATHS.artifacts / "impactkv_common_agent_capacity64_20260812"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_report(run: Path) -> dict[str, Any]:
    wrapper = read_json(run / "OFFICIAL_RESULT.json")
    report = wrapper.get("report") or {}
    if wrapper.get("returncode") != 0 or report.get("error_instances") != 0:
        raise RuntimeError("source Dense official evaluation is not valid")
    return report


def prepare(output: Path, source: Path = SOURCE) -> dict[str, Any]:
    registration_path = output / "CAPACITY64_REGISTRATION.json"
    if registration_path.exists():
        return read_json(registration_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)

    dense_run = source / "runs/formal/cacheblend_dense/all"
    report = official_report(dense_run)
    if report.get("resolved_instances") != 0 or report.get("total_instances") != 24:
        raise RuntimeError("capacity diagnosis requires the observed valid 0/24 Dense floor")

    trajectories: list[dict[str, Any]] = []
    for path in sorted(dense_run.glob("*/*.traj.json")):
        value = read_json(path)
        info = value.get("info") or {}
        api_calls = int((info.get("model_stats") or {}).get("api_calls") or 0)
        trajectories.append(
            {
                "instance_id": str(value.get("instance_id") or path.parent.name),
                "exit_status": info.get("exit_status"),
                "api_calls": api_calls,
                "submission_empty": not bool(str(info.get("submission") or "").strip()),
                "trajectory_sha256": sha256(path),
            }
        )
    if len(trajectories) != 24:
        raise RuntimeError(f"expected 24 source trajectories, found {len(trajectories)}")
    selected = [
        row
        for row in trajectories
        if row["exit_status"] == "LimitsExceeded" and row["api_calls"] == 32
    ]
    if len(selected) != 12:
        raise RuntimeError(f"expected 12 call-limited trajectories, found {len(selected)}")

    source_rows = [
        json.loads(line)
        for line in (source / "formal_dataset/test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(row["instance_id"]): row for row in source_rows}
    selected_ids = [row["instance_id"] for row in selected]
    if set(selected_ids).difference(by_id):
        raise RuntimeError("selected trajectory is absent from the frozen Fresh24 dataset")
    dataset_rows = [by_id[instance_id] for instance_id in selected_ids]

    output.mkdir(parents=True)
    write_json(output / "FROZEN_CAPACITY12.json", dataset_rows)
    write_jsonl(output / "capacity12_dataset/test.jsonl", dataset_rows)
    bridge_registration = {
        "schema_version": 1,
        "registration_id": "impactkv-common-agent-capacity64-diagnostic-20260812",
        "dataset": {"name": "princeton-nlp/SWE-bench_Verified", "split": "test"},
        "instances": [{"instance_id": value} for value in selected_ids],
    }
    write_json(output / "BRIDGE_CAPACITY12_REGISTRATION.json", bridge_registration)

    value = {
        "schema_version": 1,
        "status": "REGISTERED_BEFORE_CAPACITY64_MODEL_REQUESTS",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "motivation": {
            "source_protocol": "common Qwen2.5-Coder-7B rolling-6 Dense, 32 calls",
            "valid_official_accuracy": "0/24",
            "call_limit_exhausted": "12/24",
            "early_submitted": "12/24",
            "falsifiable_hypothesis": (
                "If the 32-call ceiling materially caused the floor, raising only the "
                "common call budget to 64 will resolve at least one of the 12 tasks "
                "that exhausted 32 calls."
            ),
        },
        "intervention": {
            "arm": "Dense only",
            "changed_field": "agent.step_limit",
            "old": 32,
            "new": 64,
            "unchanged": [
                "model and BF16 runtime",
                "rolling-6 messages and tool prompt",
                "temperature and token limits",
                "official SWE-bench evaluator",
                "task repositories and initial problem statements",
            ],
        },
        "selection": {
            "classification": "post-run mechanism diagnostic; not a final ranking cohort",
            "rule": "all and only Fresh24 Dense trajectories with LimitsExceeded at exactly 32 API calls",
            "selected": selected,
            "selected_count": len(selected),
            "reuse_outcomes_used": False,
            "claim_limit": (
                "A pass only establishes that the original accuracy protocol was "
                "capacity-limited. It does not establish a method accuracy advantage."
            ),
        },
        "gate": {
            "official_total_instances": 12,
            "official_error_instances": 0,
            "official_resolved_instances_min": 1,
            "on_fail": "drop call-budget expansion as the sole floor remedy",
            "on_pass": (
                "run all methods on the original outcome-blind Fresh24 with the same "
                "64-call budget before making accuracy rankings"
            ),
        },
        "inputs": {
            "source_campaign": str(source),
            "source_dense_official_sha256": sha256(dense_run / "OFFICIAL_RESULT.json"),
            "source_dataset_sha256": sha256(source / "formal_dataset/test.jsonl"),
            "capacity12_sha256": sha256(output / "FROZEN_CAPACITY12.json"),
        },
        "protected": {
            "prefetch": False,
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
        },
    }
    write_json(registration_path, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = prepare(args.output, args.source)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
