#!/usr/bin/env python3
"""Complete V44 summary after one non-copy client event lacked copy fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import (
    run_v44_dense_sensitive_v40_campaign as campaign,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    sha256,
    utc_now,
    write_json,
)


DEFAULT_OUTPUT = campaign.DEFAULT_OUTPUT
EXPECTED_RELATIVE_LEDGER = Path(
    "tasks/sphinx-doc__sphinx-11445/"
    "coding_grounded_observation_island_v40/CLIENT_LEDGER.jsonl"
)
EXPECTED_EVENT = "pending_source_not_reusable"
EXPECTED_MISSING_ROWS = 1
EXPECTED_NULL_TTFT_TASK = "astropy__astropy-7671"
EXPECTED_NULL_TTFT_ARMS = 3


def _read_client(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        if "copied_tokens_planned" in row:
            continue
        if row.get("event") != EXPECTED_EVENT:
            raise AssertionError(
                f"unexpected copy-field omission in {path}: {row}"
            )
        # This is a source-lifecycle diagnostic, not a target request.
        row["copied_tokens_planned"] = 0
    return rows


def run(output: Path) -> dict[str, Any]:
    stage_path = output / "V44_STAGE_STATUS.json"
    registration_path = output / "V44_REGISTRATION.json"
    affected_path = output / EXPECTED_RELATIVE_LEDGER
    stages = json.loads(stage_path.read_text(encoding="utf-8"))
    if len(stages) != 24 or any(row["returncode"] != 0 for row in stages):
        raise AssertionError("V44 run/evaluate stages are not all complete")
    official = list((output / "tasks").glob("*/V25_OFFICIAL_RESULT.json"))
    if len(official) != len(campaign.TASKS):
        raise AssertionError("V44 official results are incomplete")
    raw_rows = [
        json.loads(line)
        for line in affected_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing = [
        row for row in raw_rows if "copied_tokens_planned" not in row
    ]
    if len(missing) != EXPECTED_MISSING_ROWS:
        raise AssertionError("V44 missing-field count changed")
    if any(row.get("event") != EXPECTED_EVENT for row in missing):
        raise AssertionError("V44 missing-field event changed")
    null_ttft = [
        (path.parent.name, arm)
        for path in (output / "tasks").glob("*/V25_OFFICIAL_RESULT.json")
        for arm, row in json.loads(
            path.read_text(encoding="utf-8")
        )["arms"].items()
        if row.get("median_ttft_ms") is None
    ]
    if (
        len(null_ttft) != EXPECTED_NULL_TTFT_ARMS
        or set(null_ttft)
        != {
            (EXPECTED_NULL_TTFT_TASK, arm) for arm in campaign.ARMS
        }
    ):
        raise AssertionError("V44 null-TTFT evidence changed")

    original_client = campaign.prior._client
    original_median = campaign.statistics.median

    def median_non_null(values: Any) -> float | None:
        present = [value for value in values if value is not None]
        return original_median(present) if present else None

    campaign.prior._client = _read_client
    campaign.statistics.median = median_non_null
    try:
        value = campaign.summarize(output)
    finally:
        campaign.prior._client = original_client
        campaign.statistics.median = original_median

    repair = {
        "completed_at_utc": utc_now(),
        "status": "V44_POST_TREATMENT_SUMMARY_SCHEMA_COMPAT_APPLIED",
        "failure": {
            "exception": "KeyError: copied_tokens_planned",
            "scope": "summary only",
            "affected_relative_ledger": str(EXPECTED_RELATIVE_LEDGER),
            "affected_event": EXPECTED_EVENT,
            "missing_rows": EXPECTED_MISSING_ROWS,
            "null_ttft_task": EXPECTED_NULL_TTFT_TASK,
            "null_ttft_arms": EXPECTED_NULL_TTFT_ARMS,
        },
        "repair": {
            "rule": (
                "Assign copied_tokens_planned=0 only while reading the "
                "pending_source_not_reusable diagnostic row. That event is "
                "not a target request and cannot contain a completed copy. "
                "Exclude null per-arm TTFT values from the diagnostic median; "
                "they arise only when the task completes in shared Dense "
                "history before branching."
            ),
            "raw_ledger_modified": False,
            "task_rerun": False,
            "official_evaluation_rerun": False,
            "registered_tasks_or_gates_modified": False,
        },
        "inputs": {
            "registration_sha256": sha256(registration_path),
            "stage_status_sha256": sha256(stage_path),
            "affected_ledger_sha256": sha256(affected_path),
            "repair_script_sha256": sha256(Path(__file__)),
        },
    }
    value["post_treatment_summary_repair"] = repair
    write_json(output / "V44_RESULT.json", value)
    write_json(output / "V44_SUMMARY_SCHEMA_REPAIR.json", repair)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = run(args.output)
    print(
        {
            "status": value["status"],
            "aggregate": value["aggregate"],
        }
    )


if __name__ == "__main__":
    main()
