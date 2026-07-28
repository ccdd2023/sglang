#!/usr/bin/env python3
"""Freeze the incomplete V40A2 timeout without promoting partial outcomes."""

from __future__ import annotations

import argparse
import json
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
    ARTIFACTS / "impactkv_v40a2_grounded_observation_canary_20260728"
)
V40 = "coding_grounded_observation_island_v40"
GENERAL = "general"
DENSE = "dense"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run(output: Path) -> dict[str, Any]:
    registration_path = output / "V40A2_REGISTRATION.json"
    status_path = output / "orchestration_status/run.json"
    log_path = output / "orchestration_logs/run.log"
    server_path = output / "task/run/SERVER_LEDGER.jsonl"
    shadow_path = output / "task/run/SHADOW_LEDGER.jsonl"
    registration = read_json(registration_path)
    status = read_json(status_path)
    log = log_path.read_text(encoding="utf-8")
    server = _jsonl(server_path)
    shadow = _jsonl(shadow_path)
    clients = {
        arm: _jsonl(output / "task" / arm / "CLIENT_LEDGER.jsonl")
        for arm in (V40, GENERAL, DENSE)
    }
    copies = [
        row for row in server if row.get("event") == "target_copied"
    ]
    fallbacks = [
        row for row in server if row.get("event") == "target_fallback"
    ]
    branch_rows = [
        row for row in shadow if row.get("branch_kind") is not None
    ]
    if registration["status"] != "REGISTERED_BEFORE_V40A2_TREATMENT":
        raise AssertionError("V40A2 registration changed")
    if status["returncode"] != 1:
        raise AssertionError("V40A2 is no longer a failed run")
    if "httpcore.ReadTimeout: timed out" not in log:
        raise AssertionError("V40A2 timeout evidence missing")
    if len(branch_rows) != 1 or branch_rows[0]["call"] != 7:
        raise AssertionError("unexpected V40A2 branch")
    copy_counts = {
        arm: sum(
            row.get("event") == "target_copied"
            and row.get("policy_label") == arm
            for row in server
        )
        for arm in (V40, GENERAL, DENSE)
    }
    copied_tokens = {
        arm: sum(
            int(row["copied_k_tokens"])
            for row in copies
            if row.get("policy_label") == arm
        )
        for arm in (V40, GENERAL, DENSE)
    }
    host_copy_counts = {
        arm: sum(
            row.get("policy_label") == arm
            and row.get("source_residency") == "host"
            for row in copies
        )
        for arm in (V40, GENERAL, DENSE)
    }
    if copy_counts != {V40: 13, GENERAL: 12, DENSE: 0}:
        raise AssertionError("unexpected completed V40A2 copies")
    value = {
        "completed_at_utc": utc_now(),
        "status": "V40A2_INFRA_FAILURE_NO_ACCURACY_RESULT",
        "classification": "INCOMPLETE_ITT_DO_NOT_SCORE",
        "mechanism": {
            "branch_call": 7,
            "branch_kind": branch_rows[0]["branch_kind"],
            "copy_requests": copy_counts,
            "copied_tokens": copied_tokens,
            "host_copy_requests": host_copy_counts,
            "target_fallbacks": len(fallbacks),
            "completed_client_requests": {
                arm: len(rows) for arm, rows in clients.items()
            },
            "v40_q20_completed": len(clients[V40]) == 13,
            "general_q20_timed_out": len(clients[GENERAL]) == 12,
            "dense_q20_not_dispatched": len(clients[DENSE]) == 12,
        },
        "accuracy": {
            "official_evaluation_run": False,
            "result": None,
            "reason": (
                "The General q20 streaming request timed out before all arms "
                "reached a terminal agent outcome."
            ),
        },
        "conclusion": (
            "V40's grounded observation island executed 13 real target copies "
            "and selected fewer tokens than General, with zero fallback. "
            "However, General q20 hit the frozen 180-second mid-stream timeout "
            "before Dense q20 was dispatched. The canary is incomplete and "
            "cannot support an accuracy, speed, reliability, or superiority "
            "claim. Host-residency differences are retrospective diagnostics "
            "only."
        ),
        "next": (
            "Register a distinct short-trajectory mechanism canary selected "
            "only from V40 motivation calls/source coverage; do not rerun or "
            "replace V40A2."
        ),
        "inputs": {
            "registration_sha256": sha256(registration_path),
            "run_status_sha256": sha256(status_path),
            "run_log_sha256": sha256(log_path),
            "server_ledger_sha256": sha256(server_path),
            "shadow_ledger_sha256": sha256(shadow_path),
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
    }
    write_json(output / "V40A2_INFRA_FAILURE.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = run(args.output)
    print({"status": value["status"], "mechanism": value["mechanism"]})


if __name__ == "__main__":
    main()
