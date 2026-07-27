#!/usr/bin/env python3
"""Complete V26C after one audited mid-treatment HTTP stream failure."""

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
from benchmark.multi_workflow.run_v26_paired_replication_campaign import (
    ARMS,
    GENERAL,
    SAMPLE_SIZE,
    V23,
    _bootstrap_difference,
    _run_stage,
    _wilson,
    task_dir,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
SOURCE = ARTIFACTS / "impactkv_v26c_paired_replication_20260727"
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v26d_paired_replication_completion_20260727"
)
RETRY_INSTANCE = "pytest-dev__pytest-7432"
FIXED_INSTANCES = (
    "sphinx-doc__sphinx-9230",
    "sphinx-doc__sphinx-7440",
    "astropy__astropy-14995",
    "pydata__xarray-4075",
)
FAILED_LEDGER = (
    SOURCE / "tasks" / RETRY_INSTANCE / "run" / "SERVER_LEDGER.jsonl"
)
FAILED_SERVER_LOG = (
    SOURCE / "tasks" / RETRY_INSTANCE / "run" / "sglang_server.log"
)


def register(output: Path) -> dict[str, Any]:
    path = output / "V26D_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    source_result = read_json(SOURCE / "V26_RESULT.json")
    if source_result["status"] != "INCOMPLETE":
        raise ValueError("V26C is not the expected incomplete campaign")
    output.mkdir(parents=True)
    fixed = {}
    for instance_id in FIXED_INSTANCES:
        root = SOURCE / "tasks" / instance_id
        fixed[instance_id] = {
            "runtime_path": str(root / "V25_RESULT.json"),
            "runtime_sha256": sha256(root / "V25_RESULT.json"),
            "official_path": str(root / "V25_OFFICIAL_RESULT.json"),
            "official_sha256": sha256(
                root / "V25_OFFICIAL_RESULT.json"
            ),
        }
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_V26D_RETRY_GPU_RUN",
        "experiment": "V26D completion of frozen V26C five-task ITT set",
        "reason": (
            "The first four V26C tasks completed officially.  The fifth "
            "failed after one V23 target copy when the HTTP stream remained "
            "open after SGLang logged HTTP 200; General had not run.  There "
            "is no paired task outcome to select.  Retry that task exactly "
            "once with a 180-second request timeout, then combine it with the "
            "four immutable completed outcomes."
        ),
        "source_campaign": {
            "registration_path": str(SOURCE / "V26_REGISTRATION.json"),
            "registration_sha256": sha256(
                SOURCE / "V26_REGISTRATION.json"
            ),
            "incomplete_result_path": str(SOURCE / "V26_RESULT.json"),
            "incomplete_result_sha256": sha256(
                SOURCE / "V26_RESULT.json"
            ),
            "fixed_results": fixed,
        },
        "failed_attempt": {
            "instance_id": RETRY_INSTANCE,
            "classification": "MID_TREATMENT_HTTP_STREAM_INFRASTRUCTURE_FAILURE",
            "paired_outcome_observed": False,
            "v23_completed_requests": 1,
            "general_completed_requests": 0,
            "server_ledger_path": str(FAILED_LEDGER),
            "server_ledger_sha256": sha256(FAILED_LEDGER),
            "server_log_path": str(FAILED_SERVER_LOG),
            "server_log_sha256": sha256(FAILED_SERVER_LOG),
        },
        "retry_protocol": {
            "instance_id": RETRY_INSTANCE,
            "attempts": 1,
            "request_timeout_seconds": 180,
            "all_other_model_engine_token_and_agent_settings_unchanged": True,
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "frozen_gates": read_json(SOURCE / "V26_REGISTRATION.json")[
            "frozen_development_gates"
        ],
        "protected": {
            "prefetch": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "old_dirty_checkout_modified": False,
        },
        "source_sha256": sha256(Path(__file__)),
    }
    write_json(path, value)
    return value


def preregister_retry(output: Path) -> dict[str, Any]:
    register(output)
    child = task_dir(output, RETRY_INSTANCE) / "V25_REGISTRATION.json"
    if child.exists():
        return {"returncode": 0, "resumed": True}
    value = _run_stage(output, RETRY_INSTANCE, "register")
    if value["returncode"] != 0:
        raise RuntimeError("V26D child registration failed")
    return value


def _result_row(instance_id: str, root: Path) -> dict[str, Any]:
    runtime = read_json(root / "V25_RESULT.json")
    official = read_json(root / "V25_OFFICIAL_RESULT.json")
    resolved = {
        arm: int(official["arms"][arm]["resolved"]) for arm in ARMS
    }
    return {
        "instance_id": instance_id,
        "artifact_root": str(root),
        "resolved": resolved,
        "paired_difference": resolved[V23] - resolved[GENERAL],
        "v23_only": resolved[V23] == 1 and resolved[GENERAL] == 0,
        "general_only": resolved[V23] == 0 and resolved[GENERAL] == 1,
        "both_resolved": all(resolved.values()),
        "both_failed": not any(resolved.values()),
        "branch": runtime["branch"],
        "copy_counts": runtime["server"]["copy_counts"],
        "target_fallbacks": runtime["server"]["target_fallbacks"],
        "branched_agent_elapsed_seconds": runtime.get(
            "branched_agent_elapsed_seconds"
        ),
        "official_metrics": official["arms"],
    }


def summarize(output: Path) -> dict[str, Any]:
    register(output)
    rows = [
        _result_row(instance_id, SOURCE / "tasks" / instance_id)
        for instance_id in FIXED_INSTANCES
    ]
    retry_root = task_dir(output, RETRY_INSTANCE)
    retry_complete = (
        (retry_root / "V25_RESULT.json").exists()
        and (retry_root / "V25_OFFICIAL_RESULT.json").exists()
    )
    if retry_complete:
        rows.append(_result_row(RETRY_INSTANCE, retry_root))
    resolved = {
        arm: sum(row["resolved"][arm] for row in rows) for arm in ARMS
    }
    differences = [row["paired_difference"] for row in rows]
    v23_only = sum(row["v23_only"] for row in rows)
    general_only = sum(row["general_only"] for row in rows)
    fallbacks = sum(row["target_fallbacks"] for row in rows)
    complete = len(rows)
    gates = {
        "official_tasks_completed": complete == SAMPLE_SIZE,
        "runner_infrastructure_failures": retry_complete,
        "target_fallbacks": fallbacks == 0,
        "v23_resolved_not_below_general": resolved[V23]
        >= resolved[GENERAL],
        "v23_only_not_below_general_only": v23_only >= general_only,
        "v23_only_min": v23_only >= 1,
        "report_paired_accuracy_difference_bootstrap95": bool(rows),
        "do_not_promote_to_full225_if_gate_fails": True,
    }
    value = {
        "summarized_at_utc": utc_now(),
        "status": (
            "PASS_DEVELOPMENT_REPLICATION"
            if all(gates.values())
            else "INCOMPLETE"
            if not retry_complete
            else "FAIL_DEVELOPMENT_REPLICATION"
        ),
        "tasks": rows,
        "aggregate": {
            "tasks_complete": complete,
            "resolved": resolved,
            "accuracy": {
                arm: resolved[arm] / complete if complete else None
                for arm in ARMS
            },
            "accuracy_wilson95": {
                arm: _wilson(resolved[arm], complete) for arm in ARMS
            },
            "paired_accuracy_difference_v23_minus_general": (
                sum(differences) / complete if complete else None
            ),
            "paired_difference_bootstrap95": _bootstrap_difference(
                differences
            ),
            "v23_only": v23_only,
            "general_only": general_only,
            "both_resolved": sum(row["both_resolved"] for row in rows),
            "both_failed": sum(row["both_failed"] for row in rows),
            "target_fallbacks": fallbacks,
        },
        "gate_outcomes": gates,
        "decision": (
            "Eligible for a separately pre-registered full225 development "
            "screen; not yet a baseline or Verified promotion claim."
            if all(gates.values())
            else "Do not expand to full225; audit failures and revise."
        ),
    }
    write_json(output / "V26D_RESULT.json", value)
    return value


def run(output: Path) -> dict[str, Any]:
    preregister_retry(output)
    child = task_dir(output, RETRY_INSTANCE)
    if not (child / "V25_RESULT.json").exists():
        runtime = _run_stage(output, RETRY_INSTANCE, "run")
        if runtime["returncode"] != 0:
            return summarize(output)
    if not (child / "V25_OFFICIAL_RESULT.json").exists():
        _run_stage(output, RETRY_INSTANCE, "evaluate")
    return summarize(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("register", "preregister", "run", "summarize"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "register":
        value = register(args.output)
    elif args.command == "preregister":
        value = preregister_retry(args.output)
    elif args.command == "run":
        value = run(args.output)
    else:
        value = summarize(args.output)
    print(
        json.dumps(
            {
                "status": value.get("status"),
                "output": str(args.output),
                "gate_outcomes": value.get("gate_outcomes"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
