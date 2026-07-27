#!/usr/bin/env python3
"""Preregister the V32 replication after repairing sync-stream cleanup."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import (
    prepare_v32_outcome_independent_sample as v32,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
ORIGINAL = ARTIFACTS / "impactkv_v32_outcome_independent_sample_20260727"
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v32r_stream_close_replication_20260727"
)
PROJECT = Path(__file__).resolve().parents[2]
BRIDGE = (
    PROJECT / "benchmark/multi_workflow/bridge_reuse_litellm_model.py"
)
RUNNER = PROJECT / "benchmark/multi_workflow/run_v25_paired_agent_canary.py"
V31 = "coding_critical_event_abstain_v31"


def record_original_failure() -> dict[str, Any]:
    path = ORIGINAL / "V32_INFRASTRUCTURE_FAILURE.json"
    if path.exists():
        return read_json(path)
    task = ORIGINAL / "tasks/sympy__sympy-13551"
    if (task / "V25_RESULT.json").exists():
        raise RuntimeError("Original V32 unexpectedly has a completed result")
    server_log = task / "run/sglang_server.log"
    value = {
        "recorded_at_utc": utc_now(),
        "status": "INVALID_INFRASTRUCTURE_STREAM_CLOSE_WAIT",
        "instance_id": "sympy__sympy-13551",
        "accuracy_denominator": False,
        "speed_denominator": False,
        "replacement_task_allowed": False,
        "official_outcome_observed": False,
        "model_submission_observed": False,
        "failure": {
            "symptom": (
                "The server completed a streamed response, but the client "
                "retained multiple CLOSE_WAIT sockets and stopped consuming "
                "the terminal stream state."
            ),
            "diagnosis": (
                "BridgeReuseLitellmModel did not explicitly close LiteLLM's "
                "synchronous completion_stream after normal exhaustion."
            ),
            "manual_intervention": (
                "SIGINT was used after more than five minutes without a new "
                "prefill; it released the stuck stream and therefore makes "
                "the entire run non-promotional."
            ),
        },
        "evidence": {
            "original_registration": str(
                ORIGINAL / "V32_REGISTRATION.json"
            ),
            "original_registration_sha256": sha256(
                ORIGINAL / "V32_REGISTRATION.json"
            ),
            "server_log": str(server_log),
            "server_log_sha256": sha256(server_log),
            "result_file_absent": True,
        },
        "disposition": (
            "Exclude the interrupted run, preserve its task identity, and "
            "replicate the already-frozen sample only after a separately "
            "tested stream-lifecycle repair."
        ),
    }
    write_json(path, value)
    return value


def _paths(output: Path) -> dict[str, Path]:
    return {
        "snapshot": output / "V32R_FROZEN_SUBSET.json",
        "evaluation_registration": (
            output / "V32R_EVAL_REGISTRATION.json"
        ),
        "dataset": output / "minisweagent_dataset",
    }


def _environment(output: Path, instance_id: str) -> dict[str, str]:
    env = v32._runner_environment(output, instance_id)
    paths = _paths(output)
    env.update(
        {
            "IMPACTKV_DATASET_ROOT": str(paths["dataset"]),
            "IMPACTKV_EVAL_REGISTRATION": str(
                paths["evaluation_registration"]
            ),
            "IMPACTKV_EVAL_SNAPSHOT": str(paths["snapshot"]),
        }
    )
    return env


def task_dir(output: Path, instance_id: str) -> Path:
    return output / "tasks" / instance_id


def _runner_command(
    output: Path,
    instance_id: str,
    stage: str,
) -> list[str]:
    return [
        str(v32.PYTHON),
        str(RUNNER),
        stage,
        "--output",
        str(task_dir(output, instance_id)),
    ]


def register(output: Path) -> dict[str, Any]:
    path = output / "V32R_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    original_failure = record_original_failure()
    output.mkdir(parents=True)
    selected = v32._selection()
    expected = (
        "sympy__sympy-13551",
        "astropy__astropy-7671",
        "pylint-dev__pylint-4661",
    )
    if tuple(row["instance_id"] for row in selected) != expected:
        raise AssertionError("V32R must preserve the original V32 selection")

    paths = _paths(output)
    write_json(paths["snapshot"], selected)
    snapshot_sha = sha256(paths["snapshot"])
    evaluation_registration = {
        "schema_version": 1,
        "registration_id": (
            "impactkv_v32r_stream_close_replication_20260727"
        ),
        "registered_at_utc": utc_now(),
        "dataset": {
            "name": "princeton-nlp/SWE-bench_Verified",
            "split": "test",
            "population_size": 500,
            "local_snapshot": str(paths["snapshot"]),
            "local_snapshot_sha256": snapshot_sha,
        },
        "instances": [
            {"instance_id": row["instance_id"]} for row in selected
        ],
    }
    write_json(paths["evaluation_registration"], evaluation_registration)
    paths["dataset"].mkdir(parents=True)
    data_path = paths["dataset"] / "test.jsonl"
    data_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in selected
        ),
        encoding="utf-8",
    )
    write_json(
        paths["dataset"] / "DATASET_MANIFEST.json",
        {
            "registration_id": evaluation_registration["registration_id"],
            "source_snapshot": str(paths["snapshot"]),
            "source_snapshot_sha256": snapshot_sha,
            "instance_ids": list(expected),
            "data_file": str(data_path),
        },
    )

    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_ANY_V32R_AGENT_TREATMENT",
        "experiment": "V32R stream-close repaired paired-agent replication",
        "relation_to_v32": {
            "original_registration": str(
                ORIGINAL / "V32_REGISTRATION.json"
            ),
            "original_registration_sha256": sha256(
                ORIGINAL / "V32_REGISTRATION.json"
            ),
            "original_failure_record": str(
                ORIGINAL / "V32_INFRASTRUCTURE_FAILURE.json"
            ),
            "original_failure_record_sha256": sha256(
                ORIGINAL / "V32_INFRASTRUCTURE_FAILURE.json"
            ),
            "same_frozen_task_identities": True,
            "replacement_or_reselection": False,
            "official_task_outcomes_used": False,
            "infrastructure_symptom_used": True,
            "original_failure_status": original_failure["status"],
        },
        "motivation": (
            "Repair a connection-lifecycle defect before testing V31. The "
            "task identities remain exactly those frozen before V32 began; "
            "no official V31, General, or Dense outcome was observed."
        ),
        "repair": {
            "mechanism": (
                "Close and release LiteLLM completion_stream in a finally "
                "block after every synchronous streaming request."
            ),
            "bridge_sha256": sha256(BRIDGE),
            "tests": {
                "normal_stream_close": True,
                "exception_stream_close": True,
                "policy_and_paired_suite_passed": "37/37",
            },
        },
        "selection": {
            "rule": (
                "Reuse the exact V32 SHA-selected identities; do not replace "
                "the interrupted SymPy task."
            ),
            "selected": [
                {
                    "instance_id": row["instance_id"],
                    "difficulty": row["difficulty"],
                }
                for row in selected
            ],
        },
        "protocol": {
            "arms": [V31, "general", "dense"],
            "all_children_registered_before_first_treatment": True,
            "temperature": 0,
            "step_limit": 20,
            "request_timeout_seconds": 180,
            "official_swebench_container_each_arm": True,
            "empty_or_step_limit_exit_scored_unresolved": True,
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "frozen_gates": {
            "children_completed": len(selected),
            "mechanical_runtime_passes": len(selected),
            "target_fallbacks": 0,
            "v31_resolved_not_below_general": True,
            "v31_damage_not_above_general": True,
            "report_overall_accuracy_damage_rescue_and_ttft_separately": True,
            "do_not_replace_failed_or_incomplete_tasks": True,
        },
        "inputs": {
            "campaign_sha256": sha256(Path(__file__)),
            "runner_sha256": sha256(RUNNER),
            "dataset_sha256": sha256(data_path),
            "evaluation_snapshot_sha256": snapshot_sha,
            "evaluation_registration_sha256": sha256(
                paths["evaluation_registration"]
            ),
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
    }
    write_json(path, value)
    return value


def preregister_children(output: Path) -> list[dict[str, Any]]:
    registration = register(output)
    results = []
    for selected in registration["selection"]["selected"]:
        instance_id = selected["instance_id"]
        log_path = output / "logs" / instance_id / "register.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                _runner_command(output, instance_id, "register"),
                cwd=PROJECT,
                env=_environment(output, instance_id),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        results.append(
            {
                "instance_id": instance_id,
                "returncode": process.returncode,
                "log_path": str(log_path),
            }
        )
    write_json(output / "V32R_CHILD_REGISTRATIONS.json", results)
    if any(row["returncode"] for row in results):
        raise RuntimeError("V32R child preregistration failed")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("register", "preregister"),
        nargs="?",
        default="preregister",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "register":
        value = register(args.output)
        status = value["status"]
    else:
        preregister_children(args.output)
        status = "CHILDREN_REGISTERED_BEFORE_TREATMENT"
    print({"output": str(args.output), "status": status})


if __name__ == "__main__":
    main()
