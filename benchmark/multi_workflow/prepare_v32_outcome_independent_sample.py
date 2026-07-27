#!/usr/bin/env python3
"""Freeze and preregister an unseen three-task V32 Verified sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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
    ARTIFACTS / "impactkv_v32_outcome_independent_sample_20260727"
)
PROJECT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT / "benchmark/multi_workflow/run_v25_paired_agent_canary.py"
PYTHON = Path("/home/gfy/.venvs/mini-swe-agent-v2.3.0/bin/python")
SOURCE_SNAPSHOTS = (
    ARTIFACTS / "swebench_verified_medium_v1_20260724/frozen_subset.json",
    ARTIFACTS / "swebench_verified_complex_v1_20260724/frozen_subset.json",
)
SALT = "v32-outcome-independent-agent-v1\n"
SAMPLE_SIZE = 3
EXCLUDED = {"matplotlib__matplotlib-25775"}
V31 = "coding_critical_event_abstain_v31"


def _selection() -> list[dict[str, Any]]:
    rows = [
        row
        for snapshot in SOURCE_SNAPSHOTS
        for row in read_json(snapshot)
        if row["instance_id"] not in EXCLUDED
    ]
    selected = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            (SALT + row["instance_id"]).encode()
        ).hexdigest(),
    )[:SAMPLE_SIZE]
    return selected


def _paths(output: Path) -> dict[str, Path]:
    return {
        "snapshot": output / "V32_FROZEN_SUBSET.json",
        "registration": output / "V32_EVAL_REGISTRATION.json",
        "dataset": output / "minisweagent_dataset",
    }


def _runner_environment(output: Path, instance_id: str) -> dict[str, str]:
    paths = _paths(output)
    env = os.environ.copy()
    for key in (
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "all_proxy",
        "https_proxy",
        "http_proxy",
    ):
        env.pop(key, None)
    env.update(
        {
            "NO_PROXY": "*",
            "no_proxy": "*",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": f"{PROJECT}:{PROJECT / 'python'}",
            "IMPACTKV_PAIRED_CANDIDATE_ARM": V31,
            "IMPACTKV_PAIRED_DENSE_CONTROL": "1",
            "IMPACTKV_ALLOW_EMPTY_SUBMISSION_OUTCOME": "1",
            "IMPACTKV_PAIRED_INSTANCE_ID": instance_id,
            "IMPACTKV_REQUEST_TIMEOUT_SECONDS": "180",
            "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "1",
            "IMPACTKV_DATASET_ROOT": str(paths["dataset"]),
            "IMPACTKV_EVAL_REGISTRATION": str(paths["registration"]),
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
        str(PYTHON),
        str(RUNNER),
        stage,
        "--output",
        str(task_dir(output, instance_id)),
    ]


def register(output: Path) -> dict[str, Any]:
    path = output / "V32_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    selected = _selection()
    expected = (
        "sympy__sympy-13551",
        "astropy__astropy-7671",
        "pylint-dev__pylint-4661",
    )
    if tuple(row["instance_id"] for row in selected) != expected:
        raise AssertionError("V32 SHA selection changed")
    paths = _paths(output)
    write_json(paths["snapshot"], selected)
    snapshot_sha = sha256(paths["snapshot"])
    evaluation_registration = {
        "schema_version": 1,
        "registration_id": "impactkv_v32_outcome_independent_20260727",
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
    write_json(paths["registration"], evaluation_registration)
    paths["dataset"].mkdir(parents=True)
    data_path = paths["dataset"] / "test.jsonl"
    data_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    write_json(
        paths["dataset"] / "DATASET_MANIFEST.json",
        {
            "registration_id": evaluation_registration["registration_id"],
            "source_snapshot": str(paths["snapshot"]),
            "source_snapshot_sha256": snapshot_sha,
            "instance_ids": [row["instance_id"] for row in selected],
            "data_file": str(data_path),
        },
    )
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_ANY_V32_AGENT_TREATMENT",
        "experiment": "V32 unseen Verified paired-agent replication",
        "motivation": (
            "V31 resolved 5/5 unique valid development tasks versus 4/5 "
            "for General and Dense, but most tasks were outcome-exposed. "
            "Replicate on unseen medium/complex task identities."
        ),
        "selection": {
            "source_snapshots": [str(path) for path in SOURCE_SNAPSHOTS],
            "source_snapshot_sha256": {
                str(path): sha256(path) for path in SOURCE_SNAPSHOTS
            },
            "salt": SALT,
            "rule": (
                "Pool medium+complex Verified tasks, exclude only the known "
                "unavailable matplotlib container, sort SHA-256(salt || "
                "instance_id), take first three."
            ),
            "excluded": sorted(EXCLUDED),
            "selected": [
                {
                    "instance_id": row["instance_id"],
                    "difficulty": row["difficulty"],
                    "selection_sha256": hashlib.sha256(
                        (SALT + row["instance_id"]).encode()
                    ).hexdigest(),
                }
                for row in selected
            ],
            "v31_general_dense_agent_outcomes_used": False,
        },
        "protocol": {
            "arms": [V31, "general", "dense"],
            "all_children_registered_before_first_treatment": True,
            "shared_dense_history_and_container_snapshot": True,
            "temperature": 0,
            "step_limit": 20,
            "request_timeout_seconds": 180,
            "official_swebench_container_each_arm": True,
            "empty_or_step_limit_exit_scored_unresolved": True,
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "frozen_gates": {
            "children_completed": SAMPLE_SIZE,
            "mechanical_runtime_passes": SAMPLE_SIZE,
            "target_fallbacks": 0,
            "v31_resolved_not_below_general": True,
            "v31_damage_not_above_general": True,
            "report_all_infrastructure_failures": True,
            "do_not_replace_failed_or_incomplete_tasks": True,
        },
        "inputs": {
            "runner_sha256": sha256(RUNNER),
            "dataset_sha256": sha256(data_path),
            "evaluation_snapshot_sha256": snapshot_sha,
            "evaluation_registration_sha256": sha256(paths["registration"]),
            "campaign_sha256": sha256(Path(__file__)),
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
                env=_runner_environment(output, instance_id),
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
    write_json(output / "V32_CHILD_REGISTRATIONS.json", results)
    if any(row["returncode"] for row in results):
        raise RuntimeError("V32 child preregistration failed")
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
    else:
        value = {"children": preregister_children(args.output)}
    print({"output": str(args.output), "status": value.get("status")})


if __name__ == "__main__":
    main()
