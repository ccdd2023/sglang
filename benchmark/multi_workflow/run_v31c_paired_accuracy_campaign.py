#!/usr/bin/env python3
"""Run a five-task diagnostic V31/General/Dense paired agent campaign.

The task cohorts deliberately use already exposed outcomes.  This campaign is
therefore a development diagnostic for Dense preservation and known
General-only challenges, not held-out promotion evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_bridge_reuse_agent_experiment import DATASET
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v31c_paired_accuracy_20260727"
PROJECT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT / "benchmark/multi_workflow/run_v25_paired_agent_canary.py"
PYTHON = Path("/home/gfy/.venvs/mini-swe-agent-v2.3.0/bin/python")
V31 = "coding_critical_event_abstain_v31"
GENERAL = "general"
DENSE = "dense"
ARMS = (V31, GENERAL, DENSE)
BOOTSTRAP_SEED = 20260727
BOOTSTRAPS = 100_000
CACHEBLEND_DAMAGE_RATE = 9 / 167
TASKS = (
    {
        "instance_id": "astropy__astropy-7336",
        "cohort": "dense_preservation",
        "historical_basis": "concurrent Dense resolved in V27D",
        "offline_critical_events": 4,
    },
    {
        "instance_id": "django__django-14855",
        "cohort": "dense_preservation",
        "historical_basis": "concurrent Dense resolved in V27D",
        "offline_critical_events": 1,
    },
    {
        "instance_id": "pytest-dev__pytest-7982",
        "cohort": "dense_preservation",
        "historical_basis": "concurrent Dense resolved in V27D",
        "offline_critical_events": 2,
    },
    {
        "instance_id": "pylint-dev__pylint-7277",
        "cohort": "general_only_challenge",
        "historical_basis": "General-only resolution in V23 full18",
        "offline_critical_events": 5,
    },
    {
        "instance_id": "scikit-learn__scikit-learn-13779",
        "cohort": "general_only_challenge",
        "historical_basis": "General-only resolution in V23 full18",
        "offline_critical_events": 6,
    },
)


def task_dir(output: Path, instance_id: str) -> Path:
    return output / "tasks" / instance_id


def _command(output: Path, instance_id: str, stage: str) -> list[str]:
    return [
        str(PYTHON),
        str(RUNNER),
        stage,
        "--output",
        str(task_dir(output, instance_id)),
    ]


def _environment(instance_id: str) -> dict[str, str]:
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
        }
    )
    return env


def register(output: Path) -> dict[str, Any]:
    path = output / "V31C_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    dataset_ids = {
        json.loads(line)["instance_id"]
        for line in (DATASET / "test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    if any(row["instance_id"] not in dataset_ids for row in TASKS):
        raise ValueError("V31C task missing from frozen dataset")
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_ANY_V31C_TREATMENT_RUN",
        "experiment": "V31C paired coding-event accuracy diagnostic",
        "motivation": (
            "V31 improved same-prompt fidelity and its first independent "
            "agent canary preserved a Dense/General success. Test whether "
            "critical-event abstention preserves historically Dense-resolved "
            "tasks and can recover historically General-only challenges."
        ),
        "selection": {
            "tasks": list(TASKS),
            "outcomes_used": True,
            "classification": (
                "retrospective development diagnostic; forbidden as held-out "
                "promotion evidence"
            ),
        },
        "protocol": {
            "arms": list(ARMS),
            "shared_dense_history_before_branch": True,
            "container_snapshot_before_branch": True,
            "target_order": list(ARMS),
            "step_limit": 20,
            "temperature": 0,
            "same_model_engine_tokenization": True,
            "official_swebench_container_each_arm": True,
            "empty_patch_scored_as_official_unresolved_outcome": True,
            "limits_exceeded_scored_as_official_unresolved_outcome": True,
            "all_children_registered_before_first_treatment": True,
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_iterations": BOOTSTRAPS,
        },
        "frozen_development_gates": {
            "official_tasks_completed": len(TASKS),
            "runtime_passes": len(TASKS),
            "target_fallbacks": 0,
            "v31_resolved_not_below_general": True,
            "v31_resolved_not_below_dense": True,
            "v31_damage_not_above_general": True,
            "v31_damage_rate_below_cacheblend": CACHEBLEND_DAMAGE_RATE,
            "challenge_v31_resolved_not_below_general": True,
            "do_not_promote_from_diagnostic_set_alone": True,
        },
        "inputs": {
            "dataset": str(DATASET / "test.jsonl"),
            "dataset_sha256": sha256(DATASET / "test.jsonl"),
            "runner_sha256": sha256(RUNNER),
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


def _run_stage(
    output: Path,
    instance_id: str,
    stage: str,
) -> dict[str, Any]:
    log_path = (
        output / "orchestration_logs" / instance_id / f"{stage}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            _command(output, instance_id, stage),
            cwd=PROJECT,
            env=_environment(instance_id),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    value = {
        "instance_id": instance_id,
        "stage": stage,
        "returncode": process.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "log_path": str(log_path),
    }
    write_json(
        output / "orchestration_status" / instance_id / f"{stage}.json",
        value,
    )
    return value


def preregister_children(output: Path) -> list[dict[str, Any]]:
    register(output)
    rows = []
    for task in TASKS:
        instance_id = task["instance_id"]
        child = task_dir(output, instance_id) / "V25_REGISTRATION.json"
        rows.append(
            {
                "instance_id": instance_id,
                "stage": "register",
                "returncode": 0,
                "resumed": True,
            }
            if child.exists()
            else _run_stage(output, instance_id, "register")
        )
    write_json(output / "V31C_CHILD_REGISTRATIONS.json", rows)
    if any(row["returncode"] != 0 for row in rows):
        raise RuntimeError("one or more V31C child registrations failed")
    return rows


def _wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            p * (1 - p) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _bootstrap(values: list[int]) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    samples = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(BOOTSTRAPS)
    )
    return [
        samples[int(0.025 * BOOTSTRAPS)],
        samples[min(BOOTSTRAPS - 1, int(0.975 * BOOTSTRAPS))],
    ]


def _median_non_null(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.median(present) if present else None


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    rows = []
    for task in TASKS:
        instance_id = task["instance_id"]
        child = task_dir(output, instance_id)
        runtime_path = child / "V25_RESULT.json"
        official_path = child / "V25_OFFICIAL_RESULT.json"
        if not runtime_path.exists() or not official_path.exists():
            rows.append({**task, "status": "INCOMPLETE"})
            continue
        runtime = read_json(runtime_path)
        official = read_json(official_path)
        resolved = {
            arm: int(official["arms"][arm]["resolved"]) for arm in ARMS
        }
        rows.append(
            {
                **task,
                "status": "COMPLETE",
                "runtime_status": runtime["status"],
                "resolved": resolved,
                "v31_minus_general": resolved[V31] - resolved[GENERAL],
                "v31_damage": resolved[DENSE] == 1 and resolved[V31] == 0,
                "general_damage": (
                    resolved[DENSE] == 1 and resolved[GENERAL] == 0
                ),
                "v31_rescue": resolved[DENSE] == 0 and resolved[V31] == 1,
                "general_rescue": (
                    resolved[DENSE] == 0 and resolved[GENERAL] == 1
                ),
                "target_fallbacks": runtime["server"]["target_fallbacks"],
                "copy_counts": runtime["server"]["copy_counts"],
                "critical_abstentions": runtime["server"][
                    "candidate_critical_abstentions"
                ],
                "median_ttft_ms": {
                    arm: official["arms"][arm]["median_ttft_ms"]
                    for arm in ARMS
                },
                "branched_agent_elapsed_seconds": runtime[
                    "branched_agent_elapsed_seconds"
                ],
            }
        )
    complete = [row for row in rows if row["status"] == "COMPLETE"]
    resolved = {
        arm: sum(row["resolved"][arm] for row in complete) for arm in ARMS
    }
    dense_passes = sum(row["resolved"][DENSE] for row in complete)
    damage = {
        V31: sum(row["v31_damage"] for row in complete),
        GENERAL: sum(row["general_damage"] for row in complete),
    }
    damage_rate = {
        arm: damage[arm] / dense_passes if dense_passes else None
        for arm in (V31, GENERAL)
    }
    challenge = [
        row
        for row in complete
        if row["cohort"] == "general_only_challenge"
    ]
    fallbacks = sum(row["target_fallbacks"] for row in complete)
    gates = {
        "official_tasks_completed": len(complete) == len(TASKS),
        "runtime_passes": all(
            row["runtime_status"] == "PASS" for row in complete
        )
        and len(complete) == len(TASKS),
        "target_fallbacks": fallbacks == 0,
        "v31_resolved_not_below_general": resolved[V31] >= resolved[GENERAL],
        "v31_resolved_not_below_dense": resolved[V31] >= resolved[DENSE],
        "v31_damage_not_above_general": damage[V31] <= damage[GENERAL],
        "v31_damage_rate_below_cacheblend": (
            damage_rate[V31] is not None
            and damage_rate[V31] < CACHEBLEND_DAMAGE_RATE
        ),
        "challenge_v31_resolved_not_below_general": sum(
            row["resolved"][V31] for row in challenge
        )
        >= sum(row["resolved"][GENERAL] for row in challenge),
        "do_not_promote_from_diagnostic_set_alone": True,
    }
    value = {
        "completed_at_utc": utc_now(),
        "status": (
            "PASS_V31C_DIAGNOSTIC"
            if all(gates.values())
            else "INCOMPLETE_V31C"
            if len(complete) < len(TASKS)
            else "FAIL_V31C_DIAGNOSTIC"
        ),
        "registration_sha256": sha256(
            output / "V31C_REGISTRATION.json"
        ),
        "tasks": rows,
        "aggregate": {
            "complete_tasks": len(complete),
            "resolved": resolved,
            "accuracy": {
                arm: resolved[arm] / len(complete) if complete else None
                for arm in ARMS
            },
            "accuracy_wilson95": {
                arm: _wilson(resolved[arm], len(complete)) for arm in ARMS
            },
            "v31_minus_general_bootstrap95": _bootstrap(
                [row["v31_minus_general"] for row in complete]
            ),
            "dense_passes": dense_passes,
            "damage_count_given_dense_pass": damage,
            "damage_rate_given_dense_pass": damage_rate,
            "cacheblend_damage_rate_reference": CACHEBLEND_DAMAGE_RATE,
            "rescue_count_given_dense_fail": {
                V31: sum(row["v31_rescue"] for row in complete),
                GENERAL: sum(row["general_rescue"] for row in complete),
            },
            "challenge_resolved": {
                arm: sum(row["resolved"][arm] for row in challenge)
                for arm in ARMS
            },
            "task_median_ttft_ms": {
                arm: _median_non_null(
                    [row["median_ttft_ms"][arm] for row in complete]
                )
                for arm in ARMS
            },
            "target_fallbacks": fallbacks,
        },
        "gate_outcomes": gates,
        "interpretation": (
            "Development diagnostic only because task outcomes selected both "
            "cohorts. A pass permits an outcome-independent replication; it "
            "does not establish superiority over KVCOMM or CacheBlend."
        ),
        "registered_gates": registration["frozen_development_gates"],
    }
    write_json(output / "V31C_RESULT.json", value)
    return value


def run(output: Path) -> dict[str, Any]:
    register(output)
    preregister_children(output)
    stages = []
    for task in TASKS:
        instance_id = task["instance_id"]
        child = task_dir(output, instance_id)
        if not (child / "V25_RESULT.json").exists():
            stages.append(_run_stage(output, instance_id, "run"))
        if not (child / "V25_RESULT.json").exists():
            break
        if read_json(child / "V25_RESULT.json")["status"] != "PASS":
            break
        if not (child / "V25_OFFICIAL_RESULT.json").exists():
            stages.append(_run_stage(output, instance_id, "evaluate"))
    write_json(output / "V31C_STAGE_STATUS.json", stages)
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
        value = {"children": preregister_children(args.output)}
    elif args.command == "run":
        value = run(args.output)
    else:
        value = summarize(args.output)
    print(
        {
            "status": value.get("status"),
            "output": str(args.output),
            "gate_outcomes": value.get("gate_outcomes"),
        }
    )


if __name__ == "__main__":
    main()
