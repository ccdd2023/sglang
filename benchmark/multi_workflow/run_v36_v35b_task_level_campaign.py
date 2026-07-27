#!/usr/bin/env python3
"""Run a preregistered task-level V35B/General/Dense paired campaign."""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v36_v35b_task_level_campaign_20260727"
)
PROJECT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT / "benchmark/multi_workflow/run_v25_paired_agent_canary.py"
PYTHON = Path("/home/gfy/.venvs/mini-swe-agent-v2.3.0/bin/python")
MOTIVATION = (
    ARTIFACTS
    / "impactkv_v35b_version_validation_motivation_20260727"
    / "V35B_MOTIVATION_RESULT.json"
)
V35B = "coding_version_validation_target_v35b"
GENERAL = "general"
DENSE = "dense"
ARMS = (V35B, GENERAL, DENSE)
SALT = "v36-v35b-task-level-itt-v1\n"
SAMPLE_SIZE = 6
EXCLUDED = {
    "pydata__xarray-4075": "V35C mechanism canary",
    "pylint-dev__pylint-7277": "repeated outcome-tuned positive control",
    "scikit-learn__scikit-learn-13779": "prior incomplete timeout",
}
EXPECTED = (
    "psf__requests-5414",
    "astropy__astropy-14995",
    "astropy__astropy-7336",
    "psf__requests-1142",
    "scikit-learn__scikit-learn-12585",
    "django__django-16899",
)
BOOTSTRAP_SEED = 20260727
BOOTSTRAPS = 100_000
CACHEBLEND_DAMAGE_RATE = 9 / 167


def _selection() -> list[dict[str, Any]]:
    motivation = read_json(MOTIVATION)
    candidates = [
        row
        for row in motivation["cohorts"]["full18"]
        if row["reached"] and row["instance_id"] not in EXCLUDED
    ]
    selected = sorted(
        candidates,
        key=lambda row: hashlib.sha256(
            (SALT + row["instance_id"]).encode()
        ).hexdigest(),
    )[:SAMPLE_SIZE]
    if tuple(row["instance_id"] for row in selected) != EXPECTED:
        raise AssertionError("V36 SHA selection changed")
    return selected


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
            "IMPACTKV_PAIRED_CANDIDATE_ARM": V35B,
            "IMPACTKV_PAIRED_DENSE_CONTROL": "1",
            "IMPACTKV_ALLOW_EMPTY_SUBMISSION_OUTCOME": "1",
            "IMPACTKV_PAIRED_INSTANCE_ID": instance_id,
            "IMPACTKV_REQUEST_TIMEOUT_SECONDS": "180",
            "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "1",
        }
    )
    env.pop("IMPACTKV_REQUIRE_BRANCH", None)
    return env


def register(output: Path) -> dict[str, Any]:
    path = output / "V36_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    selected = _selection()
    dataset_path = DATASET / "test.jsonl"
    dataset_ids = {
        json.loads(line)["instance_id"]
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if any(row["instance_id"] not in dataset_ids for row in selected):
        raise ValueError("V36 task missing from frozen dataset")
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_ANY_V36_TREATMENT_RUN",
        "experiment": "V36 task-level V35B paired development campaign",
        "motivation": (
            "V35B reduced target vetoes to 13.88% on frozen traces and "
            "preserved one Dense-pass task in a causal canary, but exact "
            "offline request indices did not reproduce. Evaluate task-level "
            "intention-to-treat outcomes without requiring a particular "
            "branch step."
        ),
        "selection": {
            "source": str(MOTIVATION),
            "source_sha256": sha256(MOTIVATION),
            "rule": (
                "Take full18 tasks with frozen V35B task-level reach, exclude "
                "the V35C canary, the repeated outcome-tuned Pylint control, "
                "and the declared timeout; sort SHA-256(salt || instance_id), "
                "take six."
            ),
            "salt": SALT,
            "sample_size": SAMPLE_SIZE,
            "excluded": EXCLUDED,
            "tasks": [
                {
                    **row,
                    "selection_sha256": hashlib.sha256(
                        (SALT + row["instance_id"]).encode()
                    ).hexdigest(),
                }
                for row in selected
            ],
            "official_outcomes_used_for_selection": False,
            "outcome_exposure_class": (
                "DEVELOPMENT_POOL_PREVIOUSLY_EVALUATED_NOT_HELD_OUT"
            ),
        },
        "protocol": {
            "arms": list(ARMS),
            "shared_dense_history_before_branch": True,
            "container_snapshot_before_branch": True,
            "target_order": list(ARMS),
            "task_level_intention_to_treat": True,
            "no_branch_inherits_shared_dense_outcome": True,
            "exact_branch_index_gate": False,
            "step_limit": 20,
            "temperature": 0,
            "same_model_engine_tokenization": True,
            "official_swebench_container_each_arm": True,
            "empty_or_step_limit_scored_unresolved": True,
            "all_children_registered_before_first_treatment": True,
            "do_not_replace_failed_or_incomplete_tasks": True,
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_iterations": BOOTSTRAPS,
        },
        "frozen_development_gates": {
            "official_tasks_completed": SAMPLE_SIZE,
            "runtime_mechanics_passes": SAMPLE_SIZE,
            "tasks_with_online_branch_min": 3,
            "target_fallbacks": 0,
            "v35b_resolved_strictly_above_general": True,
            "v35b_resolved_strictly_above_dense": True,
            "v35b_damage_not_above_general": True,
            "v35b_damage_rate_below_cacheblend": CACHEBLEND_DAMAGE_RATE,
            "v35b_rescue_not_below_general": True,
            "v35b_only_vs_general_min": 1,
            "report_overall_accuracy_damage_rescue_speed_separately": True,
            "do_not_make_population_or_sota_claim": True,
        },
        "inputs": {
            "dataset": str(dataset_path),
            "dataset_sha256": sha256(dataset_path),
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
    registration = register(output)
    rows = []
    for task in registration["selection"]["tasks"]:
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
    write_json(output / "V36_CHILD_REGISTRATIONS.json", rows)
    if any(row["returncode"] != 0 for row in rows):
        raise RuntimeError("one or more V36 child registrations failed")
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


def _median(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.median(present) if present else None


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    rows = []
    for task in registration["selection"]["tasks"]:
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
                "branch_reached": runtime["branch"] is not None,
                "branch_request_index": (
                    runtime["branch"].get("branch_request_index")
                    if runtime["branch"]
                    else None
                ),
                "resolved": resolved,
                "candidate_minus_general": (
                    resolved[V35B] - resolved[GENERAL]
                ),
                "candidate_minus_dense": resolved[V35B] - resolved[DENSE],
                "candidate_damage": (
                    resolved[DENSE] == 1 and resolved[V35B] == 0
                ),
                "general_damage": (
                    resolved[DENSE] == 1 and resolved[GENERAL] == 0
                ),
                "candidate_rescue": (
                    resolved[DENSE] == 0 and resolved[V35B] == 1
                ),
                "general_rescue": (
                    resolved[DENSE] == 0 and resolved[GENERAL] == 1
                ),
                "target_fallbacks": runtime["server"]["target_fallbacks"],
                "candidate_target_vetoes": runtime["server"][
                    "candidate_target_vetoes"
                ],
                "copy_counts": runtime["server"]["copy_counts"],
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
    dense_fails = len(complete) - dense_passes
    damage = {
        V35B: sum(row["candidate_damage"] for row in complete),
        GENERAL: sum(row["general_damage"] for row in complete),
    }
    rescue = {
        V35B: sum(row["candidate_rescue"] for row in complete),
        GENERAL: sum(row["general_rescue"] for row in complete),
    }
    damage_rate = {
        arm: damage[arm] / dense_passes if dense_passes else None
        for arm in (V35B, GENERAL)
    }
    branches = sum(row["branch_reached"] for row in complete)
    fallbacks = sum(row["target_fallbacks"] for row in complete)
    candidate_only = sum(
        row["resolved"][V35B] == 1 and row["resolved"][GENERAL] == 0
        for row in complete
    )
    general_only = sum(
        row["resolved"][V35B] == 0 and row["resolved"][GENERAL] == 1
        for row in complete
    )
    gates = {
        "official_tasks_completed": len(complete) == SAMPLE_SIZE,
        "runtime_mechanics_passes": (
            len(complete) == SAMPLE_SIZE
            and all(row["runtime_status"] == "PASS" for row in complete)
        ),
        "tasks_with_online_branch_min": branches >= 3,
        "target_fallbacks": fallbacks == 0,
        "v35b_resolved_strictly_above_general": (
            resolved[V35B] > resolved[GENERAL]
        ),
        "v35b_resolved_strictly_above_dense": (
            resolved[V35B] > resolved[DENSE]
        ),
        "v35b_damage_not_above_general": (
            damage[V35B] <= damage[GENERAL]
        ),
        "v35b_damage_rate_below_cacheblend": (
            damage_rate[V35B] is not None
            and damage_rate[V35B] < CACHEBLEND_DAMAGE_RATE
        ),
        "v35b_rescue_not_below_general": rescue[V35B] >= rescue[GENERAL],
        "v35b_only_vs_general_min": candidate_only >= 1,
        "report_overall_accuracy_damage_rescue_speed_separately": True,
        "do_not_make_population_or_sota_claim": True,
    }
    value = {
        "completed_at_utc": utc_now(),
        "status": (
            "PASS_V36_DEVELOPMENT"
            if all(gates.values())
            else "INCOMPLETE_V36"
            if len(complete) < SAMPLE_SIZE
            else "FAIL_V36_DEVELOPMENT"
        ),
        "registration_sha256": sha256(output / "V36_REGISTRATION.json"),
        "tasks": rows,
        "aggregate": {
            "complete_tasks": len(complete),
            "tasks_with_online_branch": branches,
            "resolved": resolved,
            "accuracy": {
                arm: resolved[arm] / len(complete) if complete else None
                for arm in ARMS
            },
            "accuracy_wilson95": {
                arm: _wilson(resolved[arm], len(complete)) for arm in ARMS
            },
            "candidate_minus_general_bootstrap95": _bootstrap(
                [row["candidate_minus_general"] for row in complete]
            ),
            "candidate_minus_dense_bootstrap95": _bootstrap(
                [row["candidate_minus_dense"] for row in complete]
            ),
            "paired_candidate_only_vs_general_only": {
                V35B: candidate_only,
                GENERAL: general_only,
            },
            "dense_passes": dense_passes,
            "dense_fails": dense_fails,
            "damage_count_given_dense_pass": damage,
            "damage_rate_given_dense_pass": damage_rate,
            "cacheblend_damage_rate_reference": CACHEBLEND_DAMAGE_RATE,
            "rescue_count_given_dense_fail": rescue,
            "task_median_ttft_ms_fixed_order_diagnostic": {
                arm: _median(
                    [row["median_ttft_ms"][arm] for row in complete]
                )
                for arm in ARMS
            },
            "target_fallbacks": fallbacks,
        },
        "gate_outcomes": gates,
        "interpretation": (
            "Development evidence from a previously evaluated pool. A pass "
            "permits counterbalanced speed work and an outcome-independent "
            "Verified replication; it does not establish population or SOTA "
            "superiority."
        ),
        "registered_gates": registration["frozen_development_gates"],
    }
    write_json(output / "V36_RESULT.json", value)
    return value


def run(output: Path) -> dict[str, Any]:
    registration = register(output)
    preregister_children(output)
    stages = []
    for task in registration["selection"]["tasks"]:
        instance_id = task["instance_id"]
        child = task_dir(output, instance_id)
        if not (child / "V25_RESULT.json").exists():
            stages.append(_run_stage(output, instance_id, "run"))
        if (
            (child / "V25_RESULT.json").exists()
            and not (child / "V25_OFFICIAL_RESULT.json").exists()
        ):
            stages.append(_run_stage(output, instance_id, "evaluate"))
    write_json(output / "V36_STAGE_STATUS.json", stages)
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
