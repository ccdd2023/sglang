#!/usr/bin/env python3
"""Run a frozen fresh-13 Dense/General/V40 SWE-bench campaign for M55."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import run_v43_new_verified_v40_campaign as prior
from benchmark.multi_workflow import run_bridge_reuse_agent_experiment as bridge
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_m55_v40_task_disjoint_20260805"
V44_RESULT = (
    ARTIFACTS
    / "impactkv_v44_dense_sensitive_v40_20260728"
    / "V44_RESULT.json"
)
V40 = "coding_grounded_observation_island_v40"
GENERAL = "general"
DENSE = "dense"
ARMS = (V40, GENERAL, DENSE)
STEP_LIMIT = 32
SELECTION_SALT = "m55-task-disjoint-v40-fresh13-v1"
TASKS = (
    "astropy__astropy-13033",
    "django__django-12406",
    "django__django-16560",
    "psf__requests-6028",
    "pydata__xarray-3095",
    "pydata__xarray-3305",
    "pydata__xarray-6992",
    "pylint-dev__pylint-4551",
    "pylint-dev__pylint-4661",
    "pytest-dev__pytest-5787",
    "scikit-learn__scikit-learn-14087",
    "sphinx-doc__sphinx-7590",
    "sphinx-doc__sphinx-8120",
)
SELECTION_SHA256 = "9b43dfce4b7c4293846a4c9a1f34015f359ae34603328a57d7bae832cc4a3fcf"


def task_dir(output: Path, instance_id: str) -> Path:
    return output / "tasks" / instance_id


def _configure() -> None:
    prior.orchestration.V38 = V40


def _selection_hash() -> str:
    value = json.dumps(
        {"salt": SELECTION_SALT, "tasks": list(TASKS)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode()).hexdigest()


def _prepare_inputs(output: Path) -> None:
    population = {str(row["instance_id"]): row for row in prior._population_rows()}
    missing = sorted(set(TASKS) - set(population))
    if missing:
        raise ValueError(f"fresh tasks absent from local population: {missing}")
    selected = [population[task] for task in TASKS]
    snapshot = output / "M55_FROZEN_SUBSET.json"
    dataset_root = output / "minisweagent_dataset"
    dataset_path = dataset_root / "test.jsonl"
    evaluation = output / "M55_EVAL_REGISTRATION.json"
    write_json(snapshot, selected)
    dataset_root.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    write_json(
        evaluation,
        {
            "schema_version": 1,
            "registration_id": output.name,
            "registered_at_utc": utc_now(),
            "dataset": {
                "name": "princeton-nlp/SWE-bench_Verified",
                "split": "test",
                "population_size": 500,
                "local_snapshot": str(snapshot),
                "local_snapshot_sha256": sha256(snapshot),
            },
            "instances": [{"instance_id": task} for task in TASKS],
        },
    )


def register(output: Path) -> dict[str, Any]:
    path = output / "M55_TASK_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    if _selection_hash() != SELECTION_SHA256:
        raise AssertionError("fresh-13 task registration changed")
    output.mkdir(parents=True)
    _prepare_inputs(output)
    population = {str(row["instance_id"]): row for row in prior._population_rows()}
    selected = [
        {
            "instance_id": task,
            "repo": population[task]["repo"],
            "difficulty": population[task].get("difficulty"),
        }
        for task in TASKS
    ]
    value = {
        "status": "REGISTERED_BEFORE_ANY_FRESH13_TREATMENT",
        "registered_at_utc": utc_now(),
        "selection": {
            "salt": SELECTION_SALT,
            "selection_sha256": SELECTION_SHA256,
            "tasks": selected,
            "replacement_on_failure": False,
            "outcome_used_for_selection": False,
            "task_disjoint_from_m52_m53_m54": True,
            "prior_project_exposure_may_exist": True,
        },
        "protocol": {
            "arms": list(ARMS),
            "model": bridge.MODEL,
            "step_limit": STEP_LIMIT,
            "temperature": 0,
            "rolling_history_groups": 6,
            "prefetch": False,
            "official_container_evaluation": True,
            "fixed_order_ttft_is_diagnostic_only": True,
        },
        "frozen_gates": {
            "official_tasks_completed": len(TASKS),
            "v40_copy_tasks_min": 8,
            "v40_copy_requests_min": 12,
            "target_fallbacks": 0,
            "assistant_tokens_selected": 0,
            "fresh_v40_resolved_ge_general": True,
            "fresh_v40_resolved_ge_dense_minus_one": True,
            "fresh_v40_damage_le_one": True,
            "fresh_v40_damage_le_general": True,
            "pooled25_v40_resolved_ge_dense_and_general": True,
        },
        "inputs": {
            "population": str(prior.POPULATION),
            "population_sha256": sha256(prior.POPULATION),
            "dataset": str(output / "minisweagent_dataset/test.jsonl"),
            "dataset_sha256": sha256(output / "minisweagent_dataset/test.jsonl"),
            "snapshot": str(output / "M55_FROZEN_SUBSET.json"),
            "snapshot_sha256": sha256(output / "M55_FROZEN_SUBSET.json"),
            "runner_sha256": sha256(prior.orchestration.RUNNER),
            "script_sha256": sha256(Path(__file__)),
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


def _environment(output: Path, instance_id: str) -> dict[str, str]:
    env = prior.orchestration._environment(instance_id)
    env.update(
        {
            "IMPACTKV_DATASET_ROOT": str(output / "minisweagent_dataset"),
            "IMPACTKV_EVAL_REGISTRATION": str(
                output / "M55_EVAL_REGISTRATION.json"
            ),
            "IMPACTKV_EVAL_SNAPSHOT": str(output / "M55_FROZEN_SUBSET.json"),
            "IMPACTKV_AGENT_STEP_LIMIT": str(STEP_LIMIT),
            "IMPACTKV_CAPTURE_LIMIT_PATCH": "1",
        }
    )
    return env


def _run_stage(output: Path, instance_id: str, stage: str) -> dict[str, Any]:
    log_path = output / "orchestration_logs" / instance_id / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(prior.orchestration.PYTHON),
        str(prior.orchestration.RUNNER),
        stage,
        "--output",
        str(task_dir(output, instance_id)),
    ]
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=prior.orchestration.PROJECT,
            env=_environment(output, instance_id),
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
    status = output / "orchestration_status" / instance_id / f"{stage}.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    write_json(status, value)
    return value


def preregister_children(output: Path) -> list[dict[str, Any]]:
    _configure()
    registration = register(output)
    rows = [
        _run_stage(output, row["instance_id"], "register")
        for row in registration["selection"]["tasks"]
    ]
    write_json(output / "M55_CHILD_REGISTRATIONS.json", rows)
    if any(row["returncode"] != 0 for row in rows):
        raise RuntimeError("one or more fresh-13 child registrations failed")
    return rows


def _pooled_resolved(fresh: dict[str, int]) -> dict[str, int] | None:
    if not V44_RESULT.exists():
        return None
    old = read_json(V44_RESULT)["aggregate"]["resolved"]
    return {arm: int(old[arm]) + fresh[arm] for arm in ARMS}


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    rows = []
    for selected in registration["selection"]["tasks"]:
        instance_id = selected["instance_id"]
        child = task_dir(output, instance_id)
        runtime_path = child / "V25_RESULT.json"
        official_path = child / "V25_OFFICIAL_RESULT.json"
        if not runtime_path.exists() or not official_path.exists():
            rows.append({**selected, "status": "INCOMPLETE"})
            continue
        runtime = read_json(runtime_path)
        official = read_json(official_path)
        clients = {
            arm: prior._client(child / arm / "CLIENT_LEDGER.jsonl") for arm in ARMS
        }
        decisions = [
            item.get("reuse_policy_decision", {})
            for item in clients[V40]
            if item.get("reuse_policy_decision", {}).get("mode")
            == "grounded_version_valid_observation_island"
        ]
        rows.append(
            {
                **selected,
                "status": "COMPLETE",
                "runtime_status": runtime["status"],
                "resolved": {
                    arm: int(official["arms"][arm]["resolved"]) for arm in ARMS
                },
                "copy_requests": {
                    arm: sum(int(item["copied_tokens_planned"]) > 0 for item in clients[arm])
                    for arm in ARMS
                },
                "copied_tokens": {
                    arm: sum(int(item["copied_tokens_planned"]) for item in clients[arm])
                    for arm in ARMS
                },
                "assistant_tokens_selected": max(
                    [int(item["assistant_tokens_selected"]) for item in decisions],
                    default=0,
                ),
                "target_fallbacks": int(runtime["server"]["target_fallbacks"]),
                "median_ttft_ms": {
                    arm: official["arms"][arm]["median_ttft_ms"] for arm in ARMS
                },
            }
        )
    complete = [row for row in rows if row["status"] == "COMPLETE"]
    resolved = {arm: sum(row["resolved"][arm] for row in complete) for arm in ARMS}
    damage = {
        arm: sum(row["resolved"][DENSE] == 1 and row["resolved"][arm] == 0 for row in complete)
        for arm in (V40, GENERAL)
    }
    rescue = {
        arm: sum(row["resolved"][DENSE] == 0 and row["resolved"][arm] == 1 for row in complete)
        for arm in (V40, GENERAL)
    }
    copy_requests = {
        arm: sum(row["copy_requests"][arm] for row in complete) for arm in ARMS
    }
    copied_tokens = {
        arm: sum(row["copied_tokens"][arm] for row in complete) for arm in ARMS
    }
    copy_tasks = sum(row["copy_requests"][V40] > 0 for row in complete)
    pooled = _pooled_resolved(resolved) if len(complete) == len(TASKS) else None
    gates = {
        "official_tasks_completed": len(complete) == len(TASKS),
        "v40_copy_tasks_min": copy_tasks >= 8,
        "v40_copy_requests_min": copy_requests[V40] >= 12,
        "target_fallbacks": sum(row["target_fallbacks"] for row in complete) == 0,
        "assistant_tokens_selected": all(
            row["assistant_tokens_selected"] == 0 for row in complete
        ),
        "fresh_v40_resolved_ge_general": resolved[V40] >= resolved[GENERAL],
        "fresh_v40_resolved_ge_dense_minus_one": resolved[V40] >= resolved[DENSE] - 1,
        "fresh_v40_damage_le_one": damage[V40] <= 1,
        "fresh_v40_damage_le_general": damage[V40] <= damage[GENERAL],
        "pooled25_v40_resolved_ge_dense_and_general": pooled is not None
        and pooled[V40] >= pooled[DENSE]
        and pooled[V40] >= pooled[GENERAL],
    }
    value = {
        "status": (
            "INCOMPLETE_FRESH13"
            if len(complete) < len(TASKS)
            else "SUPPORTED_V40_RATIONALE"
            if all(gates.values())
            else "NOT_SUPPORTED_V40_RATIONALE"
        ),
        "completed_at_utc": utc_now(),
        "registration_sha256": sha256(output / "M55_TASK_REGISTRATION.json"),
        "tasks": rows,
        "aggregate": {
            "complete_tasks": len(complete),
            "resolved": resolved,
            "accuracy": {
                arm: resolved[arm] / len(complete) if complete else None for arm in ARMS
            },
            "accuracy_wilson95": {
                arm: prior._wilson(resolved[arm], len(complete)) for arm in ARMS
            },
            "damage_count_given_dense_pass": damage,
            "rescue_count_given_dense_fail": rescue,
            "copy_tasks_v40": copy_tasks,
            "copy_requests": copy_requests,
            "copied_tokens": copied_tokens,
            "target_fallbacks": sum(row["target_fallbacks"] for row in complete),
            "fixed_order_ttft_diagnostic_ms": {
                arm: statistics.median(row["median_ttft_ms"][arm] for row in complete)
                if complete
                else None
                for arm in ARMS
            },
            "pooled_v44_plus_fresh13_resolved": pooled,
        },
        "gate_outcomes": gates,
        "registered_gates": registration["frozen_gates"],
        "interpretation": (
            "Fresh task-disjoint V40 rationale validation. Fixed-order agent TTFT "
            "is diagnostic; formal speed requires the separate same-prompt replay."
        ),
    }
    write_json(output / "M55_TASK_RESULT.json", value)
    return value


def run(output: Path) -> dict[str, Any]:
    _configure()
    registration = register(output)
    preregister_children(output)
    stages = []
    for selected in registration["selection"]["tasks"]:
        instance_id = selected["instance_id"]
        child = task_dir(output, instance_id)
        if not (child / "V25_RESULT.json").exists():
            stage = _run_stage(output, instance_id, "run")
            stages.append(stage)
            if stage["returncode"] != 0:
                continue
        if (child / "V25_RESULT.json").exists() and not (
            child / "V25_OFFICIAL_RESULT.json"
        ).exists():
            stages.append(_run_stage(output, instance_id, "evaluate"))
    write_json(output / "M55_TASK_STAGE_STATUS.json", stages)
    return summarize(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("register", "preregister", "run", "summarize"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = (
        register(args.output)
        if args.command == "register"
        else {"children": preregister_children(args.output)}
        if args.command == "preregister"
        else summarize(args.output)
        if args.command == "summarize"
        else run(args.output)
    )
    print(json.dumps({"status": value.get("status"), "aggregate": value.get("aggregate")}, indent=2))


if __name__ == "__main__":
    main()
