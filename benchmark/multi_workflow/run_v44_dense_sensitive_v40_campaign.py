#!/usr/bin/env python3
"""Run V40 on a preregistered Dense-sensitive Verified development sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import (
    run_v43_new_verified_v40_campaign as prior,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v44_dense_sensitive_v40_20260728"
)
V43_AUDIT = (
    ARTIFACTS
    / "impactkv_v43_new_verified_v40_20260728"
    / "V43_CALL_BUDGET_COLLAPSE_AUDIT.json"
)
V40 = "coding_grounded_observation_island_v40"
GENERAL = "general"
DENSE = "dense"
ARMS = (V40, GENERAL, DENSE)
STEP_LIMIT = 32
SELECTION_SALT = "v44-new-to-v40-verified-step32-twelve-v1"
TASKS = (
    "astropy__astropy-13398",
    "psf__requests-2931",
    "sphinx-doc__sphinx-11445",
    "sympy__sympy-13551",
    "scikit-learn__scikit-learn-12682",
    "pylint-dev__pylint-6528",
    "astropy__astropy-7671",
    "scikit-learn__scikit-learn-10297",
    "mwaskom__seaborn-3069",
    "pytest-dev__pytest-7324",
    "pytest-dev__pytest-10051",
    "django__django-15561",
)
SELECTION_SHA256 = (
    "78663ee17c09445d23fb6a3cbe05e21ea6c7401871a2f93796d6aa4e171f8bf6"
)
CACHEBLEND_DAMAGE_RATE = 9 / 167
DENSE_PASS_SENSITIVITY_MIN = 2


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


def _selected_tasks() -> tuple[str, ...]:
    repo_of = {
        instance_id: repo
        for repo, instances in prior.ELIGIBLE_BY_REPO.items()
        for instance_id in instances
    }
    candidates = [
        instance_id
        for instance_id in repo_of
        if instance_id not in prior.TASKS
    ]
    ranked = sorted(
        candidates,
        key=lambda instance_id: (
            hashlib.sha256(
                f"{SELECTION_SALT}:task:{instance_id}".encode()
            ).hexdigest(),
            instance_id,
        ),
    )
    selected: list[str] = []
    per_repo: dict[str, int] = {}
    for instance_id in ranked:
        repo = repo_of[instance_id]
        if per_repo.get(repo, 0) >= 2:
            continue
        selected.append(instance_id)
        per_repo[repo] = per_repo.get(repo, 0) + 1
        if len(selected) == len(TASKS):
            break
    return tuple(selected)


def _prepare_inputs(output: Path) -> tuple[Path, Path, Path]:
    rows = prior._population_rows()
    indexed = {str(row["instance_id"]): row for row in rows}
    selected = [indexed[instance_id] for instance_id in TASKS]
    snapshot = output / "V44_FROZEN_SUBSET.json"
    dataset = output / "minisweagent_dataset"
    dataset_path = dataset / "test.jsonl"
    evaluation_registration = output / "V44_EVAL_REGISTRATION.json"
    write_json(snapshot, selected)
    dataset.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in selected
        ),
        encoding="utf-8",
    )
    write_json(
        dataset / "DATASET_MANIFEST.json",
        {
            "source": str(prior.POPULATION),
            "source_sha256": sha256(prior.POPULATION),
            "instances": list(TASKS),
            "count": len(TASKS),
            "test_jsonl_sha256": sha256(dataset_path),
        },
    )
    write_json(
        evaluation_registration,
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
            "instances": [
                {"instance_id": instance_id} for instance_id in TASKS
            ],
        },
    )
    return dataset, snapshot, evaluation_registration


def register(output: Path) -> dict[str, Any]:
    path = output / "V44_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if _selection_hash() != SELECTION_SHA256:
        raise AssertionError("V44 task selection changed")
    if _selected_tasks() != TASKS:
        raise AssertionError("V44 frozen selection rule changed")
    audit = read_json(V43_AUDIT)
    if (
        audit["status"]
        != "V43_BENCHMARK_SENSITIVITY_FAILURE_CALL_BUDGET_COLLAPSE"
    ):
        raise AssertionError("V43 call-budget audit changed")
    dataset, snapshot, evaluation_registration = _prepare_inputs(output)
    population = {
        row["instance_id"]: row for row in prior._population_rows()
    }
    selected = [
        {
            "instance_id": instance_id,
            "repo": population[instance_id]["repo"],
            "difficulty": population[instance_id].get("difficulty"),
        }
        for instance_id in TASKS
    ]
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_ANY_V44_TREATMENT",
        "experiment": (
            "V44 Dense-sensitive V40/General/Dense Verified development "
            "campaign"
        ),
        "motivation": (
            "V43 executed real V40 and General KV copies but all 18 arms "
            "exhausted the 20-call budget without submitting a patch, "
            "including every Dense control. V44 keeps the algorithm fixed, "
            "uses 12 additional tasks chosen without V44 outcomes, raises the "
            "common all-arm call budget to 32, and captures tracked worktree "
            "diffs at the limit so protocol bookkeeping cannot erase a late "
            "edit. Accuracy is split into Dense-pass preservation and "
            "Dense-fail rescue before any treatment."
        ),
        "selection": {
            "population": str(prior.POPULATION),
            "population_sha256": sha256(prior.POPULATION),
            "eligibility": (
                "Installed official evaluator image; not selected in V43; "
                "new to V40 treatment. Prior project outcome exposure may "
                "exist and is disclosed."
            ),
            "rule": (
                "Remove the six V43 tasks, rank remaining eligible instances "
                "by sha256(salt:task:instance_id), take the first 12 while "
                "capping each repository at two tasks."
            ),
            "salt": SELECTION_SALT,
            "selection_sha256": SELECTION_SHA256,
            "tasks": selected,
            "gold_patch_used": False,
            "v44_outcomes_used_for_selection": False,
            "replacement_on_failure": False,
            "outcome_exposure_class": (
                "NEW_TO_V40_DEVELOPMENT; PRIOR_PROJECT_OUTCOME_EXPOSURE_MAY_EXIST"
            ),
        },
        "protocol": {
            "arms": list(ARMS),
            "task_level_intention_to_treat": True,
            "all_children_registered_before_first_treatment": True,
            "shared_dense_history_before_branch": True,
            "container_snapshot_before_branch": True,
            "paired_accuracy_sources_reside_on_host": True,
            "formal_speed_uses_separate_device_resident_protocol": True,
            "step_limit": STEP_LIMIT,
            "temperature": 0,
            "request_timeout_seconds": 180,
            "capture_tracked_worktree_patch_at_call_limit": True,
            "terminal_capture_model_calls": 0,
            "terminal_capture_treatment_invariant": True,
            "official_swebench_container_each_completed_arm": True,
            "fixed_order_ttft_is_diagnostic_only": True,
            "accuracy_primary": (
                "damage among contemporaneous Dense-pass tasks"
            ),
            "accuracy_secondary": (
                "rescue among contemporaneous Dense-fail tasks"
            ),
            "dense_pass_cohort_rule_frozen_before_treatment": True,
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "frozen_development_gates": {
            "official_tasks_completed": len(TASKS),
            "runtime_mechanics_passes": len(TASKS),
            "dense_pass_sensitivity_min": DENSE_PASS_SENSITIVITY_MIN,
            "tasks_with_online_branch_min": 6,
            "candidate_copy_requests_min": 12,
            "candidate_copied_tokens_strictly_below_general": True,
            "candidate_assistant_tokens_selected": 0,
            "device_sources": 0,
            "target_fallbacks": 0,
            "v40_resolved_strictly_above_general": True,
            "v40_resolved_not_below_dense": True,
            "v40_damage_strictly_below_general": True,
            "v40_damage_rate_below_cacheblend": CACHEBLEND_DAMAGE_RATE,
            "v40_rescue_not_below_general": True,
            "v40_only_vs_general_min": 1,
            "report_accuracy_damage_rescue_speed_separately": True,
            "do_not_make_population_or_sota_claim": True,
        },
        "inputs": {
            "dataset": str(dataset / "test.jsonl"),
            "dataset_sha256": sha256(dataset / "test.jsonl"),
            "evaluation_snapshot": str(snapshot),
            "evaluation_snapshot_sha256": sha256(snapshot),
            "evaluation_registration": str(evaluation_registration),
            "evaluation_registration_sha256": sha256(
                evaluation_registration
            ),
            "v43_audit": str(V43_AUDIT),
            "v43_audit_sha256": sha256(V43_AUDIT),
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
                output / "V44_EVAL_REGISTRATION.json"
            ),
            "IMPACTKV_EVAL_SNAPSHOT": str(
                output / "V44_FROZEN_SUBSET.json"
            ),
            "IMPACTKV_AGENT_STEP_LIMIT": str(STEP_LIMIT),
            "IMPACTKV_CAPTURE_LIMIT_PATCH": "1",
        }
    )
    return env


def _run_stage(
    output: Path, instance_id: str, stage: str
) -> dict[str, Any]:
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
    status_path = (
        output / "orchestration_status" / instance_id / f"{stage}.json"
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(status_path, value)
    return value


def preregister_children(output: Path) -> list[dict[str, Any]]:
    _configure()
    registration = register(output)
    rows = [
        _run_stage(output, task["instance_id"], "register")
        for task in registration["selection"]["tasks"]
    ]
    write_json(output / "V44_CHILD_REGISTRATIONS.json", rows)
    if any(row["returncode"] != 0 for row in rows):
        raise RuntimeError("one or more V44 child registrations failed")
    return rows


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
        resolved = {
            arm: int(official["arms"][arm]["resolved"]) for arm in ARMS
        }
        clients = {
            arm: prior._client(child / arm / "CLIENT_LEDGER.jsonl")
            for arm in ARMS
        }
        server = prior._jsonl(child / "run" / "SERVER_LEDGER.jsonl")
        decisions = [
            row.get("reuse_policy_decision", {})
            for row in clients[V40]
            if row.get("reuse_policy_decision", {}).get("mode")
            == "grounded_version_valid_observation_island"
        ]
        rows.append(
            {
                **selected,
                "status": "COMPLETE",
                "runtime_status": runtime["status"],
                "branch_reached": runtime["branch"] is not None,
                "resolved": resolved,
                "empty_patch": {
                    arm: int(official["arms"][arm]["empty_patch"])
                    for arm in ARMS
                },
                "terminal_patch_captured": runtime.get(
                    "terminal_patch_captured",
                    {arm: False for arm in ARMS},
                ),
                "copy_requests": {
                    arm: sum(
                        int(row["copied_tokens_planned"]) > 0
                        for row in clients[arm]
                    )
                    for arm in ARMS
                },
                "copied_tokens": {
                    arm: sum(
                        int(row["copied_tokens_planned"])
                        for row in clients[arm]
                    )
                    for arm in ARMS
                },
                "assistant_tokens_selected": max(
                    [
                        int(value["assistant_tokens_selected"])
                        for value in decisions
                    ],
                    default=0,
                ),
                "device_sources": sum(
                    row.get("event") == "source_materialized"
                    for row in server
                ),
                "host_sources": sum(
                    row.get("event") == "source_materialized_host"
                    and row.get("reason") == "preferred_host_residency"
                    for row in server
                ),
                "target_fallbacks": runtime["server"]["target_fallbacks"],
                "median_ttft_ms": {
                    arm: official["arms"][arm]["median_ttft_ms"]
                    for arm in ARMS
                },
            }
        )
    complete = [row for row in rows if row["status"] == "COMPLETE"]
    resolved = {
        arm: sum(row["resolved"][arm] for row in complete) for arm in ARMS
    }
    dense_passes = sum(row["resolved"][DENSE] for row in complete)
    dense_fails = len(complete) - dense_passes
    damage = {
        arm: sum(
            row["resolved"][DENSE] == 1 and row["resolved"][arm] == 0
            for row in complete
        )
        for arm in (V40, GENERAL)
    }
    rescue = {
        arm: sum(
            row["resolved"][DENSE] == 0 and row["resolved"][arm] == 1
            for row in complete
        )
        for arm in (V40, GENERAL)
    }
    damage_rate = {
        arm: damage[arm] / dense_passes if dense_passes else None
        for arm in (V40, GENERAL)
    }
    branches = sum(row["branch_reached"] for row in complete)
    copy_requests = {
        arm: sum(row["copy_requests"][arm] for row in complete)
        for arm in ARMS
    }
    copied_tokens = {
        arm: sum(row["copied_tokens"][arm] for row in complete)
        for arm in ARMS
    }
    candidate_only = sum(
        row["resolved"][V40] == 1 and row["resolved"][GENERAL] == 0
        for row in complete
    )
    general_only = sum(
        row["resolved"][V40] == 0 and row["resolved"][GENERAL] == 1
        for row in complete
    )
    gates = {
        "official_tasks_completed": len(complete) == len(TASKS),
        "runtime_mechanics_passes": len(complete) == len(TASKS)
        and all(row["runtime_status"] == "PASS" for row in complete),
        "dense_pass_sensitivity_min": (
            dense_passes >= DENSE_PASS_SENSITIVITY_MIN
        ),
        "tasks_with_online_branch_min": branches >= 6,
        "candidate_copy_requests_min": copy_requests[V40] >= 12,
        "candidate_copied_tokens_strictly_below_general": (
            copied_tokens[V40] < copied_tokens[GENERAL]
        ),
        "candidate_assistant_tokens_selected": all(
            row["assistant_tokens_selected"] == 0 for row in complete
        ),
        "device_sources": (
            sum(row["device_sources"] for row in complete) == 0
        ),
        "target_fallbacks": (
            sum(row["target_fallbacks"] for row in complete) == 0
        ),
        "v40_resolved_strictly_above_general": (
            resolved[V40] > resolved[GENERAL]
        ),
        "v40_resolved_not_below_dense": resolved[V40] >= resolved[DENSE],
        "v40_damage_strictly_below_general": (
            damage[V40] < damage[GENERAL]
        ),
        "v40_damage_rate_below_cacheblend": (
            damage_rate[V40] is not None
            and damage_rate[V40] < CACHEBLEND_DAMAGE_RATE
        ),
        "v40_rescue_not_below_general": rescue[V40] >= rescue[GENERAL],
        "v40_only_vs_general_min": candidate_only >= 1,
        "report_accuracy_damage_rescue_speed_separately": True,
        "do_not_make_population_or_sota_claim": True,
    }
    value = {
        "completed_at_utc": utc_now(),
        "status": (
            "INCOMPLETE_V44"
            if len(complete) < len(TASKS)
            else "INCONCLUSIVE_V44_NO_DENSE_SENSITIVITY"
            if dense_passes < DENSE_PASS_SENSITIVITY_MIN
            else "PASS_V44_DEVELOPMENT"
            if all(gates.values())
            else "FAIL_V44_DEVELOPMENT"
        ),
        "registration_sha256": sha256(
            output / "V44_REGISTRATION.json"
        ),
        "tasks": rows,
        "aggregate": {
            "complete_tasks": len(complete),
            "tasks_with_online_branch": branches,
            "dense_pass_tasks": dense_passes,
            "dense_fail_tasks": dense_fails,
            "resolved": resolved,
            "accuracy": {
                arm: resolved[arm] / len(complete) if complete else None
                for arm in ARMS
            },
            "accuracy_wilson95": {
                arm: prior._wilson(resolved[arm], len(complete))
                for arm in ARMS
            },
            "paired_candidate_only_vs_general_only": {
                V40: candidate_only,
                GENERAL: general_only,
            },
            "damage_count_given_dense_pass": damage,
            "damage_rate_given_dense_pass": damage_rate,
            "damage_wilson95_given_dense_pass": {
                arm: prior._wilson(damage[arm], dense_passes)
                for arm in (V40, GENERAL)
            },
            "cacheblend_damage_rate_reference": CACHEBLEND_DAMAGE_RATE,
            "rescue_count_given_dense_fail": rescue,
            "rescue_rate_given_dense_fail": {
                arm: rescue[arm] / dense_fails if dense_fails else None
                for arm in (V40, GENERAL)
            },
            "copy_requests": copy_requests,
            "copied_tokens": copied_tokens,
            "terminal_patch_captures": {
                arm: sum(
                    row["terminal_patch_captured"][arm]
                    for row in complete
                )
                for arm in ARMS
            },
            "empty_patches": {
                arm: sum(row["empty_patch"][arm] for row in complete)
                for arm in ARMS
            },
            "device_sources": sum(
                row["device_sources"] for row in complete
            ),
            "host_sources": sum(row["host_sources"] for row in complete),
            "target_fallbacks": sum(
                row["target_fallbacks"] for row in complete
            ),
            "fixed_order_ttft_diagnostic_ms": {
                arm: statistics.median(
                    row["median_ttft_ms"][arm] for row in complete
                )
                if complete
                else None
                for arm in ARMS
            },
        },
        "gate_outcomes": gates,
        "registered_gates": registration["frozen_development_gates"],
        "interpretation": (
            "V44 is a development accuracy-sensitivity experiment, not a "
            "population or SOTA result. Dense-pass damage is the primary "
            "preservation metric; Dense-fail rescue is reported separately. "
            "Fixed-order TTFT remains diagnostic and cannot support speed "
            "claims."
        ),
    }
    write_json(output / "V44_RESULT.json", value)
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
        if (
            (child / "V25_RESULT.json").exists()
            and not (child / "V25_OFFICIAL_RESULT.json").exists()
        ):
            stages.append(_run_stage(output, instance_id, "evaluate"))
    write_json(output / "V44_STAGE_STATUS.json", stages)
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
    value = (
        register(args.output)
        if args.command == "register"
        else {"children": preregister_children(args.output)}
        if args.command == "preregister"
        else summarize(args.output)
        if args.command == "summarize"
        else run(args.output)
    )
    print(
        {
            "status": value.get("status"),
            "aggregate": value.get("aggregate"),
        }
    )


if __name__ == "__main__":
    main()
