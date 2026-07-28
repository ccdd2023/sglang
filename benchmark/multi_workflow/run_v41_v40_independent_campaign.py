#!/usr/bin/env python3
"""Run the preregistered six-task V40/General/Dense development campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import (
    run_v39_v38_independent_campaign as orchestration,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    utc_now,
    write_json,
)
from benchmark.multi_workflow.run_v40a_grounded_observation_canary import (
    _client,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v41_v40_independent_20260728"
MOTIVATION = (
    ARTIFACTS
    / "impactkv_v40_grounded_observation_motivation_20260728"
    / "V40_MOTIVATION_RESULT.json"
)
CANARY = (
    ARTIFACTS
    / "impactkv_v40a3_short_grounded_canary_20260728"
    / "V40A3_RESULT.json"
)
V40 = "coding_grounded_observation_island_v40"
GENERAL = "general"
DENSE = "dense"
ARMS = (V40, GENERAL, DENSE)
TASKS = (
    "astropy__astropy-14995",
    "django__django-16899",
    "psf__requests-1142",
    "psf__requests-5414",
    "sphinx-doc__sphinx-7440",
    "sympy__sympy-24562",
)
SELECTION_SALT = "v41-v40-all-remaining-v1"
SELECTION_SHA256 = (
    "482657736cd606cba17beb36dd3202c9db3a322c28ddd1590fec5ef7dbb698cf"
)
CACHEBLEND_DAMAGE_RATE = 9 / 167


def _configure_orchestration() -> None:
    orchestration.V38 = V40


def _selection_hash() -> str:
    value = json.dumps(
        {"salt": SELECTION_SALT, "tasks": list(TASKS)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode()).hexdigest()


def _motivation_rows() -> dict[str, dict[str, Any]]:
    value = read_json(MOTIVATION)
    return {
        row["instance_id"]: row for row in value["cohorts"]["full18"]
    }


def register(output: Path) -> dict[str, Any]:
    path = output / "V41_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if _selection_hash() != SELECTION_SHA256:
        raise AssertionError("V41 task selection changed")
    motivation = _motivation_rows()
    selected = [motivation[instance_id] for instance_id in TASKS]
    if any(not row["reached"] for row in selected):
        raise AssertionError("V41 requires V40 source reach on every task")
    if read_json(CANARY)["status"] != "PASS_V40A3_MECHANISM_CANARY":
        raise AssertionError("V40A3 canary did not pass")
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_ANY_V41_TREATMENT",
        "experiment": "V41 independent V40 development sample",
        "motivation": (
            "V40A3 proved that grounded observation islands execute real KV "
            "copies and preserve one official task. Test whether excluding "
            "assistant decisions and version-invalid observations produces "
            "strict task accuracy separation from General."
        ),
        "selection": {
            "source": str(MOTIVATION),
            "source_sha256": sha256(MOTIVATION),
            "rule": (
                "Use all six remaining full18 tasks with V40 source reach "
                "after excluding the V39 six-task set, V40 A1/A2/A3 canaries, "
                "prior tuned mechanism canaries xarray-4075, Pylint-7277 and "
                "sklearn-12585, and prior timeout sklearn-13779. No outcome "
                "value orders, drops, or replaces a selected task."
            ),
            "salt": SELECTION_SALT,
            "selection_sha256": SELECTION_SHA256,
            "tasks": selected,
            "official_outcomes_used_for_selection": False,
            "replacement_on_failure": False,
            "outcome_exposure_class": (
                "DEVELOPMENT_POOL_PREVIOUSLY_EVALUATED_NOT_HELD_OUT"
            ),
        },
        "protocol": {
            "arms": list(ARMS),
            "task_level_intention_to_treat": True,
            "all_children_registered_before_first_treatment": True,
            "continue_after_task_infrastructure_failure": True,
            "shared_dense_history_before_branch": True,
            "container_snapshot_before_branch": True,
            "step_limit": 20,
            "temperature": 0,
            "request_timeout_seconds": 180,
            "official_swebench_container_each_completed_arm": True,
            "fixed_order_ttft_is_diagnostic_only": True,
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "frozen_development_gates": {
            "official_tasks_completed": len(TASKS),
            "runtime_mechanics_passes": len(TASKS),
            "tasks_with_online_branch_min": 5,
            "candidate_copy_requests_min": 6,
            "candidate_copied_tokens_strictly_below_general": True,
            "candidate_assistant_tokens_selected": 0,
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
            "motivation_sha256": sha256(MOTIVATION),
            "canary_sha256": sha256(CANARY),
            "runner_sha256": sha256(orchestration.RUNNER),
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


def preregister_children(output: Path) -> list[dict[str, Any]]:
    _configure_orchestration()
    registration = register(output)
    rows = []
    for task in registration["selection"]["tasks"]:
        instance_id = task["instance_id"]
        child = orchestration.task_dir(output, instance_id)
        rows.append(
            orchestration._run_stage(output, instance_id, "register")
            if not (child / "V25_REGISTRATION.json").exists()
            else {
                "instance_id": instance_id,
                "stage": "register",
                "returncode": 0,
                "resumed": True,
            }
        )
    write_json(output / "V41_CHILD_REGISTRATIONS.json", rows)
    if any(row["returncode"] != 0 for row in rows):
        raise RuntimeError("one or more V41 child registrations failed")
    return rows


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    rows = []
    for selected in registration["selection"]["tasks"]:
        instance_id = selected["instance_id"]
        child = orchestration.task_dir(output, instance_id)
        runtime_path = child / "V25_RESULT.json"
        official_path = child / "V25_OFFICIAL_RESULT.json"
        if not runtime_path.exists() or not official_path.exists():
            stage_path = (
                output
                / "orchestration_status"
                / instance_id
                / "run.json"
            )
            rows.append(
                {
                    **selected,
                    "status": "INCOMPLETE",
                    "run_returncode": (
                        read_json(stage_path)["returncode"]
                        if stage_path.exists()
                        else None
                    ),
                }
            )
            continue
        runtime = read_json(runtime_path)
        official = read_json(official_path)
        resolved = {
            arm: int(official["arms"][arm]["resolved"]) for arm in ARMS
        }
        clients = {
            arm: _client(child / arm / "CLIENT_LEDGER.jsonl")
            for arm in ARMS
        }
        copy_requests = {
            arm: sum(
                int(row["copied_tokens_planned"]) > 0
                for row in clients[arm]
            )
            for arm in ARMS
        }
        copied_tokens = {
            arm: sum(
                int(row["copied_tokens_planned"])
                for row in clients[arm]
            )
            for arm in ARMS
        }
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
                "copy_requests": copy_requests,
                "copied_tokens": copied_tokens,
                "assistant_tokens_selected": max(
                    [
                        int(value["assistant_tokens_selected"])
                        for value in decisions
                    ],
                    default=0,
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
        "tasks_with_online_branch_min": branches >= 5,
        "candidate_copy_requests_min": copy_requests[V40] >= 6,
        "candidate_copied_tokens_strictly_below_general": (
            copied_tokens[V40] < copied_tokens[GENERAL]
        ),
        "candidate_assistant_tokens_selected": all(
            row["assistant_tokens_selected"] == 0 for row in complete
        ),
        "target_fallbacks": sum(
            row["target_fallbacks"] for row in complete
        )
        == 0,
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
            "PASS_V41_DEVELOPMENT"
            if all(gates.values())
            else "INCOMPLETE_V41"
            if len(complete) < len(TASKS)
            else "FAIL_V41_DEVELOPMENT"
        ),
        "registration_sha256": sha256(
            output / "V41_REGISTRATION.json"
        ),
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
                arm: orchestration._wilson(resolved[arm], len(complete))
                for arm in ARMS
            },
            "paired_candidate_only_vs_general_only": {
                V40: candidate_only,
                GENERAL: general_only,
            },
            "damage_count_given_dense_pass": damage,
            "damage_rate_given_dense_pass": damage_rate,
            "rescue_count_given_dense_fail": rescue,
            "copy_requests": copy_requests,
            "copied_tokens": copied_tokens,
            "target_fallbacks": sum(
                row["target_fallbacks"] for row in complete
            ),
            "fixed_order_ttft_diagnostic_ms": {
                arm: orchestration._median(
                    [row["median_ttft_ms"][arm] for row in complete]
                )
                for arm in ARMS
            },
        },
        "gate_outcomes": gates,
        "registered_gates": registration["frozen_development_gates"],
        "interpretation": (
            "Outcome-independent selection from a previously evaluated "
            "development pool. A pass permits counterbalanced speed work; it "
            "is not held-out population or SOTA evidence."
        ),
    }
    write_json(output / "V41_RESULT.json", value)
    return value


def run(output: Path) -> dict[str, Any]:
    _configure_orchestration()
    registration = register(output)
    preregister_children(output)
    stages = []
    for selected in registration["selection"]["tasks"]:
        instance_id = selected["instance_id"]
        child = orchestration.task_dir(output, instance_id)
        if not (child / "V25_RESULT.json").exists():
            stage = orchestration._run_stage(
                output, instance_id, "run"
            )
            stages.append(stage)
            if stage["returncode"] != 0:
                continue
        if (
            (child / "V25_RESULT.json").exists()
            and not (child / "V25_OFFICIAL_RESULT.json").exists()
        ):
            stages.append(
                orchestration._run_stage(
                    output, instance_id, "evaluate"
                )
            )
    write_json(output / "V41_STAGE_STATUS.json", stages)
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
            "gate_outcomes": value.get("gate_outcomes"),
        }
    )


if __name__ == "__main__":
    main()
