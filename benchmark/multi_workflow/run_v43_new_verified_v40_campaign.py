#!/usr/bin/env python3
"""Run V40 on six new-to-V40 SWE-bench Verified development tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
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
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v43_new_verified_v40_20260728"
POPULATION = Path(
    "/home/gfy/CodeMAS_Project/sglang-kvflow/results/"
    "repo_level_datasets/swe_verified_500_instances.json"
)
V42 = (
    ARTIFACTS
    / "impactkv_v42_host_residency_canary_20260728"
    / "V42_RESULT.json"
)
V40 = "coding_grounded_observation_island_v40"
GENERAL = "general"
DENSE = "dense"
ARMS = (V40, GENERAL, DENSE)
SELECTION_SALT = "v43-preinstalled-new-to-v40-six-repo-v1"
ELIGIBLE_BY_REPO = {
    "astropy": (
        "astropy__astropy-7671",
        "astropy__astropy-13398",
        "astropy__astropy-13033",
    ),
    "django": (
        "django__django-16560",
        "django__django-16263",
        "django__django-15561",
        "django__django-12406",
    ),
    "mwaskom": (
        "mwaskom__seaborn-3187",
        "mwaskom__seaborn-3069",
    ),
    "psf": (
        "psf__requests-6028",
        "psf__requests-2931",
    ),
    "pydata": (
        "pydata__xarray-6992",
        "pydata__xarray-3305",
        "pydata__xarray-3095",
        "pydata__xarray-2905",
    ),
    "pylint-dev": (
        "pylint-dev__pylint-6528",
        "pylint-dev__pylint-4661",
        "pylint-dev__pylint-4551",
    ),
    "pytest-dev": (
        "pytest-dev__pytest-7324",
        "pytest-dev__pytest-5840",
        "pytest-dev__pytest-5787",
        "pytest-dev__pytest-10051",
    ),
    "scikit-learn": (
        "scikit-learn__scikit-learn-14087",
        "scikit-learn__scikit-learn-12682",
        "scikit-learn__scikit-learn-10297",
    ),
    "sphinx-doc": (
        "sphinx-doc__sphinx-9461",
        "sphinx-doc__sphinx-8120",
        "sphinx-doc__sphinx-7590",
        "sphinx-doc__sphinx-11445",
    ),
    "sympy": (
        "sympy__sympy-21930",
        "sympy__sympy-13551",
    ),
}
TASKS = (
    "sphinx-doc__sphinx-9461",
    "pydata__xarray-2905",
    "sympy__sympy-21930",
    "django__django-16263",
    "mwaskom__seaborn-3187",
    "pytest-dev__pytest-5840",
)
SELECTION_SHA256 = (
    "035a82e5938e14c56cf173d62f5e3caea2ff48c1c5f0c96c89e94712bf0a3d3a"
)
CACHEBLEND_DAMAGE_RATE = 9 / 167


def task_dir(output: Path, instance_id: str) -> Path:
    return output / "tasks" / instance_id


def _configure() -> None:
    orchestration.V38 = V40


def _selection_hash() -> str:
    value = json.dumps(
        {"salt": SELECTION_SALT, "tasks": list(TASKS)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode()).hexdigest()


def _selected_tasks() -> tuple[str, ...]:
    repositories = sorted(
        ELIGIBLE_BY_REPO,
        key=lambda repo: (
            hashlib.sha256(
                f"{SELECTION_SALT}:repo:{repo}".encode()
            ).hexdigest(),
            repo,
        ),
    )[:6]
    return tuple(
        min(
            ELIGIBLE_BY_REPO[repo],
            key=lambda instance_id: (
                hashlib.sha256(
                    (
                        f"{SELECTION_SALT}:task:{instance_id}"
                    ).encode()
                ).hexdigest(),
                instance_id,
            ),
        )
        for repo in repositories
    )


def _population_rows() -> list[dict[str, Any]]:
    value = read_json(POPULATION)
    if not isinstance(value, list) or len(value) != 500:
        raise AssertionError("SWE-bench Verified population changed")
    return value


def _prepare_inputs(output: Path) -> tuple[Path, Path, Path]:
    rows = _population_rows()
    indexed = {str(row["instance_id"]): row for row in rows}
    selected = [indexed[instance_id] for instance_id in TASKS]
    snapshot = output / "V43_FROZEN_SUBSET.json"
    dataset = output / "minisweagent_dataset"
    dataset_path = dataset / "test.jsonl"
    evaluation_registration = output / "V43_EVAL_REGISTRATION.json"
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
            "source": str(POPULATION),
            "source_sha256": sha256(POPULATION),
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
    path = output / "V43_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if _selection_hash() != SELECTION_SHA256:
        raise AssertionError("V43 task selection changed")
    if _selected_tasks() != TASKS:
        raise AssertionError("V43 frozen selection rule changed")
    if (
        read_json(V42)["status"]
        != "PASS_V42_HOST_RESIDENCY_INFRA_CANARY"
    ):
        raise AssertionError("V42 host-residency canary did not pass")
    dataset, snapshot, evaluation_registration = _prepare_inputs(output)
    rows = {row["instance_id"]: row for row in _population_rows()}
    selected = [
        {
            "instance_id": instance_id,
            "repo": rows[instance_id]["repo"],
        }
        for instance_id in TASKS
    ]
    value = {
        "registered_at_utc": utc_now(),
        "status": "REGISTERED_BEFORE_ANY_V43_TREATMENT",
        "experiment": "V43 new-to-V40 Verified development campaign",
        "motivation": (
            "The original 18-task V40 pool is exhausted by prior campaigns "
            "and canaries. V42 removed the paired-arm capacity deadlock. "
            "Expand to six repositories from the 500-task Verified population "
            "to test whether V40's grounded observation reuse separates from "
            "General on new-to-V40 tasks."
        ),
        "selection": {
            "population": str(POPULATION),
            "population_sha256": sha256(POPULATION),
            "eligibility": (
                "Official evaluator image was installed before selection; "
                "instance was not in the original V40 full18 pool."
            ),
            "rule": (
                "Rank eligible repositories by sha256(salt:repo:repo), take "
                "the first six, then within each repository minimize "
                "sha256(salt:task:instance_id)."
            ),
            "salt": SELECTION_SALT,
            "selection_sha256": SELECTION_SHA256,
            "tasks": selected,
            "gold_patch_used": False,
            "official_outcomes_used_for_selection": False,
            "replacement_on_failure": False,
            "outcome_exposure_class": (
                "NEW_TO_V40_DEVELOPMENT; PRIOR_PROJECT_OUTCOME_EXPOSURE_MAY_EXIST"
            ),
        },
        "protocol": {
            "arms": list(ARMS),
            "task_level_intention_to_treat": True,
            "all_children_registered_before_first_treatment": True,
            "continue_after_task_infrastructure_failure": True,
            "paired_accuracy_sources_reside_on_host": True,
            "formal_speed_uses_separate_device_resident_protocol": True,
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
            "tasks_with_online_branch_min": 4,
            "candidate_copy_requests_min": 6,
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
            "v42_sha256": sha256(V42),
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


def _environment(output: Path, instance_id: str) -> dict[str, str]:
    env = orchestration._environment(instance_id)
    env.update(
        {
            "IMPACTKV_DATASET_ROOT": str(output / "minisweagent_dataset"),
            "IMPACTKV_EVAL_REGISTRATION": str(
                output / "V43_EVAL_REGISTRATION.json"
            ),
            "IMPACTKV_EVAL_SNAPSHOT": str(
                output / "V43_FROZEN_SUBSET.json"
            ),
        }
    )
    return env


def _run_stage(
    output: Path, instance_id: str, stage: str
) -> dict[str, Any]:
    log_path = output / "orchestration_logs" / instance_id / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(orchestration.PYTHON),
        str(orchestration.RUNNER),
        stage,
        "--output",
        str(task_dir(output, instance_id)),
    ]
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=orchestration.PROJECT,
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
    write_json(
        output / "orchestration_status" / instance_id / f"{stage}.json",
        value,
    )
    return value


def preregister_children(output: Path) -> list[dict[str, Any]]:
    _configure()
    registration = register(output)
    rows = []
    for task in registration["selection"]["tasks"]:
        instance_id = task["instance_id"]
        rows.append(_run_stage(output, instance_id, "register"))
    write_json(output / "V43_CHILD_REGISTRATIONS.json", rows)
    if any(row["returncode"] != 0 for row in rows):
        raise RuntimeError("one or more V43 child registrations failed")
    return rows


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    rows = []
    for selected in registration["selection"]["tasks"]:
        instance_id = selected["instance_id"]
        child = task_dir(output, instance_id)
        runtime_path = child / "V25_RESULT.json"
        official_path = child / "V25_OFFICIAL_RESULT.json"
        if not runtime_path.exists() or not official_path.exists():
            stage_path = (
                output / "orchestration_status" / instance_id / "run.json"
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
        server = _jsonl(child / "run" / "SERVER_LEDGER.jsonl")
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
        "tasks_with_online_branch_min": branches >= 4,
        "candidate_copy_requests_min": copy_requests[V40] >= 6,
        "candidate_copied_tokens_strictly_below_general": (
            copied_tokens[V40] < copied_tokens[GENERAL]
        ),
        "candidate_assistant_tokens_selected": all(
            row["assistant_tokens_selected"] == 0 for row in complete
        ),
        "device_sources": sum(
            row["device_sources"] for row in complete
        )
        == 0,
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
            "PASS_V43_DEVELOPMENT"
            if all(gates.values())
            else "INCOMPLETE_V43"
            if len(complete) < len(TASKS)
            else "FAIL_V43_DEVELOPMENT"
        ),
        "registration_sha256": sha256(
            output / "V43_REGISTRATION.json"
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
                arm: _wilson(resolved[arm], len(complete))
                for arm in ARMS
            },
            "paired_candidate_only_vs_general_only": {
                V40: candidate_only,
                GENERAL: general_only,
            },
            "damage_count_given_dense_pass": damage,
            "damage_rate_given_dense_pass": damage_rate,
            "cacheblend_damage_rate_reference": CACHEBLEND_DAMAGE_RATE,
            "rescue_count_given_dense_fail": rescue,
            "copy_requests": copy_requests,
            "copied_tokens": copied_tokens,
            "device_sources": sum(
                row["device_sources"] for row in complete
            ),
            "host_sources": sum(
                row["host_sources"] for row in complete
            ),
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
            "Six new-to-V40 tasks selected without outcomes from the Verified "
            "population. Prior project exposure may exist, so a pass permits "
            "further development but is not held-out population or SOTA "
            "evidence."
        ),
    }
    write_json(output / "V43_RESULT.json", value)
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
    write_json(output / "V43_STAGE_STATUS.json", stages)
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
