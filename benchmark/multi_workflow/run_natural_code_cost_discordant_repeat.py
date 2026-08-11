#!/usr/bin/env python3
"""Repeat the fresh33 discordant tasks with reversed arm order.

This is an explicitly outcome-selected stability audit.  It asks whether the
five observed rescues and two observed damages reproduce when the policy arm
runs before Dense.  Its seven outcomes must not be appended to fresh33 as
independent confirmatory tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from benchmark.multi_workflow.run_natural_code_cost_expanded_accuracy_campaign import (
    _paired_summary,
    sha256,
)


PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path("/home/gfy/CodeMAS_Project")
ARTIFACTS = ROOT / "kvflow-artifacts"
PARENT = ARTIFACTS / "impactkv_natural_code_cost_agent_expanded24_20260808"
POPULATION = (
    ROOT
    / "sglang-kvflow/results/repo_level_datasets/"
    "swe_verified_500_instances.json"
)
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_natural_code_cost_discordant7_repeat_20260809"
)
BRIDGE_RUNNER = (
    PROJECT / "benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py"
)
MINI_PYTHON = Path("/home/gfy/.venvs/mini-swe-agent-v2.3.0/bin/python")
POLICY = "coding_natural_code_cost"
ARMS = (POLICY, "dense")
EXPECTED_RESCUES = 5
EXPECTED_DAMAGES = 2
TASKS = EXPECTED_RESCUES + EXPECTED_DAMAGES


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def paired_label(
    instance_id: str, dense_ids: set[str], policy_ids: set[str]
) -> str:
    dense = instance_id in dense_ids
    policy = instance_id in policy_ids
    if policy and not dense:
        return "rescue"
    if dense and not policy:
        return "damage"
    if dense:
        return "both_resolved"
    return "both_unresolved"


def stability_summary(
    *,
    original_rescues: set[str],
    original_damages: set[str],
    repeat_dense_ids: set[str],
    repeat_policy_ids: set[str],
) -> dict[str, Any]:
    instance_ids = sorted(original_rescues | original_damages)
    transitions = []
    transition_counts: dict[str, int] = {}
    for instance_id in instance_ids:
        original = "rescue" if instance_id in original_rescues else "damage"
        repeat = paired_label(instance_id, repeat_dense_ids, repeat_policy_ids)
        transition = f"{original}->{repeat}"
        transition_counts[transition] = transition_counts.get(transition, 0) + 1
        transitions.append(
            {
                "instance_id": instance_id,
                "original": original,
                "repeat": repeat,
                "repeat_dense_resolved": instance_id in repeat_dense_ids,
                "repeat_policy_resolved": instance_id in repeat_policy_ids,
            }
        )
    repeated = _paired_summary(
        instance_ids, repeat_dense_ids, repeat_policy_ids
    )
    return {
        "selection_warning": (
            "The seven tasks were selected because fresh33 was discordant. "
            "Repeat rates and McNemar p-values are descriptive stability "
            "diagnostics, not population-level confirmatory inference."
        ),
        "original": {
            "rescues": sorted(original_rescues),
            "damages": sorted(original_damages),
            "net": len(original_rescues) - len(original_damages),
        },
        "repeat": repeated,
        "stable_rescues": sorted(
            original_rescues
            & (repeat_policy_ids - repeat_dense_ids)
        ),
        "stable_damages": sorted(
            original_damages
            & (repeat_dense_ids - repeat_policy_ids)
        ),
        "transition_counts": dict(sorted(transition_counts.items())),
        "per_task": transitions,
    }


def prepare(output: Path) -> dict[str, Any]:
    registration_path = output / "CAMPAIGN_REGISTRATION.json"
    if registration_path.exists():
        return read_json(registration_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)

    parent_result_path = PARENT / "RESULT.json"
    parent = read_json(parent_result_path)
    aggregate = parent["fresh33_transparent_aggregate"]
    original_rescues = set(aggregate["rescues"])
    original_damages = set(aggregate["damages"])
    if len(original_rescues) != EXPECTED_RESCUES:
        raise AssertionError("fresh33 rescue count changed")
    if len(original_damages) != EXPECTED_DAMAGES:
        raise AssertionError("fresh33 damage count changed")
    if original_rescues & original_damages:
        raise AssertionError("parent discordant labels overlap")

    selected_ids = sorted(original_rescues | original_damages)
    population = read_json(POPULATION)
    by_id = {str(row["instance_id"]): row for row in population}
    missing = set(selected_ids) - set(by_id)
    if missing:
        raise KeyError(f"discordant tasks absent from population: {missing}")
    selected = [by_id[instance_id] for instance_id in selected_ids]

    output.mkdir(parents=True)
    snapshot = output / "FROZEN_DISCORDANT7.json"
    dataset = output / "dataset/test.jsonl"
    bridge_registration = output / "BRIDGE_DISCORDANT7_REGISTRATION.json"
    write_json(snapshot, selected)
    write_jsonl(dataset, selected)
    write_json(
        bridge_registration,
        {
            "schema_version": 1,
            "registration_id": "impactkv-natural-code-cost-discordant7-repeat",
            "registered_at_utc": utc_now(),
            "dataset": {
                "name": "princeton-nlp/SWE-bench_Verified",
                "split": "test",
            },
            "instances": [
                {"instance_id": instance_id} for instance_id in selected_ids
            ],
        },
    )

    sources = (
        PROJECT / "benchmark/multi_workflow/coding_reuse_policy.py",
        PROJECT / "benchmark/multi_workflow/bridge_reuse_litellm_model.py",
        PROJECT / "benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py",
        Path(__file__).resolve(),
    )
    value = {
        "status": "REGISTERED_BEFORE_DISCORDANT7_REPEAT_OUTCOMES",
        "registered_at_utc": utc_now(),
        "purpose": (
            "Post-outcome stability audit of every fresh33 discordant task, "
            "with reversed arm order"
        ),
        "selection": {
            "outcome_selected": True,
            "source": "fresh33 paired rescues and damages",
            "confirmatory_population_claim_allowed": False,
            "original_rescues": sorted(original_rescues),
            "original_damages": sorted(original_damages),
            "instances": [
                {
                    "instance_id": row["instance_id"],
                    "repo": row["repo"],
                    "difficulty": row.get("difficulty"),
                    "original_label": (
                        "rescue"
                        if row["instance_id"] in original_rescues
                        else "damage"
                    ),
                }
                for row in selected
            ],
        },
        "protocol": {
            "arms": list(ARMS),
            "arm_execution_order": list(ARMS),
            "order_reversed_relative_to_fresh33": True,
            "backend": "mini-SWE-agent rolling6 + SGLang",
            "model": "Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
            "temperature": 0,
            "server_random_seed": 709609581,
            "deterministic_inference": True,
            "step_limit": 32,
            "same_system_and_agent_templates": True,
            "official_metric": "SWE-bench resolved",
            "run_both_arms_regardless_of_first_arm_outcome": True,
            "prefetch": False,
        },
        "analysis": {
            "primary": "original-label to repeat-label transition matrix",
            "secondary": "repeat paired resolved difference",
            "inference_boundary": (
                "Do not append these seven repeats to fresh33 or treat the "
                "repeat McNemar result as population-level confirmation"
            ),
            "speed": "descriptive telemetry only; prior exact-prompt speed retained",
        },
        "inputs": {
            "parent_result": str(parent_result_path),
            "parent_result_sha256": sha256(parent_result_path),
            "population": str(POPULATION),
            "population_sha256": sha256(POPULATION),
            "snapshot": str(snapshot),
            "snapshot_sha256": sha256(snapshot),
            "dataset": str(dataset),
            "dataset_sha256": sha256(dataset),
            "bridge_registration": str(bridge_registration),
            "bridge_registration_sha256": sha256(bridge_registration),
            "source_sha256": {
                str(path.relative_to(PROJECT)): sha256(path) for path in sources
            },
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "old_preregistration_thresholds_modified": False,
            "paper_modified": False,
            "prefetch": False,
        },
    }
    write_json(registration_path, value)
    return value


def run_arm(output: Path, arm: str, port: int) -> None:
    if arm not in ARMS:
        raise ValueError(arm)
    prepare(output)
    env = os.environ.copy()
    env.update(
        {
            "IMPACTKV_DATASET_ROOT": str(output / "dataset"),
            "IMPACTKV_EVAL_SNAPSHOT": str(output / "FROZEN_DISCORDANT7.json"),
            "IMPACTKV_EVAL_REGISTRATION": str(
                output / "BRIDGE_DISCORDANT7_REGISTRATION.json"
            ),
            "IMPACTKV_AGENT_STEP_LIMIT": "32",
            "PYTHONPATH": (
                str(PROJECT)
                + (
                    os.pathsep + env["PYTHONPATH"]
                    if env.get("PYTHONPATH")
                    else ""
                )
            ),
        }
    )
    command = [
        str(MINI_PYTHON),
        str(BRIDGE_RUNNER),
        "--output",
        str(output / "online"),
        "run-arm",
        "--arm",
        arm,
        "--scope",
        "full",
        "--port",
        str(port),
        "--official",
    ]
    subprocess.run(command, cwd=PROJECT, env=env, check=True)


def _official(output: Path, arm: str) -> dict[str, Any]:
    value = read_json(output / f"online/{arm}/full_{TASKS}/OFFICIAL_RESULT.json")
    report = value.get("report")
    if report is None:
        raise ValueError(f"official report absent for {arm}")
    if report.get("total_instances") != TASKS:
        raise ValueError(f"official denominator changed for {arm}")
    return dict(report)


def summarize(output: Path) -> dict[str, Any]:
    registration = prepare(output)
    original_rescues = set(registration["selection"]["original_rescues"])
    original_damages = set(registration["selection"]["original_damages"])
    dense = _official(output, "dense")
    policy = _official(output, POLICY)
    stability = stability_summary(
        original_rescues=original_rescues,
        original_damages=original_damages,
        repeat_dense_ids=set(dense["resolved_ids"]),
        repeat_policy_ids=set(policy["resolved_ids"]),
    )
    runtime = read_json(
        output / f"online/{POLICY}/full_{TASKS}/RUNTIME_SUMMARY.json"
    )
    value = {
        "status": "COMPLETE",
        "classification": (
            "post-outcome discordant7 reversed-order stability audit"
        ),
        "independent_confirmation": False,
        "stability": stability,
        "official_evaluator": {"dense": dense, POLICY: policy},
        "physical_reuse": {
            key: runtime[key]
            for key in (
                "source_materialized_events",
                "target_copy_events",
                "copied_tokens",
                "rotated_k_tokens",
                "target_fallback_events",
                "host_source_copy_events",
            )
        }
        | {"prefetch": False},
        "speed_claim_added": False,
        "parent_fresh33_result": registration["inputs"]["parent_result"],
    }
    write_json(output / "RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run-arm")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--port", type=int, default=30000)
    sub.add_parser("summarize")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        value = prepare(output)
    elif args.command == "run-arm":
        run_arm(output, args.arm, args.port)
        value = {"status": "COMPLETE", "arm": args.arm}
    else:
        value = summarize(output)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
