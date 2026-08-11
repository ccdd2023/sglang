#!/usr/bin/env python3
"""Run a task-disjoint Fresh24 comparison of Dense, flat cold, and graph LCB.

The task cohort, selector code hashes, and TTFT lower-bound calibration are
frozen before any model outcome.  Official SWE-bench resolution is primary;
free-running latency remains descriptive and exact-prompt target replay is the
causal speed measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import (
    run_natural_code_cost_expanded_accuracy_campaign as selector,
)
from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
PATHS = RuntimePaths.from_project(PROJECT)
ARTIFACTS = PATHS.artifacts
POPULATION = PATHS.population
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_dependency_graph_fresh24_20260811"
CALIBRATION = (
    ARTIFACTS
    / "impactkv_dependency_graph_lcb_20260811/CALIBRATION.json"
)
BRIDGE_RUNNER = (
    PROJECT / "benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py"
)
MINI_PYTHON = PATHS.mini_python
SELECTION_SALT = "dependency-graph-cold-lcb-fresh24-20260811-v1"
TASKS = 24
REPO_CAP = 4
DIFFICULTY_QUOTAS = {
    "<15 min fix": 10,
    "15 min - 1 hour": 9,
    "1-4 hours": 5,
}
CURRENT_ARM = "coding_dependency_cold_cost"
NEW_ARM = "coding_dependency_graph_cold_lcb"
ARMS = ("dense", CURRENT_ARM, NEW_ARM)


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_fresh(population: list[dict[str, Any]], excluded: set[str]) -> list[dict[str, Any]]:
    saved = (
        selector.SELECTION_SALT,
        selector.TASKS,
        selector.REPO_CAP,
        selector.DIFFICULTY_QUOTAS,
    )
    try:
        selector.SELECTION_SALT = SELECTION_SALT
        selector.TASKS = TASKS
        selector.REPO_CAP = REPO_CAP
        selector.DIFFICULTY_QUOTAS = DIFFICULTY_QUOTAS
        return selector.select_cohort(population, excluded)
    finally:
        (
            selector.SELECTION_SALT,
            selector.TASKS,
            selector.REPO_CAP,
            selector.DIFFICULTY_QUOTAS,
        ) = saved


def prepare(output: Path) -> dict[str, Any]:
    registration_path = output / "CAMPAIGN_REGISTRATION.json"
    if registration_path.exists():
        return read_json(registration_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    calibration = read_json(CALIBRATION)
    if calibration["status"] != "FROZEN_BEFORE_NEW_ACCURACY_OUTCOMES":
        raise AssertionError("TTFT calibration is not frozen")
    population = read_json(POPULATION)
    excluded, exclusion_audit = selector.historical_exclusions()
    selected = select_fresh(population, excluded)
    identifiers = [str(row["instance_id"]) for row in selected]
    if excluded.intersection(identifiers):
        raise AssertionError("historical task leaked into Fresh24")
    if Counter(str(row["difficulty"]) for row in selected) != Counter(
        DIFFICULTY_QUOTAS
    ):
        raise AssertionError("difficulty quota changed")

    output.mkdir(parents=True)
    snapshot = output / "FROZEN_FRESH24.json"
    dataset = output / "dataset/test.jsonl"
    bridge_registration = output / "BRIDGE_FRESH24_REGISTRATION.json"
    write_json(snapshot, selected)
    write_jsonl(dataset, selected)
    write_json(
        bridge_registration,
        {
            "schema_version": 1,
            "registration_id": "impactkv-dependency-graph-fresh24",
            "registered_at_utc": utc_now(),
            "dataset": {
                "name": "princeton-nlp/SWE-bench_Verified",
                "split": "test",
            },
            "instances": [{"instance_id": value} for value in identifiers],
        },
    )
    source_paths = (
        PROJECT / "benchmark/multi_workflow/coding_reuse_policy.py",
        PROJECT / "benchmark/multi_workflow/bridge_reuse_litellm_model.py",
        PROJECT / "benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py",
        PROJECT / "benchmark/multi_workflow/run_swebench_with_limit_patch_capture.py",
        Path(__file__).resolve(),
    )
    value = {
        "status": "REGISTERED_BEFORE_FRESH24_MODEL_OUTCOMES",
        "registered_at_utc": utc_now(),
        "purpose": (
            "test whether visible one-hop dependency protection plus a "
            "conservative single-island TTFT gate improves the accuracy-speed "
            "direction over Dense and the preceding flat cold selector"
        ),
        "selection": {
            "salt": SELECTION_SALT,
            "outcome_used_for_selection": False,
            "tasks": TASKS,
            "repository_cap": REPO_CAP,
            "difficulty_quotas": DIFFICULTY_QUOTAS,
            "capacity_amendment_before_registration": (
                "The intended 9/9/6 split was infeasible before any model "
                "execution because the remaining 1-4 hour pool contains "
                "twelve Django tasks and one SymPy task while the frozen "
                "per-repository cap is four. The closest feasible split is "
                "10/9/5."
            ),
            "historical_excluded_tasks": len(excluded),
            "exclusion_audit": exclusion_audit,
            "instances": [
                {
                    "instance_id": str(row["instance_id"]),
                    "repo": str(row["repo"]),
                    "difficulty": row.get("difficulty"),
                }
                for row in selected
            ],
        },
        "protocol": {
            "arms": list(ARMS),
            "arm_execution_order": list(ARMS),
            "run_all_arms_regardless_of_intermediate_outcomes": True,
            "backend": "mini-SWE-agent rolling6 + SGLang",
            "model": "Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
            "temperature": 0,
            "step_limit": 32,
            "limit_time_patch_capture": True,
            "same_system_agent_and_prompt_templates": True,
            "official_metric": "SWE-bench resolved",
            "prefetch": False,
        },
        "new_policy": {
            "visible_graph": [
                "normalized repository path",
                "qualified class/function/method",
                "import alias",
                "direct call/name reference",
            ],
            "hot_action": "Dense recomputation",
            "cold_action": "lossy K/V copy with K RoPE rotation",
            "max_live_pool_islands": 3,
            "max_target_islands": 1,
            "cost_admission": "frozen lower_bound_cache_ready_saving_ms > 0",
            "calibration": str(CALIBRATION),
            "calibration_sha256": sha256(CALIBRATION),
            "hidden_repository_scan": False,
            "exact_only": False,
            "prefetch": False,
        },
        "directional_gates": {
            "new_resolved_at_least_dense": True,
            "new_resolved_at_least_current_flat_cold": True,
            "new_has_physical_copy": True,
            "new_has_zero_target_fallback": True,
            "statistical_significance_required": False,
        },
        "claim_limit": (
            "Fresh24 tests a directional point estimate. Causal TTFT is "
            "reported only from exact-prompt target replay."
        ),
        "inputs": {
            "population": str(POPULATION),
            "population_sha256": sha256(POPULATION),
            "snapshot_sha256": sha256(snapshot),
            "dataset_sha256": sha256(dataset),
            "bridge_registration_sha256": sha256(bridge_registration),
            "source_sha256": {
                str(path.relative_to(PROJECT)): sha256(path)
                for path in source_paths
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
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(key, None)
    env.update(
        {
            "IMPACTKV_DATASET_ROOT": str(output / "dataset"),
            "IMPACTKV_EVAL_SNAPSHOT": str(output / "FROZEN_FRESH24.json"),
            "IMPACTKV_EVAL_REGISTRATION": str(
                output / "BRIDGE_FRESH24_REGISTRATION.json"
            ),
            "IMPACTKV_AGENT_STEP_LIMIT": "32",
            "IMPACTKV_CAPTURE_LIMIT_PATCH": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "PYTHONPATH": str(PROJECT),
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


def official(output: Path, arm: str) -> dict[str, Any]:
    value = read_json(output / f"online/{arm}/full_{TASKS}/OFFICIAL_RESULT.json")
    if value.get("report") is None:
        raise ValueError(f"official report absent for {arm}")
    return dict(value["report"])


def runtime(output: Path, arm: str) -> dict[str, Any]:
    return read_json(output / f"online/{arm}/full_{TASKS}/RUNTIME_SUMMARY.json")


def paired_rows(
    identifiers: list[str], left: set[str], right: set[str]
) -> dict[str, Any]:
    return {
        "left_resolved": len(left),
        "right_resolved": len(right),
        "right_rescues": sorted(right - left),
        "right_damages": sorted(left - right),
        "both_resolved": sorted(left & right),
        "both_unresolved": sorted(set(identifiers) - left - right),
    }


def summarize(output: Path) -> dict[str, Any]:
    registration = prepare(output)
    identifiers = [
        row["instance_id"] for row in registration["selection"]["instances"]
    ]
    reports = {arm: official(output, arm) for arm in ARMS}
    runtimes = {arm: runtime(output, arm) for arm in ARMS}
    resolved = {
        arm: set(reports[arm].get("resolved_ids") or ()) for arm in ARMS
    }
    gates = {
        "new_resolved_at_least_dense": len(resolved[NEW_ARM])
        >= len(resolved["dense"]),
        "new_resolved_at_least_current_flat_cold": len(resolved[NEW_ARM])
        >= len(resolved[CURRENT_ARM]),
        "new_has_physical_copy": runtimes[NEW_ARM]["copied_tokens"] > 0,
        "new_has_zero_target_fallback": runtimes[NEW_ARM][
            "target_fallback_events"
        ]
        == 0,
    }
    result = {
        "status": "COMPLETE",
        "classification": "task-disjoint Fresh24 directional validation",
        "official": reports,
        "paired_new_vs_dense": paired_rows(
            identifiers, resolved["dense"], resolved[NEW_ARM]
        ),
        "paired_new_vs_current": paired_rows(
            identifiers, resolved[CURRENT_ARM], resolved[NEW_ARM]
        ),
        "runtime_descriptive_only": {
            arm: {
                "requests": runtimes[arm]["requests"],
                "median_ttft_ms": runtimes[arm]["median_ttft_ms"],
                "target_copy_events": runtimes[arm]["target_copy_events"],
                "copied_tokens": runtimes[arm]["copied_tokens"],
                "rotated_k_tokens": runtimes[arm]["rotated_k_tokens"],
                "target_fallback_events": runtimes[arm][
                    "target_fallback_events"
                ],
            }
            for arm in ARMS
        },
        "directional_gates": gates,
        "decision": (
            "DIRECTION_PASS_EXACT_PROMPT_SPEED"
            if all(gates.values())
            else "DIRECTION_FAIL_DIAGNOSE"
        ),
        "claim_limit": registration["claim_limit"],
    }
    write_json(output / "RESULT.json", result)
    return result


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
        value = {"arm": args.arm, "status": "COMPLETE"}
    else:
        value = summarize(output)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
