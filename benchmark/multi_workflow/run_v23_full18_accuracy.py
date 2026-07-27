#!/usr/bin/env python3
"""Run and audit General versus V23 on the frozen official 18-task set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_bridge_reuse_agent_experiment import (
    run_arm,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    sha256,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v23_full18_accuracy_20260727"
V23_REPLAY = (
    ARTIFACTS
    / "impactkv_v23_target_prefix_replay_20260727"
    / "V23_REPLAY_RESULT.json"
)
PROJECT = Path(__file__).resolve().parents[2]
ARMS = ("general", "coding_post_mutation_target_prefix_v23")
PORTS = {"general": 32701, ARMS[1]: 32702}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def register(output: Path) -> dict[str, Any]:
    path = output / "V23_FULL18_REGISTRATION.json"
    if path.exists():
        return read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    value = {
        "registered_at_utc": utc_now(),
        "registered_before_gpu": True,
        "status": "REGISTERED_BEFORE_V23_FULL18_GPU",
        "experiment": "V23 official full18 task-accuracy gate",
        "motivation": (
            "V23 passed all frozen same-prompt mechanism, fidelity, and speed "
            "gates. The next independent layer must measure final repository "
            "correctness under free-running agent trajectories."
        ),
        "arms": list(ARMS),
        "order": list(ARMS),
        "ports": PORTS,
        "protocol": {
            "dataset": "frozen SWE-bench Verified 18-task set",
            "same_model_engine_agent_limits": True,
            "official_docker_evaluator": True,
            "free_running": True,
            "prefetch": False,
        },
        "frozen_gates": {
            "submitted_instances_each_arm": 18,
            "error_instances_each_arm_max": 0,
            "incomplete_instances_each_arm_max": 0,
            "candidate_resolved_not_below_general": True,
            "candidate_damage_not_above_rescue": True,
            "candidate_copy_events_min": 1,
            "candidate_fallbacks_max": 0,
        },
        "speed_interpretation": (
            "Exploratory only because free-running trajectories can issue "
            "different prompts and request counts; cache-ready paired speed "
            "was already gated by V23 replay."
        ),
        "inputs": {
            "v23_replay_path": str(V23_REPLAY),
            "v23_replay_sha256": sha256(V23_REPLAY),
            "source_sha256": {
                str(source.relative_to(PROJECT)): sha256(source)
                for source in (
                    PROJECT
                    / "benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py",
                    PROJECT / "benchmark/multi_workflow/coding_reuse_policy.py",
                    PROJECT
                    / "benchmark/multi_workflow/bridge_reuse_litellm_model.py",
                    PROJECT / "python/sglang/srt/mem_cache/kvcomm_exact.py",
                    Path(__file__),
                )
            },
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


def status_path(output: Path, arm: str) -> Path:
    return output / arm / "full_18" / "PIPELINE_STATUS.json"


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
    statuses = {arm: read_json(status_path(output, arm)) for arm in ARMS}
    official = {arm: statuses[arm]["official"] for arm in ARMS}
    resolved = {
        arm: set(official[arm]["resolved_ids"]) for arm in ARMS
    }
    general, candidate = ARMS
    both = sorted(resolved[general] & resolved[candidate])
    damage = sorted(resolved[general] - resolved[candidate])
    rescue = sorted(resolved[candidate] - resolved[general])
    neither = sorted(
        set(official[general]["submitted_ids"])
        - resolved[general]
        - resolved[candidate]
    )
    gates = registration["frozen_gates"]
    candidate_runtime = statuses[candidate]["runtime"]
    outcomes = {
        "complete": all(
            official[arm]["submitted_instances"]
            == gates["submitted_instances_each_arm"]
            and official[arm]["error_instances"]
            <= gates["error_instances_each_arm_max"]
            and not official[arm]["incomplete_ids"]
            for arm in ARMS
        ),
        "accuracy": len(resolved[candidate]) >= len(resolved[general]),
        "paired_transition": len(damage) <= len(rescue),
        "mechanism": (
            candidate_runtime["target_copy_events"]
            >= gates["candidate_copy_events_min"]
            and candidate_runtime["target_fallback_events"]
            <= gates["candidate_fallbacks_max"]
        ),
    }
    value = {
        "status": "V23_FULL18_COMPLETE",
        "completed_at_utc": utc_now(),
        "official_resolved": {
            arm: {
                "count": len(resolved[arm]),
                "ids": sorted(resolved[arm]),
            }
            for arm in ARMS
        },
        "paired_transition": {
            "both_resolve": both,
            "damage": damage,
            "rescue": rescue,
            "both_fail": neither,
        },
        "runtime": {
            arm: statuses[arm]["runtime"] for arm in ARMS
        },
        "gate_outcomes": outcomes,
        "promoted_to_full225": all(outcomes.values()),
        "prefetch": False,
    }
    write_json(output / "V23_FULL18_RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    run = sub.add_parser("run-arm")
    run.add_argument("--arm", choices=ARMS, required=True)
    sub.add_parser("summarize")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "register":
        value = register(output)
    elif args.command == "run-arm":
        register(output)
        value = run_arm(
            output=output,
            arm=args.arm,
            scope="full",
            port=PORTS[args.arm],
            instance_filter=None,
            official=True,
        )
    else:
        value = summarize(output)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
