#!/usr/bin/env python3
"""Same-history official canary for dependency-cold lossy KV reuse.

The selector was fixed after an equal-budget physical-splice experiment found
that copying code with a later path/symbol consumer was much less stable than
copying code without one.  This canary freezes three task/request forks using
only prompt-visible structure and predicted TTFT benefit.  Dense and treatment
receive identical messages and fresh official workspaces through the fork;
only the target inference differs.

This is an exploratory three-task canary, not a population accuracy estimate.
No prefetch, exact-only reuse, paper edit, or old preregistration edit is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import run_same_history_fork_continuation as base


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
SOURCE_ROOT = (
    ARTIFACTS
    / "impactkv_natural_module_attention_20260808/initial20/dense/full_18"
)
SOURCE_SNAPSHOT = (
    ARTIFACTS / "impactkv_natural_module_attention_20260808/FROZEN_INITIAL20.json"
)
DEFAULT_OUTPUT = (
    ARTIFACTS
    / "impactkv_dependency_cold_same_history_canary_20260810/initial3"
)
POLICY_ARM = "coding_dependency_cold_cost"
ARMS = ("dense", POLICY_ARM)
ARM_PORTS = {"dense": 32440, POLICY_ARM: 32441}

# Frozen from prompt-only capacity replay.  For every structurally eligible
# task we first selected the read-only-prefix fork with maximum predicted
# cache-ready saving, then took the top three tasks.  Official outcomes and
# treatment continuations were not used.
FORKS = {
    "sympy__sympy-17630": {
        "request": 7,
        "planned_tokens": 2733,
        "predicted_saving_ms": 182.601509443666,
    },
    "pytest-dev__pytest-10356": {
        "request": 9,
        "planned_tokens": 2409,
        "predicted_saving_ms": 139.292,
    },
    "astropy__astropy-14182": {
        "request": 7,
        "planned_tokens": 1180,
        "predicted_saving_ms": 37.110186475919996,
    },
}
ELIGIBLE_CAPACITY = {
    "target_requests": 43,
    "tasks_with_target": 8,
    "read_only_prefix_target_requests": 32,
    "tasks_with_read_only_prefix_target": 6,
}


def trajectory_path(instance_id: str) -> Path:
    return SOURCE_ROOT / instance_id / f"{instance_id}.traj.json"


def registration_payload() -> dict[str, Any]:
    rows = {str(row["instance_id"]): row for row in base.read_json(SOURCE_SNAPSHOT)}
    tasks = []
    for instance_id, selection in FORKS.items():
        path = trajectory_path(instance_id)
        trajectory = base.read_json(path)
        request = int(selection["request"])
        prefixes = base.assistant_request_prefixes(trajectory["messages"])
        if request > len(prefixes):
            raise AssertionError(f"missing frozen fork q{request}: {instance_id}")
        frozen_prefix = prefixes[request - 1]
        actions = base.prefix_actions(trajectory["messages"], request)
        commands = [str(action.get("command") or "") for action in actions]
        if any(
            any(
                marker in command
                for marker in (
                    ">",
                    "apply_patch",
                    "git checkout",
                    "git restore",
                    "rm ",
                    "mv ",
                    "sed -i",
                    "tee ",
                )
            )
            for command in commands
        ):
            raise AssertionError(f"fork prefix is not read-only: {instance_id}")
        tasks.append(
            {
                "instance_id": instance_id,
                "fork_request_index": request,
                "planned_cold_copy_tokens": selection["planned_tokens"],
                "predicted_cache_ready_saving_ms": selection[
                    "predicted_saving_ms"
                ],
                "frozen_prefix_messages": len(frozen_prefix),
                "frozen_prefix_sha256": base.canonical_hash(frozen_prefix),
                "source_trajectory": str(path),
                "source_trajectory_sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
                "prefix_commands": commands,
                "image": base.get_swebench_docker_image_name(rows[instance_id]),
            }
        )
    return {
        "status": "REGISTERED_BEFORE_DEPENDENCY_COLD_FORK_OUTCOMES",
        "registered_at_utc": base.utc_now(),
        "purpose": (
            "test whether protecting prompt-visible dependency-hot code and "
            "lossy-copying only dependency-cold code preserves official "
            "accuracy while reducing cache-ready target TTFT"
        ),
        "selection": {
            "used_official_or_treatment_outcomes": False,
            "rule": (
                "maximum predicted positive saving among read-only-prefix "
                "dependency-cold forks per task, then top three tasks"
            ),
            "capacity": ELIGIBLE_CAPACITY,
            "tasks": tasks,
        },
        "protocol": {
            "arms": list(ARMS),
            "model": base.MODEL,
            "temperature": 0,
            "same_frozen_messages_at_fork": True,
            "fresh_official_container_per_task_and_arm": True,
            "prefix_workspace_commands_replayed": True,
            "prefix_workspace_must_remain_git_clean": True,
            "reuse_source_materialization": (
                "one discarded diagnostic token per earlier frozen request"
            ),
            "diagnostic_tokens_added_to_history": False,
            "prefix_replay_counted_as_online_latency": False,
            "online_prefetch": False,
            "ordinary_radix_prefix_reuse": False,
            "lossy_kv_reuse_required": True,
            "official_metric": "SWE-bench resolved",
        },
        "gates": {
            "all_target_prompt_hashes_equal": True,
            "all_frozen_prefix_hashes_equal": True,
            "all_prefix_workspaces_clean": True,
            "policy_physical_copy_events_min": len(FORKS),
            "no_dense_damage": True,
            "median_cache_ready_ttft_saving_positive": True,
        },
        "metric_scope": (
            "three-task exploratory causal canary; report task rows and do "
            "not generalize its resolved fraction to SWE-bench"
        ),
        "protected": {
            "paper_modified": False,
            "prefetch": False,
            "old_preregistration_thresholds_modified": False,
        },
    }


def prepare(output: Path) -> dict[str, Any]:
    registration_path = output / "CAMPAIGN_REGISTRATION.json"
    if registration_path.exists():
        return base.read_json(registration_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    registration = registration_payload()
    base.write_json(registration_path, registration)
    base.write_json(
        output / "BRIDGE_REGISTRATION.json",
        {
            "schema_version": 1,
            "registration_id": output.name,
            "registered_at_utc": registration["registered_at_utc"],
            "dataset": {
                "name": "princeton-nlp/SWE-bench_Verified",
                "split": "test",
            },
            "instances": [
                {"instance_id": instance_id} for instance_id in FORKS
            ],
        },
    )
    return registration


def configure_base() -> None:
    base.SOURCE_TRAJECTORIES = SOURCE_ROOT
    base.SOURCE_SNAPSHOT = SOURCE_SNAPSHOT
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.POLICY_ARM = POLICY_ARM
    base.ARMS = ARMS
    base.ARM_PORTS = ARM_PORTS
    base.FORK_TASKS = {
        instance_id: int(value["request"])
        for instance_id, value in FORKS.items()
    }
    base.CAUSAL_SCOPE = (
        "Prompt-structure-selected three-task same-history canary. It tests "
        "these causal forks but is not a population accuracy estimate."
    )
    # Base run/evaluation functions resolve this global at execution time.
    base.prepare = prepare


def summarize(output: Path) -> dict[str, Any]:
    value = base.summarize(output)
    damages = value["accuracy"]["damages"]
    value["frozen_gates"] = {
        "no_dense_damage": not damages,
        "median_cache_ready_ttft_saving_positive": value[
            "fork_target_latency"
        ]["median_ttft_saving_percent"]
        > 0,
        "physical_copy_for_every_fork": value["physical_reuse"][
            "fork_target_copy_events"
        ]
        >= len(FORKS),
    }
    dense_completed = set(
        value["official"]["dense"]["report"].get("completed_ids") or ()
    )
    policy_completed = set(
        value["official"][POLICY_ARM]["report"].get("completed_ids") or ()
    )
    paired_completed = sorted(dense_completed & policy_completed)
    value["posthoc_validity"] = {
        "paired_official_completed_tasks": paired_completed,
        "paired_official_accuracy_resolution": bool(paired_completed),
        "interpretation": (
            "accuracy comparison is resolved on at least one paired task"
            if paired_completed
            else (
                "no task produced non-empty officially evaluated patches in "
                "both arms; retain physical-copy and TTFT results, but do not "
                "interpret 0-vs-0 as accuracy preservation"
            )
        ),
        "added_after_opening_outcomes": True,
    }
    value["decision"] = (
        "EXPAND_DEPENDENCY_COLD_SELECTOR"
        if all(value["frozen_gates"].values()) and paired_completed
        else (
            "RETAIN_SPEED_RESULT_REPEAT_ON_ACCURACY_RESOLVED_COHORT"
            if all(value["frozen_gates"].values())
            else "DO_NOT_EXPAND_WITHOUT_DIAGNOSIS"
        )
    )
    base.write_json(output / "RESULT.json", value)
    return value


def main() -> None:
    configure_base()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run-arm")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--port", type=int)
    evaluate = sub.add_parser("evaluate-arm")
    evaluate.add_argument("--arm", choices=ARMS, required=True)
    sub.add_parser("summarize")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        value = prepare(output)
    elif args.command == "run-arm":
        value = base.run_arm(output, args.arm, args.port or ARM_PORTS[args.arm])
    elif args.command == "evaluate-arm":
        value = base.evaluate_arm(output, args.arm)
    else:
        value = summarize(output)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
