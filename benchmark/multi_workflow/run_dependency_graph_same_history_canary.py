#!/usr/bin/env python3
"""Six-task same-history causal canary for the dependency-graph LCB policy.

Every arm receives the same frozen real-agent messages and reconstructed clean
workspace through the fork request.  The only treatment difference is Dense,
the preceding flat dependency-cold reuse policy, or the new graph-cold
single-island LCB policy.  Forks were selected from visible structure and the
frozen speed model without consulting continuations or official outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import run_same_history_fork_continuation as base
from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = RuntimePaths.from_project(PROJECT).artifacts
SOURCE_ROOT = (
    ARTIFACTS
    / "impactkv_natural_code_cost_agent_expanded24_20260808/online/"
    "coding_natural_code_cost/full_24"
)
SOURCE_SNAPSHOT = (
    ARTIFACTS
    / "impactkv_natural_code_cost_agent_expanded24_20260808/"
    "FROZEN_EXPANDED24.json"
)
CALIBRATION = (
    ARTIFACTS
    / "impactkv_dependency_graph_lcb_20260811/CALIBRATION.json"
)
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_dependency_graph_same_history_canary6_20260811"
)
CURRENT_ARM = "coding_dependency_cold_cost"
NEW_ARM = "coding_dependency_graph_cold_lcb"
ARMS = ("dense", CURRENT_ARM, NEW_ARM)
ARM_PORTS = {"dense": 32740, CURRENT_ARM: 32741, NEW_ARM: 32742}

# The pre-outcome shadow planner found only six tasks where the graph selector
# and frozen lower bound jointly materialize a target.  The intended N=8 was
# therefore reduced to the complete eligible task set before any fork ran.
FORKS = {
    "sphinx-doc__sphinx-8459": {
        "request": 30,
        "planned_tokens": 2894,
        "lower_bound_saving_ms": 283.82989797966223,
    },
    "django__django-15629": {
        "request": 29,
        "planned_tokens": 2523,
        "lower_bound_saving_ms": 266.34374237117,
    },
    "sympy__sympy-16597": {
        "request": 11,
        "planned_tokens": 2535,
        "lower_bound_saving_ms": 181.16706481844807,
    },
    "sphinx-doc__sphinx-9591": {
        "request": 8,
        "planned_tokens": 1475,
        "lower_bound_saving_ms": 25.94757103530253,
    },
    "scikit-learn__scikit-learn-14053": {
        "request": 23,
        "planned_tokens": 1246,
        "lower_bound_saving_ms": 21.934286922589493,
    },
    "astropy__astropy-14539": {
        "request": 11,
        "planned_tokens": 816,
        "lower_bound_saving_ms": 7.357722187824152,
    },
}

# This task was rejected before its fork inference: replaying q6 creates the
# untracked diagnostic file ``diffbug.fits``.  The campaign's frozen invariant
# requires a clean workspace at the fork, so weakening the invariant or
# deleting hidden state after replay would both be invalid.  Keep the original
# registration intact and record an objective pre-treatment exclusion instead.
PRETREATMENT_EXCLUSIONS = {
    "astropy__astropy-14539": {
        "detected_arm": "dense",
        "detected_before_target_inference": True,
        "reason": (
            "frozen prefix q6 writes untracked /testbed/diffbug.fits, so the "
            "registered clean-workspace invariant cannot be satisfied"
        ),
        "source_command_request": 6,
        "outcome_consulted": False,
    }
}
EXECUTABLE_FORKS = {
    instance_id: value
    for instance_id, value in FORKS.items()
    if instance_id not in PRETREATMENT_EXCLUSIONS
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trajectory_path(instance_id: str) -> Path:
    return SOURCE_ROOT / instance_id / f"{instance_id}.traj.json"


def registration_payload() -> dict[str, Any]:
    rows = {
        str(row["instance_id"]): row for row in base.read_json(SOURCE_SNAPSHOT)
    }
    tasks = []
    for instance_id, selection in FORKS.items():
        path = trajectory_path(instance_id)
        trajectory = base.read_json(path)
        request = int(selection["request"])
        prefixes = base.assistant_request_prefixes(trajectory["messages"])
        frozen_prefix = prefixes[request - 1]
        tasks.append(
            {
                "instance_id": instance_id,
                "fork_request_index": request,
                "planned_new_island_tokens": selection["planned_tokens"],
                "frozen_lcb_saving_ms": selection["lower_bound_saving_ms"],
                "frozen_prefix_messages": len(frozen_prefix),
                "frozen_prefix_sha256": base.canonical_hash(frozen_prefix),
                "source_trajectory": str(path),
                "source_trajectory_sha256": sha256(path),
                "image": base.get_swebench_docker_image_name(rows[instance_id]),
            }
        )
    return {
        "status": "REGISTERED_BEFORE_SAME_HISTORY_CANARY_OUTCOMES",
        "registered_at_utc": base.utc_now(),
        "purpose": (
            "causally compare Dense, flat dependency-cold reuse, and visible "
            "graph-cold single-island LCB reuse from identical coding states"
        ),
        "selection": {
            "used_fork_continuations_or_official_outcomes": False,
            "rule": (
                "maximum frozen positive graph-LCB target per eligible task"
            ),
            "capacity_amendment_before_execution": (
                "The intended eight-task canary had only six structurally "
                "eligible tasks after graph-hot protection and the frozen "
                "lower-bound gate; all six are retained."
            ),
            "tasks": tasks,
        },
        "protocol": {
            "arms": list(ARMS),
            "arm_execution_order": list(ARMS),
            "model": base.MODEL,
            "temperature": 0,
            "same_frozen_messages_at_fork": True,
            "fresh_official_container_per_task_and_arm": True,
            "prefix_workspace_commands_replayed": True,
            "prefix_workspace_must_remain_git_clean": True,
            "reuse_source_materialization": (
                "one discarded diagnostic token per earlier real prompt"
            ),
            "diagnostic_tokens_added_to_history": False,
            "prefix_replay_counted_as_online_latency": False,
            "online_prefetch": False,
            "ordinary_radix_prefix_reuse": False,
            "official_metric": "SWE-bench resolved",
            "limit_time_patch_capture": True,
        },
        "new_policy": {
            "max_target_islands": 1,
            "lossy_kv_copy": True,
            "k_rope_rotation": True,
            "exact_only": False,
            "calibration": str(CALIBRATION),
            "calibration_sha256": sha256(CALIBRATION),
        },
        "directional_gates": {
            "new_resolved_at_least_dense": True,
            "new_resolved_at_least_current": True,
            "new_median_target_ttft_saving_positive": True,
            "new_physical_copy_each_fork": True,
            "new_target_fallbacks_zero": True,
            "statistical_significance_required": False,
        },
        "claim_limit": (
            "This is a six-task causal canary, not a population accuracy "
            "estimate or an external-baseline comparison."
        ),
        "protected": {
            "paper_modified": False,
            "prefetch": False,
            "old_preregistration_thresholds_modified": False,
        },
    }


def freeze_implementation(output: Path) -> dict[str, Any]:
    path = output / "IMPLEMENTATION_FREEZE.json"
    if path.exists():
        return base.read_json(path)
    sources = (
        base.PROJECT / "benchmark/multi_workflow/coding_reuse_policy.py",
        base.PROJECT / "benchmark/multi_workflow/bridge_reuse_litellm_model.py",
        base.PROJECT / "benchmark/multi_workflow/run_same_history_fork_continuation.py",
        Path(__file__).resolve(),
    )
    value = {
        "status": "FROZEN_BEFORE_SAME_HISTORY_CANARY_OUTCOMES",
        "frozen_at_utc": base.utc_now(),
        "source_sha256": {
            str(source.relative_to(base.PROJECT)): sha256(source)
            for source in sources
        },
        "calibration_sha256": sha256(CALIBRATION),
    }
    base.write_json(path, value)
    return value


def record_pretreatment_exclusions(output: Path) -> dict[str, Any]:
    path = output / "PRETREATMENT_EXCLUSIONS.json"
    if path.exists():
        return base.read_json(path)
    value = {
        "status": "RECORDED_AFTER_PREFIX_REPLAY_FAILURE_BEFORE_TARGET",
        "recorded_at_utc": base.utc_now(),
        "registered_tasks": len(FORKS),
        "executable_tasks": len(EXECUTABLE_FORKS),
        "exclusions": PRETREATMENT_EXCLUSIONS,
        "decision_rule": (
            "exclude any fork that violates the frozen clean-workspace "
            "invariant before target inference; do not replace it after "
            "observing arm outcomes"
        ),
        "registration_mutated": False,
        "replacement_selected": False,
    }
    base.write_json(path, value)
    return value


def prepare(output: Path) -> dict[str, Any]:
    path = output / "CAMPAIGN_REGISTRATION.json"
    if path.exists():
        freeze_implementation(output)
        record_pretreatment_exclusions(output)
        return base.read_json(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    value = registration_payload()
    base.write_json(path, value)
    base.write_json(
        output / "BRIDGE_REGISTRATION.json",
        {
            "schema_version": 1,
            "registration_id": output.name,
            "registered_at_utc": value["registered_at_utc"],
            "dataset": {
                "name": "princeton-nlp/SWE-bench_Verified",
                "split": "test",
            },
            "instances": [
                {"instance_id": instance_id} for instance_id in FORKS
            ],
        },
    )
    freeze_implementation(output)
    record_pretreatment_exclusions(output)
    return value


def configure_base() -> None:
    base.SOURCE_TRAJECTORIES = SOURCE_ROOT
    base.SOURCE_SNAPSHOT = SOURCE_SNAPSHOT
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.POLICY_ARM = NEW_ARM
    base.ARMS = ARMS
    base.ARM_PORTS = ARM_PORTS
    base.FORK_TASKS = {
        instance_id: int(value["request"])
        for instance_id, value in EXECUTABLE_FORKS.items()
    }
    base.CAUSAL_SCOPE = (
        "Five executable answer-blind same-history forks after one objective "
        "pre-treatment workspace exclusion; directional causal evidence only."
    )
    base.prepare = prepare


def target_treatment(
    output: Path, arm: str, instance_id: str
) -> dict[str, Any]:
    trajectory = base.read_json(
        output / arm / instance_id / f"{instance_id}.traj.json"
    )
    message = base.assistant_message_at_request(
        trajectory, int(FORKS[instance_id]["request"])
    )
    return dict(message.get("extra", {}).get("reuse_treatment") or {})


def summarize(output: Path) -> dict[str, Any]:
    # The shared summary treats NEW_ARM as the policy and already verifies
    # prompt hashes, workspaces, physical copy rows, and Dense-vs-new TTFT.
    value = base.summarize(output)
    current_official = value["official"][CURRENT_ARM]
    current_resolved = base.resolved_ids(current_official)
    new_resolved = base.resolved_ids(value["official"][NEW_ARM])
    dense_resolved = base.resolved_ids(value["official"]["dense"])

    current_server = base.load_jsonl(
        output / CURRENT_ARM / "SERVER_LEDGER.jsonl"
    )
    current_copies = [
        row for row in current_server if row.get("event") == "target_copied"
    ]
    current_fallbacks = [
        row for row in current_server if row.get("event") == "target_fallback"
    ]
    comparisons = []
    for row in value["per_task"]:
        instance_id = row["instance_id"]
        current = target_treatment(output, CURRENT_ARM, instance_id)
        current_ttft = 1000 * float(current["ttft_seconds"])
        comparisons.append(
            {
                "instance_id": instance_id,
                "current_resolved": instance_id in current_resolved,
                "new_resolved": instance_id in new_resolved,
                "current_target_islands": int(current["target_islands"]),
                "new_target_islands": row["policy_target_islands"],
                "current_target_ttft_ms": current_ttft,
                "new_target_ttft_ms": row["policy_target_ttft_ms"],
                "new_vs_current_ttft_saving_percent": (
                    100
                    * (current_ttft - row["policy_target_ttft_ms"])
                    / current_ttft
                ),
            }
        )

    value["accuracy_three_arm"] = {
        "tasks": len(EXECUTABLE_FORKS),
        "dense_resolved": len(dense_resolved),
        "current_resolved": len(current_resolved),
        "new_resolved": len(new_resolved),
        "new_vs_current_rescues": sorted(new_resolved - current_resolved),
        "new_vs_current_damages": sorted(current_resolved - new_resolved),
    }
    value["current_physical_reuse"] = {
        "target_copy_events": len(current_copies),
        "target_fallback_events": len(current_fallbacks),
        "copied_k_tokens": sum(
            int(row.get("copied_k_tokens", 0)) for row in current_copies
        ),
    }
    value["new_vs_current_target_latency"] = {
        "pairs": len(comparisons),
        "median_ttft_saving_percent": statistics.median(
            row["new_vs_current_ttft_saving_percent"] for row in comparisons
        ),
        "new_wins": sum(
            row["new_target_ttft_ms"] < row["current_target_ttft_ms"]
            for row in comparisons
        ),
    }
    value["three_arm_per_task"] = comparisons
    gates = {
        "new_resolved_at_least_dense": len(new_resolved)
        >= len(dense_resolved),
        "new_resolved_at_least_current": len(new_resolved)
        >= len(current_resolved),
        "new_median_target_ttft_saving_positive": value[
            "fork_target_latency"
        ]["median_ttft_saving_percent"]
        > 0,
        "new_physical_copy_each_fork": value["physical_reuse"][
            "fork_target_copy_events"
        ]
        >= len(EXECUTABLE_FORKS),
        "new_target_fallbacks_zero": value["physical_reuse"][
            "fork_target_fallback_events"
        ]
        == 0,
    }
    value["directional_gates"] = gates
    value["pretreatment_exclusions"] = record_pretreatment_exclusions(output)
    value["decision"] = (
        "DIRECTION_PASS_RUN_FRESH24"
        if all(gates.values())
        else "DIRECTION_FAIL_DIAGNOSE"
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
