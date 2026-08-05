#!/usr/bin/env python3
"""Replay fresh Dense trajectories with identical prompts for V40 TTFT.

The free-running M55 campaign is the functional-accuracy experiment.  Its
three arms can take different actions and therefore do not preserve request
prompt identity.  M56 removes that confound: it rebuilds every request from
the frozen Dense trajectories, feeds the exact same prompt token IDs to Dense
and V40, and never appends the diagnostic one-token output to later requests.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark.multi_workflow import run_frozen_trajectory_replay_v18 as replay
from benchmark.multi_workflow.run_m55_v40_task_disjoint_campaign import TASKS


ROOT = Path("/home/gfy/CodeMAS_Project")
FRESH_ROOT = ROOT / "kvflow-artifacts/impactkv_m55_v40_task_disjoint_20260805"
DEFAULT_OUTPUT = ROOT / "kvflow-artifacts/impactkv_m56_v40_same_prompt_20260805/fresh13"
V40 = "coding_grounded_observation_island_v40"
ARMS = ("dense", V40)
PORTS = {"dense": 33200, V40: 33201}


def _trajectory_paths(root: Path) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in sorted((root / "tasks").glob("*/dense/**/*.traj.json")):
        value = replay.read_json(path)
        instance_id = str(value.get("instance_id") or "")
        if instance_id in TASKS and instance_id not in selected:
            selected[instance_id] = path
    return selected


def _task_campaign_complete(value: Mapping[str, Any]) -> bool:
    return int(value.get("aggregate", {}).get("complete_tasks", 0)) == len(TASKS)


def _recorded_prompt_tokens(trajectory: Mapping[str, Any]) -> list[int]:
    values = []
    for message in trajectory["messages"]:
        if message.get("role") != "assistant":
            continue
        treatment = message.get("extra", {}).get("reuse_treatment", {})
        if "prompt_tokens" not in treatment:
            raise ValueError("Dense trajectory lacks prompt-token audit metadata")
        values.append(int(treatment["prompt_tokens"]))
    return values


def _simulate(path: Path, arm: str) -> list[dict[str, Any]]:
    trajectory = replay.read_json(path)
    instance_id = str(trajectory["instance_id"])
    planner = replay.make_planner(
        arm=arm,
        manifest_path=None,
        client_ledger_path=None,
        instance_nonce=f"m56-register-{arm}",
    )
    replay.reset_planner_session(planner, instance_id=instance_id)
    rows = []
    for request_index, prefix in enumerate(
        replay.assistant_request_prefixes(trajectory["messages"]), start=1
    ):
        planned = replay.plan_request(planner, prefix)
        rows.append(
            {
                "instance_id": instance_id,
                "request_index": request_index,
                "prompt_hash": planned["prompt_hash"],
                "prompt_tokens": planned["prompt_tokens"],
                "target_registered": planned["target"] is not None,
                "target_length": (
                    int(planned["target"]["length"])
                    if planned["target"] is not None
                    else 0
                ),
                "source_registered": planned["source"] is not None,
            }
        )
    return rows


def prepare(output: Path, trajectory_root: Path) -> dict[str, Any]:
    registration_path = output / "REPLAY_REGISTRATION.json"
    if registration_path.exists():
        return replay.read_json(registration_path)
    task_registration = trajectory_root / "M55_TASK_REGISTRATION.json"
    task_result = trajectory_root / "M55_TASK_RESULT.json"
    if not task_registration.exists() or not task_result.exists():
        raise FileNotFoundError("complete M55 task registration/result required")
    task_value = replay.read_json(task_result)
    if not _task_campaign_complete(task_value):
        raise ValueError("M55 task campaign is not complete")
    trajectories = _trajectory_paths(trajectory_root)
    if set(trajectories) != set(TASKS):
        missing = sorted(set(TASKS) - set(trajectories))
        raise ValueError(f"fresh Dense trajectories missing: {missing}")

    plans = {
        arm: {
            instance_id: _simulate(trajectories[instance_id], arm)
            for instance_id in TASKS
        }
        for arm in ARMS
    }
    identities = {
        arm: [
            (
                row["instance_id"],
                row["request_index"],
                row["prompt_hash"],
            )
            for instance_id in TASKS
            for row in plans[arm][instance_id]
        ]
        for arm in ARMS
    }
    if identities["dense"] != identities[V40]:
        raise AssertionError("Dense and V40 simulated prompt IDs differ")
    for instance_id, path in trajectories.items():
        recorded = _recorded_prompt_tokens(replay.read_json(path))
        simulated = [row["prompt_tokens"] for row in plans["dense"][instance_id]]
        if recorded != simulated:
            raise AssertionError(
                f"{instance_id}: reconstructed Dense prompt lengths differ"
            )
    target_rows = [
        row
        for instance_id in TASKS
        for row in plans[V40][instance_id]
        if row["target_registered"]
    ]
    target_tasks = len({row["instance_id"] for row in target_rows})
    value = {
        "status": "REGISTERED_BEFORE_SAME_PROMPT_GPU_TREATMENT",
        "purpose": (
            "measure V40 cache-ready and build-amortized TTFT on prompt-ID "
            "identical frozen Dense coding-agent requests"
        ),
        "arms": list(ARMS),
        "arm_order": list(ARMS),
        "instances": list(TASKS),
        "requests": len(identities["dense"]),
        "v40_target_requests": len(target_rows),
        "v40_target_tasks": target_tasks,
        "trajectory_sha256": {
            instance_id: replay.sha256(path)
            for instance_id, path in trajectories.items()
        },
        "task_registration": str(task_registration),
        "task_registration_sha256": replay.sha256(task_registration),
        "task_result": str(task_result),
        "task_result_sha256": replay.sha256(task_result),
        "protocol": {
            "model": replay.MODEL,
            "temperature": 0,
            "diagnostic_new_tokens": 1,
            "diagnostic_output_never_enters_future_prompt": True,
            "same_prompt_token_ids": True,
            "reconstructed_dense_prompt_lengths_match_source_trajectory": True,
            "ordinary_radix_reuse": False,
            "prefetch": False,
            "v40_copy_cap": 4096,
            "source_build_amortization_n": 4,
        },
        "frozen_gates": {
            "minimum_v40_target_requests": 20,
            "minimum_v40_target_tasks": 8,
            "prompt_hashes_identical": True,
            "physical_copy_fraction": 1.0,
            "fallback_events_max": 0,
            "median_cache_ready_speedup_min": 1.05,
            "p95_cache_ready_speedup_min": 1.00,
            "per_request_ttft_win_fraction_min": 0.60,
            "target_first_token_agreement_min": 0.90,
        },
        "plans": plans,
        "interpretation_limits": [
            "one-token fixed-prompt replay is speed/fidelity evidence, not task accuracy",
            "functional accuracy comes only from the independent M55 official evaluation",
            "source materialization is reported separately and amortized at N=4",
        ],
        "protected": {
            "paper_modified": False,
            "old_dirty_checkout_modified": False,
            "old_preregistration_thresholds_modified": False,
        },
    }
    replay.write_json(registration_path, value)
    return value


def _close_pending(planner: Any) -> None:
    if planner._pending_source is not None:
        planner._atomic_sidecar_update(
            release_source_ids=[str(planner._pending_source["source_id"])]
        )
        planner._pending_source = None


def _preserve_partial_run(run_dir: Path) -> Path | None:
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return None
    failed = run_dir.with_name(run_dir.name + ".failed_run_1")
    if failed.exists():
        raise FileExistsError(
            f"one preserved partial run already exists: {failed}"
        )
    run_dir.rename(failed)
    return failed


def run_arm(output: Path, trajectory_root: Path, arm: str, port: int) -> dict[str, Any]:
    prepare(output, trajectory_root)
    run_dir = output / arm
    result_path = run_dir / "REPLAY_RESULTS.json"
    if result_path.exists():
        return replay.read_json(result_path)
    _preserve_partial_run(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = replay.init_manifest(run_dir, arm)
    planner = replay.make_planner(
        arm=arm,
        manifest_path=manifest if arm != "dense" else None,
        client_ledger_path=run_dir / "PLANNER_LEDGER.jsonl",
        instance_nonce=f"m56-runtime-{arm}",
    )
    process, log = replay.launch_server(
        run_dir=run_dir,
        arm=arm,
        manifest=manifest,
        port=port,
    )
    trajectories = _trajectory_paths(trajectory_root)
    rows: list[dict[str, Any]] = []
    try:
        base_url = f"http://127.0.0.1:{port}"
        first = replay.read_json(trajectories[TASKS[0]])["messages"]
        warm_prefix = replay.assistant_request_prefixes(first)[0]
        warm_planner = replay.make_planner(
            arm="dense",
            manifest_path=None,
            client_ledger_path=None,
            instance_nonce=f"m56-warm-{arm}",
        )
        warm = replay.plan_request(warm_planner, warm_prefix)
        replay.generate_one(
            base_url=base_url,
            input_ids=warm["prompt_ids"][:128],
            key=f"m56-warm-{arm}",
        )
        for instance_id in TASKS:
            replay.reset_planner_session(planner, instance_id=instance_id)
            trajectory = replay.read_json(trajectories[instance_id])
            for request_index, prefix in enumerate(
                replay.assistant_request_prefixes(trajectory["messages"]),
                start=1,
            ):
                planned = replay.plan_request(planner, prefix)
                target = planned["target"]
                generated = replay.generate_one(
                    base_url=base_url,
                    input_ids=planned["prompt_ids"],
                    key=(
                        str(target["case_id"])
                        if target is not None
                        else f"m56-{arm}-{instance_id}-q{request_index}"
                    ),
                )
                row = {
                    "arm": arm,
                    "instance_id": instance_id,
                    "request_index": request_index,
                    "prompt_hash": planned["prompt_hash"],
                    "prompt_tokens": planned["prompt_tokens"],
                    "target_registered": target is not None,
                    "target_source_id": (
                        str(target["source_id"]) if target is not None else None
                    ),
                    "target_length": (
                        int(target["length"]) if target is not None else 0
                    ),
                    "source_registered": planned["source"] is not None,
                    **generated,
                }
                rows.append(row)
                with (run_dir / "REPLAY_RESULTS.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
            if arm != "dense":
                _close_pending(planner)
    finally:
        replay.stop_server(process, log)
    value = {"arm": arm, "requests": len(rows), "rows": rows}
    replay.write_json(result_path, value)
    return value


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("empty percentile input")
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)]


def _key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["instance_id"]), int(row["request_index"])


def summarize(output: Path) -> dict[str, Any]:
    registration = replay.read_json(output / "REPLAY_REGISTRATION.json")
    rows = {
        arm: replay.read_json(output / arm / "REPLAY_RESULTS.json")["rows"]
        for arm in ARMS
    }
    indexed = {arm: {_key(row): row for row in value} for arm, value in rows.items()}
    keys = sorted(indexed["dense"])
    prompt_identity = (
        set(indexed[V40]) == set(keys)
        and all(
            indexed[V40][key]["prompt_hash"]
            == indexed["dense"][key]["prompt_hash"]
            for key in keys
        )
    )
    target_keys = [key for key in keys if indexed[V40][key]["target_registered"]]
    dense_ttft = [float(indexed["dense"][key]["ttft_ms"]) for key in target_keys]
    v40_ttft = [float(indexed[V40][key]["ttft_ms"]) for key in target_keys]
    wins = [left < right for left, right in zip(v40_ttft, dense_ttft, strict=True)]
    dense_tokens = [replay.token_id(indexed["dense"][key]) for key in target_keys]
    v40_tokens = [replay.token_id(indexed[V40][key]) for key in target_keys]
    comparable = [
        (left, right)
        for left, right in zip(dense_tokens, v40_tokens, strict=True)
        if left is not None and right is not None
    ]
    ledgers = {
        arm: replay.load_jsonl(output / arm / "SERVER_LEDGER.jsonl")
        for arm in ARMS
    }
    copies = [row for row in ledgers[V40] if row.get("event") == "target_copied"]
    fallbacks = [row for row in ledgers[V40] if row.get("event") == "target_fallback"]
    builds = {
        str(row["source_id"]): float(row["materialize_ms"])
        for row in ledgers[V40]
        if row.get("event") in ("source_materialized", "source_materialized_host")
    }
    n4 = [
        float(indexed[V40][key]["ttft_ms"])
        + builds[str(indexed[V40][key]["target_source_id"])] / 4
        for key in target_keys
        if str(indexed[V40][key]["target_source_id"]) in builds
    ]
    target_tasks = len({key[0] for key in target_keys})
    median_speedup = statistics.median(dense_ttft) / statistics.median(v40_ttft)
    p95_speedup = _percentile(dense_ttft, 0.95) / _percentile(v40_ttft, 0.95)
    agreement = (
        sum(left == right for left, right in comparable) / len(comparable)
        if comparable
        else 0.0
    )
    gate = registration["frozen_gates"]
    outcomes = {
        "minimum_v40_target_requests": len(target_keys) >= gate["minimum_v40_target_requests"],
        "minimum_v40_target_tasks": target_tasks >= gate["minimum_v40_target_tasks"],
        "prompt_hashes_identical": prompt_identity,
        "physical_copy_fraction": (
            len(copies) / len(target_keys) if target_keys else 0.0
        ) >= gate["physical_copy_fraction"],
        "fallback_events": len(fallbacks) <= gate["fallback_events_max"],
        "median_cache_ready_speedup": median_speedup >= gate["median_cache_ready_speedup_min"],
        "p95_cache_ready_speedup": p95_speedup >= gate["p95_cache_ready_speedup_min"],
        "per_request_ttft_win_fraction": (
            sum(wins) / len(wins) if wins else 0.0
        ) >= gate["per_request_ttft_win_fraction_min"],
        "target_first_token_agreement": agreement >= gate["target_first_token_agreement_min"],
    }
    value = {
        "status": "COMPLETE" if prompt_identity and len(copies) == len(target_keys) and not fallbacks else "MECHANISM_FAILURE",
        "decision": "SUPPORTED_SPEED_REPLAY" if all(outcomes.values()) else "NOT_SUPPORTED",
        "scope": "same-prompt one-token TTFT/fidelity replay; not task accuracy",
        "requests": len(keys),
        "v40_target_requests": len(target_keys),
        "v40_target_tasks": target_tasks,
        "prompt_hashes_identical": prompt_identity,
        "physical_reuse": {
            "copy_events": len(copies),
            "fallback_events": len(fallbacks),
            "copied_tokens": sum(int(row.get("copied_k_tokens", 0)) for row in copies),
        },
        "latency": {
            "dense_median_target_ttft_ms": statistics.median(dense_ttft),
            "v40_median_cache_ready_target_ttft_ms": statistics.median(v40_ttft),
            "median_cache_ready_speedup": median_speedup,
            "p95_cache_ready_speedup": p95_speedup,
            "per_request_ttft_win_fraction": sum(wins) / len(wins),
            "v40_median_n4_including_build_ms": statistics.median(n4) if n4 else None,
        },
        "fidelity": {
            "comparable_target_requests": len(comparable),
            "target_first_token_agreement": agreement,
        },
        "frozen_gate_outcomes": outcomes,
    }
    replay.write_json(output / "RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trajectory-root", type=Path, default=FRESH_ROOT)
    parser.add_argument("--stage", choices=("prepare", "run", "summarize"), default="run")
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.stage == "prepare":
        value = prepare(args.output, args.trajectory_root)
    elif args.stage == "summarize":
        value = summarize(args.output)
    else:
        prepare(args.output, args.trajectory_root)
        arms = (args.arm,) if args.arm else ARMS
        for arm in arms:
            run_arm(args.output, args.trajectory_root, arm, PORTS[arm])
        value = (
            summarize(args.output)
            if all((args.output / arm / "REPLAY_RESULTS.json").exists() for arm in ARMS)
            else {"status": "PARTIAL", "completed_arms": list(arms)}
        )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
