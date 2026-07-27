#!/usr/bin/env python3
"""Reverse-order V28/General speed replication on common frozen prompts."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import run_frozen_trajectory_replay_v18 as replay
from benchmark.multi_workflow.run_bridge_reuse_agent_experiment import (
    load_jsonl,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v28b_order_replicated_speed_20260727"
)
PROJECT = Path(__file__).resolve().parents[2]
PRIOR = (
    ARTIFACTS
    / "impactkv_v28_payoff_guard_replay_20260727"
    / "V28_REPLAY_RESULT.json"
)
V28 = "coding_post_mutation_payoff_guard_v28"
GENERAL = "general"
ARMS = (V28, GENERAL)
PORTS = {V28: 33023, GENERAL: 33024}


def _configure() -> None:
    replay.ARMS = ARMS
    replay.PORTS = PORTS


def _gpu_processes() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        command = subprocess.run(
            ["ps", "-p", str(pid), "-o", "cmd="],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
        rows.append(
            {
                "pid": pid,
                "used_memory_mib": int(parts[1].split()[0]),
                "command": command,
            }
        )
    return rows


def register(output: Path) -> dict[str, Any]:
    path = output / "REPLAY_REGISTRATION.json"
    if path.exists():
        return replay.read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _configure()
    plans = {arm: replay.simulate_arm(arm) for arm in ARMS}
    identities = {
        arm: [
            (
                row["instance_id"],
                row["request_index"],
                row["prompt_hash"],
            )
            for row in plans[arm]
        ]
        for arm in ARMS
    }
    if identities[V28] != identities[GENERAL]:
        raise AssertionError("V28B prompt identities differ")
    value = {
        "registration_id": output.name,
        "registered_at_utc": replay.utc_now(),
        "status": "REGISTERED_BEFORE_ANY_V28B_GPU_RUN",
        "experiment": "V28B reverse-order common-prompt speed replication",
        "motivation": (
            "V28A passed every mechanism and fidelity gate but failed speed. "
            "It ran after General while an unrelated GPU process appeared, "
            "and its original medians used 30 V28 targets versus 39 General "
            "targets. Replicate speed in reverse order and compare only the "
            "same 30 V28-target prompt keys. The V28A failure remains final "
            "regardless of this replication."
        ),
        "replication_of": str(PRIOR),
        "replication_of_sha256": replay.sha256(PRIOR),
        "arm_order": list(ARMS),
        "instances": list(replay.INSTANCE_IDS),
        "gpu_processes_at_registration": _gpu_processes(),
        "protocol": {
            "same_frozen_prompt_ids": True,
            "diagnostic_new_tokens": 1,
            "diagnostic_output_never_enters_future_prompt": True,
            "common_key_definition": "V28 registered-target prompt keys",
            "cache_ready_metric": "paired TTFT on common keys",
            "n4_including_build_formula": (
                "paired target TTFT + that arm's source materialize_ms / 4"
            ),
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "frozen_gates": {
            "prompt_hashes_identical": True,
            "common_target_keys_min": 1,
            "target_fallbacks_each_arm": 0,
            "v28_common_median_ttft_not_above_general": True,
            "v28_median_paired_ttft_ratio_not_above_one": True,
            "v28_common_n4_not_above_general": True,
            "do_not_override_v28a_failure": True,
            "do_not_claim_task_accuracy": True,
        },
        "offline_plans": plans,
        "inputs": {
            "source_sha256": {
                str(source.relative_to(PROJECT)): replay.sha256(source)
                for source in (
                    PROJECT
                    / "benchmark/multi_workflow/coding_reuse_policy.py",
                    PROJECT
                    / "benchmark/multi_workflow/bridge_reuse_litellm_model.py",
                    PROJECT
                    / "benchmark/multi_workflow/run_bridge_reuse_agent_experiment.py",
                    PROJECT
                    / "python/sglang/srt/mem_cache/kvcomm_exact.py",
                    Path(__file__),
                )
            }
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
    }
    replay.write_json(path, value)
    return value


def _builds(output: Path, arm: str) -> dict[str, float]:
    return {
        str(row["source_id"]): float(row["materialize_ms"])
        for row in load_jsonl(output / arm / "SERVER_LEDGER.jsonl")
        if row.get("event")
        in ("source_materialized", "source_materialized_host")
    }


def summarize(output: Path) -> dict[str, Any]:
    register(output)
    rows = {
        arm: replay.read_json(
            output / arm / "REPLAY_RESULTS.json"
        )["rows"]
        for arm in ARMS
    }
    indexed = {
        arm: {
            (row["instance_id"], int(row["request_index"])): row
            for row in rows[arm]
        }
        for arm in ARMS
    }
    common = [
        key for key, row in indexed[V28].items() if row["target_registered"]
    ]
    prompt_identity = all(
        indexed[V28][key]["prompt_hash"]
        == indexed[GENERAL][key]["prompt_hash"]
        for key in indexed[V28]
    )
    ttft = {
        arm: [float(indexed[arm][key]["ttft_ms"]) for key in common]
        for arm in ARMS
    }
    pair_ratios = [
        indexed[V28][key]["ttft_ms"] / indexed[GENERAL][key]["ttft_ms"]
        for key in common
    ]
    builds = {arm: _builds(output, arm) for arm in ARMS}
    n4 = {
        arm: [
            float(indexed[arm][key]["ttft_ms"])
            + builds[arm].get(
                str(indexed[arm][key]["target_source_id"]), 0.0
            )
            / 4
            for key in common
        ]
        for arm in ARMS
    }
    ledgers = {
        arm: load_jsonl(output / arm / "SERVER_LEDGER.jsonl")
        for arm in ARMS
    }
    fallbacks = {
        arm: sum(row.get("event") == "target_fallback" for row in ledger)
        for arm, ledger in ledgers.items()
    }
    medians = {
        arm: statistics.median(ttft[arm]) for arm in ARMS
    }
    n4_medians = {
        arm: statistics.median(n4[arm]) for arm in ARMS
    }
    gates = {
        "prompt_hashes_identical": prompt_identity,
        "common_target_keys_min": len(common) >= 1,
        "target_fallbacks_each_arm": all(
            fallbacks[arm] == 0 for arm in ARMS
        ),
        "v28_common_median_ttft_not_above_general": (
            medians[V28] <= medians[GENERAL]
        ),
        "v28_median_paired_ttft_ratio_not_above_one": (
            statistics.median(pair_ratios) <= 1
        ),
        "v28_common_n4_not_above_general": (
            n4_medians[V28] <= n4_medians[GENERAL]
        ),
        "do_not_override_v28a_failure": True,
        "do_not_claim_task_accuracy": True,
    }
    value = {
        "completed_at_utc": replay.utc_now(),
        "status": (
            "PASS_V28B_ORDER_REPLICATION"
            if all(gates.values())
            else "FAIL_V28B_ORDER_REPLICATION"
        ),
        "common_target_keys": len(common),
        "median_common_ttft_ms": medians,
        "median_common_n4_ms": n4_medians,
        "median_paired_ttft_ratio_v28_over_general": (
            statistics.median(pair_ratios)
        ),
        "common_median_ttft_ratio_v28_over_general": (
            medians[V28] / medians[GENERAL]
        ),
        "v28_faster_common_requests": sum(
            ratio < 1 for ratio in pair_ratios
        ),
        "target_fallbacks": fallbacks,
        "gpu_processes_at_completion": _gpu_processes(),
        "gate_outcomes": gates,
        "decision": (
            "Treat V28A speed failure as order/co-tenancy sensitive and move "
            "only to a counterbalanced speed estimate; still no accuracy run."
            if all(gates.values())
            else "Reject V28 speed selector before official accuracy."
        ),
    }
    replay.write_json(output / "V28B_RESULT.json", value)
    return value


def run(output: Path) -> dict[str, Any]:
    register(output)
    _configure()
    for arm in ARMS:
        replay.run_arm(output, arm, PORTS[arm])
    return summarize(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("register", "run", "summarize"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "register":
        value = register(args.output)
    elif args.command == "run":
        value = run(args.output)
    else:
        value = summarize(args.output)
    print(
        {
            "status": value.get("status"),
            "output": str(args.output),
            "gate_outcomes": value.get("gate_outcomes"),
        }
    )


if __name__ == "__main__":
    main()
