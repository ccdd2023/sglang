#!/usr/bin/env python3
"""Pre-registered frozen-prompt V28 mechanism, speed, and fidelity replay."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import run_frozen_trajectory_replay_v18 as replay
from benchmark.multi_workflow.run_bridge_reuse_agent_experiment import (
    load_jsonl,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v28_payoff_guard_replay_20260727"
PROJECT = Path(__file__).resolve().parents[2]
MOTIVATION = (
    ARTIFACTS
    / "impactkv_v27d_dense_pass_audited_completion_20260727"
    / "V28_MOTIVATION_AUDIT.json"
)
DENSE = "dense"
GENERAL = "general"
V28 = "coding_post_mutation_payoff_guard_v28"
ARMS = (DENSE, GENERAL, V28)
PORTS = {DENSE: 33020, GENERAL: 33021, V28: 33022}


def _configure_replay_module() -> None:
    replay.ARMS = ARMS
    replay.PORTS = PORTS


def register(output: Path) -> dict[str, Any]:
    path = output / "REPLAY_REGISTRATION.json"
    if path.exists():
        return replay.read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _configure_replay_module()
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
    if any(identities[arm] != identities[DENSE] for arm in ARMS[1:]):
        raise AssertionError("offline V28 prompt identities differ")
    decisions = Counter(row["decision"]["mode"] for row in plans[V28])
    value = {
        "registration_id": output.name,
        "registered_at_utc": replay.utc_now(),
        "status": "REGISTERED_BEFORE_ANY_V28_GPU_RUN",
        "experiment": "V28 frozen-prompt payoff-guard replay",
        "motivation": (
            "V27D found equal 3/5 accuracy for V23, General, and concurrent "
            "Dense, while V23 mean branch time was 8.5% above General. Test "
            "whether an online payoff guard keeps code protection only when "
            "its protected span can be repaid, falls back to General plus an "
            "exact target prefix otherwise, and abstains when the branch is "
            "too late."
        ),
        "arms": list(ARMS),
        "arm_order": list(ARMS),
        "instances": list(replay.INSTANCE_IDS),
        "protocol": {
            "same_frozen_prompt_ids": True,
            "diagnostic_new_tokens": 1,
            "diagnostic_output_never_enters_future_prompt": True,
            "return_logprob": True,
            "top_logprobs_num": 20,
            "copy_cap_tokens": 4096,
            "payoff_ratio_threshold": 0.60,
            "exact_prefix_credit_tokens": 640,
            "minimum_future_target_upper_bound": 4,
            "cache_ready_metric": "target TTFT after source exists",
            "n4_including_build_formula": (
                "target_ttft_ms + source_materialize_ms / 4"
            ),
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "offline_plan_audit": {
            "requests": len(plans[V28]),
            "decision_mode_counts": dict(sorted(decisions.items())),
            "prompt_hashes_identical": True,
            "plans": plans,
        },
        "frozen_gates": {
            "prompt_hashes_identical": True,
            "target_fallbacks": 0,
            "physical_copies_min": 1,
            "protected_decisions_min": 1,
            "general_fallback_decisions_min": 1,
            "late_dense_abstentions_min": 1,
            "target_first_token_agreement_not_below_general": True,
            "target_js_not_above_general": True,
            "cache_ready_median_ttft_not_above_general": True,
            "n4_median_not_above_general": True,
            "do_not_claim_task_accuracy": True,
        },
        "inputs": {
            "motivation_audit": str(MOTIVATION),
            "motivation_audit_sha256": replay.sha256(MOTIVATION),
            "trajectory_sha256": {
                instance_id: replay.sha256(
                    replay.trajectory_path(instance_id)
                )
                for instance_id in replay.INSTANCE_IDS
            },
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
            },
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


def run(output: Path) -> dict[str, Any]:
    register(output)
    _configure_replay_module()
    for arm in ARMS:
        replay.run_arm(output, arm, PORTS[arm])
    return summarize(output)


def _arm_summary(
    output: Path,
    arm: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ledger = load_jsonl(output / arm / "SERVER_LEDGER.jsonl")
    copies = [row for row in ledger if row.get("event") == "target_copied"]
    fallbacks = [
        row for row in ledger if row.get("event") == "target_fallback"
    ]
    builds = {
        str(row["source_id"]): float(row["materialize_ms"])
        for row in ledger
        if row.get("event")
        in ("source_materialized", "source_materialized_host")
    }
    targets = [row for row in rows if row["target_registered"]]
    n4 = [
        float(row["ttft_ms"])
        + builds.get(str(row["target_source_id"]), 0.0) / 4
        for row in targets
        if str(row["target_source_id"]) in builds
    ]
    return {
        "requests": len(rows),
        "registered_targets": len(targets),
        "physical_copies": len(copies),
        "target_fallbacks": len(fallbacks),
        "copied_tokens": sum(
            int(row.get("copied_k_tokens", 0)) for row in copies
        ),
        "median_all_ttft_ms": statistics.median(
            float(row["ttft_ms"]) for row in rows
        ),
        "median_cache_ready_target_ttft_ms": (
            statistics.median(float(row["ttft_ms"]) for row in targets)
            if targets
            else None
        ),
        "median_n4_including_build_ms": (
            statistics.median(n4) if n4 else None
        ),
    }


def _fidelity(
    dense: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    target_keys = [
        key for key, row in candidate.items() if row["target_registered"]
    ]
    agreements = []
    divergences = []
    for key in target_keys:
        left = replay.token_id(dense[key])
        right = replay.token_id(candidate[key])
        if left is not None and right is not None:
            agreements.append(left == right)
        value = replay.coarse_js(
            replay.top_distribution(dense[key]),
            replay.top_distribution(candidate[key]),
        )
        if value is not None:
            divergences.append(value)
    return {
        "target_requests": len(target_keys),
        "target_first_token_agreement": (
            sum(agreements) / len(agreements) if agreements else None
        ),
        "target_mean_top20_plus_residual_js": (
            statistics.fmean(divergences) if divergences else None
        ),
    }


def summarize(output: Path) -> dict[str, Any]:
    registration = register(output)
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
    keys = list(indexed[DENSE])
    prompt_identity = all(
        set(indexed[arm]) == set(keys)
        and all(
            indexed[arm][key]["prompt_hash"]
            == indexed[DENSE][key]["prompt_hash"]
            for key in keys
        )
        for arm in ARMS[1:]
    )
    summaries = {
        arm: _arm_summary(output, arm, rows[arm]) for arm in ARMS
    }
    fidelity = {
        arm: _fidelity(indexed[DENSE], indexed[arm])
        for arm in (GENERAL, V28)
    }
    decision_counts = Counter(
        row["decision"]["mode"] for row in rows[V28]
    )
    v28_summary = summaries[V28]
    general_summary = summaries[GENERAL]
    v28_fidelity = fidelity[V28]
    general_fidelity = fidelity[GENERAL]
    gates = {
        "prompt_hashes_identical": prompt_identity,
        "target_fallbacks": v28_summary["target_fallbacks"] == 0,
        "physical_copies_min": v28_summary["physical_copies"] >= 1,
        "protected_decisions_min": (
            decision_counts["payoff_guard_post_mutation_protected"] >= 1
        ),
        "general_fallback_decisions_min": (
            decision_counts["payoff_guard_general_middle_exact_prefix"] >= 1
        ),
        "late_dense_abstentions_min": (
            decision_counts["payoff_guard_dense_abstain_late_branch"] >= 1
        ),
        "target_first_token_agreement_not_below_general": (
            v28_fidelity["target_first_token_agreement"]
            >= general_fidelity["target_first_token_agreement"]
        ),
        "target_js_not_above_general": (
            v28_fidelity["target_mean_top20_plus_residual_js"]
            <= general_fidelity["target_mean_top20_plus_residual_js"]
        ),
        "cache_ready_median_ttft_not_above_general": (
            v28_summary["median_cache_ready_target_ttft_ms"]
            <= general_summary["median_cache_ready_target_ttft_ms"]
        ),
        "n4_median_not_above_general": (
            v28_summary["median_n4_including_build_ms"]
            <= general_summary["median_n4_including_build_ms"]
        ),
        "do_not_claim_task_accuracy": True,
    }
    value = {
        "completed_at_utc": replay.utc_now(),
        "status": (
            "PASS_V28_MECHANISM_SPEED_FIDELITY"
            if all(gates.values())
            else "FAIL_V28_MECHANISM_SPEED_FIDELITY"
        ),
        "experiment_scope": (
            "frozen-prompt mechanism/speed/first-token fidelity only; no "
            "task-accuracy claim"
        ),
        "registration_sha256": replay.sha256(
            output / "REPLAY_REGISTRATION.json"
        ),
        "prompt_hashes_identical": prompt_identity,
        "arm_summaries": summaries,
        "dense_reference_fidelity": fidelity,
        "v28_decision_mode_counts": dict(sorted(decision_counts.items())),
        "comparisons_v28_vs_general": {
            "cache_ready_ttft_ratio": (
                v28_summary["median_cache_ready_target_ttft_ms"]
                / general_summary["median_cache_ready_target_ttft_ms"]
            ),
            "n4_including_build_ratio": (
                v28_summary["median_n4_including_build_ms"]
                / general_summary["median_n4_including_build_ms"]
            ),
            "copied_tokens_ratio": (
                v28_summary["copied_tokens"]
                / general_summary["copied_tokens"]
            ),
            "target_first_token_agreement_difference": (
                v28_fidelity["target_first_token_agreement"]
                - general_fidelity["target_first_token_agreement"]
            ),
            "target_js_difference": (
                v28_fidelity["target_mean_top20_plus_residual_js"]
                - general_fidelity["target_mean_top20_plus_residual_js"]
            ),
        },
        "gate_outcomes": gates,
        "decision": (
            "Eligible for a separately preregistered same-history official "
            "accuracy canary; not a task-accuracy or baseline promotion."
            if all(gates.values())
            else "Do not run official V28 accuracy; revise or reject selector."
        ),
        "registration_snapshot": {
            "offline_decision_mode_counts": registration[
                "offline_plan_audit"
            ]["decision_mode_counts"]
        },
    }
    replay.write_json(output / "V28_REPLAY_RESULT.json", value)
    return value


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
