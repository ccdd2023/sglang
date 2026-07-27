#!/usr/bin/env python3
"""Disjoint frozen-prompt replay for the stronger V29 payoff guard."""

from __future__ import annotations

import argparse
import hashlib
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import run_frozen_trajectory_replay_v18 as replay
from benchmark.multi_workflow.run_v28_payoff_guard_replay import (
    _arm_summary,
    _fidelity,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v29_disjoint_replay_20260727"
PROJECT = Path(__file__).resolve().parents[2]
DENSE = "dense"
GENERAL = "general"
V29 = "coding_post_mutation_payoff_guard_v29"
ARMS = (DENSE, GENERAL, V29)
PORTS = {DENSE: 33026, GENERAL: 33027, V29: 33028}
SELECTION_SALT = "v29-disjoint-replay-v1\n"
V28_INSTANCES = {
    "astropy__astropy-14995",
    "psf__requests-1142",
    "sphinx-doc__sphinx-9230",
}
INSTANCE_IDS = (
    "pylint-dev__pylint-7277",
    "psf__requests-5414",
    "pydata__xarray-4075",
)


def _configure() -> None:
    replay.ARMS = ARMS
    replay.PORTS = PORTS
    replay.INSTANCE_IDS = INSTANCE_IDS


def _selection_audit() -> list[dict[str, str]]:
    available = sorted(
        path.parent.name
        for path in replay.TRAJECTORY_ROOT.glob("*/*.traj.json")
        if path.parent.name not in V28_INSTANCES
    )
    rows = [
        {
            "instance_id": instance_id,
            "selection_sha256": hashlib.sha256(
                (SELECTION_SALT + instance_id).encode()
            ).hexdigest(),
        }
        for instance_id in available
    ]
    selected = sorted(rows, key=lambda row: row["selection_sha256"])[:3]
    if tuple(row["instance_id"] for row in selected) != INSTANCE_IDS:
        raise AssertionError("V29 disjoint selection changed")
    return selected


def register(output: Path) -> dict[str, Any]:
    path = output / "V29_REGISTRATION.json"
    if path.exists():
        return replay.read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _configure()
    selection = _selection_audit()
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
        raise AssertionError("V29 prompt identities differ")
    decisions = Counter(row["decision"]["mode"] for row in plans[V29])
    value = {
        "registered_at_utc": replay.utc_now(),
        "status": "REGISTERED_BEFORE_ANY_V29_GPU_RUN",
        "experiment": "V29 disjoint frozen-prompt replay",
        "motivation": (
            "V28 improved first-token fidelity but passed only two of four "
            "speed orders. Raise the payoff threshold from 0.60 to 1.20 at "
            "the natural gap in the frozen V28 payoff distribution, causing "
            "more General-middle plus exact-prefix fallbacks. Validate on "
            "three trajectories selected without overlap or arm outcomes."
        ),
        "selection": {
            "salt": SELECTION_SALT,
            "rule": (
                "Exclude all three V28 replay instances, SHA-256 "
                "(salt || instance_id), sort ascending, take first three."
            ),
            "selected": selection,
            "outcomes_used": False,
        },
        "arms": list(ARMS),
        "arm_order": list(ARMS),
        "protocol": {
            "same_frozen_prompt_ids": True,
            "diagnostic_new_tokens": 1,
            "diagnostic_output_never_enters_future_prompt": True,
            "payoff_ratio_threshold": 1.20,
            "exact_prefix_credit_tokens": 640,
            "minimum_future_target_upper_bound": 4,
            "common_key_definition": "V29 registered-target prompt keys",
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "offline_plan_audit": {
            "requests": len(plans[V29]),
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
            "common_median_ttft_not_above_general": True,
            "common_median_paired_ratio_not_above_one": True,
            "do_not_claim_task_accuracy": True,
        },
        "inputs": {
            "trajectory_sha256": {
                instance_id: replay.sha256(
                    replay.trajectory_path(instance_id)
                )
                for instance_id in INSTANCE_IDS
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
    replay.write_json(
        output / "REPLAY_REGISTRATION.json",
        {
            "registered_at_utc": value["registered_at_utc"],
            "status": value["status"],
            "parent_registration_sha256": replay.sha256(path),
            "arms": list(ARMS),
            "instances": list(INSTANCE_IDS),
            "protected": value["protected"],
        },
    )
    return value


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
    prompt_identity = all(
        indexed[arm][key]["prompt_hash"] == indexed[DENSE][key]["prompt_hash"]
        for arm in ARMS[1:]
        for key in indexed[DENSE]
    )
    summaries = {
        arm: _arm_summary(output, arm, rows[arm]) for arm in ARMS
    }
    fidelity = {
        arm: _fidelity(indexed[DENSE], indexed[arm])
        for arm in (GENERAL, V29)
    }
    common = [
        key for key, row in indexed[V29].items() if row["target_registered"]
    ]
    ttft = {
        arm: [float(indexed[arm][key]["ttft_ms"]) for key in common]
        for arm in (GENERAL, V29)
    }
    medians = {
        arm: statistics.median(values) for arm, values in ttft.items()
    }
    paired_ratios = [
        indexed[V29][key]["ttft_ms"] / indexed[GENERAL][key]["ttft_ms"]
        for key in common
    ]
    decisions = Counter(row["decision"]["mode"] for row in rows[V29])
    v29_fidelity = fidelity[V29]
    general_fidelity = fidelity[GENERAL]
    gates = {
        "prompt_hashes_identical": prompt_identity,
        "target_fallbacks": summaries[V29]["target_fallbacks"] == 0,
        "physical_copies_min": summaries[V29]["physical_copies"] >= 1,
        "protected_decisions_min": (
            decisions["payoff_guard_post_mutation_protected"] >= 1
        ),
        "general_fallback_decisions_min": (
            decisions["payoff_guard_general_middle_exact_prefix"] >= 1
        ),
        "late_dense_abstentions_min": (
            decisions["payoff_guard_dense_abstain_late_branch"] >= 1
        ),
        "target_first_token_agreement_not_below_general": (
            v29_fidelity["target_first_token_agreement"]
            >= general_fidelity["target_first_token_agreement"]
        ),
        "target_js_not_above_general": (
            v29_fidelity["target_mean_top20_plus_residual_js"]
            <= general_fidelity["target_mean_top20_plus_residual_js"]
        ),
        "common_median_ttft_not_above_general": (
            medians[V29] <= medians[GENERAL]
        ),
        "common_median_paired_ratio_not_above_one": (
            statistics.median(paired_ratios) <= 1
        ),
        "do_not_claim_task_accuracy": True,
    }
    value = {
        "completed_at_utc": replay.utc_now(),
        "status": (
            "PASS_V29_DISJOINT_REPLAY"
            if all(gates.values())
            else "FAIL_V29_DISJOINT_REPLAY"
        ),
        "arm_summaries": summaries,
        "dense_reference_fidelity": fidelity,
        "v29_decision_mode_counts": dict(sorted(decisions.items())),
        "common_target_keys": len(common),
        "median_common_ttft_ms": medians,
        "common_median_ratio_v29_over_general": (
            medians[V29] / medians[GENERAL]
        ),
        "median_paired_ratio_v29_over_general": statistics.median(
            paired_ratios
        ),
        "v29_faster_common_requests": sum(
            ratio < 1 for ratio in paired_ratios
        ),
        "gate_outcomes": gates,
        "decision": (
            "Eligible only for reverse-order speed replication."
            if all(gates.values())
            else "Reject V29 before accuracy."
        ),
        "registration_sha256": replay.sha256(
            output / "V29_REGISTRATION.json"
        ),
        "offline_decision_mode_counts": registration[
            "offline_plan_audit"
        ]["decision_mode_counts"],
    }
    replay.write_json(output / "V29_RESULT.json", value)
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
