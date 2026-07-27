#!/usr/bin/env python3
"""Disjoint frozen replay for one-step critical coding-event abstention."""

from __future__ import annotations

import argparse
import hashlib
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import run_frozen_trajectory_replay_v18 as replay
from benchmark.multi_workflow.run_v28_payoff_guard_replay import _arm_summary


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_v31_critical_event_abstain_replay_20260727"
)
PROJECT = Path(__file__).resolve().parents[2]
DENSE = "dense"
GENERAL = "general"
V31 = "coding_critical_event_abstain_v31"
ARMS = (DENSE, GENERAL, V31)
PORTS = {DENSE: 33029, GENERAL: 33030, V31: 33031}
SELECTION_SALT = "v31-critical-event-abstain-v1\n"
PRIOR_REPLAY_INSTANCES = {
    "astropy__astropy-14995",
    "psf__requests-1142",
    "sphinx-doc__sphinx-9230",
    "pylint-dev__pylint-7277",
    "psf__requests-5414",
    "pydata__xarray-4075",
}
INSTANCE_IDS = (
    "scikit-learn__scikit-learn-12585",
    "pydata__xarray-6461",
    "pylint-dev__pylint-4970",
)


def _configure() -> None:
    replay.ARMS = ARMS
    replay.PORTS = PORTS
    replay.INSTANCE_IDS = INSTANCE_IDS


def _eligible_selection_rows() -> list[dict[str, Any]]:
    available = sorted(
        {
            path.parent.name
            for path in replay.TRAJECTORY_ROOT.glob("*/*.traj.json")
        }
        - PRIOR_REPLAY_INSTANCES
    )
    rows = []
    original = replay.INSTANCE_IDS
    try:
        for instance_id in available:
            replay.INSTANCE_IDS = (instance_id,)
            plans = replay.simulate_arm(V31)
            decisions = Counter(
                row["decision"]["mode"] for row in plans
            )
            critical = decisions["critical_event_dense_abstain"]
            reused = decisions["critical_event_general_reuse"]
            if critical < 2 or reused < 4:
                continue
            rows.append(
                {
                    "instance_id": instance_id,
                    "critical_decisions": critical,
                    "general_reuse_decisions": reused,
                    "selection_sha256": hashlib.sha256(
                        (SELECTION_SALT + instance_id).encode()
                    ).hexdigest(),
                }
            )
    finally:
        replay.INSTANCE_IDS = original
    return sorted(rows, key=lambda row: row["selection_sha256"])


def register(output: Path) -> dict[str, Any]:
    path = output / "V31_REGISTRATION.json"
    if path.exists():
        return replay.read_json(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _configure()
    eligible = _eligible_selection_rows()
    selected = eligible[:3]
    if tuple(row["instance_id"] for row in selected) != INSTANCE_IDS:
        raise AssertionError("V31 disjoint selection changed")
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
        raise AssertionError("V31 prompt identities differ")
    decisions = Counter(row["decision"]["mode"] for row in plans[V31])
    reasons = Counter(
        reason
        for row in plans[V31]
        for reason in row["decision"].get("critical_event_reasons", [])
    )
    value = {
        "registered_at_utc": replay.utc_now(),
        "status": "REGISTERED_BEFORE_ANY_V31_GPU_RUN",
        "experiment": "V31 disjoint critical-event abstention replay",
        "motivation": (
            "V30 showed that K/V KL signals do not cleanly separate stable "
            "CacheBlend task damage from matched-safe cases. Earlier Adaptive "
            "V2 also over-triggered on harmless search misses and merely "
            "shortened reuse. Test a narrow coding rule: after mutation/diff "
            "or a real executable failure, make exactly the next request "
            "Dense; otherwise use the unchanged General span."
        ),
        "selection": {
            "salt": SELECTION_SALT,
            "rule": (
                "Exclude all V28/V29 replay instances; require at least two "
                "critical abstentions and four General-reuse decisions from "
                "online trajectory events; SHA-256(salt || instance_id), "
                "sort ascending, take first three. No model outcomes used."
            ),
            "eligible": eligible,
            "selected": selected,
            "outcomes_used": False,
        },
        "arms": list(ARMS),
        "arm_order": list(ARMS),
        "protocol": {
            "same_frozen_prompt_ids": True,
            "diagnostic_new_tokens": 1,
            "diagnostic_output_never_enters_future_prompt": True,
            "critical_events": [
                "repository mutation command",
                "repository diff observation",
                "failed python/test execution",
            ],
            "explicit_noncritical_events": [
                "read-only search miss",
                "successful test execution",
            ],
            "abstention_duration_requests": 1,
            "noncritical_policy": "General contiguous 4096-token reuse",
            "prefetch": False,
            "reference_patch_or_future_output_used": False,
        },
        "offline_plan_audit": {
            "requests": len(plans[V31]),
            "decision_mode_counts": dict(sorted(decisions.items())),
            "critical_reason_counts": dict(sorted(reasons.items())),
            "prompt_hashes_identical": True,
            "plans": plans,
        },
        "frozen_gates": {
            "prompt_hashes_identical": True,
            "target_fallbacks": 0,
            "physical_copies_min": 1,
            "critical_abstentions_min": 2,
            "general_reuse_decisions_min": 4,
            "all_first_token_agreement_not_below_general": True,
            "all_js_not_above_general": True,
            "critical_first_token_agreement_not_below_general": True,
            "critical_js_not_above_general": True,
            "all_median_ttft_not_above_110pct_general": True,
            "all_median_paired_ratio_not_above_110pct": True,
            "all_median_ttft_below_dense": True,
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


def _fidelity_on_keys(
    dense: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
    keys: list[tuple[str, int]],
) -> dict[str, Any]:
    agreements = []
    divergences = []
    for key in keys:
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
        "requests": len(keys),
        "first_token_agreement": (
            sum(agreements) / len(agreements) if agreements else None
        ),
        "mean_top20_plus_residual_js": (
            statistics.mean(divergences) if divergences else None
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
    all_keys = sorted(indexed[DENSE])
    prompt_identity = all(
        indexed[arm][key]["prompt_hash"] == indexed[DENSE][key]["prompt_hash"]
        for arm in ARMS[1:]
        for key in all_keys
    )
    critical_keys = []
    for row in rows[V31]:
        if row["decision"]["mode"] != "critical_event_dense_abstain":
            continue
        key = (row["instance_id"], int(row["request_index"]) + 1)
        if key in indexed[V31]:
            critical_keys.append(key)
    fidelity = {
        cohort: {
            arm: _fidelity_on_keys(
                indexed[DENSE],
                indexed[arm],
                keys,
            )
            for arm in (GENERAL, V31)
        }
        for cohort, keys in (
            ("all", all_keys),
            ("critical_next_request", critical_keys),
        )
    }
    summaries = {
        arm: _arm_summary(output, arm, rows[arm]) for arm in ARMS
    }
    all_ttft = {
        arm: [float(indexed[arm][key]["ttft_ms"]) for key in all_keys]
        for arm in ARMS
    }
    medians = {
        arm: statistics.median(values)
        for arm, values in all_ttft.items()
    }
    paired = [
        float(indexed[V31][key]["ttft_ms"])
        / float(indexed[GENERAL][key]["ttft_ms"])
        for key in all_keys
    ]
    decisions = Counter(row["decision"]["mode"] for row in rows[V31])
    general_all = fidelity["all"][GENERAL]
    v31_all = fidelity["all"][V31]
    general_critical = fidelity["critical_next_request"][GENERAL]
    v31_critical = fidelity["critical_next_request"][V31]
    gates = {
        "prompt_hashes_identical": prompt_identity,
        "target_fallbacks": summaries[V31]["target_fallbacks"] == 0,
        "physical_copies_min": summaries[V31]["physical_copies"] >= 1,
        "critical_abstentions_min": (
            decisions["critical_event_dense_abstain"] >= 2
        ),
        "general_reuse_decisions_min": (
            decisions["critical_event_general_reuse"] >= 4
        ),
        "all_first_token_agreement_not_below_general": (
            v31_all["first_token_agreement"]
            >= general_all["first_token_agreement"]
        ),
        "all_js_not_above_general": (
            v31_all["mean_top20_plus_residual_js"]
            <= general_all["mean_top20_plus_residual_js"]
        ),
        "critical_first_token_agreement_not_below_general": (
            v31_critical["first_token_agreement"]
            >= general_critical["first_token_agreement"]
        ),
        "critical_js_not_above_general": (
            v31_critical["mean_top20_plus_residual_js"]
            <= general_critical["mean_top20_plus_residual_js"]
        ),
        "all_median_ttft_not_above_110pct_general": (
            medians[V31] <= 1.10 * medians[GENERAL]
        ),
        "all_median_paired_ratio_not_above_110pct": (
            statistics.median(paired) <= 1.10
        ),
        "all_median_ttft_below_dense": medians[V31] < medians[DENSE],
        "do_not_claim_task_accuracy": True,
    }
    passed = all(gates.values())
    value = {
        "completed_at_utc": replay.utc_now(),
        "status": (
            "PASS_V31_CRITICAL_EVENT_REPLAY"
            if passed
            else "FAIL_V31_CRITICAL_EVENT_REPLAY"
        ),
        "arm_summaries": summaries,
        "decision_mode_counts": dict(sorted(decisions.items())),
        "critical_next_request_keys": [
            {"instance_id": key[0], "request_index": key[1]}
            for key in critical_keys
        ],
        "fidelity": fidelity,
        "all_request_median_ttft_ms": medians,
        "v31_over_general_median_ttft_ratio": (
            medians[V31] / medians[GENERAL]
        ),
        "median_paired_ttft_ratio_v31_over_general": statistics.median(
            paired
        ),
        "v31_faster_all_requests": sum(ratio < 1 for ratio in paired),
        "gate_outcomes": gates,
        "decision": (
            "Eligible for paired agent accuracy canary."
            if passed
            else "Reject or revise V31 before task accuracy."
        ),
        "registration_sha256": replay.sha256(
            output / "V31_REGISTRATION.json"
        ),
        "offline_decision_mode_counts": registration[
            "offline_plan_audit"
        ]["decision_mode_counts"],
    }
    replay.write_json(output / "V31_RESULT.json", value)
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
