#!/usr/bin/env python3
"""Aggregate the causal V24/V25 paired KV-reuse campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_v25_paired_campaign_20260727"
V23 = "coding_post_mutation_target_prefix_v23"
GENERAL = "general"
RUNS = (
    (
        "scikit-learn__scikit-learn-13779",
        ARTIFACTS / "impactkv_v25b_paired_agent_canary_20260727",
        "old_independent_general_only",
    ),
    (
        "scikit-learn__scikit-learn-12585",
        ARTIFACTS
        / "impactkv_v25c_paired_agent_canary_sklearn12585_20260727",
        "old_independent_v23_only",
    ),
    (
        "pylint-dev__pylint-7277",
        ARTIFACTS
        / "impactkv_v25d_paired_agent_canary_pylint7277_20260727",
        "old_independent_general_only",
    ),
)
MECHANICAL = (
    ARTIFACTS
    / "impactkv_v24b_paired_mechanism_canary_20260727"
    / "V24_RESULT.json"
)
FROZEN_REPLAY = (
    ARTIFACTS
    / "impactkv_v23_target_prefix_replay_20260727"
    / "V23_REPLAY_RESULT.json"
)


def _paired_row(
    instance_id: str,
    root: Path,
    old_independent_transition: str,
) -> dict[str, Any]:
    runtime = read_json(root / "V25_RESULT.json")
    official = read_json(root / "V25_OFFICIAL_RESULT.json")
    arms = official["arms"]
    if arms[V23]["resolved"] > arms[GENERAL]["resolved"]:
        outcome = "v23_only"
    elif arms[V23]["resolved"] < arms[GENERAL]["resolved"]:
        outcome = "general_only"
    elif arms[V23]["resolved"]:
        outcome = "both_resolved"
    else:
        outcome = "both_failed"
    return {
        "instance_id": instance_id,
        "old_independent_transition": old_independent_transition,
        "causal_paired_outcome": outcome,
        "branch_shared_calls": runtime["branch"]["shared_calls"],
        "branch_source_lengths": runtime["branch"]["source_lengths"],
        "first_branch_prompt_hash_identical": (
            len(set(runtime["first_branch_prompt_hashes"].values())) == 1
        ),
        "target_copies": runtime["server"]["copy_counts"],
        "target_fallbacks": runtime["server"]["target_fallbacks"],
        "official": {
            arm: {
                "resolved": arms[arm]["resolved"],
                "empty_patch": arms[arm]["empty_patch"],
                "branched_model_requests": arms[arm][
                    "branched_model_requests"
                ],
                "branched_model_elapsed_seconds": arms[arm][
                    "branched_model_elapsed_seconds"
                ],
                "median_ttft_ms": arms[arm]["median_ttft_ms"],
            }
            for arm in (V23, GENERAL)
        },
    }


def summarize(output: Path) -> dict[str, Any]:
    rows = [_paired_row(*values) for values in RUNS]
    replay = read_json(FROZEN_REPLAY)
    mechanical = read_json(MECHANICAL)
    outcomes = {
        name: sum(row["causal_paired_outcome"] == name for row in rows)
        for name in ("v23_only", "general_only", "both_resolved", "both_failed")
    }
    value = {
        "summarized_at_utc": utc_now(),
        "status": "PAIRED_DEVELOPMENT_EVIDENCE_ONLY",
        "mechanical_canary": {
            "status": mechanical["status"],
            "all_gates_passed": all(mechanical["gates"].values()),
            "same_prompt_first_token": mechanical[
                "same_prompt_first_token"
            ],
        },
        "frozen_replay": {
            "status": replay["status"],
            "median_cache_ready_ratio_v23_over_general": replay["aggregate"][
                "median_cache_ready_ratio"
            ],
            "median_n4_build_inclusive_ratio_v23_over_general": replay[
                "aggregate"
            ]["median_n4_ratio"],
            "pooled_first_token_matches": replay["aggregate"][
                "pooled_first_token_matches"
            ],
        },
        "paired_official_rows": rows,
        "paired_outcomes": outcomes,
        "causal_transition_revision": {
            "old_v23_only_that_remained_v23_only": 1,
            "old_general_only_that_remained_general_only": 0,
            "old_general_only_that_became_both_failed": 2,
            "conclusion": (
                "The prior V23 rescue replicated under shared history and "
                "repository state.  Neither prior General-only damage "
                "replicated; both became joint failures.  This is encouraging "
                "for post-mutation coding awareness, but three selected "
                "transition cases are not a population estimate."
            ),
        },
        "decision": {
            "promote_to_full225": False,
            "reason": (
                "One causal V23-only resolution and no causal General-only "
                "resolution is insufficient and transition-selected.  Run a "
                "pre-registered 5-10 task paired replication first."
            ),
            "next_algorithm_hypothesis": (
                "Repository-version-aware narrow reuse is most useful after "
                "the source edit already exists, where it preserves the "
                "post-mutation validation/submission state.  Measure and gate "
                "that phase explicitly instead of claiming generic coding "
                "awareness."
            ),
            "next_primary_accuracy_metric": (
                "Official resolved rate and paired V23-only vs General-only "
                "transitions; exact output agreement is diagnostic only."
            ),
            "next_speed_metrics": (
                "Cache-ready TTFT, N=4 build-inclusive latency, and complete "
                "branched agent model/wall time.  Fixed V23-first paired TTFT "
                "must not be treated as an unbiased speed estimate."
            ),
        },
        "protected": {
            "prefetch": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "old_dirty_checkout_modified": False,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "V25_PAIRED_CAMPAIGN_RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = summarize(args.output)
    print(json.dumps(value["paired_outcomes"], sort_keys=True))


if __name__ == "__main__":
    main()
