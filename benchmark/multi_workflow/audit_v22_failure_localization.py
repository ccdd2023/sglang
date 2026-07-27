#!/usr/bin/env python3
"""Localize V22 fidelity failures and test whether its seam changed reuse."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
    sha256,
    token_id,
    utc_now,
    write_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_INPUT = ARTIFACTS / "impactkv_v22_seam32_replay_20260727"
CANDIDATE = "coding_post_mutation_seam32_v22"


def index(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row["instance_id"]), int(row["request_index"])): row
        for row in read_json(path)["rows"]
    }


def audit(root: Path) -> dict[str, Any]:
    transitions: Counter[str] = Counter()
    disagreements = []
    for repeat in (1, 2, 3):
        rows = {
            arm: index(
                root / f"repeat_{repeat}" / arm / "REPLAY_RESULTS.json"
            )
            for arm in ("dense", "general", CANDIDATE)
        }
        for key, candidate in rows[CANDIDATE].items():
            dense = rows["dense"][key]
            general = rows["general"][key]
            dense_token = token_id(dense)
            general_token = token_id(general)
            candidate_token = token_id(candidate)
            general_match = general_token == dense_token
            candidate_match = candidate_token == dense_token
            changed = (
                candidate["target_length"],
                candidate["source_length"],
            ) != (
                general["target_length"],
                general["source_length"],
            )
            registered = bool(candidate["target_registered"])
            cohort = (
                "changed_target"
                if changed
                else "unchanged_target"
                if registered
                else "unregistered"
            )
            transition = (
                "both_match"
                if general_match and candidate_match
                else "general_only"
                if general_match
                else "candidate_only"
                if candidate_match
                else "both_mismatch"
            )
            transitions[f"{cohort}:{transition}"] += 1
            if general_token != candidate_token:
                decision = candidate["decision"]
                disagreements.append(
                    {
                        "repeat": repeat,
                        "instance_id": key[0],
                        "request_index": key[1],
                        "cohort": cohort,
                        "dense_token": dense_token,
                        "general_token": general_token,
                        "candidate_token": candidate_token,
                        "transition": transition,
                        "mode": decision["mode"],
                        "repository_pathful_groups": decision.get(
                            "repository_pathful_groups"
                        ),
                    }
                )

    result_path = root / "V22_REPLAY_RESULT.json"
    result = read_json(result_path)
    return {
        "status": "V22_FAILURE_LOCALIZATION_COMPLETE",
        "completed_at_utc": utc_now(),
        "classification": "retrospective_failure_localization",
        "input": {
            "path": str(result_path),
            "sha256": sha256(result_path),
        },
        "transitions": dict(sorted(transitions.items())),
        "general_candidate_disagreements": disagreements,
        "registered_disagreements": sum(
            row["cohort"] != "unregistered" for row in disagreements
        ),
        "unregistered_disagreements": sum(
            row["cohort"] == "unregistered" for row in disagreements
        ),
        "seam_observation": {
            "registered_prefix_median": [
                row["candidate_prefix_telemetry"][
                    "median_ordinary_prefix_tokens"
                ]
                for row in result["repeats"]
            ],
            "expected_by_registration_max": 814,
            "gate_passed": result["gate_outcomes"][
                "seam_mechanism_proven"
            ],
        },
        "finding": (
            "The fixed seam did not reduce the recorded ordinary-prefix hit. "
            "Moreover, V22 disagreed with General even on an unregistered "
            "request, which the shifted coding selector cannot modify. The "
            "remaining confound is that enabling ordinary Radix reuse also "
            "changes unregistered requests and source-building requests."
        ),
        "next_candidate_constraint": (
            "Scope ordinary Radix reuse to registered target requests only; "
            "keep unregistered and source-building requests dense, retain the "
            "post-mutation shifted island, and do not prefetch."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = audit(args.input.resolve())
    output = args.output or args.input / "V22_FAILURE_AUDIT.json"
    write_json(output, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
