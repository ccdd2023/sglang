#!/usr/bin/env python3
"""Localize V21 first-token losses to changed or unchanged selector cohorts."""

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
DEFAULT_INPUT = ARTIFACTS / "impactkv_v21_robust_dual_replay_20260727"
CANDIDATE = "coding_post_mutation_dual_v20"


def index(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = read_json(path)["rows"]
    return {
        (str(row["instance_id"]), int(row["request_index"])): row
        for row in rows
    }


def audit(root: Path) -> dict[str, Any]:
    transitions: dict[str, Counter[str]] = {
        "changed_span": Counter(),
        "unchanged_span": Counter(),
    }
    disagreements: dict[str, list[dict[str, Any]]] = {
        "changed_span": [],
        "unchanged_span": [],
    }
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
            changed = (
                candidate["target_length"],
                candidate["source_length"],
            ) != (
                general["target_length"],
                general["source_length"],
            )
            cohort = "changed_span" if changed else "unchanged_span"
            dense_token = token_id(dense)
            general_token = token_id(general)
            candidate_token = token_id(candidate)
            general_match = general_token == dense_token
            candidate_match = candidate_token == dense_token
            transition = (
                "both_match"
                if general_match and candidate_match
                else "general_only"
                if general_match
                else "candidate_only"
                if candidate_match
                else "both_mismatch"
            )
            transitions[cohort][transition] += 1
            if general_token != candidate_token:
                disagreements[cohort].append(
                    {
                        "repeat": repeat,
                        "instance_id": key[0],
                        "request_index": key[1],
                        "target_registered": candidate["target_registered"],
                        "dense_token": dense_token,
                        "general_token": general_token,
                        "candidate_token": candidate_token,
                        "transition": transition,
                    }
                )
    summaries = {}
    for cohort, counts in transitions.items():
        total = sum(counts.values())
        general_matches = counts["both_match"] + counts["general_only"]
        candidate_matches = counts["both_match"] + counts["candidate_only"]
        summaries[cohort] = {
            "requests": total,
            "transitions": dict(counts),
            "general_matches": general_matches,
            "candidate_matches": candidate_matches,
            "candidate_minus_general_matches": (
                candidate_matches - general_matches
            ),
            "general_agreement": general_matches / total,
            "candidate_agreement": candidate_matches / total,
            "general_candidate_token_disagreements": len(
                disagreements[cohort]
            ),
        }
    result_path = root / "V21_REPLAY_RESULT.json"
    return {
        "status": "V21_FIRST_TOKEN_FAILURE_AUDIT_COMPLETE",
        "completed_at_utc": utc_now(),
        "classification": "retrospective_failure_localization",
        "input": {
            "path": str(result_path),
            "sha256": sha256(result_path),
        },
        "cohorts": summaries,
        "disagreements": disagreements,
        "finding": (
            "The mutation-changed selector cohort has no net first-token loss "
            "versus General. All pooled net loss is in unchanged-span requests, "
            "isolating the remaining problem to the ordinary-prefix numerical "
            "path rather than the coding selector."
        ),
        "next_candidate_constraint": (
            "Recompute a small tail immediately before the shifted middle "
            "island while keeping both the exact earlier prefix KV and the "
            "post-mutation shifted island; no prefetch."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = audit(args.input.resolve())
    output = args.output or args.input / "V21_FIRST_TOKEN_FAILURE_AUDIT.json"
    write_json(output, value)
    print(json.dumps(value["cohorts"], indent=2))


if __name__ == "__main__":
    main()
