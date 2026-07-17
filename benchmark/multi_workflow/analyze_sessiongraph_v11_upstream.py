#!/usr/bin/env python3
"""Summarize the V11 upstream-edit directional checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


DOSES = {0.0, 0.25, 0.5, 0.75, 1.0}


def analyze(observations: Path, bootstrap: int = 10_000) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in observations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups: dict[tuple[str, str], dict[float, float]] = defaultdict(dict)
    duplicates = 0
    for row in rows:
        key = (str(row["session_id"]), str(row["module_id"]))
        dose = float(row["recompute_fraction"])
        duplicates += dose in groups[key]
        groups[key][dose] = float(row["causal_splice_logit_js"])
    incomplete = [
        [*key, sorted(values)] for key, values in groups.items() if set(values) != DOSES
    ]
    if incomplete or duplicates:
        return {
            "status": "INVALID_INCOMPLETE_ARTIFACT",
            "groups": len(groups),
            "rows": len(rows),
            "duplicate_design_keys": duplicates,
            "incomplete_cells": incomplete,
        }
    dose_summary = {
        str(dose): {
            "median_js": median(group[dose] for group in groups.values()),
            "mean_js": mean(group[dose] for group in groups.values()),
        }
        for dose in sorted(DOSES)
    }
    by_session: dict[str, list[float]] = defaultdict(list)
    for (session_id, _), values in groups.items():
        if values[0.0] > 0:
            by_session[session_id].append(
                (values[0.0] - values[0.5]) / values[0.0]
            )
    reductions = [median(values) for values in by_session.values() if values]
    rng = random.Random(20260717)
    draws = sorted(
        median(rng.choice(reductions) for _ in reductions)
        for _ in range(bootstrap)
    )
    return {
        "status": "DIRECTIONAL_CHECKPOINT_NOT_A_FORMAL_GATE",
        "sessions": len(by_session),
        "groups": len(groups),
        "rows": len(rows),
        "incomplete_cells": [],
        "duplicate_design_keys": 0,
        "dose_summary": dose_summary,
        "groups_improved_at_50pct": sum(
            values[0.5] < values[0.0] for values in groups.values()
        ),
        "groups_total": len(groups),
        "session_cluster_median_relative_reduction_0_to_50pct": median(reductions),
        "session_cluster_bootstrap_ci95": [
            draws[int(0.025 * len(draws))],
            draws[min(len(draws) - 1, int(0.975 * len(draws)))],
        ],
        "bootstrap_iterations": bootstrap,
        "formal_p0_complete": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    result = analyze(args.observations, args.bootstrap)
    args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
