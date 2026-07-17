#!/usr/bin/env python3
"""Gate V11 development negative controls from one explicit artifact."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DOSES = {0.0, 0.25, 0.5, 0.75, 1.0}


def analyze(observations: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in observations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cells: dict[tuple[str, str, str], set[float]] = defaultdict(set)
    for row in rows:
        cells[
            (str(row["session_id"]), str(row["module_id"]), str(row["disturbance"]))
        ].add(float(row["recompute_fraction"]))
    sessions = {str(row["session_id"]) for row in rows}
    incomplete = [list(key) for key, values in cells.items() if values != DOSES]
    metrics = {}
    for disturbance in ("identity", "change_after"):
        values = np.asarray(
            [
                float(row["causal_splice_logit_js"])
                for row in rows
                if row["disturbance"] == disturbance
            ]
        )
        metrics[disturbance] = {
            "max_js": float(values.max()) if len(values) else math.inf,
            "median_js": float(np.median(values)) if len(values) else math.inf,
        }
    lookup = np.asarray([float(row["lookup_ms"]) for row in rows])
    reasons = []
    if len(sessions) != 32 or len(cells) != 256 or len(rows) != 1280:
        reasons.append("coverage != 32 sessions/256 cells/1280 rows")
    if incomplete:
        reasons.append(f"{len(incomplete)} cells lack all five doses")
    if any(value["max_js"] > 1e-3 for value in metrics.values()):
        reasons.append("negative-control max JS exceeds 1e-3")
    lookup_p95 = float(np.quantile(lookup, 0.95)) if len(lookup) else float("inf")
    if lookup_p95 >= 2:
        reasons.append("lookup p95 >=2ms")
    return {
        "passed": not reasons,
        "status": "PASS" if not reasons else "FALSIFIED",
        "reasons": reasons,
        "sessions": len(sessions),
        "groups": len(cells),
        "rows": len(rows),
        "doses": sorted(DOSES),
        "metrics": metrics,
        "lookup_p95_ms": lookup_p95,
        "incomplete_cells": incomplete,
        "formal_observations": str(observations),
        "invalidated_unchunked_partial_excluded": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.observations)
    args.gate_output.parent.mkdir(parents=True, exist_ok=True)
    args.gate_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
