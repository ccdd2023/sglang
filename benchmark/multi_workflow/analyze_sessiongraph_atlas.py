#!/usr/bin/env python3
"""Analyze a complete V11 causal atlas without implicit artifact discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


DISTURBANCES = frozenset(
    {
        "identity",
        "change_after",
        "upstream_edit",
        "semantic_prefix",
        "position_only",
        "module_reorder",
        "same_task",
        "cross_task",
    }
)
DOSES = frozenset({0.0, 0.25, 0.5, 0.75, 1.0})


@dataclass(frozen=True)
class Observation:
    session_id: str
    module_id: str
    module_type: str
    cache_scope: str
    disturbance: str
    recompute_fraction: float
    token_count: int
    position_norm: float
    rope_delta: int
    prefix_changed_tokens: int
    graph_distance: int | None
    causal_splice_logit_js: float
    lookup_ms: float

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Observation":
        return cls(
            session_id=str(row["session_id"]),
            module_id=str(row["module_id"]),
            module_type=str(row["module_type"]),
            cache_scope=str(row["cache_scope"]),
            disturbance=str(row["disturbance"]),
            recompute_fraction=float(row["recompute_fraction"]),
            token_count=int(row["token_count"]),
            position_norm=float(row["position_norm"]),
            rope_delta=int(row["rope_delta"]),
            prefix_changed_tokens=int(row["prefix_changed_tokens"]),
            graph_distance=None
            if row.get("graph_distance") is None
            else int(row["graph_distance"]),
            causal_splice_logit_js=float(row["causal_splice_logit_js"]),
            lookup_ms=float(row["lookup_ms"]),
        )


def _matrix(rows: Sequence[Observation], workflow: bool) -> np.ndarray:
    types = sorted({row.module_type for row in rows})
    scopes = sorted({row.cache_scope for row in rows})
    disturbances = sorted({row.disturbance for row in rows})
    output = []
    for row in rows:
        vector = [
            1.0,
            math.log1p(row.token_count),
            row.position_norm,
            math.copysign(math.log1p(abs(row.rope_delta)), row.rope_delta),
            math.log1p(row.prefix_changed_tokens),
            row.recompute_fraction,
        ]
        if workflow:
            vector.extend(
                [
                    8.0
                    if row.graph_distance is None
                    else min(8, row.graph_distance),
                    *(float(row.module_type == value) for value in types),
                    *(float(row.cache_scope == value) for value in scopes),
                    *(float(row.disturbance == value) for value in disturbances),
                ]
            )
        output.append(vector)
    return np.asarray(output, dtype=np.float64)


def grouped_r2(rows: Sequence[Observation], workflow: bool, folds: int = 5) -> float:
    sessions = sorted({row.session_id for row in rows})
    if len(sessions) < 2:
        return math.nan
    folds = min(folds, len(sessions))
    fold_of = {
        session: int(
            hashlib.sha256(f"sessiongraph-fold|{session}".encode()).hexdigest(), 16
        )
        % folds
        for session in sessions
    }
    x = _matrix(rows, workflow)
    y = np.asarray([row.causal_splice_logit_js for row in rows])
    predictions = np.zeros_like(y)
    for fold in range(folds):
        test = np.asarray([fold_of[row.session_id] == fold for row in rows])
        train = ~test
        if not test.any() or not train.any():
            continue
        scale = np.maximum(x[train].std(axis=0), 1e-9)
        scale[0] = 1.0
        tx, vx = x[train] / scale, x[test] / scale
        penalty = np.eye(tx.shape[1]) * 1e-4
        penalty[0, 0] = 0
        weights = np.linalg.pinv(tx.T @ tx + penalty) @ tx.T @ y[train]
        predictions[test] = vx @ weights
    denominator = float(np.sum((y - y.mean()) ** 2))
    return 0.0 if denominator <= 0 else 1 - float(
        np.sum((y - predictions) ** 2)
    ) / denominator


def _bootstrap_delta(
    rows: Sequence[Observation], iterations: int
) -> tuple[float, float, float]:
    observed = grouped_r2(rows, True) - grouped_r2(rows, False)
    by_session: dict[str, list[Observation]] = defaultdict(list)
    for row in rows:
        by_session[row.session_id].append(row)
    sessions = sorted(by_session)
    rng = random.Random(20260717)
    draws = []
    for draw in range(iterations):
        sample = []
        for index in range(len(sessions)):
            source = rng.choice(sessions)
            sample.extend(
                Observation(**{**row.__dict__, "session_id": f"boot:{draw}:{index}"})
                for row in by_session[source]
            )
        delta = grouped_r2(sample, True) - grouped_r2(sample, False)
        if math.isfinite(delta):
            draws.append(delta)
    draws.sort()
    return (
        observed,
        draws[int(0.025 * len(draws))] if draws else math.nan,
        draws[min(len(draws) - 1, int(0.975 * len(draws)))]
        if draws
        else math.nan,
    )


def _safe_reduction(
    rows: Sequence[Observation], iterations: int
) -> tuple[float, float]:
    grouped: dict[tuple[str, str, float], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        bucket = (
            "safe"
            if row.graph_distance is None or row.graph_distance >= 2
            else "unsafe"
        )
        grouped[
            (row.session_id, row.disturbance, row.recompute_fraction)
        ][bucket].append(row.causal_splice_logit_js)
    by_session: dict[str, list[float]] = defaultdict(list)
    for (session_id, _, _), values in grouped.items():
        if values["safe"] and values["unsafe"]:
            unsafe, safe = mean(values["unsafe"]), mean(values["safe"])
            if unsafe > 0:
                by_session[session_id].append((unsafe - safe) / unsafe)
    reductions = [mean(values) for values in by_session.values() if values]
    if not reductions:
        return math.nan, math.nan
    rng = random.Random(2718)
    draws = sorted(
        mean(rng.choice(reductions) for _ in reductions)
        for _ in range(iterations)
    )
    return mean(reductions), draws[int(0.025 * len(draws))]


def analyze(rows: Sequence[Observation], iterations: int = 10_000) -> dict[str, Any]:
    reasons = []
    sessions = {row.session_id for row in rows}
    disturbances = {row.disturbance for row in rows}
    cells: dict[tuple[str, str, str], set[float]] = defaultdict(set)
    for row in rows:
        cells[(row.session_id, row.module_id, row.disturbance)].add(
            row.recompute_fraction
        )
    duplicate_keys = len(rows) - sum(len(values) for values in cells.values())
    if (
        len(rows) != 4960
        or len(sessions) != 32
        or disturbances != DISTURBANCES
        or any(values != DOSES for values in cells.values())
        or duplicate_keys
    ):
        reasons.append("formal coverage is not exactly 32 sessions/8 disturbances/4960 rows")
    controls = [
        row for row in rows if row.disturbance in {"identity", "change_after"}
    ]
    controls_pass = bool(controls) and max(
        row.causal_splice_logit_js for row in controls
    ) <= 1e-3
    if not controls_pass:
        reasons.append("negative-control max JS exceeds 1e-3")
    delta, delta_low, delta_high = _bootstrap_delta(rows, iterations)
    if not (delta >= 0.05 and delta_low > 0):
        reasons.append("workflow-feature delta-R2 gate failed")
    reduction, reduction_low = _safe_reduction(rows, iterations)
    if not (reduction >= 0.30 and reduction_low > 0):
        reasons.append("distance>=2 harm-reduction gate failed")
    lookup_p95 = (
        float(np.quantile([row.lookup_ms for row in rows], 0.95))
        if rows
        else math.inf
    )
    if lookup_p95 >= 2:
        reasons.append("lookup p95 >=2ms")
    return {
        "passed": not reasons,
        "status": "PASS" if not reasons else "FALSIFIED",
        "reasons": reasons,
        "sessions": len(sessions),
        "disturbances": sorted(disturbances),
        "rows": len(rows),
        "duplicate_design_keys": duplicate_keys,
        "negative_controls_passed": controls_pass,
        "delta_r2": delta,
        "delta_r2_ci95": [delta_low, delta_high],
        "safe_harm_reduction": reduction,
        "safe_harm_reduction_ci_low": reduction_low,
        "lookup_p95_ms": lookup_p95,
        "bootstrap_iterations": iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    rows = [
        Observation.from_row(json.loads(line))
        for line in args.observations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = analyze(rows, args.bootstrap)
    args.gate_output.parent.mkdir(parents=True, exist_ok=True)
    args.gate_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
