#!/usr/bin/env python3
"""Generate the Phase 4 artifact manifest and Phase 5 recalculated metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


WORKSPACE = Path("/home/chris/Workspaces/kvcache-research")
WORKTREES = WORKSPACE / "worktrees"
RESULTS = WORKSPACE / "results"


def build_cost_ledger(
    *,
    source_preparation_ms: float,
    target_adapter_preparation_ms: float,
    seed_head_ms: float,
    post_pressure_reseed_ms: float,
    transfer_ms: float,
    target_only_ms: float,
) -> dict[str, float]:
    request_path_ms = (
        seed_head_ms
        + target_adapter_preparation_ms
        + post_pressure_reseed_ms
        + transfer_ms
        + target_only_ms
    )
    return {
        "request_path_ms": request_path_ms,
        "recovery_object_lifecycle_ms": source_preparation_ms + request_path_ms,
    }


def pressure_filler_count(
    *,
    capacity_tokens: int,
    rho_logical_demand: float,
    setup_used_tokens: int,
    setup_evictable_tokens: int,
    tokens_per_filler: int,
) -> int:
    if capacity_tokens <= 0 or tokens_per_filler <= 0:
        raise ValueError("capacity and tokens_per_filler must be positive")
    setup_tokens = setup_used_tokens + setup_evictable_tokens
    requested = math.ceil(rho_logical_demand * capacity_tokens)
    return max(0, math.ceil((requested - setup_tokens) / tokens_per_filler))


class ScratchNamespaceTracker:
    """Reference lifecycle model for unique extra-key scratch branches."""

    def __init__(self) -> None:
        self._references: dict[str, int] = {}

    def acquire(self, namespace: str) -> None:
        self._references[namespace] = self._references.get(namespace, 0) + 1

    def release(self, namespace: str) -> None:
        references = self._references.get(namespace)
        if references is None:
            raise KeyError(namespace)
        if references == 1:
            del self._references[namespace]
        else:
            self._references[namespace] = references - 1

    def active(self) -> tuple[str, ...]:
        return tuple(sorted(self._references))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(worktree: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
    ).strip()


def nearest_rank(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def summarize_requests(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "requests": 0,
            "ttft_mean_ms": None,
            "ttft_p50_ms": None,
            "ttft_p95_ms": None,
            "ttft_total_ms": 0.0,
            "elapsed_total_ms": 0.0,
            "misses": 0,
            "per_request_clamped_hit_fraction": None,
        }
    ttfts = [float(row["ttft_ms"]) for row in rows]
    reusable = [
        row for row in rows if int(row["expected_reusable_prefix_tokens"]) > 0
    ]
    hit_fractions = [
        min(
            1.0,
            float(row["cached_tokens"])
            / float(row["expected_reusable_prefix_tokens"]),
        )
        for row in reusable
    ]
    return {
        "requests": len(rows),
        "ttft_mean_ms": statistics.fmean(ttfts),
        "ttft_p50_ms": statistics.median(ttfts),
        "ttft_p95_ms": nearest_rank(ttfts, 0.95),
        "ttft_total_ms": sum(ttfts),
        "elapsed_total_ms": sum(float(row["elapsed_ms"]) for row in rows),
        "misses": sum(
            int(row["cached_tokens"]) < int(row["expected_reusable_prefix_tokens"])
            for row in reusable
        ),
        "per_request_clamped_hit_fraction": (
            statistics.fmean(hit_fractions) if hit_fractions else None
        ),
    }


def summarize_result_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = payload["results"]
    repeats = sorted({int(row["repeat"]) for row in rows})
    by_repeat = {}
    for repeat in repeats:
        repeat_rows = [row for row in rows if int(row["repeat"]) == repeat]
        by_repeat[str(repeat)] = {
            "workflow": summarize_requests(
                [row for row in repeat_rows if row["phase"] == "workflow"]
            ),
            "all_reusable": summarize_requests(
                [row for row in repeat_rows if int(row["occurrence"]) > 0]
            ),
            "full_trace": summarize_requests(repeat_rows),
        }

    per_role = {}
    for role in sorted({str(row["role"]) for row in rows}):
        role_rows = [row for row in rows if row["role"] == role]
        per_role[role] = {
            "all": summarize_requests(role_rows),
            "workflow": summarize_requests(
                [row for row in role_rows if row["phase"] == "workflow"]
            ),
            "all_reusable": summarize_requests(
                [row for row in role_rows if int(row["occurrence"]) > 0]
            ),
        }

    return {
        "path": str(path),
        "sha256": sha256(path),
        "run_id": payload["run_id"],
        "settings": payload["settings"],
        "actual_reusable_pressure": payload["actual_reusable_pressure"],
        "workflow": summarize_requests(
            [row for row in rows if row["phase"] == "workflow"]
        ),
        "all_reusable": summarize_requests(
            [row for row in rows if int(row["occurrence"]) > 0]
        ),
        "full_trace": summarize_requests(rows),
        "per_role": per_role,
        "per_repeat": by_repeat,
        "reset_invariant": payload["reset_invariant"],
        "idle_pool_invariant": payload["idle_pool_invariant"],
    }


def offline_variable_size_optimum(
    rows: list[dict[str, Any]], capacity_tokens: int
) -> dict[str, Any]:
    """Solve the whole-object offline caching upper bound for one trace."""
    rows = sorted(rows, key=lambda row: int(row["step"]))
    positions: dict[str, list[int]] = defaultdict(list)
    sizes: dict[str, int] = {}
    for position, row in enumerate(rows):
        object_id = str(row["object_id"])
        positions[object_id].append(position)
        sizes[object_id] = int(row["expected_reusable_prefix_tokens"])

    intervals: list[tuple[str, int, int, int]] = []
    for object_id, object_positions in positions.items():
        for start, end in zip(object_positions, object_positions[1:]):
            intervals.append((object_id, start, end, sizes[object_id]))

    if not intervals:
        return {
            "solver": "scipy.optimize.milp",
            "reuse_requests": 0,
            "optimal_hits": 0,
            "optimal_misses": 0,
        }

    constraints = lil_matrix((len(rows), len(intervals)), dtype=float)
    for interval_index, (_, start, end, size) in enumerate(intervals):
        for position in range(start + 1, end + 1):
            constraints[position, interval_index] = size

    result = milp(
        c=[-1.0] * len(intervals),
        integrality=[1] * len(intervals),
        bounds=Bounds([0.0] * len(intervals), [1.0] * len(intervals)),
        constraints=LinearConstraint(
            constraints.tocsr(),
            [-math.inf] * len(rows),
            [float(capacity_tokens)] * len(rows),
        ),
        options={"time_limit": 60.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"offline optimum failed: {result.message}")
    optimal_hits = int(sum(value >= 0.5 for value in result.x))
    return {
        "solver": "scipy.optimize.milp",
        "solver_message": result.message,
        "capacity_tokens": capacity_tokens,
        "reuse_requests": len(intervals),
        "optimal_hits": optimal_hits,
        "optimal_misses": len(intervals) - optimal_hits,
    }


def phase5_report() -> dict[str, Any]:
    scheduler_root = RESULTS / "phase5-scheduler-formal-5a87166b4"
    restart_root = RESULTS / "phase5-scheduler-restarts-5a87166b4"
    prefetch_root = RESULTS / "phase5-prefetch-formal-5a87166b4"

    scheduler_files = sorted(scheduler_root.glob("*/result.json"))
    restart_files = sorted(restart_root.glob("*/result.json"))
    prefetch_files = sorted(prefetch_root.glob("*/result.json"))

    scheduler_runs = [summarize_result_file(path) for path in scheduler_files]
    restart_runs = [summarize_result_file(path) for path in restart_files]
    prefetch_runs = [summarize_result_file(path) for path in prefetch_files]

    by_rho: dict[str, dict[str, Any]] = defaultdict(dict)
    for run in scheduler_runs:
        settings = run["settings"]
        by_rho[str(settings["target_pressure"])][settings["policy"]] = run

    paired = {}
    for rho, policies in sorted(by_rho.items(), key=lambda item: float(item[0])):
        baseline = policies["lru"]
        paired[rho] = {}
        for policy, run in sorted(policies.items()):
            if policy == "lru":
                continue
            comparisons = {}
            for scope in ("workflow", "all_reusable", "full_trace"):
                base_value = baseline[scope]["ttft_mean_ms"]
                policy_value = run[scope]["ttft_mean_ms"]
                comparisons[scope] = {
                    "lru_mean_ms": base_value,
                    "policy_mean_ms": policy_value,
                    "speedup": base_value / policy_value,
                    "delta_ms": policy_value - base_value,
                }
            paired[rho][policy] = comparisons

    offline = {}
    for rho, policies in sorted(by_rho.items(), key=lambda item: float(item[0])):
        representative_path = Path(policies["lru"]["path"])
        payload = json.loads(representative_path.read_text())
        by_repeat = {}
        for repeat in sorted({int(row["repeat"]) for row in payload["results"]}):
            repeat_rows = [
                row for row in payload["results"] if int(row["repeat"]) == repeat
            ]
            by_repeat[str(repeat)] = offline_variable_size_optimum(
                repeat_rows, int(payload["gpu_kv_capacity_tokens"])
            )
        offline[rho] = by_repeat

    return {
        "schema_version": 1,
        "source_git_sha": "5a87166b436e00fa730aa7062e949516ca823a96",
        "definitions": {
            "workflow": "phase == workflow",
            "all_reusable": "occurrence > 0",
            "full_trace": "all sequential requests; elapsed_total_ms is the sum of per-request elapsed_ms",
            "hit_fraction": "mean(min(1, cached_i / expected_i)) over reusable requests",
            "p95": "nearest-rank order statistic used by the original runner",
        },
        "scheduler_runs": scheduler_runs,
        "restart_runs": restart_runs,
        "prefetch_runs": prefetch_runs,
        "paired_against_lru": paired,
        "offline_variable_size_upper_bound": offline,
    }


def artifact(
    worktree: str,
    relative_path: str,
    *,
    status: str,
    scope: str,
    notes: list[str],
    superseded_cells: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = WORKTREES / worktree
    path = root / relative_path
    return {
        "worktree": worktree,
        "branch_head": git_head(root),
        "path": str(path),
        "relative_path": relative_path,
        "sha256": sha256(path),
        "status": status,
        "scope": scope,
        "notes": notes,
        "superseded_cells": superseded_cells or [],
    }


def phase4_manifest() -> dict[str, Any]:
    r2_pointer = {
        "body_tokens": [1024, 2048],
        "header_tokens": 64,
        "rho_logical_demand": 2.0,
        "ratio": 0.01,
        "replacement": "benchmark/approx_kv/results/phase4-r2/sm75-causal-key-rerun.json",
        "replacement_commit": "e36f1529b838c12a9eb2af7ba4dde91ae9ec124b",
    }
    r5_pointer = {
        "body_tokens": [1024, 2048],
        "header_tokens": 64,
        "rho_logical_demand": 2.0,
        "replacement": "benchmark/approx_kv/results/phase4-r5/sm75-causal-key-rerun.json",
        "replacement_commit": "abcedd62b5a5d801742734e300a5df21e1436737",
    }
    artifacts = [
        artifact(
            "raw-rope",
            "benchmark/approx_kv/results/phase4-r0/sm75-unified-pressure.json",
            status="representative",
            scope="R0 numerator body1024/2048, header64, rho2",
            notes=[
                "Dense denominator is reused from the R1 measurement.",
                "R1-k0 is the mechanism-equivalent full OAT proxy.",
            ],
        ),
        artifact(
            "epic-legolink",
            "benchmark/approx_kv/results/phase4-r1/sm75-eviction-pressure.json",
            status="authoritative_historical",
            scope="R1 pressure OAT slices",
            notes=["OAT slices, not a full Cartesian matrix."],
        ),
        artifact(
            "epic-legolink",
            "benchmark/approx_kv/results/phase4-r1/sm75-inrequest-matrix.json",
            status="historical_mechanism_only",
            scope="R1 k/head/body matrix",
            notes=[
                "No eviction pressure.",
                "Older code SHA before eviction-aware allocation.",
                "Header/body axes differ from the unified pressure contract.",
            ],
        ),
        artifact(
            "cacheblend",
            "benchmark/approx_kv/results/phase4-r2/sm75-unified-pressure.json",
            status="historical_oat",
            scope="R2 ratio/header/body/rho OAT slices",
            notes=[
                "The corrected causal rerun supersedes the specified key cells.",
                "Historical combined-positive claims must not be cited without the replacement.",
            ],
            superseded_cells=[r2_pointer],
        ),
        artifact(
            "cacheblend",
            "benchmark/approx_kv/results/phase4-r2/sm75-causal-key-rerun.json",
            status="authoritative_corrected",
            scope="R2 corrected causal body1024/2048 key cells",
            notes=["Fallback is indirectly verified because the explicit metric is absent."],
        ),
        artifact(
            "kvcomm",
            "benchmark/approx_kv/results/phase4-r4/sm75-unified-pressure.json",
            status="authoritative_historical_diagnostic",
            scope="R4 header/body/rho OAT slices",
            notes=["Setup break-even remains formula-based, not measured N reuse."],
        ),
        artifact(
            "cachetune",
            "benchmark/approx_kv/results/phase4-r5/sm75-unified-pressure.json",
            status="historical_oat",
            scope="R5 body/shape result",
            notes=[
                "Final-SHA rho_sweep_points is empty.",
                "The corrected causal rerun supersedes the specified key cells.",
            ],
            superseded_cells=[r5_pointer],
        ),
        artifact(
            "cachetune",
            "benchmark/approx_kv/results/phase4-r5/sm75-causal-key-rerun.json",
            status="authoritative_corrected",
            scope="R5 corrected causal body1024/2048 key cells",
            notes=["Explicit dense-fallback counter is zero in every formal recovery round."],
        ),
    ]

    stale_paths = [
        RESULTS / "phase4-epic-pressure-rho/dense-rho0p9.json",
        RESULTS / "phase4-epic-pressure-rho/k0-rho0p9-fixed.json",
    ]
    stale = [
        {
            "path": str(path),
            "sha256": sha256(path),
            "status": "stale_do_not_use",
            "reason": "Conflicting/incorrect eviction_observed metadata; use k0-fixed-rho0p9.json and the compact result.",
        }
        for path in stale_paths
    ]
    authoritative_raw = RESULTS / "phase4-epic-pressure-rho/k0-fixed-rho0p9.json"
    return {
        "schema_version": 1,
        "matrix_definition": "Phase 4 results are one-at-a-time slices around fixed representative points unless explicitly stated otherwise.",
        "artifacts": artifacts,
        "r1_rho0p9_raw": {
            "authoritative": {
                "path": str(authoritative_raw),
                "sha256": sha256(authoritative_raw),
            },
            "stale": stale,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-output", type=Path, required=True)
    parser.add_argument("--phase5-output", type=Path, required=True)
    args = parser.parse_args()

    args.phase4_output.write_text(
        json.dumps(phase4_manifest(), indent=2, sort_keys=True) + "\n"
    )
    args.phase5_output.write_text(
        json.dumps(phase5_report(), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
