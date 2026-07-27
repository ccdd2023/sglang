#!/usr/bin/env python3
"""CL3: Phase 5 zero-GPU recalculation.

Recomputes the Phase 5 scheduler and prefetch matrices from the committed raw
request records. No GPU cell is re-run. The recalculation adds the denominators
and statistics that the original Phase 5 summary did not separate:

- workflow-only denominator (SLA view over the fixed workflow trace);
- all-reusable denominator (primary p95 view over every reusable request);
- full-trace wall-clock;
- per-role TTFT and miss counts;
- paired-against-S0 and per-restart statistics;
- cache hit fraction clamped per request instead of after aggregation;
- S2 reported as Belady-style rather than as a true variable-size optimum.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from benchmark.approx_kv.phase6.runner import append_jsonl, write_json
from benchmark.approx_kv.phase6.schema import file_sha256, payload_sha256

POLICY_LABELS = {
    "lru": "S0 LRU",
    "workflow_steps": "S1 workflow-steps",
    "belady": "S2 Belady-style next-request-ordinal oracle",
    "recovery_value": "S3 recovery-value density",
    "hierarchical": "S4 hierarchical object class",
}

BASELINE_POLICY = "lru"
BASELINE_PREFETCH = "p0"

DENOMINATORS = {
    "workflow_only": "phase == workflow",
    "all_reusable": "expected_reusable_prefix_tokens > 0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--central-log", type=Path, required=True)
    parser.add_argument(
        "--scheduler-dir",
        type=Path,
        action="append",
        required=True,
        help="Phase 5 scheduler result directory; repeatable.",
    )
    parser.add_argument("--prefetch-dir", type=Path, action="append", default=[])
    return parser.parse_args()


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def clamped_hit_fraction(record: dict[str, Any]) -> float | None:
    expected = record.get("expected_reusable_prefix_tokens")
    if not expected:
        return None
    return min(1.0, max(0.0, float(record["cached_tokens"]) / float(expected)))


def ttft_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"requests": 0}
    ttft = [float(record["ttft_ms"]) for record in records]
    fractions = [
        value
        for value in (clamped_hit_fraction(record) for record in records)
        if value is not None
    ]
    misses = sum(1 for value in fractions if value < 1.0)
    full_misses = sum(1 for record in records if int(record["cached_tokens"]) == 0)
    return {
        "requests": len(records),
        "ttft_mean_ms": statistics.fmean(ttft),
        "ttft_p50_ms": percentile(ttft, 0.50),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "wall_clock_ms": sum(float(record["elapsed_ms"]) for record in records),
        "clamped_hit_fraction_mean": (
            statistics.fmean(fractions) if fractions else None
        ),
        "partial_or_full_miss_requests": misses,
        "full_miss_requests": full_misses,
    }


def select(records: Iterable[dict[str, Any]], denominator: str) -> list[dict[str, Any]]:
    rows = [record for record in records if record.get("sample_kind") == "measured"]
    if denominator == "workflow_only":
        return [record for record in rows if record.get("phase") == "workflow"]
    return [record for record in rows if record.get("expected_reusable_prefix_tokens")]


def load_cells(directories: Sequence[Path]) -> list[dict[str, Any]]:
    cells = []
    for directory in directories:
        for path in sorted(directory.glob("*/result.json")):
            payload = json.loads(path.read_text())
            settings = payload["settings"]
            cells.append(
                {
                    "cell_dir": str(path.parent),
                    "result_path": str(path),
                    "result_sha256": file_sha256(path),
                    "run_id": payload.get("run_id"),
                    "policy": settings["policy"],
                    "policy_label": POLICY_LABELS.get(
                        settings["policy"], settings.get("policy_label", "")
                    ),
                    "prefetch_mode": settings.get("prefetch_mode", "p0"),
                    "enable_hicache": bool(settings.get("enable_hicache", False)),
                    "target_pressure": float(settings["target_pressure"]),
                    "restart": int(settings["restart"]),
                    "formal_repeats": int(settings.get("formal_repeats", 0)),
                    "actual_reusable_pressure": payload.get("actual_reusable_pressure"),
                    "capacity_tokens": payload.get("gpu_kv_capacity_tokens"),
                    "records": payload["results"],
                }
            )
    return cells


def cell_summary(cell: dict[str, Any]) -> dict[str, Any]:
    repeats = sorted({int(record["repeat"]) for record in cell["records"]})
    summary: dict[str, Any] = {"per_repeat": {}, "denominators": {}}
    for denominator in DENOMINATORS:
        rows = select(cell["records"], denominator)
        summary["denominators"][denominator] = ttft_stats(rows)
        summary["denominators"][denominator]["per_role"] = {
            role: ttft_stats([row for row in rows if row.get("role") == role])
            for role in sorted({str(row.get("role")) for row in rows})
        }
    for repeat in repeats:
        selected = [
            record
            for record in cell["records"]
            if int(record["repeat"]) == repeat
            and record.get("sample_kind") == "measured"
        ]
        summary["per_repeat"][str(repeat)] = {
            "full_trace_wall_clock_ms": sum(
                float(record["elapsed_ms"]) for record in selected
            ),
            "requests": len(selected),
            "workflow_only": ttft_stats(
                [record for record in selected if record.get("phase") == "workflow"]
            ),
            "all_reusable": ttft_stats(
                [
                    record
                    for record in selected
                    if record.get("expected_reusable_prefix_tokens")
                ]
            ),
        }
    return summary


def paired(
    cells: Sequence[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    *,
    axis: str = "policy",
) -> list[dict[str, Any]]:
    """Pair every cell against its control on one axis.

    ``axis="policy"`` compares each eviction policy against S0 LRU at the same
    prefetch mode, pressure and restart. ``axis="prefetch"`` compares each
    prefetch mode against P0 at the same policy, pressure and restart, which is
    the only meaningful control for the Phase 5 prefetch matrix because every
    prefetch cell already uses S4.
    """
    index = {
        (
            cell["policy"],
            cell["prefetch_mode"],
            cell["enable_hicache"],
            cell["target_pressure"],
            cell["restart"],
        ): cell
        for cell in cells
    }
    rows = []
    for key, cell in index.items():
        policy, prefetch_mode, hicache, pressure, restart = key
        if axis == "policy":
            if policy == BASELINE_POLICY:
                continue
            baseline_key = (
                BASELINE_POLICY,
                prefetch_mode,
                hicache,
                pressure,
                restart,
            )
            baseline_label = POLICY_LABELS[BASELINE_POLICY]
        else:
            if prefetch_mode == BASELINE_PREFETCH:
                continue
            baseline_key = (
                policy,
                BASELINE_PREFETCH,
                hicache,
                pressure,
                restart,
            )
            baseline_label = f"{policy} + {BASELINE_PREFETCH}"
        baseline = index.get(baseline_key)
        if baseline is None:
            continue
        row: dict[str, Any] = {
            "policy": policy,
            "policy_label": POLICY_LABELS.get(policy, policy),
            "baseline": baseline_label,
            "pairing_axis": axis,
            "prefetch_mode": prefetch_mode,
            "enable_hicache": hicache,
            "target_pressure": pressure,
            "restart": restart,
        }
        for denominator in DENOMINATORS:
            treated = summaries[cell["cell_dir"]]["denominators"][denominator]
            control = summaries[baseline["cell_dir"]]["denominators"][denominator]
            if not treated.get("requests") or not control.get("requests"):
                continue
            row[denominator] = {
                "mean_speedup": control["ttft_mean_ms"] / treated["ttft_mean_ms"],
                "p50_speedup": control["ttft_p50_ms"] / treated["ttft_p50_ms"],
                "p95_ratio": treated["ttft_p95_ms"] / control["ttft_p95_ms"],
                "wall_clock_speedup": (
                    control["wall_clock_ms"] / treated["wall_clock_ms"]
                ),
                "baseline_clamped_hit_fraction": control["clamped_hit_fraction_mean"],
                "treated_clamped_hit_fraction": treated["clamped_hit_fraction_mean"],
            }
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["prefetch_mode"],
            row["policy"],
            row["target_pressure"],
            row["restart"],
        ),
    )


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    keys = sorted(
        {(row["policy"], row["prefetch_mode"], row["target_pressure"]) for row in rows}
    )
    for policy, prefetch_mode, pressure in keys:
        selected = [
            row
            for row in rows
            if row["policy"] == policy
            and row["prefetch_mode"] == prefetch_mode
            and row["target_pressure"] == pressure
        ]
        entry: dict[str, Any] = {"restarts": len(selected)}
        for denominator in DENOMINATORS:
            values = [row[denominator] for row in selected if denominator in row]
            if not values:
                continue
            entry[denominator] = {
                "median_mean_speedup": statistics.median(
                    value["mean_speedup"] for value in values
                ),
                "median_p50_speedup": statistics.median(
                    value["p50_speedup"] for value in values
                ),
                "median_p95_ratio": statistics.median(
                    value["p95_ratio"] for value in values
                ),
                "median_wall_clock_speedup": statistics.median(
                    value["wall_clock_speedup"] for value in values
                ),
                "per_restart_mean_speedup": [value["mean_speedup"] for value in values],
            }
        aggregated[f"{policy}:{prefetch_mode}:rho{pressure}"] = entry
    return aggregated


def main() -> int:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("cl3-%Y%m%dT%H%M%SZ")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    append_jsonl(
        args.central_log,
        {
            "run_id": run_id,
            "phase": "CL3",
            "status": "running",
            "output": str(args.output.resolve()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    scheduler_cells = load_cells(args.scheduler_dir)
    prefetch_cells = load_cells(args.prefetch_dir)
    if not scheduler_cells:
        raise ValueError("no Phase 5 scheduler cells were found")
    all_cells = [*scheduler_cells, *prefetch_cells]
    summaries = {cell["cell_dir"]: cell_summary(cell) for cell in all_cells}

    scheduler_paired = paired(scheduler_cells, summaries)
    prefetch_paired = (
        paired(prefetch_cells, summaries, axis="prefetch") if prefetch_cells else []
    )

    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": "CL3",
        "derivation": "zero_gpu_recalculation",
        "source_git_sha": args.source_git_sha,
        "image_digest": args.image_digest,
        "result_git_sha": None,
        "result_commit_status": "pending_result_commit",
        "performance_claim": "phase5_exact_cache_recalculation_only",
        "definitions": {
            "denominators": DENOMINATORS,
            "clamped_hit_fraction": (
                "min(1, cached_tokens / expected_reusable_prefix_tokens) "
                "computed per request before aggregation"
            ),
            "full_trace_wall_clock": (
                "sum of client elapsed_ms over every measured request in one "
                "formal repeat, including pressure fill and replay"
            ),
            "scheduler_baseline": POLICY_LABELS[BASELINE_POLICY],
            "prefetch_baseline": (
                "S4 hierarchical with prefetch mode p0; the Phase 5 prefetch "
                "matrix has no LRU arm, so P0 is the only valid control"
            ),
            "s2_naming": (
                "S2 is a Belady-style next-request-ordinal oracle over the "
                "recorded trace; it is not a variable-size offline optimum"
            ),
            "variable_size_offline_optimum": (
                "declared a Phase 7 deliverable; not computed here"
            ),
            "sample_independence": (
                "requests inside one trace are not independent experiments; "
                "per-restart and per-repeat values are reported separately"
            ),
        },
        "inputs": [
            {
                "cell_dir": cell["cell_dir"],
                "result_sha256": cell["result_sha256"],
                "run_id": cell["run_id"],
                "policy": cell["policy"],
                "prefetch_mode": cell["prefetch_mode"],
                "enable_hicache": cell["enable_hicache"],
                "target_pressure": cell["target_pressure"],
                "restart": cell["restart"],
                "actual_reusable_pressure": cell["actual_reusable_pressure"],
                "capacity_tokens": cell["capacity_tokens"],
            }
            for cell in all_cells
        ],
        "cell_summaries": {
            cell["cell_dir"]: {
                "policy": cell["policy"],
                "policy_label": cell["policy_label"],
                "prefetch_mode": cell["prefetch_mode"],
                "enable_hicache": cell["enable_hicache"],
                "target_pressure": cell["target_pressure"],
                "restart": cell["restart"],
                **summaries[cell["cell_dir"]],
            }
            for cell in all_cells
        },
        "scheduler_paired": scheduler_paired,
        "scheduler_aggregate": aggregate(scheduler_paired),
        "prefetch_paired": prefetch_paired,
        "prefetch_aggregate": aggregate(prefetch_paired) if prefetch_paired else {},
    }
    payload["raw_sha256"] = payload_sha256(payload)
    write_json(args.output, payload)
    append_jsonl(
        args.central_log,
        {
            "run_id": run_id,
            "phase": "CL3",
            "status": "completed",
            "raw_sha256": payload["raw_sha256"],
            "output": str(args.output.resolve()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    for key, value in payload["scheduler_aggregate"].items():
        workflow = value.get("workflow_only", {})
        reusable = value.get("all_reusable", {})
        if not workflow or not reusable:
            continue
        print(
            f"{key:34} workflow_mean={workflow['median_mean_speedup']:.3f} "
            f"reusable_mean={reusable['median_mean_speedup']:.3f} "
            f"reusable_p95={reusable['median_p95_ratio']:.3f} "
            f"wall={reusable['median_wall_clock_speedup']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
