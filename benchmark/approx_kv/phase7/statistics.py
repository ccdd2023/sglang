from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

from benchmark.approx_kv.run_cl3_phase5_recompute import (
    cell_summary,
    select,
    ttft_stats,
)

AMORTIZATION_N = (1, 2, 4, 8)
NOT_OBSERVED = ">8/not_observed"


def compute_amortization(
    dense_targets: Sequence[Mapping[str, Any]],
    recovery_targets: Sequence[Mapping[str, Any]],
    *,
    dense_source_materialization_ms: float,
    recovery_source_preparation_ms: float,
    n_values: Sequence[int] = AMORTIZATION_N,
) -> dict[str, Any]:
    if len(dense_targets) != len(recovery_targets):
        raise ValueError("dense and recovery target counts differ")
    dense_ids = [row["target_id"] for row in dense_targets]
    recovery_ids = [row["target_id"] for row in recovery_targets]
    if dense_ids != recovery_ids:
        raise ValueError("dense and recovery target IDs are not paired")
    if max(n_values) > len(dense_targets):
        raise ValueError("amortization prefix exceeds measured targets")

    incremental_setup = float(recovery_source_preparation_ms) - float(
        dense_source_materialization_ms
    )
    results: dict[str, Any] = {}
    for n in n_values:
        dense_prefix = dense_targets[:n]
        recovery_prefix = recovery_targets[:n]
        valid = all(
            bool(row.get("expected_outcome"))
            for row in (*dense_prefix, *recovery_prefix)
        )
        if not valid:
            results[str(n)] = {
                "valid": False,
                "reason": "invalid_prefix_outcome",
                "speedup_full_setup": None,
                "speedup_incremental_setup": None,
            }
            continue
        dense_total = sum(float(row["request_path_ms"]) for row in dense_prefix)
        recovery_requests = sum(
            float(row["request_path_ms"]) for row in recovery_prefix
        )
        full_total = float(recovery_source_preparation_ms) + recovery_requests
        incremental_total = incremental_setup + recovery_requests
        if full_total <= 0 or incremental_total <= 0:
            results[str(n)] = {
                "valid": False,
                "reason": "non_positive_total",
                "speedup_full_setup": None,
                "speedup_incremental_setup": None,
            }
            continue
        results[str(n)] = {
            "valid": True,
            "reason": None,
            "dense_total_ms": dense_total,
            "recovery_request_total_ms": recovery_requests,
            "recovery_full_setup_total_ms": full_total,
            "recovery_incremental_setup_total_ms": incremental_total,
            "speedup_full_setup": dense_total / full_total,
            "speedup_incremental_setup": dense_total / incremental_total,
        }

    def first_break_even(field: str) -> int | str:
        for n in n_values:
            row = results[str(n)]
            if row["valid"] and float(row[field]) > 1.0:
                return n
        return NOT_OBSERVED

    return {
        "n": results,
        "dense_source_materialization_ms": dense_source_materialization_ms,
        "recovery_source_preparation_ms": recovery_source_preparation_ms,
        "incremental_setup_ms": incremental_setup,
        "full_setup_break_even_observed_N": first_break_even("speedup_full_setup"),
        "incremental_setup_break_even_observed_N": first_break_even(
            "speedup_incremental_setup"
        ),
    }


def same_context_canary(
    dense_output_ids: Sequence[int],
    recovery_output_ids: Sequence[int],
) -> dict[str, Any]:
    dense = list(dense_output_ids)
    recovery = list(recovery_output_ids)
    compared = min(len(dense), len(recovery))
    positions = [
        {
            "position": index,
            "dense": dense[index],
            "recovery": recovery[index],
            "match": dense[index] == recovery[index],
        }
        for index in range(compared)
    ]
    complete = len(dense) == len(recovery) == 8
    matched = complete and all(row["match"] for row in positions)
    return {
        "dense_output_ids": dense,
        "recovery_output_ids": recovery,
        "positions": positions,
        "complete_8_tokens": complete,
        "matched": matched,
        "engineering_status": "valid" if matched else "invalid",
    }


def summarize_ceiling_repeats(
    repeats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target_ratios = []
    full_by_n: dict[str, list[float]] = defaultdict(list)
    incremental_by_n: dict[str, list[float]] = defaultdict(list)
    for repeat in repeats:
        dense = repeat["arms"]["D0"]["targets"]
        recovery = repeat["arms"]["R0"]["targets"]
        for dense_row, recovery_row in zip(dense, recovery):
            if dense_row["expected_outcome"] and recovery_row["expected_outcome"]:
                target_ratios.append(
                    float(dense_row["request_path_ms"])
                    / float(recovery_row["request_path_ms"])
                )
        amortization = repeat["amortization"]
        for n, row in amortization["n"].items():
            if not row["valid"]:
                continue
            full_by_n[n].append(float(row["speedup_full_setup"]))
            incremental_by_n[n].append(float(row["speedup_incremental_setup"]))
    return {
        "paired_target_request_path_median_speedup": (
            statistics.median(target_ratios) if target_ratios else None
        ),
        "per_repeat_paired_target_speedups": target_ratios,
        "amortization_median_speedup": {
            n: {
                "full_setup": statistics.median(full_by_n[n]),
                "incremental_setup": statistics.median(incremental_by_n[n]),
            }
            for n in sorted(full_by_n, key=int)
        },
    }


def summarize_workflow_records(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summary = cell_summary({"records": list(records)})
    summary["cl3_compatible"] = True
    return summary


def pair_scheduler_arms(
    e0_records: Sequence[dict[str, Any]],
    r0_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    key_fields = ("repeat", "request_index", "phase", "role", "object_id")

    def index(rows: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
        result = {}
        for row in rows:
            key = tuple(row[field] for field in key_fields)
            if key in result:
                raise ValueError(f"duplicate scheduler pair key: {key}")
            result[key] = row
        return result

    e0 = index(e0_records)
    r0 = index(r0_records)
    if set(e0) != set(r0):
        raise ValueError("E0/R0 scheduler request keys do not pair")
    pairs = []
    for key in sorted(e0):
        control = e0[key]
        treated = r0[key]
        control_ttft = float(control["ttft_ms"])
        treated_ttft = float(treated["ttft_ms"])
        pairs.append(
            {
                **dict(zip(key_fields, key)),
                "e0_ttft_ms": control_ttft,
                "r0_ttft_ms": treated_ttft,
                "paired_delta_ms": control_ttft - treated_ttft,
                "speedup": control_ttft / treated_ttft,
                "e0_miss": _miss(control),
                "r0_miss": _miss(treated),
            }
        )

    denominators = {}
    for denominator in ("workflow_only", "all_reusable"):
        control_rows = select(e0_records, denominator)
        treated_rows = select(r0_records, denominator)
        control_stats = ttft_stats(control_rows)
        treated_stats = ttft_stats(treated_rows)
        denominator_pairs = [
            row
            for row in pairs
            if (row["phase"] == "workflow" if denominator == "workflow_only" else True)
        ]
        denominators[denominator] = {
            "e0": control_stats,
            "r0": treated_stats,
            "mean_speedup": (
                control_stats["ttft_mean_ms"] / treated_stats["ttft_mean_ms"]
            ),
            "p50_speedup": (
                control_stats["ttft_p50_ms"] / treated_stats["ttft_p50_ms"]
            ),
            "p95_ratio": (treated_stats["ttft_p95_ms"] / control_stats["ttft_p95_ms"]),
            "wall_clock_speedup": (
                control_stats["wall_clock_ms"] / treated_stats["wall_clock_ms"]
            ),
            "paired_delta_median_ms": statistics.median(
                row["paired_delta_ms"] for row in denominator_pairs
            ),
        }
        roles = sorted({row["role"] for row in denominator_pairs})
        denominators[denominator]["per_role"] = {
            role: {
                "pairs": len(
                    selected := [
                        row for row in denominator_pairs if row["role"] == role
                    ]
                ),
                "paired_delta_median_ms": statistics.median(
                    row["paired_delta_ms"] for row in selected
                ),
                "speedup_median": statistics.median(row["speedup"] for row in selected),
                "e0_misses": sum(row["e0_miss"] for row in selected),
                "r0_misses": sum(row["r0_miss"] for row in selected),
            }
            for role in roles
        }
    return {
        "pair_count": len(pairs),
        "pairs": pairs,
        "denominators": denominators,
        "cl3_compatible": True,
    }


def _miss(record: Mapping[str, Any]) -> bool:
    expected = int(record.get("expected_reusable_prefix_tokens") or 0)
    if expected <= 0:
        return False
    return int(record["cached_tokens"]) < expected


def performance_ranking_enabled(arms: Sequence[str]) -> bool:
    return "R4-like-5x" not in arms
