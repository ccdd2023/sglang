#!/usr/bin/env python3
"""Cold one-source/one-target frontier for the full coding-evidence island.

V9 removes V8's 6144-token ceiling only at the same successful read-only
coding-evidence opportunities.  General and V8 are reconstructed from the
complete natural overlap and tail-capped, fixing the earlier replay's
head-versus-tail ambiguity.  Every source is a real preceding request and is
consumed exactly once; no prefetch is issued.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    MODEL,
    capped_tail,
    manifest_case,
    sha256_file,
    write_json,
)
from benchmark.multi_workflow.run_coding_dual_island_v8_cold import (
    run_arm as run_cold_arm,
)
from benchmark.multi_workflow.run_coding_evidence_payoff_paired import (
    read_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
SOURCE_CASES = (
    ARTIFACTS
    / "impactkv_coding_dual_island_v8_cold_20260727/COLD_CASES.json"
)
OPPORTUNITY_AUDIT = (
    ARTIFACTS
    / "impactkv_coding_evidence_payoff_v7_20260726/"
    "OPPORTUNITY_AUDIT.json"
)
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_coding_full_evidence_v9_cold_20260727"
)
ARMS = (
    "dense",
    "general_dual_4k",
    "coding_dual_v8_tail",
    "coding_full_v9",
)
REUSE_ARMS = ARMS[1:]
ROUNDS = 3
V8_CAP = 6144
GENERAL_CAP = 4096
NATIVE_CACHE_READY_REFERENCE = {
    "cacheblend": 79.01429315136923,
    "kvcomm": 88.31062959906316,
}


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def _source_paths() -> list[Path]:
    root = Path(__file__).parents[2]
    return [
        Path(__file__),
        root / "python/sglang/srt/mem_cache/kvcomm_exact.py",
        root / "python/sglang/srt/mem_cache/radix_cache.py",
        root / "python/sglang/srt/managers/schedule_batch.py",
        root / "python/sglang/srt/mem_cache/common.py",
    ]


def _full_cases() -> list[dict[str, Any]]:
    cases = read_json(SOURCE_CASES)["cases"]
    audit_rows = read_json(OPPORTUNITY_AUDIT)["rows"]
    lengths = {
        f"{row['instance_id']}-q{row['target_request_index']}": int(
            row["candidate_tokens"]
        )
        for row in audit_rows
    }
    output = []
    for case in cases:
        full_tokens = lengths[case["case_id"]]
        row = {**case, "full_evidence_tokens": full_tokens}
        if row["source_start"] + full_tokens >= len(row["source_input_ids"]):
            raise ValueError(f"{row['case_id']}: full source span is not middle")
        if row["target_start"] + full_tokens >= len(row["target_input_ids"]):
            raise ValueError(f"{row['case_id']}: full target span is not middle")
        output.append(row)
    return output


def prepare(output: Path) -> dict[str, Any]:
    cases = _full_cases()
    cases_path = output / "V9_CASES.json"
    write_json(cases_path, {"cases": cases})
    # Reuse the validated V8 cold executor, which reads this conventional
    # filename.  Both files contain the identical registered case payload.
    executor_cases_path = output / "COLD_CASES.json"
    write_json(executor_cases_path, {"cases": cases})
    manifest_hashes: dict[str, dict[str, str]] = {}
    for round_index in range(ROUNDS):
        round_dir = output / "rounds" / f"r{round_index}"
        manifest_hashes[str(round_index)] = {}
        for arm in REUSE_ARMS:
            rows = []
            for case in cases:
                full_span = {
                    "source_start": case["source_start"],
                    "target_start": case["target_start"],
                    "length": case["full_evidence_tokens"],
                }
                if arm == "general_dual_4k":
                    span = capped_tail(full_span, GENERAL_CAP)
                elif arm == "coding_dual_v8_tail":
                    span = capped_tail(full_span, V8_CAP)
                else:
                    span = full_span
                row = manifest_case(
                    case_id=case["case_id"],
                    policy_label=arm,
                    source_ids=case["source_input_ids"],
                    target_ids=case["target_input_ids"],
                    span=span,
                )
                row["target_uses"] = 1
                rows.append(row)
            path = round_dir / "manifests" / f"{arm}.json"
            write_json(
                path,
                {
                    "cache_dtype": "bfloat16",
                    "cases": rows,
                    "lease_ttl_s": 900,
                    "ledger_path": str(
                        round_dir / "server" / arm / "EXACT_LEDGER.jsonl"
                    ),
                    "model_id": MODEL,
                    "ordinary_prefix_reuse_enabled": True,
                    "rope": {
                        "base": 10_000_000,
                        "is_neox_style": True,
                        "rotary_dim": 128,
                    },
                    "version": 2,
                },
            )
            manifest_hashes[str(round_index)][arm] = sha256_file(path)

    marginal = [
        case["full_evidence_tokens"] - min(
            case["full_evidence_tokens"], V8_CAP
        )
        for case in cases
    ]
    registration = {
        "registered_before_gpu": True,
        "classification": (
            "independent-round one-target-per-source V9 speed confirmation; "
            "not task accuracy or same-workload native-SOTA evidence"
        ),
        "model": MODEL,
        "rounds": ROUNDS,
        "cases_per_round": len(cases),
        "targets_per_arm": len(cases) * ROUNDS,
        "decode_tokens": 1,
        "prefetch": False,
        "lifecycle": (
            "fresh server per round; each natural preceding source is "
            "consumed by exactly one target"
        ),
        "arms": {
            "dense": "no cross-request KV reuse",
            "general_dual_4k": (
                "lossless ordinary prefix plus complete-overlap tail 4096"
            ),
            "coding_dual_v8_tail": (
                "same ordinary prefix plus complete-overlap tail 6144"
            ),
            "coding_full_v9": (
                "same ordinary prefix plus the full successful read-only "
                "coding-evidence overlap"
            ),
        },
        "opportunity_audit": {
            "full_evidence_tokens": {
                case["case_id"]: case["full_evidence_tokens"]
                for case in cases
            },
            "v9_marginal_tokens_above_v8_total": sum(marginal),
            "cases_with_v9_marginal_tokens": sum(value > 0 for value in marginal),
        },
        "gate": {
            "v9_vs_v8_mean_ttft_reduction_percent_min": 1.0,
            "v9_vs_v8_case_mean_win_fraction_min": 0.60,
            "v9_cache_ready_saving_percent_vs_dense_min": (
                NATIVE_CACHE_READY_REFERENCE["cacheblend"]
            ),
            "stretch_v9_cache_ready_saving_percent_vs_dense": (
                NATIVE_CACHE_READY_REFERENCE["kvcomm"]
            ),
            "expected_copy_events_per_reuse_arm": len(cases) * ROUNDS,
            "fallback_events_max": 0,
        },
        "inputs": {
            "cases_sha256": sha256_file(cases_path),
            "executor_cases_sha256": sha256_file(executor_cases_path),
            "source_cases_sha256": sha256_file(SOURCE_CASES),
            "opportunity_audit_sha256": sha256_file(OPPORTUNITY_AUDIT),
            "manifest_sha256": manifest_hashes,
            "treatment_source_sha256": {
                str(path.relative_to(Path(__file__).parents[2])): sha256_file(
                    path
                )
                for path in _source_paths()
            },
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "existing_preregistration_thresholds_modified": False,
        },
        "status": "REGISTERED_BEFORE_COLD_GPU_RUN",
    }
    write_json(output / "V9_REGISTRATION.json", registration)
    return registration


def summarize(output: Path) -> dict[str, Any]:
    cases = read_json(output / "V9_CASES.json")["cases"]
    raw = {
        arm: read_json(output / "generations" / f"{arm}.json")
        for arm in ARMS
    }
    arms: dict[str, Any] = {}
    paired: dict[str, dict[tuple[int, str], float]] = {}
    for arm, value in raw.items():
        ttfts = [float(row["ttft_ms"]) for row in value["target_rows"]]
        copies = [
            row
            for row in value["ledger_rows"]
            if row.get("event") == "target_copied"
        ]
        fallbacks = [
            row
            for row in value["ledger_rows"]
            if row.get("event") == "target_fallback"
        ]
        builds = [
            row
            for row in value["ledger_rows"]
            if row.get("event")
            in ("source_materialized", "source_materialized_host")
        ]
        arms[arm] = {
            "targets": len(ttfts),
            "median_ttft_ms": statistics.median(ttfts),
            "mean_ttft_ms": statistics.mean(ttfts),
            "p95_ttft_ms": percentile(ttfts, 0.95),
            "sum_ttft_ms": sum(ttfts),
            "copy_events": len(copies),
            "fallback_events": len(fallbacks),
            "copied_tokens_total": sum(
                int(row.get("copied_k_tokens", 0)) for row in copies
            ),
            "source_materialize_total_ms": sum(
                float(row["materialize_ms"]) for row in builds
            ),
            "case_mean_ttft_ms": {
                case["case_id"]: statistics.mean(
                    float(row["ttft_ms"])
                    for row in value["target_rows"]
                    if row["case_id"] == case["case_id"]
                )
                for case in cases
            },
        }
        paired[arm] = {
            (int(row["round"]), row["case_id"]): float(row["ttft_ms"])
            for row in value["target_rows"]
        }

    comparisons = {}
    for control in ("general_dual_4k", "coding_dual_v8_tail"):
        treatment = arms["coding_full_v9"]
        base = arms[control]
        wins = sum(
            treatment["case_mean_ttft_ms"][case["case_id"]]
            < base["case_mean_ttft_ms"][case["case_id"]]
            for case in cases
        )
        comparisons[f"v9_vs_{control}"] = {
            "median_ttft_reduction_percent": 100 * (
                1 - treatment["median_ttft_ms"] / base["median_ttft_ms"]
            ),
            "mean_ttft_reduction_percent": 100 * (
                1 - treatment["mean_ttft_ms"] / base["mean_ttft_ms"]
            ),
            "p95_ttft_reduction_percent": 100 * (
                1 - treatment["p95_ttft_ms"] / base["p95_ttft_ms"]
            ),
            "case_mean_wins": wins,
            "case_mean_win_fraction": wins / len(cases),
        }

    cache_ready = {}
    n1 = {}
    for arm in REUSE_ARMS:
        build_per_target = (
            arms[arm]["source_materialize_total_ms"] / arms[arm]["targets"]
        )
        cache_ready[arm] = statistics.mean(
            100 * (1 - paired[arm][key] / dense_ttft)
            for key, dense_ttft in paired["dense"].items()
        )
        n1[arm] = statistics.mean(
            100
            * (
                1
                - (paired[arm][key] + build_per_target) / dense_ttft
            )
            for key, dense_ttft in paired["dense"].items()
        )

    expected = len(cases) * ROUNDS
    v9_vs_v8 = comparisons["v9_vs_coding_dual_v8_tail"]
    gate = {
        "v9_vs_v8_mean_passed": (
            v9_vs_v8["mean_ttft_reduction_percent"] >= 1.0
        ),
        "v9_vs_v8_case_wins_passed": (
            v9_vs_v8["case_mean_win_fraction"] >= 0.60
        ),
        "cacheblend_cache_ready_reference_passed": (
            cache_ready["coding_full_v9"]
            >= NATIVE_CACHE_READY_REFERENCE["cacheblend"]
        ),
        "kvcomm_cache_ready_stretch_passed": (
            cache_ready["coding_full_v9"]
            >= NATIVE_CACHE_READY_REFERENCE["kvcomm"]
        ),
        "copy_events_passed": all(
            arms[arm]["copy_events"] == expected for arm in REUSE_ARMS
        ),
        "fallback_passed": all(
            arms[arm]["fallback_events"] == 0 for arm in REUSE_ARMS
        ),
    }
    result = {
        "classification": (
            "independent-round one-target-per-source V9 speed confirmation; "
            "native percentages are external reference lines only"
        ),
        "prefetch": False,
        "arms": arms,
        "comparisons": comparisons,
        "paired_cache_ready_saving_percent_vs_dense": cache_ready,
        "paired_n1_materialization_inclusive_saving_percent_vs_dense": n1,
        "native_cache_ready_reference_percent": (
            NATIVE_CACHE_READY_REFERENCE
        ),
        "gate": {
            **gate,
            "promotion_passed": all(
                value
                for key, value in gate.items()
                if key != "kvcomm_cache_ready_stretch_passed"
            ),
        },
    }
    write_json(output / "V9_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run-arm")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--port", type=int, default=33340)
    sub.add_parser("summarize")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "run-arm":
        result = run_cold_arm(output, args.arm, args.port)
    else:
        result = summarize(output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
