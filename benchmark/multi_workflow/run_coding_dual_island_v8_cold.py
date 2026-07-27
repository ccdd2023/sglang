#!/usr/bin/env python3
"""Independent-round, one-target-per-source confirmation for V8.

Each round launches a fresh server and evaluates seven hash-verified natural
source/target pairs exactly once.  This matches the rolling agent lifecycle and
prevents repeated identical targets from warming their complete dense prefix.
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
    generate,
    launch_server,
    manifest_case,
    sha256_file,
    stop_server,
    write_json,
)
from benchmark.multi_workflow.run_coding_evidence_payoff_paired import (
    read_json,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
SOURCE_REPLAY = (
    ARTIFACTS / "impactkv_coding_evidence_replay_v7_20260726"
)
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_coding_dual_island_v8_cold_20260727"
)
ARMS = ("dense", "general_dual_4k", "coding_dual_v8")
REUSE_ARMS = ARMS[1:]
ROUNDS = 3


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def _source_paths() -> list[Path]:
    root = Path(__file__).parents[2]
    return [
        Path(__file__),
        root / "python/sglang/srt/mem_cache/kvcomm_exact.py",
        root / "python/sglang/srt/mem_cache/radix_cache.py",
        root / "python/sglang/srt/managers/schedule_policy.py",
        root / "python/sglang/srt/managers/schedule_batch.py",
        root / "python/sglang/srt/mem_cache/common.py",
    ]


def prepare(output: Path) -> dict[str, Any]:
    source_cases_path = SOURCE_REPLAY / "REPLAY_CASES.json"
    cases = read_json(source_cases_path)["cases"]
    cases_path = output / "COLD_CASES.json"
    write_json(cases_path, {"cases": cases})
    manifest_hashes: dict[str, dict[str, str]] = {}
    for round_index in range(ROUNDS):
        round_dir = output / "rounds" / f"r{round_index}"
        manifest_hashes[str(round_index)] = {}
        for arm in REUSE_ARMS:
            rows = []
            for case in cases:
                wide_span = {
                    "source_start": case["source_start"],
                    "target_start": case["target_start"],
                    "length": case["v7_tokens"],
                }
                span = (
                    capped_tail(wide_span, 4096)
                    if arm == "general_dual_4k"
                    else wide_span
                )
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
                        round_dir
                        / "server"
                        / arm
                        / "EXACT_LEDGER.jsonl"
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

    registration = {
        "registered_before_gpu": True,
        "classification": (
            "independent-round one-target-per-source V8 speed confirmation; "
            "not task accuracy or same-workload SOTA evidence"
        ),
        "model": MODEL,
        "cases_per_round": len(cases),
        "rounds": ROUNDS,
        "targets_per_arm": len(cases) * ROUNDS,
        "decode_tokens": 1,
        "prefetch": False,
        "lifecycle": (
            "fresh server per round; each natural source is followed by "
            "exactly one target"
        ),
        "arms": {
            "dense": "no cross-request KV reuse",
            "general_dual_4k": (
                "lossless prefix plus shifted middle capped at 4096"
            ),
            "coding_dual_v8": (
                "same prefix plus natural coding-evidence middle 5.3K--6.1K"
            ),
        },
        "gate": {
            "v8_vs_general_median_ttft_reduction_percent_min": 5.0,
            "v8_vs_general_mean_ttft_reduction_percent_min": 5.0,
            "v8_vs_general_case_mean_win_fraction_min": 0.70,
            "expected_copy_events_per_reuse_arm": len(cases) * ROUNDS,
            "fallback_events_max": 0,
        },
        "inputs": {
            "cases_sha256": sha256_file(cases_path),
            "source_cases_sha256": sha256_file(source_cases_path),
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
    write_json(output / "COLD_REGISTRATION.json", registration)
    return registration


def run_arm(output: Path, arm: str, port: int) -> dict[str, Any]:
    cases = read_json(output / "COLD_CASES.json")["cases"]
    all_sources = []
    all_targets = []
    all_ledger = []
    for round_index in range(ROUNDS):
        round_dir = output / "rounds" / f"r{round_index}"
        process, stream, base_url = launch_server(
            output=round_dir,
            arm=arm,
            port=port,
        )
        try:
            for case in cases:
                all_sources.append(
                    {
                        **generate(
                            base_url=base_url,
                            input_ids=case["source_input_ids"],
                            key=(
                                f"cold-source-r{round_index}-{arm}-"
                                f"{case['case_id']}"
                            ),
                            max_new_tokens=1,
                            stream=True,
                        ),
                        "case_id": case["case_id"],
                        "round": round_index,
                    }
                )
                all_targets.append(
                    {
                        **generate(
                            base_url=base_url,
                            input_ids=case["target_input_ids"],
                            key=(
                                f"cold-target-r{round_index}-{arm}-"
                                f"{case['case_id']}"
                            ),
                            max_new_tokens=1,
                            stream=True,
                        ),
                        "arm": arm,
                        "case_id": case["case_id"],
                        "round": round_index,
                    }
                )
        finally:
            stop_server(process, stream)
        ledger_path = (
            round_dir / "server" / arm / "EXACT_LEDGER.jsonl"
        )
        if ledger_path.exists():
            all_ledger.extend(
                {
                    **json.loads(line),
                    "round": round_index,
                }
                for line in ledger_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )
    result = {
        "arm": arm,
        "source_rows": all_sources,
        "target_rows": all_targets,
        "ledger_rows": all_ledger,
        "status": "complete",
    }
    write_json(output / "generations" / f"{arm}.json", result)
    return {
        "arm": arm,
        "targets": len(all_targets),
        "copy_events": sum(
            row.get("event") == "target_copied" for row in all_ledger
        ),
        "fallback_events": sum(
            row.get("event") == "target_fallback" for row in all_ledger
        ),
        "status": "complete",
    }


def summarize(output: Path) -> dict[str, Any]:
    cases = read_json(output / "COLD_CASES.json")["cases"]
    values = {
        arm: read_json(output / "generations" / f"{arm}.json")
        for arm in ARMS
    }
    arms: dict[str, Any] = {}
    paired: dict[str, dict[tuple[int, str], float]] = {}
    for arm, value in values.items():
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

    general = arms["general_dual_4k"]
    v8 = arms["coding_dual_v8"]
    wins = sum(
        v8["case_mean_ttft_ms"][case["case_id"]]
        < general["case_mean_ttft_ms"][case["case_id"]]
        for case in cases
    )
    comparison = {
        "median_ttft_reduction_percent": 100
        * (1 - v8["median_ttft_ms"] / general["median_ttft_ms"]),
        "mean_ttft_reduction_percent": 100
        * (1 - v8["mean_ttft_ms"] / general["mean_ttft_ms"]),
        "p95_ttft_reduction_percent": 100
        * (1 - v8["p95_ttft_ms"] / general["p95_ttft_ms"]),
        "case_mean_wins": wins,
        "case_mean_win_fraction": wins / len(cases),
    }
    cache_ready = {}
    build_inclusive_n1 = {}
    for arm in REUSE_ARMS:
        savings = []
        inclusive = []
        build_per_target = (
            arms[arm]["source_materialize_total_ms"]
            / arms[arm]["targets"]
        )
        for key, dense_ttft in paired["dense"].items():
            reuse_ttft = paired[arm][key]
            savings.append(100 * (1 - reuse_ttft / dense_ttft))
            inclusive.append(
                100
                * (1 - (reuse_ttft + build_per_target) / dense_ttft)
            )
        cache_ready[arm] = statistics.mean(savings)
        build_inclusive_n1[arm] = statistics.mean(inclusive)

    expected = len(cases) * ROUNDS
    gate = {
        "median_passed": comparison["median_ttft_reduction_percent"] >= 5,
        "mean_passed": comparison["mean_ttft_reduction_percent"] >= 5,
        "case_wins_passed": comparison["case_mean_win_fraction"] >= 0.70,
        "copy_events_passed": all(
            arms[arm]["copy_events"] == expected for arm in REUSE_ARMS
        ),
        "fallback_passed": all(
            arms[arm]["fallback_events"] == 0 for arm in REUSE_ARMS
        ),
    }
    result = {
        "classification": (
            "independent-round one-target-per-source V8 speed confirmation; "
            "not task accuracy or same-workload SOTA evidence"
        ),
        "arms": arms,
        "v8_vs_general": comparison,
        "paired_cache_ready_saving_percent_vs_dense": cache_ready,
        "paired_n1_materialization_inclusive_saving_percent_vs_dense": (
            build_inclusive_n1
        ),
        "gate": {**gate, "overall_passed": all(gate.values())},
        "prefetch": False,
    }
    write_json(output / "COLD_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    run = subparsers.add_parser("run-arm")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--port", type=int, default=33300)
    subparsers.add_parser("summarize")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "run-arm":
        result = run_arm(output, args.arm, args.port)
    else:
        result = summarize(output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
