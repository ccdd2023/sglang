#!/usr/bin/env python3
"""Fixed-request motivation experiment for dual-island coding reuse V8.

V8 composes two forms of KV reuse without prefetch:

1. lossless ordinary Radix reuse for the stable prompt prefix; and
2. shifted copy-and-RoPE reuse for the retained coding-history middle.

The matched General control uses the same prefix mechanism and a 4096-token
middle cap.  V8 changes only the middle cap on coding-evidence opportunities.
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
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_coding_dual_island_v8_20260727"
ARMS = ("dense", "general_dual_4k", "coding_dual_v8")
REUSE_ARMS = ARMS[1:]
REPETITIONS = 5
KVCOMM_CACHE_READY_SAVING_PERCENT = 88.31062959906316
CACHEBLEND_CACHE_READY_SAVING_PERCENT = 79.01429315136923
KVCOMM_N4_SAVING_PERCENT = 81.27712077018771
CACHEBLEND_N4_SAVING_PERCENT = 17.85264849288383


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def prepare(output: Path) -> dict[str, Any]:
    source_cases_path = SOURCE_REPLAY / "REPLAY_CASES.json"
    source = read_json(source_cases_path)
    cases = source["cases"]
    cases_path = output / "V8_CASES.json"
    write_json(
        cases_path,
        {
            "source": str(source_cases_path),
            "cases": cases,
        },
    )
    manifest_hashes = {}
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
            row["target_uses"] = REPETITIONS
            rows.append(row)
        path = output / "manifests" / f"{arm}.json"
        write_json(
            path,
            {
                "cache_dtype": "bfloat16",
                "cases": rows,
                "lease_ttl_s": 900,
                "ledger_path": str(
                    output / "server" / arm / "EXACT_LEDGER.jsonl"
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
        manifest_hashes[arm] = sha256_file(path)

    registration = {
        "registered_before_gpu": True,
        "classification": (
            "fixed-request V8 mechanism motivation; causal speed diagnostic, "
            "not agent accuracy or same-workload SOTA confirmation"
        ),
        "model": MODEL,
        "cases": len(cases),
        "target_repetitions": REPETITIONS,
        "decode_tokens": 1,
        "prefetch": False,
        "arms": {
            "dense": "no cross-request KV reuse",
            "general_dual_4k": (
                "lossless Radix prefix plus shifted middle capped at 4096"
            ),
            "coding_dual_v8": (
                "same lossless prefix plus coding-evidence middle at "
                "the natural 5.3K--6.1K length"
            ),
        },
        "motivation_gate": {
            "v8_vs_general_median_ttft_reduction_percent_min": 5.0,
            "v8_vs_general_p95_ttft_reduction_percent_min": 5.0,
            "v8_vs_general_case_median_win_fraction_min": 0.70,
            "expected_copy_events_per_reuse_arm": len(cases) * REPETITIONS,
            "fallback_events_max": 0,
        },
        "external_native_reference_not_same_workload": {
            "kvcomm_cache_ready_saving_percent": (
                KVCOMM_CACHE_READY_SAVING_PERCENT
            ),
            "cacheblend_cache_ready_saving_percent": (
                CACHEBLEND_CACHE_READY_SAVING_PERCENT
            ),
            "kvcomm_n4_saving_percent": KVCOMM_N4_SAVING_PERCENT,
            "cacheblend_n4_saving_percent": CACHEBLEND_N4_SAVING_PERCENT,
        },
        "inputs": {
            "cases_sha256": sha256_file(cases_path),
            "source_cases_sha256": sha256_file(source_cases_path),
            "manifest_sha256": manifest_hashes,
            "runner_sha256": sha256_file(Path(__file__)),
            "controller_sha256": sha256_file(
                Path(__file__).parents[2]
                / "python/sglang/srt/mem_cache/kvcomm_exact.py"
            ),
            "radix_cache_sha256": sha256_file(
                Path(__file__).parents[2]
                / "python/sglang/srt/mem_cache/radix_cache.py"
            ),
            "schedule_policy_sha256": sha256_file(
                Path(__file__).parents[2]
                / "python/sglang/srt/managers/schedule_policy.py"
            ),
            "schedule_batch_sha256": sha256_file(
                Path(__file__).parents[2]
                / "python/sglang/srt/managers/schedule_batch.py"
            ),
            "mem_cache_common_sha256": sha256_file(
                Path(__file__).parents[2]
                / "python/sglang/srt/mem_cache/common.py"
            ),
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "existing_preregistration_thresholds_modified": False,
        },
        "status": "REGISTERED_BEFORE_V8_GPU_RUN",
    }
    write_json(output / "V8_REGISTRATION.json", registration)
    return registration


def run_arm(output: Path, arm: str, port: int) -> dict[str, Any]:
    cases = read_json(output / "V8_CASES.json")["cases"]
    process, stream, base_url = launch_server(
        output=output,
        arm=arm,
        port=port,
    )
    source_rows = []
    target_rows = []
    try:
        for case in cases:
            source_rows.append(
                {
                    **generate(
                        base_url=base_url,
                        input_ids=case["source_input_ids"],
                        key=f"v8-source-{arm}-{case['case_id']}",
                        max_new_tokens=1,
                        stream=True,
                    ),
                    "case_id": case["case_id"],
                }
            )
            for repetition in range(REPETITIONS):
                target_rows.append(
                    {
                        **generate(
                            base_url=base_url,
                            input_ids=case["target_input_ids"],
                            key=(
                                f"v8-target-{arm}-{case['case_id']}-"
                                f"r{repetition}"
                            ),
                            max_new_tokens=1,
                            stream=True,
                        ),
                        "arm": arm,
                        "case_id": case["case_id"],
                        "repetition": repetition,
                    }
                )
    finally:
        stop_server(process, stream)
    ledger_path = output / "server" / arm / "EXACT_LEDGER.jsonl"
    ledger = (
        [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if ledger_path.exists()
        else []
    )
    result = {
        "arm": arm,
        "source_rows": source_rows,
        "target_rows": target_rows,
        "ledger_rows": ledger,
        "status": "complete",
    }
    write_json(output / "generations" / f"{arm}.json", result)
    return {
        "arm": arm,
        "sources": len(source_rows),
        "targets": len(target_rows),
        "copy_events": sum(
            row.get("event") == "target_copied" for row in ledger
        ),
        "fallback_events": sum(
            row.get("event") == "target_fallback" for row in ledger
        ),
        "status": "complete",
    }


def _paired_mean_saving(
    dense: dict[tuple[str, int], float],
    treatment: dict[tuple[str, int], float],
) -> float:
    if dense.keys() != treatment.keys():
        raise ValueError("dense/treatment target identities differ")
    return statistics.mean(
        100 * (1 - treatment[key] / dense[key]) for key in dense
    )


def summarize(output: Path) -> dict[str, Any]:
    cases = read_json(output / "V8_CASES.json")["cases"]
    values = {
        arm: read_json(output / "generations" / f"{arm}.json")
        for arm in ARMS
    }
    arms: dict[str, Any] = {}
    paired: dict[str, dict[tuple[str, int], float]] = {}
    for arm, value in values.items():
        ttfts = [float(row["ttft_ms"]) for row in value["target_rows"]]
        ledger = value["ledger_rows"]
        copies = [
            row for row in ledger if row.get("event") == "target_copied"
        ]
        fallbacks = [
            row for row in ledger if row.get("event") == "target_fallback"
        ]
        builds = [
            row
            for row in ledger
            if row.get("event")
            in ("source_materialized", "source_materialized_host")
        ]
        arms[arm] = {
            "target_rows": len(ttfts),
            "median_ttft_ms": statistics.median(ttfts),
            "p95_ttft_ms": percentile(ttfts, 0.95),
            "sum_ttft_ms": sum(ttfts),
            "copy_events": len(copies),
            "fallback_events": len(fallbacks),
            "copied_tokens": sum(
                int(row["copied_k_tokens"]) for row in copies
            ),
            "source_materializations": len(builds),
            "source_materialize_total_ms": sum(
                float(row["materialize_ms"]) for row in builds
            ),
            "case_median_ttft_ms": {
                case["case_id"]: statistics.median(
                    float(row["ttft_ms"])
                    for row in value["target_rows"]
                    if row["case_id"] == case["case_id"]
                )
                for case in cases
            },
        }
        paired[arm] = {
            (row["case_id"], int(row["repetition"])): float(row["ttft_ms"])
            for row in value["target_rows"]
        }

    general = arms["general_dual_4k"]
    v8 = arms["coding_dual_v8"]
    wins = sum(
        v8["case_median_ttft_ms"][case["case_id"]]
        < general["case_median_ttft_ms"][case["case_id"]]
        for case in cases
    )
    v8_vs_general = {
        "median_ttft_reduction_percent": 100
        * (1 - v8["median_ttft_ms"] / general["median_ttft_ms"]),
        "p95_ttft_reduction_percent": 100
        * (1 - v8["p95_ttft_ms"] / general["p95_ttft_ms"]),
        "sum_ttft_reduction_percent": 100
        * (1 - v8["sum_ttft_ms"] / general["sum_ttft_ms"]),
        "case_median_wins": wins,
        "case_median_win_fraction": wins / len(cases),
    }
    cache_ready = {
        arm: _paired_mean_saving(paired["dense"], paired[arm])
        for arm in REUSE_ARMS
    }
    n4 = {}
    for arm in REUSE_ARMS:
        per_case = []
        build_total = arms[arm]["source_materialize_total_ms"]
        build_per_case = build_total / len(cases)
        for case in cases:
            case_id = case["case_id"]
            dense_mean = statistics.mean(
                paired["dense"][(case_id, repetition)]
                for repetition in range(4)
            )
            reuse_mean = statistics.mean(
                paired[arm][(case_id, repetition)]
                for repetition in range(4)
            )
            per_case.append(
                100
                * (
                    1
                    - (reuse_mean + build_per_case / 4)
                    / dense_mean
                )
            )
        n4[arm] = statistics.mean(per_case)

    expected = len(cases) * REPETITIONS
    gate = {
        "median_passed": (
            v8_vs_general["median_ttft_reduction_percent"] >= 5
        ),
        "p95_passed": v8_vs_general["p95_ttft_reduction_percent"] >= 5,
        "case_wins_passed": (
            v8_vs_general["case_median_win_fraction"] >= 0.70
        ),
        "copy_events_passed": all(
            arms[arm]["copy_events"] == expected for arm in REUSE_ARMS
        ),
        "fallback_passed": all(
            arms[arm]["fallback_events"] == 0 for arm in REUSE_ARMS
        ),
    }
    external = {
        "classification": "reference only; workloads and engines differ",
        "v8_cache_ready_saving_percent": cache_ready["coding_dual_v8"],
        "v8_n4_saving_percent": n4["coding_dual_v8"],
        "exceeds_cacheblend_cache_ready_reference": (
            cache_ready["coding_dual_v8"]
            > CACHEBLEND_CACHE_READY_SAVING_PERCENT
        ),
        "exceeds_kvcomm_cache_ready_reference": (
            cache_ready["coding_dual_v8"]
            > KVCOMM_CACHE_READY_SAVING_PERCENT
        ),
        "exceeds_cacheblend_n4_reference": (
            n4["coding_dual_v8"] > CACHEBLEND_N4_SAVING_PERCENT
        ),
        "exceeds_kvcomm_n4_reference": (
            n4["coding_dual_v8"] > KVCOMM_N4_SAVING_PERCENT
        ),
    }
    result = {
        "classification": (
            "fixed-request V8 mechanism motivation; causal speed diagnostic, "
            "not agent accuracy or same-workload SOTA confirmation"
        ),
        "cases": len(cases),
        "arms": arms,
        "v8_vs_general": v8_vs_general,
        "paired_cache_ready_saving_percent_vs_dense": cache_ready,
        "n4_build_inclusive_saving_percent_vs_dense": n4,
        "external_native_reference": external,
        "gate": {**gate, "overall_passed": all(gate.values())},
        "prefetch": False,
    }
    write_json(output / "V8_RESULT.json", result)
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
