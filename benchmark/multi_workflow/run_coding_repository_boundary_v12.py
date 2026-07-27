#!/usr/bin/env python3
"""Confirm target-position-first coding reuse at the repository boundary.

The first target copies the shifted repository KV from its frozen source.
Later real targets first consume exact target-position Radix KV up to the
coding-labelled end of repository_context.  If only part is present, the
controller copies only the missing repository tail.  Everything after the
repository boundary remains dense.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    capped_tail,
    generate,
    manifest_case,
    sha256_file,
    stop_server,
    write_json,
)
from benchmark.multi_workflow.run_coding_header_hot_v11 import (
    generate_detailed,
)
from benchmark.multi_workflow.run_coding_native_workload_v10 import (
    CACHEBLEND_DENSE,
    CACHEBLEND_REUSE,
    GENERAL_CAP,
    KVCOMM_DENSE,
    KVCOMM_REUSE,
    MEASURED_ROUNDS,
    MODEL,
    NATIVE_REFERENCES,
    PROJECT,
    TOTAL_ROUNDS,
    WARMUPS,
    WORKLOAD,
    flush_cache,
    launch_server,
    mean_saving,
    native_result,
    percentile,
    prepare_cases,
    read_json,
    read_jsonl,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_coding_repository_boundary_v12_20260727"
)
ARMS = ("dense", "general_dual_4k", "coding_repo_boundary_v12")
REUSE_ARMS = ARMS[1:]
CACHEBLEND_8K_SAVING = 84.3322879916783
CACHEBLEND_8K_MEAN_TTFT_MS = 57.21641778945923
KVCOMM_2K4K_MEAN_TTFT_MS = 86.69399128411897


def treatment_sources() -> list[Path]:
    return [
        Path(__file__),
        PROJECT / "python/sglang/srt/mem_cache/kvcomm_exact.py",
        PROJECT / "python/sglang/srt/mem_cache/radix_cache.py",
        PROJECT / "python/sglang/srt/managers/schedule_batch.py",
        PROJECT / "python/sglang/srt/mem_cache/common.py",
    ]


def prepare(output: Path) -> dict[str, Any]:
    cases = prepare_cases()
    if {int(case["target_start"]) for case in cases} != {61}:
        raise ValueError("V12 requires the frozen 61-token target header")
    cases_path = output / "V12_CASES.json"
    write_json(cases_path, {"cases": cases})
    manifest_hashes = {}
    for arm in REUSE_ARMS:
        rows = []
        for case in cases:
            full_span = {
                "source_start": case["source_start"],
                "target_start": case["target_start"],
                "length": case["repository_tokens"],
            }
            span = (
                capped_tail(full_span, GENERAL_CAP)
                if arm == "general_dual_4k"
                else full_span
            )
            row = manifest_case(
                case_id=case["case_id"],
                policy_label=arm,
                source_ids=case["source_input_ids"],
                target_ids=case["target_input_ids"],
                span=span,
            )
            row["target_uses"] = TOTAL_ROUNDS
            if arm == "coding_repo_boundary_v12":
                row["allow_target_prefix_bypass"] = True
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
                    "base": 1_000_000,
                    "is_neox_style": True,
                    "rotary_dim": 128,
                },
                "version": 2,
            },
        )
        manifest_hashes[arm] = sha256_file(path)

    registration = {
        "date": "2026-07-27",
        "registered_before_gpu": True,
        "classification": (
            "same-model, same-token repository-boundary coding reuse "
            "confirmation on the native formal latency workload"
        ),
        "model": MODEL,
        "formal_cases": len(cases),
        "protocol": {
            "concurrency": 1,
            "decode_tokens": 1,
            "measured_rounds": MEASURED_ROUNDS,
            "one_frozen_source_request_per_case": True,
            "source_targets": TOTAL_ROUNDS,
            "target_prefetch": False,
            "warmups_per_case": WARMUPS,
        },
        "method": {
            "general_dual_4k": (
                "ordinary target-position prefix capped at shifted-tail start, "
                "then source-copy the repository tail 4K"
            ),
            "coding_repo_boundary_v12": (
                "ordinary target-position KV up to the coding-labelled end of "
                "repository_context; copy only an uncached repository tail; "
                "force the current task suffix dense"
            ),
        },
        "frozen_gates": {
            "cacheblend_8k_cache_ready_saving_percent_min": (
                CACHEBLEND_8K_SAVING
            ),
            "cacheblend_8k_absolute_mean_ttft_ms_max": (
                CACHEBLEND_8K_MEAN_TTFT_MS
            ),
            "coding_all_context_cache_ready_saving_percent_min": (
                NATIVE_REFERENCES[
                    "cacheblend_all_context_cache_ready_saving_percent"
                ]
            ),
            "coding_mean_ttft_reduction_vs_general_percent_min": 5.0,
            "fallback_events_max": 0,
            "kvcomm_2k4k_absolute_mean_ttft_ms_max": (
                KVCOMM_2K4K_MEAN_TTFT_MS
            ),
            "stretch_coding_2k4k_cache_ready_saving_percent": (
                NATIVE_REFERENCES[
                    "kvcomm_2k4k_cache_ready_saving_percent"
                ]
            ),
            "target_prefix_copy_or_bypass_events": len(cases) * TOTAL_ROUNDS,
        },
        "native_references": {
            **NATIVE_REFERENCES,
            "cacheblend_8k_cache_ready_saving_percent": (
                CACHEBLEND_8K_SAVING
            ),
            "cacheblend_8k_mean_ttft_ms": CACHEBLEND_8K_MEAN_TTFT_MS,
            "kvcomm_2k4k_mean_ttft_ms": KVCOMM_2K4K_MEAN_TTFT_MS,
        },
        "inputs": {
            "cases_sha256": sha256_file(cases_path),
            "manifest_sha256": manifest_hashes,
            "treatment_source_sha256": {
                str(path.relative_to(PROJECT)): sha256_file(path)
                for path in treatment_sources()
            },
            "workload_sha256": sha256_file(WORKLOAD),
        },
        "protected": {
            "existing_preregistration_thresholds_modified": False,
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
        },
        "status": "REGISTERED_BEFORE_GPU",
    }
    write_json(output / "V12_REGISTRATION.json", registration)
    return registration


def run_arm(output: Path, arm: str, port: int) -> dict[str, Any]:
    if not (output / "V12_REGISTRATION.json").exists():
        raise FileNotFoundError("run prepare before any GPU arm")
    cases = read_json(output / "V12_CASES.json")["cases"]
    result_path = output / "generations" / f"{arm}.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    ledger_path = output / "server" / arm / "EXACT_LEDGER.jsonl"
    if ledger_path.exists():
        raise FileExistsError(f"refusing to reuse {ledger_path}")
    process, stream, base_url = launch_server(
        output=output,
        arm=arm,
        port=port,
    )
    sources = []
    targets = []
    try:
        generate(
            base_url=base_url,
            input_ids=[100] * 128,
            key=f"v12-unrelated-server-warmup-{arm}",
            max_new_tokens=1,
            stream=True,
        )
        flush_cache(base_url)
        for case in cases:
            if arm != "dense":
                sources.append(
                    {
                        **generate(
                            base_url=base_url,
                            input_ids=case["source_input_ids"],
                            key=f"v12-source-{arm}-{case['case_id']}",
                            max_new_tokens=1,
                            stream=False,
                        ),
                        "case_id": case["case_id"],
                    }
                )
            for index in range(TOTAL_ROUNDS):
                targets.append(
                    {
                        **generate_detailed(
                            base_url=base_url,
                            input_ids=case["target_input_ids"],
                            key=(
                                f"v12-target-{arm}-{case['case_id']}-"
                                f"{index}"
                            ),
                        ),
                        "arm": arm,
                        "case_id": case["case_id"],
                        "context_bucket": case["context_bucket"],
                        "round_index": max(0, index - WARMUPS),
                        "token_ids_sha256": case[
                            "native_token_ids_sha256"
                        ],
                        "warmup": index < WARMUPS,
                    }
                )
    finally:
        stop_server(process, stream)
    ledger = read_jsonl(ledger_path) if ledger_path.exists() else []
    result = {
        "arm": arm,
        "ledger_rows": ledger,
        "source_rows": sources,
        "status": "complete",
        "target_rows": targets,
    }
    write_json(result_path, result)
    return {
        "arm": arm,
        "copy_events": sum(
            row.get("event") == "target_copied" for row in ledger
        ),
        "fallback_events": sum(
            row.get("event") == "target_fallback" for row in ledger
        ),
        "prefix_bypass_events": sum(
            row.get("event") == "target_prefix_bypass" for row in ledger
        ),
        "status": "complete",
        "targets": len(targets),
    }


def measured(value: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in value["target_rows"] if not row["warmup"]]


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["case_id"]), int(row["round_index"])


def native_unique_mean_ttft(path: Path) -> float:
    rows = [
        row
        for row in read_jsonl(path)
        if not bool(row["metadata"].get("warmup"))
    ]
    unique = {
        (str(row["case_id"]), int(row["round_index"])): float(row["ttft_ms"])
        for row in rows
    }
    return statistics.mean(unique.values())


def summarize(output: Path) -> dict[str, Any]:
    cases = read_json(output / "V12_CASES.json")["cases"]
    raw = {
        arm: read_json(output / "generations" / f"{arm}.json")
        for arm in ARMS
    }
    context = {
        str(case["case_id"]): int(case["context_bucket"]) for case in cases
    }
    selected = {
        "all": set(context),
        "2k4k": {case_id for case_id, size in context.items() if size <= 4096},
        "8k": {case_id for case_id, size in context.items() if size == 8192},
    }
    paired = {}
    arms = {}
    for arm, value in raw.items():
        rows = measured(value)
        values = [float(row["ttft_ms"]) for row in rows]
        paired[arm] = {
            row_key(row): float(row["ttft_ms"]) for row in rows
        }
        ledger = value["ledger_rows"]
        arms[arm] = {
            "copy_events": sum(
                row.get("event") == "target_copied" for row in ledger
            ),
            "fallback_events": sum(
                row.get("event") == "target_fallback" for row in ledger
            ),
            "mean_ttft_ms": statistics.mean(values),
            "median_ttft_ms": statistics.median(values),
            "p95_ttft_ms": percentile(values, 0.95),
            "prefix_bypass_events": sum(
                row.get("event") == "target_prefix_bypass"
                for row in ledger
            ),
            "targets": len(rows),
        }
    savings = {
        arm: {
            scope: mean_saving(paired["dense"], paired[arm], cases_in_scope)
            for scope, cases_in_scope in selected.items()
        }
        for arm in REUSE_ARMS
    }
    means = {
        arm: {
            scope: statistics.mean(
                value
                for (case_id, _), value in paired[arm].items()
                if case_id in cases_in_scope
            )
            for scope, cases_in_scope in selected.items()
        }
        for arm in ARMS
    }
    coding_vs_general = mean_saving(
        paired["general_dual_4k"],
        paired["coding_repo_boundary_v12"],
        selected["all"],
    )
    expected = len(cases) * TOTAL_ROUNDS
    coding_reuse_events = (
        arms["coding_repo_boundary_v12"]["copy_events"]
        + arms["coding_repo_boundary_v12"]["prefix_bypass_events"]
    )
    gates = {
        "cacheblend_8k_absolute_ttft_passed": (
            means["coding_repo_boundary_v12"]["8k"]
            <= CACHEBLEND_8K_MEAN_TTFT_MS
        ),
        "cacheblend_8k_saving_passed": (
            savings["coding_repo_boundary_v12"]["8k"]
            >= CACHEBLEND_8K_SAVING
        ),
        "cacheblend_all_saving_passed": (
            savings["coding_repo_boundary_v12"]["all"]
            >= NATIVE_REFERENCES[
                "cacheblend_all_context_cache_ready_saving_percent"
            ]
        ),
        "coding_vs_general_passed": coding_vs_general >= 5.0,
        "fallback_passed": all(
            arms[arm]["fallback_events"] == 0 for arm in REUSE_ARMS
        ),
        "kvcomm_2k4k_absolute_ttft_passed": (
            means["coding_repo_boundary_v12"]["2k4k"]
            <= KVCOMM_2K4K_MEAN_TTFT_MS
        ),
        "kvcomm_2k4k_saving_stretch_passed": (
            savings["coding_repo_boundary_v12"]["2k4k"]
            >= NATIVE_REFERENCES[
                "kvcomm_2k4k_cache_ready_saving_percent"
            ]
        ),
        "target_reuse_events_passed": coding_reuse_events == expected,
        "target_coverage_passed": all(
            arms[arm]["targets"] == len(cases) * MEASURED_ROUNDS
            for arm in ARMS
        ),
    }
    result = {
        "arms": arms,
        "coding_vs_general_mean_paired_ttft_reduction_percent": (
            coding_vs_general
        ),
        "gates": {
            **gates,
            "mechanism_overall_passed": all(
                gates[name]
                for name in (
                    "fallback_passed",
                    "target_reuse_events_passed",
                    "target_coverage_passed",
                )
            ),
            "sota_speed_goal_passed": (
                gates["cacheblend_8k_saving_passed"]
                and gates["kvcomm_2k4k_absolute_ttft_passed"]
            ),
        },
        "mean_ttft_ms_by_scope": means,
        "native_recomputed_from_validated_ledgers": {
            "cacheblend": native_result(CACHEBLEND_DENSE, CACHEBLEND_REUSE),
            "cacheblend_8k_mean_ttft_ms": CACHEBLEND_8K_MEAN_TTFT_MS,
            "kvcomm": native_result(KVCOMM_DENSE, KVCOMM_REUSE),
            "kvcomm_2k4k_unique_mean_ttft_ms": native_unique_mean_ttft(
                KVCOMM_REUSE
            ),
        },
        "paired_cache_ready_saving_percent_vs_sglang_dense": savings,
        "prefetch": False,
        "scope": (
            "same Qwen2.5-Coder-3B target IDs and native formal latency cases; "
            "accuracy is not measured"
        ),
    }
    write_json(output / "V12_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    run = subparsers.add_parser("run-arm")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--port", type=int, default=33400)
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
