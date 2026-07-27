#!/usr/bin/env python3
"""Amortize the shifted-island prefix phase with ordinary KV reuse.

V10 flushed before every source/target pair and proved that repository-full
coding reuse beats a fixed 4K General policy, but its target still needed two
prefill phases.  Every frozen native-frontier target has the same 61-token task
header.  V11 keeps that header in the ordinary Radix cache after a real target
has used it.  The exact controller caps ordinary matching at token 61, so even
identical repeated targets can never consume the shifted repository or suffix
as an accidental full-prefix hit.

Each frozen source is served once and its exact repository KV is consumed by
two warmup plus five measured targets.  There is no target prefetch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import requests

from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    capped_tail,
    generate,
    manifest_case,
    sha256_file,
    stop_server,
    write_json,
)
from benchmark.multi_workflow.run_coding_native_workload_v10 import (
    ARMS,
    CACHEBLEND_DENSE,
    CACHEBLEND_REUSE,
    GENERAL_CAP,
    KVCOMM_DENSE,
    KVCOMM_REUSE,
    MEASURED_ROUNDS,
    MODEL,
    NATIVE_REFERENCES,
    PROJECT,
    REUSE_ARMS,
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
DEFAULT_OUTPUT = ARTIFACTS / "impactkv_coding_header_hot_v11_20260727"


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
    header_lengths = {int(case["target_start"]) for case in cases}
    if header_lengths != {61}:
        raise ValueError(f"expected one 61-token target header: {header_lengths}")
    cases_path = output / "V11_CASES.json"
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
                policy_label=f"{arm}_header_hot",
                source_ids=case["source_input_ids"],
                target_ids=case["target_input_ids"],
                span=span,
            )
            row["target_uses"] = TOTAL_ROUNDS
            rows.append(row)
        manifest_path = output / "manifests" / f"{arm}.json"
        write_json(
            manifest_path,
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
        manifest_hashes[arm] = sha256_file(manifest_path)

    registration = {
        "date": "2026-07-27",
        "registered_before_gpu": True,
        "classification": (
            "same-model, same-target-token header-hot coding KV reuse "
            "confirmation on the native formal latency workload"
        ),
        "model": MODEL,
        "formal_cases": len(cases),
        "protocol": {
            "concurrency": 1,
            "decode_tokens": 1,
            "header_tokens": 61,
            "measured_rounds": MEASURED_ROUNDS,
            "one_frozen_source_request_per_case": True,
            "ordinary_prefix_match_cap": 61,
            "source_targets": TOTAL_ROUNDS,
            "target_prefetch": False,
            "warmups_per_case": WARMUPS,
        },
        "motivation": (
            "V10 copy time was only 5--6 ms, but a cold 61-token target "
            "header forced a separate first prefill phase. Reuse the identical "
            "header only after a real prior target has populated it."
        ),
        "arms": {
            "dense": "Radix disabled; seven independent dense target requests",
            "general_dual_4k": (
                "hot ordinary 61-token header plus shifted repository tail 4K"
            ),
            "coding_repo_v10": (
                "hot ordinary 61-token header plus complete shifted repository"
            ),
        },
        "frozen_gates": {
            "cached_prefix_tokens_after_first_target_min": 61,
            "coding_all_context_cache_ready_saving_vs_dense_percent_min": (
                NATIVE_REFERENCES[
                    "cacheblend_all_context_cache_ready_saving_percent"
                ]
            ),
            "coding_all_mean_ttft_reduction_vs_general_percent_min": 5.0,
            "copy_events_per_reuse_arm": len(cases) * TOTAL_ROUNDS,
            "fallback_events_max": 0,
            "stretch_coding_2k4k_cache_ready_saving_vs_dense_percent": (
                NATIVE_REFERENCES[
                    "kvcomm_2k4k_cache_ready_saving_percent"
                ]
            ),
        },
        "native_references": NATIVE_REFERENCES,
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
    write_json(output / "V11_REGISTRATION.json", registration)
    return registration


def generate_detailed(
    *,
    base_url: str,
    input_ids: list[int],
    key: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        base_url + "/generate",
        json={
            "extra_key": key,
            "input_ids": input_ids,
            "return_logprob": False,
            "sampling_params": {
                "ignore_eos": False,
                "max_new_tokens": 1,
                "temperature": 0,
            },
            "stream": True,
        },
        stream=True,
        timeout=900,
    )
    response.raise_for_status()
    value = None
    ttft_ms = math.inf
    for chunk in response.iter_lines(decode_unicode=True):
        if not chunk or not chunk.startswith("data:"):
            continue
        payload = chunk[5:].strip()
        if payload == "[DONE]":
            break
        value = json.loads(payload)
        if "error" in value:
            raise RuntimeError(value["error"])
        completion_tokens = int(
            value.get("meta_info", {}).get("completion_tokens", 0)
        )
        if math.isinf(ttft_ms) and completion_tokens:
            ttft_ms = 1000 * (time.perf_counter() - started)
    if value is None or math.isinf(ttft_ms):
        raise RuntimeError("empty generation stream")
    meta = value.get("meta_info", {})
    return {
        "cached_tokens": int(meta.get("cached_tokens", 0)),
        "completion_tokens": int(meta.get("completion_tokens", 0)),
        "elapsed_ms": 1000 * (time.perf_counter() - started),
        "finish_reason": meta.get("finish_reason"),
        "output_text": str(value.get("text") or ""),
        "ttft_ms": ttft_ms,
    }


def run_arm(output: Path, arm: str, port: int) -> dict[str, Any]:
    if not (output / "V11_REGISTRATION.json").exists():
        raise FileNotFoundError("run prepare before any GPU arm")
    cases = read_json(output / "V11_CASES.json")["cases"]
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
    source_rows = []
    target_rows = []
    try:
        # Warm an unrelated shape, then remove it. No target prefix is seeded.
        generate(
            base_url=base_url,
            input_ids=[100] * 128,
            key=f"v11-unrelated-server-warmup-{arm}",
            max_new_tokens=1,
            stream=True,
        )
        flush_cache(base_url)
        for case in cases:
            if arm != "dense":
                source_rows.append(
                    {
                        **generate(
                            base_url=base_url,
                            input_ids=case["source_input_ids"],
                            key=f"v11-source-{arm}-{case['case_id']}",
                            max_new_tokens=1,
                            stream=False,
                        ),
                        "case_id": case["case_id"],
                        "context_bucket": case["context_bucket"],
                    }
                )
            for index in range(TOTAL_ROUNDS):
                target_rows.append(
                    {
                        **generate_detailed(
                            base_url=base_url,
                            input_ids=case["target_input_ids"],
                            key=(
                                f"v11-target-{arm}-{case['case_id']}-"
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
    ledger_rows = read_jsonl(ledger_path) if ledger_path.exists() else []
    result = {
        "arm": arm,
        "ledger_rows": ledger_rows,
        "source_rows": source_rows,
        "status": "complete",
        "target_rows": target_rows,
    }
    write_json(result_path, result)
    return {
        "arm": arm,
        "copy_events": sum(
            row.get("event") == "target_copied" for row in ledger_rows
        ),
        "fallback_events": sum(
            row.get("event") == "target_fallback" for row in ledger_rows
        ),
        "measured_targets": sum(
            not row["warmup"] for row in target_rows
        ),
        "status": "complete",
        "targets": len(target_rows),
    }


def measured(value: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in value["target_rows"] if not row["warmup"]]


def key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["case_id"]), int(row["round_index"])


def summarize(output: Path) -> dict[str, Any]:
    cases = read_json(output / "V11_CASES.json")["cases"]
    raw = {
        arm: read_json(output / "generations" / f"{arm}.json")
        for arm in ARMS
    }
    context = {
        str(case["case_id"]): int(case["context_bucket"]) for case in cases
    }
    paired = {}
    arms = {}
    for arm, value in raw.items():
        rows = measured(value)
        ttfts = [float(row["ttft_ms"]) for row in rows]
        paired[arm] = {key(row): float(row["ttft_ms"]) for row in rows}
        copies = sum(
            row.get("event") == "target_copied"
            for row in value["ledger_rows"]
        )
        fallbacks = sum(
            row.get("event") == "target_fallback"
            for row in value["ledger_rows"]
        )
        arms[arm] = {
            "copy_events_all_rounds": copies,
            "fallback_events_all_rounds": fallbacks,
            "mean_ttft_ms": statistics.mean(ttfts),
            "median_ttft_ms": statistics.median(ttfts),
            "p95_ttft_ms": percentile(ttfts, 0.95),
            "targets": len(rows),
        }

    case_sets = {
        "all": set(context),
        "2k4k": {case_id for case_id, size in context.items() if size <= 4096},
    }
    for size in sorted(set(context.values())):
        case_sets[str(size)] = {
            case_id for case_id, value in context.items() if value == size
        }
    savings = {
        arm: {
            name: mean_saving(paired["dense"], paired[arm], selected)
            for name, selected in case_sets.items()
        }
        for arm in REUSE_ARMS
    }
    coding_vs_general = mean_saving(
        paired["general_dual_4k"],
        paired["coding_repo_v10"],
        case_sets["all"],
    )
    cached_prefix = {
        arm: [
            int(row["cached_tokens"])
            for row in measured(raw[arm])
        ]
        for arm in REUSE_ARMS
    }
    expected_copies = len(cases) * TOTAL_ROUNDS
    gates = {
        "cacheblend_speed_frontier_passed": (
            savings["coding_repo_v10"]["all"]
            >= NATIVE_REFERENCES[
                "cacheblend_all_context_cache_ready_saving_percent"
            ]
        ),
        "coding_vs_general_passed": coding_vs_general >= 5.0,
        "copy_events_passed": all(
            arms[arm]["copy_events_all_rounds"] == expected_copies
            for arm in REUSE_ARMS
        ),
        "fallback_passed": all(
            arms[arm]["fallback_events_all_rounds"] == 0
            for arm in REUSE_ARMS
        ),
        "header_cache_passed": all(
            values and min(values) >= 61
            for values in cached_prefix.values()
        ),
        "kvcomm_speed_stretch_passed": (
            savings["coding_repo_v10"]["2k4k"]
            >= NATIVE_REFERENCES[
                "kvcomm_2k4k_cache_ready_saving_percent"
            ]
        ),
        "target_coverage_passed": all(
            arms[arm]["targets"] == len(cases) * MEASURED_ROUNDS
            for arm in ARMS
        ),
    }
    result = {
        "arms": arms,
        "cached_prefix_tokens_measured": {
            arm: {
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
            }
            for arm, values in cached_prefix.items()
        },
        "coding_vs_general_mean_paired_ttft_reduction_percent": (
            coding_vs_general
        ),
        "gates": {
            **gates,
            "mechanism_overall_passed": all(
                gates[name]
                for name in (
                    "copy_events_passed",
                    "fallback_passed",
                    "header_cache_passed",
                    "target_coverage_passed",
                )
            ),
            "sota_speed_goal_passed": (
                gates["cacheblend_speed_frontier_passed"]
                or gates["kvcomm_speed_stretch_passed"]
            ),
        },
        "native_recomputed_from_validated_ledgers": {
            "cacheblend": native_result(CACHEBLEND_DENSE, CACHEBLEND_REUSE),
            "kvcomm": native_result(KVCOMM_DENSE, KVCOMM_REUSE),
        },
        "paired_cache_ready_saving_percent_vs_sglang_dense": savings,
        "prefetch": False,
        "scope": (
            "same Qwen2.5-Coder-3B target token IDs and formal latency cases; "
            "one real source serves seven targets; no task accuracy measured"
        ),
    }
    write_json(output / "V11_RESULT.json", result)
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
