#!/usr/bin/env python3
"""Measure SGLang TTFT for variable-length natural repository-code islands.

This is an explicitly exploratory engineering follow-up.  It is registered
after the physical-splice result showed a directional advantage for repository
code and a disadvantage for assistant interpretation.  It must not be cited as
an independent confirmation of that post-hoc code-only policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    manifest_case,
    sha256_file,
    stop_server,
    write_json,
)
from benchmark.multi_workflow.run_coding_header_hot_v11 import generate_detailed
from benchmark.multi_workflow.run_coding_native_workload_v10 import (
    MODEL,
    flush_cache,
    launch_server,
    read_json,
    read_jsonl,
)


ROOT = Path("/home/gfy/CodeMAS_Project")
ATTENTION = (
    ROOT
    / "kvflow-artifacts/impactkv_natural_module_attention_20260808/"
    "attention_initial20_r1"
)
PHYSICAL = ATTENTION / "physical_splice_minimal_reliable"
DEFAULT_OUTPUT = PHYSICAL / "stage_overhead_code_only_r2"
ARM = "natural_repository_code_module"
WARMUPS = 1
MEASURED_ROUNDS = 3
TOTAL_ROUNDS = WARMUPS + MEASURED_ROUNDS


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def select_code_cases(
    design: Mapping[str, Any], selected_candidate_keys: Sequence[str]
) -> list[dict[str, Any]]:
    """Project the outcome-blind physical selection to repository-code only."""

    selected = set(selected_candidate_keys)
    output: list[dict[str, Any]] = []
    for case in design["cases"]:
        candidates = []
        for candidate in case["candidates"]:
            key = f"{case['case_id']}::{candidate['candidate_id']}"
            if key not in selected or candidate["module_type"] != "repository_code":
                continue
            length = int(candidate["natural_length"])
            source_start = int(candidate["source_start"])
            target_start = int(candidate["target_start"])
            if (
                case["source_input_ids"][source_start : source_start + length]
                != case["target_input_ids"][target_start : target_start + length]
            ):
                raise ValueError(f"{key}: natural code module is not token-identical")
            candidates.append(
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "length": length,
                    "source_start": source_start,
                    "target_start": target_start,
                }
            )
        candidates.sort(key=lambda row: row["target_start"])
        for left, right in zip(candidates, candidates[1:]):
            if left["target_start"] + left["length"] > right["target_start"]:
                raise ValueError(f"{case['case_id']}: selected code modules overlap")
        if candidates:
            output.append(
                {
                    "case_id": str(case["case_id"]),
                    "instance_id": str(case["instance_id"]),
                    "source_input_ids": [int(value) for value in case["source_input_ids"]],
                    "target_input_ids": [int(value) for value in case["target_input_ids"]],
                    "spans": candidates,
                }
            )
    return output


def prepare(attention: Path, physical: Path, output: Path) -> dict[str, Any]:
    registration_path = output / "REGISTRATION.json"
    if registration_path.exists():
        return read_json(registration_path)
    if (output / "dense.json").exists() or (output / f"{ARM}.json").exists():
        raise RuntimeError("stage outcomes exist without a frozen registration")
    physical_result = read_json(physical / "RESULT.json")
    code_result = physical_result["module_results"]["repository_code"]
    if not (
        float(code_result["local_output_natural_boundary_median_ratio"]) < 1
        and float(code_result["local_output_natural_wins"]) > 0.5
    ):
        raise RuntimeError("repository-code physical point advantage is absent")
    physical_registration = read_json(physical / "REGISTRATION.json")
    design = read_json(attention / "DESIGN.json")
    cases = select_code_cases(
        design, physical_registration["selected_candidate_keys"]
    )
    all_target_prompts = {
        tuple(case["target_input_ids"]) for case in cases
    }
    static_replay_conflicts = [
        case["case_id"]
        for case in cases
        if tuple(case["source_input_ids"]) in all_target_prompts
    ]
    cases = [
        case
        for case in cases
        if case["case_id"] not in set(static_replay_conflicts)
    ]
    if len(cases) < 8:
        raise RuntimeError("unexpected code-only stage capacity")

    rows = []
    for case in cases:
        group_id = f"natural-code-{case['case_id']}"
        for index, span in enumerate(case["spans"]):
            row = manifest_case(
                case_id=f"{group_id}-i{index}",
                policy_label=ARM,
                source_ids=case["source_input_ids"],
                target_ids=case["target_input_ids"],
                span=span,
            )
            # Identical code text can be read under different surrounding
            # prompts.  Those K/V tensors are context-dependent and therefore
            # must not alias in the pool merely because token IDs match.
            row["content_hash"] = hashlib.sha256(
                (
                    f"{ARM}:{case['case_id']}:{span['candidate_id']}:"
                    f"{row['source_prompt_hash']}:{row['segment_token_hash']}"
                ).encode()
            ).hexdigest()
            row["source_id"] = f"{group_id}-source-{index}"
            row["target_group_id"] = group_id
            row["target_uses"] = TOTAL_ROUNDS
            rows.append(row)

    output.mkdir(parents=True, exist_ok=True)
    cases_path = output / "CASES.json"
    write_json(cases_path, {"cases": cases})
    manifest_path = output / "manifests" / f"{ARM}.json"
    write_json(
        manifest_path,
        {
            "cache_dtype": "bfloat16",
            "cases": rows,
            "lease_ttl_s": 900,
            "ledger_path": str(output / "server" / ARM / "EXACT_LEDGER.jsonl"),
            "model_id": MODEL,
            "ordinary_prefix_reuse_enabled": False,
            "rope": {
                "base": 1_000_000,
                "is_neox_style": True,
                "rotary_dim": 128,
            },
            "version": 2,
        },
    )
    lengths = [span["length"] for case in cases for span in case["spans"]]
    value = {
        "status": "REGISTERED_BEFORE_STAGE_GPU",
        "classification": "post-physical exploratory code-only stage overhead",
        "not_independent_confirmation": True,
        "static_replay_exclusions": {
            "case_ids": static_replay_conflicts,
            "count": len(static_replay_conflicts),
            "reason": (
                "a source prompt is also a registered target prompt; the static "
                "controller cannot distinguish its source-only role"
            ),
            "selected_without_stage_latency_or_output": True,
        },
        "motivation": (
            "Repository code had a 0.826 natural/boundary perturbation ratio "
            "and 65.6% pair wins; assistant interpretation was harmful."
        ),
        "pool_identity": (
            "source prompt hash + natural module identity + segment token hash; "
            "same code text in different contexts never aliases"
        ),
        "arms": {
            "dense": "same target prompts, radix disabled, no source request",
            ARM: "whole variable-length repository-code modules, K+V copied with RoPE correction",
        },
        "protocol": {
            "cache_ready_primary": True,
            "decode_tokens": 1,
            "measured_rounds": MEASURED_ROUNDS,
            "prefetch": False,
            "source_build_reported_separately": True,
            "warmups": WARMUPS,
        },
        "capacity": {
            "cases": len(cases),
            "tasks": len({case["instance_id"] for case in cases}),
            "islands": len(lengths),
            "tokens": sum(lengths),
            "length_min_median_max": [min(lengths), statistics.median(lengths), max(lengths)],
        },
        "inputs": {
            "cases_sha256": sha256_file(cases_path),
            "manifest_sha256": sha256_file(manifest_path),
            "physical_registration_sha256": sha256_file(physical / "REGISTRATION.json"),
            "physical_result_sha256": sha256_file(physical / "RESULT.json"),
        },
        "stage_outcome_used_for_selection": False,
        "protected": {
            "old_dirty_checkout_modified": False,
            "old_preregistration_thresholds_modified": False,
            "paper_modified": False,
            "prefetch": False,
        },
    }
    write_json(registration_path, value)
    return value


def run_arm(output: Path, arm: str, port: int) -> dict[str, Any]:
    if arm not in {"dense", ARM}:
        raise ValueError(arm)
    registration = read_json(output / "REGISTRATION.json")
    if registration["inputs"]["cases_sha256"] != sha256_file(output / "CASES.json"):
        raise ValueError("stage cases changed after registration")
    result_path = output / f"{arm}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    cases = read_json(output / "CASES.json")["cases"]
    process, stream, base_url = launch_server(output=output, arm=arm, port=port)
    sources = []
    targets = []
    try:
        generate_detailed(
            base_url=base_url,
            input_ids=[100] * 128,
            key=f"natural-code-unrelated-{arm}",
        )
        flush_cache(base_url)
        for case in cases:
            if arm != "dense":
                source = generate_detailed(
                    base_url=base_url,
                    input_ids=case["source_input_ids"],
                    key=f"natural-code-source-{case['case_id']}",
                )
                sources.append({**source, "case_id": case["case_id"]})
            for round_index in range(TOTAL_ROUNDS):
                target = generate_detailed(
                    base_url=base_url,
                    input_ids=case["target_input_ids"],
                    key=f"natural-code-target-{arm}-{case['case_id']}-{round_index}",
                )
                targets.append(
                    {
                        **target,
                        "case_id": case["case_id"],
                        "round_index": max(0, round_index - WARMUPS),
                        "warmup": round_index < WARMUPS,
                        "target_ids_sha256": _sha256_json(case["target_input_ids"]),
                    }
                )
    finally:
        stop_server(process, stream)
    ledger_path = output / "server" / arm / "EXACT_LEDGER.jsonl"
    ledger = read_jsonl(ledger_path) if ledger_path.exists() else []
    value = {"arm": arm, "ledger_rows": ledger, "sources": sources, "targets": targets}
    write_json(result_path, value)
    return {
        "arm": arm,
        "copy_events": sum(row.get("event") == "target_copied" for row in ledger),
        "targets": len(targets),
    }


def _paired(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    return {
        (str(row["case_id"]), int(row["round_index"])): row
        for row in rows
        if not row["warmup"]
    }


def summarize(output: Path) -> dict[str, Any]:
    cases = read_json(output / "CASES.json")["cases"]
    dense_value = read_json(output / "dense.json")
    reuse_value = read_json(output / f"{ARM}.json")
    dense = _paired(dense_value["targets"])
    reuse = _paired(reuse_value["targets"])
    if set(dense) != set(reuse):
        raise ValueError("Dense/reuse stage targets differ")
    ratios = [float(dense[key]["ttft_ms"]) / float(reuse[key]["ttft_ms"]) for key in dense]
    savings = [
        100 * (float(dense[key]["ttft_ms"]) - float(reuse[key]["ttft_ms"])) / float(dense[key]["ttft_ms"])
        for key in dense
    ]
    source_by_case = {str(row["case_id"]): float(row["elapsed_ms"]) for row in reuse_value["sources"]}
    dense_mean = statistics.fmean(float(row["ttft_ms"]) for row in dense.values())
    reuse_mean = statistics.fmean(float(row["ttft_ms"]) for row in reuse.values())
    build_mean = statistics.fmean(source_by_case.values())
    copy_events = sum(row.get("event") == "target_copied" for row in reuse_value["ledger_rows"])
    expected_copies = sum(len(case["spans"]) for case in cases) * TOTAL_ROUNDS
    fallbacks = sum(row.get("event") == "target_fallback" for row in reuse_value["ledger_rows"])
    exact_token_agreement = sum(
        dense[key].get("output_text") == reuse[key].get("output_text") for key in dense
    ) / len(dense)
    case_savings = []
    for case in cases:
        case_id = str(case["case_id"])
        dense_case = statistics.fmean(
            float(row["ttft_ms"])
            for key, row in dense.items()
            if key[0] == case_id
        )
        reuse_case = statistics.fmean(
            float(row["ttft_ms"])
            for key, row in reuse.items()
            if key[0] == case_id
        )
        case_savings.append(
            {
                "case_id": case_id,
                "island_tokens": sum(int(span["length"]) for span in case["spans"]),
                "ttft_saving_percent": 100 * (dense_case - reuse_case) / dense_case,
            }
        )
    bucket_specs = (("lt_128", 0, 128), ("128_255", 128, 256), ("256_511", 256, 512), ("ge_512", 512, 10**9))
    length_buckets = {}
    for name, lower, upper in bucket_specs:
        selected = [
            row for row in case_savings if lower <= row["island_tokens"] < upper
        ]
        if selected:
            values = [float(row["ttft_saving_percent"]) for row in selected]
            length_buckets[name] = {
                "cases": len(selected),
                "mean_ttft_saving_percent": statistics.fmean(values),
                "median_ttft_saving_percent": statistics.median(values),
                "win_rate": sum(value > 0 for value in values) / len(values),
            }
    result = {
        "status": "COMPLETE" if copy_events == expected_copies and not fallbacks else "MECHANISM_FAILURE",
        "classification": "exploratory code-only stage overhead; not an accuracy experiment",
        "coverage": {
            "cases": len(cases),
            "tasks": len({case["instance_id"] for case in cases}),
            "islands": sum(len(case["spans"]) for case in cases),
            "measured_targets_per_arm": len(dense),
        },
        "latency": {
            "dense_mean_ttft_ms": dense_mean,
            "reuse_mean_ttft_ms": reuse_mean,
            "cache_ready_speedup": dense_mean / reuse_mean,
            "paired_speedup_median": statistics.median(ratios),
            "paired_ttft_saving_percent_median": statistics.median(savings),
            "paired_ttft_win_rate": sum(value > 0 for value in savings) / len(savings),
            "mean_source_build_ms": build_mean,
            "n4_including_one_build_speedup": dense_mean / (reuse_mean + build_mean / 4),
            "posthoc_by_island_length": length_buckets,
        },
        "mechanism": {
            "copy_events": copy_events,
            "expected_copy_events": expected_copies,
            "fallback_events": fallbacks,
        },
        "one_token_diagnostic": {
            "dense_reuse_exact_output_agreement": exact_token_agreement,
            "not_official_accuracy": True,
        },
        "case_rows": case_savings,
    }
    write_json(output / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--attention", type=Path, default=ATTENTION)
    parser.add_argument("--physical", type=Path, default=PHYSICAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arm", choices=("dense", ARM))
    parser.add_argument("--port", type=int, default=31280)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.attention, args.physical, args.output)
    elif args.command == "run":
        if args.arm is None:
            parser.error("run requires --arm")
        value = run_arm(args.output, args.arm, args.port)
    else:
        value = summarize(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
