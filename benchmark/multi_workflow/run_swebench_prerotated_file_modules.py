#!/usr/bin/env python3
"""Run Dense vs prerotated SWE-bench file-module exact-prompt TTFT."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any

import time

import requests

from benchmark.multi_workflow.run_bridge_reuse_agent_experiment import launch_server, stop_server
from benchmark.multi_workflow.run_natural_code_cost_exact_prompt_speed import (
    TOTAL_ROUNDS,
    WARMUPS,
    _atomic_manifest,
    generate_detailed,
    read_json,
    read_jsonl,
    write_json,
)
from benchmark.multi_workflow.prepare_swebench_prerotated_file_modules import ARM, MODEL_DEFAULT
from benchmark.multi_workflow.template_prefetch_modes import rope_for_model

RESTART_EVERY = 20
GENERATE_RETRIES = 4


def generate_resilient(*, base_url: str, input_ids: list[int], key: str) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(GENERATE_RETRIES):
        try:
            return generate_detailed(base_url=base_url, input_ids=input_ids, key=f"{key}-a{attempt}")
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            RuntimeError,
        ) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    assert last is not None
    raise last


def _manifest(output: Path, group: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "version": 3,
        "model_id": model,
        "cache_dtype": "bfloat16",
        "lease_ttl_s": 900,
        "ledger_path": str(output / "server/reuse/SERVER_LEDGER.jsonl"),
        "rope": rope_for_model(model),
        "sources": group["sources"],
        "cases": group["cases"],
        "release_source_ids": [],
        "arm": ARM,
        "host_overflow_enabled": True,
        "ordinary_prefix_reuse_enabled": False,
        "ordinary_prefix_repair_tokens": 0,
        "ordinary_prefix_target_only": False,
    }


def _gpu_used_mib() -> int:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(raw.strip().splitlines()[0].split()[0])


def _wait_gpu_released(*, limit_mib: int = 1024, timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    used = _gpu_used_mib()
    while time.time() < deadline:
        used = _gpu_used_mib()
        if used <= limit_mib:
            return
        time.sleep(2)
    raise RuntimeError(f"GPU still holds {used} MiB after stop_server")


def _start_server(output: Path, arm: str, port: int, model: str):
    _wait_gpu_released()
    run_dir = output / f"server/{arm}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "DYNAMIC_MANIFEST.json"
    write_json(manifest_path, _manifest(output, {"sources": [], "cases": []}, model))
    process, stream = launch_server(
        run_dir=run_dir,
        arm=ARM if arm == "reuse" else "dense",
        manifest=manifest_path,
        port=port,
        mem_fraction_static=float(
            os.environ.get("IMPACTKV_MEM_FRACTION_STATIC", "0.88")
        ),
    )
    return process, stream, manifest_path, f"http://127.0.0.1:{port}"


def run_arm(output: Path, arm: str, port: int, model: str) -> dict[str, Any]:
    if arm not in {"dense", "reuse"}:
        raise ValueError(arm)
    result_path = output / f"{arm}.json"
    checkpoint = output / f"{arm}.partial.json"
    plan = read_json(output / "PLAN.json")["groups"]
    sources: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    done: set[int] = set()
    if result_path.exists():
        return {
            "arm": arm,
            "sources": 0,
            "targets": 0,
            "copy_events": 0,
            "fallback_events": 0,
            "resumed": True,
        }
    if checkpoint.exists():
        saved = read_json(checkpoint)
        sources = list(saved.get("sources") or [])
        targets = list(saved.get("targets") or [])
        done = {int(row["group_index"]) for row in targets}
    process = stream = None
    try:
        process, stream, manifest_path, base_url = _start_server(output, arm, port, model)
        generate_resilient(base_url=base_url, input_ids=[100] * 128, key=f"warm-{arm}")
        for offset, group in enumerate(plan):
            group_index = int(group["group_index"])
            if group_index in done:
                continue
            if offset and offset % RESTART_EVERY == 0:
                stop_server(process, stream)
                process, stream, manifest_path, base_url = _start_server(
                    output, arm, port, model
                )
                generate_resilient(base_url=base_url, input_ids=[100] * 128, key=f"rewarm-{arm}-{offset}")
            group_sources: list[dict[str, Any]] = []
            group_targets: list[dict[str, Any]] = []
            for group_attempt in range(2):
                group_sources = []
                group_targets = []
                try:
                    if arm == "reuse":
                        _atomic_manifest(manifest_path, _manifest(output, group, model))
                        for source_index, source_ids in enumerate(group["source_input_ids"]):
                            row = generate_resilient(
                                base_url=base_url,
                                input_ids=source_ids,
                                key=f"source-g{group_index}-{source_index}-t{group_attempt}",
                            )
                            group_sources.append(
                                {**row, "group_index": group_index, "source_index": source_index}
                            )
                    for round_index in range(TOTAL_ROUNDS):
                        row = generate_resilient(
                            base_url=base_url,
                            input_ids=group["target_input_ids"],
                            key=f"target-{arm}-g{group_index}-r{round_index}-t{group_attempt}",
                        )
                        group_targets.append(
                            {
                                **row,
                                "group_index": group_index,
                                "round_index": max(0, round_index - WARMUPS),
                                "warmup": round_index < WARMUPS,
                                "target_prompt_hash": group["target_prompt_hash"],
                            }
                        )
                    break
                except Exception:
                    if group_attempt == 1:
                        raise
                    stop_server(process, stream)
                    process, stream, manifest_path, base_url = _start_server(
                        output, arm, port, model
                    )
                    generate_resilient(
                        base_url=base_url,
                        input_ids=[100] * 128,
                        key=f"recover-{arm}-{group_index}",
                    )
            sources.extend(group_sources)
            targets.extend(group_targets)
            write_json(checkpoint, {"sources": sources, "targets": targets})
    finally:
        if process is not None:
            stop_server(process, stream)
    ledger = read_jsonl(output / f"server/{arm}/SERVER_LEDGER.jsonl")
    value = {"arm": arm, "sources": sources, "targets": targets, "ledger_rows": ledger}
    write_json(result_path, value)
    return {
        "arm": arm,
        "sources": len(sources),
        "targets": len(targets),
        "copy_events": sum(row.get("event") == "target_copied" for row in ledger),
        "fallback_events": sum(row.get("event") == "target_fallback" for row in ledger),
    }


def summarize(output: Path) -> dict[str, Any]:
    plan_meta = read_json(output / "PLAN.json")
    plan = plan_meta["groups"]
    dense = read_json(output / "dense.json")
    reuse = read_json(output / "reuse.json")
    dense_rows = {
        (int(row["group_index"]), int(row["round_index"])): row
        for row in dense["targets"]
        if not row["warmup"]
    }
    reuse_rows = {
        (int(row["group_index"]), int(row["round_index"])): row
        for row in reuse["targets"]
        if not row["warmup"]
    }
    if set(dense_rows) != set(reuse_rows):
        raise ValueError("paired targets differ")
    savings = [
        1 - float(reuse_rows[key]["ttft_ms"]) / float(dense_rows[key]["ttft_ms"])
        for key in dense_rows
    ]
    source_build = {
        index: sum(
            float(row["elapsed_ms"])
            for row in reuse["sources"]
            if int(row["group_index"]) == index
        )
        for index in range(len(plan))
    }
    group_savings = []
    n4_dense = 0.0
    n4_reuse = 0.0
    for group in plan:
        index = int(group["group_index"])
        dense_mean = statistics.fmean(
            float(row["ttft_ms"]) for key, row in dense_rows.items() if key[0] == index
        )
        reuse_mean = statistics.fmean(
            float(row["ttft_ms"]) for key, row in reuse_rows.items() if key[0] == index
        )
        group_savings.append(1 - reuse_mean / dense_mean)
        n4_dense += dense_mean * 4
        n4_reuse += reuse_mean * 4 + source_build[index]
    ledger = reuse["ledger_rows"]
    prerotated = sum(int(row.get("applied_pre_rotate_delta") or 0) != 0 for row in ledger)
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "classification": "SWE-bench exact-prompt true-lossy file-module cache-ready TTFT",
        "prefetch": False,
        "ordinary_prefix_reuse": False,
        "coverage": {
            "target_groups": len(plan),
            "islands": sum(row["islands"] for row in plan),
            "measured_pairs": len(savings),
        },
        "latency": {
            "cache_ready_speedup_ratio_of_means": (
                statistics.fmean(float(row["ttft_ms"]) for row in dense_rows.values())
                / statistics.fmean(float(row["ttft_ms"]) for row in reuse_rows.values())
            ),
            "paired_ttft_saving_median": statistics.median(savings),
            "paired_ttft_win_rate": sum(value > 0 for value in savings) / len(savings),
            "target_group_saving_median": statistics.median(group_savings),
            "n4_including_one_source_build_speedup": n4_dense / n4_reuse,
            "mean_source_build_ms_per_target_group": statistics.fmean(source_build.values()),
        },
        "mechanism": {
            "copy_events": sum(row.get("event") == "target_copied" for row in ledger),
            "expected_copy_events": sum(row["islands"] for row in plan) * TOTAL_ROUNDS,
            "fallback_events": sum(row.get("event") == "target_fallback" for row in ledger),
            "source_prerotation_events": prerotated,
        },
        "one_token_output_agreement": {
            "fraction": sum(
                dense_rows[key].get("output_text") == reuse_rows[key].get("output_text")
                for key in dense_rows
            )
            / len(dense_rows),
            "not_accuracy": True,
        },
    }
    is_7b = bool(plan_meta.get("not_30b_swebench_plan")) or plan_meta.get(
        "model"
    ) == "Qwen2.5-Coder-7B-Instruct"
    if is_7b:
        result["model"] = "Qwen2.5-Coder-7B-Instruct"
        result["not_30b_swebench_plan"] = True
        result["same_token_ids_as_96092"] = False
        result["official_96092_prefetch"] = False
        result["rope_base"] = 1_000_000
        result["qwen25_rope_ok"] = True
        if plan_meta.get("policy_label") == "general_shifted_lcs":
            result["classification"] = (
                "7B-native general shifted LCS copier vs 7B Dense; not job 96092"
            )
            result["not_96092_coding_plan"] = True
        else:
            result["classification"] = (
                "7B-native SWE-bench file-module copy vs 7B Dense; not job 96092"
            )
    elif plan_meta.get("not_96092_coding_plan"):
        result["classification"] = (
            "same-token general shifted LCS copier vs Dense; not job 96092"
        )
        result["not_96092_coding_plan"] = True
        result["same_token_ids_as_96092"] = True
        result["official_96092_prefetch"] = False
    write_json(output / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default=os.environ.get("IMPACTKV_MODEL", MODEL_DEFAULT))
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    dense = run_arm(artifact, "dense", args.port, args.model)
    reuse = run_arm(artifact, "reuse", args.port, args.model)
    result = summarize(artifact)
    print(json.dumps({"dense": dense, "reuse": reuse, "result": {
        "cache_ready_speedup": result["latency"]["cache_ready_speedup_ratio_of_means"],
        "copy_events": result["mechanism"]["copy_events"],
        "fallback_events": result["mechanism"]["fallback_events"],
        "prefetch": result["prefetch"],
    }}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
