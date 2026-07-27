#!/usr/bin/env python3
"""Run coding-aware exact KV reuse on the frozen native-frontier workload.

This runner uses the same Qwen2.5-Coder-3B model, target messages, target token
IDs, 2 warmups, and 5 measured rounds as the native KVCOMM/CacheBlend latency
frontier.  The source is the workload's frozen preceding repository request.
The target inserts a new task header before the identical repository snapshot.

Arms:
* dense: no cross-request KV reuse and Radix disabled.
* general_dual_4k: ordinary prefix reuse plus the last 4096 shifted repository
  tokens.
* coding_repo_v10: ordinary prefix reuse plus the complete task-labelled
  repository_context island.

Every source/target repetition is separated by a successful cache flush.  The
source is therefore consumed exactly once, and repeated targets cannot turn
into full-prefix Radix hits.  Source materialization is reported separately
from cache-ready target TTFT; it is never hidden in target timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

from benchmark.multi_workflow.run_bridge_reuse_pilot import (
    capped_tail,
    generate,
    manifest_case,
    sha256_file,
    stop_server,
    write_json,
)


PROJECT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
MODEL = (
    "/home/gfy/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/"
    "488639f1ff808d1d3d0ba301aef8c11461451ec5"
)
WORKLOAD = (
    ARTIFACTS
    / "impactkv_native_frontier_20260719/workload_v2/"
    "LONG_CONTEXT_WORKLOAD.json"
)
NATIVE_ROOT = ARTIFACTS / "impactkv_native_frontier_v3_20260720"
CACHEBLEND_DENSE = (
    NATIVE_ROOT
    / "runs/cacheblend/native/formal/latency.dense.dense.validated.jsonl"
)
CACHEBLEND_REUSE = (
    NATIVE_ROOT
    / "runs/cacheblend/native/formal/"
    "latency.reuse.recompute-0.05.validated.jsonl"
)
KVCOMM_DENSE = (
    NATIVE_ROOT
    / "runs/kvcomm/native/formal/latency.dense.dense.validated.jsonl"
)
KVCOMM_REUSE = (
    NATIVE_ROOT
    / "runs/kvcomm/native/formal/"
    "latency.reuse.threshold-0.1.validated.jsonl"
)
DEFAULT_OUTPUT = (
    ARTIFACTS / "impactkv_coding_native_workload_v10_20260727"
)
ARMS = ("dense", "general_dual_4k", "coding_repo_v10")
REUSE_ARMS = ARMS[1:]
WARMUPS = 2
MEASURED_ROUNDS = 5
TOTAL_ROUNDS = WARMUPS + MEASURED_ROUNDS
GENERAL_CAP = 4096
NATIVE_REFERENCES = {
    "cacheblend_all_context_cache_ready_saving_percent": 79.01429315136923,
    "cacheblend_n1_saving_percent": -165.63228548257237,
    "cacheblend_n4_saving_percent": 17.85264849288383,
    "kvcomm_2k4k_cache_ready_saving_percent": 88.31062959906316,
    "kvcomm_2k4k_n1_saving_percent": 60.17659428356137,
    "kvcomm_2k4k_n4_saving_percent": 81.27712077018771,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def render_ids(
    tokenizer: Any, messages: list[dict[str, str]]
) -> tuple[str, list[int], list[tuple[int, int]]]:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return (
        prompt,
        [int(value) for value in encoded["input_ids"]],
        [tuple(int(value) for value in pair) for pair in encoded["offset_mapping"]],
    )


def exact_text_token_span(
    *,
    case_id: str,
    prompt: str,
    offsets: list[tuple[int, int]],
    text: str,
) -> tuple[int, int]:
    char_start = prompt.find(text)
    if char_start < 0:
        raise ValueError(f"{case_id}: reusable text is absent")
    if prompt.find(text, char_start + 1) >= 0:
        raise ValueError(f"{case_id}: reusable text is not unique")
    char_end = char_start + len(text)
    positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start >= char_start and end <= char_end
    ]
    if not positions:
        raise ValueError(f"{case_id}: reusable text has no complete tokens")
    expected = list(range(positions[0], positions[-1] + 1))
    if positions != expected:
        raise ValueError(f"{case_id}: reusable tokens are not contiguous")
    return positions[0], len(positions)


def native_target_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for row in read_jsonl(CACHEBLEND_DENSE):
        case_id = str(row["case_id"])
        value = str(row["token_ids_sha256"])
        previous = hashes.setdefault(case_id, value)
        if previous != value:
            raise ValueError(f"{case_id}: native target token hash changed")
    return hashes


def prepare_cases() -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        local_files_only=True,
    )
    workload = read_json(WORKLOAD)
    native_hashes = native_target_hashes()
    cases = []
    for source in workload["cases"]:
        if source["split"] != "formal":
            continue
        case_id = str(source["case_id"])
        reusable = [
            segment
            for segment in source["segments"]
            if bool(segment["reusable"])
        ]
        if len(reusable) != 1:
            raise ValueError(f"{case_id}: expected one reusable repository")
        repository = reusable[0]
        source_messages = source["metadata"]["source_messages"]
        source_prompt, source_ids, source_offsets = render_ids(
            tokenizer, source_messages
        )
        target_prompt, target_ids, target_offsets = render_ids(
            tokenizer, source["messages"]
        )
        source_start, source_length = exact_text_token_span(
            case_id=case_id,
            prompt=source_prompt,
            offsets=source_offsets,
            text=repository["text"],
        )
        target_start, target_length = exact_text_token_span(
            case_id=case_id,
            prompt=target_prompt,
            offsets=target_offsets,
            text=repository["text"],
        )
        if source_length != target_length:
            raise ValueError(f"{case_id}: source/target repository lengths differ")
        source_segment = source_ids[
            source_start : source_start + source_length
        ]
        target_segment = target_ids[
            target_start : target_start + target_length
        ]
        if source_segment != target_segment:
            raise ValueError(f"{case_id}: repository tokens are not identical")
        if source_start <= 0 or target_start <= 0:
            raise ValueError(f"{case_id}: repository is not a middle island")
        if source_start + source_length >= len(source_ids):
            raise ValueError(f"{case_id}: source repository reaches prompt end")
        if target_start + target_length >= len(target_ids):
            raise ValueError(f"{case_id}: target repository reaches prompt end")
        target_hash = sha256_json(target_ids)
        if target_hash != native_hashes.get(case_id):
            raise ValueError(f"{case_id}: target IDs differ from native ledger")
        cases.append(
            {
                "case_id": case_id,
                "context_bucket": int(
                    source["metadata"]["context_bucket"]
                ),
                "context_tokens": len(target_ids),
                "native_token_ids_sha256": target_hash,
                "prompt_sha256": source["prompt_sha256"],
                "repository_segment_id": repository["segment_id"],
                "repository_tokens": source_length,
                "source_input_ids": source_ids,
                "source_prompt_sha256_json": sha256_json(source_ids),
                "source_start": source_start,
                "suite": source["suite"],
                "target_input_ids": target_ids,
                "target_prompt_sha256_json": target_hash,
                "target_start": target_start,
            }
        )
    if len(cases) != 24:
        raise ValueError(f"expected 24 formal cases, got {len(cases)}")
    if set(native_hashes) != {case["case_id"] for case in cases}:
        raise ValueError("native ledger and formal workload case sets differ")
    return cases


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
    cases_path = output / "V10_CASES.json"
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

    contexts: dict[str, int] = {}
    for case in cases:
        bucket = str(case["context_bucket"])
        contexts[bucket] = contexts.get(bucket, 0) + 1
    registration = {
        "date": "2026-07-27",
        "registered_before_gpu": True,
        "classification": (
            "same-model, same-target-token, same-formal-latency-workload "
            "native-frontier comparison"
        ),
        "model": MODEL,
        "workload": str(WORKLOAD),
        "formal_cases": len(cases),
        "contexts": contexts,
        "protocol": {
            "concurrency": 1,
            "decode_tokens": 1,
            "measured_rounds": MEASURED_ROUNDS,
            "server_shape_warmup_request": 1,
            "target_cache_flush_before_every_pair": True,
            "warmups_per_case": WARMUPS,
            "source_is_frozen_preceding_request": True,
            "source_materialization_excluded_from_cache_ready_ttft": True,
            "source_materialization_reported_for_n1_and_n4": True,
            "target_prefetch": False,
        },
        "arms": {
            "dense": "Radix disabled; target is fully recomputed",
            "general_dual_4k": (
                "ordinary prefix plus tail-capped 4096-token shifted island"
            ),
            "coding_repo_v10": (
                "ordinary prefix plus the complete workload-labelled "
                "repository_context shifted island"
            ),
        },
        "frozen_gates": {
            "all_target_token_hashes_match_native": True,
            "copy_events_per_reuse_arm": len(cases) * TOTAL_ROUNDS,
            "fallback_events_max": 0,
            "coding_8k_mean_ttft_reduction_vs_general_percent_min": 5.0,
            "coding_all_context_cache_ready_saving_vs_dense_percent_min": (
                NATIVE_REFERENCES[
                    "cacheblend_all_context_cache_ready_saving_percent"
                ]
            ),
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
            "native_ledgers_sha256": {
                "cacheblend_dense": sha256_file(CACHEBLEND_DENSE),
                "cacheblend_reuse": sha256_file(CACHEBLEND_REUSE),
                "kvcomm_dense": sha256_file(KVCOMM_DENSE),
                "kvcomm_reuse": sha256_file(KVCOMM_REUSE),
            },
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
    write_json(output / "V10_REGISTRATION.json", registration)
    return registration


def launch_server(
    *,
    output: Path,
    arm: str,
    port: int,
) -> tuple[subprocess.Popen[str], Any, str]:
    base_url = f"http://127.0.0.1:{port}"
    log_path = output / "server" / arm / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": f"{PROJECT / 'python'}:{PROJECT}",
            "SGLANG_KVCOMM_CORE": "0" if arm == "dense" else "1",
        }
    )
    if arm != "dense":
        env["SGLANG_KVCOMM_EXACT_CANARY_MANIFEST"] = str(
            output / "manifests" / f"{arm}.json"
        )
    command = [
        "/home/gfy/.conda/envs/sglang-kvflow/bin/python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--context-length",
        "32768",
        "--attention-backend",
        "triton",
        "--mem-fraction-static",
        "0.80",
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "16384",
        "--page-size",
        "1",
        "--disable-cuda-graph",
        "--disable-overlap-schedule",
        "--enable-deterministic-inference",
        "--enable-request-time-stats-logging",
        "--random-seed",
        "20260727",
    ]
    if arm == "dense":
        command.append("--disable-radix-cache")
    process = subprocess.Popen(
        command,
        cwd=PROJECT,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stream.flush()
                raise RuntimeError(
                    f"{arm} server exited {process.returncode}; inspect {log_path}"
                )
            try:
                response = requests.get(base_url + "/model_info", timeout=2)
                if response.ok:
                    return process, stream, base_url
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(f"{arm} server did not become ready")
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        stream.close()
        raise


def flush_cache(base_url: str) -> None:
    for _ in range(5):
        try:
            response = requests.post(base_url + "/flush_cache", timeout=30)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("cache flush failed after five attempts")


def run_arm(output: Path, arm: str, port: int) -> dict[str, Any]:
    if not (output / "V10_REGISTRATION.json").exists():
        raise FileNotFoundError("run prepare before any GPU arm")
    cases = read_json(output / "V10_CASES.json")["cases"]
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
        # SGLang documents that the first flush may race server initialization.
        # An unregistered shape warmup makes every subsequent flush explicit.
        generate(
            base_url=base_url,
            input_ids=cases[0]["target_input_ids"][:128],
            key=f"v10-server-shape-warmup-{arm}",
            max_new_tokens=1,
            stream=True,
        )
        flush_cache(base_url)
        for case in cases:
            for index in range(TOTAL_ROUNDS):
                flush_cache(base_url)
                warmup = index < WARMUPS
                round_index = max(0, index - WARMUPS)
                if arm != "dense":
                    source_rows.append(
                        {
                            **generate(
                                base_url=base_url,
                                input_ids=case["source_input_ids"],
                                key=(
                                    f"v10-source-{arm}-{case['case_id']}-"
                                    f"{index}"
                                ),
                                max_new_tokens=1,
                                stream=False,
                            ),
                            "case_id": case["case_id"],
                            "context_tokens": case["context_tokens"],
                            "round_index": round_index,
                            "warmup": warmup,
                        }
                    )
                target_rows.append(
                    {
                        **generate(
                            base_url=base_url,
                            input_ids=case["target_input_ids"],
                            key=(
                                f"v10-target-{arm}-{case['case_id']}-"
                                f"{index}"
                            ),
                            max_new_tokens=1,
                            stream=True,
                        ),
                        "arm": arm,
                        "case_id": case["case_id"],
                        "context_tokens": case["context_tokens"],
                        "round_index": round_index,
                        "token_ids_sha256": case[
                            "native_token_ids_sha256"
                        ],
                        "warmup": warmup,
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


def measured_rows(value: dict[str, Any], field: str) -> list[dict[str, Any]]:
    return [row for row in value[field] if not bool(row["warmup"])]


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["case_id"]), int(row["round_index"])


def mean_saving(
    dense: dict[tuple[str, int], float],
    reuse: dict[tuple[str, int], float],
    allowed_cases: set[str] | None = None,
) -> float:
    keys = sorted(set(dense) & set(reuse))
    if allowed_cases is not None:
        keys = [key for key in keys if key[0] in allowed_cases]
    return statistics.mean(
        100 * (1 - reuse[key] / dense[key]) for key in keys
    )


def native_result(
    dense_path: Path,
    reuse_path: Path,
) -> dict[str, Any]:
    dense_rows = [
        row
        for row in read_jsonl(dense_path)
        if not bool(row["metadata"].get("warmup"))
    ]
    reuse_rows = [
        row
        for row in read_jsonl(reuse_path)
        if not bool(row["metadata"].get("warmup"))
    ]
    dense = {
        (str(row["case_id"]), int(row["round_index"])): float(row["ttft_ms"])
        for row in dense_rows
    }
    reuse = {
        (str(row["case_id"]), int(row["round_index"])): float(row["ttft_ms"])
        for row in reuse_rows
    }
    build = {
        (str(row["case_id"]), int(row["round_index"])): float(
            row["cache_build_ms"]
        )
        for row in reuse_rows
    }
    keys = sorted(set(dense) & set(reuse))
    return {
        "cases": len({case_id for case_id, _ in keys}),
        "cache_ready_mean_saving_percent": mean_saving(dense, reuse),
        "n1_mean_saving_percent": statistics.mean(
            100 * (1 - (reuse[key] + build[key]) / dense[key])
            for key in keys
        ),
        "n4_mean_saving_percent": statistics.mean(
            100 * (1 - (reuse[key] + build[key] / 4) / dense[key])
            for key in keys
        ),
        "pairs": len(keys),
    }


def summarize(output: Path) -> dict[str, Any]:
    cases = read_json(output / "V10_CASES.json")["cases"]
    raw = {
        arm: read_json(output / "generations" / f"{arm}.json")
        for arm in ARMS
    }
    case_context = {
        str(case["case_id"]): int(case["context_bucket"]) for case in cases
    }
    arms = {}
    paired = {}
    source_elapsed = {}
    for arm, value in raw.items():
        targets = measured_rows(value, "target_rows")
        ttfts = [float(row["ttft_ms"]) for row in targets]
        paired[arm] = {row_key(row): float(row["ttft_ms"]) for row in targets}
        sources = measured_rows(value, "source_rows")
        source_elapsed[arm] = {
            row_key(row): float(row["elapsed_ms"]) for row in sources
        }
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
        arms[arm] = {
            "copy_events_all_rounds": len(copies),
            "fallback_events_all_rounds": len(fallbacks),
            "mean_ttft_ms": statistics.mean(ttfts),
            "median_ttft_ms": statistics.median(ttfts),
            "p95_ttft_ms": percentile(ttfts, 0.95),
            "targets": len(ttfts),
        }

    all_cases = set(case_context)
    supported_cases = {
        case_id
        for case_id, context in case_context.items()
        if context <= 4096
    }
    context_cases = {
        context: {
            case_id
            for case_id, value in case_context.items()
            if value == context
        }
        for context in sorted(set(case_context.values()))
    }
    savings = {}
    lifecycle = {}
    for arm in REUSE_ARMS:
        savings[arm] = {
            "all": mean_saving(
                paired["dense"], paired[arm], all_cases
            ),
            "2k4k": mean_saving(
                paired["dense"], paired[arm], supported_cases
            ),
            "by_context": {
                str(context): mean_saving(
                    paired["dense"],
                    paired[arm],
                    selected,
                )
                for context, selected in context_cases.items()
            },
        }
        keys = sorted(set(paired["dense"]) & set(paired[arm]))
        lifecycle[arm] = {
            "n1_mean_saving_percent": statistics.mean(
                100
                * (
                    1
                    - (
                        paired[arm][key] + source_elapsed[arm][key]
                    )
                    / paired["dense"][key]
                )
                for key in keys
            ),
            "n4_mean_saving_percent": statistics.mean(
                100
                * (
                    1
                    - (
                        paired[arm][key]
                        + source_elapsed[arm][key] / 4
                    )
                    / paired["dense"][key]
                )
                for key in keys
            ),
        }

    coding_vs_general = {
        "all_mean_paired_ttft_reduction_percent": mean_saving(
            paired["general_dual_4k"],
            paired["coding_repo_v10"],
            all_cases,
        ),
        "8k_mean_paired_ttft_reduction_percent": mean_saving(
            paired["general_dual_4k"],
            paired["coding_repo_v10"],
            {
                case_id
                for case_id, context in case_context.items()
                if context >= 8192
            },
        ),
    }
    expected_copies = len(cases) * TOTAL_ROUNDS
    gates = {
        "cacheblend_speed_frontier_passed": (
            savings["coding_repo_v10"]["all"]
            >= NATIVE_REFERENCES[
                "cacheblend_all_context_cache_ready_saving_percent"
            ]
        ),
        "coding_8k_vs_general_passed": (
            coding_vs_general[
                "8k_mean_paired_ttft_reduction_percent"
            ]
            >= 5.0
        ),
        "copy_events_passed": all(
            arms[arm]["copy_events_all_rounds"] == expected_copies
            for arm in REUSE_ARMS
        ),
        "fallback_passed": all(
            arms[arm]["fallback_events_all_rounds"] == 0
            for arm in REUSE_ARMS
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
    native = {
        "cacheblend": native_result(CACHEBLEND_DENSE, CACHEBLEND_REUSE),
        "kvcomm": native_result(KVCOMM_DENSE, KVCOMM_REUSE),
    }
    result = {
        "arms": arms,
        "coding_vs_general": coding_vs_general,
        "gates": {
            **gates,
            "mechanism_overall_passed": all(
                gates[name]
                for name in (
                    "copy_events_passed",
                    "fallback_passed",
                    "target_coverage_passed",
                )
            ),
            "sota_speed_goal_passed": (
                gates["cacheblend_speed_frontier_passed"]
                or gates["kvcomm_speed_stretch_passed"]
            ),
        },
        "lifecycle_source_request_inclusive": lifecycle,
        "native_recomputed_from_validated_ledgers": native,
        "paired_cache_ready_saving_percent_vs_sglang_dense": savings,
        "prefetch": False,
        "scope": (
            "same Qwen2.5-Coder-3B target token IDs and formal latency cases; "
            "accuracy is not measured by this TTFT run"
        ),
    }
    write_json(output / "V10_RESULT.json", result)
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
