#!/usr/bin/env python3
"""Corrected Phase 4 R5 key rerun with paired dense baselines."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.approx_kv.metrics import (
    metric_subset,
    usable_kv_capacity_tokens,
)
from benchmark.approx_kv.run_phase4_cachetune_canary import (
    NON_PREFIX_HEAD_TOKENS,
    NON_PREFIX_TAIL_TOKENS,
    build_eviction_pressure_workloads,
    build_non_prefix_segment_workload,
    capture_final_pool_reset_and_invariant,
    dense_generate_payload,
    ensure_target_head_resident,
    eviction_pressure_filler_count_for_rho,
    flush_and_force_gauge_refresh,
    flush_exact_radix_cache,
    metric_delta,
    metric_snapshot,
    observed_rho,
    register_eviction_pressure_objects,
    require_cached_tokens,
    require_finished_by_length,
    run_independent_round,
    timed_post,
    validate_pairwise_head_isolation,
)
from sglang.srt.mem_cache.cachetune.hardware_profile import (
    CacheTuneMode,
    HardwareMeasurement,
    RatioBounds,
    quantize_ratio,
    roofline_ratio,
)


@dataclass
class ServerProcess:
    process: subprocess.Popen
    log_file: Any
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--model-revision",
        default="c1899de289a04d12100db370d81485cdf75e47ca",
    )
    parser.add_argument(
        "--model-fingerprint",
        default="Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca",
    )
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument(
        "--image-digest",
        default="sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--central-log", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--body-tokens", default="1024,2048")
    parser.add_argument("--header-tokens", type=int, default=64)
    parser.add_argument("--target-rho", type=float, default=2.0)
    parser.add_argument("--max-segment-chunk-tokens", type=int, default=512)
    parser.add_argument(
        "--pressure-filler-head-tokens", type=int, default=NON_PREFIX_HEAD_TOKENS
    )
    parser.add_argument("--pressure-filler-body-tokens", type=int, default=512)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument("--chunked-prefill-size", type=int, default=1024)
    parser.add_argument("--t-c-ms", type=float, default=0.025747446)
    parser.add_argument("--t-i-ms", type=float, default=0.002326677)
    parser.add_argument("--t-o-ms", type=float, default=1.825835613)
    parser.add_argument("--first-recompute-layer", type=int, default=1)
    parser.add_argument("--server-start-timeout-s", type=float, default=600)
    parser.add_argument("--order-seed", type=int, default=20260724)
    parser.add_argument(
        "--allow-no-eviction",
        action="store_true",
        help="Diagnostic-only: do not fail a round whose pressure caused no eviction.",
    )
    return parser.parse_args()


def parse_body_tokens(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("body-tokens must contain positive integers")
    return values


def current_git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def server_command(args: argparse.Namespace, restart: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--revision",
        args.model_revision,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tp-size",
        "1",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--max-prefill-tokens",
        str(args.chunked_prefill_size),
        "--max-running-requests",
        "2",
        "--attention-backend",
        "torch_native",
        "--sampling-backend",
        "pytorch",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--radix-eviction-policy",
        "lru",
        "--enable-cache-report",
        "--enable-metrics",
        "--random-seed",
        str(117 + restart),
        "--log-level",
        "warning",
    ]


def start_server(
    args: argparse.Namespace, restart: int, log_path: Path
) -> ServerProcess:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
            "SGLANG_APPROX_KV_CORE": "1",
            "SGLANG_APPROX_KV_CACHETUNE": "1",
            "SGLANG_CACHETUNE_MODE": "speed_only",
            "SGLANG_CACHETUNE_T_C_MS": str(args.t_c_ms),
            "SGLANG_CACHETUNE_T_I_MS": str(args.t_i_ms),
            "SGLANG_CACHETUNE_T_O_MS": str(args.t_o_ms),
            "SGLANG_CACHETUNE_PROBE_LAYERS": "0",
            "SGLANG_CACHETUNE_FIRST_RECOMPUTE_LAYER": str(args.first_recompute_layer),
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        server_command(args, restart),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )
    return ServerProcess(process=process, log_file=log_file, log_path=log_path)


def stop_server(server: ServerProcess) -> None:
    if server.process.poll() is None:
        os.killpg(server.process.pid, signal.SIGTERM)
        try:
            server.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.process.pid, signal.SIGKILL)
            server.process.wait(timeout=10)
    server.log_file.close()


def tail_text(path: Path, lines: int = 160) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def wait_ready(args: argparse.Namespace, server: ServerProcess) -> None:
    deadline = time.monotonic() + args.server_start_timeout_s
    health_url = f"http://127.0.0.1:{args.port}/health_generate"
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            server.log_file.flush()
            raise RuntimeError(
                f"server exited during startup: {server.process.returncode}\n"
                f"{tail_text(server.log_path)}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy\n{tail_text(server.log_path)}")


def first_output_token(response: dict) -> int | None:
    output_ids = response.get("output_ids") or []
    return int(output_ids[0]) if output_ids else None


def run_dense_round(
    args: argparse.Namespace,
    *,
    tokenizer: Any,
    workload: Any,
    body_tokens: int,
    restart: int,
    repeat: int,
    warmup: bool,
) -> dict[str, Any]:
    base_url = f"http://127.0.0.1:{args.port}"
    label = (
        f"dense-b{body_tokens}-restart{restart}-repeat{repeat}-"
        f"{'warmup' if warmup else 'formal'}"
    )
    metrics_at_round_start = flush_and_force_gauge_refresh(
        base_url, tokenizer, label=f"{label} round-start"
    )
    capacity_tokens = usable_kv_capacity_tokens(metrics_at_round_start)
    flush_exact_radix_cache(base_url)

    seed_response, seed_head_ms = timed_post(
        base_url, dense_generate_payload(workload.seed_prompt_ids)
    )
    require_finished_by_length(seed_response, f"{label} seed")
    require_cached_tokens(seed_response, 0, f"{label} seed")

    metrics_after_setup = metric_snapshot(base_url)
    already_pinned_tokens = round(
        metric_delta(
            metrics_at_round_start,
            metrics_after_setup,
            "sglang:kv_used_tokens",
        )
    )
    filler_count = eviction_pressure_filler_count_for_rho(
        target_rho=args.target_rho,
        usable_capacity_tokens=capacity_tokens,
        tokens_per_filler=(
            args.pressure_filler_head_tokens + args.pressure_filler_body_tokens
        ),
        already_pinned_tokens=already_pinned_tokens,
    )
    pressure_workloads = build_eviction_pressure_workloads(
        tokenizer,
        object_count=filler_count,
        body_tokens=args.pressure_filler_body_tokens,
        head_tokens=args.pressure_filler_head_tokens,
        tail_tokens=NON_PREFIX_TAIL_TOKENS,
        salt_prefix=f"phase4-r5-key-{label}",
        reserved_first_token_ids=frozenset({workload.target_head_ids[0]}),
    )
    validate_pairwise_head_isolation(
        [
            (f"pressure-filler[{index}]", filler.target_head_ids)
            for index, filler in enumerate(pressure_workloads)
        ]
        + [(label, workload.target_head_ids)]
    )
    pressure_phase = register_eviction_pressure_objects(
        base_url,
        pressure_workloads,
        label=label,
        capacity_tokens=capacity_tokens,
        target_rho=args.target_rho,
        already_pinned_tokens=already_pinned_tokens,
    )
    head_reseed = ensure_target_head_resident(
        base_url, workload, label=f"{label} post-pressure head re-seed"
    )

    metrics_before_target = metric_snapshot(base_url)
    target_response, target_ms = timed_post(
        base_url, dense_generate_payload(workload.target_prompt_ids)
    )
    require_finished_by_length(target_response, f"{label} target")
    require_cached_tokens(
        target_response, workload.body_start_in_target, f"{label} target"
    )
    metrics_after_target = metric_snapshot(base_url)
    target_delta = {
        name: metric_delta(metrics_before_target, metrics_after_target, name)
        for name in (
            "sglang:approx_kv_cachetune_selected_tokens_total",
            "sglang:approx_kv_cachetune_recomputed_layers_total",
            "sglang:approx_kv_cachetune_precomputed_total",
            "sglang:approx_kv_dense_fallback_total",
        )
    }
    if any(value != 0 for value in target_delta.values()):
        raise RuntimeError(f"dense target moved CacheTune counters: {target_delta}")

    reseed_ms = head_reseed["ttft_ms"]
    costs = {
        "target_only_ms": target_ms,
        "adapter_combined_ms": target_ms,
        "request_path_ms": seed_head_ms + reseed_ms + target_ms,
        "full_lifecycle_ms": seed_head_ms + reseed_ms + target_ms,
        "seed_head_ms": seed_head_ms,
        "register_raw_ms": 0.0,
        "register_fresh_ms": 0.0,
        "post_pressure_reseed_ms": reseed_ms,
    }
    metrics_after_round = metric_snapshot(base_url)
    eviction_delta = metric_delta(
        metrics_at_round_start,
        metrics_after_round,
        "sglang:evicted_tokens_total",
    )
    if eviction_delta <= 0 and not args.allow_no_eviction:
        raise RuntimeError("dense matched round observed no eviction")
    return {
        "arm": "dense",
        "body_tokens": body_tokens,
        "restart": restart,
        "repeat": repeat,
        "warmup": warmup,
        "capacity_tokens": capacity_tokens,
        "already_pinned_tokens": already_pinned_tokens,
        "pressure_phase": pressure_phase,
        "head_reseed_after_pressure": head_reseed,
        "target_response": target_response,
        "target_cached_tokens": int(target_response["meta_info"]["cached_tokens"]),
        "first_output_token": first_output_token(target_response),
        "costs": costs,
        "rho_after_target": observed_rho(
            metrics_after_round, capacity_tokens=capacity_tokens
        ),
        "evicted_tokens_total_delta": eviction_delta,
        "target_counter_delta": target_delta,
        "metrics_at_round_start": metric_subset(metrics_at_round_start),
        "metrics_after_round": metric_subset(metrics_after_round),
    }


def run_cachetune_round(
    args: argparse.Namespace,
    *,
    tokenizer: Any,
    workload: Any,
    body_tokens: int,
    restart: int,
    repeat: int,
    warmup: bool,
    expected_selected_tokens: int,
    expected_recomputed_layers: int,
) -> dict[str, Any]:
    label = (
        f"cachetune-b{body_tokens}-restart{restart}-repeat{repeat}-"
        f"{'warmup' if warmup else 'formal'}"
    )
    raw_hash = f"cachetune-raw:{label}"
    fresh_hash = f"cachetune-fresh:{label}"
    result = run_independent_round(
        f"http://127.0.0.1:{args.port}",
        tokenizer,
        workload,
        raw_hash=raw_hash,
        fresh_hash=fresh_hash,
        model_fingerprint=args.model_fingerprint,
        cache_dtype="fp16",
        label=label,
        max_chunk_tokens=args.max_segment_chunk_tokens,
        target_rho=args.target_rho,
        pressure_filler_head_tokens=args.pressure_filler_head_tokens,
        pressure_filler_body_tokens=args.pressure_filler_body_tokens,
        causal_prefix_registration=True,
    )
    counter_delta = {
        name: metric_delta(
            result["metrics_at_round_start"],
            result["metrics_after_round"],
            name,
        )
        for name in (
            "sglang:approx_kv_cachetune_selected_tokens_total",
            "sglang:approx_kv_cachetune_recomputed_layers_total",
            "sglang:approx_kv_cachetune_precomputed_total",
            "sglang:approx_kv_dense_fallback_total",
        )
    }
    expected = {
        "sglang:approx_kv_cachetune_selected_tokens_total": expected_selected_tokens,
        "sglang:approx_kv_cachetune_recomputed_layers_total": (
            expected_recomputed_layers if expected_selected_tokens else 0
        ),
        "sglang:approx_kv_cachetune_precomputed_total": (
            1 if expected_selected_tokens else 0
        ),
        "sglang:approx_kv_dense_fallback_total": 0,
    }
    if counter_delta != expected:
        raise RuntimeError(
            f"CacheTune counter mismatch: observed={counter_delta}, expected={expected}"
        )
    reseed_ms = (
        result["head_reseed_after_pressure"]["ttft_ms"]
        if result["head_reseed_after_pressure"] is not None
        else 0.0
    )
    costs = {
        "target_only_ms": result["reuse_ms"],
        "adapter_combined_ms": result["register_fresh_ms"] + result["reuse_ms"],
        "request_path_ms": (
            result["seed_head_ms"]
            + result["register_fresh_ms"]
            + reseed_ms
            + result["reuse_ms"]
        ),
        "full_lifecycle_ms": (
            result["seed_head_ms"]
            + result["register_raw_ms"]
            + result["register_fresh_ms"]
            + reseed_ms
            + result["reuse_ms"]
        ),
        "seed_head_ms": result["seed_head_ms"],
        "register_raw_ms": result["register_raw_ms"],
        "register_fresh_ms": result["register_fresh_ms"],
        "post_pressure_reseed_ms": reseed_ms,
    }
    return {
        "arm": "cachetune",
        "body_tokens": body_tokens,
        "restart": restart,
        "repeat": repeat,
        "warmup": warmup,
        "capacity_tokens": result["capacity_tokens"],
        "already_pinned_tokens": result["already_pinned_tokens"],
        "pressure_phase": result["pressure_phase"],
        "head_reseed_after_pressure": result["head_reseed_after_pressure"],
        "target_cached_tokens": result["reuse_cached_tokens"],
        "first_output_token": result["first_output_token"],
        "costs": costs,
        "rho_after_target": result["observed_rho_after_target"],
        "peak_rho_observed": result["peak_rho_observed"],
        "evicted_tokens_total_delta": result["evicted_tokens_total_delta"],
        "target_counter_delta": counter_delta,
        "raw_registration": result["raw_registration"],
        "fresh_registration": result["fresh_registration"],
        "metrics_at_round_start": metric_subset(result["metrics_at_round_start"]),
        "metrics_after_round": metric_subset(result["metrics_after_round"]),
    }


def run_body(
    args: argparse.Namespace,
    *,
    tokenizer: Any,
    num_layers: int,
    body_tokens: int,
    restart: int,
) -> dict[str, Any]:
    workload = build_non_prefix_segment_workload(
        tokenizer,
        body_tokens=body_tokens,
        head_tokens=args.header_tokens,
        tail_tokens=NON_PREFIX_TAIL_TOKENS,
        salt=f"phase4-r5-key-body{body_tokens}",
    )
    measurement = HardwareMeasurement(
        t_c_ms=args.t_c_ms,
        t_i_ms=args.t_i_ms,
        t_o_ms=args.t_o_ms,
    )
    quantized = quantize_ratio(
        roofline_ratio(measurement),
        context_length=body_tokens,
        bounds=RatioBounds.for_mode(CacheTuneMode.SPEED_ONLY),
    )
    expected_selected_tokens = quantized.repair_tokens
    expected_recomputed_layers = num_layers - args.first_recompute_layer

    rng = random.Random(args.order_seed + restart * 1000 + body_tokens)
    warmup_order = ["dense", "cachetune"]
    rng.shuffle(warmup_order)
    warmups = []
    for arm in warmup_order:
        if arm == "dense":
            warmups.append(
                run_dense_round(
                    args,
                    tokenizer=tokenizer,
                    workload=workload,
                    body_tokens=body_tokens,
                    restart=restart,
                    repeat=-1,
                    warmup=True,
                )
            )
        else:
            warmups.append(
                run_cachetune_round(
                    args,
                    tokenizer=tokenizer,
                    workload=workload,
                    body_tokens=body_tokens,
                    restart=restart,
                    repeat=-1,
                    warmup=True,
                    expected_selected_tokens=expected_selected_tokens,
                    expected_recomputed_layers=expected_recomputed_layers,
                )
            )

    rows = []
    orders = []
    for repeat in range(args.repeats):
        order = ["dense", "cachetune"]
        if (restart + body_tokens + repeat) % 2:
            order.reverse()
        orders.append(order)
        by_arm = {}
        for arm in order:
            if arm == "dense":
                by_arm[arm] = run_dense_round(
                    args,
                    tokenizer=tokenizer,
                    workload=workload,
                    body_tokens=body_tokens,
                    restart=restart,
                    repeat=repeat,
                    warmup=False,
                )
            else:
                by_arm[arm] = run_cachetune_round(
                    args,
                    tokenizer=tokenizer,
                    workload=workload,
                    body_tokens=body_tokens,
                    restart=restart,
                    repeat=repeat,
                    warmup=False,
                    expected_selected_tokens=expected_selected_tokens,
                    expected_recomputed_layers=expected_recomputed_layers,
                )
        rows.append(
            {
                "repeat": repeat,
                "order": order,
                "dense": by_arm["dense"],
                "cachetune": by_arm["cachetune"],
                "first_token_match": (
                    by_arm["dense"]["first_output_token"]
                    == by_arm["cachetune"]["first_output_token"]
                ),
            }
        )
    return {
        "body_tokens": body_tokens,
        "restart": restart,
        "controller": {
            "roofline_ratio": quantized.requested_ratio,
            "executable_ratio": quantized.executable_ratio,
            "repair_tokens": quantized.repair_tokens,
        },
        "warmup_order": warmup_order,
        "warmups": warmups,
        "formal_orders": orders,
        "rows": rows,
    }


def machine_manifest() -> dict[str, Any]:
    import torch
    import transformers

    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "transformers": transformers.__version__,
    }


def summarize(results: list[dict[str, Any]], body_tokens: tuple[int, ...]) -> dict:
    summary = {}
    for body in body_tokens:
        rows = [
            row
            for restart in results
            for body_result in restart["bodies"]
            if body_result["body_tokens"] == body
            for row in body_result["rows"]
        ]
        dense_target = [row["dense"]["costs"]["target_only_ms"] for row in rows]
        cache_target = [row["cachetune"]["costs"]["target_only_ms"] for row in rows]
        cache_adapter = [
            row["cachetune"]["costs"]["adapter_combined_ms"] for row in rows
        ]
        dense_request = [row["dense"]["costs"]["request_path_ms"] for row in rows]
        cache_request = [row["cachetune"]["costs"]["request_path_ms"] for row in rows]
        dense_full = [row["dense"]["costs"]["full_lifecycle_ms"] for row in rows]
        cache_full = [row["cachetune"]["costs"]["full_lifecycle_ms"] for row in rows]
        summary[str(body)] = {
            "formal_samples_per_arm": len(rows),
            "dense_target_p50_ms": statistics.median(dense_target),
            "cachetune_target_p50_ms": statistics.median(cache_target),
            "target_only_speedup": (
                statistics.median(dense_target) / statistics.median(cache_target)
            ),
            "adapter_combined_p50_ms": statistics.median(cache_adapter),
            "adapter_combined_speedup_vs_dense_target": (
                statistics.median(dense_target) / statistics.median(cache_adapter)
            ),
            "dense_request_path_p50_ms": statistics.median(dense_request),
            "cachetune_request_path_p50_ms": statistics.median(cache_request),
            "request_path_speedup": (
                statistics.median(dense_request) / statistics.median(cache_request)
            ),
            "dense_full_lifecycle_p50_ms": statistics.median(dense_full),
            "cachetune_full_lifecycle_p50_ms": statistics.median(cache_full),
            "full_lifecycle_speedup": (
                statistics.median(dense_full) / statistics.median(cache_full)
            ),
            "first_token_match_rate": (
                sum(row["first_token_match"] for row in rows) / len(rows)
            ),
            "all_rounds_observed_eviction": all(
                row[arm]["evicted_tokens_total_delta"] > 0
                for row in rows
                for arm in ("dense", "cachetune")
            ),
            "all_cachetune_rounds_no_fallback": all(
                row["cachetune"]["target_counter_delta"][
                    "sglang:approx_kv_dense_fallback_total"
                ]
                == 0
                for row in rows
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    body_tokens = parse_body_tokens(args.body_tokens)
    if args.restarts < 1 or args.repeats < 2:
        raise ValueError("restarts must be >=1 and repeats must be >=2")
    head_sha = current_git_sha()
    if head_sha != args.source_git_sha:
        raise RuntimeError(
            f"source_git_sha={args.source_git_sha} does not match HEAD={head_sha}"
        )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision, local_files_only=True
    )
    model_config = AutoConfig.from_pretrained(
        args.model, revision=args.model_revision, local_files_only=True
    )
    num_layers = int(model_config.num_hidden_layers)

    run_id = (
        f"phase4-r5-key-rerun-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    settings = {
        "model": args.model,
        "model_revision": args.model_revision,
        "model_fingerprint": args.model_fingerprint,
        "source_git_sha": args.source_git_sha,
        "image_digest": args.image_digest,
        "body_tokens": body_tokens,
        "header_tokens": args.header_tokens,
        "target_rho": args.target_rho,
        "mode": "speed_only",
        "hardware_measurement": {
            "t_c_ms": args.t_c_ms,
            "t_i_ms": args.t_i_ms,
            "t_o_ms": args.t_o_ms,
        },
        "segment_tokens": args.max_segment_chunk_tokens,
        "restarts": args.restarts,
        "warmup_per_arm": 1,
        "formal_repeats_per_arm": args.repeats,
        "causal_prefix_registration": True,
        "paired_dense_same_server_restart": True,
        "allow_no_eviction": args.allow_no_eviction,
        "cost_ledgers": (
            "target_only",
            "adapter_combined",
            "request_path",
            "full_lifecycle",
        ),
        "server_command_template": server_command(args, 0)[:-4]
        + [
            "--random-seed",
            "<restart-dependent-seed>",
            "--log-level",
            "warning",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    append_entry = {
        "run_id": run_id,
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
        "output": str(args.output.resolve()),
    }
    args.central_log.parent.mkdir(parents=True, exist_ok=True)
    with args.central_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(append_entry, sort_keys=True) + "\n")

    results = []
    try:
        for restart in range(args.restarts):
            log_path = args.output.parent / f"server-restart{restart}.log"
            server = start_server(args, restart, log_path)
            try:
                wait_ready(args, server)
                bodies = list(body_tokens)
                random.Random(args.order_seed + restart).shuffle(bodies)
                body_results = [
                    run_body(
                        args,
                        tokenizer=tokenizer,
                        num_layers=num_layers,
                        body_tokens=body,
                        restart=restart,
                    )
                    for body in bodies
                ]
                reset = capture_final_pool_reset_and_invariant(
                    f"http://127.0.0.1:{args.port}", tokenizer
                )
                if not reset["pool_invariant"]["passed"]:
                    raise RuntimeError(
                        f"final pool invariant failed: {reset['pool_invariant']}"
                    )
                results.append(
                    {
                        "restart": restart,
                        "server_log": str(log_path.resolve()),
                        "server_command": server_command(args, restart),
                        "body_order": bodies,
                        "bodies": body_results,
                        "final_pool_metrics": metric_subset(
                            reset["metrics_post_reset"]
                        ),
                        "final_pool_invariant": reset["pool_invariant"],
                    }
                )
            finally:
                stop_server(server)

        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "scope": "corrected-r5-key-rerun",
            "settings": settings,
            "machine": machine_manifest(),
            "restarts": results,
            "summary": summarize(results, body_tokens),
            "passed": True,
        }
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        completed = {
            "run_id": run_id,
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "settings": settings,
            "output": str(args.output.resolve()),
            "result_summary": payload["summary"],
        }
        with args.central_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(completed, sort_keys=True) + "\n")
    except Exception as exc:
        failed = {
            "run_id": run_id,
            "status": "failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "settings": settings,
            "output": str(args.output.resolve()),
            "error": f"{type(exc).__name__}: {exc}",
        }
        with args.central_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(failed, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    main()
