#!/usr/bin/env python3
"""Corrected Phase 4 R2 key rerun with paired dense baselines.

Runs body 1024/2048 at header64, rho2 and ratio1% across three server
processes. Raw and fresh chunks are registered from cumulative causal
prefixes, and every cost component is reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
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
    idle_pool_invariant,
    metric_subset,
    telemetry_delta,
    usable_kv_capacity_tokens,
)
from benchmark.approx_kv.run_phase4_cacheblend_pressure import (
    FRESH_HASH_PREFIX,
    RAW_HASH_PREFIX,
    _counter_delta,
    append_log,
    build_metadata,
    build_target_segments,
    compute_filler_count,
    expected_selected_tokens,
    filler_prompt,
    metric_snapshot,
    register_source_segments,
    request,
    segment_chunks,
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
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument(
        "--image-digest",
        default="sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--central-log", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30011)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--body-tokens", default="1024,2048")
    parser.add_argument("--header-tokens", type=int, default=64)
    parser.add_argument("--target-rho", type=float, default=2.0)
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--filler-tokens", type=int, default=736)
    parser.add_argument("--segment-tokens", type=int, default=512)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument("--chunked-prefill-size", type=int, default=1024)
    parser.add_argument("--server-start-timeout-s", type=float, default=600)
    parser.add_argument("--order-seed", type=int, default=20260724)
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
        str(17 + restart),
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
            "SGLANG_APPROX_KV_CACHEBLEND": "1",
            "SGLANG_CACHEBLEND_RATIO": str(args.ratio),
            "SGLANG_CACHEBLEND_PROBE_LAYERS": "0",
            "SGLANG_CACHEBLEND_FIRST_RECOMPUTE_LAYER": "1",
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


def post_flush(base_url: str) -> None:
    request_obj = urllib.request.Request(
        f"{base_url}/flush_cache", data=b"{}", method="POST"
    )
    request_obj.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request_obj, timeout=60):
        pass
    time.sleep(0.1)


def reset_and_snapshot(base_url: str, salt: int) -> dict[str, float]:
    post_flush(base_url)
    request(base_url, [80_000 + salt, 81_000 + salt])
    snapshot = metric_snapshot(base_url)
    post_flush(base_url)
    return snapshot


def observed_rho(metrics: dict[str, float], capacity: int) -> float:
    return (
        metrics.get("sglang:kv_used_tokens", 0.0)
        + metrics.get("sglang:kv_evictable_tokens", 0.0)
    ) / capacity


def seed_prompt(target_header: list[int], body_first_token: int) -> list[int]:
    sentinel = 79_000
    if sentinel == body_first_token:
        raise RuntimeError("seed sentinel collides with body first token")
    return target_header + [sentinel]


def first_output_token(result: dict) -> int | None:
    output_ids = result.get("output_ids") or []
    return int(output_ids[0]) if output_ids else None


def run_round(
    args: argparse.Namespace,
    *,
    body_tokens: int,
    arm: str,
    restart: int,
    repeat: int,
    warmup: bool,
) -> dict[str, Any]:
    base_url = f"http://127.0.0.1:{args.port}"
    salt = restart * 100 + body_tokens + repeat * 10 + (1 if arm == "cacheblend" else 0)
    baseline = reset_and_snapshot(base_url, salt)
    capacity = usable_kv_capacity_tokens(baseline)

    body = list(range(1_000, 1_000 + body_tokens))
    source_header = list(range(50_000, 50_000 + args.header_tokens))
    target_header = list(range(60_000, 60_000 + args.header_tokens))
    chunks = segment_chunks(body, args.segment_tokens)
    chunk_lengths = [len(chunk) for chunk in chunks]
    seed_ids = seed_prompt(target_header, body[0])

    seed = request(base_url, seed_ids)
    if seed["cached_tokens"] != 0:
        raise RuntimeError(f"initial head seed was not a miss: {seed}")

    content_hash_base = (
        f"r2-key-b{body_tokens}-restart{restart}-repeat{repeat}-"
        f"{'warmup' if warmup else 'formal'}"
    )
    raw_registration = None
    fresh_registration = None
    if arm == "cacheblend":
        raw_registration = register_source_segments(
            base_url,
            header=source_header,
            chunks=chunks,
            hash_prefix=RAW_HASH_PREFIX,
            content_hash_base=content_hash_base,
            sentinel_base=900,
        )
        fresh_registration = register_source_segments(
            base_url,
            header=target_header,
            chunks=chunks,
            hash_prefix=FRESH_HASH_PREFIX,
            content_hash_base=content_hash_base,
            sentinel_base=910,
        )

    after_setup = metric_snapshot(base_url)
    already_pinned_tokens = round(
        after_setup.get("sglang:kv_used_tokens", 0.0)
        - baseline.get("sglang:kv_used_tokens", 0.0)
    )
    if already_pinned_tokens < 0:
        raise RuntimeError(f"negative setup footprint: {already_pinned_tokens}")
    filler_count = compute_filler_count(
        capacity,
        args.target_rho,
        already_pinned_tokens,
        args.filler_tokens,
    )

    pressure_start = time.perf_counter()
    for filler_index in range(filler_count):
        request(
            base_url,
            filler_prompt(filler_index, args.filler_tokens) + [950],
        )
    pressure_ms = (time.perf_counter() - pressure_start) * 1000
    after_pressure = metric_snapshot(base_url)

    reseed = request(base_url, seed_ids)
    allowed_seed_hits = (0, args.header_tokens, len(seed_ids))
    if reseed["cached_tokens"] not in allowed_seed_hits:
        raise RuntimeError(
            f"unexpected post-pressure seed hit: {reseed['cached_tokens']}"
        )

    before_target = metric_snapshot(base_url)
    metadata = None
    if arm == "cacheblend":
        metadata = build_metadata(
            operation="reuse",
            segments=build_target_segments(
                chunk_lengths,
                header_tokens=args.header_tokens,
                hash_prefix=RAW_HASH_PREFIX,
                content_hash_base=content_hash_base,
            ),
            plugin="cacheblend",
        )
    target = request(base_url, target_header + body + [901], metadata)
    after_target = metric_snapshot(base_url)

    expected_cached = (
        args.header_tokens + body_tokens if arm == "cacheblend" else args.header_tokens
    )
    if target["cached_tokens"] != expected_cached:
        raise RuntimeError(
            f"{arm} cached_tokens={target['cached_tokens']}, expected={expected_cached}"
        )

    target_delta = telemetry_delta(before_target, after_target)
    mechanism = None
    if arm == "cacheblend":
        mechanism = {
            "selected_tokens_delta": _counter_delta(
                before_target,
                after_target,
                "sglang:approx_kv_cacheblend_selected_tokens_total",
            ),
            "recomputed_layers_delta": _counter_delta(
                before_target,
                after_target,
                "sglang:approx_kv_cacheblend_recomputed_layers_total",
            ),
            "precomputed_delta": _counter_delta(
                before_target,
                after_target,
                "sglang:approx_kv_cacheblend_precomputed_total",
            ),
            "expected_selected_tokens": expected_selected_tokens(
                body_tokens, args.ratio
            ),
        }
        if mechanism["selected_tokens_delta"] != mechanism["expected_selected_tokens"]:
            raise RuntimeError(f"selected-token mismatch: {mechanism}")
        if mechanism["precomputed_delta"] != 1:
            raise RuntimeError(f"precomputed adapter not observed: {mechanism}")
        if target_delta["dense_fallbacks"] not in (None, 0):
            raise RuntimeError("CacheBlend target used dense fallback")

    raw_ms = raw_registration["total_ms"] if raw_registration is not None else 0.0
    fresh_ms = fresh_registration["total_ms"] if fresh_registration is not None else 0.0
    target_ms = target["ttft_ms"]
    seed_ms = seed["ttft_ms"]
    reseed_ms = reseed["ttft_ms"]
    costs = {
        "target_only_ms": target_ms,
        "adapter_combined_ms": fresh_ms + target_ms,
        "request_path_ms": seed_ms + fresh_ms + reseed_ms + target_ms,
        "full_lifecycle_ms": seed_ms + raw_ms + fresh_ms + reseed_ms + target_ms,
        "seed_head_ms": seed_ms,
        "register_raw_ms": raw_ms,
        "register_fresh_ms": fresh_ms,
        "post_pressure_reseed_ms": reseed_ms,
        "pressure_ms": pressure_ms,
    }
    eviction_delta = after_target.get(
        "sglang:evicted_tokens_total", 0.0
    ) - baseline.get("sglang:evicted_tokens_total", 0.0)
    if eviction_delta <= 0:
        raise RuntimeError("round did not observe real eviction")

    return {
        "arm": arm,
        "body_tokens": body_tokens,
        "restart": restart,
        "repeat": repeat,
        "warmup": warmup,
        "capacity_tokens": capacity,
        "already_pinned_tokens": already_pinned_tokens,
        "filler_count": filler_count,
        "target_rho": args.target_rho,
        "rho_after_pressure": observed_rho(after_pressure, capacity),
        "rho_after_target": observed_rho(after_target, capacity),
        "evicted_tokens_total_delta": eviction_delta,
        "seed": seed,
        "post_pressure_reseed": {
            **reseed,
            "was_evicted_by_pressure": reseed["cached_tokens"] == 0,
        },
        "target": target,
        "first_output_token": first_output_token(target),
        "costs": costs,
        "raw_registration": raw_registration,
        "fresh_registration": fresh_registration,
        "mechanism": mechanism,
        "baseline_metrics": metric_subset(baseline),
        "after_setup_metrics": metric_subset(after_setup),
        "after_pressure_metrics": metric_subset(after_pressure),
        "after_target_metrics": metric_subset(after_target),
        "target_delta": target_delta,
    }


def run_body(
    args: argparse.Namespace, *, body_tokens: int, restart: int
) -> dict[str, Any]:
    rng = random.Random(args.order_seed + restart * 1000 + body_tokens)
    warmup_order = ["dense", "cacheblend"]
    rng.shuffle(warmup_order)
    warmups = [
        run_round(
            args,
            body_tokens=body_tokens,
            arm=arm,
            restart=restart,
            repeat=-1,
            warmup=True,
        )
        for arm in warmup_order
    ]

    rows = []
    orders = []
    for repeat in range(args.repeats):
        order = ["dense", "cacheblend"]
        if (restart + body_tokens + repeat) % 2:
            order.reverse()
        orders.append(order)
        by_arm = {}
        for arm in order:
            by_arm[arm] = run_round(
                args,
                body_tokens=body_tokens,
                arm=arm,
                restart=restart,
                repeat=repeat,
                warmup=False,
            )
        rows.append(
            {
                "repeat": repeat,
                "order": order,
                "dense": by_arm["dense"],
                "cacheblend": by_arm["cacheblend"],
                "first_token_match": (
                    by_arm["dense"]["first_output_token"]
                    == by_arm["cacheblend"]["first_output_token"]
                ),
            }
        )
    return {
        "body_tokens": body_tokens,
        "restart": restart,
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


def median(values: list[float]) -> float:
    return statistics.median(values)


def summarize(results: list[dict[str, Any]], body_tokens: tuple[int, ...]) -> dict:
    summary = {}
    for body in body_tokens:
        paired_rows = [
            row
            for restart in results
            for body_result in restart["bodies"]
            if body_result["body_tokens"] == body
            for row in body_result["rows"]
        ]
        dense_target = [row["dense"]["costs"]["target_only_ms"] for row in paired_rows]
        cache_target = [
            row["cacheblend"]["costs"]["target_only_ms"] for row in paired_rows
        ]
        cache_adapter = [
            row["cacheblend"]["costs"]["adapter_combined_ms"] for row in paired_rows
        ]
        dense_request = [
            row["dense"]["costs"]["request_path_ms"] for row in paired_rows
        ]
        cache_request = [
            row["cacheblend"]["costs"]["request_path_ms"] for row in paired_rows
        ]
        dense_full = [row["dense"]["costs"]["full_lifecycle_ms"] for row in paired_rows]
        cache_full = [
            row["cacheblend"]["costs"]["full_lifecycle_ms"] for row in paired_rows
        ]
        summary[str(body)] = {
            "formal_samples_per_arm": len(paired_rows),
            "dense_target_p50_ms": median(dense_target),
            "cacheblend_target_p50_ms": median(cache_target),
            "target_only_speedup": median(dense_target) / median(cache_target),
            "adapter_combined_p50_ms": median(cache_adapter),
            "adapter_combined_speedup_vs_dense_target": (
                median(dense_target) / median(cache_adapter)
            ),
            "dense_request_path_p50_ms": median(dense_request),
            "cacheblend_request_path_p50_ms": median(cache_request),
            "request_path_speedup": median(dense_request) / median(cache_request),
            "dense_full_lifecycle_p50_ms": median(dense_full),
            "cacheblend_full_lifecycle_p50_ms": median(cache_full),
            "full_lifecycle_speedup": median(dense_full) / median(cache_full),
            "first_token_match_rate": (
                sum(row["first_token_match"] for row in paired_rows) / len(paired_rows)
            ),
            "all_rounds_observed_eviction": all(
                row[arm]["evicted_tokens_total_delta"] > 0
                for row in paired_rows
                for arm in ("dense", "cacheblend")
            ),
            "all_cacheblend_rounds_no_fallback": all(
                row["cacheblend"]["target_delta"]["dense_fallbacks"] in (None, 0)
                for row in paired_rows
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    body_tokens = parse_body_tokens(args.body_tokens)
    if args.restarts < 1 or args.repeats < 2:
        raise ValueError("restarts must be >=1 and repeats must be >=2")
    if not math.isclose(args.ratio, 0.01, rel_tol=1e-9):
        raise ValueError("the corrected key rerun is fixed to ratio=0.01")
    head_sha = current_git_sha()
    if head_sha != args.source_git_sha:
        raise RuntimeError(
            f"source_git_sha={args.source_git_sha} does not match HEAD={head_sha}"
        )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    run_id = (
        f"phase4-r2-key-rerun-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    settings = {
        "model": args.model,
        "model_revision": args.model_revision,
        "source_git_sha": args.source_git_sha,
        "image_digest": args.image_digest,
        "body_tokens": body_tokens,
        "header_tokens": args.header_tokens,
        "target_rho": args.target_rho,
        "ratio": args.ratio,
        "segment_tokens": args.segment_tokens,
        "restarts": args.restarts,
        "warmup_per_arm": 1,
        "formal_repeats_per_arm": args.repeats,
        "causal_prefix_registration": True,
        "paired_dense_same_server_restart": True,
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
    append_log(
        str(args.central_log),
        {
            "run_id": run_id,
            "status": "running",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "settings": settings,
            "output": str(args.output.resolve()),
        },
    )

    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for restart in range(args.restarts):
            log_path = args.output.parent / f"server-restart{restart}.log"
            server = start_server(args, restart, log_path)
            try:
                wait_ready(args, server)
                bodies = list(body_tokens)
                random.Random(args.order_seed + restart).shuffle(bodies)
                body_results = [
                    run_body(args, body_tokens=body, restart=restart) for body in bodies
                ]
                final_metrics = reset_and_snapshot(
                    f"http://127.0.0.1:{args.port}", 9_000 + restart
                )
                pool_invariant = idle_pool_invariant(final_metrics)
                if not pool_invariant["passed"]:
                    raise RuntimeError(f"final pool invariant failed: {pool_invariant}")
                results.append(
                    {
                        "restart": restart,
                        "server_log": str(log_path.resolve()),
                        "server_command": server_command(args, restart),
                        "body_order": bodies,
                        "bodies": body_results,
                        "final_pool_metrics": metric_subset(final_metrics),
                        "final_pool_invariant": pool_invariant,
                    }
                )
            finally:
                stop_server(server)

        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "scope": "corrected-r2-key-rerun",
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
        append_log(
            str(args.central_log),
            {
                "run_id": run_id,
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "settings": settings,
                "output": str(args.output.resolve()),
                "result_summary": payload["summary"],
            },
        )
    except Exception as exc:
        append_log(
            str(args.central_log),
            {
                "run_id": run_id,
                "status": "failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "settings": settings,
                "output": str(args.output.resolve()),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


if __name__ == "__main__":
    main()
