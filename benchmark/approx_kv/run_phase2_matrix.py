#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp

from benchmark.approx_kv.bench_phase2_pressure import (
    benchmark_server,
    request_succeeded,
    send_request,
)
from benchmark.approx_kv.metrics import (
    clean_cache_invariant,
    max_total_num_tokens,
    parse_prometheus_text,
    usable_kv_capacity_tokens,
)
from benchmark.approx_kv.workloads import (
    TraceInvocation,
    build_messages,
    build_object_catalog,
    build_workflow_trace,
    common_prefix_token_ids,
    select_objects_for_pressure,
    tokenize_messages,
    trace_physical_token_count,
    unique_prefix_token_count,
)


@dataclass
class ServerProcess:
    process: subprocess.Popen
    log_file: Any
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument("--chunked-prefill-size", type=int, default=1024)
    parser.add_argument("--catalog-size", type=int, default=24)
    parser.add_argument(
        "--target-prefix-sizes",
        default="512,1024,2048,4096",
    )
    parser.add_argument(
        "--pressure-points",
        default="0.8,1.0,1.5,2.0,3.0",
    )
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--measured-repeats", type=int, default=1)
    parser.add_argument("--order-seed", type=int, default=20260721)
    parser.add_argument("--server-seed", type=int, default=17)
    parser.add_argument("--request-timeout-s", type=float, default=300)
    parser.add_argument("--server-start-timeout-s", type=float, default=600)
    parser.add_argument("--model-name", default="default")
    parser.add_argument("--attention-backend", default="torch_native")
    parser.add_argument("--sampling-backend", default="pytorch")
    parser.add_argument(
        "--metrics-scrape-mode",
        choices=("boundary", "per_request"),
        default="boundary",
    )
    parser.add_argument("--skip-object-calibration", action="store_true")
    parser.add_argument("--probe-count", type=int, default=3)
    return parser.parse_args()


def parse_csv_numbers(value: str, cast) -> tuple:
    result = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("expected at least one comma-separated value")
    return result


def fetch_text(url: str, timeout: float = 10) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def post_empty(url: str, timeout: float = 30) -> str:
    request = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def tail_text(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def server_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
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
        args.attention_backend,
        "--sampling-backend",
        args.sampling_backend,
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--radix-eviction-policy",
        "lru",
        "--enable-cache-report",
        "--enable-metrics",
        "--random-seed",
        str(args.server_seed),
        "--log-level",
        "warning",
    ]


def start_server(args: argparse.Namespace, log_path: Path) -> ServerProcess:
    log_file = log_path.open("w")
    environment = os.environ.copy()
    environment.setdefault(
        "PYTORCH_ALLOC_CONF",
        "expandable_segments:True",
    )
    process = subprocess.Popen(
        server_command(args),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )
    return ServerProcess(process=process, log_file=log_file, log_path=log_path)


def stop_server(server: ServerProcess) -> None:
    process = server.process
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    server.log_file.close()


def wait_ready(args: argparse.Namespace, server: ServerProcess) -> None:
    deadline = time.monotonic() + args.server_start_timeout_s
    health_url = f"http://127.0.0.1:{args.port}/health_generate"
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            server.log_file.flush()
            raise RuntimeError(
                f"server exited during startup with code {server.process.returncode}\n"
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


def wait_clean_metrics(
    base_url: str,
    timeout_s: float = 60,
) -> dict[str, float]:
    deadline = time.monotonic() + timeout_s
    latest: dict[str, float] = {}
    while time.monotonic() < deadline:
        latest = parse_prometheus_text(fetch_text(f"{base_url}/metrics"))
        if clean_cache_invariant(latest)["passed"]:
            return latest
        time.sleep(0.5)
    raise RuntimeError(
        f"clean KV metrics did not settle: {clean_cache_invariant(latest)}"
    )


def wait_capacity(
    base_url: str,
    timeout_s: float = 60,
) -> tuple[int, int, dict[str, float]]:
    latest = wait_clean_metrics(base_url, timeout_s=timeout_s)
    return (
        max_total_num_tokens(latest),
        usable_kv_capacity_tokens(latest),
        latest,
    )


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


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_pressure: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        key = str(run["target_pressure"])
        by_pressure.setdefault(key, []).append(run)
    return {
        pressure: {
            "runs": len(items),
            "all_successful": all(
                item["summary"]["completion_rate"] == 1.0
                and item["clean_baseline_invariant"]["passed"]
                and item["reset_invariant"]["passed"]
                and not item["lifecycle_error"]
                for item in items
            ),
            "probe_ttft_p50_ms": [
                item["summary"]["probe_ttft_p50_ms"] for item in items
            ],
            "actual_reusable_pressure": [
                item["actual_reusable_pressure"] for item in items
            ],
            "physical_trace_pressure": [
                item["physical_trace_pressure"] for item in items
            ],
            "evicted_tokens": [
                item["telemetry_delta"]["counters"].get(
                    "sglang:evicted_tokens_total"
                )
                for item in items
            ],
        }
        for pressure, items in sorted(
            by_pressure.items(),
            key=lambda item: float(item[0]),
        )
    }


def _calibration_invocation(
    *,
    step: int,
    cache_object,
    variant: str,
) -> TraceInvocation:
    return TraceInvocation(
        step=step,
        phase="object-calibration",
        object_id=cache_object.object_id,
        role=cache_object.role,
        occurrence=step,
        suffix=f"invocation={step:06d};variant={variant}",
        next_use_step=None,
        next_use_distance=None,
        intervening_unique_prefix_tokens=None,
    )


async def calibrate_cache_objects(
    *,
    base_url: str,
    model: str,
    tokenizer,
    catalog,
    request_timeout_s: float,
) -> list[dict[str, Any]]:
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    records = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for index, cache_object in enumerate(catalog):
            cache_salt = f"object-calibration-{index:02d}"
            cold_invocation = _calibration_invocation(
                step=index * 3,
                cache_object=cache_object,
                variant="A",
            )
            variant_invocation = _calibration_invocation(
                step=index * 3 + 1,
                cache_object=cache_object,
                variant="B",
            )
            sample_kind = "object-calibration"
            repeat = 0

            def full_suffix(invocation: TraceInvocation) -> str:
                return (
                    f"{invocation.suffix};"
                    f"sample={sample_kind};"
                    f"repeat={repeat:03d}"
                )

            expected_prefix = len(
                common_prefix_token_ids(
                    tokenize_messages(
                        tokenizer,
                        build_messages(
                            cache_object,
                            full_suffix(cold_invocation),
                            cache_salt=cache_salt,
                        ),
                    ),
                    tokenize_messages(
                        tokenizer,
                        build_messages(
                            cache_object,
                            full_suffix(variant_invocation),
                            cache_salt=cache_salt,
                        ),
                    ),
                )
            )
            cold = await send_request(
                session,
                base_url=base_url,
                model=model,
                cache_object=cache_object,
                invocation=cold_invocation,
                sample_kind=sample_kind,
                repeat=repeat,
                cache_salt=cache_salt,
                is_probe=False,
                scrape_metrics=False,
            )
            variant = await send_request(
                session,
                base_url=base_url,
                model=model,
                cache_object=cache_object,
                invocation=variant_invocation,
                sample_kind=sample_kind,
                repeat=repeat,
                cache_salt=cache_salt,
                is_probe=False,
                scrape_metrics=False,
            )
            repeat_variant = await send_request(
                session,
                base_url=base_url,
                model=model,
                cache_object=cache_object,
                invocation=variant_invocation,
                sample_kind=sample_kind,
                repeat=repeat,
                cache_salt=cache_salt,
                is_probe=False,
                scrape_metrics=False,
            )
            passed = (
                request_succeeded(cold)
                and request_succeeded(variant)
                and request_succeeded(repeat_variant)
                and abs(variant.cached_tokens - expected_prefix) <= 64
                and repeat_variant.cached_tokens >= variant.cached_tokens
            )
            record = {
                "object_id": cache_object.object_id,
                "expected_variant_prefix_tokens": expected_prefix,
                "cold": asdict(cold),
                "variant": asdict(variant),
                "repeat_variant": asdict(repeat_variant),
                "tolerance_tokens": 64,
                "passed": passed,
            }
            records.append(record)
            if not passed:
                break
    return records


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_sizes = parse_csv_numbers(args.target_prefix_sizes, int)
    pressure_points = parse_csv_numbers(args.pressure_points, float)
    if args.restarts <= 0:
        raise ValueError("--restarts must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=True,
    )
    catalog = build_object_catalog(
        tokenizer,
        object_count=args.catalog_size,
        target_sizes=target_sizes,
    )
    dataset_manifest = {
        "catalog_size": len(catalog),
        "target_prefix_sizes": target_sizes,
        "sum_object_prefix_tokens": sum(
            cache_object.reusable_prefix_tokens for cache_object in catalog
        ),
        "unique_catalog_prefix_tokens": unique_prefix_token_count(catalog),
        "objects": [cache_object.manifest() for cache_object in catalog],
    }
    (args.output_dir / "dataset.json").write_text(
        json.dumps(dataset_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    base_url = f"http://127.0.0.1:{args.port}"
    if args.probe_count <= 0 or args.probe_count > len(catalog):
        raise ValueError("--probe-count must be within the catalog")
    probe_object_ids = {
        cache_object.object_id for cache_object in catalog[: args.probe_count]
    }

    reference_dir = args.output_dir / "reference-calibration"
    reference_dir.mkdir()
    reference_server = start_server(args, reference_dir / "server.log")
    try:
        wait_ready(args, reference_server)
        post_empty(f"{base_url}/flush_cache?timeout=30")
        fetch_text(f"{base_url}/health_generate")
        (
            reference_max_tokens,
            reference_capacity,
            reference_clean_metrics,
        ) = wait_capacity(base_url)
        object_calibration = []
        if not args.skip_object_calibration:
            object_calibration = asyncio.run(
                calibrate_cache_objects(
                    base_url=base_url,
                    model=args.model_name,
                    tokenizer=tokenizer,
                    catalog=catalog,
                    request_timeout_s=args.request_timeout_s,
                )
            )
            if len(object_calibration) != len(catalog) or not all(
                record["passed"] for record in object_calibration
            ):
                raise RuntimeError("object cache calibration failed")
            post_empty(f"{base_url}/flush_cache?timeout=30")
            fetch_text(f"{base_url}/health_generate")
            reference_clean_metrics = wait_clean_metrics(base_url)
        reference_record = {
            "max_total_num_tokens": reference_max_tokens,
            "usable_kv_capacity_tokens": reference_capacity,
            "clean_metrics": reference_clean_metrics,
            "clean_invariant": clean_cache_invariant(reference_clean_metrics),
            "object_calibration_skipped": args.skip_object_calibration,
            "object_calibration": object_calibration,
        }
        (reference_dir / "result.json").write_text(
            json.dumps(reference_record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    finally:
        stop_server(reference_server)

    frozen_configs: dict[float, dict[str, Any]] = {}
    for pressure in pressure_points:
        selection = select_objects_for_pressure(
            catalog,
            gpu_kv_capacity_tokens=reference_capacity,
            target_ratio=pressure,
            required_object_ids=probe_object_ids,
        )
        trace = build_workflow_trace(selection.objects)
        frozen_configs[pressure] = {
            "selection": selection,
            "trace": trace,
            "physical_trace_tokens": trace_physical_token_count(
                tokenizer,
                selection.objects,
                trace,
            ),
        }

    config_order = []
    for restart in range(args.restarts):
        restart_pressures = list(pressure_points)
        random.Random(args.order_seed + restart).shuffle(restart_pressures)
        config_order.extend(
            {
                "target_pressure": pressure,
                "restart": restart,
            }
            for pressure in restart_pressures
        )

    runs: list[dict[str, Any]] = []

    for order_index, config in enumerate(config_order):
        pressure = float(config["target_pressure"])
        restart = int(config["restart"])
        pressure_name = str(pressure).replace(".", "p")
        run_dir = args.output_dir / (
            f"{order_index:02d}-rho-{pressure_name}-restart-{restart}"
        )
        run_dir.mkdir()
        server = start_server(args, run_dir / "server.log")
        try:
            wait_ready(args, server)
            post_empty(f"{base_url}/flush_cache?timeout=30")
            fetch_text(f"{base_url}/health_generate")
            max_tokens, capacity, startup_metrics = wait_capacity(base_url)
            capacity_tolerance = max(16, round(reference_capacity * 0.01))
            if abs(capacity - reference_capacity) > capacity_tolerance:
                raise RuntimeError(
                    f"KV capacity changed across restarts: "
                    f"{capacity} != {reference_capacity}"
                )
            frozen = frozen_configs[pressure]
            selection = frozen["selection"]
            trace = frozen["trace"]
            physical_trace_tokens = int(frozen["physical_trace_tokens"])
            payload = asyncio.run(
                benchmark_server(
                    base_url=base_url,
                    model=args.model_name,
                    objects=selection.objects,
                    trace=trace,
                    warmup_repeats=args.warmup_repeats,
                    measured_repeats=args.measured_repeats,
                    request_timeout_s=args.request_timeout_s,
                    probe_object_ids=probe_object_ids,
                    metrics_scrape_mode=args.metrics_scrape_mode,
                )
            )
            run_record = {
                "order_index": order_index,
                "restart": restart,
                "target_pressure": pressure,
                "reference_reusable_pressure": selection.actual_ratio,
                "actual_reusable_pressure": (
                    selection.active_reusable_tokens / capacity
                ),
                "physical_trace_pressure": physical_trace_tokens / capacity,
                "active_reusable_tokens": selection.active_reusable_tokens,
                "physical_trace_tokens": physical_trace_tokens,
                "reference_kv_capacity_tokens": reference_capacity,
                "gpu_kv_capacity_tokens": capacity,
                "max_total_num_tokens": max_tokens,
                "selected_object_ids": [
                    cache_object.object_id for cache_object in selection.objects
                ],
                "selected_object_count": len(selection.objects),
                "probe_object_ids": sorted(probe_object_ids),
                "trace_length": len(trace),
                "trace": [asdict(invocation) for invocation in trace],
                "startup_metrics": startup_metrics,
                **payload,
            }
            (run_dir / "result.json").write_text(
                json.dumps(run_record, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            runs.append(run_record)
            if run_record["summary"]["completion_rate"] != 1.0:
                raise RuntimeError(
                    f"request failures in {run_dir}: "
                    f"{run_record['summary']}"
                )
            if not run_record["idle_pool_invariant"]["passed"]:
                raise RuntimeError(
                    f"KV pool invariant failed in {run_dir}: "
                    f"{run_record['idle_pool_invariant']}"
                )
            if not run_record["clean_baseline_invariant"]["passed"]:
                raise RuntimeError(
                    f"clean baseline failed in {run_dir}: "
                    f"{run_record['clean_baseline_invariant']}"
                )
            if not run_record["reset_invariant"]["passed"]:
                raise RuntimeError(
                    f"final reset failed in {run_dir}: "
                    f"{run_record['reset_invariant']}"
                )
            if run_record["lifecycle_error"]:
                raise RuntimeError(
                    f"lifecycle failure in {run_dir}: "
                    f"{run_record['lifecycle_error']}"
                )
        finally:
            stop_server(server)

    manifest = {
        "source_git_sha": args.source_git_sha,
        "image_digest": args.image_digest,
        "model": args.model,
        "model_revision": args.model_revision,
        "reference_calibration": {
            "max_total_num_tokens": reference_max_tokens,
            "usable_kv_capacity_tokens": reference_capacity,
            "object_calibration_skipped": args.skip_object_calibration,
        },
        "machine": machine_manifest(),
        "config": {
            **vars(args),
            "output_dir": str(args.output_dir),
            "target_prefix_sizes": target_sizes,
            "pressure_points": pressure_points,
            "server_environment": {
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            },
        },
        "randomized_config_order": config_order,
        "runs": [
            {
                "order_index": run["order_index"],
                "restart": run["restart"],
                "target_pressure": run["target_pressure"],
                "actual_reusable_pressure": run["actual_reusable_pressure"],
                "physical_trace_pressure": run["physical_trace_pressure"],
                "selected_object_count": run["selected_object_count"],
                "summary": run["summary"],
                "telemetry_delta": run["telemetry_delta"],
            }
            for run in runs
        ],
        "summary": summarize_runs(runs),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
