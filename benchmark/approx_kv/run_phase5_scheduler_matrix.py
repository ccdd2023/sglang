#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from benchmark.approx_kv.bench_phase2_pressure import benchmark_server
from benchmark.approx_kv.metrics import (
    clean_cache_invariant,
    max_total_num_tokens,
    parse_prometheus_text,
    usable_kv_capacity_tokens,
)
from benchmark.approx_kv.workloads import (
    CacheObject,
    TraceInvocation,
    build_object_catalog,
    select_objects_for_pressure,
    unique_prefix_token_count,
)

POLICY_LABELS = {
    "lru": "S0 LRU",
    "workflow_steps": "S1 steps-only",
    "belady": "S2 Belady oracle",
    "recovery_value": "S3 recovery-aware value density",
    "hierarchical": "S4 hierarchical object policy",
}


@dataclass
class ServerProcess:
    process: subprocess.Popen
    log_file: Any
    log_path: Path


def csv_values(value: str, cast) -> tuple:
    values = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated values")
    return values


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-name", default="default")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--central-log", type=Path, required=True)
    parser.add_argument(
        "--policies",
        default="lru,workflow_steps,belady,recovery_value,hierarchical",
    )
    parser.add_argument("--prefetch-modes", default="p0")
    parser.add_argument("--pressure-points", default="1.1,1.5,2.0,3.0")
    parser.add_argument("--catalog-size", type=positive_int, default=48)
    parser.add_argument("--target-prefix-sizes", default="1024")
    parser.add_argument("--workflow-cycles", type=positive_int, default=2)
    parser.add_argument("--restarts", type=positive_int, default=1)
    parser.add_argument("--warmup-repeats", type=positive_int, default=1)
    parser.add_argument("--formal-repeats", type=positive_int, default=2)
    parser.add_argument("--port", type=int, default=30011)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument("--chunked-prefill-size", type=positive_int, default=1024)
    parser.add_argument("--server-seed", type=int, default=17)
    parser.add_argument("--order-seed", type=int, default=20260724)
    parser.add_argument("--server-start-timeout-s", type=float, default=600)
    parser.add_argument("--request-timeout-s", type=float, default=300)
    parser.add_argument("--attention-backend", default="torch_native")
    parser.add_argument("--sampling-backend", default="pytorch")
    parser.add_argument(
        "--metrics-scrape-mode",
        choices=("boundary", "per_request"),
        default="boundary",
    )
    parser.add_argument("--enable-hicache", action="store_true")
    parser.add_argument("--hicache-ratio", type=float, default=2.0)
    parser.add_argument("--kv-bytes-per-token", type=positive_int, default=114688)
    return parser.parse_args()


def append_log(path: Path, entry: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(entry), sort_keys=True))
        file.write("\n")


def fetch_text(url: str, timeout: float = 10) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def post_empty(url: str, timeout: float = 30) -> str:
    request = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def tail_text(path: Path, lines: int = 160) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def server_command(
    args: argparse.Namespace,
    *,
    policy: str,
    prefetch_mode: str,
) -> list[str]:
    command = [
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
        policy,
        "--workflow-prefetch-policy",
        prefetch_mode,
        "--enable-cache-report",
        "--enable-metrics",
        "--random-seed",
        str(args.server_seed),
        "--log-level",
        "warning",
    ]
    if args.enable_hicache:
        command.extend(
            (
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                str(args.hicache_ratio),
                "--hicache-write-policy",
                "write_through",
                "--hicache-io-backend",
                "kernel",
            )
        )
    return command


def start_server(
    args: argparse.Namespace,
    *,
    policy: str,
    prefetch_mode: str,
    log_path: Path,
) -> ServerProcess:
    log_file = log_path.open("w")
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    process = subprocess.Popen(
        server_command(args, policy=policy, prefetch_mode=prefetch_mode),
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


def wait_clean_metrics(base_url: str, timeout_s: float = 60) -> dict[str, float]:
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


def capacity_snapshot(base_url: str) -> tuple[int, int, dict[str, float]]:
    post_empty(f"{base_url}/flush_cache?timeout=30")
    fetch_text(f"{base_url}/health_generate")
    metrics = wait_clean_metrics(base_url)
    return (
        max_total_num_tokens(metrics),
        usable_kv_capacity_tokens(metrics),
        metrics,
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


def active_workflow_objects(
    catalog: tuple[CacheObject, ...],
) -> tuple[CacheObject, ...]:
    by_role: dict[str, list[CacheObject]] = {}
    for cache_object in catalog:
        by_role.setdefault(cache_object.role, []).append(cache_object)
    required = {
        "architect": 1,
        "coder": 2,
        "debugger": 2,
    }
    selected = []
    for role, count in required.items():
        candidates = by_role.get(role, [])
        if len(candidates) < count:
            raise ValueError(f"catalog does not contain {count} {role} objects")
        selected.extend(candidates[:count])
    architect = next(item for item in selected if item.role == "architect")
    coders = [item for item in selected if item.role == "coder"]
    debuggers = [item for item in selected if item.role == "debugger"]
    return (architect, coders[0], debuggers[0], coders[1], debuggers[1])


def build_phase5_trace(
    *,
    selected_objects: tuple[CacheObject, ...],
    workflow_sequence: tuple[CacheObject, ...],
    workflow_cycles: int,
) -> tuple[TraceInvocation, ...]:
    workflow_ids = {item.object_id for item in workflow_sequence}
    fillers = [item for item in selected_objects if item.object_id not in workflow_ids]
    dead_count = max(1, len(fillers) // 3) if fillers else 0
    dead_fillers = fillers[:dead_count]
    live_fillers = fillers[dead_count:]
    phases: list[tuple[str, CacheObject]] = []
    phases.extend(("fill", item) for item in workflow_sequence)
    phases.extend(("backup", item) for item in workflow_sequence)
    phases.extend(("pressure_live_fill", item) for item in live_fillers)
    phases.extend(("pressure_live_backup", item) for item in live_fillers)
    phases.extend(("pressure_dead", item) for item in dead_fillers)
    for _ in range(workflow_cycles):
        phases.extend(("workflow", item) for item in workflow_sequence)
    phases.extend(("pressure_replay", item) for item in live_fillers)

    occurrences: dict[str, int] = {}
    raw = []
    for step, (phase, cache_object) in enumerate(phases):
        occurrence = occurrences.get(cache_object.object_id, 0)
        occurrences[cache_object.object_id] = occurrence + 1
        raw.append(
            {
                "step": step,
                "phase": phase,
                "object_id": cache_object.object_id,
                "role": cache_object.role,
                "occurrence": occurrence,
                "suffix": (
                    f"phase5-step={step:06d};"
                    f"phase={phase};"
                    f"occurrence={occurrence:03d}"
                ),
            }
        )

    next_use: dict[str, int] = {}
    result = []
    active_phases = {
        "backup",
        "pressure_live_backup",
        "workflow",
        "pressure_replay",
    }
    active_ordinal = {}
    ordinal = 0
    for item in raw:
        if item["phase"] in active_phases:
            ordinal += 1
        active_ordinal[item["step"]] = ordinal
    for item in reversed(raw):
        next_step = next_use.get(item["object_id"])
        result.append(
            TraceInvocation(
                **item,
                next_use_step=(
                    None if next_step is None else active_ordinal[next_step]
                ),
                next_use_distance=(
                    None if next_step is None else next_step - item["step"]
                ),
                intervening_unique_prefix_tokens=None,
                next_use_request_step=next_step,
            )
        )
        next_use[item["object_id"]] = item["step"]
    result.reverse()
    return tuple(result)


def next_stage_hints(
    trace: tuple[TraceInvocation, ...],
) -> dict[int, tuple[str, int]]:
    workflow_steps = [
        invocation.step for invocation in trace if invocation.phase == "workflow"
    ]
    hints: dict[int, tuple[str, int]] = {}
    if workflow_steps:
        pressure_steps = [
            invocation.step
            for invocation in trace
            if invocation.phase in ("pressure_live_backup", "pressure_dead")
        ]
        if pressure_steps:
            first = trace[workflow_steps[0]]
            hints[pressure_steps[-1]] = (
                first.object_id,
                workflow_steps[0],
            )
    for current_step, next_step in zip(workflow_steps, workflow_steps[1:]):
        target = trace[next_step]
        hints[current_step] = (
            target.object_id,
            next_step,
        )
    return hints


def custom_params_factory(
    *,
    selected_objects: tuple[CacheObject, ...],
    trace: tuple[TraceInvocation, ...],
    kv_bytes_per_token: int,
    enable_hicache: bool,
):
    object_map = {item.object_id: item for item in selected_objects}
    reusable_ids = {
        invocation.object_id
        for invocation in trace
        if invocation.phase in ("workflow", "pressure_replay")
    }
    hints = next_stage_hints(trace)

    def build(
        cache_object: CacheObject,
        invocation: TraceInvocation,
    ) -> Mapping[str, Any]:
        active = cache_object.object_id in reusable_ids
        metadata = {
            "object_id": cache_object.object_id,
            "protected_tokens": cache_object.reusable_prefix_tokens,
            "resident_bytes": (
                cache_object.reusable_prefix_tokens * kv_bytes_per_token
            ),
            "dense_cost_ms": cache_object.dense_cost_weight / 10.0,
            "recovery_cost_ms": cache_object.recovery_cost_weight / 10.0,
            "current_step": invocation.step,
            "next_use_step": invocation.next_use_step,
            "next_use_request_step": invocation.next_use_request_step,
            "next_use_distance": invocation.next_use_distance,
            "workflow_stage": cache_object.role,
            "object_kind": cache_object.kind.value,
            "recoverable_from_lower_tier": enable_hicache and active,
            "retired": invocation.next_use_distance is None,
        }
        result: dict[str, Any] = {"cache_protection": metadata}
        hint = hints.get(invocation.step)
        if hint is not None:
            object_id, next_step = hint
            if object_id not in object_map:
                raise RuntimeError(
                    f"prefetch hint references unknown object {object_id}"
                )
            result["cache_prefetch"] = {
                "object_id": object_id,
                "next_use_step": next_step,
            }
        return result

    return build


def workflow_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in payload["results"]
        if row["phase"] == "workflow" and not row["error"]
    ]
    ttfts = [float(row["ttft_ms"]) for row in rows]
    cached = sum(int(row["cached_tokens"]) for row in rows)
    expected = sum(int(row["expected_reusable_prefix_tokens"]) for row in rows)
    return {
        "requests": len(rows),
        "ttft_p50_ms": statistics.median(ttfts) if ttfts else 0.0,
        "ttft_mean_ms": statistics.mean(ttfts) if ttfts else 0.0,
        "ttft_total_ms": sum(ttfts),
        "ttft_p95_ms": (
            sorted(ttfts)[round((len(ttfts) - 1) * 0.95)] if ttfts else 0.0
        ),
        "cached_tokens": cached,
        "expected_reusable_tokens": expected,
        "cache_hit_fraction": min(1.0, cached / expected) if expected else 0.0,
        "per_object": {
            object_id: {
                "requests": len(items),
                "ttft_p50_ms": statistics.median(
                    [float(item["ttft_ms"]) for item in items]
                ),
                "cached_tokens": [int(item["cached_tokens"]) for item in items],
            }
            for object_id in sorted({row["object_id"] for row in rows})
            for items in [[row for row in rows if row["object_id"] == object_id]]
        },
    }


def run_config(
    args: argparse.Namespace,
    *,
    run_id: str,
    order_index: int,
    restart: int,
    policy: str,
    prefetch_mode: str,
    target_pressure: float,
    capacity: int,
    selected_objects: tuple[CacheObject, ...],
    workflow_sequence: tuple[CacheObject, ...],
) -> dict[str, Any]:
    policy_dir = policy.replace("_", "-")
    pressure_name = str(target_pressure).replace(".", "p")
    run_dir = args.output_dir / (
        f"{order_index:03d}-{policy_dir}-{prefetch_mode}-"
        f"rho{pressure_name}-restart{restart}"
    )
    run_dir.mkdir()
    server = start_server(
        args,
        policy=policy,
        prefetch_mode=prefetch_mode,
        log_path=run_dir / "server.log",
    )
    base_url = f"http://127.0.0.1:{args.port}"
    settings = {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "prefetch_mode": prefetch_mode,
        "target_pressure": target_pressure,
        "restart": restart,
        "enable_hicache": args.enable_hicache,
        "formal_repeats": args.formal_repeats,
        "warmup_repeats": args.warmup_repeats,
        "mem_fraction_static": args.mem_fraction_static,
    }
    append_log(
        args.central_log,
        {
            "run_id": run_id,
            "status": "running",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "settings": settings,
            "output": str(run_dir.resolve()),
        },
    )
    try:
        wait_ready(args, server)
        max_tokens, run_capacity, startup_metrics = capacity_snapshot(base_url)
        tolerance = max(16, round(capacity * 0.01))
        if abs(run_capacity - capacity) > tolerance:
            raise RuntimeError(
                f"KV capacity changed across restarts: {run_capacity} != {capacity}"
            )
        trace = build_phase5_trace(
            selected_objects=selected_objects,
            workflow_sequence=workflow_sequence,
            workflow_cycles=args.workflow_cycles,
        )
        payload = asyncio.run(
            benchmark_server(
                base_url=base_url,
                model=args.model_name,
                objects=selected_objects,
                trace=trace,
                warmup_repeats=args.warmup_repeats,
                measured_repeats=args.formal_repeats,
                request_timeout_s=args.request_timeout_s,
                probe_object_ids={item.object_id for item in workflow_sequence},
                metrics_scrape_mode=args.metrics_scrape_mode,
                custom_params_factory=custom_params_factory(
                    selected_objects=selected_objects,
                    trace=trace,
                    kv_bytes_per_token=args.kv_bytes_per_token,
                    enable_hicache=args.enable_hicache,
                ),
                reset_between_measured_repeats=True,
            )
        )
        (run_dir / "raw-payload.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if payload["summary"]["completion_rate"] != 1.0:
            raise RuntimeError(f"request failures: {payload['summary']}")
        for key in (
            "clean_baseline_invariant",
            "idle_pool_invariant",
            "reset_invariant",
        ):
            if not payload[key]["passed"]:
                raise RuntimeError(f"{key} failed: {payload[key]}")
        if payload["lifecycle_error"]:
            raise RuntimeError(payload["lifecycle_error"])
        eviction_delta = (
            payload["telemetry_delta"]["counters"].get("sglang:evicted_tokens_total")
            or 0
        )
        if target_pressure > 1.0 and eviction_delta <= 0:
            raise RuntimeError("pressure run did not trigger real eviction")

        record = {
            "schema_version": 1,
            "run_id": run_id,
            "order_index": order_index,
            "settings": settings,
            "max_total_num_tokens": max_tokens,
            "gpu_kv_capacity_tokens": run_capacity,
            "actual_reusable_pressure": (
                unique_prefix_token_count(selected_objects) / run_capacity
            ),
            "selected_object_ids": [item.object_id for item in selected_objects],
            "workflow_object_ids": [item.object_id for item in workflow_sequence],
            "trace": [asdict(item) for item in trace],
            "startup_metrics": startup_metrics,
            "workflow_summary": workflow_summary(payload),
            **payload,
        }
        (run_dir / "result.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        append_log(
            args.central_log,
            {
                "run_id": run_id,
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "settings": settings,
                "output": str((run_dir / "result.json").resolve()),
                "result_summary": record["workflow_summary"],
            },
        )
        return record
    except Exception as exc:
        append_log(
            args.central_log,
            {
                "run_id": run_id,
                "status": "failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "settings": settings,
                "output": str(run_dir.resolve()),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        stop_server(server)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policies = csv_values(args.policies, str)
    prefetch_modes = csv_values(args.prefetch_modes, str)
    pressure_points = csv_values(args.pressure_points, float)
    target_sizes = csv_values(args.target_prefix_sizes, int)
    unknown_policies = set(policies) - set(POLICY_LABELS)
    if unknown_policies:
        raise ValueError(f"unknown policies: {sorted(unknown_policies)}")
    if set(prefetch_modes) - {"p0", "p1", "p2", "p3"}:
        raise ValueError(f"unknown prefetch modes: {prefetch_modes}")
    if not args.enable_hicache and any(mode != "p0" for mode in prefetch_modes):
        raise ValueError("p1-p3 require --enable-hicache")
    if args.formal_repeats < 2:
        raise ValueError("--formal-repeats must be at least 2")

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
    workflow_sequence = active_workflow_objects(catalog)
    required_ids = {item.object_id for item in workflow_sequence}

    reference_dir = args.output_dir / "reference"
    reference_dir.mkdir()
    reference_server = start_server(
        args,
        policy=policies[0],
        prefetch_mode=prefetch_modes[0],
        log_path=reference_dir / "server.log",
    )
    base_url = f"http://127.0.0.1:{args.port}"
    try:
        wait_ready(args, reference_server)
        reference_max_tokens, capacity, clean_metrics = capacity_snapshot(base_url)
    finally:
        stop_server(reference_server)

    frozen: dict[float, tuple[CacheObject, ...]] = {}
    for pressure in pressure_points:
        selection = select_objects_for_pressure(
            catalog,
            gpu_kv_capacity_tokens=capacity,
            target_ratio=pressure,
            required_object_ids=required_ids,
        )
        actual_pressure = selection.actual_ratio
        largest_object_ratio = (
            max(item.reusable_prefix_tokens for item in selection.objects) / capacity
        )
        tolerance = max(0.08, largest_object_ratio)
        if abs(actual_pressure - pressure) > tolerance:
            raise RuntimeError(
                f"unable to calibrate rho={pressure}: "
                f"actual={actual_pressure:.3f}, tolerance={tolerance:.3f}"
            )
        frozen[pressure] = selection.objects

    dataset = {
        "catalog_size": len(catalog),
        "target_prefix_sizes": target_sizes,
        "workflow_sequence": [item.object_id for item in workflow_sequence],
        "reference_max_total_num_tokens": reference_max_tokens,
        "reference_capacity_tokens": capacity,
        "reference_clean_metrics": clean_metrics,
        "objects": [item.manifest() for item in catalog],
        "pressure_selections": {
            str(pressure): {
                "object_ids": [item.object_id for item in objects],
                "unique_prefix_tokens": unique_prefix_token_count(objects),
                "actual_pressure": unique_prefix_token_count(objects) / capacity,
            }
            for pressure, objects in frozen.items()
        },
    }
    (args.output_dir / "dataset.json").write_text(
        json.dumps(dataset, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    configs = [
        (policy, prefetch_mode, pressure, restart)
        for restart in range(args.restarts)
        for policy in policies
        for prefetch_mode in prefetch_modes
        for pressure in pressure_points
    ]
    random.Random(args.order_seed).shuffle(configs)
    run_id = "phase5-scheduler-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records = []
    for order_index, (policy, prefetch_mode, pressure, restart) in enumerate(configs):
        records.append(
            run_config(
                args,
                run_id=run_id,
                order_index=order_index,
                restart=restart,
                policy=policy,
                prefetch_mode=prefetch_mode,
                target_pressure=pressure,
                capacity=capacity,
                selected_objects=frozen[pressure],
                workflow_sequence=workflow_sequence,
            )
        )

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "source_git_sha": args.source_git_sha,
        "image_digest": args.image_digest,
        "model": args.model,
        "model_revision": args.model_revision,
        "machine": machine_manifest(),
        "settings": {
            **vars(args),
            "output_dir": str(args.output_dir),
            "central_log": str(args.central_log),
            "policies": policies,
            "prefetch_modes": prefetch_modes,
            "pressure_points": pressure_points,
            "target_prefix_sizes": target_sizes,
        },
        "config_order": [
            {
                "policy": policy,
                "prefetch_mode": prefetch_mode,
                "pressure": pressure,
                "restart": restart,
            }
            for policy, prefetch_mode, pressure, restart in configs
        ],
        "runs": [
            {
                "settings": record["settings"],
                "actual_reusable_pressure": record["actual_reusable_pressure"],
                "workflow_summary": record["workflow_summary"],
                "telemetry_delta": record["telemetry_delta"],
            }
            for record in records
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["runs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
