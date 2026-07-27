#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import requests

from benchmark.approx_kv.metrics import (
    clean_cache_invariant,
    counter_delta,
    max_total_num_tokens,
)
from benchmark.approx_kv.phase6.runner import (
    append_jsonl,
    execution_status,
    flush_cache,
    generate,
    launch_server,
    machine_manifest,
    metric_snapshot,
    source_provenance,
    stop_server,
    wait_ready,
    write_json,
)
from benchmark.approx_kv.phase6.schema import (
    RhoDefinitions,
    file_sha256,
    payload_sha256,
    validate_phase6_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--central-log", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30011)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument("--chunked-prefill-size", type=int, default=4096)
    parser.add_argument("--header-tokens", type=int, default=64)
    parser.add_argument("--body-tokens", type=int, default=1024)
    parser.add_argument("--segment-tokens", type=int, default=512)
    parser.add_argument("--formal-repeats", type=int, default=2)
    parser.add_argument("--server-start-timeout-s", type=float, default=600)
    parser.add_argument("--kv-bytes-per-token", type=int, default=114688)
    parser.add_argument("--host-budget-bytes", type=int, default=8 << 30)
    parser.add_argument("--max-total-tokens", type=int, default=0)
    return parser.parse_args()


def metadata(
    *,
    operation: str,
    content_hash: str,
    object_id: str,
    header_tokens: int,
    body_tokens: int,
    object_kind: str,
    residency: str | None = None,
    segment_tokens: int = 512,
) -> dict:
    segments = []
    for segment_index, offset in enumerate(range(0, body_tokens, segment_tokens)):
        segment = {
            "content_hash": f"{content_hash}:seg{segment_index}",
            "target_start": header_tokens + offset,
            "length": min(segment_tokens, body_tokens - offset),
            "object_id": f"{object_id}:seg{segment_index}",
            "object_kind": object_kind,
            "dense_cost_ms": 12.0,
            "recovery_cost_ms": 2.0,
        }
        if residency is not None:
            segment["residency"] = residency
        segments.append(segment)
    return {
        "operation": operation,
        "model_fingerprint": "p6-host-roundtrip",
        "cache_dtype": "float16",
        "segments": segments,
    }


def run_round(args: argparse.Namespace, round_index: int) -> dict:
    flush_cache(args.port)
    before = metric_snapshot(args.port)
    header = [
        2_000 + ((round_index * 97 + offset * 13) % 2_000)
        for offset in range(args.header_tokens)
    ]
    body = [
        8_000 + ((round_index * 193 + offset * 17) % 20_000)
        for offset in range(args.body_tokens)
    ]
    prompt = header + body + [31_000 + round_index]
    content_hash = f"p6-h-round-{round_index}"
    object_id = f"p6-h-object-{round_index}"
    register_metadata = metadata(
        operation="register",
        content_hash=content_hash,
        object_id=object_id,
        header_tokens=args.header_tokens,
        body_tokens=args.body_tokens,
        object_kind="materialization_scratch",
        residency="device",
        segment_tokens=args.segment_tokens,
    )
    reuse_metadata = metadata(
        operation="reuse",
        content_hash=content_hash,
        object_id=object_id,
        header_tokens=args.header_tokens,
        body_tokens=args.body_tokens,
        object_kind="materialization_scratch",
        segment_tokens=args.segment_tokens,
    )
    pressure_prompt = (
        [token + 211 for token in header]
        + [token + 307 for token in body]
        + [32_000 + round_index]
    )
    pressure_hash = f"p6-h-pressure-{round_index}"
    pressure_object_id = f"p6-h-pressure-object-{round_index}"
    pressure_metadata = metadata(
        operation="register",
        content_hash=pressure_hash,
        object_id=pressure_object_id,
        header_tokens=args.header_tokens,
        body_tokens=args.body_tokens,
        object_kind="canonical_base",
        residency="device",
        segment_tokens=args.segment_tokens,
    )
    cache_protection = {
        "object_id": pressure_object_id,
        "protected_tokens": args.header_tokens + args.body_tokens,
        "resident_bytes": (
            (args.header_tokens + args.body_tokens) * args.kv_bytes_per_token
        ),
        "object_kind": "canonical_base",
        "retired": False,
    }

    source = generate(
        port=args.port,
        input_ids=prompt,
        max_new_tokens=1,
        extra_key=f"p6-h-source-{round_index}",
    )
    registered = generate(
        port=args.port,
        input_ids=prompt,
        max_new_tokens=1,
        custom_params={"approx_kv": register_metadata},
        extra_key=f"p6-h-source-{round_index}",
    )
    pressure_source = generate(
        port=args.port,
        input_ids=pressure_prompt,
        max_new_tokens=1,
        custom_params={"cache_protection": cache_protection},
        extra_key=f"p6-h-pressure-{round_index}",
    )
    pressure_registered = generate(
        port=args.port,
        input_ids=pressure_prompt,
        max_new_tokens=1,
        custom_params={
            "cache_protection": cache_protection,
            "approx_kv": pressure_metadata,
        },
        extra_key=f"p6-h-pressure-{round_index}",
    )
    dense_namespace = f"p6-h-dense-{round_index}"
    recovery_namespace = f"p6-h-recovery-{round_index}"
    dense_seed = generate(
        port=args.port,
        input_ids=header,
        max_new_tokens=1,
        extra_key=dense_namespace,
    )
    recovery_seed = generate(
        port=args.port,
        input_ids=header,
        max_new_tokens=1,
        extra_key=recovery_namespace,
    )
    dense = generate(
        port=args.port,
        input_ids=prompt,
        max_new_tokens=8,
        extra_key=dense_namespace,
    )
    recovery_reseed = generate(
        port=args.port,
        input_ids=header,
        max_new_tokens=1,
        extra_key=recovery_namespace,
    )
    if recovery_reseed["cached_tokens"] > args.header_tokens:
        raise RuntimeError(
            "recovery header reseed produced an unexpected cached prefix: "
            f"{recovery_reseed['cached_tokens']}"
        )
    recovered = generate(
        port=args.port,
        input_ids=prompt,
        max_new_tokens=8,
        custom_params={"approx_kv": reuse_metadata},
        extra_key=recovery_namespace,
    )
    if recovered["cached_tokens"] < args.header_tokens + args.body_tokens:
        raise RuntimeError(
            "approximate reuse did not attach the registered body: "
            f"cached_tokens={recovered['cached_tokens']}, expected at least "
            f"{args.header_tokens + args.body_tokens}"
        )
    after = metric_snapshot(args.port)

    metric_names = (
        "sglang:approx_kv_host_export_tokens_total",
        "sglang:approx_kv_host_export_bytes_total",
        "sglang:approx_kv_h2d_tokens_total",
        "sglang:approx_kv_h2d_bytes_total",
        "sglang:approx_kv_host_export_duration_seconds_sum",
        "sglang:approx_kv_h2d_duration_seconds_sum",
        "sglang:approx_kv_copied_tokens_total",
        "sglang:approx_kv_dense_fallback_total",
        "sglang:cross_store_demoted_bytes_total",
        "sglang:cross_store_reservation_failures_total",
    )
    deltas = {name: counter_delta(before, after, name) for name in metric_names}
    required_token_metrics = (
        "sglang:approx_kv_host_export_tokens_total",
        "sglang:approx_kv_h2d_tokens_total",
        "sglang:approx_kv_copied_tokens_total",
    )
    for name in required_token_metrics:
        value = deltas[name]
        if value is None or value < args.body_tokens:
            raise RuntimeError(f"{name} did not cover the body: {deltas}")
    demoted_bytes = deltas["sglang:cross_store_demoted_bytes_total"]
    if (
        demoted_bytes is None
        or demoted_bytes < args.body_tokens * args.kv_bytes_per_token
    ):
        raise RuntimeError(f"device-to-host demotion was not observed: {deltas}")
    if (deltas["sglang:cross_store_reservation_failures_total"] or 0) != 0:
        raise RuntimeError(f"cross-store reservation failed: {deltas}")
    if dense["output_ids"] != recovered["output_ids"]:
        raise RuntimeError("host roundtrip output diverged from matched dense")

    flush_cache(args.port)
    post_reset = metric_snapshot(args.port)
    reset = clean_cache_invariant(post_reset)
    store_reset = {
        name: post_reset.get(name)
        for name in (
            "sglang:approx_kv_store_records",
            "sglang:approx_kv_store_device_bytes",
            "sglang:approx_kv_store_host_bytes",
            "sglang:approx_kv_store_leases",
            "sglang:approx_kv_store_orphans",
        )
    }
    if not reset["passed"] or any(
        value not in (0, 0.0) for value in store_reset.values()
    ):
        raise RuntimeError(f"post-round reset invariant failed: {reset}")
    return {
        "round_index": round_index,
        "source": source,
        "registered": registered,
        "pressure_source": pressure_source,
        "pressure_registered": pressure_registered,
        "dense_seed": dense_seed,
        "recovery_seed": recovery_seed,
        "recovery_reseed": recovery_reseed,
        "dense": dense,
        "recovered": recovered,
        "metric_deltas": deltas,
        "output_token_match": True,
        "reset_invariant": reset,
        "store_reset_gauges": store_reset,
    }


def execute(args: argparse.Namespace, run_id: str) -> dict:
    if args.formal_repeats < 2:
        raise ValueError("formal-repeats must be at least 2")
    if args.header_tokens <= 0 or args.body_tokens <= 0:
        raise ValueError("prompt lengths must be positive")
    if args.segment_tokens <= 0 or args.segment_tokens > 512:
        raise ValueError("segment-tokens must be in [1, 512]")
    if args.host_budget_bytes < args.body_tokens * args.kv_bytes_per_token:
        raise ValueError("host budget cannot hold the roundtrip object")
    provenance = source_provenance(args.source_git_sha)
    observed_sha = provenance["source_git_sha"]

    plugin_env = {
        "SGLANG_APPROX_KV_CORE": "1",
        "SGLANG_APPROX_KV_HOST": "1",
        "SGLANG_APPROX_KV_CROSS_STORE": "1",
        "SGLANG_APPROX_KV_REGISTER_EVICTS_EXACT": "1",
        "SGLANG_APPROX_KV_BYTES_PER_TOKEN": str(args.kv_bytes_per_token),
        "SGLANG_APPROX_KV_HOST_BUDGET_BYTES": str(args.host_budget_bytes),
    }
    requested_capacity_tokens = (
        args.max_total_tokens
        if args.max_total_tokens > 0
        else math.ceil((2 * args.body_tokens + args.header_tokens) * 1.15)
    )
    server = launch_server(
        model=args.model,
        model_revision=args.model_revision,
        port=args.port,
        mem_fraction_static=args.mem_fraction_static,
        chunked_prefill_size=args.chunked_prefill_size,
        policy="hierarchical",
        log_path=args.log,
        plugin_env=plugin_env,
        max_total_tokens=requested_capacity_tokens,
    )
    try:
        wait_ready(
            server,
            port=args.port,
            timeout_s=args.server_start_timeout_s,
        )
        ready_metrics = metric_snapshot(args.port)
        observed_capacity_tokens = max_total_num_tokens(ready_metrics)
        run_round(args, -1)
        rounds = [
            run_round(args, round_index) for round_index in range(args.formal_repeats)
        ]
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "phase": "P6-H",
            "source_git_sha": observed_sha,
            "source_tree_sha": provenance["source_tree_sha"],
            "result_git_sha": None,
            "result_commit_status": "pending_result_commit",
            "model": args.model,
            "model_revision": args.model_revision,
            "image_digest": args.image_digest,
            "machine": machine_manifest(),
            "server_argv": list(server.command),
            "plugin_env": server.plugin_env,
            "header_tokens": args.header_tokens,
            "body_tokens": args.body_tokens,
            "chunked_prefill_size": args.chunked_prefill_size,
            "warmup_repeats": 1,
            "formal_repeats": args.formal_repeats,
            "restarts": 1,
            "rounds": rounds,
            "mean_h2d_bytes": sum(
                float(row["metric_deltas"]["sglang:approx_kv_h2d_bytes_total"] or 0)
                for row in rounds
            )
            / len(rounds),
            "requested_capacity": {
                "tokens": requested_capacity_tokens,
                "pages": requested_capacity_tokens,
                "bytes": (requested_capacity_tokens * args.kv_bytes_per_token),
            },
            "observed_capacity": {
                "tokens": observed_capacity_tokens,
                "pages": observed_capacity_tokens,
                "bytes": (observed_capacity_tokens * args.kv_bytes_per_token),
            },
            "crosses_chunk_boundary": (
                args.header_tokens + args.body_tokens + 1 > args.chunked_prefill_size
            ),
            "segment_count": math.ceil(args.body_tokens / args.segment_tokens),
            "ledger": {
                "setup": {
                    "requests_per_round": 6,
                    "seed_head_ms_per_round": [
                        {
                            "dense": row["dense_seed"]["elapsed_ms"],
                            "recovery": row["recovery_seed"]["elapsed_ms"],
                        }
                        for row in rounds
                    ],
                },
                "materialization": {
                    "tokens_per_round": 2 * args.body_tokens,
                },
                "recovery": {
                    "tokens_per_round": args.body_tokens,
                },
                "scheduler": {
                    "policy": "hierarchical",
                },
                "transfer": {
                    "mean_h2d_bytes": sum(
                        float(
                            row["metric_deltas"]["sglang:approx_kv_h2d_bytes_total"]
                            or 0
                        )
                        for row in rounds
                    )
                    / len(rounds),
                    "mean_host_export_duration_seconds": sum(
                        float(
                            row["metric_deltas"][
                                "sglang:approx_kv_host_export_duration_seconds_sum"
                            ]
                            or 0
                        )
                        for row in rounds
                    )
                    / len(rounds),
                    "mean_h2d_duration_seconds": sum(
                        float(
                            row["metric_deltas"][
                                "sglang:approx_kv_h2d_duration_seconds_sum"
                            ]
                            or 0
                        )
                        for row in rounds
                    )
                    / len(rounds),
                },
                "temporary_peak": {
                    "bytes": args.body_tokens * args.kv_bytes_per_token,
                },
            },
            "rho": {
                **RhoDefinitions().__dict__,
                "observed_logical_demand": (
                    args.body_tokens / observed_capacity_tokens
                ),
            },
            "status": "valid",
            "performance_claim": "disabled",
            "host_backend": "allocator_cpu_copy",
            "hicache_tier_exercised": False,
            "transfer_measurement": (
                "measured payload bytes and synchronous wall-clock duration"
            ),
            "passed": True,
        }
        payload["raw_sha256"] = payload_sha256(payload)
        validate_phase6_artifact(payload)
        return payload
    finally:
        stop_server(server)


def main() -> int:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("p6-h-%Y%m%dT%H%M%SZ")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    append_jsonl(
        args.central_log,
        {
            "run_id": run_id,
            "phase": "P6-H",
            "status": "running",
            "output": str(args.output.resolve()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        payload = execute(args, run_id)
        payload["server_log_sha256"] = file_sha256(args.log)
        payload.pop("raw_sha256", None)
        payload["raw_sha256"] = payload_sha256(payload)
        validate_phase6_artifact(payload)
        write_json(args.output, payload)
        append_jsonl(
            args.central_log,
            {
                "run_id": run_id,
                "phase": "P6-H",
                "status": "completed",
                "raw_sha256": payload["raw_sha256"],
                "output": str(args.output.resolve()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return 0
    except (
        KeyError,
        MemoryError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        requests.RequestException,
    ) as exc:
        status = execution_status(exc)
        failure = {
            "schema_version": 1,
            "run_id": run_id,
            "phase": "P6-H",
            "source_git_sha": args.source_git_sha,
            "image_digest": args.image_digest,
            "status": "invalid",
            "execution_status": status,
            "error": f"{type(exc).__name__}: {exc}",
        }
        failure["raw_sha256"] = payload_sha256(failure)
        write_json(args.output, failure)
        append_jsonl(
            args.central_log,
            {
                "run_id": run_id,
                "phase": "P6-H",
                "status": status,
                "error": failure["error"],
                "raw_sha256": failure["raw_sha256"],
                "output": str(args.output.resolve()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
