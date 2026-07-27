#!/usr/bin/env python3
"""Instrumented reproduction of one P6-H host roundtrip round.

Snapshots the approximate-KV and cross-store counters after every request so
the exact step at which the demand H2D load fails can be identified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from benchmark.approx_kv.metrics import max_total_num_tokens
from benchmark.approx_kv.phase6.runner import (
    flush_cache,
    generate,
    launch_server,
    metric_snapshot,
    stop_server,
    wait_ready,
)
from benchmark.approx_kv.run_p6_h_host_roundtrip import metadata

WATCH = (
    "sglang:approx_kv_host_export_tokens_total",
    "sglang:approx_kv_h2d_tokens_total",
    "sglang:approx_kv_copied_tokens_total",
    "sglang:approx_kv_dense_fallback_total",
    "sglang:approx_kv_requests_total",
    "sglang:cross_store_demoted_bytes_total",
    "sglang:cross_store_reservation_failures_total",
    "sglang:approx_kv_store_records",
    "sglang:approx_kv_store_device_bytes",
    "sglang:approx_kv_store_host_bytes",
    "sglang:approx_kv_store_leases",
    "sglang:approx_kv_store_orphans",
)


def show(port: int, label: str, request=None) -> None:
    snap = metric_snapshot(port)
    watched = {name: snap.get(name) for name in WATCH}
    extra = ""
    if request is not None:
        extra = f" cached={request['cached_tokens']} out={request['output_ids'][:3]}"
    print(f"--- {label}{extra}")
    print("    " + json.dumps(watched, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--port", type=int, default=30012)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--chunked-prefill-size", type=int, default=4096)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument("--header-tokens", type=int, default=64)
    parser.add_argument("--body-tokens", type=int, default=1024)
    parser.add_argument("--segment-tokens", type=int, default=512)
    parser.add_argument("--kv-bytes-per-token", type=int, default=114688)
    parser.add_argument("--host-budget-bytes", type=int, default=8 << 30)
    parser.add_argument("--max-total-tokens", type=int, default=0)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--register-residency", default="device")
    parser.add_argument("--skip-pressure", action="store_true")
    args = parser.parse_args()

    plugin_env = {
        "SGLANG_APPROX_KV_CORE": "1",
        "SGLANG_APPROX_KV_HOST": "1",
        "SGLANG_APPROX_KV_CROSS_STORE": "1",
        "SGLANG_APPROX_KV_REGISTER_EVICTS_EXACT": "1",
        "SGLANG_APPROX_KV_BYTES_PER_TOKEN": str(args.kv_bytes_per_token),
        "SGLANG_APPROX_KV_HOST_BUDGET_BYTES": str(args.host_budget_bytes),
    }
    capacity = (
        args.max_total_tokens
        if args.max_total_tokens > 0
        else math.ceil((2 * args.body_tokens + args.header_tokens) * 1.15)
    )
    print(f"requested capacity tokens: {capacity}")
    server = launch_server(
        model=args.model,
        model_revision=args.model_revision,
        port=args.port,
        mem_fraction_static=args.mem_fraction_static,
        chunked_prefill_size=args.chunked_prefill_size,
        policy="hierarchical",
        log_path=args.log,
        plugin_env=plugin_env,
        max_total_tokens=capacity,
        extra_args=("--log-level", args.log_level),
    )
    try:
        wait_ready(server, port=args.port, timeout_s=600)
        ready = metric_snapshot(args.port)
        print(f"observed capacity tokens: {max_total_num_tokens(ready)}")
        round_index = 0
        flush_cache(args.port)
        show(args.port, "after flush")

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
            residency=args.register_residency,
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
        pressure_metadata = metadata(
            operation="register",
            content_hash=f"p6-h-pressure-{round_index}",
            object_id=f"p6-h-pressure-object-{round_index}",
            header_tokens=args.header_tokens,
            body_tokens=args.body_tokens,
            object_kind="canonical_base",
            residency="device",
            segment_tokens=args.segment_tokens,
        )
        cache_protection = {
            "object_id": f"p6-h-pressure-object-{round_index}",
            "protected_tokens": args.header_tokens + args.body_tokens,
            "resident_bytes": (
                (args.header_tokens + args.body_tokens) * args.kv_bytes_per_token
            ),
            "object_kind": "canonical_base",
            "retired": False,
        }

        result = generate(
            port=args.port,
            input_ids=prompt,
            max_new_tokens=1,
            extra_key=f"p6-h-source-{round_index}",
        )
        show(args.port, "1 source materialize", result)

        result = generate(
            port=args.port,
            input_ids=prompt,
            max_new_tokens=1,
            custom_params={"approx_kv": register_metadata},
            extra_key=f"p6-h-source-{round_index}",
        )
        show(args.port, "2 register device object", result)

        if not args.skip_pressure:
            result = generate(
                port=args.port,
                input_ids=pressure_prompt,
                max_new_tokens=1,
                custom_params={"cache_protection": cache_protection},
                extra_key=f"p6-h-pressure-{round_index}",
            )
            show(args.port, "3 pressure materialize", result)

            result = generate(
                port=args.port,
                input_ids=pressure_prompt,
                max_new_tokens=1,
                custom_params={
                    "cache_protection": cache_protection,
                    "approx_kv": pressure_metadata,
                },
                extra_key=f"p6-h-pressure-{round_index}",
            )
            show(args.port, "4 pressure register (expect demotion)", result)

        result = generate(
            port=args.port,
            input_ids=header,
            max_new_tokens=1,
            extra_key=f"p6-h-dense-{round_index}",
        )
        show(args.port, "5 dense seed", result)

        result = generate(
            port=args.port,
            input_ids=header,
            max_new_tokens=1,
            extra_key=f"p6-h-recovery-{round_index}",
        )
        show(args.port, "6 recovery seed", result)

        dense = generate(
            port=args.port,
            input_ids=prompt,
            max_new_tokens=8,
            extra_key=f"p6-h-dense-{round_index}",
        )
        show(args.port, "7 dense 8 tokens", dense)

        recovered = generate(
            port=args.port,
            input_ids=prompt,
            max_new_tokens=8,
            custom_params={"approx_kv": reuse_metadata},
            extra_key=f"p6-h-recovery-{round_index}",
        )
        show(args.port, "8 recover (expect H2D)", recovered)
        print(f"dense out   : {dense['output_ids']}")
        print(f"recover out : {recovered['output_ids']}")
    finally:
        stop_server(server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
