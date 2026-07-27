#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from benchmark.approx_kv.metrics import clean_cache_invariant, max_total_num_tokens
from benchmark.approx_kv.phase6.runner import (
    append_jsonl,
    execution_status,
    flush_cache,
    generate,
    launch_server,
    machine_manifest,
    metric_snapshot,
    metric_text,
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

_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+(?P<value>[-+0-9.eE]+)$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"])*)"')


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
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--server-start-timeout-s", type=float, default=600)
    parser.add_argument("--kv-bytes-per-token", type=int, default=114688)
    return parser.parse_args()


def labeled_metric_sum(
    text: str,
    name: str,
    required_labels: dict[str, str],
) -> float:
    total = 0.0
    for line in text.splitlines():
        match = _SAMPLE_RE.match(line.strip())
        if match is None or match.group("name") != name:
            continue
        labels = {
            key: bytes(value, "utf-8").decode("unicode_escape")
            for key, value in _LABEL_RE.findall(match.group("labels") or "")
        }
        if all(labels.get(key) == value for key, value in required_labels.items()):
            total += float(match.group("value"))
    return total


def labeled_metric_delta(
    before: str,
    after: str,
    name: str,
    required_labels: dict[str, str],
) -> float:
    return labeled_metric_sum(after, name, required_labels) - labeled_metric_sum(
        before,
        name,
        required_labels,
    )


def metadata(
    *,
    operation: str,
    header_tokens: int,
    body_tokens: int,
    segment_tokens: int,
) -> dict[str, Any]:
    segments = []
    for segment_index, offset in enumerate(range(0, body_tokens, segment_tokens)):
        segments.append(
            {
                "content_hash": f"p6-f-body:seg{segment_index}",
                "target_start": header_tokens + offset,
                "length": min(segment_tokens, body_tokens - offset),
                "object_id": f"p6-f-object:seg{segment_index}",
                "object_kind": "canonical_base",
                "dense_cost_ms": 10.0,
                "recovery_cost_ms": 1.0,
            }
        )
    return {
        "operation": operation,
        "model_fingerprint": "p6-f-qwen3-sm75",
        "cache_dtype": "float16",
        "segments": segments,
    }


def seed_header(
    *,
    port: int,
    header: list[int],
    namespace: str,
) -> dict[str, Any]:
    return generate(
        port=port,
        input_ids=header,
        max_new_tokens=1,
        extra_key=namespace,
    )


def run_independent_control(
    args: argparse.Namespace,
    *,
    header: list[int],
    prompt: list[int],
    register_metadata: dict[str, Any],
    reuse_metadata: dict[str, Any],
    dense_output_ids: list[int],
) -> dict[str, Any]:
    control_log = args.log.with_name(f"{args.log.stem}-control{args.log.suffix}")
    plugin_env = {
        "SGLANG_APPROX_KV_CORE": "1",
        "SGLANG_APPROX_KV_CROSS_STORE": "1",
        "SGLANG_APPROX_KV_BYTES_PER_TOKEN": str(args.kv_bytes_per_token),
        "SGLANG_APPROX_KV_HOST_BUDGET_BYTES": "0",
        "SGLANG_APPROX_KV_REGISTER_EVICTS_EXACT": "0",
        "SGLANG_APPROX_KV_TEST_ONLY": "1",
        # Deliberately omit SGLANG_APPROX_KV_TEST_RESERVATION_FAILURE.
    }
    server = launch_server(
        model=args.model,
        model_revision=args.model_revision,
        port=args.port,
        mem_fraction_static=args.mem_fraction_static,
        chunked_prefill_size=args.chunked_prefill_size,
        policy="hierarchical",
        log_path=control_log,
        plugin_env=plugin_env,
        server_seed=29,
    )
    try:
        wait_ready(
            server,
            port=args.port,
            timeout_s=args.server_start_timeout_s,
        )
        flush_cache(args.port)
        source_namespace = "p6-f-control-source"
        source = generate(
            port=args.port,
            input_ids=prompt,
            max_new_tokens=1,
            extra_key=source_namespace,
        )
        registered = generate(
            port=args.port,
            input_ids=prompt,
            max_new_tokens=1,
            custom_params={"approx_kv": register_metadata},
            extra_key=source_namespace,
        )
        recovery_namespace = "p6-f-control-recovery"
        seed = seed_header(
            port=args.port,
            header=header,
            namespace=recovery_namespace,
        )
        before = metric_text(args.port)
        recovery = generate(
            port=args.port,
            input_ids=prompt,
            max_new_tokens=args.max_new_tokens,
            custom_params={"approx_kv": reuse_metadata},
            extra_key=recovery_namespace,
        )
        after = metric_text(args.port)
        success = labeled_metric_delta(
            before,
            after,
            "sglang:approx_kv_requests_total",
            {"operation": "reuse", "outcome": "success"},
        )
        reservation_failures = labeled_metric_delta(
            before,
            after,
            "sglang:cross_store_reservation_failures_total",
            {"requires_reset": "false"},
        )
        if success != 1 or reservation_failures != 0:
            raise RuntimeError(
                "the independent injection-disabled control did not recover "
                f"normally: success={success}, failures={reservation_failures}"
            )
        if recovery["cached_tokens"] < args.header_tokens + args.body_tokens:
            raise RuntimeError("the independent control did not attach the body")
        if recovery["output_ids"] != dense_output_ids:
            raise RuntimeError("the independent control output diverged from dense")

        pre_flush = metric_snapshot(args.port)
        accounting_before_flush = {
            name: pre_flush.get(name)
            for name in (
                "sglang:cross_store_reserved_device_bytes",
                "sglang:approx_kv_provisional_tokens",
                "sglang:approx_kv_store_leases",
                "sglang:approx_kv_store_orphans",
            )
        }
        if any(value not in (0, 0.0) for value in accounting_before_flush.values()):
            raise RuntimeError(
                "independent control pre-flush accounting is not clean: "
                f"{accounting_before_flush}"
            )

        flush_cache(args.port)
        reset_metrics = metric_snapshot(args.port)
        reset = clean_cache_invariant(reset_metrics)
        store_reset = {
            name: reset_metrics.get(name)
            for name in (
                "sglang:approx_kv_store_records",
                "sglang:approx_kv_store_device_bytes",
                "sglang:approx_kv_store_host_bytes",
                "sglang:approx_kv_store_leases",
                "sglang:approx_kv_store_orphans",
                "sglang:approx_kv_provisional_tokens",
                "sglang:cross_store_reserved_device_bytes",
            )
        }
        if not reset["passed"] or any(
            value not in (0, 0.0) for value in store_reset.values()
        ):
            raise RuntimeError(
                f"independent control reset failed: {reset}, {store_reset}"
            )
        return {
            "server_argv": list(server.command),
            "plugin_env": server.plugin_env,
            "server_log": str(control_log),
            "server_log_sha256": file_sha256(control_log),
            "source": source,
            "registered": registered,
            "seed": seed,
            "recovery": recovery,
            "recovery_success": success,
            "reservation_failures": reservation_failures,
            "output_matches_dense": recovery["output_ids"] == dense_output_ids,
            "accounting_before_flush": accounting_before_flush,
            "reset_invariant": reset,
            "store_reset_gauges": store_reset,
        }
    finally:
        stop_server(server)


def execute(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    if args.header_tokens <= 0 or args.body_tokens <= 0:
        raise ValueError("prompt lengths must be positive")
    if args.segment_tokens <= 0 or args.segment_tokens > 512:
        raise ValueError("segment-tokens must be in [1, 512]")
    if args.max_new_tokens < 8:
        raise ValueError("max-new-tokens must be at least 8")

    provenance = source_provenance(args.source_git_sha)
    plugin_env = {
        "SGLANG_APPROX_KV_CORE": "1",
        "SGLANG_APPROX_KV_CROSS_STORE": "1",
        "SGLANG_APPROX_KV_BYTES_PER_TOKEN": str(args.kv_bytes_per_token),
        "SGLANG_APPROX_KV_HOST_BUDGET_BYTES": "0",
        "SGLANG_APPROX_KV_REGISTER_EVICTS_EXACT": "0",
        "SGLANG_APPROX_KV_TEST_ONLY": "1",
        "SGLANG_APPROX_KV_TEST_RESERVATION_FAILURE": "1",
    }
    server = launch_server(
        model=args.model,
        model_revision=args.model_revision,
        port=args.port,
        mem_fraction_static=args.mem_fraction_static,
        chunked_prefill_size=args.chunked_prefill_size,
        policy="hierarchical",
        log_path=args.log,
        plugin_env=plugin_env,
    )
    try:
        wait_ready(
            server,
            port=args.port,
            timeout_s=args.server_start_timeout_s,
        )
        flush_cache(args.port)
        ready_metrics = metric_snapshot(args.port)
        observed_capacity = max_total_num_tokens(ready_metrics)

        header = [12_000 + offset * 13 for offset in range(args.header_tokens)]
        body = [24_000 + offset * 17 for offset in range(args.body_tokens)]
        prompt = header + body + [48_001]
        register_metadata = metadata(
            operation="register",
            header_tokens=args.header_tokens,
            body_tokens=args.body_tokens,
            segment_tokens=args.segment_tokens,
        )
        reuse_metadata = metadata(
            operation="reuse",
            header_tokens=args.header_tokens,
            body_tokens=args.body_tokens,
            segment_tokens=args.segment_tokens,
        )

        source_namespace = "p6-f-source"
        source = generate(
            port=args.port,
            input_ids=prompt,
            max_new_tokens=1,
            extra_key=source_namespace,
        )
        registered = generate(
            port=args.port,
            input_ids=prompt,
            max_new_tokens=1,
            custom_params={"approx_kv": register_metadata},
            extra_key=source_namespace,
        )

        dense_namespace = "p6-f-dense"
        dense_seed = seed_header(
            port=args.port,
            header=header,
            namespace=dense_namespace,
        )
        dense = generate(
            port=args.port,
            input_ids=prompt,
            max_new_tokens=args.max_new_tokens,
            extra_key=dense_namespace,
        )

        fallback_namespace = "p6-f-fallback"
        fallback_seed = seed_header(
            port=args.port,
            header=header,
            namespace=fallback_namespace,
        )
        before_fallback = metric_text(args.port)
        fallback = generate(
            port=args.port,
            input_ids=prompt,
            max_new_tokens=args.max_new_tokens,
            custom_params={"approx_kv": reuse_metadata},
            extra_key=fallback_namespace,
        )
        after_fallback = metric_text(args.port)

        reservation_failures = labeled_metric_delta(
            before_fallback,
            after_fallback,
            "sglang:cross_store_reservation_failures_total",
            {"requires_reset": "false"},
        )
        reservation_fallback_tokens = labeled_metric_delta(
            before_fallback,
            after_fallback,
            "sglang:approx_kv_dense_fallback_total",
            {"reason": "cross_store_reservation_failed"},
        )
        device_allocation_fallback_tokens = labeled_metric_delta(
            before_fallback,
            after_fallback,
            "sglang:approx_kv_dense_fallback_total",
            {"reason": "device_allocation_failed"},
        )
        dense_fallback_requests = labeled_metric_delta(
            before_fallback,
            after_fallback,
            "sglang:approx_kv_requests_total",
            {"operation": "reuse", "outcome": "dense_fallback"},
        )

        if reservation_failures != 1:
            raise RuntimeError(
                "the test-only reservation failure was not observed exactly once: "
                f"{reservation_failures}"
            )
        if reservation_fallback_tokens < args.body_tokens:
            raise RuntimeError(
                "reservation-failure fallback did not cover the body: "
                f"{reservation_fallback_tokens}"
            )
        if device_allocation_fallback_tokens != 0:
            raise RuntimeError(
                "the reservation fallback was double-attributed as a generic "
                "device allocation failure: "
                f"{device_allocation_fallback_tokens}"
            )
        if dense_fallback_requests != 1:
            raise RuntimeError(
                "the integrated reuse request was not labelled dense_fallback: "
                f"{dense_fallback_requests}"
            )
        if fallback["cached_tokens"] >= args.header_tokens + args.body_tokens:
            raise RuntimeError("the injected request unexpectedly used recovered KV")
        if fallback["output_ids"] != dense["output_ids"]:
            raise RuntimeError("the dense fallback output diverged from matched dense")

        normal_namespace = "p6-f-normal-recovery"
        normal_seed = seed_header(
            port=args.port,
            header=header,
            namespace=normal_namespace,
        )
        before_normal = metric_text(args.port)
        normal_recovery = generate(
            port=args.port,
            input_ids=prompt,
            max_new_tokens=args.max_new_tokens,
            custom_params={"approx_kv": reuse_metadata},
            extra_key=normal_namespace,
        )
        after_normal = metric_text(args.port)
        normal_success = labeled_metric_delta(
            before_normal,
            after_normal,
            "sglang:approx_kv_requests_total",
            {"operation": "reuse", "outcome": "success"},
        )
        if normal_success != 1:
            raise RuntimeError(
                "the one-shot injection was not consumed: "
                f"normal recovery success delta={normal_success}"
            )
        if normal_recovery["cached_tokens"] < args.header_tokens + args.body_tokens:
            raise RuntimeError("normal recovery did not attach the registered body")
        if normal_recovery["output_ids"] != dense["output_ids"]:
            raise RuntimeError("normal same-context recovery diverged from dense")

        pre_flush_metrics = metric_snapshot(args.port)
        accounting_before_flush = {
            name: pre_flush_metrics.get(name)
            for name in (
                "sglang:cross_store_reserved_device_bytes",
                "sglang:approx_kv_provisional_tokens",
                "sglang:approx_kv_store_leases",
                "sglang:approx_kv_store_orphans",
            )
        }
        if any(value not in (0, 0.0) for value in accounting_before_flush.values()):
            raise RuntimeError(
                "pre-flush accounting is not clean: " f"{accounting_before_flush}"
            )
        observed_peak_device_bytes = pre_flush_metrics.get(
            "sglang:cross_store_peak_device_bytes"
        )

        flush_cache(args.port)
        reset_metrics = metric_snapshot(args.port)
        reset = clean_cache_invariant(reset_metrics)
        store_reset = {
            name: reset_metrics.get(name)
            for name in (
                "sglang:approx_kv_store_records",
                "sglang:approx_kv_store_device_bytes",
                "sglang:approx_kv_store_host_bytes",
                "sglang:approx_kv_store_leases",
                "sglang:approx_kv_store_orphans",
                "sglang:approx_kv_provisional_tokens",
                "sglang:cross_store_reserved_device_bytes",
            )
        }
        if not reset["passed"] or any(
            value not in (0, 0.0) for value in store_reset.values()
        ):
            raise RuntimeError(
                f"post-canary reset invariant failed: {reset}, {store_reset}"
            )

        injection_server_argv = list(server.command)
        injection_plugin_env = server.plugin_env
        stop_server(server)
        server = None
        independent_control = run_independent_control(
            args,
            header=header,
            prompt=prompt,
            register_metadata=register_metadata,
            reuse_metadata=reuse_metadata,
            dense_output_ids=dense["output_ids"],
        )

        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "phase": "P6-F",
            "source_git_sha": provenance["source_git_sha"],
            "source_tree_sha": provenance["source_tree_sha"],
            "result_git_sha": None,
            "result_commit_status": "pending_result_commit",
            "model": args.model,
            "model_revision": args.model_revision,
            "image_digest": args.image_digest,
            "machine": machine_manifest(),
            "server_argv": [
                injection_server_argv,
                independent_control["server_argv"],
            ],
            "plugin_env": {
                "injection_server": injection_plugin_env,
                "control_server": independent_control["plugin_env"],
            },
            "requested_capacity": {
                "tokens": None,
                "pages": None,
                "bytes": None,
            },
            "observed_capacity": {
                "tokens": observed_capacity,
                "pages": observed_capacity,
                "bytes": observed_capacity * args.kv_bytes_per_token,
            },
            "crosses_chunk_boundary": (
                args.header_tokens + args.body_tokens + 1 > args.chunked_prefill_size
            ),
            "segment_count": len(register_metadata["segments"]),
            "warmup_repeats": 0,
            "formal_repeats": 1,
            "restarts": 2,
            "fault_injected": True,
            "natural_pressure_reachability": False,
            "injection_point": "after_reserve",
            "injection_scope": "one-shot approximate requester only",
            "source": source,
            "registered": registered,
            "dense_seed": dense_seed,
            "dense": dense,
            "fallback_seed": fallback_seed,
            "fallback": fallback,
            "normal_seed": normal_seed,
            "normal_recovery": normal_recovery,
            "fallback_evidence": {
                "reservation_failures": reservation_failures,
                "reservation_fallback_tokens": reservation_fallback_tokens,
                "device_allocation_fallback_tokens": (
                    device_allocation_fallback_tokens
                ),
                "dense_fallback_requests": dense_fallback_requests,
                "request_completed": len(fallback["output_ids"]) == args.max_new_tokens,
                "output_matches_dense": fallback["output_ids"] == dense["output_ids"],
            },
            "one_shot_control": {
                "normal_recovery_success": normal_success,
                "normal_recovery_cached_tokens": normal_recovery["cached_tokens"],
                "output_matches_dense": normal_recovery["output_ids"]
                == dense["output_ids"],
            },
            "independent_injection_disabled_control": independent_control,
            "accounting_before_flush": accounting_before_flush,
            "reset_invariant": reset,
            "store_reset_gauges": store_reset,
            "ledger": {
                "setup": {
                    "source_ms": source["elapsed_ms"],
                    "registration_ms": registered["elapsed_ms"],
                },
                "materialization": {
                    "registered_tokens": args.body_tokens,
                },
                "recovery": {
                    "fault_injected": True,
                    "dense_fallback_tokens": reservation_fallback_tokens,
                },
                "scheduler": {
                    "dense_fallback_requests": dense_fallback_requests,
                },
                "transfer": {
                    "declared_bytes": 0,
                    "measurement": "not_applicable_no_transfer",
                },
                "temporary_peak": {
                    "observed_bytes": observed_peak_device_bytes,
                },
            },
            "rho": {
                **RhoDefinitions().__dict__,
                "observed_logical_demand": (args.body_tokens / observed_capacity),
            },
            "status": "valid",
            "performance_claim": "disabled",
            "evidence_claim": (
                "fault-injected integrated fallback path only; "
                "does not prove natural pressure reachability"
            ),
            "passed": True,
        }
        payload["raw_sha256"] = payload_sha256(payload)
        validate_phase6_artifact(payload)
        return payload
    finally:
        if server is not None:
            stop_server(server)


def main() -> int:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("p6-f-%Y%m%dT%H%M%SZ")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    append_jsonl(
        args.central_log,
        {
            "run_id": run_id,
            "phase": "P6-F",
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
                "phase": "P6-F",
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
            "phase": "P6-F",
            "source_git_sha": args.source_git_sha,
            "image_digest": args.image_digest,
            "status": "invalid",
            "execution_status": status,
            "fault_injected": True,
            "natural_pressure_reachability": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        failure["raw_sha256"] = payload_sha256(failure)
        write_json(args.output, failure)
        append_jsonl(
            args.central_log,
            {
                "run_id": run_id,
                "phase": "P6-F",
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
