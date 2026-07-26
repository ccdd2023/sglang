#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    stream_generate,
    wait_ready,
    write_json,
)
from benchmark.approx_kv.phase6.schema import file_sha256, payload_sha256

VALID_CANDIDATES = ("r0", "r1_k0", "r1_k4", "r1_k8", "r1_k16", "r1_k32")


def csv_values(value: str, cast) -> tuple:
    values = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated values")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--central-log", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--candidates", default="r0,r1_k0,r1_k4,r1_k8,r1_k16,r1_k32")
    parser.add_argument("--body-tokens", default="1024,2048")
    parser.add_argument("--header-tokens", type=int, default=64)
    parser.add_argument("--segment-tokens", type=int, default=512)
    parser.add_argument("--filler-tokens", type=int, default=512)
    parser.add_argument("--target-rho", type=float, default=2.0)
    parser.add_argument("--formal-repeats", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--port", type=int, default=30011)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument("--chunked-prefill-size", type=int, default=1024)
    parser.add_argument("--server-start-timeout-s", type=float, default=600)
    parser.add_argument("--kv-bytes-per-token", type=int, default=114688)
    return parser.parse_args()


def candidate_k(candidate: str) -> int | None:
    if candidate == "r0":
        return None
    return int(candidate.removeprefix("r1_k"))


def chunks(body: list[int], segment_tokens: int) -> list[list[int]]:
    if segment_tokens <= 0:
        raise ValueError("segment_tokens must be positive")
    return [
        body[start : start + segment_tokens]
        for start in range(0, len(body), segment_tokens)
    ]


def source_registration(
    *,
    port: int,
    candidate: str,
    body: list[int],
    source_header: list[int],
    setting_id: str,
    segment_tokens: int,
    register: bool,
) -> dict[str, Any]:
    body_chunks = chunks(body, segment_tokens)
    rows = []
    namespace = f"cl1-source:{setting_id}"
    cursor = 0
    total_ms = 0.0
    for chunk_index, chunk in enumerate(body_chunks):
        end = cursor + len(chunk)
        prompt = source_header + body[:end] + [900 + chunk_index]
        materialized = generate(
            port=port,
            input_ids=prompt,
            max_new_tokens=1,
            extra_key=namespace,
        )
        expected_materialized_cache = (
            0 if chunk_index == 0 else len(source_header) + cursor
        )
        if materialized["cached_tokens"] != expected_materialized_cache:
            raise RuntimeError(
                "cumulative source materialization mismatch: "
                f"expected {expected_materialized_cache}, observed "
                f"{materialized['cached_tokens']}"
            )
        row = {
            "chunk_index": chunk_index,
            "source_prompt_tokens": len(prompt),
            "source_start": len(source_header) + cursor,
            "length": len(chunk),
            "materialize_cached_tokens": materialized["cached_tokens"],
            "materialize_ms": materialized["elapsed_ms"],
        }
        total_ms += float(materialized["elapsed_ms"])
        if register:
            segment = {
                "content_hash": f"cl1:{setting_id}:chunk{chunk_index}",
                "target_start": len(source_header) + cursor,
                "length": len(chunk),
                "object_id": f"cl1:{setting_id}:chunk{chunk_index}",
                "object_kind": "canonical_base",
                "dense_cost_ms": 12.0,
                "recovery_cost_ms": 2.0,
            }
            registered = generate(
                port=port,
                input_ids=prompt,
                max_new_tokens=1,
                custom_params={
                    "approx_kv": {
                        "operation": "register",
                        "model_fingerprint": "cl1-qwen3-sm75",
                        "cache_dtype": "float16",
                        "segments": [segment],
                    }
                },
                extra_key=namespace,
            )
            expected_registered_cache = len(source_header) + end
            if registered["cached_tokens"] != expected_registered_cache:
                raise RuntimeError(
                    "source registration cache mismatch: "
                    f"expected {expected_registered_cache}, observed "
                    f"{registered['cached_tokens']}"
                )
            row["register_cached_tokens"] = registered["cached_tokens"]
            row["register_ms"] = registered["elapsed_ms"]
            total_ms += float(registered["elapsed_ms"])
        rows.append(row)
        cursor = end
    return {
        "causal_prefix_registration": True,
        "isolated_extra_key": namespace,
        "rows": rows,
        "total_ms": total_ms,
    }


def target_segments(
    *,
    body_tokens: int,
    header_tokens: int,
    segment_tokens: int,
    setting_id: str,
) -> list[dict[str, Any]]:
    result = []
    cursor = 0
    for chunk_index in range(math.ceil(body_tokens / segment_tokens)):
        length = min(segment_tokens, body_tokens - cursor)
        result.append(
            {
                "content_hash": f"cl1:{setting_id}:chunk{chunk_index}",
                "target_start": header_tokens + cursor,
                "length": length,
                "object_id": f"cl1:{setting_id}:chunk{chunk_index}",
                "object_kind": "canonical_base",
            }
        )
        cursor += length
    return result


def filler_prompt(index: int, length: int, salt: int) -> list[int]:
    return [
        10_000 + ((salt * 193 + index * 977 + offset * 37) % 30_000)
        for offset in range(length)
    ] + [41_000 + ((salt + index) % 1_000)]


def resident_tokens(snapshot: dict[str, float]) -> int:
    available = snapshot.get("sglang:kv_available_tokens")
    maximum = snapshot.get("sglang:max_total_num_tokens")
    if available is not None and maximum is not None:
        return max(0, int(round(maximum - available)))
    return int(
        round(
            snapshot.get("sglang:kv_used_tokens", 0)
            + snapshot.get("sglang:kv_evictable_tokens", 0)
        )
    )


def approx_metadata(
    *,
    candidate: str,
    body_tokens: int,
    header_tokens: int,
    segment_tokens: int,
    setting_id: str,
) -> dict[str, Any]:
    metadata = {
        "operation": "reuse",
        "model_fingerprint": "cl1-qwen3-sm75",
        "cache_dtype": "float16",
        "segments": target_segments(
            body_tokens=body_tokens,
            header_tokens=header_tokens,
            segment_tokens=segment_tokens,
            setting_id=setting_id,
        ),
    }
    if candidate != "r0":
        metadata["plugin"] = "epic"
    return metadata


def run_arm(
    args: argparse.Namespace,
    *,
    candidate: str,
    body_tokens: int,
    restart_index: int,
    repeat_index: int,
    arm: str,
) -> dict[str, Any]:
    flush_cache(args.port)
    baseline = metric_snapshot(args.port)
    capacity = max_total_num_tokens(baseline)
    body = [
        1_000 + ((restart_index * 101 + repeat_index * 53 + offset) % 30_000)
        for offset in range(body_tokens)
    ]
    source_header = [
        32_000 + ((restart_index * 97 + repeat_index * 19 + offset) % 4_000)
        for offset in range(args.header_tokens)
    ]
    target_header = [
        36_000 + ((restart_index * 89 + repeat_index * 23 + offset) % 4_000)
        for offset in range(args.header_tokens)
    ]
    setting_id = f"{candidate}:b{body_tokens}:r{restart_index}:q{repeat_index}:{arm}"
    registration = source_registration(
        port=args.port,
        candidate=candidate,
        body=body,
        source_header=source_header,
        setting_id=setting_id,
        segment_tokens=args.segment_tokens,
        register=arm == "approx",
    )
    after_setup = metric_snapshot(args.port)
    target_logical_tokens = math.ceil(args.target_rho * capacity)
    setup_tokens = resident_tokens(after_setup)
    filler_count = max(
        0,
        math.ceil(
            (target_logical_tokens - setup_tokens - args.header_tokens)
            / args.filler_tokens
        ),
    )
    filler_ms = 0.0
    for filler_index in range(filler_count):
        result = generate(
            port=args.port,
            input_ids=filler_prompt(
                filler_index,
                args.filler_tokens,
                salt=restart_index * 1_000 + repeat_index,
            ),
            max_new_tokens=1,
        )
        filler_ms += float(result["elapsed_ms"])

    target_namespace = f"cl1-target:{setting_id}"
    seed_head = generate(
        port=args.port,
        input_ids=target_header,
        max_new_tokens=1,
        extra_key=target_namespace,
    )
    before_target = metric_snapshot(args.port)
    custom_params = None
    if arm == "approx":
        custom_params = {
            "approx_kv": approx_metadata(
                candidate=candidate,
                body_tokens=body_tokens,
                header_tokens=args.header_tokens,
                segment_tokens=args.segment_tokens,
                setting_id=setting_id,
            )
        }
    target_prompt = target_header + body + [901]
    target = stream_generate(
        port=args.port,
        input_ids=target_prompt,
        max_new_tokens=1,
        custom_params=custom_params,
        extra_key=target_namespace,
    )
    expected_cached_tokens = (
        args.header_tokens + body_tokens if arm == "approx" else args.header_tokens
    )
    after_target = metric_snapshot(args.port)

    quality_namespace = f"cl1-quality:{setting_id}"
    generate(
        port=args.port,
        input_ids=target_header,
        max_new_tokens=1,
        extra_key=quality_namespace,
    )
    quality = generate(
        port=args.port,
        input_ids=target_prompt,
        max_new_tokens=8,
        custom_params=custom_params,
        extra_key=quality_namespace,
    )
    post_quality = metric_snapshot(args.port)
    fallback_delta = counter_delta(
        before_target,
        after_target,
        "sglang:approx_kv_dense_fallback_total",
    )
    decode_eviction_delta = counter_delta(
        after_target,
        post_quality,
        "sglang:evicted_tokens_total",
    )
    flush_cache(args.port)
    reset = clean_cache_invariant(metric_snapshot(args.port))
    quality_ms = float(quality["elapsed_ms"])
    target_ms = float(target["ttft_ms"])
    seed_head_ms = float(seed_head["elapsed_ms"])
    request_path_ms = seed_head_ms + target_ms
    return {
        "arm": arm,
        "capacity_tokens": capacity,
        "target_rho": args.target_rho,
        "setup_resident_tokens": setup_tokens,
        "filler_count": filler_count,
        "observed_pre_target_rho": (resident_tokens(before_target) / capacity),
        "registration": registration,
        "filler_ms": filler_ms,
        "target": target,
        "expected_cached_tokens": expected_cached_tokens,
        "cache_path_matched": target["cached_tokens"] == expected_cached_tokens,
        "quality": quality,
        "fallback_tokens": fallback_delta,
        "decode_eviction_tokens": decode_eviction_delta,
        "ledger": {
            "setup_ms": float(registration["total_ms"]),
            "pressure_fill_ms": filler_ms,
            "seed_head_ms": seed_head_ms,
            "target_ttft_ms": target_ms,
            "request_path_ms": request_path_ms,
            "quality_canary_ms": quality_ms,
            "adapter_combined_ms": (
                float(registration["total_ms"]) + request_path_ms
                if arm == "approx"
                else request_path_ms
            ),
            "full_trace_ms": (
                float(registration["total_ms"])
                + filler_ms
                + request_path_ms
                + quality_ms
            ),
        },
        "reset_invariant": reset,
    }


def run_paired_setting(
    args: argparse.Namespace,
    *,
    candidate: str,
    body_tokens: int,
    restart_index: int,
    repeat_index: int,
) -> dict[str, Any]:
    order = ("dense", "approx") if repeat_index % 2 == 0 else ("approx", "dense")
    arms = {
        arm: run_arm(
            args,
            candidate=candidate,
            body_tokens=body_tokens,
            restart_index=restart_index,
            repeat_index=repeat_index,
            arm=arm,
        )
        for arm in order
    }
    dense = arms["dense"]
    approx = arms["approx"]
    dense_ttft = float(dense["target"]["ttft_ms"])
    approx_ttft = float(approx["target"]["ttft_ms"])
    dense_request_path = float(dense["ledger"]["request_path_ms"])
    approx_request_path = float(approx["ledger"]["request_path_ms"])
    first_token_match = (
        dense["target"]["output_ids"][:1] == approx["target"]["output_ids"][:1]
    )
    quality_match = dense["quality"]["output_ids"] == approx["quality"]["output_ids"]
    return {
        "repeat_index": repeat_index,
        "arm_order": list(order),
        "dense": dense,
        "approx": approx,
        "target_only_speedup": dense_ttft / approx_ttft,
        "request_path_speedup": dense_request_path / approx_request_path,
        "paired_delta_ms": dense_request_path - approx_request_path,
        "first_token_match": first_token_match,
        "quality_8_token_match": quality_match,
        "passed": (
            first_token_match
            and quality_match
            and (approx["fallback_tokens"] or 0) == 0
            and dense["cache_path_matched"]
            and approx["cache_path_matched"]
            and dense["reset_invariant"]["passed"]
            and approx["reset_invariant"]["passed"]
        ),
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_body = {}
    for body_tokens in sorted({int(row["body_tokens"]) for row in rows}):
        selected = [row for row in rows if row["body_tokens"] == body_tokens]
        dense = [
            float(repeat["dense"]["ledger"]["request_path_ms"])
            for row in selected
            for repeat in row["formal"]
        ]
        approx = [
            float(repeat["approx"]["ledger"]["request_path_ms"])
            for row in selected
            for repeat in row["formal"]
        ]
        target_dense = [
            float(repeat["dense"]["target"]["ttft_ms"])
            for row in selected
            for repeat in row["formal"]
        ]
        target_approx = [
            float(repeat["approx"]["target"]["ttft_ms"])
            for row in selected
            for repeat in row["formal"]
        ]
        speedups = [
            float(repeat["request_path_speedup"])
            for row in selected
            for repeat in row["formal"]
        ]
        amortized = {}
        for reuse_count in (1, 2, 4, 8):
            samples = [
                (float(repeat["dense"]["ledger"]["request_path_ms"]) * reuse_count)
                / (
                    float(repeat["approx"]["ledger"]["setup_ms"])
                    + float(repeat["approx"]["ledger"]["request_path_ms"]) * reuse_count
                )
                for row in selected
                for repeat in row["formal"]
            ]
            amortized[str(reuse_count)] = statistics.median(samples)
        break_even = [
            (
                float(repeat["approx"]["ledger"]["setup_ms"])
                / (
                    float(repeat["dense"]["ledger"]["request_path_ms"])
                    - float(repeat["approx"]["ledger"]["request_path_ms"])
                )
                if float(repeat["dense"]["ledger"]["request_path_ms"])
                > float(repeat["approx"]["ledger"]["request_path_ms"])
                else None
            )
            for row in selected
            for repeat in row["formal"]
        ]
        finite_break_even = [value for value in break_even if value is not None]
        by_body[str(body_tokens)] = {
            "median_request_path_speedup": statistics.median(speedups),
            "paired_request_path_p95_ratio": percentile(approx, 0.95)
            / percentile(dense, 0.95),
            "paired_target_p95_ratio": percentile(target_approx, 0.95)
            / percentile(target_dense, 0.95),
            "amortized_speedup": amortized,
            "median_break_even_reuses": (
                statistics.median(finite_break_even) if finite_break_even else None
            ),
            "all_guardrails_passed": all(
                repeat["passed"] for row in selected for repeat in row["formal"]
            ),
            "per_restart_median_speedup": [
                statistics.median(
                    repeat["request_path_speedup"] for repeat in row["formal"]
                )
                for row in selected
            ],
        }
    return by_body


def promotion(candidates: dict[str, dict[str, Any]], restarts: int) -> dict[str, Any]:
    if restarts < 3:
        ranked = sorted(
            candidates,
            key=lambda candidate: candidates[candidate]["2048"][
                "median_request_path_speedup"
            ],
            reverse=True,
        )
        return {
            "status": "screening_only",
            "provisional_ranking": ranked,
            "winner": None,
        }
    passing = []
    for candidate, summary in candidates.items():
        body = summary["2048"]
        restart_wins = sum(value > 1.0 for value in body["per_restart_median_speedup"])
        if (
            body["all_guardrails_passed"]
            and restart_wins >= 2
            and body["paired_target_p95_ratio"] <= 1.05
            and body["amortized_speedup"]["8"] > 1.0
        ):
            passing.append(candidate)
    if not passing:
        return {"status": "complete", "winner": "NONE", "passing": []}
    best_speedup = max(
        candidates[candidate]["2048"]["median_request_path_speedup"]
        for candidate in passing
    )
    near_ties = [
        candidate
        for candidate in passing
        if best_speedup - candidates[candidate]["2048"]["median_request_path_speedup"]
        <= 0.02 * best_speedup
    ]
    near_ties.sort(
        key=lambda candidate: (
            candidate_k(candidate) or 0,
            -candidates[candidate]["2048"]["paired_target_p95_ratio"],
        ),
        reverse=True,
    )
    passing.sort(
        key=lambda candidate: candidates[candidate]["2048"][
            "median_request_path_speedup"
        ],
        reverse=True,
    )
    return {
        "status": "complete",
        "winner": near_ties[0],
        "passing": passing,
    }


def execute(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    candidates = csv_values(args.candidates, str)
    body_values = csv_values(args.body_tokens, int)
    unknown = set(candidates).difference(VALID_CANDIDATES)
    if unknown:
        raise ValueError(f"unknown candidates: {sorted(unknown)}")
    if args.formal_repeats < 2 or args.restarts <= 0:
        raise ValueError("formal repeats and restarts are invalid")
    provenance = source_provenance(args.source_git_sha)
    observed_sha = provenance["source_git_sha"]

    results = []
    server_manifests = []
    for candidate in candidates:
        k = candidate_k(candidate)
        for restart_index in range(args.restarts):
            env = {"SGLANG_APPROX_KV_CORE": "1"}
            if k is not None:
                env.update(
                    {
                        "SGLANG_APPROX_KV_EPIC": "1",
                        "SGLANG_APPROX_KV_EPIC_K": str(k),
                    }
                )
            log_path = args.log_dir / (f"cl1-{candidate}-restart{restart_index}.log")
            server = launch_server(
                model=args.model,
                model_revision=args.model_revision,
                port=args.port,
                mem_fraction_static=args.mem_fraction_static,
                chunked_prefill_size=args.chunked_prefill_size,
                policy="lru",
                log_path=log_path,
                plugin_env=env,
                server_seed=17 + restart_index,
            )
            try:
                wait_ready(
                    server,
                    port=args.port,
                    timeout_s=args.server_start_timeout_s,
                )
                for body_tokens in body_values:
                    run_paired_setting(
                        args,
                        candidate=candidate,
                        body_tokens=body_tokens,
                        restart_index=restart_index,
                        repeat_index=-1,
                    )
                    formal = [
                        run_paired_setting(
                            args,
                            candidate=candidate,
                            body_tokens=body_tokens,
                            restart_index=restart_index,
                            repeat_index=repeat_index,
                        )
                        for repeat_index in range(args.formal_repeats)
                    ]
                    results.append(
                        {
                            "candidate": candidate,
                            "body_tokens": body_tokens,
                            "restart_index": restart_index,
                            "formal": formal,
                        }
                    )
                server_manifests.append(
                    {
                        "candidate": candidate,
                        "restart_index": restart_index,
                        "server_argv": list(server.command),
                        "plugin_env": server.plugin_env,
                        "log_path": str(log_path),
                    }
                )
            finally:
                stop_server(server)

    summaries = {
        candidate: summarize_candidate(
            [row for row in results if row["candidate"] == candidate]
        )
        for candidate in candidates
    }
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": "CL1",
        "source_git_sha": observed_sha,
        "source_tree_sha": provenance["source_tree_sha"],
        "result_git_sha": None,
        "result_commit_status": "pending_result_commit",
        "model": args.model,
        "model_revision": args.model_revision,
        "image_digest": args.image_digest,
        "machine": machine_manifest(),
        "settings": {
            "candidates": list(candidates),
            "body_tokens": list(body_values),
            "header_tokens": args.header_tokens,
            "segment_tokens": args.segment_tokens,
            "target_rho_logical_demand": args.target_rho,
            "scheduler": "S0",
            "prefetch": "P0",
            "chunked_prefill_size": args.chunked_prefill_size,
            "formal_repeats": args.formal_repeats,
            "restarts": args.restarts,
        },
        "ledger_definitions": {
            "target_only": "target_ttft_ms",
            "adapter_combined": "approximate source materialization and "
            "registration plus target TTFT",
            "full_trace": "setup plus pressure fill plus target plus quality canary",
            "excluded": ["server startup", "model download"],
        },
        "server_manifests": server_manifests,
        "results": results,
        "summaries": summaries,
        "promotion": promotion(summaries, args.restarts),
        "performance_claim": "candidate_qualification_only",
    }
    payload["raw_sha256"] = payload_sha256(payload)
    return payload


def main() -> int:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("cl1-%Y%m%dT%H%M%SZ")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    append_jsonl(
        args.central_log,
        {
            "run_id": run_id,
            "phase": "CL1",
            "status": "running",
            "output": str(args.output.resolve()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        payload = execute(args, run_id)
        for manifest in payload["server_manifests"]:
            manifest["log_sha256"] = file_sha256(Path(manifest["log_path"]))
        payload.pop("raw_sha256", None)
        payload["raw_sha256"] = payload_sha256(payload)
        write_json(args.output, payload)
        append_jsonl(
            args.central_log,
            {
                "run_id": run_id,
                "phase": "CL1",
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
            "phase": "CL1",
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
                "phase": "CL1",
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
