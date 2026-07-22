#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

import aiohttp

from benchmark.approx_kv.metrics import parse_prometheus_text
from benchmark.approx_kv.workloads import (
    build_messages,
    build_object_catalog,
    common_prefix_token_ids,
    tokenize_messages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--cache-dtype", default="fp16")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fetch_text(url: str, timeout: float = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def post_json(url: str, payload: dict, timeout: float = 300) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def post_empty(url: str, timeout: float = 60) -> str:
    request = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def metric_snapshot(base_url: str) -> dict[str, float]:
    return parse_prometheus_text(fetch_text(f"{base_url}/metrics"))


def metric_delta(
    before: dict[str, float],
    after: dict[str, float],
    name: str,
) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


async def abort_after_first_stream_chunk(
    *,
    url: str,
    payload: dict,
) -> bool:
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json={**payload, "stream": True}) as response:
            response.raise_for_status()
            async for raw_line in response.content:
                if raw_line.decode("utf-8", errors="replace").startswith(
                    "data: "
                ):
                    response.close()
                    return True
    return False


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=True,
    )
    cache_object = build_object_catalog(
        tokenizer,
        object_count=4,
    )[0]
    source_messages = build_messages(
        cache_object,
        "phase3-source-branch",
        cache_salt="phase3-canary",
    )
    target_messages = build_messages(
        cache_object,
        "phase3-target-branch-with-different-suffix",
        cache_salt="phase3-canary",
    )
    source_ids = tokenize_messages(tokenizer, source_messages)
    target_ids = tokenize_messages(tokenizer, target_messages)
    reusable_tokens = len(common_prefix_token_ids(source_ids, target_ids))
    if reusable_tokens <= 0 or reusable_tokens >= min(
        len(source_ids),
        len(target_ids),
    ):
        raise RuntimeError("canary prompts do not have a partial stable prefix")

    def request_payload(messages, operation, max_tokens=1):
        return {
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "custom_params": {
                "approx_kv": {
                    "operation": operation,
                    "model_fingerprint": args.model_fingerprint,
                    "cache_dtype": args.cache_dtype,
                    "segments": [
                        {
                            "content_hash": "phase3-canary-artifact",
                            "target_start": 0,
                            "length": reusable_tokens,
                        }
                    ],
                }
            },
        }

    def request(messages, operation):
        return post_json(
            f"{args.base_url}/v1/chat/completions",
            request_payload(messages, operation),
        )

    metrics_before = metric_snapshot(args.base_url)
    register_response = request(source_messages, "register")
    time.sleep(0.25)
    reuse_response = request(target_messages, "reuse")
    time.sleep(0.25)

    mismatch_object = replace(
        cache_object,
        payload=f"changed-before-prefix\n{cache_object.payload}",
    )
    mismatch_messages = build_messages(
        mismatch_object,
        "phase3-target-branch-with-different-suffix",
        cache_salt="phase3-canary",
    )
    mismatch_response = request(mismatch_messages, "reuse")
    time.sleep(0.25)
    metrics_before_flush = metric_snapshot(args.base_url)

    flush_response = post_empty(f"{args.base_url}/flush_cache?timeout=30")
    post_flush_response = request(target_messages, "reuse")
    time.sleep(0.25)

    abort_register_response = request(source_messages, "register")
    abort_observed = asyncio.run(
        abort_after_first_stream_chunk(
            url=f"{args.base_url}/v1/chat/completions",
            payload=request_payload(target_messages, "reuse", max_tokens=128),
        )
    )
    if not abort_observed:
        raise RuntimeError("abort canary did not observe a streaming chunk")
    time.sleep(1)
    abort_health = fetch_text(f"{args.base_url}/health")
    final_flush_response = post_empty(
        f"{args.base_url}/flush_cache?timeout=30"
    )
    fetch_text(f"{args.base_url}/health_generate")
    time.sleep(0.25)
    metrics_after = metric_snapshot(args.base_url)
    health_status = fetch_text(f"{args.base_url}/health")

    deltas = {
        name: metric_delta(metrics_before, metrics_after, name)
        for name in (
            "sglang:approx_kv_host_export_tokens_total",
            "sglang:approx_kv_host_export_bytes_total",
            "sglang:approx_kv_h2d_tokens_total",
            "sglang:approx_kv_h2d_bytes_total",
            "sglang:approx_kv_copied_tokens_total",
            "sglang:approx_kv_dense_fallback_total",
        )
    }
    if deltas["sglang:approx_kv_host_export_tokens_total"] != 2 * reusable_tokens:
        raise RuntimeError(f"host export mismatch: {deltas}")
    if deltas["sglang:approx_kv_h2d_tokens_total"] != 2 * reusable_tokens:
        raise RuntimeError(f"H2D mismatch: {deltas}")
    if deltas["sglang:approx_kv_copied_tokens_total"] != 2 * reusable_tokens:
        raise RuntimeError(f"copy mismatch: {deltas}")
    if deltas["sglang:approx_kv_dense_fallback_total"] != 2 * reusable_tokens:
        raise RuntimeError(f"dense fallback mismatch: {deltas}")
    if any(
        response["choices"][0]["finish_reason"] != "length"
        for response in (
            register_response,
            reuse_response,
            mismatch_response,
            post_flush_response,
            abort_register_response,
        )
    ):
        raise RuntimeError("one-token canary request did not finish by length")

    maximum = metrics_after["sglang:max_total_num_tokens"]
    available = metrics_after["sglang:kv_available_tokens"]
    evictable = metrics_after["sglang:kv_evictable_tokens"]
    used = metrics_after["sglang:kv_used_tokens"]
    if available + evictable + used != maximum:
        raise RuntimeError("KV pool did not return to an accounted state")

    payload = {
        "model": args.model,
        "model_revision": args.model_revision,
        "model_fingerprint": args.model_fingerprint,
        "source_prompt_tokens": len(source_ids),
        "target_prompt_tokens": len(target_ids),
        "reusable_tokens": reusable_tokens,
        "metric_deltas": deltas,
        "metrics_before_flush": {
            name: value
            for name, value in metrics_before_flush.items()
            if "approx_kv_" in name
        },
        "metrics_after": {
            name: value
            for name, value in metrics_after.items()
            if "approx_kv_" in name
            or name
            in (
                "sglang:max_total_num_tokens",
                "sglang:kv_available_tokens",
                "sglang:kv_evictable_tokens",
                "sglang:kv_used_tokens",
            )
        },
        "flush_response": flush_response.strip(),
        "abort_observed": abort_observed,
        "abort_health_response": abort_health,
        "final_flush_response": final_flush_response.strip(),
        "health_response": health_status,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
