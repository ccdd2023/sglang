#!/usr/bin/env python3
"""Phase 4 R0 (Raw+RoPE) unified high-pressure benchmark runner.

This drives a real, running SGLang server (GPU, S0 LRU scheduler,
GPU-only tier, prefetch disabled) and compares client-observed streaming
TTFT between a fresh ``dense`` prefill and the ``raw`` (raw copy + RoPE
relocation) recovery path, under real Radix-cache eviction pressure
calibrated against the server's *actual* usable KV capacity (read back
from live Prometheus metrics -- the historical Phase 2 ~13,130-token
estimate is never assumed).

This is **not** a copy of the R1/EPIC leading-k pressure runner's repair
logic: there is no ``k`` parameter and no per-layer recompute. R0 only ever
does a raw device-to-device KV copy plus a signed RoPE position
relocation (``raw_rope.py``); this script only reuses the generic,
paper-agnostic pressure-harness scaffolding (metrics snapshot/delta,
warmup/repeat/central-log bookkeeping) that is shared across Phase 4
research branches.

Unified Phase 4 benchmark contract (values confirmed by the user):

- exact-prefix header lengths: ``0, 32, 64, 128, 256`` tokens;
- lossy body lengths: ``512, 768, 1024, 2048`` tokens;
- a canonical source body longer than 512 tokens is registered as
  multiple ``<=512``-token segments (one ``register`` request per
  segment); the target request recovers them back into one contiguous
  span;
- ``mem_fraction_static=0.35``; pressure is calibrated after startup
  against the *actual* usable KV capacity reported by
  ``sglang:kv_available_tokens``/``sglang:kv_evictable_tokens``/
  ``sglang:kv_used_tokens``, targeting pre-target rho of approximately
  ``0.9 / 1.1 / 1.5 / 2 / 3``;
- fixed S0 LRU scheduler, GPU-only residency tier, prefetch disabled;
- exactly one discarded warmup pass per setting, then formal repeats
  (default 4, rejected below 2);
- every invocation appends a running/completed/failed record (full
  settings, raw result path, and a compact summary) to the central
  ``BENCHMARK_RUN_LOG.jsonl``.

Start the server (S0 LRU / GPU-only / prefetch-off) with, e.g.:

.. code-block:: text

    SGLANG_APPROX_KV_CORE=1
    SGLANG_APPROX_KV_RAW_ROPE=1
    python3 -m sglang.launch_server \\
        --model-path Qwen/Qwen3-0.6B \\
        --mem-fraction-static 0.35 \\
        --disable-radix-cache false \\
        ...

Then run this script once per (mode, header, body, rho) setting, e.g.:

.. code-block:: text

    python3 -m benchmark.approx_kv.run_r0_pressure \\
        --base-url http://127.0.0.1:30000 \\
        --mode raw \\
        --header-tokens 64 \\
        --body-tokens 1024 \\
        --target-rho 2.0 \\
        --runner-git-sha <sha> \\
        --image-digest sha256:<digest> \\
        --output benchmark/approx_kv/results/phase4-r0/<setting>.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from benchmark.approx_kv.metrics import (
    metric_subset,
    parse_prometheus_text,
    telemetry_delta,
    usable_kv_capacity_tokens,
)

# Unified Phase 4 benchmark contract (see module docstring).
HEADER_TOKENS_CONTRACT = (0, 32, 64, 128, 256)
BODY_TOKENS_CONTRACT = (512, 768, 1024, 2048)
DEFAULT_SEGMENT_TOKENS = 512
DEFAULT_MEM_FRACTION_STATIC = 0.35
DEFAULT_CENTRAL_LOG = (
    "/home/chris/Workspaces/kvcache-research/results/BENCHMARK_RUN_LOG.jsonl"
)


def _repeat_count(value: str) -> int:
    repeats = int(value)
    if repeats < 2:
        raise argparse.ArgumentTypeError("repeats must be at least 2")
    return repeats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--mode", choices=("dense", "raw"), required=True)
    parser.add_argument(
        "--header-tokens",
        type=int,
        choices=HEADER_TOKENS_CONTRACT,
        required=True,
        help="exact-prefix header length; unified contract: 0/32/64/128/256",
    )
    parser.add_argument(
        "--body-tokens",
        type=int,
        choices=BODY_TOKENS_CONTRACT,
        required=True,
        help="lossy body length; unified contract: 512/768/1024/2048",
    )
    parser.add_argument(
        "--target-rho",
        type=float,
        required=True,
        help="target pre-target reusable rho; unified contract: ~0.9/1.1/1.5/2/3",
    )
    parser.add_argument("--filler-tokens", type=int, default=736)
    parser.add_argument(
        "--segment-tokens",
        type=int,
        default=DEFAULT_SEGMENT_TOKENS,
        help=(
            "canonical source body longer than this is registered as "
            "multiple <=segment-tokens chunks; contract default 512"
        ),
    )
    parser.add_argument("--repeats", type=_repeat_count, default=4)
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=DEFAULT_MEM_FRACTION_STATIC,
        help="recorded only -- must match the server's actual launch flag",
    )
    parser.add_argument("--runner-git-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--central-log", default=DEFAULT_CENTRAL_LOG)
    return parser.parse_args()


def append_log(path: str, entry: dict) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, sort_keys=True))
        file.write("\n")


def metric_snapshot(base_url: str) -> dict[str, float]:
    response = requests.get(f"{base_url}/metrics", timeout=30)
    response.raise_for_status()
    return parse_prometheus_text(response.text)


def request(
    base_url: str,
    input_ids: list[int],
    metadata: dict | None = None,
) -> dict:
    """Issue one streaming generate request and return client-observed TTFT.

    ``max_new_tokens=1`` and the streaming response is consumed so the TTFT
    reflects the real time-to-first-token a client would observe, not a
    server-side timestamp.
    """
    sampling_params = {"max_new_tokens": 1, "temperature": 0}
    if metadata is not None:
        sampling_params["custom_params"] = {"approx_kv": metadata}
    start = time.perf_counter()
    first = None
    payload = None
    saw_done = False
    with requests.post(
        f"{base_url}/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "stream": True,
        },
        stream=True,
        timeout=180,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if line == "data: [DONE]":
                saw_done = True
            elif line.startswith("data: ") and first is None:
                first = time.perf_counter()
                payload = json.loads(line[6:])
    if first is None or payload is None or not saw_done:
        raise RuntimeError("incomplete streaming response")
    return {
        "ttft_ms": (first - start) * 1000,
        "cached_tokens": payload["meta_info"]["cached_tokens"],
        "output_ids": payload["output_ids"],
    }


def flush(base_url: str, sentinel_salt: int) -> None:
    requests.post(
        f"{base_url}/flush_cache",
        json={},
        timeout=60,
    ).raise_for_status()
    time.sleep(0.1)
    request(base_url, [80_000 + sentinel_salt, 80_100 + sentinel_salt])
    time.sleep(0.1)


def filler_prompt(index: int, length: int) -> list[int]:
    first = 20_000 + index
    return [first] + [
        30_000 + ((index * 977 + offset * 37) % 20_000) for offset in range(length - 1)
    ]


def build_metadata(
    *,
    operation: str,
    segments: list[dict],
    plugin: str | None = None,
) -> dict:
    metadata = {
        "operation": operation,
        "model_fingerprint": "qwen3-0.6b-sm75",
        "cache_dtype": "float16",
        "segments": segments,
    }
    if plugin is not None:
        metadata["plugin"] = plugin
    return metadata


def body_segments(body: list[int], segment_tokens: int) -> list[list[int]]:
    """Split a canonical source body into <=segment_tokens-token chunks.

    Required by the unified contract: any canonical source body longer
    than ``segment_tokens`` (default 512) is registered chunk-by-chunk so
    the R0 registration path never has to allocate one oversized device
    span; the target request still recovers the full body as one
    contiguous span across all chunks.
    """
    return [
        body[start : start + segment_tokens]
        for start in range(0, len(body), segment_tokens)
    ]


def run_round(args: argparse.Namespace, round_index: int) -> dict:
    flush(args.base_url, round_index)
    baseline = metric_snapshot(args.base_url)
    capacity = usable_kv_capacity_tokens(baseline)
    persistent_tokens = args.body_tokens + args.header_tokens + 2
    target_working_tokens = int(math.ceil(args.target_rho * capacity))
    filler_count = max(
        0,
        math.ceil((target_working_tokens - persistent_tokens) / args.filler_tokens),
    )
    body = list(range(1_000, 1_000 + args.body_tokens))
    source_header = list(range(50_000, 50_000 + args.header_tokens))
    target_header = list(range(60_000, 60_000 + args.header_tokens))
    content_hash = (
        f"r0-pressure-{args.mode}-rho{args.target_rho:.3f}-round{round_index}"
    )
    chunks = body_segments(body, args.segment_tokens)
    target_segments = []
    cursor = args.header_tokens
    for chunk_index, chunk in enumerate(chunks):
        target_segments.append(
            {
                "content_hash": f"{content_hash}-chunk{chunk_index}",
                "target_start": cursor,
                "length": len(chunk),
            }
        )
        cursor += len(chunk)

    if args.mode == "raw":
        for chunk_index, chunk in enumerate(chunks):
            request(
                args.base_url,
                source_header + chunk + [900 + chunk_index],
                build_metadata(
                    operation="register",
                    segments=[
                        {
                            "content_hash": f"{content_hash}-chunk{chunk_index}",
                            "target_start": args.header_tokens,
                            "length": len(chunk),
                        }
                    ],
                ),
            )
    else:
        for chunk_index, chunk in enumerate(chunks):
            request(args.base_url, chunk + [900 + chunk_index])

    for filler_index in range(filler_count):
        request(
            args.base_url,
            filler_prompt(filler_index, args.filler_tokens) + [950],
        )

    if target_header:
        request(args.base_url, target_header)
    before_target = metric_snapshot(args.base_url)
    target_ids = target_header + body + [901]
    metadata = (
        build_metadata(
            operation="reuse",
            segments=target_segments,
            plugin="raw_rope",
        )
        if args.mode == "raw"
        else None
    )
    target = request(args.base_url, target_ids, metadata)
    after_target = metric_snapshot(args.base_url)

    expected_cached = (
        args.header_tokens + args.body_tokens
        if args.mode == "raw"
        else args.header_tokens
    )
    if target["cached_tokens"] != expected_cached:
        raise RuntimeError(
            f"unexpected cached_tokens={target['cached_tokens']}, "
            f"expected={expected_cached}"
        )
    target_delta = telemetry_delta(before_target, after_target)
    if args.mode == "raw" and target_delta["dense_fallbacks"] not in (None, 0):
        raise RuntimeError("R0 raw+RoPE pressure target used dense fallback")

    pressure_delta = telemetry_delta(baseline, before_target)
    evicted_pressure = (
        pressure_delta["counters"].get("sglang:evicted_tokens_total") or 0
    )
    evicted_target = target_delta["counters"].get("sglang:evicted_tokens_total") or 0
    declared_tokens = persistent_tokens + filler_count * args.filler_tokens
    return {
        "round_index": round_index,
        "capacity_tokens": capacity,
        "target_rho": args.target_rho,
        "actual_declared_rho": declared_tokens / capacity,
        "pre_target_rho": declared_tokens / capacity,
        "peak_rho_with_target": (declared_tokens + args.body_tokens) / capacity,
        "filler_count": filler_count,
        "declared_working_tokens": declared_tokens,
        "segment_tokens": args.segment_tokens,
        "segment_count": len(chunks),
        "target": target,
        "baseline_metrics": metric_subset(baseline),
        "before_target_metrics": metric_subset(before_target),
        "after_target_metrics": metric_subset(after_target),
        "pressure_delta": pressure_delta,
        "target_delta": target_delta,
        "evicted_tokens_pressure": evicted_pressure,
        "evicted_tokens_target": evicted_target,
        "evicted_tokens_total": evicted_pressure + evicted_target,
    }


def main() -> None:
    args = parse_args()
    if args.target_rho <= 0:
        raise ValueError("target_rho must be positive")
    if args.segment_tokens <= 0:
        raise ValueError("segment_tokens must be positive")
    prompt_tokens = args.header_tokens + args.body_tokens + 1
    crosses_chunk_boundary = prompt_tokens > 1024

    run_id = (
        f"phase4-r0-pressure-{args.mode}-"
        f"header{args.header_tokens}-body{args.body_tokens}-"
        f"rho{args.target_rho:.3f}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    settings = {
        "phase": "phase4-r0",
        "mode": args.mode,
        "header_tokens": args.header_tokens,
        "body_tokens": args.body_tokens,
        "target_rho": args.target_rho,
        "filler_tokens": args.filler_tokens,
        "segment_tokens": args.segment_tokens,
        "target_prompt_tokens": prompt_tokens,
        "crosses_1024_token_chunk_boundary": crosses_chunk_boundary,
        "global_warmup_passes": 1,
        "per_setting_warmup_passes": 1,
        "formal_repeats": args.repeats,
        "mem_fraction_static": args.mem_fraction_static,
        "scheduler": "S0 LRU",
        "tier": "GPU-only",
        "prefetch": False,
        "runner_git_sha": args.runner_git_sha,
        "image_digest": args.image_digest,
    }
    append_log(
        args.central_log,
        {
            "run_id": run_id,
            "status": "running",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "settings": settings,
            "output": str(Path(args.output).resolve()),
        },
    )
    try:
        request(args.base_url, list(range(70_000, 70_000 + args.filler_tokens)))
        warmup = run_round(args, -1)
        rows = [run_round(args, index) for index in range(args.repeats)]
        values = [row["target"]["ttft_ms"] for row in rows]
        result = {
            "schema_version": 1,
            "run_id": run_id,
            "settings": settings,
            "warmup": warmup,
            "rows": rows,
            "target_p50_ms": statistics.median(values),
            "eviction_observed_in_formal_runs": any(
                row["evicted_tokens_total"] > 0 for row in rows
            ),
            "total_evicted_tokens_in_formal_runs": sum(
                row["evicted_tokens_total"] for row in rows
            ),
            "passed": True,
        }
        output_path = Path(args.output)
        if output_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing result file: {output_path}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        append_log(
            args.central_log,
            {
                "run_id": run_id,
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "settings": settings,
                "output": str(output_path.resolve()),
                "result_summary": {
                    "target_p50_ms": result["target_p50_ms"],
                    "eviction_observed": result["eviction_observed_in_formal_runs"],
                    "total_evicted_tokens": result[
                        "total_evicted_tokens_in_formal_runs"
                    ],
                    "actual_declared_rho": [row["actual_declared_rho"] for row in rows],
                    "pre_target_rho": [row["pre_target_rho"] for row in rows],
                    "peak_rho_with_target": [
                        row["peak_rho_with_target"] for row in rows
                    ],
                    "filler_count": [row["filler_count"] for row in rows],
                },
            },
        )
    except Exception as exc:
        append_log(
            args.central_log,
            {
                "run_id": run_id,
                "status": "failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "settings": settings,
                "output": str(Path(args.output).resolve()),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


if __name__ == "__main__":
    main()
