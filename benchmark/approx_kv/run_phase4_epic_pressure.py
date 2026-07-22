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


def _repeat_count(value: str) -> int:
    repeats = int(value)
    if repeats < 2:
        raise argparse.ArgumentTypeError("repeats must be at least 2")
    return repeats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30011")
    parser.add_argument("--mode", choices=("dense", "epic"), required=True)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--target-rho", type=float, required=True)
    parser.add_argument("--body-tokens", type=int, default=736)
    parser.add_argument("--header-tokens", type=int, default=64)
    parser.add_argument("--filler-tokens", type=int, default=736)
    parser.add_argument("--repeats", type=_repeat_count, default=4)
    parser.add_argument("--runner-git-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--central-log", required=True)
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
    content_hash: str,
    header_tokens: int,
    body_tokens: int,
    plugin: str | None = None,
) -> dict:
    metadata = {
        "operation": operation,
        "model_fingerprint": "qwen3-0.6b-sm75",
        "cache_dtype": "float16",
        "segments": [
            {
                "content_hash": content_hash,
                "target_start": header_tokens,
                "length": body_tokens,
            }
        ],
    }
    if plugin is not None:
        metadata["plugin"] = plugin
    return metadata


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
        f"epic-pressure-k{args.k}-rho{args.target_rho:.3f}-round{round_index}"
    )

    if args.mode == "epic":
        request(
            args.base_url,
            source_header + body + [900],
            build_metadata(
                operation="register",
                content_hash=content_hash,
                header_tokens=args.header_tokens,
                body_tokens=args.body_tokens,
            ),
        )
    else:
        request(args.base_url, body + [900])

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
            content_hash=content_hash,
            header_tokens=args.header_tokens,
            body_tokens=args.body_tokens,
            plugin="epic",
        )
        if args.mode == "epic"
        else None
    )
    target = request(args.base_url, target_ids, metadata)
    after_target = metric_snapshot(args.base_url)

    expected_cached = (
        args.header_tokens + args.body_tokens
        if args.mode == "epic"
        else args.header_tokens
    )
    if target["cached_tokens"] != expected_cached:
        raise RuntimeError(
            f"unexpected cached_tokens={target['cached_tokens']}, "
            f"expected={expected_cached}"
        )
    target_delta = telemetry_delta(before_target, after_target)
    if args.mode == "epic" and target_delta["dense_fallbacks"] not in (None, 0):
        raise RuntimeError("EPIC pressure target used dense fallback")

    declared_tokens = persistent_tokens + filler_count * args.filler_tokens
    return {
        "round_index": round_index,
        "capacity_tokens": capacity,
        "target_rho": args.target_rho,
        "actual_declared_rho": declared_tokens / capacity,
        "filler_count": filler_count,
        "declared_working_tokens": declared_tokens,
        "target": target,
        "baseline_metrics": metric_subset(baseline),
        "before_target_metrics": metric_subset(before_target),
        "after_target_metrics": metric_subset(after_target),
        "pressure_delta": telemetry_delta(baseline, before_target),
        "target_delta": target_delta,
    }


def main() -> None:
    args = parse_args()
    if args.target_rho <= 0:
        raise ValueError("target_rho must be positive")
    prompt_tokens = args.header_tokens + args.body_tokens + 1
    crosses_chunk_boundary = prompt_tokens > 1024

    run_id = (
        f"phase4-epic-pressure-{args.mode}-k{args.k}-"
        f"rho{args.target_rho:.3f}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    settings = {
        "mode": args.mode,
        "k": args.k if args.mode == "epic" else None,
        "target_rho": args.target_rho,
        "body_tokens": args.body_tokens,
        "header_tokens": args.header_tokens,
        "filler_tokens": args.filler_tokens,
        "target_prompt_tokens": prompt_tokens,
        "crosses_1024_token_chunk_boundary": crosses_chunk_boundary,
        "global_warmup_passes": 1,
        "per_setting_warmup_passes": 1,
        "formal_repeats": args.repeats,
        "mem_fraction_static": 0.35,
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
        request(args.base_url, list(range(70_000, 70_738)))
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
                (
                    row["pressure_delta"]["counters"].get("sglang:evicted_tokens_total")
                    or 0
                )
                > 0
                for row in rows
            ),
            "passed": True,
        }
        output_path = Path(args.output)
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
                    "actual_declared_rho": [row["actual_declared_rho"] for row in rows],
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
