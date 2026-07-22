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
    parser.add_argument("--mode", choices=("dense", "kvcomm"), required=True)
    parser.add_argument("--target-rho", type=float, required=True)
    parser.add_argument("--body-tokens", type=int, required=True)
    parser.add_argument("--header-tokens", type=int, required=True)
    parser.add_argument("--segment-tokens", type=int, default=512)
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


def flush(base_url: str, salt: int) -> None:
    requests.post(
        f"{base_url}/flush_cache",
        json={},
        timeout=60,
    ).raise_for_status()
    time.sleep(0.1)
    request(base_url, [80_000 + salt, 80_100 + salt])
    time.sleep(0.1)


def filler_prompt(index: int, length: int) -> list[int]:
    return [90_000 + index] + [
        100_000 + ((index * 977 + offset * 37) % 40_000) for offset in range(length - 1)
    ]


def kvcomm_metadata(
    *,
    operation: str,
    action: str,
    segments: list[dict],
    descriptors: list[dict],
    context: str,
) -> dict:
    return {
        "operation": operation,
        "model_fingerprint": "qwen3-0.6b-sm75",
        "cache_dtype": "float16",
        "plugin": "kvcomm",
        "segments": segments,
        "plugin_params": {
            "action": action,
            "agent_id": "coder",
            "tokenizer_fingerprint": "direct-input-ids-v1",
            "template_fingerprint": "phase4-unified-pressure-v1",
            "context_fingerprint": context,
            "segments": descriptors,
            "entropy_threshold": 1.0,
            "temperature": 1.0,
            "max_anchors": 20,
            "prune_window": 5,
            "min_anchors": 2,
        },
    }


def descriptor(
    *,
    placeholder_id: str,
    source_fingerprint: str,
) -> dict:
    return {
        "segment_index": 0,
        "placeholder_id": placeholder_id,
        "role": "placeholder",
        "source_fingerprint": source_fingerprint,
    }


def register_kvcomm_setup(
    args: argparse.Namespace,
    round_index: int,
) -> tuple[list[int], list[dict], list[dict], float]:
    target_body = list(range(1_000, 1_000 + args.body_tokens))
    anchor_one_body = list(range(10_000, 10_000 + args.body_tokens))
    anchor_two_body = list(range(20_000, 20_000 + args.body_tokens))
    anchor_one_head = list(range(50_000, 50_000 + args.header_tokens))
    anchor_two_head = list(range(70_000, 70_000 + args.header_tokens))
    target_segments = []
    target_descriptors = []
    started = time.perf_counter()

    for chunk_index, start in enumerate(
        range(0, args.body_tokens, args.segment_tokens)
    ):
        target_chunk = target_body[start : start + args.segment_tokens]
        anchor_one_chunk = anchor_one_body[start : start + args.segment_tokens]
        anchor_two_chunk = anchor_two_body[start : start + args.segment_tokens]
        placeholder_id = f"unified-body-chunk-{chunk_index}"
        target_hash = f"kvcomm-r{round_index}-target-{chunk_index}"
        anchor_one_hash = f"kvcomm-r{round_index}-anchor-one-{chunk_index}"
        anchor_two_hash = f"kvcomm-r{round_index}-anchor-two-{chunk_index}"
        target_source = f"target-source-{chunk_index}"
        anchor_one_source = f"anchor-one-source-{chunk_index}"
        anchor_two_source = f"anchor-two-source-{chunk_index}"

        for token_ids, content_hash, source, context in (
            (
                target_chunk,
                target_hash,
                target_source,
                f"target-canonical-{chunk_index}",
            ),
            (
                anchor_one_chunk,
                anchor_one_hash,
                anchor_one_source,
                f"anchor-one-canonical-{chunk_index}",
            ),
            (
                anchor_two_chunk,
                anchor_two_hash,
                anchor_two_source,
                f"anchor-two-canonical-{chunk_index}",
            ),
        ):
            request(
                args.base_url,
                token_ids + [900 + chunk_index],
                kvcomm_metadata(
                    operation="register",
                    action="base",
                    segments=[
                        {
                            "content_hash": content_hash,
                            "target_start": 0,
                            "length": len(token_ids),
                        }
                    ],
                    descriptors=[
                        descriptor(
                            placeholder_id=placeholder_id,
                            source_fingerprint=source,
                        )
                    ],
                    context=context,
                ),
            )

        for head, token_ids, content_hash, source, context in (
            (
                anchor_one_head,
                anchor_one_chunk,
                anchor_one_hash,
                anchor_one_source,
                f"anchor-one-context-{chunk_index}",
            ),
            (
                anchor_two_head,
                anchor_two_chunk,
                anchor_two_hash,
                anchor_two_source,
                f"anchor-two-context-{chunk_index}",
            ),
        ):
            request(
                args.base_url,
                head + token_ids + [920 + chunk_index],
                kvcomm_metadata(
                    operation="register",
                    action="anchor",
                    segments=[
                        {
                            "content_hash": content_hash,
                            "target_start": args.header_tokens,
                            "length": len(token_ids),
                        }
                    ],
                    descriptors=[
                        descriptor(
                            placeholder_id=placeholder_id,
                            source_fingerprint=source,
                        )
                    ],
                    context=context,
                ),
            )

        target_segments.append(
            {
                "content_hash": target_hash,
                "target_start": args.header_tokens + start,
                "length": len(target_chunk),
            }
        )
        target_descriptors.append(
            {
                "segment_index": chunk_index,
                "placeholder_id": placeholder_id,
                "role": "placeholder",
                "source_fingerprint": target_source,
            }
        )

    return (
        target_body,
        target_segments,
        target_descriptors,
        (time.perf_counter() - started) * 1000,
    )


def run_round(args: argparse.Namespace, round_index: int) -> dict:
    flush(args.base_url, round_index)
    baseline = metric_snapshot(args.base_url)
    target_header = list(range(60_000, 60_000 + args.header_tokens))
    setup_ms = 0.0
    if args.mode == "kvcomm":
        (
            target_body,
            target_segments,
            target_descriptors,
            setup_ms,
        ) = register_kvcomm_setup(args, round_index)
    else:
        target_body = list(range(1_000, 1_000 + args.body_tokens))
        target_segments = []
        target_descriptors = []

    after_setup = metric_snapshot(args.base_url)
    capacity = usable_kv_capacity_tokens(baseline)
    available = after_setup.get("sglang:kv_available_tokens", capacity)
    resident_after_setup = max(0, capacity - int(round(available)))
    target_working_tokens = int(math.ceil(args.target_rho * capacity))
    filler_count = max(
        0,
        math.ceil(
            (target_working_tokens - resident_after_setup - args.header_tokens)
            / args.filler_tokens
        ),
    )
    for filler_index in range(filler_count):
        request(
            args.base_url,
            filler_prompt(filler_index, args.filler_tokens) + [950],
        )
    if target_header:
        request(args.base_url, target_header)

    before_target = metric_snapshot(args.base_url)
    metadata = (
        kvcomm_metadata(
            operation="reuse",
            action="reuse",
            segments=target_segments,
            descriptors=target_descriptors,
            context=f"target-context-round-{round_index}",
        )
        if args.mode == "kvcomm"
        else None
    )
    target = request(
        args.base_url,
        target_header + target_body + [901],
        metadata,
    )
    after_target = metric_snapshot(args.base_url)
    expected_cached = (
        args.header_tokens + args.body_tokens
        if args.mode == "kvcomm"
        else args.header_tokens
    )
    if target["cached_tokens"] != expected_cached:
        raise RuntimeError(
            f"unexpected cached_tokens={target['cached_tokens']}, "
            f"expected={expected_cached}"
        )
    target_delta = telemetry_delta(before_target, after_target)
    if args.mode == "kvcomm" and target_delta["dense_fallbacks"] not in (None, 0):
        raise RuntimeError("KVCOMM pressure target used dense fallback")
    copied_tokens_delta = after_target.get(
        "sglang:approx_kv_copied_tokens_total", 0.0
    ) - before_target.get("sglang:approx_kv_copied_tokens_total", 0.0)
    if args.mode == "kvcomm" and copied_tokens_delta != args.body_tokens:
        raise RuntimeError(
            f"KVCOMM copied {copied_tokens_delta} tokens, "
            f"expected {args.body_tokens}"
        )

    declared_tokens = (
        resident_after_setup + args.header_tokens + filler_count * args.filler_tokens
    )
    return {
        "round_index": round_index,
        "capacity_tokens": capacity,
        "target_rho": args.target_rho,
        "pre_target_rho": declared_tokens / capacity,
        "peak_rho_with_target": (declared_tokens + args.body_tokens) / capacity,
        "resident_after_setup": resident_after_setup,
        "filler_count": filler_count,
        "setup_ms": setup_ms,
        "kvcomm_copied_tokens_delta": copied_tokens_delta,
        "segment_count": math.ceil(args.body_tokens / args.segment_tokens),
        "target": target,
        "baseline_metrics": metric_subset(baseline),
        "after_setup_metrics": metric_subset(after_setup),
        "before_target_metrics": metric_subset(before_target),
        "after_target_metrics": metric_subset(after_target),
        "pressure_delta": telemetry_delta(baseline, before_target),
        "target_delta": target_delta,
    }


def main() -> None:
    args = parse_args()
    if args.target_rho <= 0:
        raise ValueError("target_rho must be positive")
    if (
        min(
            args.body_tokens,
            args.segment_tokens,
            args.filler_tokens,
        )
        <= 0
    ):
        raise ValueError("body, segment, and filler sizes must be positive")

    run_id = (
        f"phase4-kvcomm-pressure-{args.mode}-"
        f"rho{args.target_rho:.3f}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    settings = {
        "mode": args.mode,
        "target_rho": args.target_rho,
        "body_tokens": args.body_tokens,
        "header_tokens": args.header_tokens,
        "segment_tokens": args.segment_tokens,
        "filler_tokens": args.filler_tokens,
        "target_prompt_tokens": (args.header_tokens + args.body_tokens + 1),
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
        request(args.base_url, list(range(80_200, 80_938)))
        warmup = run_round(args, -1)
        rows = [run_round(args, index) for index in range(args.repeats)]
        values = [row["target"]["ttft_ms"] for row in rows]
        setup_values = [row["setup_ms"] for row in rows]
        result = {
            "schema_version": 1,
            "run_id": run_id,
            "settings": settings,
            "warmup": warmup,
            "rows": rows,
            "target_p50_ms": statistics.median(values),
            "setup_p50_ms": (
                statistics.median(setup_values) if args.mode == "kvcomm" else 0.0
            ),
            "eviction_observed_in_formal_runs": any(
                (
                    (
                        row["pressure_delta"]["counters"].get(
                            "sglang:evicted_tokens_total"
                        )
                        or 0
                    )
                    + (
                        row["target_delta"]["counters"].get(
                            "sglang:evicted_tokens_total"
                        )
                        or 0
                    )
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
                    "setup_p50_ms": result["setup_p50_ms"],
                    "eviction_observed": result["eviction_observed_in_formal_runs"],
                    "pre_target_rho": [row["pre_target_rho"] for row in rows],
                    "peak_rho_with_target": [
                        row["peak_rho_with_target"] for row in rows
                    ],
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
