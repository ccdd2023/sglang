from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def _int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30011")
    parser.add_argument("--mode", choices=("epic", "dense"), required=True)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--body-sizes", type=_int_list, default="128,256,512")
    parser.add_argument("--head-sizes", type=_int_list, default="0,16,32,64,128")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--central-log", required=True)
    parser.add_argument("--runner-git-sha", required=True)
    parser.add_argument(
        "--image-digest",
        default="sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--model-revision",
        default="c1899de289a04d12100db370d81485cdf75e47ca",
    )
    return parser.parse_args()


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


def metric_snapshot(base_url: str) -> dict[str, float]:
    response = requests.get(f"{base_url}/metrics", timeout=30)
    response.raise_for_status()
    result = {}
    for line in response.text.splitlines():
        if line and not line.startswith("#") and "approx_kv" in line:
            name, value = line.rsplit(" ", 1)
            result[name] = float(value)
    return result


def metric_total(metrics: dict[str, float], text: str) -> float:
    return sum(value for name, value in metrics.items() if text in name)


def flush(base_url: str) -> None:
    requests.post(
        f"{base_url}/flush_cache",
        json={},
        timeout=60,
    ).raise_for_status()
    time.sleep(0.1)


def append_run_log(path: str, entry: dict) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, sort_keys=True))
        file.write("\n")


def body_tokens(body_index: int, body_size: int) -> list[int]:
    start = 1_000 + body_index * 2_000
    return list(range(start, start + body_size))


def head_tokens(body_index: int, head_index: int, head_size: int) -> list[int]:
    start = 60_000 + body_index * 1_000 + head_index * 200
    return list(range(start, start + head_size))


def register_sources(args: argparse.Namespace) -> dict[int, tuple[list[int], str]]:
    sources = {}
    for body_index, body_size in enumerate(args.body_sizes):
        body = body_tokens(body_index, body_size)
        content_hash = f"epic-matrix-k{args.k}-body{body_size}"
        metadata = {
            "operation": "register",
            "model_fingerprint": "qwen3-0.6b-sm75",
            "cache_dtype": "float16",
            "segments": [
                {
                    "content_hash": content_hash,
                    "target_start": 0,
                    "length": body_size,
                }
            ],
        }
        request(
            args.base_url,
            body + [900 + body_index],
            metadata,
        )
        sources[body_size] = (body, content_hash)
        time.sleep(0.2)
    return sources


def run_epic(args: argparse.Namespace) -> list[dict]:
    sources = register_sources(args)
    points = []
    for body_index, body_size in enumerate(args.body_sizes):
        body, content_hash = sources[body_size]
        for head_index, head_size in enumerate(args.head_sizes):
            head = head_tokens(body_index, head_index, head_size)
            if head:
                request(args.base_url, head)
            metadata = {
                "operation": "reuse",
                "model_fingerprint": "qwen3-0.6b-sm75",
                "cache_dtype": "float16",
                "plugin": "epic",
                "segments": [
                    {
                        "content_hash": content_hash,
                        "target_start": head_size,
                        "length": body_size,
                    }
                ],
            }
            target = head + body + [950 + head_index]
            request(args.base_url, target, metadata)
            before = metric_snapshot(args.base_url)
            rows = [
                request(args.base_url, target, metadata) for _ in range(args.repeats)
            ]
            after = metric_snapshot(args.base_url)
            expected_cached = head_size + body_size
            if any(row["cached_tokens"] != expected_cached for row in rows):
                raise RuntimeError("EPIC did not restore the full reusable prefix")
            fallback_delta = metric_total(
                after,
                "approx_kv_dense_fallback_total",
            ) - metric_total(
                before,
                "approx_kv_dense_fallback_total",
            )
            if fallback_delta:
                raise RuntimeError("EPIC matrix point used dense fallback")
            points.append(
                {
                    "body_tokens": body_size,
                    "head_tokens": head_size,
                    "target_prompt_tokens": len(target),
                    "rows": rows,
                    "target_p50_ms": statistics.median(row["ttft_ms"] for row in rows),
                    "fallback_tokens_delta": fallback_delta,
                    "epic_layers_delta": metric_total(
                        after,
                        "approx_kv_epic_layers_recomputed_total",
                    )
                    - metric_total(
                        before,
                        "approx_kv_epic_layers_recomputed_total",
                    ),
                    "epic_leading_k_tokens_delta": metric_total(
                        after,
                        "approx_kv_epic_leading_k_tokens_total",
                    )
                    - metric_total(
                        before,
                        "approx_kv_epic_leading_k_tokens_total",
                    ),
                }
            )
    return points


def run_dense(args: argparse.Namespace) -> list[dict]:
    points = []
    for body_index, body_size in enumerate(args.body_sizes):
        body = body_tokens(body_index, body_size)
        for head_index, head_size in enumerate(args.head_sizes):
            head = head_tokens(body_index, head_index, head_size)
            target = head + body + [950 + head_index]
            flush(args.base_url)
            if head:
                request(args.base_url, head)
            request(args.base_url, target)
            rows = []
            for _ in range(args.repeats):
                flush(args.base_url)
                if head:
                    request(args.base_url, head)
                rows.append(request(args.base_url, target))
            if any(row["cached_tokens"] != head_size for row in rows):
                raise RuntimeError("dense point did not preserve the exact head")
            points.append(
                {
                    "body_tokens": body_size,
                    "head_tokens": head_size,
                    "target_prompt_tokens": len(target),
                    "rows": rows,
                    "target_p50_ms": statistics.median(row["ttft_ms"] for row in rows),
                }
            )
    return points


def main() -> None:
    args = parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = (
        f"phase4-epic-{args.mode}-k{args.k}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    settings = {
        "mode": args.mode,
        "k": args.k if args.mode == "epic" else None,
        "body_sizes": args.body_sizes,
        "head_sizes": args.head_sizes,
        "repeats_per_setting": args.repeats,
        "global_warmup_passes": 1,
        "per_setting_warmup_passes": 1,
        "scheduler": "S0 LRU",
        "tier": "GPU-only",
        "prefetch": False,
        "base_url": args.base_url,
        "model": args.model,
        "model_revision": args.model_revision,
        "image_digest": args.image_digest,
        "runner_git_sha": args.runner_git_sha,
    }
    append_run_log(
        args.central_log,
        {
            "run_id": run_id,
            "status": "running",
            "timestamp": started_at,
            "settings": settings,
            "output": str(Path(args.output).resolve()),
        },
    )
    try:
        request(args.base_url, list(range(70_000, 70_641)))
        flush(args.base_url)
        points = run_epic(args) if args.mode == "epic" else run_dense(args)
        result = {
            "run_id": run_id,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "settings": settings,
            "points": points,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        append_run_log(
            args.central_log,
            {
                "run_id": run_id,
                "status": "completed",
                "timestamp": result["completed_at"],
                "settings": settings,
                "output": str(output_path.resolve()),
                "result_summary": {
                    "points": len(points),
                    "min_p50_ms": min(point["target_p50_ms"] for point in points),
                    "max_p50_ms": max(point["target_p50_ms"] for point in points),
                },
            },
        )
    except Exception as exc:
        append_run_log(
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
