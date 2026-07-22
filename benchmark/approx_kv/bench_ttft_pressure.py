#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp

from benchmark.approx_kv.workloads import (
    PressurePoint,
    TraceKind,
    build_trace,
    deterministic_code,
    estimate_active_reusable_tokens,
    next_use_distance,
)


@dataclass(frozen=True)
class RequestResult:
    repeat: int
    step: int
    role: str
    next_use_step: int | None
    next_use_distance: int | None
    ttft_ms: float
    elapsed_ms: float
    prompt_tokens: int
    cached_tokens: int
    status: int
    error: str = ""


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def role_prompt(role: str, role_prefix_blocks: int) -> str:
    return deterministic_code(f"role:{role}", role_prefix_blocks)


async def send_request(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    model: str,
    role: str,
    role_prefix_blocks: int,
    code: str,
    suffix: str,
    max_new_tokens: int,
) -> tuple[float, float, int, int, int, str]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": role_prompt(role, role_prefix_blocks),
            },
            {
                "role": "user",
                "content": f"{code}\n\n{suffix}",
            },
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first_token_s: float | None = None
    prompt_tokens = 0
    cached_tokens = 0

    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
        ) as response:
            status = response.status
            if status != 200:
                body = await response.text()
                elapsed = (time.perf_counter() - start) * 1000
                return (
                    elapsed,
                    elapsed,
                    0,
                    0,
                    status,
                    body[:500],
                )

            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                details = usage.get("prompt_tokens_details") or {}
                cached_tokens = int(details.get("cached_tokens", cached_tokens))
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content") and first_token_s is None:
                        first_token_s = time.perf_counter()

            end = time.perf_counter()
            ttft_ms = ((first_token_s or end) - start) * 1000
            return (
                ttft_ms,
                (end - start) * 1000,
                prompt_tokens,
                cached_tokens,
                status,
                "",
            )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return (
            elapsed,
            elapsed,
            0,
            0,
            0,
            f"{type(exc).__name__}: {exc}",
        )


async def run(args: argparse.Namespace) -> list[RequestResult]:
    trace = build_trace(TraceKind(args.trace), rounds=args.rounds)
    code = deterministic_code(args.seed, args.code_blocks)
    results: list[RequestResult] = []
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_s)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for repeat in range(args.warmup_repeats + args.measured_repeats):
            measured = repeat >= args.warmup_repeats
            measured_repeat = repeat - args.warmup_repeats
            for invocation in trace:
                values = await send_request(
                    session,
                    base_url=args.base_url.rstrip("/"),
                    model=args.model,
                    role=invocation.role,
                    role_prefix_blocks=args.role_prefix_blocks,
                    code=code,
                    suffix=invocation.suffix,
                    max_new_tokens=args.max_new_tokens,
                )
                if measured:
                    results.append(
                        RequestResult(
                            repeat=measured_repeat,
                            step=invocation.step,
                            role=invocation.role,
                            next_use_step=invocation.next_use_step,
                            next_use_distance=next_use_distance(invocation),
                            ttft_ms=values[0],
                            elapsed_ms=values[1],
                            prompt_tokens=values[2],
                            cached_tokens=values[3],
                            status=values[4],
                            error=values[5],
                        )
                    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="default")
    parser.add_argument(
        "--trace",
        choices=[kind.value for kind in TraceKind],
        default=TraceKind.RETRY.value,
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", default="approx-kv")
    parser.add_argument("--code-blocks", type=int, default=256)
    parser.add_argument("--role-prefix-blocks", type=int, default=16)
    parser.add_argument("--code-tokens", type=int, default=8192)
    parser.add_argument("--role-prefix-tokens", type=int, default=512)
    parser.add_argument("--resident-variants", type=int, default=5)
    parser.add_argument("--gpu-kv-capacity-tokens", type=int, required=True)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--measured-repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--request-timeout-s", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_new_tokens != 1:
        raise ValueError("TTFT benchmark requires --max-new-tokens=1")
    active_tokens = estimate_active_reusable_tokens(
        code_tokens=args.code_tokens,
        role_prefix_tokens=args.role_prefix_tokens,
        resident_variants=args.resident_variants,
    )
    pressure = PressurePoint(
        active_reusable_tokens=active_tokens,
        gpu_kv_capacity_tokens=args.gpu_kv_capacity_tokens,
    )
    results = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            **vars(args),
            "output": str(args.output),
            "active_reusable_tokens": active_tokens,
            "pressure_ratio": pressure.ratio,
        },
        "results": [asdict(result) for result in results],
        "summary": {
            "requests": len(results),
            "successful": sum(result.status == 200 for result in results),
            "ttft_p50_ms": statistics.median(result.ttft_ms for result in results)
            if results
            else 0.0,
            "ttft_p95_ms": percentile(
                [result.ttft_ms for result in results],
                0.95,
            ),
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], sort_keys=True))
    return int(payload["summary"]["successful"] != len(results))


if __name__ == "__main__":
    raise SystemExit(main())
