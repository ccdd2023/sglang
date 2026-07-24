#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import aiohttp

from benchmark.approx_kv.metrics import (
    clean_cache_invariant,
    clean_pool_reset_invariant,
    idle_pool_invariant,
    metric_subset,
    parse_prometheus_text,
    telemetry_delta,
)
from benchmark.approx_kv.workloads import (
    CacheObject,
    TraceInvocation,
    build_messages,
)

CustomParamsFactory = Callable[
    [CacheObject, TraceInvocation],
    Mapping[str, Any] | None,
]


@dataclass(frozen=True)
class RequestResult:
    sample_kind: str
    repeat: int
    step: int
    phase: str
    object_id: str
    role: str
    occurrence: int
    next_use_step: int | None
    next_use_distance: int | None
    intervening_unique_prefix_tokens: int | None
    expected_reusable_prefix_tokens: int
    cached_prefix_fraction: float
    is_probe: bool
    first_sse_ms: float
    ttft_ms: float
    elapsed_ms: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    output_chunks: int
    saw_done: bool
    usage_present: bool
    status: int
    error: str
    metrics_after: dict[str, float]


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


async def fetch_metrics(
    session: aiohttp.ClientSession,
    base_url: str,
) -> dict[str, float]:
    async with session.get(f"{base_url}/metrics") as response:
        response.raise_for_status()
        return parse_prometheus_text(await response.text())


def request_succeeded(result: RequestResult) -> bool:
    return (
        result.status == 200
        and not result.error
        and result.first_sse_ms >= 0
        and result.output_chunks > 0
        and result.saw_done
        and result.usage_present
        and result.completion_tokens == 1
    )


async def flush_cache(
    session: aiohttp.ClientSession,
    base_url: str,
) -> str:
    async with session.post(
        f"{base_url}/flush_cache",
        params={"timeout": 30},
    ) as response:
        response.raise_for_status()
        return await response.text()


async def refresh_health(
    session: aiohttp.ClientSession,
    base_url: str,
) -> None:
    async with session.get(f"{base_url}/health_generate") as response:
        response.raise_for_status()
        await response.read()


async def send_request(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    model: str,
    cache_object: CacheObject,
    invocation: TraceInvocation,
    sample_kind: str,
    repeat: int,
    cache_salt: str,
    is_probe: bool,
    scrape_metrics: bool,
    custom_params: Mapping[str, Any] | None = None,
) -> RequestResult:
    suffix = (
        f"{invocation.suffix};"
        f"sample={sample_kind};"
        f"repeat={repeat:03d}"
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": build_messages(
            cache_object,
            suffix,
            cache_salt=cache_salt,
        ),
        "max_tokens": 1,
        "temperature": 0,
        "top_p": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if custom_params:
        payload["custom_params"] = dict(custom_params)
    start = time.perf_counter()
    first_sse_time: float | None = None
    first_token_time: float | None = None
    prompt_tokens = 0
    cached_tokens = 0
    completion_tokens = 0
    output_chunks = 0
    saw_done = False
    usage_present = False
    status = 0
    error = ""

    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
        ) as response:
            status = response.status
            if status != 200:
                error = (await response.text())[:1000]
            else:
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        saw_done = True
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if first_sse_time is None:
                        first_sse_time = time.perf_counter()
                    usage = chunk.get("usage") or {}
                    if usage:
                        usage_present = True
                    prompt_tokens = int(
                        usage.get("prompt_tokens", prompt_tokens)
                    )
                    completion_tokens = int(
                        usage.get("completion_tokens", completion_tokens)
                    )
                    details = usage.get("prompt_tokens_details") or {}
                    cached_tokens = int(
                        details.get("cached_tokens", cached_tokens)
                    )
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        reasoning = delta.get("reasoning_content")
                        if content or reasoning:
                            output_chunks += 1
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    end = time.perf_counter()
    metrics_after = {}
    if scrape_metrics:
        try:
            metrics_after = metric_subset(await fetch_metrics(session, base_url))
        except Exception as exc:
            metrics_error = f"metrics {type(exc).__name__}: {exc}"
            error = f"{error}; {metrics_error}" if error else metrics_error
    cached_prefix_fraction = min(
        1.0,
        cached_tokens / max(1, cache_object.reusable_prefix_tokens),
    )
    return RequestResult(
        sample_kind=sample_kind,
        repeat=repeat,
        step=invocation.step,
        phase=invocation.phase,
        object_id=invocation.object_id,
        role=invocation.role,
        occurrence=invocation.occurrence,
        next_use_step=invocation.next_use_step,
        next_use_distance=invocation.next_use_distance,
        intervening_unique_prefix_tokens=(
            invocation.intervening_unique_prefix_tokens
        ),
        expected_reusable_prefix_tokens=cache_object.reusable_prefix_tokens,
        cached_prefix_fraction=cached_prefix_fraction,
        is_probe=is_probe,
        first_sse_ms=(
            -1.0
            if first_sse_time is None
            else (first_sse_time - start) * 1000
        ),
        ttft_ms=((first_token_time or end) - start) * 1000,
        elapsed_ms=(end - start) * 1000,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        output_chunks=output_chunks,
        saw_done=saw_done,
        usage_present=usage_present,
        status=status,
        error=error,
        metrics_after=metrics_after,
    )


async def execute_trace(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    model: str,
    objects: Mapping[str, CacheObject],
    trace: Sequence[TraceInvocation],
    sample_kind: str,
    repeat: int,
    cache_salt: str,
    probe_object_ids: set[str],
    scrape_metrics: bool,
    custom_params_factory: CustomParamsFactory | None = None,
) -> list[RequestResult]:
    results = []
    for invocation in trace:
        result = await send_request(
            session,
            base_url=base_url,
            model=model,
            cache_object=objects[invocation.object_id],
            invocation=invocation,
            sample_kind=sample_kind,
            repeat=repeat,
            cache_salt=cache_salt,
            is_probe=(
                invocation.object_id in probe_object_ids
                and invocation.phase != "fill"
            ),
            scrape_metrics=scrape_metrics,
            custom_params=(
                None
                if custom_params_factory is None
                else custom_params_factory(
                    objects[invocation.object_id],
                    invocation,
                )
            ),
        )
        results.append(result)
        if not request_succeeded(result):
            break
    return results


async def wait_for_idle_metrics(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    timeout_s: float = 30,
) -> dict[str, float]:
    deadline = time.monotonic() + timeout_s
    latest: dict[str, float] = {}
    while time.monotonic() < deadline:
        try:
            latest = await fetch_metrics(session, base_url)
        except Exception:
            await asyncio.sleep(0.25)
            continue
        running = latest.get("sglang:num_running_reqs", 0.0)
        queued = latest.get("sglang:num_queue_reqs", 0.0)
        used = latest.get("sglang:kv_used_tokens", 0.0)
        if running == 0 and queued == 0 and used == 0:
            return latest
        await asyncio.sleep(0.25)
    return latest


async def benchmark_server(
    *,
    base_url: str,
    model: str,
    objects: Sequence[CacheObject],
    trace: Sequence[TraceInvocation],
    warmup_repeats: int,
    measured_repeats: int,
    request_timeout_s: float,
    probe_object_ids: set[str],
    metrics_scrape_mode: str,
    custom_params_factory: CustomParamsFactory | None = None,
    reset_between_measured_repeats: bool = False,
) -> dict[str, Any]:
    if metrics_scrape_mode not in ("boundary", "per_request"):
        raise ValueError("metrics_scrape_mode must be boundary or per_request")
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    object_map = {cache_object.object_id: cache_object for cache_object in objects}
    warmup_results: list[RequestResult] = []
    measured_results: list[RequestResult] = []
    lifecycle_error = ""
    warmup_metrics: dict[str, float] = {}
    clean_baseline: dict[str, float] = {}
    measured_metrics_after: dict[str, float] = {}
    post_flush_metrics: dict[str, float] = {}

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for repeat in range(warmup_repeats):
            warmup_results.extend(
                await execute_trace(
                    session,
                    base_url=base_url,
                    model=model,
                    objects=object_map,
                    trace=trace,
                    sample_kind="warmup",
                    repeat=repeat,
                    cache_salt="warmup",
                    probe_object_ids=probe_object_ids,
                    scrape_metrics=(
                        metrics_scrape_mode == "per_request"
                    ),
                    custom_params_factory=custom_params_factory,
                )
            )
            if warmup_results and not request_succeeded(warmup_results[-1]):
                break
        await refresh_health(session, base_url)
        warmup_metrics = await wait_for_idle_metrics(
            session,
            base_url=base_url,
        )
        if all(request_succeeded(result) for result in warmup_results):
            try:
                await flush_cache(session, base_url)
                await refresh_health(session, base_url)
                clean_baseline = await wait_for_idle_metrics(
                    session,
                    base_url=base_url,
                )
                for repeat in range(measured_repeats):
                    if repeat > 0 and reset_between_measured_repeats:
                        await flush_cache(session, base_url)
                        await refresh_health(session, base_url)
                        repeat_baseline = await wait_for_idle_metrics(
                            session,
                            base_url=base_url,
                        )
                        if not clean_cache_invariant(repeat_baseline)["passed"]:
                            raise RuntimeError(
                                "cache did not return to a clean state between "
                                f"measured repeats: {repeat_baseline}"
                            )
                    measured_results.extend(
                        await execute_trace(
                            session,
                            base_url=base_url,
                            model=model,
                            objects=object_map,
                            trace=trace,
                            sample_kind="measured",
                            repeat=repeat,
                            cache_salt="measured",
                            probe_object_ids=probe_object_ids,
                            scrape_metrics=(metrics_scrape_mode == "per_request"),
                            custom_params_factory=custom_params_factory,
                        )
                    )
                    if measured_results and not request_succeeded(
                        measured_results[-1]
                    ):
                        break
                await refresh_health(session, base_url)
                measured_metrics_after = await wait_for_idle_metrics(
                    session,
                    base_url=base_url,
                )
                await flush_cache(session, base_url)
                await refresh_health(session, base_url)
                post_flush_metrics = await wait_for_idle_metrics(
                    session,
                    base_url=base_url,
                )
            except Exception as exc:
                lifecycle_error = f"{type(exc).__name__}: {exc}"

    successful = [
        result
        for result in measured_results
        if request_succeeded(result)
    ]
    ttfts = [result.ttft_ms for result in successful]
    first_sse = [result.first_sse_ms for result in successful]
    probe_results = [result for result in successful if result.is_probe]
    probe_ttfts = [result.ttft_ms for result in probe_results]
    cached_tokens = sum(result.cached_tokens for result in measured_results)
    prompt_tokens = sum(result.prompt_tokens for result in measured_results)
    return {
        "warmup_results": [asdict(result) for result in warmup_results],
        "results": [asdict(result) for result in measured_results],
        "warmup_metrics": metric_subset(warmup_metrics),
        "clean_baseline_metrics": metric_subset(clean_baseline),
        "metrics_after": metric_subset(measured_metrics_after),
        "post_flush_metrics": metric_subset(post_flush_metrics),
        "telemetry_delta": telemetry_delta(
            clean_baseline,
            measured_metrics_after,
        ),
        "clean_baseline_invariant": clean_cache_invariant(clean_baseline),
        "idle_pool_invariant": idle_pool_invariant(measured_metrics_after),
        "reset_invariant": clean_pool_reset_invariant(
            clean_baseline,
            post_flush_metrics,
        ),
        "lifecycle_error": lifecycle_error,
        "summary": {
            "expected_requests": len(trace) * measured_repeats,
            "requests": len(measured_results),
            "successful": len(successful),
            "completion_rate": (
                len(successful) / (len(trace) * measured_repeats)
                if trace and measured_repeats
                else 0.0
            ),
            "first_sse_p50_ms": (
                statistics.median(first_sse) if first_sse else 0.0
            ),
            "ttft_p50_ms": statistics.median(ttfts) if ttfts else 0.0,
            "ttft_p95_ms": percentile(ttfts, 0.95),
            "ttft_mean_ms": statistics.mean(ttfts) if ttfts else 0.0,
            "probe_requests": len(probe_results),
            "probe_ttft_p50_ms": (
                statistics.median(probe_ttfts) if probe_ttfts else 0.0
            ),
            "probe_ttft_p95_ms": percentile(probe_ttfts, 0.95),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "cache_hit_token_rate": (
                cached_tokens / prompt_tokens if prompt_tokens else 0.0
            ),
            "metrics_scrape_mode": metrics_scrape_mode,
            "lifecycle_error": lifecycle_error,
        },
    }
