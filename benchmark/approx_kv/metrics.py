from __future__ import annotations

import math
import re
from typing import Mapping

_SAMPLE_PATTERN = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{[^}]*\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s+\d+)?$"
)

COUNTER_METRICS = (
    "sglang:prompt_tokens_total",
    "sglang:cached_tokens_total",
    "sglang:num_requests_total",
    "sglang:evicted_tokens_total",
    "sglang:load_back_tokens_total",
    "sglang:prefetched_tokens_total",
    "sglang:backuped_tokens_total",
    "sglang:queue_time_seconds_count",
    "sglang:queue_time_seconds_sum",
    "sglang:time_to_first_token_seconds_count",
    "sglang:time_to_first_token_seconds_sum",
)

GAUGE_METRICS = (
    "sglang:max_total_num_tokens",
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:kv_available_tokens",
    "sglang:kv_evictable_tokens",
    "sglang:kv_used_tokens",
)

FALLBACK_METRICS = (
    "sglang:approx_kv_dense_fallback_total",
    "sglang:approx_kv_fallback_total",
)


def parse_prometheus_text(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_PATTERN.match(line)
        if match is None:
            continue
        value = float(match.group("value"))
        if not math.isfinite(value):
            continue
        name = match.group("name")
        totals[name] = totals.get(name, 0.0) + value
    return totals


def counter_delta(
    before: Mapping[str, float],
    after: Mapping[str, float],
    name: str,
) -> float | None:
    if name not in before and name not in after:
        return None
    return after.get(name, 0.0) - before.get(name, 0.0)


def metric_subset(snapshot: Mapping[str, float]) -> dict[str, float]:
    return {
        name: snapshot[name]
        for name in (*COUNTER_METRICS, *GAUGE_METRICS, *FALLBACK_METRICS)
        if name in snapshot
    }


def telemetry_delta(
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> dict[str, object]:
    counters = {}
    for name in COUNTER_METRICS:
        delta = counter_delta(before, after, name)
        if delta is None and name == "sglang:evicted_tokens_total":
            delta = 0.0
        counters[name] = delta
    fallback_name = next(
        (
            name
            for name in FALLBACK_METRICS
            if name in before or name in after
        ),
        None,
    )
    fallback_count = (
        None if fallback_name is None else counter_delta(before, after, fallback_name)
    )
    return {
        "counters": counters,
        "gauges_after": {
            name: after.get(name) for name in GAUGE_METRICS if name in after
        },
        "fallback_metric": fallback_name,
        "fallback_metric_available": fallback_name is not None,
        "dense_fallbacks": fallback_count,
    }


def max_total_num_tokens(snapshot: Mapping[str, float]) -> int:
    value = snapshot.get("sglang:max_total_num_tokens", 0.0)
    if value <= 0:
        raise ValueError("sglang:max_total_num_tokens is unavailable")
    return int(round(value))


def usable_kv_capacity_tokens(snapshot: Mapping[str, float]) -> int:
    available = snapshot.get("sglang:kv_available_tokens")
    evictable = snapshot.get("sglang:kv_evictable_tokens")
    used = snapshot.get("sglang:kv_used_tokens")
    if available is not None and evictable is not None and used is not None:
        if used <= max(16.0, 0.01 * (available + evictable + used)):
            capacity = available + evictable
            if capacity > 0:
                return int(round(capacity))
    return max_total_num_tokens(snapshot)


def idle_pool_invariant(snapshot: Mapping[str, float]) -> dict[str, object]:
    maximum = snapshot.get("sglang:max_total_num_tokens")
    available = snapshot.get("sglang:kv_available_tokens")
    evictable = snapshot.get("sglang:kv_evictable_tokens")
    used = snapshot.get("sglang:kv_used_tokens")
    values_available = all(
        value is not None for value in (maximum, available, evictable, used)
    )
    if not values_available:
        return {
            "metrics_available": False,
            "passed": False,
        }
    accounted = float(available) + float(evictable) + float(used)
    tolerance = max(16.0, 0.01 * float(maximum))
    passed = float(used) <= tolerance and abs(accounted - float(maximum)) <= tolerance
    return {
        "metrics_available": True,
        "passed": passed,
        "max_total_num_tokens": maximum,
        "kv_available_tokens": available,
        "kv_evictable_tokens": evictable,
        "kv_used_tokens": used,
        "accounted_tokens": accounted,
        "tolerance_tokens": tolerance,
    }


def clean_cache_invariant(snapshot: Mapping[str, float]) -> dict[str, object]:
    idle = idle_pool_invariant(snapshot)
    if not idle["metrics_available"]:
        return idle
    maximum = float(idle["max_total_num_tokens"])
    tolerance = float(idle["tolerance_tokens"])
    evictable = float(idle["kv_evictable_tokens"])
    available = float(idle["kv_available_tokens"])
    return {
        **idle,
        "passed": (
            bool(idle["passed"])
            and evictable <= tolerance
            and abs(available - maximum) <= tolerance
        ),
    }


def clean_pool_reset_invariant(
    clean_baseline: Mapping[str, float],
    post_flush: Mapping[str, float],
) -> dict[str, object]:
    names = (
        "sglang:kv_available_tokens",
        "sglang:kv_evictable_tokens",
        "sglang:kv_used_tokens",
    )
    if any(name not in clean_baseline or name not in post_flush for name in names):
        return {
            "metrics_available": False,
            "passed": False,
        }
    maximum = clean_baseline.get(
        "sglang:max_total_num_tokens",
        clean_baseline["sglang:kv_available_tokens"],
    )
    tolerance = max(16.0, 0.01 * maximum)
    deltas = {
        name: post_flush[name] - clean_baseline[name] for name in names
    }
    return {
        "metrics_available": True,
        "passed": all(abs(delta) <= tolerance for delta in deltas.values()),
        "deltas": deltas,
        "tolerance_tokens": tolerance,
        "clean_baseline": {name: clean_baseline[name] for name in names},
        "post_flush": {name: post_flush[name] for name in names},
    }
