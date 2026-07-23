"""Phase 4 R2 CacheBlend eviction-pressure runner.

Mirrors the R1 EPIC/LegoLink pressure runner
(``run_phase4_epic_pressure.py``) so both paths share one methodology,
adapted for two things that are specific to CacheBlend:

1. **Ratio, not k.** CacheBlend's server-visible knob is the HKVD
   selective-repair ratio (``SGLANG_CACHEBLEND_RATIO``, one of
   ``1%/5%/15%/30%``), read once at server startup exactly like EPIC's
   ``k`` -- so, like ``k``, this script cannot change it mid-run; each
   ratio setting requires its own fresh server restart.
2. **Precomputed fresh-KV adapter.** This fork has no generic
   ModelRunner hook that can run a real per-layer forward for an
   arbitrary selected-token subset interleaved with an otherwise-cached
   prefix (see ``cacheblend/recompute.py`` and ``cacheblend/runtime.py``
   module docstrings). The only server-safe substitute is an *explicit
   dense preparation request* that actually computes fresh target-context
   KV under the real target header, which is then registered as a
   second ("fresh") source segment alongside the raw one. This is a real
   extra cost (a full dense forward over the body, once per segment) that
   production would not pay if a true selected-token hook existed, so it
   is timed and reported separately (``fresh_preparation_ms``) rather
   than folded into ``target_ttft_ms`` -- and ``combined_ms`` (target +
   preparation) is reported alongside, never in place of, the target-only
   number.

Genuine non-prefix reuse (not a disguised exact-prefix copy):
the *source* segments (both raw and fresh) are always registered under a
constant header-length preceding context (``source_header`` and
``target_header`` respectively -- two disjoint token-id ranges), while
the *target* request's body sits after ``target_header``. When
``header_tokens > 0`` these are two different preceding contexts, so the
raw copy is a genuine cross-context approximation with real deviation
from what the model would compute if it saw ``target_header`` before the
body (which is exactly what the fresh preparation request measures and
what HKVD scores against). Only at the ``header_tokens == 0`` boundary
setting does "preceding context" degenerate to "nothing" for both source
and target -- that convergence is an intrinsic property of an empty
header, not a disguised prefix match, and the same degenerate case is
already part of R1's accepted header sweep.

Bodies longer than 512 tokens are split into <=512-token segments; each
segment is independently registered (raw *and* fresh) and the target
request's ``segments`` metadata restores them contiguously, exactly like
R1's long-body EPIC segmentation.

Dense-mode fairness: the dense arm issues NO body-chunk priming request
of its own -- only filler traffic plus the real target request. An
earlier version primed the body by sending each chunk as a bare
(headerless) ordinary dense request before the filler loop; that
request commits a second, ~body_tokens-sized exact-cache entry that
shares no prefix with (and is never reused by) the real target request,
which silently inflated dense's true resident footprint above what
``dense_persistent_token_estimate`` declared and biased the
``target_rho`` comparison unfairly tighter for dense than for
cacheblend at the same nominal setting. See
``dense_persistent_token_estimate``'s docstring for the full accounting.
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

# The four ratios this Phase 4 R2 sweep must cover, matching
# `cacheblend.plugin.CACHEBLEND_RATIOS`.
CACHEBLEND_RATIOS: tuple[float, ...] = (0.01, 0.05, 0.15, 0.30)

CACHEBLEND_COUNTER_METRICS: tuple[str, ...] = (
    "sglang:approx_kv_cacheblend_selected_tokens_total",
    "sglang:approx_kv_cacheblend_recomputed_layers_total",
    "sglang:approx_kv_cacheblend_precomputed_total",
)

RAW_HASH_PREFIX = "cacheblend-raw:"
FRESH_HASH_PREFIX = "cacheblend-fresh:"


def _repeat_count(value: str) -> int:
    repeats = int(value)
    if repeats < 2:
        raise argparse.ArgumentTypeError("repeats must be at least 2")
    return repeats


def _ratio(value: str) -> float:
    ratio = float(value)
    if not any(math.isclose(ratio, valid, rel_tol=1e-9) for valid in CACHEBLEND_RATIOS):
        raise argparse.ArgumentTypeError(
            f"ratio must be one of {CACHEBLEND_RATIOS}, got {ratio}"
        )
    return ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30011")
    parser.add_argument("--mode", choices=("dense", "cacheblend"), required=True)
    parser.add_argument("--ratio", type=_ratio, default=0.05)
    parser.add_argument("--target-rho", type=float, required=True)
    parser.add_argument("--body-tokens", type=int, default=512)
    parser.add_argument("--header-tokens", type=int, default=64)
    parser.add_argument("--filler-tokens", type=int, default=736)
    parser.add_argument("--segment-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=_repeat_count, default=4)
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--central-log", required=True)
    return parser.parse_args()


# --------------------------------------------------------------------------
# Pure helpers (unit-tested in
# test/registered/unit/bench/test_cacheblend_pressure_runner.py without any
# network access or GPU).
# --------------------------------------------------------------------------


def segment_chunks(tokens: list[int], segment_tokens: int) -> list[list[int]]:
    """Split ``tokens`` into <=``segment_tokens``-length chunks, in order."""
    if segment_tokens <= 0:
        raise ValueError("segment_tokens must be positive")
    return [
        tokens[start : start + segment_tokens]
        for start in range(0, len(tokens), segment_tokens)
    ]


def build_target_segments(
    chunk_lengths: list[int],
    *,
    header_tokens: int,
    hash_prefix: str,
    content_hash_base: str,
) -> list[dict]:
    """Contiguous target-side restore segments for the reuse request.

    Each chunk's ``content_hash`` must match a segment independently
    registered (at a constant ``target_start=header_tokens``, see
    ``register_source_segments``) under the same base hash -- the
    restore-side ``target_start`` here is the *actual*, cumulative
    position in the target request, which is generally different from
    the position the chunk was registered under. `runtime.py` derives
    the rope rotation from that difference; it is not required (and, for
    long bodies, not even possible) for registration and restore
    positions to coincide.
    """
    segments = []
    cursor = header_tokens
    for index, length in enumerate(chunk_lengths):
        segments.append(
            {
                "content_hash": f"{hash_prefix}{content_hash_base}-chunk{index}",
                "target_start": cursor,
                "length": length,
            }
        )
        cursor += length
    return segments


def expected_selected_tokens(restore_length: int, ratio: float) -> int:
    """Mirror `cacheblend.hkvd.select_hkvd_tokens`'s final rounding rule
    (``max(1, round(total * final_ratio))``) so the runner can sanity
    check real reported telemetry against the expected count."""
    return max(1, round(restore_length * ratio))


def persistent_token_estimate(header_tokens: int, body_tokens: int) -> int:
    """Estimate the device-resident footprint one CacheBlend pressure
    "object" pins beyond ordinary filler pressure.

    Unlike EPIC/R0 (one raw registration), CacheBlend's precomputed
    fresh-KV adapter needs *two* independent source-side device
    footprints (raw + fresh), each ``body_tokens`` long, plus the
    target's own eventual committed KV (``header_tokens + body_tokens +
    1``).

    This function assumes the *only* body-sized contributions are the
    raw registration, the fresh registration, and the target's own
    commit -- see ``dense_persistent_token_estimate`` for the matching
    invariant on the dense-mode side, and why dense must never add a
    fourth, unaccounted body-sized term of its own.
    """
    return 2 * body_tokens + header_tokens + body_tokens + 1


def dense_persistent_token_estimate(header_tokens: int, body_tokens: int) -> int:
    """Dense mode's *only* persistent contribution: the target request's
    own eventual exact-cache commit (``header_tokens + body_tokens +
    1``).

    Dense mode must issue **no** body-chunk priming request of its own
    before this. A previous version of this runner primed the body by
    sending each chunk as a bare (headerless) ordinary dense request
    before the filler loop -- that request is a genuinely different
    token sequence from the real ``target_header + body`` target
    request (different prefix from position 0), so it commits a
    *second*, ~``body_tokens``-sized exact-cache entry that shares
    nothing with, and is never reused by, the target. That second entry
    was real GPU-resident pressure that this estimate never counted,
    silently inflating dense mode's true working set above what
    ``persistent_tokens`` declared and biasing ``target_rho`` unfairly
    tighter for dense than for cacheblend at the same nominal setting.
    Removing the priming (rather than trying to estimate its uncertain
    survival-to-target-time) keeps this estimate exact: nothing but the
    target's own eventual commit is resident besides filler traffic.
    """
    return header_tokens + body_tokens + 1


def compute_filler_count(
    capacity: int,
    target_rho: float,
    persistent_tokens: int,
    filler_tokens: int,
) -> int:
    if filler_tokens <= 0:
        raise ValueError("filler_tokens must be positive")
    target_working_tokens = int(math.ceil(target_rho * capacity))
    return max(
        0,
        math.ceil((target_working_tokens - persistent_tokens) / filler_tokens),
    )


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------


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
    *, operation: str, segments: list[dict], plugin: str | None = None
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


def register_source_segments(
    base_url: str,
    *,
    header: list[int],
    chunks: list[list[int]],
    hash_prefix: str,
    content_hash_base: str,
    sentinel_base: int,
) -> float:
    """Register every chunk (independently, at a constant
    ``target_start=header length``) under ``hash_prefix``. Returns the
    summed client-observed latency of these registration requests -- for
    the fresh (precomputed-adapter) hash this *is* the honest
    "preparation cost" this task requires be reported, not hidden inside
    the target-only TTFT.
    """
    total_ms = 0.0
    for index, chunk in enumerate(chunks):
        result = request(
            base_url,
            header + chunk + [sentinel_base + index],
            build_metadata(
                operation="register",
                segments=[
                    {
                        "content_hash": f"{hash_prefix}{content_hash_base}-chunk{index}",
                        "target_start": len(header),
                        "length": len(chunk),
                    }
                ],
            ),
        )
        total_ms += result["ttft_ms"]
    return total_ms


def _counter_delta(before: dict, after: dict, name: str) -> float | None:
    if name not in before and name not in after:
        return None
    return after.get(name, 0.0) - before.get(name, 0.0)


def run_round(args: argparse.Namespace, round_index: int) -> dict:
    flush(args.base_url, round_index)
    baseline = metric_snapshot(args.base_url)
    capacity = usable_kv_capacity_tokens(baseline)

    if args.mode == "cacheblend":
        persistent_tokens = persistent_token_estimate(
            args.header_tokens, args.body_tokens
        )
    else:
        persistent_tokens = dense_persistent_token_estimate(
            args.header_tokens, args.body_tokens
        )
    filler_count = compute_filler_count(
        capacity, args.target_rho, persistent_tokens, args.filler_tokens
    )

    body = list(range(1_000, 1_000 + args.body_tokens))
    source_header = list(range(50_000, 50_000 + args.header_tokens))
    target_header = list(range(60_000, 60_000 + args.header_tokens))
    content_hash_base = (
        f"cacheblend-pressure-ratio{args.ratio:.2f}-rho{args.target_rho:.3f}-"
        f"round{round_index}"
    )
    chunks = segment_chunks(body, args.segment_tokens)
    chunk_lengths = [len(chunk) for chunk in chunks]

    fresh_preparation_ms = 0.0
    if args.mode == "cacheblend":
        register_source_segments(
            args.base_url,
            header=source_header,
            chunks=chunks,
            hash_prefix=RAW_HASH_PREFIX,
            content_hash_base=content_hash_base,
            sentinel_base=900,
        )
        # The dense preparation request: real target-context KV, computed
        # under the *actual* target header, registered as the "fresh"
        # source. This is the precomputed adapter's real cost.
        fresh_preparation_ms = register_source_segments(
            args.base_url,
            header=target_header,
            chunks=chunks,
            hash_prefix=FRESH_HASH_PREFIX,
            content_hash_base=content_hash_base,
            sentinel_base=910,
        )
    # Dense mode intentionally issues NO body-chunk priming request here.
    # A bare (headerless) per-chunk dense request would commit a second,
    # ~body_tokens-sized exact-cache entry that shares no prefix with --
    # and is never reused by -- the real `target_header + body` target
    # request below, silently inflating dense's true resident footprint
    # above what `dense_persistent_token_estimate` declares (see that
    # function's docstring for the full accounting bug this avoids).

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
            segments=build_target_segments(
                chunk_lengths,
                header_tokens=args.header_tokens,
                hash_prefix=RAW_HASH_PREFIX,
                content_hash_base=content_hash_base,
            ),
            plugin="cacheblend",
        )
        if args.mode == "cacheblend"
        else None
    )
    target = request(args.base_url, target_ids, metadata)
    after_target = metric_snapshot(args.base_url)

    expected_cached = (
        args.header_tokens + args.body_tokens
        if args.mode == "cacheblend"
        else args.header_tokens
    )
    if target["cached_tokens"] != expected_cached:
        raise RuntimeError(
            f"unexpected cached_tokens={target['cached_tokens']}, "
            f"expected={expected_cached}"
        )
    target_delta = telemetry_delta(before_target, after_target)
    if args.mode == "cacheblend" and target_delta["dense_fallbacks"] not in (None, 0):
        raise RuntimeError("CacheBlend pressure target used dense fallback")

    cacheblend_telemetry = None
    if args.mode == "cacheblend":
        selected_delta = _counter_delta(
            before_target,
            after_target,
            "sglang:approx_kv_cacheblend_selected_tokens_total",
        )
        recomputed_layers_delta = _counter_delta(
            before_target,
            after_target,
            "sglang:approx_kv_cacheblend_recomputed_layers_total",
        )
        precomputed_delta = _counter_delta(
            before_target,
            after_target,
            "sglang:approx_kv_cacheblend_precomputed_total",
        )
        expected_selected = expected_selected_tokens(args.body_tokens, args.ratio)
        if not selected_delta:
            raise RuntimeError(
                "CacheBlend pressure target reported zero selected tokens"
            )
        cacheblend_telemetry = {
            "selected_tokens_delta": selected_delta,
            "recomputed_layers_delta": recomputed_layers_delta,
            "precomputed_delta": precomputed_delta,
            "expected_selected_tokens": expected_selected,
        }

    declared_tokens = persistent_tokens + filler_count * args.filler_tokens
    combined_ms = target["ttft_ms"] + fresh_preparation_ms
    return {
        "round_index": round_index,
        "capacity_tokens": capacity,
        "target_rho": args.target_rho,
        "pre_target_rho": declared_tokens / capacity,
        "peak_rho_with_target": (declared_tokens + args.body_tokens) / capacity,
        "filler_count": filler_count,
        "declared_working_tokens": declared_tokens,
        "segment_tokens": args.segment_tokens,
        "segment_count": len(chunks),
        "target": target,
        "fresh_preparation_ms": fresh_preparation_ms,
        "combined_ms": combined_ms,
        "cacheblend_telemetry": cacheblend_telemetry,
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
    if args.segment_tokens <= 0:
        raise ValueError("segment_tokens must be positive")
    prompt_tokens = args.header_tokens + args.body_tokens + 1
    crosses_chunk_boundary = prompt_tokens > 1024

    run_id = (
        f"phase4-cacheblend-pressure-{args.mode}-"
        f"ratio{args.ratio:.2f}-rho{args.target_rho:.3f}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    settings = {
        "mode": args.mode,
        "ratio": args.ratio if args.mode == "cacheblend" else None,
        "target_rho": args.target_rho,
        "body_tokens": args.body_tokens,
        "header_tokens": args.header_tokens,
        "filler_tokens": args.filler_tokens,
        "segment_tokens": args.segment_tokens,
        "target_prompt_tokens": prompt_tokens,
        "crosses_1024_token_chunk_boundary": crosses_chunk_boundary,
        "global_warmup_passes": 1,
        "per_setting_warmup_passes": 1,
        "formal_repeats": args.repeats,
        "mem_fraction_static": 0.35,
        "scheduler": "S0 LRU",
        "tier": "GPU-only",
        "prefetch": False,
        "model": args.model,
        "model_revision": args.model_revision,
        "runner_git_sha": args.runner_git_sha,
        "image_digest": args.image_digest,
        "known_limitations": [
            "The generic online ModelRunner selected-token forward hook "
            "remains unavailable; cacheblend mode uses an explicit dense "
            "preparation request (fresh_preparation_ms) as a server-safe "
            "precomputed-adapter substitute.",
            "Source/target contexts are genuinely non-prefix (distinct "
            "header token ranges) whenever header_tokens > 0; at "
            "header_tokens == 0 both contexts are empty, which is an "
            "intrinsic boundary case, not a disguised exact-prefix copy.",
        ],
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
        target_values = [row["target"]["ttft_ms"] for row in rows]
        prep_values = [row["fresh_preparation_ms"] for row in rows]
        combined_values = [row["combined_ms"] for row in rows]
        result = {
            "schema_version": 1,
            "run_id": run_id,
            "settings": settings,
            "warmup": warmup,
            "rows": rows,
            "target_p50_ms": statistics.median(target_values),
            "fresh_preparation_p50_ms": statistics.median(prep_values),
            "combined_p50_ms": statistics.median(combined_values),
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
                    "fresh_preparation_p50_ms": result["fresh_preparation_p50_ms"],
                    "combined_p50_ms": result["combined_p50_ms"],
                    "eviction_observed": result["eviction_observed_in_formal_runs"],
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
