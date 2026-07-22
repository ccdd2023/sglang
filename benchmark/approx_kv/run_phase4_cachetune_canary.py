#!/usr/bin/env python3
from __future__ import annotations

"""Phase 4 R5 CacheTune SM75 canary runner.

Connects to a live SGLang server that already has the CacheTune recovery
plugin registered (``SGLANG_APPROX_KV_CACHETUNE=1``, plus a deployment-wide
hardware measurement from ``SGLANG_CACHETUNE_T_C_MS`` / ``_T_I_MS`` /
``_T_O_MS`` -- see ``cachetune/plugin.py``) and issues real, blocking HTTP
``/v1/chat/completions`` requests to exercise the genuine, non-simulated
CacheTune request path end to end:

1. Run the dense baseline (no ``approx_kv`` metadata at all) *entirely* to
   completion, flushing the exact-match radix cache before every dense
   request. This must happen before anything is registered, and before any
   ``reuse`` request targets the same messages -- otherwise a real dense
   forward's exact-cache entry could silently serve a later "reuse"
   request via the scheduler's own prefix match, before CacheTune's
   plugin dispatch ever runs (see the ``measurement hygiene`` note below).
2. Register a "raw" source-context segment (``cachetune-raw:<artifact>``)
   from one real dense forward (the "source" branch).
3. Register a "fresh" precomputed target-context segment
   (``cachetune-fresh:<artifact>``) from a *separate* real dense forward
   over the actual target branch -- this is the genuine, separately-timed
   "preparation" cost the precomputed fresh-KV adapter needs on this fork,
   since no ModelRunner hook exists here for a real inline per-layer
   forward on an arbitrary token subset (see
   ``cachetune/precomputed.py``/``cachetune/runtime.py``).
4. Issue ``plugin=cachetune`` reuse requests referencing the raw segment
   and record the server's real ``sglang:approx_kv_cachetune_*``
   Prometheus counter deltas plus each request's genuine client-observed
   wall-clock latency (``max_tokens=1``, so request latency approximates
   TTFT, matching every other benchmark script in this project).
5. Additionally exercises a handful of *different* real restore lengths
   against the *same* running controller/server (no restart) to prove
   real, per-request deterministic ratio re-quantization -- CacheTune's
   controller re-derives an executable token count from the *exact*
   context length of every request, not a fixed sweep parameter (see
   ``hardware_profile.quantize_ratio``).
6. Measurement hygiene (mandatory, applies to every setting above and to
   every length-sweep point): each setting first runs one *discarded*
   warmup pass, then ``--repeats`` (``>= 2``) formal repeats, recording
   every repeat's raw wall-clock sample -- never just a derived median --
   and cross-checking Prometheus telemetry deltas using only the formal
   repeat count (the warmup's own telemetry effect is already baked into
   the "before" snapshot, which is always taken *after* warmup completes).
   Dense additionally flushes the exact radix cache before its warmup,
   before every formal repeat, and once more right before any raw/fresh
   registration begins. Register/reuse requests need no such flush:
   ``schedule_batch.Req.skip_radix_cache_insert`` is forced True whenever
   ``approx_kv_metadata`` is present (register *or* reuse), so they can
   never populate the exact radix tree themselves -- only dense requests
   (which carry no ``approx_kv`` metadata) can, which is why only dense
   needs inter-repeat flushing. Flushing between formal register/reuse
   repeats would be actively wrong: ``/flush_cache`` also resets
   ``ApproxKVManager``'s segment store (see
   ``mem_cache/approx_kv/manager.py``), which would delete the very
   "raw"/"fresh" segments those repeats depend on. Every invocation
   writes JSONL lifecycle records (``running`` / ``completed`` /
   ``failed``) to ``--central-log``, carrying the full settings, the
   image/model/git identity, the warmup/repeat counts, the output path,
   and (on success) a short result summary -- see ``append_run_log``.

Every reported number is a genuine client-observed duration or a real
server-reported Prometheus counter delta; nothing is fabricated, and the
"fresh" preparation cost is always reported and folded into
``combined_p50_ms`` -- never excluded before claiming an end-to-end
result (see ``research/cacheblend``'s and ``research/epic-legolink``'s
own Phase 4 results for the established honest-reporting precedent this
script follows; the central-log/warmup/repeat discipline itself mirrors
``research/epic-legolink``'s ``run_phase4_epic_inrequest_matrix.py``).

The controller's own per-request decision (ratio, selected tokens,
recomputed layers, precomputed-adapter usage) is *not* exposed in the
``/v1/chat/completions`` JSON response body -- it is only observable in
aggregate via the ``/metrics`` Prometheus endpoint. This script always
cross-checks the *observed* telemetry deltas against an independently
computed expectation, using this same package's real
``roofline_ratio``/``quantize_ratio``/``predict_ttft_ms`` functions
(imported directly, not reimplemented) applied to the exact
``t_c``/``t_i``/``t_o`` measurement and mode the operator also used to
start the server -- this is real white-box cross-validation of the
running server's behaviour, not a tautology, since it would catch any
mismatch between what the server actually does and what the controller
contract promises.
"""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.approx_kv.metrics import idle_pool_invariant, parse_prometheus_text
from benchmark.approx_kv.workloads import (
    CacheObject,
    build_messages,
    build_object_catalog,
    common_prefix_token_ids,
    tokenize_messages,
)
from sglang.srt.mem_cache.cachetune.hardware_profile import (
    CacheTuneMode,
    HardwareMeasurement,
    RatioBounds,
    predict_ttft_ms,
    quantize_ratio,
    roofline_ratio,
)

CACHE_SALT = "phase4-r5-cachetune"

# Every setting (dense, the main CacheTune point, and each length-sweep
# point) runs exactly this many *discarded* passes before the formal
# repeats begin. This is a fixed measurement-protocol constant, not a CLI
# knob, so every canary result is comparable under the same discipline.
WARMUP_PASSES_PER_SETTING = 1


def _repeat_count(value: str) -> int:
    """argparse ``type=`` validator: reject ``--repeats`` below 2 up front.

    A single formal repeat cannot be distinguished from measurement noise.
    The entire point of separating "formal repeats" from the discarded
    warmup pass is to give ``statistics.median`` more than one real
    sample, so this is enforced as a hard CLI-level error rather than a
    silently-clamped default.
    """
    repeats = int(value)
    if repeats < 2:
        raise argparse.ArgumentTypeError(
            f"--repeats must be >= 2, got {repeats} (need at least two "
            "formal measurements to compute a meaningful median and to "
            "distinguish real signal from single-sample noise)"
        )
    return repeats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--cache-dtype", default="fp16")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[mode.value for mode in CacheTuneMode],
        help="Must equal the running server's SGLANG_CACHETUNE_MODE.",
    )
    parser.add_argument(
        "--t-c-ms",
        type=float,
        required=True,
        help="Must equal the running server's SGLANG_CACHETUNE_T_C_MS.",
    )
    parser.add_argument(
        "--t-i-ms",
        type=float,
        required=True,
        help="Must equal the running server's SGLANG_CACHETUNE_T_I_MS.",
    )
    parser.add_argument(
        "--t-o-ms",
        type=float,
        required=True,
        help="Must equal the running server's SGLANG_CACHETUNE_T_O_MS.",
    )
    parser.add_argument(
        "--first-recompute-layer",
        type=int,
        default=1,
        help="Must equal the running server's SGLANG_CACHETUNE_FIRST_RECOMPUTE_LAYER.",
    )
    parser.add_argument("--target-prefix-tokens", type=int, default=256)
    parser.add_argument(
        "--length-sweep",
        default="128,512",
        help="Comma-separated additional reusable-prefix token targets used "
        "to prove real per-length deterministic re-quantization within the "
        "same running server/controller (no server restart).",
    )
    parser.add_argument("--repeats", type=_repeat_count, default=4)
    parser.add_argument("--runner-git-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--central-log",
        type=Path,
        required=True,
        help="Shared JSONL log every test/benchmark run must append to: "
        "one 'running' record at start, then one 'completed' or 'failed' "
        "record at the end (see append_run_log).",
    )
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
    """POST with an empty body.

    Matches the exact idiom already established by this directory's
    ``run_phase3_canary.py`` and ``run_phase2_matrix.py`` for
    ``/flush_cache`` -- any non-2xx response raises ``urllib.error.HTTPError``
    unhandled, which is intentional: a silently-ignored flush failure
    would silently reintroduce the exact-cache pollution this script
    exists to prevent.
    """
    request = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def flush_exact_radix_cache(base_url: str) -> str:
    """Flush the server's exact-match radix cache.

    Only *dense* baseline requests (no ``approx_kv`` metadata) are ever
    inserted into the exact radix tree: ``schedule_batch.Req``'s
    ``skip_radix_cache_insert`` is forced True whenever
    ``approx_kv_metadata`` is present -- i.e. for *every* register or
    reuse request (see ``python/sglang/srt/managers/schedule_batch.py``).
    That means register/reuse requests can never pollute the exact cache
    themselves, but a real dense forward over the same token sequence a
    later ``reuse`` request targets absolutely can: it would let the
    scheduler's own prefix match resolve the entire prompt before
    CacheTune's plugin dispatch ever runs, silently skipping the whole
    approximate-repair path. Call this before every dense repeat (each is
    a real, cache-writing forward pass) and once more before any
    raw/fresh registration begins. Do **not** call this between formal
    register+reuse repeats: doing so would also invoke
    ``ApproxKVManager.reset()`` (see
    ``python/sglang/srt/mem_cache/approx_kv/manager.py``), which wipes
    the very "raw"/"fresh" segments those repeats depend on.
    """
    response = post_empty(f"{base_url}/flush_cache?timeout=30")
    time.sleep(0.1)
    return response


def calibrate_length_point(
    tokenizer: Any,
    target_prefix_tokens: int,
    branch_tag: str,
) -> tuple[CacheObject, list[dict[str, str]], list[dict[str, str]], int]:
    cache_object = build_object_catalog(
        tokenizer,
        object_count=4,
        target_sizes=(target_prefix_tokens,),
    )[0]
    source_messages = build_messages(
        cache_object,
        f"cachetune-source-branch-{branch_tag}",
        cache_salt=CACHE_SALT,
    )
    target_messages = build_messages(
        cache_object,
        f"cachetune-target-branch-{branch_tag}-with-different-suffix",
        cache_salt=CACHE_SALT,
    )
    source_ids = tokenize_messages(tokenizer, source_messages)
    target_ids = tokenize_messages(tokenizer, target_messages)
    reusable_tokens = len(common_prefix_token_ids(source_ids, target_ids))
    if reusable_tokens <= 0 or reusable_tokens >= min(len(source_ids), len(target_ids)):
        raise RuntimeError(
            f"canary prompts for target={target_prefix_tokens} do not have a "
            "partial stable prefix"
        )
    return cache_object, source_messages, target_messages, reusable_tokens


def register_payload(
    *,
    messages: list[dict[str, str]],
    content_hash: str,
    reusable_tokens: int,
    model_fingerprint: str,
    cache_dtype: str,
) -> dict:
    return {
        "model": "default",
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0,
        "custom_params": {
            "approx_kv": {
                "operation": "register",
                "model_fingerprint": model_fingerprint,
                "cache_dtype": cache_dtype,
                "segments": [
                    {
                        "content_hash": content_hash,
                        "target_start": 0,
                        "length": reusable_tokens,
                    }
                ],
            }
        },
    }


def reuse_payload(
    *,
    messages: list[dict[str, str]],
    raw_content_hash: str,
    reusable_tokens: int,
    model_fingerprint: str,
    cache_dtype: str,
) -> dict:
    return {
        "model": "default",
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0,
        "custom_params": {
            "approx_kv": {
                "operation": "reuse",
                "plugin": "cachetune",
                "model_fingerprint": model_fingerprint,
                "cache_dtype": cache_dtype,
                "segments": [
                    {
                        "content_hash": raw_content_hash,
                        "target_start": 0,
                        "length": reusable_tokens,
                    }
                ],
            }
        },
    }


def dense_payload(messages: list[dict[str, str]]) -> dict:
    return {
        "model": "default",
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0,
    }


def timed_post(base_url: str, payload: dict) -> tuple[dict, float]:
    start = time.perf_counter()
    response = post_json(f"{base_url}/v1/chat/completions", payload)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return response, elapsed_ms


def require_finished_by_length(response: dict, label: str) -> None:
    finish_reason = response["choices"][0]["finish_reason"]
    if finish_reason != "length":
        raise RuntimeError(f"{label} request did not finish by length: {finish_reason}")


def metric_snapshot(base_url: str) -> dict[str, float]:
    return parse_prometheus_text(fetch_text(f"{base_url}/metrics"))


def metric_delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def expected_repair_totals(
    *,
    repair_tokens_per_call: int,
    recomputed_layers_per_call: int,
    repeats: int,
) -> dict[str, Any]:
    """Pure computation of the telemetry totals CacheTune's Prometheus
    counters must show after exactly ``repeats`` *formal* reuse calls.

    Never includes the discarded warmup pass: its telemetry effect is
    already baked into the "before" snapshot, which this runner always
    takes only after warmup has completed (see ``run_canary``), so
    ``repeats`` here must be the formal-repeat count alone.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if repair_tokens_per_call < 0:
        raise ValueError(
            f"repair_tokens_per_call must be >= 0, got {repair_tokens_per_call}"
        )
    if recomputed_layers_per_call < 0:
        raise ValueError(
            "recomputed_layers_per_call must be >= 0, got "
            f"{recomputed_layers_per_call}"
        )
    expect_precomputed_adapter = repair_tokens_per_call > 0
    return {
        "expect_precomputed_adapter": expect_precomputed_adapter,
        "expected_selected_tokens_total": repair_tokens_per_call * repeats,
        "expected_recomputed_layers_total": (
            recomputed_layers_per_call * repeats if expect_precomputed_adapter else 0
        ),
        "expected_precomputed_total": repeats if expect_precomputed_adapter else 0,
    }


def append_run_log(path: Path, entry: dict[str, Any]) -> None:
    """Append one JSONL lifecycle record to the shared central log.

    Mirrors the schema already established by
    ``research/epic-legolink``'s ``run_phase4_epic_inrequest_matrix.py``
    (``run_id`` / ``status`` / ``timestamp`` / ``settings`` / ``output``,
    plus ``result_summary`` on success or ``error`` on failure) so a
    human or tool reading multiple sibling canaries' central logs sees
    one uniform shape. The file is opened in append mode and never
    truncated: many independent runs share the same ``--central-log``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True))
        handle.write("\n")


def build_settings(args: argparse.Namespace) -> dict[str, Any]:
    """JSON-safe snapshot of every setting relevant to reproducing a run.

    Shared verbatim by the ``running`` / ``completed`` / ``failed``
    central-log records for one invocation.
    """
    return {
        "base_url": args.base_url,
        "model": args.model,
        "model_revision": args.model_revision,
        "model_fingerprint": args.model_fingerprint,
        "cache_dtype": args.cache_dtype,
        "mode": args.mode,
        "t_c_ms": args.t_c_ms,
        "t_i_ms": args.t_i_ms,
        "t_o_ms": args.t_o_ms,
        "first_recompute_layer": args.first_recompute_layer,
        "target_prefix_tokens": args.target_prefix_tokens,
        "length_sweep": args.length_sweep,
        "repeats_per_setting": args.repeats,
        "warmup_passes_per_setting": WARMUP_PASSES_PER_SETTING,
        "runner_git_sha": args.runner_git_sha,
        "image_digest": args.image_digest,
        "scheduler": "S0 LRU",
        "tier": "GPU-only",
        "prefetch": False,
        "accuracy_metric": False,
    }


def run_canary(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full canary against a live server and return the
    result payload (also written to ``args.output`` and printed)."""
    mode = CacheTuneMode(args.mode)
    bounds = RatioBounds.for_mode(mode)
    measurement = HardwareMeasurement(
        t_c_ms=args.t_c_ms,
        t_i_ms=args.t_i_ms,
        t_o_ms=args.t_o_ms,
    )
    r0 = roofline_ratio(measurement)

    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=True,
    )
    model_config = AutoConfig.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=True,
    )
    num_layers = int(model_config.num_hidden_layers)
    expected_recomputed_layers = num_layers - args.first_recompute_layer
    if expected_recomputed_layers <= 0:
        raise RuntimeError("first_recompute_layer leaves no layers to recompute")

    # ---- Main TTFT benchmark point: calibrate prompts up front ----------
    main_object, main_source_messages, main_target_messages, main_reusable_tokens = (
        calibrate_length_point(tokenizer, args.target_prefix_tokens, "main")
    )
    main_quantized = quantize_ratio(
        r0,
        context_length=main_reusable_tokens,
        bounds=bounds,
    )
    main_predicted_ttft_ms = predict_ttft_ms(
        measurement,
        num_layers=num_layers,
        context_length=main_reusable_tokens,
        ratio=main_quantized.executable_ratio,
    )
    expected_selected_tokens_per_call = main_quantized.repair_tokens

    # ---- Dense baseline: runs to completion BEFORE any raw/fresh
    # registration begins (see flush_exact_radix_cache's docstring for
    # why this ordering and the per-repeat flushing are both mandatory).
    flush_exact_radix_cache(args.base_url)
    warmup_dense_response, _ = timed_post(
        args.base_url, dense_payload(main_target_messages)
    )
    require_finished_by_length(warmup_dense_response, "dense warmup (discarded)")

    dense_ms_samples: list[float] = []
    for _ in range(args.repeats):
        flush_exact_radix_cache(args.base_url)
        dense_response, dense_ms = timed_post(
            args.base_url,
            dense_payload(main_target_messages),
        )
        require_finished_by_length(dense_response, "dense baseline")
        dense_ms_samples.append(dense_ms)
        time.sleep(0.1)
    # Final flush: dense's last formal repeat left main_target_messages'
    # full sequence in the exact radix tree. Clear it before registering
    # anything, so the CacheTune reuse requests below cannot be silently
    # served by that stale exact-cache entry instead of the real
    # approximate-repair path.
    flush_exact_radix_cache(args.base_url)

    raw_hash = "cachetune-raw:phase4-r5-main"
    fresh_hash = "cachetune-fresh:phase4-r5-main"
    register_raw_response = post_json(
        f"{args.base_url}/v1/chat/completions",
        register_payload(
            messages=main_source_messages,
            content_hash=raw_hash,
            reusable_tokens=main_reusable_tokens,
            model_fingerprint=args.model_fingerprint,
            cache_dtype=args.cache_dtype,
        ),
    )
    require_finished_by_length(register_raw_response, "raw register")
    time.sleep(0.1)

    # Discarded warmup (register fresh + reuse): unlike dense, register
    # and reuse requests always set skip_radix_cache_insert=True, so they
    # can never pollute the exact cache and repeating them needs no
    # flush. This warmup exists only to absorb any first-call cost
    # (e.g. lazy controller/profile-cache initialization) so it does not
    # bleed into the formal measurement below.
    warmup_fresh_response, _ = timed_post(
        args.base_url,
        register_payload(
            messages=main_target_messages,
            content_hash=fresh_hash,
            reusable_tokens=main_reusable_tokens,
            model_fingerprint=args.model_fingerprint,
            cache_dtype=args.cache_dtype,
        ),
    )
    require_finished_by_length(
        warmup_fresh_response, "cachetune warmup fresh preparation (discarded)"
    )
    warmup_reuse_response, _ = timed_post(
        args.base_url,
        reuse_payload(
            messages=main_target_messages,
            raw_content_hash=raw_hash,
            reusable_tokens=main_reusable_tokens,
            model_fingerprint=args.model_fingerprint,
            cache_dtype=args.cache_dtype,
        ),
    )
    require_finished_by_length(
        warmup_reuse_response, "cachetune warmup reuse (discarded)"
    )

    # Snapshot AFTER warmup: the warmup's own telemetry contribution must
    # not be counted as part of the formal-repeat delta below.
    metrics_before_cachetune = metric_snapshot(args.base_url)
    fresh_ms_samples: list[float] = []
    cachetune_ms_samples: list[float] = []
    for _ in range(args.repeats):
        register_fresh_response, fresh_ms = timed_post(
            args.base_url,
            register_payload(
                messages=main_target_messages,
                content_hash=fresh_hash,
                reusable_tokens=main_reusable_tokens,
                model_fingerprint=args.model_fingerprint,
                cache_dtype=args.cache_dtype,
            ),
        )
        require_finished_by_length(register_fresh_response, "fresh preparation")

        reuse_response, cachetune_ms = timed_post(
            args.base_url,
            reuse_payload(
                messages=main_target_messages,
                raw_content_hash=raw_hash,
                reusable_tokens=main_reusable_tokens,
                model_fingerprint=args.model_fingerprint,
                cache_dtype=args.cache_dtype,
            ),
        )
        require_finished_by_length(reuse_response, "cachetune reuse")

        fresh_ms_samples.append(fresh_ms)
        cachetune_ms_samples.append(cachetune_ms)
        time.sleep(0.1)
    metrics_after_cachetune = metric_snapshot(args.base_url)

    cachetune_deltas = {
        name: metric_delta(metrics_before_cachetune, metrics_after_cachetune, name)
        for name in (
            "sglang:approx_kv_cachetune_selected_tokens_total",
            "sglang:approx_kv_cachetune_recomputed_layers_total",
            "sglang:approx_kv_cachetune_precomputed_total",
            "sglang:approx_kv_dense_fallback_total",
        )
    }
    main_expected = expected_repair_totals(
        repair_tokens_per_call=expected_selected_tokens_per_call,
        recomputed_layers_per_call=expected_recomputed_layers,
        repeats=args.repeats,
    )
    expect_precomputed_adapter = main_expected["expect_precomputed_adapter"]
    expected_selected_tokens_total = main_expected["expected_selected_tokens_total"]
    expected_recomputed_layers_total = main_expected["expected_recomputed_layers_total"]
    expected_precomputed_total = main_expected["expected_precomputed_total"]
    telemetry_checks = {
        "selected_tokens_total_matches_controller_decision": (
            cachetune_deltas["sglang:approx_kv_cachetune_selected_tokens_total"]
            == expected_selected_tokens_total
        ),
        "recomputed_layers_total_matches_first_recompute_layer": (
            cachetune_deltas["sglang:approx_kv_cachetune_recomputed_layers_total"]
            == expected_recomputed_layers_total
        ),
        "precomputed_adapter_used_every_call": (
            cachetune_deltas["sglang:approx_kv_cachetune_precomputed_total"]
            == expected_precomputed_total
        ),
        "no_unexpected_dense_fallback": (
            cachetune_deltas["sglang:approx_kv_dense_fallback_total"] == 0
        ),
    }
    if not all(telemetry_checks.values()):
        raise RuntimeError(f"telemetry cross-validation failed: {telemetry_checks}")

    combined_ms_samples = [
        fresh_ms + cachetune_ms
        for fresh_ms, cachetune_ms in zip(fresh_ms_samples, cachetune_ms_samples)
    ]
    dense_p50_ms = statistics.median(dense_ms_samples)
    cachetune_target_p50_ms = statistics.median(cachetune_ms_samples)
    fresh_preparation_p50_ms = statistics.median(fresh_ms_samples)
    combined_p50_ms = statistics.median(combined_ms_samples)

    # ---- Real per-length re-quantization sweep (no server restart) -----
    length_sweep_targets = [
        int(value) for value in args.length_sweep.split(",") if value.strip()
    ]
    length_sweep_points: list[dict[str, Any]] = []
    for target_prefix_tokens in length_sweep_targets:
        _, source_messages, target_messages, reusable_tokens = calibrate_length_point(
            tokenizer,
            target_prefix_tokens,
            f"sweep-{target_prefix_tokens}",
        )
        quantized = quantize_ratio(r0, context_length=reusable_tokens, bounds=bounds)
        artifact = f"phase4-r5-cachetune-sweep-{target_prefix_tokens}"
        sweep_raw_hash = f"cachetune-raw:{artifact}"
        sweep_fresh_hash = f"cachetune-fresh:{artifact}"

        register_raw = post_json(
            f"{args.base_url}/v1/chat/completions",
            register_payload(
                messages=source_messages,
                content_hash=sweep_raw_hash,
                reusable_tokens=reusable_tokens,
                model_fingerprint=args.model_fingerprint,
                cache_dtype=args.cache_dtype,
            ),
        )
        require_finished_by_length(
            register_raw, f"sweep[{target_prefix_tokens}] raw register"
        )

        # Discarded warmup (register fresh + reuse) -- same rationale as
        # the main setting above: register/reuse requests never touch
        # the exact radix tree, so no flush is required here or between
        # the formal repeats below.
        warmup_fresh = post_json(
            f"{args.base_url}/v1/chat/completions",
            register_payload(
                messages=target_messages,
                content_hash=sweep_fresh_hash,
                reusable_tokens=reusable_tokens,
                model_fingerprint=args.model_fingerprint,
                cache_dtype=args.cache_dtype,
            ),
        )
        require_finished_by_length(
            warmup_fresh,
            f"sweep[{target_prefix_tokens}] warmup fresh preparation (discarded)",
        )
        warmup_reuse = post_json(
            f"{args.base_url}/v1/chat/completions",
            reuse_payload(
                messages=target_messages,
                raw_content_hash=sweep_raw_hash,
                reusable_tokens=reusable_tokens,
                model_fingerprint=args.model_fingerprint,
                cache_dtype=args.cache_dtype,
            ),
        )
        require_finished_by_length(
            warmup_reuse, f"sweep[{target_prefix_tokens}] warmup reuse (discarded)"
        )

        # Snapshot AFTER warmup, matching the main setting's discipline.
        metrics_before_point = metric_snapshot(args.base_url)
        sweep_fresh_ms_samples: list[float] = []
        sweep_reuse_ms_samples: list[float] = []
        for _ in range(args.repeats):
            register_fresh, fresh_ms = timed_post(
                args.base_url,
                register_payload(
                    messages=target_messages,
                    content_hash=sweep_fresh_hash,
                    reusable_tokens=reusable_tokens,
                    model_fingerprint=args.model_fingerprint,
                    cache_dtype=args.cache_dtype,
                ),
            )
            require_finished_by_length(
                register_fresh, f"sweep[{target_prefix_tokens}] fresh preparation"
            )

            reuse_response, reuse_ms = timed_post(
                args.base_url,
                reuse_payload(
                    messages=target_messages,
                    raw_content_hash=sweep_raw_hash,
                    reusable_tokens=reusable_tokens,
                    model_fingerprint=args.model_fingerprint,
                    cache_dtype=args.cache_dtype,
                ),
            )
            require_finished_by_length(
                reuse_response, f"sweep[{target_prefix_tokens}] reuse"
            )

            sweep_fresh_ms_samples.append(fresh_ms)
            sweep_reuse_ms_samples.append(reuse_ms)
            time.sleep(0.1)
        metrics_after_point = metric_snapshot(args.base_url)

        observed_selected_tokens_total = metric_delta(
            metrics_before_point,
            metrics_after_point,
            "sglang:approx_kv_cachetune_selected_tokens_total",
        )
        observed_dense_fallback = metric_delta(
            metrics_before_point,
            metrics_after_point,
            "sglang:approx_kv_dense_fallback_total",
        )
        point_expected = expected_repair_totals(
            repair_tokens_per_call=quantized.repair_tokens,
            recomputed_layers_per_call=0,  # not tracked per sweep point
            repeats=args.repeats,
        )
        expected_selected_tokens_for_point = point_expected[
            "expected_selected_tokens_total"
        ]
        sweep_combined_ms_samples = [
            fresh_ms + reuse_ms
            for fresh_ms, reuse_ms in zip(
                sweep_fresh_ms_samples, sweep_reuse_ms_samples
            )
        ]
        length_sweep_points.append(
            {
                "reusable_tokens": reusable_tokens,
                "repeats": args.repeats,
                "expected_selected_tokens_per_call": quantized.repair_tokens,
                "expected_selected_tokens_total": expected_selected_tokens_for_point,
                "expected_executable_ratio": quantized.executable_ratio,
                "observed_selected_tokens_total": observed_selected_tokens_total,
                "observed_dense_fallback": observed_dense_fallback,
                "fresh_ms_samples": sweep_fresh_ms_samples,
                "reuse_ms_samples": sweep_reuse_ms_samples,
                "combined_ms_samples": sweep_combined_ms_samples,
                "fresh_p50_ms": statistics.median(sweep_fresh_ms_samples),
                "reuse_p50_ms": statistics.median(sweep_reuse_ms_samples),
                "combined_p50_ms": statistics.median(sweep_combined_ms_samples),
                "passed": (
                    observed_selected_tokens_total == expected_selected_tokens_for_point
                    and observed_dense_fallback == 0
                ),
            }
        )
        time.sleep(0.1)

    if not all(point["passed"] for point in length_sweep_points):
        raise RuntimeError(f"length sweep validation failed: {length_sweep_points}")

    metrics_final = metric_snapshot(args.base_url)
    pool_invariant = idle_pool_invariant(metrics_final)
    health_status = fetch_text(f"{args.base_url}/health")

    known_limitations = [
        "Only one (roofline-derived) ratio configuration received a real "
        f"SM75 server canary in this result: r0={r0:.4f} under mode={mode.value}.",
        "Fresh target-context KV is generated by an explicit dense "
        "preparation request; its cost is included in combined_p50_ms.",
        "The generic online ModelRunner selected-token forward hook "
        "remains unavailable on this fork, so every successful repair in "
        "this canary used the precomputed fresh-KV adapter path, not a "
        "genuine inline per-layer recompute.",
        "Recompute and transfer critical paths are not executed with "
        "genuine wall-clock overlap in this backend; the roofline model "
        "chooses the ratio faithfully, but execution uses this project's "
        "available-hardware adaptation (see cachetune/runtime.py).",
        "This is a CacheTune hardware-controller inspired subset: "
        "frequency-domain token selection, sparse transfer, "
        "multi-stream overlap, and deferred RoPE from the full paper are "
        "out of scope for this branch.",
        "No accuracy/quality benchmark was run; success criteria are "
        "TTFT, real repair-token/telemetry accounting, and absence of "
        "crash/OOM/allocator corruption only.",
    ]
    if mode is CacheTuneMode.SPEED_ONLY:
        known_limitations.append(
            "mode=speed_only allows a 0% repair-token floor; this is this "
            "project's own non-paper setting, not the paper's r_min=15% "
            "quality floor (paper_mechanism)."
        )
    else:
        known_limitations.append(
            "mode=paper_mechanism reproduces the paper's r_min=15% quality "
            "floor; this project does not evaluate output quality, so the "
            "floor is exercised here purely as a ratio-selection behavior."
        )

    payload = {
        "schema_version": 2,
        "runner_git_sha": args.runner_git_sha,
        "image_digest": args.image_digest,
        "model": args.model,
        "model_revision": args.model_revision,
        "scope": {
            "recovery": (
                "CacheTune hardware-aware roofline repair-ratio controller "
                "plus ported CacheBlend-style selected-token repair "
                "(precomputed fresh-KV adapter)"
            ),
            "mode": mode.value,
            "scheduler": "S0 LRU",
            "tier": "GPU-only",
            "prefetch": False,
            "accuracy_metric": False,
        },
        "measurement_protocol": {
            "warmup_passes_per_setting": WARMUP_PASSES_PER_SETTING,
            "warmup_passes_discarded": True,
            "formal_repeats": args.repeats,
            "dense_flush_before_warmup": True,
            "dense_flush_before_each_formal_repeat": True,
            "dense_flush_after_formal_repeats_before_registration": True,
            "cachetune_reuse_flush_between_repeats": False,
            "cachetune_reuse_flush_rationale": (
                "register/reuse requests always set "
                "schedule_batch.Req.skip_radix_cache_insert=True whenever "
                "approx_kv_metadata is present, so they never populate "
                "the exact-match radix tree and cannot exact-hit each "
                "other; only dense baseline requests (no approx_kv "
                "metadata) do, which is why only dense needs inter-repeat "
                "flushing."
            ),
        },
        "hardware_measurement": {
            "t_c_ms": args.t_c_ms,
            "t_i_ms": args.t_i_ms,
            "t_o_ms": args.t_o_ms,
            "roofline_ratio_r0": r0,
        },
        "server_validation": {
            "target_tokens": main_reusable_tokens,
            "num_layers": num_layers,
            "first_recompute_layer": args.first_recompute_layer,
            "controller_executable_ratio": main_quantized.executable_ratio,
            "controller_predicted_ttft_ms": main_predicted_ttft_ms,
            "selected_tokens_per_call": expected_selected_tokens_per_call,
            "recomputed_layers_per_call": (
                expected_recomputed_layers if expect_precomputed_adapter else 0
            ),
            "cachetune_deltas": cachetune_deltas,
            "expected_selected_tokens_total": expected_selected_tokens_total,
            "expected_recomputed_layers_total": expected_recomputed_layers_total,
            "expected_precomputed_total": expected_precomputed_total,
            "telemetry_checks": telemetry_checks,
            "fresh_target_kv_from_dense_preparation": True,
            "raw_body_context_differs_from_target": True,
            "last_prompt_token_real_forward": True,
            "passed": all(telemetry_checks.values()),
        },
        "ttft": {
            "repeats_per_mode": args.repeats,
            "dense_ms_samples": dense_ms_samples,
            "fresh_ms_samples": fresh_ms_samples,
            "cachetune_ms_samples": cachetune_ms_samples,
            "combined_ms_samples": combined_ms_samples,
            "dense_p50_ms": dense_p50_ms,
            "cachetune_target_p50_ms": cachetune_target_p50_ms,
            "fresh_preparation_p50_ms": fresh_preparation_p50_ms,
            "combined_p50_ms": combined_p50_ms,
            "target_only_speedup": dense_p50_ms / cachetune_target_p50_ms,
            "combined_speedup": dense_p50_ms / combined_p50_ms,
        },
        "length_sweep_points": length_sweep_points,
        "pool_invariant": pool_invariant,
        "health_response": health_status,
        "known_limitations": known_limitations,
        "passed": all(telemetry_checks.values())
        and all(point["passed"] for point in length_sweep_points)
        and bool(pool_invariant.get("passed")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> int:
    args = parse_args()
    settings = build_settings(args)
    run_id = (
        f"phase4-r5-cachetune-{args.mode}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_path_str = str(args.output.resolve())
    append_run_log(
        args.central_log,
        {
            "run_id": run_id,
            "status": "running",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "settings": settings,
            "output": output_path_str,
        },
    )
    try:
        if args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        payload = run_canary(args)
    except Exception as exc:
        append_run_log(
            args.central_log,
            {
                "run_id": run_id,
                "status": "failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "settings": settings,
                "output": output_path_str,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise

    append_run_log(
        args.central_log,
        {
            "run_id": run_id,
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "settings": settings,
            "output": output_path_str,
            "result_summary": {
                "passed": payload["passed"],
                "mode": args.mode,
                "roofline_ratio_r0": payload["hardware_measurement"][
                    "roofline_ratio_r0"
                ],
                "dense_p50_ms": payload["ttft"]["dense_p50_ms"],
                "cachetune_target_p50_ms": payload["ttft"]["cachetune_target_p50_ms"],
                "fresh_preparation_p50_ms": payload["ttft"]["fresh_preparation_p50_ms"],
                "combined_p50_ms": payload["ttft"]["combined_p50_ms"],
                "target_only_speedup": payload["ttft"]["target_only_speedup"],
                "combined_speedup": payload["ttft"]["combined_speedup"],
                "length_sweep_points": len(payload["length_sweep_points"]),
            },
        },
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
