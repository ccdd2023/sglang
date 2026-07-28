#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from benchmark.approx_kv.metrics import (
    clean_cache_invariant,
    counter_delta,
    max_total_num_tokens,
)
from benchmark.approx_kv.phase6.manifest import (
    REPRESENTATION_PROFILES,
    build_fixed40_manifest,
    fixed_object_token_ids,
)
from benchmark.approx_kv.phase6.runner import (
    append_jsonl,
    execution_status,
    flush_cache,
    generate,
    launch_server,
    machine_manifest,
    metric_snapshot,
    metric_text,
    source_provenance,
    stop_server,
    wait_ready,
    write_json,
)
from benchmark.approx_kv.phase6.schema import (
    RhoDefinitions,
    file_sha256,
    payload_sha256,
    validate_phase6_artifact,
)
from benchmark.approx_kv.phase7.common import (
    CAPACITY_RUNNER,
    Phase7ContractError,
    build_inactive_counter_assertion,
    capacity_error_tolerance,
    ensure_artifact_path_layout,
    finalize_artifact_hash,
    formal_arm_order,
    inactive_counter_observations,
    load_execution_context,
    pending_result_provenance,
    validate_phase7_artifact,
)

_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"])*)"')

PROFILE_KINDS = {
    name: tuple(profile["representation_kinds"])
    for name, profile in REPRESENTATION_PROFILES.items()
}
RUNNER_KEY = "capacity_pilot"
PHASE7_MODE_FIELDS = (
    "phase7_manifest",
    "phase7_setting_id",
    "phase7_restart_index",
)


def csv_values(value: str, cast) -> tuple:
    values = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated values")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision")
    parser.add_argument("--source-git-sha")
    parser.add_argument("--image-digest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--central-log", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--port", type=int, default=30011)
    parser.add_argument("--mem-fraction-static", type=float, default=0.65)
    parser.add_argument("--chunked-prefill-size", type=int, default=1024)
    parser.add_argument(
        "--chunk-source",
        choices=("cl2", "provisional_worst_case"),
        default="provisional_worst_case",
    )
    parser.add_argument("--rhos", default="1.1,1.5,2.0,3.0")
    parser.add_argument(
        "--profiles",
        default="exact_only,r0_like,r1_like_k32,r2_like,r4_like",
    )
    parser.add_argument("--formal-repeats", type=int, default=2)
    parser.add_argument("--server-start-timeout-s", type=float, default=600)
    parser.add_argument("--kv-bytes-per-token", type=int, default=114688)
    parser.add_argument("--capacity-tolerance", type=float, default=0.05)
    parser.add_argument("--phase7-manifest", type=Path)
    parser.add_argument("--phase7-setting-id")
    parser.add_argument("--phase7-restart-index", type=int)
    return parser.parse_args()


def phase7_mode_requested(args: argparse.Namespace) -> bool:
    present = [getattr(args, field, None) is not None for field in PHASE7_MODE_FIELDS]
    if any(present) and not all(present):
        raise Phase7ContractError(
            "--phase7-manifest, --phase7-setting-id, and "
            "--phase7-restart-index must be provided together"
        )
    return all(present)


def configure_phase7_args(
    args: argparse.Namespace,
    context,
) -> argparse.Namespace:
    manifest = context.manifest
    setting = context.setting
    if args.log is None:
        raise Phase7ContractError("Phase7 capacity mode requires --log")
    if args.log_dir is not None:
        raise Phase7ContractError("Phase7 capacity mode uses --log, not --log-dir")
    restart_seeds = manifest["server_template"]["restart_seeds"]
    if context.restart_index >= len(restart_seeds):
        raise Phase7ContractError("Phase7 restart index lacks a pinned server seed")
    args.model = manifest["environment"]["model"]
    args.model_revision = manifest["environment"]["model_revision"]
    args.source_git_sha = context.source["source_git_sha"]
    args.image_digest = manifest["environment"]["image_digest"]
    args.mem_fraction_static = float(setting["mem_fraction_static"])
    args.chunked_prefill_size = int(setting["chunked_prefill_size"])
    args.chunk_source = (
        "cl2" if args.chunked_prefill_size == 4096 else "provisional_worst_case"
    )
    args.rhos = str(setting["rho_logical_demand"])
    args.profiles = ",".join(setting["arms"])
    args.formal_repeats = int(setting["formal_repeats"])
    args.warmup_repeats = int(setting["warmup_repeats"])
    args.capacity_tolerance = capacity_error_tolerance(setting)
    args.kv_bytes_per_token = int(
        manifest["server_template"]["plugin_env"]["SGLANG_APPROX_KV_BYTES_PER_TOKEN"]
    )
    args.phase7_policy = str(setting["policy"])
    args.phase7_rho = float(setting["rho_logical_demand"])
    args.phase7_max_total_tokens = int(setting["max_total_tokens"])
    args.phase7_server_seed = int(restart_seeds[context.restart_index])
    args.phase7_plugin_env = dict(manifest["server_template"]["plugin_env"])
    args.phase7_attention_backend = manifest["server_template"]["attention_backend"]
    args.phase7_sampling_backend = manifest["server_template"]["sampling_backend"]
    args.phase7_context = context
    return args


def validate_historical_args(args: argparse.Namespace) -> None:
    missing = [
        flag
        for flag, value in (
            ("--model-revision", args.model_revision),
            ("--source-git-sha", args.source_git_sha),
            ("--image-digest", args.image_digest),
            ("--log-dir", args.log_dir),
        )
        if value is None
    ]
    if missing:
        raise Phase7ContractError(f"historical P6 mode requires {', '.join(missing)}")
    if args.log is not None:
        raise Phase7ContractError("historical P6 mode uses --log-dir, not --log")


def execution_cells(args: argparse.Namespace) -> list[tuple[str, float]]:
    if hasattr(args, "phase7_context"):
        return [(args.phase7_policy, args.phase7_rho)]
    return launch_cells(csv_values(args.rhos, float))


def capacity_log_path(
    args: argparse.Namespace,
    *,
    policy: str,
    rho: float,
) -> Path:
    if hasattr(args, "phase7_context"):
        return args.log
    return args.log_dir / f"p6-4-{policy}-rho{rho:.1f}-server.log"


def labeled_metric_sum(
    text: str,
    name: str,
    required_labels: dict[str, str],
) -> float:
    total = 0.0
    for line in text.splitlines():
        match = _SAMPLE_RE.match(line.strip())
        if match is None or match.group("name") != name:
            continue
        labels = {
            key: bytes(value, "utf-8").decode("unicode_escape")
            for key, value in _LABEL_RE.findall(match.group("labels") or "")
        }
        if all(labels.get(key) == value for key, value in required_labels.items()):
            total += float(match.group("value"))
    return total


def labeled_metric_delta(
    before: str,
    after: str,
    name: str,
    required_labels: dict[str, str],
) -> float:
    return labeled_metric_sum(after, name, required_labels) - labeled_metric_sum(
        before,
        name,
        required_labels,
    )


def exact_kind(item: dict[str, Any]) -> str:
    role = item["role"]
    if role == "architect":
        return "canonical_base"
    if role == "debugger":
        return "repair_metadata"
    if role in {"live_filler", "dead_filler"}:
        return "filler"
    return "exact_variant"


def prompt_for(
    item: dict[str, Any],
    *,
    round_index: int,
) -> tuple[list[int], list[int]]:
    order = int(item["order"])
    header = [
        31_000 + ((round_index * 193 + order * 97 + offset * 13) % 8_000)
        for offset in range(64)
    ]
    body = fixed_object_token_ids(order, int(item["logical_tokens"]))
    return header + body + [49_000 + ((round_index + order) % 1_000)], body


def cache_protection(
    item: dict[str, Any],
    *,
    profile: str,
    round_index: int,
    bytes_per_token: int,
) -> dict[str, Any]:
    return {
        "object_id": f"exact:{profile}:{round_index}:{item['object_id']}",
        "protected_tokens": 64 + int(item["logical_tokens"]),
        "resident_bytes": (64 + int(item["logical_tokens"])) * bytes_per_token,
        "dense_cost_ms": float(item["dense_cost_ms"]),
        "recovery_cost_ms": float(item["recovery_cost_ms"]),
        "next_use_request_step": (
            None if item["retired"] else int(item["order"]) + 100
        ),
        "object_kind": exact_kind(item),
        "recoverable_from_lower_tier": False,
        "retired": bool(item["retired"]),
    }


def representation_metadata(
    item: dict[str, Any],
    *,
    profile: str,
    representation_index: int,
    object_kind: str,
    round_index: int,
    segment_tokens_max: int,
    temporary: bool = False,
) -> dict[str, Any]:
    body_tokens = int(item["logical_tokens"])
    segments = []
    for segment_index, start in enumerate(range(0, body_tokens, segment_tokens_max)):
        length = min(segment_tokens_max, body_tokens - start)
        object_id = (
            f"approx:{profile}:{round_index}:{item['object_id']}:"
            f"rep{representation_index}:seg{segment_index}"
        )
        dependencies = []
        if representation_index > 0:
            dependency_representation = (
                0
                if object_kind in {"repair_state", "precomputed_adapter", "anchor"}
                else representation_index - 1
            )
            dependencies.append(
                f"approx:{profile}:{round_index}:{item['object_id']}:"
                f"rep{dependency_representation}:seg{segment_index}"
            )
        segments.append(
            {
                "content_hash": (
                    f"p6-4:{profile}:{round_index}:{item['object_id']}:"
                    f"rep{representation_index}:seg{segment_index}"
                ),
                "target_start": 64 + start,
                "length": length,
                "object_id": object_id,
                "object_kind": object_kind,
                "dependencies": dependencies,
                "dense_cost_ms": float(item["dense_cost_ms"]),
                "recovery_cost_ms": float(item["recovery_cost_ms"]),
                "next_use_ordinal": (
                    None if item["retired"] else int(item["order"]) + 100
                ),
                "retired": bool(item["retired"]) or temporary,
                "residency": "device",
            }
        )
    return {
        "operation": "register",
        "model_fingerprint": "p6-fixed40",
        "cache_dtype": "float16",
        "segments": segments,
    }


def reuse_metadata(
    item: dict[str, Any],
    *,
    profile: str,
    round_index: int,
    object_kind: str,
    segment_tokens_max: int,
) -> dict[str, Any]:
    metadata = representation_metadata(
        item,
        profile=profile,
        representation_index=0,
        object_kind=object_kind,
        round_index=round_index,
        segment_tokens_max=segment_tokens_max,
    )
    metadata["operation"] = "reuse"
    return metadata


def run_round(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    profile: str,
    representation_kinds: tuple[str, ...],
    round_index: int,
) -> dict[str, Any]:
    flush_cache(args.port)
    before_metrics = metric_snapshot(args.port)
    before_text = metric_text(args.port)
    materialization_ms = 0.0
    registration_ms = 0.0
    request_error = None
    expected_capacity_error = None
    registered_segments = 0
    registration_rows = []
    try:
        for item in manifest["objects"]:
            prompt, body = prompt_for(item, round_index=round_index)
            if payload_sha256(body) != item["token_ids_sha256"]:
                raise RuntimeError(f"token manifest drift for {item['object_id']}")
            protection = cache_protection(
                item,
                profile=profile,
                round_index=round_index,
                bytes_per_token=args.kv_bytes_per_token,
            )
            namespace = f"p6-4-source:{profile}:{round_index}:{item['object_id']}"
            source = generate(
                port=args.port,
                input_ids=prompt,
                max_new_tokens=1,
                custom_params={"cache_protection": protection},
                extra_key=namespace,
            )
            materialization_ms += float(source["elapsed_ms"])
            for representation_index, object_kind in enumerate(representation_kinds):
                metadata = representation_metadata(
                    item,
                    profile=profile,
                    representation_index=representation_index,
                    object_kind=object_kind,
                    round_index=round_index,
                    segment_tokens_max=manifest["segment_tokens_max"],
                    temporary=(
                        representation_index
                        >= REPRESENTATION_PROFILES[profile]["resident_multiplicity"]
                    ),
                )
                before_registration = metric_text(args.port)
                registered = generate(
                    port=args.port,
                    input_ids=prompt,
                    max_new_tokens=1,
                    custom_params={
                        "cache_protection": protection,
                        "approx_kv": metadata,
                    },
                    extra_key=namespace,
                )
                after_registration = metric_text(args.port)
                registration_outcomes = {
                    outcome: labeled_metric_delta(
                        before_registration,
                        after_registration,
                        "sglang:approx_kv_requests_total",
                        {"operation": "register", "outcome": outcome},
                    )
                    for outcome in ("success", "partial", "dense_only", "error")
                }
                registration_ms += float(registered["elapsed_ms"])
                if registration_outcomes["success"] > 0:
                    registration_outcome = "success"
                    registered_segments += len(metadata["segments"])
                else:
                    registration_outcome = next(
                        (
                            outcome
                            for outcome in ("partial", "dense_only", "error")
                            if registration_outcomes[outcome] > 0
                        ),
                        "unknown",
                    )
                    expected_capacity_error = (
                        "registration outcome "
                        f"{registration_outcome} for {item['object_id']} "
                        f"representation {representation_index}"
                    )
                registration_rows.append(
                    {
                        "object_id": item["object_id"],
                        "representation_index": representation_index,
                        "expected_segments": len(metadata["segments"]),
                        "outcome": registration_outcome,
                        "outcome_deltas": registration_outcomes,
                    }
                )

        replay_rows = []
        for item in manifest["objects"][:5]:
            prompt, _ = prompt_for(item, round_index=round_index)
            if representation_kinds:
                custom_params = {
                    "approx_kv": reuse_metadata(
                        item,
                        profile=profile,
                        round_index=round_index,
                        object_kind=representation_kinds[0],
                        segment_tokens_max=manifest["segment_tokens_max"],
                    )
                }
                replay_namespace = (
                    f"p6-4-replay:{profile}:{round_index}:{item['object_id']}"
                )
                seed = generate(
                    port=args.port,
                    input_ids=prompt[:64],
                    max_new_tokens=1,
                    extra_key=replay_namespace,
                )
            else:
                custom_params = None
                replay_namespace = (
                    f"p6-4-source:{profile}:{round_index}:{item['object_id']}"
                )
                seed = None
            before_replay_text = metric_text(args.port)
            before_replay_metrics = metric_snapshot(args.port)
            recovered = generate(
                port=args.port,
                input_ids=prompt,
                max_new_tokens=1,
                custom_params=custom_params,
                extra_key=replay_namespace,
            )
            after_replay_text = metric_text(args.port)
            after_replay_metrics = metric_snapshot(args.port)
            expected_cached = 64 + int(item["logical_tokens"])
            if representation_kinds:
                request_outcomes = {
                    outcome: labeled_metric_delta(
                        before_replay_text,
                        after_replay_text,
                        "sglang:approx_kv_requests_total",
                        {"operation": "reuse", "outcome": outcome},
                    )
                    for outcome in (
                        "success",
                        "dense_fallback",
                        "exact",
                        "exact_host_preferred",
                    )
                }
                if request_outcomes["success"] > 0:
                    outcome = "approximate_gpu_recovery"
                elif request_outcomes["dense_fallback"] > 0:
                    outcome = "dense_fallback"
                elif request_outcomes["exact"] > 0:
                    outcome = "exact_gpu_hit"
                elif request_outcomes["exact_host_preferred"] > 0:
                    outcome = "host_demand_load"
                else:
                    outcome = "unknown"
            else:
                # No approximate metadata was attached, so this request never
                # entered the recovery path. A short prefix here is an ordinary
                # exact-cache miss, NOT an approximate dense fallback. Calling
                # it dense_fallback previously caused exact-only misses to be
                # cited as evidence that the fallback path had executed.
                request_outcomes = {}
                outcome = (
                    "exact_gpu_hit"
                    if recovered["cached_tokens"] >= expected_cached
                    else "exact_cache_miss"
                )
            telemetry_consistent = (
                outcome in {"dense_fallback", "exact_cache_miss", "unknown"}
                or recovered["cached_tokens"] >= expected_cached
            )
            replay_rows.append(
                {
                    "object_id": item["object_id"],
                    "cached_tokens": recovered["cached_tokens"],
                    "expected_cached_tokens": expected_cached,
                    "outcome": outcome,
                    "request_outcomes": request_outcomes,
                    "telemetry_consistent": telemetry_consistent,
                    "seed_head_ms": (None if seed is None else seed["elapsed_ms"]),
                    "reservation_failures": (
                        counter_delta(
                            before_replay_metrics,
                            after_replay_metrics,
                            "sglang:cross_store_reservation_failures_total",
                        )
                        or 0
                    ),
                    "elapsed_ms": recovered["elapsed_ms"],
                }
            )
    except requests.HTTPError as exc:
        response_text = exc.response.text if exc.response is not None else str(exc)
        if "failed to register approximate KV source segments" in response_text:
            expected_capacity_error = f"{type(exc).__name__}: {response_text[:500]}"
        else:
            request_error = f"{type(exc).__name__}: {response_text[:500]}"
        replay_rows = []
    except (KeyError, RuntimeError, ValueError, requests.RequestException) as exc:
        request_error = f"{type(exc).__name__}: {exc}"
        replay_rows = []

    after_metrics = metric_snapshot(args.port)
    after_text = metric_text(args.port)
    metrics = {
        "exact_evicted_bytes": labeled_metric_delta(
            before_text,
            after_text,
            "sglang:cross_store_evicted_bytes_total",
            {"provenance": "exact"},
        ),
        "approx_evicted_bytes": labeled_metric_delta(
            before_text,
            after_text,
            "sglang:cross_store_evicted_bytes_total",
            {"provenance": "approximate"},
        ),
        "exact_requester_approx_victim_bytes": labeled_metric_delta(
            before_text,
            after_text,
            "sglang:cross_store_evicted_bytes_total",
            {"requester": "exact", "provenance": "approximate"},
        ),
        "approx_requester_exact_victim_bytes": labeled_metric_delta(
            before_text,
            after_text,
            "sglang:cross_store_evicted_bytes_total",
            {"requester": "approximate", "provenance": "exact"},
        ),
        "demoted_bytes": counter_delta(
            before_metrics,
            after_metrics,
            "sglang:cross_store_demoted_bytes_total",
        ),
        "reservation_failures": counter_delta(
            before_metrics,
            after_metrics,
            "sglang:cross_store_reservation_failures_total",
        ),
        "dense_fallback_tokens": counter_delta(
            before_metrics,
            after_metrics,
            "sglang:approx_kv_dense_fallback_total",
        ),
        "peak_device_bytes": after_metrics.get("sglang:cross_store_peak_device_bytes"),
        "orphan_count": after_metrics.get("sglang:approx_kv_store_orphans"),
    }
    flush_cache(args.port)
    post_reset = metric_snapshot(args.port)
    reset = clean_cache_invariant(post_reset)
    store_reset = {
        name: post_reset.get(name)
        for name in (
            "sglang:approx_kv_store_records",
            "sglang:approx_kv_store_device_bytes",
            "sglang:approx_kv_store_host_bytes",
            "sglang:approx_kv_store_leases",
            "sglang:approx_kv_store_orphans",
        )
    }
    valid = (
        request_error is None
        and bool(reset["passed"])
        and all(value in (0, 0.0) for value in store_reset.values())
        and metrics["orphan_count"] in (0, 0.0)
        and all(row["telemetry_consistent"] for row in replay_rows)
        and all(row["outcome"] != "unknown" for row in replay_rows)
    )
    outcomes = {
        outcome: sum(row["outcome"] == outcome for row in replay_rows)
        for outcome in (
            "exact_gpu_hit",
            "approximate_gpu_recovery",
            "host_demand_load",
            "dense_fallback",
            "exact_cache_miss",
            "unknown",
        )
    }
    fallback_reachable = (
        any(
            row["outcome"] == "dense_fallback" and row["reservation_failures"] > 0
            for row in replay_rows
        )
        and request_error is None
    )
    reachability = (
        "invalid"
        if not valid
        else (
            "diagnostic-unavailable"
            if expected_capacity_error is not None
            else "reachable"
        )
    )
    return {
        "round_index": round_index,
        "representation_multiplicity": len(representation_kinds),
        "registered_segments": registered_segments,
        "registrations": registration_rows,
        "materialization_ms": materialization_ms,
        "registration_ms": registration_ms,
        "replay": replay_rows,
        "metrics": metrics,
        "cache_outcomes": outcomes,
        "fallback_reachable": fallback_reachable,
        "reachability": reachability,
        "request_error": request_error,
        "expected_capacity_error": expected_capacity_error,
        "reset_invariant": reset,
        "store_reset_gauges": store_reset,
        "valid": valid,
    }


def run_profile(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    profile: str,
    representation_kinds: tuple[str, ...],
) -> dict[str, Any]:
    warmup = run_round(
        args,
        manifest,
        profile=profile,
        representation_kinds=representation_kinds,
        round_index=-1,
    )
    formal = [
        run_round(
            args,
            manifest,
            profile=profile,
            representation_kinds=representation_kinds,
            round_index=index,
        )
        for index in range(args.formal_repeats)
    ]
    return {
        "profile": profile,
        "representation_kinds": list(representation_kinds),
        "warmup": warmup,
        "formal": formal,
        "reachability": (
            "invalid"
            if any(row["reachability"] == "invalid" for row in formal)
            else (
                "diagnostic-unavailable"
                if any(
                    row["reachability"] == "diagnostic-unavailable" for row in formal
                )
                else "reachable"
            )
        ),
        "valid": all(row["valid"] for row in formal),
    }


def profile_summary(
    profile: str,
    *,
    representation_kinds: tuple[str, ...],
    warmup: list[dict[str, Any]],
    formal: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-profile view of repeat-major execution.

    ``formal`` is ordered by ``round_index`` so that every existing cell and
    status computation keeps working unchanged; the authoritative execution
    order lives in the cell-level ``formal_repeats`` record.
    """
    return {
        "profile": profile,
        "representation_kinds": list(representation_kinds),
        "warmup": warmup,
        "warmup_repeats": len(warmup),
        "formal": formal,
        "reachability": (
            "invalid"
            if any(row["reachability"] == "invalid" for row in formal)
            else (
                "diagnostic-unavailable"
                if any(
                    row["reachability"] == "diagnostic-unavailable" for row in formal
                )
                else "reachable"
            )
        ),
        "valid": all(row["valid"] for row in formal),
    }


def run_phase7_repeat_major(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    profiles: tuple[str, ...],
    setting: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute Phase 7 capacity profiles repeat-major.

    Every formal repeat runs each profile once, in the preregistered
    ``arm_order_by_repeat`` order, so that profile position inside a repeat is
    counterbalanced instead of frozen by manifest order. Warmup keeps the fixed
    manifest arm order and is indexed by warmup repeat.
    """
    warmup_rounds: dict[str, list[dict[str, Any]]] = {
        profile: [] for profile in profiles
    }
    formal_rounds: dict[str, list[dict[str, Any]]] = {
        profile: [] for profile in profiles
    }
    if tuple(profiles) != tuple(setting["arms"]):
        raise Phase7ContractError(
            f"{setting['setting_id']}: executed profiles differ from the "
            "preregistered arms"
        )
    for warmup_index in range(int(setting["warmup_repeats"])):
        for profile in profiles:
            warmup_rounds[profile].append(
                run_round(
                    args,
                    manifest,
                    profile=profile,
                    representation_kinds=PROFILE_KINDS[profile],
                    round_index=-(warmup_index + 1),
                )
            )
    formal_repeats = []
    for repeat_index in range(int(args.formal_repeats)):
        order = formal_arm_order(setting, repeat_index)
        executed = []
        for execution_index, profile in enumerate(order):
            row = run_round(
                args,
                manifest,
                profile=profile,
                representation_kinds=PROFILE_KINDS[profile],
                round_index=repeat_index,
            )
            formal_rounds[profile].append(row)
            executed.append(
                {
                    "profile": profile,
                    "execution_index": execution_index,
                    "round_index": repeat_index,
                    "representation_kinds": list(PROFILE_KINDS[profile]),
                    "reachability": row["reachability"],
                    "valid": row["valid"],
                    # binds this ordered entry to the full round payload kept
                    # in the compatibility per-profile summary
                    "round_sha256": payload_sha256(row),
                }
            )
        formal_repeats.append(
            {
                "repeat_index": repeat_index,
                "arm_order": list(order),
                "profiles": executed,
            }
        )
    profile_results = [
        profile_summary(
            profile,
            representation_kinds=PROFILE_KINDS[profile],
            warmup=warmup_rounds[profile],
            formal=formal_rounds[profile],
        )
        for profile in profiles
    ]
    return profile_results, formal_repeats


def launch_cells(rhos: tuple[float, ...]) -> list[tuple[str, float]]:
    cells = [("hierarchical", rho) for rho in rhos if rho != 2.0]
    insert_at = next(
        (index for index, (_, rho) in enumerate(cells) if rho > 2.0),
        len(cells),
    )
    cells[insert_at:insert_at] = [("lru", 2.0), ("hierarchical", 2.0)]
    return cells


def execute(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    phase7_context = getattr(args, "phase7_context", None)
    rhos = (
        (args.phase7_rho,)
        if phase7_context is not None
        else csv_values(args.rhos, float)
    )
    profiles = csv_values(args.profiles, str)
    unknown = set(profiles).difference(PROFILE_KINDS)
    if unknown:
        raise ValueError(f"unknown profiles: {sorted(unknown)}")
    if args.formal_repeats < 2:
        raise ValueError("formal-repeats must be at least 2")
    if phase7_context is None:
        provenance = source_provenance(args.source_git_sha)
        observed_sha = provenance["source_git_sha"]
    else:
        provenance = {
            "source_git_sha": phase7_context.source["source_git_sha"],
            "source_tree_sha": phase7_context.source["source_tree_sha"],
        }
        observed_sha = provenance["source_git_sha"]

    manifest = build_fixed40_manifest(
        chunked_prefill_size=args.chunked_prefill_size,
        chunk_source=args.chunk_source,
    )
    logical_tokens = sum(int(item["logical_tokens"]) for item in manifest["objects"])
    plugin_env = (
        dict(args.phase7_plugin_env)
        if phase7_context is not None
        else {
            "SGLANG_APPROX_KV_CORE": "1",
            "SGLANG_APPROX_KV_CROSS_STORE": "1",
            "SGLANG_APPROX_KV_REGISTER_EVICTS_EXACT": "1",
            "SGLANG_APPROX_KV_BYTES_PER_TOKEN": str(args.kv_bytes_per_token),
            "SGLANG_APPROX_KV_HOST_BUDGET_BYTES": "0",
        }
    )
    cell_results = []
    server_manifests = []
    for cell_index, (policy, rho) in enumerate(execution_cells(args)):
        requested_capacity_tokens = (
            args.phase7_max_total_tokens
            if phase7_context is not None
            else math.ceil(logical_tokens / rho)
        )
        log_path = capacity_log_path(args, policy=policy, rho=rho)
        server = launch_server(
            model=args.model,
            model_revision=args.model_revision,
            port=args.port,
            mem_fraction_static=args.mem_fraction_static,
            chunked_prefill_size=args.chunked_prefill_size,
            policy=policy,
            log_path=log_path,
            plugin_env=plugin_env,
            max_total_tokens=requested_capacity_tokens,
            server_seed=(
                args.phase7_server_seed
                if phase7_context is not None
                else 17 + cell_index
            ),
            attention_backend=(
                args.phase7_attention_backend
                if phase7_context is not None
                else "torch_native"
            ),
            sampling_backend=(
                args.phase7_sampling_backend
                if phase7_context is not None
                else "pytorch"
            ),
        )
        try:
            wait_ready(
                server,
                port=args.port,
                timeout_s=args.server_start_timeout_s,
            )
            observed_capacity_tokens = max_total_num_tokens(metric_snapshot(args.port))
            before_profiles_text = (
                metric_text(args.port) if phase7_context is not None else None
            )
            formal_repeat_records = None
            if phase7_context is not None:
                profile_results, formal_repeat_records = run_phase7_repeat_major(
                    args,
                    manifest,
                    profiles=profiles,
                    setting=phase7_context.setting,
                )
            else:
                profile_results = [
                    run_profile(
                        args,
                        manifest,
                        profile=profile,
                        representation_kinds=PROFILE_KINDS[profile],
                    )
                    for profile in profiles
                ]
            after_profiles_text = (
                metric_text(args.port) if phase7_context is not None else None
            )
            capacity_error = (
                abs(observed_capacity_tokens - requested_capacity_tokens)
                / requested_capacity_tokens
            )
            exact_evicted = sum(
                float(round_row["metrics"]["exact_evicted_bytes"])
                for profile_row in profile_results
                for round_row in profile_row["formal"]
            )
            approx_evicted = sum(
                float(round_row["metrics"]["approx_evicted_bytes"])
                for profile_row in profile_results
                for round_row in profile_row["formal"]
            )
            exact_to_approx = sum(
                float(round_row["metrics"]["exact_requester_approx_victim_bytes"])
                for profile_row in profile_results
                for round_row in profile_row["formal"]
            )
            approx_to_exact = sum(
                float(round_row["metrics"]["approx_requester_exact_victim_bytes"])
                for profile_row in profile_results
                for round_row in profile_row["formal"]
            )
            approx_recoveries = sum(
                int(round_row["cache_outcomes"]["approximate_gpu_recovery"])
                for profile_row in profile_results
                for round_row in profile_row["formal"]
            )
            if capacity_error > args.capacity_tolerance or any(
                not row["valid"] for row in profile_results
            ):
                cell_status = "invalid"
            elif any(
                row["reachability"] == "diagnostic-unavailable"
                for row in profile_results
            ):
                cell_status = "diagnostic-unavailable"
            elif (
                exact_evicted <= 0
                or approx_evicted <= 0
                or exact_to_approx <= 0
                or approx_to_exact <= 0
                or approx_recoveries <= 0
            ):
                cell_status = "diagnostic-unavailable"
            else:
                cell_status = "valid"
            cell_results.append(
                {
                    "policy": "S4" if policy == "hierarchical" else "S0",
                    "policy_argument": policy,
                    "rho_logical_demand_requested": rho,
                    "rho_logical_demand_observed": (
                        logical_tokens / observed_capacity_tokens
                    ),
                    "requested_capacity_tokens": requested_capacity_tokens,
                    "observed_capacity_tokens": observed_capacity_tokens,
                    "capacity_relative_error": capacity_error,
                    "profiles": profile_results,
                    **(
                        {
                            "capacity_relative_error_tolerance": (
                                args.capacity_tolerance
                            ),
                            "execution_order": "repeat_major",
                            "formal_repeats": formal_repeat_records,
                            "inactive_counter_observations": (
                                inactive_counter_observations(
                                    before_profiles_text,
                                    after_profiles_text,
                                )
                            ),
                        }
                        if phase7_context is not None
                        else {}
                    ),
                    "evidence": {
                        "exact_evicted_bytes": exact_evicted,
                        "approx_evicted_bytes": approx_evicted,
                        "exact_requester_approx_victim_bytes": exact_to_approx,
                        "approx_requester_exact_victim_bytes": approx_to_exact,
                        "approximate_recoveries": approx_recoveries,
                        "bidirectional_pressure": (
                            exact_to_approx > 0 and approx_to_exact > 0
                        ),
                    },
                    "status": cell_status,
                }
            )
            server_manifests.append(
                {
                    "policy": policy,
                    "rho": rho,
                    "server_argv": list(server.command),
                    "plugin_env": server.plugin_env,
                    "log_path": str(log_path),
                }
            )
        except (
            KeyError,
            MemoryError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            requests.RequestException,
        ) as exc:
            # A single unreachable cell must not destroy the rest of the
            # matrix. Record it explicitly and continue; the contract allows
            # a cell to be reported unreachable, but it requires the other
            # cells to still produce evidence.
            cell_results.append(
                {
                    "policy": "S4" if policy == "hierarchical" else "S0",
                    "policy_argument": policy,
                    "rho_logical_demand_requested": rho,
                    "requested_capacity_tokens": requested_capacity_tokens,
                    "observed_capacity_tokens": None,
                    "profiles": [],
                    **(
                        {
                            "capacity_relative_error_tolerance": (
                                args.capacity_tolerance
                            ),
                            "execution_order": "repeat_major",
                            "formal_repeats": [],
                            "inactive_counter_observations": {},
                        }
                        if phase7_context is not None
                        else {}
                    ),
                    "evidence": {},
                    "status": "diagnostic-unavailable",
                    "unreachable_reason": f"{type(exc).__name__}: {exc}",
                    "server_log": str(log_path),
                }
            )
            server_manifests.append(
                {
                    "policy": policy,
                    "rho": rho,
                    "server_argv": list(server.command),
                    "plugin_env": server.plugin_env,
                    "log_path": str(log_path),
                    "unreachable": True,
                }
            )
        finally:
            stop_server(server)

    all_formal = [
        row
        for cell in cell_results
        for profile in cell["profiles"]
        for row in profile["formal"]
    ]
    exact_evicted = sum(
        float(row["metrics"]["exact_evicted_bytes"]) for row in all_formal
    )
    approx_evicted = sum(
        float(row["metrics"]["approx_evicted_bytes"]) for row in all_formal
    )
    exact_to_approx = sum(
        float(row["metrics"]["exact_requester_approx_victim_bytes"])
        for row in all_formal
    )
    approx_to_exact = sum(
        float(row["metrics"]["approx_requester_exact_victim_bytes"])
        for row in all_formal
    )
    fallback_reachable_rounds = sum(
        bool(row["fallback_reachable"]) for row in all_formal
    )
    statuses = {cell["status"] for cell in cell_results}
    if "invalid" in statuses:
        overall_status = "invalid"
    elif (
        statuses == {"valid"}
        and exact_evicted > 0
        and approx_evicted > 0
        and exact_to_approx > 0
        and approx_to_exact > 0
        and fallback_reachable_rounds > 0
    ):
        overall_status = "valid"
    else:
        overall_status = "inconclusive"
    requested_capacities = [cell["requested_capacity_tokens"] for cell in cell_results]
    observed_capacities = [cell["observed_capacity_tokens"] for cell in cell_results]
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": "P6-4",
        "source_git_sha": observed_sha,
        "source_tree_sha": provenance["source_tree_sha"],
        "result_git_sha": None,
        "result_commit_status": "pending_result_commit",
        "model": args.model,
        "model_revision": args.model_revision,
        "image_digest": args.image_digest,
        "machine": machine_manifest(),
        "server_argv": [item["server_argv"] for item in server_manifests],
        "plugin_env": plugin_env,
        "server_manifests": server_manifests,
        "workload": manifest,
        "requested_capacity": {
            "tokens": requested_capacities,
            "pages": requested_capacities,
            "bytes": [
                value * args.kv_bytes_per_token for value in requested_capacities
            ],
        },
        "observed_capacity": {
            "tokens": observed_capacities,
            "pages": observed_capacities,
            "bytes": [
                None if value is None else value * args.kv_bytes_per_token
                for value in observed_capacities
            ],
        },
        "crosses_chunk_boundary": any(
            item["crosses_chunk_boundary"] for item in manifest["objects"]
        ),
        "segment_count": sum(
            int(item["segment_count"]) for item in manifest["objects"]
        ),
        "warmup_repeats": (
            int(phase7_context.setting["warmup_repeats"])
            if phase7_context is not None
            else 1
        ),
        "formal_repeats": args.formal_repeats,
        "restarts": 1,
        "cells": cell_results,
        "ledger": {
            "setup": {
                "server_starts": len(server_manifests),
                "fixed_objects": manifest["object_count"],
            },
            "materialization": {
                "logical_tokens_per_round": logical_tokens,
            },
            "recovery": {
                "workflow_replays_per_round": 5,
            },
            "scheduler": {
                "policies": ["S0", "S4"],
            },
            "transfer": {
                "host_enabled": False,
            },
            "temporary_peak": {
                "max_device_bytes": max(
                    (
                        float(row["metrics"]["peak_device_bytes"] or 0)
                        for row in all_formal
                    ),
                    default=0,
                ),
            },
        },
        "rho": {
            **RhoDefinitions().__dict__,
            "logical_values": list(rhos),
            "physical_profiles": {
                profile: {
                    **{
                        key: value
                        for key, value in REPRESENTATION_PROFILES[profile].items()
                        if key != "representation_kinds"
                    },
                    "peak_multiplicity": max(
                        REPRESENTATION_PROFILES[profile]["resident_multiplicity"],
                        REPRESENTATION_PROFILES[profile]["temporary_multiplicity"],
                    ),
                    "requested_physical_demand_by_cell": [
                        (
                            None
                            if cell["observed_capacity_tokens"] is None
                            else sum(
                                64
                                + int(item["logical_tokens"])
                                * (
                                    1
                                    + max(
                                        REPRESENTATION_PROFILES[profile][
                                            "resident_multiplicity"
                                        ],
                                        REPRESENTATION_PROFILES[profile][
                                            "temporary_multiplicity"
                                        ],
                                    )
                                )
                                for item in manifest["objects"]
                            )
                            / cell["observed_capacity_tokens"]
                        )
                        for cell in cell_results
                    ],
                }
                for profile in profiles
            },
        },
        "bidirectional_pressure": {
            "exact_evicted_bytes": exact_evicted,
            "approx_evicted_bytes": approx_evicted,
            "exact_requester_approx_victim_bytes": exact_to_approx,
            "approx_requester_exact_victim_bytes": approx_to_exact,
            "passed": exact_to_approx > 0 and approx_to_exact > 0,
        },
        "fallback_reachability": {
            "rounds": fallback_reachable_rounds,
            "passed": fallback_reachable_rounds > 0,
        },
        "performance_claim": "disabled",
        "status": overall_status,
    }
    if phase7_context is not None:
        manifest_payload = phase7_context.manifest
        setting = phase7_context.setting
        inactive_assertion = build_inactive_counter_assertion(
            manifest_payload,
            [
                cell["inactive_counter_observations"]
                for cell in cell_results
                if cell["inactive_counter_observations"]
            ],
        )
        outcome_counts = {
            outcome: 0 for outcome in manifest_payload["outcome_taxonomy"]
        }
        outcome_mapping = {
            "exact_gpu_hit": "exact_gpu_hit",
            "approximate_gpu_recovery": "approximate_gpu_recovery",
            "host_demand_load": "host_demand_load",
            "dense_fallback": "approximate_recovery_failed_dense",
            "exact_cache_miss": "ordinary_exact_cache_miss",
        }
        for row in all_formal:
            for source_outcome, count in row["cache_outcomes"].items():
                target_outcome = outcome_mapping.get(source_outcome)
                if target_outcome is not None:
                    outcome_counts[target_outcome] += int(count)
        payload.update(
            {
                "phase": "Phase7-capacity",
                "phase7_mode": True,
                "execution_envelope": phase7_context.envelope,
                "manifest_revision": manifest_payload["manifest_revision"],
                "preregistered_manifest_sha256": manifest_payload[
                    "preregistered_manifest_sha256"
                ],
                "manifest_file_sha256": phase7_context.manifest_file_sha256,
                "plan": manifest_payload["plan"],
                "setting_id": setting["setting_id"],
                "setting": setting,
                "restart_index": phase7_context.restart_index,
                "runner": {
                    "module": phase7_context.runner_module,
                    "path": phase7_context.runner_path,
                    "sha256": phase7_context.runner_sha256,
                },
                "server_log_path": str(args.log.resolve()),
                "server_log_sha256": None,
                "outcome": {
                    "taxonomy": manifest_payload["outcome_taxonomy"],
                    "counts": outcome_counts,
                    "exclusive_terminal_reasons": manifest_payload[
                        "exclusive_terminal_reasons"
                    ],
                    "terminal_reason_counts": {},
                },
                "reset": {
                    str(cell_index): {
                        profile["profile"]: [
                            {
                                "reset_invariant": row["reset_invariant"],
                                "store_reset_gauges": row["store_reset_gauges"],
                            }
                            for row in profile["formal"]
                        ]
                        for profile in cell["profiles"]
                    }
                    for cell_index, cell in enumerate(cell_results)
                },
                "inactive_counter_assertion": inactive_assertion,
                "provenance": {
                    "manifest_path": str(phase7_context.manifest_path.resolve()),
                    "manifest_file_sha256": phase7_context.manifest_file_sha256,
                    "implementation": manifest_payload["implementation"],
                    "runner_sha256": phase7_context.runner_sha256,
                    "source": phase7_context.source,
                },
                "phase7_parameters": {
                    "policy": args.phase7_policy,
                    "rho_logical_demand": args.phase7_rho,
                    "chunked_prefill_size": args.chunked_prefill_size,
                    "max_total_tokens": args.phase7_max_total_tokens,
                    "mem_fraction_static": args.mem_fraction_static,
                    "profiles": list(profiles),
                    "warmup_repeats": int(setting["warmup_repeats"]),
                    "formal_repeats": args.formal_repeats,
                    "arm_order_by_repeat": dict(setting["arm_order_by_repeat"]),
                    "execution_order": "repeat_major",
                    "capacity_relative_error_tolerance": args.capacity_tolerance,
                    "restart_index": phase7_context.restart_index,
                    "server_seed": args.phase7_server_seed,
                },
            }
        )
        if not inactive_assertion["passed"]:
            payload["status"] = "invalid"
    payload["raw_sha256"] = payload_sha256(payload)
    validate_phase6_artifact(payload)
    return payload


def phase7_failure_artifact(
    *,
    args: argparse.Namespace,
    context,
    run_id: str,
    error: Exception,
) -> dict[str, Any]:
    manifest = context.manifest
    setting = context.setting
    workload = build_fixed40_manifest(
        chunked_prefill_size=int(setting["chunked_prefill_size"]),
        chunk_source=(
            "cl2"
            if int(setting["chunked_prefill_size"]) == 4096
            else "provisional_worst_case"
        ),
    )
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": "Phase7-capacity",
        "phase7_mode": True,
        "source_git_sha": context.source["source_git_sha"],
        "source_tree_sha": context.source["source_tree_sha"],
        **pending_result_provenance(),
        "execution_envelope": context.envelope,
        "raw_sha256": "",
        "server_argv": [],
        "plugin_env": dict(args.phase7_plugin_env),
        "machine": {"runtime_probe": "unavailable_due_to_failure"},
        "image_digest": manifest["environment"]["image_digest"],
        "requested_capacity": {
            "tokens": [int(setting["max_total_tokens"])],
            "pages": [int(setting["max_total_tokens"])],
            "bytes": [int(setting["max_total_tokens"]) * args.kv_bytes_per_token],
        },
        "observed_capacity": {
            "tokens": [None],
            "pages": [None],
            "bytes": [None],
        },
        "crosses_chunk_boundary": any(
            item["crosses_chunk_boundary"] for item in workload["objects"]
        ),
        "segment_count": sum(
            int(item["segment_count"]) for item in workload["objects"]
        ),
        "warmup_repeats": int(setting["warmup_repeats"]),
        "formal_repeats": int(setting["formal_repeats"]),
        "restarts": 1,
        "ledger": {
            "setup": {},
            "materialization": {},
            "recovery": {},
            "scheduler": {},
            "transfer": {},
            "temporary_peak": {},
        },
        "rho": {
            **RhoDefinitions().__dict__,
            "logical_values": [float(setting["rho_logical_demand"])],
        },
        "status": "invalid",
        "manifest_revision": manifest["manifest_revision"],
        "preregistered_manifest_sha256": manifest["preregistered_manifest_sha256"],
        "manifest_file_sha256": context.manifest_file_sha256,
        "plan": manifest["plan"],
        "setting_id": setting["setting_id"],
        "setting": setting,
        "restart_index": context.restart_index,
        "runner": {
            "module": context.runner_module,
            "path": context.runner_path,
            "sha256": context.runner_sha256,
        },
        "server_log_path": str(args.log.resolve()),
        "server_log_sha256": file_sha256(args.log) if args.log.exists() else None,
        "outcome": {
            "taxonomy": manifest["outcome_taxonomy"],
            "counts": {},
            "exclusive_terminal_reasons": manifest["exclusive_terminal_reasons"],
            "terminal_reason_counts": {},
        },
        "reset": {},
        "inactive_counter_assertion": build_inactive_counter_assertion(manifest, []),
        "provenance": {
            "manifest_path": str(context.manifest_path.resolve()),
            "manifest_file_sha256": context.manifest_file_sha256,
            "implementation": manifest["implementation"],
            "runner_sha256": context.runner_sha256,
            "source": context.source,
        },
        "phase7_parameters": {
            "policy": args.phase7_policy,
            "rho_logical_demand": args.phase7_rho,
            "chunked_prefill_size": args.chunked_prefill_size,
            "max_total_tokens": args.phase7_max_total_tokens,
            "mem_fraction_static": args.mem_fraction_static,
            "profiles": list(setting["arms"]),
            "warmup_repeats": int(setting["warmup_repeats"]),
            "formal_repeats": args.formal_repeats,
            "arm_order_by_repeat": dict(setting["arm_order_by_repeat"]),
            "execution_order": "repeat_major",
            "capacity_relative_error_tolerance": capacity_error_tolerance(setting),
            "restart_index": context.restart_index,
            "server_seed": args.phase7_server_seed,
        },
        "execution_status": execution_status(error),
        "error": f"{type(error).__name__}: {error}",
    }
    finalize_artifact_hash(payload)
    validate_phase7_artifact(payload, manifest=manifest)
    return payload


def main() -> int:
    args = parse_args()
    phase7_mode = phase7_mode_requested(args)
    context = None
    if phase7_mode:
        context = load_execution_context(
            manifest_path=args.phase7_manifest,
            setting_id=args.phase7_setting_id,
            restart_index=args.phase7_restart_index,
            runner_key=RUNNER_KEY,
            runner_module=CAPACITY_RUNNER,
            runner_file=Path(__file__),
        )
        configure_phase7_args(args, context)
        ensure_artifact_path_layout(
            output=args.output,
            log=args.log,
            central_log=args.central_log,
            staging_root=context.manifest["artifact_templates"]["runtime_staging_root"],
        )
        run_id = (
            f"p7-capacity-{args.phase7_setting_id}-"
            f"r{args.phase7_restart_index}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        )
        phase = "Phase7-capacity"
    else:
        validate_historical_args(args)
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        run_id = datetime.now(timezone.utc).strftime("p6-4-%Y%m%dT%H%M%SZ")
        phase = "P6-4"
    append_jsonl(
        args.central_log,
        {
            "run_id": run_id,
            "phase": phase,
            "status": "running",
            "output": str(args.output.resolve()),
            **(
                {
                    "setting_id": context.setting["setting_id"],
                    "restart_index": context.restart_index,
                    "manifest_sha256": context.manifest[
                        "preregistered_manifest_sha256"
                    ],
                }
                if context is not None
                else {}
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        payload = execute(args, run_id)
        for manifest in payload["server_manifests"]:
            manifest["log_sha256"] = file_sha256(Path(manifest["log_path"]))
        if context is not None:
            payload["server_log_sha256"] = file_sha256(args.log)
            finalize_artifact_hash(payload)
            validate_phase7_artifact(payload, manifest=context.manifest)
        else:
            payload.pop("raw_sha256", None)
            payload["raw_sha256"] = payload_sha256(payload)
            validate_phase6_artifact(payload)
        write_json(args.output, payload)
        append_jsonl(
            args.central_log,
            {
                "run_id": run_id,
                "phase": phase,
                "status": "completed",
                "raw_sha256": payload["raw_sha256"],
                "output": str(args.output.resolve()),
                **(
                    {
                        "setting_id": context.setting["setting_id"],
                        "restart_index": context.restart_index,
                    }
                    if context is not None
                    else {}
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return 0
    except (
        KeyError,
        MemoryError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        requests.RequestException,
    ) as exc:
        status = execution_status(exc)
        if context is not None:
            failure = phase7_failure_artifact(
                args=args,
                context=context,
                run_id=run_id,
                error=exc,
            )
        else:
            failure = {
                "schema_version": 1,
                "run_id": run_id,
                "phase": "P6-4",
                "source_git_sha": args.source_git_sha,
                "image_digest": args.image_digest,
                "status": "invalid",
                "execution_status": status,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failure["raw_sha256"] = payload_sha256(failure)
        write_json(args.output, failure)
        append_jsonl(
            args.central_log,
            {
                "run_id": run_id,
                "phase": phase,
                "status": status,
                "error": failure["error"],
                "raw_sha256": failure["raw_sha256"],
                "output": str(args.output.resolve()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
