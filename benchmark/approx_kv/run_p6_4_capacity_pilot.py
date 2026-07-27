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


def csv_values(value: str, cast) -> tuple:
    values = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated values")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--central-log", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
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
    return parser.parse_args()


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
                request_outcomes = {}
                outcome = (
                    "exact_gpu_hit"
                    if recovered["cached_tokens"] >= expected_cached
                    else "dense_fallback"
                )
            telemetry_consistent = (
                outcome in {"dense_fallback", "unknown"}
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


def launch_cells(rhos: tuple[float, ...]) -> list[tuple[str, float]]:
    cells = [("hierarchical", rho) for rho in rhos if rho != 2.0]
    insert_at = next(
        (index for index, (_, rho) in enumerate(cells) if rho > 2.0),
        len(cells),
    )
    cells[insert_at:insert_at] = [("lru", 2.0), ("hierarchical", 2.0)]
    return cells


def execute(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    rhos = csv_values(args.rhos, float)
    profiles = csv_values(args.profiles, str)
    unknown = set(profiles).difference(PROFILE_KINDS)
    if unknown:
        raise ValueError(f"unknown profiles: {sorted(unknown)}")
    if args.formal_repeats < 2:
        raise ValueError("formal-repeats must be at least 2")
    provenance = source_provenance(args.source_git_sha)
    observed_sha = provenance["source_git_sha"]

    manifest = build_fixed40_manifest(
        chunked_prefill_size=args.chunked_prefill_size,
        chunk_source=args.chunk_source,
    )
    logical_tokens = sum(int(item["logical_tokens"]) for item in manifest["objects"])
    plugin_env = {
        "SGLANG_APPROX_KV_CORE": "1",
        "SGLANG_APPROX_KV_CROSS_STORE": "1",
        "SGLANG_APPROX_KV_REGISTER_EVICTS_EXACT": "1",
        "SGLANG_APPROX_KV_BYTES_PER_TOKEN": str(args.kv_bytes_per_token),
        "SGLANG_APPROX_KV_HOST_BUDGET_BYTES": "0",
    }
    cell_results = []
    server_manifests = []
    for cell_index, (policy, rho) in enumerate(launch_cells(rhos)):
        requested_capacity_tokens = math.ceil(logical_tokens / rho)
        log_path = args.log_dir / (f"p6-4-{policy}-rho{rho:.1f}-server.log")
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
            server_seed=17 + cell_index,
        )
        try:
            wait_ready(
                server,
                port=args.port,
                timeout_s=args.server_start_timeout_s,
            )
            observed_capacity_tokens = max_total_num_tokens(metric_snapshot(args.port))
            profile_results = [
                run_profile(
                    args,
                    manifest,
                    profile=profile,
                    representation_kinds=PROFILE_KINDS[profile],
                )
                for profile in profiles
            ]
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
        "warmup_repeats": 1,
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
                            sum(
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
    payload["raw_sha256"] = payload_sha256(payload)
    validate_phase6_artifact(payload)
    return payload


def main() -> int:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("p6-4-%Y%m%dT%H%M%SZ")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    append_jsonl(
        args.central_log,
        {
            "run_id": run_id,
            "phase": "P6-4",
            "status": "running",
            "output": str(args.output.resolve()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        payload = execute(args, run_id)
        for manifest in payload["server_manifests"]:
            manifest["log_sha256"] = file_sha256(Path(manifest["log_path"]))
        payload.pop("raw_sha256", None)
        payload["raw_sha256"] = payload_sha256(payload)
        validate_phase6_artifact(payload)
        write_json(args.output, payload)
        append_jsonl(
            args.central_log,
            {
                "run_id": run_id,
                "phase": "P6-4",
                "status": "completed",
                "raw_sha256": payload["raw_sha256"],
                "output": str(args.output.resolve()),
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
                "phase": "P6-4",
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
