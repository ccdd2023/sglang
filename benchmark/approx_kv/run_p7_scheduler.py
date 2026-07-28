#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests

from benchmark.approx_kv.build_phase7_manifest import token_list_sha
from benchmark.approx_kv.phase6.manifest import REPRESENTATION_PROFILES
from benchmark.approx_kv.phase6.runner import (
    append_jsonl,
    execution_status,
    flush_cache,
    generate,
    launch_server,
    machine_manifest,
    metric_snapshot,
    metric_text,
    stop_server,
    stream_generate,
    wait_ready,
    write_json,
)
from benchmark.approx_kv.phase6.schema import file_sha256
from benchmark.approx_kv.phase7.common import (
    SCHEDULER_RUNNER,
    Phase7ContractError,
    Phase7RunError,
    classify_request_outcome,
    cross_store_metrics,
    ensure_artifact_path_layout,
    finalize_artifact_hash,
    load_execution_context,
    memory_footprint,
    observed_capacity,
    pending_result_provenance,
    phase7_reset_invariant,
    registration_outcome_observations,
    request_outcome_observations,
    rho_payload,
    terminal_reason_observations,
    validate_phase7_artifact,
    w_workload,
)
from benchmark.approx_kv.phase7.statistics import (
    pair_scheduler_arms,
    performance_ranking_enabled,
    summarize_workflow_records,
)
from benchmark.approx_kv.run_p6_4_capacity_pilot import (
    exact_kind,
    representation_metadata,
    reuse_metadata,
)

RUNNER_KEY = "scheduler"
HEADER_TOKENS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--restart-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--central-log", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--server-start-timeout-s", type=float, default=600)
    parser.add_argument("--request-timeout-s", type=float, default=300)
    return parser.parse_args()


def formal_arm_order(setting: Mapping[str, Any], repeat_index: int) -> tuple[str, ...]:
    order = setting["arm_order_by_repeat"].get(str(repeat_index))
    if order is None or sorted(order) != sorted(setting["arms"]):
        raise Phase7ContractError(
            f"{setting['setting_id']}: invalid arm order for repeat {repeat_index}"
        )
    return tuple(order)


def scheduler_performance_contract(arms: list[str]) -> dict[str, Any]:
    ranking = performance_ranking_enabled(arms)
    return {
        "arm_label": "R4-like-5x" if not ranking else "E0/R0",
        "performance_ranking_enabled": ranking,
        "early_stop": "ES-ENGINEERING-only",
        "r0_mde_applies": False,
        "claim": (
            "synthetic_footprint_and_victim_diagnostic_only_not_kvcomm"
            if not ranking
            else "paired_E0_R0_scheduler_trace"
        ),
    }


def full_reset(
    *,
    port: int,
    clean_baseline: Mapping[str, float] | None,
    strict: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    flush_cache(port)
    snapshot = metric_snapshot(port)
    invariant = phase7_reset_invariant(
        snapshot,
        strict=strict,
        clean_baseline=clean_baseline,
    )
    if not invariant["passed"]:
        raise RuntimeError(f"Phase7 full reset failed: {invariant}")
    return snapshot, invariant


def cache_protection(
    item: Mapping[str, Any],
    *,
    object_id: str,
    protected_tokens: int,
    bytes_per_token: int,
    current_step: int | None,
    next_use_request_step: int | None,
    retired: bool,
) -> dict[str, Any]:
    return {
        "object_id": object_id,
        "protected_tokens": protected_tokens,
        "resident_bytes": protected_tokens * bytes_per_token,
        "dense_cost_ms": float(item["dense_cost_ms"]),
        "recovery_cost_ms": float(item["recovery_cost_ms"]),
        "current_step": current_step,
        "next_use_request_step": next_use_request_step,
        "workflow_stage": item["role"],
        "object_kind": exact_kind(dict(item)),
        "recoverable_from_lower_tier": False,
        "retired": retired,
    }


def representation_contract(arm: str) -> tuple[str | None, tuple[str, ...]]:
    if arm == "E0":
        return None, ()
    if arm == "R0":
        return "r0_like", ("canonical_base",)
    if arm == "R4-like-5x":
        profile = REPRESENTATION_PROFILES["r4_like"]
        return "r4_like", tuple(profile["representation_kinds"])
    raise Phase7ContractError(f"unsupported scheduler arm {arm!r}")


def first_use_by_object(workload: Mapping[str, Any]) -> dict[str, int]:
    result = {}
    for request in workload["request_order"]:
        result.setdefault(request["object_id"], int(request["request_index"]))
    return result


def setup_arm(
    *,
    args: argparse.Namespace,
    workload: Mapping[str, Any],
    arm: str,
    repeat_index: int,
    bytes_per_token: int,
) -> dict[str, Any]:
    profile, representation_kinds = representation_contract(arm)
    segment_tokens_max = int(workload["segment_tokens_max"])
    first_use = first_use_by_object(workload)
    rows = []
    victim_sequence = []
    materialize_ms = 0.0
    register_ms = 0.0
    registration_failed = False
    registration_failed_by_object: dict[str, bool] = {}
    for event_index, object_id in enumerate(workload["fill_order"]):
        object_payload = workload["objects_by_id"][object_id]
        item = object_payload["spec"]
        body = object_payload["body"]
        header = (
            object_payload["target_header"]
            if arm == "E0"
            else object_payload["source_header"]
        )
        prompt = header + body + [54_000 + int(item["order"])]
        namespace = (
            f"p7-w:{arm}:{repeat_index}:{object_id}:"
            f"{'target' if arm == 'E0' else 'source'}"
        )
        protection = cache_protection(
            item,
            object_id=f"p7-w-exact:{arm}:{repeat_index}:{object_id}",
            protected_tokens=HEADER_TOKENS + len(body),
            bytes_per_token=bytes_per_token,
            current_step=None,
            next_use_request_step=first_use.get(object_id),
            retired=bool(item["retired"]),
        )
        before_object_text = metric_text(args.port)
        before_object_snapshot = metric_snapshot(args.port)
        materialized = generate(
            port=args.port,
            input_ids=prompt,
            max_new_tokens=1,
            custom_params={"cache_protection": protection},
            extra_key=namespace,
            timeout_s=args.request_timeout_s,
        )
        materialize_ms += float(materialized["elapsed_ms"])
        registrations = []
        object_registration_failed = False
        if profile is not None:
            for representation_index, object_kind in enumerate(representation_kinds):
                metadata = representation_metadata(
                    dict(item),
                    profile=profile,
                    representation_index=representation_index,
                    object_kind=object_kind,
                    round_index=repeat_index,
                    segment_tokens_max=segment_tokens_max,
                )
                for segment in metadata["segments"]:
                    segment["next_use_ordinal"] = first_use.get(object_id)
                    segment["retired"] = bool(item["retired"])
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
                    timeout_s=args.request_timeout_s,
                )
                after_registration = metric_text(args.port)
                observations = registration_outcome_observations(
                    before_registration,
                    after_registration,
                )
                expected_cached = HEADER_TOKENS + len(body)
                cache_verified = registered["cached_tokens"] >= expected_cached
                if observations["verification"] == "unknown" and cache_verified:
                    observations["verification"] = "indirectly_verified"
                    observations["indirect_evidence"] = (
                        "registered request retained the full object prefix"
                    )
                failed = (
                    observations["registration_failed"]
                    or not cache_verified
                    or len(observations["positive_outcomes"]) > 1
                )
                registration_failed = registration_failed or failed
                object_registration_failed = object_registration_failed or failed
                register_ms += float(registered["elapsed_ms"])
                registrations.append(
                    {
                        "representation_index": representation_index,
                        "object_kind": object_kind,
                        "segments": len(metadata["segments"]),
                        "elapsed_ms": registered["elapsed_ms"],
                        "cached_tokens": registered["cached_tokens"],
                        "observations": observations,
                        "registration_failed": failed,
                    }
                )
                if failed and arm == "R4-like-5x":
                    break
        registration_failed_by_object[object_id] = object_registration_failed
        after_object_text = metric_text(args.port)
        after_object_snapshot = metric_snapshot(args.port)
        object_metrics = cross_store_metrics(
            before_text=before_object_text,
            after_text=after_object_text,
            before_snapshot=before_object_snapshot,
            after_snapshot=after_object_snapshot,
        )
        for victim in object_metrics["victim_evict_bytes"]["rows"]:
            victim_sequence.append(
                {
                    "event_index": event_index,
                    "trigger_phase": "setup",
                    "trigger_object_id": object_id,
                    **victim,
                }
            )
        rows.append(
            {
                "object_id": object_id,
                "prompt_token_sha256": token_list_sha(prompt),
                "body_token_sha256": item["token_ids_sha256"],
                "namespace": namespace,
                "materialize_ms": materialized["elapsed_ms"],
                "materialize_cached_tokens": materialized["cached_tokens"],
                "registrations": registrations,
                "metrics": object_metrics,
            }
        )
        if registration_failed and arm == "R4-like-5x":
            break
    return {
        "arm": arm,
        "profile": profile,
        "representation_kinds": list(representation_kinds),
        "representation_multiplicity": len(representation_kinds),
        "rows": rows,
        "materialize_ms": materialize_ms,
        "register_ms": register_ms,
        "setup_ms": materialize_ms + register_ms,
        "registration_failed": registration_failed,
        "registration_failed_by_object": registration_failed_by_object,
        "victim_sequence": victim_sequence,
    }


def target_namespace(arm: str, repeat_index: int, object_id: str) -> str:
    return f"p7-w:{arm}:{repeat_index}:{object_id}:target"


def run_request(
    *,
    args: argparse.Namespace,
    workload: Mapping[str, Any],
    request: Mapping[str, Any],
    arm: str,
    repeat_index: int,
    measured: bool,
    bytes_per_token: int,
    registration_failed: bool,
) -> dict[str, Any]:
    object_payload = workload["objects_by_id"][request["object_id"]]
    item = object_payload["spec"]
    body = object_payload["body"]
    header = object_payload["target_header"]
    request_index = int(request["request_index"])
    prompt = header + body + [50_000 + request_index]
    namespace = target_namespace(arm, repeat_index, request["object_id"])
    protection = cache_protection(
        item,
        object_id=f"p7-w-target:{arm}:{repeat_index}:{request['object_id']}",
        protected_tokens=HEADER_TOKENS + len(body),
        bytes_per_token=bytes_per_token,
        current_step=request_index,
        next_use_request_step=request["next_use_request_index"],
        retired=request["next_use_request_index"] is None,
    )
    seed = None
    if arm in {"R0", "R4-like-5x"}:
        seed = generate(
            port=args.port,
            input_ids=header,
            max_new_tokens=1,
            custom_params={"cache_protection": protection},
            extra_key=namespace,
            timeout_s=args.request_timeout_s,
        )
    before_text = metric_text(args.port)
    before_snapshot = metric_snapshot(args.port)
    custom_params: dict[str, Any] = {"cache_protection": protection}
    if arm in {"R0", "R4-like-5x"}:
        profile = "r0_like" if arm == "R0" else "r4_like"
        metadata = reuse_metadata(
            dict(item),
            profile=profile,
            round_index=repeat_index,
            object_kind="canonical_base",
            segment_tokens_max=int(workload["segment_tokens_max"]),
        )
        for segment in metadata["segments"]:
            segment["next_use_ordinal"] = request["next_use_request_index"]
            segment["retired"] = request["next_use_request_index"] is None
        custom_params["approx_kv"] = metadata
    result = stream_generate(
        port=args.port,
        input_ids=prompt,
        max_new_tokens=1,
        custom_params=custom_params,
        extra_key=namespace,
        timeout_s=args.request_timeout_s,
    )
    after_text = metric_text(args.port)
    after_snapshot = metric_snapshot(args.port)
    expected_cached = HEADER_TOKENS + len(body)
    request_observations = (
        request_outcome_observations(before_text, after_text, operation="reuse")
        if arm in {"R0", "R4-like-5x"}
        else None
    )
    terminal_observations = (
        terminal_reason_observations(before_text, after_text)
        if arm in {"R0", "R4-like-5x"}
        else None
    )
    outcome = classify_request_outcome(
        arm=arm,
        cached_tokens=int(result["cached_tokens"]),
        expected_cached_tokens=expected_cached,
        request_observations=request_observations,
        terminal_observations=terminal_observations,
        registration_failed=registration_failed,
        expected_outcomes=(
            (
                "approximate_gpu_recovery",
                "exact_gpu_hit",
                "approximate_recovery_failed_dense",
            )
            if arm in {"R0", "R4-like-5x"}
            else ("exact_gpu_hit", "ordinary_exact_cache_miss")
        ),
    )
    seed_ms = 0.0 if seed is None else float(seed["elapsed_ms"])
    seed_elapsed_ms = seed_ms
    request_path_ttft = seed_ms + float(result["ttft_ms"])
    request_path_elapsed = seed_elapsed_ms + float(result["elapsed_ms"])
    return {
        "sample_kind": "measured" if measured else "warmup",
        "repeat": repeat_index,
        "request_index": request_index,
        "phase": request["phase"],
        "role": request["role"],
        "object_id": request["object_id"],
        "next_use_request_index": request["next_use_request_index"],
        "prompt_token_sha256": token_list_sha(prompt),
        "ttft_ms": request_path_ttft,
        "target_ttft_ms": result["ttft_ms"],
        "elapsed_ms": request_path_elapsed,
        "target_elapsed_ms": result["elapsed_ms"],
        "seed_head_ms": seed_ms,
        "cached_tokens": result["cached_tokens"],
        "expected_reusable_prefix_tokens": expected_cached,
        "output_ids": result["output_ids"],
        **outcome,
        "request_outcome_observations": request_observations,
        "terminal_reason_observations": terminal_observations,
        "metrics": cross_store_metrics(
            before_text=before_text,
            after_text=after_text,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
        ),
        "memory_footprint_after": memory_footprint(
            after_snapshot,
            bytes_per_token=bytes_per_token,
        ),
    }


def run_arm(
    *,
    args: argparse.Namespace,
    context,
    workload: Mapping[str, Any],
    clean_baseline: Mapping[str, float],
    arm: str,
    repeat_index: int,
    measured: bool,
    bytes_per_token: int,
) -> dict[str, Any]:
    approximate_arm = arm in {"R0", "R4-like-5x"}
    _, pre_reset = full_reset(
        port=args.port,
        clean_baseline=clean_baseline,
        strict=False,
    )
    before_text = metric_text(args.port)
    before_snapshot = metric_snapshot(args.port)
    diagnostic_error = None
    diagnostic_error_stage = None
    diagnostic_request_index = None
    records = []
    try:
        setup = setup_arm(
            args=args,
            workload=workload,
            arm=arm,
            repeat_index=repeat_index,
            bytes_per_token=bytes_per_token,
        )
    except requests.HTTPError as error:
        if arm != "R4-like-5x":
            raise
        setup = {
            "arm": arm,
            "profile": "r4_like",
            "representation_kinds": list(
                REPRESENTATION_PROFILES["r4_like"]["representation_kinds"]
            ),
            "representation_multiplicity": 5,
            "rows": [],
            "materialize_ms": 0.0,
            "register_ms": 0.0,
            "setup_ms": 0.0,
            "registration_failed": True,
            "registration_failed_by_object": {},
            "victim_sequence": [],
        }
        diagnostic_error = f"{type(error).__name__}: {error}"
        diagnostic_error_stage = "setup"
    else:
        if not (arm == "R4-like-5x" and setup["registration_failed"]):
            for request in workload["request_order"]:
                try:
                    records.append(
                        run_request(
                            args=args,
                            workload=workload,
                            request=request,
                            arm=arm,
                            repeat_index=repeat_index,
                            measured=measured,
                            bytes_per_token=bytes_per_token,
                            registration_failed=(
                                setup["registration_failed_by_object"].get(
                                    request["object_id"], False
                                )
                            ),
                        )
                    )
                except requests.HTTPError as error:
                    if arm != "R4-like-5x":
                        raise
                    diagnostic_error = f"{type(error).__name__}: {error}"
                    diagnostic_error_stage = "request"
                    diagnostic_request_index = int(request["request_index"])
                    break

    after_text = metric_text(args.port)
    after_snapshot = metric_snapshot(args.port)
    metrics = cross_store_metrics(
        before_text=before_text,
        after_text=after_text,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
    )
    victim_sequence = list(setup["victim_sequence"])
    for record in records:
        for victim in record["metrics"]["victim_evict_bytes"]["rows"]:
            victim_sequence.append(
                {
                    "event_index": int(record["request_index"]),
                    "trigger_phase": record["phase"],
                    "trigger_object_id": record["object_id"],
                    **victim,
                }
            )
    if diagnostic_error is not None and not victim_sequence:
        for victim in metrics["victim_evict_bytes"]["rows"]:
            victim_sequence.append(
                {
                    "event_index": -1,
                    "trigger_phase": f"{diagnostic_error_stage}_error",
                    "trigger_object_id": (
                        "unknown"
                        if diagnostic_request_index is None
                        else workload["request_order"][diagnostic_request_index][
                            "object_id"
                        ]
                    ),
                    **victim,
                }
            )
    _, post_reset = full_reset(
        port=args.port,
        clean_baseline=clean_baseline,
        strict=approximate_arm,
    )
    diagnostic_status = (
        "diagnostic_unavailable"
        if arm == "R4-like-5x"
        and (setup["registration_failed"] or diagnostic_error is not None)
        else "available"
    )
    seed_head_ms = sum(float(record["seed_head_ms"]) for record in records)
    target_only_ms = sum(float(record["target_ttft_ms"]) for record in records)
    request_path_ms = sum(float(record["ttft_ms"]) for record in records)
    full_trace_wall_clock_ms = sum(float(record["elapsed_ms"]) for record in records)
    full_lifecycle_ms = float(setup["setup_ms"]) + request_path_ms
    return {
        "arm": arm,
        "repeat_index": repeat_index,
        "measured": measured,
        "pre_reset": pre_reset,
        "setup": setup,
        "records": records,
        "statistics": (
            summarize_workflow_records(records)
            if measured and records and arm != "R4-like-5x"
            else None
        ),
        "metrics": metrics,
        "memory_footprint_after": memory_footprint(
            after_snapshot,
            bytes_per_token=bytes_per_token,
        ),
        "victim_sequence": victim_sequence,
        "diagnostic_status": diagnostic_status,
        "diagnostic_error": diagnostic_error,
        "diagnostic_error_stage": diagnostic_error_stage,
        "request_diagnostic": {
            "status": (
                "unavailable"
                if diagnostic_error_stage == "request"
                else "not_applicable"
            ),
            "failed_request_index": diagnostic_request_index,
            "completed_request_records": len(records),
            "error": (
                diagnostic_error if diagnostic_error_stage == "request" else None
            ),
        },
        "representation_metadata": {
            "arm_label": arm,
            "profile": setup["profile"],
            "resident_multiplicity": setup["representation_multiplicity"],
            "executes_kvcomm": False,
            "performance_ranking_enabled": arm != "R4-like-5x",
        },
        "ledger": {
            "source_preparation_ms": setup["setup_ms"],
            "materialize_ms": setup["materialize_ms"],
            "register_copy_ms": setup["register_ms"],
            "seed_head_ms": seed_head_ms,
            "target_only_ms": target_only_ms,
            "request_path_ms": request_path_ms,
            "request_path_definition": "seed_head_ms + target_only_ms",
            "full_trace_wall_clock_ms": full_trace_wall_clock_ms,
            "full_lifecycle_ms": full_lifecycle_ms,
            "source_preparation_scope": "initial_setup_only",
            "adapter_ms": 0.0,
            "adapter_ms_note": "no adapter arm executes: R2 is disabled_not_comparable",
            "post_pressure_reseed_ms": 0.0,
            "post_pressure_reseed_note": "the scheduler never re-registers sources",
            "transfer_ms": "not_measured",
            "protocol_overhead_ms": "not_measured",
            "non_overlapping_components": [
                "source_preparation_ms",
                "request_path_ms",
            ],
            "non_overlapping_total_ms": full_lifecycle_ms,
        },
        "post_reset": post_reset,
    }


def execute(args: argparse.Namespace, context, run_id: str) -> dict[str, Any]:
    manifest = context.manifest
    setting = context.setting
    workload = w_workload(manifest)
    if workload["workload_id"] != "W-fixed40-v1":
        raise Phase7ContractError("scheduler requires W-fixed40-v1")
    server_template = manifest["server_template"]
    bytes_per_token = int(
        server_template["plugin_env"]["SGLANG_APPROX_KV_BYTES_PER_TOKEN"]
    )
    contract = scheduler_performance_contract(list(setting["arms"]))
    launch_started = time.perf_counter()
    server = launch_server(
        model=manifest["environment"]["model"],
        model_revision=manifest["environment"]["model_revision"],
        port=args.port,
        mem_fraction_static=float(setting["mem_fraction_static"]),
        chunked_prefill_size=int(setting["chunked_prefill_size"]),
        policy=str(setting["policy"]),
        log_path=args.log,
        plugin_env=server_template["plugin_env"],
        max_total_tokens=int(setting["max_total_tokens"]),
        server_seed=int(server_template["restart_seeds"][context.restart_index]),
        attention_backend=server_template["attention_backend"],
        sampling_backend=server_template["sampling_backend"],
    )
    try:
        wait_ready(
            server,
            port=args.port,
            timeout_s=args.server_start_timeout_s,
        )
        server_cold_start_ms = (time.perf_counter() - launch_started) * 1000.0
        startup_snapshot, clean_baseline_reset = full_reset(
            port=args.port,
            clean_baseline=None,
            strict=False,
        )
        clean_baseline = dict(startup_snapshot)
        capacity = observed_capacity(startup_snapshot, bytes_per_token)
        requested_capacity = int(setting["max_total_tokens"])
        tolerance = max(16, round(requested_capacity * 0.01))
        if abs(capacity["tokens"] - requested_capacity) > tolerance:
            raise RuntimeError(
                "scheduler capacity pin mismatch: "
                f"{capacity['tokens']} != {requested_capacity}"
            )
        logical_tokens = sum(
            int(item["logical_tokens"]) for item in workload["objects"]
        )
        realized_rho = logical_tokens / capacity["tokens"]
        if abs(realized_rho - float(setting["rho_logical_demand"])) > 0.05:
            raise RuntimeError(
                f"W logical rho drift: {realized_rho} != "
                f"{setting['rho_logical_demand']}"
            )

        warmup = []
        for warmup_index in range(int(setting["warmup_repeats"])):
            warmup.append(
                {
                    arm: run_arm(
                        args=args,
                        context=context,
                        workload=workload,
                        clean_baseline=clean_baseline,
                        arm=arm,
                        repeat_index=-(warmup_index + 1),
                        measured=False,
                        bytes_per_token=bytes_per_token,
                    )
                    for arm in setting["arms"]
                }
            )
        formal = []
        for repeat_index in range(int(setting["formal_repeats"])):
            order = formal_arm_order(setting, repeat_index)
            arms = {}
            for arm in order:
                arms[arm] = run_arm(
                    args=args,
                    context=context,
                    workload=workload,
                    clean_baseline=clean_baseline,
                    arm=arm,
                    repeat_index=repeat_index,
                    measured=True,
                    bytes_per_token=bytes_per_token,
                )
            formal.append(
                {
                    "repeat_index": repeat_index,
                    "arm_order": list(order),
                    "arms": arms,
                }
            )

        arm_records = {
            arm: [
                record for repeat in formal for record in repeat["arms"][arm]["records"]
            ]
            for arm in setting["arms"]
        }
        arm_statistics = {
            arm: summarize_workflow_records(records)
            for arm, records in arm_records.items()
            if records and arm != "R4-like-5x"
        }
        paired = None
        paired_per_repeat = []
        if contract["performance_ranking_enabled"]:
            paired = pair_scheduler_arms(
                arm_records["E0"],
                arm_records["R0"],
            )
            for repeat in formal:
                paired_per_repeat.append(
                    {
                        "repeat_index": repeat["repeat_index"],
                        "paired": pair_scheduler_arms(
                            repeat["arms"]["E0"]["records"],
                            repeat["arms"]["R0"]["records"],
                        ),
                    }
                )
        taxonomy_invalid = any(
            not record["taxonomy_valid"]
            for records in arm_records.values()
            for record in records
        )
        diagnostic_unavailable = any(
            arm_data["diagnostic_status"] == "diagnostic_unavailable"
            for repeat in formal
            for arm_data in repeat["arms"].values()
        )
        status = (
            "invalid"
            if taxonomy_invalid
            else ("inconclusive" if diagnostic_unavailable else "valid")
        )
        result = {
            "startup_snapshot": startup_snapshot,
            "clean_baseline_reset": clean_baseline_reset,
            "capacity": capacity,
            "logical_working_set_tokens": logical_tokens,
            "realized_logical_rho": realized_rho,
            "warmup": warmup,
            "formal": formal,
            "arm_statistics": arm_statistics,
            "paired_E0_R0": paired,
            "paired_per_repeat": paired_per_repeat,
            "performance_contract": contract,
            "server_cold_start_ms": server_cold_start_ms,
            "status": status,
            "machine": machine_manifest(),
            "server_argv": list(server.command),
            "plugin_env": server.plugin_env,
        }
    except Exception as error:
        raise Phase7RunError(
            error,
            server_argv=server.command,
            plugin_env=server.plugin_env,
        ) from error
    finally:
        stop_server(server)

    log_sha = file_sha256(args.log)
    all_records = [
        record
        for repeat in result["formal"]
        for arm in repeat["arms"].values()
        for record in arm["records"]
    ]
    outcome_counts = {
        outcome: sum(record["outcome"] == outcome for record in all_records)
        for outcome in manifest["outcome_taxonomy"]
    }
    reason_counts = {
        reason: sum(record["terminal_reason"] == reason for record in all_records)
        for reason in manifest["exclusive_terminal_reasons"]
    }
    segment_tokens_max = int(workload["segment_tokens_max"])
    segment_count = sum(
        math.ceil(int(item["logical_tokens"]) / segment_tokens_max)
        for item in workload["objects"]
    )
    artifact = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": "Phase7-scheduler",
        "source_git_sha": context.source["source_git_sha"],
        "source_tree_sha": context.source["source_tree_sha"],
        **pending_result_provenance(),
        "execution_envelope": context.envelope,
        "raw_sha256": "",
        "server_argv": result["server_argv"],
        "plugin_env": result["plugin_env"],
        "machine": result["machine"],
        "image_digest": manifest["environment"]["image_digest"],
        "requested_capacity": {
            "mode": setting["capacity_mode"],
            "tokens": setting["max_total_tokens"],
            "pages": None,
            "bytes": int(setting["max_total_tokens"]) * bytes_per_token,
        },
        "observed_capacity": result["capacity"],
        "crosses_chunk_boundary": False,
        "segment_count": segment_count,
        "segment_tokens_max": segment_tokens_max,
        "warmup_repeats": int(setting["warmup_repeats"]),
        "formal_repeats": int(setting["formal_repeats"]),
        "restarts": 1,
        "ledger": {
            "setup": {
                "server_cold_start_ms": result["server_cold_start_ms"],
                "per_repeat_per_arm": {
                    str(repeat["repeat_index"]): {
                        arm: data["ledger"]["source_preparation_ms"]
                        for arm, data in repeat["arms"].items()
                    }
                    for repeat in result["formal"]
                },
            },
            "materialization": "see formal[*].arms[*].setup",
            "recovery": "see formal[*].arms[R0].records",
            "scheduler": {
                "arm_statistics": result["arm_statistics"],
                "paired_E0_R0": result["paired_E0_R0"],
                "paired_per_repeat": result["paired_per_repeat"],
            },
            "transfer": "not_measured",
            "temporary_peak": {
                "semantics": "arm_high_water_since_last_full_reset",
                "per_repeat_per_arm": {
                    str(repeat["repeat_index"]): {
                        arm: data["metrics"]["arm_interval_peak_device_bytes"]
                        for arm, data in repeat["arms"].items()
                    }
                    for repeat in result["formal"]
                },
            },
        },
        "rho": {
            **rho_payload(),
            "rho_logical_demand": setting["rho_logical_demand"],
            "rho_realization": setting["rho_realization"],
            "logical_working_set_tokens": result["logical_working_set_tokens"],
            "realized_logical_rho": result["realized_logical_rho"],
            "arm_interval_peak_by_repeat_arm": {
                "semantics": "arm_high_water_since_last_full_reset",
                "values": {
                    str(repeat["repeat_index"]): {
                        arm: data["metrics"]["arm_interval_peak_device_bytes"]
                        for arm, data in repeat["arms"].items()
                    }
                    for repeat in result["formal"]
                },
            },
        },
        "status": result["status"],
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
        "server_log_sha256": log_sha,
        "outcome": {
            "taxonomy": manifest["outcome_taxonomy"],
            "counts": outcome_counts,
            "exclusive_terminal_reasons": manifest["exclusive_terminal_reasons"],
            "terminal_reason_counts": reason_counts,
        },
        "reset": {
            "startup": result["clean_baseline_reset"],
            "formal": {
                str(repeat["repeat_index"]): {
                    arm: {
                        "pre": data["pre_reset"],
                        "post": data["post_reset"],
                    }
                    for arm, data in repeat["arms"].items()
                }
                for repeat in result["formal"]
            },
        },
        "provenance": {
            "manifest_path": str(context.manifest_path.resolve()),
            "manifest_file_sha256": context.manifest_file_sha256,
            "implementation": manifest["implementation"],
            "runner_sha256": context.runner_sha256,
            "source": context.source,
        },
        "workload": {
            "workload_id": workload["workload_id"],
            "workload_manifest_sha256": workload["manifest_sha256"],
            "request_order_sha256": workload["request_order_sha256"],
            "request_count": len(workload["request_order"]),
            "object_count": len(workload["objects"]),
        },
        "early_stop": {
            "applicable_rule": "ES-ENGINEERING",
            "r0_mde_applies": False,
        },
        "performance_contract": result["performance_contract"],
        "warmup": result["warmup"],
        "formal": result["formal"],
        "arm_statistics": result["arm_statistics"],
        "paired_E0_R0": result["paired_E0_R0"],
        "paired_per_repeat": result["paired_per_repeat"],
    }
    finalize_artifact_hash(artifact)
    validate_phase7_artifact(artifact)
    return artifact


def failure_artifact(
    *,
    args: argparse.Namespace,
    context,
    run_id: str,
    error: Exception,
) -> dict[str, Any]:
    manifest = context.manifest
    setting = context.setting
    bytes_per_token = int(
        manifest["server_template"]["plugin_env"]["SGLANG_APPROX_KV_BYTES_PER_TOKEN"]
    )
    try:
        workload = w_workload(manifest)
    except Phase7ContractError:
        workload = None
    segment_tokens_max = (
        None if workload is None else int(workload["segment_tokens_max"])
    )
    segment_count = (
        None
        if workload is None
        else sum(
            math.ceil(int(item["logical_tokens"]) / segment_tokens_max)
            for item in workload["objects"]
        )
    )
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": "Phase7-scheduler",
        "source_git_sha": context.source["source_git_sha"],
        "source_tree_sha": context.source["source_tree_sha"],
        **pending_result_provenance(),
        "execution_envelope": context.envelope,
        "raw_sha256": "",
        "server_argv": list(getattr(error, "server_argv", ())),
        "plugin_env": dict(
            getattr(
                error,
                "plugin_env",
                manifest["server_template"]["plugin_env"],
            )
        ),
        "machine": {
            "runtime_probe": "unavailable_due_to_failure",
            "expected_gpu": manifest["environment"]["gpu"],
            "expected_driver": manifest["environment"]["driver"],
        },
        "image_digest": manifest["environment"]["image_digest"],
        "requested_capacity": {
            "mode": setting["capacity_mode"],
            "tokens": setting["max_total_tokens"],
            "pages": None,
            "bytes": int(setting["max_total_tokens"]) * bytes_per_token,
        },
        "observed_capacity": {"tokens": None, "pages": None, "bytes": None},
        "crosses_chunk_boundary": False,
        "segment_count": segment_count,
        "segment_tokens_max": segment_tokens_max,
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
            **rho_payload(),
            "rho_logical_demand": setting["rho_logical_demand"],
        },
        "status": "invalid",
        "manifest_revision": manifest["manifest_revision"],
        "preregistered_manifest_sha256": manifest["preregistered_manifest_sha256"],
        "manifest_file_sha256": context.manifest_file_sha256,
        "plan": manifest["plan"],
        "setting_id": setting["setting_id"],
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
        "provenance": {
            "manifest_path": str(context.manifest_path.resolve()),
            "implementation": manifest["implementation"],
            "runner_sha256": context.runner_sha256,
            "source": context.source,
        },
        "performance_contract": scheduler_performance_contract(list(setting["arms"])),
        "execution_status": execution_status(error),
        "error": f"{type(error).__name__}: {error}",
    }
    finalize_artifact_hash(payload)
    validate_phase7_artifact(payload)
    return payload


def main() -> int:
    args = parse_args()
    ensure_artifact_path_layout(
        output=args.output,
        log=args.log,
        central_log=args.central_log,
    )
    context = load_execution_context(
        manifest_path=args.manifest,
        setting_id=args.setting_id,
        restart_index=args.restart_index,
        runner_key=RUNNER_KEY,
        runner_module=SCHEDULER_RUNNER,
        runner_file=Path(__file__),
    )
    run_id = (
        f"p7-scheduler-{args.setting_id}-r{args.restart_index}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    append_jsonl(
        args.central_log,
        {
            "run_id": run_id,
            "phase": "Phase7-scheduler",
            "status": "running",
            "setting_id": args.setting_id,
            "restart_index": args.restart_index,
            "manifest_sha256": context.manifest["preregistered_manifest_sha256"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        artifact = execute(args, context, run_id)
        write_json(args.output, artifact)
        append_jsonl(
            args.central_log,
            {
                "run_id": run_id,
                "phase": "Phase7-scheduler",
                "status": "completed",
                "setting_id": args.setting_id,
                "restart_index": args.restart_index,
                "raw_sha256": artifact["raw_sha256"],
                "artifact_status": artifact["status"],
                "output": str(args.output.resolve()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return 0
    except Exception as error:
        failure = failure_artifact(
            args=args,
            context=context,
            run_id=run_id,
            error=error,
        )
        write_json(args.output, failure)
        append_jsonl(
            args.central_log,
            {
                "run_id": run_id,
                "phase": "Phase7-scheduler",
                "status": "failed",
                "setting_id": args.setting_id,
                "restart_index": args.restart_index,
                "raw_sha256": failure["raw_sha256"],
                "output": str(args.output.resolve()),
                "error": failure["error"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
