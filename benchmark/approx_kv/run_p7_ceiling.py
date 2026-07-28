#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from benchmark.approx_kv.metrics import max_total_num_tokens
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
    CEILING_RUNNER,
    Phase7ContractError,
    Phase7RunError,
    a8_tokens,
    classify_request_outcome,
    cross_store_metrics,
    ensure_artifact_path_layout,
    filler_pool_tokens,
    finalize_artifact_hash,
    load_execution_context,
    memory_footprint,
    observed_capacity,
    pending_result_provenance,
    phase7_reset_invariant,
    registration_outcome_observations,
    request_outcome_observations,
    required_resident_tokens,
    rho_payload,
    select_filler_prefix,
    terminal_reason_observations,
    validate_phase7_artifact,
)
from benchmark.approx_kv.phase7.statistics import (
    compute_amortization,
    same_context_canary,
    summarize_ceiling_repeats,
)
from benchmark.approx_kv.run_cl1_qualification import chunks

RUNNER_KEY = "ceiling"
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
    parser.add_argument("--mde-gate-passed", action="store_true")
    return parser.parse_args()


def formal_arm_order(setting: Mapping[str, Any], repeat_index: int) -> tuple[str, ...]:
    order = setting["arm_order_by_repeat"].get(str(repeat_index))
    if order is None or sorted(order) != sorted(setting["arms"]):
        raise Phase7ContractError(
            f"{setting['setting_id']}: invalid arm order for repeat {repeat_index}"
        )
    return tuple(order)


def ceiling_early_stop_contract(
    setting: Mapping[str, Any],
    *,
    restart_index: int,
    mde_gate_passed: bool,
) -> dict[str, Any]:
    supplement = restart_index in setting["supplement_restarts"]
    gate = setting.get("supplement_gate")
    if supplement and gate == "ES-R0-MDE" and not mde_gate_passed:
        raise Phase7ContractError(
            f"{setting['setting_id']}: restart {restart_index} requires "
            "--mde-gate-passed"
        )
    return {
        "supplement": supplement,
        "supplement_gate": gate,
        "mde_gate_required": supplement and gate == "ES-R0-MDE",
        "mde_gate_passed": (
            mde_gate_passed if supplement and gate == "ES-R0-MDE" else None
        ),
        "r0_mde_applies": gate == "ES-R0-MDE",
        "engineering_stop_always_applies": True,
    }


def source_segments(
    *,
    workload_id: str,
    body_tokens: int,
    segment_tokens: int,
) -> list[dict[str, Any]]:
    segments = []
    cursor = 0
    for segment_index in range(math.ceil(body_tokens / segment_tokens)):
        length = min(segment_tokens, body_tokens - cursor)
        segments.append(
            {
                "content_hash": f"p7:{workload_id}:body:seg{segment_index}",
                "target_start": HEADER_TOKENS + cursor,
                "length": length,
                "object_id": f"p7:{workload_id}:canonical:seg{segment_index}",
                "object_kind": "canonical_base",
                "dense_cost_ms": 12.0,
                "recovery_cost_ms": 2.0,
                "next_use_ordinal": 0,
                "retired": False,
                "residency": "device",
            }
        )
        cursor += length
    return segments


def approx_metadata(
    *,
    workload_id: str,
    body_tokens: int,
    segment_tokens: int,
    operation: str,
    pin_until_reset: bool = False,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "operation": operation,
        "model_fingerprint": "phase7-qwen3-sm75",
        "cache_dtype": "float16",
        "segments": source_segments(
            workload_id=workload_id,
            body_tokens=body_tokens,
            segment_tokens=segment_tokens,
        ),
    }
    if pin_until_reset:
        metadata["pin_until_reset"] = True
    return metadata


def cache_protection(
    *,
    object_id: str,
    protected_tokens: int,
    object_kind: str,
    bytes_per_token: int,
    current_step: int | None,
    next_use_request_step: int | None,
    retired: bool,
) -> dict[str, Any]:
    return {
        "object_id": object_id,
        "protected_tokens": protected_tokens,
        "resident_bytes": protected_tokens * bytes_per_token,
        "dense_cost_ms": 12.0,
        "recovery_cost_ms": 2.0,
        "current_step": current_step,
        "next_use_request_step": next_use_request_step,
        "workflow_stage": "a8",
        "object_kind": object_kind,
        "recoverable_from_lower_tier": False,
        "retired": retired,
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


def materialize_source(
    *,
    args: argparse.Namespace,
    workload: Mapping[str, Any],
    arm: str,
    register: bool,
    bytes_per_token: int,
    pass_name: str,
    segment_tokens: int,
    pin_until_reset: bool,
) -> dict[str, Any]:
    body = list(workload["body"])
    header = list(workload["source_header"])
    workload_id = workload["spec"]["workload_id"]
    namespace = f"p7-source:{workload_id}:{arm}"
    rows = []
    materialize_ms = 0.0
    register_ms = 0.0
    registration_failed = False
    cursor = 0
    body_chunks = chunks(body, segment_tokens)
    for segment_index, chunk in enumerate(body_chunks):
        cursor += len(chunk)
        prompt = header + body[:cursor] + [60_000 + segment_index]
        protection = cache_protection(
            object_id=f"p7-source-exact:{workload_id}:{arm}",
            protected_tokens=HEADER_TOKENS + cursor,
            object_kind="canonical_base",
            bytes_per_token=bytes_per_token,
            current_step=None,
            next_use_request_step=0 if register else None,
            retired=not register,
        )
        materialized = generate(
            port=args.port,
            input_ids=prompt,
            max_new_tokens=1,
            custom_params={"cache_protection": protection},
            extra_key=namespace,
            timeout_s=args.request_timeout_s,
        )
        materialize_ms += float(materialized["elapsed_ms"])
        row: dict[str, Any] = {
            "segment_index": segment_index,
            "pass": pass_name,
            "prompt_tokens": len(prompt),
            "materialize_cached_tokens": materialized["cached_tokens"],
            "materialize_ms": materialized["elapsed_ms"],
        }
        if register:
            metadata = approx_metadata(
                workload_id=workload_id,
                body_tokens=len(body),
                segment_tokens=segment_tokens,
                operation="register",
                pin_until_reset=pin_until_reset,
            )
            metadata["segments"] = [metadata["segments"][segment_index]]
            before = metric_text(args.port)
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
            after = metric_text(args.port)
            observations = registration_outcome_observations(before, after)
            expected_cached = HEADER_TOKENS + cursor
            cache_verified = registered["cached_tokens"] >= expected_cached
            if observations["verification"] == "unknown" and cache_verified:
                observations["verification"] = "indirectly_verified"
                observations["indirect_evidence"] = (
                    "registered request retained the full causal source prefix"
                )
            failed = (
                observations["registration_failed"]
                or not cache_verified
                or len(observations["positive_outcomes"]) > 1
            )
            registration_failed = registration_failed or failed
            register_ms += float(registered["elapsed_ms"])
            row.update(
                {
                    "register_cached_tokens": registered["cached_tokens"],
                    "expected_register_cached_tokens": expected_cached,
                    "register_ms": registered["elapsed_ms"],
                    "pin_until_reset": pin_until_reset,
                    "registration_observations": observations,
                    "registration_failed": failed,
                }
            )
        rows.append(row)
    return {
        "pass": pass_name,
        "register": register,
        "namespace": namespace,
        "causal_prefix_registration": True,
        "segment_tokens": segment_tokens,
        "segment_count": len(rows),
        "pin_until_reset": pin_until_reset if register else False,
        "rows": rows,
        "materialize_ms": materialize_ms,
        "register_ms": register_ms,
        "source_preparation_ms": materialize_ms + register_ms,
        "registration_failed": registration_failed,
    }


def source_pin_state(
    *,
    args: argparse.Namespace,
    arm: str,
    expected_segments: int,
    pin_until_reset: bool,
    stage: str,
) -> dict[str, Any]:
    """Assert the persistent registration lease survived filler pressure."""
    if arm != "R0":
        return {
            "arm": arm,
            "stage": stage,
            "mechanism": "not_applicable",
            "verification": "not_applicable",
        }
    snapshot = metric_snapshot(args.port)
    leases = snapshot.get("sglang:approx_kv_store_leases")
    records = snapshot.get("sglang:approx_kv_store_records")
    state = {
        "arm": arm,
        "stage": stage,
        "mechanism": "persistent_registration_lease_until_reset",
        "pin_until_reset": pin_until_reset,
        "expected_segments": expected_segments,
        "observed_store_leases": leases,
        "observed_store_records": records,
        "verification": (
            "direct" if leases is not None and records is not None else "unknown"
        ),
    }
    if state["verification"] != "direct":
        raise RuntimeError(f"persistent source lease is unobservable: {state}")
    if float(leases) < expected_segments or float(records) < expected_segments:
        raise RuntimeError(f"persistent source lease did not survive pressure: {state}")
    return state


def materialize_fillers(
    *,
    args: argparse.Namespace,
    selected: list[Mapping[str, Any]],
    arm: str,
    repeat_index: int,
    bytes_per_token: int,
) -> dict[str, Any]:
    rows = []
    total_ms = 0.0
    for item in selected:
        prompt = list(item["token_ids"]) + [
            120_000 + int(item["filler_id"].removeprefix("p7-filler-"))
        ]
        result = generate(
            port=args.port,
            input_ids=prompt,
            max_new_tokens=1,
            custom_params={
                "cache_protection": cache_protection(
                    object_id=f"p7-filler:{item['filler_id']}",
                    protected_tokens=int(item["tokens"]),
                    object_kind="filler",
                    bytes_per_token=bytes_per_token,
                    current_step=None,
                    next_use_request_step=None,
                    retired=bool(item["retired"]),
                )
            },
            extra_key=f"p7-filler:{arm}:{repeat_index}:{item['filler_id']}",
            timeout_s=args.request_timeout_s,
        )
        total_ms += float(result["elapsed_ms"])
        rows.append(
            {
                "filler_id": item["filler_id"],
                "tokens": item["tokens"],
                "retired": item["retired"],
                "elapsed_ms": result["elapsed_ms"],
                "cached_tokens": result["cached_tokens"],
            }
        )
    return {"rows": rows, "total_ms": total_ms}


def run_target(
    *,
    args: argparse.Namespace,
    workload: Mapping[str, Any],
    target: Mapping[str, Any],
    arm: str,
    target_index: int,
    bytes_per_token: int,
    registration_failed: bool,
    selected_filler_tokens: int,
    capacity_tokens: int,
    segment_tokens: int,
) -> dict[str, Any]:
    spec = target["spec"]
    namespace = spec["extra_keys_by_arm"][arm]
    header = list(target["header"])
    body = list(target["body"])
    seed = None
    if arm in {"E0", "R0"}:
        seed = generate(
            port=args.port,
            input_ids=header,
            max_new_tokens=1,
            custom_params={
                "cache_protection": cache_protection(
                    object_id=f"p7-target:{spec['target_id']}:{arm}",
                    protected_tokens=HEADER_TOKENS,
                    object_kind="exact_variant",
                    bytes_per_token=bytes_per_token,
                    current_step=target_index,
                    next_use_request_step=target_index,
                    retired=False,
                )
            },
            extra_key=namespace,
            timeout_s=args.request_timeout_s,
        )
    before_text = metric_text(args.port)
    before_snapshot = metric_snapshot(args.port)
    custom_params: dict[str, Any] = {
        "cache_protection": cache_protection(
            object_id=f"p7-target:{spec['target_id']}:{arm}",
            protected_tokens=HEADER_TOKENS + len(body),
            object_kind="exact_variant",
            bytes_per_token=bytes_per_token,
            current_step=target_index,
            next_use_request_step=None,
            retired=True,
        )
    }
    if arm == "R0":
        custom_params["approx_kv"] = approx_metadata(
            workload_id=workload["spec"]["workload_id"],
            body_tokens=len(body),
            segment_tokens=segment_tokens,
            operation="reuse",
        )
    result = stream_generate(
        port=args.port,
        input_ids=list(target["prompt"]),
        max_new_tokens=1,
        custom_params=custom_params,
        extra_key=namespace,
        timeout_s=args.request_timeout_s,
    )
    after_text = metric_text(args.port)
    after_snapshot = metric_snapshot(args.port)
    expected_cached = {
        "D0": 0,
        "E0": HEADER_TOKENS,
        "R0": HEADER_TOKENS + len(body),
    }[arm]
    request_observations = (
        request_outcome_observations(before_text, after_text, operation="reuse")
        if arm == "R0"
        else None
    )
    terminal_observations = (
        terminal_reason_observations(before_text, after_text) if arm == "R0" else None
    )
    outcome = classify_request_outcome(
        arm=arm,
        cached_tokens=int(result["cached_tokens"]),
        expected_cached_tokens=expected_cached,
        request_observations=request_observations,
        terminal_observations=terminal_observations,
        registration_failed=registration_failed,
    )
    cache_path_matched = (
        int(result["cached_tokens"]) == 0
        if arm == "D0"
        else int(result["cached_tokens"]) >= expected_cached
    )
    outcome["expected_outcome"] = (
        bool(outcome["expected_outcome"]) and cache_path_matched
    )
    observed_resident = required_resident_tokens(after_snapshot)
    seed_ms = 0.0 if seed is None else float(seed["elapsed_ms"])
    return {
        "target_id": spec["target_id"],
        "target_index": target_index,
        "extra_key": namespace,
        "cached_tokens": result["cached_tokens"],
        "expected_cached_tokens": expected_cached,
        "cache_path_matched": cache_path_matched,
        "ttft_ms": result["ttft_ms"],
        "elapsed_ms": result["elapsed_ms"],
        "output_ids": result["output_ids"],
        "seed_head_ms": seed_ms,
        "target_only_ms": float(result["ttft_ms"]),
        "request_path_ms": seed_ms + float(result["ttft_ms"]),
        **outcome,
        "request_outcome_observations": request_observations,
        "terminal_reason_observations": terminal_observations,
        "metrics": cross_store_metrics(
            before_text=before_text,
            after_text=after_text,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
        ),
        "pressure_trajectory": {
            "target_index": target_index,
            "logical_components_tokens": {
                "filler": selected_filler_tokens,
                "prior_targets": (target_index + 1) * (HEADER_TOKENS + len(body)),
                "pinned_source": len(body) if arm == "R0" else 0,
            },
            "observed_resident_tokens": observed_resident,
            "rho_resident": observed_resident / capacity_tokens,
            "memory_footprint": memory_footprint(
                after_snapshot,
                bytes_per_token=bytes_per_token,
            ),
        },
    }


def run_canary(
    *,
    args: argparse.Namespace,
    workload: Mapping[str, Any],
    bytes_per_token: int,
    segment_tokens: int,
) -> dict[str, Any]:
    spec = workload["spec"]["same_context_canary"]
    header = list(workload["source_header"])
    prompt = header + list(workload["body"]) + list(workload["canary_suffix"])
    recovery_namespace = f"{spec['extra_key']}-R0"
    dense_namespace = f"{spec['extra_key']}-dense-reference"
    recovery_seed = generate(
        port=args.port,
        input_ids=header,
        max_new_tokens=1,
        extra_key=recovery_namespace,
        timeout_s=args.request_timeout_s,
    )
    before = metric_text(args.port)
    recovery = generate(
        port=args.port,
        input_ids=prompt,
        max_new_tokens=8,
        custom_params={
            "cache_protection": cache_protection(
                object_id=f"p7-canary:{spec['target_id']}:R0",
                protected_tokens=HEADER_TOKENS + len(workload["body"]),
                object_kind="exact_variant",
                bytes_per_token=bytes_per_token,
                current_step=8,
                next_use_request_step=None,
                retired=True,
            ),
            "approx_kv": approx_metadata(
                workload_id=workload["spec"]["workload_id"],
                body_tokens=len(workload["body"]),
                segment_tokens=segment_tokens,
                operation="reuse",
            ),
        },
        extra_key=recovery_namespace,
        timeout_s=args.request_timeout_s,
    )
    after = metric_text(args.port)
    dense_seed = generate(
        port=args.port,
        input_ids=header,
        max_new_tokens=1,
        extra_key=dense_namespace,
        timeout_s=args.request_timeout_s,
    )
    dense = generate(
        port=args.port,
        input_ids=prompt,
        max_new_tokens=8,
        extra_key=dense_namespace,
        timeout_s=args.request_timeout_s,
    )
    comparison = same_context_canary(
        dense["output_ids"],
        recovery["output_ids"],
    )
    request_observations = request_outcome_observations(
        before, after, operation="reuse"
    )
    terminal_observations = terminal_reason_observations(before, after)
    recovery_outcome = classify_request_outcome(
        arm="R0",
        cached_tokens=int(recovery["cached_tokens"]),
        expected_cached_tokens=HEADER_TOKENS + len(workload["body"]),
        request_observations=request_observations,
        terminal_observations=terminal_observations,
        expected_outcomes=(
            "approximate_gpu_recovery",
            "approximate_recovery_failed_dense",
        ),
    )
    return {
        **comparison,
        "placement": "after_target_8_before_reset",
        "included_in_amortization": False,
        "execution_order": ["recovery", "dense_reference"],
        "recovery_seed_ms": recovery_seed["elapsed_ms"],
        "dense_seed_ms": dense_seed["elapsed_ms"],
        "recovery_elapsed_ms": recovery["elapsed_ms"],
        "dense_elapsed_ms": dense["elapsed_ms"],
        "total_ms": (
            float(recovery_seed["elapsed_ms"])
            + float(recovery["elapsed_ms"])
            + float(dense_seed["elapsed_ms"])
            + float(dense["elapsed_ms"])
        ),
        "recovery_cached_tokens": recovery["cached_tokens"],
        "dense_cached_tokens": dense["cached_tokens"],
        "recovery_outcome": recovery_outcome,
    }


def run_arm(
    *,
    args: argparse.Namespace,
    context,
    workload: Mapping[str, Any],
    fillers: list[Mapping[str, Any]],
    clean_baseline: Mapping[str, float],
    arm: str,
    repeat_index: int,
    measured: bool,
    bytes_per_token: int,
) -> dict[str, Any]:
    segment_tokens = int(workload["segment_tokens_max"])
    pin_until_reset = workload["source_pin_until_reset"] is True
    approximate_arm = arm == "R0"
    _, pre_reset = full_reset(
        port=args.port,
        clean_baseline=clean_baseline,
        strict=False,
    )
    arm_before_text = metric_text(args.port)
    arm_before_snapshot = metric_snapshot(args.port)
    source_initial = None
    if arm in {"D0", "R0"}:
        source_initial = materialize_source(
            args=args,
            workload=workload,
            arm=arm,
            register=approximate_arm,
            bytes_per_token=bytes_per_token,
            pass_name="initial",
            segment_tokens=segment_tokens,
            pin_until_reset=pin_until_reset,
        )
    setup_snapshot = metric_snapshot(args.port)
    setup_resident = required_resident_tokens(setup_snapshot)
    selection = select_filler_prefix(
        fillers,
        capacity_tokens=max_total_num_tokens(setup_snapshot),
        rho_logical_demand=float(context.setting["rho_logical_demand"]),
        setup_resident_tokens=setup_resident,
    )
    filler_result = materialize_fillers(
        args=args,
        selected=selection["selected"],
        arm=arm,
        repeat_index=repeat_index,
        bytes_per_token=bytes_per_token,
    )
    expected_segments = (
        0 if source_initial is None else int(source_initial["segment_count"])
    )
    post_pressure_pin = source_pin_state(
        args=args,
        arm=arm,
        expected_segments=expected_segments,
        pin_until_reset=pin_until_reset,
        stage="after_pressure_before_targets",
    )
    registration_failed = bool(
        source_initial is not None and source_initial["registration_failed"]
    )
    capacity = max_total_num_tokens(setup_snapshot)
    targets = [
        run_target(
            args=args,
            workload=workload,
            target=target,
            arm=arm,
            target_index=index,
            bytes_per_token=bytes_per_token,
            registration_failed=registration_failed,
            selected_filler_tokens=int(selection["selected_filler_tokens"]),
            capacity_tokens=capacity,
            segment_tokens=segment_tokens,
        )
        for index, target in enumerate(workload["targets"])
    ]
    post_sequence_pin = source_pin_state(
        args=args,
        arm=arm,
        expected_segments=expected_segments,
        pin_until_reset=pin_until_reset,
        stage="after_target_8_before_reset",
    )
    canary = (
        run_canary(
            args=args,
            workload=workload,
            bytes_per_token=bytes_per_token,
            segment_tokens=segment_tokens,
        )
        if measured and approximate_arm
        else None
    )
    arm_after_text = metric_text(args.port)
    arm_after_snapshot = metric_snapshot(args.port)
    _, post_reset = full_reset(
        port=args.port,
        clean_baseline=clean_baseline,
        strict=approximate_arm,
    )
    source_materialize_ms = (
        0.0 if source_initial is None else float(source_initial["materialize_ms"])
    )
    source_register_ms = (
        0.0 if source_initial is None else float(source_initial["register_ms"])
    )
    source_preparation_ms = source_materialize_ms + source_register_ms
    seed_ms = sum(float(row["seed_head_ms"]) for row in targets)
    target_only_ms = sum(float(row["target_only_ms"]) for row in targets)
    request_path_ms = sum(float(row["request_path_ms"]) for row in targets)
    canary_ms = 0.0 if canary is None else float(canary["total_ms"])
    return {
        "arm": arm,
        "repeat_index": repeat_index,
        "measured": measured,
        "pre_reset": pre_reset,
        "source_initial": source_initial,
        "source_pin_contract": {
            "source_must_remain_available_for_sequence": approximate_arm,
            "persistent_runner_visible_lease": approximate_arm,
            "mechanism": (
                "opt-in persistent registration lease until reset plus the "
                "per-reuse runtime lease"
                if approximate_arm
                else "not_applicable"
            ),
            "pin_until_reset": pin_until_reset if approximate_arm else False,
            "post_pressure_reregistration": False,
            "verification": (
                "store lease and record gauges are asserted after pressure and "
                "after target 8; the arm-final full reset releases the lease"
                if approximate_arm
                else "not_applicable"
            ),
            "after_pressure": post_pressure_pin,
            "after_target_8": post_sequence_pin,
        },
        "pressure": {
            **{key: value for key, value in selection.items() if key != "selected"},
            "setup_memory_footprint": memory_footprint(
                setup_snapshot,
                bytes_per_token=bytes_per_token,
            ),
            "post_sequence_memory_footprint": memory_footprint(
                arm_after_snapshot,
                bytes_per_token=bytes_per_token,
            ),
        },
        "fillers": filler_result,
        "targets": targets,
        "same_context_canary": canary,
        "metrics": cross_store_metrics(
            before_text=arm_before_text,
            after_text=arm_after_text,
            before_snapshot=arm_before_snapshot,
            after_snapshot=arm_after_snapshot,
        ),
        "ledger": {
            "source_preparation_ms": source_preparation_ms,
            "source_preparation_scope": "initial_setup_only",
            "materialize_ms": source_materialize_ms,
            "register_copy_ms": source_register_ms,
            "adapter_ms": 0.0,
            "adapter_ms_note": "no adapter arm executes: R2 is disabled_not_comparable",
            "post_pressure_reseed_ms": 0.0,
            "post_pressure_reseed_note": (
                "the persistent registration lease replaces post-pressure "
                "re-registration; no reseed work is performed"
            ),
            "transfer_ms": "not_measured",
            "seed_head_ms": seed_ms,
            "target_only_ms": target_only_ms,
            "request_path_ms": request_path_ms,
            "request_path_definition": "seed_head_ms + target_only_ms",
            "pressure_fill_ms": float(filler_result["total_ms"]),
            "cold_start_ms": "reported_at_server_level",
            "protocol_overhead_ms": "not_measured",
            "same_context_canary_ms": canary_ms,
            "non_overlapping_components": [
                "source_preparation_ms",
                "pressure_fill_ms",
                "request_path_ms",
                "same_context_canary_ms",
            ],
            "full_lifecycle_ms": (
                source_preparation_ms
                + float(filler_result["total_ms"])
                + request_path_ms
                + canary_ms
            ),
        },
        "post_reset": post_reset,
    }


def frozen_segment_tokens(manifest: Mapping[str, Any]) -> int:
    segment_tokens = (
        manifest.get("workloads", {}).get("A8", {}).get("segment_tokens_max")
    )
    if not isinstance(segment_tokens, int) or segment_tokens <= 0:
        raise Phase7ContractError("A8 workload does not freeze segment_tokens_max")
    return segment_tokens


def reported_segment_count(manifest: Mapping[str, Any], body_tokens: int) -> int | None:
    """Segment count for a failure artifact, or ``None`` when unpinned."""
    try:
        return math.ceil(body_tokens / frozen_segment_tokens(manifest))
    except Phase7ContractError:
        return None


def execute(args: argparse.Namespace, context, run_id: str) -> dict[str, Any]:
    manifest = context.manifest
    setting = context.setting
    if "R2" in setting["arms"]:
        raise Phase7ContractError(
            "R2 execution is unavailable: disposition is disabled_not_comparable"
        )
    workload = a8_tokens(manifest, body_tokens=int(setting["body_tokens"]))
    fillers = filler_pool_tokens(manifest)
    server_template = manifest["server_template"]
    bytes_per_token = int(
        server_template["plugin_env"]["SGLANG_APPROX_KV_BYTES_PER_TOKEN"]
    )
    restart_seeds = server_template["restart_seeds"]
    early_stop = ceiling_early_stop_contract(
        setting,
        restart_index=context.restart_index,
        mde_gate_passed=bool(args.mde_gate_passed),
    )
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
        max_total_tokens=setting["max_total_tokens"],
        server_seed=int(restart_seeds[context.restart_index]),
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
        warmup = []
        for warmup_index in range(int(setting["warmup_repeats"])):
            warmup.append(
                {
                    arm: run_arm(
                        args=args,
                        context=context,
                        workload=workload,
                        fillers=fillers,
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
                    fillers=fillers,
                    clean_baseline=clean_baseline,
                    arm=arm,
                    repeat_index=repeat_index,
                    measured=True,
                    bytes_per_token=bytes_per_token,
                )
            amortization = compute_amortization(
                arms["D0"]["targets"],
                arms["R0"]["targets"],
                dense_source_materialization_ms=float(
                    arms["D0"]["ledger"]["materialize_ms"]
                ),
                recovery_source_preparation_ms=float(
                    arms["R0"]["ledger"]["source_preparation_ms"]
                ),
            )
            formal.append(
                {
                    "repeat_index": repeat_index,
                    "arm_order": list(order),
                    "arms": arms,
                    "amortization": amortization,
                }
            )
        engineering_invalid = any(
            not row["taxonomy_valid"]
            for repeat in formal
            for arm in repeat["arms"].values()
            for row in arm["targets"]
        ) or any(
            (
                repeat["arms"]["R0"]["same_context_canary"]["engineering_status"]
                == "invalid"
                or not repeat["arms"]["R0"]["same_context_canary"]["recovery_outcome"][
                    "taxonomy_valid"
                ]
            )
            for repeat in formal
        )
        dense_invalid = any(
            not row["expected_outcome"]
            for repeat in formal
            for row in repeat["arms"]["D0"]["targets"]
        )
        exact_invalid = any(
            not row["expected_outcome"]
            for repeat in formal
            for row in repeat["arms"]["E0"]["targets"]
        )
        prefix_invalid = any(
            not row["expected_outcome"]
            for repeat in formal
            for row in repeat["arms"]["R0"]["targets"]
        )
        status = (
            "invalid"
            if engineering_invalid or dense_invalid or exact_invalid
            else ("inconclusive" if prefix_invalid else "valid")
        )
        summary = summarize_ceiling_repeats(formal)
        result = {
            "startup_snapshot": startup_snapshot,
            "clean_baseline_reset": clean_baseline_reset,
            "warmup": warmup,
            "formal": formal,
            "summary": summary,
            "early_stop": early_stop,
            "server_cold_start_ms": server_cold_start_ms,
            "status": status,
            "machine": machine_manifest(),
            "server_argv": list(server.command),
            "plugin_env": server.plugin_env,
            "observed_capacity": capacity,
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
    all_targets = [
        target
        for repeat in result["formal"]
        for arm in repeat["arms"].values()
        for target in arm["targets"]
    ]
    outcome_counts = {
        outcome: sum(row["outcome"] == outcome for row in all_targets)
        for outcome in manifest["outcome_taxonomy"]
    }
    reason_counts = {
        reason: sum(row["terminal_reason"] == reason for row in all_targets)
        for reason in manifest["exclusive_terminal_reasons"]
    }
    artifact = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": "Phase7-ceiling",
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
            "bytes": (
                None
                if setting["max_total_tokens"] is None
                else int(setting["max_total_tokens"]) * bytes_per_token
            ),
        },
        "observed_capacity": result["observed_capacity"],
        "crosses_chunk_boundary": (
            HEADER_TOKENS + int(setting["body_tokens"]) + 1
            > int(setting["chunked_prefill_size"])
        ),
        "segment_count": math.ceil(
            int(setting["body_tokens"]) / int(workload["segment_tokens_max"])
        ),
        "segment_tokens_max": int(workload["segment_tokens_max"]),
        "source_pin_until_reset": workload["source_pin_until_reset"] is True,
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
            "materialization": "see formal[*].arms[*].ledger",
            "recovery": "see formal[*].amortization",
            "scheduler": "not_applicable",
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
            "per_repeat_per_arm": {
                str(repeat["repeat_index"]): {
                    arm: data["pressure"] for arm, data in repeat["arms"].items()
                }
                for repeat in result["formal"]
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
            "workload_id": workload["spec"]["workload_id"],
            "body_tokens": len(workload["body"]),
            "targets_per_setup": len(workload["targets"]),
            "workload_manifest_sha256": manifest["workloads"]["A8"]["manifest_sha256"],
            "filler_manifest_sha256": manifest["workloads"]["filler_pool"][
                "manifest_sha256"
            ],
        },
        "warmup": result["warmup"],
        "formal": result["formal"],
        "summary": result["summary"],
        "early_stop": result["early_stop"],
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
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": "Phase7-ceiling",
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
            "bytes": (
                None
                if setting["max_total_tokens"] is None
                else int(setting["max_total_tokens"]) * bytes_per_token
            ),
        },
        "observed_capacity": {"tokens": None, "pages": None, "bytes": None},
        "crosses_chunk_boundary": (
            HEADER_TOKENS + int(setting["body_tokens"]) + 1
            > int(setting["chunked_prefill_size"])
        ),
        "segment_count": reported_segment_count(manifest, int(setting["body_tokens"])),
        "segment_tokens_max": (
            frozen_segment_tokens(manifest)
            if reported_segment_count(manifest, int(setting["body_tokens"])) is not None
            else None
        ),
        "source_pin_until_reset": (
            manifest.get("workloads", {}).get("A8", {}).get("source_pin_until_reset")
        ),
        "warmup_repeats": int(setting["warmup_repeats"]),
        "formal_repeats": int(setting["formal_repeats"]),
        "restarts": 1,
        "ledger": {
            "setup": {},
            "materialization": {},
            "recovery": {},
            "scheduler": "not_applicable",
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
        runner_module=CEILING_RUNNER,
        runner_file=Path(__file__),
    )
    ceiling_early_stop_contract(
        context.setting,
        restart_index=context.restart_index,
        mde_gate_passed=bool(args.mde_gate_passed),
    )
    run_id = (
        f"p7-ceiling-{args.setting_id}-r{args.restart_index}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    append_jsonl(
        args.central_log,
        {
            "run_id": run_id,
            "phase": "Phase7-ceiling",
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
                "phase": "Phase7-ceiling",
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
                "phase": "Phase7-ceiling",
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
