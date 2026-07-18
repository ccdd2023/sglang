#!/usr/bin/env python3
"""Reference measurement and sequential composition runner for ProbeHead V12."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark.multi_workflow.measure_sessiongraph_atlas import (
    MODEL_ID,
    _atlas_prompt_hash,
    _cosine_deviation,
    _dense,
    _js,
    _layers,
    _rope_shift,
    build_prompt_pair,
)
from benchmark.multi_workflow.probehead_v12 import (
    HEAD_CANDIDATES,
    MAX_COPY_ISLANDS,
    PROFILE,
    ProbeCandidate,
    probe_score,
    shuffled_exact_budget,
)
from benchmark.multi_workflow.sessiongraph_v11 import CostModel, read_jsonl


@dataclass(frozen=True)
class RuntimeCandidate:
    policy: ProbeCandidate
    source_span: tuple[int, int]
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module(turn: Mapping[str, Any], module_id: str) -> Mapping[str, Any]:
    return next(
        row for row in turn["modules"] if str(row["module_id"]) == module_id
    )


def _tokens(tokenizer: Any, turn: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in tokenizer.encode(
            str(turn["rendered_prompt"]), add_special_tokens=False
        )
    )


def _probe_metrics(
    *,
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    source_span: tuple[int, int],
    target_span: tuple[int, int],
    head_tokens: int,
    theta: float,
    comparison_device: str | torch.device | None = None,
) -> tuple[float, float, float]:
    length = target_span[1] - target_span[0]
    head = min(head_tokens, length)
    source_start, target_start = source_span[0], target_span[0]
    delta = target_start - source_start
    source_keys = torch.stack(
        [
            key[:, source_start : source_start + head]
            for key, _ in source_cache
        ]
    )
    source_values = torch.stack(
        [
            value[:, source_start : source_start + head]
            for _, value in source_cache
        ]
    )
    target_keys = torch.stack(
        [
            key[:, target_start : target_start + head]
            for key, _ in target_cache
        ]
    )
    target_values = torch.stack(
        [
            value[:, target_start : target_start + head]
            for _, value in target_cache
        ]
    )
    if comparison_device is not None:
        source_keys = source_keys.to(comparison_device)
        source_values = source_values.to(comparison_device)
        target_keys = target_keys.to(comparison_device)
        target_values = target_values.to(comparison_device)
    if source_keys.is_cuda:
        torch.cuda.synchronize(source_keys.device)
    started = time.perf_counter()
    key_deviation = _cosine_deviation(
        _rope_shift(source_keys, delta, theta), target_keys
    )
    value_deviation = _cosine_deviation(source_values, target_values)
    if source_keys.is_cuda:
        torch.cuda.synchronize(source_keys.device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return key_deviation, value_deviation, elapsed_ms


@torch.inference_mode()
def _splice_head(
    *,
    model: Any,
    target_ids: Sequence[int],
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    source_span: tuple[int, int],
    target_span: tuple[int, int],
    head_tokens: int,
    theta: float,
    chunk_size: int,
) -> torch.Tensor:
    from transformers.cache_utils import DynamicCache

    source_start, source_end = source_span
    target_start, target_end = target_span
    length = target_end - target_start
    head = min(head_tokens, length)
    delta = target_start - source_start
    layers = []
    for (target_key, target_value), (source_key, source_value) in zip(
        target_cache, source_cache, strict=True
    ):
        key = target_key[:, : target_start + head]
        value = target_value[:, : target_start + head]
        if head < length:
            key = torch.cat(
                (
                    key,
                    _rope_shift(
                        source_key[:, source_start + head : source_end],
                        delta,
                        theta,
                    ),
                ),
                dim=1,
            )
            value = torch.cat(
                (value, source_value[:, source_start + head : source_end]), dim=1
            )
        layers.append((key.unsqueeze(0).cuda(), value.unsqueeze(0).cuda()))
    cache = DynamicCache(layers, config=model.config)
    suffix = target_ids[target_end:]
    if not suffix:
        raise ValueError("selected module has no target suffix")
    logits = None
    for offset in range(0, len(suffix), chunk_size):
        output = model(
            input_ids=torch.tensor(
                [suffix[offset : offset + chunk_size]], device="cuda"
            ),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        logits = output.logits[0, -1].detach().float().cpu()
    assert logits is not None
    return logits


def _runtime_probe(
    *,
    cache: Any,
    runtime: RuntimeCandidate,
    head: int,
    theta: float,
) -> tuple[float, float, float]:
    candidate = runtime.policy
    source_start = runtime.source_span[0]
    target_start = candidate.target_start
    delta = target_start - source_start
    target_layers = _layers(cache)
    device = target_layers[0][0].device
    # Materialization/transfer is part of the existing KV copy path.  The
    # registered probe latency measures only the incremental comparison.
    source_keys = torch.stack(
        [
            key[:, source_start : source_start + head]
            for key, _ in runtime.source_cache
        ]
    ).to(device)
    source_values = torch.stack(
        [
            value[:, source_start : source_start + head]
            for _, value in runtime.source_cache
        ]
    ).to(device)
    target_keys = torch.stack(
        [
            key[0, :, target_start : target_start + head]
            for key, _ in target_layers
        ]
    )
    target_values = torch.stack(
        [
            value[0, :, target_start : target_start + head]
            for _, value in target_layers
        ]
    )
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    key_deviation = _cosine_deviation(
        _rope_shift(source_keys, delta, theta), target_keys
    )
    value_deviation = _cosine_deviation(source_values, target_values)
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return key_deviation, value_deviation, elapsed_ms


def _append_source_body(
    *,
    model: Any,
    cache: Any,
    runtime: RuntimeCandidate,
    head: int,
    theta: float,
) -> Any:
    from transformers.cache_utils import DynamicCache

    candidate = runtime.policy
    source_start = runtime.source_span[0]
    source_end = runtime.source_span[1]
    delta = candidate.target_start - source_start
    layers = []
    for (target_key, target_value), (source_key, source_value) in zip(
        _layers(cache), runtime.source_cache, strict=True
    ):
        body_key = _rope_shift(
            source_key[:, source_start + head : source_end].to(target_key.device),
            delta,
            theta,
        ).unsqueeze(0)
        body_value = source_value[
            :, source_start + head : source_end
        ].to(target_value.device).unsqueeze(0)
        layers.append(
            (
                torch.cat((target_key, body_key), dim=2),
                torch.cat((target_value, body_value), dim=2),
            )
        )
    return DynamicCache(layers, config=model.config)


@torch.inference_mode()
def _advance(
    model: Any, cache: Any | None, token_ids: Sequence[int]
) -> tuple[Any, torch.Tensor | None]:
    if not token_ids:
        return cache, None
    output = model(
        input_ids=torch.tensor([token_ids], device="cuda"),
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
        logits_to_keep=1,
    )
    return output.past_key_values, output.logits[0, -1].detach().float().cpu()


@torch.inference_mode()
def execute_composed(
    *,
    model: Any,
    target_ids: Sequence[int],
    runtimes: Sequence[RuntimeCandidate],
    head_tokens: int,
    theta: float,
    chunk_size: int,
    mode: str,
    threshold: float | None,
    cost_model: CostModel,
    forced_heads: Mapping[str, int] | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Execute one target prompt left-to-right with dynamic ProbeHead decisions."""
    ordered = sorted(runtimes, key=lambda value: value.policy.target_start)
    if not ordered:
        raise ValueError("composed execution requires candidates")
    cache: Any | None = None
    cursor = 0
    logits: torch.Tensor | None = None
    islands = 0
    previous_copy_end: int | None = None
    observations = []
    for runtime in ordered:
        candidate = runtime.policy
        if candidate.target_start < cursor:
            raise ValueError("runtime candidates overlap")
        cache, gap_logits = _advance(
            model, cache, target_ids[cursor : candidate.target_start]
        )
        if gap_logits is not None:
            logits = gap_logits
        head = min(head_tokens, candidate.length)
        cache, head_logits = _advance(
            model,
            cache,
            target_ids[candidate.target_start : candidate.target_start + head],
        )
        if head_logits is not None:
            logits = head_logits
        if cache is None:
            raise AssertionError("probe head did not produce a cache")
        k_dev, v_dev, probe_ms = _runtime_probe(
            cache=cache, runtime=runtime, head=head, theta=theta
        )
        score = probe_score(k_dev, v_dev)
        body = candidate.length - head
        adjacent = previous_copy_end == candidate.target_start
        new_island = not adjacent

        if forced_heads is not None:
            decision_head = int(forced_heads[candidate.module_id])
            if decision_head < head or decision_head > candidate.length:
                raise ValueError("forced head violates the frozen probe budget")
            extra = decision_head - head
            cache, body_logits = _advance(
                model,
                cache,
                target_ids[
                    candidate.target_start
                    + head : candidate.target_start
                    + decision_head
                ],
            )
            if body_logits is not None:
                logits = body_logits
            copy = candidate.length - decision_head
            reason = "forced_copy" if copy else "forced_dense"
            head = decision_head
        else:
            if threshold is None:
                raise ValueError("dynamic probe execution requires a threshold")
            saving = (
                cost_model.net_saving_us(body, islands=int(new_island))
                - probe_ms * 1000
            )
            copy = body
            reason = "probe_copy"
            if body <= 0:
                copy, reason = 0, "probe_consumes_complete_module"
            elif score > threshold:
                copy, reason = 0, "probe_score_above_threshold"
            elif new_island and islands >= MAX_COPY_ISLANDS:
                copy, reason = 0, "probe_island_limit"
            elif saving <= 0:
                copy, reason = 0, "probe_cost_negative"

        if copy:
            if new_island:
                islands += 1
            cache = _append_source_body(
                model=model, cache=cache, runtime=runtime, head=head, theta=theta
            )
            previous_copy_end = candidate.target_start + candidate.length
            island_index: int | None = islands - 1
        else:
            cache, body_logits = _advance(
                model,
                cache,
                target_ids[
                    candidate.target_start + head : candidate.target_start
                    + candidate.length
                ],
            )
            if body_logits is not None:
                logits = body_logits
            previous_copy_end = None
            island_index = None
        cursor = candidate.target_start + candidate.length
        observations.append(
            {
                "module_id": candidate.module_id,
                "head_tokens": head,
                "probe_k_deviation": k_dev,
                "probe_v_deviation": v_dev,
                "probe_score": score,
                "probe_ms": probe_ms,
                "copied_tokens": copy,
                "island_index": island_index,
                "decision_reason": reason,
            }
        )
    for offset in range(cursor, len(target_ids), chunk_size):
        cache, suffix_logits = _advance(
            model, cache, target_ids[offset : offset + chunk_size]
        )
        if suffix_logits is not None:
            logits = suffix_logits
    if logits is None:
        raise AssertionError("composed execution did not produce logits")
    return logits, observations


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _probe_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["case_kind"]),
        str(row["cohort"]),
        str(row["session_id"]),
        int(row["turn_id"]),
        str(row["module_id"]),
        str(row["disturbance"]),
        int(row["head_tokens"]),
    )


def _request_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["session_id"]),
        int(row["turn_id"]),
        int(row["head_tokens"]),
    )


def _validate_registration(
    registration_path: Path, design_path: Path
) -> dict[str, Any]:
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("policy") != PROFILE:
        raise ValueError("measurement registration is not ProbeHead V12")
    if registration.get("design_sha256") != _sha(design_path):
        raise ValueError("measurement design hash differs from registration")
    return registration


def _validate_executor_amendment(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "accepted": True,
        "probe_warmup_iterations": 3,
        "timed_scope": "vectorized KV comparison only",
        "thresholds_changed": False,
        "holdout_opened": False,
    }
    mismatches = {
        key: (value.get(key), expected_value)
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"reference executor amendment mismatch: {mismatches}")
    return value


def _warm_probe_comparison(model: Any, theta: float, iterations: int) -> None:
    config = model.config
    layers = int(getattr(config, "num_hidden_layers"))
    heads = int(getattr(config, "num_key_value_heads"))
    head_dim = int(getattr(config, "hidden_size")) // int(
        getattr(config, "num_attention_heads")
    )
    shape = (layers, heads, max(HEAD_CANDIDATES), head_dim)
    left = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    right = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    for _ in range(iterations):
        _cosine_deviation(_rope_shift(left, 1, theta), right)
        _cosine_deviation(left, right)
    torch.cuda.synchronize()
    del left, right


def plan_rows(
    *,
    design_path: Path,
    cohort: str,
    mode: str,
    calibration_lock_path: Path | None,
    development_gate_path: Path | None = None,
) -> tuple[list[dict[str, Any]], int | None, float | None]:
    rows = [row for row in read_jsonl(design_path) if row["cohort"] == cohort]
    chosen_head = None
    threshold = None
    if mode == "compose" or cohort == "holdout":
        if calibration_lock_path is None or not calibration_lock_path.exists():
            raise ValueError("composition/holdout requires CALIBRATION_LOCK.json")
        lock = json.loads(calibration_lock_path.read_text(encoding="utf-8"))
        if lock.get("status") != "LOCKED":
            raise ValueError("calibration is not locked")
        chosen_head = int(lock["head_tokens"])
        threshold = float(lock["threshold"])
        rows = [row for row in rows if int(row["head_tokens"]) == chosen_head]
        if cohort == "holdout":
            if development_gate_path is None or not development_gate_path.exists():
                raise ValueError("holdout requires a passed development compose gate")
            gate = json.loads(development_gate_path.read_text(encoding="utf-8"))
            if (
                gate.get("stage") != "development-compose"
                or gate.get("passed") is not True
                or gate.get("inputs", {}).get("calibration_lock")
                != _sha(calibration_lock_path)
            ):
                raise ValueError("development compose gate is invalid or stale")
    elif cohort == "development":
        rows = [
            row for row in rows if int(row["head_tokens"]) in HEAD_CANDIDATES
        ]
    return rows, chosen_head, threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--executor-amendment", type=Path, required=True)
    parser.add_argument("--cost-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cohort", choices=("development", "holdout"), required=True)
    parser.add_argument("--mode", choices=("probes", "compose"), required=True)
    parser.add_argument("--calibration-lock", type=Path)
    parser.add_argument("--development-gate", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--session-ids", default="")
    parser.add_argument("--turn-ids", default="")
    parser.add_argument("--module-ids", default="")
    parser.add_argument("--head-tokens", default="")
    parser.add_argument("--case-kinds", default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--splice-chunk-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    registration = _validate_registration(args.registration, args.design)
    amendment = _validate_executor_amendment(args.executor_amendment)
    rows, chosen_head, threshold = plan_rows(
        design_path=args.design,
        cohort=args.cohort,
        mode=args.mode,
        calibration_lock_path=args.calibration_lock,
        development_gate_path=args.development_gate,
    )
    session_filter = set(filter(None, args.session_ids.split(",")))
    turn_filter = {
        int(value) for value in args.turn_ids.split(",") if value.strip()
    }
    kind_filter = set(filter(None, args.case_kinds.split(",")))
    module_filter = set(filter(None, args.module_ids.split(",")))
    head_filter = {
        int(value) for value in args.head_tokens.split(",") if value.strip()
    }
    rows = [
        row
        for row in rows
        if (not session_filter or str(row["session_id"]) in session_filter)
        and (not turn_filter or int(row["turn_id"]) in turn_filter)
        and (not kind_filter or str(row["case_kind"]) in kind_filter)
        and (not module_filter or str(row["module_id"]) in module_filter)
        and (not head_filter or int(row["head_tokens"]) in head_filter)
    ]
    completed_probe_keys: set[tuple[Any, ...]] = set()
    completed_request_keys: set[tuple[str, int, int]] = set()
    if args.resume and args.mode == "probes" and args.output.exists():
        completed_probe_keys = {
            _probe_key(row)
            for row in read_jsonl(args.output)
            if row.get("status") == "ok"
        }
        rows = [row for row in rows if _probe_key(row) not in completed_probe_keys]
    if args.resume and args.mode == "compose" and args.request_output.exists():
        completed_request_keys = {
            _request_key(row)
            for row in read_jsonl(args.request_output)
            if row.get("status") == "ok"
        }
        if args.output.exists():
            partial = {
                _request_key(row)
                for row in read_jsonl(args.output)
                if row.get("status") == "ok"
            } - completed_request_keys
            if partial:
                raise ValueError(
                    "compose output contains a partial request group; "
                    "use fresh output paths"
                )
        rows = [
            row
            for row in rows
            if (
                str(row["session_id"]),
                int(row["turn_id"]),
                int(row["head_tokens"]),
            )
            not in completed_request_keys
        ]
    plan = {
        "passed": True,
        "plan_only": args.plan_only,
        "cohort": args.cohort,
        "mode": args.mode,
        "selected_design_rows": len(rows),
        "sessions": len({str(row["session_id"]) for row in rows}),
        "head_tokens": chosen_head,
        "threshold": threshold,
        "completed_rows": len(completed_probe_keys)
        if args.mode == "probes"
        else len(completed_request_keys),
        "holdout_locked": args.cohort != "holdout"
        or args.calibration_lock is not None,
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU substitution is forbidden")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    ).eval()
    theta = float(getattr(model.config, "rope_theta", 1_000_000.0))
    _warm_probe_comparison(
        model, theta, int(amendment["probe_warmup_iterations"])
    )
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    turns = {
        (str(row["session_id"]), int(row["turn_id"])): row for row in replay
    }
    final_turns = {}
    for key, turn in turns.items():
        if key[0] not in final_turns or key[1] > int(
            final_turns[key[0]]["turn_id"]
        ):
            final_turns[key[0]] = turn
    cost_gate = json.loads(args.cost_gate.read_text(encoding="utf-8"))
    cost_model = CostModel(**cost_gate["cost_model"])

    module_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    written_module_rows = 0
    written_request_rows = 0
    errors = []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if args.mode == "compose" and row["case_kind"] != "workflow":
            continue
        if args.mode == "probes":
            key = (
                (
                    row["case_kind"],
                    row["session_id"],
                    row["turn_id"],
                )
                if row["case_kind"] == "workflow"
                else (
                    row["case_kind"],
                    row["session_id"],
                    row["turn_id"],
                    row["module_id"],
                    row["disturbance"],
                )
            )
        else:
            key = (
                row["case_kind"],
                row["session_id"],
                row["turn_id"],
                row["head_tokens"],
            )
        grouped[key].append(row)

    total_groups = len(grouped)
    for group_index, (key, selected) in enumerate(sorted(grouped.items()), 1):
        try:
            first = selected[0]
            session_id = str(first["session_id"])
            target_turn = turns[(session_id, int(first["turn_id"]))]
            if args.mode == "probes":
                entries = []
                source_caches = {}
                if first["case_kind"] == "stress":
                    pair = build_prompt_pair(
                        tokenizer=tokenizer,
                        turns=turns,
                        final_turns=final_turns,
                        session_id=session_id,
                        module_id=str(first["module_id"]),
                        disturbance=str(first["disturbance"]),
                    )
                    source_ids, target_ids = pair.source_ids, pair.target_ids
                    source_span, target_span = pair.source_span, pair.target_span
                    source_cache, _ = _dense(model, source_ids)
                    source_caches["stress"] = source_cache
                    entries = [
                        (row, source_ids, source_cache, source_span, target_span)
                        for row in selected
                    ]
                else:
                    target_ids = _tokens(tokenizer, target_turn)
                    for row in selected:
                        source_turn_id = int(row["source_turn_id"])
                        if source_turn_id not in source_caches:
                            source_turn = turns[(session_id, source_turn_id)]
                            source_ids = _tokens(tokenizer, source_turn)
                            source_caches[source_turn_id] = (
                                source_turn,
                                source_ids,
                                _dense(model, source_ids)[0],
                            )
                        source_turn, source_ids, source_cache = source_caches[
                            source_turn_id
                        ]
                        module_id = str(row["module_id"])
                        entries.append(
                            (
                                row,
                                source_ids,
                                source_cache,
                                tuple(
                                    map(
                                        int,
                                        _module(source_turn, module_id)[
                                            "token_span"
                                        ],
                                    )
                                ),
                                tuple(
                                    map(
                                        int,
                                        _module(target_turn, module_id)[
                                            "token_span"
                                        ],
                                    )
                                ),
                            )
                        )
                target_cache, target_logits = _dense(model, target_ids)
                for row, source_ids, source_cache, source_span, target_span in entries:
                    head = int(row["head_tokens"])
                    if (
                        source_ids[slice(*source_span)]
                        != target_ids[slice(*target_span)]
                    ):
                        raise ValueError("probe source/target token slices differ")
                    k_dev, v_dev, probe_ms = _probe_metrics(
                        source_cache=source_cache,
                        target_cache=target_cache,
                        source_span=source_span,
                        target_span=target_span,
                        head_tokens=head,
                        theta=theta,
                        comparison_device="cuda",
                    )
                    splice_logits = _splice_head(
                        model=model,
                        target_ids=target_ids,
                        target_cache=target_cache,
                        source_cache=source_cache,
                        source_span=source_span,
                        target_span=target_span,
                        head_tokens=head,
                        theta=theta,
                        chunk_size=args.splice_chunk_size,
                    )
                    module_rows.append(
                        {
                            **row,
                            "status": "ok",
                            "token_count": target_span[1] - target_span[0],
                            "source_start": source_span[0],
                            "target_start": target_span[0],
                            "target_prompt_tokens": len(target_ids),
                            "remaining_body_tokens": max(
                                0, target_span[1] - target_span[0] - head
                            ),
                            "probe_k_deviation": k_dev,
                            "probe_v_deviation": v_dev,
                            "probe_score": probe_score(k_dev, v_dev),
                            "probe_ms": probe_ms,
                            "causal_splice_logit_js": _js(
                                target_logits, splice_logits
                            ),
                            "splice_top1_changed": int(target_logits.argmax())
                            != int(splice_logits.argmax()),
                            "source_prompt_hash": _atlas_prompt_hash(source_ids),
                            "target_prompt_hash": _atlas_prompt_hash(target_ids),
                            "measurement_model": MODEL_ID,
                            "dtype": "bfloat16",
                            "attention_implementation": "sdpa",
                        }
                    )
                del source_caches, target_cache
            else:
                head = int(first["head_tokens"])
                target_ids = _tokens(tokenizer, target_turn)
                teacher_cache, teacher_logits = _dense(model, target_ids)
                del teacher_cache
                source_caches = {}
                runtimes = []
                for row in selected:
                    source_turn_id = int(row["source_turn_id"])
                    if source_turn_id not in source_caches:
                        source_turn = turns[(session_id, source_turn_id)]
                        source_ids = _tokens(tokenizer, source_turn)
                        source_caches[source_turn_id] = (
                            source_turn,
                            source_ids,
                            _dense(model, source_ids)[0],
                        )
                    source_turn, source_ids, source_cache = source_caches[
                        source_turn_id
                    ]
                    module_id = str(row["module_id"])
                    source_span = tuple(
                        map(int, _module(source_turn, module_id)["token_span"])
                    )
                    target_span = tuple(
                        map(int, _module(target_turn, module_id)["token_span"])
                    )
                    if (
                        source_ids[slice(*source_span)]
                        != target_ids[slice(*target_span)]
                    ):
                        raise ValueError("composed source/target token slices differ")
                    runtimes.append(
                        RuntimeCandidate(
                            policy=ProbeCandidate(
                                session_id=session_id,
                                turn_id=int(row["turn_id"]),
                                module_id=module_id,
                                source_start=source_span[0],
                                target_start=target_span[0],
                                length=target_span[1] - target_span[0],
                                prompt_tokens=len(target_ids),
                            ),
                            source_span=source_span,
                            source_cache=source_cache,
                        )
                    )
                probe_logits, probe_modules = execute_composed(
                    model=model,
                    target_ids=target_ids,
                    runtimes=runtimes,
                    head_tokens=head,
                    theta=theta,
                    chunk_size=args.splice_chunk_size,
                    mode="probe",
                    threshold=threshold,
                    cost_model=cost_model,
                )
                copied_budget = sum(
                    int(row["copied_tokens"]) for row in probe_modules
                )
                copy_all_heads = {
                    runtime.policy.module_id: min(
                        head, runtime.policy.length
                    )
                    for runtime in runtimes
                }
                copy_all_logits, _ = execute_composed(
                    model=model,
                    target_ids=target_ids,
                    runtimes=runtimes,
                    head_tokens=head,
                    theta=theta,
                    chunk_size=args.splice_chunk_size,
                    mode="copy_all",
                    threshold=None,
                    cost_model=cost_model,
                    forced_heads=copy_all_heads,
                )
                shuffled = shuffled_exact_budget(
                    candidates=[runtime.policy for runtime in runtimes],
                    copied_token_budget=copied_budget,
                    head_tokens=head,
                )
                shuffled_heads = {
                    decision.candidate.module_id: decision.head_tokens
                    for decision in shuffled
                }
                shuffled_logits, _ = execute_composed(
                    model=model,
                    target_ids=target_ids,
                    runtimes=runtimes,
                    head_tokens=head,
                    theta=theta,
                    chunk_size=args.splice_chunk_size,
                    mode="shuffled",
                    threshold=None,
                    cost_model=cost_model,
                    forced_heads=shuffled_heads,
                )
                selected_by_module = {
                    str(row["module_id"]): row for row in selected
                }
                module_rows.extend(
                    {
                        **selected_by_module[str(measurement["module_id"])],
                        **measurement,
                        "status": "ok",
                        "measurement_model": MODEL_ID,
                    }
                    for measurement in probe_modules
                )
                probe_times = [float(row["probe_ms"]) for row in probe_modules]
                request_rows.append(
                    {
                        "status": "ok",
                        "cohort": args.cohort,
                        "session_id": session_id,
                        "turn_id": int(first["turn_id"]),
                        "head_tokens": head,
                        "threshold": threshold,
                        "prompt_tokens": len(target_ids),
                        "candidate_modules": len(runtimes),
                        "copied_tokens": copied_budget,
                        "cost_positive_copy_fraction": copied_budget
                        / len(target_ids),
                        "copy_islands": len(
                            {
                                row["island_index"]
                                for row in probe_modules
                                if row["island_index"] is not None
                            }
                        ),
                        "probe_p95_ms": float(
                            np.quantile(probe_times, 0.95)
                        ),
                        "probe_composed_js": _js(
                            teacher_logits, probe_logits
                        ),
                        "copy_all_composed_js": _js(
                            teacher_logits, copy_all_logits
                        ),
                        "shuffled_composed_js": _js(
                            teacher_logits, shuffled_logits
                        ),
                        "probe_top1_changed": int(teacher_logits.argmax())
                        != int(probe_logits.argmax()),
                        "target_prompt_hash": _atlas_prompt_hash(target_ids),
                        "measurement_model": MODEL_ID,
                    }
                )
                del source_caches
            gc.collect()
            torch.cuda.empty_cache()
            _write_jsonl(args.output, module_rows)
            _write_jsonl(args.request_output, request_rows)
            written_module_rows += len(module_rows)
            written_request_rows += len(request_rows)
            module_rows.clear()
            request_rows.clear()
            if (
                args.progress_every > 0
                and (
                    group_index % args.progress_every == 0
                    or group_index == total_groups
                )
            ):
                print(
                    json.dumps(
                        {
                            "progress_groups": group_index,
                            "total_groups": total_groups,
                            "module_rows": written_module_rows,
                            "request_rows": written_request_rows,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        except Exception as error:
            errors.append(
                {
                    "key": list(key),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            break

    summary = {
        **plan,
        "passed": not errors,
        "module_rows": written_module_rows,
        "request_rows": written_request_rows,
        "errors": errors,
        "model": MODEL_ID,
        "model_source": args.model,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "splice_suffix_chunk_size": args.splice_chunk_size,
        "registration_sha256": _sha(args.registration),
        "executor_amendment_sha256": _sha(args.executor_amendment),
        "probe_warmup_iterations": int(
            amendment["probe_warmup_iterations"]
        ),
        "design_sha256": registration["design_sha256"],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
