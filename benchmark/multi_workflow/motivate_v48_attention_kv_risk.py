#!/usr/bin/env python3
"""Measure attention x K/V-drift as an oracle risk signal for V46 islands.

The measurement is deliberately offline and diagnostic.  It computes Dense
target K/V before deciding, so it must not be presented as an online policy.
Its purpose is to establish whether model-internal evidence can explain the
causal logit harm of copying an observation island before engineering a cheap
SGLang probe.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from benchmark.multi_workflow.measure_probehead_v12 import _advance
from benchmark.multi_workflow.measure_sessiongraph_atlas import (
    _cpu_cache,
    _js,
    _layers,
    _rope_shift,
    _rotate_half,
)
from benchmark.multi_workflow.motivate_v47_task_conditioned_pool import (
    TOKENS_PER_ISLAND,
    _candidate_spans,
)
from benchmark.multi_workflow.run_bridge_reuse_pilot import write_json
from benchmark.multi_workflow.run_v40_repobench_control import (
    DEFAULT_WORKLOAD,
    MODEL,
    SOURCE_PREFIX,
    SOURCE_SUFFIX,
    _render,
)


ROOT = Path("/home/gfy/CodeMAS_Project")
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_m48_attention_kv_risk_20260805"
    / "canary8"
)
DEFAULT_M47_RESULT = (
    ROOT
    / "kvflow-artifacts/impactkv_m47_task_conditioned_pool_20260805"
    / "full50/RESULT.json"
)
PROBE_LAYERS = (0, 8, 17, 26, 35)
QUERY_TAIL_TOKENS = 32
SPLICE_CHUNK_SIZE = 512
RANDOM_SEED = 20260805


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_theta(config: Any) -> float:
    if hasattr(config, "rope_theta"):
        return float(config.rope_theta)
    parameters = getattr(config, "rope_parameters", None) or {}
    return float(parameters.get("rope_theta", 1_000_000.0))


def prepare_case(tokenizer: Any, case: Mapping[str, Any]) -> dict[str, Any]:
    reusable = [
        str(segment["text"])
        for segment in case["segments"]
        if bool(segment["reusable"])
    ]
    source_messages = [
        {
            "role": "system",
            "content": (
                "You inspect repository code returned by read-only tools. "
                "Return only a one-word acknowledgement."
            ),
        },
        {
            "role": "user",
            "content": SOURCE_PREFIX + "".join(reusable) + SOURCE_SUFFIX,
        },
    ]
    source_prompt, source_ids, source_offsets = _render(
        tokenizer, source_messages
    )
    target_prompt, target_ids, target_offsets = _render(
        tokenizer, list(case["messages"])
    )
    candidates = _candidate_spans(
        source_prompt=source_prompt,
        source_ids=source_ids,
        source_offsets=source_offsets,
        target_prompt=target_prompt,
        target_ids=target_ids,
        target_offsets=target_offsets,
        reusable=reusable,
    )
    if len(candidates) < 3:
        raise ValueError(f"{case['case_id']}: fewer than three candidates")
    candidate_rows = [
        {
            "candidate_id": f"context-{row['context_index']}",
            "context_index": int(row["context_index"]),
            "length": int(row["length"]),
            "source_start": int(row["source_start"]),
            "target_start": int(row["target_start"]),
        }
        for row in candidates
    ]
    v46_indices = sorted(
        range(len(candidate_rows)),
        key=lambda index: candidate_rows[index]["context_index"],
        reverse=True,
    )[:3]
    v46_ids = sorted(candidate_rows[index]["candidate_id"] for index in v46_indices)
    answer = str(case["metadata"]["answers"][0])
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids:
        raise ValueError(f"{case['case_id']}: answer has no tokens")
    return {
        "answer": answer,
        "answer_first_token_id": int(answer_ids[0]),
        "case_id": str(case["case_id"]),
        "candidates": candidate_rows,
        "source_input_ids": source_ids,
        "target_input_ids": target_ids,
        "v46_candidate_ids": v46_ids,
    }


def prepare(workload: Path, output: Path, limit: int) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    value = json.loads(workload.read_text(encoding="utf-8"))
    source_cases = value["cases"][:limit] if limit > 0 else value["cases"]
    cases = [prepare_case(tokenizer, case) for case in source_cases]
    output.mkdir(parents=True, exist_ok=True)
    design_path = output / "DESIGN.json"
    write_json(
        design_path,
        {
            "cases": cases,
            "model": MODEL,
            "probe_layers": list(PROBE_LAYERS),
            "query_tail_tokens": QUERY_TAIL_TOKENS,
        },
    )
    registration = {
        "status": "REGISTERED_BEFORE_GPU",
        "purpose": (
            "test whether target attention and cross-prefix K/V drift explain "
            "causal harm from V46 observation-island reuse"
        ),
        "cases": len(cases),
        "dataset": "RepoBench-P",
        "model": MODEL,
        "design_sha256": _sha256(design_path),
        "probe_layers_zero_based": list(PROBE_LAYERS),
        "query_tail_tokens": QUERY_TAIL_TOKENS,
        "candidate_contract": {
            "candidate_tokens": TOKENS_PER_ISLAND,
            "token_identical_source_target": True,
            "all_eligible_candidates_measured": True,
            "v46_triple_composed_left_to_right": True,
        },
        "signals": [
            "target-query attention mass",
            "RoPE-corrected K cosine drift",
            "V cosine drift",
            "relative K/V L2 drift",
            "head-aware attention-times-drift q90",
        ],
        "causal_labels": [
            "single-island final-logit JS versus Dense",
            "single-island answer-first-token NLL delta",
            "V46 three-island composed final-logit JS and NLL delta",
        ],
        "scope_warning": (
            "oracle motivation only: full Dense target K/V is measured and "
            "cannot be charged as an online controller"
        ),
        "canary_expansion_rule": (
            "expand to full50 if every case has finite attention, drift, "
            "single-island causal labels, and V46 composed labels; signal "
            "direction is not an expansion gate"
        ),
        "promotion_rule": (
            "no runtime promotion from this experiment; a positive oracle "
            "result must be followed by a separately frozen cheap-proxy test"
        ),
    }
    write_json(output / "REGISTRATION.json", registration)
    return registration


def _cosine_deviation_by_head(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    left = F.normalize(left.float(), dim=-1)
    right = F.normalize(right.float(), dim=-1)
    return 1 - (left * right).sum(-1).mean(-1)


def _relative_l2_by_head(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    numerator = (left.float() - right.float()).square().sum(dim=(-2, -1))
    denominator = right.float().square().sum(dim=(-2, -1)).clamp_min(1e-12)
    return numerator / denominator


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


@torch.inference_mode()
def _target_forward(
    *,
    model: Any,
    target_ids: Sequence[int],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, dict[str, Any]]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    layers = model.model.layers
    for layer_index in PROBE_LAYERS:
        def capture(
            _module: Any,
            args: tuple[Any, ...],
            index: int = layer_index,
        ) -> None:
            captured[index] = args[0][:, -QUERY_TAIL_TOKENS:].detach()

        handles.append(layers[layer_index].register_forward_pre_hook(capture))
    inputs = torch.tensor([target_ids], device="cuda", dtype=torch.long)
    try:
        output = model(
            input_ids=inputs,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(PROBE_LAYERS):
        raise RuntimeError("not all probe-layer hidden states were captured")
    cache_layers = _layers(output.past_key_values)
    prompt_tokens = len(target_ids)
    query_tokens = min(QUERY_TAIL_TOKENS, prompt_tokens)
    candidate_attention: dict[str, dict[str, Any]] = {
        str(row["candidate_id"]): {"layers": []} for row in candidates
    }
    num_attention_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    groups = num_attention_heads // num_kv_heads
    head_dim = int(model.config.hidden_size) // num_attention_heads
    query_positions = torch.arange(
        prompt_tokens - query_tokens,
        prompt_tokens,
        device="cuda",
        dtype=torch.long,
    ).unsqueeze(0)
    key_positions = torch.arange(prompt_tokens, device="cuda")
    for layer_index in PROBE_LAYERS:
        attention = layers[layer_index].self_attn
        hidden = captured[layer_index][:, -query_tokens:]
        query = attention.q_proj(hidden).view(
            1, query_tokens, num_attention_heads, head_dim
        ).transpose(1, 2)
        cosine, sine = model.model.rotary_emb(hidden, query_positions)
        cosine = cosine.unsqueeze(1)
        sine = sine.unsqueeze(1)
        query = query * cosine + _rotate_half(query) * sine
        key = cache_layers[layer_index][0].repeat_interleave(groups, dim=1)
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
        scores /= math.sqrt(head_dim)
        causal_mask = key_positions.view(1, 1, 1, -1) > query_positions.view(
            1, 1, -1, 1
        )
        weights = torch.softmax(scores.masked_fill(causal_mask, -torch.inf), dim=-1)
        for candidate in candidates:
            start = int(candidate["target_start"])
            end = start + int(candidate["length"])
            per_head = weights[..., start:end].sum(-1).mean(-1)[0]
            values = [float(value) for value in per_head.cpu().tolist()]
            candidate_attention[str(candidate["candidate_id"])]["layers"].append(
                {
                    "layer": layer_index,
                    "per_query_head_mass": values,
                    "mean": statistics.fmean(values),
                    "q90": _quantile(values, 0.9),
                    "max": max(values),
                }
            )
        del hidden, query, key, scores, weights
    cache = _cpu_cache(output.past_key_values)
    logits = output.logits[0, -1].detach().float().cpu()
    del output, inputs, captured
    torch.cuda.empty_cache()
    return cache, logits, candidate_attention


def _candidate_internal_metrics(
    *,
    candidate: Mapping[str, Any],
    attention: Mapping[str, Any],
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    theta: float,
    num_attention_heads: int,
) -> dict[str, Any]:
    source_start = int(candidate["source_start"])
    target_start = int(candidate["target_start"])
    length = int(candidate["length"])
    delta = target_start - source_start
    layer_rows = []
    products = []
    attention_values = []
    drift_values = []
    k_l2_values = []
    v_l2_values = []
    attention_by_layer = {
        int(row["layer"]): row for row in attention["layers"]
    }
    for layer_index in PROBE_LAYERS:
        source_key, source_value = source_cache[layer_index]
        target_key, target_value = target_cache[layer_index]
        source_key = _rope_shift(
            source_key[:, source_start : source_start + length],
            delta,
            theta,
        )
        source_value = source_value[:, source_start : source_start + length]
        target_key = target_key[:, target_start : target_start + length]
        target_value = target_value[:, target_start : target_start + length]
        key_deviation = _cosine_deviation_by_head(source_key, target_key)
        value_deviation = _cosine_deviation_by_head(source_value, target_value)
        key_l2 = _relative_l2_by_head(source_key, target_key)
        value_l2 = _relative_l2_by_head(source_value, target_value)
        kv_heads = len(key_deviation)
        repeats = num_attention_heads // kv_heads
        drift = torch.maximum(key_deviation, value_deviation).repeat_interleave(
            repeats
        )
        masses = torch.tensor(
            attention_by_layer[layer_index]["per_query_head_mass"],
            dtype=torch.float32,
        )
        product = masses * drift
        products.extend(float(value) for value in product.tolist())
        attention_values.extend(float(value) for value in masses.tolist())
        drift_values.extend(float(value) for value in drift.tolist())
        k_l2_values.extend(float(value) for value in key_l2.tolist())
        v_l2_values.extend(float(value) for value in value_l2.tolist())
        layer_rows.append(
            {
                "layer": layer_index,
                "attention_per_query_head_mass": [
                    float(value) for value in masses.tolist()
                ],
                "key_cosine_deviation_per_kv_head": [
                    float(value) for value in key_deviation.tolist()
                ],
                "value_cosine_deviation_per_kv_head": [
                    float(value) for value in value_deviation.tolist()
                ],
                "key_relative_l2_per_kv_head": [
                    float(value) for value in key_l2.tolist()
                ],
                "value_relative_l2_per_kv_head": [
                    float(value) for value in value_l2.tolist()
                ],
                "attention_times_drift_per_query_head": [
                    float(value) for value in product.tolist()
                ],
            }
        )
    return {
        "layer_metrics": layer_rows,
        "attention_mean": statistics.fmean(attention_values),
        "attention_q90": _quantile(attention_values, 0.9),
        "kv_cosine_drift_mean": statistics.fmean(drift_values),
        "kv_cosine_drift_q90": _quantile(drift_values, 0.9),
        "key_relative_l2_mean": statistics.fmean(k_l2_values),
        "value_relative_l2_mean": statistics.fmean(v_l2_values),
        "risk_product_mean": statistics.fmean(products),
        "risk_product_q90": _quantile(products, 0.9),
        "risk_product_max": max(products),
    }


def _cache_from_dense_prefix(
    *,
    model: Any,
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    prefix_tokens: int,
) -> DynamicCache:
    return DynamicCache(
        [
            (
                key[:, :prefix_tokens].unsqueeze(0).cuda(),
                value[:, :prefix_tokens].unsqueeze(0).cuda(),
            )
            for key, value in target_cache
        ],
        config=model.config,
    )


def _append_source_island(
    *,
    model: Any,
    cache: Any,
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    candidate: Mapping[str, Any],
    theta: float,
) -> Any:
    layers = []
    source_start = int(candidate["source_start"])
    target_start = int(candidate["target_start"])
    length = int(candidate["length"])
    delta = target_start - source_start
    for (target_key, target_value), (source_key, source_value) in zip(
        _layers(cache), source_cache, strict=True
    ):
        copied_key = _rope_shift(
            source_key[:, source_start : source_start + length].to(
                target_key.device
            ),
            delta,
            theta,
        ).unsqueeze(0)
        copied_value = source_value[
            :, source_start : source_start + length
        ].to(target_value.device).unsqueeze(0)
        layers.append(
            (
                torch.cat((target_key, copied_key), dim=2),
                torch.cat((target_value, copied_value), dim=2),
            )
        )
    return DynamicCache(layers, config=model.config)


@torch.inference_mode()
def _compose_splice(
    *,
    model: Any,
    target_ids: Sequence[int],
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    candidates: Sequence[Mapping[str, Any]],
    theta: float,
) -> torch.Tensor:
    ordered = sorted(candidates, key=lambda row: int(row["target_start"]))
    if not ordered:
        raise ValueError("at least one splice candidate is required")
    first_start = int(ordered[0]["target_start"])
    cache: Any = _cache_from_dense_prefix(
        model=model,
        target_cache=target_cache,
        prefix_tokens=first_start,
    )
    cursor = first_start
    logits: torch.Tensor | None = None
    for candidate in ordered:
        target_start = int(candidate["target_start"])
        if target_start < cursor:
            raise ValueError("splice candidates overlap")
        cache, gap_logits = _advance(model, cache, target_ids[cursor:target_start])
        if gap_logits is not None:
            logits = gap_logits
        cache = _append_source_island(
            model=model,
            cache=cache,
            source_cache=source_cache,
            candidate=candidate,
            theta=theta,
        )
        cursor = target_start + int(candidate["length"])
    for offset in range(cursor, len(target_ids), SPLICE_CHUNK_SIZE):
        cache, suffix_logits = _advance(
            model,
            cache,
            target_ids[offset : offset + SPLICE_CHUNK_SIZE],
        )
        if suffix_logits is not None:
            logits = suffix_logits
    if logits is None:
        raise RuntimeError("splice did not produce final logits")
    del cache
    torch.cuda.empty_cache()
    return logits


def _first_token_nll(logits: torch.Tensor, token_id: int) -> float:
    return float(-F.log_softmax(logits.float(), dim=-1)[token_id])


def _measurement_complete(row: Mapping[str, Any]) -> bool:
    if row.get("status") != "ok" or not row.get("candidates"):
        return False
    values = []
    for candidate in row["candidates"]:
        for key in (
            "attention_mean",
            "kv_cosine_drift_mean",
            "risk_product_q90",
            "causal_splice_logit_js",
            "answer_first_token_nll_delta",
        ):
            values.append(float(candidate[key]))
    triple = row.get("v46_composed") or {}
    values.extend(
        float(triple[key])
        for key in ("causal_splice_logit_js", "answer_first_token_nll_delta")
    )
    return all(math.isfinite(value) for value in values)


def measure(
    *,
    output: Path,
    max_cases: int,
    local_files_only: bool,
) -> dict[str, Any]:
    design_path = output / "DESIGN.json"
    registration = json.loads((output / "REGISTRATION.json").read_text())
    if registration["design_sha256"] != _sha256(design_path):
        raise ValueError("design changed after registration")
    design = json.loads(design_path.read_text())
    cases = design["cases"][:max_cases] if max_cases > 0 else design["cases"]
    observations_path = output / "OBSERVATIONS.jsonl"
    completed: set[str] = set()
    if observations_path.exists():
        for line in observations_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if _measurement_complete(row):
                    completed.add(str(row["case_id"]))
    pending = [row for row in cases if row["case_id"] not in completed]
    if not pending:
        return {
            "status": "COMPLETE",
            "selected_cases": len(cases),
            "completed_cases": len(completed),
            "new_cases": 0,
        }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU substitution is forbidden")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
        local_files_only=local_files_only,
    ).eval()
    theta = _model_theta(model.config)
    written = 0
    errors = []
    for case_index, case in enumerate(pending, 1):
        try:
            source_cache, _ = _dense_source(model, case["source_input_ids"])
            target_cache, dense_logits, attention = _target_forward(
                model=model,
                target_ids=case["target_input_ids"],
                candidates=case["candidates"],
            )
            dense_nll = _first_token_nll(
                dense_logits, int(case["answer_first_token_id"])
            )
            candidate_rows = []
            by_id = {}
            for candidate in case["candidates"]:
                candidate_id = str(candidate["candidate_id"])
                internal = _candidate_internal_metrics(
                    candidate=candidate,
                    attention=attention[candidate_id],
                    source_cache=source_cache,
                    target_cache=target_cache,
                    theta=theta,
                    num_attention_heads=int(model.config.num_attention_heads),
                )
                splice_logits = _compose_splice(
                    model=model,
                    target_ids=case["target_input_ids"],
                    target_cache=target_cache,
                    source_cache=source_cache,
                    candidates=[candidate],
                    theta=theta,
                )
                splice_nll = _first_token_nll(
                    splice_logits, int(case["answer_first_token_id"])
                )
                measured = {
                    **candidate,
                    **internal,
                    "position_fraction": int(candidate["target_start"])
                    / len(case["target_input_ids"]),
                    "causal_splice_logit_js": _js(dense_logits, splice_logits),
                    "causal_splice_top1_changed": int(dense_logits.argmax())
                    != int(splice_logits.argmax()),
                    "answer_first_token_nll": splice_nll,
                    "answer_first_token_nll_delta": splice_nll - dense_nll,
                }
                candidate_rows.append(measured)
                by_id[candidate_id] = candidate
                del splice_logits
            v46_candidates = [
                by_id[candidate_id] for candidate_id in case["v46_candidate_ids"]
            ]
            v46_logits = _compose_splice(
                model=model,
                target_ids=case["target_input_ids"],
                target_cache=target_cache,
                source_cache=source_cache,
                candidates=v46_candidates,
                theta=theta,
            )
            v46_nll = _first_token_nll(
                v46_logits, int(case["answer_first_token_id"])
            )
            row = {
                "status": "ok",
                "case_id": case["case_id"],
                "answer_first_token_id": case["answer_first_token_id"],
                "source_tokens": len(case["source_input_ids"]),
                "target_tokens": len(case["target_input_ids"]),
                "dense_answer_first_token_nll": dense_nll,
                "dense_top1_token_id": int(dense_logits.argmax()),
                "candidates": candidate_rows,
                "v46_composed": {
                    "candidate_ids": case["v46_candidate_ids"],
                    "causal_splice_logit_js": _js(dense_logits, v46_logits),
                    "causal_splice_top1_changed": int(dense_logits.argmax())
                    != int(v46_logits.argmax()),
                    "answer_first_token_nll": v46_nll,
                    "answer_first_token_nll_delta": v46_nll - dense_nll,
                },
            }
            if not _measurement_complete(row):
                raise RuntimeError("case produced incomplete/non-finite metrics")
            with observations_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
            print(
                json.dumps(
                    {
                        "case": case_index,
                        "case_id": case["case_id"],
                        "pending": len(pending),
                        "v46_js": row["v46_composed"][
                            "causal_splice_logit_js"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del source_cache, target_cache, dense_logits, v46_logits, attention
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as error:
            errors.append(
                {
                    "case_id": case["case_id"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(json.dumps(errors[-1], sort_keys=True), flush=True)
            break
    summary = {
        "status": "COMPLETE" if not errors and written == len(pending) else "PARTIAL",
        "selected_cases": len(cases),
        "previously_completed_cases": len(completed),
        "new_cases": written,
        "errors": errors,
        "observations": str(observations_path),
        "model": MODEL,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
    }
    write_json(output / "MEASUREMENT_STATUS.json", summary)
    return summary


@torch.inference_mode()
def _dense_source(
    model: Any, ids: Sequence[int]
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    inputs = torch.tensor([ids], device="cuda", dtype=torch.long)
    output = model(
        input_ids=inputs,
        use_cache=True,
        return_dict=True,
        logits_to_keep=1,
    )
    cache = _cpu_cache(output.past_key_values)
    logits = output.logits[0, -1].detach().float().cpu()
    del output, inputs
    torch.cuda.empty_cache()
    return cache, logits


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0:
        return math.nan
    return sum(
        a * b for a, b in zip(left_centered, right_centered, strict=True)
    ) / denominator


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_ranks(left), _ranks(right))


def _finite_mean(values: Sequence[float]) -> float:
    selected = [value for value in values if math.isfinite(value)]
    return statistics.fmean(selected) if selected else math.nan


def analyze(output: Path, m47_result_path: Path) -> dict[str, Any]:
    observations = [
        json.loads(line)
        for line in (output / "OBSERVATIONS.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not observations or any(not _measurement_complete(row) for row in observations):
        raise ValueError("observations are missing or incomplete")
    m47 = json.loads(m47_result_path.read_text()) if m47_result_path.exists() else None
    m47_rows = (
        {
            (row["arm"], row["case_id"]): row
            for row in m47["rows"]
        }
        if m47
        else {}
    )
    candidate_rows = [
        {"case_id": row["case_id"], **candidate}
        for row in observations
        for candidate in row["candidates"]
    ]
    signals = (
        "attention_mean",
        "kv_cosine_drift_mean",
        "kv_cosine_drift_q90",
        "value_relative_l2_mean",
        "risk_product_mean",
        "risk_product_q90",
        "risk_product_max",
        "position_fraction",
    )
    labels = (
        "causal_splice_logit_js",
        "answer_first_token_nll_delta",
    )
    correlations = {}
    for signal in signals:
        correlations[signal] = {}
        for label in labels:
            global_value = _spearman(
                [float(row[signal]) for row in candidate_rows],
                [float(row[label]) for row in candidate_rows],
            )
            by_position = defaultdict(list)
            for row in candidate_rows:
                by_position[int(row["context_index"])].append(row)
            fixed_position = [
                _spearman(
                    [float(row[signal]) for row in rows],
                    [float(row[label]) for row in rows],
                )
                for rows in by_position.values()
                if len(rows) >= 3
            ]
            by_case = defaultdict(list)
            for row in candidate_rows:
                by_case[str(row["case_id"])].append(row)
            within_case = [
                _spearman(
                    [float(row[signal]) for row in rows],
                    [float(row[label]) for row in rows],
                )
                for rows in by_case.values()
                if len(rows) >= 3
            ]
            correlations[signal][label] = {
                "global_spearman": global_value,
                "mean_fixed_position_spearman": _finite_mean(fixed_position),
                "mean_within_case_spearman": _finite_mean(within_case),
            }
    request_rows = []
    for row in observations:
        selected = [
            candidate
            for candidate in row["candidates"]
            if candidate["candidate_id"]
            in set(row["v46_composed"]["candidate_ids"])
        ]
        request = {
            "case_id": row["case_id"],
            "risk_product_q90_max": max(
                float(candidate["risk_product_q90"]) for candidate in selected
            ),
            "risk_product_q90_mean": statistics.fmean(
                float(candidate["risk_product_q90"]) for candidate in selected
            ),
            "kv_drift_q90_max": max(
                float(candidate["kv_cosine_drift_q90"]) for candidate in selected
            ),
            "v46_composed_logit_js": float(
                row["v46_composed"]["causal_splice_logit_js"]
            ),
            "v46_composed_nll_delta": float(
                row["v46_composed"]["answer_first_token_nll_delta"]
            ),
        }
        key = ("v46_recency_m47", row["case_id"])
        if key in m47_rows:
            m47_row = m47_rows[key]
            request.update(
                sglang_prediction_changed=not bool(
                    m47_row["prediction_identical_to_dense"]
                ),
                sglang_code_sim_delta=float(m47_row["code_sim"])
                - float(m47_row["dense_code_sim"]),
                sglang_exact_damage=bool(m47_row["dense_exact_line"])
                and not bool(m47_row["exact_line"]),
            )
        request_rows.append(request)
    request_correlations = {}
    for signal in (
        "risk_product_q90_max",
        "risk_product_q90_mean",
        "kv_drift_q90_max",
    ):
        request_correlations[signal] = {
            "composed_logit_js_spearman": _spearman(
                [row[signal] for row in request_rows],
                [row["v46_composed_logit_js"] for row in request_rows],
            ),
            "composed_nll_delta_spearman": _spearman(
                [row[signal] for row in request_rows],
                [row["v46_composed_nll_delta"] for row in request_rows],
            ),
        }
        if request_rows and "sglang_code_sim_delta" in request_rows[0]:
            request_correlations[signal]["sglang_abs_code_sim_change_spearman"] = (
                _spearman(
                    [row[signal] for row in request_rows],
                    [abs(row["sglang_code_sim_delta"]) for row in request_rows],
                )
            )
    value = {
        "status": "COMPLETE",
        "cases": len(observations),
        "candidate_observations": len(candidate_rows),
        "signals": correlations,
        "v46_request_correlations": request_correlations,
        "v46_requests": request_rows,
        "scope": (
            "oracle model-internal motivation; not online latency or functional "
            "accuracy evidence"
        ),
    }
    write_json(output / "RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--limit", type=int, default=8)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    measure_parser.add_argument("--max-cases", type=int, default=0)
    measure_parser.add_argument("--local-files-only", action="store_true")
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    analyze_parser.add_argument(
        "--m47-result", type=Path, default=DEFAULT_M47_RESULT
    )
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.workload, args.output, args.limit)
    elif args.command == "measure":
        value = measure(
            output=args.output,
            max_cases=args.max_cases,
            local_files_only=args.local_files_only,
        )
    else:
        value = analyze(args.output, args.m47_result)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
