#!/usr/bin/env python3
"""Validate an attention-weighted K/V perturbation bound on real reuse islands.

The experiment reuses the frozen 26-case cohort built from the exact-same-prompt
V40 campaign.  It is a mechanism experiment, not an accuracy or latency result.
For each real copied observation island it:

1. computes Dense source and Dense target K/V with a local 3B coder model;
2. replaces the target island with source K only, source V only, or both;
3. measures the local pre-output-projection attention error for final prompt
   queries and the end-to-end final-logit JS after a physical suffix replay;
4. checks an exact finite perturbation bound and evaluates whether
   attention-weighted drift ranks local harm better than raw drift.

The 3B model is a mechanism proxy because the native 30B AWQ SGLang model does
not expose full attention tensors through the available Transformers stack.
The source/target prompt text, copied observation identity, and RoPE-shifted
middle-span splice semantics are preserved from the current SGLang method.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


ROOT = Path("/home/gfy/CodeMAS_Project")
SOURCE_DESIGN = (
    ROOT
    / "kvflow-artifacts/impactkv_global_block_attention_20260806/"
    "frozen26_r2/DESIGN.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_attention_kv_bound_20260806/frozen26"
)
MODEL = Path(
    "/home/gfy/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/"
    "488639f1ff808d1d3d0ba301aef8c11461451ec5"
)
PROBE_LAYERS = (0, 8, 17, 26, 35)
QUERY_TAIL_TOKENS = 32
FORWARD_CHUNK = 512
BOUND_TOLERANCE = 2e-5


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o644)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _layers(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return [(row[0], row[1]) for row in cache]


def _cpu_cache(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            key[0].detach().to("cpu", dtype=torch.bfloat16).contiguous(),
            value[0].detach().to("cpu", dtype=torch.bfloat16).contiguous(),
        )
        for key, value in _layers(cache)
    ]


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def _rope_shift(keys: torch.Tensor, delta: int, theta: float) -> torch.Tensor:
    if delta == 0 or not keys.numel():
        return keys
    dim = keys.shape[-1]
    inv = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, device=keys.device, dtype=torch.float32)
            / dim
        )
    )
    frequency = delta * inv
    cosine = torch.cat((frequency.cos(), frequency.cos()))
    sine = torch.cat((frequency.sin(), frequency.sin()))
    return (
        keys.float() * cosine + _rotate_half(keys.float()) * sine
    ).to(keys.dtype)


def _model_theta(config: Any) -> float:
    if hasattr(config, "rope_theta"):
        return float(config.rope_theta)
    parameters = getattr(config, "rope_parameters", None) or {}
    return float(parameters.get("rope_theta", 1_000_000.0))


def _softmax_l1_bound(epsilon: torch.Tensor) -> torch.Tensor:
    """L1 bound when every logit perturbation is at most epsilon in magnitude.

    If ||z' - z||_infinity <= epsilon, the probability likelihood ratio is in
    [exp(-2 epsilon), exp(2 epsilon)].  Hence total variation is at most
    tanh(epsilon), and L1 distance is at most 2*tanh(epsilon).
    """

    return 2.0 * torch.tanh(epsilon.float())


def _softmax_island_l1_bound(
    epsilon: torch.Tensor, dense_island_mass: torch.Tensor
) -> torch.Tensor:
    """Mass-aware L1 bound when only one island's logits are perturbed.

    In addition to the global ``2*tanh(epsilon)`` bound, writing the new
    softmax in terms of the old distribution gives

        ||a' - a||_1 <= 2 A_S (exp(2 epsilon) - 1),

    where ``A_S`` is the Dense attention mass on the perturbed island.  The
    minimum of the two valid bounds keeps the expression finite for large
    epsilon and exposes the useful low-attention regime.
    """

    epsilon = epsilon.float()
    dense_island_mass = dense_island_mass.float()
    mass_bound = 2.0 * dense_island_mass * torch.expm1(
        (2.0 * epsilon).clamp(max=20.0)
    )
    return torch.minimum(_softmax_l1_bound(epsilon), mass_bound.clamp(max=2.0))


def _js(left: torch.Tensor, right: torch.Tensor) -> float:
    left_log = F.log_softmax(left.float(), dim=-1)
    right_log = F.log_softmax(right.float(), dim=-1)
    left_p, right_p = left_log.exp(), right_log.exp()
    middle = 0.5 * (left_p + right_p)
    value = 0.5 * (
        (left_p * (left_log - middle.log())).sum()
        + (right_p * (right_log - middle.log())).sum()
    )
    return float(value)


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
    }


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


def _head_query_bound_metrics(
    *,
    dense_scores: torch.Tensor,
    stale_scores: torch.Tensor,
    dense_values: torch.Tensor,
    stale_values: torch.Tensor,
    island_start: int,
    island_end: int,
) -> dict[str, torch.Tensor]:
    """Return per-head/query perturbation terms for one attention layer.

    Shapes are [heads, queries, keys] for scores and
    [heads, keys, head_dim] for values.  Scores must already include the causal
    mask.  Only the island slice may differ between dense and stale tensors.
    """

    dense_weights = torch.softmax(dense_scores.float(), dim=-1)
    stale_weights = torch.softmax(stale_scores.float(), dim=-1)
    dense_v = dense_values.float()
    stale_v = stale_values.float()
    dense_output = torch.matmul(dense_weights, dense_v)
    key_only_output = torch.matmul(stale_weights, dense_v)
    value_only_output = torch.matmul(dense_weights, stale_v)
    stale_output = torch.matmul(stale_weights, stale_v)

    island = slice(island_start, island_end)
    delta_scores = stale_scores[..., island].float() - dense_scores[..., island].float()
    epsilon = delta_scores.abs().amax(dim=-1)
    delta_values = stale_v[:, island] - dense_v[:, island]
    delta_value_norm = delta_values.norm(dim=-1)
    max_delta_value = delta_value_norm.amax(dim=-1).unsqueeze(-1)
    max_dense_value = dense_v.norm(dim=-1).amax(dim=-1).unsqueeze(-1)
    dense_mass = dense_weights[..., island].sum(dim=-1)
    stale_mass = stale_weights[..., island].sum(dim=-1)
    attention_l1 = (stale_weights - dense_weights).abs().sum(dim=-1)

    actual_key = (key_only_output - dense_output).norm(dim=-1)
    actual_value = (value_only_output - dense_output).norm(dim=-1)
    actual_kv = (stale_output - dense_output).norm(dim=-1)
    value_triangle = (
        stale_weights[..., island]
        * delta_value_norm.unsqueeze(1)
    ).sum(dim=-1)
    exact_finite_bound = value_triangle + attention_l1 * max_dense_value
    analytic_bound = (
        stale_mass * max_delta_value
        + _softmax_l1_bound(epsilon) * max_dense_value
    )
    mass_aware_analytic_bound = (
        stale_mass * max_delta_value
        + _softmax_island_l1_bound(epsilon, dense_mass) * max_dense_value
    )
    first_order_score = (
        dense_mass * max_delta_value
        + 2.0 * dense_mass * epsilon * max_dense_value
    )
    scale = dense_output.norm(dim=-1).clamp_min(1e-8)

    return {
        "dense_attention_mass": dense_mass,
        "stale_attention_mass": stale_mass,
        "key_logit_epsilon": epsilon,
        "max_delta_value_norm": max_delta_value.expand_as(epsilon),
        "max_dense_value_norm": max_dense_value.expand_as(epsilon),
        "attention_l1": attention_l1,
        "actual_key_output_delta": actual_key,
        "actual_value_output_delta": actual_value,
        "actual_kv_output_delta": actual_kv,
        "actual_key_output_relative": actual_key / scale,
        "actual_value_output_relative": actual_value / scale,
        "actual_kv_output_relative": actual_kv / scale,
        "exact_finite_bound": exact_finite_bound,
        "analytic_bound": analytic_bound,
        "mass_aware_analytic_bound": mass_aware_analytic_bound,
        "first_order_score": first_order_score,
        "exact_finite_bound_relative": exact_finite_bound / scale,
        "analytic_bound_relative": analytic_bound / scale,
        "mass_aware_analytic_bound_relative": mass_aware_analytic_bound / scale,
        "first_order_score_relative": first_order_score / scale,
    }


def prepare(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    source = json.loads(SOURCE_DESIGN.read_text(encoding="utf-8"))
    cases = source["cases"]
    if len(cases) != 26 or len({row["instance_id"] for row in cases}) != 13:
        raise ValueError("source design must contain 26 cases from 13 tasks")
    design = {
        "source_design": str(SOURCE_DESIGN),
        "source_design_sha256": _sha256(SOURCE_DESIGN),
        "case_ids": [row["case_id"] for row in cases],
        "canary_case_ids": source["canary_case_ids"],
        "analysis_model": str(MODEL),
        "probe_layers_zero_based": list(PROBE_LAYERS),
        "query_tail_tokens": QUERY_TAIL_TOKENS,
    }
    output.mkdir(parents=True)
    design_path = output / "DESIGN.json"
    _write_json(design_path, design)
    registration = {
        "status": "REGISTERED_BEFORE_GPU",
        "purpose": (
            "validate an attention-weighted finite K/V perturbation bound and "
            "separate K-only, V-only, and K+V causal effects on real V40 islands"
        ),
        "design_sha256": _sha256(design_path),
        "source_design_sha256": design["source_design_sha256"],
        "cases": len(cases),
        "tasks": len({row["instance_id"] for row in cases}),
        "canary_case_ids": design["canary_case_ids"],
        "mechanism_claim": (
            "For a fixed query, stale-island attention-output error decomposes "
            "into a value-content term and a key-induced attention-redistribution "
            "term. Attention mass and K/V deviation jointly control local harm."
        ),
        "finite_bound": (
            "||o'-o||_2 <= sum_{i in S} a'_i ||dv_i||_2 + "
            "||a'-a||_1 max_i ||v_i||_2 <= A'_S max||dv|| + "
            "2 tanh(epsilon) max||v||, where epsilon=max_{i in S}|q·dk_i|/sqrt(d)."
        ),
        "mass_aware_refinement": (
            "When only island S changes, ||a'-a||_1 is also at most "
            "2 A_S (exp(2 epsilon)-1); use the minimum with 2 tanh(epsilon)."
        ),
        "primary_gates": {
            "finite_bound_coverage_min": 1.0,
            "bound_tolerance": BOUND_TOLERANCE,
            "attention_weighted_drift_local_spearman_min": 0.30,
            "attention_weighted_minus_raw_drift_spearman_min": 0.05,
            "local_delta_vs_end_to_end_js_spearman_min": 0.30,
        },
        "canary_expansion_rule": (
            "expand after all frozen canary cases finish with finite K/V component "
            "labels and no finite-bound violation above tolerance; result direction "
            "is not an expansion gate"
        ),
        "scope_warnings": [
            "mechanism motivation, not task accuracy or latency",
            "full Dense target K/V and attention are offline oracles",
            "Qwen2.5-Coder-3B proxy on unchanged real V40 source/target prompts",
            "the finite bound applies to one fixed-query attention layer before output projection",
            "end-to-end K/V splices preserve current middle-span RoPE semantics",
        ],
        "protected": {
            "paper_modified": False,
            "prefetch_modified": False,
            "old_dirty_checkout_modified": False,
            "old_preregistration_thresholds_modified": False,
        },
    }
    _write_json(output / "REGISTRATION.json", registration)
    return registration


@torch.inference_mode()
def _dense_forward(
    model: Any,
    ids: Sequence[int],
    *,
    capture_queries: bool,
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor]],
    torch.Tensor,
    dict[int, torch.Tensor],
]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    if capture_queries:
        for layer_index in PROBE_LAYERS:
            def capture(
                _module: Any,
                args: tuple[Any, ...],
                index: int = layer_index,
            ) -> None:
                captured[index] = args[0][:, -QUERY_TAIL_TOKENS:].detach()

            handles.append(model.model.layers[layer_index].register_forward_pre_hook(capture))
    inputs = torch.tensor([ids], device="cuda", dtype=torch.long)
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
    if capture_queries and set(captured) != set(PROBE_LAYERS):
        raise RuntimeError("not all query hidden states were captured")
    cache = _cpu_cache(output.past_key_values)
    logits = output.logits[0, -1].detach().float().cpu()
    del output, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return cache, logits, captured


def _cosine_drift(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return 1.0 - (
        F.normalize(left.float(), dim=-1)
        * F.normalize(right.float(), dim=-1)
    ).sum(dim=-1).mean(dim=-1)


@torch.inference_mode()
def _local_layer_metrics(
    *,
    model: Any,
    case: Mapping[str, Any],
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    captured: Mapping[int, torch.Tensor],
    theta: float,
) -> list[dict[str, Any]]:
    source_start = int(case["source_start"])
    target_start = int(case["target_start"])
    length = int(case["length"])
    target_tokens = len(case["target_input_ids"])
    query_tokens = min(QUERY_TAIL_TOKENS, target_tokens)
    query_positions = torch.arange(
        target_tokens - query_tokens,
        target_tokens,
        device="cuda",
        dtype=torch.long,
    ).unsqueeze(0)
    key_positions = torch.arange(target_tokens, device="cuda")
    rows = []
    for layer_index in PROBE_LAYERS:
        attention = model.model.layers[layer_index].self_attn
        num_heads = int(model.config.num_attention_heads)
        num_kv_heads = int(model.config.num_key_value_heads)
        groups = num_heads // num_kv_heads
        head_dim = int(getattr(model.config, "head_dim", 0)) or (
            int(model.config.hidden_size) // num_heads
        )
        hidden = captured[layer_index][:, -query_tokens:]
        query = attention.q_proj(hidden).view(
            1, query_tokens, num_heads, head_dim
        ).transpose(1, 2)
        if hasattr(attention, "q_norm"):
            query = attention.q_norm(query)
        cosine, sine = model.model.rotary_emb(hidden, query_positions)
        query = query * cosine.unsqueeze(1) + _rotate_half(query) * sine.unsqueeze(1)

        target_key, target_value = target_cache[layer_index]
        source_key, source_value = source_cache[layer_index]
        target_key = target_key.to("cuda")
        target_value = target_value.to("cuda")
        shifted_key = _rope_shift(
            source_key[:, source_start : source_start + length].to("cuda"),
            target_start - source_start,
            theta,
        )
        stale_value_island = source_value[
            :, source_start : source_start + length
        ].to("cuda")
        stale_key = target_key.clone()
        stale_value = target_value.clone()
        stale_key[:, target_start : target_start + length] = shifted_key
        stale_value[:, target_start : target_start + length] = stale_value_island

        dense_key_h = target_key.repeat_interleave(groups, dim=0)
        stale_key_h = stale_key.repeat_interleave(groups, dim=0)
        dense_value_h = target_value.repeat_interleave(groups, dim=0)
        stale_value_h = stale_value.repeat_interleave(groups, dim=0)
        dense_scores = torch.matmul(
            query[0].float(), dense_key_h.float().transpose(-1, -2)
        ) / math.sqrt(head_dim)
        stale_scores = torch.matmul(
            query[0].float(), stale_key_h.float().transpose(-1, -2)
        ) / math.sqrt(head_dim)
        causal = key_positions.view(1, 1, -1) > query_positions.view(1, -1, 1)
        dense_scores = dense_scores.masked_fill(causal, -torch.inf)
        stale_scores = stale_scores.masked_fill(causal, -torch.inf)
        metrics = _head_query_bound_metrics(
            dense_scores=dense_scores,
            stale_scores=stale_scores,
            dense_values=dense_value_h,
            stale_values=stale_value_h,
            island_start=target_start,
            island_end=target_start + length,
        )
        key_drift = _cosine_drift(shifted_key, target_key[:, target_start : target_start + length])
        value_drift = _cosine_drift(
            stale_value_island,
            target_value[:, target_start : target_start + length],
        )
        drift = torch.maximum(key_drift, value_drift).repeat_interleave(groups)
        attention_mass = metrics["dense_attention_mass"]
        attention_times_drift = attention_mass * drift.unsqueeze(-1)
        actual = metrics["actual_kv_output_delta"]
        exact_bound = metrics["exact_finite_bound"]
        analytic_bound = metrics["analytic_bound"]
        mass_aware_bound = metrics["mass_aware_analytic_bound"]
        analytic_tolerance = BOUND_TOLERANCE * torch.maximum(
            torch.ones_like(actual), analytic_bound
        )
        exact_tolerance = BOUND_TOLERANCE * torch.maximum(
            torch.ones_like(actual), exact_bound
        )
        analytic_violations = actual > analytic_bound + analytic_tolerance
        exact_violations = actual > exact_bound + exact_tolerance
        mass_aware_tolerance = BOUND_TOLERANCE * torch.maximum(
            torch.ones_like(actual), mass_aware_bound
        )
        mass_aware_violations = actual > mass_aware_bound + mass_aware_tolerance

        # A raw ratio is ill-conditioned when both the measured error and its
        # bound are at floating-point zero.  Report ratios only above the same
        # absolute numerical floor used by the preregistered coverage check;
        # max excess and violation counts remain the authoritative checks.
        reliable_exact = exact_bound >= BOUND_TOLERANCE
        reliable_analytic = analytic_bound >= BOUND_TOLERANCE
        reliable_mass_aware = mass_aware_bound >= BOUND_TOLERANCE

        def reliable_ratio(bound: torch.Tensor, mask: torch.Tensor) -> float:
            if not bool(mask.any()):
                return 0.0
            return float((actual[mask] / bound[mask]).max())

        row: dict[str, Any] = {
            "layer": layer_index,
            "points": int(actual.numel()),
            "key_cosine_drift_mean": float(key_drift.mean()),
            "value_cosine_drift_mean": float(value_drift.mean()),
            "raw_kv_drift_mean": float(drift.mean()),
            "attention_mass_mean": float(attention_mass.mean()),
            "attention_times_drift_mean": float(attention_times_drift.mean()),
            "key_logit_epsilon_mean": float(metrics["key_logit_epsilon"].mean()),
            "actual_key_output_delta_mean": float(metrics["actual_key_output_delta"].mean()),
            "actual_value_output_delta_mean": float(metrics["actual_value_output_delta"].mean()),
            "actual_kv_output_delta_mean": float(actual.mean()),
            "actual_kv_output_relative_mean": float(metrics["actual_kv_output_relative"].mean()),
            "first_order_score_mean": float(metrics["first_order_score"].mean()),
            "first_order_score_relative_mean": float(metrics["first_order_score_relative"].mean()),
            "exact_finite_bound_mean": float(exact_bound.mean()),
            "analytic_bound_mean": float(analytic_bound.mean()),
            "analytic_bound_relative_mean": float(metrics["analytic_bound_relative"].mean()),
            "mass_aware_analytic_bound_mean": float(mass_aware_bound.mean()),
            "mass_aware_analytic_bound_relative_mean": float(
                metrics["mass_aware_analytic_bound_relative"].mean()
            ),
            "bound_violations": int(analytic_violations.sum().item()),
            "exact_bound_violations": int(exact_violations.sum().item()),
            "mass_aware_bound_violations": int(mass_aware_violations.sum().item()),
            "max_actual_minus_analytic_bound": float((actual - analytic_bound).max()),
            "max_actual_minus_exact_bound": float((actual - exact_bound).max()),
            "max_actual_minus_mass_aware_bound": float(
                (actual - mass_aware_bound).max()
            ),
            "max_actual_to_analytic_bound_ratio": reliable_ratio(
                analytic_bound, reliable_analytic
            ),
            "max_actual_to_exact_bound_ratio": reliable_ratio(
                exact_bound, reliable_exact
            ),
            "max_actual_to_mass_aware_bound_ratio": reliable_ratio(
                mass_aware_bound, reliable_mass_aware
            ),
        }
        rows.append(row)
        del (
            hidden,
            query,
            target_key,
            target_value,
            shifted_key,
            stale_value_island,
            stale_key,
            stale_value,
            dense_key_h,
            stale_key_h,
            dense_value_h,
            stale_value_h,
            dense_scores,
            stale_scores,
            metrics,
        )
        torch.cuda.empty_cache()
    return rows


def _cache_from_prefix(
    model: Any,
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    length: int,
) -> DynamicCache:
    return DynamicCache(
        [
            (
                key[:, :length].unsqueeze(0).to("cuda"),
                value[:, :length].unsqueeze(0).to("cuda"),
            )
            for key, value in target_cache
        ],
        config=model.config,
    )


def _append_component_island(
    *,
    model: Any,
    cache: Any,
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    source_start: int,
    target_start: int,
    length: int,
    theta: float,
    mode: str,
) -> DynamicCache:
    if mode not in {"key_only", "value_only", "kv"}:
        raise ValueError(mode)
    layers = []
    for (prefix_key, prefix_value), (source_key, source_value), (
        target_key,
        target_value,
    ) in zip(_layers(cache), source_cache, target_cache, strict=True):
        if mode in {"key_only", "kv"}:
            island_key = _rope_shift(
                source_key[:, source_start : source_start + length].to("cuda"),
                target_start - source_start,
                theta,
            )
        else:
            island_key = target_key[
                :, target_start : target_start + length
            ].to("cuda")
        if mode in {"value_only", "kv"}:
            island_value = source_value[
                :, source_start : source_start + length
            ].to("cuda")
        else:
            island_value = target_value[
                :, target_start : target_start + length
            ].to("cuda")
        layers.append(
            (
                torch.cat((prefix_key, island_key.unsqueeze(0)), dim=2),
                torch.cat((prefix_value, island_value.unsqueeze(0)), dim=2),
            )
        )
    return DynamicCache(layers, config=model.config)


@torch.inference_mode()
def _physical_splice_logits(
    *,
    model: Any,
    case: Mapping[str, Any],
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    theta: float,
    mode: str,
) -> torch.Tensor:
    start = int(case["target_start"])
    end = start + int(case["length"])
    cache = _cache_from_prefix(model, target_cache, start)
    cache = _append_component_island(
        model=model,
        cache=cache,
        source_cache=source_cache,
        target_cache=target_cache,
        source_start=int(case["source_start"]),
        target_start=start,
        length=int(case["length"]),
        theta=theta,
        mode=mode,
    )
    logits = None
    for offset in range(end, len(case["target_input_ids"]), FORWARD_CHUNK):
        output = model(
            input_ids=torch.tensor(
                [case["target_input_ids"][offset : offset + FORWARD_CHUNK]],
                device="cuda",
                dtype=torch.long,
            ),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        logits = output.logits[0, -1].detach().float().cpu()
        del output
    if logits is None:
        raise RuntimeError("physical splice suffix produced no logits")
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    return logits


def _measurement_complete(row: Mapping[str, Any]) -> bool:
    if row.get("status") != "ok" or not row.get("layers"):
        return False
    values = []
    for layer in row["layers"]:
        values.extend(
            float(layer[key])
            for key in (
                "attention_times_drift_mean",
                "actual_kv_output_delta_mean",
                "analytic_bound_mean",
                "mass_aware_analytic_bound_mean",
            )
        )
    for mode in ("key_only", "value_only", "kv"):
        values.append(float(row["physical_splice"][mode]["final_logit_js"]))
    return all(math.isfinite(value) for value in values)


def measure(output: Path, max_cases: int, canary_only: bool) -> dict[str, Any]:
    design_path = output / "DESIGN.json"
    registration = json.loads((output / "REGISTRATION.json").read_text())
    if registration["design_sha256"] != _sha256(design_path):
        raise ValueError("design changed after registration")
    design = json.loads(design_path.read_text())
    if design["source_design_sha256"] != _sha256(SOURCE_DESIGN):
        raise ValueError("source design changed")
    source = json.loads(SOURCE_DESIGN.read_text())
    cases_by_id = {row["case_id"]: row for row in source["cases"]}
    selected_ids = (
        design["canary_case_ids"] if canary_only else design["case_ids"]
    )
    if max_cases > 0:
        selected_ids = selected_ids[:max_cases]
    selected = [cases_by_id[case_id] for case_id in selected_ids]
    observations_path = output / "OBSERVATIONS.jsonl"
    completed = set()
    if observations_path.exists():
        for line in observations_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if _measurement_complete(row):
                    completed.add(str(row["case_id"]))
    pending = [row for row in selected if row["case_id"] not in completed]
    if not pending:
        return {
            "status": "COMPLETE",
            "selected_cases": len(selected),
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
        local_files_only=True,
    ).eval()
    theta = _model_theta(model.config)
    written = 0
    errors = []
    for index, case in enumerate(pending, 1):
        try:
            source_cache, _, _ = _dense_forward(
                model, case["source_input_ids"], capture_queries=False
            )
            target_cache, dense_logits, captured = _dense_forward(
                model, case["target_input_ids"], capture_queries=True
            )
            layers = _local_layer_metrics(
                model=model,
                case=case,
                source_cache=source_cache,
                target_cache=target_cache,
                captured=captured,
                theta=theta,
            )
            physical = {}
            for mode in ("key_only", "value_only", "kv"):
                logits = _physical_splice_logits(
                    model=model,
                    case=case,
                    source_cache=source_cache,
                    target_cache=target_cache,
                    theta=theta,
                    mode=mode,
                )
                physical[mode] = {
                    "final_logit_js": _js(dense_logits, logits),
                    "top1_changed": int(dense_logits.argmax()) != int(logits.argmax()),
                }
                del logits
            row = {
                "status": "ok",
                "case_id": case["case_id"],
                "instance_id": case["instance_id"],
                "source_tokens": len(case["source_input_ids"]),
                "target_tokens": len(case["target_input_ids"]),
                "copy_tokens": int(case["length"]),
                "source_start": int(case["source_start"]),
                "target_start": int(case["target_start"]),
                "layers": layers,
                "physical_splice": physical,
            }
            if not _measurement_complete(row):
                raise RuntimeError("case produced incomplete metrics")
            with observations_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
            print(
                json.dumps(
                    {
                        "case": index,
                        "pending": len(pending),
                        "case_id": case["case_id"],
                        "kv_js": physical["kv"]["final_logit_js"],
                        "analytic_bound_violations": sum(
                            layer["bound_violations"] for layer in layers
                        ),
                        "exact_bound_violations": sum(
                            layer["exact_bound_violations"] for layer in layers
                        ),
                        "mass_aware_bound_violations": sum(
                            layer["mass_aware_bound_violations"] for layer in layers
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del source_cache, target_cache, dense_logits, captured
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
        "selected_cases": len(selected),
        "previously_completed_cases": len(completed),
        "new_cases": written,
        "errors": errors,
        "model": str(MODEL),
        "dtype": "bfloat16",
        "observations": str(observations_path),
    }
    _write_json(output / "MEASUREMENT_STATUS.json", summary)
    return summary


def analyze(output: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (output / "OBSERVATIONS.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not rows or any(not _measurement_complete(row) for row in rows):
        raise ValueError("observations are missing or incomplete")
    layer_rows = [
        {"case_id": row["case_id"], "instance_id": row["instance_id"], **layer}
        for row in rows
        for layer in row["layers"]
    ]
    target = [float(row["actual_kv_output_relative_mean"]) for row in layer_rows]
    signals = {
        "attention_mass": [float(row["attention_mass_mean"]) for row in layer_rows],
        "raw_kv_drift": [float(row["raw_kv_drift_mean"]) for row in layer_rows],
        "attention_times_drift": [
            float(row["attention_times_drift_mean"]) for row in layer_rows
        ],
        "first_order_score": [
            float(row["first_order_score_relative_mean"]) for row in layer_rows
        ],
        "analytic_bound": [
            float(row["analytic_bound_relative_mean"]) for row in layer_rows
        ],
        "mass_aware_analytic_bound": [
            float(row["mass_aware_analytic_bound_relative_mean"])
            for row in layer_rows
        ],
    }
    local_correlations = {
        name: _spearman(values, target) for name, values in signals.items()
    }
    layer_correlations = {}
    for layer_index in PROBE_LAYERS:
        selected = [
            row for row in layer_rows if int(row["layer"]) == layer_index
        ]
        selected_target = [
            float(row["actual_kv_output_relative_mean"]) for row in selected
        ]
        layer_correlations[str(layer_index)] = {
            "raw_kv_drift": _spearman(
                [float(row["raw_kv_drift_mean"]) for row in selected],
                selected_target,
            ),
            "attention_times_drift": _spearman(
                [float(row["attention_times_drift_mean"]) for row in selected],
                selected_target,
            ),
            "first_order_score": _spearman(
                [float(row["first_order_score_relative_mean"]) for row in selected],
                selected_target,
            ),
        }
    case_rows = []
    for row in rows:
        case_rows.append(
            {
                "case_id": row["case_id"],
                "instance_id": row["instance_id"],
                "local_kv_output_relative_mean": statistics.fmean(
                    float(layer["actual_kv_output_relative_mean"])
                    for layer in row["layers"]
                ),
                "attention_times_drift_mean": statistics.fmean(
                    float(layer["attention_times_drift_mean"])
                    for layer in row["layers"]
                ),
                "first_order_score_relative_mean": statistics.fmean(
                    float(layer["first_order_score_relative_mean"])
                    for layer in row["layers"]
                ),
                "key_only_js": float(row["physical_splice"]["key_only"]["final_logit_js"]),
                "value_only_js": float(row["physical_splice"]["value_only"]["final_logit_js"]),
                "kv_js": float(row["physical_splice"]["kv"]["final_logit_js"]),
            }
        )
    end_to_end = [row["kv_js"] for row in case_rows]
    case_correlations = {
        "local_kv_output_vs_kv_js": _spearman(
            [row["local_kv_output_relative_mean"] for row in case_rows],
            end_to_end,
        ),
        "attention_times_drift_vs_kv_js": _spearman(
            [row["attention_times_drift_mean"] for row in case_rows],
            end_to_end,
        ),
        "first_order_score_vs_kv_js": _spearman(
            [row["first_order_score_relative_mean"] for row in case_rows],
            end_to_end,
        ),
    }
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        by_task[row["instance_id"]].append(row)
    pair_agreement = {}
    for predictor in (
        "local_kv_output_relative_mean",
        "attention_times_drift_mean",
        "first_order_score_relative_mean",
    ):
        trials = wins = ties = 0
        for task_rows in by_task.values():
            if len(task_rows) != 2:
                continue
            left, right = task_rows
            predicted = math.copysign(1, left[predictor] - right[predictor]) if left[predictor] != right[predictor] else 0
            actual = math.copysign(1, left["kv_js"] - right["kv_js"]) if left["kv_js"] != right["kv_js"] else 0
            if not predicted or not actual:
                ties += 1
                continue
            trials += 1
            wins += int(predicted == actual)
        pair_agreement[predictor] = {
            "wins": wins,
            "trials": trials,
            "ties": ties,
            "fraction": wins / trials if trials else math.nan,
        }
    total_points = sum(int(row["points"]) for row in layer_rows)
    analytic_violations = sum(
        int(row["bound_violations"]) for row in layer_rows
    )
    exact_violations = sum(
        int(row["exact_bound_violations"]) for row in layer_rows
    )
    mass_aware_violations = sum(
        int(row["mass_aware_bound_violations"]) for row in layer_rows
    )
    max_analytic_excess = max(
        float(row["max_actual_minus_analytic_bound"]) for row in layer_rows
    )
    max_exact_excess = max(
        float(row["max_actual_minus_exact_bound"]) for row in layer_rows
    )
    max_mass_aware_excess = max(
        float(row["max_actual_minus_mass_aware_bound"]) for row in layer_rows
    )
    gates = json.loads((output / "REGISTRATION.json").read_text())["primary_gates"]
    weighted = local_correlations["attention_times_drift"]
    raw = local_correlations["raw_kv_drift"]
    local_to_js = case_correlations["local_kv_output_vs_kv_js"]
    gate_results = {
        "finite_bound_coverage": (
            analytic_violations == 0
            and exact_violations == 0
            and mass_aware_violations == 0
        ),
        "attention_weighted_drift_local": weighted
        >= gates["attention_weighted_drift_local_spearman_min"],
        "attention_weighting_adds_value": weighted - raw
        >= gates["attention_weighted_minus_raw_drift_spearman_min"],
        "local_delta_links_to_end_to_end": local_to_js
        >= gates["local_delta_vs_end_to_end_js_spearman_min"],
    }
    component = {
        mode: _quantiles([row[f"{mode}_js"] for row in case_rows])
        for mode in ("key_only", "value_only", "kv")
    }
    component["top1_changed_counts"] = {
        mode: sum(
            int(row["physical_splice"][mode]["top1_changed"]) for row in rows
        )
        for mode in ("key_only", "value_only", "kv")
    }
    component["dominance_counts"] = {
        "key_only_gt_value_only": sum(
            row["key_only_js"] > row["value_only_js"] for row in case_rows
        ),
        "value_only_gt_key_only": sum(
            row["value_only_js"] > row["key_only_js"] for row in case_rows
        ),
        "equal": sum(
            row["value_only_js"] == row["key_only_js"] for row in case_rows
        ),
    }
    result = {
        "status": "COMPLETE",
        "decision": (
            "SUPPORTED_LOCAL_MECHANISM"
            if all(gate_results.values())
            else "PARTIAL_OR_FALSIFIED"
        ),
        "cases": len(rows),
        "tasks": len({row["instance_id"] for row in rows}),
        "case_layer_points": len(layer_rows),
        "head_query_layer_points": total_points,
        "finite_bound": {
            "analytic_violations": analytic_violations,
            "exact_violations": exact_violations,
            "mass_aware_violations": mass_aware_violations,
            "joint_coverage": 1.0
            - max(
                analytic_violations,
                exact_violations,
                mass_aware_violations,
            )
            / total_points,
            "max_actual_minus_analytic_bound": max_analytic_excess,
            "max_actual_minus_exact_bound": max_exact_excess,
            "max_actual_minus_mass_aware_bound": max_mass_aware_excess,
            "max_actual_to_analytic_bound_ratio": max(
                float(row["max_actual_to_analytic_bound_ratio"])
                for row in layer_rows
            ),
            "max_actual_to_exact_bound_ratio": max(
                float(row["max_actual_to_exact_bound_ratio"])
                for row in layer_rows
            ),
            "max_actual_to_mass_aware_bound_ratio": max(
                float(row["max_actual_to_mass_aware_bound_ratio"])
                for row in layer_rows
            ),
        },
        "local_correlations_with_attention_output_error": local_correlations,
        "local_correlations_by_layer": layer_correlations,
        "case_correlations_with_physical_kv_splice_js": case_correlations,
        "within_task_pair_direction": pair_agreement,
        "physical_component_js": component,
        "gate_results": gate_results,
        "case_rows": case_rows,
        "scope": (
            "fixed-query, pre-output-projection attention bound plus 3B physical "
            "K-only/V-only/KV suffix replay on unchanged real V40 prompts; not "
            "functional accuracy, online selection, native-30B attention, or latency"
        ),
    }
    _write_json(output / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    measure_parser.add_argument("--max-cases", type=int, default=0)
    measure_parser.add_argument("--canary-only", action="store_true")
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.output)
    elif args.command == "measure":
        value = measure(args.output, args.max_cases, args.canary_only)
    else:
        value = analyze(args.output)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
