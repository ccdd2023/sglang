#!/usr/bin/env python3
"""RcLLM-inspired attention-sparsity motivation on frozen coding requests.

This is a Dense-only diagnostic, not an online selector or an accuracy result.
It asks whether final-query attention is concentrated on a small token set and
on a small number of version-valid read-only observation chunks, whether the
dominant observation is merely the newest one, and whether the dominant block
persists across layers.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.multi_workflow.coding_reuse_policy import (
    is_successful_readonly_evidence,
    repository_paths,
)
from benchmark.multi_workflow.measure_sessiongraph_atlas import _layers, _rotate_half
from benchmark.multi_workflow.motivate_v50_coding_provenance import (
    MODEL,
    _render_rolling,
    _sha256,
    _token_ids_hash,
    _turn_groups,
)
from benchmark.multi_workflow.run_bridge_reuse_pilot import write_json


ROOT = Path("/home/gfy/CodeMAS_Project")
SOURCE_DESIGN = (
    ROOT
    / "kvflow-artifacts/impactkv_m52_path_dependency_20260805/matched20/DESIGN.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_attention_sparsity_20260806/frozen20"
)
PROBE_LAYERS = (0, 8, 17, 26, 35)
QUERY_TAIL_TOKENS = 32
CHUNK_TOKENS = 128
PROFILE_BINS = 64
TOKEN_FRACTIONS = (0.01, 0.05, 0.10, 0.20, 0.50)


def _mass_at_fraction(scores: torch.Tensor, fraction: float) -> float:
    count = max(1, math.ceil(len(scores) * fraction))
    total = scores.sum().clamp_min(1e-20)
    return float(torch.topk(scores, count).values.sum().div(total).item())


def _fraction_for_mass(scores: torch.Tensor, target: float) -> float:
    ordered = torch.sort(scores, descending=True).values
    cumulative = ordered.cumsum(0)
    threshold = target * ordered.sum().clamp_min(1e-20)
    count = int(torch.searchsorted(cumulative, threshold).item()) + 1
    return count / len(scores)


def _binned(scores: torch.Tensor, bins: int) -> list[float]:
    boundaries = np.linspace(0, len(scores), bins + 1, dtype=int)
    values = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        if right <= left:
            values.append(0.0)
        else:
            values.append(float(scores[left:right].sum().item()))
    total = sum(values) or 1.0
    return [value / total for value in values]


def _observation_chunks(
    tokenizer: Any, case: Mapping[str, Any]
) -> tuple[list[int], list[dict[str, Any]]]:
    trajectory = json.loads(Path(case["trajectory_path"]).read_text(encoding="utf-8"))
    messages = trajectory["messages"]
    base = messages[:2]
    groups = _turn_groups(messages[2:])
    target_completed = int(case["target_request_index"]) - 1
    target_ids, spans = _render_rolling(tokenizer, base, groups[:target_completed])
    if _token_ids_hash(target_ids) != case["target_prompt_hash"]:
        raise ValueError(f"{case['case_id']}: reconstructed target prompt changed")
    latest_paths = set(case["latest_paths"])
    chunks: list[dict[str, Any]] = []
    dropped = max(0, target_completed - 6)
    for group_index in range(dropped, target_completed):
        group = groups[group_index]
        if not is_successful_readonly_evidence(group):
            continue
        paths = repository_paths(group)
        for message_index, message in enumerate(group):
            key = (group_index, message_index)
            if message.get("role") != "tool" or key not in spans:
                continue
            left, right = spans[key]
            if right - left < CHUNK_TOKENS:
                continue
            chunks.append(
                {
                    "candidate_id": f"group-{group_index}-message-{message_index}",
                    "group_index": group_index,
                    "message_index": message_index,
                    "start": right - CHUNK_TOKENS,
                    "length": CHUNK_TOKENS,
                    "full_message_tokens": right - left,
                    "paths": sorted(paths),
                    "path_relevant": bool(paths & latest_paths),
                }
            )
    return target_ids, chunks


def prepare(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    source = json.loads(SOURCE_DESIGN.read_text(encoding="utf-8"))
    cases = []
    for source_case in source["cases"]:
        target_ids, chunks = _observation_chunks(tokenizer, source_case)
        if len(chunks) < 2:
            continue
        cases.append(
            {
                "case_id": source_case["case_id"],
                "instance_id": source_case["instance_id"],
                "target_input_ids": target_ids,
                "target_prompt_hash": source_case["target_prompt_hash"],
                "chunks": chunks,
            }
        )
    output.mkdir(parents=True)
    design_path = output / "DESIGN.json"
    write_json(design_path, {"cases": cases, "model": str(MODEL)})
    candidate_counts = [len(case["chunks"]) for case in cases]
    capacity = {
        "cases": len(cases),
        "tasks": len({case["instance_id"] for case in cases}),
        "median_observation_chunks": statistics.median(candidate_counts)
        if candidate_counts
        else 0,
        "cases_with_at_least_three_chunks": sum(value >= 3 for value in candidate_counts),
    }
    registration = {
        "status": "REGISTERED_BEFORE_GPU",
        "purpose": (
            "RcLLM-inspired test of token and observation-block attention "
            "sparsity on frozen coding-agent target prompts"
        ),
        "source_design": str(SOURCE_DESIGN),
        "source_design_sha256": _sha256(SOURCE_DESIGN),
        "design_sha256": _sha256(design_path),
        "capacity": capacity,
        "model": str(MODEL),
        "probe_layers_zero_based": list(PROBE_LAYERS),
        "query_tail_tokens": QUERY_TAIL_TOKENS,
        "observation_chunk_tokens": CHUNK_TOKENS,
        "frozen_support_rules": {
            "minimum_cases": 16,
            "minimum_tasks": 8,
            "median_global_top10_attention_mass_min": 0.50,
            "median_top_observation_vs_uniform_ratio_min": 1.50,
            "median_within_observation_top20_mass_min": 0.50,
            "median_cross_layer_top_observation_stability_min": 0.60,
            "newest_observation_top_rate_max_for_recency_insufficiency": 0.70,
        },
        "interpretation_limits": [
            "Dense target attention is an offline oracle diagnostic",
            "tail-query attention is not a full pairwise attention heatmap",
            "attention sparsity motivates selection or repair but is not task accuracy",
            "no reuse, latency, or causal splice label is read in this experiment",
        ],
    }
    write_json(output / "REGISTRATION.json", registration)
    write_json(output / "CAPACITY.json", capacity)
    return registration


@torch.inference_mode()
def _profile_case(model: Any, case: Mapping[str, Any]) -> dict[str, Any]:
    target_ids = case["target_input_ids"]
    prompt_tokens = len(target_ids)
    query_tokens = min(QUERY_TAIL_TOKENS, prompt_tokens)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    layers = model.model.layers
    for layer_index in PROBE_LAYERS:
        def capture(
            _module: Any,
            args: tuple[Any, ...],
            index: int = layer_index,
        ) -> None:
            captured[index] = args[0][:, -query_tokens:].detach()

        handles.append(layers[layer_index].register_forward_pre_hook(capture))
    inputs = torch.tensor([target_ids], device="cuda", dtype=torch.long)
    try:
        output = model(input_ids=inputs, use_cache=True, return_dict=True, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    cache_layers = _layers(output.past_key_values)
    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    groups = num_heads // num_kv_heads
    head_dim = int(model.config.hidden_size) // num_heads
    query_positions = torch.arange(
        prompt_tokens - query_tokens, prompt_tokens, device="cuda", dtype=torch.long
    ).unsqueeze(0)
    key_positions = torch.arange(prompt_tokens, device="cuda")
    layer_rows = []
    for layer_index in PROBE_LAYERS:
        attention = layers[layer_index].self_attn
        hidden = captured[layer_index]
        query = attention.q_proj(hidden).view(1, query_tokens, num_heads, head_dim).transpose(1, 2)
        cosine, sine = model.model.rotary_emb(hidden, query_positions)
        query = query * cosine.unsqueeze(1) + _rotate_half(query) * sine.unsqueeze(1)
        key = cache_layers[layer_index][0].repeat_interleave(groups, dim=1)
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(head_dim)
        causal_mask = key_positions.view(1, 1, 1, -1) > query_positions.view(1, 1, -1, 1)
        weights = torch.softmax(scores.masked_fill(causal_mask, -torch.inf), dim=-1)
        token_scores = weights.mean(dim=(0, 1, 2))
        chunk_rows = []
        for chunk in case["chunks"]:
            start = int(chunk["start"])
            end = start + int(chunk["length"])
            values = token_scores[start:end]
            mass = float(values.sum().item())
            chunk_rows.append(
                {
                    **chunk,
                    "attention_mass": mass,
                    "within_top20_mass": _mass_at_fraction(values, 0.20),
                }
            )
        observation_total = sum(row["attention_mass"] for row in chunk_rows)
        for row in chunk_rows:
            row["observation_attention_share"] = (
                row["attention_mass"] / observation_total if observation_total else 0.0
            )
        top = max(chunk_rows, key=lambda row: row["attention_mass"])
        newest = max(chunk_rows, key=lambda row: (row["group_index"], row["message_index"]))
        largest = max(
            chunk_rows,
            key=lambda row: (
                row["full_message_tokens"],
                row["group_index"],
                row["message_index"],
            ),
        )
        layer_rows.append(
            {
                "layer": layer_index,
                "global_attention_mass_by_token_fraction": {
                    str(fraction): _mass_at_fraction(token_scores, fraction)
                    for fraction in TOKEN_FRACTIONS
                },
                "token_fraction_for_attention_mass": {
                    str(target): _fraction_for_mass(token_scores, target)
                    for target in (0.50, 0.80, 0.90)
                },
                "attention_bins": _binned(token_scores, PROFILE_BINS),
                "observation_attention_mass": observation_total,
                "top_observation_id": top["candidate_id"],
                "top_observation_share": top["observation_attention_share"],
                "top_observation_vs_uniform_ratio": top["observation_attention_share"]
                * len(chunk_rows),
                "top_observation_is_newest": top["candidate_id"] == newest["candidate_id"],
                "top_observation_is_largest": top["candidate_id"] == largest["candidate_id"],
                "top_observation_is_path_relevant": bool(top["path_relevant"]),
                "chunks": chunk_rows,
            }
        )
        del hidden, query, key, scores, weights, token_scores
    top_ids = [row["top_observation_id"] for row in layer_rows]
    modal_count = Counter(top_ids).most_common(1)[0][1]
    result = {
        "status": "ok",
        "case_id": case["case_id"],
        "instance_id": case["instance_id"],
        "prompt_tokens": prompt_tokens,
        "observation_chunks": len(case["chunks"]),
        "cross_layer_top_observation_stability": modal_count / len(layer_rows),
        "first_last_top_observation_same": top_ids[0] == top_ids[-1],
        "layers": layer_rows,
    }
    del output, inputs, captured, cache_layers
    gc.collect()
    torch.cuda.empty_cache()
    return result


def measure(output: Path, max_cases: int) -> dict[str, Any]:
    design_path = output / "DESIGN.json"
    registration = json.loads((output / "REGISTRATION.json").read_text())
    if registration["design_sha256"] != _sha256(design_path):
        raise ValueError("design changed after registration")
    cases = json.loads(design_path.read_text())["cases"]
    if max_cases > 0:
        cases = cases[:max_cases]
    observations_path = output / "OBSERVATIONS.jsonl"
    completed = set()
    if observations_path.exists():
        for line in observations_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "ok":
                    completed.add(row["case_id"])
    pending = [case for case in cases if case["case_id"] not in completed]
    if pending:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU substitution is forbidden")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).to("cuda").eval()
        for index, case in enumerate(pending, 1):
            row = _profile_case(model, case)
            with observations_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                json.dumps(
                    {
                        "case": index,
                        "pending": len(pending),
                        "case_id": row["case_id"],
                        "chunks": row["observation_chunks"],
                        "stability": row["cross_layer_top_observation_stability"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    status = {
        "status": "COMPLETE",
        "selected_cases": len(cases),
        "previously_completed_cases": len(completed),
        "new_cases": len(pending),
    }
    write_json(output / "MEASUREMENT_STATUS.json", status)
    return status


def analyze(output: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (output / "OBSERVATIONS.jsonl").read_text().splitlines()
        if line.strip()
    ]
    registration = json.loads((output / "REGISTRATION.json").read_text())
    rules = registration["frozen_support_rules"]
    case_summaries = []
    for row in rows:
        layers = row["layers"]
        within = [
            chunk["within_top20_mass"]
            for layer in layers
            for chunk in layer["chunks"]
        ]
        case_summaries.append(
            {
                "case_id": row["case_id"],
                "instance_id": row["instance_id"],
                "observation_chunks": row["observation_chunks"],
                "global_top10_mass": statistics.median(
                    layer["global_attention_mass_by_token_fraction"]["0.1"]
                    for layer in layers
                ),
                "top_observation_vs_uniform_ratio": statistics.median(
                    layer["top_observation_vs_uniform_ratio"] for layer in layers
                ),
                "within_observation_top20_mass": statistics.median(within),
                "cross_layer_top_observation_stability": row[
                    "cross_layer_top_observation_stability"
                ],
                "newest_top_rate": statistics.fmean(
                    float(layer["top_observation_is_newest"]) for layer in layers
                ),
                "largest_top_rate": statistics.fmean(
                    float(layer["top_observation_is_largest"]) for layer in layers
                ),
                "path_relevant_top_rate": statistics.fmean(
                    float(layer["top_observation_is_path_relevant"]) for layer in layers
                ),
            }
        )
    aggregate = {
        "cases": len(rows),
        "tasks": len({row["instance_id"] for row in rows}),
        "median_observation_chunks": statistics.median(
            row["observation_chunks"] for row in rows
        ),
        "median_global_top10_attention_mass": statistics.median(
            row["global_top10_mass"] for row in case_summaries
        ),
        "median_top_observation_vs_uniform_ratio": statistics.median(
            row["top_observation_vs_uniform_ratio"] for row in case_summaries
        ),
        "median_within_observation_top20_mass": statistics.median(
            row["within_observation_top20_mass"] for row in case_summaries
        ),
        "median_cross_layer_top_observation_stability": statistics.median(
            row["cross_layer_top_observation_stability"] for row in case_summaries
        ),
        "newest_observation_top_rate": statistics.fmean(
            row["newest_top_rate"] for row in case_summaries
        ),
        "largest_observation_top_rate": statistics.fmean(
            row["largest_top_rate"] for row in case_summaries
        ),
        "path_relevant_observation_top_rate": statistics.fmean(
            row["path_relevant_top_rate"] for row in case_summaries
        ),
    }
    gates = {
        "minimum_cases": aggregate["cases"] >= rules["minimum_cases"],
        "minimum_tasks": aggregate["tasks"] >= rules["minimum_tasks"],
        "global_token_sparsity": aggregate["median_global_top10_attention_mass"]
        >= rules["median_global_top10_attention_mass_min"],
        "observation_block_sparsity": aggregate[
            "median_top_observation_vs_uniform_ratio"
        ]
        >= rules["median_top_observation_vs_uniform_ratio_min"],
        "within_observation_heavy_hitters": aggregate[
            "median_within_observation_top20_mass"
        ]
        >= rules["median_within_observation_top20_mass_min"],
        "cross_layer_block_stability": aggregate[
            "median_cross_layer_top_observation_stability"
        ]
        >= rules["median_cross_layer_top_observation_stability_min"],
        "recency_is_insufficient": aggregate["newest_observation_top_rate"]
        <= rules["newest_observation_top_rate_max_for_recency_insufficiency"],
    }
    result = {
        "status": "COMPLETE",
        "decision": "SUPPORTED" if all(gates.values()) else "PARTIALLY_SUPPORTED",
        "aggregate": aggregate,
        "gates": gates,
        "case_summaries": case_summaries,
        "representative_profile": rows[0],
        "scope": (
            "Dense-only tail-query attention motivation; supports observation "
            "selection/selective-repair hypotheses, not task accuracy or latency"
        ),
    }
    write_json(output / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    measure_parser.add_argument("--max-cases", type=int, default=0)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.output)
    elif args.command == "measure":
        value = measure(args.output, args.max_cases)
    else:
        value = analyze(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
