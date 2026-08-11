#!/usr/bin/env python3
"""Task-disjoint, module-conditioned Attention/KV motivation experiment.

The experiment has sealed stages:

* ``prepare`` freezes version-valid 128-token observation candidates without
  looking at model-internal labels;
* ``measure-internals`` opens only Dense attention and K/V-deviation labels;
* ``freeze-cells`` assigns module-specific 2x2 cells and freezes the physical
  intervention subset;
* ``measure-splices`` opens local output and K-only/V-only/K+V final-logit
  outcomes only after the cell registration passes its capacity gates;
* ``summarize`` performs task-clustered confirmatory analysis.

This is an offline mechanism study.  Dense oracle attention and target K/V are
never presented as an online policy or as functional coding accuracy.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.multi_workflow.analyze_attention_kv_factorial import assign_cell
from benchmark.multi_workflow.coding_reuse_policy import (
    is_successful_readonly_evidence,
    repository_paths,
)
from benchmark.multi_workflow import motivate_v50_coding_provenance as m50
from benchmark.multi_workflow import motivate_v55_two_stage_selector as m55
from benchmark.multi_workflow.motivate_v48_attention_kv_risk import _compose_splice
from benchmark.multi_workflow.motivate_attention_kv_perturbation_bound import (
    _head_query_bound_metrics,
    _js,
    _model_theta,
    _physical_splice_logits,
    _rope_shift,
)


ROOT = Path("/home/gfy/CodeMAS_Project")
TRAJECTORY_ROOT = (
    ROOT
    / "kvflow-artifacts/impactkv_attention_kv_task_disjoint_20260807_r1/"
    "dense/full_18"
)
COHORT_REGISTRATION = (
    ROOT
    / "kvflow-artifacts/impactkv_attention_kv_task_disjoint_20260807_r1/"
    "COHORT_REGISTRATION.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_module_conditioned_attention_kv_20260807/"
    "task_disjoint20"
)
MODEL = m50.MODEL
PROBE_LAYERS = (0, 8, 17, 26, 35)
CANDIDATE_TOKENS = 128
MAX_PROMPT_TOKENS = 30_000
MAX_REQUESTS = 80
MAX_REQUESTS_PER_TASK = 4
MAX_CANDIDATES_PER_REQUEST = 4
QUERY_CHUNK = 64
FORWARD_CHUNK = 512
PREPARE_SEED = 2026080702
CELL_SEED = 2026080703
MAX_PHYSICAL_CANDIDATES = 96
CELL_TARGET = 12
CELL_TASK_MIN = 6
MIN_TASKS = 12
MIN_REQUESTS = 48
MIN_CANDIDATES = 128
MIN_MODULE_TASKS = 8
MIN_MODULE_POINTS = 48
PRIMARY_MODULES = (
    "assistant_action",
    "read_observation_path_relevant",
    "read_observation_path_disjoint",
    "other_tool_result",
    "generation_marker",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o644)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _token_hash(ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _trajectory_paths(root: Path) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in sorted(root.glob("**/*.traj.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        instance_id = str(value.get("instance_id") or "")
        if instance_id and instance_id not in selected:
            selected[instance_id] = path
    return selected


def _category_for_message(
    message: Mapping[str, Any], group: Sequence[Mapping[str, Any]]
) -> tuple[str, list[str]]:
    role = str(message.get("role") or "")
    if role == "assistant":
        return "assistant_action", []
    if role == "tool":
        paths = sorted(repository_paths(group))
        return (
            "read_observation" if is_successful_readonly_evidence(group) else "other_tool_result",
            paths,
        )
    return "other_tool_result", []


def _render_with_blocks(
    tokenizer: Any,
    base: Sequence[dict[str, Any]],
    groups: Sequence[Sequence[dict[str, Any]]],
) -> tuple[list[int], dict[tuple[int, int], tuple[int, int]], list[dict[str, Any]]]:
    ids: list[int] = []
    message_spans: dict[tuple[int, int], tuple[int, int]] = {}
    blocks: list[dict[str, Any]] = []

    def append(literal: str, category: str, paths: Sequence[str], label: str) -> tuple[int, int]:
        start = len(ids)
        ids.extend(tokenizer.encode(literal, add_special_tokens=False))
        end = len(ids)
        if end > start:
            blocks.append(
                {
                    "start": start,
                    "end": end,
                    "tokens": end - start,
                    "category": category,
                    "paths": list(paths),
                    "label": label,
                }
            )
        return start, end

    for index, message in enumerate(base):
        append(
            m50._render_message_literal(message),
            "system_instruction" if index == 0 else "user_task",
            [],
            "system" if index == 0 else "coding task",
        )
    dropped = max(0, len(groups) - m50.ROLLING_GROUPS)
    if dropped:
        append(
            m50._render_message_literal(
                {
                    "role": "user",
                    "content": m50.ROLLING_NOTICE.format(dropped=dropped),
                }
            ),
            "compaction_notice",
            [],
            "history compaction",
        )
    for group_index in range(dropped, len(groups)):
        group = groups[group_index]
        for message_index, message in enumerate(group):
            category, paths = _category_for_message(message, group)
            span = append(
                m50._render_message_literal(message),
                category,
                paths,
                f"group {group_index} message {message_index}",
            )
            message_spans[(group_index, message_index)] = span
    append("<|im_start|>assistant\n", "generation_marker", [], "next action")
    for index, block in enumerate(blocks):
        block["block_id"] = f"b{index:02d}"
    if not blocks or blocks[0]["start"] != 0 or blocks[-1]["end"] != len(ids):
        raise ValueError("prompt blocks do not cover the rendered prompt")
    if any(left["end"] != right["start"] for left, right in zip(blocks, blocks[1:])):
        raise ValueError("prompt blocks are not contiguous")
    return ids, message_spans, blocks


def _directory_overlap(left: set[str], right: set[str]) -> bool:
    left_dirs = {str(PurePosixPath(value).parent) for value in left}
    right_dirs = {str(PurePosixPath(value).parent) for value in right}
    return bool(left_dirs & right_dirs)


def _request_candidates(
    *,
    tokenizer: Any,
    trajectory_path: Path,
) -> list[dict[str, Any]]:
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    instance_id = str(trajectory["instance_id"])
    messages = trajectory["messages"]
    base = messages[:2]
    groups = m50._turn_groups(messages[2:])
    rows: list[dict[str, Any]] = []
    for target_completed in range(7, len(groups)):
        source_ids, source_spans, source_blocks = _render_with_blocks(
            tokenizer, base, groups[: target_completed - 1]
        )
        target_ids, target_spans, target_blocks = _render_with_blocks(
            tokenizer, base, groups[:target_completed]
        )
        if len(source_ids) > MAX_PROMPT_TOKENS or len(target_ids) > MAX_PROMPT_TOKENS:
            continue
        latest_paths = set(repository_paths(groups[target_completed - 1]))
        candidates: list[dict[str, Any]] = []
        group_indices = sorted({key[0] for key in source_spans} & {key[0] for key in target_spans})
        for group_index in group_indices:
            group = groups[group_index]
            paths = set(repository_paths(group))
            if not paths or not is_successful_readonly_evidence(group):
                continue
            for message_index, message in enumerate(group):
                if message.get("role") != "tool":
                    continue
                candidate = m50._message_candidate(
                    category="version_valid_read_observation",
                    group_index=group_index,
                    message_index=message_index,
                    source_ids=source_ids,
                    source_spans=source_spans,
                    target_ids=target_ids,
                    target_spans=target_spans,
                )
                if candidate is None:
                    continue
                candidate["repository_paths"] = sorted(paths)
                if not m55._version_valid_at_target(candidate, groups, target_completed - 1):
                    continue
                exact_path = bool(paths & latest_paths)
                same_directory = not exact_path and _directory_overlap(paths, latest_paths)
                candidate.update(
                    candidate_id=(
                        f"g{group_index}-m{message_index}-t{candidate['target_start']}"
                    ),
                    exact_path=exact_path,
                    same_directory=same_directory,
                    interaction_distance=target_completed - 1 - group_index,
                    version_valid_at_target=True,
                )
                candidates.append(candidate)
        if not candidates:
            continue
        candidates = sorted(
            candidates,
            key=lambda row: (
                int(row["group_index"]),
                int(row["target_start"]),
                str(row["candidate_id"]),
            ),
            reverse=True,
        )[:MAX_CANDIDATES_PER_REQUEST]
        case_id = f"{instance_id}-q{target_completed + 1}"
        rows.append(
            {
                "case_id": case_id,
                "instance_id": instance_id,
                "request_index": target_completed + 1,
                "source_input_ids": source_ids,
                "target_input_ids": target_ids,
                "source_prompt_hash": _token_hash(source_ids),
                "target_prompt_hash": _token_hash(target_ids),
                "source_blocks": source_blocks,
                "target_blocks": target_blocks,
                "latest_paths": sorted(latest_paths),
                "candidates": candidates,
                "trajectory_path": str(trajectory_path),
                "trajectory_sha256": _sha256(trajectory_path),
            }
        )
    return rows


def _balanced_requests(rows: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["instance_id"])].append(row)
    queues: dict[str, list[dict[str, Any]]] = {}
    for task, task_rows in by_task.items():
        ordered = sorted(task_rows, key=lambda row: (int(row["request_index"]), row["case_id"]))
        if len(ordered) > MAX_REQUESTS_PER_TASK:
            indices = np.linspace(0, len(ordered) - 1, MAX_REQUESTS_PER_TASK).round().astype(int)
            ordered = [ordered[int(index)] for index in sorted(set(indices))]
        queues[task] = ordered
    selected: list[dict[str, Any]] = []
    tasks = sorted(queues, key=lambda task: _stable_hash(task, PREPARE_SEED))
    depth = 0
    while len(selected) < limit:
        added = False
        for task in tasks:
            if depth < len(queues[task]):
                selected.append(queues[task][depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def prepare(output: Path, trajectory_root: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if not COHORT_REGISTRATION.exists():
        raise FileNotFoundError(COHORT_REGISTRATION)
    trajectories = _trajectory_paths(trajectory_root)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    all_rows = [
        row
        for task in sorted(trajectories)
        for row in _request_candidates(tokenizer=tokenizer, trajectory_path=trajectories[task])
    ]
    selected = _balanced_requests(all_rows, MAX_REQUESTS)
    tasks = len({str(row["instance_id"]) for row in selected})
    candidates = sum(len(row["candidates"]) for row in selected)
    capacity = {
        "trajectory_tasks": len(trajectories),
        "eligible_tasks": len({str(row["instance_id"]) for row in all_rows}),
        "eligible_requests": len(all_rows),
        "eligible_candidates_after_per_request_cap": sum(len(row["candidates"]) for row in all_rows),
        "selected_tasks": tasks,
        "selected_requests": len(selected),
        "selected_candidates": candidates,
        "minimum_tasks": MIN_TASKS,
        "minimum_requests": MIN_REQUESTS,
        "minimum_candidates": MIN_CANDIDATES,
    }
    gates = {
        "tasks": tasks >= MIN_TASKS,
        "requests": len(selected) >= MIN_REQUESTS,
        "candidates": candidates >= MIN_CANDIDATES,
        "source_target_candidate_tokens_identical": all(
            row["source_input_ids"][int(candidate["source_start"]): int(candidate["source_start"]) + CANDIDATE_TOKENS]
            == row["target_input_ids"][int(candidate["target_start"]): int(candidate["target_start"]) + CANDIDATE_TOKENS]
            for row in selected
            for candidate in row["candidates"]
        ),
        "all_version_valid": all(
            bool(candidate["version_valid_at_target"])
            for row in selected
            for candidate in row["candidates"]
        ),
    }
    output.mkdir(parents=True)
    design_path = output / "DESIGN.json"
    _write_json(
        design_path,
        {
            "cases": selected,
            "analysis_model": str(MODEL),
            "probe_layers_zero_based": list(PROBE_LAYERS),
            "candidate_tokens": CANDIDATE_TOKENS,
            "trajectory_root": str(trajectory_root),
        },
    )
    registration = {
        "status": "REGISTERED_BEFORE_INTERNAL_LABELS" if all(gates.values()) else "STOPPED_BEFORE_INTERNAL_LABELS",
        "purpose": "task-disjoint module-conditioned Attention/KV factorial confirmation",
        "design_sha256": _sha256(design_path),
        "script_sha256": _sha256(Path(__file__)),
        "cohort_registration": str(COHORT_REGISTRATION),
        "cohort_registration_sha256": _sha256(COHORT_REGISTRATION),
        "capacity": capacity,
        "capacity_gates": gates,
        "selection_contract": {
            "maximum_requests": MAX_REQUESTS,
            "maximum_requests_per_task": MAX_REQUESTS_PER_TASK,
            "maximum_candidates_per_request": MAX_CANDIDATES_PER_REQUEST,
            "candidate_tokens": CANDIDATE_TOKENS,
            "successful_read_only_observation": True,
            "version_valid_at_target": True,
            "outcome_used_for_selection": False,
        },
        "sealed_stages": [
            "Dense attention and K/V deviation",
            "cell registration",
            "local output and physical splice outcomes",
        ],
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
    }
    _write_json(output / "CAPACITY.json", capacity)
    _write_json(output / "REGISTRATION.json", registration)
    return registration


def module_for_block(block: Mapping[str, Any], candidate_paths: set[str]) -> str:
    category = str(block["category"])
    if category != "read_observation":
        return category
    return (
        "read_observation_path_relevant"
        if set(str(value) for value in block.get("paths", [])) & candidate_paths
        else "read_observation_path_disjoint"
    )


def _cache_layers(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return [(row[0], row[1]) for row in cache]


def _cpu_cache(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            key[0].detach().to("cpu", dtype=torch.bfloat16).contiguous(),
            value[0].detach().to("cpu", dtype=torch.bfloat16).contiguous(),
        )
        for key, value in _cache_layers(cache)
    ]


def _register_full_hooks(model: Any) -> tuple[dict[int, torch.Tensor], list[Any]]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in PROBE_LAYERS:
        def capture(_module: Any, args: tuple[Any, ...], index: int = layer_index) -> None:
            captured[index] = args[0].detach()

        handles.append(model.model.layers[layer_index].register_forward_pre_hook(capture))
    return captured, handles


@torch.inference_mode()
def _dense_full(
    model: Any, ids: Sequence[int], capture_hidden: bool
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, dict[int, torch.Tensor]]:
    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []
    if capture_hidden:
        captured, handles = _register_full_hooks(model)
    inputs = torch.tensor([ids], device="cuda", dtype=torch.long)
    try:
        output = model(input_ids=inputs, use_cache=True, return_dict=True, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    if capture_hidden and set(captured) != set(PROBE_LAYERS):
        raise RuntimeError("not all full-prompt hidden states were captured")
    cache = _cpu_cache(output.past_key_values)
    logits = output.logits[0, -1].detach().float().cpu()
    del output, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return cache, logits, captured


def _cosine_drift(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        (
            1.0
            - (
                F.normalize(left.float(), dim=-1)
                * F.normalize(right.float(), dim=-1)
            ).sum(dim=-1).mean(dim=-1)
        ).mean()
    )


@torch.inference_mode()
def _attention_mass_for_block(
    *,
    model: Any,
    layer_index: int,
    hidden: torch.Tensor,
    target_key: torch.Tensor,
    block: Mapping[str, Any],
    candidate_spans: Mapping[str, tuple[int, int]],
) -> dict[str, float]:
    attention = model.model.layers[layer_index].self_attn
    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    groups = num_heads // num_kv_heads
    head_dim = int(getattr(model.config, "head_dim", 0)) or int(model.config.hidden_size) // num_heads
    sums = {candidate_id: 0.0 for candidate_id in candidate_spans}
    left, right = int(block["start"]), int(block["end"])
    for query_left in range(left, right, QUERY_CHUNK):
        query_right = min(right, query_left + QUERY_CHUNK)
        query_hidden = hidden[:, query_left:query_right]
        positions = torch.arange(query_left, query_right, device="cuda", dtype=torch.long).unsqueeze(0)
        query = attention.q_proj(query_hidden).view(1, query_right - query_left, num_heads, head_dim).transpose(1, 2)
        if hasattr(attention, "q_norm"):
            query = attention.q_norm(query)
        cosine, sine = model.model.rotary_emb(query_hidden, positions)
        half = query.shape[-1] // 2
        rotated = torch.cat((-query[..., half:], query[..., :half]), dim=-1)
        query = query * cosine.unsqueeze(1) + rotated * sine.unsqueeze(1)
        expanded_key = target_key.repeat_interleave(groups, dim=1)
        scores = torch.matmul(query.float(), expanded_key.float().transpose(-1, -2)) / math.sqrt(head_dim)
        key_positions = torch.arange(target_key.shape[-2], device="cuda")
        causal = key_positions.view(1, 1, 1, -1) > positions.view(1, 1, -1, 1)
        weights = torch.softmax(scores.masked_fill(causal, -torch.inf), dim=-1).mean(dim=1)[0]
        for candidate_id, (start, end) in candidate_spans.items():
            sums[candidate_id] += float(weights[:, start:end].sum())
        del query_hidden, positions, query, rotated, expanded_key, scores, causal, weights
    tokens = right - left
    return {candidate_id: value / tokens for candidate_id, value in sums.items()}


def _internals_complete(row: Mapping[str, Any]) -> bool:
    if row.get("status") != "ok" or not row.get("candidates"):
        return False
    return all(candidate.get("layers") and candidate.get("module_attention") for candidate in row["candidates"])


@torch.inference_mode()
def measure_internals(output: Path, max_cases: int) -> dict[str, Any]:
    registration = json.loads((output / "REGISTRATION.json").read_text())
    design_path = output / "DESIGN.json"
    if registration["status"] != "REGISTERED_BEFORE_INTERNAL_LABELS":
        raise RuntimeError("offline capacity gate did not pass")
    if registration["design_sha256"] != _sha256(design_path):
        raise ValueError("design changed after registration")
    cases = json.loads(design_path.read_text())["cases"]
    if max_cases > 0:
        cases = cases[:max_cases]
    observations = output / "INTERNALS.jsonl"
    completed = set()
    if observations.exists():
        for line in observations.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if _internals_complete(row):
                    completed.add(str(row["case_id"]))
    pending = [case for case in cases if str(case["case_id"]) not in completed]
    if not pending:
        return {"status": "COMPLETE", "selected_cases": len(cases), "new_cases": 0}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    theta = _model_theta(model.config)
    errors = []
    written = 0
    for index, case in enumerate(pending, 1):
        try:
            source_cache, _, _ = _dense_full(model, case["source_input_ids"], False)
            target_cache, _, hidden = _dense_full(model, case["target_input_ids"], True)
            spans = {
                str(candidate["candidate_id"]): (
                    int(candidate["target_start"]),
                    int(candidate["target_start"]) + CANDIDATE_TOKENS,
                )
                for candidate in case["candidates"]
            }
            block_masses: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
            for layer_index in PROBE_LAYERS:
                target_key = target_cache[layer_index][0].unsqueeze(0).to("cuda")
                for block in case["target_blocks"]:
                    if int(block["end"]) <= min(start for start, _ in spans.values()):
                        continue
                    block_masses[str(layer_index)][str(block["block_id"])] = _attention_mass_for_block(
                        model=model,
                        layer_index=layer_index,
                        hidden=hidden[layer_index],
                        target_key=target_key,
                        block=block,
                        candidate_spans=spans,
                    )
                del target_key
            measured = []
            for candidate in case["candidates"]:
                candidate_id = str(candidate["candidate_id"])
                source_start = int(candidate["source_start"])
                target_start = int(candidate["target_start"])
                layer_rows = []
                module_values: dict[str, list[tuple[float, int]]] = defaultdict(list)
                for layer_index in PROBE_LAYERS:
                    source_key, source_value = source_cache[layer_index]
                    target_key, target_value = target_cache[layer_index]
                    shifted = _rope_shift(
                        source_key[:, source_start : source_start + CANDIDATE_TOKENS].to("cuda"),
                        target_start - source_start,
                        theta,
                    ).cpu()
                    stale_value = source_value[:, source_start : source_start + CANDIDATE_TOKENS]
                    dense_key = target_key[:, target_start : target_start + CANDIDATE_TOKENS]
                    dense_value = target_value[:, target_start : target_start + CANDIDATE_TOKENS]
                    key_drift = _cosine_drift(shifted, dense_key)
                    value_drift = _cosine_drift(stale_value, dense_value)
                    layer_rows.append(
                        {
                            "layer": layer_index,
                            "key_cosine_drift": key_drift,
                            "value_cosine_drift": value_drift,
                            "raw_kv_drift": max(key_drift, value_drift),
                        }
                    )
                    for block in case["target_blocks"]:
                        if int(block["start"]) < target_start + CANDIDATE_TOKENS:
                            continue
                        module = module_for_block(block, set(candidate["repository_paths"]))
                        mass = block_masses[str(layer_index)][str(block["block_id"])][candidate_id]
                        module_values[module].append((mass, int(block["tokens"])))
                module_attention = {
                    module: {
                        "attention_mass": sum(value * tokens for value, tokens in values) / sum(tokens for _, tokens in values),
                        "query_tokens_layer_weighted": sum(tokens for _, tokens in values),
                    }
                    for module, values in module_values.items()
                    if values and sum(tokens for _, tokens in values) > 0
                }
                measured.append({**candidate, "layers": layer_rows, "module_attention": module_attention})
            row = {
                "status": "ok",
                "case_id": case["case_id"],
                "instance_id": case["instance_id"],
                "request_index": case["request_index"],
                "source_prompt_hash": case["source_prompt_hash"],
                "target_prompt_hash": case["target_prompt_hash"],
                "candidates": measured,
            }
            if not _internals_complete(row):
                raise RuntimeError("incomplete internal labels")
            with observations.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
            print(json.dumps({"case": index, "pending": len(pending), "case_id": case["case_id"]}), flush=True)
            del source_cache, target_cache, hidden, block_masses
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as error:
            errors.append({"case_id": case["case_id"], "error": f"{type(error).__name__}: {error}"})
            print(json.dumps(errors[-1]), flush=True)
            break
    del model
    gc.collect()
    torch.cuda.empty_cache()
    status = {
        "status": "COMPLETE" if not errors and written == len(pending) else "PARTIAL",
        "selected_cases": len(cases),
        "previously_completed": len(completed),
        "new_cases": written,
        "errors": errors,
        "dtype": "bfloat16",
        "model": str(MODEL),
    }
    _write_json(output / "INTERNAL_MEASUREMENT_STATUS.json", status)
    return status


def _internal_rows(output: Path) -> list[dict[str, Any]]:
    path = output / "INTERNALS.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(not _internals_complete(row) for row in rows):
        raise ValueError("internal observations are missing or incomplete")
    return rows


def _candidate_module_points(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points = []
    for row in rows:
        for candidate in row["candidates"]:
            raw_drift = statistics.fmean(float(layer["raw_kv_drift"]) for layer in candidate["layers"])
            for module, value in candidate["module_attention"].items():
                if module not in PRIMARY_MODULES:
                    continue
                points.append(
                    {
                        "case_id": str(row["case_id"]),
                        "instance_id": str(row["instance_id"]),
                        "candidate_id": str(candidate["candidate_id"]),
                        "point_id": f"{row['case_id']}::{candidate['candidate_id']}::{module}",
                        "module": module,
                        "attention_mass": float(value["attention_mass"]),
                        "raw_kv_drift": raw_drift,
                    }
                )
    return points


def _select_physical_candidates(
    points: Sequence[Mapping[str, Any]], qualifying: Sequence[str]
) -> list[str]:
    cells = (
        "low_attention__low_drift",
        "high_attention__low_drift",
        "low_attention__high_drift",
        "high_attention__high_drift",
    )
    point_deficits = {
        (module, cell): CELL_TARGET for module in qualifying for cell in cells
    }
    selected_tasks: dict[tuple[str, str], set[str]] = {
        (module, cell): set() for module in qualifying for cell in cells
    }
    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for point in points:
        if point["module"] in qualifying:
            by_candidate[f"{point['case_id']}::{point['candidate_id']}"].append(point)
    selected: list[str] = []
    while (
        any(value > 0 for value in point_deficits.values())
        or any(len(tasks) < CELL_TASK_MIN for tasks in selected_tasks.values())
    ) and len(selected) < MAX_PHYSICAL_CANDIDATES:
        best = None
        for candidate_key, candidate_points in by_candidate.items():
            if candidate_key in selected:
                continue
            point_gain = 0
            task_gain = 0
            for point in candidate_points:
                key = (str(point["module"]), str(point["cell"]))
                if point_deficits.get(key, 0) > 0:
                    point_gain += 1
                if (
                    len(selected_tasks.get(key, set())) < CELL_TASK_MIN
                    and str(point["instance_id"]) not in selected_tasks.get(key, set())
                ):
                    task_gain += 1
            # A new task is the scarcer resource: one task can supply many
            # candidate points, whereas six distinct tasks cannot be recovered
            # by adding more points from an already represented task.
            score = (2 * task_gain + point_gain, task_gain, point_gain, _stable_hash(candidate_key, CELL_SEED))
            if best is None or score[:3] > best[0][:3] or (
                score[:3] == best[0][:3] and score[3] < best[0][3]
            ):
                best = (score, candidate_key, candidate_points)
        if best is None or best[0][0] == 0:
            break
        _, candidate_key, candidate_points = best
        selected.append(candidate_key)
        for point in candidate_points:
            key = (str(point["module"]), str(point["cell"]))
            if point_deficits.get(key, 0) > 0:
                point_deficits[key] -= 1
            if key in selected_tasks:
                selected_tasks[key].add(str(point["instance_id"]))
    return selected


def freeze_cells(output: Path) -> dict[str, Any]:
    destination = output / "CELL_REGISTRATION.json"
    if destination.exists():
        return json.loads(destination.read_text())
    internals = output / "INTERNALS.jsonl"
    rows = _internal_rows(output)
    design = json.loads((output / "DESIGN.json").read_text())
    if len(rows) != len(design["cases"]):
        raise ValueError("all registered internal cases must finish before cell freeze")
    points = _candidate_module_points(rows)
    drift_median = statistics.median(float(point["raw_kv_drift"]) for point in points)
    attention_medians = {
        module: statistics.median(
            float(point["attention_mass"]) for point in points if point["module"] == module
        )
        for module in sorted({str(point["module"]) for point in points})
    }
    for point in points:
        point["cell"] = assign_cell(
            attention=float(point["attention_mass"]),
            drift=float(point["raw_kv_drift"]),
            attention_median=attention_medians[str(point["module"])],
            drift_median=drift_median,
        )
    module_capacity = {}
    qualifying = []
    for module in PRIMARY_MODULES:
        module_points = [point for point in points if point["module"] == module]
        cells = {}
        for cell in (
            "low_attention__low_drift",
            "high_attention__low_drift",
            "low_attention__high_drift",
            "high_attention__high_drift",
        ):
            cell_points = [point for point in module_points if point["cell"] == cell]
            cells[cell] = {
                "points": len(cell_points),
                "tasks": len({point["instance_id"] for point in cell_points}),
            }
        capable = (
            len(module_points) >= MIN_MODULE_POINTS
            and len({point["instance_id"] for point in module_points}) >= MIN_MODULE_TASKS
            and all(value["points"] >= CELL_TARGET and value["tasks"] >= CELL_TASK_MIN for value in cells.values())
        )
        module_capacity[module] = {
            "points": len(module_points),
            "tasks": len({point["instance_id"] for point in module_points}),
            "cells": cells,
            "qualifies": capable,
        }
        if capable:
            qualifying.append(module)
    has_generation = "generation_marker" in qualifying
    has_coding_module = any(
        module in qualifying
        for module in (
            "read_observation_path_relevant",
            "read_observation_path_disjoint",
            "other_tool_result",
        )
    )
    module_gate = len(qualifying) >= 3 and has_generation and has_coding_module
    selected = _select_physical_candidates(points, qualifying) if module_gate else []
    selected_set = set(selected)
    selected_capacity = {}
    for module in qualifying:
        selected_capacity[module] = {}
        for cell in (
            "low_attention__low_drift",
            "high_attention__low_drift",
            "low_attention__high_drift",
            "high_attention__high_drift",
        ):
            chosen = [
                point
                for point in points
                if point["module"] == module
                and point["cell"] == cell
                and f"{point['case_id']}::{point['candidate_id']}" in selected_set
            ]
            selected_capacity[module][cell] = {
                "points": len(chosen),
                "tasks": len({point["instance_id"] for point in chosen}),
            }
    intervention_gate = (
        module_gate
        and len(selected) <= MAX_PHYSICAL_CANDIDATES
        and all(
            value["points"] >= CELL_TARGET and value["tasks"] >= CELL_TASK_MIN
            for module in selected_capacity.values()
            for value in module.values()
        )
    )
    value = {
        "status": "REGISTERED_BEFORE_PHYSICAL_OUTCOMES" if intervention_gate else "STOPPED_BEFORE_PHYSICAL_OUTCOMES",
        "internals_sha256": _sha256(internals),
        "design_sha256": _sha256(output / "DESIGN.json"),
        "thresholds": {
            "raw_kv_drift_global_median": drift_median,
            "attention_median_by_module": attention_medians,
            "inclusive_median_assignment": True,
        },
        "qualifying_modules": qualifying,
        "module_capacity": module_capacity,
        "selected_candidate_keys": selected,
        "selected_candidate_count": len(selected),
        "selected_capacity": selected_capacity,
        "gates": {
            "at_least_three_modules": len(qualifying) >= 3,
            "generation_marker_included": has_generation,
            "coding_evidence_or_feedback_included": has_coding_module,
            "physical_subset_capacity": intervention_gate,
        },
        "cell_points": points,
        "outcomes_unopened_at_registration": True,
    }
    _write_json(destination, value)
    return value


def _query_tensor(
    *,
    model: Any,
    layer_index: int,
    hidden: torch.Tensor,
    query_left: int,
    query_right: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    attention = model.model.layers[layer_index].self_attn
    num_heads = int(model.config.num_attention_heads)
    head_dim = int(getattr(model.config, "head_dim", 0)) or int(model.config.hidden_size) // num_heads
    query_hidden = hidden[:, query_left:query_right]
    positions = torch.arange(query_left, query_right, device="cuda", dtype=torch.long).unsqueeze(0)
    query = attention.q_proj(query_hidden).view(
        1, query_right - query_left, num_heads, head_dim
    ).transpose(1, 2)
    if hasattr(attention, "q_norm"):
        query = attention.q_norm(query)
    cosine, sine = model.model.rotary_emb(query_hidden, positions)
    half = query.shape[-1] // 2
    query = query * cosine.unsqueeze(1) + torch.cat(
        (-query[..., half:], query[..., :half]), dim=-1
    ) * sine.unsqueeze(1)
    return query, positions


@torch.inference_mode()
def _local_module_metrics(
    *,
    model: Any,
    case: Mapping[str, Any],
    candidate: Mapping[str, Any],
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    hidden: Mapping[int, torch.Tensor],
    theta: float,
) -> dict[str, dict[str, float]]:
    source_start = int(candidate["source_start"])
    target_start = int(candidate["target_start"])
    target_end = target_start + CANDIDATE_TOKENS
    target_tokens = len(case["target_input_ids"])
    candidate_paths = set(str(value) for value in candidate["repository_paths"])
    metric_names = (
        "dense_attention_mass",
        "actual_key_output_relative",
        "actual_value_output_relative",
        "actual_kv_output_relative",
        "attention_l1",
        "first_order_score_relative",
        "mass_aware_analytic_bound_relative",
    )
    accumulators: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    points: Counter[str] = Counter()
    key_positions = torch.arange(target_tokens, device="cuda")
    for layer_index in PROBE_LAYERS:
        num_heads = int(model.config.num_attention_heads)
        num_kv_heads = int(model.config.num_key_value_heads)
        groups = num_heads // num_kv_heads
        head_dim = int(getattr(model.config, "head_dim", 0)) or int(model.config.hidden_size) // num_heads
        target_key, target_value = target_cache[layer_index]
        source_key, source_value = source_cache[layer_index]
        dense_key = target_key.to("cuda")
        dense_value = target_value.to("cuda")
        stale_key = dense_key.clone()
        stale_value = dense_value.clone()
        stale_key[:, target_start:target_end] = _rope_shift(
            source_key[:, source_start : source_start + CANDIDATE_TOKENS].to("cuda"),
            target_start - source_start,
            theta,
        )
        stale_value[:, target_start:target_end] = source_value[
            :, source_start : source_start + CANDIDATE_TOKENS
        ].to("cuda")
        dense_key_h = dense_key.repeat_interleave(groups, dim=0)
        stale_key_h = stale_key.repeat_interleave(groups, dim=0)
        dense_value_h = dense_value.repeat_interleave(groups, dim=0)
        stale_value_h = stale_value.repeat_interleave(groups, dim=0)
        for block in case["target_blocks"]:
            left, right = int(block["start"]), int(block["end"])
            if left < target_end:
                continue
            module = module_for_block(block, candidate_paths)
            if module not in PRIMARY_MODULES:
                continue
            for query_left in range(left, right, QUERY_CHUNK):
                query_right = min(right, query_left + QUERY_CHUNK)
                query, positions = _query_tensor(
                    model=model,
                    layer_index=layer_index,
                    hidden=hidden[layer_index],
                    query_left=query_left,
                    query_right=query_right,
                )
                dense_scores = torch.matmul(
                    query[0].float(), dense_key_h.float().transpose(-1, -2)
                ) / math.sqrt(head_dim)
                stale_scores = torch.matmul(
                    query[0].float(), stale_key_h.float().transpose(-1, -2)
                ) / math.sqrt(head_dim)
                causal = key_positions.view(1, 1, -1) > positions.view(1, -1, 1)
                dense_scores = dense_scores.masked_fill(causal, -torch.inf)
                stale_scores = stale_scores.masked_fill(causal, -torch.inf)
                metrics = _head_query_bound_metrics(
                    dense_scores=dense_scores,
                    stale_scores=stale_scores,
                    dense_values=dense_value_h,
                    stale_values=stale_value_h,
                    island_start=target_start,
                    island_end=target_end,
                )
                count = int(metrics["actual_kv_output_relative"].numel())
                for name in metric_names:
                    accumulators[module][name] += float(metrics[name].sum())
                points[module] += count
                del query, positions, dense_scores, stale_scores, causal, metrics
        del (
            dense_key,
            dense_value,
            stale_key,
            stale_value,
            dense_key_h,
            stale_key_h,
            dense_value_h,
            stale_value_h,
        )
        torch.cuda.empty_cache()
    result = {}
    for module, values in accumulators.items():
        count = points[module]
        result[module] = {
            **{f"{name}_mean": value / count for name, value in values.items()},
            "attention_row_tv_mean": values["attention_l1"] / count / 2.0,
            "head_query_layer_points": count,
        }
    return result


def _splice_complete(row: Mapping[str, Any]) -> bool:
    if row.get("status") != "ok" or not row.get("candidates"):
        return False
    for candidate in row["candidates"]:
        if not candidate.get("local_modules"):
            return False
        physical = candidate.get("physical_splice") or {}
        if set(physical) != {"key_only", "value_only", "kv"}:
            return False
        if not all(math.isfinite(float(value["final_logit_js"])) for value in physical.values()):
            return False
    return True


@torch.inference_mode()
def measure_splices(output: Path, max_cases: int) -> dict[str, Any]:
    cell_path = output / "CELL_REGISTRATION.json"
    if not cell_path.exists():
        raise FileNotFoundError(cell_path)
    registration = json.loads(cell_path.read_text())
    if registration["status"] != "REGISTERED_BEFORE_PHYSICAL_OUTCOMES":
        raise RuntimeError("cell/intervention capacity gate did not pass")
    if registration["internals_sha256"] != _sha256(output / "INTERNALS.jsonl"):
        raise ValueError("internal labels changed after cell registration")
    design = json.loads((output / "DESIGN.json").read_text())
    cases_by_id = {str(case["case_id"]): case for case in design["cases"]}
    selected_by_case: dict[str, set[str]] = defaultdict(set)
    for key in registration["selected_candidate_keys"]:
        case_id, candidate_id = str(key).split("::", 1)
        selected_by_case[case_id].add(candidate_id)
    selected_cases = [cases_by_id[case_id] for case_id in selected_by_case]
    if max_cases > 0:
        selected_cases = selected_cases[:max_cases]
    observations = output / "SPLICE_OBSERVATIONS.jsonl"
    completed = set()
    if observations.exists():
        for line in observations.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if _splice_complete(row):
                    completed.add(str(row["case_id"]))
    pending = [case for case in selected_cases if str(case["case_id"]) not in completed]
    if not pending:
        return {"status": "COMPLETE", "selected_cases": len(selected_cases), "new_cases": 0}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    theta = _model_theta(model.config)
    errors = []
    written = 0
    for index, case in enumerate(pending, 1):
        try:
            source_cache, _, _ = _dense_full(model, case["source_input_ids"], False)
            target_cache, dense_logits, hidden = _dense_full(model, case["target_input_ids"], True)
            measured = []
            for candidate in case["candidates"]:
                candidate_id = str(candidate["candidate_id"])
                if candidate_id not in selected_by_case[str(case["case_id"])]:
                    continue
                local = _local_module_metrics(
                    model=model,
                    case=case,
                    candidate=candidate,
                    source_cache=source_cache,
                    target_cache=target_cache,
                    hidden=hidden,
                    theta=theta,
                )
                physical = {}
                physical_case = {
                    **case,
                    "source_start": int(candidate["source_start"]),
                    "target_start": int(candidate["target_start"]),
                    "length": CANDIDATE_TOKENS,
                }
                for mode in ("key_only", "value_only", "kv"):
                    logits = _physical_splice_logits(
                        model=model,
                        case=physical_case,
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
                measured.append(
                    {
                        "candidate_id": candidate_id,
                        "source_start": candidate["source_start"],
                        "target_start": candidate["target_start"],
                        "repository_paths": candidate["repository_paths"],
                        "local_modules": local,
                        "physical_splice": physical,
                    }
                )
            row = {
                "status": "ok",
                "case_id": case["case_id"],
                "instance_id": case["instance_id"],
                "candidates": measured,
            }
            if not _splice_complete(row):
                raise RuntimeError("incomplete physical splice labels")
            with observations.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
            print(json.dumps({"case": index, "pending": len(pending), "case_id": case["case_id"], "candidates": len(measured)}), flush=True)
            del source_cache, target_cache, dense_logits, hidden
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as error:
            errors.append({"case_id": case["case_id"], "error": f"{type(error).__name__}: {error}"})
            print(json.dumps(errors[-1]), flush=True)
            break
    del model
    gc.collect()
    torch.cuda.empty_cache()
    status = {
        "status": "COMPLETE" if not errors and written == len(pending) else "PARTIAL",
        "selected_cases": len(selected_cases),
        "previously_completed": len(completed),
        "new_cases": written,
        "errors": errors,
    }
    _write_json(output / "SPLICE_MEASUREMENT_STATUS.json", status)
    return status


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2
        for index in order[cursor:end]:
            result[index] = rank
        cursor = end
    return result


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    x = np.asarray(_ranks(left), dtype=np.float64)
    y = np.asarray(_ranks(right), dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator else math.nan


def _design_matrix(
    points: Sequence[Mapping[str, Any]], modules: Sequence[str], *, interaction: bool
) -> np.ndarray:
    log_attention = np.log(np.maximum([float(point["attention_mass"]) for point in points], 1e-12))
    log_drift = np.log(np.maximum([float(point["raw_kv_drift"]) for point in points], 1e-12))
    columns = [np.ones(len(points)), log_drift]
    for module in modules[1:]:
        columns.append(np.asarray([float(point["module"] == module) for point in points]))
    if interaction:
        columns.append(log_attention)
        for module in modules:
            mask = np.asarray([float(point["module"] == module) for point in points])
            columns.append(mask * log_attention * log_drift)
    return np.column_stack(columns)


def _leave_one_task_out_predictions(
    points: Sequence[Mapping[str, Any]], modules: Sequence[str], *, interaction: bool
) -> list[float]:
    predictions = [math.nan] * len(points)
    tasks = sorted({str(point["instance_id"]) for point in points})
    target = np.log1p(
        np.asarray([float(point["actual_kv_output_relative_mean"]) for point in points]) * 1e6
    )
    for task in tasks:
        train_indices = [index for index, point in enumerate(points) if point["instance_id"] != task]
        test_indices = [index for index, point in enumerate(points) if point["instance_id"] == task]
        train = [points[index] for index in train_indices]
        test = [points[index] for index in test_indices]
        train_design = _design_matrix(train, modules, interaction=interaction)
        test_design = _design_matrix(test, modules, interaction=interaction)
        coefficients, _, _, _ = np.linalg.lstsq(train_design, target[train_indices], rcond=None)
        for index, value in zip(test_indices, test_design @ coefficients, strict=True):
            predictions[index] = float(value)
    return predictions


def _bootstrap_mean(values: Sequence[float], seed: int, draws: int = 4000) -> dict[str, float]:
    rng = random.Random(seed)
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"q025": math.nan, "median": math.nan, "q975": math.nan}
    samples = [statistics.fmean(rng.choice(finite) for _ in finite) for _ in range(draws)]
    return {
        "q025": float(np.quantile(samples, 0.025)),
        "median": float(np.quantile(samples, 0.5)),
        "q975": float(np.quantile(samples, 0.975)),
    }


def summarize(output: Path) -> dict[str, Any]:
    destination = output / "RESULT.json"
    cells = json.loads((output / "CELL_REGISTRATION.json").read_text())
    if cells["status"] != "REGISTERED_BEFORE_PHYSICAL_OUTCOMES":
        raise RuntimeError("physical outcome registration did not pass")
    internal_points = {str(point["point_id"]): point for point in cells["cell_points"]}
    splice_path = output / "SPLICE_OBSERVATIONS.jsonl"
    splice_rows = [json.loads(line) for line in splice_path.read_text().splitlines() if line.strip()]
    if not splice_rows or any(not _splice_complete(row) for row in splice_rows):
        raise ValueError("physical splice observations are incomplete")
    points = []
    physical_candidates = []
    for row in splice_rows:
        for candidate in row["candidates"]:
            physical_candidates.append(
                {
                    "case_id": row["case_id"],
                    "instance_id": row["instance_id"],
                    "candidate_id": candidate["candidate_id"],
                    **{
                        f"{mode}_final_logit_js": candidate["physical_splice"][mode]["final_logit_js"]
                        for mode in ("key_only", "value_only", "kv")
                    },
                }
            )
            for module, metrics in candidate["local_modules"].items():
                point_id = f"{row['case_id']}::{candidate['candidate_id']}::{module}"
                if point_id not in internal_points or module not in cells["qualifying_modules"]:
                    continue
                points.append(
                    {
                        **internal_points[point_id],
                        **metrics,
                        "kv_final_logit_js": candidate["physical_splice"]["kv"]["final_logit_js"],
                    }
                )
    modules = list(cells["qualifying_modules"])
    baseline_prediction = _leave_one_task_out_predictions(points, modules, interaction=False)
    interaction_prediction = _leave_one_task_out_predictions(points, modules, interaction=True)
    target = [float(point["actual_kv_output_relative_mean"]) for point in points]
    per_task_delta = []
    for task in sorted({str(point["instance_id"]) for point in points}):
        indices = [index for index, point in enumerate(points) if point["instance_id"] == task]
        if len(indices) < 3:
            continue
        baseline = _spearman([baseline_prediction[index] for index in indices], [target[index] for index in indices])
        interaction = _spearman([interaction_prediction[index] for index in indices], [target[index] for index in indices])
        if math.isfinite(baseline) and math.isfinite(interaction):
            per_task_delta.append(interaction - baseline)
    overall_baseline = _spearman(baseline_prediction, target)
    overall_interaction = _spearman(interaction_prediction, target)
    delta_bootstrap = _bootstrap_mean(per_task_delta, 2026080704)
    factorial = {}
    pair_directions = []
    high_drift_ratios = []
    for module in modules:
        module_points = [point for point in points if point["module"] == module]
        factorial[module] = {}
        for cell in (
            "low_attention__low_drift",
            "high_attention__low_drift",
            "low_attention__high_drift",
            "high_attention__high_drift",
        ):
            values = [
                float(point["actual_kv_output_relative_mean"])
                for point in module_points
                if point["cell"] == cell
            ]
            factorial[module][cell] = {
                "points": len(values),
                "median_local_output_change": statistics.median(values),
            }
        high_high = factorial[module]["high_attention__high_drift"]["median_local_output_change"]
        low_high = factorial[module]["low_attention__high_drift"]["median_local_output_change"]
        high_drift_ratios.append(high_high / max(low_high, 1e-20))
        for task in sorted({str(point["instance_id"]) for point in module_points}):
            high_values = [float(point["actual_kv_output_relative_mean"]) for point in module_points if point["instance_id"] == task and point["cell"] == "high_attention__high_drift"]
            low_values = [float(point["actual_kv_output_relative_mean"]) for point in module_points if point["instance_id"] == task and point["cell"] == "low_attention__high_drift"]
            if high_values and low_values:
                pair_directions.append(statistics.median(high_values) > statistics.median(low_values))
    high_drift_ratio = statistics.median(high_drift_ratios)
    pair_fraction = statistics.fmean(pair_directions) if pair_directions else 0.0
    gate_results = {
        "held_out_spearman_improvement_min_0_05": overall_interaction - overall_baseline >= 0.05,
        "task_bootstrap_delta_lower_bound_positive": delta_bootstrap["q025"] > 0,
        "high_attention_high_drift_ratio_min_1_25": high_drift_ratio >= 1.25,
        "paired_task_module_direction_fraction_min_0_60": pair_fraction >= 0.60,
    }
    result = {
        "status": "COMPLETE",
        "decision": "SUPPORTED_MODULE_CONDITIONED_LOCAL_RISK" if all(gate_results.values()) else "NOT_SUPPORTED_FOR_POLICY_PROMOTION",
        "tasks": len({str(point["instance_id"]) for point in points}),
        "candidate_module_points": len(points),
        "physical_candidates": len(physical_candidates),
        "qualifying_modules": modules,
        "leave_one_task_out": {
            "baseline_drift_module_spearman": overall_baseline,
            "module_attention_interaction_spearman": overall_interaction,
            "improvement": overall_interaction - overall_baseline,
            "per_task_improvement_mean": statistics.fmean(per_task_delta) if per_task_delta else math.nan,
            "per_task_improvement_bootstrap": delta_bootstrap,
        },
        "factorial": factorial,
        "high_attention_vs_low_attention_at_high_drift_median_ratio": high_drift_ratio,
        "paired_task_module_high_attention_higher_fraction": pair_fraction,
        "gate_results": gate_results,
        "physical_component_js": {
            mode: {
                "median": statistics.median(float(row[f"{mode}_final_logit_js"]) for row in physical_candidates),
                "top_level_points": len(physical_candidates),
            }
            for mode in ("key_only", "value_only", "kv")
        },
        "local_to_final": {
            "local_output_change_to_kv_final_js_spearman": _spearman(
                target, [float(point["kv_final_logit_js"]) for point in points]
            )
        },
        "scope": (
            "task-disjoint 3B mechanism proxy; local risk motivation only. "
            "Not functional accuracy, TTFT, or an online selector."
        ),
    }
    _write_json(destination, result)
    with (output / "CONFIRMATORY_POINTS.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = sorted({key for point in points for key in point if not isinstance(point[key], (dict, list))})
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(points)
    return result


def _physical_training_points(
    output: Path, cells: Mapping[str, Any]
) -> list[dict[str, Any]]:
    internal_points = {str(point["point_id"]): point for point in cells["cell_points"]}
    path = output / "SPLICE_OBSERVATIONS.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    points = []
    for row in rows:
        for candidate in row["candidates"]:
            for module, metrics in candidate["local_modules"].items():
                point_id = f"{row['case_id']}::{candidate['candidate_id']}::{module}"
                if point_id in internal_points and module in cells["qualifying_modules"]:
                    points.append({**internal_points[point_id], **metrics})
    return points


def _crossfit_candidate_risk(
    *,
    training_points: Sequence[Mapping[str, Any]],
    all_points: Sequence[Mapping[str, Any]],
    modules: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    risks: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    tasks = sorted({str(point["instance_id"]) for point in all_points})
    for task in tasks:
        train = [point for point in training_points if point["instance_id"] != task]
        test = [
            point
            for point in all_points
            if point["instance_id"] == task and point["module"] in modules
        ]
        if not train or not test:
            continue
        target = np.log1p(
            np.asarray([float(point["actual_kv_output_relative_mean"]) for point in train])
            * 1e6
        )
        coefficients, _, _, _ = np.linalg.lstsq(
            _design_matrix(train, modules, interaction=True), target, rcond=None
        )
        prediction = _design_matrix(test, modules, interaction=True) @ coefficients
        thresholds[task] = float(np.quantile(target, 0.75))
        grouped: dict[str, list[float]] = defaultdict(list)
        for point, value in zip(test, prediction, strict=True):
            grouped[f"{point['case_id']}::{point['candidate_id']}"].append(float(value))
        for key, values in grouped.items():
            risks[key] = max(values)
    return risks, thresholds


def prepare_multi(output: Path) -> dict[str, Any]:
    destination = output / "MULTI_REGISTRATION.json"
    if destination.exists():
        return json.loads(destination.read_text())
    single_result = json.loads((output / "RESULT.json").read_text())
    if single_result["decision"] != "SUPPORTED_MODULE_CONDITIONED_LOCAL_RISK":
        value = {
            "status": "STOPPED_BEFORE_MULTI_OUTCOMES",
            "reason": "single-island module-conditioned gates did not all pass",
            "single_result_sha256": _sha256(output / "RESULT.json"),
        }
        _write_json(destination, value)
        return value
    cells = json.loads((output / "CELL_REGISTRATION.json").read_text())
    modules = list(cells["qualifying_modules"])
    training_points = _physical_training_points(output, cells)
    all_points = [
        point for point in cells["cell_points"] if point["module"] in modules
    ]
    risks, thresholds = _crossfit_candidate_risk(
        training_points=training_points, all_points=all_points, modules=modules
    )
    design = json.loads((output / "DESIGN.json").read_text())
    eligible = []
    for case in design["cases"]:
        task = str(case["instance_id"])
        threshold = thresholds.get(task)
        if threshold is None or len(case["candidates"]) < 4:
            continue
        available = []
        for candidate in case["candidates"]:
            key = f"{case['case_id']}::{candidate['candidate_id']}"
            if key in risks:
                available.append((candidate, risks[key]))
        if len(available) < 4:
            continue
        current = [candidate["candidate_id"] for candidate, _ in available[:3]]
        safe = [(candidate, risk) for candidate, risk in available if risk <= threshold]
        if len(safe) < 3:
            continue
        safe.sort(
            key=lambda row: (
                -int(bool(row[0].get("exact_path"))),
                -int(bool(row[0].get("same_directory"))),
                int(row[0].get("interaction_distance", 10**9)),
                row[1],
                str(row[0]["candidate_id"]),
            )
        )
        risk_aware = [candidate["candidate_id"] for candidate, _ in safe[:3]]
        random_ids = [
            candidate["candidate_id"]
            for candidate, _ in sorted(
                available,
                key=lambda row: _stable_hash(
                    f"{case['case_id']}::{row[0]['candidate_id']}", CELL_SEED
                ),
            )[:3]
        ]
        if len(set(current)) != 3 or len(set(risk_aware)) != 3 or len(set(random_ids)) != 3:
            raise ValueError("multi-island arms require three unique candidates")
        eligible.append(
            {
                "case_id": case["case_id"],
                "instance_id": task,
                "request_index": case["request_index"],
                "arms": {
                    "current_recency": current,
                    "module_risk_then_path_utility": risk_aware,
                    "seeded_random": random_ids,
                },
                "candidate_risk": {
                    str(candidate["candidate_id"]): risk for candidate, risk in available
                },
                "crossfit_safe_threshold": threshold,
            }
        )
    selected = _balanced_requests(eligible, 24)
    tasks = len({row["instance_id"] for row in selected})
    gates = {"requests_min_24": len(selected) >= 24, "tasks_min_8": tasks >= 8}
    value = {
        "status": "REGISTERED_BEFORE_MULTI_OUTCOMES" if all(gates.values()) else "STOPPED_BEFORE_MULTI_OUTCOMES",
        "single_result_sha256": _sha256(output / "RESULT.json"),
        "cell_registration_sha256": _sha256(output / "CELL_REGISTRATION.json"),
        "training_splices_sha256": _sha256(output / "SPLICE_OBSERVATIONS.jsonl"),
        "selection": (
            "leave-one-task-out module risk; retain candidates below the training-fold "
            "75th percentile; rank the safe set by exact path, same directory, recency"
        ),
        "copy_contract": {"islands_per_arm": 3, "tokens_per_island": 128, "tokens_per_arm": 384},
        "eligible_requests": len(eligible),
        "selected_requests": len(selected),
        "selected_tasks": tasks,
        "gates": gates,
        "cases": selected,
        "outcomes_unopened_at_registration": True,
    }
    _write_json(destination, value)
    return value


@torch.inference_mode()
def _local_multi_metrics(
    *,
    model: Any,
    case: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    hidden: Mapping[int, torch.Tensor],
    theta: float,
) -> dict[str, float]:
    earliest_end = min(int(candidate["target_start"]) + CANDIDATE_TOKENS for candidate in candidates)
    target_tokens = len(case["target_input_ids"])
    key_positions = torch.arange(target_tokens, device="cuda")
    delta_sum = tv_sum = mass_sum = 0.0
    points = 0
    for layer_index in PROBE_LAYERS:
        num_heads = int(model.config.num_attention_heads)
        num_kv_heads = int(model.config.num_key_value_heads)
        groups = num_heads // num_kv_heads
        head_dim = int(getattr(model.config, "head_dim", 0)) or int(model.config.hidden_size) // num_heads
        target_key, target_value = target_cache[layer_index]
        source_key, source_value = source_cache[layer_index]
        dense_key = target_key.to("cuda")
        dense_value = target_value.to("cuda")
        stale_key = dense_key.clone()
        stale_value = dense_value.clone()
        spans = []
        for candidate in candidates:
            source_start = int(candidate["source_start"])
            target_start = int(candidate["target_start"])
            target_end = target_start + CANDIDATE_TOKENS
            spans.append((target_start, target_end))
            stale_key[:, target_start:target_end] = _rope_shift(
                source_key[:, source_start : source_start + CANDIDATE_TOKENS].to("cuda"),
                target_start - source_start,
                theta,
            )
            stale_value[:, target_start:target_end] = source_value[
                :, source_start : source_start + CANDIDATE_TOKENS
            ].to("cuda")
        dense_key_h = dense_key.repeat_interleave(groups, dim=0)
        stale_key_h = stale_key.repeat_interleave(groups, dim=0)
        dense_value_h = dense_value.repeat_interleave(groups, dim=0)
        stale_value_h = stale_value.repeat_interleave(groups, dim=0)
        for block in case["target_blocks"]:
            left, right = int(block["start"]), int(block["end"])
            if left < earliest_end:
                continue
            for query_left in range(left, right, QUERY_CHUNK):
                query_right = min(right, query_left + QUERY_CHUNK)
                query, positions = _query_tensor(
                    model=model,
                    layer_index=layer_index,
                    hidden=hidden[layer_index],
                    query_left=query_left,
                    query_right=query_right,
                )
                dense_scores = torch.matmul(query[0].float(), dense_key_h.float().transpose(-1, -2)) / math.sqrt(head_dim)
                stale_scores = torch.matmul(query[0].float(), stale_key_h.float().transpose(-1, -2)) / math.sqrt(head_dim)
                causal = key_positions.view(1, 1, -1) > positions.view(1, -1, 1)
                dense_weights = torch.softmax(dense_scores.masked_fill(causal, -torch.inf), dim=-1)
                stale_weights = torch.softmax(stale_scores.masked_fill(causal, -torch.inf), dim=-1)
                dense_output = torch.matmul(dense_weights, dense_value_h.float())
                stale_output = torch.matmul(stale_weights, stale_value_h.float())
                scale = dense_output.norm(dim=-1).clamp_min(1e-8)
                delta = (stale_output - dense_output).norm(dim=-1) / scale
                tv = (stale_weights - dense_weights).abs().sum(dim=-1) / 2.0
                mass = sum(dense_weights[..., start:end].sum(dim=-1) for start, end in spans)
                delta_sum += float(delta.sum())
                tv_sum += float(tv.sum())
                mass_sum += float(mass.sum())
                points += int(delta.numel())
                del query, positions, dense_scores, stale_scores, causal, dense_weights, stale_weights, dense_output, stale_output, scale, delta, tv, mass
        del dense_key, dense_value, stale_key, stale_value, dense_key_h, stale_key_h, dense_value_h, stale_value_h
        torch.cuda.empty_cache()
    return {
        "actual_kv_output_relative_mean": delta_sum / points,
        "attention_row_tv_mean": tv_sum / points,
        "dense_attention_mass_to_three_islands_mean": mass_sum / points,
        "head_query_layer_points": points,
    }


def _multi_complete(row: Mapping[str, Any]) -> bool:
    return (
        row.get("status") == "ok"
        and set(row.get("arms", {}))
        == {"current_recency", "module_risk_then_path_utility", "seeded_random"}
        and all(math.isfinite(float(value["final_logit_js"])) for value in row["arms"].values())
    )


@torch.inference_mode()
def measure_multi(output: Path, max_cases: int) -> dict[str, Any]:
    registration = json.loads((output / "MULTI_REGISTRATION.json").read_text())
    if registration["status"] != "REGISTERED_BEFORE_MULTI_OUTCOMES":
        raise RuntimeError("multi-island capacity gate did not pass")
    design = json.loads((output / "DESIGN.json").read_text())
    cases_by_id = {str(case["case_id"]): case for case in design["cases"]}
    selected = registration["cases"][:max_cases] if max_cases > 0 else registration["cases"]
    path = output / "MULTI_OBSERVATIONS.jsonl"
    completed = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if _multi_complete(row):
                    completed.add(str(row["case_id"]))
    pending = [row for row in selected if str(row["case_id"]) not in completed]
    if not pending:
        return {"status": "COMPLETE", "selected_cases": len(selected), "new_cases": 0}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    theta = _model_theta(model.config)
    errors = []
    written = 0
    for index, selected_row in enumerate(pending, 1):
        case = cases_by_id[str(selected_row["case_id"])]
        try:
            source_cache, _, _ = _dense_full(model, case["source_input_ids"], False)
            target_cache, dense_logits, hidden = _dense_full(model, case["target_input_ids"], True)
            candidates_by_id = {str(candidate["candidate_id"]): candidate for candidate in case["candidates"]}
            unique_ids = sorted({candidate_id for ids in selected_row["arms"].values() for candidate_id in ids})
            single_js = {}
            for candidate_id in unique_ids:
                candidate = candidates_by_id[candidate_id]
                logits = _physical_splice_logits(
                    model=model,
                    case={**case, "source_start": candidate["source_start"], "target_start": candidate["target_start"], "length": CANDIDATE_TOKENS},
                    source_cache=source_cache,
                    target_cache=target_cache,
                    theta=theta,
                    mode="kv",
                )
                single_js[candidate_id] = _js(dense_logits, logits)
                del logits
            arms = {}
            for arm, candidate_ids in selected_row["arms"].items():
                arm_candidates = [candidates_by_id[candidate_id] for candidate_id in candidate_ids]
                logits = _compose_splice(
                    model=model,
                    target_ids=case["target_input_ids"],
                    target_cache=target_cache,
                    source_cache=source_cache,
                    candidates=arm_candidates,
                    theta=theta,
                )
                local = _local_multi_metrics(
                    model=model,
                    case=case,
                    candidates=arm_candidates,
                    source_cache=source_cache,
                    target_cache=target_cache,
                    hidden=hidden,
                    theta=theta,
                )
                values = [single_js[candidate_id] for candidate_id in candidate_ids]
                arms[arm] = {
                    "candidate_ids": candidate_ids,
                    "final_logit_js": _js(dense_logits, logits),
                    "top1_changed": int(dense_logits.argmax()) != int(logits.argmax()),
                    "single_js_max": max(values),
                    "single_js_sum": sum(values),
                    **local,
                }
                del logits
            row = {
                "status": "ok",
                "case_id": case["case_id"],
                "instance_id": case["instance_id"],
                "single_candidate_js": single_js,
                "arms": arms,
            }
            if not _multi_complete(row):
                raise RuntimeError("incomplete multi-island labels")
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
            print(json.dumps({"case": index, "pending": len(pending), "case_id": case["case_id"]}), flush=True)
            del source_cache, target_cache, dense_logits, hidden
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as error:
            errors.append({"case_id": case["case_id"], "error": f"{type(error).__name__}: {error}"})
            print(json.dumps(errors[-1]), flush=True)
            break
    del model
    gc.collect()
    torch.cuda.empty_cache()
    status = {
        "status": "COMPLETE" if not errors and written == len(pending) else "PARTIAL",
        "selected_cases": len(selected),
        "previously_completed": len(completed),
        "new_cases": written,
        "errors": errors,
    }
    _write_json(output / "MULTI_MEASUREMENT_STATUS.json", status)
    return status


def summarize_multi(output: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (output / "MULTI_OBSERVATIONS.jsonl").read_text().splitlines()
        if line.strip()
    ]
    registration = json.loads((output / "MULTI_REGISTRATION.json").read_text())
    if len(rows) != registration["selected_requests"] or any(not _multi_complete(row) for row in rows):
        raise ValueError("multi-island observations are incomplete")
    arms = ("current_recency", "module_risk_then_path_utility", "seeded_random")
    summary = {}
    for arm in arms:
        summary[arm] = {
            metric: statistics.median(float(row["arms"][arm][metric]) for row in rows)
            for metric in ("final_logit_js", "actual_kv_output_relative_mean", "attention_row_tv_mean")
        }
    current = "current_recency"
    treatment = "module_risk_then_path_utility"
    js_ratio = summary[treatment]["final_logit_js"] / max(summary[current]["final_logit_js"], 1e-20)
    tv_ratio = summary[treatment]["attention_row_tv_mean"] / max(summary[current]["attention_row_tv_mean"], 1e-20)
    win_fraction = statistics.fmean(
        float(row["arms"][treatment]["final_logit_js"] < row["arms"][current]["final_logit_js"])
        for row in rows
    )
    result = {
        "status": "COMPLETE",
        "decision": "SUPPORTED_MULTI_ISLAND_RISK_SELECTION" if js_ratio <= 0.9 and tv_ratio <= 0.9 and win_fraction >= 0.60 else "NOT_SUPPORTED_FOR_MULTI_ISLAND_PROMOTION",
        "requests": len(rows),
        "tasks": len({str(row["instance_id"]) for row in rows}),
        "arms": summary,
        "risk_aware_vs_current": {
            "final_logit_js_ratio": js_ratio,
            "attention_row_tv_ratio": tv_ratio,
            "final_logit_js_win_fraction": win_fraction,
        },
        "composition_audit": {
            arm: {
                "actual_to_max_single_js_median_ratio": statistics.median(
                    float(row["arms"][arm]["final_logit_js"]) / max(float(row["arms"][arm]["single_js_max"]), 1e-20)
                    for row in rows
                ),
                "actual_to_sum_single_js_median_ratio": statistics.median(
                    float(row["arms"][arm]["final_logit_js"]) / max(float(row["arms"][arm]["single_js_sum"]), 1e-20)
                    for row in rows
                ),
            }
            for arm in arms
        },
    }
    _write_json(output / "MULTI_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--trajectory-root", type=Path, default=TRAJECTORY_ROOT)
    internals = sub.add_parser("measure-internals")
    internals.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    internals.add_argument("--max-cases", type=int, default=0)
    cells = sub.add_parser("freeze-cells")
    cells.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    splices = sub.add_parser("measure-splices")
    splices.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    splices.add_argument("--max-cases", type=int, default=0)
    summary = sub.add_parser("summarize")
    summary.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    multi_prepare = sub.add_parser("prepare-multi")
    multi_prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    multi_measure = sub.add_parser("measure-multi")
    multi_measure.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    multi_measure.add_argument("--max-cases", type=int, default=0)
    multi_summary = sub.add_parser("summarize-multi")
    multi_summary.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.output, args.trajectory_root)
    elif args.command == "measure-internals":
        value = measure_internals(args.output, args.max_cases)
    elif args.command == "freeze-cells":
        value = freeze_cells(args.output)
    elif args.command == "measure-splices":
        value = measure_splices(args.output, args.max_cases)
    elif args.command == "summarize":
        value = summarize(args.output)
    elif args.command == "prepare-multi":
        value = prepare_multi(args.output)
    elif args.command == "measure-multi":
        value = measure_multi(args.output, args.max_cases)
    else:
        value = summarize_multi(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
