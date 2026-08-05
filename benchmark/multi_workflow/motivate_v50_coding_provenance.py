#!/usr/bin/env python3
"""Test whether grounded coding facts are safer to reuse than agent decisions.

M50 uses frozen Dense SWE-bench agent trajectories.  In each adjacent rolling
request pair it matches one successful read-only tool observation to the
nearest assistant reasoning/tool-call block.  Both candidates are exactly 128
tokens, share the same source and target prompts, and undergo the same rolling
history transition.  The experiment measures RoPE-corrected K/V drift and the
causal error after physically splicing source K/V into the Dense target.

This is an offline motivation experiment.  It does not implement a runtime
selector and it must not be reported as functional accuracy or latency.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from benchmark.multi_workflow.coding_reuse_policy import (
    is_successful_readonly_evidence,
)
from benchmark.multi_workflow.measure_probehead_v12 import _advance
from benchmark.multi_workflow.measure_sessiongraph_atlas import (
    _cpu_cache,
    _js,
    _layers,
    _rope_shift,
)
from benchmark.multi_workflow.motivate_v48_attention_kv_risk import (
    _cache_from_dense_prefix,
    _cosine_deviation_by_head,
    _dense_source,
    _model_theta,
    _relative_l2_by_head,
)
from benchmark.multi_workflow.run_bridge_reuse_pilot import write_json


ROOT = Path("/home/gfy/CodeMAS_Project")
MODEL = Path(
    "/home/gfy/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/"
    "488639f1ff808d1d3d0ba301aef8c11461451ec5"
)
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_m50_coding_provenance_20260805"
    / "matched20"
)
TRAJECTORY_ROOTS = (
    ROOT
    / "kvflow-artifacts/impactkv_v44_dense_sensitive_v40_20260728/tasks",
    ROOT
    / "kvflow-artifacts/impactkv_bridge_agent_accuracy_speed_20260726/"
    "dense/full_18",
    ROOT
    / "kvflow-artifacts/impactkv_v43_new_verified_v40_20260728/tasks",
)
ROLLING_GROUPS = 6
CANDIDATE_TOKENS = 128
ANSWER_TOKENS = 16
MAX_POSITION_DISTANCE_TOKENS = 512
MAX_POSITION_DISTANCE_FRACTION = 0.05
PROBE_LAYERS = (0, 8, 17, 26, 35)
SPLICE_CHUNK_SIZE = 512
RANDOM_SEED = 20260805

ROLLING_NOTICE = (
    '<history_compaction dropped_turn_groups="{dropped}">'
    "Earlier interaction details were omitted to stay within the rolling "
    "history budget. Repository state persists; the most recent complete "
    "interactions follow."
    "</history_compaction>"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_ids_hash(ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _turn_groups(messages: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "assistant" and current:
            groups.append(current)
            current = []
        current.append(message)
    if current:
        groups.append(current)
    return groups


def _render_message_literal(message: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(message))
    role = str(value["role"])
    if role == "assistant" and value.get("tool_calls"):
        rendered = "<|im_start|>assistant\n"
        if value.get("content"):
            rendered += str(value["content"]).strip() + "\n"
        for wrapped_call in value["tool_calls"]:
            call = wrapped_call.get("function", wrapped_call)
            rendered += f"<tool_call>\n<function={call['name']}>\n"
            arguments = call.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw_arguments": arguments}
            for name, argument in arguments.items():
                rendered += f"<parameter={name}>{argument}</parameter>\n"
            rendered += "</function>\n</tool_call>\n"
        return rendered + "<|im_end|>\n"
    if role == "tool":
        return (
            "<|im_start|>user\n<tool_response>\n"
            + str(value.get("content") or "")
            + "\n</tool_response><|im_end|>\n"
        )
    return (
        f"<|im_start|>{role}\n"
        + str(value.get("content") or "")
        + "<|im_end|>\n"
    )


def _render_rolling(
    tokenizer: Any,
    base: Sequence[dict[str, Any]],
    groups: Sequence[Sequence[dict[str, Any]]],
) -> tuple[list[int], dict[tuple[int, int], tuple[int, int]]]:
    """Render a rolling prompt and return exact per-message token spans."""

    ids: list[int] = []
    spans: dict[tuple[int, int], tuple[int, int]] = {}

    def append_literal(literal: str) -> tuple[int, int]:
        start = len(ids)
        ids.extend(tokenizer.encode(literal, add_special_tokens=False))
        return start, len(ids)

    for message in base:
        append_literal(_render_message_literal(message))
    dropped = max(0, len(groups) - ROLLING_GROUPS)
    if dropped:
        append_literal(
            _render_message_literal(
                {
                    "role": "user",
                    "content": ROLLING_NOTICE.format(dropped=dropped),
                }
            )
        )
    for group_index in range(dropped, len(groups)):
        for message_index, message in enumerate(groups[group_index]):
            spans[(group_index, message_index)] = append_literal(
                _render_message_literal(message)
            )
    append_literal("<|im_start|>assistant\n")
    return ids, spans


def _trajectory_paths() -> dict[str, Path]:
    """Return one deterministic Dense trajectory per task, root-prioritized."""

    selected: dict[str, Path] = {}
    for root in TRAJECTORY_ROOTS:
        for path in sorted(root.glob("**/*.traj.json")):
            if "/dense/" not in str(path) and "full_18" not in str(path):
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            instance_id = value.get("instance_id")
            if instance_id and str(instance_id) not in selected:
                selected[str(instance_id)] = path
    return selected


def _message_candidate(
    *,
    category: str,
    group_index: int,
    message_index: int,
    source_ids: Sequence[int],
    source_spans: Mapping[tuple[int, int], tuple[int, int]],
    target_ids: Sequence[int],
    target_spans: Mapping[tuple[int, int], tuple[int, int]],
) -> dict[str, Any] | None:
    key = (group_index, message_index)
    if key not in source_spans or key not in target_spans:
        return None
    source_left, source_right = source_spans[key]
    target_left, target_right = target_spans[key]
    if (
        source_right - source_left < CANDIDATE_TOKENS
        or target_right - target_left < CANDIDATE_TOKENS
    ):
        return None
    source_start = source_right - CANDIDATE_TOKENS
    target_start = target_right - CANDIDATE_TOKENS
    segment = list(source_ids[source_start:source_right])
    if segment != list(target_ids[target_start:target_right]):
        raise ValueError("message tail is not token-identical across prompts")
    return {
        "category": category,
        "group_index": group_index,
        "message_index": message_index,
        "length": CANDIDATE_TOKENS,
        "segment_token_hash": _token_ids_hash(segment),
        "source_start": source_start,
        "target_start": target_start,
    }


def _candidate_pool(tokenizer: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for instance_id, path in _trajectory_paths().items():
        value = json.loads(path.read_text(encoding="utf-8"))
        messages = value["messages"]
        base = messages[:2]
        groups = _turn_groups(messages[2:])
        # target_completed is the number of completed groups in the target
        # prompt.  The following group supplies the frozen next-action label.
        for target_completed in range(7, len(groups)):
            source_ids, source_spans = _render_rolling(
                tokenizer, base, groups[: target_completed - 1]
            )
            target_ids, target_spans = _render_rolling(
                tokenizer, base, groups[:target_completed]
            )
            if len(source_ids) > 30_000 or len(target_ids) > 30_000:
                continue
            overlap = sorted(set(source_spans) & set(target_spans))
            grounded: list[dict[str, Any]] = []
            decisions: list[dict[str, Any]] = []
            for group_index in sorted({key[0] for key in overlap}):
                group = groups[group_index]
                grounded_group = is_successful_readonly_evidence(group)
                for message_index, message in enumerate(group):
                    role = message.get("role")
                    if role == "tool" and grounded_group:
                        candidate = _message_candidate(
                            category="grounded_readonly_tool",
                            group_index=group_index,
                            message_index=message_index,
                            source_ids=source_ids,
                            source_spans=source_spans,
                            target_ids=target_ids,
                            target_spans=target_spans,
                        )
                        if candidate:
                            grounded.append(candidate)
                    elif role == "assistant":
                        candidate = _message_candidate(
                            category="assistant_decision",
                            group_index=group_index,
                            message_index=message_index,
                            source_ids=source_ids,
                            source_spans=source_spans,
                            target_ids=target_ids,
                            target_spans=target_spans,
                        )
                        if candidate:
                            decisions.append(candidate)
            if not grounded or not decisions:
                continue
            possible = [
                (abs(tool["target_start"] - decision["target_start"]), tool, decision)
                for tool in grounded
                for decision in decisions
            ]
            distance, tool, decision = min(
                possible,
                key=lambda item: (
                    item[0],
                    item[1]["target_start"],
                    item[2]["target_start"],
                ),
            )
            if distance > MAX_POSITION_DISTANCE_TOKENS:
                continue
            if distance / len(target_ids) > MAX_POSITION_DISTANCE_FRACTION:
                continue
            answer_message = groups[target_completed][0]
            answer_literal = str(answer_message.get("content") or "")
            answer_ids = tokenizer.encode(answer_literal, add_special_tokens=False)
            if not answer_ids:
                continue
            case_id = f"{instance_id}-q{target_completed + 1}"
            rows.append(
                {
                    "answer_ids": answer_ids[:ANSWER_TOKENS],
                    "case_id": case_id,
                    "instance_id": instance_id,
                    "position_distance_fraction": distance / len(target_ids),
                    "position_distance_tokens": distance,
                    "source_input_ids": source_ids,
                    "source_prompt_hash": _token_ids_hash(source_ids),
                    "target_input_ids": target_ids,
                    "target_prompt_hash": _token_ids_hash(target_ids),
                    "target_request_index": target_completed + 1,
                    "trajectory_path": str(path),
                    "trajectory_sha256": _sha256(path),
                    "candidates": [tool, decision],
                }
            )
    return rows


def _balanced_select(rows: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Deterministically take one row per task before any second row."""

    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{RANDOM_SEED}:{row['case_id']}".encode()
        ).hexdigest(),
    )
    selected: list[dict[str, Any]] = []
    selected_case_ids: set[str] = set()
    used: dict[str, int] = {}
    round_index = 0
    while len(selected) < limit:
        added = False
        for row in ranked:
            instance_id = str(row["instance_id"])
            case_id = str(row["case_id"])
            if case_id in selected_case_ids:
                continue
            if used.get(instance_id, 0) != round_index:
                continue
            selected.append(row)
            selected_case_ids.add(case_id)
            used[instance_id] = round_index + 1
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
        round_index += 1
    return selected


def prepare(output: Path, limit: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    eligible = _candidate_pool(tokenizer)
    cases = _balanced_select(eligible, limit)
    if len(cases) < limit:
        raise ValueError(f"only {len(cases)} eligible cases for requested {limit}")
    output.mkdir(parents=True)
    design_path = output / "DESIGN.json"
    write_json(
        design_path,
        {
            "cases": cases,
            "eligible_cases_before_sampling": len(eligible),
            "model": str(MODEL),
        },
    )
    registration = {
        "status": "REGISTERED_BEFORE_GPU",
        "purpose": (
            "test the coding-specific claim that externally grounded, "
            "read-only repository facts are safer lossy-KV sources than "
            "assistant reasoning/tool-call decisions"
        ),
        "design_sha256": _sha256(design_path),
        "model": str(MODEL),
        "cases": len(cases),
        "tasks": len({row["instance_id"] for row in cases}),
        "eligible_cases_before_sampling": len(eligible),
        "matching_contract": {
            "same_real_dense_agent_source_target_prompts": True,
            "same_rolling_transition_within_pair": True,
            "candidate_tokens_each": CANDIDATE_TOKENS,
            "candidate_token_identical_source_target": True,
            "maximum_position_distance_tokens": MAX_POSITION_DISTANCE_TOKENS,
            "maximum_position_distance_fraction": MAX_POSITION_DISTANCE_FRACTION,
            "one_case_per_task_before_second_case": True,
            "answer_tokens": ANSWER_TOKENS,
        },
        "primary_labels": [
            "causal final-logit JS versus Dense",
            "16-token frozen next-action NLL delta versus Dense",
        ],
        "mechanism_labels": [
            "RoPE-corrected K cosine drift",
            "V cosine drift",
            "relative K/V L2 drift",
        ],
        "frozen_support_rule": {
            "grounded_lower_JS_pair_fraction_min": 0.65,
            "equal_position_adjusted_geometric_JS_ratio_max": 0.85,
            "minimum_complete_cases": 16,
        },
        "interpretation_limits": [
            "offline Qwen2.5-Coder-3B diagnostic, not Qwen3 agent accuracy",
            "next-action NLL is not SWE-bench resolve rate",
            "the pair is near-position matched, not literally co-located",
            "no runtime selector may be promoted from M50 alone",
        ],
        "next_stage_if_supported": (
            "run same-path versus path-disjoint mutation and multi-island "
            "interaction controls before changing V46"
        ),
    }
    write_json(output / "REGISTRATION.json", registration)
    return registration


def _kv_metrics(
    *,
    candidate: Mapping[str, Any],
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    theta: float,
) -> dict[str, float]:
    source_start = int(candidate["source_start"])
    target_start = int(candidate["target_start"])
    length = int(candidate["length"])
    delta = target_start - source_start
    key_cosine: list[float] = []
    value_cosine: list[float] = []
    key_l2: list[float] = []
    value_l2: list[float] = []
    for layer_index in PROBE_LAYERS:
        source_key, source_value = source_cache[layer_index]
        target_key, target_value = target_cache[layer_index]
        source_key = _rope_shift(
            source_key[:, source_start : source_start + length], delta, theta
        )
        source_value = source_value[:, source_start : source_start + length]
        target_key = target_key[:, target_start : target_start + length]
        target_value = target_value[:, target_start : target_start + length]
        key_cosine.extend(
            float(value)
            for value in _cosine_deviation_by_head(source_key, target_key).tolist()
        )
        value_cosine.extend(
            float(value)
            for value in _cosine_deviation_by_head(source_value, target_value).tolist()
        )
        key_l2.extend(
            float(value)
            for value in _relative_l2_by_head(source_key, target_key).tolist()
        )
        value_l2.extend(
            float(value)
            for value in _relative_l2_by_head(source_value, target_value).tolist()
        )
    return {
        "key_cosine_drift_mean": statistics.fmean(key_cosine),
        "value_cosine_drift_mean": statistics.fmean(value_cosine),
        "kv_cosine_drift_mean": statistics.fmean(
            [max(key, value) for key, value in zip(key_cosine, value_cosine, strict=True)]
        ),
        "key_relative_l2_mean": statistics.fmean(key_l2),
        "value_relative_l2_mean": statistics.fmean(value_l2),
    }


def _append_source_candidate(
    *,
    model: Any,
    cache: Any,
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    candidate: Mapping[str, Any],
    theta: float,
) -> Any:
    source_start = int(candidate["source_start"])
    target_start = int(candidate["target_start"])
    length = int(candidate["length"])
    delta = target_start - source_start
    layers = []
    for (target_key, target_value), (source_key, source_value) in zip(
        _layers(cache), source_cache, strict=True
    ):
        copied_key = _rope_shift(
            source_key[:, source_start : source_start + length].to(target_key.device),
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
def _splice_with_cache(
    *,
    model: Any,
    target_ids: Sequence[int],
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    candidate: Mapping[str, Any],
    theta: float,
) -> tuple[Any, torch.Tensor]:
    start = int(candidate["target_start"])
    length = int(candidate["length"])
    cache: Any = _cache_from_dense_prefix(
        model=model, target_cache=target_cache, prefix_tokens=start
    )
    cache = _append_source_candidate(
        model=model,
        cache=cache,
        source_cache=source_cache,
        candidate=candidate,
        theta=theta,
    )
    cursor = start + length
    logits: torch.Tensor | None = None
    for offset in range(cursor, len(target_ids), SPLICE_CHUNK_SIZE):
        cache, next_logits = _advance(
            model, cache, target_ids[offset : offset + SPLICE_CHUNK_SIZE]
        )
        if next_logits is not None:
            logits = next_logits
    if logits is None:
        raise RuntimeError("candidate has no suffix for causal measurement")
    return cache, logits


@torch.inference_mode()
def _continuation_nll(
    model: Any,
    cache: Any,
    initial_logits: torch.Tensor,
    answer_ids: Sequence[int],
) -> float:
    labels = torch.tensor(answer_ids, dtype=torch.long)
    losses = [
        float(-F.log_softmax(initial_logits.float(), dim=-1)[int(labels[0])])
    ]
    if len(answer_ids) > 1:
        input_ids = torch.tensor(
            [answer_ids[:-1]], device="cuda", dtype=torch.long
        )
        output = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        logits = output.logits[0].float().cpu()
        token_losses = F.cross_entropy(
            logits,
            labels[1:],
            reduction="none",
        )
        losses.extend(float(value) for value in token_losses.tolist())
        del output, logits, input_ids
    return statistics.fmean(losses)


def _measurement_complete(row: Mapping[str, Any]) -> bool:
    if row.get("status") != "ok" or len(row.get("candidates", [])) != 2:
        return False
    values = []
    for candidate in row["candidates"]:
        for key in (
            "kv_cosine_drift_mean",
            "causal_splice_logit_js",
            "next_action_nll_delta",
        ):
            values.append(float(candidate[key]))
    return all(math.isfinite(value) for value in values)


def measure(output: Path, max_cases: int) -> dict[str, Any]:
    design_path = output / "DESIGN.json"
    registration = json.loads((output / "REGISTRATION.json").read_text())
    if _sha256(design_path) != registration["design_sha256"]:
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
        return {"status": "COMPLETE", "cases": len(cases), "new_cases": 0}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU substitution is forbidden")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda").eval()
    theta = _model_theta(model.config)
    written = 0
    errors = []
    for index, case in enumerate(pending, 1):
        try:
            source_cache, _ = _dense_source(model, case["source_input_ids"])
            target_cache, dense_logits = _dense_source(
                model, case["target_input_ids"]
            )
            dense_continuation_cache = _cache_from_dense_prefix(
                model=model,
                target_cache=target_cache,
                prefix_tokens=len(case["target_input_ids"]),
            )
            dense_nll = _continuation_nll(
                model,
                dense_continuation_cache,
                dense_logits,
                case["answer_ids"],
            )
            measured = []
            for candidate in case["candidates"]:
                metrics = _kv_metrics(
                    candidate=candidate,
                    source_cache=source_cache,
                    target_cache=target_cache,
                    theta=theta,
                )
                splice_cache, splice_logits = _splice_with_cache(
                    model=model,
                    target_ids=case["target_input_ids"],
                    target_cache=target_cache,
                    source_cache=source_cache,
                    candidate=candidate,
                    theta=theta,
                )
                splice_nll = _continuation_nll(
                    model,
                    splice_cache,
                    splice_logits,
                    case["answer_ids"],
                )
                measured.append(
                    {
                        **candidate,
                        **metrics,
                        "position_fraction": candidate["target_start"]
                        / len(case["target_input_ids"]),
                        "prefix_shift_tokens": candidate["target_start"]
                        - candidate["source_start"],
                        "causal_splice_logit_js": _js(dense_logits, splice_logits),
                        "causal_splice_top1_changed": int(dense_logits.argmax())
                        != int(splice_logits.argmax()),
                        "next_action_nll": splice_nll,
                        "next_action_nll_delta": splice_nll - dense_nll,
                    }
                )
                del splice_cache, splice_logits
            row = {
                "status": "ok",
                "case_id": case["case_id"],
                "instance_id": case["instance_id"],
                "source_tokens": len(case["source_input_ids"]),
                "target_tokens": len(case["target_input_ids"]),
                "dense_next_action_nll": dense_nll,
                "position_distance_tokens": case["position_distance_tokens"],
                "candidates": measured,
            }
            if not _measurement_complete(row):
                raise RuntimeError("case produced incomplete/non-finite metrics")
            with observations_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
            print(
                json.dumps(
                    {
                        "case": index,
                        "case_id": case["case_id"],
                        "pending": len(pending),
                        "tool_js": measured[0]["causal_splice_logit_js"],
                        "decision_js": measured[1]["causal_splice_logit_js"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del source_cache, target_cache, dense_logits, dense_continuation_cache
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
    status = {
        "status": "COMPLETE" if not errors and written == len(pending) else "PARTIAL",
        "selected_cases": len(cases),
        "previously_completed_cases": len(completed),
        "new_cases": written,
        "errors": errors,
    }
    write_json(output / "MEASUREMENT_STATUS.json", status)
    return status


def _paired_rows(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for observation in observations:
        candidates = {
            str(row["category"]): row for row in observation["candidates"]
        }
        rows.append(
            {
                "case_id": observation["case_id"],
                "instance_id": observation["instance_id"],
                "tool": candidates["grounded_readonly_tool"],
                "decision": candidates["assistant_decision"],
            }
        )
    return rows


def _equal_position_ratio(
    rows: Sequence[Mapping[str, Any]], metric: str, epsilon: float
) -> dict[str, float]:
    """Regress paired log damage difference on paired position difference."""

    x = [
        float(row["tool"]["position_fraction"])
        - float(row["decision"]["position_fraction"])
        for row in rows
    ]
    y = [
        math.log(float(row["tool"][metric]) + epsilon)
        - math.log(float(row["decision"][metric]) + epsilon)
        for row in rows
    ]
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    slope = (
        sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
        / denominator
        if denominator
        else 0.0
    )
    intercept = y_mean - slope * x_mean
    return {
        "equal_position_geometric_ratio": math.exp(intercept),
        "position_slope": slope,
        "mean_position_difference_fraction": x_mean,
    }


def _binomial_lower_tail_probability(wins: int, trials: int) -> float:
    return sum(math.comb(trials, value) for value in range(wins, trials + 1)) / (
        2**trials
    )


def analyze(output: Path) -> dict[str, Any]:
    observations = [
        json.loads(line)
        for line in (output / "OBSERVATIONS.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not observations or any(not _measurement_complete(row) for row in observations):
        raise ValueError("observations are missing or incomplete")
    rows = _paired_rows(observations)
    metrics = {}
    for metric, epsilon in (
        ("causal_splice_logit_js", 1e-8),
        ("kv_cosine_drift_mean", 1e-8),
    ):
        tool = [float(row["tool"][metric]) for row in rows]
        decision = [float(row["decision"][metric]) for row in rows]
        wins = sum(a < b for a, b in zip(tool, decision, strict=True))
        metrics[metric] = {
            "tool_mean": statistics.fmean(tool),
            "tool_median": statistics.median(tool),
            "assistant_mean": statistics.fmean(decision),
            "assistant_median": statistics.median(decision),
            "tool_lower_pair_fraction": wins / len(rows),
            "one_sided_sign_probability": _binomial_lower_tail_probability(
                wins, len(rows)
            ),
            **_equal_position_ratio(rows, metric, epsilon),
        }
    nll_tool = [float(row["tool"]["next_action_nll_delta"]) for row in rows]
    nll_decision = [
        float(row["decision"]["next_action_nll_delta"]) for row in rows
    ]
    js = metrics["causal_splice_logit_js"]
    registration = json.loads((output / "REGISTRATION.json").read_text())
    gates = registration["frozen_support_rule"]
    complete_gate = len(rows) >= int(gates["minimum_complete_cases"])
    win_gate = js["tool_lower_pair_fraction"] >= float(
        gates["grounded_lower_JS_pair_fraction_min"]
    )
    ratio_gate = js["equal_position_geometric_ratio"] <= float(
        gates["equal_position_adjusted_geometric_JS_ratio_max"]
    )
    decision = "SUPPORTED" if complete_gate and win_gate and ratio_gate else "NOT_SUPPORTED"
    value = {
        "status": "COMPLETE",
        "decision": decision,
        "cases": len(rows),
        "tasks": len({str(row["instance_id"]) for row in rows}),
        "metrics": metrics,
        "next_action_nll_delta": {
            "tool_mean": statistics.fmean(nll_tool),
            "tool_median": statistics.median(nll_tool),
            "assistant_mean": statistics.fmean(nll_decision),
            "assistant_median": statistics.median(nll_decision),
            "tool_lower_pair_fraction": sum(
                a < b for a, b in zip(nll_tool, nll_decision, strict=True)
            )
            / len(rows),
        },
        "frozen_gate_results": {
            "minimum_complete_cases": complete_gate,
            "grounded_lower_JS_pair_fraction": win_gate,
            "equal_position_adjusted_geometric_JS_ratio": ratio_gate,
        },
        "scope": (
            "coding-provenance motivation under physical lossy-KV splice; "
            "not functional accuracy, TTFT, or a runtime policy"
        ),
        "next_step": (
            "same-path/disjoint mutation and multi-island controls"
            if decision == "SUPPORTED"
            else "do not use grounded provenance alone as a novelty claim"
        ),
    }
    write_json(output / "RESULT.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--limit", type=int, default=20)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    measure_parser.add_argument("--max-cases", type=int, default=0)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.output, args.limit)
    elif args.command == "measure":
        value = measure(args.output, args.max_cases)
    else:
        value = analyze(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
