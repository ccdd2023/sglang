#!/usr/bin/env python3
"""Resumable Hugging Face reference measurement for the V11 causal atlas."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.multi_workflow.sessiongraph_v11 import read_jsonl


MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _atlas_prompt_hash(token_ids: Sequence[int]) -> str:
    """Preserve the frozen atlas' native int32 token-hash encoding."""
    return hashlib.sha256(
        np.asarray(tuple(token_ids), dtype=np.int32).tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class PromptPair:
    source_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    source_span: tuple[int, int]
    target_span: tuple[int, int]
    disturbance: str

    def validate(self) -> None:
        source = self.source_ids[slice(*self.source_span)]
        target = self.target_ids[slice(*self.target_span)]
        if not source or source != target:
            raise ValueError("source/target module token slices are not identical")
        for span, values in (
            (self.source_span, self.source_ids),
            (self.target_span, self.target_ids),
        ):
            if span[0] < 0 or span[1] > len(values) or span[0] >= span[1]:
                raise ValueError("invalid module span")


def _module(turn: Mapping[str, Any], module_id: str) -> Mapping[str, Any]:
    return next(row for row in turn["modules"] if row["module_id"] == module_id)


def _replacement(values: Sequence[int], length: int, salt: str) -> list[int]:
    if not values:
        raise ValueError("empty replacement source")
    offset = int(_sha(salt), 16) % len(values)
    rotated = [*values[offset:], *values[:offset]]
    return (rotated * ((length + len(rotated) - 1) // len(rotated)))[:length]


def _replace(
    values: Sequence[int], span: tuple[int, int], replacement: Sequence[int]
) -> tuple[int, ...]:
    start, end = span
    if len(replacement) != end - start:
        raise ValueError("replacement changes token length")
    return tuple([*values[:start], *replacement, *values[end:]])


def build_prompt_pair(
    *,
    tokenizer: Any,
    turns: Mapping[tuple[str, int], Mapping[str, Any]],
    final_turns: Mapping[str, Mapping[str, Any]],
    session_id: str,
    module_id: str,
    disturbance: str,
) -> PromptPair:
    """Construct a deterministic disturbance using public prompt context only."""
    target_turn = final_turns[session_id]
    selected = _module(target_turn, module_id)
    target_ids = tuple(
        tokenizer.encode(target_turn["rendered_prompt"], add_special_tokens=False)
    )
    target_span = tuple(map(int, selected["token_span"]))
    source_ids, source_span = target_ids, target_span
    others = sorted(
        (value for value in final_turns if value != session_id),
        key=lambda value: _sha(f"sessiongraph-atlas-other|{session_id}|{value}"),
    )
    if disturbance in {
        "semantic_prefix",
        "upstream_edit",
        "change_after",
        "cross_task",
    } and not others:
        raise ValueError("disturbance requires another public session")
    other_ids = (
        tuple(
            tokenizer.encode(
                final_turns[others[0]]["rendered_prompt"],
                add_special_tokens=False,
            )
        )
        if others
        else ()
    )
    if disturbance == "identity":
        pass
    elif disturbance == "position_only":
        padding = tuple(
            tokenizer.encode("\n<!-- reserved position shift -->\n", add_special_tokens=False)
        )
        target_ids = (*padding, *target_ids)
        target_span = (
            target_span[0] + len(padding),
            target_span[1] + len(padding),
        )
    elif disturbance == "module_reorder":
        earlier = [
            row
            for row in target_turn["modules"]
            if int(row["token_span"][1]) <= target_span[0]
            and row["module_id"] != module_id
        ]
        if not earlier:
            raise ValueError("no earlier module for reorder")
        moved = min(
            earlier,
            key=lambda row: _sha(
                "sessiongraph-atlas-reorder|"
                f"{session_id}|{module_id}|{row['module_id']}"
            ),
        )
        left, right = map(int, moved["token_span"])
        selected_left, selected_right = target_span
        source_ids = tuple(
            [
                *target_ids[:left],
                *target_ids[right:selected_left],
                *target_ids[selected_left:selected_right],
                *target_ids[left:right],
                *target_ids[selected_right:],
            ]
        )
        source_span = (
            selected_left - (right - left),
            selected_right - (right - left),
        )
    elif disturbance in {"semantic_prefix", "upstream_edit"}:
        earlier = [
            row
            for row in target_turn["modules"]
            if int(row["token_span"][1]) <= target_span[0]
        ]
        if disturbance == "upstream_edit":
            scoped = [
                row
                for row in earlier
                if row["cache_scope"] in {"session", "workspace"}
                and row["module_type"] not in {"system", "role_instruction"}
            ]
            earlier = scoped or earlier
            changed = max(earlier, key=lambda row: int(row["token_span"][1]))
        else:
            changed = min(earlier, key=lambda row: int(row["position"]))
        span = tuple(map(int, changed["token_span"]))
        source_ids = _replace(
            target_ids,
            span,
            _replacement(other_ids, span[1] - span[0], f"{session_id}|{module_id}|{disturbance}"),
        )
    elif disturbance == "change_after":
        later = [
            row
            for row in target_turn["modules"]
            if int(row["token_span"][0]) >= target_span[1]
        ]
        if not later:
            raise ValueError("no later module for change-after")
        changed = max(later, key=lambda row: int(row["position"]))
        span = tuple(map(int, changed["token_span"]))
        target_ids = _replace(
            target_ids,
            span,
            _replacement(other_ids, span[1] - span[0], f"{session_id}|{module_id}|change-after"),
        )
    elif disturbance == "same_task":
        candidates = []
        for (candidate_session, turn_id), turn in turns.items():
            if candidate_session != session_id or turn_id >= int(target_turn["turn_id"]):
                continue
            try:
                old = _module(turn, module_id)
            except StopIteration:
                continue
            candidates.append((turn_id, turn, old))
        if not candidates:
            raise ValueError("module absent from earlier same-task turn")
        _, source_turn, old = max(candidates)
        source_ids = tuple(
            tokenizer.encode(source_turn["rendered_prompt"], add_special_tokens=False)
        )
        source_span = tuple(map(int, old["token_span"]))
    elif disturbance == "cross_task":
        module_ids = target_ids[slice(*target_span)]
        insert = min(
            len(other_ids) - 1,
            max(1, round(target_span[0] / len(target_ids) * len(other_ids))),
        )
        source_ids = (*other_ids[:insert], *module_ids, *other_ids[insert:])
        source_span = (insert, insert + len(module_ids))
    else:
        raise ValueError(f"unsupported disturbance: {disturbance}")
    pair = PromptPair(source_ids, target_ids, source_span, target_span, disturbance)
    pair.validate()
    return pair


def graph_distance(turn: Mapping[str, Any], module_id: str) -> int | None:
    parents = {
        str(row["module_id"]): tuple(map(str, row.get("dependencies", ())))
        for row in turn["modules"]
    }
    distances = {
        str(row["module_id"]): 0
        for row in turn["modules"]
        if row["module_type"] == "target"
    }
    queue = deque(distances)
    while queue:
        child = queue.popleft()
        for parent in parents.get(child, ()):
            if parent not in distances:
                distances[parent] = distances[child] + 1
                queue.append(parent)
    return distances.get(module_id)


def _prefix_changed(
    source_ids: Sequence[int],
    target_ids: Sequence[int],
    source_start: int,
    target_start: int,
) -> int:
    left = source_ids[:source_start]
    right = target_ids[:target_start]
    shared = min(len(left), len(right))
    return abs(len(left) - len(right)) + sum(
        left[index] != right[index] for index in range(shared)
    )


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
        ** (torch.arange(0, dim, 2, device=keys.device, dtype=torch.float32) / dim)
    )
    frequency = delta * inv
    cosine = torch.cat((frequency.cos(), frequency.cos()))
    sine = torch.cat((frequency.sin(), frequency.sin()))
    return (keys.float() * cosine + _rotate_half(keys.float()) * sine).to(keys.dtype)


def _cosine_deviation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = F.normalize(left.float().reshape(-1, left.shape[-1]), dim=-1)
    right = F.normalize(right.float().reshape(-1, right.shape[-1]), dim=-1)
    return float(1.0 - (left * right).sum(-1).mean().item())


def _kv_metrics(
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    source_span: tuple[int, int],
    target_span: tuple[int, int],
    theta: float,
) -> tuple[float, float]:
    source_start, source_end = source_span
    target_start, target_end = target_span
    delta = target_start - source_start
    key_deviations, value_deviations = [], []
    for (source_key, source_value), (target_key, target_value) in zip(
        source_cache, target_cache, strict=True
    ):
        key_deviations.append(
            _cosine_deviation(
                _rope_shift(
                    source_key[:, source_start:source_end], delta, theta
                ),
                target_key[:, target_start:target_end],
            )
        )
        value_deviations.append(
            _cosine_deviation(
                source_value[:, source_start:source_end],
                target_value[:, target_start:target_end],
            )
        )
    return float(np.mean(key_deviations)), float(np.mean(value_deviations))


def _js(left: torch.Tensor, right: torch.Tensor) -> float:
    left, right = F.log_softmax(left.float(), -1), F.log_softmax(right.float(), -1)
    midpoint = torch.logaddexp(left, right) - math.log(2)
    return max(
        0.0,
        float(
            0.5
            * (
                torch.sum(left.exp() * (left - midpoint))
                + torch.sum(right.exp() * (right - midpoint))
            )
        ),
    )


@torch.inference_mode()
def _dense(model: Any, ids: Sequence[int]) -> tuple[list[Any], torch.Tensor]:
    inputs = torch.tensor([ids], device="cuda", dtype=torch.long)
    output = model(input_ids=inputs, use_cache=True, return_dict=True, logits_to_keep=1)
    cache = _cpu_cache(output.past_key_values)
    logits = output.logits[0, -1].detach().float().cpu()
    del output, inputs
    torch.cuda.empty_cache()
    return cache, logits


@torch.inference_mode()
def _splice(
    model: Any,
    target_ids: Sequence[int],
    target_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    source_span: tuple[int, int],
    target_span: tuple[int, int],
    fraction: float,
    theta: float,
    chunk_size: int,
) -> torch.Tensor:
    from transformers.cache_utils import DynamicCache

    source_start, source_end = source_span
    target_start, target_end = target_span
    length = target_end - target_start
    head = min(length, max(0, round(length * fraction)))
    delta = target_start - source_start
    layers = []
    for (target_key, target_value), (source_key, source_value) in zip(
        target_cache, source_cache, strict=True
    ):
        key = target_key[:, : target_start + head]
        value = target_value[:, : target_start + head]
        if head < length:
            key = torch.cat(
                (key, _rope_shift(source_key[:, source_start + head : source_end], delta, theta)),
                dim=1,
            )
            value = torch.cat(
                (value, source_value[:, source_start + head : source_end]), dim=1
            )
        layers.append((key.unsqueeze(0).cuda(), value.unsqueeze(0).cuda()))
    cache = DynamicCache(layers, config=model.config)
    suffix = target_ids[target_end:]
    if not suffix:
        raise ValueError("selected module has no suffix")
    logits = None
    size = chunk_size or len(suffix)
    for offset in range(0, len(suffix), size):
        output = model(
            input_ids=torch.tensor([suffix[offset : offset + size]], device="cuda"),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        logits = output.logits[0, -1].float().cpu()
    assert logits is not None
    return logits


DesignKey = tuple[str, str, str, float]


def _design_key(row: Mapping[str, Any]) -> DesignKey:
    return (
        str(row["session_id"]),
        str(row["module_id"]),
        str(row["disturbance"]),
        float(row["recompute_fraction"]),
    )


def _completed(paths: Sequence[Path]) -> set[DesignKey]:
    completed: set[DesignKey] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row.get("status", "ok") != "ok":
                continue
            key = _design_key(row)
            if key in completed:
                raise ValueError(f"duplicate completed design key: {key}")
            completed.add(key)
    return completed


def pending_design_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = [
        row for row in read_jsonl(args.design) if row["cohort"] == args.cohort
    ]
    filters = {
        "disturbance": set(args.disturbances.split(","))
        if args.disturbances
        else None,
        "session_id": set(args.session_ids.split(",")) if args.session_ids else None,
        "module_id": set(args.module_ids.split(",")) if args.module_ids else None,
        "dose": {float(value) for value in args.doses.split(",")}
        if args.doses
        else None,
    }
    selected = [
        row
        for row in selected
        if (filters["disturbance"] is None or row["disturbance"] in filters["disturbance"])
        and (filters["session_id"] is None or str(row["session_id"]) in filters["session_id"])
        and (filters["module_id"] is None or str(row["module_id"]) in filters["module_id"])
        and (
            filters["dose"] is None
            or float(row["recompute_fraction"]) in filters["dose"]
        )
    ]
    completed_paths = [*args.resume_from]
    if args.resume and args.output.exists():
        completed_paths.append(args.output)
    completed = _completed(completed_paths)
    pending = [row for row in selected if _design_key(row) not in completed]
    groups = {
        (str(row["session_id"]), str(row["module_id"]), str(row["disturbance"]))
        for row in pending
    }
    summary = {
        "cohort": args.cohort,
        "selected_design_rows": len(selected),
        "completed_selected_rows": len(selected) - len(pending),
        "pending_rows": len(pending),
        "pending_groups": len(groups),
        "pending_sessions": len({key[0] for key in groups}),
        "pending_by_disturbance": dict(
            sorted(
                (
                    disturbance,
                    sum(row["disturbance"] == disturbance for row in pending),
                )
                for disturbance in {str(row["disturbance"]) for row in pending}
            )
        ),
    }
    return pending, summary


def measure(args: argparse.Namespace) -> dict[str, Any]:
    pending_rows, plan = pending_design_rows(args)
    if args.plan_output is not None:
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(
            json.dumps(
                {"summary": plan, "pending_design": pending_rows},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.plan_only:
        return {"passed": True, "plan_only": True, **plan}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU substitution is forbidden")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    turns = {
        (str(row["session_id"]), int(row["turn_id"])): row for row in replay
    }
    final_turns = {}
    for (session_id, turn_id), row in turns.items():
        if session_id not in final_turns or turn_id > int(final_turns[session_id]["turn_id"]):
            final_turns[session_id] = row
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pending_rows:
        grouped[
            (str(row["session_id"]), str(row["module_id"]), str(row["disturbance"]))
        ].append(row)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    ).eval()
    theta = float(getattr(model.config, "rope_theta", 1_000_000.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written, errors = 0, []
    active_session: str | None = None
    base_ids: tuple[int, ...] = ()
    base_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    base_logits: torch.Tensor | None = None
    for (session_id, module_id, disturbance), rows in sorted(grouped.items()):
        try:
            if session_id != active_session:
                if base_cache is not None:
                    del base_cache, base_logits
                    gc.collect()
                active_session = session_id
                base_ids = tuple(
                    tokenizer.encode(
                        final_turns[session_id]["rendered_prompt"],
                        add_special_tokens=False,
                    )
                )
                if len(base_ids) > args.max_context:
                    raise ValueError(
                        f"base context exceeds {args.max_context}: {len(base_ids)}"
                    )
                base_cache, base_logits = _dense(model, base_ids)
            pair = build_prompt_pair(
                tokenizer=tokenizer,
                turns=turns,
                final_turns=final_turns,
                session_id=session_id,
                module_id=module_id,
                disturbance=disturbance,
            )
            if max(len(pair.source_ids), len(pair.target_ids)) > args.max_context:
                raise ValueError(
                    f"context exceeds {args.max_context}: "
                    f"{len(pair.source_ids)}/{len(pair.target_ids)}"
                )
            if pair.source_ids == base_ids:
                source_cache, source_logits = base_cache, base_logits
                source_is_base = True
            else:
                source_cache, source_logits = _dense(model, pair.source_ids)
                source_is_base = False
            if pair.target_ids == pair.source_ids:
                target_cache, target_logits = source_cache, source_logits
                target_is_base = source_is_base
            elif pair.target_ids == base_ids:
                target_cache, target_logits = base_cache, base_logits
                target_is_base = True
            else:
                target_cache, target_logits = _dense(model, pair.target_ids)
                target_is_base = False
            assert source_cache is not None and source_logits is not None
            assert target_cache is not None and target_logits is not None
            key_deviation, value_deviation = _kv_metrics(
                source_cache,
                target_cache,
                pair.source_span,
                pair.target_span,
                theta,
            )
            teacher_js = _js(source_logits, target_logits)
            samples = []
            for _ in range(100):
                started = time.perf_counter()
                distance = graph_distance(final_turns[session_id], module_id)
                samples.append((time.perf_counter() - started) * 1000)
            for row in sorted(rows, key=lambda value: float(value["recompute_fraction"])):
                dose = float(row["recompute_fraction"])
                logits = (
                    target_logits
                    if dose == 1.0
                    else _splice(
                        model,
                        pair.target_ids,
                        target_cache,
                        source_cache,
                        pair.source_span,
                        pair.target_span,
                        dose,
                        theta,
                        args.splice_chunk_size,
                    )
                )
                observation = {
                    **row,
                    "status": "ok",
                    "token_count": pair.target_span[1] - pair.target_span[0],
                    "position_norm": pair.target_span[0] / len(pair.target_ids),
                    "rope_delta": pair.target_span[0] - pair.source_span[0],
                    "prefix_changed_tokens": _prefix_changed(
                        pair.source_ids,
                        pair.target_ids,
                        pair.source_span[0],
                        pair.target_span[0],
                    ),
                    "graph_distance": distance,
                    "k_deviation": key_deviation,
                    "v_deviation": value_deviation,
                    "attention_mass": None,
                    "attention_mass_measured": False,
                    "teacher_logit_js": teacher_js,
                    "teacher_top1_changed": int(source_logits.argmax())
                    != int(target_logits.argmax()),
                    "causal_splice_logit_js": _js(target_logits, logits),
                    "lookup_ms": float(np.quantile(samples, 0.95)),
                    "source_tokens": len(pair.source_ids),
                    "target_tokens": len(pair.target_ids),
                    "source_prompt_hash": _atlas_prompt_hash(pair.source_ids),
                    "target_prompt_hash": _atlas_prompt_hash(pair.target_ids),
                    "measurement_model": MODEL_ID,
                }
                with args.output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(observation, sort_keys=True) + "\n")
                written += 1
            if not source_is_base:
                del source_cache, source_logits
            if not target_is_base and pair.target_ids != pair.source_ids:
                del target_cache, target_logits
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as error:
            errors.append(
                {
                    "session_id": session_id,
                    "module_id": module_id,
                    "disturbance": disturbance,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if args.fail_fast:
                raise
    summary = {
        "passed": not errors,
        **plan,
        "groups": len(grouped),
        "written_rows": written,
        "errors": errors,
        "output": str(args.output),
        "model": MODEL_ID,
        "model_source": args.model,
        "dtype": "bfloat16",
        "attention_implementation": args.attn_implementation,
        "splice_suffix_chunk_size": args.splice_chunk_size,
        "attention_mass_status": "not_measured_no_proxy",
        "resume_from": [str(path) for path in args.resume_from],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--resume-from",
        type=Path,
        action="append",
        default=[],
        help="Read completed rows from an immutable artifact; may be repeated.",
    )
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cohort", choices=("development", "holdout"), default="development")
    parser.add_argument("--session-ids", default="")
    parser.add_argument("--module-ids", default="")
    parser.add_argument("--disturbances", default="")
    parser.add_argument("--doses", default="")
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--splice-chunk-size", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    args = parser.parse_args()
    print(json.dumps(measure(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
