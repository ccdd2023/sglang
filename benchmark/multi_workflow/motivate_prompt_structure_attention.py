#!/usr/bin/env python3
"""Map Dense attention onto the actual structure of coding-agent prompts.

The earlier sparsity probe measured only how concentrated tail-query attention
was.  This follow-up preserves the exact frozen prompts but labels every token
as task text, assistant action, path-relevant/disjoint read observation, other
tool result, compaction notice, or the final generation marker.  It reports
causal query-category -> key-category attention and one preregistered
chronological message-block heatmap.

This is a Dense-only mechanism diagnostic.  It is not accuracy, latency, an
online selector, or evidence that path relevance is independent of recency.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.multi_workflow.coding_reuse_policy import (
    is_successful_readonly_evidence,
    repository_paths,
)
from benchmark.multi_workflow.measure_sessiongraph_atlas import _layers, _rotate_half
from benchmark.multi_workflow.motivate_v50_coding_provenance import (
    MODEL,
    ROLLING_GROUPS,
    ROLLING_NOTICE,
    _render_message_literal,
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
    ROOT / "kvflow-artifacts/impactkv_prompt_structure_attention_20260806/frozen20"
)
PROBE_LAYERS = (0, 8, 17, 26, 35)
QUERY_TOKENS_PER_REGION = 16
MAX_QUERY_TOKENS_PER_CATEGORY = 64
EXAMPLE_CASE_ID = "mwaskom__seaborn-3069-path-q26"

CATEGORIES = (
    "system_instruction",
    "user_task",
    "compaction_notice",
    "assistant_action",
    "read_observation_path_disjoint",
    "read_observation_path_relevant",
    "other_tool_result",
    "generation_marker",
)


def _preview(value: str, limit: int = 92) -> str:
    compact = " ".join(value.replace("/testbed/", "").split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _assistant_summary(message: Mapping[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    if calls:
        wrapped = calls[0]
        call = wrapped.get("function", wrapped)
        arguments = call.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw_arguments": arguments}
        command = str(arguments.get("command") or arguments)
        return _preview(f"action: {command}")
    return _preview(f"reasoning: {message.get('content') or ''}")


def _append_region(
    regions: list[dict[str, Any]],
    *,
    start: int,
    end: int,
    category: str,
    label: str,
    paths: Sequence[str] = (),
    path_relevant: bool = False,
) -> None:
    if end <= start:
        return
    regions.append(
        {
            "region_id": f"r{len(regions):02d}",
            "start": start,
            "end": end,
            "tokens": end - start,
            "category": category,
            "label": label,
            "paths": list(paths),
            "path_relevant": path_relevant,
        }
    )


def _render_structured(
    tokenizer: Any, case: Mapping[str, Any]
) -> tuple[list[int], list[dict[str, Any]]]:
    trajectory = json.loads(Path(case["trajectory_path"]).read_text(encoding="utf-8"))
    messages = trajectory["messages"]
    base = messages[:2]
    groups = _turn_groups(messages[2:])
    target_completed = int(case["target_request_index"]) - 1
    active_groups = groups[:target_completed]
    target_ids, spans = _render_rolling(tokenizer, base, active_groups)
    if _token_ids_hash(target_ids) != case["target_prompt_hash"]:
        raise ValueError(f"{case['case_id']}: reconstructed target prompt changed")

    latest_paths = set(case["latest_paths"])
    regions: list[dict[str, Any]] = []
    cursor = 0
    for base_index, message in enumerate(base):
        literal_ids = tokenizer.encode(
            _render_message_literal(message), add_special_tokens=False
        )
        end = cursor + len(literal_ids)
        category = "system_instruction" if base_index == 0 else "user_task"
        label = "system: agent rules" if base_index == 0 else _preview(
            f"task: {message.get('content') or ''}"
        )
        _append_region(
            regions,
            start=cursor,
            end=end,
            category=category,
            label=label,
        )
        cursor = end

    dropped = max(0, len(active_groups) - ROLLING_GROUPS)
    if dropped:
        literal = _render_message_literal(
            {
                "role": "user",
                "content": ROLLING_NOTICE.format(dropped=dropped),
            }
        )
        end = cursor + len(tokenizer.encode(literal, add_special_tokens=False))
        _append_region(
            regions,
            start=cursor,
            end=end,
            category="compaction_notice",
            label=f"history compaction: {dropped} older turn groups omitted",
        )
        cursor = end

    for group_index in range(dropped, len(active_groups)):
        group = active_groups[group_index]
        readonly = is_successful_readonly_evidence(group)
        paths = sorted(repository_paths(group))
        relevant = bool(set(paths) & latest_paths)
        for message_index, message in enumerate(group):
            start, end = spans[(group_index, message_index)]
            if start != cursor:
                raise ValueError(
                    f"{case['case_id']}: non-contiguous region at {group_index}/{message_index}"
                )
            role = str(message.get("role") or "")
            if role == "assistant":
                category = "assistant_action"
                label = _assistant_summary(message)
            elif role == "tool" and readonly:
                category = (
                    "read_observation_path_relevant"
                    if relevant
                    else "read_observation_path_disjoint"
                )
                relation = "path relevant" if relevant else "path disjoint"
                path_text = ", ".join(paths[:2]) or "unparsed path"
                label = _preview(f"observation ({relation}): {path_text}")
            elif role == "tool":
                category = "other_tool_result"
                label = _preview(f"other tool result: {message.get('content') or ''}")
            else:
                category = "other_tool_result"
                label = _preview(f"{role}: {message.get('content') or ''}")
            _append_region(
                regions,
                start=start,
                end=end,
                category=category,
                label=label,
                paths=paths,
                path_relevant=relevant,
            )
            cursor = end

    marker_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    if cursor + len(marker_ids) != len(target_ids):
        raise ValueError(f"{case['case_id']}: final marker boundary changed")
    _append_region(
        regions,
        start=cursor,
        end=len(target_ids),
        category="generation_marker",
        label="next assistant action",
    )
    return target_ids, regions


def prepare(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    source = json.loads(SOURCE_DESIGN.read_text(encoding="utf-8"))
    cases = []
    for source_case in source["cases"]:
        target_ids, regions = _render_structured(tokenizer, source_case)
        cases.append(
            {
                "case_id": source_case["case_id"],
                "instance_id": source_case["instance_id"],
                "target_input_ids": target_ids,
                "target_prompt_hash": source_case["target_prompt_hash"],
                "regions": regions,
            }
        )
    output.mkdir(parents=True)
    design_path = output / "DESIGN.json"
    write_json(design_path, {"cases": cases, "model": str(MODEL)})
    capacity = {
        "cases": len(cases),
        "tasks": len({case["instance_id"] for case in cases}),
        "example_case_id": EXAMPLE_CASE_ID,
        "median_prompt_tokens": statistics.median(
            len(case["target_input_ids"]) for case in cases
        ),
        "median_structural_regions": statistics.median(
            len(case["regions"]) for case in cases
        ),
    }
    registration = {
        "status": "REGISTERED_BEFORE_GPU",
        "purpose": (
            "Map causal attention onto exact coding-agent prompt regions and "
            "show which structural/content blocks attend to which prior blocks"
        ),
        "source_design": str(SOURCE_DESIGN),
        "source_design_sha256": _sha256(SOURCE_DESIGN),
        "design_sha256": _sha256(design_path),
        "model": str(MODEL),
        "probe_layers_zero_based": list(PROBE_LAYERS),
        "query_tokens_per_region": QUERY_TOKENS_PER_REGION,
        "max_query_tokens_per_category": MAX_QUERY_TOKENS_PER_CATEGORY,
        "categories": list(CATEGORIES),
        "frozen_example_case_id": EXAMPLE_CASE_ID,
        "capacity": capacity,
        "frozen_reporting_questions": [
            "What key categories receive next-action attention mass?",
            "Do path-relevant observations receive higher attention density than disjoint observations?",
            "How do assistant actions and observations attend to earlier prompt regions?",
            "Is the dominant next-action key category stable across layers?",
        ],
        "interpretation_limits": [
            "Attention is causal: row query regions attend only to earlier key regions",
            "Category attention mass is not task accuracy or KV splice safety",
            "Path relevance remains confounded with recency in this frozen cohort",
            "The probe samples region-tail queries rather than materializing every token pair",
        ],
    }
    write_json(output / "CAPACITY.json", capacity)
    write_json(output / "REGISTRATION.json", registration)
    return registration


def _query_positions(
    regions: Sequence[Mapping[str, Any]], category: str
) -> list[int]:
    positions: list[int] = []
    for region in regions:
        if region["category"] != category:
            continue
        start = max(int(region["start"]), int(region["end"]) - QUERY_TOKENS_PER_REGION)
        positions.extend(range(start, int(region["end"])))
    return positions[-MAX_QUERY_TOKENS_PER_CATEGORY:]


def _attention_to_regions(
    *,
    model: Any,
    layer_index: int,
    hidden: torch.Tensor,
    key: torch.Tensor,
    query_positions: Sequence[int],
    regions: Sequence[Mapping[str, Any]],
    group_by: str,
) -> tuple[dict[str, float], dict[str, float]]:
    if not query_positions:
        return {}, {}
    attention = model.model.layers[layer_index].self_attn
    positions = torch.tensor(query_positions, device="cuda", dtype=torch.long)
    query_hidden = hidden[:, positions]
    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    groups = num_heads // num_kv_heads
    head_dim = int(model.config.hidden_size) // num_heads
    query = attention.q_proj(query_hidden).view(
        1, len(query_positions), num_heads, head_dim
    ).transpose(1, 2)
    cosine, sine = model.model.rotary_emb(query_hidden, positions.unsqueeze(0))
    query = query * cosine.unsqueeze(1) + _rotate_half(query) * sine.unsqueeze(1)
    expanded_key = key.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query.float(), expanded_key.float().transpose(-1, -2))
    scores = scores / math.sqrt(head_dim)
    key_positions = torch.arange(key.shape[-2], device="cuda")
    causal = key_positions.view(1, 1, 1, -1) > positions.view(1, 1, -1, 1)
    weights = torch.softmax(scores.masked_fill(causal, -torch.inf), dim=-1).mean(
        dim=(0, 1)
    )

    masses: dict[str, float] = {}
    densities: dict[str, float] = {}
    for region in regions:
        name = str(region[group_by])
        start, end = int(region["start"]), int(region["end"])
        per_query_mass = weights[:, start:end].sum(dim=1)
        visible = (
            positions[:, None]
            >= torch.arange(start, end, device="cuda")[None, :]
        ).sum(dim=1)
        visible_fraction = visible.float() / (positions.float() + 1.0)
        valid = visible_fraction > 0
        mass_value = float(per_query_mass.mean().item())
        density_value = float(
            (per_query_mass[valid] / visible_fraction[valid]).mean().item()
        ) if bool(valid.any()) else 0.0
        masses[name] = masses.get(name, 0.0) + mass_value
        # Density is defined over the union for categories; individual regions
        # are used only by the chronological example and therefore unique.
        if group_by == "region_id":
            densities[name] = density_value

    if group_by == "category":
        for category in CATEGORIES:
            members = [r for r in regions if r["category"] == category]
            if not members:
                densities[category] = 0.0
                continue
            mask = torch.zeros(key.shape[-2], device="cuda", dtype=torch.bool)
            for region in members:
                mask[int(region["start"]):int(region["end"])] = True
            per_query_mass = weights[:, mask].sum(dim=1)
            visible_counts = torch.stack(
                [mask[: int(position) + 1].sum() for position in query_positions]
            ).float()
            visible_fraction = visible_counts / (positions.float() + 1.0)
            valid = visible_fraction > 0
            densities[category] = float(
                (per_query_mass[valid] / visible_fraction[valid]).mean().item()
            ) if bool(valid.any()) else 0.0
    del query_hidden, query, expanded_key, scores, causal, weights
    return masses, densities


@torch.inference_mode()
def _profile_case(model: Any, case: Mapping[str, Any]) -> dict[str, Any]:
    target_ids = case["target_input_ids"]
    regions = case["regions"]
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in PROBE_LAYERS:
        def capture(
            _module: Any,
            args: tuple[Any, ...],
            index: int = layer_index,
        ) -> None:
            captured[index] = args[0].detach()

        handles.append(model.model.layers[layer_index].register_forward_pre_hook(capture))
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
    cache_layers = _layers(output.past_key_values)
    layer_rows = []
    for layer_index in PROBE_LAYERS:
        query_rows = {}
        for category in CATEGORIES:
            positions = _query_positions(regions, category)
            if not positions:
                continue
            masses, densities = _attention_to_regions(
                model=model,
                layer_index=layer_index,
                hidden=captured[layer_index],
                key=cache_layers[layer_index][0],
                query_positions=positions,
                regions=regions,
                group_by="category",
            )
            query_rows[category] = {
                "query_tokens": len(positions),
                "attention_mass": masses,
                "attention_density_vs_uniform": densities,
            }
        layer_rows.append({"layer": layer_index, "query_categories": query_rows})

    example = None
    if case["case_id"] == EXAMPLE_CASE_ID:
        example_rows = []
        layer_index = 17
        for region in regions:
            start = max(int(region["start"]), int(region["end"]) - QUERY_TOKENS_PER_REGION)
            positions = list(range(start, int(region["end"])))
            masses, _ = _attention_to_regions(
                model=model,
                layer_index=layer_index,
                hidden=captured[layer_index],
                key=cache_layers[layer_index][0],
                query_positions=positions,
                regions=regions,
                group_by="region_id",
            )
            example_rows.append(
                {
                    "query_region_id": region["region_id"],
                    "attention_mass": masses,
                }
            )
        example = {
            "layer": layer_index,
            "regions": regions,
            "query_to_key_attention": example_rows,
        }

    row = {
        "status": "ok",
        "case_id": case["case_id"],
        "instance_id": case["instance_id"],
        "prompt_tokens": len(target_ids),
        "regions": regions,
        "layers": layer_rows,
        "chronological_example": example,
    }
    del output, inputs, captured, cache_layers
    gc.collect()
    torch.cuda.empty_cache()
    return row


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
                        "prompt_tokens": row["prompt_tokens"],
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
    mass_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    density_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    final_top_categories = []
    final_top_stabilities = []
    per_case = []
    for row in rows:
        case_top = []
        final_observation_mass = []
        relevant_density = []
        disjoint_density = []
        for layer in row["layers"]:
            for query_category, values in layer["query_categories"].items():
                for key_category in CATEGORIES:
                    mass_values[(query_category, key_category)].append(
                        float(values["attention_mass"].get(key_category, 0.0))
                    )
                    density_values[(query_category, key_category)].append(
                        float(
                            values["attention_density_vs_uniform"].get(
                                key_category, 0.0
                            )
                        )
                    )
            final = layer["query_categories"].get("generation_marker")
            if final:
                masses = final["attention_mass"]
                top = max(masses, key=masses.get)
                case_top.append(top)
                final_top_categories.append(top)
                final_observation_mass.append(
                    masses.get("read_observation_path_relevant", 0.0)
                    + masses.get("read_observation_path_disjoint", 0.0)
                )
                relevant_density.append(
                    final["attention_density_vs_uniform"].get(
                        "read_observation_path_relevant", 0.0
                    )
                )
                disjoint_density.append(
                    final["attention_density_vs_uniform"].get(
                        "read_observation_path_disjoint", 0.0
                    )
                )
        modal = Counter(case_top).most_common(1)[0][1] / len(case_top)
        final_top_stabilities.append(modal)
        density_ratios = [
            relevant / disjoint
            for relevant, disjoint in zip(
                relevant_density, disjoint_density, strict=True
            )
            if disjoint > 0
        ]
        per_case.append(
            {
                "case_id": row["case_id"],
                "instance_id": row["instance_id"],
                "median_final_query_observation_mass": statistics.median(
                    final_observation_mass
                ),
                "median_relevant_vs_disjoint_density_ratio": statistics.median(
                    density_ratios
                ) if density_ratios else None,
                "final_top_category_stability": modal,
            }
        )

    matrices = {
        query: {
            key: statistics.median(values)
            for (query_name, key), values in mass_values.items()
            if query_name == query
        }
        for query in CATEGORIES
    }
    mean_matrices = {
        query: {
            key: statistics.fmean(values)
            for (query_name, key), values in mass_values.items()
            if query_name == query
        }
        for query in CATEGORIES
    }
    density_matrices = {
        query: {
            key: statistics.median(values)
            for (query_name, key), values in density_values.items()
            if query_name == query
        }
        for query in CATEGORIES
    }
    example = next(
        row["chronological_example"]
        for row in rows
        if row["case_id"] == EXAMPLE_CASE_ID
    )
    aggregate = {
        "cases": len(rows),
        "tasks": len({row["instance_id"] for row in rows}),
        "median_final_query_observation_mass": statistics.median(
            row["median_final_query_observation_mass"] for row in per_case
        ),
        "median_relevant_vs_disjoint_density_ratio": statistics.median(
            row["median_relevant_vs_disjoint_density_ratio"]
            for row in per_case
            if row["median_relevant_vs_disjoint_density_ratio"] is not None
        ),
        "median_final_top_category_stability": statistics.median(
            final_top_stabilities
        ),
        "final_top_category_layer_case_points": len(final_top_categories),
        "final_top_category_counts": dict(Counter(final_top_categories)),
    }
    result = {
        "status": "COMPLETE",
        "aggregate": aggregate,
        "mean_query_to_key_attention_mass": mean_matrices,
        "median_query_to_key_attention_mass": matrices,
        "median_query_to_key_attention_density_vs_uniform": density_matrices,
        "case_summaries": per_case,
        "chronological_example": example,
        "scope": (
            "Dense-only causal attention mapped to exact prompt structure; "
            "not accuracy, latency, selector quality, or splice safety"
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
