#!/usr/bin/env python3
"""Confirmatory Attention study for natural coding-prompt modules.

Stages are sealed. ``prepare`` opens only fresh Dense trajectories and freezes
natural-module, boundary, and relation controls. ``measure`` then opens 3B
Dense Attention. ``summarize`` evaluates the preregistered task-clustered gates.
No K/V splice or runtime outcome is opened by this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.multi_workflow import motivate_module_conditioned_attention_kv as old_attention
from benchmark.multi_workflow import motivate_v50_coding_provenance as m50
from benchmark.multi_workflow.audit_natural_prompt_module_capacity import (
    boundary_control,
    recency_control,
)
from benchmark.multi_workflow.natural_prompt_modules import (
    render_natural_prompt_modules,
)


ROOT = Path("/home/gfy/CodeMAS_Project")
CAMPAIGN = ROOT / "kvflow-artifacts/impactkv_natural_module_attention_20260808"
TRAJECTORY_ROOT = CAMPAIGN
COHORT_REGISTRATION = CAMPAIGN / "COHORT_REGISTRATION.json"
DEFAULT_OUTPUT = CAMPAIGN / "attention_initial20"
MODEL = old_attention.MODEL
PROBE_LAYERS = (0, 8, 17, 26, 35)
PRIMARY_TYPES = ("repository_code", "assistant_interpretation")
MIN_LENGTH = 32
MAX_LENGTH = 4096
MAX_CASES = 80
MAX_CASES_PER_TASK = 4
MIN_CASES = 64
MIN_TASKS = 16
MIN_MODULES = 48
MIN_MODULE_TASKS = 8
MIN_MATCHED_PAIRS = 32
MIN_MATCHED_TASKS = 8
PREPARE_SALT = "natural-module-attention-cases-20260808-v1"
BOOTSTRAP_SEED = 2026080804
BOOTSTRAP_SAMPLES = 2000


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable(value: str) -> str:
    return hashlib.sha256(f"{PREPARE_SALT}:{value}".encode()).hexdigest()


def _trajectory_paths(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(root.glob("**/*.traj.json")):
        value = _read_json(path)
        instance_id = str(value.get("instance_id") or "")
        if instance_id and instance_id not in paths:
            paths[instance_id] = path
    return paths


def _module_matches(
    source: Sequence[Mapping[str, Any]], target: Sequence[Mapping[str, Any]]
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    target_map = {
        (row["parent_interaction_id"], row["module_type"], row["content_hash"]): row
        for row in target
    }
    return [
        (row, target_map[key])
        for row in source
        if (
            key := (row["parent_interaction_id"], row["module_type"], row["content_hash"])
        )
        in target_map
    ]


def _source_span_for_target_control(
    *,
    source_modules: Sequence[Mapping[str, Any]],
    target_modules: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    target_span: tuple[int, int],
) -> tuple[int, int] | None:
    parent = str(candidate["parent_interaction_id"])
    source_parent = [row for row in source_modules if row["parent_interaction_id"] == parent]
    target_parent = [row for row in target_modules if row["parent_interaction_id"] == parent]
    if not source_parent or not target_parent:
        return None
    delta = int(source_parent[0]["token_start"]) - int(target_parent[0]["token_start"])
    return target_span[0] + delta, target_span[1] + delta


def _consumer(
    candidate: Mapping[str, Any],
    modules: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    by_id = {str(row["module_id"]): row for row in modules}
    choices: list[tuple[tuple[Any, ...], Mapping[str, Any], Mapping[str, Any]]] = []
    for relation in relations:
        if relation["key_module_id"] != candidate["module_id"]:
            continue
        query = by_id[str(relation["query_module_id"])]
        if query["module_type"] not in ("assistant_interpretation", "tool_command"):
            continue
        signals = sum(
            bool(relation[name])
            for name in ("exact_path", "shared_symbol", "interpretation_grounding")
        )
        if not signals:
            continue
        priority = (
            0 if query["module_type"] == "assistant_interpretation" else 1,
            int(relation["interaction_distance"]),
            -signals,
            str(query["module_id"]),
        )
        choices.append((priority, query, relation))
    if not choices:
        return None
    _, query, relation = min(choices, key=lambda row: row[0])
    return query, relation


def _candidate_rows(
    *,
    source_ids: Sequence[int],
    target_ids: Sequence[int],
    source_modules: Sequence[Mapping[str, Any]],
    target_modules: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, target in _module_matches(source_modules, target_modules):
        module_type = str(target["module_type"])
        length = int(target["natural_length"])
        if module_type not in PRIMARY_TYPES or not MIN_LENGTH <= length <= MAX_LENGTH:
            continue
        if target.get("invalidating_event") is not None:
            continue
        source_start = int(source["token_start"])
        target_start = int(target["token_start"])
        if list(source_ids[source_start : source_start + length]) != list(
            target_ids[target_start : target_start + length]
        ):
            raise AssertionError("matched natural module changed across prompts")
        boundary = boundary_control(target, target_modules)
        if boundary is None:
            continue
        source_boundary = _source_span_for_target_control(
            source_modules=source_modules,
            target_modules=target_modules,
            candidate=target,
            target_span=boundary,
        )
        if source_boundary is None or list(source_ids[source_boundary[0] : source_boundary[1]]) != list(
            target_ids[boundary[0] : boundary[1]]
        ):
            continue
        consumer = _consumer(target, target_modules, relations)
        recent = recency_control(target, target_modules, len(target_ids))
        relation_payload = None
        if consumer is not None and recent is not None:
            query, relation = consumer
            control_length = length
            control_start = int(recent["token_end"]) - control_length
            source_recent = next(
                (
                    row
                    for row in source_modules
                    if row["parent_interaction_id"] == recent["parent_interaction_id"]
                    and row["module_type"] == recent["module_type"]
                    and row["content_hash"] == recent["content_hash"]
                ),
                None,
            )
            source_control_start = (
                int(source_recent["token_end"]) - control_length
                if source_recent is not None
                else -1
            )
            if (
                control_start >= int(recent["token_start"])
                and int(recent["token_end"]) <= int(query["token_start"])
                and source_recent is not None
                and source_control_start >= int(source_recent["token_start"])
                and list(source_ids[source_control_start : source_control_start + control_length])
                == list(target_ids[control_start : control_start + control_length])
            ):
                relation_payload = {
                    "query_module_id": query["module_id"],
                    "query_module_type": query["module_type"],
                    "query_start": query["token_start"],
                    "query_end": query["token_end"],
                    "control_module_id": recent["module_id"],
                    "control_start": control_start,
                    "control_end": control_start + control_length,
                    "source_control_start": source_control_start,
                    "source_control_end": source_control_start + control_length,
                    "control_interaction_distance": (
                        int(query["source_request_index"])
                        - int(recent["source_request_index"])
                    ),
                    "relation": dict(relation),
                }
        rows.append(
            {
                "candidate_id": str(target["module_id"]),
                "module_type": module_type,
                "natural_length": length,
                "source_start": source_start,
                "target_start": target_start,
                "target_end": target_start + length,
                "source_boundary_start": source_boundary[0],
                "target_boundary_start": boundary[0],
                "target_boundary_end": boundary[1],
                "source_request_index": target["source_request_index"],
                "repository_epoch": target["repository_epoch"],
                "paths": target["paths"],
                "symbols": target["symbols"],
                "relation_control": relation_payload,
            }
        )
    return rows


def _balanced_cases(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["instance_id"])].append(row)
    for task in by_task:
        by_task[task].sort(
            key=lambda row: (
                -sum(
                    candidate["relation_control"] is not None
                    for candidate in row["candidates"]
                ),
                -len({candidate["module_type"] for candidate in row["candidates"]}),
                _stable(str(row["case_id"])),
            )
        )
    selected: list[dict[str, Any]] = []
    for round_index in range(MAX_CASES_PER_TASK):
        for task in sorted(by_task, key=_stable):
            if round_index < len(by_task[task]) and len(selected) < MAX_CASES:
                selected.append(by_task[task][round_index])
    return selected


def prepare(output: Path, trajectory_root: Path) -> dict[str, Any]:
    if output.exists():
        return _read_json(output / "REGISTRATION.json")
    trajectories = _trajectory_paths(trajectory_root)
    if len(trajectories) < MIN_TASKS:
        raise RuntimeError(f"only {len(trajectories)} fresh Dense trajectories")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    rows: list[dict[str, Any]] = []
    for instance_id, path in trajectories.items():
        trajectory = _read_json(path)
        messages = trajectory["messages"]
        base = messages[:2]
        groups = m50._turn_groups(messages[2:])
        for target_completed in range(7, len(groups)):
            source_ids, source_modules, _ = render_natural_prompt_modules(
                tokenizer, base, groups[: target_completed - 1]
            )
            target_ids, target_modules, relations = render_natural_prompt_modules(
                tokenizer, base, groups[:target_completed]
            )
            if len(target_ids) > old_attention.MAX_PROMPT_TOKENS:
                continue
            candidates = _candidate_rows(
                source_ids=source_ids,
                target_ids=target_ids,
                source_modules=source_modules,
                target_modules=target_modules,
                relations=relations,
            )
            if not candidates:
                continue
            # Freeze at most one candidate of each type per request, selected
            # by a salted identifier and never by Attention/KV labels.
            chosen = []
            for module_type in PRIMARY_TYPES:
                typed = [row for row in candidates if row["module_type"] == module_type]
                if typed:
                    chosen.append(
                        min(
                            typed,
                            key=lambda row: (
                                row["relation_control"] is None,
                                _stable(
                                    f"{instance_id}:{target_completed}:{row['candidate_id']}"
                                ),
                            ),
                        )
                    )
            rows.append(
                {
                    "case_id": f"{instance_id}-q{target_completed + 1}",
                    "instance_id": instance_id,
                    "request_index": target_completed + 1,
                    "source_input_ids": source_ids,
                    "target_input_ids": target_ids,
                    "target_modules": target_modules,
                    "candidates": chosen,
                }
            )
    selected = _balanced_cases(rows)
    counts = Counter(
        candidate["module_type"] for row in selected for candidate in row["candidates"]
    )
    tasks_by_type = {
        module_type: {
            row["instance_id"]
            for row in selected
            if any(candidate["module_type"] == module_type for candidate in row["candidates"])
        }
        for module_type in PRIMARY_TYPES
    }
    relation_counts = Counter(
        candidate["module_type"]
        for row in selected
        for candidate in row["candidates"]
        if candidate["relation_control"] is not None
    )
    relation_tasks = {
        module_type: {
            row["instance_id"]
            for row in selected
            if any(
                candidate["module_type"] == module_type
                and candidate["relation_control"] is not None
                for candidate in row["candidates"]
            )
        }
        for module_type in PRIMARY_TYPES
    }
    gates = {
        "tasks_at_least_16": len({row["instance_id"] for row in selected}) >= MIN_TASKS,
        "target_prompts_at_least_64": len(selected) >= MIN_CASES,
    }
    for module_type in PRIMARY_TYPES:
        gates[f"{module_type}_at_least_48_modules_8_tasks"] = (
            counts[module_type] >= MIN_MODULES and len(tasks_by_type[module_type]) >= MIN_MODULE_TASKS
        )
        gates[f"{module_type}_at_least_32_relation_controls_8_tasks"] = (
            relation_counts[module_type] >= MIN_MATCHED_PAIRS
            and len(relation_tasks[module_type]) >= MIN_MATCHED_TASKS
        )
    output.mkdir(parents=True)
    design = {
        "analysis_model": str(MODEL),
        "probe_layers_zero_based": list(PROBE_LAYERS),
        "trajectory_root": str(trajectory_root),
        "cases": selected,
    }
    _write_json(output / "DESIGN.json", design)
    capacity = {
        "fresh_trajectories": len(trajectories),
        "eligible_cases": len(rows),
        "selected_cases": len(selected),
        "selected_tasks": len({row["instance_id"] for row in selected}),
        "modules": dict(counts),
        "module_tasks": {key: len(value) for key, value in tasks_by_type.items()},
        "relation_controls": dict(relation_counts),
        "relation_control_tasks": {key: len(value) for key, value in relation_tasks.items()},
        "gates": gates,
    }
    _write_json(output / "CAPACITY.json", capacity)
    registration = {
        "status": "REGISTERED_BEFORE_ATTENTION" if all(gates.values()) else "CAPACITY_SHORTFALL_BEFORE_ATTENTION",
        "design_sha256": _sha256(output / "DESIGN.json"),
        "cohort_registration": str(COHORT_REGISTRATION),
        "cohort_registration_sha256": _sha256(COHORT_REGISTRATION),
        "outcome_used_for_case_or_candidate_selection": False,
        "analysis_bounds": {
            "natural_tokens": [MIN_LENGTH, MAX_LENGTH],
            "maximum_cases": MAX_CASES,
            "maximum_cases_per_task": MAX_CASES_PER_TASK,
            "probe_layers": list(PROBE_LAYERS),
        },
        "confirmatory_gates": {
            "intra_module_residual_ratio_median": 1.20,
            "intra_module_paired_direction": 0.65,
            "task_bootstrap_ratio_q025": 1.0,
            "enhanced_spearman_improvement": 0.10,
            "task_bootstrap_spearman_improvement_q025": 0.0,
        },
        "sealed_next_stage": "physical splice remains closed until every Attention gate passes",
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
            "prefetch": False,
        },
    }
    _write_json(output / "REGISTRATION.json", registration)
    return registration


def _point(
    *,
    case: Mapping[str, Any],
    candidate: Mapping[str, Any],
    layer: int,
    kind: str,
    key_start: int,
    key_end: int,
    query_start: int,
    query_end: int,
    attention_mass: float,
    relation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    key_tokens = key_end - key_start
    query_tokens = query_end - query_start
    return {
        "case_id": case["case_id"],
        "instance_id": case["instance_id"],
        "candidate_id": candidate["candidate_id"],
        "module_type": candidate["module_type"],
        "query_module_type": (relation or {}).get(
            "query_module_type", candidate["module_type"]
        ),
        "kind": kind,
        "layer": layer,
        "key_start": key_start,
        "key_end": key_end,
        "query_start": query_start,
        "query_end": query_end,
        "key_tokens": key_tokens,
        "query_tokens": query_tokens,
        "token_distance": max(0, query_start - key_end),
        "key_position": key_start / max(len(case["target_input_ids"]), 1),
        "query_position": query_start / max(len(case["target_input_ids"]), 1),
        "interaction_distance": (
            int((relation or {}).get("interaction_distance", 0))
        ),
        "attention_mass": attention_mass,
        "attention_density": attention_mass / max(key_tokens, 1),
        "exact_path": bool((relation or {}).get("exact_path", False)),
        "same_directory": bool((relation or {}).get("same_directory", False)),
        "shared_symbol": bool((relation or {}).get("shared_symbol", False)),
        "interpretation_grounding": bool(
            (relation or {}).get("interpretation_grounding", False)
        ),
    }


@torch.inference_mode()
def measure(output: Path) -> dict[str, Any]:
    registration = _read_json(output / "REGISTRATION.json")
    design_path = output / "DESIGN.json"
    if registration["status"] != "REGISTERED_BEFORE_ATTENTION":
        raise RuntimeError("fresh capacity gates did not pass")
    if registration["design_sha256"] != _sha256(design_path):
        raise ValueError("Attention design changed after registration")
    cases = _read_json(design_path)["cases"]
    observations = output / "ATTENTION.jsonl"
    completed: set[str] = set()
    if observations.exists():
        for line in observations.read_text().splitlines():
            if line.strip():
                completed.add(str(json.loads(line)["case_id"]))
    pending = [case for case in cases if str(case["case_id"]) not in completed]
    if not pending:
        return {"status": "COMPLETE", "cases": len(cases), "new_cases": 0}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    errors = []
    written = 0
    for index, case in enumerate(pending, 1):
        try:
            target_cache, _, hidden = old_attention._dense_full(
                model, case["target_input_ids"], True
            )
            points: list[dict[str, Any]] = []
            for layer in PROBE_LAYERS:
                key = target_cache[layer][0].unsqueeze(0).to("cuda")
                for candidate in case["candidates"]:
                    natural = (
                        int(candidate["target_start"]),
                        int(candidate["target_end"]),
                    )
                    boundary = (
                        int(candidate["target_boundary_start"]),
                        int(candidate["target_boundary_end"]),
                    )
                    for kind, span in (("intra_natural", natural), ("intra_boundary", boundary)):
                        mass = old_attention._attention_mass_for_block(
                            model=model,
                            layer_index=layer,
                            hidden=hidden[layer],
                            target_key=key,
                            block={"start": span[0], "end": span[1]},
                            candidate_spans={kind: span},
                        )[kind]
                        points.append(
                            _point(
                                case=case,
                                candidate=candidate,
                                layer=layer,
                                kind=kind,
                                key_start=span[0],
                                key_end=span[1],
                                query_start=span[0],
                                query_end=span[1],
                                attention_mass=mass,
                            )
                        )
                    relation_control = candidate.get("relation_control")
                    if relation_control is not None:
                        query_span = (
                            int(relation_control["query_start"]),
                            int(relation_control["query_end"]),
                        )
                        control_span = (
                            int(relation_control["control_start"]),
                            int(relation_control["control_end"]),
                        )
                        masses = old_attention._attention_mass_for_block(
                            model=model,
                            layer_index=layer,
                            hidden=hidden[layer],
                            target_key=key,
                            block={"start": query_span[0], "end": query_span[1]},
                            candidate_spans={"relation": natural, "relation_control": control_span},
                        )
                        for kind, span in (("relation", natural), ("relation_control", control_span)):
                            points.append(
                                _point(
                                    case=case,
                                    candidate=candidate,
                                    layer=layer,
                                    kind=kind,
                                    key_start=span[0],
                                    key_end=span[1],
                                    query_start=query_span[0],
                                    query_end=query_span[1],
                                    attention_mass=masses[kind],
                                    relation=(
                                        {
                                            **relation_control["relation"],
                                            "query_module_type": relation_control[
                                                "query_module_type"
                                            ],
                                        }
                                        if kind == "relation"
                                        else {
                                            "interaction_distance": relation_control[
                                                "control_interaction_distance"
                                            ],
                                            "query_module_type": relation_control[
                                                "query_module_type"
                                            ],
                                        }
                                    ),
                                )
                            )
                del key
            with observations.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "status": "ok",
                            "case_id": case["case_id"],
                            "instance_id": case["instance_id"],
                            "points": points,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            written += 1
            print(json.dumps({"case": index, "pending": len(pending), "case_id": case["case_id"]}), flush=True)
            del target_cache, hidden, points
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as error:
            errors.append({"case_id": case["case_id"], "error": f"{type(error).__name__}: {error}"})
            break
    del model
    gc.collect()
    torch.cuda.empty_cache()
    status = {
        "status": "COMPLETE" if not errors and written == len(pending) else "PARTIAL",
        "cases": len(cases),
        "previously_completed": len(completed),
        "new_cases": written,
        "errors": errors,
    }
    _write_json(output / "ATTENTION_STATUS.json", status)
    return status


def _rank(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    sorted_values = np.asarray(values)[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 3:
        return float("nan")
    a, b = _rank(left), _rank(right)
    if float(np.std(a)) == 0 or float(np.std(b)) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _features(row: Mapping[str, Any], enhanced: bool) -> list[float]:
    layer_one_hot = [float(int(row["layer"]) == value) for value in PROBE_LAYERS[1:]]
    values = [
        1.0,
        math.log1p(float(row["key_tokens"])),
        math.log1p(float(row["query_tokens"])),
        math.log1p(float(row["token_distance"])),
        float(row["key_position"]),
        float(row["query_position"]),
        float(row["interaction_distance"]),
        *layer_one_hot,
    ]
    if enhanced:
        values.extend(
            [
                float(row["module_type"] == "repository_code"),
                float(row["query_module_type"] == "assistant_interpretation"),
                float(row["query_module_type"] == "tool_command"),
                float(row["kind"] == "intra_natural"),
                float(row["kind"] == "relation"),
                float(row["exact_path"]),
                float(row["same_directory"]),
                float(row["shared_symbol"]),
                float(row["interpretation_grounding"]),
            ]
        )
    return values


def crossfit_predictions(rows: Sequence[Mapping[str, Any]], enhanced: bool) -> list[float]:
    predictions = [float("nan")] * len(rows)
    tasks = sorted({str(row["instance_id"]) for row in rows})
    for task in tasks:
        train = [row for row in rows if str(row["instance_id"]) != task]
        held_indices = [index for index, row in enumerate(rows) if str(row["instance_id"]) == task]
        x = np.asarray([_features(row, enhanced) for row in train], dtype=float)
        y = np.asarray([math.log(max(float(row["attention_density"]), 1e-12)) for row in train])
        ridge = np.eye(x.shape[1]) * 1e-6
        ridge[0, 0] = 0
        beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
        for index in held_indices:
            predictions[index] = float(np.asarray(_features(rows[index], enhanced)) @ beta)
    return predictions


def _task_bootstrap(
    task_values: Mapping[str, Sequence[float]], statistic: Any
) -> tuple[float, float, float]:
    tasks = sorted(task_values)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_SAMPLES):
        selected = rng.choice(tasks, size=len(tasks), replace=True)
        values = [value for task in selected for value in task_values[str(task)]]
        draws.append(float(statistic(values)))
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.5, 0.975]))


def summarize(output: Path) -> dict[str, Any]:
    status = _read_json(output / "ATTENTION_STATUS.json")
    if status["status"] != "COMPLETE":
        raise RuntimeError("Attention measurements are incomplete")
    rows = [
        point
        for line in (output / "ATTENTION.jsonl").read_text().splitlines()
        if line.strip()
        for point in json.loads(line)["points"]
    ]
    baseline_prediction = crossfit_predictions(rows, False)
    enhanced_prediction = crossfit_predictions(rows, True)
    observed = [math.log(max(float(row["attention_density"]), 1e-12)) for row in rows]
    baseline_rho = spearman(baseline_prediction, observed)
    enhanced_rho = spearman(enhanced_prediction, observed)
    improvement = enhanced_rho - baseline_rho

    case_layer = {
        (str(row["case_id"]), str(row["candidate_id"]), int(row["layer"]), str(row["kind"])): row
        for row in rows
    }
    row_indices = {
        (str(row["case_id"]), str(row["candidate_id"]), int(row["layer"]), str(row["kind"])): index
        for index, row in enumerate(rows)
    }
    type_results: dict[str, Any] = {}
    all_gates: dict[str, bool] = {}
    for module_type in PRIMARY_TYPES:
        residual_ratios_by_case: dict[tuple[str, str], list[float]] = defaultdict(list)
        raw_ratios_by_case: dict[tuple[str, str], list[float]] = defaultdict(list)
        task_ratios: dict[str, list[float]] = defaultdict(list)
        for key, natural in case_layer.items():
            case_id, candidate_id, layer, kind = key
            if kind != "intra_natural" or natural["module_type"] != module_type:
                continue
            boundary = case_layer[(case_id, candidate_id, layer, "intra_boundary")]
            raw_ratio = float(natural["attention_density"]) / max(
                float(boundary["attention_density"]), 1e-12
            )
            natural_index = row_indices[(case_id, candidate_id, layer, "intra_natural")]
            boundary_index = row_indices[(case_id, candidate_id, layer, "intra_boundary")]
            residual_ratio = math.exp(
                (observed[natural_index] - baseline_prediction[natural_index])
                - (observed[boundary_index] - baseline_prediction[boundary_index])
            )
            raw_ratios_by_case[(case_id, candidate_id)].append(raw_ratio)
            residual_ratios_by_case[(case_id, candidate_id)].append(residual_ratio)
        raw_case_ratios = {
            key: statistics.geometric_mean(max(item, 1e-12) for item in values)
            for key, values in raw_ratios_by_case.items()
        }
        for (case_id, candidate_id), values in residual_ratios_by_case.items():
            natural = case_layer[(case_id, candidate_id, PROBE_LAYERS[0], "intra_natural")]
            task_ratios[str(natural["instance_id"])].append(
                statistics.geometric_mean(max(item, 1e-12) for item in values)
            )
        flattened = [value for values in task_ratios.values() for value in values]
        bootstrap = _task_bootstrap(task_ratios, statistics.median)
        result = {
            "paired_cases": len(flattened),
            "tasks": len(task_ratios),
            "raw_natural_to_boundary_median_density_ratio": statistics.median(
                raw_case_ratios.values()
            ),
            "median_baseline_adjusted_natural_to_boundary_density_ratio": statistics.median(flattened),
            "raw_natural_to_boundary_paired_direction": (
                sum(value > 1 for value in raw_case_ratios.values())
                / len(raw_case_ratios)
            ),
            "task_bootstrap_adjusted_ratio_q025_q50_q975": bootstrap,
        }
        type_results[module_type] = result
        all_gates[f"{module_type}_residual_ratio_at_least_1_20"] = (
            result["median_baseline_adjusted_natural_to_boundary_density_ratio"] >= 1.20
        )
        all_gates[f"{module_type}_paired_direction_at_least_65pct"] = (
            result["raw_natural_to_boundary_paired_direction"] >= 0.65
        )
        all_gates[f"{module_type}_bootstrap_q025_above_1"] = bootstrap[0] > 1.0

        relation_by_case: dict[tuple[str, str], list[float]] = defaultdict(list)
        relation_tasks: dict[str, list[float]] = defaultdict(list)
        for key, relation in case_layer.items():
            case_id, candidate_id, layer, kind = key
            if kind != "relation" or relation["module_type"] != module_type:
                continue
            control = case_layer[(case_id, candidate_id, layer, "relation_control")]
            relation_by_case[(case_id, candidate_id)].append(
                float(relation["attention_density"])
                / max(float(control["attention_density"]), 1e-12)
            )
        for (case_id, candidate_id), values in relation_by_case.items():
            relation = case_layer[(case_id, candidate_id, PROBE_LAYERS[0], "relation")]
            relation_tasks[str(relation["instance_id"])].append(
                statistics.geometric_mean(max(value, 1e-12) for value in values)
            )
        relation_values = [
            value for values in relation_tasks.values() for value in values
        ]
        type_results[module_type]["source_to_consumer_vs_recency_control"] = {
            "paired_cases": len(relation_values),
            "tasks": len(relation_tasks),
            "median_density_ratio": statistics.median(relation_values),
            "paired_direction": (
                sum(value > 1 for value in relation_values) / len(relation_values)
            ),
            "task_bootstrap_ratio_q025_q50_q975": _task_bootstrap(
                relation_tasks, statistics.median
            ),
            "confirmatory_gate": False,
        }

    # Task bootstrap the full cross-fitted Spearman comparison by resampling
    # already-held-out task predictions; the fitted task never leaks back in.
    indices_by_task: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_task[str(row["instance_id"])].append(index)
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    improvements = []
    tasks = sorted(indices_by_task)
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(tasks, size=len(tasks), replace=True)
        indices = [index for task in sampled for index in indices_by_task[str(task)]]
        improvements.append(
            spearman([enhanced_prediction[index] for index in indices], [observed[index] for index in indices])
            - spearman([baseline_prediction[index] for index in indices], [observed[index] for index in indices])
        )
    improvement_ci = tuple(float(value) for value in np.quantile(improvements, [0.025, 0.5, 0.975]))
    all_gates["enhanced_spearman_improvement_at_least_0_10"] = improvement >= 0.10
    all_gates["enhanced_spearman_bootstrap_q025_above_0"] = improvement_ci[0] > 0
    result = {
        "status": "PASS" if all(all_gates.values()) else "STOP_BEFORE_PHYSICAL_SPLICE",
        "type_results": type_results,
        "prediction": {
            "baseline_task_leave_one_out_spearman": baseline_rho,
            "enhanced_task_leave_one_out_spearman": enhanced_rho,
            "improvement": improvement,
            "task_bootstrap_improvement_q025_q50_q975": improvement_ci,
        },
        "gates": all_gates,
        "next_action": (
            "open_registered_physical_splice_stage"
            if all(all_gates.values())
            else "stop_physical_and_runtime_stages"
        ),
    }
    _write_json(output / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "measure", "summarize"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trajectory-root", type=Path, default=TRAJECTORY_ROOT)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.output, args.trajectory_root)
    elif args.command == "measure":
        value = measure(args.output)
    else:
        value = summarize(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
