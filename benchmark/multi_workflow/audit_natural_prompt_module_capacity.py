#!/usr/bin/env python3
"""Development-only capacity audit for variable-length prompt modules.

This script opens no new model outcomes.  It reparses the frozen 64-case Dense
design, verifies byte-for-byte prompt-token identity, and counts whether the
registered confirmatory experiment can support natural-module controls before
any fresh cohort is selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from benchmark.multi_workflow import motivate_v50_coding_provenance as m50
from benchmark.multi_workflow.natural_prompt_modules import (
    render_natural_prompt_modules,
)


ROOT = Path("/home/gfy/CodeMAS_Project")
DESIGN = (
    ROOT
    / "kvflow-artifacts/impactkv_module_conditioned_attention_kv_20260807/"
    "task_disjoint20/DESIGN.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "kvflow-artifacts/impactkv_natural_prompt_modules_20260808/development64"
)
MIN_LENGTH = 32
MAX_LENGTH = 4096
PRIMARY_TYPES = ("repository_code", "assistant_interpretation")


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


def _trajectory_paths(root: Path) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in sorted(root.glob("**/*.traj.json")):
        value = _read_json(path)
        instance_id = str(value.get("instance_id") or "")
        if instance_id and instance_id not in selected:
            selected[instance_id] = path
    return selected


def _same_module_map(
    source: Sequence[Mapping[str, Any]], target: Sequence[Mapping[str, Any]]
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    target_by_key = {
        (
            module["parent_interaction_id"],
            module["module_type"],
            module["content_hash"],
        ): module
        for module in target
    }
    matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for module in source:
        key = (
            module["parent_interaction_id"],
            module["module_type"],
            module["content_hash"],
        )
        if key in target_by_key:
            matches.append((module, target_by_key[key]))
    return matches


def boundary_control(
    candidate: Mapping[str, Any], modules: Sequence[Mapping[str, Any]]
) -> tuple[int, int] | None:
    """Choose a same-parent, same-length window crossing a natural boundary."""

    parent = [
        module
        for module in modules
        if module["parent_interaction_id"] == candidate["parent_interaction_id"]
    ]
    if len(parent) < 2:
        return None
    length = int(candidate["natural_length"])
    parent_start = int(parent[0]["token_start"])
    parent_end = int(parent[-1]["token_end"])
    if parent_end - parent_start < length:
        return None
    choices: list[tuple[int, int, int]] = []
    for left, right in zip(parent, parent[1:]):
        boundary = int(left["token_end"])
        low = max(parent_start, boundary - length + 1)
        high = min(boundary - 1, parent_end - length)
        if low > high:
            continue
        ideal = boundary - length // 2
        start = min(max(ideal, low), high)
        end = start + length
        if start == int(candidate["token_start"]) and end == int(candidate["token_end"]):
            continue
        choices.append((abs(start - int(candidate["token_start"])), start, end))
    if not choices:
        return None
    _, start, end = min(choices)
    return start, end


def recency_control(
    candidate: Mapping[str, Any], modules: Sequence[Mapping[str, Any]], prompt_length: int
) -> Mapping[str, Any] | None:
    """Choose a recent same-type module able to host an equal-length span."""

    length = int(candidate["natural_length"])
    position = int(candidate["token_start"]) / max(prompt_length, 1)
    alternatives = [
        module
        for module in modules
        if module["module_type"] == candidate["module_type"]
        and module["module_id"] != candidate["module_id"]
        and int(module["natural_length"]) >= length
        and int(module["source_request_index"]) <= int(candidate["source_request_index"])
    ]
    if not alternatives:
        return None
    return min(
        alternatives,
        key=lambda module: (
            abs(int(module["token_start"]) / max(prompt_length, 1) - position),
            -int(module["source_request_index"]),
            -int(module["token_start"]),
        ),
    )


def audit(design_path: Path, output: Path) -> dict[str, Any]:
    design = _read_json(design_path)
    trajectory_root = Path(design["trajectory_root"])
    trajectories = _trajectory_paths(trajectory_root)
    tokenizer = AutoTokenizer.from_pretrained(
        design.get("analysis_model") or m50.MODEL,
        trust_remote_code=True,
    )
    registration = {
        "status": "DEVELOPMENT_CAPACITY_ONLY",
        "purpose": (
            "Natural-module parser and matched-control capacity audit; no new "
            "attention, splice, task-accuracy, or runtime outcome is opened."
        ),
        "design": str(design_path),
        "design_sha256": _sha256(design_path),
        "trajectory_root": str(trajectory_root),
        "minimum_natural_tokens": MIN_LENGTH,
        "maximum_physical_reuse_tokens": MAX_LENGTH,
        "confirmatory_use_allowed": False,
    }
    if output.exists() and (output / "REGISTRATION.json").exists():
        if _read_json(output / "REGISTRATION.json") != registration:
            raise ValueError("development registration differs from frozen audit")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / "REGISTRATION.json", registration)

    module_counts: Counter[str] = Counter()
    eligible_counts: Counter[str] = Counter()
    module_tasks: dict[str, set[str]] = defaultdict(set)
    eligible_tasks: dict[str, set[str]] = defaultdict(set)
    boundary_counts: Counter[str] = Counter()
    boundary_tasks: dict[str, set[str]] = defaultdict(set)
    recency_counts: Counter[str] = Counter()
    recency_tasks: dict[str, set[str]] = defaultdict(set)
    lengths: dict[str, list[int]] = defaultdict(list)
    case_rows: list[dict[str, Any]] = []

    for case in design["cases"]:
        instance_id = str(case["instance_id"])
        trajectory = _read_json(trajectories[instance_id])
        messages = trajectory["messages"]
        base = messages[:2]
        groups = m50._turn_groups(messages[2:])
        target_completed = int(case["request_index"]) - 1
        source_ids, source_modules, _ = render_natural_prompt_modules(
            tokenizer, base, groups[: target_completed - 1]
        )
        target_ids, target_modules, relations = render_natural_prompt_modules(
            tokenizer, base, groups[:target_completed]
        )
        if source_ids != list(case["source_input_ids"]):
            raise AssertionError(f"source prompt mismatch: {case['case_id']}")
        if target_ids != list(case["target_input_ids"]):
            raise AssertionError(f"target prompt mismatch: {case['case_id']}")

        for module in target_modules:
            module_type = str(module["module_type"])
            module_counts[module_type] += 1
            module_tasks[module_type].add(instance_id)
            lengths[module_type].append(int(module["natural_length"]))

        candidates: list[dict[str, Any]] = []
        for source_module, target_module in _same_module_map(source_modules, target_modules):
            module_type = str(target_module["module_type"])
            length = int(target_module["natural_length"])
            if module_type not in PRIMARY_TYPES or not MIN_LENGTH <= length <= MAX_LENGTH:
                continue
            if target_module.get("invalidating_event") is not None:
                continue
            eligible_counts[module_type] += 1
            eligible_tasks[module_type].add(instance_id)
            boundary = boundary_control(target_module, target_modules)
            recent = recency_control(target_module, target_modules, len(target_ids))
            if boundary is not None:
                boundary_counts[module_type] += 1
                boundary_tasks[module_type].add(instance_id)
            if recent is not None:
                recency_counts[module_type] += 1
                recency_tasks[module_type].add(instance_id)
            candidates.append(
                {
                    "module_id": target_module["module_id"],
                    "module_type": module_type,
                    "natural_length": length,
                    "source_start": source_module["token_start"],
                    "target_start": target_module["token_start"],
                    "boundary_control": boundary,
                    "recency_control_module_id": (
                        recent["module_id"] if recent is not None else None
                    ),
                }
            )
        case_rows.append(
            {
                "case_id": case["case_id"],
                "instance_id": instance_id,
                "request_index": case["request_index"],
                "prompt_tokens": len(target_ids),
                "modules": target_modules,
                "relations": relations,
                "eligible_candidates": candidates,
            }
        )

    def type_summary(module_type: str) -> dict[str, Any]:
        values = lengths[module_type]
        return {
            "modules": module_counts[module_type],
            "tasks": len(module_tasks[module_type]),
            "eligible_reuse_instances": eligible_counts[module_type],
            "eligible_tasks": len(eligible_tasks[module_type]),
            "same_parent_boundary_pairs": boundary_counts[module_type],
            "boundary_pair_tasks": len(boundary_tasks[module_type]),
            "same_type_recency_pairs": recency_counts[module_type],
            "recency_pair_tasks": len(recency_tasks[module_type]),
            "length_min": min(values) if values else None,
            "length_median": statistics.median(values) if values else None,
            "length_max": max(values) if values else None,
        }

    primary = {module_type: type_summary(module_type) for module_type in PRIMARY_TYPES}
    gates = {
        "at_least_16_tasks": len({row["instance_id"] for row in case_rows}) >= 16,
        "at_least_64_target_prompts": len(case_rows) >= 64,
        "code_at_least_48_modules_8_tasks": (
            primary["repository_code"]["eligible_reuse_instances"] >= 48
            and primary["repository_code"]["eligible_tasks"] >= 8
        ),
        "interpretation_at_least_48_modules_8_tasks": (
            primary["assistant_interpretation"]["eligible_reuse_instances"] >= 48
            and primary["assistant_interpretation"]["eligible_tasks"] >= 8
        ),
        "code_at_least_32_boundary_pairs_8_tasks": (
            primary["repository_code"]["same_parent_boundary_pairs"] >= 32
            and primary["repository_code"]["boundary_pair_tasks"] >= 8
        ),
        "interpretation_at_least_32_boundary_pairs_8_tasks": (
            primary["assistant_interpretation"]["same_parent_boundary_pairs"] >= 32
            and primary["assistant_interpretation"]["boundary_pair_tasks"] >= 8
        ),
        "code_at_least_32_recency_pairs_8_tasks": (
            primary["repository_code"]["same_type_recency_pairs"] >= 32
            and primary["repository_code"]["recency_pair_tasks"] >= 8
        ),
        "interpretation_at_least_32_recency_pairs_8_tasks": (
            primary["assistant_interpretation"]["same_type_recency_pairs"] >= 32
            and primary["assistant_interpretation"]["recency_pair_tasks"] >= 8
        ),
    }
    result = {
        "status": "PASS" if all(gates.values()) else "CAPACITY_SHORTFALL",
        "cases": len(case_rows),
        "tasks": len({row["instance_id"] for row in case_rows}),
        "prompt_identity_verified": True,
        "module_types": {
            module_type: type_summary(module_type)
            for module_type in sorted(module_counts)
        },
        "primary_types": primary,
        "gates": gates,
        "next_action": (
            "register_fresh_task_disjoint_cohort"
            if all(gates.values())
            else "revise_parser_or_controls_without_opening_new_outcomes"
        ),
    }
    with (output / "CASES.jsonl").open("w", encoding="utf-8") as handle:
        for row in case_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write_json(output / "CAPACITY.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=DESIGN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(audit(args.design, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

