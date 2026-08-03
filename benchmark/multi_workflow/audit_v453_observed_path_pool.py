#!/usr/bin/env python3
"""Audit output-derived file provenance for the bounded V45.1 pool.

The V45.1 audit excluded otherwise eligible coding observations when the tool
command did not itself contain a concrete repository file.  Search results
and inspected patches often contain those paths in the tool output.  This
audit extracts only literal online-visible paths.  Directory-wide search
results are conservatively bound to the whole repository and are invalidated
by any later repository mutation.  It makes no model or GPU request.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from jinja2 import StrictUndefined, Template
from tokenizers import Tokenizer

import benchmark.multi_workflow.audit_v452_position_bound_pool as base
from benchmark.multi_workflow.audit_v451_multi_observation_pool import (
    COPY_CAP,
    MAX_ISLANDS,
    MIN_TOKENS,
    _candidate_literal,
    file_sha256,
)
from benchmark.multi_workflow.audit_v45_selected_target_guard import (
    V45,
    _planner,
)
from benchmark.multi_workflow.bridge_reuse_litellm_model import (
    BridgeReuseLitellmModel,
    capped_tail,
    token_ids_hash,
)
from benchmark.multi_workflow.coding_reuse_policy import (
    _mutation_effect_on_evidence,
    _tool_command,
    is_successful_readonly_evidence,
    repository_commit_phase_event,
    repository_observation_symbols,
    repository_paths,
    tool_observation_sha256,
)


_REPOSITORY_FILE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"((?:/testbed/|\./|a/|b/)?"
    r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*"
    r"\.(?:py|pyi|toml|yaml|yml|json|rst|md|cfg|ini|txt))\b"
)
_REPOSITORY_SCOPE_SEARCH = re.compile(
    r"(?:^|&&\s*|;\s*|\|\|\s*)find\s+"
    r"|\bgrep\b[^\n;&|]*(?:\s-[A-Za-z]*[rR][A-Za-z]*\b|--recursive\b)",
    re.I,
)


def _normalize_path(value: str) -> str:
    value = value.strip().lstrip("./")
    if value.startswith("testbed/"):
        value = value[len("testbed/") :]
    if value.startswith(("a/", "b/")):
        value = value[2:]
    return value


def observed_path_provenance(
    group: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    commands = "\n".join(
        command for message in group if (command := _tool_command(message))
    )
    observations = "\n".join(
        str(message.get("content") or "")
        for message in group
        if message.get("role") == "tool"
    )
    command_paths = repository_paths(group)
    literal_paths = {
        _normalize_path(match.group(1))
        for match in _REPOSITORY_FILE.finditer(
            commands + "\n" + observations
        )
    }
    paths = {value for value in command_paths | literal_paths if value}
    scope_dependency = bool(_REPOSITORY_SCOPE_SEARCH.search(commands))
    return {
        "paths": sorted(paths),
        "command_paths": sorted(command_paths),
        "observation_added_paths": sorted(paths - command_paths),
        "repository_scope_dependency": scope_dependency,
    }


def _observed_candidates(
    retained_groups: Sequence[Sequence[dict[str, Any]]],
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    candidates: list[list[dict[str, Any]]] = []
    evidence: list[dict[str, Any]] = []
    for index, group in enumerate(retained_groups):
        if not is_successful_readonly_evidence(group):
            continue
        provenance = observed_path_provenance(group)
        source_paths = set(provenance["paths"])
        if not source_paths:
            continue
        source_symbols = repository_observation_symbols(group)
        invalid = False
        for later in retained_groups[index + 1 :]:
            if not repository_commit_phase_event(later):
                continue
            if provenance["repository_scope_dependency"]:
                invalid = True
                break
            effect = _mutation_effect_on_evidence(
                source_paths=source_paths,
                source_symbols=source_symbols,
                mutation=later,
            )
            if effect["invalidates"] or effect["reason"] == (
                "same_file_symbol_disjoint_mutation"
            ):
                invalid = True
                break
        if invalid:
            continue
        tool_messages = [
            message for message in group if message.get("role") == "tool"
        ]
        if not tool_messages:
            continue
        candidates.append(tool_messages)
        evidence.append(
            {
                "group_index": index,
                "paths": sorted(source_paths),
                "symbols": sorted(source_symbols),
                "observation_sha256": tool_observation_sha256(group),
                "path_provenance": provenance,
            }
        )
    return candidates, evidence


def _source_candidates(
    *,
    model: BridgeReuseLitellmModel,
    prompt_ids: list[int],
    selected_groups: list[list[dict[str, Any]]],
    counts: Counter[str],
) -> list[dict[str, Any]]:
    retained = (
        selected_groups
        if len(selected_groups) < model.config.rolling_history_groups
        else selected_groups[1:]
    )
    candidates, evidence_rows = _observed_candidates(retained)
    group_hashes = [base.group_identity_sha256(group) for group in retained]
    rows: list[dict[str, Any]] = []
    for candidate, evidence in zip(candidates, evidence_rows, strict=True):
        group = retained[int(evidence["group_index"])]
        group_hash = base.group_identity_sha256(group)
        if group_hashes.count(group_hash) != 1:
            counts["source_group_identity_not_unique"] += 1
            continue
        group_span = base._unique_group_span(
            model=model,
            prompt_ids=prompt_ids,
            group=group,
        )
        if group_span is None:
            counts["source_group_token_span_not_unique"] += 1
            continue
        segment_ids = model._tokenizer.encode(
            _candidate_literal(candidate), add_special_tokens=False
        ).ids
        if len(segment_ids) < MIN_TOKENS:
            counts["source_below_minimum_tokens"] += 1
            continue
        source_start, global_matches = base._position_inside_group(
            prompt_ids=prompt_ids,
            segment_ids=segment_ids,
            group_span=group_span,
        )
        if source_start is None:
            counts["source_segment_not_unique_inside_group"] += 1
            continue
        if global_matches > 1:
            counts["source_global_duplicate_resolved"] += 1
        uncapped_tokens = len(segment_ids)
        segment_ids, source_start = capped_tail(
            segment_ids, source_start, COPY_CAP
        )
        if source_start <= 0 or source_start + len(segment_ids) >= len(prompt_ids):
            counts["source_segment_not_strictly_middle"] += 1
            continue
        if prompt_ids[source_start : source_start + len(segment_ids)] != segment_ids:
            counts["source_token_slice_mismatches"] += 1
            continue
        if evidence["path_provenance"]["observation_added_paths"]:
            counts["observation_path_candidate_rows"] += 1
        if evidence["path_provenance"]["repository_scope_dependency"]:
            counts["repository_scope_candidate_rows"] += 1
        rows.append(
            {
                "evidence": evidence,
                "source_group_sha256": group_hash,
                "segment_ids": segment_ids,
                "segment_token_hash": token_ids_hash(segment_ids),
                "source_start": source_start,
                "uncapped_tokens": uncapped_tokens,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            len(row["segment_ids"]),
            row["evidence"]["group_index"],
        ),
        reverse=True,
    )


_BASE_POSITION_GUARD = base._position_bound_guard


def _observed_path_guard(
    pending: dict[str, Any],
    retained_groups: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    result = _BASE_POSITION_GUARD(pending, retained_groups)
    if not result["target_evidence_valid"]:
        return result
    provenance = pending["evidence"].get("path_provenance") or {}
    if not provenance.get("repository_scope_dependency"):
        return result
    source_index = int(result["source_group_index"])
    for later_index, later in enumerate(
        retained_groups[source_index + 1 :], start=source_index + 1
    ):
        if repository_commit_phase_event(later):
            return {
                **result,
                "target_evidence_valid": False,
                "reason": "repository_scope_mutated",
                "invalidating_group_index": later_index,
            }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--frozen-v45-audit", type=Path, required=True)
    parser.add_argument("--v451-result", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(
        path
        for path in args.trajectory_root.rglob("*.traj.json")
        if "coding_grounded_observation_island_v40" in path.parts
    )
    frozen = json.loads(args.frozen_v45_audit.read_text(encoding="utf-8"))
    frozen_hashes = {
        str(row["instance_id"]): {
            int(request["request_index"]): str(request["prompt_hashes"][V45])
            for request in row["requests"]
        }
        for row in frozen["trajectories"]
    }
    v451 = json.loads(args.v451_result.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    chat_template = Template(
        args.chat_template.read_text(encoding="utf-8"),
        undefined=StrictUndefined,
    )

    base._source_candidates = _source_candidates
    base._position_bound_guard = _observed_path_guard
    trajectories = []
    for path in paths:
        instance_id = json.loads(path.read_text(encoding="utf-8"))["instance_id"]
        trajectories.append(
            base.audit_trajectory(
                path,
                tokenizer=tokenizer,
                chat_template=chat_template,
                frozen_prompt_hashes=frozen_hashes[instance_id],
            )
        )

    totals: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    all_uses: list[int] = []
    productive_uses: list[int] = []
    max_islands = 0
    max_pool = 0
    for row in trajectories:
        counts = row["counts"]
        max_islands = max(max_islands, int(counts.get("max_islands_on_one_target", 0)))
        max_pool = max(max_pool, int(counts.get("max_pool_size", 0)))
        totals.update(
            {
                key: value
                for key, value in counts.items()
                if key not in {"max_islands_on_one_target", "max_pool_size"}
            }
        )
        reasons.update(row["release_reasons"])
        all_uses.extend(row["source_uses"])
        productive_uses.extend(row["productive_source_uses"])
    totals["max_islands_on_one_target"] = max_islands
    totals["max_pool_size"] = max_pool

    copied_fraction = totals["copied_tokens"] / totals["prompt_tokens"]
    target_rate = totals["target_requests_with_copy"] / totals["requests"]
    mean_productive_uses = (
        statistics.fmean(productive_uses) if productive_uses else 0.0
    )
    target_gain = (
        totals["target_requests_with_copy"]
        - int(v451["totals"]["target_requests_with_copy"])
    )
    gates = {
        "prompt_hash_mismatches_zero": totals["prompt_hash_mismatches"] == 0,
        "source_token_slice_mismatches_zero": totals["source_token_slice_mismatches"] == 0,
        "target_token_slice_mismatches_zero": totals["target_token_slice_mismatches"] == 0,
        "max_islands_at_most_3": totals["max_islands_on_one_target"] <= 3,
        "max_pool_size_at_most_3": totals["max_pool_size"] <= 3,
        "multi_island_target_requests_min_20": totals["multi_island_target_requests"] >= 20,
        "copied_prompt_token_fraction_min_0_25": copied_fraction >= 0.25,
        "target_request_rate_min_0_65": target_rate >= 0.65,
        "mean_productive_source_uses_min_1_5": mean_productive_uses >= 1.5,
        "target_request_gain_vs_v451_min_7": target_gain >= 7,
        "observation_path_candidate_rows_min_7": totals["observation_path_candidate_rows"] >= 7,
    }
    result = {
        "schema": "impactkv-v453-observed-path-pool-audit-v1",
        "scope": {
            "trajectory_count": len(trajectories),
            "max_islands_per_target": MAX_ISLANDS,
            "copy_cap": COPY_CAP,
            "minimum_tokens": MIN_TOKENS,
            "model_requests": 0,
            "gpu_requests": 0,
            "prefetch": False,
            "assistant_reasoning_selected": False,
            "symbol_disjoint_relaxation": False,
            "directory_search_dependency": "repository-wide",
            "task_accuracy_claimed": False,
            "latency_claimed": False,
        },
        "totals": dict(sorted(totals.items())),
        "release_reasons": dict(sorted(reasons.items())),
        "copied_prompt_token_fraction": copied_fraction,
        "target_request_rate": target_rate,
        "target_request_gain_vs_v451": target_gain,
        "source_handles": len(all_uses),
        "productive_source_handles": len(productive_uses),
        "mean_productive_source_uses": mean_productive_uses,
        "frozen_gates": gates,
        "runtime_implementation_eligible": all(gates.values()),
        "inputs": {
            "frozen_v45_audit": str(args.frozen_v45_audit),
            "frozen_v45_audit_sha256": file_sha256(args.frozen_v45_audit),
            "v451_result": str(args.v451_result),
            "v451_result_sha256": file_sha256(args.v451_result),
            "tokenizer_sha256": file_sha256(args.tokenizer),
            "chat_template_sha256": file_sha256(args.chat_template),
        },
        "trajectories": trajectories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "runtime_implementation_eligible": result["runtime_implementation_eligible"],
                "totals": result["totals"],
                "release_reasons": result["release_reasons"],
                "copied_prompt_token_fraction": copied_fraction,
                "target_request_rate": target_rate,
                "target_request_gain_vs_v451": target_gain,
                "mean_productive_source_uses": mean_productive_uses,
                "frozen_gates": gates,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
