#!/usr/bin/env python3
"""Audit position-bound identities for a bounded V45 observation pool.

V45.1 deliberately failed closed when a tool observation occurred more than
once anywhere in the rolling prompt.  That content-only identity rejects
otherwise distinguishable coding evidence such as repeated test summaries or
file reads issued by different commands.  This audit binds evidence to the
complete assistant/tool turn and resolves its token span only inside that
turn.  The V45 file-version guard, three-handle pool, and three-island target
limit remain unchanged.  No model or GPU request is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from jinja2 import StrictUndefined, Template
from tokenizers import Tokenizer

from benchmark.multi_workflow.audit_v451_multi_observation_pool import (
    COPY_CAP,
    MAX_ISLANDS,
    MIN_TOKENS,
    _candidate_literal,
    _nonoverlapping_top,
    _prompt_state,
    file_sha256,
)
from benchmark.multi_workflow.audit_v45_selected_target_guard import (
    V45,
    _groups,
    _planner,
)
from benchmark.multi_workflow.bridge_reuse_litellm_model import (
    BridgeReuseLitellmModel,
    capped_tail,
    find_sublist,
    token_ids_hash,
)
from benchmark.multi_workflow.coding_reuse_policy import (
    _mutation_effect_on_evidence,
    repository_commit_phase_event,
    tool_observation_sha256,
    versioned_grounded_observation_candidates,
)


def group_identity_sha256(group: Sequence[dict[str, Any]]) -> str:
    """Hash the online-visible assistant command and tool observation."""

    payload = json.dumps(
        list(group),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _group_literal(group: Sequence[dict[str, Any]]) -> str:
    return "".join(
        BridgeReuseLitellmModel._render_message_literal(message)
        for message in group
    )


def _unique_group_span(
    *,
    model: BridgeReuseLitellmModel,
    prompt_ids: list[int],
    group: Sequence[dict[str, Any]],
) -> tuple[int, int] | None:
    group_ids = model._tokenizer.encode(
        _group_literal(group), add_special_tokens=False
    ).ids
    positions = find_sublist(prompt_ids, group_ids)
    if len(positions) != 1:
        return None
    return positions[0], positions[0] + len(group_ids)


def _position_inside_group(
    *,
    prompt_ids: list[int],
    segment_ids: list[int],
    group_span: tuple[int, int],
) -> tuple[int | None, int]:
    """Return one segment occurrence inside the bound group and global count."""

    positions = find_sublist(prompt_ids, segment_ids)
    left, right = group_span
    inside = [
        start
        for start in positions
        if start >= left and start + len(segment_ids) <= right
    ]
    return (inside[0] if len(inside) == 1 else None), len(positions)


def _position_bound_guard(
    pending: dict[str, Any],
    retained_groups: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    expected_group = str(pending.get("source_group_sha256") or "")
    expected_observation = str(
        pending.get("evidence", {}).get("observation_sha256") or ""
    )
    source_paths = {
        str(value) for value in pending.get("evidence", {}).get("paths") or ()
    }
    source_symbols = {
        str(value)
        for value in pending.get("evidence", {}).get("symbols") or ()
    }
    group_matches = [
        index
        for index, group in enumerate(retained_groups)
        if expected_group and group_identity_sha256(group) == expected_group
    ]
    observation_matches = [
        index
        for index, group in enumerate(retained_groups)
        if expected_observation
        and tool_observation_sha256(group) == expected_observation
    ]
    result: dict[str, Any] = {
        "target_evidence_valid": False,
        "reason": "source_group_identity_not_unique",
        "source_group_matches": len(group_matches),
        "source_observation_matches": len(observation_matches),
        "source_group_index": (
            group_matches[0] if len(group_matches) == 1 else None
        ),
        "later_mutation_groups": 0,
        "source_paths": sorted(source_paths),
        "source_symbols": sorted(source_symbols),
    }
    if len(group_matches) != 1 or not source_paths:
        return result

    source_index = group_matches[0]
    if tool_observation_sha256(retained_groups[source_index]) != (
        expected_observation
    ):
        result["reason"] = "source_group_observation_mismatch"
        return result

    for later_index, later in enumerate(
        retained_groups[source_index + 1 :], start=source_index + 1
    ):
        if not repository_commit_phase_event(later):
            continue
        result["later_mutation_groups"] += 1
        effect = _mutation_effect_on_evidence(
            source_paths=source_paths,
            source_symbols=source_symbols,
            mutation=later,
        )
        if effect["reason"] == "same_file_symbol_disjoint_mutation":
            result.update(
                {
                    "reason": "same_file_symbol_disjoint_not_enabled",
                    "invalidating_group_index": later_index,
                    "changed_paths": effect["changed_paths"],
                    "changed_symbols": effect["changed_symbols"],
                }
            )
            return result
        if effect["invalidates"]:
            result.update(
                {
                    "reason": effect["reason"],
                    "invalidating_group_index": later_index,
                    "changed_paths": effect["changed_paths"],
                    "changed_symbols": effect["changed_symbols"],
                }
            )
            return result

    result.update(
        {
            "target_evidence_valid": True,
            "reason": "position_bound_version_evidence_valid",
        }
    )
    return result


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
    candidates, decision = versioned_grounded_observation_candidates(retained)
    group_hashes = [group_identity_sha256(group) for group in retained]
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        evidence = decision["candidate_evidence"][index]
        if not evidence["paths"]:
            counts["source_unlocalized"] += 1
            continue
        group = retained[int(evidence["group_index"])]
        group_hash = group_identity_sha256(group)
        if group_hashes.count(group_hash) != 1:
            counts["source_group_identity_not_unique"] += 1
            continue
        group_span = _unique_group_span(
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
        source_start, global_matches = _position_inside_group(
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


def _pool_key(row: dict[str, Any]) -> str:
    return ":".join(
        (
            str(row["source_group_sha256"]),
            ",".join(row["evidence"]["paths"]),
            str(row["segment_token_hash"]),
        )
    )


def audit_trajectory(
    path: Path,
    *,
    tokenizer: Tokenizer,
    chat_template: Template,
    frozen_prompt_hashes: dict[int, str],
) -> dict[str, Any]:
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    base = trajectory["messages"][:2]
    groups = _groups(trajectory["messages"][2:])
    calls = int(trajectory["info"]["model_stats"]["api_calls"])
    request_count = min(calls, len(groups) + 1)
    model = _planner(V45, tokenizer=tokenizer, chat_template=chat_template)
    pool: dict[str, dict[str, Any]] = {}
    source_records: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    rows = []

    for completed_count in range(request_count):
        request_index = completed_count + 1
        messages = list(base)
        for group in groups[:completed_count]:
            messages.extend(group)
        prompt_ids, selected_groups = _prompt_state(model, messages)
        if frozen_prompt_hashes.get(request_index) != token_ids_hash(prompt_ids):
            counts["prompt_hash_mismatches"] += 1
        counts["requests"] += 1
        counts["prompt_tokens"] += len(prompt_ids)

        target_rows = []
        released = []
        for key, handle in list(pool.items()):
            guard = _position_bound_guard(handle, selected_groups)
            if not guard["target_evidence_valid"]:
                reasons[str(guard["reason"])] += 1
                released.append(key)
                continue
            group = selected_groups[int(guard["source_group_index"])]
            group_span = _unique_group_span(
                model=model,
                prompt_ids=prompt_ids,
                group=group,
            )
            if group_span is None:
                reasons["target_group_token_span_not_unique"] += 1
                released.append(key)
                continue
            target_start, global_matches = _position_inside_group(
                prompt_ids=prompt_ids,
                segment_ids=handle["segment_ids"],
                group_span=group_span,
            )
            if target_start is None:
                reasons["target_segment_not_unique_inside_group"] += 1
                released.append(key)
                continue
            if global_matches > 1:
                counts["target_global_duplicate_islands"] += 1
                if guard["source_observation_matches"] > 1:
                    counts["target_duplicate_observation_islands"] += 1
            if (
                target_start <= 0
                or target_start + len(handle["segment_ids"]) >= len(prompt_ids)
            ):
                reasons["target_segment_not_strictly_middle"] += 1
                released.append(key)
                continue
            if prompt_ids[
                target_start : target_start + len(handle["segment_ids"])
            ] != handle["segment_ids"]:
                counts["target_token_slice_mismatches"] += 1
                released.append(key)
                continue
            target_rows.append(
                {
                    **handle,
                    "target_start": target_start,
                    "duplicate_observation_bound": (
                        guard["source_observation_matches"] > 1
                    ),
                }
            )
        for key in released:
            pool.pop(key, None)

        copied = _nonoverlapping_top(target_rows, limit=MAX_ISLANDS)
        if copied:
            counts["target_requests_with_copy"] += 1
        if any(row["duplicate_observation_bound"] for row in copied):
            counts["duplicate_observation_target_requests"] += 1
        if len(copied) > 1:
            counts["multi_island_target_requests"] += 1
        counts["target_islands"] += len(copied)
        copied_tokens = sum(len(row["segment_ids"]) for row in copied)
        counts["copied_tokens"] += copied_tokens
        counts["max_islands_on_one_target"] = max(
            counts["max_islands_on_one_target"], len(copied)
        )
        for handle in copied:
            source_records[handle["pool_key"]]["uses"] += 1

        proposed: dict[str, dict[str, Any]] = {}
        for candidate in _source_candidates(
            model=model,
            prompt_ids=prompt_ids,
            selected_groups=selected_groups,
            counts=counts,
        ):
            key = _pool_key(candidate)
            if key in pool:
                continue
            proposed[key] = {
                **candidate,
                "pool_key": key,
                "source_request_index": request_index,
            }
        ranked_keys = [
            key
            for key, _ in sorted(
                {**pool, **proposed}.items(),
                key=lambda item: (
                    len(item[1]["segment_ids"]),
                    item[1]["source_request_index"],
                ),
                reverse=True,
            )[:MAX_ISLANDS]
        ]
        keep = set(ranked_keys)
        for key in list(pool):
            if key not in keep:
                pool.pop(key)
                reasons["pool_capacity_eviction"] += 1
        registered = 0
        for key in ranked_keys:
            if key in pool:
                continue
            handle = proposed[key]
            pool[key] = handle
            source_records[key] = {
                "source_request_index": request_index,
                "tokens": len(handle["segment_ids"]),
                "uses": 0,
            }
            registered += 1
        counts["source_registrations"] += registered
        counts["max_pool_size"] = max(counts["max_pool_size"], len(pool))
        rows.append(
            {
                "request_index": request_index,
                "prompt_tokens": len(prompt_ids),
                "pool_size_after_registration": len(pool),
                "sources_registered": registered,
                "target_islands": len(copied),
                "copied_tokens": copied_tokens,
            }
        )

    uses = [row["uses"] for row in source_records.values()]
    productive_uses = [value for value in uses if value > 0]
    return {
        "instance_id": trajectory["instance_id"],
        "trajectory": str(path),
        "trajectory_sha256": file_sha256(path),
        "counts": dict(sorted(counts.items())),
        "release_reasons": dict(sorted(reasons.items())),
        "source_handles": len(source_records),
        "productive_source_handles": len(productive_uses),
        "source_uses": uses,
        "productive_source_uses": productive_uses,
        "requests_detail": rows,
    }


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
    trajectories = []
    for path in paths:
        instance_id = json.loads(path.read_text(encoding="utf-8"))["instance_id"]
        trajectories.append(
            audit_trajectory(
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
        "duplicate_observation_target_requests_min_7": totals["duplicate_observation_target_requests"] >= 7,
    }
    result = {
        "schema": "impactkv-v452-position-bound-pool-audit-v1",
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
