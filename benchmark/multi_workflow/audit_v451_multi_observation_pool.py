#!/usr/bin/env python3
"""Audit a bounded, persistent V45 multi-observation pool without GPU use.

The proposal keeps up to three independent grounded tool-observation handles.
Each handle is registered from a real preceding prompt, may serve more than
one later target, and is invalidated independently by the existing V45 file-
version guard.  The audit uses the production tokenizer, chat template,
rolling compaction, and exact token occurrence checks.  It does not execute a
model request or claim task accuracy/latency.
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
    versioned_evidence_target_guard,
    versioned_grounded_observation_candidates,
)


MAX_ISLANDS = 3
COPY_CAP = 4096
MIN_TOKENS = 128


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_state(
    model: BridgeReuseLitellmModel,
    messages: list[dict[str, Any]],
) -> tuple[list[int], list[list[dict[str, Any]]]]:
    rolling_messages, selected_groups, _ = model._rolling_messages(messages)
    compacted_messages, _ = model.compact_messages(rolling_messages)
    return model._render_prompt_ids(compacted_messages), selected_groups


def _candidate_literal(candidate: Sequence[dict[str, Any]]) -> str:
    return "".join(
        BridgeReuseLitellmModel._render_message_literal(message)
        for message in candidate
    )


def _source_candidates(
    *,
    model: BridgeReuseLitellmModel,
    prompt_ids: list[int],
    selected_groups: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    # Before the six-group window is full, no group rolls out on the next
    # request. Once full, the oldest group will roll and cannot be a source.
    retained = (
        selected_groups
        if len(selected_groups) < model.config.rolling_history_groups
        else selected_groups[1:]
    )
    candidates, decision = versioned_grounded_observation_candidates(retained)
    rows = []
    for index, candidate in enumerate(candidates):
        evidence = decision["candidate_evidence"][index]
        if not evidence["paths"]:
            continue
        segment_ids = model._tokenizer.encode(
            _candidate_literal(candidate), add_special_tokens=False
        ).ids
        positions = find_sublist(prompt_ids, segment_ids)
        if len(segment_ids) < MIN_TOKENS or len(positions) != 1:
            continue
        segment_ids, source_start = capped_tail(
            segment_ids, positions[0], COPY_CAP
        )
        if source_start <= 0 or source_start + len(segment_ids) >= len(prompt_ids):
            continue
        rows.append(
            {
                "evidence": evidence,
                "segment_ids": segment_ids,
                "segment_token_hash": token_ids_hash(segment_ids),
                "source_start": source_start,
                "uncapped_tokens": len(
                    model._tokenizer.encode(
                        _candidate_literal(candidate),
                        add_special_tokens=False,
                    ).ids
                ),
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
    evidence = row["evidence"]
    return ":".join(
        (
            str(evidence["observation_sha256"]),
            ",".join(evidence["paths"]),
            str(row["segment_token_hash"]),
        )
    )


def _nonoverlapping_top(
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    intervals: list[tuple[int, int]] = []
    for row in sorted(
        rows,
        key=lambda value: (
            len(value["segment_ids"]),
            value["source_request_index"],
        ),
        reverse=True,
    ):
        start = int(row["target_start"])
        end = start + len(row["segment_ids"])
        if any(start < right and left < end for left, right in intervals):
            continue
        selected.append(row)
        intervals.append((start, end))
        if len(selected) >= limit:
            break
    return sorted(selected, key=lambda row: row["target_start"])


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
        prompt_hash = token_ids_hash(prompt_ids)
        if frozen_prompt_hashes.get(request_index) != prompt_hash:
            counts["prompt_hash_mismatches"] += 1
        counts["requests"] += 1
        counts["prompt_tokens"] += len(prompt_ids)

        target_rows = []
        released = []
        for key, handle in list(pool.items()):
            guard = versioned_evidence_target_guard(
                {
                    "source_observation_sha256": handle["evidence"][
                        "observation_sha256"
                    ],
                    "source_paths": handle["evidence"]["paths"],
                    "source_symbols": handle["evidence"]["symbols"],
                },
                selected_groups,
                allow_symbol_disjoint=False,
            )
            if not guard["target_evidence_valid"]:
                reasons[str(guard["reason"])] += 1
                released.append(key)
                continue
            positions = find_sublist(prompt_ids, handle["segment_ids"])
            if len(positions) != 1:
                reasons["target_segment_not_unique"] += 1
                released.append(key)
                continue
            target_start = positions[0]
            if (
                target_start <= 0
                or target_start + len(handle["segment_ids"]) >= len(prompt_ids)
            ):
                reasons["target_segment_not_strictly_middle"] += 1
                released.append(key)
                continue
            target_rows.append({**handle, "target_start": target_start})
        for key in released:
            pool.pop(key, None)

        copied = _nonoverlapping_top(target_rows, limit=MAX_ISLANDS)
        if copied:
            counts["target_requests_with_copy"] += 1
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
            int(request["request_index"]): str(
                request["prompt_hashes"][V45]
            )
            for request in row["requests"]
        }
        for row in frozen["trajectories"]
    }
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    chat_template = Template(
        args.chat_template.read_text(encoding="utf-8"),
        undefined=StrictUndefined,
    )
    trajectories = []
    for path in paths:
        instance_id = json.loads(path.read_text(encoding="utf-8"))[
            "instance_id"
        ]
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
    all_uses = []
    productive_uses = []
    max_islands_on_one_target = 0
    max_pool_size = 0
    for row in trajectories:
        counts = row["counts"]
        max_islands_on_one_target = max(
            max_islands_on_one_target,
            int(counts.get("max_islands_on_one_target", 0)),
        )
        max_pool_size = max(
            max_pool_size,
            int(counts.get("max_pool_size", 0)),
        )
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
    totals["max_islands_on_one_target"] = max_islands_on_one_target
    totals["max_pool_size"] = max_pool_size
    copied_fraction = totals["copied_tokens"] / totals["prompt_tokens"]
    target_rate = totals["target_requests_with_copy"] / totals["requests"]
    mean_productive_uses = (
        statistics.fmean(productive_uses) if productive_uses else 0.0
    )
    gates = {
        "prompt_hash_mismatches_zero": totals["prompt_hash_mismatches"] == 0,
        "max_islands_at_most_3": totals["max_islands_on_one_target"] <= 3,
        "max_pool_size_at_most_3": totals["max_pool_size"] <= 3,
        "multi_island_target_requests_min_20": (
            totals["multi_island_target_requests"] >= 20
        ),
        "copied_prompt_token_fraction_min_0_25": copied_fraction >= 0.25,
        "target_request_rate_min_0_65": target_rate >= 0.65,
        "mean_productive_source_uses_min_1_5": mean_productive_uses >= 1.5,
    }
    result = {
        "schema": "impactkv-v451-multi-observation-pool-audit-v1",
        "scope": {
            "trajectory_count": len(trajectories),
            "max_islands_per_target": MAX_ISLANDS,
            "copy_cap": COPY_CAP,
            "minimum_tokens": MIN_TOKENS,
            "model_requests": 0,
            "gpu_requests": 0,
            "prefetch": False,
            "task_accuracy_claimed": False,
            "latency_claimed": False,
        },
        "totals": dict(sorted(totals.items())),
        "release_reasons": dict(sorted(reasons.items())),
        "copied_prompt_token_fraction": copied_fraction,
        "target_request_rate": target_rate,
        "source_handles": len(all_uses),
        "productive_source_handles": len(productive_uses),
        "mean_productive_source_uses": mean_productive_uses,
        "frozen_gates": gates,
        "runtime_implementation_eligible": all(gates.values()),
        "inputs": {
            "frozen_v45_audit": str(args.frozen_v45_audit),
            "frozen_v45_audit_sha256": file_sha256(args.frozen_v45_audit),
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
                "runtime_implementation_eligible": result[
                    "runtime_implementation_eligible"
                ],
                "totals": result["totals"],
                "copied_prompt_token_fraction": copied_fraction,
                "target_request_rate": target_rate,
                "mean_productive_source_uses": mean_productive_uses,
                "frozen_gates": gates,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
