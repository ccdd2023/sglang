#!/usr/bin/env python3
"""Replay V40 prompts through the exact V40 and narrowed V45 planners.

The audit uses the production tokenizer, chat template, rolling compaction,
source selection, capped token segment, uniqueness checks, and target guard.
It never executes a model request or SGLang kernel and therefore makes no
accuracy or latency claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from jinja2 import StrictUndefined, Template
from tokenizers import Tokenizer

from benchmark.multi_workflow.bridge_reuse_litellm_model import (
    BridgeReuseLitellmModel,
    token_ids_hash,
)
from benchmark.multi_workflow.context_bounded_litellm_model import (
    ContextBoundedLitellmModel,
)


V40 = "coding_grounded_observation_island_v40"
V45 = "coding_versioned_evidence_guard_v45"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _planner(
    arm: str,
    *,
    tokenizer: Tokenizer,
    chat_template: Template,
) -> BridgeReuseLitellmModel:
    model = object.__new__(BridgeReuseLitellmModel)
    model.config = SimpleNamespace(
        reuse_arm=arm,
        rolling_history_groups=6,
        reuse_copy_cap=4096,
        reuse_min_tokens=128,
        reuse_manifest_path=None,
        reuse_client_ledger_path=None,
        prompt_token_limit=28_000,
        max_tool_observation_chars=6_000,
        max_assistant_reasoning_chars=3_000,
        emergency_message_chars=1_500,
    )
    model._tokenizer = tokenizer
    model._chat_template = chat_template
    model._instance_nonce = f"offline-{arm}"
    model._request_index = 0
    model._session_index = 0
    model._last_message_count = 0
    model._pending_source = None
    model._commit_phase_latched = False
    model._last_stream_stats = {}
    return model


def _groups(messages: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return ContextBoundedLitellmModel._turn_groups(list(messages))


def _target_identity(prepared: dict[str, Any]) -> str | None:
    target = prepared["target"]
    if target is None:
        return None
    return ":".join(
        (
            str(target["segment_token_hash"]),
            str(target["target_start"]),
            str(target["length"]),
        )
    )


def replay_trajectory(
    path: Path,
    *,
    tokenizer: Tokenizer,
    chat_template: Template,
) -> dict[str, Any]:
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    base = trajectory["messages"][:2]
    groups = _groups(trajectory["messages"][2:])
    calls = int(trajectory["info"]["model_stats"]["api_calls"])
    planners = {
        arm: _planner(
            arm,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        for arm in (V40, V45)
    }
    counts: Counter[str] = Counter()
    guard_reasons: Counter[str] = Counter()
    differing_prompt_requests: list[int] = []
    rows: list[dict[str, Any]] = []

    request_count = min(calls, len(groups) + 1)
    for completed_count in range(request_count):
        messages = list(base)
        for group in groups[:completed_count]:
            messages.extend(group)
        prepared = {
            arm: planner.prepare_reuse_query(messages, write_sidecar=False)
            for arm, planner in planners.items()
        }
        prompt_hashes = {
            arm: token_ids_hash(value["prompt_ids"])
            for arm, value in prepared.items()
        }
        if len(set(prompt_hashes.values())) != 1:
            differing_prompt_requests.append(completed_count + 1)
        row = {
            "request_index": completed_count + 1,
            "prompt_hashes": prompt_hashes,
            "arms": {},
        }
        for arm, value in prepared.items():
            source = value["source"]
            target = value["target"]
            if source is not None:
                counts[f"{arm}_sources"] += 1
            if target is not None:
                counts[f"{arm}_targets"] += 1
                counts[f"{arm}_target_tokens"] += int(target["length"])
            row["arms"][arm] = {
                "source_registered": source is not None,
                "source_tokens": int(source["length"]) if source else 0,
                "target_registered": target is not None,
                "target_tokens": int(target["length"]) if target else 0,
                "target_identity": _target_identity(value),
                "releases": value["releases"],
                "decision": value["policy_decision"],
            }

        v40_target = prepared[V40]["target"]
        v45_target = prepared[V45]["target"]
        if v40_target is not None and v45_target is None:
            counts["v40_target_removed_by_v45"] += 1
            guard = prepared[V45]["policy_decision"].get(
                "target_evidence_guard", {}
            )
            reason = str(guard.get("reason") or "no_v45_pending_source")
            guard_reasons[reason] += 1
            if reason.startswith("same_file_") or reason == (
                "unlocalized_repository_mutation"
            ):
                counts["v40_target_removed_by_version_write_guard"] += 1
        elif v40_target is None and v45_target is not None:
            counts["v45_target_without_v40_target"] += 1
        elif v40_target is not None and v45_target is not None:
            counts["both_arms_target"] += 1
            if _target_identity(prepared[V40]) != _target_identity(
                prepared[V45]
            ):
                counts["both_target_but_different_segment"] += 1
        rows.append(row)

    return {
        "instance_id": trajectory.get("instance_id"),
        "trajectory": str(path),
        "trajectory_sha256": file_sha256(path),
        "api_calls": calls,
        "replayed_requests": request_count,
        "counts": dict(sorted(counts.items())),
        "guard_reasons": dict(sorted(guard_reasons.items())),
        "differing_prompt_requests": differing_prompt_requests,
        "requests": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(
        path
        for path in args.trajectory_root.rglob("*.traj.json")
        if V40 in path.parts
    )
    if not paths:
        raise SystemExit("no V40 trajectories found")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    chat_template = Template(
        args.chat_template.read_text(encoding="utf-8"),
        undefined=StrictUndefined,
    )
    trajectories = [
        replay_trajectory(
            path,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        for path in paths
    ]
    totals: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    differing_prompt_requests = 0
    for row in trajectories:
        totals.update(row["counts"])
        reasons.update(row["guard_reasons"])
        differing_prompt_requests += len(row["differing_prompt_requests"])
    v40_targets = totals[f"{V40}_targets"]
    v45_targets = totals[f"{V45}_targets"]
    retention = v45_targets / v40_targets if v40_targets else 0.0
    gates = {
        "identical_prompts": differing_prompt_requests == 0,
        "shared_target_segments_identical": (
            totals["both_target_but_different_segment"] == 0
        ),
        "v45_targets_are_v40_subset": (
            totals["v45_target_without_v40_target"] == 0
        ),
        "v40_runtime_eligible_targets_min_12": v40_targets >= 12,
        "version_write_targets_removed_min_1": (
            totals["v40_target_removed_by_version_write_guard"] >= 1
        ),
        "v45_runtime_eligible_targets_min_10": v45_targets >= 10,
        "v45_target_retention_min_0_50": retention >= 0.50,
    }
    result = {
        "schema": "impactkv-v45-selected-target-guard-audit-v1",
        "scope": {
            "trajectory_count": len(trajectories),
            "model_requests": 0,
            "gpu_requests": 0,
            "task_accuracy_claimed": False,
            "latency_claimed": False,
            "production_tokenizer": str(args.tokenizer),
            "production_chat_template": str(args.chat_template),
            "production_bridge_planner": True,
        },
        "totals": dict(sorted(totals.items())),
        "guard_reasons": dict(sorted(reasons.items())),
        "differing_prompt_requests": differing_prompt_requests,
        "v45_target_retention_vs_v40": retention,
        "frozen_gates": gates,
        "gpu_canary_registration_eligible": all(gates.values()),
        "inputs": {
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
                "gpu_canary_registration_eligible": result[
                    "gpu_canary_registration_eligible"
                ],
                "totals": result["totals"],
                "guard_reasons": result["guard_reasons"],
                "v45_target_retention_vs_v40": retention,
                "frozen_gates": gates,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
