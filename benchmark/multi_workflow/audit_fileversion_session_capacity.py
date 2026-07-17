#!/usr/bin/env python3
"""Audit FileVersion SessionGraphKV capacity from explicit frozen artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from benchmark.multi_workflow.sessiongraph_raw_provenance import code_paths
from benchmark.multi_workflow.sessiongraph_v11 import (
    CostModel,
    canonical_mutations,
    read_jsonl,
    select_fileversion_modules,
    write_jsonl,
)


def audit(
    replay_path: Path,
    runtime_exact_path: Path,
    cost_gate_path: Path,
    mutations_path: Path,
    capacity_output: Path,
    gate_output: Path,
) -> dict[str, Any]:
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    exact_rows = read_jsonl(runtime_exact_path)
    cost_gate = json.loads(cost_gate_path.read_text(encoding="utf-8"))
    cost = CostModel(**cost_gate["cost_model"])
    mutations_rows = read_jsonl(mutations_path)
    mutations_by_session = canonical_mutations(mutations_rows)
    exact_by_turn = {
        (str(row["session_id"]), int(row["turn_id"])): set(
            str(value) for value in row["runtime_exact_module_ids"]
        )
        for row in exact_rows
    }
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for turn in replay:
        by_session[str(turn["session_id"])].append(turn)
    rows = []
    reason_tokens: dict[str, int] = defaultdict(int)
    stable_source_tokens = 0
    unresolved_source_tokens = 0
    for session_id, turns in sorted(by_session.items()):
        turns.sort(key=lambda row: int(row["turn_id"]))
        for previous, current in zip(turns, turns[1:]):
            resources = {
                str(module["module_id"]): code_paths(str(module.get("text", "")))
                for module in current["modules"]
                if module["module_type"] == "source_view"
            }
            selected, reasons = select_fileversion_modules(
                previous,
                current,
                runtime_exact_ids=exact_by_turn.get(
                    (session_id, int(current["turn_id"])), set()
                ),
                source_view_resources=resources,
                mutations=mutations_by_session.get(session_id, {}),
                cost_model=cost,
            )
            by_id = {
                str(module["module_id"]): module for module in current["modules"]
            }
            copied_tokens = sum(
                int(by_id[value]["token_span"][1])
                - int(by_id[value]["token_span"][0])
                for value in selected
            )
            prompt_tokens = sum(
                int(module["token_span"][1]) - int(module["token_span"][0])
                for module in current["modules"]
            )
            for module_id, reason in reasons.items():
                tokens = (
                    int(by_id[module_id]["token_span"][1])
                    - int(by_id[module_id]["token_span"][0])
                )
                reason_tokens[reason] += tokens
                if by_id[module_id]["module_type"] == "source_view":
                    if reason == "file_version_stable":
                        stable_source_tokens += tokens
                    elif reason == "source_path_unresolved":
                        unresolved_source_tokens += tokens
            positions = {
                str(module["module_id"]): int(module["position"])
                for module in current["modules"]
            }
            island_count = sum(
                index == 0
                or positions[value] != positions[selected[index - 1]] + 1
                for index, value in enumerate(selected)
            )
            rows.append(
                {
                    "session_id": session_id,
                    "turn_id": int(current["turn_id"]),
                    "prompt_tokens": prompt_tokens,
                    "copied_tokens": copied_tokens,
                    "reusable_fraction": copied_tokens / prompt_tokens
                    if prompt_tokens
                    else 0.0,
                    "copy_islands": island_count,
                    "copied_module_ids": selected,
                    "runtime_exact_module_ids": sorted(
                        exact_by_turn.get(
                            (session_id, int(current["turn_id"])), set()
                        )
                    ),
                    "source_view_resources": {
                        key: list(value) for key, value in resources.items() if value
                    },
                    "selection_reasons": reasons,
                }
            )
    session_fractions = {
        session_id: median(
            row["reusable_fraction"]
            for row in rows
            if row["session_id"] == session_id
        )
        for session_id in by_session
    }
    fraction = median(session_fractions.values()) if session_fractions else 0.0
    sessions_with_two = sum(
        sum(
            row["copied_tokens"] > 0
            for row in rows
            if row["session_id"] == session_id
        )
        >= 2
        for session_id in by_session
    )
    canonical_sha = hashlib.sha256(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in mutations_rows
        ).encode("utf-8")
    ).hexdigest()
    session_islands = {
        session_id: median(
            row["copy_islands"]
            for row in rows
            if row["session_id"] == session_id
        )
        for session_id in by_session
    }
    result = {
        "status": "PASS" if fraction >= 0.20 else "FALSIFIED",
        "passed": bool(rows) and fraction >= 0.20,
        "sessions": len(by_session),
        "turn_requests": len(rows),
        "measured_cost_rows": int(cost_gate["measured_rows"]),
        "median_file_version_reusable_fraction": fraction,
        "median_cost_positive_reusable_fraction": fraction,
        "median_copy_islands": median(session_islands.values())
        if session_islands
        else 0,
        "sessions_with_two_reusable_later_turns_fraction": (
            sessions_with_two / len(by_session) if by_session else 0.0
        ),
        "stable_source_view_tokens": stable_source_tokens,
        "unresolved_source_view_tokens_fail_closed": unresolved_source_tokens,
        "canonical_provenance_sha256": canonical_sha,
        "reason_tokens": dict(sorted(reason_tokens.items())),
        "model_or_kv_outputs_read_for_selection": False,
    }
    write_jsonl(capacity_output, rows)
    gate_output.parent.mkdir(parents=True, exist_ok=True)
    gate_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--runtime-exact", type=Path, required=True)
    parser.add_argument("--cost-gate", type=Path, required=True)
    parser.add_argument("--mutations", type=Path, required=True)
    parser.add_argument("--capacity-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                args.replay,
                args.runtime_exact,
                args.cost_gate,
                args.mutations,
                args.capacity_output,
                args.gate_output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
