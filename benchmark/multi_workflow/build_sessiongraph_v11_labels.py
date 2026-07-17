#!/usr/bin/env python3
"""Build runtime-exact V11 labels and matched controls from explicit inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.sessiongraph_v11 import (
    PROFILE,
    assert_online_safe,
    build_label_rows,
    content_hash,
    exact_prefix_module_ids,
    prompt_hash,
    token_hash,
    write_jsonl,
)


def _tokens(tokenizer: Any, turn: dict[str, Any]) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in tokenizer.encode(
            turn["rendered_prompt"], add_special_tokens=False
        )
    )


def build(
    *,
    tokenizer: Any,
    replay_path: Path,
    capacity_path: Path,
    registration_path: Path,
    capacity_gate_path: Path,
    labels_dir: Path,
    prompts_output: Path,
    gate_output: Path,
) -> dict[str, Any]:
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    capacity_gate = json.loads(capacity_gate_path.read_text(encoding="utf-8"))
    if registration["policy"] != PROFILE or not capacity_gate.get("passed"):
        raise RuntimeError("V11 is not registered or capacity gate failed")
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert_online_safe(replay)
    selected_by_turn = {
        (str(row["session_id"]), int(row["turn_id"])): list(
            str(value) for value in row["copied_module_ids"]
        )
        for row in (
            json.loads(line)
            for line in capacity_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for turn in replay:
        by_session[str(turn["session_id"])].append(turn)
    modes = {mode: [] for mode in ("fileversion", "uniform", "shuffled", "type_only")}
    prompts = []
    mismatches = 0
    selection_mismatches = []
    for session_id, turns in sorted(by_session.items()):
        turns.sort(key=lambda row: int(row["turn_id"]))
        for previous, current in zip(turns, turns[1:]):
            warm = _tokens(tokenizer, previous)
            target = _tokens(tokenizer, current)
            previous_by_id = {
                str(module["module_id"]): module for module in previous["modules"]
            }
            current_by_id = {
                str(module["module_id"]): module for module in current["modules"]
            }
            exact_ids = set()
            for module_id in set(previous_by_id) & set(current_by_id):
                left = previous_by_id[module_id]
                right = current_by_id[module_id]
                left_span = slice(*map(int, left["token_span"]))
                right_span = slice(*map(int, right["token_span"]))
                if warm[left_span] == target[right_span] and warm[left_span]:
                    exact_ids.add(module_id)
                else:
                    mismatches += 1
            eligible = sorted(
                exact_ids - exact_prefix_module_ids(previous, current),
                key=lambda value: int(current_by_id[value]["position"]),
            )
            chunks = []
            for position, module_id in enumerate(eligible):
                module = current_by_id[module_id]
                start, end = map(int, module["token_span"])
                ids = target[start:end]
                if len(ids) <= 4:
                    continue
                slot_id = f"session:{module_id}"
                chunks.append(
                    {
                        "module_id": module_id,
                        "module_type": str(module["module_type"]),
                        "slot_id": slot_id,
                        "chunk_signature": hashlib.sha256(
                            f"{slot_id}\0{module.get('text', '')}".encode("utf-8")
                        ).hexdigest()[:16],
                        "chunk_len": len(ids),
                        "token_hash": token_hash(ids),
                        "position": position,
                    }
                )
            chunk_ids = {str(row["module_id"]) for row in chunks}
            key = (session_id, int(current["turn_id"]))
            selected = selected_by_turn[key]
            absent = sorted(set(selected) - chunk_ids)
            if absent:
                selection_mismatches.append(
                    {
                        "session_id": session_id,
                        "turn_id": key[1],
                        "selected_not_runtime_eligible": absent,
                    }
                )
            selected = [value for value in selected if value in chunk_ids]
            rows_by_mode = build_label_rows(
                case_id=f"{session_id}:t{key[1]}",
                chunks=chunks,
                copied_module_ids=selected,
            )
            for mode, rows in rows_by_mode.items():
                modes[mode].extend(rows)
            prompts.append(
                {
                    "session_id": session_id,
                    "turn_id": key[1],
                    "impact_case_id": f"{session_id}:t{key[1]}",
                    "warm_prompt_hash": prompt_hash(warm),
                    "target_prompt_hash": prompt_hash(target),
                    "warm_content_hash": content_hash(previous["rendered_prompt"]),
                    "target_content_hash": content_hash(current["rendered_prompt"]),
                    "warm_token_count": len(warm),
                    "target_token_count": len(target),
                    "eligible_modules": eligible,
                    "copied_modules": selected,
                }
            )
    for mode, rows in modes.items():
        write_jsonl(labels_dir / f"{mode}.jsonl", rows)
    prompts_output.parent.mkdir(parents=True, exist_ok=True)
    prompts_output.write_text(
        json.dumps(prompts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {
        "passed": bool(prompts)
        and all(modes.values())
        and not selection_mismatches
        and mismatches == 0,
        "turn_requests": len(prompts),
        "runtime_token_slice_mismatches": mismatches,
        "selection_mismatches": selection_mismatches,
        "labels": {mode: len(rows) for mode, rows in modes.items()},
        "prompt_mode_identity": "one frozen token hash per turn for every mode",
        "whole_slot_policy": True,
        "policy_profile": PROFILE,
    }
    gate_output.parent.mkdir(parents=True, exist_ok=True)
    gate_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--capacity-gate", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--prompts-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    print(
        json.dumps(
            build(
                tokenizer=tokenizer,
                replay_path=args.replay,
                capacity_path=args.capacity,
                registration_path=args.registration,
                capacity_gate_path=args.capacity_gate,
                labels_dir=args.labels_dir,
                prompts_output=args.prompts_output,
                gate_output=args.gate_output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
