#!/usr/bin/env python3
"""Compare global block attention under Dense and actual V40 KV reuse.

This is a mechanism experiment, not an accuracy or latency benchmark.  It
reconstructs real source/target requests from the M56 same-prompt campaign,
then retokenizes the unchanged prompt text with Qwen2.5-Coder-3B-Instruct so a
five-layer attention probe fits on one 24 GB GPU.  Dense executes every target
token.  The reuse arm executes the Dense prefix, inserts the source-time K/V
island with the same RoPE delta used by the V40 reference executor, and only
executes the suffix.  Copied target rows are therefore explicitly unavailable;
their source-time formation attention is reported separately.

The persistent output is block-level.  Query tokens are processed in bounded
chunks and no token-by-token N x N attention matrix is written to disk.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import gc
import hashlib
import json
import math
import statistics
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path("/home/gfy/CodeMAS_Project")
M56_ROOT = ROOT / "kvflow-artifacts/impactkv_m56_v40_same_prompt_20260805/fresh13"
M55_ROOT = ROOT / "kvflow-artifacts/impactkv_m55_v40_task_disjoint_20260805"
M56_MANIFEST = M56_ROOT / "coding_grounded_observation_island_v40/DYNAMIC_MANIFEST.json"
MODEL = Path(
    "/home/gfy/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/"
    "488639f1ff808d1d3d0ba301aef8c11461451ec5"
)
ORIGINAL_MODEL = "/home/gfy/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
DEFAULT_OUTPUT = (
    ROOT / "kvflow-artifacts/impactkv_global_block_attention_20260806/frozen26_r2"
)
V40 = "coding_grounded_observation_island_v40"
PROBE_LAYERS = (0, 8, 17, 26, 35)
QUERY_CHUNK = 64
FORWARD_CHUNK = 512
MIN_ANALYSIS_COPY_TOKENS = 128
GENERATION_MARKER = "<|im_start|>assistant\n"

CATEGORIES = (
    "system_instruction",
    "user_task",
    "compaction_notice",
    "assistant_action",
    "read_observation_path_disjoint",
    "read_observation_path_relevant",
    "copied_observation_island",
    "other_tool_result",
    "generation_marker",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _token_hash(ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _find_sublist(haystack: Sequence[int], needle: Sequence[int]) -> list[int]:
    if not needle or len(needle) > len(haystack):
        return []
    first = needle[0]
    length = len(needle)
    return [
        index
        for index, value in enumerate(haystack[: len(haystack) - length + 1])
        if value == first and list(haystack[index : index + length]) == list(needle)
    ]


def _preview(value: str, limit: int = 100) -> str:
    compact = " ".join(value.replace("/testbed/", "").split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _longest_common_slice(
    source: Sequence[int], target: Sequence[int]
) -> tuple[int, int, int]:
    """Return source offset, target offset, and longest exact common run."""

    blocks = difflib.SequenceMatcher(
        None, list(source), list(target), autojunk=False
    ).get_matching_blocks()
    best = max(blocks, key=lambda row: (row.size, -row.a, -row.b))
    return int(best.a), int(best.b), int(best.size)


def _max_distance_pair(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    if len(rows) < 2:
        raise ValueError("each task requires at least two eligible targets")
    best: tuple[float, str, str, int, int] | None = None
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            dx = float(rows[left]["normalized_log_prompt"]) - float(
                rows[right]["normalized_log_prompt"]
            )
            dy = float(rows[left]["normalized_log_copy"]) - float(
                rows[right]["normalized_log_copy"]
            )
            distance = math.hypot(dx, dy)
            tie_left, tie_right = sorted(
                (str(rows[left]["case_id"]), str(rows[right]["case_id"]))
            )
            candidate = (distance, tie_left, tie_right, left, right)
            if best is None or candidate[:1] > best[:1] or (
                candidate[0] == best[0] and candidate[1:3] < best[1:3]
            ):
                best = candidate
    assert best is not None
    return best[3], best[4]


def _split_blocks_at_span(
    blocks: Sequence[Mapping[str, Any]], start: int, end: int
) -> list[dict[str, Any]]:
    if start < 0 or end <= start:
        raise ValueError("invalid copied span")
    output: list[dict[str, Any]] = []
    for original in blocks:
        left, right = int(original["start"]), int(original["end"])
        cuts = sorted({left, right, *[x for x in (start, end) if left < x < right]})
        for piece_left, piece_right in zip(cuts, cuts[1:]):
            row = dict(original)
            row.update(
                start=piece_left,
                end=piece_right,
                tokens=piece_right - piece_left,
                copied=piece_left >= start and piece_right <= end,
            )
            if row["copied"]:
                row["category"] = "copied_observation_island"
                row["label"] = "copied V40 read observation"
            output.append(row)
    if sum(row["tokens"] for row in output if row["copied"]) != end - start:
        raise ValueError("copied span is not fully covered by blocks")
    return output


def _finalize_blocks(
    blocks: Sequence[Mapping[str, Any]], ids: Sequence[int], prefix: str
) -> list[dict[str, Any]]:
    output = []
    cursor = 0
    for index, source in enumerate(blocks):
        row = dict(source)
        if int(row["start"]) != cursor or int(row["end"]) <= cursor:
            raise ValueError("structural blocks are not a contiguous partition")
        cursor = int(row["end"])
        row["block_id"] = f"{prefix}{index:02d}"
        row["token_hash"] = _token_hash(ids[int(row["start"]) : cursor])
        output.append(row)
    if cursor != len(ids):
        raise ValueError("structural blocks do not cover every prompt token")
    return output


def _map_source_blocks(
    source_blocks: Sequence[Mapping[str, Any]],
    target_blocks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    queues: dict[tuple[str, int, bool], deque[str]] = defaultdict(deque)
    for row in target_blocks:
        key = (str(row["token_hash"]), int(row["tokens"]), bool(row["copied"]))
        queues[key].append(str(row["block_id"]))
    output = []
    for original in source_blocks:
        row = dict(original)
        key = (str(row["token_hash"]), int(row["tokens"]), bool(row["copied"]))
        row["mapped_target_block_id"] = (
            queues[key].popleft() if queues[key] else "source_only_context"
        )
        output.append(row)
    copied_maps = {
        row["mapped_target_block_id"] for row in output if row["copied"]
    }
    if copied_maps == {"source_only_context"}:
        raise ValueError("copied source block did not map to copied target block")
    return output


def _char_regions(
    *,
    planner: Any,
    prompt: str,
    messages: Sequence[Mapping[str, Any]],
    source_paths: Sequence[str],
) -> list[dict[str, Any]]:
    """Make a complete, chronological character partition of one prompt."""

    from benchmark.multi_workflow.coding_reuse_policy import (
        is_successful_readonly_evidence,
        repository_paths,
    )

    if not prompt.endswith(GENERATION_MARKER):
        raise ValueError("rendered prompt lacks the final generation marker")
    marker_start = len(prompt) - len(GENERATION_MARKER)
    message_rows: list[dict[str, Any]] = []
    cursor = 0
    for message_index, message in enumerate(messages):
        if message_index == 0 and message.get("role") == "system":
            continue
        literal = planner._render_message_literal(dict(message))
        start = prompt.find(literal, cursor, marker_start)
        if start < 0:
            raise ValueError(f"message literal {message_index} is absent from prompt")
        if message_rows and start != cursor:
            # Jinja whitespace belongs to the preceding structural message.
            message_rows[-1]["char_end"] = start
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if message_index == 1:
            category = "user_task"
            label = _preview("task: " + content)
            paths: list[str] = []
        elif content.startswith("<history_compaction"):
            category = "compaction_notice"
            label = _preview(content)
            paths = []
        elif role == "assistant":
            category = "assistant_action"
            calls = message.get("tool_calls") or []
            label = _preview(
                "action: " + json.dumps(calls[0], sort_keys=True)
                if calls
                else "reasoning: " + content
            )
            paths = []
        elif role == "tool":
            preceding = messages[message_index - 1] if message_index else {}
            group = [dict(preceding), dict(message)]
            readonly = is_successful_readonly_evidence(group)
            paths = sorted(repository_paths(group))
            relevant = bool(set(paths) & set(source_paths))
            if readonly:
                category = (
                    "read_observation_path_relevant"
                    if relevant
                    else "read_observation_path_disjoint"
                )
                label = _preview(
                    "read observation: " + (", ".join(paths[:2]) or content)
                )
            else:
                category = "other_tool_result"
                label = _preview("other tool result: " + content)
        else:
            category = "other_tool_result"
            label = _preview(f"{role}: {content}")
            paths = []
        message_rows.append(
            {
                "char_start": start,
                "char_end": start + len(literal),
                "category": category,
                "label": label,
                "paths": paths,
                "message_index": message_index,
            }
        )
        cursor = start + len(literal)
    first = message_rows[0]["char_start"] if message_rows else marker_start
    rows = [
        {
            "char_start": 0,
            "char_end": first,
            "category": "system_instruction",
            "label": "system instructions and bash tool schema",
            "paths": [],
            "message_index": 0,
        },
        *message_rows,
    ]
    if rows[-1]["char_end"] != marker_start:
        rows[-1]["char_end"] = marker_start
    rows.append(
        {
            "char_start": marker_start,
            "char_end": len(prompt),
            "category": "generation_marker",
            "label": "next assistant action",
            "paths": [],
            "message_index": len(messages),
        }
    )
    for left, right in zip(rows, rows[1:]):
        if left["char_end"] != right["char_start"]:
            raise ValueError("message character regions are not contiguous")
    return rows


def _token_blocks(
    *,
    planner: Any,
    prompt: str,
    messages: Sequence[Mapping[str, Any]],
    encoding: Any,
    copied_start: int,
    copied_end: int,
    source_paths: Sequence[str],
    prefix: str,
) -> list[dict[str, Any]]:
    regions = _char_regions(
        planner=planner,
        prompt=prompt,
        messages=messages,
        source_paths=source_paths,
    )
    offsets = list(encoding.offsets)
    blocks = []
    token_cursor = 0
    for region_index, region in enumerate(regions):
        if region_index + 1 == len(regions):
            token_end = len(encoding.ids)
        else:
            char_end = int(region["char_end"])
            token_end = token_cursor
            while token_end < len(offsets) and int(offsets[token_end][0]) < char_end:
                token_end += 1
        if token_end > token_cursor:
            blocks.append(
                {
                    "start": token_cursor,
                    "end": token_end,
                    "tokens": token_end - token_cursor,
                    "category": region["category"],
                    "label": region["label"],
                    "paths": region["paths"],
                    "copied": False,
                }
            )
        token_cursor = token_end
    blocks = _split_blocks_at_span(blocks, copied_start, copied_end)
    return _finalize_blocks(blocks, encoding.ids, prefix)


def _analysis_span(
    *,
    source_prompt: str,
    target_prompt: str,
    source_encoding_original: Any,
    original_start: int,
    original_length: int,
    tokenizer: Any,
) -> dict[str, Any]:
    original_end = original_start + original_length
    offsets = source_encoding_original.offsets
    char_start = int(offsets[original_start][0])
    char_end = int(offsets[original_end - 1][1])
    segment_text = source_prompt[char_start:char_end]
    if source_prompt.count(segment_text) != 1 or target_prompt.count(segment_text) != 1:
        raise ValueError("copied text is not unique in both prompts")
    target_char_start = target_prompt.index(segment_text)
    target_char_end = target_char_start + len(segment_text)
    source_encoding = tokenizer.encode(source_prompt, add_special_tokens=False)
    target_encoding = tokenizer.encode(target_prompt, add_special_tokens=False)
    source_positions = [
        index
        for index, (left, right) in enumerate(source_encoding.offsets)
        if int(right) > char_start and int(left) < char_end
    ]
    target_positions = [
        index
        for index, (left, right) in enumerate(target_encoding.offsets)
        if int(right) > target_char_start and int(left) < target_char_end
    ]
    if not source_positions or not target_positions:
        raise ValueError("copied text has no analysis-model tokens")
    source_slice = [source_encoding.ids[index] for index in source_positions]
    target_slice = [target_encoding.ids[index] for index in target_positions]
    trimmed_source = trimmed_target = 0
    if source_slice != target_slice:
        source_offset, target_offset, common = _longest_common_slice(
            source_slice, target_slice
        )
        if common < MIN_ANALYSIS_COPY_TOKENS:
            raise ValueError("retokenized copied span has no 128-token common run")
        trimmed_source = len(source_slice) - common
        trimmed_target = len(target_slice) - common
        source_positions = source_positions[source_offset : source_offset + common]
        target_positions = target_positions[target_offset : target_offset + common]
        source_slice = source_slice[source_offset : source_offset + common]
        target_slice = target_slice[target_offset : target_offset + common]
    if source_slice != target_slice:
        raise AssertionError("analysis source/target token slices differ")
    source_start, target_start = source_positions[0], target_positions[0]
    length = len(source_slice)
    if length < MIN_ANALYSIS_COPY_TOKENS:
        raise ValueError("retokenized copied span is below the 128-token minimum")
    if (
        source_start <= 0
        or source_start + length >= len(source_encoding.ids)
        or target_start <= 0
        or target_start + length >= len(target_encoding.ids)
    ):
        raise ValueError("retokenized copied span is not strictly middle")
    if _find_sublist(source_encoding.ids, source_slice) != [source_start]:
        raise ValueError("analysis copied tokens are not unique in source")
    if _find_sublist(target_encoding.ids, target_slice) != [target_start]:
        raise ValueError("analysis copied tokens are not unique in target")
    return {
        "source_encoding": source_encoding,
        "target_encoding": target_encoding,
        "source_start": source_start,
        "target_start": target_start,
        "length": length,
        "segment_token_hash": _token_hash(source_slice),
        "segment_text_sha256": hashlib.sha256(segment_text.encode()).hexdigest(),
        "original_segment_chars": len(segment_text),
        "retokenization_trimmed_source_tokens": trimmed_source,
        "retokenization_trimmed_target_tokens": trimmed_target,
    }


def _candidate_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rebuild every M56 V40 target and retain analysis-model-valid cases."""

    from minisweagent.models.utils.actions_toolcall import BASH_TOOL
    from tokenizers import Tokenizer

    from benchmark.multi_workflow import run_frozen_trajectory_replay_v18 as replay
    from benchmark.multi_workflow.run_m56_v40_same_prompt_replay import (
        FRESH_ROOT,
        _trajectory_paths,
    )

    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    manifest = json.loads(M56_MANIFEST.read_text())
    manifest_cases = {str(row["case_id"]): row for row in manifest["cases"]}
    eligible: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    # M56 used one planner for the whole task sequence.  Its session counter is
    # therefore part of every manifest case_id and must advance identically.
    planner = replay.make_planner(
        arm=V40,
        manifest_path=None,
        client_ledger_path=None,
        instance_nonce="global-block-attention-prepare",
    )
    for instance_id, trajectory_path in _trajectory_paths(FRESH_ROOT).items():
        trajectory = replay.read_json(trajectory_path)
        replay.reset_planner_session(planner, instance_id=instance_id)
        sources: dict[str, dict[str, Any]] = {}
        for request_index, prefix in enumerate(
            replay.assistant_request_prefixes(trajectory["messages"]), start=1
        ):
            rolling_messages, _, rolling = planner._rolling_messages(prefix)
            compacted_messages, compaction = planner.compact_messages(
                rolling_messages
            )
            prompt = planner._chat_template.render(
                messages=planner._template_messages(compacted_messages),
                tools=[BASH_TOOL],
                add_generation_prompt=True,
            )
            previous_pending = copy.deepcopy(planner._pending_source)
            planned = replay.plan_request(planner, prefix)
            if planned["target"] is not None:
                case_id = str(planned["target"]["case_id"])
                try:
                    runtime_case = manifest_cases[case_id]
                    for field in (
                        "source_id",
                        "source_start",
                        "target_start",
                        "length",
                        "source_prompt_hash",
                        "target_prompt_hash",
                        "segment_token_hash",
                    ):
                        if runtime_case[field] != planned["target"][field]:
                            raise ValueError(f"M56 manifest mismatch in {field}")
                    if previous_pending is None:
                        raise ValueError("target has no pending source handle")
                    source_id = str(planned["target"]["source_id"])
                    source = sources[source_id]
                    original_source_encoding = planner._tokenizer.encode(
                        source["prompt"], add_special_tokens=False
                    )
                    analysis = _analysis_span(
                        source_prompt=source["prompt"],
                        target_prompt=prompt,
                        source_encoding_original=original_source_encoding,
                        original_start=int(planned["target"]["source_start"]),
                        original_length=int(planned["target"]["length"]),
                        tokenizer=tokenizer,
                    )
                    metadata_source_paths = sorted(
                        {
                            path
                            for message in source["messages"]
                            for path in (
                                message.get("extra", {}).get("repository_paths")
                                or []
                            )
                        }
                    )
                    preliminary_source_blocks = _token_blocks(
                        planner=planner,
                        prompt=source["prompt"],
                        messages=source["messages"],
                        encoding=analysis["source_encoding"],
                        copied_start=analysis["source_start"],
                        copied_end=analysis["source_start"] + analysis["length"],
                        source_paths=metadata_source_paths,
                        prefix="s",
                    )
                    # V40 pending handles intentionally do not persist path
                    # metadata.  Recover the selected observation's paths from
                    # the actual copied tool block rather than labeling every
                    # other read as path-disjoint.
                    copied_block_paths = sorted(
                        {
                            path
                            for block in preliminary_source_blocks
                            if block["copied"]
                            for path in block["paths"]
                        }
                    )
                    source_paths = copied_block_paths or metadata_source_paths
                    source_blocks = _token_blocks(
                        planner=planner,
                        prompt=source["prompt"],
                        messages=source["messages"],
                        encoding=analysis["source_encoding"],
                        copied_start=analysis["source_start"],
                        copied_end=analysis["source_start"] + analysis["length"],
                        source_paths=source_paths,
                        prefix="s",
                    )
                    target_blocks = _token_blocks(
                        planner=planner,
                        prompt=prompt,
                        messages=compacted_messages,
                        encoding=analysis["target_encoding"],
                        copied_start=analysis["target_start"],
                        copied_end=analysis["target_start"] + analysis["length"],
                        source_paths=source_paths,
                        prefix="t",
                    )
                    source_blocks = _map_source_blocks(source_blocks, target_blocks)
                    eligible.append(
                        {
                            "case_id": case_id,
                            "instance_id": instance_id,
                            "request_index": request_index,
                            "trajectory_path": str(trajectory_path),
                            "trajectory_sha256": _sha256(trajectory_path),
                            "source_input_ids": list(analysis["source_encoding"].ids),
                            "target_input_ids": list(analysis["target_encoding"].ids),
                            "source_start": analysis["source_start"],
                            "target_start": analysis["target_start"],
                            "length": analysis["length"],
                            "segment_token_hash": analysis["segment_token_hash"],
                            "segment_text_sha256": analysis["segment_text_sha256"],
                            "source_blocks": source_blocks,
                            "target_blocks": target_blocks,
                            "source_compaction": source["compaction"],
                            "target_compaction": compaction,
                            "source_rolling": source["rolling"],
                            "target_rolling": rolling,
                            "analysis_prompt_tokens": len(
                                analysis["target_encoding"].ids
                            ),
                            "analysis_source_tokens": len(
                                analysis["source_encoding"].ids
                            ),
                            "analysis_copied_tokens": analysis["length"],
                            "retokenization_trimmed_source_tokens": analysis[
                                "retokenization_trimmed_source_tokens"
                            ],
                            "retokenization_trimmed_target_tokens": analysis[
                                "retokenization_trimmed_target_tokens"
                            ],
                            "original_m56": {
                                key: runtime_case[key]
                                for key in (
                                    "case_id",
                                    "source_id",
                                    "source_start",
                                    "target_start",
                                    "length",
                                    "source_prompt_hash",
                                    "target_prompt_hash",
                                    "segment_token_hash",
                                )
                            },
                        }
                    )
                except Exception as error:
                    dropped.append(
                        {
                            "case_id": case_id,
                            "instance_id": instance_id,
                            "request_index": request_index,
                            "reason": f"{type(error).__name__}: {error}",
                        }
                    )
            if planned["source"] is not None:
                pending = copy.deepcopy(planner._pending_source)
                if pending is None:
                    raise AssertionError("registered source lacks pending handle")
                sources[str(pending["source_id"])] = {
                    "prompt": prompt,
                    "messages": compacted_messages,
                    "compaction": compaction,
                    "rolling": rolling,
                }
    return eligible, dropped


def _select_cohort(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompt_logs = [math.log1p(row["analysis_prompt_tokens"]) for row in rows]
    copy_logs = [math.log1p(row["analysis_copied_tokens"]) for row in rows]

    def normalize(value: float, values: Sequence[float]) -> float:
        width = max(values) - min(values)
        return 0.0 if width == 0 else (value - min(values)) / width

    for row, prompt_log, copy_log in zip(rows, prompt_logs, copy_logs, strict=True):
        row["normalized_log_prompt"] = normalize(prompt_log, prompt_logs)
        row["normalized_log_copy"] = normalize(copy_log, copy_logs)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["instance_id"])].append(row)
    selected = []
    for task in sorted(by_task):
        task_rows = sorted(by_task[task], key=lambda row: str(row["case_id"]))
        left, right = _max_distance_pair(task_rows)
        selected.extend((task_rows[left], task_rows[right]))
    return sorted(selected, key=lambda row: (row["instance_id"], row["request_index"]))


def _representative_case_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    median_copy = statistics.median(row["analysis_copied_tokens"] for row in rows)
    ranked = [
        sorted(rows, key=lambda row: (row["analysis_copied_tokens"], row["case_id"])),
        sorted(
            rows,
            key=lambda row: (
                abs(row["analysis_copied_tokens"] - median_copy),
                row["case_id"],
            ),
        ),
        sorted(
            rows,
            key=lambda row: (-row["analysis_copied_tokens"], row["case_id"]),
        ),
    ]
    selected, tasks = [], set()
    for candidates in ranked:
        row = next(row for row in candidates if row["instance_id"] not in tasks)
        selected.append(str(row["case_id"]))
        tasks.add(str(row["instance_id"]))
    return selected


def prepare(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    eligible, dropped = _candidate_records()
    selected = _select_cohort(eligible)
    if len(selected) != 26 or len({row["instance_id"] for row in selected}) != 13:
        raise ValueError("frozen cohort must contain 2 targets for each of 13 tasks")
    canary = []
    canary_tasks = set()
    for row in sorted(selected, key=lambda row: _stable_hash(str(row["case_id"]))):
        if row["instance_id"] not in canary_tasks:
            canary.append(str(row["case_id"]))
            canary_tasks.add(str(row["instance_id"]))
        if len(canary) == 6:
            break
    representatives = _representative_case_ids(selected)
    output.mkdir(parents=True)
    design_path = output / "DESIGN.json"
    _write_json(
        design_path,
        {
            "cases": selected,
            "analysis_model": str(MODEL),
            "original_m56_model": ORIGINAL_MODEL,
            "probe_layers": list(PROBE_LAYERS),
            "query_chunk_tokens": QUERY_CHUNK,
            "forward_chunk_tokens": FORWARD_CHUNK,
            "canary_case_ids": canary,
            "representative_case_ids": representatives,
            "eligible_targets": len(eligible),
            "dropped_targets": dropped,
        },
    )
    capacity = {
        "selected_cases": len(selected),
        "selected_tasks": len({row["instance_id"] for row in selected}),
        "eligible_targets": len(eligible),
        "dropped_targets": len(dropped),
        "canary_cases": len(canary),
        "canary_tasks": len(canary_tasks),
        "analysis_prompt_tokens_min": min(row["analysis_prompt_tokens"] for row in selected),
        "analysis_prompt_tokens_median": statistics.median(
            row["analysis_prompt_tokens"] for row in selected
        ),
        "analysis_prompt_tokens_max": max(row["analysis_prompt_tokens"] for row in selected),
        "analysis_copied_tokens_min": min(row["analysis_copied_tokens"] for row in selected),
        "analysis_copied_tokens_median": statistics.median(
            row["analysis_copied_tokens"] for row in selected
        ),
        "analysis_copied_tokens_max": max(row["analysis_copied_tokens"] for row in selected),
    }
    registration = {
        "status": "REGISTERED_BEFORE_GPU",
        "purpose": (
            "Compare full structural-block attention under Dense execution and "
            "runtime-faithful single-island V40 lossy KV reuse."
        ),
        "design_sha256": _sha256(design_path),
        "source_m56_manifest": str(M56_MANIFEST),
        "source_m56_manifest_sha256": _sha256(M56_MANIFEST),
        "analysis_model": str(MODEL),
        "original_m56_model": ORIGINAL_MODEL,
        "model_scope_warning": (
            "The M56 request text and V40 source/target policy are preserved, but "
            "attention is measured with a Qwen2.5-Coder-3B mechanism proxy. The "
            "packed 30B MoE AWQ checkpoint cannot be loaded by the available "
            "Transformers stack on the 24 GB GPU. This is not native-SGLang 30B attention."
        ),
        "revision_reason": (
            "R2 recovers copied-observation paths from the parsed copied tool "
            "block. R1 relied on absent trajectory extra.repository_paths and "
            "therefore mislabeled all non-copied reads as path-disjoint; token "
            "spans, cohort selection, splice execution, and gates are unchanged."
        ),
        "selection": (
            "For each of 13 tasks, choose the pair with maximum Euclidean distance "
            "in globally normalized log1p(prompt tokens), log1p(copy tokens) space."
        ),
        "probe_layers_zero_based": list(PROBE_LAYERS),
        "query_contract": {
            "all_query_tokens": True,
            "query_chunk_tokens": QUERY_CHUNK,
            "forward_chunk_tokens": FORWARD_CHUNK,
            "persistent_token_attention_matrix": False,
        },
        "semantics": {
            "dense_rows": "all target structural blocks",
            "reuse_prefix_rows": "copied exactly from Dense as a negative control",
            "reuse_copied_rows": "N/A because target-time attention is not executed",
            "reuse_suffix_rows": "measured from the hybrid V40 cache",
            "copied_formation_rows": (
                "source-time copied-query attention, mapped to matching target blocks "
                "plus source_only_context"
            ),
        },
        "mechanical_canary_gates": {
            "cases": 6,
            "distinct_tasks": 6,
            "all_tokens_block_covered": True,
            "source_target_segment_token_identical": True,
            "attention_row_sum_abs_error_max": 1e-4,
            "prefix_matrix_abs_delta_max": 1e-7,
            "instrumented_reference_logit_abs_delta_max": 1e-5,
            "finite_no_oom": True,
            "result_direction_gate": False,
        },
        "frozen_canary_case_ids": canary,
        "frozen_representative_case_ids": representatives,
        "capacity": capacity,
        "protected": {
            "paper_modified": False,
            "prefetch_modified": False,
            "old_dirty_checkout_modified": False,
            "old_preregistration_thresholds_modified": False,
        },
    }
    _write_json(output / "CAPACITY.json", capacity)
    _write_json(output / "REGISTRATION.json", registration)
    return registration


def _layers(cache: Any) -> list[tuple[Any, Any]]:
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return [(row[0], row[1]) for row in cache]


def _cpu_cache(cache: Any, torch: Any) -> list[tuple[Any, Any]]:
    return [
        (
            key[0].detach().to("cpu", dtype=torch.bfloat16).contiguous(),
            value[0].detach().to("cpu", dtype=torch.bfloat16).contiguous(),
        )
        for key, value in _layers(cache)
    ]


def _rotate_half(value: Any, torch: Any) -> Any:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def _rope_shift(keys: Any, delta: int, theta: float, torch: Any) -> Any:
    if delta == 0 or not keys.numel():
        return keys
    dim = keys.shape[-1]
    inv = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, device=keys.device, dtype=torch.float32)
            / dim
        )
    )
    frequency = delta * inv
    cosine = torch.cat((frequency.cos(), frequency.cos()))
    sine = torch.cat((frequency.sin(), frequency.sin()))
    return (
        keys.float() * cosine + _rotate_half(keys.float(), torch) * sine
    ).to(keys.dtype)


def _model_theta(config: Any) -> float:
    if hasattr(config, "rope_theta"):
        return float(config.rope_theta)
    parameters = getattr(config, "rope_parameters", None) or {}
    return float(parameters.get("rope_theta", 1_000_000.0))


def _attention_block_rows(
    *,
    model: Any,
    layer_index: int,
    hidden: Any,
    key: Any,
    global_start: int,
    query_blocks: Sequence[Mapping[str, Any]],
    key_blocks: Sequence[Mapping[str, Any]],
    torch: Any,
) -> dict[str, dict[str, float]]:
    """Sum attention for all query tokens present in this hidden-state chunk."""

    attention = model.model.layers[layer_index].self_attn
    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    groups = num_heads // num_kv_heads
    head_dim = int(getattr(model.config, "head_dim", 0)) or (
        int(model.config.hidden_size) // num_heads
    )
    output: dict[str, dict[str, float]] = {}
    hidden_end = global_start + hidden.shape[1]
    for block in query_blocks:
        left = max(int(block["start"]), global_start)
        right = min(int(block["end"]), hidden_end)
        if right <= left:
            continue
        sums = {str(row["block_id"]): 0.0 for row in key_blocks}
        for query_left in range(left, right, QUERY_CHUNK):
            query_right = min(right, query_left + QUERY_CHUNK)
            local_left, local_right = query_left - global_start, query_right - global_start
            query_hidden = hidden[:, local_left:local_right]
            positions = torch.arange(
                query_left, query_right, device="cuda", dtype=torch.long
            ).unsqueeze(0)
            query = attention.q_proj(query_hidden).view(
                1, query_right - query_left, num_heads, head_dim
            ).transpose(1, 2)
            if hasattr(attention, "q_norm"):
                query = attention.q_norm(query)
            cosine, sine = model.model.rotary_emb(query_hidden, positions)
            query = query * cosine.unsqueeze(1) + _rotate_half(query, torch) * sine.unsqueeze(1)
            expanded_key = key.repeat_interleave(groups, dim=1)
            scores = torch.matmul(query.float(), expanded_key.float().transpose(-1, -2))
            scores /= math.sqrt(head_dim)
            key_positions = torch.arange(key.shape[-2], device="cuda")
            causal = key_positions.view(1, 1, 1, -1) > positions.view(1, 1, -1, 1)
            weights = torch.softmax(scores.masked_fill(causal, -torch.inf), dim=-1)
            mean_weights = weights.mean(dim=1)[0]
            for key_block in key_blocks:
                key_left = int(key_block["start"])
                key_right = min(int(key_block["end"]), key.shape[-2])
                if key_right > key_left:
                    sums[str(key_block["block_id"])] += float(
                        mean_weights[:, key_left:key_right].sum().item()
                    )
            del query_hidden, positions, query, expanded_key, scores, causal, weights, mean_weights
        output[str(block["block_id"])] = sums
    return output


def _merge_row_sums(
    accumulator: dict[str, dict[str, float]],
    additions: Mapping[str, Mapping[str, float]],
) -> None:
    for row_id, masses in additions.items():
        row = accumulator.setdefault(str(row_id), defaultdict(float))
        for key_id, value in masses.items():
            row[str(key_id)] += float(value)


def _normalize_rows(
    sums: Mapping[str, Mapping[str, float]],
    blocks_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, float]], float]:
    rows = {}
    max_error = 0.0
    for row_id, masses in sums.items():
        tokens = int(blocks_by_id[row_id]["tokens"])
        normalized = {key: float(value) / tokens for key, value in masses.items()}
        error = abs(sum(normalized.values()) - 1.0)
        max_error = max(max_error, error)
        rows[row_id] = normalized
    return rows, max_error


def _register_hooks(model: Any) -> tuple[dict[int, Any], list[Any]]:
    captured: dict[int, Any] = {}
    handles = []
    for layer_index in PROBE_LAYERS:
        def capture(_module: Any, args: tuple[Any, ...], index: int = layer_index) -> None:
            captured[index] = args[0].detach()

        handles.append(model.model.layers[layer_index].register_forward_pre_hook(capture))
    return captured, handles


def _dense_profile(
    *,
    model: Any,
    ids: Sequence[int],
    blocks: Sequence[Mapping[str, Any]],
    query_block_ids: set[str] | None,
    torch: Any,
) -> tuple[list[tuple[Any, Any]], Any, dict[str, dict[str, dict[str, float]]], float]:
    captured, handles = _register_hooks(model)
    inputs = torch.tensor([ids], device="cuda", dtype=torch.long)
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
    if set(captured) != set(PROBE_LAYERS):
        raise RuntimeError("not all probe-layer hidden states were captured")
    cache_layers = _layers(output.past_key_values)
    query_blocks = [
        row for row in blocks
        if query_block_ids is None or str(row["block_id"]) in query_block_ids
    ]
    blocks_by_id = {str(row["block_id"]): row for row in blocks}
    matrices = {}
    max_error = 0.0
    for layer_index in PROBE_LAYERS:
        sums = _attention_block_rows(
            model=model,
            layer_index=layer_index,
            hidden=captured[layer_index],
            key=cache_layers[layer_index][0],
            global_start=0,
            query_blocks=query_blocks,
            key_blocks=blocks,
            torch=torch,
        )
        rows, error = _normalize_rows(sums, blocks_by_id)
        matrices[str(layer_index)] = rows
        max_error = max(max_error, error)
    cache = _cpu_cache(output.past_key_values, torch)
    logits = output.logits[0, -1].detach().float().cpu()
    del output, inputs, captured, cache_layers
    gc.collect()
    torch.cuda.empty_cache()
    return cache, logits, matrices, max_error


def _cache_from_prefix(model: Any, target_cache: Sequence[tuple[Any, Any]], length: int, torch: Any) -> Any:
    from transformers.cache_utils import DynamicCache

    return DynamicCache(
        [
            (
                key[:, :length].unsqueeze(0).to("cuda"),
                value[:, :length].unsqueeze(0).to("cuda"),
            )
            for key, value in target_cache
        ],
        config=model.config,
    )


def _append_island(
    *,
    model: Any,
    cache: Any,
    source_cache: Sequence[tuple[Any, Any]],
    source_start: int,
    target_start: int,
    length: int,
    theta: float,
    torch: Any,
) -> Any:
    from transformers.cache_utils import DynamicCache

    layers = []
    delta = target_start - source_start
    for (target_key, target_value), (source_key, source_value) in zip(
        _layers(cache), source_cache, strict=True
    ):
        copied_key = _rope_shift(
            source_key[:, source_start : source_start + length].to("cuda"),
            delta,
            theta,
            torch,
        ).unsqueeze(0)
        copied_value = source_value[:, source_start : source_start + length].to("cuda").unsqueeze(0)
        layers.append(
            (
                torch.cat((target_key, copied_key), dim=2),
                torch.cat((target_value, copied_value), dim=2),
            )
        )
    return DynamicCache(layers, config=model.config)


def _reuse_profile(
    *,
    model: Any,
    case: Mapping[str, Any],
    target_cache: Sequence[tuple[Any, Any]],
    source_cache: Sequence[tuple[Any, Any]],
    dense_matrix: Mapping[str, Any],
    theta: float,
    torch: Any,
) -> tuple[Any, dict[str, Any], float, float]:
    target_start = int(case["target_start"])
    copy_end = target_start + int(case["length"])
    blocks = case["target_blocks"]
    blocks_by_id = {str(row["block_id"]): row for row in blocks}
    cache = _cache_from_prefix(model, target_cache, target_start, torch)
    cache = _append_island(
        model=model,
        cache=cache,
        source_cache=source_cache,
        source_start=int(case["source_start"]),
        target_start=target_start,
        length=int(case["length"]),
        theta=theta,
        torch=torch,
    )
    suffix_blocks = [row for row in blocks if int(row["start"]) >= copy_end]
    sums_by_layer: dict[str, dict[str, dict[str, float]]] = {
        str(layer): {} for layer in PROBE_LAYERS
    }
    logits = None
    for offset in range(copy_end, len(case["target_input_ids"]), FORWARD_CHUNK):
        token_ids = case["target_input_ids"][offset : offset + FORWARD_CHUNK]
        captured, handles = _register_hooks(model)
        try:
            output = model(
                input_ids=torch.tensor([token_ids], device="cuda", dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        finally:
            for handle in handles:
                handle.remove()
        cache = output.past_key_values
        logits = output.logits[0, -1].detach().float().cpu()
        cache_layers = _layers(cache)
        for layer_index in PROBE_LAYERS:
            additions = _attention_block_rows(
                model=model,
                layer_index=layer_index,
                hidden=captured[layer_index],
                key=cache_layers[layer_index][0],
                global_start=offset,
                query_blocks=suffix_blocks,
                key_blocks=blocks,
                torch=torch,
            )
            _merge_row_sums(sums_by_layer[str(layer_index)], additions)
        del output, captured, cache_layers
    if logits is None:
        raise RuntimeError("reuse suffix did not execute")
    matrices: dict[str, Any] = {}
    max_error = 0.0
    prefix_delta = 0.0
    for layer_index in PROBE_LAYERS:
        layer = str(layer_index)
        suffix_rows, error = _normalize_rows(sums_by_layer[layer], blocks_by_id)
        max_error = max(max_error, error)
        rows: dict[str, Any] = {}
        for block in blocks:
            row_id = str(block["block_id"])
            if int(block["end"]) <= target_start:
                rows[row_id] = dense_matrix[layer][row_id]
                prefix_delta = max(
                    prefix_delta,
                    max(
                        abs(rows[row_id][key] - dense_matrix[layer][row_id][key])
                        for key in rows[row_id]
                    ),
                )
            elif bool(block["copied"]):
                rows[row_id] = None
            else:
                rows[row_id] = suffix_rows[row_id]
        matrices[layer] = rows
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    return logits, matrices, max_error, prefix_delta


def _reference_reuse_logits(
    *,
    model: Any,
    case: Mapping[str, Any],
    target_cache: Sequence[tuple[Any, Any]],
    source_cache: Sequence[tuple[Any, Any]],
    theta: float,
    torch: Any,
) -> Any:
    start = int(case["target_start"])
    end = start + int(case["length"])
    cache = _cache_from_prefix(model, target_cache, start, torch)
    cache = _append_island(
        model=model,
        cache=cache,
        source_cache=source_cache,
        source_start=int(case["source_start"]),
        target_start=start,
        length=int(case["length"]),
        theta=theta,
        torch=torch,
    )
    logits = None
    for offset in range(end, len(case["target_input_ids"]), FORWARD_CHUNK):
        output = model(
            input_ids=torch.tensor(
                [case["target_input_ids"][offset : offset + FORWARD_CHUNK]],
                device="cuda",
                dtype=torch.long,
            ),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        logits = output.logits[0, -1].detach().float().cpu()
        del output
    if logits is None:
        raise RuntimeError("reference reuse suffix did not execute")
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    return logits


def _distribution_metrics(
    left: Mapping[str, float], right: Mapping[str, float]
) -> dict[str, Any]:
    keys = sorted(set(left) | set(right))
    p = [float(left.get(key, 0.0)) for key in keys]
    q = [float(right.get(key, 0.0)) for key in keys]
    tv = 0.5 * sum(abs(a - b) for a, b in zip(p, q, strict=True))
    js = 0.0
    for a, b in zip(p, q, strict=True):
        middle = 0.5 * (a + b)
        if a > 0:
            js += 0.5 * a * math.log(a / middle)
        if b > 0:
            js += 0.5 * b * math.log(b / middle)
    return {
        "tv": tv,
        "js_nats": js,
        "top_block_agreement": max(left, key=left.get) == max(right, key=right.get),
        "left_top_block": max(left, key=left.get),
        "right_top_block": max(right, key=right.get),
    }


def _weighted_distribution(
    matrix: Mapping[str, Mapping[str, float] | None],
    row_blocks: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    valid = [row for row in row_blocks if matrix.get(str(row["block_id"])) is not None]
    total = sum(int(row["tokens"]) for row in valid)
    if total == 0:
        return {}
    output: dict[str, float] = defaultdict(float)
    for row in valid:
        weight = int(row["tokens"]) / total
        for key, value in matrix[str(row["block_id"])].items():
            output[key] += weight * float(value)
    return dict(output)


def _softmax_js(left: Any, right: Any, torch: Any) -> float:
    p = torch.softmax(left.float(), dim=-1)
    q = torch.softmax(right.float(), dim=-1)
    middle = 0.5 * (p + q)
    value = 0.5 * (
        (p * (p.clamp_min(1e-30).log() - middle.clamp_min(1e-30).log())).sum()
        + (q * (q.clamp_min(1e-30).log() - middle.clamp_min(1e-30).log())).sum()
    )
    return float(value)


def _case_metrics(
    *,
    case: Mapping[str, Any],
    dense: Mapping[str, Any],
    reuse: Mapping[str, Any],
    source_formation: Mapping[str, Any],
) -> dict[str, Any]:
    target_blocks = case["target_blocks"]
    source_blocks = case["source_blocks"]
    suffix = [
        row
        for row in target_blocks
        if int(row["start"]) >= int(case["target_start"]) + int(case["length"])
    ]
    copied_target = [row for row in target_blocks if row["copied"]]
    copied_source = [row for row in source_blocks if row["copied"]]
    generation = next(row for row in target_blocks if row["category"] == "generation_marker")
    copied_key_ids = {str(row["block_id"]) for row in copied_target}
    layers = []
    for layer_index in PROBE_LAYERS:
        layer = str(layer_index)
        dense_suffix = _weighted_distribution(dense[layer], suffix)
        reuse_suffix = _weighted_distribution(reuse[layer], suffix)
        suffix_metrics = _distribution_metrics(dense_suffix, reuse_suffix)
        suffix_metrics["dense_mass_to_copied_island"] = sum(
            dense_suffix.get(key, 0.0) for key in copied_key_ids
        )
        suffix_metrics["reuse_mass_to_copied_island"] = sum(
            reuse_suffix.get(key, 0.0) for key in copied_key_ids
        )
        suffix_metrics["copied_island_mass_delta"] = (
            suffix_metrics["reuse_mass_to_copied_island"]
            - suffix_metrics["dense_mass_to_copied_island"]
        )
        generation_metrics = _distribution_metrics(
            dense[layer][generation["block_id"]],
            reuse[layer][generation["block_id"]],
        )
        dense_copied = _weighted_distribution(dense[layer], copied_target)
        source_raw = _weighted_distribution(source_formation[layer], copied_source)
        mapped_source: dict[str, float] = defaultdict(float)
        source_by_id = {str(row["block_id"]): row for row in source_blocks}
        for key_id, value in source_raw.items():
            mapped = str(source_by_id[key_id]["mapped_target_block_id"])
            mapped_source[mapped] += value
        dense_copied["source_only_context"] = 0.0
        formation_metrics = _distribution_metrics(dense_copied, mapped_source)
        formation_metrics["source_only_context_mass"] = mapped_source.get(
            "source_only_context", 0.0
        )
        layers.append(
            {
                "layer": layer_index,
                "recomputed_suffix": suffix_metrics,
                "generation_marker": generation_metrics,
                "copied_row_source_time_vs_dense_target": formation_metrics,
            }
        )
    return {"layers": layers}


def _measurement_complete(row: Mapping[str, Any]) -> bool:
    return row.get("status") == "ok" and len(row.get("metrics", {}).get("layers", [])) == len(PROBE_LAYERS)


def measure(output: Path, max_cases: int) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    # The probe reconstructs attention from captured hidden states and never
    # backpropagates.  Leaving autograd enabled retains every 4k--11k-token
    # layer graph and can consume the entire 24 GB card before block reduction.
    torch.set_grad_enabled(False)

    design_path = output / "DESIGN.json"
    registration = json.loads((output / "REGISTRATION.json").read_text())
    if registration["design_sha256"] != _sha256(design_path):
        raise ValueError("design changed after preregistration")
    design = json.loads(design_path.read_text())
    cases = design["cases"]
    if max_cases > 0:
        wanted = set(design["canary_case_ids"][:max_cases])
        cases = [row for row in cases if row["case_id"] in wanted]
    observations_path = output / "OBSERVATIONS.jsonl"
    completed = set()
    if observations_path.exists():
        for line in observations_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if _measurement_complete(row):
                    completed.add(str(row["case_id"]))
    pending = [row for row in cases if row["case_id"] not in completed]
    if not pending:
        return {
            "status": "COMPLETE",
            "selected_cases": len(cases),
            "completed_cases": len(completed),
            "new_cases": 0,
        }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU substitution is forbidden")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda").eval()
    if len(model.model.layers) != 36:
        raise ValueError("analysis model layer count changed")
    theta = _model_theta(model.config)
    written = 0
    errors = []
    for index, case in enumerate(pending, 1):
        try:
            copied_source_ids = case["source_input_ids"][
                case["source_start"] : case["source_start"] + case["length"]
            ]
            copied_target_ids = case["target_input_ids"][
                case["target_start"] : case["target_start"] + case["length"]
            ]
            if copied_source_ids != copied_target_ids:
                raise ValueError("source/target copied tokens changed after registration")
            copied_source_block_ids = {
                str(row["block_id"]) for row in case["source_blocks"] if row["copied"]
            }
            source_cache, _, source_matrix, source_error = _dense_profile(
                model=model,
                ids=case["source_input_ids"],
                blocks=case["source_blocks"],
                query_block_ids=copied_source_block_ids,
                torch=torch,
            )
            target_cache, dense_logits, dense_matrix, dense_error = _dense_profile(
                model=model,
                ids=case["target_input_ids"],
                blocks=case["target_blocks"],
                query_block_ids=None,
                torch=torch,
            )
            reuse_logits, reuse_matrix, reuse_error, prefix_delta = _reuse_profile(
                model=model,
                case=case,
                target_cache=target_cache,
                source_cache=source_cache,
                dense_matrix=dense_matrix,
                theta=theta,
                torch=torch,
            )
            reference_logits = _reference_reuse_logits(
                model=model,
                case=case,
                target_cache=target_cache,
                source_cache=source_cache,
                theta=theta,
                torch=torch,
            )
            logit_delta = float((reuse_logits - reference_logits).abs().max())
            max_row_error = max(source_error, dense_error, reuse_error)
            row = {
                "status": "ok",
                "case_id": case["case_id"],
                "instance_id": case["instance_id"],
                "request_index": case["request_index"],
                "target_tokens": len(case["target_input_ids"]),
                "source_tokens": len(case["source_input_ids"]),
                "copied_tokens": case["length"],
                "source_start": case["source_start"],
                "target_start": case["target_start"],
                "dense_matrix": dense_matrix,
                "reuse_matrix": reuse_matrix,
                "source_formation_matrix": source_matrix,
                "metrics": _case_metrics(
                    case=case,
                    dense=dense_matrix,
                    reuse=reuse_matrix,
                    source_formation=source_matrix,
                ),
                "mechanical_checks": {
                    "attention_row_sum_abs_error_max": max_row_error,
                    "prefix_matrix_abs_delta_max": prefix_delta,
                    "instrumented_reference_logit_abs_delta_max": logit_delta,
                    "dense_reuse_final_logit_js": _softmax_js(
                        dense_logits, reuse_logits, torch
                    ),
                    "dense_reuse_top1_changed": int(dense_logits.argmax())
                    != int(reuse_logits.argmax()),
                    "dense_top1_token_id": int(dense_logits.argmax()),
                    "reuse_top1_token_id": int(reuse_logits.argmax()),
                },
            }
            if max_row_error > 1e-4:
                raise RuntimeError(f"attention row error {max_row_error} exceeds gate")
            if prefix_delta > 1e-7:
                raise RuntimeError(f"prefix matrix delta {prefix_delta} exceeds gate")
            if logit_delta > 1e-5:
                raise RuntimeError(f"instrumented/reference delta {logit_delta} exceeds gate")
            if not _measurement_complete(row):
                raise RuntimeError("case output is incomplete")
            with observations_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
            print(
                json.dumps(
                    {
                        "case": index,
                        "pending": len(pending),
                        "case_id": case["case_id"],
                        "target_tokens": len(case["target_input_ids"]),
                        "copied_tokens": case["length"],
                        "dense_reuse_logit_js": row["mechanical_checks"][
                            "dense_reuse_final_logit_js"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del source_cache, target_cache, dense_logits, reuse_logits, reference_logits
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as error:
            errors.append(
                {
                    "case_id": case["case_id"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(json.dumps(errors[-1], sort_keys=True), flush=True)
            break
    status = {
        "status": "COMPLETE" if not errors and written == len(pending) else "PARTIAL",
        "selected_cases": len(cases),
        "previously_completed_cases": len(completed),
        "new_cases": written,
        "errors": errors,
        "analysis_model": str(MODEL),
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
    }
    _write_json(output / "MEASUREMENT_STATUS.json", status)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return status


def _median_iqr(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "median": statistics.median(ordered),
        "q25": ordered[max(0, math.floor(0.25 * (len(ordered) - 1)))],
        "q75": ordered[min(len(ordered) - 1, math.ceil(0.75 * (len(ordered) - 1)))],
    }


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return math.nan
    lm, rm = statistics.fmean(left), statistics.fmean(right)
    lc, rc = [x - lm for x in left], [x - rm for x in right]
    denominator = math.sqrt(sum(x * x for x in lc) * sum(x * x for x in rc))
    return math.nan if denominator == 0 else sum(a * b for a, b in zip(lc, rc, strict=True)) / denominator


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_ranks(left), _ranks(right))


def _aggregate_category_matrix(
    rows: Sequence[Mapping[str, Any]],
    design_by_id: Mapping[str, Mapping[str, Any]],
    arm: str,
) -> dict[str, dict[str, float]]:
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    weights: dict[str, float] = defaultdict(float)
    for observation in rows:
        case = design_by_id[str(observation["case_id"])]
        blocks = case["target_blocks"]
        by_id = {str(row["block_id"]): row for row in blocks}
        for layer in PROBE_LAYERS:
            matrix = observation[f"{arm}_matrix"][str(layer)]
            for query_id, distribution in matrix.items():
                if distribution is None:
                    continue
                query = by_id[query_id]
                qcat = str(query["category"])
                weight = float(query["tokens"])
                weights[qcat] += weight
                for key_id, mass in distribution.items():
                    sums[qcat][str(by_id[key_id]["category"])] += weight * float(mass)
    return {
        query: {key: value / weights[query] for key, value in values.items()}
        for query, values in sums.items()
    }


def analyze(output: Path) -> dict[str, Any]:
    design = json.loads((output / "DESIGN.json").read_text())
    design_by_id = {str(row["case_id"]): row for row in design["cases"]}
    rows = [
        json.loads(line)
        for line in (output / "OBSERVATIONS.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("no observations")
    signals: dict[str, list[float]] = defaultdict(list)
    case_summaries = []
    for row in rows:
        layer_rows = row["metrics"]["layers"]
        summary = {
            "case_id": row["case_id"],
            "instance_id": row["instance_id"],
            "target_tokens": row["target_tokens"],
            "copied_tokens": row["copied_tokens"],
        }
        for name, path in (
            ("suffix_tv", ("recomputed_suffix", "tv")),
            ("suffix_js_nats", ("recomputed_suffix", "js_nats")),
            ("suffix_copied_mass_delta", ("recomputed_suffix", "copied_island_mass_delta")),
            ("generation_tv", ("generation_marker", "tv")),
            ("formation_tv", ("copied_row_source_time_vs_dense_target", "tv")),
            ("formation_source_only_mass", ("copied_row_source_time_vs_dense_target", "source_only_context_mass")),
        ):
            values = [float(layer[path[0]][path[1]]) for layer in layer_rows]
            summary[f"median_{name}_over_layers"] = statistics.median(values)
            signals[name].extend(values)
        summary["generation_top_block_agreement_fraction"] = statistics.fmean(
            float(layer["generation_marker"]["top_block_agreement"])
            for layer in layer_rows
        )
        summary["suffix_top_block_agreement_fraction"] = statistics.fmean(
            float(layer["recomputed_suffix"]["top_block_agreement"])
            for layer in layer_rows
        )
        summary["final_logit_js"] = row["mechanical_checks"]["dense_reuse_final_logit_js"]
        summary["final_top1_changed"] = row["mechanical_checks"]["dense_reuse_top1_changed"]
        case_summaries.append(summary)
    aggregate = {
        "cases": len(rows),
        "tasks": len({row["instance_id"] for row in rows}),
        **{name: _median_iqr(values) for name, values in signals.items()},
        "generation_top_block_agreement_fraction": statistics.fmean(
            row["generation_top_block_agreement_fraction"] for row in case_summaries
        ),
        "suffix_top_block_agreement_fraction": statistics.fmean(
            row["suffix_top_block_agreement_fraction"] for row in case_summaries
        ),
        "final_top1_changed_fraction": statistics.fmean(
            float(row["final_top1_changed"]) for row in case_summaries
        ),
        "final_logit_js": _median_iqr([row["final_logit_js"] for row in case_summaries]),
        "mechanical_maxima": {
            key: max(float(row["mechanical_checks"][key]) for row in rows)
            for key in (
                "attention_row_sum_abs_error_max",
                "prefix_matrix_abs_delta_max",
                "instrumented_reference_logit_abs_delta_max",
            )
        },
    }
    prompt_tokens = [float(row["target_tokens"]) for row in rows]
    copy_tokens = [float(row["copied_tokens"]) for row in rows]
    prefix_shift = [
        abs(float(row["target_start"]) - float(row["source_start"])) for row in rows
    ]
    suffix_tv = [
        float(summary["median_suffix_tv_over_layers"]) for summary in case_summaries
    ]
    correlations = {
        "suffix_tv_vs_prompt_tokens_spearman": _spearman(suffix_tv, prompt_tokens),
        "suffix_tv_vs_copy_tokens_spearman": _spearman(suffix_tv, copy_tokens),
        "suffix_tv_vs_abs_prefix_shift_spearman": _spearman(suffix_tv, prefix_shift),
    }
    result = {
        "status": "COMPLETE" if len(rows) == len(design["cases"]) else "CANARY_COMPLETE",
        "aggregate": aggregate,
        "correlations": correlations,
        "case_summaries": case_summaries,
        "dense_category_matrix": _aggregate_category_matrix(rows, design_by_id, "dense"),
        "reuse_category_matrix": _aggregate_category_matrix(rows, design_by_id, "reuse"),
        "representative_case_ids": design["representative_case_ids"],
        "scope": (
            "Qwen2.5-Coder-3B mechanism proxy on unchanged M56 trajectory text and "
            "runtime-faithful V40 splice semantics; not 30B native-engine attention, "
            "accuracy, or latency."
        ),
    }
    _write_json(output / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    measure_parser = commands.add_parser("measure")
    measure_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    measure_parser.add_argument("--max-cases", type=int, default=0)
    analyze_parser = commands.add_parser("analyze")
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
