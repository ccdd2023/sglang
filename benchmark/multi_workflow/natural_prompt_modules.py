#!/usr/bin/env python3
"""Deterministic natural-module segmentation for coding-agent prompts.

The module boundary is derived only from text already visible to the online
agent.  It does not use a gold patch, evaluator outcome, Dense attention, or
future model output.  Rendering is deliberately delegated to the fixed M50
backend and this module only partitions the resulting token sequence; callers
can therefore verify that segmentation never rewrites the compared prompt.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from benchmark.multi_workflow import motivate_v50_coding_provenance as m50
from benchmark.multi_workflow.coding_reuse_policy import (
    _tool_command,
    critical_coding_event_reasons,
    is_high_value_executable_failure,
    is_successful_readonly_evidence,
    observed_repository_path_provenance,
    repository_commit_phase_event,
    repository_mutation_symbols,
    repository_observation_symbols,
    repository_paths,
)


MODULE_TYPES = (
    "system_instruction",
    "task_specification",
    "assistant_interpretation",
    "tool_command",
    "repository_code",
    "repository_search",
    "test_or_execution_feedback",
    "diff_or_mutation_feedback",
    "other_tool_result",
    "generation_marker",
)

_SEARCH = re.compile(r"(?:^|[;&|]\s*)(?:rg|grep|find)\b", re.I)
_READ_ONLY = re.compile(r"(?:^|[;&|]\s*)(?:rg|grep|find|sed|cat|head|tail)\b", re.I)
_RETURN_CODE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>", re.I)
_EXECUTION = re.compile(
    r"\b(?:python\d*|pytest|tox|unittest|bash|sh)\b"
    r"|\bmake\s+(?:test|check)\b",
    re.I,
)
_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"((?:/testbed/|\./|a/|b/)?[A-Za-z0-9_.-]+"
    r"(?:/[A-Za-z0-9_.-]+)*\."
    r"(?:py|pyi|toml|yaml|yml|json|rst|md|cfg|ini|txt))\b"
)
_SYMBOL = re.compile(
    r"`([A-Za-z_]\w*)`|\b(?:class|def|function|method)\s+([A-Za-z_]\w*)\b"
)
_TRACEBACK = re.compile(
    r"traceback \(most recent call last\)|\b[A-Za-z_]+Error\b"
    r"|\bassert(?:ion)?(?:error| failed)\b",
    re.I,
)
_FILE_HEADER = re.compile(
    r"(?m)^(?:==>\s*|###\s*(?:File:\s*)?|---\s*)"
    r"((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\."
    r"(?:py|pyi|toml|yaml|yml|json|rst|md|cfg|ini|txt))"
    r"(?:\s*<==)?\s*$"
)


def _normalize_path(value: str) -> str:
    value = value.strip()
    for prefix in ("/testbed/", "./", "a/", "b/"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _text_paths(value: str) -> set[str]:
    return {_normalize_path(match.group(1)) for match in _PATH.finditer(value)}


def _text_symbols(value: str) -> set[str]:
    return {
        next(part for part in match.groups() if part)
        for match in _SYMBOL.finditer(value)
    }


def _token_hash(ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def classify_tool_result(group: Sequence[Mapping[str, Any]]) -> str:
    """Classify a tool result using only its completed interaction."""

    copied = [dict(message) for message in group]
    commands = "\n".join(
        command for message in copied if (command := _tool_command(message))
    )
    risks = set(critical_coding_event_reasons(copied))
    if risks & {"repository_mutation_command", "repository_diff_observed"}:
        return "diff_or_mutation_feedback"
    if is_high_value_executable_failure(copied) or _EXECUTION.search(commands):
        return "test_or_execution_feedback"
    observations = "\n".join(
        str(message.get("content") or "")
        for message in copied
        if message.get("role") == "tool"
    )
    return_codes = [int(value) for value in _RETURN_CODE.findall(observations)]
    successful_read = (
        bool(_READ_ONLY.search(commands))
        and bool(return_codes)
        and all(value == 0 for value in return_codes)
    )
    if successful_read or is_successful_readonly_evidence(copied):
        return "repository_search" if _SEARCH.search(commands) else "repository_code"
    return "other_tool_result"


def _encode_offsets(tokenizer: Any, literal: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        literal,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = list(encoded["input_ids"])
    offsets = [tuple(value) for value in encoded["offset_mapping"]]
    if ids != list(tokenizer.encode(literal, add_special_tokens=False)):
        raise ValueError("offset tokenizer changed rendered token ids")
    if len(ids) != len(offsets):
        raise ValueError("token offsets do not align with token ids")
    return ids, offsets


def _token_boundary(offsets: Sequence[tuple[int, int]], char_boundary: int) -> int:
    """Return a deterministic nearest token boundary for a character edge."""

    for index, (start, end) in enumerate(offsets):
        if start >= char_boundary:
            return index
        if start < char_boundary < end:
            return index + int(char_boundary - start >= end - char_boundary)
    return len(offsets)


def _directory_overlap(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return False
    left_dirs = {str(PurePosixPath(value).parent) for value in left}
    right_dirs = {str(PurePosixPath(value).parent) for value in right}
    return bool(left_dirs & right_dirs)


def _tool_content_parts(content: str, fallback_paths: set[str]) -> list[dict[str, Any]]:
    """Split multi-file output only when explicit file headers are present."""

    headers = list(_FILE_HEADER.finditer(content))
    if len(headers) < 2:
        return [{"char_start": 0, "char_end": len(content), "paths": fallback_paths}]
    parts: list[dict[str, Any]] = []
    for index, header in enumerate(headers):
        start = 0 if index == 0 else header.start()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(content)
        parts.append(
            {
                "char_start": start,
                "char_end": end,
                "paths": {_normalize_path(header.group(1))},
            }
        )
    return parts


def render_natural_prompt_modules(
    tokenizer: Any,
    base: Sequence[Mapping[str, Any]],
    groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    rolling_groups: int = m50.ROLLING_GROUPS,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    """Render one prompt and return exact modules plus causal relation rows.

    Every returned module is a contiguous token interval.  The intervals are
    non-overlapping, cover the full prompt, and concatenate to the exact M50
    prompt token IDs.
    """

    prompt_ids: list[int] = []
    modules: list[dict[str, Any]] = []

    def append_piece(
        piece_ids: Sequence[int],
        *,
        module_type: str,
        role: str,
        parent: str,
        source_request_index: int,
        paths: Sequence[str] = (),
        symbols: Sequence[str] = (),
        epoch: int = 0,
        repository_scope_dependency: bool = False,
        contains_traceback: bool = False,
    ) -> None:
        if not piece_ids:
            return
        if module_type not in MODULE_TYPES:
            raise ValueError(f"unknown module type: {module_type}")
        start = len(prompt_ids)
        prompt_ids.extend(int(value) for value in piece_ids)
        end = len(prompt_ids)
        modules.append(
            {
                "module_id": f"m{len(modules):03d}",
                "parent_interaction_id": parent,
                "module_type": module_type,
                "role": role,
                "token_start": start,
                "token_end": end,
                "natural_length": end - start,
                "paths": sorted(set(paths)),
                "symbols": sorted(set(symbols)),
                "repository_epoch": epoch,
                "grounding_module_ids": [],
                "content_hash": _token_hash(piece_ids),
                "source_request_index": source_request_index,
                "invalidating_event": None,
                "repository_scope_dependency": repository_scope_dependency,
                "contains_traceback": contains_traceback,
            }
        )

    def append_whole(
        literal: str,
        *,
        module_type: str,
        role: str,
        parent: str,
        source_request_index: int,
        paths: Sequence[str] = (),
        symbols: Sequence[str] = (),
        epoch: int = 0,
        repository_scope_dependency: bool = False,
        contains_traceback: bool = False,
    ) -> None:
        append_piece(
            tokenizer.encode(literal, add_special_tokens=False),
            module_type=module_type,
            role=role,
            parent=parent,
            source_request_index=source_request_index,
            paths=paths,
            symbols=symbols,
            epoch=epoch,
            repository_scope_dependency=repository_scope_dependency,
            contains_traceback=contains_traceback,
        )

    for index, raw_message in enumerate(base):
        message = dict(raw_message)
        role = str(message.get("role") or "")
        append_whole(
            m50._render_message_literal(message),
            module_type=("system_instruction" if role == "system" else "task_specification"),
            role=role,
            parent=f"base-{index}",
            source_request_index=-1,
            paths=_text_paths(str(message.get("content") or "")),
            symbols=_text_symbols(str(message.get("content") or "")),
        )

    copied_groups = [[dict(message) for message in group] for group in groups]
    dropped = max(0, len(copied_groups) - rolling_groups)
    if dropped:
        append_whole(
            m50._render_message_literal(
                {
                    "role": "user",
                    "content": m50.ROLLING_NOTICE.format(dropped=dropped),
                }
            ),
            module_type="system_instruction",
            role="user",
            parent="history-compaction",
            source_request_index=dropped,
        )

    epoch = sum(repository_commit_phase_event(group) for group in copied_groups[:dropped])
    mutation_events: list[dict[str, Any]] = []
    for group_index in range(dropped, len(copied_groups)):
        group = copied_groups[group_index]
        parent = f"interaction-{group_index:03d}"
        provenance = observed_repository_path_provenance(group)
        group_paths = set(provenance["paths"])
        group_symbols = repository_observation_symbols(group)
        result_type = classify_tool_result(group)
        group_start_module = len(modules)
        for message_index, message in enumerate(group):
            role = str(message.get("role") or "")
            literal = m50._render_message_literal(message)
            content = str(message.get("content") or "")
            if role == "assistant":
                assistant_paths = _text_paths(content) | set(repository_paths(group))
                assistant_symbols = _text_symbols(content)
                if content.strip() and message.get("tool_calls"):
                    literal_ids, offsets = _encode_offsets(tokenizer, literal)
                    header = "<|im_start|>assistant\n"
                    content_end = len(header) + len(content.strip())
                    boundary = _token_boundary(offsets, content_end)
                    append_piece(
                        literal_ids[:boundary],
                        module_type="assistant_interpretation",
                        role=role,
                        parent=parent,
                        source_request_index=group_index,
                        paths=assistant_paths,
                        symbols=assistant_symbols,
                        epoch=epoch,
                        contains_traceback=bool(_TRACEBACK.search(content)),
                    )
                    append_piece(
                        literal_ids[boundary:],
                        module_type="tool_command",
                        role=role,
                        parent=parent,
                        source_request_index=group_index,
                        paths=repository_paths(group),
                        symbols=repository_mutation_symbols(group),
                        epoch=epoch,
                    )
                else:
                    append_whole(
                        literal,
                        module_type=(
                            "assistant_interpretation" if content.strip() else "tool_command"
                        ),
                        role=role,
                        parent=parent,
                        source_request_index=group_index,
                        paths=assistant_paths,
                        symbols=assistant_symbols,
                        epoch=epoch,
                        contains_traceback=bool(_TRACEBACK.search(content)),
                    )
            elif role == "tool":
                content_start = len("<|im_start|>user\n<tool_response>\n")
                parts = _tool_content_parts(content, group_paths)
                literal_ids, offsets = _encode_offsets(tokenizer, literal)
                boundaries = [0]
                for part in parts[1:]:
                    boundaries.append(
                        _token_boundary(offsets, content_start + int(part["char_start"]))
                    )
                boundaries.append(len(literal_ids))
                for part_index, part in enumerate(parts):
                    part_text = content[int(part["char_start"]) : int(part["char_end"])]
                    append_piece(
                        literal_ids[boundaries[part_index] : boundaries[part_index + 1]],
                        module_type=result_type,
                        role=role,
                        parent=parent,
                        source_request_index=group_index,
                        paths=part["paths"],
                        symbols=(
                            repository_observation_symbols(group)
                            | repository_mutation_symbols(group)
                            | _text_symbols(part_text)
                        ),
                        epoch=epoch,
                        repository_scope_dependency=bool(
                            provenance["repository_scope_dependency"]
                        ),
                        contains_traceback=bool(_TRACEBACK.search(part_text)),
                    )
            else:
                append_whole(
                    literal,
                    module_type="other_tool_result",
                    role=role,
                    parent=parent,
                    source_request_index=group_index,
                    paths=_text_paths(content),
                    symbols=_text_symbols(content),
                    epoch=epoch,
                )
        if repository_commit_phase_event(group):
            mutation_events.append(
                {
                    "event_id": f"mutation-{group_index:03d}",
                    "group_index": group_index,
                    "paths": sorted(repository_paths(group)),
                    "symbols": sorted(repository_mutation_symbols(group)),
                    "module_ids": [
                        module["module_id"] for module in modules[group_start_module:]
                    ],
                }
            )
            epoch += 1

    append_whole(
        "<|im_start|>assistant\n",
        module_type="generation_marker",
        role="assistant",
        parent="generation",
        source_request_index=len(copied_groups),
        epoch=epoch,
    )

    _attach_grounding_and_invalidation(modules, mutation_events)
    _validate_modules(prompt_ids, modules)
    return prompt_ids, modules, build_module_relations(modules)


def _attach_grounding_and_invalidation(
    modules: list[dict[str, Any]], mutation_events: Sequence[Mapping[str, Any]]
) -> None:
    evidence_types = {
        "repository_code",
        "repository_search",
        "test_or_execution_feedback",
        "diff_or_mutation_feedback",
    }
    for query_index, query in enumerate(modules):
        if query["module_type"] != "assistant_interpretation":
            continue
        query_paths = set(query["paths"])
        query_symbols = set(query["symbols"])
        grounded: list[str] = []
        for key in modules[:query_index]:
            if key["module_type"] not in evidence_types:
                continue
            if (
                query_paths & set(key["paths"])
                or query_symbols & set(key["symbols"])
                or (query["contains_traceback"] and key["contains_traceback"])
            ):
                grounded.append(str(key["module_id"]))
        query["grounding_module_ids"] = grounded

    reusable_types = {"repository_code", "repository_search"}
    for module in modules:
        if module["module_type"] not in reusable_types:
            continue
        module_paths = set(module["paths"])
        for event in mutation_events:
            if int(event["group_index"]) <= int(module["source_request_index"]):
                continue
            changed_paths = set(event["paths"])
            if module["repository_scope_dependency"] or not module_paths or not changed_paths:
                module["invalidating_event"] = dict(event)
                break
            if module_paths & changed_paths:
                module["invalidating_event"] = dict(event)
                break


def build_module_relations(modules: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return causal key-module to query-module coding relations."""

    rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(modules):
        query_paths = set(query["paths"])
        query_symbols = set(query["symbols"])
        grounding = set(query.get("grounding_module_ids") or ())
        for key in modules[:query_index]:
            key_paths = set(key["paths"])
            key_symbols = set(key["symbols"])
            distance = int(query["source_request_index"]) - int(
                key["source_request_index"]
            )
            same_interaction = key["parent_interaction_id"] == query["parent_interaction_id"]
            failure_next_action = (
                key["module_type"] == "test_or_execution_feedback"
                and query["module_type"] == "assistant_interpretation"
                and distance == 1
            )
            feedback_to_patch = (
                key["module_type"] == "test_or_execution_feedback"
                and query["module_type"] == "diff_or_mutation_feedback"
                and bool(key_paths & query_paths)
            )
            rows.append(
                {
                    "key_module_id": key["module_id"],
                    "query_module_id": query["module_id"],
                    "exact_path": bool(key_paths & query_paths),
                    "same_directory": _directory_overlap(key_paths, query_paths),
                    "shared_symbol": bool(key_symbols & query_symbols),
                    "interpretation_grounding": key["module_id"] in grounding,
                    "feedback_to_patch": feedback_to_patch,
                    "failure_to_next_action": failure_next_action,
                    "same_interaction": same_interaction,
                    "interaction_distance": distance,
                    "same_repository_epoch": (
                        key["repository_epoch"] == query["repository_epoch"]
                    ),
                }
            )
    return rows


def _validate_modules(ids: Sequence[int], modules: Sequence[Mapping[str, Any]]) -> None:
    if not modules or int(modules[0]["token_start"]) != 0:
        raise ValueError("natural modules do not begin at token zero")
    if int(modules[-1]["token_end"]) != len(ids):
        raise ValueError("natural modules do not cover the rendered prompt")
    for index, module in enumerate(modules):
        start = int(module["token_start"])
        end = int(module["token_end"])
        if end <= start or end - start != int(module["natural_length"]):
            raise ValueError("natural module has an invalid token interval")
        if module["content_hash"] != _token_hash(ids[start:end]):
            raise ValueError("natural module token hash mismatch")
        if index and int(modules[index - 1]["token_end"]) != start:
            raise ValueError("natural modules are not contiguous")
