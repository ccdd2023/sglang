#!/usr/bin/env python3
"""Canonical raw-tool provenance for file-versioned SessionGraphKV."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_CODE_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"((?:/[A-Za-z0-9_.-]+)+|(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+)"
    r"\.py(?:\.[A-Za-z0-9_.-]+)?"
)
_PATCH_PATH = re.compile(
    r"(?:\*\*\* (?:Update|Add|Delete) File:|\+\+\+ b/|--- a/)\s*([^\s]+)"
)
_SUSPICIOUS_WRITE = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:sed\s+-i|apply_patch\b|patch\b|tee\b|cp\b|mv\b|"
    r"rm\b|touch\b)|(?:>|>>)\s*[^\s;&|]+|"
    r"(?:write_text|write_bytes|open\s*\([^)]*,\s*['\"][wa])",
    re.IGNORECASE,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _clean_path(value: str) -> str:
    value = value.strip().strip("`'\".,:()[]{}")
    while value.startswith("./"):
        value = value[2:]
    return value


def code_paths(text: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                _clean_path(match.group(0))
                for match in _CODE_PATH.finditer(text)
                if _clean_path(match.group(0))
            }
        )
    )


def _python_related(path: str) -> bool:
    return bool(re.search(r"\.py(?:$|\.)", Path(path).name))


def _shell_segments(command: str) -> list[list[str]]:
    output = []
    for raw in re.split(r"\s*(?:&&|;|\|\|)\s*", command):
        if not raw.strip():
            continue
        try:
            values = shlex.split(raw, comments=False, posix=True)
        except ValueError:
            values = raw.split()
        if values:
            output.append(values)
    return output


@dataclass(frozen=True)
class MutationEvent:
    session_id: str
    source: str
    event_id: str
    operation: str
    classification: str
    changed_paths: tuple[str, ...]
    removed_paths: tuple[str, ...]
    active_path_before: str | None
    active_path_after: str | None
    raw_action_sha256: str
    parser: str

    def row(self) -> dict[str, Any]:
        row = asdict(self)
        row["changed_paths"] = list(self.changed_paths)
        row["removed_paths"] = list(self.removed_paths)
        return row


def _event(
    session_id: str,
    source: str,
    index: int,
    operation: str,
    changed: tuple[str, ...],
    removed: tuple[str, ...],
    active_before: str | None,
    active_after: str | None,
    raw: str,
    parser: str,
    *,
    unresolved: bool = False,
) -> MutationEvent:
    if unresolved:
        classification = "global_fail_closed"
    elif changed or removed:
        classification = (
            "resolved_python"
            if any(_python_related(path) for path in (*changed, *removed))
            else "non_python"
        )
    else:
        classification = "non_python"
    return MutationEvent(
        session_id,
        source,
        f"event:{index:04d}",
        operation,
        classification,
        tuple(sorted(set(changed))),
        tuple(sorted(set(removed))),
        active_before,
        active_after,
        sha256_bytes(raw.encode("utf-8")),
        parser,
    )


def _shell_mutation(
    session_id: str,
    source: str,
    index: int,
    command: str,
    active: str | None,
) -> MutationEvent | None:
    changed: set[str] = set()
    removed: set[str] = set()
    operations: list[str] = []
    for match in _PATCH_PATH.finditer(command):
        path = _clean_path(match.group(1))
        if _python_related(path):
            changed.add(path)
            operations.append("patch")
    segments = _shell_segments(command)
    for segment in segments:
        name = Path(segment[0]).name
        args = [value for value in segment[1:] if not value.startswith("-")]
        if name == "sed" and any(value.startswith("-i") for value in segment[1:]):
            changed.update(_clean_path(value) for value in args if _python_related(value))
            operations.append("sed_in_place")
        elif name in {"cp", "mv"} and len(args) >= 2:
            if _python_related(args[-1]):
                changed.add(_clean_path(args[-1]))
            if name == "mv" and _python_related(args[-2]):
                removed.add(_clean_path(args[-2]))
            operations.append(name)
        elif name == "rm":
            removed.update(_clean_path(value) for value in args if _python_related(value))
            operations.append("rm")
        elif name in {"touch", "tee"}:
            changed.update(_clean_path(value) for value in args if _python_related(value))
            operations.append(name)
    for match in re.finditer(r"(?:>|>>)\s*([^\s;&|]+)", command):
        path = _clean_path(match.group(1))
        if _python_related(path):
            changed.add(path)
            operations.append("redirect")
    if changed or removed:
        return _event(
            session_id,
            source,
            index,
            "+".join(sorted(set(operations))) or "shell_write",
            tuple(changed),
            tuple(removed),
            active,
            active,
            command,
            "raw-shell-v1",
        )
    if _SUSPICIOUS_WRITE.search(command):
        concrete = any(
            value
            for segment in segments
            for value in segment[1:]
            if not value.startswith("-")
        )
        return _event(
            session_id,
            source,
            index,
            "shell_non_python_write" if concrete else "shell_write_unresolved",
            (),
            (),
            active,
            active,
            command,
            "raw-shell-v1",
            unresolved=not concrete,
        )
    return None


def parse_sweagent(session_id: str, value: dict[str, Any]) -> list[MutationEvent]:
    active: str | None = None
    output = []
    for index, row in enumerate(value["history"]):
        if str(row.get("role", "")).lower() != "assistant":
            continue
        action = str(row.get("action", "")).strip()
        if not action:
            continue
        command, _, remainder = action.splitlines()[0].partition(" ")
        command = command.lower()
        try:
            values = [
                _clean_path(item)
                for item in shlex.split(remainder)
                if item and not item.startswith("-")
            ]
        except ValueError:
            values = [
                _clean_path(item)
                for item in remainder.split()
                if item and not item.startswith("-")
            ]
        path = values[0] if values else ""
        before = active
        if command in {"open", "create"} and path:
            active = path
        if command == "open":
            continue
        if command in {"edit", "create"}:
            output.append(
                _event(
                    session_id,
                    "sweagent",
                    index,
                    f"editor_{command}",
                    ((active if command == "edit" else path),)
                    if (active if command == "edit" else path)
                    else (),
                    (),
                    before,
                    active,
                    action,
                    "swe-editor-v1",
                    unresolved=not (active if command == "edit" else path),
                )
            )
        elif command in {"rm", "delete"}:
            output.append(
                _event(
                    session_id,
                    "sweagent",
                    index,
                    command,
                    (),
                    tuple(values),
                    before,
                    active,
                    action,
                    "swe-editor-v1",
                )
            )
        elif command == "mv":
            output.append(
                _event(
                    session_id,
                    "sweagent",
                    index,
                    "mv",
                    tuple(values[-1:]) if len(values) >= 2 else (),
                    tuple(values[-2:-1]) if len(values) >= 2 else (),
                    before,
                    active,
                    action,
                    "swe-editor-v1",
                    unresolved=len(values) < 2,
                )
            )
        else:
            mutation = _shell_mutation(session_id, "sweagent", index, action, active)
            if mutation is not None:
                output.append(mutation)
    return output


def parse_openhands(
    session_id: str, value: list[dict[str, Any]]
) -> list[MutationEvent]:
    active: str | None = None
    output = []
    for index, row in enumerate(value):
        if str(row.get("role", "")).lower() != "assistant":
            continue
        for call in row.get("tool_calls") or ():
            function = call.get("function", {}) if isinstance(call, dict) else {}
            name = str(function.get("name", ""))
            raw = str(function.get("arguments", "{}"))
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                arguments = {}
            command = str(arguments.get("command", ""))
            before = active
            if name == "str_replace_editor":
                path = _clean_path(str(arguments.get("path", "")))
                if command == "view" and path:
                    active = path
                    continue
                if command in {"create", "str_replace", "insert", "undo_edit"}:
                    if path:
                        active = path
                    output.append(
                        _event(
                            session_id,
                            "openhands",
                            index,
                            f"str_replace_editor:{command}",
                            (path,) if path else (),
                            (),
                            before,
                            active,
                            raw,
                            "openhands-tool-v1",
                            unresolved=not path,
                        )
                    )
            elif name == "execute_bash":
                mutation = _shell_mutation(
                    session_id, "openhands", index, command, active
                )
                if mutation is not None:
                    output.append(mutation)
    return output


def build_canonical_manifest(
    root: Path, manifest: list[dict[str, Any]]
) -> tuple[list[MutationEvent], dict[str, Any]]:
    events = []
    source_hashes = {}
    for item in manifest:
        payload = (root / item["local_path"]).read_bytes()
        if sha256_bytes(payload) != item["sha256"]:
            raise ValueError(f"raw hash mismatch: {item['session_id']}")
        session_id = str(item["session_id"])
        source_hashes[session_id] = str(item["sha256"])
        value = json.loads(payload)
        if item["source"] == "sweagent":
            events.extend(parse_sweagent(session_id, value))
        elif item["source"] == "openhands":
            events.extend(parse_openhands(session_id, value))
        else:
            raise ValueError(f"unsupported trajectory source: {item['source']}")
    events.sort(key=lambda row: (row.session_id, row.event_id, row.operation))
    rows = [event.row() for event in events]
    counts = {
        classification: sum(
            event.classification == classification for event in events
        )
        for classification in (
            "resolved_python",
            "non_python",
            "global_fail_closed",
        )
    }
    manifest_hash = sha256_bytes(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ).encode("utf-8")
    )
    return events, {
        "passed": counts["global_fail_closed"] == 0,
        "sessions": len(manifest),
        "write_like_events": len(events),
        "classification_counts": counts,
        "canonical_manifest_sha256": manifest_hash,
        "raw_source_hashes": source_hashes,
        "parser_versions": [
            "swe-editor-v1",
            "openhands-tool-v1",
            "raw-shell-v1",
        ],
        "model_or_kv_outputs_read": False,
        "test_outcomes_read": False,
    }
