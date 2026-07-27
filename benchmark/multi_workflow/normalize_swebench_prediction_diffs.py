#!/usr/bin/env python3
"""Normalize unified-diff syntax without changing the proposed edits."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "swebench_verified_complex_v1_20260724"
)
DEFAULT_INPUT = DEFAULT_ROOT / "dense_qwen25_7b_oracle_ast_v1/predictions.jsonl"
DEFAULT_OUTPUT = (
    DEFAULT_ROOT / "dense_qwen25_7b_oracle_ast_v1_recount/predictions.jsonl"
)
HUNK = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<suffix>.*)$"
)


def range_text(start: int, count: int) -> str:
    return str(start) if count == 1 else f"{start},{count}"


def recount_patch(
    patch: str,
    *,
    repair_empty_hunk_lines: bool = False,
) -> tuple[str, dict[str, Any]]:
    lines = patch.splitlines()
    output = []
    changed_headers = 0
    repaired_empty_lines = 0
    hunk_count = 0
    index = 0
    while index < len(lines):
        match = HUNK.match(lines[index])
        if match is None:
            output.append(lines[index])
            index += 1
            continue
        hunk_count += 1
        end = index + 1
        while end < len(lines):
            if HUNK.match(lines[end]) or lines[end].startswith("diff --git "):
                break
            end += 1
        body = lines[index + 1 : end]
        if repair_empty_hunk_lines:
            repaired_body = []
            for line in body:
                if line == "":
                    repaired_body.append(" ")
                    repaired_empty_lines += 1
                else:
                    repaired_body.append(line)
            body = repaired_body
        old_count = sum(line[:1] in (" ", "-") for line in body)
        new_count = sum(line[:1] in (" ", "+") for line in body)
        header = (
            f"@@ -{range_text(int(match.group('old_start')), old_count)} "
            f"+{range_text(int(match.group('new_start')), new_count)} "
            f"@@{match.group('suffix')}"
        )
        if header != lines[index]:
            changed_headers += 1
        output.append(header)
        output.extend(body)
        index = end
    normalized = "\n".join(output)
    if patch.endswith("\n") or normalized:
        normalized += "\n"
    return normalized, {
        "hunk_count": hunk_count,
        "changed_headers": changed_headers,
        "repaired_empty_hunk_lines": repaired_empty_lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--repair-empty-hunk-lines",
        action="store_true",
        help=(
            "Prefix empty lines inside hunks with one diff-context space. "
            "This repairs syntax only; the represented file content stays empty."
        ),
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stats = []
    normalized_rows = []
    for row in rows:
        patch, patch_stats = recount_patch(
            row["model_patch"],
            repair_empty_hunk_lines=args.repair_empty_hunk_lines,
        )
        label_suffix = (
            "-deterministic-diff-syntax"
            if args.repair_empty_hunk_lines
            else "-deterministic-recount"
        )
        normalized_rows.append(
            {
                **row,
                "model_name_or_path": row["model_name_or_path"] + label_suffix,
                "model_patch": patch,
            }
        )
        stats.append({"instance_id": row["instance_id"], **patch_stats})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized_rows),
        encoding="utf-8",
    )
    (args.output.parent / "RECOUNT_STATS.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"normalized {len(rows)} predictions; "
        f"changed {sum(row['changed_headers'] for row in stats)} hunk headers; "
        "repaired "
        f"{sum(row['repaired_empty_hunk_lines'] for row in stats)} empty hunk lines"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
