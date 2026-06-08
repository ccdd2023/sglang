#!/usr/bin/env python3
"""Build coding-structure KV sensitivity prompts.

This experiment keeps the target code bytes identical and changes only the
coding-agent prompt structure around it. It is meant to answer a question that
the older AST/context experiments do not: how sensitive is a code segment's KV
state to Planner/Coder/Reviewer workflow structure?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
DEFAULT_OUT = ROOT / "results" / "coding_structure_kv_sensitivity" / "data"
SEED_SEGMENTS = ROOT / "results" / "same_code_context_variation" / "data" / "segments.json"


SYSTEM_PROMPTS = {
    "planner": "You are AgentTemplateKV Planner. Identify files, dependencies, and reusable codebase blocks. Do not write the final patch.",
    "coder": "You are AgentTemplateKV Coder. Produce the minimal implementation patch using the provided codebase blocks.",
    "reviewer": "You are AgentTemplateKV Reviewer. Check the proposed change against the issue and codebase.",
    "tester": "You are AgentTemplateKV Tester. Infer failing behavior and suggest targeted validation from the codebase.",
}

CODING_STRUCTURES = [
    "code_first",
    "issue_first",
    "planner_trace_before_code",
    "review_trace_before_code",
    "neighbor_file_before_code",
    "previous_output_before_code",
]


@dataclass
class Segment:
    seg_id: str
    ast_type: str
    length_bin: str
    token_count: int
    source: str
    source_id: str = ""


def sha1_short(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def approx_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def synthetic_segments() -> list[Segment]:
    snippets = [
        (
            "function_error_path",
            "def normalize_path(path):\n"
            "    if path is None:\n"
            "        raise ValueError('path is required')\n"
            "    return path.replace('\\\\', '/').strip('/')\n",
            "FunctionDef",
        ),
        (
            "class_state_update",
            "class RetryBudget:\n"
            "    def __init__(self, limit):\n"
            "        self.limit = limit\n"
            "        self.used = 0\n"
            "    def consume(self):\n"
            "        if self.used >= self.limit:\n"
            "            return False\n"
            "        self.used += 1\n"
            "        return True\n",
            "ClassDef",
        ),
        (
            "loop_filter",
            "def collect_enabled(items):\n"
            "    out = []\n"
            "    for item in items:\n"
            "        if item.get('enabled'):\n"
            "            out.append(item['name'])\n"
            "    return out\n",
            "ForIf",
        ),
    ]
    out = []
    for name, code, ast_type in snippets:
        n = approx_tokens(code)
        out.append(
            Segment(
                seg_id=f"syn__{name}",
                ast_type=ast_type,
                length_bin="50-200" if n >= 50 else "<50",
                token_count=n,
                source=code,
                source_id=name,
            )
        )
    return out


def load_segments(path: Path, max_segments: int) -> list[Segment]:
    segs: list[Segment] = []
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            source = row.get("source", "")
            if not source.strip():
                continue
            segs.append(
                Segment(
                    seg_id=row.get("seg_id", f"seg_{len(segs)}"),
                    ast_type=row.get("ast_type", "unknown"),
                    length_bin=row.get("length_bin", "unknown"),
                    token_count=int(row.get("token_count", approx_tokens(source))),
                    source=source.rstrip(),
                    source_id=row.get("source_id", ""),
                )
            )
            if len(segs) >= max_segments:
                return segs
    for seg in synthetic_segments():
        if all(existing.seg_id != seg.seg_id for existing in segs):
            segs.append(seg)
        if len(segs) >= max_segments:
            break
    return segs


def code_block(idx: int, path: str, code: str) -> str:
    return f"## code_base{idx}: {path}\n```python\n{code}\n```"


def build_user_prompt(seg: Segment, role: str, structure: str) -> str:
    issue = (
        "## Issue\n"
        "A regression appears when edge-case inputs flow through this module. "
        "Find the minimal implementation change.\n"
    )
    tests = (
        "## FAIL_TO_PASS tests\n"
        "- test_handles_missing_values\n"
        "- test_preserves_existing_behavior\n"
    )
    planner_trace = (
        "## Planner trace\n"
        "1. Inspect the relevant implementation file.\n"
        "2. Keep code_base1 resident for Coder and Reviewer.\n"
        "3. Avoid touching unrelated tests.\n"
    )
    review_trace = (
        "## Reviewer context\n"
        "The previous patch may overfit the failing test. Re-check invariants, "
        "imports, and error paths before accepting it.\n"
    )
    previous_output = (
        "## Previous agent output\n"
        "Planner selected this file as the shared anchor. Coder should patch only "
        "the displayed implementation block.\n"
    )
    neighbor = code_block(
        0,
        "repo/helpers.py",
        "def helper_identity(value):\n    return value\n\nDEFAULT_TIMEOUT = 30\n",
    )
    target = code_block(1, f"repo/{seg.source_id or seg.seg_id}.py", seg.source)
    instruction = (
        f"## Agent instruction ({role})\n"
        "Return a compact patch plan or patch fragment. Do not repeat the codebase.\n"
    )
    if structure == "code_first":
        parts = [target, instruction, issue, tests]
    elif structure == "issue_first":
        parts = [issue, tests, target, instruction]
    elif structure == "planner_trace_before_code":
        parts = [issue, planner_trace, target, instruction, tests]
    elif structure == "review_trace_before_code":
        parts = [issue, review_trace, target, instruction, tests]
    elif structure == "neighbor_file_before_code":
        parts = [issue, neighbor, target, instruction, tests]
    elif structure == "previous_output_before_code":
        parts = [issue, previous_output, target, instruction, tests]
    else:
        raise ValueError(f"unknown structure: {structure}")
    return "\n\n".join(parts)


def build_variations(segments: list[Segment]) -> list[dict]:
    rows: list[dict] = []
    for seg in segments:
        content_signature = sha1_short(seg.source)
        for role in SYSTEM_PROMPTS:
            for structure in CODING_STRUCTURES:
                rows.append(
                    {
                        "seg_id": seg.seg_id,
                        "source_id": seg.source_id,
                        "ast_type": seg.ast_type,
                        "length_bin": seg.length_bin,
                        "token_count": seg.token_count,
                        "content_signature": content_signature,
                        "agent_role": role,
                        "coding_structure": structure,
                        "target_code": seg.source,
                        "system_prompt": SYSTEM_PROMPTS[role],
                        "user_prompt": build_user_prompt(seg, role, structure),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, default=SEED_SEGMENTS)
    parser.add_argument("--max-segments", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    segments = load_segments(args.segments, args.max_segments)
    variations = build_variations(segments)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "segments.json").write_text(
        json.dumps([asdict(s) for s in segments], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.out_dir / "variations.json").write_text(
        json.dumps(variations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[structure_sampler] segments={len(segments)} variations={len(variations)}")
    print(f"[structure_sampler] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
