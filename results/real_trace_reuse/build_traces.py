"""Build 3-agent traces from SWE-bench Verified for KVCOMM hit-rate study.

For each of the 500 SWE-bench tasks, construct a synthetic multi-agent
trace:

    Planner     -> "Plan for issue <X> in repo <R> based on these files"
                   + lists the relevant code blocks (from the issue's
                     'patch' and 'problem_statement' context)
    Implementer -> "Implement the plan in repo <R>"
                   + same code blocks (this is where KVCOMM should hit)
    Reviewer    -> "Review the implementation"
                   + same code blocks again (KVCOMM should hit twice)

Each request carries:
    - the agent's system_prompt_class (planner / coder / reviewer)
    - the code blocks as the "code_base" segment, with
      code_anchor_token_spans marking them
    - a stable content_signature per code block, so the same block in
      Planner / Implementer / Reviewer has the SAME content_signature

The trace JSONL is consumed by replay_server.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

PROJECT_ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
OUT_DIR = PROJECT_ROOT / "results" / "real_trace_reuse" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Simple system prompts mirroring MAScoder's 4 roles.
SYSTEM_PROMPTS = {
    "planner": (
        "You are a senior planner. Given an issue and a few code snippets, "
        "produce a 5-step implementation plan."
    ),
    "coder": (
        "You are a senior engineer. Given a plan and a few code snippets, "
        "implement the change."
    ),
    "reviewer": (
        "You are a senior reviewer. Given a code change and the original "
        "snippets, find any defects or edge cases."
    ),
}


def _extract_code_blocks(patch: str, problem_statement: str, n_blocks: int = 3) -> list[str]:
    """Pull a small number of code-ish blocks from the SWE-bench patch /
    problem_statement. We use the diff hunks of the patch and the markdown
    code fences in the problem statement."""
    blocks: list[str] = []
    # 1) Pull from diff hunks
    for line in patch.splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            stripped = line[1:].strip()
            if 20 < len(stripped) < 200 and ("=" not in stripped):
                blocks.append(stripped)
                if len(blocks) >= n_blocks:
                    return blocks[:n_blocks]
    # 2) Pull from markdown code fences in problem statement
    in_fence = False
    buf: list[str] = []
    for line in problem_statement.splitlines():
        if line.strip().startswith("```"):
            if in_fence:
                if 50 < sum(len(l) for l in buf) < 1000:
                    blocks.append("\n".join(buf))
                buf = []
                in_fence = False
            else:
                in_fence = True
        elif in_fence:
            buf.append(line)
        if len(blocks) >= n_blocks:
            return blocks[:n_blocks]
    return blocks[:n_blocks]


def _code_content_sig(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()[:32]


def _approx_token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def _build_one_trace(task: dict) -> list[dict] | None:
    """Return a list of 3 request payloads for one SWE-bench task, or None
    if the task doesn't have enough code to form a meaningful trace."""
    blocks = _extract_code_blocks(task.get("patch", ""), task.get("problem_statement", ""))
    if len(blocks) < 1:
        return None
    code_block = "\n".join(blocks)
    content_sig = _code_content_sig(code_block)
    n_tokens = _approx_token_count(code_block)
    # Build a system message and a user message that includes the code block
    # (the SAME block for all 3 agents, so KVCOMM can match).
    system_msg = "You are a senior {role} agent. Be concise."
    out: list[dict] = []
    role_tasks = [
        ("planner", f"Plan how to resolve: {task['problem_statement'][:400]}"),
        ("coder",   f"Implement the plan in repo {task['repo']}."),
        ("reviewer", f"Review the implementation for defects."),
    ]
    for agent_role, task_desc in role_tasks:
        user_msg = (
            f"```python\n{code_block}\n```\n\n"
            f"Task: {task_desc}\n"
        )
        # Token span: assume the code block starts after the first ~10
        # tokens of the system+user template. We won't be exact; the
        # server tolerates approximate spans.
        approx_start = 30
        approx_end = approx_start + n_tokens
        out.append({
            "instance_id": task["instance_id"],
            "repo": task["repo"],
            "agent": agent_role,
            "code_content_signature": content_sig,
            "code_block": code_block,
            "n_tokens": n_tokens,
            "approx_token_span": [approx_start, approx_end],
            "system": system_msg.format(role=agent_role),
            "user": user_msg,
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-tasks", type=int, default=500)
    p.add_argument("--out", default=str(OUT_DIR / "swe_bench_traces.jsonl"))
    args = p.parse_args()

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    n = min(args.max_tasks, len(ds))
    print(f"[build_traces] using {n} / {len(ds)} SWE-bench tasks")

    n_traces = 0
    n_skipped = 0
    with open(args.out, "w") as f:
        for i in range(n):
            task = dict(ds[i])
            trace = _build_one_trace(task)
            if trace is None:
                n_skipped += 1
                continue
            for req in trace:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
            n_traces += 1
    print(f"[build_traces] wrote {n_traces} traces ({n_skipped} skipped) to {args.out}")


if __name__ == "__main__":
    main()
