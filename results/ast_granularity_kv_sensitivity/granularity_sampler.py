#!/usr/bin/env python3
"""Sample exact codebase spans at multiple AST granularities.

The question is not whether AST-similar but text-different code can be reused.
It cannot. The question is: for byte-identical codebase content, which AST
granularity is a good reuse object?
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
DEFAULT_MANIFEST = ROOT / "results" / "repo_level_datasets" / "manifest_10.json"
DEFAULT_OUT = ROOT / "results" / "ast_granularity_kv_sensitivity" / "data"


@dataclass
class Span:
    span_id: str
    instance_id: str
    repo: str
    path: str
    granularity: str
    ast_type: str
    start_line: int
    end_line: int
    line_count: int
    approx_tokens: int
    content_signature: str
    text: str


def sha1_short(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def approx_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def slice_lines(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start - 1 : end]).rstrip()


def add_span(
    out: list[Span],
    sample: dict,
    file_row: dict,
    granularity: str,
    ast_type: str,
    start: int,
    end: int,
    lines: list[str],
    max_span_tokens: int,
) -> None:
    text = slice_lines(lines, start, end)
    toks = approx_tokens(text)
    if toks < 8 or toks > max_span_tokens or not text.strip():
        return
    out.append(
        Span(
            span_id=sha1_short(f"{sample['instance_id']}:{file_row['path']}:{granularity}:{start}:{end}:{text}"),
            instance_id=sample["instance_id"],
            repo=sample["repo"],
            path=file_row["path"],
            granularity=granularity,
            ast_type=ast_type,
            start_line=start,
            end_line=end,
            line_count=end - start + 1,
            approx_tokens=toks,
            content_signature=sha1_short(text),
            text=text,
        )
    )


def bounded_end(lines: list[str], start: int, preferred_end: int, max_span_tokens: int) -> int:
    end = min(preferred_end, len(lines))
    while end > start and approx_tokens(slice_lines(lines, start, end)) > max_span_tokens:
        end = start + max(1, (end - start) // 2)
    return end


def collect_spans(sample: dict, file_row: dict, max_file_chars: int, max_span_tokens: int) -> list[Span]:
    path = Path(file_row["local_path"])
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")[:max_file_chars]
    lines = text.splitlines()
    if not lines:
        return []
    out: list[Span] = []
    file_end = bounded_end(lines, 1, min(len(lines), 240), max_span_tokens)
    add_span(out, sample, file_row, "file_prefix", "Module", 1, file_end, lines, max_span_tokens)
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return out

    for node in ast.walk(tree):
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            continue
        start, end = int(node.lineno), int(node.end_lineno)
        if end <= start or end > len(lines):
            continue
        if isinstance(node, ast.ClassDef):
            add_span(out, sample, file_row, "class", "ClassDef", start, min(end, start + 180), lines, max_span_tokens)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parent = "method" if any(isinstance(p, ast.ClassDef) and node in getattr(p, "body", []) for p in ast.walk(tree)) else "function"
            add_span(out, sample, file_row, parent, type(node).__name__, start, min(end, start + 120), lines, max_span_tokens)
        elif isinstance(node, (ast.For, ast.While, ast.If, ast.Try, ast.With)):
            add_span(out, sample, file_row, "control_block", type(node).__name__, start, min(end, start + 60), lines, max_span_tokens)

    # Statement windows: exact smaller-granularity chunks inside real files.
    for i in range(1, len(lines), 40):
        end = min(len(lines), i + 11)
        add_span(out, sample, file_row, "statement_window", "StmtWindow", i, end, lines, max_span_tokens)
    return out


def choose_balanced(spans: list[Span], per_granularity: int) -> list[Span]:
    chosen: list[Span] = []
    seen = set()
    for granularity in ["file_prefix", "class", "function", "method", "control_block", "statement_window"]:
        bucket = [s for s in spans if s.granularity == granularity]
        bucket.sort(key=lambda s: (abs(s.approx_tokens - 160), s.path, s.start_line))
        for span in bucket:
            if span.content_signature in seen:
                continue
            chosen.append(span)
            seen.add(span.content_signature)
            if sum(1 for s in chosen if s.granularity == granularity) >= per_granularity:
                break
    return chosen


SYSTEMS = {
    "planner": "You are an AgentTemplateKV planner. Identify reusable code objects for downstream agents.",
    "coder": "You are an AgentTemplateKV coder. Use the provided code object to implement the minimal fix.",
    "reviewer": "You are an AgentTemplateKV reviewer. Check whether the code object supports the proposed fix.",
}


def prompt_for(span: Span, agent_role: str) -> dict:
    user = (
        f"## Issue\nA repository bug may involve `{span.path}`.\n\n"
        f"## code_object granularity={span.granularity} ast={span.ast_type} path={span.path}:{span.start_line}-{span.end_line}\n"
        "```python\n"
        f"{span.text}\n"
        "```\n\n"
        "## Task\nSummarize the reusable invariant of this exact code object in one sentence."
    )
    return {
        "span_id": span.span_id,
        "agent_role": agent_role,
        "system_prompt": SYSTEMS[agent_role],
        "user_prompt": user,
        "target_code": span.text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--files-per-sample", type=int, default=2)
    parser.add_argument("--max-file-chars", type=int, default=120000)
    parser.add_argument("--max-span-tokens", type=int, default=2500)
    parser.add_argument("--per-granularity", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = manifest["samples"][: args.max_samples]
    spans: list[Span] = []
    for sample in samples:
        for file_row in sample.get("files", [])[: args.files_per_sample]:
            spans.extend(collect_spans(sample, file_row, args.max_file_chars, args.max_span_tokens))
    chosen = choose_balanced(spans, args.per_granularity)
    variations = []
    for span in chosen:
        for role in SYSTEMS:
            variations.append(prompt_for(span, role))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "spans.json").write_text(
        json.dumps([asdict(s) for s in chosen], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.out_dir / "variations.json").write_text(
        json.dumps(variations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    counts = {}
    for span in chosen:
        counts[span.granularity] = counts.get(span.granularity, 0) + 1
    print(f"[granularity_sampler] spans={len(chosen)} variations={len(variations)} counts={counts}")
    print(f"[granularity_sampler] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
