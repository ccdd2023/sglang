"""Long-code (500-3000 token) segment extractor from bigcode/the-stack-smol-xs.

Purpose:
    The original `context_sampler.py` corpus is HumanEval + 6 synthetic fixtures
    (max ~275 tokens). That is too short to reveal K/V divergence caused by
    long context (the user's note: "我们的长代码块都是自己造的，需要一个完整
    开源数据集来测试"). This extractor pulls real-world Python functions/classes
    (500-3000 tokens) from `the-stack-smol-xs` so the same-code × different-
    context experiment can be re-run on long code.

Output (drop-in compatible with the existing pipeline):
    data/segments_long.json    — list[Segment]
    data/variations_long.json  — list[VariationDict] (Cartesian product)

Format mirrors context_sampler.Segment / build_variations() exactly so the
existing `kv_distance_analyzer.py` runs with no changes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from typing import Iterable

# Ensure we can reuse _length_bin / _approximate_token_count / variation builder
# from the sibling context_sampler.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from context_sampler import (  # noqa: E402
    Segment,
    POSITION_OFFSETS,
    SYSTEM_PROMPT_CLASSES,
    SURROUNDING_CODE_CLASSES,
    SYSTEM_PROMPTS,
    SURROUNDING_WRAPPERS,
    _padding_tokens,
    _approximate_token_count,
    _length_bin,
    build_variations,
    _classify_by_token_count,
)


# ---- Tokenizer-aware token counting (more accurate than whitespace heuristic)

def _hf_token_count(tokenizer, text: str) -> int:
    """Use the actual HF tokenizer to count tokens. Falls back to whitespace
    heuristic if tokenizer is unavailable."""
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return _approximate_token_count(text)


# ---- AST-based extraction -------------------------------------------------

def _extract_python_units(source: str) -> list[tuple[str, str]]:
    """Return list of (unit_name, unit_source) for each top-level function or
    class in `source`. Comments/docstrings preserved verbatim."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    lines = source.splitlines(keepends=True)
    out: list[tuple[str, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", None)
            if end is None:
                # Fallback: scan forward until dedent
                end = start + 1
                base_indent = len(lines[start]) - len(lines[start].lstrip())
                while end < len(lines):
                    cur = lines[end]
                    if cur.strip() and (len(cur) - len(cur.lstrip())) <= base_indent:
                        break
                    end += 1
            unit_src = "".join(lines[start:end])
            out.append((node.name, unit_src))
    return out


# ---- Quality filters ------------------------------------------------------

# Patterns that signal a unit is too noisy / not real-world code to reuse.
_BAD_NAME_PATTERNS = re.compile(
    r"^(test_|fixture_|conftest_|__|main$|run$|setup$|teardown$)",
    re.IGNORECASE,
)


def _is_interesting(name: str, src: str, min_tokens: int, max_tokens: int) -> bool:
    n = _approximate_token_count(src)
    if n < min_tokens or n > max_tokens:
        return False
    if _BAD_NAME_PATTERNS.match(name):
        return False
    # Must have at least one executable statement besides the signature.
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # body of a function — must have at least one statement
            if len(node.body) == 0:
                continue
            # Skip if the entire body is just a docstring or pass
            non_trivial = [
                s for s in node.body
                if not isinstance(s, (ast.Pass, ast.Expr)) or (
                    isinstance(s, ast.Expr) and not isinstance(s.value, ast.Constant)
                )
            ]
            if non_trivial:
                return True
        elif isinstance(node, ast.ClassDef):
            if len(node.body) > 0:
                return True
    return False


# ---- Main sampling loop ---------------------------------------------------

def sample_from_stack(
    tokenizer_name: str | None,
    target_segments: int,
    min_tokens: int,
    max_tokens: int,
    max_files_to_scan: int,
    seed: int = 0,
) -> list[Segment]:
    """Stream the-stack-smol-xs and collect up to `target_segments` Python
    function/class units in the [min_tokens, max_tokens] token range."""
    from datasets import load_dataset

    tokenizer = None
    if tokenizer_name:
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            print(f"[long_code] loaded tokenizer {tokenizer_name}", flush=True)
        except Exception as e:
            print(f"[long_code] tokenizer load failed: {e}; falling back to heuristic", flush=True)
            tokenizer = None

    ds = load_dataset(
        "bigcode/the-stack-smol-xs",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    segs: list[Segment] = []
    files_scanned = 0
    for rec in ds:
        files_scanned += 1
        if files_scanned > max_files_to_scan:
            break
        if rec.get("lang") != "Python":
            continue
        content = rec.get("content", "")
        if not content or len(content) < 1000:
            continue
        for name, unit_src in _extract_python_units(content):
            if not _is_interesting(name, unit_src, min_tokens, max_tokens):
                continue
            tok_count = _hf_token_count(tokenizer, unit_src) if tokenizer else _approximate_token_count(unit_src)
            if tok_count < min_tokens or tok_count > max_tokens:
                continue
            # source_id is a stable id from the file path + name + content hash
            sid = hashlib.sha1(
                (rec.get("id", "") + "::" + name + "::" + unit_src[:64]).encode()
            ).hexdigest()[:12]
            segs.append(Segment(
                seg_id=f"stack__{sid}_{name[:24]}",
                ast_type=_classify_by_token_count(unit_src),
                length_bin=_length_bin(tok_count),
                token_count=tok_count,
                source=unit_src,
                source_id=f"{rec.get('repo_name', '?')}::{name}",
            ))
            if len(segs) >= target_segments:
                break
        if len(segs) >= target_segments:
            break
        if files_scanned % 50 == 0:
            print(f"[long_code] scanned {files_scanned} files, collected {len(segs)} segments", flush=True)
    return segs


# ---- Variation builder (re-uses context_sampler.build_variations) --------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-segments", type=int, default=12)
    p.add_argument("--min-tokens", type=int, default=500)
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--max-files", type=int, default=2000)
    p.add_argument("--tokenizer", default="/home/gfy/models/Qwen2.5-3B-Instruct",
                   help="HF tokenizer for accurate token counts (default Qwen2.5-3B)")
    p.add_argument("--out-segments", default=os.path.join(_THIS_DIR, "data", "segments_long.json"))
    p.add_argument("--out-variations", default=os.path.join(_THIS_DIR, "data", "variations_long.json"))
    args = p.parse_args()

    segs = sample_from_stack(
        tokenizer_name=args.tokenizer,
        target_segments=args.target_segments,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_files_to_scan=args.max_files,
    )
    print(f"[long_code] {len(segs)} segments collected", flush=True)
    from collections import Counter
    print(f"[long_code] length_bin distribution: {dict(Counter(s.length_bin for s in segs))}", flush=True)
    print(f"[long_code] ast_type distribution: {dict(Counter(s.ast_type for s in segs))}", flush=True)

    os.makedirs(os.path.dirname(args.out_segments), exist_ok=True)
    with open(args.out_segments, "w") as f:
        json.dump([s.__dict__ for s in segs], f, indent=2, ensure_ascii=False)
    print(f"[long_code] wrote {args.out_segments}", flush=True)

    variations = build_variations(segs)
    with open(args.out_variations, "w") as f:
        json.dump(variations, f, indent=2, ensure_ascii=False)
    print(f"[long_code] wrote {args.out_variations} ({len(variations)} variations)", flush=True)


if __name__ == "__main__":
    main()
