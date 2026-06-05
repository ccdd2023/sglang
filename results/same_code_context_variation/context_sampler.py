"""Context-variation code sampler.

Curates a set of code segments and emits a Cartesian product of prompt-context
variations (position offset × system prompt class × surrounding code wrap).

Compared to `ast_kv_distance/ast_sampler.py`, this sampler keeps CODE CONTENT
constant across variations. The only thing that changes is the PROMPT CONTEXT
around the code block. This is the variable that the user wants to study:
does the same code, placed in different prompts, have different K/V caches?

Output:
  data/segments.json  — N entries (one per code sample)
  data/variations.json — N × |offsets| × |systems| × |surroundings| entries
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass, asdict, field
from typing import Iterable

# Reuse the synthetic fixtures and humaneval loader from the prior experiment.
_PYTHON_ROOT = "/home/gfy/CodeMAS_Project/sglang-kvflow/python"
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

# Pull in 4 MAScoder system prompts verbatim (matching /home/gfy/CodeMAS_Project/MAScoder/src/mascoder/prompts.py).
PLANNER_SYSTEM_PROMPT = """
You are the MAScoder planning agent. Your tasks:
1) Understand the request and break it into executable steps.
2) Identify required tools and file operations.
3) Output a clear, actionable plan.
Constraint: Do not write final code; output the plan only.
""".strip()

CODER_SYSTEM_PROMPT = """
You are the MAScoder coding agent. Your tasks:
1) Implement the plan in code.
2) Use available tools for file read/write/search when needed.
3) Keep changes minimal and consistent with the existing style.
""".strip()

REVIEWER_SYSTEM_PROMPT = """
You are the MAScoder review agent. Your tasks:
1) Check whether the implementation meets requirements.
2) Identify defects, edge cases, and readability issues.
3) Provide improvement suggestions.
Constraint: Do not rewrite code; output review notes only.
""".strip()

TESTER_SYSTEM_PROMPT = """
You are the MAScoder testing agent (Code Agent). Your tasks:
1) Design and run critical test cases.
2) Collect failures and pinpoint issues.
3) Provide a concise test summary.
""".strip()

SYSTEM_PROMPTS = {
    "planner": PLANNER_SYSTEM_PROMPT,
    "coder": CODER_SYSTEM_PROMPT,
    "reviewer": REVIEWER_SYSTEM_PROMPT,
    "tester": TESTER_SYSTEM_PROMPT,
}

# Variation axes.
POSITION_OFFSETS = [0, 5, 10, 25, 50, 100]   # tokens of padding before code block
SYSTEM_PROMPT_CLASSES = ["planner", "coder", "reviewer", "tester"]
SURROUNDING_CODE_CLASSES = ["none", "class_wrap", "try_wrap", "imports_wrap"]


# ---- Surrounding-code wrappers -------------------------------------------

def _wrap_class_method(code: str) -> str:
    indent = "\n".join("    " + line for line in code.splitlines())
    return f"class _Wrapper:\n    def method(self):\n{indent}\n"


def _wrap_try(code: str) -> str:
    indent = "\n".join("    " + line for line in code.splitlines())
    return f"try:\n{indent}\nexcept Exception:\n    pass\n"


def _wrap_imports(code: str) -> str:
    return (
        "import os\nimport sys\nimport json\nfrom collections import defaultdict\n"
        "from typing import List, Dict, Optional\nfrom itertools import chain\n\n" + code
    )


SURROUNDING_WRAPPERS = {
    "none": lambda c: c,
    "class_wrap": _wrap_class_method,
    "try_wrap": _wrap_try,
    "imports_wrap": _wrap_imports,
}


# ---- Padding tokens (deterministic) -------------------------------------

_PADDING_WORDS = [
    "alpha", "beta", "gamma", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu",
]


def _padding_tokens(n: int) -> str:
    """Return n tokens of padding as a single string. The intent is to push the
    code block to position_offset tokens within the prompt. We deliberately use
    a fixed deterministic stream so the same offset always produces the same
    prefix."""
    if n <= 0:
        return ""
    out = []
    i = 0
    while sum(len(w.split()) + 1 for w in out) < n:
        out.append(_PADDING_WORDS[i % len(_PADDING_WORDS)])
        i += 1
    return " ".join(out)


# ---- Code corpus ----------------------------------------------------------

def _approximate_token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def _length_bin(tokens: int) -> str:
    if tokens < 50:
        return "<50"
    if tokens < 200:
        return "50-200"
    if tokens < 500:
        return "200-500"
    return ">500"


def _load_humaneval(path: str, max_n: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = obj.get("prompt", "").strip()
            if prompt:
                out.append((obj.get("name", "humaneval"), prompt))
                if len(out) >= max_n:
                    break
    return out


def _synthetic_fixtures() -> list[tuple[str, str]]:
    """Reused from ast_kv_distance/ast_sampler.py: ranges across decorators,
    classes, comprehensions, control flow, and imports. We keep them as
    STANDALONE snippets so the surrounding-wrap transform is meaningful
    (wrapping a class in another class would be malformed)."""
    return [
        ("syn_decorator", (
            "from functools import lru_cache\n\n"
            "@lru_cache(maxsize=128)\n"
            "def fib(n: int) -> int:\n"
            "    if n < 2:\n"
            "        return n\n"
            "    return fib(n - 1) + fib(n - 2)\n"
        )),
        ("syn_class", (
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self.n = 0\n"
            "    def inc(self):\n"
            "        self.n += 1\n"
            "    def value(self):\n"
            "        return self.n\n"
        )),
        ("syn_comprehension", (
            "def even_squares(xs):\n"
            "    return [x * x for x in xs if x % 2 == 0]\n\n"
            "def char_freq(s):\n"
            "    return {c: s.count(c) for c in set(s)}\n"
        )),
        ("syn_control_flow", (
            "def find(xs, target):\n"
            "    for i, x in enumerate(xs):\n"
            "        if x == target:\n"
            "            return i\n"
            "    return -1\n\n"
            "def safe_div(a, b):\n"
            "    try:\n"
            "        return a / b\n"
            "    except ZeroDivisionError:\n"
            "        return 0\n"
        )),
        ("syn_imports_only", (
            "import os\n"
            "import sys\n"
            "import json\n"
            "from pathlib import Path\n"
            "from typing import List, Dict, Optional\n"
        )),
        ("syn_long_function", (
            "def render_dashboard(metrics):\n"
            "    lines = []\n"
            "    lines.append('=== Dashboard ===')\n"
            "    for k, v in metrics.items():\n"
            "        if isinstance(v, (int, float)):\n"
            "            lines.append(f'{k:>20}: {v:>10.2f}')\n"
            "        else:\n"
            "            lines.append(f'{k:>20}: {v}')\n"
            "    if 'errors' in metrics and metrics['errors'] > 0:\n"
            "        lines.append('!! Errors detected; check logs.')\n"
            "    elif 'warnings' in metrics and metrics['warnings'] > 0:\n"
            "        lines.append('* Warnings present.')\n"
            "    else:\n"
            "        lines.append('OK')\n"
            "    return '\\n'.join(lines)\n"
        )),
    ]


# ---- Variation assembly --------------------------------------------------

@dataclass
class Segment:
    seg_id: str
    ast_type: str
    length_bin: str
    token_count: int
    source: str
    source_id: str = ""


def _classify_by_token_count(text: str) -> str:
    """Heuristic AST classification for our short code corpus. We don't run
    full AST here because the experiment is about *the code as input to the
    LLM*, not about parsing it. We bucket by the dominant construct."""
    n = _approximate_token_count(text)
    if text.lstrip().startswith(("class ", "class\t")):
        return "ClassDef"
    if text.lstrip().startswith(("def ", "def\t", "async def ")):
        return "FunctionDef"
    if text.lstrip().startswith(("import ", "from ")):
        return "Import"
    if "@" in text and ("def " in text or "class " in text):
        return "FunctionDef"   # decorated function — same bucket as plain fn
    if any(kw in text for kw in ("for ", "if ", "while ", "try:")):
        return "ForIfTry"
    if "[" in text and "for" in text and "in" in text:
        return "Comprehension"
    return "Other"


def build_segments(humaneval_path: str, humaneval_n: int) -> list[Segment]:
    segs: list[Segment] = []
    # Pull humaneval prompts, sorted by length so we get good length-bin coverage.
    he = _load_humaneval(humaneval_path, humaneval_n * 2)
    he.sort(key=lambda kv: _approximate_token_count(kv[1]))
    # Take evenly-spaced samples across lengths.
    step = max(1, len(he) // humaneval_n)
    for i in range(0, len(he), step):
        name, prompt = he[i]
        if len(segs) >= humaneval_n:
            break
        n = _approximate_token_count(prompt)
        segs.append(Segment(
            seg_id=f"he__{name}",
            ast_type=_classify_by_token_count(prompt),
            length_bin=_length_bin(n),
            token_count=n,
            source=prompt,
            source_id=name,
        ))
    # Add synthetic fixtures — they fill the longer bins and non-FunctionDef types.
    for name, text in _synthetic_fixtures():
        n = _approximate_token_count(text)
        segs.append(Segment(
            seg_id=f"syn__{name}",
            ast_type=_classify_by_token_count(text),
            length_bin=_length_bin(n),
            token_count=n,
            source=text,
            source_id=name,
        ))
    return segs


def _build_variation_prompt(
    code: str, position_offset: int, system_class: str, surrounding_class: str
) -> tuple[str, int, str]:
    """Return (system_prompt, user_prompt, surrounding_hash).

    The code body is identical across variations — only the prompt shape changes.
    """
    system_prompt = SYSTEM_PROMPTS[system_class]
    wrapped_code = SURROUNDING_WRAPPERS[surrounding_class](code)
    padding = _padding_tokens(position_offset)
    # Stable hash of the surrounding wrap so it can be cached
    surrounding_hash_input = f"{surrounding_class}:{position_offset}"
    surrounding_hash = hashlib.sha1(surrounding_hash_input.encode()).hexdigest()[:16]
    user_prompt = (
        (padding + "\n" if padding else "")
        + "```python\n"
        + wrapped_code
        + "\n```\n"
        + "Summarise the purpose of this code in one sentence."
    )
    return system_prompt, user_prompt, surrounding_hash


def build_variations(segments: list[Segment]) -> list[dict]:
    """Cartesian product of position_offset × system_prompt_class × surrounding_code_class
    applied to each segment."""
    out: list[dict] = []
    for seg in segments:
        for offset in POSITION_OFFSETS:
            for sys_cls in SYSTEM_PROMPT_CLASSES:
                for surr_cls in SURROUNDING_CODE_CLASSES:
                    sys_p, usr_p, surr_hash = _build_variation_prompt(
                        seg.source, offset, sys_cls, surr_cls
                    )
                    out.append({
                        "seg_id": seg.seg_id,
                        "ast_type": seg.ast_type,
                        "length_bin": seg.length_bin,
                        "token_count": seg.token_count,
                        "position_offset": offset,
                        "system_prompt_class": sys_cls,
                        "surrounding_code_class": surr_cls,
                        "surrounding_code_hash": surr_hash,
                        "system_prompt": sys_p,
                        "user_prompt": usr_p,
                    })
    return out


# ---- Main ----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--humaneval", default="/home/gfy/KVCOMM/datasets/humaneval/humaneval-py.jsonl")
    p.add_argument("--humaneval-n", type=int, default=18)
    p.add_argument("--out-segments", default="/home/gfy/CodeMAS_Project/sglang-kvflow/results/same_code_context_variation/data/segments.json")
    p.add_argument("--out-variations", default="/home/gfy/CodeMAS_Project/sglang-kvflow/results/same_code_context_variation/data/variations.json")
    args = p.parse_args()

    segs = build_segments(args.humaneval, args.humaneval_n)
    print(f"[context_sampler] {len(segs)} segments", flush=True)
    by_ast = {}
    for s in segs:
        by_ast[s.ast_type] = by_ast.get(s.ast_type, 0) + 1
    print(f"[context_sampler] by ast_type: {by_ast}", flush=True)
    by_bin = {}
    for s in segs:
        by_bin[s.length_bin] = by_bin.get(s.length_bin, 0) + 1
    print(f"[context_sampler] by length_bin: {by_bin}", flush=True)

    variations = build_variations(segs)
    print(f"[context_sampler] {len(variations)} variations ({len(variations)//max(1,len(segs))} per segment)", flush=True)

    os.makedirs(os.path.dirname(args.out_segments), exist_ok=True)
    with open(args.out_segments, "w") as f:
        json.dump([asdict(s) for s in segs], f, indent=2, ensure_ascii=False)
    with open(args.out_variations, "w") as f:
        json.dump(variations, f, indent=2, ensure_ascii=False)
    print(f"[context_sampler] wrote {args.out_segments} and {args.out_variations}", flush=True)


if __name__ == "__main__":
    main()
