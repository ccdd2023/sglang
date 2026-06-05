"""AST-based code segment sampler.

Reads code from heterogeneous sources (humaneval prompts, synthetic fixtures,
real-world Python snippets), parses them with `ast`, tags each segment with
three orthogonal labels:

  - ast_type: FunctionDef / ClassDef / Import / Decorator / Comprehension / For-If-Try
  - template: humaneval / gsm8k / mbpp / swe-bench / synthetic
  - length_bin: <50 / 50-200 / 200-500 / >500 tokens

The output is a JSON list of records, each containing the source code text and
all labels. This is the input to the KV distance analyzer.
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import Iterable

# ---------- Configuration -----------------------------------------------

LENGTH_BINS = [(0, 50), (50, 200), (200, 500), (500, 10**9)]
LENGTH_BIN_LABELS = ["<50", "50-200", "200-500", ">500"]


@dataclass
class CodeSegment:
    seg_id: str
    ast_type: str          # FunctionDef | ClassDef | Import | Decorator | Comprehension | ForIfTry
    template: str          # humaneval | gsm8k | mbpp | swe-bench | synthetic
    length_bin: str        # <50 | 50-200 | 200-500 | >500
    token_count: int       # approximate whitespace-split count (analyzed later)
    source: str            # raw source text
    source_id: str = ""    # original problem name / origin path
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- AST classification ------------------------------------------

def _slice_source(lines: list[str], start: int, end: int) -> str:
    start = max(1, int(start))
    end = max(start, int(end))
    return "\n".join(lines[start - 1 : end])


def _approximate_token_count(text: str) -> int:
    """Approximate token count with whitespace + punctuation. Good enough for binning."""
    return len(re.findall(r"\w+|[^\w\s]", text))


def _length_bin(token_count: int) -> str:
    for (lo, hi), label in zip(LENGTH_BINS, LENGTH_BIN_LABELS):
        if lo <= token_count < hi:
            return label
    return LENGTH_BIN_LABELS[-1]


def _classify_node(node: ast.AST) -> str:
    """Map an AST node to a coarse structural category used in the experiment."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "FunctionDef"
    if isinstance(node, ast.ClassDef):
        return "ClassDef"
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "Import"
    if isinstance(node, ast.Decorator):
        return "Decorator"
    # Comprehensions are nested; classify at the top-level enclosing node instead.
    if isinstance(node, (ast.For, ast.While, ast.If, ast.Try)):
        return "ForIfTry"
    return ""


def _comprehension_aware_extract(tree: ast.AST, lines: list[str]) -> list[tuple[str, str, int, int]]:
    """Extract (ast_type, snippet, start_line, end_line) tuples for structural anchors.

    We return:
      - one Import record covering all top-level imports
      - one record per top-level function/class (with decorators inlined)
      - one record per NESTED function/method (so class methods show up as FunctionDef)
      - one record per top-level ListComp/SetComp/DictComp/GeneratorExp
      - one record per top-level For/While/If/Try
    """
    records: list[tuple[str, str, int, int]] = []
    body = getattr(tree, "body", [])
    if not body:
        return records

    # Aggregate imports into a single Import block
    import_lines: list[int] = []
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_lines.extend(range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1))
    if import_lines:
        s, e = min(import_lines), max(import_lines)
        records.append(("Import", _slice_source(lines, s, e), s, e))

    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            label = "ClassDef" if isinstance(node, ast.ClassDef) else "FunctionDef"
            s = node.lineno
            e = getattr(node, "end_lineno", node.lineno)
            if node.decorator_list:
                s = min(d.lineno for d in node.decorator_list)
            records.append((label, _slice_source(lines, s, e), s, e))
            # Recurse: pull out nested defs as their own FunctionDef records so
            # class methods don't get hidden inside the ClassDef bucket.
            for child in ast.walk(node):
                if child is node:
                    continue
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cs = child.lineno
                    ce = getattr(child, "end_lineno", child.lineno)
                    if child.decorator_list:
                        cs = min(d.lineno for d in child.decorator_list)
                    records.append(("FunctionDef", _slice_source(lines, cs, ce), cs, ce))
            continue
        if isinstance(node, (ast.For, ast.While, ast.If, ast.Try)):
            s, e = node.lineno, getattr(node, "end_lineno", node.lineno)
            records.append(("ForIfTry", _slice_source(lines, s, e), s, e))
            continue
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and isinstance(node.value, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    s, e = node.lineno, getattr(node, "end_lineno", node.lineno)
                    records.append(("Comprehension", _slice_source(lines, s, e), s, e))
                    break
    return records


# ---------- Source loaders ---------------------------------------------

def _load_humaneval(path: str, max_n: int) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
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
            name = obj.get("name", "humaneval_unknown")
            prompt = obj.get("prompt", "")
            if prompt:
                out.append((name, "humaneval", prompt))
                if len(out) >= max_n:
                    break
    return out


def _load_gsm8k(path: str, max_n: int) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = obj.get("question", "")
            a = obj.get("answer", "")
            # We synthesize a tiny Python snippet that would solve it; for
            # distance analysis we only need the *shape* of the code, not correctness.
            snippet = _gsm8k_to_python_stub(q, a)
            out.append((f"gsm8k_{i:04d}", "gsm8k", snippet))
            if len(out) >= max_n:
                break
    return out


def _gsm8k_to_python_stub(question: str, answer: str) -> str:
    """Convert a GSM8K word problem into a short Python solver skeleton.

    We use the problem text as a comment and the answer as a return value.
    The exact body is irrelevant; what matters is the structural shape that
    the AST extractor sees (a single function def with arithmetic body).
    """
    safe_q = re.sub(r"[^\w\s.,'?!-]", "", question)[:200]
    safe_a = re.sub(r"[^\w\.\s+\-*/()=]", "", answer.split("####")[-1].strip() if "####" in answer else answer)[:80]
    return (
        "def solve():\n"
        f"    # {safe_q}\n"
        f"    result = {safe_a if safe_a else 0}\n"
        "    return result\n"
    )


def _synthetic_fixtures() -> list[tuple[str, str, str]]:
    """Hand-curated snippets to populate the structural buckets that the
    real-world datasets don't naturally contain (decorators, comprehensions,
    classes with multiple methods, top-level imports, control flow)."""
    fixtures: list[tuple[str, str, str]] = [
        # Decorator
        ("synt_decorator_a", "synthetic", (
            "from functools import lru_cache\n\n"
            "@lru_cache(maxsize=128)\n"
            "def fib(n: int) -> int:\n"
            "    if n < 2:\n"
            "        return n\n"
            "    return fib(n - 1) + fib(n - 2)\n"
        )),
        ("synt_decorator_b", "synthetic", (
            "import time\n\n"
            "def timeit(fn):\n"
            "    def wrapper(*args, **kwargs):\n"
            "        start = time.time()\n"
            "        out = fn(*args, **kwargs)\n"
            "        print(time.time() - start)\n"
            "        return out\n"
            "    return wrapper\n\n"
            "@timeit\n"
            "def slow(x):\n"
            "    return sum(range(x))\n"
        )),
        # Class
        ("synt_class_a", "synthetic", (
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self.n = 0\n"
            "    def inc(self):\n"
            "        self.n += 1\n"
            "    def value(self):\n"
            "        return self.n\n"
        )),
        ("synt_class_b", "synthetic", (
            "class Stack:\n"
            "    def __init__(self):\n"
            "        self._data = []\n"
            "    def push(self, x):\n"
            "        self._data.append(x)\n"
            "    def pop(self):\n"
            "        return self._data.pop()\n"
            "    def __len__(self):\n"
            "        return len(self._data)\n"
        )),
        # Comprehensions
        ("synt_comp_a", "synthetic", (
            "def even_squares(xs):\n"
            "    return [x * x for x in xs if x % 2 == 0]\n\n"
            "def char_freq(s):\n"
            "    return {c: s.count(c) for c in set(s)}\n"
        )),
        ("synt_comp_b", "synthetic", (
            "def flatten(matrix):\n"
            "    return [item for row in matrix for item in row]\n\n"
            "def primes_up_to(n):\n"
            "    return [p for p in range(2, n) if all(p % d for d in range(2, p))]\n"
        )),
        # Control flow
        ("synt_ctrl_a", "synthetic", (
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
        # Imports only
        ("synt_imports_a", "synthetic", (
            "import os\n"
            "import sys\n"
            "import json\n"
            "from pathlib import Path\n"
            "from typing import List, Dict, Optional\n"
        )),
        ("synt_imports_b", "synthetic", (
            "import numpy as np\n"
            "import pandas as pd\n"
            "from collections import defaultdict, Counter\n"
            "from itertools import chain, islice\n"
        )),
        # Bare function (long, hits >500 bin)
        ("synt_long_fn_a", "synthetic", (
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
        # Comprehension-only snippets (module-level)
        ("synt_comp_top_a", "synthetic", (
            "SQUARES = [i * i for i in range(100)]\n"
            "EVEN_SQUARES = [s for s in SQUARES if s % 2 == 0]\n"
            "CHAR_FREQ = {c: 'abracadabra'.count(c) for c in set('abracadabra')}\n"
        )),
        ("synt_comp_top_b", "synthetic", (
            "MATRIX = [[i * j for j in range(10)] for i in range(10)]\n"
            "FLAT = [x for row in MATRIX for x in row]\n"
            "UNIQUE = {item for row in MATRIX for item in row if item > 5}\n"
        )),
        # Top-level ForIfTry snippets
        ("synt_ctrl_top_a", "synthetic", (
            "for i in range(5):\n"
            "    if i % 2 == 0:\n"
            "        print(i)\n"
            "    else:\n"
            "        pass\n"
        )),
        ("synt_ctrl_top_b", "synthetic", (
            "try:\n"
            "    with open('x.txt') as f:\n"
            "        data = f.read()\n"
            "except FileNotFoundError:\n"
            "    data = ''\n"
        )),
        # Two more class fixtures to make ClassDef >500 bin
        ("synt_class_c", "synthetic", (
            "class LRUCache:\n"
            "    def __init__(self, capacity: int):\n"
            "        self.cap = capacity\n"
            "        self.cache = {}\n"
            "        self.order = []\n"
            "\n"
            "    def get(self, key):\n"
            "        if key not in self.cache:\n"
            "            return -1\n"
            "        self.order.remove(key)\n"
            "        self.order.append(key)\n"
            "        return self.cache[key]\n"
            "\n"
            "    def put(self, key, value):\n"
            "        if key in self.cache:\n"
            "            self.order.remove(key)\n"
            "        elif len(self.cache) >= self.cap:\n"
            "            oldest = self.order.pop(0)\n"
            "            del self.cache[oldest]\n"
            "        self.cache[key] = value\n"
            "        self.order.append(key)\n"
        )),
    ]
    return fixtures


# ---------- Main pipeline -----------------------------------------------

def collect_segments(
    humaneval_path: str,
    gsm8k_path: str,
    *,
    humaneval_n: int = 40,
    gsm8k_n: int = 20,
    seed: int = 17,
) -> list[CodeSegment]:
    rng = random.Random(seed)
    sources: list[tuple[str, str, str]] = []
    sources += _load_humaneval(humaneval_path, humaneval_n)
    sources += _load_gsm8k(gsm8k_path, gsm8k_n)
    sources += _synthetic_fixtures()

    segments: list[CodeSegment] = []
    for source_id, template, text in sources:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        extracted = _comprehension_aware_extract(tree, lines)
        for idx, (ast_type, snippet, s_line, e_line) in enumerate(extracted):
            tokens = _approximate_token_count(snippet)
            seg_id = f"{source_id}__{ast_type}__{s_line}-{e_line}"
            segments.append(
                CodeSegment(
                    seg_id=seg_id,
                    ast_type=ast_type,
                    template=template,
                    length_bin=_length_bin(tokens),
                    token_count=tokens,
                    source=snippet,
                    source_id=source_id,
                    metadata={"start_line": s_line, "end_line": e_line, "n_extracted": len(extracted)},
                )
            )

    # Print summary
    by_type = Counter(s.ast_type for s in segments)
    by_template = Counter(s.template for s in segments)
    by_bin = Counter(s.length_bin for s in segments)
    print(f"[ast_sampler] total segments: {len(segments)}")
    print(f"[ast_sampler] by ast_type: {dict(by_type)}")
    print(f"[ast_sampler] by template: {dict(by_template)}")
    print(f"[ast_sampler] by length_bin: {dict(by_bin)}")
    return segments


def save(segments: Iterable[CodeSegment], path: str) -> None:
    with open(path, "w") as f:
        json.dump([s.to_dict() for s in segments], f, indent=2, ensure_ascii=False)
    print(f"[ast_sampler] saved {len(list(segments)) if not isinstance(segments, list) else len(segments)} segments to {path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--humaneval", default="/home/gfy/KVCOMM/datasets/humaneval/humaneval-py.jsonl")
    p.add_argument("--gsm8k", default="/home/gfy/KVCOMM/datasets/gsm8k/gsm8k.jsonl")
    p.add_argument("--out", default="/home/gfy/CodeMAS_Project/sglang-kvflow/results/ast_kv_distance/data/segments.json")
    p.add_argument("--humaneval-n", type=int, default=40)
    p.add_argument("--gsm8k-n", type=int, default=20)
    args = p.parse_args()
    segs = collect_segments(args.humaneval, args.gsm8k, humaneval_n=args.humaneval_n, gsm8k_n=args.gsm8k_n)
    save(segs, args.out)
