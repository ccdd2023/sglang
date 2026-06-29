"""
Server-side AST chunker for Direction #3 (AST-boundary chunked prefill).

This is a server-side mirror of MAScoder's
``MAScoder/src/mascoder/code_anchor.py`` (branch ``feature/code-anchor-integration``,
commit ``f244398``, 2026-06-27). The byte-offset math and signature seed
formulas are copied verbatim from that file so that the two implementations
agree byte-for-byte on the same input. Keep the two in sync.

Why mirror instead of import: sglang ships as a self-contained wheel;
importing MAScoder at request time would create a hard runtime dependency
on a non-pip package. The cost of duplication is small (~150 LOC), and the
``test_anchored_byte_offsets_match_mascoder`` regression test in
``test/registered/unit/mem_cache/test_ast_chunker.py`` catches drift.

Phase A (this file): pure-Python stdlib ``ast`` chunker + ``ChunkSpan``
dataclass. No tree-sitter, no multi-language. Read-path wiring (Phase B)
will live in ``radix_cache.py::_store_placeholder_anchor_kv``.

Public API:
    ChunkSpan            — frozen dataclass for one AST-aligned chunk
    ASTChunker           — chunker for Python source
    chunk_text           — convenience wrapper

Usage:
    from sglang.srt.mem_cache.ast_chunker import ASTChunker, ChunkSpan

    chunker = ASTChunker()
    chunks = chunker.chunk_text(source_text)
    for c in chunks:
        # c.byte_start, c.byte_end, c.signature, ...
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ChunkSpan:
    """One AST-aligned chunk of source text.

    ``byte_start``/``byte_end`` are absolute byte offsets within the
    original (un-stripped) input string. ``start_line``/``end_line`` are
    1-based AST line numbers (matches ``ast.Node.lineno``). ``signature``
    is the same sha1[:16] seed MAScoder uses for its
    ``CodeAnchor.signature`` field, so consumers can join by signature
    without recomputing.
    """

    byte_start: int
    byte_end: int
    start_line: int
    end_line: int
    anchor_type: str  # "function" | "class" | "for" | "while" | "if" | "try"
    name: str
    signature: str  # sha1[:16] of "lang:type:name:normalized"
    nesting_depth: int


# Maximum anchors per parse, mirroring MAScoder's bounds at line 107.
# Keeps request payload sizes stable for very long files.
_MAX_ANCHORS = 32


def _normalize_snippet(text: str) -> str:
    """Mirror of MAScoder._normalize_snippet (code_anchor.py:206-208).

    Collapses whitespace via ``re.sub(r"\s+", " ", ...)`` and truncates
    to 240 chars. Used as input to the ``signature`` sha1 seed.
    """
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    return normalized[:240]


def _slice_source(lines: list[str], start: int, end: int) -> str:
    """Mirror of MAScoder._slice_source (code_anchor.py:211-214)."""
    lo = max(1, start)
    hi = max(lo, end)
    return "\n".join(lines[lo - 1 : hi])


def _nesting_depth(parents: dict[int, ast.AST], node: ast.AST) -> int:
    """Mirror of MAScoder._nesting_depth (code_anchor.py:217-228).

    Each FunctionDef/AsyncFunctionDef/ClassDef/Lambda ancestor adds 1.
    """
    depth = 0
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(
            cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            depth += 1
        cur = parents.get(id(cur))
    return depth


def _build_chunk_span(
    language: str,
    anchor_type: str,
    name: str,
    snippet: str,
    node: ast.AST,
    parents: dict[int, ast.AST],
    byte_offsets: list[int],
    leading_offset: int,
) -> ChunkSpan:
    """Build one ChunkSpan. Mirror of MAScoder._build_anchor
    (code_anchor.py:231-268) but emitting a ChunkSpan instead of CodeAnchor.
    """
    normalized = _normalize_snippet(snippet)
    exact_text = snippet or ""
    signature_seed = f"{language}:{anchor_type}:{name}:{normalized}"
    signature = hashlib.sha1(signature_seed.encode("utf-8")).hexdigest()[:16]

    start_line = getattr(node, "lineno", 0) or 0
    end_line = getattr(node, "end_lineno", start_line) or start_line

    byte_start = 0
    byte_end = 0
    if byte_offsets and 0 < start_line <= len(byte_offsets):
        line_off = byte_offsets[start_line - 1]
        col_off = getattr(node, "col_offset", 0) or 0
        byte_start = leading_offset + line_off + col_off
    if byte_offsets and 0 < end_line <= len(byte_offsets):
        end_line_off = byte_offsets[end_line - 1]
        end_col_off = getattr(node, "end_col_offset", 0) or 0
        byte_end = leading_offset + end_line_off + end_col_off

    return ChunkSpan(
        byte_start=byte_start,
        byte_end=byte_end,
        start_line=start_line,
        end_line=end_line,
        anchor_type=anchor_type,
        name=name,
        signature=signature,
        nesting_depth=_nesting_depth(parents, node),
    )


class ASTChunker:
    """Chunker for Python source code (stdlib ast only).

    Mirrors MAScoder's ``PythonCodeAnchorExtractor.extract``
    (code_anchor.py:57-107) but produces ``ChunkSpan`` objects instead of
    ``CodeAnchor``. The byte-offset math is identical, so consumers can
    cross-reference anchors by ``signature`` (which uses the same seed
    formula).
    """

    language: str = "python"

    # Process-wide cache: text-hash → chunk list. The same code text is chunked
    # by every agent in a multi-agent run (5 agents × 5 spans = 25 identical
    # AST parses + signature hashes per request); caching avoids that repeat
    # cost on the TTFT critical path. Bounded to _CHUNK_CACHE_MAX entries (LRU
    # via dict insertion-order eviction).
    _chunk_cache: "dict[int, list[ChunkSpan]]" = {}
    _CHUNK_CACHE_MAX = 512

    def chunk_text(self, text: str) -> list[ChunkSpan]:
        """Parse ``text`` and return up to ``_MAX_ANCHORS`` AST-aligned chunks.

        Returns ``[]`` for empty input or syntax errors.

        Coarse mode (``SGLANG_CHUNK_COARSE=1``): return a SINGLE chunk spanning
        the entire text (anchor_type="module"). One chunk = one whole-slot copy
        in the C2 read path, so a 7000-token code_base is copied in ONE
        alloc+move+RoPE (fast, like L3's whole-slot copy) instead of ~88
        per-function copies whose per-chunk overhead negates the reuse speedup.
        The signature still derives from the (normalized) text, so byte-exact
        matching across requests is preserved.
        """
        source = (text or "").strip()
        if not source:
            return []
        # Cache lookup (keyed by text hash + the two granularity env flags so
        # toggling coarse/toplevel invalidates). The AST parse + signature
        # hashing is the dominant per-request cost in the C2 read path; the
        # same code text is chunked by every agent, so this turns 25 parses
        # into 5.
        cache_key = hash((
            text,
            os.environ.get("SGLANG_CHUNK_COARSE", "0"),
            os.environ.get("SGLANG_CHUNK_TOPLEVEL", "0"),
        ))
        cached = ASTChunker._chunk_cache.get(cache_key)
        if cached is not None:
            return cached
        result = self._chunk_text_uncached(text, source)
        if len(ASTChunker._chunk_cache) >= ASTChunker._CHUNK_CACHE_MAX:
            ASTChunker._chunk_cache.pop(next(iter(ASTChunker._chunk_cache)))
        ASTChunker._chunk_cache[cache_key] = result
        return result

    def _chunk_text_uncached(self, text: str, source: str) -> list[ChunkSpan]:
        if os.environ.get("SGLANG_CHUNK_COARSE", "0") == "1":
            normalized = _normalize_snippet(source)
            signature = hashlib.sha1(
                f"{self.language}:module:module:{normalized}".encode("utf-8")
            ).hexdigest()[:16]
            return [
                ChunkSpan(
                    byte_start=0,
                    byte_end=len(text or ""),
                    start_line=1,
                    end_line=len(source.splitlines()) or 1,
                    anchor_type="module",
                    name="module",
                    signature=signature,
                    nesting_depth=0,
                )
            ]
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        # Rebuild parent map (ast.walk loses it).
        parents: dict[int, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[id(child)] = parent

        lines = source.splitlines()
        # Pre-compute byte offsets per line for fast (line, col) -> byte.
        leading_offset = len(source) - len(source.lstrip())
        line_byte_offsets: list[int] = []
        running = 0
        for line in lines:
            line_byte_offsets.append(running)
            running += len(line) + 1  # +1 for the splitlines() drop of \n

        chunks: list[ChunkSpan] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                snippet = _slice_source(
                    lines, node.lineno, getattr(node, "end_lineno", node.lineno)
                )
                chunks.append(
                    _build_chunk_span(
                        self.language,
                        "function",
                        node.name,
                        snippet,
                        node,
                        parents,
                        line_byte_offsets,
                        leading_offset,
                    )
                )
            elif isinstance(node, ast.ClassDef):
                snippet = _slice_source(
                    lines, node.lineno, getattr(node, "end_lineno", node.lineno)
                )
                chunks.append(
                    _build_chunk_span(
                        self.language,
                        "class",
                        node.name,
                        snippet,
                        node,
                        parents,
                        line_byte_offsets,
                        leading_offset,
                    )
                )
            elif isinstance(node, (ast.For, ast.While, ast.If, ast.Try)):
                # Top-level mode (SGLANG_CHUNK_TOPLEVEL=1): skip control-flow
                # chunks, keeping only function/class chunks. Fewer, larger
                # chunks → fewer copies in the C2 read path → lower per-chunk
                # overhead → better speedup, while staying AST-aligned.
                if os.environ.get("SGLANG_CHUNK_TOPLEVEL", "0") == "1":
                    continue
                node_type = type(node).__name__.lower()
                snippet = _slice_source(
                    lines, node.lineno, getattr(node, "end_lineno", node.lineno)
                )
                chunks.append(
                    _build_chunk_span(
                        self.language,
                        node_type,
                        node_type,
                        snippet,
                        node,
                        parents,
                        line_byte_offsets,
                        leading_offset,
                    )
                )

        chunks.sort(key=lambda c: (c.start_line, c.anchor_type, c.name))
        return chunks[:_MAX_ANCHORS]


def chunk_text(text: str) -> list[ChunkSpan]:
    """Convenience wrapper around ``ASTChunker().chunk_text(text)``."""
    return ASTChunker().chunk_text(text)


__all__ = [
    "ASTChunker",
    "ChunkSpan",
    "chunk_text",
]