"""
Unit tests for ``python/sglang/srt/mem_cache/ast_chunker.py``.

These tests verify:
1. The chunker parses common Python structures correctly.
2. Byte-offset math matches MAScoder's ``PythonCodeAnchorExtractor``
   (the regression test catches mirror-drift).
3. Edge cases (empty input, syntax error, leading whitespace) are handled.

Run:
    python -m pytest test/registered/unit/mem_cache/test_ast_chunker.py -v

The MAScoder parity test (#3) needs MAScoder to be importable. We
sys.path.insert the MAScoder src directory if not already on path;
CI may pre-install MAScoder as an editable dep.
"""

from __future__ import annotations

import os
import sys
import textwrap
import unittest

# Ensure MAScoder is importable for the parity test. If the repo lives
# next to sglang-kvflow in the user's checkout, we pick it up here.
_MASCODER_SRC = "/home/gfy/CodeMAS_Project/MAScoder/src"
if os.path.isdir(_MASCODER_SRC) and _MASCODER_SRC not in sys.path:
    sys.path.insert(0, _MASCODER_SRC)

from sglang.srt.mem_cache.ast_chunker import (  # noqa: E402
    ASTChunker,
    ChunkSpan,
    chunk_text,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=10, suite="stage-b-test-small-1-gpu-amd")


class TestASTChunkerSimple(unittest.TestCase):
    def test_simple_function_chunk(self):
        """Single ``def`` produces one function chunk with byte span covering the body."""
        text = "def foo():\n    return 1\n"
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 1)
        c = chunks[0]
        self.assertEqual(c.anchor_type, "function")
        self.assertEqual(c.name, "foo")
        self.assertEqual(c.start_line, 1)
        self.assertEqual(c.end_line, 2)
        self.assertEqual(c.byte_start, 0)
        # byte_end is the byte at end_col_offset of the last line. With
        # "    return 1\n", the end_lineno is 2 and end_col_offset is 12
        # (just past "    return 1"). So byte_end = leading_offset(0) +
        # line_byte_offsets[1] + end_col_offset(12). line_byte_offsets[1]
        # is len("def foo():\n") = 11. byte_end = 0 + 11 + 12 = 23.
        self.assertEqual(c.byte_end, 23)
        # signature is sha1[:16] of "python:function:foo:def foo() : return 1"
        self.assertEqual(len(c.signature), 16)
        self.assertEqual(c.nesting_depth, 0)

    def test_nested_class_chunk(self):
        """Class + method produces two chunks; method has nesting_depth=1."""
        text = textwrap.dedent(
            """\
            class Foo:
                def bar(self):
                    return 1
            """
        )
        chunks = chunk_text(text)
        types = [c.anchor_type for c in chunks]
        names = [c.name for c in chunks]
        self.assertIn("class", types)
        self.assertIn("function", types)
        self.assertIn("Foo", names)
        self.assertIn("bar", names)

        foo_class = next(c for c in chunks if c.anchor_type == "class")
        bar_method = next(c for c in chunks if c.anchor_type == "function")
        self.assertEqual(foo_class.nesting_depth, 0)
        self.assertEqual(bar_method.nesting_depth, 1)
        # Class spans lines 1-3, method spans 2-3.
        self.assertEqual(foo_class.start_line, 1)
        self.assertEqual(bar_method.start_line, 2)

    def test_anchored_byte_offsets_match_mascoder(self):
        """Parity check: chunker byte ranges agree byte-for-byte with
        MAScoder's ``PythonCodeAnchorExtractor`` on the same input.

        This is the mirror-drift regression test. If MAScoder's byte-offset
        math changes, this test catches it (and a corresponding update to
        ``ast_chunker.py`` is required).
        """
        try:
            from mascoder.code_anchor import PythonCodeAnchorExtractor
        except ImportError as exc:
            self.skipTest(f"MAScoder not importable in this env: {exc}")

        text = textwrap.dedent(
            """\
            import os

            def top_level(arg: int) -> int:
                if arg > 0:
                    return arg
                return -arg

            class Helper:
                def run(self):
                    for i in range(3):
                        print(i)
            """
        )

        ours = ASTChunker().chunk_text(text)
        theirs = PythonCodeAnchorExtractor().extract(text)

        # Build a signature -> ChunkSpan map for fast lookup.
        ours_by_sig = {c.signature: c for c in ours}
        # MAScoder's signature seed is identical ("lang:type:name:normalized")
        # so signatures agree when names + types agree.
        theirs_by_sig = {a.signature: a for a in theirs}

        # Both should produce the same set of signatures for this input.
        self.assertEqual(
            set(ours_by_sig.keys()),
            set(theirs_by_sig.keys()),
            msg=(
                "ASTChunker and MAScoder produced different signature sets.\n"
                f"ours:   {sorted(ours_by_sig.keys())}\n"
                f"theirs: {sorted(theirs_by_sig.keys())}"
            ),
        )

        for sig, our_chunk in ours_by_sig.items():
            their_anchor = theirs_by_sig[sig]
            self.assertEqual(
                our_chunk.byte_start,
                their_anchor.byte_start,
                msg=f"byte_start mismatch for signature={sig}",
            )
            self.assertEqual(
                our_chunk.byte_end,
                their_anchor.byte_end,
                msg=f"byte_end mismatch for signature={sig}",
            )
            self.assertEqual(our_chunk.start_line, their_anchor.start_line)
            self.assertEqual(our_chunk.end_line, their_anchor.end_line)
            self.assertEqual(
                our_chunk.nesting_depth,
                their_anchor.nesting_depth,
                msg=f"nesting_depth mismatch for signature={sig}",
            )

    def test_empty_text_returns_empty(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n  \n"), [])

    def test_syntax_error_returns_empty(self):
        text = "def foo(:\n    return 1\n"  # invalid syntax
        self.assertEqual(chunk_text(text), [])

    def test_byte_start_zero_leading_offset(self):
        """Input with no leading whitespace: byte_start=0."""
        text = "def f():\n    pass\n"
        chunks = chunk_text(text)
        self.assertEqual(chunks[0].byte_start, 0)

    def test_byte_start_nonzero_leading_offset(self):
        """Input with leading whitespace: byte_start=0 because chunk_text
        strips the input first (matches MAScoder's behavior at
        ``code_anchor.py:58``). Use ``chunk_text_strict`` (Phase B) or
        pre-strip the input yourself if you need leading-offset preservation.
        """
        # 4 leading spaces then a function.
        text = "    def f():\n        pass\n"
        chunks = chunk_text(text)
        # ``chunk_text`` strips input (code_anchor.py:58 / ast_chunker.py
        # mirror); leading_offset = 0; byte_start = 0 + 0 + 0 = 0.
        self.assertEqual(chunks[0].byte_start, 0)

    def test_no_end_lineno_attribute(self):
        """AST nodes without ``end_lineno`` fall back to ``lineno``."""
        # Use a real parse then monkeypatch the resulting node to drop
        # end_lineno/end_col_offset, simulating older Python (<3.8) ASTs.
        import ast

        text = "def synthetic():\n    pass\n"
        tree = ast.parse(text)
        fn_node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        # Pre-3.8 AST nodes don't carry end_lineno; delete to simulate.
        if hasattr(fn_node, "end_lineno"):
            del fn_node.end_lineno
        if hasattr(fn_node, "end_col_offset"):
            del fn_node.end_col_offset

        chunker = ASTChunker()
        # Build the lines/byte_offsets manually since we bypass chunk_text.
        from sglang.srt.mem_cache.ast_chunker import (
            _build_chunk_span,
            _nesting_depth,
        )

        lines = text.splitlines()
        line_byte_offsets: list[int] = []
        running = 0
        for line in lines:
            line_byte_offsets.append(running)
            running += len(line) + 1
        parents: dict[int, ast.AST] = {id(fn_node): None}
        chunk = _build_chunk_span(
            language="python",
            anchor_type="function",
            name=fn_node.name,
            snippet="\n".join(lines),
            node=fn_node,
            parents=parents,
            byte_offsets=line_byte_offsets,
            leading_offset=0,
        )
        # Without end_lineno, fallback to lineno for both start and end.
        self.assertEqual(chunk.start_line, fn_node.lineno)
        self.assertEqual(chunk.end_line, fn_node.lineno)
        self.assertEqual(chunk.nesting_depth, 0)

    def test_chunk_for_anchor_from_mascoder_dict(self):
        """``ChunkSpan`` dataclass round-trips through ``dataclasses.asdict``.

        (We don't directly consume a MAScoder to_dict() because the
        server-side chunker operates on text, not on MAScoder payloads.
        This test documents the dataclass shape.)
        """
        from dataclasses import asdict

        text = "def only():\n    return 0\n"
        chunks = chunk_text(text)
        d = asdict(chunks[0])
        expected_keys = {
            "byte_start",
            "byte_end",
            "start_line",
            "end_line",
            "anchor_type",
            "name",
            "signature",
            "nesting_depth",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_byte_span_is_zero_indexed(self):
        """First chunk of input that begins with ``def`` has byte_start=0."""
        text = "def alpha():\n    pass\n\ndef beta():\n    pass\n"
        chunks = chunk_text(text)
        first = chunks[0]
        self.assertEqual(first.byte_start, 0)
        self.assertEqual(first.name, "alpha")
        # Second chunk should start after the first chunk + blank line.
        second = chunks[1]
        self.assertEqual(second.name, "beta")
        self.assertGreater(second.byte_start, first.byte_end)


class TestASTChunkerEdgeCases(unittest.TestCase):
    def test_control_block_chunks(self):
        """For/While/If/Try produce anchor_type matching the lowercase node type."""
        text = textwrap.dedent(
            """\
            def f():
                for i in range(3):
                    pass
                while True:
                    break
                if True:
                    pass
                try:
                    pass
                except Exception:
                    pass
            """
        )
        chunks = chunk_text(text)
        types = sorted(c.anchor_type for c in chunks if c.anchor_type != "function")
        self.assertEqual(types, ["for", "if", "try", "while"])

    def test_caps_at_max_anchors(self):
        """Lots of top-level functions → truncated to 32 anchors."""
        text = "\n".join(f"def f{i}():\n    return {i}\n" for i in range(50))
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 32)

    def test_chunks_are_sorted_by_start_line(self):
        text = textwrap.dedent(
            """\
            def zzz():
                pass
            def aaa():
                pass
            """
        )
        chunks = chunk_text(text)
        starts = [c.start_line for c in chunks]
        self.assertEqual(starts, sorted(starts))


if __name__ == "__main__":
    unittest.main()