"""7B SWE-bench PLAN relocates islands and does not keep 30B token ids."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

from benchmark.multi_workflow.prepare_7b_swebench_file_modules_plan import (
    MODEL_7B,
    TOK_7B,
    locate_text_span,
)


def test_locate_text_span_keeps_nonzero_middle() -> None:
    tok = Tokenizer.from_file(str(TOK_7B))
    full = "HEAD file.py\nprint(1)\nprint(2)\nTAIL"
    piece = "print(1)\nprint(2)"
    span = locate_text_span(tok, full, piece, 0.4)
    assert span is not None
    start, length = span
    ids = list(tok.encode(full).ids)
    assert start > 0
    assert start + length < len(ids)
    assert length > 0


def test_locate_text_span_rejects_missing_piece() -> None:
    tok = Tokenizer.from_file(str(TOK_7B))
    assert locate_text_span(tok, "abc def", "not-present", 0.0) is None


def test_7b_rope_base_is_one_million() -> None:
    from benchmark.multi_workflow.template_prefetch_modes import rope_for_model

    seven = rope_for_model("/models/Qwen2.5-Coder-7B-Instruct")
    thirty = rope_for_model("/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit")
    assert seven["base"] == 1_000_000
    assert thirty["base"] == 10_000_000


def test_7b_tokenizer_is_not_30b_file() -> None:
    assert TOK_7B.name == "tokenizer.json"
    assert "Qwen2.5-Coder-7B-Instruct" in str(TOK_7B)
    assert MODEL_7B == "Qwen2.5-Coder-7B-Instruct"
    assert Path(__file__).parent.joinpath(
        "prepare_7b_swebench_file_modules_plan.py"
    ).is_file()
