from __future__ import annotations

import pytest
from tokenizers import Tokenizer

from benchmark.multi_workflow.analyze_prompt_module_attention_kv import (
    aggregate_case_module_rows,
    category_distribution,
    largest_matrix_changes,
)
from benchmark.multi_workflow.build_prompt_module_full_prompt_appendix import (
    MODEL,
    annotated_target,
    fence,
    validate_blocks,
)


def test_category_distribution_merges_blocks_in_the_same_module() -> None:
    blocks = {
        "a": {"category": "system_instruction"},
        "b": {"category": "other_tool_result"},
        "c": {"category": "other_tool_result"},
    }
    result = category_distribution({"a": 0.2, "b": 0.3, "c": 0.5}, blocks)
    assert result == {"system_instruction": 0.2, "other_tool_result": 0.8}


def test_case_module_aggregation_is_token_weighted() -> None:
    rows = [
        {
            "case_id": "case",
            "query_module": "assistant_action",
            "query_tokens": 1,
            "row_tv": 0.1,
            "dense_copied_mass": 0.2,
            "reuse_copied_mass": 0.1,
            "copied_mass_delta": -0.1,
            "raw_kv_drift": 0.2,
            "key_cosine_drift": 0.1,
            "value_cosine_drift": 0.2,
            "attention_times_drift": 0.04,
        },
        {
            "case_id": "case",
            "query_module": "assistant_action",
            "query_tokens": 3,
            "row_tv": 0.5,
            "dense_copied_mass": 0.6,
            "reuse_copied_mass": 0.5,
            "copied_mass_delta": -0.1,
            "raw_kv_drift": 0.4,
            "key_cosine_drift": 0.3,
            "value_cosine_drift": 0.4,
            "attention_times_drift": 0.24,
        },
    ]
    result = aggregate_case_module_rows(rows)
    assert len(result) == 1
    assert result[0]["row_tv"] == pytest.approx(0.4)
    assert result[0]["dense_copied_mass"] == pytest.approx(0.5)
    assert result[0]["attention_times_drift"] == pytest.approx(0.19)


def test_largest_matrix_changes_are_sorted_by_absolute_delta() -> None:
    dense = {"assistant_action": {"system_instruction": 0.4, "user_task": 0.6}}
    reuse = {"assistant_action": {"system_instruction": 0.3, "user_task": 0.7}}
    changes = largest_matrix_changes(
        dense, reuse, ("assistant_action",), limit=2
    )
    assert len(changes) == 2
    assert all(abs(row["delta_percentage_points"]) == pytest.approx(10.0) for row in changes)


def test_annotated_target_preserves_all_decoded_prompt_text() -> None:
    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    ids = tokenizer.encode("<|im_start|>system\nrule<|im_end|>\n", add_special_tokens=False).ids
    annotation, prefix, reused, suffix = annotated_target(
        tokenizer,
        ids,
        target_start=2,
        source_start=7,
        length=3,
    )
    assert prefix + reused + suffix == tokenizer.decode(ids, skip_special_tokens=False)
    assert "[[LOSSY_REUSE_BEGIN" in annotation
    assert "source=[7,10)" in annotation


def test_prompt_block_validation_and_dynamic_fence() -> None:
    validate_blocks(
        [
            {"block_id": "a", "start": 0, "end": 2, "tokens": 2},
            {"block_id": "b", "start": 2, "end": 5, "tokens": 3},
        ],
        5,
    )
    fenced = fence("inside ~~~~ marker")
    assert fenced.startswith("~~~~~text\n")
    assert fenced.endswith("\n~~~~~")
