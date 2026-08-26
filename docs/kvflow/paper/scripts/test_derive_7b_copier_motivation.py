"""Copier motivation must read frozen PLANs and COMPLETE outputs."""

from __future__ import annotations

from derive_7b_copier_motivation import (
    ART,
    CODING_REUSE,
    SPAN_CLASSES,
    analyze,
    analyze_plans,
    classify_decoded_extra,
    coverage_mask,
    extra_spans,
    read_json,
    tokens_from_char_counts,
)


def test_coverage_mask_marks_only_the_span() -> None:
    mask = coverage_mask(8, [{"target_start": 2, "length": 3}])
    assert mask == [False, False, True, True, True, False, False, False]


def test_extra_spans_mark_file_adjacency() -> None:
    file_mask = [False, True, True, False, False, False]
    clone_mask = [True, True, True, True, False, True]
    spans = extra_spans(file_mask, clone_mask)
    assert spans == [(0, 1, True), (3, 4, True), (5, 6, False)]


def test_classify_decoded_extra_prefers_tool_xml() -> None:
    text = (
        "assistant\nLet me inspect the file.\n"
        "<tool_call>\n<function=bash>\n<parameter=command>\ncat a.py\n"
        "</parameter>\n</function>\n</tool_call>\n"
        "user\n<tool_response>\n<returncode>0</returncode>\n"
        "<output>\ndef f():\n    return 1\n</output>\n</tool_response>\n"
    )
    counts = classify_decoded_extra(text)
    assert set(counts) == set(SPAN_CLASSES)
    assert counts["tool_log"] > 0
    assert counts["tool_command"] > 0
    assert counts["assistant"] > 0
    assert sum(counts.values()) == len(text)


def test_tokens_from_char_counts_preserves_length() -> None:
    chars = classify_decoded_extra(
        "<tool_response>log</tool_response>assistant\nnote\n"
    )
    allocated = tokens_from_char_counts(17, chars)
    assert sum(allocated.values()) == 17
    assert allocated["tool_log"] > 0


def test_kvcomm_extra_tokens_are_computed_from_plans() -> None:
    coding = {
        "groups": [
            {
                "group_index": 0,
                "target_input_ids": list(range(10)),
                "cases": [{"target_start": 4, "length": 3}],
            }
        ]
    }
    kvcomm = {
        "groups": [
            {
                "group_index": 0,
                "target_input_ids": list(range(10)),
                "cases": [{"target_start": 2, "length": 6}],
            }
        ]
    }
    stats = analyze_plans(coding, kvcomm)
    assert stats["file_module_copied_tokens"] == 3
    assert stats["kvcomm_copied_tokens"] == 6
    assert stats["shared_copied_tokens"] == 3
    assert stats["kvcomm_extra_tokens"] == 3
    assert stats["groups_with_kvcomm_extra"] == 1


def test_frozen_137400_unconstrained_copies_extra_and_disagrees_more() -> None:
    stats = analyze(ART, CODING_REUSE)
    spans = stats["spans"]
    agr = stats["agreement"]
    result = read_json(ART / "RESULT.json")
    assert result["status"] == "COMPLETE"
    assert spans["kvcomm_extra_tokens"] > 0
    assert spans["kvcomm_copied_tokens"] > spans["file_module_copied_tokens"]
    assert spans["groups"] == 235
    assert spans["kvcomm_extra_tokens"] > spans["groups"] * 100
    assert spans["groups_with_kvcomm_extra"] == spans["groups"]
    assert agr["pairs"] == 705
    assert agr["file_module_agrees"] > agr["kvcomm_agrees"]
    assert agr["file_module_agrees"] > agr["cacheblend_agrees"]
    assert agr["file_agrees_kvcomm_differs"] > agr["kvcomm_agrees_file_differs"]
    assert agr["not_accuracy"] is True
    coding_frac = agr["file_module_agrees"] / agr["pairs"]
    assert abs(coding_frac - result["coding"]["one_token_output_agreement"]["fraction"]) < 1e-12
    kinds = stats["span_types"]
    campaign = kinds["campaign"]
    subset = kinds["file_agrees_kvcomm_differs"]
    assert kinds["not_a_new_gpu_arm"] is True
    assert kinds["not_admitted_repository_code"] is True
    assert campaign["extra_tokens"] == spans["kvcomm_extra_tokens"]
    assert sum(campaign["by_class"].values()) == campaign["extra_tokens"]
    assert campaign["by_class"]["tool_log"] > campaign["by_class"]["tool_command"]
    assert campaign["by_class"]["tool_command"] > campaign["by_class"]["assistant"]
    assert campaign["adjacent_to_file_island"] > campaign["disjoint_from_file_island"]
    assert agr["file_agrees_kvcomm_differs"] == 51
    assert subset["pairs"] == 51
    assert subset["groups"] == 17
    assert subset["all_three_measured_rounds"] is True
    assert subset["disjoint_from_file_island"] == 0
    assert subset["adjacent_to_file_island"] == subset["extra_tokens"]
    assert subset["by_class"]["tool_log"] > subset["by_class"]["assistant"]
