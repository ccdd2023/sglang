"""General LCS PLAN keeps 96092 token ids and only admits Δ≠0 middles."""

from __future__ import annotations

from pathlib import Path

from benchmark.multi_workflow.prepare_swebench_general_lcs_plan import (
    PLAN_96092,
    build_groups,
    longest_shifted_middle,
)


def test_longest_shifted_middle_rejects_prefix_and_zero_shift() -> None:
    source = [1, 2, 3, 4, 5, 9]
    target = [1, 2, 3, 4, 5, 8]
    assert longest_shifted_middle(source, target) is None
    source = [9, 2, 3, 4, 5, 1]
    target = [8, 2, 3, 4, 5, 0]
    assert longest_shifted_middle(source, target) is None


def test_longest_shifted_middle_picks_nonzero_delta() -> None:
    source = [1, 2, 3, 4, 5, 8, 9]
    target = [1, 7, 2, 3, 4, 5, 9]
    span = longest_shifted_middle(source, target)
    assert span == (1, 2, 4)
    s, t, n = span
    assert source[s : s + n] == target[t : t + n]


def test_build_groups_drops_overlapping_target_spans() -> None:
    # Both sources keep a Δ≠0 strict-middle run; the shorter is nested in the longer.
    source_a = [99, 10, 11, 12, 13, 14, 15, 88]
    source_b = [77, 20, 11, 12, 13, 14, 15, 66]
    target = [1, 2, 10, 11, 12, 13, 14, 15, 3]
    span_a = longest_shifted_middle(source_a, target)
    span_b = longest_shifted_middle(source_b, target)
    assert span_a == (1, 2, 6)
    assert span_b == (2, 3, 5)
    assert span_a[1] < span_b[1] + span_b[2] and span_b[1] < span_a[1] + span_a[2]
    official = [
        {
            "group_index": 2,
            "original_target_group_id": "overlap-g2",
            "target_prompt_hash": "h",
            "target_input_ids": target,
            "source_prompt_hashes": ["a", "b"],
            "source_input_ids": [source_a, source_b],
            "sources": [],
            "cases": [],
            "islands": 2,
            "copied_tokens": 0,
            "pre_rotate_delta": -1,
        }
    ]
    groups = build_groups(official)
    starts = [case["target_start"] for case in groups[0]["cases"]]
    lengths = [case["length"] for case in groups[0]["cases"]]
    for left, right in zip(starts, starts[1:]):
        assert left < right
    for start, length, nxt in zip(starts, lengths, starts[1:]):
        assert start + length <= nxt
    assert groups[0]["islands"] == 1


def test_general_plan_keeps_96092_targets_and_changes_spans() -> None:
    import json

    official = json.loads(PLAN_96092.read_text(encoding="utf-8"))["groups"][:8]
    groups = build_groups(official)
    assert len(groups) == 8
    longer = 0
    for ours, theirs in zip(groups, official):
        assert ours["target_input_ids"] == theirs["target_input_ids"]
        assert ours["cases"][0]["target_start"] != ours["cases"][0]["source_start"]
        if ours["copied_tokens"] > theirs["copied_tokens"]:
            longer += 1
        starts = [case["target_start"] for case in ours["cases"]]
        lengths = [case["length"] for case in ours["cases"]]
        for start, length, nxt in zip(starts, lengths, starts[1:]):
            assert start + length <= nxt
        src = ours["source_input_ids"][0]
        tgt = ours["target_input_ids"]
        ss = ours["cases"][0]["source_start"]
        ts = ours["cases"][0]["target_start"]
        n = ours["cases"][0]["length"]
        assert src[ss : ss + n] == tgt[ts : ts + n]
        assert ss > 0 and ts > 0
    assert longer >= 1
    assert Path(__file__).parent.joinpath(
        "prepare_swebench_general_lcs_plan.py"
    ).is_file()
