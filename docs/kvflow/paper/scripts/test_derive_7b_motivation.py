"""Motivation deriver must use PLAN token ids, not a hard-coded fraction."""

from __future__ import annotations

from derive_7b_motivation import (
    DEFAULT_PLAN,
    analyze_plan,
    best_source_lcp,
    group_coverage_series,
    lcp_len,
)


def test_lcp_stops_at_first_mismatch() -> None:
    assert lcp_len([1, 2, 3, 9], [1, 2, 8, 9]) == 2
    assert lcp_len([], [1]) == 0


def test_disjoint_radix_and_island_on_synthetic_plan(tmp_path) -> None:
    plan = tmp_path / "PLAN.json"
    source = [1, 2, 3, 10, 11, 12, 99]
    target = [1, 2, 3, 7, 10, 11, 12, 8]
    plan.write_text(
        __import__("json").dumps(
            {
                "model": "Qwen2.5-Coder-7B-Instruct",
                "groups": [
                    {
                        "group_index": 0,
                        "target_input_ids": target,
                        "source_input_ids": [source],
                        "copied_tokens": 3,
                        "islands": 1,
                        "cases": [
                            {
                                "target_start": 4,
                                "source_start": 3,
                                "length": 3,
                                "content_hash": "seg",
                                "source_prefix_token_hash": "pre",
                            }
                        ],
                    }
                ],
            }
        )
        + "\n"
    )
    stats = analyze_plan(plan)
    group = __import__("json").loads(plan.read_text())["groups"][0]
    assert stats["lcp_island_overlap_tokens"] == 0
    assert stats["disjoint_radix_and_file_islands"] is True
    assert stats["groups_with_both_radix_and_lossy"] == 1
    assert best_source_lcp(group) == 3
    assert abs(stats["mean_radix_fraction"] - 3 / 8) < 1e-12
    assert abs(stats["mean_lossy_fraction"] - 3 / 8) < 1e-12


def test_frozen_7b_plan_radix_and_file_islands_are_disjoint() -> None:
    stats = analyze_plan(DEFAULT_PLAN)
    assert stats["groups"] == 235
    assert stats["lcp_island_overlap_tokens"] == 0
    assert stats["disjoint_radix_and_file_islands"] is True
    assert stats["groups_with_both_radix_and_lossy"] == 235
    assert stats["mean_lossy_fraction"] > stats["mean_radix_fraction"] > 0.2
    series = group_coverage_series(DEFAULT_PLAN)
    assert len(series["copied_frac"]) == 235
    assert min(series["rest_frac"]) >= 0.0
    assert abs(sum(series["copied_tokens"]) / 235 - stats["mean_copied_tokens"]) < 1e-9
