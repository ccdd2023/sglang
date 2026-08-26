"""CacheBlend-style shrink keeps token equality and a strictly smaller copy."""

from __future__ import annotations

from benchmark.multi_workflow.prepare_7b_sota_copier_plans import (
    cacheblend_groups,
    shrink_island_for_blend,
)
from benchmark.multi_workflow.prepare_swebench_general_lcs_plan import (
    build_groups,
)


def test_shrink_keeps_delta_and_cuts_fifteen_percent() -> None:
    shrunk = shrink_island_for_blend(10, 40, 100, ratio=0.15)
    assert shrunk == (25, 55, 85, 15)
    assert shrunk[1] - shrunk[0] == 40 - 10


def test_shrink_rejects_too_short_span() -> None:
    assert shrink_island_for_blend(1, 4, 1) is None


def test_cacheblend_copies_fewer_tokens_than_unconstrained_lcs() -> None:
    source = [1] + list(range(10, 30)) + [9]
    target = [1, 7] + list(range(10, 30)) + [8]
    official = [
        {
            "group_index": 0,
            "original_target_group_id": "g0",
            "target_prompt_hash": "h",
            "target_input_ids": target,
            "source_prompt_hashes": ["s"],
            "source_input_ids": [source],
            "sources": [],
            "cases": [],
            "islands": 1,
            "copied_tokens": 0,
            "pre_rotate_delta": -1,
        }
    ]
    kvcomm = build_groups(official)
    blend = cacheblend_groups(kvcomm)
    assert len(blend) == 1
    assert blend[0]["copied_tokens"] < kvcomm[0]["copied_tokens"]
    assert blend[0]["blend_recompute_tokens"] > 0
    case = blend[0]["cases"][0]
    src = blend[0]["source_input_ids"][0]
    tgt = blend[0]["target_input_ids"]
    s = case["source_start"]
    t = case["target_start"]
    n = case["length"]
    assert src[s : s + n] == tgt[t : t + n]
    assert s != t
