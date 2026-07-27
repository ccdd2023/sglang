from pathlib import Path

from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    ARMS,
    INSTANCE_IDS,
    assistant_request_prefixes,
    coarse_js,
    prepare,
    simulate_arm,
)


def test_assistant_prefixes_never_include_current_assistant():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "a2"},
    ]
    prefixes = assistant_request_prefixes(messages)
    assert [len(value) for value in prefixes] == [2, 4]
    assert all(value[-1]["role"] != "assistant" for value in prefixes)


def test_coarse_js_identity_and_symmetry():
    left = {1: 0.6, 2: 0.2}
    right = {1: 0.4, 3: 0.3}
    assert coarse_js(left, left) == 0
    assert coarse_js(left, right) == coarse_js(right, left)
    assert coarse_js(left, right) > 0


def test_registration_freezes_identical_prompt_hashes(tmp_path: Path):
    registration = prepare(tmp_path)
    plans = registration["plans"]
    dense = [
        (row["instance_id"], row["request_index"], row["prompt_hash"])
        for row in plans["dense"]
    ]
    assert {row[0] for row in dense} == set(INSTANCE_IDS)
    assert len(dense) > len(INSTANCE_IDS)
    for arm in ARMS[1:]:
        assert dense == [
            (row["instance_id"], row["request_index"], row["prompt_hash"])
            for row in plans[arm]
        ]


def test_version_graph_plan_is_not_general_alias():
    general = simulate_arm("general")
    graph = simulate_arm("coding_version_graph_v17")
    assert any(
        left["source_tokens_planned"] != right["source_tokens_planned"]
        or left["source_registered"] != right["source_registered"]
        for left, right in zip(general, graph)
    )
