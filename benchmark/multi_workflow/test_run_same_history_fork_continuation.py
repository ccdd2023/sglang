from __future__ import annotations

from benchmark.multi_workflow.run_same_history_fork_continuation import (
    assistant_message_at_request,
    assistant_request_prefixes,
    canonical_hash,
    prefix_actions,
    public_response_signature,
    resolved_ids,
)


def sample_messages():
    return [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {
            "role": "assistant",
            "content": None,
            "extra": {"actions": [{"command": "read one"}]},
        },
        {"role": "tool", "content": "one"},
        {
            "role": "assistant",
            "content": None,
            "extra": {"actions": [{"command": "read two"}]},
        },
        {"role": "tool", "content": "two"},
    ]


def test_assistant_request_prefixes_end_before_assistant():
    prefixes = assistant_request_prefixes(sample_messages())
    assert len(prefixes) == 2
    assert [row["role"] for row in prefixes[0]] == ["system", "user"]
    assert [row["role"] for row in prefixes[1]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]


def test_prefix_actions_excludes_fork_request():
    assert prefix_actions(sample_messages(), 2) == [{"command": "read one"}]


def test_canonical_hash_ignores_mapping_order():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_resolved_ids_reads_official_report():
    assert resolved_ids({"report": {"resolved_ids": ["a", "b"]}}) == {"a", "b"}
    assert resolved_ids({"report": None}) == set()


def test_assistant_message_and_public_signature():
    message = assistant_message_at_request({"messages": sample_messages()}, 2)
    assert public_response_signature(message) == {
        "content": None,
        "actions": [{"command": "read two"}],
    }
