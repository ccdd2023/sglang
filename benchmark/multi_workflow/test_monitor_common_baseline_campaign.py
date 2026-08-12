import pytest

from benchmark.multi_workflow.monitor_common_baseline_campaign import (
    first_request_identity_by_session,
)


def test_first_request_identity_by_session_uses_first_completed_request() -> None:
    rows = [
        {"event": "request_started", "session_index": 1},
        {
            "event": "request_complete",
            "session_index": 1,
            "messages_sha256": "messages-1-first",
            "input_ids_sha256": "tokens-1-first",
        },
        {
            "event": "request_complete",
            "session_index": 1,
            "messages_sha256": "messages-1-later",
            "input_ids_sha256": "tokens-1-later",
        },
        {
            "event": "request_complete",
            "session_index": 2,
            "messages_sha256": "messages-2-first",
            "input_ids_sha256": "tokens-2-first",
        },
    ]

    assert first_request_identity_by_session(rows) == {
        1: ("messages-1-first", "tokens-1-first"),
        2: ("messages-2-first", "tokens-2-first"),
    }


def test_first_request_identity_by_session_rejects_missing_identity() -> None:
    with pytest.raises(ValueError, match="missing input identity"):
        first_request_identity_by_session(
            [
                {
                    "event": "request_complete",
                    "session_index": 1,
                    "messages_sha256": "messages",
                }
            ]
        )
