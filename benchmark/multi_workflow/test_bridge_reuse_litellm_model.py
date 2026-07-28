from types import SimpleNamespace

import pytest

from benchmark.multi_workflow import bridge_reuse_litellm_model as bridge


class _UnderlyingStream:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _LiteLLMStream:
    def __init__(self, chunks) -> None:
        self._chunks = chunks
        self.completion_stream = _UnderlyingStream()

    def __iter__(self):
        return iter(self._chunks)


def _bare_model() -> bridge.BridgeReuseLitellmModel:
    model = object.__new__(bridge.BridgeReuseLitellmModel)
    model.config = SimpleNamespace(model_kwargs={}, model_name="test-model")
    model._instance_nonce = "test"
    model._session_index = 0
    model._request_index = 1
    model._last_stream_stats = {}
    return model


def test_v33b_state_transition_releases_and_vetoes_current_target() -> None:
    mutation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": (
                                "python -c \"from pathlib import Path; "
                                "Path('pkg/a.py').write_text('x')\""
                            )
                        },
                    }
                }
            ],
        },
        {"role": "tool", "content": "<returncode>0</returncode>"},
    ]
    target = {"source_id": "source-1"}

    guarded, releases, decision = bridge.apply_current_target_veto(
        arm="coding_state_transition_target_v33b",
        selected_groups=[mutation],
        target=target,
        releases=[],
    )

    assert guarded is None
    assert releases == ["source-1"]
    assert decision == {
        "target_vetoed": True,
        "target_veto_reasons": ["repository_mutation_command"],
    }


def test_v34_critical_event_vetoes_without_phase_cooldown() -> None:
    mutation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": (
                                "python -c \"open('pkg/a.py', 'w').write('x')\""
                            )
                        },
                    }
                }
            ],
        },
        {"role": "tool", "content": "<returncode>0</returncode>"},
    ]
    prior_mutation = list(mutation)

    guarded, releases, decision = bridge.apply_current_target_veto(
        arm="coding_critical_current_target_v34",
        selected_groups=[prior_mutation, mutation],
        target={"source_id": "source-2"},
        releases=[],
    )

    assert guarded is None
    assert releases == ["source-2"]
    assert decision == {
        "target_vetoed": True,
        "target_veto_reasons": ["repository_mutation_command"],
    }


def test_v35b_vetoes_first_validation_target_only() -> None:
    mutation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": (
                                "python -c \"open('pkg/a.py', 'w').write('x')\""
                            )
                        },
                    }
                }
            ],
        },
        {"role": "tool", "content": "<returncode>0</returncode>"},
    ]
    validation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": "python -m pytest tests/test_a.py"
                        },
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": "<returncode>0</returncode><output>1 passed</output>",
        },
    ]

    guarded, releases, decision = bridge.apply_current_target_veto(
        arm="coding_version_validation_target_v35b",
        selected_groups=[mutation, validation],
        target={"source_id": "source-3"},
        releases=[],
    )
    assert guarded is None
    assert releases == ["source-3"]
    assert decision["target_veto_reasons"] == [
        "first_validation_of_version_before_submit"
    ]

    retained, _, second = bridge.apply_current_target_veto(
        arm="coding_version_validation_target_v35b",
        selected_groups=[mutation, validation, validation],
        target={"source_id": "source-4"},
        releases=[],
    )
    assert retained == {"source_id": "source-4"}
    assert second["target_vetoed"] is False


def test_v37_vetoes_patch_decision_and_recognizes_shell_write() -> None:
    shell_write = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {
                            "command": "cat > /testbed/pkg/a.py <<'EOF'\nx = 1\nEOF"
                        },
                    }
                }
            ],
        },
        {"role": "tool", "content": "<returncode>0</returncode>"},
    ]
    validation = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {"command": "pytest tests/test_a.py"},
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": "<returncode>0</returncode><output>1 passed</output>",
        },
    ]
    diff = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {"command": "git diff"},
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": (
                "<returncode>0</returncode><output>"
                "diff --git a/pkg/a.py b/pkg/a.py</output>"
            ),
        },
    ]

    _, _, after_validation = bridge.apply_current_target_veto(
        arm="coding_patch_lifecycle_target_v37",
        selected_groups=[shell_write, validation],
        target={"source_id": "source-v"},
        releases=[],
    )
    assert after_validation["target_veto_reasons"] == [
        "first_validation_of_version_before_submit"
    ]

    guarded, releases, after_diff = bridge.apply_current_target_veto(
        arm="coding_patch_lifecycle_target_v37",
        selected_groups=[shell_write, validation, diff],
        target={"source_id": "source-d"},
        releases=[],
    )
    assert guarded is None
    assert releases == ["source-d"]
    assert after_diff["target_veto_reasons"] == [
        "patch_diff_before_submission_decision"
    ]
    retained, source_decision = bridge.select_reuse_groups(
        "coding_patch_lifecycle_target_v37",
        [shell_write, validation, diff],
    )
    assert retained == [validation, diff]
    assert source_decision["mode"] == (
        "patch_lifecycle_target_general_source"
    )


def test_query_closes_underlying_sync_stream(monkeypatch) -> None:
    chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content="done",
                    tool_calls=None,
                    reasoning_content=None,
                )
            )
        ]
    )
    stream = _LiteLLMStream([chunk])
    underlying = stream.completion_stream
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(tool_calls=None, content=None)
            )
        ]
    )
    monkeypatch.setattr(bridge.litellm, "completion", lambda **_: stream)
    monkeypatch.setattr(
        bridge.litellm,
        "stream_chunk_builder",
        lambda *_args, **_kwargs: response,
    )

    assert _bare_model()._query([{"role": "user", "content": "x"}]) is response
    assert underlying.close_calls == 1
    assert stream.completion_stream is None


def test_query_closes_underlying_sync_stream_on_iteration_error(
    monkeypatch,
) -> None:
    class BrokenStream(_LiteLLMStream):
        def __iter__(self):
            raise RuntimeError("stream failed")

    stream = BrokenStream([])
    underlying = stream.completion_stream
    monkeypatch.setattr(bridge.litellm, "completion", lambda **_: stream)

    with pytest.raises(RuntimeError, match="stream failed"):
        _bare_model()._query([{"role": "user", "content": "x"}])

    assert underlying.close_calls == 1
    assert stream.completion_stream is None
