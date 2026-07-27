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
