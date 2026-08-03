from types import SimpleNamespace

import pytest
from minisweagent.exceptions import Submitted
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent

from benchmark.multi_workflow import run_swebench_with_limit_patch_capture as capture


class _Environment:
    def __init__(self, output: str, returncode: int = 0) -> None:
        self.output = output
        self.returncode = returncode
        self.commands: list[tuple[dict, int]] = []

    def execute(self, action: dict, *, timeout: int) -> dict:
        self.commands.append((action, timeout))
        return {"returncode": self.returncode, "output": self.output}


def test_capture_tracked_patch_returns_successful_diff() -> None:
    environment = _Environment("diff --git a/x b/x\n")
    agent = SimpleNamespace(env=environment)

    assert capture.capture_tracked_patch(agent).startswith("diff --git")
    assert environment.commands == [
        ({"command": "git diff --binary --no-ext-diff"}, 120)
    ]


def test_capture_tracked_patch_rejects_failed_diff() -> None:
    environment = _Environment("failure", returncode=1)
    agent = SimpleNamespace(env=environment)

    assert capture.capture_tracked_patch(agent) == ""


def test_fill_empty_submitted_result_from_live_tracked_diff() -> None:
    environment = _Environment("diff --git a/module.py b/module.py\n")
    agent = SimpleNamespace(env=environment)
    error = Submitted(
        {
            "role": "exit",
            "content": "",
            "extra": {"exit_status": "Submitted", "submission": ""},
        }
    )

    assert capture.fill_empty_submission(agent, error) is True
    message = error.messages[0]
    assert message["extra"]["submission"].startswith("diff --git")
    assert message["extra"]["terminal_patch_capture"] is True
    assert message["extra"]["terminal_patch_capture_reason"] == "Submitted"
    assert message["content"] == message["extra"]["submission"]
    assert environment.commands == [
        ({"command": "git diff --binary --no-ext-diff"}, 120)
    ]


def test_nonempty_submission_is_never_replaced() -> None:
    environment = _Environment("unexpected diff")
    agent = SimpleNamespace(env=environment)
    error = Submitted(
        {
            "role": "exit",
            "content": "model patch",
            "extra": {
                "exit_status": "Submitted",
                "submission": "model patch",
            },
        }
    )

    assert capture.fill_empty_submission(agent, error) is False
    assert error.messages[0]["extra"]["submission"] == "model patch"
    assert environment.commands == []


def test_execute_actions_captures_empty_submitted_result(monkeypatch) -> None:
    def submit_without_patch(
        _agent: object, _message: dict
    ) -> list[dict]:
        raise Submitted(
            {
                "role": "exit",
                "content": "",
                "extra": {
                    "exit_status": "Submitted",
                    "submission": "",
                },
            }
        )

    monkeypatch.setattr(
        ProgressTrackingAgent,
        "execute_actions",
        submit_without_patch,
    )
    agent = object.__new__(capture.LimitPatchCaptureAgent)
    agent.env = _Environment("diff --git a/module.py b/module.py\n")

    with pytest.raises(Submitted) as raised:
        agent.execute_actions({})

    extra = raised.value.messages[0]["extra"]
    assert extra["submission"].startswith("diff --git")
    assert extra["terminal_patch_capture_reason"] == "Submitted"
