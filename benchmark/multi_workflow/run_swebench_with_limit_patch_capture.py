#!/usr/bin/env python3
"""Run mini-SWE-agent SWE-bench with terminal tracked-diff capture.

mini-SWE-agent 2.3.0 records an empty submission when ``step_limit`` is
reached, even if the agent has already edited tracked files.  This launcher
keeps the stock batch runner and changes only that terminal bookkeeping:
before propagating ``LimitsExceeded``, it captures ``git diff --binary
--no-ext-diff`` from the still-live task container and stores it as the
submission. Some models also print only the completion sentinel even after
creating and inspecting ``patch.txt``. An empty ``Submitted`` result receives
the same tracked-diff fallback. Neither path performs another model request.
"""

from __future__ import annotations

from typing import Any

from minisweagent.exceptions import LimitsExceeded, Submitted
from minisweagent.run.benchmarks import swebench
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent


def capture_tracked_patch(agent: Any) -> str:
    output = agent.env.execute(
        {"command": "git diff --binary --no-ext-diff"},
        timeout=120,
    )
    if output.get("returncode") != 0:
        return ""
    return str(output.get("output") or "")


def fill_empty_submission(agent: Any, error: Any) -> bool:
    """Fill only empty terminal submissions from the live tracked diff."""

    empty_messages = [
        message
        for message in error.messages
        if not message.get("extra", {}).get("submission")
    ]
    if not empty_messages:
        return False
    submission = capture_tracked_patch(agent)
    for message in empty_messages:
        extra = message.setdefault("extra", {})
        extra["submission"] = submission
        extra["terminal_patch_capture"] = bool(submission)
        extra["terminal_patch_capture_reason"] = type(error).__name__
        if submission:
            message["content"] = submission
    return bool(submission)


class LimitPatchCaptureAgent(ProgressTrackingAgent):
    def query(self) -> dict:
        try:
            return super().query()
        except LimitsExceeded as error:
            fill_empty_submission(self, error)
            raise

    def execute_actions(self, message: dict) -> list[dict]:
        try:
            return super().execute_actions(message)
        except Submitted as error:
            fill_empty_submission(self, error)
            raise


swebench.ProgressTrackingAgent = LimitPatchCaptureAgent


if __name__ == "__main__":
    swebench.app()
