#!/usr/bin/env python3
"""Run mini-SWE-agent SWE-bench with tracked-diff capture at call limit.

mini-SWE-agent 2.3.0 records an empty submission when ``step_limit`` is
reached, even if the agent has already edited tracked files.  This launcher
keeps the stock batch runner and changes only that terminal bookkeeping:
before propagating ``LimitsExceeded``, it captures ``git diff --binary
--no-ext-diff`` from the still-live task container and stores it as the
submission.  It performs no additional model request.
"""

from __future__ import annotations

from typing import Any

from minisweagent.exceptions import LimitsExceeded
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


class LimitPatchCaptureAgent(ProgressTrackingAgent):
    def query(self) -> dict:
        try:
            return super().query()
        except LimitsExceeded as error:
            submission = capture_tracked_patch(self)
            for message in error.messages:
                extra = message.setdefault("extra", {})
                extra["submission"] = submission
                extra["terminal_patch_capture"] = bool(submission)
            raise


swebench.ProgressTrackingAgent = LimitPatchCaptureAgent


if __name__ == "__main__":
    swebench.app()
