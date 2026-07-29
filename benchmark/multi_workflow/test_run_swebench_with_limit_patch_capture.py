from benchmark.multi_workflow.run_swebench_with_limit_patch_capture import (
    capture_tracked_patch,
)


class _Environment:
    def __init__(self, value):
        self.value = value

    def execute(self, action, timeout):
        assert action == {"command": "git diff --binary --no-ext-diff"}
        assert timeout == 120
        return self.value


class _Agent:
    def __init__(self, value):
        self.env = _Environment(value)


def test_capture_tracked_patch_returns_successful_diff():
    assert capture_tracked_patch(
        _Agent({"returncode": 0, "output": "diff --git a/x b/x\n"})
    ).startswith("diff --git")


def test_capture_tracked_patch_rejects_failed_diff():
    assert capture_tracked_patch(
        _Agent({"returncode": 1, "output": "failure"})
    ) == ""
