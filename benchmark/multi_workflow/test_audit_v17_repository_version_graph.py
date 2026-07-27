from __future__ import annotations

from benchmark.multi_workflow.audit_v17_repository_version_graph import (
    command_paths,
    complete_groups,
)


def test_command_paths_extracts_patch_and_testbed_paths() -> None:
    command = """
cat /testbed/pkg/core.py
apply_patch <<'PATCH'
*** Update File: pkg/utils.py
PATCH
"""
    assert command_paths(command) == {"pkg/core.py", "pkg/utils.py"}


def test_complete_groups_ignores_unpaired_final_answer() -> None:
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
        {"role": "assistant", "tool_calls": []},
        {"role": "tool", "content": ""},
        {"role": "assistant", "content": "done"},
    ]
    groups = complete_groups(messages)
    assert len(groups) == 1
    assert [row["role"] for row in groups[0]] == ["assistant", "tool"]
