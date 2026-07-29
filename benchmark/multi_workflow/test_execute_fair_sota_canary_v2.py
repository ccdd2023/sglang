from pathlib import Path

import pytest

from benchmark.multi_workflow.execute_fair_sota_canary_v2 import (
    clone_for_retry,
    execute_command,
    execution_status_root,
    expected_output,
    select_commands,
)


def _command(tmp_path, command_id="ok"):
    metrics = tmp_path / f"{command_id}.jsonl"
    return {
        "command_id": command_id,
        "comparison_layer": "native",
        "method": "fake",
        "mode": "dense",
        "workdir": str(tmp_path),
        "argv": [
            "/bin/sh",
            "-c",
            f"touch '{metrics}'",
            "--metrics",
            str(metrics),
        ],
        "env": {},
    }


def test_expected_output_reads_metrics_argument(tmp_path):
    command = _command(tmp_path)
    assert expected_output(command) == tmp_path / "ok.jsonl"


def test_execute_command_records_status_and_refuses_append(tmp_path):
    command = _command(tmp_path)
    status_root = tmp_path / "status"

    status = execute_command(command, status_root=status_root)

    assert status["exit_code"] == 0
    assert Path(status["expected_output"]).exists()
    assert (status_root / "ok.status.json").exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        execute_command(command, status_root=status_root)


def test_select_commands_preserves_plan_order():
    plan = {"commands": [{"command_id": "a"}, {"command_id": "b"}]}
    assert [
        command["command_id"]
        for command in select_commands(
            plan, ["b", "a"], run_all=False
        )
    ] == ["b", "a"]
    assert select_commands(plan, [], run_all=True) == plan["commands"]
    with pytest.raises(ValueError, match="unknown"):
        select_commands(plan, ["missing"], run_all=False)


def test_retry_clone_changes_all_outputs_without_mutating_original(tmp_path):
    command = _command(tmp_path)
    command["argv"].extend(
        ["--output-dir", str(tmp_path / "outputs"), "--run-id", "run"]
    )

    retry = clone_for_retry(command, "retry1")

    assert retry["command_id"] == "ok-retry1"
    assert expected_output(retry) == tmp_path / "ok-retry1.jsonl"
    assert str(tmp_path / "outputs-retry1") in retry["argv"]
    assert "run-retry1" in retry["argv"]
    assert command["command_id"] == "ok"
    with pytest.raises(ValueError, match="path-safe"):
        clone_for_retry(command, "bad tag")


def test_status_root_separates_canary_and_static_logs(tmp_path):
    assert execution_status_root(
        tmp_path / "CANARY_COMMAND_PLAN.json"
    ) == tmp_path / "canary/execution"
    assert execution_status_root(
        tmp_path / "STATIC_COMMAND_PLAN.json"
    ) == tmp_path / "static/execution"
