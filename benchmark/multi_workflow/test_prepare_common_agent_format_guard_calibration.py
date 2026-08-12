import json
from pathlib import Path

from benchmark.multi_workflow.prepare_common_agent_format_guard_calibration import prepare


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_prepare_freezes_one_high_loop_task_per_repo(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dense = source / "runs/formal/cacheblend_dense/all"
    dataset_rows = []
    for index in range(24):
        repo = f"repo{index % 6}"
        instance_id = f"{repo}__task-{index}"
        dataset_rows.append({"instance_id": instance_id, "problem_statement": str(index)})
        high = index < 6
        messages = [
            {
                "role": "user",
                "extra": {"interrupt_type": "FormatError"},
                "content": "bad",
            }
            for _ in range(28 if high else 0)
        ] + [{"role": "tool", "content": "ok"} for _ in range(4)]
        dump(
            dense / instance_id / f"{instance_id}.traj.json",
            {
                "instance_id": instance_id,
                "messages": messages,
                "info": {
                    "model_stats": {"api_calls": 32},
                    "exit_status": "LimitsExceeded",
                    "submission": "",
                },
            },
        )
    dataset = source / "formal_dataset/test.jsonl"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in dataset_rows), encoding="utf-8"
    )

    value = prepare(tmp_path / "output", source)

    selected = value["selection"]["selected"]
    assert len(selected) == 4
    assert len({row["repo"] for row in selected}) == 4
    assert all(row["format_errors"] == 28 for row in selected)
    assert value["intervention"]["step_limit"] == 32
    assert value["intervention"]["notice_invents_model_command"] is False
