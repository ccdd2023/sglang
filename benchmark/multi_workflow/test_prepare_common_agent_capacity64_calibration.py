import json
from pathlib import Path

from benchmark.multi_workflow.prepare_common_agent_capacity64_calibration import prepare


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_prepare_selects_only_32_call_limit_exits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dense = source / "runs/formal/cacheblend_dense/all"
    dump(
        dense / "OFFICIAL_RESULT.json",
        {
            "returncode": 0,
            "report": {
                "resolved_instances": 0,
                "total_instances": 24,
                "error_instances": 0,
            },
        },
    )
    rows = []
    for index in range(24):
        instance_id = f"repo__repo-{index}"
        rows.append({"instance_id": instance_id, "problem_statement": str(index)})
        limited = index % 2 == 0
        dump(
            dense / instance_id / f"{instance_id}.traj.json",
            {
                "instance_id": instance_id,
                "info": {
                    "model_stats": {"api_calls": 32 if limited else 4},
                    "exit_status": "LimitsExceeded" if limited else "Submitted",
                    "submission": "" if limited else "diff",
                },
            },
        )
    dataset = source / "formal_dataset/test.jsonl"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    output = tmp_path / "capacity"
    value = prepare(output, source)

    assert value["selection"]["selected_count"] == 12
    assert value["intervention"]["old"] == 32
    assert value["intervention"]["new"] == 64
    assert value["gate"]["official_resolved_instances_min"] == 1
    frozen = json.loads((output / "FROZEN_CAPACITY12.json").read_text())
    assert [row["instance_id"] for row in frozen] == sorted(
        f"repo__repo-{index}" for index in range(0, 24, 2)
    )
