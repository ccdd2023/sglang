import json

from benchmark.multi_workflow import (
    analyze_search_file_section_canary_counterfactual as audit,
)


def trajectory(path, *, command: str, treated: bool, submission: str) -> None:
    value = {
        "info": {"exit_status": "Submitted", "submission": submission},
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": {"command": command},
                        }
                    }
                ],
                "extra": {
                    "reuse_treatment": {
                        "request_index": 4,
                        "input_ids_sha256": "same-prompt",
                        "target_registered": treated,
                        "copied_tokens_planned": 321 if treated else 0,
                    }
                },
            }
        ],
    }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_compare_task_attributes_successful_copy_without_claiming_accuracy(
    tmp_path,
) -> None:
    dense = tmp_path / "dense/task.traj.json"
    search = tmp_path / "search/task.traj.json"
    trajectory(dense, command="sed -n 1,20p a.py", treated=False, submission="patch")
    trajectory(search, command="sed -n 1,20p a.py", treated=True, submission="patch")

    row = audit.compare_task(dense, search)

    assert row["common_inputs_identical"] == 1
    assert row["common_actions_identical"] == 1
    assert row["successful_copy_exposed_tokens"] == 321
    assert row["all_successful_copy_actions_match_dense"] is True
    assert row["final_submission_identical"] is True
