import json

from benchmark.multi_workflow.register_three_method_coding_benchmark import (
    METHODS,
    opportunity_rows,
    registration,
    select_mechanism_cohort,
)


def _motivation():
    return {
        "cohorts": {
            "a": [
                {
                    "instance_id": "org1__repo-1",
                    "eligible_target_requests": 10,
                    "requests_with_source": 5,
                    "selected_tokens": [100] * 5,
                    "version_invalidated_observations": 2,
                },
                {
                    "instance_id": "org1__repo-2",
                    "eligible_target_requests": 10,
                    "requests_with_source": 5,
                    "selected_tokens": [80] * 5,
                    "version_invalidated_observations": 1,
                },
                {
                    "instance_id": "org1__repo-3",
                    "eligible_target_requests": 10,
                    "requests_with_source": 5,
                    "selected_tokens": [70] * 5,
                    "version_invalidated_observations": 1,
                },
                {
                    "instance_id": "org2__repo-4",
                    "eligible_target_requests": 10,
                    "requests_with_source": 4,
                    "selected_tokens": [60] * 4,
                    "version_invalidated_observations": 0,
                },
            ]
        }
    }


def test_mechanism_selection_uses_opportunity_and_caps_repositories():
    selected = select_mechanism_cohort(
        opportunity_rows(_motivation()), size=3, per_repo_cap=2
    )

    assert [row["instance_id"] for row in selected] == [
        "org1__repo-1",
        "org1__repo-2",
        "org2__repo-4",
    ]
    assert selected[0]["expected_copied_tokens_per_target"] == 50


def test_registration_contains_only_requested_competitors(tmp_path):
    motivation = tmp_path / "motivation.json"
    motivation.write_text(json.dumps(_motivation()), encoding="utf-8")
    repobench = tmp_path / "repobench-p.jsonl"
    repobench.write_text(
        "\n".join(json.dumps({"_id": f"case-{i}"}) for i in range(4)) + "\n",
        encoding="utf-8",
    )

    value = registration(
        motivation_path=motivation,
        repobench_path=repobench,
        mechanism_size=2,
        control_size=3,
    )

    assert value["scope"]["competing_methods"] == list(METHODS)
    assert value["scope"]["prefetch"] is False
    assert (
        value["datasets"]["swebench_verified_mechanism"][
            "selection_uses_accuracy_or_method_outputs"
        ]
        is False
    )
    assert len(value["datasets"]["repobench_p_control"]["task_ids"]) == 3
    assert value["protected"]["old_preregistration_gates_modified"] is False
