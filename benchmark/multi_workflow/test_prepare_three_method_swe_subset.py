import json

from benchmark.multi_workflow.prepare_three_method_swe_subset import prepare


def test_prepare_preserves_registered_order(tmp_path):
    registration = tmp_path / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "datasets": {
                    "swebench_verified_mechanism": {
                        "tasks": [
                            {"instance_id": "org__two"},
                            {"instance_id": "org__one"},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    population = tmp_path / "population.json"
    population.write_text(
        json.dumps(
            [
                {"instance_id": "org__one", "problem_statement": "one"},
                {"instance_id": "org__two", "problem_statement": "two"},
            ]
        ),
        encoding="utf-8",
    )

    manifest = prepare(registration, population, tmp_path / "out")

    assert manifest["instances"] == ["org__two", "org__one"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "out/minisweagent_dataset/test.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["instance_id"] for row in rows] == [
        "org__two",
        "org__one",
    ]
