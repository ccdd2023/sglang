from benchmark.multi_workflow.run_v12_full225_accuracy import (
    _manifest_rows,
    _paired_delta,
)


def test_manifest_rows_enable_v12_boundary_bypass():
    cases = [
        {
            "case_id": "v12:test",
            "segment_tokens": 3,
            "source_input_ids": [1, 2, 10, 11, 12, 3],
            "source_start": 2,
            "target_input_ids": [1, 4, 5, 10, 11, 12, 6],
            "target_start": 3,
        }
    ]
    row = _manifest_rows(cases, "coding_repo_boundary_v12")[0]
    assert row["allow_target_prefix_bypass"] is True
    assert row["length"] == 3
    assert row["target_uses"] == 1


def test_paired_delta_uses_v12_oriented_sign():
    value = _paired_delta(
        {"a": True, "b": False, "c": False, "d": True},
        {"a": True, "b": True, "c": False, "d": False},
        7,
    )
    assert value["treatment_minus_control_pp"] == 0.0
    assert value["transitions"] == {
        "both_fail": 1,
        "both_pass": 1,
        "control_only": 1,
        "treatment_only": 1,
    }
