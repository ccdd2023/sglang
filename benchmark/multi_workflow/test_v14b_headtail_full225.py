from benchmark.multi_workflow.run_v14b_headtail_full225 import manifest_rows


def test_manifest_keeps_sixteen_tokens_dense_at_both_boundaries():
    case = {
        "case_id": "x",
        "original_case_id": "x",
        "segment_tokens": 40,
        "source_input_ids": [1, *range(10, 50), 2],
        "source_start": 1,
        "target_input_ids": [3, 4, *range(10, 50), 5],
        "target_start": 2,
    }
    row = manifest_rows([case])[0]
    assert row["source_start"] == 17
    assert row["target_start"] == 18
    assert row["length"] == 8
