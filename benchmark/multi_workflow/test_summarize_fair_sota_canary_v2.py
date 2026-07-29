import json

from benchmark.multi_workflow.fair_sota_comparison_v2 import (
    canonical_sha256,
)
from benchmark.multi_workflow.summarize_fair_sota_canary_v2 import (
    _v40_token_records,
)


def test_v40_records_use_common_token_hash(tmp_path):
    (tmp_path / "CASES.json").write_text(
        json.dumps(
            {
                "cases": [
                    {"case_id": "a", "target_input_ids": [1, 2, 3]},
                    {"case_id": "b", "target_input_ids": [4, 5]},
                ]
            }
        ),
        encoding="utf-8",
    )

    records = _v40_token_records(tmp_path)

    assert records[0]["token_ids_sha256"] == canonical_sha256([1, 2, 3])
    assert records[1]["case_id"] == "b"
