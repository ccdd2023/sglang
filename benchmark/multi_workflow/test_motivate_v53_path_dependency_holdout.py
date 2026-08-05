from benchmark.multi_workflow.motivate_v53_path_dependency_holdout import (
    _candidate_pair_key,
)


def test_candidate_pair_key_is_order_independent():
    left = {
        "candidates": [
            {"segment_token_hash": "b"},
            {"segment_token_hash": "a"},
        ]
    }
    right = {
        "candidates": [
            {"segment_token_hash": "a"},
            {"segment_token_hash": "b"},
        ]
    }
    assert _candidate_pair_key(left) == _candidate_pair_key(right)
