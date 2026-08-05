from benchmark.multi_workflow.motivate_v54_dependency_drift_hybrid import (
    _pair_accuracy,
)


def test_pair_accuracy_compares_score_and_harm_order():
    rows = [
        {
            "candidates": [
                {"score": 2.0, "causal_splice_logit_js": 3.0},
                {"score": 1.0, "causal_splice_logit_js": 1.0},
            ]
        },
        {
            "candidates": [
                {"score": 2.0, "causal_splice_logit_js": 1.0},
                {"score": 1.0, "causal_splice_logit_js": 3.0},
            ]
        },
    ]
    assert _pair_accuracy(rows, "score") == 0.5
