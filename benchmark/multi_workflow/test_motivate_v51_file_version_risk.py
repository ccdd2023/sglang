import math

from benchmark.multi_workflow.motivate_v51_file_version_risk import (
    _adjusted_ratio,
    _match_score,
    _sign_probability,
)


def test_match_score_is_zero_for_equal_features():
    row = {
        "position_fraction": 0.5,
        "prefix_shift_tokens": -100,
        "target_input_ids": [1, 2, 3],
    }
    assert _match_score(row, row) == 0


def test_adjusted_ratio_recovers_intercept():
    rows = []
    for delta in (-0.1, 0.0, 0.1, 0.2):
        rows.append(
            {
                "treatment": {
                    "metric": 2 * math.exp(delta),
                    "position_fraction": delta,
                    "prefix_shift_tokens": 0,
                    "target_tokens": 1000,
                },
                "control": {
                    "metric": 1.0,
                    "position_fraction": 0.0,
                    "prefix_shift_tokens": 0,
                    "target_tokens": 1000,
                },
            }
        )
    value = _adjusted_ratio(rows, "metric")
    assert math.isclose(
        value["covariate_adjusted_geometric_ratio"], 2.0, rel_tol=1e-7
    )


def test_sign_probability_is_one_sided_tail():
    assert math.isclose(_sign_probability(2, 2), 0.25)
