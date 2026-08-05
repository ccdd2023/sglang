import math

from benchmark.multi_workflow.motivate_v52_path_dependency import (
    _adjusted_ratio,
)


def test_adjusted_ratio_recovers_equal_position_intercept():
    rows = []
    for delta in (-0.2, -0.1, 0.1, 0.2):
        rows.append(
            {
                "candidates": [
                    {
                        "candidate_id": "path_relevant",
                        "metric": 2 * math.exp(delta),
                        "position_fraction": delta,
                        "prefix_shift_tokens": 0,
                    },
                    {
                        "candidate_id": "path_disjoint",
                        "metric": 1.0,
                        "position_fraction": 0.0,
                        "prefix_shift_tokens": 0,
                    },
                ]
            }
        )
    value = _adjusted_ratio(rows, "metric")
    assert math.isclose(
        value["position_adjusted_geometric_ratio"], 2.0, rel_tol=1e-7
    )
