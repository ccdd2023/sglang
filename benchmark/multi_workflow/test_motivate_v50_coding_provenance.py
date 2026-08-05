import math

from benchmark.multi_workflow.motivate_v50_coding_provenance import (
    _balanced_select,
    _binomial_lower_tail_probability,
    _equal_position_ratio,
    _turn_groups,
)


def test_turn_groups_start_at_assistant_boundaries():
    messages = [
        {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "x"},
        {"role": "assistant", "content": "b"},
        {"role": "tool", "content": "y"},
    ]
    assert _turn_groups(messages) == [messages[:2], messages[2:]]


def test_balanced_select_takes_one_per_task_first():
    rows = [
        {"case_id": "a1", "instance_id": "a"},
        {"case_id": "a2", "instance_id": "a"},
        {"case_id": "b1", "instance_id": "b"},
        {"case_id": "b2", "instance_id": "b"},
    ]
    selected = _balanced_select(rows, 3)
    assert len({row["instance_id"] for row in selected[:2]}) == 2
    assert len({row["case_id"] for row in selected}) == 3


def test_equal_position_ratio_recovers_paired_intercept():
    rows = []
    for position_delta in (-0.1, 0.0, 0.1):
        decision = 2.0
        tool = decision * 0.5 * math.exp(position_delta)
        rows.append(
            {
                "tool": {"position_fraction": position_delta, "metric": tool},
                "decision": {"position_fraction": 0.0, "metric": decision},
            }
        )
    value = _equal_position_ratio(rows, "metric", 0.0)
    assert math.isclose(value["equal_position_geometric_ratio"], 0.5)


def test_binomial_probability_is_one_sided_tail():
    assert math.isclose(_binomial_lower_tail_probability(2, 2), 0.25)
