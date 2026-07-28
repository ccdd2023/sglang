from benchmark.multi_workflow.run_v40a_grounded_observation_canary import (
    ARMS,
    V40,
)


def test_v40a_arm_contract() -> None:
    assert ARMS == (V40, "general", "dense")
    assert V40 == "coding_grounded_observation_island_v40"
