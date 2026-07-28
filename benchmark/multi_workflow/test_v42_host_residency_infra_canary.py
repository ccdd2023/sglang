from benchmark.multi_workflow.run_v42_host_residency_infra_canary import (
    ARMS,
    GENERAL,
    INSTANCE_ID,
    V40,
)


def test_v42_is_failure_directed_infra_only() -> None:
    assert INSTANCE_ID == "astropy__astropy-14995"
    assert V40 == "coding_grounded_observation_island_v40"
    assert GENERAL == "general"
    assert ARMS == (V40, GENERAL, "dense")
