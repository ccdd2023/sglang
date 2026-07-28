from benchmark.multi_workflow.audit_v41_capacity_deadlock import (
    FAILED,
    GENERAL,
    KV_CAPACITY_TOKENS,
    MAX_NEW_TOKENS,
    V40,
)


def test_v41_capacity_audit_constants_are_frozen() -> None:
    assert FAILED == (
        "astropy__astropy-14995",
        "psf__requests-1142",
    )
    assert V40 == "coding_grounded_observation_island_v40"
    assert GENERAL == "general"
    assert MAX_NEW_TOKENS == 2048
    assert KV_CAPACITY_TOKENS == 14482
