from benchmark.multi_workflow.summarize_v44_schema_compat import (
    EXPECTED_EVENT,
    EXPECTED_MISSING_ROWS,
    EXPECTED_NULL_TTFT_ARMS,
    EXPECTED_NULL_TTFT_TASK,
    EXPECTED_RELATIVE_LEDGER,
)


def test_v44_summary_schema_repair_is_narrowly_frozen() -> None:
    assert EXPECTED_RELATIVE_LEDGER.as_posix() == (
        "tasks/sphinx-doc__sphinx-11445/"
        "coding_grounded_observation_island_v40/CLIENT_LEDGER.jsonl"
    )
    assert EXPECTED_EVENT == "pending_source_not_reusable"
    assert EXPECTED_MISSING_ROWS == 1
    assert EXPECTED_NULL_TTFT_TASK == "astropy__astropy-7671"
    assert EXPECTED_NULL_TTFT_ARMS == 3
