from benchmark.multi_workflow import analyze_search_file_section_effective_speed as audit


def test_actual_materialization_counts_only_incremental_snapshot_cost() -> None:
    plan = {
        "groups": [
            {"cases": [{"source_id": "source-a"}]},
            {"cases": [{"source_id": "source-a"}]},
            {"cases": [{"source_id": "source-b"}]},
        ]
    }
    server = [
        {"event": "source_materialized", "source_id": "source-a", "materialize_ms": 1.0},
        {"event": "source_materialized", "source_id": "source-b", "materialize_ms": 2.0},
        {"event": "source_materialized", "source_id": "unused", "materialize_ms": 9.0},
    ]
    exact = [
        {"median_dense_ttft_ms": 100.0, "median_reuse_ttft_ms": 80.0},
        {"median_dense_ttft_ms": 100.0, "median_reuse_ttft_ms": 80.0},
    ]

    result = audit.actual_materialization(plan, server, exact)

    assert result["incremental_materialization_ms"] == 3.0
    assert result["source_prompt_replay_excluded"] is True
    assert result["observed_online_lifecycle_speedup"] == 200.0 / 163.0


def test_partition_summary_reports_ratio_and_win_coverage() -> None:
    result = audit.partition_summary(
        [
            {
                "median_dense_ttft_ms": 100.0,
                "median_reuse_ttft_ms": 50.0,
                "cache_ready_speedup": 2.0,
                "reusable_tokens": 400,
            },
            {
                "median_dense_ttft_ms": 200.0,
                "median_reuse_ttft_ms": 100.0,
                "cache_ready_speedup": 2.0,
                "reusable_tokens": 600,
            },
        ]
    )

    assert result["targets"] == 2
    assert result["cache_ready_speedup_ratio_of_sums"] == 2.0
    assert result["targets_cache_ready_faster"] == 2
    assert result["reusable_tokens_median"] == 500
