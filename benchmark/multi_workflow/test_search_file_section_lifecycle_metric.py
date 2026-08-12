from benchmark.multi_workflow import (
    analyze_search_file_section_exact_lifecycle as analysis,
)
from benchmark.multi_workflow import (
    prepare_search_file_section_lifecycle_metric as prep,
)


def arm(groups, ttft, *, reuse=False, build=0.0):
    targets = [
        {"group_index": group, "warmup": False, "ttft_ms": ttft}
        for group in groups
    ]
    sources = (
        [
            {"group_index": group, "elapsed_ms": build}
            for group in groups
        ]
        if reuse
        else []
    )
    return {"targets": targets, "sources": sources}


def test_lifecycle_counts_one_build_per_persistent_source() -> None:
    plan = {
        "groups": [
            {"group_index": 0, "cases": [{"source_id": "source-a"}]},
            {"group_index": 1, "cases": [{"source_id": "source-a"}]},
            {"group_index": 2, "cases": [{"source_id": "source-b"}]},
        ]
    }
    passes = {
        sequence: {
            "dense": arm(range(3), 100.0),
            "reuse": arm(range(3), 50.0, reuse=True, build=30.0),
        }
        for sequence in ("ab", "ba")
    }

    result = analysis.summarize_lifecycle(plan, passes)

    assert prep.source_usage(plan) == {"source-a": 2, "source-b": 1}
    assert result["distinct_source_build_sum_ms"] == 60.0
    assert result["cache_ready_speedup_ratio_of_sums"] == 2.0
    assert result["observed_lifecycle_n1_speedup"] == 300.0 / 210.0
