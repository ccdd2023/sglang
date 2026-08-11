from benchmark.multi_workflow.motivate_dense_reuse_global_block_attention import (
    _distribution_metrics,
    _finalize_blocks,
    _map_source_blocks,
    _max_distance_pair,
    _split_blocks_at_span,
    _weighted_distribution,
)


def test_max_distance_pair_uses_both_frozen_axes() -> None:
    rows = [
        {
            "case_id": "middle",
            "normalized_log_prompt": 0.5,
            "normalized_log_copy": 0.5,
        },
        {
            "case_id": "low",
            "normalized_log_prompt": 0.0,
            "normalized_log_copy": 0.0,
        },
        {
            "case_id": "high",
            "normalized_log_prompt": 1.0,
            "normalized_log_copy": 1.0,
        },
    ]
    assert _max_distance_pair(rows) == (1, 2)


def test_split_blocks_marks_exact_copied_partition() -> None:
    blocks = [
        {
            "start": 0,
            "end": 10,
            "tokens": 10,
            "category": "read_observation_path_relevant",
            "label": "read",
            "paths": ["a.py"],
            "copied": False,
        },
        {
            "start": 10,
            "end": 15,
            "tokens": 5,
            "category": "generation_marker",
            "label": "next",
            "paths": [],
            "copied": False,
        },
    ]
    split = _split_blocks_at_span(blocks, 3, 8)
    assert [(row["start"], row["end"]) for row in split] == [
        (0, 3),
        (3, 8),
        (8, 10),
        (10, 15),
    ]
    assert sum(row["tokens"] for row in split if row["copied"]) == 5
    assert next(row for row in split if row["copied"])["category"] == (
        "copied_observation_island"
    )


def test_source_mapping_has_explicit_source_only_bucket() -> None:
    ids = list(range(12))
    target = _finalize_blocks(
        [
            {
                "start": 0,
                "end": 4,
                "tokens": 4,
                "category": "system_instruction",
                "label": "system",
                "paths": [],
                "copied": False,
            },
            {
                "start": 4,
                "end": 8,
                "tokens": 4,
                "category": "copied_observation_island",
                "label": "copy",
                "paths": [],
                "copied": True,
            },
            {
                "start": 8,
                "end": 12,
                "tokens": 4,
                "category": "generation_marker",
                "label": "next",
                "paths": [],
                "copied": False,
            },
        ],
        ids,
        "t",
    )
    source_ids = [99, 98, 97, 96, *ids[4:8], 95, 94, 93, 92]
    source = _finalize_blocks(
        [
            {
                "start": 0,
                "end": 4,
                "tokens": 4,
                "category": "system_instruction",
                "label": "old system",
                "paths": [],
                "copied": False,
            },
            {
                "start": 4,
                "end": 8,
                "tokens": 4,
                "category": "copied_observation_island",
                "label": "copy",
                "paths": [],
                "copied": True,
            },
            {
                "start": 8,
                "end": 12,
                "tokens": 4,
                "category": "generation_marker",
                "label": "old next",
                "paths": [],
                "copied": False,
            },
        ],
        source_ids,
        "s",
    )
    mapped = _map_source_blocks(source, target)
    assert mapped[0]["mapped_target_block_id"] == "source_only_context"
    assert mapped[1]["mapped_target_block_id"] == "t01"


def test_distribution_metrics_and_weighted_rows() -> None:
    dense = {"r0": {"a": 0.75, "b": 0.25}, "r1": {"a": 0.25, "b": 0.75}}
    blocks = [
        {"block_id": "r0", "tokens": 1},
        {"block_id": "r1", "tokens": 3},
    ]
    weighted = _weighted_distribution(dense, blocks)
    assert weighted == {"a": 0.375, "b": 0.625}
    metrics = _distribution_metrics(weighted, {"a": 0.375, "b": 0.625})
    assert metrics["tv"] == 0.0
    assert metrics["js_nats"] == 0.0
    assert metrics["top_block_agreement"] is True
