from __future__ import annotations

from benchmark.multi_workflow.validate_single_island_action_divergence import (
    common_prefix_tokens,
    levenshtein,
    sequence_metrics,
)


def test_token_sequence_metrics_distinguish_prefix_and_edit_distance() -> None:
    dense = [1, 2, 3, 4]
    splice = [1, 2, 9, 4, 5]
    value = sequence_metrics(dense, splice)
    assert common_prefix_tokens(dense, splice) == 2
    assert levenshtein(dense, splice) == 2
    assert value["common_prefix_tokens"] == 2
    assert value["token_edit_distance"] == 2
    assert value["normalized_token_edit_distance"] == 0.4
    assert not value["exact_match"]


def test_token_sequence_metrics_handle_exact_and_empty() -> None:
    assert sequence_metrics([7, 8], [7, 8])["exact_match"]
    assert sequence_metrics([], [3])["normalized_token_edit_distance"] == 1.0
