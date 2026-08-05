import collections

from benchmark.multi_workflow.motivate_v47_task_conditioned_pool import (
    _select_indices,
    coding_symbol_scores,
)


def _candidates():
    return [
        {"context_index": index, "text": text}
        for index, text in enumerate(
            (
                "def unrelated_socket_timeout(): pass",
                "class Gradient: def apply_shader(self): pass",
                "def draw_colored_triangles(values): pass",
                "def old_helper(): pass",
                "def recent_helper(): pass",
                "def newest_helper(): pass",
            )
        )
    ]


def test_symbol_scores_prefer_cursor_local_code_identifiers():
    candidates = _candidates()
    scores = coding_symbol_scores(
        "if gradient: gradient.apply_shader(); graphics.draw_colored_triangles(",
        [row["text"] for row in candidates],
    )
    assert scores[1] > scores[0]
    assert scores[2] > scores[0]


def test_recency_and_symbol_selectors_are_distinct_and_budget_neutral():
    candidates = _candidates()
    recency, _ = _select_indices(
        "v46_recency_m47", "case", candidates, "ignored"
    )
    symbol, _ = _select_indices(
        "coding_symbol_overlap_m47",
        "case",
        candidates,
        "gradient.apply_shader(); graphics.draw_colored_triangles(",
    )
    assert set(recency) == {3, 4, 5}
    assert {1, 2}.issubset(set(symbol))
    assert len(recency) == len(symbol) == 3


def test_matched_random_is_stable_per_case():
    candidates = _candidates()
    first, _ = _select_indices(
        "matched_random_m47", "frozen-case", candidates, "ignored"
    )
    second, _ = _select_indices(
        "matched_random_m47", "frozen-case", candidates, "ignored"
    )
    assert first == second
    assert len(first) == 3
    assert len(collections.Counter(first)) == 3
