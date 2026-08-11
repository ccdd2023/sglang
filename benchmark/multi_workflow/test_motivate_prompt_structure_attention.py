from benchmark.multi_workflow.motivate_prompt_structure_attention import (
    MAX_QUERY_TOKENS_PER_CATEGORY,
    _preview,
    _query_positions,
)


def test_query_positions_uses_region_tails_and_global_cap():
    regions = [
        {"category": "assistant_action", "start": 0, "end": 40},
        {"category": "assistant_action", "start": 40, "end": 80},
        {"category": "user_task", "start": 80, "end": 100},
        {"category": "assistant_action", "start": 100, "end": 140},
        {"category": "assistant_action", "start": 140, "end": 180},
        {"category": "assistant_action", "start": 180, "end": 220},
    ]
    positions = _query_positions(regions, "assistant_action")
    assert len(positions) == MAX_QUERY_TOKENS_PER_CATEGORY
    assert positions[0] == 64
    assert positions[-1] == 219
    assert all(position not in positions for position in range(80, 100))


def test_preview_normalizes_testbed_path_and_whitespace():
    assert _preview("  cat   /testbed/src/parser.py\n") == "cat src/parser.py"
