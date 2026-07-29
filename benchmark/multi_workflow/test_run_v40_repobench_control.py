from benchmark.multi_workflow.run_v40_repobench_control import (
    _prediction_line,
)


def test_prediction_line_ignores_fences_and_comments():
    assert _prediction_line("\n```python\n# comment\n    return x\n```") == "    return x"


def test_prediction_line_keeps_plain_code():
    assert _prediction_line("    continue\nextra") == "    continue"
