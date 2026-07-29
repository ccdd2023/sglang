import pytest

from benchmark.multi_workflow.run_v40_repobench_control import (
    _prediction_line,
    prepare_case,
)


def test_prediction_line_ignores_fences_and_comments():
    assert _prediction_line("\n```python\n# comment\n    return x\n```") == "    return x"


def test_prediction_line_keeps_plain_code():
    assert _prediction_line("    continue\nextra") == "    continue"


def test_prepare_case_rejects_copy_cap_below_minimum():
    with pytest.raises(ValueError, match="at least 128"):
        prepare_case(None, {}, copy_cap=127)
