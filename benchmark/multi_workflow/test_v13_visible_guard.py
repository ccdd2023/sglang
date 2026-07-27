from benchmark.multi_workflow.analyze_v13_visible_guard import visible_check


CASE = {
    "metadata": {"official_entry_point": "inc"},
    "segments": [
        {
            "reusable": True,
            "text": "def inc(x):\n    \"\"\"\n    >>> inc(1)\n    2\n    \"\"\"",
        }
    ],
}


def test_visible_check_requires_interface_and_examples():
    assert visible_check(CASE, "def inc(x):\n    return x + 1")["passed"]
    wrong = visible_check(CASE, "def inc(x):\n    return x - 1")
    assert not wrong["passed"]
    assert wrong["doctest_failed"] == 1
    assert not visible_check(CASE, "def increase(x):\n    return x + 1")["passed"]
