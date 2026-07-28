from benchmark.multi_workflow.audit_v39_v38_equivalence import (
    _common_prefix,
)


def test_common_prefix() -> None:
    assert _common_prefix([], []) == 0
    assert _common_prefix([1, 2], [1, 3]) == 1
    assert _common_prefix([1, 2], [1, 2, 3]) == 2
