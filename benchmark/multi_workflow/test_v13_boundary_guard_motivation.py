from benchmark.multi_workflow.run_v13_boundary_guard_motivation import (
    guarded_span,
)


CASE = {
    "case_id": "x",
    "segment_tokens": 64,
    "source_start": 10,
    "target_start": 20,
}


def test_guarded_spans_keep_requested_boundaries_dense():
    assert guarded_span(CASE, "head16") == {
        "source_start": 26,
        "target_start": 36,
        "length": 48,
    }
    assert guarded_span(CASE, "tail16") == {
        "source_start": 10,
        "target_start": 20,
        "length": 48,
    }
    assert guarded_span(CASE, "head16_tail16") == {
        "source_start": 26,
        "target_start": 36,
        "length": 32,
    }
