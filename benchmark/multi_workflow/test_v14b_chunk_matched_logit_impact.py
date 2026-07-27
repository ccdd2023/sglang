from benchmark.multi_workflow.probe_v14b_chunk_matched_logit_impact import (
    CANDIDATES,
)


def test_candidates_exclude_controls():
    assert "full_copy" not in CANDIDATES
    assert "dense_replay" not in CANDIDATES
    assert "repair_middle12" in CANDIDATES
