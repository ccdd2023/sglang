from __future__ import annotations

import inspect

from sglang.srt.mem_cache.coding_aware import policy
from sglang.srt.mem_cache.coding_aware.policy import (
    CodingRisk,
    CodingSegment,
    build_coding_reuse_plan,
)
from sglang.srt.mem_cache.kvcomm.types import (
    KVSegmentHandle,
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
    token_ids_hash,
)


def _handle(tokens, start=0):
    key = KVSegmentKey(
        content_hash=f"content-{tokens}",
        token_hash=token_ids_hash(tokens),
        token_count=len(tokens),
        model_id="test",
        cache_dtype="bf16",
        kind=SegmentKind.MIDDLE,
    )
    return KVSegmentHandle(
        key=key,
        generation=1,
        residency=ResidencyTier.DEVICE,
        source_start=start,
        token_ids=tuple(tokens),
        backend_ref="kv",
    )


def test_critical_segment_is_fully_dense():
    tokens = tuple(range(10))
    plan = build_coding_reuse_plan(
        target_token_ids=tokens,
        segments=(
            CodingSegment(
                slot_id="target.py:f",
                target_start=0,
                token_ids=tokens,
                risk=CodingRisk.CRITICAL,
                source=_handle(tokens),
            ),
        ),
    )
    assert not plan.copied_spans
    assert [(item.target_start, item.length) for item in plan.dense_ranges] == [
        (0, 10)
    ]


def test_stable_segment_recomputes_head_and_copies_body():
    tokens = tuple(range(100))
    plan = build_coding_reuse_plan(
        target_token_ids=tokens,
        segments=(
            CodingSegment(
                slot_id="distractor.py:f",
                target_start=0,
                token_ids=tokens,
                risk=CodingRisk.STABLE,
                source=_handle(tokens),
                head_tokens=20,
            ),
        ),
    )
    assert [(item.target_start, item.length) for item in plan.dense_ranges] == [
        (0, 20)
    ]
    span = plan.copied_spans[0]
    assert (span.source_offset, span.target_start, span.length) == (20, 20, 80)


def test_position_move_computes_rope_delta_from_source_and_target():
    prefix = (99, 98)
    tokens = tuple(range(6))
    plan = build_coding_reuse_plan(
        target_token_ids=prefix + tokens,
        segments=(
            CodingSegment(
                slot_id="moved.py:f",
                target_start=2,
                token_ids=tokens,
                risk=CodingRisk.STABLE,
                source=_handle(tokens, start=20),
            ),
        ),
    )
    assert plan.copied_spans[0].rope_delta == -18
    assert plan.dense_ranges[0].reason == "outside_coding_segments"


def test_missing_or_token_mismatched_source_is_dense():
    target = tuple(range(8))
    plan = build_coding_reuse_plan(
        target_token_ids=target,
        segments=(
            CodingSegment(
                slot_id="missing",
                target_start=0,
                token_ids=target[:4],
                risk=CodingRisk.STABLE,
                source=None,
            ),
            CodingSegment(
                slot_id="mismatch",
                target_start=4,
                token_ids=target[4:],
                risk=CodingRisk.STABLE,
                source=_handle((20, 21, 22, 23)),
            ),
        ),
    )
    assert not plan.copied_spans
    assert {item.reason for item in plan.dense_ranges} == {
        "missing_source",
        "source_token_mismatch",
    }


def test_policy_module_has_no_prefetch_or_scheduler_dependency():
    source = inspect.getsource(policy)
    assert "managers.scheduler" not in source
    assert "ensure_resident" not in source
