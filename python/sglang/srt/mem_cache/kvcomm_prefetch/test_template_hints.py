from __future__ import annotations

from sglang.srt.mem_cache.kvcomm.types import (
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
    token_ids_hash,
)
from sglang.srt.mem_cache.kvcomm_prefetch.template_hints import (
    TemplatePrefetchIsland,
    compile_template_prefetch_hints,
)


def _key(name: str, token: int = 1) -> KVSegmentKey:
    tokens = (token,)
    return KVSegmentKey(
        content_hash=name,
        token_hash=token_ids_hash(tokens),
        token_count=1,
        model_id="test",
        cache_dtype="bf16",
        kind=SegmentKind.MIDDLE,
    )


def _island(
    name: str,
    *,
    remaining_uses: int = 2,
    next_group_index: int | None = 0,
    eligible: bool = True,
    token_ids_match: bool = True,
    version_valid: bool = True,
    delta_nonzero: bool = True,
    single_file: bool = True,
    token: int = 1,
) -> TemplatePrefetchIsland:
    return TemplatePrefetchIsland(
        source_id=name,
        key=_key(name, token),
        remaining_uses=remaining_uses,
        next_group_index=next_group_index,
        eligible=eligible,
        token_ids_match=token_ids_match,
        version_valid=version_valid,
        delta_nonzero=delta_nonzero,
        single_file_repository_code=single_file,
    )


def test_whitelist_skips_ineligible_and_zero_shift_and_stale():
    plan = compile_template_prefetch_hints(
        (
            _island("ok", remaining_uses=3, token=1),
            _island("issue_text", eligible=False, token=2),
            _island("stale", version_valid=False, token=3),
            _island("zero", delta_nonzero=False, token=4),
            _island("mismatch", token_ids_match=False, token=5),
            _island("spent", remaining_uses=0, token=6),
            _island("multi", single_file=False, token=7),
        )
    )
    assert [hint.key.content_hash for hint in plan.hints] == ["ok"]
    assert plan.hints[0].priority == 3
    assert plan.hints[0].target_tier == ResidencyTier.DEVICE
    assert "issue_text:not_eligible" in plan.skip_reasons
    assert "stale:version_invalid" in plan.skip_reasons
    assert "zero:zero_shift" in plan.skip_reasons
    assert "mismatch:token_ids_mismatch" in plan.skip_reasons
    assert "spent:no_remaining_uses" in plan.skip_reasons
    assert "multi:not_single_file_repository_code" in plan.skip_reasons


def test_priority_is_remaining_uses_and_deadline_is_next_group_eta():
    plan = compile_template_prefetch_hints(
        (
            _island("late", remaining_uses=9, next_group_index=8, token=1),
            _island("soon", remaining_uses=2, next_group_index=1, token=2),
        ),
        group_eta_s={1: 1.5, 8: 10.0},
        now_s=0.5,
    )
    assert [hint.key.content_hash for hint in plan.hints] == ["soon", "late"]
    assert plan.hints[0].deadline_s == 1.0
    assert plan.hints[0].priority == 2
    assert plan.hints[1].deadline_s == 9.5
    assert plan.hints[1].priority == 9


def test_same_key_keeps_tighter_deadline_and_does_not_grow_admit_set():
    key = _key("shared", 11)
    first = TemplatePrefetchIsland(
        source_id="a",
        key=key,
        remaining_uses=1,
        next_group_index=4,
    )
    second = TemplatePrefetchIsland(
        source_id="b",
        key=key,
        remaining_uses=5,
        next_group_index=1,
    )
    plan = compile_template_prefetch_hints(
        (first, second),
        group_eta_s={1: 2.0, 4: 8.0},
        now_s=0.0,
    )
    assert len(plan.hints) == 1
    assert plan.hints[0].deadline_s == 2.0
    assert plan.hints[0].priority == 5


def test_compile_does_not_rotate_or_invent_prefix_keys():
    plan = compile_template_prefetch_hints((_island("file", token=3),))
    assert plan.hints[0].key.kind is SegmentKind.MIDDLE
    assert "prefix" not in plan.hints[0].key.content_hash
