"""Held-out next-island hints: protocol signals only, no future target_uses."""

from __future__ import annotations

from sglang.srt.mem_cache.kvcomm.types import (
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
    token_ids_hash,
)
from sglang.srt.mem_cache.kvcomm_prefetch.template_hints import (
    PREFIX_PRIORITY_FLOOR,
    NextIslandObservation,
    TemplatePrefetchIsland,
    compile_next_island_prefetch_hints,
    compile_template_prefetch_hints,
    protocol_later_roles,
)


def _key(name: str, token: int, kind=SegmentKind.MIDDLE) -> KVSegmentKey:
    tokens = (token,)
    return KVSegmentKey(
        content_hash=name,
        token_hash=token_ids_hash(tokens),
        token_count=1,
        model_id="test",
        cache_dtype="bf16",
        kind=kind,
    )


def test_next_island_hints_are_whitelist_subset_without_future_uses():
    admitted = _key("repo-file-v3", 1)
    ineligible = _key("tool-log", 2)
    last_role = _key("last-role-file", 3)
    observations = (
        NextIslandObservation(
            source_id="planner-read",
            key=admitted,
            later_roles_in_protocol=3,
        ),
        NextIslandObservation(
            source_id="debugger-note",
            key=ineligible,
            later_roles_in_protocol=3,
            eligible=False,
        ),
        NextIslandObservation(
            source_id="stale",
            key=_key("stale-file", 4),
            later_roles_in_protocol=2,
            version_valid=False,
        ),
        NextIslandObservation(
            source_id="zero-shift",
            key=_key("prefix-looking", 5),
            later_roles_in_protocol=2,
            delta_nonzero=False,
        ),
        NextIslandObservation(
            source_id="finalizer",
            key=last_role,
            later_roles_in_protocol=0,
        ),
    )
    # Future uses are hidden: oracle compile would skip remaining_uses=0.
    oracle = compile_template_prefetch_hints(
        (
            TemplatePrefetchIsland(
                source_id="planner-read",
                key=admitted,
                remaining_uses=0,
            ),
        )
    )
    assert oracle.hints == ()

    plan = compile_next_island_prefetch_hints(observations)
    hinted = {hint.key.content_hash for hint in plan.hints}
    whitelist = {admitted.content_hash}
    assert hinted <= whitelist
    assert hinted == whitelist
    assert plan.hints[0].priority == 3
    assert ineligible.content_hash not in hinted
    assert last_role.content_hash not in hinted
    assert "debugger-note:not_eligible" in plan.skip_reasons
    assert "stale:version_invalid" in plan.skip_reasons
    assert "zero-shift:zero_shift" in plan.skip_reasons
    assert "finalizer:no_protocol_reread" in plan.skip_reasons


def test_next_island_does_not_grow_admit_set_on_duplicate_keys():
    key = _key("shared-file", 9)
    plan = compile_next_island_prefetch_hints(
        (
            NextIslandObservation(
                source_id="early",
                key=key,
                later_roles_in_protocol=1,
            ),
            NextIslandObservation(
                source_id="later",
                key=key,
                later_roles_in_protocol=3,
            ),
        )
    )
    assert len(plan.hints) == 1
    assert plan.hints[0].priority == 3


def test_protocol_later_roles_is_not_remaining_uses() -> None:
    assert protocol_later_roles("coding_aware") == 3
    assert protocol_later_roles("coding_natural_code_cost", explicit=0) == 0
    assert protocol_later_roles("general") == 0


def test_later_roles_zero_emits_no_hint() -> None:
    plan = compile_next_island_prefetch_hints(
        (
            NextIslandObservation(
                source_id="reviewer",
                key=_key("last-file", 1),
                later_roles_in_protocol=0,
            ),
        )
    )
    assert plan.hints == ()
    assert "reviewer:no_protocol_reread" in plan.skip_reasons


def test_already_resident_sequential_next_use_emits_no_hint() -> None:
    plan = compile_next_island_prefetch_hints(
        (
            NextIslandObservation(
                source_id="planner-read",
                key=_key("repo-file", 1),
                later_roles_in_protocol=3,
                residency=ResidencyTier.DEVICE,
                sequential_next_use=True,
            ),
        )
    )
    assert plan.hints == ()
    assert "planner-read:no_overlap_window" in plan.skip_reasons


def test_host_later_roles_emits_hint_without_remaining_uses() -> None:
    plan = compile_next_island_prefetch_hints(
        (
            NextIslandObservation(
                source_id="planner-read",
                key=_key("repo-file", 1),
                later_roles_in_protocol=3,
                residency=ResidencyTier.HOST,
                sequential_next_use=False,
            ),
        )
    )
    assert len(plan.hints) == 1
    assert plan.hints[0].priority == 3
    assert not hasattr(plan.hints[0], "rope_delta")
    assert "rope" not in plan.hints[0].__dataclass_fields__


def test_remaining_uses_zero_oracle_is_not_next_island_rule() -> None:
    key = _key("repo-file", 1)
    oracle = compile_template_prefetch_hints(
        (
            TemplatePrefetchIsland(
                source_id="planner-read",
                key=key,
                remaining_uses=0,
            ),
        )
    )
    served = compile_next_island_prefetch_hints(
        (
            NextIslandObservation(
                source_id="planner-read",
                key=key,
                later_roles_in_protocol=3,
                residency=ResidencyTier.HOST,
            ),
        )
    )
    assert oracle.hints == ()
    assert len(served.hints) == 1
    assert served.hints[0].key == key


def test_prefix_and_middle_hints_emit_together_on_host() -> None:
    prefix = _key("prompt-prefix", 1, kind=SegmentKind.PREFIX)
    island = _key("repo-file", 2, kind=SegmentKind.MIDDLE)
    plan = compile_next_island_prefetch_hints(
        (
            NextIslandObservation(
                source_id="planner-read:prefix",
                key=prefix,
                later_roles_in_protocol=3,
                residency=ResidencyTier.HOST,
                sequential_next_use=False,
                delta_nonzero=False,
                single_file_repository_code=False,
                span_kind=SegmentKind.PREFIX,
            ),
            NextIslandObservation(
                source_id="planner-read",
                key=island,
                later_roles_in_protocol=3,
                residency=ResidencyTier.HOST,
                sequential_next_use=False,
                span_kind=SegmentKind.MIDDLE,
            ),
        )
    )
    kinds = {hint.key.kind for hint in plan.hints}
    assert kinds == {SegmentKind.PREFIX, SegmentKind.MIDDLE}
    assert plan.hints[0].key.kind == SegmentKind.PREFIX
    assert plan.hints[0].priority >= PREFIX_PRIORITY_FLOOR
    assert plan.hints[0].priority > plan.hints[1].priority
    assert not any(hasattr(hint, "rope_delta") for hint in plan.hints)


def test_middle_priority_override_uses_template_score() -> None:
    island = _key("repo-file", 2, kind=SegmentKind.MIDDLE)
    plan = compile_next_island_prefetch_hints(
        (
            NextIslandObservation(
                source_id="planner-read",
                key=island,
                later_roles_in_protocol=3,
                residency=ResidencyTier.HOST,
                priority_override=5,
            ),
        )
    )
    assert plan.hints[0].priority == 5


def test_prefix_device_sequential_still_skips() -> None:
    plan = compile_next_island_prefetch_hints(
        (
            NextIslandObservation(
                source_id="planner-read:prefix",
                key=_key("prompt-prefix", 1, kind=SegmentKind.PREFIX),
                later_roles_in_protocol=3,
                residency=ResidencyTier.DEVICE,
                sequential_next_use=True,
                delta_nonzero=False,
                span_kind=SegmentKind.PREFIX,
            ),
        )
    )
    assert plan.hints == ()
    assert "planner-read:prefix:no_overlap_window" in plan.skip_reasons


def test_middle_zero_shift_still_not_treated_as_prefix() -> None:
    plan = compile_next_island_prefetch_hints(
        (
            NextIslandObservation(
                source_id="zero-shift",
                key=_key("prefix-looking", 5),
                later_roles_in_protocol=2,
                delta_nonzero=False,
                span_kind=SegmentKind.MIDDLE,
            ),
        )
    )
    assert plan.hints == ()
    assert "zero-shift:zero_shift" in plan.skip_reasons
