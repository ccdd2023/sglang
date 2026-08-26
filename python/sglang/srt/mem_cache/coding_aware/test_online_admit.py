from __future__ import annotations

import inspect

from sglang.srt.mem_cache.coding_aware import online_admit
from sglang.srt.mem_cache.coding_aware.online_admit import (
    BindAction,
    LeasedIsland,
    SourceObservation,
    admit_source_island,
    bind_leased_islands,
    build_online_reuse_plan,
    locate_unique_span,
    protocol_later_roles,
)
from sglang.srt.mem_cache.kvcomm.types import (
    KVSegmentHandle,
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
    token_ids_hash,
)


def _obs(**overrides) -> SourceObservation:
    tokens = (10, 11, 12, 13)
    values = dict(
        source_id="read:mod.py",
        source_start=4,
        token_ids=tokens,
        content_hash=token_ids_hash(tokens),
        source_prefix_hash="pfx",
        single_file_repository_code=True,
        version_valid=True,
        later_roles_in_protocol=3,
        seq=0,
        policy_label="coding",
    )
    values.update(overrides)
    return SourceObservation(**values)


def _lease(tokens, source_start=4, source_id="src", seq=0, handle=None):
    return LeasedIsland(
        source_id=source_id,
        source_start=source_start,
        token_ids=tuple(tokens),
        content_hash=token_ids_hash(tokens),
        source_prefix_hash="pfx",
        seq=seq,
        handle=handle,
    )


def _handle(tokens, start=4):
    tokens = tuple(tokens)
    key = KVSegmentKey(
        content_hash=token_ids_hash(tokens),
        token_hash=token_ids_hash(tokens),
        token_count=len(tokens),
        model_id="test",
        cache_dtype="bf16",
        kind=SegmentKind.MIDDLE,
        source_prefix_hash="pfx",
        pre_rotate_delta=0,
    )
    return KVSegmentHandle(
        key=key,
        generation=1,
        residency=ResidencyTier.DEVICE,
        source_start=start,
        token_ids=tokens,
        backend_ref="kv",
    )


def test_admit_does_not_consult_a_target():
    source = inspect.getsource(admit_source_island)
    assert "target_start" not in source
    assert "target_hash" not in source
    assert "target_uses" not in source
    assert admit_source_island(_obs()) is None


def test_admit_rejects_protocol_and_version_failures():
    assert admit_source_island(_obs(later_roles_in_protocol=0)) == (
        "no_protocol_reread"
    )
    assert admit_source_island(_obs(version_valid=False)) == "version_invalid"
    assert (
        admit_source_island(_obs(single_file_repository_code=False))
        == "not_single_file_repository_code"
    )
    assert admit_source_island(_obs(source_start=0)) == "not_strictly_middle"


def test_coding_policy_implies_later_roles_without_oracle_uses():
    assert protocol_later_roles("coding_natural_code_cost") == 3
    assert protocol_later_roles("general") == 0
    assert protocol_later_roles("coding", explicit=0) == 0


def test_bind_locates_shifted_span_without_a_planned_t():
    island = (2, 3, 4, 5)
    target = (9, 8, 7, *island, 1)
    binds = bind_leased_islands(target, (_lease(island, source_start=1),))
    assert len(binds) == 1
    bind = binds[0]
    assert bind.action is BindAction.COPY
    assert bind.target_start == 3
    assert bind.rope_delta == 2


def test_zero_shift_is_dropped_not_copied():
    island = (2, 3, 4)
    target = (0, 1, *island, 9)
    binds = bind_leased_islands(target, (_lease(island, source_start=2),))
    assert binds[0].action is BindAction.DROP
    assert binds[0].reason == "zero_shift"


def test_missing_or_ambiguous_span_is_dense():
    island = (2, 3, 4)
    absent = bind_leased_islands((9, 8, 7, 6), (_lease(island),))
    assert absent[0].action is BindAction.DENSE
    assert absent[0].reason == "not_in_target"
    duplicated = (1, *island, 0, *island, 9)
    ambiguous = bind_leased_islands(duplicated, (_lease(island),))
    assert ambiguous[0].action is BindAction.DENSE
    assert ambiguous[0].reason == "not_in_target"


def test_newer_duplicate_content_hash_wins():
    island = (2, 3, 4)
    target = (9, *island, 8)
    older = _lease(island, source_start=8, source_id="old", seq=1)
    newer = _lease(island, source_start=1, source_id="new", seq=7)
    binds = bind_leased_islands(target, (older, newer))
    by_id = {item.source_id: item for item in binds}
    assert "old" not in by_id
    assert by_id["new"].action is BindAction.DROP
    assert by_id["new"].reason == "zero_shift"


def test_newer_shifted_duplicate_copies_from_recent_source():
    island = (2, 3, 4)
    target = (9, 8, *island, 1)
    older = _lease(island, source_start=2, source_id="old", seq=1)
    newer = _lease(island, source_start=6, source_id="new", seq=7)
    binds = bind_leased_islands(target, (older, newer))
    copies = [item for item in binds if item.copies]
    assert len(copies) == 1
    assert copies[0].source_id == "new"
    assert copies[0].target_start == 2
    assert copies[0].rope_delta == -4


def test_reuse_plan_copies_unrotated_k_with_full_delta():
    island = (2, 3, 4)
    target = (9, 8, 7, *island, 1)
    handle = _handle(island, start=1)
    plan, binds = build_online_reuse_plan(
        target_token_ids=target,
        leases=(_lease(island, source_start=1, handle=handle),),
    )
    assert binds[0].copies
    assert len(plan.copied_spans) == 1
    assert plan.copied_spans[0].rope_delta == 2
    assert plan.copied_spans[0].target_start == 3
    assert plan.copied_spans[0].length == 3


def test_online_admit_module_has_no_prefetch_or_oracle_plan():
    source = inspect.getsource(online_admit)
    assert "ensure_resident" not in source
    assert "PLAN.json" not in source
    assert "managers.scheduler" not in source
    assert "target_start" not in inspect.getsource(admit_source_island)


def test_locate_unique_span():
    assert locate_unique_span((1, 2, 3, 4), (2, 3)) == 1
    assert locate_unique_span((1, 2, 3, 2, 3), (2, 3)) is None
    assert locate_unique_span((1, 2, 3), (9,)) is None
    assert locate_unique_span((1, 1, 1, 1), (1, 1, 1)) is None
    assert locate_unique_span((7, 8, 9), (7, 8, 9)) == 0
    assert locate_unique_span((1, 2), ()) is None
    assert locate_unique_span((), (1,)) is None


def _naive_unique_span(haystack, needle):
    n = len(needle)
    if n == 0 or n > len(haystack):
        return None
    hay = tuple(int(value) for value in haystack)
    need = tuple(int(value) for value in needle)
    found = None
    for index in range(len(hay) - n + 1):
        if hay[index : index + n] == need:
            if found is not None:
                return None
            found = index
    return found


def test_locate_unique_span_matches_naive_scan():
    import random

    rng = random.Random(0)
    for _ in range(40):
        hay = [rng.randrange(-80, 80) for _ in range(rng.randint(8, 90))]
        width = rng.randint(1, min(7, len(hay)))
        start = rng.randint(0, len(hay) - width)
        need = hay[start : start + width]
        assert locate_unique_span(hay, need) == _naive_unique_span(hay, need)
