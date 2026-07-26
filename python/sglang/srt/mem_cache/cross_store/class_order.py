from __future__ import annotations

_S4_KIND_CLASS = {
    "materialization_scratch": 0,
    "host_copy": 1,
    "repair_state": 2,
    "repair_metadata": 2,
    "precomputed_adapter": 2,
    "delta": 2,
    "anchor": 3,
    "exact_variant": 4,
    "stage_variant": 4,
    "filler": 4,
    "canonical_base": 5,
}


def s4_class(
    kind: str,
    *,
    retired: bool,
    recoverable_from_lower_tier: bool,
) -> int:
    if retired:
        return 0
    if kind in {"exact_variant", "stage_variant"} and recoverable_from_lower_tier:
        return 1
    return _S4_KIND_CLASS.get(kind, 5)


def s4_next_use_key(
    next_use: int | None,
    fallback_distance: int | None = None,
) -> int:
    resolved = (
        next_use
        if next_use is not None
        else (fallback_distance if fallback_distance is not None else (1 << 63) - 1)
    )
    return -resolved
