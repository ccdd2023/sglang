"""Source-time admit and target-time bind for true-lossy file islands.

The frozen PLAN used by the headline campaign is an offline oracle: it
sees the target token IDs before it decides what to lease. This module is
the online policy that does not.

Admit inspects only the source observation (single-file repository_code,
version-valid, later-roles in the coding protocol). It never reads a
target index, a target hash, or remaining target_uses.

Bind runs when a target prompt is already in hand. It locates each leased
island by token identity, computes Δ = t − s, and fail-closes to Dense.
K is stored unrotated (pre_rotate_delta = 0); rotation happens at copy.
"""

from __future__ import annotations

import array
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from sglang.srt.mem_cache.coding_aware.policy import (
    CodingRisk,
    CodingSegment,
    build_coding_reuse_plan,
)
from sglang.srt.mem_cache.kvcomm.types import KVReusePlan, KVSegmentHandle


def protocol_later_roles(
    policy_label: str, explicit: int | None = None
) -> int:
    """Later roles from the coding protocol, not remaining target_uses."""
    if explicit is not None:
        return int(explicit)
    if "coding" in (policy_label or ""):
        return 3
    return 0


@dataclass(frozen=True)
class SourceObservation:
    """What the source request can see. No target fields."""

    source_id: str
    source_start: int
    token_ids: tuple[int, ...]
    content_hash: str
    source_prefix_hash: str
    single_file_repository_code: bool
    version_valid: bool
    later_roles_in_protocol: int
    seq: int = 0
    policy_label: str = "coding"


def mechanical_source_gates(obs: SourceObservation) -> str | None:
    """Hard gates that never use a target prompt or learned scores."""
    if not obs.source_id:
        return "missing_source_id"
    if not obs.single_file_repository_code:
        return "not_single_file_repository_code"
    if not obs.version_valid:
        return "version_invalid"
    if obs.source_start <= 0 or not obs.token_ids:
        return "not_strictly_middle"
    if not obs.content_hash or not obs.source_prefix_hash:
        return "incomplete_page_identity"
    return None


def admit_source_island(obs: SourceObservation) -> str | None:
    """Protocol-only admit. Return a skip reason, or None to lease.

    This function must not take a target start, target hash, or Δ.
    Later-roles is the cold-start prior; the class template can override
    it after observing bind outcomes.
    """
    reason = mechanical_source_gates(obs)
    if reason is not None:
        return reason
    if obs.later_roles_in_protocol <= 0:
        return "no_protocol_reread"
    return None


@dataclass(frozen=True)
class LeasedIsland:
    source_id: str
    source_start: int
    token_ids: tuple[int, ...]
    content_hash: str
    source_prefix_hash: str
    seq: int = 0
    handle: KVSegmentHandle | None = None


class BindAction(str, Enum):
    COPY = "copy"
    DENSE = "dense"
    DROP = "drop"


@dataclass(frozen=True)
class BindResult:
    action: BindAction
    reason: str
    source_id: str | None = None
    source_start: int | None = None
    target_start: int | None = None
    length: int = 0
    rope_delta: int = 0
    content_hash: str = ""
    handle: KVSegmentHandle | None = None

    @property
    def copies(self) -> bool:
        return self.action is BindAction.COPY


def locate_unique_span(
    haystack: Sequence[int], needle: Sequence[int]
) -> int | None:
    """Return the unique start of needle in haystack, else None.

    Packed-int ``bytes.find`` stays on the target TTFT path; a naive
    Python scan of a 5k-token prompt was ~8 ms per island.
    """
    n = len(needle)
    if n == 0 or n > len(haystack):
        return None
    packed_hay = array.array("q", (int(value) for value in haystack)).tobytes()
    packed_need = array.array("q", (int(value) for value in needle)).tobytes()
    width = array.array("q").itemsize
    found: int | None = None
    start = 0
    while True:
        idx = packed_hay.find(packed_need, start)
        if idx < 0:
            return found
        if idx % width == 0:
            if found is not None:
                return None
            found = idx // width
            start = idx + width
        else:
            start = idx + 1


def bind_leased_islands(
    target_token_ids: Sequence[int],
    leases: Sequence[LeasedIsland],
) -> tuple[BindResult, ...]:
    """Match leased source pages onto a target prompt that is already known.

    Uses the target request's own tokens. Does not read a precomputed (s, t)
    pair. Duplicate content hashes keep the newest seq. Zero-shift matches
    are dropped so the path cannot collapse to prefix reuse.
    """
    target = tuple(int(value) for value in target_token_ids)
    newest: dict[str, LeasedIsland] = {}
    for lease in leases:
        current = newest.get(lease.content_hash)
        if current is None or lease.seq >= current.seq:
            newest[lease.content_hash] = lease

    ranked = sorted(
        newest.values(),
        key=lambda item: (-len(item.token_ids), -item.seq, item.source_id),
    )
    occupied: set[int] = set()
    results: list[BindResult] = []
    for lease in ranked:
        start = locate_unique_span(target, lease.token_ids)
        length = len(lease.token_ids)
        if start is None:
            results.append(
                BindResult(
                    action=BindAction.DENSE,
                    reason="not_in_target",
                    source_id=lease.source_id,
                    source_start=lease.source_start,
                    length=length,
                    content_hash=lease.content_hash,
                    handle=lease.handle,
                )
            )
            continue
        span = set(range(start, start + length))
        if occupied & span:
            results.append(
                BindResult(
                    action=BindAction.DROP,
                    reason="overlap_superseded",
                    source_id=lease.source_id,
                    source_start=lease.source_start,
                    target_start=start,
                    length=length,
                    content_hash=lease.content_hash,
                    handle=lease.handle,
                )
            )
            continue
        if tuple(target[start : start + length]) != lease.token_ids:
            results.append(
                BindResult(
                    action=BindAction.DENSE,
                    reason="token_ids_mismatch",
                    source_id=lease.source_id,
                    source_start=lease.source_start,
                    target_start=start,
                    length=length,
                    content_hash=lease.content_hash,
                    handle=lease.handle,
                )
            )
            continue
        delta = start - lease.source_start
        if delta == 0:
            results.append(
                BindResult(
                    action=BindAction.DROP,
                    reason="zero_shift",
                    source_id=lease.source_id,
                    source_start=lease.source_start,
                    target_start=start,
                    length=length,
                    content_hash=lease.content_hash,
                    handle=lease.handle,
                )
            )
            continue
        occupied |= span
        results.append(
            BindResult(
                action=BindAction.COPY,
                reason="online_bind",
                source_id=lease.source_id,
                source_start=lease.source_start,
                target_start=start,
                length=length,
                rope_delta=delta,
                content_hash=lease.content_hash,
                handle=lease.handle,
            )
        )
    results.sort(
        key=lambda item: (
            item.target_start is None,
            item.target_start if item.target_start is not None else 0,
            item.source_id or "",
        )
    )
    return tuple(results)


def build_online_reuse_plan(
    *,
    target_token_ids: Sequence[int],
    leases: Sequence[LeasedIsland],
) -> tuple[KVReusePlan, tuple[BindResult, ...]]:
    """Translate online binds into a complete KVCOMM plan.

    COPY spans use the leased handle. Everything else stays Dense.
    Stored K is unrotated, so rope_delta is the full t − s.
    """
    binds = bind_leased_islands(target_token_ids, leases)
    segments: list[CodingSegment] = []
    for index, bind in enumerate(binds):
        if bind.action is not BindAction.COPY or bind.target_start is None:
            continue
        if bind.handle is None:
            continue
        segments.append(
            CodingSegment(
                slot_id=bind.source_id or f"online-{index}",
                target_start=bind.target_start,
                token_ids=bind.handle.token_ids,
                risk=CodingRisk.STABLE,
                source=bind.handle,
            )
        )
    return (
        build_coding_reuse_plan(
            target_token_ids=target_token_ids,
            segments=tuple(segments),
        ),
        binds,
    )
