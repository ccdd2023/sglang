from __future__ import annotations

from typing import Protocol

from .store import ApproxKVSegmentStore
from .types import (
    KVLayerTransferResult,
    KVReusePlan,
    KVTransferStats,
    ResidencyTier,
)


class KVTransferBackend(Protocol):
    def copy_and_rotate(
        self,
        *,
        source_ref: object,
        source_offset: int,
        target_start: int,
        length: int,
        rope_delta: int,
    ) -> KVLayerTransferResult: ...

    def dense_prefill(
        self,
        *,
        target_start: int,
        length: int,
        reason: str,
    ) -> None: ...


class KVTransferInvariantError(RuntimeError):
    pass


def _range(start: int, length: int) -> range:
    return range(start, start + length)


def _validate_bounds(plan: KVReusePlan) -> None:
    target_len = len(plan.target_token_ids)
    occupied: set[int] = set()

    for dense in plan.dense_ranges:
        if dense.target_start + dense.length > target_len:
            raise ValueError("dense range exceeds target token sequence")
        positions = set(_range(dense.target_start, dense.length))
        if occupied & positions:
            raise ValueError("reuse plan contains overlapping target ranges")
        occupied |= positions

    for span in plan.copied_spans:
        if span.target_start + span.length > target_len:
            raise ValueError("copy range exceeds target token sequence")
        if span.chunk_start + span.chunk_length > target_len:
            raise ValueError("copy chunk exceeds target token sequence")
        if span.source_offset + span.length > len(span.source.token_ids):
            raise ValueError("copy range exceeds source token sequence")
        positions = set(_range(span.target_start, span.length))
        if occupied & positions:
            raise ValueError("reuse plan contains overlapping target ranges")
        occupied |= positions

    if plan.require_full_coverage and occupied != set(range(target_len)):
        raise ValueError("reuse plan leaves an unowned target gap")


def _contiguous_ranges(positions: list[int]) -> list[tuple[int, int]]:
    if not positions:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            ranges.append((start, previous - start + 1))
            start = position
        previous = position
    ranges.append((start, previous - start + 1))
    return ranges


def execute_reuse_plan(
    *,
    plan: KVReusePlan,
    store: ApproxKVSegmentStore,
    backend: KVTransferBackend,
) -> KVTransferStats:
    _validate_bounds(plan)
    stats = KVTransferStats(
        recovery_mode=plan.recovery_mode,
        target_tokens=len(plan.target_token_ids),
    )
    fallback_chunks: dict[tuple[int, int], str] = {}

    for span in plan.copied_spans:
        chunk = (span.chunk_start, span.chunk_length)
        if not store.is_current(span.source):
            stats.stale_handle += 1
            fallback_chunks[chunk] = "stale_handle"
            continue
        if span.source.residency != ResidencyTier.DEVICE:
            stats.residency_miss += 1
            fallback_chunks[chunk] = "residency_miss"
            continue

        source_tokens = span.source.token_ids[
            span.source_offset : span.source_offset + span.length
        ]
        target_tokens = plan.target_token_ids[
            span.target_start : span.target_start + span.length
        ]
        if source_tokens != target_tokens:
            stats.source_slice_mismatch += 1
            fallback_chunks[chunk] = "source_slice_mismatch"

    dense_positions: set[int] = set()
    for (chunk_start, chunk_length), reason in sorted(fallback_chunks.items()):
        backend.dense_prefill(
            target_start=chunk_start,
            length=chunk_length,
            reason=reason,
        )
        dense_positions.update(_range(chunk_start, chunk_length))
        stats.fallback_reasons.append(reason)

    for dense in plan.dense_ranges:
        remaining = [
            position
            for position in _range(dense.target_start, dense.length)
            if position not in dense_positions
        ]
        for start, length in _contiguous_ranges(remaining):
            backend.dense_prefill(
                target_start=start,
                length=length,
                reason=dense.reason,
            )
            dense_positions.update(_range(start, length))

    stats.recomputed_tokens = len(dense_positions)

    for span in plan.copied_spans:
        if (span.chunk_start, span.chunk_length) in fallback_chunks:
            continue
        result = backend.copy_and_rotate(
            source_ref=span.source.backend_ref,
            source_offset=span.source_offset,
            target_start=span.target_start,
            length=span.length,
            rope_delta=span.rope_delta,
        )
        if (
            result.copied_k_tokens != span.length
            or result.copied_v_tokens != span.length
        ):
            raise KVTransferInvariantError(
                "backend did not copy the complete requested K/V slice"
            )
        if result.rotated_k_tokens != result.copied_k_tokens:
            raise KVTransferInvariantError(
                "all copied K tokens must receive full RoPE correction"
            )
        stats.copied_k_tokens += result.copied_k_tokens
        stats.rotated_k_tokens += result.rotated_k_tokens
        stats.copied_v_tokens += result.copied_v_tokens
        stats.copy_ms += result.copy_ms
        stats.rope_ms += result.rope_ms

    if plan.require_full_coverage and not stats.mechanically_valid:
        raise KVTransferInvariantError(
            "reuse execution did not account for every target token"
        )
    return stats
