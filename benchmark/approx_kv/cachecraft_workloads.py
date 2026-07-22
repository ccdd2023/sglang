"""Non-prefix segmented workload construction for Cache-Craft benchmarking.

Cache-Craft's decision logic (``cachecraft_metrics.py``) is fundamentally
about *reordered* and *partially reused* chunk sequences -- a new prompt
whose chunk order differs from the order the same chunks were originally
cached in (Eq. (6)-(8) Prefix Overlap Score / Order Penalty) -- not simple
contiguous-prefix reuse. The shared ``benchmark/approx_kv/workloads.py``
catalog (Phase 2) only ever measures plain prefix reuse, so it cannot
exercise this. This module builds a deterministic, GPU-free "non-prefix
segmented" workload: several canonical source chunks, each split into
<=512-token segments (the unified Phase 4 contract's canonical-source
segmentation rule), plus one or more *reordered* target chunk sequences
that reuse the same chunks in a different order.

This is scaffolding for a future real Cache-Craft server hook (see
``cachecraft_capability.py`` for the current, honest "not wired yet"
finding): nothing here talks to a model, a server, or a GPU. It only
produces plain Python token-id sequences and chunk metadata so that once a
real attention-profile/selected-token hook exists, a benchmark runner can
immediately build realistic non-prefix requests without redesigning the
workload shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Unified Phase 4 high-pressure contract (shared across all six research
# worktrees; see PROJECT.md "2026-07-22T12:37:07-07:00 High-pressure
# eviction设置纠正" and later entries). Defined once here so Cache-Craft's
# runner/tests never hand-copy these numbers.
UNIFIED_EXACT_HEADER_TOKENS: tuple[int, ...] = (0, 32, 64, 128, 256)
UNIFIED_LOSSY_BODY_TOKENS: tuple[int, ...] = (512, 768, 1024, 2048)
UNIFIED_TARGET_RHO: tuple[float, ...] = (0.9, 1.1, 1.5, 2.0, 3.0)
UNIFIED_MEM_FRACTION_STATIC: float = 0.35
UNIFIED_CANONICAL_SEGMENT_TOKENS: int = 512
UNIFIED_WARMUP_PASSES: int = 1
UNIFIED_DEFAULT_FORMAL_REPEATS: int = 4
UNIFIED_MIN_FORMAL_REPEATS: int = 2


@dataclass(frozen=True)
class CacheCraftChunkSegment:
    """One <=512-token physical registration unit of a canonical chunk."""

    chunk_id: str
    segment_index: int
    content_hash: str
    token_ids: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class CacheCraftChunk:
    """A canonical source chunk, segmented per the unified contract."""

    chunk_id: str
    token_ids: tuple[int, ...]
    segments: tuple[CacheCraftChunkSegment, ...]

    @property
    def length(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class NonPrefixSegmentedWorkload:
    """A set of canonical chunks plus one reordered target chunk sequence.

    ``target_chunk_order`` lists chunk ids in the order they appear in the
    *new* prompt; it may differ from ``canonical_chunk_order`` (the order
    the chunks were originally registered in), which is exactly the
    "non-prefix" scenario Cache-Craft's Order Penalty (Eq. (7)) is designed
    to score.
    """

    chunks: tuple[CacheCraftChunk, ...]
    canonical_chunk_order: tuple[str, ...]
    target_chunk_order: tuple[str, ...]
    header_token_ids: tuple[int, ...]
    final_token_id: int

    def chunk(self, chunk_id: str) -> CacheCraftChunk:
        for candidate in self.chunks:
            if candidate.chunk_id == chunk_id:
                return candidate
        raise KeyError(f"unknown chunk_id: {chunk_id}")

    @property
    def target_token_ids(self) -> tuple[int, ...]:
        body: list[int] = []
        for chunk_id in self.target_chunk_order:
            body.extend(self.chunk(chunk_id).token_ids)
        return (*self.header_token_ids, *body, self.final_token_id)

    @property
    def is_reordered(self) -> bool:
        return self.target_chunk_order != self.canonical_chunk_order


def segment_into_canonical_chunks(
    token_ids: Sequence[int],
    *,
    max_segment_tokens: int = UNIFIED_CANONICAL_SEGMENT_TOKENS,
) -> tuple[tuple[int, ...], ...]:
    """Split ``token_ids`` into <=``max_segment_tokens`` contiguous runs.

    This is the unified contract's "body>512 canonical source按<=512-token
    segments" rule, extracted into a standalone, independently testable
    function (mirrors the inline splitting R1's EPIC pressure runner does
    for its own long-body canonical sources).
    """
    if max_segment_tokens <= 0:
        raise ValueError("max_segment_tokens must be positive")
    token_ids = tuple(int(token_id) for token_id in token_ids)
    return tuple(
        token_ids[start : start + max_segment_tokens]
        for start in range(0, len(token_ids), max_segment_tokens)
    )


def build_canonical_chunk(
    chunk_id: str,
    token_ids: Sequence[int],
    *,
    max_segment_tokens: int = UNIFIED_CANONICAL_SEGMENT_TOKENS,
) -> CacheCraftChunk:
    token_ids = tuple(int(token_id) for token_id in token_ids)
    raw_segments = segment_into_canonical_chunks(
        token_ids,
        max_segment_tokens=max_segment_tokens,
    )
    segments = tuple(
        CacheCraftChunkSegment(
            chunk_id=chunk_id,
            segment_index=index,
            content_hash=f"{chunk_id}-segment{index}",
            token_ids=segment_tokens,
        )
        for index, segment_tokens in enumerate(raw_segments)
    )
    return CacheCraftChunk(chunk_id=chunk_id, token_ids=token_ids, segments=segments)


def _deterministic_chunk_tokens(
    chunk_id: str, length: int, base: int
) -> tuple[int, ...]:
    if length <= 0:
        raise ValueError("length must be positive")
    return tuple(base + offset for offset in range(length))


def build_non_prefix_segmented_workload(
    *,
    body_tokens: int,
    header_tokens: int = 0,
    num_chunks: int = 3,
    reorder_seed: int = 1,
    max_segment_tokens: int = UNIFIED_CANONICAL_SEGMENT_TOKENS,
    final_token_id: int = 9_001,
) -> NonPrefixSegmentedWorkload:
    """Build ``num_chunks`` canonical source chunks whose combined length is
    ``body_tokens``, each segmented per the unified <=512-token contract,
    plus a deterministic *reordered* target chunk sequence.

    ``header_tokens`` is the exact-match prefix length that precedes the
    (reordered) chunk body in the target prompt, per the unified contract's
    exact-header sweep. The reordering is a fixed cyclic rotation seeded by
    ``reorder_seed`` (not random per-call) so results are reproducible.
    """
    if body_tokens <= 0:
        raise ValueError("body_tokens must be positive")
    if header_tokens < 0:
        raise ValueError("header_tokens must not be negative")
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")

    base_length = body_tokens // num_chunks
    remainder = body_tokens - base_length * num_chunks
    chunk_lengths = [
        base_length + (1 if index < remainder else 0) for index in range(num_chunks)
    ]
    if any(length <= 0 for length in chunk_lengths):
        raise ValueError("body_tokens too small for num_chunks")

    chunks = []
    cursor = 100_000
    for index, length in enumerate(chunk_lengths):
        chunk_id = f"cachecraft-chunk-{index}"
        chunks.append(
            build_canonical_chunk(
                chunk_id,
                _deterministic_chunk_tokens(chunk_id, length, cursor),
                max_segment_tokens=max_segment_tokens,
            )
        )
        cursor += length + 1_000  # gap avoids accidental token-id collisions

    canonical_order = tuple(chunk.chunk_id for chunk in chunks)
    rotation = reorder_seed % num_chunks if num_chunks > 1 else 0
    target_order = canonical_order[rotation:] + canonical_order[:rotation]
    if num_chunks > 1 and target_order == canonical_order:
        # A rotation by a multiple of num_chunks is a no-op; force a
        # genuine reorder by swapping the first two chunks instead so the
        # workload is never accidentally prefix-identical.
        target_order = (canonical_order[1], canonical_order[0], *canonical_order[2:])

    header_token_ids = tuple(range(1, header_tokens + 1))
    return NonPrefixSegmentedWorkload(
        chunks=tuple(chunks),
        canonical_chunk_order=canonical_order,
        target_chunk_order=target_order,
        header_token_ids=header_token_ids,
        final_token_id=final_token_id,
    )
