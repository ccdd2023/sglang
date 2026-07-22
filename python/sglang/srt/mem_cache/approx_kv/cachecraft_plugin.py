"""Cache-Craft `RecoveryPlugin` implementation: direct-reuse vs
partial-repair vs full-recompute decision, built on the frozen common-core
`RecoveryPlugin` protocol (`plugins.py`), `KVReusePlan`/`TransferSpan`/
`DenseRange` (`types.py`), and `ApproxKVSegmentStore` (`store.py`).

This module only adds new, Cache-Craft-specific logic; it does not modify
any common-core file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .cachecraft_metrics import (
    CacheCraftDecision,
    ChunkContextProfile,
    adjusted_beta,
    compute_beta,
    compute_cci,
    compute_cfo,
    decide,
    kendall_tau_order_penalty,
    select_recompute_positions,
)
from .plugins import RecoveryRequestContext
from .store import ApproxKVSegmentStore
from .types import (
    DenseRange,
    KVReusePlan,
    KVSegmentKey,
    RecoveryMode,
    SchedulerMetadata,
    TransferSpan,
)


class CacheCraftProfileStore:
    """Registry of `ChunkContextProfile`s, keyed by chunk id.

    Deliberately separate from `ApproxKVSegmentStore`: this only ever holds
    real attention statistics (see `cachecraft_attention.py`), never raw K/V
    payloads, so it cannot become a second path for approximate results to
    reach the exact Radix cache.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, tuple[int, ChunkContextProfile]] = {}

    def register(
        self,
        profile: ChunkContextProfile,
        *,
        generation: int = 1,
    ) -> None:
        if generation <= 0:
            raise ValueError("profile generation must be positive")
        self._profiles[profile.chunk_id] = (generation, profile)

    def get(
        self,
        chunk_id: str,
        *,
        generation: int,
    ) -> ChunkContextProfile | None:
        record = self._profiles.get(chunk_id)
        if record is None or record[0] != generation:
            return None
        return record[1]

    def reset(self) -> None:
        self._profiles.clear()

    def __len__(self) -> int:
        return len(self._profiles)


@dataclass(frozen=True)
class CacheCraftDecisionTrace:
    """Deterministic, inspectable record of one Cache-Craft decision.

    Exists so tests (and telemetry) can assert on the exact CCI / order
    penalty / CFO values that produced a decision, not just the decision
    itself.
    """

    chunk_id: str
    cci: float
    beta: float
    gamma: float
    beta_prime: float
    cfo: float
    decision: CacheCraftDecision
    recompute_positions: tuple[int, ...]


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


class CacheCraftPlugin:
    """`RecoveryPlugin` implementation for Cache-Craft chunk-cache reuse.

    One `build_plan` call handles exactly one chunk: `context.custom_metadata`
    must contain

    - `chunk_id` (str): the chunk being considered for reuse.
    - `chunk_key` (`KVSegmentKey`): the store lookup key for its cached KV.
    - `chunk_start` / `chunk_length` (int): the chunk's offset/length within
      `context.target_token_ids`.
    - `new_prefix_order` (sequence[str]): the ordered chunk ids preceding
      this chunk in the *new* prompt (`S_new`, Eq. (6)-(8)).

    The plan's `target_token_ids`/positions are expressed in chunk-local
    coordinates (0-based from `chunk_start`); the runtime layer
    (`cachecraft_runtime.py`) is responsible for translating them into
    absolute request/pool positions.
    """

    name = "cachecraft"

    def __init__(
        self,
        profiles: CacheCraftProfileStore,
        *,
        alpha: float = 1.0,
        full_recompute_threshold: float = 1.0,
    ) -> None:
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if not (0.0 < full_recompute_threshold <= 1.0):
            raise ValueError("full_recompute_threshold must be in (0, 1]")
        self.profiles = profiles
        self.alpha = alpha
        self.full_recompute_threshold = full_recompute_threshold
        self.last_trace: CacheCraftDecisionTrace | None = None

    def build_plan(
        self,
        context: RecoveryRequestContext,
        store: ApproxKVSegmentStore,
    ) -> KVReusePlan:
        meta: Mapping[str, object] = context.custom_metadata
        chunk_id = str(meta["chunk_id"])
        chunk_key: KVSegmentKey = meta["chunk_key"]  # type: ignore[assignment]
        chunk_start = int(meta["chunk_start"])  # type: ignore[arg-type]
        chunk_length = int(meta["chunk_length"])  # type: ignore[arg-type]
        new_prefix_order = tuple(str(c) for c in meta.get("new_prefix_order", ()))

        if chunk_start < 0 or chunk_length <= 0:
            raise ValueError("chunk_start/chunk_length must be valid bounds")
        target_tokens = context.target_token_ids[
            chunk_start : chunk_start + chunk_length
        ]
        if len(target_tokens) != chunk_length:
            raise ValueError("chunk_start/chunk_length exceed target token sequence")

        handle = store.lookup(chunk_key)
        profile = (
            None
            if handle is None
            else self.profiles.get(
                chunk_id,
                generation=handle.generation,
            )
        )
        cache_hit = profile is not None and handle is not None

        if not cache_hit:
            self.last_trace = CacheCraftDecisionTrace(
                chunk_id=chunk_id,
                cci=1.0,
                beta=0.0,
                gamma=0.0,
                beta_prime=0.0,
                cfo=1.0,
                decision=CacheCraftDecision.FULL_RECOMPUTE,
                recompute_positions=tuple(range(chunk_length)),
            )
            return KVReusePlan(
                target_token_ids=target_tokens,
                recovery_mode=RecoveryMode.DENSE,
                dense_ranges=(
                    DenseRange(0, chunk_length, "cachecraft_no_profile_or_handle"),
                ),
                require_full_coverage=True,
            )

        assert profile is not None and handle is not None
        common_ids = set(profile.old_prefix_order) & set(new_prefix_order)
        beta = compute_beta(profile, new_prefix_order)
        gamma = kendall_tau_order_penalty(
            profile.old_prefix_order,
            new_prefix_order,
            common_ids,
        )
        beta_prime = adjusted_beta(beta, gamma)
        cci = compute_cci(profile)
        cfo = compute_cfo(cci, beta_prime, self.alpha)
        decision = decide(
            cfo,
            cache_hit=True,
            full_recompute_threshold=self.full_recompute_threshold,
        )

        if decision is CacheCraftDecision.DIRECT_REUSE:
            positions: tuple[int, ...] = ()
            plan = KVReusePlan(
                target_token_ids=target_tokens,
                recovery_mode=RecoveryMode.COPY,
                copied_spans=(
                    TransferSpan(
                        source=handle,
                        source_offset=0,
                        target_start=0,
                        length=chunk_length,
                        rope_delta=0,
                        chunk_start=0,
                        chunk_length=chunk_length,
                    ),
                ),
                require_full_coverage=True,
            )
        elif decision is CacheCraftDecision.FULL_RECOMPUTE:
            positions = tuple(range(chunk_length))
            plan = KVReusePlan(
                target_token_ids=target_tokens,
                recovery_mode=RecoveryMode.DENSE,
                dense_ranges=(DenseRange(0, chunk_length, "cachecraft_cfo_full"),),
                require_full_coverage=True,
            )
        else:
            positions = select_recompute_positions(profile, cfo)
            occupied = set(positions)
            dense_ranges = tuple(
                DenseRange(start, length, "cachecraft_partial_repair")
                for start, length in _contiguous_ranges(sorted(occupied))
            )
            copy_positions = [
                position for position in range(chunk_length) if position not in occupied
            ]
            copied_spans = tuple(
                TransferSpan(
                    source=handle,
                    source_offset=start,
                    target_start=start,
                    length=length,
                    rope_delta=0,
                    chunk_start=0,
                    chunk_length=chunk_length,
                )
                for start, length in _contiguous_ranges(copy_positions)
            )
            plan = KVReusePlan(
                target_token_ids=target_tokens,
                recovery_mode=RecoveryMode.COPY,
                dense_ranges=dense_ranges,
                copied_spans=copied_spans,
                require_full_coverage=True,
            )

        self.last_trace = CacheCraftDecisionTrace(
            chunk_id=chunk_id,
            cci=cci,
            beta=beta,
            gamma=gamma,
            beta_prime=beta_prime,
            cfo=cfo,
            decision=decision,
            recompute_positions=positions,
        )
        return plan

    def scheduler_metadata(
        self,
        context: RecoveryRequestContext,
    ) -> tuple[SchedulerMetadata, ...]:
        del context
        return ()
