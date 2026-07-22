"""Real, capability-gated selected-token recompute hook for Cache-Craft
partial repair.

Cache-Craft's core mechanism only makes sense if partial repair triggers an
actual forward computation for the selected tokens -- copying/planning alone
would silently degrade quality with no real correction. This module defines
the seam (`ChunkRecomputeHook`) through which that real recompute must flow,
and a `RadixKVTransferBackend`-compatible wrapper that invokes it from
`dense_prefill` instead of treating every dense range as a whole-request
fallback signal.

Capability gate / current blocker: a real production hook must perform an
actual model forward (e.g. an eager attention pass, or SGLang's
`ForwardMode.TARGET_VERIFY` machinery used by speculative decoding) that
writes genuine per-layer K/V into the caller-specified physical indices for
*exactly* the selected token positions, leaving already-copied positions
untouched. `TARGET_VERIFY` today is wired end-to-end only inside the
EAGLE/ngram speculative-decoding workers (see
`sglang.srt.speculative.eagle_worker_v2`, `spec_utils.py`) and is not
reachable as a standalone API from request-level code such as this module.
Until that (or an equivalent eager-attention forward) is exposed, no real
hook is available on the live GPU server path, and
`restore_request_via_cachecraft` (see `cachecraft_runtime.py`) must dense-
fallback the whole chunk whenever `recompute_hook is None`. This module is
written so that plugging in a real hook, once available, requires no
changes to the decision logic or execution plumbing above it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch

from .types import KVLayerTransferResult


class ChunkRecomputeHook(Protocol):
    """A real per-token recompute hook.

    Implementations must perform an actual forward computation for
    `token_ids` at physical positions `target_indices` and write genuine
    per-layer K/V values for every layer of `kvcache` into those indices.
    There is no metadata-only success path: returning a result without
    writing real K/V values violates this contract.
    """

    def recompute(
        self,
        *,
        kvcache: Any,
        target_indices: torch.Tensor,
        token_ids: tuple[int, ...],
        reason: str,
    ) -> KVLayerTransferResult: ...


class CacheCraftUnsupportedError(RuntimeError):
    """Raised when Cache-Craft partial repair needs a real selected-token
    recompute hook but none is available (unsupported model/layout)."""


@dataclass
class RecomputeInvocation:
    target_start: int
    length: int
    reason: str
    result: KVLayerTransferResult


@dataclass
class CacheCraftRecomputeBackend:
    """Wraps a real `RadixKVTransferBackend` so `dense_prefill` invokes a
    genuine selected-token recompute hook, instead of only recording a
    fallback reason for the whole request.

    `inner` supplies the real, already-validated `copy_and_rotate`
    implementation (device index copy + RoPE correction) from common core;
    this wrapper only changes what happens for the ranges Cache-Craft
    decided to recompute.
    """

    inner: Any
    kvcache: Any
    target_indices: Callable[[int, int], torch.Tensor]
    token_ids: Callable[[int, int], tuple[int, ...]]
    recompute_hook: ChunkRecomputeHook | None
    invocations: list[RecomputeInvocation] = field(default_factory=list)
    unsupported_reasons: list[str] = field(default_factory=list)

    def copy_and_rotate(self, **kwargs) -> KVLayerTransferResult:
        return self.inner.copy_and_rotate(**kwargs)

    def dense_prefill(self, *, target_start: int, length: int, reason: str) -> None:
        if self.recompute_hook is None:
            self.unsupported_reasons.append(reason)
            return
        indices = self.target_indices(target_start, length)
        tokens = self.token_ids(target_start, length)
        if len(tokens) != length:
            raise ValueError("token_ids callback returned the wrong length")
        result = self.recompute_hook.recompute(
            kvcache=self.kvcache,
            target_indices=indices,
            token_ids=tokens,
            reason=reason,
        )
        if result.copied_k_tokens != length or result.copied_v_tokens != length:
            raise CacheCraftUnsupportedError(
                "selected-token recompute hook did not cover the full "
                f"requested range ({reason})"
            )
        if result.rotated_k_tokens != result.copied_k_tokens:
            raise CacheCraftUnsupportedError(
                "selected-token recompute hook must produce fully "
                f"position-correct keys ({reason})"
            )
        self.invocations.append(
            RecomputeInvocation(
                target_start=target_start,
                length=length,
                reason=reason,
                result=result,
            )
        )

    @property
    def recomputed_tokens(self) -> int:
        return sum(invocation.length for invocation in self.invocations)
