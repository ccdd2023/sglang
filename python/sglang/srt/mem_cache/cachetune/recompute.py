from __future__ import annotations

"""Per-layer selected-token recomputation for CacheTune.

Ported from `research/cacheblend`'s
`python/sglang/srt/mem_cache/cacheblend/recompute.py`: the mechanism that
makes a "repair" step more than "raw reuse" is that, for the controller
-selected repair-token positions, every layer from the first repair layer
onward is *really* recomputed against the model's actual weights (genuine
K/V projections in the current context), while every other position keeps
the raw copied+RoPE-corrected KV untouched. CacheTune reuses this
mechanism unchanged in substance; only the names (`CacheBlend` ->
`CacheTune`) and the docstrings (dropping CacheBlend-specific HKVD
terminology in favor of "repair ratio") differ from the source.

Historical negative lesson (see `research/cacheblend/recompute.py` module
docstring and git history, `9e84d2f94` "P3 True CacheBlend Path A ->
FALSIFIED (5th falsification)"): an earlier attempt implemented selective
recompute as one small forward launch ("minipre") *per selected token*.
That regressed TTFT by +1129 ms (38x over the practical gate) and had
per-token p95 latency of 18 ms (2.3x over its own gate), because
per-token GPU kernel launches and Python scheduling overhead dominate.
The coordinator below deliberately forbids that pattern: every layer is
recomputed with exactly one batched call covering *all* selected
positions at once, identified by both their physical KV slot index
(where to write) and their absolute logical token position (needed for
correct RoPE / attention masking against the real context).
"""

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch


@dataclass(frozen=True)
class LayerRecomputeResult:
    """Outcome of one real, batched per-layer recompute call."""

    layer_id: int
    recomputed_slot_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.recomputed_slot_indices:
            raise ValueError("a layer recompute result must cover at least one slot")
        if len(set(self.recomputed_slot_indices)) != len(self.recomputed_slot_indices):
            raise ValueError("recomputed_slot_indices must not contain duplicates")


class CacheTuneLayerRecomputeBackend(Protocol):
    """Real per-layer selective recompute hook.

    A conforming implementation MUST run a single batched forward for
    ``layer_id`` covering every entry in ``slot_indices``/``token_positions``
    in one call (never a per-token loop), using the model's actual weights
    and the real current-request context, and write the resulting K/V
    directly into the KV cache buffer at ``slot_indices`` before returning.
    ``token_positions[i]`` is the absolute logical sequence position (for
    RoPE) of the token whose physical KV slot is ``slot_indices[i]``.
    """

    def recompute_layer(
        self,
        *,
        layer_id: int,
        slot_indices: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> LayerRecomputeResult: ...


class CacheTuneProbeBackend(Protocol):
    """Real shallow-layer forward hook used only for repair-token scoring.

    Must return genuine freshly-computed K (shape
    ``[len(slot_indices), ...]``) for ``layer_id``; it must not fabricate,
    interpolate, or reuse a cached value -- that would silently defeat the
    deviation measurement (see ``token_selection.compute_token_deviation``).
    """

    def probe_layer(
        self,
        *,
        layer_id: int,
        slot_indices: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor: ...


class CacheTuneCapabilityError(RuntimeError):
    """Raised when a real probe/recompute backend is not bound.

    This is the explicit, honest signal that the deep per-layer model hook
    required by true inline selective recompute is unavailable in the
    current process; callers must treat it as a capability gap and fall
    back to a dense (fully real) forward -- never silently substitute the
    raw copied KV and call it a CacheTune repair.
    """


class LayerRecomputeCoordinator:
    """Drives real per-layer recompute for a fixed, pre-selected token set.

    One instance is used per reuse-restore call. ``recompute_selected``
    issues exactly one backend call per layer (``layer_num -
    first_recompute_layer`` calls total), each covering the *entire*
    selected set, and validates that the backend actually covered exactly
    the requested slots -- never more, never fewer, never one at a time.
    """

    def __init__(
        self,
        backend: CacheTuneLayerRecomputeBackend,
        *,
        first_recompute_layer: int,
        layer_num: int,
    ) -> None:
        if first_recompute_layer < 0:
            raise ValueError("first_recompute_layer must be non-negative")
        if layer_num <= first_recompute_layer:
            raise ValueError("layer_num must be greater than first_recompute_layer")
        self._backend = backend
        self._first_recompute_layer = first_recompute_layer
        self._layer_num = layer_num

    @property
    def recomputed_layer_ids(self) -> tuple[int, ...]:
        return tuple(range(self._first_recompute_layer, self._layer_num))

    def recompute_selected(
        self,
        *,
        slot_indices: Sequence[int],
        token_positions: Sequence[int],
    ) -> tuple[LayerRecomputeResult, ...]:
        if len(slot_indices) != len(token_positions):
            raise ValueError("slot_indices and token_positions must align 1:1")
        if not slot_indices:
            return ()
        ordered_pairs = sorted(zip(slot_indices, token_positions))
        ordered_slots = [int(slot) for slot, _ in ordered_pairs]
        if len(set(ordered_slots)) != len(ordered_slots):
            raise ValueError("slot_indices must not contain duplicates")
        ordered_positions = [int(position) for _, position in ordered_pairs]
        slots_tensor = torch.as_tensor(ordered_slots, dtype=torch.long)
        positions_tensor = torch.as_tensor(ordered_positions, dtype=torch.long)
        expected = set(ordered_slots)

        results = []
        for layer_id in self.recomputed_layer_ids:
            result = self._backend.recompute_layer(
                layer_id=layer_id,
                slot_indices=slots_tensor,
                token_positions=positions_tensor,
            )
            if result.layer_id != layer_id:
                raise RuntimeError(
                    "layer recompute backend returned a mismatched layer_id"
                )
            if set(result.recomputed_slot_indices) != expected:
                raise RuntimeError(
                    "layer recompute backend did not cover exactly the "
                    "selected slots in a single batched call"
                )
            results.append(result)
        return tuple(results)
