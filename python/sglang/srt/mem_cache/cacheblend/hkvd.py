from __future__ import annotations

"""Real HKVD (high KV deviation) measurement and gradual filtering.

This module implements the actual selection mechanism from CacheBlend
(arXiv paper terminology: "HKVD" tokens are the tokens whose reused KV
most deviates from what the model would compute for them in the current
(blended) context). It is intentionally narrow: it only scores and
selects token positions. It never fabricates a deviation score and never
mutates a KV buffer itself -- that is the job of ``recompute.py``.

Historical negative lesson (see git history on
``archive/fix-placeholder-pool-activation-20260717``,
``feat(hkvd-multi-signal)`` and the "True CacheBlend Path A" falsification):
a prior research line tried to *predict* HKVD tokens from static
code-structure signals (AST node kind, control-flow vs data-flow, def/use,
import distance). All 5 of those structural proxies were falsified
(p-values 0.97-1.0, several reversed in direction) -- code structure does
not predict KV deviation. This module therefore only scores tokens from
*real* freshly-computed K (and optionally V) tensors, never from AST or
any other structural/static proxy.
"""

from dataclasses import dataclass
from typing import Sequence

import torch


def compute_token_deviation(
    fresh_keys: torch.Tensor,
    reused_keys: torch.Tensor,
    *,
    fresh_values: torch.Tensor | None = None,
    reused_values: torch.Tensor | None = None,
    value_weight: float = 0.0,
) -> torch.Tensor:
    """Compute a real, per-token HKVD deviation score.

    ``fresh_keys``/``reused_keys`` must be ``[num_tokens, ...]`` tensors of a
    single probe layer's K (and optionally V), aligned by token position:
    ``fresh_keys[i]`` is what the model actually computes for token ``i`` in
    the current request context; ``reused_keys[i]`` is the value physically
    sitting in the KV buffer after the raw copy+RoPE reuse step. The
    deviation is a relative L2 distance, matching CacheBlend's KV deviation
    metric. By default only K is used (``value_weight=0.0``), matching the
    paper's finding that K deviation is the informative signal; V deviation
    can optionally be blended in for experimentation.
    """
    if fresh_keys.shape != reused_keys.shape:
        raise ValueError("fresh and reused key tensors must have matching shape")
    if fresh_keys.ndim < 1 or fresh_keys.shape[0] == 0:
        raise ValueError("key tensors must contain at least one token")
    if not (0.0 <= value_weight <= 1.0):
        raise ValueError("value_weight must be in [0, 1]")

    flat_fresh = fresh_keys.reshape(fresh_keys.shape[0], -1).float()
    flat_reused = reused_keys.reshape(reused_keys.shape[0], -1).float()
    key_norm = torch.linalg.norm(flat_reused, dim=-1).clamp_min(1e-6)
    key_deviation = torch.linalg.norm(flat_fresh - flat_reused, dim=-1) / key_norm

    if value_weight == 0.0:
        return key_deviation

    if fresh_values is None or reused_values is None:
        raise ValueError("value_weight > 0 requires fresh_values and reused_values")
    if fresh_values.shape != reused_values.shape:
        raise ValueError(
            "fresh and reused value tensors must have matching shape"
        )
    flat_fresh_v = fresh_values.reshape(fresh_values.shape[0], -1).float()
    flat_reused_v = reused_values.reshape(reused_values.shape[0], -1).float()
    value_norm = torch.linalg.norm(flat_reused_v, dim=-1).clamp_min(1e-6)
    value_deviation = (
        torch.linalg.norm(flat_fresh_v - flat_reused_v, dim=-1) / value_norm
    )
    return (1.0 - value_weight) * key_deviation + value_weight * value_deviation


@dataclass(frozen=True)
class GradualFilterStage:
    """One probe stage of the coarse-to-fine HKVD selection funnel.

    Each stage measures deviation (via a caller-supplied real forward hook)
    only for the *current* candidate set -- not the full reusable range --
    and keeps the top ``keep_ratio`` fraction of that (shrinking) set. This
    is what makes the filtering "gradual": later, more expensive/precise
    probe layers only ever see a small, already-narrowed candidate pool.
    """

    probe_layer_id: int
    keep_ratio: float

    def __post_init__(self) -> None:
        if self.probe_layer_id < 0:
            raise ValueError("probe_layer_id must be non-negative")
        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be in (0, 1]")


@dataclass(frozen=True)
class HKVDSelection:
    """Result of gradual filtering: the final selected token positions."""

    candidate_positions: tuple[int, ...]
    selected_positions: tuple[int, ...]
    stage_scores: tuple[tuple[int, torch.Tensor], ...]
    final_scores: torch.Tensor

    def __post_init__(self) -> None:
        if not set(self.selected_positions) <= set(self.candidate_positions):
            raise ValueError(
                "selected positions must be a subset of the candidate positions"
            )


def select_hkvd_tokens(
    candidate_positions: Sequence[int],
    *,
    stages: Sequence[GradualFilterStage],
    final_ratio: float,
    deviation_fn,
) -> HKVDSelection:
    """Run gradual (coarse-to-fine) HKVD filtering down to ``final_ratio``.

    ``deviation_fn(probe_layer_id, positions) -> torch.Tensor`` must return a
    real, per-token deviation score (see ``compute_token_deviation``) aligned
    with ``positions`` -- computed from an actual probe-layer forward, never
    fabricated. The final selection always uses the *last* stage's probe
    layer (the deepest, most informative one available) re-scored on the
    surviving candidate pool, so the final ``final_ratio`` selection is
    driven by real HKVD scores, not just candidate-pool membership.
    """
    if not (0.0 < final_ratio <= 1.0):
        raise ValueError("final_ratio must be in (0, 1]")
    if not candidate_positions:
        raise ValueError("candidate_positions must not be empty")

    candidates = list(candidate_positions)
    total = len(candidates)
    stage_scores: list[tuple[int, torch.Tensor]] = []
    last_scores: torch.Tensor | None = None
    last_layer_id: int | None = None

    for stage in stages:
        if not candidates:
            break
        positions_tensor = torch.as_tensor(candidates, dtype=torch.long)
        scores = deviation_fn(stage.probe_layer_id, positions_tensor)
        if scores.shape[0] != len(candidates):
            raise ValueError(
                "deviation_fn must return one score per candidate position"
            )
        stage_scores.append((stage.probe_layer_id, scores))
        last_layer_id = stage.probe_layer_id
        keep_count = max(1, round(len(candidates) * stage.keep_ratio))
        order = torch.argsort(scores, descending=True)
        kept_indices = order[:keep_count]
        candidates = [candidates[int(i)] for i in kept_indices.tolist()]
        # Keep the per-candidate scores aligned with the *filtered and
        # reordered* candidates list so a later reuse of `last_scores`
        # indexes the same tokens it was computed for.
        last_scores = scores[kept_indices]

    final_count = max(1, round(total * final_ratio))
    final_count = min(final_count, len(candidates)) if candidates else 0

    if not candidates:
        return HKVDSelection(
            candidate_positions=tuple(candidate_positions),
            selected_positions=(),
            stage_scores=tuple(stage_scores),
            final_scores=torch.empty(0),
        )

    # Re-score the surviving pool with the deepest available probe layer so
    # the *final* ranking (not just funnel membership) is HKVD-driven.
    positions_tensor = torch.as_tensor(candidates, dtype=torch.long)
    if last_layer_id is not None and len(candidates) == (
        last_scores.shape[0] if last_scores is not None else -1
    ):
        final_scores = last_scores
    else:
        final_layer_id = stages[-1].probe_layer_id if stages else 0
        final_scores = deviation_fn(final_layer_id, positions_tensor)
        stage_scores.append((final_layer_id, final_scores))

    order = torch.argsort(final_scores, descending=True)
    selected = tuple(sorted(candidates[int(i)] for i in order[:final_count].tolist()))

    return HKVDSelection(
        candidate_positions=tuple(candidate_positions),
        selected_positions=selected,
        stage_scores=tuple(stage_scores),
        final_scores=final_scores,
    )
