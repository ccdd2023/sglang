from __future__ import annotations

"""Count-driven token selection for CacheTune (ported from CacheBlend).

Ported and adapted from `research/cacheblend`'s
`python/sglang/srt/mem_cache/cacheblend/hkvd.py` gradual (coarse-to-fine)
deviation-based token-selection funnel. The real-deviation scoring
primitive (`compute_token_deviation`) is carried over unchanged in
substance (same genuine per-token relative-L2 K deviation measurement
over real freshly-computed tensors, never a static/structural proxy --
see the historical falsification note below), but the *final selection*
is changed from **ratio**-driven to **count**-driven:

CacheTune's `controller.py` already performs one authoritative rounding
of a continuous ratio down to an executable integer `repair_tokens`
count (`hardware_profile.quantize_ratio`). If this module re-derived its
own rounded count from a ratio (as CacheBlend's `select_hkvd_tokens`
does via `max(1, round(total * final_ratio))`), a second, independently
-rounded count could silently disagree with the controller's decision by
an off-by-one. This module instead takes the controller's `final_count`
directly as an integer and both selects and *self-validates* (via
`TokenSelection.__post_init__`) that it produced exactly that many
tokens -- never more, never fewer -- so any future bug here fails loudly
at construction time rather than silently drifting from the controller's
decision.

`final_count = 0` is a legitimate, common outcome for this module (never
for CacheBlend's, whose minimum ratio was a fixed 1%): CacheTune's
`speed_only` mode explicitly allows a 0% floor (this project does not
optimize output quality), meaning "repair nothing, keep the raw
copied+RoPE-corrected KV as-is" is an honest, first-class decision here.

Historical negative lesson (see `research/cacheblend/hkvd.py` module
docstring and git history on
`archive/fix-placeholder-pool-activation-20260717`,
`feat(hkvd-multi-signal)`, the "True CacheBlend Path A" falsification): a
prior research line tried to *predict* high-KV-deviation tokens from
static code-structure signals (AST node kind, control-flow vs data-flow,
def/use, import distance). All 5 of those structural proxies were
falsified (p-values 0.97-1.0, several reversed in direction) -- code
structure does not predict KV deviation. This module therefore only
scores tokens from real, freshly-computed K (and optionally V) tensors,
never from AST or any other structural/static proxy.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .hardware_profile import round_half_up


def compute_token_deviation(
    fresh_keys: torch.Tensor,
    reused_keys: torch.Tensor,
    *,
    fresh_values: torch.Tensor | None = None,
    reused_values: torch.Tensor | None = None,
    value_weight: float = 0.0,
) -> torch.Tensor:
    """Compute a real, per-token KV deviation score.

    ``fresh_keys``/``reused_keys`` must be ``[num_tokens, ...]`` tensors of a
    single probe layer's K (and optionally V), aligned by token position:
    ``fresh_keys[i]`` is what the model actually computes for token ``i`` in
    the current request context; ``reused_keys[i]`` is the value physically
    sitting in the KV buffer after the raw copy+RoPE reuse step. The
    deviation is a relative L2 distance. By default only K is used
    (``value_weight=0.0``), matching CacheBlend's finding that K deviation
    is the informative signal; V deviation can optionally be blended in.
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
        raise ValueError("fresh and reused value tensors must have matching shape")
    flat_fresh_v = fresh_values.reshape(fresh_values.shape[0], -1).float()
    flat_reused_v = reused_values.reshape(reused_values.shape[0], -1).float()
    value_norm = torch.linalg.norm(flat_reused_v, dim=-1).clamp_min(1e-6)
    value_deviation = (
        torch.linalg.norm(flat_fresh_v - flat_reused_v, dim=-1) / value_norm
    )
    return (1.0 - value_weight) * key_deviation + value_weight * value_deviation


@dataclass(frozen=True)
class GradualFilterStage:
    """One probe stage of the coarse-to-fine selection funnel.

    Each stage measures deviation (via a caller-supplied real forward hook)
    only for the *current* candidate set -- not the full reusable range --
    and keeps the top ``keep_ratio`` fraction of that (shrinking) set,
    never fewer than the controller's authoritative ``final_count`` (see
    ``select_repair_tokens``). This is what makes the filtering "gradual":
    later, more expensive/precise probe layers only ever see a small,
    already-narrowed candidate pool.
    """

    probe_layer_id: int
    keep_ratio: float

    def __post_init__(self) -> None:
        if self.probe_layer_id < 0:
            raise ValueError("probe_layer_id must be non-negative")
        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be in (0, 1]")


@dataclass(frozen=True)
class TokenSelection:
    """Result of count-driven gradual filtering.

    ``requested_count`` is the controller's authoritative repair-token
    count this selection was asked to produce; the invariant
    ``len(selected_positions) == requested_count`` is enforced in
    ``__post_init__`` so any selection-logic bug fails immediately at
    construction time rather than silently reaching the caller.
    """

    candidate_positions: tuple[int, ...]
    requested_count: int
    selected_positions: tuple[int, ...]
    stage_scores: tuple[tuple[int, torch.Tensor], ...]

    def __post_init__(self) -> None:
        if not set(self.selected_positions) <= set(self.candidate_positions):
            raise ValueError(
                "selected positions must be a subset of the candidate positions"
            )
        if len(set(self.selected_positions)) != len(self.selected_positions):
            raise ValueError("selected positions must not contain duplicates")
        if len(self.selected_positions) != self.requested_count:
            raise RuntimeError(
                "token selection produced "
                f"{len(self.selected_positions)} positions but the "
                f"controller requested exactly {self.requested_count}"
            )


DeviationFn = Callable[[int, torch.Tensor], torch.Tensor]


def select_repair_tokens(
    candidate_positions: Sequence[int],
    *,
    stages: Sequence[GradualFilterStage],
    final_count: int,
    deviation_fn: DeviationFn,
) -> TokenSelection:
    """Run gradual (coarse-to-fine) filtering down to exactly ``final_count``.

    ``deviation_fn(probe_layer_id, positions) -> torch.Tensor`` must return a
    real, per-token deviation score (see ``compute_token_deviation``) aligned
    with ``positions`` -- computed from an actual probe-layer forward, never
    fabricated. The final selection always re-scores the surviving candidate
    pool with the *last* stage's probe layer (the deepest, most informative
    one available), so the final ``final_count`` selection is driven by real
    deviation scores, not just candidate-pool membership.

    ``final_count == 0`` and ``final_count == len(candidate_positions)``
    are both short-circuited without calling ``deviation_fn`` at all: when
    nothing will be repaired, or everything will be, no scoring is needed
    to decide *which* tokens to repair.
    """
    total = len(candidate_positions)
    if final_count < 0 or final_count > total:
        raise ValueError(f"final_count must be within [0, {total}], got {final_count}")

    if final_count == 0:
        return TokenSelection(
            candidate_positions=tuple(candidate_positions),
            requested_count=0,
            selected_positions=(),
            stage_scores=(),
        )
    if final_count == total:
        return TokenSelection(
            candidate_positions=tuple(candidate_positions),
            requested_count=final_count,
            selected_positions=tuple(sorted(candidate_positions)),
            stage_scores=(),
        )

    candidates = list(candidate_positions)
    stage_scores: list[tuple[int, torch.Tensor]] = []
    last_scores: torch.Tensor | None = None
    last_layer_id: int | None = None

    for stage in stages:
        if len(candidates) <= final_count:
            break
        positions_tensor = torch.as_tensor(candidates, dtype=torch.long)
        scores = deviation_fn(stage.probe_layer_id, positions_tensor)
        if scores.shape[0] != len(candidates):
            raise ValueError(
                "deviation_fn must return one score per candidate position"
            )
        stage_scores.append((stage.probe_layer_id, scores))
        last_layer_id = stage.probe_layer_id
        # Never shrink the candidate pool below `final_count`: the funnel
        # must always retain at least as many candidates as the
        # controller's authoritative repair-token count, so the final
        # re-scoring step below always has a large-enough pool to choose
        # from and can reach exactly `final_count` selections.
        keep_count = max(final_count, round_half_up(len(candidates) * stage.keep_ratio))
        keep_count = min(keep_count, len(candidates))
        order = torch.argsort(scores, descending=True)
        kept_indices = order[:keep_count]
        candidates = [candidates[int(i)] for i in kept_indices.tolist()]
        # Keep the per-candidate scores aligned with the *filtered and
        # reordered* candidates list so a later reuse of `last_scores`
        # indexes the same tokens it was computed for.
        last_scores = scores[kept_indices]

    positions_tensor = torch.as_tensor(candidates, dtype=torch.long)
    if (
        last_layer_id is not None
        and last_scores is not None
        and last_scores.shape[0] == len(candidates)
    ):
        final_scores = last_scores
    else:
        final_layer_id = stages[-1].probe_layer_id if stages else 0
        final_scores = deviation_fn(final_layer_id, positions_tensor)
        stage_scores.append((final_layer_id, final_scores))

    order = torch.argsort(final_scores, descending=True)
    selected = tuple(sorted(candidates[int(i)] for i in order[:final_count].tolist()))

    return TokenSelection(
        candidate_positions=tuple(candidate_positions),
        requested_count=final_count,
        selected_positions=selected,
        stage_scores=tuple(stage_scores),
    )
