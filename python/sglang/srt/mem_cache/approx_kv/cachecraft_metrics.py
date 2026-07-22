"""Cache-Craft chunk contextualization and reuse-decision math.

Implements the Cache-Craft (arXiv:2502.15734, Agarwal et al.) chunk-cache
reusability metrics as new, additive logic layered on top of the frozen
`approx_kv` common core. Nothing here mutates existing common-core files;
this module only consumes their public types (`KVSegmentKey`, etc.) from
sibling modules.

Equation references are to the paper:

- Eq. (3) inter(Ci, Cj): cumulative attention weight from chunk Ci's tokens
  to chunk Cj's tokens (Ci precedes Cj).
- Eq. (4) intra(Ci): cumulative attention weight within chunk Ci from later
  tokens to earlier tokens.
- Eq. (6) beta: Prefix Overlap Score between a chunk-cache's original
  prefix (S_old) and a new prompt's prefix (S_new).
- Eq. (7) gamma: Order Penalty Score, the normalized Kendall's Tau distance
  between the common-chunk orderings of S_old and S_new.
- Eq. (8) beta': Adjusted Prefix Overlap Score, beta * (1 - gamma).
- Eq. (9)-(10) a(Ci), b(Ci): length-normalized, layer-averaged external
  ("inter") vs self ("intra") contextualization of a chunk.
- Eq. (11) CCI: Cache Context Impact, sigmoid(mean_a / mean_b).
- Eq. (12) CFO: Cache Fix Overhead, alpha * CCI * (1 - beta'), the fraction
  of a chunk's tokens that must be recomputed.
- Eq. (14) top-N token selection: the N = ceil(CFO * |Ci|) tokens with the
  highest external ("inter") attention are chosen for recomputation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ChunkContextProfile:
    """Real attention statistics captured when a chunk's KV was cached.

    All fields are derived from genuine per-layer attention weights (see
    `cachecraft_attention.py`); nothing here is a placeholder or synthetic
    label. `token_inter_scores` backs Eq. (14) token selection and
    `inter_attention_by_layer`/`intra_attention_by_layer` back Eq. (9)-(10).
    """

    chunk_id: str
    length: int
    old_prefix_order: tuple[str, ...]
    prefix_chunk_lengths: Mapping[str, int]
    inter_attention_by_layer: Mapping[str, tuple[float, ...]]
    intra_attention_by_layer: tuple[float, ...]
    token_inter_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id must be non-empty")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if not self.intra_attention_by_layer:
            raise ValueError("intra_attention_by_layer must not be empty")
        layer_num = len(self.intra_attention_by_layer)
        if len(self.token_inter_scores) != self.length:
            raise ValueError("token_inter_scores must have one entry per token")
        for chunk_id in self.old_prefix_order:
            if chunk_id not in self.prefix_chunk_lengths:
                raise ValueError(f"missing prefix_chunk_lengths entry for {chunk_id!r}")
            if chunk_id not in self.inter_attention_by_layer:
                raise ValueError(
                    f"missing inter_attention_by_layer entry for {chunk_id!r}"
                )
            if len(self.inter_attention_by_layer[chunk_id]) != layer_num:
                raise ValueError(
                    "inter_attention_by_layer entries must match layer_num"
                )
        if len(set(self.old_prefix_order)) != len(self.old_prefix_order):
            raise ValueError("old_prefix_order must not repeat a chunk id")

    @property
    def layer_num(self) -> int:
        return len(self.intra_attention_by_layer)


def layer_average(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def compute_a(profile: ChunkContextProfile) -> float:
    """Layer-averaged, length-normalized external contextualization, Eq. (9)-(10)."""
    layer_sums = [0.0] * profile.layer_num
    for prefix_id, per_layer in profile.inter_attention_by_layer.items():
        denom = profile.length * profile.prefix_chunk_lengths[prefix_id]
        if denom <= 0:
            raise ValueError("chunk lengths must be positive for normalization")
        for layer, value in enumerate(per_layer):
            layer_sums[layer] += value / denom
    return layer_average(layer_sums)


def compute_b(profile: ChunkContextProfile) -> float:
    """Layer-averaged, length-normalized self contextualization, Eq. (9)-(10)."""
    denom = profile.length * profile.length
    per_layer = [value / denom for value in profile.intra_attention_by_layer]
    return layer_average(per_layer)


def compute_cci(profile: ChunkContextProfile) -> float:
    """Cache Context Impact, Eq. (11): sigmoid(mean_a / mean_b)."""
    a = compute_a(profile)
    b = compute_b(profile)
    if b == 0.0:
        # A chunk with zero self-attention mass is fully defined by its
        # prefix (or by nothing at all if a is also zero); treat the ratio
        # as +inf when a > 0 (maximally contextualized) and as 0 otherwise.
        if a > 0.0:
            return 1.0
        return 0.5
    ratio = a / b
    return 1.0 / (1.0 + math.exp(-ratio))


def total_inter(profile: ChunkContextProfile, prefix_chunk_id: str) -> float:
    per_layer = profile.inter_attention_by_layer.get(prefix_chunk_id)
    if per_layer is None:
        return 0.0
    return sum(per_layer)


def compute_beta(
    profile: ChunkContextProfile,
    new_prefix_order: Sequence[str],
) -> float:
    """Prefix Overlap Score, Eq. (6)."""
    old_set = set(profile.old_prefix_order)
    new_set = set(new_prefix_order)
    common = old_set & new_set
    denominator = sum(total_inter(profile, chunk_id) for chunk_id in old_set)
    if denominator == 0.0:
        # No recorded old-prefix attention mass: the chunk was effectively
        # prefix-independent when cached, so there is nothing to lose by
        # reusing it in a different prefix.
        return 1.0
    numerator = sum(total_inter(profile, chunk_id) for chunk_id in common)
    return numerator / denominator


def kendall_tau_order_penalty(
    old_order: Sequence[str],
    new_order: Sequence[str],
    common_ids: Sequence[str] | None = None,
) -> float:
    """Order Penalty Score gamma, Eq. (7): normalized Kendall's Tau distance
    between the orderings of the chunks common to both S_old and S_new."""
    common = (
        set(common_ids) if common_ids is not None else (set(old_order) & set(new_order))
    )
    ordered_old = tuple(chunk_id for chunk_id in old_order if chunk_id in common)
    ordered_new = tuple(chunk_id for chunk_id in new_order if chunk_id in common)
    m = len(ordered_old)
    if m != len(ordered_new):
        raise ValueError("old/new orderings disagree on the common chunk set")
    if m < 2:
        return 0.0
    new_rank = {chunk_id: index for index, chunk_id in enumerate(ordered_new)}
    discordant = 0
    for i in range(m):
        for j in range(i + 1, m):
            if new_rank[ordered_old[i]] > new_rank[ordered_old[j]]:
                discordant += 1
    total_pairs = m * (m - 1) / 2
    return discordant / total_pairs


def adjusted_beta(beta: float, gamma: float) -> float:
    """Adjusted Prefix Overlap Score beta', Eq. (8)."""
    return beta * (1.0 - gamma)


def compute_cfo(cci: float, beta_prime: float, alpha: float = 1.0) -> float:
    """Cache Fix Overhead, Eq. (12), clamped to the paper's [0, 1] range
    (CFO = 1 means the entire chunk must be recomputed)."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    value = alpha * cci * (1.0 - beta_prime)
    return min(max(value, 0.0), 1.0)


def select_recompute_positions(
    profile: ChunkContextProfile,
    cfo: float,
) -> tuple[int, ...]:
    """Top-N token selection, Eq. (14): the N = ceil(CFO * |Ci|) tokens with
    the highest external ("inter") attention score are selected."""
    if not (0.0 <= cfo <= 1.0):
        raise ValueError("cfo must be in [0, 1]")
    n = min(profile.length, math.ceil(cfo * profile.length))
    if n <= 0:
        return ()
    ranked = sorted(
        range(profile.length),
        key=lambda position: (-profile.token_inter_scores[position], position),
    )
    return tuple(sorted(ranked[:n]))


class CacheCraftDecision(str, Enum):
    DIRECT_REUSE = "direct_reuse"
    PARTIAL_REPAIR = "partial_repair"
    FULL_RECOMPUTE = "full_recompute"


def decide(
    cfo: float,
    *,
    cache_hit: bool,
    full_recompute_threshold: float = 1.0,
) -> CacheCraftDecision:
    """Direct-reuse vs partial-repair vs full-recompute decision rule.

    - No usable chunk-cache entry (store miss) always forces a full
      recompute, matching the "No reuse" case in Fig. 11.
    - CFO == 0 means the chunk needs no recomputation at all (Case 1,
      "Direct Reuse").
    - CFO >= full_recompute_threshold means Cache-Craft would recompute the
      whole chunk anyway, so it is cheaper to fall back to a plain dense
      recompute of the chunk.
    - Otherwise a strict subset of tokens is recomputed (Case 3, "Selectively
      recompute").
    """
    if not (0.0 < full_recompute_threshold <= 1.0):
        raise ValueError("full_recompute_threshold must be in (0, 1]")
    if not cache_hit:
        return CacheCraftDecision.FULL_RECOMPUTE
    if not (0.0 <= cfo <= 1.0):
        raise ValueError("cfo must be in [0, 1]")
    if cfo <= 0.0:
        return CacheCraftDecision.DIRECT_REUSE
    if cfo >= full_recompute_threshold:
        return CacheCraftDecision.FULL_RECOMPUTE
    return CacheCraftDecision.PARTIAL_REPAIR
