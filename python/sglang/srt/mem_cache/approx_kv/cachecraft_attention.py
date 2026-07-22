"""Real per-layer attention-weight capture for Cache-Craft chunk profiles.

This performs genuine dense causal self-attention (`softmax(QK^T / sqrt(d))`
with a lower-triangular mask) over real query/key tensors and slices the
result into the `inter`/`intra` sums defined in Cache-Craft Eq. (3)-(4). It
is a faithful reference computation, not a placeholder: given real Q/K
tensors it produces exactly the attention weights a production model would
use for those tokens.

Capability gate: SGLang's production attention backends (FlashInfer /
FlashAttention / Triton fused kernels) never materialize the full
`(seq_len, seq_len)` attention-probability matrix, so this capture path only
applies where real per-layer Q/K tensors (or an eager/reference attention
fallback) are available -- e.g. a dedicated registration-time forward using
an eager attention implementation. There is no way to recover exact
attention weights from the fused kernels used on the hot request path
without such an eager fallback; that gap is the documented Cache-Craft
"real profile capture" server-integration blocker (see HANDOFF/PROJECT
tracking for this worktree).
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

from .cachecraft_metrics import ChunkContextProfile


def causal_attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Real dense causal self-attention weights for one layer.

    `query`/`key` are `(seq_len, head_dim)` (already RoPE-applied, as they
    would be inside a real attention module). Returns `(seq_len, seq_len)`
    row-normalized weights where `weights[l, k]` is the probability that
    query position `l` places on key position `k` (`k <= l`).
    """
    if query.dim() != 2 or key.dim() != 2:
        raise ValueError("query/key must be 2-D (seq_len, head_dim) tensors")
    if query.shape != key.shape:
        raise ValueError("query/key must have matching shape for reference capture")
    seq_len, head_dim = query.shape
    if seq_len == 0:
        raise ValueError("query/key must have at least one token")
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    scores = (query.float() @ key.float().transpose(-1, -2)) * scale
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    return torch.softmax(scores, dim=-1)


def capture_chunk_profile(
    *,
    chunk_id: str,
    weights_per_layer: Sequence[torch.Tensor],
    chunk_spans: Mapping[str, tuple[int, int]],
    old_prefix_order: tuple[str, ...],
) -> ChunkContextProfile:
    """Build a `ChunkContextProfile` from real per-layer attention weights.

    `weights_per_layer[l]` must be the full `(total_seq_len, total_seq_len)`
    causal attention-weight matrix (e.g. from `causal_attention_weights`) for
    the concatenated `old_prefix_order + [chunk_id]` sequence at layer `l`.
    `chunk_spans` maps every chunk id in that sequence (including `chunk_id`
    itself) to its `(start, end)` token-position span within that sequence.
    """
    if chunk_id not in chunk_spans:
        raise ValueError("chunk_spans must include the target chunk_id")
    start, end = chunk_spans[chunk_id]
    length = end - start
    if length <= 0:
        raise ValueError("chunk span must be non-empty")
    layer_num = len(weights_per_layer)
    if layer_num == 0:
        raise ValueError("weights_per_layer must not be empty")

    intra_by_layer: list[float] = []
    inter_by_layer: dict[str, list[float]] = {
        prefix_id: [0.0] * layer_num for prefix_id in old_prefix_order
    }
    token_scores = [0.0] * length

    for layer, weights in enumerate(weights_per_layer):
        if weights.shape[0] != weights.shape[1]:
            raise ValueError("attention weights must be square")
        chunk_block = weights[start:end, start:end]
        intra_by_layer.append(float(chunk_block.tril(diagonal=-1).sum()))

        for prefix_id in old_prefix_order:
            prefix_start, prefix_end = chunk_spans[prefix_id]
            cross_block = weights[start:end, prefix_start:prefix_end]
            inter_by_layer[prefix_id][layer] = float(cross_block.sum())

        if start > 0:
            external = weights[start:end, 0:start].sum(dim=1)
            for position in range(length):
                token_scores[position] += float(external[position])

    return ChunkContextProfile(
        chunk_id=chunk_id,
        length=length,
        old_prefix_order=old_prefix_order,
        prefix_chunk_lengths={
            prefix_id: chunk_spans[prefix_id][1] - chunk_spans[prefix_id][0]
            for prefix_id in old_prefix_order
        },
        inter_attention_by_layer={
            prefix_id: tuple(values) for prefix_id, values in inter_by_layer.items()
        },
        intra_attention_by_layer=tuple(intra_by_layer),
        token_inter_scores=tuple(token_scores),
    )
