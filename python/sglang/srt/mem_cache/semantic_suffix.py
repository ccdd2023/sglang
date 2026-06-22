"""Semantic suffix-copy length decider.

Replaces the hand-tuned per-case `max_suffix_copy_len` cap with a content-derived
copy length. For each candidate anchor entry, we compute 64-token chunk
embeddings at store time and compare to the request's chunk embeddings at the
candidate position. The largest prefix where every chunk has cosine >=
`SGLANG_SEMANTIC_SUFFIX_MIN_COSINE` becomes the effective copy length.

This module is read-only with respect to the runtime cache; it only adds two
helpers used by `radix_cache.py`:

- `compute_chunk_embeddings(token_ids, tokenizer, model, chunk_size)` -> Tensor[N, D]
- `cosine_profile(req_chunks, entry_chunks, min_cosine)` -> int (length in tokens)

If `sentence-transformers` is unavailable or model load fails, the helpers
fall back to a "no semantic data" sentinel and the runtime uses the legacy
cap-based length.

Environment variables:
- SGLANG_SEMANTIC_SUFFIX_ENABLED (default 1): master switch.
- SGLANG_SEMANTIC_SUFFIX_CHUNK_TOKENS (default 64): chunk size in tokens.
- SGLANG_SEMANTIC_SUFFIX_MIN_COSINE (default 0.70): chunk cosine floor.
- SGLANG_SEMANTIC_SUFFIX_MIN_CHUNKS (default 1): refuse copy of <min_chunks chunks.
- SGLANG_SEMANTIC_SUFFIX_MODEL (default sentence-transformers/all-MiniLM-L6-v2).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment knobs (read lazily so tests can monkey-patch).
# ---------------------------------------------------------------------------

def is_enabled() -> bool:
    """Master switch. Default ON per the v9 mainline plan."""
    return os.environ.get("SGLANG_SEMANTIC_SUFFIX_ENABLED", "1") == "1"


def chunk_tokens() -> int:
    return int(os.environ.get("SGLANG_SEMANTIC_SUFFIX_CHUNK_TOKENS", "64"))


def min_cosine() -> float:
    return float(os.environ.get("SGLANG_SEMANTIC_SUFFIX_MIN_COSINE", "0.70"))


def min_chunks() -> int:
    return int(os.environ.get("SGLANG_SEMANTIC_SUFFIX_MIN_CHUNKS", "1"))


def model_name() -> str:
    return os.environ.get(
        "SGLANG_SEMANTIC_SUFFIX_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )


def cache_dir() -> Optional[str]:
    d = os.environ.get("SGLANG_SEMANTIC_SUFFIX_CACHE_DIR", "")
    return d or None


# ---------------------------------------------------------------------------
# Embedder (lazy-loaded singleton).
# ---------------------------------------------------------------------------


@dataclass
class _Embedder:
    tokenizer: object  # transformers.PreTrainedTokenizerBase
    model: object  # transformers.PreTrainedModel
    dim: int
    device: str


_EMBEDDER: Optional[_Embedder] = None
_EMBEDDER_LOAD_FAILED = False


def load_embedder(force: bool = False) -> Optional[_Embedder]:
    """Load sentence-transformers/all-MiniLM-L6-v2 once per process.

    Returns None if load fails (FFmpeg / torchcodec / network errors). When
    None, callers must treat the semantic layer as unavailable and fall back
    to the legacy cap-based copy length.
    """
    global _EMBEDDER, _EMBEDDER_LOAD_FAILED
    if _EMBEDDER is not None:
        return _EMBEDDER
    if _EMBEDDER_LOAD_FAILED and not force:
        return None
    try:
        # Bypass sentence-transformers library (FFmpeg/torchcodec dep issue
        # on this host) and load the model directly via transformers.
        from transformers import AutoModel, AutoTokenizer

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        cache_kwargs = {"cache_dir": cache_dir()} if cache_dir() else {}
        tokenizer = AutoTokenizer.from_pretrained(model_name(), **cache_kwargs)
        model = AutoModel.from_pretrained(model_name(), **cache_kwargs)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        dim = int(model.config.hidden_size)
        _EMBEDDER = _Embedder(tokenizer=tokenizer, model=model, dim=dim, device=device)
        logger.info(
            "[semantic_suffix] loaded %s on %s (dim=%d)",
            model_name(), device, dim,
        )
        return _EMBEDDER
    except Exception as e:  # pragma: no cover - dependency / network errors
        _EMBEDDER_LOAD_FAILED = True
        logger.warning(
            "[semantic_suffix] embedder load failed; semantic suffix disabled: %s: %s",
            type(e).__name__, e,
        )
        return None


# ---------------------------------------------------------------------------
# Chunk embedding computation.
# ---------------------------------------------------------------------------


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings, masking out padding."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def _embed_texts(texts: list[str], emb: _Embedder) -> torch.Tensor:
    """Run MiniLM on a batch of texts. Returns normalized embeddings [N, D]."""
    enc = emb.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=emb.model.config.max_position_embeddings,
        return_tensors="pt",
    )
    enc = {k: v.to(emb.device) for k, v in enc.items()}
    out = emb.model(**enc)
    pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
    pooled = F.normalize(pooled, p=2, dim=1)
    return pooled.cpu()


def compute_chunk_embeddings(
    token_ids: torch.Tensor,
    tokenizer,  # the LLM tokenizer (different from the embedder tokenizer)
    emb: _Embedder,
    chunk_size: int | None = None,
) -> torch.Tensor | None:
    """Decode token_ids in `chunk_size`-token windows and embed each window.

    Returns Tensor[N, D] where N = ceil(len(token_ids) / chunk_size).
    Each row is the L2-normalized mean-pooled embedding of one window's text.
    Returns None if token_ids is too short to produce even one chunk.
    """
    if chunk_size is None:
        chunk_size = chunk_tokens()
    n = int(len(token_ids))
    if n <= 0:
        return None
    n_chunks = max(1, (n + chunk_size - 1) // chunk_size)
    if n_chunks < min_chunks():
        # Below the floor; caller should treat this as "no semantic data".
        return None

    texts: list[str] = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n)
        # The LLM tokenizer and the embedder tokenizer may differ. Decode with
        # the LLM tokenizer (which is what produced these token_ids) so the
        # resulting text is faithful to the anchor content.
        try:
            text = tokenizer.decode(
                token_ids[start:end].detach().cpu().tolist(),
                skip_special_tokens=True,
            )
        except Exception:
            text = ""
        if not text:
            text = " "
        texts.append(text)
    return _embed_texts(texts, emb)


def cosine_profile(
    req_chunks: torch.Tensor,
    entry_chunks: torch.Tensor,
    min_cosine_threshold: float | None = None,
    chunk_token_size: int | None = None,
    min_chunk_count: int | None = None,
) -> int:
    """Largest prefix length (in tokens) where every chunk has cosine >= threshold.

    Inputs are L2-normalized chunk embeddings [N_req, D] and [N_entry, D].
    Returns the token count of the longest aligned prefix that satisfies
    the cosine floor for every included chunk.

    - If `entry_chunks` is shorter than `req_chunks`, the answer is bounded by
      `len(entry_chunks) * chunk_token_size`.
    - If fewer than `min_chunk_count` chunks qualify, returns 0.
    - If the two chunk counts are equal but a chunk fails the cosine test at
      index k, returns `k * chunk_token_size` (i.e. the partial prefix).
    """
    if req_chunks is None or entry_chunks is None:
        return 0
    if req_chunks.numel() == 0 or entry_chunks.numel() == 0:
        return 0
    if min_cosine_threshold is None:
        min_cosine_threshold = min_cosine()
    if chunk_token_size is None:
        chunk_token_size = chunk_tokens()
    if min_chunk_count is None:
        min_chunk_count = min_chunks()

    n_common = min(int(req_chunks.shape[0]), int(entry_chunks.shape[0]))
    if n_common == 0:
        return 0

    # Cosine similarity per aligned chunk pair. Both inputs are already
    # L2-normalized, so a dot product suffices.
    sims = (req_chunks[:n_common] * entry_chunks[:n_common]).sum(dim=-1)
    # Find the longest prefix where every cosine >= threshold.
    n_keep = 0
    for i in range(n_common):
        if float(sims[i]) >= min_cosine_threshold:
            n_keep = i + 1
        else:
            break

    if n_keep < min_chunk_count:
        return 0
    return n_keep * chunk_token_size


# ---------------------------------------------------------------------------
# Convenience: compute both sides for an anchor and a request position.
# ---------------------------------------------------------------------------


def entry_chunks_for(
    token_ids: torch.Tensor,
    llm_tokenizer,
) -> torch.Tensor | None:
    """Compute and cache chunk embeddings for an anchor entry. Returns None
    if the embedder is unavailable or the entry is too short."""
    if not is_enabled():
        return None
    emb = load_embedder()
    if emb is None:
        return None
    return compute_chunk_embeddings(token_ids, llm_tokenizer, emb)


def request_chunks_for(
    token_ids: torch.Tensor,
    llm_tokenizer,
) -> torch.Tensor | None:
    """Compute chunk embeddings for the request at the candidate anchor's
    position. Same shape contract as `entry_chunks_for`."""
    return entry_chunks_for(token_ids, llm_tokenizer)


def embed_single_text(
    text: str,
    emb: Optional["_Embedder"] = None,
) -> Optional[torch.Tensor]:
    """Return one L2-normalized [D] embedding for a single text string.

    Used by the per-placeholder k-NN pool to compute the corpus embedding
    for a placeholder span at write-back time and the query embedding at
    read time.  Returns None if disabled or load failed (graceful degrade).
    """
    if not is_enabled():
        return None
    if emb is None:
        emb = load_embedder()
    if emb is None:
        return None
    out = _embed_texts([text or " "], emb)
    if out is None or out.numel() == 0:
        return None
    return out[0]  # [D] L2-normalized


# LRU cache for query embeddings.  Keyed by bounded text content.
# Saves ~24ms (MiniLM forward) per repeat lookup in the k-NN body.
from collections import OrderedDict
_EMBED_CACHE: "OrderedDict[str, torch.Tensor]" = OrderedDict()
_EMBED_CACHE_MAX = int(os.environ.get("SGLANG_SEMANTIC_SUFFIX_EMBED_CACHE_MAX", "1024"))
_EMBED_CACHE_KEY_LEN = int(os.environ.get("SGLANG_SEMANTIC_SUFFIX_EMBED_CACHE_KEY_LEN", "2048"))


def embed_single_text_cached(
    text: str,
    emb: Optional["_Embedder"] = None,
) -> Optional[torch.Tensor]:
    """Like embed_single_text but with an LRU cache by text content.
    Saves ~24ms per repeat lookup in the k-NN body.  Cache key is bounded
    by _EMBED_CACHE_KEY_LEN to bound memory.  Disabled when env var
    SGLANG_SEMANTIC_SUFFIX_EMBED_CACHE_ENABLED=0.
    """
    if os.environ.get("SGLANG_SEMANTIC_SUFFIX_EMBED_CACHE_ENABLED", "1") != "1":
        return embed_single_text(text, emb=emb)
    key = (text or " ")[:_EMBED_CACHE_KEY_LEN]
    cached = _EMBED_CACHE.get(key)
    if cached is not None:
        _EMBED_CACHE.move_to_end(key)
        return cached
    out = embed_single_text(key, emb=emb)
    if out is not None and len(_EMBED_CACHE) < _EMBED_CACHE_MAX:
        _EMBED_CACHE[key] = out
    elif out is not None:
        # Evict oldest to make room
        _EMBED_CACHE.popitem(last=False)
        _EMBED_CACHE[key] = out
    return out


def reset_embed_cache_for_tests() -> None:
    """Clear embedding LRU cache; for unit tests."""
    _EMBED_CACHE.clear()


def reset_for_tests() -> None:
    """Clear cached embedder; for unit tests."""
    global _EMBEDDER, _EMBEDDER_LOAD_FAILED
    _EMBEDDER = None
    _EMBEDDER_LOAD_FAILED = False
