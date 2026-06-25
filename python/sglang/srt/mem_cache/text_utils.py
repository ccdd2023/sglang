"""Lightweight text utilities shared by radix_cache and benchmark.

Kept dependency-free (no torch, no transformers) so it can be imported
freely from runtime code without creating an import cycle.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, List, Tuple


def token_f1(a: str, b: str) -> float:
    """Whitespace-tokenized F1 between two strings.

    Returns 1.0 for two empty strings, 0.0 if exactly one is empty.
    Otherwise computes precision/recall over the multiset intersection
    of whitespace-split tokens (Counter-based, so repetition counts).
    Returns 0.0 when overlap is zero.
    """
    aa = a.split()
    bb = b.split()
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    ca, cb = Counter(aa), Counter(bb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(aa)
    recall = overlap / len(bb)
    return 2 * precision * recall / (precision + recall)


def token_bounds_for_text(
    tokenizer: Any,
    full_text: str,
    segment_text: str,
    char_start: int = 0,
) -> Tuple[int, int, int]:
    """Locate ``segment_text`` within ``full_text`` after ``char_start``
    and return ``(start_token, end_token, char_end)``.

    Both endpoints are token counts (the LLM tokenizer's encoding of the
    prefix up to the boundary).  Raises ``ValueError`` if the segment
    text cannot be found.
    """
    char_pos = full_text.find(segment_text, char_start)
    if char_pos < 0:
        raise ValueError("segment text not found in prompt")
    start = len(tokenizer.encode(full_text[:char_pos], add_special_tokens=False))
    end = len(tokenizer.encode(full_text[: char_pos + len(segment_text)], add_special_tokens=False))
    return start, end, char_pos + len(segment_text)


def all_slot_token_bounds(
    tokenizer: Any,
    full_text: str,
    slot_texts: List[str],
) -> List[Tuple[int, int]]:
    """Apply :func:`token_bounds_for_text` across a list of slot texts,
    advancing the cursor each iteration.  Each entry returned is
    ``(start_token, end_token)``.  Slots whose text is not found are
    silently skipped (caller decides what to do).
    """
    out: List[Tuple[int, int]] = []
    cursor = 0
    for seg_text in slot_texts:
        if not seg_text:
            continue
        try:
            start, end, cursor = token_bounds_for_text(
                tokenizer, full_text, seg_text, char_start=cursor,
            )
            out.append((start, end))
        except ValueError:
            continue
    return out
