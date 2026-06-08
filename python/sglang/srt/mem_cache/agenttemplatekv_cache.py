"""AgentTemplateKVCache: subclass of RadixCache that exposes the device-first
codebase prefetch path as a first-class public method.

Upstream ``RadixCache`` carries the underlying ATK data structures
(``anchor_kv_store``, ``AnchorKVEntry``, ``_agenttemplatekv_*`` helpers) and
the lossy-fuzzy-match path that consumes them. The ``AgentTemplateKVCache``
subclass exists so that:

1. The scheduler's "should I run the ATK prefetch path?" check is
   ``isinstance(self.tree_cache, AgentTemplateKVCache)`` instead of
   ``hasattr(self.tree_cache, "agenttemplatekv_prefetch_codebases")``.
   This is the upstreamable form: a non-ATK RadixCache is *not* a
   subclass of ``AgentTemplateKVCache``, so the scheduler never invokes
   the ATK path on stock SGLang.
2. New ATK-specific methods (prefetch, plan-store, future extensions)
   can be added to this subclass without polluting the public surface
   of ``RadixCache``.

The subclass overrides :meth:`prefetch_codebases` and delegates to the
private helpers on ``RadixCache`` (``_agenttemplatekv_protect_entry``,
``_agenttemplatekv_release_expired_prefetch_entries``). All ATK state
lives on the base class because the lossy-fuzzy-match path in
``_try_lossy_fuzzy_match`` and the anchor-store GC in ``_delete_leaf``
need to read/write it directly.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.radix_cache import RadixCache

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class AgentTemplateKVCache(RadixCache):
    """RadixCache with the AgentTemplateKV device-first prefetch path enabled.

    Use this class as a drop-in replacement for ``RadixCache`` when the
    scheduler is configured to use AgentTemplateKV. The base RadixCache
    class still implements the underlying anchor store, protect/release
    helpers, and lossy-fuzzy-match consumer; this subclass only adds
    the public ``prefetch_codebases`` entry point.
    """

    def prefetch_codebases(
        self,
        req: "Req",
        tokenizer=None,
        max_hints: int = 8,
    ) -> None:
        """Device-first codebase prefetch for AgentTemplateKV.

        Walks ``req.codebase_prefetch_hints``, looks up the exact-content
        anchor for each hint in ``self.anchor_kv_store``, pins the matched
        anchor for the next template step, and records a real device hit.
        This path is HiCache-independent.

        ``tokenizer`` is used to re-encode ``hint.text`` (when provided)
        and verify the cached entry's token sequence matches the hint
        text. Pass ``None`` to skip the text-verify step (the cached
        ``token_ids`` is then used as-is).
        """
        if not self._agenttemplatekv_enabled():
            return
        self._agenttemplatekv_release_expired_prefetch_entries(req)

        hints = getattr(req, "codebase_prefetch_hints", None) or []
        if not hints:
            return

        for hint in hints[:max_hints]:
            if not isinstance(hint, dict):
                continue
            content_signature = str(
                hint.get("content_signature")
                or hint.get("code_content_signature")
                or ""
            )
            if not content_signature:
                continue

            entries = []
            with self.anchor_kv_store_lock:
                entries = list(self.anchor_kv_store.get(content_signature, []))
            if not entries:
                req.agenttemplatekv_prefetch_miss_count += 1
                continue

            text = hint.get("text") or hint.get("code") or hint.get("content")
            text_token_ids = None
            if text and tokenizer is not None:
                try:
                    text_token_ids = tokenizer.encode(text, add_special_tokens=False)
                except Exception:
                    text_token_ids = None

            matched_entry = None
            for entry in entries:
                if entry.code_content_signature != content_signature:
                    continue
                if (
                    text_token_ids is not None
                    and list(entry.token_ids.tolist()) != list(text_token_ids)
                ):
                    continue
                matched_entry = entry
                break

            if matched_entry is None:
                req.agenttemplatekv_prefetch_miss_count += 1
                continue

            steps_to_use = int(hint.get("steps_to_use") or 1)
            locked_now = self._agenttemplatekv_protect_entry(
                matched_entry,
                req=req,
                steps_to_use=max(1, steps_to_use),
            )
            matched_tokens = len(matched_entry.token_ids)
            req.codebase_prefetch_matched_tokens += matched_tokens
            req.codebase_prefetch_success_count += 1
            req.codebase_prefetch_device_hit_count += 1
            req.agenttemplatekv_prefetch_hit_count += 1
            req.agenttemplatekv_prefetch_protected_tokens += matched_tokens
            if locked_now:
                req.agenttemplatekv_prefetch_newly_protected_tokens += matched_tokens


__all__ = ["AgentTemplateKVCache"]
