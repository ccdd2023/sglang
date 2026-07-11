"""Phase A1 (2026-07-11): Tool-output + system-prompt KV cache prototype.

External validation:
  - TokenCake (arXiv:2510.18586): >=47% latency reduction on coding workloads
  - Anthropic cache_control: ephemeral: ~97% of to-B cache hits from
    single-turn system-prompt sharing

This module is the **measurement-first** instrumentation layer. It
detects "cacheable buckets" in incoming chat-completion requests and
emits hit-rate telemetry without yet wiring the actual KV-pool sharing.

The 4 buckets we recognize:
  1. System-prompt bucket (role=system, content hash)
  2. Tool-definition bucket (hash of json.dumps(tools))
  3. Tool-call bucket (per assistant message, hash of name+args)
  4. Tool-output bucket (per tool message, hash of content)

For each bucket we track:
  - first_seen_ts (process-clock seconds at first sighting)
  - hit_count (number of times this bucket was re-seen after first_seen)
  - in_process (whether the bucket has been seen in the running process)

This gives us the **hit-rate distribution** (the most important metric
for justifying the KV-pool wiring cost) without committing to the pool
itself. If hit rate is ~0% across 100 requests, we don't bother building
the pool. If hit rate is ~50%, we build it.

Default OFF (SGLANG_TOOL_OUTPUT_CACHE=0). No code-path mutation when
disabled; buckets dict is empty and telemetry write sites are no-ops.

Telemetry emission: 4 counters exposed on the cache object:
  - tool_output_cache_hit_count  (re-seen bucket hits)
  - tool_output_cache_miss_count (new buckets first seen)
  - tool_output_cache_total_buckets (unique buckets observed)
  - tool_output_cache_in_process_buckets (currently held)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple


def _stable_hash(data: Any) -> str:
    """SHA256 of a JSON-serializable object, hex-digest[:32].

    Used as the bucket key. Output is stable across processes (no
    timestamp, no memory address). 32 chars keeps the dict small.
    """
    if isinstance(data, str):
        b = data.encode("utf-8")
    elif isinstance(data, bytes):
        b = data
    else:
        b = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:32]


@dataclass
class CacheBucket:
    """One observation about a stable hash bucket."""
    bucket_key: str
    first_seen_ts: float = field(default_factory=time.monotonic)
    hit_count: int = 0
    bucket_size_chars: int = 0  # rough size proxy


class ToolOutputCache:
    """Process-scope (per RadixCache) bucket registry.

    Thread-safety: not thread-safe. SGLang's scheduler runs single-threaded
    per RadixCache instance so the lack of locking is intentional.

    Lifecycle: owned by RadixCache; constructed lazily at first observation.
    """

    def __init__(self, max_buckets: int = 4096) -> None:
        self._max_buckets = max_buckets
        self._buckets: Dict[Tuple[str, str], CacheBucket] = {}
        self.hit_count: int = 0
        self.miss_count: int = 0

    @property
    def enabled(self) -> bool:
        return os.environ.get("SGLANG_TOOL_OUTPUT_CACHE", "0") == "1"

    @property
    def total_buckets(self) -> int:
        return len(self._buckets)

    @property
    def in_process_buckets(self) -> int:
        # Without LRU eviction, all observed buckets are in-process.
        # (Future: track per-bucket LRU.)
        return len(self._buckets)

    def observe_system_prompt(self, content: str) -> bool:
        """Record a system-prompt bucket. Return True if cache HIT, else MISS."""
        return self._observe(("system", _stable_hash(content)), size_chars=len(content))

    def observe_tool_definitions(self, tools: Iterable[dict]) -> bool:
        """Record a tool-definition bucket. The tools arg is iterable of dicts."""
        try:
            stable = json.dumps(list(tools), sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return False
        return self._observe(("tools_def", _stable_hash(stable)),
                              size_chars=len(stable))

    def observe_tool_call(self, name: str, args: Any) -> bool:
        """Record a (name, args) bucket for an assistant tool_call."""
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (TypeError, ValueError):
                pass
        return self._observe(("tool_call", _stable_hash((name, args))),
                              size_chars=len(str(args)))

    def observe_tool_output(self, tool_call_id: Optional[str], content: str) -> bool:
        """Record a tool-output bucket (role=tool message content).

        tool_call_id is optional: we hash by content. When tool_call_id is
        present, two outputs from the same call could potentially differ in
        whitespace; in practice the content is the cache signal.
        """
        bucket_key = _stable_hash(content)
        # Distinguish by tool_call_id when present (more granular) but
        # observed unique-by-content so two tool messages with same content
        # but different tool_call_ids hash to the same bucket.
        # This is intentional: same content -> same KV -> cache hit.
        del tool_call_id  # not used in hashing; kept for caller clarity
        return self._observe(("tool_output", bucket_key),
                              size_chars=len(content))

    def _observe(self, key: Tuple[str, str], size_chars: int) -> bool:
        if not self.enabled:
            # When disabled, return False (no hit signal) and do not retain.
            return False
        bucket = self._buckets.get(key)
        if bucket is None:
            # Cap retained buckets to keep memory bounded.
            if len(self._buckets) >= self._max_buckets:
                return False
            bucket = CacheBucket(bucket_key=key[1], bucket_size_chars=size_chars)
            self._buckets[key] = bucket
            self.miss_count += 1
            return False
        else:
            bucket.hit_count += 1
            self.hit_count += 1
            return True

    def snapshot(self) -> Dict[str, Any]:
        """Process-wide snapshot of cache state. Cheap to call."""
        return {
            "enabled": self.enabled,
            "total_buckets": self.total_buckets,
            "in_process_buckets": self.in_process_buckets,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "max_buckets": self._max_buckets,
        }
