"""Scheduling-side prefetch coordinator for KVCOMM segments."""

from sglang.srt.mem_cache.kvcomm_prefetch.coordinator import (
    KVPrefetchCoordinator,
    PrefetchResult,
)

__all__ = ["KVPrefetchCoordinator", "PrefetchResult"]
