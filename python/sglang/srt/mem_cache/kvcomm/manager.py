from __future__ import annotations

from typing import Any

from sglang.srt.mem_cache.kvcomm.config import KVCommFeatureConfig
from sglang.srt.mem_cache.kvcomm.store import (
    KVSegmentStore,
    ReleaseBackend,
    ResidencyLoader,
)
from sglang.srt.mem_cache.kvcomm.transfer import (
    KVTransferBackend,
    execute_reuse_plan,
)
from sglang.srt.mem_cache.kvcomm.types import (
    KVReusePlan,
    KVSegmentHandle,
    KVSegmentKey,
    KVTransferStats,
    ResidencyTier,
)


class KVCommManager:
    """Policy-neutral facade attached to a cache implementation."""

    def __init__(
        self,
        config: KVCommFeatureConfig | None = None,
        store: KVSegmentStore | None = None,
    ) -> None:
        self.config = config or KVCommFeatureConfig()
        self.store = store or KVSegmentStore()

    def register_segment(
        self,
        *,
        key: KVSegmentKey,
        token_ids: tuple[int, ...] | list[int],
        source_start: int,
        residency: ResidencyTier,
        backend_ref: Any,
        release_backend: ReleaseBackend | None = None,
    ) -> KVSegmentHandle | None:
        if not self.config.core_enabled:
            return None
        return self.store.register(
            key=key,
            token_ids=token_ids,
            source_start=source_start,
            residency=residency,
            backend_ref=backend_ref,
            release_backend=release_backend,
        )

    def execute(
        self, plan: KVReusePlan, backend: KVTransferBackend
    ) -> KVTransferStats:
        if not self.config.core_enabled:
            backend.dense_prefill(
                target_start=0,
                length=len(plan.target_token_ids),
                reason="kvcomm_core_disabled",
            )
            return KVTransferStats(
                target_tokens=len(plan.target_token_ids),
                recomputed_tokens=len(plan.target_token_ids),
                fallback_reasons=["kvcomm_core_disabled"],
            )
        return execute_reuse_plan(plan=plan, store=self.store, backend=backend)

    def ensure_resident(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
        loader: ResidencyLoader,
    ) -> KVSegmentHandle:
        if not self.config.core_enabled:
            raise RuntimeError("KVCOMM core is disabled")
        return self.store.ensure_resident(handle, target_tier, loader)

    def reset(self) -> None:
        self.store.reset()
