from __future__ import annotations

from typing import Any

from .config import ApproxKVFeatureConfig
from .store import (
    ApproxKVSegmentStore,
    ReleaseBackend,
    ResidencyLoader,
)
from .transfer import KVTransferBackend, execute_reuse_plan
from .types import (
    KVReusePlan,
    KVSegmentHandle,
    KVSegmentKey,
    KVTransferStats,
    RecoveryMode,
    ResidencyTier,
)


class ApproxKVManager:
    def __init__(
        self,
        config: ApproxKVFeatureConfig | None = None,
        store: ApproxKVSegmentStore | None = None,
    ) -> None:
        self.config = config or ApproxKVFeatureConfig()
        self.store = store or ApproxKVSegmentStore()

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
        self,
        plan: KVReusePlan,
        backend: KVTransferBackend,
    ) -> KVTransferStats:
        if not self.config.core_enabled:
            backend.dense_prefill(
                target_start=0,
                length=len(plan.target_token_ids),
                reason="approx_kv_core_disabled",
            )
            return KVTransferStats(
                recovery_mode=RecoveryMode.DENSE,
                target_tokens=len(plan.target_token_ids),
                recomputed_tokens=len(plan.target_token_ids),
                fallback_reasons=["approx_kv_core_disabled"],
            )
        return execute_reuse_plan(
            plan=plan,
            store=self.store,
            backend=backend,
        )

    def ensure_resident(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
        loader: ResidencyLoader,
    ) -> KVSegmentHandle:
        if not self.config.core_enabled:
            raise RuntimeError("approximate KV core is disabled")
        return self.store.ensure_resident(handle, target_tier, loader)

    def reset(self) -> None:
        self.store.reset()
