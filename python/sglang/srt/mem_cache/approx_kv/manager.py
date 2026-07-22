from __future__ import annotations

import threading
from typing import Any

from .async_transfer import ApproxKVPrefetchTicket
from .config import ApproxKVFeatureConfig
from .plugins import RecoveryPlugin, RecoveryPluginRegistry
from .store import (
    AsyncResidencyLoader,
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
        metrics_collector: object | None = None,
    ) -> None:
        self.config = config or ApproxKVFeatureConfig()
        self.store = store or ApproxKVSegmentStore()
        self.plugins = RecoveryPluginRegistry()
        self.metrics_collector = metrics_collector
        self.residency_backend: Any | None = None
        self.rope_config: Any | None = None
        self.model_runner: Any | None = None
        self.epic_forward_batch_factory: Any | None = None
        self._async_loader: AsyncResidencyLoader | None = None
        self._tickets: dict[str, ApproxKVPrefetchTicket] = {}
        self._ticket_lock = threading.Lock()
        if self.config.epic_enabled:
            from .epic_plugin import EPICLeadingKPlugin

            self.register_plugin(
                EPICLeadingKPlugin(
                    k=self.config.epic_k,
                    attention_sink=self.config.epic_attention_sink,
                )
            )

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
            stats = KVTransferStats(
                recovery_mode=RecoveryMode.DENSE,
                target_tokens=len(plan.target_token_ids),
                recomputed_tokens=len(plan.target_token_ids),
                fallback_reasons=["approx_kv_core_disabled"],
            )
            self._record_transfer_stats(stats)
            return stats
        stats = execute_reuse_plan(
            plan=plan,
            store=self.store,
            backend=backend,
        )
        self._record_transfer_stats(stats)
        return stats

    def ensure_resident(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
        loader: ResidencyLoader,
    ) -> KVSegmentHandle:
        if not self.config.core_enabled:
            raise RuntimeError("approximate KV core is disabled")
        source_tier = handle.residency
        result = self.store.ensure_resident(handle, target_tier, loader)
        if source_tier != ResidencyTier.DEVICE and target_tier == ResidencyTier.DEVICE:
            self._record_h2d(result.key.token_count)
        return result

    def bind_async_loader(self, loader: AsyncResidencyLoader) -> None:
        if not self.config.host_residency_enabled:
            raise RuntimeError("approximate KV host residency is disabled")
        self._async_loader = loader

    def bind_residency_backend(self, backend: Any) -> None:
        if not self.config.host_residency_enabled:
            raise RuntimeError("approximate KV host residency is disabled")
        self.residency_backend = backend
        if hasattr(backend, "begin_load"):
            self.bind_async_loader(backend)

    def bind_rope_config(self, rope_config: Any) -> None:
        self.rope_config = rope_config

    def bind_model_runner(self, model_runner: Any) -> None:
        """Bind the live model runner EPIC needs to recompute leading-k.

        This is analogous to ``bind_rope_config``/``bind_residency_backend``:
        binding is optional and only required by callers that actually need
        genuine per-layer recompute (EPIC's ``epic_runtime.py``). Nothing in
        the R0 raw-copy path reads this attribute.
        """
        self.model_runner = model_runner

    def bind_epic_forward_batch_factory(self, factory: Any) -> None:
        """Bind the production builder for EPIC's leading-k forward batch."""
        self.epic_forward_batch_factory = factory

    def export_to_host(self, device_ref: Any):
        if not self.config.host_residency_enabled:
            raise RuntimeError("approximate KV host residency is disabled")
        backend = self.residency_backend
        if backend is None:
            raise RuntimeError("no approximate KV residency backend is bound")
        return backend.export_to_host(device_ref)

    def ensure_device(self, handle: KVSegmentHandle) -> KVSegmentHandle:
        if handle.residency == ResidencyTier.DEVICE:
            return handle
        backend = self.residency_backend
        if backend is None:
            raise RuntimeError("no approximate KV residency backend is bound")
        if self.config.async_prefetch_enabled and hasattr(backend, "begin_load"):
            return self.begin_prefetch(handle).wait()
        return self.ensure_resident(handle, ResidencyTier.DEVICE, backend)

    def begin_prefetch(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier = ResidencyTier.DEVICE,
    ) -> ApproxKVPrefetchTicket:
        if not self.config.async_prefetch_enabled:
            raise RuntimeError("approximate KV async prefetch is disabled")
        loader = self._async_loader
        if loader is None:
            raise RuntimeError("no approximate KV async loader is bound")
        transfer = loader.begin_load(handle, target_tier)
        ticket = ApproxKVPrefetchTicket(
            store=self.store,
            handle=handle,
            target_tier=target_tier,
            transfer=transfer,
            on_finish=self._forget_ticket,
            on_complete=self._record_async_load,
        )
        with self._ticket_lock:
            self._tickets[ticket.ticket_id] = ticket
        return ticket

    def register_plugin(self, plugin: RecoveryPlugin) -> None:
        self.plugins.register(plugin)

    def record_request(self, operation: str, outcome: str) -> None:
        collector = self.metrics_collector
        if collector is None:
            return
        callback = getattr(collector, "increment_approx_kv_request", None)
        if callback is not None:
            callback(operation, outcome)

    def record_fallback(self, reason: str, num_tokens: int) -> None:
        collector = self.metrics_collector
        if collector is None:
            return
        callback = getattr(collector, "increment_approx_kv_fallback", None)
        if callback is not None:
            callback(reason, num_tokens)

    def record_host_export(self, num_tokens: int, num_bytes: int) -> None:
        collector = self.metrics_collector
        if collector is None:
            return
        callback = getattr(collector, "increment_approx_kv_host_export", None)
        if callback is not None:
            callback(num_tokens, num_bytes)

    def record_epic_layer_recompute(
        self,
        *,
        layers_recomputed: int,
        leading_k_tokens: int,
        genuinely_layerwise: bool,
    ) -> None:
        collector = self.metrics_collector
        if collector is None:
            return
        callback = getattr(collector, "record_approx_kv_epic_layer_recompute", None)
        if callback is not None:
            callback(
                layers_recomputed,
                leading_k_tokens,
                genuinely_layerwise,
            )

    @property
    def active_ticket_count(self) -> int:
        with self._ticket_lock:
            return len(self._tickets)

    def reset(self) -> None:
        with self._ticket_lock:
            tickets = tuple(self._tickets.values())
        for ticket in tickets:
            ticket.cancel()
        self.store.reset()

    def _forget_ticket(self, ticket_id: str) -> None:
        with self._ticket_lock:
            self._tickets.pop(ticket_id, None)

    def _record_transfer_stats(self, stats: KVTransferStats) -> None:
        collector = self.metrics_collector
        if collector is None:
            return
        callback = getattr(collector, "record_approx_kv_transfer", None)
        if callback is not None:
            callback(stats)

    def _record_h2d(self, num_tokens: int) -> None:
        collector = self.metrics_collector
        if collector is None:
            return
        callback = getattr(collector, "increment_approx_kv_h2d_tokens", None)
        if callback is not None:
            callback(num_tokens)

    def _record_async_load(self, result) -> None:
        self._record_h2d(result.num_tokens)
        collector = self.metrics_collector
        if collector is None:
            return
        callback = getattr(collector, "observe_approx_kv_h2d", None)
        if callback is not None:
            callback(
                result.num_tokens,
                result.bytes_transferred,
                result.duration_ms,
            )
