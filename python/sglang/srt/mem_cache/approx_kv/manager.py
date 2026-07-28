from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any

from sglang.srt.mem_cache.cross_store.allocator import (
    AppliedAction,
    CrossStoreResource,
)

from .async_transfer import ApproxKVPrefetchTicket
from .config import ApproxKVFeatureConfig
from .plugins import RecoveryPlugin, RecoveryPluginRegistry
from .store import (
    ApproxKVLease,
    ApproxKVSegmentStore,
    ApproxKVStoreCapacityError,
    AsyncResidencyLoader,
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
        self.store = store or ApproxKVSegmentStore(
            max_host_bytes=(
                self.config.cross_store_host_budget_bytes
                if self.config.cross_store_enabled
                and self.config.host_residency_enabled
                else None
            ),
            bytes_per_token=self.config.cross_store_bytes_per_token,
        )
        if (
            store is not None
            and self.config.cross_store_enabled
            and self.config.host_residency_enabled
        ):
            self.store.configure_byte_limits(
                max_host_bytes=self.config.cross_store_host_budget_bytes,
            )
        if (
            self.config.cross_store_enabled
            and self.store.bytes_per_token != self.config.cross_store_bytes_per_token
        ):
            raise ValueError(
                "approximate store and cross-store coordinator must use "
                "the same bytes_per_token"
            )
        self.plugins = RecoveryPluginRegistry()
        self.metrics_collector = metrics_collector
        self.residency_backend: Any | None = None
        self.rope_config: Any | None = None
        self.model_runner: Any | None = None
        self.epic_forward_batch_factory: Any | None = None
        self._async_loader: AsyncResidencyLoader | None = None
        self._tickets: dict[str, ApproxKVPrefetchTicket] = {}
        self._ticket_lock = threading.Lock()
        self._provisional_lock = threading.Lock()
        self._provisional_tokens = 0
        self._persistent_lease_lock = threading.Lock()
        self._persistent_leases: dict[KVSegmentKey, ApproxKVLease] = {}
        self._cross_store_coordinator = None
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
        resident_bytes: int | None = None,
        object_id: str | None = None,
        object_kind=None,
        dependencies: frozenset[str] = frozenset(),
        dense_cost_ms: float | None = None,
        recovery_cost_ms: float | None = None,
        next_use_ordinal: int | None = None,
        retired: bool = False,
    ) -> KVSegmentHandle | None:
        if not self.config.core_enabled:
            return None
        kwargs = {}
        if object_kind is not None:
            kwargs["object_kind"] = object_kind
        try:
            handle = self.store.register(
                key=key,
                token_ids=token_ids,
                source_start=source_start,
                residency=residency,
                backend_ref=backend_ref,
                release_backend=release_backend,
                resident_bytes=resident_bytes,
                object_id=object_id,
                dependencies=dependencies,
                dense_cost_ms=dense_cost_ms,
                recovery_cost_ms=recovery_cost_ms,
                next_use_ordinal=next_use_ordinal,
                retired=retired,
                **kwargs,
            )
        except ApproxKVStoreCapacityError:
            self.record_fallback("registration_store_capacity", len(token_ids))
            return None
        self._record_store_state()
        return handle

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
        result, transfer = self.store.load_resident(
            handle,
            target_tier,
            loader,
        )
        if source_tier != ResidencyTier.DEVICE and target_tier == ResidencyTier.DEVICE:
            num_tokens = transfer.num_tokens or result.key.token_count
            num_bytes = (
                transfer.bytes_transferred
                if transfer.bytes_transferred > 0
                else num_tokens * self.config.cross_store_bytes_per_token
            )
            if self.config.cross_store_enabled:
                self.validate_transfer_bytes(
                    num_bytes,
                    expected_bytes=(
                        num_tokens * self.config.cross_store_bytes_per_token
                    ),
                )
            self._record_h2d(num_tokens)
            collector = self.metrics_collector
            if collector is not None:
                callback = getattr(collector, "observe_approx_kv_h2d", None)
                if callback is not None:
                    callback(
                        num_tokens,
                        num_bytes,
                        transfer.duration_ms,
                    )
        self._record_store_state()
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

    def cross_store_resources(self) -> tuple[CrossStoreResource, ...]:
        resources = self.store.cross_store_resources()
        if not self.config.host_residency_enabled or self.residency_backend is None:
            return resources
        wrapped = []
        for resource in resources:
            handle = self.store.handle_for_object_id(resource.item.object_id)
            if handle.residency != ResidencyTier.DEVICE:
                wrapped.append(resource)
                continue

            def demote(
                handle=handle,
                item=resource.item,
            ):
                result = self.export_to_host(handle.backend_ref)
                try:
                    transfer_bytes = self.validate_transfer_bytes(
                        result.bytes_transferred,
                        expected_bytes=item.resident_bytes,
                    )
                except ValueError:
                    if result.release_backend is not None:
                        result.release_backend(
                            result.backend_ref,
                            ResidencyTier.HOST,
                        )
                    raise
                host_handle = self.store.commit_residency(
                    handle,
                    target_tier=ResidencyTier.HOST,
                    result=result,
                )
                self.record_host_export(
                    handle.key.token_count,
                    transfer_bytes,
                    duration_ms=result.duration_ms,
                )
                self._record_store_state()

                def undo() -> None:
                    self.restore_after_failed_demotion(host_handle)

                return AppliedAction(undo=undo)

            wrapped.append(
                CrossStoreResource(
                    item=replace(resource.item, demotable=True),
                    evict=resource.evict,
                    demote=demote,
                )
            )
        return tuple(wrapped)

    def cross_store_coordinator(self, tree_cache):
        if self._cross_store_coordinator is None:
            from sglang.srt.mem_cache.cross_store import CrossStoreCoordinator

            self._cross_store_coordinator = CrossStoreCoordinator(
                tree_cache,
                bytes_per_token=self.config.cross_store_bytes_per_token,
                host_budget_bytes=self.config.cross_store_host_budget_bytes,
            )
        return self._cross_store_coordinator

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

    @staticmethod
    def validate_transfer_bytes(
        measured_bytes: int,
        *,
        expected_bytes: int,
    ) -> int:
        if measured_bytes > 0 and measured_bytes != expected_bytes:
            raise ValueError(
                "approximate KV transfer byte count does not match the "
                f"configured device accounting: measured={measured_bytes}, "
                f"expected={expected_bytes}"
            )
        return expected_bytes

    def ensure_device(self, handle: KVSegmentHandle) -> KVSegmentHandle:
        if handle.residency == ResidencyTier.DEVICE:
            return handle
        backend = self.residency_backend
        if backend is None:
            raise RuntimeError("no approximate KV residency backend is bound")
        if self.config.async_prefetch_enabled and hasattr(backend, "begin_load"):
            return self.begin_prefetch(handle).wait()
        return self.ensure_resident(handle, ResidencyTier.DEVICE, backend)

    def restore_after_failed_demotion(
        self,
        handle: KVSegmentHandle,
    ) -> KVSegmentHandle:
        backend = self.residency_backend
        if backend is None:
            raise RuntimeError("no approximate KV residency backend is bound")
        if not hasattr(backend, "load_for_rollback"):
            return self.ensure_resident(
                handle,
                ResidencyTier.DEVICE,
                backend,
            )
        transfer = backend.load_for_rollback(
            handle,
            ResidencyTier.DEVICE,
        )
        restored = self.store.commit_residency(
            handle,
            target_tier=ResidencyTier.DEVICE,
            result=transfer,
        )
        num_tokens = transfer.num_tokens or restored.key.token_count
        num_bytes = (
            transfer.bytes_transferred
            if transfer.bytes_transferred > 0
            else num_tokens * self.config.cross_store_bytes_per_token
        )
        self.validate_transfer_bytes(
            num_bytes,
            expected_bytes=(num_tokens * self.config.cross_store_bytes_per_token),
        )
        self._record_h2d(num_tokens)
        self._record_store_state()
        return restored

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

    def record_host_export(
        self,
        num_tokens: int,
        num_bytes: int,
        *,
        duration_ms: float = 0.0,
    ) -> None:
        collector = self.metrics_collector
        if collector is None:
            return
        callback = getattr(collector, "increment_approx_kv_host_export", None)
        if callback is not None:
            callback(num_tokens, num_bytes)
        observe = getattr(collector, "observe_approx_kv_host_export", None)
        if observe is not None:
            observe(duration_ms)

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

    def record_cross_store_eviction(
        self,
        item,
        *,
        demoted: bool,
        requester: str,
    ) -> None:
        collector = self.metrics_collector
        if collector is None:
            return
        callback = getattr(
            collector,
            (
                "increment_cross_store_demotion"
                if demoted
                else "increment_cross_store_eviction"
            ),
            None,
        )
        if callback is not None:
            callback(
                requester=requester,
                provenance=item.provenance.value,
                object_kind=item.kind.value,
                num_bytes=item.resident_bytes,
            )

    def record_cross_store_reservation_failure(self, requires_reset: bool) -> None:
        collector = self.metrics_collector
        if collector is None:
            return
        callback = getattr(
            collector,
            "increment_cross_store_reservation_failure",
            None,
        )
        if callback is not None:
            callback(requires_reset=requires_reset)

    def record_cross_store_result(self, result) -> None:
        collector = self.metrics_collector
        if collector is None:
            return
        callback = getattr(collector, "record_cross_store_result", None)
        if callback is not None:
            callback(
                committed=result.committed,
                destroyed_bytes=result.destroyed_bytes,
                peak_device_bytes=result.peak_device_bytes,
                reserved_device_bytes=result.reserved_device_bytes,
            )
        self._record_store_state()

    def add_provisional_tokens(self, num_tokens: int) -> None:
        if num_tokens <= 0:
            raise ValueError("provisional token count must be positive")
        with self._provisional_lock:
            self._provisional_tokens += num_tokens
        self._record_store_state()

    def remove_provisional_tokens(self, num_tokens: int) -> None:
        if num_tokens <= 0:
            raise ValueError("provisional token count must be positive")
        with self._provisional_lock:
            if num_tokens > self._provisional_tokens:
                raise ValueError("provisional token count underflow")
            self._provisional_tokens -= num_tokens
        self._record_store_state()

    @property
    def provisional_tokens(self) -> int:
        with self._provisional_lock:
            return self._provisional_tokens

    def pin_registration(
        self,
        handle: KVSegmentHandle,
    ) -> ApproxKVLease:
        """Hold an opt-in persistent lease on a freshly registered segment.

        The lease survives request completion so a registered source object
        stays reusable across a target sequence; it is released by
        :meth:`reset`.
        """
        if not self.config.allow_persistent_pins:
            raise RuntimeError("persistent registration pins are disabled")
        with self._persistent_lease_lock:
            previous = self._persistent_leases.get(handle.key)
            if previous is not None and previous.generation == handle.generation:
                return previous
            additional = 0 if previous is not None else 1
            if (
                len(self._persistent_leases) + additional
                > self.config.max_persistent_pins
            ):
                raise RuntimeError(
                    "persistent registration pin cap exceeded: "
                    f"max={self.config.max_persistent_pins}"
                )
            if previous is not None:
                self.store.unpin(previous)
                self._persistent_leases.pop(handle.key, None)
            lease = self.store.pin(handle, ttl_s=None)
            self._persistent_leases[handle.key] = lease
        self._record_store_state()
        return lease

    def validate_persistent_registration_request(self, requested_pins: int) -> None:
        if requested_pins <= 0:
            raise ValueError("requested_pins must be positive")
        if not self.config.allow_persistent_pins:
            raise RuntimeError("persistent registration pins are disabled")
        with self._persistent_lease_lock:
            if (
                len(self._persistent_leases) + requested_pins
                > self.config.max_persistent_pins
            ):
                raise RuntimeError(
                    "persistent registration pin cap exceeded: "
                    f"requested={requested_pins}, "
                    f"active={len(self._persistent_leases)}, "
                    f"max={self.config.max_persistent_pins}"
                )

    def release_segment(self, handle: KVSegmentHandle) -> bool:
        released = self.store.release(handle)
        if released:
            self._record_store_state()
        return released

    @property
    def persistent_lease_count(self) -> int:
        with self._persistent_lease_lock:
            return len(self._persistent_leases)

    def _release_persistent_leases(self) -> None:
        with self._persistent_lease_lock:
            leases = tuple(self._persistent_leases.values())
            self._persistent_leases.clear()
        for lease in leases:
            self.store.unpin(lease)

    def _record_store_state(self) -> None:
        collector = self.metrics_collector
        if collector is None:
            return
        callback = getattr(collector, "set_approx_kv_store_state", None)
        if callback is not None:
            callback(
                records=self.store.record_count,
                device_bytes=self.store.device_owned_bytes,
                host_bytes=self.store.host_owned_bytes,
                leases=self.store.lease_count,
                orphans=self.store.orphan_count,
                provisional_tokens=self.provisional_tokens,
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
        self._release_persistent_leases()
        self.store.reset()
        with self._provisional_lock:
            self._provisional_tokens = 0
        self._reset_cross_store_accounting()
        self._record_store_state()

    def _reset_cross_store_accounting(self) -> None:
        coordinator = self._cross_store_coordinator
        if coordinator is not None:
            reset_accounting = getattr(coordinator, "reset_accounting", None)
            if reset_accounting is not None:
                reset_accounting(force=True)
        collector = self.metrics_collector
        if collector is None:
            return
        callback = getattr(collector, "set_cross_store_device_accounting", None)
        if callback is not None:
            callback(peak_device_bytes=0, reserved_device_bytes=0)

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
