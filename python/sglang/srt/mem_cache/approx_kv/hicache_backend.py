from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

from sglang.srt.managers.cache_controller import (
    device_module,
    make_timing_event_pair,
)

from .radix_backend import DeviceKVRef
from .store import (
    AsyncResidencyTransfer,
    ResidencyLoadResult,
)
from .types import KVSegmentHandle, ResidencyTier


@dataclass(frozen=True)
class HiCacheHostKVRef:
    indices: torch.Tensor


class _EventResidencyTransfer(AsyncResidencyTransfer):
    def __init__(
        self,
        *,
        finish_event: Any,
        start_event: Any,
        timing_enabled: bool,
        result_factory: Callable[[float], ResidencyLoadResult],
        release_pending: Callable[[], None],
        keep_alive: tuple[object, ...],
    ) -> None:
        self._finish_event = finish_event
        self._start_event = start_event
        self._timing_enabled = timing_enabled
        self._result_factory = result_factory
        self._release_pending = release_pending
        self._keep_alive = keep_alive
        self._result: ResidencyLoadResult | None = None
        self._cancelled = False
        self._lock = threading.Lock()

    @property
    def done(self) -> bool:
        with self._lock:
            if self._result is not None or self._cancelled:
                return True
        return bool(self._finish_event.query())

    def wait(self, timeout_s: float | None = None) -> ResidencyLoadResult:
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while not self._finish_event.query():
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("HiCache approximate KV transfer timed out")
            time.sleep(0.001)
        self._finish_event.synchronize()
        with self._lock:
            if self._cancelled:
                raise RuntimeError("HiCache approximate KV transfer was cancelled")
            if self._result is None:
                duration_ms = (
                    self._start_event.elapsed_time(self._finish_event)
                    if self._timing_enabled
                    else 0.0
                )
                self._result = self._result_factory(duration_ms)
                self._keep_alive = ()
            return self._result

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled:
                return
        while not self._finish_event.query():
            time.sleep(0.001)
        self._finish_event.synchronize()
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            self._keep_alive = ()
        self._release_pending()


class HiCacheResidencyBackend:
    def __init__(self, cache_controller: Any) -> None:
        self._controller = cache_controller
        self._lock = threading.Lock()

    def begin_export(self, device_ref: DeviceKVRef) -> AsyncResidencyTransfer:
        controller = self._controller
        token_count = len(device_ref.indices)
        self._validate_page_alignment(token_count)
        host_indices = controller.mem_pool_host.alloc(token_count)
        if host_indices is None:
            raise MemoryError("unable to allocate HiCache host slots")

        try:
            with self._lock:
                transfer_host, transfer_device = controller.move_indices(
                    host_indices,
                    device_ref.indices,
                )
                start_event, finish_event, timing_enabled = make_timing_event_pair()
                start_event.record()
                with device_module.stream(controller.write_stream):
                    start_event.wait(controller.write_stream)
                    controller.mem_pool_host.backup_from_device_all_layer(
                        controller.mem_pool_device,
                        transfer_host,
                        transfer_device,
                        controller.io_backend,
                    )
                    if controller.has_draft:
                        controller.mem_pool_host_draft.backup_from_device_all_layer(
                            controller.mem_pool_device_draft,
                            transfer_host,
                            transfer_device,
                            controller.io_backend,
                        )
                    finish_event.record()
                    self._record_stream(
                        transfer_host,
                        transfer_device,
                        controller.write_stream,
                    )
        except Exception:
            self._synchronize_stream(controller.write_stream)
            controller.evict_host(host_indices)
            raise

        bytes_transferred = self._estimate_bytes(token_count)
        return _EventResidencyTransfer(
            finish_event=finish_event,
            start_event=start_event,
            timing_enabled=timing_enabled,
            result_factory=lambda duration_ms: ResidencyLoadResult(
                backend_ref=HiCacheHostKVRef(host_indices),
                release_backend=self.release_host,
                num_tokens=token_count,
                bytes_transferred=bytes_transferred,
                duration_ms=duration_ms,
            ),
            release_pending=lambda: controller.evict_host(host_indices),
            keep_alive=(
                device_ref,
                host_indices,
                transfer_host,
                transfer_device,
            ),
        )

    def export_to_host(self, device_ref: DeviceKVRef) -> ResidencyLoadResult:
        return self.begin_export(device_ref).wait()

    def begin_load(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
    ) -> AsyncResidencyTransfer:
        if target_tier != ResidencyTier.DEVICE:
            raise NotImplementedError("HiCache backend loads only to device")
        if not isinstance(handle.backend_ref, HiCacheHostKVRef):
            raise TypeError("host-resident handle must carry HiCacheHostKVRef")

        controller = self._controller
        host_indices = handle.backend_ref.indices
        token_count = len(host_indices)
        self._validate_page_alignment(token_count)
        device_indices = controller.mem_pool_device_allocator.alloc(token_count)
        if device_indices is None or len(device_indices) != token_count:
            if device_indices is not None:
                controller.evict_device(device_indices)
            raise MemoryError("unable to allocate device slots for HiCache load")

        try:
            with self._lock:
                transfer_host, transfer_device = controller.move_indices(
                    host_indices,
                    device_indices,
                )
                start_event, finish_event, timing_enabled = make_timing_event_pair()
                start_event.record()
                with device_module.stream(controller.load_stream):
                    start_event.wait(controller.load_stream)
                    for layer_id in range(controller.layer_num):
                        controller.mem_pool_host.load_to_device_per_layer(
                            controller.mem_pool_device,
                            transfer_host,
                            transfer_device,
                            layer_id,
                            controller.io_backend,
                        )
                        if (
                            controller.has_draft
                            and layer_id < controller.mem_pool_host_draft.layer_num
                        ):
                            controller.mem_pool_host_draft.load_to_device_per_layer(
                                controller.mem_pool_device_draft,
                                transfer_host,
                                transfer_device,
                                layer_id,
                                controller.io_backend,
                            )
                    finish_event.record()
                    self._record_stream(
                        transfer_host,
                        transfer_device,
                        controller.load_stream,
                    )
        except Exception:
            self._synchronize_stream(controller.load_stream)
            controller.evict_device(device_indices)
            raise

        bytes_transferred = self._estimate_bytes(token_count)
        return _EventResidencyTransfer(
            finish_event=finish_event,
            start_event=start_event,
            timing_enabled=timing_enabled,
            result_factory=lambda duration_ms: ResidencyLoadResult(
                backend_ref=DeviceKVRef(device_indices),
                release_backend=self.release_device,
                release_on_stale=self.release_device,
                num_tokens=token_count,
                bytes_transferred=bytes_transferred,
                duration_ms=duration_ms,
            ),
            release_pending=lambda: controller.evict_device(device_indices),
            keep_alive=(
                handle,
                host_indices,
                device_indices,
                transfer_host,
                transfer_device,
            ),
        )

    def load(
        self,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
    ) -> ResidencyLoadResult:
        return self.begin_load(handle, target_tier).wait()

    def release_host(
        self,
        backend_ref: object,
        residency: ResidencyTier,
    ) -> None:
        if residency != ResidencyTier.HOST or not isinstance(
            backend_ref,
            HiCacheHostKVRef,
        ):
            raise TypeError("HiCache releaser received a non-host KV ref")
        self._controller.evict_host(backend_ref.indices)

    def release_device(
        self,
        backend_ref: object,
        residency: ResidencyTier,
    ) -> None:
        if residency != ResidencyTier.DEVICE or not isinstance(
            backend_ref,
            DeviceKVRef,
        ):
            raise TypeError("HiCache releaser received a non-device KV ref")
        self._controller.evict_device(backend_ref.indices)

    def _validate_page_alignment(self, token_count: int) -> None:
        page_size = int(self._controller.mem_pool_host.page_size)
        if token_count <= 0 or token_count % page_size:
            raise ValueError(
                "approximate KV HiCache transfers must be positive and page aligned"
            )

    def _estimate_bytes(self, token_count: int) -> int:
        pool = self._controller.mem_pool_device
        try:
            key_bytes, value_bytes = pool.get_kv_size_bytes()
            total_bytes = int(key_bytes) + int(value_bytes)
            pool_size = int(self._controller.mem_pool_device_allocator.size_full)
            return 0 if pool_size <= 0 else total_bytes * token_count // pool_size
        except (AttributeError, TypeError, ValueError):
            return 0

    @staticmethod
    def _record_stream(
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        stream: Any,
    ) -> None:
        if host_indices.is_cuda:
            host_indices.record_stream(stream)
        if device_indices.is_cuda:
            device_indices.record_stream(stream)

    @staticmethod
    def _synchronize_stream(stream: Any) -> None:
        synchronize = getattr(stream, "synchronize", None)
        if synchronize is not None:
            synchronize()
        else:
            device_module.synchronize()
