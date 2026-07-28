from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetSnapshot:
    device_limit_bytes: int
    host_limit_bytes: int
    device_used_bytes: int
    host_used_bytes: int
    device_reserved_bytes: int
    peak_device_bytes: int

    @property
    def device_available_bytes(self) -> int:
        return (
            self.device_limit_bytes
            - self.device_used_bytes
            - self.device_reserved_bytes
        )


class CrossStoreBudget:
    def __init__(self, *, device_limit_bytes: int, host_limit_bytes: int) -> None:
        if device_limit_bytes <= 0 or host_limit_bytes < 0:
            raise ValueError("invalid byte budgets")
        self.device_limit_bytes = device_limit_bytes
        self.host_limit_bytes = host_limit_bytes
        self.device_used_bytes = 0
        self.host_used_bytes = 0
        self.device_reserved_bytes = 0
        self.peak_device_bytes = 0
        self._lock = threading.RLock()

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                device_limit_bytes=self.device_limit_bytes,
                host_limit_bytes=self.host_limit_bytes,
                device_used_bytes=self.device_used_bytes,
                host_used_bytes=self.host_used_bytes,
                device_reserved_bytes=self.device_reserved_bytes,
                peak_device_bytes=self.peak_device_bytes,
            )

    def reserve_device(
        self,
        num_bytes: int,
        *,
        allow_overcommit: bool = False,
    ) -> None:
        if num_bytes <= 0:
            raise ValueError("reservation must be positive")
        with self._lock:
            if (
                not allow_overcommit
                and self.snapshot().device_available_bytes < num_bytes
            ):
                raise MemoryError("insufficient device budget")
            self.device_reserved_bytes += num_bytes
            self._update_peak()

    def release_device_reservation(self, num_bytes: int) -> None:
        with self._lock:
            if num_bytes < 0 or num_bytes > self.device_reserved_bytes:
                raise ValueError("invalid device reservation release")
            self.device_reserved_bytes -= num_bytes

    def commit_device(self, num_bytes: int) -> None:
        with self._lock:
            self.release_device_reservation(num_bytes)
            self.device_used_bytes += num_bytes
            self._update_peak()

    def release_device(self, num_bytes: int) -> None:
        with self._lock:
            if num_bytes < 0 or num_bytes > self.device_used_bytes:
                raise ValueError("invalid device release")
            self.device_used_bytes -= num_bytes

    def restore_device(self, num_bytes: int) -> None:
        with self._lock:
            if num_bytes < 0:
                raise ValueError("invalid device restore")
            if self.snapshot().device_available_bytes < num_bytes:
                raise MemoryError("insufficient device budget for restore")
            self.device_used_bytes += num_bytes
            self._update_peak()

    def release_host(self, num_bytes: int) -> None:
        with self._lock:
            if num_bytes < 0 or num_bytes > self.host_used_bytes:
                raise ValueError("invalid host release")
            self.host_used_bytes -= num_bytes

    def restore_host(self, num_bytes: int) -> None:
        with self._lock:
            if num_bytes < 0:
                raise ValueError("invalid host restore")
            if self.host_used_bytes + num_bytes > self.host_limit_bytes:
                raise MemoryError("insufficient host budget for restore")
            self.host_used_bytes += num_bytes

    def demote(self, num_bytes: int) -> None:
        with self._lock:
            if num_bytes > self.device_used_bytes:
                raise ValueError("cannot demote more bytes than device usage")
            if self.host_used_bytes + num_bytes > self.host_limit_bytes:
                raise MemoryError("insufficient host budget")
            self.device_used_bytes -= num_bytes
            self.host_used_bytes += num_bytes

    def promote(self, num_bytes: int) -> None:
        with self._lock:
            if num_bytes > self.host_used_bytes:
                raise ValueError("cannot promote more bytes than host usage")
            if self.snapshot().device_available_bytes < num_bytes:
                raise MemoryError("insufficient device budget")
            self.host_used_bytes -= num_bytes
            self.device_used_bytes += num_bytes
            self._update_peak()

    def seed_usage(self, *, device_bytes: int = 0, host_bytes: int = 0) -> None:
        with self._lock:
            if self.device_reserved_bytes:
                raise RuntimeError("cannot seed usage with an active reservation")
            if (
                device_bytes < 0
                or host_bytes < 0
                or device_bytes > self.device_limit_bytes
                or host_bytes > self.host_limit_bytes
            ):
                raise ValueError("invalid seeded usage")
            self.device_used_bytes = device_bytes
            self.host_used_bytes = host_bytes
            self._update_peak()

    def reconcile_usage(self, *, device_bytes: int, host_bytes: int) -> None:
        self.seed_usage(device_bytes=device_bytes, host_bytes=host_bytes)

    def reset_accounting(self, *, force: bool = False) -> None:
        """Clear usage and the peak high-water mark after a full store reset.

        Without ``force``, an active reservation still indicates a live
        allocation. A full store reset uses ``force`` to discard stale
        accounting left by an interrupted allocation.
        """
        with self._lock:
            if self.device_reserved_bytes and not force:
                raise RuntimeError(
                    "cannot reset cross-store accounting with an active reservation"
                )
            self.device_used_bytes = 0
            self.host_used_bytes = 0
            self.device_reserved_bytes = 0
            self.peak_device_bytes = 0

    def _update_peak(self) -> None:
        self.peak_device_bytes = max(
            self.peak_device_bytes,
            self.device_used_bytes + self.device_reserved_bytes,
        )
