from __future__ import annotations

import heapq
import threading
import time
from math import inf

from sglang.srt.mem_cache.kvcomm.manager import KVCommManager
from sglang.srt.mem_cache.kvcomm.types import (
    KVPrefetchHint,
    KVSegmentHandle,
    KVSegmentKey,
    ResidencyTier,
)
from sglang.srt.mem_cache.kvcomm_prefetch.coordinator import (
    KVPrefetchCoordinator,
    PrefetchResult,
)


class MiddleKVPrefetchError(RuntimeError):
    pass


class PrefetchTicket:
    """Completion and lease handle for one background residency request."""

    def __init__(
        self,
        *,
        key: KVSegmentKey,
        manager: KVCommManager,
        coordinator: KVPrefetchCoordinator,
    ) -> None:
        self.key = key
        self._manager = manager
        self._coordinator = coordinator
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._result: PrefetchResult | None = None
        self._released = False

    @property
    def done(self) -> bool:
        return self._event.is_set()

    @property
    def result(self) -> PrefetchResult | None:
        """Return the result once complete, or ``None`` while queued/running."""

        with self._lock:
            return self._result

    @property
    def successful(self) -> bool:
        with self._lock:
            result = self._result
        if result is None:
            return False
        handle = self._manager.store.lookup(self.key)
        return (
            not result.disabled
            and result.failed == 0
            and result.store_misses == 0
            and handle is not None
            and handle.residency == ResidencyTier.DEVICE
        )

    def wait(self, timeout_s: float | None = None) -> KVSegmentHandle:
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        if not self._event.wait(timeout_s):
            raise MiddleKVPrefetchError("timed out waiting for middle KV prefetch")
        with self._lock:
            result = self._result
        if result is None:  # pragma: no cover - defensive event invariant
            raise MiddleKVPrefetchError("prefetch completed without a result")
        if result.disabled:
            raise MiddleKVPrefetchError("KV prefetch is disabled")
        if result.store_misses:
            raise MiddleKVPrefetchError(f"middle KV segment not found: {self.key}")
        if result.failed:
            details = "; ".join(result.failure_reasons)
            raise MiddleKVPrefetchError(f"middle KV prefetch failed: {details}")
        handle = self._manager.store.lookup(self.key)
        if handle is None or handle.residency != ResidencyTier.DEVICE:
            raise MiddleKVPrefetchError(
                "prefetch completed without a device-resident segment"
            )
        return handle

    def device_indices(self):
        # Imported lazily so the scheduler remains independent of torch.
        from sglang.srt.mem_cache.kvcomm.radix_backend import DeviceKVRef

        handle = self.wait()
        if not isinstance(handle.backend_ref, DeviceKVRef):
            raise MiddleKVPrefetchError(
                "device-resident segment does not carry DeviceKVRef"
            )
        return handle.backend_ref.indices

    def release(self) -> bool:
        """Release now, or arrange release if the load is still in flight."""

        with self._lock:
            if self._released:
                return False
            self._released = True
            result = self._result
        if result is not None:
            self._coordinator.release(result)
        return True

    def _complete(self, result: PrefetchResult) -> None:
        with self._lock:
            self._result = result
            release_on_completion = self._released
        if release_on_completion:
            self._coordinator.release(result)
        self._event.set()

    def __enter__(self) -> "PrefetchTicket":
        self.wait()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


class AsyncKVPrefetchScheduler:
    """Deadline/priority queue that performs residency loads off-thread.

    ``deadline_s`` is interpreted as a relative queue-and-load budget from
    submission. It controls pending work admission; an already-running backend
    copy is not forcefully interrupted.
    """

    def __init__(
        self,
        *,
        manager: KVCommManager,
        coordinator: KVPrefetchCoordinator,
        worker_count: int = 1,
    ) -> None:
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        self._manager = manager
        self._coordinator = coordinator
        self._condition = threading.Condition()
        self._queue: list[
            tuple[float, int, int, float, KVPrefetchHint, PrefetchTicket]
        ] = []
        self._sequence = 0
        self._closed = False
        self._workers = [
            threading.Thread(
                target=self._run,
                name=f"kv-prefetch-{index}",
                daemon=True,
            )
            for index in range(worker_count)
        ]
        for worker in self._workers:
            worker.start()

    def submit(self, hint: KVPrefetchHint) -> PrefetchTicket:
        if hint.deadline_s is not None and hint.deadline_s < 0:
            raise ValueError("deadline_s must be non-negative")
        submitted_at = time.monotonic()
        absolute_deadline = (
            inf
            if hint.deadline_s is None
            else submitted_at + hint.deadline_s
        )
        ticket = PrefetchTicket(
            key=hint.key,
            manager=self._manager,
            coordinator=self._coordinator,
        )
        with self._condition:
            if self._closed:
                raise RuntimeError("prefetch scheduler is closed")
            sequence = self._sequence
            self._sequence += 1
            heapq.heappush(
                self._queue,
                (
                    absolute_deadline,
                    -hint.priority,
                    sequence,
                    submitted_at,
                    hint,
                    ticket,
                ),
            )
            self._condition.notify()
        return ticket

    def close(self, *, wait: bool = True, cancel_pending: bool = False) -> None:
        cancelled: list[PrefetchTicket] = []
        with self._condition:
            if not self._closed:
                self._closed = True
                if cancel_pending:
                    while self._queue:
                        *_, ticket = heapq.heappop(self._queue)
                        cancelled.append(ticket)
                self._condition.notify_all()
        for ticket in cancelled:
            ticket._complete(
                PrefetchResult(
                    requested=1,
                    unique_requested=1,
                    failed=1,
                    failure_reasons=["scheduler_closed_before_start"],
                )
            )
        if wait:
            for worker in self._workers:
                worker.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if not self._queue:
                    return
                (
                    absolute_deadline,
                    _,
                    _,
                    submitted_at,
                    hint,
                    ticket,
                ) = heapq.heappop(self._queue)

            started_at = time.monotonic()
            if started_at >= absolute_deadline:
                result = PrefetchResult(
                    requested=1,
                    unique_requested=1,
                    failed=1,
                    expired=1,
                    failure_reasons=["deadline_expired_before_start"],
                )
            else:
                result = self._coordinator.prefetch((hint,))
            result.queue_wait_s = started_at - submitted_at
            result.load_elapsed_s = time.monotonic() - started_at
            ticket._complete(result)

    def __enter__(self) -> "AsyncKVPrefetchScheduler":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
