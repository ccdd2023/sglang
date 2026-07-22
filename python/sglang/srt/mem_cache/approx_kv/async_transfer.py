from __future__ import annotations

import threading
import uuid
from enum import Enum
from typing import Callable

from .store import (
    ApproxKVSegmentStore,
    AsyncResidencyTransfer,
    ResidencyLoadResult,
)
from .types import KVSegmentHandle, ResidencyTier


class AsyncTransferState(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ApproxKVPrefetchTicket:
    def __init__(
        self,
        *,
        store: ApproxKVSegmentStore,
        handle: KVSegmentHandle,
        target_tier: ResidencyTier,
        transfer: AsyncResidencyTransfer,
        on_finish: Callable[[str], None] | None = None,
        on_complete: Callable[[ResidencyLoadResult], None] | None = None,
    ) -> None:
        self.ticket_id = uuid.uuid4().hex
        self._store = store
        self._handle = handle
        self._target_tier = target_tier
        self._transfer = transfer
        self._on_finish = on_finish
        self._on_complete = on_complete
        self._state = AsyncTransferState.PENDING
        self._result: KVSegmentHandle | None = None
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._wait_lock = threading.Lock()

    @property
    def state(self) -> AsyncTransferState:
        with self._lock:
            return self._state

    @property
    def done(self) -> bool:
        with self._lock:
            if self._state != AsyncTransferState.PENDING:
                return True
        return self._transfer.done

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def wait(self, timeout_s: float | None = None) -> KVSegmentHandle:
        with self._wait_lock:
            with self._lock:
                if self._state == AsyncTransferState.COMPLETED:
                    assert self._result is not None
                    return self._result
                if self._state == AsyncTransferState.CANCELLED:
                    raise RuntimeError("approximate KV prefetch was cancelled")
                if self._state == AsyncTransferState.FAILED:
                    assert self._error is not None
                    raise RuntimeError("approximate KV prefetch failed") from self._error

            load_result = None
            try:
                load_result = self._transfer.wait(timeout_s)
                result = self._store.commit_residency(
                    self._handle,
                    target_tier=self._target_tier,
                    result=load_result,
                )
                if self._on_complete is not None:
                    self._on_complete(load_result)
            except TimeoutError:
                raise
            except BaseException as exc:
                if load_result is None:
                    self._transfer.cancel()
                with self._lock:
                    self._state = AsyncTransferState.FAILED
                    self._error = exc
                self._finish()
                raise

            with self._lock:
                self._state = AsyncTransferState.COMPLETED
                self._result = result
            self._finish()
            return result

    def cancel(self) -> bool:
        with self._wait_lock:
            with self._lock:
                if self._state != AsyncTransferState.PENDING:
                    return False
                self._state = AsyncTransferState.CANCELLED
            try:
                self._transfer.cancel()
            finally:
                self._finish()
            return True

    def _finish(self) -> None:
        callback = self._on_finish
        if callback is not None:
            self._on_finish = None
            callback(self.ticket_id)
