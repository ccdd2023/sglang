from __future__ import annotations

import threading
import time

import pytest

from sglang.srt.mem_cache.kvcomm.config import KVCommFeatureConfig
from sglang.srt.mem_cache.kvcomm.manager import KVCommManager
from sglang.srt.mem_cache.kvcomm.types import (
    KVPrefetchHint,
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
    token_ids_hash,
)
from sglang.srt.mem_cache.kvcomm_prefetch.coordinator import (
    KVPrefetchCoordinator,
)
from sglang.srt.mem_cache.kvcomm_prefetch.scheduler import (
    AsyncKVPrefetchScheduler,
    MiddleKVPrefetchError,
)


def _key(name: str, token: int) -> KVSegmentKey:
    tokens = (token,)
    return KVSegmentKey(
        content_hash=name,
        token_hash=token_ids_hash(tokens),
        token_count=1,
        model_id="test",
        cache_dtype="bf16",
        kind=SegmentKind.MIDDLE,
    )


class BlockingLoader:
    def __init__(self, blocked_name: str = "blocker") -> None:
        self.blocked_name = blocked_name
        self.started = threading.Event()
        self.unblock = threading.Event()
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def load(self, handle, target_tier):
        with self._lock:
            self.calls.append(handle.key.content_hash)
        if handle.key.content_hash == self.blocked_name:
            self.started.set()
            assert self.unblock.wait(2)
        return f"device-{handle.key.content_hash}"


def _scheduler(*names: str):
    manager = KVCommManager(
        KVCommFeatureConfig(core_enabled=True, prefetch_enabled=True)
    )
    keys = {}
    for token, name in enumerate(names, start=1):
        key = _key(name, token)
        keys[name] = key
        manager.register_segment(
            key=key,
            token_ids=(token,),
            source_start=token,
            residency=ResidencyTier.HOST,
            backend_ref=f"host-{name}",
        )
    loader = BlockingLoader()
    coordinator = KVPrefetchCoordinator(manager=manager, loader=loader)
    scheduler = AsyncKVPrefetchScheduler(
        manager=manager,
        coordinator=coordinator,
        worker_count=1,
    )
    return manager, keys, loader, scheduler


def test_submit_returns_before_background_load_finishes():
    _, keys, loader, scheduler = _scheduler("blocker")
    ticket = scheduler.submit(KVPrefetchHint(keys["blocker"]))
    assert loader.started.wait(1)
    assert not ticket.done
    with pytest.raises(MiddleKVPrefetchError, match="timed out"):
        ticket.wait(timeout_s=0)
    loader.unblock.set()
    assert ticket.wait(timeout_s=1).residency == ResidencyTier.DEVICE
    assert ticket.done
    assert ticket.successful
    assert ticket.result is not None
    assert ticket.result.queue_wait_s >= 0
    assert ticket.result.load_elapsed_s >= 0
    ticket.release()
    scheduler.close()


def test_pending_work_is_ordered_by_deadline_then_priority():
    _, keys, loader, scheduler = _scheduler(
        "blocker", "low", "high", "deadline"
    )
    blocker = scheduler.submit(KVPrefetchHint(keys["blocker"]))
    assert loader.started.wait(1)
    low = scheduler.submit(KVPrefetchHint(keys["low"], priority=1))
    high = scheduler.submit(KVPrefetchHint(keys["high"], priority=9))
    deadline = scheduler.submit(
        KVPrefetchHint(keys["deadline"], deadline_s=1, priority=-10)
    )
    loader.unblock.set()
    blocker.wait(timeout_s=1)
    deadline.wait(timeout_s=1)
    high.wait(timeout_s=1)
    low.wait(timeout_s=1)
    assert loader.calls == ["blocker", "deadline", "high", "low"]
    blocker.release()
    deadline.release()
    high.release()
    low.release()
    scheduler.close()


def test_expired_pending_request_is_not_loaded():
    manager, keys, loader, scheduler = _scheduler("blocker", "expired")
    blocker = scheduler.submit(KVPrefetchHint(keys["blocker"]))
    assert loader.started.wait(1)
    expired = scheduler.submit(
        KVPrefetchHint(keys["expired"], deadline_s=0.01, priority=100)
    )
    time.sleep(0.03)
    loader.unblock.set()
    blocker.wait(timeout_s=1)
    with pytest.raises(MiddleKVPrefetchError, match="deadline_expired"):
        expired.wait(timeout_s=1)
    assert expired.result is not None
    assert expired.result.expired == 1
    assert loader.calls == ["blocker"]
    assert manager.store.lookup(keys["expired"]).residency == ResidencyTier.HOST
    blocker.release()
    scheduler.close()


def test_release_before_completion_does_not_leak_a_lease():
    manager, keys, loader, scheduler = _scheduler("blocker")
    ticket = scheduler.submit(KVPrefetchHint(keys["blocker"]))
    assert loader.started.wait(1)
    assert ticket.release()
    assert not ticket.release()
    loader.unblock.set()
    ticket.wait(timeout_s=1)
    assert manager.store.lease_count == 0
    scheduler.close()
