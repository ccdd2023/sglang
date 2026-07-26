from __future__ import annotations

import threading


class CrossStoreEventClock:
    def __init__(self) -> None:
        self._ordinal = 0
        self._lock = threading.Lock()

    def tick(self) -> int:
        with self._lock:
            self._ordinal += 1
            return self._ordinal

    @property
    def current(self) -> int:
        with self._lock:
            return self._ordinal

    def reset(self) -> None:
        with self._lock:
            self._ordinal = 0


_GLOBAL_EVENT_CLOCK = CrossStoreEventClock()


def global_event_clock() -> CrossStoreEventClock:
    return _GLOBAL_EVENT_CLOCK
