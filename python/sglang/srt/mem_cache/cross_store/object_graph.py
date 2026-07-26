from __future__ import annotations

import threading
from collections import defaultdict

from .event_clock import CrossStoreEventClock
from .types import CrossStoreObject


class CrossStoreObjectGraph:
    def __init__(self, clock: CrossStoreEventClock | None = None) -> None:
        self._owns_clock = clock is None
        self.clock = clock if clock is not None else CrossStoreEventClock()
        self._objects: dict[str, CrossStoreObject] = {}
        self._dependents: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def register(self, item: CrossStoreObject) -> CrossStoreObject:
        with self._lock:
            missing = item.dependencies.difference(self._objects)
            if missing:
                raise KeyError(f"missing dependencies: {sorted(missing)}")
            if item.object_id in self._objects:
                raise KeyError(f"object already exists: {item.object_id}")
            stored = item.touched(self.clock.tick())
            self._objects[stored.object_id] = stored
            for dependency in stored.dependencies:
                self._dependents[dependency].add(stored.object_id)
            if self._has_cycle():
                self._remove_without_validation(stored.object_id)
                raise ValueError("dependency cycle detected")
            return stored

    def get(self, object_id: str) -> CrossStoreObject:
        with self._lock:
            return self._objects[object_id]

    def touch(self, object_id: str) -> CrossStoreObject:
        with self._lock:
            item = self._objects[object_id].touched(self.clock.tick())
            self._objects[object_id] = item
            return item

    def replace(self, item: CrossStoreObject) -> None:
        with self._lock:
            current = self._objects[item.object_id]
            if item.dependencies != current.dependencies:
                raise ValueError("dependencies cannot be changed in-place")
            self._objects[item.object_id] = item

    def eviction_closure(self, object_id: str) -> tuple[CrossStoreObject, ...]:
        with self._lock:
            closure_ids: set[str] = set()
            stack = [object_id]
            while stack:
                current = stack.pop()
                if current in closure_ids:
                    continue
                closure_ids.add(current)
                stack.extend(self._dependents.get(current, ()))
            return tuple(
                sorted(
                    (self._objects[item_id] for item_id in closure_ids),
                    key=lambda item: item.object_id,
                )
            )

    def remove_closure(self, object_id: str) -> tuple[CrossStoreObject, ...]:
        with self._lock:
            closure = self.eviction_closure(object_id)
            for item in reversed(closure):
                self._remove_without_validation(item.object_id)
            self.assert_no_orphans()
            return closure

    def values(self) -> tuple[CrossStoreObject, ...]:
        with self._lock:
            return tuple(self._objects.values())

    def usage_bytes(self, *, tier: str) -> int:
        with self._lock:
            return sum(
                item.resident_bytes
                for item in self._objects.values()
                if item.tier.value == tier
            )

    def assert_no_orphans(self) -> None:
        with self._lock:
            for item in self._objects.values():
                missing = item.dependencies.difference(self._objects)
                if missing:
                    raise AssertionError(
                        f"orphan object {item.object_id}: {sorted(missing)}"
                    )

    def reset(self) -> None:
        with self._lock:
            self._objects.clear()
            self._dependents.clear()
            if self._owns_clock:
                self.clock.reset()

    def _remove_without_validation(self, object_id: str) -> None:
        item = self._objects.pop(object_id)
        for dependency in item.dependencies:
            dependents = self._dependents.get(dependency)
            if dependents is not None:
                dependents.discard(object_id)
                if not dependents:
                    self._dependents.pop(dependency, None)
        self._dependents.pop(object_id, None)

    def _has_cycle(self) -> bool:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(object_id: str) -> bool:
            if object_id in visiting:
                return True
            if object_id in visited:
                return False
            visiting.add(object_id)
            for dependency in self._objects[object_id].dependencies:
                if visit(dependency):
                    return True
            visiting.remove(object_id)
            visited.add(object_id)
            return False

        return any(visit(object_id) for object_id in self._objects)
