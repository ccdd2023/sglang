from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .budget import CrossStoreBudget
from .policy import CrossStorePolicy
from .types import CrossStoreObject, CrossStoreTier

Undo = Callable[[], None]


@dataclass(frozen=True)
class AppliedAction:
    undo: Undo
    commit: Callable[[], None] = lambda: None


Action = Callable[[], AppliedAction | None]


class AllocationFailurePoint(str, Enum):
    AFTER_RESERVE = "after_reserve"
    AFTER_VICTIM_SELECTION = "after_victim_selection"
    AFTER_EVICTION = "after_eviction"
    AFTER_ALLOCATION = "after_allocation"
    BEFORE_COMMIT = "before_commit"


@dataclass(frozen=True)
class CrossStoreResource:
    item: CrossStoreObject
    evict: Action
    demote: Action | None = None


@dataclass(frozen=True)
class ReservationResult:
    allocation: Any | None
    victim_ids: tuple[str, ...]
    demoted_ids: tuple[str, ...]
    committed: bool
    requires_reset: bool
    victim_items: tuple[CrossStoreObject, ...] = ()
    demoted_items: tuple[CrossStoreObject, ...] = ()
    irreversible_actions: bool = False
    destroyed_bytes: int = 0
    demoted_bytes: int = 0
    peak_device_bytes: int = 0
    reserved_device_bytes: int = 0
    rolled_back_ids: tuple[str, ...] = ()
    failure: str | None = None


class CrossStoreAllocator:
    def __init__(
        self,
        *,
        budget: CrossStoreBudget,
        policy: CrossStorePolicy,
        fault_injector: Callable[[AllocationFailurePoint], None] | None = None,
    ) -> None:
        self.budget = budget
        self.policy = policy
        self.fault_injector = fault_injector
        self._lock = threading.RLock()

    def allocate(
        self,
        *,
        required_device_bytes: int,
        resources: tuple[CrossStoreResource, ...],
        resource_provider: Callable[[], tuple[CrossStoreResource, ...]] | None = None,
        allocate_backend: Callable[[], Any | None],
        release_allocation: Callable[[Any], None],
    ) -> ReservationResult:
        with self._lock:
            applied: list[tuple[CrossStoreResource, str, AppliedAction | None]] = []
            stale_victims = 0
            allocation = None
            reserved = False
            requires_reset = False
            committed_action_indexes: set[int] = set()
            rolled_back_indexes: set[int] = set()
            commit_phase = False
            try:
                self.budget.reserve_device(
                    required_device_bytes,
                    allow_overcommit=True,
                )
                reserved = True
                self._fault(AllocationFailurePoint.AFTER_RESERVE)

                excluded_roots: set[tuple[str, int, str]] = set()
                inactive_resources: set[tuple[str, int, str]] = set()
                selection_faulted = False
                current_resources = resources
                refresh_resources = False
                while self.budget.snapshot().device_available_bytes < 0:
                    if refresh_resources and resource_provider is not None:
                        current_resources = resource_provider()
                        refresh_resources = False
                    active_resources = tuple(
                        resource
                        for resource in current_resources
                        if self._resource_identity(resource) not in inactive_resources
                    )
                    candidates = [
                        resource
                        for resource in active_resources
                        if resource.item.tier == CrossStoreTier.DEVICE
                        and not resource.item.protected
                        and self._resource_identity(resource) not in excluded_roots
                    ]
                    if not candidates and resource_provider is not None:
                        current_resources = resource_provider()
                        active_resources = tuple(
                            resource
                            for resource in current_resources
                            if self._resource_identity(resource)
                            not in inactive_resources
                        )
                        candidates = [
                            resource
                            for resource in active_resources
                            if resource.item.tier == CrossStoreTier.DEVICE
                            and not resource.item.protected
                            and self._resource_identity(resource) not in excluded_roots
                        ]
                    if not candidates:
                        raise MemoryError("insufficient cross-store victims")
                    selected = None
                    selected_actions = None
                    for resource in sorted(
                        candidates,
                        key=lambda candidate: self.policy.victim_key(candidate.item),
                    ):
                        can_demote = (
                            resource.demote is not None
                            and resource.item.demotable
                            and self.budget.snapshot().host_used_bytes
                            + resource.item.resident_bytes
                            <= self.budget.host_limit_bytes
                        )
                        if can_demote:
                            selected = resource
                            selected_actions = ((resource, "demote"),)
                            break
                        closure = self._eviction_closure(
                            resource,
                            active_resources,
                        )
                        if all(
                            item.item.evictable and not item.item.protected
                            for item in closure
                        ):
                            selected = resource
                            selected_actions = tuple(
                                (
                                    item,
                                    (
                                        "evict_device"
                                        if item.item.tier == CrossStoreTier.DEVICE
                                        else "evict_host"
                                    ),
                                )
                                for item in closure
                            )
                            break
                        excluded_roots.add(self._resource_identity(resource))
                    if selected is None or selected_actions is None:
                        raise MemoryError("insufficient cross-store victims")
                    if not selection_faulted:
                        self._fault(AllocationFailurePoint.AFTER_VICTIM_SELECTION)
                        selection_faulted = True

                    for resource, mode in selected_actions:
                        inactive_resources.add(self._resource_identity(resource))
                        action = resource.demote if mode == "demote" else resource.evict
                        assert action is not None
                        try:
                            action_result = action()
                        except KeyError:
                            # The victim was detached or replaced after this
                            # snapshot was taken. It is already marked inactive,
                            # so refresh and choose another one instead of
                            # failing an allocation that other valid victims
                            # could still satisfy.
                            stale_victims += 1
                            refresh_resources = True
                            break
                        applied.append((resource, mode, action_result))
                        if mode == "demote":
                            self.budget.demote(resource.item.resident_bytes)
                        elif mode == "evict_device":
                            self.budget.release_device(resource.item.resident_bytes)
                        else:
                            self.budget.release_host(resource.item.resident_bytes)
                    else:
                        if any(
                            resource.item.provenance.value == "exact"
                            for resource, _ in selected_actions
                        ):
                            refresh_resources = True

                if not selection_faulted:
                    self._fault(AllocationFailurePoint.AFTER_VICTIM_SELECTION)
                self._fault(AllocationFailurePoint.AFTER_EVICTION)

                allocation = allocate_backend()
                if allocation is None:
                    raise MemoryError("backend allocation failed")
                self._fault(AllocationFailurePoint.AFTER_ALLOCATION)
                self._fault(AllocationFailurePoint.BEFORE_COMMIT)
                commit_phase = True
                for index, (_, _, action_result) in enumerate(applied):
                    if action_result is not None:
                        committed_action_indexes.add(index)
                        action_result.commit()
                commit_phase = False
                self.budget.commit_device(required_device_bytes)
                reserved = False
                victims = tuple(
                    resource.item for resource, mode, _ in applied if mode != "demote"
                )
                demoted = tuple(
                    resource.item for resource, mode, _ in applied if mode == "demote"
                )
                return ReservationResult(
                    allocation=allocation,
                    victim_ids=tuple(item.object_id for item in victims),
                    demoted_ids=tuple(item.object_id for item in demoted),
                    committed=True,
                    requires_reset=False,
                    victim_items=victims,
                    demoted_items=demoted,
                    irreversible_actions=any(
                        action_result is None for _, _, action_result in applied
                    ),
                    destroyed_bytes=sum(item.resident_bytes for item in victims),
                    demoted_bytes=sum(item.resident_bytes for item in demoted),
                    peak_device_bytes=self.budget.snapshot().peak_device_bytes,
                )
            except (
                AssertionError,
                KeyError,
                MemoryError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                if commit_phase:
                    requires_reset = True
                if allocation is not None:
                    release_allocation(allocation)
                if reserved:
                    self.budget.release_device_reservation(required_device_bytes)
                for index in reversed(range(len(applied))):
                    resource, mode, action_result = applied[index]
                    if action_result is None:
                        continue
                    if index in committed_action_indexes:
                        continue
                    try:
                        action_result.undo()
                        if mode == "demote":
                            self.budget.promote(resource.item.resident_bytes)
                        elif mode == "evict_device":
                            self.budget.restore_device(resource.item.resident_bytes)
                        else:
                            self.budget.restore_host(resource.item.resident_bytes)
                        rolled_back_indexes.add(index)
                    except (
                        AssertionError,
                        KeyError,
                        MemoryError,
                        RuntimeError,
                        ValueError,
                    ):
                        requires_reset = True
                victims = tuple(
                    resource.item
                    for index, (resource, mode, _) in enumerate(applied)
                    if mode != "demote" and index not in rolled_back_indexes
                )
                demoted = tuple(
                    resource.item
                    for index, (resource, mode, _) in enumerate(applied)
                    if mode == "demote" and index not in rolled_back_indexes
                )
                return ReservationResult(
                    allocation=None,
                    victim_ids=tuple(item.object_id for item in victims),
                    demoted_ids=tuple(item.object_id for item in demoted),
                    committed=False,
                    requires_reset=requires_reset,
                    victim_items=victims,
                    demoted_items=demoted,
                    irreversible_actions=any(
                        action_result is None for _, _, action_result in applied
                    ),
                    destroyed_bytes=sum(item.resident_bytes for item in victims),
                    demoted_bytes=sum(item.resident_bytes for item in demoted),
                    peak_device_bytes=self.budget.snapshot().peak_device_bytes,
                    rolled_back_ids=tuple(
                        applied[index][0].item.object_id
                        for index in sorted(rolled_back_indexes)
                    ),
                    failure=f"{type(exc).__name__}: {exc}",
                )

    def _fault(self, point: AllocationFailurePoint) -> None:
        if self.fault_injector is not None:
            self.fault_injector(point)

    @staticmethod
    def _resource_identity(
        resource: CrossStoreResource,
    ) -> tuple[str, int, str]:
        return (
            resource.item.object_id,
            resource.item.generation,
            resource.item.provenance.value,
        )

    @staticmethod
    def _eviction_closure(
        root: CrossStoreResource,
        resources: tuple[CrossStoreResource, ...],
    ) -> tuple[CrossStoreResource, ...]:
        by_id = {resource.item.object_id: resource for resource in resources}
        if len(by_id) != len(resources):
            raise ValueError("cross-store object_id values must be unique")
        dependents: dict[str, list[str]] = {}
        for resource in resources:
            for dependency in resource.item.dependencies:
                if dependency in by_id:
                    dependents.setdefault(dependency, []).append(
                        resource.item.object_id
                    )
        ordered: list[CrossStoreResource] = []
        visited: set[str] = set()

        def visit(object_id: str) -> None:
            if object_id in visited:
                return
            visited.add(object_id)
            for dependent in sorted(dependents.get(object_id, ())):
                visit(dependent)
            ordered.append(by_id[object_id])

        visit(root.item.object_id)
        return tuple(ordered)
