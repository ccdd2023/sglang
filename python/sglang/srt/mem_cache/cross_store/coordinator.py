from __future__ import annotations

import threading
from typing import Any

from .allocator import (
    AllocationFailurePoint,
    CrossStoreAllocator,
    ReservationResult,
)
from .budget import CrossStoreBudget
from .policy import CrossStorePolicy, PolicyKind


class CrossStoreCoordinator:
    def __init__(
        self,
        tree_cache: Any,
        *,
        bytes_per_token: int,
        host_budget_bytes: int,
    ) -> None:
        if bytes_per_token <= 0 or host_budget_bytes < 0:
            raise ValueError("invalid cross-store coordinator budget")
        self.tree_cache = tree_cache
        self.bytes_per_token = bytes_per_token
        self.host_budget_bytes = host_budget_bytes
        self._budget: CrossStoreBudget | None = None
        self._allocation_lock = threading.RLock()
        self._allocating = False
        self._test_reservation_failure_consumed = False

    def allocate_tokens(
        self,
        num_tokens: int,
        *,
        requester: str = "approximate",
    ) -> ReservationResult:
        return self._run_allocation(
            num_tokens,
            requester=requester,
            allocate_backend=lambda: (
                self.tree_cache.token_to_kv_pool_allocator.alloc(num_tokens)
            ),
            release_allocation=lambda allocation: (
                self.tree_cache.token_to_kv_pool_allocator.free(allocation)
            ),
        )

    def make_room(
        self,
        num_tokens: int,
        *,
        requester: str,
    ) -> ReservationResult:
        marker = object()
        return self._run_allocation(
            num_tokens,
            requester=requester,
            allocate_backend=lambda: (
                marker
                if self.tree_cache.token_to_kv_pool_allocator.available_size()
                >= num_tokens
                else None
            ),
            release_allocation=lambda allocation: None,
        )

    def _run_allocation(
        self,
        num_tokens: int,
        *,
        requester: str,
        allocate_backend,
        release_allocation,
    ) -> ReservationResult:
        with self._allocation_lock:
            if self._allocating:
                raise RuntimeError("nested cross-store allocation is not supported")
            self._allocating = True
            try:
                return self._allocate_tokens(
                    num_tokens,
                    requester=requester,
                    allocate_backend=allocate_backend,
                    release_allocation=release_allocation,
                )
            finally:
                self._allocating = False

    def _allocate_tokens(
        self,
        num_tokens: int,
        *,
        requester: str,
        allocate_backend,
        release_allocation,
    ) -> ReservationResult:
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        token_allocator = self.tree_cache.token_to_kv_pool_allocator
        manager = self.tree_cache.approx_kv
        approx_store = manager.store
        exact_tokens = int(self.tree_cache.total_size())
        approx_device_tokens = int(approx_store.device_owned_tokens)
        available_tokens = int(token_allocator.available_size())
        capacity_tokens = int(
            getattr(
                token_allocator,
                "size_full",
                getattr(token_allocator, "size", 0),
            )
        )
        if capacity_tokens <= 0:
            capacity_tokens = available_tokens + exact_tokens + approx_device_tokens
        device_limit_bytes = capacity_tokens * self.bytes_per_token
        if self._budget is None:
            self._budget = CrossStoreBudget(
                device_limit_bytes=device_limit_bytes,
                host_limit_bytes=self.host_budget_bytes,
            )
        elif self._budget.device_limit_bytes != device_limit_bytes:
            raise RuntimeError("cross-store device capacity changed during a run")
        self._budget.reconcile_usage(
            device_bytes=(capacity_tokens - available_tokens) * self.bytes_per_token,
            host_bytes=approx_store.host_owned_bytes,
        )
        policy = CrossStorePolicy(
            PolicyKind.S4_HIERARCHICAL
            if self.tree_cache.eviction_policy == "hierarchical"
            else PolicyKind.S0_LRU
        )

        def resources():
            return (
                *self.tree_cache.cross_store_resources(self.bytes_per_token),
                *manager.cross_store_resources(),
            )

        initial_resources = resources()
        fault_injector = None
        if (
            manager.config.cross_store_test_reservation_failure
            and requester == "approximate"
            and not self._test_reservation_failure_consumed
        ):

            def fault_injector(point: AllocationFailurePoint) -> None:
                if point != AllocationFailurePoint.AFTER_RESERVE:
                    return
                self._test_reservation_failure_consumed = True
                raise RuntimeError("test-only injected cross-store reservation failure")

        allocator = CrossStoreAllocator(
            budget=self._budget,
            policy=policy,
            fault_injector=fault_injector,
        )
        result = allocator.allocate(
            required_device_bytes=num_tokens * self.bytes_per_token,
            resources=initial_resources,
            resource_provider=resources,
            allocate_backend=allocate_backend,
            release_allocation=release_allocation,
        )
        for item in result.victim_items:
            manager.record_cross_store_eviction(
                item,
                demoted=False,
                requester=requester,
            )
        for item in result.demoted_items:
            manager.record_cross_store_eviction(
                item,
                demoted=True,
                requester=requester,
            )
        manager.record_cross_store_result(result)
        if not result.committed:
            manager.record_cross_store_reservation_failure(result.requires_reset)
        return result
