from __future__ import annotations

import time
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

from sglang.srt.mem_cache.approx_kv.config import ApproxKVFeatureConfig
from sglang.srt.mem_cache.approx_kv.manager import ApproxKVManager
from sglang.srt.mem_cache.approx_kv.radix_backend import DeviceKVRef
from sglang.srt.mem_cache.approx_kv.request import parse_request_metadata
from sglang.srt.mem_cache.approx_kv.runtime import (
    allocate_recovery_slots,
    pin_reuse_sources,
    resolve_reuse_spans,
)
from sglang.srt.mem_cache.approx_kv.store import (
    ApproxKVSegmentStore,
    ResidencyLoadResult,
)
from sglang.srt.mem_cache.approx_kv.types import (
    KVSegmentHandle,
    KVSegmentKey,
    ResidencyTier,
    SegmentKind,
    token_ids_hash,
)
from sglang.srt.mem_cache.common import evict_from_tree_cache
from sglang.srt.mem_cache.cross_store import (
    AllocationFailurePoint,
    AppliedAction,
    CrossStoreAllocator,
    CrossStoreBudget,
    CrossStoreCoordinator,
    CrossStoreKind,
    CrossStoreObject,
    CrossStoreObjectGraph,
    CrossStorePolicy,
    CrossStoreResource,
    CrossStoreTier,
    ObjectProvenance,
    PolicyKind,
    ReservationResult,
)
from sglang.srt.mem_cache.cross_store.class_order import s4_next_use_key
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode


def item(
    object_id: str,
    *,
    kind: CrossStoreKind = CrossStoreKind.EXACT_VARIANT,
    tier: CrossStoreTier = CrossStoreTier.DEVICE,
    bytes_: int = 100,
    dependencies: frozenset[str] = frozenset(),
    demotable: bool = False,
    pinned: bool = False,
) -> CrossStoreObject:
    return CrossStoreObject(
        object_id=object_id,
        kind=kind,
        tier=tier,
        provenance=(
            ObjectProvenance.EXACT
            if kind == CrossStoreKind.EXACT_VARIANT
            else ObjectProvenance.APPROXIMATE
        ),
        token_count=1,
        resident_bytes=bytes_,
        event_ordinal=0,
        dependencies=dependencies,
        demotable=demotable,
        pinned=pinned,
    )


class TestCrossStoreObjectGraph(unittest.TestCase):
    def test_eviction_closure_removes_dependents_without_orphans(self):
        graph = CrossStoreObjectGraph()
        graph.register(item("base", kind=CrossStoreKind.CANONICAL_BASE))
        graph.register(
            item(
                "delta",
                kind=CrossStoreKind.DELTA,
                dependencies=frozenset({"base"}),
            )
        )
        graph.register(
            item(
                "adapter",
                kind=CrossStoreKind.PRECOMPUTED_ADAPTER,
                dependencies=frozenset({"delta"}),
            )
        )
        closure = graph.eviction_closure("base")
        self.assertEqual(
            {entry.object_id for entry in closure}, {"base", "delta", "adapter"}
        )
        graph.remove_closure("base")
        self.assertEqual(graph.values(), ())

    def test_missing_dependency_is_rejected(self):
        graph = CrossStoreObjectGraph()
        with self.assertRaises(KeyError):
            graph.register(item("child", dependencies=frozenset({"missing"})))

    def test_event_clock_is_shared_across_tiers(self):
        graph = CrossStoreObjectGraph()
        device = graph.register(item("device"))
        host = graph.register(
            item("host", kind=CrossStoreKind.HOST_COPY, tier=CrossStoreTier.HOST)
        )
        self.assertLess(device.event_ordinal, host.event_ordinal)
        touched = graph.touch("device")
        self.assertGreater(touched.event_ordinal, host.event_ordinal)


class TestCrossStorePolicy(unittest.TestCase):
    def test_s4_protects_canonical_base_over_scratch(self):
        policy = CrossStorePolicy(PolicyKind.S4_HIERARCHICAL)
        scratch = item("scratch", kind=CrossStoreKind.MATERIALIZATION_SCRATCH)
        base = item("base", kind=CrossStoreKind.CANONICAL_BASE)
        self.assertLess(policy.victim_key(scratch), policy.victim_key(base))

    def test_unknown_next_use_is_conservatively_evicted_first(self):
        self.assertLess(s4_next_use_key(None), s4_next_use_key(10))


class TestCrossStoreAllocator(unittest.TestCase):
    def test_demotes_before_evicting_when_host_has_space(self):
        budget = CrossStoreBudget(device_limit_bytes=200, host_limit_bytes=200)
        budget.seed_usage(device_bytes=200)
        actions = []
        resource = CrossStoreResource(
            item=item(
                "approx",
                kind=CrossStoreKind.PRECOMPUTED_ADAPTER,
                demotable=True,
            ),
            evict=lambda: actions.append("evict"),
            demote=lambda: actions.append("demote"),
        )
        allocator = CrossStoreAllocator(
            budget=budget,
            policy=CrossStorePolicy(PolicyKind.S0_LRU),
        )
        result = allocator.allocate(
            required_device_bytes=100,
            resources=(resource,),
            allocate_backend=lambda: object(),
            release_allocation=lambda allocation: None,
        )
        self.assertTrue(result.committed)
        self.assertEqual(result.demoted_ids, ("approx",))
        self.assertEqual(actions, ["demote"])
        self.assertEqual(budget.snapshot().host_used_bytes, 100)

    def test_fault_injection_rolls_back_reversible_actions(self):
        for point in AllocationFailurePoint:
            with self.subTest(point=point):
                budget = CrossStoreBudget(
                    device_limit_bytes=200,
                    host_limit_bytes=0,
                )
                budget.seed_usage(device_bytes=200)
                action_state = {"evicted": False}

                def evict():
                    action_state["evicted"] = True

                    def undo():
                        action_state["evicted"] = False

                    return AppliedAction(undo=undo)

                resource = CrossStoreResource(item=item("exact"), evict=evict)

                def inject(observed):
                    if observed == point:
                        raise RuntimeError("injected")

                allocator = CrossStoreAllocator(
                    budget=budget,
                    policy=CrossStorePolicy(PolicyKind.S0_LRU),
                    fault_injector=inject,
                )
                result = allocator.allocate(
                    required_device_bytes=100,
                    resources=(resource,),
                    allocate_backend=lambda: object(),
                    release_allocation=lambda allocation: None,
                )
                self.assertFalse(result.committed)
                self.assertFalse(action_state["evicted"])
                self.assertEqual(budget.snapshot().device_used_bytes, 200)
                self.assertEqual(budget.snapshot().device_reserved_bytes, 0)

    def test_assertion_error_after_reserve_uses_rollback_path(self):
        budget = CrossStoreBudget(device_limit_bytes=200, host_limit_bytes=0)
        budget.seed_usage(device_bytes=100)

        def inject(point):
            if point == AllocationFailurePoint.AFTER_RESERVE:
                raise AssertionError("high-confidence invariant")

        result = CrossStoreAllocator(
            budget=budget,
            policy=CrossStorePolicy(PolicyKind.S0_LRU),
            fault_injector=inject,
        ).allocate(
            required_device_bytes=50,
            resources=(),
            allocate_backend=lambda: object(),
            release_allocation=lambda allocation: None,
        )
        self.assertFalse(result.committed)
        self.assertIn("AssertionError", result.failure)
        self.assertEqual(budget.snapshot().device_reserved_bytes, 0)
        self.assertEqual(budget.snapshot().device_used_bytes, 100)

    def test_reservation_failure_degrades_to_dense_fallback(self):
        """A failed reservation must degrade, and be counted as a fallback.

        The allocator rollback is covered above. This covers the rest of the
        chain: allocate_recovery_slots must turn a non-committed reservation
        into None so the caller goes dense, and it must attribute the
        fallback to that reservation failure rather than leaving the
        explicit counter silent.
        """
        from sglang.srt.mem_cache.approx_kv.config import ApproxKVFeatureConfig
        from sglang.srt.mem_cache.approx_kv.manager import ApproxKVManager
        from sglang.srt.mem_cache.approx_kv.runtime import allocate_recovery_slots

        manager = ApproxKVManager(
            ApproxKVFeatureConfig(
                core_enabled=True,
                cross_store_enabled=True,
                cross_store_bytes_per_token=16,
            )
        )
        fallbacks = []
        manager.metrics_collector = SimpleNamespace(
            increment_approx_kv_fallback=lambda reason, num_tokens: (
                fallbacks.append((reason, num_tokens))
            ),
            increment_approx_kv_request=lambda operation, outcome: None,
        )
        refused = ReservationResult(
            allocation=None,
            victim_ids=(),
            demoted_ids=(),
            committed=False,
            requires_reset=False,
        )
        manager.cross_store_coordinator = lambda tree_cache: SimpleNamespace(
            allocate_tokens=lambda num_tokens: refused
        )
        allocator = _FakeTokenAllocator()
        tree = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            approx_kv=manager,
            cross_store_resources=lambda bytes_per_token: (),
        )

        self.assertIsNone(allocate_recovery_slots(tree, 8))

        self.assertEqual(fallbacks, [("cross_store_reservation_failed", 8)])
        # Nothing may be handed out when the reservation was refused.
        self.assertEqual(allocator.next_index, 100)
        self.assertEqual(allocator.freed, [])

    def test_irreversible_victim_remains_accounted_on_failure(self):
        budget = CrossStoreBudget(device_limit_bytes=200, host_limit_bytes=0)
        budget.seed_usage(device_bytes=200)
        resource = CrossStoreResource(item=item("exact"), evict=lambda: None)
        allocator = CrossStoreAllocator(
            budget=budget,
            policy=CrossStorePolicy(PolicyKind.S0_LRU),
        )
        result = allocator.allocate(
            required_device_bytes=100,
            resources=(resource,),
            allocate_backend=lambda: None,
            release_allocation=lambda allocation: None,
        )
        self.assertFalse(result.committed)
        self.assertFalse(result.requires_reset)
        self.assertTrue(result.irreversible_actions)
        self.assertEqual(result.destroyed_bytes, 100)
        self.assertEqual(budget.snapshot().device_used_bytes, 100)
        self.assertEqual(budget.snapshot().device_reserved_bytes, 0)

    def test_dependency_closure_evicts_dependents_before_base(self):
        budget = CrossStoreBudget(device_limit_bytes=200, host_limit_bytes=0)
        budget.seed_usage(device_bytes=200)
        actions = []
        base = CrossStoreResource(
            item=item("base"),
            evict=lambda: actions.append("base"),
        )
        dependent = CrossStoreResource(
            item=item("dependent", dependencies=frozenset({"base"})),
            evict=lambda: actions.append("dependent"),
        )
        result = CrossStoreAllocator(
            budget=budget,
            policy=CrossStorePolicy(PolicyKind.S0_LRU),
        ).allocate(
            required_device_bytes=100,
            resources=(base, dependent),
            allocate_backend=lambda: object(),
            release_allocation=lambda allocation: None,
        )
        self.assertTrue(result.committed)
        self.assertEqual(actions, ["dependent", "base"])
        self.assertEqual(result.victim_ids, ("dependent", "base"))

    def test_unrelated_orphan_does_not_block_device_victim(self):
        budget = CrossStoreBudget(device_limit_bytes=100, host_limit_bytes=100)
        budget.seed_usage(device_bytes=100, host_bytes=100)
        actions = []
        root = CrossStoreResource(
            item=item("root"),
            evict=lambda: actions.append("root"),
        )
        unrelated_orphan = CrossStoreResource(
            item=item(
                "orphan",
                tier=CrossStoreTier.HOST,
                dependencies=frozenset({"missing"}),
            ),
            evict=lambda: actions.append("orphan"),
        )
        result = CrossStoreAllocator(
            budget=budget,
            policy=CrossStorePolicy(PolicyKind.S0_LRU),
        ).allocate(
            required_device_bytes=100,
            resources=(root, unrelated_orphan),
            allocate_backend=lambda: object(),
            release_allocation=lambda allocation: None,
        )
        self.assertTrue(result.committed)
        self.assertEqual(actions, ["root"])

    def test_protected_dependent_keeps_base_out_of_closure(self):
        budget = CrossStoreBudget(device_limit_bytes=200, host_limit_bytes=0)
        budget.seed_usage(device_bytes=200)
        actions = []
        base = CrossStoreResource(
            item=item("base"),
            evict=lambda: actions.append("base"),
        )
        dependent = CrossStoreResource(
            item=item(
                "dependent",
                dependencies=frozenset({"base"}),
                pinned=True,
            ),
            evict=lambda: actions.append("dependent"),
        )
        result = CrossStoreAllocator(
            budget=budget,
            policy=CrossStorePolicy(PolicyKind.S0_LRU),
        ).allocate(
            required_device_bytes=100,
            resources=(base, dependent),
            allocate_backend=lambda: object(),
            release_allocation=lambda allocation: None,
        )
        self.assertFalse(result.committed)
        self.assertEqual(actions, [])

    def test_coordinator_rejects_nested_allocation_before_mutation(self):
        coordinator = CrossStoreCoordinator(
            object(),
            bytes_per_token=1,
            host_budget_bytes=0,
        )
        coordinator._allocating = True
        with self.assertRaisesRegex(RuntimeError, "nested"):
            coordinator.allocate_tokens(1)

    def test_test_only_reservation_failure_is_one_shot_and_requester_scoped(self):
        class Allocator:
            size_full = 8

            def __init__(self):
                self.available = self.size_full

            def available_size(self):
                return self.available

            def alloc(self, num_tokens):
                if self.available < num_tokens:
                    return None
                start = self.size_full - self.available
                self.available -= num_tokens
                return torch.arange(start, start + num_tokens, dtype=torch.int64)

            def free(self, indices):
                self.available += len(indices)

        allocator = Allocator()
        failures = []
        config = ApproxKVFeatureConfig(
            core_enabled=True,
            cross_store_enabled=True,
            cross_store_bytes_per_token=1,
            test_mode_enabled=True,
            cross_store_test_reservation_failure=True,
        )
        store = SimpleNamespace(
            device_owned_tokens=0,
            host_owned_bytes=0,
        )
        manager = SimpleNamespace(
            config=config,
            store=store,
            cross_store_resources=lambda: (),
            record_cross_store_eviction=lambda *args, **kwargs: None,
            record_cross_store_result=lambda result: None,
            record_cross_store_reservation_failure=failures.append,
        )
        tree = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            approx_kv=manager,
            eviction_policy="lru",
            total_size=lambda: 0,
            cross_store_resources=lambda bytes_per_token: (),
        )
        coordinator = CrossStoreCoordinator(
            tree,
            bytes_per_token=1,
            host_budget_bytes=0,
        )

        # The injection is scoped to approximate recovery, not exact pressure.
        exact = coordinator.allocate_tokens(1, requester="exact")
        self.assertTrue(exact.committed)

        injected = coordinator.allocate_tokens(1, requester="approximate")
        self.assertFalse(injected.committed)
        self.assertIn("test-only injected", injected.failure)
        self.assertEqual(failures, [False])

        # It is one-shot: the next approximate allocation behaves normally.
        recovered = coordinator.allocate_tokens(1, requester="approximate")
        self.assertTrue(recovered.committed)
        self.assertEqual(failures, [False])


class TestApproxStoreByteBudget(unittest.TestCase):
    def test_orphan_count_detects_missing_dependency(self):
        store = ApproxKVSegmentStore(bytes_per_token=1)
        base_key = self._key("base")
        dependent_key = self._key("dependent")
        store.register(
            key=base_key,
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref=object(),
            resident_bytes=1,
            object_id="base",
        )
        store.register(
            key=dependent_key,
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref=object(),
            resident_bytes=1,
            object_id="dependent",
            dependencies=frozenset({"base"}),
        )
        self.assertEqual(store.orphan_count, 0)

        # Corrupt only the object index to prove the gauge is computed rather
        # than a hard-coded zero. Normal store operations prevent this state.
        with store._lock:
            store._object_keys.pop("base")
        self.assertEqual(store.orphan_count, 1)

    def _key(self, name: str) -> KVSegmentKey:
        tokens = (1,)
        return KVSegmentKey(
            content_hash=name,
            token_hash=token_ids_hash(tokens),
            token_count=1,
            model_fingerprint="model",
            cache_dtype="fp16",
            kind=SegmentKind.ARTIFACT,
        )

    def test_device_byte_budget_evicts_oldest_device_record(self):
        released = []
        store = ApproxKVSegmentStore(max_device_bytes=100)
        store.register(
            key=self._key("a"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="a",
            resident_bytes=60,
            release_backend=lambda ref, tier: released.append((ref, tier)),
        )
        store.register(
            key=self._key("b"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="b",
            resident_bytes=60,
            release_backend=lambda ref, tier: released.append((ref, tier)),
        )
        self.assertEqual(store.record_count, 1)
        self.assertEqual(store.device_owned_bytes, 60)
        self.assertEqual(released, [("a", ResidencyTier.DEVICE)])

    def test_cross_store_resource_releases_physical_record(self):
        released = []
        store = ApproxKVSegmentStore()
        store.register(
            key=self._key("a"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="a",
            resident_bytes=60,
            release_backend=lambda ref, tier: released.append((ref, tier)),
        )
        resource = store.cross_store_resources()[0]
        applied = resource.evict()
        self.assertIsNone(applied)
        self.assertEqual(store.record_count, 0)
        self.assertEqual(released, [("a", ResidencyTier.DEVICE)])

    def test_device_pressure_does_not_evict_host_record_first(self):
        released = []
        store = ApproxKVSegmentStore(
            max_device_bytes=100,
            max_host_bytes=100,
        )
        store.register(
            key=self._key("host"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.HOST,
            backend_ref="host",
            resident_bytes=60,
            release_backend=lambda ref, tier: released.append((ref, tier)),
        )
        store.register(
            key=self._key("device-a"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="device-a",
            resident_bytes=60,
            release_backend=lambda ref, tier: released.append((ref, tier)),
        )
        store.register(
            key=self._key("device-b"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="device-b",
            resident_bytes=60,
            release_backend=lambda ref, tier: released.append((ref, tier)),
        )
        self.assertIsNotNone(store.lookup(self._key("host")))
        self.assertEqual(released, [("device-a", ResidencyTier.DEVICE)])

    def test_thousand_register_release_cycles_leave_no_state(self):
        store = ApproxKVSegmentStore()
        for index in range(1000):
            key = self._key(f"cycle-{index}")
            handle = store.register(
                key=key,
                token_ids=(1,),
                source_start=0,
                residency=ResidencyTier.DEVICE,
                backend_ref=index,
                resident_bytes=1,
            )
            self.assertTrue(store.release(handle))
        self.assertEqual(store.record_count, 0)
        self.assertEqual(store.lease_count, 0)
        self.assertEqual(store.device_owned_tokens, 0)
        self.assertEqual(store.device_owned_bytes, 0)

    def test_manager_exposes_real_device_to_host_demotion(self):
        released = []

        class HostBackend:
            def export_to_host(self, device_ref):
                return ResidencyLoadResult(
                    backend_ref=("host", device_ref),
                    release_backend=lambda ref, tier: None,
                    bytes_transferred=8,
                )

            def load_for_rollback(self, handle, target_tier):
                return ResidencyLoadResult(
                    backend_ref="device-restored",
                    release_backend=lambda ref, tier: released.append((ref, tier)),
                    num_tokens=1,
                    bytes_transferred=8,
                )

        manager = ApproxKVManager(
            config=ApproxKVFeatureConfig(
                core_enabled=True,
                host_residency_enabled=True,
                cross_store_bytes_per_token=8,
            )
        )
        manager.bind_residency_backend(HostBackend())
        manager.register_segment(
            key=self._key("demote"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="device",
            release_backend=lambda ref, tier: released.append((ref, tier)),
            resident_bytes=8,
            object_id="demote",
            object_kind=CrossStoreKind.CANONICAL_BASE,
        )

        resource = manager.cross_store_resources()[0]
        self.assertTrue(resource.item.demotable)
        self.assertIsNotNone(resource.demote)
        action = resource.demote()
        self.assertIsInstance(action, AppliedAction)
        handle = manager.store.handle_for_object_id("demote")
        self.assertEqual(handle.residency, ResidencyTier.HOST)
        action.undo()
        handle = manager.store.handle_for_object_id("demote")
        self.assertEqual(handle.residency, ResidencyTier.DEVICE)
        self.assertEqual(released, [("device", ResidencyTier.DEVICE)])

    def test_failed_allocation_rolls_back_real_demotion(self):
        class HostBackend:
            def export_to_host(self, device_ref):
                return ResidencyLoadResult(
                    backend_ref=("host", device_ref),
                    release_backend=lambda ref, tier: None,
                    bytes_transferred=8,
                )

            def load_for_rollback(self, handle, target_tier):
                return ResidencyLoadResult(
                    backend_ref="device-restored",
                    release_backend=lambda ref, tier: None,
                    num_tokens=1,
                    bytes_transferred=8,
                )

        manager = ApproxKVManager(
            config=ApproxKVFeatureConfig(
                core_enabled=True,
                host_residency_enabled=True,
                cross_store_bytes_per_token=8,
            )
        )
        manager.bind_residency_backend(HostBackend())
        manager.register_segment(
            key=self._key("rollback-demote"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="device",
            resident_bytes=8,
            object_id="rollback-demote",
            object_kind=CrossStoreKind.CANONICAL_BASE,
        )
        budget = CrossStoreBudget(device_limit_bytes=8, host_limit_bytes=8)
        budget.seed_usage(device_bytes=8)
        result = CrossStoreAllocator(
            budget=budget,
            policy=CrossStorePolicy(PolicyKind.S0_LRU),
        ).allocate(
            required_device_bytes=8,
            resources=manager.cross_store_resources(),
            allocate_backend=lambda: None,
            release_allocation=lambda allocation: None,
        )
        self.assertFalse(result.committed)
        self.assertEqual(result.demoted_ids, ())
        self.assertEqual(result.rolled_back_ids, ("rollback-demote",))
        self.assertFalse(result.irreversible_actions)
        self.assertEqual(
            manager.store.handle_for_object_id("rollback-demote").residency,
            ResidencyTier.DEVICE,
        )
        self.assertEqual(budget.snapshot().device_used_bytes, 8)
        self.assertEqual(budget.snapshot().host_used_bytes, 0)

    def test_commit_failure_does_not_invoke_undo(self):
        budget = CrossStoreBudget(device_limit_bytes=100, host_limit_bytes=0)
        budget.seed_usage(device_bytes=100)
        state = {"committed": False, "undone": False}

        def evict():
            def commit():
                state["committed"] = True
                raise RuntimeError("commit failed")

            def undo():
                state["undone"] = True

            return AppliedAction(undo=undo, commit=commit)

        result = CrossStoreAllocator(
            budget=budget,
            policy=CrossStorePolicy(PolicyKind.S0_LRU),
        ).allocate(
            required_device_bytes=100,
            resources=(CrossStoreResource(item=item("commit"), evict=evict),),
            allocate_backend=lambda: object(),
            release_allocation=lambda allocation: None,
        )
        self.assertFalse(result.committed)
        self.assertTrue(result.requires_reset)
        self.assertTrue(state["committed"])
        self.assertFalse(state["undone"])

    def test_failed_replacement_restores_previous_record(self):
        released = []
        store = ApproxKVSegmentStore(max_device_bytes=4)
        key = self._key("replace")
        previous = store.register(
            key=key,
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="old",
            release_backend=lambda ref, tier: released.append((ref, tier)),
            resident_bytes=4,
        )
        with self.assertRaises(RuntimeError):
            store.register(
                key=key,
                token_ids=(1,),
                source_start=0,
                residency=ResidencyTier.DEVICE,
                backend_ref="new",
                release_backend=lambda ref, tier: released.append((ref, tier)),
                resident_bytes=8,
            )
        restored = store.lookup(key)
        self.assertEqual(restored.generation, previous.generation)
        self.assertEqual(restored.backend_ref, "old")
        self.assertEqual(released, [("new", ResidencyTier.DEVICE)])

    def test_replacing_depended_on_object_id_is_rejected(self):
        released = []
        store = ApproxKVSegmentStore()
        base_key = self._key("replace-base")
        base = store.register(
            key=base_key,
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="base",
            object_id="base",
        )
        store.register(
            key=self._key("replace-dependent"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="dependent",
            object_id="dependent",
            dependencies=frozenset({"base"}),
        )
        with self.assertRaisesRegex(RuntimeError, "dependents"):
            store.register(
                key=base_key,
                token_ids=(1,),
                source_start=0,
                residency=ResidencyTier.DEVICE,
                backend_ref="new-base",
                release_backend=lambda ref, tier: released.append((ref, tier)),
                object_id="renamed-base",
            )
        self.assertEqual(released, [("new-base", ResidencyTier.DEVICE)])
        self.assertTrue(store.is_current(base))
        self.assertEqual(store.orphan_count, 0)

    def test_manager_capacity_rejection_releases_backend_once(self):
        released = []
        manager = ApproxKVManager(
            config=ApproxKVFeatureConfig(core_enabled=True),
            store=ApproxKVSegmentStore(max_device_bytes=4),
        )
        handle = manager.register_segment(
            key=self._key("too-large"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="too-large",
            release_backend=lambda ref, tier: released.append((ref, tier)),
            resident_bytes=8,
        )
        self.assertIsNone(handle)
        self.assertEqual(released, [("too-large", ResidencyTier.DEVICE)])
        self.assertEqual(manager.store.record_count, 0)

    def test_host_budget_evicts_previous_host_record(self):
        released = []
        manager = ApproxKVManager(
            config=ApproxKVFeatureConfig(
                core_enabled=True,
                host_residency_enabled=True,
                cross_store_enabled=True,
                cross_store_bytes_per_token=8,
                cross_store_host_budget_bytes=8,
            )
        )
        for name in ("first", "second"):
            manager.register_segment(
                key=self._key(f"host-{name}"),
                token_ids=(1,),
                source_start=0,
                residency=ResidencyTier.HOST,
                backend_ref=name,
                release_backend=lambda ref, tier: released.append((ref, tier)),
                resident_bytes=8,
                object_id=f"host-{name}",
            )
        self.assertEqual(manager.store.record_count, 1)
        self.assertEqual(manager.store.host_owned_bytes, 8)
        self.assertEqual(released, [("first", ResidencyTier.HOST)])

    def test_production_store_victim_failure_keeps_budget_consistent(self):
        store = ApproxKVSegmentStore(bytes_per_token=100)
        store.register(
            key=self._key("victim"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="device",
            resident_bytes=100,
            object_id="victim",
        )
        budget = CrossStoreBudget(device_limit_bytes=100, host_limit_bytes=0)
        budget.seed_usage(device_bytes=100)
        result = CrossStoreAllocator(
            budget=budget,
            policy=CrossStorePolicy(PolicyKind.S0_LRU),
        ).allocate(
            required_device_bytes=100,
            resources=store.cross_store_resources(),
            allocate_backend=lambda: None,
            release_allocation=lambda allocation: None,
        )
        self.assertFalse(result.committed)
        self.assertTrue(result.irreversible_actions)
        self.assertEqual(store.device_owned_bytes, 0)
        self.assertEqual(budget.snapshot().device_used_bytes, 0)

    def test_eight_hundred_object_eviction_stays_bounded(self):
        store = ApproxKVSegmentStore()
        for index in range(800):
            store.register(
                key=self._key(f"scale-{index}"),
                token_ids=(1,),
                source_start=0,
                residency=ResidencyTier.DEVICE,
                backend_ref=index,
                resident_bytes=1,
                object_id=f"scale-{index}",
            )
        budget = CrossStoreBudget(device_limit_bytes=800, host_limit_bytes=0)
        budget.seed_usage(device_bytes=800)
        started = time.perf_counter()
        result = CrossStoreAllocator(
            budget=budget,
            policy=CrossStorePolicy(PolicyKind.S0_LRU),
        ).allocate(
            required_device_bytes=400,
            resources=store.cross_store_resources(),
            resource_provider=store.cross_store_resources,
            allocate_backend=lambda: object(),
            release_allocation=lambda allocation: None,
        )
        elapsed = time.perf_counter() - started
        self.assertTrue(result.committed)
        self.assertEqual(len(result.victim_ids), 400)
        self.assertLess(elapsed, 1.5)

    def test_pinned_source_is_not_a_cross_store_candidate(self):
        manager = ApproxKVManager(
            config=ApproxKVFeatureConfig(
                core_enabled=True,
                cross_store_enabled=True,
            )
        )
        handle = manager.register_segment(
            key=self._key("pinned"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="device",
            resident_bytes=1,
            object_id="pinned",
        )
        lease = manager.store.pin(handle, ttl_s=10)
        try:
            self.assertTrue(manager.cross_store_resources()[0].item.protected)
        finally:
            manager.store.unpin(lease)

    def test_store_rejects_orphans_and_releases_dependents_first(self):
        released = []
        store = ApproxKVSegmentStore()
        with self.assertRaises(KeyError):
            store.register(
                key=self._key("orphan"),
                token_ids=(1,),
                source_start=0,
                residency=ResidencyTier.DEVICE,
                backend_ref="orphan",
                release_backend=lambda ref, tier: released.append((ref, tier)),
                object_id="orphan",
                dependencies=frozenset({"missing"}),
            )
        self.assertEqual(released, [("orphan", ResidencyTier.DEVICE)])

        base = store.register(
            key=self._key("base"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="base",
            object_id="base",
        )
        dependent = store.register(
            key=self._key("dependent"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref="dependent",
            object_id="dependent",
            dependencies=frozenset({"base"}),
        )
        self.assertFalse(store.release(base))
        self.assertTrue(store.release(dependent))
        self.assertTrue(store.release(base))


class _FakeTokenAllocator:
    device = "cpu"

    def __init__(self, available: int = 0) -> None:
        self.available = available
        self.next_index = 100
        self.freed = []

    def free(self, values):
        released = [int(value) for value in values]
        self.freed.extend(released)
        self.available += len(released)

    def available_size(self):
        return self.available

    def alloc(self, num_tokens):
        if self.available < num_tokens:
            return None
        values = torch.arange(
            self.next_index,
            self.next_index + num_tokens,
            dtype=torch.int64,
        )
        self.next_index += num_tokens
        self.available -= num_tokens
        return values


class TestRadixCrossStoreAdapter(unittest.TestCase):
    def test_leaf_is_exposed_as_exact_resource(self):
        allocator = _FakeTokenAllocator()
        cache = RadixCache.create_simulated(mock_allocator=allocator)
        node = TreeNode()
        node.parent = cache.root_node
        node.key = RadixKey(array("q", [1]))
        node.value = torch.tensor([7], dtype=torch.int64)
        cache.root_node.children[node.key.child_key(1)] = node
        cache.evictable_leaves.add(node)
        cache.evictable_size_ = 1

        resource = cache.cross_store_resources(bytes_per_token=16)[0]
        self.assertEqual(resource.item.resident_bytes, 16)
        self.assertEqual(resource.item.provenance, ObjectProvenance.EXACT)
        self.assertIsNone(resource.evict())
        self.assertEqual(allocator.freed, [7])

    def _leaf_cache(self):
        allocator = _FakeTokenAllocator()
        cache = RadixCache.create_simulated(mock_allocator=allocator)
        node = TreeNode()
        node.parent = cache.root_node
        node.key = RadixKey(array("q", [1]))
        node.value = torch.tensor([7], dtype=torch.int64)
        cache.root_node.children[node.key.child_key(1)] = node
        cache.evictable_leaves.add(node)
        cache.evictable_size_ = 1
        return allocator, cache, node

    def test_unlocked_prefix_is_an_eviction_candidate(self):
        """Baseline for the recovery prefix guard.

        This is why protect_request_prefix exists: an unlocked node that a
        request is already using as its prefix is a perfectly legal victim.
        """
        _, cache, _ = self._leaf_cache()
        self.assertEqual(len(cache.cross_store_resources(bytes_per_token=16)), 1)

    def test_locked_prefix_is_never_offered_as_a_victim(self):
        _, cache, node = self._leaf_cache()
        cache.inc_lock_ref(node)
        self.assertEqual(cache.cross_store_resources(bytes_per_token=16), ())
        cache.dec_lock_ref(node)
        self.assertEqual(len(cache.cross_store_resources(bytes_per_token=16)), 1)

    def test_stale_victim_is_skipped_so_a_valid_victim_still_runs(self):
        """A dead victim must not fail an allocation others could satisfy.

        The snapshot can name a node that was detached after it was taken.
        Aborting on it strands real capacity: the exact-pressure path then
        reports nothing evictable and an ordinary prefill dies with OOM.
        """
        allocator = _FakeTokenAllocator()
        cache = RadixCache.create_simulated(mock_allocator=allocator)
        stale = TreeNode()
        stale.parent = cache.root_node
        stale.key = RadixKey(array("q", [1]))
        stale.value = torch.tensor([7], dtype=torch.int64)
        cache.root_node.children[stale.key.child_key(1)] = stale
        good = TreeNode()
        good.parent = cache.root_node
        good.key = RadixKey(array("q", [2]))
        good.value = torch.tensor([8], dtype=torch.int64)
        cache.root_node.children[good.key.child_key(1)] = good
        cache.evictable_leaves.update({stale, good})
        cache.evictable_size_ = 2

        snapshot = cache.cross_store_resources(bytes_per_token=1)
        # Detach one victim after the snapshot was taken.
        cache.root_node.children.pop(stale.key.child_key(1))

        budget = CrossStoreBudget(device_limit_bytes=2, host_limit_bytes=0)
        budget.seed_usage(device_bytes=2)
        result = CrossStoreAllocator(
            budget=budget,
            policy=CrossStorePolicy(PolicyKind.S0_LRU),
        ).allocate(
            required_device_bytes=1,
            resources=snapshot,
            resource_provider=lambda: cache.cross_store_resources(bytes_per_token=1),
            allocate_backend=lambda: allocator.alloc(1),
            release_allocation=lambda allocation: None,
        )

        self.assertTrue(result.committed)
        self.assertEqual(allocator.freed, [8])
        # The detached node must stop being advertised.
        self.assertNotIn(stale, cache.evictable_leaves)

    def test_stale_victim_raises_keyerror_instead_of_asserting(self):
        """A detached victim must degrade to dense fallback, not crash.

        KeyError is inside the exception tuple the cross-store allocator
        already rolls back on; an AssertionError from _delete_leaf kills the
        scheduler process instead.
        """
        _, cache, node = self._leaf_cache()
        resource = cache.cross_store_resources(bytes_per_token=16)[0]
        cache.root_node.children.clear()
        with self.assertRaises(KeyError):
            resource.evict()

    def test_resource_provider_exposes_parent_after_leaf_eviction(self):
        allocator = _FakeTokenAllocator()
        cache = RadixCache.create_simulated(mock_allocator=allocator)
        parent = TreeNode()
        parent.parent = cache.root_node
        parent.key = RadixKey(array("q", [1]))
        parent.value = torch.tensor([6], dtype=torch.int64)
        cache.root_node.children[parent.key.child_key(1)] = parent
        child = TreeNode()
        child.parent = parent
        child.key = RadixKey(array("q", [2]))
        child.value = torch.tensor([7], dtype=torch.int64)
        parent.children[child.key.child_key(1)] = child
        cache.evictable_leaves.add(child)
        cache.evictable_size_ = 2

        budget = CrossStoreBudget(device_limit_bytes=2, host_limit_bytes=0)
        budget.seed_usage(device_bytes=2)
        provider = lambda: cache.cross_store_resources(bytes_per_token=1)
        result = CrossStoreAllocator(
            budget=budget,
            policy=CrossStorePolicy(PolicyKind.S0_LRU),
        ).allocate(
            required_device_bytes=2,
            resources=provider(),
            resource_provider=provider,
            allocate_backend=lambda: allocator.alloc(2),
            release_allocation=allocator.free,
        )
        self.assertTrue(result.committed)
        self.assertEqual(len(result.victim_ids), 2)
        self.assertEqual(cache.total_size(), 0)


class TestCrossStoreConfig(unittest.TestCase):
    def test_cross_store_requires_core(self):
        with self.assertRaises(ValueError):
            ApproxKVFeatureConfig.from_env(
                {
                    "SGLANG_APPROX_KV_CROSS_STORE": "1",
                }
            )

    def test_cross_store_byte_settings(self):
        config = ApproxKVFeatureConfig.from_env(
            {
                "SGLANG_APPROX_KV_CORE": "1",
                "SGLANG_APPROX_KV_CROSS_STORE": "1",
                "SGLANG_APPROX_KV_BYTES_PER_TOKEN": "16",
                "SGLANG_APPROX_KV_HOST_BUDGET_BYTES": "64",
            }
        )
        self.assertTrue(config.cross_store_enabled)
        self.assertEqual(config.cross_store_bytes_per_token, 16)
        self.assertEqual(config.cross_store_host_budget_bytes, 64)

    def test_persistent_pin_settings_are_gated_and_positive(self):
        defaults = ApproxKVFeatureConfig.from_env({})
        self.assertFalse(defaults.allow_persistent_pins)
        self.assertEqual(defaults.max_persistent_pins, 16)

        enabled = ApproxKVFeatureConfig.from_env(
            {
                "SGLANG_APPROX_KV_ALLOW_PERSISTENT_PINS": "1",
                "SGLANG_APPROX_KV_MAX_PERSISTENT_PINS": "3",
            }
        )
        self.assertTrue(enabled.allow_persistent_pins)
        self.assertEqual(enabled.max_persistent_pins, 3)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            ApproxKVFeatureConfig.from_env(
                {"SGLANG_APPROX_KV_MAX_PERSISTENT_PINS": "0"}
            )

    def test_reservation_failure_injection_requires_both_test_gates(self):
        base = {
            "SGLANG_APPROX_KV_CORE": "1",
            "SGLANG_APPROX_KV_CROSS_STORE": "1",
            "SGLANG_APPROX_KV_TEST_RESERVATION_FAILURE": "1",
        }
        with self.assertRaisesRegex(ValueError, "test_mode_enabled"):
            ApproxKVFeatureConfig.from_env(base)

        config = ApproxKVFeatureConfig.from_env(
            {
                **base,
                "SGLANG_APPROX_KV_TEST_ONLY": "1",
            }
        )
        self.assertTrue(config.test_mode_enabled)
        self.assertTrue(config.cross_store_test_reservation_failure)

    def test_reservation_failure_injection_requires_cross_store(self):
        with self.assertRaisesRegex(ValueError, "cross_store_enabled"):
            ApproxKVFeatureConfig.from_env(
                {
                    "SGLANG_APPROX_KV_CORE": "1",
                    "SGLANG_APPROX_KV_TEST_ONLY": "1",
                    "SGLANG_APPROX_KV_TEST_RESERVATION_FAILURE": "1",
                }
            )

    def test_segment_cross_store_metadata_parses(self):
        metadata = parse_request_metadata(
            {
                "approx_kv": {
                    "operation": "register",
                    "segments": [
                        {
                            "content_hash": "artifact",
                            "target_start": 0,
                            "length": 1,
                            "object_id": "object-1",
                            "object_kind": "canonical_base",
                            "dense_cost_ms": 10,
                            "recovery_cost_ms": 2,
                            "next_use_ordinal": 7,
                        }
                    ],
                }
            }
        )
        segment = metadata.segments[0]
        self.assertEqual(segment.object_id, "object-1")
        self.assertEqual(segment.object_kind, CrossStoreKind.CANONICAL_BASE)
        self.assertEqual(segment.next_use_ordinal, 7)

    def test_legacy_object_kind_and_device_residency_parse(self):
        metadata = parse_request_metadata(
            {
                "approx_kv": {
                    "operation": "register",
                    "segments": [
                        {
                            "content_hash": "legacy",
                            "target_start": 0,
                            "length": 1,
                            "object_kind": "repair_metadata",
                            "residency": "device",
                            "dependencies": ["base"],
                        }
                    ],
                }
            }
        )
        segment = metadata.segments[0]
        self.assertEqual(segment.object_kind, CrossStoreKind.REPAIR_STATE)
        self.assertEqual(segment.residency, ResidencyTier.DEVICE)
        self.assertEqual(segment.dependencies, frozenset({"base"}))

    def test_seeded_header_enters_segment_reuse_path(self):
        tokens = (1, 2, 3)
        key = KVSegmentKey(
            content_hash="seeded",
            token_hash=token_ids_hash(tokens),
            token_count=3,
            model_fingerprint="runtime",
            cache_dtype="auto",
            kind=SegmentKind.ARTIFACT,
        )
        handle = KVSegmentHandle(
            key=key,
            generation=1,
            residency=ResidencyTier.DEVICE,
            source_start=64,
            token_ids=tokens,
            backend_ref=object(),
        )

        class Store:
            def __init__(self):
                self.unpinned = []

            def lookup(self, observed_key):
                return handle if observed_key == key else None

            def pin(self, observed_handle, ttl_s):
                self.asserted_ttl = ttl_s
                return "lease"

            def unpin(self, lease):
                self.unpinned.append(lease)

        store = Store()
        manager = SimpleNamespace(
            store=store,
            ensure_device=lambda observed: observed,
            rope_config=None,
            record_fallback=lambda *args: None,
            record_request=lambda *args: None,
        )
        metadata = parse_request_metadata(
            {
                "approx_kv": {
                    "operation": "reuse",
                    "segments": [
                        {
                            "content_hash": "seeded",
                            "target_start": 64,
                            "length": 3,
                        }
                    ],
                }
            }
        )
        req = SimpleNamespace(
            full_untruncated_fill_ids=[0] * 64 + list(tokens) + [9],
            prefix_indices=torch.arange(64, dtype=torch.int64),
        )
        resolved = resolve_reuse_spans(None, req, metadata, manager)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.restore_length, 3)
        with pin_reuse_sources(manager, resolved) as pinned:
            self.assertTrue(pinned)
        self.assertEqual(store.unpinned, ["lease"])


class TestCrossStoreRuntimeIntegration(unittest.TestCase):
    def test_recovery_prefers_lower_class_approx_victim(self):
        allocator = _FakeTokenAllocator(available=1)
        cache = RadixCache.create_simulated(mock_allocator=allocator)
        cache.eviction_policy = "hierarchical"
        cache.approx_kv.config = ApproxKVFeatureConfig(
            core_enabled=True,
            cross_store_enabled=True,
            cross_store_bytes_per_token=1,
        )

        exact = TreeNode()
        exact.parent = cache.root_node
        exact.key = RadixKey(array("q", [1]))
        exact.value = torch.tensor([7], dtype=torch.int64)
        cache.root_node.children[exact.key.child_key(1)] = exact
        cache.evictable_leaves.add(exact)
        cache.evictable_size_ = 1

        cache.approx_kv.register_segment(
            key=TestApproxStoreByteBudget()._key("approx"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref=DeviceKVRef(torch.tensor([8], dtype=torch.int64)),
            release_backend=lambda ref, tier: allocator.free(ref.indices),
            resident_bytes=1,
            object_kind=CrossStoreKind.PRECOMPUTED_ADAPTER,
        )

        allocated = allocate_recovery_slots(cache, 2)
        self.assertIsNotNone(allocated)
        self.assertEqual(cache.approx_kv.store.record_count, 0)
        self.assertIn(exact, cache.evictable_leaves)
        self.assertIn(8, allocator.freed)

    def test_exact_request_pressure_can_evict_approximate_victim(self):
        allocator = _FakeTokenAllocator(available=0)
        cache = RadixCache.create_simulated(mock_allocator=allocator)
        cache.approx_kv.config = ApproxKVFeatureConfig(
            core_enabled=True,
            cross_store_enabled=True,
            cross_store_bytes_per_token=1,
        )
        cache.approx_kv.register_segment(
            key=TestApproxStoreByteBudget()._key("exact-request-victim"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref=DeviceKVRef(torch.tensor([8], dtype=torch.int64)),
            release_backend=lambda ref, tier: allocator.free(ref.indices),
            resident_bytes=1,
            object_id="exact-request-victim",
            object_kind=CrossStoreKind.PRECOMPUTED_ADAPTER,
        )
        evict_from_tree_cache(cache, 1)
        self.assertEqual(cache.approx_kv.store.record_count, 0)
        self.assertEqual(allocator.available_size(), 1)


class TestCrossStoreResetAccounting(unittest.TestCase):
    def test_budget_reset_clears_usage_and_device_peak(self):
        budget = CrossStoreBudget(device_limit_bytes=1000, host_limit_bytes=1000)
        budget.reserve_device(400)
        budget.commit_device(400)
        self.assertEqual(budget.snapshot().peak_device_bytes, 400)
        budget.release_device(400)
        self.assertEqual(budget.snapshot().peak_device_bytes, 400)

        budget.reset_accounting()
        snapshot = budget.snapshot()
        self.assertEqual(snapshot.peak_device_bytes, 0)
        self.assertEqual(snapshot.device_used_bytes, 0)
        self.assertEqual(snapshot.host_used_bytes, 0)
        self.assertEqual(snapshot.device_reserved_bytes, 0)

    def test_budget_reset_is_rejected_while_a_reservation_is_active(self):
        budget = CrossStoreBudget(device_limit_bytes=1000, host_limit_bytes=0)
        budget.reserve_device(100)
        with self.assertRaisesRegex(RuntimeError, "active reservation"):
            budget.reset_accounting()
        self.assertEqual(budget.snapshot().peak_device_bytes, 100)

    def test_forced_budget_reset_clears_stale_reservation_and_usage(self):
        budget = CrossStoreBudget(device_limit_bytes=1000, host_limit_bytes=1000)
        budget.seed_usage(device_bytes=300, host_bytes=200)
        budget.reserve_device(100)

        budget.reset_accounting(force=True)
        snapshot = budget.snapshot()
        self.assertEqual(snapshot.device_used_bytes, 0)
        self.assertEqual(snapshot.host_used_bytes, 0)
        self.assertEqual(snapshot.device_reserved_bytes, 0)
        self.assertEqual(snapshot.peak_device_bytes, 0)

    def test_coordinator_reset_is_a_noop_without_a_budget(self):
        coordinator = CrossStoreCoordinator(
            object(),
            bytes_per_token=1,
            host_budget_bytes=0,
        )
        self.assertFalse(coordinator.reset_accounting())
        coordinator._budget = CrossStoreBudget(
            device_limit_bytes=100,
            host_limit_bytes=0,
        )
        coordinator._budget.reserve_device(10)
        coordinator._budget.commit_device(10)
        self.assertTrue(coordinator.reset_accounting())
        self.assertEqual(coordinator._budget.snapshot().peak_device_bytes, 0)

    def test_coordinator_reset_is_rejected_during_allocation(self):
        coordinator = CrossStoreCoordinator(
            object(),
            bytes_per_token=1,
            host_budget_bytes=0,
        )
        coordinator._allocating = True
        with self.assertRaisesRegex(RuntimeError, "during an allocation"):
            coordinator.reset_accounting(force=True)

    def test_manager_reset_zeroes_peak_and_reserved_gauges(self):
        class RecordingCollector:
            def __init__(self):
                self.accounting = []
                self.store_state = []

            def set_cross_store_device_accounting(
                self,
                *,
                peak_device_bytes,
                reserved_device_bytes,
            ):
                self.accounting.append((peak_device_bytes, reserved_device_bytes))

            def record_cross_store_result(self, **kwargs):
                self.set_cross_store_device_accounting(
                    peak_device_bytes=kwargs["peak_device_bytes"],
                    reserved_device_bytes=kwargs["reserved_device_bytes"],
                )

            def set_approx_kv_store_state(self, **kwargs):
                self.store_state.append(kwargs)

        collector = RecordingCollector()
        manager = ApproxKVManager(
            ApproxKVFeatureConfig(
                core_enabled=True,
                cross_store_enabled=True,
                cross_store_bytes_per_token=1,
            ),
            metrics_collector=collector,
        )
        coordinator = manager.cross_store_coordinator(object())
        coordinator._budget = CrossStoreBudget(
            device_limit_bytes=1000,
            host_limit_bytes=0,
        )
        coordinator._budget.reserve_device(500)
        coordinator._budget.commit_device(500)
        coordinator._budget.release_device(500)
        manager.record_cross_store_result(
            SimpleNamespace(
                committed=True,
                destroyed_bytes=0,
                peak_device_bytes=500,
                reserved_device_bytes=0,
            )
        )
        self.assertEqual(collector.accounting[-1], (500, 0))

        manager.reset()
        self.assertEqual(collector.accounting[-1], (0, 0))
        self.assertEqual(coordinator._budget.snapshot().peak_device_bytes, 0)
        self.assertEqual(collector.store_state[-1]["records"], 0)

    def test_manager_full_reset_forces_stale_reservation_cleanup(self):
        manager = ApproxKVManager(
            ApproxKVFeatureConfig(
                core_enabled=True,
                cross_store_enabled=True,
                cross_store_bytes_per_token=1,
            )
        )
        coordinator = manager.cross_store_coordinator(object())
        coordinator._budget = CrossStoreBudget(
            device_limit_bytes=1000,
            host_limit_bytes=100,
        )
        coordinator._budget.seed_usage(device_bytes=300, host_bytes=100)
        coordinator._budget.reserve_device(200)

        manager.reset()
        snapshot = coordinator._budget.snapshot()
        self.assertEqual(snapshot.device_used_bytes, 0)
        self.assertEqual(snapshot.host_used_bytes, 0)
        self.assertEqual(snapshot.device_reserved_bytes, 0)
        self.assertEqual(snapshot.peak_device_bytes, 0)


class TestPersistentRegistrationLease(unittest.TestCase):
    def test_pin_until_reset_is_optional_boolean_registration_metadata(self):
        default = parse_request_metadata(
            {
                "approx_kv": {
                    "operation": "register",
                    "segments": [
                        {"content_hash": "artifact", "target_start": 0, "length": 1}
                    ],
                }
            }
        )
        self.assertFalse(default.pin_until_reset)

        pinned = parse_request_metadata(
            {
                "approx_kv": {
                    "operation": "register",
                    "pin_until_reset": True,
                    "segments": [
                        {"content_hash": "artifact", "target_start": 0, "length": 1}
                    ],
                }
            }
        )
        self.assertTrue(pinned.pin_until_reset)

        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            parse_request_metadata(
                {
                    "approx_kv": {
                        "operation": "register",
                        "pin_until_reset": 1,
                        "segments": [
                            {"content_hash": "artifact", "target_start": 0, "length": 1}
                        ],
                    }
                }
            )
        with self.assertRaisesRegex(ValueError, "pin_ttl_s is unsupported"):
            parse_request_metadata(
                {
                    "approx_kv": {
                        "operation": "register",
                        "pin_ttl_s": 3600,
                        "segments": [
                            {
                                "content_hash": "artifact",
                                "target_start": 0,
                                "length": 1,
                            }
                        ],
                    }
                }
            )

    def test_manager_pin_registration_holds_and_reset_releases(self):
        store = ApproxKVSegmentStore(bytes_per_token=1)
        manager = ApproxKVManager(
            ApproxKVFeatureConfig(
                core_enabled=True,
                allow_persistent_pins=True,
            ),
            store=store,
        )
        handle = manager.register_segment(
            key=TestApproxStoreByteBudget()._key("pinned-source"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref=object(),
            resident_bytes=1,
            object_id="pinned-source",
        )
        self.assertEqual(store.lease_count, 0)

        lease = manager.pin_registration(handle)
        self.assertIsNotNone(lease)
        self.assertIsNone(lease.expires_at_s)
        self.assertEqual(store.lease_count, 1)
        self.assertEqual(manager.persistent_lease_count, 1)

        manager.reset()
        self.assertEqual(store.lease_count, 0)
        self.assertEqual(manager.persistent_lease_count, 0)
        self.assertEqual(store.record_count, 0)

    def test_pin_registration_gate_cap_and_duplicate(self):
        manager = ApproxKVManager(ApproxKVFeatureConfig(core_enabled=True))
        handle = manager.register_segment(
            key=TestApproxStoreByteBudget()._key("gate"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref=object(),
            resident_bytes=1,
            object_id="gate",
        )
        with self.assertRaisesRegex(RuntimeError, "pins are disabled"):
            manager.pin_registration(handle)

        store = ApproxKVSegmentStore(bytes_per_token=1)
        capped = ApproxKVManager(
            ApproxKVFeatureConfig(
                core_enabled=True,
                allow_persistent_pins=True,
                max_persistent_pins=1,
            ),
            store=store,
        )
        first = capped.register_segment(
            key=TestApproxStoreByteBudget()._key("first-persistent"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref=object(),
            resident_bytes=1,
            object_id="first-persistent",
        )
        first_lease = capped.pin_registration(first)
        duplicate = capped.pin_registration(first)
        self.assertIs(duplicate, first_lease)
        self.assertEqual(capped.persistent_lease_count, 1)
        self.assertEqual(store.lease_count, 1)

        second = capped.register_segment(
            key=TestApproxStoreByteBudget()._key("second-persistent"),
            token_ids=(1,),
            source_start=0,
            residency=ResidencyTier.DEVICE,
            backend_ref=object(),
            resident_bytes=1,
            object_id="second-persistent",
        )
        with self.assertRaisesRegex(RuntimeError, "pin cap exceeded"):
            capped.pin_registration(second)
        self.assertEqual(capped.persistent_lease_count, 1)
        self.assertEqual(store.lease_count, 1)

        capped.reset()
        self.assertEqual(capped.persistent_lease_count, 0)
        self.assertEqual(store.lease_count, 0)


if __name__ == "__main__":
    unittest.main()
