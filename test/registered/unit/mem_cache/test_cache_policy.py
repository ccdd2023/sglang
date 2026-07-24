from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import torch

from benchmark.approx_kv.run_phase5_scheduler_matrix import (
    build_phase5_trace,
    custom_params_factory,
)
from benchmark.approx_kv.workloads import (
    CacheObject,
)
from benchmark.approx_kv.workloads import CacheObjectKind as WorkloadObjectKind
from benchmark.approx_kv.workloads import (
    ReuseClass,
)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.cache_policy import (
    CacheObjectKind,
    CachePrefetchHint,
    CacheProtectionMetadata,
    CacheProtectionState,
    PrefetchCandidate,
    PrefetchMode,
    parse_cache_prefetch_hints,
    parse_cache_protection_metadata,
    plan_prefetch,
)
from sglang.srt.mem_cache.evict_policy import (
    BeladyStrategy,
    HierarchicalObjectStrategy,
    RecoveryValueStrategy,
    WorkflowStepsStrategy,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams


def metadata(object_id: str, **kwargs) -> CacheProtectionMetadata:
    return CacheProtectionMetadata(object_id=object_id, **kwargs)


def node(*items: CacheProtectionMetadata, last_access_time: float = 0.0):
    state = CacheProtectionState()
    state.update(items)
    return SimpleNamespace(
        cache_protection=state,
        last_access_time=last_access_time,
    )


class TestCacheProtectionMetadata(unittest.TestCase):
    def test_parse_single_object_and_alias(self):
        parsed = parse_cache_protection_metadata(
            {
                "cache_protection": {
                    "object_id": "coder-bundle",
                    "protected_tokens": 1024,
                    "resident_bytes": 4096,
                    "dense_cost_ms": 40.0,
                    "recovery_cost_ms": 10.0,
                    "current_step": 4,
                    "next_use_step": 3,
                    "next_use_request_step": 12,
                    "next_use_distance": 2,
                    "workflow_stage": "coder",
                    "object_kind": "stage_variant",
                    "recoverable_from_lower_tier": True,
                }
            }
        )
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].object_kind, CacheObjectKind.EXACT_VARIANT)
        self.assertEqual(parsed[0].protected_tokens, 1024)
        self.assertEqual(parsed[0].next_use_request_step, 12)
        self.assertEqual(parsed[0].saved_ms, 30.0)
        self.assertTrue(parsed[0].has_future_use)

    def test_parse_multiple_objects(self):
        parsed = parse_cache_protection_metadata(
            {
                "cache_protection": {
                    "objects": [
                        {"object_id": "base", "object_kind": "canonical_base"},
                        {"object_id": "anchor", "object_kind": "anchor"},
                    ]
                }
            }
        )
        self.assertEqual(
            [item.object_kind for item in parsed],
            [CacheObjectKind.CANONICAL_BASE, CacheObjectKind.ANCHOR],
        )

    def test_invalid_boolean_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            parse_cache_protection_metadata(
                {
                    "cache_protection": {
                        "object_id": "bad",
                        "retired": "false",
                    }
                }
            )

    def test_state_update_replaces_same_object(self):
        state = CacheProtectionState()
        state.update((metadata("shared", next_use_step=9),))
        state.update((metadata("shared", next_use_step=2),))
        self.assertEqual(state.objects["shared"].next_use_step, 2)

    def test_parse_prefetch_hints(self):
        hints = parse_cache_prefetch_hints(
            {
                "cache_prefetch": {
                    "object_id": "coder",
                    "next_use_step": 7,
                }
            }
        )
        self.assertEqual(
            hints,
            (CachePrefetchHint("coder", next_use_step=7),),
        )


class TestWorkflowEvictionStrategies(unittest.TestCase):
    def test_steps_only_evicts_retired_then_farthest(self):
        strategy = WorkflowStepsStrategy()
        retired = node(metadata("retired", retired=True))
        far = node(metadata("far", next_use_step=12))
        near = node(metadata("near", next_use_step=2))
        self.assertLess(strategy.get_priority(retired), strategy.get_priority(far))
        self.assertLess(strategy.get_priority(far), strategy.get_priority(near))

    def test_shared_steps_use_most_urgent_dependency(self):
        strategy = WorkflowStepsStrategy()
        shared = node(
            metadata("architect", next_use_step=20),
            metadata("coder", next_use_step=3),
        )
        distant = node(metadata("debugger", next_use_step=10))
        self.assertGreater(
            strategy.get_priority(shared), strategy.get_priority(distant)
        )

    def test_belady_uses_exact_next_use_distance(self):
        strategy = BeladyStrategy()
        far = node(metadata("far", next_use_request_step=18))
        near = node(metadata("near", next_use_request_step=11))
        self.assertLess(strategy.get_priority(far), strategy.get_priority(near))

    def test_recovery_value_uses_saving_per_byte_and_distance(self):
        strategy = RecoveryValueStrategy()
        low = node(
            metadata(
                "low",
                resident_bytes=100,
                dense_cost_ms=20,
                recovery_cost_ms=10,
                current_step=4,
                next_use_request_step=6,
            )
        )
        high = node(
            metadata(
                "high",
                resident_bytes=100,
                dense_cost_ms=80,
                recovery_cost_ms=10,
                current_step=4,
                next_use_request_step=6,
            )
        )
        strategy.observe((metadata("clock", current_step=4),))
        self.assertLess(strategy.get_priority(low), strategy.get_priority(high))

    def test_hierarchical_object_order(self):
        strategy = HierarchicalObjectStrategy()
        retired = node(metadata("retired", retired=True))
        lower_tier_exact = node(
            metadata(
                "exact",
                next_use_request_step=6,
                recoverable_from_lower_tier=True,
            )
        )
        anchor = node(
            metadata(
                "anchor",
                object_kind=CacheObjectKind.ANCHOR,
                next_use_request_step=6,
            )
        )
        canonical = node(
            metadata(
                "base",
                object_kind=CacheObjectKind.CANONICAL_BASE,
                next_use_request_step=6,
            )
        )
        ordered = sorted(
            [canonical, anchor, lower_tier_exact, retired],
            key=strategy.get_priority,
        )
        self.assertEqual(
            ordered,
            [retired, lower_tier_exact, anchor, canonical],
        )


class TestPrefetchAdmission(unittest.TestCase):
    def candidate(self, candidate_id, token_count, *items, access=0.0):
        state = CacheProtectionState()
        state.update(items)
        return PrefetchCandidate(
            candidate_id=candidate_id,
            token_count=token_count,
            protection=state,
            last_access_time=access,
        )

    def test_p0_is_always_disabled(self):
        decision = plan_prefetch(
            mode=PrefetchMode.OFF,
            required_tokens=4,
            available_tokens=10,
            candidates=(),
        )
        self.assertFalse(decision.admitted)
        self.assertEqual(decision.reason, "prefetch_disabled")

    def test_p1_uses_only_free_space(self):
        admitted = plan_prefetch(
            mode=PrefetchMode.FREE_SPACE_ONLY,
            required_tokens=4,
            available_tokens=4,
            candidates=(),
        )
        rejected = plan_prefetch(
            mode=PrefetchMode.FREE_SPACE_ONLY,
            required_tokens=4,
            available_tokens=3,
            candidates=(self.candidate(1, 10, metadata("dead", retired=True)),),
        )
        self.assertTrue(admitted.admitted)
        self.assertFalse(rejected.admitted)
        self.assertEqual(rejected.victim_ids, ())

    def test_p2_evicts_only_known_dead_objects(self):
        decision = plan_prefetch(
            mode=PrefetchMode.DEAD_OBJECT_ONLY,
            required_tokens=8,
            available_tokens=2,
            candidates=(
                self.candidate(1, 2),
                self.candidate(2, 4, metadata("live", next_use_request_step=10)),
                self.candidate(3, 6, metadata("dead", retired=True)),
            ),
        )
        self.assertTrue(decision.admitted)
        self.assertEqual(decision.victim_ids, (3,))

    def test_p2_rejects_when_dead_capacity_is_insufficient(self):
        decision = plan_prefetch(
            mode=PrefetchMode.DEAD_OBJECT_ONLY,
            required_tokens=8,
            available_tokens=2,
            candidates=(
                self.candidate(1, 4, metadata("dead", retired=True)),
                self.candidate(2, 10, metadata("live", next_use_request_step=10)),
            ),
        )
        self.assertFalse(decision.admitted)
        self.assertEqual(decision.victim_ids, (1,))

    def test_p3_may_evict_only_farther_or_dead_objects(self):
        decision = plan_prefetch(
            mode=PrefetchMode.ORACLE_NEXT_STAGE,
            required_tokens=10,
            available_tokens=1,
            target_next_use_step=2,
            candidates=(
                self.candidate(1, 4, metadata("near", next_use_request_step=1)),
                self.candidate(
                    2,
                    5,
                    metadata(
                        "far",
                        next_use_request_step=8,
                        recoverable_from_lower_tier=True,
                    ),
                ),
                self.candidate(3, 4, metadata("dead", retired=True)),
            ),
        )
        self.assertTrue(decision.admitted)
        self.assertEqual(decision.victim_ids, (3, 2))


class TestRadixCacheProtectionIntegration(unittest.TestCase):
    def setUp(self):
        allocator = Mock()
        allocator.device = torch.device("cpu")
        allocator.available_size.return_value = 0
        self.cache = RadixCache.create_simulated(mock_allocator=allocator)
        self.allocator = allocator

    def insert(self, tokens, item, *, request_priority=0):
        return self.cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=torch.tensor(tokens, dtype=torch.int64),
                priority=request_priority,
                cache_protection=(item,),
            )
        )

    def test_split_preserves_and_merges_object_metadata(self):
        first = self.insert([1, 2, 3], metadata("first", next_use_step=6))
        second = self.insert([1, 2, 4], metadata("second", next_use_step=2))

        shared = self.cache.root_node.children[1]
        self.assertEqual(set(shared.cache_protection.objects), {"first", "second"})
        self.assertEqual(set(shared.children[3].cache_protection.objects), {"first"})
        self.assertEqual(set(shared.children[4].cache_protection.objects), {"second"})
        self.assertIsNotNone(first.last_device_node)
        self.assertIsNotNone(second.last_device_node)

    def test_workflow_policy_ignores_request_priority(self):
        self.cache.eviction_strategy = WorkflowStepsStrategy()
        self.insert(
            [1, 1],
            metadata("far", next_use_step=10),
            request_priority=100,
        )
        self.insert(
            [2, 2],
            metadata("near", next_use_step=1),
            request_priority=-100,
        )

        result = self.cache.evict(EvictParams(num_tokens=2))

        self.assertEqual(result.num_tokens_evicted, 2)
        self.assertNotIn(1, self.cache.root_node.children)
        self.assertIn(2, self.cache.root_node.children)
        self.allocator.free.assert_called_once()

    def test_object_boundary_excludes_dynamic_suffixes(self):
        self.insert(
            [1, 2, 3, 9],
            metadata(
                "shared",
                protected_tokens=3,
                next_use_request_step=10,
            ),
        )
        self.insert(
            [1, 2, 3, 8],
            metadata(
                "shared",
                protected_tokens=3,
                current_step=2,
                next_use_request_step=12,
            ),
        )

        boundary = self.cache.find_cache_object_node("shared")
        self.assertIsNotNone(boundary)
        self.assertEqual(list(boundary.key.token_ids), [1, 2, 3])
        self.assertEqual(
            set(boundary.cache_protection.objects),
            {"shared"},
        )
        self.assertTrue(
            all(
                "shared" not in child.cache_protection.objects
                for child in boundary.children.values()
            )
        )

    def test_shorter_object_boundary_replaces_old_boundary(self):
        self.insert(
            [1, 2, 3, 4, 9],
            metadata(
                "moving",
                protected_tokens=4,
                next_use_request_step=10,
            ),
        )
        old_boundary = self.cache.find_cache_object_node("moving")
        self.assertEqual(list(old_boundary.key.token_ids), [1, 2, 3, 4])

        self.insert(
            [1, 2, 3, 8],
            metadata(
                "moving",
                protected_tokens=3,
                current_step=2,
                next_use_request_step=12,
            ),
        )

        new_boundary = self.cache.find_cache_object_node("moving")
        self.assertEqual(list(new_boundary.key.token_ids), [1, 2, 3])
        self.assertNotIn("moving", old_boundary.cache_object_boundaries)

    def test_eviction_cleans_root_metadata(self):
        for index in range(20):
            self.insert(
                [100 + index, 200 + index],
                metadata(
                    f"retired-{index}",
                    protected_tokens=2,
                    retired=True,
                ),
            )
        self.assertEqual(len(self.cache.root_node.cache_protection.objects), 20)

        self.cache.evict(EvictParams(num_tokens=40))

        self.assertEqual(self.cache.root_node.cache_protection.objects, {})

    def test_prefetch_candidate_covers_dynamic_suffix_subtree(self):
        self.insert(
            [1, 2, 3, 9],
            metadata(
                "dead",
                protected_tokens=3,
                retired=True,
            ),
        )
        boundary = self.cache.find_cache_object_node("dead")
        decision = self.cache.plan_cache_prefetch(
            mode=PrefetchMode.DEAD_OBJECT_ONLY,
            required_tokens=4,
            target_next_use_step=None,
        )
        self.assertTrue(decision.admitted)
        self.assertEqual(decision.victim_ids, (boundary.id,))
        self.assertEqual(decision.victim_tokens, 4)

    def test_reset_clears_recovery_value_clock(self):
        strategy = RecoveryValueStrategy()
        strategy.observe((metadata("clock", current_step=40),))
        self.cache.eviction_strategy = strategy

        self.cache.reset()

        self.assertEqual(strategy.current_step, 0)


class TestPhase5WorkloadMetadata(unittest.TestCase):
    def cache_object(self, index, role, kind):
        return CacheObject(
            object_id=f"object-{index}",
            role=role,
            kind=kind,
            reuse_class=ReuseClass.HOT,
            artifact_group=f"group-{index}",
            target_prefix_tokens=4,
            reusable_prefix_token_ids=(index, index + 10, index + 20, index + 30),
            payload=f"payload-{index}",
            dense_cost_weight=40.0 + index,
            recovery_cost_weight=4.0 + index,
        )

    def test_trace_has_backup_pressure_and_retry_cycles(self):
        workflow = (
            self.cache_object(1, "architect", WorkloadObjectKind.CANONICAL_BASE),
            self.cache_object(2, "coder", WorkloadObjectKind.STAGE_VARIANT),
            self.cache_object(3, "debugger", WorkloadObjectKind.ANCHOR),
            self.cache_object(4, "coder", WorkloadObjectKind.REPAIR_METADATA),
            self.cache_object(5, "debugger", WorkloadObjectKind.CANONICAL_BASE),
        )
        filler = self.cache_object(6, "cold-filler-a", WorkloadObjectKind.STAGE_VARIANT)
        live_filler = self.cache_object(
            7,
            "cold-filler-b",
            WorkloadObjectKind.REPAIR_METADATA,
        )
        trace = build_phase5_trace(
            selected_objects=(*workflow, filler, live_filler),
            workflow_sequence=workflow,
            workflow_cycles=2,
        )
        self.assertEqual([row.phase for row in trace[:5]], ["fill"] * 5)
        self.assertEqual([row.phase for row in trace[5:10]], ["backup"] * 5)
        self.assertEqual(
            [row.phase for row in trace[10:13]],
            ["pressure_live_fill", "pressure_live_backup", "pressure_dead"],
        )
        self.assertEqual(
            [row.object_id for row in trace[-11:-1]],
            [item.object_id for item in workflow] * 2,
        )
        self.assertEqual(trace[-1].phase, "pressure_replay")
        backup = next(
            row
            for row in trace
            if row.phase == "backup" and row.object_id == workflow[0].object_id
        )
        self.assertIsNotNone(backup.next_use_step)
        self.assertEqual(
            backup.next_use_request_step,
            backup.step + backup.next_use_distance,
        )

    def test_custom_params_include_policy_metadata_and_next_stage_hint(self):
        workflow = (
            self.cache_object(1, "architect", WorkloadObjectKind.CANONICAL_BASE),
            self.cache_object(2, "coder", WorkloadObjectKind.STAGE_VARIANT),
            self.cache_object(3, "debugger", WorkloadObjectKind.ANCHOR),
            self.cache_object(4, "coder", WorkloadObjectKind.REPAIR_METADATA),
            self.cache_object(5, "debugger", WorkloadObjectKind.CANONICAL_BASE),
        )
        filler = self.cache_object(6, "cold-filler-a", WorkloadObjectKind.STAGE_VARIANT)
        selected = (*workflow, filler)
        trace = build_phase5_trace(
            selected_objects=selected,
            workflow_sequence=workflow,
            workflow_cycles=1,
        )
        factory = custom_params_factory(
            selected_objects=selected,
            trace=trace,
            kv_bytes_per_token=100,
            enable_hicache=True,
        )
        first_workflow = next(row for row in trace if row.phase == "workflow")
        params = factory(workflow[0], first_workflow)
        self.assertEqual(
            params["cache_protection"]["resident_bytes"],
            workflow[0].reusable_prefix_tokens * 100,
        )
        self.assertEqual(
            params["cache_protection"]["protected_tokens"],
            workflow[0].reusable_prefix_tokens,
        )
        self.assertTrue(params["cache_protection"]["recoverable_from_lower_tier"])
        self.assertEqual(
            params["cache_prefetch"]["object_id"],
            workflow[1].object_id,
        )

        sampling = SamplingParams(custom_params=dict(params))
        restored = msgspec.msgpack.decode(
            msgspec.msgpack.encode(sampling),
            type=SamplingParams,
        )
        self.assertEqual(restored.custom_params, sampling.custom_params)


if __name__ == "__main__":
    unittest.main()
