from __future__ import annotations

import importlib
import sys
import types as python_types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_DIR = REPO_ROOT / "python/sglang/srt/mem_cache/approx_kv"
PACKAGE_NAME = "approx_kv_scheduler_under_test"

package = python_types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_DIR)]
sys.modules[PACKAGE_NAME] = package

recovery_module = importlib.import_module(f"{PACKAGE_NAME}.recovery")
scheduling_module = importlib.import_module(f"{PACKAGE_NAME}.scheduling")
types_module = importlib.import_module(f"{PACKAGE_NAME}.types")

CacheCandidate = scheduling_module.CacheCandidate
CacheObjectKind = scheduling_module.CacheObjectKind
EvictionPolicy = scheduling_module.EvictionPolicy
PrefetchMode = scheduling_module.PrefetchMode
PrefetchRequest = scheduling_module.PrefetchRequest
RecoveryMeasurement = recovery_module.RecoveryMeasurement
HardwareAwareRecoverySelector = recovery_module.HardwareAwareRecoverySelector
RecoveryMode = types_module.RecoveryMode
ResidencyTier = types_module.ResidencyTier


def candidate(
    object_id,
    *,
    resident_bytes=100,
    last_access_step=0,
    dense_cost_ms=20,
    recovery_cost_ms=5,
    kind=CacheObjectKind.EXACT_VARIANT,
    steps_to_execution=None,
    oracle_next_use_step=None,
    retired=False,
    pinned=False,
):
    return CacheCandidate(
        object_id=object_id,
        resident_bytes=resident_bytes,
        last_access_step=last_access_step,
        dense_cost_ms=dense_cost_ms,
        recovery_cost_ms=recovery_cost_ms,
        kind=kind,
        steps_to_execution=steps_to_execution,
        oracle_next_use_step=oracle_next_use_step,
        retired=retired,
        pinned=pinned,
    )


class TestApproxKVScheduler(unittest.TestCase):
    def test_hardware_selector_uses_nearest_profile_and_lowest_ttft(self):
        selector = HardwareAwareRecoverySelector(
            (
                RecoveryMeasurement(
                    RecoveryMode.DENSE,
                    4096,
                    ResidencyTier.HOST,
                    dense_ms=80,
                    last_token_ms=2,
                ),
                RecoveryMeasurement(
                    RecoveryMode.RAW_ROPE,
                    4096,
                    ResidencyTier.HOST,
                    dense_ms=80,
                    h2d_ms=20,
                    recovery_ms=5,
                    last_token_ms=2,
                ),
                RecoveryMeasurement(
                    RecoveryMode.EPIC_FIXED_K,
                    8192,
                    ResidencyTier.HOST,
                    dense_ms=150,
                    h2d_ms=35,
                    recovery_ms=20,
                    last_token_ms=2,
                ),
            )
        )
        selection = selector.select(
            token_count=5000,
            source_tier=ResidencyTier.HOST,
        )
        self.assertEqual(selection.mode, RecoveryMode.RAW_ROPE)
        self.assertEqual(selection.predicted_ttft_ms, 27)
        self.assertEqual(selection.saved_ms, 55)

    def test_lru_steps_and_oracle_rank_differently(self):
        items = (
            candidate(
                "a",
                last_access_step=8,
                steps_to_execution=9,
                oracle_next_use_step=11,
            ),
            candidate(
                "b",
                last_access_step=1,
                steps_to_execution=1,
                oracle_next_use_step=None,
            ),
            candidate(
                "c",
                last_access_step=4,
                steps_to_execution=None,
                oracle_next_use_step=30,
            ),
        )
        self.assertEqual(
            [
                item.object_id
                for item in scheduling_module.rank_for_eviction(
                    items,
                    policy=EvictionPolicy.LRU,
                    current_step=10,
                )
            ],
            ["b", "c", "a"],
        )
        self.assertEqual(
            [
                item.object_id
                for item in scheduling_module.rank_for_eviction(
                    items,
                    policy=EvictionPolicy.STEPS_ONLY,
                    current_step=10,
                )
            ],
            ["c", "a", "b"],
        )
        self.assertEqual(
            [
                item.object_id
                for item in scheduling_module.rank_for_eviction(
                    items,
                    policy=EvictionPolicy.BELADY_ORACLE,
                    current_step=10,
                )
            ],
            ["b", "c", "a"],
        )

    def test_value_density_and_hierarchy_protect_high_value_base(self):
        items = (
            candidate(
                "cheap-anchor",
                resident_bytes=200,
                dense_cost_ms=10,
                recovery_cost_ms=9,
                steps_to_execution=1,
                kind=CacheObjectKind.CONTEXT_ANCHOR,
            ),
            candidate(
                "shared-base",
                resident_bytes=100,
                dense_cost_ms=100,
                recovery_cost_ms=5,
                steps_to_execution=1,
                kind=CacheObjectKind.CANONICAL_BASE,
            ),
            candidate(
                "retired",
                kind=CacheObjectKind.EXACT_VARIANT,
                retired=True,
            ),
        )
        value_order = scheduling_module.rank_for_eviction(
            items,
            policy=EvictionPolicy.VALUE_DENSITY,
            current_step=0,
        )
        self.assertEqual(
            [item.object_id for item in value_order],
            ["retired", "cheap-anchor", "shared-base"],
        )
        hierarchical = scheduling_module.rank_for_eviction(
            items,
            policy=EvictionPolicy.HIERARCHICAL,
            current_step=0,
        )
        self.assertEqual(
            [item.object_id for item in hierarchical],
            ["retired", "cheap-anchor", "shared-base"],
        )

    def test_victim_selection_skips_pinned_objects(self):
        items = (
            candidate("pinned", resident_bytes=100, pinned=True),
            candidate("first", resident_bytes=60, last_access_step=1),
            candidate("second", resident_bytes=60, last_access_step=2),
        )
        victims = scheduling_module.select_victims(
            items,
            bytes_to_free=100,
            policy=EvictionPolicy.LRU,
            current_step=0,
        )
        self.assertEqual(
            [item.object_id for item in victims],
            ["first", "second"],
        )

    def test_prefetch_free_space_retired_and_oracle_guards(self):
        request = PrefetchRequest(
            object_id="target",
            resident_bytes=100,
            miss_cost_ms=30,
            oracle_next_use_step=5,
        )
        free = scheduling_module.admit_prefetch(
            request,
            mode=PrefetchMode.FREE_SPACE_ONLY,
            free_bytes=100,
            candidates=(),
            current_step=0,
        )
        self.assertTrue(free.admitted)

        retired = candidate(
            "retired",
            resident_bytes=100,
            recovery_cost_ms=4,
            retired=True,
        )
        retired_decision = scheduling_module.admit_prefetch(
            request,
            mode=PrefetchMode.RETIRED_ONLY,
            free_bytes=0,
            candidates=(retired,),
            current_step=0,
        )
        self.assertTrue(retired_decision.admitted)
        self.assertEqual(
            [item.object_id for item in retired_decision.victims],
            ["retired"],
        )

        urgent = candidate(
            "urgent",
            resident_bytes=100,
            recovery_cost_ms=2,
            oracle_next_use_step=4,
        )
        rejected = scheduling_module.admit_prefetch(
            request,
            mode=PrefetchMode.ORACLE,
            free_bytes=0,
            candidates=(urgent,),
            current_step=0,
        )
        self.assertFalse(rejected.admitted)
        self.assertEqual(
            rejected.reason,
            "victim_needed_no_later_than_target",
        )


if __name__ == "__main__":
    unittest.main()
