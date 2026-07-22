from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKLOADS_PATH = REPO_ROOT / "benchmark/approx_kv/workloads.py"

spec = importlib.util.spec_from_file_location(
    "approx_kv_workloads",
    WORKLOADS_PATH,
)
workloads = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = workloads
spec.loader.exec_module(workloads)


class TestApproxKVWorkloads(unittest.TestCase):
    def test_retry_trace_next_use(self):
        trace = workloads.build_trace(workloads.TraceKind.RETRY)
        self.assertEqual(
            [item.role for item in trace],
            [
                "architect",
                "coder",
                "debugger",
                "coder",
                "debugger",
            ],
        )
        self.assertEqual(
            [item.next_use_step for item in trace],
            [None, 3, 4, None, None],
        )
        self.assertEqual(
            [workloads.next_use_distance(item) for item in trace],
            [None, 2, 2, None, None],
        )

    def test_pressure_ratio_and_working_set(self):
        active = workloads.estimate_active_reusable_tokens(
            code_tokens=8192,
            role_prefix_tokens=512,
            resident_variants=5,
        )
        self.assertEqual(active, 43520)
        point = workloads.PressurePoint(
            active_reusable_tokens=active,
            gpu_kv_capacity_tokens=21760,
        )
        self.assertEqual(point.ratio, 2.0)

    def test_synthetic_code_is_deterministic(self):
        first = workloads.deterministic_code("seed", 4)
        second = workloads.deterministic_code("seed", 4)
        other = workloads.deterministic_code("other", 4)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(first.count("def synthetic_"), 4)

    def test_interleaved_workflow_objects(self):
        independent = workloads.build_interleaved_object_trace(
            kind=workloads.TraceKind.RETRY,
            rounds=1,
            workflows=2,
            share_roles=False,
        )
        self.assertEqual(
            independent[:4],
            (
                "workflow-0:architect",
                "workflow-1:architect",
                "workflow-0:coder",
                "workflow-1:coder",
            ),
        )
        shared = workloads.build_interleaved_object_trace(
            kind=workloads.TraceKind.RETRY,
            rounds=1,
            workflows=2,
            share_roles=True,
        )
        self.assertEqual(shared[:4], ("architect", "architect", "coder", "coder"))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            workloads.build_trace(workloads.TraceKind.RETRY, rounds=0)
        with self.assertRaises(ValueError):
            workloads.deterministic_code("seed", 0)
        with self.assertRaises(ValueError):
            workloads.PressurePoint(1, 0).ratio
        with self.assertRaises(ValueError):
            workloads.build_interleaved_object_trace(
                kind=workloads.TraceKind.RETRY,
                rounds=1,
                workflows=0,
                share_roles=False,
            )


if __name__ == "__main__":
    unittest.main()
