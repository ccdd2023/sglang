from __future__ import annotations

import unittest
from argparse import Namespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")

from benchmark.approx_kv.phase6.manifest import build_fixed40_manifest
from benchmark.approx_kv.run_p6_0_contract import build_contract, verify_contract
from benchmark.approx_kv.run_p6_4_capacity_pilot import (
    labeled_metric_delta,
    launch_cells,
    representation_metadata,
)
from benchmark.approx_kv.run_p6_h_host_roundtrip import metadata as host_metadata


class TestPhase6Manifest(unittest.TestCase):
    def test_manifest_is_deterministic_and_fixed(self):
        first = build_fixed40_manifest()
        second = build_fixed40_manifest()
        self.assertEqual(first, second)
        self.assertEqual(first["object_count"], 40)
        self.assertEqual(len({item["object_id"] for item in first["objects"]}), 40)
        lengths = {item["logical_tokens"] for item in first["objects"]}
        self.assertGreaterEqual(len(lengths), 2)
        workflow_lengths = {
            item["logical_tokens"]
            for item in first["objects"]
            if item["object_id"].startswith("workflow-")
        }
        self.assertEqual(workflow_lengths, {1024, 2048})
        self.assertTrue(all(item["token_ids_sha256"] for item in first["objects"]))

    def test_dead_live_identity_is_frozen(self):
        manifest = build_fixed40_manifest()
        dead = [item for item in manifest["objects"] if item["retired"]]
        live = [item for item in manifest["objects"] if item["active"]]
        self.assertEqual(len(dead), 12)
        self.assertEqual(len(live), 28)

    def test_contract_records_provisional_chunk_and_profiles(self):
        contract = build_contract(
            Namespace(
                source_git_sha="abc",
                source_tree_sha="tree",
                image_digest="sha256:image",
                model="model",
                model_revision="revision",
                chunked_prefill_size=1024,
                chunk_source="provisional_worst_case",
            )
        )
        self.assertEqual(contract["settings"]["chunk_source"], "provisional_worst_case")
        self.assertEqual(
            contract["representation_profiles"]["r4_like"]["resident_multiplicity"],
            5,
        )
        self.assertFalse(contract["performance_ranking_enabled"])
        self.assertIn("exact_only", contract["representation_profiles"])
        self.assertEqual(contract["workload"]["chunked_prefill_size"], 1024)
        contract["run_id"] = "test-run"
        verify_contract(contract)

    def test_contract_verification_rejects_drift(self):
        contract = build_contract(
            Namespace(
                source_git_sha="abc",
                source_tree_sha="tree",
                image_digest="sha256:image",
                model="model",
                model_revision="revision",
                chunked_prefill_size=1024,
                chunk_source="provisional_worst_case",
            )
        )
        contract["workload"]["objects"][0]["logical_tokens"] += 1
        with self.assertRaises(ValueError):
            verify_contract(contract)

    def test_capacity_runner_uses_segment_bounded_representations(self):
        item = build_fixed40_manifest()["objects"][1]
        metadata = representation_metadata(
            item,
            profile="r4_like",
            representation_index=4,
            object_kind="delta",
            round_index=0,
            segment_tokens_max=512,
        )
        self.assertEqual(sum(row["length"] for row in metadata["segments"]), 2048)
        self.assertTrue(all(row["length"] <= 512 for row in metadata["segments"]))
        self.assertEqual(
            len({row["object_id"] for row in metadata["segments"]}),
            len(metadata["segments"]),
        )
        self.assertTrue(
            all("rep3:" in row["dependencies"][0] for row in metadata["segments"])
        )

    def test_host_roundtrip_uses_actual_bounded_segments(self):
        payload = host_metadata(
            operation="register",
            content_hash="host",
            object_id="host",
            header_tokens=64,
            body_tokens=1024,
            object_kind="materialization_scratch",
            residency="device",
            segment_tokens=512,
        )
        self.assertEqual(len(payload["segments"]), 2)
        self.assertEqual(
            [segment["target_start"] for segment in payload["segments"]],
            [64, 576],
        )
        self.assertTrue(
            all(segment["length"] <= 512 for segment in payload["segments"])
        )

    def test_capacity_launch_order_pairs_s0_s4_at_rho_two(self):
        cells = launch_cells((1.1, 1.5, 2.0, 3.0))
        self.assertEqual(
            cells,
            [
                ("hierarchical", 1.1),
                ("hierarchical", 1.5),
                ("lru", 2.0),
                ("hierarchical", 2.0),
                ("hierarchical", 3.0),
            ],
        )

    def test_labeled_metric_delta_preserves_provenance(self):
        before = (
            "sglang:cross_store_evicted_bytes_total"
            '{requester="approximate",provenance="exact",'
            'object_kind="filler"} 10\n'
        )
        after = (
            "sglang:cross_store_evicted_bytes_total"
            '{requester="approximate",provenance="exact",'
            'object_kind="filler"} 30\n'
            "sglang:cross_store_evicted_bytes_total"
            '{requester="exact",provenance="approximate",'
            'object_kind="delta"} 40\n'
        )
        self.assertEqual(
            labeled_metric_delta(
                before,
                after,
                "sglang:cross_store_evicted_bytes_total",
                {"provenance": "exact"},
            ),
            20,
        )
        self.assertEqual(
            labeled_metric_delta(
                before,
                after,
                "sglang:cross_store_evicted_bytes_total",
                {"requester": "exact", "provenance": "approximate"},
            ),
            40,
        )


if __name__ == "__main__":
    unittest.main()
