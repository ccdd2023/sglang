from __future__ import annotations

import unittest
from pathlib import Path

from sglang.srt.mem_cache.approx_kv.request import (
    ApproxKVRequestOperation,
    parse_request_metadata,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_DIR = REPO_ROOT / "python/sglang/srt/mem_cache/approx_kv"


class TestApproxKVIntegrationSource(unittest.TestCase):
    def test_request_metadata_reserves_last_prompt_token(self):
        metadata = parse_request_metadata(
            {
                "approx_kv": {
                    "operation": "reuse",
                    "model_fingerprint": "model",
                    "cache_dtype": "fp16",
                    "segments": [
                        {
                            "content_hash": "artifact",
                            "target_start": 2,
                            "length": 4,
                        }
                    ],
                }
            }
        )
        self.assertEqual(metadata.operation, ApproxKVRequestOperation.REUSE)
        metadata.validate_prompt_length(7)
        with self.assertRaisesRegex(ValueError, "final prompt token"):
            metadata.validate_prompt_length(6)

    def test_request_and_cache_lifecycle_wiring(self):
        schedule_source = (
            REPO_ROOT / "python/sglang/srt/managers/schedule_batch.py"
        ).read_text()
        radix_source = (
            REPO_ROOT / "python/sglang/srt/mem_cache/radix_cache.py"
        ).read_text()
        hiradix_source = (
            REPO_ROOT / "python/sglang/srt/mem_cache/hiradix_cache.py"
        ).read_text()
        unified_source = (
            REPO_ROOT / "python/sglang/srt/mem_cache/unified_radix_cache.py"
        ).read_text()
        common_source = (
            REPO_ROOT / "python/sglang/srt/mem_cache/common.py"
        ).read_text()
        self.assertIn("parse_request_metadata", schedule_source)
        self.assertIn("self.approx_kv_metadata", schedule_source)
        self.assertIn("or self.approx_kv_metadata is not None", schedule_source)
        self.assertIn("restore_request_prefix(tree_cache, self)", schedule_source)
        self.assertIn("self.approx_kv = ApproxKVManager", radix_source)
        self.assertIn("self.approx_kv.reset()", radix_source)
        self.assertIn("HiCacheResidencyBackend", hiradix_source)
        self.assertIn("self.approx_kv = ApproxKVManager", unified_source)
        self.assertIn("HiCacheResidencyBackend", unified_source)
        self.assertIn("register_request_segments(tree_cache, req)", common_source)
        self.assertIn("except ApproxKVRegistrationError", common_source)

    def test_kvcomm_is_wired_without_unrelated_algorithms(self):
        source = "\n".join(path.read_text() for path in PACKAGE_DIR.glob("*.py"))
        runtime_source = (PACKAGE_DIR / "runtime.py").read_text()
        manager_source = (PACKAGE_DIR / "manager.py").read_text()
        self.assertIn("class KVCOMMRecoveryPlugin", source)
        self.assertIn("execute_kvcomm_reconstruction", runtime_source)
        self.assertIn("_restore_kvcomm_prefix", runtime_source)
        self.assertIn("KVCOMMRecoveryPlugin", manager_source)

        source = source.lower()
        for forbidden in (
            "epic_fixed_k",
            "selective_repair",
            "hardware_selector",
            "allow_token_mismatch",
        ):
            self.assertNotIn(forbidden, source)

    def test_metrics_are_exposed(self):
        source = (
            REPO_ROOT / "python/sglang/srt/observability/metrics_collector.py"
        ).read_text()
        for metric in (
            "sglang:approx_kv_requests_total",
            "sglang:approx_kv_copied_tokens_total",
            "sglang:approx_kv_dense_fallback_total",
            "sglang:approx_kv_h2d_tokens_total",
            "sglang:approx_kv_host_export_tokens_total",
        ):
            self.assertIn(metric, source)


if __name__ == "__main__":
    unittest.main()
