from __future__ import annotations

import ast
import importlib
import sys
import types as python_types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_DIR = REPO_ROOT / "python/sglang/srt/mem_cache/approx_kv"
PACKAGE_NAME = "approx_kv_integration_under_test"

package = python_types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_DIR)]
sys.modules[PACKAGE_NAME] = package

request_module = importlib.import_module(f"{PACKAGE_NAME}.request")
types_module = importlib.import_module(f"{PACKAGE_NAME}.types")

RecoveryMode = types_module.RecoveryMode


class TestApproxKVIntegrationSource(unittest.TestCase):
    def test_request_metadata_reserves_last_prompt_token(self):
        metadata = request_module.parse_request_metadata(
            {
                "approx_kv": {
                    "recovery_mode": "raw_rope",
                    "speed_only": True,
                    "register_source": False,
                    "segments": [
                        {
                            "source_content_hash": "code",
                            "target_start": 2,
                            "length": 4,
                        }
                    ],
                }
            }
        )
        self.assertEqual(metadata.recovery_mode, RecoveryMode.RAW_ROPE)
        self.assertTrue(metadata.speed_only)
        self.assertFalse(metadata.register_source)
        metadata.validate_prompt_length(7)
        with self.assertRaisesRegex(ValueError, "final prompt token"):
            metadata.validate_prompt_length(6)

    def test_radix_backend_has_full_layer_copy_and_rotation(self):
        path = PACKAGE_DIR / "radix_backend.py"
        tree = ast.parse(path.read_text())
        backend = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RadixKVTransferBackend"
        )
        methods = {
            node.name: node
            for node in backend.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("copy_and_rotate", methods)
        self.assertIn("reconstruct_and_rotate", methods)
        self.assertIn("_copy_all_layers", methods)
        self.assertIn("_rotate_all_copied_keys", methods)

        copy_source = ast.unparse(methods["_copy_all_layers"])
        rotate_source = ast.unparse(methods["_rotate_all_copied_keys"])
        self.assertIn("range(kvcache.layer_num)", copy_source)
        self.assertIn("get_key_buffer", copy_source)
        self.assertIn("get_value_buffer", copy_source)
        self.assertIn("range(kvcache.layer_num)", rotate_source)
        self.assertIn("apply_rotary_emb", rotate_source)

    def test_integration_does_not_contain_host_driver_operations(self):
        source = (
            (PACKAGE_DIR / "radix_backend.py").read_text()
            + (PACKAGE_DIR / "runtime.py").read_text()
        ).lower()
        forbidden = (
            "modprobe",
            "dkms",
            "/lib/modules",
            "nvidia-smi",
            "apt install",
        )
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_restored_slots_remain_request_owned(self):
        runtime_source = (PACKAGE_DIR / "runtime.py").read_text()
        self.assertNotIn(
            "req.cache_protected_len = len(req.prefix_indices)",
            runtime_source,
        )

    def test_request_and_radix_lifecycle_wiring(self):
        schedule_source = (
            REPO_ROOT / "python/sglang/srt/managers/schedule_batch.py"
        ).read_text()
        radix_source = (
            REPO_ROOT / "python/sglang/srt/mem_cache/radix_cache.py"
        ).read_text()
        common_source = (
            REPO_ROOT / "python/sglang/srt/mem_cache/common.py"
        ).read_text()
        self.assertIn("parse_request_metadata", schedule_source)
        self.assertIn("self.approx_kv_metadata", schedule_source)
        self.assertIn(
            "or self.approx_kv_metadata is not None",
            schedule_source,
        )
        self.assertIn("restore_request_prefix(tree_cache, self)", schedule_source)
        self.assertIn("self.approx_kv = ApproxKVManager", radix_source)
        self.assertIn("self.approx_kv.reset()", radix_source)
        self.assertIn(
            "register_request_segments(tree_cache, req)",
            common_source,
        )


if __name__ == "__main__":
    unittest.main()
