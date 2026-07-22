import ast
import importlib.util
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
COMPAT_PATH = REPO_ROOT / "python/sglang/srt/layers/sm75_compat.py"

spec = importlib.util.spec_from_file_location("sm75_compat", COMPAT_PATH)
sm75_compat = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sm75_compat)


class TestSM75Compatibility(unittest.TestCase):
    def test_capability_gate(self):
        cases = [
            (False, (7, 5), False),
            (True, (7, 5), True),
            (True, (8, 0), False),
            (True, (9, 0), False),
            (True, (12, 0), False),
            (True, (None, None), True),
        ]
        for is_cuda_platform, capability, expected in cases:
            with self.subTest(
                is_cuda_platform=is_cuda_platform,
                capability=capability,
            ):
                self.assertEqual(
                    sm75_compat.should_use_native_cuda_fallback(
                        is_cuda_platform,
                        capability,
                    ),
                    expected,
                )

    def test_fused_entry_points_guard_native_fallback_first(self):
        cases = [
            (
                REPO_ROOT / "python/sglang/srt/layers/activation.py",
                "SiluAndMul",
                "forward_cuda",
            ),
            (
                REPO_ROOT / "python/sglang/srt/layers/activation.py",
                "GeluAndMul",
                "_forward_impl",
            ),
            (
                REPO_ROOT / "python/sglang/srt/layers/layernorm.py",
                "RMSNorm",
                "forward_cuda",
            ),
            (
                REPO_ROOT / "python/sglang/srt/layers/layernorm.py",
                "GemmaRMSNorm",
                "forward_cuda",
            ),
        ]

        for path, class_name, method_name in cases:
            with self.subTest(class_name=class_name, method_name=method_name):
                tree = ast.parse(path.read_text())
                class_node = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == class_name
                )
                method_node = next(
                    node
                    for node in class_node.body
                    if isinstance(node, ast.FunctionDef) and node.name == method_name
                )
                first_statement = method_node.body[0]
                self.assertIsInstance(first_statement, ast.If)
                self.assertEqual(
                    ast.unparse(first_statement.test),
                    "_use_native_cuda_fallback",
                )


if __name__ == "__main__":
    unittest.main()
