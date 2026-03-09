import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"


def _run_probe(code: str) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(PYTHON_DIR)
        if not env.get("PYTHONPATH")
        else f"{PYTHON_DIR}{os.pathsep}{env['PYTHONPATH']}"
    )
    env.setdefault("PYTHONNOUSERSITE", "1")

    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


class TestSchedulerImportRegression(unittest.TestCase):
    def test_import_sglang_stays_lazy(self):
        result = _run_probe(
            """
            import json
            import sys

            import sglang

            print(json.dumps({
                "sglang.lang.api": "sglang.lang.api" in sys.modules,
                "openai": any(m == "openai" or m.startswith("openai.") for m in sys.modules),
                "transformers": any(m == "transformers" or m.startswith("transformers.") for m in sys.modules),
                "compressed_tensors": any(m == "compressed_tensors" or m.startswith("compressed_tensors.") for m in sys.modules),
            }))
            """
        )

        self.assertFalse(result["sglang.lang.api"])
        self.assertFalse(result["openai"])
        self.assertFalse(result["transformers"])
        self.assertFalse(result["compressed_tensors"])

    def test_moe_utils_import_does_not_load_runner(self):
        result = _run_probe(
            """
            import json
            import sys

            from sglang.srt.layers.moe.utils import initialize_moe_config

            print(json.dumps({
                "initialize_moe_config": callable(initialize_moe_config),
                "sglang.srt.layers.moe.moe_runner": "sglang.srt.layers.moe.moe_runner" in sys.modules,
                "sglang.srt.layers.moe.moe_runner.deep_gemm": "sglang.srt.layers.moe.moe_runner.deep_gemm" in sys.modules,
            }))
            """
        )

        self.assertTrue(result["initialize_moe_config"])
        self.assertFalse(result["sglang.srt.layers.moe.moe_runner"])
        self.assertFalse(result["sglang.srt.layers.moe.moe_runner.deep_gemm"])

    def test_scheduler_import_avoids_heavy_optional_modules(self):
        result = _run_probe(
            """
            import json
            import sys

            from sglang.srt.managers.scheduler import Scheduler

            print(json.dumps({
                "Scheduler": Scheduler.__name__,
                "sglang.lang.api": "sglang.lang.api" in sys.modules,
                "openai": any(m == "openai" or m.startswith("openai.") for m in sys.modules),
                "transformers": any(m == "transformers" or m.startswith("transformers.") for m in sys.modules),
                "compressed_tensors": any(m == "compressed_tensors" or m.startswith("compressed_tensors.") for m in sys.modules),
                "flashinfer": any(m == "flashinfer" or m.startswith("flashinfer.") for m in sys.modules),
                "deep_gemm_wrapper": any(m.startswith("sglang.srt.layers.deep_gemm_wrapper") for m in sys.modules),
                "moe_runner.deep_gemm": "sglang.srt.layers.moe.moe_runner.deep_gemm" in sys.modules,
                "torch.utils.cpp_extension": "torch.utils.cpp_extension" in sys.modules,
            }))
            """
        )

        self.assertEqual(result["Scheduler"], "Scheduler")
        self.assertFalse(result["sglang.lang.api"])
        self.assertFalse(result["openai"])
        self.assertFalse(result["transformers"])
        self.assertFalse(result["compressed_tensors"])
        self.assertFalse(result["flashinfer"])
        self.assertFalse(result["deep_gemm_wrapper"])
        self.assertFalse(result["moe_runner.deep_gemm"])
        self.assertFalse(result["torch.utils.cpp_extension"])

    def test_model_config_import_does_not_load_transformers(self):
        result = _run_probe(
            """
            import json
            import sys

            from sglang.srt.configs.model_config import ModelConfig

            print(json.dumps({
                "ModelConfig": ModelConfig.__name__,
                "transformers": any(m == "transformers" or m.startswith("transformers.") for m in sys.modules),
                "compressed_tensors": any(m == "compressed_tensors" or m.startswith("compressed_tensors.") for m in sys.modules),
                "hf_transformers_utils": "sglang.srt.utils.hf_transformers_utils" in sys.modules,
                "deepseekvl2": "sglang.srt.configs.deepseekvl2" in sys.modules,
            }))
            """
        )

        self.assertEqual(result["ModelConfig"], "ModelConfig")
        self.assertFalse(result["transformers"])
        self.assertFalse(result["compressed_tensors"])
        self.assertFalse(result["hf_transformers_utils"])
        self.assertFalse(result["deepseekvl2"])

    def test_reasoning_parser_import_does_not_load_openai(self):
        result = _run_probe(
            """
            import json
            import sys

            from sglang.srt.parser.reasoning_parser import ReasoningParser

            print(json.dumps({
                "ReasoningParser": ReasoningParser.__name__,
                "openai": any(m == "openai" or m.startswith("openai.") for m in sys.modules),
                "openai_protocol": "sglang.srt.entrypoints.openai.protocol" in sys.modules,
            }))
            """
        )

        self.assertEqual(result["ReasoningParser"], "ReasoningParser")
        self.assertFalse(result["openai"])
        self.assertFalse(result["openai_protocol"])

    def test_server_args_import_does_not_load_heavy_deps(self):
        result = _run_probe(
            """
            import json
            import sys

            from sglang.srt.server_args import ServerArgs

            print(json.dumps({
                "ServerArgs": ServerArgs.__name__,
                "openai": any(m == "openai" or m.startswith("openai.") for m in sys.modules),
                "transformers": any(m == "transformers" or m.startswith("transformers.") for m in sys.modules),
                "hf_transformers_utils": "sglang.srt.utils.hf_transformers_utils" in sys.modules,
            }))
            """
        )

        self.assertEqual(result["ServerArgs"], "ServerArgs")
        self.assertFalse(result["openai"])
        self.assertFalse(result["transformers"])
        self.assertFalse(result["hf_transformers_utils"])

    def test_configs_package_does_not_eagerly_import_model_configs(self):
        result = _run_probe(
            """
            import json
            import sys

            from sglang.srt.configs import DeepseekVL2Config

            print(json.dumps({
                "DeepseekVL2Config": DeepseekVL2Config.__name__,
                "exaone": "sglang.srt.configs.exaone" in sys.modules,
                "chatglm": "sglang.srt.configs.chatglm" in sys.modules,
                "dbrx": "sglang.srt.configs.dbrx" in sys.modules,
            }))
            """
        )

        self.assertEqual(result["DeepseekVL2Config"], "DeepseekVL2Config")
        self.assertFalse(result["exaone"])
        self.assertFalse(result["chatglm"])
        self.assertFalse(result["dbrx"])

    def test_compatibility_imports_still_work(self):
        result = _run_probe(
            """
            import json

            from sglang import ServerArgs, gen
            from sglang.srt.layers.moe import MoeRunner, MoeRunnerConfig, initialize_moe_config
            from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config

            print(json.dumps({
                "gen": callable(gen),
                "ServerArgs": ServerArgs.__name__,
                "MoeRunner": MoeRunner.__name__,
                "MoeRunnerConfig": MoeRunnerConfig.__name__,
                "initialize_moe_config": callable(initialize_moe_config),
                "initialize_fp8_gemm_config": callable(initialize_fp8_gemm_config),
            }))
            """
        )

        self.assertTrue(result["gen"])
        self.assertEqual(result["ServerArgs"], "ServerArgs")
        self.assertEqual(result["MoeRunner"], "MoeRunner")
        self.assertEqual(result["MoeRunnerConfig"], "MoeRunnerConfig")
        self.assertTrue(result["initialize_moe_config"])
        self.assertTrue(result["initialize_fp8_gemm_config"])


if __name__ == "__main__":
    unittest.main()
