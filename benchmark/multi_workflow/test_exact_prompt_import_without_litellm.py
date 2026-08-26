"""Exact-prompt TTFT replay must import without LiteLLM / mini-SWE-agent."""

from __future__ import annotations

import ast
from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_exact_prompt_speed_has_no_module_level_litellm_import() -> None:
    path = HERE / "run_natural_code_cost_exact_prompt_speed.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or "", *[alias.name for alias in node.names]]
        else:
            continue
        joined = " ".join(names)
        assert "litellm" not in joined
        assert "bridge_reuse_litellm_model" not in joined


def test_prerotated_runner_imports_without_litellm() -> None:
    from benchmark.multi_workflow.run_natural_code_cost_exact_prompt_speed import (
        TOTAL_ROUNDS,
        WARMUPS,
        generate_detailed,
    )
    from benchmark.multi_workflow.run_swebench_prerotated_file_modules import (
        generate_resilient,
        summarize,
    )

    assert TOTAL_ROUNDS == 4
    assert WARMUPS == 1
    assert callable(generate_detailed)
    assert callable(generate_resilient)
    assert callable(summarize)
