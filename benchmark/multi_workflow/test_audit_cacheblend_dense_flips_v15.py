from __future__ import annotations

import hashlib

import pytest

from benchmark.multi_workflow.audit_cacheblend_dense_flips_v15 import (
    audit_rows,
)


def _row(case_id: str, *, mode: str, passed: bool, output: str) -> dict:
    reuse = mode == "reuse"
    return {
        "blend_layers_executed": 1 if reuse else 0,
        "concurrency": 1,
        "config_id": "recompute-0.05" if reuse else "dense",
        "context_tokens": 128,
        "dtype": "native",
        "engine": "vllm-blend",
        "engine_commit": "frozen",
        "error": None,
        "method": "cacheblend",
        "mode": mode,
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "passed": passed,
        "phase": "accuracy",
        "physical_reuse_proven": reuse,
        "prompt_sha256": f"prompt-{case_id}",
        "recomputed_tokens": 8 if reuse else 0,
        "request_topology": "cacheblend-vllm-segmented-request",
        "reused_k_tokens": 120 if reuse else 0,
        "reused_v_tokens": 120 if reuse else 0,
        "split": "formal",
        "suite": "synthetic",
        "target_tokens": 128,
        "token_ids_sha256": f"tokens-{case_id}",
        "case_id": case_id,
        "metadata": {"output_text": output},
    }


def test_audit_separates_damage_rescue_and_fidelity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "benchmark.multi_workflow.audit_cacheblend_dense_flips_v15."
        "EXPECTED_CASES",
        4,
    )
    dense = [
        _row("both-pass", mode="dense", passed=True, output="a"),
        _row("damage", mode="dense", passed=True, output="b"),
        _row("rescue", mode="dense", passed=False, output="c"),
        _row("both-fail", mode="dense", passed=False, output="d"),
    ]
    reuse = [
        _row("both-pass", mode="reuse", passed=True, output="a"),
        _row("damage", mode="reuse", passed=False, output="B"),
        _row("rescue", mode="reuse", passed=True, output="C"),
        _row("both-fail", mode="reuse", passed=False, output="d"),
    ]

    result = audit_rows(dense, reuse)

    assert result["transitions"] == {
        "both_pass": 1,
        "dense_only": 1,
        "reuse_only": 1,
        "both_fail": 1,
    }
    assert result["task_correctness"]["dense_passed"] == 2
    assert result["task_correctness"]["cacheblend_passed"] == 2
    assert result["dense_preservation"]["damage_rate_given_dense_pass"] == 0.5
    assert result["dense_preservation"]["rescue_rate_given_dense_fail"] == 0.5
    assert result["fidelity"]["exact_output_matches"] == 2
    assert {row["case_id"] for row in result["flips"]} == {
        "damage",
        "rescue",
    }


def test_audit_rejects_unpaired_prompt_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "benchmark.multi_workflow.audit_cacheblend_dense_flips_v15."
        "EXPECTED_CASES",
        1,
    )
    dense = [_row("case", mode="dense", passed=True, output="a")]
    reuse = [_row("case", mode="reuse", passed=True, output="a")]
    reuse[0]["token_ids_sha256"] = "different"

    with pytest.raises(ValueError, match="unpaired fields"):
        audit_rows(dense, reuse)
