"""Prefix-on increment summarizer is COMPLETE without combined prefetch."""

from __future__ import annotations

from pathlib import Path

import json

from benchmark.multi_workflow.run_swebench_7b_prefix_on import MODES, summarize
from benchmark.multi_workflow.template_prefetch_modes import mode_manifest


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _arm(tmp: Path, name: str, ttft: float, copies: int = 0, prefix_hits: int = 0) -> None:
    targets = [
        {
            "group_index": 0,
            "round_index": round_index,
            "warmup": False,
            "ttft_ms": ttft,
            "output_text": "x",
        }
        for round_index in range(3)
    ]
    ledger = [{"event": "target_copied"} for _ in range(copies)]
    ledger.extend(
        {
            "event": "target_ordinary_prefix_matched",
            "ordinary_prefix_tokens": 10,
        }
        for _ in range(prefix_hits)
    )
    write_json(tmp / f"{name}.json", {"targets": targets, "ledger_rows": ledger})


def test_prefix_on_modes_exclude_combined() -> None:
    assert "combined" not in MODES
    assert MODES == ("dense", "prefix_only", "lossy_only", "dual")


def test_summarize_complete_without_combined(tmp_path: Path) -> None:
    write_json(
        tmp_path / "PLAN.json",
        {
            "model": "Qwen2.5-Coder-7B-Instruct",
            "not_30b_swebench_plan": True,
            "groups": [{"group_index": 0, "islands": 2}],
        },
    )
    _arm(tmp_path, "dense", 400.0)
    _arm(tmp_path, "prefix_only", 350.0, prefix_hits=3)
    _arm(tmp_path, "lossy_only", 250.0, copies=8)
    _arm(tmp_path, "dual", 200.0, copies=8, prefix_hits=3)
    result = summarize(tmp_path)
    assert result["status"] == "COMPLETE"
    assert result["prefetch"] is False
    assert result["ordinary_prefix_reuse"] is True
    assert result["not_eval_summary"] is True
    assert result["not_7b_dual_island"] is True
    assert "combined" not in result["latency"]
    assert result["algorithm_bars"]["dual_vs_dense"] == 2.0
    assert result["algorithm_bars"]["copy_on_prefix"] == 350.0 / 200.0
    assert result["one_token_output_agreement"]["not_accuracy"] is True


def test_summarize_partial_without_dual(tmp_path: Path) -> None:
    write_json(
        tmp_path / "PLAN.json",
        {"groups": [{"group_index": 0, "islands": 1}]},
    )
    _arm(tmp_path, "dense", 400.0)
    _arm(tmp_path, "prefix_only", 350.0)
    result = summarize(tmp_path)
    assert result["status"] == "PARTIAL"
    assert "dual" in result["missing_modes"]


def test_dual_manifest_keeps_copy_and_prefix() -> None:
    group = {"sources": [], "cases": [{"reuse_enabled": True}]}
    dual = mode_manifest(Path("/unused"), group, "Qwen2.5-Coder-7B-Instruct", "dual")
    assert dual["ordinary_prefix_reuse_enabled"] is True
    assert dual["prefetch_spill_device"] is False
    prefix = mode_manifest(Path("/unused"), group, "Qwen2.5-Coder-7B-Instruct", "prefix_only")
    assert prefix["cases"][0]["copy_middle"] is False
