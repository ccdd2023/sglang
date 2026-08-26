"""Dual-island mode isolation and 7B PLAN is not 30B token ids."""

from __future__ import annotations

from pathlib import Path

from benchmark.multi_workflow.template_prefetch_modes import (
    MODEL_7B,
    combined_vs_coding_speedup,
    mode_manifest,
)


def test_reuse_modes_isolate_prefix_and_lossy() -> None:
    group = {"sources": [], "cases": [{"reuse_enabled": True}]}
    output = Path("/tmp/unused")
    dense = mode_manifest(output, group, "m", "dense")
    assert dense.get("prefer_host_sources") is False
    assert dense.get("ordinary_prefix_reuse_enabled") is False

    prefix = mode_manifest(output, group, "m", "prefix_only")
    assert prefix["ordinary_prefix_reuse_enabled"] is True
    assert prefix["cases"][0]["reuse_enabled"] is True
    assert prefix["cases"][0]["copy_middle"] is False
    assert prefix["prefetch_spill_device"] is False
    assert prefix["prefer_host_sources"] is False

    lossy = mode_manifest(output, group, "m", "lossy_only")
    assert lossy["ordinary_prefix_reuse_enabled"] is False
    assert lossy["cases"][0]["reuse_enabled"] is True
    assert lossy["prefetch_spill_device"] is False

    dual = mode_manifest(output, group, "m", "dual")
    assert dual["ordinary_prefix_reuse_enabled"] is True
    assert dual["cases"][0]["reuse_enabled"] is True
    assert dual["prefetch_spill_device"] is False
    assert dual["prefer_host_sources"] is False

    combined = mode_manifest(output, group, "m", "combined")
    assert combined["ordinary_prefix_reuse_enabled"] is True
    assert combined["cases"][0]["reuse_enabled"] is True
    assert combined["prefetch_spill_device"] is True
    assert combined["prefer_host_sources"] is True


def test_official_96092_manifest_stays_prefix_and_prefetch_off() -> None:
    src = (
        Path(__file__).resolve().parent
        / "run_swebench_prerotated_file_modules.py"
    ).read_text(encoding="utf-8")
    start = src.index("def _manifest(")
    end = src.index("\ndef ", start + 1)
    body = src[start:end]
    assert '"ordinary_prefix_reuse_enabled": False' in body
    assert "prefer_host_sources" not in body
    assert "prefetch_spill_device" not in body


def test_ledger_counts_ignore_zero_token_prefix_events() -> None:
    from benchmark.multi_workflow.template_prefetch_modes import ledger_counts

    rows = [
        {"event": "target_ordinary_prefix_matched", "ordinary_prefix_tokens": 0},
        {"event": "target_ordinary_prefix_matched", "ordinary_prefix_tokens": 878},
        {"event": "target_copied"},
        {"event": "target_middle_left_dense"},
    ]
    counts = ledger_counts(rows)
    assert counts["ordinary_prefix_matched"] == 1
    assert counts["copy_events"] == 1
    assert counts["middle_left_dense"] == 1


def test_combined_vs_dual_is_prefetch_increment_not_vs_dense() -> None:
    latency = {
        "dual": {"cache_ready_speedup_ratio_of_means": 1.39},
        "combined": {"cache_ready_speedup_ratio_of_means": 1.46},
    }
    assert abs(combined_vs_coding_speedup(latency) - 1.46 / 1.39) < 1e-12


def test_7b_plan_is_not_30b_swebench_token_ids() -> None:
    from benchmark.multi_workflow.prepare_7b_dual_island_plan import (
        PLAN_30B,
        build_groups,
        assert_not_30b_plan,
    )
    import json

    cases = json.loads(
        Path(
            "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
            "impactkv_coding_dual_island_v8_cold_20260727/COLD_CASES.json"
        ).read_text(encoding="utf-8")
    )["cases"]
    groups = build_groups(cases)
    assert_not_30b_plan(groups)
    thirty = json.loads(PLAN_30B.read_text(encoding="utf-8"))
    assert groups[0]["target_prompt_hash"] != thirty["groups"][0]["target_prompt_hash"]
    assert groups[0]["model"] == MODEL_7B
    assert groups[0]["retokenized_from_30b"] is True
    assert groups[0]["sources"][0]["source_start"] > 0
    assert groups[0]["cases"][0]["target_start"] != groups[0]["cases"][0]["source_start"]
    src = groups[0]["source_input_ids"][0]
    tgt = groups[0]["target_input_ids"]
    ss = groups[0]["cases"][0]["source_start"]
    ts = groups[0]["cases"][0]["target_start"]
    n = groups[0]["cases"][0]["length"]
    assert src[ss : ss + n] == tgt[ts : ts + n]
    assert src != cases[0]["source_input_ids"]
