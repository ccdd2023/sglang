from pathlib import Path

from benchmark.multi_workflow.template_prefetch_modes import (
    mode_env,
    mode_manifest,
    staircase_increments,
)


def test_mode_env_keeps_dual_prefetch_off():
    assert mode_env("dual")["SGLANG_KV_PREFETCH"] == "0"
    assert mode_env("combined")["SGLANG_KV_PREFETCH"] == "1"
    assert mode_env("dense")["SGLANG_KVCOMM_CORE"] == "0"


def test_staircase_all_keep_lossy_copy_and_fair_host():
    group = {
        "sources": [{"source_id": "s0"}],
        "cases": [{"case_id": "c0", "reuse_enabled": True}],
    }
    lossy = mode_manifest(Path("/tmp/art"), group, "m", "lossy_host")
    prefix = mode_manifest(Path("/tmp/art"), group, "m", "prefix_prefetch")
    templ = mode_manifest(Path("/tmp/art"), group, "m", "template_prefetch")
    for row in (lossy, prefix, templ):
        assert row["prefer_host_sources"] is True
        assert row["cases"][0]["reuse_enabled"] is True
        assert row["cases"][0].get("copy_middle", True) is True
    assert lossy["prefetch_spill_device"] is False
    assert lossy["ordinary_prefix_reuse_enabled"] is False
    assert prefix["prefetch_spill_device"] is True
    assert prefix["prefetch_middle"] is False
    assert prefix["ordinary_prefix_reuse_enabled"] is True
    assert templ["prefetch_spill_device"] is True
    assert templ["prefetch_middle"] is True
    assert mode_env("prefix_prefetch")["SGLANG_KV_PREFETCH_MIDDLE"] == "0"
    assert mode_env("template_prefetch")["SGLANG_KV_PREFETCH_MIDDLE"] == "1"


def test_staircase_increments_are_vs_lossy_not_headline():
    lat = {
        "lossy_host": {"cache_ready_speedup_ratio_of_means": 2.0},
        "prefix_prefetch": {"cache_ready_speedup_ratio_of_means": 2.2},
        "template_prefetch": {"cache_ready_speedup_ratio_of_means": 2.6},
    }
    steps = staircase_increments(lat)
    assert abs(steps["prefix_prefetch_vs_lossy"] - 1.1) < 1e-9
    assert abs(steps["template_prefetch_vs_prefix"] - (2.6 / 2.2)) < 1e-9
    assert "1.492" not in str(steps)
