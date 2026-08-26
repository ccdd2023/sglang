"""7B slices must match frozen 137185 RESULT, not a literal speedup."""

from __future__ import annotations

from derive_7b_137185_slices import DEFAULT_ART, analyze
from derive_96092_slices import read_json


def test_n4_matches_frozen_137185_result() -> None:
    result = read_json(DEFAULT_ART / "RESULT.json")
    assert result["status"] == "COMPLETE"
    assert result["qwen25_rope_ok"] is True
    slices = analyze(DEFAULT_ART)
    n4 = float(result["latency"]["n4_including_one_source_build_speedup"])
    assert abs(slices["n_use_including_one_source_build"]["4"] - n4) < 1e-6
    assert slices["status"] == "DERIVED_FROM_FROZEN_137185"
    assert slices["not_a_new_gpu_arm"] is True
    assert slices["dataset"]["target_groups"] == 235
    assert slices["dataset"]["tasks"] == 24
    assert sum(slices["dataset"]["repos"].values()) == 235
    ttft = slices["ttft_ms"]
    assert ttft["pairs"] == 705
    assert abs(
        ttft["dense_mean"] / ttft["reuse_mean"]
        - result["latency"]["cache_ready_speedup_ratio_of_means"]
    ) < 1e-12
    assert abs(
        ttft["paired_saving_median"] - result["latency"]["paired_ttft_saving_median"]
    ) < 1e-12
    assert ttft["reuse_p50"] < ttft["dense_p50"]
    assert ttft["reuse_p90"] < ttft["dense_p90"]
    assert ttft["reuse_p99"] < ttft["dense_p99"]
    assert len(slices["group_scatter"]["copied_fraction"]) == 235
