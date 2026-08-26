"""Figure values must come from frozen RESULT JSON, not literals."""

from __future__ import annotations

from build_7b_eval_figures import COPIERS, DEFAULT_ART, PREFIX_ON, figure_values
from derive_96092_slices import read_json


def test_speedup_bars_match_frozen_137185_result() -> None:
    result = read_json(DEFAULT_ART / "RESULT.json")
    values = figure_values(DEFAULT_ART)
    assert result["status"] == "COMPLETE"
    assert abs(values["cache_ready"] - result["latency"]["cache_ready_speedup_ratio_of_means"]) < 1e-12
    assert abs(values["n4"] - result["latency"]["n4_including_one_source_build_speedup"]) < 1e-12
    assert values["copied"] == result["mechanism"]["copy_events"]
    assert values["fallback"] == 0
    assert values["copied"] == values["planned"]
    assert sum(values["island_groups"].values()) == 235
    ttft = values["ttft"]
    assert abs(ttft["dense_mean"] / ttft["reuse_mean"] - values["cache_ready"]) < 1e-12
    assert ttft["reuse_p50"] < ttft["dense_p50"]
    assert len(values["scatter_frac"]) == 235


def test_prefix_and_copier_bars_match_frozen_results() -> None:
    values = figure_values(DEFAULT_ART)
    prefix = read_json(PREFIX_ON / "RESULT.json")
    copiers = read_json(COPIERS / "RESULT.json")
    assert prefix["status"] == "COMPLETE"
    assert copiers["status"] == "COMPLETE"
    assert abs(
        values["prefix_on"]["dual"]
        - prefix["latency"]["dual"]["cache_ready_speedup_ratio_of_means"]
    ) < 1e-12
    assert abs(
        values["copiers"]["kvcomm"]
        - copiers["kvcomm_style"]["latency"]["cache_ready_speedup_ratio_of_means"]
    ) < 1e-12
    assert abs(
        values["copiers"]["file_agree"]
        - copiers["coding"]["one_token_output_agreement"]["fraction"]
    ) < 1e-12
