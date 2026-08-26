"""Motivation charts must match frozen 3B / 7B PLAN JSON, not literals."""

from __future__ import annotations

import json
from pathlib import Path

from build_motivation_heatmaps import extra_token_series, module_tv_by_case
from derive_7b_motivation import DEFAULT_PLAN, group_coverage_series

ATTN = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_global_block_attention_20260806/frozen26_r2/RESULT.json"
)
FOUR = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_common_prompt_attention_kv_mechanism_20260813/"
    "FOUR_ARM_RESULT.json"
)
MOTIVATION = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_swebench_7b_file_modules_prefixkey_20260824/MOTIVATION.json"
)
COPIER = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_swebench_7b_sota_copiers_20260824/COPIER_MOTIVATION.json"
)
FIG = Path(__file__).resolve().parents[1] / "figures"


def test_tv_locus_matches_frozen26_and_four_arm() -> None:
    attn = json.loads(ATTN.read_text(encoding="utf-8"))
    four = json.loads(FOUR.read_text(encoding="utf-8"))
    assert attn["aggregate"]["cases"] == 26
    assert abs(attn["aggregate"]["suffix_tv"]["median"] - 0.00264263) < 5e-6
    assert abs(attn["aggregate"]["formation_tv"]["median"] - 0.046176) < 5e-5
    assert four["status"] == "COMPLETE"
    assert four["summaries"]["coding_aware"]["attention_tv_median"] < 0.01
    assert four["summaries"]["cacheblend"]["attention_tv_median"] > 0.03
    assert (FIG / "fig_tv_locus.pdf").exists()
    assert (FIG / "fig_module_tv.pdf").exists()
    assert (FIG / "fig_attn_heatmap.png").exists()
    assert (FIG / "fig_kv_heatmap.png").exists()
    assert (FIG / "fig_motivation_coverage.pdf").exists()
    assert (FIG / "fig_motivation_extra.pdf").exists()
    assert (FIG / "fig_attn_proxy.pdf").exists()


def test_coverage_and_extra_series_match_frozen_json() -> None:
    mot = json.loads(MOTIVATION.read_text(encoding="utf-8"))
    cov = group_coverage_series(DEFAULT_PLAN)
    n = len(cov["copied_frac"])
    assert n == int(mot["groups"]) == 235
    assert abs(sum(cov["copied_tokens"]) / n - float(mot["mean_copied_tokens"])) < 1e-9
    assert abs(sum(cov["lcp_tokens"]) / n - float(mot["mean_radix_lcp_tokens"])) < 1e-9
    extra = extra_token_series()
    spans = json.loads(COPIER.read_text(encoding="utf-8"))["spans"]
    assert sum(extra["file"]) == int(spans["file_module_copied_tokens"])
    assert sum(extra["kvcomm"]) == int(spans["kvcomm_copied_tokens"])
    assert sum(extra["extra"]) == int(spans["kvcomm_extra_tokens"]) == 194624


def test_module_tv_lines_are_eight_aligned_prompts() -> None:
    series = module_tv_by_case()
    assert len(series["coding"]) == 8
    assert len(series["cacheblend"]) == 8
    assert len(series["kvcomm"]) == 8
    four = json.loads(FOUR.read_text(encoding="utf-8"))
    coding = [v for v in series["coding"] if v == v]
    assert statistics_median_close(
        coding, four["summaries"]["coding_aware"]["attention_tv_median"]
    )


def statistics_median_close(values: list[float], frozen: float, tol: float = 5e-3) -> bool:
    import statistics

    # Per-layer jsonl median can differ slightly from FOUR_ARM summary median.
    return abs(statistics.median(values) - float(frozen)) < tol
