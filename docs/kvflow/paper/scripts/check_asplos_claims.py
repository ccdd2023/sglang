#!/usr/bin/env python3
"""Check ASPLOS submission claims against the frozen SWE-bench RESULT.json.

Drives the shipped paper sources (main.tex + sections/*.tex) and the
frozen campaign artifact. Exit 0 only if format, headlines, and
forbidden-claim guards hold.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from impactkv_paths import artifact_root, engine_root

PAPER = Path(__file__).resolve().parents[1]
ARTIFACTS = artifact_root()
ENGINE_ROOT = engine_root()
RESULT = (
    ARTIFACTS
    / "impactkv_swebench_7b_file_modules_prefixkey_20260824/RESULT.json"
)
RESULT_30B = (
    ARTIFACTS / "impactkv_swebench_prerotated_file_modules_20260818/RESULT.json"
)
MOTIVATION = (
    ARTIFACTS
    / "impactkv_swebench_7b_file_modules_prefixkey_20260824/MOTIVATION.json"
)
COPIER_MOTIVATION = (
    ARTIFACTS / "impactkv_swebench_7b_sota_copiers_20260824/COPIER_MOTIVATION.json"
)
SLICES_7B = (
    ARTIFACTS
    / "impactkv_swebench_7b_file_modules_prefixkey_20260824/SLICES.json"
)
SLICES_30B = RESULT_30B.with_name("SLICES.json")
PLAN_7B = RESULT.with_name("PLAN.json")
PLAN_30B = RESULT_30B.with_name("PLAN.json")
ATTN_BLOCK = (
    ARTIFACTS
    / "impactkv_global_block_attention_20260806/frozen26_r2/RESULT.json"
)
ATTN_SPARSITY = (
    ARTIFACTS / "impactkv_attention_sparsity_20260806/frozen20/RESULT.json"
)
ATTN_FOUR = (
    ARTIFACTS
    / "impactkv_common_prompt_attention_kv_mechanism_20260813/FOUR_ARM_RESULT.json"
)
ENGINE = ENGINE_ROOT / "python/sglang/srt/mem_cache/kvcomm_exact.py"


def paper_tex() -> str:
    parts = [(PAPER / "main.tex").read_text(encoding="utf-8")]
    for path in sorted((PAPER / "sections").glob("*.tex")):
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def seven_b_body_text() -> str:
    """Headline 7B sources: abstracts + body sections, not the 30B appendix."""
    parts: list[str] = []
    for path in (PAPER / "main.tex", PAPER / "main_article.tex"):
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    for path in sorted((PAPER / "sections").glob("*.tex")):
        if path.name == "appendix.tex":
            continue
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def leftover_30b_win_rate(body: str) -> str | None:
    if "96.5" in body:
        return "30B 96.5 paired win rate leftover in 7B body"
    return None


def _tex_has_int(text: str, n: int) -> bool:
    raw = str(int(n))
    tokens = [raw]
    if len(raw) > 3:
        tokens.append(f"{raw[:-3]},{raw[-3:]}")
        tokens.append(f"{raw[:-3]}{{,}}{raw[-3:]}")
    return any(re.search(rf"(?<![\d{{,]){re.escape(token)}(?![\d,}}])", text) for token in tokens)


def thirty_b_plan_mean_lengths(slices: dict) -> tuple[int, int]:
    zero = slices["skipped_zero_shift"]
    groups = int(slices["dataset"]["target_groups"])
    copied = int(round(zero["plan_copied_tokens"] / groups))
    prompt = int(round(zero["plan_target_tokens"] / groups))
    return copied, prompt


def leftover_30b_plan_lengths(body: str, copied_30: int, prompt_30: int) -> str | None:
    hits = []
    if _tex_has_int(body, copied_30):
        hits.append("copied")
    if _tex_has_int(body, prompt_30):
        hits.append("prompt")
    if hits:
        return f"30B PLAN {'/'.join(hits)} length leftover in 7B body"
    return None


def plan_target_length_percentiles(plan_path: Path) -> tuple[int, int, int]:
    import statistics

    groups = json.loads(plan_path.read_text(encoding="utf-8"))["groups"]
    lengths = sorted(len(group["target_input_ids"]) for group in groups)
    if len(lengths) < 2:
        raise ValueError(f"{plan_path} PLAN too small")
    p90 = lengths[round(0.9 * (len(lengths) - 1))]
    return int(statistics.median(lengths)), int(p90), int(lengths[-1])


def main() -> int:
    failures: list[str] = []
    tex = paper_tex()
    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    result = json.loads(RESULT.read_text(encoding="utf-8"))

    if "sigplan,anonymous,review,nonacm" not in main_tex:
        failures.append("documentclass missing ASPLOS options")
    if "ACM-Reference-Format" not in main_tex:
        failures.append("bibliography is not ACM-Reference-Format")
    if "biblatex" in main_tex:
        failures.append("biblatex must not be used")

    title_m = re.search(r"\\title\{([^}]+)\}", main_tex)
    title = title_m.group(1) if title_m else ""
    if re.search(r"\bServing\b", title) and "prefill" not in title.lower():
        failures.append("title uses Serving without a prefill qualifier")
    if re.search(r"SWE-bench", title, re.I):
        failures.append("title must not name SWE-bench")
    if not re.search(r"sequential one-token prefill", tex):
        failures.append("paper must scope eval as sequential one-token prefill")
    leftover = leftover_30b_win_rate(seven_b_body_text())
    if leftover:
        failures.append(leftover)
    eval_tex = (PAPER / "sections/evaluation.tex").read_text(encoding="utf-8")
    main_tex_full = (PAPER / "main.tex").read_text(encoding="utf-8")
    if "99.3" not in eval_tex or "99.3" not in main_tex_full:
        failures.append("7B body/abstract must report 137185 paired win rate 99.3")
    article = PAPER / "main_article.tex"
    article_tex = article.read_text(encoding="utf-8") if article.exists() else ""
    if article.exists() and "99.3" not in article_tex:
        failures.append("main_article abstract must report 137185 paired win rate 99.3")
    if article.exists():
        art_title_m = re.search(r"\\title\{([^}]+)\}", article_tex)
        art_title = art_title_m.group(1) if art_title_m else ""
        if art_title != title:
            failures.append("main_article title diverges from main.tex")
        if re.search(r"Dense-recomputes the span if residual RoPE", article_tex):
            failures.append("main_article stale fail-closed: residual Δ≠0 always discarded")

    lat = result["latency"]
    mech = result["mechanism"]
    agr = result["one_token_output_agreement"]
    checks = [
        (abs(lat["cache_ready_speedup_ratio_of_means"] - 1.4919) < 5e-4, "cache-ready"),
        (abs(lat["n4_including_one_source_build_speedup"] - 0.9048) < 5e-4, "n4"),
        (abs(lat["paired_ttft_win_rate"] - 0.9929) < 5e-4, "paired win rate"),
        (mech["copy_events"] == 1684, "copies"),
        (mech["expected_copy_events"] == 1684, "expected copies"),
        (mech["fallback_events"] == 0, "fallback"),
        (agr["not_accuracy"] is True, "agreement not accuracy"),
        (result["prefetch"] is False, "prefetch off"),
        (result["ordinary_prefix_reuse"] is False, "prefix reuse off"),
        (result["status"] == "COMPLETE", "COMPLETE"),
        (result.get("qwen25_rope_ok") is True, "qwen25 rope"),
        (result.get("not_30b_swebench_plan") is True, "not 30b plan"),
    ]
    thirty = json.loads(RESULT_30B.read_text(encoding="utf-8"))
    if thirty.get("status") != "COMPLETE":
        failures.append("appendix 30B RESULT is not COMPLETE")
    if abs(thirty["latency"]["cache_ready_speedup_ratio_of_means"] - 1.3748) > 5e-4:
        failures.append("appendix 30B cache-ready drifted")
    for ok, name in checks:
        if not ok:
            failures.append(f"RESULT.json field drifted: {name}")

    for needle in (
        r"1\.492",
        r"93\.6",
        r"99\.3",
        r"1\.375",
        r"1684/1684",
        r"94\.8",
        r"fail-closed",
        r"prefetch",
        r"tab:eval-30b",
        r"tab:eval-scales",
        r"fig:motivation-coverage",
        r"fig:motivation-extra",
        r"fig:attn-proxy",
        r"tab:admit-ablation",
        r"fig:ttft-cdf",
        r"fig:eval-slices",
        r"fig:copied-speedup",
        r"fig:prefix-on",
        r"fig:admit",
        r"fig:tv-locus",
        r"fig:attn-heatmap",
        r"fig:kv-heatmap",
        r"585\.9",
        r"392\.7",
        r"1178",
        r"CacheSlide",
        r"RedKnot",
        r"KVLink",
        r"Notes-at-Prefill",
        r"offline",
        r"oracle",
    ):
        if not re.search(needle, tex):
            failures.append(f"paper missing {needle}")

    if re.search(r"\bSOTA\b", tex):
        failures.append("submission tex contains SOTA")
    if "tab:sota-copiers" in tex:
        failures.append("submission tex still uses tab:sota-copiers")
    if "zero observed fallbacks do not justify" not in tex.lower():
        failures.append("fail-closed must not be justified by zero fallbacks")
    if not re.search(r"offline.{0,80}oracle|oracle.{0,80}offline", tex, re.I):
        failures.append("PLAN must be named an offline oracle")
    if "do not full-decode the 17" not in tex:
        failures.append("must decline full-decode of the 17 copier groups")
    prefix_on_path = ARTIFACTS / "impactkv_swebench_7b_prefix_on_20260825/RESULT.json"
    if prefix_on_path.exists():
        prefix_on = json.loads(prefix_on_path.read_text(encoding="utf-8"))
        if prefix_on.get("status") == "COMPLETE":
            if "tab:7b-prefix-on" not in tex:
                failures.append("COMPLETE 7B prefix-on RESULT requires tab:7b-prefix-on")
            if "do not turn ordinary prefix on" in tex:
                failures.append("COMPLETE prefix-on RESULT still declined in the paper")
            if prefix_on.get("prefetch") is not False:
                failures.append("prefix-on RESULT must keep prefetch off")
            if prefix_on.get("not_eval_summary") is not True:
                failures.append("prefix-on RESULT must set not_eval_summary")
            if not re.search(r"1\.526", tex):
                failures.append("paper missing prefix-only 1.526")
            if not re.search(r"1\.408", tex):
                failures.append("paper missing same-job lossy-only 1.408")
            if not re.search(r"2\.120", tex):
                failures.append("paper missing dual 2.120")
            if not re.search(r"1\.390", tex):
                failures.append("paper missing copy-on-prefix 1.390")
            if re.search(r"1\.375\\times[\s\S]{0,80}2\.120", tex) or re.search(
                r"2\.120\\times[\s\S]{0,80}1\.375", tex
            ):
                failures.append("prefix-on dual 2.120x bound to 1.375x")
            if re.search(r"1\.492\\times[\s\S]{0,80}2\.120", tex) or re.search(
                r"2\.120\\times[\s\S]{0,80}1\.492", tex
            ):
                failures.append("prefix-on dual 2.120x bound to 1.492x")
        elif "do not turn ordinary prefix on" not in tex:
            failures.append("incomplete prefix-on RESULT must stay declined")
    elif "do not turn ordinary prefix on" not in tex:
        failures.append("must decline a prefix-on increment of the headline job")
    if re.search(r"DS-1000", tex) and re.search(
        r"tab:eval-summary[\s\S]{0,800}DS-1000", tex
    ):
        failures.append("DS-1000 mixed into SWE-bench headline table")
    if re.search(r"1\.375\\times\s+end-to-end", tex):
        failures.append("1.375x claimed as end-to-end")
    if "Several directions remain" in tex:
        failures.append("do not leave copier/P99/500-task as owed future work")
    if "do not run a same-token 30B CacheBlend" not in tex and (
        "do not run a same-token 30B" not in tex
    ):
        failures.append("must explicitly decline a 30B CacheBlend/KVCOMM arm")
    if "do not replay all of SWE-bench" not in tex:
        failures.append("must explicitly decline a full Verified replay")

    prefetch_campaign = (
        ARTIFACTS
        / "impactkv_swebench_template_prefetch_nextisland_20260821/RESULT.json"
    )
    pref = None
    if prefetch_campaign.exists():
        pref = json.loads(prefetch_campaign.read_text(encoding="utf-8"))
    if "tab:template-prefetch" in tex:
        if pref is None:
            failures.append("tab:template-prefetch without prefetch RESULT.json")
        elif pref.get("status") != "COMPLETE":
            failures.append(
                "tab:template-prefetch cited while prefetch RESULT is not COMPLETE"
            )
        if not re.search(r"later-roles|later\\_roles", tex):
            failures.append("prefetch table must name later-roles / next-island")
        if not re.search(r"next-island", tex):
            failures.append("prefetch table must name next-island")
        if not re.search(r"not a copy win", tex):
            failures.append("prefetch table must say prefetch-only is not a copy win")
        if re.search(r"Hints are oracle", tex):
            failures.append("prefetch table still claims oracle remaining_uses")
    elif pref is not None and pref.get("status") == "COMPLETE":
        failures.append("COMPLETE prefetch RESULT requires tab:template-prefetch")
    if re.search(r"1\.375\\times[\s\S]{0,80}1\.390", tex) or re.search(
        r"1\.390\\times[\s\S]{0,80}1\.375", tex
    ):
        failures.append("1.390x bound to 1.375x")
    if re.search(r"1\.375\\times[\s\S]{0,80}1\.392", tex) or re.search(
        r"1\.392\\times[\s\S]{0,80}1\.375", tex
    ):
        failures.append("1.392x bound to 1.375x")
    if re.search(r"1\.375\\times[\s\S]{0,80}0\.996", tex) or re.search(
        r"0\.996\\times[\s\S]{0,80}1\.375", tex
    ):
        failures.append("0.996x bound to 1.375x")
    if "\\vspace" in tex:
        failures.append("vertical-space squeeze (\\vspace)")

    # 3B activation probe must stay a separate figure from 30B TTFT.
    if "fig:attn-proxy" not in tex:
        failures.append("missing fig:attn-proxy")
    if r"\label{tab:attn-proxy}" in tex:
        failures.append("attn-proxy must be a figure, not a table")
    if "Qwen2.5-Coder-3B" not in tex:
        failures.append("3B probe must name Qwen2.5-Coder-3B")
    if "not 30B native" not in tex:
        failures.append("3B probe must say it is not 30B native")
    for needle in (r"0\.00264", r"0\.0462", r"80\.1"):
        if not re.search(needle, tex):
            failures.append(f"paper missing probe {needle}")
    headline = re.search(
        r"\\label\{tab:eval-summary\}[\s\S]{0,2500}\\end\{tabular\}", tex
    )
    if headline and re.search(
        r"0\.00264|0\.0462|Qwen2.5-Coder-3B|\\bTV\\b", headline.group(0)
    ):
        failures.append("3B TV mixed into SWE-bench headline table")
    if headline and re.search(
        r"1\.390|1\.392|0\.996|0\.970|prefetch-only|prefetch_only",
        headline.group(0),
    ):
        failures.append("prefetch campaign mixed into headline table")
    if headline and re.search(
        r"1\.155|4\.526|8\.490",
        headline.group(0),
    ):
        failures.append("7B dual-island mixed into headline table")
    if headline and re.search(
        r"tab:7b-prefix-on|copy_on_prefix|1\.526|1\.408|2\.120|1\.390",
        headline.group(0),
    ):
        failures.append("prefix-on increment mixed into headline table")
    if headline and re.search(r"1\.375", headline.group(0)):
        failures.append("30B 1.375 mixed into 7B headline table")
    scales = re.search(
        r"\\label\{tab:eval-scales\}[\s\S]{0,800}?\\end\{tabular\}", tex
    )
    if not scales:
        failures.append("missing tab:eval-scales table body")
    else:
        if "137185" not in scales.group(0) or "96092" not in scales.group(0):
            failures.append("scale table must name both 137185 and 96092")
        if re.search(r"1\.492\\times|1\.375\\times", scales.group(0)):
            failures.append("scale table must not bind the two speedups")
        if "96.5" in scales.group(0):
            failures.append("30B win rate mixed into scale table")
    if "Qwen2.5-Coder-7B-Instruct" not in tex:
        failures.append("headline campaign must name Qwen2.5-Coder-7B-Instruct")
    if "fig:motivation-coverage" not in tex:
        failures.append("missing fig:motivation-coverage")
    if r"\label{tab:motivation-coverage}" in tex or r"\label{tab:motivation-extra}" in tex:
        failures.append("motivation coverage/extra must be figures")
    if headline and re.search(
        r"1\.525|1\.523|2\.136|0\.933|1\.407|80\.4|74\.5|66\.4",
        headline.group(0),
    ):
        failures.append("7B SWE-bench campaign mixed into headline table")
    if headline and re.search(
        r"2\.032|1\.312|948/948|LCS copier|2\.100|1\.883|89\.4|91\.9|1\.384|1\.287|117.?965",
        headline.group(0),
    ):
        failures.append("general LCS copier mixed into headline table")
    if headline and re.search(
        r"1\.231|1\.347|1\.685|1\.331|1\.588|1\.119|1\.551|1\.131|1\.917|1\.175|194.624|convenience sample|rolling-6 turns",
        headline.group(0),
    ):
        failures.append("dataset/ablation slices mixed into headline table")
    if "tab:dataset" not in tex:
        failures.append("missing tab:dataset")
    if "tab:ablate-islands" not in tex:
        failures.append("missing tab:ablate-islands")
    if "tab:ablate-delta" not in tex:
        failures.append("missing tab:ablate-delta")
    if "tab:ablate-frac" not in tex:
        failures.append("missing tab:ablate-frac")
    if "tab:nuse" in tex or "fig:nuse" in tex or "fig_nuse" in tex:
        failures.append("paper must not bill source-inclusive N-use")
    if re.search(r"0\.905", tex):
        failures.append("paper still bills 7B N=4 0.905")
    if re.search(r"0\.841", tex):
        failures.append("paper still bills 30B N=4 0.841")
    if re.search(r"1\.271\\times", tex):
        failures.append("paper bills unique-source-once N=4 counterfactual")
    if "tab:repo-slice" not in tex:
        failures.append("missing tab:repo-slice")
    if "not a ranking" not in tex.lower() and "do not rank" not in tex.lower():
        failures.append("repo slice must not be a ranking")
    if "rolling-6" not in tex:
        failures.append("dataset card must say rolling-6 turns")
    if "convenience sample" not in tex:
        failures.append("dataset card must name convenience sample")
    if "not a new GPU arm" not in tex:
        failures.append("ablation caption must say not a new GPU arm")
    if "Replaying 30B token" not in tex and "replaying 30B token" not in tex:
        failures.append("must forbid replaying 30B token ids on 7B")
    slices_path = SLICES_30B
    if slices_path.exists():
        slices = json.loads(slices_path.read_text(encoding="utf-8"))
        if slices.get("status") != "DERIVED_FROM_FROZEN_96092":
            failures.append("SLICES.json not derived from frozen 96092")
        if abs(slices["island_count_slices"]["3"]["cache_ready_speedup"] - 1.6853) > 5e-3:
            failures.append("island-count 3-slice drifted")
        if slices["dataset"]["tasks"] != 24:
            failures.append("dataset card tasks must be 24")
        if abs(slices["abs_delta_slices"][">=3000"]["cache_ready_speedup"] - 1.588) > 5e-3:
            failures.append("delta >=3000 slice drifted")
        if abs(slices["repo_slices"]["django"]["cache_ready_speedup"] - 1.551) > 5e-3:
            failures.append("django repo slice drifted")
        frac = slices["copied_fraction_quartiles"]["slices"]
        if abs(frac["Q4"]["cache_ready_speedup"] - 1.917) > 5e-3:
            failures.append("copied-fraction Q4 drifted")
        shared = slices["cross_group_source_amortization"]
        if shared.get("not_headline") is not True:
            failures.append("cross-group N=4 must be marked not_headline")
        if abs(shared["n4_per_group_build"] - thirty["latency"]["n4_including_one_source_build_speedup"]) > 1e-6:
            failures.append("shared-slice n4 drifted from 30B RESULT")
    if SLICES_7B.exists():
        s7 = json.loads(SLICES_7B.read_text(encoding="utf-8"))
        if s7.get("status") != "DERIVED_FROM_FROZEN_137185":
            failures.append("7B SLICES.json not derived from frozen 137185")
        if abs(s7["island_count_slices"]["3"]["cache_ready_speedup"] - 1.8332) > 5e-3:
            failures.append("7B island-count 3-slice drifted")
        if abs(
            s7["cross_group_source_amortization"]["n4_per_group_build"]
            - result["latency"]["n4_including_one_source_build_speedup"]
        ) > 1e-6:
            failures.append("7B shared-slice n4 drifted from headline RESULT")
        ttft = s7.get("ttft_ms") or {}
        if int(ttft.get("pairs", 0)) != 705:
            failures.append("7B SLICES ttft pairs drifted")
        if abs(
            ttft.get("dense_mean", 0) / ttft.get("reuse_mean", 1)
            - result["latency"]["cache_ready_speedup_ratio_of_means"]
        ) > 1e-9:
            failures.append("7B SLICES mean TTFT drifted from cache-ready")
        if abs(
            ttft.get("paired_saving_median", 0)
            - result["latency"]["paired_ttft_saving_median"]
        ) > 1e-9:
            failures.append("7B SLICES median saving drifted from RESULT")
    if MOTIVATION.exists():
        mot = json.loads(MOTIVATION.read_text(encoding="utf-8"))
        if mot.get("disjoint_radix_and_file_islands") is not True:
            failures.append("motivation must prove disjoint radix and file islands")
        if int(mot.get("lcp_island_overlap_tokens", -1)) != 0:
            failures.append("motivation LCP/island overlap must be 0")
        copied_7 = int(round(float(mot["mean_copied_tokens"])))
        prompt_7 = int(round(float(mot["mean_target_tokens"])))
        body_7 = seven_b_body_text()
        if not _tex_has_int(body_7, copied_7):
            failures.append("7B body missing frozen MOTIVATION mean copied tokens")
        if not _tex_has_int(body_7, prompt_7):
            failures.append("7B body missing frozen MOTIVATION mean prompt tokens")
        template = (PAPER / "sections/template.tex").read_text(encoding="utf-8")
        if not _tex_has_int(template, copied_7) or not _tex_has_int(template, prompt_7):
            failures.append("M0 compiler paragraph missing 7B PLAN mean lengths")
        if SLICES_30B.exists():
            copied_30, prompt_30 = thirty_b_plan_mean_lengths(
                json.loads(SLICES_30B.read_text(encoding="utf-8"))
            )
            leftover_len = leftover_30b_plan_lengths(body_7, copied_30, prompt_30)
            if leftover_len:
                failures.append(leftover_len)
        if PLAN_7B.exists():
            med, p90, mx = plan_target_length_percentiles(PLAN_7B)
            for n, name in ((med, "median"), (p90, "p90"), (mx, "max")):
                if not _tex_has_int(template, n):
                    failures.append(f"M0 missing 7B PLAN {name} target length")
        if PLAN_30B.exists():
            med30, p90_30, max30 = plan_target_length_percentiles(PLAN_30B)
            for n, name in ((med30, "median"), (p90_30, "p90"), (max30, "max")):
                if _tex_has_int(body_7, n):
                    failures.append(f"30B PLAN {name} target length leftover in 7B body")
    elif "fig:motivation-coverage" in tex:
        failures.append("fig:motivation-coverage without MOTIVATION.json")
    if COPIER_MOTIVATION.exists():
        extra = json.loads(COPIER_MOTIVATION.read_text(encoding="utf-8"))
        if extra.get("status") != "DERIVED_FROM_FROZEN_137400":
            failures.append("copier motivation not derived from 137400")
        spans = extra["spans"]
        agr_c = extra["agreement"]
        if int(spans["kvcomm_extra_tokens"]) != 194624:
            failures.append("unconstrained extra tokens drifted")
        if int(agr_c["file_agrees_kvcomm_differs"]) != 51:
            failures.append("file-vs-KVCOMM disagreement drifted")
        if re.search(
            r"194\{,\}624 extra tokens in every group"
            r"|extra unconstrained tokens in\s+every group",
            tex,
        ):
            failures.append("194624 extra tokens claimed per group")
        if "campaign total" not in tex:
            failures.append("paper must call 194624 extra tokens a campaign total")
        if "194" not in tex or "624" not in tex:
            failures.append("paper must report unconstrained extra tokens")
        if not re.search(r"51\$?\s*pairs", tex):
            failures.append("paper must report file-module pairs KVCOMM misses")
        kinds = extra.get("span_types")
        if not kinds:
            failures.append("COPIER_MOTIVATION.json missing span_types")
        else:
            camp = kinds["campaign"]
            subset = kinds["file_agrees_kvcomm_differs"]
            if int(camp["extra_tokens"]) != 194624:
                failures.append("span-type extra tokens drifted")
            if int(camp["by_class"]["tool_log"]) != 117965:
                failures.append("span-type tool_log drifted")
            if int(subset["pairs"]) != 51 or int(subset["groups"]) != 17:
                failures.append("span-type 51-pair subset drifted")
            if int(subset["disjoint_from_file_island"]) != 0:
                failures.append("51-pair extra is not all file-adjacent")
            if kinds.get("not_admitted_repository_code") is not True:
                failures.append("span types must not be admitted repository_code")
            if "tab:copier-extra-kinds" not in tex:
                failures.append("missing tab:copier-extra-kinds")
            if not re.search(r"117\{,\}965|117,965", tex):
                failures.append("paper must report extra tool-log tokens")
            if not re.search(r"17\$?\s*groups", tex):
                failures.append("paper must report the 17 disagreement groups")
            if headline and re.search(
                r"117.?965|12.?639|8.?156|tab:copier-extra-kinds",
                headline.group(0),
            ):
                failures.append("copier span types mixed into headline table")
    elif "fig:motivation-extra" in tex:
        failures.append("fig:motivation-extra without COPIER_MOTIVATION.json")
    if "tab:7b-dual-island" in tex:
        if "Qwen2.5-Coder-7B-Instruct" not in tex:
            failures.append("7B dual-island table must name Qwen2.5-Coder-7B-Instruct")
        if not re.search(
            r"tab:7b-dual-island[\s\S]{0,600}not the 30B PLAN",
            tex,
        ) and "Not the 30B PLAN" not in tex:
            failures.append("7B dual-island table must say it is not the 30B PLAN")
    if re.search(r"1\.375\\times[\s\S]{0,80}0\.00264", tex) or re.search(
        r"0\.00264[\s\S]{0,80}1\.375\\times", tex
    ):
        failures.append("suffix TV bound to 1.375x")
    if re.search(r"1\.492\\times[\s\S]{0,80}0\.00264", tex) or re.search(
        r"0\.00264[\s\S]{0,80}1\.492\\times", tex
    ):
        failures.append("suffix TV bound to 1.492x")
    if re.search(r"1\.375\\times[\s\S]{0,80}2\.032", tex) or re.search(
        r"2\.032\\times[\s\S]{0,80}1\.375", tex
    ):
        failures.append("general LCS 2.032x bound to 1.375x")
    if re.search(r"1\.375\\times[\s\S]{0,80}1\.312", tex) or re.search(
        r"1\.312\\times[\s\S]{0,80}1\.375", tex
    ):
        failures.append("general LCS 1.312x bound to 1.375x")
    lcs_campaign = ARTIFACTS / "impactkv_swebench_general_lcs_20260822/RESULT.json"
    if lcs_campaign.exists():
        lcs = json.loads(lcs_campaign.read_text(encoding="utf-8"))
        if lcs.get("status") == "COMPLETE":
            if "tab:general-lcs" in tex:
                failures.append("unconstrained LCS must not be a comparison table")
            if "not a comparison target" not in tex:
                failures.append("paper must decline unconstrained LCS as a comparison target")
            if "87.1" not in tex:
                failures.append("paper must report the LCS one-token drop")
            if lcs.get("not_96092_coding_plan") is not True:
                failures.append("general LCS RESULT must set not_96092_coding_plan")
    if re.search(r"1\.375\\times[\s\S]{0,80}1\.525", tex) or re.search(
        r"1\.525\\times[\s\S]{0,80}1\.375", tex
    ):
        failures.append("7B coding 1.525x bound to 1.375x")
    if re.search(r"1\.375\\times[\s\S]{0,80}1\.523", tex) or re.search(
        r"1\.523\\times[\s\S]{0,80}1\.375", tex
    ):
        failures.append("7B coding 1.523x bound to 1.375x")
    if re.search(r"1\.375\\times[\s\S]{0,80}1\.492", tex) or re.search(
        r"1\.492\\times[\s\S]{0,80}1\.375", tex
    ):
        failures.append("7B prefix-key 1.492x bound to 1.375x")
    if re.search(r"1\.375\\times[\s\S]{0,80}2\.100", tex) or re.search(
        r"2\.100\\times[\s\S]{0,80}1\.375", tex
    ):
        failures.append("KVCOMM-style 2.100x bound to 1.375x")
    if re.search(r"1\.375\\times[\s\S]{0,80}1\.883", tex) or re.search(
        r"1\.883\\times[\s\S]{0,80}1\.375", tex
    ):
        failures.append("CacheBlend-style 1.883x bound to 1.375x")
    if re.search(r"1\.375\\times[\s\S]{0,80}2\.136", tex) or re.search(
        r"2\.136\\times[\s\S]{0,80}1\.375", tex
    ):
        failures.append("7B LCS 2.136x bound to 1.375x")
    sota_path = ARTIFACTS / "impactkv_swebench_7b_sota_copiers_20260824/RESULT.json"
    if sota_path.exists():
        sota = json.loads(sota_path.read_text(encoding="utf-8"))
        if sota.get("status") == "COMPLETE":
            if "tab:admit-ablation" not in tex:
                failures.append("COMPLETE 7B copier clones require tab:admit-ablation")
            if sota.get("not_native_cacheblend_or_kvcomm_stack") is not True:
                failures.append("copier RESULT must set not_native_cacheblend_or_kvcomm_stack")
            if "same-engine" not in tex.lower():
                failures.append("copier table must say same-engine")
            if "not native" not in tex.lower():
                failures.append("copier table must say not native stacks")
            for arm_name in ("coding", "kvcomm_style", "cacheblend_style"):
                arm = sota.get(arm_name) or {}
                if arm.get("status") != "COMPLETE":
                    failures.append(f"copier arm {arm_name} not COMPLETE")
                if arm.get("qwen25_rope_ok") is not True:
                    failures.append(f"copier arm {arm_name} missing qwen25_rope_ok")
    seven_b = (
        ARTIFACTS
        / "impactkv_swebench_7b_file_modules_prefixkey_20260824/RESULT.json"
    )
    if seven_b.exists():
        seven = json.loads(seven_b.read_text(encoding="utf-8"))
        if seven.get("status") == "COMPLETE" and seven.get("qwen25_rope_ok") is True:
            if "tab:7b-swebench" not in tex:
                failures.append("COMPLETE 7B SWE-bench RESULT requires tab:7b-swebench")
            if seven.get("model") != "Qwen2.5-Coder-7B-Instruct":
                failures.append("7B SWE-bench RESULT must name Qwen2.5-Coder-7B-Instruct")
            if seven.get("not_30b_swebench_plan") is not True:
                failures.append("7B SWE-bench RESULT must set not_30b_swebench_plan")
            mech = seven["mechanism"]
            if mech["copy_events"] != mech["expected_copy_events"]:
                failures.append("7B SWE-bench copies drifted")
            if mech["fallback_events"] != 0:
                failures.append("7B SWE-bench fallback drifted")
            if re.search(
                r"tab:7b-swebench[\s\S]{0,1200}\\textbf\{LCS\}",
                tex,
            ):
                failures.append("7B SWE-bench table must not treat LCS as a comparison column")
            if not re.search(
                r"tab:7b-swebench[\s\S]{0,800}not the seven-group dual-island",
                tex,
            ) and "not the seven-group dual-island" not in tex:
                failures.append("7B SWE-bench table must not be the dual-island table")
    if "CacheWise" not in tex:
        failures.append("paper must cite CacheWise / ForeCache")

    attn = json.loads(ATTN_BLOCK.read_text(encoding="utf-8"))
    sparse = json.loads(ATTN_SPARSITY.read_text(encoding="utf-8"))
    four = json.loads(ATTN_FOUR.read_text(encoding="utf-8"))
    if abs(attn["aggregate"]["suffix_tv"]["median"] - 0.00264263) > 5e-6:
        failures.append("frozen26 suffix TV drifted")
    if abs(attn["aggregate"]["formation_tv"]["median"] - 0.046176) > 5e-5:
        failures.append("frozen26 formation TV drifted")
    if abs(sparse["aggregate"]["median_global_top10_attention_mass"] - 0.80068) > 5e-4:
        failures.append("frozen20 sparsity mass drifted")
    if four["status"] != "COMPLETE":
        failures.append("four-arm probe not COMPLETE")
    if not re.search(r"does not\s+estimate Attention", tex):
        failures.append("admit must still refuse Attention scores")
    if tex.count("sec:attn-proxy") < 4:
        failures.append("sec:attn-proxy under-cited (need method/eval refs)")
    if re.search(r"Dense-recomputes the span if residual RoPE", tex):
        failures.append("stale fail-closed: residual Δ≠0 always discarded")
    if "mechanically invalid" not in tex:
        failures.append("fail-closed must be mechanical invalid copy")

    engine = ENGINE.read_text(encoding="utf-8")
    if "copy_rope_delta = logical_rope_delta - source_pre_rotate_delta" not in engine:
        failures.append("engine residual formula missing")
    if '"event": "target_fallback"' not in engine:
        failures.append("engine target_fallback event missing")

    inputs = re.findall(r"\\input\{sections/([^}]+)\}", main_tex)
    expected_body = [
        "introduction",
        "background",
        "motivation",
        "problem",
        "template",
        "kv-management",
        "implementation",
        "evaluation",
    ]
    if inputs[:8] != expected_body:
        failures.append(f"main.tex body input order {inputs[:8]}")

    background = (PAPER / "sections/background.tex").read_text(encoding="utf-8")
    motivation = (PAPER / "sections/motivation.tex").read_text(encoding="utf-8")
    design = (PAPER / "sections/problem.tex").read_text(encoding="utf-8")
    template = (PAPER / "sections/template.tex").read_text(encoding="utf-8")
    kvman = (PAPER / "sections/kv-management.tex").read_text(encoding="utf-8")
    if r"\subsection{Limitation}" not in background:
        failures.append("Background missing Limitation subsection")
    if r"\subsection{Opportunities}" not in motivation:
        failures.append("Motivation missing Opportunities subsection")
    if r"\begin{table" in motivation:
        failures.append("Motivation still uses tables instead of charts")
    for lab in (
        "fig:tv-locus",
        "fig:attn-heatmap",
        "fig:kv-heatmap",
        "fig:motivation-coverage",
        "fig:motivation-extra",
        "fig:attn-proxy",
    ):
        if rf"\label{{{lab}}}" not in motivation:
            failures.append(f"Motivation missing {lab}")
    heat_pos = motivation.find(r"\label{fig:attn-heatmap}")
    opp_pos = motivation.find(r"\subsection{Opportunities}")
    if heat_pos == -1 or opp_pos == -1 or heat_pos > opp_pos:
        failures.append("Opportunities must follow attention heatmap")
    if r"\label{fig:architecture}" not in design:
        failures.append("Design missing fig:architecture")
    if r"\label{fig:template-process}" not in template:
        failures.append("M0 missing fig:template-process")
    if r"\label{fig:kv-reuse}" not in kvman:
        failures.append("M2 missing fig:kv-reuse")
    impl = (PAPER / "sections/implementation.tex").read_text(encoding="utf-8")
    if "ordinary_prefix_reuse_enabled" in impl:
        failures.append("listing key too long; wraps the M1 comment")
    if "# M1 off" not in impl or "# M3 off" not in impl:
        failures.append("listing must keep intact M1/M3 off comments")
    if r"\begin{figure*}" in eval_tex:
        failures.append("eval body uses figure*")
    eval_nocomment = re.sub(r"(?<!\\)%.*", "", eval_tex)
    for spec in re.findall(r"\\includegraphics\[([^\]]*)\]", eval_nocomment):
        if r"\columnwidth" not in spec:
            failures.append(f"eval includegraphics not columnwidth: {spec}")
        if r"\textwidth" in spec:
            failures.append("eval includegraphics uses textwidth")
    for needle in ("Hardware and software", "Baselines", "Dataset"):
        if needle not in eval_tex:
            failures.append(f"Setup missing {needle}")
    eval_heads = [
        (r"\\subsection\{Setup\}", "Setup"),
        (r"\\subsection\{Overall results\}", "Overall"),
        (r"\\subsection\{Ablation of each module\}", "Ablation"),
        (r"\\subsection\{Sensitivity\}", "Sensitivity"),
    ]
    head_pos: list[int] = []
    for pat, name in eval_heads:
        m = re.search(pat, eval_tex)
        if not m:
            failures.append(f"eval missing {name} heading")
            continue
        head_pos.append(m.start())
    if len(head_pos) == 4 and head_pos != sorted(head_pos):
        failures.append("eval heading order is not Setup/Overall/Ablation/Sensitivity")
    for lab in (
        "tab:admit-ablation",
        "fig:admit",
        "tab:7b-prefix-on",
        "fig:prefix-on",
        "tab:copier-extra-kinds",
        "fig:template-process",
        "fig:architecture",
        "fig:kv-reuse",
        "fig:mechanism",
        "fig:coverage",
        "tab:eval-scales",
        "fig:motivation-coverage",
        "fig:motivation-extra",
        "fig:attn-proxy",
    ):
        n = len(re.findall(rf"\\label\{{{re.escape(lab)}\}}", tex))
        if n != 1:
            failures.append(f"label {lab} count {n}")

    if failures:
        print("FAIL")
        for item in failures:
            print("-", item)
        return 1
    print("PASS")
    print("cache_ready", lat["cache_ready_speedup_ratio_of_means"])
    print("n4", lat["n4_including_one_source_build_speedup"])
    print("copies", mech["copy_events"], "/", mech["expected_copy_events"])
    print("fallback", mech["fallback_events"])
    print("agreement", agr["fraction"], "not_accuracy", agr["not_accuracy"])
    print("prefetch", result["prefetch"])
    return 0


def test_asplos_submission_claims() -> None:
    assert main() == 0


def test_leftover_30b_win_rate_flags_abstract_text() -> None:
    assert leftover_30b_win_rate(r"paired TTFT win rate $96.5\%$") is not None
    assert leftover_30b_win_rate(r"paired TTFT win rate $99.3\%$") is None


def test_leftover_30b_plan_lengths_flags_96092_means() -> None:
    slices = json.loads(SLICES_30B.read_text(encoding="utf-8"))
    copied_30, prompt_30 = thirty_b_plan_mean_lengths(slices)
    mot = json.loads(MOTIVATION.read_text(encoding="utf-8"))
    copied_7 = int(round(float(mot["mean_copied_tokens"])))
    prompt_7 = int(round(float(mot["mean_target_tokens"])))
    assert copied_30 != copied_7
    assert prompt_30 != prompt_7
    def latex_int(n: int) -> str:
        raw = str(int(n))
        if len(raw) <= 3:
            return raw
        return raw[:-3] + "{,}" + raw[-3:]

    assert leftover_30b_plan_lengths(
        rf"mean copied ${copied_30}$ against ${prompt_30}$ tokens",
        copied_30,
        prompt_30,
    ) is not None
    assert leftover_30b_plan_lengths(
        rf"mean copied ${latex_int(copied_30)}$ prompt ${latex_int(prompt_30)}$",
        copied_30,
        prompt_30,
    ) is not None
    assert leftover_30b_plan_lengths(
        rf"mean copied ${copied_7}$ against ${prompt_7}$ tokens",
        copied_30,
        prompt_30,
    ) is None
    body = seven_b_body_text()
    assert leftover_30b_plan_lengths(body, copied_30, prompt_30) is None
    assert _tex_has_int(body, copied_7)
    assert _tex_has_int(body, prompt_7)
    template = (PAPER / "sections/template.tex").read_text(encoding="utf-8")
    assert _tex_has_int(template, copied_7)
    assert _tex_has_int(template, prompt_7)
    med, p90, mx = plan_target_length_percentiles(PLAN_7B)
    assert _tex_has_int(template, med)
    assert _tex_has_int(template, p90)
    assert _tex_has_int(template, mx)


def test_paper_does_not_bill_source_inclusive_n4() -> None:
    tex = paper_tex()
    assert "tab:nuse" not in tex
    assert "fig:nuse" not in tex
    assert "fig_nuse" not in tex
    assert "0.905" not in tex
    assert "0.841" not in tex
    assert "source-inclusive $N$-use speedup" in tex


def test_submission_drops_sota_table_and_cites_shifted_kv() -> None:
    tex = paper_tex()
    assert "tab:sota-copiers" not in tex
    assert "tab:admit-ablation" in tex
    assert not re.search(r"\bSOTA\b", tex)
    for name in ("CacheSlide", "RedKnot", "KVLink", "Notes-at-Prefill"):
        assert name in tex
    assert "offline" in tex.lower() and "oracle" in tex.lower()
    assert leftover_30b_win_rate(seven_b_body_text()) is None


def test_narrative_order_and_columnwidth() -> None:
    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    inputs = re.findall(r"\\input\{sections/([^}]+)\}", main_tex)
    assert inputs[:8] == [
        "introduction",
        "background",
        "motivation",
        "problem",
        "template",
        "kv-management",
        "implementation",
        "evaluation",
    ]
    eval_tex = (PAPER / "sections/evaluation.tex").read_text(encoding="utf-8")
    assert eval_tex.find(r"\subsection{Setup}") < eval_tex.find(
        r"\subsection{Overall results}"
    )
    assert eval_tex.find(r"\subsection{Overall results}") < eval_tex.find(
        r"\subsection{Ablation of each module}"
    )
    assert eval_tex.find(r"\subsection{Ablation of each module}") < eval_tex.find(
        r"\subsection{Sensitivity}"
    )
    assert r"\label{tab:eval-scales}" in eval_tex
    assert eval_tex.find(r"\subsection{Overall results}") < eval_tex.find(
        r"\label{tab:eval-scales}"
    )
    assert r"\begin{figure*}" not in eval_tex
    for spec in re.findall(r"\\includegraphics\[([^\]]*)\]", eval_tex):
        assert r"\columnwidth" in spec
        assert r"\textwidth" not in spec
    motivation = (PAPER / "sections/motivation.tex").read_text(encoding="utf-8")
    assert r"\subsection{Opportunities}" in motivation
    assert r"\begin{table" not in motivation
    assert r"\label{fig:motivation-coverage}" in motivation
    assert r"\label{fig:motivation-extra}" in motivation
    assert r"\label{fig:attn-proxy}" in motivation
    assert motivation.find(r"\label{fig:attn-heatmap}") < motivation.find(
        r"\subsection{Opportunities}"
    )
    template = (PAPER / "sections/template.tex").read_text(encoding="utf-8")
    kvman = (PAPER / "sections/kv-management.tex").read_text(encoding="utf-8")
    assert r"\label{fig:template-process}" in template
    assert r"\label{fig:kv-reuse}" in kvman


def test_seven_b_body_including_abstracts_has_no_96_5() -> None:
    body = seven_b_body_text()
    assert leftover_30b_win_rate(body) is None
    assert "99.3" in body
    assert "Qwen2.5-Coder-7B-Instruct" in body
    article = PAPER / "main_article.tex"
    assert article.exists()
    article_tex = article.read_text(encoding="utf-8")
    assert "96.5" not in article_tex
    assert "99.3" in article_tex
    assert "1.492" in article_tex
    assert "0.905" not in article_tex
    assert "0.841" not in article_tex
    assert "93.6" in article_tex


if __name__ == "__main__":
    sys.exit(main())
