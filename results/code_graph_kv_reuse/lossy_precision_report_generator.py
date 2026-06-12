#!/usr/bin/env python3
"""Generate the unified code-graph lossy precision report."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
BASE = ROOT / "results" / "code_graph_kv_reuse"
OUT = BASE / "CODE_GRAPH_LOSSY_PRECISION_REPORT.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bundle_rows(summary: dict, key: str = "by_bundle_type") -> str:
    rows = []
    for bundle, stat in summary[key].items():
        rows.append(
            f"| `{bundle}` | {stat['count']} | {stat['mean']:.3f} | {stat['p90']:.3f} | {stat['tail_rate_050']:.2f} |"
        )
    return "\n".join(rows)


def drift_rows(summary: dict) -> str:
    rows = []
    for bundle, stat in summary["by_candidate_bundle_type"].items():
        rows.append(
            f"| `{bundle}` | {stat['n']} | {stat['mean_token_f1']:.3f} | {stat['json_valid_rate']:.2f} | "
            f"{stat['reuse_risk_match_rate']:.2f} | {stat['high_risk_drift_rate']:.2f} |"
        )
    return "\n".join(rows)


def passrate_overlap() -> dict:
    cg_cases = set()
    with (BASE / "data" / "code_graph_bundle_table.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cg_cases.add(row["instance_id"])
    pass_cases = set()
    pass_path = ROOT / "results" / "swe_generated_patch_kvcomm" / "qwen2_5_7b_json_30" / "passrate_table.csv"
    with pass_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pass_cases.add(row["instance_id"])
    overlap = sorted(cg_cases & pass_cases)
    (BASE / "data" / "graph_pass1_overlap_cases.json").write_text(
        json.dumps(overlap, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pass_summary = load_json(ROOT / "results" / "passrate_28" / "per_case_summary.json")
    return {
        "pass_cases": len(pass_cases),
        "code_graph_cases": len(cg_cases),
        "overlap": overlap,
        "overlap_n": len(overlap),
        "lossless_pass1": pass_summary.get("lossless_pass1", 0),
        "lossy_pass1": pass_summary.get("lossy_pass1", 0),
        "regressions": pass_summary.get("regressions", []),
    }


def drift_failure_rows(path: Path) -> str:
    counts = Counter()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["failure_mode"]] += 1
    return "\n".join(f"| `{k}` | {v} |" for k, v in sorted(counts.items()))


def readiness_summary(path: Path) -> tuple[str, str]:
    if not path.exists():
        return (
            "| not run | - | - | - | - | - | - |",
            "Graph-aware live patch readiness has not been run yet.",
        )
    by_mode: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_mode.setdefault(row["mode"], []).append(row)

    rows = []
    graph_note = ""
    for mode, records in by_mode.items():
        n = len(records)
        diff = sum(row["diff_extracted"] == "True" for row in records)
        apply_ok = sum(row["apply_ok"] == "True" for row in records)
        gen_err = sum(bool(row["generation_error"]) for row in records)
        search_nf = sum("search not found" in row["synthesis_error"] for row in records)
        exact = sum(row["match_reason"] == "exact_code_content_signature" for row in records)
        cached = sum(float(row["cached_tokens"] or 0) for row in records) / n
        rows.append(f"| `{mode}` | {n} | {diff} | {apply_ok} | {gen_err} | {search_nf} | {exact} | {cached:.1f} |")
        if mode == "graph_aware_lossy":
            graph_note = (
                f"`graph_aware_lossy` reached exact signature match on {exact}/{n} cases and produced "
                f"git-applyable JSON edits on {apply_ok}/{n} cases. This is a readiness signal, not pass@1."
            )
    return "\n".join(rows), graph_note


def main() -> None:
    kv3 = load_json(BASE / "qwen2_5_3b_precision_kv_12targets" / "summary.json")
    kv7 = load_json(BASE / "qwen2_5_7b_precision_kv_8targets" / "summary.json")
    drift = load_json(BASE / "qwen2_5_3b_output_drift_12targets" / "summary.json")
    overlap = passrate_overlap()
    readiness_rows, readiness_note = readiness_summary(BASE / "pass1_graph_aware_13_skiptest" / "mode_diagnostics.csv")
    overlap_list = ", ".join(f"`{x}`" for x in overlap["overlap"])
    regression_rows = "\n".join(
        f"| `{r['instance_id']}` | `{r.get('lossy_fail_step', '')}` | {r.get('lossy_cached_tokens', '')} | {r.get('lossless_cached_tokens', '')} |"
        for r in overlap["regressions"]
    ) or "| none | - | - | - |"
    md = f"""# Code Graph Lossy Reuse Precision Report

## 1. Evidence Stack

这份报告把 code-specific lossy reuse 的精度证据分成三层：

1. **KV stability**：同一个 exact code bundle 在 planner/coder/reviewer prompt 下的 KV 表示是否稳定。
2. **Output drift**：graph-aware bundle 是否改变模型的 JSON risk judgment。
3. **Pass@1 readiness**：已有 pass@1 case 中有多少能映射到 code graph bundle，下一步如何跑 graph-aware lossy。

重要边界：这里仍然只主张 non-degradation readiness，不主张 accuracy improvement。

## 2. KV Stability

### 3B cross-task diagnostic

- Result: `results/code_graph_kv_reuse/qwen2_5_3b_precision_kv_12targets/`
- Records: {kv3['config']['n_records']}; sampled bundle groups: {kv3['config']['sampled_groups']}
- Overall mean/p90/max d_norm: {kv3['summary']['overall']['mean']:.3f} / {kv3['summary']['overall']['p90']:.3f} / {kv3['summary']['overall']['max']:.3f}
- Tail `d_norm>0.5`: {kv3['summary']['overall']['tail_rate_050']:.2f}

| bundle | n | mean d_norm | p90 | tail d_norm>0.5 |
|---|---:|---:|---:|---:|
{bundle_rows(kv3['summary'])}

### 7B robustness sanity

- Result: `results/code_graph_kv_reuse/qwen2_5_7b_precision_kv_8targets/`
- Records: {kv7['config']['n_records']}; sampled bundle groups: {kv7['config']['sampled_groups']}
- Overall mean/p90/max d_norm: {kv7['summary']['overall']['mean']:.3f} / {kv7['summary']['overall']['p90']:.3f} / {kv7['summary']['overall']['max']:.3f}
- Tail `d_norm>0.5`: {kv7['summary']['overall']['tail_rate_050']:.2f}

| bundle | n | mean d_norm | p90 | tail d_norm>0.5 |
|---|---:|---:|---:|---:|
{bundle_rows(kv7['summary'])}

Interpretation: KV 层面上，`import_dependency_bundle` 和 `call_neighborhood_1hop` 在 3B/7B 上都没有出现高 tail risk，是值得进入输出和 pass@1 验证的候选。

## 3. Output Drift

- Result: `results/code_graph_kv_reuse/qwen2_5_3b_output_drift_12targets/`
- Model: `{drift['config']['model']}`
- Baseline: same target/role 的 `ast_function_only` deterministic JSON output
- Candidates: graph-aware bundles
- Pairs: {drift['summary']['overall']['pairs']}
- Overall mean token F1: {drift['summary']['overall']['mean_token_f1']['mean']:.3f}
- JSON valid rate: {drift['summary']['overall']['json_valid_rate']:.2f}
- Reuse-risk match rate: {drift['summary']['overall']['reuse_risk_match_rate']:.2f}

| candidate bundle | n | mean token F1 | JSON valid | reuse-risk match | high-risk drift |
|---|---:|---:|---:|---:|---:|
{drift_rows(drift['summary'])}

Failure breakdown:

| failure mode | n |
|---|---:|
{drift_failure_rows(BASE / 'qwen2_5_3b_output_drift_12targets' / 'output_drift_table.csv')}

Interpretation: 输出层比 KV 层敏感得多。`call_neighborhood_1hop` 的 reuse-risk match 最好，但 relevant-symbol/missing-context 字段仍然经常漂移。因此 graph-aware lossy 进入 pass@1 前必须加 output-level gate，不能只凭 KV distance 放行。

## 4. Pass@1 Readiness Audit

- Existing paired pass@1 cases: {overlap['pass_cases']}
- Code graph census cases: {overlap['code_graph_cases']}
- Overlap available for graph-aware pass@1: {overlap['overlap_n']}
- Overlap cases: {overlap_list}
- Current paired pass@1 baseline: lossless {overlap['lossless_pass1']}/28, current lossy {overlap['lossy_pass1']}/28

Current lossy regression(s):

| case | lossy fail step | lossy cached | lossless cached |
|---|---|---:|---:|
{regression_rows}

Interpretation: P3 可以先在这 {overlap['overlap_n']} 个 overlap cases 上跑，而不是重新扩 100/500 cases。成功标准是 `graph_aware_lossy` regression count 不超过 current lossy，并解释所有 failure mode。

### Live patch-harness readiness, 13 overlap cases

- Result: `results/code_graph_kv_reuse/pass1_graph_aware_13_skiptest/`
- Candidate tests were skipped in this run; it checks generation, JSON-edit synthesis, `git apply --check`, and reuse metadata.

| mode | n | diff extracted | apply ok | generation errors | search-not-found | exact signature match | mean cached tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
{readiness_rows}

Interpretation: {readiness_note} The dominant remaining failure is `search not found`, so the next pass@1 run should either evaluate only applyable patches first or tighten the JSON-edit prompt to force search strings from the graph bundle.

## 5. Policy Implication

- 默认候选：`call_neighborhood_1hop`，因为 KV 稳定且 output reuse-risk match 最高。
- 保守候选：`import_dependency_bundle`，KV 稳定但 output risk drift 需要 gate。
- 任务诊断：`test_target_bundle`，只用于 SWE-style failure analysis，不作为默认 runtime policy。
- 拒绝条件：JSON invalid、baseline symbol coverage 低、reuse-risk label 改变、或 KV `d_norm>0.5`。

## 6. Next Required Experiment

Run `graph_aware_lossy` paired pass@1 on the {overlap['overlap_n']} overlap cases using the policy above. This is the only missing step before writing a paper-level non-degradation claim for code graph-aware lossy reuse.
"""
    OUT.write_text(md, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
