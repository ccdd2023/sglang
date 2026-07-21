#!/usr/bin/env python3
"""Build the append-only 2026-07-21 weekly audit revision."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import build_weekly_report_archive as legacy


ROOT = Path("/home/gfy/CodeMAS_Project")
SOURCE = (
    ROOT
    / "sglang-kvflow-worktrees/coding-aware/docs/kvflow/"
    "WEEKLY_RESEARCH_AUDIT_20260718.md"
)
OUTPUT = ROOT / "kvflow-reports/weekly_reports_20260718"
ARTIFACT = ROOT / "kvflow-artifacts/impactkv_native_frontier_v3_20260720"
E0 = ROOT / "kvflow-artifacts/impactkv_exact_middle_e0_20260718/E0_FINAL_VERDICT.json"
E1 = ROOT / "kvflow-artifacts/impactkv_exact_middle_e1_20260718/E1_FINAL_VERDICT.json"
E2 = ROOT / "kvflow-artifacts/impactkv_exact_middle_e2_20260718/server/E2_RESULT.json"
C2 = ROOT / "kvflow-artifacts/impactkv_component_c2_20260718/COMPONENT_ROUTE_FINAL_VERDICT.json"
STEM = "2026-07-21_IMPACTKV_KVFLOW_WEEKLY_RESEARCH_AUDIT_REVISION"


TRACKS = (
    ("cacheblend", "native", "CacheBlend native"),
    ("cacheblend", "float16", "CacheBlend FP16"),
    ("kvcomm", "native", "KVCOMM native FP32"),
    ("kvcomm", "float16", "KVCOMM FP16"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result(method: str, dtype: str) -> dict[str, Any]:
    path = ARTIFACT / f"runs/{method}/{dtype}/formal/FORMAL_RESULT.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["_path"] = str(path)
    value["_sha256"] = _sha(path)
    return value


def _candidate(method: str, dtype: str) -> dict[str, Any]:
    values = _result(method, dtype)["candidates"]
    if len(values) != 1:
        raise ValueError(f"expected one frozen candidate for {method}/{dtype}")
    return values[0]


def _accuracy_counts(method: str, dtype: str) -> tuple[int, int]:
    root = ARTIFACT / f"runs/{method}/{dtype}/formal"
    output: list[int] = []
    for mode in ("dense", "reuse"):
        paths = sorted(root.glob(f"accuracy.{mode}.*.scored.jsonl"))
        if len(paths) != 1:
            raise ValueError(f"missing accuracy ledger: {method}/{dtype}/{mode}")
        rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line]
        rows = [
            row
            for row in rows
            if not row.get("metadata", {}).get("source_observation")
        ]
        if len(rows) != 225 or any(row.get("error") for row in rows):
            raise ValueError(f"invalid formal accuracy coverage: {paths[0]}")
        output.append(sum(bool(row.get("passed")) for row in rows))
    return output[0], output[1]


def _integrity() -> dict[str, Any]:
    issues: list[str] = []
    markers = list(ARTIFACT.glob("runs/*/*/formal/*.complete.json"))
    for marker in markers:
        value = json.loads(marker.read_text(encoding="utf-8"))
        raw = Path(value["raw_ledger"])
        rows = sum(bool(line.strip()) for line in raw.open(encoding="utf-8"))
        if (
            value.get("status") != "complete"
            or rows != value.get("records")
            or _sha(raw) != value.get("sha256")
        ):
            issues.append(str(marker))
    if len(markers) != 16 or issues:
        raise ValueError(
            f"formal integrity failed: markers={len(markers)}, issues={issues}"
        )
    return {"formal_completion_markers": len(markers), "issues": issues}


def _fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def _baseline_markdown() -> str:
    rows = []
    context_rows = []
    amortized_rows = []
    evidence = []
    for method, dtype, label in TRACKS:
        result = _result(method, dtype)
        candidate = result["candidates"][0]
        accuracy = candidate["accuracy_drop_pp"]
        latency = candidate["latency"]
        dense, reuse = _accuracy_counts(method, dtype)
        scope = "/".join(latency["context"].keys())
        rows.append(
            f"| {label} | `{candidate['config_id']}` | {scope} | "
            f"{dense}/225 → {reuse}/225 | {_fmt(accuracy['estimate'])} "
            f"[{_fmt(accuracy['low'])}, {_fmt(accuracy['high'])}] | "
            f"{_fmt(latency['mean_saving_percent'])}% "
            f"[{_fmt(latency['saving_bootstrap']['low'])}, "
            f"{_fmt(latency['saving_bootstrap']['high'])}] |"
        )
        context = latency["context"]
        context_rows.append(
            f"| {label} | "
            + " | ".join(
                f"{_fmt(context[key]['estimate'])}%" if key in context else "unsupported"
                for key in ("2048", "4096", "8192")
            )
            + " |"
        )
        amortized_rows.append(
            f"| {label} | {_fmt(latency['mean_cache_build_ms'])} ms | "
            f"{_fmt(latency['amortized']['1']['mean_saving_percent'])}% | "
            f"{_fmt(latency['amortized']['4']['mean_saving_percent'])}% | "
            f"{_fmt(latency['amortized']['8']['mean_saving_percent'])}% | "
            f"{_fmt(latency['amortized']['16']['mean_saving_percent'])}% |"
        )
        evidence.append(f"- `{result['_path']}`  \n  SHA-256: `{result['_sha256']}`")

    e0 = json.loads(E0.read_text(encoding="utf-8"))
    e1 = json.loads(E1.read_text(encoding="utf-8"))
    e2 = json.loads(E2.read_text(encoding="utf-8"))
    c2 = json.loads(C2.read_text(encoding="utf-8"))
    return f"""
## 8A. 7 月 18 日后完成的 exact executor 与外部对比基线

本节是 2026-07-21 修订新增的权威更新。它不改写 7 月 18 日报告原件，
也不改变 V11/V12 registration、threshold、verdict 或 sealed holdout。

### 8A.1 统一协议和解释边界

- 模型：Qwen2.5-Coder-3B-Instruct；硬件：单张 RTX 4090；
- accuracy：64-case calibration 后冻结配置，formal 为 225 cases；
- latency：2 warmups + 5 measured rounds，bootstrap unit 为 case；
- errors、fallbacks 和缺失观测不得从分母中静默删除；
- 每个方法只和自己的原生 dense engine 比较；跨引擎绝对毫秒仅作描述；
- 在线 TTFT 不含首次 cache build，因此必须同时报告摊销结果。

### 8A.2 225-case formal accuracy 与在线 TTFT

accuracy drop 定义为 Dense pass rate 减 Reuse pass rate；负值只表示本次
reuse 多通过少量 cases。所有区间都包含零，不能据此声称 lossy reuse 提升精度。

| 方法 | 冻结配置 | latency scope | Dense → Reuse | accuracy drop pp [95% CI] | online TTFT saving [95% CI] |
|---|---|---|---:|---:|---:|
{chr(10).join(rows)}

所有正式 latency rows 均为 physical reuse=100%、fallback=0、error=0。
KVCOMM native FP32 的 8K 注册 cases 全部 OOM，故正式 scope 明确限制为 2K/4K；
该失败范围没有被静默排除后冒充完整结果。

### 8A.3 上下文长度与收益

| 方法 | 2K | 4K | 8K |
|---|---:|---:|---:|
{chr(10).join(context_rows)}

### 8A.4 cache build 与 N 次复用摊销

| 方法 | mean build | N=1 | N=4 | N=8 | N=16 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(amortized_rows)}

CacheBlend 的约 79% 是 cache 已建好后的在线收益；N=1/2 总体更慢，约第三次
复用才 break even。KVCOMM 的 build 更容易摊销，但其 Transformers 多代理
执行拓扑与 SGLang/CacheBlend 不同，不能用绝对毫秒直接排名。

### 8A.5 E0–E2 exact executor 和 component route

- E0：`{e0['decision']}`，identity={e0['e0_formal_identity']}，
  mechanical mismatch={e0['e0_mechanical_mismatch_tokens']}；
- E1：{e1['concurrent_arrival_executor']['completion_identity']} completion identity，
  但 serialized staging 的 exact/dense makespan ratio=
  {_fmt(e1['posthoc_performance']['exact_over_dense_round_makespan_ratio'], 3)}，因此无并发加速 claim；
- E2：status=`{e2['status']}`，completion identity={e2['completion_identity']}/120，
  p95 makespan ratio={_fmt(e2['p95_round_makespan_ratio_exact_over_dense'], 3)}，
  即受控 same-position exact 场景约节省
  {_fmt(100 * (1 - e2['p95_round_makespan_ratio_exact_over_dense']), 2)}%；
- component split：`{c2['final_status']}`。V-only AUC 未过 0.6，K-only 虽过
  AUC，但 paired improvement CI low≤0；不授权沿 C1/C2 继续 threshold sweep。

E2 是受控 exact、same-position、same-causal-prefix server gate，不是 lossy
coding-aware formal，也不代表通用 production policy 已完成。

### 8A.6 对下一路线的约束

当前可实现底座是 E2 的约 17.7% exact system saving。新的 lossy 路线必须：

1. 以 N=4、包含 cache build 的系统收益为 primary metric；
2. 第一晋级线为 N=4≥20%，先超过 CacheBlend 的约 18%；
3. 以约 71%（KVCOMM FP16）和 81%（KVCOMM native）为描述性高阶目标；
4. accuracy drop 95% CI high≤3 pp；
5. 从“挑可复制模块”改为“加载大共享上下文，再 task-guided repair 高风险 tokens”；
6. K/V 联合 repair，不继续 C1/C2 component-only 路线；
7. formal 配置必须在 64-case calibration 后冻结，225 cases 不得重调。

### 8A.7 权威制品

{chr(10).join(evidence)}

- `{E2}`  
  SHA-256: `{_sha(E2)}`
- `{C2}`  
  SHA-256: `{_sha(C2)}`
- `{ARTIFACT / 'invalidated/kvcomm_native_formal_thread_timeout_20260721/INVALIDATION.json'}`
"""


def build_markdown(source: Path) -> str:
    original = source.read_text(encoding="utf-8")
    revised = original.replace(
        "# ImpactKV / KVFlow 本周研究审计与版本演进报告",
        "# ImpactKV / KVFlow 本周研究审计、基线复现与路线修订报告",
        1,
    )
    revised = revised.replace(
        "日期：2026-07-18  \n覆盖范围：2026-07-14 至 2026-07-18；必要时回溯 7 月 10–13 日结果",
        "原始审计日期：2026-07-18  \n修订日期：2026-07-21  \n覆盖范围：2026-07-14 至 2026-07-21；必要时回溯 7 月 10–13 日结果",
        1,
    )
    revised = revised.replace(
        "当前结论：coding-aware V9–V12 均未通过各自预注册门槛；KVCOMM 分支接口已解耦，但生产 model-server 路径仍待 canary",
        "当前结论：V9–V12 与 component-split 路线均未过门槛；受控 E2 exact executor 已通过，CacheBlend/KVCOMM 原生对比基线已完成，但 coding-aware 尚无可与之竞争的正式系统结果",
        1,
    )
    revised = revised.replace(
        "- 不能声称 KVCOMM 已完成 production model-server GPU 验证。",
        "- 外部 KVCOMM Transformers 原生 GPU 基线已完成；仍不能将它表述为 SGLang shared-core 的通用 production policy 验证。",
        1,
    )
    marker = "## 9. 建议的下一阶段"
    if marker not in revised:
        raise ValueError("weekly audit insertion marker is missing")
    revised = revised.replace(marker, _baseline_markdown() + "\n\n" + marker, 1)
    revised = revised.replace(
        "优先级上，V13 低于 KVCOMM production-path 修复。",
        "2026-07-21 修订：外部原生基线和受控 E2 已完成。下一步优先 E3 common-harness 对齐，再注册独立 Task-Guided Repair；不得恢复 V11/V12/C1/C2 threshold sweep。",
        1,
    )
    revised += (
        "\n\n---\n\n修订版生成规则：7 月 18 日原始 MD/HTML/PDF 保持字节不变；"
        "本文件的新增数字来自上述 SHA 锁定机器制品。\n"
    )
    return revised


def _slides() -> str:
    values = {label: _candidate(method, dtype) for method, dtype, label in TRACKS}
    e2 = json.loads(E2.read_text(encoding="utf-8"))
    slides = [
        legacy._render_slide(
            43,
            "V3 PROTOCOL",
            "原生对比基线：先冻结配置，再跑 225-case formal",
            """
            <div class="metrics"><div class="metric"><small>Accuracy</small><strong>225</strong><p>129 HumanEval + 96 MBPP</p></div><div class="metric"><small>Latency</small><strong>2 + 5</strong><p>warmups + measured rounds</p></div><div class="metric"><small>Inference unit</small><strong>CASE</strong><p>不把重复轮次当独立样本</p></div></div>
            <div class="callout">errors、fallbacks、OOM 与缺失观测保留在覆盖审计中；每个方法只和自己的原生 Dense 比较。</div>
            """,
        ),
        legacy._render_slide(
            44,
            "FORMAL FRONTIER",
            "225-case 精度没有显示显著损失；在线 TTFT 节省 79%–88%",
            """
            <table><thead><tr><th>方法</th><th>Dense → Reuse</th><th>Accuracy drop 95% CI</th><th>Online saving</th></tr></thead><tbody>
            <tr><td>CacheBlend native</td><td>167 → 169</td><td>[-4.89, 3.11] pp</td><td>79.01%</td></tr>
            <tr><td>CacheBlend FP16</td><td>167 → 167</td><td>[-3.56, 3.56] pp</td><td>78.66%</td></tr>
            <tr><td>KVCOMM native</td><td>161 → 164</td><td>[-3.56, 0.44] pp</td><td>88.31%</td></tr>
            <tr><td>KVCOMM FP16</td><td>161 → 163</td><td>[-3.56, 1.33] pp</td><td>79.27%</td></tr></tbody></table>
            <div class="callout warn">负 accuracy drop 只表示本次 reuse 多过少量 cases；CI 包含 0，不能声称精度提升。</div>
            """,
        ),
        legacy._render_slide(
            45,
            "CONTEXT SCALING",
            "上下文越长，在线 reuse 收益越高",
            """
            <table><thead><tr><th>方法</th><th>2K</th><th>4K</th><th>8K</th></tr></thead><tbody>
            <tr><td>CacheBlend native</td><td>71.45%</td><td>81.26%</td><td>84.33%</td></tr>
            <tr><td>CacheBlend FP16</td><td>71.89%</td><td>79.77%</td><td>84.33%</td></tr>
            <tr><td>KVCOMM native</td><td>83.67%</td><td>92.95%</td><td>OOM / unsupported</td></tr>
            <tr><td>KVCOMM FP16</td><td>57.83%</td><td>85.39%</td><td>94.59%</td></tr></tbody></table>
            <div class="callout">KVCOMM native 的 8K OOM 被明确记为 unsupported，而不是从分母删除。</div>
            """,
        ),
        legacy._render_slide(
            46,
            "SYSTEM AMORTIZATION",
            "在线 headline 不等于系统收益：cache build 必须摊销",
            """
            <table><thead><tr><th>方法</th><th>Build</th><th>N=1</th><th>N=4</th><th>N=8</th></tr></thead><tbody>
            <tr><td>CacheBlend native</td><td>419 ms</td><td>-165.63%</td><td>17.85%</td><td>48.43%</td></tr>
            <tr><td>CacheBlend FP16</td><td>416 ms</td><td>-164.20%</td><td>17.95%</td><td>48.31%</td></tr>
            <tr><td>KVCOMM native</td><td>209 ms</td><td>60.18%</td><td>81.28%</td><td>84.79%</td></tr>
            <tr><td>KVCOMM FP16</td><td>145 ms</td><td>47.11%</td><td>71.23%</td><td>75.25%</td></tr></tbody></table>
            <div class="callout ok">下一阶段 primary metric 冻结为 N=4 system saving；第一晋级线为 20%。</div>
            """,
        ),
        legacy._render_slide(
            47,
            "INTERNAL EXECUTOR",
            "E2 exact 已建立约 17.7% 的受控 SGLang 系统底座",
            f"""
            <div class="metrics"><div class="metric ok"><small>E2 status</small><strong>{html.escape(e2['status'])}</strong><p>120/120 completion identity</p></div><div class="metric"><small>p95 ratio</small><strong>{_fmt(e2['p95_round_makespan_ratio_exact_over_dense'], 3)}</strong><p>exact / Dense makespan</p></div><div class="metric"><small>Lifecycle</small><strong>0</strong><p>allocator / lease growth</p></div></div>
            <div class="twocol"><div class="panel selected"><h3>已建立</h3><ul><li>same position</li><li>same causal prefix</li><li>controlled exact middle</li><li>positive concurrent makespan CI</li></ul></div><div class="panel warn"><h3>仍未建立</h3><ul><li>lossy coding policy</li><li>position-shift exactness</li><li>225-case workflow accuracy</li><li>通用 production policy</li></ul></div></div>
            """,
        ),
        legacy._render_slide(
            48,
            "NEXT ROUTE",
            "从 safe-reuse 分类转向 Task-Guided Repair",
            """
            <div class="flow"><div class="node">加载大共享 K/V</div><div class="arrow">→</div><div class="node accent">任务特征定位高风险 token</div><div class="arrow">→</div><div class="node">K/V 联合 repair</div><div class="arrow">→</div><div class="node">N=4 系统收益</div></div>
            <div class="metrics"><div class="metric"><small>Advance</small><strong>≥20%</strong><p>先超过 CacheBlend N=4</p></div><div class="metric"><small>Parity</small><strong>≥71%</strong><p>KVCOMM FP16 level</p></div><div class="metric"><small>Stretch</small><strong>≥82%</strong><p>best baseline level</p></div></div>
            <div class="callout bad">accuracy CI high 必须 ≤3 pp；不恢复 V11/V12/C1/C2 threshold sweep。</div>
            """,
        ),
    ]
    return "".join(slides)


def build_deck() -> str:
    deck = legacy.build_deck().replace("2026-07-18", "2026-07-21")
    deck = deck.replace(
        "本周研究审计与版本演进",
        "本周研究审计、基线复现与路线修订",
    )
    return deck.replace("</main><script>", _slides() + "</main><script>", 1)


def _update_index(output: Path) -> None:
    revision = f"""
    <section class="current"><h2>7 月 21 日完整修订版</h2>
    <code>{STEM}</code><p>新增 E0–E2、component-route 与 CacheBlend/KVCOMM V3 正式基线；7 月 18 日原件保持不变。</p>
    <a href="{STEM}.html">HTML</a><a href="{STEM}.pdf">PDF</a><a href="{STEM}.md">Markdown</a></section>
    """
    index = legacy.build_index(output)
    index = index.replace("<main>", "<main>" + revision, 1)
    (output / "INDEX.html").write_text(index, encoding="utf-8")


def _write_manifest(output: Path) -> None:
    files = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "MANIFEST.sha256"
    )
    (output / "MANIFEST.sha256").write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    integrity = _integrity()
    markdown = build_markdown(source)
    md_path = output / f"{STEM}.md"
    html_path = output / f"{STEM}.html"
    pdf_path = output / f"{STEM}.pdf"
    qa_path = output / "VISUAL_VALIDATION_20260721.json"
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(build_deck(), encoding="utf-8")
    visual = legacy.render_pdf(html_path, pdf_path, qa_path)
    if not visual["passed"] or visual["slides"] != 48:
        raise RuntimeError(f"revision visual validation failed: {visual}")
    _update_index(output)
    readme = output / "README.md"
    prior = readme.read_text(encoding="utf-8") if readme.exists() else ""
    note = (
        "\n## 2026-07-21 完整修订版\n\n"
        f"- `{STEM}.md` / `.html` / `.pdf`：截至 7 月 21 日的新权威快照；\n"
        "- 7 月 18 日原始三件套保持不变；\n"
        f"- slides/pages：{visual['slides']}/{visual['pdf_pages']}，overflow=0。\n"
    )
    if "## 2026-07-21 完整修订版" not in prior:
        readme.write_text(prior.rstrip() + "\n" + note, encoding="utf-8")
    _write_manifest(output)
    print(
        json.dumps(
            {
                "stem": STEM,
                "integrity": integrity,
                "markdown_lines": len(markdown.splitlines()),
                "visual": visual,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
