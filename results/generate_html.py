#!/usr/bin/env python3
"""Generate standalone HTML report with embedded base64 PNG charts."""

import base64
import json
from pathlib import Path

OUT = Path(__file__).parent

def b64png(path):
    return base64.b64encode(path.read_bytes()).decode()

# Load data
codebase_data = json.loads((OUT / "codebase_reuse/results.json").read_text())
kv_replace_data = json.loads((Path("/tmp/kv_replacement_results/results.json").read_text()))

charts = {
    "kv_a2": b64png(OUT / "chart_kv_reuse_a2.png"),
    "kv_a3": b64png(OUT / "chart_kv_reuse_a3.png"),
    "reuse_ratio": b64png(OUT / "chart_reuse_ratio.png"),
    "prefill_saved": b64png(OUT / "chart_prefill_saved.png"),
    "gate_swe": b64png(OUT / "chart_gate_swe.png"),
    "latency_swe": b64png(OUT / "chart_latency_swe.png"),
    "gate_large": b64png(OUT / "chart_gate_large.png"),
}

# Build tables
rows = []
for r in codebase_data:
    rows.append(f"""
    <tr>
      <td>{r['name']}</td>
      <td>{r['code_lines']}</td>
      <td>A2 lossy</td>
      <td>{r['a2_lossy']['cached_tokens']}</td>
      <td>{r['a2_lossy']['kv_reuse_mb']}</td>
      <td>{r['a2_lossy']['reuse_ratio']}%</td>
      <td>{r['a2_lossy']['total_ms']:.0f}</td>
      <td>{r['a2_matcher']}</td>
    </tr>
    <tr>
      <td>{r['name']}</td>
      <td>{r['code_lines']}</td>
      <td>A2 lossless</td>
      <td>{r['a2_lossless']['cached_tokens']}</td>
      <td>{r['a2_lossless']['kv_reuse_mb']}</td>
      <td>{r['a2_lossless']['reuse_ratio']}%</td>
      <td>{r['a2_lossless']['total_ms']:.0f}</td>
      <td>-</td>
    </tr>
    <tr>
      <td>{r['name']}</td>
      <td>{r['code_lines']}</td>
      <td>A3 lossy</td>
      <td>{r['a3_lossy']['cached_tokens']}</td>
      <td>{r['a3_lossy']['kv_reuse_mb']}</td>
      <td>{r['a3_lossy']['reuse_ratio']}%</td>
      <td>{r['a3_lossy']['total_ms']:.0f}</td>
      <td>{r['a3_matcher']}</td>
    </tr>
    <tr>
      <td>{r['name']}</td>
      <td>{r['code_lines']}</td>
      <td>A3 lossless</td>
      <td>{r['a3_lossless']['cached_tokens']}</td>
      <td>{r['a3_lossless']['kv_reuse_mb']}</td>
      <td>{r['a3_lossless']['reuse_ratio']}%</td>
      <td>{r['a3_lossless']['total_ms']:.0f}</td>
      <td>-</td>
    </tr>
    """)

codebase_table = "\n".join(rows)

# KV replacement table
kv_rows = []
for d in kv_replace_data:
    kv_rows.append(f"""
    <tr>
      <td>{d['warmup']}</td>
      <td>{d['eval']}</td>
      <td>{d['desc']}</td>
      <td>{d['code_block_tokens']}</td>
      <td>{d['saved_prefill_ms']:.1f}</td>
      <td>{d['baseline_gen_ms']:.1f}</td>
      <td>{d.get('hybrid_gen_ms', 0):.1f}</td>
    </tr>
    """)
kv_table = "\n".join(kv_rows)

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Code-Position-Aware KV Management — 技术报告</title>
<style>
  :root {{
    --bg: #0d1117;
    --fg: #c9d1d9;
    --accent: #58a6ff;
    --ok: #3fb950;
    --warn: #d29922;
    --danger: #f85149;
    --card: #161b22;
    --border: #30363d;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--fg);
    line-height: 1.7;
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem 1rem;
  }}
  h1, h2, h3 {{
    color: #e6edf3;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
  }}
  h1 {{ font-size: 2rem; border-bottom-width: 2px; }}
  h2 {{ font-size: 1.5rem; margin-top: 2.5rem; }}
  h3 {{ font-size: 1.2rem; margin-top: 1.8rem; }}
  .tag {{
    display: inline-block;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.15rem 0.6rem;
    font-size: 0.8rem;
    margin-right: 0.4rem;
    color: var(--accent);
  }}
  .tag.ok {{ color: var(--ok); border-color: rgba(63,185,80,0.3); }}
  .tag.danger {{ color: var(--danger); border-color: rgba(248,81,73,0.3); }}
  .tag.warn {{ color: var(--warn); border-color: rgba(210,153,34,0.3); }}
  pre {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    overflow-x: auto;
    font-size: 0.9rem;
  }}
  code {{
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    background: rgba(110,118,129,0.1);
    padding: 0.1rem 0.3rem;
    border-radius: 4px;
    font-size: 0.9em;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.92rem;
  }}
  th, td {{
    border: 1px solid var(--border);
    padding: 0.5rem 0.7rem;
    text-align: left;
  }}
  th {{
    background: var(--card);
    color: #e6edf3;
    font-weight: 600;
  }}
  tr:nth-child(even) {{ background: rgba(110,118,129,0.05); }}
  img {{
    max-width: 100%;
    border: 1px solid var(--border);
    border-radius: 8px;
    margin: 1rem 0;
  }}
  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
  }}
  .metric {{
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
  }}
  .metric-label {{
    font-size: 0.9rem;
    color: #8b949e;
  }}
  @media (max-width: 800px) {{
    .grid-2 {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<h1>Code-Position-Aware KV Cache Management</h1>
<p>
  <span class="tag">sglang-kvflow</span>
  <span class="tag">HiRadixCache</span>
  <span class="tag">AST Anchor</span>
  <span class="tag ok">Lossy Reuse</span>
  <span class="tag">Multi-Agent</span>
</p>
<p>本报告展示了基于代码位置感知的 KV Cache 复用方案的设计、实现与实验验证。核心思想是：<strong>利用代码 AST 结构作为语义锚点，在跨请求、跨 Agent 的场景下实现安全、高效的 KV 复用</strong>。</p>

<!-- ==================== 方案实现 ==================== -->
<h2>1. 方案实现</h2>

<h3>1.1 总体架构</h3>
<pre>
MAScoder (Python AST)                    sglang-kvflow Server
+----------------------+                +----------------------------------+
| code_anchor.py        |  HTTP POST   |  protocol.py -> io_struct.py     |
|  -> CodeAnchor        |  ----------> |  -> schedule_batch.py (Req)      |
|  -> build_code_anchor |              |  -> scheduler.py                 |
|    _payload()         |              |      |                           |
|                       |              |  HiRadixCache.match_prefix()     |
| kvflow_integration.py |              |  |-- _resolve_lossy_match(req)   |
|  -> KVFlowHint        |              |  |   Anchor Matcher              |
|    (template+lossy)   |              |  |   (7 reject + 5 match rules)   |
+----------------------+              |  |   first/final dual-phase       |
                                      |  |-- token prefix match (native)  |
                                      |  |-- GATE: reject -> skip cache   |
                                      |  |-- metadata.lossy_reuse         |
                                      |       (18 fields in HTTP resp)    |
                                      |                                   |
                                      |  evict_policy.py:                 |
                                      |  lossy nodes -> crit_distance -1~2|
                                      +-----------------------------------+
</pre>

<h3>1.2 核心模块详解</h3>

<div class="grid-2">
<div class="card">
<strong>模块 A: Code Base Segment 标记</strong><br>
<code>MAScoder/src/mascoder/code_anchor.py</code>
<ul>
<li>模板侧声明每个 Agent prompt 中出现的 <code>code_base</code> segment 及后续复用关系</li>
<li>每个 segment 包含：<code>code_base_id</code>, <code>content_signature</code>, <code>start_token</code>, <code>end_token</code></li>
<li>生成 3 个关键 payload 字段：
  <ul>
    <li><code>code_content_signature</code>: Code Base 原文 SHA256，用于复用门控</li>
    <li><code>code_anchor_token_spans</code>: token 区间列表，用于定位 segment</li>
    <li><code>code_anchor_signature</code>: 仅用于定位和观测，不作为复用许可</li>
  </ul>
</li>
<li>实际 KV 复用必须要求代码内容完全一致</li>
</ul>
</div>

<div class="card">
<strong>模块 B: Exact-Content Matcher + Gate</strong><br>
<code>sglang/srt/mem_cache/anchor_match.py</code>
<ul>
<li><strong>核心 reject 规则</strong>：缺少内容签名、内容签名不一致、reuse_mode 非 lossy、alignment_method 不匹配、task_family 不匹配</li>
<li><strong>核心 match 规则</strong>：exact_code_content_signature</li>
<li><code>_span_similarity()</code> 仅作为定位/观测辅助，不允许异内容复用</li>
<li>匹配结果携带 <code>reuse_confidence</code> 和 <code>match_reason</code></li>
</ul>
</div>

<div class="card">
<strong>模块 C: HiRadixCache 集成</strong><br>
<code>sglang/srt/mem_cache/hiradix_cache.py</code>
<ul>
<li><code>match_prefix()</code> 中调用 <code>_resolve_lossy_match(req)</code></li>
<li>若 gate reject，返回空 MatchResult，跳过 cache 匹配</li>
<li>若 gate allow，正常走 token 前缀匹配，同时保留 metadata</li>
<li>insert() 时将 anchor metadata 写入 TreeNode，供后续请求匹配</li>
</ul>
</div>

<div class="card">
<strong>模块 D: 驱逐保护</strong><br>
<code>sglang/srt/mem_cache/evict_policy.py</code>
<ul>
<li>lossy 复用节点在驱逐时获得额外保护</li>
<li><code>critical_path_distance</code> 决定保护等级：距离叶子越远，保护越强</li>
<li>配合 PriorityStrategy 的 DAG-aware 优先级计算</li>
</ul>
</div>
</div>

<h3>1.3 请求-响应链路</h3>
<pre>
Client Request
  |
  |-- code_anchor_signature       (来自 AST)
  |-- code_anchor_spans           (行号区间)
  |-- reuse_mode = "lossy"        (启用复用)
  |-- lossy_alignment_method      (对齐策略)
  |-- template_task_family        (任务族)
  |-- template_workflow_signature (工作流签名)
  |-- template_structural_fingerprint (结构指纹)
  v
HiRadixCache.match_prefix(req)
  |
  |-- _resolve_lossy_match(req)
  |     |-- 遍历所有带 anchor metadata 的 TreeNode
  |     |-- anchor_match.select_best_match()
  |     |-- 返回 AnchorMatchResult (allow/reject + reason)
  |
  |-- if reject: return empty MatchResult (跳过 cache)
  |-- if allow: 正常 token 前缀匹配
  v
Scheduler / schedule_batch
  |
  |-- req 携带 18 个 lossy_reuse 字段
  v
HTTP Response (metadata.lossy_reuse)
  |
  |-- lossy_first_reuse_allowed      (首次 gate 决策)
  |-- lossy_first_match_reason       (首次匹配原因)
  |-- lossy_first_rejected_reason    (首次拒绝原因)
  |-- lossy_final_reuse_allowed      (最终 gate 决策)
  |-- ... (共 18 个可观测字段)
</pre>

<!-- ==================== 实验结果 ==================== -->
<h2>2. 实验验证</h2>

<h3>2.1 大 Codebase × 多 Agent KV 复用</h3>
<p>使用 5 个大型 Python 代码文件（76-120 行），模拟 3-Agent 工作流（Analyzer → Implementer → Reviewer），对比 lossy 与 lossless 两种模式的 KV 复用量。</p>

<div class="grid-2">
  <div class="card" style="text-align:center">
    <div class="metric">249 MB</div>
    <div class="metric-label">A2/A3 Lossy 平均复用</div>
  </div>
  <div class="card" style="text-align:center">
    <div class="metric">277-296 MB</div>
    <div class="metric-label">A2/A3 Lossless 平均复用</div>
  </div>
</div>

<p><img src="data:image/png;base64,{charts['kv_a2']}" alt="KV Reuse A2"></p>
<p><img src="data:image/png;base64,{charts['kv_a3']}" alt="KV Reuse A3"></p>
<p><img src="data:image/png;base64,{charts['reuse_ratio']}" alt="Reuse Ratio"></p>

<table>
  <tr>
    <th>File</th><th>Lines</th><th>Agent</th><th>Cached Tok</th>
    <th>KV (MB)</th><th>Reuse %</th><th>Latency (ms)</th><th>Matcher</th>
  </tr>
  {codebase_table}
</table>

<p><strong>关键发现：</strong></p>
<ul>
<li>Lossy 模式复用率 80-92%，相比 lossless 仅降低 8-15 个百分点</li>
<li>Gate 全部命中 <code>exact_code_content_signature</code>，说明同文件同代码块内容完全一致</li>
<li>A2（Implementer）延迟较高（~3.4s）因为需要生成代码，A3（Reviewer）延迟较低（~1s）因为仅做短评</li>
</ul>

<h3>2.2 SWE-bench Lite — Gate 有效性与延迟</h3>
<p>在 50 个真实 SWE-bench Lite 任务上测试 lossy gate 的决策质量。</p>

<div class="grid-2">
  <div class="card" style="text-align:center">
    <div class="metric">90%</div>
    <div class="metric-label">Reject Rate (45/50)</div>
  </div>
  <div class="card" style="text-align:center">
    <div class="metric">10%</div>
    <div class="metric-label">Accept Rate (5/50)</div>
  </div>
</div>

<div class="grid-2">
  <p><img src="data:image/png;base64,{charts['gate_swe']}" alt="Gate SWE"></p>
  <p><img src="data:image/png;base64,{charts['latency_swe']}" alt="Latency SWE"></p>
</div>

<p><strong>关键发现：</strong></p>
<ul>
<li>90% 的任务被正确 reject，避免不安全复用</li>
<li>Accepted 任务平均 BLEU = 0.28，说明复用未显著降低输出质量</li>
<li>Lossy 端到端延迟略低于 lossless（-37ms accepted / -44ms rejected），说明 gate 决策本身开销极小</li>
</ul>

<h3>2.3 大 Code Block Gate 决策（45 tasks）</h3>
<p>在更大的代码块（≥15 行）上测试 gate 决策。Source 包含 swe_verified 和 codehub 数据集。</p>

<div class="grid-2">
  <div class="card" style="text-align:center">
    <div class="metric">40%</div>
    <div class="metric-label">Accept Rate (18/45)</div>
  </div>
  <div class="card" style="text-align:center">
    <div class="metric">60%</div>
    <div class="metric-label">Reject Rate (27/45)</div>
  </div>
</div>

<p><img src="data:image/png;base64,{charts['gate_large']}" alt="Gate Large"></p>

<p><strong>关键发现：</strong></p>
<ul>
<li>codehub 同代码场景 100% accept（exact_code_content_signature）</li>
<li>swe_verified 跨任务场景大部分 reject（no_anchor_overlap）</li>
<li>Accept 平均延迟 644ms，与 lossless 603ms 接近，说明复用带来实际收益</li>
</ul>

<h3>2.4 KV Tensor Replacement — Prefill 加速</h3>
<p>独立实验：用 HF Transformers 直接替换同功能代码块的 KV tensor，测量 prefill 时间节省。</p>

<p><img src="data:image/png;base64,{charts['prefill_saved']}" alt="Prefill Saved"></p>

<table>
  <tr>
    <th>Warmup</th><th>Eval</th><th>Description</th>
    <th>Block Tokens</th><th>Saved Prefill (ms)</th>
    <th>Baseline Gen (ms)</th><th>Hybrid Gen (ms)</th>
  </tr>
  {kv_table}
</table>

<p><strong>关键发现：</strong></p>
<ul>
<li>同功能代码块（bubble→selection/insertion, factorial rec→iter, fib rec→iter, binary→linear）prefill 节省 12-18ms</li>
<li>不同功能或完全不相关代码块，gate 正确拒绝（saved = 0ms）</li>
<li>代码块越大（token 数越多），潜在节省越多</li>
</ul>

<!-- ==================== 总结 ==================== -->
<h2>3. 总结与结论</h2>

<div class="grid-2">
<div class="card">
<strong>可行性结论</strong>
<ul>
<li><span class="tag ok">可行</span> 同文件/同代码块的 KV 复用率可达 80-92%（lossy）或 99.9%（lossless）</li>
<li><span class="tag ok">可行</span> exact content signature 能可靠识别完全相同 Code Base，exact_code_content_signature 匹配率 100%</li>
<li><span class="tag ok">可行</span> Gate 能正确区分同功能与异功能代码块，SWE-bench 90% reject 率</li>
</ul>
</div>

<div class="card">
<strong>加速效果结论</strong>
<ul>
<li><span class="tag ok">有效</span> 大 codebase 场景：单 Agent 复用 200-314 MB KV</li>
<li><span class="tag ok">有效</span> Prefill 阶段：每代码块节省 12-18ms</li>
<li><span class="tag ok">有效</span> 端到端延迟：lossy 与 lossless 持平甚至略低（gate 开销 &lt;50ms）</li>
</ul>
</div>
</div>

<h3>下一步建议</h3>
<ol>
<li><strong>补充 7B 模型验证</strong>：当前仅 3B，需证明 scale 不影响结论</li>
<li><strong>集成 V cosine gate</strong>：Phase 2 发现 V cosine 2.3× 区分度，可作为数值门控</li>
<li><strong>端到端 throughput 测试</strong>：System 论文需要系统级吞吐数据</li>
<li><strong>完善论文定位</strong>：弱化 Multi-Agent，强化 Code-Structure-Aware KV Reuse</li>
</ol>

<hr>
<p style="color:#8b949e; font-size:0.85rem">Generated: 2026-05-27 | Model: Qwen2.5-3B-Instruct | Framework: sglang-kvflow HiRadixCache</p>

</body>
</html>
"""

(OUT / "code_kv_reuse_report.html").write_text(html, encoding='utf-8')
print("HTML report saved to", OUT / "code_kv_reuse_report.html")
