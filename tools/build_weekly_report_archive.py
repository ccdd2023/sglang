#!/usr/bin/env python3
"""Build a standalone HTML/PDF archive of the ImpactKV weekly reports."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path


DEFAULT_SOURCE = Path(
    "/home/gfy/CodeMAS_Project/sglang-kvflow/results/weekly_reports"
)
DEFAULT_AUDIT = Path(
    "/home/gfy/CodeMAS_Project/sglang-kvflow-worktrees/coding-aware/"
    "docs/kvflow/WEEKLY_RESEARCH_AUDIT_20260718.md"
)
DEFAULT_OUTPUT = Path(
    "/home/gfy/CodeMAS_Project/kvflow-reports/weekly_reports_20260718"
)

CURRENT_STEM = "2026-07-18_IMPACTKV_KVFLOW_WEEKLY_RESEARCH_AUDIT"

HISTORICAL_REPORTS = (
    (
        "2026-07-07_CODE_AWARE_LOSSY_KV_PROGRESS",
        "R1–R27 初期探索",
        "历史探索；不能替代修复后的正式证据。",
    ),
    (
        "2026-07-09_CODE_AWARE_LOSSY_KV_PROGRESS_R28_R39",
        "R28–R39 FRAC / position",
        "包含后来被 runtime audit 撤回的旧 partial-copy 结果。",
    ),
    (
        "2026-07-10_CODE_AWARE_LOSSY_KV_PROGRESS_NODEKIND",
        "Node-kind 中间判决",
        "旧 AST proxy；受 runtime 与 launcher 审计限制。",
    ),
    (
        "2026-07-10_CODE_AWARE_LOSSY_KV_PROGRESS_FINAL",
        "上周最终总结",
        "31.2% headline 已被 7 月 16 日修复后重测推翻。",
    ),
    (
        "2026-07-15_CODE_AWARE_LOSSY_KV_PROGRESS_TASKCONE_L2",
        "TaskCone L2",
        "calibration 观察；matched-control gate 失败，MBPP 未打开。",
    ),
    (
        "2026-07-16_CODE_AWARE_LOSSY_KV_PROGRESS_ASTSPANKV",
        "ASTSpanKV 修复后重测",
        "正式 P1 失败；是本次审计报告的直接版式底稿。",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


INTRO_SLIDES = 12


def _render_slide(
    number: int, label: str, title: str, content: str, *, appendix: bool = False
) -> str:
    kind = "appendix" if appendix else "main"
    return (
        f'<section class="slide {kind}" data-slide="{number}">'
        f'<div class="meta"><span>{label}</span><span>{number:02d}</span></div>'
        f"<h1>{title}</h1><div class=\"content\">{content}</div>"
        f"<footer><span>ImpactKV · KVFlow · 2026-07-18</span>"
        f"<span>{number:02d}</span></footer></section>"
    )


def slide(
    number: int, label: str, title: str, content: str, *, appendix: bool = False
) -> str:
    """Render one of the original audit slides after the reader introduction."""
    return _render_slide(
        number + INTRO_SLIDES,
        label,
        title,
        content,
        appendix=appendix,
    )


def build_deck() -> str:
    slides: list[str] = []
    slides.append(
        _render_slide(
            1,
            "PROJECT · COLLABORATION · AUDIT",
            "ImpactKV / KVFlow：面向 Coding Agent 的中间 KV 复用研究",
            """
            <div class="hero">
              <div class="eyebrow">PROJECT PRIMER · DECOUPLED COLLABORATION · WEEKLY EVIDENCE</div>
              <h2>先解释项目和两位合作者如何分块，再审计本周所有算法版本</h2>
            </div>
            <div class="metrics">
              <div class="metric"><small>研究问题</small><strong>WHAT</strong><p>哪些已计算 KV 可以安全复用？</p></div>
              <div class="metric"><small>合作者问题</small><strong>WHEN / WHERE</strong><p>何时把 KV 搬到哪个设备层级？</p></div>
              <div class="metric warn"><small>本周状态</small><strong>NO HEADLINE</strong><p>V9–V12 均未过预注册门槛</p></div>
            </div>
            <div class="callout">读者不需要预先了解 SGLang、KV cache、AST 或此前版本；前 12 页建立完整背景和协作边界。</div>
            """,
        )
    )
    slides.append(
        _render_slide(
            2,
            "PROJECT IN ONE PAGE",
            "一句话理解项目：少做重复 prefill，同时不破坏代码任务答案",
            """
            <div class="twocol">
              <div class="panel selected"><h3>今天的普通执行</h3><pre>完整 prompt tokens
  → Transformer prefill
  → 为每个 token 生成 K/V
  → 产生第一个输出 token</pre><p>长 prompt 的 prefill 是 TTFT 的主要组成之一。</p></div>
              <div class="panel selected"><h3>我们想做的执行</h3><pre>已有、安全的 prompt 区域
  → 复制历史 K/V
风险区域
  → Dense 重算
  → 更早产生第一个 token</pre><p>加速只能来自少算了多少 prefill。</p></div>
            </div>
            <div class="callout"><b>核心目标：</b>在相同 target prompt 和相同模型下，用“选择性复制 + 选择性重算”降低 TTFT，并以客观测试确认答案没有退化。</div>
            """,
        )
    )
    slides.append(
        _render_slide(
            3,
            "BACKGROUND · KV CACHE",
            "KV cache 是什么，为什么复用它可能节省时间",
            """
            <div class="flow">
              <div><b>Token embeddings</b><span>prompt 被 tokenizer 转成 token IDs</span></div><i>→</i>
              <div><b>Transformer layers</b><span>每层 attention 产生 Key / Value</span></div><i>→</i>
              <div class="selected"><b>KV cache</b><span>保留历史 token 的 attention 状态</span></div><i>→</i>
              <div><b>Decode</b><span>新 token 查询这些 K/V</span></div>
            </div>
            <div class="three">
              <div class="panel"><h3>Prefill</h3><p>首次处理整段 prompt，计算所有层的 K/V；长上下文时成本很高。</p></div>
              <div class="panel"><h3>TTFT</h3><p>Time To First Token：请求发出到第一个模型输出 token 出现的时间。</p></div>
              <div class="panel"><h3>Reuse</h3><p>如果某段 K/V 已经存在，就可能跳过对应 token 的重复 Transformer 计算。</p></div>
            </div>
            <div class="callout warn">K/V 不是纯文本缓存：同一 token 在不同前文下可能产生不同状态，因此“文本相同”不自动等于“KV 安全”。</div>
            """,
        )
    )
    slides.append(
        _render_slide(
            4,
            "BACKGROUND · EXACT VS LOSSY",
            "Exact prefix reuse 很安全；middle lossy reuse 更有机会也更难",
            """
            <div class="twocol">
              <div class="panel ok"><h3>Exact prefix reuse</h3><pre>Request A: [system][repo][task A]
Request B: [system][repo][task B]
           └ exact prefix ┘</pre><p>相同 token、相同前文和相同位置，KV 可以直接复用。</p></div>
              <div class="panel warn"><h3>Middle lossy reuse</h3><pre>Earlier: [issue][old events][foo.py]
Later:   [issue][new events][foo.py][target]
                            └ middle ┘</pre><p>foo.py token 相同，但前文已变化，K/V 只近似相同。</p></div>
            </div>
            <div class="callout"><b>Lossy 的含义：</b>不是压缩数值或修改 token，而是用旧上下文产生的 KV 近似当前上下文本应产生的 KV。</div>
            <table><thead><tr><th>问题</th><th>Exact prefix</th><th>Middle lossy</th></tr></thead><tbody>
              <tr><td>位置</td><td>请求开头</td><td>请求中部任意合法模块</td></tr>
              <tr><td>上下文</td><td>完全相同</td><td>通常不同</td></tr>
              <tr><td>主要风险</td><td>identity / residency</td><td>语义状态偏差与 downstream harm</td></tr>
            </tbody></table>
            """,
        )
    )
    slides.append(
        _render_slide(
            5,
            "WHY CODING WORKFLOWS",
            "Coding Agent 会反复携带代码、工具输出和工作区历史",
            """
            <div class="flow">
              <div><b>Issue</b><span>用户任务与系统指令</span></div><i>→</i>
              <div><b>Inspect</b><span>source views · search · tests</span></div><i>→</i>
              <div><b>Edit</b><span>workspace mutations</span></div><i>→</i>
              <div class="selected"><b>Later turn</b><span>再次携带部分旧模块</span></div>
            </div>
            <div class="twocol">
              <div class="panel"><h3>复用机会</h3><ul><li>同一文件被多次查看</li><li>issue/system instruction 重复出现</li><li>较早 tool output 被后续角色消费</li><li>多 agent 共享同一 session evidence</li></ul></div>
              <div class="panel"><h3>风险来源</h3><ul><li>文件后来被修改</li><li>新测试改变模型理解</li><li>模块位置发生变化</li><li>直接依赖 target 的内容更敏感</li></ul></div>
            </div>
            <div class="callout">项目不是“所有相同代码都复制”，而是寻找同一 session 中仍合法、成本为正且模型状态足够稳定的区域。</div>
            """,
        )
    )
    slides.append(
        _render_slide(
            6,
            "GOALS · FAIRNESS · NON-GOALS",
            "成功必须同时满足机械、准确率、公平性和系统收益",
            """
            <div class="three">
              <div class="panel ok"><h3>目标</h3><ul><li>复用 prefix 之外的 middle KV</li><li>降低 TTFT</li><li>保持官方功能测试</li><li>证明 coding signal 优于 controls</li></ul></div>
              <div class="panel selected"><h3>公平性</h3><ul><li>相同 target prompt</li><li>相同 token IDs</li><li>相同 source pool</li><li>相同 eligible set / budget</li><li>固定分母与 bootstrap</li></ul></div>
              <div class="panel warn"><h3>本分支不做</h3><ul><li>并发调度技巧</li><li>eviction / residency scheduling</li><li>HiCache 预取策略</li><li>用模型自评代替 tests</li></ul></div>
            </div>
            <div class="callout bad">如果 speedup 来自 batching、请求顺序或预取，而不是更多安全 KV 复用，就不能归功于 coding-aware policy。</div>
            """,
        )
    )
    slides.append(
        _render_slide(
            7,
            "COLLABORATION · LEGACY PROBLEM",
            "为什么必须先把你和合作者的工作从旧大分支中拆开",
            """
            <div class="three">
              <div class="panel bad"><h3>你这一侧</h3><p>AST、workflow、SessionGraph、风险标签、copy/dense 决策。</p></div>
              <div class="panel bad"><h3>合作者一侧</h3><p>scheduler、prefetch、residency、eviction、deadline 和 priority。</p></div>
              <div class="panel bad"><h3>旧共享实现</h3><p>identity、source pool、KV movement、RoPE、生命周期和实验自动开关。</p></div>
            </div>
            <div class="callout bad">旧结构把三类逻辑混入同一个 cache 路径，两个 owner 都可能修改 <code>radix_cache.py</code>，导致 merge 冲突和结果无法归因。</div>
            <div class="twocol">
              <div class="panel"><h3>具体风险</h3><ul><li>coding 变更影响 scheduler baseline</li><li>prefetch timing 被误报为 coding speedup</li><li>results JSON 意外启用 runtime</li><li>paper、launcher 和核心代码一起 merge</li></ul></div>
              <div class="panel"><h3>旧状态如何处理</h3><p>完整保存在只读 archive，不整支合回新结构；后续 unpublished 变更按 owner 逐个迁移。</p><code>archive/context-aware-kv-reuse-20260717<br>@ 015d58c969cb</code></div>
            </div>
            """,
        )
    )
    slides.append(
        _render_slide(
            8,
            "COLLABORATION · BRANCH MAP",
            "四个分支形成“两个研究 owner + 一个中立数据面 + 一个组合层”",
            """
            <div class="branch-grid">
              <div class="branch shared"><b>kvflow/shared-core</b><span>中立 KVCOMM：identity、generation、lease、validated transfer</span></div>
              <div class="branch coding"><b>research/coding-aware-lossy</b><span>你这一侧：决定哪些 token Dense，哪些 KV 可以 copy</span></div>
              <div class="branch prefetch"><b>research/prefetch</b><span>合作者一侧：决定何时、按何种优先级把 KV 搬到 device</span></div>
              <div class="branch integration"><b>integration/coding-aware-prefetch</b><span>只合并两边做 composition tests 和 thin adapters</span></div>
            </div>
            <div class="callout ok">研究逻辑留在各自 owner 分支；共享 bug 进入 shared-core；组合问题进入 integration，避免互相污染 baseline。</div>
            """,
        )
    )
    slides.append(
        _render_slide(
            9,
            "COLLABORATION · OWNERSHIP",
            "双方分别回答不同问题，互不读取对方的研究信号",
            """
            <table><thead><tr><th>Owner</th><th>输入</th><th>输出</th><th>不得负责</th></tr></thead><tbody>
              <tr><td>你：coding-aware</td><td>online-visible AST / workflow / file version / probe</td><td><code>KVReusePlan</code></td><td>scheduler、residency、eviction、ensure_resident</td></tr>
              <tr><td>合作者：prefetch</td><td>segment key、deadline、priority、tier</td><td>device-resident <code>KVSegmentHandle</code></td><td>AST、coding risk、实验 labels</td></tr>
              <tr><td>shared-core</td><td>plan、handle、target token IDs</td><td>validated transfer + telemetry</td><td>任何 policy 或 result-dependent activation</td></tr>
              <tr><td>integration</td><td>两个 owner 的已通过 commits</td><td>四模式组合结果</td><td>长期 fork 的新算法逻辑</td></tr>
            </tbody></table>
            <div class="callout">Coding owner 回答 <b>what may be reused</b>；Prefetch owner 回答 <b>when and where it becomes resident</b>。</div>
            """,
        )
    )
    slides.append(
        _render_slide(
            10,
            "COLLABORATION · CONTRACT",
            "双方只通过两个稳定对象交接：Handle 与 Plan",
            """
            <div class="twocol">
              <div class="panel selected"><h3>合作者交付 Handle</h3><pre>KVSegmentKey
  content/token/model/dtype identity
KVPrefetchHint
  deadline + priority + target tier
PrefetchTicket.wait()
  → KVSegmentHandle(DEVICE)</pre></div>
              <div class="panel selected"><h3>你交付 Plan</h3><pre>DenseRange[]
  current request must recompute
TransferSpan[]
  source handle + offset + target
KVReusePlan
  complete non-overlapping coverage</pre></div>
            </div>
            <div class="callout">Coding policy 不需要知道 KV 如何到达 GPU；prefetch policy 不需要知道为什么某段代码被判定为稳定。</div>
            """,
        )
    )
    slides.append(
        _render_slide(
            11,
            "COLLABORATION · END-TO-END HANDOFF",
            "一个 middle-KV 段如何从 earlier turn 交给 later request",
            """
            <div class="flow">
              <div><b>1 · Earlier turn</b><span>计算 exact token/KV slice</span></div><i>→</i>
              <div><b>2 · Export</b><span>host handle；原 device slots 仍归原请求</span></div><i>→</i>
              <div><b>3 · Prefetch</b><span>deadline / priority → device handle</span></div><i>→</i>
              <div class="selected"><b>4 · Reuse</b><span>coding plan 把 handle 放入 TransferSpan</span></div>
            </div>
            <div class="flow">
              <div><b>5 · Validate</b><span>generation · residency · token slice · bounds</span></div><i>→</i>
              <div><b>6 · Move</b><span>K full RoPE delta；V byte copy</span></div><i>→</i>
              <div><b>7 · Telemetry</b><span>copied / rotated / recomputed / fallback</span></div><i>→</i>
              <div><b>8 · Release</b><span>ticket lease + drop lifecycle</span></div>
            </div>
            <div class="callout bad">任何 stale、mismatch、nonresident 或 partial-transfer 都 fail closed 到 Dense。</div>
            """,
        )
    )
    slides.append(
        _render_slide(
            12,
            "COLLABORATION · DAILY WORKFLOW",
            "日常开发、合并和验收如何避免相互阻塞",
            """
            <div class="twocol">
              <div class="panel"><h3>你：coding-aware worktree</h3><pre>merge kvflow/shared-core
core=1 · coding=1 · prefetch=0
freeze experiment
run coding-only tests / benchmarks
emit plan + evidence</pre></div>
              <div class="panel"><h3>合作者：prefetch worktree</h3><pre>merge kvflow/shared-core
core=1 · coding=0 · prefetch=1
connect loader / scheduler
run residency + lifecycle tests
emit resident handle + telemetry</pre></div>
            </div>
            <div class="flow">
              <div><b>Owner tests</b><span>每边先独立通过</span></div><i>→</i>
              <div><b>Small merge</b><span>不直接跨 research cherry-pick</span></div><i>→</i>
              <div class="selected"><b>Integration</b><span>OFF / coding / prefetch / both</span></div><i>→</i>
              <div><b>Attribution</b><span>分别报告 reuse 与 residency 收益</span></div>
            </div>
            <div class="callout">这套拆分保证任一 owner 可以独立推进；只有 shared contract 改动需要双方同步。</div>
            """,
        )
    )
    slides.append(
        slide(
            1,
            "WEEKLY AUDIT",
            "从历史 31.2% 到 V12：本周所有版本的可审计结论",
            """
            <div class="hero">
              <div class="eyebrow">RUNTIME VALIDITY · FROZEN GATES · BRANCH DECOUPLING</div>
              <h2>V9–V12 均未通过预注册门槛；没有端到端成功 headline</h2>
            </div>
            <div class="metrics">
              <div class="metric bad"><small>V11 formal P0</small><strong>FALSIFIED</strong><p>容量通过，workflow signal 失败</p></div>
              <div class="metric bad"><small>V12 calibration</small><strong>0 / 4,639</strong><p>无可行 head × threshold 配置</p></div>
              <div class="metric warn"><small>Runtime status</small><strong>INTERFACE</strong><p>server canary 仍待完成</p></div>
            </div>
            <div class="callout">旧周报保留为历史快照；本报告只把能绑定完整 verdict、gate 和 artifact 的数字作为当前结论。</div>
            """,
        )
    )
    slides.append(
        slide(
            2,
            "AUDIT TAXONOMY",
            "“虚假数据”需要拆成五类，不能混为一个问题",
            """
            <div class="three">
              <div class="panel bad"><h3>Runtime-invalid</h3><p>body offset、partial RoPE、zero-gap 或 prompt cascade 使结果不能代表所声称方法。</p></div>
              <div class="panel bad"><h3>Schema error</h3><p>模块类型、依赖边或统计分母错误；修正后数字显著改变。</p></div>
              <div class="panel warn"><h3>Stale partial</h3><p>阶段性 PASS 文件在完整 stage 结束后仍存在，但已被 final gate 取代。</p></div>
            </div>
            <div class="twocol">
              <div class="panel warn"><h3>Claim overreach</h3><p>把 calibration、teacher logits 或机械测试外推为 accuracy、coding-specific 或 TTFT。</p></div>
              <div class="panel selected"><h3>Valid falsification</h3><p>制品与执行有效，但没有达到冻结门槛；这是真实负结果。</p></div>
            </div>
            <div class="callout">没有发现需要声称“人为伪造原始观测”的证据；主要问题是机械有效性、schema、证据版本和 claim 边界。</div>
            """,
        )
    )
    slides.append(
        slide(
            3,
            "EVIDENCE ORDER",
            "同名报告冲突时，机器可读 gate 高于演示文稿",
            """
            <div class="flow">
              <div class="selected"><b>1 · Frozen inputs</b><span>registration · design · split · SHA</span></div><i>→</i>
              <div><b>2 · Aggregate</b><span>完整 denominator · gate JSON</span></div><i>→</i>
              <div><b>3 · Final verdict</b><span>stage 结束后的判决</span></div><i>→</i>
              <div><b>4 · Narrative</b><span>partial · handoff · weekly deck</span></div>
            </div>
            <div class="twocol">
              <div class="panel bad"><h3>V11 陈旧文件</h3><p>旧 checkout 的 <code>FINAL_VERDICT.md</code> 仍写 P0 PARTIAL/PASS。</p></div>
              <div class="panel ok"><h3>V11 权威文件</h3><p>外部 4,960-row <code>P0_FINAL_VERDICT.md</code>：FALSIFIED。</p></div>
            </div>
            <div class="callout">每个 headline 必须同时报告 stage、denominator、gate、verdict 和 artifact。</div>
            """,
        )
    )
    slides.append(
        slide(
            4,
            "TASK CONSTRUCTION",
            "TaskFix V5–V7 与 Oracle-localized：先证明 Dense 任务有效",
            """
            <table><thead><tr><th>版本</th><th>目标</th><th>观测</th><th>判决</th></tr></thead><tbody>
              <tr><td>TaskFix V5</td><td>test-backed repair</td><td>55% balanced accuracy；1/10 pair-correct</td><td class="bad">P0 FALSIFIED</td></tr>
              <tr><td>TaskFix V6</td><td>gold-minus-one + 受限 patch</td><td>R2 仅 2/20 regression-safe</td><td class="bad">Dense gate FAIL</td></tr>
              <tr><td>TaskFix V7</td><td>单表达式 replacement</td><td>最多可行 23；要求 24</td><td class="bad">Construction FAIL</td></tr>
              <tr><td>Oracle-localized</td><td>直接提供局部上下文</td><td>Dense 0/13</td><td class="bad">Anchor FAIL</td></tr>
            </tbody></table>
            <div class="callout">这些版本没有产生 KV policy 速度结论；它们排除了“任务本身 Dense 都做不好，却继续比较有损策略”的无效路线。</div>
            """,
        )
    )
    slides.append(
        slide(
            5,
            "TASKCONE",
            "TaskCone 的速度观察真实存在，但 coding-specific gate 没过",
            """
            <div class="metrics">
              <div class="metric ok"><small>L2 follow-up preservation</small><strong>30 / 30</strong><p>HumanEval calibration official tests</p></div>
              <div class="metric ok"><small>Paired median TTFT</small><strong>+82.94%</strong><p>95% CI [82.70, 83.15]%</p></div>
              <div class="metric bad"><small>Control CI low</small><strong>0</strong><p>Uniform 与 Shuffled 均未过 gate</p></div>
            </div>
            <div class="twocol">
              <div class="panel bad"><h3>V1</h3><p>错误 body offset、head-only RoPE、zero-gap；全部 policy 证据撤回。</p></div>
              <div class="panel warn"><h3>L2 follow-up</h3><p>允许报告 calibration 内功能保留和速度；不允许报告 unseen 或 coding-specific。</p></div>
            </div>
            <div class="callout">MBPP 未打开；82.94% 不能被包装成一般 coding-agent 加速。</div>
            """,
        )
    )
    slides.append(
        slide(
            6,
            "ASTSPANKV",
            "真实 AST token spans：准确率和执行碎片化同时失败",
            """
            <div class="metrics">
              <div class="metric ok"><small>H0 Dense</small><strong>134 / 164</strong><p>official HumanEval pass@1</p></div>
              <div class="metric bad"><small>Dense→AST regression</small><strong>1 / 32</strong><p>零退化门槛失败</p></div>
              <div class="metric bad"><small>TTFT improvement</small><strong>-74.29%</strong><p>诊断 B0；明显慢于 Dense</p></div>
            </div>
            <div class="flow">
              <div><b>AST parse</b><span>真实 node spans</span></div><i>→</i>
              <div><b>Critical Dense</b><span>if/for/return/raise…</span></div><i>→</i>
              <div><b>Stable copy</b><span>同预算 matched controls</span></div><i>→</i>
              <div class="selected"><b>66.5 stages</b><span>固定开销超过 prefill 节省</span></div>
            </div>
            <div class="callout bad">P1 FAIL；P2 未打开。不能外推成“所有 AST signal 都无效”。</div>
            """,
        )
    )
    slides.append(
        slide(
            7,
            "AST-ISLAND",
            "限制 copy islands 恢复功能，但仍没有速度 Pareto",
            """
            <div class="metrics">
              <div class="metric ok"><small>B2/B4/B8/B16</small><strong>8 / 8</strong><p>四个配置均保留功能</p></div>
              <div class="metric bad"><small>最快配置 B8</small><strong>-5.04%</strong><p>paired-median TTFT improvement</p></div>
              <div class="metric bad"><small>Stage</small><strong>S0 FAIL</strong><p>controls / P1 / P2 未打开</p></div>
            </div>
            <div class="callout">减少 fragmentation 能消除 ASTSpan 的大部分性能灾难，但当前 prefix-staged executor 上仍慢于 Dense。</div>
            <div class="flow">
              <div><b>ASTSpan</b><span>大量细碎 stable runs</span></div><i>→</i>
              <div><b>Merge</b><span>最多 B 个 islands</span></div><i>→</i>
              <div class="selected"><b>AST-Island</b><span>功能恢复，速度仍负</span></div>
            </div>
            """,
        )
    )
    slides.append(
        slide(
            8,
            "V9–V12",
            "从 workflow 容量到动态状态 probe：每一版解决不同瓶颈",
            """
            <table><thead><tr><th>版本</th><th>新增信号</th><th>容量/设计</th><th>失败点</th></tr></thead><tbody>
              <tr><td>V9 WorkflowModule</td><td>稳定非代码模块</td><td>0.33%</td><td class="bad">真实容量不足</td></tr>
              <tr><td>V10 SessionGraph</td><td>同 session + graph + workspace</td><td>9.12% / 9.59%</td><td class="bad">容量门槛失败</td></tr>
              <tr><td>V11 FileVersion</td><td>event→file provenance</td><td>21.43%</td><td class="bad">静态 signal gate 失败</td></tr>
              <tr><td>V12 ProbeHead</td><td>实测 head K/V deviation</td><td>4,639 configs</td><td class="bad">0 feasible</td></tr>
            </tbody></table>
            <div class="callout">V12 不扩展 V11 候选，只在合法候选上动态拒绝高风险模块并重算固定 head。</div>
            """,
        )
    )
    slides.append(
        slide(
            9,
            "V10 SCHEMA AUDIT",
            "约 32.7% 的早期容量为何无效",
            """
            <div class="twocol">
              <div class="panel bad"><h3>错误 normalizer</h3><ul><li>把后续 user observation 当 immutable issue</li><li>漏掉 current-observation dependency</li><li>edit/apply-patch 被当普通 agent message</li><li>workspace guard 被绕过</li></ul></div>
              <div class="panel ok"><h3>固定 64-session 修正</h3><ul><li>non-prefix：9.12%</li><li>cost-positive：9.59%</li><li>later-turn coverage：100%</li><li>token mismatch：0</li></ul></div>
            </div>
            <div class="metrics compact">
              <div class="metric bad"><small>Superseded</small><strong>32.66%</strong><p>不得引用</p></div>
              <div class="metric bad"><small>Corrected vs 20%</small><strong>9.12%</strong><p>R0 FAIL</p></div>
              <div class="metric bad"><small>Corrected vs 15%</small><strong>9.59%</strong><p>C0 FAIL</p></div>
            </div>
            """,
        )
    )
    slides.append(
        slide(
            10,
            "V11 FILEVERSION",
            "FileVersion 恢复合法容量，但 workflow 特征不预测伤害",
            """
            <div class="metrics">
              <div class="metric ok"><small>Reusable / cost-positive</small><strong>21.43%</strong><p>64 sessions · 192 requests</p></div>
              <div class="metric bad"><small>Delta-R²</small><strong>0.02467</strong><p>95% CI high 0.04697 &lt; 0.05 gate</p></div>
              <div class="metric bad"><small>Safe harm reduction</small><strong>-119.711</strong><p>CI low -211.419</p></div>
            </div>
            <div class="flow">
              <div><b>Exact module</b><span>同 session、token-identical</span></div><i>→</i>
              <div><b>File provenance</b><span>对应文件之后未写入</span></div><i>→</i>
              <div><b>Capacity PASS</b><span>206,378 stable tokens</span></div><i>→</i>
              <div class="selected"><b>Signal FAIL</b><span>4,960-row complete atlas</span></div>
            </div>
            <div class="callout bad">机械与负对照通过不等于 P0 通过；P1 accuracy/TTFT 保持关闭。</div>
            """,
        )
    )
    slides.append(
        slide(
            11,
            "V12 ALGORITHM",
            "ProbeHead：先在当前上下文重算 head，再决定是否复制 body",
            """
            <div class="flow">
              <div><b>V11 candidate</b><span>合法、token-identical、cost-positive</span></div><i>→</i>
              <div><b>Dense head</b><span>H ∈ {8,16,32,64}</span></div><i>→</i>
              <div><b>K/V compare</b><span>K 做 RoPE shift；V 直接比较</span></div><i>→</i>
              <div class="selected"><b>Copy / Dense</b><span>score ≤ frozen threshold</span></div>
            </div>
            <div class="twocol">
              <div class="panel"><h3>Probe score</h3><pre>dK = 1 - mean cosine(RoPE(Ks), Kt)
dV = 1 - mean cosine(Vs, Vt)
score = max(dK, dV)</pre></div>
              <div class="panel"><h3>Online guards</h3><ul><li>body 必须有正净收益</li><li>最多 4 copy islands</li><li>mismatch/stale/nonresident → Dense</li><li>head 消耗完整模块 → Dense</li></ul></div>
            </div>
            """,
        )
    )
    slides.append(
        slide(
            12,
            "V12 CALIBRATION",
            "绝对误差低，但容量与风险排序无法同时过门槛",
            """
            <div class="metrics">
              <div class="metric ok"><small>Development coverage</small><strong>4,784</strong><p>96 requests · 1,196 modules</p></div>
              <div class="metric bad"><small>Configurations</small><strong>0 / 4,639</strong><p>feasible / evaluated</p></div>
              <div class="metric warn"><small>Holdout read</small><strong>FALSE</strong><p>sequential / P1 未运行</p></div>
            </div>
            <div class="twocol">
              <div class="panel warn"><h3>容量优先的最近点</h3><p>head=16：capacity 约 19.12%，harm reduction 约 24.14%，低于 30%。</p></div>
              <div class="panel warn"><h3>安全优先的上限</h3><p>满足 harm reduction ≥30% 时，最大 capacity 约 7.53%，低于 15%。</p></div>
            </div>
            <div class="callout bad">失败是 probe specificity：score 无法足够好地把高伤害模块排到阈值之外。</div>
            """,
        )
    )
    slides.append(
        slide(
            13,
            "ROOT CAUSES",
            "本周结果揭示四个彼此独立的门槛",
            """
            <div class="flow">
              <div class="selected"><b>1 · Mechanics</b><span>token slice、RoPE、coverage</span></div><i>→</i>
              <div><b>2 · Capacity</b><span>是否有足够合法 middle KV</span></div><i>→</i>
              <div><b>3 · Signal</b><span>能否识别低伤害区域</span></div><i>→</i>
              <div><b>4 · System</b><span>是否转化为 TTFT</span></div>
            </div>
            <table><thead><tr><th>失败层</th><th>代表版本</th><th>含义</th></tr></thead><tbody>
              <tr><td>Mechanics</td><td>历史 FRAC / TaskCone V1</td><td>结果撤回，不是算法证据</td></tr>
              <tr><td>Capacity</td><td>V9 / V10</td><td>运行 policy 前停止</td></tr>
              <tr><td>Signal</td><td>V11 / V12</td><td>合法候选存在，但排序不足</td></tr>
              <tr><td>Execution shape</td><td>ASTSpan / AST-Island</td><td>copied tokens 不自动产生加速</td></tr>
            </tbody></table>
            """,
        )
    )
    slides.append(
        slide(
            14,
            "CLAIM BOUNDARY",
            "现在能说什么，不能说什么",
            """
            <div class="twocol">
              <div class="panel ok"><h3>允许</h3><ul><li>旧 31.2% 未在修复后复现</li><li>V11 合法容量达到 21.43%</li><li>V11/V12 signal gates 均失败</li><li>KVCOMM 已完成接口级解耦</li><li>holdout 与 P1 仍关闭</li></ul></div>
              <div class="panel bad"><h3>禁止</h3><ul><li>当前方法已加速 coding agent</li><li>V12 精度没有损失</li><li>TaskCone 证明 coding-specific</li><li>V11 P0 PASS</li><li>V10 合法容量约 33%</li><li>生产级异步 prefetch 已完成</li></ul></div>
            </div>
            <div class="callout">teacher top-1、JS 和机械 identity 都不能替代官方 workflow tests。</div>
            """,
        )
    )
    slides.append(
        slide(
            15,
            "LEGACY COLLABORATION",
            "旧实现把 what、when、where 和 movement 混在同一 cache 路径",
            """
            <div class="three">
              <div class="panel bad"><h3>Coding policy</h3><p>AST selector、workflow labels、context confidence。</p></div>
              <div class="panel bad"><h3>Prefetch policy</h3><p>scheduler hooks、residency、eviction、priority。</p></div>
              <div class="panel bad"><h3>Data plane</h3><p>identity、source pool、KV movement、RoPE、lifecycle。</p></div>
            </div>
            <div class="callout bad">两个 owner 同时修改 <code>radix_cache.py</code> 时，无法判断收益来自 coding signal、预取时序还是隐藏的默认开关。</div>
            <div class="twocol">
              <div class="panel"><h3>Archive</h3><code>archive/context-aware-kv-reuse-20260717<br>@ 015d58c969cb</code></div>
              <div class="panel"><h3>Migration rule</h3><p>旧 collaborator branch 不整支 merge；后续变更以小 PR 进入对应 owner 分支。</p></div>
            </div>
            """,
        )
    )
    slides.append(
        slide(
            16,
            "BRANCH TOPOLOGY",
            "新结构：两个研究 owner 只通过 policy-neutral KVCOMM 组合",
            """
            <div class="branch-grid">
              <div class="branch shared"><b>kvflow/shared-core</b><span>identity · generation · lease · validated transfer</span></div>
              <div class="branch coding"><b>research/coding-aware-lossy</b><span>你这一侧：what may be copied / recomputed</span></div>
              <div class="branch prefetch"><b>research/prefetch</b><span>合作者一侧：when / where to load KV</span></div>
              <div class="branch integration"><b>integration/coding-aware-prefetch</b><span>composition tests · thin adapters only</span></div>
            </div>
            <div class="callout">两个 research branches 不直接 cherry-pick；共享 bug 先进入 shared-core，组合只进入 integration。</div>
            """,
        )
    )
    slides.append(
        slide(
            17,
            "SHARED CONTRACT",
            "Coding 输出 plan；Prefetch 输出 resident handle；Core 统一验证",
            """
            <div class="twocol">
              <div class="panel selected"><h3>Coding owner</h3><pre>online-visible signals
  → DenseRange[]
  → TransferSpan[]
  → KVReusePlan</pre></div>
              <div class="panel selected"><h3>Prefetch owner</h3><pre>KVPrefetchHint
  → PrefetchTicket.wait()
  → KVSegmentHandle(DEVICE)</pre></div>
            </div>
            <div class="flow">
              <div><b>Handle identity</b><span>model · dtype · token hash · generation</span></div><i>→</i>
              <div><b>TransferSpan</b><span>source offset · target · rope delta</span></div><i>→</i>
              <div class="selected"><b>KVComm execute</b><span>validate or fail closed</span></div>
            </div>
            """,
        )
    )
    slides.append(
        slide(
            18,
            "RESOURCE OWNERSHIP",
            "Middle-KV v1 的 export、prefetch、consume 和 release 生命周期",
            """
            <div class="flow">
              <div><b>Source request</b><span>computes exact KV slice</span></div><i>→</i>
              <div><b>export_middle_kv</b><span>host handle；原 slots 不被 free</span></div><i>→</i>
              <div><b>prefetch</b><span>deadline · priority · ticket</span></div><i>→</i>
              <div class="selected"><b>consume</b><span>device handle → reuse plan</span></div>
            </div>
            <div class="three">
              <div class="panel"><h3>Lease</h3><p>ticket 在 admission/finish 后 release，防止无限 pin。</p></div>
              <div class="panel"><h3>Drop</h3><p>副本不再 cacheable 时显式释放。</p></div>
              <div class="panel warn"><h3>当前限制</h3><p>ticket 同步；尚非 CUDA-event 异步实现。</p></div>
            </div>
            """,
        )
    )
    slides.append(
        slide(
            19,
            "OWNER WORKFLOW",
            "双方独立验证，再在 integration 做四模式组合",
            """
            <table><thead><tr><th>Owner</th><th>Feature flags</th><th>负责</th><th>禁止</th></tr></thead><tbody>
              <tr><td>Coding-aware</td><td>core=1 · coding=1 · prefetch=0</td><td>cohort、signal、plan、accuracy/TTFT</td><td>scheduler、eviction、ensure_resident</td></tr>
              <tr><td>Prefetch</td><td>core=1 · coding=0 · prefetch=1</td><td>export、deadline、residency、lease</td><td>AST、workflow label、coding result</td></tr>
              <tr><td>Integration</td><td>四模式矩阵</td><td>composition、thin adapter</td><td>新研究逻辑、paper、results</td></tr>
            </tbody></table>
            <div class="metrics compact">
              <div class="metric"><small>Mode 1</small><strong>OFF</strong><p>feature-off baseline</p></div>
              <div class="metric"><small>Mode 2 / 3</small><strong>ONE</strong><p>coding-only / prefetch-only</p></div>
              <div class="metric"><small>Mode 4</small><strong>BOTH</strong><p>composition without attribution loss</p></div>
            </div>
            """,
        )
    )
    slides.append(
        slide(
            20,
            "BRANCH SNAPSHOT",
            "当前 HEAD 与 7 月 18 日复跑结果",
            """
            <table><thead><tr><th>Branch</th><th>HEAD</th><th>Verification</th></tr></thead><tbody>
              <tr><td>kvflow/shared-core</td><td><code>c16bfbb8e</code> · rc3</td><td>interface base</td></tr>
              <tr><td>research/prefetch</td><td><code>fa86f8f16</code></td><td class="ok">30 passed · scope OK</td></tr>
              <tr><td>research/coding-aware-lossy</td><td><code>957468577</code> + V12 worktree</td><td class="ok">34 passed · scope OK</td></tr>
              <tr><td>integration/coding-aware-prefetch</td><td><code>d4a7ec132</code></td><td class="ok">1 composition passed · scope OK</td></tr>
            </tbody></table>
            <div class="callout warn">这些是 unit/reference contract 证据，不是 model-server TTFT、并发调度或生产 allocator 证据。</div>
            """,
        )
    )
    slides.append(
        slide(
            21,
            "SERVER GAPS",
            "为什么当前只能叫 INTERFACE_COMPLETE / SERVER_CANARY_PENDING",
            """
            <div class="three">
              <div class="panel warn"><h3>Shared core</h3><ul><li>真实 model-server GPU request</li><li>exact-transfer Dense identity</li><li>sustained lease/ref leak</li></ul></div>
              <div class="panel warn"><h3>Prefetch</h3><ul><li>production allocator</li><li>HiCache storage payload</li><li>scheduler prediction</li><li>CUDA async transfer</li></ul></div>
              <div class="panel warn"><h3>Integration</h3><ul><li>feature-off</li><li>coding-only</li><li>prefetch-only</li><li>combined server matrix</li></ul></div>
            </div>
            <div class="callout">V12 的 7B RTX 4090 reference executor canary 不是 SGLang model-server canary。</div>
            """,
        )
    )
    slides.append(
        slide(
            22,
            "NEXT",
            "下一步：coding signal 与 prefetch runtime 分别推进",
            """
            <div class="twocol">
              <div class="panel selected"><h3>Coding-aware V13（需新注册）</h3><ul><li>保持 holdout 密封</li><li>layer/head-wise tail features</li><li>module/body length 与 prefix delta</li><li>先做 ranking / top-risk capture</li><li>再做 sequential composition</li></ul></div>
              <div class="panel selected"><h3>Prefetch owner</h3><ul><li>接 scheduler admission</li><li>production host→device payload</li><li>CUDA event/stream ticket</li><li>deadline miss 与 dedup</li><li>并发 lifecycle audit</li></ul></div>
            </div>
            <div class="callout bad">不继续扩大 V12 threshold sweep，不放宽既有 gate，也不打开 holdout/P1。</div>
            """,
        )
    )
    slides.append(
        slide(
            23,
            "APPENDIX · ARTIFACTS",
            "完整 verdict 与机器可读 lock 的权威位置",
            """
            <div class="twocol">
              <div class="artifact"><h3>V11 complete</h3>
                <code>kvflow-artifacts/<br>impactkv_sessiongraph_v11_20260717/<br>P0_FINAL_VERDICT.md</code>
                <p class="source">SHA 628c42be…0cfefa</p>
              </div>
              <div class="artifact"><h3>V12 development</h3>
                <code>kvflow-artifacts/<br>impactkv_probehead_v12_20260717/<br>DEVELOPMENT_CALIBRATION_REPORT.json<br>CALIBRATION_LOCK.json</code>
                <p class="source">holdout_measurements_read=false</p>
              </div>
            </div>
            <div class="callout">详细审计正文与本 HTML/PDF 一起保存在独立归档目录。</div>
            """,
            appendix=True,
        )
    )
    slides.append(
        slide(
            24,
            "FINAL",
            "本周真正的成果：失败变得可归因，协作变得可组合",
            """
            <div class="hero"><h2>当前没有正 speedup headline，但已经建立可信的研究与协作边界</h2></div>
            <div class="flow">
              <div class="selected"><b>Mechanics</b><span>基本解决</span></div><i>→</i>
              <div class="selected"><b>Capacity</b><span>V11 证明存在</span></div><i>→</i>
              <div><b>Signal</b><span>V11/V12 当前失败</span></div><i>→</i>
              <div><b>TTFT + accuracy</b><span>尚未授权测量</span></div>
            </div>
            <div class="three">
              <div class="panel"><h3>Coding owner</h3><p>what to copy / recompute</p></div>
              <div class="panel"><h3>Prefetch owner</h3><p>when / where to load</p></div>
              <div class="panel"><h3>Shared + Integration</h3><p>safe transfer / composition</p></div>
            </div>
            <div class="callout ok">下一轮即使失败，也能区分 signal、capacity、movement、residency、scheduler 与 execution shape。</div>
            """,
        )
    )
    slides.append(
        slide(
            25,
            "REPOSITORY CLEANUP",
            "删的是重复入口与陈旧导航，不删失败证据和可复现链",
            """
            <div class="metrics">
              <div class="metric ok"><small>Project MD</small><strong>7 → 4</strong><p>保留总览、handoff、架构与综合审计</p></div>
              <div class="metric ok"><small>Obsolete scripts</small><strong>−2</strong><p>被完整 V11 aggregate 替代</p></div>
              <div class="metric warn"><small>Historical evidence</small><strong>KEPT</strong><p>失败结果与注册/测量链仍可复现</p></div>
            </div>
            <div class="twocol">
              <div class="panel"><h3>已删除脚本</h3><code>analyze_sessiongraph_v11_negative_controls.py<br>analyze_sessiongraph_v11_upstream.py</code><p>前者被正式 aggregate 覆盖；后者自称非正式 checkpoint。</p></div>
              <div class="panel"><h3>已合并文档</h3><code>_archive/handovers/README.md<br>docs/kvflow/HANDOFF.md<br>docs/kvflow/STATUS.md</code><p>当前状态收敛到 KVFLOW 与本审计，不再维护多份冲突真相。</p></div>
            </div>
            <div class="callout">上游 SGLang 自带文档、V11/V12 注册与测量脚本、raw artifacts 均未删除。</div>
            """,
        )
    )
    slides.append(
        slide(
            26,
            "SGLANG DIFF AUDIT",
            "相对最初 SGLang，原框架侵入很小，但新增层尚未进入请求执行",
            """
            <div class="metrics">
              <div class="metric"><small>Baseline</small><strong>3343a794</strong><p>origin/main initial comparison point</p></div>
              <div class="metric ok"><small>Shared-core diff</small><strong>18 files</strong><p>+1,923 lines；主要为新增 KVCOMM</p></div>
              <div class="metric ok"><small>Original SGLang edits</small><strong>11 lines</strong><p>cache params + RadixCache init/reset</p></div>
            </div>
            <div class="twocol">
              <div class="panel ok"><h3>结构优势</h3><ul><li>policy-neutral shared core</li><li>coding 与 prefetch owner 分离</li><li>default-off、fail-closed</li><li>完整 coverage / token / RoPE 检查</li></ul></div>
              <div class="panel warn"><h3>关键事实</h3><ul><li>coding commit 主要是 offline benchmark</li><li>生产 request path 没有 register_segment</li><li>生产 request path 没有 manager.execute</li><li>因此当前不是 server-integrated feature</li></ul></div>
            </div>
            """,
        )
    )
    slides.append(
        slide(
            27,
            "STRUCTURAL BLOCKERS",
            "最危险的问题不是代码多，而是生命周期与真实执行器脱节",
            """
            <table><thead><tr><th>Severity</th><th>问题</th><th>后果</th></tr></thead><tbody>
              <tr><td class="bad">P0</td><td><code>ensure_resident</code> 换 backend ref 却不 bump generation</td><td>旧 HOST handle 仍被判 current；并发 loader 可引用已释放 ref</td></tr>
              <tr><td class="bad">P0</td><td>KVSegmentStore 与 Radix/HiCache 双重记录生命周期</td><td>allocator slots 已释放，handle 仍可能表面有效</td></tr>
              <tr><td class="warn">P1</td><td>Dense fallback callback 未接 sparse/interleaved prefill</td><td>验证失败后的真实重算路径未证明</td></tr>
              <tr><td class="warn">P1</td><td>key identity 不含 revision/rank/layout/backend/RoPE/page config</td><td>不同执行配置可能错误命中</td></tr>
              <tr><td>P2</td><td>无 CUDA stream/event ownership；按 record 数量限容</td><td>异步正确性与显存/host 成本不可控</td></tr>
            </tbody></table>
            <div class="callout bad">最小复现已确认：迁移后 old_handle_is_current=true、old/new generation 相同，但 residency/backend_ref 已变化。</div>
            """,
        )
    )
    slides.append(
        slide(
            28,
            "TARGET ARCHITECTURE",
            "先把 exact contiguous reuse 接进 Scheduler / ModelRunner，再谈 lossy 扩张",
            """
            <div class="flow">
              <div><b>Scheduler / ModelRunner</b><span>请求与 allocator 的真实 owner</span></div><i>→</i>
              <div><b>Reuse Planner</b><span>先 exact；以后再接 coding policy</span></div><i>→</i>
              <div><b>Registry Adapter</b><span>锚定 Radix / HiCache 生命周期</span></div><i>→</i>
              <div><b>Plan Executor</b><span>copy islands + dense gaps</span></div>
            </div>
            <div class="three">
              <div class="panel selected"><h3>Phase A</h3><p>一个 contiguous exact island，真实 server request，默认关闭。</p></div>
              <div class="panel"><h3>Phase B</h3><p>修 generation、lease、eviction、stream/event 与 byte budget。</p></div>
              <div class="panel"><h3>Phase C</h3><p>接 prefetch residency 和 bounded islands；最后才开放 lossy plan。</p></div>
            </div>
            <div class="callout">只有 executor 与 lifecycle 可验证后，TTFT 才能归因于 KV reuse，而不是 reference harness 或隐含调度变化。</div>
            """,
        )
    )
    slides.append(
        slide(
            29,
            "LOSSY ROUTE AUDIT",
            "路线并非证明“不可能”，而是选择信号和评价指标尚未匹配系统目标",
            """
            <table><thead><tr><th>版本</th><th>留下的有效信息</th><th>失败点</th></tr></thead><tbody>
              <tr><td>TaskCone / AST</td><td>某些结构区域有稳定性差异</td><td>matched controls / fragmentation / net speed</td></tr>
              <tr><td>V9–V11</td><td>合法 FileVersion 容量可超过 20%</td><td>coding-specific signal、harm 与收益不能同过</td></tr>
              <tr><td>V12 ProbeHead</td><td>存在低误差 module/head 个例</td><td>4,639 个配置无共同可行点</td></tr>
            </tbody></table>
            <div class="twocol">
              <div class="panel bad"><h3>V12 metric mismatch</h3><ul><li>per-module mean，不按 token/request 加权</li><li>独立替换，不是 sequential composition</li><li>final-prompt logits 不是生成/任务准确率</li><li>head-only 不能可靠预测 body harm</li></ul></div>
              <div class="panel warn"><h3>系统经济性</h3><ul><li>稀碎 islands 吃掉 kernel/copy 收益</li><li>metadata/probe/host→device 都有成本</li><li>尾部高风险会支配请求失败</li><li>低平均 harm 不等于可上线</li></ul></div>
            </div>
            """,
        )
    )
    slides.append(
        slide(
            30,
            "DECISION · NEXT GATES",
            "暂停扩大 signal sweep；按系统可证伪顺序重启",
            """
            <div class="flow">
              <div class="selected"><b>1 · Exact executor</b><span>真实 server 单 island</span></div><i>→</i>
              <div><b>2 · Cost model</b><span>token/byte/copy/kernel/probe</span></div><i>→</i>
              <div><b>3 · Workload</b><span>真实 session file-version 分布</span></div><i>→</i>
              <div><b>4 · Lossy ranking</b><span>捕获 tail risk</span></div>
            </div>
            <div class="flow">
              <div><b>5 · Composition</b><span>1–2 个大岛 sequential</span></div><i>→</i>
              <div><b>6 · Objective</b><span>official tests / generation</span></div><i>→</i>
              <div><b>7 · System</b><span>TTFT、吞吐、显存、并发</span></div><i>→</i>
              <div class="selected"><b>8 · Integration</b><span>四模式归因</span></div>
            </div>
            <div class="callout bad">V13 必须新注册；不得放宽 V11/V12 门槛、打开冻结 holdout，或用更多 threshold sweep 替代 executor 证据。</div>
            <div class="callout ok"><b>当前最稳健结论：</b>mechanics 基本可行、容量存在；廉价选择信号、顺序安全性、真实精度与端到端收益均未证明。</div>
            """,
        )
    )

    css = r"""
    :root{--bg:#0b1525;--panel:#14233a;--line:#33445f;--text:#e8edf5;--dim:#aab5c5;--cyan:#56b4e9;--green:#009e73;--orange:#e69f00;--rust:#d55e00}
    *{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif;overflow:hidden}
    #deck{display:flex;width:100vw;height:100vh;overflow-x:auto;scroll-snap-type:x mandatory}
    .slide{flex:0 0 100vw;width:100vw;height:100vh;padding:27px 52px 40px;position:relative;overflow:hidden;scroll-snap-align:start;border-left:9px solid var(--cyan);display:flex;flex-direction:column}
    .meta{display:flex;justify-content:space-between;color:var(--cyan);font:12px ui-monospace;letter-spacing:.15em}
    .slide h1{font-size:29px;line-height:1.14;margin:9px 0 12px}.content{flex:1;min-height:0;font-size:16px;line-height:1.46}
    .content h2{font-size:27px;line-height:1.22;margin:7px 0 9px}.content h3{font-size:17px;color:var(--cyan);margin:0 0 7px}
    .content p,.content li{font-size:15px;line-height:1.45}.content ul{margin:5px 0;padding-left:22px}.content li{margin:4px 0}
    footer{position:absolute;bottom:12px;left:52px;right:52px;display:flex;justify-content:space-between;color:#758196;font:11px ui-monospace}
    .hero{border-top:2px solid var(--cyan);padding-top:10px}.eyebrow{font:12px ui-monospace;color:var(--cyan);letter-spacing:.13em}
    .metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:11px;margin:12px 0}.metrics.compact{margin:4px 0}
    .metrics.compact .metric{min-height:84px;padding:7px 11px}.metrics.compact .metric strong{font-size:21px}
    .metric{background:var(--panel);border:1px solid var(--line);border-top:3px solid var(--cyan);padding:10px 13px;min-height:99px}
    .metric.ok{border-top-color:var(--green)}.metric.bad{border-top-color:var(--rust)}.metric.warn{border-top-color:var(--orange)}
    .metric small{display:block;color:var(--dim);font:11px ui-monospace}.metric strong{display:block;font-size:26px;line-height:1.14;margin:4px 0}
    .metric p{font-size:13px;margin:0;color:var(--dim)}.callout{border-left:4px solid var(--cyan);background:var(--panel);padding:9px 14px;margin:9px 0;font-size:15px}
    .callout.bad{border-left-color:var(--rust)}.callout.ok{border-left-color:var(--green)}.callout.warn{border-left-color:var(--orange)}
    .flow{display:flex;align-items:stretch;gap:8px;margin:20px 0}.flow div{flex:1;background:var(--panel);border:1px solid var(--line);padding:12px}
    .flow div.selected{border-color:var(--cyan)}.flow b,.flow span{display:block}.flow span{color:var(--dim);font-size:13px;margin-top:5px}
    .flow i{align-self:center;color:var(--cyan);font-size:22px}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:13px;margin:8px 0}
    .three{display:grid;grid-template-columns:repeat(3,1fr);gap:11px;margin:10px 0}.panel{background:var(--panel);border:1px solid var(--line);padding:12px;min-height:110px}
    .panel.selected{border:2px solid var(--cyan)}.panel.bad{border-top:3px solid var(--rust)}.panel.ok{border-top:3px solid var(--green)}.panel.warn{border-top:3px solid var(--orange)}
    code{font:13px ui-monospace;color:var(--cyan)}pre{font:13px ui-monospace;line-height:1.42;color:var(--text);white-space:pre-wrap;margin:3px 0}
    table{width:100%;border-collapse:collapse;font-size:13px;margin:7px 0}th,td{border:1px solid var(--line);padding:6px 8px;text-align:left}
    th{color:var(--dim);background:var(--panel);font:11px ui-monospace}td.ok,.ok{color:#59d0a7}td.bad,.bad{color:#f07a50}.warn{color:#f0c75e}
    .source{color:var(--dim);font-size:12px!important}.artifact{background:var(--panel);border:1px solid var(--line);padding:16px}
    .branch-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}.branch{background:var(--panel);border:1px solid var(--line);padding:18px}
    .branch.shared{grid-column:1/3;border-top:3px solid var(--green)}.branch.coding{border-top:3px solid var(--cyan)}.branch.prefetch{border-top:3px solid var(--orange)}
    .branch.integration{grid-column:1/3;border-top:3px solid var(--cyan)}.branch b,.branch span{display:block}.branch span{color:var(--dim);margin-top:6px}
    @media print{html,body{overflow:visible;background:var(--bg)}#deck{display:block;width:100%;height:auto;overflow:visible}.slide{width:100%;height:100vh;page-break-after:always;break-after:page}.slide:last-child{page-break-after:auto}}
    """
    js = (
        "const s=[...document.querySelectorAll('.slide')];let i=0;"
        "function go(n){i=Math.max(0,Math.min(s.length-1,n));"
        "s[i].scrollIntoView({behavior:'instant'})}"
        "addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' ']"
        ".includes(e.key))go(i+1);if(['ArrowLeft','PageUp']"
        ".includes(e.key))go(i-1)});"
    )
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width">'
        "<title>ImpactKV / KVFlow Weekly Audit 2026-07-18</title>"
        f"<style>{css}</style></head><body><main id=\"deck\">"
        f'{"".join(slides)}</main><script>{js}</script></body></html>'
    )


def render_pdf(html_path: Path, pdf_path: Path, qa_path: Path) -> dict[str, object]:
    from playwright.sync_api import sync_playwright
    from pypdf import PdfReader

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        slide_count = page.locator(".slide").count()
        overflow = []
        for index in range(slide_count):
            sizes = page.locator(".slide").nth(index).evaluate(
                "el=>({sw:el.scrollWidth,cw:el.clientWidth,"
                "sh:el.scrollHeight,ch:el.clientHeight})"
            )
            if sizes["sw"] > sizes["cw"] + 1 or sizes["sh"] > sizes["ch"] + 1:
                overflow.append({"slide": index + 1, **sizes})
        page.emulate_media(media="print")
        page.pdf(
            path=str(pdf_path),
            width="12.8in",
            height="7.2in",
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )
        browser.close()
    pdf_pages = len(PdfReader(str(pdf_path)).pages)
    result = {
        "slides": slide_count,
        "pdf_pages": pdf_pages,
        "overflow": overflow,
        "passed": slide_count == pdf_pages and not overflow,
    }
    qa_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"visual validation failed: {result}")
    return result


def build_index(output: Path) -> str:
    cards = []
    for stem, title, note in HISTORICAL_REPORTS:
        cards.append(
            '<article class="card historical">'
            f"<h2>{html.escape(title)}</h2><code>{html.escape(stem)}</code>"
            f"<p>{html.escape(note)}</p>"
            f'<a href="{stem}.html">HTML</a><a href="{stem}.pdf">PDF</a>'
            "</article>"
        )
    cards.append(
        '<article class="card current"><h2>2026-07-18 综合审计与分支协作</h2>'
        f"<code>{CURRENT_STEM}</code>"
        "<p>V9–V12 完整判决、历史数据修正、双分支解耦与下一步。</p>"
        f'<a href="{CURRENT_STEM}.html">HTML</a>'
        f'<a href="{CURRENT_STEM}.pdf">PDF</a>'
        f'<a href="{CURRENT_STEM}.md">Markdown</a></article>'
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>ImpactKV / KVFlow 周报归档</title>
<style>
:root{{--bg:#0b1525;--panel:#14233a;--line:#33445f;--text:#e8edf5;--dim:#aab5c5;--cyan:#56b4e9;--green:#009e73;--rust:#d55e00}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:1180px;margin:auto;padding:42px 28px 70px}}h1{{font-size:36px;margin:0 0 10px}}.lead{{color:var(--dim);font-size:17px;line-height:1.6}}
.notice{{border-left:4px solid var(--rust);background:var(--panel);padding:14px 18px;margin:24px 0;line-height:1.55}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:16px;margin-top:28px}}
.card{{background:var(--panel);border:1px solid var(--line);border-top:4px solid var(--line);padding:20px;min-height:210px}}
.card.current{{border-top-color:var(--green)}}.card h2{{font-size:20px;margin:0 0 10px}}.card code{{font-size:12px;color:var(--cyan);overflow-wrap:anywhere}}
.card p{{color:var(--dim);line-height:1.5}}a{{display:inline-block;color:var(--cyan);border:1px solid var(--cyan);padding:7px 12px;margin:8px 8px 0 0;text-decoration:none}}
footer{{color:var(--dim);margin-top:30px;font:12px ui-monospace}}
</style></head><body><main>
<h1>ImpactKV / KVFlow 周报归档</h1>
<p class="lead">历史 HTML/PDF 保持字节不变；2026-07-18 报告以相同 16:9 深色 slide-deck 格式生成，同时保留详细 Markdown 正文。</p>
<div class="notice"><b>审计提示：</b>历史文件是研究过程快照，不自动代表当前有效结论。31.2%、旧 AST/chunker 和 V10 约 32.7% 等数字已在 2026-07-18 审计报告中撤回或修正。</div>
<section class="grid">{''.join(cards)}</section>
<footer>MANIFEST.sha256 · VISUAL_VALIDATION.json · README.md</footer>
</main></body></html>"""


def build_readme(
    output: Path, visual: dict[str, object], audit_line_count: int
) -> str:
    rows = [
        "# ImpactKV / KVFlow 周报独立归档",
        "",
        "生成日期：2026-07-18",
        "",
        "本目录从旧脏 checkout 只读复制历史 HTML/PDF；源文件保持字节不变。",
        "2026-07-18 综合审计报告由当前 coding-aware worktree 生成。",
        "",
        "## 查阅入口",
        "",
        "- `INDEX.html`：浏览器索引；",
        f"- `{CURRENT_STEM}.html`：今日 16:9 slide deck；",
        f"- `{CURRENT_STEM}.pdf`：今日 PDF；",
        f"- `{CURRENT_STEM}.md`：{audit_line_count} 行详细正文；",
        "- `MANIFEST.sha256`：归档文件校验值；",
        "- `VISUAL_VALIDATION.json`：今日 HTML/PDF 分页与 overflow 检查。",
        "",
        "## 今日报告视觉验证",
        "",
        f"- HTML slides：{visual['slides']}",
        f"- PDF pages：{visual['pdf_pages']}",
        f"- overflow：{len(visual['overflow'])}",
        f"- passed：{str(visual['passed']).lower()}",
        "",
        "## 解释规则",
        "",
        "历史报告保留研究时间线，但其中被 runtime/schema 审计推翻的数字不得作为",
        "当前结论。以 2026-07-18 综合审计及其引用的机器可读 final gates 为准。",
        "",
    ]
    return "\n".join(rows)


def write_manifest(output: Path) -> None:
    files = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "MANIFEST.sha256"
    )
    lines = [f"{sha256(path)}  {path.name}" for path in files]
    (output / "MANIFEST.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    for stem, _, _ in HISTORICAL_REPORTS:
        for suffix in (".html", ".pdf"):
            source = args.source / f"{stem}{suffix}"
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, output / source.name)

    if not args.audit.is_file():
        raise FileNotFoundError(args.audit)
    audit_copy = output / f"{CURRENT_STEM}.md"
    shutil.copy2(args.audit, audit_copy)
    audit_line_count = sum(
        1 for _ in audit_copy.open("r", encoding="utf-8")
    )

    html_path = output / f"{CURRENT_STEM}.html"
    pdf_path = output / f"{CURRENT_STEM}.pdf"
    qa_path = output / "VISUAL_VALIDATION.json"
    html_path.write_text(build_deck(), encoding="utf-8")
    visual = render_pdf(html_path, pdf_path, qa_path)

    (output / "INDEX.html").write_text(build_index(output), encoding="utf-8")
    (output / "README.md").write_text(
        build_readme(output, visual, audit_line_count), encoding="utf-8"
    )
    write_manifest(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "historical_pairs": len(HISTORICAL_REPORTS),
                "current_html": str(html_path),
                "current_pdf": str(pdf_path),
                "visual": visual,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
