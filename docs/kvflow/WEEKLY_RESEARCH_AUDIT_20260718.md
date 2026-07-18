# ImpactKV / KVFlow 本周研究审计与版本演进报告

日期：2026-07-18  
覆盖范围：2026-07-14 至 2026-07-18；必要时回溯 7 月 10–13 日结果  
报告性质：对 2026-07-16 ASTSpanKV 周报的审计式续篇  
当前结论：coding-aware V9–V12 均未通过各自预注册门槛；KVCOMM 分支接口已解耦，但生产 model-server 路径仍待 canary

## 0. 报告说明

本报告以此前的
`results/weekly_reports/2026-07-16_CODE_AWARE_LOSSY_KV_PROGRESS_ASTSPANKV.html`
为叙事底稿，但不直接修改该文件。旧报告位于历史脏 checkout，本报告在
`research/coding-aware-lossy` 的独立 worktree 中新建，避免改写历史证据、
paper 或已经注册的实验门槛。

旧报告 SHA-256：

```text
7555cac1a533bb25aa24a4eb3379eb4d057813e2788e848e137b31548bfa98f9
```

用户指出旧报告中存在大量“虚假数据”。从本次文件与实验制品审计看，
更精确的分类是：

1. **运行时无效（runtime-invalid）**：实验确实运行过，但 KV body offset、
   RoPE 或 zero-gap 实现错误，结果不能代表所声称的方法。
2. **数据/schema 错误**：统计来自错误的事件归类或错误的分母，修正后数字
   显著变化。
3. **过期的阶段性结论**：partial report 在完整实验结束后没有被原位更新，
   文件名仍像“最终结果”，但权威 verdict 已改变。
4. **外推过度**：局部 calibration、teacher-logit 或机械测试被误读为
   workflow accuracy、coding-specific 优势或端到端加速。
5. **真实的预注册失败**：制品有效，但没有达到预先冻结的容量、精度、统计
   或速度门槛。

本次没有发现需要声称“人为捏造原始观测”的证据；问题主要是实验机械错误、
schema 错误、证据版本管理和 claim 边界失控。下面统一使用
“无效、撤回、被取代、未建立或 falsified”描述。

## 1. 执行摘要

### 1.1 本周真正建立的事实

- 历史 Uniform head 30% 的 **31.2% TTFT 改善已撤回**。修复 KV copy
  路径、冻结 prompt 并重跑后，只得到约 **1.86%（报告中四舍五入为
  1.9%）** 的 paired-median TTFT 改善。
- TaskCone L2 follow-up 在固定 HumanEval calibration anchors 上得到
  **30/30 功能保留**和 **82.94%** paired-median TTFT 改善，但相对
  Uniform 与 Shuffled 的 preservation CI 下界均为 0，因此没有建立
  coding-specific 定位优势，也没有打开 MBPP unseen。
- ASTSpanKV 在正式 calibration 中产生 **1/32 Dense→wrong regression**，
  且诊断 TTFT 改善为 **-74.29%**，即显著慢于 Dense。
- AST-IslandKV 将功能保留改善到 **8/8**，但最快 B8 仍为
  **-5.04%**，没有速度 Pareto。
- WorkflowModuleKV V9 的真实 prompt 稳定非代码容量中位数只有
  **0.33%**，在运行 policy 前被容量门槛证伪。
- SessionGraphKV V10 修正 schema 后，合法 non-prefix 容量只有
  **9.12%**，cost-positive 容量 **9.59%**；此前约 32.7% 的结果无效。
- FileVersion SessionGraphKV V11 把合法容量提高到 **21.43%**，但完整
  4,960-row P0 的 workflow-feature delta-\(R^2\) 和 safe-vs-unsafe
  gate 都失败，最终 verdict 为 **FALSIFIED**。
- ProbeHead StateSensitivityKV V12 完成 4,784 条 development workflow
  观测、评估 4,639 个配置，**可行配置为 0**。holdout 未读取，sequential
  development composition、P1 accuracy 和 TTFT 均未运行。

### 1.2 当前不能声称的内容

- 不能声称当前 coding-aware 方法带来端到端 SGLang TTFT 加速。
- 不能声称 V11 或 V12 保留真实 coding workflow accuracy。
- 不能用 V12 的 teacher top-1 不变代替官方功能测试。
- 不能声称 TaskCone 已证明 coding-specific 优势或 unseen generalization。
- 不能声称 KVCOMM 已完成 production model-server GPU 验证。
- 不能把 prefetch 接口测试解释为并发调度或 HiCache 预取已经完成。

### 1.3 本周总判决

研究结论不是“lossy middle-KV 完全不可行”，而是：

> 机械上可以安全搬运 token-identical 的 middle KV，也存在足够大的
> file-version 候选容量；当前失败发生在“如何用低成本信号准确识别真正
> 低伤害区域”以及“如何让碎片化执行转化为端到端收益”这两层。

## 2. 证据优先级与防止陈旧结论的规则

同一版本出现多个互相冲突的文件时，本报告使用以下优先级：

1. 冻结 registration、design、split、artifact manifest 和输入 SHA；
2. 完整 stage aggregate 及其机器可读 gate JSON；
3. 完整 stage 结束后生成的 final verdict；
4. partial report、status、handoff；
5. weekly HTML/PDF、演示图表和口头摘要；
6. 被移动到 `invalid_runs/` 或明确列入 forbidden inputs 的历史产物。

因此：

- 旧 checkout 中 V11 的 `FINAL_VERDICT.md` 写着
  “P0 PARTIAL — MECHANICS AND DIRECTIONAL SIGNAL PASS”，它只是 1,920-row
  阶段快照。
- 权威结果是外部只读制品中的
  `impactkv_sessiongraph_v11_20260717/P0_FINAL_VERDICT.md`：完整 4,960 rows，
  verdict 为 **FALSIFIED**。
- coding-aware 已删除的旧 `docs/kvflow/STATUS.md` 曾写
  “V12 DEVELOPMENT_PARTIAL”；该状态已被 7 月 18 日完成的开发校准取代，
  当前摘要合并到根目录 `KVFLOW.md`。

以后每个 headline 必须同时给出：

- stage；
- denominator；
- gate；
- verdict；
- artifact 路径；
- 如果是派生分析，明确标记“非注册 headline”。

## 3. 被撤回、修正或限制的旧主张

| 旧主张或旧数字 | 问题类型 | 审计后的结论 |
|---|---|---|
| Uniform head 30% 改善 31.2% | runtime-invalid | body source offset、partial RoPE、zero-gap 和 prompt cascade 被审计出问题；修复后约 1.86% |
| 7 月 10–13 日 AST/chunker 表证明 AST-aware 有效或无效 | runtime-invalid + launcher bug | 旧 interface/control-flow 只是 proxy，chunker launcher 还包含字面量 `\\n`；不能作为正式 AST 结论 |
| HumanEval TaskCone V1 preservation/speed | runtime-invalid | V1 使用错误 body offset、head-only RoPE 和 zero KV gap；不得引用 |
| TaskCone L2 的 82.94% 等于 coding-specific 加速 | 外推过度 | calibration 内功能和速度观察可保留；matched-control CI gate 失败，MBPP 未打开 |
| V10 non-prefix 容量约 32.66%，cost-positive 约 32.94% | schema 错误 | 后续 `user` observation 被错误当作 immutable issue；修正后为 9.12% / 9.59% |
| V11 “P0 PASS” | 过期 partial | 仅机械、负对照和部分方向性通过；完整 P0 的两个 coding-specific gate 失败 |
| V12 单模块 JS 很低，因此 accuracy 已保留 | 指标外推 | JS/top-1 是 teacher-logit P0 指标，不是 workflow functional accuracy |
| middle-KV prefetch 已完成 | 接口与生产状态混淆 | CPU/fake allocator 的 API 已验证；真实 scheduler、异步 CUDA transfer 和 HiCache storage payload 尚未验证 |

## 4. 本周版本时间线

### 4.1 数据集和任务构造路线

| 版本 | 目标 | 关键观测 | 判决 |
|---|---|---|---|
| TaskFix V5 | 在 test-backed repair 上建立 Dense 任务有效性 | balanced accuracy 55%，仅 1/10 pair-correct | P0 FALSIFIED |
| TaskFix V6 | 用 gold-minus-one、受限 patch 格式减少输出失败 | R2 仅 2/20 regression-safe passes | Dense discovery gate 失败 |
| TaskFix V7 | 单表达式 byte-addressed replacement | 最多可行 23，门槛要求 24 | task construction FALSIFIED |
| Oracle-localized | 直接给局部编辑上下文 | Dense 0/13 | anchor gate 失败 |
| HumanEval TaskCone V1 | 目标函数感知的 KV 分配 | 后续发现 runtime invalid | 全部 policy 证据撤回 |
| HumanEval TaskCone V2 | 修复 KV transfer 并使用官方测试 | 无 profile 达到零 regression；P80 慢 10.49% | mechanism not validated |
| TaskCone L2 | shape-aligned 目标函数路线 | 30/30 tests，82.67%，但 2 个 completion SHA 不同 | strict prereg gate 失败 |
| TaskCone L2 follow-up | 把功能测试与 SHA identity 分开 | 30/30，82.94%；control CI low=0 | 整体 gate 失败，MBPP 关闭 |

这些版本主要解决“有没有一个 Dense 能做、测试可判、输出能应用的客观 coding
任务”。它们不是 V9–V12 的直接祖先，但解释了为什么项目后来转向真实
multi-turn workflow module 和 offline causal atlas。

### 4.2 AST 定位路线

#### ASTSpanKV

方法：

- 解析真实 AST span；
- control-flow、return、raise 等 token run 强制 Dense；
- stable run 复制；
- Uniform/Shuffled 使用相同 eligible set 和整数预算。

结果：

- H0 Dense：134/164 official HumanEval pass@1；
- calibration：Dense/Uniform/ASTSpan/Shuffled =
  32/30/31/30；
- Dense→ASTSpan regression：1/32；
- 诊断 paired-median TTFT improvement：-74.29%；
- 中位每请求 dense/copy stages：66.5；
- P2 未打开。

解释：

> 真 AST 位置没有保住零 regression；细粒度交错又产生大量 stage，固定
> overhead 超过省下的 prefill。

#### AST-IslandKV

方法：

- 不再复制大量细碎 stable span；
- 把可复制区域压缩成最多 B 个稳定 island；
- 冻结 B2/B4/B8/B16。

结果：

- 四个配置均 8/8 功能保留；
- 最快 B8 仍为 -5.04%；
- S0 失败，controls/P1/P2 未打开。

解释：

> 减少 fragmentation 能恢复准确率和大部分性能损失，但当前 prefix-staged
> executor 上仍没有正 speedup。

### 4.3 Workflow / SessionGraph 路线

| 版本 | 新增信号 | 合法/有效容量 | 核心 gate | 最终结果 |
|---|---|---:|---|---|
| V9 WorkflowModuleKV | task/system、agent、tool、workspace 模块 | 0.33% 稳定非代码中位容量 | ≥20% | R0 FALSIFIED |
| V10 SessionGraphKV | 同一 session、graph distance、workspace version | 9.12%；cost-positive 9.59% | ≥20% / ≥15% | R0/C0 FALSIFIED |
| V11 FileVersion SessionGraphKV | canonical event→file provenance；未被写入的旧 source view 可延续 | 21.43% / 21.43% | P0 signal gates | P0 FALSIFIED |
| V12 ProbeHead StateSensitivityKV | 在 V11 候选上实测 head K/V deviation | 配置相关 | ≥15% 容量且 ≥30% harm reduction | development calibration FALSIFIED |

#### V9：模块定义正确，但真实容量不足

V9 试图复用稳定的非代码 workflow 模块，例如 instruction、agent message、
tool output 和 workspace trace。真实 prompt 审计发现这些模块通常是
turn-local、已经变化或只是普通 prefix，真正合法的 middle stable capacity
中位数只有 0.33%。因此没有运行 GPU policy，也没有 accuracy/speed 结论。

#### V10：引入 session graph，但 schema 修正推翻早期容量

V10 把真实 session 拆成模块和依赖边：

- target 为 graph distance 0；
- target 的直接依赖为 distance 1；
- 只允许更远、token-identical、早先出现且 workspace version 合法的模块；
- 当前 observation、target、turn-local 和 stale workspace 模块保持 Dense。

初始 normalizer 把 SWE-agent 后续 `user` observation 错当成 immutable issue，
导致模块类型、依赖边和 workspace scope 错误，容量虚高到约 32.7%。
保持 64-session 固定分母修正后：

- non-prefix capacity：9.12%；
- cost-positive capacity：9.59%；
- later turns reuse coverage：100%；
- token-slice mismatch：0。

容量门槛失败，P0 causal atlas 没有打开。

#### V11：FileVersion 恢复容量，但静态 workflow 信号不预测伤害

V11 不放宽 V10 门槛，而是改变合法 reuse scope：

> 一个旧 source view 即使跨越 workspace version，只要 canonical raw-tool
> provenance 能证明对应文件之后没有被写入，就仍可复用。

固定 64 sessions / 192 later-turn requests：

- file-version reusable fraction median：21.43%；
- cost-positive fraction median：21.43%；
- reuse in at least two later turns：100%；
- median copy islands：4；
- stable source-view tokens：206,378；
- unresolved source-view tokens forced Dense：510。

容量和机械负对照通过，但完整 development causal atlas 显示：

- 32 sessions；
- 8 disturbances；
- exactly 4,960 rows；
- missing/extra/duplicate keys：0；
- lookup p95：0.04795 ms，门槛 `<2 ms`，通过；
- workflow-feature delta-\(R^2\)：0.02467，
  95% CI [0.01062, 0.04697]，门槛 `≥0.05`，失败；
- distance≥2 safe-vs-unsafe harm reduction：-119.711，
  CI low -211.419，门槛 `≥30%` 且 CI low `>0`，失败。

V11 证明了合法容量和机械执行，但没有证明静态 workflow feature 能找到
低伤害模块，因此 P1 workflow accuracy 和 SGLang TTFT 保持关闭。

#### V12：从静态 feature 转向模型状态 probe

V12 完全继承 V11 合法候选，不增加任何新复用区域。对每个候选：

1. 在 target 当前上下文中 Dense 重算固定 head：
   \(H\in\{8,16,32,64\}\)；
2. 比较 source/target 的全层 K/V cosine deviation；
3. source K 按位置差执行 RoPE shift，V 不旋转；
4. 令
   \(score=\max(d_K,d_V)\)；
5. score 不高于冻结 threshold、body cost-positive 且不超过 4 islands 时，
   复制 body，否则 body Dense。

本次 development calibration：

- 32 development sessions；
- 96 later-turn requests；
- 1,196 unique candidate modules；
- 4,784 complete workflow observations；
- 4,639 `(head, threshold)` configurations；
- feasible configurations：0；
- calibration lock：FALSIFIED；
- holdout measurements read：false。

注册可行性要求同时满足：

- median cost-positive copied fraction ≥15%；
- selected per-module splice p95 JS ≤`1e-3`；
- teacher top-1 changes = 0；
- copy-all harm reduction ≥30%。

复核派生分析（不是单独注册的 headline）显示，最接近容量门槛的 head=16
配置可达到约 19.12% capacity，但 harm reduction 只有约 24.14%；反过来，
满足 ≥30% harm reduction 的配置最多只有约 7.53% capacity。由此判断当前
失败是 probe 风险排序的 specificity 不足，而不是单模块 JS/top-1 绝对门槛
先失败。

由于 calibration 失败：

- development stress completion 未运行；
- formal sequential composition 未运行；
- copy-all/shuffled request-level composed gate 未运行；
- holdout 未打开；
- P1 objective accuracy 和 TTFT 未运行。

## 5. 方法演进揭示的共同问题

### 5.1 “token-identical”只保证可搬运，不保证状态等价

相同 token 在不同 prefix 下会产生不同 hidden state 和 KV。V11 的
file-version 稳定性解决的是“内容与文件版本是否合法”，没有解决
“模型是否以相同方式理解这段内容”。

### 5.2 静态 coding signal 的预测力不足

AST node kind、graph distance、module type、workspace version 和
file provenance 都有清晰语义，但目前没有一个信号在 matched controls
下稳定预测 downstream logit harm。

### 5.3 当前 ProbeHead 压缩了过多结构信息

V12 把：

- transformer layer；
- KV head；
- head token position；
- K 与 V 的不同变化模式

压缩成两个平均 cosine deviation，再取最大值。少数关键层/head 的危险变化
可能被平均掉；大量无害的小变化也可能抬高 score。因此 score 与 causal
splice harm 的排序仍有较大重叠。

### 5.4 capacity、安全性和执行形状是三个不同门槛

- V9/V10 首先死于合法容量；
- V11 有容量但死于信号；
- V12 有低绝对误差但无法同时满足容量和相对 harm reduction；
- ASTSpan 有大量 copied tokens，却死于准确率和 fragmentation；
- AST-Island 降低 fragmentation 后仍未产生正 TTFT。

只优化 copied-token ratio 不足以预测端到端速度。

## 6. 两位合作者如何在两个研究分支上解耦

### 6.1 旧协作方式的问题

旧实现把以下逻辑累积在同一个大 cache/branch 中：

- coding/AST selector；
- KV segment identity；
- source pool；
- physical KV movement；
- scheduler hooks；
- prefetch；
- residency/eviction；
- experiment-result auto-activation；
- paper 和大规模 benchmark launcher。

此外，旧 `feature/context-aware-kv-reuse` 是原活动分支的祖先，不是一个
真正隔离的 collaborator branch。直接继续在两个长生命周期分支上开发会
导致：

- 两边同时修改 `radix_cache.py`；
- 无法判断 speedup 来自 coding policy 还是 prefetch timing；
- feature flag 相互隐式启用；
- merge 时把实验结果、paper 和 runtime 一起带入；
- 一个 owner 的变更可能改变另一个 owner 的 baseline。

旧 collaborator 状态已只读归档为：

```text
archive/context-aware-kv-reuse-20260717 @ 015d58c969cb
```

归档包括当时没有进入远端的四个本地 commits，但不允许整支 merge 回新结构。

### 6.2 新分支拓扑

```text
                         kvflow/shared-core
                   policy-neutral KVCOMM contract
                     /                       \
                    / merge shared updates   \ merge shared updates
                   v                           v
 research/coding-aware-lossy             research/prefetch
 你这一侧：决定“复制什么”              合作者一侧：决定“何时/搬到哪里”
                   \                           /
                    \ only composition merges /
                     v                       v
                integration/coding-aware-prefetch
                 只做组合测试和薄 adapter
```

实际分支：

| 分支 | owner 责任 | 明确禁止 |
|---|---|---|
| `kvflow/shared-core` | segment identity、generation、lease/resource lifecycle、transfer validation、Radix adapter | coding feature、prefetch policy、results、paper |
| `research/coding-aware-lossy` | AST/workflow/session/probe 信号；生成 Dense/copy plan | scheduler、prefetch coordinator、eviction、`ensure_resident` |
| `research/prefetch` | prefix/middle KV residency、deadline/priority、host/storage→device loader | AST/coding selector、coding experiment labels |
| `integration/coding-aware-prefetch` | 合并两个 owner 分支，运行四模式 composition 和薄 adapter | 新研究逻辑、results、paper |

### 6.3 共享的最小数据契约

Coding owner 不调用预取，只产生：

```text
KVReusePlan
  ├── target_token_ids
  ├── copied_spans: TransferSpan[]
  └── dense_ranges: DenseRange[]
```

Prefetch owner不判断 coding 风险，只产生 device-resident handle：

```text
KVPrefetchHint
  -> MiddleKVPrefetchAPI.prefetch(...)
  -> PrefetchTicket.wait()
  -> KVSegmentHandle(residency=DEVICE)
```

二者唯一的交点是共享 handle 和 plan：

```text
prefetch owner:
Host/Storage KV
   -> device-resident KVSegmentHandle
                                \
                                 -> TransferSpan.source
                                /
coding owner:
online-visible signals
   -> copy/dense decision
   -> KVReusePlan
```

KVCOMM core 执行前统一验证：

- model/cache identity；
- token count 和 token hash；
- generation 是否 stale；
- source/target slice 是否完全一致；
- source residency；
- span bounds 和 non-overlap；
- full coverage；
- copied K 是否全部 RoPE-rotated；
- copied K/V token 数是否一致。

失败时对受影响 chunk fail closed 到 Dense。

### 6.4 双方的日常协作规则

Coding-aware owner（你这一侧）：

```bash
git switch research/coding-aware-lossy
git merge kvflow/shared-core
export SGLANG_KVCOMM_CORE=1
export SGLANG_CODING_AWARE_LOSSY=1
export SGLANG_KV_PREFETCH=0
```

职责：

- 冻结 cohort、registration、threshold 和 controls；
- 只使用 online-visible coding/workflow 信号；
- 输出 `KVReusePlan`；
- 在 prefetch 关闭时独立验证 accuracy/TTFT；
- 不修改 scheduler、residency 或 eviction。

Prefetch owner（合作者一侧）：

```bash
git switch research/prefetch
git merge kvflow/shared-core
export SGLANG_KVCOMM_CORE=1
export SGLANG_CODING_AWARE_LOSSY=0
export SGLANG_KV_PREFETCH=1
```

职责：

- export/register middle KV；
- 根据 deadline 和 priority 调度 residency；
- 管理 ticket、lease、drop 和 host/device ownership；
- 连接 scheduler 和未来 HiCache storage loader；
- 不读取 AST、workflow label 或 coding experiment result。

共同规则：

1. 两个 research 分支之间不直接 cherry-pick。
2. 共享 bug 先修到 `kvflow/shared-core`，再分别 merge。
3. 组合行为只在 `integration/coding-aware-prefetch` 测试。
4. integration 发现 policy bug 时回到对应 owner 分支修，不在 integration
   长期维护 fork。
5. paper 和大 experiment directory 不作为代码依赖。
6. 所有 feature flag 默认关闭，runtime 不通过“发现 results JSON”自动启用。
7. coding-only、prefetch-only 和 combined 必须分别报告，防止把预取收益归给
   coding signal。

### 6.5 Middle-KV 的资源所有权

Prefetch 分支已经实现的 v1 API：

```text
source request computes KV
  -> export_middle_kv
  -> host KVSegmentHandle
  -> prefetch(key, deadline, priority)
  -> PrefetchTicket
  -> wait()
  -> device KVSegmentHandle
  -> KVReusePlan / KVCommManager.execute
  -> ticket.release()
  -> drop() when no longer cacheable
```

source request 的原 device slots 仍由原 request/RadixCache 管理；
`export_middle_kv` 不会偷偷 free 或 pin 它们。新 host/device 副本通过独立
generation、lease 和 disposer 管理。

当前 ticket 是同步实现，但 API 为未来 CUDA event/transfer stream 保留了
异步兼容形状。不能因此宣称异步预取已经完成。

## 7. 当前分支和验证状态

审计时分支快照：

| worktree | branch | HEAD | 状态 |
|---|---|---|---|
| `shared-core` | `kvflow/shared-core` | `c16bfbb8e8cc83a8b23858808f52833be9091101` | clean；tag `kvcomm-core-v0.1-rc3` |
| `prefetch` | `research/prefetch` | `fa86f8f16e6cf08fa3e51f9f9fd5b12cfc303fc0` | clean |
| `coding-aware` | `research/coding-aware-lossy` | `9574685777ab1a87781d4356cc3fbca4a537afb3` | local ahead 1；V12 与本报告仍在 worktree changes |
| `integration` | `integration/coding-aware-prefetch` | `d4a7ec132d80597c7b55a562beb8432e804ab127` | clean |

2026-07-18 复跑：

- coding suite：34 passed；branch scope OK；
- prefetch suite：30 passed；branch scope OK；
- integration composition：1 passed；branch scope OK。

这些是 unit/reference contract 证据，不是 model-server TTFT 证据。

当前工程分类仍应是：

```text
INTERFACE_COMPLETE / SERVER_CANARY_PENDING
```

尚未完成：

- shared Radix adapter 的真实 model-server GPU request canary；
- production allocator 上的 middle-KV export/prefetch/consume；
- HiCache storage payload 验证；
- scheduler prediction 与真正异步 transfer；
- feature-off / coding-only / prefetch-only / combined server matrix；
- sustained concurrency 下的 lease/ref/allocator leak 检查。

V12 的 RTX 4090 canary 是 Qwen2.5-Coder-7B reference executor 的 probe 和
组合机械 canary，不等于 SGLang model-server canary。

## 8. 当前可以对外使用的表述

允许：

- “修复 runtime 后，历史 Uniform 30% 的 31.2% headline 未复现。”
- “FileVersion 将合法 middle-KV 容量提高到 21.43%，但 V11 的
  coding-specific P0 signal gates 失败。”
- “V12 的动态 K/V probe 在 development 上无法同时满足 15% 容量和
  30% harm-reduction 门槛。”
- “KVCOMM 已把 coding policy 与 prefetch residency 解耦，并通过
  unit/reference composition tests。”
- “当前结果是多条候选机制的诚实 falsification，而不是端到端成功。”

禁止：

- “当前算法已加速 coding agent。”
- “V12 精度没有损失。”
- “TaskCone 已证明 coding-specific 选择优于通用复用。”
- “V11 P0 通过。”
- “V10 有约 33% 合法容量。”
- “middle-KV 已经实现生产级异步预取。”
- “所有 AST 或 workflow 信号都不可能有效。”

## 9. 建议的下一阶段

### 9.1 Coding-aware

不要在 V12 registration 内放宽 threshold 或 gate。若继续，应注册独立 V13：

- 保持 holdout 密封；
- 使用 layer-wise/head-wise quantile、max-tail 或 learned-on-development
  的低维风险特征，避免全局均值抹平关键变化；
- 明确加入 module/body length、source-target prefix delta 和局部 attention
  amplification；
- 首先做 ranking/AUC 或 top-risk capture 的离线验证；
- 再做 capacity–harm Pareto；
- 只有 development sequential composition 通过，才打开 holdout；
- 只有 P0 holdout 通过，才注册 objective workflow accuracy 和 TTFT。

### 9.2 Prefetch

Prefetch owner可以独立继续，不等待 V13：

- 把 `MiddleKVPrefetchAPI` 接到真实 scheduler admission；
- 在 production allocator 上验证 host→device payload；
- 将同步 ticket 替换为 CUDA event/stream-backed ticket；
- 测量 deadline miss、dedup、lease lifecycle 和并发资源增长；
- 使用 exact-transfer workload，避免和 lossy signal 混合归因。

### 9.3 Integration

只有两条 owner 路线各自通过独立 gate 后，再运行：

1. feature-off；
2. coding-only；
3. prefetch-only；
4. coding+prefetch。

四种模式必须使用同一 target prompt、source pool、eligible set 和请求顺序。
最终报告分别给出：

- coding policy 增加了多少 reuse；
- prefetch 减少了多少 residency wait；
- 两者组合是否存在非线性交互；
- 离线 KV 构建是否计入系统边界。

## 10. 权威制品索引

### 10.1 历史只读 checkout

```text
/home/gfy/CodeMAS_Project/sglang-kvflow
```

不得 reset、cleanup 或在其中继续写 paper/report。主要证据：

```text
results/impactkv_astspan_retest_20260716/PRIOR_RUN_INVALIDATION.md
results/impactkv_astspan_retest_20260716/FINAL_VERDICT.md
results/impactkv_astisland_v1_20260716/FINAL_VERDICT.md
results/impactkv_workflowmodule_v9_20260716/FINAL_VERDICT.md
results/impactkv_sessiongraph_v10_20260717/R0_SCHEMA_CORRECTION_REPORT.md
results/impactkv_sessiongraph_v10_20260717/FINAL_VERDICT.md
results/impactkv_sessiongraph_v11_20260717/R0_FILE_VERSION_CAPACITY_REPORT.md
```

### 10.2 V11 完整外部制品

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_sessiongraph_v11_20260717/
```

权威 verdict：

```text
P0_FINAL_VERDICT.md
SHA-256:
628c42be00eb9476e7b9a8365bb37410a17d76fd16d782d3a17d5bae970cfefa
```

### 10.3 V12 外部制品

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_probehead_v12_20260717/
```

```text
DEVELOPMENT_CALIBRATION_REPORT.json
SHA-256:
fa168dc7ade15f67a23be557d303cde6235833cce31f5e9b3d3861300694a996

CALIBRATION_LOCK.json
SHA-256:
97c653f151877a3bf218fedc7702ca3bda21baa640617780137204354de01117
```

`CALIBRATION_LOCK.json` 的 `holdout_measurements_read=false` 是当前 holdout
仍密封的机器可读证据。

## 11. 仓库清理与 SGLang 结构审计

### 11.1 本次清理结果

删除了两个已经被完整 V11 aggregate 取代的阶段性脚本：

```text
benchmark/multi_workflow/analyze_sessiongraph_v11_negative_controls.py
benchmark/multi_workflow/analyze_sessiongraph_v11_upstream.py
```

删除理由：

- negative-control 独立脚本只处理 1,280-row 子集，完整 4,960-row
  aggregate 已验证同一负对照并产生 final gate；
- upstream 脚本的输出状态明确写着
  `DIRECTIONAL_CHECKPOINT_NOT_A_FORMAL_GATE` 和
  `formal_p0_complete=false`；
- 两个脚本均没有 runtime/import caller，只剩旧 handoff 中的路径引用；
- 保留它们会继续制造“partial checkpoint 等于 final gate”的误读风险。

没有删除的 V11/V12 脚本：

- registration/design builder；
- canonical provenance、capacity 和 label builder；
- reference measurement executor；
- formal aggregate/gate；
- artifact validator；
- unit tests。

这些代码对应的假设虽然被 falsify，但仍构成完整复现与审计链。负结果不是
“无效脚本”的删除理由。

Markdown 文档从本分支新增的 7 个入口压缩为 4 个：

```text
KVFLOW.md
CODING_AWARE_HANDOFF_20260717.md
docs/kvflow/ARCHITECTURE.md
docs/kvflow/WEEKLY_RESEARCH_AUDIT_20260718.md
```

删除：

```text
_archive/handovers/README.md
docs/kvflow/HANDOFF.md
docs/kvflow/STATUS.md
```

其中旧 `HANDOFF.md` 仍停留在 `kvcomm-core-v0.1-rc1`，而当前 shared core
已经是 rc3；`STATUS.md` 与 `KVFLOW.md`、coding-aware handoff 大量重复。
当前 runtime/research 状态已合并到 `KVFLOW.md`。

### 11.2 与最初 SGLang 的 diff

比较基线：

```text
origin/main @ 3343a79466aa714d34a14d08d3929f7953a47212
```

`kvflow/shared-core@c16bfbb8e` 相对基线：

- 18 files changed；
- 1,923 insertions；
- 原 SGLang 文件只修改两个：
  - `cache_init_params.py`：增加可选 `kvcomm_config`，5 行；
  - `radix_cache.py`：创建/reset `KVCommManager`，6 行；
- 其余代码均位于新建的 `mem_cache/kvcomm/`、测试、工具和文档。

这说明源码侵入面很小，分支方向也基本正确：

```text
SGLang original runtime
  ├── CacheInitParams：一个可选注入点
  └── RadixCache：一个 KVCOMM facade

new isolated modules
  ├── kvcomm/
  ├── coding_aware/
  ├── kvcomm_prefetch/      # prefetch branch
  └── kvflow_integration/   # integration branch
```

`research/coding-aware-lossy@957468577` 的 committed diff 为 35 files /
5,952 insertions。新增体积主要来自 `benchmark/multi_workflow/` 的离线 V11
研究 executor，而不是 production SGLang runtime。V12 与本报告目前仍是
worktree changes，尚未作为稳定 runtime commit。

### 11.3 结构上做对的部分

- shared core 不导入 AST、workflow result 或 scheduler policy；
- coding policy 不导入 prefetch，也不调用 `ensure_resident`；
- feature flags 默认关闭；
- plan 使用完整 Dense/copy coverage，而不是隐式 zero gaps；
- source/target token、bounds、generation、residency 和 full-RoPE 都有
  fail-closed 检查；
- prefetch 的 host/device resource 有 release callback 和 lease；
- 原始 SGLang 文件改动只有 11 行，未来容易 rebase/upstream；
- coding-only、prefetch-only 和 integration worktree 可以独立测试。

### 11.4 当前结构的关键问题

| 严重度 | 问题 | 证据与影响 |
|---|---|---|
| P0 | 没有 production execute path | `RadixCache` 只实例化/reset manager；生产代码没有调用 `register_segment()` 或 `KVCommManager.execute()` |
| P0 | residency transition 不使旧 handle 失效 | `ensure_resident()` 更换 backend ref 但 generation 不变；旧 host handle 仍被 `is_current()` 判为 true |
| P0 | KVCOMM store 与 Radix/HiCache lifecycle 双重记账 | Radix node/allocator 可能释放 slots，而独立 store 仍持有逻辑 current handle |
| P0 | Dense fallback 尚不能映射到真实 prefill scheduler | `dense_prefill(start,length)` 只是 backend callback；SGLang 没有接入任意交错 Dense/copy span 的 request execution |
| P1 | Handle identity 不足以覆盖所有缓存兼容条件 | key 缺少 model revision、TP/rank、KV layout、attention backend、RoPE scaling、quantization/page configuration |
| P1 | RoPE adapter 只验证了当前简单布局 | dynamic/scaled RoPE、MLA、multimodal positions、非标准 KV layout 尚未覆盖 |
| P1 | 缺少 CUDA stream/event 所有权 | copy、rotation、prefill 之间没有 production stream dependency 或并发 hazard contract |
| P1 | Store 容量按 record 数而不是 token/byte | `max_records=4096` 不能表达真实 GPU/host memory budget |
| P2 | 全覆盖验证按 token 构造 Python set | 长上下文为 O(prompt tokens) 内存；应改为排序 interval validation |
| P2 | 测试位于 runtime package | `python/sglang/.../test_*.py` 应最终移动到 SGLang 既有 `test/registered` 层级 |

residency handle 最小复现：

```text
register HOST handle generation=1, backend=host-ref
ensure_resident(...) -> DEVICE handle generation=1, backend=device-ref

store.is_current(old_host_handle) == True
old_host_handle.backend_ref == host-ref
```

在并发 load 中，一个线程还可能返回随后被另一个线程释放的 device ref。
修复方案必须二选一：

1. 每次 residency/backend ref 改变都提升 generation，使所有旧 handle
   fail closed；或
2. handle 只引用稳定 record/indirection，backend ref 不复制到不可更新的
   value object，并用 lease/version 保护转移。

### 11.5 建议的目标结构

不建议继续把完整 runtime owner 放在 `RadixCache` 对象内部。更合理的结构：

```text
Scheduler / ModelRunner request path
  │
  ├── CodingReusePlanner
  │     online-visible signals -> KVReusePlan
  │
  ├── KVSegmentRegistry adapter
  │     anchored to real Radix/HiCache node and allocator lifecycle
  │
  ├── ResidencyService
  │     prefetch owner; host/storage/device + stream/event
  │
  └── PrefillPlanExecutor
        contiguous exact copy first
        dense fallback through real scheduler
        telemetry and transactional failure handling
```

建议接入顺序：

1. 只支持一个大 contiguous、same-context、exact middle segment；
2. 在真实 model server 中证明 Dense completion/logits identity；
3. 接入 allocator invalidation 和 lifecycle；
4. 接入 prefetch residency；
5. 支持少量 bounded islands；
6. 最后才允许 lossy coding policy 使用该 executor。

在第 1–4 步完成之前，继续优化 V12/V13 signal 会把算法误差与尚未存在的
production executor 混在一起。

## 12. Lossy 技术路线总审视

### 12.1 路线迭代真正说明了什么

| 路线 | 得到的正证据 | 失败或限制 |
|---|---|---|
| Uniform FRAC | 证明“少算 prefill”可能影响 TTFT | 旧 31.2% runtime-invalid；修复后仅约 1.9% |
| TaskCone L2 | shape-aligned、强 target signal 下可出现 30/30 与大速度观察 | controls CI low=0；不是 unseen/coding-specific |
| ASTSpan / AST-Island | AST 可形成明确 Dense/copy partition；bounded islands 改善执行形状 | accuracy regression 或速度仍负 |
| WorkflowModule V9 | 建立真实 prompt 模块审计 | 合法容量仅 0.33% |
| SessionGraph V10 | same-session 和 dependency guards 合理 | schema 修正后容量 9.12% |
| FileVersion V11 | 合法 non-prefix capacity 达 21.43% | 静态 workflow feature 不预测 splice harm |
| ProbeHead V12 | 单模块绝对 JS 低，top-1 可保持 | 容量与 ≥30% harm reduction 无共同可行点 |

因此项目还保留两个值得继续的事实：

1. 在真实 file-version scope 中，确实存在超过 20% 的合法 middle-token
   机会；
2. 一部分 token-identical 模块的 source KV 对最终单步 logits 的影响很小。

但至今没有证明：

- 能以低成本在线区分这些模块；
- 多模块顺序组合仍安全；
- 生成轨迹和官方 workflow tests 保留；
- 在真实 SGLang executor 上产生正 TTFT；
- coding signal 优于 exact-budget shuffled/uniform；
- 离线构建、residency 和 memory 成本后系统仍有净收益。

### 12.2 V12 指标存在的根本错位

当前 calibration 的 harm reduction 基于：

```text
baseline = mean(per-module splice JS over all candidates)
selected = mean(per-module splice JS over accepted candidates)
reduction = (baseline - selected) / baseline
```

问题：

- 没有按 copied tokens 或 body length 加权；
- 比较的是独立单模块 splice，不是请求级 sequential composition；
- accepted 为空时 selected 被设为 0，会产生 100% 表面 reduction，虽然
  15% capacity gate 阻止了最极端退化；
- 最后一个 prompt token 的 teacher JS/top-1 不代表多步 generation 或
  functional accuracy；
- score 把 layer、KV head 和 token position 全部平均，再对 K/V 取 max，
  可能淹没少数关键层/head 的危险变化；
- 固定 head 只观察模块开头，无法保证 body 中后部的状态偏差不会增长。

因此 V12 的失败不只是“阈值没调好”，而是 proxy、aggregation 和最终任务
目标没有充分对齐。

### 12.3 经济性约束

当前单卡 cost model 表明 copy/token 明显便宜于 Dense/token，但这不足以
推出 end-to-end speedup。真实净收益还包括：

```text
saved dense prefill
  - KV materialization / copy
  - full K rotation
  - island launch cost
  - probe head recompute
  - probe comparison
  - allocator / residency wait
  - scheduler fragmentation
  - offline build and memory cost
```

ASTSpan 的 66.5 个 stages 和 AST-Island B8 仍为 -5.04% 已经说明：
copy token 数不是正确的速度代理。新路线必须优先最大化“少数大 contiguous
islands 的净节省”，而不是最大化离散 selected modules。

### 12.4 技术路线判决

当前建议不是彻底终止 lossy middle-KV，而是：

> **暂停新的 signal sweep，把项目从 policy-first 改成
> executor-and-workload-first。**

Go/no-go 顺序：

1. **Exact executor gate**：真实 server 中一个大 middle island 与 Dense
   logits/completion identity，mechanical invalid=0；
2. **Cost gate**：包含 allocator、copy、rotation 和 request ordering 后，
   contiguous exact middle reuse 有正 TTFT CI；
3. **Workload gate**：真实而非重组偏差过大的 coding session 中，
   file-version capacity 仍达到注册门槛；
4. **Ranking gate**：新 signal 在 development 和 sealed holdout 上，
   token-weighted risk ranking 优于 shuffled；
5. **Sequential safety gate**：请求级 composed logits 与 top-1；
6. **Objective gate**：官方 workflow tests 零 regression；
7. **System gate**：coding-only 相对 exact prefix/Dense 有正 TTFT，且优于
   matched generic controls；
8. **Integration gate**：再与 prefetch owner 的 residency 收益组合。

### 12.5 如果注册 V13，应改变什么

V13 不应只是换一组 V12 threshold。最低要求：

- 使用 token-weighted/request-level objective；
- signal 设计冻结在 holdout 之前；
- layer/head tail quantile、局部 max 和 body checkpoints，而非全局均值；
- 限制为 1–2 个大 contiguous file-version islands；
- 把 probe 和 copy 的完整在线成本放入选择；
- development calibration 后先做真实 sequential composition；
- matched shuffled 必须保持相同 copied tokens、island lengths 和执行形状；
- teacher-logit gate 后仍需 objective tests；
- 只有 shared exact executor gate 通过，才允许执行 V13。

优先级上，V13 低于 KVCOMM production-path 修复。

## 13. 最终总结

本周最重要的进展不是得到一个新的正 speedup，而是把研究问题分成了四个
可独立证伪的层次：

```text
机械搬运是否合法？
        ↓ 已基本解决
是否存在足够的合法 middle-KV 容量？
        ↓ V11 证明存在
能否准确识别低伤害区域？
        ↓ V11/V12 当前失败
能否转化为 objective accuracy-preserving TTFT speedup？
        ↓ 尚未授权测量
```

同时，工程协作从“两个 owner 修改同一混合 cache 实现”转为：

```text
coding owner 负责 what
prefetch owner 负责 when/where
shared core 负责 identity/lifecycle/safe transfer
integration 负责 composition
```

这使下一轮即使再次失败，也能明确知道失败来自信号、容量、transfer、
residency、scheduler 还是执行形状，而不会再用一个无法归因的 TTFT 数字覆盖
所有问题。
