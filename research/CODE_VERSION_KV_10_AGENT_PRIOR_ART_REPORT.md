# Codebase Source-Version-Aware KV：2025–2026 分段 Prior-Art 复核

最后更新：2026-07-15T19:43:11-07:00

状态：**已完成。最终范围为 2025-01-01 至 2026-07-15，七个保留分段全部返回并完成统一复核。**

## 1. 复核目的

本轮最初将 2024-01-01 至 2026-07-15 连续均分为十段。2026-07-15，用户决定取消 2025 年之前的分段，只保留 2025 和 2026 年工作。因此最终证据范围收缩为 **2025-01-01 至 2026-07-15**；`version-scan-01` 至 `version-scan-03` 已收到停止指令且其结果不得纳入，`version-scan-04` 只保留 2025-01-01 至 2025-01-06。

保留分段继续用于降低以下漏检风险：

- 论文标题没有出现 Git、version 或 KV cache；
- arXiv 首次提交与正式 venue 跨年；
- workshop、OpenReview 或代码先于 DBLP；
- 同一工作以 prompt editing、persistent KV、function object、incremental context 等邻近术语出现；
- 最新 7/30/90 天工作尚未进入引用索引。

目标是再次挑战以下核心 thesis：

> 是否已有系统将 repository/source version 作为 Transformer attention KV 的一等 identity、validity 和 coherence 信息，并统一实现跨版本复用、源码/依赖失效、增量 repair/rematerialization、branch/worktree isolation 与物理 memory-tier lifecycle。

## 2. 代理状态与最终负责区间

论文按**首次公开日期**归属。后续 revision、正式 venue 和官方代码追踪至 2026-07-15，但不会改变归属区间。

| Agent | 首次公开日期范围 | 天数 | 状态 |
| --- | --- | ---: | --- |
| `version-scan-01` | 2024-01-01 – 2024-04-02 | 93 | 已停止；排除 |
| `version-scan-02` | 2024-04-03 – 2024-07-04 | 93 | 已停止；排除 |
| `version-scan-03` | 2024-07-05 – 2024-10-05 | 93 | 已停止；排除 |
| `version-scan-04` | 2025-01-01 – 2025-01-06 | 6 | 已完成 |
| `version-scan-05` | 2025-01-07 – 2025-04-09 | 93 | 已完成 |
| `version-scan-06` | 2025-04-10 – 2025-07-11 | 93 | 已完成 |
| `version-scan-07` | 2025-07-12 – 2025-10-12 | 93 | 已完成 |
| `version-scan-08` | 2025-10-13 – 2026-01-12 | 92 | 已完成 |
| `version-scan-09` | 2026-01-13 – 2026-04-14 | 92 | 已完成 |
| `version-scan-10` | 2026-04-15 – 2026-07-15 | 92 | 已完成 |

第十段额外要求专项覆盖截至 2026-07-15 的最近 7、30 和 90 天。

最终报告只整合 `version-scan-04` 至 `version-scan-10` 的 2025–2026 证据。即使被停止的代理稍后返回 2024 结果，也只保留为运行记录，不进入候选计数、去重矩阵或 novelty verdict。

## 3. 严格分类

### A：直接先例

Repository/source version 明确控制普通 attention K/V tensor 的：

- identity；
- validity；
- reuse；
- invalidation；
- repair；
- rematerialization；
- coherence。

### B：强邻近

处理 mutable prompt、context 或 code edit 后的 attention-KV：

- splice；
- relocation；
- offset reconstruction；
- selective recompute；
- repair。

但没有完整 repository version lifecycle。

### C：系统 primitive

提供：

- content-addressed/persistent KV；
- CPU/GPU/SSD tier；
- modular/non-prefix/function/chunk KV；
- runtime checkpoint/version log。

但不理解 Git/source version。

### D：非 KV 邻近

处理：

- Git/Merkle repository index；
- version-aware RAG；
- embedding/graph；
- agent text memory；
- incremental build。

但不保存 Transformer attention K/V tensor。

## 4. 防止误报的边界

- Token/content hash 不等于 repository source-version coherence。
- Runtime checkpoint epoch 不等于 Git commit、branch 或 worktree version。
- Git-aware RAG/index 不等于 attention KV。
- 普通 exact-prefix cache 不等于跨 source version reuse。
- AST/function ID 只作为索引，不自动构成 version-aware coherence。
- 论文 revision 落在某区间，不代表其首次公开日期属于该区间。

## 5. 每个保留代理必须返回的证据

1. 报告首行明确负责日期范围。
2. 数据库、检索式和引用链。
3. 候选论文首次日期、最新 revision、venue、代码和来源。
4. 正文机制证据，而非仅摘要。
5. A/B/C/D 分类及排除理由。
6. Boundary spillover。
7. A 类计数、本段 verdict、遗漏风险和置信度。

## 6. 最终整合规则

七份保留 memo 返回后将执行：

1. 按 arXiv ID、DOI、标题和代码仓库去重。
2. 将同一工作的 arXiv、OpenReview 和正式 venue 合并。
3. 对所有 A 类候选重新阅读全文，至少进行一次主会话独立复核。
4. 对 B/C/D 最接近工作建立机制矩阵。
5. 单列与已有报告相比的新发现、改判和撤回项。
6. 负搜索只写为“本次检索未发现”，不写成绝对不存在。
7. 更新安全 novelty claim、论文 thesis 和实现优先级。

## 7. 分段结果

### 7.4 `version-scan-04`：2025-01-01 – 2025-01-06

- 全文核查 2 篇区间内候选。
- 分类结果：**A/B/C/D = 0/0/2/0**。
- A 类计数：**0**。
- A=0 置信度：高。
- FlashInfer 提供 KV layout、paged/radix attention kernel 和执行 primitive，但 cache identity 与 validity 由上层 serving engine 决定。
- HybridServe 在 host/GPU 间混合保存 KV 与 activation checkpoint，并按成本选择加载或重物化，但其版本是当前请求执行状态，不是 source artifact version。

本段结论：

> 2025 年最前 6 天未发现直接先例；相关工作只提供可被未来版本化系统复用的底层 attention execution 与 rematerialization primitive。

### 7.5 `version-scan-05`：2025-01-07 – 2025-04-09

- 全文核查 11 篇区间内候选。
- 分类结果：**A/B/C/D = 0/3/6/2**。
- A 类计数：**0**。
- A=0 置信度：高，代理估计约 `0.92`。
- B 类为 Cache-Craft、MPIC、KVShare：均可在上下文变化后选择性修复或重算 K/V，但使用 chunk、文件引用或 token hash，不理解 commit、branch、worktree、dirty patch epoch 和 dependency lineage。
- C 类包括 Parallel KV-Fusion、PRESERVE、KVLink、SentenceKV、FlowKV、HyperRAG：覆盖模块化表示、位置重定位、持久存储或 GPU/CPU/SSD tier primitive，但没有 repository source-version coherence。
- D 类为 SyncMind 与 Repository-level Code Search：两者确实使用 commit、repository state 或历史变更，但不保存 Transformer attention K/V。
- Boundary spillover 中的 PIE、CacheBlend、EPIC、HCache 均按首次公开日期排除出本段；它们也不构成严格 A 类。

本段结论：

> 本次检索未发现首次公开于 2025-01-07 至 2025-04-09、并把 repository/source version 作为普通 attention-KV 一等 identity、validity、invalidation 或 coherence 信息的系统。

### 7.6 `version-scan-06`：2025-04-10 – 2025-07-11

- 核心候选 18 篇。
- 分类结果：**A/B/C/D = 0/1/12/5**。
- A 类计数：**0**。
- A=0 置信度：高，代理估计约 `0.90`。
- 唯一 B 类是 EFIM：通过重排 FIM prompt 保持 prefix/suffix exact cache，但没有 commit、artifact lineage、dependency invalidation 或跨版本 catalog。
- MemOS、LAG、FastLibra、KVFlow 等分别覆盖 activation memory versioning、持久 latent KV、dependency-aware residency 和 CPU/GPU lifecycle。
- `Towards the Versioning of LLM-Agent-Based Software`、SWE-Bench-CL、Code Graph Model 等覆盖 Git SHA、artifact manifest、repository evolution 或 graph，但不保存 attention K/V。
- MemOS 是本段最危险的相邻项：论文同时讨论 memory versioning 和 activation KV，但其实现仍以随机 UUID、source text 和 extraction time 管理 KV，没有 Git/source-version validity 或 invalidation。

本段结论：

> 已有工作覆盖 code-edit KV reuse、通用 memory versioning、dependency-aware KV residency、persistent latent KV 和 repository evolution；本次检索仍未发现 source version 直接控制 attention-KV correctness/coherence 的系统。

### 7.7 `version-scan-07`：2025-07-12 – 2025-10-12

- 全文核查 21 篇候选。
- 分类结果：**A/B/C/D = 0/5/14/2**。
- A 类计数：**0**。
- A=0 置信度：高，代理估计约 `0.90`。
- B 类包括 SamKV、KVCOMM、CIFLEX、SemShareKV、CacheClip，分别覆盖跨 context repair、base-KV + offset、推理分支/回滚、语义匹配复用和 chunk splice，但均没有 source-version validity。
- C 类包括 LMCache、AdaptCache、Halo、Oneiros、SmartCache、HiFC 等，覆盖持久化、workflow lineage、CPU/GPU/SSD tier 与 runtime lifecycle，但不理解 repository artifact/dependency lineage。
- D 类 LinkAnchor 与 Repository Memory/RepoMem 显式使用 branch、commit、diff 和 repository history，却只保存文本、图或摘要，不保存 attention K/V。
- 最危险的组合式邻近是 PIE 的代码编辑 KV 修正、LMCache 的持久多层管理，以及 RepoMem/LinkAnchor 的 Git history identity；这些能力仍分散在不同系统。

本段结论：

> 本次检索仍未发现将 repository source version 直接映射为 attention-KV identity、validity、dependency invalidation、branch isolation 和 stale audit 的统一协议。

### 7.8 `version-scan-08`：2025-10-13 – 2026-01-12

- 全文核查 12 篇区间内候选。
- 分类结果：**A/B/C/D = 0/1/10/1**。
- A 类计数：**0**。
- A=0 置信度：高，代理估计约 `0.87`。
- 唯一 B 类 Warp-Cortex 通过隐藏 reference token 更新 live `past_key_values`，但没有 repository/version identity，也不是 changed source artifact 的精确 repair。
- C 类包括 BanaServe、Cortex、TokenCake、Jarvis、KVTC、Continuum、ContextPilot、KVSwap、SGLANG-LSM 和 Don't Break the Cache，覆盖 workflow scheduling、持久化、压缩、tiering、TTL、content hash、prefix index 与 runtime eviction sync，但都不理解 source-version coherence。
- D 类 PortGPT 明确使用 Git history、branch 和 patch backport 语义，却只检索代码、diff 和历史文本，不保存或失效 attention K/V。
- ContextPilot 是最容易误报的候选：content-defined chunk/hash 只表示文本相等，runtime eviction sync 只表示缓存物理存在性，二者均不等于 commit/tree/blob、dirty patch 或 dependency validity。

本段结论：

> 该区间仍呈现清晰的三段分离：B 类会更新可变上下文 K/V，C 类会管理持久和分层 K/V，D 类理解 Git/branch；但没有工作把 source lineage 直接变成 attention-KV 的 identity 与 coherence contract。

### 7.9 `version-scan-09`：2026-01-13 – 2026-04-14

- 全文升级审查 20 篇候选。
- 分类结果：**A/B/C/D = 0/8/8/4**。
- A 类计数：**0**。
- A=0 置信度：高，代理估计约 `0.91`。
- 最接近 mutable-update/invalidation 的是 KEEP：根据 embodied memory 更新拆分 static/dynamic KV、选择性重算并从 CPU 加载，但版本对象是机器人环境记忆，不是 repository/source version。
- 最接近模块化或 tiered KV 的是 TableCache、ContiguousKV、COMB、KV Packet。
- 最接近 code-specific KV 的是 CodeComp；最接近 coding-agent session lifecycle 的是 MARS。
- Git/source 语义出现在 CAID、Lore、Repository Intelligence Graph 中，但它们不保存 attention K/V。
- 本段仍显示同一断裂：有 version/Git semantics 的工作没有 K/V；有 K/V update/tier semantics 的工作没有 repository source version。

本段安全 verdict：

> 本次检索未发现首次公开于 2026-01-13 至 2026-04-14、并把 repository/source version graph 作为普通 attention-KV 一等 coherence domain 的系统。

其 memo 同时指出以下 broad claims 已不安全：

- 首次处理动态内容更新后的 KV 失效；
- 首次做 code-structure-aware KV；
- 首次做 modular/tiered persistent KV；
- 首次给 coding agents 做 KV lifecycle；
- 首次把 Git branch/worktree 用于 agent isolation；
- 首次把 commit 作为 agent memory。

其最安全主张仍是：

```text
repository-version-aware attention-KV coherence
/ versioned causal KV materialized views for evolving codebases
```

### 7.10 `version-scan-10`：2026-04-15 – 2026-07-15

- 全文核查 21 篇候选。
- 互斥分类：**A/B/C/D = 0/5/15/1**。
- A 类计数：**0**。
- A=0 置信度：中高，代理估计约 `0.87`。
- 最近 7、30、90 天专项检索均得到 A=0。
- FCGraft 主类为 C、次标签 B：核心是函数级持久 KV 库和 GPU/DRAM lifecycle，局部 patch 是附加编辑能力。
- Models Take Notes 主类为 B、次标签 C：核心是字段变化后的 KV edit、erratum 和 selective recompute。
- Irminsul 最接近跨 edit unchanged chunk reuse，但只使用 CDC/content hash，不具有 repository version semantics。
- Leyline、KVEraser、Models Take Notes 和 SmoothAgent 已覆盖 mutable-context repair。
- ResidentClaim、Concordia、Execution-State Capsules 已覆盖 runtime identity、checkpoint、fork 或 rollback，但其 version 是 execution state，不是 source revision。
- Code Isn't Memory 已覆盖 working-copy Merkle diff 和 affected-chunk re-index，但不保存 attention K/V。

本段认为单一文献 novelty 风险仍较低，但组合显而易见性风险已经上升：

```text
Code Isn't Memory 的 source-update semantics
+ Irminsul/RedKnot 的 reusable KV objects
+ Leyline/Models Take Notes 的 edit repair
+ ResidentClaim/Concordia/Capsules 的 lifecycle/rollback
```

因此最终贡献必须严格落在：

1. repository snapshot / dirty-patch epoch 进入 KV identity；
2. source/dependency change 到 K/V invalidation 的协议；
3. commit/branch/worktree 多版本隔离；
4. unchanged artifact 的跨版本 exact alias；
5. source-aware stale audit 和 rollback。

## 8. 跨段去重与全文复核

七个保留分段共核查 **105 篇**主候选。按各 memo 的互斥主类汇总：

| 首次公开日期范围 | A | B | C | D | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2025-01-01 – 2025-01-06 | 0 | 0 | 2 | 0 | 2 |
| 2025-01-07 – 2025-04-09 | 0 | 3 | 6 | 2 | 11 |
| 2025-04-10 – 2025-07-11 | 0 | 1 | 12 | 5 | 18 |
| 2025-07-12 – 2025-10-12 | 0 | 5 | 14 | 2 | 21 |
| 2025-10-13 – 2026-01-12 | 0 | 1 | 10 | 1 | 12 |
| 2026-01-13 – 2026-04-14 | 0 | 8 | 8 | 4 | 20 |
| 2026-04-15 – 2026-07-15 | 0 | 5 | 15 | 1 | 21 |
| **总计** | **0** | **23** | **67** | **15** | **105** |

去重与归属规则：

- 论文只按最早可核验公开日期归属一个分段；
- arXiv、OpenReview、正式 proceedings 和同一官方代码实现合并为一项；
- 后续 revision 和 venue 不改变首次归属；
- boundary spillover 不进入分段计数；
- 三个已取消的 2024 分段完全排除。

所有分段 A 类均为 0，因此没有待升级复核的 A 类候选。主会话同时用此前年度报告中已阅读全文的高风险工作交叉检查边界，未发现需要从 B/C/D 改判为 A 的项目。

### 8.1 Closest-prior-art matrix

| 能力轴 | 最接近工作 | 已覆盖 | 仍缺失 |
| --- | --- | --- | --- |
| Mutable-context KV repair | Cache-Craft、KVCOMM、CacheClip、KEEP、Leyline、Models Take Notes | offset、splice、selective recompute、局部 repair | repository snapshot identity 与 source/dependency validity |
| Persistent/tiered KV lifecycle | LMCache、MemOS、KVFlow、ContextPilot、SGLANG-LSM、KV Packet | GPU/CPU/SSD、持久化、prefetch、eviction、runtime metadata | source event 驱动的 coherence state transition |
| Content/function KV objects | MEPIC、Irminsul、FCGraft、CodeComp | content hash、函数对象、position repair、代码结构信号 | commit/branch/worktree、多版本并存与依赖失效 |
| Git/repository semantics | SyncMind、RepoMem、LinkAnchor、PortGPT、CAID、Lore、Repository Intelligence Graph、Code Isn't Memory | commit、branch、diff、working-copy、repository graph | 普通 Transformer attention K/V tensor |
| Runtime version/rollback | Concordia、ResidentClaim、Execution-State Capsules | checkpoint、fork、epoch、rollback、durability | source revision 与 causal-context validity |
| Workflow-aware cache | KVFlow、Halo、Cortex、MARS | stage/DAG priority、residency、prefetch | evolving repository 的 artifact-version coherence |

跨段结果持续呈现同一个断裂：

```text
理解源码版本和 Git lifecycle 的工作通常不保存 attention K/V；
管理、修复或分层 attention K/V 的工作通常不理解源码版本。
```

## 9. 最终 Verdict

### 9.1 直接先例结论

截至 2026-07-15，本次对 2025-01-01 至 2026-07-15 的分段检索**未发现**公开论文或官方实现同时满足：

1. repository snapshot、commit、branch、worktree 或 dirty-patch epoch 进入普通 attention-KV identity；
2. source diff 与 dependency change 直接触发 K/V validity、invalidation 或 rematerialization；
3. unchanged artifact 支持跨版本 exact alias；
4. 多版本 reader/writer、rollback 与 GC 具有 MVCC-like isolation；
5. 上述逻辑与 GPU/CPU/SSD physical-page lifecycle、stale audit 闭环。

因此，当前仍相对安全的 thesis 是：

```text
repository-version-aware attention-KV coherence
/ versioned causal KV materialized views for evolving codebases
```

这里的结论必须表述为“本次检索未发现”，不能表述为绝对不存在，也不是专利或法律上的新颖性意见。

### 9.2 不安全的宽泛 claim

- 首个 mutable-context KV repair；
- 首个 function/code-structure-aware KV；
- 首个 persistent、modular 或 tiered KV；
- 首个 workflow-aware agent cache；
- 首个 Git-aware coding-agent memory；
- 首个使用 branch、commit、rollback 或 version 概念的 agent/runtime system。

### 9.3 相对安全的贡献边界

- source snapshot 与 causal-context fingerprint 共同定义 KV identity；
- source/dependency event 到 `KEEP/ALIAS/DIRTY/INVALIDATE/VERIFY/REMATERIALIZE/GC` 的可执行 coherence protocol；
- commit/branch/worktree/dirty-patch 下的多版本 KV isolation；
- unchanged artifact 的 guarded cross-version exact alias；
- source-aware stale audit、rollback 和 physical-tier consistency。

### 9.4 风险判断

单篇直接先例风险目前较低，但**组合显而易见性风险较高**。现有论文已经分别提供 Git/Merkle update semantics、function/content-addressed KV objects、mutable-context repair、runtime checkpoint 和 tiered storage。因此论文不能只做组件拼接，必须通过明确的 coherence protocol、正确性约束和真实 Git trace 证明：

- dependency-aware invalidation 比 file-level 全失效更精细且安全；
- guarded cross-version reuse 能显著降低 prefill/H2D 成本；
- branch/worktree isolation 不产生 stale read；
- metadata、存储和调度开销不会抵消收益。

对公开学术文献的 A=0 判断为高置信，但仍可能遗漏闭源工业实现、专利、未索引 workshop artifact 或匿名代码。

## 10. Presentation Summary

我们的核心 idea 是把超大、持续演化 Codebase 的 KV Cache，从“相同 prompt 才能命中”升级为“由 repository version 管理的因果物化视图”。系统按函数、模块前置区等非重叠 artifact 预计算 KV 并分层存放在 CPU/GPU；当 commit、branch、dirty patch 或依赖发生变化时，满足完整内容与上下文指纹的部分可以跨版本复用，受影响部分才失效并按需重算。潜在 novelty 不在 AST 索引或 CPU offload 本身，而在 source version/dependency 与 attention-KV identity、validity 和 coherence 的直接耦合。

主要 concern 是源码内容未变化并不保证它在不同 prompt、位置或 agent role 下产生完全相同的 K/V，因此跨上下文复用必须经过严格 fingerprint、质量 gate 和 dense fallback。依赖图还可能漏失效或过度传播，metadata、CPU→GPU 传输、存储和多 branch/worktree 状态也可能抵消 prefill 收益；这些问题必须通过真实 Git trace、正确率和端到端延迟实验验证。

最小 prototype 从干净 SGLang 基线开始：按 function/method、module preamble 等 canonical artifact 建立 source/dependency registry，将 exact KV 存入 CPU，先实现跨版本 exact alias、变更失效和 dense fallback。正确性稳定后，再加入 faithful KVCOMM 的跨上下文重建以及 KVFlow 的 workflow priority 和 CPU/GPU tier scheduling。
