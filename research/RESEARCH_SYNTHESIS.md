# 研究综合：面向超大 Codebase 的 Coding Agent KV Cache

最后更新：2026-07-15T23:09:45-07:00

## 核心结论

本项目不是单纯复刻某一篇论文，而是要在 SGLang 上构建一个 **Codebase-aware、workflow-aware、cross-context、CPU/GPU 分层的 KV Cache 系统**：

- KVFlow 决定哪些 cache 应保留、淘汰或从 CPU 调回 GPU。
- KVCOMM 决定同一代码 artifact 在不同 agent role/prefix 下如何经过位置对齐和 context offset 修正后复用。
- AST/代码索引决定整个 Codebase 如何切分、标注、检索、失效和映射到物理 KV pages。
- 固定 `Architect -> Coder -> Debugger` workflow 提供稳定的未来执行顺序，使 priority 与预取更可预测。

系统的最终形态更接近“位于 Coding Agent 之下的预计算 Codebase latent memory”，而不只是传统的相同 prompt prefix cache。

最新 prior-art 与多模型评估见 `research/AST_INDEXED_KV_CACHE_NOVELTY_REPORT.md`；逐项教学式解释见 `research/AST_INDEXED_KV_CACHE_STEP_BY_STEP.md`。该评估要求将论文主线进一步收窄为 **versioned causal KV materialized views**：AST index、function-level KV、CPU/GPU tier 和 workflow priority 均已有直接或强邻近先例，真正需要证明的是 evolving repository 中的一致性、依赖失效、cross-role reconstruction 和 artifact-level planning 闭环。

2024–2026 Git/repository/source-version-aware attention-KV 专项调研见 `research/GIT_CODEBASE_VERSIONED_KV_CACHE_PRIOR_ART.md`。专项结果进一步确认：

- PIE、Leyline 等已覆盖 mutable prompt/code edit 后的 KV repair；
- Irminsul、MEPIC、LMCache 等已覆盖 content-addressed/persistent KV objects；
- FCGraft 已覆盖 function-level KV object patch/update 和 GPU/DRAM lifecycle；
- Streaming Knowledge Compilation 已覆盖 time-evolving content 的 staleness 和 affected-entity recompilation；
- Code Isn't Memory 已覆盖 Git working-copy Merkle diff 和 incremental repository index，但不存 attention KV；
- Concordia 已覆盖 runtime checkpoint version/epoch coherence，但不是 source version；
- 本次检索未发现 repository source version 被直接用作 attention-KV 的一等 identity、dependency invalidation 和 tier-coherence 协议。

因此更精确的系统 thesis 是 **RepoKV-MVCC / repository-version-aware causal KV materialized views**。

2026-07-15 启动的更细粒度复核后来按用户要求收缩为 2025-01-01 至 2026-07-15。三个纯 2024 分段已停止并排除，七个保留分段全部完成，共核查 105 篇主候选，A/B/C/D=`0/23/67/15`。最终仍未发现 repository/source version 直接成为普通 attention-KV identity、dependency invalidation、branch/worktree isolation 和 physical-tier coherence domain 的公开系统；完整报告见 `research/CODE_VERSION_KV_10_AGENT_PRIOR_ART_REPORT.md`。

## Novelty 调研后的关键修正

- CodeComp `2604.10235` 已用 Joern CPG（AST/CFG/PDG）直接控制 repository-level KV budget、protected spans 和 pruning。
- FCGraft `2606.13097` 已把函数作为 KV object，支持 function-ID 索引、stitching、局部 patch、成功后更新和 GPU/DRAM residency。
- MEPIC `2512.16822` 与 MiniPIC `2606.13126` 已覆盖 code chunk/file span、canonical pages、position-independent reuse 和 memory tier primitive。
- 因此不能再主张“首个 AST-aware、function-level 或 code-specific hierarchical KV cache”。
- 原始组合 novelty 约 `2/5`；若实现 version consistency、dependency invalidation、calibrated reconstruction 和 artifact-level cache planning，保守上限约 `3.3–3.6/5`。
- prefix 或 role context 改变会使后续 hidden states 变化，suffix 的 K/V 都可能变化；RoPE relocation 只修位置。
- 全库应建立完整 logical index，但 physical KV 必须以 hotset 为主、按需惰性物化。

## 已验证资料与环境

| 项目 | 结果 |
| --- | --- |
| 历史工作区 | `/home/chris/Workspaces/kvcache-research` |
| SGLang prototype | `/home/chris/Workspaces/kvcache-research/kvflow-sglang` |
| 同步分支 | `feature/workflow-priority` |
| 同步提交 | `5bb9afc9234aa9caa9df51e87f119e5bfaf186de`，本地与 `ccdd2023/sglang` 远程一致 |
| 本机兼容版本 | `/home/chris/Workspaces/kvcache-research/sglang-running`，本地分支 `fix/qwen3-0.6b-docker-sm75` |
| 本机硬件 | NVIDIA RTX 2080 SUPER，SM75，8GB VRAM |
| Docker | `lmsysorg/sglang:dev` 镜像存在；调查时没有运行中的 SGLang 容器 |
| 远程 prototype 仓库 | `https://github.com/ccdd2023/sglang` |

## 实验基础设施：Vast.ai RTX PRO 6000

完整评估见 `research/VASTAI_RTX_PRO_6000_WORKFLOW.md`。

本项目采用混合工作流：

- 本地负责 Git、文档、metadata/unit tests、trace 分析、实验编排和结果持久化。
- Vast.ai RTX PRO 6000 S 负责短时 7B/8B、长上下文、KV pressure、HiCache、H2D 和 workflow benchmark。
- 最终硬件 claim 仍需少量 H100 calibration，不能把 SM120/GDDR7 结果直接等同于 H100 SM90/HBM。

兼容性审计确认：

- 本机 `lmsysorg/sglang:dev` 是 CUDA 12.9.1、PyTorch 2.9.1+cu129，并包含 `sm_120`/`compute_120`。
- `sglang-running` 源码有 RTX PRO 6000 特定 kernel 路径。
- 现有 host-side `docker run` 脚本不能在 Vast standard Docker instance 内嵌套执行，必须转换为 template/on-start/entrypoint。
- 当前 DeepEP build arch list 不含 SM120，dense single-GPU 主线先行。

正式实验必须记录 Git SHA、Docker digest、model revision、GPU/CPU/RAM、PCIe、disk、network、offer 和 machine ID，以隔离 Vast marketplace 异构性。

`sglang-running` 中已有 SM75 patch、Docker build/run 脚本和 Qwen3 运行说明，但其本地分支当前未在远程仓库发现同名引用。另一个本地 `feature/workflow-prefetch` 分支也未发现远程同名引用。

## 现有 AgentTemplateKV 分支审查结论

完整审查见 `research/YU_GUOFAN_BRANCH_REVIEW.md`。

最近两个月的研究线为：

```text
para_temp
-> feature/context-aware-kv-reuse
-> agenttemplatekv-eurosys-2026-06
-> phase-2.7-prerot
-> fix/placeholder-pool-activation
```

该工作线积累了大量 benchmark、telemetry、AST chunk、HKVD、CPU host pool、selective recompute 和负结果，但不能作为 KVCOMM 的现成复刻。

### 与 KVCOMM 的核心差异

当前分支没有实现：

- 无外部 context 的 placeholder base KV；
- placeholder `ΔK/ΔV`；
- neighboring-prefix `ΔK/ΔV`；
- multi-anchor soft interpolation；
- KVCOMM 的 length/embedding/entropy shareability gate；
- dense fallback 后的 online base/offset anchor update。

实际实现主要是某次真实 context KV 的 copy、Key position rotation、单邻居 k-NN、AST chunk 和固定比例 partial recompute。因此，最新论文稿已经正确地把 KVCOMM 写成 prior work / inspiration，而不是本文贡献。

### 安全路径必须分层看待

- L2 whole-slot exact path 在 copy 前比较当前 span token IDs 与 stored token IDs，具有真实 token equality guard。
- C2 AST chunk path 只使用 whitespace-normalized 前 240 字符的 signature，再比较 byte range；它不比较完整 token/content，不能满足全局 exact-content invariant。
- AST chunker 对 Unicode 使用 character count 计算“byte offset”，离线 compiler 又假设 tokenizer 对 `preamble + "\n" + text` 可加，均会造成 source/token/KV 错位。

### 实现基线决策

本项目不从 `fix/placeholder-pool-activation` 继续开发，而是：

1. 从接近 upstream 的干净分支或 `feature/workflow-priority` 开始；
2. 选择性移植 metadata plumbing、benchmark、AST/HKVD、RoPE helper；
3. 重写 KVCOMM core、pool lifecycle、offline writer/loader、context gate 和 selective recompute；
4. KVCOMM correctness 通过后再接 CPU tier 和 AST index。

## 论文文件

| 论文 | arXiv | 本地文件 | SHA-256 |
| --- | --- | --- | --- |
| KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows | [`2507.07400`](https://arxiv.org/abs/2507.07400) | `research/papers/KVFlow-2507.07400.pdf` | `2dc3180f91c34df920a7547f14fe92988527a76a277d2d17ddf6b558205d2f89` |
| KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems | [`2510.12872`](https://arxiv.org/abs/2510.12872) | `research/papers/KVCOMM-2510.12872.pdf` | `3acffcc4468b17cf5193a64ec998c6517af42acb247fc230171da92f0f9eb8df` |

两篇论文均已通过 alphaXiv 定位、收藏、读取并下载。KVCOMM 另有一篇同名论文 [`2510.03346`](https://arxiv.org/abs/2510.03346)，其主题是跨模型选择性共享非连续 Transformer layer 的 KV，不是本项目所需的跨 context cache 修正方案。

## KVFlow 的准确作用

KVFlow 解决的是 **cache management 和 scheduling**，不是跨不同上下文改变 KV 表示。

### 核心机制

1. 将 agent workflow 表示为 Agent Step Graph。
2. 为每个 agent invocation 计算 `steps-to-execution`。
3. 将未来执行距离传播到 radix-tree 的 KV node。
4. 多 agent 共享节点取最紧急的最小 step。
5. 动态 suffix 最先淘汰，较远未来才使用的固定 prefix 优先淘汰。
6. 被淘汰的固定 prefix KV 保存在 CPU。
7. 根据下一步 agent 提前执行 CPU→GPU prefetch。
8. cache node 具有 GPU、CPU、loading、offloading 四种状态；scheduler 跳过仍在 loading 的请求。

KVFlow 只复用逐 token 完全一致的 prefix。它不解决“相同代码段出现在不同 role prompt 之后”导致 KV 数值变化的问题。

### 历史实现结果

历史 SGLang 工作已完成：

- priority eviction；
- HiCache CPU backup/load-back；
- Qwen3 小模型与 AWQ 模型 benchmark；
- workflow-aware prefetch 实验；
- SM75 Docker 运行兼容。

既有实验的重要结论是：

- cache pressure 越大，workflow-aware priority 越有价值；
- working set 能完全放入 cache 时，priority 的额外收益很小；
- 单一 sequential cycle 中，强制 prefetch 可能引起 eviction churn，反而降低性能；
- 本地实现中无 prefetch 的 Priority + HiCache 在部分配置下最好；
- prefetch 更可能在 concurrent workflows、有 GPU 空闲空间或可被其他 ready request 隐藏传输延迟时获益。

当前 arXiv 版本的 KVFlow 论文明确包含 GPU-only SGLang 和 SGLang w/ HiCache 两个 baseline；历史笔记中“论文缺少 LRU + HiCache baseline”的判断与当前论文文本不一致，应以当前论文为准。

## KVCOMM 的准确作用

本项目相关的是 arXiv `2510.12872`。它解决的是：

> 同一 shared text 位于不同 agent role prefix、不同上游输出和不同绝对位置之后时，传统 exact-prefix cache 无法命中，如何近似恢复目标 context 中的 KV。

### Prompt 模型

每个 agent prompt 被划分为固定 prefix 与动态 placeholder：

```text
[fixed prefix 0]
[placeholder 1]
[fixed prefix 1]
[placeholder 2]
[fixed prefix 2]
```

placeholder 可以是：

- 用户问题；
- tool execution result；
- 其他 agent response；
- 历史轮次内容。

### Base KV + context offset

KVCOMM 将目标 context 中的 KV 理解为：

```text
target KV = canonical/base KV + context-dependent offset
```

它为 placeholder 维护 anchor pool。每个 anchor 记录：

- base placeholder KV；
- 某个 agent/context 下 placeholder 的真实 KV offset；
- placeholder 后相邻固定 prefix 的 KV offset；
- embedding、长度、使用频率等匹配信息。

命中时执行：

1. 根据长度和 embedding/KV distance 找到 anchors。
2. 对权重分布做 entropy gating。
3. 对 Key 进行 RoPE de-rotation/re-rotation，修正绝对位置。
4. 对多个 anchor 的 `ΔK/ΔV` 做 soft interpolation。
5. 同时修正 placeholder 和其后固定 prefix。
6. 按 prompt 顺序拼接 cache，直接进入 decoding。

无法安全复用时，系统执行 dense prefill，并把真实 offset 加入 anchor pool。

### “可变编码”的精确定义

KVCOMM 没有提出名为 Variable Encoding 的编码格式，也没有实现量化、异构精度或通用压缩。用户所说的“可变编码”在当前方案中应理解为：

- canonical/base KV；
- 随 role、prefix 和 position 变化的 `ΔKV`；
- Key 的动态 RoPE 重定位；
- 根据 anchor 置信度选择不同的重建路径。

论文附录观察到 anchor delta 较稀疏，并把压缩列为未来工作。因此，将 delta 做稀疏编码、量化或多精度存储属于本项目可研究的扩展，不是 KVCOMM 已实现的贡献。

### 与 CPU Memory 的边界

KVCOMM 官方实现基于 Hugging Face `DynamicCache`，核心 fast path 偏 GPU-resident。论文测过将 anchors offload 到 CPU 的高开销，但没有实现 SGLang 式的高效 CPU/GPU hierarchical cache。

所以：

- KVCOMM 提供“cache 如何变换和复用”；
- KVFlow/HiCache 提供“cache 在哪里、何时搬运和淘汰”。

“根据 AST index 从 CPU 加载经 context 修正的 cache”是两者组合后的本项目创新。

## AST 与代码结构的历史研究结论

历史离线实验已经验证：

- embedding distance 与 KV distance 的相关性存在，但不强；
- AST 结构距离与 embedding distance 近乎正交；
- 同时控制 token 位置和 embedding 后，AST 结构距离对 Key/Value distance 仍有约 `+0.17 ~ +0.24` 的 partial Spearman，两个小模型的置信区间均不含 0；
- raw Key distance 会被 RoPE 位置严重污染；
- AST 信号适合 candidate gating、anchor ranking、temperature 调整和分段边界，不适合替代 embedding 判据。

因此 AST 在本项目中至少承担四种职责：

1. 将 Codebase 切分为稳定 artifact：module、class、function、statement、basic block。
2. 建立 symbol、dependency、call graph、test relation 和 source span 索引。
3. 为 cache retrieval 与 anchor matching 提供结构先验。
4. 在代码修改后按 content hash 与依赖关系进行局部失效和重算。

## 对“预计算整个 Codebase”的准确理解

不能把整个 Codebase 当成一条巨大连续 prompt 生成单一 KV Cache，原因包括：

- 超出模型 context window；
- KV 是 causal、position-dependent、context-dependent 的；
- 任意拼接独立 KV chunk 会缺失跨 chunk attention；
- 代码持续变化，需要局部失效。

可实现的定义是：

> 对整个 Codebase 的所有可索引 artifact，在 canonical context 下分别预计算 base KV，并维护足以在不同 role/context 中重建目标 KV 的 metadata 与 anchor offsets。

每个 artifact 至少需要：

```text
repo_commit
content_hash
file_path
language
AST stable ID / node kind
qualified symbol
source span
token span
model/tokenizer/template fingerprint
base position
KV page IDs
storage tier
dtype/layout/compression mode
anchor IDs and confidence
last access / reuse frequency
workflow priority
```

### 推荐的 artifact 切分

- repository/package/module/file/class 作为**逻辑容器和检索视图**；
- function/method、module-level declaration、class field/init block 作为首选的**非重叠 canonical physical unit**；
- 超长函数才继续按 statement/basic-block 子段切分，并保存父子关系与原始顺序；
- imports、常量、宏、全局初始化和配置文件单独形成 module preamble/config artifact；
- file/class view 只引用下层 canonical units，不重复存储嵌套函数的完整 KV。

这意味着“预计算整个 Codebase”是覆盖全库 logical artifacts，而不是把仓库拼成一个连续 prompt，也不是为每个层级重复保存完整 BF16 KV。

### Dependency analysis 的准确作用

依赖图由现有工具组合生成，不要求从零发明 parser：

- tree-sitter/LSP/compiler symbol graph；
- import、call、inheritance、type/ABI、def-use 和 build graph；
- test-to-symbol mapping；
- 可选的 runtime call/test trace。

更新流程是：

```text
Git/token/AST diff
-> changed canonical artifacts
-> content/interface/dependency hash comparison
-> reverse dependency cone
-> affected context signatures
-> KEEP / DIRTY / INVALIDATE / VERIFY / REMATERIALIZE
```

依赖分析只是 conservative invalidation prior，不能证明 attention independence。最终 exact reuse 仍要求完整 token、causal-context 和 model fingerprint 相同；近似复用必须经过 probe/gate，失败则 dense fallback。

已有研究已经覆盖 AST/CPG、函数级对象、Git/Merkle 增量索引和传统依赖图。候选 novelty 不在“发明依赖分析”，而在将 source-version/dependency events 直接连接到 attention-KV identity、coherence、physical-page lifecycle 和 stale audit。

### Git 不是唯一索引

系统同时维护：

| 索引 | 作用 |
| --- | --- |
| path/symbol/AST | 定位函数、类和 module preamble |
| embedding/query relevance | 判断当前任务需要哪些 artifacts |
| Git snapshot/patch epoch | 确定 artifacts 属于哪个源码版本 |
| content/token hash | 判断内容是否完全相同 |
| dependency graph | 传播变更风险和失效 |
| causal-context signature | 判断某份 KV 是否可以 exact reuse |
| physical-page index | 定位 KV 当前在 GPU、CPU 或 SSD 的位置 |

因此准确总结是：

```text
结构负责切块
检索负责选块
Git 负责定版本
依赖负责传播失效
context fingerprint 负责 exactness
physical index 负责加载
```

artifact 也不是越小越好。通常以数百到数千 token 的 function/method 或 module preamble 为单位；全库 logical index 可以很大，但只为 hot artifacts 和常用 composed context 稀疏物化 physical KV。独立预计算的小块不能任意拼接成 exact KV，新顺序或新前缀必须经过 KVCOMM reconstruction、selective recompute 或 dense fallback。

### Physical KV 与仓库 bootstrap

Attention KV 是每一层为历史 token 产生的真实 Key/Value tensor：

```text
K[layer, token, kv_head, head_dim]
V[layer, token, kv_head, head_dim]
```

它不是源码文本、embedding 或 Git object。每个 physical object 以 token pages 保存这些 tensors，并附带 artifact version、context signature、position basis、model fingerprint、dtype/layout 和 tier。

系统不从仓库第一个 commit 开始。推荐流程是：

1. 选择部署或实验使用的 seed snapshot，例如 `main@<fixed SHA>`。
2. 对该 snapshot 建立全库 logical artifact catalog。
3. 只为 hot/relevant artifacts 生成 canonical base KV，默认落在 CPU。
4. 新 commit、branch 或 dirty patch 到来后只处理 diff。
5. unchanged artifact 在新 snapshot 中 alias 旧 physical pages；changed artifact 重算；interface/dependency change 触发保守验证或失效。
6. old pages 在仍有 session/branch 引用时保留，无引用后 GC。

历史 commit 只用于 workload replay、anchor 收集或论文评测，不要求从 repository genesis 逐个预计算。

### 函数粒度需要区分四种“单位”

| 层次 | 默认单位 | 含义 |
| --- | --- | --- |
| logical artifact | function/method 为主 | 版本、依赖、检索和 invalidation 的主要单位 |
| exact reuse | 完整 causal prefix/context signature | 函数文本相同但前置上下文不同，不能自动 exact |
| KVCOMM reconstruction | placeholder span | 可以映射为函数，也可以是 class/module chunk；论文还处理后续 fixed-prefix offset 和 whole-agent gate |
| physical storage | token pages | 一个函数通常跨多个 K/V pages，page 才是实际搬运和分配单位 |

因此函数是推荐的第一版逻辑粒度，而不是所有路径的统一物理或 exact-match 粒度。超长函数需要拆分，极短且强耦合的 declarations 可以合并；module preamble、class context、宏/配置和 top-level code 仍需独立 artifacts。

### Source dependency graph 与 prompt causal graph

系统需要两张不同的图：

1. **Source dependency graph**：描述 import、call、type、inheritance、def-use、build、test 和 observed runtime edges，用于 retrieval、impact analysis 和 conservative dirty cone。
2. **Prompt causal graph**：记录某个已编译 prompt 中真实的 segment 顺序，以及每个 KV span 前面实际出现了哪些 artifact versions；用于 exact invalidation。

静态源码依赖不等于 attention 因果依赖。某个 callee 即使语义上依赖于 caller，只要它没有出现在某份 KV 的 causal prefix 中，就不应仅因静态边而强制失效；反之，只要旧 artifact 文本实际出现在 causal prefix 中，即使静态分析没有边，对应 context variant 也必须失效。

### 三类存储，而不是“每个函数两份完整 KV”

```text
1. Exact Radix/Bundle Cache
   - key: 完整连续 prompt token prefix + fingerprint
   - value: 可跨多个函数、模块和日志的 exact KV pages

2. Canonical Artifact Base Store
   - key: artifact version + canonical context signature
   - value: function/method/preamble 等 span 的 base KV
   - 用途: KVCOMM reconstruction，不代表任意 prompt 下 exact

3. Context Variant / Anchor Store
   - key: artifact + role/context/length/position signature
   - value: bounded residual/offset/anchor metadata，优先避免保存完整重复 KV
```

部分 cold artifacts 只有 logical metadata，没有 physical KV。Exact bundle 由真实请求惰性产生；canonical base 可以离线或首次访问时产生；anchor/residual 数量有上限。

### Prompt Compiler

Prompt Compiler 输入 task、workflow stage、repository snapshot、retrieval candidates、dependency graph 和 token budget，输出确定性的 `PromptPlan`：

```text
PromptPlan {
  ordered_segments[]
  source_versions[]
  rendered_token_ids[]
  position_ranges[]
  causal_context_signatures[]
  exact/base/reconstruct/dense decisions[]
}
```

“决定代码顺序”是指决定哪些 artifact 先进入 Transformer causal sequence。推荐约束是：

- static system/stage template 最前；
- definitions、types、callees 和 stable module context 通常先于 consumers；
- strongly connected components 合并后稳定排序；
- 高复用、低变动内容尽量靠前；
- patch、test failure、stack trace 和用户本轮动态信息靠后。

这样 Debugger 可以 exact reuse 前面的大段稳定 code bundle，只在尾部追加新的错误日志。顺序既影响模型质量，也直接决定最长 exact-prefix hit。

### Debugger Bundle 示例

一次 SGLang cache bug 调试可能编译出：

```text
system/debugger template
Req / memory-pool types
RadixCache.match_prefix
RadixCache.insert
RadixCache.cache_finished_req
release_kv_cache
相关 tests
当前 patch
最新 stack trace
用户问题
```

前七段可以形成 5k–20k token 的 exact bundle。下一轮只改变 stack trace 时，可以复用整个稳定 prefix，而不是只复用一个函数。若中间的 `cache_finished_req` 改变，则普通 Radix exact 只能复用变化点之前的 prefix；变化点之后可 dense 重算，或由 KVCOMM 对一个或多个 placeholder spans 做近似重建。

### 两个 Python 文件的最小 Git/workflow 示例

```text
calc.py
  add()
  divide()
  average() -> divide()

report.py
  total()   -> add()
  ratio()   -> divide()
  summary() -> total(), average()
```

在 `commit C0`：

- 六个函数各自有 logical artifact version；
- canonical base KV 可以按函数保存；
- Architect 首次使用六函数 prompt 后，产生一个跨六函数的 exact bundle。

Coder 从 `C0` 建立 dirty worktree `W1=C0+patch1`，只修改 `divide()`。Git 的作用是产生 snapshot identity 与 diff：

```text
add      -> ALIAS C0 base
divide   -> DIRTY / REMATERIALIZE
average  -> source unchanged; semantic dependent; context variants VERIFY
total    -> ALIAS
ratio    -> source unchanged; semantic dependent; context variants VERIFY
summary  -> source unchanged; transitive dependent; context variants VERIFY
```

如果某 exact bundle 的 token prefix 中实际包含旧 `divide()`，变化点之后的 KV 都不能继续宣称 exact。未变函数的 canonical base 仍可跨版本 alias；它们在新 Coder/Debugger context 中通过 KVCOMM reconstruction 或 dense prefill 得到新 variants。

Coder commit 后形成 `C1`：

- `C0` pages 仍供旧 Architect session 使用；
- `C1` 对 unchanged artifacts alias 旧 base pages；
- `divide@C1` 使用新 pages；
- worktree patch visibility 变成 committed snapshot visibility。

Debugger 绑定 `C1`，编译六函数 code bundle，并在尾部加入 patch、test failure 和 stack trace。第一轮产生 `ExactBundle(Debugger,C1,plan_hash)`；后续只变 stack trace 时复用整个六函数 prefix。Debugger 失败回到 Coder 时，workflow priority 保留该 dependency cone 的 base/variant pages，并预取下一阶段最可能使用的对象。

这个例子体现的候选 novelty 不是 Git diff 本身，而是：

```text
Git snapshot / dirty patch
-> artifact version visibility
-> cross-version physical-page alias
-> prompt causal invalidation
-> KVCOMM/dense rematerialization
-> branch/session isolation
-> workflow-aware tier priority
```

### 跨 Architect/Coder/Debugger 的共享边界

不同 System Prompt 位于代码 token 之前，会改变代码 token 在所有层的 hidden state 和 K/V。因此默认情况下：

```text
ExactBundle(Architect, snapshot, plan)
ExactBundle(Coder, snapshot, plan)
ExactBundle(Debugger, snapshot, plan)
```

是三个不同的 exact variants，不能直接 raw-copy 互换。

但不意味着三套数据完全独立。推荐共享结构是：

```text
CanonicalBaseKV(artifact version)       # 跨角色共享
ContextOffset/Anchor(Architect)         # 角色/上下文 variant
ContextOffset/Anchor(Coder)
ContextOffset/Anchor(Debugger)
Hot ExactBundle(stage, snapshot, plan)  # 惰性生成、stage-specific
```

KVCOMM 的目标正是利用 shared base 与 context-dependent offsets，将同一代码 artifact 从一个 role/prefix 重建到另一个 role/prefix。重建受 length/entropy gate 约束，失败则 dense prefill；Architect cache 不能保证一定被 Coder/Debugger 接受。

faithful KVCOMM 的 offsets 可能接近完整 K/V 大小，不能默认视为很小；必须限制 anchor 数量，并在后续研究中验证低秩、稀疏或量化 residual 是否可行。

如果三个角色采用完全相同的前导 system prefix，并把 stage directive 放到后面，则 role 分叉之前仍可 exact reuse；但这属于 prompt-template co-design，需要单独验证 agent quality，不能作为默认假设。

### Canonical Base KV 的精确定义

`canonical` 不是新的 tensor 类型、压缩格式或符号表示。Canonical Base KV 是普通模型 forward/prefill 在固定 reference condition 下生成的真实 K/V tensors：

```text
CanonicalBaseKV =
Model(
  fixed model/tokenizer/template,
  fixed canonical prompt,
  fixed artifact tokens,
  fixed position basis
).K/V
```

所有 KV 都是计算出来的；`canonical` 只表示这次计算的 prompt/context/position 被选作公共参考系。

它只在 canonical prompt 本身下是 exact。Architect、Coder 或 Debugger 的实际前置上下文不同后，canonical base 不能直接当作 target exact KV；KVCOMM 使用 anchors/context-dependent `ΔK/ΔV` 和 RoPE relocation 估计目标 K/V，gate 失败则 dense。

记录至少包含：

```text
artifact version
canonical prompt/token hash
token IDs
base positions
model/tokenizer/template/RoPE fingerprint
actual K tensors
actual V tensors
```

所以更准确的说法是：**Canonical Base KV = 普通 KV tensor + 被固定并完整记录的 reference provenance。**

KVCOMM 在历史 SGLang 上的完整复现可行性见 `research/KVCOMM_SGLANG_FEASIBILITY_REPORT.md`。功能性忠实复现可行，推荐先从 clean fixed SHA 建立 GPU-only、TP=1 版本；token/position、full K/V offset、RoPE、approximate provenance 和 lifecycle 是主要 P0，HiCache/KVFlow 与 codebase registry 均应后置。

## 统一系统架构

### 1. 离线 Codebase KV Compiler

1. 固定 repository snapshot。
2. 使用 tree-sitter/编译器解析 AST、symbol、dependency 和测试关系。
3. 按稳定 artifact 边界切分代码。
4. tokenizer 对齐 source span 与 token span。
5. 在 canonical prompt/context 中预计算 base KV。
6. 建立逻辑索引与物理 cache 索引。
7. 将大部分 cache 持久化到 CPU Memory；GPU 只保留 hot working set。

### 2. 两类索引

**逻辑索引**

- file/symbol/AST；
- call graph、dependency、test mapping；
- retrieval relevance；
- artifact version 与 invalidation。

**物理索引**

- GPU/CPU cache page；
- exact/base/approximate 类型；
- position 与 RoPE metadata；
- anchor delta；
- dtype、压缩方式、传输成本；
- 当前状态与 priority。

### 3. 在线请求路径

```text
用户任务
→ 当前 workflow stage
→ 检索相关 AST/artifact segments
→ exact RadixAttention hit 优先
→ exact miss 时尝试 KVCOMM base+offset 重建
→ 根据 priority 从 CPU load 到 GPU
→ 低置信或不兼容 segment 执行 dense prefill
→ 记录真实 offset，更新 anchor/index
→ 进入普通 decode
```

正确的优先顺序是：

1. exact cache；
2. verified cross-context reconstructed cache；
3. dense fallback。

## 固定三阶段 Workflow

固定流程：

```text
Architect -> Coder -> Debugger
```

### Architect

- 读取 repo map、架构文档、接口和跨模块依赖。
- 需要较广但不一定很深的 Codebase coverage。
- 运行期间可准备 Coder 所需的目标文件、依赖和设计结果 cache。

### Coder

- 读取 Architect plan、目标文件、symbol dependency、相关测试。
- 需要高精度代码 cache，错误复用风险高。
- 运行期间可准备 Debugger 所需的修改文件、测试、错误处理路径和 tool schema。

### Debugger

- 读取 patch、编译结果、test logs、stack trace、调用链。
- 成功时 workflow 结束；失败时可能回到 Coder。
- `Debugger -> Coder` 应作为条件分支进入 Agent Step Graph。

同一 code artifact 会出现在三个不同 role prefix 之后，这正是 KVCOMM 相比 exact-prefix cache 的价值所在。

## Priority 设计

KVFlow 原始 priority 只使用 `steps-to-execution`。本项目需要组合：

```text
priority = f(
  steps_to_execution,
  retrieval_relevance,
  AST/dependency relevance,
  expected reuse frequency,
  recompute cost,
  CPU→GPU transfer cost,
  anchor confidence,
  correctness risk
)
```

原则：

- 当前和下一阶段所需 artifact 最高保留优先级。
- 高重算成本、高复用概率的 base KV 优先保留。
- 低置信 approximate cache 不应挤占关键 exact cache。
- prefetch 不得为了加载较低价值 cache 强制驱逐更紧急的 cache。
- 单 sequential workflow 默认保守 prefetch；有空闲 GPU slot、明确下一阶段或并发 workflow 时再积极预取。

## 论文复刻与项目创新的边界

| 能力 | 来源 |
| --- | --- |
| Agent Step Graph、steps-to-execution、radix-node priority | KVFlow |
| CPU backup、下一 agent prefetch、status-aware scheduling | KVFlow |
| base KV + context offset | KVCOMM |
| RoPE Key 重定位 | KVCOMM |
| anchor matching、soft interpolation、dense fallback | KVCOMM |
| SGLang 上的 KVCOMM 实现 | 本项目 |
| AST stable-ID 与 Codebase artifact index | 本项目实现机制；CodeComp/FCGraft 已否定其单独 novelty |
| 全 Codebase logical artifact index + lazy physical KV | 本项目候选系统贡献，必须与版本一致性联合证明 |
| anchor/base KV 的 HiCache CPU tier | 本项目集成；通用 tier 已有 LMCache/MEPIC/MiniPIC |
| AST relevance + workflow step 联合 priority | 本项目集成；必须对比 KVFlow，不能单独主张 novelty |
| 可变精度、稀疏或量化 delta 编码 | 本项目候选扩展 |
| `Architect -> Coder -> Debugger` 专用调度 | 本项目 |
| build/test correctness 反馈驱动的 cache guardrail | 本项目 |
| source/dependency incremental invalidation | 本项目最强候选系统贡献 |
| versioned logical-artifact-to-physical-page consistency | 本项目最强候选系统贡献 |
| calibrated cross-role reconstruction + dense fallback | 本项目最强候选 correctness 贡献 |

## 建议的 Prototype 路线

### 前置阶段：可行性测量

- 采集真实 workflow 中跨阶段、跨 commit 的 artifact reuse。
- 测量同一 artifact 在不同 role、query、顺序和位置下的 KV variance。
- 建立 H2D load 与 dense recompute 的 break-even 曲线。
- 回放真实 commit/patch，测量 source/dependency invalidation ratio。
- 若 reuse、load advantage 或可标定 reconstruction 不成立，应缩小论文 thesis，而不是继续扩大 SGLang 改动。

### 阶段 0：固定基线

- 以远程同步的 `feature/workflow-priority` 为基础。
- 不使用 `fix/placeholder-pool-activation` 作为核心实现基线；该分支仅作为实验档案和 helper donor。
- 使用 `sglang-running` 的 SM75 Docker patch 在本机运行。
- 固定小模型、prompt、workflow 和 correctness baseline。

### 阶段 1：在 SGLang 复刻 KVCOMM

- 先不做 AST，也不做复杂 CPU 分层。
- 实现 prompt segmentation、base KV、anchor pool、RoPE relocation、offset interpolation 和 dense fallback。
- 对比 dense、exact-prefix 与 KVCOMM。

### 阶段 2：接入 HiCache

- 将 base KV 和 anchor delta 映射为 SGLang cache pages。
- 支持 GPU L1 / CPU L2。
- 增加 transfer、load state 和 cache type metadata。

### 阶段 3：Codebase 预计算与 AST Index

- 建立离线 artifact compiler。
- 先支持一种语言和 function/class 粒度。
- 实现 content-hash invalidation 和 source/token mapping。

### 阶段 4：三阶段 Workflow 集成

- Architect、Coder、Debugger 使用不同 role prefix。
- 同一 artifact 跨 role 做 KVCOMM 重建。
- 根据 step 和 artifact relevance 管理 CPU→GPU load。

### 阶段 5：评测

- cache hit/reuse/fallback rate；
- prefill tokens、TTFT、H2D bytes、overlap ratio；
- GPU/CPU memory；
- anchor approximation error、next-token KL；
- patch correctness、编译成功率、unit-test pass rate；
- 冷启动、代码修改、错误 anchor 和 cache pressure 场景。

## 主要风险

1. 独立预计算 chunk 缺失跨 chunk attention，不能仅靠物理拼接保证正确。
2. approximate KV 对代码语法和标识符错误非常敏感。
3. anchor pool 与完整 Codebase KV 的 CPU 内存成本仍可能很高。
4. 模型、tokenizer、RoPE、chat template 或 repository revision 变化都必须使 cache 失效。
5. 单工作流 prefetch 可能引发 cache churn。
6. 本机 8GB VRAM 只能做小模型机制复现，不能复现 KVCOMM 的 H100 主实验指标。
7. 最终系统必须把 exact 与 approximate cache 明确区分，并提供 dense fallback、抽样审计和 correctness guardrail。
8. 截断内容 hash、byte-range-only gate 或 client-provided signature 不能单独作为 exact reuse 证明。
9. cache pool 必须拥有或显式 pin 其 KV pages；不能保存可能被 radix allocator 回收的裸 slot 引用。
10. 离线 KV artifact 必须绑定 model、revision、tokenizer、chat template、RoPE、dtype/layout、canonical preamble 和 repository commit。
11. 第三方算法的近似实现必须标为 inspired；固定 leading-FRAC 不是 CacheBlend，head-only Key rotation 不是 EPIC。
12. 论文表格与图必须能从已提交的 compact source artifact 重生成。
