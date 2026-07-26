# AST-Indexed Repository KV Cache：Prior Art、Novelty 与研究路线

最后更新：2026-07-12

## 1. 执行摘要

本轮由一个专项 arXiv 调研代理和四个独立模型评估代理完成。结论不是“没有人做过 AST-aware KV”，而是：

1. **已有直接先例。**
   - [CodeComp `2604.10235`](https://arxiv.org/abs/2604.10235) 已用 Code Property Graph（CPG，包含 AST/CFG/PDG）直接决定 repository-level 代码 span 的 KV 预算、保护和淘汰。
   - [Functional Cache Grafting / FCGraft `2606.13097`](https://arxiv.org/abs/2606.13097) 已把函数作为 KV 对象，用函数 ID 索引文本与 KV，并管理检索、拼接、修补、更新和 GPU/DRAM 驻留。
2. **原始方案 novelty 偏弱。** “AST 分段/索引 + CPU/GPU tier + workflow priority + KVCOMM/CacheBlend 式重建”容易被评价为 CodeComp/FCGraft + KVFlow + KVCOMM + MEPIC/MiniPIC 的系统组合，当前约为 **2/5，weak reject**。
3. **仍存在可辩护空白。** 本次检索尚未发现一个系统同时面向持续演化的软件仓库，实现：
   - 多粒度 AST artifact compiler；
   - logical artifact 到 physical paged-KV 的版本化映射；
   - 源码和 dependency change 驱动的增量失效；
   - 同一 artifact 跨 role/prefix 的受控重建；
   - artifact 粒度的 CPU/GPU workflow scheduling；
   - stale/approximate cache 的 correctness contract。
4. **论文 thesis 必须更换。** 不再把“AST index”本身当核心创新，而应把代码 KV 定义为：

   > 带有源码版本、因果上下文、数值保真度和风险界限的 materialized view；由 cache-plan compiler 联合决定 exact reuse、context reconstruction、selective recompute、dense fallback 与 tier placement。

5. **最值得主打的机制：**
   - patch/dependency-cone incremental invalidation；
   - versioned logical-artifact-to-physical-page KV store；
   - exactness lattice、可标定 cross-role reconstruction 与 dense fallback；
   - causal cache-plan compiler；
   - 测试/trace 反馈驱动的 cache refresh；
   - 结构条件化 KV reconstruction，作为必须实测胜过现有工作的条件性算法方向。

建议论文方向为：

> **RepoMV: Causality-Aware, Versioned KV Materialized Views for Repository-Scale Coding Agents**

---

## 2. 调研与评估方法

### 2.1 专项文献调研

专项代理使用 GPT-5.6 Sol Max，对 arXiv/alphaXiv 全文、引用链和可验证官方代码进行检索，截止日期为 2026-07-12。检索重点不是标题是否包含 AST，而是程序结构是否直接参与：

- KV object identity；
- KV index 或 lookup；
- retention/eviction；
- CPU/GPU residency；
- source update 后的 invalidation/lifecycle。

### 2.2 四模型独立评估

| 代理 | 模型 | 主要视角 |
| --- | --- | --- |
| Sol | GPT-5.6 Sol Max | 系统 novelty、materialized-view abstraction、实验与 kill criteria |
| Opus 4.8 | Claude Opus 4.8 Max | 顶会审稿、causal correctness、最强拒稿理由 |
| Opus 4.6 | Claude Opus 4.6 Max | 数据模型、priority、状态机和 workflow contract |
| Gemini | Gemini 3.1 Pro 最高推理档 | 发散机制、AST/图、tier、feedback 与 workflow 优化 |

用户所写 “Observe 4.6” 在可用模型中按 Claude Opus 4.6 执行。

### 2.3 结果校正规则

最终报告以专项文献证据和 causal Transformer 基本约束为准。以下代理表述已被纠正：

- EPIC 正确编号为 [`2410.15332`](https://arxiv.org/abs/2410.15332)；LegoLink 是 training-free 的静态 leading-token recompute，不是 `2405.12119`，也不是必须重训。
- prefix 发生变化时，suffix hidden states 会变化，因此 suffix 的 **K 和 V 都可能变化**。
- AST-isomorphic position ID、独立 block position 或定制 attention mask 会改变模型语义，不能直接称为原模型下 exact。
- Coder generation KV 在 Debugger role/prefix 改变后通常不能直接嫁接；必须满足相同因果上下文，或执行 KVCOMM 式校正/重算。
- “AST-topological KV 完全空白”已被 CodeComp 和 FCGraft 否定。

---

## 3. Direct-prior-art verdict

### 3.1 最准确的结论

不能再声称：

- “首个 AST-aware KV cache”；
- “首个 function-level KV cache”；
- “首个 code-specific hierarchical KV cache”；
- “首个 workflow-aware coding-agent KV manager”；
- “首个预计算 code chunk KV 的系统”。

可以谨慎声称：

> 本次检索尚未发现一个 repository-scale serving system，把已有源码中的 file/class/function/AST-span 物化为独立可寻址 KV objects，并同时维护跨源码版本、物理页面、内存层级、role context 和 agent workflow 的一致性。

所有正式论文表述都应使用 “to our knowledge” 或“本次检索未发现”，不能使用绝对不存在。

### 3.2 A/B/C/D 分类

- **A 类：直接先例**  
  程序结构直接决定 KV object identity、index、retention 或 lifecycle。
- **B 类：强代码邻近工作**  
  明确把 code file/chunk 当作可复用 KV span，但没有 AST/程序结构索引。
- **C 类：通用 KV primitive**  
  modular/non-prefix reuse、reconstruction、tiering、agent scheduling。
- **D 类：代码结构与 agent memory，但不存 attention KV**。

### 3.3 A 类直接先例

#### CodeComp

- 论文：[`2604.10235`](https://arxiv.org/abs/2604.10235)
- 使用 Joern CPG 提取 call、control、return、assign、CFG/PDG 特征。
- 按 method/function 边界切块。
- 程序结构直接决定：
  - chunk structural score；
  - KV budget；
  - protected spans；
  - 哪些 token KV 不能被 attention-only pruning 删除。
- 场景是 repository fault localization 与 patch generation。
- 边界：
  - 属于单请求内 structure-aware KV compression；
  - 不做跨请求持久 KV object store；
  - 没有 CPU/GPU tier、artifact registry、源码版本失效或 cross-role reconstruction。

#### Functional Cache Grafting / FCGraft

- 论文：[`2606.13097`](https://arxiv.org/abs/2606.13097)
- 直接把函数作为 KV object，并用函数 ID 索引 interface、implementation、文本与 KV。
- 支持：
  - function retrieval；
  - cache stitching；
  - localized patching；
  - 成功执行后更新；
  - recency、frequency、co-occurrence、semantic score；
  - GPU/DRAM residency。
- 边界：
  - 场景是 embodied-agent Code-as-Policies；
  - 不是持续演化的软件仓库；
  - 没有 AST/CPG parser、class/file/span、多版本依赖失效、physical KV page catalog 或跨 role context reconstruction。

### 3.4 强 B 类先例

#### MEPIC

- 论文：[`2512.16822`](https://arxiv.org/abs/2512.16822)
- 面向 document/code chunks。
- 提供 canonical paged layout、完整 chunk hash、NoPE KV、运行时 RoPE、首 block request-specific recompute、chunk LRU/refcount、HBM 与 LMCache remote tier。
- 已非常接近 physical page mapping 和 persistent chunk residency。
- 缺少 AST artifact identity、源码依赖失效和 workflow semantics。

#### MiniPIC

- 论文：[`2606.13126`](https://arxiv.org/abs/2606.13126)
- 把 document/code file 定义为可复用 span。
- 保存 unrotated K，在 attention kernel 内按请求位置施加 RoPE。
- 原生兼容 CPU offload。
- span 由调用者定义，没有自动 AST compiler、版本一致性和依赖失效。

### 3.5 关键 C/D 类工作

| 工作 | 主要能力 | 对本项目的边界 |
| --- | --- | --- |
| [KVCOMM `2510.12872`](https://arxiv.org/abs/2510.12872) | base KV、context offset anchors、RoPE relocation、multi-anchor interpolation、dense fallback | 覆盖跨 role/prefix reconstruction；无代码结构和 artifact tier |
| [KVFlow `2507.07400`](https://arxiv.org/abs/2507.07400) | Agent Step Graph、steps-to-execution、CPU→GPU prefetch | 覆盖 workflow priority；单位是 agent exact-prefix node，不是代码 artifact |
| [CacheBlend `2405.16444`](https://arxiv.org/abs/2405.16444) | non-prefix chunk KV、high-deviation token selective recompute | 重建基线；无代码结构 |
| [EPIC `2410.15332`](https://arxiv.org/abs/2410.15332) | immutable chunk compile/link、leading-token recompute | modular chunk 基线；无 repository lifecycle |
| [KVLink `2502.16002`](https://arxiv.org/abs/2502.16002) | 独立文档 KV、link tokens、跨块关系恢复 | 需要训练；不做代码 artifact lifecycle |
| [Prompt Cache `2311.04934`](https://arxiv.org/abs/2311.04934) | PML module→KV、固定 position、GPU/CPU | 人工 schema，不是 AST/source index |
| [RAGCache `2404.12457`](https://arxiv.org/abs/2404.12457) | knowledge tree、GPU/host hierarchy、PGDSF | 树是文档顺序，不是 AST |
| [LMCache `2510.09665`](https://arxiv.org/abs/2510.09665) | CPU/SSD/remote KV tier 与控制 API | 可作物理 tier；无程序结构 |
| [MORI `2606.00866`](https://arxiv.org/abs/2606.00866) | coding-agent session 级 GPU/CPU residency | 粒度是完整 session，不是 file/function |
| [Code Isn’t Memory `2606.22417`](https://arxiv.org/abs/2606.22417) | whole-repo AST/graph index、Merkle diff、增量重建 | 只检索文本/图，不存 Transformer KV |
| [LocAgent `2503.09089`](https://arxiv.org/abs/2503.09089) | file/class/function 异构图索引 | 只做代码检索 |

其他已核查的强邻近工作还包括：

- [Irminsul `2605.05696`](https://arxiv.org/abs/2605.05696)：coding-agent prompt 的 content-defined chunking 与 arbitrary-position reuse，但不是程序结构。
- [Cache-Craft `2502.15734`](https://arxiv.org/abs/2502.15734)：RAG chunk repair、reuse score 与 eviction。
- [TurboRAG `2410.07590`](https://arxiv.org/abs/2410.07590)：离线 chunk KV compiler、独立 attention mask 与模型适配。
- [Leyline `2606.01065`](https://arxiv.org/abs/2606.01065)：agent cache edit directive 与 prefix-trimmed re-prefill。
- [SeKV `2606.31145`](https://arxiv.org/abs/2606.31145)：semantic span、CPU low-rank KV 与动态 zoom-in。
- [Cartridges at Scale `2606.04557`](https://arxiv.org/abs/2606.04557)：大规模持久 KV cartridges，但不是普通 source-span prefill KV。

---

## 4. Closest-prior-art matrix

| 系统 | 程序结构直接控制 KV | 持久 KV object key | 跨请求复用 | 跨 role/context 校正 | CPU/GPU tier | 源码/依赖增量失效 | workflow priority |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CodeComp | ✓ CPG retention | — | — | — | — | — | — |
| FCGraft | ✓ function object | ✓ | ✓ | — | ✓ | △ 成功后局部更新 | △ runtime utility |
| MEPIC | △ code chunk | ✓ hash/page | ✓ | △ 首块重算 | ✓ | — | — |
| MiniPIC | △ code-file span | ✓ | ✓ | △ position-independent primitive | ✓ | — | — |
| KVCOMM | — | ✓ anchor | ✓ | ✓ | △ | — | — |
| KVFlow | — | △ exact-prefix node | ✓ | — | ✓ | — | ✓ |
| LMCache | — | ✓ chunk | ✓ | △ 插件 | ✓ | — | — |
| Code Isn’t Memory | ✓ AST/graph | — KV | — KV | — | — KV | ✓ | — |
| 目标系统 | ✓ AST/CPG artifact | ✓ versioned page object | ✓ | ✓ | ✓ | ✓ | ✓ artifact-level |

“没有单篇覆盖所有列”本身不能证明 novelty。必须证明这些能力之间存在不可被独立组件表达的联合机制，例如：

- 依赖失效如何约束 reconstruction；
- stage fidelity 如何进入 risk budget；
- tier placement 如何同时考虑 load/recompute cost 与 stale risk；
- patch/test feedback 如何改变下一轮 cache plan。

---

## 5. Novelty 评估

### 5.1 原始方案

原始表述：

> 中间段可变代码 + 先 AST index + KVCOMM/CacheBlend 式复用 + KVFlow priority 在 SGLang CPU/GPU tier 中浮现。

评分约为 **2/5**：

- AST/CPG 控制 KV 已有 CodeComp；
- 函数级 KV object 与 GPU/DRAM 生命周期已有 FCGraft；
- code chunk 的 canonical page 和 tier 已有 MEPIC/MiniPIC；
- cross-context reconstruction 已有 KVCOMM；
- non-prefix repair 已有 CacheBlend/EPIC；
- workflow promotion 已有 KVFlow；
- 通用 CPU/GPU/SSD tier 已有 HiCache/LMCache/RAGCache。

如果只完成组件接线，最可能的审稿意见是：

> 这是一个合理但增量的 code-adapted KV serving integration，没有隔离出独立的新算法或新的 correctness guarantee。

### 5.2 改造后的潜力

四模型在看到 CodeComp、FCGraft、MEPIC 和 MiniPIC 后进行二次评审，保守上限下调为约 **3.3–3.6/5**。要达到这一范围，系统至少需要实现以下闭环：

1. repository version 与 model fingerprint 共同定义 KV identity；
2. dependency-cone invalidation 控制哪些 materialized views 失效；
3. structure-conditioned reconstruction 明显优于 KVCOMM generic anchors；
4. calibrated gate 给出 coverage-risk 曲线，越界 dense fallback；
5. cache-plan compiler 联合优化 reuse mode、object order、tier 和 prefetch；
6. 在真实 patch/test loop 中保持 resolve rate 与 dense baseline 无统计显著下降。

---

## 6. 因果正确性：不能绕过的约束

对于第 \(l\) 层第 \(i\) 个 token：

\[
KV_l(i)=f_l(x_{\le i})
\]

代码 artifact 的 KV 不是代码文本自身的静态属性，而是：

```text
artifact text
+ causal ancestors
+ absolute/relative positions
+ model/tokenizer/template/RoPE fingerprint
+ numerical layout
```

### 6.1 RoPE 的边界

\[
K_l(i)=R_i W_K h_l(i)
\]

de-rotate/re-rotate 只修正 \(R_i\)，不修正 prefix 改变引起的 \(h_l(i)\) 变化。V 没有 RoPE，但同样由 \(h_l(i)\) 产生，因此 prefix 变化时 K/V 都可能改变。

### 6.2 Exactness lattice

| 级别 | 条件 | 允许路径 |
| --- | --- | --- |
| L0 | 完整 token prefix 与所有 fingerprint 完全相同 | strict exact reuse |
| L1 | causal ancestors 完全相同，仅整段位置等价平移，模型/attention 语义不变 | 可验证的 RoPE relocation |
| L2 | ancestors 有有限变化，误差可被标定 | anchor reconstruction / selective recompute + probe gate |
| L3 | 结构相关 ancestors、role、query 或大范围 prefix 发生变化 | dense fallback |

任何“独立预计算后可 exact 拼接任意 non-prefix code KV”的表述都不成立。

在 `Architect -> Coder -> Debugger` 和持续编辑场景中，L1 条件几乎总被 role prefix、query、artifact order 或 patch 破坏，因此实际 exact fast path 主要仍是 L0；L1 只能作为窄且必须验证的特例。

### 6.3 Dense fallback 的真实范围

中间位置变化后，理论上所有后续 token 都可能受影响。“局部重算”只有在以下条件下才安全：

- selective recompute 经 dense 对照标定；
- 使用 block-independent attention/link-token 等改变模型语义的机制；
- 或系统接受 approximate，并由风险 gate 控制。

---

## 7. 最强的六个机制方向

二次评审后的系统贡献优先级是：

1. source/dependency incremental invalidation；
2. 持久 versioned logical-to-physical artifact store 与 cache planner；
3. calibrated cross-role reconstruction；
4. structure-conditioned reconstruction，只有在实测胜过 KVCOMM/FCGraft/MEPIC/MiniPIC 后才升级为核心算法贡献。

### 7.1 结构条件化重建先验

核心假设：

> 代码模型对 artifact 的 representational offset 主要由 in-scope symbol、def-use、type、caller/callee 邻域决定。

用结构特征选择 KVCOMM anchors 或预测低秩 residual：

\[
\widehat{KV}=KV_{base}+U_o a(\text{role},\text{dependency closure},\text{query class})
\]

必须通过以下实验成立：

- 结构邻居比随机/embedding-only anchor 更能预测真实 \(\Delta KV\)；
- 在相同 quality 下，storage、recompute 或 latency 优于 faithful KVCOMM；
- 跨 repository、role 和 object order 仍能泛化。

这是最强的**条件性算法**方向，但不是当前可以直接成立的 claim，也是最容易被证伪的方向。

### 7.2 Patch/dependency-cone incremental invalidation

代码修改产生 AST/token edit script，并沿 reverse dependency graph 传播 dirty frontier：

\[
\text{Dirty}(E)=E\cup\text{dependency-closure}(E)
\]

需要区分：

- lexical/content hash；
- normalized AST hash；
- symbol/ABI hash；
- dependency-closure Merkle root；
- prompt-plan/context signature；
- patch epoch/branch/worktree。

它不应宣称静态依赖图能证明 attention independence；作用是 conservative invalidation prior，最终仍由 probe/dense 校验兜底。

### 7.3 Exactness lattice + conformal/probe safety gate

选择最便宜且满足风险预算的方法：

\[
m^*=\arg\min_m Cost(m)
\quad
\text{s.t.}\quad
\widehat{Risk}(m)\le\varepsilon
\]

候选 \(m\)：

```text
exact
-> RoPE relocation
-> anchor/residual reconstruction
-> selective recompute
-> dense
```

probe 可以覆盖：

- symbol signature；
- modified span 首尾；
- branch/callsite tokens；
- 最后层 hidden/logit KL；
- 首个 decode token 或短 continuation。

论文必须报告 calibration coverage、accepted error 和 fallback rate，不能只报告平均 KL。

### 7.4 Causal Cache Plan Compiler

输入：

- retrieval 产生的 artifact DAG；
- order/dependency constraints；
- exactness level；
- resident tier；
- load/recompute cost；
- stage SLO；
- stale/risk score。

输出：

- artifact order；
- exact/reconstruct/recompute/dense mode；
- CPU/GPU placement；
- prefetch/eviction schedule。

目标是联合最小化：

\[
Latency+\lambda Transfer+\rho Risk+\sigma EvictionExternality
\]

这比“给每个 page 算一个 priority”更有机会构成新系统 abstraction。

### 7.5 Versioned Semantic Merkle Cache

为不同一致性维度维护独立 root：

```text
content root
AST root
symbol/ABI root
dependency-closure root
context-plan root
model fingerprint
```

由 Git diff/AST diff 更新逻辑索引，只失效不安全的 context variants，尽可能保留 canonical base。该机制把 repository evolution 与 KV lifecycle 连接起来，是 FCGraft/MEPIC/KVFlow 尚未覆盖的关键差异。

### 7.6 Test/trace-feedback cache learner

每次 patch 记录：

- retrieval set；
- 每个 artifact 的 cache mode；
- patch epoch；
- test outcome；
- stack trace；
- 必要时 dense counterfactual replay。

失败后区分：

- retrieval 错误；
- approximate KV 错误；
- generation 本身错误；
- source/cache version mismatch。

下一轮提高 trace symbols 的保真度，淘汰被证伪 anchor，并限制重复失败。

---

## 8. 固定 Workflow 的 cache contract

### 8.1 Architect

- 目标：广覆盖、低深度。
- 对象：repo map、module/class/API、call/type graph、文档。
- 允许：经标定的量化或 approximate KV。
- 产出：Coder target symbols 和 dependency hotset。
- 策略：生成 plan 时流式预取 Coder 工作集；plan 完成后快速降级 broad context。

### 8.2 Coder

- 目标：窄覆盖、高精度。
- 对象：target functions、types、direct dependencies、tests。
- 规则：
  - 当前修改对象必须 exact 或 dense；
  - 只读依赖可以经过校准 reconstruction；
  - 所有 pages 绑定当前 patch epoch；
  - 测试启动时预取 Debugger slice。

### 8.3 Debugger

- 目标：dirty slice 和失败路径最高保真。
- 对象：changed spans、failing tests、stack trace、callers/callees。
- 规则：
  - patch 和 trace 命中对象 exact/dense；
  - tool logs 为 ephemeral，短 TTL 或不持久化；
  - test 运行期间重建下一轮可能需要的 Coder pages。

### 8.4 Debugger → Coder loop

维护一个受版本约束的 loop kernel：

```text
issue
+ current plan
+ current diff
+ latest failing test/stack trace
+ target objects
+ fixed role prefixes
+ patch epoch
```

旧 patch epoch 不得与新 epoch 混用。失败后：

1. 提升 trace symbols 和相关 dependency cone；
2. 对高风险 approximate objects 做 dense replay；
3. 证伪错误 anchor；
4. 测试运行期间预取下一轮 Coder hotset；
5. 加入 session quota、aging 和 stale-prefetch cancellation，防止循环饿死其他请求。

---

## 9. 存储与成本约束

KV 大小近似：

\[
B/token=2\cdot L\cdot H_{kv}\cdot D_h\cdot bytes(dtype)
\]

完整仓库、多个粒度、多个 role/context anchor 会迅速导致存储爆炸。正确策略应是：

- **全库 logical index**；
- **non-overlapping canonical artifact partition**；
- **physical KV sparse/lazy materialization**；
- file/class 作为逻辑 view，不重复存储嵌套 function KV；
- 只物化 hotset；
- residual 采用低秩、稀疏或量化形式；
- 在线比较：

\[
T_{load}=bytes/BW_{effective}+queue
\]

与：

\[
T_{recompute}=prefill\ cost
\]

若 \(T_{load}\ge T_{recompute}\)，即使命中也应直接重算。

---

## 10. 推荐系统数据模型

### 10.1 Logical Artifact

```text
repository
branch/worktree
commit/patch_epoch
file_path
language
qualified_symbol
AST/CPG stable identity
source span
token IDs
content/AST/ABI/dependency hashes
retrieval and workflow metadata
```

### 10.2 Physical KV Object

```text
artifact version
context signature
position basis
model/tokenizer/template/RoPE fingerprint
layer/token page extents
KV dtype/layout/compression
GPU/CPU/disk tier
base/residual/exact-snapshot type
confidence/error budget
ownership/refcount/pin state
```

### 10.3 Runtime state

```text
ABSENT
CPU_RESIDENT
LOADING_H2D
GPU_EXACT
GPU_APPROXIMATE
VERIFYING
DIRTY
RECOMPUTING
INVALID
```

fast path：

```text
exact
-> verified relocation
-> calibrated reconstruction
-> selective recompute
-> dense fallback
```

---

## 11. 建议 Prototype 路线

### Phase 0：先测是否值得做

在修改 SGLang 核心前采集：

- object overlap/reuse across stages；
- 同一 artifact 在 role/query/order 改变下的 KV variance；
- H2D 与 recompute break-even；
- code churn 和 dependency invalidation ratio；
- logical index hit 与真实 agent access 的差距。

若真实 reuse 或 load advantage 很低，应立即缩小 thesis。

### Phase 1：安全基线

- 从接近 upstream 的干净 SGLang 或 `feature/workflow-priority` 开始。
- 使用现有 SM75 Docker patch。
- 实现完整 fingerprint、content/token equality、ownership/refcount。
- 保留 RadixAttention exact prefix 与 HiCache/LMCache tier。

### Phase 2：faithful KVCOMM

- base KV；
- placeholder/artifact \(\Delta K/\Delta V\)；
- neighboring-prefix offsets；
- Key de-rotation/re-rotation；
- multi-anchor interpolation；
- shareability gate；
- dense fallback 与 online anchor update。

先证明机制正确，再加入 AST。

### Phase 3：Repository artifact registry

- 先支持一种语言；
- function/class 粒度；
- source-token alignment；
- logical-to-physical page catalog；
- lazy physical materialization；
- content/model fingerprint invalidation。

### Phase 4：只选择一个核心 novelty

优先顺序：

1. patch/dependency invalidation + version consistency；
2. calibrated cross-role reconstruction + probe gate；
3. cache-plan compiler；
4. structure-conditioned reconstruction，作为前述稳定后再验证的算法分支。

不要同时实现所有 brainstorm idea。

### Phase 5：Workflow 与 tier

- stage-specific fidelity；
- artifact-level future-use；
- test-time prefetch；
- Debugger→Coder loop；
- fairness、quota、aging、prefetch cancellation。

---

## 12. 评测设计

### 12.1 关键研究问题

1. repository coding traces 中是否存在足够稳定的 artifact reuse？
2. 结构特征能否预测真实 context-induced KV offset？
3. CPU→GPU load 在哪些对象长度、模型和硬件上优于 recompute？
4. dependency invalidation 在真实 commit churn 下是否安全且足够精细？
5. calibrated gate 能否在跨 repository OOD 下保持风险界限？
6. 完整 workflow 的 resolve rate、session latency 和公平性是否优于 baseline？

### 12.2 必须包含的 baseline

- dense prefill；
- SGLang RadixAttention；
- RadixAttention + HiCache/LMCache；
- KVFlow；
- faithful KVCOMM；
- CacheBlend；
- EPIC；
- MEPIC/MiniPIC；
- CodeComp；
- FCGraft 的函数对象/lifecycle 设计；
- 现有 AgentTemplateKV raw copy + RoPE + fixed fraction 作为负对照。

### 12.3 工作负载

- SWE-bench Verified；
- RepoBench / CrossCodeEval；
- DebugBench / Commit0 或真实 commit replay；
- 大型 monorepo；
- 真实 `Architect -> Coder -> Debugger -> Coder` traces。

### 12.4 指标

正确性：

- resolve/pass rate；
- hidden tests；
- compile/patch validity；
- token divergence；
- next-token logit KL；
- stale false hit；
- calibration coverage。

系统：

- TTFT；
- session completion latency；
- workflows/s；
- p95/p99；
- GPU/CPU/SSD footprint；
- H2D bytes；
- load/recompute ratio；
- fallback rate；
- wasted prefetch；
- starvation/fairness。

### 12.5 防止虚假 speedup

- 分开统计 system prompt prefix hit 与 code artifact hit；
- 同时报 cold/warm cache；
- 回放真实代码编辑和失效；
- 包含离线 compilation 和存储摊销；
- 扫描中间段大小与位置；
- 固定相同 retrieval objects/order/tokens，对比不同 KV path；
- 不用 BLEU/ROUGE 代替 test pass；
- 不隐藏 fallback 成本。

### 12.6 Kill criteria

出现任一项应缩小或终止主 thesis：

- correctness/resolve rate 下降超过约 1 个百分点；
- dense fallback 超过约 40%，综合速度无收益；
- residual 超过完整 KV 的 20–25% 仍不稳定；
- 多数对象 \(T_{load}\ge T_{recompute}\)；
- 跨 stage/commit 的 artifact reuse 不足以摊销预计算；
- 结构特征对 \(\Delta KV\) 的预测不优于 embedding-only；
- CodeComp + MEPIC/MiniPIC + KVCOMM + KVFlow + LMCache 的联合 baseline 已达到相同结果。

---

## 13. 论文 thesis 与安全措辞

### 13.1 推荐 thesis

**RepoMV: Causality-Aware, Versioned KV Materialized Views for Repository-Scale Coding Agents**

> 代码 artifact KV 不是可任意搬移的静态块，而是由源码版本与因果上下文参数化的 materialized view。RepoMV 使用版本化 logical-to-physical page index、dependency-aware invalidation、calibrated reconstruction 与 workflow-aware tier planning，在不降低 patch/test 成功率的前提下减少 coding-agent session 的 prefill 和数据移动。

### 13.2 更聚焦的备选

1. **PatchLoopKV**  
   聚焦 patch epoch、dependency invalidation、测试期间 rematerialization 和 Coder–Debugger loop。
2. **StructDeltaKV**  
   聚焦结构条件化 anchor/residual 与 conformal safety gate。
3. **RepoCache Compiler**  
   聚焦 artifact DAG 到 exact/reconstruct/recompute/tiered cache plan 的联合编译。

### 13.3 应避免的表述

- AST 本身证明 KV 可复用；
- non-prefix code KV 可以 strict exact；
- RoPE relocation 可以修复 prefix 语义变化；
- workflow priority 是本项目原创；
- function-level 或 code-specific hierarchical KV 是首次提出；
- 只用 speedup、平均 KL 或小样本任务证明 correctness-preserving。

---

## 14. 最终建议

### 14.1 研究主线

继续项目，但更换 thesis：

```text
从：
AST index + KVCOMM + KVFlow + CPU/GPU tier

改为：
versioned causal KV materialized views
+ structure-conditioned reconstruction
+ patch/dependency invalidation
+ calibrated correctness contract
+ artifact-level workflow planning
```

### 14.2 实施优先级

1. 先做真实 trace instrumentation 和 break-even 测量。
2. 在干净 SGLang 基线上 faithful 复刻 KVCOMM。
3. 建立安全的 repository artifact registry 与 lazy physical KV。
4. 先完成 source/dependency invalidation、patch epoch 和 stale-cache audit。
5. 加入 calibrated cross-role reconstruction 与 dense fallback。
6. 再用先导实验验证“代码注意力/offset 是否受结构邻域支配”；成立时升级为算法贡献，不成立时保留 PatchLoopKV 系统 thesis。
7. 最后接 KVFlow/HiCache；priority/index 只作为系统支撑，不作为核心 novelty。

最终判断：

> **Broad AST-aware KV novelty 已不存在；repository-scale、version-consistent、causality-aware KV materialized-view co-design 仍有研究空间，但必须通过真实编辑循环、强联合 baseline 和明确 kill criteria 来证明。**
