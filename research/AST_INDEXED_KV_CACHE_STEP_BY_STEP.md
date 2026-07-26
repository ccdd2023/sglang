# AST-Indexed Repository KV Cache：逐步研究详解

最后更新：2026-07-13T01:13:56-07:00

## 文档定位

本文件保存对 AST-indexed repository KV Cache 研究结论的教学式、逐步解释。

- 正式 prior-art、novelty、实验和论文路线报告：
  `research/AST_INDEXED_KV_CACHE_NOVELTY_REPORT.md`
- 本文件：
  从问题定义开始，逐步解释为什么原始方案需要更换 thesis，以及新的系统应该如何设计、实现和评测。

最核心的结论是：

> AST 索引仍然要做，但它不能再作为论文的核心创新。真正值得研究的是：如何把持续变化的软件仓库中的代码 KV，管理成带版本、带因果上下文、可失效、可重建、可验证、可分层调度的 materialized views。

---

## Step 1：我们真正想解决什么问题

目标场景是一个超大 Codebase 上的固定 Coding Agent Workflow：

```text
Architect -> Coder -> Debugger
                         |
                         └── failure -> Coder
```

整个仓库可能包含数百万甚至上亿 token：

- 不可能全部放进模型上下文；
- 不可能把全部 KV 长期放在 GPU；
- CPU 内存虽然更大，但 CPU→GPU 搬运不一定比重新 prefill 更快；
- Coder 会持续修改代码，缓存不断失效；
- 同一个函数在 Architect、Coder、Debugger 的不同 prompt 中，前缀不同，因此 KV 数值也不同。

所以系统需要同时解决：

1. **容量问题**：哪些代码 KV 值得物化，存在哪里。
2. **身份问题**：一个 KV page 对应哪个 repository、commit、文件、函数和模型配置。
3. **因果问题**：同样的代码出现在不同 prefix 后，旧 KV 是否还能用。
4. **一致性问题**：代码修改后，哪些缓存必须失效或重算。

这不是普通的“根据 AST 找到函数，然后把 KV 从 CPU 搬到 GPU”问题，而是一个完整的版本化缓存一致性问题。

---

## Step 2：原始方案可以拆成哪些组件

最初方案大致是：

```text
整个 Codebase
-> AST 分段和索引
-> 离线预计算代码 KV
-> 大部分放 CPU
-> 根据 Architect/Coder/Debugger priority 搬到 GPU
-> 中间代码变化时进行 KV relocation/reconstruction
```

拆成已有研究组件后：

| 原始能力 | 对应已有工作 |
| --- | --- |
| AST/代码结构控制 KV | CodeComp |
| 函数作为 KV 对象 | FCGraft |
| code chunk/file 的独立 KV pages | MEPIC、MiniPIC |
| 跨 context 的 KV 重建 | KVCOMM |
| non-prefix chunk selective recompute | CacheBlend、EPIC |
| workflow-aware priority | KVFlow |
| CPU/GPU/SSD KV tier | HiCache、LMCache、RAGCache |
| repository AST/graph 与增量索引 | Code Isn’t Memory、LocAgent，但它们不存 KV |

因此，原始方案里的主要组件都已经存在直接或强邻近先例。

这就是为什么不能再简单声称：

> “我们首次用 AST 对 KV Cache 做索引。”

---

## Step 3：最重要的两个直接先例

### 3.1 CodeComp：程序结构已经直接参与 KV 管理

论文：

- [CodeComp `2604.10235`](https://arxiv.org/abs/2604.10235)

它使用 Joern 提取 Code Property Graph：

```text
CPG = AST + CFG + PDG
```

然后使用程序结构决定：

- 哪些 method/function chunk 更重要；
- 每个 chunk 获得多少 KV budget；
- 哪些代码 span 必须保留；
- 哪些 token KV 可以被删除；
- 哪些结构节点不能被普通 attention pruning 丢掉。

它已经占据了这个 broad claim：

> “程序结构直接控制 KV Cache 的保留和淘汰。”

但 CodeComp 的边界也很清楚：

- 它是单次请求内部的 KV compression；
- 不把函数 KV 做成跨请求持久对象；
- 没有 CPU/GPU artifact store；
- 不处理源码版本变化；
- 不处理 Architect/Coder/Debugger 跨 role 重建；
- 不建立 logical artifact 到 physical page 的持久映射。

所以它击中了“AST-aware KV”这个 broad novelty，但没有完成 repository lifecycle。

### 3.2 Functional Cache Grafting：函数级 KV 对象也已经存在

论文：

- [Functional Cache Grafting / FCGraft `2606.13097`](https://arxiv.org/abs/2606.13097)

它已经实现：

- 函数 ID 作为 cache key；
- 函数 interface 和 implementation 的双层表示；
- 函数文本与函数 KV 的关联；
- function retrieval；
- KV stitching；
- localized patching；
- 成功执行后更新函数库；
- 根据 recency、frequency、co-occurrence、semantic score 决定保留；
- GPU/DRAM residency。

因此以下 claim 已经不安全：

- “首个 function-level KV cache”；
- “首个将函数作为 KV object”；
- “首个 code KV 的 GPU/CPU 分层系统”；
- “首个支持函数 KV patch/update 的系统”。

但 FCGraft 的场景是机器人或 embodied agent 的 Code-as-Policies：

- 函数库相对独立；
- 不是真实的大型软件仓库；
- 没有 class/file/AST span 多粒度对象；
- 没有 Git commit、branch、worktree；
- 没有跨函数 dependency invalidation；
- 没有不同 agent role prefix 下的 KV reconstruction。

因此它占据了“函数对象系统”，但没有覆盖“持续演化的软件仓库一致性”。

---

## Step 4：MEPIC 和 MiniPIC 为什么也很危险

### 4.1 MEPIC

论文：

- [MEPIC `2512.16822`](https://arxiv.org/abs/2512.16822)

它已经覆盖：

- document/code chunk；
- canonical paged layout；
- 完整 chunk hash；
- NoPE KV；
- 运行时 RoPE；
- 首 block request-specific recompute；
- 后续 blocks 跨请求共享；
- refcount、LRU；
- HBM 和 LMCache remote tier。

所以“把代码 chunk 编译成独立 physical KV pages，再按需装入 GPU”已经不是空白。

它缺少的是：

- 自动 AST artifact identity；
- repository version；
- source dependency invalidation；
- agent workflow semantics；
- cross-role context correction。

### 4.2 MiniPIC

论文：

- [MiniPIC `2606.13126`](https://arxiv.org/abs/2606.13126)

它已经把 code file 视为独立 span：

- 保存 unrotated K；
- 在运行时按真实位置施加 RoPE；
- 支持 position-independent reuse；
- 可以与 CPU offload 结合。

因此“代码文件 KV 在不同位置重复使用”也已经有很强的先例。

---

## Step 5：其他组件分别被谁覆盖

### KVCOMM

- [KVCOMM `2510.12872`](https://arxiv.org/abs/2510.12872)

覆盖：

```text
base KV
+ context-dependent ΔK/ΔV
+ RoPE relocation
+ multi-anchor interpolation
+ shareability gate
+ dense fallback
```

而且已经包含 collaborative coding 场景。

因此跨 Architect/Coder/Debugger 的 base+offset 思路主要来自 KVCOMM，不是本项目原创。

### KVFlow

- [KVFlow `2507.07400`](https://arxiv.org/abs/2507.07400)

覆盖：

```text
Agent Step Graph
-> steps-to-execution
-> cache priority
-> CPU backup
-> CPU→GPU overlapped prefetch
```

所以固定 Workflow 驱动 priority 已经被覆盖。

### CacheBlend

- [CacheBlend `2405.16444`](https://arxiv.org/abs/2405.16444)

覆盖：

- 多个离线 chunk KV 的融合；
- 逐层识别高 KV deviation token；
- selective recompute；
- 只重算一部分 token。

### EPIC

- [EPIC `2410.15332`](https://arxiv.org/abs/2410.15332)

覆盖：

- immutable chunk compile/link；
- 每个非首 chunk 的 leading-token recompute；
- training-free LegoLink。

### Prompt Cache

- [Prompt Cache `2311.04934`](https://arxiv.org/abs/2311.04934)

已经有：

- 命名 module；
- module→KV 映射；
- 固定 position；
- GPU/CPU 存储。

区别是它依赖人工 schema，而不是 AST 自动生成。

### LMCache

- [LMCache `2510.09665`](https://arxiv.org/abs/2510.09665)

已经提供：

- CPU；
- SSD；
- remote KV；
- pin、move、lookup、clear；
- 与 non-prefix reuse 结合的基础设施。

所以 CPU/GPU/SSD tier 本身不是新贡献。

---

## Step 6：哪些 claim 必须撤回

| 不安全 claim | 原因 |
| --- | --- |
| 首个 AST-aware KV Cache | CodeComp |
| 首个 function-level KV Cache | FCGraft |
| 首个 code-specific hierarchical KV Cache | FCGraft、MEPIC、MiniPIC |
| 首个 code chunk KV compiler | MEPIC、MiniPIC、EPIC、TurboRAG |
| 首个 workflow-aware coding-agent KV manager | KVFlow、MORI |
| 首个 CPU/GPU code KV tier | FCGraft、MEPIC、MiniPIC |
| 首个 non-prefix code KV reuse | MEPIC、MiniPIC、KVCOMM |
| AST index 本身是核心算法创新 | CodeComp/FCGraft 已否定 |

更安全的表述是：

> 据本次检索，尚未发现一个 repository-scale serving system，同时维护已有源码 artifact 的版本化 logical-to-physical KV page 映射、源码依赖失效、跨 role 重建和 artifact-level workflow tier scheduling。

重点从“某个组件第一次出现”，变成“完整一致性闭环尚未出现”。

---

## Step 7：为什么原方案只有大约 2/5 novelty

审稿人很可能会把原方案理解为：

```text
CodeComp/FCGraft 的 code object
+ MEPIC/MiniPIC 的 code pages
+ KVCOMM 的跨 context 重建
+ KVFlow 的 workflow priority
+ LMCache/HiCache 的 tier
+ AST 作为 key
```

然后追问：

> 如果把这些已有论文的组件全部去掉，你自己的技术心脏还剩什么？

原方案下，剩余内容很少：

| 维度 | 评分 |
| --- | ---: |
| 问题重要性 | 3/5 |
| 单独机制 novelty | 1/5 |
| 应用场景 novelty | 2–2.5/5 |
| 原始系统组合 | 2/5 |
| 强 venue 可发表性 | Weak Reject |

这不代表系统没有工程价值，而是“工程上有价值”和“论文上有新机制”是两件不同的事。

---

## Step 8：最根本的技术约束是 causal attention

对于第 \(l\) 层、第 \(i\) 个 token：

\[
KV_l(i)=f_l(x_{\le i})
\]

这意味着一个函数的 KV 并不只是函数文本的属性，而是：

```text
函数文本
+ 前面所有 token
+ token 位置
+ role/system prompt
+ 用户 query
+ 前面排列的其他代码
+ 模型和 tokenizer 配置
```

例如同一个函数：

```python
def validate(user):
    return user.is_active
```

在 Architect 中可能是：

```text
[Architect system prompt]
[用户需求]
[repo map]
[validate 函数]
```

在 Coder 中可能是：

```text
[Coder system prompt]
[Architect plan]
[目标文件]
[validate 函数]
```

在 Debugger 中可能是：

```text
[Debugger system prompt]
[patch]
[test failure]
[stack trace]
[validate 函数]
```

虽然函数 token 完全相同，但 causal ancestors 不同，因此 hidden states 不同，最终 K 和 V 都可能不同。

---

## Step 9：RoPE relocation 到底能修什么

Key 通常可以写成：

\[
K_l(i)=R_iW_Kh_l(i)
\]

这里有两种不同误差。

### 9.1 Position error

同样的 hidden state，只是 token 从位置 100 移到位置 500。

这时可以：

```text
de-rotate old position
-> re-rotate new position
```

修正 \(R_i\)。

### 9.2 Representation error

如果 prefix 变了，hidden state \(h_l(i)\) 也变了。

RoPE relocation 无法修复 \(h_l(i)\)。

而且 Value 虽然没有 RoPE，但：

\[
V_l(i)=W_Vh_l(i)
\]

所以 prefix 改变后：

- K 会变；
- V 也会变。

因此：

> raw KV copy + Key RoPE shift 只能解决位置，不等于跨 context reconstruction。

这也是 AgentTemplateKV 当前分支和 faithful KVCOMM 之间最核心的区别。

---

## Step 10：Exactness 应该分成四级

### L0：严格 exact

条件：

- 完整 token prefix 相同；
- 所有 token ID 相同；
- 模型、tokenizer、chat template、RoPE、dtype/layout 全部相同。

这就是标准 RadixAttention exact-prefix hit。

### L1：纯位置等价

条件非常严格：

- 所有 causal ancestors 完全相同；
- 只发生整体位置平移；
- attention 语义不变；
- 标准 RoPE 条件成立。

这时 relocation 可能 exact。

但在 Architect/Coder/Debugger 中，因为 role prefix 和 query 几乎总不同，L1 实际接近空集。

### L2：受控近似

条件：

- prefix 有变化；
- 但真实误差经过标定后可控制。

允许使用：

- KVCOMM anchors；
- residual reconstruction；
- selective recompute；
- probe verification。

### L3：不安全

例如：

- role 完全改变；
- 前序代码大范围变化；
- 修改的是目标对象的重要依赖；
- query 与旧 anchor 明显 OOD；
- probe 不通过。

此时必须 dense prefill。

最终路径应该是：

```text
L0 exact
-> 窄 L1 relocation
-> L2 calibrated reconstruction
-> selective recompute
-> L3 dense fallback
```

---

## Step 11：AST 能做什么，不能做什么

### AST 能做

1. **Artifact segmentation**
   - module；
   - class；
   - function；
   - method；
   - block；
   - statement。

2. **Logical identity**
   - qualified symbol；
   - source span；
   - content hash；
   - AST hash。

3. **Dependency graph**
   - import；
   - call；
   - def-use；
   - type relation；
   - test relation。

4. **Retrieval prior**
   - 当前任务可能需要哪些 artifact。

5. **Invalidation prior**
   - 某个 symbol 修改后，哪些 dependents 可能受影响。

6. **Reconstruction prior**
   - 哪些 prefix differences 可能与当前函数有关。

### AST 不能做

1. 不能证明两个 KV 数值相同。
2. 不能替代完整 token equality。
3. 不能证明 Transformer attention 只沿 call graph 传播。
4. 不能保证动态语言、反射、宏和自然语言注释的依赖都被捕获。
5. 不能让独立计算的 non-prefix KV 自动变成 exact。
6. 不能直接替代 KVCOMM 的 context distance 或实际 probe。

因此，AST 应该是：

```text
index
+ retrieval prior
+ invalidation prior
+ reconstruction prior
+ gating feature
```

而不是 correctness proof。

---

## Step 12：新的核心 abstraction——KV Materialized View

数据库中的 materialized view：

- 从基础数据计算出来；
- 有版本；
- 数据变化后可能 stale；
- 可以增量刷新；
- 也可以全部重建。

代码 KV 应被看作同样的对象：

```text
KV View =
source artifact
+ source version
+ causal context
+ model fingerprint
+ position basis
+ numerical format
```

例如一个函数 `validate` 不应该只有一个 KV：

```text
validate -> KV
```

而应该是：

```text
validate@commit-A
  ├── canonical base KV
  ├── Architect-context variant
  ├── Coder-context residual
  ├── Debugger-context residual
  └── exact snapshots / calibrated variants
```

每个 variant 都有：

- 来源版本；
- context signature；
- error/risk；
- resident tier；
- physical pages；
- 是否 stale；
- 是否经过验证。

这就是 versioned causal KV materialized views。

---

## Step 13：安全的论文 thesis

推荐论文名称：

> **RepoMV: Causality-Aware, Versioned KV Materialized Views for Repository-Scale Coding Agents**

安全的核心 claim：

> 我们把软件仓库中的代码 artifact KV 建模为由源码版本和因果上下文共同参数化的 materialized views。系统维护版本化 logical-to-physical page index，通过 dependency-aware invalidation、calibrated cross-role reconstruction 和 workflow-aware tier planning，在不显著降低 patch/test 成功率的情况下减少 coding-agent session 的 prefill 和数据移动。

与现有工作的区别：

| 工作 | 缺少什么 |
| --- | --- |
| CodeComp | 无持久对象、无跨请求、无版本生命周期 |
| FCGraft | 无真实 evolving repository dependency graph |
| MEPIC/MiniPIC | 无 AST artifact identity 和 source invalidation |
| KVCOMM | 无 repository、AST、tier lifecycle |
| KVFlow | 只管理 exact-prefix agent nodes |
| Code Isn’t Memory | 有 AST/Merkle，但没有 Transformer KV |

贡献不再是“发明其中一个组件”，而是建立这些机制之间的 repository consistency contract。

---

## Step 14：完整的数据模型

### 14.1 Logical Artifact Record

至少包含：

```text
repository ID
branch / worktree
commit
patch epoch
file path
language
qualified symbol
AST/CPG stable ID
source span
token IDs
content hash
AST hash
symbol/ABI hash
dependency-closure Merkle root
retrieval metadata
workflow metadata
```

### 14.2 Physical KV Object

cache key 应接近：

```text
(
  artifact_id,
  artifact_version,
  context_signature,
  position_basis,
  model_fingerprint,
  dtype,
  layout
)
```

物理信息包括：

```text
layer range
token range
page IDs
base / residual / exact snapshot
GPU / CPU / SSD
compression
confidence
error budget
refcount
pin ownership
dirty state
```

### 14.3 完整 model fingerprint

必须覆盖：

```text
checkpoint / weight hash
adapter / LoRA
tokenizer
normalizer
special tokens
chat template
RoPE theta/scaling
attention architecture
KV dtype/layout/quantization
必要时 kernel/numerical implementation
```

任何 fingerprint 不同都不能直接 exact reuse。

---

## Step 15：为什么全库只能 logical-full，不能 physical-full

KV 的体积非常大。

近似公式：

\[
B/token=2\cdot L\cdot H_{kv}\cdot D_h\cdot bytes(dtype)
\]

以某些 8B GQA 模型为例，可能约为：

```text
128 KiB / token
```

那么：

```text
1,000,000 tokens ≈ 122 GiB KV
```

如果还同时保存：

- file KV；
- class KV；
- function KV；
- 三个 role；
- 多个 context anchors；
- 多个 commit；
- base 和 residual；

存储会迅速膨胀到不可接受。

所以必须采用：

```text
全库 logical index
+ 非重叠 canonical artifact partition
+ physical KV lazy materialization
+ hotset only
```

每个函数都可以在索引中存在，但不是每个函数都立即保存完整 KV。只有预计会复用、且 load 比 recompute 划算的对象才物化。

---

## Step 16：加载还是重算必须有成本模型

不能采用：

> 只要 CPU 中命中，就一定搬到 GPU。

必须在线比较：

\[
T_{load}=bytes/BW_{effective}+queue\ delay
\]

和：

\[
T_{recompute}=prefill\ cost
\]

如果：

\[
T_{load}\ge T_{recompute}
\]

就应该直接重算。

影响因素包括：

- artifact token 长度；
- 模型层数；
- KV dtype；
- PCIe/NVLink；
- pinned memory；
- page fragmentation；
- GPU 当前空闲算力；
- DMA queue；
- 是否能与 tool execution 重叠。

因此 Phase 0 必须先测 break-even，而不是直接大改 SGLang。

---

## Step 17：最强机制一——源码和依赖增量失效

当 Coder 修改代码时，不能只按文件名清缓存，也不能假设只有被修改函数受影响。

需要维护多层 hash：

```text
content hash
AST hash
symbol/ABI hash
dependency-closure Merkle root
context-plan hash
patch epoch
```

例如：

```text
A.py::foo
-> 被 B.py::bar 调用
-> bar 被 test_foo 使用
```

修改 `foo` 后：

1. `foo` 的旧 content variant 失效；
2. `foo` 的 exact snapshots 失效；
3. 依赖旧 `foo` interface/context 的 residual 失效；
4. `bar` 和相关 test 被标记为 dirty candidates；
5. 未受影响的 canonical objects 继续保留；
6. 下次实际访问时 lazy refresh。

dependency graph 只能指导 conservative invalidation，不能证明没有图边的代码在 attention 中完全无关。

因此最终仍需要 probe 或 dense fallback。

这部分是最强、最稳妥的系统贡献，因为 FCGraft 主要 patch 单函数，CodeComp 又不维护跨请求对象库。

---

## Step 18：最强机制二——可标定的跨 role 重建

对于每个 artifact，候选路径包括：

```text
exact
RoPE relocation
KVCOMM anchor reconstruction
low-rank residual
selective recompute
dense
```

系统选择最便宜且满足风险预算的方法：

\[
m^*=\arg\min_m Cost(m)
\]

约束：

\[
\widehat{Risk}(m)\le\varepsilon
\]

Risk 不能只使用中间层 KV L2，可以组合：

- layer-wise KV distance；
- 最后层 hidden-state distance；
- next-token logit KL；
- top-1 consistency；
- 短 continuation divergence；
- task-level failure probability。

### Probe 可以选择

- 函数 signature token；
- 修改点附近 token；
- branch predicate；
- callsite；
- 首尾 token；
- stack trace 命中行；
- 少量真实 context 下的重算 token。

路径：

```text
approximate assembly
-> probe
   -> pass: accept
   -> uncertain: selective recompute
   -> fail: dense
```

论文必须同时报告：

- accepted coverage；
- fallback rate；
- accepted samples 的实际错误率；
- OOD repository 上的 calibration。

不能只报告平均 KL 很小。

---

## Step 19：结构条件化 reconstruction 为什么只是条件性创新

我们希望验证：

> 一个函数在新 prefix 下的 KV offset，主要由它的 in-scope symbol、caller/callee、def-use 和 type neighborhood 决定。

如果假设成立，可以：

- 使用结构邻居选择 KVCOMM anchors；
- 预测 residual cluster；
- 避免扫描全局 anchor pool；
- 更准确地识别需要重算的 token。

例如：

\[
\widehat{KV}=KV_{base}+U_o a(
role,
dependency\ closure,
query\ class
)
\]

但它不能直接作为当前 claim，因为：

- CodeComp 已经使用 CPG 影响 KV；
- FCGraft 已有函数级对象和 patch；
- KVCOMM 已有 context-conditioned offset；
- 静态程序图不一定对应模型真实 attention。

它必须实测证明：

```text
structure-aware anchor selection
>
embedding-only KVCOMM anchor selection
```

并且在跨 repository、role、query、artifact order 和 commit 时仍然成立。

如果不成立，就应该放弃这个算法 thesis。

---

## Step 20：最强机制三——Causal Cache Plan Compiler

简单 priority 只是：

```text
每个 page 算一个分数
```

更强的设计是 cache-plan compiler。

输入：

```text
检索得到的 artifact DAG
artifact order constraints
dependency graph
exactness level
resident tier
load cost
recompute cost
risk
workflow stage
memory pressure
```

输出：

```text
加载哪些对象
哪些 exact
哪些 reconstruct
哪些 selective recompute
哪些 dense
对象放 CPU 还是 GPU
何时 prefetch
何时 eviction
```

优化目标：

\[
Latency
+\lambda Transfer
+\rho Risk
+\sigma EvictionExternality
\]

Eviction externality 指：

- 为当前 request 搬入一个 speculative object；
- 却把另一个即将运行 session 的关键 exact cache 驱逐掉。

这比 KVFlow 的单一 steps-to-execution 更强，但必须与 KVFlow 做直接对比。

---

## Step 21：Architect 阶段如何管理

Architect 的特点：

- 看得广；
- 不一定深入每一行；
- 主要需要 repo map、API、类型、模块关系；
- 会产生 Coder 下一步需要的 symbol set。

建议：

```text
广覆盖
+ 较低精度
+ 较多 CPU-resident objects
+ 只将下一阶段 hotset 提升到 GPU
```

允许：

- 经标定的量化 KV；
- approximate background dependencies；
- module/class-level logical views。

但不能直接假设 Architect 不在乎代码细节，这仍需在真实 benchmark 上验证。

Architect 输出的 plan 应显式包含：

```text
target files
target symbols
dependencies
tests
expected edit regions
```

这组信息成为 Coder prefetch plan。

---

## Step 22：Coder 阶段如何管理

Coder 的特点：

- 工作集较窄；
- 对标识符、类型和语法精度要求高；
- 会产生新的 patch epoch。

规则：

1. 当前修改对象：
   - 必须 exact 或 dense；
   - 不应直接使用未经验证的 approximate KV。

2. 直接类型/接口依赖：
   - high fidelity；
   - 可以使用经过 probe 的 reconstruction。

3. 远程只读依赖：
   - 可使用较低精度；
   - 不得挤占当前编辑对象。

4. Coder 生成 patch 后：
   - patch epoch 加一；
   - 旧 exact snapshots 失效；
   - dependency dirty frontier 更新；
   - 测试启动时预取 Debugger 工作集。

Coder 的 generation KV 不能直接交给 Debugger 并宣称 exact，因为 Debugger 的 role prefix 已经改变。

---

## Step 23：Debugger 阶段如何管理

Debugger 需要：

```text
current patch
failing tests
stack trace
changed functions
callers/callees
test infrastructure
```

精度策略：

- changed spans：exact/dense；
- stack trace 命中函数：exact/dense；
- 直接 callers/callees：高保真；
- 远程依赖：经过校准的 approximate；
- test logs/stdout/stderr：ephemeral，不做长期 artifact cache。

Debugger 有动态反馈：

```text
stack trace
compiler error
failing assertion
test outcome
```

这些信号可以：

- 提高某些 symbol 的 priority；
- 提高其 fidelity；
- 证伪某些 approximate anchors；
- 决定下一轮 Coder 的 dense refresh；
- 判断一次失败是否可能来自 stale cache。

---

## Step 24：Debugger → Coder 循环如何管理

维护一个明确的 loop kernel：

```text
issue
current plan
current diff
patch epoch
latest failure
stack trace
target objects
role prefixes
```

当 Debugger 失败时：

1. 解析 stack trace；
2. 提升 trace symbols；
3. 计算 dependency cone；
4. 对使用 approximate KV 的高风险对象做 dense counterfactual replay；
5. 如果 dense 与 approximate 输出明显不同，标记 anchor 失效；
6. 测试执行期间预取下一轮 Coder hotset；
7. 禁止旧 patch epoch 的 page 混入新请求。

还要处理公平性：

- Debugger↔Coder 热循环不能永久占用 GPU；
- 每个 session 有 pinned-byte cap；
- priority 加 aging；
- stale speculative prefetch 及时取消。

---

## Step 25：完整在线请求路径

最终请求流程：

```text
1. 用户任务进入
2. 确定当前 workflow stage
3. 检索相关 artifact
4. 构造 artifact DAG 和 cache plan
5. 检查 repository/model/context fingerprints
6. 对每个 artifact 选择：
   exact
   relocation
   reconstruction
   selective recompute
   dense
7. 比较 CPU load 与 recompute cost
8. reserve / pin GPU pages
9. CPU→GPU promotion
10. approximate path 执行 probe
11. probe 不通过则升级重算
12. 执行普通 decode
13. 记录真实 KV offset 和 telemetry
14. Coder patch 后更新版本与 dirty frontier
15. test/trace 反馈更新下一轮 cache plan
```

核心 fast path 必须始终是：

```text
exact
-> verified approximate
-> dense fallback
```

不能让低置信 approximate cache 抢占 correctness。

---

## Step 26：Prototype 实施顺序

### Phase 0：先测是否值得做

先不大改 SGLang，测四件事。

#### 0.1 Artifact reuse

统计真实 Workflow 中：

- 同一函数是否跨 Architect/Coder/Debugger 重复出现；
- Debugger→Coder 循环中的重复率；
- 跨不同 task 的复用率；
- 跨 commit 的存活时间。

#### 0.2 Context variance

对同一函数放入不同 role、query、artifact order、prefix 和 position，计算真实 K/V 与 dense ground truth 的差异。

#### 0.3 Load/recompute break-even

在本机 SM75 Docker 中测：

```text
CPU pinned memory -> GPU
vs
GPU dense prefill
```

按对象长度扫描。

#### 0.4 Edit churn

回放真实 commits：

- 每次修改多少函数；
- dependency closure 多大；
- 有多少缓存仍可保留；
- 全失效、文件失效、symbol 失效之间差异。

### Phase 1：建立安全 SGLang 基线

从接近 upstream 的干净分支或 `feature/workflow-priority` 开始。

不使用 `fix/placeholder-pool-activation` 作为核心基线。

先实现：

- ownership/refcount；
- page pinning；
- full token equality；
- full content hash；
- 完整 fingerprint；
- exact-prefix baseline；
- dense correctness baseline。

### Phase 2：faithful 复刻 KVCOMM

必须真正实现：

- canonical/base placeholder KV；
- context-specific \(\Delta K/\Delta V\)；
- neighboring-prefix offsets；
- Key de-rotation/re-rotation；
- multi-anchor soft interpolation；
- length/embedding/entropy gate；
- dense fallback；
- dense 后 online anchor update。

不能用：

```text
从某个旧请求 copy raw KV
+ Key shift
+ 单 nearest neighbor
```

代替。

### Phase 3：Versioned Artifact Registry

先支持一种语言，例如 Python。

粒度先做：

```text
function
class
```

不要一开始同时做 statement/basic block。

实现：

- tree-sitter/compiler parsing；
- source-token alignment；
- content/AST/symbol hash；
- logical artifact catalog；
- physical page catalog；
- lazy materialization；
- patch epoch；
- stale audit。

### Phase 4：Dependency Invalidation

实现：

```text
Git diff
-> token/AST edit
-> changed symbols
-> reverse dependency closure
-> invalidate context variants
-> lazy rematerialization
```

先以 correctness 为目标，不追求极小 invalidation set。

### Phase 5：Calibrated Reconstruction

加入：

- KVCOMM reconstruction；
- probe tokens；
- risk score；
- selective recompute；
- dense fallback；
- coverage-risk reporting。

### Phase 6：Workflow 与 Tier

最后加入：

- Architect/Coder/Debugger fidelity；
- artifact-level priority；
- test-time prefetch；
- CPU/GPU placement；
- fairness/aging/quota；
- Debugger→Coder loop。

这个顺序很重要：不能先写复杂 scheduler，再发现 KV reconstruction 本身不可靠。

---

## Step 27：实验问题

### RQ1：真实 artifact reuse 是否足够高

如果真实重复率很低，整个预计算系统无法摊销。

### RQ2：结构信息能否预测 KV offset

比较：

```text
embedding-only anchors
vs
AST/CPG-conditioned anchors
vs
random anchors
vs
dense
```

### RQ3：load 是否真的快于 recompute

扫描：

- artifact size；
- GPU；
- PCIe；
- dtype；
- quantization；
- fragmentation；
- concurrency。

### RQ4：增量失效是否安全

比较：

```text
全库失效
文件级失效
symbol-level 失效
dependency-cone 失效
```

统计：

- invalidated fraction；
- stale false hit；
- rebuild latency。

### RQ5：端到端 Workflow 是否受益

最终指标不能只看 TTFT，而要看：

```text
完成一个 repository task 的总时间
```

包括 Architect、Coder、test tools、Debugger、失败循环和 cache fallback。

---

## Step 28：必须比较的 Baseline

至少包括：

1. Dense prefill。
2. SGLang RadixAttention。
3. RadixAttention + HiCache。
4. KVFlow。
5. Faithful KVCOMM。
6. CacheBlend。
7. EPIC。
8. MEPIC/MiniPIC。
9. LMCache。
10. CodeComp 对应的 structure-aware compression。
11. FCGraft 对应的 function-object lifecycle。
12. 当前 AgentTemplateKV：

    ```text
    raw copy + RoPE + fixed-FRAC
    ```

    作为负对照。

最危险的 baseline 是联合 baseline：

```text
CodeComp structure
+ MEPIC/MiniPIC pages
+ KVCOMM reconstruction
+ KVFlow scheduling
+ LMCache tier
```

如果这个联合 baseline 已达到相同效果，我们的系统就没有独立贡献。

---

## Step 29：指标

### Correctness

- SWE-bench resolve rate；
- hidden-test pass；
- compile success；
- patch validity；
- token divergence；
- next-token KL；
- top-1 consistency；
- stale false hit；
- fallback rate；
- calibration coverage。

### Performance

- TTFT；
- total session latency；
- workflows/s；
- p95/p99；
- GPU/CPU/SSD footprint；
- H2D bytes；
- overlap ratio；
- load/recompute ratio；
- wasted prefetch；
- per-stage hit rate。

### Fairness

- cold session wait time；
- starvation；
- pinned-byte fairness；
- p99 latency。

---

## Step 30：如何防止虚假 speedup

以下情况容易让结果看起来比真实情况好：

1. 把固定 system prompt 的普通 prefix hit 算成代码 KV 收益。
2. 所有任务共享 warm cache，但不报告 cold start。
3. 不模拟代码修改。
4. 中间可变段非常小。
5. 忽略离线预计算成本。
6. 忽略 CPU 存储占用。
7. 忽略 fallback。
8. 只测短输出，让 TTFT 占据全部时间。
9. 用 BLEU/ROUGE 掩盖代码测试失败。
10. retrieval set 不同，导致性能变化实际来自检索。

必须使用 paired experiment：

```text
相同 artifact
相同顺序
相同 token
只改变 KV execution path
```

---

## Step 31：何时停止或缩小 thesis

明确的 kill criteria：

1. Correctness 或 resolve rate 下降超过约 1 个百分点。
2. Dense fallback 超过约 40%，综合速度无收益。
3. residual 超过完整 KV 的 20–25% 仍不稳定。
4. 大多数对象满足：

   \[
   T_{load}\ge T_{recompute}
   \]

5. 跨 stage/commit artifact reuse 太低。
6. 结构特征不能比 embedding-only 更好地预测 \(\Delta KV\)。
7. dependency invalidation 经常漏掉 stale cache。
8. 强联合 baseline 达到相同结果。

对应退路：

### 如果结构条件化 reconstruction 成立

主打：

```text
RepoMV + StructDeltaKV
```

### 如果结构 reconstruction 不成立，但版本失效有效

缩小为：

```text
PatchLoopKV
```

重点做 patch epoch、dependency invalidation、test-time rematerialization。

### 如果 CPU load 不如 recompute

减少 CPU KV，转向：

- GPU hotset；
- compression；
- exact-prefix；
- workflow scheduling；
- tool-time recompute。

### 如果真实 artifact reuse 很低

停止 whole-codebase KV thesis，回到 KVFlow/KVCOMM 的普通 agent-context 优化。

---

## Step 32：五个代理的最终共识

### 专项 Research Agent

确认：

- CodeComp、FCGraft 是 A 类直接先例；
- MEPIC、MiniPIC 是最危险的 code-specific 邻近工作；
- 完整 evolving-repository lifecycle 尚未发现。

### GPT-5.6 Sol

最重要建议：

- 把 KV 视为 materialized view；
- logical index 可以全库；
- physical KV 必须 sparse/lazy；
- 先做 break-even 和 kill criteria。

### Claude Opus 4.8

最重要建议：

- 原方案是典型 weak-reject 工程组合；
- exact/approximate/dense 必须形式化；
- dependency invalidation 和 calibrated error budget 比 index/priority 更有价值。

### Claude Opus 4.6

最有价值部分：

- artifact/page 数据模型；
- cache state machine；
- workflow contract；
- multi-signal scheduling。

第一轮中的两个错误已经纠正：

- EPIC 编号和机制；
- suffix V 不受 prefix 影响这一错误。

### Gemini 3.1 Pro

贡献了：

- role-specific fidelity；
- semantic subgraph prefetch；
- test feedback；
- speculative branch planning。

第一轮中的两个大胆想法不能直接称为 exact：

- AST-isomorphic positions；
- Coder generation KV 直接嫁接 Debugger。

---

## Step 33：最终应该做什么

不再把重点放在：

```text
AST label
AST index
简单 priority
把 KV 放 CPU
RoPE shift
```

这些仍然需要实现，但属于基础设施。

真正的中心应是：

```text
1. repository version consistency
2. source/dependency incremental invalidation
3. causal context-aware reconstruction
4. calibrated correctness and fallback
5. logical artifact -> physical page lifecycle
6. artifact-level workflow cache planning
```

最准确的一句话是：

> 代码仓库不是一组静态、可任意搬移的 KV blocks；每个代码 KV 都是由源码版本和因果上下文共同决定的 materialized view。系统的研究价值，在于如何在持续编辑、跨角色和多级内存中安全地维护、验证、失效和调度这些 views。

这就是 prior art、四模型评估和二次修正后真正收敛下来的研究方向。
