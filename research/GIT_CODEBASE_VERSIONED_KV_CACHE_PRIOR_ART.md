# Git / Codebase Version-Aware KV Cache：2024–2026 Prior Art 调研

最后更新：2026-07-13T04:11:21-07:00

## 1. 执行摘要

本轮由三个独立的 GPT-5.6 Sol Max 代理分别研究 2024、2025、2026 年工作，并追踪到 2026-07-13 的最新 revision、正式 venue、DBLP 和官方代码。三个代理均优先尝试 alphaXiv/arXiv MCP；alphaXiv discovery/full-text 接口在执行期间多次返回 HTTP 429，因此代理继续使用 arXiv PDF/HTML、arXiv API、DBLP、正式会议页面和官方仓库交叉核查。主会话另外下载并抽查了八篇关键论文全文。

### 1.1 最核心结论

按照严格定义，本次检索在 2024、2025、2026 三个年份中均得到：

> **A 类直接先例为 0。**

截至 2026-07-13，本次检索未发现一个系统把以下 repository/source version 信息直接作为普通 Transformer attention KV 的一等身份、有效性或一致性协议：

```text
repo_id
+ commit / tree / blob
+ branch / worktree epoch
+ artifact path / symbol / span
+ patch lineage
+ model / tokenizer / template / RoPE fingerprint
```

也未发现完整实现以下闭环的工作：

```text
Git/source update
-> changed artifact detection
-> dependency-aware KV invalidation
-> cross-version exact reuse / repair / rematerialization
-> GPU/CPU/SSD physical-page lifecycle
-> stale audit / rollback / dense fallback
```

### 1.2 但不能声称“没人做过版本变化后的 KV 处理”

已有工作分别占据了多个组成部分：

- **2024 PIE**：代码被编辑后保留 prefix KV、重算编辑 span、移动 suffix Key。
- **CacheBlend / Cache-Craft / KVCOMM / Leyline / KVEraser**：处理上下文或 prompt 改变后的 KV repair、offset 或 selective recompute。
- **Irminsul / MEPIC / MiniPIC**：内容寻址、position-independent span/chunk KV 和 canonical physical pages。
- **FCGraft**：function ID→KV object、局部 patch/update 和 GPU/DRAM lifecycle。
- **Streaming Knowledge Compilation**：time-evolving content 的 staleness 和受影响 entity 重编译。
- **Code Isn't Memory**：Git working copy 的 Merkle diff 和 repository index 增量更新，但不保存 attention KV。
- **Concordia**：带 version/epoch 的运行时 KV checkpoint coherence，但 version 是系统 checkpoint，不是源码版本。

真正的空白是：

> **将 source-version semantics 与 attention-KV correctness、logical object identity 和 physical tier lifecycle 统一起来。**

### 1.3 对当前项目的直接影响

用户此前已经明确：AST 从来不是主要研究切入点。此次调研进一步支持这一点。

AST、CPG、LSP 或 compiler frontend 只是实现以下能力的一种手段：

- artifact identity；
- source diff mapping；
- dependency extraction；
- invalidation propagation。

论文主线应放在：

```text
repository version graph
+ cache coherence
+ cross-version reuse
+ incremental rematerialization
+ persistent KV object lifecycle
+ workflow-aware physical placement
```

---

## 2. 研究问题与分类

## 2.1 严格研究问题

本轮重点判断是否已有论文实现以下任一能力：

1. Git commit、branch、worktree、repository revision、source version 或 patch epoch 进入 attention-KV cache key。
2. source update 直接触发 attention-KV invalidation。
3. 跨 commit/patch 复用 unchanged attention KV。
4. 根据 source diff 增量修复或 rematerialize KV。
5. dependency graph 决定跨 artifact KV invalidation。
6. 多 branch/worktree 共享 KV，同时维护版本隔离。
7. versioned logical artifact 映射到 GPU/CPU/SSD physical pages。
8. stale KV 能被检测、审计、回滚或升级为 dense fallback。

## 2.2 A/B/C/D 分类

### A：直接先例

repository/source version 明确控制普通 attention KV 的：

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
- rotation；
- steering；
- selective recompute；
- offset reconstruction；
- prompt-layout transformation。

但没有完整 repository version lifecycle。

### C：系统 primitive

提供：

- content-addressed KV；
- persistent KV；
- CPU/GPU/SSD tier；
- modular/non-prefix KV；
- checkpoint/version log；
- message/function object。

但不理解源码版本或 Git DAG。

### D：非 KV 邻近

处理：

- Git/Merkle repository index；
- version-aware RAG；
- embedding；
- graph；
- agent text memory；
- incremental build。

但不保存 Transformer attention K/V tensor。

---

## 3. 总体 Verdict

| 年份 | A 类 | 最接近的工作 | 年度判断 |
| --- | ---: | --- | --- |
| 2024 | 0 | PIE | 已处理 code edit 后 KV splice/relocation，但没有 source-version identity 或 coherence |
| 2025 | 0 | Cache-Craft、EFIM、KVCOMM、MEPIC | 已有 contextual repair、prompt-layout harness 和 content-hash objects，但没有 repository revision lifecycle |
| 2026 | 0 | Leyline、FCGraft、Irminsul、Streaming Knowledge Compilation、Code Isn't Memory | edit directives、function objects、content addressing、evolving content、Git/Merkle index 已分别出现，但尚未统一到 versioned attention-KV system |

最安全的正式结论是：

> **To our knowledge, existing systems separately support mutable-prompt KV repair, content-addressed KV objects, persistent memory tiers, runtime checkpoint versions, or Git/Merkle repository updates; we did not find a system that makes repository source versions first-class attention-KV identities and maintains coherence across source edits, branches, worktrees, dependencies, and memory tiers.**

---

## 4. 2024 年工作

## 4.1 2024 Verdict

2024 年已有一篇直接处理 mutable code prompt 的强 B 类工作：

- PIE：`Let the Code LLM Edit Itself When You Edit the Code`

但本次检索未发现严格 A 类系统。

## 4.2 最相关论文

### PIE：Let the Code LLM Edit Itself When You Edit the Code

- arXiv：[`2407.03157`](https://arxiv.org/abs/2407.03157)
- 首次提交：2024-07-03
- 最新 revision：2025-03-04，v2
- Venue：ICLR 2025
- 代码：[zhenyuhe00/PIE](https://github.com/zhenyuhe00/PIE)
- 分类：**B**

机制：

```text
old prompt:
prefix + old edited span + suffix

new prompt:
same prefix + replacement span + same suffix
```

PIE：

1. 保留 unchanged prefix KV；
2. 重算 replacement span；
3. 根据新位置重旋 suffix Key；
4. 保留 suffix Value。

它首次非常直接地回答了：

> “开发者修改代码后，能否避免重新 prefill 整个代码上下文？”

但其边界同样关键：

- 没有 commit、branch、worktree 或 source version key；
- 没有 content hash 或 artifact identity；
- 没有 dependency invalidation；
- 没有 persistent object catalog；
- 旧 suffix hidden states 已包含旧编辑 span 的因果影响；
- Key relocation 只能修正位置，不能消除旧语义影响；
- suffix Value 同样可能因 prefix edit 而与 dense ground truth 不一致。

因此 PIE 是当前项目必须比较的 edit-repair baseline，但不是 repository versioning 系统。

### CacheBlend

- arXiv：[`2405.16444`](https://arxiv.org/abs/2405.16444)
- 首次提交：2024-05-26
- 最新 revision：2025-04-03，v3
- Venue：EuroSys 2025
- 代码：[YaoJiayi/CacheBlend](https://github.com/YaoJiayi/CacheBlend)
- 分类：**B**

机制：

- 把离线 chunk KV 放到新的组合上下文；
- 测量 high KV deviation tokens；
- 逐层 selective recompute；
- 其余 token 继续复用；
- 与 KV I/O pipeline。

边界：

- 处理的是同一 immutable chunk 在新前序和顺序下的 contextualization；
- 不处理 chunk 内容被修改后的 revision lineage；
- 没有 source invalidation。

### EPIC

- arXiv：[`2410.15332`](https://arxiv.org/abs/2410.15332)
- 首次提交：2024-10-20
- 最新 revision：2025-05-27，v3
- Venue：ICML 2025
- 代码：[DerekHJH/epic](https://github.com/DerekHJH/epic)
- 分类：**C / B 邻近**

机制：

- immutable chunks 独立 compile；
- link 时重算每个非首 chunk 的少量 leading tokens；
- 处理 chunk boundary attention sink。

边界：

- 明确以 immutable chunk 为前提；
- 内容修改后需要重新 compile；
- 没有 revision graph 或 patch repair。

### CachedAttention

- arXiv：[`2403.19708`](https://arxiv.org/abs/2403.19708)
- Venue：USENIX ATC 2024
- 分类：**C**

机制：

- 多轮会话 KV 分层保存；
- 位置解耦；
- context overflow 后截断旧 KV；
- reload 时重新注入位置。

边界：

- 主要处理 append-only session 和头部截断；
- 没有 source version。

### RAGCache

- arXiv：[`2404.12457`](https://arxiv.org/abs/2404.12457)
- 分类：**C**

机制：

- ordered document IDs 构成 knowledge tree；
- GPU/host hierarchy；
- PGDSF 管理 KV。

边界：

- document ID 不等于 document revision；
- 同 ID 内容变化后的 cache validity 未定义。

### MemServe

- arXiv：[`2406.17565`](https://arxiv.org/abs/2406.17565)
- 分类：**C**

机制：

- token-ID radix tree；
- `insert/match/delete/evict/transfer`；
- 全局 prompt tree 记录实例位置。

边界：

- TTL 处理的是控制面与本地 eviction 的陈旧；
- 不是 source-version coherence。

### Mooncake

- arXiv：[`2407.00079`](https://arxiv.org/abs/2407.00079)
- Venue：FAST 2025
- 代码：[kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)
- 分类：**C**

机制：

```text
block_key = Hash(previous_hash || token_block)
```

支持：

- exact content prefix identity；
- CPU/DRAM/SSD/RDMA；
- 热点复制；
- 调度。

边界：

- 内容变化导致当前 block 及后续 hash miss；
- 这是 exact-prefix content identity；
- 不是 source revision lineage 或跨版本 repair。

### ChunkAttention

- arXiv：[`2402.15220`](https://arxiv.org/abs/2402.15220)
- Venue：ACL 2024
- 代码：[microsoft/chunk-attention](https://github.com/microsoft/chunk-attention)
- 分类：**C**

机制：

- exact shared-prefix chunk tree；
- 请求生命周期 insert/delete。

边界：

- insert/delete 不是 source edit；
- 没有 arbitrary prompt revision。

## 4.3 2024 最危险 Baseline

1. PIE。
2. CacheBlend。
3. EPIC。

物理系统 baseline：

- Mooncake；
- MemServe；
- RAGCache。

---

## 5. 2025 年工作

## 5.1 2025 Verdict

2025 年的工作大幅增强了：

- context-sensitive chunk repair；
- code infilling layout；
- cross-context offset reconstruction；
- persistent KV tiers；
- content-hash KV objects。

但仍未出现严格 A 类 repository-version-aware attention-KV system。

## 5.2 最相关论文

### Cache-Craft

- arXiv：[`2502.15734`](https://arxiv.org/abs/2502.15734)
- 首次提交：2025-02-05
- Venue：SIGMOD 2025
- 分类：**B**

机制：

- 比较旧/新 preceding chunks；
- 估算 Contextualized Cache Impact；
- 根据 inter-attention 选择 top-N token 重算；
- 为 RAG chunk 维护独立 hash→KV-block 映射。

边界：

- 处理 surrounding context 和 chunk order；
- 不处理 chunk 内容 revision；
- 没有 Git/source version。

### KVLink

- arXiv：[`2502.16002`](https://arxiv.org/abs/2502.16002)
- Venue：NeurIPS 2025
- 代码：[UCSB-NLP-Chang/KVLink](https://github.com/UCSB-NLP-Chang/KVLink)
- 分类：**C**

机制：

- 文档独立编码；
- 保存 position-independent KV；
- link tokens 恢复跨文档关系。

边界：

- 文档视为 immutable；
- 需要训练；
- 没有 version invalidation。

### EFIM

- arXiv：[`2505.21889`](https://arxiv.org/abs/2505.21889)
- Venue：Euro-Par 2025
- 代码：[gty111/EFIM](https://github.com/gty111/EFIM)
- 分类：**B**

机制：

- 解决 code infilling 中 prefix 尾部和 suffix 头部增长造成的 cache invalidation；
- 将增量 fragment 移到 prompt 尾部；
- 通过训练恢复 tokenization 和 infilling 能力。

边界：

- 是 prompt layout harness；
- 不处理任意 interior source diff；
- 没有 file/commit identity。

### KVFlow

- arXiv：[`2507.07400`](https://arxiv.org/abs/2507.07400)
- Venue：NeurIPS 2025
- 代码：[PanZaifeng/KVFlow](https://github.com/PanZaifeng/KVFlow)
- 分类：**C**

机制：

- Agent Step Graph；
- `steps-to-execution`；
- GPU eviction；
- CPU→GPU prefetch。

边界：

- exact token prefix；
- workflow state 不等于 source version。

### LMCache

- arXiv：[`2510.09665`](https://arxiv.org/abs/2510.09665)
- 代码：[LMCache/LMCache](https://github.com/LMCache/LMCache)
- 分类：**C**

机制：

- GPU/CPU/disk/remote；
- lookup、move、clear、pin、compress；
- token rolling hash；
- model namespace。

边界：

- 内容变化产生新 token hash；
- 结果是 miss，不是跨版本 repair；
- 没有 source lineage 或 dependency invalidation。

### KVCOMM

- arXiv：[`2510.12872`](https://arxiv.org/abs/2510.12872)
- Venue：NeurIPS 2025
- 代码：[FastMAS/KVCOMM](https://github.com/FastMAS/KVCOMM)
- 分类：**B**

机制：

```text
base KV
+ weighted context ΔKV
+ RoPE relocation
+ shareability gate
+ dense fallback
+ anchor update
```

边界：

- context identity 由 placeholder、agent、embedding、长度和 anchors 表示；
- 不理解 source version 或 patch lineage。

### MEPIC

- arXiv：[`2512.16822`](https://arxiv.org/abs/2512.16822)
- 首次提交：2025-12-18
- 分类：**C**

机制：

- padded token sequence hash；
- canonical paged layout；
- first-block request-specific recompute；
- remaining pages 共享；
- reference count；
- object LRU；
- HBM 和 remote KV。

边界：

- 是最接近 versioned physical KV object store 的 primitive；
- 但 key 仍是 token content hash；
- 修改后形成新对象并重算；
- 没有 repo、commit、dependency 或 supersession relation。

### Prompt Choreography

- arXiv：[`2512.23049`](https://arxiv.org/abs/2512.23049)
- Venue：TACL 2026
- 代码：[tjbai/choreo](https://github.com/tjbai/choreo)
- 分类：**B / C**

机制：

- append-only message KV store；
- message ID 和 parent lists；
- branch、backtrack、reorder；
- dynamic attention mask；
- RoPE reposition。

边界：

- message objects 是 immutable；
- branch 是 workflow/message branch，不是 Git branch；
- 没有修改既有 source artifact 的 coherence。

## 5.3 2025 最危险 Baseline

1. MEPIC。
2. Cache-Craft。
3. EFIM。
4. KVCOMM。
5. LMCache。

联合 baseline：

```text
MEPIC object/page identity
+ Cache-Craft repair
+ KVCOMM context reconstruction
+ LMCache tier
+ KVFlow scheduling
```

---

## 6. 2026 年最新工作

## 6.1 2026 Verdict

2026 年出现了多个非常接近本项目系统 thesis 的工作，但它们仍然分散在不同轴上：

- mutable prompt directives；
- content-addressed spans；
- function-level lifecycle；
- time-evolving knowledge；
- Git/Merkle code index；
- runtime checkpoint coherence。

截至 2026-07-13，本次检索仍未发现严格 A 类统一系统。

## 6.2 最接近的工作

### Irminsul

- arXiv：[`2605.05696`](https://arxiv.org/abs/2605.05696)
- 首次提交：2026-05-07
- 分类：**C**

机制：

- Content-Defined Chunking / Gear hash；
- `xxHash64(chunk)` 内容寻址；
- position-independent MLA cache；
- 命中 unchanged chunk 后重旋小型 RoPE component；
- 首部 carve-out 重算。

边界：

- 最接近跨 edit 保留 unchanged spans；
- 但不了解 file、repo、commit 或 artifact lineage；
- xxHash64 只代表当前 content identity；
- 不表达 old version→new version relation；
- 主要机制依赖 MLA。

### Leyline

- arXiv：[`2606.01065`](https://arxiv.org/abs/2606.01065)
- 首次提交：2026-05-31
- 分类：**B**

机制：

```text
Directive = (start, end, replacement, mode)
```

- `AMORTIZE`：splice replacement，修正 downstream MLA positional component；
- `FORGET`：prefix-trimmed re-prefill。

边界：

- 是当前最明确的 mutable-agent-context harness；
- `AMORTIZE` 有意保留旧 span 对 suffix hidden states 的历史影响；
- 不等价于 dense new prompt；
- 没有 Git/source identity。

### Functional Cache Grafting

- arXiv：[`2606.13097`](https://arxiv.org/abs/2606.13097)
- 首次提交：2026-06-11
- Venue：ICML 2026
- 分类：**B / C**

机制：

- function ID→interface/code/KV；
- function retrieval；
- cache stitching；
- localized patch；
- 成功执行后 update；
- GPU/DRAM placement。

边界：

- 最接近 code-object lifecycle；
- 面向 embodied-agent code policies；
- 没有 Git commit、branch、worktree；
- 没有依赖图；
- 没有多版本并存或 source merge。

### Models Take Notes at Prefill

- arXiv：[`2606.17107`](https://arxiv.org/abs/2606.17107)
- 首次提交：2026-06-14
- 代码：[19PINE-AI/programmable-kv](https://github.com/19PINE-AI/programmable-kv)
- 分类：**B / C**

机制：

- 分析 mutable field 的信息已传播到 downstream aggregator KV；
- append erratum；
- selective note recompute；
- programmable/composable KV notes。

边界：

- 明确认识到“只替换 field KV 不足以忘掉旧内容”；
- erratum 是追加修正，不是删除旧状态；
- 没有 source-version graph。

### KVEraser

- arXiv：[`2606.17034`](https://arxiv.org/abs/2606.17034)
- 分类：**B**

机制：

- 训练 eraser 生成 steering KV；
- 替换被删除 span；
- 继续复用受旧内容污染的后缀。

边界：

- learned approximate counterfactual erasure；
- 不是 source-version consistency；
- 不提供可审计 exactness。

### Streaming Knowledge Compilation

- arXiv：[`2606.09877`](https://arxiv.org/abs/2606.09877)
- 首次提交：2026-06-03
- 分类：**C，最接近 generic content-version lifecycle**

机制：

- timestamped knowledge；
- staleness decay；
- dynamic pins；
- 新内容到达后识别 affected entities；
- 对 affected entity 执行 KV-prefix recompilation；
- 周期性 full recompilation。

边界：

- 面向 evolving wiki/knowledge stream；
- 不是软件仓库；
- 没有 Git DAG；
- affected entity 的处理仍偏完整 prefix recompilation；
- 没有跨 revision delta reuse 或 artifact dependency coherence。

### Code Isn't Memory

- arXiv：[`2606.22417`](https://arxiv.org/abs/2606.22417)
- 首次提交：2026-06-21
- 代码：[TransformerOptimus/supercoder-eval](https://github.com/TransformerOptimus/supercoder-eval)
- 分类：**D**

机制：

- per-repository structural index；
- working-copy Merkle diff；
- source edit 只 re-index affected chunks；
- tree-sitter AST；
- call graph；
- vector/BM25/graph retrieval。

边界：

- 它拥有本项目需要的 repository version/update semantics；
- 但完全不保存 attention KV；
- 是“版本轴有、KV 轴没有”的最强 baseline。

### MORI

- arXiv：[`2606.00866`](https://arxiv.org/abs/2606.00866)
- 分类：**C**

机制：

- coding-agent session；
- GPU/CPU/Waiting tier；
- idleness 和 typed eviction；
- replica affinity。

边界：

- object 粒度是整个 agent session；
- repository base commit 只是 workload setup；
- 不参与 KV identity。

### Concordia

- arXiv：[`2606.23521`](https://arxiv.org/abs/2606.23521)
- 分类：**C**

机制：

- runtime KV checkpoint；
- region ID；
- version；
- epoch；
- dirty pages；
- checksum；
- commit marker；
- base snapshot + delta replay。

边界：

- 证明了 version/epoch/MVCC-like coherence 可用于 KV runtime durability；
- 但 version 是 checkpoint sequence；
- 不是 source repository version。

### Cache Merging as a Convergent Replicated State

- arXiv：[`2607.01308`](https://arxiv.org/abs/2607.01308)
- 首次提交：2026-07-01
- 分类：**C**

机制：

- content-addressed latent fragments；
- K/V bytes hash；
- set-union CvRDT；
- commutative、associative、idempotent merge。

边界：

- 解决 fragment delivery、重复与 merge order；
- 不表达 source revision supersession、delete 或 invalidation。

## 6.3 最近 30/90 天需要持续跟踪的工作

### 最近 30 天

- Models Take Notes at Prefill；
- KVEraser；
- Code Isn't Memory；
- Concordia；
- SmoothAgent；
- Cache Merging as a Convergent Replicated State；
- HYPIC。

### 最近 90 天

- CodeComp；
- Irminsul；
- MORI；
- Leyline；
- Streaming Knowledge Compilation；
- FCGraft；
- MiniPIC。

这些论文中，未来最可能扩展为 A 类的方向是：

1. Streaming Knowledge Compilation 增加 repository/Git semantics；
2. FCGraft 增加 multi-version source/dependency lifecycle；
3. Leyline 增加 source edit compiler 和 correctness protocol；
4. Irminsul/MEPIC 增加 artifact/version catalog；
5. Code Isn't Memory 将其 Merkle index 与 attention KV store 连接。

---

## 7. Closest-Prior-Art Matrix

| 工作 | Source/Git version | Content hash | Mutable edit | KV repair | Persistent KV object | CPU/GPU tier | Dependency invalidation | Attention KV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PIE | — | — | ✓ code edit | ✓ approximate suffix relocation | — | — | — | ✓ |
| CacheBlend | — | △ chunk ID | △ context composition | ✓ selective recompute | △ | △ | — | ✓ |
| Cache-Craft | — | ✓ chunk hash | △ surrounding context | ✓ selective repair | ✓ | △ | — | ✓ |
| KVCOMM | — | △ placeholder/anchor | △ context change | ✓ offset reconstruction | ✓ anchor | △ | — | ✓ |
| MEPIC | — | ✓ token hash | — content treated immutable | △ first-block recompute | ✓ canonical pages | ✓ | — | ✓ |
| Irminsul | — | ✓ CDC + xxHash64 | △ unchanged chunks survive edits | △ positional repair | ✓ | △ | — | ✓ |
| Leyline | — | — | ✓ span replace | ✓ AMORTIZE/FORGET | — | — | — | ✓ |
| FCGraft | — | function ID | ✓ local function patch | ✓ localized patch | ✓ | ✓ | — | ✓ |
| Models Take Notes | — | — | ✓ mutable field | ✓ erratum/note recompute | ✓ notes | △ | — | ✓ |
| Streaming Knowledge Compilation | timestamp/version-like | △ entity identity | ✓ evolving content | full affected-entity recompile | ✓ pins | △ | entity-level | ✓ |
| Code Isn't Memory | ✓ working-copy/Merkle | ✓ | ✓ source edit | — | index objects | persistent index | ✓ affected chunks | — |
| Concordia | runtime checkpoint version | checksum | dirty pages | delta replay | ✓ | ✓ | runtime page-level | ✓ |
| 目标系统 | ✓ commit/branch/worktree | ✓ artifact content | ✓ source patch | exact/repair/rematerialize | ✓ versioned object | ✓ | ✓ source/dependency | ✓ |

这个表揭示了最重要的研究空白：

> **目前最接近的能力被拆散在不同论文中，没有一篇同时覆盖最后一行。**

但“组合所有列”本身仍不足以成为强论文。必须提出一个统一 consistency abstraction，并证明它改变了：

- correctness；
- invalidation granularity；
- cross-version reuse；
- storage cost；
- scheduling decision。

---

## 8. 最准确的系统研究空白

## 8.1 Versioned KV Identity

现有 KV system 通常使用：

```text
token hash
model namespace
chunk ID
message ID
function ID
```

目标系统需要：

```text
KVIdentity {
  repository_id
  source_snapshot
  branch_id
  worktree_epoch
  artifact_id
  source_span
  content_hash
  dependency_version
  context_signature
  model_fingerprint
  position_basis
  numerical_layout
}
```

其中 `source_snapshot` 可以是：

- Git commit/tree；
- dirty worktree synthetic snapshot；
- patch epoch；
- ephemeral generated-code snapshot。

## 8.2 Repository Version Graph

Git history 不是单一递增版本号，而是 DAG：

```text
branch
merge
rebase
cherry-pick
worktree
dirty patch
```

系统需要判断：

- 两个版本是否共享同一个 unchanged artifact；
- 是否可以共享 canonical base KV；
- context variants 是否仍有效；
- merge 后是否需要重新验证；
- branch-local object 何时可 GC；
- rollback 是否可以恢复旧 physical pages。

## 8.3 Source Diff → KV Invalidation

需要建立：

```text
Git/token/AST diff
-> changed artifact
-> interface/ABI change
-> dependency cone
-> context variants at risk
-> exact invalidation
-> repair candidate
-> rematerialization plan
```

AST 只是这一管道中的一种 diff/dependency implementation。

可以替换或补充为：

- compiler IR；
- LSP symbol graph；
- build graph；
- import graph；
- test trace；
- runtime call trace。

## 8.4 Cross-Version Reuse Lattice

对于 old version \(v_o\) 与 new version \(v_n\)：

```text
R0 exact physical reuse
R1 exact content reuse under identical causal ancestors
R2 position-only relocation
R3 calibrated repair / residual reconstruction
R4 selective rematerialization
R5 full dense recompute
```

系统需要明确：

- 哪一级是 exact；
- 哪一级是 approximate；
- 哪一级需要 probe；
- 哪一级必须 dense fallback。

## 8.5 MVCC-like KV Lifecycle

数据库 MVCC 的类比非常自然：

- logical artifact 有多个 source versions；
- reader 绑定某个 repository snapshot；
- writer 产生新 patch epoch；
- old readers 可继续使用旧 KV pages；
- new readers 不得读取 stale version；
- commit 后更新 visibility；
- rollback 后丢弃 uncommitted variants；
- GC 回收不再被 branch/session 引用的 pages。

但需要注意：

> KV 并不是普通 deterministic row value；它还依赖 causal context 和 model fingerprint。

所以准确概念是：

```text
Versioned Causal KV Materialized View
```

而不只是 “Git blob cache”。

## 8.6 Physical Tier Coherence

同一 logical version 可能存在：

- GPU exact page；
- CPU compressed base；
- SSD quantized snapshot；
- reconstructed GPU variant；
- stale speculative prefetch。

需要维护：

```text
ABSENT
CPU_VALID
LOADING
GPU_EXACT
GPU_APPROX_VERIFIED
DIRTY
STALE
RECOMPUTING
INVALID
GC_PENDING
```

Git/source events 必须能原子地影响这些 states。

---

## 9. 安全与不安全的 Novelty Claim

## 9.1 不安全

- 首个支持代码编辑的 KV Cache。
- 首个 mutable prompt KV system。
- 首个 content-addressed KV store。
- 首个 function-level code KV object。
- 首个 versioned KV system。
- 首个 KV checkpoint version/epoch。
- 首个 Git-aware code index。

这些分别会被 PIE、Leyline、Irminsul/MEPIC、FCGraft、Concordia、Code Isn't Memory 等反驳。

## 9.2 相对安全

> To our knowledge, this is the first repository-version-aware attention-KV system that binds logical cache objects to Git/worktree snapshots, propagates source and dependency changes into KV validity, and manages exact reuse, verified repair, rematerialization, and tier placement under a unified coherence protocol.

需要主动区分：

- PIE/Leyline：mutable prompt repair，但无 repository identity。
- Irminsul/MEPIC：content-addressed objects，但无 version graph。
- FCGraft：function lifecycle，但无 repository/dependency coherence。
- Streaming Knowledge Compilation：evolving content recompilation，但无 source DAG 和 cross-version KV repair。
- Code Isn't Memory：Git/Merkle invalidation，但不存 attention KV。
- Concordia：runtime checkpoint versioning，但不是 source versioning。

---

## 10. 推荐论文 Thesis

### 10.1 主推荐

**RepoKV-MVCC: Versioned Causal KV Materialized Views for Evolving Codebases**

核心论点：

> Coding-agent KV cache cannot be safely keyed by tokens or function IDs alone. It must be bound to a repository snapshot, causal context, and model fingerprint. RepoKV-MVCC turns Git/worktree changes into cache-coherence events and jointly selects exact reuse, verified repair, incremental rematerialization, and physical tier placement.

### 10.2 更系统化的表述

> 软件仓库的 source versions 构成一个 DAG，而不是一条 append-only prompt。我们为 attention KV 引入 repository-snapshot isolation、dependency-aware invalidation 和 MVCC-like physical-page lifecycle，使并发 Architect/Coder/Debugger session 能安全共享 unchanged code KV，同时隔离 dirty worktrees 和 patch epochs。

### 10.3 更聚焦的备选

**PatchMVKV**

- 只研究 dirty worktree 和 patch epoch；
- 避免一开始覆盖 merge/rebase；
- 聚焦 Coder→Debugger→Coder loop；
- 评估 source diff、incremental invalidation 和 rematerialization。

---

## 11. 最值得实现的机制

## 11.1 Repository Snapshot Descriptor

```text
RepositorySnapshot {
  repo_id
  base_commit
  branch
  worktree_id
  patch_epoch
  dirty_tree_hash
}
```

## 11.2 Artifact Version Graph

每个 artifact 维护：

```text
ArtifactVersion {
  artifact_id
  parent_versions[]
  source_snapshot
  content_hash
  interface_hash
  dependency_root
  token_ids
}
```

支持：

- unchanged version aliasing；
- branch fork；
- merge validation；
- rollback；
- GC。

## 11.3 Coherence Protocol

源码事件：

```text
EDIT
COMMIT
CHECKOUT
BRANCH
MERGE
REBASE
ROLLBACK
```

转换成：

```text
KEEP
ALIAS
DIRTY
INVALIDATE
VERIFY
REMATERIALIZE
GC
```

## 11.4 Cross-Version Reuse Planner

对每个 artifact 选择：

```text
exact same-version page
exact cross-version unchanged-content alias
position relocation
PIE/Leyline-style repair
KVCOMM/residual reconstruction
selective suffix rematerialization
dense
```

## 11.5 Stale Audit

每个 approximate path 记录：

- old/new source version；
- edit span；
- causal suffix length；
- repair method；
- probe result；
- dense counterfactual sample；
- downstream test result。

## 11.6 Tier + Version GC

优先保留：

- 被多个 branches 共享的 canonical base；
- 当前 worktree 的 exact objects；
- 即将被 Debugger/Coder 使用的 patch epoch；
- load 明显快于 recompute 的大对象。

优先回收：

- abandoned patch epochs；
- merged 后不再可达的 branch-local variants；
- 低 reuse residual；
- 低置信 approximate snapshots。

---

## 12. 评测设计

## 12.1 Workload

- SWE-bench Verified tasks；
- 真实 Git commit histories；
- 多 branch/worktree replay；
- patch→test→debug loops；
- merge conflict 与 rollback；
- concurrent repository sessions。

## 12.2 Baseline

### Edit/repair

- PIE；
- Leyline；
- CacheBlend；
- Cache-Craft；
- KVCOMM；
- KVEraser；
- Models Take Notes。

### Object/tier

- Irminsul；
- MEPIC；
- MiniPIC；
- LMCache；
- Mooncake；
- FCGraft。

### Version/update but non-KV

- Code Isn't Memory；
- Streaming Knowledge Compilation。

### Runtime coherence

- Concordia。

### End-to-end

- dense prefill；
- SGLang RadixAttention；
- HiCache；
- KVFlow。

## 12.3 指标

### Version correctness

- stale false-hit rate；
- wrong-version page load；
- cross-branch contamination；
- rollback correctness；
- merge/rebase invalidation correctness。

### Reuse

- exact same-version hit；
- exact cross-version alias hit；
- repair acceptance；
- rematerialized tokens；
- dense fallback。

### Performance

- TTFT；
- workflow/session latency；
- H2D bytes；
- recompute FLOPs；
- CPU/GPU/SSD footprint；
- version catalog overhead；
- GC cost。

### Agent correctness

- patch compile；
- tests；
- SWE-bench resolve；
- Debugger→Coder loop count；
- dense-vs-cache output divergence。

## 12.4 关键 Ablation

1. token hash only vs repository snapshot identity。
2. file invalidation vs artifact invalidation。
3. content-only vs dependency-aware invalidation。
4. no version aliasing vs cross-version aliasing。
5. no repair vs PIE/Leyline vs calibrated repair。
6. no MVCC vs snapshot isolation。
7. LRU vs version-aware GC。

---

## 13. Kill Criteria

应缩小或停止 thesis 的条件：

1. 大多数 commit 中 unchanged hot artifacts 太少，cross-version reuse 无法摊销 catalog。
2. dependency invalidation 几乎总扩散到整个工作集。
3. repair path 的 dense fallback 超过约 40%。
4. stale false-hit 无法压到接近 0。
5. repository snapshot lookup 和 version catalog 开销超过节省的 prefill。
6. load 大多数时候不如 recompute。
7. 简单的 token hash + MEPIC/LMCache 已达到相同收益。
8. Code Isn't Memory + dense prefill 的文本检索方案在端到端 agent latency 上已经足够好。

---

## 14. 最终判断

### 已有的

```text
代码编辑后的 KV splice        -> PIE
通用 mutable context directive -> Leyline
context-aware repair           -> CacheBlend / Cache-Craft / KVCOMM
content-addressed KV           -> Irminsul / MEPIC / LMCache
function object lifecycle      -> FCGraft
evolving knowledge recompilation -> Streaming Knowledge Compilation
Git/Merkle incremental index   -> Code Isn't Memory
runtime versioned checkpoint   -> Concordia
```

### 尚未发现完整先例的

```text
Git/source snapshot
+ attention-KV identity
+ dependency-aware invalidation
+ cross-version exact alias / repair
+ MVCC-like branch/worktree isolation
+ physical tier coherence
+ stale audit / rollback
```

因此此次年度调研对项目的结论不是“version-based KV 完全没人做”，而是：

> **相关能力已经在多个方向上迅速逼近，尤其是 2026 年；但 repository source-version semantics 与 attention-KV coherence 之间仍存在明确、可实现、可证伪的系统研究空白。**

下一步不应再扩大 broad prior-art 搜索，而应尽快验证三个决定性问题：

1. 真实 commit/patch trace 中有多少 hot artifact 可以跨版本 exact alias。
2. source/dependency invalidation 能否明显优于 file-level 全失效。
3. version catalog、H2D 和 repair overhead 是否低于节省的 prefill。
