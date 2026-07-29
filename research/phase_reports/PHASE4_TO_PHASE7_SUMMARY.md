# Phase 4 → Phase 7 跨阶段总研究报告

> 报告类型：跨阶段综合研究报告（自包含、可审计）
> 覆盖范围：Phase 4 恢复机制筛选 → Phase 5 调度隔离 → Phase 6 底座与可行性 → Phase 7 集成评测，含 Closeout CL1/CL2/CL3
> 撰写时间：2026-07-28
> 报告状态：`最终权威`
> 最终整体判定：`engineering = VALID` / `r0_mechanism = NEGATIVE` / `w_system_behaviour = INCONCLUSIVE-DESCRIPTIVE` / `publication = READY WITH CAVEATS`

**四份阶段报告**（本报告不重复其细节，只做综合与反转分析）：

| 阶段 | 报告 | 一句话结论 |
| --- | --- | --- |
| Phase 4 | [跨上下文近似 KV 恢复机制（R0–R5）](PHASE4_RECOVERY_METHODS_REPORT.md) | target-only 有收益，single-use combined 为负；body 叙事被 chunk1024 confound 限定 |
| Phase 5 | [Workflow-Aware Cache Scheduling 与 Prefetch 隔离](PHASE5_WORKFLOW_SCHEDULING_REPORT.md) | 只测 exact Radix；S4 相对其它策略的描述性数值分离只在 workflow-only 口径出现 |
| Phase 6 | [Cross-Store Substrate、正确性与容量可行性](PHASE6_CROSS_STORE_SUBSTRATE_REPORT.md) | 底座 `PASS WITH CAVEATS`；自然压力 fallback 未证明 |
| Phase 7 | [集成评测、R0 Ceiling 与 Workflow Scheduler 描述性结果](PHASE7_INTEGRATED_EVALUATION_REPORT.md) | chunk4096 下 R0 = `NEGATIVE`；W = 描述性；publication `READY WITH CAVEATS` |

---

## 0. 引用约定

| 前缀 | 含义 | 绝对根路径 |
| --- | --- | --- |
| `docs:` | 文档仓库（本报告所在仓库） | `/home/chris/Workspaces/code-agent-kvcache` |
| `impl:` | 实现/结果仓库（cross-store-substrate worktree） | `/home/chris/Workspaces/kvcache-research/worktrees/cross-store-substrate` |
| `wt:<name>:` | Phase4 各机制的独立 worktree | `/home/chris/Workspaces/kvcache-research/worktrees/<name>` |

状态标签：`最终权威` / `历史/已被替代` / `diagnostic/proxy`。

---

## 1. 整体实验动机与最初的假设

### 1.1 系统层动机

本项目的目标不是复刻某一篇论文，而是在 SGLang 上构建一个 **Codebase-aware、workflow-aware、cross-context、CPU/GPU 分层的 KV Cache 系统**（`docs:research/RESEARCH_SYNTHESIS.md` 核心结论）：

| 组件 | 职责边界（严格） |
| --- | --- |
| **KVFlow** | workflow-aware cache priority、eviction、CPU backup、prefetch 与 scheduling |
| **KVCOMM**（`2510.12872`） | base KV、context-dependent offset、RoPE relocation、anchor interpolation、dense fallback |
| **AST / 代码索引** | 决定 codebase 如何切分、标注、检索、失效并映射到物理 KV pages；**是结构索引与辅助 gating 信号，不替代 embedding distance** |
| **固定 workflow** | `Architect -> Coder -> Debugger`（Debugger 失败后可条件返回 Coder），提供稳定的未来执行顺序，使 priority 与预取可预测 |

系统的最终形态更接近「位于 Coding Agent 之下的**预计算 Codebase latent memory**」，而不只是传统的相同 prompt prefix cache。经 novelty 调研后，论文主线被进一步收窄为 **versioned causal KV materialized views / RepoKV-MVCC**。

**必须遵守的表述边界**（贯穿全部四个阶段）：

1. 「可变编码」**不是 KVCOMM 原文术语**；不得把 delta compression、AST index 或 SGLang HiCache 写成论文已有能力。
2. 超大 Codebase 必须按 **artifact / AST span** 预计算和索引，**不得描述为单一连续 KV Cache**；全库建立完整**逻辑**索引，但**物理** KV 必须以 hotset 为主、按需惰性物化。
3. 实现顺序必须优先保证正确性：**exact cache → 受控 KVCOMM reconstruction → dense fallback**。

### 1.2 最初想测什么

从工程角度，本轮实验（Phase4–7）最初要回答的是一个**朴素而具体**的问题：

> 在一个固定的 Coding Agent workflow 中，同一段代码 body 会被不同 role / 不同 header / 不同 causal context 反复读取。exact prefix cache 对此无能为力，只能重新 dense prefill。**能否用受控的 KV 恢复替代一部分 dense prefill？每条路径真正要付出多少额外成本？**

（`docs:research/PHASE4_STAGE_REPORT_SLIDES.md` §1，该节文字未被后续结果推翻）

以及配套的第二个问题：

> 在 cache 容量不足时，**用 workflow 未来执行距离驱动的 priority 能否显著优于 LRU**？强制 prefetch 是帮忙还是制造 churn？

### 1.3 初始假设（后续被逐条检验）

| 编号 | 初始假设 | 最终处置 |
| --- | --- | --- |
| H1 | 跨上下文 KV 恢复能显著降低目标请求 TTFT | **部分成立**：只在 target-only 口径与 chunk1024 配置下成立；chunk4096 下 ceiling 为负 |
| H2 | body 越长收益越大，存在明确的 crossover 点 | **被 confound 限定**：在 body768/1024 上观察到显著的 coupled chunk/max-prefill effect，现有双变量实验不能宣布 body 或 chunk 的全局主导性 |
| H3 | 修复更多 token（更大 repair budget）能换来更好的质量/性能折中 | **被推翻（性能侧）**：R2 的 1% 快于 5%/15%/30%；EPIC 的 leading-k 在 request-path 口径下是净成本 |
| H4 | 恢复出的 KV 在保守输出门槛下可用 | **未通过**：CL1 冻结规则下 `practical family = NONE` |
| H5 | workflow-aware priority 在高压下显著优于 LRU | **被口径与证据强度限定**：workflow-only 下出现描述性分离；all-reusable 下四策略接近，restart 不足以排序 |
| H6 | 强制 prefetch 能提前把对象搬回 GPU 从而降低 TTFT | **本 canary 未支持**：P1–P3 未观察到稳定 mean 改善；host 大于工作集、同步 H2D 与 restart 不足阻止一般化 |
| H7 | exact 与 approximate 对象可以在同一 GPU 预算下安全共存 | **成立（工程层）**：Phase6 双向 pressure 实测，`PASS WITH CAVEATS` |
| H8 | Phase7 R0 的收益能通过多次复用摊销 | **未观察到**：chunk4096 下 N≤8 全部 `>8/not_observed`；不外推到未执行的其它恢复机制 |

---

## 2. 架构与研究问题的演变

### 2.1 演化主线

```text
Phase 4 ── 单机制、单请求、target-only 上限
   │        「哪条恢复路径最快？」
   │        [审计] causal-key 缺陷 → corrected rerun
   │        [审计] 四本账口径 → single-use combined 转负
   ▼
Phase 5 ── 隔离调度，刻意不用恢复
   │        「workflow priority 能否打败 LRU？」
   │        [审计] 分母 → workflow-only vs all-reusable
   ▼
Phase 6 ── 底座、正确性、容量可行性（禁止性能声称）
   │        「exact 与 approximate 能否在同一预算下安全竞争？」
   │        [审计] 自我驱逐 P0、provisional 泄漏、采样间隔、fallback taxonomy
   ▼
Phase 7 ── 预注册、集成、双模型 review、字节级 provenance
            「在正确的 chunk 配置下，机制到底有没有收益？」
            [结论] R0 = NEGATIVE；W = DESCRIPTIVE
```

### 2.2 研究问题的四次收窄

| 阶段 | 核心研究问题 | 相对上一阶段的收窄 |
| --- | --- | --- |
| Phase 4 | 六条恢复机制在统一压力下的 TTFT 与额外成本 | — |
| Phase 5 | 在**排除恢复**的前提下，scheduler policy 的独立价值 | 从「机制 + 调度混合」收窄为「纯 exact 调度隔离」 |
| Phase 6 | 底座能否让两类对象**安全竞争**；哪些配置**根本不可达** | 从「性能比较」收窄为「正确性与可行性」，并**显式禁止性能声称** |
| Phase 7 | 在 primary chunk=4096 下，R0 ceiling 是否过预注册 MDE；W 的系统行为 | 从「五路径 × 五策略」收窄为「R0 ceiling + S0/S4 + 两个 synthetic footprint diagnostic」 |

### 2.3 架构演变

| 阶段 | 架构形态 |
| --- | --- |
| Phase 4 | 每条机制一个独立 worktree（`raw-rope` / `epic-legolink` / `cacheblend` / `cachecraft` / `kvcomm` / `cachetune`），共享 common core 与统一 pressure contract |
| Phase 5 | 单 worktree（`scheduler-policies` 血统），独立 priority metadata 链，不复用 `Req.priority` |
| Phase 6 | 合并到 `cross-store-substrate`：新增 `mem_cache/cross_store/`（1073 行）——对象 DAG、预算、策略序、原子 allocator、coordinator |
| Phase 7 | 同一 worktree + 三个受授权门约束的 runner + 离线 consolidator + 版本化 review/evidence + `RESULT_MANIFEST` 治理层 |

### 2.4 治理成熟度的演变（这是本项目最显著的进步之一）

| 阶段 | 治理形态 | 典型失效 |
| --- | --- | --- |
| Phase 4 | 结果写进 slides / HANDOFF；无机器可读 supersede 索引 | `1.14x` / `1.04x` 被推翻后仍在多处传播 |
| Phase 5 | 同上；Closeout 项靠人工勾选 | FINDING-GAP-1：CL3 被认为完成但从未执行 |
| Phase 6 | 引入 `RESULT_MANIFEST.json` + `build_result_manifest.py --check`；`.gitignore` 排除的 JSONL 被强制纳入 | 首版 manifest 32 项中只有 29 项可验证 |
| Phase 7 | 预注册 manifest（rev6→rev12）+ design payload hash + code pin/execution envelope 分层 + byte-frozen plan + 版本化 CPU/review evidence + 自哈希 compact/summary + `88/88` 且 `known_gaps=[]` | 无开放 P0/P1 |

---

## 3. 逐阶段综合（实验 → 修复 → lesson → 结论）

> 本节只做**跨阶段的综合与对照**；逐条细节见四份阶段报告。

### 3.1 Phase 4 — 恢复机制筛选

| 维度 | 内容 |
| --- | --- |
| 实验 | 五条机制（R0/R1/R2/R4/R5）在统一 header/body/rho contract 下的 OAT 切片；R3 defer（无 GPU 结果） |
| 最强正面结果 | R2 corrected target-only `1.659x`（body1024）/ `2.044x`（body2048）；R5 `1.614x` / `1.978x`；R4 `1.37x` / `1.76x`；R1-k0 `1.70x` / `2.07x` |
| 最强负面结果 | corrected single-use：adapter-combined `0.406–0.449x`，request-path `0.433–0.527x`，full-lifecycle `0.246–0.327x` |
| 关键修复 | causal-key ground-truth 缺陷（fresh registration 只用 "header + current chunk"）；eviction-aware allocation；RoPE resolver；gauge 滞后；header seed 多匹配 |
| 关键 lesson | 只报 target-only 会得到**方向性错误**的结论；配置未披露会制造伪机制结论 |
| 结论 | 恢复能降低目标请求 TTFT，但**单次使用是净亏损**；机制之间不可排序 |

### 3.2 Phase 5 — 调度隔离

| 维度 | 内容 |
| --- | --- |
| 实验 | S0–S4 × rho{1.1,1.5,2,3}；prefetch S4 × P0–P3 × rho{1.5,2,3}；restart validation（8 runs）；CL3 零 GPU 重算（40 cell） |
| 最强正面结果 | workflow-only 口径：S4 = `1.4568 / 1.3210 / 1.1484 / 1.1468`（rho1.1/1.5/2/3），S1–S3 在 rho≥2 掉到 ≈`1.00` |
| 最强负面结果 | all-reusable 口径：S1–S4 = `1.087–1.187`，**四策略几乎不可区分**；prefetch paired = `0.9885–1.0038` |
| 关键修复 | CL3 从未执行（FINDING-GAP-1）；hit fraction 改 per-request clamp；prefetch 对照臂改为同策略 P0；S2 命名从「Belady oracle 上界」改为「Belady-style next-request-ordinal oracle」 |
| 关键 lesson | **分母决定结论**；零 GPU 重算是高性价比审计手段；命名即声称 |
| 结论 | S4 + P0 为保守默认；workflow-only 下出现 S4 的描述性数值分离，all-reusable 下顺序不稳定，现有 restart 不足以排序 |

### 3.3 Phase 6 — 底座与可行性

| 维度 | 内容 |
| --- | --- |
| 实验 | P6-0 合同冻结；P6-4 fixed40 capacity pilot（5 cell × 5 profile）；P6-H host roundtrip + same-context canary；P6-F fault-injected fallback canary（v1→v3）；诊断 C/D |
| 最强正面结果 | 双向 pressure `47.5 GB` / `58.8 GB`；`r1_like_k32` 在三个 S4 cell 可达；P6-H 8-token 输出与 dense 完全一致；P6-F 注入下 fallback 正确且输出匹配 |
| 最强负面结果 | 每个 cell 顶层均 `diagnostic-unavailable`（因 `r4_like` 5x 不可达）；S0/rho2 与 S4/rho3 真实容量不可达；自然压力 fallback 未证明 |
| 关键修复 | **P0 请求自身 prefix 未加锁导致自我覆写**；provisional slot 泄漏（三处清理路径）；stale victim；SWA release metadata；fallback taxonomy 误标；`.gitignore` 排除 JSONL；采样间隔 0.4s→0.05s |
| 关键 lesson | 采样间隔必须短于分配动态；`num_used_tokens` 不可与 store gauge 相加；cell 级与 profile 级状态必须分述；fault-injected ≠ natural |
| 结论 | `PASS WITH CAVEATS`；`RESULT_MANIFEST 48/48`；`phase7_authorized=false` |

### 3.4 Phase 7 — 集成评测

| 维度 | 内容 |
| --- | --- |
| 实验 | 22 primary starts（wave-0 ×2、A8 ×4、chunk1024 sensitivity ×2、W ×12、R4-like ×2）+ 1 evidence correction |
| 最强正面结果 | 工程有效性：22 个 artifact 全部通过 hash/provenance/reset/inactive 校验；S4 把 workflow 对象 hit fraction 维持在 `1.0`（S0 仅 `0.03–0.06`） |
| 最强负面结果 | A8 request-path `0.7723–0.9362x`，N8 full-setup `0.6086–0.6419x`，全部未过 5% MDE → `NEGATIVE`；W latency 比值仅 `1.0021–1.0442x` 且混有 `45.9%–72.1%` dense fallback |
| 关键修复 | runtime 写 repo（P0）；CPU evidence 内容绑定；capacity runner 授权门；plan byte-frozen；p95 命名；launch block 相邻性披露；wave-0 缺 terminal reason 的 correction run |
| 关键 lesson | 缺失的证据无法离线重建；停止规则必须机器执行；`result_git_sha=null` ≠ provenance 缺失 |
| 结论 | `VALID / NEGATIVE / INCONCLUSIVE-DESCRIPTIVE / READY WITH CAVEATS`，open P0/P1 = `0/0`，`RESULT_MANIFEST 88/88` |

---

## 4. 六次关键方法论反转

这是本报告最重要的一节。每一次反转都改变了「什么算作证据」。

### 4.1 反转一：target-only vs request-path / lifecycle

**反转前**：Phase4 早期以 target-only TTFT 作为主指标，得出「长 body 恢复可达 `2.0x`」的结论，并据此宣称 R2 的 single-use combined 也是正的（`1.14x`）。

**反转后**：corrected causal rerun 显示同一批机制在四本账上的表现符号相反：

| 口径 | R2 body2048 | R5 body2048 |
| --- | ---: | ---: |
| target-only | `2.044x` | `1.978x` |
| adapter-combined | `0.407x` | `0.406x` |
| request-path | `0.434x` | `0.433x` |
| full-lifecycle | `0.246x` | `0.246x` |

**方法论含义**：

1. 任何「上限」指标必须显式标注为上限，**禁止称作 end-to-end**；
2. 必须同时公布 setup、adapter、request-path、lifecycle 四层；
3. 摊销必须实测，**禁止公式外推**（Phase7 起改为 `speedup_N` 实际累计 + `>8/not_observed`）。

**触发这次反转的不是新实验，而是一次审计**：C-13/PRC-13 发现 fresh registration 用的是「header + current chunk」而非完整 cumulative causal prefix——这是 ground-truth 构造缺陷。

### 4.2 反转二：chunk1024 confound

**反转前**：Phase4 的核心叙事是「body 长度是主导变量，crossover 在 768–1024 之间」。

**反转后**：CL2 chunk gate 显示同一 body1024 在两种 chunk 下差异巨大：

| chunk | body | dense target | approx target | target-only |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | `297.8ms` | `171.8ms` | `1.733x` |
| 4096 | 1024 | `178.4ms` | `172.8ms` | `1.032x` |

机制：`launch_server` 把 `--max-prefill-tokens` 与 `chunked_prefill_size` 绑定；body1024 的 prompt 长 `1089` token，在 chunk1024 下 dense 必须跨两个 chunk。approximate 臂几乎不变（只需 prefill 最后 1 个 token）。body768（prompt `833`）作为对照组两种配置都 ≈`1.0x`。

**Phase7 的定量确认**：同一 body2048/rho2/S0 在 chunk1024 下 request-path `1.7370x`，在 chunk4096 下 `0.9362x`。

**方法论含义**：

1. 任何 recovery speedup claim 必须同时声明 `chunked_prefill_size` 与 `max_prefill_tokens`，并附一个 prompt 可单 chunk 容纳的对照点；
2. **不得把 Phase4 chunk1024 数值与 Phase7 chunk4096 结果合并统计或排名**（硬性规则）；
3. CL2 本身也必须限定：只测了 body768/1024，且同时改动了两个配置项，正式 status 为 `inconclusive`。

### 4.3 反转三：workflow-only vs all-reusable

**反转前**：Phase5 结论是「S4 是唯一在高压下稳定优于 S0 的策略」。

**反转后**：CL3 用同一批 raw 数据换一个分母，得到完全不同的区分度：

| 分母 | rho2.0 下 S1 / S2 / S3 / S4 |
| --- | --- |
| workflow-only（20 个 workflow 请求） | `0.9959` / `1.0041` / `1.0046` / **`1.1484`** |
| all-reusable（全部可复用请求） | `1.1487` / `1.1554` / `1.1577` / `1.1510` |

**方法论含义**：

1. **分母决定结论**；任何 cache policy 结果都必须声明「哪些请求计入」；
2. 精确表述是「S4 相对 S1–S3 的**独特性**消失，相对 S0 仍有改善」，而不是「S4 优势消失」；
3. 现有独立 restart（多数 cell 为 1 次）**不足以做策略排序**；
4. 禁止「within noise」这类未经检验的统计判断。

### 4.4 反转四：exact / approximate 共预算（从「两个独立 cache」到「一个竞争预算」）

**反转前**：Phase4/5 隐含假设 approximate store 与 exact Radix 是两个互不干扰的空间——Phase4 只在 exact 侧加 pressure，Phase5 完全不碰 approximate。

**反转后**：Phase6 把两者放进同一 device 预算并实测双向驱逐（`47.5 GB` exact→approx，`58.8 GB` approx→exact），随即暴露出三类此前不可见的缺陷：

1. **请求自身 prefix 未加锁**：恢复窗口早于 `add_one_req` 取锁，而 victim 条件恰为 `lock_ref == 0` → 请求驱逐并覆写自己的 KV；
2. **provisional slot 泄漏**：admission 之前分配的 slot 在 reject/abort 路径不释放；
3. **stale victim**：同一轮驱逐内树结构变化使后续 victim 失效。

**方法论含义**：

1. 恢复路径必须复用 exact 路径已有的不变量（prefix 锁）；
2. admission 之前分配的资源必须有明确的所有权协议，覆盖 reject / abort / rematch / teardown 四条路径；
3. 表示多重性（1x / 2x / 5x）是一等容量约束——`r4_like` 的 5x 在本硬件下几乎处处不可达。

### 4.5 反转五：自然 fallback vs fault-injected fallback

**反转前**：Phase6 Exit 首轮提交的证据中声称观察到「12 次 GPU dense fallback」。

**反转后**：Review A 发现这 12 次全部来自 `exact_only` profile 的**普通 exact-cache miss**（runner 仅凭 `cached_tokens < expected` 就打了 `dense_fallback` 标签）；`r4_like` 的 4096 fallback token 实为 registration 容量失败。两条结论均被撤回。

真正可辩护的证据是 P6-F v3 的 **fault-injected** canary：`reservation_failures=1`、`reuse_dense_fallback_requests=1`、`cross_store_reservation_failed_tokens=1024`、输出与 dense 匹配。但 artifact 永久标注 `fault_injected=true; natural_pressure_reachability=false`。

**Phase7 的后续观察**：自然压力下 **0 次** reservation failure；wave-0 `fallback_reachability = {"passed": false, "rounds": 0}`；correction run 显示 40/40 的 approximate 失败原因是 `unsupported <- store_miss`，**不是** reservation failure。

**方法论含义**：

1. **fault-injected 与 natural-pressure 是两种不同的证据强度**，必须显式标注并原样带入下游 claim；
2. `dense_fallback` 必须与 `ordinary_exact_cache_miss` 严格区分；
3. 带 label 的 Prometheus counter 未触发时不输出 series → 只能记 `indirectly_verified`，**不得记显式 0**；
4. artifact 命名即声称（`p6-4-fallback-injection.json` 里没有任何 injection，已更名）。

### 4.6 反转六：从「事后叙述」到「预注册 + 双模型 review + 字节级 provenance」

**反转前（Phase4/5）**：结果写进 slides 与 HANDOFF；被推翻的 `1.14x` / `1.04x` 长期在多份文档中传播；Closeout CL3 被认为完成但从未执行。

**反转后（Phase6/7）**：

| 机制 | 内容 |
| --- | --- |
| 预注册 | Phase7 primary manifest rev6→rev12；早期 design hash 随修订变化，完成 V7 review 的 rev11→rev12 才保持 `50003145…` 不变；MDE 与停止规则在执行前冻结 |
| 授权分层 | Phase6 通过 ≠ Phase7 授权；rev12 才是 `authorized`；三个 runner 共享同一授权门 |
| plan 冻结 | V7 plan blob byte-frozen；activation 移到 `PROJECT.md`/`HANDOFF.md`，避免 design-hash 循环 |
| 双模型 review | Sol 与 Opus 独立 review → 交叉 consolidate → targeted delta → final verify → publication-ready，共 8 份版本化报告 |
| provenance | `RESULT_MANIFEST.json` 提供 file→commit 映射与内容哈希；`--check` 从 `48/48`（Phase6）到 `88/88` 且 `known_gaps=[]`（Phase7） |
| 证据补正 | 缺失的 terminal reason 无法离线重建 → 单独 1 个 correction run，独立计账，原 raw/log 字节不变 |

**方法论含义**：

1. **`result_git_sha=null` 不等于 provenance 缺失**——runner 无法知道将来容纳自己输出的 commit；manifest 才是权威映射；
2. **不要建立 summary ↔ manifest 的双向 hash 循环**（summary 显式记录 `hash_omitted_to_avoid_summary_result_manifest_cycle=true`）；
3. code pin 与 execution envelope 必须分层；
4. 停止规则必须由机器执行（`ES-R0-MDE` 直接省下 8 个 start）。

---

## 5. 最终整体结论

### 5.1 工程层（VALID）

1. 在 SM75 / 8GB / Qwen3-0.6B / Docker 环境下，**exact KV 对象与 approximate KV 对象可以在同一 device 预算下安全竞争**：双向驱逐实测、分配失败可回滚可逆动作、完成的 run 无泄漏无孤儿。
2. 恢复路径的**正确性前提**已被明确并实现：请求自身 prefix 必须加锁；admission 前分配的 slot 必须走 provisional 所有权协议；stale victim 必须隔离刷新重试。
3. 整条证据链可在字节级复核：22 个 raw 自哈希、22 个 compact 自哈希、1 个 summary 自哈希、88 个 artifact 的 file→commit 映射全部验证通过。

### 5.2 机制层（NEGATIVE，限定作用域）

1. **在 chunk4096（primary）下，R0 speed-only ceiling 的 paired request-path speedup 为 `0.7723–0.9362x`，摊销到 N=8 为 `0.6086–0.6419x`，全部未达预注册 5% MDE。** 这是 Phase7 对 R0 的负面结论；R2 未执行，R4-like 不是 KVCOMM，因此不能据此替未测机制下结论。
2. **在 chunk1024 下同配置为 `1.7370x` / `1.1890x`** —— 这只证明 chunk 耦合，不是机制固有收益。
3. **corrected R2/R5 的 single-use request-path 与 lifecycle 均为净亏损**（`0.246–0.527x`）；该结论不外推到未按同一合同测量的其它路径。
4. **R2 与 R5 不可排序**（`1%` vs `8.3%` repair ratio）；两者都是 precomputed oracle，不是 practical candidate。
5. **在冻结的 exact-output promotion 规则下 `practical family = NONE`**，且该结论严格限定于：本模型、合成 prompt 族、SM75、`chunk=max-prefill=1024`、exact-output 不变量。已排除已修复的 eviction-dependent prefix-overwrite 缺陷，但**未证明 context 差异是唯一原因，也未排除 header-dependent 实现缺陷**。

### 5.3 调度层（DESCRIPTIVE）

1. **exact-only 场景（Phase5）**：S4 在 workflow-only 口径下相对 LRU 有 `1.15x–1.46x` 的描述性改善；all-reusable 下 S1–S4 接近（`1.087–1.187`），现有 restart 不支持策略排序。
2. **含 approximate 臂的场景（Phase7 W）**：S4 相对 S0 的 latency 比值仅 `1.0021x–1.0442x`，miss 减少 `-14`/`-10`（rho1.5）与 `+2`/`-10`（rho2.0）；**比较为 seed-matched 但非相邻 launch block，且混有 `45.9%–72.1%` dense fallback**。
3. **prefetch canary 未观察到稳定 mean 改善**：P1–P3 相对同策略 P0 为 `0.9885–1.0038`；P2 p95 增加 `2.91%–4.89%`，P3 增加 `3.75%–3.98%`。该证据不构成一般性能否定。
4. **S4 确实保住了 workflow 对象**（clamped hit fraction `1.0` vs S0 的 `0.03–0.06`），但这一机械效果**没有转化为可声称的延迟收益**。

### 5.4 容量与表示层

| 表示多重性 | profile | 可达性（本硬件） |
| ---: | --- | --- |
| 0x | `exact_only` | Phase6 三个已完成 S4 cell 与 Phase7 rho2 wave-0 可达 |
| 1x | `r0_like` | Phase6 三个已完成 S4 cell 与 Phase7 rho2 wave-0 可达 |
| 1x + 临时 2x | `r1_like_k32` | Phase6 三个已完成 S4 cell 与 Phase7 rho2 wave-0 可达 |
| 2x | `r2_like` | Phase6 三个已完成 S4 cell 与 Phase7 rho2 wave-0 可达；仅为 synthetic footprint |
| **5x** | `r4_like` | Phase6 全部 cell 不可达；Phase7 chunk4096 下 S0 可 registration 但 122 请求只有 12 次 recovery，S4 registration 失败 |

Phase6 的 S0/rho2 与 S4/rho3 是独立测得的容量死亡点，不能用上表的已完成
cell 结果把它们改写为可达。

---

## 6. 从结论导出的决策规则

以下规则是四阶段实验的**可操作产物**，适用于后续任何在本系统上做 KV 复用的工作。

### 6.1 度量与报告规则

| 编号 | 规则 |
| --- | --- |
| D1 | 任何 speedup claim 必须声明 `chunked_prefill_size` 与 `max_prefill_tokens`，并附一个 prompt 可单 chunk 容纳的对照点 |
| D2 | 必须同时报告 target-only、request-path、lifecycle 与 full/incremental 两种摊销口径；禁止只报最有利的一版 |
| D3 | `target-only` 禁止称作 end-to-end |
| D4 | break-even 必须实测；N≤8 未观察到写 `>8/not_observed`，禁止插值或公式外推 |
| D5 | cache policy 结果必须声明分母（workflow-only / all-reusable / full-trace） |
| D6 | p95 比值必须命名为 `ratio_of_marginal_p95s` 并标注 `nonpaired` |
| D7 | 独立复制单元是 server restart；formal repeats、targets、同 trace 内的请求都不是独立样本 |
| D8 | 禁止「within noise」；未做检验只能写「数值上几乎不可区分」 |
| D9 | canary 的判别力必须量化（如 `distinct_output_tokens`） |

### 6.2 证据与遥测规则

| 编号 | 规则 |
| --- | --- |
| E1 | 遥测采样间隔必须短于 workload 的分配动态，并在证据中声明 |
| E2 | 「容量不可达」必须附死亡瞬间的 store gauge 快照 |
| E3 | `num_used_tokens` 已含 approximate store slot，禁止与 store gauge 相加；`nonfree_resident_bytes` 已含 `approx_device_bytes`，禁止相加 |
| E4 | 带 label 的 counter 未触发时记 `indirectly_verified`，不得记显式 0 |
| E5 | `dense_fallback` 与 `ordinary_exact_cache_miss` 必须区分；approximate 失败必须有且只有一个 exclusive terminal reason |
| E6 | cell 级状态与 profile 级状态必须分别陈述 |
| E7 | fault-injected 与 natural-pressure 是不同证据强度，必须显式标注并向下游传递 |
| E8 | 缺失的强制字段无法离线重建，只能通过独立 correction run 补证，且必须独立计账、原 raw 不变 |

### 6.3 治理与 provenance 规则

| 编号 | 规则 |
| --- | --- |
| G1 | 结果 artifact 的 `result_git_sha` 天然为 null；必须另行维护 `RESULT_MANIFEST.json` 并在同一 commit 内 `--check` |
| G2 | 不建立 summary ↔ manifest 的双向 hash 循环 |
| G3 | plan 必须 byte-frozen，activation 记录在别处 |
| G4 | code pin 与 execution envelope 分层；逐 blob 验证 runner 与 manifest |
| G5 | MDE、停止规则与 promotion 规则必须在执行前冻结，看到结果后不得修改 |
| G6 | 阶段通过不自动授权下一阶段 |
| G7 | 被推翻的结论必须同时从叙述性文档中移除或显式标注 `historical/superseded` |
| G8 | runtime 结果不得写入 code worktree |

### 6.4 实现优先级规则（贯穿全项目）

```text
exact cache  →  受控 KVCOMM reconstruction  →  dense fallback
```

即：**正确性优先**。任何近似路径都必须在无法安全执行时可靠退化为 dense，并且退化必须可观测（exclusive terminal reason）。

---

## 7. 可预测事项（跨阶段，可证伪预测与预注册问题）

| 编号 | 预测 | 证伪条件 | 来源 |
| --- | --- | --- | --- |
| X1 | 若 coupled chunk/max-prefill effect 在 body≫chunk 时仍占主要份额，dense 跨更多 chunk 后 R0 speedup 应上升 | 若不升，则 copy/setup 等开销更可能主导；现有数据不预言必然越过 1.0 | Phase4 §8 P4-5、Phase7 §8 P7-1 |
| X2 | device limit 超过 measured live footprint 后，R4-like registration 可能转为可达；显存、带宽与算力需分开 factorial | 预算明确足够仍失败才支持实现缺陷；算力对 R0 的方向无先验 | Phase6 §8 P6-4、Phase7 §8 P7-2 |
| X3 | 现有 workload 下自然 reservation failure 可能继续稀少 | 非零事件只证明 reachability；关闭 caveat 还需 fallback 完成、输出匹配与 clean accounting | Phase6 §8 P6-2、Phase7 §8 P7-5 |
| X4 | 提高 recovery 覆盖率会减少 fallback 混杂，但不保证扩大 policy latency 差异 | 必须在 matched fallback/coverage 下估计策略效应 | Phase5 §8 P5-5、Phase7 §8 P7-3 |
| X5 | A8 restart 1–2 是待补的复制实验，当前 n=1 不给方向先验 | 新 restart 达门槛则更新结论；否则增强 NEGATIVE 的复制支持 | Phase7 §8 P7-4 |
| X6 | 真实 trace 的 reuse-count 分布决定 full-setup 与 incremental-setup 的报告权重 | reuse≤8 时保留 full-setup 主口径；显著高于8时再提高 incremental 权重 | Phase7 §8 P7-6 |
| X7 | 真实 metadata 可能改变 victim 序列和效果量，现有轮转 kind 不给方向先验 | 多 restart 下结果等价则说明 kind 构造在该 workload 中不重要 | Phase5 §8 P5-1 |
| X8 | 固定对象集合的 rho sweep 应预注册比较单峰、单调和无规律模型，而非预先选择形状 | 由新实验选择支持的模型 | Phase5 §8 P5-2 |

---

## 8. 仍未解决的研究问题

### 8.1 机制层

| 编号 | 问题 | 需要什么新证据 |
| --- | --- | --- |
| Q1 | 跨上下文恢复在**大 body / 大模型 / 长 context** 下是否存在真正的收益区间？ | 在 chunk4096（或更大）下，body ≫ chunk 的 A8 ceiling 实测，含 N=1/2/4/8 实测摊销 |
| Q2 | CL1 的输出偏离究竟是「跨上下文近似误差」还是 **header-dependent 实现缺陷**？ | 真正的 `same/different header × low/high pressure` factorial（同一 runner/policy/chunk/env/SHA/重复数），而不是拼接 P6-H 与 CL1 |
| Q3 | R2 与 R5 在**同一 repair ratio** 下是否仍有机制性差异？ | matched repair ratio 对照实验 |
| Q4 | R3 Cache-Craft 的 CCI/CFO 决策在真实系统上是否可行？ | 跨 scheduler / model / attention backend 的改动 + 专项 GPU 验证；当前 SGLang 无通用 selected-token recompute hook |
| Q5 | R4 KVCOMM 的 **neighboring-prefix delta group** 在统一压力矩阵下表现如何？ | 把该子机制纳入统一 header/body/rho 矩阵重测（现仅有更早的小 canary） |
| Q6 | 恢复质量能否用比「输出 token 一致」更有意义的指标衡量？ | logit / top-k 差异、下游 coding correctness 端到端评测 |

### 8.2 调度层

| 编号 | 问题 | 需要什么新证据 |
| --- | --- | --- |
| Q7 | S1/S2/S3/S4 能否被真正排序？ | 每策略 ≥3 次独立 server restart，且在 all-reusable 口径下比较 |
| Q8 | variable-size offline optimum 是多少？S4 距离它有多远？ | 在变尺寸 KV 对象上求解真正的离线最优（CL3 声明为 Phase7 交付物，实际未在主线计算） |
| Q9 | 异步 H2D + host 侧真实压力下，prefetch 是否能转正？ | async H2D 实现 + host 工作集大于 host 容量的 workload |
| Q10 | 真实 object DAG（而非轮转 kind 标签）下，分层策略的价值如何？ | 接入 Phase6 cross-store 真实 metadata 的 S4 重测 |

### 8.3 系统与工程层

| 编号 | 问题 | 需要什么新证据 |
| --- | --- | --- |
| Q11 | 自然压力下 reservation-failure fallback 是否可达？ | 构造能自然触发 reservation failure 的 workload（Phase6 fault-injected + Phase7 0 次自然观察，仍未证明） |
| Q12 | exact–host / HiCache 统一后行为如何？ | 实现统一并重跑 P6-H 级别的 canary（当前 `hicache_tier_exercised=false`） |
| Q13 | 并发 / 多租户下的调度与恢复干扰？ | 全程未测（Phase4–7 均为串行请求） |
| Q14 | 在真实 repository 上，按 artifact / AST span 预计算的索引与失效协议是否成立？ | 真实仓库 + source/dependency invalidation + branch/worktree isolation 的端到端实验 |

---

## 9. 明确没有做的事 / 明确不能声称的事

### 9.1 明确没有做

| 类别 | 未做项 |
| --- | --- |
| 机制 | R3 Cache-Craft 的真实 GPU 结果；R2/R5 matched repair ratio 对照；R2/R5 实测 N=1/2/4/8 摊销；R0/R1/R4 backfill 到 CL1/CL2 contract；R4 neighboring-prefix delta group 纳入统一矩阵 |
| 配置 | chunk4096 下的 `practical` 重新 qualification；CL2 的 body2048 chunk factorial；P6-4 rho1.1/1.5/3 在 chunk4096 下的重新验证；A8 restart 1–2；rho3 conditional |
| 调度 | S1/S2/S3 的独立 restart；固定对象集合的 rho sweep；variable-size offline optimum；async H2D prefetch；真实 object DAG 上的 S4 |
| 系统 | exact–host / HiCache 统一；host / prefetch / async 轨道（Phase7 预算为 0）；并发/多租户；自然压力 reservation failure |
| 质量 | accuracy / semantic correctness / bitwise KV / logit fidelity；端到端 coding correctness |
| 规模 | 更大模型、更长 context、更强 GPU、真实 repository、真实 agent trace |

### 9.2 明确不能声称（汇总）

1. **不能**声称任何恢复路径是 production-ready 或 practical winner。
2. **不能**发布任何 R0 speedup headline（`headline_speedup_allowed=false`）。
3. **不能**把 `practical=NONE` 写成「跨上下文 KV 恢复普遍不可行」——它是**冻结规则在特定 scope 下**的结论。
4. **不能**声称 CL1 的输出失配「已证明是真实近似误差、不是 bug」；正确措辞是「与预期的跨上下文近似一致，且无法由已修复的压力损坏缺陷解释」。也**不能**把它写成数据损坏。
5. **不能**跨路径排名 R0–R5，**不能**声称 R2 优于 R5 或反之。
6. **不能**把 chunk1024 数值与 chunk4096 结果合并统计或排名。
7. **不能**在任何口径下排序 S1/S2/S3/S4；**不能**把 S2 称为 variable-size offline optimum 或理论上界。
8. **不能**把 Phase5 的 kind 层级结果外推到真实 approximate object DAG。
9. **不能**把 prefetch 矩阵当作性能结论。
10. **不能**把 `r2_like` / `R4-like` 的任何结果归因于 CacheBlend 或 KVCOMM；**R4-like 只是 synthetic 5x footprint proxy**。
11. **不能**声称自然压力下 reservation-failure fallback 可达。
12. **不能**把 P6-H 的 8-token canary 说成 KV 保真或 HiCache qualification。
13. **不能**把 wave-0 的 registration reachability 说成 approximate-recovery success。
14. **不能**据 artifact 的 `result_git_sha` 字段声称 provenance 完整。
15. **不能**声称 W 的 latency 差异是实用收益（预注册规则不允许，且混有大量 dense fallback）。
16. **不能**把「可变编码」、delta compression、AST index 或 SGLang HiCache 写成 KVCOMM 论文已有能力。
17. **不能**把超大 codebase 的 KV 描述为单一连续 KV cache。
18. **不能**用 AST 替代 embedding distance——它是结构索引与辅助 gating 信号。

### 9.3 推进各方向所需的新证据（最小集）

| 方向 | 最小新证据 |
| --- | --- |
| 重新打开「恢复有收益」的可能性 | chunk4096 下 body ≫ chunk 的 A8 ceiling，且 request-path median 在 ≥3 个独立 restart 上稳定 `>1.05` |
| 关闭 header-dependent 缺陷这一替代解释 | 单一 runner/配置下的真 factorial：`same/different header × low/high pressure`，四格数据同源 |
| 允许机制排序 | 全部路径 backfill 到同一 causal/paired/four-ledger/guardrail contract + matched repair ratio |
| 允许策略排序 | 每策略 ≥3 独立 restart，all-reusable 口径，且给出 variable-size offline optimum 参照 |
| 允许 prefetch 进入主结果 | mean 相对 P0 改善 ≥3%、p95 不恶化、wasted/churn bytes 受控、不驱逐更早使用的高价值对象 |
| 关闭自然 fallback caveat | 在无 fault injection 条件下观察到非零 `cross_store_reservation_failed`，关联 dense fallback 完成、输出匹配，并且 pre-flush / post-reset accounting 干净 |
| 支持 repository 级 thesis | 真实仓库 + AST/artifact span 索引 + source/dependency invalidation + branch/worktree isolation 的端到端实验 |

---

## 10. Artifact 与权威索引

### 10.1 四份阶段报告

- [`PHASE4_RECOVERY_METHODS_REPORT.md`](PHASE4_RECOVERY_METHODS_REPORT.md)
- [`PHASE5_WORKFLOW_SCHEDULING_REPORT.md`](PHASE5_WORKFLOW_SCHEDULING_REPORT.md)
- [`PHASE6_CROSS_STORE_SUBSTRATE_REPORT.md`](PHASE6_CROSS_STORE_SUBSTRATE_REPORT.md)
- [`PHASE7_INTEGRATED_EVALUATION_REPORT.md`](PHASE7_INTEGRATED_EVALUATION_REPORT.md)

### 10.2 权威 artifact（跨阶段）

| 阶段 | 权威索引 / disposition | 关键校验 |
| --- | --- | --- |
| Phase 4 | `docs:research/PHASE4_RESULT_MANIFEST.json` | 8 个 artifact 的 `status` + `superseded_cells` |
| Phase 4 corrected | `wt:cacheblend:benchmark/approx_kv/results/phase4-r2/sm75-causal-key-rerun.json`（sha256 `84d28044…`）、`wt:cachetune:benchmark/approx_kv/results/phase4-r5/sm75-causal-key-rerun.json`（sha256 `007099d6…`） | `authoritative_corrected` |
| Phase 4/5 审计 | `docs:CONSOLIDATED_PHASE4_PHASE6_REVIEW.txt`（761 行） | C-01…C-65 / PRC-01…PRC-23 register |
| Phase 5 | `impl:benchmark/approx_kv/results/phase5-scheduler/*.json` | `authoritative_historical` |
| Phase 5 修正 | `impl:benchmark/approx_kv/results/phase6/cl3-phase5-recalculation.json`（`raw_sha256=17f010b7…`） | 分母修正权威 |
| Phase 6 | `impl:benchmark/approx_kv/results/phase6/PHASE6_EXIT_DISPOSITION.json` + `RESULT_MANIFEST.json` | 10 gate；`48/48` |
| Phase 7 | `impl:benchmark/approx_kv/results/phase7/PHASE7_FINAL_DISPOSITION.json`（self `4013f054…`）+ `phase7-consolidated-summary.json`（canonical `9d0aafcd…`）+ `RESULT_MANIFEST.json`（file `6b6b0af1…`） | `88/88`，`known_gaps=[]` |
| 计划 | `docs:IMPLEMENTATION_PLAN_LATEST.md`（V7，byte-frozen，plan commit `c80ec165…`） | plan of record |
| 时间线 | `docs:TRACKING.md`（append-only，不可改写） | 逐轮操作证据 |
| 事实源 | `docs:PROJECT.md` | 项目事实与决策 |
| 研究综合 | `docs:research/RESEARCH_SYNTHESIS.md` | 系统 thesis 与边界 |

### 10.3 共同基础环境与逐阶段差异

| 项目 | 值 |
| --- | --- |
| 执行方式 | **全部在 Docker 内执行** |
| 镜像 digest | `sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781` |
| 模型 / revision | `Qwen/Qwen3-0.6B` / `c1899de289a04d12100db370d81485cdf75e47ca` |
| tokenizer revision | `c1899de289a04d12100db370d81485cdf75e47ca`；chat template = `model-revision-bound` |
| GPU | NVIDIA GeForce RTX 2080 SUPER，SM75（cc 7.5），8192 MiB（`gpu_memory_bytes=8163426304`） |
| driver | `580.173.02` |
| CUDA / Torch / Transformers / Python | `12.9` / `2.9.1+cu129` / `5.12.1` / `3.12.3` |
| 容器参数 | `--runtime=nvidia --gpus all --ipc=host --shm-size=8g --user 1000:1000` |

上述 image/model/GPU 是共同基础，不代表 source/config 完全一致：

- Phase4 各机制使用独立 worktree/source pin；corrected R2 为 `c73c9c5ab…`，
  R5 为 `46d1f85c2…`；
- Phase5 主矩阵 source 为 `5a87166b4…`；
- Phase6 不同 artifact 也有独立 source pin（例如 P6-H `c405343c…`、
  P6-4 `fb284cad…`）；
- Phase7 primary execution pin 为 `81405f42…`，correction pin 为
  `a950ab91…`。

chunk 配置同样不同：Phase4 与 Closeout CL1 为
`chunk = max-prefill = 1024`；Phase6 contract 冻结为 `1024`
（`provisional_worst_case`），P6-H 因 OOM 改用 `4096`；Phase7 primary
为 `4096`，sensitivity 为 `1024`。跨阶段引用必须逐 artifact 核对
source、chunk、policy 和 estimator。

### 10.4 GPU 预算总览（Phase7）

| 项 | 值 |
| --- | ---: |
| primary starts | `22` |
| primary GPU-equivalent hours | `1.310141803888889` |
| primary wall-clock span | `1.528835326388889` h |
| evidence correction start | `1` |
| correction GPU-equivalent hours | `0.09833181611111111` |
| hard cap | `36 starts / 6 GPUh` |
| 实际占硬上限比例 | `0.21835696731481483` |
| 因 `ES-R0-MDE` 跳过 | `8` starts |
| rho3 conditional 未执行 | `1` start |

---

## 11. 结语：这轮研究真正确立了什么

1. **一套可以反驳自己的证据体系。** 本轮最有价值的产物不是某个
   speedup 数字，而是一条能把历史结论逐条收窄、撤回并留痕的证据链：
   R2 single-use combined 从 `1.14x` 更正为 `0.407x`；body-only
   叙事被 coupled chunk/max-prefill effect 限定；S4 的区分度被证明依赖
   分母；所谓「12 次 fallback」被撤回为普通 exact-cache miss，另一个
   独立事实是 Phase7 观察到 0 次自然 reservation failure。

2. **一条清晰且有边界的负面结论。** 在本硬件 / 本模型 / 合成
   prompt 与预注册 primary `chunk=4096` 配置下，**R0 raw KV 复制 +
   位置修正不产生收益**，且摊销到 8 次复用仍不转正。该结论只针对
   Phase7 R0，不替未执行的恢复机制下结论。

3. **一个可复用的正确性底座。** cross-store substrate 明确了恢复路径必须遵守的三条不变量（prefix 锁、provisional 所有权、stale victim 隔离），以及记账不可双计的规则。

4. **一组可操作的决策规则**（§6），覆盖度量、证据、治理三个层面，可直接用于后续任何 KV 复用工作。

5. **一份诚实的未完成清单**（§8、§9），标明了每个方向需要什么新证据才能推进，而不是留下模糊的「future work」。

**Phase 8 不会自动触发；若继续，必须另行版本化计划并获得明确授权。**
