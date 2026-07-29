# Phase 4 正式研究报告：跨上下文近似 KV 恢复机制（R0–R5）

> 报告类型：正式阶段研究报告（自包含、可审计）
> 覆盖阶段：Phase 4（R0/R1/R2/R3/R4/R5 恢复机制筛选）+ 直接修正 Phase 4 结论的 Closeout CL1/CL2
> 撰写时间：2026-07-28
> 报告状态：`最终权威`（叙事层）；具体数值的权威性以本文逐条状态标签为准
> 关联报告：[Phase5](PHASE5_WORKFLOW_SCHEDULING_REPORT.md)｜[Phase6](PHASE6_CROSS_STORE_SUBSTRATE_REPORT.md)｜[Phase7](PHASE7_INTEGRATED_EVALUATION_REPORT.md)｜[跨阶段总报告](PHASE4_TO_PHASE7_SUMMARY.md)

---

## 0. 引用约定

本报告全部证据引用使用相对路径，并区分两个仓库：

| 前缀 | 含义 | 绝对根路径 |
| --- | --- | --- |
| `docs:` | 文档仓库（本报告所在仓库） | `/home/chris/Workspaces/code-agent-kvcache` |
| `impl:` | 实现/结果仓库（cross-store-substrate worktree） | `/home/chris/Workspaces/kvcache-research/worktrees/cross-store-substrate` |
| `wt:<name>:` | Phase4 各机制的独立 worktree | `/home/chris/Workspaces/kvcache-research/worktrees/<name>` |

对已经只存在于 git 历史中的 artifact，同时给出 `commit + 仓库内相对路径`，可用 `git show <sha>:<path>` 复核。

本报告使用三类状态标签：

| 标签 | 含义 |
| --- | --- |
| `最终权威` | 当前可直接引用的结论/数值 |
| `历史/已被替代` | 只能作为演进过程引用，不得作为当前结论 |
| `diagnostic/proxy` | 诊断或代理性证据，不构成性能或机制声称 |

---

## 1. 文档定位、证据状态与 Executive Summary

### 1.1 文档定位

本报告是 Phase 4 的正式研究报告，目标是把「六条跨上下文 KV 恢复路径的 SM75 对照实验」完整、可审计地固化下来，包括：原始实验、被推翻的结论、corrected rerun、以及后续 Closeout 阶段对 Phase 4 叙事的反向修正。

本报告**取代** `docs:research/PHASE4_STAGE_REPORT_SLIDES.md` 中的「结果总览」「跨路径观察」等数值章节的结论地位。旧 slides 仅在术语/动机说明上仍可引用。

### 1.2 证据状态总览

| 证据源 | 状态 | 说明 |
| --- | --- | --- |
| `docs:research/PHASE4_RESULT_MANIFEST.json` | `最终权威` | 唯一机器可读的 artifact supersede 索引，逐 artifact 给出 `status` 与 `superseded_cells` |
| `docs:CONSOLIDATED_PHASE4_PHASE6_REVIEW.txt` | `最终权威` | corrected R2/R5 rerun 的双模型审计全文 |
| `docs:IMPLEMENTATION_PLAN_LATEST.md` §2.2/§5/§15.2 | `最终权威` | V7 对 Phase4 结论的最新收窄措辞（byte-frozen） |
| `docs:TRACKING.md` | `最终权威`（时间线） | 不可改写的逐轮操作证据 |
| `wt:cacheblend:benchmark/approx_kv/results/phase4-r2/sm75-causal-key-rerun.json` | `最终权威` | R2 corrected 数值 |
| `wt:cachetune:benchmark/approx_kv/results/phase4-r5/sm75-causal-key-rerun.json` | `最终权威` | R5 corrected 数值 |
| `wt:kvcomm:benchmark/approx_kv/results/phase4-r4/sm75-unified-pressure.json` | `authoritative_historical_diagnostic` | R4 真实 KVCOMM 部分复现，未纳入 causal 修复范围 |
| `wt:epic-legolink:benchmark/approx_kv/results/phase4-r1/sm75-eviction-pressure.json` | `authoritative_historical` | R1 压力 OAT 切片 |
| `wt:epic-legolink:benchmark/approx_kv/results/phase4-r1/sm75-inrequest-matrix.json` | `historical_mechanism_only` | 无 eviction pressure、旧代码 SHA，只证明机制曾被验证 |
| `wt:cacheblend/cachetune:.../sm75-unified-pressure.json` | `historical_oat` | body1024/2048@header64@rho2 关键 cell 已被 corrected rerun 取代 |
| `docs:research/PHASE4_STAGE_REPORT_SLIDES.md` 数值表 | `历史/已被替代` | pre-correction 快照 |
| `docs:HANDOFF.md` 尾部 worktree 状态表 | `历史/已被替代` | 未随 corrected rerun 更新 |

### 1.3 Executive Summary

1. **Phase 4 的核心问题是「能否用受控 KV 恢复替代一部分 dense prefill，以及每条路径真正要付出多少额外成本」**，只测请求可跑通与 TTFT，不涉及 accuracy 或输出等价（`docs:research/PHASE4_STAGE_REPORT_SLIDES.md` §1，该节文字未被推翻）。

2. **五条恢复机制（R0/R1/R2/R4/R5）完成了统一 SM75 pressure 对照；R3 Cache-Craft 被 defer，从未产生真实 GPU 结果。** R3 的阻塞点是 SGLang 上不存在通用 selected-token recompute hook，`schedule_batch.py` 对任何 `approx_kv_metadata` 无条件走通用 raw-copy 路径（`wt:cachecraft` 完成态 `d1110066a`，runner 保留 exit code 3）。

3. **`target-only` 口径下的恢复收益成立且被 corrected rerun 确认；`single-use combined` 的正收益被彻底推翻。** R2 body2048 旧值 `1.14x` → 新值 `0.407x`；R5 body2048 旧值 `1.04x` → 新值 `0.406x`。

4. **corrected R2/R5 是 Phase 4 唯一可引用的 R2/R5 数值**（`最终权威`）：

   | 路径 | body | chunk/max-prefill | target-only | adapter-combined | request-path | full-lifecycle |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: |
   | R2 | 1024 | `1024/1024` | `1.659x` | `0.441x` | `0.526x` | `0.324x` |
   | R2 | 2048 | `1024/1024` | `2.044x` | `0.407x` | `0.434x` | `0.246x` |
   | R5 | 1024 | `1024/1024` | `1.614x` | `0.449x` | `0.527x` | `0.327x` |
   | R5 | 2048 | `1024/1024` | `1.978x` | `0.406x` | `0.433x` | `0.246x` |

5. **R2 与 R5 不可按现有数据排序。** 两者分别在 repair ratio `1%` 与 `8.3%` 上运行，处于同一条「speed vs repair ratio」曲线的不同点；把 target 差异写成机制优劣是无效推断（PRC-16 撤回）。

6. **Phase 4 关于 body 长度的核心叙事被 CL2 chunk1024 confound 严重限定。** 在 `chunked_prefill_size = max_prefill_tokens = 1024` 下 body1024 的 `1.733x` target-only，在 chunk4096 下降到 `1.032x`；主要原因是 dense 基线在小 chunk 下被迫跨两个 prefill chunk 而被「惩罚」，不是恢复机制的固有收益。

7. **CL1 冻结的 exact-output promotion 规则下 `practical family = NONE`。** 该结论完全由 correctness guardrail 决定，性能条件全部满足；作用域严格限定为**本模型、合成 prompt 族、SM75、chunk=1024、冻结的 exact-output promotion 规则**。已排除已修复的 eviction-dependent prefix-overwrite 缺陷，但**未证明 context 差异是唯一原因，也未排除 header-dependent 实现缺陷**。

8. **R4 是本项目唯一一次真实 KVCOMM 部分复现**（canonical base + anchor + context-delta multi-anchor interpolation）。Phase6/Phase7 中出现的 `R4-like` 只是 synthetic 5x footprint proxy，**绝不能写成 KVCOMM 执行**。

---

## 2. Phase 4 动机、研究问题、冻结假设与非目标

### 2.1 动机

Coding Agent 的固定 workflow（`Architect -> Coder -> Debugger`）会反复读取同一段代码，但 role、header 与 causal context 一直在变。exact prefix cache 无法复用这类「内容相同、前置上下文不同」的 body KV，只能重新 dense prefill；body 越长重复计算越贵，working set 超出 cache 后真实 eviction 会继续推高 TTFT（`docs:research/PHASE4_STAGE_REPORT_SLIDES.md` §1）。

因此 Phase 4 在 Phase1–3 common core 冻结之后，为六条恢复机制建立**统一的 SM75 pressure 对照**（`docs:TRACKING.md:710`「纠正 Phase 4/5 完成口径与实验顺序」）。

### 2.2 研究问题

| 编号 | 研究问题 | 本报告结论位置 |
| --- | --- | --- |
| RQ4-1 | 受控 KV 恢复能否在真实 eviction 压力下把目标请求 TTFT 降到 dense prefill 以下？ | §7.1 |
| RQ4-2 | 每条路径的真实额外成本（setup / adapter / lifecycle）是多少？单次使用是否还能盈利？ | §4.6、§7.1 |
| RQ4-3 | body 长度、header 长度、pressure(rho) 中哪一个是主导变量？ | §4.2、§4.7、§7.2 |
| RQ4-4 | repair budget（重算 token 比例）越大是否越好？ | §4.3、§7.1 |
| RQ4-5 | 恢复出来的 KV 是否能通过保守的输出一致性门槛？ | §4.8、§7.3 |

### 2.3 冻结假设（contract 轴）

统一 contract（`docs:TRACKING.md:1305`、`impl:benchmark/approx_kv/README.md` Phase4 小节、`docs:IMPLEMENTATION_PLAN_LATEST.md` §5）：

- header ∈ `{0, 32, 64, 128, 256}`（header = 目标请求中位于 body 之前、可 exact match 的 prefix 长度，**不是 attention head 数**）
- body ∈ `{512, 768, 1024, 2048}`；`>512` 时按 `≤512-token` segment 注册 canonical source
- rho（logical demand target）≈ `{0.9, 1.1, 1.5, 2.0, 3.0}`
- `mem_fraction_static = 0.35`
- 调度侧固定：S0 LRU、GPU-only、prefetch off（P0）
- warmup = 1；formal repeats 默认 4，最少 2
- R2 额外轴：repair ratio ∈ `{1%, 5%, 15%, 30%}`
- R5 额外轴：controller ratio `r`，由 roofline 求解，`speed_only` 模式允许 `r=0`

**R0/R1/R2/R4/R5 是五条恢复机制，不是五套独立数据 workload**；R3 defer 不阻塞其余五条的完成判定（`docs:TRACKING.md:1063`）。

### 2.4 非目标（明确不做）

- 不做 accuracy / semantic correctness / 输出等价性声明；输出一致性只作为保守 guardrail。
- 不做 scheduler / eviction policy 比较（属于 Phase 5）。
- 不做跨机制性能排名（R0–R5 未 backfill 到同一 causal/paired/four-ledger/guardrail contract）。
- 不做 production-readiness 声明。
- 不做并发/多租户场景。

---

## 3. 环境、实现范围、方法与测量口径

### 3.1 执行环境（全部实验均在 Docker 内执行）

| 项目 | 值 |
| --- | --- |
| 容器镜像 digest | `sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781` |
| 模型 | `Qwen/Qwen3-0.6B` |
| model / tokenizer revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| GPU | NVIDIA GeForce RTX 2080 SUPER，SM75（compute capability 7.5），8192 MiB（`gpu_memory_bytes=8163426304`） |
| CUDA / Torch / Transformers / Python | `12.9` / `2.9.1+cu129` / `5.12.1` / `3.12.3` |
| 容器参数 | `--runtime=nvidia --gpus all --ipc=host --shm-size=8g --user 1000:1000` |

证据：`wt:epic-legolink:benchmark/approx_kv/results/phase4-r1/sm75-eviction-pressure.json` 的 `image_digest` / `model` / `model_revision` 字段；同 digest 亦见于 `impl:benchmark/approx_kv/results/phase6/RESULT_MANIFEST.json` 的 `environment`。

> 注意：Phase4 全部实验的 `chunked_prefill_size` 与 `max_prefill_tokens` 均为 `1024`（`launch_server` 把 `--max-prefill-tokens` 同步设为 `chunked_prefill_size`）。这一点在原始 Phase4 报告中未被披露，是 CL2 反向修正的直接起因（见 §4.7、§5.5）。

### 3.2 六条路径的定义与真实实现边界

| 路径 | 定义 | 真实实现边界（诚实阻塞点） | worktree | 关键 commit |
| --- | --- | --- | --- | --- |
| **R0 Raw+RoPE** | 复制 body K/V，只做符号化 RoPE 位置修正；speed-only 上界 | 无 context-dependent repair；**显式非忠实 KVCOMM 复现**；`raw-rope` 独立 plugin 化（`RawRoPERecoveryPlugin`），经 `manager.plugins` registry 派发 | `raw-rope` | `61c39791e`（`docs:TRACKING.md:798`） |
| **R1 EPIC/LegoLink** | 固定 leading-k tokens 逐层重算，其余 body 复用 | 服务器接线含明确记录的阻塞点；k32 为选定实用参数，未在 body2048/rho2 前做完整 k 扫描（由 PRC-17 记为未完成项） | `epic-legolink` | `984bfd873`（`docs:TRACKING.md:812`） |
| **R2 CacheBlend** | 按 HKVD（K deviation）打分选择约 `1%/5%/15%/30%` body token 做 batched 逐层 recompute | **只用 precomputed fresh-KV adapter**，不是通用 ModelRunner selected-token inline hook（该 hook 在 SGLang 上不存在）；`build_plan` 保留 dense-only 保守协议 | `cacheblend` | 首次 `91874f18b`；GPU 矩阵 `e6dd5eab3`；corrected rerun impl `c73c9c5ab` / result `e36f1529b`；closeout `ce55860a9` |
| **R3 Cache-Craft** | 按 CCI(Eq.11)/CFO(Eq.12) 决定 direct/partial/full recompute | **Deferred，无任何真实 GPU 结果**：scheduler 无 dispatch；`TARGET_VERIFY` 仅 spec-decode 内部可达；生产融合 attention 不物化完整注意力矩阵；仅 CPU-only 48/64 测试通过 | `cachecraft` | `d1110066a`（保留 CPU 证据 + blocked runner，exit code 3） |
| **R4 KVCOMM** | canonical base + anchor + context-delta multi-anchor interpolation 重建目标 KV | 统一 header/body/rho 矩阵版本**省略 neighboring-prefix delta group**（该子机制仅由更早的小 canary 证明）；长 body 使用多个独立 512-token placeholder pool，非单一连续 canonical source | `kvcomm` | `6f709a739`(runner)/`ec015cae3`(result)/`562fce6f5`(校验)；后续 `cd81c3e92` |
| **R5 CacheTune** | 硬件感知 repair controller，roofline `T_layer(r)=max(r·N·t_c,(1-r)·N·t_i)+t_o`，golden-section search 求 `r*` | `speed_only` 模式允许 0% repair floor（**非论文 `r_min=15%` quality floor**）；仍是 precomputed fresh-KV adapter，非频域 token selection / sparse transfer / 多流 overlap / deferred RoPE；实测只有 `r0=0.0829` 一档真实 SM75 canary | `cachetune` | `8acb95e5a`；corrected rerun impl `46d1f85c2` / result `abcedd62b`；closeout `71f15d5d1` |

### 3.3 测量口径（四本账 / four-ledger）

Phase4 及其 corrected rerun 使用四层成本口径（定义见 `docs:IMPLEMENTATION_PLAN_LATEST.md` §5.4）：

| 口径 | 定义 | 用途 |
| --- | --- | --- |
| `target-only` | 只计目标请求本身的 TTFT | 恢复机制的上界收益；**禁止称作 end-to-end** |
| `adapter-combined` | target-only + 该次使用所需的 adapter/fresh 准备成本 | R2/R5 这类需要 per-target 准备的路径的单次成本 |
| `request-path` | `seed_head + target_adapter_preparation + post_pressure_reseed + transfer + target_only` | 真实请求路径成本，Phase7 的预注册 MDE 指标 |
| `recovery-object lifecycle` | `source_preparation + request_path` | 含一次性 source 构建的完整对象生命周期成本 |

派生的摊销口径：

```text
recovery_total_N   = source_preparation + Σ request_path_i   (i = 1..N)
dense_total_N      = Σ request_path_i (matched dense)
speedup_N          = dense_total_N / recovery_total_N
incremental_setup  = recovery_source_preparation - dense_source_materialization
```

### 3.4 rho 的四个口径

`docs:IMPLEMENTATION_PLAN_LATEST.md` §5.6 明确禁止只写 `rho`：

- `rho_logical_demand` = logical reusable working set / configured capacity
- `rho_physical_demand` = 含全部 representation 与 scratch 的请求物理页 / capacity
- `rho_resident` = 采样得到的 `(used + evictable)` / capacity
- `rho_host` = host working set / host capacity

**Phase4 corrected rerun 中 `rho_resident ≈ 0.96–0.99`，与历史 demand/oversubscription rho ≈ 2.1 不是同一个量，二者不可混用。**

---

## 4. 全部实验：矩阵、执行顺序、核心数值、成功/失败/跳过

### 4.1 执行顺序

```text
Phase1-3 common core 冻结
  → R1 EPIC 完整 sweep（body / header / rho）
  → R4 KVCOMM sweep
  → R2 CacheBlend ratio + body sweep
  → R5 CacheTune body sweep
  → R0 Raw+RoPE 代表点（dense 分母复用 R1 测量）
  → R3 Cache-Craft：CPU-only 验证后 defer
  → [审计发现 causal-key 缺陷]
  → R2 / R5 corrected causal-key rerun
  → [Closeout] CL1 screening → 3-restart confirm → P0 修复后 rerun
  → [Closeout] CL2 chunk gate
```

### 4.2 R1 EPIC 完整 sweep — `authoritative_historical`

证据：`docs:TRACKING.md:966-994`；artifact `wt:epic-legolink:benchmark/approx_kv/results/phase4-r1/sm75-eviction-pressure.json`（sha256 `1f2e822f…`）。全部为 chunk1024 条件下的 target-only speedup。

body sweep（header64，rho≈2）：

| body | k0 | k32 |
| ---: | ---: | ---: |
| 512 | `0.96x` | `0.76x` |
| 768 | `1.00x` | `0.83x` |
| 1024 | `1.70x` | `1.53x` |
| 2048 | `2.07x` | `1.98x` |

header sweep（body1024，rho≈2）：

| header | 0 | 32 | 64 | 128 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| k0 | `1.69x` | `1.73x` | `1.69x` | `1.74x` | `1.76x` |
| k32 | `1.46x` | `1.50x` | `1.51x` | `1.53x` | `1.59x` |

rho sweep（body1024，header64）：k0 稳定约 `1.73x`；k32 约 `1.49–1.56x`（pre-target rho `0.924→3.054`，peak rho `1.002→3.132`）。

原始结论「`k>0` 全部负收益只成立于 `body≤512`；crossover 位于 768 与 1024 之间」在 chunk1024 作用域内**仍成立**，但已被 CL2 限定（§4.7）。

### 4.3 R2 CacheBlend ratio / body sweep — `历史/已被替代`（关键 cell）

证据：`docs:TRACKING.md:1077-1092`；artifact `git show e6dd5eab3:benchmark/approx_kv/results/phase4-r2/sm75-unified-pressure.json`。

ratio sweep（body1024 / header64 / rho≈2，target-only latency）：

| repair ratio | target TTFT | 说明 |
| ---: | ---: | --- |
| 1% | `182.26ms`（`1.64x`） | 最快 |
| 5% | `189.04ms` | |
| 15% | `196.69ms` | |
| 30% | `211.86ms` | 最慢 |

**结论：repair budget 不是越大越好；在本实现下 1% 优于 5%/15%/30%。** 该趋势未被 corrected rerun 推翻（corrected rerun 只覆盖 1% 的 body1024/2048@rho2 关键点）。

body sweep（ratio 1%）：body512/768 target-only `<1x`；body1024 target `1.64x`；body2048 target `2.02x`（`486.60ms` vs dense `980.87ms`），**single-use combined `1.14x`（`862.02ms` vs `980.87ms`）→ 该值已被推翻，见 §4.6**。

### 4.4 R4 KVCOMM sweep — `authoritative_historical_diagnostic`

证据：`docs:TRACKING.md:1029-1046`；artifact `wt:kvcomm:benchmark/approx_kv/results/phase4-r4/sm75-unified-pressure.json`（sha256 `5ae9762b…`，branch head `cd81c3e92`）。

| 配置 | 结果 |
| --- | --- |
| body512 / body768 | 慢于 dense |
| body1024，rho≈2 | target-only `1.37x`（dense `299.34ms` vs `218.69ms`）；setup ≈ `1.08s`；break-even ≈ 14 次 reuse |
| body2048，rho≈2 | target-only `1.76x`（dense `980.87ms` vs `558.67ms`）；setup ≈ `2.16s`；break-even ≈ 6 次 reuse |
| body1024，peak rho `1.03–3.11` | 稳定 `1.36–1.38x` |
| header `0 → 256` | speedup `1.30x → 1.46x` |

R4 结构：**每 512-token placeholder = 1 target canonical base + 2 anchor bases + 2 context-delta anchors → 物理 setup footprint ≈ 5x body tokens**。这正是 Phase6/Phase7 中 `R4-like` synthetic 5x footprint proxy 的来源；但 proxy **不执行** KVCOMM 重建。

R4 未纳入 corrected causal rerun 范围（causal 修复只应用于 R2/R5），因此其数值状态为 `authoritative_historical_diagnostic`：可引用，但不得与 corrected R2/R5 并列排名。

### 4.5 R5 CacheTune sweep — `历史/已被替代`（关键 cell）

证据：`docs:TRACKING.md:1313-1326`。target/combined：

| body | target-only | single-use combined |
| ---: | ---: | ---: |
| 512 | `0.94x` | `0.48x` |
| 768 | `0.93x` | `0.44x` |
| 1024 | `1.50x` | `0.76x` |
| 2048 | `1.80x` | `1.04x` |

body1024 主点的机制字段：selected tokens = `85`（executable ratio ≈ `8.3%`），recomputed layers = `27`，cached tokens 严格 `1088`，`0` fallback。

**body2048 的 `1.04x` single-use combined 已被推翻（§4.6）。**

### 4.6 Corrected R2/R5 causal-key rerun — `最终权威`

#### 4.6.1 触发原因

`docs:CONSOLIDATED_PHASE4_PHASE6_REVIEW.txt` C-13 / PRC-13：**R2/R5 的 fresh registration 对长 body 使用的是「header + current chunk」，而不是完整 cumulative target causal prefix**。也就是说，target 侧的「新鲜」KV 并未真实反映它本应依赖的完整因果上下文。这是 ground-truth 构造缺陷，不是记账口径问题，必须修复后重跑。

#### 4.6.2 Provenance

| 项 | R2 | R5 |
| --- | --- | --- |
| impl commit | `c73c9c5ab3ab705996c0ff901314a5fe41e1f8a6` | `46d1f85c22a98b7305b4f3ef299da56c65d2a025` |
| result commit | `e36f1529b838c12a9eb2af7ba4dde91ae9ec124b` | `abcedd62b5a5d801742734e300a5df21e1436737` |
| raw sha256 | `bd6452f7c1c9e34e79e2d7435dae9c350fb89effdb50edeed2f77dc6f819af3a` | `007099d686cc9a1ff24d63182009be421d1f9a1de770d9a783c935a24e81c262` |
| 中央日志 run_id | `phase4-r2-key-rerun-20260725T042040Z` | `phase4-r5-key-rerun-20260725T042651Z` |
| closeout head | `ce55860a9` | `71f15d5d1` |

两次 rerun 均以 `ccdd2023` 身份 push，remote SHA 与本地一致。

#### 4.6.3 数值（唯一可引用的 R2/R5 值）

| 路径 | body | target-only | adapter-combined | request-path | full-lifecycle |
| --- | ---: | ---: | ---: | ---: | ---: |
| R2 | 1024 | `1.659x` | `0.441x` | `0.526x` | `0.324x` |
| R2 | 2048 | `2.044x` | `0.407x` | `0.434x` | `0.246x` |
| R5 | 1024 | `1.614x` | `0.449x` | `0.527x` | `0.327x` |
| R5 | 2048 | `1.978x` | `0.406x` | `0.433x` | `0.246x` |

底层测量值（`summary` 字段，已逐字段核对）：

- R2 body1024：dense target `298.182ms` → cacheblend target `179.756ms`；dense request-path `416.659ms` → `791.742ms`；full-lifecycle `416.659ms` → `1284.551ms`。
- R2 body2048：dense target `981.940ms` → `480.338ms`；request-path `1099.735ms` → `2531.881ms`；full-lifecycle `1099.735ms` → `4461.715ms`。
- R5 body1024：dense target `307.743ms` → cachetune target `190.694ms`；request-path `425.206ms` → `806.649ms`；lifecycle → `1300.484ms`。
- R5 body2048：dense target `988.000ms` → `499.584ms`；request-path `1105.076ms` → `2553.519ms`；lifecycle → `4485.257ms`。
- 两者 `first_token_match_rate = 1.0`，`formal_samples_per_arm = 6`，`all_rounds_observed_eviction = true`。

#### 4.6.4 被明确推翻的旧结论

| 旧结论 | 新结论 |
| --- | --- |
| R2 body2048 single-use combined `1.14x` | `0.407x`（**推翻**） |
| R5 body2048 single-use combined `1.04x` | `0.406x`（**推翻**） |
| R2/R5 target-only 收益 | **确认，未被推翻** |
| 「R2 快于 R5，说明机制更优」 | **撤回（PRC-16）**：差异来自 `1%` vs `8.3%` repair ratio 配置 |

#### 4.6.5 R2/R5 不可排序的定量理由

R2(1%) 与 R5(8.3%) 处在同一条「speed vs repair ratio」曲线上：R5−R2 的 target 差异为 `0.128–0.146 ms / 额外选中 token`，与 R2 自身历史 ratio sweep 的边际范围一致；把 R2 外推到 85 个选中 token 预测 `190.7ms`，几乎等于 R5 实测 `190.694ms`。因此 **R5 被默认排除的理由是「冗余」，不是「被 R2 性能支配」**。

#### 4.6.6 Break-even：公式外推，不是实测

review 推导的 formula break-even：body2048 `N=4`（fresh-only setup）/ `N=8`（含 raw setup）；body1024 `N=5` / `N=9`；理想化 one-request 协议下投影 `N=2` / `N=4`。**这些全部是理论投影，Phase4 从未实测 N=1/2/4/8 摊销序列**（PRC-04 / PRC-17 列为未完成项）。

#### 4.6.7 corrected rerun 自身的遗留不对称性（不得省略）

1. recovery setup 留下 `2250–4306` 个额外可驱逐的 exact-namespace token，导致 recovery 轮次比 dense 轮次多驱逐 `10.6%–22.4%` token；
2. R5 的 dense / recovery 两臂 filler token 内容不同（salt 含 arm label），R2 两臂共用同一 filler family；
3. R2 fallback metric 不可用（**不是显式 0**），只能由 `indirect_full_prefix_and_mechanism_counters` 间接验证；R5 则为显式 0 fallback；
4. `full_lifecycle` 排除 pressure 生成、server 启停、namespace 清理，**不是全实验 wall-clock**；
5. `rho_resident ≈ 0.96–0.99` 与历史 demand rho ≈ 2.1 不可混用。

### 4.7 CL2 chunk1024 confounding — 对 Phase4 body 叙事的关键反向修正

证据：`docs:TRACKING.md:1766-1800`；artifact `impl:benchmark/approx_kv/results/phase6/cl2-chunk-gate.json`，`raw_sha256=ab384e6594d1cf293bb5ad9b8a9dbe5fa68dcd4babfcbe8cbe29b0b1250abfc2`。

| chunk | body | dense target TTFT | approx target TTFT | target-only | request-path |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 768 | `129.8ms` | `126.4ms` | `1.027x` | `1.0172x` |
| 1024 | 1024 | `297.8ms` | `171.8ms` | `1.733x` | `1.5467x` |
| 4096 | 768 | `129.3ms` | `127.6ms` | `1.013x` | `1.0051x` |
| 4096 | 1024 | `178.4ms` | `172.8ms` | `1.032x` | `1.0252x` |

表中的 request-path 列使用 artifact 的
`summaries.*.median_request_path_speedup`，即 formal repeat 内 paired
speedup 的中位数。历史叙述中的约 `1.549x` 是由边际中位数再相除得到的另一
estimator；本报告不再混用二者。

机制：`launch_server` 把 `--max-prefill-tokens` 与 `chunked_prefill_size` 同步设置。body1024 的 target prompt 长度为 `64 + 1024 + 1 = 1089` token，在 chunk=1024 下 dense 必须跨两个 prefill chunk（TTFT `297.8ms`），在 chunk=4096 下单 chunk（`178.4ms`）；approximate 臂两种配置几乎不变（`171.8ms` vs `172.8ms`），因为它只需 prefill 最后 1 个 token。body768（prompt `833` token）在两种 chunk 下都是单 chunk，speedup 均 ≈ `1.0x`，作为对照组与该解释完全一致。

**结论：在已测 body768/1024 上，观察到显著的 coupled
chunk/max-prefill effect；body1024 的大部分表观收益可由 dense baseline
跨 chunk 的额外成本解释。** 这不是完整的 chunk-size 单变量因果结论。

必须保留的限定（Review B 强制措辞弱化，`docs:TRACKING.md:2516-2519`）：

- 只测了 body768/1024；
- 同时改动了两个配置项（chunk 与 max-prefill-tokens 绑定），**不得泛化为完整的 chunk-size 单变量结论**；
- CL2 正式 status 记为 `inconclusive`（promotion gate 因 correctness guardrail 未通过而无法产出选中 chunk）；
- CL2 未覆盖 CL1 的 body2048，只追加了一个**显式 out-of-contract diagnostic** 敏感性点，不作为 CL2 正式结果。

V7 处置：R2 只保留 Phase4 chunk1024 历史引用并标 `disabled_not_comparable`；Phase7 primary chunk 迁移到 `4096`，sensitivity = `1024`。**硬性规则：不得把 Phase4 R2 数值与 Phase7 chunk4096 结果放入同一排名或合并统计**（`docs:IMPLEMENTATION_PLAN_LATEST.md:948`）。

### 4.8 CL1 screening → 3-restart confirm：`practical family = NONE`

证据：`docs:TRACKING.md:1686-1763`；artifact `impl:benchmark/approx_kv/results/phase6/cl1-screening.json`、`cl1-confirm.json`（`raw_sha256=7736f0e7f641ce7d9d628a4ea7bf1b6697ede4019bf6e6214b37efb57fff8945`）、以及 P0 修复后的 `cl1-rerun-screening.json` / `cl1-rerun-confirm.json`。

#### Screening（6 candidate：r0、r1_k0/k4/k8/k16/k32；body1024/2048；restart=1；formal=4；48 paired repeats）

| 观测 | 数值 |
| --- | --- |
| body2048 median request-path speedup | 全部落在 `1.952x–1.984x`，候选间差异 `<1.6%` |
| body1024 | r0 / r1_k0 = `1.554x` / `1.555x`；k≥4 = `1.451x–1.467x` |
| paired target p95 ratio | `0.476–0.632` |
| N=1 摊销 | `0.420–0.488` |
| N=8 摊销 | `1.156x–1.357x` |
| break-even | `3.75–4.54` 次复用 |

注意：**EPIC 的 leading-k 在 request-path 口径下是净成本**（k≥4 慢于 k0），与 target-only 口径下的结论方向相反。

#### FINDING-CL1-A（阻塞 promotion）

48 个 paired repeat 中 `quality_8_token_match` 失败 `17` 次、`first_token_match` 失败 `6` 次 → `all_guardrails_passed=false`，对全部 6 个 candidate 均成立。cache path / reset invariant / pool 恢复 `48/48` 全通过 → **这是恢复质量结果，不是 harness 故障**。

#### 3-restart confirm（r0、r1_k0）

`promotion.status=complete, passing=[], winner=NONE`。

| 项 | r0 | r1_k0 |
| --- | --- | --- |
| body2048 per-restart median | `1.972` / `1.965` / `1.978` | `1.969` / `1.972` / `1.974` |
| 3/3 restart > 1.0x | 是 | 是 |
| p95 ratio | `0.480` | `0.479` |
| N=8 摊销 | `1.353x` | `1.351x` |
| guardrail 失败 | `quality_8_token_match` 12 次、`first_token_match` 4 次（48 次中） | 同上合计 |

**关键点：`NONE` 完全由 correctness guardrail 决定，性能条件全部满足。**

#### P0 修复前后对照（因果归因）

首轮怀疑 CL1 输出偏离由「prefix 驱逐缺陷」（eviction-dependent P0，见 §5.6）造成。修复该 P0 后重跑：**guardrail 失败计数在修复前后完全一致**（screening `17+6/48`；confirm `12+4/48`），性能数字几乎不变（confirm body2048 per-restart `1.987/1.977/1.966`）。因此「因果归因无效」的怀疑被解除。

机制解释：CL1 的 `source_header` 起始 `32_000`、`target_header` 起始 `36_000`——在一个前缀下算出的 body KV 被拿到**另一个前缀**下使用，KV 本来就是近似的；而 P6-H 的 source/target 使用同一 header，修复后能正确复现 dense。

#### Review 强制的措辞弱化（必须原样保留）

1. **不得**写「已证明是真实近似误差、不是 bug」。正确措辞：「该偏离与预期的跨上下文近似一致，且**无法由已修复的压力损坏缺陷解释**」——P6-H 与 CL1 在 scheduler/chunk/residency/harness 上均有差异，不能完全排除 CL1 特有的残留问题。
2. `practical=NONE` 是**规则范围内**结论，不是普遍不可行性；作用域 = 本模型、合成 prompt 族、exact-output 不变量、本 GPU、`chunk=1024`。
3. 所报「paired p95」实为 pooled `p95(approx)/p95(dense)`，**不是配对统计量**；N=8 摊销是外推值，不是真实测得的 8 次复用。
4. 独立复制单元很少：CL1 只有 3 个 restart 级单元、CL2 为 2、CL3 多数为 1。
5. `practical=NONE` 的 `same/different header × low/high pressure` 2×2 **并非真正 factorial**（拼接 P6-H 与 CL1，runner/policy/chunk/env/SHA/重复数均不同）。**最强剩余替代解释是 header-dependent 实现缺陷，不依赖 eviction**；已排除已修复的 eviction-dependent P0，但未证明 context 差异是唯一原因，也未排除 header-dependent 实现缺陷。

#### V7 冻结措辞（`docs:IMPLEMENTATION_PLAN_LATEST.md:790-804`）

> 在本模型、合成 prompt 族、SM75、`chunk=max-prefill=1024` 与冻结 exact-output promotion 规则下，没有 candidate 通过。已排除已修复的 eviction-dependent prefix-overwrite 缺陷，但未证明 context 差异是唯一原因，也未排除 header-dependent 实现缺陷。V7 将 primary 迁移到 4096，`NONE` 在 4096 下未重新 qualification，跳过 practical 是 V7 scope 决策，不是新的经验结论。

### 4.9 Guardrail 语义分层（避免过度声明）

`docs:IMPLEMENTATION_PLAN_LATEST.md` §5.9：

| Guardrail | 条件 | 失配含义 |
| --- | --- | --- |
| §5.9.A same-context corruption canary | `source header == target header` | 任一输出 token 失配 = `INVALID` 工程缺陷。证据模板 P6-H、P6-F。**仍不等价于 bitwise KV 或 logit fidelity** |
| §5.9.B cross-context exact-output promotion gate | `source header != target header` | 输出失配是**设计内近似结果**，不自动等于 corruption / semantic failure / 一般不可用；exact-output equality 只是一项**保守产品 promotion 策略**；必须记录逐位置一致率，不得把失配引为数据损坏证据 |

CL1 促成的 promotion 规则：body2048 request-path 至少 2/3 restart `>1.0x`；p95 恶化 `≤5%`；无通过者则 `practical family = NONE`。**该规则在 CL1 执行前冻结，看到结果后不得修改。** FINDING-CL1-C 记录了计划 §5.9 与冻结 runner 在「8-token 是否为硬门」上的不一致，按严格「以已冻结实现为准」处置，差异留给后续版本。

Fallback 证据分级（FINDING-CL1-B）：带 label 的 Prometheus counter 在未发生事件时不输出任何 series，因此「counter 缺失」只能记 `indirectly_verified`，**不得记为显式 `0`**。该规则同样反向应用于 R2 corrected rerun 的 fallback 字段。

### 4.10 成功 / 失败 / 被跳过项汇总

| 项 | 状态 | 说明 |
| --- | --- | --- |
| R0 代表点 | 成功 | dense 分母复用 R1 测量；R1-k0 为机制等价 full OAT proxy（manifest `notes`） |
| R1 body/header/rho sweep | 成功 | `authoritative_historical` |
| R2 ratio/body/header/rho OAT | 成功但关键 cell 被取代 | `historical_oat` |
| R3 Cache-Craft GPU | **失败/Deferred** | 无任何真实 GPU 结果；runner 保留 exit code 3 |
| R4 KVCOMM sweep | 成功（诊断级） | 省略 neighboring-prefix delta group |
| R5 body sweep | 成功但关键 cell 被取代 | `historical_oat`；仅 `r0=0.0829` 一档真实 canary |
| R2/R5 corrected causal rerun | 成功 | `authoritative_corrected` |
| R2/R5 matched repair ratio 对照 | **未做** | 保留任何机制速度排序的必要前提 |
| R2/R5 实测 N=1/2/4/8 摊销 | **未做** | 现有 break-even 全部是公式外推 |
| R0/R1/R4 backfill 到 CL1/CL2 contract | **未做** | 因此不能跨路径排名（PRC-21） |
| CL2 body2048 chunk factorial | **未做** | 仅有 out-of-contract diagnostic 点 |
| chunk4096 下重新 qualification `practical` | **未做** | V7 scope 决策，非经验结论 |

---

## 5. 发现并修复的问题

以下每一项均给出：症状 → 根因 → 修复 → 验证 → 对旧结论的影响。

### 5.1 eviction-aware allocation 缺失

- **症状**：body1024 首次高压 k0 实验崩溃。
- **根因**：ApproxKV 恢复路径直接调用 `allocator.alloc`，未先驱逐 exact evictable victim（`docs:TRACKING.md:2145-2160`）。
- **修复**：引入共享 `allocate_recovery_slots()`，R0/R1/R2/R4/R5 全部迁移使用。
- **验证**：body1024 高压 cell 可稳定完成。
- **对旧结论影响**：修复前的 body1024 高压数据不可用；修复后重跑的数据才进入统一 pressure contract。

### 5.2 RoPE resolver 缺失

- **症状**：R2/R5 body1024 首次崩溃。
- **根因**：分支从未生产绑定 `resolve_model_rope_config` / `bind_rope_config`，第二个 512-token segment 的 `+512` RoPE delta 必然触发 `rope_config_unavailable` fallback。
- **修复**：补齐 resolver/binding。
- **验证**：修复后才能真实测得 body1024。
- **对旧结论影响**：body1024 以上的 R2/R5 数据全部在修复后产生。

### 5.3 gauge 滞后 / `already_pinned_tokens` 负值（R5）

- **症状**：跨 setting 切换时出现负 pinned token（`docs:TRACKING.md:1291-1299`）。
- **根因**：`/flush_cache` 同步清空但 scheduler gauge 异步刷新。
- **修复**：每 round 发送固定 sentinel 强制刷新。
- **验证**：负值消失，accounting 自洽。
- **对旧结论影响**：影响记账可读性，不改变 latency 结论。

### 5.4 header seed 偶然多匹配 1 token（R5）

- **症状**：exact match 长度比预期多 1（`docs:TRACKING.md:1283`）。
- **根因**：裸 `target_head_ids` 生成的 token 偶然等于 body 首 token。
- **修复**：显式追加不冲突的 sentinel token。
- **验证**：cached tokens 严格等于设计值（body1024 主点为 `1088`）。
- **对旧结论影响**：修正了 cached-token 记账，未改变 speedup 量级。

### 5.5 chunk 披露缺失（配置层缺陷）

- **症状**：Phase4 所有 speedup 均未披露 `chunked_prefill_size` 与 `max_prefill_tokens`。
- **根因**：`launch_server` 隐式绑定二者，且报告模板未要求披露。
- **修复**：V7 强制「任何 recovery speedup claim 必须同时声明 `chunked_prefill_size` 与 `max_prefill_tokens`，并附一个 prompt 可单 chunk 容纳的对照点」（`docs:IMPLEMENTATION_PLAN_LATEST.md` §15.2 第 3 条）。
- **验证**：CL2 chunk gate 实测。
- **对旧结论影响**：**最大的一次**——R0/R1 长 body 的历史性能结论全部被限定在 chunk1024 作用域内。

### 5.6 P0：请求自身 exact prefix 未加锁导致自我驱逐 / 自我覆写

- **症状**：CL1 输出偏离；P6-4 出现 `_delete_leaf` 断言。
- **根因**：`Req.init_next_round_input` 调用 `restore_request_prefix` 发生在 `schedule_policy.add_one_req` 获取 prefix 锁**之前**，此时 `req.last_node.lock_ref == 0`，而 victim 枚举条件恰为 `lock_ref == 0`，两者叠加使请求可驱逐并覆写自己即将 attend 的 KV。
- **修复**：新增 `protect_request_prefix` 上下文管理器（提交 `af81934e4`），整个 recovery 窗口持有标准 prefix 锁。
- **验证**：GPU 上先前必然损坏的配置修复后逐 token 与 dense 完全一致；新增 5 个回归测试。
- **对旧结论影响**：**修复后 CL1 guardrail 失败计数完全不变**，因此该缺陷**不是** CL1 输出偏离的成因；但它解除了「因果归因无效」这一阻塞，使 `practical=NONE` 成为可归因的结论（在 §4.8 的措辞限定内）。

### 5.7 采样间隔过粗导致「死亡瞬间」结论完全错误

- **症状**：诊断 C v1 以 `0.4s` 轮询，得出「这是我方缺陷」的结论。
- **根因**：临近死亡的最后 `1.3s` 内 `num_used_tokens` 从 `5376` 涨到 `10688`，`0.4s` 采样点必然早于致命请求。
- **修复**：改为 `0.05s` 采样。
- **验证**：结论完全反转为「真实容量不可达」。
- **对旧结论影响**：这是 Phase4→Phase6 期间最重要的**测量方法论**教训，写入 `docs:IMPLEMENTATION_PLAN_LATEST.md` §15.2 第 14 条。详见 [Phase6 报告](PHASE6_CROSS_STORE_SUBSTRATE_REPORT.md) §5。

### 5.8 R2/R5 causal-key ground-truth 构造缺陷

见 §4.6.1–§4.6.4。这是 Phase4 唯一一次**推翻已发布数值**的修复。

---

## 6. Lessons Learned

### 6.1 机制层

1. **body 与恢复收益的关联受 chunk/max-prefill 配置显著调制。** 在
   chunk1024 下观察到 768–1024 之间的 crossover；把耦合配置改为 4096
   后，body1024 的收益几乎消失。现有双变量实验不足以宣布某个变量是全局
   主导因素。
2. **repair budget 不是越大越好。** R2 的 `1%` 优于 `5%/15%/30%`；重算成本随选中 token 数近似线性上升（`0.128–0.146 ms/token`）。
3. **target-only 收益与 request-path/lifecycle 收益可以符号相反。** R2/R5 在 target-only 上是 `1.6–2.0x`，在 request-path 上是 `0.43–0.53x`，在 lifecycle 上是 `0.25–0.33x`。**只报告 target-only 会得到方向性错误的结论。**
4. **EPIC 的 leading-k 在不同口径下方向相反**：target-only 下 k>0 相对 k0 有代价但仍可能盈利；request-path 下 k≥4 是净成本。
5. **precomputed adapter 不是 practical 方案。** R2/R5 自始至终是 precomputed fresh-KV oracle，corrected rerun 进一步确认了这一定性。

### 6.2 系统层

6. **恢复路径必须与 exact cache 的分配/驱逐语义共同设计。** eviction-aware allocation 与 prefix lock 是两个独立但同源的缺陷（都源于「恢复路径绕过了 exact 路径已有的不变量」）。
7. **多 segment 恢复必须显式绑定 RoPE 配置**，否则会静默 fallback。
8. **服务端 gauge 是异步的**，跨 setting 切换必须强制刷新，否则会产生自相矛盾的记账。

### 6.3 测量层

9. **配置披露不足会制造伪机制结论。** chunk1024 confound 是本项目最贵的一课：一个未被记录的服务器参数把「机制收益」放大了约 `1.7x`。
10. **任何「上限」口径都必须显式标注为上限**，并且禁止称作 end-to-end。
11. **摊销必须实测。** Phase4 全部 break-even 都是公式外推；Phase7 才引入真实 N=1/2/4/8 累计测量。

### 6.4 统计层

12. **独立复制单元是 server restart，不是请求，也不是 formal repeat。** CL1 只有 3 个 restart 级单元。
13. **禁止使用「within noise」这类未经检验的统计判断**；只能写「数值上几乎不可区分」。
14. **pooled `p95(approx)/p95(dense)` 不是配对统计量**，必须如实命名。

### 6.5 治理 / provenance 层

15. **必须维护机器可读的 supersede 索引。** `docs:research/PHASE4_RESULT_MANIFEST.json` 逐 artifact 标注 `status` 与 `superseded_cells`（含被替换的 `body_tokens/header_tokens/ratio/rho_logical_demand` 坐标），否则被推翻的数字会通过旧 slides/HANDOFF 无限期传播。
16. **pre-correction 快照必须显式标记。** slides 与 HANDOFF 尾表至今仍含 `1.14x`/`1.04x`，只能作为历史演进节点引用。
17. **promotion 规则必须在看到结果前冻结**，并且发现规则与实现不一致时，按已冻结实现判定、把差异留给下一版本，而不是当场改规则。

---

## 7. 最终结论

### 7.1 当前仍成立的结论

| 结论 | 作用域限定 | 证据 |
| --- | --- | --- |
| 受控 KV 恢复能显著降低**目标请求本身**的 TTFT | chunk1024；body≥1024；本模型/本 GPU | §4.2、§4.3、§4.4、§4.6 |
| R2/R5 target-only 收益（`1.61x–2.04x`） | 同上 | §4.6.3 |
| **单次使用（single-use）恢复在 request-path 与 lifecycle 口径下是净亏损** | R2/R5 corrected；chunk1024 | §4.6.3 |
| repair budget 越大越慢（1% > 5% > 15% > 30%） | R2；body1024；rho≈2 | §4.3 |
| body512/768 下恢复无收益 | chunk1024 | §4.2、§4.3、§4.4、§4.5 |
| R4 KVCOMM 的 setup footprint ≈ 5x body tokens | 结构性事实 | §4.4 |
| R2/R5 是 precomputed oracle，不是 practical candidate | 全局 | §3.2、§4.6 |
| 恢复轮次会额外增加 exact-namespace 驱逐（`10.6%–22.4%`） | corrected rerun | §4.6.7 |

### 7.2 被收窄的结论

| 原结论 | 收窄后表述 |
| --- | --- |
| 「body 长度是最重要变量，crossover 在 768–1024」 | 只能保留为 `chunk=max-prefill=1024` 下的历史关联；CL2 显示 coupled chunk/max-prefill 配置会把 body1024 收益降到 `1.032x`，因此不能宣布全局主导变量 |
| 「R0/R1 长 body 高压下仍有收益」 | 性能层面成立（CL1 3/3 restart `>1.0x`），但（a）部分是 chunk1024 artifact，（b）correctness guardrail 未通过 |
| 「Phase4 rho sweep 显示 speedup 随 rho 单调」 | **从未成立**：`docs:TRACKING.md:2547` 明确 rho sweep 本身并未证明单调性；body length 的历史关联也只在固定 chunk1024 配置下成立 |
| 「R4 break-even ≈ 6/14 次 reuse」 | 是 setup/收益比的公式推导，不是实测摊销序列 |

### 7.3 被推翻的结论

| 被推翻结论 | 替代结论 | 证据 |
| --- | --- | --- |
| R2 body2048 single-use combined `1.14x` | `0.407x` | §4.6.3 |
| R5 body2048 single-use combined `1.04x` | `0.406x` | §4.6.3 |
| 「R2 比 R5 更快，机制更优」 | 撤回；差异由 repair ratio 配置解释 | §4.6.5 |
| 「CL1 输出偏离是 prefix 驱逐缺陷造成的」 | 修复前后 guardrail 失败计数完全一致，该缺陷不是成因 | §4.8 |
| （诊断 C v1）「S0/rho2 OOM 是我方回收路径缺陷」 | 0.05s 采样后反转为真实容量不可达 | §5.7 |

### 7.4 明确**不能**声称的内容

1. **不能**声称任一恢复路径是 production-ready 或 practical winner。
2. **不能**声称跨路径性能排名（R0–R5 未在同一 causal/paired/four-ledger/guardrail contract 下 backfill）。
3. **不能**声称 R2 优于 R5 或反之。
4. **不能**把 chunk1024 下的 speedup 迁移到其它 chunk 配置。
5. **不能**把 Phase4 R2 数值与 Phase7 chunk4096 结果合并统计或排名（硬性规则）。
6. **不能**把 `practical=NONE` 写成「跨上下文 KV 恢复普遍不可行」。
7. **不能**把 CL1 的输出失配写成「已证明是真实近似误差、不是 bug」；也**不能**写成「数据损坏」。
8. **不能**声称 R3 Cache-Craft 被实验验证或证伪——它从未产生真实 GPU 结果。
9. **不能**把 R4 的 `1.37x/1.76x` 用作 KVCOMM 论文机制的完整复现结论（省略了 neighboring-prefix delta group）。
10. **不能**把 `R4-like` proxy 的任何数字归因于 KVCOMM。
11. **不能**把 formula break-even 写成实测摊销。
12. **不能**把 `rho_resident` 与 demand rho 混用。

---

## 8. 该结论能预测什么（可证伪预测与待验证假设）

以下均为**预测**，不是已测事实；每条给出证伪条件。

| 编号 | 预测 | 证伪条件 |
| --- | --- | --- |
| P4-1 | 在 chunk4096 下重跑 R0/R1 的 body1024/2048 ceiling，request-path speedup 将显著低于 chunk1024 下的值，且很可能 `<1.0x` | 若 chunk4096 下 request-path 仍 `≥1.5x`，则 chunk confound 解释被证伪 |
| P4-2 | R2/R5 若在同一 repair ratio 下重测，target-only 差异将落在 `0.128–0.146 ms/token` 的边际曲线可解释范围内 | 若同 ratio 下仍有系统性差异，则「同一曲线不同点」的解释被证伪 |
| P4-3 | R2/R5 的实测 N=1/2/4/8 序列可能偏离公式 break-even，因为公式没有建立独立 process-level 摊销证据 | 若实测序列与公式在各 N 上一致，则公式已充分描述该 scope；否则以实测为准 |
| P4-4 | 在同一 runner/config 下补齐 same/different header × low/high pressure 真 factorial，可区分 context effect 与 header-dependent 实现缺陷 | 若 same-context 仍失配，只能证明存在额外实现/config 因素，并会削弱「context 是唯一解释」；它不能单独否定 different-header 中同时存在 context effect |
| P4-5 | **若** CL2 观察到的 coupled chunk/max-prefill effect 在更大 body 上仍占主导，则 dense 跨更多 chunk 时 request-path speedup 应上升 | 若 dense 跨更多 chunk 后 speedup 不升，则 copy/setup 等固定或线性开销更可能主导 |
| P4-6 | 5x resident multiplicity 在紧容量下比 1x/2x profile 更容易不可达 | 若同一容量合同下 R4-like 与低 multiplicity profile 同样稳定可达，则 multiplicity 不是主要容量约束；该预测仅涉及 footprint proxy，不代表 KVCOMM 执行 |

### 8.1 对下一 Phase 的直接推论

- Phase5 必须把 **scheduler 效应与 recovery 效应隔离**，否则无法归因（Phase5 因此只测 exact Radix）。
- Phase6 必须先建立 **exact 与 approximate 对象在同一 GPU 预算下真实竞争** 的底座，否则 Phase4 的单机制结论无法组合。
- Phase7 必须把 primary chunk 迁到 `4096`，并把 `1024` 降级为 sensitivity 诊断。

---

## 9. 局限、未完成项与 artifact / provenance 索引

### 9.1 局限

1. 单 GPU（SM75，8GB）、单模型（Qwen3-0.6B）、合成 prompt 族；无真实 repository / 真实 agent trace。
2. 全部为串行请求，无并发/多租户干扰。
3. 恢复质量只用输出 token 一致率衡量，**不建立 bitwise KV 或 logit fidelity**。
4. Phase4 的 OAT（one-at-a-time）设计：绝大多数结果是围绕代表点的单变量切片，不是完整笛卡尔矩阵（`docs:research/PHASE4_RESULT_MANIFEST.json` 的 `matrix_definition` 明确记录这一点）。
5. corrected rerun 只覆盖 R2/R5 的 body1024/2048@header64@rho2 关键 cell。

### 9.2 未完成项（明确 open，不得声称已解决）

| 编号 | 未完成项 | 影响 |
| --- | --- | --- |
| PRC-21 | R0/R1/R4 未 backfill CL1/CL2 causal/paired/four-ledger/guardrail contract | 不能跨路径排名 |
| PRC-16 反面 | R2/R5 matched repair ratio 对照实验 | 保留任何机制速度排序的前提 |
| PRC-04 / PRC-17 | R2/R5 真实 N=1/2/4/8 测量摊销序列 | 现有 break-even 全为公式外推 |
| PRC-23 | CL2 chunk factorial 未覆盖 body2048；rho1.1/rho3 robustness 未做 | chunk 结论不能泛化 |
| — | R3 Cache-Craft 深层实现 | 需要跨 scheduler/model/attention backend 改动 + 专项 GPU 验证 |
| — | chunk4096 下 `practical` 重新 qualification | V7 scope 决策，非经验结论 |
| — | header-dependent 实现缺陷的排除 | §4.8 第 5 条的最强替代解释仍未被证伪 |
| — | R4 neighboring-prefix delta group 纳入统一矩阵 | 现仅由更早小 canary 证明 |

### 9.3 Artifact / provenance 索引

**权威索引**：`docs:research/PHASE4_RESULT_MANIFEST.json`（`schema_version=1`，8 个 artifact 条目 + `r1_rho0p9_raw` 段）。

| artifact（仓库内相对路径） | worktree / branch head | sha256 | status |
| --- | --- | --- | --- |
| `benchmark/approx_kv/results/phase4-r0/sm75-unified-pressure.json` | `raw-rope` @ `61c39791e…` | `29010d11…` | `representative` |
| `benchmark/approx_kv/results/phase4-r1/sm75-eviction-pressure.json` | `epic-legolink` @ `984bfd873…` | `1f2e822f…` | `authoritative_historical` |
| `benchmark/approx_kv/results/phase4-r1/sm75-inrequest-matrix.json` | `epic-legolink` @ `984bfd873…` | `bd7d4ac9…` | `historical_mechanism_only` |
| `benchmark/approx_kv/results/phase4-r2/sm75-unified-pressure.json` | `cacheblend` @ `ce55860a9…` | `d7183e35…` | `historical_oat`（body1024/2048 @header64 @ratio 0.01 @rho2 被替代） |
| `benchmark/approx_kv/results/phase4-r2/sm75-causal-key-rerun.json` | `cacheblend` @ `ce55860a9…` | `84d28044…` | **`authoritative_corrected`** |
| `benchmark/approx_kv/results/phase4-r4/sm75-unified-pressure.json` | `kvcomm` @ `cd81c3e92…` | `5ae9762b…` | `authoritative_historical_diagnostic` |
| `benchmark/approx_kv/results/phase4-r5/sm75-unified-pressure.json` | `cachetune` @ `71f15d5d1…` | `08b128ef…` | `historical_oat`（body1024/2048 @header64 @rho2 被替代） |
| `benchmark/approx_kv/results/phase4-r5/sm75-causal-key-rerun.json` | `cachetune` @ `71f15d5d1…` | `007099d6…` | **`authoritative_corrected`** |

Closeout artifact（位于 `impl:` 仓库）：

| 文件 | raw_sha256 | 说明 |
| --- | --- | --- |
| `benchmark/approx_kv/results/phase6/cl1-screening.json` / `cl1-confirm.json` | confirm `7736f0e7…` | CL1 screening / 3-restart confirm |
| `benchmark/approx_kv/results/phase6/cl1-rerun-screening.json` / `cl1-rerun-confirm.json` | — | P0 修复后重跑 |
| `benchmark/approx_kv/results/phase6/cl2-chunk-gate.json` | `ab384e65…` | CL2 chunk gate |
| `benchmark/approx_kv/results/phase6/context-vs-pressure-2x2.json` | — | same/different header × 压力 2×2（非真 factorial） |
| `benchmark/approx_kv/results/phase6/RESULT_MANIFEST.json` | — | Phase6 file→commit 映射，`--check` 通过 `48/48` |

`r1_rho0p9_raw` 特别说明：权威文件为 `/home/chris/Workspaces/kvcache-research/results/phase4-epic-pressure-rho/k0-fixed-rho0p9.json`（sha256 `fbfec1c1…`）；同目录下 `dense-rho0p9.json`（`69ebc7f9…`）与 `k0-rho0p9-fixed.json`（`c7a14a28…`）因 `eviction_observed` 元数据冲突被标 `stale_do_not_use`。

### 9.4 不得引用为权威数值的历史快照

1. `docs:research/PHASE4_STAGE_REPORT_SLIDES.md` §3/§6/§7 的数值表（含 R2 `1.14x`）——2026-07-22 pre-correction 快照。可引用其术语/动机文字（header 定义、repair ratio 含义），**不可引用数值表**。
2. `docs:HANDOFF.md` 尾部 worktree 状态表（含 `1.14x` / `1.04x`）——同样是 pre-correction 快照。
3. R2/R5 `sm75-unified-pressure.json` 中被 `superseded_cells` 标注的 cell。
4. `sm75-inrequest-matrix.json`（`historical_mechanism_only`）——无 eviction pressure、旧代码 SHA、轴定义不同，只能作为「机制曾被验证」的证据。

---

## 10. 与其它阶段报告的关系

| 关系 | 说明 |
| --- | --- |
| → [Phase5](PHASE5_WORKFLOW_SCHEDULING_REPORT.md) | Phase5 刻意**不使用**任何 Phase4 恢复路径，以隔离 scheduler 效应；两阶段 speedup 分母不同，不可直接排名 |
| → [Phase6](PHASE6_CROSS_STORE_SUBSTRATE_REPORT.md) | Phase6 为「exact 与 approximate 对象在同一预算下真实竞争」建立底座；CL1/CL2 与 Phase6 的 P0 修复相互纠缠（见 §5.6） |
| → [Phase7](PHASE7_INTEGRATED_EVALUATION_REPORT.md) | Phase7 在 chunk4096 下重测 R0 ceiling，结论为 `NEGATIVE`；这与本报告 P4-1 预测方向一致 |
| → [跨阶段总报告](PHASE4_TO_PHASE7_SUMMARY.md) | 汇总四阶段的方法论反转与最终决策规则 |
