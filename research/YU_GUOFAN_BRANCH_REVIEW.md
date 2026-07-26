# Yu Guofan / AgentTemplateKV 研究分支审查

最后更新：2026-07-12T19:40:52-07:00

## 结论先行

该研究线做了大量真实的工程、实验和论文调研，尤其在 workflow metadata、代码段 KV reuse、AST chunk、CPU host pool、HKVD 测量、selective recompute 和负结果记录方面有明显投入。

但它**不是 KVCOMM `2510.12872` 的忠实复刻**。当前实现的核心更接近：

```text
exact/raw KV copy
+ Key 的 RoPE position shift
+ MiniLM/AST/heuristic gate
+ 固定比例或固定位置的 partial recompute
+ workflow hint 与 device-resident retention
```

而 KVCOMM 的核心是：

```text
placeholder base KV
+ agent/context-specific ΔK/ΔV
+ neighboring-prefix ΔK/ΔV
+ multi-anchor soft interpolation
+ RoPE de-rotation/re-rotation
+ length/embedding/entropy gating
+ dense fallback 与在线 anchor update
```

因此，最新分支适合作为**研究档案和少量 helper 的 donor**，不适合作为本项目的 KVCOMM 实现基线，也不建议把整个 7,143 行 `radix_cache.py` 继续向前叠加功能。

## 审查范围与归属边界

用户所说的 Yu Guofan 对应 GitHub 账号 `flaminyu`。最近两个月的主要研究分支呈线性继承：

| 阶段 | 分支 | tip | 日期 | 相对上一阶段新增提交 |
| --- | --- | --- | --- | ---: |
| 1 | `para_temp` | `f893c6488` | 2026-05-20 | 4 |
| 2 | `feature/context-aware-kv-reuse` | `3681e8fa2` | 2026-06-05 | 10 |
| 3 | `agenttemplatekv-eurosys-2026-06` | `5fb934751` | 2026-06-12 | 18 |
| 4 | `phase-2.7-prerot` | `003782eb8` | 2026-06-23 | 11 |
| 5 | `fix/placeholder-pool-activation` | `9e84d2f94` | 2026-07-11 | 78 |

最新分支相对 `main` 有 121 个提交。author 统计为：

| author | 提交数 |
| --- | ---: |
| `AgentTemplateKV EuroSys Submission` | 102 |
| `flaminyu` | 12 |
| `claude` / `Claude` | 5 |
| `cw` | 1 |
| 异常编码的 `flaminyu` | 1 |

所以本文使用“该研究分支”“该工作线”，不把 121 个提交全部表述为 Yu Guofan 个人手写。

还需特别区分：

- `5bb9afc92 Priority eviction for SGLang` 的 author 是 `cw`，是这条研究线继承的已有基础。
- `a7960c7a3 Add KVFlow benchmarks and priority cache integration` 才是 `flaminyu` 在该基础上继续加入的工作。
- 后期大量提交使用 synthetic submission identity，并带有 Claude 协作痕迹。

## 这条研究线实际做了什么

### 1. KVFlow priority 与 benchmark

早期 `para_temp` 基于已有 priority eviction，补充 KVFlow benchmark、cache priority integration 和实验脚本。

这一阶段的价值主要是：

- 验证 workflow-aware priority 在 SGLang radix/HiCache 中的接入点；
- 构造 multi-agent cache pressure benchmark；
- 为后续 AgentTemplateKV 的 metadata 和 prefetch hint 提供基础。

### 2. Context-aware exact reuse

`feature/context-aware-kv-reuse` 开始研究：

- 相同代码位于不同 system prompt、位置和 surrounding context 时的 KV drift；
- exact-content anchor；
- Key 的 RoPE delta rotation；
- 基于离线表格的 `context_aware_confidence`。

这一阶段已经意识到“文本相同不代表 KV 相同”，方向正确，但解决方案仍是对某次真实上下文中的 KV 做 copy/rotation，而不是 KVCOMM 的 base+offset reconstruction。

### 3. AgentTemplateKV 包装

`agenttemplatekv-eurosys-2026-06` 将工作重新组织为：

1. coding MAS workflow template；
2. template-derived codebase prefetch hints；
3. exact-content code segment reuse。

最新论文稿已经明确把 KVFlow 和 KVCOMM 写成 prior work / implementation inspiration，而不是本文贡献。这一收缩是正确的，因为当前代码确实没有实现 KVCOMM 原算法。

### 4. Placeholder k-NN 与 pre-rotation

`phase-2.7-prerot` 加入：

- per-placeholder MiniLM embedding pool；
- top-k 候选搜索，但实际消费 `neighbors[0]`；
- raw KV copy；
- head-only Key RoPE；
- 多个 delta 的预旋转 Key；
- cost gate、overlap gate 和 benchmark mode reorder。

该路径后续被标记为 research-only / production deprecated，这是合理决定。

### 5. AST chunk、离线 KV、CPU host pool 与 selective recompute

最新分支继续加入：

- Python AST chunker；
- per-chunk KV pool；
- 离线 Codebase KV compiler；
- CPU host pool 与 H2D load；
- R32 固定 leading fraction recompute；
- position-aware、node-kind、control-flow 等多轮 ablation；
- “True CacheBlend”逐 token mini-prefill 原型；
- tool-output cache hit-rate telemetry；
- 大量 R28-R40 报告与负结果归档。

这是整条工作线中实验最扎实的部分，但其中大部分属于新的 code-chunk reuse 研究，而不是 KVCOMM。

## 与 KVCOMM 的逐项对照

KVCOMM 原文的 anchor pool 为每个 placeholder 保存：

1. 无外部 context 的 base KV；
2. placeholder 在各 agent context 下相对 base 的 offset；
3. placeholder 后 neighboring fixed-prefix 的 offset；
4. embedding、长度和使用信息。

命中时，KVCOMM 使用多个 anchor 做 soft interpolation，并在必要时 dense prefill、在线新增 anchor。

| KVCOMM 必要机制 | 当前分支状态 | 判断 |
| --- | --- | --- |
| placeholder base KV | `AnchorKVEntry` 只保存某次真实 context 下的 `token_ids`、`kv_indices`、位置和 embedding | 缺失 |
| placeholder `ΔK/ΔV` | 未存储 base 与 actual 的差值 | 缺失 |
| neighboring-prefix offset | 没有独立 prefix offset 数据结构；部分路径改为 gap staging 或 zero gap | 缺失 |
| multi-anchor soft interpolation | k-NN 搜索后固定使用 `neighbors[0]` | 缺失 |
| Key de-rotate → offset estimation → re-rotate | 仅对 copied Key 做 position delta rotation | 不等价 |
| length compatibility + entropy gate | 使用 cosine、长度上限、cost、overlap 等经验 gate | 缺失 |
| 所有 placeholder 均可共享才跳过整段 prefill | 当前按 slot/chunk 局部 copy | 缺失 |
| unsafe 时 dense fallback | 多数路径有 fallback | 部分实现 |
| dense 后在线写入新 base/offset anchor | 只写 raw KV entry，没有 base/offset update | 缺失 |
| LFU-among-oldest anchor pruning | 简化为 per-slot LRU | 不等价 |

直接证据：

- `AnchorKVEntry` 字段位于  
  `origin/fix/placeholder-pool-activation:python/sglang/srt/mem_cache/radix_cache.py:126-180`。
- placeholder k-NN 固定使用第一候选位于  
  `radix_cache.py:5528-5542`。
- copy 路径直接移动 raw KV 位于  
  `radix_cache.py:5832-5876`。
- RoPE 路径只是按 source/target position delta 旋转 Key，位于  
  `radix_cache.py:5891-5976`。

**最终判断：**“KVCOMM-style”只能表示灵感来源，不能表示论文算法复刻。

## 当前实现中值得肯定的部分

### 1. Whole-slot exact path 有真实 token equality guard

需要区分两条路径：

- L2 whole-slot exact anchor path；
- 后来的 C2 AST chunk path。

L2 路径不只是比较客户端 metadata。它在真正 copy 前还会把当前 request span token 与已存 entry token 做 `torch.equal`：

`radix_cache.py:4574-4628`。

这使 L2 exact path 比单纯 hash gate 更安全。它仍然是 cross-context lossy KV copy，不是 KVCOMM，但不能把它与后面有缺陷的 C2 chunk gate混为一谈。

### 2. 负结果与 claim 回撤较诚实

该工作线后期主动记录了多项失败：

- placeholder k-NN 的 headline 受 mode order/cold-warm 状态影响；
- R38b 在 n=15 不优于 R32；
- node-kind、type-aware 和 control-flow policy 没有超过固定 R32；
- “True CacheBlend”逐 token scheduler path 开销过高；
- 7B coding pipeline 的绝对 patch 能力很弱；
- host-backed prefetch 尚未成为主实验路径。

`results/SCALE15_HKVD_REPORT.md` 将早期 n=5 精度结论修正为：

- lossless：TTFT 1032ms，type match 10.7%；
- R32：745ms，9.8%；
- R38b：753ms，6.7%。

最诚实的定位是约 `1.38-1.43x` TTFT 加速，伴随约 `13%` 的相对 type-match consistency 损失，而不是 accuracy-preserving。

### 3. 实验基础设施有复用价值

可保留的工程包括：

- request/API/telemetry metadata plumbing；
- benchmark 结果分解和 fair-A/B analyzer 的思路；
- AST artifact slicing 框架；
- full-key RoPE delta primitive；
- HKVD 离线测量脚本；
- objective `git apply --check` judge；
- 负结果和 superseded round 的归档习惯。

## 高风险 correctness 问题

### P0：必须先修，否则不能作为实现基线

#### 1. protected lock 释放 off-by-one

`inc_lock_ref(max_ancestors=2)` 使用 `steps <= cap`，实际锁定 leaf + 2 ancestors，共 3 个节点：

`radix_cache.py:6462-6506`。

release 端将 `len(locked)==3` 作为 `max_ancestors=3` 传给同样使用 `steps <= cap` 的 `dec_lock_ref`，会尝试释放 leaf + 3 ancestors，共 4 个节点：

`radix_cache.py:1442-1466`、`6557-6595`。

现有测试只构造了 root 前恰好 3 个节点，因此第四次循环碰到 root 后停止，掩盖了更深树上的错误：

`test_anchor_match.py:1770-1805`。

潜在影响是负 `lock_ref`、`protected_size_` / `evictable_size_` 错账和缓存生命周期损坏。

#### 2. whole-slot placeholder pool 持有可能失效的 radix KV slot

whole-slot pool 直接保存 `kv_indices[start:end].clone()`，但没有为 source radix node 持锁或复制成 pool-owned slot：

`radix_cache.py:1794-1895`。

同时：

- `reset()` 只清 `anchor_kv_store`，不清 `placeholder_anchor_pool` 或 `placeholder_chunk_pool`；
- radix leaf GC 只回收 `anchor_kv_store`；
- read path 把消费项写入 `_consumed_placeholder_entries`，但 `cache_finished_req()` 只处理 `_consumed_anchor_entries`。

证据：

- `radix_cache.py:951-964`
- `radix_cache.py:5987-6058`
- `radix_cache.py:6069-6138`
- `radix_cache.py:6940-7020`

这会造成 stale slot、引用泄漏或 reuse 已被 allocator 重用的 KV page。

#### 3. L3 非连续 slot 被当作连续 prefix

当 slot `start > prefix_len` 时，代码没有填补 `[prefix_len, start)` 的 gap，却直接把 copied slots append 到 `exact_values`：

`radix_cache.py:5647-5682`、`6018-6021`。

copy 后的实际逻辑位置是当前 `exact_len`，但 RoPE delta 仍按原 span `start` 计算：

`radix_cache.py:5891-5905`。

这会同时破坏 prefix continuity 和 position alignment。

#### 4. offset-LCP 只返回 entry skip，不返回 request skip

`_ast_gate_offset_lcp()` 尝试 symmetric、request-only 和 entry-only realignment，却只返回 `(entry_skip, lcp_len)`：

`radix_cache.py:5271-5340`。

消费端始终把 copied region 放到未平移的 request 位置：

`radix_cache.py:5683-5705`。

request-only 或 symmetric skip 因此会把 source body KV 放到错误 target token 上。

#### 5. C2 的“byte-exact”不是完整内容相等

AST signature 只使用：

```text
language + anchor_type + name + whitespace-normalized first 240 chars
```

`ast_chunker.py:99-106`、`257-260`。

读取时 `_find_byte_exact_chunk_entry()` 只比较 `byte_start` 和 `byte_end`，不比较完整文本或 token IDs：

`radix_cache.py:3310-3334`。

本次独立验证构造了两个：

- 同函数名；
- 同长度；
- 前 240 个 normalized chars 相同；
- 尾部 `return "alpha"` / `return "omega"` 不同；

的合法函数。结果为：

```text
same_signature=True
same_range=True
different_text=True
```

因此 C2 可以在代码实际不同的情况下错误命中。`size_mismatch` 分支也事实上不可达，因为 helper 对相同 byte range 直接返回 entry，不检查 token count。

注意：这项缺陷针对 C2 AST chunk path；L2 whole-slot path 有额外 token equality guard。

#### 6. AST “byte offset” 对 Unicode 错误

`line_byte_offsets` 使用 `len(line)`，计算的是 Python character count，不是 UTF-8 byte count：

`ast_chunker.py:452-459`。

独立验证：

```text
unicode_reported_byte_start=7
unicode_actual_byte_start=11
```

此外 `chunk_text()` 先 `strip()`，会丢失原始 leading whitespace offset，却在 dataclass 文档中把 offset 描述为原始输入的 absolute byte offset。

#### 7. 离线 KV compiler 的 token offset 不可靠

实际 prefill 文本为：

```python
preamble + "\n" + text
```

但 chunk offset 使用：

```python
len(tokenize(preamble)) + tokenize(file_prefix)
```

`scripts/precompute_codebase_kv.py:414-447`。

这既遗漏了换行，也假设 BPE tokenization 对字符串拼接可加。两者都不成立，可能从错误 token slot 提取 KV。

#### 8. loader fingerprint 与失败回收不完整

`codebase_kv_loader.py` 只严格核对：

- `head_num`
- `head_dim`
- `layer_num`

没有完整核对：

- model/revision；
- tokenizer vocabulary/config；
- chat template；
- RoPE config；
- dtype/layout；
- preamble；
- repository revision。

token 验证只 decode/re-encode 并比较长度，即使 drift 也“keeping entry anyway”：

`codebase_kv_loader.py:248-257`。

此外 device/host slot 在检查 bin 是否存在之前已经分配；文件缺失或读取失败时直接 `continue`，没有释放：

`codebase_kv_loader.py:204-244`。

#### 9. host copy 失败后仍把 dst slot 记为 cached prefix

H2D load 异常被捕获后，代码注释称会 dense recompute，但实际上仍将 `new_slots` append 到 `exact_values`：

`radix_cache.py:4053-4065`、`4252-4253`。

这会把未初始化或部分初始化的 KV slot 暴露给后续 attention。

#### 10. “True CacheBlend” mini-prefill 的 target position 语义不成立

第一次 mini-prefill 会截短 `prefix_indices`。后续非连续 position 只设置 `extend_input_len=1`，实际重算的是当前 prefix 后的下一 token，不保证是原计划的 absolute target position：

`schedule_policy.py:608-645`。

所以该实验不仅执行路径低效，consumer 本身也存在位置正确性问题。

### P1：实验或策略层面的问题

#### context-aware confidence 并非可靠的 4D safety gate

主要问题：

- analyzer 比较整段 prompt 的 KV，没有切出相同 code span；
- 没有先 de-rotate Key，position drift 会污染距离；
- 192-cell table 由 marginal mean 加法合成，不是 joint measurement；
- runtime 不使用 `nesting_depth`；
- `surrounding_code_hash` 无法映射回 class，实际固定为 `"none"`；
- 策略只看 request context，不比较 request 与 candidate context。

更严重的是，代码注释称缺表时返回 safe no-op，但：

```text
baseline=1.0
d_max=1.0
multiplier=0.5
0.95 * 0.5 = 0.475
```

本次在 `SGLANG_CONTEXT_AWARE_CONFIDENCE=1` 且 table 不存在时独立验证得到：

```text
allowed=False
confidence=0.475
multiplier=0.5
reason=context_aware_confidence_below_floor
```

即显式开启功能但 artifact 缺失时会拒绝全部 exact matches，与“safe no-op”注释相反。

#### prefetch 主要是 warm device hit，不是 CPU→GPU prefetch

`agenttemplatekv_cache.py` 只在 `anchor_kv_store` 中查找已经存在的 entry，验证 token 后 pin/protect，并计为 `device_hit`：

`agenttemplatekv_cache.py:55-130`。

它不主动从 host tier 加载。最新论文也承认主表禁用了 host-backed storage。因此 4-5x TTFT 应描述为 bounded warm device-resident exact reuse，不应解释为 Codebase CPU prefetch 已完成。

#### tool-output cache 目前只是 telemetry

`tool_output_cache.py` 统计 system prompt、tool definition、tool call 和 tool output 的 hash hit/miss，但没有 KV pool write/read 或 round-trip。

它可以作为 measurement-first 设计保留，但不能称为已实现的 tool-output KV cache。

## 实验可信度评估

### 可信或有价值的结果

1. **n=15 scale-up 主动推翻 n=5 的乐观精度结论。**
2. **HKVD-by-position、node-kind、control-flow 等离线 measurement 有研究价值。**
3. **R32 的速度-精度 trade-off 有可复用信息：约 1.38-1.43x，而非无损加速。**
4. **coding pipeline 引入 `git apply --check`，比简单 F1/type label 更接近真实代码质量。**
5. **大量 negative ablation 被记录，没有全部包装成正结果。**

### 需要降级解释的结果

#### 1. E7 agent-scaling benchmark 跨 mode 累积 prompt

`upstream` 在进入 mode loop 前初始化，但每个 mode 结束后没有重置：

`benchmark/multi_workflow/bench_kvcomm_ttft_stress.py:1375-1414`。

后运行的 mode 会收到前面 mode 累积的额外文本，prompt 不再相同。

#### 2. v44 headline 受 mode order / cache warmness 影响

代码注释明确说明将 `placeholder_knn_reuse` 移到最前，以避免它被前面模式的 cache 写入和 eviction 影响。该改动可以提高该 mode 的观测速度，但也使跨 mode 对比混入执行顺序和 cache state。

后续 MATCH=0 control 已证明早期 3x headline 的大部分不是 k-NN body 的算法收益。

#### 3. R32 不是 CacheBlend

R32 对每个 chunk 固定重算 contiguous leading fraction：

`radix_cache.py:2729-2743`。

CacheBlend 是每层根据上一层观测到的 KV deviation/HKVD 选择少量 token 并执行 partial forward。R32 可以叫 CacheBlend-inspired static recompute，不能作为论文等价复刻。

#### 4. “True CacheBlend falsified”只否定了 Path A

当前原型使用：

- uniform token positions；
- 每个 token 一次完整 scheduler mini-prefill。

其 p95 18.06ms、TTFT +1129ms 只说明这种 scheduler mapping 不可行，不能否定 layer-wise masked CacheBlend。

#### 5. EPIC 的结论被错误迁移

EPIC LegoLink `k=2` 是在 live context 中完整重算每个非首 chunk 的前两个 token，包含完整层状态与 K/V。

当前 head-only path 只旋转前两个 token 的 Key，Value 和 hidden-state/cross-chunk attention 都不重算：

`radix_cache.py:5891-5976`。

因此不能使用 EPIC 的质量结论支持该实现。

## 论文 artifact 可复现性

最新论文稿比早期报告更克制，已经明确：

- KVFlow/KVCOMM 是 prior work；
- host-backed prefetch 是 limitation；
- 8k single-segment TTFT 是 bounded evidence；
- 16k/32k 是 fast-path diagnostics。

但 artifact 仍不完整：

- `paper/data_manifest.json` 列出 27 个 source entries；
- 当前 branch 只提交了其中 5 个；
- 22 个 source path 缺失；
- 缺失项包括 safety、RoPE、logit、pass@1、prefetch 和多项 selective-reuse summary CSV/JSON。

在独立 worktree 运行：

```bash
python3 paper/scripts/generate_paper_figures.py
```

立即因缺少：

```text
results/kvcomm_ablation_package/gate_safety_ablation.csv
```

失败。

因此当前已提交的表格和图不能从 branch 内 artifact 完整重生成。

此外：

- `bench_kvcomm_ttft_stress.py` 依赖 sibling `MAScoder/src`，该依赖未提交；
- 多个 benchmark 包含 `/home/gfy/...` 硬编码路径；
- `test_ast_chunker.py` 仍断言旧 `ChunkSpan` 字段集合，而当前 dataclass 已新增 4 个字段，测试确定性过期；
- 本环境运行目标 pytest 时先因缺少 `pybase64` 在 collection 阶段失败，但独立加载模块已经确认字段断言不匹配。

## 其他论文：哪些真的进入了实现

| 论文/系统 | 分支中的作用 | 实际状态 |
| --- | --- | --- |
| KVFlow `2507.07400` | workflow priority、future use、prefetch hint 的核心架构来源 | 直接影响；但未完整复刻 Agent Step Graph propagation/prefetch overlap |
| CacheBlend `2405.16444`，EuroSys 2025 | R32、HKVD、selective recompute 的主要来源 | 直接影响；没有忠实复刻 |
| EPIC `2410.15332`，ICML 2025 | `k=2`、position-independent chunk linking 的来源 | 直接影响；head-only RoPE 是机制误用，R32 反而更接近 LegoLink |
| Prompt Cache `2311.04934` | template/module/stable object identity 的概念先驱 | 强相关概念；无代码移植或公平 baseline |
| LMCache `2510.09665` | CPU/GPU/SSD tiering 和 connector 的部署参照 | 有 runbook 和自研 host pool；最终公平 baseline 未运行 |
| DroidSpeak `2411.02820` | cross-model/cross-LoRA selective layer recompute | 只调研，未实现；与同一模型不同 role/prefix 的问题不同 |
| SnapKV `2404.14469` | capacity-side KV compression 背景 | 仅背景 |
| Mooncake、MemServe、CortexCache、Position-Aware Recomputation、KVLink、Tokencake、Continuum | related work 与候选方向 | 大多没有直接实现 |

需要纠正分支文档中的引用错误：

- CacheBlend 是 **EuroSys 2025**，不是部分文档写的 ICML 2025 或 NeurIPS 2024。
- DroidSpeak 的 arXiv ID 是 `2411.02820`；`2404.14469` 是 SnapKV。
- 最新论文 related-work 没有正式引用最接近 R32 的 CacheBlend 和 EPIC，落后于实际算法来源。

## 对当前项目可直接继承的部分

1. `feature/workflow-priority` 上已有的 priority eviction / HiCache 基础。
2. request schema、scheduler 和 response telemetry 的 metadata plumbing。
3. full-key RoPE delta primitive，但必须建立明确数学测试。
4. AST parser/chunker 框架，修复 full-content hash、UTF-8 offset 和 source/token mapping 后使用。
5. HKVD 离线 measurement 与 negative-result archive。
6. objective patch judge、paired A/B 和 workload manifest 思路。
7. CPU host transfer 的接口经验，但不继承当前 bump-pool ownership 和 loader validation。

## 必须重写的部分

1. KVCOMM 的 base KV、placeholder offset、neighboring-prefix offset、multi-anchor interpolation 和 entropy gate。
2. placeholder whole-slot/chunk pool 的 ownership、pinning、eviction、reset 和 request-finish lifecycle。
3. 离线 KV writer/loader 的 tokenizer boundary、fingerprint、transactional allocation 和失败回收。
4. context-aware confidence pipeline。
5. AgentTemplateKV prefetch/retention 层。
6. R32/True CacheBlend 执行路径。
7. 任何依赖截断 hash 或 byte-range 代替完整 token equality 的“exact” gate。

## 推荐的实现基线

不要从 `fix/placeholder-pool-activation` 继续开发。推荐：

1. 从接近 upstream 的干净分支或已同步的 `feature/workflow-priority` 开始。
2. 先单独实现 faithful KVCOMM，不接 AST、不接 CPU tier：
   - prompt segmentation；
   - canonical base KV；
   - per-agent placeholder/prefix `ΔKV`；
   - Key de-rotation/re-rotation；
   - multi-anchor interpolation；
   - length/entropy/embedding gate；
   - dense fallback；
   - online anchor update。
3. 用 full token IDs + cryptographic full-content hash 建立 exact invariant。
4. 为 cache entry 固化完整 fingerprint：
   - model weights/revision；
   - tokenizer；
   - chat template；
   - RoPE config；
   - dtype/layout；
   - repository commit；
   - canonical preamble。
5. KVCOMM 数学和 correctness 通过后，再接 HiCache CPU tier。
6. 最后加入 AST artifact index 与 `Architect -> Coder -> Debugger` workflow priority。

建议把 KVCOMM 实现拆成独立模块，不继续把策略、pool、实验开关和 production path 堆进 `radix_cache.py`。

## 最终评价

### 做得好的地方

- 研究范围广，动手实现和实验量大；
- 能够发现并记录大量负结果；
- 对 code-aware KV reuse 的工程瓶颈积累了真实经验；
- AST、HKVD、benchmark 和 telemetry 资产值得选择性继承；
- 最新论文已经主动收缩 KVCOMM claim，方向更诚实。

### 核心不足

- 对 KVCOMM 的算法理解没有落实为 base+offset reconstruction；
- 多处把论文机制简化成不等价的 raw copy、head RoPE 或 leading-FRAC；
- cache lifecycle、位置连续性、离线 token mapping 和 exact gate 存在 correctness blocker；
- benchmark 和 artifact 仍不足以支持 production-ready 或完整复现 claim；
- 当前代码组织已经过度集中，继续堆叠会放大风险。

**总判断：研究工作有价值，但实现质量不足以直接继承为核心。最合理的做法是保留其 benchmark、measurement、AST 和 telemetry 经验，从干净 SGLang 基线重建 faithful KVCOMM。**
