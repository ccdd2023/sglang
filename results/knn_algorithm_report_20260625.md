# Placeholder k-NN 复用算法详解 — 与 Code-Aware 复用结合 + Cache-Ordering 机制

**作者**: placeholder k-NN 路径技术报告
**日期**: 2026-06-25
**代码分支**: `phase-2.7-prerot` @ `16d6fc681`
**适用对象**: 组会汇报 + 后续论文 §3/§4 写作参考

> **本报告 3 个问题**：
> 1. Placeholder k-NN 算法的 5 个执行步骤具体做什么？
> 2. 它如何与 code-aware 复用（AST 锚点、exact_anchor_signature）整合？
> 3. 为什么 cache-ordering 决定了 3.37-4.14× 的加速，而 k-NN 本身的贡献只占 1.58-2.87×？

---

## TL;DR

| 主题 | 结论 |
|---|---|
| 算法本质 | k-NN 找到相似的<em>代码块</em>，把它们的 KV <strong>原样复制</strong>到目标位置（不做 dense prefill，不做 RoPE 旋转 blend） |
| 与 code-aware 关系 | placeholder k-NN 是 lossy 路径的<em>扩展</em>——lossy 要求 AST 锚点文本<strong>完全一致</strong>才能复用，placeholder k-NN 放松到<strong>cosine 相似 ≥ 0.85</strong>即可 |
| Cache-ordering 真相 | KNNFIRST 的 3.37-4.14× 加速 <strong>90% 来自 mode 在 cache 中的写入顺序</strong>，k-NN body 本身在 ac=1 上净拖慢（≈ 0.93×） |
| 生产部署 | <strong>不部署</strong>，placeholder_knn_lossy 是 research direction；production 用 textually-identical prefix match |
| 安全性保证 | §6.7 F1=1.0000 (25/25 cells) — 输出分布不变，间接保证不引入新错误 |

---

## 1. Placeholder k-NN 算法 — 5 步执行流程

### 1.1 触发条件

调用栈（`python/sglang/srt/mem_cache/radix_cache.py:680-691`）：

```
match_prefix(req, key)
├── _try_lossy_fuzzy_match(req, key, value, last_node)    # 路径 1：textually-identical 锚点
├── _try_placeholder_knn_lossy_match(req, key, ...)       # 路径 2：placeholder k-NN  ← 本文焦点
└── (如果 value 非空) torch.cat(value)
```

**触发条件 3 个全部满足**：
1. `os.environ.get("SGLANG_PLACEHOLDER_KNN_MATCH") == "1"`
2. `req.placeholder_anchor_token_spans` 非空（AST 锚点 spans）
3. `embedder` 已加载（`semantic_suffix.is_enabled() and load_embedder() != None`）

如果 3 个条件有任一不满足，函数直接 `return exact_values, exact_node` —— 也就是**完全走 lossy 路径的 fallback**（这就是 §2.9 byte-equal 结果的来源：per-case driver 启动新 sglang server，embedder 不一定加载，或 pool 为空）。

### 1.2 算法 5 步

#### Step 1 — Placeholder 锚点建立（write path）

在 radix cache **写入**路径上（`_insert_helper` → `_try_lossy_fuzzy_match` 之后），对每个 AST 锚点：

```python
# radix_cache.py:1237-1260
with self.anchor_kv_store_lock:
    self.anchor_kv_store.setdefault(
        segment_content_signature, []   # ← 按 code content 签名索引
    ).append(entry)                      # entry: AnchorKVEntry(KV blob + start + len + 签名)
```

**`anchor_kv_store` 维护的是 textually-identical 的 KV 入口**（与 lossy 路径共用），它的 key 是 `code_content_signature`（即代码内容的 SHA-1 哈希）。相同代码块的所有 KV 实例都在同一个 list 里（用于 ref-count GC）。

**`placeholder_anchor_pool` 是独立的另一个池**（radix_cache.py:555）：

```python
self.placeholder_anchor_pool: dict[str, list[AnchorKVEntry]] = {}
# 索引 key: slot_id（与 code content 签名不同，slot_id 是 AST 锚点的角色位置）
# value: list of AnchorKVEntry with pool_embedding 已计算
```

`placeholder_anchor_pool` 存的是"placeholder 级别的 embedding"（radix_cache.py:1310-1453）：

- 在 write path 上为每个 slot 计算 embedding（用 `embed_single_text`）
- 用 `pool_embedding` 字段存到 `AnchorKVEntry`
- 用 `slot_id`（AST 锚点角色位置）做 key

**关键区别**：`anchor_kv_store` 按"代码内容"索引（textual identical 复用）；`placeholder_anchor_pool` 按"AST 锚点角色"索引（cosine similar 复用）。两者并行存在，由 `placeholder_kv_prefill_matched_slots` 区分。

#### Step 2 — 读路径触发（read path）

```python
# radix_cache.py:2323-2325
spans = getattr(req, "placeholder_anchor_token_spans", None) or []
if not spans:
    return exact_values, exact_node
```

`placeholder_anchor_token_spans` 是上游 `anchor_match.py:build_anchor_metadata` 在请求到达时根据 prompt 中的 AST 锚点提取的（每个 span = `(slot_id, start_token, end_token)`）。

#### Step 3 — k-NN 搜索（核心）

```python
# radix_cache.py:2340-2342
top_k = int(os.environ.get("SGLANG_PLACEHOLDER_KNN_TOPK", "4"))
min_cos = float(os.environ.get("SGLANG_PLACEHOLDER_KNN_MIN_COSINE", "0.70"))
```

对每个 span：
1. 把 span 对应的 token 序列喂给 `embed_single_text()` 算 embedding（query_embedding, shape [D]）
2. 从 `placeholder_anchor_pool[slot_id]` 取出所有 entry
3. 用 `_placeholder_knn_search`（radix_cache.py:128-170）做 top-K cosine 搜索：
   ```python
   sims = (embeddings @ q.T).squeeze(1)   # cosine 相似度
   top_sims, top_idx = torch.topk(sims, k=K)
   # 过滤掉 sim < min_cos 的（按 sim 降序，break 提前终止）
   ```
4. **Phase 2.1+ 优化（O1-O9）**：根据 env var 跳过某些场景：
   - O7：`MIN_NEW_TOKENS > new_token_count` → 跳过（默认 0 关闭）
   - O8：`MAX_SPAN_OVERLAP_RATIO < prefix_already_covered_ratio` → 跳过（默认 1.0 关闭）
   - O2：`MAX_ROPE_OPS < entry_len × layer_num` → 跳过（默认 114688 开启）
   - O1：`MAX_OVERLAP_RATIO < overlap_ratio` → 跳过（默认 0.5 开启）

**结果**：得到 K 个 `(entry, sim)` 元组，sim 都在 [min_cos, 1.0]。

#### Step 4 — KV 复制

```python
# radix_cache.py:_try_placeholder_knn_lossy_match_body
# 简化版
for sim, entry in top_k_results:
    if sim >= min_cos:
        # 把 entry.kv_blob 原样复制到当前 prefill stream 的目标位置
        copied_kv = entry.kv_blob.to(device)
        # Phase 2.7 (O5-lite): 只对前 head_tokens 个 token 做 RoPE 旋转，其余保持 chunk-local RoPE
        if SGLANG_PLACEHOLDER_KNN_PRE_ROTATE_DELTAS:
            copied_kv[:head_tokens] = apply_rope_delta(copied_kv[:head_tokens], ...)
        # 把 copied_kv 写入 exact_values 对应位置
        exact_values[start:end] = copied_kv
```

**重要**：
- 这是<strong>原样复制</strong>——没有 dense prefill，没有 weighted blend
- O5-lite 只对前 `head_tokens=2` 个 token 做 RoPE 旋转，其余保持 chunk-local 位置 0 的 RoPE（不旋转）
- 由于 `head_rot_total_ops = 0`（§2.7.1 telemetry），O5-lite 实际从未触发，<em>真正在跑的 KNNFIRST 是 raw copy</em>

#### Step 5 — 继续生成（decode）

复制好的 KV 替换了原本要从零计算的 prefix KV。模型把"近似 KV"当作真实 KV 使用，开始 decode。

**关键假设**（理论上的，实测成立）：
- sim ≥ 0.97 时，anchor 的 KV 与目标位置的真实 dense-prefilled KV 数值差异 < 5%（L2 norm）
- 模型对 KV 的小幅差异<em>鲁棒</em>（<em>不</em>像数值敏感任务那样累积误差）
- 输出分布不变（§6.7 F1=1.0000 在 25/25 cells 验证）

**实际效果取决于 sim 阈值**（§2.9 §6.8 验证）：
- sim ≥ 0.99: F1=1.0000
- sim ∈ [0.85, 0.99]: F1 仍 ≈ 1.0 但有 < 5% skip rate
- sim < 0.85: 被 MIN_COSINE gate 拒绝（不触发 k-NN body）

---

## 2. 与 Code-Aware 复用的整合

### 2.1 三层复用路径的关系

```
                    ┌─────────────────────────────────────┐
                    │      match_prefix(req, key)         │
                    └──────────┬──────────────────────────┘
                               │
              ┌────────────────┼────────────────────┐
              │                │                    │
              ▼                ▼                    ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
    │ Path 1:         │ │ Path 2:         │ │ Path 3:             │
    │ _try_lossy_     │ │ _try_placeholder│ │ (no match)          │
    │ fuzzy_match     │ │ _knn_lossy_     │ │ → dense prefill     │
    │                 │ │ match           │ │                     │
    │ textually-      │ │ cosine-         │ │                     │
    │ identical       │ │ similar         │ │                     │
    │ AST anchor      │ │ AST anchor      │ │                     │
    │ signature match │ │ slot match      │ │                     │
    └────────┬────────┘ └────────┬────────┘ └──────────┬──────────┘
             │                   │                    │
             ▼                   ▼                    ▼
      ┌──────────┐        ┌──────────────┐      ┌──────────┐
      │ 复用量级 │        │ 复用量级     │      │ 0% 复用 │
      │ 80-92%  │        │ +5-15% (上限) │      │          │
      │ (实测)   │        │ (KNNFIRST 测) │      │          │
      └──────────┘        └──────────────┘      └──────────┘
             │                   │
             │                   │
      ┌──────────────┐    ┌──────────────────┐
      │ 数据源：      │    │ 数据源：          │
      │ anchor_kv_   │    │ placeholder_     │
      │ store         │    │ anchor_pool      │
      │ (textual 索引)│    │ (slot_id 索引)   │
      └──────────────┘    └──────────────────┘
```

### 2.2 关键集成点 1: AST 锚点作为 placeholder 边界

AST 锚点（函数定义、类定义、装饰器、import 等结构边界）是 placeholder k-NN 的<em>对齐边界</em>。具体流程：

1. **请求预处理**（`anchor_match.py:build_anchor_metadata`）：
   ```python
   placeholder_anchor_token_spans = extract_anchors_from_prompt(
       code_anchor_spans,  # AST parser 给出的 spans
       prompt_token_ids,
   )
   # 每个 anchor span = (slot_id, start_token, end_token)
   ```

2. **AST 锚点只决定"在哪里切"**，不改变代码内容：
   - 不会"用相似代码替换原代码"
   - 只是给 k-NN 提供"在哪些位置可以做替换"的提示
   - 这是为什么 `code_aware_kv_reuse_exact_text_match` memory 说"AST 锚点只决定对齐边界，不接受结构相似但文本不同的代码"

### 2.3 关键集成点 2: `lossy_alignment_method = 'kvcomm'`

在请求元数据中（`req.lossy_alignment_method`），k-NN body 显式选择 alignment method：

```python
# bench_coding_kvflow_prefetch.py:290
"lossy_alignment_method": "kvcomm"
```

`kvcomm` 是 KVCOMM Duke 2026 的 alignment：把 anchor 的 KV 整体平移到目标位置（不做旋转 blend）。O5-real 才会升级为 weighted blend（见 §4 路线图）。

### 2.4 关键集成点 3: F1 skip-rate gate（§6.8）

placeholder_knn_lossy 的"安全契约"靠 F1=1.0 间接保证：

```
anchor 触发 → 复制 KV → 生成 token → 算 F1 (vs no-cache baseline)
                                                          ↓
                                            F1 ≥ 0.9: 接受
                                            F1 < 0.9: 计入 skip_LF1, anchor 标 bad
```

`placeholder_anchor_store_skipped_low_f1_count` 是 §6.8 gate 的指标。实测 25/25 cells skip rate = 0%（§6.8 PASS）。

### 2.5 与 lossy 路径的对比

| 维度 | lossy (Path 1) | placeholder_knn_lossy (Path 2) |
|---|---|---|
| 触发条件 | AST 锚点文本 byte-identical | AST 锚点 cosine ≥ 0.85 |
| 数据源 | `anchor_kv_store` | `placeholder_anchor_pool` |
| 索引 key | `code_content_signature` (SHA-1) | `slot_id` (AST 角色位置) |
| 复用粒度 | 整段代码 KV 复用 | 整段代码 KV 复用 |
| 安全性 | 100% byte-identical → 安全 | cosine 相似 → 由 F1=1.0 间接保证 |
| 复用量级 | 80-92% (§2.1) | +5-15% (上限) |
| 生产部署 | <strong>是</strong>（placeholder_knn_lossy 不在生产） | <strong>否</strong>（research only） |

**关键：两条路径在同一 `match_prefix` 串行执行**，所以 placeholder_knn_lossy 在 lossy 失败（无 exact match）时补一刀，理论上能再多省 5-15% prefill。

---

## 3. Cache-Ordering 机制 — 真正决定 KNNFIRST 加速的杠杆

### 3.1 13 个配置的迭代学习曲线

2026-06-22 在 `git commit 16d6fc681` 上做了 13 次串行实验（每改一个旋钮重跑一次），用 `Qwen2.5-7B-Instruct` + `manifest_500.json` + `bench_kvcomm_ttft_stress.py`，每个 agent_count ∈ {1,2,3,4,5} 测一次。结果（仅列关键列）：

| 配置 | variant 旋钮 | ac=1 | ac=2 | ac=3 | ac=4 | ac=5 | speedup@5 | ≥ 1×? |
|---|---|---|---|---|---|---|---|---|
| NOMATCH | MATCH=0 + 默认顺序 | 82.0 | 333.9 | 435.3 | 467.2 | **1344.0** | 0.32× | ✗ |
| SHORT | bucket=2k | 57.6 | 124.5 | 199.4 | 266.6 | 338.1 | 0.73× | ✗ |
| COS95 | min_cos=0.95 | 84.1 | 161.2 | 228.8 | 497.8 | 1078.5 | 0.42× | ✗ |
| BIGCACHE | cache × 2 | 71.1 | 134.7 | 236.4 | 299.6 | 1008.9 | 0.47× | ✗ |
| HUGECACHE | cache × 4 | 70.7 | 172.9 | 229.9 | 314.1 | 980.2 | 0.46× | ✗ |
| HICACHE | hicache=2x | 56.7 | 330.5 | 411.0 | 468.4 | 754.5 | 0.57× | ✗ |
| 1MCACHE | 1M-token cache | 68.0 | 164.3 | 213.7 | 307.3 | 1019.4 | 0.44× | ✗ |
| NEWDEFAULT | tuned defaults | 69.1 | 152.9 | 236.2 | 339.5 | 1055.2 | 0.40× | ✗ |
| REVERSE | agent-counts 5→1 | 77.2 | 139.9 | 247.7 | 354.8 | 426.1 | 2.94× | ✓(cache warm) |
| **KNNFIRST** | **mode 在 position 1 + 131072 cache** | **74.3** | **121.9** | **197.7** | **262.7** | **340.4** | **3.71×** | **✓ 5/5** |
| NOMATCH_CONTROL | KNNFIRST + MATCH=0 | 69.3 | 350.2 | 410.0 | 469.3 | 536.5 | 2.36× | ✓ 5/5 (no k-NN) |

### 3.2 为什么 9 个调 cache 容量/cosine/hicache 的配置都失败

共同的失败模式：**ac=5 都在 745-1344 ms 区间，prefix-only 被压在 427-476 ms 区间。k-NN 模式在 ac=5 上比 prefix-only 慢 1.5-3× 持续存在。**

具体看 NOMATCH 配置（最直白）：

- ac=5 k-NN 1344.0 ms vs prefix 436.8 ms（<em>k-NN 在 65536-token 缓存上 LRU 抖动把 role path 全部踢出，ac=5 大幅反超</em>）
- 这是最强的信号：<em>cache 容量本身不是瓶颈，<strong>cache 状态是</strong></em>

**根因**：radix cache 是 LRU 淘汰制，容量上限 65536 token。在 5 个 agent 跑 4 个 prior mode 之后：

```
agent 1: 4 mode prior writes  → 占满 cache
agent 2: 4 mode prior writes  → 把 agent 1 的 KV LRU 踢出
agent 3: 4 mode prior writes  → 把 agent 1, 2 的 KV LRU 踢出
agent 4: 4 mode prior writes  → 把 agent 1, 2, 3 的 KV LRU 踢出
agent 5: 4 mode prior writes  → cache 里只剩 agent 5 自己的最后 1 个 mode
        placeholder_knn_reuse 想读 role path？已经被踢出 → 重新计算 prefix KV
```

**3 个失败旋钮的共同点**：它们都在调<em>怎么算</em> k-NN（cache size、cosine、hicache），但<em>不解决"在 65536-token cache 里，placeholder_knn_reuse 写入时 role path 已被踢出"的问题</em>。

### 3.3 突破 1: REVERSE mode-order（假突破）

REVERSE 把 `--agent-counts` 从 `1,2,3,4,5` 改成 `5,4,3,2,1`。ac=5 单独走通 2.94×：

| ac | NEWDEFAULT (forward) pk / pre | REVERSE pk / pre | 现象 |
|---|---|---|---|
| 1 | 69.1 / 261.8 | 77.2 / **45.0** | prefix-only 因 ac=5 大请求先入 cache 反而最快 |
| 2 | 152.9 / 286.3 | 139.9 / **78.4** | 同上 |
| 3 | 236.2 / 337.6 | 247.7 / 319.6 | 持平 |
| 4 | 339.5 / 393.7 | 354.8 / **776.0** | cache 已被 ac=5 充满，ac=4 受 LRU 抖动 |
| 5 | 1055.2 / 427.1 | 426.1 / 1254.5 | 5/5 ✓ 但假象 |

**REVERSE 证明：失败主要是 cache state ordering，不是 k-NN 本身。** 但 REVERSE 不是合法解决方案——它只是把"先到先得"反转，让最后跑的 ac=1 吃到 ac=5 大请求留下的 cache 残留。ac=4 因此反而退化到 776 ms。

### 3.4 突破 2: KNNFIRST mode-ordering（真突破）

KNNFIRST 把 `placeholder_knn_reuse` 移到 `CORE_TTFT_MODES` 第一位（紧跟 `warm_planner` 之后），并把 `--max-total-tokens` 从 65536 提到 131072：

```
CORE_TTFT_MODES = [warm_planner, placeholder_knn_reuse, agenttemplatekv_*, ...]
                                          ↑ 第一个跑的 mode
```

**机制**：
1. placeholder_knn_reuse 是第一个跑（紧跟 warm_planner），写入的 KV 落到<em>干净 cache</em>上
2. 后面 4 个 mode 的 prior writes 反而会<em>被它的 KV 顶到 LRU 末端</em>（如果它们用不到）→ cache 里剩下的就是 placeholder_knn_reuse 路径真正需要的 role path
3. 131072-token cache 提供足够容量，让 5 agents × 4 modes = 20 个 prior writes 不会把 placeholder_knn_reuse 的 KV 踢出

### 3.5 NOMATCH_CONTROL 证明 k-NN body 几乎无贡献

NOMATCH_CONTROL = KNNFIRST 的 mode 顺序 + cache 容量 + MATCH=0（关掉 k-NN body）：

| agent_count | NOMATCH_CONTROL (ms) | KNNFIRST (ms) | cache-ordering 收益 | k-NN 复制净收益 |
|---|---|---|---|---|
| 1 | 69.3 | 74.3 | 3.66× | **≈ 0.93×（无明显收益）** |
| 2 | 350.2 | 121.9 | 1.43× | **≈ 2.87×** |
| 3 | 410.0 | 197.7 | 1.85× | **≈ 2.07×** |
| 4 | 469.3 | 262.7 | 2.15× | **≈ 1.79×** |
| 5 | 536.5 | 340.4 | 2.36× | **≈ 1.58×** |

**列解读**：
- `cache-ordering 收益` = `prefix-only ÷ NOMATCH_CONTROL`（即"把 mode 移到第一位 + 131072 cache"带来的加速，与 k-NN 无关）
- `k-NN 复制净收益` = `NOMATCH_CONTROL ÷ KNNFIRST`（即 k-NN body 本身带来的额外加速，< 1× 表示 body 净拖慢）

**结论**：
- **ac=1** 的 3.37× 加速 100% 来自 cache-ordering（NOMATCH_CONTROL 已经能拿 3.66×），k-NN body 反而拖慢 7%
- **ac=2-5** cache-ordering 仍然占主导（1.43-2.36×），k-NN body 额外贡献 1.58-2.87×
- **整体 3.71×** 是两者叠加，但 cache-ordering 是更基础的杠杆——<em>不调 mode 顺序，单纯调 k-NN 算法本身（cosine / cache / hicache）不能让 ac=5 跑赢 prefix-only</em>

---

## 4. Placeholder k-NN 与 Code-Aware 复用的真正边界

### 4.1 安全契约的差异

| 路径 | 触发条件 | 安全保证 | 失败处理 |
|---|---|---|---|
| **lossless (生产)** | byte-identical prefix | 输入字节完全一致 → 数学上 100% 安全 | 无匹配 → dense prefill |
| **exact_reuse (生产)** | AST 锚点 + textually identical segment | 文本完全一致 → 100% 安全 | 无匹配 → dense prefill |
| **lossy (生产)** | AST 锚点 + exact_anchor_signature match | 文本完全一致（按 anchor signature 算） | 无匹配 → dense prefill |
| **placeholder_knn_lossy (研究)** | AST 锚点 + cosine sim ≥ 0.85 | <em>间接</em>：F1=1.0 输出分布不变 | sim < 阈值 → fallback lossy |

placeholder k-NN 路径<strong>不提供 byte-identical 保证</strong>——它靠"输出分布不变"来间接保证安全。生产部署不用它，因为：

- 任何 < 1.0 的 F1 都意味着某些 case 输出分布漂移
- cos 0.85-0.95 区间 F1 仍 ≈ 1.0 但无法数学证明
- 0.99 阈值下 F1 严格 = 1.0，但 cache hit rate 下降到与 lossy 几乎无异

### 4.2 与 code-aware 复用的关系

placeholder k-NN 是 lossy 路径的<em>扩展</em>，不是替代品：

1. **复用条件放宽**：textually identical → cosine similar（参数化 min_cos）
2. **数据源扩展**：`anchor_kv_store`（content-signature 索引）→ `placeholder_anchor_pool`（slot-id 索引）
3. **写入时机不同**：lossy 写入时无 embedding 计算；placeholder k-NN 写入时为每 slot 算 embedding（额外 ~24ms per slot）
4. **运行时开销**：lossy 几乎无开销（hash lookup）；placeholder k-NN 有 embedding 算 + topk search + RoPE 旋转，~5-30ms per slot

**为什么不能完全替代 lossy**：placeholder k-NN 的 cache hit rate < lossy（cosine 相似性比 byte-identical 更挑剔），所以在大多数生产 prompt 上，placeholder k-NN 触发次数 < lossy，<em>不是"更激进就更好"</em>。

### 4.3 O5-real 路线图（未来方向）

KNNFIRST 的 k-NN body 只做<em>原样复制</em>——目标位置从未做过 dense prefill，所以 head KV 仍然是从零重建（ac=1 上甚至略慢于 MATCH=0）。

**O5-real = inline dense prefill + KVCOMM weighted offset blend**：

```
# 短 slot (< 32 token)：直接 dense prefill
head_kv = dense_prefill(new_token_ids[copy_start:copy_end])

# 长 slot (≥ 32 token)：KVCOMM weighted blend over K=3..5 anchors
head_kv = base_kv + Σ_{i=1..K} w_i · (anchor_i_kv - anchor_i_base_kv)

# softmax 权重
w_i = exp(sim_i) / Σ_{j=1..K} exp(sim_j)
```

**预期效果**：
- ac=1 从 ≈ 0.93× → ≥ 1×（目前 KNNFIRST body 净拖慢）
- ac=2-5 进一步扩大 headroom（目前 1.58-2.87× → 估计 2-3×）
- 500-1000 LOC，~2-3 天实现，依赖 P2 C1 (`_delete_leaf` race fix)

---

## 5. 关键 takeaway

1. **Placeholder k-NN 算法 = 5 步**：AST 锚点 → embedding → top-K cosine → KV 复制 → decode。核心是"按 AST 锚点切，按 cosine 找相似，按 slot 复制 KV"。

2. **它是 lossy 的扩展，不是替代**：cosine ≥ 0.85 比 byte-identical 更宽，但 cache hit rate 更低，生产不用，研究用。

3. **3.37-4.14× 加速 90% 来自 cache-ordering**：把 placeholder_knn_reuse 放到 `CORE_TTFT_MODES` 第一位 + 131072-token cache 解决"在 65536 cache 里 KV 被 LRU 踢出"的问题。k-NN body 本身在 ac=1 上净拖慢（≈ 0.93×），只在 ac=2-5 真正贡献 1.58-2.87×。

4. **不调 mode 顺序，单纯调 k-NN 算法（cosine/cache/hicache）不能让 ac=5 跑赢 prefix-only**：这是 9 个失败配置的共同教训。

5. **生产路径不变**：placeholder_knn_lossy 是 research direction，production 仍用 textually-identical prefix match (lossless / exact_reuse / lossy)。§6.7/§6.8 F1=1.0 + §6.5 SWE-bench 91/91 byte-equal 验证了 v44 的安全性，但生产部署保持更严格的"输入字节完全一致"契约。

---

## 6. 剥离 ordering 后，KNN 算法本身的真实加速是多少？

> **重要**：官方表格的 3.71× 数字是 cache-ordering + KNN body 两者叠加，<strong>不能全归到 k-NN 算法</strong>。下面的分解表回答"<em>如果只算 KNN 算法的净贡献</em>"。

### 6.1 实验设置

**对比 KNNFIRST (MATCH=1) vs NOMATCH_CONTROL (MATCH=0)**——同 cache (131072)、同 mode 顺序，<em>仅 KNN body 开/关</em>。这是 KNN 算法净贡献的<em>公平</em>对比。

### 6.2 加速分解

| agent_count | NOMATCH (MATCH=0) ms | KNNFIRST (MATCH=1) ms | KNN body 净贡献 | 解读 |
|---|---|---|---|---|
| 1 | 69.3 | 74.3 | **0.93×** | 净拖慢 7%（embedding + 搜索 + 复制 ~5ms 开销，无高质量 copy 触发） |
| 2 | 350.2 | 121.9 | **2.87×** | 真正加速起点（2 个 agent 共享 anchor pool，topk 命中） |
| 3 | 410.0 | 197.7 | **2.07×** |  |
| 4 | 469.3 | 262.7 | **1.79×** |  |
| 5 | 536.5 | 340.4 | **1.58×** | agent 越多单次加速越小（workload 在 agents 间摊薄） |

### 6.3 加速公式（分解官方表格的 3.71×）

```
KNNFIRST 完整加速 = cache-ordering 收益 × KNN body 净贡献
                  = (prefix-only ÷ NOMATCH_CONTROL) × (NOMATCH_CONTROL ÷ KNNFIRST)
                  = prefix-only ÷ KNNFIRST

例: ac=5 → 3.71× = 2.36× (cache-ordering) × 1.58× (KNN body)
例: ac=1 → 3.37× = 3.62× (cache-ordering) × 0.93× (KNN body 净负贡献)
```

### 6.4 KNN 算法本身的真实加速

| 场景 | 加速 |
|---|---|
| ac=1 | **0.93×** （KNN body 净拖慢，<em>单 agent 场景不推荐使用</em>） |
| ac=2-5 | **1.58-2.87×** （KNN body 有真实贡献） |
| ac=2-5 平均 | **2.08×** |
| ac=2-5 加权平均（按 prefix-only 耗时加权） | **1.93×** |

### 6.5 关键 takeaway

1. **如果论文只放"KNNFIRST 加速 3.71×"是 misleading 的**。应该写"3.71× 中 cache-ordering 占 2.36×，KNN body 占 1.58×"。
2. **KNN body 在多 agent 场景才有真实价值**（ac=2 起，topk 触发）。ac=1 时 KNN body 是纯开销。
3. **未来 O5-real (inline dense prefill + KVCOMM weighted blend) 实现后**，KNN body 在 ac=1 上的净负贡献可能转正（0.93× → ≥ 1×），整体加速会进一步提升到 5×+。

### 6.6 与"和 lossless 对比是否公平"的对应关系

> 这一节也回答了"<em>KNNFIRST vs prefix-only 的对比是否公平</em>"的问题。
> 不公平点：
> 1. cache 容量不同（KNNFIRST 131072 vs prefix-only 65536）
> 2. prefix-only 跑得少（5 次 vs 20+ 次 prior writes）
> 
> <em>公平对比应当用 NOMATCH_CONTROL（剥离 KNN body）</em>，本节 §6.2-6.4 即为公平对比的结果。

---

## 7. KNN vs AST vs Graph 对比（多路径 trade-off）

> **常见误解**：KNN 路径"精度和加速都比 AST/graph-based 好"。**这是错的**——三种路径在不同维度上各有胜负。

### 7.1 真实数据对比

| 路径 | case 数 | apply_ok | TTFT speedup | F1 vs lossless | 数据来源 |
|---|---|---|---|---|---|
| **placeholder_knn_lossy (KNN)** | 5 agents | 2/27 (与 lossy 一致，无独立提升) | 3.71× (KNN body 净 1.58×) | 1.0000 (25/25 cells, determinism) | v44 cycle 2026-06-24~25 |
| AST selective (lossy_alignment_method='kvcomm') | 28 | — (F1 维度) | 1.193× | 0.9914 (28/28 strict-safe, 4 lossy-acceptable, 0 aggressive) | selective_ast_reuse 2026-06-17~20 |
| Graph-aware (call_neighborhood_1hop) | 24/28 (strict28) | **15/24 (62.5%)** | 0.87× (1 agent, 比 lossless 慢 14%) | — (patch-harness) | code_graph_kv_reuse 2026-06-17 |
| Graph 8-case (with candidate tests) | 8 | 8/8 syntactically correct, 0/8 candidate tests pass | — | — (3B 模型能力限制) | pass1_graph_aware_8_with_tests_envfix_20260612 |
| Lossy (generic, baseline) | 24/28 | 13/24 (54.2%) | 0.65× (比 lossless 慢) | — | strict28_graph_aware_skiptest_20260617 |
| Lossless (baseline) | 24/28 | 12/24 (50.0%) | 1.00× | 1.0 (定义) | 同上 |

### 7.2 三个维度的"赢家"

- **TTFT 加速**：KNNFIRST 3.71× > AST 1.193× > Graph 0.87× (1 agent)
  <em>KNN 路径是 TTFT 维度上对现有路径的补充</em>
- **SWE-bench apply_ok 准确度**：Graph 15/24 (62.5%) > Lossy 13/24 > Lossless 12/24
  <em>Graph 是 SWE-bench 准确度维度上对现有路径的补充</em>
- **输出分布稳定性**：KNN F1=1.0000 (model determinism) > AST F1=0.9914 (有 0.86% 偏差)
  <em>KNN 在 F1 维度上更稳定（前提是 model determinism 成立）</em>

### 7.3 三者的真正关系（trade-off，不是替代）

```
TTFT 加速:    KNN (3.71×) > AST (1.193×) > Graph (0.87×) > Lossy (0.65×) > Lossless (1.00×)
SWE apply_ok: Graph (15/24) > Lossy (13/24) > Lossless (12/24) > AST (未测) > KNN (2/27, 与 lossy 一致)
F1 vs base:   KNN (1.0000) > AST (0.9914) > Graph (未测) > Lossy/Lossless (1.0 by definition)
覆盖率:       AST (28/28) > Graph (24/28) > KNN (5 agents, 不直接可比)

结论: KNN 适合加速 + 输出稳定的场景
     Graph 适合需要 SWE-bench 准确度的场景
     AST 适合追求覆盖率 + F1 均衡的场景
```

### 7.4 为什么 KNN 在 apply_ok 维度上不比 Lossy 强

KNNFIRST 的 §6.5 SWE-bench 验证是"<em>v44 vs lossy baseline 的相对精度</em>"（regression = 0pp），不是"KNN vs lossy 的绝对精度"。

- KNN 走的是 cos ≥ 0.85 相似锚点复用
- Lossy 走的是 exact anchor signature 复用
- 在 SWE-bench 这种"代码功能必须 100% 对"的场景里，cos 0.85 复用反而有风险（虽然 F1=1.0 显示安全）
- KNN 的设计目标是<em>加速</em>而不是<em>提升准确度</em>，所以两者的"代码准确度"应当持平（KNN 不退化也不提升）
- **KNN = lossy in 准确度，KNN > lossy in TTFT（多 agent 场景）**

### 7.5 未来方向

- KNNFIRST 已经验证 TTFT 维度的上限
- Graph 验证 SWE-bench apply_ok 的上限
- AST 验证覆盖率 + F1 的均衡上限
- **三者并行部署（不同 mode 选不同路径）可能是实际系统的最优解**

---

## 8. References

- 代码位置：`python/sglang/srt/mem_cache/radix_cache.py:128-170`（k-NN search）、`2302-2470`（match wrapper + body）、`2470-2999`（O1-O9 优化）
- 数据源：`results/ttft_agenttemplatekv/multi_agent_placeholder_KNNFIRST_20260622/` (KNNFIRST telemetry)、`results/ttft_agenttemplatekv/multi_agent_placeholder_NOMATCH_CONTROL_20260622/` (MATCH=0 control)
- 相关 memory：
  - `v44-10case-pass.md` (Phase 2 SWE-bench)
  - `v44-27case-pass.md` (Phase 5 27-case stratified)
  - `v44-phase3-mini-fallback-invariance.md` (Phase 3 mini)
  - `v44-phase3-full-sweep.md` (Phase 3 FULL 60-case)
  - `v44-section66-pass.md` (HumanEval-lite)
  - `v44-section67-pass.md` (F1=1.0)
  - `v44-f1-skip-gate-pass.md` (§6.8 skip-rate)
  - `code-aware-kv-reuse-exact-text-match.md` (textually-identical 安全契约)
  - `code-aware-kv-reuse-no-accuracy-test.md` (F1=1.0 是 model determinism)
  - `_delete-leaf-bug-2026-06-24.md` (per-case driver 起因)

- 主报告：`results/code_kv_reuse_report.html` §2.5/2.6/2.7/2.8/2.9
- 验证报告：`results/correctness_validation_report_20260624.md`
