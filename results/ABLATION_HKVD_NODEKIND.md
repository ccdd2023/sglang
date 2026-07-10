# HKVD-by-node-kind - Decisive Mechanism Measurement

**Date**: 2026-07-10
**Status**: ❌ **STRUCTURE SIGNAL NOT REAL at KV level - "code structure decides recompute" line dead (incl. P3)**
**Script**: `results/hkvd_by_node_kind_20260710/measure_hkvd_by_node_kind.py`
**Outputs**: `results/hkvd_by_node_kind_20260710/hkvd_by_node_kind.json` + `_per_chunk.jsonl`

---

## TL;DR

The decisive experiment for the entire "code-aware lossy KV recompute" research line. Direction A (contiguous node-kind) was FALSIFIED at equal budget (-3.3pp vs R32), Direction B (dataflow) FALSIFIED at P0, but neither told us whether the **structure signal exists at the KV level**. This script measures it directly:

**Question**: Under the canonical→live prefix swap (the real pool-precompute scenario), do AST interface tokens (signature + docstring) drift MORE than body tokens in their KV state?

**Answer**: **NO.** Body tokens drift slightly MORE than interface tokens (K_dev). The Direction A hypothesis is not just unsupported - it is **inverted**.

```
              interface    body      delta(iface-body)
K_dev overall   0.0843    0.0886     -0.0043   (body +5.0% more)
V_dev overall   0.0100    0.0061     +0.0039   (iface higher, but V_dev ~6× smaller)

paired (n=40 chunks): mean_delta = -0.0043, std=0.0066
  iface>body: 9   body>iface: 31   tie: 0
  Wilcoxon one-sided (iface>body) p = 0.9999   <- hypothesis strongly REJECTED
```

**Implication**: Direction A recomputed the K-insensitive interface and copied the K-sensitive body - the **worst** selective strategy. This mechanically explains the -3.3pp falsification. More broadly, AST node-kind (interface vs body) carries **no usable KV-deviation signal**, so the "code structure decides what to recompute" line is dead **including P3 (True CacheBlend with AST-targeted per-token mask)** - the targeting signal it would exploit does not exist.

---

## 1. 方法

### 1.1 测量定义

对每个有 AST interface boundary 的 pool chunk C（pandas_15case_v1, 120 chunks, 100% fire rate），把 C 的 token 分成两组：

- **interface tokens** = `[chunk_start, interface_end_byte)` - signature + leading docstring（Direction A recompute 的部分）
- **body tokens** = `[interface_end_byte, chunk_end)` - function/class body（Direction A copy 的部分）

每组测：

```
deviation_g = 1 - cosine( KV(g | canonical_prefix), KV(g | live_prefix) )
```

averaged over layers + chunks。`canonical_prefix` = pool 预计算用的占位符 preamble（ROLE/CASE/UPSTREAM literal）；`live_prefix` = 填入真实 role/case/upstream 的 preamble。这是 pool precompute 的**真实场景**（pool 用 canonical，runtime 用 live）。

### 1.2 实现

- HuggingFace Qwen2.5-Coder-7B 直接 forward（不依赖 sglang，同 `measure_hkvd_by_position.py` 模式）
- `ASTChunker` 拿 `interface_end_byte`（chunk text 坐标系，与 `compute_nodekind_budget.py` 一致）
- 采样 40 chunks（8 class + 32 function），deterministic seed=42
- per-chunk paired：同一 chunk 的 interface_dev − body_dev，消除 chunk-level variance
- Wilcoxon signed-rank one-sided（H₁: interface_dev > body_dev）

### 1.3 判决逻辑

- `interface_dev > body_dev` → structure signal real → P3 (per-token mask) 有动机
- `interface_dev ≤ body_dev` → structure signal NOT real → 整条 code-structure-recompute 线死亡，含 P3

---

## 2. 结果

### 2.1 Overall

| group | n | K_dev | V_dev | K_std |
|---|---|---|---|---|
| interface | 40 | 0.0843 | 0.0100 | 0.0053 |
| body | 40 | 0.0886 | 0.0061 | 0.0046 |

K_dev: **body > interface** by 5.0%（与 Direction A 假设相反）。
V_dev: interface > body，但 V_dev 整体比 K_dev 小 ~6×（0.006-0.010 vs 0.084-0.089），attention 匹配主要由 K 决定，K_dev 是主导信号。

### 2.2 Per anchor_type

| type | group | n | K_dev | V_dev |
|---|---|---|---|---|
| class | interface | 8 | 0.0877 | 0.0056 |
| class | body | 8 | 0.0878 | 0.0044 |
| function | interface | 32 | 0.0835 | 0.0111 |
| function | body | 32 | 0.0888 | 0.0065 |

- class: interface ≈ body（K_dev 几乎相等，0.0877 vs 0.0878）
- function: body > interface by 6.3%（K_dev 0.0888 vs 0.0835）- 信号主要来自 function chunks

### 2.3 Paired test（per-chunk）

```
n=40  mean_delta(iface − body) = -0.0043  std=0.0066
iface>body: 9   body>iface: 31   tie: 0
Wilcoxon one-sided (iface>body) p = 0.9999
```

31/40 chunks 的 body K_dev 高于 interface。interface > body 的假设被 p=0.9999 **强烈拒绝**。

---

## 3. 机制解释

### 3.1 为什么 body 的 K_dev 更高

推测（机制层面）：
- **body** 包含具体逻辑、`self.attr` 引用、函数调用 - 这些 token 的 **key** 需要编码"该关注哪些上下文"，而上下文（role/case）正是 prefix swap 改变的部分 → key 对 prefix 敏感
- **interface**（signature + docstring）是声明性的、自包含的 - 描述"函数签名是什么"，对 role/case context 不敏感 → key 稳定

### 3.2 为什么 Direction A 是 -3.3pp

Direction A 策略：**recompute interface, copy body**。

但 HKVD 显示：interface 的 K 稳定（deviation 低），body 的 K 敏感（deviation 高）。

所以 Direction A 恰好 recompute 了**不敏感的**部分，copy 了**敏感的**部分 - 即保留了 stale 的 body KV，重算了本来就稳定的 interface KV。这是**最坏** selective 策略，机械地解释了 -3.3pp vs R32（R32 至少随机 recompute 前缀，有概率覆盖敏感 body tokens）。

### 3.3 V_dev 的反方向

V_dev: interface (0.0100) > body (0.0061) - interface 的 **value** 更敏感。可能因为 value 编码"token 的语义内容"，而 signature/docstring 的语义内容（函数是什么）确实随 case context 变化。但 V_dev 整体小 6×，在 attention 匹配中不主导。这是个有趣的次级信号，但不足以支撑 selective recompute 策略（K 和 V 必须一起 recompute 或一起 copy，不能分开）。

---

## 4. 对研究路线的影响

### 4.1 三重证伪完成

| 实验 | 结果 | 证伪层面 |
|---|---|---|
| Direction A (contiguous node-kind) | -3.3pp vs R32 @ equal B | 算法层面：contiguous head 无法利用结构 |
| Direction B (dataflow) P0 | FRAC 0.156 (≈R32_f015) 或 0.845 (≈lossless) | 算法层面：contiguous head 无法表达 selective |
| **HKVD-by-node-kind** | interface_dev ≤ body_dev, p=0.9999 | **机制层面：结构信号在 KV 层不存在** |

前两个证伪了"contiguous head 机制"，但留下了"也许信号存在，只是机制错"的可能。HKVD-by-node-kind **直接否定了信号本身** - 即使有完美的 per-token 机制（P3 True CacheBlend），也没有 AST-based targeting 信号可利用。

### 4.2 P3 True CacheBlend 动机消失

P3 的原始动机：用 per-token mask 精准 recompute AST interface tokens（比 contiguous head 更精准）。但 HKVD 显示 interface tokens **不比** body tokens 敏感 - 精准 targeting interface 没有收益（甚至有害，如 Direction A 所示）。

P3 的通用机制（attention-kernel hook for per-token selective recompute）仍可能有用，但需要**非-AST**的 per-token 信号（如 attention-score-based importance、per-token HKVD）。这偏离了 "code-aware" 卖点 - 变成 generic per-token importance，失去 coding-MAS 的差异化。

### 4.3 "code-aware lossy KV" 卖点的现状

核心卖点 "用代码结构决定 recompute 什么" 失去机制支撑：
- AST interface/body → 无 KV deviation 差异（本实验）
- AST dataflow (cross-chunk globals) → contiguous head 无法表达（Direction B P0）
- AST node-kind → contiguous head -3.3pp（Direction A）

剩余的真实增益只有 **R32 uniform-along-position FRAC**（1.43× TTFT 换 ~13% type-match 一致性，见 P1'' retraction）- 这是 **position-aware** 不是 **code-aware**，且是 speed-accuracy 权衡非 accuracy-preserving。

---

## 5. 下一步研究方向（重新评估）

"code-structure-driven selective recompute" 路线已穷尽。可选新方向：

| 方向 | 依据 | 估时 | novelty |
|---|---|---|---|
| **per-token HKVD (non-AST)** | 直接测每个 token 的 deviation，找真正敏感的 tokens（不分 interface/body） | 1-2 天 | 中 - 偏离 code-aware，generic per-token |
| **attention-score-based recompute** | 用 attention entropy / magnitude 决定 recompute 哪些 token | 1 周 | 中 - 已有 literature |
| **P4 R40 zmq pickle 修复** | 解锁 7 个 timing 字段 + chunk-pool telemetry | 半天-1 天 | 基础设施（无 novelty，但解锁测量） |
| **重新定位为 position-aware** | 承认 code-aware 无信号，专注 R32 position-proxy + HKVD-by-position（已 +7.2%） | - | 低 - 现有结果 |
| **换 benchmark / task** | verdict task 已饱和；试 code-gen task（R40-P2 git apply）看 code-aware 是否在生成任务有信号 | 多天 | 待测 |

**推荐**：先完成 P4（解锁测量基础设施），再用 per-token HKVD 探索"非-AST"信号是否存在。若 per-token HKVD 也无强信号，则整个 lossy-KV-reuse 线路应重新定位为纯 speed-accuracy 权衡（R32），研究重心转向 **prefill scheduling / KV cache management** 而非 **selective recompute**。

---

## 6. 引用

- `results/hkvd_by_node_kind_20260710/measure_hkvd_by_node_kind.py` - 测量脚本
- `results/hkvd_by_node_kind_20260710/hkvd_by_node_kind.json` - 聚合 + per-chunk
- `results/hkvd_by_position_20260709/measure_hkvd_by_position.py` - 前序（by-position, +7.2%）
- `results/compute_nodekind_budget.py` - interface boundary 提取（同源）
- `results/ABLATION_NODEKIND_REPORT.md` - Direction A FALSIFICATION
- `results/ABLATION_DATAFLOW_P0.md` - Direction B P0 FALSIFICATION
- `results/ABLATION_R32_F045_CONFIRMATION.md` - P1'' R32 speed-accuracy tradeoff
- CLAUDE.md §2c/§2d/§6 P0