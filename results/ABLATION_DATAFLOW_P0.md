# P1′ P0 — Dataflow (Direction B) Cheap Signal Report

**Date**: 2026-07-10
**Status**: ❌ **P0 PARTIAL → RECOMMEND FALSIFY (do not proceed P1)**
**Script**: `results/compute_dataflow_budget.py`
**Outputs**: `results/codebase_kv/pandas_15case_v1/dataflow_budget{,_per_chunk.jsonl}`

---

## TL;DR

Direction B（dataflow: recompute only tokens that reference cross-chunk globals）的 P0 cheap signal 检查结果：

| 信号维度 | 结果 | 评估 |
|---|---|---|
| Cross-use fire rate | **82/120 = 68.3%** chunks 有 cross-use tokens | ✅ 有信号 |
| Pure-dataflow FRAC (per-token mask) | **0.0507**（median 0.031, p90 0.146） | ⚠️ 远低于所有 baseline |
| Contiguous-head approx "first cross" | 0.156 ≈ R32_f015 | ⚠️ 不是 novel lever — 另一形态 uniform |
| Contiguous-head approx "last cross" | 0.845 ≈ lossless | ❌ 过度 recompute, 无 selective 优势 |
| 实现代价 | per-token mask 需 CacheBlend 多段重写 pool chunk | ❌ 多周工程 |

**结论**：Direction B 在「contiguous head」机制下**结构性等价**于已测过的 R32 sweep（uniform 沿位置），无法表达 selective per-token targeting。P1 路径需要 multi-segment CacheBlend 重写 per pool-chunk — 与 Direction A 的核心承诺（"code structure drives recompute"）相比风险/收益不对称。

**Recommendation**: Falsify Direction B at P0; do not proceed P1. Move to **P1'' R32_f045 confirmation**（CLAUDE.md §6 第二条，~3h, 高 yield）或 **P3 True CacheBlend attention-kernel hook**（multi-week，需 sign-off）。

---

## 1. 方法论

### 1.1 数据流定义

Chunk C 的 **cross-use set** = {names USED in C, Load context} ∩ {module-level globals defined in C's file} − {names redefined inside C}

- **USED in C**: `ast.Name(ctx=Load)` + `ast.Attribute(ctx=Load)` chain 的 base Name（filter 掉 builtins/dunders via `_BUILTINS` frozenset，~80 项）
- **Module-level globals**: `ast.walk(tree.body)` 收集 `FunctionDef/AsyncFunctionDef/ClassDef.name` + `Assign` LHS Names + `AnnAssign` LHS + `Import/ImportFrom` aliases
- **Local DEFs in C**: `ast.FunctionDef/ClassDef/Assign/AnnAssign` + `ast.arguments` formal params（任何 depth）

`cross_uses(C)` = USEs that are visible globals defined OUTSIDE this chunk's own body — i.e. references whose binding can change when another chunk in the same file is touched.

### 1.2 Token-level recompute mask

对每个 Name/Attribute-base 在 `cross_uses(C)`，取其 byte range（`(lineno, col_offset)` → `(end_lineno, end_col_offset)` 经 `line_byte_offsets`），merge 重叠区间，然后用现有 `_lookup_byte_offset` 风格 bisect_right on token end offsets 算覆盖的 unique token 数。`cross_use_frac(C) = n_covered / n_tokens(C)`。

### 1.3 Baselines

| Baseline | B (over total_chunk_len=24684) | 已有 ablation 表现 |
|---|---|---|
| Uniform 0.15 | 3702 | R32_f015 type_match = 6.6% |
| Uniform 0.30 | 7405 | R32_f030 = 9.8% |
| Uniform 0.45 | 11107 | R32_f045 = **11.5% ≈ lossless** |
| Node-kind interface | 6443 | Direction A = **6.6%**（FALSIFIED, -3.3pp vs R32） |

---

## 2. P0 结果

### 2.1 Cross-use 分布（per chunk）

```
chunks: 120  (with cross-use: 82, fire rate 68.3%)
B_dataflow = 1252  /  total chunk_len = 24684
dataflow overall FRAC = 0.0507

per-anchor-type:
        type     n  fire%  cross_tok  chunk_len    frac
       class     9 100.0%        404       7422  0.0544
    function   111  65.8%        848      17262  0.0491

frac distribution (per chunk):
       min = 0.0000
       p10 = 0.0000
       p25 = 0.0000
    median = 0.0308
       p75 = 0.0860
       p90 = 0.1463
       p95 = 0.1818
       max = 0.3056
      mean = 0.0554
```

**观察**：
- 9/9 class 全部 fire（class methods 大量 reference self/super/外部 globals）
- 73/111 function fire（35% 是纯 local，不引用任何 module global）
- 即使 fire 的 chunk，cross-use 也只占 3-15% 的 token（p90=0.146）

### 2.2 Contiguous-head 近似（Design A）

因为现有 radix_cache head-recompute 机制是 **contiguous prefix K**，不能表达 per-token mask，dataflow 必须近似为 contiguous：

#### Design A: K = last cross-use byte
```
total_K_DA = 19367  /  total_chunk_len = 22924
FRAC = 0.8448
min=0.087 p10=0.341 p25=0.688 p50=0.793 p75=0.943 p90=0.979 max=0.998
```

→ **几乎全 recompute**（>80%）。等价于"几乎 lossless"。**完全没有 selective 优势**。

#### Design A': K = first cross-use byte
```
total_K = 3582  /  total_chunk_len = 22924
FRAC = 0.1563
min=0.003 p10=0.015 p25=0.063 p50=0.165 p75=0.366 p90=0.647 max=0.828
```

→ FRAC = 0.156 ≈ uniform 0.15 ≈ R32_f015 baseline。但**没有任何结构信号**：只是「early prefix up to first reference」 — 与 R32_f015（uniform prefix）数学上同形，只是 K 的 per-chunk 来源不同。

### 2.3 与 HKVD 对齐

CLAUDE.md §6 P0 = HKVD-by-node-kind 实测。已有 HKVD 数据是 **by slot position (1-5)**，不是 per-chunk，无法直接对齐 dataflow 信号。但方向 A 的 FALSIFICATION 已经表明：即使有 AST 结构信号（interface byte），与 HKVD 的对齐也没买精度（Direction A -3.3pp vs R32）。

类比预测：dataflow 即使有「cross-use」结构信号，在 contiguous-head 约束下也无法转化为精度优势 — 因为它和 uniform 的差异只在 K 的选择上，不在 K 的「semantic 精准度」上。

---

## 3. 为什么 P1 风险/收益不对称

### 3.1 工程代价

要让 dataflow "真的 selective"，必须实现 **per-token mask**：
- 不能用现有 contiguous head K
- 必须用 CacheBlend-style 多段 recompute per pool-chunk
- 需要修改 `_build_chunk_plan` + `_apply_chunk_recompute`：每个 chunk 接受 `[start, end]` 列表而不是单一 K
- 改动 radix_cache.py 主 KV copy 路径，影响 cache hit 关键路径

预估：**1.5-2 周**（vs Direction A 的 ~3 天，因为需要新机制而不是只新增 env branch）。

### 3.2 期望收益

如果 per-token mask 成功：dataflow FRAC = 0.05（pure），5% selective recompute 比 30% uniform 少 6× B。
- 假设 dataflow accuracy @ 5% B ≥ uniform accuracy @ 30% B → **type_match 不退化 + TTFT 更快**（更少 token recompute）
- 但 5% B 已经远低于 R32_f015（6.6%）的 critical mass（CLAUDE.md §2a 显示 R32_f015 type_match 已经崩到 6.6%）
- 极可能：5% B 不够 → accuracy 跌穿 6.6% → 数据流"过度 selective"也 falsify

### 3.3 类比：Direction A 的教训

Direction A（contiguous interface-recompute）已经证伪：-3.3pp vs R32 @ equal B。即使 AST 结构信号存在（100% interface fire rate，R34 教训 #1 命中），在 contiguous-head 约束下也无法转化为精度优势。

Direction B（dataflow）有相同的结构性限制：code structure signal 存在（cross-use fire rate 68%），但表达机制（contiguous K）无法利用这个 signal。

### 3.4 R34 教训再核对

R34 retired 的根本原因（CLAUDE.md §6 P1 历史）：「pandas 0.x untyped；gate effect = global FRAC bump」 — 也就是说 type-annotation signal 存在，但当表达机制把它转换为 uniform FRAC 时，所有 signal 都被 wash out 成"提高总体 B"。

Dataflow 在 contiguous-head 约束下会落入相同的陷阱：per-chunk signal 被 wash 成"per-chunk K 的小调整"，与 uniform sweep 无法区分。

---

## 4. 决策

**Falsify Direction B at P0. Do not proceed P1.**

理由：
1. contiguous-head 近似下 dataflow ≠ novel lever（要么 0.156 ≈ R32_f015，要么 0.845 ≈ lossless）
2. per-token mask 需 CacheBlend 多段重写 per pool-chunk，工程量 1.5-2 周
3. 期望收益边际（5% B 已在 R32_f015 critical mass 之下）
4. Direction A 的连续失败（FALSIFIED 2026-07-10）强烈预示 contiguous-head + structure signal 的组合无法买精度

**Next best leverage**（按 yield / cost 排序）：

| Option | Effort | Yield | 说明 |
|---|---|---|---|
| **P1'' R32_f045 确认** | ~3h | 中-高 | 次要 positive：n=15+OOM CI 宽，需复跑确认 R32@0.45 是否真稳定 ≈ lossless @ 1.43× |
| **P0 HKVD-by-node-kind 实测** | ~2h | 中 | CLAUDE.md §6 第一条；HKVD 验证 signature 节点 deviation — 即使 Direction A 证伪，HKVD 信号仍是机制层面的硬证据 |
| **P3 True CacheBlend attention-kernel hook** | 多周 | 极高（潜在 EuroSys-level novelty） | 真正 novel 路径，需 user sign-off |
| **P4 R40 zmq pickle 边界修复** | 半天-1 天 | 中 | 6 个 timing 字段 + chunk-pool telemetry 通道 |

---

## 5. 引用

- `results/compute_dataflow_budget.py` — P0 脚本（stdlib-only AST，无 sglang runtime 修改）
- `results/codebase_kv/pandas_15case_v1/dataflow_budget.json` — 聚合
- `results/codebase_kv/pandas_15case_v1/dataflow_budget_per_chunk.jsonl` — per-chunk detail
- `results/compute_nodekind_budget.py` — Direction A 等价脚本（参照实现）
- `results/ABLATION_NODEKIND_REPORT.md` — Direction A FALSIFICATION 报告
- CLAUDE.md §2c, §6 P0/P1/P1'/P1''