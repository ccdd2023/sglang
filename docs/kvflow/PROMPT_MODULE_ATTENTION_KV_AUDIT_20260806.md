# Prompt 模块级 Attention 热力图与 K/V Deviation 联合审计

日期：2026-08-06  
定位：基于相同 26 个当前方法真实 island 的只读机制分析；比较 Dense 全量计算与有损 KV splice，不是 accuracy 或线上 selector 结果。

## 结论先行

可以把完整 coding-agent prompt 分成稳定的语义模块，并在模块级比较 Dense 与有损 KV 复用后的 attention。现有 global-block artifact 已经覆盖完整 prompt，因此本轮不需要重跑 GPU。

最主要的发现有三点：

1. **全局模块路由基本保持。** Dense 与 Lossy 两张热力图肉眼接近；在复制之前的 System、Coding task、Context control 是机械负对照，差异严格为 0。
2. **变化集中在 repository evidence。** 复制之后，path-relevant repository evidence 对 copied evidence 的平均 attention 从 `15.37%` 降到 `14.83%`，即 `-0.54` 个百分点，是最大的模块级变化。
3. **K/V deviation 的作用是 module-conditional。** 在五类 suffix query 中，`attention to copied island × KV drift` 在四类中比裸 KV drift 更能解释 attention-row TV；但将模块混在一起后，相关性反而从 `0.680` 降到 `0.516`。因此不能再使用一个跨模块统一的 Attention×KV 标量，模块身份本身必须进入风险模型。

这为当前方法补充的是“哪类 coding 信息怎样读取旧 cache”的机制证据，仍然不能替代官方 execution accuracy。

## 1. Prompt 怎样分模块

每个 token 先按真实 chat/tool block 定位，再映射到九类模块：

| 模块 | 内容例子 | Coding 语义 |
|---|---|---|
| System prompt | agent 规则、bash/tool schema | 全局行为约束 |
| Coding task | issue 描述、用户目标、repository task | 当前需要解决的问题 |
| Context control | history compaction notice | 上下文管理信息 |
| Agent action | assistant reasoning 后产生的 tool call | 下一步操作与命令 |
| Copied repo evidence | 当前方法实际复用的只读 observation island | 有损 KV 干预对象 |
| Path-relevant repo evidence | 路径与最新 coding action 相交的其他 observation | 与当前代码路径相关的信息 |
| Other repo evidence | 路径不相交的 read observation | 其他 repository 内容 |
| Tool/runtime feedback | 测试输出、traceback、命令结果等 | 环境反馈与执行状态 |
| Next action | prompt 末尾的 generation marker | 模型即将生成的 action |

热力图的行是 **query module**，列是 **被读取的 key module**。例如 `Relevant evidence → Copied evidence = 15.37%` 表示 path-relevant observation 的 query token 平均把 15.37% attention 分配给当前复制的旧 observation。

## 2. Dense 与 Lossy 的公平比较口径

Dense 是在当前完整 prompt 下重新计算全部 K/V。Lossy arm 使用当前保守单-island 方法：source K 做 RoPE 位置修正、source V 直接复制、island 前后 Dense 计算。

必须处理一个容易误画的问题：Lossy arm 不会在 target 时重新执行 copied island 自己的 query 行，因此这行不存在，而不是 attention 等于 0。本报告采用：

- copied island **不作为 Dense/Reuse 对比的 query 行**；
- copied island继续作为 key 列，测量后续模块怎样读取它；
- Full-prompt 图保留所有两臂共同执行的 query 模块；
- Post-copy 图只保留 island 之后真正可能受干预影响的 query block；
- 五层、全部 query token 做 token-weighted 聚合，不保存难以解释的逐 token 方阵。

机械检查：

| 检查 | 结果 |
|---|---:|
| Dense/Reuse 匹配 case | 26/26 |
| Category row-sum 最大误差 | `5.69e-9` |
| Prefix 负对照最大差异 | `0` |
| Post-copy block-layer rows | 750 |
| Case-module 聚合点 | 86 |

### 2.1 完整 Prompt、切分边界与复用文本附件

为了能够直接核对“模型看到了什么”和“哪一段没有重新计算”，本报告新增了 [26 例完整 Prompt 附件](PROMPT_MODULE_ATTENTION_KV_FULL_PROMPTS_20260806.md)。它不是从 trajectory 摘录出来的近似文本，而是使用冻结 `source_input_ids`、`target_input_ids` 和实验 tokenizer 原样 decode，保留 `<|im_start|>` 等 chat-template special tokens。

附件对每个 case 都写入：

1. 完整 Source prompt：旧 K/V 实际建立时的全部输入；
2. 完整 Target prompt：Dense 与 Lossy 两臂共同使用的全部输入；
3. Source/Target 的逐模块 token 区间表；
4. source 与 target 中完全相同的 reused token span；
5. 单独列出的实际复用文本；
6. 加入 `DENSE_TARGET_PREFIX`、`LOSSY_REUSE`、`DENSE_TARGET_SUFFIX` 标记的完整 Target prompt；
7. token hash、文本 hash、RoPE position delta 和 repository path。

这 26 例总计包含 `113,896` 个 source token、`114,004` 个 target token和 `28,711` 个 copied token。机械检查确认：

| 完整 Prompt 检查 | 结果 |
|---|---:|
| Source/target reused `input_ids` 完全相同 | 26/26 |
| Source/target block 无缝覆盖完整 prompt | 26/26 |
| 删除审计标记后重建原始 target decoded prompt | 26/26 |

例如附件中的 Case 1：

```text
Source prompt: 5,388 tokens
  Dense source prefix [0, 1308)
  Source K/V origin   [1308, 3600)   <- astropy/timeseries/sampled.py
  Dense source suffix [3600, 5388)

Target prompt: 5,584 tokens（Dense 与 Lossy 的 token 完全一致）
  Dense target prefix [0, 1211)
  Lossy K/V reuse     [1211, 3503)   <- source [1308, 3600)
  Dense target suffix [3503, 5584)

RoPE position delta = 1211 - 1308 = -97
K: shifted by -97 positions
V: copied unchanged
```

`[[LOSSY_REUSE_BEGIN ...]]` 等标记只存在于附件，方便人阅读；它们没有发送给模型。原始、无标记字符串与逐 case 结构化数据也保存在 [FULL_PROMPTS.jsonl](assets/prompt_module_attention_kv_20260806/FULL_PROMPTS.jsonl)，紧凑索引在 [PROMPT_INDEX.csv](assets/prompt_module_attention_kv_20260806/PROMPT_INDEX.csv)。

## 3. 完整 Prompt 模块热力图

![完整 Prompt 的 Dense、Lossy 与差值热力图](assets/prompt_module_attention_kv_20260806/01_full_prompt_module_heatmaps.png)

前面三个 query 模块完全不变是因果结构决定的：System、Task 和 Context 位于 copied island 之前，causal attention 不会被后面的 cache 替换反向影响。这是一项 instrumentation 负对照，不表示这些模块在任意复用位置都“天然安全”。

全局图中最大的差异仍是：

```text
Path-relevant evidence → Copied evidence: -0.20 percentage points
Next action           → Copied evidence: -0.12 percentage points
Tool feedback         → Copied evidence: -0.06 percentage points
Agent action          → Copied evidence: -0.05 percentage points
```

这个全局均值包含同类模块在 island 前后的 query，因此会被严格相同的 prefix rows 稀释。判断有损复用的直接影响，应继续看 post-copy 图。

## 4. 只看复制之后的因果影响区

![复制后的模块热力图](assets/prompt_module_attention_kv_20260806/02_post_copy_module_heatmaps.png)

| Post-copy query 模块 | Dense→Copied | Lossy→Copied | 差值 |
|---|---:|---:|---:|
| Agent action | 11.83% | 11.72% | -0.11 pp |
| **Path-relevant repo evidence** | **15.37%** | **14.83%** | **-0.54 pp** |
| Other repo evidence | 9.90% | 9.91% | +0.00 pp |
| Tool/runtime feedback | 6.82% | 6.61% | -0.21 pp |
| Next action | 4.49% | 4.37% | -0.12 pp |

Path-relevant evidence 不仅最强地读取 copied evidence，也发生最大的平均注意力下降。Lossy 后这部分 mass 主要重新分配给：

- path-relevant evidence 自身：`+0.18 pp`；
- agent action：`+0.18 pp`；
- coding task：`+0.08 pp`；
- system prompt：`+0.07 pp`。

这不是“模型主动发现旧 cache 不可靠”的证明。它只说明 source-time K 搬到新 prefix 后，query 对 copied island 的相对 compatibility 略有下降，attention mass 被 softmax 重新分配。

## 5. 哪些模块依赖 copied island，哪些模块变化更大

下面先在每个 case 内跨五层和同类 query block 做 token-weighted 聚合，再对 case 取中位数，避免把 750 个 block-layer row 当成完全独立样本。

![模块依赖与 attention 路由变化](assets/prompt_module_attention_kv_20260806/03_module_dependency_and_tv.png)

| Query 模块 | 覆盖 case | Dense 对 copied evidence 的中位 attention | Dense↔Lossy row TV 中位数 |
|---|---:|---:|---:|
| Agent action | 26 | 11.27% | 0.339 pp |
| **Path-relevant evidence** | 14 | **16.24%** | **0.463 pp** |
| Other evidence | 8 | 9.33% | 0.154 pp |
| Tool feedback | 12 | 6.95% | 0.433 pp |
| Next action | 26 | 4.38% | 0.463 pp |

这张表说明 attention-to-copy 与完整 row TV 不是同一个量。Next action 对 copied evidence 的直接 mass 只有 4.38%，但其完整分布 TV 与 relevant evidence 同为约 0.463 pp，因为它还会在 agent action、relevant evidence、tool feedback 等列之间重新分配。

因此，比起只问“复制块被关注多少”，更准确的问题是：

> 当前模块读取 copied island 的强度、K/V 发生的偏移、以及它对其他 prompt 模块的竞争关系，共同造成多少路由变化？

## 6. 与 K/V Deviation 联合分析

每个 case、每层使用同一个实际 copied island 的：

```text
raw drift = max(mean K cosine drift, mean V cosine drift)
module-weighted drift = Dense module attention to copied island × raw drift
target = 该 query 模块 Dense 与 Lossy category distribution 的 TV
```

![不同模块中 K/V drift 与路由变化的相关性](assets/prompt_module_attention_kv_20260806/04_module_kv_correlations.png)

| Query 模块 | Raw drift → row TV | Attention×drift → row TV | 变化 |
|---|---:|---:|---:|
| Agent action | 0.894 | 0.893 | -0.001 |
| Path-relevant evidence | 0.930 | **0.978** | +0.048 |
| Other evidence | 0.881 | **0.952** | +0.071 |
| Tool feedback | 0.259 | **0.580** | +0.322 |
| Next action | 0.391 | **0.508** | +0.117 |

在四类模块内，加入 attention mass 后相关性更高；Agent action 基本不变。这支持“KV deviation 的局部后果由当前模块是否读取 copied island 调制”。

但如果忽略模块身份，把 86 个 case-module 点全部混在一起：

| Pooled score | 与 row TV 的 Spearman |
|---|---:|
| Raw KV drift | **0.680** |
| Attention×drift | 0.516 |

这不是矛盾，而是模块混合造成的异质性：不同 query 模块有不同的注意力基线、self-attention 结构和下游角色。一个统一乘法分数会把 module identity 丢掉，重现此前 request-level scalar aggregation 失败的问题。

## 7. 对当前算法的含义

本轮结果支持模块化风险描述：

```text
LocalRisk(module, island)
    = f_module(
        attention from this module to island,
        ΔK of island,
        ΔV of island,
        competing prompt modules
      )
```

而不支持：

```text
GlobalRisk = one_attention_number × one_KV_distance
```

对 coding-aware 方法最有价值的结论是：

1. path-relevant repository evidence 是当前样本中最依赖 copied island 的内容模块；
2. 它的 attention routing 也对有损复用最敏感；
3. tool/runtime feedback 虽然直接 attention 较低，但 attention weighting 对其风险解释提升最大；
4. system/task/context 作为 prefix 负对照验证了因果方向与 instrumentation；
5. 下一步应建立 module-conditioned K/V risk，而不是继续搜索一个跨模块统一 proxy。

这些指标依赖 Dense oracle K/V 和 full attention，目前只能用于 motivation 和 offline policy design。它们不能直接进入线上 SGLang，除非后续找到不抵消 TTFT 收益的廉价估计器。

## 8. 不能从热力图推出什么

- 热力图接近不等于生成代码功能正确；
- `-0.54 pp` 不等于 accuracy 损失 0.54%；
- row TV 与 KV drift 相关不等于能校准 task failure；
- 26 例来自同一冻结 cohort，相关性是 post-hoc mechanism evidence；
- path-relevant、tool feedback 等模块覆盖分别只有 14、12 或 8 个 case，需要 task-disjoint replication；
- 模型仍是 Qwen2.5-Coder-3B BF16 proxy，不是原生 30B SGLang attention。

最终方法仍需用相同 prompt 下的 TTFT 与官方 execution accuracy 晋级。

## 9. 复现入口

```text
benchmark/multi_workflow/analyze_prompt_module_attention_kv.py
benchmark/multi_workflow/build_prompt_module_full_prompt_appendix.py

docs/kvflow/PROMPT_MODULE_ATTENTION_KV_FULL_PROMPTS_20260806.md

docs/kvflow/assets/prompt_module_attention_kv_20260806/
  RESULT.json
  CASE_MODULE_ROWS.csv
  FULL_PROMPTS.jsonl
  PROMPT_INDEX.csv
  01_full_prompt_module_heatmaps.png
  02_post_copy_module_heatmaps.png
  03_module_dependency_and_tv.png
  04_module_kv_correlations.png
```

输入保持只读：

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_global_block_attention_20260806/frozen26_r2/
  impactkv_attention_kv_bound_20260806/frozen26_mass_aware/
```

本轮没有修改旧脏 checkout、paper、prefetch 或任何既有预注册门槛。
