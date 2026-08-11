# Coding-aware lossy KV reuse：显式依赖图与保守收益门控

日期：2026-08-11  
工作分支：`research/coding-aware-v45-multi-observation-20260803`

## 本轮要解决的问题

上一种在线方法会复用“成功读取、版本仍有效、之后没有出现同路径或同名符号消费者”的代码观察，并用平均 TTFT 回归值决定是否值得复制。Fresh8 中它确实执行了 53 次物理复制，且没有 fallback，但还有三个未解决的问题：

1. `__init__`、`__call__` 这类跨文件大量重复的平面符号会产生伪依赖；
2. “平均预计能省一点”不足以保证单个目标真的更快；
3. 一个请求最多复制三个岛，额外的控制、搬运和旋转开销可能吞掉小岛收益。

本轮没有改成 exact reuse，也没有加入 prefetch。仍然是在目标 prompt 的中间位置物理复制旧 K/V，其中 K 按目标位置做 RoPE 旋转，V 直接复制。

## 新的在线算法

### 第一步：只考虑自然出现的代码观察

候选必须同时满足：

- 是已经执行成功的只读工具观察；
- 对应一个明确的仓库文件；
- 不是全仓搜索、测试输出、修改反馈或助手推理；
- 此后没有使该文件版本失效的写操作；
- 源片段在源 prompt 和目标 prompt 中都只有一个明确位置。

这些条件沿用已有的 fail-closed 文件版本协议。

### 第二步：从已经展示给模型的代码构造一跳依赖图

对于 Python 观察，解析以下可见信息：

- 规范化路径，例如 `pkg/parser.py`；
- 类、函数和方法的限定名，例如 `Parser.parse`；
- import 与别名，例如 `from pkg.parser import Parser as P`；
- 直接调用和名称引用，例如 `P().parse()`。

只读取 prompt 中已经存在的文本，不扫描隐藏 checkout，也不读取未来工具结果。Python 片段不完整或解析失败时，退回较保守的路径和词法匹配。

如果后续可见历史存在以下任一关系，旧代码被视为 dependency-hot，目标请求重新计算它：

1. 精确路径相同；
2. 引用了源代码的限定符号；
3. import 一跳到源模块或源符号；
4. 直接调用源代码声明的非歧义符号。

否则它是 dependency-cold，才有资格进入 lossy KV pool。孤立的 `__init__`、`__call__` 等 dunder 名称不能单独造成跨文件 hot。

### 第三步：用 TTFT 下界而不是均值决定是否复制

校准数据来自已经完成的 56 个 exact-prompt 目标，每个目标 Dense/reuse 各测三轮中位数。五折验证按 agent task 分组，避免同一轨迹的相邻请求同时进入训练和验证。

拟合模型为：

```text
predicted_saving_ms
  = 0.1572862349 × (island_tokens × prompt_tokens / 10000)
    + 0.2543561958
```

任务分组交叉验证残差的第 10 百分位为 `-78.79832246 ms`，所以在线门控使用：

```text
lower_bound_saving_ms = predicted_saving_ms - 78.79832246
reuse only if lower_bound_saving_ms > 0
```

这不是统计置信区间，而是预先冻结的工程保守下界。它对应约 `island_tokens × prompt_tokens > 4.99M token²`；旧均值门控约为 `1.11M token²`，新门控约严格 4.5 倍。

### 第四步：目标请求只复制一个岛

pool 仍最多保留三个在线自然来源，以便较长的后续 prompt 可能跨过收益门槛；但一个目标只选择下界收益最高且不重叠的一个岛。这样把比较对象固定为“一个可信的代码模块”，并减少多岛控制开销。

## 伪代码

```text
for each completed request:
    candidates = successful_version_valid_single_file_code_reads(history)
    for source in candidates:
        graph = parse_visible_python_only(source)
        if visible_one_hop_consumer(graph, later_history):
            protect_for_dense_recompute(source)
        else:
            keep_in_pool(source, max_pool=3)

for the next target request:
    revalidate_file_version_and_graph_relation(pool)
    for source in valid_cold_pool:
        lcb = frozen_ttft_lower_bound(source_tokens, target_prompt_tokens)
    choose at most one nonoverlapping source with lcb > 0
    physically_copy_V_and_rope_rotate_K()
```

## 已完成的实现验证

核心策略、桥接模型和限额补丁捕获共有 78 个回归测试全部通过。新增测试覆盖：

- import alias 和限定符号解析；
- dunder 跨文件伪依赖排除；
- source-time 与 target-time 图关系复核；
- 正均值但负下界的小岛拒绝；
- 多个可用来源时目标只选择一个岛；
- `LimitsExceeded` 或空 `Submitted` 时捕获工作区 `git diff`。

## 冻结历史上的 shadow audit

在旧 Fresh8 的 8 个任务、210 个请求上，平面选择器和新图选择器只在 5 个请求上发生差异，去重后只有两个“任务 × 来源 × 方向”差异：

| 观察 | 数量 |
|---|---:|
| 请求总数 | 210 |
| 有任意选择差异的请求 | 5 |
| 旧方法判冷、新图判热 | 2 |
| 新图判冷、旧方法判热 | 3 |
| 去重差异来源 | 2 |

关系触发累计以精确路径为主：精确路径 526、词法 fallback 173、一跳 import 9、直接调用 1。候选重复评估中，完整 AST 解析 25 次，片段不完整而回退词法 407 次。

因此必须如实区分两件事：

- 显式图修复了可解释的限定名/import/dunder 问题；
- 但在现有滚动轨迹中，它并未大幅改变候选集合。本轮速度行为更可能主要来自保守 LCB 和单岛限制，而不是图结构本身。

这也是后续需要 same-history 因果实验，而不能只凭选择器计数宣称改进的原因。

## 实验阶段门

### 协议恢复审计

先在原 Fresh8 的同八题上重跑 Dense 与旧 dependency-cold，并在步数耗尽时捕获已有补丁。旧结果已经公开，所以这一轮只诊断“0/8 是否来自空提交协议”，不作为新算法独立证据。

恢复审计已经完成。Dense 与旧 dependency-cold 都是 `2/8`
resolved、`4/8` 有非空补丁，但只共同解决 `sphinx-7910`：

| 配对结果 | 任务 |
|---|---|
| 两者都解决 | `sphinx-7910` |
| 旧方法 rescue | `sympy-19954` |
| 旧方法 damage | `sphinx-8269` |
| 两者都未解决 | 其余 5 题 |

Dense 的 `sympy-13798` 在步数耗尽时捕获到 1550 字符补丁，直接证明旧协议会把已有工作误记为空；另外三条 Dense 非空轨迹来自重复执行本身，因此不能把全部恢复都归因于 patch capture。旧方法进行了 53 次物理复制、复制 52,236 个 token、0 次 fallback，但自由 agent 运行的整体中位 TTFT 是 `329.4 ms`，高于 Dense 的 `293.7 ms`。由于两臂请求数和动作路径不同，这个 TTFT 只作描述，不能当作相同 prompt 的速度比较。

结论是：旧 Fresh8 的 `0/8 Dense vs 1/8 reuse` 优势必须撤回；修复后两者净 accuracy 相同，而且存在一项 rescue 和一项 damage。新方法的判断必须依靠下述相同历史 fork 与尚未暴露的 Fresh24，不能继承旧结论。

### 同历史因果 canary

从既有轨迹中只按可见结构和冻结 LCB 选择 fork。原计划八题，但满足“图冷且 LCB>0”的任务完整集合只有六题，因此在执行前冻结为六题。执行 Dense 的第六题前处理时，安全检查发现 `astropy-14539` 的冻结历史会写入未跟踪文件 `diffbug.fits`。该任务在目标推理前被客观排除，不用已经产生的臂结果补选，因此最终可执行集合为五题。每题从相同消息和相同干净工作区分叉：

- rolling Dense；
- 旧的平面 dependency-cold；
- 新的显式图 + LCB + 单岛。

比较官方 resolved、目标 TTFT、物理 copy 和 fallback。显著性不是门槛，只要求方向优势。五题结果如下：

| 指标 | Dense | 旧平面 dependency-cold | 新图 + LCB + 单岛 |
|---|---:|---:|---:|
| 官方 resolved | 1/5 | 1/5 | 1/5 |
| resolved 任务 | `scikit-learn-14053` | 同左 | 同左 |
| fork 目标物理 copy | 0 | 5/5 | 5/5 |
| fork 目标 fallback | 0 | 0 | 0 |
| 相对 Dense 目标 TTFT 中位节省 | — | 未作为阶段门 | **33.0%** |
| 相对 Dense 配对 speedup 中位数 | 1.00× | 未作为阶段门 | **1.49×** |

新方法在五个相同 prompt 上都比 Dense 的目标 TTFT 低，复制 10,673 个目标 K/V token，且所有 K 都执行了位置旋转。不过它相对旧方法只赢 2/5 个目标，中位 TTFT 反而慢 `1.72%`；因此 canary 支持“相对 Dense 有方向性速度优势且未见 accuracy 退化”，不支持“新选择器已经比旧方法更快”。

一条 scikit-learn 前缀诊断命令没有固定决策树随机种子，导致重放观察文本不能逐字复现。该重放输出不会加入冻结消息，三臂目标 prompt 哈希仍完全相同，工作区也都干净；报告仍把它列为协议局限，而不声称六题无瑕疵因果证据。

### Fresh24

已经在任何新模型结果出现前冻结 24 个历史未暴露任务，三臂顺序固定为 Dense、旧方法、新方法。原计划难度配额 9/9/6，但剩余 `1-4 hours` 任务在 repo cap=4 下最多只能选择五题，因此执行前改为最接近的 10/9/5。

同历史 canary 通过预先冻结的方向门后，三臂均已完成。正式精度使用
SWE-bench 官方容器 resolved；三臂请求路径不同，所以自由运行 TTFT 只作描述：

| 指标 | Dense | 旧平面 dependency-cold | 新图 + LCB + 单岛 |
|---|---:|---:|---:|
| 官方 resolved | 5/24 | 5/24 | **6/24** |
| 非空补丁 | 12/24 | 16/24 | 13/24 |
| agent 请求 | 654 | 648 | 678 |
| 物理 copy | 0 | 46 | 37 |
| copied / RoPE-rotated token | 0 | 27,158 | **64,606** |
| fallback | 0 | 0 | 0 |
| 描述性中位 TTFT | 308.0 ms | 321.7 ms | 340.4 ms |

新方法相对 Dense 的逐题分解是 2 rescue、1 damage、4 both-resolved；相对旧方法
也是 2 rescue、1 damage、4 both-resolved。因此四个冻结方向门均通过，允许进入精确
prompt 测速。但是，`6/24 > 5/24` 还不能直接归因于 lossy reuse：agent 可能在首次
copy 之前分叉，甚至整题从未发生 copy。下文给出处理暴露审计。

### Fresh24 处理暴露审计：净增一题不等于 reuse 净增一题

把 trajectory 中的进程 nonce、线上 manifest、37 个 copy 目标和官方结果逐题连接后，
只有 7/24 题实际暴露于新方法的有损 K/V copy；37 次 copy 只占 678 个 agent 请求的
`5.46%`。

| 相对 Dense | 有 copy 的 7 题 | 无 copy 的 17 题 | 全部 24 题 |
|---|---:|---:|---:|
| rescue | **1** (`scikit-learn-25232`) | **1** (`sympy-12096`) | 2 |
| damage | **1** (`sphinx-10449`) | 0 | 1 |
| both resolved | 0 | 4 | 4 |
| both unresolved | 5 | 12 | 17 |

因此应采用更严格的解释：

- 新方法的全体点估计是 `6/24`，方向上高于两个 `5/24` 对照；
- 但 `sympy-12096` 全程没有 copy，其 rescue 是重复执行/代理路径方差，不能记作算法收益；
- 真正 copy-exposed 的任务相对 Dense 是 1 rescue、1 damage，**净 accuracy 为 0**；
- 相对旧方法，copy-exposed 任务是 1 rescue、0 damage，说明保守图方法在受处理子集上
  改善了旧平面选择器，但这仍是事后分层，不是随机化 accuracy 估计；
- `sphinx-10449` 在 4 次 copy 后由 Dense resolved 变为空补丁，是下一轮必须解释和修复的
  稳定性目标，不能用总分上升掩盖。

这项审计避免了此前项目中常见的错误：把不同独立 agent 运行的最终分数差异全部归因
于 KV 操作。原始连接结果逐题保存于 `ATTRIBUTION_AUDIT.json`。

### 精确 prompt 速度

对 Fresh24 中新方法实际发生物理复制的全部目标，使用完全相同的 token IDs 做 Dense/reuse AB/BA 测量：一轮 warmup、五轮正式测量。主指标是 cache-ready TTFT；source build 单独报告，不伪装成在线 prefetch。

该实验已经完成，共覆盖 37 个真实目标组、37 个单岛和 370 个正式 Dense/reuse 配对：

| 指标 | 结果 |
|---|---:|
| Dense 平均 TTFT | 569.51 ms |
| reuse 平均 TTFT | 369.17 ms |
| ratio-of-means cache-ready speedup | **1.543×** |
| 配对 TTFT saving 中位数 | **35.02%** |
| 配对胜率 | **370/370 (100%)** |
| AB 中位 saving | 35.03% |
| BA 中位 saving | 35.01% |
| 物理 copy | 444/444 expected |
| fallback | 0 |
| source 完整物化均值 | 580.14 ms |

AB 与 BA 几乎一致，且每个配对使用相同 target token IDs，因此“命中后目标计算更快”
是目前最强的因果速度证据。生命周期结论要更谨慎：若把一次完整 source 物化成本均摊
到 `N` 次 target reuse，得到：

| source 使用次数 | 含 build 后相对 Dense speedup |
|---:|---:|
| N=1 | 0.600× |
| N=4 | **1.108×** |
| N=16 | **1.405×** |

37 个 source/target 组的中位盈亏平衡点是 3 次 reuse。真实 rolling agent 的 source
来自本来就会发生的代码读取，不是额外 prefetch；上表仍用“重新执行完整 source prompt”
作为保守上界。反过来，自由 agent 的中位 TTFT 不能用来否定精确测速，因为三臂生成了
不同数量和内容的请求。当前可以声称局部 cache-ready 加速和 N=4 生命周期方向优势，
不能声称完整 agent wall-clock 已经加速。

### 外部基线

最后在共同的 Qwen2.5-Coder-3B 模型和同一冻结 coding workload 上运行 CacheBlend 与 KVCOMM 原生引擎。跨引擎主要比较各自相对 native Dense 的 accuracy delta 与 TTFT speedup；不同运行拓扑下的绝对 TTFT 不直接排名。

若新方法精度更高但速度没有同时超过两者，只报告 accuracy-speed Pareto 优势，不写成全面 SOTA。

两个原生 3B 结果已经完成并经过 artifact hash 复核：

| 原生方法 | 冻结任务 | Exact-line Dense→reuse | Accuracy delta | Cache-ready | N=4 incl. build |
|---|---:|---:|---:|---:|---:|
| CacheBlend | RepoBench-P 50 | 5→4 | -2 pp | 1.501× | 0.827× |
| KVCOMM | RepoBench-P 50 | 4→5 | +2 pp | 13.849× | 8.636× |

这张表是后续共同 adapter 的目标线，不是当前 Fresh24 的直接排名。外部结果使用
Qwen2.5-Coder-3B 的静态 next-line prompt；Fresh24 使用 Qwen3-Coder-30B rolling
tool agent 和官方任务 resolved。KVCOMM 还保留其原生三代理 + FinalRefer 拓扑，token
hash 与 SGLang/CacheBlend 不同。当前的 `1.543×` 不能据此写成“超过 CacheBlend”，
`6/24` 也不能与 `4/50` 或 `5/50` exact-line 混为同一个 accuracy 指标。

## 下一轮开发计划：先消除可归因 damage，再扩大可摊销覆盖

### A. `sphinx-10449` 四个 copy 点的同历史最小反事实

从该题的冻结轨迹在四个 copy 目标前分别分叉，只改变当前目标是否物理复制，保持消息、
工作区、采样和 prompt token hash 一致。逐点记录下一工具动作、补丁是否产生和最终官方
resolved。目标不是再比较 KV distance，而是定位“哪一个自然代码模块的 stale K/V 改变了
后续行动”。四个点全部纳入，不能只挑能恢复 Dense 的点。

晋级条件只要求方向优势，不要求强显著性：至少找出一个可重复 damage 点，并得到一个仅依赖
copy 前可见信息的保护规则；若四点都不能复现，则把 Fresh24 damage 标为 agent 方差，不为它
增加新规则。

### B. 把 coding-aware 保护从“出现依赖”细化为“依赖影响当前决策”

当前一跳图只知道后续历史是否引用 source 符号，不知道当前目标是否正在据此决定补丁。下一候选
规则只使用 prompt 中已出现的信息：当前任务路径、最近一次写入/测试失败路径、source 的
qualified symbol/import/call 图，以及当前 target 的直接调用和类型/异常名称。若 source 到当前
决策证据存在可见一跳关系，则重新计算；只有与当前行动图断开的自然代码模块才 copy。

这一步必须先在已冻结轨迹 shadow：报告它是否保护 `sphinx-10449` 的可复现 damage 点，同时保留
`scikit-learn-25232` 的 rescue copy。不能因为 KV deviation 变小就晋级。

### C. 生命周期门控改为“预计可复用次数 × 单次收益”

LCB 已保证 37/37 目标在 cache-ready 条件下更快，但 source 完整构建的中位盈亏平衡为 3 次，且
线上只覆盖 7/24 题、5.46% 请求。下一成本规则同时估计：

```text
net_visible_utility
  = expected_future_uses × lower_bound_target_saving
    - incremental_source_retention_cost
```

`expected_future_uses` 只能由当前可见的剩余 step budget、同一路径后续读取次数和 pool lease 得出；
不扫描未来、不预取。源 KV 若来自本来就执行的观察，单列增量保留成本，不能重复计入完整 source
prefill。目标是在不增加 copy-exposed damage 的前提下，提高处理请求覆盖，并在同 prompt
N=4 指标上保持 `>1×`。

### D. 完成真正可排名的共同 adapter

固定一批 rolling coding-agent request records，使 SGLang、CacheBlend 和 fixed-prompt KVCOMM
都消费完全相同的 messages、target token IDs、输出预算和 evaluator；source 都只能来自同一历史
自然观察。每个引擎保留 native Dense，分别报告 official task accuracy、copy/fallback、
cache-ready TTFT、source build、N=1/4/16。若 KVCOMM 必须保留原生三代理 prompt，则只报告其
within-native delta，不能进入相同 prompt 排名。

在该共同协议完成前，本轮结论是：**新方法获得了可靠的命中速度优势，Fresh24 总分方向为正，
但 copy-exposed accuracy 仍是 1 rescue / 1 damage，尚未证明有损复用本身提高任务精度。**

## 可复查产物

- LCB 校准：`kvflow-artifacts/impactkv_dependency_graph_lcb_20260811/CALIBRATION.json`
- shadow audit：`kvflow-artifacts/impactkv_dependency_graph_lcb_20260811/SHADOW_AUDIT.json`
- Fresh8 恢复登记：`kvflow-artifacts/impactkv_dependency_cold_fresh8_patch_capture_20260811/RECOVERY_REGISTRATION.json`
- 同历史 canary 登记：`kvflow-artifacts/impactkv_dependency_graph_same_history_canary6_20260811/CAMPAIGN_REGISTRATION.json`
- 同历史 canary 结果：`kvflow-artifacts/impactkv_dependency_graph_same_history_canary6_20260811/RESULT.json`
- 同历史 pre-treatment 排除：`kvflow-artifacts/impactkv_dependency_graph_same_history_canary6_20260811/PRETREATMENT_EXCLUSIONS.json`
- Fresh24 登记：`kvflow-artifacts/impactkv_dependency_graph_fresh24_20260811/CAMPAIGN_REGISTRATION.json`
- Fresh24 三臂结果：`kvflow-artifacts/impactkv_dependency_graph_fresh24_20260811/RESULT.json`
- exact-prompt AB/BA：`kvflow-artifacts/impactkv_dependency_graph_fresh24_20260811/exact_prompt_speed_abba/RESULT.json`
- copy-exposure 归因审计：`kvflow-artifacts/impactkv_dependency_graph_fresh24_20260811/ATTRIBUTION_AUDIT.json`

Fresh8 恢复、同历史五题 canary、Fresh24、精确速度和外部基线审计均已写入。外部 3B
数字只作为 native reference；共同 rolling adapter 仍是下一轮必须完成的比较实验。
