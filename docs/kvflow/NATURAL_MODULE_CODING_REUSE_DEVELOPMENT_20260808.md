# Coding Prompt 自然模块 KV 复用：开发与证据记录

日期：2026-08-08（UTC）  
状态：原强门槛停止记录保留；按用户随后确认的最小优势策略，已完成独立注册的物理 splice 与代码模块 SGLang 开销实验

## 1. 为什么要换实验单位

此前的 128-token island 是一个公平的等预算实验单位，不是一个合理的
coding 语义单位。它可能从函数中间开始，也可能把 assistant 的解释、工具命令
和工具输出切在同一个片段里。这样做可以回答“同样复制 128 token 时谁更好”，
却不能回答“coding prompt 自己是否提供了更好的复用边界”。

本轮把问题改成：**prompt 中自然形成的代码、搜索结果和 assistant 解释，是否
比同长度但跨越语义边界的 token 更内聚、更适合有损 K/V 复用？**

例如下面一次交互不再被视为一个固定长度窗口：

```text
assistant_interpretation
  “报错指向 parser.py 的 parse()；先确认这个函数如何处理空输入。”

tool_command
  sed -n '80,180p' src/parser.py

repository_code
  def parse(...):
      ...                         # 长度由真实输出决定
```

三部分有各自的自然长度。`repository_code` 可以是 102 token，也可以是 3,207
token；实验不会为了凑 128 token 而截尾或补齐。

## 2. 确定性模块切分

实现位于
`benchmark/multi_workflow/natural_prompt_modules.py`。切分只读取 agent 在线已经
看到的消息、命令和工具结果，不读取 gold patch、官方 evaluator、Dense Attention
或未来输出。

当前冻结的模块类型如下：

| 模块 | 通俗含义 | 例子 |
|---|---|---|
| `system_instruction` | agent 规则和历史压缩说明 | 工具使用规则 |
| `task_specification` | 用户给出的 coding 任务 | issue 描述、复现信息 |
| `assistant_interpretation` | agent 对证据的解释和下一步意图 | “失败来自旧 formatter” |
| `tool_command` | 真正序列化到 prompt 的工具调用 | `rg parse src/` |
| `repository_code` | 成功读取的代码或配置 | `sed/cat/head/tail` 输出 |
| `repository_search` | 成功的仓库搜索结果 | `rg/grep/find` 输出 |
| `test_or_execution_feedback` | 测试或程序运行反馈 | pytest traceback |
| `diff_or_mutation_feedback` | 修改或 diff 反馈 | `apply_patch`、`git diff` |
| `other_tool_result` | 不属于上述类别的工具结果 | 环境查询 |
| `generation_marker` | 下一次 assistant 生成起点 | chat-template marker |

短读取仍然属于代码模块。“至少 400 字符”是旧观察复用策略的 eligibility 条件，
不能用来定义内容本身是什么；自然模块实验仅在纳入分析时应用 32-token 下限。

每个模块记录：token 起止位置、自然长度、路径、符号、仓库 epoch、内容 hash、
来源 request、grounding 模块和后续失效事件。模块关系图记录 exact path、同目录、
共享符号、解释与其证据、失败到下一行动、interaction distance 和同一仓库 epoch。

最重要的实现不变量是：**切分前后的完整 prompt token ID 完全相同**。模块只是在
既有 token 序列上增加不重叠、连续、全覆盖的区间，不改写 prompt。

## 3. 开发集容量审计

审计使用上一轮已存在的 64 个 prompt / 16 个任务，不产生新模型结果。修正短读取
的语义分类后，所有旧 prompt 的 token ID 仍逐项相同。

| 自然模块 | prompt 中模块数 | 可复用实例（32–4096 token） | 覆盖任务 | 同 parent 跨边界等长对照 | 同类型 recency 等长对照 |
|---|---:|---:|---:|---:|---:|
| 代码读取 | 174 | 126 | 16 | 126 / 16 tasks | 42 / 14 tasks |
| assistant 解释 | 366 | 131 | 16 | 131 / 16 tasks | 46 / 13 tasks |

代码模块的自然长度中位数为 369.5 token，范围 28–3,207；assistant 解释的中位数
为 24 token，范围 9–268。后者也说明 32-token 分析门槛会排除很多非常短的连接语，
而不是强行把它们扩成 128-token island。

开发审计的八项容量门槛全部通过。原始结果位于：

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_natural_prompt_modules_20260808/development64/CAPACITY.json
```

## 4. 新任务 cohort 如何冻结

新 Attention 证据不能继续来自调过方法的旧任务。新 cohort 在新 Dense 轨迹产生前
完成冻结：

- 从本地 SWE-bench Verified-500 选择；
- 排除 69 个任何旧 trajectory 或上一轮 frozen cohort 使用过的任务；
- 初始 20 题按难度固定为 7 个 `15 min–1 hour`、7 个 `1–4 hours`、6 个
  `<15 min`；
- 初始 cohort 每个 repository 最多 2 题；
- 同时预先封存最多 29 题、每 repo 最多 3 题的容量上限；额外 9 题只能在开封
  Attention 前因容量不足而启用；
- 选择只由固定 salt、难度和 repo 配额决定，不使用模型结果。

固定的 20/29 题列表、输入 hash 和保护声明位于：

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_natural_module_attention_20260808/COHORT_REGISTRATION.json
```

两道基础设施失败来自 Matplotlib 官方镜像内部 UID 超出本机 rootless Docker 的
subuid 范围；它们发生在 agent 推理前，不能记作算法失败。其余冻结镜像已提前
补齐，最终是否需要启用预封存扩展，只由 Attention 开封前的模块容量决定。

最终得到 18 条完整 Dense trajectory 和 476 次模型请求。Dense 中位 TTFT 为
328.0 ms，p95 为 621.0 ms；ledger 中 source/target 注册、KV copy 和 prefetch
事件全部为 0。18 个任务已足够通过容量门槛，因此没有启用额外 9 题。

## 5. Attention 实验将回答什么

使用 Qwen2.5-Coder-3B-Instruct BF16，在第 0/8/17/26/35 层读取全 prompt 的
block-level Attention。Attention 仍对全局 key 做 softmax，只把结果汇总到自然
模块，不把研究退化成几十个局部 token。

### 5.1 模块内部凝聚度

对每个 32–4096 token 的代码模块或 assistant 解释，比较：

1. 整个自然模块；
2. 同一次 parent interaction 内、相同长度、但跨越自然边界的片段。

指标是 `attention_mass / key_tokens`。基线模型只知道 key/query 长度、token 距离、
绝对位置、interaction distance 和 layer；增强模型再加入模块类型和 coding 关系。
所有预测按 task leave-one-out，避免同一题泄漏。

### 5.2 来源到消费模块

关系图进一步比较：

- 代码读取 → 后续引用相同路径/符号的 assistant 解释；
- assistant 解释 → 它生成的具体工具命令；
- 与同类型、同长度、位置尽量接近的 recency 控制片段。

这比“observation 是否被下一次 tool call 读取”更完整，因为代码内容、解释、命令、
测试反馈和仓库版本都成为显式模块。

### 5.3 冻结门槛

只有以下条件全部成立才允许 physical splice：

- 代码和解释的基线校正后模块/边界比值中位数分别 ≥ 1.20；
- 两类的 paired direction 分别 ≥ 65%；
- task bootstrap 的比值下界分别 > 1；
- 加入模块/关系特征后，task-LOO Spearman 相对长度/位置基线至少提高 0.10，且
  task bootstrap 下界 > 0。

## 6. 后续门控，而不是预设成功

若 Attention 通过，physical splice 会按两类模块各冻结 32 对、至少 8 个任务，
比较整自然模块、同 parent 跨边界等长片段、同类型 recency 等长片段。固定 128
token tail 只保留为诊断，不参与主要结论。

物理干预还必须满足：自然模块相对边界的 Attention density 比值 ≥ 1.25、方向
≥65%、bootstrap 下界 >1，同时条件化局部输出扰动的中位数 ≤ 边界的 0.90，
paired win ≥60%。任何一项失败，variable-length SGLang stage-overhead 和 runtime
arm 都保持关闭。

因此，本轮不是先写一个新的 runtime policy 再寻找支持数据，而是依次回答：

```text
自然边界是否真实存在
  → 在物理 K/V splice 下是否更稳定
    → 长度/位置/多 island 的真实开销能否被可靠预测
      → 最后才允许进入同 prompt 的 accuracy + speed 实验
```

## 7. 实验结果：边界信号存在，但没有强到足以打开 splice

### 7.1 为什么出现过一次容量失败

第一版 `prepare` 在每个 prompt 内用 salted random 选择同类型模块。72 个 prompt
已经有 61 个代码模块、65 个解释模块，但随机选择只保留了 14/5 个 source→consumer
对照。这里的失败属于设计器没有利用已经冻结的结构关系，不是任务池容量不足。

在没有打开 Attention 的前提下，新注册只做了一处 outcome-free 修正：若同一
prompt 内存在多个同类型模块，优先保留具有等长 recency 对照的模块，再用 salt
打破平局。修正后的设计仍为 72 prompts / 18 tasks，代码/解释模块为 60/63；
source→consumer 对照为 42/46，均覆盖 16 tasks。所有容量门槛通过，额外 9 题保持
未启用。

### 7.2 自然边界比跨边界片段更内聚吗

![自然模块与跨边界对照](assets/natural_module_attention_20260808/01_natural_boundary_attention.png)

| 模块 | 配对 case / tasks | Raw 中位比值 | Raw 胜率 | 几何因素校正后中位比值 | task bootstrap 95% 区间 | 1.20 门槛 |
|---|---:|---:|---:|---:|---:|---:|
| 代码读取 | 60 / 17 | 1.022× | 70.0% | 1.028× | [1.009, 1.065] | 失败 |
| assistant 解释 | 63 / 18 | 1.150× | 100.0% | 1.144× | [1.126, 1.172] | 失败 |

这不是“完全没有 coding signal”。两个 bootstrap 下界都大于 1，说明自然模块相对
跨边界 token 的优势可复现。问题是强度：代码读取边界只有约 2.8% 的校正后优势；
assistant 解释约 14.4%，仍未达到预先冻结的 20%。因此不能把“自然模块”本身当作
一个足够强的有损复用安全单元。

### 7.3 coding 结构能否改善 Attention 预测

![task leave-one-out 预测](assets/natural_module_attention_20260808/02_crossfit_prediction_gain.png)

只使用 key/query 长度、token 距离、位置、interaction distance 和 layer 的
task-LOO 基线已经达到 Spearman 0.901。加入模块类型、自然边界、path、directory、
symbol 和 grounding 关系后达到 0.948，增量为 +0.047；task bootstrap 95% 区间为
[+0.018, +0.076]。

所以 coding 结构带来了显著的额外信息，但没有达到冻结的 +0.10。更重要的是，
绝对相关性很高主要因为 Attention 大量受长度、距离和位置解释；不能把 0.948
全部归因于 coding awareness。

### 7.4 最值得保留的新观察：路径/符号关系强于“整个模块”

![source 到 consumer 的 Attention](assets/natural_module_attention_20260808/03_source_consumer_attention.png)

作为不改变门槛的描述性汇总，真实 source→consumer 与同类型、同长度的 recency
对照相比：

| Source 类型 | 配对 case / tasks | Attention density 中位比值 | 配对方向 | task bootstrap 95% 区间 |
|---|---:|---:|---:|---:|
| 代码读取 → path/symbol-linked consumer | 42 / 16 | 5.159× | 97.6% | [3.554, 7.020] |
| assistant 解释 → linked consumer | 46 / 16 | 2.860× | 97.8% | [1.846, 3.834] |

这组结果不能用来绕过停止门槛。它说明的是“谁会读取谁”很 coding-specific，而不是
“source-time 的旧 K/V 可以安全替代 target-time 的 K/V”。换句话说，任务结构非常
适合做 relevance/utility 预测，但 relevance 高甚至可能意味着有损误差更容易传播，
仍需独立的物理稳定性证据。

## 8. 原强门槛下的决策

原注册的最终状态为 `STOP_BEFORE_PHYSICAL_SPLICE`：

- 代码和解释的 1.20 内聚度门槛均失败；
- enhanced predictor 的 +0.10 增量门槛失败；
- paired direction、task bootstrap 正方向和 bootstrap Spearman 正增量通过；
- 按冻结协议，不运行 physical splice；
- 因此也不运行 variable-length stage-overhead，不实现 SGLang runtime arm，不产生
  新 accuracy/speed 声明。

本轮排除了一个过宽的算法假设：**“自然模块边界本身足以定义 lossy KV reuse
island”不成立。** 同时它给出下一轮更窄、也更有 coding 特征的假设：

> 不要复制所有自然代码模块；只研究具有显式 path/symbol source→consumer 边、
> 且在 consumer 前仍版本有效的模块，并把高 relevance 与物理敏感度分开建模。

若继续这个方向，必须另建物理干预注册，不能修改这一份 Attention 的 1.20 或
+0.10 门槛后宣称原实验通过。下一节正是按这个要求保存原失败记录后建立的新阶段。

## 9. 放宽为“有可复现优势即可”后的物理检验

### 9.1 新协议没有改写旧结论

用户在看到强门槛结果后明确表示，不要求 20% 或 +0.10 这种大效应，只要存在优势
即可。因此新建了 `minimal-reliable` 物理注册，而不是修改旧 `RESULT.json`。打开
物理阶段所用的证据是：两类自然模块的 task-bootstrap Attention 比值下界均大于
1，paired direction 大于随机方向，结构特征预测增量的 bootstrap 下界大于 0。

物理结果开封前，冻结了两类各 32 对、各覆盖 16 个任务。每个候选同时比较：

1. 完整自然模块；
2. 同 parent、同长度但跨模块边界的片段；
3. 同类型、同长度、位置较近的 recency 片段；
4. 自然模块足够长时的 128-token tail，仅作诊断。

这里是真实的有损 K/V splice：从较早 source prompt 复制 K 和 V；K 按目标位置做
RoPE shift，V 直接搬运；模块以外 token 正常 Dense 计算。随后在显式 path/symbol
consumer 上比较五个层的 attention-output 相对扰动，并检查完整 suffix 后的最终
logit。数值越小，表示复制旧 K/V 对当前计算的扰动越小。

### 9.2 物理结果不是“所有自然模块都安全”，而是明显分型

![物理 splice 与 SGLang TTFT](assets/natural_module_attention_20260808/04_physical_splice_and_stage_ttft_r1.png)

| 模块 | 自然块/跨边界局部扰动中位比 | 自然块胜率 | task-bootstrap 95% 区间 | 最终 logit JS 中位数：自然 / 边界 | 结论 |
|---|---:|---:|---:|---:|---|
| 仓库代码 | **0.826×** | **65.6%** | [0.728, 1.092] | 2.70e-4 / 4.01e-4 | 有方向优势，但区间仍跨 1 |
| assistant 解释 | **1.190×** | 28.1% | [1.076, 1.272] | 3.91e-4 / 2.96e-4 | 明确更差，应重算 |

代码模块的优势不是由单题制造：以每个任务的配对中位数计，16 个任务中有 10 个
优于跨边界对照。但 bootstrap 上界仍为 1.092，因此严格的
`minimal-reliable` 统计门槛没有通过，结果文件仍是
`STOP_BEFORE_STAGE_OVERHEAD`，不能写成“已可靠证明”。

这组结果也纠正了一个关键直觉：assistant 解释的模块内 Attention 凝聚度比代码
更强（1.144× 对 1.028×），但复制后反而更不稳定。高 Attention 回答“后续是否会
读取这段信息”，属于 utility；物理扰动回答“旧 K/V 能否安全代替当前 K/V”，属于
risk。二者不能互相替代。

### 9.3 按最小工程优势继续的代码-only SGLang 检验

在用户接受方向优势后，后续只把仓库代码投影到 SGLang；assistant 解释、命令、
搜索和其他 token 全部重算。这是物理结果之后形成的探索性 policy，不是独立
confirmatory 证据。

实验发现并修复了一个 KV pool 身份问题：相同代码 token 可能出现在不同 source
prompt 中，但其 K/V 受上下文影响，不能只按代码 token hash 合并。新 pool key 同时
包含 source prompt、自然模块 ID 和 token hash。第一次旧 key 的 leased-record
冲突和第二次静态 rolling source/target 角色冲突都保留为失败 artifact；最终纯机制
复测只排除 source prompt 同时也是注册 target prompt 的 8 个静态回放歧义 case，
筛除规则不读取 TTFT 或模型输出。

最终覆盖 24 个自然代码模块、16 个任务；模块长度 38–1,584 token，中位 222。
每个 target 1 次 warmup、3 次测量，无 prefetch：

| 指标 | 结果 | 如何理解 |
|---|---:|---|
| 真实 copy / 预期 copy | 96 / 96 | 所有注册请求都走了 K+V copy |
| fallback | 0 | 没有混入 Dense fallback |
| Dense / reuse 平均 TTFT | 284.27 / 276.15 ms | 按总平均为 **1.029×** 加速 |
| 配对 TTFT saving 中位数 | **-1.72%** | 典型短模块仍略慢 |
| 配对胜率 | 41.7% | 优势不是普遍存在 |
| 1-token Dense/reuse 一致 | 100% | 仅是机制诊断，不是任务 accuracy |
| N=4 且人为计入一次 source build | 0.830× | 若 source 不是自然历史而需额外构建，则不划算 |

“平均加速但中位数变慢”并不矛盾：长 prompt、长代码岛节省的绝对毫秒更大，主导
了总体平均；大量几十到两百 token 的岛，copy 和分阶段调度开销超过省下的计算。
按模块长度做事后诊断：

| 自然代码长度 | case 数 | 平均 TTFT saving | 中位 saving | 胜率 |
|---|---:|---:|---:|---:|
| <128 | 4 | -3.81% | -5.32% | 25.0% |
| 128–255 | 11 | -4.49% | -2.46% | 18.2% |
| 256–511 | 4 | -2.51% | -2.12% | 50.0% |
| ≥512 | 5 | **+11.09%** | **+7.75%** | **100%** |

这个分桶是看到结果后的诊断，样本仅 5 个，不能把“512 token”直接写成已确认的
算法阈值。它提供的是下一次 fresh cohort 的明确可证伪假设。

### 9.4 当前算法应如何收窄

本轮证据支持的最窄设计不是“复用所有自然模块”，而是：

```text
先用 coding parser 找到完整 repository-code 模块
  → 用 path/symbol、版本 epoch 和后续 consumer 判断它是否有用且仍有效
    → assistant interpretation 与其他高状态依赖模块保持 Dense
      → 只有足够长、预计计算收益高于 copy/stage 开销的代码模块才复用 K+V
```

下一轮需在 fresh task 上预先冻结长度/成本规则，并测试完整 agent accuracy。当前
结果只能声明两点：自然代码模块存在比跨边界片段更低物理扰动的方向性机会；长
代码模块在真实 SGLang 路径上存在速度优势。它还不能声明 accuracy 已提高，也不能
声明已经超过 CacheBlend 或 KVCOMM。

完整结果位于：

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_natural_module_attention_20260808/
    initial20/dense/full_18/PIPELINE_STATUS.json
    attention_initial20_r1/CAPACITY.json
    attention_initial20_r1/ATTENTION.jsonl
    attention_initial20_r1/RESULT.json
    attention_initial20_r1/physical_splice_minimal_reliable/
      REGISTRATION.json
      OBSERVATIONS.jsonl
      RESULT.json
      stage_overhead_code_only/      # pool-key 冲突失败，保留
      stage_overhead_code_only_r1/   # rolling 角色冲突诊断，保留
      stage_overhead_code_only_r2/   # 24-case 纯机制结果
```

## 10. 从事后分桶变成线上成本策略

### 10.1 为什么不能直接写死 `512 tokens`

上节的 `≥512` 分桶来自看到 TTFT 之后的诊断，而且同样长度的代码块放进 2k 与
10k prompt，省下的 Dense attention 工作并不相同。线上实现因此没有把 512 写成
固定门槛，而是重新用 24 个 case 的配对 TTFT 中位数拟合一个只有两个量的工程模型：

```text
预计 cache-ready saving (ms)
  = 0.13169242 × (代码模块 tokens × 当前 prompt tokens / 10,000)
    - 14.66811245
```

这里乘积近似“若不复用，这个模块参加当前 prefill attention 所需的计算”；负截距
近似 source 查找、K 的 RoPE 旋转、K/V copy 与分阶段调度的固定成本。24 个观测上的
`R²=0.8750`。这仍是事后工程拟合，不是独立确认，也不是 accuracy predictor；它只
决定一次物理 copy 预计是否值得。

例如同为 300-token 代码模块：在 2,500-token prompt 中预测亏约 4.8 ms，因此
Dense；在 5,000-token prompt 中预测赚约 5.1 ms，因此允许 lossy K/V copy。这比
“所有任务一律 512”更符合任务相关模块长度和当前上下文共同决定成本的事实。

### 10.2 当前线上算法的完整执行顺序

每个真实 agent 请求按以下顺序处理，始终没有 synthetic prefetch：

1. 在最近六个完整交互中识别成功的只读工具结果；
2. 只保留单文件直接读取形成的完整 `repository_code` tool-result 模块；
3. `rg/grep/find` 搜索、assistant 解释、工具命令、测试结果、修改反馈和无法可靠
   切开的多文件结果全部 Dense；
4. 用 source group hash、observation hash、路径和 repository write 重新验证文件
   版本；同文件修改或无法定位的修改 fail closed；
5. pool 最多保留三个 source，身份包含 source prompt、交互、路径与 token hash，
   因此相同代码文本在不同 prefix 下不会误共享 K/V；
6. 在当前 target 中再次定位完整模块，按上式计算预计 saving；只有严格大于 0 才
   注册 copy case，最多选三个不重叠模块；
7. 复制旧 K 和 V，K 按 source/target 位置差做 RoPE 修正，其余 token 正常 Dense
   计算；版本有效但本次成本为负的 source 留在小 pool 中，等待更长的后续 prompt。

### 10.3 fresh9 与首个 canary

新 agent 实验没有复用 initial20 的任务结果。它在开任何新 outcome 前冻结了原
capacity ceiling 中尚未运行的九题，并固定 Dense 与上述策略使用相同
mini-SWE-agent 模板、rolling6、32 steps、temperature 0 和 SGLang 模型。

首题 `pytest-dev__pytest-6202` 的结果是一个重要的负 canary：

| 臂 | 官方 resolved | 请求数 | source materialization | target copy | fallback |
|---|---:|---:|---:|---:|---:|
| Dense | 1/1 | 11 | 0 | 0 | 0 |
| 自然代码 + 成本门控 | 1/1 | 17 | 2 | **0** | 0 |

策略观察到 240-token 和 173-token 两个版本有效代码模块；在约 3,054-token prompt
上分别预测亏 5.02 ms 与 7.71 ms，因此正确拒绝 copy。官方任务没有变差，但由于
没有 treatment，这一题不能证明 lossy reuse 的速度或精度。两条独立启动的 agent
在第二个请求已自然分叉，请求数和 prompt 长度不同，所以也不能把策略臂 287 ms
与 Dense 217 ms 的中位 TTFT直接当成 slowdown。原先登记的“canary 至少一次
copy”门槛明确记为失败；其余 frozen cohort 仅作为探索性容量与 accuracy 收集，
速度结论必须来自真实 `target_copied` 事件以及同 prompt 分析。

## 11. fresh9 完整结果：任务精度有方向优势，cache-ready 速度已明确改善

这一节把两个容易混在一起的问题分开：

- **任务做对了吗？** 用官方 SWE-bench evaluator 判定最终 patch；
- **同一个 prompt 是否更快？** 固定完全相同的 token ID，再比较 Dense prefill 与
  有损 K/V copy 的 TTFT。

不能直接拿两条自由运行 agent 轨迹的平均 TTFT 相减。一个早期命令不同，就会导致
后续读取的文件、prompt 长度和总请求数都不同；这种差异既含算法影响，也含轨迹
分叉，不能作为纯速度因果证据。

### 11.1 官方任务 accuracy

Dense 和“自然代码 + 成本门控”都使用同一个 mini-SWE-agent、system prompt、工具
协议、rolling-6 上下文、32-step 上限、temperature 0 和同一 SGLang 模型。九道题
在 treatment 前冻结，最终 patch 只交给官方容器测试：

| SWE-bench Verified 任务 | Dense | 自然代码 + 成本门控 | 配对结果 |
|---|---:|---:|---|
| `django__django-13343` | 通过 | 未通过 | damage |
| `sympy__sympy-22914` | 未通过 | 通过 | rescue |
| `sphinx-doc__sphinx-7757` | 未通过 | 未通过 | 同未通过 |
| `pytest-dev__pytest-6202` | 通过 | 通过 | 同通过 |
| `matplotlib__matplotlib-25287` | 未通过 | 未通过 | 公共容器基础设施失败 |
| `scikit-learn__scikit-learn-13142` | 未通过 | 未通过 | 同未通过 |
| `pydata__xarray-7229` | 未通过 | 未通过 | 同未通过 |
| `astropy__astropy-14309` | 未通过 | 通过 | rescue |
| `pylint-dev__pylint-6903` | 通过 | 通过 | 同通过 |
| **合计** | **3/9** | **4/9** | **2 rescue，1 damage** |

![fresh agent accuracy 与完全相同 prompt 的 TTFT](assets/natural_module_attention_20260808/05_fresh_agent_accuracy_and_exact_speed.png)

这是目前最重要的任务级新证据：在确实发生 79 个 target copy、累计复制 85,335
个 K/V token、零 fallback 的完整 agent 运行中，策略没有出现“靠少算换取总体
accuracy 下降”，反而是 4/9 对 3/9 的方向优势。

但 `n=9` 太小，且两条自由运行轨迹在第一次物理 copy 之前就可能分叉。因此这组
数据证明“方法可以在真实任务上保持可用、没有观察到总体精度退化”，还不能证明
两个 rescue 必然由 lossy reuse 带来，更不能作统计显著的 accuracy superiority
声明。图中的逐 token 一致率没有参与 accuracy 判定。

### 11.2 线上运行证明了机制发生，但不能单独证明速度

| 指标 | Dense 自由运行 | 自然代码 + 成本门控自由运行 |
|---|---:|---:|
| 请求数 | 201 | 191 |
| prompt token 中位数 / p95 | 3,208 / 5,721 | 3,322 / 6,688 |
| TTFT 中位数 / p95 | 310.6 / 666.3 ms | 289.3 / 758.9 ms |
| source materialization | 0 | 43（其中 host 3） |
| target copy / fallback | 0 / 0 | 79 / 0 |

中位 TTFT 看似下降 6.9%，p95 却上升 13.9%；两边 prompt 分布、请求数和 agent
动作都不同，所以两项都只作运行画像。策略臂内部给每个 copy 请求匹配最接近长度
的普通 Dense 请求时，中位节省 17.35%、胜率 89.3%，但它仍不是完全相同 prompt，
只能作为描述性诊断。

### 11.3 完全相同 token 的配对速度实验

为消除轨迹混杂，accuracy 开封后另行冻结了线上**全部**发生物理 copy 的目标：56
个不同 target prompt、79 个自然代码 island。每个 Dense/reuse 请求使用完全相同的
token ID、只生成 1 token、1 次 warmup 加 3 次测量；普通 Radix prefix cache 关闭。
source prompt 只为重建线上已经自然出现过的 KV snapshot，不是预取。

| 严格配对指标 | 结果 | 含义 |
|---|---:|---|
| measured pairs | 168 | 56 prompts × 3 轮 |
| 物理 copy | 316 / 316 | 79 islands × 4 次（含 warmup） |
| fallback | 0 | 所有配对都实际接受 treatment |
| cache-ready ratio-of-means speedup | **1.359×** | 相同目标 prompt 的平均 TTFT |
| 配对 TTFT saving 中位数 | **19.52%** | 典型单次目标请求的节省 |
| 配对胜率 | **89.29%** | 150 / 168 个测量对更快 |
| target-group saving 中位数 / 胜率 | **19.52% / 89.29%** | 先在每个 prompt 内聚合 |
| 1-token 输出一致率 | 94.64% | 机制诊断，**不是 accuracy** |
| N=4 + 每组重新完整 replay source | **0.926×** | 极保守、非线上自然历史口径 |

最后一行故意采用悲观口径：每个 target group 都重新执行其全部 source prompt，
即使 56 组实际只涉及 21 个唯一 source，也把 79 次 source replay 的完整推理时间
反复计入。它回答“如果为了每四次复用都额外从零构建 source，划算吗？”答案是
不划算。这不等于线上算法需要 prefetch：真实 agent 本来就执行过这些代码读取，
Dense 流程也支付了该 source 请求的正常推理；reuse 只在其自然产生后保留 snapshot。
21 个唯一 snapshot 在控制器中的实际 materialization 记录合计 72.7 ms。

因此速度结论必须完整表述为：

> 对 agent 历史中已自然产生、版本仍有效、预计收益为正的完整代码模块，有损 K/V
> copy 在完全相同目标 prompt 上达到 1.359× cache-ready 加速；若人为要求低复用
> 次数下额外重建 source，则该收益不足以覆盖完整重放成本。

### 11.4 成本门控究竟学到了什么

canary 中两个短模块被拒绝，完整 cohort 中则有 56 个目标通过门控。这说明模型没有
简单退化成“见到代码就复制”：

```text
短代码 + 短上下文
  → 节省的 attention 工作小于约 14.7 ms 固定开销
  → Dense

较长代码，或同一代码进入更长的后续上下文
  → 预计节省转正
  → 从版本有效的 pool source 复制 K/V
```

线上实际结果与拟合动机一致：早期 24-case 纯机制实验只有长模块明显获益；fresh9
中经二维成本门控留下的真实目标，在严格配对测试中有 89.3% 获益。后者是新的
out-of-sample 工程验证，但成本公式仍来自小样本，后续需要用更多任务校准误差区间，
而不是把当前系数视为跨模型常数。

### 11.5 目前仍不能宣称超过 CacheBlend 或 KVCOMM

现有 CacheBlend 与 KVCOMM 原生复现是 fixed-prompt verifier：CacheBlend 接收其原生
固定输入，KVCOMM adapter 只支持 system + user 两条消息。fresh9 则是 rolling
coding agent，包含工具调用、代码 observation、修改反馈和不断增长的历史。它们既
不是同一 prompt，也不是同一任务执行协议。

所以本节只能给出 Dense 对照，不能把旧 CacheBlend/KVCOMM 数字拼进同一排名表。
下一阶段真正公平的比较必须满足：

1. 三个后端接收同一个 MAS 序列化后的逐请求 token ID；
2. 三者运行同一批冻结 coding 任务和官方 evaluator；
3. accuracy 报最终 resolved，速度同时报 exact-prompt cache-ready TTFT、自然 source
   生命周期成本和完整 agent wall time；
4. 保留各方案原生 K/V 处理机制，但不允许各自改写 system/user/tool prompt。

在这个 adapter 完成前，任何“已超过两个 SOTA”的表述都属于越过证据。

### 11.6 当前决策与下一工程目标

当前方向应保留，但范围必须保持窄：只复用成功的单文件只读 `repository_code`，
搜索列表、assistant 解释、测试反馈、修改后的失效版本和成本为负的目标全部 Dense。
下一步优先级不是继续扩大可复制内容，而是：

1. 把 21 个自然 source 的生命周期去重和 lease 复用做成明确 telemetry，避免保守
   benchmark 中的重复 source replay 被误解为线上成本；
2. 在 fresh 任务上扩大官方 paired accuracy，报告首次 copy 前分叉率和多次重复，
   判断 4/9 对 3/9 是否稳定；
3. 为 CacheBlend、KVCOMM 增加同一 rolling MAS prompt 的 adapter，再运行公平三方
   accuracy + TTFT；
4. 只有在共同协议上至少保持 accuracy 优势后，再扩展 path/symbol consumer 选择，
   而不是重新纳入已经被物理实验否定的 assistant 解释。

本阶段完整机器可读结果位于：

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_natural_code_cost_agent_20260808/
    CAMPAIGN_REGISTRATION.json
    RESULT.json
    online/dense/full_9/OFFICIAL_RESULT.json
    online/coding_natural_code_cost/full_9/OFFICIAL_RESULT.json
    exact_prompt_speed/
      REGISTRATION.json
      REGISTRATION_AMENDMENT_01.json
      dense.json
      reuse.json
      RESULT.json
```

## 12. expanded24：方向优势在更大的全新任务集上延续，但还没有统计显著

fresh9 的 `4/9` 对 `3/9` 只能作为小样本信号。为避免根据这九题的结果挑选后续
任务，扩展实验在任何新 outcome 产生前扫描并排除了历史 artifact 中已经暴露的
151 题，从剩余 SWE-bench Verified 任务中冻结 24 题。任务来自 7 个 repository，
每个 repository 最多 4 题；难度配额为 9 道 `<15 min fix`、9 道
`15 min–1 hour`、6 道 `1–4 hours`。Matplotlib 因已知 rootless-Docker subuid
启动故障在冻结前按基础设施规则排除，而不是根据任务结果排除。

两臂继续使用完全相同的 mini-SWE-agent rolling6 prompt、工具协议、30B 模型、
32-step 上限和 temperature 0；唯一处理差异是 eligible repository-code 模块在策略
臂中可以经过成本门控后复制 K/V。24 题全部提交官方 SWE-bench evaluator：

| expanded24 官方结果 | Dense | 自然代码 + 成本门控 | 差值 |
|---|---:|---:|---:|
| resolved | 3/24（12.50%） | **5/24（20.83%）** | **+2 题 / +8.33 pp** |
| 有非空 patch | 5/24 | 6/24 | +1 题 |
| 空 patch | 19/24 | 18/24 | -1 题 |
| evaluator error | 0 | 0 | 0 |

逐题配对比单看比例更重要：

| 配对类型 | 数量 | 任务 |
|---|---:|---|
| rescue：Dense 失败、策略通过 | **3** | `django-13837`、`sympy-14711`、`sympy-17139` |
| damage：Dense 通过、策略失败 | **1** | `xarray-7233` |
| 两者都通过 | 2 | `xarray-4629`、`scikit-learn-11578` |
| 两者都失败 | 18 | 其余冻结任务 |

因此本轮观察到的净变化是 `3 - 1 = +2` 题，而不是靠不同分母得到的视觉差异。
但只有 4 个 discordant pair，exact two-sided McNemar `p=0.625`；它没有达到常用
显著性水平。正确表述是“独立扩展集中再次观察到 accuracy 方向优势”，不是“已经
证明 lossy reuse 提升 accuracy”。尤其是 18/24 两臂都失败，当前 agent/model 的
绝对通过率较低，统计功效受到明显限制。

### 12.1 fresh9 + expanded24 的透明汇总

两批任务互不重叠，且 expanded24 没有用 fresh9 outcome 选择任务。把两批作为透明
汇总，而不是伪装成单次预注册实验，可得到：

| 33 道全新任务汇总 | Dense | 自然代码 + 成本门控 | 配对变化 |
|---|---:|---:|---|
| resolved | 6/33（18.18%） | **9/33（27.27%）** | **+3 题 / +9.09 pp** |
| rescue / damage | — | — | **5 / 2** |
| 两者都通过 / 都失败 | — | — | 4 / 22 |
| McNemar exact two-sided | — | — | `p=0.453125` |

这比 fresh9 更支持“没有观察到总体 accuracy 下降”：两次独立批次都是净正向，且
33 题上不是依靠极少复用得到结果。但 5:2 的 discordant pair 仍太少，置信结论仍
应停留在方向优势；下一轮应提高可解任务比例或做重复种子，而不是只盲目增加大量
两臂都输出空 patch 的题。

### 12.2 expanded24 中确实执行了大规模有损 K/V 复用

| 策略臂物理 telemetry | 数值 |
|---|---:|
| agent 请求 | 685 |
| materialized source | 207（device 183，host 24） |
| target copy | **469** |
| copied / RoPE-rotated K tokens | **413,128 / 413,128** |
| host-source copy | 85 |
| fallback | **0** |
| prefetch | **否** |

这排除了“策略退化为 Dense 或 exact reuse，所以 accuracy 看起来安全”的解释：处理
臂在自然工具历史中真实保留 source snapshot，并在后续不同 prefix/position 的
target 中复制了超过 41 万 token；K 按位置差旋转，V 直接复制。eligible 代码内容
本身相同，但上下文和位置不同，因此仍是 lossy KV reuse，不是普通 exact-prefix
cache hit。

### 12.3 运行恢复边界与速度结论

Dense 原始进程在完成 20 条轨迹后失去父进程，SGLang 成为 orphan；日志中没有算法
或 GPU error。恢复注册在再次运行前封存了这 20 条轨迹及 hash，只补跑尚无最终
轨迹的 4 题，并且只合并官方 task outcome。这样 `3/24` accuracy 是可组合、可审计
的；中断前后的 arm wall time 和自由运行 latency 则明确不合并。

因此 expanded24 **没有新增自由运行 speedup 声明**。当前纯速度因果证据仍是第
11.3 节的完全相同 56 个 target prompt：cache-ready `1.359×`、TTFT saving 中位
`19.52%`、胜率 `89.29%`。expanded24 的 469 次 copy 证明机制覆盖，而不是替代
exact-prompt 配对速度实验。若人为为每四次 target 额外从零重建 source，先前的
悲观口径仍只有 `0.926×`；线上 source 来自 agent 本来就做过的代码读取，没有
synthetic prefetch。

### 12.4 更新后的研究判断

目前最窄、证据最完整的结论是：

> 只对版本有效的单文件自然代码 observation 做二维成本门控的 lossy K/V copy，
> 在相同 prompt 上具有明确 cache-ready TTFT 优势；在两批共 33 个全新官方 coding
> 任务上观察到 9/33 对 6/33 的 accuracy 方向优势，但配对统计尚不显著。

这支持继续开发该方向，同时否定两种过度表述：不能说 accuracy rescue 由 KV copy
因果产生，也不能把尚未接入同一 rolling prompt 的 CacheBlend/KVCOMM 旧结果放进
排名。下一工程目标应是同 prompt 三后端 adapter，以及提高任务可解率后的重复
accuracy；不是重新扩大到已被实验否定的搜索、assistant 解释或 stale code。

expanded24 的注册、恢复边界与机器可读汇总位于：

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_natural_code_cost_agent_expanded24_20260808/
    CAMPAIGN_REGISTRATION.json
    RECOVERY_REGISTRATION.json
    DENSE_RECOVERED_OFFICIAL_RESULT.json
    RESULT.json
    online/coding_natural_code_cost/full_24/
      RUNTIME_SUMMARY.json
      OFFICIAL_RESULT.json
```

## 13. discordant7 反序复跑：rescue 大多复现，原 damage 没有复现

fresh33 的 5 rescue / 2 damage 仍可能来自单次 agent 波动。为检查这一点，下一轮没有
挑选策略成功题，而是把**全部七个成对分歧题**一起冻结复跑，包括两个原 damage；
执行顺序从此前的 Dense→策略反转为策略→Dense，以排除“策略总是后跑”的简单顺序
解释。模型、rolling6 prompt、32 steps、temperature 0 和官方 evaluator 均不变。

这是一项明确的**事后稳定性审计**：任务正因为 fresh33 中发生过 flip 才入选，所以
重复得到的 `4/7` 对 `0/7` 不能追加到 fresh33，不能作为新的独立总体 accuracy 样本，
也不能用其 McNemar p-value 做 population-level superiority 检验。

| 任务 | fresh33 原标签 | 反序复跑策略 | 反序复跑 Dense | 新标签 |
|---|---|---:|---:|---|
| `astropy__astropy-14309` | rescue | 通过 | 未通过 | **rescue** |
| `django__django-13343` | damage | 未通过 | 未通过 | 共同失败 |
| `django__django-13837` | rescue | 通过 | 未通过 | **rescue** |
| `pydata__xarray-7233` | damage | 未通过 | 未通过 | 共同失败 |
| `sympy__sympy-14711` | rescue | 通过 | 未通过 | **rescue** |
| `sympy__sympy-17139` | rescue | 未通过 | 未通过 | 共同失败 |
| `sympy__sympy-22914` | rescue | 通过 | 未通过 | **rescue** |
| **合计** | 5 rescue / 2 damage | **4/7** | **0/7** | 4 rescue / 0 damage |

转移矩阵更直接：

| 原标签 → 复跑标签 | 数量 |
|---|---:|
| rescue → rescue | **4** |
| rescue → 共同失败 | 1 |
| damage → damage | **0** |
| damage → 共同失败 | 2 |

因此 5 个原 rescue 有 4 个在反转 arm 顺序后仍满足“策略通过、Dense 未通过”，稳定
率为 80%；两个原 damage 均没有复现。复跑本身为 4 rescue / 0 damage，描述性的 exact
McNemar `p=0.125`，但因为 cohort 是按原 discordance 选择的，这个 p-value 不能用于
总体推断。更可靠的用途是修正我们对单题 flip 的解释：原 damage 不是稳定算法伤害，
而至少 4 个 rescue 不是简单的 arm 执行顺序假象。

策略组在这七题中有 160 个请求、47 次 source materialization、106 次 target copy，
复制并旋转 100,221 个 K token，16 次从 host source 复制、0 fallback、0 prefetch。
因此反序结果仍发生了大量物理 lossy treatment。Dense 有 186 个请求，但仅 1 个非空
patch 且官方未通过；策略有 5 个非空 patch，4 个官方通过。这也暴露了 agent outcome
自身的高波动：即使 temperature 0，工具动作和最终是否产出 patch 仍不具备逐次完全
可重复性。

本轮没有增加速度声明。自由运行中策略中位 TTFT 280.2 ms、Dense 268.2 ms，轨迹与
prompt 分布不同，不能据此称加速或减速；纯速度仍引用完全相同 prompt 的 1.359×
cache-ready 实验。

### 13.1 首次 treatment 前审计改变了 rescue 的解释

反序复跑的 `4/7` 对 `0/7` 仍只是两个独立运行的 agent。为了判断四个稳定 rescue
能否归因于物理 KV copy，我们逐题定位策略账本中的首次 `target_copied`，并把此前
每轮 assistant 内容、工具参数和工具返回与 Dense 轨迹对齐。仅忽略每个进程生成的
tool-call ID；system/user 初始 prompt 必须完全相同。

| 稳定 rescue | physical copy | 首次 copy 请求 | copy 前完整历史相同？ | 因果判定 |
|---|---:|---:|---:|---|
| `astropy-14309` | **0** | — | — | 未接受 treatment |
| `django-13837` | **0** | — | — | 未接受 treatment |
| `sympy-14711` | 6 / 5,982 tokens | 6 | 否；第 1 请求已分叉 | treatment 前混杂 |
| `sympy-22914` | 13 / 18,185 tokens | 5 | 否；第 1 请求已分叉 | treatment 前混杂 |

七题总体有 5 题发生物理 copy，但 `0/5` 在首次 copy 前保持完整 interaction history
一致，甚至 `0/5` 保持全部工具动作一致。四个稳定 rescue 中两个没有 treatment，
另外两个在 treatment 前已经形成不同 prompt。因此：

> 反序复跑支持“策略配置下的 outcome 方向具有一定稳定性”，但其中 **0 个 rescue**
> 满足把最终 accuracy 改善因果归于 lossy KV copy 的条件。

这不是语义上的小修正。未发生 copy 的 rescue 直接测量了独立 agent 运行波动；发生
copy 但历史已不同的 rescue 同时混入早期生成差异。它解释了为什么 NLL、KV
deviation、请求数和单次 final accuracy 都不能互相替代，也意味着后续不能再用这些
rescue 反向训练所谓“低风险 module selector”。

下一项有效实验必须采用 forked continuation：先冻结同一条自然历史、workspace 与
完全相同 target token IDs，在该点分叉 Dense 和 reuse，仅让物理 KV treatment 不同，
随后各自继续 agent 并由官方 evaluator 判最终 patch。只有这种设计才能同时回答
“相同 prompt 是否更快”和“首次 lossy perturbation 是否改变最终任务成功”。

机器可读注册与结果：

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_natural_code_cost_discordant7_repeat_20260809/
    CAMPAIGN_REGISTRATION.json
    RESULT.json
    TREATMENT_ATTRIBUTION.json
    online/coding_natural_code_cost/full_7/OFFICIAL_RESULT.json
    online/dense/full_7/OFFICIAL_RESULT.json
```

## 14. 同历史分叉：首次把一个 accuracy rescue 归因到 KV treatment

第 13.1 节的问题是：Dense 与复用 agent 都从 request 1 独立生成，即使 temperature
为 0，也可能在第一次 KV copy 之前选择不同命令。解决方法不是再跑一次相同双臂，
而是把实验单位改成“同一个已经发生的 agent 状态”：

1. 从自然策略轨迹中取首次 eligible copy 前的完整 system、任务、assistant、tool
   历史；
2. 在两个全新 SWE-bench 官方容器中逐条重放此前只读命令，要求每条 tool observation
   与原轨迹逐字一致，并要求 `git status`、`git diff` 都为空；
3. Dense 与自然代码复用使用完全相同的 target token IDs，只在该请求是否复制 K/V
   上分叉；
4. 从不同的 target response 开始，各自继续运行 agent，最后用官方测试判 patch。

复用臂在分叉前向 SGLang 重放冻结 prompt、只生成并丢弃一个 token，以重建此前自然
source 的 KV。这只是因果实验的状态搭台，不写入 agent history，也不计入线上延迟；
线上算法仍从真实代码读取请求留下 source，不启用 prefetch。

### 14.1 身份与物理执行门禁全部通过

两题选自“反序后仍稳定、且确实接受 treatment”的 rescue，因此本轮是 outcome-selected
机制 canary，不是总体 accuracy 抽样。分叉前的控制条件为：

| 控制条件 | 结果 |
|---|---:|
| Dense/策略冻结消息 hash 相同 | 2/2 |
| target prompt token hash 相同 | 2/2 |
| 重放 tool observation 逐字相同 | 全部相同 |
| 两臂分叉前工作区干净 | 4/4 容器 |
| 分叉请求实际 `target_copied` | 2/2 |
| 分叉请求 fallback | **0** |

这排除了第 13.1 节最主要的混杂：结果差异不再可能来自第一次 copy 前已经不同的工具
路径、文件内容或 prompt。

### 14.2 官方 accuracy 与同 prompt TTFT

| 官方任务 | 分叉请求 | 实际复制 K/V | Dense TTFT | 复用 TTFT | TTFT 节省 | Dense | 复用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sympy-14711` | q6 | 997 tokens | 271.5 ms | 265.4 ms | 2.22% | 未通过 | **通过** |
| `sympy-22914` | q5 | 2,840 tokens | 466.8 ms | 200.0 ms | 57.16% | 通过 | **通过** |
| **合计/中位** | — | **3,837 tokens** | — | — | **29.69%** | **1/2** | **2/2** |

两个分叉 target 的复用 TTFT 都更低，paired speedup 中位为 `1.678×`。但 `n=2`，且
题目按既有 rescue 选择，不能用这一速度点替代 56 个 outcome-independent exact-prompt
target 的 `1.359×` 主速度结果。

更重要的是，官方判题第一次出现了 causally clean rescue：`sympy-14711` 的 Dense
在相同 q6 prompt 上继续搜索 `other == 0`，之后错误地破坏了 `__and__` 定义并在 32
次调用上限停止；复用臂在 q6 转向检查 `_check_vector`，最终只在 `Vector.__add__`
加入 `other == 0` 时返回自身，官方测试通过。`sympy-22914` 虽然 q5 的下一条工具命令
也发生变化，但两臂最终提交同一份 `_print_Min/_print_Max` patch，均通过。这给出一正
一安全的局部观测：有损 KV 可以改变搜索路径，但并不必然改变最终正确 patch。

从分叉到结束，策略共执行 13 次 physical copy、复制并旋转 16,605 个 K token、
0 fallback；所以 `2/2` 不是第一次 copy 后悄悄退化为 Dense。

### 14.3 能证明什么，仍不能证明什么

本轮把此前“0 个可归因 rescue”推进为：

> 在两个冻结的稳定-rescue continuation 中，自然代码 lossy KV treatment 相对同状态
> Dense 产生 1 个官方 rescue、0 damage，并在两个完全相同 target prompt 上都降低
> TTFT。

这是目前支持算法合理性的最直接证据，但不能宣称 population accuracy 从 50% 提升
到 100%：两题是看过既有 outcome 后选择的，而且都来自 SymPy。下一轮应在不知道
Dense/策略最终结果的前提下，先收集一批自然出现且满足物理 copy 的 fork points，
冻结后统一跑官方 continuation；同时纳入历史 damage、共同成功与共同失败来源，而
不是继续只复跑 rescue。

机器可读证据：

```text
/home/gfy/CodeMAS_Project/kvflow-artifacts/
  impactkv_natural_code_cost_same_history_fork_20260809/
    CAMPAIGN_REGISTRATION.json
    RESULT.json
    dense/OFFICIAL_RESULT.json
    coding_natural_code_cost/OFFICIAL_RESULT.json
    coding_natural_code_cost/SERVER_LEDGER.jsonl
```
