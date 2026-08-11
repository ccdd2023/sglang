# Coding-aware KV reuse：高依赖代码重算，低依赖代码有损复用

日期：2026-08-10  
状态：方向性实验完成；可进入更强 accuracy backend 的扩大验证  
范围：只修改 coding-aware SGLang worktree 与新实验 artifact；未修改 paper、prefetch 分支或旧预注册门槛

## 1. 这轮解决了什么问题

此前容易把“后续 decoding 对某段代码 Attention 较高”理解为“这段代码值得复用”。
这个方向是反的：Attention 高说明后续答案正在依赖该段信息，因此旧上下文中算出的
KV 一旦有偏差，影响更容易被放大。这样的代码应该重新计算。真正适合有损复用的是：

> 代码内容仍然有效，但在当前可见的后续推理中没有形成直接路径或符号依赖的模块。

本轮把这个判断落成了可在线执行的策略：

- **dependency-hot：重算**；
- **dependency-cold：通过收益门后才有损复制 KV**；
- 其余 system prompt、用户问题、assistant 推理、工具命令、搜索结果、测试反馈、修改反馈继续 Dense。

它不是 exact reuse，也没有 prefetch。实际复制的是旧上下文生成的 K/V；K 根据新位置做
RoPE 旋转，V 直接复制。由于该代码块的隐状态是在旧前缀下形成的，即使代码文本相同，
这仍是 lossy KV reuse。

## 2. 为什么“相关就重算”在理论上更合理

对某个 decoding query，Attention 输出可写为：

\[
o = \sum_i a_i v_i.
\]

有损复制改变 K/V 后，一阶误差可以直观写成：

\[
\Delta o \approx \sum_i a_i\Delta v_i + \sum_i\Delta a_i v_i.
\]

这里有两条风险通路：

1. 原本 Attention 权重 \(a_i\) 高时，同样大小的 \(\Delta v_i\) 会被更强地传到输出；
2. K 的变化会改变 Attention 权重本身，即 \(\Delta a_i\)，从而改变模型读取信息的方式。

所以 Attention/路径相关性在这里首先是 **risk signal**，不是“复用收益”。收益仍由 token
长度与目标 prompt 长度决定；风险和收益必须分开。

这也解释了为什么 KV deviation 不能单独当最终优化目标：它只描述内部状态距离，没有
包含任务是否依赖这段状态，也没有等价地约束最终 patch 是否通过测试。

## 3. 在线算法如何工作

### 3.1 先把 agent prompt 按自然交互分组

一个简化的 coding agent 历史如下：

```text
[system] 角色、工具协议、提交格式
[user]   issue：修复 FileField 对 callable storage 的序列化

[assistant/tool-call] 读取 tests/field_deconstruction/tests.py
[tool/observation]     单文件测试代码 A

[assistant/tool-call] 读取 django/db/models/fields/files.py
[tool/observation]     单文件实现代码 B

[assistant]            讨论 FileField.deconstruct
[assistant/tool-call] 再次 grep files.py 的 deconstruct
[tool/observation]     实现代码 C

[current request]      决定下一步修改
```

算法不再用固定 token 长度假定一个 island，而是把成功的单文件直接读取结果视为一个自然
repository-code 模块。搜索列表、多文件混合输出、测试执行结果和写操作反馈都不进入 lossy
候选池。

### 3.2 dependency-hot 判定只使用当前已经可见的信息

对候选代码观察 \(G_s\)，检查它之后、当前请求之前的可见分组：

- 后续是否再次明确提到同一路径；
- 后续是否明确提到该源码中抽取出的符号；
- 源码之后是否出现覆盖同文件/符号的修改，使版本失效。

满足前两项之一就是 dependency-hot；第三项直接使旧源失效。该判定不读取未来 Attention、
未来模型输出或官方判题结果，因此可以在线执行。

在上面的 Django 例子中：

- `django/db/models/fields/files.py` 被后续多次读取并讨论 `FileField.deconstruct`，属于 hot，
  必须 Dense 重算；
- 较早的 `tests/field_deconstruction/tests.py` 在当前决策链中没有再次被路径或符号指向，
  属于 cold，可以进入收益门。

真实的同历史 q14 实验正是这样处理：复制 383 个测试文件 token，同时保护 3 个与
`files.py` 直接相关的代码观察。

### 3.3 cold 还不等于一定复用

候选还必须同时满足：

1. 直接、成功、只读的单文件源码观察；
2. 源码版本仍有效；
3. target 中能唯一定位原分组；
4. 预测 cache-ready 收益为正：

\[
0.13169242 \times \frac{L_{island}L_{prompt}}{10000} - 14.66811245 > 0.
\]

当前最多保留 3 个 live sources、一个 target 最多复制 3 个 islands。没有通过上述门的
内容全部 Dense 计算。

## 4. 第一层证据：同任务、同复制预算的 Hot/Cold 方向实验

为了回答“相关代码到底应复用还是重算”，实验对每一臂都只复制最后 128 tokens，并在
同一 coding task 内配对 Hot 与 Cold 模块。这样不会把 token 数量优势误当成语义优势。

| 指标 | 复制 Cold 代码 | 复制 Hot 代码 | 解释 |
|---|---:|---:|---|
| 64-token Dense continuation 完全一致 | 7/8（87.5%） | 3/8（37.5%） | Cold 明显更稳定 |
| normalized token edit median | 0.0000 | 0.2578 | Cold 输出偏移更小 |
| 有信息配对胜负 | 5 胜 | 1 胜 | 另 2 对平局 |
| Cold−Hot edit bootstrap 95% 区间 | `[-0.7500, -0.2578, 0.0000]` | — | 中位方向支持 Cold |

这组结果只验证局部行为方向，不是官方 accuracy。但它足以否定“Attention/依赖越高越应该
复用”的原假设，并支持 **Hot 重算、Cold 复用**。

Artifact：
`kvflow-artifacts/impactkv_hot_cold_recompute_direction_20260810/same_task_fixed128/RESULT.json`

## 5. 第二层证据：相同历史、相同目标 prompt 的因果 TTFT

三题 canary 固定了相同消息、相同目标 prompt hash 与干净只读 workspace，只改变目标请求
是否物理复制 dependency-cold KV。

| 任务 | Dense TTFT | Cold TTFT | TTFT 节省 |
|---|---:|---:|---:|
| SymPy-17630 | 653.5 ms | 357.3 ms | 45.3% |
| pytest-10356 | 520.5 ms | 302.8 ms | 41.8% |
| Astropy-14182 | 326.0 ms | 260.4 ms | 20.1% |
| **中位数** | — | — | **41.8%，1.719×** |

三题全部发生物理 copy，K 全部 RoPE 旋转，0 fallback。由于三题的两臂均没有形成可配对的
官方非空 patch，这里只能保留因果速度结论，不能声称 accuracy preservation。

Artifact：
`kvflow-artifacts/impactkv_dependency_cold_same_history_canary_20260810/initial3_r1/RESULT.json`

## 6. 第三层证据：有官方判题分辨率的同历史机制样本

这两题是从历史 Dense-resolved 轨迹中按结果挑选出来的，作用是验证机制，不可当总体
accuracy 估计。

| 同历史 fork | Cold copy | Dense official | Cold official | TTFT 节省 | 结论 |
|---|---:|---:|---:|---:|---|
| SymPy-22914 q9 | 620 tokens | resolved | resolved | 18.4% | 机制通过：正确性保持且更快 |
| Django-13343 q14 | 383 tokens | unresolved | resolved | 7.9% | Cold rescue，但没有复现历史 Dense resolved，按预注册机制门记失败 |

Django 的失败门记录非常重要：即使 Cold 这次通过，也不能把它改写成“稳定提升 accuracy”。
它说明 agent 轨迹对生成路径敏感，单次 discordant outcome 需要重复或扩大样本。

Artifacts：

- `kvflow-artifacts/impactkv_dependency_cold_resolved_mechanism_20260810/sympy22914_q9/RESULT.json`
- `kvflow-artifacts/impactkv_dependency_cold_resolved_mechanism_20260810/django13343_q14/RESULT.json`

## 7. 第四层证据：任务不重叠的 fresh8 完整 agent 实验

在任何本轮模型输出之前，从 SWE-bench Verified 500 中排除历史 artifact 暴露过的 175 题，
再以固定 salt、难度 3/3/2、每仓库最多 2 题选出 8 题。Dense 与 Cold 使用同一 mini-SWE-agent
rolling6、同一 prompt 模板、temperature 0、32 步上限和官方判题。

### 7.1 最终任务结果

| 指标 | Dense | Dependency-cold reuse |
|---|---:|---:|
| 官方 resolved | 0/8 | 1/8 |
| 非空、完成判题的 patch | 0/8 | 4/8 |
| 空 patch | 8/8 | 4/8 |
| 自由运行 median TTFT | 355.1 ms | 306.3 ms |

Cold 唯一 resolved 为 `sphinx-doc__sphinx-7910`，因此冻结的方向门得到 1 个 rescue、0 damage，
accuracy 点估计为 +12.5 pp。自由运行 median TTFT 描述性降低 13.75%。

但是这里必须同时写出两个限制：

1. Dense 8 题全部在 32 步达到上限而没有显式提交，因而没有 Dense-completed 子集可用于估计
   “Dense 正确答案是否被保持”；
2. 两臂产生 256 与 210 个不同请求，prompt 轨迹已经分叉，所以 13.75% 不是配对因果 speedup。

因此正确结论是 **方向有利但协议分辨率有限**，不是“fresh8 已证明 accuracy 非劣”。

### 7.2 物理复用与选择器是否真的工作

| 审计项 | 数值 |
|---|---:|
| Cold arm 请求数 | 210 |
| 注册 target 的请求 | 44 |
| 原始可复用候选观察 | 432 |
| 判为 dependency-hot 并保护 | 284 |
| dependency-cold 决策 | 148 |
| target copy events | 53 |
| copied / RoPE-rotated K tokens | 49,470 / 49,470 |
| fallback | 0 |

这排除了“结果来自 exact prefix reuse”“结果来自 prefetch”或“策略实际上从不复制”的解释。

Artifacts：

- `kvflow-artifacts/impactkv_dependency_cold_fresh8_20260810/CAMPAIGN_REGISTRATION.json`
- `kvflow-artifacts/impactkv_dependency_cold_fresh8_20260810/RESULT.json`
- `kvflow-artifacts/impactkv_dependency_cold_fresh8_20260810/POSTHOC_AUDIT.json`

## 8. 目前可以与不可以声称什么

### 可以声称

1. 在相同 128-token 预算下，复制 dependency-cold 模块比复制 dependency-hot 模块更稳定；
2. 因而当前正确方向是 Hot 重算、Cold 有损复用，而不是相关模块优先复用；
3. 同 prompt 的 cache-ready fork 上，Cold 物理 KV copy 观察到 7.9%–45.3% TTFT 节省；
4. 至少一个官方 resolved 的同历史任务保持正确且快 18.4%；
5. fresh8 上没有观察到 resolved damage，并出现 1 个 rescue，但这只是有利方向。

### 还不能声称

1. 不能声称已经统计证明 accuracy 非劣或优于 Dense；
2. 不能把自由运行 13.75% 直接当作因果 speedup；
3. 不能声称已超过 CacheBlend 或 KVCOMM；本轮没有在它们的原生等价协议上运行新对比；
4. 不能把局部 continuation 一致率、NLL、KV deviation 当官方任务 accuracy；
5. 不能声称 lexical path/symbol guard 已捕获所有跨文件、动态分派或隐式语义依赖。

## 9. 为什么我认为当前“方向修正”已经达成

本轮不是只得到一张 Attention 图，而是形成了四层相互约束的证据：

1. **等预算方向实验**回答“Hot 还是 Cold 更安全”；
2. **同 prompt fork**回答“物理有损复制能否真的降低 TTFT”；
3. **official mechanism case**回答“速度收益能否与最终 resolved 共存”；
4. **task-disjoint full-agent run**检查策略是否在新任务中真实触发，并给出最终 accuracy 的有利但
   有限信号。

因此“高依赖重算、低依赖复用”已经达到继续开发所需的方向门。尚未达到的是论文级总体
accuracy 与外部 SOTA 排名，这两者需要下一轮先修复 accuracy backend 的空 patch 分辨率。

## 10. 下一轮唯一优先事项

fresh8 的主要瓶颈已不是 selector，而是 agent 在达到步数上限时丢弃工作区 diff。下一轮应：

1. 在不改变 prompt、模型和工具历史的情况下，冻结“达到上限也抓取当前 `git diff`”的提交协议；
2. 先用本轮 8 题做协议恢复审计，确认 Dense-completed 数显著上升；
3. 再注册完全未见过的新 cohort，报告 paired official resolved、rescue/damage 与 McNemar 区间；
4. 速度仍只用同 prompt fork 作因果结论；
5. accuracy backend 有分辨率之后，再与 CacheBlend/KVCOMM 做等 prompt、等 agent、各自原生引擎的
   accuracy/TTFT 比较。

## 11. 验证

- policy / bridge / same-history：73 tests passed；
- Hot/Cold 配对与固定预算：2 tests passed；
- fresh8：8/8 两臂均完成运行与官方 evaluator，0 evaluator error；
- 所有 Cold 实验：physical copy、K RoPE rotation、0 prefetch、0 exact-only gate。
