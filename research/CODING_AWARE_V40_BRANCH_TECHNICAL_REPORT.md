# Coding-aware V40 分支技术审查报告：方法定位、先例映射、代码阻塞项、证据强度、可行性与集成计划

> 报告类型：正式技术审查 + 可行性评估 + 可执行实验计划
> 报告日期：2026-07-29T02:09:59-07:00
> 审查对象：合作者分支 `review/coding-aware-v40-prefetch-20260729`
> 审查方：本项目（`code-agent-kvcache` docs 仓库 + `kvcache-research` 实现工作区）
> 最终结论：**`NOT APPROVED AS-IS`**（存在已复现的 P0 正确性阻塞项）

---

## 0. 引用约定、证据分级与本报告的自我约束

### 0.1 引用前缀

| 前缀 | 含义 | 物理位置 |
| --- | --- | --- |
| `v40:` | 被审查的合作者分支 worktree | `/home/chris/Workspaces/kvcache-research/worktrees/coding-aware-v40-prefetch` |
| `xs:` | 本项目当前底座 worktree | `/home/chris/Workspaces/kvcache-research/worktrees/cross-store-substrate` |
| `docs:` | 本文档仓库 | `/home/chris/Workspaces/code-agent-kvcache` |
| `ext:` | 作者机器上的仓库外 artifact | `/home/gfy/CodeMAS_Project/kvflow-artifacts/**`（当前环境不存在） |

### 0.2 固定 pin

| 项 | 值 | 验证方式 |
| --- | --- | --- |
| V40 review ref | `origin/review/coding-aware-v40-prefetch-20260729` | review worktree 为 detached HEAD；验证 `git rev-parse HEAD == git rev-parse origin/review/coding-aware-v40-prefetch-20260729 == 13671eb708da…` |
| V40 HEAD | `13671eb708da689137a654946b0d34ba924efb29` | `git log --oneline -1` = `13671eb70 docs(kvflow): hand off V40 for collaborator review` |
| `origin/main` | `bd47ec97ff7a2881f9bb0316a4a657000a50c020` | `git ls-remote origin refs/heads/main`（2026-07-29T05:26:28-07:00 复核） |
| merge base（`origin/main` × review） | `3343a79466aa714d34a14d08d3929f7953a47212` | `git merge-base origin/main 13671eb708da` → 一致 |
| `origin/research/cross-store-substrate` | `0206f17b4255e4b248dafaaeb943be57428dae2f` | `git ls-remote origin` |
| kvflow shared-core | `c16bfbb8e8cc83a8b23858808f52833be9091101` | `git cat-file -t` = `commit`；且是 V40 HEAD 的祖先 |
| 底座 publication HEAD | `0206f17b4255e4b248dafaaeb943be57428dae2f` | `results: bind the complete Phase7 publication record`（2026-07-28 20:37:05 -0700） |
| 底座 primary pin | `81405f4278b034911bc613c4ee17c79d15ee8f35` | `fix: include nested Phase7 artifacts in provenance`（2026-07-28 08:58:01 -0700） |
| 固定 Docker image | `ghcr.io/ccdd2023/sglang@sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781` | `docker images --digests` 本机存在 |

### 0.3 证据分级（全文强制使用）

| 级别 | 定义 | 允许的表述 |
| --- | --- | --- |
| `verified-local` | 本次审查在本机（含固定 Docker image）实际执行并观察到输出 | 可直接陈述为事实 |
| `verified-code` | 直接从被审查源码读取，附文件与行号 | 可直接陈述为事实 |
| `external unverified claim` | 数值/结论只存在于作者机器 `ext:` 或分支文档文字中，本环境**无法**复核 | 必须逐条标注，**禁止**写成"已验证" |
| `derived` | 由上述证据推导得到的判断 | 必须显式标注为判断 |

### 0.4 本报告明确不做的事

- 不把任何 `external unverified claim` 升级为已验证结果。
- 不把 V40 描述为 prefetch、KVCOMM 重建、CacheBlend selective repair 或 Cache-Craft。
- 不对 V40 与 KVCOMM/CacheBlend 的历史 225-task 基线做排名。
- 不修改被审查分支，不修改本仓库其它文件，不提交、不推送。

---

## 1. Executive Verdict 与可行性总表

### 1.1 一句话方法定义（最准确表述）

> **V40 从真实的近期 agent 请求中，选择一个"成功的、只读的、token 完全相同的、在目标 prompt 中唯一出现的、严格位于中部的、且未被后续同路径写操作失效的" tool observation 岛；对该岛的 V 原样拷贝，对 K 施加 source→target 的 RoPE 位置 delta 旋转；岛之外全部 dense 重算。它没有 context repair，没有 synthetic replay，也不是 prefetch。**

### 1.2 最终 verdict

| 维度 | 结论 |
| --- | --- |
| 方法定位 | 恢复 primitive = `R0 Raw+RoPE`（既有）；新增点主要是 `grounded coding selector` 的 admission / selection / invalidation policy。**不是新的恢复公式，不是 KVCOMM reconstruction / selected-token repair / prefetch** |
| 代码正确性 | **`NOT APPROVED AS-IS`** — 2 类已在固定 Docker image 内复现的 P0 失效漏检（违反 freshness/abstention policy；**非 data corruption**，token identity 仍通过，见 §5.1.5） |
| 文档与代码一致性 | 不一致（review request 列出的 active entry point 有一个没有生产调用者） |
| 现有实验证据 | 全部关键数值为 `external unverified claim`；样本量与设计均不足以支持优越性主张 |
| 是否可整分支 merge | **否**（相对 shared-core 191 文件 / 52,549 insertions，含大量 V8–V44 历史驱动） |
| 是否值得在本项目底座上重实现最小 payload | **是**，作为 candidate `C40` 进入受控评测 |
| 底座集成缺口 | copy/RoPE backend 与现有 cross-store lifecycle primitives 可复用；store metadata需补`C40` provenance/approx-depth映射；**request execution seam 必须新写** middle-span staging controller（§10.3.1） |
| 授权状态 | **`PENDING USER AUTHORIZATION`** —— 含 Gate 0 在内**没有任何 Gate 已被授权**；此前的 `≤62 starts / ≤28.2 GPUh` 上界已撤回 |
| 推荐的下一步（待授权） | **Track A**：zero-GPU 的代码 + provenance 修复（Gate 0/1 + middle-span controller）。未完成前不申请任何 GPU 预算 |

### 1.3 可行性总表（feasibility table）

| # | 子问题 | 可行性 | 难度 | 主要阻塞 | 证据 |
| --- | --- | --- | --- | --- | --- |
| F1 | 在本项目底座上复现 V40 的 selector 语义（结构化版本） | 高 | 2/5 | 需要 tool wrapper 提供结构化 read/write paths | §5.6、§10.3 |
| F2 | 在本项目底座上执行 C40 的数据面（copy V + K RoPE delta） | backend 已具备 | 1/5 | backend 零改动；但 **request execution seam 需新写 middle-span controller**（现有 `runtime.py` 只支持连续 span） | §10.2、§10.3.1 |
| F3 | 证明 V40 在 same-context 下 bit/数值正确 | 高 | 2/5 | 需要 GPU canary | §9.4 |
| F4 | 证明 V40 在 cross-context 下不损伤输出质量 | 中 | 4/5 | 需要 logit/KL + task-level 评测，样本量要求高 | §9.6 |
| F5 | 证明 C40 有 TTFT 收益 | **未知，先验不利** | 5/5 | Phase 7 已判定**同一恢复 primitive** R0 在同 image/同模型/chunk4096 下为 `NEGATIVE`；C40 只加 selector，**不得默认转正**（也不得预先断言必负），须由 pilot 检验；报告必须同时给出 `r` / `w` / `C_selector` / `E_cond` / `E_work` 并附 span-matched R0 对照 | §6.6、§8.3.1、§9.10b、§9.19 |
| F6 | 与 scheduler（S0/S4）组合评测 | 中 | 3/5 | 需 seed-matched 相邻 launch block | §9.9 |
| F7 | 与 prefetch（P0/P1）组合评测 | 低（当前不可复现） | 4/5 | prefetch 分支引用在当前 origin 不可获得 | §7.5 |
| F8 | 作为论文级贡献独立成立 | 低 | 5/5 | 恢复 primitive 与 Prompt-Cache 族非 prefix 复用均有先例 | §4 |

---

## 2. 分支来源、范围与分叉度

### 2.1 分叉度（`verified-local`，全部以**显式 remote ref** 度量）

**测量时点**：2026-07-29T05:26:28-07:00。远端 tip 先用
`git ls-remote` 复核，本地计数只使用与这些 tip 对应的显式
`origin/*` remote-tracking ref；不使用本地 `main` 分支名。review worktree
为 detached HEAD，使用 `HEAD == origin/review/...` 校验，而不是
`git branch --show-current`。

固定输入：

```text
origin/main                              = bd47ec97ff7a2881f9bb0316a4a657000a50c020
origin/research/cross-store-substrate    = 0206f17b4255e4b248dafaaeb943be57428dae2f
review/coding-aware-v40-prefetch-20260729= 13671eb708da689137a654946b0d34ba924efb29
merge-base(origin/main, review)          = 3343a79466aa714d34a14d08d3929f7953a47212
```

| 度量 | 命令 | 结果 |
| --- | --- | --- |
| review 分支自有 commit 数 | `git rev-list --count 3343a794..13671eb7` | **`82`** |
| `origin/main` 相对 merge base 前进 | `git rev-list --count 3343a794..origin/main` | **`4945`** |
| `origin/research/cross-store-substrate` 相对 merge base 前进 | `git rev-list --count 3343a794..origin/research/cross-store-substrate` | **`4786`** |
| `origin/main` 与 cross-store 的双向独有量 | `git rev-list --left-right --count origin/main...origin/research/cross-store-substrate` | **`289` / `130`** |

即：`origin/main` 有 `289` 个 cross-store 没有的 commit，cross-store 有 `130` 个 `origin/main` 没有的 commit；两者已双向分叉，**不是**简单的"cross-store = main + 130"关系。

**结论（`derived`）**：review 分支的 merge base 落后 `origin/main` **4,945** 个 commit、落后 `origin/research/cross-store-substrate` **4,786** 个 commit。任何"把 V40 分支整体 merge 回主线/底座"的方案都不是一次策略合并，而是一次跨约 4.9k commit 的大范围回滚风险事件。

### 2.2 相对 shared-core 的 payload 规模（`verified-local`）

`c16bfbb8e`（`kvflow/shared-core`）是 V40 HEAD 的祖先，`git merge-base c16bfbb8e HEAD == c16bfbb8e`。

```text
git diff --shortstat c16bfbb8e..HEAD
  191 files changed, 52549 insertions(+), 220 deletions(-)
```

与审查请求给出的"约 52,549 insertions / 191 files"**逐字一致**。

对 merge base `3343a794` 的差异略大（含 shared-core 自身）：

```text
git diff --shortstat 3343a794..HEAD
  194 files changed, 54266 insertions(+), 14 deletions(-)
```

按目录拆分（`git diff --stat c16bfbb8e..HEAD -- python/ tools/ docs/ KVFLOW.md`）：

```text
  26 files changed, 4751 insertions(+), 203 deletions(-)
```

即：**52,549 行中只有约 4,751 行在 runtime/工具/文档侧**，其余全部集中在 `benchmark/multi_workflow/`。该目录当前共 **163 个 tracked 文件**（`git ls-files benchmark/multi_workflow | wc -l`），涵盖 V8–V44 的全部历史驱动、audit、probe 与 preregistration 脚本。

runtime 侧真正被改动的 16 个 `python/sglang` 文件：

| 文件 | 变更行 | 性质 |
| --- | ---: | --- |
| `python/sglang/srt/mem_cache/kvcomm_exact.py` | `1169`（新增） | V40 的实际执行控制器 |
| `python/sglang/srt/mem_cache/test_kvcomm_exact.py` | `765`（新增） | 上者的测试 |
| `python/sglang/srt/mem_cache/kvcomm/radix_backend.py` | `214` | copy + RoPE 后端 |
| `python/sglang/srt/mem_cache/coding_aware/policy.py` | `126`（新增） | **未接线 seam**（见 §5.3） |
| `python/sglang/srt/mem_cache/coding_aware/test_policy.py` | `131`（新增） | 上者的测试 |
| `python/sglang/srt/managers/scheduler.py` | `54` | canary 控制器初始化 |
| `python/sglang/srt/managers/schedule_policy.py` | `44` | copy 触发点 |
| `python/sglang/srt/mem_cache/radix_cache.py` | `40` | attach/limit/finish hook |
| `python/sglang/srt/managers/schedule_batch.py` | `31` | copy 触发点 + 状态字段 |
| 其余 7 个 | ≤ `17` 每个 | allocator / common / transfer / mixin / `__init__` |

### 2.3 V40 本身的实现 commit（`verified-local`）

```text
git show --stat 03ba74050
03ba7405054f5f1bc6e87058df3fa6162dcc7797 2026-07-28 01:46:52 +0000 Implement V40 grounded observation reuse
 benchmark/multi_workflow/audit_v39_v38_equivalence.py                    | 257 +
 benchmark/multi_workflow/bridge_reuse_litellm_model.py                   |  74 +-
 benchmark/multi_workflow/coding_reuse_policy.py                          |  57 +
 benchmark/multi_workflow/motivate_v40_grounded_observation_island.py     | 400 +
 benchmark/multi_workflow/test_audit_v39_v38_equivalence.py               |   9 +
 benchmark/multi_workflow/test_bridge_reuse_litellm_model.py              |  70 +
 benchmark/multi_workflow/test_coding_reuse_policy.py                     |  38 +
 benchmark/multi_workflow/test_motivate_v40_grounded_observation_island.py|  60 +
 8 files changed, 953 insertions(+), 12 deletions(-)
```

**关键事实（`verified-local`）**：`03ba74050` 一行 `python/sglang/` 都没有改。V40 的 **算法增量只有 953 行、8 个文件，全部在 `benchmark/multi_workflow/`**；它所依赖的 runtime（`kvcomm_exact.py`、`kvcomm/*`、scheduler/radix hooks）来自更早的 shared-core 与前置 commit。

### 2.4 branch scope 检查（`verified-local`）

```text
python3 tools/check_kvflow_branch_scope.py --role coding --base c16bfbb8e...
  coding branch scope: OK                      (rc=0)

python3 tools/check_kvflow_branch_scope.py --role shared --base c16bfbb8e...
  shared branch contains out-of-scope paths:   (rc=1)
    benchmark/multi_workflow/** (163 项)
    python/sglang/srt/mem_cache/coding_aware/{__init__,policy,test_policy}.py
```

即：分支自检工具认定它是合法的 **coding-role** 分支，但显式确认它**不是**可直接进入 shared-core 的内容。这与"不能整分支 merge"的结论一致。

### 2.5 prefetch / integration 引用不可获得（`verified-local`）

审查请求与分支文档反复引用的三个 ref，在当前 `origin`（`git@github.com:ccdd2023/sglang.git`）上**全部无法解析**：

```text
git rev-parse --verify research/prefetch-p8-async-20260722   -> fatal: Needed a single revision
git rev-parse --verify 0ab4fc942   (integration-v2)          -> fatal: Needed a single revision
git rev-parse --verify e44ce40dc   (prefetch tip)            -> fatal: Needed a single revision
```

当前 origin 上只有 `origin/research/prefetch`：

```text
fa86f8f16e6cf08fa3e51f9f9fd5b12cfc303fc0  2026-07-17  docs(kvflow): document middle-KV handoff
git merge-base --is-ancestor origin/research/prefetch HEAD  -> rc=1（不是祖先）
git rev-list --left-right --count origin/research/prefetch...HEAD -> 7  77
git merge-base origin/research/prefetch HEAD -> c16bfbb8e（= shared-core）
```

同时，被审查分支中 **`kvcomm_prefetch` 不存在**：

```text
git ls-files | grep -i prefetch      -> 空
```

`kvcomm_prefetch` 一词在本分支中只出现在 scope guard 的**负例**里（`v40:tools/check_kvflow_branch_scope.py:16,23`、`v40:tools/test_check_kvflow_branch_scope.py:27,34`）。而 `origin/research/prefetch`（非本分支祖先）确实含 `python/sglang/srt/mem_cache/kvcomm_prefetch/{__init__,coordinator,middle_kv,test_coordinator,test_middle_kv}.py`。

#### 2.5.1 对象库级复核（`verified-local`，本轮补充）

上面的 `git rev-parse` 只证明"不是一个可解析的 revision"。为排除"对象在本地但没有 ref 指向它"的可能，进一步做**对象库级**检查：

```text
git cat-file -t 0ab4fc942   -> fatal: Not a valid object name 0ab4fc942
git cat-file -t e44ce40dc   -> fatal: Not a valid object name e44ce40dc
git cat-file -t 0ab4fc9     -> fatal: Not a valid object name 0ab4fc9
git cat-file -t e44ce40     -> fatal: Not a valid object name e44ce40
```

即：**这两个 commit 对象根本不在本地 object 库中**（不是"有对象但无 ref"，而是"对象不存在"）。

再对 `origin` 做一次**在线** `git ls-remote origin` 全量 ref 枚举（`verified-local`，本轮实际联网执行），完整结果中与本议题相关的行只有：

```text
d4a7ec132d80597c7b55a562beb8432e804ab127  refs/heads/integration/coding-aware-prefetch
fa86f8f16e6cf08fa3e51f9f9fd5b12cfc303fc0  refs/heads/research/prefetch
a580c1498b5c9703ef2c6712a6aed89f14c0750f  refs/heads/research/coding-aware-lossy
13671eb708da689137a654946b0d34ba924efb29  refs/heads/review/coding-aware-v40-prefetch-20260729
ef0c665a3d07285f2f02a66f56594721b28072f4  refs/heads/experiment/v40-sota-fair-comparison-v2-20260729
c16bfbb8e8cc83a8b23858808f52833be9091101  refs/heads/kvflow/shared-core
```

结论逐条：

| 分支文档中的引用 | origin 上的真实情况 | 判定 |
| --- | --- | --- |
| `integration/coding-aware-prefetch-v2 @ 0ab4fc942` | **该 ref 不存在**；只有 `integration/coding-aware-prefetch @ d4a7ec132`（即 `v40:KVFLOW.md:127` 自己称为 "old integration" 的那个 stale head） | **不可获得** |
| prefetch tip `e44ce40dc` | `research/prefetch @ fa86f8f16`（2026-07-17，非本分支祖先）；不存在 `research/prefetch-p8-async-20260722` | **不可获得** |
| "frozen from `research/coding-aware-lossy` at `525a03c6b`" | origin 的 `research/coding-aware-lossy` 停在 `a580c1498`，比 review HEAD **落后 71 个 commit**；`525a03c6b` 只存在于 review 分支上 | 表述不精确 |

**结论（`derived`，强化）**：`v40:KVFLOW.md:132`、`v40:docs/kvflow/ARCHITECTURE.md:143-146` 与 `v40:docs/kvflow/COLLABORATOR_QUICKSTART_20260729.md:104-118` 所称的"合并后单元面 **113 tests** 通过（含 `kvcomm_prefetch/test_{coordinator,middle_kv,scheduler}.py` 与 `kvflow_integration/test_composition_v2.py`）"，其**两个构成 commit（`0ab4fc942`、`e44ce40dc`）既不在本地 object 库、也不是 origin 上任何 ref 的 tip**，对应的 `kvcomm_prefetch/` 与 `kvflow_integration/` 目录在被审查分支中也不存在。

因此：**该 113 tests 只能标记为 `external unverified claim`**，不得作为"组合可行"的证据，也不得作为 Gate 5 的 entry 依据。`docs/kvflow/COLLABORATOR_QUICKSTART_20260729.md:110-116` 给出的复现命令在当前 origin 上**不可能**执行成功。

**结论（`derived`）**：`v40:KVFLOW.md:132` 与 `v40:docs/kvflow/ARCHITECTURE.md` 所称的"合并后单元面 **113 tests** 通过"在当前环境**无法独立复现**，因为其输入 commit 既不在 object 库、也不在 origin 的任何 ref 上（详见 §2.5.1）。该数字必须标记为 `external unverified claim`。

---

## 3. 方法的精确定义与真实调用链

### 3.1 用户问题 1 的直接回答：他用了什么方法做有损 KV 恢复？

**答：他使用的恢复 primitive 就是 `R0 Raw+RoPE`——复制 V、按位置 delta 旋转 K；他在此之上新增的，主要是一套 admission / selection / invalidation policy。它不是一条新的恢复公式，也不是 KVCOMM reconstruction、不是 selected-token repair、不是 prefetch。**

拆成两层：

- **数据面 / 恢复 primitive（沿用既有算法）**：`R0 Raw+RoPE` —— `V` 逐 token 原样复制、`K` 施加 `rope_delta = target_start - source_start` 的旋转、其余全部 dense。**没有 context-dependent 修正、没有 selected-token 重算、没有 anchor 插值**。
- **策略面 / admission·selection·invalidation（新增点所在）**：一套 coding-agent 专用的准入与失效规则，决定"哪一段历史 tool observation 允许被搬到下一个请求，以及何时必须放弃"。

因此"有损"来自：token 序列相同，但这段 token 在**新的左上下文**下本应产生不同的 K/V。RoPE 只修正位置坐标，不修正 causal context 差异。这一点分支文档自己也写了（`v40:KVFLOW.md:66-68`）：

> "The reuse remains lossy because a token-identical tool observation was originally encoded under an older left context. RoPE correction fixes position coordinates, not that contextual-state difference."

### 3.2 六步选择算法（`verified-code`）

以下逐步给出 `v40:benchmark/multi_workflow/coding_reuse_policy.py` 与 `v40:benchmark/multi_workflow/bridge_reuse_litellm_model.py` 的行号。

**Step 1 — 滚动窗口与 roll-out 假设**
`bridge_reuse_litellm_model.py:105` `rolling_history_groups: int = Field(default=6, ge=4)`；
`bridge_reuse_litellm_model.py:461` 若 `len(selected_groups) < rolling_history_groups` 则直接放弃（`mode="insufficient_rolling_history"`）；
`bridge_reuse_litellm_model.py:490` 只把 `selected_groups[1:]`（即下一轮 roll 之后仍会保留的 5 组）交给 selector，因为 `selected_groups[0]` 必然滚出。

**Step 2 — "成功只读证据"分类器**
`coding_reuse_policy.py:348` `is_successful_readonly_evidence(group)`，判据为：
1. 命令匹配 `_READONLY_EVIDENCE_COMMAND`（`coding_reuse_policy.py:80-84`，即 `rg|grep|find|sed|cat|head|tail`）；
2. 命令**不**匹配 `_EXECUTION_OR_STATE_COMMAND`（`:62-67`）、`_MUTATION_MARKERS`（`:31-38`）、`_SHELL_MUTATION`（`:39-42`）、`_INPLACE_MUTATION`（`:43-46`）；
3. tool 消息中所有 `<returncode>` 均为 `0`（`:18` 的 `_RETURN_CODE`）；
4. tool 消息总长度 `>= 400` 字符。

**Step 3 — 后续同路径写失效**
`coding_reuse_policy.py:386-440` `grounded_observation_candidates(retained_groups)`：
- `:401-404` 遍历每个 group，非"成功只读"直接跳过；
- `:405` `source_paths = repository_paths(group)`（路径正则见 `:85-89` `_REPOSITORY_PATH`、`:90-94` `_PATCH_PATH`）；
- `:407-412` 对**其后**每个 group 调用 `critical_coding_event_reasons(later)`，仅当返回值含 `"repository_mutation_command"` 才视为写事件；
- `:413-420` 若 `source_paths` 为空、或 `changed_paths` 为空、或两者相交 → 判定 invalid（fail-closed 的意图正确）；
- `:424-429` 只取 `role == "tool"` 的消息进入候选，**显式排除 assistant reasoning 与 tool_calls 文本**（`:437` `"assistant_tokens_selected": 0`）。

> **这一步正是 P0 的所在**：`critical_coding_event_reasons` → `latest_group_risk_reasons` 的写检测覆盖面严重不足。详见 §5.1。

**Step 4 — token 化、唯一性、最小长度**
`bridge_reuse_litellm_model.py:486-521`：
- `:490` 调 `grounded_observation_candidates(selected_groups[1:])`；
- `:493-496` 对每个候选做 `encoded_groups`（`:471-483`），返回 `(ids, find_sublist(prompt_ids, ids))`；
- `:498-503` 只保留 `len(ids) >= reuse_min_tokens` **且** `len(positions) == 1`（在目标 prompt 中**恰好唯一出现**）；
- `bridge_reuse_litellm_model.py:107` `reuse_min_tokens: int = Field(default=128, ge=32)`；
- `:504-509` 无合格候选 → `skip_reason="no_unique_version_valid_observation_at_minimum_size"`（fail closed）。

**Step 5 — 选一个岛 + copy cap + 严格中部**
- `:513-521` `max(...)` 以 `(min(len(ids), reuse_copy_cap), candidate_group_index)` 为 key，即"截断后 token 数最大，同分取更新的那个"；
- `bridge_reuse_litellm_model.py:106` `reuse_copy_cap: int = Field(default=4096, ge=128)`；`coding_reuse_policy.py:890-920` `effective_copy_cap` 对 V40 arm 不放宽（走 `:920 return base_cap`）；
- `:600-611` `capped_tail(...)`（定义在 `:135`）后强制 `source_start > 0` 且 `source_start + len(segment_ids) < len(prompt_ids)`，否则 `skip_reason="span_not_strictly_middle"`（`:610`）；此前 `:582` 还再次强制 `len(segment_ids) >= reuse_min_tokens and len(positions) == 1`。

**Step 6 — 身份指纹与 sidecar 注册**
`:613-641` 生成 `source_prompt_hash`（`:613`）、`segment_token_hash`、`source_prefix_token_hash`、`content_hash`（`:620`，`sha256(arm + ":" + source_prompt_hash + ":" + segment_token_hash)`），并把 `source` 写入 version-3 sidecar（`:309-382` `_atomic_sidecar_update`；`bridge_reuse_litellm_model.py:5` 明确 "no HTTP field may select KV spans"）。

### 3.3 真实执行调用链（`verified-code`）

```text
[客户端 / bridge adapter 侧]
coding_reuse_policy.grounded_observation_candidates()        v40:coding_reuse_policy.py:386
        │  （candidates + auditable decision dict）
        ▼
BridgeReuseLitellmModel._future_source()                     v40:bridge_reuse_litellm_model.py:432-641
        │  （token 化 / 唯一性 / 最小长度 / copy cap / 严格中部 / hash）
        ▼
_atomic_sidecar_update()  ->  version-3 reuse manifest        v40:bridge_reuse_litellm_model.py:309
        │  （本地文件；SGLANG_KVCOMM_EXACT_CANARY_MANIFEST 指向它）
════════════════════════ 进程边界 ════════════════════════
[SGLang server 侧]
Scheduler.init_kvcomm_exact_canary()                          v40:python/sglang/srt/managers/scheduler.py:792
        │  guard: manifest 非空、TP=PP=1、page_size=1、无 spec-decode、
        │         非 multimodal/SWA/SSM、tree_cache 必须是 Python RadixCache、
        │         tree_cache.kvcomm.config.core_enabled（= SGLANG_KVCOMM_CORE=1）
        ▼
ExactMiddleCanaryController.from_manifest()                   v40:kvcomm_exact.py:262
        │  version ∈ {1,2,3}（:275）；version==3 才启用动态 sidecar（:351,:369,:463）
        ▼
RadixCache.match_prefix() hook                                v40:radix_cache.py:435-447
        │  controller.is_target_request() / maybe_attach_target() / ordinary_prefix_match_limit()
        ▼
schedule_policy.py:608-612  或  schedule_batch.py:1483-1504
        │  if controller.copy_ready(req): controller.copy_into_request(req)
        ▼
ExactMiddleCanaryController.copy_into_request()               v40:kvcomm_exact.py:988
        │  构造 KVReusePlan（:1018）：1 个 TransferSpan + dense_prefix + dense_suffix
        │  rope_delta = case.target_start - case.source_start        (:1026)
        │  require_full_coverage=True                                 (:1032)
        ▼
KVCommManager.execute(plan, RadixKVTransferBackend)           v40:kvcomm/manager.py + transfer.py
        │  机械校验：handle current / token slice 相同 / 边界合法 / 无重叠无空洞
        ▼
RadixKVTransferBackend                                        v40:kvcomm/radix_backend.py
        │  V 原样 copy；K 全部按 rope_delta 旋转；任一部分拷贝或部分旋转 = 硬不变量错误
        ▼
RadixCache.cache_finished_req() hook                          v40:radix_cache.py:488-490
           controller.finish_request(req)  ->  store.unpin(lease)    (:1111,:1115)
           controller.maybe_materialize_source(req)                   (:582)
```

### 3.4 失败即 dense（fail-closed 路径，`verified-code`）

`v40:kvcomm_exact.py:1144` `_fallback(req, reason)`；`copy_into_request` 中的 fallback 触发点：
`missing_request_pool_slot`（`:993`）、`empty_dynamic_copy`（`:1005`）、
`stats.copied_k_tokens != copy_length or not stats.mechanically_valid` → 取 `stats.fallback_reasons[0]` 或 `mechanical_validation_failed`（`:1057-1066`）、
`target_allocation_capacity`（`MemoryError`，`:1073`）、`copy_exception`（`:1076`，会 re-raise）。

**判断（`derived`）**：数据面的机械安全性设计是合格的——token 相等、边界、完整覆盖、部分拷贝均被检查，失败一律 dense。**问题不在数据面，而在准入策略面**（§5）。

### 3.5 明确不是什么（`verified-code` + `derived`）

| 说法 | 是否成立 | 依据 |
| --- | --- | --- |
| V40 是 prefetch | **否** | 分支中无 `kvcomm_prefetch`；`v40:KVFLOW.md:38-39` 自述 "pure KV-reuse method; it does not prefetch"；多个 audit 显式写 `"prefetch": False`（如 `audit_v40a2_timeout_failure.py:147`、`audit_v41_capacity_deadlock.py:200`、`audit_v43_call_budget_collapse.py:277`） |
| V40 是 KVCOMM 重建 | **否** | 无 canonical base、无 `ΔK/ΔV`、无 anchor pool、无 multi-anchor 插值、无 entropy/length shareability gate、无 neighboring-prefix offset。KVCOMM 的九项构成见 `docs:research/KVCOMM_SGLANG_FEASIBILITY_REPORT.md` §3 |
| V40 是 CacheBlend selective repair | **否** | 无 HKVD/K-deviation 打分，无 selected-token 逐层重算；只有"全岛复用 + 位置修正"这一半底座 |
| V40 是 Cache-Craft / CacheTune | **否** | 无 CCI/CFO 判据，无 roofline repair-ratio controller |
| V40 是 KVFlow | **否** | 无 steps-to-execution、无 priority eviction、无 CPU 分层调度 |
| V40 的 source 是 synthetic replay | **否** | source 由**服务端**在前一个真实 agent 请求完成时物化：`v40:python/sglang/srt/mem_cache/kvcomm_exact.py:582` `ExactMiddleCanaryController.maybe_materialize_source()`，经 `v40:python/sglang/srt/mem_cache/radix_cache.py:488-490` 的 `cache_finished_req` hook 触发（**不是** bridge 侧的 `bridge_reuse_litellm_model.py:582`） |
| token 完全相同 ⇒ KV 完全相同 | **否** | 见 §3.1；这正是"lossy"的来源 |

---

## 4. 先例映射：跟哪个方法最像，或是哪几个研究的组合

### 4.1 用户问题 2 的直接回答

**答：恢复 primitive 就是本项目 Phase 4 定义的 `R0 Raw+RoPE`（不是新公式）。V40 的新增点主要在 admission / selection / invalidation policy 这一层。整体 = R0 恢复 primitive + Prompt-Cache/PIC 式 non-prefix modular reuse 的问题设定 + 一套 V40 独有的 coding observation 准入/失效策略 + 通用 segment lifecycle。**

用一行公式表达：

```text
V40  ≡  R0(Raw + RoPE)  ⊕  PromptCache/PIC-style non-prefix modular reuse 问题设定
        ⊕  V40-specific coding observation selection & invalidation policy
        ⊕  generic segment identity / lease / transfer lifecycle
```

### 4.2 与本项目 `R0 Raw+RoPE` 的对照（最近邻，`verified-code`）

本项目 Phase 4 对 R0 的定义（`docs:research/phase_reports/PHASE4_RECOVERY_METHODS_REPORT.md:149`）：

> "**R0 Raw+RoPE** | 复制 body K/V，只做符号化 RoPE 位置修正；speed-only 上界 | 无 context-dependent repair；**显式非忠实 KVCOMM 复现**"

底座实现（`xs:python/sglang/srt/mem_cache/approx_kv/`）：

| 项 | `xs:` (R0) | `v40:` | 是否等价 |
| --- | --- | --- | --- |
| 模式常量 | `RecoveryMode.COPY = "copy"`（`types.py:20`，注释 "copy + RoPE-correction (R0 path)"） | `KVReusePlan` + 单 `TransferSpan` | 等价 |
| 变换函数 | `RadixKVTransferBackend.copy_and_rotate()`（`radix_backend.py:196`） | `RadixKVTransferBackend`（`v40:kvcomm/radix_backend.py`） | 同名同构 |
| V 处理 | `move_kv_cache(target, source)` 全层原样搬运（`radix_backend.py:213`），V 不旋转 | V 不旋转 | 等价 |
| K 处理 | `_rotate_all_copied_keys(rope_delta)`（`:217-221`）→ `apply_rotary_emb`（`:340`） | 同 | 等价 |
| delta 定义 | `rope_delta = overlap_start - source_position`（`runtime.py:537`） | `rope_delta = case.target_start - case.source_start`（`v40:kvcomm_exact.py:1029`） | 等价 |
| 覆盖要求 | `require_full_coverage=True`（`transfer.py:84`） | `require_full_coverage=True`（`v40:kvcomm_exact.py:1034`） | 等价 |

**结论（`derived`）**：V40 的数据面与本项目 R0 在数学与实现上**没有实质差异**。因此 V40 **不应**被命名为新的恢复 primitive（不应叫 R6/L0）。

### 4.3 与 EPIC / LegoLink（R1）的关系

`xs:python/sglang/srt/mem_cache/approx_kv/config.py:37-38`：

```python
# k=0 degenerates to the plain raw-copy (R0) path.
SUPPORTED_EPIC_K_VALUES: tuple[int, ...] = (0, 2, 4, 8, 16, 32)
```

`EPICLeadingKPlugin.build_plan()` 在 `k>0` 时把前 k 个 token 划为 `DenseRange(0, k, EPIC_LEADING_K_REPAIR_REASON)`，其余走 COPY。

**V40 只与 `k = 0`（raw-link）端点接近**：它没有 leading-k attention-sink 逐层重算。V40 的 dense prefix 是"岛之外的自然 prefix"，不是 EPIC 意义上的 *repair head*。因此：**V40 ≠ EPIC**，只是落在 EPIC 参数族的退化端点上。

### 4.4 与 CacheBlend 的关系

共享的只有底座的两条：**（a）整段复用；（b）位置修正**。
不共享的是 CacheBlend 的核心：**HKVD / K-deviation 打分 + selected-token 逐层 recompute**（本项目 R2，`docs:research/phase_reports/PHASE4_RECOVERY_METHODS_REPORT.md:151`）。

分支代码中 `CACHEBLEND_DAMAGE_RATE = 9 / 167` 只是一个被冻结的**历史参考常数**，出现在 `v40:run_v41_v40_independent_campaign.py:54`、`v40:run_v43_new_verified_v40_campaign.py:116`、`v40:run_v44_dense_sensitive_v40_campaign.py:59`；它不代表 V40 实现了 CacheBlend。

**禁止表述**：不得把 V40 称为 "CacheBlend-style selective repair"，也不得把 V40 的 dense prefix/suffix 称为 "repair"。

### 4.5 与 Prompt Cache / PIC 家族的关系（问题设定层面的最近邻）

V40 真正共享的"问题设定"是 **non-prefix modular KV reuse**：允许把一个已经计算过的、位于 prompt **中部**的模块（这里是一次 tool observation）在新的 prompt 中以新的绝对位置重新装配，并对位置编码做重标定。这正是 Prompt Cache / PIC（position-independent caching）族的核心设定；本项目研究综合中列出的 MEPIC `2512.16822`、MiniPIC `2606.13126` 也覆盖了 "code chunk/file span、canonical pages、position-independent reuse"（`docs:research/RESEARCH_SYNTHESIS.md`）。

**差异点**：Prompt-Cache/PIC 族通常要求模块边界由 prompt 模板显式声明；V40 的贡献是**从真实 agent trajectory 里自动、保守地"发现"这样一个模块**，并用"文件版本失效"作为其有效期判据。

### 4.6 与 KVFlow / KVCOMM 的职责边界（本项目既定约束）

按本项目既定边界（`docs:research/RESEARCH_SYNTHESIS.md`、仓库协作指令）：

- KVFlow 负责 workflow-aware cache priority / eviction / CPU backup / prefetch / scheduling；
- KVCOMM `2510.12872` 负责 base KV、context-dependent offset、RoPE relocation、anchor interpolation、dense fallback。

V40 **两者都不做**：它不做调度，也不做 context offset。它做的是**准入判定**——这是一个此前在两篇论文中都没有被作为一等问题处理的维度（KVCOMM 用 length + entropy gate，V40 用 *repository 语义事件* gate）。

### 4.7 组合定位小结（`derived`）

| 层 | V40 的成分 | 最近先例 | 新颖性判断 |
| --- | --- | --- | --- |
| 数据面 primitive | copy V + RoPE K | 本项目 R0 / EPIC k=0 / PIC 族 | **无新颖性** |
| 复用几何 | 非 prefix、中部单岛、唯一匹配 | Prompt Cache / PIC / MEPIC | 低 |
| 身份与 lifecycle | token hash + generation + lease | 通用 KV store 设计（含本项目 `xs:` 底座） | 无 |
| **准入 gate** | 成功只读命令 + 路径提取 + 后续同路径写失效 | 未见完全相同的公开系统；与 Streaming Knowledge Compilation 的 staleness、FCGraft 的 function-object patch 相邻 | **这是唯一可能有增量的一层** |
| 触发时机 | 由真实前序请求自然产生 source | 通用 | 无 |

**因此**：如果要为这条线写论文，可主张的增量只能是 **"repository-event-grounded admission control for non-prefix KV reuse in coding agents"**，且必须在与 R0/PIC 相同的数据面下做对照，而**不能**主张新的恢复算法。这与本项目 `docs:research/RESEARCH_SYNTHESIS.md` 中"组合 novelty 约 2/5，实现 version consistency / dependency invalidation / calibrated reconstruction / artifact-level planning 后保守上限约 3.3–3.6/5"的判断一致。

---

## 5. 代码审计与阻塞项

### 5.0 阻塞项总表

| ID | 等级 | 标题 | 状态 | 证据 |
| --- | --- | --- | --- | --- |
| B-01 | **P0** | 后续同路径写事件大面积漏检，本应按freshness policy失效的 observation 仍 `eligible=1 / invalidated=0` | **本地复现** | §5.1 |
| B-02 | **P0** | 同一 group 内"读+写"混合命令被判为 `is_successful_readonly_evidence=True` 并直接 eligible | **本地复现** | §5.2 |
| B-03 | P1 | review request 列出的 active entry point `build_coding_reuse_plan` 无生产调用者 | 已确认 | §5.3 |
| B-04 | P1 | `SGLANG_CODING_AWARE_LOSSY` / `coding_aware_lossy_enabled` 是 dead flag | 已确认 | §5.4 |
| B-05 | P1 | `gc_expired_leases` 无生产调用；`_is_leased` 不检查过期 → TTL 不能自动解锁 | 已确认 | §5.5 |
| B-06 | P1 | 实验 runner/审计器硬编码 `/home/gfy/**` 绝对路径，导致 2 个测试在任何其他机器上必失败 | **本地复现** | §7.3 |
| B-07 | P2 | `KVFLOW.md` / `ARCHITECTURE.md` 引用的 prefetch/integration ref 在当前 origin 不可获得 | 已确认 | §2.5 |

### 5.1 B-01（P0）：后续同路径写事件大面积漏检

#### 5.1.1 违反的硬不变量

`v40:docs/kvflow/CODING_AWARE_V40_REVIEW_REQUEST_20260729.md:66-72` 自述的 required invariants 中有两条：

> - "an unknown path is treated as proof that a read survived a later write;"
> - "a failed or **mutating** tool interaction can become a reusable observation;"

并在 `:33` 明确要求候选必须 "has not been invalidated by a later write to the same repository path"。

#### 5.1.2 根因（`verified-code`）

失效链是：

```text
grounded_observation_candidates            coding_reuse_policy.py:407-412
   -> critical_coding_event_reasons(later)  coding_reuse_policy.py:297
        -> latest_group_risk_reasons(later) coding_reuse_policy.py:205
```

`latest_group_risk_reasons` 的写检测只有四个来源（`coding_reuse_policy.py:217-224`）：

```python
217:    if any(marker in command_lower for marker in _MUTATION_MARKERS):
218:        reasons.append("repository_mutation_command")
219:    elif (
220:        _SHELL_MUTATION.search(commands)
221:        or _INPLACE_MUTATION.search(commands)
222:        or _OPEN_WRITE_MUTATION.search(commands)
223:    ):
224:        reasons.append("repository_mutation_command")
```

其覆盖面为：

| 常量 | 行 | 实际覆盖 |
| --- | --- | --- |
| `_MUTATION_MARKERS` | `:31-38` | `.write_text(`、`.write_bytes(`、`apply_patch`、`git apply`、`git checkout `、`git restore ` |
| `_SHELL_MUTATION` | `:39-42` | 行首/`&&`/`;`/`\|\|` 后的 `rm` / `mv` / `cp` |
| `_INPLACE_MUTATION` | `:43-46` | `sed ... -i`、`tee` |
| `_OPEN_WRITE_MUTATION` | `:47-50` | `open('...', 'w|a|x|+')` |

而分支中**已经定义了**一个专门的 shell 重定向写检测器：

```python
51: _SHELL_SOURCE_WRITE = re.compile(
52:     r"\b(?:cat|printf|echo)\b[^\n]*(?:>>|>)\s*"
53:     r"(?:/testbed/|\./)?[^\s;&|]+"
54:     r"\.(?:py|pyi|toml|yaml|yml|json|cfg|ini)\b",
55:     re.I,
56: )
```

它只被 `is_shell_source_write`（`:489`）和 `repository_commit_phase_event`（`:502`，服务于已冻结的 V37/V38 arm）使用，**从未进入 V40 的失效链**。

#### 5.1.3 复现（`verified-local`，**在固定 Docker image 内、只读挂载**）

> **R12 约定**：本报告的**全部正式复现命令一律在固定 Docker image 内执行**，被审查 worktree 以 `:ro` 只读挂载。宿主机 Python 直接执行**不作为证据**。

在 `ghcr.io/ccdd2023/sglang@sha256:0be6e16e…` 内构造 `[读 pkg/a.py] → [写 pkg/a.py]` 两组，实测输出：

```text
== later-same-path write invalidation ==
cat > pkg/a.py         risk=-                              eligible=1 invalidated=0
echo x > pkg/a.py      risk=-                              eligible=1 invalidated=0
perl -pi               risk=-                              eligible=1 invalidated=0
dd of=                 risk=-                              eligible=1 invalidated=0
truncate -s 0          risk=-                              eligible=1 invalidated=0
git mv                 risk=-                              eligible=1 invalidated=0
rsync                  risk=-                              eligible=1 invalidated=0
bare cp (control)      risk=repository_mutation_command    eligible=0 invalidated=1
tee (control)          risk=repository_mutation_command    eligible=0 invalidated=1
```

复现命令（固定 image、只读挂载、不写宿主任何文件）：

```bash
cd /home/chris/Workspaces/kvcache-research/worktrees/coding-aware-v40-prefetch
docker run --rm -i --user 1000:1000 \
  -v "$PWD":/w:ro -w /w/benchmark/multi_workflow -e HOME=/tmp \
  ghcr.io/ccdd2023/sglang@sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781 \
  python - <<'PY'
import sys, json; sys.path.insert(0, '.')
import coding_reuse_policy as P
def grp(cmd, obs):
    return [{"role":"assistant","tool_calls":[{"function":{"name":"bash",
             "arguments": json.dumps({"command": cmd})}}]},
            {"role":"tool","content": obs}]
OK = "x"*420 + "\n<returncode>0</returncode>"
for name, w in [("cat >","cat > pkg/a.py <<'EOF'\nprint(1)\nEOF"),
                ("echo >","echo x > pkg/a.py"),
                ("perl -pi","perl -pi -e 's/a/b/' pkg/a.py"),
                ("dd of=","dd if=/dev/null of=pkg/a.py"),
                ("truncate","truncate -s 0 pkg/a.py"),
                ("git mv","git mv pkg/a.py pkg/b.py"),
                ("rsync","rsync -a /src/a.py pkg/a.py"),
                ("cp","cp /src/a.py pkg/a.py"),
                ("tee","tee pkg/a.py < /src/a.py")]:
    g = [grp("cat pkg/a.py", OK), grp(w, "ok\n<returncode>0</returncode>")]
    _, d = P.grounded_observation_candidates(g)
    print("%-10s eligible=%d invalidated=%d risk=%s" % (
        name, d["eligible_observations"], d["version_invalidated_observations"],
        P.critical_coding_event_reasons(g[1])))
PY
```

**mixed group 部分**同样在同一容器内测得：

```text
== mixed read+write in one group ==
cat;echo>   readonly=True critical=[] eligible=1
sed;perl    readonly=True critical=[] eligible=1
head;trunc  readonly=True critical=[] eligible=1
```

#### 5.1.4 逐条根因

| 写方式 | 为什么漏检 |
| --- | --- |
| `cat > f.py` / `echo x > f.py` | 只有 `_SHELL_SOURCE_WRITE` 能匹配，而它不在失效链中 |
| `perl -pi -e ... f.py` | 四个常量都不含 `perl` |
| `dd ... of=f.py` | 不含 `dd` |
| `truncate -s 0 f.py` | 不含 `truncate` |
| `git mv a.py b.py` | `_MUTATION_MARKERS` 只有 `git apply/checkout /restore `；`_SHELL_MUTATION` 要求 `mv` 出现在行首或 `&&`/`;`/`\|\|` 之后，`git mv` 中的 `mv` 前面是 `git `，不匹配 |
| `rsync ... f.py` | 不含 `rsync` |

补充：`git mv` 还会同时造成**路径重命名**，即使检出为写，`repository_paths(later)` 也会把 `pkg/b.py` 算进 `changed_paths`，与 `source_paths={pkg/a.py}` 不相交 —— 属于第二重漏洞（路径别名/重命名未建模）。

#### 5.1.5 影响的精确界定（**必须按此表述，不得夸大**）

先明确**没有**发生什么：

| 不成立的说法 | 为什么不成立 |
| --- | --- |
| "旧 KV 被配到了新的 target token 上" | **不会**。复用要求该 observation 的 token 序列在 target prompt 中**逐 token 相同**且**唯一出现**（`bridge_reuse_litellm_model.py:500-503`、`:582`），执行前 `copy_into_request` 还会再做一次 token slice 相等的机械校验（`kvcomm_exact.py:1057-1066`）。token identity **仍然通过**。 |
| "这是 data corruption / KV 与 token 错位" | **不是**。span 的 `source_start`/`target_start`/`length`/`rope_delta` 全部一致，V 原样、K 按 delta 旋转，机械不变量成立。 |
| "dense 就不会看到这段旧文本" | **不对**。被选中的 observation 文本仍然实实在在地存在于**当前** target prompt 中（否则唯一匹配不可能成立）。Dense 基线会对**同一段旧文本**做完整 prefill。 |

再明确**实际**发生了什么：

> 漏检的失效**违反的是 freshness / abstention policy**，不是数据面正确性。
>
> V40 的设计承诺是："当一段 observation 所描述的仓库状态**已经被后续写操作改变**时，就应当**放弃**复用它、退回 dense"（`v40:docs/kvflow/CODING_AWARE_V40_REVIEW_REQUEST_20260729.md:33,70`）。漏检使得一个**语义上已经过时**的 historical observation 仍然被送进 lossy reuse 路径。

因此真实风险是**两层叠加**，而不是错位：

1. **策略层（本 P0 的直接后果）**：本应 abstain 的样本没有 abstain。该段文本描述的是旧版文件内容，而 agent 的后续推理很可能依赖它——但这一点 Dense 也一样（同样的旧文本在 prompt 里）。**所以这一层本身不改变模型看到的 token，只改变"哪些请求走了 lossy 路径"**。
2. **机制层（V40 固有的 lossy 性质）**：一旦进入复用路径，这段 KV 就是在**旧的左上下文**下算出来的，与当前 target 左上下文不同（§3.1）。RoPE 只修位置。

两层叠加的净效应是：**V40 的准入 gate 无法履行其声明的保守性，本应被排除的、语义已过时的 observation 会以 lossy KV 的形式进入推理路径**，而 gate 正是分支用来论证"复用是安全的"的唯一依据。

**为什么仍然必须 block approval**：
- 它直接违反分支自己列为 required invariants 的两条（`:70` "a failed or mutating tool interaction can become a reusable observation"、`:33` "has not been invalidated by a later write to the same repository path"）；
- 数据面的 token 相等校验**在设计上就不可能**发现策略层的版本判定错误（两者检查的是不同的东西）；
- 因此这是一个**只能在策略层修复**的缺陷，且当前策略层的覆盖率无法通过正则枚举补全（§5.6）。

### 5.2 B-02（P0）：同一 group 内"读 + 写"混合命令直接 eligible

#### 5.2.1 复现（`verified-local`，固定 Docker image 内、只读挂载；命令同 §5.1.3）

```text
== mixed read+write inside the SAME group ==
cat ; echo >       readonly_evidence=True critical=-  eligible=1
sed ; perl -pi     readonly_evidence=True critical=-  eligible=1
head ; truncate    readonly_evidence=True critical=-  eligible=1
```

具体命令串：
- `cat pkg/a.py; echo x > pkg/a.py`
- `sed -n '1,50p' pkg/a.py; perl -pi -e 's/a/b/' pkg/a.py`
- `head -n 50 pkg/a.py; truncate -s 0 pkg/a.py`

#### 5.2.2 根因（`verified-code`）

`is_successful_readonly_evidence`（`coding_reuse_policy.py:348-401`）的排除集合是：

```python
366:    if (
367:        _EXECUTION_OR_STATE_COMMAND.search(commands)
368:        or any(marker in command_lower for marker in _MUTATION_MARKERS)
369:        or _SHELL_MUTATION.search(commands)
370:        or _INPLACE_MUTATION.search(commands)
371:    ):
372:        return False
```

注意：**连 `_OPEN_WRITE_MUTATION` 都没有包含**（`latest_group_risk_reasons` 有，这里没有），更不含 `_SHELL_SOURCE_WRITE`。此外 `_INPLACE_MUTATION` 的 `\bsed\b[^\n;&|]*\s-i` 中的字符类显式排除了 `;`、`&`、`|`，所以它**无法跨越命令分隔符**匹配。

结果：一个既读又写同一文件的交互，被归类为"成功的只读证据"，并在同一轮内直接成为可复用 source。

#### 5.2.3 影响

比 B-01 更严重：B-01 至少需要一个"后续 group"才出错；B-02 在**当前 group 内**就把一个 mutating interaction 当成 reusable observation，直接违反 `v40:docs/kvflow/CODING_AWARE_V40_REVIEW_REQUEST_20260729.md:70` 的 "a failed or mutating tool interaction can become a reusable observation" 阻断条件。

### 5.3 B-03（P1）：文档声明的 active entry point 与代码事实矛盾

`v40:docs/kvflow/CODING_AWARE_V40_REVIEW_REQUEST_20260729.md:55-60` 写：

```text
The active V40 entry points are:
  grounded_observation_candidates(...)
  reuse_arm="coding_grounded_observation_island_v40"
  build_coding_reuse_plan(...)
```

实测（`verified-local`）：

```text
grep -rn "build_coding_reuse_plan" --include=*.py .
  python/sglang/srt/mem_cache/coding_aware/policy.py:53   (定义)
  python/sglang/srt/mem_cache/coding_aware/__init__.py:6,9 (re-export)
  python/sglang/srt/mem_cache/coding_aware/test_policy.py:9,41,61,84,102 (仅测试)
  tools/{check,test_check}_kvflow_branch_scope.py         (仅路径字符串)
  docs/kvflow/CODING_AWARE_V40_REVIEW_REQUEST_20260729.md (仅文档)
```

**没有任何生产代码路径调用它**。真正被 server 执行的是 `v40:kvcomm_exact.py:988 copy_into_request()`，它自己构造 `KVReusePlan`（`:1018-1033`），完全不经过 `coding_aware/policy.py`。

因此 `python/sglang/srt/mem_cache/coding_aware/policy.py`（126 行）是一个**未接线的 seam**：设计意图良好（`CodingRisk.CRITICAL` → dense、token mismatch → dense、`head_tokens` 预算、`outside_coding_segments` 兜底覆盖），但当前对 V40 的实际行为**零影响**。

**要求**：review request 必须更正为"active = `coding_reuse_policy.grounded_observation_candidates` + `bridge_reuse_litellm_model._future_source` + `kvcomm_exact.copy_into_request`；`coding_aware/policy.py` = 未接线的未来 seam"。否则审查者会把审查精力放到不生效的代码上。

### 5.4 B-04（P1）：dead feature flag

`v40:python/sglang/srt/mem_cache/kvcomm/config.py`：

```python
28:    coding_aware_lossy_enabled: bool = False
42:                "SGLANG_CODING_AWARE_LOSSY",
62:        coding = _read_bool(env, "SGLANG_CODING_AWARE_LOSSY", False)
64:        if (coding or prefetch) and not core:
65:            raise ValueError("SGLANG_KVCOMM_CORE=1 is required when ...")
71:            coding_aware_lossy_enabled=coding,
```

全仓库对 `coding_aware_lossy_enabled` 的读取只有 `v40:python/sglang/srt/mem_cache/kvcomm/test_core.py:77,98` 两处断言。**没有任何 runtime 分支依赖它**。

而实际开关是（`verified-code`）：

1. `SGLANG_KVCOMM_EXACT_CANARY_MANIFEST` **非空**（`v40:scheduler.py:795-798`）——这是真正的总开关；
2. `SGLANG_KVCOMM_CORE=1`（`v40:scheduler.py:815-819` 检查 `tree_cache.kvcomm.config.core_enabled`）；
3. 一组硬 guard：`tp_size==1 and pp_size==1`（`:799`）、`page_size==1`（`:801`）、无 spec-decode（`:803`）、非 multimodal / 非 hybrid SWA / 非 hybrid SSM（`:805-812`）、`type(tree_cache) is RadixCache`（`:813`）。

**后果**：`v40:KVFLOW.md:153` 与 `v40:docs/kvflow/COLLABORATOR_QUICKSTART_20260729.md:94` 指导用户用 `SGLANG_CODING_AWARE_LOSSY=0/1` 切换 coding-only 与 Dense 模式，但这个变量**不控制任何行为**。据此设计的"四模式对照实验"（Dense / Coding-only / Prefetch-only / Combined）会得到**错误的模式标签**。这是实验设计层面的高危缺陷。

### 5.5 B-05（P1）：lease TTL 不能自动解锁

`v40:python/sglang/srt/mem_cache/kvcomm/store.py`：

```python
139:    def pin(self, handle: KVSegmentHandle, ttl_s: float) -> KVLease:
149:                expires_at_s=time.monotonic() + ttl_s,

159:    def gc_expired_leases(self, now_s: float | None = None) -> int:
165:                if lease.expires_at_s <= now_s

235:    def _is_leased(self, key: KVSegmentKey, generation: int) -> bool:
236:        return any(
237:            lease.key == key and lease.generation == generation
238:            for lease in self._leases.values()
239:        )
```

两个事实：

1. `gc_expired_leases` 在整个仓库中**只有测试调用**（`v40:kvcomm/test_core.py:224`），无生产调用者；
2. `_is_leased`（`:235`）**不检查 `expires_at_s`**。

而 `_is_leased` 正是三个关键路径的守卫：`register` 拒绝替换被 lease 的段（`:97`）、`release` 拒绝释放（`:212`）、`_evict_unleased_if_needed` 挑选 victim（`:250`）。

`v40:kvcomm_exact.py:179` 默认 `lease_ttl_s = 300.0`，`:765` 与 `:870` 都会 `pin(...)`，正常释放在 `finish_request`（`:1111-1115`）。

**后果（`derived`）**：一旦请求异常路径漏掉 `finish_request`（abort、reject、exception、reset），该 lease **永久有效**：
- `_evict_unleased_if_needed`（`:241-257`）会一路找不到 victim，最终 `raise RuntimeError("KV segment store capacity is fully pinned")`；
- TTL 存在但形同虚设，因为唯一的清理函数不会被调用。

这与本项目 Phase 6 的教训直接对应：`xs:` 底座为此专门补了 `release_provisional_recovery_slots` 在 **rejection**（`xs:scheduler.py:3045`）与 **abort**（`xs:scheduler.py:4090`）两条路径上的调用（commit `15634baf6` "fix: release recovery slots on rejection and on abort"）。V40 分支的 store 没有等价保障。

### 5.6 推荐的根修方案（不要继续堆命令正则）

**不要**再往 `_SHELL_MUTATION` / `_INPLACE_MUTATION` 里加 `perl|dd|truncate|rsync|install|ln|python -c open(...)|>>`。命令行写文件的方式在实践中是**开放集合**，正则枚举必然继续漏。

推荐改为**结构化 provenance + fail-closed**：

1. **由 tool wrapper 提供结构化字段**，而不是从命令字符串反推：
   ```json
   {
     "tool": "bash",
     "read_paths":  ["pkg/a.py"],
     "write_paths": [],
     "unknown_effect": false,
     "repo": "/testbed",
     "worktree_generation": 17,
     "source_content_sha256": {"pkg/a.py": "…"}
   }
   ```
   实现方式建议：在 agent 侧对每条命令前后各做一次 `git status --porcelain=v2 -z` + `git stash list` 或 worktree 内容哈希快照，diff 出真实 `write_paths`；或直接用 `strace`/`LD_PRELOAD`/`fanotify` 采集 open(2) 写标志。至少要做到"命令执行前后 worktree generation 变化 ⇒ 该 group 标记为写"。
2. **worktree generation 单调计数**：任何一次 generation 变化都使该 generation 之前采集的所有 observation 岛失效，除非能逐路径证明该路径内容哈希未变。
3. **source path content hash 绑定**：注册 source 时记录被读文件的 `sha256`；复用前在 target 侧重新计算并比对，不一致直接 dense。这可以把"版本失效"从"猜命令语义"变成"验证内容"。
4. **未知效应 fail closed**：`unknown_effect=true`（无法解析的命令、非 bash 工具、超时、被截断输出）一律视为写。当前实现相反——不认识的命令默认视为无写。
5. **路径规范化与重命名建模**：`git mv` / `mv` / 符号链接 / 相对路径 / `/testbed` 前缀必须归一化到 repo-relative 规范路径，并把 rename 建模为"旧路径写 + 新路径写"。
6. **对抗性 + property 测试**（见 §9.1）：任何新增写工具都必须先加入 adversarial matrix，再改代码。

---

## 6. 现有证据强度审计

### 6.1 总原则

> **所有详细实验的 raw 数据都在作者机器 `/home/gfy/CodeMAS_Project/kvflow-artifacts` 下，不在 Git 中，本环境不存在。因此所有具体数字只能标注为 `external unverified claim`。**

本地实测（`verified-local`）：仓库中**没有任何 raw 实验结果 JSON**。三个被 tracked 的 JSON（`swebench_verified_{bridge,complex,medium}_v1.json`）都是 preregistration 的任务清单，且其 `local_snapshot` 字段（各文件第 10 行）仍指向 `/home/gfy/CodeMAS_Project/kvflow-artifacts/**`。

### 6.2 两套 12-task cohort 必须严格区分（**不可合并**）

分支文档里出现了两组"12 个任务"的数字，它们**任务集不同、Dense 基线不同、报告口径不同**：

#### Cohort A — V44 Dense-sensitive development campaign（`external unverified claim`）

来源：`v40:KVFLOW.md:82-95`；runner `v40:benchmark/multi_workflow/run_v44_dense_sensitive_v40_campaign.py`（`TASKS` 见 `:41-52`，`SELECTION_SHA256 = 78663ee1…` 见 `:55-58`，`STEP_LIMIT = 32` 见 `:40`，`DENSE_PASS_SENSITIVITY_MIN = 2` 见 `:60`）。

| 指标 | Dense | General contiguous reuse | V40 |
| --- | ---: | ---: | ---: |
| Official resolved | `3/12` | `3/12` | `4/12` |
| Wilson 95% CI | 8.9–53.2% | 8.9–53.2% | 13.8–60.9% |
| Damage among 3 Dense-pass | — | `1/3` | `0/3` |
| Rescue among 9 Dense-fail | — | `1/9` | `1/9` |
| Copied tokens | 0 | 487,144 | 171,139 |
| Fixed-order host-resident median TTFT | `357.6 ms` | `335.7 ms` | `327.5 ms` |

#### Cohort B — three-method development cohort（`external unverified claim`）

来源：`v40:docs/kvflow/CODING_AWARE_V40_REVIEW_REQUEST_20260729.md:111-116`、`v40:docs/kvflow/COLLABORATOR_QUICKSTART_20260729.md:37-39`。

| 指标 | Dense | V40 |
| --- | ---: | ---: |
| Twelve-task SWE-bench development cohort | `6/12` | `4/12` |
| median TTFT | `295.5 ms` | `258.3 ms` |
| RepoBench-P static control cache-ready TTFT speedup | — | `1.089x` |
| RepoBench-P exact-line agreement | `10%` | `8%` |

#### 为什么不可合并

1. **Dense 基线不同**：Cohort A 的 Dense 是 `3/12`，Cohort B 的 Dense 是 `6/12`。同一 Dense 在同一任务集上不可能同时是 3 和 6，因此任务集必然不同。
2. **runner 不同**：Cohort A 来自 `run_v44_dense_sensitive_v40_campaign.py`（`STEP_LIMIT=32`），Cohort B 来自 `register_three_method_coding_benchmark.py` + `summarize_three_method_coding_benchmark.py` 链路。
3. **TTFT 口径不同**：A 是 "fixed-order host-resident median TTFT"（分支自述为 diagnostic），B 是 "median TTFT"，未声明 host residency 与顺序控制。
4. **结论方向相反**：A 中 V40 `4/12` **高于** Dense `3/12`；B 中 V40 `4/12` **低于** Dense `6/12`。把两者放进同一张表会制造虚假的一致性。

**强制要求**：任何后续引用都必须写成 "Cohort A (V44) …" 或 "Cohort B (three-method) …"，并且必须带 `external unverified claim` 标记。

### 6.3 `6/12 vs 4/12` 不是稳定的 loss 因果估计

分支的 summarizer **在读到作者机器上的外部输入后会生成**以下限制文字
（`v40:benchmark/multi_workflow/summarize_three_method_coding_benchmark.py:257`）：

> "Dense repeats failed. The 6/12 versus 4/12 result remains a valid single-run point estimate, but its -16.7 pp difference is not a stable causal estimate of lossy-reuse damage."

分支据此声称：两个"Dense 通过 / V40 失败"的任务，在 Dense 重复运行时
也失败。由于对应 repeat raw 不在本环境，这一运行结果仍是
`external unverified claim`。若外部输入属实，则 −16.7 pp 不能作为稳定
的 lossy-reuse 因果损伤估计。

`v40:benchmark/multi_workflow/test_summarize_three_method_coding_benchmark.py:72` 还断言：

```python
assert audit["decision"]["v40_beats_both_static_exact_line"] is False
```

该测试只证明：在测试构造的输入下，审计逻辑会把
"V40 在 static exact-line 上胜过两个基线"判为 **False**；它不验证作者
真实实验输入或输出。

### 6.4 V41 / V42 / V43 / V44 的证据等级逐代评估

> **证据级别的关键区分（R04，必须严格遵守）**：源码**只能**证明"这些 audit/runner 的 schema、判据与冻结常量是什么"，属于 `verified-code`；它**不能**证明"那次实际运行发生了什么"。因此下表把两者拆成两列：**运行结果**一律 `external unverified claim`，**源码可核实的部分**才标 `verified-code`。

| 代 | 性质 | 运行结果（**全部 `external unverified claim`**） | 源码可核实部分（`verified-code`） |
| --- | --- | --- | --- |
| **V40 motivation** | 离线选择器度量 | 是否跑过、得到什么 opportunity 分布 —— **未验证** | 硬编码依赖 `/home/gfy/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit/tokenizer.json`（`motivate_v40_grounded_observation_island.py:44-45`）与 `ARTIFACTS`（`:34`）；本环境该路径不存在（`verified-local`） |
| **V40A / V40A2 / V40A3** | 单任务 canary | 三次 canary 的实际结果、V40A2 是否真的超时 —— **未验证** | `audit_v40a2_timeout_failure.py:98` 定义了状态串 `"V40A2_INFRA_FAILURE_NO_ACCURACY_RESULT"`；三个 runner 的 `ARTIFACTS`/`MOTIVATION`/`*_FAILURE` 路径常量（各 `:21`、`:24-33`） |
| **V41** | 6 任务独立 campaign | "因 paired-source capacity deadlock 未完成"、两个任务失败 —— **未验证**（无 raw JSON） | `audit_v41_capacity_deadlock.py` 存在且冻结了 `FAILED = ("astropy__astropy-14995","psf__requests-1142")`（`:15-18`）、`MAX_NEW_TOKENS=2048`（`:21`）、`KV_CAPACITY_TOKENS=14482`（`:22`）；`test_audit_v41_capacity_deadlock.py` 断言这些常量 |
| **V42** | host residency infra canary | 该 canary 的实际结论 —— **未验证** | runner 的任务常量 `INSTANCE_ID = "astropy__astropy-14995"`、`ARMS = (V40, GENERAL, "dense")`；其设计意图为 infra canary（不含 accuracy 判据） |
| **V43** | 6 个新 Verified 任务 | "6 个任务全部耗尽 20-call 预算、产生空提交（0/6）" —— **未验证**（该说法只见于 `v40:KVFLOW.md:100-103` 的自述文字，无 raw JSON） | `audit_v43_call_budget_collapse.py` 冻结 `STEP_LIMIT=20`（`:33`）、`SHARED_CALLS=7`（`:34`）、`BRANCH_REQUEST_INDEX=8`（`:35`）与 6 任务元组；`test_audit_v43_call_budget_collapse.py` 断言之 |
| **V44** | 12 任务 development 运行 | 全部数值（`3/12`、`3/12`、`4/12`、TTFT `357.6/335.7/327.5 ms`、copied tokens 等）—— **未验证** | runner 冻结 `STEP_LIMIT=32`（`:40`）、12 任务元组（`:41-52`）、`SELECTION_SHA256=78663ee1…`（`:55-58`）、`DENSE_PASS_SENSITIVITY_MIN=2`（`:60`）；`test_v44_dense_sensitive_v40_campaign.py` 断言选择哈希与任务数 |
| **three-method cohort** | 12 任务 + RepoBench-P | 全部数值（`6/12`、`4/12`、`295.5→258.3 ms`、`1.089x`、`10%→8%`）—— **未验证** | summarizer 的聚合逻辑与稳定性声明（`summarize_three_method_coding_benchmark.py:257`）、`test_…:72` 断言 `v40_beats_both_static_exact_line is False` |

**判断（`derived`）**：按分支自述，V44 是其声称的完整 12-task
development 运行，其余各代分别被描述为 motivation、infra canary、
capacity deadlock 与 call-budget collapse。由于 raw 均不可得，这一演化
叙事只能作为作者声明；不能用"做了 V40–V44 五代实验"作为已独立验证的
成熟度证据。

**同时必须注意**：上一段本身也建立在**分支自述**之上。本环境没有任何一代的 raw 结果，因此"V41 deadlock"、"V40A2 timeout"、"V43 0/6"这些**运行层事实全部是 `external unverified claim`**；可被独立核实的只有 audit 脚本的 schema、判据与冻结常量。**不得**因为 audit 脚本存在就把对应的运行结论升级为 `verified-code`。

### 6.5 与 KVCOMM / CacheBlend 225-task 基线不可排名

`v40:KVFLOW.md:105-112`（`external unverified claim`）：

> - KVCOMM: 164/225, 8.55× cache-ready, 5.34× at N=4 including build;
> - CacheBlend: 169/225, 4.77× cache-ready, 1.22× at N=4 including build.

分支自己也写了 "These figures cannot be directly ranked against V44."

**必须遵守**：这两组数字与 V40 之间 **model / prompt / task order / generation limits / engine 全部不同**。`v40:docs/kvflow/COLLABORATOR_QUICKSTART_20260729.md:40-41` 补充："Native KVCOMM uses a different multi-agent prompt topology."。任何排名表述都被禁止（见 §12）。

### 6.6 本项目自身的先验证据（对 V40 收益预期的直接约束）

这一节是**本项目已有的、已通过双模型审查的 GPU 证据**，它对 V40 的 TTFT 收益预期构成强约束：

`docs:research/phase_reports/PHASE7_INTEGRATED_EVALUATION_REPORT.md` §1.3(2)：在预注册 primary `chunk = 4096` 下，R0（即 V40 的同族数据面）的 paired request-path median **全部低于 1.0**：

| body | rho | request-path median | N8 full-setup | N8 incremental |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1.5 | `0.7723084788319753` | `0.6086457910880934` | `0.6838342404592734` |
| 1024 | 2.0 | `0.7750652993475325` | `0.6100955039216343` | `0.6854804712027397` |
| 2048 | 1.5 | `0.9333835627802327` | `0.6397908630435616` | `0.7607318873368634` |
| 2048 | 2.0 | `0.9361732730155323` | `0.6418893640462963` | `0.7640829718628741` |

全部未达预注册 MDE（`max(5%, 2×sample_sd)`，`mde_fraction=0.05`），触发停止规则 `ES-R0-MDE`，Phase7 disposition 为 `r0_mechanism = NEGATIVE`。

同时 §1.3(3) 记录了 chunk 混淆：同一 body2048/rho2/S0 配置在 **chunk1024** 下 request-path median = `1.7370152775837997`，但 artifact 显式标注 `headline=false`、`interpretation="chunk-coupled sensitivity diagnostic; not a mechanism-intrinsic headline"`。

**推论（`derived`，且是本报告最重要的可行性判断之一）**：
V40 报告的 TTFT 改善（Cohort A `357.6 → 327.5 ms`，即 `1.092x`；Cohort B `295.5 → 258.3 ms`，即 `1.144x`；RepoBench `1.089x`）落在与本项目 chunk1024 敏感性诊断相同的量级区间，而**远低于**任何机制性 headline 所需的幅度。在评估其真实性之前，必须先排除 **chunk / max-prefill 配置耦合**这一已知混淆源。分支现有材料中**没有**披露 V40 实验的 `chunked_prefill_size` / `max_prefill_tokens`。这是必须补齐的第一项元数据。

---

## 7. Docker 验证结果（`verified-local`）

### 7.1 环境

固定 image：`ghcr.io/ccdd2023/sglang@sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781`
（本机 `docker images --digests` 确认存在；容器内 `CUDA 12.9.1`；无 GPU 驱动挂载，全部为 CPU 测试）。

挂载方式：分支 worktree 以 **只读** 方式挂载到 `/w`，`--user 1000:1000`，`--rm`，不向被审查仓库写入任何内容。

### 7.2 通过的测试面

| 套件 | 选择器 | 结果 |
| --- | --- | --- |
| policy / selector / KVCOMM core / radix | `coding_aware/test_policy.py` + `test_coding_reuse_policy.py` + `kvcomm/test_core.py` + `kvcomm/test_radix_backend.py` | **`66 passed`** |
| bridge adapter | `benchmark/multi_workflow/test_bridge_reuse_litellm_model.py` | **`8 passed`** |
| branch scope 单元 | `tools/test_check_kvflow_branch_scope.py` | **`3 passed`** |
| branch scope 实跑 | `check_kvflow_branch_scope.py --role coding --base c16bfbb8e` | **`coding branch scope: OK`（rc=0）** |
| self-contained V40–V44 schema/audit | 见下方精确 selector（11 文件） | **`12 passed`** |
| three-method + RepoBench summarizer | `test_summarize_three_method_*` + `test_run_v40_repobench_control` + `test_summarize_kvcomm_repobench` + `test_register_three_method_*` + `test_prepare_three_method_*` | **`8 passed`** |
| kvcomm_exact runtime | `python/sglang/srt/mem_cache/test_kvcomm_exact.py` | **`23 passed`** |

**self-contained V40–V44 schema/audit 的精确 selector（统一为 `12 passed`）**：

```bash
python -m pytest -q -p no:cacheprovider \
  benchmark/multi_workflow/test_v40a_grounded_observation_canary.py \
  benchmark/multi_workflow/test_audit_v40a2_timeout_failure.py \
  benchmark/multi_workflow/test_audit_v39_v38_equivalence.py \
  benchmark/multi_workflow/test_v41_v40_independent_campaign.py \
  benchmark/multi_workflow/test_audit_v41_capacity_deadlock.py \
  benchmark/multi_workflow/test_v42_host_residency_infra_canary.py \
  benchmark/multi_workflow/test_v43_new_verified_v40_campaign.py \
  benchmark/multi_workflow/test_audit_v43_call_budget_collapse.py \
  benchmark/multi_workflow/test_v44_dense_sensitive_v40_campaign.py \
  benchmark/multi_workflow/test_summarize_v44_schema_compat.py \
  benchmark/multi_workflow/test_motivate_v40_grounded_observation_island.py
# -> 12 passed
```

（此前版本因遗漏 `test_audit_v39_v38_equivalence.py` 而记为 `11 passed`，现统一为 **`12 passed`**，与审查请求一致。）

**本报告的 Docker 计数汇总（全部 `verified-local`）**：`66` / `8` / `3` / `12` / `8` / `23` 通过，另有 **2 个外部依赖导致的 failure**（§7.3）。

复现命令（示例，policy/selector/core/radix 面）：

```bash
cd /home/chris/Workspaces/kvcache-research/worktrees/coding-aware-v40-prefetch
docker run --rm --user 1000:1000 \
  -v "$PWD":/w:ro -w /w -e PYTHONPATH=/w/python -e HOME=/tmp \
  ghcr.io/ccdd2023/sglang@sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781 \
  python -m pytest -q -p no:cacheprovider \
    python/sglang/srt/mem_cache/coding_aware/test_policy.py \
    benchmark/multi_workflow/test_coding_reuse_policy.py \
    python/sglang/srt/mem_cache/kvcomm/test_core.py \
    python/sglang/srt/mem_cache/kvcomm/test_radix_backend.py
# -> 66 passed
```

### 7.3 失败的测试（B-06）

```text
FAILED benchmark/multi_workflow/test_v40a2_grounded_observation_canary.py::test_v40a2_selection_is_outcome_independent_and_source_rich
FAILED benchmark/multi_workflow/test_v40a3_short_grounded_observation_canary.py::test_v40a3_selection_is_short_and_outcome_independent
2 failed, 9 passed
```

失败原因（容器内实际 traceback）：

```text
FileNotFoundError: [Errno 2] No such file or directory:
  '/home/gfy/CodeMAS_Project/kvflow-artifacts/
   impactkv_v40_grounded_observation_motivation_20260728/V40_MOTIVATION_RESULT.json'
```

对应硬编码（`verified-code`）：
`v40:benchmark/multi_workflow/run_v40a2_grounded_observation_canary.py:21` `ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")`、`:24-28` `MOTIVATION = ARTIFACTS / "impactkv_v40_grounded_observation_motivation_20260728" / "V40_MOTIVATION_RESULT.json"`；
`v40:benchmark/multi_workflow/run_v40a3_short_grounded_observation_canary.py:21`、`:24-28` 同构。

这两个测试断言的是**选择结果的冻结值**：

```python
assert v40a2._selected_task() == ("sphinx-doc__sphinx-9230", 14)
assert v40a3._selected_task() == ("pytest-dev__pytest-7982", 13, 6)
```

即 selection 的 provenance 完全依赖一个不在 Git 中的文件。**在作者机器之外，V40 的任务选择不可复核。**

### 7.4 依赖不自包含（B-06 的第二面）

`benchmark/multi_workflow/` 的 runner/audit 测试**传递依赖** `litellm` 与 `minisweagent`：

```text
benchmark/multi_workflow/bridge_reuse_litellm_model.py:24: import litellm
      -> ModuleNotFoundError: No module named 'litellm'
benchmark/multi_workflow/context_bounded_litellm_model.py:12:
      from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
      -> ModuleNotFoundError: No module named 'minisweagent'
```

**基础镜像既有版本（`verified-local`）**：`tokenizers == 0.22.2`（无需额外安装）。

基础镜像自身的 `python -m pip check` 已有 5 条基线不一致：

```text
sglang requires transformers==4.57.1, installed 5.12.1
torch requires nvidia-cudnn-cu12==9.10.2.21, installed 9.16.0.29
torch requires nvidia-nccl-cu12==2.27.5, installed 2.28.3
datasets requires fsspec<=2026.2.0, installed 2026.6.0
compressed-tensors requires transformers<5.0.0, installed 5.12.1
```

因此对派生 layer 要求 "`pip check` 零输出" 在不先重建基础镜像时不可达。

**安装方式（`verified-local`，更正此前表述）**：使用 `pip install --target <dir>` + `PYTHONPATH` 即可，**不需要** `--break-system-packages`：

```bash
pip install --target /tmp/deps "mini-swe-agent==2.3.0" litellm   # rc=0，无 externally-managed 错误
export PYTHONPATH=/w:/w/python:/tmp/deps
```

（此前版本称"必须 `--break-system-packages`"是**不准确**的；那只是在 `--user` 安装路径下才会触发 PEP 668 拒绝。）

**未锁版本时的 resolver 告警（容器内原文，`verified-local`）**：

```text
ERROR: pip's dependency resolver does not currently take into account all the
packages that are installed. ... the following dependency conflicts.
transformers 5.12.1 requires tokenizers<=0.23.0,>=0.22.0, but you have tokenizers 0.23.1
sglang 0.0.0.dev1+g7a1ca5380 requires openai==2.6.1, but you have openai 2.50.0
sglang 0.0.0.dev1+g7a1ca5380 requires transformers==4.57.1, but you have transformers 5.12.1
torch 2.9.1+cu129 requires nvidia-cudnn-cu12==9.10.2.21, but you have 9.16.0.29
torch 2.9.1+cu129 requires nvidia-nccl-cu12==2.27.5, but you have 2.28.3
```

**如何解读这些告警（R09，必须按此表述）**：
- 这些告警是**在没有版本锁的情况下装最新版**产生的，它证明的是"**该实验环境没有被锁定**"；
- 它**不**证明"依赖冲突不可避免"。把 `mini-swe-agent` / `litellm` / `openai` / `transformers` / `tokenizers` 钉到与基础镜像兼容的版本后，冲突可以消除；
- 因此这是一个**可修复的工程缺陷**，修复要求见 §9.18.1：提供
  `requirements.lock`（含 hash）并构建专用 Docker layer。若沿用当前基础
  镜像，验收采用**baseline-delta**：不新增任何 `pip check` 冲突，尤其不得
  新增 `litellm`/`mini-swe-agent`/`openai`/`tokenizers` 相关项；若重建新的
  clean base image，则可把 `pip check` 零输出作为更强验收。

**结论（`derived`，修正）**：分支的实验面**当前不是自包含的**，原因是 (a) 依赖未锁且未随镜像分发、(b) 一批结果 artifact 只存在于 `/home/gfy/**` 而不在 Git 中。其中 (a) 可通过依赖锁与专用 layer 修复；(b) 需要外部导入（§13.6）。

### 7.5 prefetch composition 无法独立复现（`external unverified claim`）

三层证据（详见 §2.5 与 §2.5.1）：

1. **ref 层**：`git rev-parse --verify` 对 `research/prefetch-p8-async-20260722`、`0ab4fc942`、`e44ce40dc` 全部返回 `fatal: Needed a single revision`；
2. **对象库层**：`git cat-file -t 0ab4fc942` / `git cat-file -t e44ce40dc` 返回 `fatal: Not a valid object name` —— 这两个 commit **对象本身不存在**；
3. **远端层**：在线 `git ls-remote origin` 全量枚举中，既没有 `integration/coding-aware-prefetch-v2`，也没有 `research/prefetch-p8-async-20260722`；只有 stale 的 `integration/coding-aware-prefetch @ d4a7ec132` 与 `research/prefetch @ fa86f8f16`。

同时，被审查分支中 `git ls-files | grep -i prefetch` 为空，`python/sglang/srt/mem_cache/kvcomm_prefetch/` 与 `python/sglang/srt/mem_cache/kvflow_integration/` 目录均不存在。

**因此**：`v40:KVFLOW.md:132` 与 `v40:docs/kvflow/ARCHITECTURE.md:143-146` 所称的 **113 composition tests**，以及 `v40:docs/kvflow/COLLABORATOR_QUICKSTART_20260729.md:110-116` 给出的复现命令，在本环境（以及任何只能访问当前 `origin` 的环境）**均不可复现**。该结果必须标记为 `external unverified claim`，**不得**用作 composition 可行性的证据，也**不得**作为 Gate 5 的 entry 条件。

### 7.6 Docker 验证小结

| 判定 | 依据 |
| --- | --- |
| 数据面与身份/lifecycle 单元测试是**健康**的 | 66 + 23 + 8 全绿 |
| 策略面的**测试覆盖不足**（未覆盖 §5.1/§5.2 的对抗输入） | 66 全绿但 §5.1/§5.2 的 P0 仍然存在 |
| **历史 end-to-end 实验结果不可复现**（**不是**"任何实验都不可复现"） | 2 failed（硬编码 `/home/gfy/**`）+ 依赖未锁 + raw artifact 不在 Git。**focused / unit 测试面则完全可复现**：本报告已在固定 image 内实测 `66/8/3/12/8/23` 全部通过 |
| composition 面**不可复现** | 三个 ref 全部不可获得 |

---

## 8. 可行性评估（用户问题 3 的第一半）

### 8.1 机制可行性

| 断言 | 判定 | 理由 |
| --- | --- | --- |
| "token 相同的 tool observation 可以被搬到下一请求并保持机械合法" | **成立** | `xs:` 与 `v40:` 的 transfer 层都做了 token slice 相等 + 边界 + 完整覆盖检查；单元测试全绿 |
| "这样搬运是无损的" | **不成立** | 左上下文不同 ⇒ 各层 hidden state 不同 ⇒ K/V 都不同；RoPE 只修位置 |
| "选择器能保证被搬运的段落对应当前 repo 版本" | **当前不成立** | §5.1、§5.2 已本地复现反例 |
| "该方法在 chunk4096 下能带来 TTFT 收益" | **未证明，且先验不利** | 本项目 Phase7 对同族 R0 的判定为 `NEGATIVE`（§6.6） |
| "该方法不会损害任务准确率" | **未证明** | Cohort A/B 方向相反；分支自身 summarizer 拒绝把差值当因果估计 |

### 8.2 工程可行性（在本项目底座上重实现）

| 组件 | 现状 | 增量工作量（估计，标注为 estimate） |
| --- | --- | --- |
| copy V + RoPE K 执行器（backend） | `xs:approx_kv/radix_backend.py:196` 已具备并通过 GPU test（`test_approx_kv_cuda.py`） | 0 |
| **middle-span staging controller + 状态机 + scheduler 接线** | **缺口**：`xs:approx_kv/runtime.py:441-442,449-467` 只支持从 `exact_length` 连续开始的 span，中部岛会被判 `prefix_gap` 整体 dense（§10.3.1） | **`2–3 人周`（estimate）+ 一轮独立 review** |
| 段身份 / generation / lease | `xs:approx_kv/store.py:99-413` 已具备 | 0 |
| cross-store 预算 / 驱逐 / 降级 | `xs:cross_store/{allocator,budget,policy,object_graph}.py` 已具备并含 Phase 6 的**六项**正确性修复（§10.2） | 0 |
| fallback 分类学与遥测 | `xs:approx_kv/manager.py` + `observability/metrics_collector.py:1951-2110` 已具备 14 个 fallback reason | 小改：新增 `coding_version_stale` 等 reason |
| **结构化 selector** | 无 | `2–3 人周`（estimate），主要成本在 tool wrapper 的 read/write path 采集 |
| sidecar / adapter | `v40:` 有一份可借鉴实现（version-3） | `1–1.5 人周`（estimate），需去掉硬编码路径并加入 provenance |
| 四模式 / 四本账 遥测接线 | Phase7 已有 request_path / target_only / full_lifecycle / speedup_N 四本账 | `0.5 人周`（estimate） |

**总计（estimate）**：把 C40 的最小 payload 在 `xs:` 上重实现约 **6–9 人周**，其中约一半是 selector 的结构化 provenance，另一半是 middle-span execution seam（§10.3.1）。copy/RoPE 数学本身为 0。

### 8.3 收益可行性（最关键的负面判断）

Phase7 已经在**同一台 SM75 机器、同一 image、同一模型**下证明：R0 数据面在 primary chunk4096 下 request-path median 为 `0.772–0.936`（即**变慢**），且摊销到 N=8 仍为 `0.609–0.642`。V40 的数据面与 R0 等价（§4.2）。

因此，V40 若要展示正收益，只可能来自以下三条之一，且必须**逐条隔离验证**：

1. **不同的 chunk / max-prefill 配置**（已知会产生 `1.737x` 量级的假性收益，Phase7 §1.3(3) 明确标为非 headline）；
2. **不同的模型规模**（V40 用 Qwen3-Coder-30B-A3B-AWQ / Qwen2.5-Coder-3B，Phase7 用 Qwen3-0.6B；更大模型的 prefill 成本占比更高，理论上更有利）；
3. **不同的 body 长度分布**（V40 的岛为 128–4096 token，Phase7 测的是 body 1024/2048）。

#### 8.3.1 关键推论：C40 **不得默认**性能转正（强制条款）

**前提（`verified`）**：在本项目 Phase 7 的固定环境（image `sha256:0be6e16e…`、`Qwen/Qwen3-0.6B @ c1899de2…`、SM75 / RTX 2080 SUPER 8 GiB、primary `chunk = max-prefill = 4096`）中，**R0 已被判定为 `NEGATIVE`**：四个 A8 restart-0 setting 的 paired request-path median 全部 `< 1.0`（`0.7723 / 0.7751 / 0.9334 / 0.9362`），N8 full-setup 仍为 `0.6086–0.6419`，未达预注册 MDE，触发 `ES-R0-MDE` 并跳过 8 个 supplement starts（`docs:research/phase_reports/PHASE7_INTEGRATED_EVALUATION_REPORT.md` §1.3(2)、§4.6）。

**C40 与该结论的关系（`derived`，必须写入任何 C40 计划）**：

> C40 **使用与 R0 完全相同的数据面**（copy V + K RoPE delta，§4.2 已逐项对照证明数学等价），它相对 R0 增加的**只有一个 selector**（`G40` 准入 gate）。selector 只能决定"哪些请求进入复用路径"，**不能改变进入之后每个 token 的搬运成本**。
>
> 因此：**C40 在同一环境下不得默认性能转正。** 任何"C40 比 R0 快"的观察，在排除以下三个来源之前，都不能归因于方法本身：
> 1. **selection 效应（真实收益）**：selector 选中的 span 恰好更长 / 位置更靠前 / 更容易命中，使**单位 span 的搬运收益更高**；
> 2. **幸存者偏差（survivorship bias，虚假收益）**：selector 把所有"复用会变慢"的请求过滤掉了，剩下的样本天然好看 —— 这**不是**方法收益，而是**样本选择**；
> 3. **配置耦合（已知混淆）**：chunk / max-prefill 差异（Phase 7 已证 `chunk1024` 可产生 `1.737x` 的假性 headline，且被显式标注 `headline=false`）。

**合法 estimand 定义（R06，取代此前的 median 合成与 Amdahl 硬检查）**

所有量必须以**每请求配对时间之和**（paired sums / ratio-of-sums）定义，**不得**用"median speedup"再做合成，也**不得**用 `1/((1-f)+f/s)` 这类 request-fraction × median-speedup 的公式当作可核对的硬约束（该式要求同质性与时间比例可乘，两者在真实 workload 中都不成立）。

| 记号 | 定义（全部为**同一批请求上的配对总时间比**） |
| --- | --- |
| **E_cond** | `Σ_{i∈Elig} T_dense(i)  /  Σ_{i∈Elig} T_C40(i)`，其中 `Elig` 是**预先冻结**的 eligible 请求集合，`T` 取 `request_path`（或 `target_only`，需分别报告）。这是 ratio-of-sums，不是 median 的比 |
| **E_work** | `Σ_{i∈All} T_dense-full(i) / Σ_{i∈All} T_C40-full(i)`；Dense 与 C40 都必须各自**真实执行完整、有状态的请求流**，保留相同顺序、cache pressure、source 生产/消费和 ineligible dense 路径。不得把 eligible-only 结果与另一条 Dense trace 事后拼接 |
| **time-weighted coverage** `w` | `Σ_{i∈Elig} T_dense(i)  /  Σ_{i∈All} T_dense(i)`。注意分子分母**都取 dense 基线时间**，因此 `w` 与 C40 是否变快无关，是纯粹的"可作用时间占比" |
| **eligibility rate（计数口径）** `r` | `|Elig| / |All|`，附 Wilson 95% CI 与 skip_reason 直方图。**仅作描述**，不参与任何加速度换算 |
| **C_selector** | selector 自身的 CPU 开销（含路径提取、tokenize、唯一性搜索、sidecar I/O），**在 ineligible 请求上也会发生**，必须单独测量并计入 `E_work` 的分母 |

三者关系只作为**方向性说明**而非硬检查：`E_work` 受 `w` 上界约束（`w` 小则 `E_work` 必然接近 1），但不要求满足任何闭式等式。

**counterfactual 设计（R06 核心修正）**

机制归因实验必须先冻结 trajectory 与 selector plan，再让各臂**完整执行同一
请求流**，避免"不同臂选出不同请求"和事后拼接：

```text
Step 1  冻结一条可重放 trajectory：每个请求的完整 prompt、工具 observation、
        顺序、source-producing event 与预期 repository generation。
Step 2  对该冻结 trajectory 离线运行 selector，冻结:
          Elig 集合 + 每个 i 的 span 元组
          (source_start, target_start, length, rope_delta, source_key, generation)
        冻结清单写入 manifest 并做 sha256 绑定。
Step 3  三臂分别从clean reset开始，**完整执行All请求**:
          臂 A  Dense-full      : 所有请求dense，仍执行完整source lifecycle
          臂 B  C40-full        : eligible请求执行C40；ineligible请求在同臂dense
          臂 C  SpanR0-full     : eligible请求回放同span R0；ineligible请求同臂dense
        三臂都保留请求顺序、cache pressure、source register/release与下一请求状态。
Step 4  E_work直接由三条完整trace的paired total elapsed计算。E_cond只是从
        同一完整trace中切出冻结的Elig请求做次级分析，不产生合成workload。
```

**B / C 差分的正确含义（不得称为"selection 收益"）**

臂 B 与臂 C 跑的是**同一完整请求流**，且 eligible 请求使用同一批 span；
唯一设计差别是 B 在线执行 selector 判定与控制路径，C 回放冻结结果。因此：

> `E_cond(C) / E_cond(B)` **只度量 selector 判定与控制路径的开销**（`C_selector` + 状态机分支成本），**不是** selection 带来的收益，**也不是** survivorship 的度量。报告中必须命名为 **`overhead_selector_control`**，禁止写成 "selection gain"。

**survivorship 不通过差分估计**

此前版本用"臂 C（eligible 子集）vs 臂 D（全体）"来定义 `Δ_survivorship` 是无效的：两者请求集合不同，差值同时混入了请求难度、长度与顺序，无法归因。**取消该差分。** survivorship 的正确处理方式是**结构性的**：

- 用 `w`（time-weighted coverage）显式披露"C40 只能作用于 dense 基线时间的百分之几"；
- 所有 conditional 结论**必须**与 `w` 一起出现；
- 全 workload 结论**只**看由完整 C40-full trace 实测的 `E_work`；它真实
  包含 ineligible dense、selector overhead、cache pressure、source
  lifecycle 和顺序效应，不允许事后合成。

**headline 判据（重写）**

| 结论级别 | 条件 |
| --- | --- |
| 允许 workload-level headline | `E_work` 的 restart-cluster bootstrap 95% CI **下界 > 1 + MDE**，且 `w`、`r`、`C_selector` 同时披露，且 chunk 配置为 primary |
| 只允许 conditional 表述 | `E_cond` 显著 > 1 但 `E_work` CI 覆盖 1 —— 必须写成"仅在占 dense 基线时间 `w` 的 eligible 子集上观察到 `E_cond`；全 workload 未观察到显著改善" |
| 判 `NEGATIVE` | `E_cond` 与 `E_work` 的 CI 均覆盖或低于 1（与 Phase 7 R0 结论一致） |
| 判 `INVALID`（不进入 disposition） | 没有完整执行 Dense-full/C40-full trace、缺 `E_work`/`w`/`C_selector`，或未执行臂 C |

**汇报模板（强制）**：

```text
C40 = G40 × R0 | chunk=<...> | model=<id@rev> | image=<digest> | restarts=<n>
  r  eligibility rate (count)      : e/N = ___   Wilson95 [___, ___]
  w  time-weighted coverage        : ___         (dense-baseline time share)
  C_selector overhead              : ___ ms/req  (measured on ALL requests)
  E_cond (ratio-of-sums, eligible) : ___         cluster-bootstrap 95% CI [___, ___]
  E_work (ratio-of-sums, ALL)      : ___         cluster-bootstrap 95% CI [___, ___]
  arm C span-matched R0            : ___   -> overhead_selector_control = ___
  Phase7 R0 reference (same image/model, chunk4096) : 0.772-0.936  (NEGATIVE)
```

**可行性结论（`derived`，措辞已按 R08/R11 收紧）**：
- Phase 7 对 R0 的 `NEGATIVE` 判定构成一个**不利先验**（prior），**不是**对 C40 的既定结论。C40 在同硬件同模型下是否也为负，**必须由 pilot 检验**（§9.19 Stage 1），不得预先断言"必负"；
- 因此当前最值得投入的是 **正确性 + 覆盖率 + 质量损伤上界**（Gate 1–3、Gate 6）以及 **配置耦合的隔离**（Gate 3 的 chunk 因子），而不是速度 headline；
- 更大模型 / 更长 span / 更大 KV footprint 的硬件（如 RTX PRO 6000、H100）是一条**条件性扩展路径**：只有当 pilot 显示收益随 span 长度或模型规模单调上升、且本机 pilot 未被 `E_work` CI 排除时才值得申请，**不是**"必须换硬件"的结论，也不是发布任何 claim 的前提。

### 8.4 三个可证伪预测（预注册用）

| ID | 预测 | 证伪条件 |
| --- | --- | --- |
| PR-C40-1 | 在结构化 selector 修复后，C40 在 SWE-bench 类 trajectory 上的 eligible 比例（`r` 与 `w` 同时）会**显著下降**（因为大量 mixed read-write group 将被正确排除） | 修复前后 eligible 率差异不显著（配对 bootstrap，α=0.05） |
| PR-C40-2 | 在固定 chunk4096 + Qwen3-0.6B 下，C40 的 **`E_work`**（完整 workload paired ratio-of-sums）cluster-bootstrap CI **不会**整体高于 `1 + MDE` | `E_work` 的 CI 下界 > `1 + MDE`（MDE 须由 pilot 先行冻结）。**注**：这是把不利先验形式化为可证伪命题，不是断言 C40 必负 |
| PR-C40-3 | same-context canary 下，`max|ΔK|`/`max|ΔV|`/`max|Δlogit|` 全部落在预冻结容差内且贪心输出一致 | same-context 超出容差 → 数据面实现缺陷。（**不对 cross-context 的 top-1 一致率做方向性预测**；cross-context 的"确实跨上下文"由 prefix hash 与 `rope_delta≠0` 断言保证，不由 top-1 反推） |

---

## 9. 详细测试与实验设计（用户问题 3 的第二半）

### 9.0 全局统计与口径约束（先于所有实验）

| 约束 | 内容 |
| --- | --- |
| **独立复制单元** | **server restart**。同一 server 内的 formal request **不是**独立样本，禁止当作独立样本做统计；所有 CI 用 **restart-level cluster bootstrap** |
| **任务准确率单位** | **task**，不是 request。一个 task 的多个请求只贡献一个 pass/fail；配对比较用 **McNemar**，且必须**先估 discordant rate** 再定样本量 |
| **四本账 latency ledger** | `target_only` / `request_path`（= `seed_head_ms + target_only_ms`，**预注册 MDE 指标**）/ `full_lifecycle` / `speedup_N`（N ∈ {1,2,4,8}，**必须实测，禁止插值外推**） |
| **estimand** | 一律 **ratio-of-sums**（`E_cond` / `E_work`，§8.3.1），禁止 median-of-ratios 合成 |
| **两阶段统计计划** | **restart-0 只做 engineering screening**（可达性、无 capacity error、遥测完整），**不得**用它的未知 `sample_sd` 判定最终 `NEGATIVE`；必须先跑 **≥ 2 个独立 restart 的 pilot** 估方差与 discordance，再**冻结 MDE 与 confirmatory plan**（§9.19） |
| **MDE** | 由 pilot 方差估计后冻结；形式为 `max(5%, 2 × sample_sd_pilot)`。**冻结前不得**引用任何 MDE 判定 |
| **配对方式** | 同一 `(body, rho, restart)` 下相邻 launch block；不相邻只能称 `seed_matched_non_adjacent_restart_comparison` |
| **p95** | `ratio_of_marginal_p95s` **不是**配对统计量，必须标 `p95_pairing="nonpaired"` |
| **必须披露的配置** | `chunked_prefill_size`、`max_prefill_tokens`、`page_size`、`tp/pp`、eviction policy、HiCache on/off、model+tokenizer revision、image digest |

### 9.1 G1 — Selector 对抗矩阵（CPU，最高优先级，Gate 0 的核心）

**目的**：把 §5.1 / §5.2 变成回归测试，并把"写检测"从命令正则升级为结构化判定。

#### 9.1.1 写工具对抗矩阵（每格必须产生 `invalidated=1`）

| 类别 | 用例（≥ 每类 5 个变体） |
| --- | --- |
| shell 重定向 | `cat > f`、`cat >> f`、`echo x > f`、`printf ... > f`、`tee f`、`tee -a f`、`cat <<EOF > f` |
| in-place 编辑 | `sed -i`、`sed --in-place`、`perl -pi -e`、`perl -i.bak -pe`、`ruby -i -pe`、`ed`、`ex -sc` |
| 截断/写块 | `truncate -s 0 f`、`dd of=f`、`dd of=f conv=notrunc`、`: > f`、`> f` |
| 复制/移动/链接 | `cp`、`cp -a`、`install -m`、`mv`、`git mv`、`rsync`、`ln -sf`、`ln -f` |
| 打补丁 | `patch -p1`、`git apply`、`git am`、`git checkout --`、`git restore`、`git stash pop`、`git revert` |
| 语言内写 | `python -c "open(...,'w')"`、`python - <<PY`、`pathlib.write_text`、`json.dump(fp)`、`shutil.copy`、`os.rename`、`np.save` |
| 构建副作用 | `make`、`python setup.py build_ext --inplace`、`pip install -e .`、`pytest --lf` 写 cache |
| 归档解包 | `tar -xf`、`unzip -o`、`git clone` 覆盖 |
| 编码/引用绕过 | 反引号、`$( )`、`eval`、变量拼路径、base64 解码后写、多行续行 `\` |
| 路径变体 | `/testbed/pkg/a.py`、`./pkg/a.py`、`pkg/a.py`、`../repo/pkg/a.py`、软链接指向、大小写、Unicode 路径 |

#### 9.1.2 混合 group 矩阵（每格必须 `is_successful_readonly_evidence=False`）

至少覆盖：`read; write`、`write; read`、`read && write`、`read || write`、`read | tee f`、`read; (write)`、`read; bash -c "write"`、`for f in *; do write; done`、以及 §5.2 的三个已复现用例。

#### 9.1.3 Property 测试（Hypothesis 或等价）

- **P1**：对任意随机命令串，若结构化 `write_paths ∩ source_paths ≠ ∅`，则 `invalidated == 1`。
- **P2**：对任意随机命令串，若 `unknown_effect == True`，则该 group **不可**成为 source。
- **P3**：单调性——向 trajectory 尾部追加任意 group，eligible 集合只能收缩不能扩张（保守性）。
- **P4**：路径规范化幂等——`normalize(normalize(p)) == normalize(p)`，且 `/testbed/x` 与 `./x` 与 `x` 归一到同一键。

#### 9.1.4 差分测试（**必须在 Docker 内的临时可写副本上执行**）

用真实 SWE-bench trajectory（或合成 agent trace）**真正执行**命令，用 `git status --porcelain=v2 -z` 采集 ground-truth `write_paths`，与 selector 的判定做差分。**FN（漏检写）必须为 0**；FP 可以有，因为 fail-closed 方向安全。

**执行约束（R12，硬性）**：
- 被测仓库以 `:ro` 挂载；测试开始时在**容器内**复制到临时可写目录（`--tmpfs /scratch` 或容器可写层）；
- 所有 mutation 命令只作用于该副本；
- **绝不**在宿主 worktree 上执行任何写操作，也不向宿主目录挂载可写卷；
- 容器 `--rm` 退出即销毁，差分结果经 stdout 或显式 artifact 卷输出。

```bash
docker run --rm --user 1000:1000 \
  -v "$PWD":/w:ro --tmpfs /scratch:rw,size=2g -w /scratch -e HOME=/tmp \
  ghcr.io/ccdd2023/sglang@sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781 \
  bash -c 'cp -a /w /scratch/repo && cd /scratch/repo && python -m selector_differential --out -'
```

**成功标准**：对抗矩阵 100% 通过；差分测试 FN = 0；property 测试 1000 例无反例。
**失败标准**：任一 FN > 0 → Gate 0 不通过。

### 9.2 G2 — 身份与指纹测试（CPU）

覆盖：`token_hash` / `token_count` 不匹配拒绝；同 key 重注册产生新 `generation`；stale handle 不能 pin/load/release；model id、tokenizer revision、chat template、dtype、layout、page_size、TP rank 全部进入 fingerprint；跨 fingerprint 复用必须拒绝。

**新增（相对 `v40:`）**：`source_content_sha256` 与 `worktree_generation` 必须进入 key；缺失即拒绝（fail closed）。

### 9.3 G3 — Dense / copy 覆盖测试（CPU）

对每个 plan 断言：`copied_spans` ∪ `dense_ranges` == `[0, len(target))`，无重叠、无空洞（`require_full_coverage=True`）。
边界用例：岛在位置 1；岛结束于 `len-1`；岛长度 = `reuse_min_tokens`；岛长度 = `reuse_copy_cap`；岛长度 = `cap+1`（必须截断且仍严格中部）；多次唯一匹配（必须拒绝）；零次匹配（必须拒绝）。

### 9.4 G4 — K/V + RoPE 张量测试（GPU，Gate 2）

1. **same-context 恒等**：source 与 target 的左上下文**完全相同**、位置也相同（`rope_delta = 0`）→ 判据为**张量容差 + logit 容差 + 输出匹配**三者：`max|ΔK|`、`max|ΔV|` 在 backend dtype 容差内；`max|Δlogit|` 在预先冻结的 logit 容差内；贪心解码输出串完全一致。**不以 top-1 一致率作为唯一判据**。
2. **纯位移**：同左上下文、不同绝对位置（`rope_delta ≠ 0`，正负都测）→ 复用后 K 必须等于 dense 在新位置的 K（容差内）；V 必须 bitwise 相等。
3. **全层全 head 覆盖**：断言 `rotated_k_tokens == copied_k_tokens == span_len`，且逐层校验（防止只旋转了部分层）。
4. **rotary_dim < head_dim** 情形：只有前 `rotary_dim` 维被改，其余维必须 bitwise 不变。
5. **corruption canary**：故意注入错误 `rope_delta`（±1）→ 必须被 §9.5 的 same-context canary 检出。

### 9.5 G5 — same-context corruption canary（GPU，常驻回归）

构造一个"source 与 target 左上下文严格相同"的请求对。此时复用**在数学上应当无损**。任何 logit 差异都说明实现缺陷（而非方法固有损失）。该 canary 必须在每次 GPU 实验前作为 smoke 运行。

**成功标准**：`max|ΔK|`、`max|ΔV|` 在 dtype 容差内；`max|Δlogit|` 低于预冻结 logit 容差；贪心输出串逐字符一致。
**失败标准**：任一超出容差 → 停止本轮全部 GPU 实验。
**注意**：same-context canary 的作用是**发现实现缺陷**，不是证明方法无损；它的通过**不构成**任何 cross-context 结论。

### 9.6 G6 — cross-context 输出 / logit / 任务质量（GPU，Gate 3 + Gate 6）

三个层级，**从便宜到昂贵**：

| 层级 | 指标 | 样本 | 说明 |
| --- | --- | --- | --- |
| L1 token 级 | 首 token top-1 一致率、top-5 一致率、`KL(dense ‖ reuse)`、`max|Δlogit|` | 每配置 ≥ 200 个复用点 | 最灵敏、最便宜，先跑 |
| L2 序列级 | 生成前 64 token 的 exact-prefix 长度、编辑距离、命令一致率 | 每配置 ≥ 100 个请求 | 对应 RepoBench-P exact-line |
| L3 任务级 | SWE-bench Verified official resolved（**以 task 为单位**） | ≥ 40 task/臂（见 §9.15 功效） | 最贵，最后跑 |

**关键设计要求**：
1. L1/L2 必须在**同一 server、同一 seed、同一任务顺序**下用 paired 设计采集 dense 与 reuse 两条曲线；L3 必须每个 task 至少 `3` 次重复以估计 run-to-run 方差（three-method 的教训：Dense 自身不稳定）。
2. **cross-context 必须先证明"上下文确实不同"**，否则整组测量无意义。每个复用点必须记录并断言：
   - `source_prefix_token_hash != target_prefix_token_hash`（左上下文确实不同）；
   - `rope_delta = target_start - source_start != 0`（位置确实不同）；
   - 二者任一不成立 → 该样本归入 same-context 组，不得计入 cross-context 统计。
3. **不预设 top-1 一致率必须 < 100%**。cross-context 下 `top-1 = 100%` 完全可能（尤其在贪心解码且 margin 很大时），它**不能**证明"测试没跨上下文"——设计是否真的跨上下文由第 2 条的 hash/delta 断言决定，不由结果反推。
4. 报告必须同时给出：`KL(dense ‖ reuse)` 的均值/p95、`max|Δlogit|`、top-1 与 top-5 一致率、以及 task-level 指标的 **CI**（restart/task cluster bootstrap）。单看 top-1 不作结论。

### 9.7 G7 — exclusive fallback 分类学测试（CPU + GPU）

每个失败的复用**有且仅有一个** terminal reason。至少覆盖：
`store_miss`、`stale_handle`、`source_pin_stale`、`source_slice_mismatch`、`prefix_gap`、`device_allocation_failed`、`cross_store_reservation_failed`、`cross_store_error`、`registration_dependency_missing`、`registration_store_capacity`、`residency_miss`、`residency_load_failed`、`rope_config_unavailable`、`approx_kv_core_disabled`，**新增** `coding_version_stale`、`coding_unknown_tool_effect`、`coding_path_ambiguous`。

断言：`Σ terminal_reason_counts == approximate_recovery_failed_dense`（防止 Phase7 wave-0 出现过的 `terminal_reason_counts={}` 缺失），且**不得双计**（Phase6 `5e47904ec` / `15634baf6` 的教训）。

### 9.8 G8 — lease / abort / reject / timeout / reset soak（CPU + GPU）

针对 §5.5：

| 场景 | 期望 |
| --- | --- |
| 正常完成 | lease 释放，`store_leases` 归零 |
| 请求被 scheduler 拒绝 | 立即释放（对应 `xs:scheduler.py:3045`） |
| 等待队列中被 abort | 立即释放（对应 `xs:scheduler.py:4090`） |
| copy 过程抛异常 | provisional slot 释放，lease 释放 |
| TTL 到期但 owner 未调用 unpin | **必须**由周期性 GC 回收（当前 `v40:` 不满足） |
| 全局 reset | 所有 lease / record / 依赖图清空，`orphans == 0` |

**Soak**：连续 `≥ 10,000` 次请求，断言 `store_records`、`store_leases`、`store_orphans`、`provisional_tokens`、device peak bytes **全部不单调增长**；结束后全部归零。

### 9.9 G9 — 双向压力（bidirectional pressure）

exact cache 与 approximate segment **共享同一预算**（Phase6 的既定反转四）。必须测两个方向：
1. **approximate 挤压 exact**：注册大量 source 段，观察 exact hit rate 下降与 self-eviction 是否被 `protect_request_prefix`（`xs:runtime.py:83`）挡住；
2. **exact 挤压 approximate**：制造高 exact 命中压力，观察 source 段被驱逐后是否**干净地** fail-closed 到 dense（而不是产生 stale copy）。

### 9.10 G10 — chunk × body 全因子实验（隔离已知混淆源）

**这是必须做的隔离实验**（§6.6）。

| 因子 | 水平 |
| --- | --- |
| `chunked_prefill_size` / `max_prefill_tokens` | `1024`、`2048`、`4096`（primary = `4096`） |
| island body 长度 | `256`、`512`、`1024`、`2048`、`4096` |
| `rho`（KV 压力） | `1.5`、`2.0` |
| arm | `dense` / `exact` / `C40` |

**报告要求**：任何速度数字必须同时给出其 chunk 水平；只有 `chunk=4096` 的结果可以作为 headline，`chunk=1024` 只能作为 sensitivity diagnostic（`headline=false`）。

### 9.10b G10b — eligibility 条件化报告与合法 estimand（**必做**）

**依据**：§8.3.1。C40 与 Phase 7 已判 `NEGATIVE` 的 R0 共用同一恢复 primitive，因此**性能不得默认转正**；而 conditional speedup 单独存在时不可解释。

**三臂 + 冻结 counterfactual**（同一冻结 trajectory、同一 seed 计划、
相同请求顺序；每臂从 clean reset 独立执行完整 workload）：

| 臂 | 名称 | 执行范围 | 用途 |
| --- | --- | --- | --- |
| A | `dense-full` | 完整 workload；所有请求 dense，同时保留 source-producing/release 事件 | `E_work` 分子与 dense baseline |
| B | `C40-full = G40 × R0` | 完整 workload；eligible执行C40，ineligible在**同一臂**dense | 被评系统；直接产出真实 `E_work` |
| C | `span-matched-R0-full` | 完整 workload；eligible回放同一span，ineligible同臂dense | 度量 selector/control-path overhead |

> **取消**先前版本的"臂 D unconditional R0"与 `Δ_survivorship` 差分：
> 请求集合不同无法归因。survivorship 用 `w` 结构性披露；workload effect
> 只能来自臂 B 的完整、有状态 trace，不能用 eligible-only 与 Dense
> 片段拼接。

**冻结流程**（Step 1–4 见 §8.3.1）：先冻结完整 trajectory，再离线生成
`Elig` 与每个请求的 span manifest。A/B/C 都执行全部请求；臂 C 只在
eligible 请求回放冻结清单，不重新跑 selector。

**必须输出的字段**（每个 cell）：

```text
coverage:
  eligibility_rate_count r, wilson_ci_95, skip_reason_histogram{...}
  time_weighted_coverage w = sum_dense_time(Elig) / sum_dense_time(All)
overhead:
  selector_overhead_ms_per_request        (measured on ALL requests, incl. ineligible)
  sidecar_io_ms, tokenize_ms, unique_match_search_ms
estimands (ratio-of-sums, NOT median composition):
  E_cond_request_path, E_cond_target_only
  E_work_request_path                       (actual full-trace ratio-of-sums)
uncertainty:
  restart_cluster_bootstrap_ci95 for E_cond and E_work
  n_restarts, n_requests_total, n_requests_eligible
control:
  arm_C_E_cond, overhead_selector_control = E_cond(C)/E_cond(B)
```

**统计口径**：
- **独立复制单元 = server restart**；单个 server 内的 formal request **不是**独立样本；
- 置信区间一律用 **restart-level cluster bootstrap**（对 restart 重采样，restart 内保留全部请求）；
- `E_cond` / `E_work` 均为 **ratio-of-sums**，不得先算 per-request speedup 再取 median 再合成；
- warm-up 请求必须显式丢弃，formal repeats 数量在执行前冻结（§9.18）。

**成功标准**：A/B/C 三条完整trace、`r`、`w`、`C_selector`、`E_cond`、
`E_work` 全部齐备，且 CI 由 restart-cluster bootstrap 给出。
**失败标准（判 `INVALID`，不进入 disposition）**：B没有执行全部请求，
缺 `E_work`/`w`/`C_selector`/臂 C，或用 eligible-only片段、median
speedup事后合成 workload 数字。

**判读**（与 §8.3.1 headline 判据一致）：workload headline 需 `E_work` CI 下界 `> 1 + MDE`；否则最多只能给 conditional 表述且必须同时给出 `w`。

### 9.11 G11 — 四本账 + 实测 N（1/2/4/8）

对每个 cell 输出：

```text
target_only_ms, request_path_ms(MDE), full_lifecycle_ms,
speedup_1, speedup_2, speedup_4, speedup_8              (full-setup)
speedup_incremental_1..8
break_even_N   (若 N≤8 未观察到，写 ">8 / not_observed")
```

`speedup_N = dense_total_N / (source_preparation + Σ_{i≤N} request_path_i)`，全部来自**实际累计**。

### 9.12 G12 — S0 / S4 matched coverage

在 `xs:` 上，S0 = `PolicyKind.S0_LRU`，S4 = `PolicyKind.S4_HIERARCHICAL`（`xs:cross_store/coordinator.py:107`，由 `tree_cache.eviction_policy == "hierarchical"` 选择）。

**必须修正 Phase7 的已知缺陷**：S0 与 S4 必须作为**相邻 launch block** 交替启动（`S0,S4,S0,S4,...`），而不是"S0 三次跑完再跑 S4"。否则只能报 `seed_matched_non_adjacent_restart_comparison`。

同时必须报告 **matched coverage**：两臂的 `expected_reusable_prefix_tokens > 0` 的请求集合必须一致，否则分母不同（Phase5 CL3 的教训）。

### 9.13 G13 — 四模式（Dense / Coding / Prefetch / Combined）

| 模式 | core | coding | prefetch |
| --- | ---: | ---: | ---: |
| Dense | 1 | 0 | 0 |
| Coding-only | 1 | 1 | 0 |
| Prefetch-only | 1 | 0 | 1 |
| Combined | 1 | 1 | 1 |

**前置条件**：必须先修复 §5.4 的 dead flag，否则模式标签无效。在 `xs:` 上对应的真实开关是 `SGLANG_APPROX_KV_CORE` / `SGLANG_APPROX_KV_HOST` / `SGLANG_APPROX_KV_PREFETCH`（`xs:approx_kv/config.py:125-127`），且 `PREFETCH` 依赖 `HOST`、`HOST` 依赖 `CORE`。

**组合验收**：Combined 选中的 token span 必须与 Coding-only **逐 token 相同**；关闭 prefetch 后必须**精确恢复** Coding-only 行为（span、copied tokens、fallback reason 全等）。

### 9.14 G14 — late / cancel / stale prefetch

| 场景 | 期望 |
| --- | --- |
| ticket 迟到（超过 deadline） | 该 span fail-closed 到 **coding plan 已声明的** dense range，不得改变 span |
| ticket 被取消 | lease 释放，无残留 worker/CUDA event |
| generation 已过期 | 拒绝，记 `stale_handle` |
| token hash 不匹配 | 拒绝，记 `source_slice_mismatch` |
| residency 缺失 | 记 `residency_miss`，dense |

### 9.15 G15 — RepoBench-P 与 SWE-bench 质量评测（功效计算）

**RepoBench-P**（静态、便宜、可大样本）：`exact-line agreement`、`edit similarity`、`cache-ready TTFT`。样本 ≥ 1000 例；用 paired bootstrap 报告差值 CI。

**SWE-bench Verified**（昂贵）：配对比较用 **McNemar**，其功效由 **discordant pair 数**决定，而不是总 task 数。因此：

1. **必须先由 pilot 估 discordant rate**（§9.19 Stage 1）——即"Dense 通过而 C40 失败" + "Dense 失败而 C40 通过"的 task 对占比；
2. 在拿到该估计之前，任何 `n` 都只是**占位数量级**。作为量级参考（`estimate`，**不得**当作冻结样本量）：若 Dense pass rate ≈ `0.30` 且 discordant rate 落在常见的 `0.10–0.20` 区间，检出 **10 pp** 绝对差约需 `n` 在**百量级**，检出 **5 pp** 约需 `n` 在**数百量级**；
3. 正式样本量在 pilot 之后与 MDE 一并冻结，并作为**单独的二次授权**事项提出（§11.9）。

**因此**：12 个任务只能做极粗的筛查，**无法**支持"C40 不损伤准确率"的结论。报告必须写成"该样本量与实测 discordant rate 下，无法排除 ≤ X pp 的损伤"，并给出 X 及其推导（含 discordant 计数）。

### 9.16 G16 — same-engine 对照（R0 / CacheBlend / KVCOMM）

要做任何跨方法比较，必须在**同一 engine、同一 model+revision、同一 prompt、同一任务顺序、同一 generation limits、同一 chunk 配置**下重跑：
- `R0`：`xs:` 已有（`RecoveryMode.COPY`）；
- `EPIC k∈{0,2,8,32}`：`xs:` 已有（`epic_plugin.py`）；
- `CacheBlend`：`wt:cacheblend`（本项目 R2，corrected rerun `c73c9c5ab`）；
- `KVCOMM`：`wt:kvcomm`（本项目 R4，`authoritative_historical_diagnostic`）。

**禁止**引用 `164/225`、`169/225`、`8.55×`、`4.77×` 与 C40 并列。

### 9.17 G17 — manifest / provenance 测试

每个结果目录必须含自哈希的 `RESULT_MANIFEST.json`，绑定：image digest、model+tokenizer revision、code pin（commit + tree）、runner path + sha256、plan/manifest revision、每个 raw 文件的 sha256、`known_gaps` 列表。`--check` 必须能在只读模式下重放校验（Phase7 的 `88/88` 模式）。

**额外要求（针对 B-06）**：所有 runner 的输入/输出路径必须来自 CLI 或环境变量，**禁止**任何 `/home/<user>/` 硬编码；CI 增加一条 lint：`grep -rn "/home/" benchmark/ python/ | grep -v test_` 必须为空。

---

### 9.18 G18 — 执行环境与测量合同（Docker 依赖锁 / central log / warm-up / formal repeats）

**这是所有 GPU 与 CPU 实验的公共前置合同，缺任一项判 `INVALID`。**

#### 9.18.1 Docker 依赖锁

| 要求 | 内容 |
| --- | --- |
| 基础镜像 | 以 digest 固定：`ghcr.io/ccdd2023/sglang@sha256:0be6e16e…`；**禁止**使用 tag |
| 依赖安装方式 | 构建**专用 Docker layer**，而不是运行时 `pip install`。若必须运行时安装，使用 `pip install --target <dir>` + `PYTHONPATH`（**不需要** `--break-system-packages`，见 §7.4） |
| 锁文件 | 必须提供 `requirements.lock`（`pip-compile` 或 `uv pip compile` 产出，含全部传递依赖与 hash），把 `openai` / `transformers` / `tokenizers` / `litellm` / `mini-swe-agent` 全部钉死到与基础镜像**兼容**的版本 |
| 验收 | 先版本化当前基础镜像的5条`pip check` baseline；派生layer不得新增任何冲突，并不得新增agent依赖相关项。若另行构建clean base image并固定新digest，则要求`pip check`零输出 |
| 记录 | 每次运行把 `pip freeze` 全量写入 artifact，并计入 manifest 的 sha256 |

#### 9.18.2 central log 合同

所有 runner 追加写入**单一 central JSONL**（每行一个事件，含 `run_id` / `phase` / `arm` / `restart` / `request_index` / `ts` / `event` / payload），并满足：

- 只追加，不重写；文件级 sha256 进 manifest；
- 每个 server start 有唯一 `run_id`，包含 image digest、model+tokenizer revision、code pin（commit + tree）、全部 `SGLANG_*` 环境变量快照、chunk / max-prefill / page_size / tp / pp / eviction policy；
- 每个请求记录：`is_warmup`、`is_formal`、`repeat_index`、`eligible`、`skip_reason`、四本账时间、cache outcome、exclusive terminal reason；
- 离线 consolidator 只读该 JSONL 与 raw 输出，产出自哈希 compact/summary（Phase 7 模式）。

#### 9.18.3 warm-up 丢弃与 formal repeats 合同（执行前冻结）

| 项 | 规则 |
| --- | --- |
| warm-up | 每个 arm 在每次 server start 后先跑 `W` 个 warm-up 请求，**全部丢弃**且不进入任何统计；`W` 在执行前冻结（建议 `W ≥ 2`，需由 pilot 的时间序列确认已进入稳态） |
| formal repeats | 每个 arm 的 formal 请求数 `M` 执行前冻结；arm 之间按 formal repeat **交替**（`A,B,C,A,B,C,…`），避免顺序效应 |
| reset | arm 之间做完整 reset；`arm_interval_peak_device_bytes` 自上次完整 reset 起计 |
| 丢弃可审计 | 被丢弃的 warm-up 请求仍必须写入 central log（`is_warmup=true`），以便复核"丢了几条、为什么" |
| 禁止 | 事后按结果决定丢弃哪些请求；任何 post-hoc 丢弃必须单独记录并在 disposition 中披露 |

### 9.19 G19 — 两阶段统计计划（pilot → confirmatory）

**R07 的强制流程。**

| 阶段 | 内容 | 允许的结论 |
| --- | --- | --- |
| **Stage 0 — engineering screening（restart-0）** | 单个 restart，只验证：配置可达、无 capacity error、遥测字段完整、reset/orphan/lease 归零、fallback 分类学自洽 | **只能**得出"工程可执行 / 不可执行"。**不得**用其未知 `sample_sd` 判定 `NEGATIVE` 或 `POSITIVE`，也不得据此触发 early stop |
| **Stage 1 — pilot（≥ 2 个独立 restart）** | 估计 restart 间方差、请求内自相关、`E_cond`/`E_work` 的 cluster bootstrap 分布宽度；对 accuracy 估计 **discordant pair rate** | 产出方差与 discordance 估计；据此**冻结 MDE 与 confirmatory plan**（样本量、restart 数、early-stop 规则） |
| **Stage 2 — confirmatory** | 按冻结计划执行。**默认每个确认性 cell 4 个 restart**；`2` 个 restart **仅**在 feasibility 受限时允许，且必须在 disposition 中标注为"最低可行配置，统计功效不足" | 允许给出 `POSITIVE` / `NEGATIVE` / `INCONCLUSIVE` |

**task repeats 的主分析规则必须在 pilot 前冻结**：

- 每个 task/arm 使用 3 个固定且配对的 seed；
- primary task-level binary outcome =
  `majority_resolved`（3 次中至少 2 次 official resolved；3 次无 tie）；
- McNemar 只对 Dense 与 C40 的 paired `majority_resolved` 做检验；
- `n_discordant` 指**聚合后的 task pair**中一臂通过、另一臂失败的 task 数；
- secondary sensitivity 报告每次 repeat 的原始结果，并用 task random
  intercept 的层次二项模型或 task-cluster bootstrap；不得把 3 次 repeat
  当成 3 个独立 task。

McNemar 的功效计算必须基于上述 task-level discordant rate，而不是 Dense
pass rate或总 repeat 数。pilot 先估计 discordance，再冻结 confirmatory
task 数；此前的 `n=40/130/500` 只能是量级占位。

**cluster bootstrap 规则**：latency 按 `restart` 聚簇重采样；accuracy 按 `task` 聚簇重采样（同一 task 的多次 repeat 整体进出）。**formal requests 不是独立样本**，禁止对请求做朴素 bootstrap。

## 10. 集成架构（用户问题 4 的第一半）

### 10.1 绝对不能 merge 整个分支

| 理由 | 证据 |
| --- | --- |
| 规模 | 相对 shared-core `191 files / 52,549 insertions / 220 deletions`（§2.2） |
| 内容 | 其中 163 个文件是 `benchmark/multi_workflow/` 的 V8–V44 历史驱动、audit、probe、preregistration；`shared` role 的 scope 检查直接判 rc=1（§2.4） |
| 算法增量极小 | V40 的实现 commit `03ba74050` 只有 `953 insertions / 8 files`，且一行 runtime 都没改（§2.3） |
| merge base 落后 | 落后 `origin/main` 4,945 commits、落后 current cross-store 4,786 commits（§2.1） |
| 不可复现 | 实验面依赖 `/home/gfy/**` 与冲突的第三方依赖（§7.3、§7.4） |

### 10.2 更不能把分支的旧 `kvcomm store/manager/scheduler` 带进当前底座

这会**回归 Phase 6 的六项正确性修复**。全文统一按**六项**计（A 自身 prefix 保护、B SWA lock metadata、C stale victim、D provisional slot 生命周期、E fallback taxonomy / 双计防护、F object graph 无孤儿）。host tier / HiCache backend 与依赖注册属于 Phase 6 的**功能面**，不计入这六项正确性修复。逐项对照（`xs:` 侧证据）：

| Phase6 修复 | `xs:` 位置与 commit | 若引入 `v40:` 旧实现的风险 |
| --- | --- | --- |
| **prefix self-eviction** | `xs:approx_kv/runtime.py:83-114` `protect_request_prefix()`；commit `af81934e4` "fix(cache): protect a request's own prefix during KV recovery" | 恢复过程中把请求**自己的 prefix** 当作合法 cross-store victim 释放，再把同一批 slot 发回给自己 |
| **SWA metadata on lock release** | `xs:approx_kv/runtime.py:107-114`（`to_dec_params`）；commit `db2d18ff0` | 释放锁时越过 SWA window，误减另一个请求仍持有的祖先引用计数 |
| **stale victim** | `xs:cross_store/allocator.py:180-194`（`stale_victims += 1; refresh_resources = True; break`）；commit `3379e6699` | 一个已被替换的 victim 直接让整次 allocation 失败，而其它合法 victim 本可满足 |
| **provisional slot 生命周期** | `xs:approx_kv/runtime.py:37`（release）/`:69`（commit）；调用点 `schedule_batch.py:1220,2257`、`scheduler.py:3045,4090`；commit `391bb8990` + `15634baf6` | 拒绝/中止路径漏放 recovery slot → allocator 泄漏 |
| **fallback taxonomy / 双计** | `xs:approx_kv/runtime.py:449-466`（`prefix_gap` 记为 dense fallback 而非 exact hit，commit `5e47904ec`）与 `:589-597`（cross-store 已记录终因时不再记 `device_allocation_failed`） | exact hit 被高估、fallback 被双计，指标不可信 |
| **object graph 无孤儿** | `xs:cross_store/object_graph.py:51`（`eviction_closure`）/`:68`（`remove_closure`）/`:88`（`assert_no_orphans`） | 依赖对象先于被依赖对象被释放，产生悬挂引用 |

**结论**：`v40:python/sglang/srt/mem_cache/kvcomm/*`（store/manager/transfer/radix_backend/types/config）与 `kvcomm_exact.py` **不得**移植到 `xs:`。它们是 shared-core 时代的产物，早于上述全部修复。

### 10.3 推荐方案：在当前 cross-store 上重实现最小 payload

```text
┌──────────────────────── 新增（最小 payload）────────────────────────┐
│ 1. 结构化 coding selector                                          │
│    输入: agent trajectory + 每个 tool group 的结构化 provenance     │
│          {read_paths, write_paths, unknown_effect,                 │
│           worktree_generation, source_content_sha256}              │
│    输出: 至多一个候选岛 + 完整可审计 decision dict                  │
│    位置建议: benchmark/approx_kv/coding/selector.py（实验面）        │
│              或 python/sglang/srt/mem_cache/approx_kv/coding/       │
│              （若要进 runtime，需 shared-core owner 同意）           │
│                                                                    │
│ 2. sidecar / adapter                                               │
│    version-4 manifest：在 v40 的 version-3 基础上增加               │
│      worktree_generation / source_content_sha256 /                 │
│      selector_version / tool_provenance_schema_version             │
│    路径全部来自 CLI/env，禁止硬编码                                  │
└────────────────────────────────────────────────────────────────────┘
                              │ KVReusePlan（只声明 what）
                              ▼
┌───────────── 复用（零改动：backend 与 store/lifecycle 层）───────────┐
│ 3. copy/RoPE backend（可直接复用，无需改动）                         │
│    RecoveryMode.COPY  (xs:approx_kv/types.py:20)                    │
│    copy_and_rotate()  (xs:approx_kv/radix_backend.py:196)           │
│    execute_reuse_plan()(xs:approx_kv/transfer.py:84)                │
│                                                                    │
│ 4. 当前 cross_store lifecycle / stats（可直接复用）                  │
│    CrossStoreCoordinator/Allocator/Budget/Policy/ObjectGraph        │
│    ApproxKVSegmentStore（register/lookup/pin/unpin/                 │
│      gc_expired_leases/load_resident/commit_residency/release/reset）│
│    17 个 Prometheus counter/gauge + 14 个 fallback reason           │
└────────────────────────────────────────────────────────────────────┘
┌──────── 必须新写（request execution seam，**不是零改动**）───────────┐
│ 5. middle-span staging controller + 状态机（§10.3.1）                │
│    当前 restore_request_prefix()/resolve_reuse_spans() 只支持         │
│    "从 exact_length 连续开始"的 span，中间带 dense gap 的 middle      │
│    span 会被判为 prefix_gap 并整体 dense（见下）。                    │
└────────────────────────────────────────────────────────────────────┘
```

### 10.3.1 当前底座的 middle-span 集成缺口（**必须新写的部分**）

**这是本报告最容易被误读的一点：copy/RoPE backend 可以零改动复用，但 request execution seam 必须新实现。**

`xs:python/sglang/srt/mem_cache/approx_kv/runtime.py:430-467` 的 `resolve_reuse_spans()` 逐字如下：

```python
430:     reusable_limit = len(req.full_untruncated_fill_ids) - 1
431:     exact_length = len(req.prefix_indices)
...
437:     next_target = exact_length
438:     for segment in ordered_segments:
439:         if segment.target_end <= exact_length:
440:             continue
441:         if segment.target_start > next_target:
442:             break                      # <-- 任何"跨 gap"的 segment 在此被丢弃
443:         active_segments.append(segment)
444:         next_target = max(next_target, segment.target_end)
...
447:     restore_end = min(next_target, reusable_limit)
448:     restore_length = restore_end - exact_length
449:     if restore_length <= 0:
...
463:             manager.record_fallback("prefix_gap", pending_length)
464:             manager.record_request("reuse", "dense_fallback")
...
467:         return None
```

语义（`verified-code`）：**当前底座只处理"从 `exact_length` 起连续开始"的恢复 span**。V40/C40 的几何恰恰相反——它的岛**严格位于中部**，`target_start > exact_length` 几乎总是成立，中间必然存在一段需要 dense 计算的 prefix gap。按现有代码，这类 span 在 `:441-442` 被 `break` 丢弃，随后在 `:449-467` 被记为 `prefix_gap` 并整体退回 dense。

**因此**：如果只是"把 selector 接到现有 `restore_request_prefix()` 上"，C40 会**恒等于 dense**，实验将毫无意义。必须新写一个 middle-span staging controller。

#### 建议的状态机

```text
        ┌────────────────────────────────────────────────────────────┐
        │  INIT                                                      │
        │  match_prefix 完成，metadata/manifest 已解析                │
        └───────────────┬────────────────────────────────────────────┘
                        │ selector 判定 eligible 且 span 严格中部
                        ▼
        ┌────────────────────────────────────────────────────────────┐
        │  DENSE_PREFIX                                              │
        │  [exact_length, target_start) 交给正常 extend 路径计算       │
        │  期间: 持有 protect_request_prefix 锁；source handle 已 pin  │
        │  失败/中止 -> ABORTED                                       │
        └───────────────┬────────────────────────────────────────────┘
                        │ dense prefix 已写入 req_to_token 且位置对齐
                        ▼
        ┌────────────────────────────────────────────────────────────┐
        │  COPY_READY                                                │
        │  校验: handle current / generation 一致 / token slice 相等   │
        │        / source 已 device-resident / 目标槽已 provisional 分配│
        │  执行: execute_reuse_plan() -> copy_and_rotate()            │
        │  任一校验失败 -> DENSE_FALLBACK（唯一 terminal reason）      │
        └───────────────┬────────────────────────────────────────────┘
                        │ copied_k == copied_v == length 且 mechanically_valid
                        ▼
        ┌────────────────────────────────────────────────────────────┐
        │  DENSE_SUFFIX                                              │
        │  [target_start+length, len(prompt)) 正常 extend             │
        └───────────────┬────────────────────────────────────────────┘
                        ▼
        ┌────────────────────────────────────────────────────────────┐
        │  COMMIT  -> 释放 lease、provisional 槽转正、写 stats         │
        └────────────────────────────────────────────────────────────┘

  任意状态 --(reject / abort / exception / reset / timeout)--> ABORTED
      ABORTED: release_provisional_recovery_slots + unpin + 唯一 fallback 归因
```

#### Rolling 请求的双角色生命周期与锁交接

V40 的 rolling 请求通常同时承担两个角色：

1. **consume**：消费上一请求注册的 source，执行当前 middle-span reuse；
2. **produce**：当前请求结束后，从当前 prompt/KV 中物化下一请求可能使用的
   source。

因此 metadata 不能是互斥的 `REGISTER` 或 `REUSE` 单枚举。建议 request
state 同时保存：

```text
consume_state:
  source_handle / source_lease / target_span / terminal_reason
produce_state:
  candidate_span / output_key / source_provenance / approx_depth
```

生命周期顺序必须冻结为：

1. pre-admission 只解析/校验 metadata，**不分配 target slot、不驱逐**；
2. `add_one_req` 成功后由 scheduler 的常规 request lock 接管 prefix
   ownership；
3. 在该常规锁已生效后执行 dense prefix 与 copy reservation。若未来确实
   需要在 admission 前做会触发驱逐的动作，必须引入可跨调用的显式
   prefix-guard lease，并在 scheduler lock 获取成功后做一次可审计 handoff；
   不能依赖一个已经退出的临时 context manager；
4. copied island 的 slot 在进入 `DENSE_SUFFIX` 前必须写入
   `req_to_token` 并完成所有权转正，使 suffix forward 能 attention 到
   `[dense prefix + copied island]` 的完整逻辑前缀；
5. 请求结束时先完成 consume cleanup，再处理 produce：
   - release consume lease；
   - 若当前请求使用过 approximate copy，则新 source 必须标
     `provenance=APPROXIMATE` 与 `approx_depth>=1`，**禁止写入 exact
     Radix**；
   - 默认 C40 qualification 只允许从 `approx_depth=0` 的 dense/exact
     请求物化下一 source，防止误差跨轮累积；
   - 若未来研究 chained reuse，必须作为独立实验轴，冻结最大 depth 并逐层
     报告质量衰减。

上述双角色模型还要求 `finish/abort/reject/reset` 同时清理 consume 与
produce 两套状态，不能因 source materialization 成功而漏掉 target lease，
也不能因 target fallback 而留下 pending source。

状态必须再分成两层：

- **request-lifetime**：`consume_state`、`produce_state`、source lease、
  middle-span cursor、当前状态机阶段；跨多个 chunk/调度轮持续存在；
- **per-round transient**：本轮尚未转正的 provisional indices、临时
  allocator reservation、一次性 copy transaction。

`init_next_round_input` 只能清理上一轮遗留的 per-round transient。它**不能**
清空 request-lifetime 状态或 source lease，否则多轮 `DENSE_PREFIX` 尚未到达
`COPY_READY` 就会丢失计划。request-lifetime 状态只在 terminal
finish/abort/reject/reset，或重新 match 后明确证明冻结 plan 已失效时清理。

#### 建议的 scheduler hook 位置（全部为**当前底座**的既有 seam）

| Hook | `xs:` 位置 | C40 需要做什么 |
| --- | --- | --- |
| 进入 `init_next_round_input` | `schedule_batch.py:1216-1220`（已调 `release_provisional_recovery_slots`） | 只释放上一轮未转正的provisional/transient资源；保留request-lifetime consume/produce状态、source lease与middle cursor。若rematch使冻结plan失效，记录exclusive reason后才转ABORTED并清理 |
| `match_prefix` 之后、恢复之前 | `schedule_batch.py:1315-1332`（`if self.approx_kv_metadata is not None:` → `with protect_request_prefix(...)`） | **在此新增第三分支**：`elif approx_kv_manager.config.coding_middle_span_enabled: stage_middle_span(tree_cache, self)`，与既有 `restore_request_prefix_epic` / `restore_request_prefix` 并列；**必须仍在 `protect_request_prefix` 上下文内** |
| admission / 常规锁获取 | `schedule_policy.add_one_req` | metadata staging 不得在此之前触发allocation/eviction；成功admit后记录常规prefix lock已接管 |
| dense prefix 完成、copy 执行点 | `schedule_policy.py` 的 `add_one_req` 侧与 `schedule_batch.prepare_for_extend` 之间 | `DENSE_PREFIX → COPY_READY` 的触发点；参考 `v40:` 的 `copy_ready()/copy_into_request()` 双点触发（`v40:schedule_policy.py:608-612`、`v40:schedule_batch.py:1483-1504`），但要重写为使用 `xs:` 的 provisional 槽与 cross-store 预算 |
| 所有权转正 | `schedule_batch.py:2257` 的 `commit_provisional_recovery_slots` 作为既有语义先例 | middle-span controller必须在suffix forward前完成等价的`req_to_token`写入与所有权转正；若现有调用时机太晚，应新增copy-boundary hook，但复用同一release/commit记账语义 |
| 拒绝路径 | `scheduler.py:3045` | `→ ABORTED`，释放槽与 lease |
| 中止路径 | `scheduler.py:4090` | 同上 |
| 通用 teardown | `common.py:167-170`（`release_kv_cache`） | 同上，兜底 |
| 请求结束 / source 物化 | RadixCache 的 `cache_finished_req` | 同时闭合consume与produce；释放本次source，再按provenance/approx_depth规则物化下一source（对应 `v40:kvcomm_exact.py:582`） |

#### 必须保留的既有不变量（**新代码不得绕过**）

| 不变量 | `xs:` 位置 | 对 C40 的具体要求 |
| --- | --- | --- |
| prefix ownership | `approx_kv/runtime.py:83-114` + scheduler常规lock | 推荐方案在admission前不分配/驱逐；admit后由scheduler常规request lock覆盖`DENSE_PREFIX→COPY_READY→DENSE_SUFFIX`。若存在任何pre-admission驱逐动作，须用显式guard lease保护并在常规lock获取后审计式handoff；释放参数必须保留`to_dec_params()`的SWA metadata |
| provisional ownership | `approx_kv/runtime.py:37`（release）/`:69`（commit） | copy 目标槽必须走 provisional 分配；四条异常路径（`schedule_batch.py:1220`、`scheduler.py:3045`、`scheduler.py:4090`、`common.py:170`）全部要能释放 |
| stale victim 容忍 | `cross_store/allocator.py:180-194` | 为 middle span 腾空间时必须复用现有 allocator，不得自建驱逐循环 |
| exclusive fallback / 不双计 | `approx_kv/runtime.py:449-466`、`:589-597` | 新增 `coding_*` terminal reason 必须与既有 14 个互斥；`Σ reason == approximate_recovery_failed_dense` |
| object graph 无孤儿 | `cross_store/object_graph.py:51,68,88` | source 段与其 host copy/依赖必须注册进图，驱逐走 `remove_closure` |

#### 结论（R05）

- **可直接复用的核心语义**：`copy_and_rotate()` / `execute_reuse_plan()` /
  `ApproxKVSegmentStore` / `CrossStore*` 的现有分配、驱逐、lease与记账路径；
  但需用薄适配层增加C40 method/provenance/approx-depth字段，不能替换底座实现；
- **必须新实现**：middle-span staging controller、上述状态机、以及 `schedule_batch.py:1315-1332` 处的第三分支接线；
- **不得写成**"R0 runtime 零改动即可支撑 C40"——这与 `runtime.py:441-442,449-467` 的实际语义矛盾。
- 工作量修正（`estimate`）：§8.2 中"sidecar/adapter 1–1.5 人周"之外，**另加 middle-span controller + 状态机 + 四条异常路径接线 `2–3 人周`**，并需要一轮独立 review。

**接线原则**（与分支自身的 ownership 契约一致，`v40:docs/kvflow/ARCHITECTURE.md`）：
- selector 只产生 `KVReusePlan`，**不得**调用 `ensure_resident`，不得做 eviction/priority/deadline；
- 位置修正**只在** transfer backend 内发生一次；
- 任何缺失/过期/不匹配/迟到 → fail closed 到 plan 已声明的 dense range。

### 10.4 命名：不要引入新的恢复 primitive

**禁止**把当前方法命名为 `R6` 或 `L0`。理由见 §4.2：其数据面与 `R0` 数学等价。

**推荐的系统 candidate ID**：

```text
C40  =  G40 (grounded coding selector)  ×  R0 (Raw + RoPE executor)
```

- `G40` = "grounded observation island selector"，是**策略/准入**维度的第一个正式候选；
- `R0` = 已有的恢复 primitive，不变。

正交维度（与既有命名体系对齐）：

| 维度 | 取值 | 权威定义 |
| --- | --- | --- |
| 恢复机制 R | `R0`(Raw+RoPE) / `R1`(EPIC) / `R2`(CacheBlend) / `R3`(Cache-Craft, deferred) / `R4`(KVCOMM) / `R5`(CacheTune) | `docs:research/phase_reports/PHASE4_RECOVERY_METHODS_REPORT.md:149-153` |
| 准入 G | `G0`(无准入/全量) / `G40`(grounded coding observation) | 本报告新增 |
| 调度 S | `S0`(LRU) / `S1`(workflow steps) / `S2`(Belady-style oracle) / `S3`(recovery value) / `S4`(hierarchical) | `docs:research/phase_reports/PHASE5_WORKFLOW_SCHEDULING_REPORT.md:135-139` |
| 驻留/预取 P | `P0`(off) / `P1`(free-space-only) / `P2`(known-dead-only eviction) / `P3`(oracle-farther-use) | `docs:research/phase_reports/PHASE5_WORKFLOW_SCHEDULING_REPORT.md:147-150` |

因此一次完整实验点写作：`C40 = G40 × R0`，配 `S ∈ {S0, S4}`、`P ∈ {P0, P1}`。

### 10.5 不改写 Phase 4–7

Phase 4/5/6/7 的报告、manifest、disposition **一律不改写**（`docs:TRACKING.md` append-only 原则；Phase7 `RESULT_MANIFEST=88/88`、`known_gaps=[]` 已封版）。

C40 必须走**新的 versioned candidate plan/manifest**：

```text
docs:IMPLEMENTATION_PLAN_C40_V1.md            (新计划，byte-frozen 后 pin)
impl:benchmark/approx_kv/results/c40/         (新结果目录)
impl:benchmark/approx_kv/results/c40/RESULT_MANIFEST.json
impl:benchmark/approx_kv/results/c40/C40_DISPOSITION.json
```

引用 Phase7 结论时，只以只读方式引用，不得回填、不得合并统计。

### 10.6 正确的实现顺序（本项目既定原则）

```text
1. exact cache                 （先证明 exact 路径完全正确、无自我驱逐、无污染）
2. controlled C40 reconstruction（在 exact 之上加受控的 G40 × R0，全程 fail-closed）
3. dense fallback              （任何不确定一律回落，且必须可归因到唯一 terminal reason）
```

**不得**跳过第 1 步直接做第 2 步；不得让第 2 步的近似 KV 写入 exact Radix（除非经过 dense materialization）。

---

## 11. 建议的版本化执行计划（**全部为建议，待用户授权**）

> **授权状态声明（R08，置于本节最前）**
>
> **本节的全部 Gate、Track、预算与样本量均为"建议"，当前 `AUTHORIZATION = PENDING USER AUTHORIZATION`，包括 Gate 0 在内没有任何一个 Gate 已被授权。**
> 本报告能给出的唯一结论是：**推荐的下一步是 Track A（zero-GPU 的代码与 provenance 修复）**。
> 任何 GPU 执行、任何预算占用、任何样本量扩张，都必须先获得明确的用户授权，并按 Phase 7 的治理链完成"预注册 → pin → 授权 → 执行"。

### 11.1 Track 划分与授权状态

| Track | 内容 | 资源性质 | 授权状态 |
| --- | --- | --- | --- |
| **Track A** | Zero-GPU 的代码与 provenance 修复（Gate 0 + Gate 1） | CPU only，`0 GPUh` | `PENDING USER AUTHORIZATION`（**推荐的下一步**） |
| **Track B** | 小规模 Docker GPU pilot（Gate 2 + Gate 3 的 Stage 0/Stage 1） | GPU，**hard cap 见 §11.10** | `PENDING USER AUTHORIZATION`（须在 Track A Exit 后单独申请） |
| **Track C** | 条件性 confirmatory 扩张（Gate 3 Stage 2、Gate 4、Gate 6） | GPU + task-runs | `NOT REQUESTED` —— **只有在 Track B pilot 完成且获得新的授权后**才可提出 |
| **Track D** | 外部 prefetch 分支与外部 raw evidence 的解锁（§13.6） | 无本地 GPU 占用 | `BLOCKED_EXTERNAL`，**不占用当前预算** |

**硬性规则**：
- Track B 的任何一个 start 都不得在 Track A Exit 前执行；
- Track C **不得**由 Track B 的结果自动触发，必须重新申请授权；
- Gate 3 的 16-start 扩展路径与 SWE-bench 扩样（`n>40`）各自需要**单独的二次授权**，不包含在 Track B 或 Track C 的初始授权中；
- Track D 不产生本地资源消耗，可与 Track A 并行推进。

### 11.2 Gate 总览（映射到 Track）

| Gate | 名称 | Track | 资源 | 授权状态 |
| --- | --- | --- | --- | --- |
| Gate 0 | Code / provenance fix | A | CPU | `PENDING USER AUTHORIZATION` |
| Gate 1 | CPU 正确性面 | A | CPU | `PENDING USER AUTHORIZATION` |
| Gate 2 | GPU same-context canary | B | GPU（pilot cap 内） | `PENDING USER AUTHORIZATION` |
| Gate 3 | Cross-context microbenchmark（Stage 0/1 属 B；Stage 2 属 C） | B / C | GPU | Stage 0/1 `PENDING`；Stage 2 `NOT REQUESTED` |
| Gate 4 | Workflow / scheduler | C | GPU | `NOT REQUESTED` |
| Gate 5 | Prefetch composition | C（依赖 D） | GPU | `BLOCKED_EXTERNAL` |
| Gate 6 | RepoBench / SWE 质量 | C | GPU + task-runs | `NOT REQUESTED` |
| Gate 7 | 双模型 review + publication | A/C | 无 GPU | `PENDING` |

### 11.3 Gate 0 — Code / provenance fix（Track A，`PENDING USER AUTHORIZATION`）

**Entry**：用户授权。
**内容**：
1. 实现 tool provenance schema v1（`read_paths` / `write_paths` / `unknown_effect` / `worktree_generation` / `source_content_sha256`）；
2. 用结构化字段重写失效判定，`_SHELL_MUTATION` / `_INPLACE_MUTATION` 降级为**只能加严、不能放宽**的次级信号；
3. `unknown_effect` fail closed；路径规范化 + rename 建模；
4. 补齐 §9.1 的对抗矩阵、property 测试与差分测试（差分测试的真实 mutation 必须在 Docker 内的**临时可写副本**上执行，见 §9.1.4 与 §13.5）；
5. 修复 B-03（更正 review request 的 active entry point 表述）、B-04（移除或真正接线 `coding_aware_lossy_enabled`）、B-05（周期性 `gc_expired_leases` + `_is_leased` 检查过期，或在 reject/abort/exception/reset 四条路径显式 unpin）；
6. 消除全部 `/home/<user>/` 硬编码，加入 CI lint；
7. 建立 §9.18 的 Docker 依赖锁与专用 layer；沿用旧base时
   `pip check`不得比版本化baseline新增任何项，使用新clean base时才要求
   零输出。

**Exit（全部必须满足）**：
- §9.1 对抗矩阵 100% 通过；差分测试 **FN = 0**；property 测试 1000 例无反例；
- `grep -rn "/home/" benchmark/ python/ | grep -v test_` 为空；
- 四模式开关在代码中真实生效（可用断言测试证明关闭 coding 时 copied_tokens ≡ 0）；
- soak 10k 请求后 lease/record/orphan/provisional 全部归零；
- `pip install --require-hashes -r requirements.lock` 后，
  `pip check` 相对基础镜像 baseline 无新增冲突；若Track A选择重建clean
  base，则新digest下为零输出。

**预算（`estimate`，建议）**：`2–3 人周`，GPU **`0`** 小时。

### 11.4 Gate 1 — CPU 正确性面（Track A）

**Entry**：Gate 0 Exit。
**内容**：G1、G2、G3、G7、G8（CPU 部分）、G17、G18。
**Exit**：全部测试绿；`RESULT_MANIFEST --check` 通过；fallback 分类学满足 `Σ terminal_reason_counts == approximate_recovery_failed_dense` 且无双计；central log 合同生效。
**预算（`estimate`）**：`0.5 人周`，GPU `0` 小时。

> **注**：middle-span controller（§10.3.1）的实现与 review 也在 Track A 内完成（`2–3 人周` estimate）；否则 Gate 2/3 无对象可测。

### 11.5 Gate 2 — GPU same-context canary（Track B）

**Entry**：Gate 1 Exit **且** 用户对 Track B 的单独授权。
**建议设置**：
```text
model            Qwen/Qwen3-0.6B @ c1899de289a04d12100db370d81485cdf75e47ca
image            ghcr.io/ccdd2023/sglang@sha256:0be6e16e...
tp=pp=1, page_size=1, chunked_prefill=4096, spec-decode=off
SGLANG_APPROX_KV_CORE=1, CROSS_STORE=1, HOST=0, PREFETCH=0, EPIC=0
```
**内容**：G4（K/V + RoPE 张量：正负 delta、全层全 head、`rotary_dim<head_dim`）、G5（same-context canary，容差判据见 §9.5）。
**Exit**：`max|ΔK|`/`max|ΔV|` 在 dtype 容差内、`max|Δlogit|` 在预冻结容差内、贪心输出逐字符一致；`rotated_k_tokens == copied_k_tokens == span_len` 逐层成立；注入 ±1 `rope_delta` 的 corruption canary 被检出。
**Early stop**：任一超出容差 → 停止 Track B，回到 Track A。
**预算（`estimate`，建议）**：`≤ 2 server starts`。

### 11.6 Gate 3 — Cross-context microbenchmark（Stage 0/1 属 Track B；Stage 2 属 Track C）

**Entry**：Gate 2 Exit。
**建议设计**：`chunk {1024, 4096} × body {512, 2048} × rho {2.0} × arm {A dense, B C40, C span-matched R0}`（三臂定义见 §9.10b）。**pilot 阶段不做全因子**；全因子（含 `chunk 2048`、更多 body、`rho 1.5`）属于 Stage 2 扩展。
**预注册基准线（写入 manifest）**：Phase 7 在同 image / 同模型 / `chunk4096` 下对 R0 的判定为 `NEGATIVE`（`0.7723 / 0.7751 / 0.9334 / 0.9362`）。C40 沿用同一恢复 primitive，**这是一个不利先验，需由 pilot 检验**，不是既定结论。
**Stage 划分（R07）**：
- **Stage 0（screening，1 restart）**：只判工程可执行性，**不得**据此判 `NEGATIVE`；
- **Stage 1（pilot，≥ 2 独立 restart）**：估方差与 discordance，**冻结 MDE 与 confirmatory plan**；
- **Stage 2（confirmatory，默认 4 restart/cell）**：需**二次授权**；`2 restart` 仅在 feasibility 受限时允许并标注功效不足。
**Exit（Stage 2；缺一判 `INVALID`）**：
1. 四本账 + 实测 `speedup_{1,2,4,8}` + `break_even_N` 齐备；
2. `r`（计数 eligibility）、`w`（time-weighted coverage）、`C_selector`、`E_cond`、`E_work` **全部报告**，estimand 为 ratio-of-sums；
3. CI 由 **restart-level cluster bootstrap** 给出；
4. 臂 C 已执行，`overhead_selector_control` 被量化（**不得**称为 selection 收益）；
5. chunk 因子被显式分离；headline 仅允许来自 primary chunk 且需 `E_work` CI 下界 `> 1 + MDE`。
**预算（`estimate`，建议）**：Stage 0+1 合计 `≤ 6 server starts`（含在 Track B 的 pilot cap 内）；**Stage 2 的 16-start 扩展路径需单独二次授权**。

### 11.7 Gate 4 — Workflow / scheduler（Track C，`NOT REQUESTED`）

**建议设计**：`S ∈ {S0, S4} × rho ∈ {1.5, 2.0} × restart ∈ {0,1,2,3}`，**S0/S4 相邻交替**启动；同时报告 all-reusable 与 workflow-only 两个口径，并报告 matched coverage。
**Exit**：若 C40 臂的 dense fallback 率 > 40%，结论只能写 `DESCRIPTIVE`。

### 11.8 Gate 5 — Prefetch composition（Track C，依赖 Track D，`BLOCKED_EXTERNAL`）

**Entry**：Gate 4 Exit **且** prefetch 分支可在受控环境中获得（当前**不满足**，§2.5.1）。
**硬性验收**：Combined 的选中 span 与 Coding-only **逐 token 相同**；关闭 prefetch 精确恢复 Coding-only；无 lease/worker/CUDA event 泄漏。
**若 Track D 未解锁**：本 Gate 保持 `BLOCKED_EXTERNAL`，不阻塞 Gate 6/7，也**不占用**任何预算。

### 11.9 Gate 6 — RepoBench / SWE 质量（Track C，`NOT REQUESTED`）

**建议设计**：
- RepoBench-P：≥ 1000 例，restart/task cluster bootstrap，报告 `exact-line agreement` 与 `edit similarity` 差值的 95% CI；
- SWE-bench Verified：先 `n = 40` task × 3 个配对 seed；按
  `majority_resolved` 聚合为每 task/arm 一个 primary binary outcome，再估
  McNemar discordant pair rate与run-to-run方差。扩样到 `n = 130` 或
  `n = 500` 属于**单独的二次授权**事项。
**Exit**：给出"该样本量下无法排除 ≤ X pp 损伤"的显式 X 值；**禁止**写成"无损伤"。
**Early stop**：若 40 task × 3 repeats 中 Dense 自身的 run-to-run 翻转率 > 15%，先修 harness 稳定性。
**计量单位（R08）**：质量 campaign 以 **task-runs** 单列计量（`n_tasks × n_repeats × n_arms`），**不得**折算成 server starts 混入 GPU start 预算。

### 11.10 建议预算（**全部 `estimate`，待授权**）

> 此前版本给出的"`≤ 62 starts / ≤ 28.2 GPUh` 保守上界"**予以撤回**：在完成 1-start 校准之前，任何总量上界都缺乏依据。以下按 Track 分列，且只对 Track A/B 给出可申请的数字。

| Track | 人力（`estimate`） | GPU server starts | GPUh | task-runs | 授权状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| **A**（Gate 0+1 + middle-span controller） | `4.5–6.5 人周` | **0** | **0** | 0 | `PENDING`（推荐先做） |
| **B**（Gate 2 + Gate 3 Stage 0/1） | `1 人周` | **建议 hard cap `≤ 8 starts`** | **建议 hard cap `≤ 2 GPUh`（`estimate`）** | 0 | `PENDING`（须单独申请） |
| **C**（Gate 3 Stage 2 / Gate 4 / Gate 6） | 待定 | 待定 | 待定 | 待定 | `NOT REQUESTED`，须 pilot 后重新授权 |
| **D**（外部解锁） | `0.5 人周` | 0 | 0 | 0 | `BLOCKED_EXTERNAL`，不占预算 |

**Track B 的 cap 使用规则**：
- 先执行 **1 个 start 的校准**（single-start calibration），实测该配置下每 start 的 wall-clock 与 GPU-equivalent 小时；
- **以校准结果冻结**剩余 starts 的分配；上表的 `≤ 8 starts / ≤ 2 GPUh` 只是申请时的初始建议上界，不是已核实的容量；
- 一旦达到 cap，**停止并重新申请**，不得自行追加。

**参考量级**：Phase 7 实际使用 `22 starts / 1.310 GPUh`（其硬上限为 `36 starts / 6 GPUh`，实际占 `21.8%`）。

### 11.11 总体成功 / 失败判据

| 判据 | 成功 | 失败 |
| --- | --- | --- |
| **正确性** | Gate 0–2 全绿；same-context 三类容差全部满足 | 差分测试 FN > 0，或 same-context 超容差 → 方法不可用 |
| **速度** | primary chunk 下 `E_work` 的 cluster-bootstrap CI 下界 `> 1 + MDE`（MDE 由 pilot 冻结） | `E_cond` 与 `E_work` 的 CI 均覆盖或低于 1 → 记 `NEGATIVE`，不发布任何 speedup headline |
| **覆盖率** | `w` 与 `r` 均被披露，conditional 结论始终与 `w` 同时出现 | 只报 conditional 而缺 `w`/`E_work` → 判 `INVALID` |
| **质量** | 给出可信的损伤上界 X pp（基于 discordant rate 的功效计算） | Dense 自身翻转率 > 15% → 先修 harness |
| **系统行为** | S0/S4 在相邻 launch block 下可区分，fallback 率 < 40% | 否则只能写 `DESCRIPTIVE` |
| **组合** | Combined span ≡ Coding-only span | 否则停止合并 |
| **治理** | 每个 Gate 执行前均有明确用户授权与 pinned manifest | 无授权执行 → 结果不进入 disposition |

## 12. 允许与禁止的表述

### 12.1 允许的表述（在标注证据级别的前提下）

1. "V40 是一个 grounded coding admission policy，其数据面等价于本项目的 `R0 Raw+RoPE`。"（`verified-code`）
2. "V40 的选择器显式排除 assistant reasoning 与 tool-call 文本，只复用 `role == "tool"` 的消息。"（`verified-code`，`coding_reuse_policy.py:424-429,437`）
3. "V40 要求 token 在目标 prompt 中唯一出现、严格位于中部、长度 ∈ [128, 4096]。"（`verified-code`）
4. "V40 的数据面机械校验（token 相等 / 边界 / 完整覆盖 / 部分拷贝检测）是健全的，失败一律 dense。"（`verified-code` + `verified-local` 66/23 passed）
5. "在固定 image 内，policy/selector/KVCOMM core/radix 面 66 passed；bridge adapter 8 passed；branch scope 3 passed。"（`verified-local`）
6. "V40 的路径失效判定对多种常见写工具漏检，且同 group 内读写混合会被判为成功只读证据。"（`verified-local`，§5.1/§5.2）
7. "V44 报告 Dense 3/12、General 3/12、V40 4/12，TTFT 357.6/335.7/327.5 ms —— **external unverified claim**，raw 数据不在 Git 中。"
8. "three-method development cohort 报告 Dense 6/12、V40 4/12，median TTFT 295.5 → 258.3 ms —— **external unverified claim**，且与 V44 cohort 任务集与 Dense 基线均不同，不可合并。"
9. "该 −16.7 pp 差值不是稳定的因果损伤估计（分支自身 summarizer 已如此声明）。"（`verified-code`，`summarize_three_method_coding_benchmark.py:257`）
10. "本项目 Phase7 在 chunk4096 下判定同族 R0 机制为 `NEGATIVE`。"（`verified`，`docs:research/phase_reports/PHASE7_INTEGRATED_EVALUATION_REPORT.md` §1.3）
11. "C40 与该 R0 共用同一恢复 primitive、只增加一个 selector，因此在同环境下**不得默认**性能转正；其速度结论必须由 `r` / `w` / `C_selector` / `E_cond` / `E_work` 与 span-matched R0 对照共同支撑，且 CI 用 restart-level cluster bootstrap。"（`derived`，§8.3.1、§9.10b）
12. "`0ab4fc942` 与 `e44ce40dc` 既不在本地 object 库（`git cat-file -t` → `Not a valid object name`），也不是 origin 上任何 ref 的 tip（在线 `git ls-remote origin` 全量枚举）。"（`verified-local`，§2.5.1）

### 12.2 明确禁止的表述

| # | 禁止 | 原因 |
| --- | --- | --- |
| 1 | "V40 是 prefetch" / "V40 的收益来自预取" | 分支无 `kvcomm_prefetch`；分支自述为 pure KV-reuse（§3.5） |
| 2 | "V40 实现了 KVCOMM" / "V40 是 KVCOMM 的工程化" | 无 base KV、无 ΔK/ΔV、无 anchor、无插值、无 entropy gate（§3.5、§4.6） |
| 3 | "V40 是 CacheBlend selective repair" / "V40 做了 KV repair" | 无 HKVD 打分、无 selected-token 逐层重算（§4.4） |
| 4 | "V40 是新的恢复 primitive（R6/L0）" | 数据面与 R0 数学等价（§4.2） |
| 5 | "token 完全相同所以 KV 也完全相同 / 复用是无损的" | 左上下文不同 ⇒ K/V 不同；RoPE 只修位置（§3.1） |
| 6 | 把 V44 cohort 与 three-method cohort 的数字并表 | 任务集与 Dense 基线不同、结论方向相反（§6.2） |
| 7 | "V40 相对 Dense 有 X pp 的准确率损伤" | 分支声称 Dense repeats 也失败，但 raw 不可得；即使该声明属实，单次 −16.7 pp 也不是稳定因果估计（§6.3） |
| 8 | "V40 优于 / 劣于 KVCOMM 164/225 或 CacheBlend 169/225" | model/prompt/order/limits/engine 全不同（§6.5） |
| 9 | "合并后 113 tests 通过证明组合可行" | `0ab4fc942`/`e44ce40dc` **对象不存在**且不在 origin 任何 ref 上；`kvcomm_prefetch/`、`kvflow_integration/` 在被审查分支中也不存在。只能标 `external unverified claim`，不得作为 Gate 5 entry 依据（§2.5.1、§7.5） |
| 10 | 把 `V40_MOTIVATION_RESULT.json` 等外部 artifact 的数字写成"已验证" | 文件不在 Git，本环境不存在（§7.3） |
| 11 | "V40–V44 五代实验证明了方法成熟度" | 分支把 V41/V42/V43分别描述为未完成、infra canary和0/6协议失败，但raw均不可得；这些声明本身也不能作为已验证成熟度证据（§6.4） |
| 12 | 引用任何 speed 数字而不同时给出 chunk / max-prefill 配置 | Phase7 已证明 chunk 强耦合（§6.6） |
| 13 | "V40 已通过审查" | 本报告结论为 `NOT APPROVED AS-IS`（§1.2、§5） |
| 14 | "C40 加了 selector，所以性能会好于 Phase7 的 R0" | selector 只改变**哪些请求复用**，不改变**复用后每 token 的成本**；同数据面下不得默认转正（§8.3.1） |
| 15 | 只报 `E_cond`（eligible 子集）而不报 `E_work`、`w`、`C_selector` | 缺一即判 `INVALID`；`E_cond` 高而 `E_work` ≈ 1 是典型的低时间覆盖率结局（§8.3.1、§9.10b） |
| 16 | 把 `E_cond` 直接写成 "workload 加速 X 倍"，或用 `1/((1-f)+f/s)` 之类的 request-fraction × median-speedup 公式换算 | workload 数字只能来自 Dense-full 与 C40-full **各自真实执行完整、有状态请求流**后的 `E_work` ratio-of-sums；不得用eligible片段与另一条Dense trace拼接（§8.3.1） |
| 17 | 把臂 B / 臂 C 的差分称为 "selection 收益" 或 "survivorship 修正" | 二者请求集合与 span 完全相同，差分**只**度量 `overhead_selector_control`；survivorship 由 `w` 与 `E_work` 的构造消除，不由差分估计（§9.10b） |
| 18 | 用 median-of-ratios 合成 workload speedup，或对 formal request 做朴素 bootstrap | estimand 必须是 ratio-of-sums；CI 必须用 restart-level cluster bootstrap（§9.0、§9.19） |
| 19 | 用 restart-0 的未知 `sample_sd` 判定最终 `NEGATIVE`/`POSITIVE` | restart-0 只做 engineering screening；MDE 须由 ≥2 restart 的 pilot 估计后冻结（§9.19） |
| 20 | 声称"C40 在本硬件必然为负"或"必须换更大硬件才能有收益" | Phase 7 的 R0 `NEGATIVE` 只是**不利先验**，须由 pilot 检验；大模型/长 span 是条件性扩展路径，不是结论（§8.3） |
| 21 | 写成"任何 V40 实验都无法复现" | 准确表述是"**历史 end-to-end 结果**不可复现"；focused/unit 面已实测 `66/8/3/12/8/23` 全通过（§7.6） |
| 22 | 声称任何 Gate（含 Gate 0）已获授权，或引用已撤回的 `≤62 starts / ≤28.2 GPUh` | 当前全部为 `PENDING USER AUTHORIZATION`；该上界已撤回（§11.1、§11.10） |
| 23 | 把 middle-span 集成写成"R0 runtime 零改动即可支撑 C40" | `xs:approx_kv/runtime.py:441-442,449-467` 只支持连续 span，中部岛会被判 `prefix_gap` 整体 dense（§10.3.1） |
| 24 | 因为 audit 脚本存在，就把 V41 deadlock / V40A2 failure / V43 0/6 等**运行结果**标为 `verified-code` | 源码只能验证 schema、判据与冻结常量；运行结果一律 `external unverified claim`（§6.4） |
| 25 | 预设 "cross-context 的 top-1 一致率必须 < 100%" | 跨上下文由 prefix hash 与 `rope_delta≠0` 断言保证；`top-1=100%` 完全可能偶然发生，不能反推设计有误（§9.6） |
| 26 | 引用 `integration/coding-aware-prefetch-v2` 或 `research/prefetch-p8-async-20260722` 作为已存在的分支 | origin 上不存在这两个 ref；只有 stale 的 `integration/coding-aware-prefetch @ d4a7ec132` 与 `research/prefetch @ fa86f8f16`（§2.5.1） |

---

## 13. Artifact 与索引

### 13.1 被审查分支的关键文件（`v40:` 前缀）

| 文件 | 行数 | 角色 |
| --- | ---: | --- |
| `KVFLOW.md` | 164 | 分支入口文档；含 V44 表（`:82-95`）与 merge readiness（`:117-135`） |
| `docs/kvflow/ARCHITECTURE.md` | 156 | ownership / composition 契约 |
| `docs/kvflow/CODING_AWARE_V40_REVIEW_REQUEST_20260729.md` | 144 | 审查请求；required invariants 在 `:66-72`；证据在 `:109-116` |
| `docs/kvflow/COLLABORATOR_QUICKSTART_20260729.md` | 146 | 四模式说明；dead flag 出现在 `:94` |
| `benchmark/multi_workflow/coding_reuse_policy.py` | 974 | **V40 选择器**；关键行见 §3.2、§5.1、§5.2 |
| `benchmark/multi_workflow/bridge_reuse_litellm_model.py` | 912 | **请求适配器**；V40 分支 `:486-521`；sidecar `:309-382` |
| `python/sglang/srt/mem_cache/kvcomm_exact.py` | 1169 | **实际执行控制器**；`copy_into_request` `:988` |
| `python/sglang/srt/mem_cache/kvcomm/{store,manager,transfer,radix_backend,types,config}.py` | 1096 | 数据面（**不得移植到 `xs:`**） |
| `python/sglang/srt/mem_cache/coding_aware/policy.py` | 126 | **未接线 seam**（B-03） |
| `python/sglang/srt/managers/scheduler.py:792-841` | — | canary 初始化与硬 guard |
| `python/sglang/srt/mem_cache/radix_cache.py:435-447,488-490,694` | — | attach（`:435-440`）/ prefix limit（`:441-447`）/ `cache_finished_req`（`:488-490`）/ 第三处 controller hook（`:694`） |
| `python/sglang/srt/managers/schedule_policy.py:608-612`、`schedule_batch.py:1483-1504` | — | copy 触发点 |
| `tools/check_kvflow_branch_scope.py` | 69 | branch scope 守卫 |

### 13.2 历史实验 artifact（全部依赖仓库外路径）

| 文件 | 行数 | 外部依赖（行号） |
| --- | ---: | --- |
| `motivate_v40_grounded_observation_island.py` | 400 | `:34` ARTIFACTS、`:44-45` tokenizer |
| `run_v40a_grounded_observation_canary.py` | 330 | `:22`、`:28` PYTHON、`:29-37` MOTIVATION/AUDIT |
| `run_v40a2_grounded_observation_canary.py` | 271 | `:21`、`:24-33` |
| `run_v40a3_short_grounded_observation_canary.py` | 276 | `:21`、`:24-33` |
| `audit_v40a2_timeout_failure.py` | 163 | `:19-21`；状态串 `:98`；`"prefetch": False` `:147` |
| `run_v40_repobench_control.py` | 445 | `:39` ROOT、`:40-45` MODEL、`:46-53` workload/output |
| `run_v41_v40_independent_campaign.py` | 456 | `:26-36`；`TASKS` `:40-49`；`SELECTION_SHA256` `:50-52`；`CACHEBLEND_DAMAGE_RATE` `:54` |
| `audit_v41_capacity_deadlock.py` | 221 | `:13-14`；`FAILED` `:15-18`；`MAX_NEW_TOKENS` `:21`；`KV_CAPACITY_TOKENS` `:22` |
| `run_v42_host_residency_infra_canary.py` | 274 | `:22-27` |
| `run_v43_new_verified_v40_campaign.py` | 697 | `:31-40`；`TASKS` `:100-107`；`SELECTION_SHA256` `:108-110` |
| `audit_v43_call_budget_collapse.py` | 298 | `:14-15`；`STEP_LIMIT` `:33`；`SHARED_CALLS` `:34`；`BRANCH_REQUEST_INDEX` `:35` |
| `run_v44_dense_sensitive_v40_campaign.py` | 663 | `:27-34`；`STEP_LIMIT` `:40`；`TASKS` `:41-52`；`SELECTION_SHA256` `:55-58`；`DENSE_PASS_SENSITIVITY_MIN` `:60` |
| `summarize_v44_schema_compat.py` | 159 | 无（CLI 路径） |
| `register_three_method_coding_benchmark.py` | 319 | `:29`、`:31-33`、`:41-43` |
| `prepare_three_method_swe_subset.py` | 136 | `:13-16` |
| `summarize_three_method_coding_benchmark.py` | 429 | 无（CLI 路径）；稳定性声明 `:257` |
| `summarize_kvcomm_repobench.py` | 66 | 无（CLI 路径）；KVCOMM 配置 `:34-37` |
| `swebench_verified_{bridge,complex,medium}_v1.json` | 285/180/329 | 各文件 `:10` `local_snapshot` |

### 13.3 本项目底座关键位置（`xs:` 前缀）

| 功能 | 位置 |
| --- | --- |
| 恢复模式枚举 | `python/sglang/srt/mem_cache/approx_kv/types.py:18-20` |
| R0 copy + RoPE | `approx_kv/radix_backend.py:196`（`copy_and_rotate`）、`:213`、`:217-221`、`:240`（per-layer）、`:325`/`:340`（rotary kernel）、`:380` |
| rope_delta 计算 | `approx_kv/runtime.py:537` |
| plan 执行 | `approx_kv/transfer.py:84`（`execute_reuse_plan`） |
| 请求级入口 | `approx_kv/runtime.py:652`（`restore_request_prefix`）、`:415`、`:566` |
| EPIC 插件 | `approx_kv/epic_plugin.py`、`epic_runtime.py`；k 值集合 `approx_kv/config.py:37-38` |
| prefix 保护（Phase6 A） | `approx_kv/runtime.py:83-114`；commit `af81934e4` |
| SWA lock metadata（B） | `approx_kv/runtime.py:107-114`；commit `db2d18ff0` |
| stale victim（C） | `cross_store/allocator.py:180-194`；commit `3379e6699` |
| provisional slot（D） | `approx_kv/runtime.py:37`/`:69`；`schedule_batch.py:1220,2257`；`scheduler.py:3045,4090`；commit `391bb8990` + `15634baf6` |
| fallback taxonomy（E） | `approx_kv/runtime.py:449-466`（`5e47904ec`）、`:589-597` |
| object graph（F） | `cross_store/object_graph.py:18,38,44,51,68,88,97` |
| segment store | `approx_kv/store.py:99,146,261,271,276,299,304,326,358,400,413` |
| cross-store 协调 | `cross_store/coordinator.py:16,34,51,70,107`；`allocator.py:57,70`；`policy.py:22`；`class_order.py:3` |
| 遥测 | `observability/metrics_collector.py:1951-2110`（17 项）；`approx_kv/manager.py:393,401,409,426,444,471,483,497,504,586` |
| 环境开关 | `approx_kv/config.py:122-162`（14 个 `SGLANG_APPROX_KV_*`） |
| 测试 | `test/registered/unit/mem_cache/{test_cross_store_substrate,test_approx_kv_runtime,test_approx_kv_core,test_epic_leadingk,test_approx_kv_hicache_backend,test_cache_policy,test_approx_kv_cuda}.py` |

### 13.4 本项目权威文档（`docs:` 前缀）

| 文档 | 用途 |
| --- | --- |
| `PROJECT.md` | 项目事实、决策与约束的固定事实来源 |
| `HANDOFF.md` | 当前状态快照（2026-07-29T02:18:08-07:00 已记录 V40 `NOT APPROVED AS-IS`） |
| `TRACKING.md` | append-only 时间线 |
| `research/RESEARCH_SYNTHESIS.md` | 系统 thesis、KVFlow/KVCOMM 边界、AST 职责、canonical base 定义 |
| `research/KVCOMM_SGLANG_FEASIBILITY_REPORT.md` | faithful KVCOMM 的九项构成、P0/P1/P2 blocker、复用边界 |
| `research/phase_reports/PHASE4_RECOVERY_METHODS_REPORT.md` | R0–R5 定义与 corrected 数值 |
| `research/phase_reports/PHASE5_WORKFLOW_SCHEDULING_REPORT.md` | S0–S4 / P0–P3 定义与 CL3 修正 |
| `research/phase_reports/PHASE6_CROSS_STORE_SUBSTRATE_REPORT.md` | 底座与不变量 |
| `research/phase_reports/PHASE7_INTEGRATED_EVALUATION_REPORT.md` | R0 = `NEGATIVE`、chunk 混淆、四本账、治理链 |
| `research/phase_reports/PHASE4_TO_PHASE7_SUMMARY.md` | 六次方法论反转与决策规则 |
| `research/YU_GUOFAN_BRANCH_REVIEW.md` | 更早一次对同一合作者工作线的审查 |

### 13.5 复现命令索引

> **R12 约定**：除 `git` provenance 查询（只读元数据操作）外，**所有测试与代码执行类复现命令一律在固定 Docker image 内、以 `:ro` 挂载执行**。本报告不把宿主机 Python 的直接执行作为证据。

```bash
cd /home/chris/Workspaces/kvcache-research/worktrees/coding-aware-v40-prefetch
ROOT=$PWD
IMG=ghcr.io/ccdd2023/sglang@sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781
DK=(docker run --rm -i --user 1000:1000
    -v "$ROOT:/w:ro" -w /w -e HOME=/tmp "$IMG")

# 1) 远端tip复核（网络只读，不更新本地refs）
git ls-remote origin \
  refs/heads/main \
  refs/heads/research/cross-store-substrate \
  refs/heads/review/coding-aware-v40-prefetch-20260729

# 下列本地计数要求对应对象已存在；若需同步对象，`git fetch`会更新本地
# metadata/refs，不能称为只读验证，应作为单独的显式准备步骤记录。
git rev-parse origin/main    # bd47ec97ff7a2881f9bb0316a4a657000a50c020
test "$(git rev-parse HEAD)" = \
     "$(git rev-parse origin/review/coding-aware-v40-prefetch-20260729)"
git rev-list --count 3343a79466aa714d34a14d08d3929f7953a47212..origin/main   # 4945
git rev-list --count 3343a79466aa714d34a14d08d3929f7953a47212..origin/research/cross-store-substrate  # 4786
git rev-list --left-right --count origin/main...origin/research/cross-store-substrate                # 289  130
git rev-list --count 3343a79466aa714d34a14d08d3929f7953a47212..13671eb708da689137a654946b0d34ba924efb29  # 82
git diff --shortstat c16bfbb8e8cc83a8b23858808f52833be9091101..HEAD          # 191/52549/220
git ls-files benchmark/multi_workflow | wc -l                                # 163
git ls-files | grep -i prefetch                                              # 空
"${DK[@]}" python tools/check_kvflow_branch_scope.py \
  --role coding --base c16bfbb8e8cc83a8b23858808f52833be9091101

# 1b) 不可获得引用的三层复核（§2.5.1）
git rev-parse --verify research/prefetch-p8-async-20260722   # fatal: Needed a single revision
git cat-file -t 0ab4fc942                                    # fatal: Not a valid object name
git cat-file -t e44ce40dc                                    # fatal: Not a valid object name
git ls-remote origin | grep -iE 'prefetch|integration|coding-aware-lossy'
#   d4a7ec132...  refs/heads/integration/coding-aware-prefetch   (stale, 非 v2)
#   fa86f8f16...  refs/heads/research/prefetch                   (非 e44ce40dc)
#   a580c1498...  refs/heads/research/coding-aware-lossy         (落后 review HEAD 71 commits)

# 2) P0 复现（固定 image、只读挂载，完整脚本见 §5.1.3）
#    差分测试若需真实 mutation：--tmpfs /scratch，容器内 cp -a 后再改，见 §9.1.4

# 3) Docker 测试面（见 §7.2；12-test selector 见 §7.2 代码块）

# 4) 底座对照
cd /home/chris/Workspaces/kvcache-research/worktrees/cross-store-substrate
git show -s --format='%H %ci %s' 0206f17b4255e4b248dafaaeb943be57428dae2f
git show -s --format='%H %ci %s' 81405f4278b034911bc613c4ee17c79d15ee8f35
```

---

### 13.6 外部 raw / ref 获取与 SHA 导入清单（Track D）

当前本环境缺失的全部外部输入，以及"导入即可解除哪个阻塞"的对应关系。**每一项在导入时必须同时提供 sha256，并登记进 `RESULT_MANIFEST`；未登记 sha256 的输入不得进入任何 disposition。**

#### 13.6.1 Git ref / commit（解除 composition 阻塞）

| 需要的对象 | 用途 | 解除的阻塞 | 导入方式与校验 |
| --- | --- | --- | --- |
| `integration/coding-aware-prefetch-v2` 的实际 commit（文档称 `0ab4fc942`） | 复现 113 composition tests | §7.5、Gate 5 entry | 推送到 `origin` 或提供 bundle；导入后 `git cat-file -t <sha>` 必须为 `commit`，并记录完整 40 位 SHA |
| prefetch tip（文档称 `e44ce40dc`）与分支 `research/prefetch-p8-async-20260722` | 四模式 / prefetch composition | §7.5、Gate 5 | 同上；同时提供该分支与 `kvflow/shared-core` 的 merge-base |
| `research/coding-aware-lossy` 的实际冻结点（文档称 `525a03c6b`） | 澄清"frozen from" 的准确性 | §2.5.1 表格第 3 行 | 把 origin 上的 `research/coding-aware-lossy` 推进到该点，或说明其为 review 分支独有 |

导入后必须重跑并记录：`git ls-remote origin`、`git merge-base --is-ancestor`、以及 §7.5 的三层复核。

#### 13.6.2 实验 raw artifact（解除证据阻塞）

| 需要的文件（作者机路径） | 内容 | 解除的阻塞 | 必须随附 |
| --- | --- | --- | --- |
| `…/impactkv_v40_grounded_observation_motivation_20260728/V40_MOTIVATION_RESULT.json` | selector opportunity 分布；V40A2/V40A3 的任务选择依据 | §7.3 的 2 个 failing 测试；§6.4 的 V40 motivation 行 | sha256、生成它的 code pin、tokenizer 路径与 revision |
| `…/impactkv_v40a{,2,3}_*/V40A*_RESULT.json`、`V40A2_INFRA_FAILURE.json` | 三次 canary 的实际结果 | §6.4 V40A/V40A2/V40A3 行 | sha256 + 完整 server argv + 环境变量快照 |
| `…/impactkv_v41_v40_independent_20260728/V41_CAPACITY_DEADLOCK_AUDIT.json` | V41 deadlock 的实际证据 | §6.4 V41 行（当前 `external unverified claim`） | sha256 + central log |
| `…/impactkv_v42_host_residency_canary_20260728/V42_RESULT.json` | V42 infra canary | §6.4 V42 行 | sha256 |
| `…/impactkv_v43_new_verified_v40_20260728/V43_RESULT.json`、`V43_CALL_BUDGET_COLLAPSE_AUDIT.json` | V43 的 0/6 说法 | §6.4 V43 行 | sha256 + 六个 task 的 trajectory |
| `…/impactkv_v44_dense_sensitive_v40_20260728/V44_RESULT.json` | Cohort A 全部数字 | §6.2 Cohort A | sha256 + 每 task 的 official resolved 判定与 harness 版本 |
| `…/impactkv_three_method_coding_benchmark_20260728/THREE_METHOD_AUDIT.{json,md}` + `repobench-p/**` | Cohort B 全部数字 + RepoBench-P | §6.2 Cohort B、§6.3 | sha256 + workload JSON + 模型 snapshot 标识 |
| KVCOMM / CacheBlend 225-task 基线的 raw | 若要做任何跨方法讨论 | §6.5 | sha256 + 完整 harness 与 prompt 拓扑说明 |

#### 13.6.3 配置元数据（**即使没有 raw 也应立即提供**）

以下缺失项当前直接阻碍对 Cohort A/B 数字的解读，且提供成本很低：

1. 每次实验的 `chunked_prefill_size` / `max_prefill_tokens`（**最高优先级**，见 §6.6）；
2. `page_size`、`tp`/`pp`、eviction policy、HiCache 开关；
3. 模型与 tokenizer 的精确 revision（AWQ 量化配置）；
4. server argv 与全部 `SGLANG_*` 环境变量快照；
5. warm-up 请求数、formal repeats 数、arm 顺序；
6. 每个 arm 的 eligibility 计数与 skip_reason 直方图（用于事后估 `r` 与 `w`）。

#### 13.6.4 导入验收

导入完成后，Track D 的 Exit 条件为：
- 所有导入文件的 sha256 已登记进 `RESULT_MANIFEST`，`--check` 通过；
- §7.3 的 2 个 failing 测试转为 pass（或被改写为不依赖绝对路径的等价测试）；
- §7.5 的三层复核对 `0ab4fc942` / `e44ce40dc` 返回"存在"；
- §6.2 / §6.4 中被标 `external unverified claim` 的条目，逐条注明"已导入并校验"或"仍缺失"。

**Track D 不占用当前 GPU 预算，可与 Track A 并行推进。**

## 14. 结论

1. **方法**：`C40 = G40 grounded coding selector × R0 Raw+RoPE executor`。**恢复 primitive 就是 R0 Raw+RoPE**（不是新公式）；V40 的新增点主要在 **admission / selection / invalidation policy**。它**不是** KVCOMM reconstruction、**不是** selected-token repair（CacheBlend/Cache-Craft/CacheTune）、**不是** prefetch。

2. **最像谁**：恢复 primitive 与本项目 `R0` 及 EPIC `k=0` 端点等价；问题设定最像 Prompt Cache / PIC 家族的 non-prefix modular reuse；唯一可能有增量的是"以 repository 事件为依据的准入 gate"。

3. **审查结论**：**`NOT APPROVED AS-IS`**。两类 P0 已在固定 Docker image 内、只读挂载下复现：常见写工具（`cat >`、`echo >`、`perl -pi`、`dd`、`truncate`、`git mv`、`rsync`）全部漏检；同 group 内读写混合被判为成功只读证据。这直接违反分支自己写的两条 required invariants。

4. **P0 影响的准确界定**：target prompt 中**仍然包含**同一段旧 observation 文本，Dense 也会看到它；token identity 与机械校验**仍然通过**。因此这**不是** old-KV-配-new-token，**不是** data corruption。它违反的是 **freshness / abstention policy**：本应放弃复用的、语义已过时的 historical observation 被放进了 lossy reuse 路径，而该 gate 正是分支论证"复用安全"的唯一依据（§5.1.5）。

5. **修法**：不要继续堆命令正则。由 tool wrapper 提供结构化 `read_paths` / `write_paths`、repo/worktree generation、source path content hash；未知效应 fail closed；补齐对抗矩阵、property 测试与 Docker 内可写副本上的差分测试（§5.6、§9.1）。

6. **证据**：所有关键运行结果均为 `external unverified claim`——包括 V41 deadlock、V40A2 failure、V43 的 0/6、Cohort A（V44）与 Cohort B（three-method）的全部数字。源码**只能**验证 audit 的 schema、判据与冻结常量，**不得**据此把运行结论升级为 `verified-code`（§6.4）。两套 12-task cohort 任务集与 Dense 基线不同，不可合并；与 KVCOMM/CacheBlend 的 225-task 基线不可排名。

7. **性能先验（不是结论）**：Phase 7 已在同 image / 同模型 / `chunk4096` 下判定 R0 为 `NEGATIVE`（`0.772–0.936`）。C40 沿用**同一恢复 primitive**，selector 只影响"哪些请求复用"、不影响"复用后每 token 的成本"，因此 **C40 不得默认性能转正**；但这只是一个**不利先验**，是否成立**必须由 pilot 检验**，不得预先断言"必负"或"必须换硬件"。

8. **合法 estimand（强制）**：任何 C40 速度结论必须同时给出 —— `r`（计数 eligibility rate + Wilson CI + skip_reason 直方图）、**`w`（time-weighted coverage，dense 基线时间口径）**、`C_selector`（在**全部**请求上测得的 selector 开销）、**`E_cond`**（eligible 子集的 paired ratio-of-sums）、**`E_work`**（Dense-full 与 C40-full 各自真实执行完整、有状态请求流后的 paired ratio-of-sums）。CI 一律用 **restart-level cluster bootstrap**；**禁止** median-of-ratios 合成，**禁止**用 request-fraction × median-speedup 公式或eligible-only片段拼接workload结果。臂 B/C 同 span 的差分只度量 **`overhead_selector_control`**，**不得**称为 selection 收益；survivorship 不用差分估计，而由 `w` 与完整trace `E_work` 结构性披露（§8.3.1、§9.10b）。

9. **统计计划**：restart-0 **只做 engineering screening**，不得用未知 `sample_sd` 判定最终 `NEGATIVE`；必须先跑 **≥ 2 独立 restart 的 pilot** 估方差与 discordance，再冻结 MDE 与 confirmatory plan；确认性 cell **默认 4 restarts**（`2` 仅限 feasibility 且须标注功效不足）；task accuracy 用 **paired McNemar** 且先估 discordant rate；formal requests **不是**独立样本（§9.19）。

10. **集成**：绝不整分支 merge；绝不把分支的旧 `kvcomm store/manager/scheduler` 带回底座（会回归 Phase 6 的**六项**正确性修复，§10.2）。在当前 cross-store 上重实现最小 payload：结构化 selector + provenance、sidecar/adapter。copy/RoPE backend 与现有 cross-store lifecycle **核心语义可复用**，但需增加 C40 provenance/approx-depth 薄适配；**request execution seam 必须新写**。现有 `xs:approx_kv/runtime.py:441-442,449-467` 只支持"从 `exact_length` 连续开始"的 span，中部岛会被判 `prefix_gap` 整体 dense；需新增 `DENSE_PREFIX → COPY_READY → DENSE_SUFFIX` 状态机、consume/produce双角色生命周期与scheduler接线，同时保留 prefix ownership、provisional ownership、stale victim、SWA metadata、exclusive fallback/不双计、object graph 六项不变量（§10.3.1）。

11. **命名**：不引入新恢复 primitive。系统 candidate ID = `C40 = G40 × R0`；正交维度 `S0/S4`、`P0/P1`。

12. **composition 证据**：分支自述的 integration-v2 **113 tests** 依赖 `0ab4fc942` 与 `e44ce40dc`。二者经三层复核（ref / object 库 / 在线 `git ls-remote origin`）**均不存在**；origin 上只有 stale 的 `integration/coding-aware-prefetch @ d4a7ec132` 与 `research/prefetch @ fa86f8f16`，分支内也无 `kvcomm_prefetch/` 与 `kvflow_integration/`。该 113 tests 只能标 `external unverified claim`，不得作为组合可行性证据或 Gate 5 entry 依据（§2.5.1、§7.5）。

13. **可复现性的准确表述**：**历史 end-to-end 实验结果不可复现**（硬编码 `/home/gfy/**`、依赖未锁、raw 不在 Git）；但 **focused / unit 测试面完全可复现** —— 本报告已在固定 image 内实测 `66 / 8 / 3 / 12 / 8 / 23` 全部通过，另有 2 个外部依赖导致的 failure（§7.2、§7.3、§7.6）。依赖告警证明的是"环境未锁"，**不**证明"冲突不可避免"；修复方式是 `requirements.lock` + 专用 Docker layer。沿用当前base时验收为`pip check`相对已版本化5项baseline无新增冲突；只有重建clean base并固定新digest时才要求零输出（§7.4、§9.18.1）。

14. **顺序**：exact cache → 受控 C40 reconstruction → dense fallback。不得跳步。

15. **治理与授权状态**：不改写 Phase 4–7；为 C40 创建新的 versioned plan/manifest，走与 Phase 7 相同的"预注册 → pin → 授权 → 执行 → consolidate → 双模型 review → disposition"链路。
    **当前 `AUTHORIZATION = PENDING USER AUTHORIZATION`，包括 Gate 0 在内没有任何 Gate 已被授权。** 本报告能给出的唯一结论是：**推荐的下一步是 Track A（zero-GPU 的代码与 provenance 修复）**。此前版本给出的"`≤ 62 starts / ≤ 28.2 GPUh` 保守上界"**予以撤回**；Track B 的建议 hard cap 为 `≤ 8 starts / ≤ 2 GPUh`（`estimate`，须经 1-start 校准后冻结），Gate 3 的 16-start 扩展路径与 SWE-bench 扩样各需**单独二次授权**，质量 campaign 以 **task-runs** 单列计量（§11）。

---

*本报告只创建/修改自身一个文件，未修改被审查分支或本仓库其它文件，未提交、未推送。除只读 `git` 元数据查询外，全部 `verified-local` 结论均在固定 Docker image `sha256:0be6e16e…` 内、以 `:ro` 挂载复现，命令见 §13.5。*
