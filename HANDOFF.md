# 会话交接

最后更新：2026-07-30T11:17:00-07:00

## 新会话启动顺序

1. 阅读本文件，获取当前状态和下一步。
2. 阅读 `PROJECT.md`，确认项目事实、决策和约束。
3. 阅读 `IMPLEMENTATION_PLAN_LATEST.md`，获取当前可执行计划。
4. 阅读 `TODO_LOCAL.txt`，获取可跨session接续的完整待办。
5. 阅读 `TRACKING.md` 的最新记录，了解最近讨论过程。
6. 开始工作后持续维护上述文件，不把重要信息只留在聊天中。

## 当前快照

### 2026-07-30T11:07:48-07:00 Phase7.5计划已reviewed，等待用户授权

- Plan=`IMPLEMENTATION_PLAN_PHASE7_5_C40.md@V1-r13`；
  status=`Reviewed Candidate / PENDING USER AUTHORIZATION`。
- 最终plan SHA=`b894e276960b72c1cf7963ca7f5957f89f62471b11eb006548fc867d7f28bedf`。
- 最终review=`PASS / READY_FOR_PLAN_FREEZE`，open P0/P1/P2=`0/0/0`，
  F-01..F-11全部CLOSED。
- review artifact：
  `evidence/review/plan-review-phase7_5-c40-v1-r13.json`，
  self=`63bf3b83...`，绑定最终plan SHA。
- G0q artifact：
  `evidence/review/c40-quarantine-manifest.json`，
  status=`AVAILABLE`，contains_source_text=false，self=`a0fe94ae...`。
- 主发布commit=`ce3f786699cf813cfb84d8191752a0dbf236cd48`，
  已推送至`origin/project/code-agent-kvcache-docs`。
- Phase7状态不变：engineering VALID、R0 NEGATIVE、W
  INCONCLUSIVE/DESCRIPTIVE、publication READY WITH CAVEATS。
- 未创建Phase7.5 branch，未修改implementation repo，未运行Docker测试或GPU。
- 下一步：等待用户显式授权branch creation；之后严格按
  `G1a→G0b→G1b→G2...`推进，每个Gate/Track独立授权。

### 2026-07-30T10:32:19-07:00 Phase7.5 C40计划第十二轮review = PASS（0/0/0）

- 计划文件：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表 `V1-r13`，5743行，
  sha256 `25cbac98c65c6b67e6d2de23c0cc33e28d7c6f6b3ca60b13cd49a6a3b3770a42`。
- 第十二轮只读全文件逐API + 跨章节 closure review 完成；**未改计划文件、
  未建branch、未跑测试/GPU**。底座固定 `worktrees/cross-store-substrate@0206f17b`（clean）。
- Verdict = **`PASS / READY_FOR_PLAN_FREEZE`**；open = **`0 P0 / 0 P1 / 0 P2`**。
  §23.1 的 review closure 硬条件（open P0/P1 = 0/0）已满足，且 P2 亦归零。
- 第十一轮唯一 P2 与三项观察全部 CLOSED：
  (a) §19.5 模板 `:4818` 新增 `ci_method=<bootstrap|t|conservative_of_both>`，
  与 §19.2b `:4718-4719`/`:4749` 闭合，新增 `TC-97` `:2294` 断言；
  (b) §8.6.3 步骤1 `:1904-1905` 的 B-2 行显式分 `clip flag=0/1`（后者转 CL-I）；
  (c) §18.6b `:4453-4458` 冻结 `c40_effective_task_rate` 与 `copy_coverage(task)`
  并规定与 `w_quality` 分列不互换，新增 `TC-99` `:2296`；
  (d) §14.1 拆出 `collector_capabilities`/`collector_security_opts` 并新增
  `ebpf_authority`（`BPF`+`PERFMON`，security_opts 空），新增 `TC-98` `:2295`。
- 全文件核验：TC-1..TC-99 共 99 条定义，无缺号/重号/悬空引用，§7.1 `:864`、
  WP5 `:3756`、G3 Exit `:5033` 口径统一；§19.5 已覆盖 §19.2b 全部强制字段；
  四态判定互斥穷尽；统计复算（`2856.9`/`2473.0`/`124-248-495`/`90`+`8` cells/
  `12+4` starts/Track B `6<=8`/人周 `8.25–12.25`、`4.00–8.00`）一致；
  126 个 `c40_*` 与 3 个 `p75_*` 标识符均有定义；全部 §x.y 引用可解析；
  底座行号抽样（`schedule_policy.py:435-438/1073/1126-1174/1158-1166`、
  `scheduler.py:2630/2745-2756/3017-3018/3025-3058/3068`、
  `schedule_batch.py:1080/1184`、`memory_pool.py:273`、`common.py:103-107/198-202`）
  全部属实。
- F 矩阵：**F-01..F-11 全部 CLOSED**（F-05 由 PARTIAL 转 CLOSED）。
- 非阻断观察（不计入 open）：§19.5 `:4823`/`:4829` 的 `E_work_pooled` 槽重复且
  `CI95` 续行现排在 `ci_method` 之下；`w_quality`/`eligible_task_rate` 只有名字
  无独立公式；TC-97/98/99 归在 WP5/G3 而对象属 WP8/WP0b/WP12（WP8 早于 G3，
  无循环）；§24.2 交付索引少列 4 个文件（已声明"见 §7.1"）；
  §14.2 绑定句未点名 security opts。
- **下一步**：(1) 产出 versioned `evidence/review/plan-review-*.json`
  （自哈希 + 绑定文档 sha256、记载 `open P0/P1 = 0/0`）；
  (2) 产出 G0q 的 `evidence/review/c40-quarantine-manifest.json`（隔离 reviewer）；
  **两份 artifact 齐备后**才可由主会话把状态升级为
  `Reviewed Candidate / PENDING USER AUTHORIZATION`，再申请 branch 授权。
- 授权边界不变：只授权计划编制/审阅/文档同步；branch、实现、Docker与GPU
  仍为`PENDING USER AUTHORIZATION`。

### 2026-07-30T10:01:48-07:00 Phase7.5 C40计划第十一轮review = PASS_WITH_P2

- 计划文件：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表 `V1-r12`，5730行，
  sha256 `a1a461b0ccec1b59787c26ade72ac772000d7df32301e5fed1dea7a2ef0c6b26`。
- 第十一轮只读全文件逐API + 跨章节 closure review 完成；**未改计划文件、
  未建branch、未跑测试/GPU**。底座固定 `worktrees/cross-store-substrate@0206f17b`（clean）。
- Verdict = `PASS_WITH_P2`；open = **`0 P0 / 0 P1 / 1 P2`**。
  §23.1 的 review closure 硬条件（open P0/P1 = 0/0）**已满足**；
  因仍有 1 项 P2，**不**宣告无条件 `0/0/0 READY_FOR_PLAN_FREEZE`。
- 第十轮 **1 P1 + 3 P2 全部 CLOSED**：
  (a) `CONTINUE + not-added` 已彻底清除——§7.4 `:1133` 改"保持底座
  `OTHER`/not-added 路径并 defer"，TC-37 `:2233` 改写为跨轮 owner 场景并与
  TC-45 `:2241` 统一，`:1839` 明令禁止自造 `CONTINUE+not-added`，
  底座 `scheduler.py:3025-3058` 的 not-added 清理与 `assert self.chunked_req
  is None` 均恢复可达；
  (b) CL-I 已完整复用 B-3 零 dense-prefix admission/copy 生命周期
  （`:1559-1565`/`:1660`/`:1763-1764`/`:1928-1931`/`:1938`/`:1957`/`:2082`，
  TC-89/TC-95）；
  (c) §19.5 模板 `:4820-4823` 已含 exact-hit / insert-suppressed /
  stash-suppressed / `final_radix_insert_policy`（TC-96）；
  (d) `rollback_for_deferral` `:1961-1966` 已复位
  `lifecycle_ever_entered`/`recovery_attempted`/`radix_insert_suppressed`
  并清 pending reason，与 §8.1.2 cleanup domain、§12.4 族1/族2、TC-46/90/92 一致。
- **唯一 open 项（P2-1，非阻断）**：§19.2b `:4736-4737` 把 "CI 方法" 列为
  四态判定强制同现项且"缺任一项判 `INVALID`"，并要求 n 小时同时报
  bootstrap 与 t 的 90% CI 取更保守者；但 §19.5 强制模板 `:4808` 只有单一
  `CI90 [L90,U90] (decision CI)` 槽、无 `ci_method`（`:4806` 的 `method=`
  属 `n_confirmatory` 样本量校正方法，`:4816` 的 `E_cond` 行反而写明了
  `cluster-bootstrap`）。默认 `n_min=4` 即小样本路径，逐字遵循模板会缺该必报项。
  最小修复：`mu_theta` 行补 `ci_method=<bootstrap|t|conservative_of_both>`
  并在 TC-96 断言。
- 非阻断观察（不计入 open）：步骤1 `:2121` 的 B-2 行未带 `CLIP=1 ⇒ CL-I` 旁注
  （已由"按 §8.6.2 分类"委派 `:1760-1764` 解决）；§18.6b `:4468` 的
  `c40_effective_task_rate` 未在 §13/§18 定义；§19.5 `:4811`/`:4817`
  的 `E_work_pooled` 槽重复。
- F 矩阵：F-01/F-02/**F-03**/F-04/**F-06**/F-07/F-08/F-09/**F-10**/F-11 = CLOSED；
  F-05 = PARTIAL（本轮 P2-1）。
- **下一步**：(1) 可选地补 `ci_method` 槽位清零 P2；(2) 产出 versioned
  `evidence/review/plan-review-*.json`（自哈希 + 绑定文档 sha256、记载
  `open P0/P1 = 0/0`）与 G0q 的 `c40-quarantine-manifest.json`；
  **两份 artifact 齐备后**才可由主会话把状态升级为
  `Reviewed Candidate / PENDING USER AUTHORIZATION`，再申请 branch 授权。
- 授权边界不变：只授权计划编制/审阅/文档同步；branch、实现、Docker与GPU
  仍为`PENDING USER AUTHORIZATION`。

### 2026-07-30T09:30:55-07:00 Phase7.5 C40计划第十轮review = FAIL

- 计划文件：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表 `V1-r11`，5723行，
  sha256 `8d824a3170427ab1b0f4b8a7304bbf5a9a129d595ecb194dd030154de48aa010`。
- 第十轮只读全文件逐API + 跨章节 closure review 完成；**未改计划文件、
  未建branch、未跑测试/GPU**。底座固定 `worktrees/cross-store-substrate@0206f17b`（clean）。
- Verdict = `FAIL / NOT READY`；open = `0 P0 / 1 P1 / 3 P2`。
- 第九轮 **2 P1 + 5 P2 全部 CLOSED**：admission 顺序已由零资源 PRE-FLIGHT
  （owner/gate/预计算 `AdmissionPlan` 全在 `STAGED` 完成，§8.6.3 步骤2 `:1916-1927`）
  + "commit 后到 append 禁止普通 return" `:1961-1963` + 步骤3 `:1990-1993`
  执行冻结 plan 闭合，§8.2 矩阵已删 `DENSE_PREFIX --admit_deferred-->` 边（TC-91）；
  两条 rollback 分支 `:1964-1978` 完整复位 skip/live/recovery_attempted/pending
  reason 并给出 postcondition（TC-92）；G0q 只产 reference signatures、
  branch-match 裁决在 G1b Exit（`:4941`/`:4952`/`:4993`，TC-94）；
  中间 stash 抑制新增 `p75_radix_stash_suppressed_chunks_total{arm,outcome}`
  与 §13.5.1 日志字段、§19.6 `INVALID` 判据（TC-93）；
  §12.4 族2 归属改看 sticky audit `:2891-2893`（TC-90）。
- **新 P1（阻断）**：**`CONTINUE + not-added` 语义只改了三分之二**。
  §8.6.2a `:1830`/`:1837`、§8.6.3 步骤2 `:1922-1925` 与 TC-45 `:2238` 已冻结
  "owner conflict 保持底座 `AddReqResult.OTHER`、不自造 CONTINUE"，但
  §7.4 `:1133` 与 TC-37 `:2230` 仍写"返回 `CONTINUE` 且 `added=false`"。
  后果：(a) TC-37 与 TC-45 对同一 owner-conflict 场景断言互斥，
  §20 G3 Exit 的"TC-1..TC-94 全绿"永不可达；
  (b) 底座 `scheduler.py:3025-3058` 的 not-added 清理只在 `res != CONTINUE`
  分支执行，返回 `CONTINUE` 会跳过 §8.6.2a 强制的 `cleanup_not_added_req(req)`
  且让 waiting 循环继续加请求（`assert self.chunked_req is None` `:3074` 风险）；
  (c) TC-37 的"同轮两个 forced-middle C40"在 r11 下不可复现——首个 owner 的
  `reserve_exclusive_chunked_owner()` 把 `rem_chunk_tokens=0` ⇒
  `budget_state()` 返回 `OTHER` ⇒ scheduler 立即 `break`，第二个请求本轮
  根本不会 staging；owner-conflict guard 只在 `has_chunked_req == True`
  （跨轮 owner）时可达。最小修复：§7.4 `:1133` 改"保持底座 `OTHER`"；
  TC-37 改写为跨轮 owner 场景并与 TC-45 统一口径。
- P2（3项）：(1) §8.6.1 `:1559` 冻结 `staging_case ∈ {B-3,B-4,CL-I}`，但步骤2 的
  `needs_forced_middle`/`projected_prefix_len`、步骤5 copy-hook 映射、
  `scheduling_prefix_len` `:1697` 与 §8.6.2 B 表均只覆盖 B-3/B-4，
  T23-6 `:4260` 要求的零 dense-prefix CL-I admission 路径无规格；
  (2) §19.5 强制汇报模板 `:4812-4814` 缺 stash-suppression 槽位，
  而 §19.6 `:4830` 把该字段缺失判 `INVALID`、§19.9 `:4877-4879` 要求按 arm 披露
  ⇒ 逐字遵循模板即自动 INVALID；
  (3) `rollback_for_deferral` `:1968-1972` 复位 `recovery_attempted` 却不复位
  sticky `lifecycle_ever_entered`（§8.6.1 `:1580`、TC-92 同样省略），
  在 "B-3 commit → 同轮 gate 拒绝 → 下轮判 `dense_ineligible`" 序列上
  §8.1.2 `:1339` 的 cleanup domain 与 TC-46 `:2239`"ineligible 不写 cleanup"冲突。
- F-01 由 PARTIAL 转 **CLOSED**；F-02/F-04/F-05/F-07/F-08/F-09/F-11 = CLOSED；
  F-03 = PARTIAL（P1）；F-06 = PARTIAL（P2-1）；F-10 = PARTIAL（P2-2、P2-3）。
- **下一步**：修订 1 P1（并清 3 P2）后做第十一轮 closure review；
  open P0/P1归零并产出versioned `plan-review-*.json`之前，
  **不得**升级状态、**不得**plan freeze、**不得**申请branch授权。
- 授权边界不变：只授权计划编制/审阅/文档同步；branch、实现、Docker与GPU
  仍为`PENDING USER AUTHORIZATION`。

### 2026-07-30T08:47:42-07:00 Phase7.5 C40计划第九轮review = FAIL

- 计划文件：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表 `V1-r10`，5702行，
  sha256 `d3e4610bb1203d5dd73b509f8dad2eb15014d4a070f9cf12456da668a4e4940b`。
- 第九轮只读全文件逐API + 跨章节 closure review 完成；**未改计划文件、
  未建branch、未跑测试/GPU**。底座固定 `worktrees/cross-store-substrate@0206f17b`（clean）。
- Verdict = `FAIL / NOT READY`；open = `0 P0 / 2 P1 / 5 P2`。
- 第八轮 **1 P1 + 5 P2 全部 CLOSED**：每臂 exact-Radix 插入策略已由
  §8.2.1 规则6 `:1397-1416` / §15.1 `:3403-3412` 冻结，并新增
  `p75_exact_prefix_hit_tokens_total{arm}` 与
  `p75_radix_insert_suppressed_requests_total{arm,outcome}`
  （§13.1 `:2977-2979`、§19.5 `:4796-4798`、§19.9 `:4859-4866`、TC-85/86/87）；
  approx-metadata 前提（§7.4 `:1129-1133`）、audit sticky schema（§8.6.1 `:1573-1578`）、
  `outcome_geometry` 映射（`:1556-1562`）、collector profiles（§14.1 含
  `ebpf_v1` 的 `BPF`/`PERFMON`）、族1/族2 的 `recovery_attempted` 判据均已落地。
- **新 P1-1（阻断）**：**admission-commit 与 forced-middle 钳制的顺序自相矛盾，
  两种读法都不可执行**。§8.6.3 步骤3 的 `needs_forced_middle` `:1977` 与
  owner-exclusion `:1982-1987` 都要求 `c40_state == DENSE_PREFIX`，而该状态只由
  步骤2 的 `c40_on_admission_commit` `:1928-1949` 建立（r10 把 lifecycle 入口
  移到 admission 成功之后）。底座 `schedule_policy.py:1123-1180` 中
  `set_extend_range` 与 `can_run_list.append` 相邻，因此：
  (a) commit 放在钳制**之后** ⇒ B-4 首轮谓词恒假 ⇒ 不钳制、不设 `new_chunked_req`
  ⇒ 必然越过 `target_start` ⇒ 恒触发 `c40_boundary_overrun` + 族4
  `engineering_valid=false`，primary 路径整体作废；
  (b) commit 放在钳制**之前**（与 §7.4 `:1138`、步骤5 `:2072` 的
  "B-3 → add_one_req pre-range hook" 一致）⇒ owner-exclusion 的
  `return AddReqResult.CONTINUE` `:1987` 正是步骤2 `:1950` 明令禁止的普通 return，
  且此时 consume lease 已 pin，违反步骤2 postcondition `:1953-1957`
  （`consume_lease is None`）与 §8.1.2 / §8.2 的 ADMISSION_DEFERRED 不变量。
  最小修复：冻结 commit 相对 `set_extend_range` 的位置；owner-exclusion 改为
  commit **之前**的 gate 并把谓词改用 `STAGED` + staging plan；
  §8.2 删除 `DENSE_PREFIX --admit_deferred--> ADMISSION_DEFERRED` 边；补 TC。
- **新 P1-2（阻断）**：**post-commit deferral / rollback 未复位
  `skip_radix_cache_insert`、`lifecycle_live`、`recovery_attempted`**。
  步骤2 `:1951-1957`、步骤3 `:1986`、步骤7 `:2160-2163` 只列资源，不含标志位；
  §8.2.1 规则5 `:1409` 只覆盖"pre-admission 失败 / `state=NONE` / pure retract"，
  不覆盖"已进入 lifecycle 后转 ADMISSION_DEFERRED"。后果：
  (1) flag 保持 True ⇒ `common.py:200` → `cache_finished_req(is_insert=False)`
  使该请求即使最终全 dense 也永不写 exact Radix，违反 §7.5 与 §15.1 冻结臂策略；
  (2) 下一次 commit `:1942-1943` 把泄漏值 latch 进
  `original_skip_radix_cache_insert`，污染永久化；
  (3) `radix_insert_suppressed` 只在 copy commit `:2131` 置位 ⇒
  `p75_radix_insert_suppressed_requests_total` 不记录该泄漏 ⇒
  §15.1/§19.9 的"臂策略漂移 ⇒ engineering invalid"判据检测不到；
  (4) `recovery_attempted` `:1929` 泄漏后若该请求终判 `dense_ineligible`，
  与 §12.4 族1 的 `recovery_attempted=false` 定义冲突，TC-90 两条规则互斥。
- P2（5项）：G0q Exit `:4915` / G1a Entry `:4933` / §20.1 `:5104` 要求裁决的
  alert 只能由 G1b 的 CR-3/CR-3b 在 branch diff 上产生（Exit 空洞或阻断链）；
  `reserve_exclusive_chunked_owner()` `:2003` 使 owner-conflict 的 `CONTINUE`
  分支不可达（底座先返回 `OTHER`），与 §8.6.2a `:1823-1830` 与 TC-37/TC-45 冲突；
  中间 chunk 抑制 `cache_unfinished_req`（`common.py:104` ← `scheduler.py:2630`）
  形成**第二类臂间系统差异**且无度量/披露，§8.2 `:1386` 的"最终 exact Radix 状态
  与 never-eligible 一致"与 `radix_cache.py:551-605` 的分块建树不符；
  §12.4 族2 `:2879` 仍把 attempted fallback 绑定 `DENSE_ISLAND_FALLBACK`，
  与步骤2 `:1960-1962` 的 `state=NONE` 路径冲突；新 per-arm Radix 指标缺
  §13.5.1 `:3101-3120` 日志字段与 §19.2b/§19.6 `:4804-4813` 的 `INVALID` 判据。
- F-02/F-03/F-05/F-06/F-07/F-08/F-09/F-11 = CLOSED；F-01 = PARTIAL（P1-1）；
  F-04 = PARTIAL（P2-1）；F-10 = PARTIAL（P1-2、P2-4）。
- **下一步**：修订 2 P1（并清 5 P2）后做第十轮 closure review；
  open P0/P1归零并产出versioned `plan-review-*.json`之前，
  **不得**升级状态、**不得**plan freeze、**不得**申请branch授权。
- 授权边界不变：只授权计划编制/审阅/文档同步；branch、实现、Docker与GPU
  仍为`PENDING USER AUTHORIZATION`。

### 2026-07-30T07:58:28-07:00 Phase7.5 C40计划第八轮review = FAIL

- 计划文件：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表 `V1-r8`，5622行，
  sha256 `f1feb6de2851cd63aa16004f9e117ea275d9f7dad6de3298b6afed5feca1aa9a`。
- 第八轮只读全文件逐API + 跨章节 closure review 完成；**未改计划文件、
  未建branch、未跑测试/GPU**。底座固定 `worktrees/cross-store-substrate@0206f17b`（clean）。
- Verdict = `FAIL / NOT READY`；open = `0 P0 / 1 P1 / 5 P2`。
- 第七轮 **2 P1 + 4 P2 全部 CLOSED**：pre-admission fallback 同轮 admit 已由
  §8.1.1 `:1300-1303` / §8.5 `:1486` / §8.6.3 步骤2 `:1937` / §8.2 矩阵 /
  TC-81 的 `c40_state=NONE` 规则闭合；skip-Radix seam 已由 §7.4 `:1129` /
  §8.6.3 步骤1 `:1890` / §8.6.4 `:2142` / TC-82 显式置位闭合；
  ToolEvent 层级、矩阵 defer 边、`geometry`/`suppress_produce_once` 字段、
  collector `collector_profiles` + `selected_profile_by_gate` 亦全部闭合。
- **新 P1（阻断）**：**exact Radix 插入抑制的臂间不对称未冻结、未度量，
  且证伪两条既有不变量**。§8.6.3 步骤1 对每个 B-3/B-4 请求置
  `skip_radix_cache_insert := true`，§8.2.1 规则2（`:1404-1406`）要求整个
  lifecycle（含 `DENSE_ISLAND_FALLBACK` 后的纯 dense 重算）不得解除；
  底座 `common.py:103-107` / `common.py:198` → `radix_cache.py:496-551` 表明
  该 flag 为真时请求 KV **永不进入 exact Radix**。后果：
  (1) §8.2 `:1386`“`DENSE_ISLAND_FALLBACK` 其后行为与‘从未 eligible’逐字段一致
  （可用差分测试断言）”为假，§7.5 `:1157`“exact 命中路径与 C40 关闭时逐位一致”
  同样被证伪；(2) 计划未冻结 `D0`/`E0`/`C40-D` 是否同样抑制插入
  （§15.1 `:3338-3341`、§19.4 Step 3 只说“仍执行完整 source lifecycle”），
  若对照臂正常插树而 C40 臂的全部 staged 请求（`c40_copied` 与
  `dense_attempted_fallback`）不插树，则 C40 臂在 W4a 滚动轨迹上逐请求丢失
  exact 前缀命中，该臂间系统性差异被静默折进 primary estimand
  `theta_j`/`E_cond`，§13 无 metric、§19.5/§19.6 无披露与 INVALID 判据、
  §19.4 臂C/臂D 差分无法分离；(3) 该 flag 无任何复位路径
  （`clear_target_staging()`+`state=NONE` 与底座 `reset_for_retract`
  `schedule_batch.py:1567-1601` 均不复位）。最小修复：限定 §8.2/§7.5 的等价性
  断言范围；在 §15.1/§19.4 冻结每臂 exact-Radix 插入策略；新增 per-arm
  `exact_prefix_hit_tokens` 与 `c40_radix_insert_suppressed_requests_total`
  并写入 §19.5/§19.9；定义复位语义并补 TC。
- P2（5项）：§7.4 约束4 `:1116` 与 §8.2.1 规则1 `:1404` 仍以“带 approx metadata
  恒为 True”为前提（对 C40 为假）；§8.6.1 `:1556-1558` 的 `req.c40_audit`
  未声明 fallback reason/outcome 字段（§8.1.1/§8.5/§8.6.3 均要求其在
  `state=NONE` 后存活，TC-83 自称 schema 完整）；`geometry` 两套冲突值域
  （§8.6.1 `:1552` vs §12.3 `:2798-2800`/§13.1 `:2964`）；collector capability
  manifest 漂移（§9.2.3 `:2385`、§17.1.4 `:3925` 引用已不存在的顶层
  `docker_capabilities`；`collector_capabilities` 混入 security-opt；
  `ebpf_v1` 因 `allowed_max` 不含 `BPF`/`PERFMON` 而不可达）；
  §12.4 族1“从未构造 plan”与 deferred 重验后判 `dense_ineligible` 冲突。
- F-01..F-07 / F-09 / F-11 = CLOSED；F-08 = PARTIAL（P2-4）；
  F-10 = PARTIAL（P2-2、P2-5）。
- **下一步**：修订 1 P1（并清 5 P2）后做第九轮 closure review；
  open P0/P1归零并产出versioned `plan-review-*.json`之前，
  **不得**升级状态、**不得**plan freeze、**不得**申请branch授权。
- 授权边界不变：只授权计划编制/审阅/文档同步；branch、实现、Docker与GPU
  仍为`PENDING USER AUTHORIZATION`。

### 2026-07-30T06:47:22-07:00 Phase7.5 C40计划第七轮review = FAIL

- 计划文件：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表 `V1-r7`，5592行，
  sha256 `9067dd9bc5c3351652296ebcfe5053e949a63792de04012c5b26c522469e049c`。
- 第七轮只读全文件逐API + 跨章节 closure review 完成；**未改计划文件、
  未建branch、未跑测试/GPU**。底座固定 `worktrees/cross-store-substrate@0206f17b`（clean）。
- Verdict = `FAIL / NOT READY`；open = `0 P0 / 2 P1 / 4 P2`。
- 第六轮 **7 项 P2 全部 CLOSED**（§14.1 provenance 四字段+TC-72；§7.2 三分支守卫；
  §13.3 `c40_admission_guard_ms`+§19.9 豁免；§8.2 矩阵 defer 边+TC-77；
  §8.6.1a `:1101`/`:1160-1163` 统一 `scheduling_prefix_len`+TC-78；
  B-4 `c40_copy_budget_insufficient`+TC-79；logprob sentinel `-1`+TC-80）。
  第六轮 1 P1 = **PARTIAL**：helper 按 `req_pool_idx is None` 退化、
  §8.1.1 `clear_target_staging()`、TC-75 均已落地，但只覆盖"未 admit"分支。
- **P1-1（新，阻断）**：pre-admission 转 `DENSE_ISLAND_FALLBACK`/`dense_ineligible`
  的请求会在**同一调度轮**被 `add_one_req` 接纳（`scheduler.py:3017→:3018`），
  首次 `alloc_for_extend`（`allocation.py:333`）分配 `req_pool_idx` 后 helper 的
  退化条件立即失效，`effective_prefix_len()` 返回已被清空的 `middle_cursor=0`、
  `effective_prefix_indices()` 返回空 tensor ⇒ §8.6.3 步骤4B 的
  `assert snapshot.extend_start == middle_cursor` 必然失败（族4，整块作废）；
  放宽断言则 `set_extend_range` 起点回 0、`allocation.py:316` 收到空 prefix
  tensor，borrowed exact 段从 `req_to_token` 消失。与 §8.1（fallback 时
  `middle_cursor = exact_length`）及 §8.2"与从未 eligible 逐字段一致"矛盾；
  TC-75 只断言"未 admit"分支。最小修复：pre-admission 的
  `clear_target_staging()` 同时置 `c40_state := NONE`，或在 admission-commit
  seam 以当前 `prefix_indices` 重建 borrowed/effective/cursor，并补配对 TC。
- **P1-2（新，阻断）**：C40 请求的 `skip_radix_cache_insert` **没有任何 seam
  会置 `True`**。底座 `schedule_batch.py:1080-1082` 仅在
  `bootstrap_host == FAKE_BOOTSTRAP_HOST or approx_kv_metadata is not None`
  时为真且在 `Req.__init__` 内求值；C40 span 决策禁止走 HTTP 字段（§14.2b 规则5），
  §7.4 的 staging hook 又是 `if approx_kv_metadata is not None:`（底座 `:1315`）
  的 `elif` 旁路 ⇒ 该 flag 实为 `False`。后果：`stash_chunked_request` →
  `maybe_cache_unfinished_req`（`common.py:103-107`）不再早退，
  `cache_unfinished_req` 每轮改写 `prefix_indices`/`cache_protected_len`
  （破坏 INV-6 与 `radix_cache.py:496-551` 的 free 分界），并在 cursor 越过
  `target_end` 后把近似 copied island 插入 exact Radix ⇒ 触发 SR-4。
  §7.4 约束4 / §8.6 / §8.6.1a / §8.2.1 / TC-10 / TC-62 全部依赖该前提，
  §8.6.4 也未把 `:1080` 列入最小修改清单。
- P2（4项）：§9.1 ToolEvent 仍内嵌 `collector_impl`/`collector_capabilities`
  与 TC-76"不嵌入 ToolEvent"冲突；§8.2 缺"deferred/staged 重验后判
  `dense_ineligible`（→NONE）"转移边；§8.6.1 容器字段表未声明 `geometry`
  与 `suppress_produce_once`（却在 `:1633`/`:1388`/`:2102`/`:2136` 被规范引用）；
  TC-66/TC-72 的"`docker_capabilities` 与命令逐字段一致"与 §9.2.3
  （SYS_PTRACE）/§17.1.4（+SYS_ADMIN）/§14.1（超集）不可同时成立。
- F-01 / F-02 = **PARTIAL**（分别受 P1-1、P1-2 影响）；F-03..F-11 = CLOSED。
- **下一步**：修订 2 P1（并清 4 P2）后做第八轮 closure review；
  open P0/P1归零并产出versioned `plan-review-*.json`之前，
  **不得**升级状态、**不得**plan freeze、**不得**申请branch授权。
- 授权边界不变：只授权计划编制/审阅/文档同步；branch、实现、Docker与GPU
  仍为`PENDING USER AUTHORIZATION`。

### 2026-07-30T06:29:17-07:00 Phase7.5 C40计划第六轮review = FAIL

- 计划文件：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表 `V1-r6`，5576行，
  sha256 `51fda84c6ae04bda6f5aefb3d03c24f16363fa89500c2b05e5317c1e6e448e62`。
- 第六轮只读全文件逐API + 跨章节 closure review 完成；**未改计划文件、
  未建branch、未跑测试/GPU**。底座固定 `worktrees/cross-store-substrate@0206f17b`（clean）。
- Verdict = `FAIL / NOT READY`；open = `0 P0 / 1 P1 / 7 P2`。
- 第五轮 1 P1 **CLOSED**（§8.6.3步骤4B 已加 `c40_state == DENSE_PREFIX` 守卫，
  `DENSE_SUFFIX`/`DENSE_ISLAND_FALLBACK` 只推进 cursor，配 TC-70）；
  5 P2 中 3 项 CLOSED（B-3 加 `target_end < len(prompt)` + B-5 穷尽；
  §13.5.0 双 mount 措辞；`_update_prefill_budget(prefix_len=)` 首轮 borrowed
  exact / 后续轮字面 `0` + TC-74），1 项 PARTIAL（`scheduling_prefix_len` 已加，
  `:1160-1163` 行未同步），1 项 NOT CLOSED（provenance 字段加进 §9.1 而非 §14.1）。
- **P1（新，阻断）**：**pre-admission 的 `DENSE_ISLAND_FALLBACK` /
  `dense_ineligible` 缺 target-side reset**。§8.5 阶段2 允许 deferred 请求
  重验 source 失败后转这两个状态，此时 `req_pool_idx is None`、零 owned KV；
  但 §8.6.1a helper 只在 `state is NONE` 时退化、§7.4 要求跨轮保留
  `middle_cursor`、§8.2 的 fallback 清理只覆盖 provisional/lease、
  admission final `torch.equal` guard 只对 `STAGED` 执行。若该请求本轮未被
  admit（capacity 场景常见），下一轮 `match_prefix` 重建 `prefix_indices` /
  `cache_protected_len` 后，陈旧 `borrowed_exact_indices` 与 `middle_cursor`
  会经 `allocation.py:316` → `write_cache_indices` 写回 `req_to_token`
  ⇒ 静默跨请求 KV 污染，teardown 还可能对 Radix 所有 index 二次 free。
  与第三轮已闭合的 P0-2 同类。最小修复：helper 改以"未 admit"
  （`req.req_pool_idx is None`）为退化判据；admission 前进入 fallback/ineligible
  时按 `ADMISSION_DEFERRED` 规则清空 target-side；补一条对照 TC-58 的新 TC。
- P2（7项）：§14.1 缺 `provenance.{collector_impl,primary_oracle_impl,
  supplemental_oracle_impl,collector_capabilities}`（被误加到 §9.1）；
  §7.2 data-flow 仍是无守卫三分支；`c40_admission_guard_ms` 未进 §13.3；
  §8.2 矩阵缺 `DENSE_PREFIX/DENSE_ISLAND_FALLBACK --admit_deferred-->
  ADMISSION_DEFERRED`；§8.6.1a `:1160-1163` 行与步骤2/TC-71 冲突且未列 `:1101`
  （page_size=1 下为精确 no-op）；B-4 pre-copy capacity gate 拒绝无 terminal
  reason；logprob fail-close 把 sentinel `-1` 误判（常见路径已被
  `scheduler.py:2336-2353` 归一化，影响仅限 `token_ids_logprob` 窄路径）。
- F-01..F-11 = 全部 CLOSED。
- **下一步**：修订 1 P1（并清 7 P2）后做第七轮 closure review；
  open P0/P1归零并产出versioned `plan-review-*.json`之前，
  **不得**升级状态、**不得**plan freeze、**不得**申请branch授权。
- 授权边界不变：只授权计划编制/审阅/文档同步；branch、实现、Docker与GPU
  仍为`PENDING USER AUTHORIZATION`。

### 2026-07-30T05:44:53-07:00 Phase7.5 C40计划第五轮review = FAIL

- 计划文件：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表 `V1-r5`，5557行。
- 第五轮只读全文件逐API + 跨章节 closure review 完成；**未改计划文件、
  未建branch、未跑测试/GPU**。底座固定 `worktrees/cross-store-substrate@0206f17b`（clean）。
- Verdict = `FAIL / NOT READY`；open = `0 P0 / 1 P1 / 5 P2`。
- 第四轮全部 open（2 P1 + 7 P2）**均已 CLOSED**，逐条经底座核实：
  `already_computed += island_len` 在 copy commit 上闭合 exact/copy telemetry
  （B-3 `new_cached=exact`、B-4 suffix 轮 `0`）；两处最小 rw mount 与
  capability/security-opt 已落进 §13.5.0 与 §14.1；membership assert 限定
  `lifecycle_live`；INV-3 限定 admission 后 lifecycle；TC-51 明确
  validation=0 仍跑 `torch.equal` guard；§7.3.1 覆盖 `[0,target_start)`；
  staging 增 dynamic-chunking / deterministic / multimodal fail-close；
  新增 `c40_logprob_range_unsupported`；版本升 `V1-r5`。
- **P1（新，阻断）**：§8.6.3 步骤4B / §7.2 的 enqueued-range commit hook
  三分支状态判定缺 `c40_state == DENSE_PREFIX` 守卫。§8.6.3 步骤6、§8.6.1b、
  TC-6/TC-27/TC-50、INV-3 都要求 `middle_cursor` 在 `DENSE_SUFFIX` 继续推进
  到 `prompt_end` 且无第二套机制 ⇒ 每个 copy 成功的请求在 suffix 提交时
  `middle_cursor > target_start`，被误判 `c40_boundary_overrun`，outcome
  退化为 `dense_attempted_fallback`，族4 使整块 `engineering_valid=false`
  ⇒ primary confirmatory 无有效数据；并与 §8.2 转移矩阵、§12.2 reason 定义
  矛盾。最小修复：把三分支判定包进 `if c40_state == DENSE_PREFIX:`，
  `DENSE_SUFFIX`/`DENSE_ISLAND_FALLBACK` 只推进 cursor 与 owned 集合。
- P2（5项）：§8.6.1a 的 `cand_extend_input_len`/`:1160-1163` 与 §8.6.3 步骤2
  的 `projected_prefix_len`（B-3=`target_end`）冲突；B-3 判据缺
  `target_end < len(prompt)` 遮蔽 B-5 兜底使 TC-69 不可满足；§14.1 manifest
  缺 `provenance.collector_impl` 等字段（与 §9.2.3/§17.1.4/FD-7/TC-66 冲突）；
  §13.5.0 残留"唯一允许的宿主可写挂载点"与"恰好两处"冲突；
  `_update_prefill_budget(prefix_len=)` 行未限定只适用 `add_one_req`。
- F-01..F-11 = 全部 CLOSED（F-11 附残留 P2-4 文本冲突）。
- **下一步**：修订 1 P1（并清 5 P2）后做第六轮 closure review；
  open P0/P1归零并产出versioned `plan-review-*.json`之前，
  **不得**升级状态、**不得**plan freeze、**不得**申请branch授权。
- 授权边界不变：只授权计划编制/审阅/文档同步；branch、实现、Docker与GPU
  仍为`PENDING USER AUTHORIZATION`。

### 2026-07-30T04:23:18-07:00 Phase7.5 C40计划第四轮review = FAIL

- 计划文件：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表仍标 `V1-r3`，5494行。
- 第四轮只读逐API closure review完成；**未改计划文件、未建branch、
  未跑测试/GPU**。底座固定 `worktrees/cross-store-substrate@0206f17b`（clean）。
- Verdict = `FAIL / NOT READY`；open = `0 P0 / 2 P1 / 7 P2`。
- 第三轮4项阻断findings（P0-1 B-4 parked、P0-2 deferral快照、
  P1-1 copy双计、P1-2 dense契约）**全部 CLOSED**，并经底座逐API核实；
  全部 `verified-code` 行号抽查属实。
- **P1-1**：§8.6.1a 把 `prefix_lens` 改 effective 后，
  `schedule_batch.py:2317-2318` 的
  `new_cached = pre_len - already_computed` 会在 copy 提交轮把 `island_len`
  加进 `req.cached_tokens`（经 `output_streamer.py:85-87/198` 用户可见），
  违反计划自订的 exact/copy 分离与 TC-60；计划只改了 2337。
  最小修复：copy commit 处 `req.already_computed = max(.., target_end)`
  + 把 2314-2318 列入 §8.6.1a + 扩 TC-60 断言。
- **P1-2**：§13.5.0/§14.1 的 `mounts` 仍把宿主 `results/` 根目录整体 rw
  挂载（与同段"禁止整体 rw"、CR-7/CR-8 矛盾），且缺
  `/results/phase7_5_c40` 条目 ⇒ 按"未列明则不得挂载" runner 无处写。
  F-11 由 CLOSED 回归 PARTIAL。
- P2（7项）：`inflight++` 前 membership assert 未排除底座 hybrid-SWA parked
  分支；INV-3 单调性与 deferred target-side 重建矛盾；TC-51 零 D2H sync 与
  强制 `torch.equal` admission guard 冲突；§7.3.1 未声明 dense rule 须覆盖
  `[0,target_start)`（否则 `_validate_bounds` 必抛 unowned gap）；staging
  fail-closed 缺 `enable_dynamic_chunking`/multimodal-encoder/
  `truncation_align_size`；TC-29 logprob 对齐在 island 上不可达且无
  fail-closed；版本号仍 `V1-r3` 但该变更行已被就地改写（违反 §28.1）。
- F-01..F-10 = CLOSED；F-11 = PARTIAL。
- **下一步**：修订 2 P1（并顺带清 7 P2）后做第五轮 closure review；
  open P0/P1归零并产出versioned `plan-review-*.json`之前，
  **不得**升级状态、**不得**plan freeze、**不得**申请branch授权。
- 授权边界不变：只授权计划编制/审阅/文档同步；branch、实现、Docker与GPU
  仍为`PENDING USER AUTHORIZATION`。

### 2026-07-30T03:29:27-07:00 Phase7.5 C40计划第三轮review = FAIL

- 计划文件：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表标 `V1-r3`，5455行。
- 第三轮只读逐API closure review完成；**未改计划文件、未建branch、
  未跑测试/GPU**。底座固定
  `worktrees/cross-store-substrate@0206f17b`（clean）。
- Verdict = `FAIL / NOT READY`；open = `2 P0 / 2 P1 / 6 P2`。
- 第二轮4项阻断findings（P0-A/P0-B/P1-A/P1-B）**全部 CLOSED**；
  第三轮自称9项修复中6项经底座核实成立。
- **P0-1**：§8.6.3步骤7新引入的 **B-4 parked continuation**
  （`add_chunked_req`返回原req但不入`can_run_list`）与
  `scheduler.py:3073-3079`冲突——`assert self.chunked_req is None`
  可被非C40长请求触发崩溃；`inflight_middle_chunks += 1`无membership判据，
  递减只在`batch_result_processor.py:283/336`遍历`batch.reqs`时发生，
  计数器只增不减 ⇒ 最终chunk被判为middle chunk、不采样、空`output_ids`
  进decode。底座`schedule_policy.py:841-842`要求chunked owner必须入list。
- **P0-2**：§8.5/§8.1.1/TC-8要求`ADMISSION_DEFERRED`时
  `borrowed_exact_indices`逐字段保持，但deferred请求不持有任何lock，
  其matched prefix可被驱逐重分配；等长不等值时INV-6仍通过，陈旧index
  经`effective_prefix_indices`→`allocation.py:316`→`write_cache_indices`
  写回`req_to_token` ⇒ 静默跨请求KV污染；与步骤1"重新staging"自相矛盾。
- **P1-1**：`charge_c40_copy_allocation`在island物理分配后再加
  `rem_total/cur_rem` offsets，而二者是读`available_size()`的live property
  （`schedule_policy.py:585-620`）⇒ 双计。
- **P1-2**：§7.3.1的dense disposition契约与底座API不符
  （`transfer.py:120-140`是`dense_prefill(target_start=,length=,reason=)`，
  且先回调fallback chunk reason、再用`_contiguous_ranges`重切range），
  三元组精确查表在普通fallback下必miss ⇒ 具体reason被吞成
  `c40_copy_exception`，违反§12.1规则4。
- P2：island计入`log_hit_tokens`/`cached_tokens`；forced-middle轮变单请求
  batch未披露；§8.2.1规则3与`cache_unfinished_req`冲突；
  §7.4 `scheduler.py:2966/2967`应为`2967/2968`；`enable_mixed_chunk`
  未fail-closed；`process_pending_chunked_abort`未列入五条final hook。
- F-01/F-10 = PARTIAL，其余 F-02..F-09 / F-11 = CLOSED。
- **下一步**：修订2 P0 + 2 P1（并顺带清6 P2）后做第四轮closure review；
  open P0/P1归零并产出versioned `plan-review-*.json`之前，
  **不得**升级状态、**不得**plan freeze、**不得**申请branch授权。
- 授权边界不变：只授权计划编制/审阅/文档同步；branch、实现、Docker与GPU
  仍为`PENDING USER AUTHORIZATION`。

### 2026-07-29T23:58:44-07:00 Phase7.5 C40计划第二轮review = FAIL

- 计划文件：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表标 `V1-r2`。
- 第二轮只读targeted closure review完成；**未改计划文件、未建branch、
  未跑测试/GPU**。
- Verdict = `FAIL / NOT READY`；open = `2 P0 / 2 P1 / 6 P2`。
- 上一轮13项：9 CLOSED / 3 PARTIAL（P0-01、P2-01、P2-05）/ 1 OPEN（P2-07）。
- **P0-A**：底座同时只允许一个chunked请求
  （`scheduler.py:3075 assert self.chunked_req is None`、`:3079`单点
  `inflight_middle_chunks += 1`），而§8.6.3步骤3要求每个被钳制的C40请求
  都设`new_chunked_req`；钳制还少消耗chunk预算，解除了原有隐式互斥。
- **P0-B**：全文0处提及retraction/preemption；
  `release_kv_cache + reset_for_retract`（`schedule_batch.py:1567,1790,1795`）
  会清零`cache_protected_len/kv_committed_len`但不清`req.c40`，
  破坏五路cleanup恒等式与INV-1/6/7，并可能把已释放index写回`req_to_token`。
- **P1-A**：B-3的`req_to_token_pool.alloc([req])`缺`None`失败分支。
- **P1-B**：`needs_host_load_back()`在staging后改写`prefix_indices`与
  `cache_protected_len`，使staged快照/INV-4/INV-6失效。
- P2：族4 kind域缺2项+族3 cleanup失败双计；§25.1仍允许host构建manifest；
  `req_to_token`是int32而Triton硬转int64指针；`add_one_req_ignore_eos`/
  `_add_dllm_req`两个extend起点未覆盖；§28.2版本号与版本表不一致且§0.4
  未列G0q；F-04/06/07/08/10/11无锚点。
- 已确认属实的修订：effective-prefix行号、`prefix_lens`/`prefix_tensors`
  成对、底座单一free ledger、identity membership、adapter-local dense
  disposition、collector authority分层、primary estimand、最小rw mount。
- **下一步**：修订2 P0 + 2 P1（并顺带清P2）后做第三轮closure review；
  open P0/P1归零并产出versioned `plan-review-*.json`之前，
  **不得**升级状态、**不得**plan freeze、**不得**申请branch授权。
- 授权边界不变：只授权计划编制/审阅/文档同步；branch、实现、Docker与GPU
  仍为`PENDING USER AUTHORIZATION`。

### 2026-07-29T17:42:31-07:00 Phase7.5 C40 clean-room计划编制中

- Phase7 final status仍为VALID / NEGATIVE / INCONCLUSIVE-DESCRIPTIVE。
- publication=`READY WITH CAVEATS`；Phase7 RESULT_MANIFEST=`88/88`。
- `research/phase_reports/`新增Phase4、5、6、7四份详尽报告与跨阶段总报告。
- Opus 5 Max Thinking完成执笔；GPT-5.6 Sol Max完成全文与closure review。
- 报告初审0 P0 / 9 P1 / 12 P2；全部修复后open P0/P1/P2=`0/0/0`。
- 报告完整覆盖实验、fix、lesson、结论、预测、限制与provenance。
- 本轮仅整理既有证据，未运行GPU或server；历史实验仍全部来自Docker。
- V40已pin=`13671eb708da`；概念上是grounded selector × R0 Raw+RoPE，不是prefetch。
- 当前`NOT APPROVED AS-IS`：常见write与mixed read-write可绕过path invalidation。
- 完整报告：`research/CODING_AWARE_V40_BRANCH_TECHNICAL_REPORT.md`。
- GPT-5.6 Sol Max最终报告review=`PASS WITH CAVEATS / READY WITH CAVEATS`，
  open report P0/P1/P2=`0/0/0`。
- 分支自身仍有2类P0 freshness/abstention blocker；报告通过不等于分支通过。
- 用户决定把clean-room C40 reproduction定义为Phase7.5，只复用思路、
  **不复用合作者代码**，并尽量实现原分支尚未实现的能力。
- 当前只授权编制和审阅完整执行计划；新branch、代码、Docker实验和GPU
  Gate仍为`PENDING USER AUTHORIZATION`。

### 2026-07-28T19:37:55-07:00 correction/reconsolidation完成

- S0 correction：40/40 direct `unsupported <- store_miss`。
- correction耗时0.09833h；原22 starts/1.31014h不变。
- 原raw/log不可变；22 compact与summary全量重生成。
- summary SHA=`e2deec5a...`；状态VALID / NEGATIVE / INCONCLUSIVE。
- RESULT_MANIFEST=`79/79`。
- 下一步：等待Sol/Opus targeted delta closure与final disposition。

### 2026-07-28T19:19:44-07:00 Correction治理review通过，等待review artifact

- 只读Opus delta review：`PASS_WITH_CAVEATS`，open P0/P1=`0/0`。
- entry=`READY_AFTER_REVIEW_ARTIFACT_AND_AUTH_CORRECTION`。
- 两个P1均CLOSED：RESULT_MANIFEST `--check` 复跑`75/75`；数值不变已独立验证。
- correction manifest rev1自哈希重算=`2a907fee...`，`--check`通过。
- runtime授权门与authorized rev2模拟共13用例行为正确；post-pin变更3文件
  全在4项allowlist内；envelope仅因review artifact未创建而阻塞。
- consolidator按设计在correction执行前不可端到端跑通，非缺陷。
- 测试复现：targeted `119 passed`、`36 passed`、bench `144 passed`。
- 6条P2记录在`PROJECT.md`，不阻塞授权。
- 下一步：correction review artifact → authorized rev2 → Docker S0 correction
  → `--force`重生成22 compact+summary并同commit重建RESULT_MANIFEST。

### 2026-07-28T18:33:37-07:00 等待S0 correction授权

- 原Phase7：22 starts / 1.31014h，raw/log不可变。
- result review cross-consolidation：NOT_READY，0 P0 / 6 P1。
- 只需1个S0 terminal-reason supplementary correction；其他GPU不重跑。
- correction code pin=`a950ab91...`；runner=`c92225d4...`。
- correction manifest rev1 self=`2a907fee...`，status=pinned_blocked。
- correction Opus PASS_WITH_CAVEATS；RESULT_MANIFEST=`75/75`。
- correction后全量重生成22 compact+summary，原raw/log保持不变。
- 下一步：correction review artifact → authorized rev2 → Docker S0 correction。

### 2026-07-28T11:30:50-07:00 Phase7 offline consolidation实现完成

- `cross-store-substrate`新增纯离线consolidator与CPU tests，未提交。
- 输入契约：authorized rev12/design hash，22 raw、22 logs、central JSONL。
- 输出：22个自哈希compact JSON + 1个自哈希consolidated summary；默认拒绝
  覆盖，`--force`显式允许。
- 真实staging验证通过：
  - elapsed GPU-equivalent=`1.31014h`；
  - engineering=`VALID`；
  - A8 mechanism=`NEGATIVE`；
  - W system behaviour=`INCONCLUSIVE/DESCRIPTIVE`；
  - S0 R4-like 5x available/valid，S4 diagnostic_unavailable；均非KVCOMM。
- 验证：`7 passed`，isort/ruff/black通过，22 compact与summary自哈希复核通过。
- 额外绑定验证：execution envelope、source SHA、manifest file SHA、runner
  path/hash、central run_id/phase/output；compact list同时提供canonical/file hash。
- 下一步：对新增两文件做独立review；之后在指定结果目录正式运行consolidator，
  再生成/更新`RESULT_MANIFEST`。不要把`r2_like`写成R2执行。

### 2026-07-28T10:00:31-07:00 A8 screening为NEGATIVE

- 四个restart-0 settings全部工程有效。
- request-path median：
  - body1024=`0.772–0.775x`；
  - body2048=`0.933–0.936x`。
- N8 full-setup=`0.609–0.642x`，未达5% MDE。
- 跳过8个primary supplement starts。
- 下一步：chunk1024 sensitivity 2 starts，然后W/R4 scheduler矩阵。

### 2026-07-28T09:45:04-07:00 Phase7 wave-0完成

- S4/rho2/chunk4096：四个non-R4 profile可达；R4-like容量诊断不可达。
- S0/rho2/chunk4096：五个profile全部可达。
- 两cell capacity error=0，reset/inactive/orphan均通过。
- 无ES-ENGINEERING failure；rho3未运行。
- 下一步：A8 body1024/2048 × rho1.5/2.0 restart-0 screening。

### 2026-07-28T09:28:47-07:00 rev12 authorized

- V7 Current/Latest。
- rev12 self=`2d66a1bc...`，design=`50003145...`。
- status=`authorized`，blockers=[]，review=`PASS WITH CAVEATS`。
- primary commit=`759476c9...`，envelope HEAD=`d42a5d15...`，
  RESULT_MANIFEST=`5/5`。
- Phase7下一步：wave-0 S4/rho2/chunk4096，然后S0/rho2/chunk4096。
- rho3 disabled；runtime只写`/results/phase7`。

### 2026-07-28T09:23:55-07:00 V7已激活，等待authorized rev12

- V7 plan commit=`c80ec165...`，由PROJECT/HANDOFF激活为Current/Latest；
  plan blob byte-frozen。
- rev11 self=`48c86bf0...`，design=`50003145...`，
  code pin=`81405f42...`。
- final Opus verdict=`PASS WITH CAVEATS`，open P0/P1=`0/0`。
- review artifact commit=`b0837505...`，artifact SHA=`2c83f7dd...`。
- 下一步只剩生成design-preserving authorized rev12。
- Docker必须同时只读挂载：
  - cross-store worktree；
  - `/home/chris/Workspaces/kvcache-research/sglang-experiments`
    （linked-worktree gitdir，保持相同绝对路径）。
- 波次策略：全部run写`/results/phase7`；所有GPU wave结束后一次性版本化，
  wave之间不提交raw/compact/log。
- rev12完成前不得启动Phase7。

### 2026-07-28T08:52:08-07:00 V7等待final delta review

- V5/V6已归档；V7 plan为byte-frozen plan of record。
- V6 final Opus的1 P0/3 P1与后续capacity/evidence findings已闭合。
- runtime结果写`/results/phase7` staging，三个runner共享authorized gate。
- final code pin=`81405f4278b034911bc613c4ee17c79d15ee8f35`。
- CPU evidence HEAD=`a01642fdd6cdc869cb2c991c23003ff600665f37`。
- Docker evidence：
  - full targeted=`269 passed + 22 subtests`；
  - capacity=`24 passed`；
  - ceiling=`55 passed`；
  - scheduler=`12 passed`。
- 下一步：提交V7 plan → 生成V7 pinned manifest → Opus final delta review →
  versioned review artifact → authorized revision。
- Phase7仍未开始。

### 2026-07-28T04:31:07-07:00 V6 candidate与runner完成

- V5已通过不可变commit/hash归档。
- V6 candidate已创建；R2=`disabled_not_comparable`，删除Phase7 R2 GPU cells。
- 两个Phase7 runner及共享模块已实现。
- Docker targeted suite=`152 passed + 10 subtests`。
- 三轮targeted review无开放P0/P1。
- final code pin=`5d9a5793d73121f088890aa6c02cfebc31cd97be`。
- 最终矩阵：13 committed/30 starts；含rho3条件项14/31；
  expected 3.8 GPUh，hard cap 6 GPUh。
- rev7 manifest已pin：self=`3eebdbfb...`，design=`ec5aaf59...`，
  blocker仅`final_opus_review_pending`；result manifest `1/1`通过。
- rev9现为final review-of-record：self=`5daa95a3...`，
  design=`ae8d8e1e...`，plan=`14a573eb...`，envelope HEAD=`eb7daf63...`。
- 下一步：Opus 5 Max Thinking final review → closure → 条件性授权生效。
- final review artifact路径：
  `benchmark/approx_kv/results/phase7/phase7-final-opus-review.json`。
- rev6在final review前因result-manifest builder陈旧状态描述而supersede。
- Phase7仍未开始。

### 2026-07-28T00:07:40-07:00 新增最终Opus审阅门

- 用户已选择方案B，并条件性授权完成前置工作后执行Phase7。
- rev6 pin后、任何Phase7 GPU执行前，必须使用
  **Claude Opus 5 / Max Thinking**审阅最终plan、manifest、runner、
  R2 disposition与implementation binding。
- 必须闭合全部accepted feedback；若plan design改变，归档旧版、创建新版、
  更新design hash并重新review。
- 只有最终无开放P0/P1，条件性授权才生效并转`authorized`。

### 2026-07-27T23:50:58-07:00 Phase7 Entry前推荐顺序

1. 实现`run_p7_ceiling.py`和`run_p7_scheduler.py`；
2. 在Docker内完成CPU tests和Sol/Opus targeted review；
3. 对R2做有界0-GPU feasibility：
   - 非侵入则实现adapter；
   - 侵入性过高则冻结`disabled_not_comparable`；
4. 生成manifest rev6并pin final SHA/tree/runner hashes，转
   `pinned_blocked`；
5. Opus 5 Max Thinking审阅最终定稿并闭合feedback；
6. 应用用户已给出的条件性授权，转`authorized`。

当前推荐证据增强路线：只在R2 adapter不改变core recovery语义时实现它；
不建议在Entry前扩大到R1/R5/真实R4/host/prefetch。P7-0b chunk4096
feasibility属于授权后的wave-0，不能提前执行。

### 2026-07-27T19:30:00-07:00 V5与primary manifest已封版

- V4已归档为`IMPLEMENTATION_PLAN_V4_ARCHIVED.md`。
- latest为V5 Current / Latest。
- 8个accepted P0已关闭；targeted review新增MDE定义P0，现冻结`MDE=5%`；
  final minimal delta无开放P0，计划已足以生成primary manifest。
- Phase7已收窄为R0 ceiling、条件R2、R4-like proxy与R0 W×S0/S4；
  practical/host/prefetch/async轨道默认全部跳过。
- primary chunk=`4096`；`1024`只作diagnostic sensitivity。
- 预算：committed 13 GPU settings/30 starts；含条件项16/33；
  hard cap36 starts/6 GPUh。
- Phase7 primary manifest rev4已获Sol/Opus最终PASS；rev5仅绑定最终V5 plan，
  settings/workloads未变。
- manifest status=`preregistered_blocked`，仍有5个显式execution blocker。
- 仍未进入Phase7。

- Phase7 primary manifest现已生成并自检通过：
  - 16 settings；
  - 30 committed / 33 total starts；
  - status=`preregistered_blocked`；
  - 5个显式execution blocker；
  - `phase7_execution_authorized=false`。
- Sol/Opus manifest targeted review进行中。
- Manifest rev2已获两位reviewer `PASS WITH CAVEATS`，无P0。
- rev3正在关闭两个赋值级P1：supplement gate作用域与GPUh headroom。

### 2026-07-27T17:15:00-07:00 V5 full review FAIL已修订，等待targeted delta

- Sol/Opus两份full review均为FAIL，已全文互换并cross-consolidate。
- 8个accepted P0已全部定点修订；整体result-bound结构未推翻。
- 新的关键门禁：
  - P7-0 runner/manifest CPU gate；
  - P7-0b P6-4Δ-4096；
  - A8与W workload分离；
  - D0/E0/R0三臂；
  - R4-like只跑S0/S4；
  - expanded outcome taxonomy。
- committed预算30 starts，含条件项33，hard cap36。
- V5仍不是Current / Latest；下一步只做targeted delta review。

### 2026-07-27T16:10:00-07:00 Phase6 technical Exit为PASS WITH CAVEATS

- 用户以test-only集成路线取代此前方案C治理性豁免。
- 最终P6-F v3：
  - reservation failure=1、`reuse/dense_fallback`=1；
  - `cross_store_reservation_failed=1024`、`device_allocation_failed=0`；
  - dense/fallback namespace均严格为64-token exact header；
  - 8-token输出完成且与dense一致；
  - same-process下一请求正常recovery；
  - 独立无注入server重新注册并正常recovery；
  - pre-flush与post-reset reserved/provisional/leases/orphans均为0。
- artifact明确`fault_injected=true`、
  `natural_pressure_reachability=false`。
- P6-F v3及两个最终log、primary P6-H/P6-4 logs、raw JSONL均已版本化；
  manifest现`48/48`通过。
- P6-4旧误标签已用独立correction artifact更正，raw不改写。
- 两个P6-F targeted reviewer最终均判PASS；两个formal Exit reviewer均关闭
  P0-1/P0-3，最终判`PASS WITH CAVEATS`，无新P0/P1。

- 全部实验在Docker SM75镜像内执行。
- 实现branch head：`5966983236599bd07bf3e582474e41bc01a22f6b`（含manifest rev5；Phase7 runner实现后须由rev6重新pin）。

| 门禁 | 状态 |
| --- | --- |
| CL1 | 完成，`winner=NONE`（仅为被测实现与冻结规则下的promotion结果；未排除header-dependent缺陷） |
| CL2 | 完成，waive为provisional chunk `1024` |
| CL3 | 完成 |
| P6-H | **通过**，`status=valid`（1 restart/2 round的8-token输出canary；不是KV或logit保真证明） |
| P6-4 | **完整跑通**；三个S4 cell中的四个non-R4 profile可达，所有顶层cell仍为`diagnostic-unavailable` |
| CL4 | 正式双模型Exit review完成，最终结论为`PASS WITH CAVEATS` |

#### Phase6 Exit逐条

最终状态：

```text
technical_exit = PASS WITH CAVEATS
```

caveat：reservation-failure-associated fallback仅在test-only fault-injected
canary强度验证；`natural_pressure_reachability=false`。

直接证据还包括：exact/device-approximate共享device budget、双向pressure
（exact→approx `47.5GB`、approx→exact `58.8GB`）、R1-like worst-case
（k32）profile可达、generic host canary以及已完成run的clean reset。
host使用独立host limit；exact-host/HiCache unification未实现。

#### 本轮共5处修复（全部已推送）

1. `af81934e4` P0：recovery期间请求自身prefix未加锁 → 自我覆写。
2. `c405343c8` P6-H reseed断言（满命中报告`N-1`）。
3. `db2d18ff0` P1-1：回传SWA释放元数据。
4. `40f09c1fe` P1-3：recovery slot provisional所有权，杜绝admission拒绝时泄漏。
5. `3379e6699` P1-2：stale victim跳过并重试，detached节点移出`evictable_leaves`。
   另有`0f379eb04`/`fb284cad4`使P6-4 runner逐cell容错。

#### 重要判定

- S0/rho2与S4/rho3的OOM是**真实容量不可达**，不是实现缺陷。
  诊断C以`0.05s`采样确证：死亡瞬间approximate store为`0`字节`0`记录，
  可用`704` token而请求需`1024`；且exact压力此前已成功从approximate对象
  回收`2.2GB`，回收路径工作正常。
  （注：诊断C v1曾以`0.4s`粗采样得出相反结论，已撤回——采样间隔必须短于
  workload的分配动态。）
- `r4_like`（约5x）在所有cell不可达，属计划预先允许的R4例外。
- 全目录回归`935 failed`是**改动前既有基线**（已用`git stash`对照两次），
  本次净增3个pass。

### 下一步

1. 实现并review `run_p7_ceiling.py`与`run_p7_scheduler.py`；
2. 决定R2 strategy（adapter或disabled/not_comparable）；
3. 原子更新manifest/HANDOFF/PROJECT/TODO并pin Phase7 implementation SHA；
4. 等待用户明确授权Phase7。

### 不要重做

- 不重跑Phase4/5完整矩阵，不重复R2/R5 rho2矩阵。
- 不重跑CL1/CL2/CL3/P6-H。
- `test_radix_cache_unit.py::test_memory_allocated`及全目录935 failed均为
  既有失败，不要当作回归。


### 2026-07-27T01:50:00-07:00 P0已修复，P6-H通过，CL1定稿NONE，仅剩P6-4阻塞

- 所有实验均在Docker SM75镜像内执行
  （`ghcr.io/ccdd2023/sglang@sha256:0be6e16e...`、`--runtime=nvidia --gpus all`、
  `--user 1000:1000`、worktree只读挂载、`/results`读写挂载、HF离线只读）。
- 实现branch head：`7bb7365361d0541603c6a7ca1d1199f303310472`。

| 门禁 | 状态 | 结论 |
| --- | --- | --- |
| CL1 | **完成** | `winner=NONE`（冻结规则下的机械结果；未排除header-dependent缺陷） |
| CL2 | 完成 | `inconclusive`，waive为provisional chunk `1024` |
| CL3 | 完成 | S4优势仅在workflow-only分母成立 |
| P6-H | **通过** | `status=valid`，输出逐token一致 |
| P6-4 | **阻塞** | S0/LRU rho2.0确定性device OOM |
| CL4 | 进行中 | 双模型review已启动 |

#### 已修复的P0（根因与先前推断不同）

- 根因：`Req.init_next_round_input`执行recovery时，请求自身的prefix
  **尚未加锁**（`_req_inc_lock_ref`在`schedule_policy.add_one_req`中才发生），
  而`cross_store_resources()`恰好以`lock_ref == 0`为victim条件，于是请求
  自己的prefix被驱逐、slot回到free list、又被当作recovery目的地发回，
  造成自我覆写。
- 修复：新增`protect_request_prefix`在整个recovery窗口持有标准prefix锁；
  加固exact victim guard使stale victim抛`KeyError`而非触发断言。
- GPU验证：先前必然损坏的配置现在与dense逐token一致。
- 锁对称性已用容器内对照实验验证，无泄漏。

#### 最重要的科学结论

CL1在修复前后的guardrail失败计数**完全一致**，因此`NONE`不是缺陷造成的。
CL1用不同header注册/复用（`32_000+` vs `36_000+`），KV本来就是近似的；
P6-H用同一header，修复后逐token一致。两者共同证明：
**跨上下文raw KV复制的恢复误差是真实的，practical family = NONE成立。**

#### 唯一剩余阻塞：P6-4 S0/LRU rho2.0

- 3次独立复现（完整profile、仅rho2.0、缩减profile）均为
  `Available tokens: 0 (available_size=0 + evictable_size=0)`，
  在`alloc_token_slots`抛`RuntimeError`杀死scheduler。
- 已排除：本方lock泄漏（对称性实验）、ordinary prefill缺少cross-store感知
  （`evict_from_tree_cache`确实调用`make_room`）、coordinator重入。
- `launch_cells`无条件把`lru2.0`排在`hier2.0`/`hier3.0`之前，因此该cell
  阻塞其后所有cell；目前已知通过的只有`hier1.1`与`hier1.5`。
- **待确认假设**：P0修复正确地把请求自身prefix移出victim池后，recovery必须
  占用新slot，峰值device需求真实上升，S0/LRU在rho2.0下确实无解。若成立
  应记为`diagnostic-unavailable`而非实现缺陷；但当前表现为硬崩溃而非优雅
  降级，属独立鲁棒性缺口。

### 下一步

1. 判定P6-4 S0/rho2是真容量不可达还是可修复的鲁棒性缺口；
   最直接的对照是在修复前的commit上用相同缩减profile复跑该cell。
2. 让`alloc_token_slots`在cross-store场景下优雅降级而不是杀死scheduler。
3. 完成CL4双模型review与disposition。
4. 之后才考虑归档V4、创建V5（用户已明确指示本轮不升级）。
5. 仍然严格停在Phase7前，等待用户授权。

### 不要重做

- 不重跑Phase4/5完整矩阵，也不重复已修正的R2/R5 rho2矩阵。
- 不重跑CL1（screening与3-restart确认均已在修复后底座完成）。
- 不重跑CL3、P6-H。
- `test_radix_cache_unit.py::test_memory_allocated`是**改动前既有失败**，
  已用`git stash`对照确认，不要当作回归。


### 2026-07-26T20:25:00-07:00 Phase6全部门禁已跑完，Exit被单一P0阻塞

- 所有实验均在Docker SM75镜像内执行
  （`ghcr.io/ccdd2023/sglang@sha256:0be6e16e...`、`--runtime=nvidia --gpus all`、
  `--user 1000:1000`、worktree只读挂载、`/results`读写挂载、HF离线只读）。
- 实现branch head：`248e2cb4774dbee8bb123b64d9b63cbd69f4ff5f`，
  本地与`ccdd2023/sglang:research/cross-store-substrate`远程SHA一致。

| 门禁 | 状态 | 结论 |
| --- | --- | --- |
| CL1 | 完成 | `practical family = NONE`（性能条件全过，仅correctness guardrail不过） |
| CL2 | 完成 | `inconclusive`，显式waive为provisional chunk `1024` |
| CL3 | 完成 | Phase5零GPU重算，S4优势仅在workflow-only分母成立 |
| P6-H | **失败** | 机械证据全通过，数据保真失败（P0） |
| P6-4 | **失败** | 完整矩阵`invalid`；exact-only baseline有效 |
| CL4 | 未开始 | Phase6 Exit未通过，前提不成立 |

- **唯一根本阻塞项（P0）**：`cross_store/allocator.py`在一次驱逐迭代内按
  快照顺序执行整个eviction closure，快照只在下一轮循环开头才刷新。同轮内
  先执行的驱逐会让后续resource的radix节点变成stale，于是
  - P6-4：对stale节点再次`evict`触发`_delete_leaf`断言，scheduler崩溃；
  - P6-H：stale节点被重复释放，device slot回到free list后被覆写，
    压力下近似reuse返回损坏KV。
- 5次隔离实验证明触发条件是“reuse执行时存在真实device压力”，与residency
  tier无关、与是否demotion无关、也不是紧容量本身；零近似exact-cache对照
  16/16完全一致，排除prefill数值不确定性。
- **CL1的`NONE`因果归因无效**：CL1所有臂都在`rho=2.0`压力下执行，其
  quality/first-token失败与该P0完全混淆。`NONE`仍是冻结规则下程序正确的
  结论，但必须在修复后重跑CL1才能重新判定。
- 本轮已修复并推送两个次要缺陷：
  - `resolve_reuse_spans`把prefix-gap的整段dense prefill误记为
    `reuse/exact`且0 fallback（`5e47904ec`）；
  - P6-H canary的recovery header被paired dense驱逐，导致H2D永不触发。
- 本轮已补齐从未执行的Closeout CL3（`0b5e4f7b5`）。
- 相关回归：容器内`164 passed, 5 skipped` + closeout runner `9 passed`；
  isort/black/ruff(F401,F821,UP037)/`git diff --check`全部通过。

### 下一步（严格顺序）

1. 设计并实施cross-store allocator stale-resource P0修复
   （两个候选方向见`IMPLEMENTATION_PLAN_LATEST.md`§15.3），补压力态保真回归。
2. 在修复后的底座上重跑CL1，重新判定practical family。
3. 重跑P6-H与P6-4完整矩阵。
4. 执行CL4双模型review并形成Phase6 Exit disposition。
5. 之后才归档V4、创建result-bound V5，并按§1走完双模型review。
6. 仍然严格停在Phase7前，等待用户明确授权。

### 不要重做

- 不重跑Phase4/5完整矩阵，也不重复已修正的R2/R5 rho2矩阵。
- 不重跑CL3（零GPU重算已完成且不依赖被污染的底座）。
- 不把P6-H/P6-4的失败写成Phase6 negative result：它们是实现缺陷，不是
  容量不可达，也不是机制结论。


### 2026-07-26T14:00:08-07:00 Phase6零GPU实现完成主体，GPU验证阻塞

- Phase6 worktree/branch：
  - `/home/chris/Workspaces/kvcache-research/worktrees/cross-store-substrate`
  - `research/cross-store-substrate`
  - base `research/scheduler-policies@c185428fd`
- 已实现：
  - P6-0 fixed40/token hash/chunk/schema/contract verification；
  - P6-1 exact/approx/host对象、event clock、S0/S4、dependency closure；
  - P6-2 byte budget、reserve/commit/failure账本、双向pressure；
  - device→host demotion、cross-store-aware H2D load；
  - P6-3 lifecycle、orphan拒绝、dependency pin、reset/store gauges；
  - P6-H、P6-4、CL1、CL2 runner。
- Opus三轮review的CR-01至CR-22、CR2-01至CR2-16、CR3-01至CR3-03
  已全部处理；最后一次delta确认无遗留前置finding。
- 800对象、400 victim路径为`0.188s`；格式、lint和diff检查通过。
- 独立GPT-5.6 Sol Max最终review提出8项P1，全部修复；最后delta结论为
  “无剩余P0/P1”。
- 最终CPU验证：`169 passed, 1 skipped`。
- 核心提交：`391bb89901cebebd50ffc9f27a648b09a99abf7e`。
- P6-0/远程branch head：
  `c487e36af5f7ce4da556da1b88c85df750a0b14d`。
- P6-0 contract/workload SHA256：
  - `a498daa36449993ff166dd70870005be22a1da0a7d09e97e8f779d72cbf3fb30`
  - `30c9ae8de429a6389e58bbdcdf096101cf6296ff14d4e6fcf5c2b87c6b1f0749`
- 尚未完成：
  - CL1/CL2/P6-H/P6-4 GPU运行；
  - GPU结果双模型review。
- GPU阻塞：
  - loaded NVIDIA module `580.159.03`；
  - userspace/NVML `580.173.02`；
  - 图形会话占用模块，安全修复通常需要重启。
  - `/var/run/reboot-required`存在；因有活动SSH/tmux会话，未擅自重启。
- 下一步：
  1. 用户安排安全系统重启以加载NVIDIA `580.173.02`；
  2. 复核`nvidia-smi`、保持无无关Docker容器；
  3. 运行CL1→CL2→P6-H→P6-4；
  4. 对GPU结果做Sol/Opus双模型review；
  5. 严格停在Phase7 Entry前汇报。

### 关键约束

- 不得把GPU环境阻塞写成Phase6 negative result。
- P6-4在CL2未运行时只能使用预注册
  `provisional_worst_case` chunk，CL2最终值不同时必须重跑受影响cell。
- R4-like不可达时只允许降低representation multiplicity，不改变对象长度。
- HiRadix exact cross-store eviction当前明确unsupported；P6-H使用标准Radix的
  approximate host路径，不宣称HiCache exact-host统一已完成。
- 所有旧Docker容器已停止；不要恢复无关长期容器。
- 不得进入Phase7。
- 当前实现未改变Phase7经验结论；它只收紧Phase7证据合同与entry gate。
- 不重跑完整Phase4/5或已修正R2/R5 rho2矩阵；只执行CL1、CL2、P6-H、
  P6-4及明确claim触发的条件性补点。
- 驱动恢复只需安全重启以加载已安装的`580.173.02`，不是升级驱动，也不需要
  重做Phase6 patch。
- host已重启并验证恢复：
  - loaded/installed/NVML均为`580.173.02`；
  - SM75镜像内PyTorch CUDA smoke通过；
  - 当前运行Docker容器数为0；
  - CL1/CL2/P6-H/P6-4不再受驱动阻塞。
- 下一执行项为CL1 R0/R1 candidate qualification。
- 完整跨session待办固定在根目录`TODO_LOCAL.txt`。
- Phase7不立即整体改版：先执行CL1/CL2/P6-H/P6-4；结果review后再创建
  result-bound新latest版本。
- 已知的Phase7合同修正先作为冻结边界保留：request-path/N摊销、完整candidate
  enum、方向性pressure、clean-tree provenance，以及P7-3专用HiCache adapter gate。

### 2026-07-25T23:47:04-07:00 V4 latest plan完成双模型review并定稿

- V3已完整归档为`IMPLEMENTATION_PLAN_V3_ARCHIVED.md`。
- 当前latest为`IMPLEMENTATION_PLAN_LATEST.md` V4，状态`Current / Latest`。
- V4阶段结构：
  - Phase4/5 Closeout Lane，不新增phase编号；
  - Phase6 Cross-Store Substrate & Feasibility；
  - Phase7 Integrated Recovery × Scheduling Evaluation；
  - Phase8仅保留Potential Scope，不创建矩阵。
- 双模型使用GPT-5.6 Sol Max Thinking与Claude Opus 5 Max Thinking，完成独立review、全文互换、交叉consolidate和最终delta verification。
- V4补齐：
  - CL2→P6-4依赖；
  - unified schema/rho/ledger/memory指标；
  - R1-like footprint；
  - `P6-H` generic host roundtrip canary；
  - practical=NONE完整停止路径；
  - R4具体victim diagnostic；
  - Review disposition mapping和预算。
- Phase6 Entry仍由CL0阻塞；未创建Phase6分支、未运行新GPU实验。

### 2026-07-25T11:09:10-07:00 V3最终delta verification通过

- GPT-5.6 Sol Max Thinking与Claude Opus 5 Max Thinking均确认V3定稿P0全部闭合，可维持`Current / Latest`。
- 最后4处非阻塞errata已修正：
  - 全文统一`rho_logical_demand`；
  - 展开`steady_target_path`公式；
  - PROJECT追加`T0 = tier GPU-only`历史更正；
  - 统一使用P6-3a/P6-3b命名。
- G0仍未执行，因此Implementation Entry继续blocked；这不影响V3作为当前latest计划。

### 2026-07-25T10:53:22-07:00 V3 latest plan完成双模型review并定稿

- V2已完整归档为`IMPLEMENTATION_PLAN_V2_ARCHIVED.md`。
- 当前latest为`IMPLEMENTATION_PLAN_LATEST.md` V3，状态`Current / Latest`。
- review模型：
  - GPT-5.6 Sol / Max Thinking / long context；
  - Claude Opus 5 / Max Thinking / long context。
- 两模型先独立review，再全文互换并交叉consolidate；所有VA/VB finding已映射到VC修订清单。
- 关键计划变化：
  - 区分Implementation Entry与Experiment Entry；
  - G0与Plan Review阻塞P6-0/P6-1；
  - R0/R1 qualification移到P6-3a，不阻塞allocator实现；
  - chunk实验改为执行或显式waive的配置门；
  - R1改称candidate family，通过预注册promotion rule后才是practical；
  - R4只做独立diagnostic，不参与practical winner；
  - 固定唯一matched-state、flush/reset、四类hit、block/ledger/rho定义；
  - 增加event clock、S4 class order、lock order、rollback、extra-key GC；
  - 统一all-reusable p95恶化`<=5%`；
  - host basic demand-load与`rho_host>=1`压力claim分开；
  - 双模型finding只有主会话接受为blocking P0才阻塞。
- Phase6前GPU结论：
  - Implementation Entry前不需要新的Phase4/5 GPU重跑；
  - P6-3前必须完成R0/R1 k qualification；
  - chunk配置必须执行或显式waive；
  - R2/R5 matched-ratio、rho1.1/3、R2 fallback补点均为条件性。
- 当前G0尚未完成，Implementation Entry仍blocked；Phase6分支未创建、GPU实验未启动。

### 2026-07-24T22:08:11-07:00 Post-rerun双代理复核完成，review已更新

- `CONSOLIDATED_PHASE4_PHASE6_REVIEW.txt`已写入：
  - pre-rerun 65条建议；
  - R2/R5 corrected key rerun完整结果；
  - 同配置Sol/Opus独立复核与交叉consolidate；
  - 原建议status delta；
  - 新PRC-01至PRC-23；
  - 最终Phase4/Phase6 assessment。
- 双代理共同结论：
  - 当前同一R2/R5矩阵不需要再次强制重跑；
  - 旧target-only收益被确认；
  - 旧single-use combined正收益被推翻；
  - R2/R5仍是precomputed target oracle，不是practical candidate；
  - R2与R5的target差异由1%与8.3% repair ratio解释，不应再写成机制性能排序；
  - first-token一致只是不发生粗暴损坏的guardrail，不是semantic correctness。
- 新增重要待办：
  - 旧artifact加superseded pointer；
  - 区分demand rho与resident rho；
  - 报告N=`1/2/4/8`摊销与cold start；
  - R2 fallback counter缺失时不得写成显式0；
  - pressure预算应考虑setup evictable footprint；
  - Phase6 P6-5 host demotion先做可行性canary。
- R2/R5结果分支已用显式`ccdd2023` SSH身份push并核对：
  - `research/cacheblend@e36f1529b838c12a9eb2af7ba4dde91ae9ec124b`
  - `research/cachetune@abcedd62b5a5d801742734e300a5df21e1436737`
- `IMPLEMENTATION_PLAN_LATEST.md`仍未修改；Phase6仍未启动。

### 2026-07-24T21:38:03-07:00 R2/R5 corrected key rerun完成，等待双代理复核

- 固定review文件：`CONSOLIDATED_PHASE4_PHASE6_REVIEW.txt`。
- R2 CacheBlend：
  - 实现提交`c73c9c5ab`，结果提交`e36f1529b`；
  - body1024/2048、ratio1%、header64、rho2、3 server restarts、每臂warmup1+formal2；
  - target-only speedup=`1.659x/2.044x`；
  - causal adapter-combined speedup=`0.441x/0.407x`；
  - request-path speedup=`0.526x/0.434x`；
  - full-lifecycle speedup=`0.324x/0.246x`。
- R5 CacheTune：
  - 实现提交`46d1f85c2`，结果提交`abcedd62b`；
  - body1024/2048、header64、rho2、3 server restarts、每臂warmup1+formal2；
  - target-only speedup=`1.614x/1.978x`；
  - causal adapter-combined speedup=`0.449x/0.406x`；
  - request-path speedup=`0.527x/0.433x`；
  - full-lifecycle speedup=`0.327x/0.246x`。
- 两条路径均使用增量dense物化完整累计causal prefix，再从隔离exact-cache命名空间注册当前<=512-token chunk；dense与recovery在同一server restart内配对。
- 两条路径的所有12个formal pair均首token一致；所有recovery round均0 fallback、真实eviction；三次server的pool reset invariant均通过。
- 当前正在用原GPT-5.6 Sol与Claude Opus 5代理配置复核新代码、raw结果及Phase6影响；review文件尚未写入最终post-rerun assessment。

### 2026-07-24T18:20:12-07:00 Phase 4–6 双代理审计完成，建议待用户评审

- 已按用户要求使用两个 long-context / max-reasoning 代理完成独立审计和交叉汇总：
  - A：GPT-5.6 Sol；
  - B：当前可用的 Claude Opus 5（作为用户指定 Opus 5.5x Thinking 的最近可用配置）。
- 两代理读取了 Phase 4/5 compact JSON、raw results、中央日志、runner/runtime/policy/test、全部相关 worktree 与 `IMPLEMENTATION_PLAN_LATEST.md`。
- 交叉汇总后形成 65 条去重建议与 6 个未决分歧；这些均为待评审意见，不代表已经接受、修改计划或决定重跑。
- 共同结论范围：
  - 不建议完整重跑 Phase 4/5；
  - Phase 5 的 workflow-only 与 all-reusable/full-trace 口径会改变策略排名，需并列报告；
  - R2/R5 的 long-body fresh-KV causal context 与 R5 combined 成本边界需要重点复核；
  - 现有 prefetch 矩阵在 host tier 饱和且同步 H2D 下主要是功能/开销 canary；
  - Phase 6 在 matched-state、cross-store accounting、paired baseline、quality guardrail、统计门槛和 async prefetch 前置条件上存在待评审建议。
- `IMPLEMENTATION_PLAN_LATEST.md` 未修改，Phase 6 未启动，未运行任何新 GPU 实验。
- 下一步：由用户审阅本轮最终建议及逐条点评后，再决定是否修订计划、执行 0-GPU 重算或安排定向重跑。

### 2026-07-24T06:58:18-07:00 Phase 5完成并已推送

- 分支：`research/scheduler-policies@c185428fd`；实现提交`5a87166b4`，结果提交`c185428fd`；远程SHA已核对。
- S0-S4、P0-P3、独立cache-protection metadata、absolute next-use、reusable-prefix boundary、metadata GC、atomic suffix-subtree victim和telemetry均已完成。
- 测试：CPU `226 passed + 27 subtests`；SM75 HiRadix GPU targeted `3 passed`。
- commit-bound正式矩阵：
  - S4 vs LRU mean speedup：rho1.5 `1.32x`、rho2 `1.15x`、rho3 `1.13x`；
  - 三次独立进程范围：rho1.5 `1.32–1.34x`、rho2 `1.11–1.15x`。
- S4+HiCache P2/P3已机械证明真实主动load与admission eviction，但没有稳定TTFT收益且p95更差。
- Phase5最终默认：**S4 hierarchical + P0 off**。
- 结果文件位于`benchmark/approx_kv/results/phase5-scheduler/`。
- 当前停止在Phase 6之前；不自动进行跨恢复路径组合、RTX PRO 6000或并发实验。
- Phase5实验没有并行GPU/request干扰；setting逐个运行、独立server、请求串行、顺序随机、repeat间flush。
- “压力越大priority越有价值”只适用于从无压力进入oversubscription的趋势，不应表述为无限rho下speedup单调增加。
- 当前mean speedup下降的直接原因是S4 hit fraction从rho1.1的1.0降到rho2/3的0.705，而LRU已稳定在约0.51；p50 speedup在rho1.5/2/3仍约1.44/1.45/1.42x。
- 当前rho sweep通过增加对象把working set从15扩至20/27/40个，存在composition混杂；若复核单调性，必须固定同一对象集合并只改变capacity。
- Phase5 S4 baseline是Phase5内的S0 LRU exact-cache trace；Phase5 prefetch baseline是S4+HiCache+P0。
- Phase4 recovery speedup与Phase5 scheduler speedup分母不同，不能直接比较大小。
- 当前Phase5没有真实组合Phase4 R0-R5：无`approx_kv`请求，S3仍用synthetic cost，S4对象类别仍是exact Radix标签；cross-recovery组合属于Phase6。
- 每个setting确实执行1次warm-up；“discarded”仅表示不纳入formal统计。warm-up后flush，formal repeat之间也flush。
- Phase4是一个统一恢复workload contract下比较R0/R1/R2/R4/R5，不是五套完全不同数据workload。
- Phase5只使用一套exact-Radix scheduler trace family，在rho设置下比较S0-S4；prefetch矩阵固定S4+HiCache比较P0-P3。
- Phase5没有执行R0/R1/R2/R4/R5，也没有做`5 recovery × 5 scheduler`；该组合属于Phase6。
- Phase5 workflow无任何有损KV恢复；hit是exact Radix/HiCache，miss是dense。
- pressure定义是reusable KV working set / 实测GPU KV pool capacity，不是整卡VRAM占用率。
- 实测capacity约13,130 tokens，实际rho=`1.153/1.537/2.075/3.073`，所有正式setting都有真实eviction和通过的pool reset invariant。
- Phase6计划已重写但尚未启动：先接通exact+approx cross-store eviction/allocator，再做固定workload capacity sweep。
- Phase6主策略缩减为S0 vs S4，P0默认；S2仅诊断，P1-P3后置。
- recovery候选分为R0 speed ceiling、R1-k32 practical、R2 precomputed oracle、R4 anchor diagnostic；不做五乘五。
- Phase6必须固定对象集合、只调capacity，并同时报告logical/physical rho。
- S1-S3与P1-P3没有彻底取消，只从大矩阵移到revalidation gate。
- Scheduler gate：final practical recovery在body2048、rho1.5/3重跑S0-S4；满足mean>=5%且p95不恶化才晋级。
- Prefetch gate：S4+HiCache winner在body2048、rho2/3重跑P0-P3；没有真实async overlap时只作安全canary，不作性能claim。
- `PROJECT.md`中的Phase6主计划现已直接包含P6-3.5与P6-5.5，不再只存在于后续补充说明。
- 当前计划文件：`IMPLEMENTATION_PLAN_LATEST.md`（**V4** / Latest）。（历史快照中的“V2”已过时。）
- 旧版计划文件：`IMPLEMENTATION_PLAN_V1_ARCHIVED.md`（V1 / Archived）。
- 原时间戳文件名保留为兼容指针，不是执行依据。

### 2026-07-24T01:25:06-07:00 Phase 5 已获授权并开始

- 用户明确要求：Phase 4 若已完成，则无需再次等待，直接自主执行 Phase 5，全部完成后再汇报。
- Phase 4 的权威完成状态为：R0/R1/R2/R4/R5 已收尾；R3 Cache-Craft 按既有决定 defer，不阻塞 Phase 5。因此此前人工确认门已解除。
- 当前执行范围固定为 S0-S4 eviction/scheduler 与 P0-P3 prefetch：
  - S0 LRU；
  - S1 steps-only；
  - S2 Belady oracle next-use；
  - S3 recovery-aware value density；
  - S4 hierarchical object policy；
  - P0 off、P1 free-space-only、P2 dead-object-only、P3 oracle next-stage。
- 必须使用独立 cache-protection metadata，不能复用 request scheduling `priority`。
- 当前先审计历史 `feature/workflow-priority` donor 与 frozen common-core 接线点，再创建独立 Phase 5 worktree/branch；随后完成 Docker 测试、真实 SM75 高压力筛选、中央日志、提交、显式 `ccdd2023` push 和文档收尾。
- 在 Phase 5 全部完成前不需要中途向用户索取范围确认；遇到实现细节歧义按上述保守原则自主决策。
- 当前实现 worktree/branch 已建立：`worktrees/scheduler-policies` / `research/scheduler-policies`，基于 `research/epic-legolink@984bfd873`。
- S0-S4、P0-P3、独立 metadata、absolute next-use、reusable-prefix boundary、atomic subtree victim、metrics 和正式 runner 均已实现。
- CPU regression `224 passed + 27 subtests`；SM75 HiRadix targeted tests `3 passed`。
- S0/S1与P0-P3真实smoke均通过；P2/P3机械验证主动load/eviction与零pool leak。正式S0-S4四压力矩阵正在运行。
- 正式S0-S4四压力矩阵已完成；S4在rho1.5/2/3的mean speedup约`1.33x/1.15x/1.15x`，是唯一跨高压档稳定优于LRU的策略。
- 正式S4+P0-P3矩阵已完成；P2/P3真实主动load/eviction，但无稳定TTFT收益且p95更差。当前默认固定为S4+P0。
- 两次额外restart正在运行，结束后进行format/lint、commit/push和最终文档收尾。

### 2026-07-23T06:47:21-07:00 R5 CacheTune完成；现有恢复实验暂停

- `research/cachetune` 已完成真实SM75 server路径、统一pressure runner与最终结果，远程分支为 `research/cachetune@8acb95e5a`；最终代码修复点为 `afcbcb027`。
- 最终口径固定为header64、body=`512/768/1024/2048`、target rho=2、512-token pressure fillers、S0 LRU、GPU-only、prefetch off；每个setting包含1次discarded warmup和2次formal repeats，并写入中央日志。
- 四个body的target-only / single-use combined speedup：
  - 512：`0.94x / 0.48x`
  - 768：`0.93x / 0.44x`
  - 1024：`1.50x / 0.76x`
  - 2048：`1.80x / 1.04x`
- 每个formal round均发生真实eviction；恢复cached tokens分别为`576/832/1088/2112`，与header+body严格一致；selected-token telemetry与controller决策一致，dense fallback为0，reset后pool invariant通过。
- 结论：CacheTune在body1024开始获得target-only收益，但只有body2048在把fresh preparation计入后仍有single-use正收益。当前实现仍是precomputed fresh-KV adapter，未实现论文完整frequency-domain selection、真实transfer/recompute overlap或通用inline selected-token hook。
- 结果：`benchmark/approx_kv/results/phase4-r5/sm75-unified-pressure.json`及四个`sm75-body*-rho2.json`。
- R0/R1/R2/R4/R5当前恢复实验已收尾；R3继续defer。现在必须停止，不得进入任何scheduler/eviction/prefetch实现或实验，等待用户明确批准。
- 阶段slides仍按用户要求排除CacheTune，不因本次技术收尾自动修改。

### 2026-07-22T16:41:21-07:00 R3 Cache-Craft：迁移 allocate_recovery_slots + 统一 Phase 4 contract（本地提交，未 push）

- worktree `/home/chris/Workspaces/kvcache-research/worktrees/cachecraft`，分支 `research/cachecraft`；本次本地提交 `57fc991fc`（前一次已 push 基线 `ca054f7a8`）；全部 git/测试/lint 在 Docker CPU-only 容器内完成，未启动 GPU，未 push。
- 已迁移共享 `allocate_recovery_slots`（从 R1 EPIC donor移植，只移植该 helper 本身）到 common runtime 的 `restore_request_prefix` 与 Cache-Craft partial-repair 的 `restore_request_via_cachecraft`，均在分配恢复 buffer 前先驱逐 exact Radix victims；新增高压力/无泄漏测试各 2–3 个。
- 审计出比既有 docstring 更精确的生产阻塞：`schedule_batch.py` 从不检查 `metadata.plugin`，对任何真实请求 `restore_request_via_cachecraft` 目前零可达性。新增 `cachecraft_capability.py::inspect_scheduler_dispatch_capability()`（零网络零 GPU 源码内省），当前诚实返回 `supported=False`。
- 新增统一 Phase 4 contract 模块 `cachecraft_workloads.py`（header 0/32/64/128/256、body 512/768/1024/2048、body>512 按 <=512-token segments、mem_fraction_static=0.35、rho 约 0.9/1.1/1.5/2/3、S0 LRU/GPU-only/prefetch-off、warmup1、formal repeats默认4/最少2）与 GPU-free 非-prefix 乱序 workload builder，为未来真实 hook 准备。
- 新增诚实的 `run_phase4_cachecraft_pressure.py` pressure runner scaffold：能力检查在最前面，当前始终走 blocked 路径（`status:"blocked"` 中央日志、零网络/GPU调用、不产出结果文件、exit code 3）；"真实运行"代码路径完整实现同一 settings/warmup/repeats/央志 contract 但结构性不可达，仅 fake transport 单元测试，绝不伪造 server 成功。
- README 新增 Phase 4 小节（未覆盖既有 Phase2/Phase3 内容）；目标测试集合格式化前后各跑两遍稳定 `114 passed/0 failed`；black/isort/ruff 通过（唯一例外是一处早于本次改动就存在、与本次 diff 无关的未使用 import，未做无关修复）。
- 详见 `PROJECT.md`/`TRACKING.md` 对应条目。

### 2026-07-22T06:50:12-07:00 Phase 4 权威门禁状态

- Phase 1–3已完成并冻结；当前保留的Phase 4恢复实验已收尾，Phase 5仍严格blocked并等待用户批准。
- 所有 research worktree 均基于 `experiment/common-core@674278379`。当前远程分支：
  - R0 `research/raw-rope@61c39791e`
  - R1 `research/epic-legolink@984bfd873`
  - R2 `research/cacheblend@e6dd5eab3`
  - R3 `research/cachecraft@d1110066a`
  - R4 `research/kvcomm@cd81c3e92`
  - R5 `research/cachetune@8acb95e5a`
- R0统一contract已push；独立GPU验证body1024/2048、header64、rho≈2分别`1.73x/2.07x`，完整k0矩阵由R1同物理路径覆盖。
- R1生产in-request seam、长body与真实eviction压力已完成：
  - 真实Qwen3/torch-native、TP/PP/DP=1，临时req-table row + leading-k `ForwardBatch`；
  - k=`0/2/4/8/16/32`，head=`0/16/32/64/128`，body=`128/256/512`；
  - body≤512时k>0负收益，但长body发生crossover；
  - body1024/rho≈2：k0 `1.70x`、k32 `1.53x`；
  - body2048/rho≈2：k0 `2.07x`、k32 `1.98x`；
  - rho≈0.9–3全部发生真实eviction，收益稳定；
  - header0/32/64/128/256全部通过，较大header相对speedup反而提高；
  - 结果：`sm75-inrequest-matrix.json`、`sm75-eviction-pressure.json`。
- 共享关键修复：approx recovery allocation必须先`evict_from_tree_cache`；长body canonical source在SM75上按≤512-token segments注册。
- R2统一contract已完成：ratio1/5/15/30%、header/body/rho全矩阵；body1024 ratio1% target `1.64x`但single-use combined `0.82x`；body2048 target `2.02x`、combined `1.14x`。仍依赖precomputed fresh adapter。
- R3按用户决定当前**DEFER/SKIP**：保留`d1110066a`的CPU core、allocation和blocked runner；因无scheduler dispatch、production attention profile与selected-token recompute hook，不做GPU、不再阻塞当前Phase4/5门禁。
- R4 KVCOMM已升级到统一Phase4 contract并push：
  - header=`0/32/64/128/256`，body=`512/768/1024/2048`，rho=`0.9/1.1/1.5/2/3`；
  - long body按≤512-token multi-placeholders，每个placeholder有target base、2个anchor bases和2个context deltas；
  - body1024/rho≈2：target-only约`1.37x`，setup约1.08s，约14次reuse break-even；
  - body2048/rho≈2：target-only约`1.76x`，setup约2.16s，约6次reuse break-even；
  - body1024在peak rho≈1.03–3.11保持约`1.36–1.38x`；
  - body1024/rho2四次formal request均验证copied tokens=1024、cached=1088、0 fallback；
  - 结果：`sm75-server.json`、`sm75-unified-pressure.json`。
- CacheTune论文控制器已用alphaXiv MCP复核：`T_layer(r)=max(rNt_c,(1-r)Nt_i)+t_o`，`r0=t_i/(t_c+t_i)`，再以calibration TTFT做warm-start golden-section search。必须分开论文15%质量下限和本项目0% speed-only下限。
- 历史CPU-only“完成”条目保留作过程记录，但不得再用作Phase 4 server完成证据。六条路径尚未完成统一Phase 2 high-pressure评测。
- 中央test/benchmark日志固定为 `/home/chris/Workspaces/kvcache-research/results/BENCHMARK_RUN_LOG.jsonl`。所有后续setting必须先warmup、正式重复至少2次并保存原始样本；当前GPU默认4次。
- High-pressure门已在R1完成：header=`0/32/64/128/256`，body=`512/768/1024/2048`，peak rho约`1.0/1.23/1.62/2.12/3.13`。
- **Phase 5人工确认硬门**：Phase 4全部完成后必须停止并取得用户明确授权；未经授权不得进行任何Phase 5 scheduler/eviction/prefetch实现、测试、benchmark或分支操作。
- 阶段报告slides位于`research/PHASE4_STAGE_REPORT_SLIDES.md`，当前8页brief版；最终human-tone pass已完成，CacheTune不在本版slides中。

### 2026-07-22T04:15:00-07:00 Phase 4 R1 EPIC/LegoLink 实现完成（CPU-only，服务器接线含明确记录阻塞）

- worktree `/home/chris/Workspaces/kvcache-research/worktrees/epic-legolink`，分支 `research/epic-legolink`，从冻结 common core `674278379` 创建；本地提交 SHA `dd4f54919e2c6cddf56383c3caaf4b2376bb62aa`，未 push。
- 核心机制：`epic_recompute.py::LayerwiseEpicExecutor` 对每一层先真实调用 `layer.forward(...)` 重算 leading-k 部分，再在同一层内调用新增的 `radix_backend.py::copy_and_rotate_layer()`（只搬运/旋转该单一层，不用融合的 `move_kv_cache`）搬运 body，逐层交替，而非规划 k 值或整体一次性复制。`EpicRecomputeStats.genuinely_layerwise` 机械校验真实调用交替顺序，检测并拒绝任何重排的 success-shaped stub。
- 支持 k∈{0,2,4,8,16,32}（`config.py`）与 attention-sink 语义（`epic_plugin.py::EPICLeadingKPlugin`/`carve_leading_k`）；k=0 直接复用 R0 精确路径（`runtime.py` 重构后的 `finalize_copy_reuse()`），不重复实现；exact-cache-first、任意不支持模型/布局 dense fallback、末 token 永远真实 forward、不写入 exact Radix 均由 `epic_runtime.py::restore_request_prefix_epic()` 保证，guard 结构与 common-core `restore_request_prefix()` 一致。能力门 `epic_capability.py` 基于 AST 签名核对，对照真实 `qwen3.py::Qwen3DecoderLayer.forward` 验证通过。
- 服务器接线（config-gated，默认关闭，零行为变化）：`scheduler.py` 绑定 `model_runner`（复刻既有 `bind_residency_backend` 接线模式）；`schedule_batch.py` 按 `epic_enabled` 分派到 EPIC 钩子或原 R0 钩子。唯一诚实记录的未解决生产阻塞：`EpicForwardBatchFactory` seam（构造独立 leading-k-only `ForwardBatch`）未绑定且未证明无 GPU 下安全，每次尝试在此安全 dense fallback，不伪造成功。
- 新增 28 个测试（`test_epic_leadingk.py`），含真实逐层交替顺序证明、stub 检测回归、k=4 全链路真实 recompute+copy 证明（fake layer 用真实张量仿射推导新 K/V，非复制固定值）、k=0/2/4/8/16/32 全扫描。连同既有 15 个 approx_kv baseline 测试，Docker CPU-only 容器内共 43 passed（含 12 subtests）、0 failed；`test_approx_kv_runtime.py` 等因预先存在的重 CUDA 依赖链无法在轻量环境导入（与本次改动无关），已用 scratch 副本手工回归验证。
- 明确排除 CacheBlend/Cache-Craft/KVCOMM/CacheTune/scheduler policy；固定 S0 LRU/GPU-only/prefetch-off、Phase2 dataset 未被触碰，无 accuracy metric。
- 详见 `PROJECT.md` 与 `TRACKING.md` 对应条目。

### 2026-07-22T03:40:00-07:00 Phase 4 R0 Raw+RoPE 实现完成（CPU-only）

- worktree `/home/chris/Workspaces/kvcache-research/worktrees/raw-rope`，分支 `research/raw-rope`，从冻结 common core `674278379` 创建；本地提交 SHA `41c4c0b25`，未 push。
- 明确定位：R0 是速度上限（speed-only upper bound），非忠实 KVCOMM 复现；不引入 EPIC/CacheBlend/Cache-Craft/KVCOMM/CacheTune/scheduler/prefetch 逻辑，无 accuracy metric。
- 新增 `python/sglang/srt/mem_cache/approx_kv/raw_rope.py`：`RawRoPERecoveryPlugin` 实现 common-core `RecoveryPlugin` 协议 + 纯函数 `build_raw_rope_plan()`/`select_contiguous_segments()`。原始 K/V copy + 整段 RoPE 位置重定位（zero/positive/negative delta 统一按有符号旋转角处理）；覆盖范围在"本次尝试的连续段"内是 all-or-nothing；`RawRoPERecoveryUnavailable` 用于缺失 segment 或已收窄 run 内部仍有 gap 的场景。诚实记录已知硬限制：只恢复锚定在 exact-prefix 边界的**前导连续段**；遇到不连续 segment，`select_contiguous_segments`（在 `runtime.py` 里、调用 plugin 之前执行）会把恢复范围裁剪到第一个 gap 之前，gap 之后的部分完全不尝试（既不静默修复，也不强制整请求 dense fallback），交由调度器当作普通 prefill 处理——这与最初草稿的"gap 即整请求 dense fallback"假设不同，是调试测试后修正的准确表述。
- 改动 common-core（仅新增/门控，未修改冻结不变式）：`config.py` 新增 `raw_rope_plugin_enabled` 字段 + `SGLANG_APPROX_KV_RAW_ROPE` env 门（需要 `core_enabled`）；`manager.py` 在门开启时自动注册 `RawRoPERecoveryPlugin()`；`runtime.py::restore_request_prefix()` 重构为通过 `RecoveryPlugin` registry 派发（把此前硬编码的 inline plan 构造正式化为 plugin 分发），I/O residency promotion（`ensure_device`）保留在 orchestration 层，因为 `RecoveryPlugin.build_plan` 协议本身不暴露 manager/backend 访问；exact-match-first（`schedule_batch.py` 恒在调用本函数前先跑 Radix 精确匹配）、末 token 永远真实 forward、all-or-nothing dense fallback 不变式均保持不变。`__init__.py` 导出新增公共 API。
- 新增 18 个分支专属 CPU 测试 `test/registered/unit/mem_cache/test_raw_rope_plugin.py`：覆盖 zero/positive/negative RoPE delta（含逐 bit 旋转数值校验）、连续多 segment 恢复、dense/exact head 之后的 interior segment 恢复、显式 plugin 门（开/关）、缺失/不连续覆盖的正确处理。
- 新增可复现、无需 GPU/server 的 canary：`benchmark/approx_kv/run_r0_raw_rope_cpu_canary.py` + `benchmark/approx_kv/results/phase4-r0/cpu-canary.json`，直接在进程内驱动真实 `restore_request_prefix()` 请求路径，token 序列来自真实 Phase 2 24-object catalog（`benchmark.approx_kv.workloads.build_object_catalog` + 真实 Qwen3-0.6B tokenizer），无 accuracy metric，只验证结构正确性（恢复 token 数、独立复算的 RoPE 旋转逐 bit 比对、value 原样拷贝、末 token 保留、门/gap/缺失 segment 处理）；8/8 场景通过。
- 调试过程中修正了两类真实 bug/测试误解：(1) 早期测试假设 exact_prefix_length 可以与 segment 起点不一致，实际 `select_contiguous_segments` 要求前导 segment 必须紧贴 exact 边界，否则整体视为"exact，无需恢复"；(2) fake harness 里 `ensure_device` 走 host residency 提升会真实调用 allocator 二次分配，若 next_index 起点与"位置即物理 index"约定的源数据区间重叠会导致 IndexError，需要给 allocator 起点留足够 headroom。
- 测试证据：Docker `ghcr.io/ccdd2023/sglang@sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781`（CPU-only，未启动 GPU server）内运行 `test_approx_kv_core.py`/`test_approx_kv_runtime.py`/`test_approx_kv_integration_source.py`/`test_approx_kv_hicache_backend.py`/`test_raw_rope_plugin.py`：42 passed，`test_approx_kv_cuda.py` 1 skipped（无 CUDA），0 failed，无回归。`ruff`（F401/F821/UP037）与 `isort` 对全部新增/改动文件通过（`config.py` 一处 UP037 为改动前既有问题，未触碰）。
- 诚实阻塞点：真实 GPU 上的 RoPE 正确性（针对真实模型前向）与 TTFT 基准测试未在本次 CPU-only 会话执行，需要主会话 GPU 验证；本任务按要求未启动 GPU server（其它 research 分支共享主机 GPU 并行工作）。
- 详见 `PROJECT.md` 与 `TRACKING.md` 对应条目。

### 2026-07-22T03:25:00-07:00 Phase 4 R3 Cache-Craft 实现完成（CPU-only）

- worktree `/home/chris/Workspaces/kvcache-research/worktrees/cachecraft`，分支 `research/cachecraft`，从冻结 common core `674278379` 创建；本地提交 SHA `e2b7d047e`，未 push。
- 新增 `python/sglang/srt/mem_cache/approx_kv/cachecraft_*.py`（同层平级文件，未改动任何 common core 冻结文件）：
  - `cachecraft_metrics.py`：忠实实现论文 Eq.(3)(4) inter/intra attention 求和、Eq.(6) Prefix Overlap Score β、Eq.(7) 基于 Kendall's Tau 的 Order Penalty γ、Eq.(8) 调整后 β'、Eq.(9)-(10) 逐层平均的 a(Ci)/b(Ci)、Eq.(11) CCI=sigmoid(ā/b̄)、Eq.(12) CFO、Eq.(14) top-N selected-token 选择，以及 direct-reuse/partial-repair/full-recompute 决策规则。
  - `cachecraft_attention.py`：真实（非占位）dense causal self-attention（genuine softmax(QK^T)+下三角掩码）捕获，用于从真实注意力权重构建 chunk profile。
  - `cachecraft_plugin.py`：`CacheCraftPlugin` 实现 common-core `RecoveryPlugin` 协议，按决策产出 direct-reuse（整段 copy）/full-recompute（整段 dense）/partial-repair（`dense_ranges`+`copied_spans` 混合）三种 `KVReusePlan`。
  - `cachecraft_recompute.py`：`CacheCraftRecomputeBackend` 包装真实 `RadixKVTransferBackend`，使 partial repair 的 `dense_prefill` 真正调用 `ChunkRecomputeHook.recompute(...)`（而非只记录 fallback 原因），并校验 hook 结果的完整性/RoPE 对齐。
  - `cachecraft_runtime.py`：`restore_request_via_cachecraft` 复用 common-core exact-cache-first/末 token 真实 forward/dense fallback 结构，串联 plugin 决策与真实 recompute backend；`schedule_batch.py` 已有的 `skip_radix_cache_insert` 和 `request.py` 的 `validate_prompt_length`（reusable_limit=prompt_length-1）无需改动即天然满足"近似结果不进 exact Radix"与"末 token 必真实 forward"。
- 48个新CPU-only测试（Docker `ghcr.io/ccdd2023/sglang@sha256:0be6e16e...`，`PYTHONPATH` 需保留镜像自带 `/opt/sm75-site` 前缀避免 transformers 版本错乱）：证明上下文/顺序变化真实改变 CCI/β/γ/CFO/决策（含仅改变顺序即翻转决策的用例）、真实 causal attention 与 chunk profile 捕获数值正确且对顺序敏感、`CacheCraftRecomputeBackend.dense_prefill` 真实调用 recompute hook 并写入可区分标记值（同时拒绝不完整/RoPE错位的 hook 结果）、plugin 三种决策分支产出正确 `KVReusePlan`、端到端 runtime 测试证明 partial repair 对被选中 token 真实调用 recompute hook、对其余 token 走真实设备 copy，且无 hook 时安全 dense fallback（不泄漏分配、不污染状态）。连同既有16个approx_kv baseline测试，共 64 passed/0 failed。
- 诚实阻塞（写入模块 docstring）：(1) 生产侧无独立可调用的 selected-token recompute 钩子——`ForwardMode.TARGET_VERIFY` 只在 speculative-decoding worker pipeline（`eagle_worker_v2.py`/`spec_utils.py`）内部可达，不是通用 request-level API；`recompute_hook` 在真实 GPU server 上目前只能是 `None`，partial repair 会安全 dense fallback。(2) 冻结的 wire-level request schema 无字段表达新 prompt 的 chunk order，`cachecraft_runtime.py` 通过 out-of-band 请求属性 `req.approx_kv_new_prefix_order` 读取。(3) 未改动 `schedule_batch.py`/`radix_cache.py` 做 scheduler dispatch 接线——因真正的 recompute hook 尚不存在，现在接线不会产生功能性差异，且本次会话不允许 GPU/并发 server 验证，风险/收益不对等，故推迟。
- 详见 `PROJECT.md` 与 `TRACKING.md` 对应条目。

### 2026-07-22T03:10:00-07:00 Phase 4 R2 CacheBlend 实现完成（CPU-only）

- worktree `/home/chris/Workspaces/kvcache-research/worktrees/cacheblend`，分支 `research/cacheblend`，从冻结 common core `674278379` 创建；本地提交 SHA `91874f18b`，未 push。
- 新增 `python/sglang/srt/mem_cache/cacheblend/` 包：真实 HKVD 测量（K deviation 相对 L2）+ gradual filtering、`LayerRecomputeCoordinator`（每层恰好一次 batched recompute，拒绝逐 token/部分覆盖）、`CacheBlendConfig`/能力门/env 注册、`restore_request_prefix_cacheblend` 真实请求路径（exact优先→segment overlap load→baseline copy+RoPE→HKVD测量→选中token逐层recompute→末token留真实forward→不支持布局dense fallback、不写入exact Radix）。
- 修复了 `select_hkvd_tokens` 中候选池收窄后 score/candidate 错位的真实 bug。
- 46个新CPU-only测试（Docker `ghcr.io/ccdd2023/sglang@sha256:0be6e16e...`）证明HKVD分数驱动1/5/15/30%选择、每层恰好一次batched recompute、能力门dense fallback无泄漏、末token不被恢复、多segment load/recompute overlap接口生效；连同既有24个approx_kv测试共66 passed/1 skipped/0 failed。
- 诚实阻塞：ModelRunner无“任意token子集与其余cached前缀交错、每层一次batched前向”钩子；生产注册`probe_backend`/`recompute_backend`为`None`，能力门正确dense fallback。未做GPU/server端到端或并行运行。
- 详见 `PROJECT.md` 与 `TRACKING.md` 对应条目。

### 2026-07-22T02:14:26-07:00 Phase 3 门禁完成

- frozen common core：`experiment/common-core@6742783798ab0b41ce4670bb48d423216ba2681c`。
- shared机制已完成：identity/store/lease/slot ownership、exact isolation、CPU/HiCache residency、async H2D event、K/V copy、RoPE、fallback、last-token forward、cleanup、metrics、plugin/scheduler metadata接口。
- paper-specific recovery与scheduler代码均未进入common core。
- targeted tests：41 passed；SM75 fresh canary与stream abort通过。
- canary reusable prefix 513 tokens；两次host export/H2D/copy共1,026 tokens；两次dense fallback共1,026 tokens。
- final pool fully accounted，exact radix未被approx请求污染。
- 结果：`benchmark/approx_kv/results/phase3/sm75-canary.json`。
- Phase 3验收通过；Phase 4六条research worktree必须全部从`674278379`创建，Phase 5不得提前。

### 2026-07-22T00:03:07-07:00 Phase 2 门禁完成

- `experiment/phase2-benchmark` 当前头：`05bb93bda`；full matrix runner SHA：`333ebb65710a629ee8f859a7182db5f471c3e38c`。
- 24-object catalog、fixed probe、workflow retry/filler/fan-out trace、rho calibration、TTFT/metrics/reset runner均已实现。
- full matrix：5 rho × 3 restart，15/15 run、471/471 measured requests，完成率100%，clean/idle/reset全部通过。
- actual reusable rho：`0.813/1.007/1.514/2.017/3.023`；physical estimate：`0.870/1.067/1.569/2.094/3.152`。
- 低压三次均0 eviction；`rho>=1.007`稳定eviction且restart间token count完全一致。
- 24-object cached-token校准误差为0；正式metrics使用boundary-only。
- compact结果：`benchmark/approx_kv/results/phase2/sm75-summary.json`。
- Phase 2验收通过；当前唯一允许开始Phase 3 common-core，Phase 4–5不得提前。

### 2026-07-21T22:19:43-07:00 Phase 1 门禁完成

- `experiment/phase1-image` 当前头：`dc09064ab`；镜像源码 SHA：`5a0fd2606bb62c6bcca004a4b2784ace745a580a`。
- SM75 image digest：`sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781`；CI `29892292070`。
- SM80/SM120 runtime digest：`sha256:2e36099165cedb0d328c98ee6c37f88c7c626d1a953a35de28748d1aa6183482`；CI `29892292080`。
- 两个镜像均使用 `crane v0.20.3` 流式组装并公开推送到 `ghcr.io/ccdd2023/sglang`，避免在标准 runner 解包大型 base。
- SM75 最终镜像已通过真实 CUDA fallback、Qwen3-0.6B health/model-info/1-token chat。
- SM120 image 已完成 CUDA13/PyTorch/Transformers/sgl-kernel/SM100 binary 和 native gate 静态验证；真实 SM120 GPU smoke 后置到 RTX PRO 6000。
- manifest：`ccdd2023/sglang:experiment/phase1-image/docker/phase1-image-manifest.json`。
- Phase 1 七项验收全部通过；当前唯一允许开始的是 Phase 2，Phase 3–5 不得提前。

### 2026-07-21T20:50:25-07:00 实验逻辑修正

- R0 raw是唯一server end-to-end路径；R1 EPIC、R2 selective、R3 KVCOMM目前只有planner/microbenchmark，不能标为完成。
- 下一步优先完成R1-R3真实repair/reconstruction执行，不先去Pro 6000。
- recovery完成后构建多对象 `rho≈1/1.5/2/3` dataset，再把S0-S4接入真实eviction。
- HiCache和prefetch放在recovery+scheduler归因清楚后。



### 2026-07-21T18:23:24-07:00 实施快照

- `ccdd2023/sglang:latest-main` 最终头：`f1e91b9cb80d9d4c036099fd0fa23a03400769e1`。
- GitHub guest source CI：`29888035426`，结论 success。
- SM75 read-only/tmpfs guest 已通过 Qwen3-0.6B、CUDA fallback、Radix copy/RoPE 和 sequential high-pressure MVP。
- raw whole-prefix speed-only 在 `rho=1.533/1.888` 相对 exact 分别降低 TTFT `8.57%/7.63%`；在 `rho=0.840` 回归 `1.10%`。
- compact results：`benchmark/approx_kv/results/sm75/`。
- Pro 6000 runner 已实现，但远程实例仍因 Vast 凭据未配置而阻塞。
- 不再在宿主机执行 Docker image build；不得删除现有 host worktree、images 或 builder。


历史成果接续、论文综合、Yu Guofan / AgentTemplateKV 分支审查，以及 **AST-indexed whole-codebase KV Cache 的 prior-art 与多模型 novelty 评估均已完成**。

最新 novelty verdict 见 `research/AST_INDEXED_KV_CACHE_NOVELTY_REPORT.md`；完整 33 步解释见 `research/AST_INDEXED_KV_CACHE_STEP_BY_STEP.md`。统一设计见 `research/RESEARCH_SYNTHESIS.md`，现有分支的完整审查见 `research/YU_GUOFAN_BRANCH_REVIEW.md`。

用户已明确 AST 不是主要研究切入点。**Git/repository/codebase-version-aware KV Cache 的 2024/2025/2026 年度 prior-art 调研已完成**，完整报告见 `research/GIT_CODEBASE_VERSIONED_KV_CACHE_PRIOR_ART.md`。

三年严格 A 类均为 0：截至 2026-07-13，本次检索未发现 Git commit、branch、worktree、repository/source version 或 patch epoch 被用作普通 attention-KV 的一等 identity、validity、invalidation 和 coherence 协议。

Vast.ai RTX PRO 6000 S 接入评估也已完成，报告见 `research/VASTAI_RTX_PRO_6000_WORKFLOW.md`。

用户对 source-version-aware attention-KV novelty 仍有担忧。高覆盖率分段复核最初覆盖 2024-01-01 至 2026-07-15；用户于 2026-07-15 取消 2025 年之前的 subagent。最终报告只保留 2025-01-01 至 2026-07-15 的七个 GPT-5.6 Sol Max 分段，见 `research/CODE_VERSION_KV_10_AGENT_PRIOR_ART_REPORT.md`。

2025-01-01 至 2026-07-15 的七个保留分段已**全部完成**：

- `version-scan-04`：2025-01-01 至 2025-01-06，A/B/C/D=`0/0/2/0`；
- `version-scan-05`：2025-01-07 至 2025-04-09，A/B/C/D=`0/3/6/2`；
- `version-scan-06`：2025-04-10 至 2025-07-11，A/B/C/D=`0/1/12/5`；
- `version-scan-07`：2025-07-12 至 2025-10-12，A/B/C/D=`0/5/14/2`；
- `version-scan-08`：2025-10-13 至 2026-01-12，A/B/C/D=`0/1/10/1`；
- `version-scan-09`：2026-01-13 至 2026-04-14，A/B/C/D=`0/8/8/4`；
- `version-scan-10`：2026-04-15 至 2026-07-15，A/B/C/D=`0/5/15/1`，最近 7/30/90 天 A 均为 0。

总计 105 篇主候选，A/B/C/D=`0/23/67/15`，七段严格 A 类均为 0。最终报告未发现 repository/source version 直接控制普通 attention-KV identity、validity、dependency invalidation、branch/worktree isolation 和 physical-tier coherence 的公开系统。`version-scan-01` 至 `version-scan-03` 均已停止，2024 结果不纳入。

KVCOMM SGLang 可行性调研已经完成，报告见 `research/KVCOMM_SGLANG_FEASIBILITY_REPORT.md`。结论是 GPU-only faithful functional reproduction 可行、难度 4/5；论文级性能复现条件可行、难度 5/5，受 H100 和精确环境阻塞。

`integration/coding-aware-prefetch` 最新分支审查也已完成。当前头为 `d4a7ec132`；最新增量是 middle-KV handoff API。分支已形成 shared segment identity/store、lease/resource lifecycle、copy-and-RoPE transfer、coding reuse plan、prefetch coordinator 和 Radix adapter，但实际 SGLang 接线仍只到 `CacheInitParams` 与 `RadixCache` manager 初始化/reset。官方状态是 `INTERFACE_COMPLETE / SERVER_CANARY_PENDING`，并非 faithful KVCOMM 或 production-ready coding-aware prefetch。

2026-07-21 用户进一步明确当前实验方向：**暂时不做 AST、自动切分、label、indexing、版本一致性或正确率优化，先专门研究有损跨上下文 KV 恢复与 KVFlow-style scheduler 在 high GPU cache pressure 下能否降低 TTFT。**

这里的“有损”不是量化，而是同一固定代码段在不同 role/prefix/context 下不做完整目标-context prefill，改用 raw KV reuse + RoPE、KVCOMM base/offset/anchor、EPIC fixed-k 或 CacheBlend/Cache-Craft/CacheTune 风格局部 repair。客户端 TTFT 是唯一性能主目标；最低要求只是请求不崩溃并返回首 token。

第一阶段只做 sequential `Architect -> Coder -> Debugger` 与 `Debugger -> Coder` retry；并发明确后置。必须同时尝试多条恢复路径和多种 scheduler，并使用 synthetic trace 的 oracle next-use 作为上界，不能预设哪种方法最好。

当前完整计划见`IMPLEMENTATION_PLAN_LATEST.md`；旧版见`IMPLEMENTATION_PLAN_V1_ARCHIVED.md`。

## 已完成

- 初始化 Git 仓库。
- 建立 `README.md` 文档入口。
- 建立 `PROJECT.md` 固定项目事实来源。
- 建立 `TRACKING.md` 讨论追踪时间线。
- 建立本交接文件。
- 建立 `.github/copilot-instructions.md`，固化后续会话的中文交流和文档维护规则。
- 确认系统中保存了 `ccdd2023` GitHub 账号。
- 显式使用 `ccdd2023` 身份验证目标仓库，确认具有 `ADMIN` 权限且默认分支为 `main`。
- 定位历史工作区 `/home/chris/Workspaces/kvcache-research`。
- 确认 `kvflow-sglang` 的 `feature/workflow-priority` 与远程同名分支同步，提交为 `5bb9afc9234aa9caa9df51e87f119e5bfaf186de`。
- 定位本机 SM75 Docker 运行版本 `sglang-running` 及其补丁和脚本。
- 通过 alphaXiv 获取并下载 KVFlow `2507.07400` 与 KVCOMM `2510.12872`。
- 两个独立 subagent 已完成两篇论文研究。
- 明确同名 KVComm 中 `2510.12872` 才是当前目标。
- 完成 KVFlow、KVCOMM、AST index、HiCache 与固定三阶段 workflow 的统一系统设计。
- 创建 `research/RESEARCH_SYNTHESIS.md`。
- 确认 Yu Guofan 对应 GitHub 账号 `flaminyu`。
- 审查最近两个月五个线性继承的研究分支及 121 个提交的归属边界。
- 完成 KVCOMM 原文机制与当前实现的逐项对照。
- 完成 CacheBlend、EPIC、KVFlow、Prompt Cache、LMCache、DroidSpeak 等论文的实际落地分类。
- 识别 cache ownership、lock、位置连续性、AST signature、Unicode offset、离线 token 对齐、host load 和 benchmark 公平性问题。
- 独立验证截断 signature 碰撞、Unicode byte offset 错位、context gate 缺表拒绝全部 exact matches，以及论文图表无法从已提交数据重生成。
- 创建 `research/YU_GUOFAN_BRANCH_REVIEW.md`。
- 一个 GPT-5.6 Sol Max arXiv research agent 和四个不同模型的 novelty/brainstorm agent 已全部完成。
- 四个评估代理在收到专项 prior-art 结果后又完成一次针对性修正。
- 确认 CodeComp `2604.10235` 与 FCGraft `2606.13097` 是 A 类直接先例；MEPIC `2512.16822` 与 MiniPIC `2606.13126` 是强邻近工作。
- 撤回 broad “首个 AST-aware/function-level/code-specific hierarchical KV cache”主张。
- 创建 `research/AST_INDEXED_KV_CACHE_NOVELTY_REPORT.md`。
- 创建 `research/AST_INDEXED_KV_CACHE_STEP_BY_STEP.md`，保存完整逐步推导。
- 将论文主线收窄为 evolving repository 的 versioned causal KV materialized views。
- 明确 AST 仅是可选结构信号，主要研究切入点是 repository/source version、cache consistency、incremental invalidation/rematerialization 和 serving lifecycle。
- 启动 2024/2025/2026 三个全 GPT-5.6 Sol Max 年度版本化 KV 文献调研代理。
- 三个年度 Sol Max 报告均已完成，并由主会话抽查关键论文全文。
- 创建 `research/GIT_CODEBASE_VERSIONED_KV_CACHE_PRIOR_ART.md`。
- 确认 PIE/Leyline 已覆盖 mutable prompt repair，Irminsul/MEPIC 已覆盖 content-addressed objects，FCGraft 已覆盖 function lifecycle，Code Isn't Memory 已覆盖 Git/Merkle index，Concordia 已覆盖 runtime checkpoint version。
- 确认仍未发现上述能力在 repository source-version-aware attention-KV coherence 中的统一实现。
- 完成 Vast.ai Docker hosting、账号连接、安全、成本、container lifecycle 和 RTX PRO 6000 实验价值评估。
- 验证本地 `lmsysorg/sglang:dev` 为 CUDA 12.9.1、PyTorch 2.9.1+cu129，并包含 SM120/compute_120。
- 验证 `sglang-running` 源码已有 RTX PRO 6000 特定 kernel 路径。
- 确认本地 `docker run` 脚本不能在 Vast standard instance 内嵌套执行；应使用 template/on-start 或 registry image。
- 决定采用本地控制面 + Vast.ai 按需 GPU 执行面的混合 workflow。
- 创建 `research/VASTAI_RTX_PRO_6000_WORKFLOW.md`。
- 将 2024-01-01 至 2026-07-15 均分为十个连续首次公开日期区间。
- 同时启动十个 GPT-5.6 Sol Max research agent，严格检索 repository/source-version-aware attention-KV 及 A/B/C/D 邻近工作。
- 建立 SQLite `prior_art_segments` 表记录 agent、日期范围、天数和状态。
- 创建 `research/CODE_VERSION_KV_10_AGENT_PRIOR_ART_REPORT.md` 报告骨架。
- 将最终分段复核范围收缩为 2025-01-01 至 2026-07-15；停止并排除前三个 2024 分段。
- 收到 `version-scan-05` 结果：11 篇候选，A/B/C/D=`0/3/6/2`，A=0 置信度约 0.92。
- 收到 `version-scan-08` 结果：12 篇候选，A/B/C/D=`0/1/10/1`，A=0 置信度约 0.87；2026 年范围已完整覆盖。
- 收到 `version-scan-07` 结果：21 篇候选，A/B/C/D=`0/5/14/2`，A=0 置信度约 0.90。
- 收到 `version-scan-04` 结果：2 篇候选，A/B/C/D=`0/0/2/0`。
- 完成 2025–2026 七分段最终报告、closest-prior-art matrix、claim 边界和 presentation summary。
- 完成 `integration/coding-aware-prefetch` 审查，确认 middle-KV handoff、shared data plane、coding policy、prefetch coordinator、Radix adapter、lifecycle 修复及其未完成的生产接线边界。
- 完成 Phase 4 R2 CacheBlend：独立 worktree `research/cacheblend`，本地提交 `91874f18b`；真实 HKVD 测量+gradual filtering、每层batched selected-token recompute、能力门、真实server请求路径（dense fallback on不支持布局，不写入exact Radix）；46个新CPU-only测试通过，连同既有测试共66 passed/1 skipped/0 failed；ModelRunner缺乏per-layer selective forward钩子的诚实阻塞已记录，未做GPU/server验证。
- 完成 Phase 4 R3 Cache-Craft：独立 worktree `research/cachecraft`，本地提交 `e2b7d047e`；忠实实现CCI/β/γ/β'/CFO/top-N selected-token(Eq.3-4,6-12,14)、direct-reuse/partial-repair/full-recompute决策、`CacheCraftRecomputeBackend`使partial repair真实调用selected-token recompute hook（非仅metadata/planning）、真实server请求路径（exact优先、末token真实forward、不支持布局dense fallback、不写入exact Radix，均复用common-core既有不变式无需改动）；48个新CPU-only测试通过（含端到端runtime测试证明真实hook调用与真实设备copy），连同既有测试共64 passed/0 failed；ModelRunner缺乏独立可调用selected-token recompute钩子（TARGET_VERIFY仅spec-decode内部可达）的诚实阻塞已记录，未做GPU/server验证，未接线scheduler dispatch。
- 完成 Phase 4 R0 Raw+RoPE：独立 worktree `research/raw-rope`，本地提交 `41c4c0b25`；`RawRoPERecoveryPlugin`（common-core `RecoveryPlugin` 协议）+ 纯函数 `build_raw_rope_plan`/`select_contiguous_segments`；原始K/V copy+整段RoPE重定位（zero/positive/negative delta统一按有符号旋转角处理）；`restore_request_prefix` 重构为经 plugin registry 派发（而非硬编码inline逻辑），显式 `raw_rope_plugin_enabled`/`SGLANG_APPROX_KV_RAW_ROPE` 门；诚实文档化硬限制：不连续segment时只恢复前导连续段、gap之后部分完全不尝试（既不静默修复也不强制整请求dense fallback）；18个新CPU-only测试+8/8场景CPU canary（真实Phase2 24-object catalog token，无GPU/server，无accuracy metric）通过，连同既有测试共42 passed/1 skipped/0 failed；GPU上真实RoPE正确性与TTFT基准未做，未启动GPU server。

## 下一步

1. Phase 5已完成，保持暂停，不自动进入Phase 6。
2. 下一阶段若获授权，按计划选择前两条恢复路径与S4/P0做cross-recovery组合，而不是重新展开完整笛卡尔积。
3. RTX PRO 6000复测、Phase 2统一dataset横评和R3深层接线仍未完成，属于后续范围。
4. 阶段slides保持当前8页版本并继续排除CacheTune与Phase5结果，除非用户明确要求更新。

## 必须遵守的约束

- 所有面向用户的回复使用中文；仅在代码、命令和技术标识确有需要时使用英文。
- 不使用日语。
- 所有更新、可共享思路、讨论结论、进度和计划写入 `PROJECT.md`。
- 每轮有效讨论追加到 `TRACKING.md`。
- 发生阶段切换、关键决策、功能完成、重大阻塞或下一步变化时更新本文件。
- 对 `ccdd2023/sglang` 执行 GitHub 操作时必须显式使用 `ccdd2023` 账号；不要假设当前默认账号正确。
- 不得在日志、文档或回复中暴露 GitHub token 或其他凭据。
- 本机是 RTX 2080 SUPER、SM75、8GB VRAM；所有 SGLang 测试必须在 Docker 中运行，并使用 `--runtime=nvidia --gpus all`。
- 源码和 Git worktree 保存在 host，并只读挂载进不同 container；guest 不保留重复源码。
- 依赖安装、镜像组装、测试、server 和 benchmark 必须在 Docker 或远程 true guest 中执行；host 禁止 image build、driver/DKMS/kernel module 修改。
- 当前实验不以正确率、代码质量、输出一致性或 KL 为目标；只要求请求不崩溃并返回首 token。
- 当前实验的论文事实只允许来自已配置的 arXiv/alphaXiv MCP。
- 当前首版不做并发；并发是找到有效 sequential 方法后的后置阶段。
- 所有test/benchmark必须写入中央日志 `/home/chris/Workspaces/kvcache-research/results/BENCHMARK_RUN_LOG.jsonl`，记录完整settings、代码/镜像/模型版本、原始结果路径和汇总。
- 所有正式benchmark setting必须先做discarded warmup，warmup不得混入formal samples。
- 每个benchmark setting必须正式重复至少2次；当前GPU默认4次，保留全部raw samples并使用p50等稳健统计量。
- KVCOMM 论文主实验使用 7B/8B 模型与 H100，本机只能先做小模型功能性复现，不能声称复现论文主指标。
- Vast provider 可技术性访问其 host 文件；使用 Secure Cloud、scoped key、只读模型 token，不把 GitHub 写凭据或唯一实验数据放在实例内。
- RTX PRO 6000 是 SM120/GDDR7 路径，不等于 H100 SM90/HBM；可验证系统机制，最终 H100 主张仍需 calibration。

## 当前关键决策

- `PROJECT.md` 保存当前有效事实。
- `TRACKING.md` 保存完整讨论时间线。
- `HANDOFF.md` 保存下一会话所需的最新快照。
- `ccdd2023/sglang` 是项目交流和 prototype 代码实现的统一仓库。
- 目标仓库操作使用 `ccdd2023` 身份；最近一次验证权限为 `ADMIN`。
- 固定 Coding Agent workflow 是 `Architect -> Coder -> Debugger`。
- 当前 acceleration-only 实验使用 `Architect -> Coder -> Debugger -> Coder -> Debugger` retry trace，并允许 sequential synthetic stress 变体；不改变项目固定 workflow 定义。
- 当前“有损 KV”指跨 context 近似恢复，不是量化、低比特或普通 KV pruning。
- 当前唯一主指标是客户端 TTFT；所有恢复与调度路径必须在同一 high-pressure harness 下比较。
- 当前必须同时尝试 raw+RoPE、EPIC-like、selective repair、KVCOMM anchor 和 hardware-aware selector，不预设最佳路径。
- 当前必须同时比较 LRU、steps-only、Belady oracle、value-density 和 hierarchical object policy。
- 当前 fork main 可 fast-forward 到 upstream 当前 SHA；实施分支固定为同步后的 `latest-main`。
- 当前源码隔离采用 host Git worktree，进程和依赖隔离采用 container；性能运行一次只允许一个 GPU container。
- KVFlow 用于 workflow-aware priority、eviction 和 CPU→GPU 调度。
- KVCOMM `2510.12872` 用于同一 artifact 在不同 role/prefix 下的 base-KV + offset/anchor 跨上下文复用。
- AST 用于 artifact 分段、索引和候选 gating；不替代 embedding-distance 安全判据。
- Codebase 预计算必须按 artifact/span 进行，不能把整个仓库视为一条单一 KV 序列。
- module/class/file 只作为 logical view；physical KV 使用非重叠 function/method、module preamble、class init/field block，超长单元才切 statement/basic block。
- dependency graph 复用现有 AST/LSP/compiler/build/test 工具，只生成 conservative invalidation cone；exact reuse 仍要求完整 causal-context fingerprint，近似路径仍需 probe 与 dense fallback。
- novelty 不在发明 AST 或依赖分析，而在 source/dependency event 到 attention-KV identity、coherence、rematerialization 和 tier state 的可执行协议。
- 系统不是 Git-only index：symbol/AST 与 embedding 负责检索，Git 负责版本，content hash 负责相等性，dependency graph 负责失效，causal-context signature 负责 exactness，physical index 负责 tier location。
- artifact 通常是 function/method 或 module preamble，不是单个 import/token；独立 artifact KV 在新前缀或新顺序下不能盲目 exact 拼接。
- physical KV 是逐层逐 token 的真实 K/V tensor pages，并绑定 artifact/source/context/model provenance。
- bootstrap 从当前目标 fixed SHA 开始，不从首个 Git commit 开始；全库 logical index 完整，physical KV 只为 hotset 稀疏物化，后续按 diff alias、失效、重算和 GC。
- function/method 是默认 logical artifact；exact reuse 以完整 causal prefix/context signature 为准，KVCOMM 以 placeholder span 为重建对象，物理分配/传输以 token pages 为单位。
- 不为每个函数固定保存“原文 exact + system-context exact”两份完整 KV；采用 exact bundle cache、canonical artifact base 和 bounded residual/anchor 三层。
- source dependency graph 用于检索/影响分析，prompt causal graph 用于记录实际前序和 exact invalidation。
- Prompt Compiler 把多个 artifacts 编译为稳定有序 bundle，Debugger 可 exact reuse 跨多个函数的大段 prefix；动态 patch/test/stack trace 尽量放尾部。
- Git 负责 snapshot/version visibility、dirty worktree、branch isolation、diff event 和 unchanged physical-page alias；不负责 KV 计算或 semantic retrieval。
- 固定 workflow 将 Git 变更后的 dependency cone 映射到 Architect/Coder/Debugger 的保留、预取和 rematerialization priority。
- Architect/Coder/Debugger 的 exact bundles 默认分别存储；跨角色共享 canonical artifact base，并通过 KVCOMM context offset/anchor 重建，失败则 dense。
- faithful KVCOMM offsets 可能接近完整 K/V 大小，需限制 anchor 数量；不能把它描述为天然小 residual。
- canonical base KV 是固定 reference prompt/context/position 下真实计算的普通 K/V tensor，不是新编码格式；只在该 reference 下 exact。
- “可变编码”不是 KVCOMM 原文术语；当前定义是 base KV、context-dependent delta、RoPE relocation 和受控重建。
- runtime 优先级必须是 exact cache、受控 KVCOMM reconstruction、dense fallback。
- `fix/placeholder-pool-activation` 不是 faithful KVCOMM，不作为实现基线。
- 最新分支的 L2 whole-slot exact path 有 token equality guard；C2 AST chunk path 的截断 signature + byte range 不满足完整 exact invariant。
- 可选择性继承 benchmark、telemetry、AST/HKVD、RoPE helper 和 priority eviction 经验。
- KVCOMM core、placeholder/chunk pool、offline writer/loader、context confidence 和 selective recompute 必须重写。
- CacheBlend 和 EPIC 的近似实现只能标为 inspired，不能沿用论文的质量结论。
- 论文 artifact 必须提交可重生成表格/图的 compact source CSV/JSON。
- CodeComp 已占据 CPG/AST 驱动的 KV retention；FCGraft 已占据函数级 KV object、stitching、patch/update 和 GPU/DRAM lifecycle。
- 不能把 AST index、函数级 KV、code-specific tier 或 workflow priority 单独写成首次贡献。
- 全库使用完整 logical index，但 physical KV 默认按 hotset 惰性物化。
- 新主线是 versioned causal KV materialized views；系统优先级是 dependency invalidation、持久 artifact/page lifecycle、calibrated cross-role reconstruction。
- role/prefix 改变时 suffix K/V 均可能变化；RoPE 只修位置，不能让任意 non-prefix code KV 变成 exact。
- AST 不是主要研究切入点；当前优先验证 repository/source version 是否已被用于 attention-KV identity、invalidation 和 cross-version lifecycle。
- 普通 content hash、runtime checkpoint version 和 repository source version 是不同概念，不能混为同一 prior art。
- 安全的 thesis 是 repository-version-aware attention-KV coherence，而不是笼统的“versioned KV cache”。
- 实验基础设施固定为 local-control/remote-execution；Vast 实例是短生命周期 worker，不是长期开发机或事实来源。
- 首轮 Vast 使用 on-demand Secure Cloud/Verified RTX PRO 6000 S；正式 benchmark 固定 image digest、Git SHA、model revision 和 machine manifest。
- 当前 Dockerfile 的 DeepEP arch list 不含 SM120；dense single-GPU 路线先行，DeepEP/MoE 后续单独修复。
- 2025–2026 七分段复核已经完成；结论是“本次检索未发现 A 类”，不能写成绝对不存在或法律意义的新颖性结论。
- `integration/coding-aware-prefetch` 的可继承部分是 segment identity/store、generation/lease/resource disposal、transfer invariant、Radix adapter 和 middle-KV handoff contract。
- 该分支未实现 KVCOMM canonical base、context-dependent `ΔK/ΔV`、multi-anchor interpolation 和 entropy/shareability gate；coding policy 也未接真实 AST/dependency signals。
- 当前只在 `RadixCache` 构造和 reset manager，没有 scheduler/request 自动调用链；成熟度固定表述为 `INTERFACE_COMPLETE / SERVER_CANARY_PENDING`。

## 历史实现位置

| 路径 | 分支 | 状态 |
| --- | --- | --- |
| `/home/chris/Workspaces/kvcache-research/kvflow-sglang` | `feature/workflow-priority` | 与 `ccdd2023/sglang` 远程同名分支同步 |
| `/home/chris/Workspaces/kvcache-research/sglang-running` | `fix/qwen3-0.6b-docker-sm75` | 本地 SM75 Docker 运行版本；远程未发现同名分支 |
| `/home/chris/Workspaces/kvcache-research/mini-sglang` | `main` | KVFlow 早期研究验证实现，工作区存在大量历史改动 |
| `origin/fix/placeholder-pool-activation` | `9e84d2f94` | AgentTemplateKV 最新研究线；仅作为审查档案和 helper donor，不作为实现基线 |
| `origin/integration/coding-aware-prefetch` | `d4a7ec132` | shared data plane 与 middle-KV 接口已成形；server canary、scheduler/HiCache 接线及 faithful KVCOMM 未完成 |
| `/home/chris/Workspaces/kvcache-research/worktrees/raw-rope` | `research/raw-rope@61c39791e` | R0统一contract完成；body1024/2048、header64、rho≈2为`1.73x/2.07x` |
| `/home/chris/Workspaces/kvcache-research/worktrees/epic-legolink` | `research/epic-legolink@984bfd873` | R1生产seam、长body分段source、eviction-aware allocation与rho≈1–3矩阵完成；body1024/2048 k32为`1.53x/1.98x` |
| `/home/chris/Workspaces/kvcache-research/worktrees/cacheblend` | `research/cacheblend@e6dd5eab3` | R2统一矩阵完成；body2048 ratio1% target `2.02x`、single-use combined `1.14x`；仍为precomputed adapter |
| `/home/chris/Workspaces/kvcache-research/worktrees/cachecraft` | `research/cachecraft@d1110066a` | R3 CPU core、allocation与blocked runner已push；当前defer，不产GPU结果 |
| `/home/chris/Workspaces/kvcache-research/worktrees/kvcomm` | `research/kvcomm@cd81c3e92` | R4统一header/body/rho完成；body1024/2048约`1.37x/1.76x`，setup break-even约14/6次 |
| `/home/chris/Workspaces/kvcache-research/worktrees/cachetune` | `research/cachetune@8acb95e5a` | R5真实SM75统一pressure结果完成；body1024/2048 target-only `1.50x/1.80x`，body2048 combined `1.04x` |

## 论文文件

| 论文 | 文件 |
| --- | --- |
| KVFlow `2507.07400` | `research/papers/KVFlow-2507.07400.pdf` |
| KVCOMM `2510.12872` | `research/papers/KVCOMM-2510.12872.pdf` |

## 关键研究文档

- `research/RESEARCH_SYNTHESIS.md`：统一技术理解、系统架构、论文边界、路线与风险。
- `research/YU_GUOFAN_BRANCH_REVIEW.md`：AgentTemplateKV 分支演进、KVCOMM 忠实度、代码问题、实验边界、论文地图和继承建议。
- `research/AST_INDEXED_KV_CACHE_NOVELTY_REPORT.md`：direct prior art、novelty verdict、causal correctness、机制排序、workflow contract、prototype 与 kill criteria。
- `research/AST_INDEXED_KV_CACHE_STEP_BY_STEP.md`：从问题定义到最终实施决策的 33 步教学式解释。
- `research/GIT_CODEBASE_VERSIONED_KV_CACHE_PRIOR_ART.md`：2024–2026 Git/source-version-aware attention-KV 年度调研、closest matrix 和 RepoKV-MVCC thesis。
- `research/VASTAI_RTX_PRO_6000_WORKFLOW.md`：Vast hosting、Docker 兼容、账号安全、offer 筛选、成本收益和分阶段实验流程。
- `research/CODE_VERSION_KV_10_AGENT_PRIOR_ART_REPORT.md`：原十代理任务及收缩后的 2025-01-01 至 2026-07-15 七分段复核、去重、全文复核和最终 verdict。
- `research/KVCOMM_SGLANG_FEASIBILITY_REPORT.md`：faithful KVCOMM 的可行性、难度、blocker、复用边界、实施路线和验证方案。
- `/home/chris/Workspaces/kvcache-research/KVFLOW_MIGRATION_PROGRESS.md`：历史 KVFlow SGLang 移植进度。
- `/home/chris/Workspaces/kvcache-research/DESIGN_NOTES.md`：历史 KVFlow 设计和 benchmark 结论。
- `/home/chris/Workspaces/kvcache-research/paper-reading/2510.12872-kvcomm.md`：历史 KVCOMM 论文笔记。
- `/home/chris/Workspaces/kvcache-research/paper-reading/2510.12872-kvcomm.code-agent-kv.md`：KVCOMM 与 Coding Agent 的历史映射。
- `/home/chris/Workspaces/kvcache-research/paper-reading/2510.12872-kvcomm.structural-distance-experiment.md`：AST 结构距离离线实验。

## 外部仓库访问

| 项目 | 当前值 |
| --- | --- |
| 仓库 | `https://github.com/ccdd2023/sglang` |
| 用途 | 项目交流与 prototype 代码实现 |
| 指定账号 | `ccdd2023` |
| 最近验证 | 2026-07-23T06:47:21-07:00 |
| 验证权限 | SSH身份为 `ccdd2023`；`research/cachetune` dry-run及实际push成功；历史API权限为 `ADMIN` |
| 默认分支 | `main` |
| 注意事项 | GitHub CLI 默认账号可能不是 `ccdd2023`，每次操作需显式选择身份 |

## 当前文件

```text
.
├── .github/
│   └── copilot-instructions.md
├── HANDOFF.md
├── PROJECT.md
├── README.md
├── research/
│   ├── RESEARCH_SYNTHESIS.md
│   ├── AST_INDEXED_KV_CACHE_NOVELTY_REPORT.md
│   ├── AST_INDEXED_KV_CACHE_STEP_BY_STEP.md
│   ├── GIT_CODEBASE_VERSIONED_KV_CACHE_PRIOR_ART.md
│   ├── YU_GUOFAN_BRANCH_REVIEW.md
│   └── papers/
│       ├── KVCOMM-2510.12872.pdf
│       └── KVFlow-2507.07400.pdf
└── TRACKING.md
```
