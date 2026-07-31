# 项目主文档

> 本文件是项目更新、可共享思路、讨论结论、进度、计划和决策的固定事实来源。

最后更新：2026-07-30T11:17:00-07:00

## 项目概况

| 项目 | 当前值 |
| --- | --- |
| 名称 | `code-agent-kvcache` |
| 阶段 | Phase7.5 C40计划已review并版本化；状态为Reviewed Candidate，等待用户授权 |
| 业务目标 | 在 SGLang 上比较多种跨 context 近似 KV 恢复与 workflow-aware cache scheduling，降低 Coding Agent TTFT |
| 技术栈 | SGLang、HiCache、Docker、KVFlow、KVCOMM、CacheBlend、Cache-Craft、EPIC、CacheTune |
| 默认分支 | `main` |
| 协作与原型仓库 | [`ccdd2023/sglang`](https://github.com/ccdd2023/sglang) |
| 仓库操作账号 | `ccdd2023` |

## 已确认的协作要求

- 与用户的所有交流均使用中文；代码、命令、API 名称等必要技术标识可保留英文。
- 不使用日语，也不使用非必要的英文叙述。
- 所有项目更新、可共享思路、讨论结论、进度和计划都必须写入本文件。
- 每轮有效讨论都必须追加到 `TRACKING.md`。
- 重大更新必须同步更新 `HANDOFF.md`，使全新会话可以直接接续。
- 重要状态不得只保留在聊天上下文中。

## 当前状态

### 2026-07-28T20:25:48-07:00 Phase7 final disposition

```text
phase7_execution = COMPLETE
engineering_status = VALID
r0_mechanism_status = NEGATIVE
w_system_behaviour = INCONCLUSIVE / DESCRIPTIVE
publication = READY WITH CAVEATS
```

- 原authorized执行：
  - 22 starts；
  - 1.310141804 GPU-equivalent hours；
  - 8个A8 supplements按ES-R0-MDE跳过；
  - rho3 conditional未执行。
- supplementary evidence correction：
  - 1个S0 start；
  - 0.098331816h；
  - 40/40 direct `unsupported <- store_miss`；
  - 单独计账，不改变原22-start execution set。
- final summary：
  - canonical SHA=`9d0aafcd776305c7ac679b14845e4a5c79e44c04bb696ecf2d83644fca4c2c69`；
  - file SHA=`05acb52d51ea68bcff95c392e13348f73292dad0921b154efaf1e02e150eb135`；
  - publication package HEAD=`0206f17b4255e4b248dafaaeb943be57428dae2f`。
- RESULT_MANIFEST：
  - `88/88`；
  - file SHA=`6b6b0af19b70b9958866dc9674102026b348b1d5a4254e7f2f9d732a77548a65`；
  - `known_gaps=[]`。
- final disposition：
  - path=`benchmark/approx_kv/results/phase7/PHASE7_FINAL_DISPOSITION.json`；
  - file SHA=`731ace74108bc6ec0b6beb160bb3a55fcdbaae2fd2959c1a3f3d0b7aff466af5`；
  - self SHA=`4013f054...`。
- final dual review：
  - Opus=`PASS_WITH_CAVEATS / READY`，open P0/P1=`0/0`；
  - Sol=`PASS_WITH_CAVEATS / READY_WITH_CAVEATS`，open P0/P1/P2=`0/0/0`；
  - authority SHA同步已闭合；
  - 完整original/exchange/cross/delta/final reports均已版本化；
  - 无额外GPU或offline work。

核心结论：

1. chunk4096下R0 request-path=`0.7723–0.9362x`，N8 full-setup=
   `0.6086–0.6419x`，mechanism=`NEGATIVE`，不得发布speedup headline。
2. chunk1024 sensitivity request-path median=`1.7370x`、N8 full=`1.1890x`，
   仅证明chunk coupling，不是机制固有收益。
3. W S4 vs S0仅描述性：
   - rho1.5 all=`1.0021x`，workflow=`1.0276x`；
   - rho2 all=`1.0250x`，workflow=`1.0442x`；
   - 比较为seed-matched但非相邻launch block；
   - R0臂45.9%–72.1%请求走dense fallback。
4. R4-like是synthetic 5x footprint proxy，不执行KVCOMM：
   - S0真实recovery仅12/122；
   - S4 registration diagnostic-unavailable。
5. natural-pressure reservation-failure fallback仍未证明；Phase7观察到0次自然
   reservation failure。

必须随结果保留的caveats：

- `practical=NONE`严格限定于被测scope；
- R0是ceiling，不是practical candidate；
- R2=`disabled_not_comparable`且未执行；
- `r2_like`/R4-like均为synthetic footprint profile；
- 独立复制单元是server restart，A8 primary每setting仅n=1；
- body1024 canary只有1个distinct output token，判别力有限；
- host/prefetch/async/HiCache未生成；
- P6-4 rho1.1/1.5/3仍限定chunk1024。

Phase8不会自动触发。若未来继续，必须另行版本化与授权。

### 2026-07-28T19:37:55-07:00 S0 correction与reconsolidation完成

- 专用correction rev2授权并执行1个S0 supplementary start。
- correction结果：
  - 40/40 approximate dense fallback；
  - 全部direct exclusive `unsupported <- store_miss`；
  - 无reservation failure、ambiguity、reset或inactive failure。
- correction elapsed=`0.098331816h`，单列为post-hoc evidence：
  - 不计入原22 authorized starts；
  - 不改原1.310141804h primary execution；
  - 原22 raw/log保持字节不变。
- 全部22 compact与summary已`--force`重生成：
  - summary SHA=`e2deec5afbbfffb7b22de84d7218e3ca1aac0ba7327abcad5b1ae402753e4134`；
  - engineering=`VALID`；
  - mechanism=`NEGATIVE`；
  - W=`INCONCLUSIVE/DESCRIPTIVE`。
- publication层已补W victim/footprint、R4 fallback/5x accounting、
  wave0 cell/profile/outcome/reason、A8 n=1与request-path定义、
  W fallback比例/non-adjacent protocol、p95与完整provenance。
- RESULT_MANIFEST=`79/79`。
- Sol/Opus targeted delta closure进行中。

### 2026-07-28T19:19:44-07:00 Correction治理delta review为PASS_WITH_CAVEATS

- 只读targeted delta review（Claude Opus 5 Max，无修改、无GPU、无server）。
- 审查范围：pin `a950ab91...`→HEAD `e3ee91ef8...`，correction manifest rev1
  self=`2a907fee...`，RESULT_MANIFEST，correction/runtime/consolidator。
- 上一轮两个P1均判定为CLOSED：
  - P1-1（correction manifest/evidence未纳入RESULT_MANIFEST）：两条目已在
    manifest内（`b33403304`/`7b92f02ae`），`build_result_manifest --check`
    复跑得`OK: 75 artifacts verified`；且consolidator现把RESULT_MANIFEST
    作为硬输入校验publication inputs的存在与hash，属结构性闭合而非一次性重建。
  - P1-2（pin同时含publication reconsolidation）：PROJECT/HANDOFF/TRACKING
    已披露pin范围与correction后`--force`全量重生成22 compact+summary、
    raw/log不改写；独立验证数值不变成立——pre-pin consolidator可逐字节复现
    已提交22 compact与summary，新旧输出跨470条key path比对无任何既有数值改变，
    差异仅为新增字段、重命名与hash字段。
- 可执行性验证（全部CPU、只读）：
  - `build_phase7_correction_manifest --check`=`OK pinned_blocked revision 1`；
    manifest自哈希重算=`2a907fee...`，与pin值一致。
  - runtime授权门6个用例全部正确拒绝：未授权rev1、错误setting、错误restart、
    错误original raw sha、缺correction flag、primary manifest误用。
  - 内存模拟authorized rev2通过校验，7个篡改向量（带blocker、错误supersede、
    放宽allowed_setting、改pinned setting、放宽post-pin allowlist、复用rev12
    primary pin、evidence含open P1）全部被拒。
  - post-pin变更3个文件全部落在4项allowlist内，worktree clean；envelope当前
    仅因review artifact尚未创建而阻塞，正是既定下一步。
  - consolidator按设计要求`--correction-dir`，correction未执行前不可端到端跑通；
    correction路径已有专用CPU测试覆盖（含拒绝多于1个supplementary raw）。
  - 测试复现：4个Phase7 targeted文件`119 passed`；
    `test_phase6_manifest.py`=`36 passed`与pinned CPU evidence一致；
    bench目录（排除2个GPU依赖文件）=`144 passed`。
- 结论：`PASS_WITH_CAVEATS`，open P0/P1=`0/0`；
  entry=`READY_AFTER_REVIEW_ARTIFACT_AND_AUTH_CORRECTION`。
- 剩余P2（不阻塞授权）：
  1. consolidator `engineering_basis`与wave0 S0 `status_reason`无条件声称
     correction已通过；当前数据集不可达，换数据集会成为虚假证据陈述。
  2. `_ensure_output_paths`保护raw/log/central/manifest，但未保护
     `RESULT_MANIFEST.json`、entry review与`evidence/*-cpu.json`。
  3. A8 `_numeric_summary`输出`n=16/2`，与同artifact的`n_per_setting=1`
     并列，易被误读为样本量。
  4. 新版summary删除`within_policy…independent_restart_values`，而rev12
     statistics规则为`report_all_three_values_and_min_max`（值仍可从
     `w_scheduler.per_restart`取得）。
  5. 文档未披露字段重命名与该删除项，只写了“数值不变”。
  6. correction manifest/review/correction CPU evidence未列入consolidator的
     `publication_inputs`，只靠`build_result_manifest --check`的全盘枚举兜底。

### 2026-07-28T18:33:37-07:00 Result review与S0 capacity correction

- 原Phase7执行已完成并版本化：22 starts / 1.31014 GPU-equivalent hours。
- Sol/Opus独立review、全文互换与cross-consolidation：
  - 核心数值与hash全部PASS；
  - publication package=`NOT_READY`；
  - accepted P0/P1/P2=`0/6/9`。
- 唯一GPU correction：
  - 原S0 wave0有40次dense fallback但未保存exclusive reason；
  - 原raw/log中reason已在metrics聚合时丢失，不能离线直接恢复；
  - 只需1个supplementary S0 correction；S4/A8/W/R4不重跑。
- 原22 raw/log与authorized rev12保持不可变；correction独立补充证据。
- 已实现并通过Docker测试：
  - capacity逐请求terminal reason采集与exclusive validation；
  - 专用correction manifest/review/authorization链；
  - W victim/footprint、R4 fallback、wave0 outcome、A8 n=1、
    W fallback比例、p95/provenance等publication修复；
  - targeted=`119 passed`。
- correction code pin=`a950ab914e6b029f86d8ef666a3f770fc42980a7`；
  runner SHA=`c92225d4...`；CPU evidence=`36 passed`。
- correction manifest rev1：
  - self=`2a907feec24f07d8601be0b48a21fc98cc6bdc08b6499031e2d284cf8b00a3f1`；
  - status=`pinned_blocked`；
  - allowed setting仅S0/rho2/chunk4096 restart0；
  - original raw SHA=`80e8e83d...`。
- correction Opus review为PASS_WITH_CAVEATS；两个治理P1已闭合：
  - RESULT_MANIFEST现`75/75`；
  - correction完成后会`--force`重生成全部22 compact与summary；
    原raw/log不改写。
- 下一步：versioned correction review artifact → authorized correction rev2 →
  Docker单一S0 correction。

### 2026-07-28T11:30:50-07:00 Phase7离线consolidator已实现

- 在`cross-store-substrate` worktree新增：
  - `benchmark/approx_kv/consolidate_phase7_results.py`；
  - `test/registered/unit/bench/test_consolidate_phase7_results.py`。
- consolidator纯离线读取`results/phase7` staging，验证authorized rev12、
  design hash、22个实际start、8个`ES-R0-MDE` skip、rho3 disabled、raw/log/
  central JSONL hash与工程不变量。
- 每个raw生成自哈希compact JSON；summary包含执行时长/预算、wave0、A8、
  chunk1024 sensitivity、W cross-policy、within-policy、R4与完整source hash
  lists。
- 真实staging验证结果：
  - 22 raw + 22 logs；
  - actual elapsed GPU-equivalent=`1.31014h`；
  - `engineering_status=VALID`；
  - `mechanism_status=NEGATIVE`；
  - `system_behaviour_status=INCONCLUSIVE/DESCRIPTIVE`。
- W paired-restart median：
  - rho1.5 all=`1.00208`、workflow=`1.02757`、miss delta=`-14`、
    peak ratio=`1.01123`；
  - rho2 all=`1.02503`、workflow=`1.04418`、miss delta=`+2`、
    peak ratio=`0.99806`。
- host验证：新增CPU tests `7 passed`；isort/ruff/black通过；真实staging
  生成22 compact并验证全部compact/summary自哈希。试运行输出已清理。
- 最终加固还验证execution envelope/source SHA、manifest file SHA、central
  run_id/phase/output绑定、runner path/hash、inactive counter逐项语义；summary
  同时记录compact canonical hash与序列化文件hash。R4 test已改为纯合成语义测试。
- target worktree未提交；下一步为独立review后生成正式compact/summary并纳入
  后续`RESULT_MANIFEST`。

### 2026-07-28T10:00:31-07:00 A8 restart-0 screening完成：R0 NEGATIVE

- 四个primary settings全部`status=valid`：
  - D0/E0/R0 outcome全部符合预期；
  - 每setting 2 formal × 8 targets均为真实请求；
  - R0全部为`approximate_gpu_recovery`；
  - same-context 8-token canary全部逐token一致；
  - persistent source pin、reset、orphan、inactive assertions全部通过。
- 结果：

| body | rho | paired request-path median | N8 full-setup | N8 incremental |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1.5 | `0.7723x` | `0.6086x` | `0.6838x` |
| 1024 | 2.0 | `0.7751x` | `0.6101x` | `0.6855x` |
| 2048 | 1.5 | `0.9334x` | `0.6398x` | `0.7607x` |
| 2048 | 2.0 | `0.9362x` | `0.6419x` | `0.7641x` |

- 所有setting均低于1.0，更不满足冻结MDE=`5%`；R0 ceiling track记
  `NEGATIVE`。
- 按`ES-R0-MDE`跳过primary restart1–2，共节省8 starts。
- 不发布speedup headline；结果严格为chunk4096、当前SM75/model/workload下的
  ceiling结论。
- W与chunk1024 sensitivity使用`ES-W-UNCONDITIONAL`，仍继续执行。

### 2026-07-28T09:45:04-07:00 Phase7 wave-0 required完成

- 全部实验在authorized rev12与固定Docker镜像内执行。
- `p6delta-s4-rho2-chunk4096`：
  - requested/observed capacity=`11392/11392`；
  - exact/r0/r1-k32/r2-like均reachable/valid；
  - R4-like在第5 representation出现预期partial registration，
    `diagnostic-unavailable`；
  - 两个formal repeat结果一致；
  - reset、orphan、inactive counter assertion全部通过。
- `p6delta-s0-rho2-chunk4096`：
  - requested/observed capacity=`11392/11392`；
  - 五个profile全部reachable/valid；
  - reset、orphan、inactive counter assertion全部通过。
- 两cell均无unexpected OOM、server death、stale handle、double free或
  accounting failure；S4 R4-like属于容量型diagnostic-unavailable，不是
  engineering failure。
- rho3保持`disabled_scoped_chunk1024`，未运行。
- runtime artifacts保留在`/results/phase7` staging，未修改implementation repo。
- 下一步：A8 restart-0四个primary settings。

### 2026-07-28T09:28:47-07:00 Phase7 manifest rev12已授权

- rev12：
  - status=`authorized`；
  - `phase7_execution_authorized=true`；
  - blockers=`[]`；
  - self=`2d66a1bcdb6dc92a72c59fefc581212fcd541accbc8ededa221495d30d039bef`；
  - design=`50003145f2e7f0e866613dbd420e73ba3983a6c182a360d6918098b1d1f7b987`；
  - supersedes rev11 self=`48c86bf0...`；
  - design hash保持不变。
- rev12 primary containing commit=`759476c901b1ac30eeeb96f83b9257586a103c4e`；
  authorized result envelope HEAD=`d42a5d1546680de458122f605f8a486b8faf5564`；
  RESULT_MANIFEST=`5/5`。
- Docker image/GPU/linked-worktree gitdir挂载前置均验证通过。
- 下一步为Phase7 wave-0 required：
  - `p6delta-s4-rho2-chunk4096`；
  - `p6delta-s0-rho2-chunk4096`。
- rho3 conditional保持disabled；所有输出写`/results/phase7` staging。

### 2026-07-28T09:23:55-07:00 V7 final Opus PASS WITH CAVEATS

- V7 plan commit=`c80ec165713772c533e17ef4c50f083c36dc9d72`，
  通过外部authority激活为`Current / Latest`；plan blob不再修改。
- rev11 review-of-record：
  - self=`48c86bf0f4df6f5c8baa41fc6871e3bcdfba59ea46f9022875f8453f2d5a5236`；
  - design=`50003145f2e7f0e866613dbd420e73ba3983a6c182a360d6918098b1d1f7b987`；
  - code pin=`81405f4278b034911bc613c4ee17c79d15ee8f35`；
  - status=`pinned_blocked`；
  - blocker仅`final_opus_review_pending`。
- final Opus delta verdict=`PASS WITH CAVEATS`：
  - open P0=`0`；
  - open P1=`0`；
  - entry=`READY_AFTER_REVIEW_ARTIFACT_AND_AUTH_REVISION`。
- review artifact：
  `benchmark/approx_kv/results/phase7/phase7-final-opus-review.json`；
  artifact SHA=`2c83f7ddec41d65a937a6b71cab851230c0fb06ff725f09cb36dd596e81e8649`；
  containing commit=`b0837505abc4efe7df914c3b504693956fe71df9`。
- 固定执行策略：
  - implementation worktree只读；
  - linked-worktree gitdir所属主仓库
    `/home/chris/Workspaces/kvcache-research/sglang-experiments`
    必须以相同绝对路径只读挂载；
  - 所有GPU wave输出保留在`/results/phase7` staging；
  - **全部GPU wave结束后一次性复制、hash、版本化**，波次间不修改repo envelope。
- 仍需生成rev12：
  - supersede rev11 self hash；
  - 保持design hash=`50003145...`；
  - 绑定final review artifact；
  - status=`authorized`，blockers=[]。
- rev12完成前仍未进入Phase7。

### 2026-07-28T08:52:08-07:00 V7闭合最终Opus review findings

- V6 final Opus review结论=`FAIL`，发现1 P0/3 P1：
  - repo内实时artifact导致第二次run无法满足clean envelope；
  - ceiling required CPU test证据陈旧；
  - P7-0b capacity runner未接入授权/manifest门；
  - plan自改`Current / Latest`与design hash保持互斥。
- 已创建V7并归档V6：
  - runtime raw/compact/log/central-log只写`/results/phase7` staging；
  - 三个runner均在server launch前验证authorized manifest与runner blob；
  - capacity runner Phase7模式只执行单一setting，并按formal repeat执行
    正序/逆序profiles；
  - capacity tolerance=`0.05`、warmup/formal/seed均由manifest冻结；
  - CPU test evidence改为版本化JSON并绑定Docker digest、exact command、
    exit code、summary、runner blob；
  - final Opus evidence绑定被审manifest self/design hash、code pin和runner hashes；
  - plan文件byte-frozen，`Current / Latest` activation由PROJECT/HANDOFF记录。
- 最终Docker证据：
  - targeted regression=`269 passed + 22 subtests`；
  - capacity required command=`24 passed`；
  - ceiling required command=`55 passed`；
  - scheduler required command=`12 passed`。
- final code pin=`81405f4278b034911bc613c4ee17c79d15ee8f35`。
- CPU evidence envelope HEAD=`a01642fdd6cdc869cb2c991c23003ff600665f37`。
- R2仍为`disabled_not_comparable`；矩阵仍为13 committed/30 starts，
  含rho3条件项14/31，3.8 expected GPUh。
- 下一步：生成V7 pinned manifest、执行最终增量Opus review、闭合feedback，
  再生成authorized revision。仍未进入Phase7。

### 2026-07-28T04:31:07-07:00 V6 Candidate、runner完成与R2最终disposition

- 用户选择方案B后，已按预定停止条件完成bounded R2 feasibility：
  - 历史CacheBlend package与core hooks已删除；
  - 恢复R2必须修改冻结的scheduler/runtime dispatch与store lifecycle；
  - 因此非侵入adapter不可行，按方案B决策树冻结
    `R2=disabled_not_comparable`。
- 已实现：
  - `run_p7_ceiling.py`；
  - `run_p7_scheduler.py`；
  - Phase7 common/statistics模块；
  - A8 persistent `pin_until_reset` source lease；
  - code-pin/execution-envelope验证；
  - per-arm peak、memory/churn/terminal taxonomy修复。
- 固定Docker镜像targeted suite：
  - `152 passed`；
  - `10 subtests passed`。
- 三轮targeted review已关闭全部P0/P1；最后一轮无开放P0/P1。
- final implementation code pin：
  `5d9a5793d73121f088890aa6c02cfebc31cd97be`。
- V5完整内容由commit `d314cc41...`与SHA-256归档；
  `IMPLEMENTATION_PLAN_V5_ARCHIVED.md`为定位入口。
- V6矩阵：
  - 13 committed GPU settings / 30 starts；
  - 1 conditional rho3 setting / 1 start；
  - 14 settings / 31 starts total；
  - expected `3.8 GPUh`，hard cap `6 GPUh`。
- 当前仍需：
  1. Opus 5 Max Thinking最终review与feedback closure；
  2. 条件性授权生效后才转`authorized`。
- final review evidence将版本化为
  `benchmark/approx_kv/results/phase7/phase7-final-opus-review.json`；
  rev7保持`pinned_blocked`，后续授权revision必须保持V6 design hash。
- rev6曾绑定前一code pin；因`build_result_manifest.py`仍含
  “runner未实现”的陈旧known-gap文字，在final review前被supersede。
- rev7：
  - status=`pinned_blocked`；
  - blocker仅`final_opus_review_pending`；
  - self hash=`3eebdbfb...`；
  - design hash=`ec5aaf59...`；
  - 14 settings / 31 starts；
  - Phase7 result manifest `1/1`通过。
- rev8仅绑定review-ready状态；rev9进一步将计划措辞改为revision-stable，
  settings/workloads均未改变。
- rev9为final Opus review-of-record：
  - status=`pinned_blocked`；
  - blocker仅`final_opus_review_pending`；
  - self hash=`5daa95a3...`；
  - design hash=`ae8d8e1e...`；
  - plan commit=`14a573eb...`；
  - envelope HEAD=`eb7daf63...`；
  - result manifest `1/1`通过。
- 仍未进入Phase7。

### 2026-07-28 最终Phase7计划增加Opus 5 Max Thinking强制复核门

- 用户已明确选择方案B，并授权在前置Gate完成后执行Phase7。
- 该授权是条件性授权：在GPU执行前的最后定稿阶段，必须让
  **Claude Opus 5 / Max Thinking**独立审阅最终Phase7 plan与执行包。
- 最终审阅输入至少包括：
  - final `IMPLEMENTATION_PLAN_LATEST.md`；
  - rev6 pinned manifest及revision/design chain；
  - 两个runner及CPU test结果；
  - R2 adapter feasibility、实现与review disposition；
  - final implementation SHA/tree/runner blob hashes；
  - Entry blocker、预算、early-stop和Docker执行合同。
- 主会话必须逐条处理Opus feedback：
  - accepted finding必须修改并重新验证；
  - rejected finding必须给出证据化理由；
  - 如修改plan design，必须归档旧版、创建新版、更新design hash并再次复核。
- 只有最终Opus verdict无开放P0/P1，且反馈闭合后，条件性用户授权才生效，
  manifest才能转为`authorized`并开始Phase7。

### 2026-07-27 Phase7正式Entry前剩余Gate与推荐路径

当前没有隐藏的Phase6 blocker。五个manifest blocker对应四类操作待办：

| Gate | 当前状态 | 关闭条件 |
| --- | --- | --- |
| Ceiling runner | 未实现 | 实现`run_p7_ceiling.py`，在Docker内通过CPU tests与targeted review |
| Scheduler runner | 未实现 | 实现`run_p7_scheduler.py`，在Docker内通过CPU tests与targeted review |
| R2 strategy | `pending` | 选择并冻结`adapter`或`disabled_not_comparable` |
| Implementation pin | 未生成 | 生成manifest rev6，pin final commit/tree/runner hashes并更新authority docs |
| User authorization | 未获得 | 用户在看到rev6 pinned状态后明确授权Phase7 |

推荐执行顺序：

1. 先完成两个runner及其Docker CPU tests、schema/golden tests和Sol/Opus
   targeted review；
2. 对R2做有界的0-GPU adapter feasibility检查：
   - 若只需harness/plugin plumbing且不改变core recovery语义，则实现、
     测试并选择`adapter`；
   - 若需要侵入性修改或威胁已冻结substrate，则选择
     `disabled_not_comparable`；
3. 生成manifest rev6，状态转为`pinned_blocked`，保持rev5 design hash；
4. 向用户呈现最终pin、runner review和R2 disposition；
5. 只有用户明确授权后才转为`authorized`。

可选路线：

- **最小准入**：R2=`disabled_not_comparable`。最快、风险最低，Phase7运行
  13 committed settings/最多30 starts，但没有新R2 GPU数据。
- **证据增强**：实现R2 adapter。若上述feasibility检查证明改动受控，这是
  当前推荐路线；Phase7可增加2个R2 conditional settings/2 starts，
  解决旧chunk1024 R2不可比较问题。
- **扩大research范围**：重新加入R1性能、R5 matched comparison、真实R4/
  KVCOMM或host/prefetch。当前不建议；这会改变V5设计，必须创建新版计划、
  新design hash并重做双模型review，而不是简单关闭Entry gate。

不属于正式Entry前Gate：

- P7-0b chunk4096 feasibility是授权后的Phase7 wave-0，不能提前偷跑；
- natural-pressure fallback reachability仍是caveat，不阻塞Entry；
- S4/rho3 chunk4096只在需要rho3 claim时激活；
- Phase4/5完整矩阵、CL1/CL2/CL3/P6-H和完整P6-4默认不重做。

### 2026-07-27 Recovery数据代际：metric schema不等于Phase7新数据

必须区分：

1. metric/ledger定义是否已纠正；
2. raw data是否在Phase7匹配配置下重新采集。

目前没有任何recovery family拥有Phase7数值结果：

| 路径 | 现有最新证据 | Phase7匹配数据 |
| --- | --- | --- |
| R0 | post-fix CL1为body1024/2048、chunk1024；CL2为body768/1024、chunk1024/4096 | 无A8/W、真实N=1/2/4/8或Phase7 scheduler结果 |
| R1 | post-fix CL1全k轴仍为chunk1024；CL2只覆盖与R0同物理family的k0 | 无Phase7性能数据；`winner=NONE` |
| R2 | corrected rerun已使用纠正ledger，但raw仍为body1024/2048、chunk1024旧harness | 无chunk4096 cross-store adapter数据 |
| R4 | Phase6只有`R4-like` synthetic footprint/capacity profile，非真实KVCOMM | 无真实R4机制数据；Phase7仅计划新proxy diagnostic |
| R5 | corrected rerun已使用纠正ledger，但raw仍为body1024/2048、chunk1024 | 无Phase7数据，且V5默认排除 |

因此R2/R5并非“仍使用错误指标”，而是“指标已纠正，但底层实验配置与
Phase7不匹配”。缺少的数据包括chunk4096、A8真实连续targets、当前
cross-store substrate、统一D0/E0配对、当前taxonomy/accounting和Phase7
restart合同。

CL2直接说明配置不可平移：R0/R1-k0 body1024 request-path从chunk1024约
`1.547x`降到chunk4096约`1.025x`。因此任何chunk1024历史收益都不能自动
作为Phase7结果。

P6-4中的R0/R1/R2/R4-like是capacity footprint profile，不是相应机制的
新性能/正确性实验；Phase7 wave-0的P6-4Delta-4096尚未执行。

### 2026-07-27 R0/R2定位、Phase7矩阵与指标合同

- R0是`Raw+RoPE/raw_copy_ceiling`：
  - 复制source body的K/V；
  - 对K执行完整RoPE位置校正；
  - 不修复header/context变化造成的hidden-state差异；
  - 因此只代表最低恢复开销下的speed ceiling，不是practical candidate。
- R2是项目中的CacheBlend-style precomputed oracle family代表：
  - 复用大部分预计算KV；
  - 通过显式repair比例生成target-specific adapter/recovery object；
  - setup/adapter成本必须计入完整ledger；
  - 历史点约为1% repair ratio，不等同于完整faithful CacheBlend实现。

Phase7不是“只跑一个R0 run”，而是以R0为唯一primary recovery family：

- R0 A8 ceiling：4个primary settings，最多12 starts；
- R0 chunk1024 sensitivity：1个setting、2 starts；
- R0 W × S0/S4：4个settings、12 starts；
- P6-4Delta wave-0：2个required settings，内部含`r0_like`等footprint profile；
- R4-like：2个settings、2 starts；
- R2：2个conditional settings、2 starts，仅在adapter实现后运行；
- rho3 P6-4Delta：1个conditional setting、1 start。

总计13 committed settings/30 starts，另有3 conditional settings/3 starts。
R1只在wave-0以capacity footprint profile出现，不做Phase7性能研究；R3 defer；
R5因与R2冗余默认排除；practical/HiCache/host/prefetch/async轨道跳过。

Phase7尚无数值结果，当前冻结的是metrics contract，覆盖：

- latency/cost ledger与N=1/2/4/8真实amortization/break-even；
- TTFT、request-path、full lifecycle、workflow wall-clock和per-role视图；
- first-token/8-token逐位置一致率及same-context canary；
- exact/approx outcome、fallback与exclusive terminal reason；
- rho、resident/temporary peak、victim/evict/demote/transfer/churn/orphan；
- paired per-restart统计、MDE=5%、p95口径；
- validity/mechanism状态和完整raw/log/hash/commit provenance。

### 2026-07-27 Phase7 `preregistered_blocked`与五个准入门解释

`preregistered_blocked`是有意设置的安全状态，不是review失败：

- `preregistered`表示V5实验设计已在GPU执行前冻结并由manifest绑定，包括
  workload、arms、settings、repeat、MDE、early-stop、预算、环境、
  outcome taxonomy和artifact provenance；
- `blocked`表示manifest仍禁止Phase7 GPU执行，
  `phase7_execution_authorized=false`，且最终可执行实现尚未完整pin。

五个显式execution blocker的性质不同：

| Blocker | 当前事实 | 是否需要实现 |
| --- | --- | --- |
| `missing_runner:ceiling` | `run_p7_ceiling.py`文件不存在，CPU测试与review均pending | 是；实现A8 D0/E0/R0、真实N=1/2/4/8和结果记账 |
| `missing_runner:scheduler` | `run_p7_scheduler.py`文件不存在，CPU测试与review均pending | 是；实现独立W、S0/S4和各统计视图 |
| `phase7_pinned_implementation_sha_pending` | final runner code尚未完成，因此不能冻结最终commit/tree/runner hash | 不是新机制；在runner与R2 disposition完成后生成manifest rev6并pin |
| `r2_strategy_pending` | conditional R2尚未决定走真实cross-store adapter还是`disabled_not_comparable` | 二选一；只有选择adapter才需要实现/测试额外代码 |
| `explicit_user_authorization_missing` | 用户尚未明确授权进入Phase7 | 不需要代码；是最后的人为治理门 |

状态机为：

1. `preregistered_blocked`：设计已冻结，但未pin且仍有blocker；
2. `pinned_blocked`：runner/R2 disposition已完成，final implementation已pin，
   但仍至少缺用户授权；
3. `authorized`：已pin、blocker清零且用户明确授权，才允许在Docker内启动
   Phase7 GPU执行。

当前问题与解释不构成Phase7授权。Phase7之前可继续完成前四项工程/决策工作。
P7-0b chunk4096 feasibility属于授权后的wave-0实验门，不在上述五个
pre-execution blocker中；它通过后才允许继续后续Phase7结果解释。

### 2026-07-27 V5与Phase7 primary manifest完成review并封版

- `IMPLEMENTATION_PLAN_V4_ARCHIVED.md`为V4只读归档。
- `IMPLEMENTATION_PLAN_LATEST.md`为V5 Current / Latest。
- V5 full review发现的8个P0已全部关闭；targeted delta review新增MDE定义P0，
  已用CL2 body768/chunk4096 noise model冻结为`MDE=5%`。
- 最终minimal delta：
  - Sol：`PASS WITH CAVEATS`，唯一retained rho3 scope已修；
  - Opus：`PASS WITH CAVEATS`，确认计划足以生成primary manifest；
  - 无开放P0，budget算术无误。
- V5实际范围：
  - practical=`NONE`；
  - R0 ceiling、条件R2、R4-like synthetic footprint proxy；
  - primary chunk=`4096`，chunk=`1024`仅作sensitivity；
  - A8 recovery与W workflow严格分离；
  - Phase7主矩阵为R0 W × S0/S4；
  - HiCache/host/prefetch/async默认全部跳过；
  - committed 13 GPU settings/30 starts；含条件项16/33；hard cap36/6 GPUh；
  - N=1/2/4/8改为一次setup后的真实连续8 targets，不再外推；
  - fallback仅有fault-injected canary证据，natural pressure claim必须重新取证。
- Phase7 primary manifest rev4已获Sol/Opus最终PASS，无新P0/P1。
- rev5仅绑定最终V5 Current/Latest plan commit，settings/workloads未变；
  自检与Phase7 result-manifest均通过。
- manifest仍为`preregistered_blocked`，禁止GPU执行。
- 当前剩余执行blocker：
  - `run_p7_ceiling.py`未实现/测试/review；
  - `run_p7_scheduler.py`未实现/测试/review；
  - Phase7 pinned implementation SHA未生成；
  - R2 strategy未决；
  - 用户未授权Phase7。
- 仍未进入Phase7。

### 2026-07-27 Phase7 primary manifest已预注册，状态明确blocked

- `build_phase7_manifest.py`已实现并通过`--check`。
- `phase7-primary-manifest.json`绑定V5 plan commit
  `8fac933371ab26891dcd4c981a3d1156bf2e9814`。
- manifest冻结16个GPU setting、30 committed starts、3 conditional starts、
  33 total starts、MDE=5%、完整A8/W/filler与early-stop/taxonomy。
- status=`preregistered_blocked`，`phase7_execution_authorized=false`。
- 明确blockers：
  - `run_p7_ceiling.py`未实现；
  - `run_p7_scheduler.py`未实现；
  - Phase7 pinned implementation SHA未生成；
  - R2 strategy未完成；
  - 用户未授权Phase7。
- manifest review全部完成；仍未进入Phase7。

- Manifest review结果：
  - Sol：`PASS WITH CAVEATS`；
  - Opus：`PASS WITH CAVEATS`，接受rev2为pre-registration of record；
  - 无新P0。
- 两个赋值级P1已在rev3关闭：
  - W/sensitivity supplements使用无条件gate，不受A8 MDE早停；
  - GPUh按start类型重估，并保留>=15% hard-cap headroom。

### 2026-07-27 V5双模型full review与交叉consolidation完成，revision待delta review

- Sol与Opus独立full review均判`FAIL`；两份报告已全文互换。
- cross-consolidation接受8个P0，均已在V5 revision中定点修订：
  1. A8真实N合同与旧single-target/公式冲突；
  2. A8 microbenchmark与W workflow workload必须分离；
  3. R4 S1–S3无cross-store实现，删除；
  4. cache outcome taxonomy必须区分ordinary exact miss与approx fallback；
  5. chunk4096触发P6-4Δ兼容性复核；
  6. Phase4 R2/R5收益必须标`chunk-confounded`；
  7. P7 runner/manifest/R2工程前置缺失；
  8. P7-1必须恢复D0/E0/R0三臂。
- 预算修订为：
  - committed：13 GPU settings / 30 starts；
  - 条件项：R2 2 starts、S4/rho3 P6-4Δ 1 start；
  - 全部条件项：16 GPU settings / 33 starts；
  - hard cap仍为36 starts / 6 GPUh。
- （历史状态）当时V5仍为Draft；现已完成targeted delta与manifest review，
  并标记Current / Latest。

### 2026-07-27 当前无Phase6 blocker；剩余项均为Phase7 Entry

- Phase6 technical Exit已是`PASS WITH CAVEATS`，无开放P0/P1。
- 永久scope caveat：reservation-failure-associated fallback仅在test-only
  fault-injected canary强度验证，`natural_pressure_reachability=false`。
- 该caveat不再阻塞Phase6，但Phase7若依赖自然压力fallback，必须重新取证。
- V4已归档，result-bound V5 draft已创建。
- 当前真正阻塞Phase7 Entry的事项只有：
  1. 两个Phase7 runner实现、CPU测试与targeted review；
  2. R2 strategy决议；
  3. 原子pin Phase7 implementation SHA；
  4. 用户明确授权Phase7。
- `winner=NONE`含义已冻结为被测实现、模型、prompt族、GPU、chunk配置和
  exact-output promotion规则下的结果；未证明context差异是唯一原因，
  未排除header-dependent实现缺陷。
- V5将primary chunk从1024迁移到4096，已触发`P6-4Δ-4096`兼容性复核：
  S4/rho2、S0/rho2为required，S4/rho3为条件项。该复核属于Phase7 P7-0b
  工程前置，不得与旧P6-4全矩阵重跑混淆。

### 2026-07-27 Phase6 technical Exit最终为PASS WITH CAVEATS

- 用户明确用V4 §15.1冻结的test-only路线取代此前方案C豁免。
- 最终实现：
  - 双test gate，默认关闭；
  - one-shot，仅`requester=approximate`；
  - 在`AllocationFailurePoint.AFTER_RESERVE`注入；
  - 不修改共享上游`alloc_token_slots`。
- 最终GPU artifact：`p6-f-v3-fallback-canary.json`，`status=valid`：

| 验收项 | 结果 |
| --- | --- |
| reservation failure | `1` |
| `cross_store_reservation_failed` token | `1024` |
| `device_allocation_failed` token | `0`（exclusive terminal reason） |
| `reuse/dense_fallback` request | `1` |
| 请求完成 | 是，8 token |
| dense/fallback exact prefix | 均严格为`64` token |
| fallback输出与dense一致 | 是 |
| 注入后的下一请求正常recovery | 成功，`cached_tokens=1088` |
| 独立无注入server | recovery成功、failures=0、输出一致 |
| pre-flush accounting | reserved/provisional/leases/orphans均为0 |
| reset/store gauge | 全部归零 |

- artifact明确写入`fault_injected=true`、
  `natural_pressure_reachability=false`，不把注入证据写成自然可达。
- JSON与两个最终server log均已版本化；primary P6-H/P6-4 logs也已版本化；
  `RESULT_MANIFEST.json`现`48/48`通过自动file→commit/hash验证。
- P6-4旧artifact未改写；新增
  `p6-4-outcome-correction.json`：
  - 12个`exact_only`旧`dense_fallback`全部更正为`exact_cache_miss`；
  - corrected approximate fallback数为`0`；
  - R4 replay共30次均为`approximate_gpu_recovery`，registration容量失败
    不再作为replay fallback证据。
- P6-F两位targeted reviewer最终均判**PASS**，无新P0/P1。
- Formal Exit两位reviewer均关闭P0-1/P0-3，最终判：

```text
technical_exit = PASS WITH CAVEATS
```

- caveat：fallback仅在**fault-injected canary强度**验证；
  `natural_pressure_reachability=false`，不得声称自然压力可达。
- 主会话最终disposition已写入`PHASE6_EXIT_DISPOSITION.json`。
- Phase6通过不等于Phase7获授权；仍需result-bound V5、Phase7 primary
  manifest与用户明确授权。

### 2026-07-27 V4先更新、再执行blocker的顺序已冻结

- 不升级版本；`IMPLEMENTATION_PLAN_LATEST.md`继续为V4 Current / Latest。
- 在任何blocker代码或实验之前，先原地冻结§15.1的验收合同，避免再次出现
  “看到结果后改变证据标准”。
- **以下是P6-F执行前的历史状态，现已被上节最终disposition取代：**

```text
technical_exit         = FAIL
governance_disposition = 已对一项未验证条件接受豁免
```

- 若用户决定把technical Exit真正转为PASS，执行路线固定为：
  1. benchmark/test-only fault injection，默认关闭；
  2. 不改共享上游`alloc_token_slots`；
  3. 必须经过`finalize_copy_reuse`、scheduler与真实dense执行；
  4. 必须断言reservation失败标签、`reuse/dense_fallback`、输出完成、
     provisional/device/store/lease/orphan全部归零；
  5. artifact标注`fault_injected=true`、
     `natural_pressure_reachability=false`；
  6. 更新并验证`RESULT_MANIFEST.json`；
  7. 只做该blocker/provenance的targeted dual-review delta verification。
- technical Exit通过前不创建V5、不进入Phase7。

### 2026-07-26 Phase4/5 Closeout与Phase6实现进展

- 已按用户要求停止全部既有Docker容器；后续只允许任务自身的短期容器。
- Phase6实现worktree为
  `/home/chris/Workspaces/kvcache-research/worktrees/cross-store-substrate`，
  分支为`research/cross-store-substrate`，基于
  `research/scheduler-policies@c185428fd`。
- 已实现P6-0合同：
  - fixed40对象、确定性token SHA256、dead/live、长度、segment和chunk语义；
  - provisional chunk配置进入manifest hash；
  - 统一artifact schema、contract hash与漂移校验。
- 已实现P6-1/P6-2/P6-3 substrate：
  - exact/approx/host统一对象、byte-authoritative budget和统一event ordinal；
  - S0与S4跨store策略；
  - reserve-before-victim、allocation、commit和失败账本；
  - exact/approx双向pressure；
  - approximate device→host真实demotion及cross-store-aware demand H2D load；
  - dependency wire metadata、注册期dependency pin、dependent-first atomic closure；
  - orphan拒绝、stale/double-free防护、store lifecycle/reset gauge；
  - cross-store victim/demotion/failure/wasted/peak telemetry。
- 第一轮Claude Opus 5 Max代码review提出CR-01至CR-22；阻断项已逐项修复或显式封锁：
  - 请求执行中不再调用全量`tree_cache.reset()`；
  - 不可逆victim后的失败保留真实资源状态和byte ledger；
  - HiRadix exact cross-store路径在专用语义完成前明确unsupported，避免host ref泄漏；
  - fixed40、schema、S4 class order、wire alias和event clock已统一。
- 已实现GPU runner：
  - `run_p6_h_host_roundtrip.py`；
  - `run_p6_4_capacity_pilot.py`；
  - `run_cl1_qualification.py`；
  - `run_cl2_chunk_gate.py`。
- Claude Opus 5 Max完成三轮代码review：
  - 第一轮CR-01至CR-22；
  - 第二轮CR2-01至CR2-16；
  - 最终delta CR3-01至CR3-03；
  - 上述finding均已修复并增加对应回归。
- 800对象、400 victim的CPU压力路径由`3.087s`优化到`0.188s`。
- 当前相关CPU回归为`169 passed, 1 skipped`；isort/Black/ruff和
  `git diff --check`全部通过。
- 独立GPT-5.6 Sol Max最终review提出8项P1；全部修复后完成两次delta核对，
  最终结论为“无剩余P0/P1”。
- Phase6核心提交：
  `391bb89901cebebd50ffc9f27a648b09a99abf7e`。
- P6-0 artifact提交及远程branch head：
  `c487e36af5f7ce4da556da1b88c85df750a0b14d`。
- 远程分支：
  `ccdd2023/sglang:research/cross-store-substrate`，本地与远程SHA已核对一致。
- P6-0合同：
  - contract SHA256
    `a498daa36449993ff166dd70870005be22a1da0a7d09e97e8f779d72cbf3fb30`；
  - fixed40 workload SHA256
    `30c9ae8de429a6389e58bbdcdf096101cf6296ff14d4e6fcf5c2b87c6b1f0749`；
  - source tree
    `2ec26d8503d1a2f7515f379ed3a4f60c2dba42c2`；
  - provisional chunk为`1024`。
- CL0 authority manifest已重新生成，R2/R5 final heads更新为
  `ce55860a9`与`71f15d5d1`。
- GPU仍不可用：
  - loaded kernel module：`580.159.03`；
  - installed userspace/NVML：`580.173.02`；
  - NVIDIA模块被图形会话占用，安全恢复通常需要系统重启。
  - `/var/run/reboot-required`明确存在，当前还有活动SSH/tmux会话，因此未擅自重启。
- 因此CL1、CL2、P6-H和P6-4尚未运行；这不是实验负结果，而是环境阻塞。
- 当前严格停在Phase7 Entry之前，不执行Phase7 integrated evaluation。

### 2026-07-26 Phase7影响与剩余重测边界

- Phase6实现没有产生新的GPU结果，因此没有改变任何Phase7性能、winner或
  scheduler收益结论。
- 实现改变的是Phase7的证据合同和进入条件：
  - exact→approx与approx→exact必须按requester→victim方向分别证明；
  - P6-4必须含exact-only baseline、服务端cache outcome、physical rho、
    fallback关联和clean-tree/model-revision provenance；
  - CL1 promotion使用包含seed head的request-path，并报告N=1/2/4/8与break-even；
  - P6-H只证明generic `AllocatorCPUResidencyBackend` roundtrip，不证明HiCache；
  - HiRadix/UnifiedRadix cross-store在专用语义完成前启动期拒绝。
- Phase7前必须执行：
  1. 安全重启并验证GPU；
  2. CL1 candidate qualification；
  3. CL2 chunk gate；
  4. P6-H generic host canary；
  5. P6-4 fixed40 capacity pilot；
  6. GPU结果Sol/Opus双模型review与CL4 disposition。
- 不需要再次完整重跑Phase4或Phase5，也不需要重复已修正的R2/R5 rho2矩阵。
- 条件性重跑：
  - CL2最终chunk不是provisional `1024`时，重新冻结合同并重跑受影响P6-4 cell；
  - 保留R2/R5机制排名时才做matched repair ratio/pressure；
  - 发布rho鲁棒性时才补R2/R5 rho1.1/3；
  - 需要R2直接zero-fallback证据时才补显式counter单cell；
  - Phase7 HiCache track仍需独立H4/RH4 feasibility，不能由P6-H替代。

### NVIDIA驱动恢复说明

- 不需要安装“更新的最新驱动”，也不需要重新实现或重新移植Phase6 patch。
- 当前磁盘上的NVIDIA kernel module与userspace均已是`580.173.02`；
  `modinfo -F version nvidia`返回`580.173.02`。
- 当前运行内核仍持有启动时加载的旧module `580.159.03`，因此NVML
  `580.173.02`无法连接，`nvidia-smi`报告driver/library mismatch。
- 正确恢复方式是保存活动会话后安全重启本地主机，使内核加载已安装的
  `580.173.02`。不建议在图形会话和活动SSH/tmux存在时强制卸载NVIDIA模块。
- 重启后只需验证module/NVML/CUDA smoke；仅当真实GPU测试暴露API兼容问题时
  才需要新的代码修复，不预先重做patch。

### NVIDIA重启后验证结果

- host已于`2026-07-26 17:55`重启，`reboot-required`标记已清除。
- loaded module、installed module与NVML现均为`580.173.02`。
- `nvidia-smi`正常识别`NVIDIA GeForce RTX 2080 SUPER`、8192 MiB。
- 正式SM75镜像
  `ghcr.io/ccdd2023/sglang@sha256:0be6e16e...`内验证：
  - PyTorch `2.9.1+cu129`；
  - CUDA build `12.9`；
  - `torch.cuda.is_available() == True`；
  - compute capability `(7, 5)`；
  - CUDA tensor smoke结果`28.0`。
- 当前无运行中的Docker容器；GPU实验门禁已解除。
- Phase6 patch无需重做；下一步按既定顺序执行CL1。

### Phase7计划更新时机

- 当前建议不立即重写整个Phase7，也不等待所有门禁结束后才处理任何计划问题，
  而是分两步：
  1. 现在记录已知、与实验结果无关的合同修正；
  2. CL1/CL2/P6-H/P6-4完成并双模型review后，再创建新的latest plan版本，
     冻结真实candidate、chunk、矩阵和停止分支。
- 当前已知合同修正：
  - CL1 promotion以完整request-path和N=1/2/4/8摊销为依据；
  - CL2必须支持R0、R1-k0、selected R1-k和NONE；
  - Phase7记录exact→approx与approx→exact requester/victim方向；
  - Phase7 provenance必须含model revision、clean source tree SHA与独立result commit；
  - P6-H只证明generic allocator-CPU host roundtrip，不能解锁HiCache track；
  - P7-3若practical存在，仍需专用HiRadix/Unified cross-store adapter gate；
    practical=NONE时不实现该adapter并直接跳过practical host/prefetch track。
- 必须等待门禁结果才能决定：
  - practical candidate或NONE；
  - 最终chunk为1024或4096；
  - P6-4哪些footprint可达；
  - P7-1/P7-2实际矩阵裁剪；
  - 是否值得实现P7-3 host adapter及执行P7-4。
- 因此当前V4阶段结构保持有效；门禁完成后再归档V4并用Sol/Opus review新的
  result-bound latest版本。
- 根目录`TODO_LOCAL.txt`保存全部当前、条件性和待授权任务，作为聊天/session
  中断后的固定恢复入口。

### 2026-07-26 CL1 screening结果与Phase7前缺口审计

Docker执行合同（本轮全部实验均遵守）：

- 镜像`ghcr.io/ccdd2023/sglang@sha256:0be6e16e...`；
- `--runtime=nvidia --gpus all --ipc=host --shm-size=8g --user 1000:1000`；
- 实现worktree与`sglang-experiments`按host路径只读挂载，保证git worktree
  gitdir可解析；
- `results`目录读写挂载为`/results`；HF cache只读挂载并使用离线模式；
- `PYTHONPATH=<worktree>:<worktree>/python:/opt/sm75-site`。

CL1 screening（6 candidate × body 1024/2048 × restart 1 × formal 4 = 48个
paired repeat）已完成：

- artifact `/results/phase6-gpu/cl1-screening.json`，
  `raw_sha256=a122e1981af1d6ee92943b8f937dd91ac4cbd18998032248d5f65b84ba081cf6`；
- provisional ranking `r0 > r1_k4 > r1_k0 > r1_k8 > r1_k32 > r1_k16`。

| candidate | body | median request-path | paired target p95 ratio | 摊销N=1 | 摊销N=8 | break-even |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r0 | 1024 | `1.554x` | `0.577` | `0.487` | `1.221x` | `3.95` |
| r0 | 2048 | `1.984x` | `0.476` | `0.423` | `1.357x` | `3.75` |
| r1_k0 | 1024 | `1.555x` | `0.575` | `0.488` | `1.221x` | `3.94` |
| r1_k0 | 2048 | `1.969x` | `0.479` | `0.424` | `1.353x` | `3.76` |
| r1_k4 | 2048 | `1.974x` | `0.480` | `0.422` | `1.352x` | `3.78` |
| r1_k8 | 2048 | `1.966x` | `0.480` | `0.421` | `1.348x` | `3.80` |
| r1_k16 | 2048 | `1.952x` | `0.483` | `0.420` | `1.340x` | `3.84` |
| r1_k32 | 2048 | `1.962x` | `0.479` | `0.420` | `1.345x` | `3.82` |

固定结论：

- body2048上6个candidate的差异小于`1.6%`，不足以支撑机制排序；
- body1024上`r0`/`r1_k0`（`1.554x`/`1.555x`）明显优于`k>=4`
  （`1.451x–1.467x`），说明EPIC leading-k重算在request-path口径下是净成本；
- paired target p95全部改善（`0.476–0.632`），远优于`<=1.05`门槛；
- N=1摊销为`0.420–0.488`，break-even需`3.75–4.54`次复用，N=8才达
  `1.156x–1.357x`；single-use为负的结论与corrected R2/R5一致。

本轮新增finding：

- FINDING-CL1-A：48个paired repeat中`quality_8_token_match`失败17次、
  `first_token_match`失败6次，`all_guardrails_passed`对全部candidate为
  `false`。cache path、reset invariant与pool恢复48/48通过，因此这是恢复质量
  结果而非harness故障。按冻结的promotion规则，practical family方向为`NONE`。
- 零GPU派生`/results/phase6-gpu/cl1-screening-consistency.json`补齐§5.9要求
  的逐token一致率：first-token一致率`0.875`、8-token完全一致率`0.646`、
  逐token一致率中位数`1.000`、均值`0.799`；body1024比body2048更易发散。
- FINDING-CL1-B（P0证据缺陷）：`approx.fallback_tokens`全部为`null`。
  `sglang:approx_kv_dense_fallback_total`是带`reason`标签的Counter，
  未发生fallback时不输出任何series；冻结runner用`(x or 0) == 0`把
  “counter缺失”静默判为“显式0 fallback”，违反“counter缺失只能记为
  `indirectly_verified`”的既定规则。派生artifact已按`indirectly_verified`
  记录，该缺陷不改变本次promotion方向。
- FINDING-CL1-C：计划§5.9把8-token canary定义为“记录逐token一致率、不扩展
  semantic correctness claim”，冻结runner却把8-token完全一致作为promotion硬门。
  因“CL1执行前冻结、看到结果后不得改规则”，本轮严格按冻结实现判定，差异交由
  CL4与新版plan处置。
- FINDING-GAP-1：Closeout CL3（Phase5零GPU重算）从未执行，
  `scheduler-policies` worktree无代码与artifact。计划§8.1要求CL0–CL4全部完成
  才能进入Phase7，因此这是真实阻塞项。`TODO_LOCAL.txt`原先误标为已完成，且
  执行顺序漏掉CL3，均已更正。“variable-size offline optimum”属于计划§12的
  Phase7交付物，不是Closeout阻塞项，原条目把两者混为一谈。
- FINDING-GAP-2：`IMPLEMENTATION_PLAN_LATEST.md`§14“当前状态”仍写着
  “Phase6分支未创建、CL0未完成、未启动GPU实验”，已更正为当前事实。
- FINDING-GAP-3：计划§8.1把“Phase7 primary manifest已预注册”列为Phase7
  Entry条件，但该manifest不存在，也没有生成它的runner或模板；必须在Phase7
  授权前补齐。

### 2026-07-26 CL1定稿：practical family = NONE

- 3-restart确认运行（`r0`、`r1_k0`，body 1024/2048，formal=4，48个paired
  repeat）已完成，artifact `/results/phase6-gpu/cl1-confirm.json`，
  `raw_sha256=7736f0e7f641ce7d9d628a4ea7bf1b6697ede4019bf6e6214b37efb57fff8945`。
- `promotion` 结果为 `status=complete`、`passing=[]`、`winner=NONE`。
- **固定结论：practical family = NONE。**
- 该NONE完全由correctness guardrail决定，四条性能条件全部满足：

| 条件 | 要求 | r0 | r1_k0 | 结果 |
| --- | --- | --- | --- | --- |
| body2048 request-path >1.0x | 至少2/3 restart | `1.972/1.965/1.978` | `1.969/1.972/1.974` | 3/3通过 |
| paired target p95 ratio | `<=1.05` | `0.480` | `0.479` | 通过 |
| N=8摊销 | `>1.0x` | `1.353x` | `1.351x` | 通过 |
| all guardrails | 必须通过 | `false` | `false` | **不通过** |

- guardrail失败明细：48个paired repeat中`quality_8_token_match`失败12次、
  `first_token_match`失败4次；cache path、fallback、reset invariant与pool
  恢复48/48通过。
- 零GPU派生`/results/phase6-gpu/cl1-confirm-consistency.json`：
  first-token一致率`0.917`、8-token完全一致率`0.750`、逐token一致率中位数
  `1.000`、均值`0.859`；fallback证据等级`indirectly_verified`。
- 三次restart的body2048 speedup最大相对偏差`<0.7%`，性能测量稳定，NONE不是
  噪声导致，也不是环境阻塞。
- 直接后果（计划§8.4/§8.5/§8.6的`practical=NONE`分支）：
  - 跳过practical scheduler revalidation，不生成PR-S0/PR-S4；
  - 跳过practical HiCache与RH4矩阵，不实现P7-3专用cross-store adapter；
  - 跳过prefetch性能track；
  - 保留R0 ceiling、R2 oracle与R4 diagnostic；
  - exact-only prefetch若执行只能标为Phase5回归canary。

### 2026-07-26 CL2 chunk gate与chunk配置伪影

- CL2以`--selected-candidate NONE`执行（冻结逻辑回退到`r1_k0`为gate臂），
  artifact `/results/phase6-gpu/cl2-chunk-gate.json`，
  `raw_sha256=ab384e6594d1cf293bb5ad9b8a9dbe5fa68dcd4babfcbe8cbe29b0b1250abfc2`。
- `status=inconclusive`、`selected_chunked_prefill_size=null`；原因与CL1同源，
  gate要求`all_guardrails_passed`而correctness guardrail不通过。
- 按计划§6 CL2显式waive分支处置：P6-4继续使用预注册worst-case provisional
  chunk `1024`，所有结论限定在该预注册配置。
- FINDING-CL2-A（重大）：measured recovery speedup几乎完全是
  `chunked_prefill_size`配置伪影。

| chunk | body | dense target TTFT | approx target TTFT | target-only | request-path |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 768 | `129.8ms` | `126.4ms` | `1.027x` | `1.018x` |
| 1024 | 1024 | `297.8ms` | `171.8ms` | `1.733x` | `1.549x` |
| 4096 | 768 | `129.3ms` | `127.6ms` | `1.013x` | `1.004x` |
| 4096 | 1024 | `178.4ms` | `172.8ms` | `1.032x` | `1.025x` |

- 机制：`launch_server`把`--max-prefill-tokens`同步设为`chunked_prefill_size`。
  body1024的target prompt为`64+1024+1=1089`token，chunk`1024`下dense必须分两个
  prefill chunk（`297.8ms`），chunk`4096`下是单chunk（`178.4ms`）；approximate臂
  只需prefill最后1个token，两种配置几乎不变。
- 措辞限定（按review修正）：**在CL2的body1024这一个点上**，表观speedup主要
  来自coupled `chunked_prefill_size`/`max_prefill_tokens`配置造成的dense
  chunk-boundary penalty。注意该实验**同时改变了两个配置项**，且
  **body2048未被测量**，因此**不能泛化到全部CL1结果**。

### 2026-07-26 P6-H暴露压力下近似KV数据损坏（P0，阻塞Phase6 Exit）

- P6-H运行中发现并修复两个缺陷，修复提交
  `5e47904ecba6b8d7b5d03693277360a1cecfa679`：
  1. `resolve_reuse_spans`在exact prefix短于首个segment `target_start`时把
     整段dense prefill误记为`reuse/exact`且不计任何fallback token，
     违反“counter缺失不得写成显式0”的证据规则；现改为记
     `prefix_gap` dense fallback，并保留真正“已被exact覆盖”的`exact`语义。
  2. P6-H canary在tight capacity下让paired dense请求驱逐了recovery namespace
     的header，使reuse永远无法挂载、demand H2D不可能触发；现于reuse前重新
     seed header并断言reuse确实挂载registered body。
- 新增2个回归测试；容器内相关回归`164 passed, 5 skipped`；
  isort/black/ruff(F401,F821,UP037)/`git diff --check`全部通过。
- 该修复不影响CL1/CL2已测路径：CL1的`cache_path_matched`为48/48通过，
  exact_length始终等于header长度，从未进入被修改的分支。
- **FINDING-P6H-A（P0）**：修复后P6-H全部机械证据通过——host export
  `1024`token、`cross_store_demoted_bytes_total`=`117440512`、demand H2D
  `1024`token、host bytes归零、leases`2`、0 reservation failure、0 orphan、
  reset通过——但recovered输出与matched dense不一致。P6-H的source与target
  上下文完全相同，正确的copy必须逐token复现dense输出。
- 5次隔离实验（artifact
  `/results/phase6-gpu/p6-h-pressure-corruption-isolation.json`）：

| max_total_tokens | 竞争性registration | 注册residency | demotion | 与dense一致 |
| ---: | :--- | :--- | :--- | :--- |
| 8000 | 有 | device | 无 | 一致 |
| 8000 | 无 | host | 无 | 一致 |
| 3400 | 无 | host | 无 | 一致 |
| 3400 | 有 | device | 有 | **不一致** |
| 3400 | 有 | host | 无 | **不一致** |

- 触发条件是“reuse执行时存在真实device内存压力（竞争性近似registration +
  紧容量）”；与residency tier无关、与是否demotion无关，也不是紧容量本身。
- **FINDING-P6H-B**：零近似exact-cache对照
  `/results/phase6-gpu/control-exact-cache-guardrail.json`
  （body1024/2048各8轮，第二臂由普通exact radix命中服务）first-token、
  8-token与逐token一致率均为`1.000`，16/16全部一致，排除prefill路径数值
  不确定性。
- **FINDING-P6H-C（重大，影响CL1结论）**：CL1所有臂都在
  `rho_logical_demand=2.0`压力下执行，因此CL1的`quality_8_token_match`与
  `first_token_match`失败与本缺陷完全混淆，不能归因于跨上下文近似误差。
  `practical family = NONE`仍是冻结规则下程序正确的结论，但**因果归因无效**，
  必须在缺陷修复后重跑CL1才能重新判定。
- 未做投机性修补：该P0涉及pinned近似source的device slot与同一请求
  `allocate_recovery_slots`之间的保护契约，需要专门设计与双模型review。
- 结论：**Phase6 Exit当前不可通过；Phase7不得在该底座上启动。**

### 2026-07-27 P0根因更正、修复与验证

**先前推断已更正。** 根因不是cross-store allocator的victim快照刷新时机，
而是**请求自身的exact prefix在recovery期间完全未受保护**。

证据链：

- `Req.init_next_round_input`在`schedule_batch.py`中调用
  `restore_request_prefix`；
- 而请求的prefix锁`_req_inc_lock_ref(req)`是在
  `schedule_policy.add_one_req`中才获取，发生在**之后**；
- 因此recovery执行时`req.last_node.lock_ref == 0`；
- `RadixCache.cross_store_resources()`的过滤条件恰好是`node.lock_ref == 0`，
  于是请求自己的prefix节点成为**合法victim**；
- 压力下`allocate_recovery_slots`驱逐它→`allocator.free(node.value)`→
  slot回到free list→紧接着`allocate_backend()`把**同一批slot**作为recovery
  目的地返回→请求即将attend的自身prefix KV被静默覆写。

这完整解释了“机械证据全过、只有输出错”：byte/token/lease/reset记账全部正确，
被破坏的是数据本身。

修复（`af81934e4`，最终head `c405343c8`）：

1. 新增`protect_request_prefix`上下文管理器，在整个recovery窗口持有标准
   prefix锁；`inc_lock_ref`一路walk到root，因此保护整条matched chain并将其
   移出`evictable_leaves`，同时覆盖嵌套的`ensure_device` H2D分配路径。
2. 在`schedule_batch.py`唯一调用点包裹，同时覆盖EPIC与普通两条路径。
3. 加固exact victim guard：额外校验节点仍挂在父节点上；stale victim现在抛
   `KeyError`（allocator已有回滚处理），不再触发`_delete_leaf`断言杀死
   scheduler进程。

验证：

- 容器内对照实验确认`inc_lock_ref`/`dec_lock_ref`完全对称
  （`(2,0,0,0)`→`(0,2,1,1)`→`(2,0,0,0)`），加锁期间victim数为0，无锁泄漏。
- 相关回归`204 passed, 5 skipped`；新增5个回归。唯一失败
  `test_radix_cache_unit.py::test_memory_allocated`经`git stash`对照确认为
  **改动前既有失败**。

### 2026-07-27 P6-H首次通过

`run_id=p6-h-20260727T071106Z`，`status=valid`：

- 2个formal round输出均与matched dense**逐token一致**；
- host export与demand H2D均为`1024` token / `117440512` bytes；
- `cross_store_demoted_bytes_total=117440512`，真实device→host demotion；
- reset invariant通过，store五项gauge全部归零；
- 作用域仍为`host_backend=allocator_cpu_copy`、
  `hicache_tier_exercised=false`，**不解锁**Phase7 HiCache track。

### 2026-07-27 CL1重跑：NONE获得有效因果归因（重大）

在修复后的底座上重跑CL1 screening
（`raw_sha256=fe05d3dc34594a25ef8a...`）：

- 48个paired repeat的guardrail失败为`quality_8=17`、`first_token=6`，
  与修复前**完全一致**；
- 性能数字同样几乎不变（body2048 median request-path `1.947x–1.970x`）。

**这是本轮最重要的科学结论**：P0修复没有改变CL1的guardrail结果，因此
CL1的输出偏离**不是**由该缺陷造成的，先前记录的“因果归因无效”警告已解除。

机制解释（两个实验的差异是决定性的）：

| 实验 | source header | target header | 上下文 | 正确期望 | 实测 |
| --- | --- | --- | --- | --- | --- |
| P6-H | 同一个 | 同一个 | 完全相同 | 必须逐token一致 | 修复后一致 |
| CL1 | `32_000+` | `36_000+` | **不同** | 天然有损 | 稳定偏离 |

即CL1把在`source_header`下计算的body KV复制到`target_header`之后使用，
前缀不同导致attention上下文不同，KV**本来就是近似的**。

按独立review要求补做的`header × pressure` 2×2对照
（artifact `/results/phase6-gpu/context-vs-pressure-2x2.json`）：

| header | 压力 | 是否发生eviction | 输出与dense一致 |
| --- | --- | --- | --- |
| 相同 | 低（8000 token） | 否 | **一致** |
| 相同 | 高（3400 token，真实demotion+H2D） | 是 | **一致** |
| 不同 | 高（rho2.0） | 是 | 不一致（48中12例） |
| 不同 | **低（rho0.5，observed 0.519）** | **否** | **不一致（4中1例）** |

决定性一格是最后一行：在**完全没有发生eviction**的条件下，不同header的复用
**仍然偏离**，且body1024 repeat0的偏离序列与高压力下**完全相同**
（dense `[82,198,271,...]` vs approx `[82,198,198,...]`）。
已修复的缺陷必须依赖eviction才能触发，因此**它无法解释该偏离**。

**措辞（按两轮review修正后采纳）**：`NONE`在冻结exact-output promotion规则
下成立。数据**排除了已修复的eviction-dependent P0**是这些偏离的原因，
但**没有证明context差异是唯一原因**，也**未排除header-dependent实现缺陷**。

需明确的方法学限制：所谓2×2**并非真正的factorial设计**——它拼接了P6-H与
CL1两套实验，二者的runner、policy、chunk配置、plugin env、代码SHA与重复数
均不同；低压CL1的eviction counter也未覆盖关键target allocation窗口。
最强剩余替代解释是**不同header路径特有的position/index/cache-metadata
对齐缺陷**，且该解释不依赖eviction。

**结论：`practical family = NONE`成立。** 其作用域限定为本模型、合成
prompt族、exact-output不变量、本GPU与chunk配置下**冻结promotion规则**的
结论，**不是**普遍不可行性claim。

统计口径更正（同样按review采纳）：

- 所报“paired target p95”实际是pooled样本上的`p95(approx)/p95(dense)`，
  **不是**配对统计量；
- N=1/2/4/8摊销是**外推值**，不是真实测得的多次复用；
- 独立复制单元很少：CL1为3个restart级单元、CL2为2、CL3多数为1；
  同一trace内的请求可用于描述性p95，但不能当作独立重复；
- P6-H只有1 restart/2 round，且只校验output token，不是bitwise KV/logit保真。

### 2026-07-27 P6-4完整矩阵与三处后续修复

按独立review的P1-2/P1-3继续修复（详见`TRACKING.md`）：

1. `40f09c1fe`：recovery slot在admission前挂载、被拒绝时无人释放。改为
   provisional所有权模型。全目录回归对照确认既有基线为`935 failed`，
   本次净增3个pass，该935与本次改动无关。
2. `3379e6699`：stale victim导致整个allocation放弃；改为跳过并刷新重试，
   同时把detached节点移出`evictable_leaves`。
3. `0f379eb04`+`fb284cad4`：P6-4 runner逐cell容错，单个不可达cell记为
   `diagnostic-unavailable`并继续，不再中止整个矩阵。

**重要更正**：P1-3与P1-2**都不是**S0/rho2 OOM的成因；两次修复后仍确定性
OOM，因此该cell归类为**真实容量不可达**而非实现缺陷。
诊断C（v2，`0.05s`采样）已独立确证这一点：死亡瞬间approximate store为空，
且exact压力此前已成功回收`2.2GB` approximate内存。

P6-4最终结果（`run_id=p6-4-20260727T104820Z`）：

| cell | requested/observed | 结论 |
| --- | --- | --- |
| S4 rho1.1 | `20713`/`20713` | 非R4 profile可达；cell级仍为`diagnostic-unavailable` |
| S4 rho1.5 | `15190`/`15190` | 非R4 profile可达；cell级仍为`diagnostic-unavailable` |
| S4 rho2.0 | `11392`/`11392` | 非R4 profile可达；cell级仍为`diagnostic-unavailable` |
| S0 rho2.0 | `11392`/不可达 | device耗尽 |
| S4 rho3.0 | `7595`/不可达 | device耗尽 |

- **双向pressure首次`passed=True`**：exact→approx victim `47.5 GB`；
  approx→exact victim `58.8 GB`。这是Phase6 Exit的核心要求之一。
- 措辞限定（按Exit review修正）：完整矩阵中**每个cell的顶层状态都是
  `diagnostic-unavailable`**；成立的是“三个S4 cell中的四个非R4 profile
  （`exact_only`/`r0_like`/`r1_like_k32`/`r2_like`）reachable且valid”，
  **应按profile级而非cell级陈述**。**R1-like worst-case（k32）footprint可达**。
- `r4_like`（约5x）在所有cell均不可达，属计划预先允许的R4例外。
- 整体`status=inconclusive`：无cell达到全`valid`，且
  `fallback_reachability.rounds=0`。

### 2026-07-27 诊断C（v2，已更正v1）：S0/rho2确实是容量不可达

**v1结论已撤回。** v1以`0.4s`间隔轮询，而临近死亡时workload分配速度远快于
该间隔（最后`1.3s`内`num_used_tokens`由`5376`涨到`10688`），因此“最后一个
成功样本”早于致命请求，据此得出的“有可回收内存却未回收”属于采样伪影。

以`0.05s`重采样的死亡瞬间真实状态：

```
approx_kv_store_device_bytes = 0.0     ← 已完全清空
approx_kv_store_records      = 0
approx_kv_store_leases       = 0
num_used_tokens / capacity   = 10688 / 11392   (token_usage 0.94)
可用 704 token，而请求需要 1024
cross_store_reservation_failures_total = 从未出现
```

回收路径被证明工作正常，死亡瞬间累计`cross_store_evicted_bytes_total`：

| 方向 | 字节 |
| --- | ---: |
| exact requester → approximate victim | `2,202,009,600` |
| approximate requester → approximate victim | `411,041,792` |
| exact requester → exact victim | `8,592,424,960` |
| approximate requester → exact victim | `1,767,800,832` |

exact压力已成功从approximate对象回收`2.2GB`；到死亡时approximate store已被
榨干，**没有可回收资源残留**。

**固定结论**：

- S0/LRU rho2.0**强烈支持容量耗尽**，`diagnostic-unavailable`标签正确；
- **S4 rho3.0已于2026-07-27单独实测**（`0.05s`采样，`cap=7595`）：
  死亡瞬间`approx_kv_store_device_bytes=0`、`records=0`，
  且exact压力已从approximate对象回收`1,746,927,616` bytes。
  与S0/rho2**签名一致**，同为真实容量不可达。该判定不再依赖外推；
- **不存在cross-store回收缺陷**；
- v1所称“832个token无法归属”是伪影：`num_used_tokens`已包含approximate
  store占用的slot，不能与之相加。

**对最后一项Exit证据的影响**：既然无缺陷可修，reservation-failure关联的
fallback不能靠修bug自然获得。用户于2026-07-27选定**方案C**：该项以
`indirectly_verified`强度结案，不改共享上游路径，也不使用fault injection。

已直接证明（经两轮Exit review修正后）：

- allocator在**任意**失败点正确回滚（遍历全部`AllocationFailurePoint`，CPU）；
- `allocate_recovery_slots`把被拒reservation转为None并归因为
  `cross_store_reservation_failed`（CPU，mutation验证）——
  **但该测试只调用`allocate_recovery_slots`，未经过`finalize_copy_reuse`、
  scheduler或真实dense推理**；
- exact压力真实回收`2.2GB` approximate内存（GPU）。

**已撤回的两条错误证据**（由Exit review A发现）：

1. “GPU上真实执行12次dense fallback”——这12次全部来自`exact_only` profile；
   该profile没有approximate metadata，runner在
   `run_p6_4_capacity_pilot.py:413-419`仅凭`cached_tokens < expected`就把
   **普通exact-cache miss**标为`dense_fallback`，其`request_outcomes`为空、
   reservation失败为0。**它们不是approximate dense fallback。**
2. “`r4_like`的4096 fallback token”——那是registration容量失败，
   其replay outcome是`approximate_gpu_recovery`，不是dense fallback。

未证明：任何**集成请求**在cross-store reservation失败后真正走完dense路径
并完成输出。

**附带发现的报告缺陷**：runner把无approximate metadata时的exact-cache miss
标为`dense_fallback`，该混淆必须在Phase7 fallback报告前修复。

`p6-4.json`未被修改，仍如实记录`fallback_reachability.passed=false`；
disposition单独记录在`phase6-exit-fallback-disposition.json`与计划§7.9.1。

**以下为P6-F执行前的历史状态，现已被最终disposition取代**：

```
technical_exit = PASS WITH CAVEATS
```

治理性豁免已被用户明确撤销并由P6-F test-only集成证据取代。
仍未进入Phase7。

### Phase6 Exit当前逐条状态

| Exit要求 | 状态 |
| --- | --- |
| exact/approx/host同budget安全竞争 | **满足** |
| 双向pressure有效 | **满足**（首次） |
| allocation失败可回滚 | 满足 |
| fixed40四rho可运行或明确不可达 | **满足**：三个S4 cell中的四个non-R4 profile可达；S0/rho2与S4/rho3有独立capacity-limit证据；所有顶层full-matrix cell仍为`diagnostic-unavailable` |
| R1-like worst-case footprint | **满足**（k32可达） |
| generic host roundtrip canary | **满足**（P6-H valid） |
| 无泄漏、无orphan | 满足（store gauge全归零） |
| 近似reuse压力下数据保真 | **满足（范围受限）**：1 restart/2 round的8-token输出canary与dense一致；未验证bitwise KV、logit等价或低概率损坏 |
| raw/commit/env/test provenance | **满足formal Exit package**：manifest 48/48；raw JSONL与primary P6-H/P6-4/P6-F logs已版本化 |
| dense fallback可达性 | **满足（fault-injected canary强度）**；自然压力可达性未证明 |

关于最后一项的精确事实（**此处原有的“12次dense fallback”表述已撤回**）：

- CPU测试覆盖了**可逆动作的回滚**，以及在`allocate_recovery_slots`边界上的
  **reservation失败归因**；
- GPU运行覆盖了**双向回收**；
- P6-F v3已提供一个集成approximate请求在reservation失败后走完dense执行的
  test-only canary；自然压力可达性仍未证明。

原先据以声称“已观测12次dense fallback”的记录，实为`exact_only` profile的
**普通exact-cache miss被runner误标**；该标签缺陷已在`84c4e5202`修复为
`exact_cache_miss`，但**既有artifact仍保留旧标签**，读取时必须注意。

拿不到自然证据的原因：唯一会真正触发reservation失败的cell（S0/rho2、
S4/rho3）在此之前就因`alloc_token_slots`抛`RuntimeError`杀死server；
且诊断C已证明这两个cell是真实容量耗尽，无缺陷可修。

补充结果（2026-07-27）：只跑`exact_only`+`r2_like`时，
**S4 rho1.1与S4 rho2.0首次达到`status=valid`**。这确定完整矩阵中所有cell被
判为`diagnostic-unavailable`的唯一原因是`r4_like`（约5x）不可达，
即计划预先允许的R4例外，而非底座缺陷。

reservation失败无法用配置获得，已实测排除两条路径：

- rho2.5同样device耗尽，“可运行/耗尽”之间窗口过窄；
- `--kv-bytes-per-token`放大4倍仍`resv_fail=0`，因为`CrossStoreBudget`
  用同一单位换算limit与已用量，缩放在等式两边互相抵消。

该手段随后已由用户明确采用，并实现为P6-F test-only集成canary。
其证据强度固定为“fallback路径在注入失败下可用”，不升级为自然可达。

**影响范围判定（2026-07-27）：该项不波及全部research。**
CL1/CL2实测未启用`SGLANG_APPROX_KV_CROSS_STORE`，CL3为零GPU重算，
P6-H与P6-4可达cell的`reservation_failures`均为0，因此
`practical=NONE`、chunk伪影、双向pressure、压力下保真等结论均不依赖该项。

**证据覆盖度**：CPU回归遍历全部`AllocationFailurePoint`证明可逆回滚；
P6-F v3补齐集成链路后半段
（reservation失败→`reuse/dense_fallback`→真实dense模型执行→输出完成），
并有独立无注入server作为control。仅缺**自然压力**下发生一次reservation失败，
该缺口被明确保留为scope caveat，而不是技术Exit blocker。

**前瞻风险**：`alloc_token_slots`在cross-store腾不出空间时抛`RuntimeError`
杀死scheduler，会影响任何在容量极限附近运行的后续实验（含Phase7高rho），
不限于本项，应在Phase7高压力矩阵前修复。

因此Phase6 Exit只剩这一项未完全取得证据，其余全部满足。仍未进入Phase7。

### 2026-07-26 P6-4结果与CL3 Phase5零GPU重算

- P6-4完整profile矩阵在`hierarchical/rho1.5`的`r0_like` profile崩溃，
  status为`invalid`（实现缺陷，不是`diagnostic-unavailable`）：

```
restore_request_prefix -> finalize_copy_reuse -> allocate_recovery_slots
  -> cross_store/coordinator.allocate_tokens -> cross_store/allocator.allocate
  -> radix_cache.evict -> radix_cache._delete_leaf
AssertionError: parent does not have child key
```

- 根因：`cross_store/allocator.py`驱逐循环在一次迭代内按快照顺序执行整个
  eviction closure，而快照只在“上一轮驱逐过exact资源”时于下一次循环开头刷新。
  同轮内先执行的驱逐可能已把后续resource对应的radix节点从父节点摘除，
  对该stale节点再次`evict`即触发断言。这同时解释了P6-H的压力下KV损坏：
  stale节点被重复释放会把仍被近似对象引用的device slot放回free list并被覆写。
  **P6-H数据保真失败与P6-4结构断言是同一缺陷类。**
- 已单独取得不受该缺陷影响的exact-only capacity baseline
  （`/results/phase6-gpu/p6-4-exact-only.json`，
  `run_id=p6-4-20260727T030317Z`）：
  - 5个cell（S4 rho`1.1/1.5/2.0/3.0` + S0 rho2 paired）全部完成；
  - requested与observed capacity完全相等：
    `20713/15190/11392/11392/7595` token，容差检查5/5通过；
  - `exact_evicted_bytes=18007851008`，四个rho均发生真实exact eviction；
  - 因只跑exact_only，`bidirectional_pressure`与`fallback_reachability`
    按定义不成立，整体status为`inconclusive`。
- CL3 Phase5零GPU重算已完成（本次新增runner
  `benchmark/approx_kv/run_cl3_phase5_recompute.py`，不重跑任何Phase5 GPU
  cell）：artifact `/results/phase6-gpu/cl3-phase5-recalculation.json`，
  `run_id=cl3-20260727T031459Z`，
  `raw_sha256=17f010b75e5f18dd38c675550ef041a90d12e93211f2384093e759b13bd3af41`，
  覆盖40个已提交cell、18个scheduler paired行与9个prefetch paired行。
- **FINDING-CL3-A（重大）**：S4的优势完全依赖分母选择。

| 策略（相对S0 LRU） | 分母 | rho1.1 | rho1.5 | rho2.0 | rho3.0 |
| --- | --- | ---: | ---: | ---: | ---: |
| S4 hierarchical | workflow-only | `1.457` | `1.321` | `1.148` | `1.147` |
| S1 workflow-steps | workflow-only | `1.454` | `1.135` | `0.996` | `1.007` |
| S2 Belady-style | workflow-only | `1.451` | `1.125` | `1.004` | `1.008` |
| S3 recovery-value | workflow-only | `1.466` | `1.138` | `1.005` | `1.003` |
| S4 hierarchical | all-reusable | `1.180` | `1.089` | `1.151` | `1.097` |
| S1 workflow-steps | all-reusable | `1.177` | `1.096` | `1.149` | `1.107` |
| S2 Belady-style | all-reusable | `1.178` | `1.088` | `1.155` | `1.109` |
| S3 recovery-value | all-reusable | `1.187` | `1.100` | `1.158` | `1.102` |

- 精确表述（按review修正）：**S4相对其他非LRU策略（S1–S3）的独特高rho优势
  只出现在workflow-only口径**；在all-reusable口径下S1–S4相对S0**均有相近的
  描述性改善**（`1.09x–1.19x`），现有restart数**不足以支持策略排序**。
  注意“S4的优势消失”是不准确的——消失的是它相对S1–S3的**独特性**，
  它相对S0仍有数值改善。
- all-reusable的p95比值全部在`0.984–1.009`，四种策略均未恶化p95。
- 措辞更正：CL3多数cell只有1个restart，因此只能写“数值上几乎不可区分”，
  **不得**写成“within noise”这类统计判断。
- **FINDING-CL3-B**：prefetch矩阵改为与同策略P0配对（Phase5 prefetch矩阵没有
  LRU臂，原先与LRU比较不成立）后，P1/P2/P3在两种分母下都是`0.989–1.004`，
  即无收益。Phase5“默认S4+P0”的结论在正确对照下依然成立。
- CL3同时落实：per-request clamp的hit fraction、full-trace wall-clock、
  per-role TTFT与miss、paired/per-restart统计，以及S2只称Belady-style。
  “variable-size offline optimum”按计划§12归为Phase7交付物，不是Closeout阻塞项。

- 用户已将当前实验明确收窄为：不研究 AST、label、自动分段或 indexing；手工固定一个大代码段，在不同 role/prefix/context 下做有损 KV 恢复与调度实验。
- “有损 KV”专指避免完整目标-context prefill，通过 raw KV reuse + RoPE、KVCOMM base/offset/anchor 或局部 recompute/repair 近似恢复 KV；不指量化、低比特或普通 KV pruning。
- 当前唯一性能主目标是客户端观测 TTFT（包含排队），同时记录 server-side 分解；不评估语义正确率、代码正确率或输出一致性，最低门槛只是请求不崩溃并返回首 token。
- 第一阶段只做 sequential `Architect -> Coder -> Debugger` 与 `Debugger -> Coder` retry；并发 workflow 和并发 prefetch 明确后置。
- 用户要求同时尝试多条恢复路径和多种 scheduler/eviction 组合，不能预设某一种方法最好；synthetic code-like data 可用于制造可控 high pressure。
- 论文发现和机制事实只使用已配置的 arXiv/alphaXiv MCP；其他来源只可用于代码定位。
- 只读核对确认 `ccdd2023/sglang:main` 为 `3343a79466aa714d34a14d08d3929f7953a47212`，upstream `sgl-project/sglang:main` 为 `c0ed009f5b566be023661bd4e93065b8b4b8b31f`；前者是后者祖先，落后 4,654 commits，可 fast-forward-only。
- 远程当前不存在 `latest-main`；实施时将在 Docker 内 fast-forward fork main 后创建并推送该分支。
- Git fetch、checkout、编辑、依赖下载、构建、测试、server 和 benchmark 全部必须在 Docker 内执行；宿主机只负责启动容器、挂载目录和收集结果。
- 当前实施计划为`IMPLEMENTATION_PLAN_LATEST.md`（**V5 Current / Latest**）；V1/V2/V3/V4均已归档。
- 已在空目录中初始化 Git 仓库，默认分支为 `main`。
- 已建立项目主文档、讨论追踪文件和会话交接文件。
- 已建立 Copilot 仓库指令，确保后续会话先读取交接信息并持续维护文档。
- 已确认 `https://github.com/ccdd2023/sglang` 是项目交流和 prototype 代码实现仓库。
- 已显式使用系统保存的 `ccdd2023` 身份验证该仓库，权限为 `ADMIN`，默认分支为 `main`。
- 验证时 GitHub CLI 的当前默认账号不是 `ccdd2023`；后续操作必须显式选择该账号，不能依赖默认账号。
- 已定位历史研究工作区 `/home/chris/Workspaces/kvcache-research`。
- 历史 SGLang 移植位于 `kvflow-sglang`，当前分支 `feature/workflow-priority` 与 `ccdd2023/sglang` 远程同名分支精确同步，提交为 `5bb9afc9234aa9caa9df51e87f119e5bfaf186de`。
- 本机 SM75 兼容运行版本位于 `sglang-running`，分支为 `fix/qwen3-0.6b-docker-sm75`；Docker 运行脚本和镜像存在，但当前没有正在运行的 SGLang 容器。
- 已通过 alphaXiv 定位并收藏 KVFlow `2507.07400`、KVCOMM `2510.12872`，并下载 PDF 到 `research/papers/`。
- 两个独立 subagent 已分别完成 KVFlow 与 KVCOMM 深入研究。
- 已形成统一研究综合文档 `research/RESEARCH_SYNTHESIS.md`。
- 已确认用户所指 KVCOMM 是 `2510.12872`；同名 `2510.03346` 是不同的跨模型 selective layer KV sharing 工作。
- 已确认 KVCOMM 没有实现名为“可变编码”的格式；当前准确解释是 base KV、context-dependent offset、RoPE 重定位与 anchor interpolation。
- 已确认用户所指的 Yu Guofan 对应 GitHub 账号 `flaminyu`。
- 已审查最近两个月的线性研究分支：
  `para_temp -> feature/context-aware-kv-reuse -> agenttemplatekv-eurosys-2026-06 -> phase-2.7-prerot -> fix/placeholder-pool-activation`。
- 已形成 `research/YU_GUOFAN_BRANCH_REVIEW.md`，记录分支归属、KVCOMM 忠实度、其他论文、代码缺陷、实验边界和可继承模块。
- 已判定该研究线不是 KVCOMM 的完整复刻；最新论文稿也已把 KVFlow/KVCOMM 降为 prior work / implementation inspiration。
- 已判定最新分支不应作为本项目继续开发的核心基线；应从接近 upstream 的干净分支或 `feature/workflow-priority` 重建 faithful KVCOMM。
- 已确认可选择性继承的内容包括 benchmark/telemetry、AST slicing、HKVD measurement、RoPE helper 和 priority eviction 经验。
- 已确认必须重写的内容包括 KVCOMM base/offset/interpolation、placeholder/chunk pool lifecycle、offline KV writer/loader、context confidence 和 selective recompute。
- 一个 GPT-5.6 Sol Max 专项 research agent 和四个独立 novelty/brainstorm agent 已全部完成；四个评估代理在看到文献结果后又完成一次针对性修正。
- 专项检索发现两篇 A 类直接先例：
  - CodeComp `2604.10235` 已用 Joern CPG（AST/CFG/PDG）直接控制 repository-level 代码 span 的 KV 预算、保护和淘汰，但属于单请求内 compression。
  - Functional Cache Grafting / FCGraft `2606.13097` 已把函数作为 KV 对象，支持 function-ID 索引、stitching、局部 patch、成功后更新和 GPU/DRAM residency，但面向机器人 Code-as-Policies。
- 强邻近工作 MEPIC `2512.16822` 和 MiniPIC `2606.13126` 已覆盖 code chunk/file span 的 canonical pages、position-independent reuse 和 memory tier primitive。
- 因此不能再声称“首个 AST-aware、function-level 或 code-specific hierarchical KV cache”。
- 当前原始方案 novelty 约为 `2/5`；若收窄为 evolving repository 的 version consistency、dependency invalidation、calibrated cross-role reconstruction 和 artifact-level planning，保守上限约为 `3.3–3.6/5`。
- 已创建 `research/AST_INDEXED_KV_CACHE_NOVELTY_REPORT.md`，包含 A/B/C/D prior art、closest matrix、causal correctness、机制排序、workflow contract、实验矩阵和 kill criteria。
- 已创建 `research/AST_INDEXED_KV_CACHE_STEP_BY_STEP.md`，完整保存从问题定义到 prior art、因果边界、materialized-view thesis、系统架构、三阶段 workflow、prototype、评测和 kill criteria 的 33 步教学式推导。
- 用户所写 “Observe 4.6” 暂按最接近的可用模型 Claude Opus 4.6 执行。
- 当前主线改为：把代码 KV 建模为带源码版本、因果上下文与风险界限的 materialized view；AST index、priority 和 CPU/GPU tier 只作为系统支撑，不单独作为核心 novelty。
- 用户进一步澄清：AST 从来不是主要研究切入点；本项目更重要的是 repository/version lifecycle、cache consistency、cross-version reuse、incremental invalidation/rematerialization 和 serving harnessing。
- 按用户纠正后的明确要求，最终由三个独立 **GPT-5.6 Sol Max** 同步研究代理分别完成 2024、2025、2026 年调研。
- 三个 Sol 代理均优先尝试 alphaXiv/arXiv MCP 的论文发现与全文阅读，并追踪至 2026-07-13 的最新 revision、引用链和实现状态。
- 早期误启或被运行时取消的代理没有产出，被最终三份同步 Sol Max 报告完全取代。
- 调研核心是 Git commit、branch、worktree、repository/source version、patch epoch 是否直接进入 Transformer attention KV 的 key、identity、reuse、invalidation、repair、rematerialization 或 memory-tier lifecycle。
- 三个最终 GPT-5.6 Sol Max 同步研究代理已分别完成 2024、2025、2026 年报告；alphaXiv MCP 在调研期间多次返回 HTTP 429，代理继续以 arXiv PDF/API、DBLP、正式 venue 和官方代码交叉核查，主会话另行抽查八篇关键论文全文。
- 三年严格 A 类 direct prior art 均为 0：本次检索未发现 Git commit/branch/worktree/repository source version 被用作普通 attention-KV 的一等 identity、validity 或 coherence 协议。
- 2024 最接近的是 PIE `2407.03157`：直接处理代码 edit 后的 prefix reuse、replacement recompute 和 suffix Key relocation，但没有 source-version identity，且 suffix K/V 的旧语义影响仍在。
- 2025 最接近的是 Cache-Craft、EFIM、KVCOMM、MEPIC：分别覆盖 contextual repair、code-infilling layout、cross-context offset 和 content-hash physical objects，但都没有 repository revision lifecycle。
- 2026 最接近的能力被分散在：
  - Leyline：mutable context directives；
  - FCGraft：function KV objects、patch/update、GPU/DRAM lifecycle；
  - Irminsul/MEPIC/MiniPIC：content-addressed、position-independent objects；
  - Streaming Knowledge Compilation：time-evolving content 的 staleness 与 affected-entity recompilation；
  - Code Isn't Memory：Git working-copy Merkle diff 和 incremental repository index，但不保存 attention KV；
  - Concordia：runtime checkpoint version/epoch coherence，但不是 source version。
- 已创建 `research/GIT_CODEBASE_VERSIONED_KV_CACHE_PRIOR_ART.md`，完整记录年度证据、closest matrix、系统空白、RepoKV-MVCC thesis、机制、实验和 kill criteria。
- 已完成 Vast.ai RTX PRO 6000 S 的 hosting、Docker、账号连接、安全、成本和实验收益评估。
- Vast.ai 标准实例是 provider 机器上的 Linux Docker container；GPU 运行期间独占，CPU/RAM 按 GPU 份额分配，disk 在创建时固定，标准实例不支持 Docker-in-Docker。
- 当前 `lmsysorg/sglang:dev` 已验证为 CUDA 12.9.1、PyTorch 2.9.1+cu129，编译架构包含 `sm_120`/`compute_120`，可用于 RTX PRO 6000 首轮 smoke test。
- `sglang-running` 源码已有 RTX PRO 6000/SM120 专用检测和 Triton 路径；现有本地 `docker run` 脚本不能在 Vast container 内原样执行，必须转换为 Vast template/on-start/entrypoint。
- 现有 Dockerfile 的 DeepEP arch list 没有 SM120；dense single-GPU 7B/8B 主线可先运行，DeepEP/MoE 必须后续单独修复验证。
- 当前系统未安装 `vastai` CLI，也没有配置 `~/.config/vastai/vast_api_key`；尚未实际连接用户账号或租用实例。
- 推荐固定为混合工作流：本地负责 Git、文档、代码、单元测试和结果分析；Vast.ai 只负责短时 GPU integration、7B/8B、长上下文、HiCache/KVCOMM 和性能实验。
- 已创建 `research/VASTAI_RTX_PRO_6000_WORKFLOW.md`，包含账号安全流程、offer 筛选、模板映射、实验 manifest、分阶段计划与验收条件。
- 用户仍担心 codebase source-version-aware KV idea 已有直接先例，因此最初启动十代理独立复核。
- 用户于 2026-07-15 将最终复核范围收缩为 2025 和 2026：`version-scan-01` 至 `version-scan-03` 已收到停止指令且结果不纳入；`version-scan-04` 只保留 2025-01-01 至 2025-01-06。
- 最终报告只覆盖 2025-01-01 至 2026-07-15 的七个保留分段，即 `version-scan-04` 至 `version-scan-10`。
- 保留代理均为 GPT-5.6 Sol Max research agent；每个代理必须在报告首行记录负责区间，并按论文首次公开日期归属。
- 所有代理使用统一 A/B/C/D 定义，必须记录全文证据、负搜索、boundary spillover、代码和截至 2026-07-15 的最新 revision。
- 第十代理额外覆盖最近 7/30/90 天尚未进入 DBLP 的最新 arXiv/OpenReview/workshop 工作。
- 已建立 `research/CODE_VERSION_KV_10_AGENT_PRIOR_ART_REPORT.md` 报告骨架和 SQLite `prior_art_segments` 区间登记。
- 最终交付除完整技术报告外，还必须包含一到两段面向 presentation 的简短中文 summary，用易懂语言说明问题、核心方案和价值。
- 十代理中的 `version-scan-09` 已完成 2026-01-13 至 2026-04-14 区间：全文审查 20 篇，A/B/C/D=`0/8/8/4`，未发现 repository-source-version-aware attention-KV coherence 的 A 类直接先例。
- 第九段最接近工作分散在 KEEP、TableCache、COMB、KV Packet、CodeComp、MARS、CAID、Lore 和 Repository Intelligence Graph；仍呈现“有 Git/version semantics 的工作没有 K/V，有 K/V update/tier semantics 的工作没有 repository version”的断裂。
- `version-scan-10` 已完成 2026-04-15 至 2026-07-15：全文核查 21 篇，互斥 A/B/C/D=`0/5/15/1`，最近 7/30/90 天 A 均为 0。
- 第十段确认宽泛 primitive 已高度拥挤，组合显而易见性风险上升；核心 claim 必须收窄到 source snapshot identity、dependency-driven K/V invalidation、branch/worktree isolation、cross-version exact alias 和 source-aware stale audit。
- `version-scan-06` 已完成 2025-04-10 至 2025-07-11：18 篇核心候选，A/B/C/D=`0/1/12/5`，A=0 置信度约 0.90。
- 第六段确认 MemOS 是重要高风险邻近项：它同时讨论 memory versioning 与 activation KV，但仍未把 Git/source version 连接到 KV validity、invalidation 或 coherence。
- `version-scan-05` 已完成 2025-01-07 至 2025-04-09：11 篇候选，A/B/C/D=`0/3/6/2`，A=0 置信度约 0.92。
- 第五段最接近的是 Cache-Craft、MPIC、KVShare 的 selective repair，以及 HyperRAG、KVLink、SentenceKV 的 modular/tiered KV；SyncMind 和 Repository-level Code Search 具有 Git/repository 语义但不保存 attention K/V。
- `version-scan-08` 已完成 2025-10-13 至 2026-01-12：12 篇候选，A/B/C/D=`0/1/10/1`，A=0 置信度约 0.87。
- 第八段显示 mutable-KV、persistent/tiered KV 与 Git/branch-aware coding agent 仍分属不同工作；PortGPT 有明确 Git/branch 语义但不保存 attention K/V。
- 2026-01-01 至 2026-07-15 已由 `version-scan-08`、`version-scan-09`、`version-scan-10` 完整覆盖，三段严格 A 类均为 0。
- 对外概括 core idea 时使用简短分段：先解释现有 KV Cache 缺少 evolving repository 的 source version、依赖关系与固定 Coding Agent workflow 的统一一致性管理，再解释 versioned causal KV materialized views、主要 concerns 和最小 prototype。
- 已启动独立 GPT-5.6 Sol Max 后台工程调研，评估在历史修改版 SGLang 上完整复现 KVCOMM `2510.12872` 的功能可行性、论文级复现难度、P0/P1/P2 blocking points、代码复用边界和分阶段实施路线。
- `version-scan-07` 已完成 2025-07-12 至 2025-10-12：21 篇候选，A/B/C/D=`0/5/14/2`，A=0 置信度约 0.90。
- 第七段确认最接近的组合由 KVCOMM/CacheClip 的 mutable-context repair、LMCache/AdaptCache 的持久分层管理和 RepoMem/LinkAnchor 的 Git history 语义组成，但 source lineage 与 attention-KV coherence 仍未被直接耦合。
- Whole-codebase KV 的准确实现不是单条连续 prefill，而是全库 logical index + 非重叠 canonical artifact units + hotset physical KV。module/class/file 是逻辑容器，首选物理单元是 function/method、module preamble、class init/field block，超长单元才继续切 statement/basic block。
- Dependency analysis 复用 tree-sitter、LSP、compiler/CPG、build graph、test trace 等已有技术，并产生保守 reverse dependency cone；它不能独立证明 KV 可复用，exact path 仍由 token/context/model fingerprint 保证，approximate path 仍需 probe 和 dense fallback。
- 研究候选 novelty 不在原创 parser、AST 或依赖图，而在把 source-version/dependency events 直接耦合到 attention-KV identity、invalidation、rematerialization、tier lifecycle 和 stale audit。
- 2025-01-01 至 2026-07-15 的七个保留分段已全部完成，共核查 105 篇主候选，A/B/C/D=`0/23/67/15`。
- 所有分段严格 A 类均为 0；最终复核未发现 repository/source version 直接作为普通 attention-KV identity、validity、dependency invalidation 和 physical-tier coherence domain 的公开系统。
- 最终安全 thesis 保持为 `repository-version-aware attention-KV coherence / versioned causal KV materialized views for evolving codebases`。
- 宽泛组件均已有先例，组合显而易见性风险较高；下一步必须以 coherence protocol、真实 Git trace、correctness 和端到端收益证明贡献。
- artifact 不是任意细碎 token，也不是只靠 Git 索引。推荐单位是 function/method、module preamble、class init/field block；系统同时使用 symbol/AST、embedding relevance、Git snapshot、content hash、dependency graph、causal-context signature 和 physical-page index。
- Git 只回答“这是哪个源码版本”，不负责相关性检索，也不能单独保证 KV exactness。独立 artifact KV 不能在新顺序下盲目拼接；必须满足相同 causal context，或经过 KVCOMM/重算/fallback。
- KVCOMM `2510.12872` 的 GPU-only faithful functional reproduction 在 SGLang 上技术可行，难度约 4/5；论文级性能复现条件可行、难度约 5/5。
- 推荐从接近 upstream 的 clean fixed SHA 开始，`feature/workflow-priority` 可作为第二选择；AgentTemplateKV 最新分支只作为实验资产 donor。
- 最大 P0 是 token/template 对齐、连续 prefix、完整 K/V offset 与 RoPE、approximate provenance 隔离、slot ownership/lifecycle 和强 fingerprint。
- 当前估计单人功能版 8–14 人周，加入 lifecycle/tiering 14–24 人周，论文性能版累计 22–36 人周；论文级 claim 仍受 H100 和精确环境阻塞。
- Physical KV 是模型每层、每个 token 的真实 K/V tensors，不是源码、embedding 或 Git object；按 token pages 存储并绑定 artifact version、causal context、position、model/template fingerprint 和 tier。
- 仓库 bootstrap 从一个当前目标 fixed SHA 开始，不从第一个 commit 开始。先建立全库 logical catalog，再稀疏物化 hot canonical KV；后续 commit/worktree 通过 diff 做 alias、失效、重算和 GC。
- function/method 是第一版默认 logical artifact，但 exact reuse 的单位是完整 causal prefix/context signature，KVCOMM 的单位是 placeholder span，physical KV 的单位是 token page；四者不能混为同一粒度。
- 最终存储模型不是每个函数保存两份完整 KV，而是三层：跨多个 artifacts 的 exact Radix/bundle cache、函数级 canonical base store、以及有上限的 context residual/anchor store。
- 系统同时维护 source dependency graph 与 prompt causal graph；前者做检索和保守影响分析，后者记录真实 prompt 前序并决定 exact invalidation。
- Prompt Compiler 生成确定性的 ordered prompt plan，通常把稳定 definitions/dependencies 放前、patch/test/stack trace 放后，以提高大段 exact-prefix reuse。
- Debugger 的最大 exact reuse 粒度可以是包含多个函数和测试的 5k–20k token bundle，不受单函数上限约束。
- Git 的核心角色是 snapshot identity、visibility、branch/worktree isolation、diff event 和 unchanged-page alias，不负责计算 KV 或相关性检索。
- 最小六函数示例确认：函数源码未变时 canonical base 可跨 commit alias；包含旧 changed function 的 prompt variants 必须按 causal prefix 失效或验证；旧 session 继续读取旧 snapshot pages。
- 固定 workflow 进一步决定 cache priority：Architect 后预取 Coder variants，Coder patch 后保留 dependency cone 并准备 Debugger bundle，Debugger 失败时条件返回 Coder。
- Architect/Coder/Debugger 的 exact bundles 默认 stage-specific，不能因代码相同直接 raw-copy；共享发生在 canonical artifact base 与 KVCOMM context reconstruction 层。
- hot stage exact bundles 惰性生成；role/context offsets 或 anchors 有上限。faithful KVCOMM offset 可能接近完整 K/V 大小，不能默认认为 residual 很小。
- Canonical Base KV 不是特殊编码，而是在固定 model/template/prompt/token/position reference 下真实 prefill 得到的普通 K/V tensors；它只在该 canonical context 下 exact，在其他 role/context 中只能作为 KVCOMM reconstruction base。
- 已审查远程分支 `integration/coding-aware-prefetch`，当前头为 `d4a7ec132d80597c7b55a562beb8432e804ab127`，提交主题为 `merge: document middle-KV handoff`。
- 该分支从 upstream 基线 `3343a794` 开始，新增 29 个文件、约 3,324 行；核心是 policy-neutral `KVCOMM` shared data plane、coding-aware reuse policy、prefix/middle-KV prefetch coordinator、Radix transfer/residency adapter 和 middle-KV handoff API。
- shared store 已加入完整 token hash、model/cache identity、generation、residency、pin/lease、LRU eviction、stale-handle guard 和 backend resource disposer；transfer path 会在 stale、nonresident 或 token mismatch 时整 chunk dense fallback。
- coding-aware policy 目前只把调用者已分类的 `STABLE`/`CRITICAL` segment 转成 reuse plan；尚未接入真实 AST、SessionGraph、dependency signal 或 scheduler decision builder。
- 最新 middle-KV handoff 支持从请求设备槽导出 host payload、同步预取回 device、以 `PrefetchTicket` 管理 lease、暴露 device indices，并把 handle 交给共享 `KVReusePlan` 消费。
- SGLang 生产接入仍很薄：只在 `CacheInitParams` 增加可选配置，并在 `RadixCache` 初始化/reset `KVCommManager`；scheduler/request admission 尚未自动调用 coding policy、prefetch 或 middle-KV API。
- 该分支官方分类为 `INTERFACE_COMPLETE / SERVER_CANARY_PENDING`；CPU/fake allocator 与确定性 tensor backend 有测试，但尚无真实模型服务器 GPU canary、真实 HiCache storage payload、异步 CUDA transfer 或端到端性能数据。
- 该分支不是 faithful KVCOMM `2510.12872`：未实现 canonical base、context-dependent `ΔK/ΔV`、multi-anchor interpolation、entropy/shareability gate 和论文完整算法。它适合作为共享搬运、身份和生命周期骨架的候选 donor，不应被描述为 KVCOMM 完整复现或 production-ready coding-aware prefetch。

## 当前计划

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 1 | 初始化版本库和跨会话文档机制 | 已完成 |
| 2 | 接续历史 SGLang、KVFlow、KVCOMM 与 AST 结构距离研究 | 已完成 |
| 3 | 形成超大 Codebase 预计算、索引、CPU/GPU 分层与三阶段 workflow 的统一设计 | 已完成 |
| 4 | 审查 Yu Guofan / AgentTemplateKV 最近两个月分支和其他论文工作 | 已完成 |
| 5 | 调研 AST-indexed whole-codebase KV Cache 直接先例并评估 novelty | 已完成 |
| 5B | 按 2024/2025/2026 调研 Git/repository-version-aware KV Cache | 已完成 |
| 5C | 评估 Vast.ai RTX PRO 6000 hosting、容器兼容、账号连接和实验收益 | 已完成 |
| 5D | 七个保留分段复核 2025-01-01 至 2026-07-15 的 codebase source-version-aware KV prior art | 已完成 |
| 5E | 评估在历史修改版 SGLang 上 faithful reproduction KVCOMM | 已完成 |
| 5F | 审查 `integration/coding-aware-prefetch` 最新实现、接线和成熟度 | 已完成 |
| 6 | Docker 内同步 upstream main、创建 `latest-main`、迁移 SM75 patch 并跑通最新 SGLang | 待开始 |
| 7 | 建立 sequential high-pressure TTFT harness、approximate KV 独立数据面和统一 telemetry | 待开始 |
| 8 | 并列实现 raw+RoPE、EPIC fixed-k、selective repair、KVCOMM anchor 与 hardware-aware selector | 待开始 |
| 9 | 比较 LRU、KVFlow steps-only、Belady oracle、recovery-aware value-density 和 hierarchical policy | 待开始 |
| 10 | 本地筛选有效组合，并在 RTX PRO 6000 上按同一口径复测 | 待开始 |
| 11 | 仅在 sequential 方法有效后研究并发 workflow、status-aware scheduling 和并发 prefetch | 后置 |

## 可共享工作思路

- 当前目录原本为空，且尚无产品需求；先建立持久化上下文和协作约束，可以避免过早选择技术栈或生成错误代码骨架。
- `PROJECT.md` 保存当前有效事实，`TRACKING.md` 保留时间线，`HANDOFF.md` 保存下一会话立即需要的状态，三者职责分离以减少信息漂移。
- 项目交流和 prototype 代码统一落在 `ccdd2023/sglang`；执行 GitHub 操作前必须确认使用 `ccdd2023` 身份，避免误用系统当前默认账号。
- KVFlow 负责利用固定 workflow 的未来执行距离进行 cache priority、eviction 和 CPU→GPU 调度；KVCOMM 负责解决相同代码段在不同 agent role/prefix 下无法 exact-prefix 命中的跨上下文复用问题。
- 超大 Codebase 不能作为单一连续 prompt 直接预计算；需要按代码 artifact/AST span 分段，在 canonical context 下生成 base KV，并建立内容、结构、位置、模型与缓存位置索引。
- AST 适合作为分段、索引和 anchor gating 的结构信号，不应直接替代 KVCOMM 的 embedding-distance 判据。
- runtime 应始终按 `exact cache -> verified KVCOMM reconstruction -> dense fallback` 的顺序执行。
- 单 sequential workflow 的 prefetch 必须保守，不能为了加载低价值 cache 强制驱逐更紧急的 cache。
- 论文机制、历史实现结果和本项目新增方案必须明确区分，避免把新的 CPU 分层、AST 索引或 Coding Agent 设计误写成论文原结论。
- AgentTemplateKV 最新分支有大量有价值的实验资产，但核心路径已混合 exact copy、k-NN、AST chunk、R32 和几十个环境开关，不适合作为 faithful KVCOMM 的实现基线。
- L2 whole-slot exact path 有实际 token equality guard；C2 AST chunk path 则使用截断 normalized signature + byte range，必须明确区分，不能笼统评价为同一个安全级别。
- 任何名为 exact-content 的 production path 都必须同时满足完整 token equality、完整内容 hash 和完整 model/tokenizer/template fingerprint，不能使用截断文本签名。
- CacheBlend 和 EPIC 的论文机制不能用固定 leading fraction 或 head-only Key rotation 代替；近似实现只能标为 inspired/ablation。
- 论文 artifact 必须能够从提交的数据源重新生成；只提交图表和 data manifest 而缺少 manifest 指向的 compact CSV/JSON 不足以支持复现。
- CodeComp 和 FCGraft 已分别占据 structure-aware KV retention 与 function-level KV object/lifecycle；AST stable ID、函数 KV 对象化或 CPU/GPU 分层不能单独作为论文核心贡献。
- 整个仓库应建立完整 logical index，但 physical KV 默认按热度惰性物化；不要预计算并长期保存所有粒度、所有 role 和所有 context 的完整 BF16 KV。
- prefix 改变会改变 suffix hidden states，因此 suffix K/V 都可能变化；RoPE relocation 只修位置，不修 context-induced representation offset。
- 新的系统 thesis 是 versioned causal KV materialized views：源码/依赖增量失效、logical-to-physical page consistency、calibrated reconstruction、dense fallback 与 artifact-level cache planning。
- 当前最强系统贡献顺序为：dependency invalidation、持久 artifact/page lifecycle、可标定 cross-role reconstruction；structure-conditioned reconstruction 只有在实测胜过 KVCOMM/FCGraft/MEPIC/MiniPIC 后才升级为算法贡献。
- AST 只是一种可选的结构信号和实现工具，不是项目的主要研究切入点；优先研究 Git/repository version、incremental update、cache consistency 和 serving lifecycle 等系统机制。
- “Version-aware KV”必须严格区分三种不同概念：
  - token/content hash 命中；
  - runtime checkpoint version/epoch；
  - repository source-version coherence。
- PIE、Leyline 等证明 mutable prompt repair 已不是空白；Irminsul/MEPIC 证明 content-addressed objects 已不是空白；Concordia 证明 runtime versioned checkpoint 已不是空白。
- 仍成立的核心系统空白是：Git/worktree snapshot identity、source/dependency invalidation、cross-version exact alias/repair、MVCC-like branch isolation、physical tier coherence 和 stale audit 的统一闭环。
- 下一阶段应优先验证：真实 commit trace 中的 cross-version hot-artifact reuse、dependency invalidation 粒度，以及 version catalog/H2D/repair overhead。
- 实验基础设施采用 local-control/remote-execution：本地是唯一 Git、凭据、文档和长期结果控制面；Vast instance 只拉取固定 commit、运行实验并返回 compact artifacts。
- Vast 首轮使用 on-demand Secure Cloud/Verified RTX PRO 6000 S；至少 128GB host RAM，HiCache 优先 256GB，disk 建议 200GB 起，并记录 offer/machine/PCIe/disk/network 以控制 marketplace 异构性。
- 首轮可用官方 SGLang `dev` 验证 SM120；正式实验必须使用 immutable release 或与 Git SHA 绑定的自定义镜像，不能依赖 mutable `dev` tag。
- Vast API key 只保留本地，并使用最小权限 scoped key；远端不放 GitHub 写凭据，SGLang API 默认通过 SSH tunnel 访问。
- 2025–2026 保留分段以首次公开日期划分而非 revision 日期划分，防止跨年 revision 重复计数和时间空洞。
- 任何潜在 A 类候选都不能直接采用代理结论；最终必须由主会话阅读全文并核对 source-version semantics 是否真正进入 attention-KV coherence。

## 决策记录

| 编号 | 日期 | 决策 | 原因 |
| --- | --- | --- | --- |
| D-001 | 2026-07-12 | 使用 `PROJECT.md` 作为项目固定事实来源 | 集中维护更新、计划、进度和决策 |
| D-002 | 2026-07-12 | `TRACKING.md` 采用只追加的时间线 | 保留讨论与执行过程，避免覆盖历史 |
| D-003 | 2026-07-12 | `HANDOFF.md` 维护最新快照，而非完整历史 | 让新会话可以快速恢复上下文 |
| D-004 | 2026-07-12 | 使用 `.github/copilot-instructions.md` 固化协作规则 | 让后续 Copilot 会话自动获得维护要求 |
| D-005 | 2026-07-12 | 使用 `ccdd2023/sglang` 进行项目交流和 prototype 实现 | 用户指定该仓库为统一协作位置 |
| D-006 | 2026-07-12 | 目标仓库操作必须显式使用 `ccdd2023` 账号 | 系统当前默认 GitHub 账号可能不是目标账号 |
| D-007 | 2026-07-12 | 以 KVFlow `2507.07400` 与 KVCOMM `2510.12872` 作为当前核心论文 | 与历史研究文件及用户描述一致 |
| D-008 | 2026-07-12 | 固定 Coding Agent workflow 为 `Architect -> Coder -> Debugger` | 用户明确该流程不变 |
| D-009 | 2026-07-12 | 以 Codebase artifact/AST span 为预计算和索引单元 | 整体 Codebase 超出显存与上下文容量，必须分段管理 |
| D-010 | 2026-07-12 | 将“可变编码”定义为 base KV + context offset + RoPE relocation 的组合概念 | KVCOMM 原文没有名为 Variable Encoding 的格式 |
| D-011 | 2026-07-12 | cache 使用顺序为 exact、受控近似、dense fallback | 优先保证 Coding Agent 的正确性 |
| D-012 | 2026-07-12 | AST 作为结构索引和辅助 gating，不替代 embedding distance | 历史实验显示 AST 信号互补但相关性有限 |
| D-013 | 2026-07-12 | 不把 `fix/placeholder-pool-activation` 作为 KVCOMM 实现基线 | 缺失 base/offset/interpolation 等核心机制，并存在 cache lifecycle、位置和离线 KV correctness blocker |
| D-014 | 2026-07-12 | 从接近 upstream 的干净分支或 `feature/workflow-priority` 重建 | 降低 7,143 行 `radix_cache.py` 与大量实验开关带来的耦合和回归风险 |
| D-015 | 2026-07-12 | 只选择性移植 benchmark、telemetry、AST/HKVD 与 RoPE helper | 这些模块有独立价值，且不要求继承错误的 KVCOMM 核心 |
| D-016 | 2026-07-12 | exact reuse 必须使用 full token equality、full-content hash 和完整 fingerprint | 防止截断签名碰撞、tokenizer drift、模型/模板不兼容和错误 KV 命中 |
| D-017 | 2026-07-12 | CacheBlend、EPIC 等第三方机制必须忠实实现或明确标注为 inspired | 避免把固定 FRAC、head-only RoPE 等近似误写成论文复刻 |
| D-018 | 2026-07-12 | 不再主张首个 AST-aware、function-level 或 code-specific hierarchical KV cache | CodeComp、FCGraft、MEPIC 和 MiniPIC 已覆盖这些 broad claims |
| D-019 | 2026-07-12 | 将论文主线改为 versioned causal KV materialized views | 真正空白在 evolving repository 的一致性、失效、cross-role reconstruction 和 artifact-level planning 闭环 |
| D-020 | 2026-07-12 | 全库只保证 logical index 完整，physical KV 按 hotset 惰性物化 | 避免全库、多粒度、多 context KV 的存储爆炸 |
| D-021 | 2026-07-12 | 先测 reuse、context variance、H2D/recompute break-even 和 edit churn，再扩大实现 | 用可证伪数据判断系统是否值得继续 |
| D-022 | 2026-07-12 | structure-conditioned reconstruction 是条件性算法方向，不是已成立贡献 | 必须在真实 repository/role/order 变化下实测优于强 baseline |
| D-023 | 2026-07-13 | AST 不作为主要研究切入点 | 用户明确核心关注是 version/lifecycle/consistency 等系统 harnessing |
| D-024 | 2026-07-13 | 使用三个 GPT-5.6 Sol Max research agent 分别覆盖 2024、2025、2026 | 用户明确要求统一使用 Sol Max，并以 alphaXiv/arXiv MCP 为主要全文证据来源 |
| D-025 | 2026-07-13 | 将严格 A 类定义为 repository/source version 直接控制 attention-KV identity、validity 或 coherence | 避免把普通 token hash、runtime checkpoint epoch 或 Git-aware RAG 误计为直接先例 |
| D-026 | 2026-07-13 | 暂以 RepoKV-MVCC / Versioned Causal KV Materialized Views 作为系统 thesis | 当前空白在 source snapshot、dependency invalidation、cross-version reuse 和 tier coherence 的统一协议 |
| D-027 | 2026-07-13 | 实施前先回放真实 Git commit/patch traces | 必须证明 cross-version hot-artifact reuse 足以摊销 catalog、repair 和数据移动开销 |
| D-028 | 2026-07-13 | 采用本地控制面 + Vast.ai 按需 GPU 执行面的混合工作流 | 本地 8GB 适合开发但不足以验证 7B/8B、长上下文和大 KV tier；Vast marketplace 又不适合作为唯一开发与存储环境 |
| D-029 | 2026-07-13 | Vast 首轮使用官方 CUDA 12.9+/SM120 SGLang 镜像，正式实验使用 Git-SHA 绑定镜像 | 先降低接入风险，再保证 prototype 代码、native kernel 和实验环境可复现 |
| D-030 | 2026-07-13 | Vast 凭据仅驻留本地，远端只读拉取代码并通过 SSH tunnel 暴露服务 | provider 可技术性访问 host 文件，必须最小化凭据和公网暴露 |
| D-031 | 2026-07-15 | 用十个连续时间段代理重新复核 source-version-aware attention-KV prior art | 用户对 novelty 仍有合理担忧，需要以更细时间粒度、独立负搜索和最新 7/30/90 天覆盖降低漏检风险 |
| D-032 | 2026-07-15 | 论文按首次公开日期唯一归属，A 类候选必须主会话二次全文复核 | 避免把 revision、venue 版本或邻近 Git/RAG 工作误算为多个直接先例 |
| D-033 | 2026-07-15 | 最终分段复核只保留 2025-01-01 至 2026-07-15 | 用户决定取消 2025 年之前的 subagent；2024 分段结果不得进入最终 verdict |
| D-034 | 2026-07-15 | module/class 仅作逻辑 view，物理 KV 使用非重叠 canonical artifacts | 避免单一全库 prompt、嵌套 KV 重复存储和粗粒度全模块失效 |
| D-035 | 2026-07-15 | 依赖图只作为保守失效 prior，不作为 exactness 证明 | attention KV 还依赖 causal context、位置、模型和模板，必须由 fingerprint/probe/fallback 保证正确性 |
| D-036 | 2026-07-15 | 采用 repository-version-aware attention-KV coherence 作为最终安全 claim | 七分段 105 篇候选均无 A 类，但各基础组件已有强先例，必须避免宽泛首创叙事 |
| D-037 | 2026-07-15 | 仓库 KV bootstrap 从目标 fixed SHA 开始，不从 genesis commit 开始 | 只需服务当前及未来 snapshots；全历史预计算成本高且没有运行必要性 |
| D-038 | 2026-07-15 | logical catalog 全库覆盖，physical KV 仅按 hotset 稀疏物化 | 实际 K/V tensor 体积很大，不能为所有 artifact/context/version 组合长期保存完整副本 |
| D-039 | 2026-07-15 | 分离 source dependency graph 与 prompt causal graph | 静态代码依赖不等于 Transformer causal context，exact invalidation 必须基于真实 prompt 前序 |
| D-040 | 2026-07-15 | exact cache 保存完整连续 bundle/prefix，函数只作为主要 artifact/invalidation 单元 | 使 Debugger 等大上下文可复用跨多个函数的大段 KV |
| D-041 | 2026-07-15 | 采用 exact bundle、canonical base、bounded residual/anchor 三层存储 | 避免为每个函数和所有 context 保存两份或多份完整 KV |
| D-042 | 2026-07-15 | Prompt Compiler 采用稳定、依赖约束和 volatility-aware ordering | 增大最长 exact prefix，并把 patch、日志等动态信息放在尾部 |
| D-043 | 2026-07-17 | 将 `integration/coding-aware-prefetch` 定位为 interface-complete shared-data-plane donor，而非 faithful KVCOMM 或 production baseline | identity、transfer、lease 和 middle-KV API 已成形，但 scheduler、GPU server、HiCache 与 KVCOMM base/delta/anchor 算法仍未完成 |
| D-044 | 2026-07-21 | 当前实验只研究跨 context 有损 KV 恢复与 scheduling，不研究 AST、自动分段、label 或 indexing | 用户明确把这些问题交给其他合作者，当前任务只优化恢复和调度 |
| D-045 | 2026-07-21 | “有损”定义为不做完整目标-context prefill 的近似 KV 恢复，不包括量化、低比特或普通 KV pruning | 用户纠正了此前把有损误解为压缩/量化的错误 |
| D-046 | 2026-07-21 | 客户端 TTFT 是唯一性能主目标；只要求请求返回首 token | 用户明确不关心正确率，本轮不以语义、代码质量或输出一致性筛选方案 |
| D-047 | 2026-07-21 | 第一阶段只做 sequential workflow，并发明确后置 | 先隔离恢复与 eviction 的因果关系，找到有效方法后再增加并发复杂度 |
| D-048 | 2026-07-21 | 同时比较多条恢复路径与多种 scheduler，并加入 synthetic trace 的 Belady oracle 上界 | 当前不知道哪种方法最好，不能押注单一路径 |
| D-049 | 2026-07-21 | fork main fast-forward 到 upstream 后创建 `latest-main`，所有 Git/编辑/build/test 均在 Docker 内完成 | 用户明确指定同步和分支流程，并禁止宿主机构建或下载更新 |
| D-050 | 2026-07-21 | approximate KV store 与 exact Radix 隔离，完整覆盖到倒数第二个 prompt token，最后 token真实 forward | 保证机械连续性、首 token生成和实验稳定性，同时避免近似 KV 污染 exact cache |

## 更新记录

### 2026-07-12T17:59:01-07:00

- 完成空项目初始化。
- 固化中文交流限制。
- 建立项目主文档、讨论追踪和会话交接机制。
- 下一阶段等待定义项目需求。

### 2026-07-12T18:03:17-07:00

- 将 `https://github.com/ccdd2023/sglang` 确认为项目交流和 prototype 代码实现仓库。
- 使用系统已有的 `ccdd2023` 凭据完成身份与权限验证。
- 确认 `ccdd2023` 对目标仓库具有 `ADMIN` 权限，仓库默认分支为 `main`。
- 固化后续 GitHub 操作必须显式选择 `ccdd2023` 账号的要求。

### 2026-07-12T18:09:54-07:00

- 接入历史研究工作区并定位本机 Docker 版 SGLang、KVFlow 移植分支和已有 benchmark 结果。
- 确认 `kvflow-sglang` 的 `feature/workflow-priority` 与远程仓库同名分支同步。
- 通过 alphaXiv 获取 KVFlow 与 KVCOMM，并下载论文 PDF。
- 启动两个独立 subagent 深入研究两篇论文。
- 将项目目标明确为：在 SGLang 上构建面向超大 Codebase Coding Agent 的分层、跨上下文 KV Cache 复用系统。

### 2026-07-12T18:18:37-07:00

- 两个论文 subagent 完成研究并返回详细报告。
- 判定相关 KVCOMM 为 arXiv `2510.12872`，排除同名但机制不同的 `2510.03346`。
- 明确 KVFlow、KVCOMM、AST index、HiCache 和三阶段 workflow 的职责边界。
- 形成 `research/RESEARCH_SYNTHESIS.md`，记录统一系统架构、原始论文边界、prototype 路线、指标和风险。
- 下一阶段是先在 SGLang 上完成 KVCOMM 的最小原样复刻。

### 2026-07-12T19:40:52-07:00

- 确认 Yu Guofan 对应 `flaminyu`，并核对五个研究分支的线性演进、阶段提交数和 author 归属边界。
- 两个独立 subagent 分别完成 KVCOMM 实现审查和其他论文追踪；主会话逐项复核关键代码、实验报告和论文稿。
- 判定该研究线实现的是 exact/raw KV copy、RoPE shift、heuristic gate、AST chunk 和 selective recompute 的组合，而不是 KVCOMM 的 base+offset+multi-anchor reconstruction。
- 识别 lock off-by-one、stale placeholder pool、非连续 prefix、截断签名碰撞、Unicode byte offset、离线 token 边界、loader fingerprint、host copy failure 和 True CacheBlend position 等 correctness blocker。
- 独立验证 C2 截断签名可碰撞、Unicode byte offset 错位、context gate 缺表时拒绝全部 exact matches。
- 审计论文 artifact：`paper/data_manifest.json` 的 27 个 source entries 中只有 5 个存在于分支；图表生成脚本无法从当前提交重跑。
- 区分其他论文的实际状态：CacheBlend/EPIC 直接影响但未忠实复刻，KVFlow 是架构来源，Prompt Cache/LMCache 是概念或部署参照，DroidSpeak 仅调研。
- 创建 `research/YU_GUOFAN_BRANCH_REVIEW.md`。
- 下一阶段改为从干净 SGLang 基线重建 faithful KVCOMM，而不是继续扩展最新 AgentTemplateKV 分支。

### 2026-07-12T23:04:43-07:00

- 启动 GPT-5.6 Sol Max 专职 arXiv research agent，调查 AST/程序结构直接标注或索引 whole-codebase KV Cache 的先例。
- 并行启动 GPT-5.6 Sol Max、Claude Opus 4.8 Max、Claude Opus 4.6 Max 和 Gemini 3.1 Pro 四个独立评估代理。
- 四个评估代理统一分析：
  - “先 index，再改变中间代码段，并以 priority 在 SGLang CPU/GPU tier 中浮现”的 novelty；
  - closest prior art 与已有组件拼接边界；
  - causal correctness、RoPE、cross-chunk attention、storage/invalidation 和 H2D 风险；
  - `Architect -> Coder -> Debugger` 的 workflow-specific 新机制；
  - paper thesis、prototype 路线和可证伪实验。
- 研究任务在后台运行，不阻塞当前会话；完成后将创建统一 consolidated report 并更新所有交接文档。

### 2026-07-12T23:36:13-07:00

- 五个后台代理全部完成；四个评估代理在收到专项文献结果后又提交了二次修正。
- 发现 A 类直接先例 CodeComp `2604.10235` 和 FCGraft `2606.13097`，以及强邻近 MEPIC `2512.16822`、MiniPIC `2606.13126`。
- 撤回 broad “首个 AST-aware/function-level/code-specific hierarchical KV cache”叙事。
- 将原始方案 novelty 下调为约 `2/5`；收窄后的系统论文上限保守估计为 `3.3–3.6/5`。
- 明确因果边界：role/prefix 改变会导致 suffix K/V 均变化，RoPE relocation 不能修复 representation offset，L1 exact 条件在真实 workflow 中接近空集。
- 创建 `research/AST_INDEXED_KV_CACHE_NOVELTY_REPORT.md`。
- 下一阶段先做 trace instrumentation 和 faithful KVCOMM，再建立 versioned artifact registry、dependency invalidation 和 calibrated reconstruction。

### 2026-07-13T01:13:56-07:00

- 将对 consolidated verdict 的 33 步详细解释完整落盘为 `research/AST_INDEXED_KV_CACHE_STEP_BY_STEP.md`。
- 保留 `AST_INDEXED_KV_CACHE_NOVELTY_REPORT.md` 作为正式研究报告；新文件承担教学式推导、新会话恢复和逐项讨论入口。
- 更新 README、研究综合、项目主文档和 handoff 中的文档索引。

### 2026-07-13T02:14:15-07:00

- 用户澄清 AST 不是主要研究切入点，重点应放在 Git/repository version、incremental invalidation、cross-version reuse 和 serving lifecycle 等系统 harnessing。
- 最初误按 Sol/Terra/Luna 分配；用户随后明确要求三个代理均使用 GPT-5.6 Sol Max。
- 已重新启动三个全 Sol Max 年度代理，分别覆盖 2024、2025、2026，并强制以 alphaXiv/arXiv MCP 发现和全文证据为主。
- 早期代理结果只作为补充交叉检查，不作为最终主要证据。
- 待三份年度报告完成后，将创建 `research/GIT_CODEBASE_VERSIONED_KV_CACHE_PRIOR_ART.md` 并形成跨年份 direct-prior-art verdict。

### 2026-07-13T04:11:21-07:00

- 三个 GPT-5.6 Sol Max 同步代理完成 2024、2025、2026 年 version-aware attention-KV 调研。
- alphaXiv MCP 多次返回 HTTP 429；研究没有停止，改用 arXiv PDF/API、DBLP、venue 和官方代码核查，主会话另行抽查八篇关键论文全文。
- 三年严格 A 类均为 0。
- 确认已有能力被拆分在 PIE、Leyline、FCGraft、Irminsul/MEPIC、Streaming Knowledge Compilation、Code Isn't Memory 和 Concordia 等不同系统中。
- 创建 `research/GIT_CODEBASE_VERSIONED_KV_CACHE_PRIOR_ART.md`。
- 推荐 thesis 收窄为 RepoKV-MVCC：repository snapshot isolation、source/dependency invalidation、cross-version reuse/rematerialization 与 physical-tier coherence。

### 2026-07-13T08:58:42-07:00

- 调研 Vast.ai 的 Docker hosting、SSH/CLI/API key、storage、networking、Secure Cloud 和实例生命周期。
- 审计本地 SGLang 镜像与源码：当前官方 dev image 含 CUDA 12.9、SM120 kernels 和 SSH；源码已有 RTX PRO 6000 特定路径。
- 确认现有 `docker run` 脚本不能在 Vast standard Docker instance 内原样执行，正式使用应转为 template/on-start 或预构建 registry image。
- 识别现有 DeepEP build arch list 缺少 SM120，暂不把 RTX PRO 6000 用于该路径。
- 决定采用本地控制面 + Vast 短时执行面的混合 workflow，并创建 `research/VASTAI_RTX_PRO_6000_WORKFLOW.md`。

### 2026-07-15T18:47:34-07:00

- 用户取消 2025 年之前的分段调研；已向 `version-scan-01` 至 `version-scan-03` 发送停止指令，并明确排除其结果。
- `version-scan-04` 的有效范围收缩为 2025-01-01 至 2025-01-06，保留 `version-scan-05` 至 `version-scan-10`。
- `version-scan-05` 同时完成：11 篇候选，A/B/C/D=`0/3/6/2`，严格 A 类为 0。
- 当前最终范围为 2025-01-01 至 2026-07-15，七个保留分段完成 4/7。

### 2026-07-15T18:52:54-07:00

- 核对 2026 年覆盖状态：2026-01-13 至 2026-07-15 已完成，2026-01-01 至 2026-01-12 尚在 `version-scan-08` 的负责范围内。
- 因此当前不能声称 2026 年全部检索完成；需等待第八段返回。

### 2026-07-15T18:56:10-07:00

- 用户要求暂时搁置当前分段调研，仅基于历史方案用两句话概括 novelty。
- 对外表述聚焦两个层次：现有缓存缺少 repository version/dependency/workflow 的统一一致性管理；本项目以 versioned causal KV materialized views 实现 artifact KV 的跨版本复用、失效、重算和分层调度。

### 2026-07-15T18:56:54-07:00

- 用户纠正交付格式：需要两段简短文字，而不是两个单句。
- 保持“为什么有 novelty”和“novelty 是什么”各占一段，并增加必要解释但不展开成长报告。

### 2026-07-15T18:58:28-07:00

- `version-scan-08` 完成 2025-10-13 至 2026-01-12：12 篇候选，A/B/C/D=`0/1/10/1`，严格 A 类为 0。
- 由此 2026-01-01 至 2026-07-15 已完整覆盖；此前“尚缺 2026 年前 12 天”的状态已被新结果关闭。
- 七个保留分段当前完成 5/7，剩余 `version-scan-04` 和 `version-scan-07`。

### 2026-07-15T18:59:42-07:00

- `version-scan-02`、`version-scan-03` 已明确确认停止。
- 两个代理此前生成的 2024 memo 均按用户要求排除，不进入最终计数、矩阵或 novelty verdict；`version-scan-01` 的停止确认仍待返回。

### 2026-07-15T19:03:06-07:00

- 用户将 presentation 文案格式明确为两段：第一段同时概括 novelty 与 main idea，第二段概括问题和 concerns。
- 第一段应突出 repository source lineage/dependency 与 attention-KV identity、validity、invalidation、rematerialization 的直接耦合，而不是把 AST index、CPU offload 或 workflow priority 单独当作 novelty。
- 第二段应明确 causal correctness、跨 role/prefix 复用误差、依赖失效精度、metadata/H2D/storage 成本、branch/worktree 状态爆炸和 stale-cache 风险。

### 2026-07-15T19:04:13-07:00

- 用户要求在 presentation 文案后增加一段简短 prototype 路线。
- 最小 prototype 从干净 SGLang 基线开始，只实现 artifact 切分、source/dependency metadata、CPU KV registry、跨版本 exact reuse、变更失效和 dense fallback；KVCOMM reconstruction 与 KVFlow priority 后置。
- 首轮用真实 Git trace 测量 hit rate、TTFT、H2D、显存占用和输出正确性。

### 2026-07-15T19:11:00-07:00

- 用户要求后台评估是否能在此前修改过的 SGLang 上完整、忠实复现 KVCOMM，以及难度和 blocking points。
- 已启动 `kvcomm-sglang-feasibility` GPT-5.6 Sol Max 后台代理。
- 代理将只读核查 KVCOMM 原文、`feature/workflow-priority`、`sglang-running`、AgentTemplateKV 历史分支和实际 SGLang 集成点。
- 最终交付包括功能性复现与论文级复现的分别结论、推荐基线、工作量、P0/P1/P2 blocker、代码复用矩阵、阶段验收和验证方案。

### 2026-07-15T19:37:48-07:00

- `version-scan-07` 完成 2025-07-12 至 2025-10-12 的检索。
- 21 篇候选，A/B/C/D=`0/5/14/2`，严格 A 类为 0。
- 最接近的 B 类为 KVCOMM、CacheClip、CIFLEX、SamKV、SemShareKV；最接近 Git/source 语义的是 RepoMem 与 LinkAnchor，但它们不保存 attention K/V。
- 七个保留分段完成 6/7，当前只缺 2025-01-01 至 2025-01-06。

### 2026-07-15T19:38:49-07:00

- `version-scan-01` 已确认停止，其 2024-01-01 至 2024-04-02 结果不纳入最终报告。
- 至此三个纯 2024 分段均已停止并排除。

### 2026-07-15T19:39:13-07:00

- 用户确认 whole-codebase prefill 是否应按模块和依赖分段，并追问依赖分析是否需要原创。
- 明确采用 logical hierarchy + non-overlapping canonical physical artifacts：module/class/file 用于组织，function/method/preamble/init block 用于实际 KV object。
- 依赖分析复用已有 parser、symbol/call/build/test graph；项目新增点是 source/dependency change 到 attention-KV coherence action 的协议，而不是重新发明静态分析。

### 2026-07-15T19:43:11-07:00

- 收到最后一个保留分段 `version-scan-04`：2025-01-01 至 2025-01-06 共 2 篇候选，A/B/C/D=`0/0/2/0`。
- 七个保留分段全部完成，共 105 篇主候选，A/B/C/D=`0/23/67/15`。
- 完成跨段归属、closest-prior-art matrix、最终 verdict、claim 边界和三段 presentation summary。
- `research/CODE_VERSION_KV_10_AGENT_PRIOR_ART_REPORT.md` 已成为本轮最终报告；当前阶段转向 KVCOMM SGLang 可行性评估和 prototype 准备。

### 2026-07-15T19:46:33-07:00

- 用户进一步确认是否只保存大量小 KV 并仅用 Git 索引。
- 明确采用多索引设计：结构切分、symbol/embedding 检索、Git 定版本、dependency 传播失效、context fingerprint 判定 exactness、physical index 定位 KV。
- artifact 通常是中等粒度函数或 module preamble；physical KV 稀疏物化，独立块不能在新因果顺序下直接拼接。
- 完成 `kvcomm-sglang-feasibility` 后台调研和主会话代码抽查。
- 创建 `research/KVCOMM_SGLANG_FEASIBILITY_REPORT.md`，记录可行性、推荐基线、难度、P0/P1/P2 blocker、复用矩阵、实施路线和验证方案。

### 2026-07-15T19:54:51-07:00

- 用户要求用一段话快速概括最终 literature review。
- 摘要应包含七分段、105 篇候选、A=0、已有能力分散在 mutable KV/persistent tier/Git memory 三类，以及安全 claim 与组合显而易见性风险。

### 2026-07-15T19:56:02-07:00

- 用户追问 attention KV 的实际含义、SGLang 大仓库如何切分和存储，以及是否从第一个 commit 开始。
- 明确 physical KV 是逐层逐 token 的 K/V tensor pages；以当前目标 SHA bootstrap，不回放整个 Git 历史。
- 全库建立 logical artifact catalog，physical KV 只物化 hot/relevant canonical artifacts；新版本通过 diff 做 page alias、局部失效、重算和 MVCC-like GC。

### 2026-07-15T20:03:39-07:00

- 用户确认 exact 与 KVCOMM 是否都以函数为 match 单位。
- 明确函数是默认 logical artifact 和常见 KVCOMM placeholder，但普通 exact cache 仍匹配完整连续 causal prefix，physical allocator 仍按 token pages 工作。
- 同一函数文本在不同前置上下文中不能自动 exact；KVCOMM 也包含 placeholder 后 fixed-prefix offset 与 whole-agent shareability gate。

### 2026-07-15T20:04:36-07:00

- 用户要求用极短文字概括 KVCOMM SGLang 可行性报告。
- 摘要聚焦：功能性复现可行、论文级复现受环境阻塞、推荐 clean baseline，以及 token/position/RoPE/provenance/lifecycle 是主要难点。

### 2026-07-15T22:11:55-07:00

- 用户要求正面说明 dependency graph、Prompt Compiler、KV artifact 存储模型和 Debugger 大段代码复用。
- 明确不采用“每个函数两份完整 KV”；函数是主要版本/失效单元，exact cache 是跨函数连续 bundle，KVCOMM store 保存 canonical base 与 bounded residual/anchor。
- 新增 source dependency graph 与 prompt causal graph 的职责分离，以及 stable/dependency/volatility-aware prompt ordering。

### 2026-07-15T22:29:13-07:00

- 用户要求用两个 Python 文件、每个三个函数的最小例子解释 Git 与固定 workflow 的作用。
- 示例以 `C0 -> dirty worktree W1 -> commit C1` 展示 unchanged base page alias、changed artifact rematerialization、dependent context verification、旧 snapshot isolation 和 Debugger bundle reuse。
- 明确 novelty 不是 Git diff，而是 Git/source events 到 attention-KV visibility、coherence、rematerialization 和 workflow-aware tier state 的闭环。

### 2026-07-15T23:03:06-07:00

- 用户确认三个 workflow role 是否各自维护独立 exact KV，以及能否直接跨角色复用。
- 明确 stage-specific exact bundle 默认隔离；跨角色通过 shared canonical base + KVCOMM offset/anchor reconstruction，而不是 raw-copy exact cache。
- 如 prompt template 共享完全相同前导 prefix，可在 role 分叉点前 exact reuse，但属于待验证的 template co-design。

### 2026-07-15T23:09:45-07:00

- 用户询问 canonical base KV 是否是真实 KV，以及它与普通 KV 的差异。
- 明确 canonical KV 是真实 prefill 产生的普通 K/V tensor；“canonical”只表示其 prompt、位置和 fingerprint 被固定为 reference。
- 它不是任意 runtime context 下的 exact KV，跨角色使用时必须经过 KVCOMM offset/RoPE reconstruction 或 dense fallback。

### 2026-07-17T22:35:36-07:00

- 审查 `ccdd2023/sglang` 的 `integration/coding-aware-prefetch`，确认当前头为 `d4a7ec132`，最新功能提交为 middle-KV handoff API，随后补充接口文档。
- 分支建立了共享 segment identity/store、generation/lease/resource lifecycle、copy-and-RoPE transfer、coding reuse plan、prefetch coordinator、Radix adapter 和 CPU 示例。
- 确认真实 SGLang 接线仅包含 `CacheInitParams.kvcomm_config`、`RadixCache.kvcomm` 初始化与 reset；scheduler、request admission、生产 allocator、HiCache storage 和异步 CUDA prefetch 尚未接通。
- 确认该实现不是 KVCOMM `2510.12872` 的 base/offset/multi-anchor 算法；当前成熟度应表述为 `INTERFACE_COMPLETE / SERVER_CANARY_PENDING`。
- 当前环境尝试运行目标 pytest 时在 collection 阶段因缺少 `pybase64` 依赖停止；这不是代码断言失败，也不能替代分支文档记录的 CPU/fake-backend 测试结果。

### 2026-07-17T22:59:25-07:00

- 将 `integration/coding-aware-prefetch` 概括为：搭建跨请求共享、搬运和复用 KV segment 的数据与接口骨架，并尝试让编码工作流提前规划和预取 prefix/middle KV。
- 明确未完成项仍包括真实 scheduler/request 自动接线、AST/依赖信号、HiCache 与异步 GPU 搬运、模型服务器验证，以及 KVCOMM 的 base/delta/anchor 重建算法。

### 2026-07-21T02:26:32-07:00

- 用户将当前实施重点切换为有损跨上下文 KV 恢复与 KVFlow-style priority/eviction 的 TTFT 加速实验。
- 明确有损不是量化，而是 raw reuse + RoPE、KVCOMM anchor/offset、EPIC/CacheBlend/Cache-Craft/CacheTune 风格局部 repair 等“不做完整目标-context prefill”的恢复方式。
- 明确正确率不作为指标；只要求请求稳定返回首 token，客户端 TTFT 为唯一主目标。
- 第一阶段只做 sequential `Architect -> Coder -> Debugger` 与 retry，使用 synthetic trace 制造 GPU KV oversubscription；并发后置。
- 确认同时尝试多条恢复路径、多种 eviction/scheduler 和 oracle next-use 上界，不能预设最佳方法。
- 只读核对 fork main 可 fast-forward 到 upstream 当前 SHA，远程尚无 `latest-main`。
- 完成历史 SM75、KVFlow、approximate KV 分支资产审计；该次原始计划现归档为`IMPLEMENTATION_PLAN_V1_ARCHIVED.md`。

### 2026-07-21T18:23:24-07:00

- 远程 `ccdd2023/sglang:latest-main` 已完成 SM75 source fallback、approximate KV 独立数据面、raw/EPIC/selective/KVCOMM recovery、hardware-aware selector、五种 eviction policy、sequential pressure harness、Radix lifecycle 接线和 whole-prefix raw-speed MVP。
- 最终远程头为 `f1e91b9cb80d9d4c036099fd0fa23a03400769e1`；GitHub-hosted guest source CI `29888035426` 成功。
- 所有代码编辑、测试和 GPU 运行均在 guest/container 内完成；未修改宿主机驱动、内核模块或 apt/dpkg 状态。
- 本地 SM75 使用只读 rootfs + tmpfs guest，Qwen3-0.6B server health/chat smoke 和真实 CUDA fallback/Radix copy-RoPE 测试均通过。
- 三档串行 end-to-end 结果：
  - `rho=0.840`：raw 相对 exact 回归 `1.10%`；
  - `rho=1.533`：raw 相对 exact TTFT 降低 `8.57%`，speedup `1.094x`；
  - `rho=1.888`：raw 相对 exact TTFT 降低 `7.63%`，speedup `1.083x`。
- 结果、原始 JSON、microbenchmark、concurrency simulation 和说明文档位于 `benchmark/approx_kv/results/sm75/`。
- 两个误并发提交到同一 server 的 run 已明确排除且未提交，没有删除其 guest 文件。
- RTX PRO 6000 scale runner 已实现为 `benchmark/approx_kv/run_scale_matrix.py`，但实际租用受 Vast API key/SSH guest 凭据缺失阻塞。

### 2026-07-21T20:32:35-07:00

- “完整构建官方 runtime image”指使用 SGLang 仓库自带 `docker/Dockerfile` 的 `runtime` target，从固定源码重新构建项目自定义镜像；不是 Docker 公司提供的官方镜像，也不等于直接使用 `lmsysorg/sglang:dev`。
- 历史分支的实际成功路径主要是 `lmsysorg/sglang:dev` + bind-mounted source + guest 内 editable install；虽然存在 full build script，但历史记录没有证明完整 runtime image 曾构建到底。
- immutable image 指以 Git SHA、固定依赖和 image digest 标识、不可被同名 tag 静默替换的镜像；当前 SM75 guest shim 不是最终 immutable image。
- `rho` 定义为 active reusable KV working set / GPU evictable KV capacity；`rho<1` 基本放得下，`rho>1` 必须淘汰或恢复。
- HiCache 是 SGLang 自带的 exact hierarchical cache；当前缺少的是 approximate base/anchor/object 与 HiCache controller 的真实接线，不是重新实现 HiCache。
- Phase 3 是 policy-neutral shared data plane，综合 SGLang lifecycle、KVFlow tier/state 思路、KVCOMM object需求和历史 `integration/coding-aware-prefetch` donor，不是某一篇论文的 faithful reproduction。
- 当前唯一 end-to-end 的 R0 whole-prefix speed-only 是激进速度上界，不是 KVCOMM；KVCOMM 对应 R3，尚未完成 faithful server end-to-end。
- Phase 5 当前完成 policy代码和simulation，但未修改真实 Radix/HiCache eviction；原因是 upstream request priority 语义冲突、tiering未接、单一 source trace无法区分S1-S4，以及先隔离 recovery收益。
- Phase 6 是 recovery axis × scheduler axis 的组合实验，不属于单篇论文：B0/B1/B2来自SGLang，B3来自KVFlow，R3才对应KVCOMM。

### 2026-07-21T20:50:25-07:00

- 用户指出 Phase 4/5 的完成口径错误：planner、policy和microbenchmark完成不能等同于server end-to-end完成。
- 已将 R1 EPIC、R2 selective、R3 KVCOMM、hardware selector、真实scheduler和完整screening重新标记为进行中/待完成；R0 raw仍是唯一端到端恢复路径。
- 修正后的实验顺序：
  1. 固定LRU/GPU-only，把R0-R3全部做成相同server path并比较recovery TTFT；
  2. 固定前两条recovery，在多对象synthetic trace上扫描 `rho≈1/1.5/2/3`，真实比较S0-S4；
  3. 最后加入HiCache与P0-P3 prefetch。
- 高压力本身不足以区分eviction policy；必须同时存在多个不同大小、不同next-use distance和不同recovery cost的cache对象。单一巨大对象即使 `rho=3` 也不能形成有意义的victim选择。
- 当前SM75三档数据仅证明R0 feasibility与pressure threshold，不足以完成Phase 4算法比较或Phase 5 scheduler结论。

## 2026-07-21T22:19:43-07:00 Phase 1 严格门禁完成

- Phase 1 实验分支当前头为 `experiment/phase1-image@dc09064ab`；镜像对应源码 SHA 为 `5a0fd2606bb62c6bcca004a4b2784ace745a580a`。
- 标准 GitHub runner 无法本地展开大型 SGLang base image，两个普通 Docker build 均以 exit code `102` 失败。最终改用固定版本 `crane v0.20.3` 做 registry-to-registry OCI 流式组装，只追加小型 wrapper/Transformers layer，不在 runner 解包 base。
- SM75 compatibility image：
  - `ghcr.io/ccdd2023/sglang:sm75-5a0fd2606bb62c6bcca004a4b2784ace745a580a`
  - digest `sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781`
  - CI run `29892292070`
  - 本地只读 GPU container 已验证 RTX 2080 SUPER/SM75、PyTorch `2.9.1+cu129`、Transformers `5.12.1`、`sgl_kernel 0.3.21`、activation/RMSNorm native fallback、Qwen3-0.6B health/model-info/1-token chat。
- SM80/SM120 runtime image：
  - `ghcr.io/ccdd2023/sglang:runtime-5a0fd2606bb62c6bcca004a4b2784ace745a580a`
  - digest `sha256:2e36099165cedb0d328c98ee6c37f88c7c626d1a953a35de28748d1aa6183482`
  - base 为固定的 official `v0.5.15.post1-cu130-runtime` digest
  - CI run `29892292080`
  - 只读 container 静态验证 PyTorch `2.11.0+cu130`、CUDA `13.0`、Transformers `5.12.1`、`sglang-kernel 0.4.4`、SM100 binary 和 SM80/SM120 native gate；真实 SM120 GPU smoke 等 RTX PRO 6000。
- 正式 manifest 位于 `docker/phase1-image-manifest.json`，记录 base/image digest、依赖、CI、验证范围和 host safety。
- 没有在 host 构建镜像，也没有修改 driver、DKMS、kernel module；源码保存在 host worktree，只读挂载进 container。
- Phase 1 七项验收全部通过。严格门禁已切换到 Phase 2；Phase 3–5 仍不得提前开始。

## 2026-07-22T00:03:07-07:00 Phase 2 严格门禁完成

- Phase 2 分支当前头为 `experiment/phase2-benchmark@05bb93bda`；实际矩阵 runner SHA 为 `333ebb65710a629ee8f859a7182db5f471c3e38c`。
- 新增 tokenizer-calibrated 24-object synthetic catalog，目标 reusable prefix 为 512/1024/2048/4096 tokens，并使用 chat-template token LCP 与 unique-token trie计算 reusable pressure。
- 固定 sequential trace 包含 fill、`Architect -> Coder -> Debugger`、cold filler、Coder/Debugger retry、branch/fan-out、replay 和 hot tail；三个固定 probe objects 在所有 rho 中保持一致。
- 每个配置使用独立 server restart；每个 restart 内随机化五个 pressure point 顺序。warmup使用不同 cache salt，随后 flush、health gauge refresh、clean baseline、measured trace、final flush/reset。
- 最终环境：SM75 image digest `sha256:0be6e16e...`、Qwen3-0.6B revision `c1899de...`、`mem_fraction_static=0.35`、实际 usable KV capacity `13,130` tokens。
- 24 个对象的 server-side cold/variant/repeat 校准全部通过；observed variant `cached_tokens` 与 expected token-LCP 的误差为 0。
- 5 个 pressure point × 3 独立 restart 共 15 个 run、471 个 measured requests，完成率 100%；所有 clean baseline、idle pool 和 final reset invariant 均通过。
- 实际 reusable rho 为 `0.813/1.007/1.514/2.017/3.023`；estimated physical rho 为 `0.870/1.067/1.569/2.094/3.152`。
- `rho=0.813` 三次均 0 eviction；从 `rho=1.007` 开始稳定出现 eviction，三次 token count 完全一致，最高 `rho=3.023` 为 `96,642` evicted tokens。
- 固定 probe 的 TTFT p50 在低压约 `147–149ms`，进入 eviction threshold 后约 `278–281ms`；Phase 2只证明pressure harness可稳定区分无eviction/有eviction，不代表任何有损恢复或scheduler收益。
- boundary-only 与 per-request metrics scrape诊断在 `rho=1.5` 的 probe p50 相差 `-0.43%`，evicted tokens一致；正式结果仍使用boundary-only。
- 离线trace validator确认在 `rho=1/1.5/2/3` 上 LRU、Belady和synthetic value-density的victim序列均不同；synthetic cost只作trace metadata，不是实测恢复成本。
- compact结果位于 `benchmark/approx_kv/results/phase2/sm75-summary.json`；host raw结果目录为 `/home/chris/Workspaces/kvcache-research/results/phase2-full-20260721T232020`。
- Phase 2 验收全部通过。严格门禁已切换到 Phase 3；旧 `latest-main` data-plane/MVP 不得视为 Phase 3 完成。

## 2026-07-22T02:14:26-07:00 Phase 3 严格门禁完成

- Phase 3 policy-neutral common core 已冻结为 `experiment/common-core@6742783798ab0b41ce4670bb48d423216ba2681c`；核心实现提交为 `04d168a75100f3c21c81a921eb64dbef70a81048`。
- 只迁移并重写 shared identity/store/transfer 基础；明确排除 EPIC、CacheBlend、Cache-Craft、KVCOMM、CacheTune 和 scheduler policy。
- 已完成 segment content/token/model fingerprint、generation、lease、pin/unpin、backend ownership、device-slot accounting、exact/approx Radix隔离和full coverage validation。
- 已完成 CPU payload export/load、真实HiCache host pool D2H、CUDA-event async H2D ticket、full-layer K/V copy、非零RoPE relocation、dense fallback和last prompt token真实forward。
- ApproxKV manager已接入 RadixCache、HiRadixCache和当前实际使用的UnifiedRadixCache；feature关闭时不启用KV copy或新路径。
- 请求生命周期已覆盖register/reuse、finish、registration error、H2D failure、stream abort、flush/reset；approx source slots纳入scheduler invariant accounting。
- 新增request-scoped Prometheus telemetry：register/reuse outcome、host export tokens/bytes、H2D tokens/bytes/duration、copied tokens和dense fallback tokens/reason。
- 新增Phase 4 recovery plugin接口与Phase 5 scheduler metadata接口；common core自身不注册任何论文算法或调度策略。
- 五轮high-confidence review先后发现并修复：
  - async exception释放前未同步stream；
  - registration capacity failure双重释放backend；
  - registration error向上抛导致同batch后续请求不释放；
  - exact host load-back与approx restore的ownership冲突；
  - 同generation双H2D并发commit释放首个caller buffer。
- targeted suite最终 `41 passed`，包含SM75 CUDA全layer copy/非零RoPE、HiCache event、CPU runtime、abort/error/reset和metrics DI。
- fresh committed SM75 canary使用Phase 2 object，513-token stable prefix：
  - 两次host export合计1,026 tokens / 117,674,874 bytes；
  - 两次async H2D合计1,026 tokens / 117,674,874 bytes；
  - copied tokens 1,026；
  - mismatch与flush后store miss dense fallback合计1,026 tokens；
  - streaming abort被观察，server保持健康；
  - final pool `23591 available + 0 evictable + 2 used = 23593 total`。
- compact结果位于 `benchmark/approx_kv/results/phase3/sm75-canary.json`；raw结果位于 `/home/chris/Workspaces/kvcache-research/results/phase3-final-04d168a75/`。
- Phase 3 验收全部通过。Phase 4所有research worktree必须从同一冻结SHA `674278379` 创建；Phase 5仍不得开始。

## 2026-07-22T03:10:00-07:00 Phase 4 R2 CacheBlend 实现完成（CPU-only targeted 验证）

- 独立 worktree `/home/chris/Workspaces/kvcache-research/worktrees/cacheblend`，分支 `research/cacheblend`，从冻结 common-core `674278379` 创建；只在该 worktree 内提交，未触碰其它 worktree/global/driver/文档，未 push 远程。
- 新提交 SHA `91874f18b`；只新增 `python/sglang/srt/mem_cache/cacheblend/` 包（`hkvd.py`/`recompute.py`/`plugin.py`/`runtime.py`/`__init__.py`）与四个新测试文件；对 `schedule_batch.py`/`radix_cache.py`/`unified_radix_cache.py` 仅在既有扩展点追加调用，未改动 common-core 冻结语义。
- `hkvd.py` 实现真实 HKVD：对某 probe layer 的新鲜 K 与已复用（raw copy+RoPE）K 做逐 token 相对 L2 偏差评分，并支持 gradual filtering（多阶段由粗到细收窄候选池，最终阶段用最深 probe layer 重新排序）；不使用 AST 或任何静态结构代理（`fix-placeholder-pool-activation` 历史分支已证伪 5 种此类信号）。
- `recompute.py` 的 `LayerRecomputeCoordinator` 强制每层恰好一次 batched 调用覆盖全部被选 token（拒绝逐 token 循环、拒绝部分覆盖、重复 slot、层 id 不匹配）；直接对应历史 “minipre”逐 token 前向被证伪的教训（TTFT +1129ms，38x over gate）。
- `plugin.py` 定义 `CacheBlendConfig`（ratio 限定 1/5/15/30%、probe stages、first_recompute_layer、`from_env`）与 `CacheBlendRecoveryPlugin`；`capable` 属性是显式能力门（要求真实 probe backend 与真实 recompute backend 同时绑定），`build_plan` 仅返回保守 dense-only 计划（因为通用 `KVReusePlan` 无法表达“同一 span 内散点 token 与 reuse token 交错”的语义）。
- `runtime.py` 的 `restore_request_prefix_cacheblend` 是真实请求路径：exact cache 优先（不变）→ 多 segment 提前 `begin_prefetch` 实现 load/recompute overlap → 复用 common-core `execute_reuse_plan`/`RadixKVTransferBackend` 做 baseline copy+RoPE → 对整个恢复 span 做真实 HKVD 测量+gradual filtering → 对被选 token 逐层 batched 真实 recompute → 恢复请求的最后一个 prompt token 永远留给真实 forward；任何不支持的布局/不变量（store miss、prefix gap、stale handle、能力缺口、RoPE config 缺失）都释放已分配 slot 并 dense fallback，绝不向 exact Radix 写入近似结果。
- 新增 46 个针对性测试（`test_cacheblend_hkvd.py`/`test_cacheblend_recompute.py`/`test_cacheblend_plugin.py`/`test_cacheblend_runtime.py`），在 Docker（`ghcr.io/ccdd2023/sglang@sha256:0be6e16e...`，CPU-only、无 GPU）中运行：证明 HKVD 分数（而非候选池成员或任何静态信号）驱动 1/5/15/30% 全部四档的最终 token 选择；证明 recompute coordinator 对被选 slot 每层恰好一次 batched 调用，拒绝部分/重复/层不匹配覆盖；证明能力门在任一 backend 缺失时 dense fallback 且无 allocator 泄漏；证明 token 不匹配时 dense fallback 且无泄漏；证明最后一个 prompt token 永不被恢复；证明两个 segment 的 host→device load 均在被 wait 之前就已发出（load/recompute overlap 接口生效）。与该 worktree 内既有 24 个 approx_kv 测试一起共 66 passed、1 skipped（CUDA-only）、0 failed。
- 诚实阻塞点：SGLang ModelRunner 当前没有暴露“对任意 token 子集、与其余 cached 前缀交错、每层一次 batched 前向”的钩子；因此生产环境注册时 `probe_backend`/`recompute_backend` 均为 `None`，能力门会正确触发 dense fallback，而不是伪造 CacheBlend 结果。真实 GPU/server 端到端验证仍被这一缺失的 ModelRunner 钩子阻塞，未做 GPU/server 并行验证。

## 2026-07-22T03:25:00-07:00 Phase 4 R3 Cache-Craft 实现完成（CPU-only targeted 验证）

- 独立 worktree `/home/chris/Workspaces/kvcache-research/worktrees/cachecraft`，分支 `research/cachecraft`，从冻结 common-core `674278379` 创建；只在该 worktree 内提交，未触碰其它 worktree/global/driver/文档，未 push 远程。
- 新提交 SHA `e2b7d047e`；只新增 `python/sglang/srt/mem_cache/approx_kv/cachecraft_*.py`（`cachecraft_metrics.py`/`cachecraft_attention.py`/`cachecraft_plugin.py`/`cachecraft_recompute.py`/`cachecraft_runtime.py`，与 `approx_kv/` 既有平级文件同层）与五个新测试文件；未改动任何 common-core 冻结文件（`types.py`/`manager.py`/`transfer.py`/`radix_backend.py`/`plugins.py`/`store.py`/`request.py`/`config.py`/`runtime.py`/`__init__.py`/`async_transfer.py`/`schedule_batch.py`/`radix_cache.py`）。
- `cachecraft_metrics.py` 忠实实现论文 arXiv:2502.15734 的精确公式：Eq.(3)(4) inter(Ci,Cj)/intra(Ci) attention 求和、Eq.(6) Prefix Overlap Score β（分母为零时按"无从谈起、直接可复用"处理为 β=1.0）、Eq.(7) 基于归一化 Kendall's Tau 距离的 Order Penalty Score γ、Eq.(8) 调整后 β'=β·(1-γ)、Eq.(9)-(10) 逐层平均、按长度归一化的 a(Ci)/b(Ci)、Eq.(11) CCI=sigmoid(ā/b̄)、Eq.(12) CFO=α·CCI·(1-β')（clamp到[0,1]）、Eq.(14) top-N selected-token（按最高外部 attention score 选取 N=⌈CFO·|Ci|⌉ 个 token）；决策规则为 store-miss→FULL_RECOMPUTE，CFO≤0→DIRECT_REUSE，CFO≥可调 `full_recompute_threshold`（默认1.0）→FULL_RECOMPUTE，否则→PARTIAL_REPAIR。
- `cachecraft_attention.py` 实现真实（非占位）dense causal self-attention：genuine `softmax(QK^T/√d)` 加下三角掩码，对真实 Q/K 张量给出与生产模型完全一致的注意力权重；`capture_chunk_profile` 从真实逐层权重切片构建 `ChunkContextProfile`。能力门：生产融合 kernel（FlashInfer/FlashAttention/Triton）不物化完整注意力矩阵，此路径仅在存在 eager/reference attention 前向时可用，这是记录在模块 docstring 中的"真实 profile 采集"服务器接线阻塞点。
- `cachecraft_plugin.py` 的 `CacheCraftPlugin` 实现 common-core `RecoveryPlugin` 协议的 `build_plan`/`scheduler_metadata`；`CacheCraftProfileStore` 与 `ApproxKVSegmentStore` 严格分离（只存注意力统计量，不存 K/V payload，杜绝近似结果绕道进入 exact Radix 的第二条路径）；`CacheCraftDecisionTrace` 记录每次决策的完整 CCI/β/γ/β'/CFO/decision/recompute_positions，供测试与遥测检查。三种决策分支：direct-reuse 产出整段 `copied_spans`；full-recompute 产出整段 `dense_ranges`；partial-repair 用 top-N selected positions 构造 `dense_ranges`（选中的 token）与 `copied_spans`（其余 token）混合计划，复用既有测试 `test_complete_copy_and_dense_head` 已验证的"同一 `KVReusePlan` 可混合 copy+dense"能力，未修改 common-core 类型。
- `cachecraft_recompute.py` 的 `CacheCraftRecomputeBackend` 包装真实 `RadixKVTransferBackend`，使 `dense_prefill` 真正调用调用方注入的 `ChunkRecomputeHook.recompute(...)`（而非仅记录 fallback 原因）；校验 hook 结果覆盖完整请求范围且 RoPE 位置修正到位，不完整/错位结果会抛 `CacheCraftUnsupportedError`；`recompute_hook=None` 时安全记录 unsupported reason 但不写入任何数据。这是"partial repair 必须触发真实 selected-token recompute，而不仅是 metadata/planning"这一要求的核心机制。
- `cachecraft_runtime.py` 的 `restore_request_via_cachecraft` 复刻 common-core `runtime.restore_request_prefix` 的 exact-cache-first/末 token 真实 forward/dense fallback 结构，串联 plugin 决策与真实 recompute backend；`schedule_batch.py:1064` 既有的 `skip_radix_cache_insert = (... or self.approx_kv_metadata is not None)` 与 `request.py` 的 `validate_prompt_length`（`reusable_limit = prompt_length - 1`）无需任何改动即天然满足"近似结果不进 exact Radix"与"末 token 必真实 forward"两条要求。
- 48 个新 CPU-only 测试（Docker `ghcr.io/ccdd2023/sglang@sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781`；`PYTHONPATH` 必须保留镜像自带 `/opt/sm75-site` 前缀，否则 transformers 会解析成错误版本）：
  - `test_cachecraft_metrics.py`（25 个）：证明上下文变化（更多外部 attention）真实提高 a(Ci)/CCI；证明仅改变 chunk 顺序（β 不变、γ 从 0 变到 1）即可在固定阈值下翻转 direct-reuse↔full-recompute 决策；证明 CCI 差异通过可调阈值真实翻转 partial-repair↔full-recompute 决策；覆盖 β 分母为零、CCI 分母为零等边界情况；证明 top-N token 选择按最高外部 attention score 排序。
  - `test_cachecraft_attention.py`（7 个）：证明真实 causal softmax 每行归一、无未来信息泄露；证明手算小样例与实现完全吻合；证明 chunk profile 捕获对 chunk 物理位置/顺序敏感（交换两个 prefix chunk 的物理跨度会使各自捕获到的外部 attention 总量随之互换）；证明多层平均正确反映各层真实统计量。
  - `test_cachecraft_recompute.py`（6 个）：核心证据——证明 `dense_prefill` 真实调用注入的 recompute hook 并传入正确的物理 target indices 和真实 token ids；hook 对每个被选 token 执行真实按 token/位置派生的计算并写入可区分标记值到 K/V buffer 的恰好被选 slot，未触及其余 slot；证明不完整/RoPE 错位的 hook 结果被拒绝为 `CacheCraftUnsupportedError`；证明无 hook 时安全记录 unsupported reason 且不写入任何数据。
  - `test_cachecraft_plugin.py`（6 个）：证明 store-miss（无 profile 或无 handle）强制 FULL_RECOMPUTE；证明相同 prefix 相同顺序触发 DIRECT_REUSE；证明高 CCI+零 overlap 在可调阈值下触发 FULL_RECOMPUTE；证明部分 overlap 触发 PARTIAL_REPAIR 且严格选中外部 attention score 最高的 token 子集，`dense_ranges`+`copied_spans` 精确覆盖全部 target token；证明仅改变已在场 prefix chunk 的顺序即可翻转 DIRECT_REUSE↔FULL_RECOMPUTE。
  - `test_cachecraft_runtime.py`（4 个，端到端最关键）：证明 DIRECT_REUSE 通过真实设备 `move_kv_cache` 复制扩展 `req.prefix_indices`，且未调用任何 recompute hook；证明 PARTIAL_REPAIR 对被选中的两个 token 位置真实调用 recompute hook（校验物理 target indices、真实 token ids、写入的 marker 值），对其余 token 位置走真实设备 copy（校验值与源 K/V 完全一致），并将两者装配进最终 `req.prefix_indices`；证明无 recompute hook 时安全 dense fallback（返回 False、已分配的 device slot 被正确释放而非泄漏、`req.prefix_indices` 保持空）；证明 store-miss 触发 FULL_RECOMPUTE 且未发起任何设备分配。
  - 连同该 worktree 内既有 16 个 approx_kv baseline 测试，共 `64 passed / 0 failed`；黑色格式化（black 26.3.0）+ isort（8.0.1，Docker 镜像内置版本）已对全部新文件运行，重新测试确认格式化未破坏任何用例。
- 诚实阻塞点（写入各模块 docstring）：
  1. 生产侧无独立可调用的 selected-token recompute 钩子：`ForwardMode.TARGET_VERIFY` 只在 speculative-decoding worker pipeline（`eagle_worker_v2.py`/`spec_utils.py`）内部可达，不是通用 request-level API；因此 `recompute_hook` 在真实 GPU server 上目前只能是 `None`，partial repair 会安全 dense fallback 整个 chunk。
  2. 冻结的 wire-level request schema（`request.py`）无字段表达新 prompt 的 chunk order；`cachecraft_runtime.py` 通过 out-of-band 请求属性 `req.approx_kv_new_prefix_order` 读取，而非扩展 `custom_params.approx_kv` 本身（会要求改动冻结文件）。
  3. 未接线 scheduler dispatch（`schedule_batch.py`/`radix_cache.py`）：因真正的 recompute hook 尚不存在，现在接线不会带来任何功能性差异（一定命中 dense fallback），且本次会话不允许 GPU/并发 server 验证，无法验证接线正确性，风险/收益不对等，故推迟到阻塞点 1 解决之后。
- 明确排除 EPIC/CacheBlend/KVCOMM/CacheTune/scheduler policy；固定 S0 LRU/GPU-only/prefetch-off、Phase2 dataset、TTFT-only 范围未被触碰。

## 2026-07-22T03:40:00-07:00 Phase 4 R0 Raw+RoPE 实现完成（CPU-only targeted 验证）

- 独立 worktree `/home/chris/Workspaces/kvcache-research/worktrees/raw-rope`，分支 `research/raw-rope`，从冻结 common-core `674278379` 创建；只在该 worktree 内提交，未触碰其它 worktree/global/driver/文档，未启动 GPU server（其它 research 分支共享主机 GPU 并行工作），未 push 远程。
- 明确定位：R0 是速度上限（speed-only upper bound），显式非忠实 KVCOMM（`2510.12872`）复现；不含 base KV、context-dependent offset、anchor interpolation 或 dense fallback 之外的任何 KVCOMM 特有机制；未引入 EPIC/CacheBlend/Cache-Craft/CacheTune/scheduler/prefetch policy 逻辑；无 accuracy metric。
- 新提交 SHA `41c4c0b25`；新增 `python/sglang/srt/mem_cache/approx_kv/raw_rope.py`、`test/registered/unit/mem_cache/test_raw_rope_plugin.py`、`benchmark/approx_kv/run_r0_raw_rope_cpu_canary.py`、`benchmark/approx_kv/results/phase4-r0/cpu-canary.json`；对 common-core `config.py`/`manager.py`/`runtime.py`/`__init__.py`/`test_approx_kv_runtime.py` 仅做新增/门控式改动（新增字段、新增注册分支、重构 dispatch 但保留全部既有行为契约），未改动冻结类型/存储/lease/HiCache 语义。
- `raw_rope.py` 实现真正的 `RecoveryPlugin`（common-core `plugins.py` 协议）：`select_contiguous_segments` 纯函数负责挑出从当前 exact-prefix 边界起、彼此首尾相接的最长前导 segment 连续段（遇 gap 即在 gap 处截断，不看 gap 之后的内容）；`build_raw_rope_plan` 对该前导连续段逐段做 `store.lookup` 校验、以 `rope_delta = overlap_start(目标位置) - source_position`（有符号整数，统一覆盖 zero/positive/negative）为参数构造 `KVReusePlan` 的 `copied_spans`；`RawRoPERecoveryPlugin.build_plan` 包装以上纯函数，`capable` 恒为 True（无外部 backend 依赖，唯一门是显式配置开关）。V 只被 copy 从不旋转；K 在 copy 之后由既有 common-core `radix_backend.py::_rotate_all_copied_keys` 按 `rope_delta` 旋转，`raw_rope.py` 本身不重复实现旋转数学，只负责产出正确的 delta 与 span 元数据。
- `config.py` 新增独立于通用 `core_enabled` 的显式门 `raw_rope_plugin_enabled: bool`（`SGLANG_APPROX_KV_RAW_ROPE` 环境变量,要求同时 `core_enabled=True` 才生效）；`manager.py` 在 `ApproxKVManager.__init__` 里门开启时自动注册 `RawRoPERecoveryPlugin()`；`runtime.py::restore_request_prefix` 从原先硬编码 inline 的 raw-copy+RoPE 逻辑重构为经 `manager.plugins` registry 派发（调用注册的 plugin 的 `build_plan`），保留 exact-match-first、末 token 永远真实 forward、all-or-nothing dense fallback、`ensure_device` 失败先于任何分配发生的既有顺序不变式。
- 调试中发现并修正的关键认知偏差：最初错误假设"segment 不连续 → 必须整请求 dense fallback"；实测（并与既有 common-core 测试 `test_noncontiguous_segments_fall_back_to_dense` 交叉核对）确认真实契约是 `select_contiguous_segments` 在 orchestration 层于调用 plugin **之前**就把 segment 列表裁剪到 gap 之前的前导连续段，`restore_request_prefix` 对此返回 `True` 并只恢复前导段，gap 之后部分完全不进入这次调用（既不静默修复也不算这次调用的 dense fallback，隐式留给调度器当普通 prefill 处理）；据此改正了 `raw_rope.py` 模块 docstring 与 canary 的场景期望/`known_hard_limitation` 字段表述，避免把"declared segments 不连续"与"已裁剪 run 内部仍缺失/失效"两种不同语义混为一谈——这正是任务要求的"如果任意不连续 selective prefill 需要更深模型改动，实现最安全的、server 可真实支持的子集，并在代码/结果元数据里诚实记录剩余硬限制"的具体落地。
- 新增 18 个分支专属 CPU 测试（`test_raw_rope_plugin.py`）：9 个纯函数测试（无 I/O，覆盖 zero/positive/negative delta、多 segment、interior-after-head、末 token 保留、缺失 segment、不连续 gap、协议包装、payload 校验）+ 9 个经真实 `restore_request_prefix` 的端到端集成测试（含真实 RoPE 旋转数值校验，门开关，各 delta 符号，多 segment，interior segment，不连续/缺失 fallback）。
- 新增可复现、无需 GPU/server 的 canary `benchmark/approx_kv/run_r0_raw_rope_cpu_canary.py`（结果落盘 `benchmark/approx_kv/results/phase4-r0/cpu-canary.json`）：直接在进程内驱动真实 `restore_request_prefix()` 请求路径（非简化 mock），token 序列取自真实 Phase 2 对象生成（`benchmark.approx_kv.workloads.build_object_catalog`，24 个对象）+ 真实 Qwen3-0.6B tokenizer（仅分词，不加载权重，纯 CPU）；8 个场景（zero/positive/negative delta、连续多 segment、interior-after-dense-head、不连续 gap 裁剪为前导段、缺失 segment dense fallback、显式门关闭时插件从未注册）全部通过；对每个 copied span 独立按 neox-style `rotate_half` 公式复算 RoPE 旋转并与实现输出逐 bit `torch.allclose` 比对（而非只信任被测代码自身输出），确认与 `radix_backend.py::_rotate_all_copied_keys` 实际使用的 `apply_rotary_emb` 数学等价；无 accuracy metric，只报告结构正确性、恢复 token 数、rope_delta 等结构化元数据。
- 测试证据：Docker `ghcr.io/ccdd2023/sglang@sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781`（CPU-only）内，`test_approx_kv_core.py`/`test_approx_kv_runtime.py`/`test_approx_kv_integration_source.py`/`test_approx_kv_hicache_backend.py`/`test_raw_rope_plugin.py` 共 42 passed，`test_approx_kv_cuda.py` 1 skipped（容器内无 CUDA），0 failed，较改动前 24 个 baseline 测试无回归；canary 8/8 场景通过，exit code 0；`isort`（8.0.1）与临时安装的 `ruff --select=F401,F821,UP037` 对全部新增/直接改动文件检查干净（修正了本次新增代码中的一处未使用导入与一处 import 顺序；`config.py` 中一处改动前已存在的 UP037 未被本次 diff 触及，未做无关修复）。
- 诚实阻塞点：真实 GPU 上针对真实模型前向的 RoPE 数值正确性验证（对照真实模型 forward 的逐 token K 值）与 TTFT 基准测试未在本次 CPU-only 会话执行；本任务按要求未启动 GPU server（其它 research 分支共享主机 GPU 并行工作），需要主会话安排 GPU 验证。

## 2026-07-22T04:15:00-07:00 Phase 4 R1 EPIC/LegoLink 实现完成（CPU-only targeted 验证，服务器接线含明确记录阻塞）

- 独立 worktree `/home/chris/Workspaces/kvcache-research/worktrees/epic-legolink`，分支 `research/epic-legolink`，从冻结 common-core `674278379` 创建；只在该 worktree 内提交，未触碰其它 worktree/global/driver/文档，未启动 GPU server（其它 research 分支共享主机 GPU 并行工作），未 push 远程。
- 本地提交 SHA `dd4f54919e2c6cddf56383c3caaf4b2376bb62aa`；新增 `epic_capability.py`/`epic_plugin.py`/`epic_recompute.py`/`epic_runtime.py`（均与 `approx_kv/` 既有平级文件同层）与 `test/registered/unit/mem_cache/test_epic_leadingk.py`；对 common-core `config.py`/`manager.py`/`radix_backend.py`/`runtime.py`/`__init__.py`/`observability/metrics_collector.py` 仅做新增/门控式改动；额外对 `managers/schedule_batch.py`/`managers/scheduler.py` 做了显式 config-gated 请求钩子接线（默认关闭、零行为变化）。
- 核心机制：EPIC 的"固定 leading-k 修复"要求对每个 transformer 层，前 k 个 target-context token 必须真正逐层重新计算，而不是整体一次性 forward 或直接复制。实现为 `epic_recompute.py::LayerwiseEpicExecutor`：对每一层，先调用真实 `layer.forward(positions, hidden_states, forward_batch, residual)` 重新计算 leading-k 部分的 K/V（写入 `EpicRecomputeStats` 的调用顺序记录），再（同一层内、在下一层开始前）调用新的 `radix_backend.py::copy_and_rotate_layer()` 只搬运/旋转该层的 body KV（直接索引 `get_key_buffer`/`get_value_buffer`，不使用融合的 `move_kv_cache`），如此逐层交替，而非"规划 k 值"或"整体一次性复制"的 success-shaped stub。
- 支持 k∈{0,2,4,8,16,32}（`config.py::SUPPORTED_EPIC_K_VALUES`，`epic_plugin.py::EPICLeadingKPlugin.leading_k_window()` 做窗口裁剪）与 attention-sink 修复语义（`epic_attention_sink` 配置字段、`carve_leading_k` 保证 leading-k 段始终锚定在请求最前端）；k=0 直接复用 R0 精确路径的 `finalize_copy_reuse()`（从 `runtime.py::restore_request_prefix()` 重构提取的公共构件），不重复实现。
- exact-cache-first、任意不支持模型/布局/不变量时 dense fallback、最后一个 prompt token 永远真实 forward、绝不写入 exact Radix 均由 `epic_runtime.py::restore_request_prefix_epic()` 保证，其 guard 结构与 common-core `runtime.restore_request_prefix()` 完全一致（同样先做 `resolve_reuse_spans()`）。
- 能力门 `epic_capability.py::inspect_layerwise_recompute_capability()`：AST 级检查 `model.model.layers[i].forward` 签名是否真的含 `positions`/`hidden_states`/`forward_batch` 形参，对照真实 `qwen3.py::Qwen3DecoderLayer.forward` 验证通过；不匹配的模型/布局一律 dense fallback，不假装支持。
- 真实"recompute 是真的"证据（而非规划/成功态 stub）：`epic_recompute.py::EpicRecomputeStats.genuinely_layerwise` 属性机械地校验记录的调用顺序确实是"recompute(layer0)→copy(layer0)→recompute(layer1)→copy(layer1)→…"逐层交替，任何重排（如整体 recompute 完再整体 copy）都会被判定为非 genuinely layerwise 并触发 dense fallback；新测试 `test_genuinely_layerwise_detects_reordered_stub` 专门验证这一检测本身有效。测试用的 `FakeDecoderLayer` 对每层输入做真实张量仿射变换推导新 K/V（而非复制固定值），端到端测试 `test_leading_k_repair_recomputes_and_copies_per_layer_genuinely` 校验被恢复的 leading-k 位置的值确实等于该层真实推导值，而 body 位置的值确实等于源 K/V。
- 服务器接线（config-gated，默认关闭，无 GPU/并发验证下安全）：`scheduler.py` 在 `self.tree_cache` 与 `self.tp_worker.model_runner` 均可用处调用 `approx_kv_manager.bind_model_runner(...)`（复刻既有 `bind_residency_backend` 的接线模式，而非像 `bind_rope_config` 那样长期悬空）；`schedule_batch.py` 在既有 R0 请求钩子调用点，按 `approx_kv_manager.config.epic_enabled` 分派到 `restore_request_prefix_epic` 或原 `restore_request_prefix`，默认 `epic_enabled=False` 时行为与改动前逐字节一致。
- 诚实记录的唯一未解决生产阻塞：`epic_runtime.py` 模块 docstring 记录的"PRODUCTION WIRING GAP"——`EpicForwardBatchFactory` seam（为 leading-k-only 前向构造一个独立、正确填充的 `ForwardBatch`）在本 worktree 内未被绑定且未被证明在无 GPU 情况下安全可行；调度器分块（chunked-prefill 边界控制）替代方案超出范围（scheduler policy 被显式排除）。即使 `epic_enabled=True` 且 `model_runner` 已绑定，只要该 factory 未绑定，每次尝试都会在该单一、被清楚记录的点上安全 dense fallback，绝不会伪造成功。
- 测试：新增 28 个测试（`test_epic_leadingk.py`），覆盖能力门（5，含真实 Qwen3DecoderLayer 签名核对）、`LayerwiseEpicExecutor` 交替顺序证明与 stub 检测回归（5）、`EPICLeadingKPlugin` 全部 k 值与 span 切分（5）、config 校验与 env（4）、端到端 runtime 集成（9：k=0 委托、model_runner 未绑定/能力不支持/factory 缺失三种 dense fallback、k=4 全链路真实逐层证明、k=0/2/4/8/16/32 全扫、末 token 不变式、无可复用区间时不触碰能力门/factory）。连同该 worktree 既有 15 个 approx_kv baseline 测试（`test_approx_kv_core.py` 11 个、`test_approx_kv_integration_source.py` 4 个），Docker CPU-only 容器（`python:3.12-slim` + 手动安装的最小 CPU 依赖集，见 TRACKING.md 对应条目）内共 `43 passed`（含 12 个 subtests）、0 failed；`test_approx_kv_runtime.py`/`test_approx_kv_cuda.py`/`test_approx_kv_hicache_backend.py` 因预先存在、与本次改动无关的重 CUDA 依赖链（`dill`→`sgl-deep-gemm`等）无法在轻量 CPU 环境导入，通过手工 scratch 副本回归验证 `restore_request_prefix` 重构行为未变。`black 26.1.0`/`isort 7.0.0`/`ruff 0.15.1`（与 `.pre-commit-config.yaml` 版本一致）对全部新增/改动文件通过；意外重排的、与本次改动无关的既有代码格式已手工还原。
- 明确排除 CacheBlend/Cache-Craft/KVCOMM/CacheTune/scheduler policy；固定 S0 LRU/GPU-only/prefetch-off、Phase2 dataset、TTFT-only 范围未被触碰，无 accuracy metric。

## 2026-07-22T06:50:12-07:00 Phase 4 严格状态纠正与 KVCOMM SM75 结果

本节覆盖此前由并行实现记录产生的 CPU-only“完成”口径。历史记录不删除，但当前 Phase 4/5 门禁以本节为准：只有真实模型 server 请求成功、实际恢复 telemetry 可见且完成同口径 TTFT 对照，才算对应 server MVP；CPU/fake-tensor 测试只证明算法核心或安全 fallback。

| 路径 | 当前分支 | 严格状态 | SM75 结果 |
| --- | --- | --- | --- |
| R0 Raw+RoPE | `research/raw-rope@61c39791e`，已 push | 统一header/body/rho runner、segmented long source与eviction-aware target allocation完成 | 独立验证body1024/2048、header64、rho≈2为`1.73x/2.07x`；完整k0矩阵由R1同物理路径覆盖 |
| R1 EPIC/LegoLink | `research/epic-legolink@984bfd873`，已 push | production in-request seam、eviction-aware recovery allocation、长body分段source和S0 LRU pressure矩阵完成；统一多路径Phase2 trace仍未完成 | body1024/rho≈2：k0 `1.70x`、k32 `1.53x`；body2048：k0 `2.07x`、k32 `1.98x`；rho≈0.9–3收益稳定 |
| R2 CacheBlend | `research/cacheblend@e6dd5eab3`，已 push | 统一ratio/header/body/rho、segmented raw/fresh、RoPE binding、eviction-aware allocation与安全fallback完成 | ratio1%：body1024 target `1.64x`但combined `0.82x`；body2048 target `2.02x`、single-use combined `1.14x` |
| R3 Cache-Craft | `research/cachecraft@d1110066a`，已 push | **当前阶段按用户决定DEFER/SKIP**；统一allocation/contract/blocked runner保留 | 无scheduler dispatch、production attention profile capture、selected-token recompute hook；不产GPU结果 |
| R4 KVCOMM | `research/kvcomm@cd81c3e92`，已 push | 统一header/body/rho、multi-placeholder long body、eviction-aware allocation、压力评测与scheduler-safe fallback完成 | body1024约`1.37x`、setup约1.08s/14次复用break-even；body2048约`1.76x`、setup约2.16s/6次break-even |
| R5 CacheTune controller | `research/cachetune@8acb95e5a`，已 push | 真实server路径、统一pressure runner与SM75 body sweep完成 | body1024/2048 target-only `1.50x/1.80x`；body2048 single-use combined `1.04x` |

KVCOMM 本次结果文件为 `benchmark/approx_kv/results/phase4-r4/sm75-server.json`。功能门禁通过：注册3个 canonical bases、2个 context anchors，fixed-neighbor 路径请求成功并返回首 token。初始短canary在exact-prefix匹配后只实际恢复26 tokens；补充的direct-`input_ids`长度扫描由Prometheus逐请求核对实际恢复576和944 tokens、均为0 fallback。对应结果：

- 576-token恢复、611-token target：KVCOMM p50 `145.28ms`，dense p50 `94.83ms`，`0.653x`，回归 `53.20%`。
- 944-token恢复、979-token target：KVCOMM p50 `198.65ms`，dense p50 `158.34ms`，`0.797x`，回归 `25.46%`。

随着长度增加，reconstruction的相对差距缩小，但在当前稳定的 `<1024` token单chunk SM75配置内仍没有正crossover。未继续跨1024 chunk边界，因为该SM75 torch-native路径已有cross-chunk allocator不稳定记录。

CacheTune `2605.24022v1` 的硬件控制器事实已通过 alphaXiv MCP 复核：论文将 recompute 与 transfer 建模为可重叠 critical paths，

```text
T_layer(r) = max(r * N * t_c, (1-r) * N * t_i) + t_o
r0 = t_i / (t_c + t_i)
```

并用少量 calibration TTFT 对 `[r_min, r_max]` 做 roofline warm-start golden-section search。论文为质量保留 `r_min=15%`；本项目只优化速度，因此实现必须明确区分 `paper-mechanism`（15%下限）与 `speed-only`（可为0%）模式，后者不能称为论文原设定。若只复用既有 repair backend 而未实现 frequency-domain selection、sparse transfer、multi-stream overlap 和 deferred RoPE，则只能标为 CacheTune hardware-controller inspired subset。

当前严格结论：

1. Phase 4 总体未完成：R1生产实现已完成，但R2、R3、R5仍有真实生产或覆盖缺口，且六条路径尚未使用 Phase 2 统一 high-pressure dataset 做同口径筛选。
2. Phase 5 仍 blocked：不得开始真实 scheduler/eviction 评测，直到 Phase 4 server E2E 与统一 recovery benchmark 门禁通过。
3. R1的结论已被长body/high-pressure实验修正：body≤512时k>0负收益，但body1024/2048时k32在真实eviction压力下达到约`1.53x/1.98x`。R2 controlled combined与R4当前测点仍为负收益。
4. 继续优先扩大恢复长度、消除额外准备请求、补 production hook，并以客户端 TTFT 验证；不能因为算法核心或 fake backend 测试通过就进入 scheduler 阶段。

## 2026-07-22T12:14:05-07:00 EPIC production seam、head/body/k矩阵与benchmark规范

R1新增并验证了真实生产子集：

- `TorchNativeEpicForwardBatchFactory`在`Req.init_next_round_input`阶段分配临时request-table row，映射exact prefix physical slots与leading-k目标slots，构造单请求extend `ForwardBatch`。
- 使用真实Qwen input embedding与28层`layer.forward(...)`，每层先重算leading-k，再复制/重定位该层body KV；最后prompt token仍由正常scheduler真实forward。
- 仅支持`torch_native`、TP/PP/DP=1；SWA、LoRA、MRoPE、multimodal和embedding override显式dense fallback。
- 临时request slot在成功/异常路径均释放；CUDA失败同步本身报错时也继续资源清理。
- 58项targeted regression连续运行两次，均为58 passed、0 failed；production server实际telemetry确认k>0每请求重算28层、完整prefix cached、0 fallback。

按用户要求扫描：

```text
k = 0, 2, 4, 8, 16, 32
exact head = 0, 16, 32, 64, 128 tokens
lossy body = 128, 256, 512 tokens
```

共90个EPIC settings与15个fresh dense settings。每个setting先做1次discarded warmup，再正式重复4次并取客户端TTFT p50。结果：

- 所有90个EPIC组合成功返回首token、完整prefix restored/cached、0 fallback。
- 只有`k=0` raw copy+RoPE获得5/15个小幅胜点；最佳是body128/head0，`1.041x`。
- 所有`k>0` genuine repair共75点均慢于dense；最佳是k32/body512/head0，`0.829x`。
- exact head从0增至128时，k>0路径额外增加约8–22ms；leading-k需要对更长exact context做attention，head越长通常越不利。
- body从128增至512能摊薄固定逐层dispatch/copy开销，但仍不足以让k>0交叉到正收益。

compact结果：

`benchmark/approx_kv/results/phase4-r1/sm75-inrequest-matrix.json`

中央运行日志：

`/home/chris/Workspaces/kvcache-research/results/BENCHMARK_RUN_LOG.jsonl`

用户新增的永久benchmark规则：

1. 每次test/benchmark必须把运行设置、代码/镜像/模型版本、原始结果路径和汇总写入专用中央日志。
2. 正式记录前必须有明确discarded warmup passes，warmup不得混入formal samples。
3. 每个setting必须正式重复多次；当前GPU benchmark默认4次，至少不得少于2次，并保存所有原始样本后使用稳健统计量。

## 2026-07-22T12:37:07-07:00 High-pressure eviction设置纠正

用户指出此前body仅128/256/512且server内总working set很小，主要是恢复路径microbenchmark，不能代表GPU cache eviction压力。纠正后的设计不使用单个超大prompt挤爆显存，而使用“单请求安全的较大body × 多对象working set”制造可控oversubscription。

按用户修正后的单请求尺寸：

```text
exact header/prefix = 0, 32, 64, 128, 256 tokens
lossy body          = 512, 768, 1024, 2048 tokens
final real token    = 1
```

这里的header是目标请求中先exact match的context长度，不是attention head数量。body 1024/2048并非放不进本机KV pool：以Phase 2约13,130-token capacity估算，单个2048-token body约占15.6%。此前选择736的原因只是让最大header下仍保持单chunk，先隔离算法开销与SM75 cross-chunk风险。

新矩阵必须显式分组：

- **single-chunk control**：`header + body + 1 <= 1024`的组合，用于最干净的算法归因；
- **cross-chunk long-body**：body 1024/2048，以及body768+header256；继续使用chunked prefill，但单独标记，先做无压力功能canary，再进入pressure sweep。

最大组合`256+2048+1=2305`会跨多个1024-token chunk。它的主要风险不是KV capacity，而是SM75 torch-native prefill workspace和此前观察到的cross-chunk allocator不稳定；因此不能与single-chunk结果混在一起解释。

压力设置：

- server使用`mem_fraction_static=0.35`，启动后从allocator/cache metrics重新读取实际usable KV capacity；Phase 2历史值约13,130 tokens仅作初始估计。
- pressure object使用当前body长度；runner按实测capacity自动反算对象数，使实际reusable rho覆盖约`0.9/1.1/1.5/2.0/3.0`。body2048在rho约3时仍可保留十余个独立对象，足以产生victim diversity。
- 单个对象始终小于单请求稳定上限；通过对象数量与role/context variant数量提高总working set，确保`rho>1`时发生真实Radix eviction，而不是单请求OOM。
- recovery source对象与exact filler对象分离：approx source只占必要device slots；多对象exact filler负责触发可观测eviction和LRU victim choice。
- 每个setting fresh server restart；1次discarded warmup；正式4次重复；结果与settings追加到中央`BENCHMARK_RUN_LOG.jsonl`。

分阶段矩阵，避免立即跑无意义的完整笛卡尔积：

1. **长body功能门**：body512/768/1024/2048分别做无压力canary；cross-chunk组合若出现allocator/OOM必须单独记录，不能静默降级。
2. **压力校准**：优先固定body1024/header64，比较dense、k0 raw、k32 genuine EPIC，在5个rho档位确认无eviction/eviction阈值和无OOM区间；若SM75 cross-chunk门失败，退回body768完成本地pressure基线。
3. **header扫描**：选择首个稳定eviction档与约2x高压档，扫描header `0/32/64/128/256`。
4. **body扫描**：固定header64，在上述压力档扫描body `512/768/1024/2048`。
5. 只有实际`evicted_tokens`、pool invariant和请求成功率均通过，才把该setting纳入后续Phase 4统一筛选。

Phase 5 scheduler仍未解锁；此处只在固定S0 LRU下建立真实pressure归因基线。

## 2026-07-22T14:06:57-07:00 EPIC长body与真实eviction压力结果

用户关于body过小、没有触发eviction的判断得到验证。关键修正分两步：

1. **长body source分段注册**：单次approx register在body1024跨chunk时会耗尽allocator状态，scheduler随后尝试分配1024 tokens时看到`available=0/evictable=0`并退出。改为每个canonical source使用最多512-token segments分别注册，目标请求再连续恢复为1024/2048 body。
2. **恢复分配先驱逐Radix victims**：原ApproxKV路径直接`allocator.alloc`，绕过SGLang标准`evict_from_tree_cache -> allocator.alloc`顺序，在高压下会崩溃。新增共享`allocate_recovery_slots()`，R0/k0与k>0 EPIC均在分配恢复buffer前触发exact Radix eviction。

两次59-test targeted regression均通过，review未发现剩余高置信问题。

### 无压力长body功能门

固定header64、每setting 1次discarded warmup + 4次formal repeats：

| body | dense p50 | k0 p50 / speedup | k32 p50 / speedup |
| ---: | ---: | ---: | ---: |
| 1024 | 297.29ms | 173.32ms / `1.72x` | 194.60ms / `1.53x` |
| 2048 | 971.32ms | 475.57ms / `2.04x` | 492.05ms / `1.97x` |

### rho压力扫描

固定body1024/header64、capacity=13,130 tokens、filler=736 tokens、S0 LRU：

- pre-target rho约`0.924/1.148/1.540/2.045/3.054`；
- peak rho约`1.002/1.226/1.618/2.123/3.132`；
- 所有formal runs均发生真实eviction、成功返回首token、EPIC 0 fallback；
- k0在五档保持约`1.73x`；
- k32保持约`1.49–1.56x`；
- 四次formal runs累计evicted tokens从约5.9K增长到115K，收益没有因高pressure消失。

### body扫描（peak rho约2、header64）

| body | dense p50 | k0 speedup | k32 speedup |
| ---: | ---: | ---: | ---: |
| 512 | 87.31ms | `0.96x` | `0.76x` |
| 768 | 127.82ms | `1.00x` | `0.83x` |
| 1024 | 299.34ms | `1.70x` | `1.53x` |
| 2048 | 980.87ms | `2.07x` | `1.98x` |

明确crossover位于768与1024之间；此前body≤512矩阵不足以代表长上下文收益。

### header扫描（body1024、peak rho约2）

header=`0/32/64/128/256`均通过。header增大时绝对TTFT上升，但dense增长更快：

- k0 speedup约`1.69/1.73/1.69/1.74/1.76x`；
- k32 speedup约`1.46/1.50/1.51/1.53/1.59x`。

compact结果：

`benchmark/approx_kv/results/phase4-r1/sm75-eviction-pressure.json`

当前R1结论：真正的leading-k repair不是普遍无效，而是存在明显长度门槛；在SM75/Qwen3-0.6B上，body1024起开始出现显著TTFT收益，并在rho≈3的真实eviction压力下保持。

## 2026-07-22T17:08:52-07:00 Phase 4 全路径统一contract

所有R0–R5必须统一使用：

```text
header = 0, 32, 64, 128, 256
body   = 512, 768, 1024, 2048
rho    = 0.9, 1.1, 1.5, 2.0, 3.0
source segment <= 512 tokens for long bodies
mem_fraction_static = 0.35
scheduler = S0 LRU
tier = GPU-only
prefetch = off
warmup = 1 discarded pass / setting
formal repeats = 4 (minimum 2)
central log = results/BENCHMARK_RUN_LOG.jsonl
```

所有恢复buffer必须使用eviction-aware allocation：先`evict_from_tree_cache`，再`allocator.alloc`。source registration不因保存未来近似对象而主动驱逐exact victims；只有当前目标请求为继续执行所需的恢复分配可以触发驱逐。

当前compliance状态：

| 路径 | 统一代码contract | 统一GPU结果 |
| --- | --- | --- |
| R0 Raw+RoPE | 完成并push `61c39791e` | 代表性body1024/2048 GPU通过；完整contract runner就绪 |
| R1 EPIC | 完成并push `984bfd873` | 完成：header/body/rho全矩阵 |
| R2 CacheBlend | 完成并push `e6dd5eab3` | 完成：ratio/header/body/rho统一GPU矩阵 |
| R3 Cache-Craft | **DEFERRED/SKIPPED for now**；`d1110066a`保留CPU证据与blocked runner | 从当前Phase4完成门禁排除，不允许伪造GPU结果 |
| R4 KVCOMM | 完成并push `cd81c3e92` | 完成：header/body/rho全矩阵 |
| R5 CacheTune | 完成并push `research/cachetune@8acb95e5a`；non-prefix、streaming、独立round、真实eviction、telemetry与pool reset门均通过 | body512/768/1024/2048 target-only `0.94x/0.93x/1.50x/1.80x`；combined `0.48x/0.44x/0.76x/1.04x` |

### R4 KVCOMM统一结果

长body被拆为多个≤512-token placeholders。每个placeholder分别建立target canonical base、两个anchor bases和两个context deltas；目标请求连续重建所有groups。统一long-body runner不包含neighbor group，neighbor机制仍由旧小canary证明。

- body512/768仍慢于dense；
- body1024、rho≈2：dense 299.34ms，KVCOMM target 218.69ms，`1.37x`；setup约1.08s，约14次target reuse摊销；
- body2048、rho≈2：dense 980.87ms，KVCOMM target 558.67ms，`1.76x`；setup约2.16s，约6次reuse摊销；
- body1024在peak rho≈1.03–3.11保持约`1.36–1.38x`；
- header从0增至256时target-only speedup从约`1.30x`增至`1.46x`；
- 代表性body1024/rho2四次formal request均机械验证`copied_tokens_delta=1024`、cached=1088、0 fallback。

结果：

`benchmark/approx_kv/results/phase4-r4/sm75-unified-pressure.json`

### R3 Cache-Craft当前阶段跳过决定

用户明确允许：若R3修复代价过高，当前先跳过并完整记录原因。R3不再阻塞当前Phase 4剩余路径或后续Phase 5门禁。

跳过原因：

1. `schedule_batch.py`没有按`metadata.plugin=="cachecraft"`分派，`restore_request_via_cachecraft`对真实请求不可达。
2. FlashInfer/FlashAttention/Triton生产融合attention不物化完整attention矩阵，无法直接生成论文所需CCI/CFO profile。
3. SGLang没有通用request-level selected-token recompute hook；`TARGET_VERIFY`仅存在于speculative decoding内部。
4. 修复需要同时改scheduler dispatch、model forward/profile capture和selected-token execution，不是局部benchmark接线，且必须重新做GPU安全验证。

保留内容：论文公式、CPU profile/decision/recompute core、eviction-aware allocation、统一workload contract、显式blocked runner。后续只有在单独批准R3深层实现阶段才恢复。

状态澄清：当前所说“卡住并暂时跳过”的research就是 **R3 Cache-Craft**。R5 CacheTune仍在正常收尾，不属于同一阻塞；后续工作则是人工确认门，并非技术失败。

### R2 CacheBlend统一结果

- ratio=`1%/5%/15%/30%`均完成真实selected-token telemetry；
- ratio越高，target latency单调增加；body1024/rho≈2的最佳是1%：
  - dense 299.34ms；
  - fresh preparation 185.58ms；
  - target 182.26ms=`1.64x` target-only；
  - combined 367.28ms=`0.82x`，单次仍慢于dense；
  - fresh成本约2次target reuse可摊销。
- body crossover：
  - 512/768 target-only仍<1x；
  - body1024 target-only `1.64x`；
  - body2048 target `2.02x`，fresh 375.56ms，combined 862.02ms vs dense 980.87ms，single-use combined `1.14x`。
- body1024 ratio1%在peak rho约0.99–3.12保持稳定；header0/32/64/128/256全部完成。

仍然是precomputed fresh-KV adapter，不是通用inline ModelRunner selected-token hook；fresh与target串行执行。

结果：`benchmark/approx_kv/results/phase4-r2/sm75-unified-pressure.json`

## 2026-07-22T18:05:02-07:00 Phase 5人工确认硬门

用户明确要求：进入Phase 5之前必须停止并获得进一步的明确确认。

执行规则：

1. 当前只允许完成R2、R5及Phase 4统一评测/文档收尾。
2. 即使所有Phase 4技术门禁全部通过，也不得自动创建Phase 5分支、修改scheduler/eviction/prefetch代码、运行Phase 5测试或benchmark。
3. 到达Phase 4完成点后必须先汇报完整结果、未解决风险和建议的Phase 5范围，然后暂停。
4. 只有收到用户明确授权后，才可以把`scheduler-policies`、Phase 5 local screening或prefetch工作从blocked改为进行中。

该人工审批门高于自动pilot推进和普通依赖门禁。

## 2026-07-22T18:43:49-07:00 阶段报告slides状态

- 阶段报告草稿：`research/PHASE4_STAGE_REPORT_SLIDES.md`。
- 当前为8页brief slides，只保留主要问题、单页research简介、preliminary findings、关键结果和最优路径。
- R0–R4合并在一页“我们复现的 research”中，每条只保留一句直观机制简介；数字和状态放在后续结果页。
- `Results overview`按R0、R1、R2、R4、R3列出当前slides保留的research，最佳两条R0/R1置顶。
- 报告slides已按用户要求完全删除CacheTune/R5；即使R5技术结果现已完成，也不自动加入本版presentation。
- slides中的CacheBlend表述已澄清为`repair ratio = 1%`，即约选择1%的body tokens修复，不是“repel”或其他术语。
- slides同时明确header size是body之前可exact match的prefix/context token长度，不是attention head数量或维度；1% repair也明确不是1% speedup。
- 全文已删除Phase级别措辞；最后一页为`Next`，只保留HiCache协同、RTX 6000与更长context统一对比。
- 已删除runtime建设步骤、关键代码列表、平台型号、审计过程等presentation不需要的细节。
- R5技术结果已在独立分支完成；slides内容继续保持用户确认过的8页版本，除非用户明确要求再加入R5。
- 最终human-tone编辑已完成：8页结构与数字不变，改为短句和直接判断，去除AI模板语气、机械对称bullet与空泛连接词。
- 全文确认不含`Phase 4`、`Phase 5`、`CacheTune`、`R5`或`Vast AI`。

## 2026-07-22T16:41:21-07:00 R3 Cache-Craft 迁移共享 allocate_recovery_slots 与统一 Phase 4 contract（CPU-only、未 push）

独立 worktree `/home/chris/Workspaces/kvcache-research/worktrees/cachecraft`，分支 `research/cachecraft`；只在该 worktree 内本地提交，未触碰其它 worktree/global/docs，未启动 GPU/server，未 push 远程。所有 git/测试/lint 均在 Docker（`ghcr.io/ccdd2023/sglang@sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781`，CPU-only）内执行；容器挂载路径与宿主机 worktree 路径保持一致（含主仓库 `sglang-experiments` 目录）以便 `git worktree` 的 gitdir 引用正确解析。

**任务1：迁移共享 `allocate_recovery_slots`**——从 R1 EPIC donor 的 `runtime.py` 移植 `allocate_recovery_slots(tree_cache, num_tokens)`（分配前先检测 `tree_cache.evict`/`is_chunk_cache` 与 `allocator.available_size` 均存在时，调用 common-core `evict_from_tree_cache` 驱逐 exact Radix victims，再 `allocator.alloc`），只移植该 helper 本身，未移植 donor 中范围更大的 `resolve_reuse_spans`/`finalize_copy_reuse` 重构（超出任务范围）。`runtime.py::restore_request_prefix` 与 `cachecraft_runtime.py::restore_request_via_cachecraft`（partial-repair 恢复 buffer 分配）均改为调用该共享 helper 而非直接 `allocator.alloc`。新增高压力/无泄漏测试：`test_approx_kv_runtime.py`（3 个新测试，`PressureAllocator`/`PressureTree` fixture，证明 eviction-then-alloc 成功、端到端通过、以及持续 OOM 时 dense fallback 且无 buffer 泄漏）与 `test_cachecraft_runtime.py`（2 个新测试，同样的 partial-repair 路径证明）。

**任务2：诚实审计生产阻塞点 + 统一 pressure runner scaffold**——通过 grep 精确定位到比既有 docstring 更严重的阻塞：`schedule_batch.py` 对任何携带 `approx_kv_metadata` 的请求无条件调用通用 `runtime.restore_request_prefix`，从不检查 `metadata.plugin`，意味着即使真实发出 `plugin: "cachecraft"` 请求也会被错误的通用 raw-copy 路径处理，`restore_request_via_cachecraft` 目前对任何真实 server 请求都零可达性。新增 `cachecraft_capability.py::inspect_scheduler_dispatch_capability()`：对真实可 import 的 `schedule_batch` 模块做源码内省（`inspect.getsource` 搜索 `restore_request_via_cachecraft` 符号），零网络零 GPU，返回冻结的 `CacheCraftServerCapability(supported, reason)`；当前诚实返回 `supported=False`。新增 `cachecraft_workloads.py`：统一 Phase 4 contract 常量（header `0/32/64/128/256`、body `512/768/1024/2048`、body>512 canonical source 按 `<=512`-token segments 注册、`mem_fraction_static=0.35`、rho约`0.9/1.1/1.5/2.0/3.0`、S0 LRU/GPU-only/prefetch-off、warmup=1、formal repeats 默认4/最少2）与 GPU-free 的 `build_non_prefix_segmented_workload`（为未来真实 hook 准备的非-prefix 乱序 chunk workload）。新增 `run_phase4_cachecraft_pressure.py`：完整复刻 R1 已完成 runner 的 settings/warmup/repeats/中央 JSONL/streaming TTFT client 契约，但 `main()` 首先调用能力检查；当前（诚实）状态下始终走 blocked 路径——写一条 `status: "blocked"` 的中央日志（含完整 settings 与真实阻塞原因）、零网络/GPU 调用、不产出结果文件、exit code 3；"真实运行"代码路径（`run_real`/`run_round`/HTTP client）完整实现同一契约但结构性不可达，仅通过 fake HTTP transport 做单元测试，绝不用 fake backend 伪造 Cache-Craft server 成功。

**任务3：README/结果metadata contract**——在 `benchmark/approx_kv/README.md` 追加新的"Phase 4 unified high-pressure contract (Cache-Craft / R3)"小节（未删除/覆盖既有 Phase 2/Phase 3 内容），记录统一 contract 数值、中央日志 metadata 形状（`status` 取值 `running/completed/failed/blocked`、按路径分子目录、绝不覆盖既有历史结果文件）与 Cache-Craft 当前诚实的能力门控状态；同步更新 `cachecraft_runtime.py` 模块 docstring，加入比原文更精确的 scheduler dispatch 阻塞条目并交叉引用新的 `cachecraft_capability.py`。

**任务4：测试与格式化**——目标测试集合（`test_approx_kv_core/runtime/integration_source/hicache_backend` + 全部 `test_cachecraft_*` + 新增 `test_cachecraft_capability/workloads` + `test_run_phase4_cachecraft_pressure`）在格式化前后各运行两遍，稳定 `114 passed / 0 failed`。`black`：5 个全新文件已重新格式化；4 个被修改的既有文件中，只对本次新增的代码段做了格式化，明确保留其余既有、与本次改动无关的历史格式（遵循"不修复无关既有问题"的原则）。`isort`：全部文件无需改动。`ruff --select=F401,F821,UP037`：全部干净，唯一例外是 `test_cachecraft_runtime.py` 中一处早于本次改动就存在的未使用 `from typing import Any` 导入，确认与本次 diff 无关，未做无关修复。

**提交**：本地提交 `57fc991fc`（`research/cachecraft` 分支），11 个文件、1762 行新增/5 行删除，含 Copilot co-author trailer；未 push。

明确排除 EPIC/CacheBlend/KVCOMM/CacheTune/scheduler policy；固定 S0 LRU/GPU-only/prefetch-off、Phase2 dataset范围未被触碰，无 accuracy metric，无 GPU/server 端到端声称。

## 2026-07-23T06:47:21-07:00 R5 CacheTune最终SM75结果与暂停门

`research/cachetune` 已完成并push到 `8acb95e5a`。其中 `afcbcb027` 是最后一个runner/runtime修复提交，解决flush后Prometheus gauge滞后：每个独立round在抓取baseline前发送固定dense sentinel刷新scheduler与gauge，不对负值做静默clamp。最终提交新增统一结果和四个独立main-setting结果。

固定口径：

```text
header = 64
body = 512, 768, 1024, 2048
target rho = 2
pressure filler body = 512
source segment <= 512
mode = speed_only
scheduler = S0 LRU
tier = GPU-only
prefetch = off
warmup = 1 discarded
formal repeats = 2
```

| body | dense p50 | target p50 | target-only | combined p50 | combined |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 94.35ms | 100.65ms | `0.94x` | 195.71ms | `0.48x` |
| 768 | 131.53ms | 141.83ms | `0.93x` | 296.60ms | `0.44x` |
| 1024 | 289.28ms | 193.17ms | `1.50x` | 379.59ms | `0.76x` |
| 2048 | 917.47ms | 509.24ms | `1.80x` | 880.95ms | `1.04x` |

机械有效性：

- 所有setting和formal round均通过，且每轮 `evicted_tokens_total_delta=13,766`，确认是真实exact-Radix eviction压力。
- 每次恢复的`cached_tokens`严格等于header+body：`576/832/1088/2112`。
- controller选择的selected tokens为每请求`42/64/85/170`；累计telemetry与期望完全一致。
- `approx_kv_dense_fallback_total`增量为0；每个setting结束后的pool invariant通过。
- 中央日志已写入每个run的running/completed记录、完整settings、raw output和summary。

结论：

- CacheTune的target-only crossover也位于768与1024之间。
- body1024有明显target-only收益，但fresh preparation使single-use combined仍为负。
- body2048是首个同时获得target-only和single-use combined正收益的CacheTune点。
- 当前实现是CacheTune-inspired hardware-aware controller与precomputed fresh-KV adapter，不是论文完整实现；缺少frequency-domain selection、真实transfer/recompute overlap和通用inline selected-token recompute hook。

结果文件：

- `benchmark/approx_kv/results/phase4-r5/sm75-unified-pressure.json`
- `benchmark/approx_kv/results/phase4-r5/sm75-body512-rho2.json`
- `benchmark/approx_kv/results/phase4-r5/sm75-body768-rho2.json`
- `benchmark/approx_kv/results/phase4-r5/sm75-body1024-rho2.json`
- `benchmark/approx_kv/results/phase4-r5/sm75-body2048-rho2.json`

当前保留的R0/R1/R2/R4/R5恢复实验已经收尾，R3继续defer。按照用户的人工门禁，现已停止；未经明确批准，不得进入scheduler/eviction/prefetch阶段，也不自动启动新的统一screening。RTX PRO 6000复测和跨分支Phase 2 dataset横评仍未完成。

## 2026-07-24T01:25:06-07:00 Phase 5 获得授权并启动

- 用户明确授权：若 Phase 4 已全部完成，则自主继续 Phase 5，全部完成后再统一汇报。该指令解除 2026-07-22 建立的 Phase 5 人工确认硬门。
- Phase 4 完成判定保持不变：R0/R1/R2/R4/R5 已完成当前统一 SM75 高压力收尾；R3 Cache-Craft 依据既有决定继续 defer，不再阻塞阶段切换。
- Phase 5 默认按完整计划执行：
  - S0 LRU；
  - S1 KVFlow steps-only；
  - S2 Belady oracle next-use；
  - S3 recovery-aware value density；
  - S4 hierarchical object policy；
  - P0 off、P1 free-space-only、P2 dead-object-only eviction、P3 oracle next-stage prefetch。
- 实施必须新增独立 cache-protection metadata，不复用 request scheduling 的通用 `priority` 字段。
- 首轮仍以 sequential 固定 workflow、真实 SM75、high-pressure、客户端 TTFT 为主；HiCache/prefetch 接线必须遵守保守 admission，不允许为了预取驱逐仍有未来用途的活跃高价值对象。
- 计划采用独立 worktree/branch，完成策略实现、Docker 测试、真实 GPU 筛选、中央日志、原始结果、提交与远程持久化后再向用户汇报。

## 2026-07-24T04:04:09-07:00 Phase 5 实现完成并进入正式矩阵

- 独立 worktree：`/home/chris/Workspaces/kvcache-research/worktrees/scheduler-policies`；分支：`research/scheduler-policies`；基线：`research/epic-legolink@984bfd873`。
- 已实现独立 cache-protection 数据链，不读取或覆盖 `Req.priority`：
  - `workflow_steps`；
  - `belady`；
  - `recovery_value`；
  - `hierarchical`；
  - 原生 `lru` 继续作为 S0。
- metadata 使用 absolute workflow/request ordinal，避免相对 next-use 随 trace 推进失效；flush/reset 会重置 S3 当前 step。
- `protected_tokens` 会在 Radix/HiRadix/UnifiedRadix 内切出 reusable-prefix 边界；对象 metadata 不再传播到 invocation-specific dynamic suffix。
- P0-P3 已实现：
  - P0 off；
  - P1 free-space-only；
  - P2 known-dead-object-only；
  - P3 oracle-farther-use 且对象必须可恢复。
- prefetch victim 以“对象边界 + 全部 dynamic suffix 后代”为原子子树；HiRadix H2D 会启动 transfer、同步等待 completion，并由 `loading_check` 释放锁。write-back 下需要 victim eviction 的 P2/P3 当前明确拒绝，不伪造支持。
- 已新增 Prometheus telemetry：
  - `workflow_cache_evicted_tokens_total`；
  - `workflow_prefetch_requests_total`；
  - `workflow_prefetch_loaded_tokens_total`；
  - `workflow_prefetch_evicted_tokens_total`。
- 验证：
  - Docker CPU targeted regression：`224 passed`、`27 subtests passed`；
  - SM75 HiRadix metadata/split、P1 load-back/lock release、P2 dynamic-suffix subtree eviction：`3 passed`。
- 真实 SM75 smoke：
  - S0 LRU、rho≈1.537：workflow cache-hit fraction约`1.56%`；
  - S1 steps-only、同压力：约`40.93%`，总 eviction 从`48,966`降至`26,882` tokens；
  - HiCache P0/P1均保持完整workflow hit；P1在高压下没有主动load，符合free-space-only；
  - P2/P3各完成`6,050` tokens主动load与`6,260` tokens admission eviction，pool reset和accounting通过；当前同步实现的workflow TTFT慢于P0/P1。
- 正式 S0-S4、rho=`1.1/1.5/2/3`、warmup1、formal repeats2 矩阵正在运行；完成后再决定P0-P3正式扩展点。

## 2026-07-24T05:12:34-07:00 Phase 5正式矩阵结果

- S0-S4 GPU-only正式矩阵完成：20个setting，rho=`1.1/1.5/2/3`，每个setting warmup1 + formal repeats2，全部请求、eviction、reset invariant通过。
- S4 hierarchical是唯一在三个高压档均稳定优于S0 LRU的策略：
  - rho1.5：mean `163.52ms` vs LRU `217.04ms`，`1.33x`；p50 `150.52ms`，`1.44x`；
  - rho2.0：mean `188.25ms` vs `216.25ms`，`1.15x`；p50 `149.16ms`，`1.45x`；
  - rho3.0：mean `188.91ms` vs `216.64ms`，`1.15x`；p50 `149.64ms`，`1.45x`。
- S4在rho1.5/2/3的workflow hit fraction约`0.903/0.705/0.705`；LRU约`0.510/0.510/0.511`。
- S1/S2/S3在rho2/3退化到约`0.508`命中率，mean TTFT与LRU基本相同；Belady上界未优于S4，说明当前对象层级比单纯next-use更关键。
- S4+HiCache P0-P3正式矩阵完成：12个setting，rho=`1.5/2/3`。
  - 所有模式workflow hit fraction均为1.0；
  - P0 mean约`149.27/150.06/152.62ms`；
  - P1无主动load，结果与P0接近；
  - P2每档主动load `2,016` tokens、admission eviction `2,088` tokens；
  - P3在rho3主动load `4,032`、eviction `4,104` tokens；
  - P2/P3没有稳定mean收益，且p95约`158–161ms`，高于P0约`152–156ms`。
- 当前Phase5默认选择固定为 **S4 + P0**；P1/P2/P3保留为已实现、已验证的实验变体，不作为sequential默认。
- 正在补充rho1.5/2.0的两次独立重启；与正式矩阵已有一次合计三次，用于最终趋势复核。

## 2026-07-24T06:58:18-07:00 Phase 5最终完成

- 远程分支：`research/scheduler-policies@c185428fd`，已显式使用`ccdd2023` SSH身份完成dry-run、push与remote SHA核对。
- 实现提交：`5a87166b4`；结果提交：`c185428fd`。
- 最终测试：
  - Docker CPU：`226 passed`、`27 subtests passed`；
  - SM75 HiRadix targeted GPU tests：`3 passed`；
  - black对新增及本次实现文件通过；isort对全部改动Python文件通过；当前固定镜像未安装ruff，因此未临时安装新lint工具。
- 所有正式GPU结果均重跑并绑定干净实现提交`5a87166b436e00fa730aa7062e949516ca823a96`。
- commit-bound S0-S4正式矩阵：
  - rho1.5：S4 mean `163.46ms`，LRU `215.93ms`，`1.32x`；
  - rho2.0：S4 `188.96ms`，LRU `216.56ms`，`1.15x`；
  - rho3.0：S4 `189.31ms`，LRU `214.37ms`，`1.13x`；
  - S4 hit fraction约`0.903/0.705/0.705`，LRU约`0.510/0.510/0.511`。
- 三次独立server进程验证：
  - rho1.5 S4 mean speedup `1.32–1.34x`；
  - rho2.0 `1.11–1.15x`。
- commit-bound S4+HiCache P0-P3：
  - P0 mean约`149.94/151.03/151.47ms`；
  - P1没有主动load，差异仅为运行噪声；
  - P2每档主动load `2,016`、admission eviction `2,088` tokens；
  - P3在rho3主动load `5,040`、eviction `5,112` tokens；
  - P2/P3无稳定mean收益，p95约`159–161ms`，高于P0约`153–154ms`。
- 最终默认固定为 **S4 hierarchical + P0 off**。
- 紧凑结果：
  - `benchmark/approx_kv/results/phase5-scheduler/sm75-scheduler-matrix.json`；
  - `benchmark/approx_kv/results/phase5-scheduler/sm75-prefetch-matrix.json`；
  - `benchmark/approx_kv/results/phase5-scheduler/sm75-restart-validation.json`。
- 当前停止，不自动进入Phase 6跨恢复路径组合或RTX PRO 6000 scale；这些属于下一阶段。

## 2026-07-24T08:20:07-07:00 rho与speedup口径澄清

- 用户指出历史结论是“cache pressure越大，workflow-aware priority越有价值”，但Phase5汇报中的mean speedup从rho1.1到rho3下降。
- Phase5 setting没有并行干扰：
  - 整个matrix一次只运行一个SGLang server/GPU setting；
  - 每个setting独立启动和停止server；
  - client逐请求`await`，实际请求串行；
  - warmup丢弃，formal repeat之间flush；
  - setting顺序随机化；
  - rho1.5/2又做了三次独立server进程验证。
- 历史结论应准确解释为：从“working set完全可容纳”进入“开始oversubscribe”的区间，workflow-aware policy价值通常上升；它不是rho无限增大时speedup必然单调上升的定律。
- 当前Phase5从rho1.1开始时，S4优势已经接近峰值：
  - rho1.1：S4/LRU hit fraction=`1.000/0.510`，mean speedup约`1.46x`；
  - rho1.5：`0.903/0.510`，约`1.32x`；
  - rho2：`0.705/0.510`，约`1.15x`；
  - rho3：`0.705/0.511`，约`1.13x`。
- LRU已经接近约0.51的命中下限，继续加压时恶化不大；S4则从full hit降到0.903再降到0.705，因此两者差距缩小。
- mean与p50必须分开：
  - rho1.5/2/3的p50 speedup仍约`1.44x/1.45x/1.42x`；
  - mean下降主要来自更多约280ms级miss进入平均值；
  - 当前分布是“多数fast hit + 少数slow miss”的双峰，而非每个请求都均匀变慢。
- 当前rho sweep还同时扩大了working set：所选对象数约为`15/20/27/40`，并增加live/dead pressure对象，不是固定同一对象集合只改变capacity。因此它不能单独证明或否定“固定workload下speedup随rho单调变化”。
- 若后续专门复核该历史claim，正确实验应固定同一40-object trace和对象类别，只改变GPU capacity/mem_fraction来扫rho，避免working-set composition成为混杂变量。

## 2026-07-24T14:18:09-07:00 Phase 5 baseline、Phase 4对比与warm-up口径

- Phase5 scheduler matrix中，S4的baseline是**S0 LRU**，不是Phase4 dense，也不是R0/R1：
  - 同一Qwen3-0.6B/SM75；
  - 同一exact Radix workload、对象集合、rho和P0；
  - 只替换`radix_eviction_policy`。
- S0-S4含义：
  - S0：LRU；
  - S1：coarse workflow steps-to-execution；
  - S2：exact next-request ordinal的Belady-style oracle；
  - S3：synthetic saved-cost/resident-byte value density；
  - S4：dead/recoverable exact/repair/anchor/exact/canonical-base层级。
- Phase5 prefetch matrix的baseline又不同：固定S4+HiCache，对比P0；P1-P3均相对该P0。
- Phase5 raw最佳metrics：
  - 全局最低mean：S4/rho1.1 `148.50ms`；
  - 全局最低p50：S3/rho1.5 `148.49ms`，但其mean/p95为`187.84/280.36ms`，属于hit/miss双峰，不能作为整体最佳；
  - 全局最低p95：S3/rho1.1 `149.33ms`；
  - 高压力跨rho最稳：S4。
- S0-S4 mean speedup相对S0：
  - S1：rho1.1/1.5/2/3=`1.446/1.144/1.006/0.994x`；
  - S2：`1.428/1.148/0.999/0.996x`；
  - S3：`1.454/1.150/1.011/0.990x`；
  - S4：`1.456/1.321/1.146/1.132x`。
- Phase4与Phase5的speedup分母不同，不能直接排名：
  - Phase4：`dense target prefill / approximate recovery target`，主要衡量单请求恢复机制；
  - Phase5：`S0 LRU workflow TTFT / policy workflow TTFT`，主要衡量对象是否仍resident。
- Phase4 body2048 target-only最佳点：
  - R0 raw/k0 `2.07x`；
  - R2 CacheBlend 1% `2.02x`，combined `1.14x`；
  - R1 EPIC k32 `1.98x`；
  - R5 CacheTune `1.80x`，combined `1.04x`；
  - R4 KVCOMM `1.76x`，setup break-even约6次。
- Phase4的rho sweep本身并未证明speedup随rho单调增加：
  - R1 k0在rho0.9–3约`1.73x`稳定；
  - R1 k32约`1.49–1.56x`；
  - R2约`1.61–1.64x`；
  - R4约`1.36–1.38x`。
- Phase4真正明显的单调变量是body length：512/768通常无收益，1024开始crossover，2048收益最大。
- 当前Phase5不是“Phase4最优恢复路径+S4”的端到端组合：
  - runner没有发送`approx_kv` register/reuse；
  - S3使用Phase2 synthetic cost weight，不是Phase4实测R0-R5 profile；
  - S4的canonical/anchor/repair是exact Radix对象标签，不是approx store中真实物理对象。
- 真正的Phase4+5组合属于Phase6：选择前两条恢复路径，与S4/P0在同一trace下联测。
- warm-up实际存在：
  - 每个setting执行1次完整warm-up trace；
  - warm-up使用独立`cache_salt="warmup"`；
  - warm-up结果保留但不计入mean/p50/p95；
  - 随后flush，并以`cache_salt="measured"`运行formal repeats；
  - formal repeats之间也flush。
- “discarded warm-up”只表示不计入正式统计，不表示没有运行。
- warm-up用于消除首次kernel/runtime/allocator冷启动；若把它计入formal，会放大cold-start噪声；若保留其cache内容不flush，则会人为预热exact cache并夸大speedup。

## 2026-07-24T14:40:33-07:00 Phase 4/5 workload与策略维度澄清

- Phase4不是五个完全不同的数据workload，而是同一个统一non-prefix恢复workload contract下的五条恢复路径：
  - R0 Raw+RoPE；
  - R1 EPIC fixed-k；
  - R2 CacheBlend selective repair；
  - R4 KVCOMM；
  - R5 CacheTune；
  - R3 Cache-Craft defer。
- Phase4共同setting轴为header/body/rho、S0 LRU、GPU-only、P0；各R路径另有自身参数轴，如k、repair ratio、anchor setup或controller ratio。
- Phase5没有把R0/R1/R2/R4/R5分别与S0-S4做组合。
- Phase5使用一套新的exact-Radix scheduler microbenchmark trace：
  - 5个固定workflow对象：Architect 1、Coder 2、Debugger 2；
  - live/dead pressure fillers；
  - 两轮`Architect -> Coder -> Debugger -> Coder -> Debugger`；
  - live filler replay。
- Phase5 scheduler matrix是在该单一trace family下比较：
  - S0 LRU baseline；
  - S1-S4只替换eviction policy；
  - rho=`1.1/1.5/2/3`。
- Phase5 prefetch matrix仍使用同一逻辑trace，固定S4+HiCache，以P0为baseline比较P1-P3。
- Phase5分支虽然基于R1 EPIC代码分支创建，但runner没有发送`approx_kv` metadata，R0/R1恢复路径并未执行。
- 因此Phase5当前结论只是“exact-cache scheduler policy isolation”；不是“五条Phase4 recovery workload × 五条scheduler”的组合矩阵。
- 真正组合实验应在Phase6执行：从Phase4选前两条恢复路径，再与S4/P0及必要baseline联测，避免完整五乘五笛卡尔积。

## 2026-07-24T14:54:28-07:00 Phase 5是否有损恢复及pressure定义

- Phase5 workflow明确没有执行有损KV恢复：
  - 无`approx_kv register/reuse`；
  - 无raw+RoPE、EPIC、CacheBlend、KVCOMM或CacheTune target；
  - 所有workflow hit均为exact Radix/HiCache hit，miss走普通dense prefill。
- Phase5 pressure针对的是SGLang可分配/可淘汰的GPU KV token pool，不是把整张GPU物理VRAM字节全部占满。
- server使用`mem_fraction_static=0.35`；运行时metrics测得usable KV capacity约`13,130` tokens。
- runner从实际allocator/cache metrics读取capacity，再选择对象使working set oversubscribe；不是只用理论VRAM估算。
- commit-bound实际rho为：
  - target1.1 -> `1.153`，15个对象；
  - target1.5 -> `1.537`，20个对象；
  - target2.0 -> `2.075`，27个对象；
  - target3.0 -> `3.073`，40个对象。
- 所有正式setting均发生真实Radix eviction。S0累计evicted tokens约：
  - rho1.1 `32,074`；
  - rho1.5 `49,234`；
  - rho2 `105,064`；
  - rho3 `169,852`。
- S4对应约`10,678/34,002/69,628/134,412` evicted tokens。
- 每个setting结束后KV pool accounting与flush/reset invariant通过，因此pressure是“真实超出KV pool并反复淘汰”，不是OOM或假压力。
- 该pressure只能证明exact-cache scheduler在KV pool oversubscription下有效；不能证明scheduler对Phase4 approximate KV store/source objects有效。

## 2026-07-24T15:10:08-07:00 Phase 6修订计划

### 为什么原Phase 6必须修改

- 原计划“Phase4前两条恢复路径 × S0-S4 × P1-P3”不再合适：
  - Phase5只验证了exact Radix，未管理approx store；
  - S1-S3在高压下未稳定优于S0，S4才是主策略；
  - P2/P3在sequential下无稳定收益且p95更差；
  - Phase5 rho sweep改变了对象组成；
  - Phase4 target-only、combined和production readiness并不一致；
  - R2/R5仍是precomputed adapter。

### Phase 6新的核心目标

在同一固定sequential workflow、同一逻辑对象集合和同一GPU预算中，让exact Radix对象与Phase4 approximate source/adapter/anchor对象真实竞争，比较：

```text
dense
vs exact S0/S4
vs lossy recovery S0/S4
vs S4 + HiCache demand load
```

只做速度/系统结论，不扩展semantic quality claim。

### 计划改动

1. 不做五条recovery乘五条scheduler的完整笛卡尔积。
2. 主scheduler只保留S0与S4；S2只做oracle-style victim诊断。
3. 主流程固定P0；P1-P3继续后置，直到有真正异步overlap或并发隐藏H2D。
4. 先把S4 metadata、eviction和admission接入approx store，不再只管理exact Radix。
5. 建立cross-store reservation/victim/commit/rollback协议，统一exact与approx allocator pressure。
6. 固定同一逻辑对象集合和顺序，只改变capacity扫rho；不再通过增加对象改变rho。
7. 同时报告logical rho与真实physical rho/pages/bytes，包含raw/fresh/anchor/delta多份表示。
8. Phase4候选分轨：
   - R0/k0：speed-only ceiling；
   - R1/k32：genuine in-request practical candidate；
   - R2/1%：precomputed oracle track，未online前不参与production winner；
   - R4：anchor结构诊断点；
   - R5当前被R2支配且combined margin小，不进主矩阵；
   - R3继续defer。
9. 选择路径不再只看target-only TTFT；使用完整workflow、combined cost、p95、fallback和lifecycle。
10. warm-up继续执行但排除formal，同时单独保存cold-start sample。

### Phase 6完整执行计划

#### P6-0：冻结公平性合同

- 固定workflow：`Architect -> Coder -> Debugger -> Coder -> Debugger`。
- 固定5个active workflow对象与固定live/dead filler类别、ID、顺序。
- body主点=`1024/2048`，header=`64`，long source segment=`<=512`。
- 主口径为matched-state：
  - approximate target不写回exact Radix；
  - exact baseline使用变化的target suffix，避免后续轮次退化为完整exact hit。
- 另保留native-system口径，单独展示真实写回行为。

#### P6-1：cross-store对象模型与原子分配

- 定义真实object DAG/state：
  - exact stage variant/bundle；
  - canonical raw segment chain；
  - EPIC repair state；
  - R2 fresh adapter；
  - R4 canonical/anchor/delta；
  - host copy。
- metadata进入approx store/handle，不只进入Radix node。
- pin、lease、in-flight H2D、active recovery reservation不可驱逐。
- allocation使用：
  1. reserve；
  2. 在exact+approx候选中选victim；
  3. 原子evict/demote；
  4. commit；
  5. 失败rollback或dense fallback。
- R0 segmented chain按连续可恢复前缀原子计价；禁止保留无用孤儿suffix。
- reset/fallback后验证generation、lease、slot、host ref全部归零。

#### P6-2：固定workload与capacity feasibility pilot

- 保持40个逻辑对象和类别固定，但一次性校准filler长度，使最低压力点在SM75安全capacity内可达。
- 优先通过`max_total_tokens`改变KV capacity；若接口不稳定才使用`mem_fraction_static`。
- 先用R0+S0、body2048做单点pilot，验证单请求可运行且rho1.1–3物理可达。
- setting固定同一对象集合；只改变capacity。
- 记录：
  - `rho_logical`；
  - `rho_physical_exact`；
  - `rho_physical_approx`；
  - total device/host pages。

#### P6-3：恢复路径重新筛选

- 候选：R0、R1-k32、R2-1%、R4结构诊断。
- setting：body1024/2048，logical rho1.5/2，S0、T0、P0。
- R0与R1-k0不得作为两个独立候选。
- R2必须把fresh preparation放入计时；未online时标为oracle/precomputed。
- R4报告setup以及N=`1/2/4/8` amortized workflow cost。
- 每格warmup1、formal2、独立完整round。
- 选择：
  - R0固定保留为speed ceiling；
  - 另选一个production-practical winner；
  - R2只在实现online lifecycle后才可成为practical winner。

#### P6-3.5：scheduler revalidation gate

- cross-store接线和恢复路径重新筛选完成后，先不直接删除S1-S3。
- 对最终practical recovery运行：
  - body2048；
  - rho1.5与rho3；
  - S0-S4；
  - warmup1、formal2、单restart。
- 对R4 canonical/anchor/delta结构在body2048、rho2另跑一次S0-S4，检查真实object DAG上的victim sequence。
- S1/S2/S3满足任一条件则重新进入主scheduler矩阵：
  - 相对S0 mean改善>=5%，且p95恶化<=5%；
  - fallback率、physical footprint或victim correctness明显优于S4。
- 未通过资格赛的策略不进入大矩阵，但保留完整诊断结果。

#### P6-4：主scheduler组合矩阵

- 默认矩阵为两条入选路径 × `{S0,S4}` × T0/P0。
- 若P6-3.5中S1/S2/S3重新获得资格，则只把通过门槛的策略加入，而不是自动恢复全部S0-S4。
- body1024只跑rho1.5/2作为control。
- body2048跑rho1.1/1.5/2/3。
- 共享baseline按body/rho只测一次：
  - D0：dense，无reuse；
  - E0：exact S0、GPU-only、P0；
  - E4：exact S4、GPU-only、P0。
- S2只在body2048的rho1.1与rho3做victim-sequence诊断，不进入主排名。
- setting顺序使用blocked random/Latin-square，仍保持一次只运行一个GPU server。

#### P6-5：HiCache demand-load矩阵

- 只对最终practical winner执行。
- 固定S4+P0、body2048、rho1.5/2/3。
- baseline H4：exact S4+HiCache+P0。
- approximate与exact使用相同GPU/host budget和write policy。
- 验证approx source真实demote到host、后续demand load或安全fallback。
- 该矩阵先以P0建立可归因的demand-load baseline。

#### P6-5.5：prefetch revalidation gate

- final practical winner完成S4+HiCache+P0 demand-load后，运行：
  - body2048；
  - rho2与rho3；
  - P0-P3；
  - warmup1、formal2、单restart。
- 如果实现仍是同步等待H2D：
  - P1-P3只作为功能、安全和churn canary；
  - 不发布正式prefetch性能claim。
- 只有实现真正async H2D，并能与当前stage执行或GPU idle窗口重叠时，才进入正式性能比较。
- P1/P2/P3重新进入后续主结果的条件：
  - mean相对P0改善>=3%；
  - p95不恶化；
  - wasted/churn bytes受控；
  - 不驱逐next-use更早的高价值对象。
- 未通过时，P0继续作为Phase6最终默认。

#### P6-6：metrics与统计

- 客户端：
  - workflow wall-clock；
  - mean/p50/p95 TTFT；
  - 每角色Architect/Coder/Debugger；
  - 每restart paired delta。
- cache分类：
  - exact hit；
  - approximate hit；
  - host demand load；
  - dense fallback。
- recovery：
  - source/register preparation；
  - intent-to-treat target TTFT，包含fallback；
  - effective contiguous recovered tokens；
  - N=`1/2/4/8` amortized cost与break-even。
- memory：
  - exact/approx各类resident bytes/pages；
  - victim count/tokens/bytes by object kind；
  - H2D bytes/time；
  - wasted objects/orphan segments必须为0。
- warm-up/cold-start：
  - warm-up完整保存但不进formal；
  - 另报告cold-start sample；
  - warm-up后清空exact/approx/HiCache/metadata/generation。

#### P6-7：restart与最终门禁

- screening格先做1次server process × formal2。
- 最终primary cells做3次独立server restart，每次formal2。
- 独立统计单位是server restart，不把同trace内请求伪装成独立实验。
- 最低有效性：
  - 100%请求完成，无OOM/allocator corruption；
  - exact+approx+host pool accounting通过；
  - 至少观察一次approx source被evict/demote，后续成功恢复、load或dense fallback；
  - S4相对同恢复路径S0在至少两个rho>=1.5点mean改善>=5%；
  - p95不得恶化超过5%，否则不声明scheduler收益。
- 若不满足，诚实结论为S4对lossy store无增益，不继续扩大矩阵。

### 预期交付

- unified cross-store policy实现与测试；
- fixed-trace/capacity-sweep runner；
- recovery re-screen JSON；
- S0/S4主矩阵；
- HiCache demand-load矩阵；
- 三restart compact manifests；
- 明确区分speed ceiling、precomputed oracle和practical winner的结论。

（历史记录，已过时：当时Phase6仅完成计划修订。当前为V5 Current / Latest，Phase6 technical Exit已通过。）

## 2026-07-24T15:43:46-07:00 S1-S3与P1-P3不彻底取消，增加revalidation gate

- 用户质疑Phase6为何主矩阵只保留S0/S4、P0。
- 原缩减依据：
  - Phase5 rho2/3时S1/S2/S3 mean speedup相对S0约为：
    - S1 `1.006/0.994x`；
    - S2 `0.999/0.996x`；
    - S3 `1.011/0.990x`；
    - S4 `1.146/1.132x`。
  - Phase5 P1没有主动load；P2/P3有真实load/eviction但无稳定mean收益，且rho3 p95从P0约`154.47ms`恶化到P2/P3约`158.96/160.62ms`。
- 但Phase5只测exact cache，不能证明S1-S3/P1-P3在真实approx objects上永远无效。
- 因此修订为“主矩阵缩减 + 小矩阵重新资格赛”，不是彻底删除。

### 新增P6-3.5：scheduler revalidation gate

- cross-store接线并完成recovery re-screen后，选择最终practical recovery：
  - body2048；
  - rho1.5与rho3；
  - S0-S4全部运行；
  - warmup1、formal2、单restart。
- R4 anchor结构另在rho2运行一次S0-S4诊断，检查真实canonical/anchor/delta victim sequence。
- S1/S2/S3满足任一条件则重新进入主矩阵：
  - 相对S0 mean改善>=5%且p95恶化<=5%；
  - 在fallback、physical footprint或victim correctness上提供S4没有的明确收益。
- 否则主矩阵继续只做S0/S4。

### 新增P6-5.5：prefetch revalidation gate

- final practical winner完成S4+HiCache+P0 demand-load后：
  - body2048；
  - rho2与rho3；
  - P0-P3小矩阵。
- 如果仍是当前同步prefetch实现：
  - P1-P3只做功能/安全canary；
  - 不作为正式性能结论。
- 只有实现真正async H2D，并能在当前stage执行期间或GPU idle窗口重叠时，才进入formal性能矩阵。
- P1/P2/P3重新进入主结果的门槛：
  - mean相对P0改善>=3%；
  - p95不恶化；
  - wasted/churn bytes受控；
  - 不驱逐更早使用的高价值对象。
- 因此P0仍是主流程默认，用于隔离recovery与eviction归因；其他P保留但需重新获得资格。

## 2026-07-24T18:20:12-07:00 Phase 4–6 双代理独立审计与交叉汇总完成

- 用户要求两个1M-context配置彻底复核Phase 4/5实验、判断是否需要重跑，并评估最新Phase 6计划。
- 实际调用配置：
  - sub-agent A：GPT-5.6 Sol、`long_context`、max reasoning；
  - sub-agent B：Claude Opus 5、`long_context`、max reasoning。当前工具没有字面上的“Opus 5.5”，因此使用最近可用的Opus 5配置，并已向用户透明说明。
- 两代理均只读审计了：
  - `HANDOFF.md`、`PROJECT.md`、`TRACKING.md`、`IMPLEMENTATION_PLAN_LATEST.md`、`research/RESEARCH_SYNTHESIS.md`；
  - R0–R5与scheduler-policies全部worktree、指定提交和相关runner/runtime/policy/test；
  - Phase 4/5全部compact JSON、可用raw result、server log与中央`BENCHMARK_RUN_LOG.jsonl`。
- 两代理先独立报告，再全文交换给对方交叉复核。最终交叉register包含`C-01`至`C-65`共65条去重建议，并保留6个未决分歧：
  1. 固定40对象、只调capacity的S0/S4矩阵应立即作为Phase5补跑，还是并入Phase6 P6-2；
  2. R5 combined在不同成本边界和dense分母下的权威口径；
  3. R0是否还需自己的完整矩阵，或R1-k0机制等价代理已足够；
  4. 已发布R1-k32 pressure结果是否需要定向重抓EPIC机制counter；
  5. prefetch是否值得补跑，以及必须先去除host-tier饱和；
  6. long-body分段注册缺跨chunk causal attention对R0/R1/R4应定性为缺陷还是诚实限制。
- 交叉审计的重要事实/推断：
  - Phase5正式`workflow_summary`只统计20个workflow请求；改用全部可复用请求或全trace后，S1/S2/S3与S4的相对排序发生变化。`S4 > S0`仍可在两类口径下成立，但“唯一稳定优于S0”和“Belady上界未优于S4”需要按分母重新表述。
  - Phase5 S4的object kind由轮转标签构造，不能直接外推为真实approximate-object DAG的验证。
  - Phase5 prefetch的host容量大于工作集，P0整条trace无miss；同步H2D又可能把成本落在相邻请求间隙，因此现有结果更适合作功能/安全/开销canary。
  - Phase4 body crossover与dense侧1024-token chunk边界共线；现有数据能提示但不能完全分离body长度和chunking因果。
  - R0/R1/R2/R4共用同一dense分母，R5使用另一次dense；R5 body2048 combined会随成本ledger和分母选择落在约`0.698–1.113x`区间，现有`1.041x`不应被当作唯一权威点估计。
  - R2/R5 long-body fresh adapter按`header + 当前chunk`构造，缺少此前body chunk的target causal context；R0/R1 source注册也使用同构分段方式，但其语义定性仍属未决项。
  - 双方一致认为不需要完整重跑Phase4/5；建议分成0-GPU重算/文档治理、决定R2/R5结论的key rerun，以及按claim触发的targeted rerun。
  - Phase6待评审建议集中在matched-state exact-hit控制、state ledger/provenance、cross-store byte budget与反向压力、DAG原子闭包、R2/R4 worst-footprint pilot、统一cost ledger、quality guardrail、block-paired baseline、host pressure、async前置条件及统计门槛。
- 本轮没有接受或执行这些建议，没有修改`IMPLEMENTATION_PLAN_LATEST.md`，没有修改prototype，也没有启动新GPU实验。下一步由用户先审阅最终65条建议及主代理逐条点评。

## 2026-07-24T21:38:03-07:00 R2/R5 corrected causal key rerun完成

- 用户要求把合并审计写入文件，并按审计建议重跑R2与R5；固定文件为`CONSOLIDATED_PHASE4_PHASE6_REVIEW.txt`。
- 共同修复合同：
  - 每个raw/fresh chunk先在独立`extra_key`命名空间中增量dense物化`header + body[:chunk_end]`；
  - 再从该exact KV注册当前<=512-token chunk及其真实绝对位置；
  - 真实target仍只使用默认cache命名空间，因此exact prefix严格停在64-token target head；
  - dense/recovery在同一server restart内配对，分别报告target-only、adapter-combined、request-path和full-lifecycle；
  - 每条路径3个server restart，每臂warmup1+formal2，首tokenguardrail、telemetry、eviction和pool reset全部落盘。
- 调试过程中确认：
  - 直接发送累计长register prompt会在SM75 chunked-prefill路径OOM；
  - 增量dense物化避免一次性长extend；
  - register请求的最大exact hit为`prompt_len-1`，因为最后一个prompt token必须真实forward；
  - R5最初误启用host tier会在pressure下引入额外H2D并OOM；正式合同固定GPU-only。
- R2：
  - 实现提交`c73c9c5ab3ab705996c0ff901314a5fe41e1f8a6`；
  - 结果提交`e36f1529b838c12a9eb2af7ba4dde91ae9ec124b`；
  - 完整结果`benchmark/approx_kv/results/phase4-r2/sm75-causal-key-rerun.json`；
  - raw目录`/home/chris/Workspaces/kvcache-research/results/phase4-r2-corrected-c73c9c5ab/`；
  - body1024：target-only`1.659x`、adapter-combined`0.441x`、request-path`0.526x`、full-lifecycle`0.324x`；
  - body2048：`2.044x/0.407x/0.434x/0.246x`。
- R5：
  - 实现提交`46d1f85c22a98b7305b4f3ef299da56c65d2a025`；
  - 结果提交`abcedd62b5a5d801742734e300a5df21e1436737`；
  - 完整结果`benchmark/approx_kv/results/phase4-r5/sm75-causal-key-rerun.json`；
  - raw目录`/home/chris/Workspaces/kvcache-research/results/phase4-r5-corrected-46d1f85c2/`；
  - body1024：target-only`1.614x`、adapter-combined`0.449x`、request-path`0.527x`、full-lifecycle`0.327x`；
  - body2048：`1.978x/0.406x/0.433x/0.246x`。
- 两条路径所有formal pair首token一致率均为1.0；0 dense fallback；所有round有真实eviction；三次server的pool reset invariant均通过。
- 当前结论尚未经过用户要求的同配置双代理复核；`IMPLEMENTATION_PLAN_LATEST.md`仍未修改，Phase6仍未启动。

## 2026-07-24T22:08:11-07:00 R2/R5 post-rerun双代理复核完成

- 原GPT-5.6 Sol与Claude Opus 5代理实例被重新唤醒，分别独立核查新runner、实现/结果commit、全部raw样本和审计文件；之后全文互换并交叉consolidate。
- 固定review文件`CONSOLIDATED_PHASE4_PHASE6_REVIEW.txt`已更新，包含完整post-rerun结果、原C-01至C-65状态变化、新PRC-01至PRC-23及最终assessment。
- 双方一致：
  - causal-prefix修复真实生效；
  - committed JSON与raw逐字节一致，四类ledger算术全部闭合；
  - R2/R5旧target-only收益成立；
  - R2 body2048 combined从`1.14x`修正为`0.407x`；
  - R5 body2048从`1.04x`修正为`0.406x`；
  - 两者仍是precomputed target oracle，不是practical candidate；
  - 相同矩阵无需再次强制重跑。
- 交叉新增解释：
  - R2修复1% tokens，R5修复约8.3%；两者target差异约`0.128–0.146ms/selected token`，与R2历史ratio sweep边际成本一致；
  - 将R2外推到85 selected tokens得到约`190.7ms`，与R5实测`190.694ms`吻合；
  - 因此“R5被R2性能支配”不再成立，应改为“与R2 precomputed track高度冗余且无practical ledger收益”。
- 剩余量化缺口：
  - recovery setup额外留下`2250/4306` evictable exact-namespace tokens，filler预算只扣used tokens，approx侧eviction多约`10.6–22.4%`；
  - R5两臂的filler salt含arm label，token内容不同；
  - R2 fallback metric为`None`，不能把`None in (None,0)`写成显式0 fallback证据；
  - `full_lifecycle`不含pressure/server/cleanup；
  - 当前rho2结果不证明rho全轴稳健性；
  - first-token仅一个greedy token，不构成semantic quality。
- 公式摊销投影：
  - body2048 fresh-only约N=4 break-even，含raw约N=8；
  - body1024约N=5/N=9；
  - 当前两请求materialize+register协议中，register阶段约占48–49% setup时间，但包含真实KV copy，不能全部视作可删除协议开销。
- 结果已用显式`ccdd2023`身份push并核对远程SHA：
  - `research/cacheblend@e36f1529b838c12a9eb2af7ba4dde91ae9ec124b`
  - `research/cachetune@abcedd62b5a5d801742734e300a5df21e1436737`
- `IMPLEMENTATION_PLAN_LATEST.md`未自动修改，Phase6未启动。若未来保留R2/R5排序claim，最小新增实验必须matched ratio、matched pressure和共同filler manifest；否则无需再次重跑当前矩阵。

## 2026-07-25T00:12:14-07:00 Consolidated review point状态复核

- 用户要求逐项检查新的consolidated review，明确哪些review point已经update完成。
- 当前严格分类：
  - 完整完成/被新结果正式取代：`C-04`、`C-12`、`C-14`，以及`C-13`的R2/R5部分；`PRC-13`已完成。
  - 部分完成：`C-03/C-05/C-08/C-09/C-10/C-11/C-15/C-19/C-43/C-49/C-50/C-56/C-61/C-62/C-63/C-64`。
  - 有新证据但仍未完成：`C-06/C-16/C-17/C-18/C-29/C-33/C-34/C-35/C-39/C-47/C-52/C-55/C-57`。
  - 其余原始C项没有因R2/R5重跑自动完成。
- 新增`PRC-01`至`PRC-23`中，只有远程持久化`PRC-13`已经落实；其余均仍是待执行、条件执行或计划建议。
- 本轮只做状态解释，没有修改latest Phase6计划、prototype或实验结果。

## 2026-07-25T10:53:22-07:00 实施计划V3完成双模型review并定稿

- 将原`IMPLEMENTATION_PLAN_LATEST.md` V2归档为`IMPLEMENTATION_PLAN_V2_ARCHIVED.md`；V1继续保留。
- 新`IMPLEMENTATION_PLAN_LATEST.md`为V3，已标记`Current / Latest`。
- 使用用户指定的两个review配置：
  - GPT-5.6 Sol / Max Thinking / long context；
  - Claude Opus 5 / Max Thinking / long context。
- 两模型独立review后全文互换，形成两份cross-consolidated draft；VA-01..25、VB-01..37全部映射到VC-01..40，无静默丢弃。
- 双模型主要分歧及最终处置：
  - G1不阻塞P6-0/P6-1，移为P6-3a并阻塞实验主线；
  - G2不阻塞P6-1，改为primary性能矩阵前“执行或显式waive”的chunk配置门；
  - R4不加入G1，只按V3合同独立重测且不参与practical winner；
  - host basic demand-load不强制`rho_host>=1`，但host eviction/admission claim必须至少一个`rho_host>=1` cell。
- V3新增/修正：
  - Implementation Entry与Experiment Entry；
  - R1 candidate family、预注册promotion rule和`practical family=NONE`分支；
  - 唯一matched-state方案；
  - paired launch block定义；
  - 逐路径ledger公式及`protocol_overhead=not_measured`守卫；
  - warmup/formal间完整flush/reset；
  - exact/approx/host-load/fallback四分类；
  - logical/physical-demand/resident/host rho；
  - unified event ordinal、S4 class/value/tie-break；
  - lock order、failure injection和rollback语义；
  - extra-key无flush GC与ledger/pressure CPU tests；
  - all-reusable p95恶化`<=5%`全局规则；
  - R4独立diagnostic、N连续8-target序列；
  - write-through host feasibility、prefetch最小setting和churn阈值；
  - 双模型accepted-blocking-P0、override和model fallback规则；
  - logical cells/server startups/rounds/GPU-hour预算及early-stop。
- Phase6开始前的不确定性结论：
  - Implementation Entry前无需新的Phase4/5 GPU重跑；
  - 必须完成G0 0-GPU收尾、authority文档版本化和本计划review disposition；
  - P6-3前必须跑R0/R1 k qualification及R0/R1-k0等价检查；
  - chunked-prefill配置必须实测决定或显式waive并缩窄claim；
  - R2/R5 matched ratio、rho稳健性、R2显式fallback补点均为条件性。
- 当前G0未完成，因此Phase6 Implementation Entry仍blocked；未创建Phase6实现分支。
- 历史V2条目中的`T0`统一解释为`tier = GPU-only`；V3不再使用`T0`缩写。

## 2026-07-25T11:09:10-07:00 V3最终delta verification通过

- 同一GPT-5.6 Sol Max Thinking与Claude Opus 5 Max Thinking实例对最终V3及同步文档进行delta-only复核。
- 两模型均确认：V3定稿阻塞P0已闭合，可维持`Current / Latest`；G0待执行属于Implementation Entry门，不是计划文本缺陷。
- 修正最终非阻塞errata：
  - `rho_demand`统一为`rho_logical_demand`；
  - `amortized_ms_N`公式展开，不再引用未定义`steady_target_path`；
  - 追加历史`T0 = tier GPU-only`解释；
  - G1/G2命名收敛为P6-3a/P6-3b。
- Phase6仍未开始，Implementation Entry仍由G0阻塞。

## 2026-07-25T16:29:22-07:00 V3前四项门禁上下文澄清

- 向用户逐项解释：
  1. Implementation Entry前无需新GPU重跑，但G0是结果治理、schema、测试和authority文档版本化门；
  2. P6-3a的R0/R1 k qualification用于决定是否存在真正practical family，不阻塞P6-0/P6-1；
  3. P6-3b chunk配置门决定Phase6数字继承的`chunked_prefill_size`，可执行或显式waive并缩窄claim；
  4. P6-2 fixed-40 pilot在cross-store实现后统一验证capacity/footprint，因此无需先独立重跑Phase5。
- 本轮仅做解释，没有修改V3、代码或实验状态。

## 2026-07-25T16:37:23-07:00 Phase6/Phase7重新划分建议（待确认）

- 用户指出当前V3把过多不同性质工作放进Phase6，并明确前阶段补跑/收尾不应单独成为新phase。
- 建议采用并行closeout lane加两阶段主线：
  - Phase4/5 Closeout Lane：G0、R0/R1新合同确认、chunk配置、Phase5零GPU重算；不新增phase编号，可与Phase6工程并行，但必须在Phase7前完成。
  - Phase6：Cross-Store Substrate & Feasibility，只负责合同、对象DAG、统一allocator/eviction/rollback、lifecycle/GC及fixed40 capacity pilot，不做winner或性能claim。
  - Phase7：Integrated Recovery × Scheduling Evaluation，负责candidate freeze、N摊销、scheduler主矩阵、HiCache demand-load、prefetch及最终结论。
- 建议把当前V3的P6-0/P6-1/P6-2保留到新Phase6，把P6-3a至P6-5.5整体迁移到Phase7。
- 暂不预建Phase8；只有host/prefetch工作在Phase7 feasibility后确认独立膨胀时再拆分。
- 本条为讨论建议，尚未修改latest plan。

## 2026-07-25T23:47:04-07:00 V3归档、V4双模型review并定稿

- 用户确认将Phase4/5补跑作为closeout lane，不新增phase，并将原Phase6拆成Phase6底座与Phase7集成评测。
- V3归档为`IMPLEMENTATION_PLAN_V3_ARCHIVED.md`；V4成为`IMPLEMENTATION_PLAN_LATEST.md`，状态`Current / Latest`。
- V4主结构：
  - Closeout CL0–CL4可与Phase6的0-GPU工程并行，但单卡GPU任务串行；
  - Phase6只做cross-store substrate、correctness、lifecycle和fixed40 feasibility，不做性能winner；
  - Phase7做recovery、scheduler、HiCache、prefetch和最终结论；
  - Phase8仅定义Potential Scope及触发条件。
- GPT-5.6 Sol Max Thinking与Claude Opus 5 Max Thinking完成两轮独立review、交叉consolidate和最终delta验证。
- 交叉review发现并修复：
  - CL2→P6-4配置依赖；
  - generic host roundtrip canary；
  - unified manifest/schema；
  - rho公式、per-path ledger和break-even；
  - shared memory/transfer指标；
  - R1-like worst-case footprint；
  - practical=NONE完整停止分支；
  - PR-S0/PR-S4命名；
  - R4 rho2 S0-S4 victim diagnostic；
  - review mapping和分阶段GPU预算。
- generic host canary命名为`P6-H`，避免复用历史V3的`P6-5` HiCache标签。
- 当前CL0尚未完成，因此Phase6 Entry继续blocked；未创建Phase6实现分支。

## 2026-07-29T01:35:14-07:00 Phase4–7正式报告集完成

已新增五份自包含、可审计的正式文档：

- `research/phase_reports/PHASE4_RECOVERY_METHODS_REPORT.md`
- `research/phase_reports/PHASE5_WORKFLOW_SCHEDULING_REPORT.md`
- `research/phase_reports/PHASE6_CROSS_STORE_SUBSTRATE_REPORT.md`
- `research/phase_reports/PHASE7_INTEGRATED_EVALUATION_REPORT.md`
- `research/phase_reports/PHASE4_TO_PHASE7_SUMMARY.md`

报告统一覆盖实验、矩阵、执行顺序、问题修复、lesson、最终结论、可证伪
预测、限制与artifact/provenance。总报告额外覆盖整体动机、最初问题、架构
演化和跨阶段方法论反转。

写作与审阅：

- 用户指定的Opus 3.5 Thinking当前不可用，使用最接近的
  Claude Opus 5 Max Thinking执笔；
- 用户指定的GPT-4o Max当前不可用，使用GPT-5.6 Sol Max全文审阅；
- 初审为`FAIL / NOT_READY`，findings=`0 P0 / 9 P1 / 12 P2`；
- 完成两轮closure后最终为
  `PASS_WITH_CAVEATS / READY_WITH_CAVEATS`，open P0/P1/P2=`0/0/0`。

关键修正包括：

- Phase6 Exit gate汇总改为3项`satisfied`，总数闭合为10；
- Phase5删除S2/S4排序与prefetch普遍无收益表述；
- Phase7 R0 NEGATIVE严格限定于R0/chunk4096，不外推未测机制；
- CL2只声明coupled chunk/max-prefill effect，不宣布全局主导变量；
- Phase6 host使用独立limit，不与device对象共用同一GPU预算；
- 无数据支撑的定向预测改为条件预测或预注册研究问题；
- Phase7说明89个physical files与88个manifest entries的self-exclusion；
- source pin、summary-binding commit与publication HEAD分层陈述。

本轮只整理和审查既有证据，没有运行GPU或server。被引用的Phase4–7实验
仍全部在固定Docker环境中执行。

用户指定的下一项工作：

- 审查合作者分支
  `review/coding-aware-v40-prefetch-20260729`；
- 解释其有损KV恢复方法；
- 映射最接近的既有研究或组合来源；
- 设计Docker-only正确性、性能、质量、压力、chunk、fallback、
  scheduling与prefetch实验；
- 给出如何纳入当前实验taxonomy与未来版本化计划的完整集成方案。

该分支审查不得改写已冻结的Phase4–7历史artifact，也不得把新方法追写成
旧阶段已完成能力。

## 2026-07-29T02:18:08-07:00 V40分支审查发现阻断性失效漏洞

审查对象：

- branch=`review/coding-aware-v40-prefetch-20260729`
- pin=`13671eb708da689137a654946b0d34ba924efb29`
- merge-base=`3343a79466aa714d34a14d08d3929f7953a47212`

初步机制结论：

- V40选择一个来自真实前序agent请求的successful read-only tool observation；
- 要求token-identical、target中唯一、strict middle、未被后续同路径write失效；
- 执行时copy V，对K做source→target RoPE delta，岛外dense；
- 恢复primitive最接近R0 Raw+RoPE；不是prefetch、KVCOMM reconstruction或
  CacheBlend selective repair。

阻断性finding：

- V40的失效路径依赖命令文本正则；
- Docker对抗测试证明以下later same-path write均未被识别，candidate仍为
  `eligible=1`、`invalidated=0`：
  - `cat > pkg/a.py`
  - `echo x > pkg/a.py`
  - `perl -pi ... pkg/a.py`
  - `dd ... of=pkg/a.py`
  - `truncate -s 0 pkg/a.py`
  - `git mv pkg/a.py pkg/b.py`
  - `rsync ... pkg/a.py`
- mixed group `cat pkg/a.py; echo x > pkg/a.py`、`sed ...; perl -pi ...`、
  `head ...; truncate ...`甚至被
  `is_successful_readonly_evidence=True`直接接受。
- server端只验证token/model/generation，不理解repository path，因此没有
  第二道file-version防线。

该问题违反分支review request自己的硬不变量：

- mutating interaction不得成为reusable observation；
- later write必须使同路径read失效。

当前处置：

```text
branch_approval = NOT APPROVED AS-IS
integration = BLOCKED
```

推荐根修不是继续扩充deny-list正则，而是由tool wrapper提供结构化
`read_paths/write_paths`、repo/worktree generation和path content hash；
未知effect必须fail closed，并增加对抗/property tests。

其它已确认缺口：

- `coding_aware/policy.py::build_coding_reuse_plan`没有生产调用者；
- `SGLANG_CODING_AWARE_LOSSY`当前不是实际runtime gate；
- lease TTL需要显式调用无人接线的GC，异常漏`unpin`可能永久阻止驱逐；
-详细raw/result与integration-v2证据只在作者绝对路径或不可获得commit中，
  当前无法独立复现；
- 分支与当前cross-store底座严重分叉，不能直接merge旧store/manager/scheduler。

Docker只读验证：

- policy/selector/KV core/radix：`66 passed`；
- bridge adapter：`8 passed`，但依赖需临时安装且resolver报告冲突；
- branch scope：`OK`，scope tests=`3 passed`；
- self-contained campaign/schema tests：`12 passed`；
- V40A2/V40A3两个test因硬编码、缺失的`/home/gfy/...` artifact失败。

完整技术报告、实验可行性和集成计划正在撰写。

## 2026-07-29T05:46:31-07:00 V40完整技术报告封版

正式报告：

`research/CODING_AWARE_V40_BRANCH_TECHNICAL_REPORT.md`

报告共2266行，完整回答：

1. V40如何做有损KV恢复；
2. 与R0、Prompt Cache/PIC、EPIC、CacheBlend、KVCOMM、KVFlow的关系；
3. selector、tensor、correctness、quality、latency、pressure、scheduler、
   prefetch与provenance测试可行性；
4. 如何以`C40 = G40 selector × R0 executor`纳入未来版本化实验设计。

最终方法定位：

> V40的恢复primitive是R0 Raw+RoPE：从真实近期agent请求选择一个
> token-identical的read-only tool observation岛，copy V、对K做RoPE
> relocation，岛外dense。其主要新增点是grounded selection/invalidation
> policy，不是新的恢复公式，也不是prefetch、KVCOMM reconstruction或
> CacheBlend selective repair。

分支判定：

```text
branch_approval = NOT APPROVED AS-IS
integration = BLOCKED
```

两个阻断类别均已在固定Docker image中复现：

- 常见same-path write漏检；
- 同一group的mixed read/write被当成readonly evidence。

影响边界：

- 这不是old-KV配new-token或data corruption；target仍包含同一历史文本，
  token identity与机械校验可通过；
- 真正违反的是freshness/abstention policy：本应放弃的语义过时historical
  observation被送入lossy reuse路径。

其它重要发现：

- active链路是selector→bridge→sidecar→`kvcomm_exact`→transfer backend；
- `coding_aware/policy.py`是未接线seam；
- `SGLANG_CODING_AWARE_LOSSY`不是实际runtime gate；
- lease TTL没有生产GC调用；
-历史end-to-end raw、integration-v2 refs与113-test证据当前不可复现；
- 当前cross-store只支持从exact boundary连续恢复；C40严格middle island需要
  新的`DENSE_PREFIX→COPY_READY→DENSE_SUFFIX`controller；
- rolling request必须同时维护consume/produce双角色、approx provenance和
  request-lifetime状态；不得在每轮`init_next_round_input`清空。

Docker验证：

- policy/selector/KV core/radix=`66 passed`；
- bridge adapter=`8 passed`；
- branch scope=`OK`，scope tests=`3 passed`；
- self-contained campaign/schema=`12 passed`；
- three-method/RepoBench summarizer=`8 passed`；
- `kvcomm_exact` runtime=`23 passed`；
- V40A2/V40A3=`2 failed`，原因是缺失硬编码external artifact。

报告review：

- GPT-5.6 Sol Max首轮发现`0 P0 / 8 P1 / 5 P2`；
- closure review继续发现middle-span lifecycle、完整workload estimand、
  task repeat聚合与依赖baseline问题；
- 全部修复后最终为
  `PASS WITH CAVEATS / READY WITH CAVEATS`，open report
  P0/P1/P2=`0/0/0`。

建议集成方式：

- 不merge整个82-commit/52k-line research branch；
- 不带入旧`kvcomm store/manager/scheduler`；
- 在当前`approx_kv/cross_store`底座重实现结构化selector/provenance；
- 复用现有copy/RoPE与cross-store核心语义，新增middle-span request seam；
- 新candidate ID=`C40 = G40 × R0`；scheduler与prefetch分别为正交轴；
- 不改写Phase4–7，创建新的versioned plan/manifest。

建议执行Tracks：

- Track A：zero-GPU code/provenance/middle-span修复；
- Track B：小规模Docker GPU pilot，建议hard cap估计`≤8 starts / ≤2 GPUh`；
- Track C：只有pilot后另行授权才进入confirmatory/workflow/quality；
- Track D：获取并版本化external raw与prefetch/integration refs。

**所有Track均为建议，当前`PENDING USER AUTHORIZATION`。**

## 2026-07-29T14:45:40-07:00 V40实现成熟度与功效评价

针对用户对实现状态、speedup和工程复杂度的追问，当前评价如下：

1. **实现成熟度：机制链路可运行，但不可直接采用。**
   - active链路已经覆盖selector、bridge、sidecar、middle-KV controller与
     copy/RoPE backend；
   - focused/unit Docker测试可复现并通过；
   - 但selector存在两类P0 freshness/abstention漏洞，历史end-to-end raw与
     composition refs不可复现，feature gate、lease GC和依赖锁也未闭合；
   - 因此它是research prototype，不是production-ready implementation。
2. **现有speedup只属于作者外部、未验证的诊断性声称。**
   - V44 cohort声称Dense `357.6ms`→V40 `327.5ms`，约`1.092x`；
   - another three-method cohort声称`295.5ms`→`258.3ms`，约`1.144x`；
   - RepoBench声称cache-ready `1.089x`；
   - 三组数字来自不同protocol，raw不在Git，chunk/max-prefill未披露，
     不能合并或作为headline；
   - 本项目已验证的同primitive R0在chunk4096为`0.772–0.936x`，
     所以C40只有不利先验，实际功效必须以完整workload pilot重新判断。
3. **工程复杂度分为“可复用底座”和“必须新写的C40层”。**
   - 可直接复用：copy V/K RoPE relocation、segment identity/store、
     cross-store allocator/budget/object graph、prefix protection、
     provisional lifecycle、fallback taxonomy/accounting、telemetry与manifest；
   - 必须新写：结构化read/write provenance selector、C40 adapter/metadata、
     strict-middle request controller、consume/produce双角色生命周期、
     approx provenance/chain-depth和真实feature gate；
   - 当前底座只支持从exact boundary连续恢复，不能直接执行V40的strict
     middle island。
4. **估计工作量：约6–9人周，不需要重写恢复数学或cache基座。**
   - selector/provenance约2–3人周；
   - middle-span controller与异常路径约2–3人周；
   - adapter、telemetry、tests与独立review约2–3人周。
5. **外部依赖。**
   - coding-only C40不需要prefetch；
   - 需要agent/tool wrapper提供结构化`read_paths/write_paths`、worktree
     generation与content hash；
   - 需要当前cross-store substrate承担生命周期与压力管理；
   - 需要RepoBench/SWE-bench质量harness和锁定的Docker依赖层；
   - prefetch只在后续Combined正交实验中需要，且当前相关ref尚不可获得。

## 2026-07-29T17:42:31-07:00 Phase7.5 C40 clean-room规划决策

用户决定新增 **Phase7.5：C40 Clean-Room Reproduction & Extended
Evaluation**。

目标：

1. 只采用合作者V40/C40的研究思路，**不使用、不cherry-pick、不复制其代码**；
2. 从当前`research/cross-store-substrate@0206f17b...`创建全新branch；
3. 在当前经过Phase6/7验证的`approx_kv/cross_store`底座上独立重实现
   `C40 = G40 selector × R0 Raw+RoPE executor`；
4. 完整覆盖合作者已实现的功能；
5. 尽可能补齐其尚未实现或未接线的能力；
6. 将实现与本项目已有可行性测试、workflow、RepoBench/SWE-bench和其它
   合理测试统一评估。

Clean-room边界：

- 不从`review/coding-aware-v40-prefetch-20260729`复制源码、测试或runtime；
- 允许把已审计的**行为规范、失败模式和公开研究机制**作为需求输入；
- 新代码必须使用新的模块结构、类型、tests、manifest和runner；
- 自动检查不得导入旧`kvcomm`/`coding_aware`模块；
- 历史Phase4–7 artifact保持冻结，Phase7.5使用独立结果目录与manifest。

拟议branch：

`research/phase7.5-c40-cleanroom`

拟议base：

`origin/research/cross-store-substrate@0206f17b4255e4b248dafaaeb943be57428dae2f`

强制能力分层：

- **Parity core**：structured read/write provenance、单岛选择、唯一token匹配、
  strict-middle copy、V copy、K RoPE relocation、岛外dense、fail-closed；
- **Correctness extensions**：repo/worktree generation、content hash、rename/
  symlink规范化、unknown-effect fail-closed、真实feature gate、lease GC；
- **Runtime extensions**：middle-span state machine、consume/produce双角色、
  request-lifetime状态、approx provenance/depth、异常路径cleanup；
- **Research extensions**：multi-island、copy-budget optimizer、optional
  leading-k repair、AST/embedding辅助gate、host/demand-load、prefetch接口、
  concurrency与branch/worktree isolation。

正确性顺序保持：

```text
exact cache -> controlled C40 reconstruction -> dense fallback
```

当前授权边界：

```text
phase7_5_plan_authorized = true
branch_creation_authorized = false
implementation_authorized = false
docker_test_execution_authorized = false
gpu_execution_authorized = false
```

下一步仅为创建、双模型review并发布
`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`。任何branch、代码或实验必须在计划定稿
后再次取得用户明确授权。

## 2026-07-29T23:58:44-07:00 Phase7.5 C40计划第二轮只读closure review

对`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`（V1-r2）做第二轮targeted closure
review，全程只读，未改计划文件、未建branch、未跑测试/GPU。

复核方式：逐条对照冻结底座
`origin/research/cross-store-substrate@0206f17b...`的真实API与断言
（本地`sglang-latest-main`内含该commit，用`git show`只读比对）。

上一轮13项结果：**9 CLOSED / 3 PARTIAL / 1 OPEN**。

- CLOSED：P1-01、P1-02、P1-03、P1-04、P1-05、P2-02、P2-03、P2-04、P2-06。
- PARTIAL：P0-01（单请求已闭合，多chunk并存未闭合）、P2-01、P2-05。
- OPEN：P2-07（§25.1仍允许host构建manifest）。

本轮判定：**FAIL / NOT READY**，open `2 P0 / 2 P1 / 6 P2`。

两个新P0（均由本次修订本身引入或未覆盖）：

1. **forced-middle与底座"同时只允许一个chunked请求"不变量冲突**。
   底座`scheduler.py:3075`有`assert self.chunked_req is None`，
   `new_chunked_req`与`chunked_req`都是单值，`:3079`只对单个请求
   `inflight_middle_chunks += 1`，`batch_result_processor`亦注明
   "There is only at most one request being currently chunked"。
   计划§8.6.3步骤3要求任何被钳制的C40请求都设`new_chunked_req`，
   且钳制会**少消耗**chunk预算，反而解除了底座原本的隐式互斥，
   因此第二个chunked请求可触发assert崩溃，或被静默覆盖后在
   `target_start`采样（等于P0-01原failure mode换了触发条件）。

2. **retraction / preemption生命周期完全缺失**。
   全文0处提到retract/preempt。底座`schedule_batch.py:1790,1795`
   在KV压力下执行`release_kv_cache(...)`+`reset_for_retract()`
   （`:1567`），会清零`prefix_indices`/`cache_protected_len`/
   `kv_committed_len`/`inflight_middle_chunks`，但不会清理`req.c40`。
   计划的五路cleanup恒等式、INV-1/6/7与lease归零断言因此不成立；
   陈旧`middle_cursor`会经`effective_prefix_indices`进入
   `write_cache_indices`，把**已释放的index**写回`req_to_token`。
   `rho ∈ {1.5,2.0}`与T9双向压力正是primary路径，可达性高。

两个P1：B-3提前`req_to_token_pool.alloc([req])`缺失返回`None`的失败分支
（会破坏`release_kv_cache`的`(req_pool_idx is None) == (req.kv is None)`
断言）；以及`add_one_req`内`needs_host_load_back()`会在staging之后再次
增长`prefix_indices`并改写`cache_protected_len`，使staged快照与INV-4/INV-6
失效。

六个P2：族4 `kind`域缺`cleanup_failed`/`prefill_completion_misclassified`
且族3恒等式在cleanup失败时双计；§25.1的host manifest构建；`req_to_token`
为`int32`而Triton kernel硬转`int64`指针，计划的"dtype相同"只靠隐式类型提升
成立；`add_one_req_ignore_eos`/`_add_dllm_req`两个`set_extend_range`调用点
未纳入effective-prefix与钳制覆盖；§28.2的`Version: V1`与版本表`V1-r2`不一致
且§0.4未列入G0q；F-04/F-06/F-07/F-08/F-10/F-11在全文无锚点、closure不可追溯。

结论：计划的底座API引用（行号、断言、free区间）本轮抽查全部属实，
`prefix_lens/prefix_tensors`成对、单一free ledger、identity membership、
adapter-local dense disposition、collector authority分层均已正确闭合；
但在**并发chunked语义**与**retraction生命周期**两处仍不可执行，
不得进入plan freeze。

## 2026-07-30T03:29:27-07:00 Phase7.5 C40计划第三轮只读closure review

范围：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md` V1-r3（5455行），全程只读。
未修改计划文件、未建branch/worktree、未执行任何测试或GPU。
复核基准：冻结底座
`/home/chris/Workspaces/kvcache-research/worktrees/cross-store-substrate@0206f17b`
的真实源码，逐API核对（event_loop_overlap / stash / result、
prepare_for_decode / release_req / retract_decode、add_one_req /
add_chunked_req、PrefillAdder offsets、ReqToTokenPool、feature gate）。

本轮判定：**FAIL / NOT READY**，open `2 P0 / 2 P1 / 6 P2`。

上一轮（第二轮）4项阻断findings全部 CLOSED：
P0-A单chunked owner（owner-exclusion + `reserve_exclusive_chunked_owner`）、
P0-B retraction生命周期（`release_req`入口区分pure retract与abort-last，
与`retract_decode`在`release_req`前设`to_finish`的实际顺序一致）、
P1-A `req_to_token_pool.alloc([req])`返回None分支、
P1-B `needs_host_load_back()`在staging fail closed。
第三轮自称的9项修复中，6项经底座核实成立
（INV-7限prefill、validation隔离、identity membership、
unsupported config来源、int64长生存prefix tensor、CONTINUE+not-added统一清理）。

两个新P0（均由本轮新引入的B-4 parked continuation与deferral快照语义导致）：

1. **B-4 parked continuation与底座single-chunked-owner记账冲突**。
   §8.6.3步骤7要求copy失败且无budget时`add_chunked_req`返回原req但不入
   `can_run_list`。底座`scheduler.py:3073-3079`在此后无条件执行
   `assert self.chunked_req is None` 与
   `self.chunked_req.inflight_middle_chunks += 1`，而递减只发生在
   `batch_result_processor.py:283/336`对`batch.reqs`的遍历中。
   后果：(a) parked轮未消耗`rem_chunk_tokens`，任一长非C40请求仍会写
   `new_chunked_req`，触发`assert`崩溃；(b) 计数器只增不减，最终chunk被
   `batch_result_processor.py:229`判为middle chunk，不采样、不写output_ids，
   请求带空`output_ids`进入decode。底座`schedule_policy.py:841-842`明确
   "chunked_req must be added to the list; otherwise, it will cause a memory
   leak"，非SWA路径恒admit（`_rem_tokens<=0`时回退为`rem_chunk_tokens`），
   parked属于计划自造的偏离。

2. **`borrowed_exact_indices`跨ADMISSION_DEFERRED轮次冻结**。
   §8.5与§8.1.1要求deferred时`borrowed_exact_indices`"全部保持"、
   "复用同一plan"，重新验证只覆盖source侧
   generation/content hash/fingerprint；TC-8进一步断言"plan逐字段保持"。
   但deferred请求不持有任何lock（`_req_inc_lock_ref`未执行，
   `protect_request_prefix`已退出），其matched prefix在轮次之间可被驱逐并
   重新分配；`schedule_policy.calc_priority`→`match_prefix_for_req`
   （`schedule_policy.py:110-143`）与`init_next_round_input`都会重写
   `prefix_indices`/`cache_protected_len`。长度相同但index不同时INV-6仍通过，
   陈旧index经`effective_prefix_indices`进入`allocation.py:316`与
   `write_cache_indices`，把**已释放并被他人复用**的slot写回`req_to_token`
   → 静默跨请求KV污染。§8.6.3步骤1的"重新staging"与上述"逐字段保持"
   相互矛盾，规范未定。

两个P1：

- `charge_c40_copy_allocation(island_len)`（§8.6.3步骤5c）在island已物理分配
  之后再加`rem_total_token_offset`/`cur_rem_token_offset`；
  `PrefillAdder.rem_total_tokens`/`cur_rem_tokens`是读取
  `token_to_kv_pool_allocator.available_size()`的live property
  （`schedule_policy.py:585-620`），分配本身已使其下降，因此island被计两次。
- §7.3.1冻结的dense disposition契约与底座API不符：
  `transfer.py:120-140`的回调是
  `backend.dense_prefill(target_start=, length=, reason=)`而非`DenseRange`
  对象，且底座会先对fallback chunk以`stale_handle`/`residency_miss`/
  `source_slice_mismatch`为reason回调，再用`_contiguous_ranges`把剩余
  dense range**重新切片**。按`(target_start,length,reason)`三元组精确查表
  在普通copy fallback下必然miss→fail closed，把可直接归因的具体reason
  吞成`c40_copy_exception`，违反§12.1规则4（Phase7 correction教训）。

六个P2：island token被计入底座`log_hit_tokens`/`req.cached_tokens*`
（用户可见的`cached_tokens`）而与exact命中混同；
`reserve_exclusive_chunked_owner`使每个forced-middle轮变成单请求batch
（`rem_chunk_tokens=0`⇒后续`add_one_req`一律`trunc_len<=0`返回OTHER并break），
该batching差异未在§19.9披露；§8.2.1规则3允许中途解除
`skip_radix_cache_insert`，而`radix_cache.py:596-605`的`cache_unfinished_req`
会改写`prefix_indices`与`cache_protected_len`，与INV-4/INV-6冲突；
§7.4引用`scheduler.py:2966/2967`实为`2967/2968`（off-by-one）；
`enable_mixed_chunk`未纳入staging fail-closed清单也未在§19.9冻结为0；
`process_pending_chunked_abort`（`scheduler.py:2646-2673`，直接调
`release_kv_cache`、`to_finish`被显式置None、`finished()`为True）未列入
§8.1.2五条final hook，而它正是C40 chunked owner最可能的abort入口。

结论：第三轮修订使retraction、single-owner、overlap commit、B-3 transaction、
validation隔离、unsupported config六个方向达到可执行强度，行号与释放区间
抽查全部属实；但**B-4 parked continuation**与**deferral快照语义**两处仍会导致
scheduler assert崩溃或静默KV污染，不得进入plan freeze，不得升级状态。

## 2026-07-30T04:23:18-07:00 Phase7.5 C40计划第四轮只读closure review

只读逐API复核 `IMPLEMENTATION_PLAN_PHASE7_5_C40.md`（版本表仍标 `V1-r3`，
5494行），底座固定 `worktrees/cross-store-substrate@0206f17b`（clean）。
未修改计划文件、未建branch、未跑任何测试、未占用GPU。

本轮 verdict = `FAIL / NOT READY`；open = `0 P0 / 2 P1 / 7 P2`。
第三轮两个P0（B-4 parked continuation、deferral快照语义）与两个P1
（copy allocation双计、dense disposition契约）**全部 CLOSED**，
经底座逐条核实：

- `add_chunked_req`（`schedule_policy.py:830-868`）在非SWA路径恒
  `can_run_list.append(req)`，故"禁止parked、本轮必须admit
  positive-length dense chunk"成立；`rem_chunk_tokens` 在
  `get_new_batch_prefill` 开头为 `chunked_prefill_size`（mixed chunk 关闭
  ⇒ `num_mixed_decode_tokens=0`），故 owner continuation 的 `new_len > 0`。
- `rem_total_tokens`/`cur_rem_tokens`（`schedule_policy.py:565-620`）确为
  读取 `available_size()+evictable_size()` 的 live property，删除
  offset charge/refund 后不再双计；island 物理分配（必要时先驱逐）净效应
  恰为 `-island_len`。
- `transfer.py:120-140` 的回调确为
  `dense_prefill(target_start=, length=, reason=)`，fallback chunk 先于
  `_contiguous_ranges` 重切的 dense range 发出，故 "reason + 区间被唯一
  rule 完整包含" 的解析规则可实现，三个底座 reason 可直通。
- `_lock_node` 内 `torch.equal(borrowed, prefix_indices)` final guard 可
  阻断 deferred 期间 prefix 被驱逐重分配导致的陈旧 index 写回。
- `process_pending_chunked_abort`（`scheduler.py:2632-2673`）先
  `prepare_abort`（`disaggregation/utils.py:1055` 置 `finished_reason`）
  再把 `to_finish=None`，故以 `finished()==true` 识别 abort 正确。
- `skip_radix_cache_insert` 只在 `schedule_batch.py:1080` 赋值一次，
  消费点仅 `common.py:104/200`，全生命周期恒真成立。
- 全部 `verified-code` 行号抽查属实：`scheduler.py:2630/2745-2756/2967/2968/
  3017/3018/3075/3079/3602`、`schedule_batch.py:863/1080/1184/1211/2213/2217/
  2337/2368/2511-2551`、`schedule_policy.py:435-438/830-868/1160-1163`、
  `allocation.py:316`、`common.py:104/167-248`、`memory_pool.py:273`。

两个新P1：

- **P1-1（exact/copy telemetry 未完全分离）**：§8.6.1a 把
  `prefix_lens`（`schedule_batch.py:2217`）改为 `effective_prefix_len` 后，
  同一批 `pre_len` 会流入 `schedule_batch.py:2317-2318`
  `new_cached = pre_len - req.already_computed; req.cached_tokens += new_cached`。
  dense chunk 轮 `new_cached == 0`，但 **copy 提交轮** `pre_len` 从
  `target_start` 跳到 `target_end`，`req.cached_tokens` 被加上 `island_len`。
  该字段经 `output_streamer.py:85-87/198` 作为用户可见 `cached_tokens`
  返回，直接违反计划自订的"copied island 禁止出现在用户可见 cached_tokens"
  与 TC-60。计划只改了 `cached_tokens_device`（2337），漏掉 2317。
  最小修复：copy-boundary commit（§8.6.3 步骤5 (d)/(e) 之间）追加
  `req.already_computed = max(req.already_computed, target_end)`；
  并把 `schedule_batch.py:2314-2318` 列入 §8.6.1a 表；TC-60 增断言
  请求结束时 `req.cached_tokens == len(borrowed_exact_indices)`。
- **P1-2（最小 rw mount 未真正落到 manifest，F-11 回归）**：§13.5.0 与
  §14.1 的 `"mounts"` 数组声明
  `{"host": ".../kvcache-research/results", "container": "/global_results",
   "mode": "rw"}`，即把宿主 `results/` 根目录整体 rw 挂载，与同段
  "**禁止**把整个宿主 `results/` 根目录以 rw 挂载"直接矛盾，也与 CR-7/CR-8
  的 Phase4–7 冻结目录只读要求矛盾；同时 `mounts` 里**没有**
  `/results/phase7_5_c40` 条目，而 §13.5.0 规定"未列明则不得挂载"，
  runner 将无处写 artifact。最小修复：两处 `mounts` 改为三条
  （`results/phase7_5_c40:/results/phase7_5_c40:rw`、
  `results/BENCHMARK_RUN_LOG.jsonl:/global_results/BENCHMARK_RUN_LOG.jsonl:rw`
  且标 `bind_type=file`、`<worktree>:/w:ro`），并在 CR-7 增加
  "任何 mount 的 host 不得等于 results 根且 mode=rw" 的静态断言。

七个P2（见 TRACKING 同时间戳记录）：`inflight` 前 membership assert 未对
底座 hybrid-SWA parked 分支设限；INV-3 单调性与 deferred target-side 重建
自相矛盾；TC-51 "measured path 零 D2H sync" 与强制 `torch.equal` admission
guard 冲突；§7.3.1 未声明 `already_materialized` 规则须覆盖 `[0, target_start)`
（含 borrowed exact），否则 `_validate_bounds` 必抛 unowned gap；staging
fail-closed 清单缺 `enable_dynamic_chunking` / multimodal-encoder /
`truncation_align_size`；TC-29 的 logprob 逐位置对齐在 `logprob_start_len <
target_end` 时不可达且无 fail-closed 规则；版本号仍为 `V1-r3` 但 `V1-r3`
变更行已被改写为"闭合四轮findings"，同一版本号指向两份不同文档。

F-01..F-11：F-11 = PARTIAL（P1-2），其余十项 CLOSED。

结论：第四轮修订使 B-4 fallback、deferred rematch、copy 预算、dense callback
四个方向达到可执行强度且与底座逐API一致；但 exact/copy telemetry 分离尚未
闭合、最小 rw mount 仍未落到 manifest，**不得** plan freeze、**不得**升级状态、
**不得**申请 branch 授权。

## 2026-07-30T05:44:53-07:00 Phase7.5 C40计划第五轮只读closure review

范围：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md` V1-r5（5557行），全文件逐API、
跨章节复核。底座固定 `worktrees/cross-store-substrate@0206f17b`（clean）。
本轮**未修改计划文件、未建 branch、未跑任何测试、未占用 GPU**。

Verdict = `FAIL / NOT READY`；open = `0 P0 / 1 P1 / 5 P2`。

### 第四轮 open 项全部 CLOSED（经底座逐API核实）

- P1-1 exact/copy telemetry 分离：§8.6.3 步骤5e 增 `req.already_computed +=
  island_len`，并新增 §8.6.1a 的 `schedule_batch.py:2314-2318` 行。逐轮推演：
  B-3 得 `new_cached = (exact+island) - island = exact`；B-4 得
  `target_start + island == target_end` 使 suffix 轮 `new_cached = 0`；
  底座 `:2317-2318/:2344`（`already_computed = seq_len`）与
  `:2339 cached_tokens_device`（C40 已 fail-close host load-back ⇒
  `device_portion == len(prefix_indices) == len(borrowed)`）均闭合。TC-60 覆盖。
- P1-2 最小 rw mount：§13.5.0 与 §14.1 的 `mounts` 现同时含
  `/results/phase7_5_c40`（directory rw）与 `BENCHMARK_RUN_LOG.jsonl`
  （file rw），并新增 `docker_capabilities` / `docker_security_opts`；
  CR-7 改为"恰好两处 + tmpfs ephemeral 例外 + 静态断言"。F-11 由 PARTIAL 回到 CLOSED。
- 七项 P2 全部 CLOSED：membership assert 仅在 `lifecycle_live` 时生效
  （底座 `schedule_policy.py:844-846` hybrid-SWA parked 路径保持 baseline）；
  INV-3 改为"admission 成功后单次 active lifecycle 内单调"；TC-51 明确
  validation=0 仍跑 `torch.equal` guard 并计 `c40_admission_guard_ms`；
  §7.3.1 显式覆盖 `[0, exact_length)` 与 `[exact_length, target_start)`
  （与 `transfer.py:_validate_bounds` 一致）；staging fail-close 增
  `enable_dynamic_chunking` / `enable_deterministic_inference`
  （`scheduler.py:1333-1348` 证明 `truncation_align_size` 被其完全蕴含）/
  multimodal-encoder；新增 `c40_logprob_range_unsupported`；版本升 `V1-r5`
  且 §28.0 保留 r1..r5 完整版本行。

### 本轮新发现

**P1-1（阻断）**：§8.6.3 步骤4B 与 §7.2 的 enqueued-range commit hook 中，
`middle_cursor` 的三分支状态判定**没有 `c40_state` 守卫**：

```
if   middle_cursor == target_start: COPY_READY
elif middle_cursor <  target_start: 保持 DENSE_PREFIX
else: c40_boundary_overrun -> DENSE_ISLAND_FALLBACK + 族4{boundary_overrun}
```

而 §8.6.3 步骤6 / §8.6.1b / TC-6 / TC-27 / TC-50 / INV-3 都要求
`middle_cursor` 在 `DENSE_SUFFIX` 阶段继续推进到 `prompt_end`，且计划未给出
第二套推进机制。因此每一个 copy 成功的请求在 suffix 提交（多 chunk 走 stash
seam，单 chunk 走步骤4C 的 terminal commit 或 teardown 幂等兜底）时必然
`middle_cursor > target_start`，被误判 `c40_boundary_overrun`：
outcome 由 `c40_copied` 退化为 `dense_attempted_fallback`，且族4
`boundary_overrun > 0` 会使**整个 restart/launch block** `engineering_valid=false`
（§12.4 族4）⇒ primary confirmatory 无有效数据。该分支同时与 §8.2 转移矩阵
（`DENSE_SUFFIX` 行无 `→ DENSE_ISLAND_FALLBACK` 边）及 §12.2 的 reason 定义
（"dense prefix 越过 target_start"）矛盾。若反向解读为"步骤4 不在 DENSE_SUFFIX
执行"，则 cursor 停在 `target_end`，每个 suffix chunk 都从 `target_end` 重算，
违反 TC-27 并使 `prepare_for_extend` 的
`assert seq_len - pre_len == extend_range.length` 语义失效。两种解读均不可执行。
最小修复：把三分支判定包在 `if c40_state == DENSE_PREFIX:` 内，
`DENSE_SUFFIX` / `DENSE_ISLAND_FALLBACK` 只推进 cursor 与 owned 集合。

**P2-1**：§8.6.1a 要求 `cand_extend_input_len` 与 `schedule_policy.py:1160-1163`
的 page-align 重算一律改用 `effective_prefix_len(req)`，与 §8.6.3 步骤2
"add_one_req 先用 projected_prefix_len（B-3 为 `target_end`）完成全部
capacity/chunk/alignment gate" 冲突。B-3 在 gate 时刻 `middle_cursor == exact_length`，
按 §8.6.1a 字面实现会算出多出 `island_len` 的 `cand_extend_input_len`，
而 copy 之后的 `set_extend_range` 起点已是 `target_end` ⇒ `extend_range.end`
超出 `prompt_end`，`input_ids` 长度与 `extend_range.length` 不符。

**P2-2**：§8.6.2 的 B-3 判据只有 `target_start == exact_length > 0`，
未含 `target_end < len(prompt)`，因而**遮蔽** B-5 兜底。§8.6.2 自称
"B-5 保证穷尽 / 不得依赖 selector 已过滤"，TC-69 亦要求
`target_end >= prompt_len` 一律 `dense_ineligible`；但
`target_start == exact_length ∧ target_end >= len(prompt)` 会先命中 B-3
进入 copy transaction，TC-69 按现有判据不可满足。

**P2-3**：§14.1 plan manifest schema 无 `provenance.{collector_impl,
primary_oracle_impl, supplemental_oracle_impl, collector_capabilities}` 段，
但 §9.2.3、§17.1.4、`FD-7` 与 TC-66 都要求授权 manifest 逐字段列明并
"缺任一即 manifest check 失败"。§9.1 的 `provenance` 块属 ToolEvent schema，
不是 plan manifest。

**P2-4**：§13.5.0 在列出两个 `-v` 之后仍保留
"这是**唯一**允许的宿主可写挂载点"，与同文件 CR-7 / §11.3 / §23.3 的
"恰好两处"冲突；CR-7 的静态断言正是按该段实现，是 F-11 曾经回归的同一处文本。

**P2-5**：§8.6.1a 的 `_update_prefill_budget(prefix_len=...)` 行未限定作用域。
底座只有 `add_one_req` 传 `prefix_len = len(req.prefix_indices)`
（`schedule_policy.py:1044`），`add_chunked_req` 与非 chunked 分支传字面 `0`
（`:855`、`:1132`）。按该行字面在续算轮改传 `len(borrowed_exact_indices)`
会使 `log_hit_tokens` 每轮重复累加 exact 前缀，污染 prefix-cache 命中率遥测。

### 本轮已逐API核实为属实的计划断言（抽样）

`schedule_batch.py` 的 `:863 / :1080 / :1184 / :2213 / :2217 / :2317-2318 /
:2339 / :2368 / :2511 / :2516 / :2526 / :2548 / :2551 / :1567 / :1769-1795`；
`schedule_policy.py` 的 `:435-438 / :830-867 / :844-846 / :1044 / :1073
（`_lock_node` 覆盖到 append）/ :1159-1162`；`scheduler.py` 的
`:2630 / :2646-2673 / :2744-2756 / :2967-2968 / :3017-3018 / :3075 / :3079 /
:3106 / :3602`；`allocation.py:316` 与 `write_cache_indices` 的
`prefix_tensors/prefix_lens` 成对语义；`common.py:104 / :167 / :197-202`；
`memory_pool.py:273 / :279-305`（`alloc` 复用 slot 且断言
`inflight_middle_chunks > 0 or kv_committed_len > 0`，B-3 先置
`kv_committed_len = exact_length` 恰好满足）；`radix_cache.py:496-553`
（`cache_finished_req(is_insert=False)` free `[cache_protected_len, key_len)`
+ tail，再 `dec_lock_ref` ⇒ INV-8 成立）；`transfer.py:25-31 / :40-66 /
:70-82 / :95-146`（`dense_prefill` kwargs、fallback chunk 与
`_contiguous_ranges` 切分、`recomputed_tokens` 口径）；
`types.py:75-87`（`DenseRange` 拒绝 `length <= 0`，印证 zero-length 省略规则）
与 `TransferSpan.chunk_start/chunk_length` 字段；
`approx_kv/runtime.py:415-449` 的 `resolve_reuse_spans` prefix_gap 语义；
§2.6 全部 reason 常量（含 `common.py:139/146` 的两个 cross-store pressure
reason 与 `epic_runtime.py:522/554/576` 的两个 prefix-family）。
统计学核对：`z_sum^2 = 6.182558`、`2473.0`、`2856.9`、
`delta1-delta0 = 0.046520` 与全部示例取整一致；§16.2 十四项上下界求和
`8.25 / 12.25`、WP11 九 lane 求和 `4.00 / 8.00`、D-3 的
`3×5×2×3 = 90` 与 `38+30+30 = 98` 全部自洽。TC-1..TC-69 无缺号无重号。

### F-01..F-11

| Finding | 状态 |
| --- | --- |
| F-01 / F-02 | CLOSED（所有权分离、effective prefix、req_pool_idx 生命周期均已逐API核实；本轮 P1-1 位于同一锚点但属状态机守卫问题） |
| F-03 | CLOSED |
| F-04 | CLOSED |
| F-05 | CLOSED |
| F-06 | CLOSED |
| F-07 | CLOSED |
| F-08 | CLOSED |
| F-09 | CLOSED |
| F-10 | CLOSED |
| F-11 | CLOSED（残留 P2-4 文本冲突） |

结论：第五轮修订把 exact/copy telemetry 分离与最小 rw mount 两项阻断
全部闭合，且七项 P2 全部消除；但 enqueued-range commit 的状态守卫缺失是
新的执行级阻断。**不得** plan freeze、**不得**升级状态、**不得**申请 branch 授权。

## 2026-07-30T06:29:17-07:00 Phase7.5 C40计划第六轮只读closure review

- 计划文件：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表 `V1-r6`，5576 行，
  sha256 = `51fda84c6ae04bda6f5aefb3d03c24f16363fa89500c2b05e5317c1e6e448e62`。
- 底座固定 `worktrees/cross-store-substrate@0206f17b42`（clean，已核）。
- 本轮全文件逐 API + 跨章节 closure review；**未改计划文件、未建 branch、
  未跑测试/GPU**。
- Verdict = `FAIL / NOT READY`；open = `0 P0 / 1 P1 / 7 P2`。

### 第五轮 open 项闭合情况（经底座逐 API 核实）

| 上轮项 | 结论 |
| --- | --- |
| P1（enqueued-range commit 缺 `DENSE_PREFIX` 守卫） | **CLOSED**：§8.6.3 步骤4B 已改为 `if c40_state == DENSE_PREFIX / elif {DENSE_SUFFIX, DENSE_ISLAND_FALLBACK} / else fail loud`，并新增 TC-70 |
| P2-1（`cand_extend_input_len` / `:1160-1163` vs `projected_prefix_len`） | **PARTIAL**：新增 `scheduling_prefix_len()` 且 `cand_extend_input_len` 行已改，但 `:1160-1163` 行仍写 `effective_prefix_len` ⇒ 本轮 P2-5 |
| P2-2（B-3 缺 `target_end < len(prompt)`） | **CLOSED**：B-3/B-4 均带该条件，B-5 兜底穷尽，TC-69 可满足 |
| P2-3（§14.1 manifest 缺 provenance 字段） | **NOT CLOSED**：字段被加进 §9.1 ToolEvent schema 而非 §14.1 plan manifest ⇒ 本轮 P2-1 |
| P2-4（§13.5.0 "唯一允许的宿主可写挂载点"） | **CLOSED**：现为"上述两处是唯一允许的宿主可写挂载"，与 CR-7 / §11.3 / §23.3 / TC-73 一致 |
| P2-5（`_update_prefill_budget(prefix_len=)` 未限定路径） | **CLOSED**：首轮传 `len(borrowed_exact_indices)`、`add_chunked_req` 及后续轮传底座字面 `0`；已对照 `schedule_policy.py:708-752/855`（`prefix_len` 只喂 `log_hit_tokens`）核实，TC-74 覆盖 |

### 本轮新 P1（阻断）

**pre-admission `DENSE_ISLAND_FALLBACK` / `dense_ineligible` 未做 target-side reset。**

- §8.5 阶段2 明确：deferred 请求下一轮重新 match / 重建 target geometry 并
  重验 source，"不通过 ⇒ 转 `DENSE_ISLAND_FALLBACK` 或 `dense_ineligible`"。
  该转移发生在**尚未 admit**（`req_pool_idx is None`、零 owned KV）的请求上。
- §8.6.1a 的 `effective_prefix_len/indices` 只在 `c40.state is NONE` 时退化为
  `len(req.prefix_indices)`；§7.4"extend 起点改用 cursor"行又要求
  `DENSE_ISLAND_FALLBACK` 下 `set_extend_range` 起点取 `middle_cursor`。
- §7.4 `init_next_round_input` 行要求跨轮**保留** `middle_cursor`；
  §8.2 的 `DENSE_ISLAND_FALLBACK` 清理只覆盖 provisional slot 与 lease；
  admission final equality guard（`torch.equal(borrowed, prefix_indices)`）
  只在 `c40_on_admission_commit` 内、即只对 `STAGED` 的 B-3/B-4 执行。
- 后果：该请求若本轮未被 admit（capacity 场景本就常见），下一轮
  `init_next_round_input` 重新 `match_prefix` 生成新的 `prefix_indices` /
  `cache_protected_len`，而 `borrowed_exact_indices` 与 `middle_cursor`
  仍是上一轮快照且无 guard。随后
  `allocation.py:316` → `write_cache_indices`（`allocation.py:55-100`）会把
  **陈旧 index** 写回 `req_to_token[req_pool_idx, 0:stale_len]`；被驱逐复用的
  slot 即造成静默跨请求 KV 污染，`common.py:167-248` teardown 还可能对
  Radix 所有的 index 二次 free。这与第三轮已闭合的 P0-2（`ADMISSION_DEFERRED`
  陈旧快照）**同类**，只是发生在 fallback / ineligible 分支。
- 最小修复：把"未 admit"作为判据而非 `state is NONE` ——
  (a) `effective_prefix_len/indices` 在 `req.req_pool_idx is None`（或未 admit）
  时一律退化为底座字面量；且
  (b) 在 admission 之前进入 `DENSE_ISLAND_FALLBACK` / `dense_ineligible` 时，
  按 `ADMISSION_DEFERRED` 同样的规则清空
  `exact_length / borrowed / owned / effective tensor / middle_cursor`；
  (c) 补一条对应 TC（对照 TC-58：跨轮驱逐并复用旧 KV index 后旧 index 零次写回）。

### 本轮 P2（7 项）

1. §14.1 plan manifest 仍缺
   `provenance.{collector_impl, primary_oracle_impl, supplemental_oracle_impl,
   collector_capabilities}`；§9.2.3 / §17.1.4 / `FD-7` / TC-66 / TC-72 都要求它在
   **授权 manifest** 中。r6 把该块加到了 §9.1 的 per-event ToolEvent schema，
   且在同一 JSON 内与 event 级 `collector_impl` / `collector_capabilities`
   同名不同型（dict-of-lists vs list）。
2. §7.2 data-flow 的 commit hook 仍写无守卫的三分支
   （`middle_cursor == target_start ? 小于/等于/大于`），与 §8.6.3 步骤4B、
   TC-70 的 `DENSE_PREFIX` 守卫不一致。
3. `c40_admission_guard_ms` 只出现在 §19.9（强制披露），未登记进 §13.3 的
   Histograms/Timers 表。
4. §8.2 转移矩阵 `DENSE_PREFIX --admit_deferred--> —`，但 §8.6.3 步骤3 的
   owner-exclusion 在 `set_extend_range` 点（按步骤5"pre-range hook"排序，
   已在 `c40_on_admission_commit` 之后）把 B-4 从 `DENSE_PREFIX` 置为
   `ADMISSION_DEFERRED`；同理步骤7 的 `DENSE_ISLAND_FALLBACK → ADMISSION_DEFERRED`
   也不在矩阵内。由 §8.2 生成的 property 测试会与可执行规格互相否定。
5. §8.6.1a 的 `schedule_policy.py:1160-1163` 行仍写"改用 `effective_prefix_len`"，
   与步骤2"用 `projected_prefix_len` 完成**全部**普通 capacity/chunk/**alignment**
   gate"及 TC-71 冲突；同表也未列 `schedule_policy.py:1101` 的
   `input_tokens` 重算点。（在冻结的 `page_size == 1` 下 1161-1163 是精确 no-op，
   故当前无行为影响。）
6. B-4 在 copy 前的 `island_len + planned_suffix_len` capacity gate **被拒**
   （非 allocator 失败）时没有对应的 `c40_*` terminal reason；
   §12.1 规则1 要求每次 attempted fallback 恰好一个 primary_reason，
   §17.7 要求每个 reason 有直接触发用例。
7. §8.6.3 步骤1 的 logprob fail-close 判据 `req.return_logprob and
   req.logprob_start_len < target_end` 把 sentinel `-1`（prompt-logprob 区间为空）
   也判为"区间覆盖 island"，与 §12.2 该 reason 的定义相矛盾。
   `scheduler.py:2336-2353` 已把常见的"只要 output logprob"归一化为
   `len(origin_input_ids)`，故 T5/T6 canary 不受影响；残留仅
   `return_logprob + token_ids_logprob` 且非 prefill-only 的窄路径被过度 fail-close。

### 本轮经底座核实为**属实**的关键点（抽样）

- `resolve_reuse_spans` 的 `next_target/break/prefix_gap`（`runtime.py:415-465`）
  与 §2.7 逐字一致；`protect_request_prefix` 确为 `@contextmanager`（`runtime.py:82`）。
- `scheduler.py:3017/3018`、`2967/2968`、`2630`、`2755-2756`、`3602`、
  `3073-3079`；`schedule_batch.py:1080/1184/1211/863/2213/2217/2316-2317/2338/2368`；
  `schedule_policy.py:434-437/830-868/1033/1044/1101/1161-1163`；
  `allocation.py:316`、`memory_pool.py:273/278-305`、`common.py:103/167-248`
  全部核对属实。
- overlap 事件循环 `scheduler.py:1592-1621` 确实先 `get_next_batch_to_run`
  后 `pop_and_process`，且 `ScheduleBatch.copy()`（`schedule_batch.py:3056`）
  携带 `extend_lens/prefix_lens` ⇒ §8.6.3 步骤4C"禁止读取共享 `req.extend_range`"
  的设计必要且正确。
- `stash_chunked_request` 对 C40 请求因 `maybe_cache_unfinished_req`
  的 `skip_radix_cache_insert` 早退而不触碰 tree_cache（`common.py:103-107`），
  但其调用点 `scheduler.py:2755` 的 `extend_range.end > len(prefix_indices)`
  在 primary 恒成立，commit hook 可靠。
- exact/copy telemetry：`already_computed += island_len` 使 B-3 首轮
  `new_cached == target_start == len(borrowed)`、B-4 suffix 轮 `0`，
  与 TC-60 / TC-74 一致。
- `cache_finished_req(is_insert=False)` free `[cache_protected_len, key_len)`
  且 `dec_lock_ref` 释放 borrowed（`radix_cache.py:496-551`）⇒ INV-8 成立。
- Triton `write_req_to_token_pool_triton` 硬转 `tl.pointer_type(tl.int64)`
  ⇒ effective prefix tensor 的 int64/contiguous/device/强引用约束必要。
- CR-4 冻结命令在底座实测输出为空（0 行），`benchmark/approx_kv/results/`
  下 `/home/` 命中 `115` 处，与计划 `verified-local` 声明一致。
- 统计学：`z_sum=2.486475`、`2856.9`、`2473.0`、`delta0/delta1`、
  §16.2 求和 `8.25/12.25`、WP11 `4.00/8.00`、D-3 `90` 与 `98` 全部自洽；
  TC-1..TC-74 无缺号无重号。

### F-01..F-11

| Finding | 状态 |
| --- | --- |
| F-01 / F-02 | CLOSED（本轮 P1 位于同一锚点，但属"未 admit 状态的 helper 退化判据"缺口，不构成 F-01/F-02 回归） |
| F-03 | CLOSED |
| F-04 | CLOSED |
| F-05 | CLOSED |
| F-06 | CLOSED |
| F-07 | CLOSED |
| F-08 | CLOSED |
| F-09 | CLOSED |
| F-10 | CLOSED |
| F-11 | CLOSED（§13.5.0 残留措辞已消除） |

结论：第六轮修订把上轮的执行级 P1 与 4/5 项 P2 闭合，但仍存在一条与
第三轮 P0-2 同类的静默 KV 污染路径。**不得** plan freeze、**不得**升级状态、
**不得**申请 branch 授权。

## 2026-07-30T06:47:22-07:00 Phase7.5 C40 计划第七轮只读 closure review = FAIL

- 对象：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表 `V1-r7`，5592 行，
  sha256 `9067dd9bc5c3351652296ebcfe5053e949a63792de04012c5b26c522469e049c`。
- 底座：`/home/chris/Workspaces/kvcache-research/worktrees/cross-store-substrate`
  @`0206f17b4255e4b248dafaaeb943be57428dae2f`（worktree clean，未改动）。
- 本轮全文件逐 API + 跨章节复核；**未修改计划文件、未建 branch、未跑测试、未占 GPU**。
- Verdict = `FAIL / NOT READY`；open = `0 P0 / 2 P1 / 4 P2`。

### 第六轮 open 项的闭合情况

| 上轮项 | 状态 |
| --- | --- |
| P1 pre-admission fallback/ineligible 缺 target reset | **PARTIAL**：helper 已加 `req.req_pool_idx is None` 退化判据（`:1620/:1626`）、§8.1.1 已加 `clear_target_staging()` 硬性规则（`:1296-1300`）、TC-75 已加；但只覆盖"未 admit"分支，**已 admit 的续算分支未闭合**（见本轮 P1-1） |
| P2-1 §14.1 缺 provenance 四字段 | CLOSED（`:3111-3121` 已含 `provenance.{collector_impl,primary_oracle_impl,supplemental_oracle_impl,collector_capabilities}`，TC-72） |
| P2-2 §7.2 data-flow 无守卫三分支 | CLOSED（`:951-953` 已加 `仅当 state==DENSE_PREFIX` 守卫） |
| P2-3 `c40_admission_guard_ms` 未进 §13.3 | CLOSED（§13.3 已登记；§19.9 已声明其为 validation=0 下的唯一豁免） |
| P2-4 §8.2 矩阵缺 defer 边 | CLOSED（`DENSE_PREFIX` / `DENSE_ISLAND_FALLBACK` 行已补 `admit_deferred`，TC-77） |
| P2-5 §8.6.1a `:1160-1163` / `:1101` | CLOSED（两行均已列入并统一到 `scheduling_prefix_len`，TC-78） |
| P2-6 B-4 pre-copy gate 无 terminal reason | CLOSED（新增 `c40_copy_budget_insufficient`，§8.6.3 步骤5 + §12.2C + TC-79） |
| P2-7 logprob sentinel `-1` 被误判 | CLOSED（`:1874-1876` 判据改为 `0 <= logprob_start_len < target_end`，TC-80；已对 `scheduler.py:2336-2353` 与 `schedule_batch.py:2364-2367` 逐行核实） |

### 本轮新 P1（阻断）

1. **P1-1：pre-admission fallback 清空 target staging 后，被同轮 admit 的续算路径没有重建入口。**
   §8.1.1（`:1296-1300`）要求任一 pre-admission 转 `DENSE_ISLAND_FALLBACK` /
   `dense_ineligible` 时调用 `clear_target_staging()`，清空
   `exact/borrowed/owned/effective/cursor`。但与 `ADMISSION_DEFERRED` 不同，
   这类请求**在同一调度轮内继续被 `add_one_req` 接纳**
   （`verified-code`：`scheduler.py:3017` `init_next_round_input(tree_cache)`
   紧接 `:3018` `add_one_req`），并以 `DENSE_ISLAND_FALLBACK` 活跃状态跑完全程。
   首次 `prepare_for_extend` → `alloc_for_extend`（`allocation.py:333-335`
   `alloc_req_slots`）分配 `req_pool_idx` 后，helper 的 `req_pool_idx is None`
   退化条件立即失效，`effective_prefix_len()` 转而返回**被清空的**
   `middle_cursor`（0），`effective_prefix_indices()` 返回**空** tensor。后果：
   - §8.6.3 步骤4B 的 `assert snapshot.extend_start == middle_cursor`（`:1995`）
     在下一轮 stash seam 必然失败（`exact_length > 0` 时），
     记族4 `invalid_engineering` ⇒ 整块 `engineering_valid=false`（§12.4）；
   - 若放宽该断言，§7.4（`:1134`）"`DENSE_ISLAND_FALLBACK` 下 `set_extend_range`
     起点取 `middle_cursor`"会使起点回到 0 ⇒ 已物化前缀被重算，且
     `allocation.py:316` 收到空 prefix tensor 与 `prefix_lens=0`，
     borrowed exact 段从 `req_to_token[req_pool_idx, :]` 中消失；
   - 与 §8.1（`:1248-1254`）"`DENSE_ISLAND_FALLBACK` 的 `middle_cursor`
     在 INIT 时 = `exact_length`"以及 §8.2"其后行为与该请求从未 eligible
     逐字段一致"直接矛盾；
   - TC-75 只断言"**且未 admit 时**"，无用例覆盖该续算分支。
   最小修复：pre-admission 的 `clear_target_staging()` 必须同时把
   `c40_state := NONE`（reason 只写 sticky audit 容器），使两个 helper
   在该请求剩余生命周期内永久退化；或等价地在 admission-commit seam 用当前
   `req.prefix_indices` 重建 `borrowed/effective/middle_cursor`。
   并补一条与 TC-75 配对的"清空 → 同轮 admit → 多 chunk dense 跑完"用例。

2. **P1-2：C40 请求的 `skip_radix_cache_insert` 没有任何 seam 会置 `True`。**
   计划的可达性与 exact 纯净性论证完全建立在该 flag 恒为 `True` 上
   （§7.4 约束4 `:1115-1122`、§8.6 "必须纠正的错误假设" `:1527`、
   §8.6.1a `:1611`、§8.2.1 规则1-3 `:1400-1406`、TC-10、TC-62）。但
   `verified-code`：`schedule_batch.py:1080-1082` 为
   `skip_radix_cache_insert = (bootstrap_host == FAKE_BOOTSTRAP_HOST
   or self.approx_kv_metadata is not None)`，且该赋值在 `Req.__init__` 内
   （`approx_kv_metadata` 由 `:832 parse_request_metadata` 从 HTTP metadata 解析）。
   而 C40 的 span 决策**禁止**走 HTTP 字段（§14.2b 规则5），
   §7.4 的 staging hook 也明确是 `if self.approx_kv_metadata is not None:` 的
   **`elif` 旁路分支**（`:1128`，与底座 `schedule_batch.py:1315` 的实际结构一致）。
   因此 C40 请求的 `approx_kv_metadata is None` ⇒ 该 flag 为 `False`。
   后果（若按计划字面实现）：`stash_chunked_request`（`scheduler.py:2630`）
   → `maybe_cache_unfinished_req`（`common.py:103-107`）不再早退 ⇒
   每轮 `cache_unfinished_req` 会改写 `req.prefix_indices` /
   `req.cache_protected_len`（破坏 INV-6 与 borrowed/owned 释放分界，
   `radix_cache.py:496-551` 的 free 区间随之错位），并在 `middle_cursor`
   越过 `target_end` 后把**近似 copied island 插入 exact Radix** ⇒ 触发 SR-4。
   §8.6.4 的既有文件最小修改清单也未列出 `schedule_batch.py:1080`。
   最小修复：把"C40 decision 已绑定该请求 ⇒ `skip_radix_cache_insert := True`"
   写成显式 seam（在 C40 decision 绑定点或 staging 成功点置位，并在
   §8.6.4 / CR-5 allowlist 中列明 `:1080` 附近的修改），
   同时把 §7.4 约束4 的 `verified-code` 措辞改为"对带 approx metadata **或**
   已绑定 C40 decision 的请求"。

### 本轮新 P2（4 项）

1. §9.1 ToolEvent schema（`:2266-2267`）仍内嵌 `collector_impl` 与
   `collector_capabilities`，而 TC-76（`:2217`）断言
   "collector/oracle/capability 对象只存在 plan manifest，**不嵌入 ToolEvent**"。
   两者不可同时成立；须把 TC-76 限定为 §14.1 的 `provenance{}` 聚合对象，
   或从 ToolEvent 删除这两个字段并改由 manifest + `collector_authority_kind` 推导。
2. §8.2 转移矩阵缺"deferred/staged 请求重验后判 `dense_ineligible`"这条边。
   §8.5 阶段2 与 §8.1.1 都明确允许该转移，但 `dense_ineligible` 是 outcome
   而非状态，矩阵也无 `→ NONE` 列；据 §8.2 生成的 property 测试会拒绝合法转移。
   须新增事件（如 `revalidate_ineligible`）与 `STAGED / ADMISSION_DEFERRED → NONE` 边。
3. §8.6.1 的 `req.c40` 容器字段表未声明 `geometry`（B-0..B-5 分类）与
   `suppress_produce_once`，但二者在 `:1633`（`scheduling_prefix_len` 判 `"B-3"`）、
   `:1388`、`:2102`、`:2136` 被规范性引用；§8.5 的 "classification" 与 `geometry`
   也未建立同名映射。须让 §8.6.1 成为唯一规范字段表。
4. TC-66 / TC-72 要求 `docker_capabilities` 与 security-opt "与实际 Docker 命令
   逐字段一致"，但 §9.2.3 的默认命令只带 `--cap-add=SYS_PTRACE`，
   §17.1.4 的差分命令带 `SYS_PTRACE + SYS_ADMIN`，而 §14.1 单一字段冻结为
   `["SYS_PTRACE","SYS_ADMIN"]`；同时 §9.2.3 又规定"未列明的 capability 不得使用"
   （即允许声明超集）。三者叠加使"逐字段一致"不可满足，须改为
   "manifest 声明集合 ⊇ 每次调用实际集合，且每次调用集合逐条记录进 evidence"。

### 本轮经底座核实为属实的关键点（抽样）

- `schedule_batch.py`：`:863` `req_pool_idx=None`、`:1080-1082` skip flag、
  `:1184` `set_extend_range`、`:1211` `init_next_round_input`（无 tree_cache 时
  **不**触碰 `prefix_indices`）、`:2213/:2217`（均在 `alloc_for_extend` **之前**）、
  `:2316-2318` `new_cached = pre_len - already_computed`、`:2337-2339`
  `cached_tokens_device`、`:2364-2371` logprob 起点与 `-1` 语义、
  `:1352-1357` `_compute_max_prefix_len`、`:1567-1597` `reset_for_retract`
  （`prefix_indices` 重置为 **int64** 空张量）、`:1769-1795` `release_req`。
- `schedule_policy.py`：`:434-437` `AddReqResult` 三值、`:708-750`
  `_update_prefill_budget(prefix_len, ...)` 累加 `log_hit_tokens`、
  `:830-868` `add_chunked_req`（`:843-845` hybrid-SWA parked `return req`）、
  `:1037-1044` `cand_extend_input_len`/`prefix_len`、`:1101-1103` `input_tokens`、
  `:1127-1139`/`:1161-1180` 两条 `set_extend_range` + 预算分支、
  `:566-587`/`:598+` `rem_total_tokens`/`cur_rem_tokens` 为 live property
  （⇒ §8.6.3 步骤5 "禁止再加 offset charge" 正确）。
- `scheduler.py`：`:2630` `stash_chunked_request`、`:2646-2673`
  `process_pending_chunked_abort`（`to_finish=None` 且 `finished()==true`）、
  `:2745-2756` stash 条件、`:2811` `get_new_batch_prefill`、`:2967-2968`
  chunked 续算、`:3017-3018`、`:3073-3079` `assert self.chunked_req is None`
  与 `inflight_middle_chunks += 1`、`:3326` `run_batch`、`:3602`
  `process_batch_result`；stash seam 确实早于 `add_chunked_req`。
- `memory_pool.py:278-305` `ReqToTokenPool.alloc` 复用已有 `req_pool_idx`
  并断言 `inflight_middle_chunks > 0 or kv_committed_len > 0`
  ⇒ 计划 B-3 先置 `kv_committed_len = exact_length(>=1)` 是必要且充分的；
  `:308-311` `free()` 会把 `req_pool_idx` 置 `None`。
- `allocation.py`：`:55-100` `write_cache_indices`（Triton 走 `data_ptr()`）、
  `:316` `prefix_tensors` 在 `:333` `alloc_req_slots` **之前**取值、
  `:372-383` 写回、`:398-401` `req.kv` 成对更新。
- `common.py:167-248` `release_kv_cache`：`assert (req_pool_idx is None)
  == (req.kv is None)`、`register_request_segments`、
  `cache_finished_req(kv_len_to_handle=effective_kv_committed_len)`、
  `_release_overallocated_kv_indices` 的 `start_p == end_p` 断言、
  最后 `req_to_token_pool.free(req)`。
- `radix_cache.py:496-551` `is_insert=False` free `[cache_protected_len, key_len)`
  + tail + `dec_lock_ref` ⇒ INV-8 成立。
- `transfer.py:42-68` `_validate_bounds`（full coverage over `target_token_ids`）、
  `:100-175` fallback chunk 先回调、`_contiguous_ranges` 重切 dense range、
  `dense_prefill(target_start=,length=,reason=)` kwargs；
  `types.py:75-87` `DenseRange` 拒绝 `length<=0`（⇒ TC-64 的"零长度必须省略"正确）。
- `runtime.py:83-113` `protect_request_prefix` 不依赖 `approx_kv_metadata`；
  `:415-465/:517/:524` `prefix_gap` 与 §2.7 一致。
- 统计与工时：`z_sum=2.486475`、`2856.9`、`2473.0`、`6.182558`、
  `tau_task` 三例、§16.2 求和 `8.25/12.25`（Track A `8.00/11.75` + WP10）、
  WP11 九 lane `4.00/8.00`、D-3 `90`/`98` arm-cells 与 `12`+4 starts 全部自洽；
  TC-1..TC-80 无缺号、无重号、无悬空引用。

### F-01..F-11

| Finding | 状态 |
| --- | --- |
| F-01 | **PARTIAL**（P1-1 使 borrowed/owned/effective ledger 在 fallback 续算上失效；P1-2 使 `cache_protected_len` 可被 `cache_unfinished_req` 改写） |
| F-02 | **PARTIAL**（P1-1：helper 的退化窗口在 `req_pool_idx` 分配后结束，无接续机制） |
| F-03 | CLOSED |
| F-04 | CLOSED |
| F-05 | CLOSED |
| F-06 | CLOSED |
| F-07 | CLOSED |
| F-08 | CLOSED（附本轮 P2-4 的 capability 一致性措辞残留） |
| F-09 | CLOSED |
| F-10 | CLOSED |
| F-11 | CLOSED |

结论：第七轮修订闭合了上轮 7 项 P2 中的全部 7 项与 P1 的"未 admit"半边，
但 pre-admission reset 的续算半边仍开放，并暴露出一条此前未被审出的
`skip_radix_cache_insert` 前提缺口。**不得** plan freeze、**不得**升级状态、
**不得**申请 branch 授权。

## 2026-07-30T07:58:28-07:00 Phase7.5 C40 计划第八轮只读 closure review = FAIL

- 对象：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表与 §28.2 均为 `V1-r8`，5622 行，
  sha256 `f1feb6de2851cd63aa16004f9e117ea275d9f7dad6de3298b6afed5feca1aa9a`。
- 底座：`/home/chris/Workspaces/kvcache-research/worktrees/cross-store-substrate`
  @`0206f17b4255e4b248dafaaeb943be57428dae2f`（worktree clean，未改动）。
- 本轮全文件逐 API + 跨章节复核；**未修改计划文件、未建 branch、未跑测试、未占 GPU**。
- Verdict = `FAIL / NOT READY`；open = `0 P0 / 1 P1 / 5 P2`。

### 第七轮 open 项的闭合情况（全部 CLOSED）

| 上轮项 | 状态 |
| --- | --- |
| P1-1 pre-admission fallback 同轮 admit 续算路径 | **CLOSED**：§8.1.1 `:1300-1303` 增"同轮以 dense admit 时 `clear_target_staging()` 必须置 `c40_state=NONE`"；§8.5 `:1486`、§8.6.3 步骤2 `:1937`、§8.2 矩阵 `STAGED/ADMISSION_DEFERRED --rematch_invalid--> NONE`、TC-81 齐备。helper `:1626-1629` 双条件退化与 `allocation.py:316` 成对，`prepare_for_extend:2273` 的 `assert seq_len - pre_len == extend_range.length` 仍成立 |
| P1-2 `skip_radix_cache_insert` 无置位 seam | **CLOSED**：§7.4 hook 表 `:1129`、§8.6.3 步骤1 `:1890` 显式 `req.skip_radix_cache_insert := true`（不依赖 HTTP 字段）、§8.6.4 `:2142` 列入最小修改、TC-82 |
| P2-1 §9.1 ToolEvent 与 TC-76 冲突 | CLOSED（ToolEvent 只带该次实际 `collector_impl`/`collector_capabilities`，plan policy dict 不内嵌） |
| P2-2 §8.2 缺 deferred/staged 重验边 | CLOSED（`rematch_invalid → NONE` 两行 + TC-77/TC-81） |
| P2-3 §8.6.1 未声明 `geometry`/`suppress_produce_once` | CLOSED（`:1552-1554` 已声明，含 `lifecycle_live`，TC-83） |
| P2-4 collector capability 三处不可同时成立 | CLOSED（§14.1 引入 `collector_profiles` + `selected_profile_by_gate`；TC-66/TC-72/TC-84 改为按 profile 精确匹配） |

### 本轮新 P1（阻断）

1. **P1：exact Radix 插入抑制的臂间不对称未冻结、未度量，且证伪两条既有不变量。**
   §8.6.3 步骤1（`:1890`）对每个 B-3/B-4 请求置 `skip_radix_cache_insert := true`，
   §8.2.1 规则2（`:1404-1406`）要求整个 lifecycle（含 `DENSE_ISLAND_FALLBACK`
   之后的纯 dense 重算）**不得解除**。`verified-code`：该 flag 为真时
   `common.py:103-107` 使 `maybe_cache_unfinished_req` 早退，
   `common.py:198` 使 `cache_finished_req(is_insert=False)`
   （`radix_cache.py:496-551`）只 free 不 insert ⇒ 该请求的 KV **永不进入 exact Radix**。
   由此产生三个未闭合后果：
   - §8.2 `:1386`"`DENSE_ISLAND_FALLBACK` 其后行为与'该请求从未 eligible'逐字段一致
     （可用差分测试断言）"为**假**：从未 eligible 的请求 `skip=False` 会正常插树，
     差分测试必然在 `skip_radix_cache_insert`、tree_cache 内容、后续请求的
     `prefix_indices`/`cached_tokens` 上失败；§7.5 `:1157`"exact 命中路径的行为与
     C40 关闭时逐位一致"同样被证伪。
   - 计划未冻结 `D0`/`E0`/`C40-D` 对照臂是否同样抑制插入（§15.1 `:3338-3341`、
     §19.4 Step 3 只说"仍执行完整 source lifecycle"）。若对照臂正常插树而
     `C40-1R0` 的全部 staged 请求（`c40_copied` **与** `dense_attempted_fallback`）
     不插树，则在 W4a 这类滚动 agent 轨迹上，C40 臂逐请求丢失本可命中的 exact 前缀；
     该臂间系统性差异被静默折进 primary estimand `theta_j`/`E_cond`，
     §13.1–§13.3 无对应 metric，§19.5 模板与 §19.6 `INVALID` 判据未要求披露，
     §19.4 的臂 C/臂 D 差分也无法把它从机制效应中分离。
   - `skip_radix_cache_insert` 一旦置真即**无复位路径**：`clear_target_staging()`
     + `state=NONE` 不复位，底座 `reset_for_retract`（`schedule_batch.py:1567-1601`）
     也不复位 ⇒ 只在某一轮短暂 staged、随后判 ineligible 或被 retract 的请求
     终生被排除在 exact Radix 之外。
   最小修复：(a) 把 §8.2 `:1386` 与 §7.5 `:1157` 的等价性断言限定到资源/所有权字段，
   显式排除 `skip_radix_cache_insert` 与 tree_cache 填充；
   (b) 在 §15.1/§19.4 冻结每臂的 exact-Radix 插入策略（四臂对同一请求集合同样抑制，
   或允许 KV 全 exact 的 `dense_attempted_fallback` 在 `cache_finished_req` 恢复插入
   并给出安全性论证）；(c) 新增 per-arm `exact_prefix_hit_tokens` 与
   `c40_radix_insert_suppressed_requests_total`，写入 §19.5 模板与 §19.9 披露项；
   (d) 定义该 flag 的复位语义并补配对 TC。

### 本轮新 P2（5 项）

| ID | 内容 |
| --- | --- |
| P2-1 | §7.4 冻结约束4（`:1116`）与 §8.2.1 规则1（`:1404`）仍以"带 approx metadata 的请求恒为 True / 继承既有语义"为前提，该前提对 C40 为假（§14.2b 规则5 禁 HTTP 字段），与同节 `:1129`、§8.6.3 `:1890` 的显式置位互相矛盾 |
| P2-2 | §8.6.1 `:1556-1558` 的 `req.c40_audit` 只声明 `lifecycle_ever_entered`，但 §8.1.1 `:1300-1302`、§8.5 `:1485-1486`、§8.6.3 步骤2 `:1937` 要求 fallback `primary_reason`/`secondary_reasons[]`/pending outcome 在 `clear_target_staging()` + `state=NONE` 后仍存活于该容器；TC-83 自称"state schema 完整" |
| P2-3 | `geometry` 有两套冲突值域：§8.6.1 `:1552` `enum{B-3,B-4,clipped}`（§8.6.1a `:1638` 以 `c40.geometry == "B-3"` 判定）vs §12.3 `:2798-2800` / §13.1 `:2964` `{strict_middle, contiguous_at_exact_boundary, clipped_at_exact_boundary}`，无声明映射 |
| P2-4 | collector capability manifest 漂移：§9.2.3 `:2385` 与 §17.1.4 `:3925` 仍引用 §14.1 已不存在的顶层 `docker_capabilities` 字段；`provenance.collector_capabilities.strace_ptrace_v1` 把 security-opt `seccomp=unconfined` 混进 capability 列表；§9.2.3 选择顺序第3步 `ebpf_v1`（需 `BPF`+`PERFMON`）在 `docker_capabilities_allowed_max=["SYS_PTRACE","SYS_ADMIN"]` 下不可达 |
| P2-5 | §12.4 族1 定义"从未构造 plan"与 §8.5/§8.2 允许 STAGED/ADMISSION_DEFERRED 请求重验后记 `dense_ineligible` 冲突；族1（请求数）与族2（请求数 + 唯一 token 恒等式）不可互换，该路径归属未消歧 |

### F 矩阵

| Finding | 状态 |
| --- | --- |
| F-01 / F-02 | CLOSED（所有权分离、effective prefix、KV ledger、allocation/retract 全部经底座逐 API 核实）；exact-purity 侧面受本轮 P1 影响 |
| F-03 | CLOSED |
| F-04 | CLOSED |
| F-05 | CLOSED |
| F-06 | CLOSED |
| F-07 | CLOSED |
| F-08 | **PARTIAL**（P2-4） |
| F-09 | CLOSED |
| F-10 | **PARTIAL**（P2-2、P2-5） |
| F-11 | CLOSED |

结论：第八轮修订闭合了上轮全部 2 P1 + 4 P2，但暴露出一条更上游的
exact-Radix 插入抑制不对称问题——它同时是不变量矛盾与 primary estimand 的
未披露混淆源。**不得** plan freeze、**不得**升级状态、**不得**申请 branch 授权。

## 2026-07-30T08:47:42-07:00 Phase7.5 C40 计划第九轮只读 closure review = FAIL

- 对象：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表 `V1-r10`，5702 行，
  sha256 `d3e4610bb1203d5dd73b509f8dad2eb15014d4a070f9cf12456da668a4e4940b`。
- 底座固定 `worktrees/cross-store-substrate@0206f17b`（clean）。全程只读：
  未改计划文件、未建 branch、未跑测试/Docker/GPU。
- **Verdict = FAIL / NOT READY**；open = **0 P0 / 2 P1 / 5 P2**。
- 第八轮 1 P1 + 5 P2 **全部 CLOSED**（见下"上轮 closure"）。

### 上轮 closure（全部核实）

| 上轮 | 状态 | 闭合锚点 |
| --- | --- | --- |
| P1 exact-Radix 插入不对称 | CLOSED | §8.2.1 规则6 `:1397-1416`；§15.1 每臂表 `:3403-3412`；§13.1 `p75_exact_prefix_hit_tokens_total{arm}` / `p75_radix_insert_suppressed_requests_total{arm,outcome}` `:2977-2979`；§19.5 `:4796-4798`；§19.9 `:4859-4866`；TC-85/86/87 |
| P2-1 approx-metadata 前提 | CLOSED | §7.4 约束4 `:1129-1133` 改为"C40 在成功 admission commit 时显式置 True"；§8.2.1 规则1；底座 `schedule_batch.py:1080`、`common.py:104,200` 复核一致 |
| P2-2 audit schema | CLOSED | §8.6.1 `:1573-1578` 增 `primary_reason`/`secondary_reasons`/`pending_outcome`/`radix_insert_suppressed`/`recovery_attempted`；TC-88 |
| P2-3 geometry 值域 | CLOSED | §8.6.1 `:1556-1562` `outcome_geometry` + B-3/B-4/CL-I 唯一映射；与 §12.3/§13.1 值域一致；TC-89 |
| P2-4 collector manifest | CLOSED | §14.1 `collector_profiles` + `selected_profile_by_gate` + `docker_capabilities_allowed_max`（含 `BPF`/`PERFMON`）；§9.2.3 与 §17.1.4 命令与 profile 逐字段一致；TC-66/72/84 |
| P2-5 族1/族2 边界 | CLOSED（但引入新边界问题，见 P1-2/P2-4） | §12.4 族1 改以 `recovery_attempted=false` 定义 `:2863-2870`；TC-90 |

### 本轮新 P1（2 项，阻断）

1. **P1-1 admission-commit 与 forced-middle 钳制的顺序自相矛盾，两种读法都不可执行。**
   §8.6.3 步骤3 的 `needs_forced_middle`（`:1977`）与 owner-exclusion（`:1982-1987`）
   都要求 `c40_state == DENSE_PREFIX`，而该状态只能由步骤2 的
   `c40_on_admission_commit`（`:1928-1949`）建立（r10 把 lifecycle 入口移到
   admission 成功之后）。底座 `schedule_policy.py:1123-1180` 中
   `set_extend_range` 与 `can_run_list.append` 相邻，因此只有两种放置：
   (a) commit 在钳制**之后** ⇒ 新请求首轮 `needs_forced_middle` 恒为假 ⇒ 不钳制、
   不设 `new_chunked_req` ⇒ 每个 B-4 请求越过 `target_start` ⇒ 恒触发
   `c40_boundary_overrun` + 族4 `engineering_valid=false`，primary 路径整体作废；
   (b) commit 在钳制**之前**（与 §7.4 `:1138`、§8.6.3 步骤5 `:2072` 的
   "B-3 → add_one_req pre-range hook" 一致）⇒ owner-exclusion 的
   `return AddReqResult.CONTINUE`（`:1987`）恰好是步骤2 `:1950`
   "从 slot 成功/copy 开始到 append 之间**禁止普通return**"所禁止的返回，
   且此时 consume lease 已 pin，违反步骤2 postcondition（`:1953-1957`
   `consume_lease is None`）、§8.1.2 ADMISSION_DEFERRED 行与 §8.2 不变量
   （`STAGED`/`ADMISSION_DEFERRED` 必然不持 lease）。
   最小修复：显式冻结 `c40_on_admission_commit` 相对 `set_extend_range` 的位置；
   把 owner-exclusion 改为 commit **之前**的 gate（谓词改用 `STAGED` + staging plan），
   §8.2 相应删除 `DENSE_PREFIX --admit_deferred--> ADMISSION_DEFERRED` 边；
   或定义完整的 post-commit rollback 并补 TC。

2. **P1-2 post-commit deferral / rollback 未复位 `skip_radix_cache_insert`、
   `lifecycle_live`、`recovery_attempted`，且 `original_skip_radix_cache_insert`
   可被 latch。** 步骤2 `:1951-1957` 的 rollback 与步骤3 `:1986`、步骤7
   `:2160-2163` 只列资源（copied/provisional/lease/slot/kv），不含标志位；
   §8.2.1 规则5（`:1409`）只覆盖"pre-admission 失败 / state NONE / pure retract"，
   **不覆盖"已进入 lifecycle 后转 ADMISSION_DEFERRED"**。后果：
   (a) `skip_radix_cache_insert` 保持 True，该请求即使最终全 dense 完成，
   `common.py:200` → `cache_finished_req(is_insert=False)` 也永不写 exact Radix，
   违反 §7.5 exact 纯净不变量与 §15.1 冻结臂策略；
   (b) 下一次 commit `:1942-1943` 会把泄漏值重新存入
   `original_skip_radix_cache_insert`，污染永久化；
   (c) `req.c40_audit.radix_insert_suppressed` 只在 copy commit（步骤5e `:2131`）
   置位，故 `p75_radix_insert_suppressed_requests_total{arm,outcome}` 不会记录该
   泄漏，§15.1/§19.9 的"臂策略漂移 ⇒ engineering invalid"判据**检测不到**；
   (d) `recovery_attempted`（`:1929`，B-3 slot 耗尽路径也会置真）泄漏后，
   若该请求后续判 `dense_ineligible`，则与 §12.4 族1 的
   "`recovery_attempted=false`"定义冲突，TC-90 的两条规则对该路径互斥。
   最小修复：在 §8.2.1 增加"曾进入 lifecycle 后转 ADMISSION_DEFERRED ⇒ 恢复
   original、清 `lifecycle_live`、按 attempt 语义重置 `recovery_attempted`"，
   并扩展 TC-39/TC-45/TC-86。

### 本轮新 P2（5 项）

| ID | 内容 |
| --- | --- |
| P2-1 | G0q Exit 条件5（`:4915`）与 G1a Entry（`:4933`）、§20.1（`:5104`）要求"全部人工复核 alert 已裁决"，但 alert（patch-id / AST 签名**命中**）只能由 G1b 的 CR-3/CR-3b 在 new-branch diff 上产生，裁决记录也在 G1b 的 `quarantine-consumed.json`（§4.6 步骤3、§4.6.3）。该 Exit 在 G0q 时刻要么空洞、要么阻断 bootstrap 链 |
| P2-2 | §8.6.3 `:2003` 的 `reserve_exclusive_chunked_owner()` 把 `rem_chunk_tokens` 置0并扣尽 `rem_input_tokens`，使后续请求在底座 `schedule_policy.py:1143-1160`（`trunc_len<=0 → OTHER`）即被拒，早于任何 C40 hook ⇒ §8.6.2a 返回值表（`:1823-1830`）与 TC-37/TC-45 的"owner conflict 返回 `CONTINUE` 且继续扫描 waiting queue"不可达 |
| P2-3 | 中间 chunk 恒 `skip=true` 使 `stash_chunked_request`（`scheduler.py:2630`）→ `maybe_cache_unfinished_req`（`common.py:104`）早退，长 prompt 下 C40 臂在 prefill 期间不做 mid-flight 插树/去重/所有权转移，而 D0 臂会做：这是与 final insertion **并列的第二类臂间系统差异**，§15.1/§19.9/§13.1 只冻结并度量 final 策略，无任何 mid-flight 度量或披露项。同时 §8.2 `:1386`"最终 exact Radix 状态与 never-eligible 一致"与底座不符（`radix_cache.py:551-605` 的分块 insert/重 match/`cache_protected_len` 推进 vs 一次性 `cache_finished_req` 插入，节点切分与驱逐粒度不同） |
| P2-4 | §12.4 族2 定义（`:2879`）把 attempted fallback 绑定到状态 `DENSE_ISLAND_FALLBACK`，与步骤2 revalidate 失败路径（`:1960-1962`：`state=NONE`、同轮纯 dense 完成、仍须计 `dense_attempted_fallback`）及 §8.2 矩阵 `STAGED --rematch_invalid--> NONE` 冲突；property 测试若按状态断言会失败 |
| P2-5 | 新冻结的 per-arm Radix 指标缺日志合同与判据：§13.5.1（`:3101-3120`）的请求级字段表不含 exact-prefix-hit / suppression 字段，consolidator 无声明来源；§19.2b 与 §19.6（`:4804-4813`）的 `INVALID` 清单也未纳入 §19.9（`:4859-4866`）新增的三项强制披露，漏披露无后果 |

### F 矩阵

| Finding | 状态 |
| --- | --- |
| F-01 | **PARTIAL**（P1-1：ADMISSION_DEFERRED 持 lease 与 postcondition 冲突） |
| F-02 | CLOSED |
| F-03 | CLOSED |
| F-04 | **PARTIAL**（P2-1） |
| F-05 | CLOSED |
| F-06 | CLOSED |
| F-07 | CLOSED |
| F-08 | CLOSED |
| F-09 | CLOSED |
| F-10 | **PARTIAL**（P1-2、P2-4） |
| F-11 | CLOSED |

### 复核事实（抽样，全部属实）

- 底座行号：`scheduler.py:2630/2646/2755-2756/2966-2968/3017-3018/3602`；
  `schedule_batch.py:1080/1184/1211/2213/2217/2281/1310-1313/1576`；
  `common.py:104/200`；`radix_cache.py:496-551/551-605`；
  `schedule_policy.py:435-438/1001-1180`；
  `transfer.py` / `radix_backend.py` 的 `dense_prefill(*, target_start, length, reason)`
  keyword-only 签名与 `_contiguous_ranges` 均如计划所述。
- `already_computed` 记账推演正确：B-3 首轮 `new_cached == len(borrowed)`、
  B-4 copy 轮 `new_cached == 0`，与 §8.6.1a 注释一致。
- 统计复算无误：`z_sum=2.486475`、`2856.9`、`6.182558`、`2473.0`
  （`p_d=0.05/0.10/0.20 ⇒ 124/248/495`）、90/98 arm-cells、12+4 starts、
  人周 `8.25–12.25` 与 `4.00–8.00`；TC-1..TC-90 无缺号、无重号。

结论：r10 闭合了上轮全部问题，但 r10 自身的"lifecycle 仅在 admission 成功后进入"
改动在 admission seam 上引入了顺序矛盾与复位缺口。**不得** plan freeze、
**不得**升级状态、**不得**申请 branch 授权。

---

## 2026-07-30T09:30:55-07:00 Phase7.5 C40 计划第十轮只读 closure review = FAIL

- 对象：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表 `V1-r11`，5723 行，
  sha256 `8d824a3170427ab1b0f4b8a7304bbf5a9a129d595ecb194dd030154de48aa010`。
- 底座固定 `worktrees/cross-store-substrate@0206f17b`（clean）。全程只读：
  未改计划文件、未建 branch、未跑测试/Docker/GPU。
- **Verdict = FAIL / NOT READY**；open = **0 P0 / 1 P1 / 3 P2**。
- 第九轮 **2 P1 + 5 P2 全部 CLOSED**（见下"上轮 closure"）。

### 上轮 closure（全部核实）

| 上轮 finding | 状态 | 闭合锚点 |
| --- | --- | --- |
| P1-1 admission-commit 与 forced-middle 钳制顺序自相矛盾 | **CLOSED** | §8.6.3 步骤2 新增零资源 PRE-FLIGHT（`:1916-1927`）：owner-exclusion + 全部 capacity/chunk/alignment gate + 预计算不可变 `AdmissionPlan`，全部在 `STAGED` 完成；commit 后到 append **禁止普通 return**（`:1961-1963`）；步骤3 `:1990-1993` 改为 `if path == add_one_req: extend_start/end/forced_middle := admission_plan.*`；§8.2 矩阵 `DENSE_PREFIX` 行 `admit_deferred = —`；TC-91 |
| P1-2 post-commit deferral/rollback 未复位标志位 | **CLOSED** | 步骤2 冻结两条 rollback 分支（`:1964-1978`）：`rollback_for_deferral` 恢复 `skip=original` / `lifecycle_live=false` / `recovery_attempted` 事务前值 / 清 pending reason / `req.kv=None`，并给出显式 postcondition；`fallback_to_dense` 保留 attempted audit 但恢复 skip/live 并 `state=NONE`；TC-92 |
| P2-1 G0q Exit 要求裁决不可能产生的 alert | **CLOSED** | G0q Exit `:4941` "只产 reference signatures，不要求 adjudication"；G1a Entry `:4952`；G1b Exit `:4993`"new-branch match 全部人工裁决"；§4.6 尾注；TC-94 |
| P2-2 owner-conflict `CONTINUE` 分支不可达 | **PARTIAL → 见本轮 P1** | §8.6.2a `:1830-1837` 与 §8.6.3 步骤2 `:1922-1925` 已改为"保持底座 `OTHER`、不自造 CONTINUE"，但 §7.4 `:1133` 与 TC-37 `:2230` 未同步 |
| P2-3 中间 stash 抑制的第二类臂间系统差异未度量 | **CLOSED** | §8.2 `:1391-1393` 限定等价性范围；新增 `p75_radix_stash_suppressed_chunks_total{arm,outcome}`（§13.1 `:3021`）、§13.5.1 日志字段、§15.1 冻结表、§19.9、TC-93 |
| P2-4 族2 定义绑定 `DENSE_ISLAND_FALLBACK` | **CLOSED** | §12.4 族2 `:2891-2893` 改为"live state 可为 `DENSE_ISLAND_FALLBACK`，也可在 pre-admission rollback 后为 `NONE`；归属只看 sticky audit"；TC-90 |
| P2-5 新 per-arm Radix 指标缺日志字段与 INVALID 判据 | **CLOSED** | §13.5.1 `:3148-3149` 四字段；§19.6 INVALID 行 `:4830` 显式含"缺 per-arm exact-hit/stash-suppression/insert-suppression 字段或 policy 不符" |

### 本轮 open findings

- **P1（阻断）— `CONTINUE + not-added` 语义残留，两条冻结 TC 互斥。**
  §7.4 `:1133` 仍写"新的 forced-middle C40 请求返回 `CONTINUE` 且 `added=false`
  → ADMISSION_DEFERRED"，TC-37 `:2230` 同样断言"第二个返回 CONTINUE 且 deferred"；
  但 §8.6.2a 返回值表 `:1830`、§8.6.2a `:1837`（"所有 C40 deferral 保持底座原
  status，不创造 `CONTINUE+not-added` 新语义"）、§8.6.3 步骤2 `:1922-1925`
  与 TC-45 `:2238` 都要求 owner conflict 保持底座 `AddReqResult.OTHER`。
  三重后果：(a) TC-37 与 TC-45 对同一 owner-conflict 场景断言互斥，
  §20 G3 Exit 要求"TC-1..TC-94 全绿"因此永不可达；
  (b) 底座 `scheduler.py:3025-3058` 的 not-added 清理只在 `res != CONTINUE`
  分支执行，返回 `CONTINUE` 会跳过 §8.6.2a 要求的 `cleanup_not_added_req(req)`
  并让 waiting 循环继续加请求，正是被禁止的新语义；
  (c) TC-37 描述的"同轮两个 forced-middle C40"在 r11 冻结设计下不可复现：
  首个 owner 的 `reserve_exclusive_chunked_owner()` 把 `rem_chunk_tokens=0`，
  `budget_state()` 返回 `OTHER`，scheduler 立即 `break`，第二个请求本轮根本
  不会进入 `init_next_round_input`/`add_one_req`；owner-conflict guard 只在
  `has_chunked_req == True`（跨轮 owner）时可达。
  最小修复：§7.4 `:1133` 改为"保持底座 `OTHER`"；TC-37 改写为跨轮 owner 场景
  并断言 `OTHER` + `added=false` + 零资源，与 TC-45 合并口径。

- **P2-1 CL-I 未纳入冻结 admission 协议的枚举。**
  §8.6.1 `:1559` 冻结 `staging_case ∈ {B-3, B-4, CL-I}`，但 §8.6.3 步骤2 的
  `needs_forced_middle`（只判 B-4）与 `projected_prefix_len`（只给 B-3/B-4）、
  步骤5 的 copy-hook 路径映射（只给 B-3/B-4）、§8.6.1a `scheduling_prefix_len`
  `:1697`（只特判 `"B-3"`）、§8.6.2 的 B-0..B-5 表均无 CL-I 行。
  而 T23-6 `:4260` 要求 CL-I `next_extend_boundary = exact_length`
  （零 dense prefix），即 B-3 式"admission 内预分配 request slot + 立即 copy"，
  该路径未被规格覆盖。影响限于 conditional lane（默认关闭、G7 门控）。

- **P2-2 §19.5 强制汇报模板缺 stash-suppression 槽位，与 §19.6 INVALID 判据冲突。**
  §19.6 `:4830` 把"缺 per-arm …stash-suppression… 字段"列为 `INVALID`，
  §19.9 `:4877-4879` 要求按 arm 报告 `p75_radix_stash_suppressed_chunks_total`，
  但 §19.5"汇报模板（强制）"`:4812-4814` 只有 `exact_prefix_hit_tokens`、
  `radix insert suppressed`、`final_radix_insert_policy` 三行。
  逐字遵循冻结模板即自动判 `INVALID`。

- **P2-3 `rollback_for_deferral` 不复位 `lifecycle_ever_entered`，使 TC-46 在
  defer→ineligible 序列上不可满足。**
  步骤2 `:1968-1972` 明确复位 `recovery_attempted` 却未列 sticky
  `req.c40_audit.lifecycle_ever_entered`（§8.6.1 `:1580` 规定其只在最终
  outcome 写出后才清），TC-92 `:2285` 的枚举同样省略。
  路径：B-3 commit 成功 → 同轮 gate 拒绝 → `rollback_for_deferral`
  （`recovery_attempted := false`，`lifecycle_ever_entered` 保持 true）
  → 下一轮 source 重验失败判 `dense_ineligible`（族1）→ 请求以纯 dense 完成。
  此时 §8.1.2 `:1339` 的 `c40_cleanup_total{path="finish"}` 会因 sticky flag
  计一次，与同节"`dense_ineligible` 等从未进入 C40 lifecycle 的请求不写该
  metric"及 TC-46 `:2239`"ineligible 不写 cleanup"直接冲突。
  最小修复：`rollback_for_deferral` 同时把 `lifecycle_ever_entered` 恢复事务前值，
  或把 TC-46/§8.1.2 的 domain 判据改用 `recovery_attempted`。

### F 矩阵

| Finding | 状态 | 说明 |
| --- | --- | --- |
| F-01 | **CLOSED** | admission 顺序与钳制已由 PRE-FLIGHT/AdmissionPlan 冻结 |
| F-02 | CLOSED | borrowed/owned 分离、effective prefix、单一 release ledger |
| F-03 | **PARTIAL** | 本轮 P1：identity-membership 返回值契约仍有 `CONTINUE+not-added` 残留 |
| F-04 | CLOSED | quarantine 五类签名、G0q/G1b 裁决时序 |
| F-05 | CLOSED | 90% two-sided decision CI、四态判定、样本量 |
| F-06 | **PARTIAL** | 本轮 P2-1：CL-I 未纳入 admission 协议枚举 |
| F-07 | CLOSED | G0a→G0q→G1a→G0b→G1b 自举链 |
| F-08 | CLOSED | collector profiles（含 `ebpf_authority` 的 BPF/PERFMON）与 oracle 分层 |
| F-09 | CLOSED | W6a/W6b disjoint、5pp pre-data margin |
| F-10 | **PARTIAL** | 本轮 P2-2（汇报模板）、P2-3（sticky flag 与 cleanup domain） |
| F-11 | CLOSED | 恰好两处宿主可写挂载 |

### 复核事实（抽样，全部属实）

- 底座行号复核：`schedule_policy.py:435-438/786-790/800-868/1001-1184`
  （`_lock_node` 内 `total_tokens`/`input_tokens`/`trunc_len` 三段 return，
  `set_extend_range`+`can_run_list.append`+`_req_inc_lock_ref`+
  `_update_prefill_budget` 相邻，`budget_state()` 在 `rem_chunk_tokens<=0` 返回
  `OTHER`）；`scheduler.py:2630/2633-2674/2755-2756/2966-2968/3017-3020/
  3040-3058/3073-3079`（`assert self.chunked_req is None`、
  `added = req is can_run_list[-1]`、`res != CONTINUE` 才清理并 break）；
  `memory_pool.py:278-291`（`alloc` 对 `reusing` 断言
  `inflight_middle_chunks > 0 or kv_committed_len > 0`，B-3 预分配依赖此点）；
  `allocation.py:252-291/395-401`；`schedule_batch.py:1211-1220/1567/2213-2217/
  2273-2281/2317/2337/2367-2369/2511-2551`；`common.py:103-107/198-205`；
  `radix_cache.py:496-605`；`approx_kv/runtime.py:415-524`（`prefix_gap`）。
- 会计推演复核：B-3 首轮 `new_cached == len(borrowed)`、B-4 copy 轮
  `new_cached == 0`；`cur_rem_tokens = available_and_evictable - offset`
  确认 island 只能由 live allocator 反映、不得再加 offset。
- 统计复算无误：`z_sum=2.486475`、`2856.9`、`6.182558`、`2473.0`
  （`p_d=0.05/0.10/0.20 ⇒ 124/248/495`）、90/98 arm-cells、12+4 starts、
  人周 `8.25–12.25` 与 `4.00–8.00`；TC-1..TC-94 无缺号、无重号、无悬空引用。

结论：r11 闭合了第九轮全部 2 P1 + 5 P2，但 owner-conflict 返回值语义只改了
三分之二，遗留一条互斥 TC 对；另有 3 项 P2。**不得** plan freeze、
**不得**升级状态、**不得**申请 branch 授权。

## 2026-07-30T10:01:48-07:00 Phase7.5 C40 计划第十一轮只读 closure review = PASS_WITH_P2

- 计划文件：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表 `V1-r12`，5730 行，
  sha256 `a1a461b0ccec1b59787c26ade72ac772000d7df32301e5fed1dea7a2ef0c6b26`。
- 底座固定 `worktrees/cross-store-substrate@0206f17b`（clean）。
  本轮**未改计划文件、未建 branch、未跑测试/Docker/GPU**。
- Verdict = `PASS_WITH_P2`；open = `0 P0 / 0 P1 / 1 P2`。
  按 §23.1，review closure 的硬条件（open P0/P1 = 0/0）**已满足**；
  P2 允许保留但必须登记并在 disposition 披露。因存在 1 项 P2，
  **不**宣告无条件 `0/0/0 READY_FOR_PLAN_FREEZE`。

### 第十轮 open 项的闭合情况（1 P1 + 3 P2 全部 CLOSED）

| 上轮项 | 状态 | 证据 |
| --- | --- | --- |
| P1 `CONTINUE + not-added` | **CLOSED** | §7.4 `:1133` 改"保持底座 `OTHER`/not-added 路径并 defer"；TC-37 `:2233` 改写为跨轮 owner 场景，与 TC-45 `:2241`、§8.6.2a `:1830-1839`、§8.6.3 步骤2 `:1925-1927` 口径统一；全文再无 `CONTINUE+not-added`（`:1839` 明令禁止），底座 `scheduler.py:3025-3058` 的 not-added 清理与 `assert self.chunked_req is None` 均可达 |
| P2-1 CL-I 无 admission 规格 | **CLOSED** | `staging_case/outcome_geometry` `:1559-1565`、`scheduling_prefix_len` `:1660`、§8.6.2 冻结说明 `:1763-1764`（clip 后重写 `target_start=exact_length`、同步 `source_offset`、复用 B-3 零 dense-prefix 生命周期）、步骤2 projected prefix + island total-token gate `:1928-1931`、commit `:1938/:1957`、步骤5 `:2082`、TC-89/TC-95 |
| P2-2 §19.5 缺 stash 槽位 | **CLOSED** | 模板 `:4820-4823` 含 exact-hit / insert-suppressed / stash-suppressed / `final_radix_insert_policy`；§13.5.1 `:3136-3137` 日志字段；§19.6 `:4836`、§19.9 `:4882-4887`；TC-96 `:2292` |
| P2-3 sticky rollback | **CLOSED** | `rollback_for_deferral` `:1961-1966` 同时复位 `lifecycle_ever_entered` / `recovery_attempted` / `radix_insert_suppressed` 并清 pending reason；与 §8.1.2 cleanup domain、§12.4 族1/族2 归属、TC-46/TC-90/TC-92 一致 |

### 本轮唯一新 open 项（P2-1，非阻断）

- **§19.5 强制汇报模板缺 `ci_method` 槽位**。§19.2b `:4736-4737` 把 "CI 方法"
  列入四态判定的强制同现项并规定"缺任一项判 `INVALID`"，且 `:4718-4719`
  要求 **n 小时同时报 bootstrap 与 t 的 90% CI 并取更保守者**；
  但 §19.5 `:4808` 只有单一 `CI90 [L90=___, U90=___] (decision CI)` 槽位，
  既无 `ci_method`，也无第二个 CI 槽。`:4806` 的 `method=<normal+t|bootstrap>`
  绑定在 `n_confirmatory`（属 §19.2a.4 的样本量校正方法），
  而 `:4816` 的 `E_cond` 行反而写明了 `cluster-bootstrap`，可见槽位缺失是
  模板侧遗漏而非有意省略。默认 `n_min = 4` 即小样本路径，因此
  **逐字遵循强制模板产出的 summary 会缺一项 §19.2b 必报字段**。
- 最小修复：`mu_theta` 行补 `ci_method=<bootstrap|t|conservative_of_both>`
  （必要时补第二个 CI90 槽），并在 TC-96 的断言清单中加入该字段。

### 非阻断观察（不计入 open，供下次修订顺手处理）

1. §8.6.3 步骤1 `:2121` 的 `B-2 -> dense_ineligible(c40_exact_overlap_unsupported)`
   未带 `CLIP=1 ⇒ CL-I` 旁注；因该步显式"按 §8.6.2 分类"、而 §8.6.2 冻结说明
   `:1760-1764` 已给出 clip 路径，不构成矛盾，仅影响逐字可执行性。
2. §18.6b `:4468` 引用 `c40_effective_task_rate`，该名字未在 §13/§18 定义；
   等价量为 `eligible_task_rate` / `effective_paired_tasks`。
3. §19.5 `:4811` 与 `:4817` 的 `E_work_pooled` 槽位重复（无害冗余）。

### F 矩阵

| Finding | 状态 | 说明 |
| --- | --- | --- |
| F-01 | CLOSED | admission PRE-FLIGHT / AdmissionPlan / 钳制顺序冻结 |
| F-02 | CLOSED | borrowed/owned 分离、effective prefix 成对替换、单一 release ledger |
| F-03 | **CLOSED** | identity membership 返回值契约统一；owner conflict 保持底座 `OTHER` |
| F-04 | CLOSED | quarantine 五类签名、G0q 只产 reference、G1b 才裁决 |
| F-05 | **PARTIAL** | 决策 CI 口径（90% two-sided、四态、样本量）本身闭合；本轮 P2-1 = 强制模板缺 `ci_method` 槽位 |
| F-06 | **CLOSED** | CL-I 复用 B-3 零 dense-prefix admission/copy 生命周期，geometry 与 headline 隔离齐备 |
| F-07 | CLOSED | G0a→G0q→G1a→G0b→G1b 自举链与 BLOCKED 中断语义 |
| F-08 | CLOSED | collector profiles / capability / primary-supplemental oracle 分层 |
| F-09 | CLOSED | W6a/W6b disjoint、5pp pre-data margin、NO_COVERAGE 阈值 pre-data 冻结 |
| F-10 | **CLOSED** | terminal taxonomy、四计数族、cleanup domain、sticky audit 与 Radix 系统指标全链闭合 |
| F-11 | CLOSED | 宿主可写挂载恰好两处 + `/scratch` `/tmp` ephemeral 例外 |

### 复核事实（抽样，全部属实）

- `schedule_policy.py:685-705`（`budget_state()` 在 `rem_chunk_tokens<=0` 返回
  `OTHER`，`rem_input_tokens<=0` 同）、`:830-868`（`add_chunked_req` 的
  hybrid-SWA parked `return req`、`_update_prefill_budget(0, ...)`、
  `return req if truncated else None`）、`:1000-1184`
  （`_lock_node` 内三段 return、`trunc_len<=0 ⇒ OTHER`、
  `has_chunked_req` 参数在底座 **未被使用**，故 C40 owner-exclusion 属新增语义）。
- `scheduler.py:2630`（`stash_chunked_request`）、`:2745-2756`（parked chunk 不 stash）、
  `:2966-2968`（chunked 续算不带 tree_cache）、`:3017-3020`、
  `:3025-3058`（`res != CONTINUE` 才做 not-added 清理并 `break`）、
  `:3073-3079`（`assert self.chunked_req is None` + `inflight_middle_chunks += 1`）。
- `memory_pool.py:273-304`：`write()`；`alloc() -> Optional[List[int]]`（耗尽返回
  `None`，B-3 事务依赖此点）；`reusing` 断言
  `inflight_middle_chunks > 0 or kv_committed_len > 0` ⇒ B-3 必须在 alloc 成功后
  立即置 `kv_committed_len = exact_length (>=1)`，计划 `:1943-1948` 正是如此。
- `allocation.py:252-282`（`alloc_req_slots` fail-loud，Hybrid/Mamba headroom 仅对
  `HybridReqToTokenPool` 生效 ⇒ 步骤1 已 fail-close，故 B-3 直连 `alloc([req])` 安全）、
  `:316-333`（`prefix_tensors` / `alloc_req_slots`）。
- `schedule_batch.py:1080`（`skip_radix_cache_insert` 初值）、`:1184`
  （`set_extend_range` 唯一写入点）、`:1211`（`init_next_round_input`）、
  `:1567`（`reset_for_retract`）、`:2213-2217`、`:2272-2273`
  （`assert seq_len - pre_len == extend_range.length`，证明 `prefix_lens` 与
  extend 起点**必须成对**改用 helper）、`:2317`（`new_cached = pre_len -
  already_computed`，随后 `already_computed = seq_len`）、`:2337-2339`
  （`cached_tokens_device` 用 `len(prefix_indices)`，计划要求保持不变）。
- `common.py:103-107`（`maybe_cache_unfinished_req` 被 skip 拦截）、
  `:198-206`（`cache_finished_req(is_insert=... and not skip)` 与
  `assert (req.req_pool_idx is None) == (req.kv is None)`）。
- 会计推演：B-3 首轮 `new_cached == len(borrowed)`；B-4 copy 轮 `new_cached == 0`；
  island 仅由 live allocator 反映，不得再加 offset charge。
- 统计复算：`z_sum=2.486475`、`2856.9`、`6.182558`、`2473.0`
  （`p_d=0.05/0.10/0.20 ⇒ 124/248/495`）、D-3 `90` arm-cells / chunk4096 start `38`、
  每 restart `98` cells / `3` starts、`n=4 ⇒ 12 (+4) starts`、
  Track B `2+1+3=6 <= 8`、人周 `8.25–12.25`（WP0a–WP10 分项求和一致）与
  CL-A..CL-I `4.00–8.00`。
- 命名与编号：TC-1..TC-96 无缺号/重号/悬空引用；全部 `c40_*` reason 均在 §12.2
  有定义，且与 §2.6 canonical inventory 及其 prefix-family 无重名。

结论：r12 闭合了第十轮全部 1 P1 + 3 P2，open P0/P1 已归零；仅余 1 项
非阻断 P2（强制模板缺 `ci_method` 槽位）。在产出 versioned
`plan-review-*.json`（记载 0/0）与 `c40-quarantine-manifest.json` 之前，
**不得**升级计划状态；branch、实现、Docker 与 GPU 仍未授权。

## 2026-07-30T10:32:19-07:00 Phase7.5 C40 计划第十二轮只读 closure review = PASS

- 计划文件：`IMPLEMENTATION_PLAN_PHASE7_5_C40.md`，版本表 `V1-r13`，5743 行，
  sha256 `25cbac98c65c6b67e6d2de23c0cc33e28d7c6f6b3ca60b13cd49a6a3b3770a42`。
- 底座固定 `worktrees/cross-store-substrate@0206f17b`（clean）。
  本轮**未改计划文件、未建 branch、未跑测试/Docker/GPU**。
- Verdict = `PASS / READY_FOR_PLAN_FREEZE`；open = **`0 P0 / 0 P1 / 0 P2`**。
  §23.1 的 review closure 硬条件（open P0/P1 = 0/0）已满足，且本轮
  **P2 亦归零**，因此明确宣告 `READY_FOR_PLAN_FREEZE`
  （freeze 仍以两份 versioned artifact 与用户授权为前提，见文末）。

### 第十一轮 open 项与观察的闭合情况

| 上轮项 | 状态 | 证据 |
| --- | --- | --- |
| P2-1 §19.5 缺 `ci_method` | **CLOSED** | 模板 `:4818` 新增 `ci_method : <bootstrap \| t \| conservative_of_both>`；与 §19.2b `:4718-4719`（n 小时同报 bootstrap/t 取更保守者）、`:4749`（"CI 方法"为强制同现项，缺任一判 `INVALID`）一致；新增 `TC-97` `:2294` 断言"小 n 同时计算 bootstrap/t、模板记录 `ci_method=conservative_of_both` 及最终 L90/U90，缺失即 INVALID" |
| 观察 1：步骤1 B-2 行缺 `CLIP` 旁注 | **CLOSED** | §8.6.3 步骤1 `:1904-1905` 改为 `clip flag=0 -> dense_ineligible(c40_exact_overlap_unsupported)` / `clip flag=1 -> 重写为 CL-I staging_case 并复用 B-3 生命周期`，与 §8.6.2 冻结说明 `:1763-1764`、TC-95 逐字一致 |
| 观察 2：`c40_effective_task_rate` 未定义 | **CLOSED** | §18.6b `:4453-4458` 新增"派生量（冻结）"：`c40_effective_task_rate = effective_paired_tasks / n_tasks`、`copy_coverage(task) = 该 task copied tokens / 该 task 全部 target prefill tokens`，并规定报告 median/p25/p75/min/max 且与 `w_quality` **不得互换**；新增 `TC-99` `:2296` |
| eBPF capability/security profile | **CLOSED** | §14.1 `provenance` 段拆出 `collector_capabilities` / `collector_security_opts` 两个映射，并新增 profile `ebpf_authority = {caps:[BPF,PERFMON], security_opts:[]}`；与 §9.2.3 选择顺序第三步、§17.1.4 collector 表、`docker_capabilities_allowed_max` 一致；新增 `TC-98` `:2295`（实际 profile 精确绑定、不得等于 allowed-max） |
| 观察 3：`E_work_pooled` 槽重复 | 保留（无害） | 模板 `:4823` 与 `:4829` 仍各有一槽，二者语义与 `estimand_role` 标注相同，不触发 §19.2b/§19.6 的任何 `INVALID` 判据，故不计入 open |

### 本轮全文件 closure 核验（要点）

- **TC 集合**：TC-1..TC-99 共 **99 条定义**，无缺号、无重号、无悬空引用；
  §7.1 `:864`、WP5 验收 `:3756`、G3 Exit `:5033` 三处口径统一为 `TC-1..TC-99`。
- **§19.5 模板 vs §19.2b 强制字段**：`n` / `n_pilot` / `s_pilot` / `delta0` /
  `delta1` / **CI 方法** / `[L90,U90]` / `exp(L90)`/`exp(U90)` / descriptive
  `[L95,U95]` / `E_work_pooled(descriptive)` / `w` / `r` / `C_selector`
  **全部在模板中有槽位**，逐字遵循模板不再自动 `INVALID`。
- **四态判定**：`POSITIVE`/`NEGATIVE`/`SMALL_POSITIVE_BELOW_MDE`/`INCONCLUSIVE`
  两两互斥且穷尽（`L90>delta0` 蕴含 `U90>delta0`；`U90<0` 蕴含 `L90<0`）。
- **统计复算**：`z_sum=2.486475`、`(2.486475/0.046520)^2=2856.86≈2856.9`、
  `s=0.04 -> 5`、`s=0.08 -> 19`；`(z.95+z.80)^2=6.182558`、`/0.05^2=2473.0`、
  `p_d=0.05/0.10/0.20 -> 124/248/495`；D-3 `90` arm-cells、chunk4096 start `38`、
  每 restart `98` cells / `3` starts、`n=4 -> 12(+4)` starts、Track B `2+1+3=6<=8`；
  人周 `8.25–12.25`（14 分项求和一致）与 CL-A..CL-I `4.00–8.00`。
- **命名空间**：全部 `c40_*`（126 个标识符）均有定义位置，`p75_exact_prefix_hit_tokens_total` /
  `p75_radix_insert_suppressed_requests_total` / `p75_radix_stash_suppressed_chunks_total`
  在 §13.1 定义并在 §15.1 / §19.5 / §19.9 / §13.5.1 一致引用。
- **交叉引用**：全文 §x.y 引用全部可解析；`§9.10` / `§10.3.1` 明标为 authority
  报告的章节，非本文件悬空引用。
- **底座抽样复核（全部属实）**：`schedule_policy.py:435-438`（`AddReqResult` 三值）、
  `:830-840`、`:1073`（`_lock_node` 覆盖 `:1126-1174` 的 `set_extend_range` /
  `can_run_list.append` / `new_chunked_req` / `_req_inc_lock_ref`）、
  `:1098-1104`、`:1158-1166`（`trunc_len<=0 -> OTHER`）；
  `scheduler.py:2630`、`:2745-2756`、`:3017-3018`、`:3025-3058`
  （not-added 清理只在 `res != CONTINUE` 分支，故计划把 owner-conflict 保持
  `OTHER` 后该清理可达）、`:3068`（`assert self.chunked_req is None`）；
  `schedule_batch.py:1080`、`:1184`；`memory_pool.py:273`；
  `common.py:103-107`、`:198-202`。

### F 矩阵

| Finding | 状态 | 说明 |
| --- | --- | --- |
| F-01 | CLOSED | admission PRE-FLIGHT / AdmissionPlan / 钳制顺序冻结 |
| F-02 | CLOSED | borrowed/owned 分离、effective prefix 成对替换、单一 release ledger |
| F-03 | CLOSED | identity membership 契约；owner conflict 保持底座 `OTHER` |
| F-04 | CLOSED | quarantine 五类签名、G0q 只产 reference、G1b 才裁决 |
| F-05 | **CLOSED** | 决策 CI 口径 + 强制模板 `ci_method` 槽位 + TC-97 断言齐备 |
| F-06 | CLOSED | CL-I 复用 B-3 零 dense-prefix 生命周期，staging 分类与 headline 隔离齐备 |
| F-07 | CLOSED | G0a→G0q→G1a→G0b→G1b 自举链与 BLOCKED 中断语义 |
| F-08 | **CLOSED** | collector capability/security 分字段 + `ebpf_authority` profile + TC-98 |
| F-09 | CLOSED | W6a/W6b disjoint、5pp pre-data margin、NO_COVERAGE 阈值与派生量冻结 |
| F-10 | CLOSED | terminal taxonomy、四计数族、cleanup domain、sticky audit、Radix 系统指标 |
| F-11 | CLOSED | 宿主可写挂载恰好两处 + `/scratch` `/tmp` ephemeral 例外 |

### 非阻断观察（不计入 open，供后续修订顺手处理）

1. §19.5 `:4823` 与 `:4829` 的 `E_work_pooled` 槽位重复；且 `CI95` 续行
   `:4819` 现位于 `ci_method` 行之下，视觉上与 `mu_theta` 行分离。
2. `w_quality`（§18.6b `:4468`、SR-8 `:5179`）参与 pre-data 冻结阈值判定，
   但只以"time-weighted coverage"指向 §19.3 的 `w`，未给出独立公式；
   `eligible_task_rate`（`:4341`/`:4427`）同样只有名字无公式。
3. TC-97/98/99 被挂在 WP5 验收 `:3756` 与 G3 Exit `:5033`，而其对象分别属
   §19.5（consolidator，WP8）、§14.1（WP0b/WP8）、§18.6b（WP12）。
   因 WP8 早于 G3，**不构成循环依赖**，仅属 WP 归属口径偏松。
4. §24.2 代码交付索引未列 `overlap_clip.py` / `reason_inventory.py` /
   `collectors/` / `live_corpus.py`（该节已声明"见 §7.1"，§7.1 完整）。
5. §14.2 的 RESULT_MANIFEST 绑定句只写 "Docker mounts 与 capabilities"，
   未显式点名 security opts（经 plan manifest 自哈希间接绑定）。

结论：`V1-r13` 闭合了第十一轮唯一 P2 与三项观察，并补齐 eBPF capability/security
profile 与 TC-97..99；本轮 open = **0 P0 / 0 P1 / 0 P2**，
Verdict = `PASS / READY_FOR_PLAN_FREEZE`。
仍需先产出 versioned `evidence/review/plan-review-*.json`（自哈希 + 绑定文档
sha256、记载 `open P0/P1 = 0/0`）与 `evidence/review/c40-quarantine-manifest.json`
（G0q，隔离 reviewer 执行），才可由主会话把状态升级为
`Reviewed Candidate / PENDING USER AUTHORIZATION`；branch、实现、Docker 与 GPU
仍为 `PENDING USER AUTHORIZATION`。

## 2026-07-30T11:07:48-07:00 Phase7.5计划artifact闭合并升级状态

- `IMPLEMENTATION_PLAN_PHASE7_5_C40.md`当前为`V1-r13`：
  `Reviewed Candidate / PENDING USER AUTHORIZATION`。
- 最终plan SHA256：
  `b894e276960b72c1cf7963ca7f5957f89f62471b11eb006548fc867d7f28bedf`。
- 最终独立review：
  - verdict=`PASS / READY_FOR_PLAN_FREEZE`；
  - open P0/P1/P2=`0/0/0`；
  - F-01..F-11=`CLOSED`。
- review artifact：
  - `evidence/review/plan-review-phase7_5-c40-v1-r13.json`；
  - file SHA256=`63159af25ec29e7f13cd254cd2823cd51aab92989bd451d7db455e15203293e9`；
  - self SHA256=`63bf3b83c040504ac731f8ab413fbd59557dff0d50a26bb514368fd38e51d1b0`。
- G0q隔离quarantine artifact：
  - `evidence/review/c40-quarantine-manifest.json`；
  - `quarantine_status=AVAILABLE`、`contains_source_text=false`；
  - file SHA256=`cbc873faf2b3d3683dc755ea261b93b30e96c454ca4af6b4c78fa2094b59da61`；
  - self SHA256=`a0fe94ae5a1122edea36580356b8c8b608e11b32729a8056dffd9fb91ff440be`；
  - coverage：82 commits、340 exclusive blobs、307 file AST signatures、
    3917 function AST signatures、172 allowed shared signatures。
- G0q只产出reference hashes/signatures；未来new-branch diff的patch-id/AST
  match与人工裁决仍属于G1b，不得提前视为已通过。
- 本轮没有创建Phase7.5 branch，没有修改implementation repo，没有运行
  Docker/CPU测试/GPU。执行授权仍全部为false。
- 计划、artifact与authority更新已发布到
  `project/code-agent-kvcache-docs`，主发布commit：
  `ce3f786699cf813cfb84d8191752a0dbf236cd48`。
- 下一执行边界：等待用户明确授权`branch_creation_authorized`及后续逐Gate/
  逐Track预算；不得因plan ready自动启动实现。

## 2026-07-31T13:37:59-07:00 Phase8及后续阶段状态澄清

- 当前**没有**正式、已评审或已授权的Phase8执行计划，也没有已编号的Phase9+
  计划。
- Phase8目前仅为`Potential Scope`，不会由Phase7或Phase7.5结果自动触发。
- 创建Phase8计划的硬前置：
  1. Phase7.5实际执行完成G11；
  2. engineering/mechanism/system/quality/publication五类disposition齐备；
  3. 用户明确授权创建Phase8计划；
  4. 使用独立plan、manifest、branch/结果目录与review链。
- 可能的Phase8方向只能由Phase7.5结果决定，当前均为候选而非项目事实：
  - 若C40正确、覆盖充分且有性能收益：做生产级规模化、并发workflow、
    host/prefetch与source-version integration；
  - 若质量安全但性能不足：优先优化selector coverage、调度/复制开销与
    hardware/model scale；
  - 若出现质量损伤：转向calibrated repair/KVCOMM-style reconstruction，
    不直接扩大部署规模。
- Phase9及以后只可在Phase8完成后另行定义；当前不预注册、不分配预算。
