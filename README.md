# code-agent-kvcache

项目已完成 AST/code-structure KV prior-art 与多模型 novelty 评估，当前准备从干净 SGLang 基线建立可测量、安全的 repository KV prototype。

## 协作与原型仓库

- 仓库：[`ccdd2023/sglang`](https://github.com/ccdd2023/sglang)
- 用途：项目交流以及 prototype 代码实现。
- 账号：涉及该仓库的 GitHub 操作必须明确使用系统中已有的 `ccdd2023` 账号，不依赖当前默认账号。
- 权限：已于 2026-07-12 使用 `ccdd2023` 身份确认具有 `ADMIN` 权限，默认分支为 `main`。

## 当前研究资料

- 历史研究工作区：`/home/chris/Workspaces/kvcache-research`
- KVFlow：[`research/papers/KVFlow-2507.07400.pdf`](research/papers/KVFlow-2507.07400.pdf)
- KVCOMM：[`research/papers/KVCOMM-2510.12872.pdf`](research/papers/KVCOMM-2510.12872.pdf)
- 研究综合：[`research/RESEARCH_SYNTHESIS.md`](research/RESEARCH_SYNTHESIS.md)
- Yu Guofan / AgentTemplateKV 分支审查：[`research/YU_GUOFAN_BRANCH_REVIEW.md`](research/YU_GUOFAN_BRANCH_REVIEW.md)
- AST-indexed KV prior art 与 novelty 报告：[`research/AST_INDEXED_KV_CACHE_NOVELTY_REPORT.md`](research/AST_INDEXED_KV_CACHE_NOVELTY_REPORT.md)
- AST-indexed KV 逐步研究详解：[`research/AST_INDEXED_KV_CACHE_STEP_BY_STEP.md`](research/AST_INDEXED_KV_CACHE_STEP_BY_STEP.md)
- Git / Codebase version-aware KV prior art：[`research/GIT_CODEBASE_VERSIONED_KV_CACHE_PRIOR_ART.md`](research/GIT_CODEBASE_VERSIONED_KV_CACHE_PRIOR_ART.md)
- 2025–2026 七分段 source-version-aware KV 最终复核：[`research/CODE_VERSION_KV_10_AGENT_PRIOR_ART_REPORT.md`](research/CODE_VERSION_KV_10_AGENT_PRIOR_ART_REPORT.md)
- KVCOMM 在历史 SGLang 上的完整复现可行性：[`research/KVCOMM_SGLANG_FEASIBILITY_REPORT.md`](research/KVCOMM_SGLANG_FEASIBILITY_REPORT.md)
- Vast.ai RTX PRO 6000 接入与实验工作流：[`research/VASTAI_RTX_PRO_6000_WORKFLOW.md`](research/VASTAI_RTX_PRO_6000_WORKFLOW.md)
- Phase 4权威artifact索引：[`research/PHASE4_RESULT_MANIFEST.json`](research/PHASE4_RESULT_MANIFEST.json)
- Phase 5请求级重算结果：[`research/PHASE5_RECALCULATED_METRICS.json`](research/PHASE5_RECALCULATED_METRICS.json)
- 两篇论文均已通过 alphaXiv 定位、收藏并下载。

## 文档入口

1. [`HANDOFF.md`](HANDOFF.md)：新会话首先阅读，了解当前状态和下一步。
2. [`PROJECT.md`](PROJECT.md)：项目事实、计划、进度、决策与可共享思路的固定来源。
3. [`IMPLEMENTATION_PLAN_LATEST.md`](IMPLEMENTATION_PLAN_LATEST.md)：当前最新、可执行的实施计划（V4）。
4. [`IMPLEMENTATION_PLAN_V3_ARCHIVED.md`](IMPLEMENTATION_PLAN_V3_ARCHIVED.md)：上一版实施计划（V3，只读归档）。
5. [`IMPLEMENTATION_PLAN_V2_ARCHIVED.md`](IMPLEMENTATION_PLAN_V2_ARCHIVED.md)：历史实施计划（V2，只读归档）。
6. [`IMPLEMENTATION_PLAN_V1_ARCHIVED.md`](IMPLEMENTATION_PLAN_V1_ARCHIVED.md)：历史实施计划（V1，只读归档）。
7. [`CONSOLIDATED_PHASE4_PHASE6_REVIEW.txt`](CONSOLIDATED_PHASE4_PHASE6_REVIEW.txt)：Phase4/5审计、corrected rerun和双模型review。
8. [`TRACKING.md`](TRACKING.md)：按时间追加的讨论与执行过程记录。
9. [`.github/copilot-instructions.md`](.github/copilot-instructions.md)：后续 Copilot 会话必须遵循的协作规则。

## 协作流程

每次开始工作前先阅读交接文档和项目主文档。每轮有效讨论后更新项目主文档并追加讨论记录；出现阶段切换、关键决策、功能完成、阻塞变化或下一步改变时，同时更新交接文档。
