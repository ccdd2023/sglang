# 仓库协作指令

## 交流语言

- 与用户交流时始终使用中文。
- 不使用日语，也不使用非必要的英文叙述。
- 代码、命令、文件名、API 名称和其他必要技术标识可以保留英文。

## 每次会话开始

1. 先阅读根目录 `HANDOFF.md`。
2. 再阅读 `PROJECT.md`。
3. 查看 `TRACKING.md` 的最新记录。
4. 涉及研究或架构时阅读 `research/RESEARCH_SYNTHESIS.md`。
5. 以这些文件中的当前状态为准，不依赖旧聊天上下文进行猜测。

## 强制文档维护

- `PROJECT.md` 是项目更新、可共享思路、讨论结论、进度、计划和决策的固定事实来源。
- 每轮有效讨论或执行后，更新 `PROJECT.md`，并向 `TRACKING.md` 追加一条带 ISO 8601 时间戳的记录。
- 不删除或改写 `TRACKING.md` 的既有记录；需要纠正时追加新记录。
- 出现阶段切换、架构或范围决策、功能完成、重大阻塞、风险变化或下一步改变时，更新 `HANDOFF.md`。
- `HANDOFF.md` 应保持为简洁的最新快照，至少包括当前状态、已完成事项、下一步、约束、关键决策和相关文件。
- 重要信息不得只存在于聊天、临时计划或代理内部状态中。

## GitHub 协作仓库

- 项目交流和 prototype 代码实现仓库是 `https://github.com/ccdd2023/sglang`。
- 目标仓库的指定操作账号是系统中已有的 `ccdd2023`。
- 执行读取以外的 GitHub 操作前必须核实身份与权限；不要假设 GitHub CLI 或其他工具的当前默认账号是 `ccdd2023`。
- 应使用账号级的显式认证方式，避免为了单次操作改变全局默认账号。
- 不得输出、记录或提交 GitHub token、认证头或其他凭据。
- 最近一次访问验证时间为 2026-07-12T18:03:17-07:00，当时 `ccdd2023` 对该仓库具有 `ADMIN` 权限，默认分支为 `main`。

## 研究与实现边界

- KVFlow 负责 workflow-aware cache priority、eviction、CPU backup、prefetch 和 scheduling。
- KVCOMM `2510.12872` 负责 base KV、context-dependent offset、RoPE relocation、anchor interpolation 和 dense fallback。
- “可变编码”不是 KVCOMM 原文术语，不得把 delta compression、AST index 或 SGLang HiCache 写成论文已有能力。
- 超大 Codebase 必须按 artifact/AST span 预计算和索引，不得描述为单一连续 KV Cache。
- AST 是结构索引和辅助 gating 信号，不替代 embedding distance。
- 实现顺序必须优先保证正确性：exact cache → 受控 KVCOMM reconstruction → dense fallback。
- 固定 workflow 为 `Architect -> Coder -> Debugger`，Debugger 失败后可条件返回 Coder。

## 实施原则

- 未确认的需求必须明确标记为待确认，不得伪造项目事实。
- 修改代码时同步更新直接相关的文档、计划和进度。
- 完成工作前验证结果，并将可供下个会话继续工作的状态写入交接文件。
