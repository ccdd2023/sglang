# HANDOFF PROMPT — sglang-kvflow (2026-07-02)

> **Quick-start card.** Paste this into a new Claude Code session at
> `/home/gfy/CodeMAS_Project/sglang-kvflow` to resume with full context
> in seconds. For the full prompt with constraints and 必读文件 order,
> see [`NEXT_SESSION_PROMPT.md`](./NEXT_SESSION_PROMPT.md). For the
> single source of truth on goal/state, see
> [`CANONICAL_TARGET.md`](./CANONICAL_TARGET.md).

---

## The Project

**sglang-kvflow** (AgentTemplateKV) — SGLang fork for Coding Multi-Agent
System serving. EuroSys 2026 submission.

**One goal:** make Coding-MAS serving **fast + correct** via code-aware
KV cache reuse ONLY (no scheduling tricks).

**Two bars:**
1. **Speed** — TTFT speedup vs lossless baseline.
2. **Accuracy** — real token-F1 vs lossless, not worse than general (non-code-aware) reuse.

---

## Current State (2026-07-02)

| Mechanism | Reuse | TTFT | F1 vs lossless | Status |
|---|---|---|---|---|
| lossless (no reuse) | 0 tok | 932 ms | 1.000 | reference |
| MULTI_SLOT (5 slots) | ~7100 tok | **124 ms (7.5×)** | **0.000** | **speed bar MET** |
| precompute SYNC (host pool) | ~870 tok | 923 ms | 0.374 | lossy, no speedup |
| precompute LAYERED (load_stream) | ~870 tok | 918 ms | 0.508 | **NOT transferable** |
| precompute DEVICE-RESIDENT (diag) | ~830 tok | 960 ms | 0.447 | slower than lossless |

### R26/R27 3-way (post-2026-07-06) — verdict-task accuracy measured

| Config | Model × Agents | Speedup | FAIL_acc | Note |
|---|---|---|---|---|
| R19 BEST | 7B-Coder × 5 | 1.29× | **60%** | accuracy-optimal |
| R26 | 3B-Instruct × 3 | **2.014×** | 27% | speed-optimal |
| R27 | 3B-Coder × 3 | 1.900× | 0% | avoid for critique |

Counterintuitive: R27 (3B-Coder) is WORST at FAIL detection — Coder training biases toward PASS.

- **Branch:** `fix/placeholder-pool-activation`, HEAD is at the R26/R27 wrap-up commit.
- **Working tree:** should be clean after wrap-up commit.
- **Speed bar MET** by MULTI_SLOT (7.5× in 7B regime, 2× in 3B × 3 regime).
- **Accuracy bar (verdict task): R19 BEST 60% > R26 27% > R27 0%.** Pick by priority.
- **Precompute pipeline built** (Phases 1-3, 4B/4C, commit `628aeab83`).
  Validated as building block for true CacheBlend. F1=0.374 lossy,
  no speedup. Phase 6 proves H2D transfer is NOT the bottleneck.
- **Phase 7 LAYERED F1=0.508 verdict:** NOT a correctness fix.
  3 fence hypotheses all FALSIFIED. Decision: not pursuing further.
- **R26/R27 session wrap:** see [results/SESSION_WRAP.md](./results/SESSION_WRAP.md).

---

## Hard Constraints (verbatim)

- 加速**只**来自更多复用，不准加 KV-cache 调度 trick。
- L3 MiniLM 语义 k-NN **默认 OFF** (research only, deprecated)。新 feature 默认 OFF。
- 实验结果统一输出到项目 `results/` 子目录，**不用 /tmp**。
- >3 case 必须加 `--disable-overlap-schedule --max-running-requests 1` (绕过 `_delete_leaf` race)；`--force-evict` **不是**真实 server flag。
- 不要重新 track `swebench_local_envs/` (21G) 或 `results/codebase_kv/` (1.2GB/run)。
- commit/push **只在用户明确要求时**；commit 结尾加 `Co-Authored-By: Claude <noreply@anthropic.com>`。
- 不要打印/外泄 SiliconFlow API key。
- **F1 测量**：读 `outputs.jsonl` 的 `output_text`，**不要信** `rows.csv` 的 `output` / `output_token_f1_vs_baseline` 列（可能是 placeholder/空）。初版 precompute A/B 误报 F1=1.000 就是这个原因。

---

## 必读文件（按顺序）

1. **[`CANONICAL_TARGET.md`](./CANONICAL_TARGET.md)** — 单一目标 + 当前状态。
2. **[`HANDOFF.md`](./HANDOFF.md)** — 当前 session 状态、bug 细节、open items。
3. **[`NEXT_SESSION_PROMPT.md`](./NEXT_SESSION_PROMPT.md)** — 完整 prompt 模板。
4. **[`results/CODE_AWARE_LOSSY_KV_PROGRESS.md`](./results/CODE_AWARE_LOSSY_KV_PROGRESS.md)** — 完整时间线 + 结果表 + fundamental limit。
5. **[`results/kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md`](./results/kvcomm_ab/precompute_ab_report/ANOMALY_FINAL.md)** — 最近一轮（precompute + Phase 7）。
6. auto-loaded memory `~/.claude/projects/-home-gfy/memory/MEMORY.md` — 关键不变量：
   - `c2-cacheblend-lossy-not-safe-2026-06-28` — fundamental limit
   - `multi-slot-copy-2026-07-01` — speed bar MET
   - `precompute-kv-ab-2026-07-02` — precompute + Phase 7 verdict
   - `cross-position-fix-works-2026-06-30` — content-derived slot_id
   - `l3-placeholder-knn-deprecated` — L3 OFF
   - `_delete-leaf-bug-2026-06-24` — >3-case 必加 flag
   - `output-path` — `results/`, not `/tmp`

---

## Retracted Claims (不要 cite)

- L4 "~1.49× production-ready" — broken over-copying path
- AST-gated L3 "1.448× both bars met" — cached_tokens 混淆 radix + code-aware (见 `fair-measurement-prefix-conflation-2026-06-30`)
- 旧 "1.31× / 20% reuse" — MiniLM 语义路径
- **LAYERED F1=0.508** — Phase 7 调查证实非 transferable 修正，仅作现象记录

---

## 接下来可能的方向（等用户指示）

| # | 方向 | 收益 | 风险 / gate |
|---|---|---|---|
| **A** | **Precompute 异步 overlap 优化**（plan Phase 4A-REVISED） | 速度 — 打破当前 0× speedup；用 SGLang `LayerDoneCounter` 让 H2D 与 prefill 在 load_stream 逐层重叠 | 用户 sign-off；~30 GPU-min |
| **B** | **True CacheBlend 实现**（attention recompute for copied chunks） | 速度 + 精度 — 唯一能同时达两个 bar 的路 | 用户 sign-off（fresh algorithmic change）；precompute 是其前置（已建） |
| **C** | partial-share niche 产品化 | 精度（AST 在 partial-share 0.62 > L2 0.51 已达标） | 无；速度 0.96× |
| **D** | deck / 文档 / 汇报材料完善 | 沟通 | 无 |
| **E** | 重跑 fair giant-codebase A/B | 真实 F1（现 2.10× 的 F1 是占位） | ~20 GPU-min |
| **F** | Cleanup commit（10 旧 doc 删除 + 未 track 原始结果） | 工作区干净 | 用户确认形状 |

**未指示前**：不要主动改算法/起服务器/commit。

---

## Common Commands

```bash
# L4 + precompute unit tests
python -m pytest test/registered/unit/mem_cache/ \
    test_ast_chunker.py test_placeholder_chunk_pool.py \
    test_placeholder_chunk_pool_read.py test_placeholder_chunk_pool_policy.py \
    test_codebase_kv_loader.py -v

# Precompute KV extraction (one-time, populates results/codebase_kv/pandas_5case/)
SGLANG_PRECOMPUTE_CODEBASE_KV=1 \
SGLANG_PRECOMPUTE_KV_DIR=results/codebase_kv/pandas_5case \
SGLANG_PRECOMPUTE_MAX_FILES=25 \
  python scripts/precompute_codebase_kv.py \
    --manifest results/giant_codebase/tasks/pandas__pandas__1000/manifest.jsonl \
    --repo-root results/giant_codebase/pandas_src \
    --model Qwen/Qwen2.5-Coder-7B-Instruct

# Precompute pipeline A/B (5 case × 5 agent, 7B-Coder)
bash results/kvcomm_ab/run_7b_precompute_ab.sh                    # LAYERED (default)
# Variants in results/kvcomm_ab/run_7b_precompute_ab_*.sh (8 total)

# Lossless reference (7B, position-shift)
bash results/kvcomm_ab/run_7b_precompute_ab_lossless.sh

# Fair A/B analysis (F1 from outputs.jsonl, NOT rows.csv)
python benchmark/multi_workflow/analyze_fair_ab.py \
    --baseline results/kvcomm_ab/7b_precompute_ab_lossless \
    --experimental results/kvcomm_ab/7b_precompute_ab_layered \
    --lossless results/kvcomm_ab/7b_precompute_ab_lossless

# MULTI_SLOT (speed bar) reference
bash results/kvcomm_ab/run_7b_multislot_l2.sh
bash results/kvcomm_ab/run_7b_lossless.sh
```

---

## Key Files

| Path | Why |
|---|---|
| `CANONICAL_TARGET.md` | Goal + state (SINGLE SOURCE OF TRUTH) |
| `HANDOFF.md` | Active session state |
| `NEXT_SESSION_PROMPT.md` | Full prompt template |
| `results/CODE_AWARE_LOSSY_KV_PROGRESS.{md,html,pdf}` | Master timeline + visual deck |
| `results/kvcomm_ab/precompute_ab_report/{COMPARISON.txt,ANOMALY_FINAL.md,SUMMARY.txt}` | Most recent cycle (precompute + Phase 7) |
| `results/kvcomm_ab/run_*.sh` | Reproducible launchers for every measured config |
| `scripts/precompute_codebase_kv.py` | Offline KV extractor |
| `python/sglang/srt/mem_cache/codebase_kv_loader.py` | Server-start disk→CPU loader |
| `python/sglang/srt/mem_cache/radix_cache.py` | L1/L2/L3(deprecated)/L4/C2/MULTI_SLOT/PRECOMPUTE + Phase 7 hooks (default OFF) |
| `python/sglang/srt/mem_cache/ast_chunker.py` | L4 server-side AST chunker |
| `benchmark/multi_workflow/bench_giant_codebase_reuse.py` | Giant-codebase driver (`--precompute-*`) |
| `benchmark/multi_workflow/analyze_fair_ab.py` | Fair A/B analyzer (decomposed counters) |

---

**Last refreshed:** 2026-07-02, after precompute + Phase 7 cycle.
**Next refresh trigger:** direction A (precompute async overlap) lands,
direction B (true CacheBlend) decision, or a fresh fair multi-case
headline number.