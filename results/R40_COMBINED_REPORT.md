# R40 — Combined report (2026-07-09)

> ⚠ **n=15 supersedes n=5 (2026-07-09)** — 本报告 Phase 2 的 `git apply 0/5 vs 1/5` 数字来自 n=5 verdict harness，已被 n=15 scale-up 推翻。
> - 当前推荐 **R32 (FRAC=0.30)** 而非 R38b（n=15 上 R38b 略慢且略差 — 6.7% vs R32 9.8% vs lossless 10.7% type_match/25）
> - 客观 judge 是 R40 Phase 2 的 5-agent coding_pipeline + `git apply --check`（已部分保留作 code-gen 信号；见 deck slide 28）
> - HKVD 机制经 `results/hkvd_by_position_20260709/` 实测验证（pos1 K_dev +7.2% > pos5）
> - **权威源**：`results/SCALE15_HKVD_REPORT.md` + `results/CODE_AWARE_LOSSY_KV_PROGRESS_R28_R39.html` deck slide 20/28

Three phases shipped: **TTFT breakdown instrumentation (P1)** +
**5-agent coding_pipeline task with objective git-apply judge (P2)** +
**type-aware FRAC override (P3)**.

All three plan files are landed in source; Phase 1 has a **known architectural
block** (documented) and Phase 3 is **byte-equal no-op verified** on pandas 0.x.
Phase 2 produced a **new objective accuracy signal** that splits the configs in
a way the old verdict task could not.

---

## Phase 1 — TTFT breakdown instrumentation (PARTIAL)

**Source landed (5 files):**
- `python/sglang/srt/observability/req_time_stats.py` — 7 setters added to
  `SchedulerReqTimeStats` (was missing `set_radix_prefix_ms` / `set_chunk_plan_ms`
  / `set_copy_ms` / `set_gap_prefill_ms` / `set_head_recompute_early_ms` /
  `set_head_recompute_late_ms` / `set_chunk_plan_done_time`).
- `python/sglang/srt/managers/tokenizer_manager.py` — propagation block
  moved out of `if self.enable_metrics` guard; per-batch delta tracking so
  additive setters don't double-count; `convert_to_output_meta_info` moved
  out of the same guard.
- `python/sglang/srt/entrypoints/openai/serving_chat.py` — `ttft_breakdown`
  surfaced in `response_metadata` for both streaming and non-streaming paths.
- `python/sglang/srt/managers/schedule_batch.py` — `time_stats` Union
  extended to include `SchedulerReqTimeStats` (so pickle survives the type
  check).
- `benchmark/multi_workflow/bench_kvcomm_ttft_stress.py` — `extract_ttft_breakdown`
  reads from response metadata + body metadata (fallback chain);
  `post_chat_stream` hoists `ttft_breakdown` to top-level metadata.

**Result (R38b 5×5 verdict task, `results/baseline_ours_r38b_5x5_verdict_r40_phase1/`):**

```
N=25
  ttft_ms                          avg=    692.18ms nonzero=25/25
  ttft_tokenize_ms                 avg=     16.56ms nonzero=25/25   ← works
  ttft_radix_prefix_ms             avg=      0.00ms nonzero= 0/25   ← architectural block
  ttft_chunk_plan_ms               avg=      0.00ms nonzero= 0/25
  ttft_copy_ms                     avg=      0.00ms nonzero= 0/25
  ttft_gap_prefill_ms              avg=      0.00ms nonzero= 0/25
  ttft_head_recompute_early_ms     avg=      0.00ms nonzero= 0/25
  ttft_head_recompute_late_ms      avg=      0.00ms nonzero= 0/25
  ttft_decode_first_token_ms       avg=      0.00ms nonzero= 0/25
```

**Architectural block — `radix walk happens AFTER first batch output is sent`.**
The radix/chunk timing lands in scheduler's in-memory `req.time_stats` AFTER
the prefill's first batch output has been pickled and sent, so the chat
completion's first content chunk picks up the empty snapshot. Memory file
`r40-ttft-breakdown-architecture-block-2026-07-09.md` documents the issue +
3 fix options (A: second zmq channel, B: /metrics endpoint, C: event-log
post-process). **Decision (2026-07-09):** keep the source fixes (they're
correct for future Option A/B/C) and pivot to Phase 2; do not block Phase 2
on this. Future-work tag in the plan.

---

## Phase 2 — 5-agent coding_pipeline task with objective git-apply judge

**Source landed:**
- `benchmark/multi_workflow/bench_kvcomm_ttft_stress.py` — new
  `task_mode="coding_pipeline"` branch in `build_slot_messages` with
  per-agent instructions + output formats (coder emits `diff --git ...`,
  tester emits `<test_result>PASS|FAIL</test_result>`, reviewer emits
  `<review>...</review><verdict>PASS|FAIL</verdict>`, refactorer emits
  `<refactor>...</refactor>`, integrator synthesizes + emits
  `<final_verdict>PASS|FAIL</final_verdict>`).
- `benchmark/multi_workflow/bench_giant_codebase_reuse.py` — the integrator
  (agent 5) gets the prior 4 agents' outputs in its `extra_context`,
  indexed by `agent_idx`.
- `results/lossy_alg_round21/scripts/score_coding_pipeline.py` — 273-line
  scorer that runs `git apply --check` against `results/giant_codebase/pandas_src`
  for the coder's diff + parses tester/integrator/reviewer tags.
- New launchers:
  - `results/coding_pipeline_5x5/launchers/run_lossless_coding_pipeline.sh`
  - `results/coding_pipeline_5x5/launchers/run_ours_r38b_coding_pipeline.sh`
- `--agent-max-tokens 768` (was 64 default, too small to fit a real diff).

**Result (5×5, `--agent-max-tokens 768`):**

| Config | diff_found | git apply | tester PASS | integrator PASS | combined |
|---|---|---|---|---|---|
| **lossless** | 5/5 = 100% | **1/5 = 20%** | 5/5 = 100% | 5/5 = 100% | **1/5 = 20%** |
| **R38b** (FRAC_EARLY=0.60, FRAC_LATE=0.15, EARLY_N=2) | 5/5 = 100% | **0/5 = 0%** | 4/5 = 80% | 2/5 = 40% (3 UNK) | **0/5 = 0%** |

**Key insight:** The new objective judge **discriminates** the two configs in
a way the old `VERDICT: PASS/FAIL` task could not (R38b was 50.0% agreement vs
lossless on the verdict task; on coding_pipeline it's 0/5 vs 1/5). The
lossless baseline produces 1 git-applyable diff; R38b produces 0. Both configs
emit "diffs" 5/5, but most are wrong (hallucinated line numbers / paths). The
7B coder on pandas 0.x can't write a clean patch from just the cross-position
shared context — even the lossless reference fails 4/5.

The chunk-reuse overhead of R38b **costs** more than it saves on the coder
agent (it produces a less-aligned diff because the prefix cache is warm to the
rotated segment, and the 60% early-FRAC + 15% late-FRAC means the head
recompute dominates the agent-1 prefill). On the other 4 agents (tester /
reviewer / refactorer / integrator) R38b's TTFT is faster, but accuracy
suffered.

**`ttft_tokenize_ms` works for the new task too:** lossless 28.8ms, R38b 17.9ms.
The other 7 fields stay 0 due to the Phase 1 architectural block.

---

## Phase 3 — Type-aware FRAC override (NO-OP VERIFIED)

**Source landed:**
- `python/sglang/srt/mem_cache/ast_chunker.py` — `ChunkSpan` now carries
  `typed_signature: str` (extracted from AST `ast.unparse()` of function args
  + return annotation) and `type_complexity: int` (0-10 score: count of
  annotated arg sites + return-type depth). Both default to `""` / `0` for
  untyped code (pandas 0.x).
- `python/sglang/srt/mem_cache/radix_cache.py` — new env var
  `SGLANG_CHUNK_TYPE_AWARE_FRAC=1` (default OFF) bumps FRAC by
  `_weight × type_complexity / 10` when the chunk has `typed_signature != ""`
  and `type_complexity > 2`. For untyped code, `chunk.typed_signature == ""`
  → the block is a no-op.
- New launcher:
  - `results/type_aware_frac_verify/launchers/run_off.sh`
  - `results/type_aware_frac_verify/launchers/run_on.sh`

**Verification (`results/type_aware_frac_verify/`):**

- `outputs.jsonl` (the actual LLM-generated text) is **byte-equal** between
  off and on runs. ✓
- `rows.csv` 26 semantic columns (case_id, agent_id, agent_idx, mode,
  radix_only_prefix_len, l2_wholeslot_reused_tokens, c2_chunk_reused_tokens,
  codeaware_reused_tokens, placeholder_chunk_pool_*, output_chars, all 6 radix
  / chunk / head_recompute / decode_first_token ms) are **byte-equal** across
  all 25 rows. ✓
- Only `ttft_ms` / `e2e_ms` / `ttft_tokenize_ms` differ, which is expected
  run-to-run wall-clock variation.

**Conclusion:** type-aware FRAC is a true no-op on pandas 0.x. To actually
exercise the path, need a typed codebase (e.g. pylint / astropy / typeshed);
out of scope for this session.

---

## Files modified (uncommitted)

- `python/sglang/srt/observability/req_time_stats.py` (+111)
- `python/sglang/srt/managers/tokenizer_manager.py` (+107)
- `python/sglang/srt/entrypoints/openai/serving_chat.py` (+17)
- `python/sglang/srt/managers/schedule_batch.py` (+2)
- `python/sglang/srt/mem_cache/radix_cache.py` (+84)
- `python/sglang/srt/mem_cache/ast_chunker.py` (+89)
- `benchmark/multi_workflow/bench_kvcomm_ttft_stress.py` (+152)
- `benchmark/multi_workflow/bench_giant_codebase_reuse.py` (+35)
- `benchmark/multi_workflow/analyze_fair_ab.py` (+56)
- `benchmark/multi_workflow/analyze_kv_isolation.py` (+363)

## New files (uncommitted)

- `results/lossy_alg_round21/scripts/score_coding_pipeline.py` (273 lines)
- `results/coding_pipeline_5x5/launchers/{run_lossless,run_ours_r38b}_coding_pipeline.sh`
- `results/type_aware_frac_verify/launchers/{run_off,run_on}.sh`

## Results (uncommitted data)

- `results/baseline_ours_r38b_5x5_verdict_r40_phase1/` — Phase 1 verification
- `results/coding_pipeline_5x5/{lossless,r38b}/` — Phase 2 A/B
- `results/type_aware_frac_verify/{off,on}/` — Phase 3 no-op verify

---

## What I did NOT do (per memory constraints)

- ❌ No commit / push (user must authorize).
- ❌ No changes to the LaTeX / progress deck.
- ❌ No LMCache integration (R24 — separate workstream).
- ❌ No HKVD attention-kernel hook (R31 — multi-week).
- ❌ Did not fix Phase 1 architectural block (documented for future work).

---

## Open questions for next session

1. **Phase 1 followup**: implement Option A (second zmq channel) or B
   (Prometheus /metrics) to surface the 6 broken TTFT fields. Need to
   decide if R41 picks this up.
2. **Phase 2 followup**: with the objective judge now in place, run a
   larger N (e.g. 25 cases) to get a stable accuracy signal. The 1/5 vs 0/5
   result on 5 cases is suggestive but not significant.
3. **Phase 3 followup**: run the verification on a typed codebase (pylint,
   astropy) to actually exercise the type-aware bump.
4. **Coding pipeline tuning**: 7B coder can't write a clean diff on pandas
   0.x without ground-truth context. Options: (a) increase the model's
   context with sibling-window files, (b) drop the coder agent entirely
   and just measure tester/integrator accuracy on a known patch.
