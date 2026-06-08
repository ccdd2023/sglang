# AgentTemplateKV acceleration implementation notes

Date: 2026-06-07

## Naming boundary

AgentTemplateKV is the name of our method. KVFlow/KVCOMM are reference
baselines and low-level mechanisms used for comparison. In code, legacy
`lossy_*` field names and `SGLANG_LOSSY_*` environment switches are kept for
backward compatibility only.

## Implemented runtime path

- Device-first codebase prefetch no longer depends on HiCache. Scheduler calls
  `RadixCache.agenttemplatekv_prefetch_codebases()` before the older
  host-storage prefetch path.
- Finished Planner-style requests can protect exact-content codebase anchors
  when their `codebase_prefetch_hints` match stored anchor signatures.
- Later Implementer/Reviewer requests carrying the same exact-content hint
  count a real device hit via `codebase_prefetch_device_hit_count` and the new
  `agenttemplatekv_*` telemetry fields.
- Protected anchors hold the source radix node lock, have a TTL controlled by
  `SGLANG_AGENTTEMPLATEKV_PREFETCH_TTL_S` (default: 60 seconds), and consume a
  `steps_to_use` budget.
- Large zero-fill gaps are rejected by default when `gap_len` exceeds
  `SGLANG_AGENTTEMPLATEKV_MAX_ZERO_GAP` (default: 16 tokens).

## Implemented template path

`benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py` now defaults to
`--prompt-layout agenttemplatekv`.

The AgentTemplateKV prompt layout keeps codebase segments stable and early:

1. Shared task preamble.
2. Canonical ordered `code_baseN` fenced blocks.
3. Agent instruction / output schema.
4. Issue, tests, allowed paths, and dynamic agent-step text.

Use `--prompt-layout legacy` to reproduce the older layout.

## Key telemetry

- Existing compatibility fields:
  - `codebase_prefetch_hint_count`
  - `codebase_prefetch_matched_tokens`
  - `codebase_prefetch_success_count`
  - `codebase_prefetch_device_hit_count`
  - `lossy_anchor_match_gap_len`
- AgentTemplateKV fields:
  - `agenttemplatekv_prefetch_hit_count`
  - `agenttemplatekv_prefetch_miss_count`
  - `agenttemplatekv_prefetch_protected_tokens`
  - `agenttemplatekv_prefetch_newly_protected_tokens`
  - `agenttemplatekv_prefetch_consumed_count`
  - `agenttemplatekv_prefetch_expired_tokens`
  - `agenttemplatekv_rejected_large_gap_count`

## Suggested replay

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow
/home/gfy/.conda/envs/sglang-kvflow/bin/python benchmark/multi_workflow/bench_swe_generated_patch_kvcomm.py \
  --prompt-layout agenttemplatekv \
  --output-schema diff \
  --out-dir results/agenttemplatekv_accel/swe_patch_agenttemplatekv
```
