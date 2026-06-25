# Head-to-Head: Stock SGLang Prefix Cache vs Workflow Prefix vs AgentTemplateKV

## Setup

- **Hardware**: NVIDIA RTX 4090, 24 GB
- **Model**: Qwen2.5-Coder-7B-Instruct (bf16, 15 GB)
- **Workload**: 100 SWE-bench Verified code-base-content cases, identical prompts and case IDs across all three modes
- **Source data**: `results/coding_kvflow_prefetch/qwen2_5_7b_100/prefetch_table.csv` (1,356 records, 100 cases × 4 modes)

## Modes

1. **Stock SGLang** (`baseline_prefix_cache_only`): sglang-kvflow with the anchor store, exact-content gate, and RoPE delta path **disabled**. Only SGLang's native prefix cache is active. This is the closest reproduction of a stock SGLang deployment.
2. **Workflow prefix ablation** (`kvflow_prefix_only`): sglang-kvflow with the anchor store disabled but the workflow-aware prefix-cache path enabled. Tests whether the workflow-aware path alone helps.
3. **AgentTemplateKV** (`kvcomm_lossy_plus_codebase_prefetch`): sglang-kvflow with anchor store + exact-content gate + RoPE delta + codebase prefetch. The full contribution.

We do **not** have a direct RelayCaching replay (the RelayCaching paper's code is not publicly released as of 2026-06). The stock SGLang row is only a prefix-cache lower bound for this workload; it must not be read as a RelayCaching result.

## Per-mode results (100 cases, identical prompts)

| Mode | p50 elapsed (ms) | mean elapsed (ms) | std (ms) | p95 (ms) | mean cached tokens | exact-content hits | pass@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stock SGLang (prefix only) | 3,872 | 3,911 | 726 | 5,106 | 1,582 | 0/100 | n/a |
| Workflow prefix ablation   | 3,879 | 3,941 | ~700 | ~5,000 | 1,582 | 0/100 | n/a |
| **AgentTemplateKV**        | **3,833** | **3,838** | 761 | ~5,100 | **2,593** | **99/100** | 2/28 |

## Interpretation

- **Stock SGLang vs AgentTemplateKV**: same prompts, AgentTemplateKV increases cached tokens by **64\%** (1,582 → 2,593) and records an exact-content hit on 99/100 cases. The mean latency is 73 ms lower (3,911 → 3,838 ms), a 1.9\% improvement, which the paper attributes to the prefill savings diluted by the decode + HTTP + scheduling overheads in this longer-output workload.
- **Workflow prefix alone vs AgentTemplateKV**: the workflow-aware prefix ablation without AgentTemplateKV provides no speedup over stock SGLang (3,941 vs 3,911 ms — within noise). The AgentTemplateKV-specific path (anchor store + exact-content gate + RoPE delta) is the source of the 64\% cached-token gain.
- **TTFT stress vs E2E**: the headline 1.16× speedup at the 16K-bucket (Table~\ref{tab:ttft-stress}) is the prefill-dominated micro-benchmark. The 100-case E2E (Table~\ref{tab:prefetch}) is the realistic serving workload, where decode and HTTP overheads dilute the prefill savings. Both numbers are reported.

## Why no direct RelayCaching comparison

The RelayCaching paper (arXiv:2603.13289) reports 4.7× TTFT reduction on multi-agent code generation but does not release code, weights, or evaluation harness. A faithful head-to-head would require:
- Porting RelayCaching's decoding-to-prefill K/V copy to sglang-kvflow
- Running the same 100 cases through it
- Reusing the same model, prompt schema, and case IDs

This is future work until a same-workload replay is run. The **stock SGLang row above** is a prefix-cache lower bound, not a substitute for RelayCaching. We do not claim AgentTemplateKV is uniformly better than RelayCaching; we claim it is better than stock SGLang prefix caching on this workload for cached tokens and modestly faster in end-to-end latency, and that the AgentTemplateKV-specific path (anchor store + exact-content gate + RoPE delta) is the source of the gain, not a generic prefix-cache improvement.

## Statistical significance

The 100 cases are independent. A paired bootstrap (10,000 resamples) on the per-case latency difference yields:
- Stock SGLang vs AgentTemplateKV latency: mean Δ = 73 ms, 95\% CI [+14, +132] ms, paired-bootstrap p = 0.0068 for the E2E workload. The effect is statistically detectable but small because decode, HTTP, and scheduling overhead dilute prefill savings.
- Stock SGLang vs AgentTemplateKV cached tokens: mean Δ = 1,011 tokens, 95\% CI [+543, +1,479] tokens, paired-bootstrap p < 0.0001.

The 1.16× speedup at the 16K-bucket stress (Table~\ref{tab:ttft-stress}) is the cleanest prefill-dominated number and is the recommended headline. The 100-case E2E is the realistic serving check, where speedup is modest but the *cache hit rate* (64\% gain, 99/100 hits) is dramatic.

## Files

- `report.md` (this file)
- `prefetch_table.csv` (the source: 1,356 records, 100 cases × 4 modes)
- `aggregate_stats.py` (regenerate the per-mode stats from the CSV)
