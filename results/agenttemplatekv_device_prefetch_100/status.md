# AgentTemplateKV Device-First Prefetch 100-Case Replay Status

Date: 2026-06-07

## Status

- A real serving smoke replay completed at `results/agenttemplatekv_device_prefetch_smoke_3/`.
- The smoke run used HiCache disabled and hierarchical cache disabled, so it directly tests the AgentTemplateKV device-first protected-anchor path.
- Protected-anchor telemetry is nonzero:
  - `agenttemplatekv_prefetch_hit_count`: 6 total
  - `codebase_prefetch_device_hit_count`: 6 total
  - `agenttemplatekv_prefetch_protected_tokens`: 55,795 total
  - `agenttemplatekv_prefetch_newly_protected_tokens`: 28,127 total
  - `agenttemplatekv_rejected_large_gap_count`: 0
- Exact-content reuse hit in the AgentTemplateKV mode for all 3 smoke cases.

## 100-Case Attempt

Command attempted:

```bash
/home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/bench_coding_kvflow_prefetch.py \
  --dataset results/repo_level_datasets/swe_verified_100_instances.json \
  --manifest results/repo_level_datasets/manifest_100.json \
  --max-cases 100 \
  --files-per-case 2 \
  --disable-hierarchical-cache \
  --out-dir results/agenttemplatekv_device_prefetch_100 \
  --port 30124 \
  --server-timeout 240
```

The SGLang server loaded the model and completed token-shape compile/capture, but the wrapper did not reach the case loop in the current turn. The server process was terminated cleanly; no 100-case result table was produced.

## Next Action

Reuse the now-updated benchmark script and rerun the same command in a longer compute window. If startup remains slow, reduce launch overhead by lowering `--max-total-tokens`, disabling token-shape capture if supported by the local SGLang version, or running a 30-case intermediate replay first.
