# Cold-cache and concurrent benchmark audit

Date: 2026-06-12.

## Existing artifacts

| Artifact | Cases | Regime | Comparable to 06-03 warm headline? | Verdict |
|---|---:|---|---|---|
| `qwen2_5_7b_100_true_cold_forceflush` | 100 | `--flush-cache-per-case`, 1 client | Partly: same model, dataset, manifest, and modes; different 06-09 harness/commit and a true cold protocol, so latency scale should not be pooled with the 06-03 warm-cache headline. | Use as robustness sanity only. |
| `qwen2_5_7b_20_concurrent4_smoke` | 20 | 4 concurrent clients | No: same model/dataset/modes, but only 20 cases and a smoke run. | Use as smoke evidence only; do not headline. |

## Safe paper claim

The 06-09 true-cold run confirms exact-content hits still occur under explicit per-case cache flush (`agenttemplatekv_exact_reuse` exact-content hit rate 1.00 on 100/100 cases), but its latency should not be pooled with the 06-03 warm-cache headline. The 20-case concurrent4 run confirms the harness path works under 4 clients, but it is not the full 100-case concurrent benchmark.

## Follow-up command

Run a full 100-case concurrent4 benchmark in a fresh out-dir before promoting concurrent latency to a headline:

```bash
/home/gfy/.conda/envs/sglang-kvflow/bin/python benchmark/multi_workflow/bench_coding_kvflow_prefetch.py \
  --model /home/gfy/models/Qwen2.5-7B-Instruct \
  --dataset results/repo_level_datasets/swe_verified_100_instances.json \
  --manifest results/repo_level_datasets/manifest_100.json \
  --max-cases 100 \
  --concurrent-clients 4 \
  --disable-hierarchical-cache \
  --port 31341 \
  --out-dir results/coding_kvflow_prefetch/qwen2_5_7b_100_concurrent4
```

## 2026-06-12 follow-up attempt

The full concurrent4 rerun was attempted from the repo cwd, from `/`, and with
`PYTHONPATH` / `VIRTUAL_ENV` unset. All direct invocations failed before the
benchmark started because `bench_coding_kvflow_prefetch.py` imports
`matplotlib.pyplot`, and the active `sglang-kvflow` Python environment has a
`.pth`/site path pointing at
`results/swebench_local_envs/repos/matplotlib__matplotlib-20488/lib`, which
shadows the real matplotlib package and lacks the compiled
`_c_internal_utils` extension. Removing the SWE-bench repo path from
`sys.path` avoids the shadowing but then reveals that the sglang env has no
regular matplotlib installation. The existing 20-case concurrent4 artifact is
therefore kept as smoke-only evidence until the driver is run from an env with
a clean matplotlib install.
