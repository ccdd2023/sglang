# Graph-Aware Lossy Smoke Recheck, 2026-06-17

This is a live smoke recheck for the graph-aware patch harness. Candidate tests
were skipped; the run validates generation, JSON edit synthesis, git
`apply --check`, and reuse metadata.

- Result dir: `results/code_graph_kv_reuse/pass1_graph_aware_smoke2_20260617`
- Model: `/home/gfy/models/Qwen2.5-7B-Instruct`
- Dataset: `results/repo_level_datasets/swe_verified_100_instances.json`
- Instance file: `results/code_graph_kv_reuse/data/graph_pass1_overlap_cases.json`
- Cases: 2
- Graph policy: `call_neighborhood_1hop`
- Candidate tests: skipped
- Command: see `summary.json`

| mode | n | diff extracted | apply ok | generation errors | search-not-found | exact signature match | mean cached | mean elapsed ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless` | 2 | 2 | 2 | 0 | 0 | 0 | 12193.0 | 2283.5 |
| `lossy` | 2 | 1 | 1 | 0 | 1 | 2 | 12194.0 | 1424.2 |
| `lossy_prefetch` | 2 | 1 | 1 | 0 | 1 | 2 | 12195.0 | 1331.6 |
| `graph_aware_lossy` | 2 | 1 | 1 | 0 | 1 | 2 | 2151.0 | 1431.4 |

## Reading

- `graph_aware_lossy` is connected to the live harness: exact content signature
  matched on `2/2` cases.
- It produced one git-applyable JSON edit out of two cases. The other failure
  is `search not found`, matching the dominant failure mode in the previous
  13-case readiness run.
- This smoke is not a pass@1 result because candidate tests were skipped.
