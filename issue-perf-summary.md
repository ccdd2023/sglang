# Import Performance Summary

Benchmarks ran inside the dev container with the shared venv at `/tmp/sglang-issue10492-venv-min2`.

- Before baseline: `/home/chris/Workspaces/sglang/.codex-tmp/main-baseline-20260308-173917`
- After baseline: `/home/chris/Workspaces/sglang`
- Warm-ups per item: `1`
- Measured runs per item: `10`
- Raw data:
  - committed compact JSON:
    - `/home/chris/Workspaces/sglang/test/manual/issue_10492/perf-results/main-10runs.json`
    - `/home/chris/Workspaces/sglang/test/manual/issue_10492/perf-results/fix-10runs.json`
  - local extended samples:
    - sample directories under `/home/chris/Workspaces/sglang/.codex-tmp/perf-results/`

## Real Time

| Benchmark | Before mean | After mean | Delta | Mean pct | Before median | After median | Delta | Median pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `import_sglang` | 0.639s | 0.024s | -0.616s | -96.3% | 0.639s | 0.024s | -0.615s | -96.3% |
| `import_model_config` | 6.944s | 1.751s | -5.193s | -74.8% | 6.928s | 1.755s | -5.173s | -74.7% |
| `import_reasoning_parser` | 3.132s | 0.036s | -3.096s | -98.9% | 3.140s | 0.036s | -3.104s | -98.9% |
| `import_moe_utils` | 6.321s | 2.146s | -4.175s | -66.0% | 6.329s | 2.147s | -4.183s | -66.1% |
| `import_engine` | 7.912s | 7.039s | -0.873s | -11.0% | 7.897s | 7.028s | -0.868s | -11.0% |
| `import_scheduler` | 7.669s | 4.694s | -2.976s | -38.8% | 7.653s | 4.690s | -2.963s | -38.7% |

## User Time

| Benchmark | Before mean | After mean | Delta | Mean pct | Before median | After median | Delta | Median pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `import_sglang` | 2.037s | 0.017s | -2.021s | -99.2% | 2.038s | 0.016s | -2.022s | -99.2% |
| `import_model_config` | 7.310s | 2.860s | -4.449s | -60.9% | 7.321s | 2.858s | -4.463s | -61.0% |
| `import_reasoning_parser` | 4.081s | 0.027s | -4.054s | -99.3% | 4.089s | 0.027s | -4.063s | -99.3% |
| `import_moe_utils` | 6.773s | 3.152s | -3.621s | -53.5% | 6.776s | 3.148s | -3.628s | -53.5% |
| `import_engine` | 8.208s | 7.359s | -0.850s | -10.4% | 8.203s | 7.360s | -0.843s | -10.3% |
| `import_scheduler` | 8.035s | 5.422s | -2.613s | -32.5% | 8.031s | 5.432s | -2.599s | -32.4% |

## Sys Time

| Benchmark | Before mean | After mean | Delta | Mean pct | Before median | After median | Delta | Median pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `import_sglang` | 0.091s | 0.007s | -0.084s | -92.4% | 0.089s | 0.007s | -0.082s | -92.2% |
| `import_model_config` | 1.099s | 0.370s | -0.729s | -66.3% | 1.101s | 0.371s | -0.731s | -66.3% |
| `import_reasoning_parser` | 0.534s | 0.009s | -0.525s | -98.3% | 0.534s | 0.009s | -0.525s | -98.3% |
| `import_moe_utils` | 1.017s | 0.472s | -0.545s | -53.5% | 1.008s | 0.472s | -0.537s | -53.2% |
| `import_engine` | 1.167s | 1.145s | -0.022s | -1.9% | 1.166s | 1.147s | -0.019s | -1.7% |
| `import_scheduler` | 1.102s | 0.743s | -0.359s | -32.6% | 1.106s | 0.749s | -0.357s | -32.3% |

## Notes

- The largest improvements are on import paths that previously dragged in frontend APIs or parser/config dependencies:
  - `import_sglang`
  - `import_reasoning_parser`
  - `import_model_config`
- `import_scheduler` improved materially, but not as much as the root package and parser paths, because it still pays for substantial runtime-side imports that are genuinely needed.
- `import_engine` improved only modestly, which is expected because it still sits close to the runtime stack and model-serving entrypoints.
