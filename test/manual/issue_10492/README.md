# Issue 10492 Manual Reproduction Artifacts

This directory keeps the benchmark harness and compact raw timing data used for the local import-time comparison.

## What Is Preserved

- `bench_imports.py`: the fresh-process import benchmark harness
- `perf-results/main-10runs.json`: raw data for the `main` baseline
- `perf-results/fix-10runs.json`: raw data for the fix branch

These JSON files keep the measured `real`, `user`, and `sys` values for each run. They are small enough to version without dragging the full temporary workspace into git history.

## What Stays Local

The full `importtime` stderr samples and the local `main` snapshot were left under `.codex-tmp/` and are intentionally not versioned here because they are much noisier and larger:

- `.codex-tmp/main-baseline-20260308-173917/`
- `.codex-tmp/perf-results/*-samples/`

Keep those local artifacts if you need deeper forensic analysis.

## Rerun Inside The Dev Container

Use the shared venv from the container session:

```bash
cd /sgl-workspace/sglang
ISSUE10492_BENCH_PYTHON=/tmp/sglang-issue10492-venv-min2/bin/python \
  python3 test/manual/issue_10492/bench_imports.py \
  --repo-root /sgl-workspace/sglang \
  --output /sgl-workspace/sglang/test/manual/issue_10492/perf-results/fix-rerun.json \
  --label fix-rerun \
  --warmups 1 \
  --runs 10
```

To compare with `main`, run the same command against a clean `main` checkout or the saved baseline snapshot.

## Notes

- All measurements in this issue were collected in the dev container, not on the host.
- The local GPU is `RTX 2080 SUPER / SM75 / 8GB`, so these artifacts validate import-path cleanup, not Hopper-only DeepGEMM behavior.
