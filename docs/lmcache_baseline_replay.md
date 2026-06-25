# LMCache Baseline Replay Runbook

This runbook prepares a same-workload LMCache baseline for the AgentTemplateKV
paper without mutating the primary `sglang-kvflow` conda environment.

## Why Isolation Is Required

The primary environment currently has:

```text
torch==2.9.1
transformers==5.3.0
sglang==0.0.0.dev10831+g97638c37f.d20260519
protobuf==7.35.0
grpcio==1.80.0
setuptools==82.0.1
numpy==2.2.6
lmcache: missing
```

It also contains an editable matplotlib path from a SWE-bench workspace:

```text
/home/gfy/.conda/envs/sglang-kvflow/lib/python3.12/site-packages/__editable__.matplotlib-3.5.0.dev1136+gb7ce415c1.d20260611.pth
/home/gfy/CodeMAS_Project/sglang-kvflow/results/swebench_local_envs/repos/matplotlib__matplotlib-20488/lib
```

`pip install --dry-run lmcache==0.4.7` would install or upgrade a large package
set, including `transformers-5.12.0`, grpc/protobuf/opentelemetry packages,
`cupy`, `nixl`, and `setuptools`. Do not install LMCache directly into the
primary environment unless the environment has first been snapshotted.

## Preflight Already Completed

The benchmark runner supports an explicit LMCache profile:

```bash
--baseline-profile lmcache
```

This profile resolves to `--enable-lmcache`, sets `LMCACHE_USE_EXPERIMENTAL=True`,
sets `LMCACHE_CONFIG_FILE`, and suppresses HiCache flags. This matters because
SGLang's scheduler chooses HiCache before LMCache; a command containing both
`--enable-hierarchical-cache` and `--enable-lmcache` would not exercise LMCache.

Dry-run artifact:

```text
results/coding_kvflow_prefetch/qwen2_5_7b_100_lmcache_dryrun_20260613/server_command.json
```

Required invariant:

```bash
/home/gfy/.conda/envs/sglang-kvflow/bin/python \
  benchmark/multi_workflow/check_lmcache_preflight.py \
  results/coding_kvflow_prefetch/qwen2_5_7b_100_lmcache_dryrun_20260613/server_command.json
```

## Option A: Clone The Primary Conda Environment

Use this when disk space is sufficient and a slow conda clone is acceptable.

```bash
/home/gfy/.conda/bin/conda create -y -n sglang-kvflow-lmcache --clone sglang-kvflow
/home/gfy/.conda/bin/conda run -n sglang-kvflow-lmcache python -m pip install lmcache==0.4.7
```

Verify imports:

```bash
/home/gfy/.conda/bin/conda run -n sglang-kvflow-lmcache python - <<'PY'
import importlib.metadata as md
for pkg in ["sglang", "torch", "transformers", "lmcache", "protobuf", "grpcio"]:
    print(f"{pkg}=={md.version(pkg)}")
PY
```

## Option B: Export Then Create A Replay Environment

Use this if clone is too slow or fails.

```bash
/home/gfy/.conda/bin/conda env export -n sglang-kvflow > /home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/sglang_kvflow_env_before_lmcache_20260613.yml
/home/gfy/.conda/bin/conda create -y -n sglang-kvflow-lmcache python=3.12
/home/gfy/.conda/bin/conda run -n sglang-kvflow-lmcache python -m pip install -e /home/gfy/CodeMAS_Project/sglang-kvflow
/home/gfy/.conda/bin/conda run -n sglang-kvflow-lmcache python -m pip install lmcache==0.4.7
```

This option may require additional SGLang build/runtime dependencies from the
primary environment before the model server starts. Prefer Option A when possible.

## 1-Case Smoke

Run this before any 100-case replay:

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow
PY=/home/gfy/.conda/envs/sglang-kvflow-lmcache/bin/python
PYTHONPATH=/home/gfy/CodeMAS_Project/sglang-kvflow:/home/gfy/CodeMAS_Project/sglang-kvflow/python \
"$PY" -m benchmark.multi_workflow.bench_coding_kvflow_prefetch \
  --model /home/gfy/models/Qwen2.5-7B-Instruct \
  --dataset results/repo_level_datasets/swe_verified_100_instances.json \
  --manifest results/repo_level_datasets/manifest_100.json \
  --max-cases 1 \
  --max-tokens 8 \
  --baseline-profile lmcache \
  --port 31341 \
  --out-dir results/coding_kvflow_prefetch/qwen2_5_7b_1_lmcache_smoke_20260613
```

Smoke acceptance:

- `summary.json` exists.
- `prefetch_table.csv` exists.
- `server_command.json` contains `--enable-lmcache`.
- `server_command.json` does not contain `--enable-hierarchical-cache`.
- `sglang_server.log` contains no OOM.
- If the server fails before ready, preserve `sglang_server.log` and classify the failure as `lmcache_env_or_server_start_failure`, not as a paper result.

## 100-Case Replay

Only run this after the 1-case smoke passes.

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow
PY=/home/gfy/.conda/envs/sglang-kvflow-lmcache/bin/python
nohup env PYTHONPATH=/home/gfy/CodeMAS_Project/sglang-kvflow:/home/gfy/CodeMAS_Project/sglang-kvflow/python \
"$PY" -m benchmark.multi_workflow.bench_coding_kvflow_prefetch \
  --model /home/gfy/models/Qwen2.5-7B-Instruct \
  --dataset results/repo_level_datasets/swe_verified_100_instances.json \
  --manifest results/repo_level_datasets/manifest_100.json \
  --max-cases 100 \
  --baseline-profile lmcache \
  --port 31342 \
  --out-dir results/coding_kvflow_prefetch/qwen2_5_7b_100_lmcache_20260613 \
  > /tmp/qwen2_5_7b_100_lmcache_20260613.stdout 2>&1 &
```

100-case acceptance:

- `summary.json` exists.
- `prefetch_table.csv` has 400 rows: 100 cases x 4 modes.
- All four modes are represented for each case.
- `server_command.json` proves LMCache was requested and HiCache was suppressed.
- `PREFETCH_REPORT.md` records the command and backend profile.
- The completed replay may be cited only as a diagnostic feasibility artifact unless it is rerun under one clean, documented configuration and then compared against the existing 100-case stock SGLang / workflow-prefix / AgentTemplateKV rows.

## Paper Use

## 2026-06-13 Replay Notes

Completed setup:

- Cloned `/home/gfy/.conda/envs/sglang-kvflow` to `/home/gfy/.conda/envs/sglang-kvflow-lmcache`.
- Installed `lmcache==0.4.7` only in the cloned environment.
- Added a small SGLang adapter compatibility fix: the local LMCache connector now receives `LMCACHE_CONFIG_FILE`, which `lmcache==0.4.7` requires.
- Changed the prefetch runner's default `--python` to `sys.executable`, so an isolated replay environment launches the server with the same Python that runs the benchmark driver.

Observed package changes in the isolated environment:

```text
torch==2.9.1
torchao==0.17.0
transformers==5.12.0
sglang==0.0.0.dev10831+g97638c37f.d20260519
protobuf==6.33.6
grpcio==1.81.1
setuptools==80.10.2
numpy==2.2.6
lmcache==0.4.7
```

Important runtime caveats:

- LMCache imports with a CUDA13 backend warning: `libcudart.so.13` is unavailable. The replay used LMCache's LocalCPUBackend.
- The default example config has `max_local_cpu_size: 10`, which produces repeated LMCache allocation-capacity warnings on long codebase prompts.
- A single 100-case server run completed only the first 48 cases before SGLang's strict idle memory checker killed the server.
- The remaining cases required `--flush-cache-per-case`, `--disable-overlap-schedule`, `--max-running-requests 1`, `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0`, and `PYTHONHASHSEED=0`.
- Therefore the combined 100-case artifact is a feasibility/diagnostic external-baseline replay, not a clean latency headline.

Completed artifacts:

```text
results/coding_kvflow_prefetch/qwen2_5_7b_1_lmcache_smoke_configfix_20260613
results/coding_kvflow_prefetch/qwen2_5_7b_100_lmcache_20260613
results/coding_kvflow_prefetch/qwen2_5_7b_100_lmcache_20260613_shard48_20_flush_noidlecheck
results/coding_kvflow_prefetch/qwen2_5_7b_100_lmcache_20260613_shard68_20_flush_noidlecheck
results/coding_kvflow_prefetch/qwen2_5_7b_100_lmcache_20260613_shard88_12_flush_noidlecheck
results/coding_kvflow_prefetch/qwen2_5_7b_100_lmcache_combined_20260613
```

Combined coverage:

- 100 cases.
- 400 rows.
- Four modes represented exactly 100 times each.

Paper use:

- Do not use this as a clean speedup/latency headline.
- It can support a limitation or appendix note: same-workload LMCache replay is operational in this fork, but needs an isolated environment and cache-flush/no-idle-check hygiene on the 24GB server.
- A publication-grade LMCache baseline should rerun from a single documented configuration, preferably with a larger LMCache CPU/disk cache and no unrelated GPU process.
