# Phase 2 Pressure Benchmark

This benchmark builds a fixed 24-object synthetic catalog and runs the
sequential Architect/Coder/Debugger retry trace under calibrated GPU KV
pressure.

The primary run uses:

- client-observed first non-empty-token TTFT;
- `max_tokens=1`;
- one warmup trace with a distinct cache salt;
- cache flush and a clean metrics baseline before measurement;
- a final cache flush and reset-state comparison;
- boundary-only Prometheus scraping to avoid perturbing the next request;
- a fixed three-object probe cohort across all pressure points;
- frozen object IDs and request order across independent server restarts.

Run from the immutable SM75 image with the source and Hugging Face cache
mounted read-only and a separate writable results mount:

```bash
python3 -m benchmark.approx_kv.run_phase2_matrix \
  --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --source-git-sha <phase2-git-sha> \
  --image-digest sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781 \
  --output-dir /results
```

The default matrix is `rho=0.8/1.0/1.5/2.0/3.0` with three independent
server restarts per point. The runner never removes an existing result
directory and refuses to write into a non-empty directory.

`rho_reusable` is based on the unique token trie for stable reusable prefixes.
`rho_physical` additionally estimates all measured prompt branches and one
generated token per request. Synthetic dense/recovery cost fields are metadata
for later trace validation and are not measured recovery costs.

## Phase 3 HiCache canary

Start SGLang with hierarchical cache plus:

```text
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
SGLANG_APPROX_KV_CORE=1
SGLANG_APPROX_KV_HOST=1
SGLANG_APPROX_KV_PREFETCH=1
```

Then run:

```bash
python3 -m benchmark.approx_kv.run_phase3_canary \
  --base-url http://127.0.0.1:30000 \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --model-fingerprint Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca \
  --output /results/phase3-canary.json
```

The canary uses the Phase 2 object generator and verifies host export, async
H2D, full-layer copy, mismatch fallback, reset-induced store miss, exact-Radix
isolation, and final KV-pool accounting.

## Phase 4 R0 (Raw+RoPE) unified high-pressure benchmark

`run_r0_pressure.py` drives a real, running SGLang server (S0 LRU scheduler,
GPU-only residency tier, prefetch disabled) and compares client-observed
streaming TTFT between a fresh `dense` prefill and the `raw` (raw copy +
signed RoPE relocation) recovery path, under real Radix eviction pressure
calibrated against the server's *actual* usable KV capacity read back from
live `/metrics` -- never an assumed constant.

It implements the unified Phase 4 benchmark contract:

- exact-prefix header lengths `0, 32, 64, 128, 256` tokens;
- lossy body lengths `512, 768, 1024, 2048` tokens;
- a canonical source body longer than 512 tokens is registered as multiple
  `<=512`-token segments (one `register` request per segment, via
  `--segment-tokens`); the target request recovers them back into one
  contiguous span;
- `--mem-fraction-static 0.35` (must match the server's actual launch flag);
  pressure is calibrated after startup so the *actual* pre-target reusable
  rho lands at approximately `0.9 / 1.1 / 1.5 / 2.0 / 3.0` (pass the matching
  `--target-rho`);
- exactly one discarded warmup pass per setting, then `--repeats` formal
  repeats (default 4, rejected below 2);
- every invocation appends a `running`/`completed`/`failed` record with the
  full settings, raw result path, and a compact summary to
  `/home/chris/Workspaces/kvcache-research/results/BENCHMARK_RUN_LOG.jsonl`
  (override with `--central-log`);
- refuses to overwrite an existing `--output` result file.

Start the server (S0 LRU / GPU-only / prefetch-off) with:

```text
SGLANG_APPROX_KV_CORE=1
SGLANG_APPROX_KV_RAW_ROPE=1
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --mem-fraction-static 0.35 \
  ...
```

Then run one setting at a time, e.g. `dense` vs `raw` at `header=64`,
`body=1024`, target rho `2.0`:

```bash
python3 -m benchmark.approx_kv.run_r0_pressure \
  --base-url http://127.0.0.1:30000 \
  --mode raw \
  --header-tokens 64 \
  --body-tokens 1024 \
  --target-rho 2.0 \
  --runner-git-sha <phase4-r0-git-sha> \
  --image-digest sha256:<image-digest> \
  --output benchmark/approx_kv/results/phase4-r0/raw-header64-body1024-rho2.0.json

python3 -m benchmark.approx_kv.run_r0_pressure \
  --base-url http://127.0.0.1:30000 \
  --mode dense \
  --header-tokens 64 \
  --body-tokens 1024 \
  --target-rho 2.0 \
  --runner-git-sha <phase4-r0-git-sha> \
  --image-digest sha256:<image-digest> \
  --output benchmark/approx_kv/results/phase4-r0/dense-header64-body1024-rho2.0.json
```

Each result JSON reports, per formal round: the actual `capacity_tokens`
read from live metrics, `pre_target_rho`/`peak_rho_with_target` computed
against that real capacity, `evicted_tokens_pressure`/`evicted_tokens_target`
counters, and the client-observed `target.ttft_ms`/`cached_tokens`. `raw`
mode additionally asserts zero dense fallback and full body recovery
(`cached_tokens == header_tokens + body_tokens`) before accepting a round.
