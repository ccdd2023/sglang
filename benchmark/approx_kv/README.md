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

## Phase 4 R5 CacheTune canary

CacheTune hardware-controller inspired subset: a roofline-based
recompute/transfer ratio controller (`sglang.srt.mem_cache.cachetune`)
driving the ported CacheBlend-style selected-token repair mechanism. This
is a controller-plus-existing-repair-backend subset, not a faithful full
CacheTune implementation -- frequency-domain token selection, sparse
transfer, multi-stream overlap and deferred RoPE from the paper are out of
scope for this branch.

Start SGLang with:

```text
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
SGLANG_APPROX_KV_CORE=1
SGLANG_APPROX_KV_HOST=1
SGLANG_APPROX_KV_CACHETUNE=1
SGLANG_CACHETUNE_MODE=speed_only        # or paper_mechanism
SGLANG_CACHETUNE_T_C_MS=19.0            # deployment-wide measurement --
SGLANG_CACHETUNE_T_I_MS=1.0             # all three are required together,
SGLANG_CACHETUNE_T_O_MS=0.5             # or every request dense-falls-back
```

Then run:

```bash
python3 -m benchmark.approx_kv.run_phase4_cachetune_canary \
  --base-url http://127.0.0.1:30000 \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --model-fingerprint Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca \
  --mode speed_only \
  --t-c-ms 19.0 --t-i-ms 1.0 --t-o-ms 0.5 \
  --runner-git-sha <cachetune-git-sha> \
  --image-digest sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781 \
  --output /results/phase4-r5/sm75-server.json
```

`--mode`/`--t-c-ms`/`--t-i-ms`/`--t-o-ms`/`--first-recompute-layer` must
match the server's own `SGLANG_CACHETUNE_*` environment exactly: the
runner uses this package's real `roofline_ratio`/`quantize_ratio`/
`predict_ttft_ms` functions (imported directly, never reimplemented) to
compute an independent expected repair-token count and cross-validates it
against the real `sglang:approx_kv_cachetune_*` Prometheus counter deltas
observed around genuine HTTP requests.

This fork has no ModelRunner hook for a genuine inline per-layer forward
over an arbitrary token subset, so every real repair in this canary uses
the precomputed fresh-KV adapter (`cachetune-raw:`/`cachetune-fresh:`
segment registration), exactly like `research/cacheblend`'s own real
canary. The controller's per-request decision (ratio, selected tokens,
recomputed layers, precomputed-adapter usage) is not exposed in the
`/v1/chat/completions` response body; it is only observable in aggregate
through `/metrics`, which is what this runner validates.

The runner also sweeps a few additional real restore lengths against the
*same* running server/controller (no restart) to prove genuine
per-request deterministic ratio re-quantization, and reports
`dense_p50_ms`, `cachetune_target_p50_ms`, `fresh_preparation_p50_ms` and
`combined_p50_ms` (target plus fresh preparation) -- never excluding
preparation cost before claiming an end-to-end result.
