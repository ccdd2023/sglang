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

## Phase 5 workflow cache policies

Phase 5 adds cache-protection metadata that is independent from request
scheduling `priority`.

Eviction policies:

- `lru`: S0 control;
- `workflow_steps`: S1 coarse steps-to-execution;
- `belady`: S2 exact next-use distance;
- `recovery_value`: S3 saved recovery time per resident byte and reuse distance;
- `hierarchical`: S4 dead objects, recoverable exact variants, repair metadata,
  anchors, exact variants, then canonical bases.

Workflow metadata is carried in `sampling_params.custom_params`:

```json
{
  "cache_protection": {
    "object_id": "coder-bundle",
    "protected_tokens": 1024,
    "resident_bytes": 117440512,
    "dense_cost_ms": 280.0,
    "recovery_cost_ms": 30.0,
    "current_step": 6,
    "next_use_step": 9,
    "next_use_request_step": 14,
    "next_use_distance": 8,
    "workflow_stage": "coder",
    "object_kind": "stage_variant",
    "recoverable_from_lower_tier": true,
    "retired": false
  },
  "cache_prefetch": {
    "object_id": "debugger-bundle",
    "next_use_step": 14
  }
}
```

`protected_tokens` anchors metadata to the reusable prefix boundary, so
request-specific dynamic suffix branches do not inherit object identity.
`next_use_step` is an absolute coarse workflow ordinal; `next_use_request_step`
is the absolute request ordinal used by Belady and P3. Absolute ordinals avoid
stale relative-distance comparisons as the trace advances.

Prefetch modes:

- `p0`: off;
- `p1`: admit only when GPU space is already free;
- `p2`: additionally evict only objects explicitly known to have no future use;
- `p3`: additionally evict recoverable objects whose oracle next use is later
  than the prefetched object.

The matrix runner uses a fixed
`Architect -> Coder -> Debugger -> Coder -> Debugger` sequence, mixes dead and
live pressure objects, performs a discarded warmup, resets between formal
repeats, and writes every setting to the central JSONL log.

GPU-only S0-S4 screening:

```bash
python3 -m benchmark.approx_kv.run_phase5_scheduler_matrix \
  --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --source-git-sha <phase5-git-sha> \
  --image-digest sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781 \
  --policies lru,workflow_steps,belady,recovery_value,hierarchical \
  --prefetch-modes p0 \
  --pressure-points 1.1,1.5,2.0,3.0 \
  --formal-repeats 2 \
  --output-dir /results/phase5-scheduler \
  --central-log /results/BENCHMARK_RUN_LOG.jsonl
```

HiCache P0-P3 screening for selected policies adds `--enable-hicache` and
`--prefetch-modes p0,p1,p2,p3`.

### SM75 results

The formal Qwen3-0.6B GPU-only matrix used 1024-token objects,
`mem_fraction_static=0.35`, one discarded warmup, and two independent formal
rounds per setting.

S4 hierarchical was the only policy with stable mean-TTFT gains over S0 LRU
at every high-pressure point:

| rho | S0 mean | S4 mean | mean speedup | S0/S4 hit fraction |
| ---: | ---: | ---: | ---: | ---: |
| 1.5 | 215.93 ms | 163.46 ms | `1.32x` | 0.510 / 0.903 |
| 2.0 | 216.56 ms | 188.96 ms | `1.15x` | 0.510 / 0.705 |
| 3.0 | 214.37 ms | 189.31 ms | `1.13x` | 0.511 / 0.705 |

Two additional server restarts reproduced S4 speedups of `1.32-1.34x` at
rho 1.5 and `1.11-1.15x` at rho 2.0.

The S4 + HiCache prefetch matrix found:

- P1 issued no proactive loads under pressure, as required by free-space-only
  admission;
- P2 loaded 2,016 tokens and evicted 2,088 admission tokens per setting;
- P3 loaded 5,040 and evicted 5,112 tokens at rho 3;
- P2/P3 did not provide a stable mean-TTFT gain and increased p95.

The sequential default is therefore **S4 + P0**. P1-P3 remain implemented
experimental variants. The current proactive H2D path waits synchronously for
completion, supports `HiRadixCache`, and rejects victim-evicting prefetch under
HiCache write-back mode.

Compact artifacts:

- `results/phase5-scheduler/sm75-scheduler-matrix.json`;
- `results/phase5-scheduler/sm75-prefetch-matrix.json`;
- `results/phase5-scheduler/sm75-restart-validation.json`.

## Phase 6 cross-store substrate

The Phase 6 path is disabled by default. Standard `RadixCache` enables it with:

```text
SGLANG_APPROX_KV_CORE=1
SGLANG_APPROX_KV_CROSS_STORE=1
SGLANG_APPROX_KV_REGISTER_EVICTS_EXACT=1
SGLANG_APPROX_KV_BYTES_PER_TOKEN=<model KV bytes per token>
SGLANG_APPROX_KV_HOST_BUDGET_BYTES=<host byte budget>
```

It adds byte-authoritative exact/approximate competition, dependency-closed
victim selection, reversible host demotion, allocator reservation telemetry,
and reset-state store gauges. `SGLANG_APPROX_KV_CROSS_STORE` with
`HiRadixCache` is rejected at startup because exact HiCache victims require
different host-reference semantics.

Freeze the commit-bound fixed40 contract first:

```bash
python3 -m benchmark.approx_kv.run_p6_0_contract \
  --source-git-sha <phase6-core-sha> \
  --image-digest <image-digest> \
  --chunked-prefill-size 1024 \
  --chunk-source provisional_worst_case \
  --output /results/phase6/p6-0-contract.json \
  --central-log /results/BENCHMARK_RUN_LOG.jsonl
```

Closeout qualification and chunk selection:

```bash
python3 -m benchmark.approx_kv.run_cl1_qualification \
  --source-git-sha <phase6-core-sha> \
  --model-revision <model-revision> \
  --image-digest <image-digest> \
  --output /results/phase6/cl1.json \
  --log-dir /results/phase6/logs \
  --central-log /results/BENCHMARK_RUN_LOG.jsonl

python3 -m benchmark.approx_kv.run_cl2_chunk_gate \
  --selected-candidate <r0|r1_k0|r1_k4|r1_k8|r1_k16|r1_k32|NONE> \
  --source-git-sha <phase6-core-sha> \
  --model-revision <model-revision> \
  --image-digest <image-digest> \
  --output /results/phase6/cl2.json \
  --log-dir /results/phase6/logs \
  --central-log /results/BENCHMARK_RUN_LOG.jsonl
```

Phase 6 validity runners:

```bash
python3 -m benchmark.approx_kv.run_p6_h_host_roundtrip \
  --source-git-sha <phase6-core-sha> \
  --model-revision <model-revision> \
  --image-digest <image-digest> \
  --output /results/phase6/p6-h.json \
  --log /results/phase6/logs/p6-h-server.log \
  --central-log /results/BENCHMARK_RUN_LOG.jsonl

python3 -m benchmark.approx_kv.run_p6_4_capacity_pilot \
  --source-git-sha <phase6-core-sha> \
  --model-revision <model-revision> \
  --image-digest <image-digest> \
  --output /results/phase6/p6-4.json \
  --log-dir /results/phase6/logs \
  --central-log /results/BENCHMARK_RUN_LOG.jsonl
```

P6-H uses synchronous `AllocatorCPUResidencyBackend` copies and explicitly
records `hicache_tier_exercised=false`; it does not qualify the later HiCache
H4/RH4 track. P6-4 disables performance ranking and reports `valid`,
`diagnostic-unavailable`, or `invalid` separately from dense fallback
reachability.

All Phase 6 and closeout runners reject a dirty source tree, record the Git
tree SHA, pass the frozen model revision to the server, and leave
`result_git_sha` pending until the result artifact is committed.
