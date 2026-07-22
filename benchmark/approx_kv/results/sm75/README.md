# SM75 Sequential Approximate-KV Results

## Scope

This is an acceleration-only MVP. It does not evaluate output quality,
semantic correctness, or code correctness.

The workload is sequential and uses the fixed trace:

```text
Architect
-> Coder
-> Debugger
-> cold filler A
-> cold filler B
-> Coder
-> Debugger
```

The raw-speed path registers one dense whole-prefix KV object, then reuses it
for different role prompts with token equality intentionally disabled. The
last prompt token still runs through a real model forward pass.

## Environment

- Git source used for the final runtime fix: `7519608c67752411eb8164c6d69a3c0a3a252a7b`
- GPU: NVIDIA GeForce RTX 2080 SUPER, SM75, 8 GB
- Base image: `lmsysorg/sglang:dev`
- Base image digest: `sha256:9bfa6494978dd1781788a73e5b096635c183b1e9e46d1d1c2bb10a1ed1630716`
- CUDA: 12.9.1
- PyTorch: 2.9.1+cu129
- Model: `Qwen/Qwen3-0.6B`
- Attention backend: `torch_native`
- GPU KV capacity: 9,954 tokens
- Server cache: radix enabled for exact runs, disabled for dense runs
- Approximate store: independent device slots, exact Radix insertion disabled
- Guest isolation: read-only container rootfs; source, dependencies, and
  results lived only in container tmpfs

The latest source requires packages newer than the base image. The guest used
temporary compatibility shims:

- Transformers runtime 5.12.1
- FlashInfer Python/cubin runtime 0.6.13 with 0.6.14 metadata
- `sgl_kernel` runtime 0.3.21 with 0.4.5 metadata

These shims are functional for the MVP but are not a release environment.
FlashInfer long-prompt kernels did not work on SM75, so the experiment used
the slower `torch_native` attention backend.

## End-to-End TTFT

All values are client-observed milliseconds. Each row contains 14 measured
requests after one warmup repeat.

| Median prompt tokens | Actual rho | Dense p50 | Exact p50 | Raw p50 | Raw vs exact | Exact-relative change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,673 | 0.840 | 345.71 | 298.23 | 301.50 | 0.989x | +1.10% |
| 3,052 | 1.533 | 946.86 | 943.66 | 862.83 | 1.094x | -8.57% |
| 3,759 | 1.888 | 1,403.98 | 1,404.20 | 1,297.06 | 1.083x | -7.63% |

The result matches the pressure hypothesis:

- Below capacity, exact Radix is slightly faster than raw approximate reuse.
- Once the five-role working set exceeds GPU KV capacity, raw whole-prefix
  reuse reduces TTFT relative to the best exact baseline.
- The benefit does not grow monotonically with prompt length because the last
  real token still performs attention over the entire restored prefix, and
  `torch_native` attention is expensive.

## GPU Recovery Microbenchmark

The microbenchmark uses 28 layers, 3,048 tokens, 8 KV heads, head dimension
128, FP16, and a position delta of 256.

| Path | p50 operation time | Upper-bound speedup vs 1,295.16 ms dense TTFT |
| --- | ---: | ---: |
| Raw copy + RoPE | 12.69 ms | 102.10x |
| EPIC body copy, repair excluded | 12.55 ms | 103.22x |
| Selective copy, repair excluded | 35.08 ms | 36.92x |
| One-anchor base + delta reconstruction | 20.66 ms | 62.70x |

These are operation-level upper bounds. EPIC and selective-repair values do
not include the real token repair computation.

## Scheduler Simulation

Using the measured 12.69 ms raw recovery cost, five role objects, and capacity
for three objects:

| Policy | Hit rate | Recoveries | Evictions | Mean simulated TTFT |
| --- | ---: | ---: | ---: | ---: |
| LRU | 25.71% | 52 | 49 | 10.71 ms |
| Steps-only | 54.29% | 32 | 29 | 8.51 ms |
| Belady oracle | 54.29% | 32 | 29 | 8.51 ms |
| Value density | 54.29% | 32 | 29 | 8.51 ms |
| Hierarchical | 54.29% | 32 | 29 | 8.51 ms |

The simple fixed trace does not distinguish the four workflow-aware policies.
More varied object sizes, recovery costs, and branch distances are needed
before selecting among them.

## Invalid Runs

The following guest files were produced by accidentally submitting two
benchmarks concurrently to one server and are excluded:

- `/work/dense-blocks40.json`
- `/work/dense-blocks100.json`

They were intentionally not committed.

## Next Validation

The next required step is to repeat the selected paths on an isolated RTX PRO
6000 guest with a normal immutable CUDA 12.9+/13 image and matching package
versions. The SM75 compatibility shims and `torch_native` backend must not be
used to make SM120 performance claims.
