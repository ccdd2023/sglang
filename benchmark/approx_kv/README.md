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

## Phase 4 unified high-pressure contract (Cache-Craft / R3)

Phase 4 unifies the settings/results-metadata contract across all six
research paths (R0 Raw+RoPE, R1 EPIC/LegoLink, R2 CacheBlend, R3 Cache-Craft,
R4 KVCOMM, R5 CacheTune). Every Phase 4 runner (including
`run_phase4_cachecraft_pressure.py`) must use exactly these fixed values:

- exact header sweep: `0, 32, 64, 128, 256` tokens (the target request's
  leading exact-match context length; *not* attention head count);
- lossy body sweep: `512, 768, 1024, 2048` tokens; any canonical source body
  over 512 tokens is registered in `<=512`-token segments (see
  `cachecraft_workloads.segment_into_canonical_chunks`), never as one
  oversized registration;
- `mem_fraction_static=0.35`, with usable KV capacity re-derived at runtime
  from live allocator/cache metrics rather than assumed from a historical
  estimate;
- actual reusable rho (working-set / usable-capacity) targets approximately
  `0.9 / 1.1 / 1.5 / 2.0 / 3.0`, realized through filler-object count, not a
  single oversized prompt;
- fixed scheduler configuration: S0 LRU, GPU-only tier, prefetch disabled;
- one discarded warmup pass per setting, then `>=2` (default `4`) formal
  repeats, using the client-observed streaming TTFT (first non-empty `data:`
  frame), exactly as in the Phase 2/3 runners above;
- every run (`running` / `completed` / `failed` / `blocked`) appends one JSON
  line to the shared central log
  (`/home/chris/Workspaces/kvcache-research/results/BENCHMARK_RUN_LOG.jsonl`
  outside this repository), carrying the full settings dict, a timestamp, and
  the result/output path or failure/blocked reason. Existing per-path result
  files under `benchmark/approx_kv/results/` (e.g. `phase2/`, `phase3/`) are
  never overwritten by Phase 4 runners; Phase 4 results are written under
  their own `phase4-r*/` subdirectory per path.

### Cache-Craft's honest current status: capability-gated, not server E2E

Unlike R1 EPIC (which has a real production in-request seam), Cache-Craft's
CCI/beta/gamma/CFO decision logic, causal-attention profile capture and
partial-repair execution (`cachecraft_metrics.py`, `cachecraft_attention.py`,
`cachecraft_plugin.py`, `cachecraft_recompute.py`, `cachecraft_runtime.py`)
are CPU-tested library code only. `cachecraft_capability.py`'s
`inspect_scheduler_dispatch_capability()` records the precise, currently-true
reason why: `schedule_batch.py` never dispatches a `plugin: "cachecraft"`
request to `cachecraft_runtime.restore_request_via_cachecraft` -- it always
calls the *generic* `runtime.restore_request_prefix` raw-copy path instead,
regardless of the request's `plugin` field. There is also no real
attention-profile capture wired to a live model's forward pass, and no
production selected-token recompute hook (the same class of blocker
documented for R1's `ForwardMode.TARGET_VERIFY`).

`run_phase4_cachecraft_pressure.py` checks this capability *first*, before
any HTTP request or GPU work:

```bash
python3 -m benchmark.approx_kv.run_phase4_cachecraft_pressure \
  --target-rho 2.0 \
  --header-tokens 64 \
  --body-tokens 1024 \
  --runner-git-sha <git-sha> \
  --image-digest sha256:<image-digest> \
  --output /results/phase4-r3/sm75-pressure.json \
  --central-log /home/chris/Workspaces/kvcache-research/results/BENCHMARK_RUN_LOG.jsonl
```

Today this always exits `3` (`BLOCKED_EXIT_CODE`), writes **one** `status:
"blocked"` entry to the central log with the exact capability reason, and
writes **no** result file and makes **no** network/GPU call at all -- it must
never be run with a fake backend standing in for a real recompute hook to
manufacture a "successful" Cache-Craft server result. The real-run code path
(`run_real`/`run_round`) is written to the exact same
settings/warmup/repeats/log shape as R1's completed runner so it needs no
redesign once real scheduler dispatch and a real recompute hook exist; until
then it is only unit-tested against a fake HTTP transport
(`test/registered/unit/bench/test_run_phase4_cachecraft_pressure.py`), never
against a real server.

`cachecraft_workloads.build_non_prefix_segmented_workload` prepares a
deterministic, GPU-free "non-prefix" request shape (several canonical chunks,
each `<=512`-token segmented, reused by the target prompt in a different
order than they were registered in) so that once a real hook exists, the
runner can immediately exercise Cache-Craft's Order-Penalty-driven decision
logic instead of plain contiguous-prefix reuse -- it produces only Python
token-id lists and is fully covered by
`test/registered/unit/bench/test_cachecraft_workloads.py`.
