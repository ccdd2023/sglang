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
  --body-tokens 256 --length-sweep 128,512 \
  --repeats 4 \
  --runner-git-sha <cachetune-git-sha> \
  --image-digest sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781 \
  --output /results/phase4-r5/sm75-server.json \
  --central-log /results/phase4-r5/central.jsonl
```

`--mode`/`--t-c-ms`/`--t-i-ms`/`--t-o-ms`/`--first-recompute-layer` must
match the server's own `SGLANG_CACHETUNE_*` environment exactly: the
runner uses this package's real `roofline_ratio`/`quantize_ratio`/
`predict_ttft_ms` functions (imported directly, never reimplemented) to
compute an independent expected repair-token count and cross-validates it
against the real `sglang:approx_kv_cachetune_*` Prometheus counter deltas
observed around genuine HTTP requests.

### Non-prefix segment workload (real cross-context repair, not exact replay)

The runner talks to the server's native `POST /generate` endpoint with
direct `input_ids` (never `/v1/chat/completions`), and every workload it
builds is a genuine **non-prefix** segment, not a shared-from-position-0
prefix:

* `source_prompt = source_head_ids + shared_body_ids`
* `target_prompt = target_head_ids + shared_body_ids + tail_ids`

with `source_head_ids != target_head_ids` (different token content) but
`shared_body_ids` byte-identical between the two. Each piece is tokenized
*separately* and the resulting integer-id lists are concatenated directly
(never re-tokenized as a joined string), so segment offsets are exact by
construction. This matters because causal attention only depends on
*preceding* tokens: if source and target shared an identical prefix from
position 0 (as an earlier version of this script did, via
`common_prefix_token_ids(source, target)` with `target_start=0`), the
registered "raw" and "fresh" segments' KV would be bit-identical --
proving only exact-content transplantation, never CacheTune's real
approximate-repair mechanism. With distinct heads, the "raw" (registered
from `source_prompt`) and "fresh" (registered from `target_prompt`)
segments capture genuinely different preceding-context KV for the same
body content, so a reuse request that only "gets away with" the raw
segment is a real repair, not a coincidence.

`source_head_ids`/`target_head_ids` are more than merely *unequal*: they
are constructed to share **zero** common exact-match token prefix, which
`NonPrefixSegmentWorkload` verifies explicitly and raises on if violated.
This is not automatic from picking two different seeds alone --
`workloads.deterministic_code`'s generated text always begins with the
same literal, seed-*independent* boilerplate before any seed-dependent
digest character appears, so two differently-seeded pieces would
otherwise reliably tokenize to several *identical* leading token ids
(confirmed empirically with Qwen3-0.6B's real tokenizer). A role-specific
literal marker text is prepended ahead of each head's generated content
(diverging at its very first character) to force divergence starting at
token 0 instead.

Head length is fixed at 34 tokens and tail length at 1 token for every
setting (`--body-tokens` only controls the shared body -- the quantity
CacheTune's controller actually sizes its repair ratio against); `--repeats`
`>= 2` is enforced (a single formal repeat cannot be distinguished from
measurement noise).

Because CacheTune's reuse path requires a request's own exact-radix-match
length to equal the registered segment's `target_start` exactly (any gap
forces dense-fallback), each setting's measurement pass runs, in order:

1. Flush the exact-match radix cache once more (see below) -- this
   function runs once per setting (the main setting and every
   length-sweep point), and settings are otherwise never isolated from
   each other, so a previous setting's own already-seeded
   `target_head_ids` would otherwise still be in the tree.
2. Seed the target head once: one plain dense `/generate` call whose
   entire prompt is exactly `target_head_ids` -- this is the only way to
   populate the exact radix tree for that specific head (register/reuse
   requests always set `skip_radix_cache_insert=True` and can never do
   this themselves).
3. Register the "raw" segment once, from `source_head_ids + shared_body_ids`.
4. One *discarded* register-fresh + reuse warmup pass.
5. `--repeats` formal register-fresh + reuse repeats.

Every formal reuse response's `meta_info.cached_tokens` (generic SGLang
exact-prefix accounting, unrelated to CacheTune's own Prometheus counters)
is cross-checked against the expected head-only length, and the output
JSON's `server_validation.body_source_context_differs_from_target` is
computed from the actual constructed workload (never hardcoded).

`--central-log` is required: every invocation appends JSONL lifecycle
records (`running`, then `completed` or `failed`) to this shared log,
carrying the full settings, image/model/git identity, warmup/repeat
counts, output path, and (on success) a short result summary.

Dense flushes the exact-match radix cache before its warmup and before
every formal repeat. Each setting's own measurement pass (step 1 above)
flushes once more of its own accord, right before its head is seeded or
any raw/fresh segment is registered -- covering both dense's own
leftover exact-cache entry and cross-setting isolation for every
subsequent length-sweep point uniformly, so a dense forward's (or a
previous setting's seeded head's) exact-cache entry can never be
silently reused by a later "reuse" request. Register/reuse repeats are
never flushed between themselves -- `/flush_cache` also resets the
`ApproxKVManager` segment store, which would delete the very
"raw"/"fresh" segments those repeats depend on, and they cannot pollute
the exact radix tree themselves since
`schedule_batch.Req.skip_radix_cache_insert` is forced `True` whenever
`approx_kv_metadata` is present.

This fork has no ModelRunner hook for a genuine inline per-layer forward
over an arbitrary token subset, so every real repair in this canary uses
the precomputed fresh-KV adapter (`cachetune-raw:`/`cachetune-fresh:`
segment registration), exactly like `research/cacheblend`'s own real
canary. The controller's per-request decision (ratio, selected tokens,
recomputed layers, precomputed-adapter usage) is not exposed in the
`/generate` response body; it is only observable in aggregate through
`/metrics`, which is what this runner validates.

The runner also sweeps a few additional real body lengths against the
*same* running server/controller (no restart) to prove genuine
per-request deterministic ratio re-quantization, and reports
`dense_p50_ms`, `cachetune_target_p50_ms`, `fresh_preparation_p50_ms` and
`combined_p50_ms` (target plus fresh preparation) -- never excluding
preparation cost before claiming an end-to-end result. The one-time
per-setting `seed_head_ms`/`register_raw_ms` setup costs are always
reported (`server_validation`/each length-sweep point) but excluded from
`combined_ms`/`combined_p50_ms`, with the rationale spelled out in
`known_limitations`: both represent context that would already exist
before the measured request in a real deployment (a prior conversation
turn's own exact-cache entry, and externally sourced/precomputed KV), the
same reasoning that already excluded raw-segment registration itself. The
output JSON (`schema_version: 3`) additionally records every raw
per-repeat sample (`dense_ms_samples`/`fresh_ms_samples`/
`cachetune_ms_samples`/`combined_ms_samples`, and per-length-sweep-point
equivalents) alongside the derived medians, so the formal-repeat
measurements are always independently reproducible from the recorded
data.
