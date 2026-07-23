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
  --main-header-tokens 64 --main-body-tokens 1024 --main-target-rho 1.5 \
  --header-tokens-choices 0,32,64,128,256 \
  --body-tokens-choices 512,768,1024,2048 \
  --target-rho-choices 0.9,1.1,1.5,2,3 \
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

### TTFT measurement methodology (client TTFT is the sole metric)

Every request (`dense_generate_payload`/`register_generate_payload`/
`reuse_generate_payload`) is sent with `stream: true`. `ttft_ms` is the
client wall-clock time from just before the request is sent to the
moment the first non-`[DONE]` SSE `data:` frame is received off the
wire -- i.e. genuine time-to-first-token, timestamped before that
frame's JSON body is even parsed. This is deliberately *not* the
blocking whole-request elapsed time an earlier version of this script
used: with `max_new_tokens=1` that number is close to TTFT, but it still
bundles in the server's full-response detokenization/serialization and
the complete HTTP body transfer that only happen strictly after the
first (and only) token was already produced -- a strictly looser upper
bound on TTFT, never TTFT itself, and TTFT is this script's sole
client-facing metric.

Every stream is still read in full through the terminal `data: [DONE]`
frame before being accepted as a success -- never abandoned right after
the first chunk -- so a dropped connection, a stream that never reaches
`[DONE]`, or a mid-stream `"error"` frame (see `http_server.py`'s
`stream_results` error branch) all raise loudly instead of being
silently treated as a completed request. The *last* non-`[DONE]` frame
observed is used as the response object for `require_finished_by_length`/
`require_cached_tokens`, so those checks keep working unchanged.

Every formal-repeat raw sample (`dense_raw_samples`/`fresh_raw_samples`/
`reuse_raw_samples`/`cachetune_raw_samples`, main setting and every
length-sweep point) is a `{"ttft_ms": ..., "cached_tokens": ...}` record
pairing that exact call's genuine streaming TTFT with its own
server-reported `meta_info.cached_tokens` -- so timing and token
accounting are always cross-referenced from the very same request,
never two independently sourced numbers a reader has to trust line up.
The existing flat `*_ms_samples` float lists and `dense_p50_ms`/
`fresh_preparation_p50_ms`/`cachetune_target_p50_ms`/`combined_p50_ms`
medians are unchanged in shape and are simply the `ttft_ms` projection
of these same raw samples.

### Non-prefix segment workload (real cross-context repair, not exact replay)

The runner talks to the server's native `POST /generate` endpoint with
direct `input_ids` (never `/v1/chat/completions`), and every workload it
builds is a genuine **non-prefix** segment, not a shared-from-position-0
prefix:

* `source_prompt = source_head_ids + shared_body_ids + tail_ids`
* `target_prompt = target_head_ids + shared_body_ids + tail_ids`
* `fresh_prompt` (registered from the *target* prompt's body offset,
  under a distinguishing content-hash prefix) is token-identical to
  `target_prompt` -- safe because every request carrying
  `approx_kv_metadata` (register AND reuse) forces
  `skip_radix_cache_insert=True`, so this fresh-register call never
  populates the live server's exact radix tree.

with `source_head_ids != target_head_ids` (different token content --
the whole point) but `shared_body_ids` (and the trailing `tail_ids`)
byte-identical between all three. Each piece is tokenized
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

Header (distinct source/target head) and shared-body token counts are
controlled by `--main-header-tokens`/`--main-body-tokens` for the main
setting and swept via `--header-tokens-choices`/`--body-tokens-choices`
for the shape sweep -- the shared body is the quantity CacheTune's
controller actually sizes its repair ratio against. Only eviction-
pressure filler objects use a separate, fixed `--pressure-filler-head-
tokens` (default 34); tail length is fixed at 1 token for every setting.
`--repeats >= 2` is enforced (a single formal repeat cannot be
distinguished from measurement noise).

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
3. Register the "raw" segment: one INDEPENDENT `/generate` call per
   `<= --max-segment-chunk-tokens` chunk of `shared_body_ids` (default
   512), never one oversized call spanning the entire body -- each
   chunk's own short prompt is `corresponding_head_ids + chunk_body_
   slice + tail_ids`. A single register call whose *stored* segment
   sizes were already chunk-bounded still let that one call's own
   live/transient per-request KV footprint scale with the FULL,
   un-chunked body, which OOM'd a real SM75 server at register time for
   body lengths above one chunk (e.g. 1024/2048); splitting register
   itself into one independent call per chunk is the fix. The REUSE
   call (steps 4/5 below) is deliberately NOT chunked this way: it
   always posts the complete target prompt in one call with the
   existing contiguous multi-segment list, since a genuine full-context
   reuse/repair forward pass is exactly what this canary measures.
4. One *discarded* register-fresh + reuse warmup pass (the fresh
   registration is chunked exactly like the raw registration above).
5. `--repeats` formal register-fresh + reuse repeats (same per-chunk
   fresh registration, every repeat).

Every formal fresh-register response's `meta_info.cached_tokens` is
cross-checked against `body_start_in_target` (the REGISTER operation
never restores anything -- see `approx_kv/runtime.py`'s
`_register_request_segments` -- so its only contribution is the
exact-match radix hit on the already-seeded target head). Every formal
reuse response's `meta_info.cached_tokens` (generic SGLang accounting,
unrelated to CacheTune's own Prometheus counters) is cross-checked
against `body_start_in_target + body_tokens` -- confirmed on a real
SM75 run that this counts the *entire* prefix already resolved without
a fresh forward pass, not just the exact-match head: a successful
CacheTune reuse always extends `req.prefix_indices` by the complete
restored body span regardless of the controller's selected repair
ratio (the ratio only decides how many of those positions get a
genuine recompute forward pass versus a straight KV copy, never how
many get restored in total; see `cachetune/runtime.py`'s
`restore_request_prefix_cachetune`). The output JSON's
`server_validation.body_source_context_differs_from_target` is
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

After every setting (main, shape sweep, rho sweep) has finished, every
raw/fresh/pressure-filler segment registered along the way is still
intentionally resident -- that residency is the entire point of
"register once, reuse across repeats," not a leak. So
`capture_final_pool_reset_and_invariant` snapshots `/metrics` first
(`pool_invariant_metrics_pre_reset` in the output JSON -- informational
only; a real SM75 run observed `kv_used_tokens=4096` here even though
`accounted_tokens` already matched `max_total_num_tokens` exactly, and
that nonzero usage must never be read as a pool leak), then flushes
(also resetting `ApproxKVManager`, releasing every such segment), posts
one small fixed sentinel `/generate` request to force a real scheduler
iteration (gauges like `kv_used_tokens` are only recomputed on the
scheduler's own next iteration, not synchronously by `/flush_cache`
itself), and only then snapshots again
(`pool_invariant_metrics_post_reset`) and runs `idle_pool_invariant` --
the ONLY invariant result `passed` is ever gated on.

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
same reasoning that already excluded raw-segment registration itself.
(`register_raw_ms`/`register_fresh_ms` are each the SUM of every
`<= --max-segment-chunk-tokens` chunk's own genuine streaming TTFT --
see item 3 above -- not a single call's ms, whenever a body spans more
than one chunk.) The
output JSON (`schema_version: 3`) additionally records every raw
per-repeat sample both as `{"ttft_ms": ..., "cached_tokens": ...}`
records (`dense_raw_samples`/`fresh_raw_samples`/`cachetune_raw_samples`,
and per-length-sweep-point `fresh_raw_samples`/`reuse_raw_samples`) and
as the existing flat `ttft_ms`-only float lists (`dense_ms_samples`/
`fresh_ms_samples`/`cachetune_ms_samples`/`combined_ms_samples`, and
per-length-sweep-point equivalents) alongside the derived medians, so the
formal-repeat measurements are always independently reproducible from
the recorded data.
