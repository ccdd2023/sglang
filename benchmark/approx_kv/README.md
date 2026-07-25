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

### Eviction-pressure fillers are plain dense objects, never CacheTune segments

Every ROUND of every setting whose `target_rho` is set (the main
setting, every shape-sweep point with a nonzero header, and every
rho-sweep point) -- the one discarded warmup round AND each of
`--repeats` formal repeats, every one a fully independent unit via
`run_independent_round` -- sends a reverse-computed count of filler
objects immediately **after** THAT SAME round's own source setup
(head-seed + raw-register + fresh-register, via `register_round_setup`)
completes -- never before it, and never reusing a filler count or
footprint measurement from any other round. Each filler is sent as
exactly **one plain, ordinary `POST /generate` request** carrying no
`approx_kv` custom_params metadata at all -- never a register/reuse
call.

This ordering (source setup before pressure) is itself a fix for a real
SM75 bug at `target_rho=2`: an earlier version of this function sent
pressure fillers *first*, before the setting's own raw-segment
registration. Register's own segment materialization is not wired to
evict exact-radix victims to make room for itself -- unlike the
reuse/repair path's own recovery-slot allocation, which explicitly
*does* evict exact-radix victims before allocating (see
`allocate_recovery_slots` in `cachetune/runtime.py` /
`mem_cache/common/runtime.py`) -- so under high pressure, source setup
itself starved for device headroom and failed. The fix always completes
that round's own source setup first, at low/no pressure, then
reverse-computes the filler count from that SAME round's own real,
*measured* (never estimated) contribution to `sglang:kv_used_tokens`
(`already_pinned_tokens`, sampled via `/metrics` immediately after that
round's own setup finishes) -- fillers are only sized to reach the
"target pre-rho" still unmet by that round's own already-resident
footprint, never blind to it and never inherited from another round
(see `eviction_pressure_filler_count_for_rho`'s own
`already_pinned_tokens` parameter). That round's own target recovery
allocation (the reuse call) is explicitly expected and allowed to evict
that SAME round's own fillers to make room for itself -- that *is* the
genuine eviction pressure this canary is constructed to exercise, at
exactly the point (recovery time) it is meant to matter; only source
setup must never depend on evicting anything.

A SECOND, later real SM75 bug at `target_rho=2` -- also fixed by
restructuring, not by this ordering alone -- came from a design where
only the discarded warmup round performed this setup-then-pressure
sequence once for the WHOLE setting, and every formal repeat merely
re-registered "fresh" and re-sent a freshly re-sized pressure batch
against that SAME already-resident raw registration. Each formal
repeat's fresh re-registration then had to transiently coexist with the
warmup's still-resident raw segment plus whatever fillers survived
LRU eviction, producing two consecutive `MemoryError`s on formal
fresh-register calls, followed by target-reuse OOM. The fix (see
"measurement pass runs, in order" below) makes every round -- the
discarded warmup round and every formal repeat alike -- a fully
independent unit via `run_independent_round`: its own flush, its own
complete raw+fresh setup, its own freshly re-sized pressure phase, and
its own reuse call, never depending on or reusing anything left
resident by any other round.

The plain-dense-filler design itself is a separate, earlier fix for a
different real SM75 bug also observed at `target_rho=2`: an even
earlier version of this phase ran every filler through the full
CacheTune register-raw/register-fresh/reuse cycle, which stores each
filler's raw/fresh body in `ApproxKVManager`'s own segment store -- a
structure the Radix LRU eviction policy cannot see or reclaim at all.
Enough fillers accumulated that way (from filler[11] onward) to fill
the pool with permanently un-evictable segments, leaving no room for
the setting's own target recovery-slot allocation -- its reuse call
then reported only the exact-match head as cached, never head+body.

A plain dense request instead populates the ordinary exact radix tree,
exactly like any other request, and is fully subject to normal LRU
eviction -- genuine, realistic cache pressure that same round's own
recovery allocation can reclaim from, matching the same plain-dense-
filler methodology `research/epic-legolink`'s own
`run_phase4_epic_pressure.py` already uses. Filler object shape
(`--pressure-filler-head-tokens` x `--pressure-filler-body-tokens`,
default 2048) is fixed across every setting AND every round; only the
reverse-computed COUNT varies, both with `target_rho` and with each
round's own independently-measured `already_pinned_tokens`, keeping
peak-rho/eviction numbers comparable across the whole matrix. Fillers
still get mutually distinct, pairwise zero-common-prefix target heads
(`build_eviction_pressure_workloads`/`validate_pairwise_head_isolation`),
so pressure objects can never spuriously exact-match each other or the
setting's own head.

Because that round's own target head is seeded *before* any filler (as
part of that SAME round's own `register_round_setup`), it is the oldest
entry in the exact radix tree once that round's own pressure phase
begins -- a plausible LRU-eviction candidate itself for any
`target_rho > 1` round. `ensure_target_head_resident` runs exactly once
per round, immediately after that round's own pressure phase, to guard
against that: one plain dense re-seed request (`workload.seed_prompt_ids`,
the same `target_head_ids + seed_sentinel_ids` prompt sent in step 2
below) tolerant of any of THREE outcomes -- a full hit (`cached_tokens ==
len(target_head_ids) + len(seed_sentinel_ids)`, both head and sentinel
survived), a head-only hit (`cached_tokens == len(target_head_ids)`, the
head survived but the sentinel's own deeper node was independently
evicted), or a full miss (`cached_tokens == 0`, the head itself was
evicted and recomputed). This is an additional, script-added safeguard --
not part of CacheTune's own design -- made necessary by sending genuine
LRU eviction pressure after the head is already seeded; without it, an
evicted head could never be restored by any later register/reuse call
(both always skip radix insertion), permanently breaking every
subsequent measurement for that round.

Because pressure fillers are now genuinely evictable,
`register_eviction_pressure_objects` itself raises immediately if a
`target_rho` value that nominally requests more tokens than the pool's
TRUE evictable headroom (live measured capacity minus
`already_pinned_tokens`, not raw capacity alone) fails to move the real
`sglang:evicted_tokens_total` Prometheus counter while registering them
-- proof that genuine device-pool eviction actually occurred is not
merely reported (`pressure_phase.evicted_tokens_total_delta`/
`peak_rho_observed` in the output JSON) but enforced. A nonzero
`sglang:approx_kv_dense_fallback_total` delta during the pressure phase
also raises immediately: a plain dense filler carries no `approx_kv`
metadata and should never be able to move that CacheTune-reuse-specific
counter.

`observed_rho()` -- the shared helper behind both
`pressure_phase.observed_rho_after_pressure` and this round's own
`observed_rho_after_target` (from which `peak_rho_observed` is the max
of the two, or just the latter when no pressure phase ran) -- reports
genuine resident pool occupancy as `(kv_used_tokens + kv_evictable_tokens)
/ capacity_tokens`, NEVER `kv_used_tokens` alone. `kv_used_tokens` is
only the pool's currently pinned/in-use tokens -- for this canary
specifically, that is a round's own raw+fresh ApproxKV segment
footprint, which register/reuse deliberately never inserts into the
exact radix tree (`skip_radix_cache_insert=True`) and so is invisible to
LRU eviction; the plain-dense pressure fillers this section describes
instead land in the ordinary LRU-evictable exact-radix tree, counted by
`kv_evictable_tokens`, not `kv_used_tokens`. An earlier version of
`observed_rho()` read `kv_used_tokens` alone -- conceptually the same
quantity the server's own `sglang:full_token_usage` gauge reports
(`full_num_used / pool_size`) -- and so drastically undercounted
genuine pressure whenever a large filler population remained resident:
a real `target_rho=2` SM75 canary reported `peak_rho_observed=0.156`
(`2048 / 13130`, `kv_used_tokens` alone) on a pool that was in fact
`(2048 + 10960) / 13130 ~= 0.991` resident once every surviving filler
was counted too. `observed_rho()` now raises immediately -- never
silently substitutes 0 or falls back to `kv_used_tokens` alone -- if
either `sglang:kv_used_tokens` or `sglang:kv_evictable_tokens` is
missing from a snapshot; `capacity_tokens` itself is unaffected by this
fix and remains the fixed, once-per-round value `usable_kv_capacity_tokens`
establishes immediately after that round's own flush-and-gauge-refresh
(`flush_and_force_gauge_refresh`, see step 1 below), never recomputed
from a later, possibly mid-pressure snapshot.

Because CacheTune's reuse path requires a request's own exact-radix-match
length to equal the registered segment's `target_start` exactly (any gap
forces dense-fallback), each setting's measurement pass runs one
discarded warmup ROUND followed by `--repeats` formal ROUNDS, every
round -- warmup and formal alike -- an identically-shaped, fully
independent unit via `run_independent_round` (no round ever depends on
or reuses anything left resident by another round). Each round runs, in
order:

1. Flush the exact-match radix cache, force one real scheduler
   iteration via a small fixed dense sentinel request, and snapshot
   `/metrics` (`flush_and_force_gauge_refresh`) -- this now runs at the
   START OF EVERY ROUND (the discarded warmup round AND each of
   `--repeats` formal repeats independently, never merely once per
   setting), so that round's own `capacity_tokens` /
   `already_pinned_tokens` baseline always reflects a genuinely idle,
   just-flushed pool, isolated both from a previous setting's own
   seeded `target_head_ids` and from any earlier round's (within this
   SAME setting) own registered segments or surviving pressure
   fillers. A bare flush alone is not enough: `/flush_cache` clears
   the actual pool/tree state synchronously, but the exported
   `sglang:kv_used_tokens` Prometheus GAUGE is only recomputed by the
   scheduler's own NEXT real request, so a flush-then-immediate-
   snapshot with no intervening request can read a value carried over
   from whatever was resident just before the flush. This is a real
   SM75 bug from a body-length sweep: a bare flush-then-snapshot let
   one setting's own final `kv_used_tokens` reading (2048, from a
   just-finished body=1024 setting's own raw+fresh footprint) leak
   into the NEXT setting/round's own baseline, producing
   `already_pinned_tokens = 1024 - 2048 = -1024` for a genuine
   body=512 setting -- a structurally negative value the
   `already_pinned_tokens < 0` `ValueError` guard in
   `eviction_pressure_filler_count_for_rho` correctly refused to
   silently clamp away. `flush_and_force_gauge_refresh` is the SAME
   flush-sentinel-snapshot pattern
   `capture_final_pool_reset_and_invariant` already used for this
   run's own final pool reset (see below) -- now shared and applied at
   every round's own start too. Because the sentinel is a plain dense
   request (not radix-insert-skipped), it remains resident in the exact
   radix tree until flushed again; a SECOND, bare flush immediately
   follows (needing no sentinel of its own) to guarantee a genuinely
   empty tree before step 2's own head-seed call below, which would
   otherwise risk an exact-match collision against the sentinel's own
   token(s).
2. Seed the target head once (this round's own copy): one plain dense
   `/generate` call whose prompt is `workload.seed_prompt_ids`, i.e.
   `target_head_ids + seed_sentinel_ids` -- never bare `target_head_ids`
   alone -- this is the only way to populate the exact radix tree for
   that specific head (register/reuse requests always set
   `skip_radix_cache_insert=True` and can never do this themselves). The
   trailing `seed_sentinel_ids` fixes a real SM75 bug at header length
   32: a bare-`target_head_ids` seed's own `max_new_tokens=1` greedy
   generation can coincidentally equal `shared_body_ids[0]`, and since
   this seed call is a plain dense request (never radix-insert-
   skipped), that generated token silently extends the tree's own
   exact-match boundary for that head by one token -- observed as a
   later fresh-register call reporting `cached_tokens=33` for a
   32-token header. `seed_sentinel_ids` is built (see
   `_build_seed_sentinel_ids_avoiding_body_first_token_collision`) to
   tokenize to a first id that provably differs from
   `shared_body_ids[0]` for THIS workload's own real tokenizer,
   independent of the seed request's own generated continuation -- so
   later `target_head_ids + shared_body_ids + ...` exact-match queries
   always diverge at exactly `len(target_head_ids)`, regardless of what
   the seed call itself generates.
3. Register the "raw" segment (this round's own copy): one INDEPENDENT
   `/generate` call per `<= --max-segment-chunk-tokens` chunk of
   `shared_body_ids` (default 512), never one oversized call spanning
   the entire body -- each chunk's own short prompt is
   `corresponding_head_ids + chunk_body_slice + tail_ids`. A single
   register call whose *stored* segment sizes were already
   chunk-bounded still let that one call's own live/transient
   per-request KV footprint scale with the FULL, un-chunked body, which
   OOM'd a real SM75 server at register time for body lengths above one
   chunk (e.g. 1024/2048); splitting register itself into one
   independent call per chunk is the fix.
4. Register the "fresh" segment (this round's own copy, same
   per-chunk chunking as raw above). Steps 2+3+4 together are THIS
   ROUND's own complete SOURCE setup (`register_round_setup`), always
   finished in full before any pressure filler THIS SAME round sends
   (see above) -- never before a previous round's own setup, and never
   shared with any other round.
5. If `target_rho` is set: measure this round's own real, post-setup
   `already_pinned_tokens` contribution, send this round's own
   reverse-computed pressure fillers sized from it (see above), then
   re-seed the target head once via `ensure_target_head_resident`
   (guard against LRU eviction of the head itself during this round's
   own pressure phase).
6. This round's own single reuse call (`run_target_reuse`) against the
   raw/fresh segments THIS SAME round just registered in steps 3-4 --
   never against a different round's own registration. The reuse call
   is deliberately NOT chunked like steps 3-4: it always posts the
   complete target prompt in one call with the existing contiguous
   multi-segment list, since a genuine full-context reuse/repair
   forward pass is exactly what this canary measures.

The discarded warmup round runs this exact same six-step sequence once
and its own result is thrown away; each of the `--repeats` formal
repeats then runs the identical six-step sequence again, completely
independently, its own result recorded. This fixes a real SM75
`target_rho=2` bug from an earlier design where only the warmup round
performed steps 1-4 once for the whole setting and every formal repeat
merely re-ran steps 4+6 (re-register fresh, then reuse) against that one
shared raw registration -- see the "Eviction-pressure fillers" section
above for the full root-cause account.

Every formal fresh-register response's `meta_info.cached_tokens` is
cross-checked against `body_start_in_target` (the REGISTER operation
never restores anything -- see `approx_kv/runtime.py`'s
`_register_request_segments` -- so its only contribution is the
exact-match radix hit on that SAME round's already-seeded target head).
Every formal reuse response's `meta_info.cached_tokens` (generic SGLang
accounting, unrelated to CacheTune's own Prometheus counters) is
cross-checked against `body_start_in_target + body_tokens` -- confirmed
on a real SM75 run that this counts the *entire* prefix already resolved
without a fresh forward pass, not just the exact-match head: a
successful CacheTune reuse always extends `req.prefix_indices` by the
complete restored body span regardless of the controller's selected
repair ratio (the ratio only decides how many of those positions get a
genuine recompute forward pass versus a straight KV copy, never how
many get restored in total; see `cachetune/runtime.py`'s
`restore_request_prefix_cachetune`). The output JSON's
`server_validation.body_source_context_differs_from_target` is
computed from the actual constructed workload (never hardcoded). Every
formal round's own complete raw telemetry (setup timings, pressure
sizing/eviction, head-reseed, reuse) is additionally available in full,
per-round, via the output JSON's `rounds` list -- never collapsed to
just a single setting-wide value, now that every round independently
re-sizes and re-sends its own pressure phase.

`--central-log` is required: every invocation appends JSONL lifecycle
records (`running`, then `completed` or `failed`) to this shared log,
carrying the full settings, image/model/git identity, warmup/repeat
counts, output path, and (on success) a short result summary.

Dense flushes the exact-match radix cache before its warmup and before
every formal repeat -- unchanged by, and unrelated to, the CacheTune
per-round restructuring below. Each CacheTune setting's own measurement
pass (step 1 above) now flushes at the START OF EVERY ROUND, not once
per setting: the discarded warmup round's own flush, and each formal
repeat's own flush, every one immediately before that SAME round's own
head is seeded or any raw/fresh segment is registered. This covers
cross-setting isolation (a previous setting's own seeded
`target_head_ids` can never still be in the tree) AND cross-round
isolation within the SAME setting (an earlier round's own registered
segments, surviving pressure fillers, or seeded head can never still be
resident when a later round's own setup begins) uniformly. Flushing
between what an earlier design called "repeats" is now not merely safe
but MANDATORY: `/flush_cache` also resets the `ApproxKVManager` segment
store, but every round -- including every formal repeat -- always
re-registers its OWN raw+fresh segments from scratch immediately
afterward (steps 2-4 above), so there is nothing left over from a
previous round for a later round to depend on. (Register/reuse calls
still cannot pollute the exact radix tree themselves, independent of
this flush, since `schedule_batch.Req.skip_radix_cache_insert` is forced
`True` whenever `approx_kv_metadata` is present.) See
`flush_exact_radix_cache`'s own docstring, and `run_independent_round`'s
own docstring, for the full real-SM75-bug account of why every round
must now flush independently.

After every setting (main, shape sweep, rho sweep) has finished, the
LAST formal round's own raw/fresh segments are still intentionally
resident in `ApproxKVManager` -- simply because nothing flushes again
until the START of the next round (the next setting's own discarded
warmup round, or this run's own final cleanup below), not because
anything is "registered once and reused across repeats" any more (every
earlier round's own registration was already wiped by ITS successor
round's own flush, per the per-round design above). This trailing
residency is expected, not a leak. (Eviction-pressure filler objects are
never registered at all -- see below -- so they hold no such residency
by construction; whichever fillers the last round's own LRU pressure did
not evict may still remain in the exact radix tree at this point, which
is likewise expected, not a leak.) So `capture_final_pool_reset_and_invariant`
snapshots `/metrics` first
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
the ONLY invariant result `passed` is ever gated on. This flush ->
sentinel -> snapshot sequence is `flush_and_force_gauge_refresh`, a
shared helper also used by every ROUND's own start (step 1 above) --
the same real-request-forces-a-gauge-refresh requirement applies
identically whether it is this run's own final reset or one setting's
own next round, and a real SM75 body-length-sweep bug (a stale,
carried-over `kv_used_tokens` reading corrupting a round's own
`already_pinned_tokens`) is what made that sharing necessary.

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
preparation cost before claiming an end-to-end result. The
`seed_head_ms`/`register_raw_ms` setup costs are now genuinely
re-measured by EVERY round (warmup and every formal repeat
independently re-seed and re-register from scratch, per the per-round
design above); the setting-level `server_validation`/each
length-sweep-point result reports the LAST formal round's own value
(every other round's own value remains available in full via that same
result's `rounds` list). Both are excluded from
`combined_ms`/`combined_p50_ms`, with the rationale spelled out in
`known_limitations`: both represent context that would already exist
before the measured request in a real deployment (a prior conversation
turn's own exact-cache entry, and externally sourced/precomputed KV), the
same reasoning that already excluded raw-segment registration itself.
(`register_raw_ms`/`register_fresh_ms` are each the SUM of every
`<= --max-segment-chunk-tokens` chunk's own genuine streaming TTFT --
see items 3-4 above -- not a single call's ms, whenever a body spans
more than one chunk.) The
output JSON (`schema_version: 3`) additionally records every raw
per-repeat sample both as `{"ttft_ms": ..., "cached_tokens": ...}`
records (`dense_raw_samples`/`fresh_raw_samples`/`cachetune_raw_samples`,
and per-length-sweep-point `fresh_raw_samples`/`reuse_raw_samples`) and
as the existing flat `ttft_ms`-only float lists (`dense_ms_samples`/
`fresh_ms_samples`/`cachetune_ms_samples`/`combined_ms_samples`, and
per-length-sweep-point equivalents) alongside the derived medians, so the
formal-repeat measurements are always independently reproducible from
the recorded data.

## Corrected Phase 4 R5 key rerun

`run_phase4_cachetune_key_rerun.py` remeasures body1024/2048 at rho=2 with
three server restarts and paired dense baselines.

The corrected path runs GPU-only and uses an isolated exact-cache namespace to
incrementally materialize each raw/fresh cumulative causal prefix before
registering only the current <=512-token chunk. It then applies pressure,
re-seeds the target head and runs the real CacheTune target.

The result keeps separate target-only, fresh-adapter-combined, request-path
and full-lifecycle ledgers. It also records the committed SM75 controller
decision, first-token equality, selected-token/recomputed-layer counters,
eviction, fallback and pool-reset evidence.

Committed result:

`benchmark/approx_kv/results/phase4-r5/sm75-causal-key-rerun.json`
