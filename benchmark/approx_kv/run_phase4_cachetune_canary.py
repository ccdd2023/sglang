#!/usr/bin/env python3
from __future__ import annotations

"""Phase 4 R5 CacheTune SM75 canary runner.

Connects to a live SGLang server that already has the CacheTune recovery
plugin registered (``SGLANG_APPROX_KV_CACHETUNE=1``, plus a deployment-wide
hardware measurement from ``SGLANG_CACHETUNE_T_C_MS`` / ``_T_I_MS`` /
``_T_O_MS`` -- see ``cachetune/plugin.py``) and issues real, streamed
native ``/generate`` requests (direct ``input_ids``, never
``/v1/chat/completions``) to exercise the genuine, non-simulated CacheTune
request path end to end.

TTFT measurement methodology (client TTFT is this script's sole metric)
------------------------------------------------------------------------
Every request this script issues sets ``stream: true`` and measures
``ttft_ms`` as the wall-clock time from just before the request is sent
to the moment the first non-``[DONE]`` SSE ``data:`` frame is received
off the wire -- i.e. genuine client time-to-first-token, timestamped
before that frame's JSON body is even parsed. This is *not* the same
number as this script's own previous approach of timing a blocking
(non-streamed) request end to end: with ``max_new_tokens=1`` that
blocking-elapsed number is close to TTFT, but it still bundles in the
server's full-response detokenization/serialization and the complete
HTTP body transfer that happen strictly *after* the first (and, given
``max_new_tokens=1``, only) token was already produced -- a strictly
looser upper bound on TTFT, never TTFT itself, and this script's sole
client-facing metric deserves the real thing rather than an
approximation. See ``timed_post``/``_stream_generate_and_measure_ttft``
for the implementation. Every stream is still read in full through the
terminal ``data: [DONE]`` frame before being accepted as a success --
never abandoned right after the first chunk -- so a connection that
drops or an error frame mid-stream fails loudly instead of silently
reporting a "successful" TTFT for a request that never actually
finished. Every raw per-repeat sample this script records (dense, fresh
preparation, reuse) carries both ``ttft_ms`` and the server-reported
``meta_info.cached_tokens`` for that exact call together (see the
``*_raw_samples`` fields in the output JSON), so token-accounting and
timing are always cross-referenced from the very same request, never
two independently-sourced numbers a reader has to trust line up.

Why /generate and non-prefix segments
--------------------------------------
An earlier version of this script built "source"/"target" prompts from the
*same* chat-templated object differing only in a trailing suffix, then
registered the "raw" segment as ``common_prefix_token_ids(source, target)``
starting at ``target_start=0``. Because causal attention only depends on
*preceding* tokens, and both prompts were byte-identical from position 0
up to that longest-common-prefix boundary, the segment's KV was
*bit-identical* whether computed under "source" or "target" context --
there was no real cross-context divergence to bridge. That configuration
only proved exact-content transplantation works; it never actually
exercised CacheTune's approximate repair mechanism (the mechanism this
canary exists to validate).

This version instead builds a genuine non-prefix workload (see
``NonPrefixSegmentWorkload``/``build_non_prefix_segment_workload``):

* ``source_prompt = source_head_ids + shared_body_ids + tail_ids``
* ``target_prompt = target_head_ids + shared_body_ids + tail_ids``
* ``fresh_prompt`` (registered from the *target* prompt's body offset,
  under a distinguishing content-hash prefix) is token-identical to
  ``target_prompt`` -- see ``NonPrefixSegmentWorkload.fresh_prompt_ids``
  for why that equality is intentional and safe.

with ``source_head_ids != target_head_ids`` (different token content --
the whole point) but ``shared_body_ids`` (and the trailing ``tail_ids``)
byte-identical between all three. The "raw" segment is registered from
the *source* prompt's body offset; the "fresh"/reuse segment is
registered/matched from the *target* prompt's body offset. Each of
source_head/target_head/shared_body/tail is tokenized *separately*
(``tokenizer.encode(text, add_special_tokens=False)``) and the resulting
integer-id lists are concatenated directly -- never re-tokenized as a
joined string -- so segment offsets are exact by construction, with no
BPE boundary-merging risk across piece boundaries.

``source_head_ids``/``target_head_ids`` are more than merely *unequal*:
they are constructed to share **zero** common exact-match token prefix
(``NonPrefixSegmentWorkload.__post_init__`` verifies this explicitly).
This is not automatic from picking two different seeds:
``workloads.deterministic_code``'s first block always begins with the
literal, seed-*independent* text ``"def synthetic_0_"`` before any
seed-dependent digest character appears, so two ``_deterministic_token_ids``
calls with different seeds alone reliably tokenize to several *identical*
leading token ids (confirmed empirically with Qwen3-0.6B's real
tokenizer: a 4-6 token shared prefix) -- exactly the kind of
deterministic, every-run collision that would make the live server's
exact radix tree report a nonzero ``cached_tokens`` for the raw-segment
register request immediately after ``target_head_ids`` is seeded,
silently corrupting the "must be exactly 0" measurement invariant. To
prevent this, ``build_non_prefix_segment_workload`` prepends a
role-specific literal marker (``_SOURCE_HEAD_LITERAL_PREFIX`` /
``_TARGET_HEAD_LITERAL_PREFIX``, diverging at their very first character)
ahead of each head's generated text via ``_deterministic_token_ids``'s
``literal_prefix`` parameter, forcing divergence starting at token 0
rather than relying on the seed-dependent digest alone.

CacheTune's reuse path requires the live request's own exact-radix-match
length to equal the registered segment's ``target_start`` *exactly* (any
other "prefix gap" forces dense-fallback; see ``cachetune/runtime.py``).
The only way to make a request's own ``exact_length`` equal
``len(target_head_ids)`` is to first send a plain dense request whose
prompt is ``workload.seed_prompt_ids`` -- ``target_head_ids`` PLUS an
explicit, per-workload sentinel token appended past it (never
``target_head_ids`` alone; see ``NonPrefixSegmentWorkload.
seed_prompt_ids`` for the real SM75 header-sweep bug a bare-head seed
caused and why the sentinel fixes it) -- populating the exact radix
tree for that head -- since dense requests are the only request
type that ever write into the exact tree (register/reuse always set
``skip_radix_cache_insert=True``). This is why every ROUND (the one
discarded warmup round and each of ``--repeats`` formal rounds alike,
all fully independent of each other -- see ``run_independent_round``)
runs its own measurement pass, in order: flush -> seed the target head
(one dense ``/generate`` call over ``target_head_ids + seed_sentinel_
ids``) -> register the raw segment (one OR MORE ``/generate`` register
calls, one per ``<= --max-segment-chunk-tokens`` chunk of
``shared_body_ids`` -- see ``register_body_chunks`` -- each posting a
short ``source_head_ids + chunk_body + tail_ids`` prompt, never one
oversized call spanning ``source_head_ids + shared_body_ids +
tail_ids`` in full)
-> register the fresh segment the same chunked way -> reuse. Every
round performs ALL of these steps itself, from scratch, every time --
never sharing a raw or fresh registration with any other round (see
``run_independent_round``'s own docstring for the real SM75
``target_rho=2`` ``MemoryError`` this per-round independence fixes).
The mandatory ``tail_ids`` (fixed at
``NON_PREFIX_TAIL_TOKENS`` token(s)) is appended to EVERY one of
``source_prompt_ids``/``fresh_prompt_ids``/``target_prompt_ids`` -- not
just the reuse target -- so every raw register, fresh register, AND
reuse request alike still has a genuine final forward pass beyond the
restored range, matching ``ApproxKVRequestMetadata``'s own invariant
that a request's last prompt token is never included in any restorable
segment (see ``NonPrefixSegmentWorkload``'s own docstring: omitting the
tail from the raw/fresh register prompts specifically -- an earlier
version of this script did -- previously raised ``ValueError`` inside
``Req.__init__`` on a real SM75 run and killed the scheduler).

The raw and fresh body registrations mentioned above are each actually
sent as one or more INDEPENDENT ``/generate`` calls -- one per
``<= --max-segment-chunk-tokens`` chunk of ``shared_body_ids``, never
one oversized call spanning the entire body -- via
``register_body_chunks`` (see that function's own docstring for the
real SM75 register-time OOM this fixes, and why the resulting per-chunk
register-vs-reuse position mismatch is safe). Only the register side is
split this way; every reuse call still posts the complete target prompt
in one request.

Measurement protocol (mandatory, applies to every setting: the main
setting, every shape-sweep point, and every rho-sweep point)
--------------------------------------------------------------------
Every setting runs one discarded warmup round plus ``--repeats``
(``>= 2``) formal rounds -- ``repeats + 1`` rounds total, ALL
structurally identical and each FULLY INDEPENDENT of every other round
(``run_independent_round``, invoked by ``run_non_prefix_setting``).
"Fully independent" means every round performs its own flush, its own
target-head seed, its own raw+fresh registration, its own
eviction-pressure phase (if enabled), and its own reuse call -- NEVER
reusing another round's already-registered raw or fresh segment, and
never carrying forward another round's own pressure fillers. This is a
deliberate architecture, not an incidental one: an earlier design
registered the raw segment ONCE per *setting* (``register_non_prefix_sources``,
since removed) and re-registered only the fresh segment on every repeat
(``run_reuse_once``, since removed), sharing that one raw registration
across the discarded warmup and every formal repeat. Under real SM75
``target_rho=2`` pressure this produced two consecutive ``MemoryError``s
on formal fresh-register calls, followed by target reuse OOM: each
repeat's fresh registration needed to transiently coexist with the
setup's still-resident raw segment plus surviving pressure fillers, and
register-side segment materialization (for BOTH raw and fresh) is not
wired to evict exact-radix victims to make room for itself (unlike the
reuse/repair path's own recovery-slot allocation, which explicitly does
evict -- see ``allocate_recovery_slots`` in ``cachetune/runtime.py`` /
``mem_cache/common/runtime.py``). Making every round fully independent
-- so each round's raw+fresh registration always runs against a
genuinely fresh, just-flushed idle pool, never atop a previous round's
already-resident footprint -- removes that transient double-footprint
entirely. Each round, in order:
1. Dense baseline (no ``approx_kv`` metadata) runs *entirely* to
   completion before anything else in this run: flush the exact-match
   radix cache before its own discarded warmup, before every one of its
   own formal repeats, and once more before CacheTune's own
   registration begins at all (see ``run_canary``'s own dense-baseline
   block, which already follows this same fully-independent-round shape
   and is the pattern ``run_independent_round`` was built to match).
   This must happen first -- a real dense forward's exact-cache entry
   over the *same* tokens a later request targets would let the
   scheduler's own prefix match resolve the whole prompt before
   CacheTune's plugin dispatch ever runs.
2. ``run_independent_round`` flushes the exact-match radix cache (and
   resets ``ApproxKVManager``'s own segment store -- see
   ``flush_exact_radix_cache``'s own docstring), then posts one small,
   fixed dense *sentinel* request to force one real scheduler iteration,
   then snapshots ``/metrics`` -- via ``flush_and_force_gauge_refresh``
   -- as its own very first action, before doing anything else -- once
   for the discarded warmup round, and independently again for every
   formal repeat, never just once per setting. The sentinel is
   mandatory, not optional: ``/flush_cache`` clears the actual pool/tree
   state synchronously, but gauges such as ``sglang:kv_used_tokens`` are
   only recomputed by the scheduler's own NEXT iteration, so a bare
   flush-then-snapshot with no intervening real request can read a
   value carried over from a PREVIOUS round or even a previous
   *setting* -- the real SM75 body-length-sweep bug this fixes: an
   earlier design's bare flush-then-snapshot let a body=512 setting's
   own ``metrics_at_round_start`` inherit a just-finished body=1024
   setting's own stale ``kv_used_tokens=2048`` reading verbatim,
   producing a structurally negative ``already_pinned_tokens=-1024``
   once that setting's own post-setup reading (a genuine 1024) was
   compared against it (see ``flush_and_force_gauge_refresh``'s own
   docstring for the full account). This is what makes steps 3-5 below
   safe for EVERY round, not merely the first: a *previous* round's own
   already-seeded ``target_head_ids``, already-registered raw/fresh
   segments, and already-sent pressure fillers would otherwise still be
   resident, either silently producing a nonzero ``cached_tokens`` for
   this round's own head-seed/register calls or -- the real SM75
   ``target_rho=2`` bug this per-round flush fixes -- forcing this
   round's own raw+fresh registration to transiently coexist with a
   previous round's still-resident footprint. This now-genuinely-fresh
   ``/metrics`` snapshot gives this round's own ``capacity_tokens``
   reference; a SECOND, bare flush (no sentinel needed) immediately
   follows to clear away the gauge-refresh sentinel's own tiny resident
   footprint before step 3 below runs.
3. Seed the target head (one dense ``/generate`` call over
   ``target_head_ids + seed_sentinel_ids`` -- never ``target_head_ids``
   alone; see ``NonPrefixSegmentWorkload.seed_prompt_ids`` for the real
   SM75 header-sweep bug a bare-head seed caused and why the sentinel
   fixes it -- expected ``cached_tokens=0`` -- always 0, since step 2's
   own second, final flush just cleared it again for this round).
4. Register the "raw" (source-context) body segment, THEN the "fresh"
   (target-context) body segment -- each one or more
   ``register_body_chunks`` calls (raw expected ``cached_tokens=0`` per
   chunk, fresh expected ``cached_tokens=body_start_in_target`` per
   chunk -- see the ``source_head_ids``/``target_head_ids``
   zero-common-prefix discussion above). Steps 3+4 together are this
   ROUND's own complete setup (``register_round_setup``) -- raw AND
   fresh both -- always finished in full BEFORE step 4a below runs:
   register's own segment materialization is not wired to evict
   exact-radix victims to make room for itself, so it must always run
   while the pool is still at (or near) THIS round's own post-flush
   idle baseline -- see ``run_independent_round``'s own docstring for
   the real SM75 ``target_rho=2`` ``MemoryError`` this per-round
   setup-before-pressure ordering fixes.
4a. If eviction pressure is enabled (the default -- see "Eviction-
   pressure phase" below), every filler object is sent here, immediately
   AFTER step 4 completes for THIS round (never before it, and freshly
   rebuilt and re-sent every round -- never reused from any other
   round), as a plain dense ``/generate`` request -- never
   registered/materialized through CacheTune's own
   register-raw/register-fresh/reuse cycle. The filler count is
   reverse-computed against THIS round's own real, measured
   contribution to ``sglang:kv_used_tokens`` (``already_pinned_tokens``),
   never blind to it and never inherited from any other round -- see
   "Eviction-pressure phase" below.
4b. If step 4a ran, one guard re-seed of the target head
   (``ensure_target_head_resident``, also over ``target_head_ids +
   seed_sentinel_ids``): step 3's own seed is the OLDEST
   exact-radix entry once THIS round's own pressure begins, a plausible
   LRU-eviction candidate for any ``target_rho > 1`` setting; this guard
   call tolerates any of three outcomes (full hit, head-only hit, or
   miss -- see ``ensure_target_head_resident``'s own docstring) and
   ensures the head is resident again before step 5 -- see that
   function's own docstring for why this additional, script-added
   safeguard is necessary under this ordering. Called once per round,
   never once per setting.
5. One reuse call against the just-registered (this round's own)
   raw/fresh segments (``run_target_reuse``). For the discarded warmup
   round, this round's entire result -- including this reuse call's own
   telemetry -- is thrown away by ``run_non_prefix_setting``. For each
   formal repeat, this reuse call's raw wall-clock TTFT and
   ``meta_info.cached_tokens`` are recorded -- never just a derived
   median. Every formal fresh-register response's (step 4's own)
   ``meta_info.cached_tokens`` is checked against
   ``body_start_in_target`` (the REGISTER operation never restores
   anything -- see ``approx_kv/runtime.py``'s
   ``_register_request_segments`` -- so its only contribution to
   ``prefix_indices`` is the exact-match radix hit on the just-seeded
   target head). Every formal reuse response's ``meta_info.cached_tokens``
   is checked against ``body_start_in_target + body_tokens`` -- a real
   GPU run confirmed that SGLang's ``cached_tokens`` accounting
   (``pre_len - already_computed`` in ``schedule_batch.py``) counts the
   *entire* prefix already resolved without a fresh forward pass, and a
   successful CacheTune reuse always extends ``req.prefix_indices`` by
   the full restored body length (``restore_length``, i.e. every
   registered segment's combined span) regardless of the controller's
   selected repair ratio -- ``decision.repair_tokens`` only decides how
   many of those already-restored positions get a genuine recompute
   forward pass versus a straight KV copy, never how many positions get
   restored in total (see ``cachetune/runtime.py``'s
   ``restore_request_prefix_cachetune``). Neither check is a tautology
   against the same telemetry this script already cross-validates in
   aggregate: both are independent, per-request, server-reported
   signals (generic SGLang accounting, unrelated to CacheTune's own
   Prometheus counters) that the live request's own prefix boundary
   landed exactly where expected.
6. Prometheus telemetry deltas are cross-checked using only the formal
   repeat count -- the discarded warmup round's own telemetry
   contribution is excluded by construction, since ``metrics_before`` is
   always the FIRST formal round's own start-of-round snapshot, taken
   strictly after the warmup round (and its own flush) has already fully
   completed.
7. After every setting above (main, shape sweep, rho sweep) has
   completed, ``capture_final_pool_reset_and_invariant`` flushes every
   still-resident raw/fresh CacheTune segment (every setting's own
   registrations) and every dense-cached exact-radix entry (including
   every eviction-pressure filler object, sent as plain dense requests
   -- see ``register_eviction_pressure_objects`` -- never registered
   through CacheTune's own segment store) this whole run produced,
   forces one real scheduler iteration with a small fixed sentinel
   ``/generate`` request, and only THEN snapshots ``/metrics``
   and runs ``idle_pool_invariant`` -- never against the pre-flush
   snapshot, whose nonzero ``kv_used_tokens`` is expected (a real SM75
   run observed exactly this: 4096 used tokens with ``accounted_tokens``
   already matching ``max_total_num_tokens``) and must never be
   misreported as a pool leak. Both the pre-reset and post-reset raw
   ``/metrics`` snapshots are saved in the output JSON
   (``pool_invariant_metrics_pre_reset``/``_post_reset``) for
   visibility, but only the post-reset snapshot's invariant gates
   ``passed``.

Flushing before every round (the discarded warmup AND every formal
repeat alike) is MANDATORY, not merely tolerated: it is what gives each
round its own genuinely independent, just-flushed idle pool to register
raw+fresh against, removing the transient "previous round's
raw/fresh/fillers still resident" double-footprint described above.
This reverses an earlier design's own rule, which forbade flushing
between formal register+reuse repeats specifically because that earlier
design shared ONE raw registration across every repeat -- flushing
would have wiped the very segment those repeats depended on. That is no
longer true: every round now registers its own raw AND fresh from
scratch, so there is nothing left for a flush to wipe out from under a
later step within the SAME round, and the flush's Counter-vs-Gauge-safe
semantics (see ``flush_exact_radix_cache``'s own docstring) make the
resulting cross-round Prometheus deltas mathematically sound regardless
of how many independent flushes separate them: ``sglang:kv_used_tokens``
is a Gauge that resets on flush (exactly what lets every round measure
its OWN idle capacity/pinned footprint), while every Counter this script
reads (``sglang:evicted_tokens_total``,
``sglang:approx_kv_dense_fallback_total``,
``sglang:approx_kv_cachetune_selected_tokens_total``, etc.) is monotonic
and unaffected by flush, so a delta spanning multiple independent
flush-separated rounds remains exactly equal to the sum of each round's
own delta.

Eviction-pressure phase (real GPU contention, not a single-object
microbenchmark, and never CLI-disableable)
--------------------------------------------------------------------
Every ROUND -- the discarded warmup round and every formal repeat
alike, for the main setting, every shape-sweep point with header > 0,
and every rho-sweep point -- always sends a freshly reverse-computed set
of distinct filler ``NonPrefixSegmentWorkload`` objects (see
``build_eviction_pressure_workloads``) immediately AFTER that round's
own setup (head-seed + raw-register + fresh-register, see
``register_round_setup``) completes, NEVER before it (see
``register_eviction_pressure_objects``). This ordering is itself a
deliberate fix for a real SM75 bug at ``target_rho=2``: an earlier
version of this phase ran BEFORE source setup, and register's own
segment materialization is not wired to evict exact-radix victims to
make room for itself (unlike the reuse/repair path's own recovery-slot
allocation, which explicitly does) -- under high pressure, setup then
starved for device headroom and failed. Pressure is resent from scratch
EVERY round -- never built once per setting and reused across rounds --
because every round is itself fully independent (see
``run_independent_round``'s own docstring for the real SM75
``target_rho=2`` ``MemoryError`` this per-round independence, including
per-round pressure, fixes). The filler object COUNT is reverse-computed
(see ``eviction_pressure_filler_count_for_rho``) from that setting's own
``target_rho`` (``--main-target-rho`` for the main setting and every
shape-sweep point, or the specific ``--target-rho-choices`` value under
test for a rho-sweep point) against a real, live, idle
``usable_kv_capacity_tokens`` snapshot (see
``benchmark.approx_kv.metrics``) taken immediately after THIS round's
own flush, NET OF ``already_pinned_tokens`` -- THIS round's own setup's
real, measured (never estimated) contribution to
``sglang:kv_used_tokens``, sampled immediately after that same round's
own setup completes -- never a fixed object count, never blind to what
that round's own setup has already consumed, and never inherited from
any other round. Every filler object's own SHAPE
(``--pressure-filler-head-tokens``, default ``NON_PREFIX_HEAD_TOKENS``,
x ``--pressure-filler-body-tokens``, default 2048) is fixed across every
round so only the reverse-computed COUNT varies with ``target_rho``,
keeping peak-rho/eviction numbers comparable across the whole matrix.

Each filler is sent as exactly ONE plain, ordinary dense ``/generate``
request (``dense_generate_payload`` over that filler's own
``target_prompt_ids``) -- carrying NO ``approx_kv`` custom_params
metadata whatsoever, never a register/reuse call. This is a deliberate
fix for a separate real, previously-observed SM75 bug at
``target_rho=2``: an earlier version of this phase ran every filler
through a full seed-head + raw-register + fresh-register + reuse
CacheTune cycle, which captured each filler's raw/fresh body into
``ApproxKVManager``'s own segment store (see ``approx_kv/runtime.py``)
-- a structure the Radix LRU eviction policy has no knowledge of and
cannot reclaim AT ALL. With enough fillers registered that way (observed:
from filler[11] onward at ``target_rho=2`` on a real run), the pool
filled with permanently un-evictable segments, leaving no room for the
setting's own target recovery-slot allocation -- its own reuse call then
only restored the exact-match head (``cached_tokens`` reported head-only,
never head+body). A plain dense request, by contrast, populates the
ordinary exact radix tree exactly like any other request and is fully
subject to normal LRU eviction -- genuine, realistic cache pressure that
round's own recovery allocation CAN reclaim from, exactly the way a real
deployment's unrelated concurrent traffic would behave (the same
"R1"/"R4"-round plain-dense-filler methodology already used by
``research/epic-legolink``'s own ``run_phase4_epic_pressure.py``).
Because every round's own flush clears the *entire* exact radix tree
(and resets ``ApproxKVManager`` too, though fillers no longer touch it
at all), filler objects cannot be built once globally -- or even once
per setting -- and expected to persist across rounds: they are rebuilt
fresh, from the same fixed shape but a per-round-appropriate count,
inside every ``run_independent_round`` call.

Because a round's own target head is seeded (by ``register_round_setup``)
BEFORE any filler, it is the OLDEST entry in the exact radix tree once
THAT round's own pressure phase begins -- a plausible LRU-eviction
candidate for any ``target_rho > 1`` setting. ``ensure_target_head_resident``
runs once per round, immediately after that round's own pressure phase
completes, to guard against exactly that (tolerant of a survived full
hit, a survived head-only hit, or an evicted-and-recomputed miss -- see
that function's own docstring for why all three are legitimate) -- see
that function's own
docstring for why this additional, script-added safeguard (not part of
the paper's own design) is necessary under this ordering, and why one
guard call per round suffices for the remainder of that same round
(never once per setting: every round protects only its own head).

``NonPrefixSegmentWorkload``'s own ``source_head_ids``/
``source_prompt_ids``/``fresh_prompt_ids`` are never sent anywhere for a
filler -- they exist on that dataclass purely to satisfy its structural
invariants (the SAME dataclass and pairwise-isolation infrastructure the
setting's own genuine repair workload uses), unused dead weight for a
filler specifically. Only ``target_prompt_ids``/``target_head_ids``
matter for a plain-dense filler.

Every filler's own target head is dense-seeded exactly like the round's
own head, and all of them (N fillers plus that round's own head) coexist
in the exact radix tree within the same flush epoch -- unlike the
main-vs-sweep case, the per-round flush does *not* isolate them from
each other WITHIN that same round. ``build_eviction_pressure_workloads``
gives each filler a mutually distinct target-head literal-prefix marker
(a different leading Unicode code point drawn from a real-tokenizer-
validated pool, never a decimal index nor a bare fixed-width letter code
-- see ``_pressure_filler_head_literal_prefix``) to keep them pairwise
zero-common-prefix, and ``validate_pairwise_head_isolation`` is a
runtime safety net that checks the actual resulting token-id sequences
(never a textual heuristic alone) and raises immediately if any two
still collide.

There is no up-front floor check against a nominal fraction (the earlier
``--eviction-pressure-min-fraction``/``validate_eviction_pressure_fraction``
design): a ``target_rho`` value ``> 1`` (the entire ``--target-rho-choices``
default set except ``0.9``) still guarantees, by construction, that
fillers alone nominally request MORE tokens than the pool's TRUE
evictable headroom (``capacity_tokens - already_pinned_tokens``, not
merely raw ``capacity_tokens`` -- see
``eviction_pressure_filler_count_for_rho``'s own docstring for the
proof this reduction preserves), guaranteeing genuine eviction pressure
by construction rather than by a separate threshold check -- and, since
fillers are now genuinely evictable plain dense objects,
``register_eviction_pressure_objects`` itself now RAISES immediately if
that construction holds (nominal filler tokens exceed the TRUE evictable
headroom) yet the live ``sglang:evicted_tokens_total`` counter failed to
move while registering them (see that function's own docstring): this
is no longer merely reported, it is enforced. The honest evidence that
real device-pool eviction actually occurred is each setting's own
``pressure_phase.evicted_tokens_total_delta`` /
``pressure_and_target_evicted_tokens_total_delta`` /
``peak_rho_observed`` in the output JSON (the genuine
``sglang:evicted_tokens_total`` Prometheus counter delta and the genuine
sampled resident-occupancy ratio -- see ``observed_rho``,
``sglang:kv_used_tokens`` PLUS ``sglang:kv_evictable_tokens`` against a
fixed capacity, never ``kv_used_tokens`` alone -- incremented/updated by
real LRU eviction and real device-pool occupancy, GPU-only tier included,
not merely GPU-to-CPU host-backup moves) -- reported exactly as observed,
never inferred or assumed from the nominal ``target_rho`` alone. A
nonzero ``sglang:approx_kv_dense_fallback_total`` delta during the
pressure phase itself also raises immediately (see
``register_eviction_pressure_objects``): a plain dense filler request
carries no ``approx_kv`` metadata at all and should never be able to
move that CacheTune-reuse-specific counter in the first place.

Every invocation writes JSONL lifecycle records (``running`` /
``completed`` / ``failed``) to ``--central-log``, carrying the full
settings, the image/model/git identity, the warmup/repeat counts, the
output path, and (on success) a short result summary -- see
``append_run_log``.

Every reported number is a genuine client-observed streaming TTFT or a
real server-reported signal (Prometheus counter delta or per-request
``meta_info``); nothing is fabricated, and the "fresh" preparation cost
and the one-time seed-head/register-raw setup costs are always reported
(see ``known_limitations`` for why the latter two are excluded from
``combined_ms``/``combined_p50_ms``, analogous to why the raw
registration step was never folded in) -- never silently excluded before
claiming an end-to-end result (see ``research/cacheblend``'s and
``research/epic-legolink``'s own Phase 4 results for the established
honest-reporting precedent this script follows; the
central-log/warmup/repeat discipline mirrors
``research/epic-legolink``'s ``run_phase4_epic_inrequest_matrix.py``).

The controller's own per-request decision (ratio, selected tokens,
recomputed layers, precomputed-adapter usage) is *not* exposed in the
``/generate`` JSON response body -- it is only observable in aggregate via
the ``/metrics`` Prometheus endpoint. This script always cross-checks the
*observed* telemetry deltas against an independently computed
expectation, using this same package's real ``roofline_ratio``/
``quantize_ratio``/``predict_ttft_ms`` functions (imported directly, not
reimplemented) applied to the exact ``t_c``/``t_i``/``t_o`` measurement
and mode the operator also used to start the server -- this is real
white-box cross-validation of the running server's behaviour, not a
tautology, since it would catch any mismatch between what the server
actually does and what the controller contract promises.
"""

import argparse
import asyncio
import json
import math
import statistics
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import aiohttp

from benchmark.approx_kv.metrics import (
    idle_pool_invariant,
    parse_prometheus_text,
    usable_kv_capacity_tokens,
)
from benchmark.approx_kv.workloads import deterministic_code
from sglang.srt.mem_cache.cachetune.hardware_profile import (
    CacheTuneMode,
    HardwareMeasurement,
    RatioBounds,
    predict_ttft_ms,
    quantize_ratio,
    roofline_ratio,
)

CACHE_SALT = "phase4-r5-cachetune"

# Fixed default head length for eviction-pressure filler objects only
# (see --pressure-filler-head-tokens): the main setting and every
# shape-sweep point instead use --main-header-tokens/--header-tokens-
# choices, which now sweep header length as its own dimension of the
# unified header x body x rho matrix (superseding this script's earlier
# "target head固定34即可" fixed-head convention).
NON_PREFIX_HEAD_TOKENS = 34
# Exactly one token must remain outside every restorable segment so each
# reuse request still performs one genuine forward pass beyond the
# repaired range (see ApproxKVRequestMetadata.validate_prompt_length).
NON_PREFIX_TAIL_TOKENS = 1

# Appended, as an explicit extra token past target_head_ids, to every
# seed/re-seed dense request for a workload's own target head (see
# NonPrefixSegmentWorkload.seed_prompt_ids). THIS IS A DELIBERATE FIX
# for a real SM75 bug observed on a header=32 sweep point: a bare
# `target_head_ids` seed request (dense_generate_payload, max_new_
# tokens=1, temperature=0) deterministically generates one token that
# can coincidentally EQUAL shared_body_ids[0]; once that single extra
# token is inserted into the exact radix tree (plain dense requests
# never set skip_radix_cache_insert), the tree's exact-match boundary
# for this head silently extends by one token, so a LATER request whose
# own prompt is target_head_ids + shared_body_ids + ... (the fresh
# register call, or the reuse call) reports one MORE token cached than
# the header's own true length -- observed as fresh-register
# cached_tokens=33 when body_start_in_target=32. Appending an explicit
# sentinel token here -- chosen, per workload, to differ from that SAME
# workload's own shared_body_ids[0] (see
# _build_seed_sentinel_ids_avoiding_body_first_token_collision) --
# anchors the tree's exact-match boundary at a FIXED, KNOWN,
# non-body-colliding token, so any later request matching
# target_head_ids + shared_body_ids + ... always diverges EXACTLY at
# len(target_head_ids), regardless of what the seed request's own
# single generated token happens to be.
NON_PREFIX_SEED_SENTINEL_TOKENS = 1
# Fixed SUFFIX text only -- the actual per-attempt CANDIDATE marker is
# this string prepended by a distinct leading Unicode code point (see
# _build_seed_sentinel_ids_avoiding_body_first_token_collision, which
# reuses _pressure_filler_marker_codepoint_for_combined_index for that
# leading character, exactly like _pressure_filler_head_literal_prefix
# does). This fixed suffix ALONE, without a varying leading character,
# would NOT let retries reach a different first token: _deterministic_
# token_ids's literal_prefix is prepended before the seed-dependent
# digest text, so a real (or word-granularity fake) tokenizer's FIRST
# token is determined entirely by this literal text's own leading
# word/subword, identically on every attempt, regardless of how many
# different `seed` values are tried against it alone.
_SEED_SENTINEL_LITERAL_PREFIX = "SEED_SENTINEL_MARKER_TEXT\n"
_MAX_SEED_SENTINEL_COLLISION_RETRIES = 64

# Prepended verbatim (never hashed) ahead of each head piece's generated
# text via ``_deterministic_token_ids``'s ``literal_prefix`` parameter.
# ``workloads.deterministic_code``'s first block always begins with the
# literal, seed-*independent* text "def synthetic_0_" before any
# seed-dependent digest character appears, so two differently-seeded head
# pieces would otherwise reliably tokenize to several *identical* leading
# token ids regardless of seed (empirically confirmed with Qwen3-0.6B's
# real tokenizer: a 4-6 token shared prefix) -- these two markers diverge
# at their very first character ('S' vs 'T'), forcing divergence starting
# at token 0 rather than relying on the seed-dependent digest alone (see
# the module docstring's "Why /generate and non-prefix segments" section).
_SOURCE_HEAD_LITERAL_PREFIX = "SOURCE_HEAD_MARKER_TEXT\n"
_TARGET_HEAD_LITERAL_PREFIX = "TARGET_HEAD_MARKER_TEXT\n"

# Unicode code-point ranges sampled by _pressure_filler_head_literal_prefix
# to build every eviction-pressure filler object's own *target* head
# marker (see build_eviction_pressure_workloads). Every filler's target
# head is dense-seeded (to populate the exact radix tree, exactly like
# the main/sweep setting's own head -- see materialize_workload_via_
# reuse), and all N fillers plus the setting's own head coexist in the
# same tree within one flush epoch (run_non_prefix_setting flushes once
# per *setting*, not once per filler object), so every one of these
# markers must tokenize to a first token id that is pairwise distinct
# from every other one (validate_pairwise_head_isolation is the runtime
# safety net that actually checks this).
#
# An earlier version of this scheme used a fixed 24-letter ASCII
# alphabet ("A".."Z" minus S/T) with a positional-numeral encoding for
# counts above 24. That was empirically PROVEN WRONG against the real
# Qwen3-0.6B tokenizer: two distinct, equal-length letter codes (e.g.
# "AA" and "AB") can still tokenize to the SAME first token id, because
# BPE merge behavior can fold a shared leading character into a shared
# merge token -- string-level prefix-freeness does not imply token-level
# first-token distinctness. Worse, ANY short "generic-looking" synthetic
# ASCII/hex text (regardless of alphabet, digit count, or length) was
# measured to plateau at only ~183-400 distinct achievable *first
# tokens* under this tokenizer's real BPE vocabulary -- nowhere near
# enough for hundreds/thousands of filler objects.
#
# These particular code-point ranges (CJK Unified Ideographs, Hiragana/
# Katakana, Hangul syllables, and several emoji/symbol blocks) were
# chosen because empirical testing against the real Qwen3-0.6B tokenizer
# showed each contains many individual characters that real-world
# multilingual training corpora represent with their own dedicated
# vocabulary entries -- together reaching thousands of distinct
# achievable first tokens (>=8000 with zero collisions in the retry
# search below), instead of the few-hundred ceiling synthetic ASCII text
# hits. This is still just an empirically-good candidate SOURCE, never
# trusted alone: _build_pressure_filler_workload_avoiding_first_token_
# collisions validates every candidate against the real tokenizer passed
# in and retries on any actual collision, so correctness never depends
# on an assumption about these ranges' behavior under any particular
# tokenizer.
_PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS: tuple[tuple[int, int], ...] = (
    (0x2190, 0x2BFF),  # Arrows, math operators, misc technical, dingbats
    (0x2C00, 0x2DFF),  # Glagolitic, Latin Extended-C, Coptic, Georgian Supp.
    (0x1F000, 0x1F0FF),  # Mahjong / domino / playing cards
    (0x1F100, 0x1F2FF),  # Enclosed alphanumeric / ideographic supplement
    (0x1F300, 0x1F5FF),  # Misc symbols and pictographs
    (0x1F600, 0x1F64F),  # Emoticons
    (0x1F680, 0x1F6FF),  # Transport and map symbols
    (0x1F700, 0x1F77F),  # Alchemical symbols
    (0x1F780, 0x1F7FF),  # Geometric shapes extended
    (0x1F800, 0x1F8FF),  # Supplemental arrows-C
    (0x1F900, 0x1F9FF),  # Supplemental symbols and pictographs
    (0x1FA00, 0x1FA6F),  # Chess symbols
    (0x1FA70, 0x1FAFF),  # Symbols and pictographs extended-A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0xAC00, 0xD7A3),  # Hangul syllables
)

_PRESSURE_FILLER_MARKER_CODEPOINT_POOL_SIZE = sum(
    high - low for low, high in _PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS
)

# Large odd multiplier (Knuth's multiplicative-hash constant) used to
# permute a combined (filler index, retry attempt) integer across the
# code-point pool above. Scanning the pool SEQUENTIALLY (combined_index
# directly as an offset) was empirically found to cluster many adjacent
# code points onto the SAME first token -- BPE merges neighboring code
# points within a block together far more often than distant ones --
# which exhausted the retry budget almost immediately (observed failure
# at the 49th filler object in one such check). This constant is
# coprime with every pool size this module can produce in practice
# (verified for the current pool via math.gcd in this module's tests),
# so multiplying by it and reducing modulo the pool size is a true
# bijection over one full pool-sized cycle of combined_index values --
# consecutive combined indices land on widely-separated pool positions
# instead of adjacent ones, which was verified empirically to let the
# retry search reach 8,000+ distinct fillers with zero exhaustion (see
# tests).
_PRESSURE_FILLER_MARKER_CODEPOINT_STRIDE = 2654435761

# Defensive upper bound on a single setting's reverse-computed filler
# object count (see eviction_pressure_filler_count_for_rho). Each filler
# object costs one real, blocking HTTP round trip (a single plain dense
# /generate call -- see register_eviction_pressure_objects's own
# docstring for why fillers are plain dense requests, never a
# register/reuse cycle) plus an explicit 0.1s sleep in that same
# function, so an unbounded count from a pathological
# (--main-target-rho/--target-rho-choices, --pressure-filler-body-tokens,
# live measured capacity) combination could otherwise silently spend
# hours issuing real requests
# before this setting's own measurement even begins -- this raises
# immediately instead, the moment the count is known, before any of that
# per-filler HTTP loop starts. Deliberately far above any plausible real
# combination this project's SM75/RTX6000-class hardware would ever need
# at the documented default --pressure-filler-body-tokens (2048): even a
# ~374k-token usable capacity (a rough RTX 6000 48GB order-of-magnitude
# estimate for this project's small Qwen3-0.6B model) at target_rho=3
# only reverse-computes to roughly 540 fillers, well under this bound.
MAX_REASONABLE_EVICTION_PRESSURE_FILLER_COUNT = 5000

# Every setting (dense, the main CacheTune point, every shape-sweep
# point, and every rho-sweep point) runs exactly this many *discarded*
# passes before the formal repeats begin. This is a fixed measurement-
# protocol constant, not a CLI knob, so every canary result is comparable
# under the same discipline.
WARMUP_PASSES_PER_SETTING = 1

# A tiny, fixed dense request posted (see
# flush_and_force_gauge_refresh) every time this script needs a
# genuinely fresh /metrics reading immediately after a /flush_cache
# call -- both at the very end of the run
# (capture_final_pool_reset_and_invariant) AND at the START of every
# independent round (run_independent_round) -- purely to force one real
# scheduler iteration so gauges such as sglang:kv_used_tokens are
# recomputed from the actually-idle pool rather than read stale
# immediately after the flush call returns. Small on purpose: its only
# job is to trigger that recompute, never to exercise CacheTune
# telemetry.
_POOL_RESET_SENTINEL_TOKENS = 4
_POOL_RESET_SENTINEL_SEED = f"{CACHE_SALT}-final-pool-reset-sentinel"


def _pressure_filler_marker_codepoint_for_combined_index(combined_index: int) -> int:
    """Map a combined ``(filler_index, retry_attempt)`` integer to one
    Unicode code point drawn from
    ``_PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS``.

    Every ``combined_index`` in ``[0, _PRESSURE_FILLER_MARKER_CODEPOINT_
    POOL_SIZE)`` maps to a DISTINCT code point (multiplying by
    ``_PRESSURE_FILLER_MARKER_CODEPOINT_STRIDE`` and reducing modulo the
    pool size is a bijection over exactly one pool-sized cycle, since the
    stride is coprime with the pool size -- see that constant's own
    comment and this module's tests); ``combined_index`` values are
    reduced modulo the pool size first, so the mapping is total (defined
    for any non-negative integer) and simply repeats every pool-sized
    cycle rather than raising once the immediate pool is exhausted --
    ``_build_pressure_filler_workload_avoiding_first_token_collisions``'s
    own bounded retry budget, not this function, is what actually caps
    how many attempts get made in practice.
    """
    if combined_index < 0:
        raise ValueError(f"combined_index must be >= 0, got {combined_index}")
    offset = combined_index % _PRESSURE_FILLER_MARKER_CODEPOINT_POOL_SIZE
    permuted_offset = (
        offset * _PRESSURE_FILLER_MARKER_CODEPOINT_STRIDE
    ) % _PRESSURE_FILLER_MARKER_CODEPOINT_POOL_SIZE
    for low, high in _PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS:
        span = high - low
        if permuted_offset < span:
            return low + permuted_offset
        permuted_offset -= span
    raise AssertionError(
        "unreachable: _PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS span "
        "accounting does not match _PRESSURE_FILLER_MARKER_CODEPOINT_"
        "POOL_SIZE -- this is a bug in this module's own constants, not "
        "a caller error"
    )


def _pressure_filler_head_literal_prefix(combined_index: int) -> str:
    """Deterministic literal marker TEXT for one CANDIDATE attempt (a
    combined ``(filler_index, retry_attempt)`` index -- see
    ``_build_pressure_filler_workload_avoiding_first_token_collisions``)
    of an eviction-pressure filler object's own *target* head (see
    ``_PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS``,
    ``build_eviction_pressure_workloads``).

    Every distinct ``combined_index`` (within one pool-sized cycle, see
    ``_pressure_filler_marker_codepoint_for_combined_index``) gets its
    own distinct leading Unicode code point, chosen from ranges (CJK
    ideographs, Hiragana/Katakana, Hangul syllables, common emoji/symbol
    blocks) empirically found to each tokenize to many distinct first
    tokens under the real Qwen3-0.6B tokenizer -- unlike short synthetic
    ASCII/hex text (this project's original scheme), which plateaus at
    only ~183-400 distinct achievable first tokens regardless of
    alphabet or length (see ``_PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS``'s
    own comment for the full empirical history).

    IMPORTANT -- exactly like the original ASCII scheme this replaces,
    a distinct leading code point is still only an empirically-good
    STARTING POINT, never a proof, of first-TOKEN distinctness on its
    own: ``build_eviction_pressure_workloads`` (via
    ``_build_pressure_filler_workload_avoiding_first_token_collisions``)
    is what actually guarantees the required token-level zero-common-
    prefix property, by validating each candidate's real
    ``target_head_ids[0]`` against the real tokenizer passed in and
    retrying with the NEXT ``combined_index`` on any detected collision.
    This function must never be trusted alone to guarantee pairwise
    isolation; only that downstream real-tokenizer validation may be.
    """
    codepoint = _pressure_filler_marker_codepoint_for_combined_index(combined_index)
    return chr(codepoint) + "-pressure-filler-marker\n"


def _repeat_count(value: str) -> int:
    """argparse ``type=`` validator: reject ``--repeats`` below 2 up front.

    A single formal repeat cannot be distinguished from measurement noise.
    The entire point of separating "formal repeats" from the discarded
    warmup pass is to give ``statistics.median`` more than one real
    sample, so this is enforced as a hard CLI-level error rather than a
    silently-clamped default.
    """
    repeats = int(value)
    if repeats < 2:
        raise argparse.ArgumentTypeError(
            f"--repeats must be >= 2, got {repeats} (need at least two "
            "formal measurements to compute a meaningful median and to "
            "distinguish real signal from single-sample noise)"
        )
    return repeats


def _non_negative_int_choice_list(value: str) -> tuple[int, ...]:
    """argparse ``type=`` validator for ``--header-tokens-choices``: parse
    a comma-separated list of non-negative integers. Rejects an empty
    list and any negative entry up front. 0 is explicitly valid (the
    header sweep's exact-context control point, see
    ``run_exact_context_control_point``)."""
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError(
            f"expected a non-empty comma-separated list, got {value!r}"
        )
    for item in values:
        if item < 0:
            raise argparse.ArgumentTypeError(
                f"every value must be >= 0, got {item} in {value!r}"
            )
    return values


def _positive_int_choice_list(value: str) -> tuple[int, ...]:
    """argparse ``type=`` validator for ``--body-tokens-choices``: parse a
    comma-separated list of positive integers. Rejects an empty list and
    any non-positive entry up front."""
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError(
            f"expected a non-empty comma-separated list, got {value!r}"
        )
    for item in values:
        if item <= 0:
            raise argparse.ArgumentTypeError(
                f"every value must be positive, got {item} in {value!r}"
            )
    return values


def _positive_float_choice_list(value: str) -> tuple[float, ...]:
    """argparse ``type=`` validator for ``--target-rho-choices``: parse a
    comma-separated list of positive floats. Rejects an empty list and
    any non-positive entry up front."""
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError(
            f"expected a non-empty comma-separated list, got {value!r}"
        )
    for item in values:
        if item <= 0:
            raise argparse.ArgumentTypeError(
                f"every value must be positive, got {item} in {value!r}"
            )
    return values


def _positive_int(value: str) -> int:
    """argparse ``type=`` validator: reject non-positive integers up
    front (used by ``--main-header-tokens``, ``--main-body-tokens``,
    ``--max-segment-chunk-tokens``, ``--pressure-filler-head-tokens``,
    ``--pressure-filler-body-tokens``)."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {parsed}")
    return parsed


def _positive_float(value: str) -> float:
    """argparse ``type=`` validator: reject non-positive floats up front
    (used by ``--main-target-rho``, ``--length-sweep-rho``)."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {parsed}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--cache-dtype", default="fp16")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[mode.value for mode in CacheTuneMode],
        help="Must equal the running server's SGLANG_CACHETUNE_MODE.",
    )
    parser.add_argument(
        "--t-c-ms",
        type=float,
        required=True,
        help="Must equal the running server's SGLANG_CACHETUNE_T_C_MS.",
    )
    parser.add_argument(
        "--t-i-ms",
        type=float,
        required=True,
        help="Must equal the running server's SGLANG_CACHETUNE_T_I_MS.",
    )
    parser.add_argument(
        "--t-o-ms",
        type=float,
        required=True,
        help="Must equal the running server's SGLANG_CACHETUNE_T_O_MS.",
    )
    parser.add_argument(
        "--first-recompute-layer",
        type=int,
        default=1,
        help="Must equal the running server's SGLANG_CACHETUNE_FIRST_RECOMPUTE_LAYER.",
    )
    parser.add_argument(
        "--main-header-tokens",
        type=_positive_int,
        default=64,
        help="Distinct-head token count (source_head_ids/target_head_ids "
        "length) for the main setting's NonPrefixSegmentWorkload. Must be "
        "positive -- header=0 is only meaningful as a shape-sweep exact-"
        "context control point (see --header-tokens-choices), never as "
        "the main setting's own shape.",
    )
    parser.add_argument(
        "--main-body-tokens",
        type=_positive_int,
        default=1024,
        help="Shared-body token count for the main setting's "
        "NonPrefixSegmentWorkload. Bodies longer than "
        "--max-segment-chunk-tokens are registered as one independent "
        "register call per <= --max-segment-chunk-tokens chunk (never "
        "within one oversized call -- see register_body_chunks) and "
        "reused as multiple <= --max-segment-chunk-tokens segments "
        "within a single reuse call (see body_segments_for_hash).",
    )
    parser.add_argument(
        "--main-target-rho",
        type=_positive_float,
        default=1.5,
        help="Target eviction-pressure ratio (fraction of the server's "
        "real, live usable_kv_capacity_tokens, see "
        "benchmark.approx_kv.metrics) for the main setting's own "
        "pre-target filler phase; the filler object count is reverse-"
        "computed from this against a real /metrics snapshot taken "
        "immediately after this setting's own flush (see "
        "eviction_pressure_filler_count_for_rho). A value > 1 means the "
        "fillers alone nominally request MORE tokens than the whole "
        "pool's measured capacity, guaranteeing genuine eviction "
        "pressure by construction.",
    )
    parser.add_argument(
        "--header-tokens-choices",
        type=_non_negative_int_choice_list,
        default="0,32,64,128,256",
        help="Comma-separated header (distinct source/target head) token "
        "counts swept, crossed with --body-tokens-choices, as the shape "
        "sweep (replaces every earlier fixed 34-token head / 128,256,512 "
        "body-only length-sweep default). header=0 cannot build a "
        "NonPrefixSegmentWorkload (source_head_ids/target_head_ids "
        "cannot differ if both are empty) so it is handled as a distinct, "
        "honestly-labeled exact-context control point instead -- see "
        "run_exact_context_control_point.",
    )
    parser.add_argument(
        "--body-tokens-choices",
        type=_positive_int_choice_list,
        default="512,768,1024,2048",
        help="Comma-separated shared-body token counts swept, crossed "
        "with --header-tokens-choices, as the shape sweep.",
    )
    parser.add_argument(
        "--target-rho-choices",
        type=_positive_float_choice_list,
        default="0.9,1.1,1.5,2,3",
        help="Comma-separated target eviction-pressure ratios swept, at "
        "the main setting's own (--main-header-tokens, --main-body-"
        "tokens) shape, as the rho sweep -- reported separately from the "
        "shape sweep so the two dimensions each stay a tractable number "
        "of real requests rather than a full combinatorial explosion.",
    )
    parser.add_argument(
        "--length-sweep-rho",
        type=_positive_float,
        default=None,
        help="Fixed target rho applied to every shape-sweep point "
        "(header x body cross product from --header-tokens-choices x "
        "--body-tokens-choices). Defaults to --main-target-rho when "
        "omitted.",
    )
    parser.add_argument(
        "--max-segment-chunk-tokens",
        type=_positive_int,
        default=512,
        help="Maximum token length of any single approx_kv segment this "
        "canary registers or reuses. Bodies longer than this are split "
        "into multiple <= this length segments (distinct content_hash "
        "per chunk) -- registered as one independent /generate call per "
        "chunk (see register_body_chunks; keeps each register call's own "
        "transient KV footprint bounded, avoiding a real SM75 OOM a "
        "single oversized multi-segment register call previously "
        "caused) but still reused within a single call's multiple "
        "contiguous segments (see body_segments_for_hash/chunk_offsets; "
        "ApproxKVRequestMetadata/register_request_segments/"
        "restore_request_prefix_cachetune natively support an arbitrary "
        "number of segments per call, so the reuse side's own multi-"
        "segment single call remains unaffected).",
    )
    parser.add_argument(
        "--pressure-filler-head-tokens",
        type=_positive_int,
        default=NON_PREFIX_HEAD_TOKENS,
        help="Head token count for every eviction-pressure filler object "
        "-- independent of --main-header-tokens/--header-tokens-choices, "
        "since fillers always need a genuine, fixed, non-zero distinct "
        "head to stay pairwise head-isolated regardless of what header "
        "value the setting under test is exercising (see "
        "_pressure_filler_head_literal_prefix / "
        "validate_pairwise_head_isolation).",
    )
    parser.add_argument(
        "--pressure-filler-body-tokens",
        type=_positive_int,
        default=2048,
        help="Shared-body token count for EACH eviction-pressure filler "
        "object, used consistently across the main setting, every shape-"
        "sweep point, and every rho-sweep point so peak-rho/eviction "
        "numbers stay comparable across points (only the reverse-"
        "computed filler COUNT varies with --main-target-rho/"
        "--target-rho-choices and the live measured capacity, never this "
        "per-object shape).",
    )
    parser.add_argument("--repeats", type=_repeat_count, default=4)
    parser.add_argument("--runner-git-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--central-log",
        type=Path,
        required=True,
        help="Shared JSONL log every test/benchmark run must append to: "
        "one 'running' record at start, then one 'completed' or 'failed' "
        "record at the end (see append_run_log).",
    )
    args = parser.parse_args()
    if args.length_sweep_rho is None:
        args.length_sweep_rho = args.main_target_rho
    return args


def fetch_text(url: str, timeout: float = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def post_empty(url: str, timeout: float = 60) -> str:
    """POST with an empty body.

    Matches the exact idiom already established by this directory's
    ``run_phase3_canary.py`` and ``run_phase2_matrix.py`` for
    ``/flush_cache`` -- any non-2xx response raises ``urllib.error.HTTPError``
    unhandled, which is intentional: a silently-ignored flush failure
    would silently reintroduce the exact-cache pollution this script
    exists to prevent.
    """
    request = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def flush_exact_radix_cache(base_url: str) -> str:
    """Flush the server's exact-match radix cache.

    Only *dense* baseline requests (no ``approx_kv`` metadata) are ever
    inserted into the exact radix tree: ``schedule_batch.Req``'s
    ``skip_radix_cache_insert`` is forced True whenever
    ``approx_kv_metadata`` is present -- i.e. for *every* register or
    reuse request (see ``python/sglang/srt/managers/schedule_batch.py``).
    That means register/reuse requests can never pollute the exact cache
    themselves, but a real dense forward over the same token sequence a
    later ``reuse`` request targets absolutely can: it would let the
    scheduler's own prefix match resolve the entire prompt before
    CacheTune's plugin dispatch ever runs, silently skipping the whole
    approximate-repair path. Call this before every dense repeat (each is
    a real, cache-writing forward pass) and, for CacheTune's own
    non-prefix-segment measurement, as the very first action of every
    single independent round (both the discarded warmup round and every
    formal repeat -- see ``run_independent_round``'s own docstring),
    never merely once per setting.

    This DOES also invoke ``ApproxKVManager.reset()`` (see
    ``python/sglang/srt/mem_cache/approx_kv/manager.py``), which wipes
    any already-registered "raw"/"fresh" segment store entries. An
    earlier design relied on ONE raw registration surviving across every
    formal repeat and therefore forbade flushing between repeats; that
    earlier design is what produced a real SM75 ``target_rho=2``
    ``MemoryError`` (see ``run_independent_round``'s own docstring for
    the full root cause). The current design instead makes every round
    -- including warmup -- fully independent: each one re-registers its
    own raw AND fresh from scratch immediately after this exact flush,
    so there is nothing left for this flush to wipe out from under a
    later step within the SAME round, and flushing before every round is
    now MANDATORY rather than forbidden.
    """
    response = post_empty(f"{base_url}/flush_cache?timeout=30")
    time.sleep(0.1)
    return response


def _deterministic_token_ids(
    tokenizer: Any, seed: str, count: int, *, literal_prefix: str = ""
) -> tuple[int, ...]:
    """Deterministically produce *exactly* ``count`` token ids from
    ``seed`` via ``workloads.deterministic_code``.

    Tokenizes the generated synthetic-code text with the same
    ``tokenizer.encode(text, add_special_tokens=False)`` convention
    already established by ``workloads.build_object_catalog`` (callers
    must concatenate the *integer id* lists this function returns, not
    the underlying text -- re-tokenizing a joined string could merge
    tokens across a piece boundary and silently break exact offsets).
    ``deterministic_code`` grows in fixed-size blocks, so this doubles
    the requested block count and retries whenever the first attempt
    tokenizes shorter than ``count``, up to a generous, finite bound --
    never a silent/partial result and never an unbounded loop.

    ``literal_prefix``, when non-empty, is prepended verbatim (as plain
    text, never hashed into the seed) ahead of the generated
    synthetic-code text on every attempt. This exists exclusively so
    ``build_non_prefix_segment_workload`` can force its two head pieces
    to diverge starting at their very first token: ``deterministic_code``
    itself always begins with the literal, seed-*independent* text
    ``"def synthetic_0_"`` before any seed-dependent digest character
    appears, so relying on ``seed`` alone would let two differently-seeded
    calls tokenize to several *identical* leading token ids (see
    ``_SOURCE_HEAD_LITERAL_PREFIX``/``_TARGET_HEAD_LITERAL_PREFIX``).
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    blocks = max(4, count // 3 + 4)
    for _ in range(20):
        text = literal_prefix + deterministic_code(seed, blocks)
        token_ids = [
            int(token_id)
            for token_id in tokenizer.encode(text, add_special_tokens=False)
        ]
        if len(token_ids) >= count:
            return tuple(token_ids[:count])
        blocks *= 2
    raise RuntimeError(
        f"could not tokenize {count} deterministic tokens for seed={seed!r} "
        f"after growing to blocks={blocks}"
    )


@dataclass(frozen=True)
class NonPrefixSegmentWorkload:
    """A genuine non-prefix cross-context repair workload.

    ``source_prompt_ids = source_head_ids + shared_body_ids + tail_ids``
    is the prompt CacheTune's "raw" segment is registered from (captures
    the body's KV under *source* context); ``target_prompt_ids =
    target_head_ids + shared_body_ids + tail_ids`` is the prompt the
    reuse request actually targets; ``fresh_prompt_ids`` is the prompt
    the "fresh" segment is registered from and is token-identical to
    ``target_prompt_ids`` (same head, body, AND tail -- see that
    property's own docstring for why that equality is intentional and
    safe). Every one of these three prompts appends the SAME trailing
    ``tail_ids`` -- never just ``head + body`` -- so that every
    register/reuse call's own body segment (which still only spans
    ``[body_start, body_start + body_tokens)``, never touching the
    tail) leaves at least that prompt's own final token for a real
    forward pass, satisfying ``ApproxKVRequestMetadata.
    validate_prompt_length``'s "approximate KV segments must leave the
    final prompt token for a real forward pass" invariant (see
    ``sglang.srt.mem_cache.approx_kv.request``). An earlier version of
    this dataclass omitted ``tail_ids`` from ``source_prompt_ids``/
    ``fresh_prompt_ids`` (only ``target_prompt_ids`` had it): that made
    the raw/fresh register call's own segment ``target_end`` land
    EXACTLY at ``len(prompt)``, tripping that same check inside
    ``Req.__init__`` (``schedule_batch.py``) synchronously on the
    scheduler's own request-admission path and killing the scheduler on
    a real SM75 run -- this is a real, previously-observed production
    bug, not a hypothetical one. ``source_head_ids`` and
    ``target_head_ids`` are required to differ: this is what makes the
    body's source and target KV genuinely distinct (not just an
    exact-content replay of the same context) -- see the module
    docstring's "Why /generate and non-prefix segments" section for the
    full rationale.

    ``seed_sentinel_ids`` (see ``seed_prompt_ids``) is an explicit extra
    token appended, only for the plain-dense head-seed/re-seed requests
    (``register_round_setup``/``ensure_target_head_resident``), past
    ``target_head_ids`` -- never part of any register/reuse segment or
    of ``target_prompt_ids``/``fresh_prompt_ids`` themselves. This is a
    REQUIRED, validated-distinct-from-``shared_body_ids[0]`` field, not
    an afterthought: a bare ``target_head_ids``-only seed request's own
    single generated token (``max_new_tokens=1``, ``temperature=0``,
    fully deterministic) can coincidentally equal ``shared_body_ids[0]``,
    which would silently extend the exact radix tree's matched boundary
    for this head by one token and corrupt every later request's own
    ``cached_tokens`` count against this workload (a real SM75 bug,
    observed on a header=32 sweep point as fresh-register
    ``cached_tokens=33`` instead of the true header length 32) -- see
    ``seed_prompt_ids``'s own docstring for the full mechanism.
    """

    source_head_ids: tuple[int, ...]
    target_head_ids: tuple[int, ...]
    shared_body_ids: tuple[int, ...]
    tail_ids: tuple[int, ...]
    seed_sentinel_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.source_head_ids:
            raise ValueError("source_head_ids must not be empty")
        if not self.target_head_ids:
            raise ValueError("target_head_ids must not be empty")
        if not self.shared_body_ids:
            raise ValueError("shared_body_ids must not be empty")
        if not self.tail_ids:
            raise ValueError("tail_ids must not be empty")
        if not self.seed_sentinel_ids:
            raise ValueError("seed_sentinel_ids must not be empty")
        if self.source_head_ids == self.target_head_ids:
            raise ValueError(
                "source_head_ids and target_head_ids must differ -- an "
                "identical head makes the body's source and target KV "
                "indistinguishable (exact-content transplant, not a real "
                "cross-context repair; see module docstring)"
            )
        common_head_prefix = 0
        for source_token, target_token in zip(
            self.source_head_ids, self.target_head_ids
        ):
            if source_token != target_token:
                break
            common_head_prefix += 1
        if common_head_prefix > 0:
            raise ValueError(
                "source_head_ids and target_head_ids share a "
                f"{common_head_prefix}-token common exact-match prefix -- "
                "being merely unequal overall is not enough: a live "
                "server's exact radix tree would still report a nonzero "
                "cached_tokens for the raw-segment register request "
                "immediately after target_head_ids is seeded, corrupting "
                "the canary's 'must be exactly 0' measurement invariant "
                "(see build_non_prefix_segment_workload's literal-prefix "
                "disambiguation, which this should never be able to reach "
                "if it is doing its job)"
            )
        if self.seed_sentinel_ids[0] == self.shared_body_ids[0]:
            raise ValueError(
                "seed_sentinel_ids[0] must differ from shared_body_ids[0] "
                "-- an identical value would let the target-head seed "
                "request's own exact-match tree entry (seed_prompt_ids = "
                "target_head_ids + seed_sentinel_ids) spuriously extend "
                "into the body on any later request matching "
                "target_head_ids + shared_body_ids + ..., corrupting that "
                "request's own cached_tokens by exactly one extra "
                "(falsely 'exact') token -- this is the real SM75 "
                "header-sweep bug seed_prompt_ids exists to prevent (see "
                "build_non_prefix_segment_workload's "
                "_build_seed_sentinel_ids_avoiding_body_first_token_"
                "collision, which this should never be able to reach if "
                "it is doing its job)"
            )

    @property
    def body_tokens(self) -> int:
        return len(self.shared_body_ids)

    @property
    def body_start_in_source(self) -> int:
        return len(self.source_head_ids)

    @property
    def body_start_in_target(self) -> int:
        return len(self.target_head_ids)

    @property
    def source_prompt_ids(self) -> tuple[int, ...]:
        """``source_head_ids + shared_body_ids + tail_ids``: the prompt
        the "raw" (source-context) segment is registered from. The
        registered segment itself still only spans
        ``[body_start_in_source, body_start_in_source + body_tokens)``
        -- ``tail_ids`` is appended solely so that span never reaches
        this prompt's own final token index (see this class's own
        docstring for why, and the real production bug this fixes)."""
        return self.source_head_ids + self.shared_body_ids + self.tail_ids

    @property
    def target_prompt_ids(self) -> tuple[int, ...]:
        return self.target_head_ids + self.shared_body_ids + self.tail_ids

    @property
    def fresh_prompt_ids(self) -> tuple[int, ...]:
        """The prompt the "fresh" segment is registered from for the
        precomputed fresh-KV adapter (see ``cachetune/precomputed.py``):
        ``target_head_ids + shared_body_ids + tail_ids`` -- token-
        identical to ``target_prompt_ids``. The fresh registration must
        capture the body's real KV under the exact same context the
        reuse request will later target, including the same trailing
        ``tail_ids``, for the same "leave the final prompt token for a
        real forward pass" reason described on ``source_prompt_ids``
        (omitting it here, as an earlier version of this property did,
        is what actually caused the real scheduler-killing bug this
        class's docstring describes). This equality is intentional and
        safe, not an accidental duplication: every request carrying
        ``approx_kv_metadata`` (register AND reuse alike) sets
        ``skip_radix_cache_insert = True`` (see
        ``sglang.srt.managers.schedule_batch.Req.__init__``), so this
        fresh-register call never populates the live server's exact
        radix tree -- a later, token-identical reuse request cannot get
        an unwanted full exact-prefix hit from it."""
        return self.target_prompt_ids

    @property
    def seed_prompt_ids(self) -> tuple[int, ...]:
        """``target_head_ids + seed_sentinel_ids``: the prompt the
        plain-dense head-seed/re-seed requests
        (``register_round_setup``/``ensure_target_head_resident``) post
        to populate the exact radix tree for this workload's own target
        head -- never ``target_head_ids`` alone.

        THIS IS A DELIBERATE FIX for a real SM75 bug observed on a
        header=32 sweep point: seeding with a bare ``target_head_ids``
        prompt (``max_new_tokens=1``, ``temperature=0``) lets the
        server's own single, fully-deterministic generated token become
        part of the exact radix tree's stored path for this head (plain
        dense requests never set ``skip_radix_cache_insert``); if that
        generated token happens to equal ``shared_body_ids[0]`` (nothing
        prevents this -- it depends only on the real model's own greedy
        decoding output for this specific head), the tree's exact-match
        boundary for ``target_head_ids`` silently extends by one token,
        so ANY later request whose own prompt is ``target_head_ids +
        shared_body_ids + ...`` (the fresh-register call, or the reuse
        call) reports ``cached_tokens`` one token higher than this
        header's own true length -- observed in production as a
        fresh-register ``cached_tokens=33`` when
        ``body_start_in_target=32``.

        Appending ``seed_sentinel_ids`` -- validated, per workload (see
        ``__post_init__``), to differ from this SAME workload's own
        ``shared_body_ids[0]`` -- anchors the tree's exact-match
        boundary at a FIXED, KNOWN, non-body-colliding token instead:
        any later request matching ``target_head_ids + shared_body_ids
        + ...`` now always diverges EXACTLY at ``len(target_head_ids)``
        (the query's own token there is ``shared_body_ids[0]``, the
        tree's stored branch there is ``seed_sentinel_ids[0]``, and
        those are guaranteed unequal), regardless of what the seed
        request's own single generated token turns out to be -- the
        match never even reaches that generated-token node, since the
        walk already diverged one position earlier. This holds
        regardless of how much of ``seed_prompt_ids`` a later re-seed
        request itself finds already resident (0, ``len(target_head_ids)``,
        or the full ``len(seed_prompt_ids)`` -- see
        ``ensure_target_head_resident``'s own docstring for why all
        three are legitimate outcomes), since the invariant this
        property protects is anchored in the PROMPT content itself, not
        in whichever of those outcomes actually occurs on a given call.
        """
        return self.target_head_ids + self.seed_sentinel_ids

    @property
    def body_source_context_differs_from_target(self) -> bool:
        """Always ``True`` once constructed (``__post_init__`` already
        enforces ``source_head_ids != target_head_ids``); exposed as an
        explicit, self-documenting fact for the output JSON rather than
        re-deriving the comparison at every call site."""
        return self.source_head_ids != self.target_head_ids


def _build_seed_sentinel_ids_avoiding_body_first_token_collision(
    tokenizer: Any,
    *,
    salt: str,
    shared_body_ids: tuple[int, ...],
) -> tuple[int, ...]:
    """Build this workload's ``seed_sentinel_ids`` (see
    ``NonPrefixSegmentWorkload.seed_prompt_ids``), trying up to
    ``_MAX_SEED_SENTINEL_COLLISION_RETRIES`` distinct candidate marker
    prefixes -- each ``chr(codepoint) + _SEED_SENTINEL_LITERAL_PREFIX``,
    reusing ``_pressure_filler_marker_codepoint_for_combined_index`` for
    a per-attempt DISTINCT leading Unicode code point, exactly the way
    ``_pressure_filler_head_literal_prefix`` already does for pressure-
    filler target-head markers -- until one candidate's first token id
    differs from ``shared_body_ids[0]``.

    Varying only the ``seed`` string passed to ``_deterministic_token_
    ids`` (while keeping ``literal_prefix`` fixed across attempts) would
    NOT work here: that function always prepends ``literal_prefix``
    verbatim, ahead of the seed-dependent digest text, so a real (or
    word-granularity fake) tokenizer's FIRST token is determined
    entirely by the fixed literal text's own leading word/subword,
    identically on every attempt -- see ``_SEED_SENTINEL_LITERAL_
    PREFIX``'s own comment. Reusing the SAME per-attempt-distinct-
    leading-code-point scheme the pressure-filler marker path already
    established (and already empirically verified, outside this test
    suite, to sustain thousands of distinct first tokens against the
    real Qwen3-0.6B tokenizer) is what actually makes each retry a
    genuinely different candidate. This sentinel-marker "namespace"
    never needs to stay mutually distinct from pressure-filler target-
    head markers or from any other workload's own sentinel -- the only
    invariant required is THIS workload's own ``seed_sentinel_ids[0] !=
    shared_body_ids[0]`` (see ``NonPrefixSegmentWorkload.__post_init__``)
    -- so reusing the identical code-point sequence starting from
    ``attempt=0`` for every call is safe.

    This directly validates against the REAL ``tokenizer`` passed in --
    not a textual heuristic -- matching the same convention
    ``_build_pressure_filler_workload_avoiding_first_token_collisions``
    already established for target-head markers. Unlike that function,
    this one's collision constraint is against a SINGLE, per-workload
    dynamic value (``shared_body_ids[0]``) rather than an accumulating
    reserved set, so no ``reserved_first_token_ids``-style mutable state
    needs to be threaded through the caller.

    Raises ``RuntimeError`` (never silently returns a colliding
    sentinel) if every attempt collides -- see
    ``NonPrefixSegmentWorkload.seed_prompt_ids``'s docstring for exactly
    why a collision here would silently corrupt a later request's own
    ``cached_tokens`` measurement (the real SM75 bug this function
    exists to prevent).
    """
    for attempt in range(_MAX_SEED_SENTINEL_COLLISION_RETRIES):
        codepoint = _pressure_filler_marker_codepoint_for_combined_index(attempt)
        marker = chr(codepoint) + _SEED_SENTINEL_LITERAL_PREFIX
        candidate = _deterministic_token_ids(
            tokenizer,
            f"{salt}-seed-sentinel-{attempt}",
            NON_PREFIX_SEED_SENTINEL_TOKENS,
            literal_prefix=marker,
        )
        if candidate[0] != shared_body_ids[0]:
            return candidate
    raise RuntimeError(
        f"could not find a shared_body_ids[0]-distinct seed sentinel for "
        f"salt={salt!r} after {_MAX_SEED_SENTINEL_COLLISION_RETRIES} "
        "attempts against the real tokenizer -- check for a "
        "tokenizer/vocab misconfiguration or an exhausted "
        "_PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS pool"
    )


def build_non_prefix_segment_workload(
    tokenizer: Any,
    *,
    body_tokens: int,
    head_tokens: int,
    tail_tokens: int,
    salt: str,
    source_head_literal_prefix: str = _SOURCE_HEAD_LITERAL_PREFIX,
    target_head_literal_prefix: str = _TARGET_HEAD_LITERAL_PREFIX,
) -> NonPrefixSegmentWorkload:
    """Build one ``NonPrefixSegmentWorkload`` from four independently
    tokenized, deterministic pieces (see ``_deterministic_token_ids``).

    ``salt`` must be unique per distinct workload the canary constructs
    (the main setting, every shape-sweep point, and every rho-sweep point
    each pass their own salt) so that different settings' bodies/tails
    never share content. Cross-setting *head* isolation in the live
    server's exact radix tree is guaranteed separately, by
    ``run_non_prefix_setting`` flushing that tree as its own first action
    for every setting -- salt uniqueness alone would not be enough for
    that, since ``source_head_ids``/``target_head_ids`` need the
    stronger, structural zero-common-prefix guarantee the literal-prefix
    markers below provide (see the module docstring's "Why /generate and
    non-prefix segments" section).

    ``source_head_literal_prefix``/``target_head_literal_prefix`` default
    to this module's own fixed markers (every existing caller -- the main
    setting, every shape-sweep point, every rho-sweep point -- gets the
    exact prior behavior unchanged). ``build_eviction_pressure_workloads``
    overrides ``target_head_literal_prefix`` per filler object with a
    mutually distinct marker (see ``_pressure_filler_head_literal_prefix``):
    eviction-pressure filler objects' target heads are ALSO dense-seeded,
    like the setting's own head, and coexist with it and with every OTHER
    filler's head within the same flush epoch -- unlike the main-vs-sweep
    case, ``run_non_prefix_setting``'s per-setting flush does not isolate
    them from each other, so they need this stronger guarantee too (see
    ``validate_pairwise_head_isolation`` for the runtime safety net).
    """
    shared_body_ids = _deterministic_token_ids(
        tokenizer, f"{salt}-shared-body", body_tokens
    )
    target_head_ids = _deterministic_token_ids(
        tokenizer,
        f"{salt}-target-head",
        head_tokens,
        literal_prefix=target_head_literal_prefix,
    )
    return NonPrefixSegmentWorkload(
        source_head_ids=_deterministic_token_ids(
            tokenizer,
            f"{salt}-source-head",
            head_tokens,
            literal_prefix=source_head_literal_prefix,
        ),
        target_head_ids=target_head_ids,
        shared_body_ids=shared_body_ids,
        tail_ids=_deterministic_token_ids(tokenizer, f"{salt}-tail", tail_tokens),
        seed_sentinel_ids=_build_seed_sentinel_ids_avoiding_body_first_token_collision(
            tokenizer, salt=salt, shared_body_ids=shared_body_ids
        ),
    )


_MAX_PRESSURE_FILLER_FIRST_TOKEN_COLLISION_RETRIES = 64


def _build_pressure_filler_workload_avoiding_first_token_collisions(
    tokenizer: Any,
    *,
    index: int,
    body_tokens: int,
    head_tokens: int,
    tail_tokens: int,
    salt_prefix: str,
    reserved_first_token_ids: set[int],
) -> NonPrefixSegmentWorkload:
    """Build filler ``index``'s workload, trying up to
    ``_MAX_PRESSURE_FILLER_FIRST_TOKEN_COLLISION_RETRIES`` distinct
    candidate target-head markers (via ``_pressure_filler_head_literal_
    prefix`` over an expanded ``index * RETRIES + attempt`` combined
    index space, so every attempt -- for every filler -- is a globally
    distinct code point, never repeating a candidate already tried by
    this or any other filler) until one produces a target head whose
    first token id is not already in ``reserved_first_token_ids``.

    This directly validates against the REAL ``tokenizer`` passed in --
    not a textual heuristic -- so it is correct by construction for
    whatever tokenizer a given deployment actually uses, unlike relying
    on ``_pressure_filler_head_literal_prefix``'s code-point choice alone
    (see that function's docstring: a distinct leading code point is an
    empirically-good starting point, never a proof, of first-token
    distinctness). ``reserved_first_token_ids`` is mutated by the caller
    (``build_eviction_pressure_workloads``) after each accepted filler,
    so every later filler also avoids every earlier filler's (and any
    externally reserved, e.g. the setting's own head's) first token.

    Raises ``RuntimeError`` (never silently returns a colliding
    workload) if every attempt collides -- this would mean fewer than
    ``_MAX_PRESSURE_FILLER_FIRST_TOKEN_COLLISION_RETRIES`` distinct first
    tokens remain reachable from this filler's candidate markers, which
    was empirically observed to require several thousand already-
    accepted fillers against the real Qwen3-0.6B tokenizer and the
    current ``_PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS`` pool (>=8,000
    fillers succeeded with zero exhaustion in that check) -- reachable
    only near or beyond ``MAX_REASONABLE_EVICTION_PRESSURE_FILLER_
    COUNT``, and would indicate a genuine tokenizer/vocab
    misconfiguration or an exhausted code-point pool worth surfacing
    loudly rather than masking.
    """
    for attempt in range(_MAX_PRESSURE_FILLER_FIRST_TOKEN_COLLISION_RETRIES):
        combined_index = (
            index * _MAX_PRESSURE_FILLER_FIRST_TOKEN_COLLISION_RETRIES + attempt
        )
        marker = _pressure_filler_head_literal_prefix(combined_index)
        candidate = build_non_prefix_segment_workload(
            tokenizer,
            body_tokens=body_tokens,
            head_tokens=head_tokens,
            tail_tokens=tail_tokens,
            salt=f"{salt_prefix}-filler-{index}",
            target_head_literal_prefix=marker,
        )
        if candidate.target_head_ids[0] not in reserved_first_token_ids:
            return candidate
    raise RuntimeError(
        f"could not find a first-token-distinct target-head marker for "
        f"pressure filler {index} after "
        f"{_MAX_PRESSURE_FILLER_FIRST_TOKEN_COLLISION_RETRIES} attempts "
        "against the real tokenizer -- this should only be reachable "
        "near or beyond MAX_REASONABLE_EVICTION_PRESSURE_FILLER_COUNT; "
        "check for a tokenizer/vocab misconfiguration or an exhausted "
        "_PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS pool"
    )


def build_eviction_pressure_workloads(
    tokenizer: Any,
    *,
    object_count: int,
    body_tokens: int,
    head_tokens: int,
    tail_tokens: int,
    salt_prefix: str,
    reserved_first_token_ids: frozenset[int] = frozenset(),
) -> tuple[NonPrefixSegmentWorkload, ...]:
    """Build ``object_count`` distinct filler ``NonPrefixSegmentWorkload``
    objects meant purely to occupy real, finite GPU KV-pool capacity
    before a setting's own measurement (see
    ``register_eviction_pressure_objects``) -- never measured for TTFT
    themselves.

    Each filler gets its own ``salt_prefix-filler-{index}`` content salt
    (so no two fillers, and no filler and any main/sweep setting, ever
    share body/tail content) AND its own target-head marker, chosen so
    that its resulting target head's FIRST TOKEN is guaranteed -- by
    direct validation against this call's real ``tokenizer``, not by
    trusting any textual convention -- to differ from every other
    filler's first token and from every id in ``reserved_first_token_ids``
    (pass the calling setting's own head's first token here so a filler
    can never collide with it either; see ``validate_pairwise_head_
    isolation``, the downstream, redundant final check).

    An earlier version of this function relied solely on a fixed-width,
    string-prefix-free letter code, on the (incorrect) assumption that
    string-level distinctness at a fixed width was sufficient. It
    empirically is NOT: e.g. at ``object_count=64``, two of the
    resulting markers were found, against the real Qwen3-0.6B tokenizer,
    to tokenize to the SAME first token id despite being distinct
    strings, and short synthetic ASCII/hex text was separately found to
    plateau at only ~183-400 distinct achievable first tokens regardless
    of alphabet or length -- far short of what hundreds/thousands of
    filler objects need. This function now draws candidate markers from
    ``_PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS`` (empirically verified to
    sustain >=8,000 zero-collision fillers) and performs a bounded,
    real-tokenizer-validated retry search per filler (see
    ``_build_pressure_filler_workload_avoiding_first_token_collisions``)
    so correctness never depends on any assumption about a specific
    tokenizer's BPE merge behaviour -- it holds for whatever tokenizer is
    actually passed in, fake or real, for any ``object_count`` up to
    ``MAX_REASONABLE_EVICTION_PRESSURE_FILLER_COUNT``.
    """
    if object_count <= 0:
        raise ValueError(f"object_count must be positive, got {object_count}")
    accepted_first_token_ids: set[int] = set(reserved_first_token_ids)
    workloads: list[NonPrefixSegmentWorkload] = []
    for index in range(object_count):
        workload = _build_pressure_filler_workload_avoiding_first_token_collisions(
            tokenizer,
            index=index,
            body_tokens=body_tokens,
            head_tokens=head_tokens,
            tail_tokens=tail_tokens,
            salt_prefix=salt_prefix,
            reserved_first_token_ids=accepted_first_token_ids,
        )
        accepted_first_token_ids.add(workload.target_head_ids[0])
        workloads.append(workload)
    return tuple(workloads)


def eviction_pressure_total_tokens(
    workloads: Sequence[NonPrefixSegmentWorkload],
) -> int:
    """Lower-bound estimate, in tokens, of the exact-radix-tree KV
    footprint that sending every pressure workload as one plain dense
    request contributes (see ``register_eviction_pressure_objects``):
    each filler's own single dense forward pass populates its FULL
    ``target_prompt_ids`` (head + body + tail) into the ordinary,
    LRU-evictable exact radix tree, so this -- the body length alone --
    is a floor on each filler's real footprint, not an exact total (it
    excludes each filler's own small, fixed head+tail contribution,
    negligible next to a multi-hundred/thousand-token body). Reported
    alongside ``observed_rho_after_pressure`` in
    ``register_eviction_pressure_objects``'s own returned dict for
    downstream debugging, AND used by that same function to gate its
    own ``evicted_tokens_total_delta`` assertion (see that function's
    docstring): this floor being enough, by itself, to exceed a
    setting's own ``capacity_tokens`` is exactly the construction that
    assertion relies on.
    """
    return sum(workload.body_tokens for workload in workloads)


def eviction_pressure_filler_count_for_rho(
    *,
    target_rho: float,
    usable_capacity_tokens: int,
    tokens_per_filler: int,
    already_pinned_tokens: int = 0,
) -> int:
    """Reverse-compute how many filler objects (each contributing
    ``tokens_per_filler`` tokens, see ``eviction_pressure_total_tokens``)
    are needed so their combined *nominal* (requested, not sampled-live)
    token footprint, PLUS ``already_pinned_tokens``, reaches at least
    ``target_rho * usable_capacity_tokens``.

    This is the "actual capacity自动反算...所需filler数" calculation: a
    real, live ``usable_kv_capacity_tokens`` reading (see
    ``benchmark.approx_kv.metrics``) drives how many filler objects a
    given ``target_rho`` requires, rather than a fixed filler count
    guessed independently of the server's real pool size. It is
    ``ceil``-rounded so the achieved nominal ratio is always >=
    ``target_rho``, never short of it by a fractional-filler rounding
    error.

    ``already_pinned_tokens`` (default 0, preserving the original
    filler-only formula for any caller that has no such footprint to
    account for) is the setting's own raw+fresh source-registration KV
    that is ALREADY resident by the time fillers are computed -- see
    ``run_non_prefix_setting``'s "register sources before pressure"
    ordering. That footprint counts toward ``target_rho`` exactly like a
    filler's own tokens do (both occupy the same live pool), so it is
    subtracted from the nominal target before dividing by
    ``tokens_per_filler``: fillers only need to make up the REMAINDER,
    the "target pre-rho" still unmet by the setup's own already-pinned
    footprint. If ``already_pinned_tokens`` alone already meets or
    exceeds the nominal target (a small pool / large setup body / high
    ``target_rho`` combination), zero fillers are needed at all -- this
    is a legitimate outcome, not an error, and is returned as ``0``
    rather than a negative count.

    "Nominal" because it is computed purely from *requested* filler
    tokens (plus the setup's own *measured*, not estimated,
    ``already_pinned_tokens``); the pool's real, observed occupancy at
    any instant also depends on whatever eviction has already reclaimed
    by the time later fillers are sent (see ``observed_rho`` for the
    genuine, sampled counterpart, read from the live
    ``sglang:kv_used_tokens`` PLUS ``sglang:kv_evictable_tokens``
    gauges -- resident occupancy, not merely pinned/in-use tokens) --
    the two are reported side by side, never conflated.

    Raises immediately -- before any pressure-phase HTTP request is ever
    made -- if the reverse-computed count exceeds
    ``MAX_REASONABLE_EVICTION_PRESSURE_FILLER_COUNT``: this is a
    defensive sanity bound only (never hit by any plausible real
    ``--main-target-rho``/``--target-rho-choices``/``--pressure-filler-
    body-tokens`` combination against a real GPU pool -- see that
    constant's own comment), catching a pathological misconfiguration
    (e.g. an accidentally tiny ``--pressure-filler-body-tokens`` against
    a large measured capacity) up front rather than letting
    ``register_eviction_pressure_objects`` silently spend a very long
    time issuing thousands of real, blocking per-filler HTTP requests.
    """
    if target_rho <= 0:
        raise ValueError(f"target_rho must be positive, got {target_rho}")
    if usable_capacity_tokens <= 0:
        raise ValueError(
            f"usable_capacity_tokens must be positive, got {usable_capacity_tokens}"
        )
    if tokens_per_filler <= 0:
        raise ValueError(f"tokens_per_filler must be positive, got {tokens_per_filler}")
    if already_pinned_tokens < 0:
        raise ValueError(
            f"already_pinned_tokens must be >= 0, got {already_pinned_tokens}"
        )
    target_total_tokens = target_rho * usable_capacity_tokens
    remaining_tokens = target_total_tokens - already_pinned_tokens
    if remaining_tokens <= 0:
        return 0
    filler_count = math.ceil(remaining_tokens / tokens_per_filler)
    if filler_count > MAX_REASONABLE_EVICTION_PRESSURE_FILLER_COUNT:
        raise ValueError(
            f"reverse-computed filler_count={filler_count} exceeds the "
            f"sanity bound of {MAX_REASONABLE_EVICTION_PRESSURE_FILLER_COUNT} "
            f"(target_rho={target_rho}, "
            f"usable_capacity_tokens={usable_capacity_tokens}, "
            f"tokens_per_filler={tokens_per_filler}, "
            f"already_pinned_tokens={already_pinned_tokens}) -- this would "
            "require thousands of real, blocking per-filler HTTP round "
            "trips before this setting's own measurement even begins; "
            "raise --pressure-filler-body-tokens (fewer, larger fillers "
            "needed for the same target_rho) or lower --main-target-rho/"
            "--target-rho-choices instead"
        )
    return filler_count


def observed_rho(snapshot: Mapping[str, float], *, capacity_tokens: int) -> float:
    """The real, sampled fraction of this ``snapshot``'s live pool that
    is genuinely RESIDENT -- ``sglang:kv_used_tokens`` (this instant's
    pinned/in-use tokens) PLUS ``sglang:kv_evictable_tokens`` (tokens
    still occupying device memory as LRU-evictable exact-radix entries,
    e.g. surviving eviction-pressure fillers, that have not actually
    been reclaimed yet) -- against a fixed ``capacity_tokens``
    reference. This is the genuine, *measured* occupancy/pressure
    fraction at the instant ``snapshot`` was taken, as opposed to
    ``eviction_pressure_filler_count_for_rho``'s nominal (requested-
    tokens) ratio.

    A REAL SM75 bug this fixes: an earlier version of this function
    used ``kv_used_tokens`` ALONE as the numerator -- the pool's
    currently pinned/in-use tokens only, conceptually the same quantity
    the server's own ``sglang:full_token_usage`` gauge reports
    (``full_num_used / pool_size``, see
    ``PoolStats.update_scheduler_stats`` server-side). That undercounts
    genuine device pressure whenever a large population of dense
    eviction-pressure fillers remains resident as LRU-evictable (not yet
    actually evicted) exact-radix entries: on a real ``target_rho=2``
    canary this reported ``peak_rho_observed=0.156`` (``kv_used_tokens``
    alone, 2048 / 13130) even though the pool was in fact ~99% resident
    (``(2048 used + 10960 evictable) / 13130 ~= 0.991``) once every
    surviving filler is counted too -- exactly the "high pressure"
    condition ``--target-rho-choices``/``--main-target-rho`` are meant
    to characterize. ``used + evictable`` is equivalent to ``capacity -
    available`` (``1 - kv_available_tokens / capacity_tokens``, up to
    the same accounting tolerance ``idle_pool_invariant`` verifies) --
    either reflects genuine resident occupancy, unlike ``used`` alone.

    ``capacity_tokens`` is deliberately a caller-supplied fixed value
    (established once, immediately after a flush, via
    ``usable_kv_capacity_tokens`` on a genuinely idle pool snapshot) --
    never recomputed from ``snapshot`` itself here, since
    ``usable_kv_capacity_tokens``'s own idle heuristic could react to a
    transient, eviction-driven usage dip in a snapshot taken mid-
    pressure and silently swap its capacity basis out from under a
    "peak rho" comparison across multiple snapshots of the same setting.

    Raises immediately -- never silently substitutes 0 or falls back to
    ``used`` alone -- if either ``sglang:kv_used_tokens`` or
    ``sglang:kv_evictable_tokens`` is unavailable in ``snapshot``: a
    ``peak_rho_observed`` computed from a partially-missing snapshot
    would be silently wrong in the exact same undercounting way this
    fix addresses, so it must never be trusted.
    """
    if capacity_tokens <= 0:
        raise ValueError(f"capacity_tokens must be positive, got {capacity_tokens}")
    used = snapshot.get("sglang:kv_used_tokens")
    evictable = snapshot.get("sglang:kv_evictable_tokens")
    missing = [
        name
        for name, value in (
            ("sglang:kv_used_tokens", used),
            ("sglang:kv_evictable_tokens", evictable),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"{', '.join(missing)} unavailable in this snapshot -- cannot "
            "compute observed_rho (used + evictable, against a fixed "
            "capacity) without both"
        )
    return (float(used) + float(evictable)) / float(capacity_tokens)


def chunk_offsets(
    total_tokens: int, max_chunk_tokens: int
) -> tuple[tuple[int, int], ...]:
    """Split ``total_tokens`` into contiguous ``(offset, length)`` chunks,
    each at most ``max_chunk_tokens`` long, offsets 0-based relative to
    the start of the span being chunked.

    Used to keep every single approx_kv segment this canary registers or
    reuses at most ``--max-segment-chunk-tokens`` long (default 512):
    with the unified body sweep now reaching up to 2048 tokens, a body
    longer than that is split into multiple segments rather than ever
    registering/reusing one oversized segment. On the REUSE side those
    segments still all travel within one single call's ``segments`` list
    (see ``body_segments_for_hash``) -- ``ApproxKVRequestMetadata``/
    ``restore_request_prefix_cachetune`` already natively iterate over
    an arbitrary number of segments per call, so this chunking changes
    nothing about that side's underlying server-side contract. On the
    REGISTER side, each chunk instead becomes its OWN independent
    ``/generate`` call (see ``register_body_chunks``): a single call
    whose STORED segment sizes were already chunk-bounded still let its
    own live/transient per-request KV footprint scale with the FULL
    un-chunked body, which OOM'd a real SM75 server at register time --
    this function's ``(offset, length)`` tuples are what both the
    reuse-side grouping and the register-side per-chunk splitting are
    built from.
    """
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")
    if max_chunk_tokens <= 0:
        raise ValueError(f"max_chunk_tokens must be positive, got {max_chunk_tokens}")
    chunks = []
    offset = 0
    while offset < total_tokens:
        length = min(max_chunk_tokens, total_tokens - offset)
        chunks.append((offset, length))
        offset += length
    return tuple(chunks)


def body_segments_for_hash(
    *,
    hash_prefix: str,
    body_start: int,
    body_tokens: int,
    max_chunk_tokens: int,
) -> list[dict[str, Any]]:
    """Build a REUSE payload ``"segments"`` list for a ``body_tokens``-
    long span anchored at ``body_start``, split into
    ``chunk_offsets(body_tokens, max_chunk_tokens)`` pieces, all
    traveling together within that single reuse call.

    NOTE: this is the REUSE-side builder only. The REGISTER side no
    longer sends its chunks this way -- ``register_body_chunks`` issues
    one independent ``/generate`` call per chunk instead (see that
    function's own docstring for the real SM75 OOM this avoids), each
    with ``target_start = len(head_ids)`` (local to that call's own
    short prompt), not ``body_start + offset`` as built here. The two
    still line up correctly at reuse time purely through the shared
    ``content_hash`` convention below (``manager.store``'s lookup key,
    ``_segment_key``, is built from ``content_hash`` + token CONTENT --
    never from ``target_start`` -- see ``approx_kv/runtime.py``).

    Every chunk gets its own distinct, deterministic ``content_hash``
    (``f"{hash_prefix}:chunk{index}"``): even chunks of the same logical
    body are independent entries in ``manager.store`` (keyed by
    ``_segment_key``, see ``cachetune/runtime.py``), never one shared
    key. ``hash_prefix`` is expected to already carry this module's
    ``cachetune-raw:``/``cachetune-fresh:`` distinguishing prefix (see
    ``_RAW_PREFIX``/``_FRESH_PREFIX`` in ``cachetune/runtime.py``):
    ``restore_request_prefix_cachetune`` discovers a segment's
    corresponding "fresh" companion via
    ``segment.content_hash.replace(_RAW_PREFIX, _FRESH_PREFIX, 1)``, so a
    caller building a raw+fresh pair for the *same* logical body MUST
    call this twice with ``hash_prefix`` values that differ *only* by
    that prefix swap (every ``raw_hash``/``fresh_hash`` pair this module
    constructs already satisfies that -- see e.g. ``run_canary``'s
    ``"cachetune-raw:phase4-r5-main"``/``"cachetune-fresh:phase4-r5-main"``)
    for the resulting per-chunk hashes to still line up correctly at
    reuse time.
    """
    return [
        {
            "content_hash": f"{hash_prefix}:chunk{index}",
            "target_start": body_start + offset,
            "length": length,
        }
        for index, (offset, length) in enumerate(
            chunk_offsets(body_tokens, max_chunk_tokens)
        )
    ]


def _first_common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    length = 0
    for token_a, token_b in zip(a, b):
        if token_a != token_b:
            break
        length += 1
    return length


def validate_pairwise_head_isolation(
    labeled_heads: Sequence[tuple[str, Sequence[int]]],
) -> None:
    """Raise a clear, actionable ``RuntimeError`` the moment any two
    simultaneously-coexisting dense-seeded target heads share a nonzero
    common token-id prefix.

    Every head in ``labeled_heads`` gets dense-seeded into the exact
    radix tree within the same flush epoch (the setting's own head, plus
    every eviction-pressure filler object's own head -- see
    ``run_non_prefix_setting``), each expecting its own seed request to
    report ``cached_tokens=0``. A shared prefix between any two of them
    would make a later seed request silently observe a nonzero match
    against an earlier head already sitting in the tree, corrupting that
    invariant. This checks the actual resulting token-id sequences
    directly -- never a textual heuristic alone -- precisely because a
    "distinct first character" literal-prefix convention cannot be
    verified against a real tokenizer's BPE merge behavior without a
    live tokenizer, so this catches any surprise immediately and loudly
    rather than assuming the textual convention worked.
    """
    for i in range(len(labeled_heads)):
        label_a, head_a = labeled_heads[i]
        for j in range(i + 1, len(labeled_heads)):
            label_b, head_b = labeled_heads[j]
            shared = _first_common_prefix_length(head_a, head_b)
            if shared > 0:
                raise RuntimeError(
                    f"{label_a!r} and {label_b!r} target heads share a "
                    f"{shared}-token common prefix "
                    f"({tuple(head_a[:shared])!r}) -- every "
                    "simultaneously dense-seeded target head (the "
                    "setting's own head plus every eviction-pressure "
                    "filler object's head) must be pairwise "
                    "zero-common-prefix, or a later seed request would "
                    "silently observe a nonzero cached_tokens match "
                    "against an earlier head already sitting in the "
                    "exact radix tree. Use more diverse literal-prefix "
                    "markers for these two heads (see "
                    "_PRESSURE_FILLER_MARKER_CODEPOINT_BLOCKS / "
                    "_SOURCE_HEAD_LITERAL_PREFIX / "
                    "_TARGET_HEAD_LITERAL_PREFIX)."
                )


def dense_generate_payload(input_ids: Sequence[int]) -> dict:
    return {
        "input_ids": list(input_ids),
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": 0,
        },
    }


def register_generate_payload(
    *,
    input_ids: Sequence[int],
    segments: Sequence[Mapping[str, Any]],
    model_fingerprint: str,
    cache_dtype: str,
) -> dict:
    """``segments`` is a ready-to-send list of ``{"content_hash",
    "target_start", "length"}`` dicts. ``register_request_segments``
    (``cachetune/runtime.py``) natively iterates over an arbitrary
    number of segments per call, so this function itself places no
    restriction on ``len(segments)`` -- but every caller in this script
    now passes exactly ONE segment per call (see
    ``register_body_chunks``): a body longer than
    ``--max-segment-chunk-tokens`` is registered as one independent call
    PER chunk, never multiple segments within one oversized call, since
    that oversized call's own transient per-request KV footprint (not
    just its eventually-stored segment sizes) previously OOM'd a real
    SM75 server."""
    return {
        "input_ids": list(input_ids),
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": 0,
            "custom_params": {
                "approx_kv": {
                    "operation": "register",
                    "model_fingerprint": model_fingerprint,
                    "cache_dtype": cache_dtype,
                    "segments": [dict(segment) for segment in segments],
                }
            },
        },
    }


def reuse_generate_payload(
    *,
    input_ids: Sequence[int],
    segments: Sequence[Mapping[str, Any]],
    model_fingerprint: str,
    cache_dtype: str,
) -> dict:
    """``segments`` is a ready-to-send list of ``{"content_hash",
    "target_start", "length"}`` dicts (see ``body_segments_for_hash``),
    each ``content_hash`` matching one of the corresponding raw-register
    call's own per-chunk hashes -- see
    ``restore_request_prefix_cachetune``'s ``_segment_key`` lookup."""
    return {
        "input_ids": list(input_ids),
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": 0,
            "custom_params": {
                "approx_kv": {
                    "operation": "reuse",
                    "plugin": "cachetune",
                    "model_fingerprint": model_fingerprint,
                    "cache_dtype": cache_dtype,
                    "segments": [dict(segment) for segment in segments],
                }
            },
        },
    }


async def _stream_generate_and_measure_ttft(
    base_url: str, payload: dict, timeout: float
) -> tuple[dict, float]:
    """POST ``payload`` to ``/generate`` with ``stream: true`` and return
    ``(response, ttft_ms)`` where ``ttft_ms`` is genuine client
    time-to-first-token.

    ``ttft_ms`` is timestamped the moment the first non-``[DONE]`` SSE
    ``data:`` frame is received off the wire -- before that frame's JSON
    body is even decoded -- never the blocking whole-request elapsed
    time this script used previously. With ``max_new_tokens=1`` (every
    payload this script builds) that blocking number happens to be
    *close* to TTFT, but it still bundles in the server's full-response
    detokenization/serialization and the complete HTTP body transfer
    that only happen strictly after the first (and only) token was
    already produced -- a strictly looser upper bound on TTFT, never
    TTFT itself, and TTFT is this script's sole client-facing metric
    (see module docstring's "TTFT measurement methodology" section).

    The stream is always read in full through the terminal
    ``data: [DONE]`` frame -- never abandoned right after the first
    chunk -- as this call's own success check: a connection that drops,
    or a stream that never reaches ``[DONE]``, raises rather than
    silently reporting a "successful" TTFT for a request that never
    actually completed. Any mid-stream frame carrying an ``"error"`` key
    (see ``http_server.py``'s ``stream_results`` error branch) also
    raises immediately rather than being folded into ``response``. The
    *last* non-``[DONE]`` frame observed is returned as ``response`` --
    guaranteed complete/final, the same semantics the old blocking
    response object had -- so ``require_finished_by_length`` and
    ``require_cached_tokens`` keep working unchanged against it.
    """
    request_payload = {**payload, "stream": True}
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    last_chunk: dict | None = None
    ttft_ms: float | None = None
    saw_done = False
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        start = time.perf_counter()
        async with session.post(
            f"{base_url}/generate", json=request_payload
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body}")
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                encoded = line[len("data: ") :]
                if encoded == "[DONE]":
                    saw_done = True
                    break
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000.0
                chunk = json.loads(encoded)
                if "error" in chunk:
                    raise RuntimeError(
                        f"/generate streaming response reported an error: "
                        f"{chunk['error']}"
                    )
                last_chunk = chunk
    if not saw_done:
        raise RuntimeError(
            "/generate streaming response ended without a terminal "
            "'data: [DONE]' frame -- treating this as a failed request, "
            "never an approximate success"
        )
    if last_chunk is None or ttft_ms is None:
        raise RuntimeError(
            "/generate streaming response reached '[DONE]' without ever "
            "emitting a data chunk -- no token was produced, so there is "
            "no TTFT to measure"
        )
    return last_chunk, ttft_ms


def timed_post(
    base_url: str, payload: dict, timeout: float = 300
) -> tuple[dict, float]:
    """POST ``payload`` to ``/generate`` and return ``(response,
    ttft_ms)``, where ``ttft_ms`` is genuine client time-to-first-token
    measured over a real streaming (``stream: true``) request -- see
    ``_stream_generate_and_measure_ttft`` for the full rationale. Every
    canary request in this script (head-seed, dense baseline,
    raw-register, fresh-register, reuse) goes through this single
    function, so this script's entire TTFT/ms telemetry is always
    genuine streamed client TTFT, never a blocking-elapsed
    approximation.
    """
    return asyncio.run(_stream_generate_and_measure_ttft(base_url, payload, timeout))


def require_finished_by_length(response: dict, label: str) -> None:
    """Native ``/generate`` shapes ``finish_reason`` as a dict (e.g.
    ``{"type": "length", "length": 1}``, see
    ``schedule_batch.FINISH_LENGTH.to_json``) -- unlike the OpenAI-
    compatible endpoint's plain string. This script only ever posts to
    ``/generate``, so no plain-string fallback is accepted here.
    """
    finish_reason = response["meta_info"]["finish_reason"]
    if finish_reason.get("type") != "length":
        raise RuntimeError(f"{label} request did not finish by length: {finish_reason}")


def require_cached_tokens(response: dict, expected: int, label: str) -> int:
    """Assert the server-reported prefix length already resolved without
    a fresh forward pass (``meta_info.cached_tokens``, generic SGLang
    accounting set from ``pre_len - already_computed`` in
    ``schedule_batch.py`` -- unrelated to any CacheTune-specific
    Prometheus counter) equals ``expected`` for this specific request,
    and return the observed value.

    IMPORTANT, confirmed on a real SM75 run: this is *not* an
    exact-match-only counter. ``pre_len = len(req.prefix_indices)``,
    and a successful CacheTune reuse extends ``req.prefix_indices`` by
    the *entire* restored body span (``restore_length``) in
    ``restore_request_prefix_cachetune`` (``cachetune/runtime.py``) --
    regardless of the controller's selected repair ratio, since
    ``decision.repair_tokens`` only picks how many already-restored
    positions get a genuine recompute forward pass versus a straight KV
    copy, never how many positions get restored in total. So for a
    reuse request the caller must pass ``body_start_in_target +
    body_tokens`` (exact-match head *plus* the full restored body), not
    ``body_start_in_target`` alone -- an earlier version of this script
    passed the head-only value here and a real SM75 canary run reported
    a mismatch (``cached_tokens`` observed as head+body, expected as
    head-only). For a REGISTER request (raw or fresh), which never calls
    ``restore_request_prefix_cachetune`` at all (see
    ``approx_kv/runtime.py``'s ``_register_request_segments``), the
    expected value is genuinely just whatever plain exact-match radix
    hit already existed before that call (0 for the raw register, since
    its unique ``source_head_ids`` was never previously seeded;
    ``body_start_in_target`` for the fresh register, since its
    ``target_head_ids`` was already seeded by the one-time dense head
    seed earlier in the same setting).

    This is an independent, per-request cross-check that the live
    request's own prefix boundary landed exactly where this canary's
    registered segment(s) expect it to -- not a tautology against the
    aggregate Prometheus deltas this script already cross-validates
    elsewhere, since it is sourced from a completely different counter.
    """
    observed = int(response["meta_info"]["cached_tokens"])
    if observed != expected:
        raise RuntimeError(
            f"{label} request reported cached_tokens={observed}, expected "
            f"exactly {expected}"
        )
    return observed


def metric_snapshot(base_url: str) -> dict[str, float]:
    return parse_prometheus_text(fetch_text(f"{base_url}/metrics"))


def metric_delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def flush_and_force_gauge_refresh(
    base_url: str, tokenizer: Any, *, label: str
) -> dict[str, float]:
    """Flush the exact-match radix cache, then force one real scheduler
    iteration via a small, fixed, dense sentinel request (see
    ``_POOL_RESET_SENTINEL_SEED``/``_POOL_RESET_SENTINEL_TOKENS``), then
    return a fresh ``/metrics`` snapshot -- the ONLY way to get a
    genuinely up-to-date ``sglang:kv_used_tokens`` (and every other
    gauge) reading immediately after a flush.

    ``/flush_cache`` clears the actual pool/tree state SYNCHRONOUSLY: a
    bare dense request posted right after a flush, with no sentinel in
    between, already correctly reports ``cached_tokens=0`` (this is
    exactly how every round's own subsequent ``register_round_setup``
    seed call has always worked). But the separately-exported
    Prometheus GAUGES (``sglang:kv_used_tokens`` and friends) are only
    recomputed by the scheduler's own NEXT iteration, not synchronously
    by ``/flush_cache`` itself -- so a bare ``/metrics`` scrape taken
    immediately after a flush, with no intervening real request, can
    still read a value carried over from whatever was resident just
    before the flush.

    THIS IS A DELIBERATE FIX for a real SM75 bug on a body-length sweep:
    ``run_independent_round`` used to flush and immediately snapshot
    ``/metrics`` with no intervening request, so a setting/round's own
    ``metrics_at_round_start`` could carry over the PREVIOUS setting's
    own final ``sglang:kv_used_tokens`` reading verbatim (e.g. 2048,
    from a just-finished body=1024 setting's own raw+fresh footprint,
    never cleared from the GAUGE by that flush alone); the NEXT
    setting's own post-setup reading (e.g. 1024, for a genuine body=512
    footprint measured from what should have been a truly-idle
    baseline) then produced ``already_pinned_tokens = 1024 - 2048 =
    -1024`` -- a structurally negative, nonsensical value. This script
    never clamps that away (see ``eviction_pressure_filler_count_for_rho``'s
    own ``already_pinned_tokens < 0`` check, which correctly raises
    ``ValueError`` rather than silently treating a negative value as
    zero) -- the ROOT problem, a stale baseline snapshot, had to be
    fixed here instead, at its source.

    Callers whose very next step performs its own exact-radix-match-
    sensitive request (e.g. ``run_independent_round``'s own upcoming
    ``register_round_setup`` seed call, which requires
    ``cached_tokens == 0``) MUST flush again themselves before that step
    -- this function's own contract is only "return an accurate
    snapshot", never "leave the tree empty after returning": the
    sentinel posted here is a plain dense request (no ``approx_kv``
    metadata), so the scheduler DOES insert it into the exact radix tree
    (see ``flush_exact_radix_cache``'s own docstring on which request
    kinds do), and its own tiny footprint remains resident (moved to
    ``sglang:kv_evictable_tokens`` once its own generation completes,
    never counted in ``sglang:kv_used_tokens``) until the NEXT flush
    clears it. A second, bare ``flush_exact_radix_cache`` call (no
    sentinel needed, since nothing reads ``/metrics`` again until after
    a further real request runs) is all a caller needs to guarantee a
    genuinely empty tree afterward -- see ``run_independent_round``'s
    own use of exactly that pattern.

    No step here catches its own exceptions: a flush failure or a
    failed/stuck sentinel request must propagate all the way up to
    whichever central-log "failed" entry the caller's own top-level
    error handling appends to, exactly like every other unrecoverable
    error in this script (see ``post_empty``'s own docstring for the
    same rationale) -- a silently-ignored failure here would silently
    hide a real stale-gauge problem behind a misleadingly "clean"
    snapshot.
    """
    flush_exact_radix_cache(base_url)
    sentinel_ids = _deterministic_token_ids(
        tokenizer, _POOL_RESET_SENTINEL_SEED, _POOL_RESET_SENTINEL_TOKENS
    )
    sentinel_response, _ = timed_post(base_url, dense_generate_payload(sentinel_ids))
    require_finished_by_length(sentinel_response, f"{label} sentinel")
    return metric_snapshot(base_url)


def capture_final_pool_reset_and_invariant(
    base_url: str, tokenizer: Any
) -> dict[str, Any]:
    """Flush every raw/fresh CacheTune segment this run registered and
    every dense-cached exact-radix-tree entry (including every
    eviction-pressure filler object, now sent as plain dense requests --
    see ``register_eviction_pressure_objects``), then capture the
    resulting genuinely-idle pool invariant -- WITHOUT ever treating
    those now-flushed registrations as if they had been a leak.

    Every setting this canary measures registers raw/fresh source-
    context segments (via each round's own ``register_round_setup``)
    and, for pressure settings, many plain-dense filler objects (see
    ``register_eviction_pressure_objects``); every round -- including
    every formal repeat -- is flushed at its own start (see
    ``flush_exact_radix_cache``'s own docstring), so only the LAST
    formal round's own raw/fresh segments (CacheTune's own segment
    store) and dense fillers (the ordinary exact radix tree) are still
    resident by the time this function runs -- an earlier round's own
    registration was already wiped by ITS OWN successor round's flush,
    never accumulated across rounds. That trailing residency is still
    intentional, not a leak: "genuinely resident until the run's own
    final cleanup" is the entire point for the last round's own
    CacheTune segments, and "genuine, realistic cache pressure" is the
    entire point for dense fillers. So immediately after the last
    measurement, ``sglang:kv_used_tokens`` is genuinely nonzero by
    design (a real SM75 run observed 4096 used tokens at exactly this
    point, with ``accounted_tokens`` already matching
    ``max_total_num_tokens`` exactly): running ``idle_pool_invariant``
    directly against that snapshot would misreport this expected,
    by-design residency as a pool leak and fail an otherwise
    fully-successful canary for no real defect.

    So this function:

    1. Snapshots ``/metrics`` first (``metrics_pre_reset``) -- kept only
       for visibility in the result JSON, never used to gate pass/fail.
    2. Calls ``flush_and_force_gauge_refresh`` (``label="final
       pool-reset"``), which flushes the exact radix tree (releasing
       every dense-cached entry, including every eviction-pressure
       filler object) AND resets ``ApproxKVManager`` (releasing every
       raw/fresh CacheTune segment this run registered), THEN posts one
       small, fixed *sentinel* ``/generate`` request to force one real
       scheduler iteration -- gauges such as ``sglang:kv_used_tokens``
       are only recomputed by the scheduler's own next iteration, not
       synchronously by ``/flush_cache`` itself (see that function's
       own docstring) -- and returns the resulting fresh ``/metrics``
       snapshot (``metrics_post_reset``).
    3. Runs ``idle_pool_invariant`` on ``metrics_post_reset`` alone --
       this is the only invariant result callers should gate pass/fail
       on, never the pre-reset snapshot from step 1.

    No step here catches its own exceptions: a flush failure or a
    failed/stuck sentinel request must propagate all the way up to
    ``main``'s existing central-log "failed" entry, exactly like every
    other unrecoverable error in this script (see ``post_empty``'s own
    docstring for the same rationale) -- a silently-ignored failure here
    would silently hide a real final-state problem behind a misleading
    "passed" result.
    """
    metrics_pre_reset = metric_snapshot(base_url)
    metrics_post_reset = flush_and_force_gauge_refresh(
        base_url, tokenizer, label="final pool-reset"
    )
    return {
        "metrics_pre_reset": metrics_pre_reset,
        "metrics_post_reset": metrics_post_reset,
        "pool_invariant": idle_pool_invariant(metrics_post_reset),
    }


def expected_repair_totals(
    *,
    repair_tokens_per_call: int,
    recomputed_layers_per_call: int,
    repeats: int,
) -> dict[str, Any]:
    """Pure computation of the telemetry totals CacheTune's Prometheus
    counters must show after exactly ``repeats`` *formal* reuse calls.

    Never includes the discarded warmup pass: its telemetry effect is
    already baked into the "before" snapshot, which this runner always
    takes only after warmup has completed (see ``run_canary``), so
    ``repeats`` here must be the formal-repeat count alone.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if repair_tokens_per_call < 0:
        raise ValueError(
            f"repair_tokens_per_call must be >= 0, got {repair_tokens_per_call}"
        )
    if recomputed_layers_per_call < 0:
        raise ValueError(
            "recomputed_layers_per_call must be >= 0, got "
            f"{recomputed_layers_per_call}"
        )
    expect_precomputed_adapter = repair_tokens_per_call > 0
    return {
        "expect_precomputed_adapter": expect_precomputed_adapter,
        "expected_selected_tokens_total": repair_tokens_per_call * repeats,
        "expected_recomputed_layers_total": (
            recomputed_layers_per_call * repeats if expect_precomputed_adapter else 0
        ),
        "expected_precomputed_total": repeats if expect_precomputed_adapter else 0,
    }


def append_run_log(path: Path, entry: dict[str, Any]) -> None:
    """Append one JSONL lifecycle record to the shared central log.

    Mirrors the schema already established by
    ``research/epic-legolink``'s ``run_phase4_epic_inrequest_matrix.py``
    (``run_id`` / ``status`` / ``timestamp`` / ``settings`` / ``output``,
    plus ``result_summary`` on success or ``error`` on failure) so a
    human or tool reading multiple sibling canaries' central logs sees
    one uniform shape. The file is opened in append mode and never
    truncated: many independent runs share the same ``--central-log``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True))
        handle.write("\n")


def build_settings(args: argparse.Namespace) -> dict[str, Any]:
    """JSON-safe snapshot of every setting relevant to reproducing a run.

    Shared verbatim by the ``running`` / ``completed`` / ``failed``
    central-log records for one invocation.
    """
    return {
        "base_url": args.base_url,
        "model": args.model,
        "model_revision": args.model_revision,
        "model_fingerprint": args.model_fingerprint,
        "cache_dtype": args.cache_dtype,
        "mode": args.mode,
        "t_c_ms": args.t_c_ms,
        "t_i_ms": args.t_i_ms,
        "t_o_ms": args.t_o_ms,
        "first_recompute_layer": args.first_recompute_layer,
        "main_header_tokens": args.main_header_tokens,
        "main_body_tokens": args.main_body_tokens,
        "main_target_rho": args.main_target_rho,
        "tail_tokens": NON_PREFIX_TAIL_TOKENS,
        "header_tokens_choices": args.header_tokens_choices,
        "body_tokens_choices": args.body_tokens_choices,
        "target_rho_choices": args.target_rho_choices,
        "length_sweep_rho": args.length_sweep_rho,
        "max_segment_chunk_tokens": args.max_segment_chunk_tokens,
        "pressure_filler_head_tokens": args.pressure_filler_head_tokens,
        "pressure_filler_body_tokens": args.pressure_filler_body_tokens,
        "repeats_per_setting": args.repeats,
        "warmup_passes_per_setting": WARMUP_PASSES_PER_SETTING,
        "runner_git_sha": args.runner_git_sha,
        "image_digest": args.image_digest,
        "scheduler": "S0 LRU",
        "tier": "GPU-only",
        "prefetch": False,
        "accuracy_metric": False,
    }


def register_body_chunks(
    base_url: str,
    *,
    head_ids: Sequence[int],
    shared_body_ids: Sequence[int],
    tail_ids: Sequence[int],
    hash_prefix: str,
    model_fingerprint: str,
    cache_dtype: str,
    max_chunk_tokens: int,
    expected_cached_tokens: int,
    label: str,
) -> dict[str, Any]:
    """Register ``shared_body_ids`` as one INDEPENDENT ``/generate``
    register call PER ``<= max_chunk_tokens`` chunk -- never one
    oversized call spanning the entire body, even when the resulting
    segments are individually ``<= max_chunk_tokens``.

    This is the fix for a real SM75 OOM: the previous design sent the
    ENTIRE ``head_ids + shared_body_ids + tail_ids`` prompt (e.g.
    ~1089 tokens for a body=1024 setting) through ONE forward pass,
    with ``chunk_offsets``/``body_segments_for_hash`` only bounding the
    STORED segment sizes (``<= max_chunk_tokens`` each) -- the call's
    own live/transient per-request KV footprint during that one forward
    pass was bounded only by the full, un-chunked body length, not by
    ``max_chunk_tokens``. With many pressure-filler objects and/or a
    large body this uncapped per-call peak OOM'd the 8GB SM75 GPU
    (observed on a real run at body=1024's register-raw step).

    Each chunk's own independent call instead posts a SHORT prompt --
    ``head_ids + this_chunk's_body_slice + tail_ids``, bounded at
    ``len(head_ids) + max_chunk_tokens + len(tail_ids)`` regardless of
    how long the logical body is -- carrying exactly ONE segment with
    ``target_start = len(head_ids)`` (this chunk's own local body-start
    position within its own short prompt -- identical for EVERY chunk
    index, since every chunk's own prompt starts fresh with the same
    head immediately followed by that chunk's own body slice) and
    ``content_hash = f"{hash_prefix}:chunk{index}"`` -- the SAME
    per-chunk hash convention ``body_segments_for_hash`` already uses
    for reuse's own (unchanged, full-prompt, multi-segment) call. Since
    ``manager.store``'s lookup key (``_segment_key``, see
    ``approx_kv/runtime.py``) is built purely from ``content_hash`` +
    a hash of the token CONTENT + ``token_count`` + ``model_fingerprint``
    + ``cache_dtype`` -- never from any absolute prompt position --
    reuse's existing per-chunk lookup for chunk N still resolves this
    exact registered entry regardless of what absolute prompt position
    THIS call itself used to compute it.

    This deliberately places every chunk index > 0 at a SMALLER
    absolute prompt position at register time than that same chunk's
    eventual reuse-time target position -- an intentional, safe
    mismatch, not a bug. ``restore_request_prefix_cachetune`` computes
    a per-segment ``rope_delta = overlap_start - source_position`` (see
    ``cachetune/runtime.py``) and, whenever that delta is nonzero,
    applies a full relative RoPE rotation of exactly that delta (see
    ``RadixKVTransferBackend._rotate_all_copied_keys`` -- a standard
    ``cos(delta * inv_freq)``/``sin(delta * inv_freq)`` relative
    rotation, mathematically correct for any integer delta magnitude),
    PROVIDED a real (non-dummy, nonzero-``rotary_dim``) RoPE config is
    bound. ``resolve_model_rope_config`` (``approx_kv/radix_backend.py``)
    guarantees exactly that for the Qwen2/Qwen3 architecture family this
    canary's SM75 target always uses, bound once at ``create_tree_cache``
    time (see ``kv_cache_builder.py``) -- the fix for an earlier real
    "second chunk nonzero delta falls back to dense" bug, which was
    caused by that binding never running at all (``rope_config`` stuck
    at the ``rotary_dim=0`` dummy sentinel), not by nonzero deltas being
    inherently unsupported. Chunk 0 happens to have ``rope_delta == 0``
    (its own local ``target_start`` already equals its true reuse-time
    position), but every later chunk genuinely relies on this
    relocation -- this is exactly what makes shrinking every register
    call's own transient footprint down to ``max_chunk_tokens`` (rather
    than the full body) safe.

    Every chunk reports the SAME ``expected_cached_tokens`` (asserted
    per chunk via ``require_cached_tokens``, never merely assumed): a
    REGISTER request never writes into the exact radix tree (see that
    function's own docstring), so every chunk's own short prompt gets
    the identical leading-``head_ids`` exact-match result, independent
    of chunk index or body length.

    Returns ``{"total_ms": float, "chunk_count": int, "cached_tokens":
    int}``: ``total_ms`` is the SUM of every chunk's own genuine
    streaming TTFT -- this body's total registration cost, however many
    independent calls it took (replaces what used to be a single call's
    own ms); ``chunk_count`` is
    ``len(chunk_offsets(len(shared_body_ids), max_chunk_tokens))``;
    ``cached_tokens`` is simply ``expected_cached_tokens`` (already
    proven true for every chunk by the per-chunk assertion above).
    """
    head_ids = tuple(int(token) for token in head_ids)
    shared_body_ids = tuple(int(token) for token in shared_body_ids)
    tail_ids = tuple(int(token) for token in tail_ids)
    body_start_local = len(head_ids)
    total_ms = 0.0
    chunk_count = 0
    for index, (offset, length) in enumerate(
        chunk_offsets(len(shared_body_ids), max_chunk_tokens)
    ):
        chunk_prompt_ids = (
            head_ids + shared_body_ids[offset : offset + length] + tail_ids
        )
        chunk_label = f"{label} chunk{index}"
        response, ttft_ms = timed_post(
            base_url,
            register_generate_payload(
                input_ids=chunk_prompt_ids,
                segments=[
                    {
                        "content_hash": f"{hash_prefix}:chunk{index}",
                        "target_start": body_start_local,
                        "length": length,
                    }
                ],
                model_fingerprint=model_fingerprint,
                cache_dtype=cache_dtype,
            ),
        )
        require_finished_by_length(response, chunk_label)
        require_cached_tokens(response, expected_cached_tokens, chunk_label)
        total_ms += ttft_ms
        chunk_count += 1
    return {
        "total_ms": total_ms,
        "chunk_count": chunk_count,
        "cached_tokens": expected_cached_tokens,
    }


def register_round_setup(
    base_url: str,
    workload: NonPrefixSegmentWorkload,
    *,
    raw_hash: str,
    fresh_hash: str,
    model_fingerprint: str,
    cache_dtype: str,
    label: str,
    max_chunk_tokens: int,
) -> dict[str, Any]:
    """Seed ``workload``'s own exact-match target head (via
    ``workload.seed_prompt_ids`` -- ``target_head_ids`` PLUS an explicit
    sentinel token, never ``target_head_ids`` alone; see
    ``NonPrefixSegmentWorkload.seed_prompt_ids`` for the real SM75 bug
    this prevents), then register its raw (source-context) AND fresh
    (target-context) body segments, in that order -- this ROUND's own
    complete, self-contained setup, always completed in full while THIS
    round's own eviction pressure is still low/absent (its own flush,
    immediately before this function runs, is what guarantees that).

    This function registers BOTH raw and fresh together, as one
    indivisible setup step, rather than registering raw once per
    *setting* and re-registering only fresh on every repeat the way an
    earlier version of this script did (that earlier version's raw
    registration lived in a since-removed ``register_non_prefix_sources``
    function, called once per setting, while fresh registration lived
    inside a since-removed ``run_reuse_once`` function called once per
    repeat). That earlier split caused a real SM75 ``target_rho=2``
    ``MemoryError``: every formal repeat's fresh registration needed to
    transiently coexist with the setup's still-resident raw segment plus
    surviving pressure fillers, and register-side segment materialization
    (for BOTH raw and fresh) is NOT wired to evict exact-radix victims to
    make room for itself (unlike the reuse/repair path's own
    recovery-slot allocation, which explicitly DOES evict exact-radix
    victims before allocating -- see ``allocate_recovery_slots`` in
    ``cachetune/runtime.py`` / ``mem_cache/common/runtime.py``). Making
    raw+fresh one atomic per-ROUND setup step -- always run immediately
    after that round's own fresh flush, always before that round's own
    pressure phase -- removes the transient double-footprint entirely:
    see ``run_independent_round``'s own docstring for the complete
    per-round ordering this function is one part of.

    Both body registrations go through ``register_body_chunks`` -- one
    INDEPENDENT ``/generate`` call per ``<= max_chunk_tokens`` chunk,
    never one oversized call spanning the entire body (see that
    function's own docstring for the real SM75 OOM this avoids and why
    the resulting per-chunk RoPE-position mismatch is safe). Fresh's own
    register call never restores anything (see ``approx_kv/runtime.py``'s
    ``_register_request_segments`` -- it bails out immediately unless
    ``operation == REUSE``), so every fresh chunk's only contribution to
    ``prefix_indices`` is the plain exact-match radix hit on
    ``target_head_ids``, just seeded above: ``body_start_in_target``, not
    0 (asserted per chunk inside ``register_body_chunks``).

    Called exactly once per ROUND -- by ``run_independent_round``, for
    the discarded warmup round and independently again for every formal
    repeat -- always as that round's first substantive step, immediately
    after that round's own flush and capacity snapshot, always before
    that round's own eviction-pressure phase (if any) and before
    ``run_target_reuse`` is ever called for that same round.

    Returns a dict with ``seed_head_ms``, ``register_raw_ms``,
    ``register_fresh_ms`` (each the genuine streaming TTFT, summed
    across every chunk where applicable -- see ``register_body_chunks``)
    and ``fresh_cached_tokens`` (always ``workload.body_start_in_target``,
    already proven true per chunk by ``register_body_chunks``'s own
    assertion), for ``run_independent_round``'s own granular per-step
    timing and telemetry.
    """
    seed_response, seed_head_ms = timed_post(
        base_url, dense_generate_payload(workload.seed_prompt_ids)
    )
    require_finished_by_length(seed_response, f"{label} seed target_head")
    require_cached_tokens(seed_response, 0, f"{label} seed target_head")

    raw_chunks = register_body_chunks(
        base_url,
        head_ids=workload.source_head_ids,
        shared_body_ids=workload.shared_body_ids,
        tail_ids=workload.tail_ids,
        hash_prefix=raw_hash,
        model_fingerprint=model_fingerprint,
        cache_dtype=cache_dtype,
        max_chunk_tokens=max_chunk_tokens,
        expected_cached_tokens=0,
        label=f"{label} raw register",
    )
    register_raw_ms = raw_chunks["total_ms"]

    fresh_chunks = register_body_chunks(
        base_url,
        head_ids=workload.target_head_ids,
        shared_body_ids=workload.shared_body_ids,
        tail_ids=workload.tail_ids,
        hash_prefix=fresh_hash,
        model_fingerprint=model_fingerprint,
        cache_dtype=cache_dtype,
        max_chunk_tokens=max_chunk_tokens,
        expected_cached_tokens=workload.body_start_in_target,
        label=f"{label} fresh preparation",
    )
    register_fresh_ms = fresh_chunks["total_ms"]

    return {
        "seed_head_ms": seed_head_ms,
        "register_raw_ms": register_raw_ms,
        "register_fresh_ms": register_fresh_ms,
        "fresh_cached_tokens": fresh_chunks["cached_tokens"],
    }


def run_target_reuse(
    base_url: str,
    workload: NonPrefixSegmentWorkload,
    *,
    raw_hash: str,
    model_fingerprint: str,
    cache_dtype: str,
    label: str,
    max_chunk_tokens: int,
) -> dict[str, Any]:
    """Issue exactly one real reuse request against ``workload``'s
    already-registered raw/fresh segments -- forcing CacheTune's genuine
    repair path to run once and materialize the raw segment onto the
    device residency tier via ``ensure_device`` (see
    ``cachetune/runtime.py``).

    ASSUMES ``register_round_setup`` has already run, for this exact
    ``workload`` AND for THIS SAME ROUND (the target head, raw segment,
    AND fresh segment must already be registered/resident): this
    function performs ONLY the reuse call itself, never any
    registration of its own. This is a deliberate split from an earlier
    ``run_reuse_once`` function, which registered fresh again on every
    call and was invoked once per repeat while sharing one raw
    registration (from a since-removed ``register_non_prefix_sources``,
    called once per *setting*) across every repeat -- see
    ``register_round_setup``'s own docstring for the real SM75
    ``target_rho=2`` ``MemoryError`` that split caused and why fresh
    registration now lives in the per-round setup step instead, always
    completed before this function is ever called for that same round.

    This function is called identically for the one discarded warmup
    round and every formal repeat -- a single implementation of "reuse",
    not two independently-maintained copies -- via
    ``run_independent_round``.

    The reuse call posts the FULL ``target_prompt_ids`` in one call with
    the existing contiguous multi-segment list (see
    ``body_segments_for_hash``), never chunked the way register calls
    are: a genuine reuse/repair forward pass over the complete target
    context is exactly what this canary measures, and the register
    side's transient per-call footprint (the OOM risk chunking there
    fixes, see ``register_body_chunks``) does not apply to this single
    call.

    Returns a dict with ``reuse_ms`` (the genuine streaming TTFT),
    ``reuse_cached_tokens`` (the reuse response's own observed
    ``meta_info.cached_tokens``, proven equal to
    ``body_start_in_target + body_tokens`` by this function's own
    assertion) and ``reuse_response`` (the reuse call's parsed JSON
    body).
    """
    reuse_response, reuse_ms = timed_post(
        base_url,
        reuse_generate_payload(
            input_ids=workload.target_prompt_ids,
            segments=body_segments_for_hash(
                hash_prefix=raw_hash,
                body_start=workload.body_start_in_target,
                body_tokens=workload.body_tokens,
                max_chunk_tokens=max_chunk_tokens,
            ),
            model_fingerprint=model_fingerprint,
            cache_dtype=cache_dtype,
        ),
    )
    require_finished_by_length(reuse_response, f"{label} reuse")
    # A successful CacheTune reuse always extends prefix_indices by the
    # FULL restored body (restore_length), never just a
    # controller-ratio-scaled fraction of it -- decision.repair_tokens
    # only picks how many of those already-restored positions get a
    # genuine recompute forward pass versus a straight KV copy (see
    # cachetune/runtime.py's restore_request_prefix_cachetune). So the
    # expected cached_tokens is exact-match head (body_start_in_target)
    # PLUS the entire body (body_tokens), not head-only -- confirmed by
    # a real SM75 run reporting exactly this head+body value where an
    # earlier version of this script expected head-only.
    reuse_cached_tokens = require_cached_tokens(
        reuse_response,
        workload.body_start_in_target + workload.body_tokens,
        f"{label} reuse",
    )

    return {
        "reuse_ms": reuse_ms,
        "reuse_cached_tokens": reuse_cached_tokens,
        "reuse_response": reuse_response,
    }


def ensure_target_head_resident(
    base_url: str,
    workload: NonPrefixSegmentWorkload,
    *,
    label: str,
) -> dict[str, Any]:
    """Re-seed ``workload``'s own exact-match target head with one plain
    dense ``/generate`` request over ``workload.seed_prompt_ids`` --
    ``target_head_ids`` PLUS an explicit sentinel token, never
    ``target_head_ids`` alone; see
    ``NonPrefixSegmentWorkload.seed_prompt_ids`` for the real SM75 bug
    this prevents -- tolerant of ANY of three outcomes:

    - ``cached_tokens == 0``: full miss, the head itself was evicted by
      the pressure phase and just got recomputed and freshly
      re-inserted (``was_evicted_by_pressure=True``);
    - ``cached_tokens == len(target_head_ids)``: partial hit, the head
      itself survived pressure but the deeper sentinel-extension node
      was independently reclaimed by LRU (eviction proceeds leaf-inward,
      and the sentinel/generated-token nodes are strictly deeper/newer
      than the head node itself);
    - ``cached_tokens == len(workload.seed_prompt_ids)`` (``==
      len(target_head_ids) + len(workload.seed_sentinel_ids)``): full
      hit, the entire head+sentinel path survived pressure intact.

    These three values are always mutually distinct (``seed_sentinel_ids``
    is validated non-empty in ``NonPrefixSegmentWorkload.__post_init__``,
    guaranteeing ``len(seed_prompt_ids) > len(target_head_ids) > 0``).
    Never requiring a SPECIFIC one of these three values, only that the
    request completes successfully with one of them -- and, either way,
    what matters for downstream correctness is only whether the HEAD
    itself (not the sentinel extension) survived, so
    ``was_evicted_by_pressure`` is set from ``cached_tokens == 0`` alone
    (both other buckets mean the head survived).

    This guards a real correctness risk this script's own "register
    sources before pressure" ordering introduces (see
    ``run_independent_round``'s docstring for that ordering and why it
    is otherwise mandatory): the target head is seeded by
    ``register_round_setup`` BEFORE any eviction-pressure filler is sent
    for THIS round, making it the OLDEST entry in the exact radix tree at
    the moment this round's own pressure begins -- a plausible
    LRU-eviction candidate for any ``target_rho > 1`` setting (genuine
    pool overflow is exactly what such a setting is constructed to
    create). If the head were evicted and nothing re-seeded it, this
    round's own subsequent ``run_target_reuse`` call would fail loudly:
    register/reuse requests always force
    ``skip_radix_cache_insert=True`` (see this module's own docstring),
    so ONLY a plain dense request like this one can ever re-populate the
    exact radix tree for that head again -- neither ``register_fresh``'s
    own ``expected_cached_tokens=body_start_in_target`` assertion nor
    reuse's own full-restore assertion could recover from a silently
    evicted head, permanently breaking the "any gap forces
    dense-fallback" invariant for the rest of this round.

    Called once per ROUND whose ``target_rho`` is set -- by
    ``run_independent_round``, for the discarded warmup round and
    independently again for every formal repeat -- positioned
    immediately after THAT round's own pressure-filler phase completes
    and before that same round's own ``run_target_reuse`` call begins:
    nothing else grows the exact radix tree within a round after this
    point (both register and reuse always skip radix insertion, see
    above), so one guard call per round is sufficient to protect that
    round's own head for the remainder of that round. Never called once
    per setting: each round re-seeds and re-pressures its own head
    completely independently of every other round (see
    ``run_independent_round``'s docstring for why no round may ever
    depend on another round's own registration or residency).

    This step is NOT part of the paper's own repair-controller design;
    it is an additional defensive measure this script adds because
    genuine LRU eviction pressure (the entire point of ``target_rho >
    1``) sent immediately after seeding the head creates exactly this
    risk. Returns ``{"ttft_ms": float, "cached_tokens": int,
    "was_evicted_by_pressure": bool}`` (the last key set from
    ``cached_tokens == 0``) for caller-side telemetry -- never silently
    discarded.
    """
    response, ttft_ms = timed_post(
        base_url, dense_generate_payload(workload.seed_prompt_ids)
    )
    require_finished_by_length(response, label)
    cached_tokens = int(response["meta_info"]["cached_tokens"])
    expected_head_tokens = len(workload.target_head_ids)
    expected_seeded_tokens = len(workload.seed_prompt_ids)
    if cached_tokens not in (0, expected_head_tokens, expected_seeded_tokens):
        raise RuntimeError(
            f"{label}: expected cached_tokens to be one of 0 (head was "
            "evicted by pressure and just got recomputed), "
            f"{expected_head_tokens} (head survived pressure, sentinel "
            f"extension independently evicted), or {expected_seeded_tokens} "
            "(head and sentinel both survived pressure), observed "
            f"{cached_tokens} instead -- this indicates a corrupted or "
            "unexpected partial radix match, not a clean hit-or-miss"
        )
    return {
        "ttft_ms": ttft_ms,
        "cached_tokens": cached_tokens,
        "was_evicted_by_pressure": cached_tokens == 0,
    }


def register_eviction_pressure_objects(
    base_url: str,
    workloads: Sequence[NonPrefixSegmentWorkload],
    *,
    label: str,
    capacity_tokens: int,
    target_rho: float,
    already_pinned_tokens: int = 0,
) -> dict[str, Any]:
    """Send every eviction-pressure filler object in ``workloads`` (see
    ``build_eviction_pressure_workloads``) as a PLAIN, ordinary dense
    ``/generate`` request -- carrying NO ``approx_kv`` custom_params
    metadata at all -- so each filler's KV lands in the server's
    ordinary, LRU-evictable exact radix tree, never in
    ``ApproxKVManager``'s own segment store.

    THIS IS A DELIBERATE FIX for a real, previously-observed SM75 bug at
    ``target_rho=2``: an earlier version of this function ran every
    filler through a full seed-head + raw-register + fresh-register +
    reuse CacheTune cycle. Register/reuse requests always set
    ``schedule_batch.Req.skip_radix_cache_insert=True`` whenever
    ``approx_kv_metadata`` is present, and their raw/fresh bodies are
    captured into ``ApproxKVManager``'s own segment store (see
    ``approx_kv/runtime.py``) -- a structure the Radix LRU eviction
    policy has no knowledge of and cannot reclaim AT ALL. With enough
    fillers materialized that way (observed: from filler[11] onward at
    ``target_rho=2`` on a real SM75 run), the pool filled with
    permanently un-evictable segments, leaving no room for the setting's
    own target recovery-slot allocation -- its own reuse call then only
    restored the exact-match head (``cached_tokens`` reported head-only,
    never head+body). Plain dense fillers, by contrast, populate the
    exact radix tree exactly like any other ordinary request and are
    fully subject to normal LRU eviction -- genuine, realistic cache
    pressure the setting's own recovery allocation CAN reclaim from,
    exactly the way a real deployment's unrelated concurrent traffic
    would behave (the same "R1"/"R4"-round plain-dense-filler
    methodology already used by ``research/epic-legolink``'s own
    ``run_phase4_epic_pressure.py``).

    Exactly one request per filler: ``dense_generate_payload`` over that
    filler's own ``target_prompt_ids`` (head + body + tail, unique per
    filler via ``build_eviction_pressure_workloads``'s pairwise
    first-token isolation). ``source_head_ids``/``source_prompt_ids``/
    ``fresh_prompt_ids`` are never sent anywhere for a filler -- they
    exist on ``NonPrefixSegmentWorkload`` purely to satisfy that
    dataclass's own structural invariants (shared with the main
    setting's genuine repair workload), unused dead weight for a filler
    specifically. Every filler's first (and only) appearance must
    report ``cached_tokens=0``: pairwise isolation plus this setting's
    own just-completed flush together guarantee no accidental exact-
    prefix hit is possible.

    ``capacity_tokens`` is the fixed, idle-pool capacity reference
    established once by the caller (immediately after its own flush, via
    ``usable_kv_capacity_tokens``) and ``target_rho`` is the nominal
    ratio ``workloads`` was reverse-sized for (see
    ``eviction_pressure_filler_count_for_rho``) -- both are used here to
    report the genuine, *sampled* ``observed_rho`` (resident occupancy:
    ``sglang:kv_used_tokens`` PLUS ``sglang:kv_evictable_tokens``, never
    ``kv_used_tokens`` alone) immediately after this pressure
    phase completes, alongside the nominal target, AND to gate the
    ``evicted_tokens_total_delta`` assertion below. ``already_pinned_tokens``
    (default 0) is THIS ROUND's own raw+fresh source-registration
    footprint, ALREADY resident by the time this phase runs (see
    ``run_independent_round``'s "register sources before pressure"
    ordering) -- it is permanently non-evictable for the remainder of
    this round (register/reuse never populate the exact radix tree
    fillers land in, but the raw/fresh segments themselves live in
    ``ApproxKVManager``'s own segment store, distinct from -- and
    additive with -- the capacity these plain-dense fillers compete for)
    and therefore reduces the TRUE evictable headroom available to
    fillers below ``capacity_tokens`` by that same amount. Every round
    measures its OWN ``already_pinned_tokens`` freshly (never inherited
    from any other round): see ``run_independent_round``'s own docstring
    for why no round may ever depend on another round's own
    registration.

    Raises immediately, before sending any filler, if
    ``already_pinned_tokens >= capacity_tokens``: the setup's own
    footprint alone already consumes the entire measured pool, an
    unrecoverable misconfiguration for this phase (there would be zero
    room left in which fillers, or anything else, could ever coexist).

    Raises immediately if ``total_pressure_tokens`` (see
    ``eviction_pressure_total_tokens``) -- the fillers' own combined
    nominal footprint ALONE -- already exceeds ``capacity_tokens -
    already_pinned_tokens`` (the TRUE evictable headroom remaining after
    the setting's own already-pinned setup footprint is set aside) yet
    the live ``sglang:evicted_tokens_total`` Prometheus counter did not
    increase while registering them: by construction, that combination
    means later fillers necessarily had to evict earlier ones just to
    fit, all within this phase alone -- a zero delta despite that would
    mean capacity accounting or eviction itself is broken, so this
    pressure phase's own "genuine device pressure" claim must not be
    silently trusted. (When ``total_pressure_tokens <= capacity_tokens -
    already_pinned_tokens`` -- e.g. ``target_rho <= 1`` -- no such
    assertion is made: the fillers may legitimately all coexist without
    any eviction at all, exactly as documented in this module's own
    "Eviction-pressure phase" docstring section.)

    Also raises immediately if
    ``sglang:approx_kv_dense_fallback_total`` increased during this
    phase: a plain dense filler request carries no ``approx_kv``
    metadata whatsoever and should NEVER be able to move this
    CacheTune-reuse-specific counter at all -- a nonzero delta here
    would mean something unexpected is happening to the server's
    approx_kv reuse path during what must be a purely-dense phase, a
    real defect worth catching immediately rather than silently ignored.

    Returns a dict with ``object_count``, ``total_pressure_tokens``
    (see ``eviction_pressure_total_tokens``), ``target_rho`` (the
    nominal ask), ``capacity_tokens`` (the fixed reference),
    ``already_pinned_tokens`` (surfaced verbatim, for transparency about
    how much of ``capacity_tokens`` this phase treated as already
    unavailable to fillers), ``observed_rho_after_pressure`` (the
    genuine sampled ratio), ``evicted_tokens_total_delta`` (the genuine,
    real evidence that device-pool eviction actually happened during
    THIS pressure phase alone -- may legitimately be 0 if the configured
    pressure was not large enough to evict anything, and this is
    reported honestly rather than hidden), and
    ``dense_fallback_total_delta`` (always 0, given the raise above,
    kept for output-schema transparency), plus the raw
    ``metrics_before``/``metrics_after`` snapshots the deltas above were
    computed from (surfaced verbatim for downstream debugging, exactly
    like ``run_non_prefix_setting``'s own ``metrics_before``/
    ``metrics_after`` keys).
    """
    if already_pinned_tokens < 0:
        raise ValueError(
            f"already_pinned_tokens must be >= 0, got {already_pinned_tokens}"
        )
    if already_pinned_tokens >= capacity_tokens:
        raise ValueError(
            f"{label}: already_pinned_tokens={already_pinned_tokens} already "
            f"meets or exceeds capacity_tokens={capacity_tokens} -- this "
            "round's own raw+fresh source-registration footprint alone "
            "consumes the entire measured pool, leaving no room for any "
            "eviction-pressure filler (or anything else) to ever coexist"
        )
    effective_available_tokens = capacity_tokens - already_pinned_tokens

    metrics_before = metric_snapshot(base_url)
    for index, filler_workload in enumerate(workloads):
        response, _ = timed_post(
            base_url, dense_generate_payload(filler_workload.target_prompt_ids)
        )
        require_finished_by_length(response, f"{label} pressure-filler[{index}]")
        require_cached_tokens(response, 0, f"{label} pressure-filler[{index}]")
        time.sleep(0.1)
    metrics_after = metric_snapshot(base_url)

    dense_fallback_delta = metric_delta(
        metrics_before, metrics_after, "sglang:approx_kv_dense_fallback_total"
    )
    if dense_fallback_delta != 0:
        raise RuntimeError(
            f"{label}: {len(workloads)} plain-dense eviction-pressure "
            f"filler object(s) produced a nonzero dense_fallback delta "
            f"of {dense_fallback_delta} -- a plain dense request carries "
            "no approx_kv metadata and should never be able to move "
            "this CacheTune-reuse-specific counter at all; something "
            "unexpected touched the approx_kv reuse path during what "
            "must be a purely-dense pressure phase"
        )

    total_pressure_tokens = eviction_pressure_total_tokens(workloads)
    evicted_tokens_total_delta = metric_delta(
        metrics_before, metrics_after, "sglang:evicted_tokens_total"
    )
    if (
        total_pressure_tokens > effective_available_tokens
        and evicted_tokens_total_delta <= 0
    ):
        raise RuntimeError(
            f"{label}: {len(workloads)} eviction-pressure filler object(s) "
            f"declare {total_pressure_tokens} nominal tokens against a "
            f"pool with only {effective_available_tokens} tokens of TRUE "
            f"evictable headroom ({capacity_tokens} usable capacity minus "
            f"{already_pinned_tokens} already pinned by this setting's own "
            "raw+fresh source registration) -- later fillers necessarily "
            "had to evict earlier ones just to fit, yet "
            "sglang:evicted_tokens_total did not increase "
            f"({evicted_tokens_total_delta}) while registering them. "
            "Either usable_kv_capacity_tokens mis-measured this pool's "
            "real capacity, or LRU eviction is not actually reclaiming "
            "these plain dense filler objects as intended -- this "
            "pressure phase cannot be trusted to have genuinely "
            "stressed the pool"
        )

    return {
        "object_count": len(workloads),
        "total_pressure_tokens": total_pressure_tokens,
        "target_rho": target_rho,
        "capacity_tokens": capacity_tokens,
        "already_pinned_tokens": already_pinned_tokens,
        "observed_rho_after_pressure": observed_rho(
            metrics_after, capacity_tokens=capacity_tokens
        ),
        "evicted_tokens_total_delta": evicted_tokens_total_delta,
        "dense_fallback_total_delta": dense_fallback_delta,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    }


def run_independent_round(
    base_url: str,
    tokenizer: Any,
    workload: NonPrefixSegmentWorkload,
    *,
    raw_hash: str,
    fresh_hash: str,
    model_fingerprint: str,
    cache_dtype: str,
    label: str,
    max_chunk_tokens: int,
    target_rho: float | None,
    pressure_filler_head_tokens: int,
    pressure_filler_body_tokens: int,
) -> dict[str, Any]:
    """Run ONE fully self-contained, independent measurement round: flush
    -> capacity snapshot -> this round's own complete setup (seed head +
    register raw + register fresh, via ``register_round_setup``) ->
    [if ``target_rho`` is set: this round's own freshly reverse-computed
    eviction-pressure phase, sized from THIS round's own real post-setup
    footprint, plus a target-head re-seed guard] -> this round's own
    reuse call (via ``run_target_reuse``) -> this round's own post-round
    metrics/rho/eviction-delta.

    This is the SINGLE primitive ``run_non_prefix_setting`` uses
    identically for its one discarded warmup round and each of its
    ``repeats`` formal rounds -- never a setup-once-then-repeat design.
    This is a deliberate architecture, fixing a real SM75
    ``target_rho=2`` bug: an earlier design (this function did not
    exist; its callers had ``register_non_prefix_sources`` register raw
    ONCE per *setting* and a since-removed ``run_reuse_once`` function
    re-register fresh on every repeat, reusing that one raw registration
    across the discarded warmup and every formal repeat) produced two
    consecutive ``MemoryError``s on formal fresh-register calls at
    ``target_rho=2``, followed by target reuse OOM: each repeat's fresh
    registration needed to transiently coexist with the setup's
    still-resident raw segment plus surviving pressure fillers, and
    register-side segment materialization (for BOTH raw and fresh) is
    NOT wired to evict exact-radix victims to make room for itself
    (unlike the reuse/repair path's own recovery-slot allocation, which
    explicitly DOES evict -- see ``allocate_recovery_slots`` in
    ``cachetune/runtime.py`` / ``mem_cache/common/runtime.py``). Making
    every round -- warmup included -- fully independent (its own flush,
    its own raw+fresh registration, its own pressure phase, its own
    reuse, NEVER reusing another round's already-registered segments or
    surviving pressure fillers) removes that transient double-footprint
    entirely: every round's raw+fresh registration always runs against a
    genuinely fresh, just-flushed idle pool.

    The flush-and-sentinel-refresh is this function's OWN first action,
    every single call -- not merely the caller's responsibility, and
    not merely once per setting the way an earlier design's
    ``run_non_prefix_setting`` did it. This is safe (never harmful,
    unlike flushing BETWEEN steps WITHIN the same round would be)
    precisely because it runs before any registration this round
    performs: see ``flush_exact_radix_cache``'s own docstring for why
    every round -- not just the first -- must start this way now, and
    why the resulting cross-round Prometheus telemetry deltas remain
    mathematically sound regardless of how many independent flushes
    separate them (Counters are monotonic and unaffected by flush; only
    the ``sglang:kv_used_tokens`` Gauge resets, which is exactly what
    lets every round measure its OWN idle capacity and pinned footprint
    freshly -- see ``flush_and_force_gauge_refresh``'s own docstring for
    why a bare flush alone is not enough to guarantee that Gauge itself
    reads fresh, and the real SM75 cross-setting bug fixed by never
    reading it without an intervening real request first).

    Immediately after that flush-and-sentinel-refresh, this round's own
    now-genuinely-fresh ``/metrics`` snapshot gives this round's own
    ``capacity_tokens`` (via ``usable_kv_capacity_tokens``) -- a fixed
    reference used for this round's own ``observed_rho`` calculations.
    A SECOND, bare flush immediately follows (clearing away the
    gauge-refresh sentinel's own tiny resident footprint, never leaving
    it to risk an unwanted exact-match collision against this round's
    own upcoming head-seed request -- see
    ``flush_and_force_gauge_refresh``'s own docstring for why this
    second flush needs no sentinel of its own). ``register_round_setup``
    then runs -- this round's own seed-head + raw-register +
    fresh-register setup, all together -- while the pool is still at (or
    near) that same idle baseline.

    If ``target_rho`` is not ``None``, a SECOND ``/metrics`` snapshot,
    taken immediately after THIS round's setup completes, gives this
    round's own ``already_pinned_tokens`` (via ``metric_delta`` against
    this round's own start-of-round snapshot) -- the setup's real,
    measured contribution to ``sglang:kv_used_tokens``, for THIS round
    alone, never inherited from any other round. The filler object count
    is reverse-computed from ``target_rho`` against this round's own
    ``capacity_tokens`` net of that already-pinned footprint (see
    ``eviction_pressure_filler_count_for_rho``), and a fresh pressure
    phase of plain dense filler requests is sent (see
    ``register_eviction_pressure_objects``). ``validate_pairwise_head_isolation``
    runs first (before any network call) to guard against a filler's
    dense-seeded target head colliding with this round's own head or with
    another filler's head in the live exact radix tree. Because this
    round's own target head is seeded (by ``register_round_setup``)
    BEFORE any filler, it is the OLDEST exact-radix entry once THIS
    round's own pressure begins -- ``ensure_target_head_resident`` runs
    immediately after, to guard against exactly that (see its own
    docstring).

    ``run_target_reuse`` then issues this round's own single reuse call
    against the just-registered (this round, not any other round's)
    raw/fresh segments.

    Returns a dict with ``seed_head_ms``, ``register_raw_ms``,
    ``register_fresh_ms``, ``fresh_cached_tokens`` (from
    ``register_round_setup``), ``reuse_ms``, ``reuse_cached_tokens``
    (from ``run_target_reuse``), ``capacity_tokens``,
    ``already_pinned_tokens`` (``None`` when ``target_rho`` is ``None``),
    ``head_reseed_after_pressure`` (``None`` when ``target_rho`` is
    ``None``, otherwise ``ensure_target_head_resident``'s own returned
    dict), ``pressure_phase`` (``None`` when ``target_rho`` is ``None``,
    otherwise ``register_eviction_pressure_objects``'s own returned
    telemetry dict), ``observed_rho_after_target`` (this round's own
    genuine, sampled resident-occupancy ratio -- see ``observed_rho``,
    ``sglang:kv_used_tokens`` PLUS ``sglang:kv_evictable_tokens`` -- right
    after this round's own reuse call completes), ``peak_rho_observed``
    (the greater of that and this round's own pressure phase's own
    ``observed_rho_after_pressure``, or just the former when no pressure
    phase ran this round), ``evicted_tokens_total_delta`` (this round's
    own cumulative real ``sglang:evicted_tokens_total`` delta, spanning
    from this round's own flush through the end of this round's own
    reuse call), and ``metrics_at_round_start``/``metrics_after_round``
    (this round's own raw snapshots, for the caller's own cross-round
    aggregation) -- everything ``run_non_prefix_setting`` needs to
    aggregate one setting's worth of otherwise fully independent rounds.
    """
    # Flush, THEN force one real scheduler iteration via a fixed
    # sentinel request, THEN snapshot -- never flush-then-snapshot
    # directly (see flush_and_force_gauge_refresh's own docstring for
    # the real SM75 body-length-sweep bug this fixes: a bare
    # flush-then-snapshot could carry over a PREVIOUS setting's own
    # stale sglang:kv_used_tokens gauge reading verbatim, producing a
    # structurally negative already_pinned_tokens for the NEXT
    # setting/round below).
    metrics_at_round_start = flush_and_force_gauge_refresh(
        base_url, tokenizer, label=f"{label} round-start"
    )
    capacity_tokens = usable_kv_capacity_tokens(metrics_at_round_start)
    # A second, bare flush clears away the gauge-refresh sentinel's own
    # tiny resident footprint (moved to sglang:kv_evictable_tokens once
    # its own generation completed, never sglang:kv_used_tokens -- see
    # flush_and_force_gauge_refresh's own docstring) so
    # register_round_setup's own seed call immediately below starts
    # from a genuinely, fully empty exact radix tree -- never at risk
    # of an unwanted exact-match collision against the sentinel's own
    # token(s), which would corrupt that seed call's own
    # ``require_cached_tokens(..., 0, ...)`` check. No sentinel is
    # needed for THIS flush: nothing reads ``/metrics`` again until
    # AFTER register_round_setup's own real dense requests (starting
    # with that same seed call) naturally force a fresh scheduler
    # iteration.
    flush_exact_radix_cache(base_url)

    # This round's own complete setup -- raw AND fresh together, always
    # finished in full BEFORE any pressure filler for THIS round. Never
    # call register_eviction_pressure_objects before this, and never
    # reuse a different round's own setup.
    setup_result = register_round_setup(
        base_url,
        workload,
        raw_hash=raw_hash,
        fresh_hash=fresh_hash,
        model_fingerprint=model_fingerprint,
        cache_dtype=cache_dtype,
        label=f"{label} setup",
        max_chunk_tokens=max_chunk_tokens,
    )

    pressure_phase: dict[str, Any] | None = None
    already_pinned_tokens: int | None = None
    head_reseed_after_pressure: dict[str, Any] | None = None
    if target_rho is not None:
        # Real, measured (never estimated) THIS ROUND's own post-setup
        # footprint: what register_round_setup itself already consumed
        # of the pool THIS round, sampled AFTER it completes -- this is
        # what lets fillers be sized against this round's own true
        # resident footprint rather than a blind guess or a stale value
        # inherited from a previous round.
        metrics_after_setup = metric_snapshot(base_url)
        already_pinned_tokens = round(
            metric_delta(
                metrics_at_round_start,
                metrics_after_setup,
                "sglang:kv_used_tokens",
            )
        )
        tokens_per_filler = pressure_filler_head_tokens + pressure_filler_body_tokens
        filler_count = eviction_pressure_filler_count_for_rho(
            target_rho=target_rho,
            usable_capacity_tokens=capacity_tokens,
            tokens_per_filler=tokens_per_filler,
            already_pinned_tokens=already_pinned_tokens,
        )
        pressure_workloads = build_eviction_pressure_workloads(
            tokenizer,
            object_count=filler_count,
            body_tokens=pressure_filler_body_tokens,
            head_tokens=pressure_filler_head_tokens,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt_prefix=f"{CACHE_SALT}-pressure-{label}",
            reserved_first_token_ids=frozenset({workload.target_head_ids[0]}),
        )
        validate_pairwise_head_isolation(
            [
                (f"pressure-filler[{index}]", filler.target_head_ids)
                for index, filler in enumerate(pressure_workloads)
            ]
            + [(label, workload.target_head_ids)]
        )
        pressure_phase = register_eviction_pressure_objects(
            base_url,
            pressure_workloads,
            label=label,
            capacity_tokens=capacity_tokens,
            target_rho=target_rho,
            already_pinned_tokens=already_pinned_tokens,
        )
        # This round's own target head was seeded above by
        # register_round_setup, BEFORE any filler -- see
        # ensure_target_head_resident's own docstring for why it is now
        # the oldest exact-radix entry and a plausible LRU-eviction
        # candidate once fillers accumulate past capacity. One guard
        # call per round is sufficient (nothing else grows the exact
        # radix tree for the remainder of this round).
        head_reseed_after_pressure = ensure_target_head_resident(
            base_url, workload, label=f"{label} post-pressure head re-seed"
        )

    reuse_result = run_target_reuse(
        base_url,
        workload,
        raw_hash=raw_hash,
        model_fingerprint=model_fingerprint,
        cache_dtype=cache_dtype,
        label=label,
        max_chunk_tokens=max_chunk_tokens,
    )

    metrics_after_round = metric_snapshot(base_url)
    observed_rho_after_target = observed_rho(
        metrics_after_round, capacity_tokens=capacity_tokens
    )
    peak_rho_observed = (
        max(observed_rho_after_target, pressure_phase["observed_rho_after_pressure"])
        if pressure_phase is not None
        else observed_rho_after_target
    )

    return {
        "seed_head_ms": setup_result["seed_head_ms"],
        "register_raw_ms": setup_result["register_raw_ms"],
        "register_fresh_ms": setup_result["register_fresh_ms"],
        "fresh_cached_tokens": setup_result["fresh_cached_tokens"],
        "reuse_ms": reuse_result["reuse_ms"],
        "reuse_cached_tokens": reuse_result["reuse_cached_tokens"],
        "capacity_tokens": capacity_tokens,
        "already_pinned_tokens": already_pinned_tokens,
        "head_reseed_after_pressure": head_reseed_after_pressure,
        "pressure_phase": pressure_phase,
        "observed_rho_after_target": observed_rho_after_target,
        "peak_rho_observed": peak_rho_observed,
        "evicted_tokens_total_delta": metric_delta(
            metrics_at_round_start,
            metrics_after_round,
            "sglang:evicted_tokens_total",
        ),
        "metrics_at_round_start": metrics_at_round_start,
        "metrics_after_round": metrics_after_round,
    }


def run_non_prefix_setting(
    *,
    base_url: str,
    tokenizer: Any,
    workload: NonPrefixSegmentWorkload,
    raw_hash: str,
    fresh_hash: str,
    model_fingerprint: str,
    cache_dtype: str,
    repeats: int,
    label: str,
    max_chunk_tokens: int,
    target_rho: float | None,
    pressure_filler_head_tokens: int,
    pressure_filler_body_tokens: int,
) -> dict[str, Any]:
    """Run one discarded warmup round, then ``repeats`` (``>= 1``) formal
    rounds, all via ``run_independent_round`` -- and aggregate their
    results into one setting-level summary. THIS FUNCTION ITSELF
    performs NO flushing, registration, pressure-sizing, or reuse calls
    directly any more: every one of those steps now lives inside
    ``run_independent_round``, called identically for the warmup round
    and each formal repeat, so that every round is fully independent of
    every other round (see that function's own docstring for why).

    The shared measurement routine used by the main CacheTune setting,
    every shape-sweep point, and every rho-sweep point.

    THIS AGGREGATOR SHAPE -- rather than an earlier design where this
    function itself did one setup (source registration) followed by a
    pressure phase followed by a warmup-then-repeats reuse loop, sharing
    that ONE setup across every round -- is a deliberate fix for a real
    SM75 ``target_rho=2`` bug: sharing one raw registration across every
    repeat while re-registering fresh separately, per repeat, required
    each repeat's fresh registration to transiently coexist with the
    shared setup's still-resident raw segment plus surviving pressure
    fillers, and register-side segment materialization is not wired to
    evict exact-radix victims to make room for itself -- this produced
    two consecutive ``MemoryError``s on formal fresh-register calls,
    followed by target reuse OOM. Making every round -- including warmup
    -- a fully independent unit (see ``run_independent_round``) removes
    that transient double-footprint entirely: NO state of any kind
    (registered segments, pressure fillers, resident head) is ever
    shared or reused between rounds. This is why this function itself
    is now a thin aggregator, never itself the owner of any round's
    setup/pressure/reuse steps.

    Every register step (raw AND fresh, inside every round's own
    ``register_round_setup``) goes through ``register_body_chunks`` --
    one INDEPENDENT ``/generate`` call per ``<= max_chunk_tokens`` chunk,
    never one oversized call spanning the entire body -- see that
    function's own docstring for the real SM75 register-time OOM this
    avoids. Every reuse call, warmup and formal alike, is deliberately
    NOT chunked this way: it always posts the complete
    ``target_prompt_ids`` in one call (see ``body_segments_for_hash``),
    since a genuine full-context reuse/repair forward pass is exactly
    what this canary measures.

    Returns a dict with ``seed_head_ms``, ``register_raw_ms`` (both from
    the LAST formal round, for backward-compatible single-value
    reporting -- every round's own value is also available per-entry in
    ``rounds``, see below), ``fresh_raw_samples``, ``reuse_raw_samples``
    (each a list of ``{"ttft_ms": float, "cached_tokens": int}`` records,
    one per formal repeat, pairing every repeat's genuine streaming TTFT
    with the server-reported ``meta_info.cached_tokens`` from that exact
    same call), ``fresh_ms_samples``, ``reuse_ms_samples``,
    ``combined_ms_samples`` (the ``ttft_ms``-only projections of the
    above, kept for existing consumers), ``observed_cached_tokens_per_call``
    (the reuse leg's ``cached_tokens`` projection, unchanged),
    ``metrics_before`` (the FIRST formal round's own
    ``metrics_at_round_start``)/``metrics_after`` (the LAST formal
    round's own ``metrics_after_round``) -- excluding the warmup round's
    own telemetry contribution exactly as before, since the warmup round
    (and its own flush) always completes entirely before the first
    formal round's own ``metrics_at_round_start`` snapshot is taken --
    ``capacity_tokens``/``already_pinned_tokens``/
    ``head_reseed_after_pressure``/``observed_rho_after_target``/
    ``pressure_phase`` (all from the LAST formal round, single-value
    reporting), ``peak_rho_observed`` (the MAXIMUM across every formal
    round, not just the last), ``pressure_and_target_evicted_tokens_total_delta``
    (the cumulative real ``sglang:evicted_tokens_total`` delta spanning
    from the first formal round's own start through the last formal
    round's own end -- mathematically equal to the sum of every formal
    round's own ``evicted_tokens_total_delta``, since
    ``sglang:evicted_tokens_total`` is a monotonic Prometheus Counter
    unaffected by the flush at the start of every round -- see
    ``flush_exact_radix_cache``'s own docstring), and ``rounds`` (the
    complete list of every formal round's own raw ``run_independent_round``
    result dict, in order -- full per-round transparency for pressure
    sizing, capacity, and eviction telemetry that is no longer
    meaningfully summarizable as a single setting-wide value now that
    every round independently re-sizes and re-sends its own pressure
    phase; the discarded warmup round's own result is never included
    here, exactly like its telemetry is excluded from every other
    aggregate above).
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    # Discarded warmup: one FULLY INDEPENDENT round (its own flush, its
    # own setup, its own pressure phase, its own reuse) -- never merely
    # a fresh-register + reuse cycle reusing an earlier round's own
    # setup. See run_independent_round's own docstring for why no round
    # may ever depend on another round's own registration or residency.
    run_independent_round(
        base_url,
        tokenizer,
        workload,
        raw_hash=raw_hash,
        fresh_hash=fresh_hash,
        model_fingerprint=model_fingerprint,
        cache_dtype=cache_dtype,
        label=f"{label} warmup (discarded)",
        max_chunk_tokens=max_chunk_tokens,
        target_rho=target_rho,
        pressure_filler_head_tokens=pressure_filler_head_tokens,
        pressure_filler_body_tokens=pressure_filler_body_tokens,
    )

    rounds: list[dict[str, Any]] = []
    for _ in range(repeats):
        round_result = run_independent_round(
            base_url,
            tokenizer,
            workload,
            raw_hash=raw_hash,
            fresh_hash=fresh_hash,
            model_fingerprint=model_fingerprint,
            cache_dtype=cache_dtype,
            label=label,
            max_chunk_tokens=max_chunk_tokens,
            target_rho=target_rho,
            pressure_filler_head_tokens=pressure_filler_head_tokens,
            pressure_filler_body_tokens=pressure_filler_body_tokens,
        )
        rounds.append(round_result)
        time.sleep(0.1)

    fresh_raw_samples = [
        {
            "ttft_ms": round_result["register_fresh_ms"],
            "cached_tokens": round_result["fresh_cached_tokens"],
        }
        for round_result in rounds
    ]
    reuse_raw_samples = [
        {
            "ttft_ms": round_result["reuse_ms"],
            "cached_tokens": round_result["reuse_cached_tokens"],
        }
        for round_result in rounds
    ]
    fresh_ms_samples = [sample["ttft_ms"] for sample in fresh_raw_samples]
    reuse_ms_samples = [sample["ttft_ms"] for sample in reuse_raw_samples]
    first_round = rounds[0]
    last_round = rounds[-1]
    return {
        "seed_head_ms": last_round["seed_head_ms"],
        "register_raw_ms": last_round["register_raw_ms"],
        "fresh_raw_samples": fresh_raw_samples,
        "reuse_raw_samples": reuse_raw_samples,
        "fresh_ms_samples": fresh_ms_samples,
        "reuse_ms_samples": reuse_ms_samples,
        "combined_ms_samples": [
            fresh_ms + reuse_ms
            for fresh_ms, reuse_ms in zip(fresh_ms_samples, reuse_ms_samples)
        ],
        "observed_cached_tokens_per_call": [
            sample["cached_tokens"] for sample in reuse_raw_samples
        ],
        "metrics_before": first_round["metrics_at_round_start"],
        "metrics_after": last_round["metrics_after_round"],
        "capacity_tokens": last_round["capacity_tokens"],
        "already_pinned_tokens": last_round["already_pinned_tokens"],
        "head_reseed_after_pressure": last_round["head_reseed_after_pressure"],
        "observed_rho_after_target": last_round["observed_rho_after_target"],
        "peak_rho_observed": max(
            round_result["peak_rho_observed"] for round_result in rounds
        ),
        "pressure_and_target_evicted_tokens_total_delta": metric_delta(
            first_round["metrics_at_round_start"],
            last_round["metrics_after_round"],
            "sglang:evicted_tokens_total",
        ),
        "pressure_phase": last_round["pressure_phase"],
        "rounds": rounds,
    }


def run_exact_context_control_point(
    *,
    base_url: str,
    tokenizer: Any,
    body_tokens: int,
    tail_tokens: int,
    salt: str,
    repeats: int,
) -> dict[str, Any]:
    """The header=0 shape-sweep control point.

    ``NonPrefixSegmentWorkload`` structurally requires
    ``source_head_ids != target_head_ids`` (see its own ``__post_init__``)
    precisely so the registered raw/fresh segments capture genuinely
    different preceding-context KV -- at header length 0 both heads
    would be empty and trivially equal, so there is no way to build a
    genuine non-prefix repair workload at all: the body would start at
    position 0 in both source and target, making them exact-content-
    identical by construction, never a lossy cross-context repair.
    Rather than silently forcing a zero header through
    ``NonPrefixSegmentWorkload`` (which would raise) or quietly coercing
    it to some nonzero value (which would misrepresent what was actually
    measured), this instead runs an honest, separately-labeled "exact-
    context control point": flush, one discarded dense warmup, then
    ``repeats`` formal flush + dense-request repeats of
    ``body_tokens + tail_tokens`` tokens -- the same per-repeat flush
    discipline the dense baseline uses (see module docstring), so every
    formal repeat is a genuine, uncached dense forward pass, never an
    accelerated repeat-of-a-repeat exact hit.

    This is the cheapest-possible ("zero restore length") reference
    point for the header sweep's low end -- never mislabeled as a
    genuine CacheTune repair measurement: ``is_exact_context_control`` is
    always ``True`` and ``body_source_context_differs_from_target`` is
    always ``False`` in the returned dict, both surfaced explicitly for
    the caller's output JSON.

    Returns a dict with ``body_tokens``, ``is_exact_context_control``
    (always ``True``), ``body_source_context_differs_from_target``
    (always ``False``), ``dense_raw_samples``, ``dense_ms_samples``, and
    ``dense_p50_ms``.
    """
    if body_tokens <= 0:
        raise ValueError(f"body_tokens must be positive, got {body_tokens}")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    prompt_ids = _deterministic_token_ids(
        tokenizer, f"{salt}-exact-control-body", body_tokens
    ) + _deterministic_token_ids(tokenizer, f"{salt}-exact-control-tail", tail_tokens)

    flush_exact_radix_cache(base_url)
    warmup_response, _ = timed_post(base_url, dense_generate_payload(prompt_ids))
    require_finished_by_length(
        warmup_response, "exact-context-control warmup (discarded)"
    )

    dense_raw_samples: list[dict[str, Any]] = []
    for _ in range(repeats):
        flush_exact_radix_cache(base_url)
        dense_response, dense_ttft_ms = timed_post(
            base_url, dense_generate_payload(prompt_ids)
        )
        require_finished_by_length(dense_response, "exact-context-control")
        dense_raw_samples.append(
            {
                "ttft_ms": dense_ttft_ms,
                "cached_tokens": int(dense_response["meta_info"]["cached_tokens"]),
            }
        )
        time.sleep(0.1)
    dense_ms_samples = [sample["ttft_ms"] for sample in dense_raw_samples]
    return {
        "body_tokens": body_tokens,
        "is_exact_context_control": True,
        "body_source_context_differs_from_target": False,
        "dense_raw_samples": dense_raw_samples,
        "dense_ms_samples": dense_ms_samples,
        "dense_p50_ms": statistics.median(dense_ms_samples),
    }


def build_sweep_point_result(
    *,
    workload: NonPrefixSegmentWorkload,
    quantized: Any,
    repeats: int,
    setting_result: dict[str, Any],
) -> dict[str, Any]:
    """Shared point-result assembly for every genuine (non-exact-control)
    sweep point: every shape-sweep point with header > 0, and every
    rho-sweep point, both built by ``run_non_prefix_setting``. Reused by
    both sweep loops in ``run_canary`` so the two stay identically
    structured and identically cross-validated against
    ``expected_repair_totals``. ``rounds`` is passed through verbatim
    from ``setting_result`` -- the complete list of every formal round's
    own raw telemetry, for full per-round pressure/capacity/eviction
    transparency now that every round independently re-sizes and
    re-sends its own pressure phase (see ``run_non_prefix_setting``'s own
    docstring).
    """
    observed_selected_tokens_total = metric_delta(
        setting_result["metrics_before"],
        setting_result["metrics_after"],
        "sglang:approx_kv_cachetune_selected_tokens_total",
    )
    observed_dense_fallback = metric_delta(
        setting_result["metrics_before"],
        setting_result["metrics_after"],
        "sglang:approx_kv_dense_fallback_total",
    )
    point_expected = expected_repair_totals(
        repair_tokens_per_call=quantized.repair_tokens,
        recomputed_layers_per_call=0,  # not tracked per sweep point
        repeats=repeats,
    )
    expected_selected_tokens_for_point = point_expected[
        "expected_selected_tokens_total"
    ]
    expected_cached_tokens_per_call = (
        workload.body_start_in_target + workload.body_tokens
    )
    # Full restored prefix (exact-match head + entire restored body),
    # never head-only: see require_cached_tokens's own docstring for why
    # a successful CacheTune reuse always extends prefix_indices by the
    # complete restore_length regardless of the controller's selected
    # ratio.
    cached_tokens_ok = all(
        observed == expected_cached_tokens_per_call
        for observed in setting_result["observed_cached_tokens_per_call"]
    )
    pressure_phase = setting_result["pressure_phase"]
    return {
        "header_tokens": len(workload.target_head_ids),
        "body_tokens": workload.body_tokens,
        "tail_tokens": len(workload.tail_ids),
        "is_exact_context_control": False,
        "body_source_context_differs_from_target": (
            workload.body_source_context_differs_from_target
        ),
        "repeats": repeats,
        "seed_head_ms": setting_result["seed_head_ms"],
        "register_raw_ms": setting_result["register_raw_ms"],
        "expected_selected_tokens_per_call": quantized.repair_tokens,
        "expected_selected_tokens_total": expected_selected_tokens_for_point,
        "expected_executable_ratio": quantized.executable_ratio,
        "observed_selected_tokens_total": observed_selected_tokens_total,
        "observed_dense_fallback": observed_dense_fallback,
        "observed_cached_tokens_per_call": (
            setting_result["observed_cached_tokens_per_call"]
        ),
        "expected_cached_tokens_per_call": expected_cached_tokens_per_call,
        "fresh_raw_samples": setting_result["fresh_raw_samples"],
        "reuse_raw_samples": setting_result["reuse_raw_samples"],
        "fresh_ms_samples": setting_result["fresh_ms_samples"],
        "reuse_ms_samples": setting_result["reuse_ms_samples"],
        "combined_ms_samples": setting_result["combined_ms_samples"],
        "fresh_p50_ms": statistics.median(setting_result["fresh_ms_samples"]),
        "reuse_p50_ms": statistics.median(setting_result["reuse_ms_samples"]),
        "combined_p50_ms": statistics.median(setting_result["combined_ms_samples"]),
        "capacity_tokens": setting_result["capacity_tokens"],
        "already_pinned_tokens": setting_result["already_pinned_tokens"],
        "head_reseed_after_pressure": setting_result["head_reseed_after_pressure"],
        "target_rho": pressure_phase["target_rho"] if pressure_phase else None,
        "observed_rho_after_target": setting_result["observed_rho_after_target"],
        "peak_rho_observed": setting_result["peak_rho_observed"],
        "pressure_and_target_evicted_tokens_total_delta": (
            setting_result["pressure_and_target_evicted_tokens_total_delta"]
        ),
        "pressure_phase": pressure_phase,
        "rounds": setting_result["rounds"],
        "passed": (
            observed_selected_tokens_total == expected_selected_tokens_for_point
            and observed_dense_fallback == 0
            and cached_tokens_ok
        ),
    }


def run_canary(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full canary against a live server and return the
    result payload (also written to ``args.output`` and printed)."""
    mode = CacheTuneMode(args.mode)
    bounds = RatioBounds.for_mode(mode)
    measurement = HardwareMeasurement(
        t_c_ms=args.t_c_ms,
        t_i_ms=args.t_i_ms,
        t_o_ms=args.t_o_ms,
    )
    r0 = roofline_ratio(measurement)

    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=True,
    )
    model_config = AutoConfig.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=True,
    )
    num_layers = int(model_config.num_hidden_layers)
    expected_recomputed_layers = num_layers - args.first_recompute_layer
    if expected_recomputed_layers <= 0:
        raise RuntimeError("first_recompute_layer leaves no layers to recompute")

    # ---- Main TTFT benchmark point: build the non-prefix workload up
    # front (see module docstring's "Why /generate and non-prefix
    # segments" for why source_head != target_head is mandatory). Real
    # GPU eviction pressure (see module docstring's "Eviction-pressure
    # phase" section) is reverse-computed and materialized fresh INSIDE
    # run_independent_round itself, per ROUND (the discarded warmup
    # round and every formal repeat alike, each fully independent -- see
    # that function's own docstring), from a real, live, idle /metrics
    # snapshot taken immediately after that round's own flush -- never
    # built once globally here, since a globally-built filler set would
    # not survive past even the very first round's own flush (see
    # flush_exact_radix_cache's docstring) and since the correct filler
    # count depends on each round's own freshly-measured capacity. ----
    main_workload = build_non_prefix_segment_workload(
        tokenizer,
        body_tokens=args.main_body_tokens,
        head_tokens=args.main_header_tokens,
        tail_tokens=NON_PREFIX_TAIL_TOKENS,
        salt=f"{CACHE_SALT}-main",
    )
    main_quantized = quantize_ratio(
        r0,
        context_length=main_workload.body_tokens,
        bounds=bounds,
    )
    main_predicted_ttft_ms = predict_ttft_ms(
        measurement,
        num_layers=num_layers,
        context_length=main_workload.body_tokens,
        ratio=main_quantized.executable_ratio,
    )
    expected_selected_tokens_per_call = main_quantized.repair_tokens

    # ---- Dense baseline: runs to completion BEFORE any raw/fresh
    # registration or head-seeding begins (see flush_exact_radix_cache's
    # docstring for why this ordering and the per-repeat flushing are
    # both mandatory).
    flush_exact_radix_cache(args.base_url)
    warmup_dense_response, _ = timed_post(
        args.base_url, dense_generate_payload(main_workload.target_prompt_ids)
    )
    require_finished_by_length(warmup_dense_response, "dense warmup (discarded)")

    dense_raw_samples: list[dict[str, Any]] = []
    for _ in range(args.repeats):
        flush_exact_radix_cache(args.base_url)
        dense_response, dense_ttft_ms = timed_post(
            args.base_url,
            dense_generate_payload(main_workload.target_prompt_ids),
        )
        require_finished_by_length(dense_response, "dense baseline")
        dense_raw_samples.append(
            {
                "ttft_ms": dense_ttft_ms,
                "cached_tokens": int(dense_response["meta_info"]["cached_tokens"]),
            }
        )
        time.sleep(0.1)
    dense_ms_samples = [sample["ttft_ms"] for sample in dense_raw_samples]
    # No explicit flush here: dense's last formal repeat left
    # main_workload's full target sequence in the exact radix tree, but
    # run_non_prefix_setting's first call to run_independent_round (for
    # its own discarded warmup round) flushes it away as that round's
    # own first action (see run_independent_round's docstring) before
    # seeding the head or registering anything -- so the CacheTune reuse
    # requests it issues cannot be silently served by that stale
    # exact-cache entry instead of the real approximate-repair path.

    raw_hash = "cachetune-raw:phase4-r5-main"
    fresh_hash = "cachetune-fresh:phase4-r5-main"
    main_result = run_non_prefix_setting(
        base_url=args.base_url,
        tokenizer=tokenizer,
        workload=main_workload,
        raw_hash=raw_hash,
        fresh_hash=fresh_hash,
        model_fingerprint=args.model_fingerprint,
        cache_dtype=args.cache_dtype,
        repeats=args.repeats,
        label="main",
        max_chunk_tokens=args.max_segment_chunk_tokens,
        target_rho=args.main_target_rho,
        pressure_filler_head_tokens=args.pressure_filler_head_tokens,
        pressure_filler_body_tokens=args.pressure_filler_body_tokens,
    )
    fresh_raw_samples = main_result["fresh_raw_samples"]
    cachetune_raw_samples = main_result["reuse_raw_samples"]
    fresh_ms_samples = main_result["fresh_ms_samples"]
    cachetune_ms_samples = main_result["reuse_ms_samples"]
    combined_ms_samples = main_result["combined_ms_samples"]
    observed_cached_tokens_per_call = main_result["observed_cached_tokens_per_call"]
    metrics_before_cachetune = main_result["metrics_before"]
    metrics_after_cachetune = main_result["metrics_after"]
    main_pressure_phase = main_result["pressure_phase"]

    cachetune_deltas = {
        name: metric_delta(metrics_before_cachetune, metrics_after_cachetune, name)
        for name in (
            "sglang:approx_kv_cachetune_selected_tokens_total",
            "sglang:approx_kv_cachetune_recomputed_layers_total",
            "sglang:approx_kv_cachetune_precomputed_total",
            "sglang:approx_kv_dense_fallback_total",
        )
    }
    main_expected = expected_repair_totals(
        repair_tokens_per_call=expected_selected_tokens_per_call,
        recomputed_layers_per_call=expected_recomputed_layers,
        repeats=args.repeats,
    )
    expect_precomputed_adapter = main_expected["expect_precomputed_adapter"]
    expected_selected_tokens_total = main_expected["expected_selected_tokens_total"]
    expected_recomputed_layers_total = main_expected["expected_recomputed_layers_total"]
    expected_precomputed_total = main_expected["expected_precomputed_total"]
    # Full restored prefix (exact-match head + entire restored body),
    # never head-only: a successful CacheTune reuse always extends
    # prefix_indices by the complete restore_length regardless of the
    # controller's selected ratio (see require_cached_tokens's own
    # docstring; confirmed against a real SM75 run).
    main_expected_cached_tokens_per_call = (
        main_workload.body_start_in_target + main_workload.body_tokens
    )
    telemetry_checks = {
        "selected_tokens_total_matches_controller_decision": (
            cachetune_deltas["sglang:approx_kv_cachetune_selected_tokens_total"]
            == expected_selected_tokens_total
        ),
        "recomputed_layers_total_matches_first_recompute_layer": (
            cachetune_deltas["sglang:approx_kv_cachetune_recomputed_layers_total"]
            == expected_recomputed_layers_total
        ),
        "precomputed_adapter_used_every_call": (
            cachetune_deltas["sglang:approx_kv_cachetune_precomputed_total"]
            == expected_precomputed_total
        ),
        "no_unexpected_dense_fallback": (
            cachetune_deltas["sglang:approx_kv_dense_fallback_total"] == 0
        ),
        "cached_tokens_matches_full_restored_prefix_every_call": all(
            observed == main_expected_cached_tokens_per_call
            for observed in observed_cached_tokens_per_call
        ),
    }
    if not all(telemetry_checks.values()):
        raise RuntimeError(f"telemetry cross-validation failed: {telemetry_checks}")

    dense_p50_ms = statistics.median(dense_ms_samples)
    cachetune_target_p50_ms = statistics.median(cachetune_ms_samples)
    fresh_preparation_p50_ms = statistics.median(fresh_ms_samples)
    combined_p50_ms = statistics.median(combined_ms_samples)

    # ---- Shape sweep: header x body cross product (from
    # --header-tokens-choices x --body-tokens-choices) at a FIXED rho
    # (--length-sweep-rho, defaults to --main-target-rho) -- the exact
    # (main_header_tokens, main_body_tokens) combo is skipped since the
    # main setting above already measured it and re-running it here would
    # be a redundant, wasteful re-measurement on real GPU time. header=0
    # cannot build a genuine NonPrefixSegmentWorkload (see
    # run_exact_context_control_point's own docstring) so it routes to
    # that dedicated exact-context control point instead of
    # run_non_prefix_setting.
    shape_sweep_points: list[dict[str, Any]] = []
    for header_tokens in args.header_tokens_choices:
        for body_tokens in args.body_tokens_choices:
            if (
                header_tokens == args.main_header_tokens
                and body_tokens == args.main_body_tokens
            ):
                continue
            artifact = f"phase4-r5-cachetune-shape-h{header_tokens}-b{body_tokens}"
            if header_tokens == 0:
                control_result = run_exact_context_control_point(
                    base_url=args.base_url,
                    tokenizer=tokenizer,
                    body_tokens=body_tokens,
                    tail_tokens=NON_PREFIX_TAIL_TOKENS,
                    salt=f"{CACHE_SALT}-{artifact}",
                    repeats=args.repeats,
                )
                shape_sweep_points.append(
                    {
                        "header_tokens": 0,
                        "tail_tokens": NON_PREFIX_TAIL_TOKENS,
                        "target_rho": args.length_sweep_rho,
                        **control_result,
                        "passed": True,
                    }
                )
                time.sleep(0.1)
                continue

            sweep_workload = build_non_prefix_segment_workload(
                tokenizer,
                body_tokens=body_tokens,
                head_tokens=header_tokens,
                tail_tokens=NON_PREFIX_TAIL_TOKENS,
                salt=f"{CACHE_SALT}-{artifact}",
            )
            quantized = quantize_ratio(
                r0, context_length=sweep_workload.body_tokens, bounds=bounds
            )
            sweep_result = run_non_prefix_setting(
                base_url=args.base_url,
                tokenizer=tokenizer,
                workload=sweep_workload,
                raw_hash=f"cachetune-raw:{artifact}",
                fresh_hash=f"cachetune-fresh:{artifact}",
                model_fingerprint=args.model_fingerprint,
                cache_dtype=args.cache_dtype,
                repeats=args.repeats,
                label=f"shape[header={header_tokens},body={body_tokens}]",
                max_chunk_tokens=args.max_segment_chunk_tokens,
                target_rho=args.length_sweep_rho,
                pressure_filler_head_tokens=args.pressure_filler_head_tokens,
                pressure_filler_body_tokens=args.pressure_filler_body_tokens,
            )
            shape_sweep_points.append(
                build_sweep_point_result(
                    workload=sweep_workload,
                    quantized=quantized,
                    repeats=args.repeats,
                    setting_result=sweep_result,
                )
            )
            time.sleep(0.1)

    if not all(point["passed"] for point in shape_sweep_points):
        raise RuntimeError(f"shape sweep validation failed: {shape_sweep_points}")

    # ---- Rho sweep: --target-rho-choices at the FIXED main
    # (--main-header-tokens, --main-body-tokens) shape -- the value equal
    # to --main-target-rho is skipped since the main setting above
    # already measured it.
    rho_sweep_points: list[dict[str, Any]] = []
    for target_rho in args.target_rho_choices:
        if target_rho == args.main_target_rho:
            continue
        artifact = f"phase4-r5-cachetune-rho-{target_rho}"
        rho_workload = build_non_prefix_segment_workload(
            tokenizer,
            body_tokens=args.main_body_tokens,
            head_tokens=args.main_header_tokens,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt=f"{CACHE_SALT}-{artifact}",
        )
        quantized = quantize_ratio(
            r0, context_length=rho_workload.body_tokens, bounds=bounds
        )
        rho_result = run_non_prefix_setting(
            base_url=args.base_url,
            tokenizer=tokenizer,
            workload=rho_workload,
            raw_hash=f"cachetune-raw:{artifact}",
            fresh_hash=f"cachetune-fresh:{artifact}",
            model_fingerprint=args.model_fingerprint,
            cache_dtype=args.cache_dtype,
            repeats=args.repeats,
            label=f"rho[{target_rho}]",
            max_chunk_tokens=args.max_segment_chunk_tokens,
            target_rho=target_rho,
            pressure_filler_head_tokens=args.pressure_filler_head_tokens,
            pressure_filler_body_tokens=args.pressure_filler_body_tokens,
        )
        rho_sweep_points.append(
            build_sweep_point_result(
                workload=rho_workload,
                quantized=quantized,
                repeats=args.repeats,
                setting_result=rho_result,
            )
        )
        time.sleep(0.1)

    if not all(point["passed"] for point in rho_sweep_points):
        raise RuntimeError(f"rho sweep validation failed: {rho_sweep_points}")

    # Every setting above intentionally left raw/fresh/pressure-filler
    # segments resident (see capture_final_pool_reset_and_invariant's own
    # docstring) -- flush them and force one real scheduler iteration
    # before trusting the pool's idle invariant, instead of gating
    # pass/fail on a pre-flush snapshot that would misreport that
    # by-design residency as a leak.
    final_pool_reset = capture_final_pool_reset_and_invariant(args.base_url, tokenizer)
    metrics_pre_reset = final_pool_reset["metrics_pre_reset"]
    metrics_post_reset = final_pool_reset["metrics_post_reset"]
    pool_invariant = final_pool_reset["pool_invariant"]
    health_status = fetch_text(f"{args.base_url}/health")

    known_limitations = [
        "Only one (roofline-derived) ratio configuration received a real "
        f"SM75 server canary in this result: r0={r0:.4f} under mode={mode.value}.",
        "Fresh target-context KV is generated by an explicit dense "
        "preparation request; its cost is included in combined_p50_ms.",
        "seed_head_ms and register_raw_ms are genuinely re-measured by "
        "EVERY independent round (the discarded warmup round and every "
        "formal repeat alike re-seed the exact-match head and "
        "re-capture the raw segment's source-context KV from scratch, "
        "via that round's own register_round_setup); the reported "
        "top-level value is the LAST formal round's own measurement "
        "(every other round's own value remains available via that "
        "same result's rounds list). Both are reported but excluded "
        "from combined_ms/combined_p50_ms -- analogous to why the raw "
        "registration step was already excluded in every prior version of "
        "this script: both represent context that would already exist "
        "before the measured request in a real deployment (a prior "
        "conversation turn's own exact-cache entry, and externally "
        "sourced/precomputed KV), not part of the request being measured.",
        "The generic online ModelRunner selected-token forward hook "
        "remains unavailable on this fork, so every successful repair in "
        "this canary used the precomputed fresh-KV adapter path, not a "
        "genuine inline per-layer recompute.",
        "Recompute and transfer critical paths are not executed with "
        "genuine wall-clock overlap in this backend; the roofline model "
        "chooses the ratio faithfully, but execution uses this project's "
        "available-hardware adaptation (see cachetune/runtime.py).",
        "This is a CacheTune hardware-controller inspired subset: "
        "frequency-domain token selection, sparse transfer, "
        "multi-stream overlap, and deferred RoPE from the full paper are "
        "out of scope for this branch.",
        "No accuracy/quality benchmark was run; success criteria are "
        "TTFT, real repair-token/telemetry accounting, and absence of "
        "crash/OOM/allocator corruption only.",
    ]
    if mode is CacheTuneMode.SPEED_ONLY:
        known_limitations.append(
            "mode=speed_only allows a 0% repair-token floor; this is this "
            "project's own non-paper setting, not the paper's r_min=15% "
            "quality floor (paper_mechanism)."
        )
    else:
        known_limitations.append(
            "mode=paper_mechanism reproduces the paper's r_min=15% quality "
            "floor; this project does not evaluate output quality, so the "
            "floor is exercised here purely as a ratio-selection behavior."
        )
    known_limitations.append(
        "Every setting that runs a genuine repair (main, every "
        "shape-sweep point with header > 0, and every rho-sweep point) "
        "runs a freshly reverse-computed eviction-pressure phase inside "
        "EVERY ROUND (the discarded warmup round and every formal "
        "repeat alike, each a fully independent run_independent_round "
        "call -- never a single setting-wide pressure phase shared "
        "across rounds), sent immediately AFTER (never before) that "
        "SAME round's own setup (head-seed + raw-register + "
        "fresh-register) completes, sized against that round's own "
        "real, live, idle usable_kv_capacity_tokens snapshot NET OF that "
        "round's own already_pinned_tokens (that round's setup's own "
        "real, measured -- not estimated -- contribution to "
        "sglang:kv_used_tokens, see eviction_pressure_filler_count_for_rho "
        "/ register_eviction_pressure_objects) -- never a single "
        "globally- or setting-wide-shared filler set and never "
        "CLI-disableable to a single-object microbenchmark. This "
        "source-setup-before-pressure ordering, and the requirement "
        "that every round rebuild its OWN pressure phase from scratch "
        "rather than reuse another round's, are both fixes for two "
        "distinct real SM75 target_rho=2 bugs: (1) register's own "
        "segment materialization is not wired to evict exact-radix "
        "victims to make room for itself (unlike the reuse/repair "
        "path's own recovery-slot allocation, which explicitly does), "
        "so sending pressure before that round's own source setup would "
        "starve setup of device headroom under high pressure; and (2) "
        "an earlier design shared ONE raw registration across the "
        "discarded warmup and every formal repeat while re-registering "
        "only fresh per repeat, which under high pressure required each "
        "repeat's fresh registration to transiently coexist with the "
        "shared setup's still-resident raw segment plus surviving "
        "pressure fillers -- producing two consecutive MemoryErrors "
        "then target reuse OOM. Making every round fully independent "
        "(its own flush, its own raw+fresh registration, its own "
        "pressure phase sized from its own post-setup footprint, its "
        "own reuse) removes both failure modes. A guard re-seed "
        "(ensure_target_head_resident) runs once per round, immediately "
        "after that round's own pressure phase (see "
        "run_independent_round's own docstring), since that round's own "
        "head, seeded before any filler in that round, is the oldest "
        "exact-radix entry once that round's own pressure begins and a "
        "plausible LRU-eviction candidate itself. Every filler object is "
        "sent as a single plain dense /generate request with no "
        "approx_kv metadata (an ordinary, LRU-evictable exact-radix-tree "
        "entry), never through CacheTune's own register/reuse repair "
        "path, so real device-pool eviction can actually reclaim filler "
        "objects the way normal concurrent traffic would in a real "
        "deployment; whether real device-pool eviction actually "
        "occurred for a given round must be read from that round's own "
        "entry in the setting's own rounds list "
        "(pressure_phase.evicted_tokens_total_delta / "
        "peak_rho_observed / observed_rho_after_target -- may "
        "legitimately differ from the nominal target_rho -- reported "
        "honestly, never hidden); the setting-level pressure_phase / "
        "observed_rho_after_target keys report only the LAST formal "
        "round's own values, while peak_rho_observed is the maximum "
        "across every formal round. The shape sweep's header=0 "
        "exact-context control points are the one deliberate exception: "
        "they run no pressure phase at all and carry no "
        "pressure_phase/capacity_tokens/peak_rho_observed/rounds keys "
        "(see the next limitation)."
    )
    known_limitations.append(
        "observed_rho_after_pressure / observed_rho_after_target / "
        "peak_rho_observed report genuine RESIDENT pool occupancy -- "
        "sglang:kv_used_tokens PLUS sglang:kv_evictable_tokens against a "
        "fixed capacity_tokens reference (see observed_rho) -- not "
        "kv_used_tokens alone. An earlier version of observed_rho used "
        "kv_used_tokens alone (the pool's currently pinned/in-use tokens "
        "only, the same quantity the server's own sglang:full_token_usage "
        "gauge reports), which undercounts genuine device pressure "
        "whenever a large population of dense eviction-pressure fillers "
        "remains resident as LRU-evictable exact-radix entries without "
        "having actually been reclaimed yet: a real SM75 target_rho=2 "
        "canary reported peak_rho_observed=0.156 under that formula "
        "(kv_used_tokens alone, 2048 / 13130) even though the pool was in "
        "fact ~99% resident once every surviving filler was counted too "
        "((2048 used + 10960 evictable) / 13130 ~= 0.991). observed_rho "
        "now raises immediately if either metric is unavailable, rather "
        "than silently falling back to a partial (and misleadingly low) "
        "reading."
    )
    known_limitations.append(
        "The shape sweep's header=0 points are an exact-context control "
        "point (dense-only, is_exact_context_control=True), never a "
        "genuine CacheTune repair measurement -- see "
        "run_exact_context_control_point's own docstring for why header=0 "
        "cannot build a NonPrefixSegmentWorkload at all."
    )
    known_limitations.append(
        "Every target-head seed/re-seed dense request (register_round_"
        "setup's initial seed and ensure_target_head_resident's "
        "post-pressure re-seed) posts target_head_ids PLUS an explicit, "
        "per-workload seed_sentinel_ids token -- never target_head_ids "
        "alone -- and expects cached_tokens in {0, len(target_head_ids), "
        "len(target_head_ids) + len(seed_sentinel_ids)}. This is a fix "
        "for a real SM75 header-sweep bug: a bare target_head_ids-only "
        "seed request (max_new_tokens=1, temperature=0, fully "
        "deterministic) inserts its own single generated token into the "
        "exact radix tree (plain dense requests never set "
        "skip_radix_cache_insert); when that generated token happened to "
        "equal shared_body_ids[0], the tree's exact-match boundary for "
        "that head silently extended by one token, so the very next "
        "fresh-register request (target_head_ids + shared_body_ids + "
        "tail_ids) reported cached_tokens=33 on a header=32 setting "
        "instead of the true 32. seed_sentinel_ids is validated, per "
        "workload, to differ from that SAME workload's own "
        "shared_body_ids[0] (see NonPrefixSegmentWorkload.seed_prompt_ids "
        "/ _build_seed_sentinel_ids_avoiding_body_first_token_collision), "
        "anchoring the tree's exact-match boundary at a fixed, known, "
        "non-body-colliding token regardless of what the seed request's "
        "own generated token turns out to be."
    )
    known_limitations.append(
        "Every run_independent_round call starts with "
        "flush_and_force_gauge_refresh (flush, then one small fixed "
        "dense sentinel request, then a /metrics snapshot), never a "
        "bare flush-then-snapshot -- and a second, bare flush "
        "immediately follows before register_round_setup's own seed "
        "call. This is a fix for a real SM75 body-length-sweep bug: "
        "/flush_cache clears the actual pool/tree state synchronously, "
        "but gauges such as sglang:kv_used_tokens are only recomputed "
        "by the scheduler's own next iteration -- so a bare "
        "flush-then-snapshot with no intervening real request could "
        "read a value carried over from a PREVIOUS round or even a "
        "previous setting. Observed on a real run: switching from a "
        "body=1024 setting (ending with kv_used_tokens=2048 resident) "
        "to the next body=512 setting produced "
        "already_pinned_tokens=1024-2048=-1024 -- a structurally "
        "negative value this script never clamps away (see "
        "eviction_pressure_filler_count_for_rho's own "
        "already_pinned_tokens < 0 check, which correctly raises "
        "ValueError instead). The sentinel forces a real scheduler "
        "iteration so the snapshot genuinely reflects the just-flushed "
        "idle pool; the second, sentinel-less flush then clears away "
        "the sentinel's own tiny resident footprint so it can never "
        "collide with this round's own head-seed request (see "
        "flush_and_force_gauge_refresh's own docstring for the full "
        "mechanism)."
    )

    payload = {
        "schema_version": 4,
        "runner_git_sha": args.runner_git_sha,
        "image_digest": args.image_digest,
        "model": args.model,
        "model_revision": args.model_revision,
        "scope": {
            "recovery": (
                "CacheTune hardware-aware roofline repair-ratio controller "
                "plus ported CacheBlend-style selected-token repair "
                "(precomputed fresh-KV adapter)"
            ),
            "mode": mode.value,
            "scheduler": "S0 LRU",
            "tier": "GPU-only",
            "prefetch": False,
            "accuracy_metric": False,
        },
        "measurement_protocol": {
            "warmup_passes_per_setting": WARMUP_PASSES_PER_SETTING,
            "warmup_passes_discarded": True,
            "formal_repeats": args.repeats,
            "dense_flush_before_warmup": True,
            "dense_flush_before_each_formal_repeat": True,
            "dense_flush_after_formal_repeats_before_registration": True,
            "exact_radix_flush_at_start_of_every_round": True,
            "exact_radix_flush_at_start_of_every_round_rationale": (
                "run_independent_round flushes the exact-match radix "
                "cache (and resets the approx_kv segment store) as its "
                "own first action, EVERY round -- the discarded warmup "
                "round and independently again for every formal repeat, "
                "never just once per setting. Otherwise a previous "
                "round's own already-seeded target_head_ids, "
                "already-registered raw/fresh segments, and "
                "already-sent pressure fillers would still be resident, "
                "either silently producing a nonzero cached_tokens for "
                "this round's own head-seed/register calls or forcing "
                "this round's own raw+fresh registration to transiently "
                "coexist with a previous round's still-resident "
                "footprint -- the real SM75 target_rho=2 MemoryError "
                "this per-round flush fixes."
            ),
            "cachetune_reuse_flush_between_repeats": True,
            "cachetune_reuse_flush_rationale": (
                "every formal repeat -- and the discarded warmup round -- "
                "is now a fully independent round (run_independent_round): "
                "each one flushes first, then registers its OWN raw and "
                "fresh segments from scratch, never reusing another "
                "round's registration. This reverses an earlier design's "
                "own rule, which forbade flushing between formal "
                "register+reuse repeats specifically because that "
                "earlier design shared ONE raw registration across every "
                "repeat -- flushing would have wiped the very segment "
                "those repeats depended on. That earlier sharing was "
                "itself the root cause of a real SM75 target_rho=2 "
                "MemoryError: each repeat's fresh registration had to "
                "transiently coexist with the shared setup's "
                "still-resident raw segment plus surviving pressure "
                "fillers, and register-side segment materialization is "
                "not wired to evict exact-radix victims to make room for "
                "itself. Flushing before every round removes that "
                "transient double-footprint entirely, and is safe for "
                "Prometheus telemetry: sglang:kv_used_tokens is a Gauge "
                "that resets on flush (exactly what lets every round "
                "measure its own idle capacity/pinned footprint), while "
                "every Counter this script reads (evicted_tokens_total, "
                "approx_kv_dense_fallback_total, "
                "approx_kv_cachetune_selected_tokens_total, etc.) is "
                "monotonic and unaffected by flush, so cross-round "
                "deltas remain mathematically sound."
            ),
            "ttft_measurement_method": (
                "every request sets stream: true; ttft_ms is the "
                "client-side wall-clock elapsed from just before the "
                "request is sent to the first non-'[DONE]' SSE 'data:' "
                "frame received (i.e. first token), timestamped before "
                "that frame's JSON body is parsed. This is never the "
                "prior blocking (non-streamed) whole-request elapsed "
                "time, which also bundles in full-response "
                "detokenization/serialization and body transfer after "
                "the first token was already produced."
            ),
            "ttft_stream_read_to_done_required": True,
            "ttft_stream_read_to_done_rationale": (
                "every stream is read in full through the terminal "
                "'data: [DONE]' frame -- never abandoned right after the "
                "first chunk -- as the success check for that request; a "
                "dropped connection, a stream that never reaches "
                "'[DONE]', or a mid-stream error frame all raise instead "
                "of being silently treated as a completed request."
            ),
            "eviction_pressure_phase_runs_once_per_round": True,
            "eviction_pressure_phase_rationale": (
                "every round's own flush (see "
                "exact_radix_flush_at_start_of_every_round) clears the "
                "entire exact radix tree and resets the approx_kv store, "
                "so eviction-pressure filler objects are freshly "
                "reverse-computed (from that setting's own target_rho "
                "against a real, live, idle /metrics snapshot taken after "
                "THIS round's own setup, net of that round's own "
                "already_pinned_tokens) and re-sent inside every "
                "run_independent_round call (the discarded warmup round "
                "and every formal repeat, for the main setting, every "
                "shape-sweep point, and every rho-sweep point alike), "
                "never built/sent once per setting and never once "
                "globally -- see eviction_pressure_filler_count_for_rho / "
                "register_eviction_pressure_objects."
            ),
            "eviction_pressure_phase_sent_after_source_setup": True,
            "eviction_pressure_phase_sent_after_source_setup_rationale": (
                "each round's own pressure phase is sent AFTER (never "
                "before) that SAME round's own setup (head-seed + "
                "raw-register + fresh-register, see register_round_setup) "
                "completes, and its filler count is reverse-computed net "
                "of that round's own already_pinned_tokens (that round's "
                "setup's own real, measured -- never estimated -- "
                "contribution to sglang:kv_used_tokens, sampled right "
                "after that round's own setup finishes). This is a fix "
                "for a real SM75 target_rho=2 bug where sending pressure "
                "BEFORE source setup starved setup of device headroom: "
                "register's own segment materialization, unlike the "
                "reuse/repair path's own recovery-slot allocation, is not "
                "wired to evict exact-radix victims to make room for "
                "itself. A guard re-seed (ensure_target_head_resident) "
                "runs once per round, immediately after that round's own "
                "pressure phase, to protect that round's own target head "
                "-- seeded before any filler, and therefore the oldest "
                "exact-radix entry in that round -- from that same "
                "round's own pressure; see run_independent_round's own "
                "docstring."
            ),
            "eviction_pressure_fillers_are_plain_dense_requests": True,
            "eviction_pressure_fillers_are_plain_dense_requests_rationale": (
                "every filler object is sent as a single plain dense "
                "/generate request with no approx_kv custom_params "
                "metadata at all, landing in the server's ordinary, "
                "LRU-evictable exact radix tree -- never through "
                "CacheTune's own register/reuse repair path (whose raw/"
                "fresh segments live in ApproxKVManager's own segment "
                "store, a structure Radix LRU eviction cannot reclaim at "
                "all). This is a deliberate fix for a real, previously-"
                "observed SM75 bug at target_rho=2 where CacheTune-path "
                "fillers accumulated as permanently un-evictable "
                "residency and starved the setting's own target "
                "recovery-slot allocation -- see "
                "register_eviction_pressure_objects's own docstring."
            ),
        },
        "workload": {
            "kind": "non_prefix_segment",
            "endpoint": "/generate",
            "main_header_tokens": args.main_header_tokens,
            "main_body_tokens": args.main_body_tokens,
            "main_target_rho": args.main_target_rho,
            "tail_tokens": NON_PREFIX_TAIL_TOKENS,
            "header_tokens_choices": list(args.header_tokens_choices),
            "body_tokens_choices": list(args.body_tokens_choices),
            "target_rho_choices": list(args.target_rho_choices),
            "length_sweep_rho": args.length_sweep_rho,
            "max_segment_chunk_tokens": args.max_segment_chunk_tokens,
            "pressure_filler_head_tokens": args.pressure_filler_head_tokens,
            "pressure_filler_body_tokens": args.pressure_filler_body_tokens,
            "description": (
                "source_prompt = source_head_ids + shared_body_ids + "
                "tail_ids; target_prompt = target_head_ids + "
                "shared_body_ids + tail_ids; fresh_prompt is token-"
                "identical to target_prompt; source_head_ids != "
                "target_head_ids by construction (see "
                "NonPrefixSegmentWorkload), so the registered raw "
                "(source-context) and fresh (target-context) segments "
                "capture genuinely different preceding-context KV for "
                "the byte-identical shared body. Every prompt's "
                "trailing tail_ids leaves at least its own final token "
                "outside every registered segment for a real forward "
                "pass (ApproxKVRequestMetadata.validate_prompt_length)."
            ),
        },
        "hardware_measurement": {
            "t_c_ms": args.t_c_ms,
            "t_i_ms": args.t_i_ms,
            "t_o_ms": args.t_o_ms,
            "roofline_ratio_r0": r0,
        },
        "server_validation": {
            "header_tokens": args.main_header_tokens,
            "body_tokens": main_workload.body_tokens,
            "num_layers": num_layers,
            "first_recompute_layer": args.first_recompute_layer,
            "controller_executable_ratio": main_quantized.executable_ratio,
            "controller_predicted_ttft_ms": main_predicted_ttft_ms,
            "selected_tokens_per_call": expected_selected_tokens_per_call,
            "recomputed_layers_per_call": (
                expected_recomputed_layers if expect_precomputed_adapter else 0
            ),
            "cachetune_deltas": cachetune_deltas,
            "expected_selected_tokens_total": expected_selected_tokens_total,
            "expected_recomputed_layers_total": expected_recomputed_layers_total,
            "expected_precomputed_total": expected_precomputed_total,
            "telemetry_checks": telemetry_checks,
            "fresh_target_kv_from_dense_preparation": True,
            "body_source_context_differs_from_target": (
                main_workload.body_source_context_differs_from_target
            ),
            "observed_cached_tokens_per_call": observed_cached_tokens_per_call,
            "expected_cached_tokens_per_call": main_expected_cached_tokens_per_call,
            "seed_head_ms": main_result["seed_head_ms"],
            "register_raw_ms": main_result["register_raw_ms"],
            "last_prompt_token_real_forward": True,
            "capacity_tokens": main_result["capacity_tokens"],
            "already_pinned_tokens": main_result["already_pinned_tokens"],
            "head_reseed_after_pressure": main_result["head_reseed_after_pressure"],
            "target_rho": args.main_target_rho,
            "observed_rho_after_target": main_result["observed_rho_after_target"],
            "peak_rho_observed": main_result["peak_rho_observed"],
            "pressure_and_target_evicted_tokens_total_delta": (
                main_result["pressure_and_target_evicted_tokens_total_delta"]
            ),
            "pressure_phase": main_pressure_phase,
            "rounds": main_result["rounds"],
            "passed": all(telemetry_checks.values()),
        },
        "ttft": {
            "repeats_per_mode": args.repeats,
            "measurement_method": (
                "client wall-clock elapsed to the first non-'[DONE]' SSE "
                "'data:' frame of a stream: true /generate request (see "
                "measurement_protocol.ttft_measurement_method); never "
                "blocking whole-request elapsed time."
            ),
            "dense_raw_samples": dense_raw_samples,
            "fresh_raw_samples": fresh_raw_samples,
            "cachetune_raw_samples": cachetune_raw_samples,
            "dense_ms_samples": dense_ms_samples,
            "fresh_ms_samples": fresh_ms_samples,
            "cachetune_ms_samples": cachetune_ms_samples,
            "combined_ms_samples": combined_ms_samples,
            "dense_p50_ms": dense_p50_ms,
            "cachetune_target_p50_ms": cachetune_target_p50_ms,
            "fresh_preparation_p50_ms": fresh_preparation_p50_ms,
            "combined_p50_ms": combined_p50_ms,
            "target_only_speedup": dense_p50_ms / cachetune_target_p50_ms,
            "combined_speedup": dense_p50_ms / combined_p50_ms,
        },
        "shape_sweep_points": shape_sweep_points,
        "rho_sweep_points": rho_sweep_points,
        "pool_invariant": pool_invariant,
        "pool_invariant_metrics_pre_reset": metrics_pre_reset,
        "pool_invariant_metrics_post_reset": metrics_post_reset,
        "pool_invariant_reset_note": (
            "pool_invariant (and its gating 'passed' bit) is computed "
            "ONLY from pool_invariant_metrics_post_reset, i.e. after "
            "flush_exact_radix_cache and one sentinel /generate request "
            "(see capture_final_pool_reset_and_invariant). "
            "pool_invariant_metrics_pre_reset is the raw /metrics "
            "snapshot from immediately before that reset -- its "
            "kv_used_tokens is expected to be nonzero (every setting's "
            "own raw/fresh CacheTune segments, plus every eviction-"
            "pressure filler's plain dense KV cache entry, are still "
            "resident by design at that point) and must never be read "
            "as a pool leak."
        ),
        "health_response": health_status,
        "known_limitations": known_limitations,
        "passed": all(telemetry_checks.values())
        and all(point["passed"] for point in shape_sweep_points)
        and all(point["passed"] for point in rho_sweep_points)
        and bool(pool_invariant.get("passed")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> int:
    args = parse_args()
    settings = build_settings(args)
    run_id = (
        f"phase4-r5-cachetune-{args.mode}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_path_str = str(args.output.resolve())
    append_run_log(
        args.central_log,
        {
            "run_id": run_id,
            "status": "running",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "settings": settings,
            "output": output_path_str,
        },
    )
    try:
        if args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        payload = run_canary(args)
    except Exception as exc:
        append_run_log(
            args.central_log,
            {
                "run_id": run_id,
                "status": "failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "settings": settings,
                "output": output_path_str,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise

    append_run_log(
        args.central_log,
        {
            "run_id": run_id,
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "settings": settings,
            "output": output_path_str,
            "result_summary": {
                "passed": payload["passed"],
                "mode": args.mode,
                "roofline_ratio_r0": payload["hardware_measurement"][
                    "roofline_ratio_r0"
                ],
                "dense_p50_ms": payload["ttft"]["dense_p50_ms"],
                "cachetune_target_p50_ms": payload["ttft"]["cachetune_target_p50_ms"],
                "fresh_preparation_p50_ms": payload["ttft"]["fresh_preparation_p50_ms"],
                "combined_p50_ms": payload["ttft"]["combined_p50_ms"],
                "target_only_speedup": payload["ttft"]["target_only_speedup"],
                "combined_speedup": payload["ttft"]["combined_speedup"],
                "shape_sweep_points": len(payload["shape_sweep_points"]),
                "rho_sweep_points": len(payload["rho_sweep_points"]),
                "main_target_rho": payload["server_validation"]["target_rho"],
                "peak_rho_observed": (
                    payload["server_validation"]["peak_rho_observed"]
                ),
            },
        },
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
