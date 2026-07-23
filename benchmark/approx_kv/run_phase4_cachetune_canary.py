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
*entire* prompt is exactly ``target_head_ids`` -- populating the exact
radix tree for that head -- since dense requests are the only request
type that ever write into the exact tree (register/reuse always set
``skip_radix_cache_insert=True``). This is why every setting's
measurement pass is, in order: seed the target head (one dense
``/generate`` call over ``target_head_ids`` alone) -> register the raw
segment once (one ``/generate`` register call over
``source_head_ids + shared_body_ids + tail_ids``) -> a discarded
register-fresh + reuse warmup -> ``--repeats`` formal register-fresh +
reuse repeats. The mandatory ``tail_ids`` (fixed at
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

Measurement protocol (mandatory, applies to every setting: the main
setting, every shape-sweep point, and every rho-sweep point)
--------------------------------------------------------------------
1. Dense baseline (no ``approx_kv`` metadata) runs *entirely* to
   completion before anything is registered: flush the exact-match radix
   cache before the discarded warmup, before every formal repeat, and
   once more right before any raw/fresh registration or head-seeding
   begins. This must happen first -- a real dense forward's exact-cache
   entry over the *same* tokens a later request targets would let the
   scheduler's own prefix match resolve the whole prompt before
   CacheTune's plugin dispatch ever runs.
2. ``run_non_prefix_setting`` flushes the exact-match radix cache once
   more of its own accord, as its very first action, before doing
   anything else. This is what actually makes step 3 below safe: this
   function runs once per *setting* (the main setting, every shape-sweep
   point with header > 0, and every rho-sweep point), settings are
   otherwise never isolated from each other, and a *previous* setting's
   own already-seeded ``target_head_ids`` would otherwise still be
   sitting in the tree, ready to silently produce a nonzero
   ``cached_tokens`` for an unrelated later setting's head or
   raw-segment register request.
2a. If eviction pressure is enabled (the default -- see "Eviction-
   pressure phase" below), every filler object is registered and
   materialized here, immediately after the flush and before step 3.
3. Seed the target head once (one dense ``/generate`` call, expected
   ``cached_tokens=0``).
4. Register the "raw" (source-context) body segment once (also expected
   ``cached_tokens=0`` -- see the ``source_head_ids``/``target_head_ids``
   zero-common-prefix discussion above).
5. One *discarded* register-fresh + reuse warmup pass.
6. ``--repeats`` (``>= 2``) formal register-fresh + reuse repeats,
   recording every repeat's raw wall-clock sample -- never just a derived
   median -- and cross-checking Prometheus telemetry deltas using only
   the formal repeat count (the warmup's own telemetry effect is already
   baked into the "before" snapshot, always taken *after* warmup
   completes). Every formal fresh-register response's
   ``meta_info.cached_tokens`` is checked against ``body_start_in_target``
   (the REGISTER operation never restores anything -- see
   ``approx_kv/runtime.py``'s ``_register_request_segments`` -- so its
   only contribution to ``prefix_indices`` is the exact-match radix hit
   on the already-seeded target head). Every formal reuse response's
   ``meta_info.cached_tokens`` is checked against
   ``body_start_in_target + body_tokens`` -- a real GPU run confirmed
   that SGLang's ``cached_tokens`` accounting (``pre_len -
   already_computed`` in ``schedule_batch.py``) counts the *entire*
   prefix already resolved without a fresh forward pass, and a
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

Register/reuse requests never need inter-repeat flushing:
``schedule_batch.Req.skip_radix_cache_insert`` is forced True whenever
``approx_kv_metadata`` is present, so they can never populate the exact
radix tree themselves; flushing between formal repeats would also invoke
``ApproxKVManager.reset()`` (see ``mem_cache/approx_kv/manager.py``),
deleting the very "raw"/"fresh" segments those repeats depend on.

Eviction-pressure phase (real GPU contention, not a single-object
microbenchmark, and never CLI-disableable)
--------------------------------------------------------------------
Every setting -- the main setting, every shape-sweep point with
header > 0, and every rho-sweep point alike -- always registers and
materializes a freshly reverse-computed set of distinct filler
``NonPrefixSegmentWorkload`` objects (see
``build_eviction_pressure_workloads``) immediately after that setting's
own flush, before that setting's own head-seed/raw-register begins (see
``register_eviction_pressure_objects``). The filler object COUNT is
reverse-computed (see ``eviction_pressure_filler_count_for_rho``) from
that setting's own ``target_rho`` (``--main-target-rho`` for the main
setting and every shape-sweep point, or the specific
``--target-rho-choices`` value under test for a rho-sweep point) against
a real, live, idle ``usable_kv_capacity_tokens`` snapshot (see
``benchmark.approx_kv.metrics``) taken immediately after that setting's
own flush -- never a fixed object count. Every filler object's own SHAPE
(``--pressure-filler-head-tokens``, default ``NON_PREFIX_HEAD_TOKENS``,
x ``--pressure-filler-body-tokens``, default 2048) is fixed across every
setting so only the reverse-computed COUNT varies with ``target_rho``,
keeping peak-rho/eviction numbers comparable across the whole matrix.
Each filler goes through the exact same register-raw + register-fresh +
one reuse cycle (``materialize_workload_via_reuse``) as the setting's own
mandatory setup, forcing its raw segment to become genuinely
device-resident via ``ensure_device`` -- a real occupant of the finite
GPU KV pool, not an artificial placeholder. Because every setting's own
flush wipes the *entire* ``approx_kv`` store (``ApproxKVManager.reset()``,
wired through ``RadixCache``/``UnifiedRadixCache``'s own ``reset()``),
filler objects cannot be built once globally and expected to persist
across settings: they are rebuilt fresh, from the same fixed shape but a
per-setting-appropriate count, inside every ``run_non_prefix_setting``
call.

Every filler's own target head is dense-seeded exactly like the
setting's own head, and all of them (N fillers plus the setting's own
head) coexist in the exact radix tree within the same flush epoch --
unlike the main-vs-sweep case, the per-setting flush does *not* isolate
them from each other. ``build_eviction_pressure_workloads`` gives each
filler a mutually distinct target-head literal-prefix marker (a
different leading Unicode code point drawn from a real-tokenizer-
validated pool, never a decimal index nor a bare fixed-width letter code
-- see ``_pressure_filler_head_literal_prefix``) to keep them pairwise
zero-common-prefix, and ``validate_pairwise_head_isolation`` is a
runtime safety net that checks the actual resulting token-id sequences
(never a textual heuristic alone) and raises immediately if any two
still collide.

There is no up-front floor check against a nominal fraction (the earlier
``--eviction-pressure-min-fraction``/``validate_eviction_pressure_fraction``
design): a ``target_rho`` value ``> 1`` (the entire ``--target-rho-choices``
default set except ``0.9``) already means the fillers alone nominally
request MORE tokens than the whole pool's measured capacity, guaranteeing
genuine eviction pressure by construction rather than by a separate
threshold check. The honest evidence that real device-pool eviction
actually occurred is each setting's own
``pressure_phase.evicted_tokens_total_delta`` /
``pressure_and_target_evicted_tokens_total_delta`` /
``peak_rho_observed`` in the output JSON (the genuine
``sglang:evicted_tokens_total`` Prometheus counter delta and the genuine
sampled ``sglang:kv_used_tokens`` gauge ratio -- incremented/updated by
real LRU eviction and real device-pool occupancy, GPU-only tier included,
not merely GPU-to-CPU host-backup moves) -- reported exactly as observed,
never inferred or assumed from the nominal ``target_rho`` alone. A
nonzero ``sglang:approx_kv_dense_fallback_total`` delta during the
pressure phase itself raises immediately (see
``register_eviction_pressure_objects``): a filler object silently
falling back to dense would mean it was never actually a genuine
CacheTune-repaired device-resident occupant at all.

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
# object costs several real, blocking HTTP round trips (register-raw,
# then a register-fresh + reuse cycle in materialize_workload_via_reuse)
# plus an explicit 0.1s sleep in register_eviction_pressure_objects, so an
# unbounded count from a pathological (--main-target-rho/--target-rho-
# choices, --pressure-filler-body-tokens, live measured capacity)
# combination could otherwise silently spend hours issuing real requests
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
        "--max-segment-chunk-tokens are registered as multiple "
        "<= --max-segment-chunk-tokens segments within the same register/"
        "reuse call (see body_segments_for_hash).",
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
        "canary registers. Bodies longer than this are split into "
        "multiple <= this length segments (distinct content_hash per "
        "chunk) within the same register/reuse call -- see "
        "body_segments_for_hash/chunk_offsets; "
        "ApproxKVRequestMetadata/register_request_segments/"
        "restore_request_prefix_cachetune already natively support an "
        "arbitrary number of segments per call.",
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
    a real, cache-writing forward pass) and once more before any
    raw/fresh registration or head-seeding begins. Do **not** call this
    between formal register+reuse repeats: doing so would also invoke
    ``ApproxKVManager.reset()`` (see
    ``python/sglang/srt/mem_cache/approx_kv/manager.py``), which wipes
    the very "raw"/"fresh" segments those repeats depend on.
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
    """

    source_head_ids: tuple[int, ...]
    target_head_ids: tuple[int, ...]
    shared_body_ids: tuple[int, ...]
    tail_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.source_head_ids:
            raise ValueError("source_head_ids must not be empty")
        if not self.target_head_ids:
            raise ValueError("target_head_ids must not be empty")
        if not self.shared_body_ids:
            raise ValueError("shared_body_ids must not be empty")
        if not self.tail_ids:
            raise ValueError("tail_ids must not be empty")
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
    def body_source_context_differs_from_target(self) -> bool:
        """Always ``True`` once constructed (``__post_init__`` already
        enforces ``source_head_ids != target_head_ids``); exposed as an
        explicit, self-documenting fact for the output JSON rather than
        re-deriving the comparison at every call site."""
        return self.source_head_ids != self.target_head_ids


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
    return NonPrefixSegmentWorkload(
        source_head_ids=_deterministic_token_ids(
            tokenizer,
            f"{salt}-source-head",
            head_tokens,
            literal_prefix=source_head_literal_prefix,
        ),
        target_head_ids=_deterministic_token_ids(
            tokenizer,
            f"{salt}-target-head",
            head_tokens,
            literal_prefix=target_head_literal_prefix,
        ),
        shared_body_ids=_deterministic_token_ids(
            tokenizer, f"{salt}-shared-body", body_tokens
        ),
        tail_ids=_deterministic_token_ids(tokenizer, f"{salt}-tail", tail_tokens),
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
    """Lower-bound estimate, in tokens, of the device-resident KV
    footprint that materializing every pressure workload's *raw* segment
    contributes (see ``register_eviction_pressure_objects`` /
    ``materialize_workload_via_reuse``): each filler object's mandatory
    register+reuse cycle forces at least this many tokens onto the
    device residency tier via ``ensure_device``. The "fresh" segment's
    own body would add roughly as much again once a filler's reuse call
    actually completes, so this is a floor on real footprint, not an
    exact total -- reported alongside ``observed_rho_after_pressure`` in
    ``register_eviction_pressure_objects``'s own returned dict for
    downstream debugging, never used to gate/validate behaviour itself.
    """
    return sum(workload.body_tokens for workload in workloads)


def eviction_pressure_filler_count_for_rho(
    *,
    target_rho: float,
    usable_capacity_tokens: int,
    tokens_per_filler: int,
) -> int:
    """Reverse-compute how many filler objects (each contributing
    ``tokens_per_filler`` tokens, see ``eviction_pressure_total_tokens``)
    are needed so their combined *nominal* (requested, not sampled-live)
    token footprint reaches at least
    ``target_rho * usable_capacity_tokens``.

    This is the "actual capacity自动反算...所需filler数" calculation: a
    real, live ``usable_kv_capacity_tokens`` reading (see
    ``benchmark.approx_kv.metrics``) drives how many filler objects a
    given ``target_rho`` requires, rather than a fixed filler count
    guessed independently of the server's real pool size. It is
    ``ceil``-rounded so the achieved nominal ratio is always >=
    ``target_rho``, never short of it by a fractional-filler rounding
    error.

    "Nominal" because it is computed purely from *requested* filler
    tokens; the pool's real, observed occupancy at any instant also
    depends on whatever eviction has already reclaimed by the time later
    fillers register (see ``observed_rho`` for the genuine, sampled
    counterpart, read from the live ``sglang:kv_used_tokens`` gauge) --
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
    target_total_tokens = target_rho * usable_capacity_tokens
    filler_count = math.ceil(target_total_tokens / tokens_per_filler)
    if filler_count > MAX_REASONABLE_EVICTION_PRESSURE_FILLER_COUNT:
        raise ValueError(
            f"reverse-computed filler_count={filler_count} exceeds the "
            f"sanity bound of {MAX_REASONABLE_EVICTION_PRESSURE_FILLER_COUNT} "
            f"(target_rho={target_rho}, "
            f"usable_capacity_tokens={usable_capacity_tokens}, "
            f"tokens_per_filler={tokens_per_filler}) -- this would require "
            "thousands of real, blocking per-filler HTTP round trips "
            "before this setting's own measurement even begins; raise "
            "--pressure-filler-body-tokens (fewer, larger fillers needed "
            "for the same target_rho) or lower --main-target-rho/"
            "--target-rho-choices instead"
        )
    return filler_count


def observed_rho(snapshot: Mapping[str, float], *, capacity_tokens: int) -> float:
    """The real, sampled ratio of this ``snapshot``'s live
    ``sglang:kv_used_tokens`` gauge to a fixed ``capacity_tokens``
    reference -- the genuine, *measured* occupancy fraction at the
    instant ``snapshot`` was taken, as opposed to
    ``eviction_pressure_filler_count_for_rho``'s nominal (requested-
    tokens) ratio.

    ``capacity_tokens`` is deliberately a caller-supplied fixed value
    (established once, immediately after a flush, via
    ``usable_kv_capacity_tokens`` on a genuinely idle pool snapshot) --
    never recomputed from ``snapshot`` itself here, since
    ``usable_kv_capacity_tokens``'s own idle heuristic could react to a
    transient, eviction-driven usage dip in a snapshot taken mid-
    pressure and silently swap its capacity basis out from under a
    "peak rho" comparison across multiple snapshots of the same setting.
    """
    if capacity_tokens <= 0:
        raise ValueError(f"capacity_tokens must be positive, got {capacity_tokens}")
    used = snapshot.get("sglang:kv_used_tokens")
    if used is None:
        raise ValueError(
            "sglang:kv_used_tokens is unavailable in this snapshot -- cannot "
            "compute observed_rho without it"
        )
    return float(used) / float(capacity_tokens)


def chunk_offsets(
    total_tokens: int, max_chunk_tokens: int
) -> tuple[tuple[int, int], ...]:
    """Split ``total_tokens`` into contiguous ``(offset, length)`` chunks,
    each at most ``max_chunk_tokens`` long, offsets 0-based relative to
    the start of the span being chunked.

    Used to keep every single approx_kv segment this canary registers at
    most ``--max-segment-chunk-tokens`` long (default 512): with the
    unified body sweep now reaching up to 2048 tokens, a body longer
    than that is split into multiple segments within one register/reuse
    call rather than ever registering one oversized segment (see
    ``body_segments_for_hash``). ``ApproxKVRequestMetadata``/
    ``register_request_segments``/``restore_request_prefix_cachetune``
    already natively iterate over an arbitrary number of segments per
    call, so this chunking changes nothing about the underlying server-
    side contract -- only how this client divides one logical body into
    that call's ``segments`` list.
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
    """Build a ``register``/``reuse`` payload ``"segments"`` list for a
    ``body_tokens``-long span anchored at ``body_start``, split into
    ``chunk_offsets(body_tokens, max_chunk_tokens)`` pieces.

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
    "target_start", "length"}`` dicts (see ``body_segments_for_hash``):
    a body longer than ``--max-segment-chunk-tokens`` is registered as
    MULTIPLE segments in this single call, never one oversized segment
    -- ``register_request_segments`` (``cachetune/runtime.py``) already
    natively iterates over an arbitrary number of segments per call."""
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


def materialize_workload_via_reuse(
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
    """Seed ``workload``'s own exact-match target head, register its raw
    (source-context) and fresh (target-context) body segments, then
    issue exactly one real reuse request against it -- forcing
    CacheTune's genuine repair path to run once and materialize the raw
    segment onto the device residency tier via ``ensure_device`` (see
    ``cachetune/runtime.py``).

    A body longer than ``max_chunk_tokens`` is registered/reused as
    multiple <= ``max_chunk_tokens`` segments within each single call
    (see ``body_segments_for_hash``), never one oversized segment.

    Shared by ``run_non_prefix_setting``'s own one-time setup + discarded
    warmup pass (seed head -> register raw -> register fresh -> reuse,
    exactly this sequence) and by ``register_eviction_pressure_objects``
    (each pressure filler object needs this exact same one-time
    materialization, never a timed/repeated measurement). Every step is
    validated the same way the rest of this script validates every
    request -- never silently swallowed.

    Returns a dict with ``seed_head_ms``, ``register_raw_ms``,
    ``register_fresh_ms``, ``reuse_ms`` (every step's own genuine
    streaming TTFT) and ``reuse_response`` (the final reuse call's parsed
    JSON body) so callers needing granular per-step timing
    (``run_non_prefix_setting``'s own setup) and callers that only need
    "did this materialize successfully" (each pressure filler object)
    can both use this single, already-validated code path.
    """
    seed_response, seed_head_ms = timed_post(
        base_url, dense_generate_payload(workload.target_head_ids)
    )
    require_finished_by_length(seed_response, f"{label} seed target_head")
    require_cached_tokens(seed_response, 0, f"{label} seed target_head")

    register_raw_response, register_raw_ms = timed_post(
        base_url,
        register_generate_payload(
            input_ids=workload.source_prompt_ids,
            segments=body_segments_for_hash(
                hash_prefix=raw_hash,
                body_start=workload.body_start_in_source,
                body_tokens=workload.body_tokens,
                max_chunk_tokens=max_chunk_tokens,
            ),
            model_fingerprint=model_fingerprint,
            cache_dtype=cache_dtype,
        ),
    )
    require_finished_by_length(register_raw_response, f"{label} raw register")
    require_cached_tokens(register_raw_response, 0, f"{label} raw register")

    register_fresh_response, register_fresh_ms = timed_post(
        base_url,
        register_generate_payload(
            input_ids=workload.fresh_prompt_ids,
            segments=body_segments_for_hash(
                hash_prefix=fresh_hash,
                body_start=workload.body_start_in_target,
                body_tokens=workload.body_tokens,
                max_chunk_tokens=max_chunk_tokens,
            ),
            model_fingerprint=model_fingerprint,
            cache_dtype=cache_dtype,
        ),
    )
    require_finished_by_length(register_fresh_response, f"{label} fresh preparation")
    # REGISTER never restores (see approx_kv/runtime.py's
    # _register_request_segments -- it bails out immediately unless
    # operation == REUSE), so this call's only contribution to
    # prefix_indices is the plain exact-match radix hit on
    # target_head_ids, already seeded above: body_start_in_target, not 0.
    require_cached_tokens(
        register_fresh_response,
        workload.body_start_in_target,
        f"{label} fresh preparation",
    )

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
    require_cached_tokens(
        reuse_response,
        workload.body_start_in_target + workload.body_tokens,
        f"{label} reuse",
    )

    return {
        "seed_head_ms": seed_head_ms,
        "register_raw_ms": register_raw_ms,
        "register_fresh_ms": register_fresh_ms,
        "reuse_ms": reuse_ms,
        "reuse_response": reuse_response,
    }


def register_eviction_pressure_objects(
    base_url: str,
    workloads: Sequence[NonPrefixSegmentWorkload],
    *,
    model_fingerprint: str,
    cache_dtype: str,
    label: str,
    max_chunk_tokens: int,
    capacity_tokens: int,
    target_rho: float,
) -> dict[str, Any]:
    """Register and materialize every eviction-pressure filler object in
    ``workloads`` (see ``build_eviction_pressure_workloads``) via
    ``materialize_workload_via_reuse``, so their raw segments become
    genuinely device-resident (via ``ensure_device``) before the
    caller's own setting measurement begins.

    Snapshots ``/metrics`` immediately before the first filler object and
    immediately after the last one, and raises loudly if
    ``sglang:approx_kv_dense_fallback_total`` increased during this
    phase: a filler object silently falling back to dense would mean its
    body was never actually restored via CacheTune's repair path at all
    (a real bug, or a misconfigured pressure request whose footprint
    cannot fit in the pool even after evicting everything else), which
    this canary must never treat as a harmless, ignorable detail.

    ``capacity_tokens`` is the fixed, idle-pool capacity reference
    established once by the caller (immediately after its own flush, via
    ``usable_kv_capacity_tokens``) and ``target_rho`` is the nominal
    ratio ``workloads`` was reverse-sized for (see
    ``eviction_pressure_filler_count_for_rho``) -- both are only used
    here to report the genuine, *sampled* ``observed_rho`` (from the
    live ``sglang:kv_used_tokens`` gauge) immediately after this pressure
    phase completes, alongside the nominal target, never to alter
    behaviour.

    Returns a dict with ``object_count``, ``total_pressure_tokens``
    (see ``eviction_pressure_total_tokens``), ``target_rho`` (the
    nominal ask), ``capacity_tokens`` (the fixed reference),
    ``observed_rho_after_pressure`` (the genuine sampled ratio),
    ``evicted_tokens_total_delta`` (the genuine, real evidence that
    device-pool eviction actually happened during THIS pressure phase
    alone -- may legitimately be 0 if the configured pressure was not
    large enough to evict anything, and this is reported honestly rather
    than hidden), and ``dense_fallback_total_delta`` (always 0, given the
    raise above, kept for output-schema transparency), plus the raw
    ``metrics_before``/``metrics_after`` snapshots the two deltas above
    were computed from (surfaced verbatim for downstream debugging,
    exactly like ``run_non_prefix_setting``'s own ``metrics_before``/
    ``metrics_after`` keys).
    """
    metrics_before = metric_snapshot(base_url)
    for index, filler_workload in enumerate(workloads):
        materialize_workload_via_reuse(
            base_url,
            filler_workload,
            raw_hash=f"cachetune-raw:phase4-r5-pressure-filler-{index}",
            fresh_hash=f"cachetune-fresh:phase4-r5-pressure-filler-{index}",
            model_fingerprint=model_fingerprint,
            cache_dtype=cache_dtype,
            label=f"{label} pressure-filler[{index}]",
            max_chunk_tokens=max_chunk_tokens,
        )
        time.sleep(0.1)
    metrics_after = metric_snapshot(base_url)

    dense_fallback_delta = metric_delta(
        metrics_before, metrics_after, "sglang:approx_kv_dense_fallback_total"
    )
    if dense_fallback_delta != 0:
        raise RuntimeError(
            f"{label}: {len(workloads)} eviction-pressure filler object(s) "
            f"produced a nonzero dense_fallback delta of "
            f"{dense_fallback_delta} while materializing -- at least one "
            "filler's own reuse silently fell back to dense instead of a "
            "genuine CacheTune repair; this pressure configuration cannot "
            "be trusted to have actually occupied device residency as "
            "intended"
        )
    return {
        "object_count": len(workloads),
        "total_pressure_tokens": eviction_pressure_total_tokens(workloads),
        "target_rho": target_rho,
        "capacity_tokens": capacity_tokens,
        "observed_rho_after_pressure": observed_rho(
            metrics_after, capacity_tokens=capacity_tokens
        ),
        "evicted_tokens_total_delta": metric_delta(
            metrics_before, metrics_after, "sglang:evicted_tokens_total"
        ),
        "dense_fallback_total_delta": dense_fallback_delta,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
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
    """Flush the exact radix cache, optionally reverse-compute and
    materialize a fresh eviction-pressure phase sized for ``target_rho``,
    seed the exact-cache target head, register the raw (source-context)
    body segment once, run one discarded register-fresh + reuse warmup,
    snapshot Prometheus metrics, then run ``repeats`` formal
    register-fresh + reuse repeats.

    The shared measurement routine used by the main CacheTune setting,
    every shape-sweep point, and every rho-sweep point (see the module
    docstring's "Why /generate and non-prefix segments" section for why
    this flush -> [pressure] -> seed -> register-raw -> warmup -> repeats
    ordering is mandatory).

    The flush is this function's *own* first action -- not merely the
    caller's responsibility -- because this function runs once per
    setting and settings are otherwise never isolated from each other: a
    previous setting's own already-seeded ``target_head_ids`` would
    otherwise still be sitting in the tree, ready to silently produce a
    nonzero ``cached_tokens`` for an unrelated later setting's head-seed
    or raw-segment register request below. Safe to do here (never
    between the formal repeats further down -- see
    ``flush_exact_radix_cache``'s own docstring for why) precisely
    because it runs before any registration this call performs.

    Immediately after that flush, a real ``/metrics`` snapshot on the now
    genuinely idle pool gives ``capacity_tokens`` (via
    ``usable_kv_capacity_tokens``) -- a fixed reference used for this
    setting's own ``observed_rho`` calculations (never recomputed later,
    since a later snapshot taken mid-pressure could react to a
    transient, eviction-driven usage dip). If ``target_rho`` is not
    ``None``, the filler object count is reverse-computed from it against
    that capacity (see ``eviction_pressure_filler_count_for_rho``) and a
    fresh pressure phase is registered and materialized (see
    ``register_eviction_pressure_objects``): the flush above resets the
    entire ``approx_kv`` store (``ApproxKVManager.reset()``, wired
    through ``RadixCache.reset``/``UnifiedRadixCache.reset`` -- see
    ``flush_exact_radix_cache``'s own docstring), so a previous setting's
    already-registered filler objects are gone the moment this function's
    own flush runs; they cannot be built once globally and expected to
    persist across multiple settings. This means every setting (the main
    setting and every shape-/rho-sweep point alike) gets its own genuine,
    freshly-sized pressure phase, never a shared/stale one.
    ``validate_pairwise_head_isolation`` runs first (before any network
    call) to guard against a filler's dense-seeded target head colliding
    with this setting's own head or with another filler's head in the
    live exact radix tree.

    Returns a dict with ``seed_head_ms``, ``register_raw_ms``,
    ``fresh_raw_samples``, ``reuse_raw_samples`` (each a list of
    ``{"ttft_ms": float, "cached_tokens": int}`` records, one per formal
    repeat, pairing every repeat's genuine streaming TTFT with the
    server-reported ``meta_info.cached_tokens`` from that exact same
    call), ``fresh_ms_samples``, ``reuse_ms_samples``,
    ``combined_ms_samples`` (the ``ttft_ms``-only projections of the
    above, kept for existing consumers), ``observed_cached_tokens_per_call``
    (the reuse leg's ``cached_tokens`` projection, unchanged),
    ``metrics_before``/``metrics_after``, ``capacity_tokens`` (this
    setting's own fixed idle-pool reference), ``observed_rho_after_target``
    (the genuine, sampled ``sglang:kv_used_tokens`` ratio right after this
    setting's own formal repeats complete), ``peak_rho_observed`` (the
    greater of that and the pressure phase's own
    ``observed_rho_after_pressure``, or just the former when no pressure
    phase ran), ``pressure_and_target_evicted_tokens_total_delta`` (the
    cumulative real ``sglang:evicted_tokens_total`` delta spanning from
    immediately after this setting's own flush through the end of its
    formal repeats -- i.e. pressure-phase eviction PLUS whatever this
    setting's own target/source registration additionally evicted), and
    ``pressure_phase`` (``None`` when ``target_rho`` is ``None``,
    otherwise ``register_eviction_pressure_objects``'s own returned
    telemetry dict) -- everything the caller needs to build its own
    output section and telemetry cross-validation.
    """
    flush_exact_radix_cache(base_url)
    metrics_at_setting_start = metric_snapshot(base_url)
    capacity_tokens = usable_kv_capacity_tokens(metrics_at_setting_start)

    pressure_phase: dict[str, Any] | None = None
    if target_rho is not None:
        tokens_per_filler = pressure_filler_head_tokens + pressure_filler_body_tokens
        filler_count = eviction_pressure_filler_count_for_rho(
            target_rho=target_rho,
            usable_capacity_tokens=capacity_tokens,
            tokens_per_filler=tokens_per_filler,
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
            model_fingerprint=model_fingerprint,
            cache_dtype=cache_dtype,
            label=label,
            max_chunk_tokens=max_chunk_tokens,
            capacity_tokens=capacity_tokens,
            target_rho=target_rho,
        )

    setup_result = materialize_workload_via_reuse(
        base_url,
        workload,
        raw_hash=raw_hash,
        fresh_hash=fresh_hash,
        model_fingerprint=model_fingerprint,
        cache_dtype=cache_dtype,
        label=f"{label} setup+warmup (discarded)",
        max_chunk_tokens=max_chunk_tokens,
    )
    seed_head_ms = setup_result["seed_head_ms"]
    register_raw_ms = setup_result["register_raw_ms"]
    # setup_result's own register_fresh_ms/reuse_ms/reuse_response are
    # this call's discarded warmup pass -- intentionally not kept, exactly
    # like the previous inline implementation discarded them.

    # Snapshot AFTER warmup: the warmup's own telemetry contribution must
    # not be counted as part of the formal-repeat delta below.
    metrics_before = metric_snapshot(base_url)
    fresh_raw_samples: list[dict[str, Any]] = []
    reuse_raw_samples: list[dict[str, Any]] = []
    for _ in range(repeats):
        register_fresh_response, fresh_ttft_ms = timed_post(
            base_url,
            register_generate_payload(
                input_ids=workload.fresh_prompt_ids,
                segments=body_segments_for_hash(
                    hash_prefix=fresh_hash,
                    body_start=workload.body_start_in_target,
                    body_tokens=workload.body_tokens,
                    max_chunk_tokens=max_chunk_tokens,
                ),
                model_fingerprint=model_fingerprint,
                cache_dtype=cache_dtype,
            ),
        )
        require_finished_by_length(
            register_fresh_response, f"{label} fresh preparation"
        )
        # Same rationale as materialize_workload_via_reuse's own fresh
        # register assertion: REGISTER never restores, so the only
        # contribution to prefix_indices is the exact-match radix hit on
        # the already-seeded target head (body_start_in_target), for
        # every formal repeat identically (register/reuse requests never
        # write into the exact radix tree, so this never drifts across
        # repeats).
        require_cached_tokens(
            register_fresh_response,
            workload.body_start_in_target,
            f"{label} fresh preparation",
        )

        reuse_response, reuse_ttft_ms = timed_post(
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
        # Full restored prefix (head + entire body), not head-only --
        # see require_cached_tokens's own docstring for why a successful
        # CacheTune reuse always extends prefix_indices by the complete
        # restore_length regardless of the controller's selected ratio.
        observed_reuse_cached_tokens = require_cached_tokens(
            reuse_response,
            workload.body_start_in_target + workload.body_tokens,
            f"{label} reuse",
        )

        fresh_raw_samples.append(
            {
                "ttft_ms": fresh_ttft_ms,
                "cached_tokens": int(
                    register_fresh_response["meta_info"]["cached_tokens"]
                ),
            }
        )
        reuse_raw_samples.append(
            {"ttft_ms": reuse_ttft_ms, "cached_tokens": observed_reuse_cached_tokens}
        )
        time.sleep(0.1)
    metrics_after = metric_snapshot(base_url)
    observed_rho_after_target = observed_rho(
        metrics_after, capacity_tokens=capacity_tokens
    )
    peak_rho_observed = (
        max(observed_rho_after_target, pressure_phase["observed_rho_after_pressure"])
        if pressure_phase is not None
        else observed_rho_after_target
    )

    fresh_ms_samples = [sample["ttft_ms"] for sample in fresh_raw_samples]
    reuse_ms_samples = [sample["ttft_ms"] for sample in reuse_raw_samples]
    return {
        "seed_head_ms": seed_head_ms,
        "register_raw_ms": register_raw_ms,
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
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "capacity_tokens": capacity_tokens,
        "observed_rho_after_target": observed_rho_after_target,
        "peak_rho_observed": peak_rho_observed,
        "pressure_and_target_evicted_tokens_total_delta": metric_delta(
            metrics_at_setting_start, metrics_after, "sglang:evicted_tokens_total"
        ),
        "pressure_phase": pressure_phase,
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
    ``expected_repair_totals``.
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
        "target_rho": pressure_phase["target_rho"] if pressure_phase else None,
        "observed_rho_after_target": setting_result["observed_rho_after_target"],
        "peak_rho_observed": setting_result["peak_rho_observed"],
        "pressure_and_target_evicted_tokens_total_delta": (
            setting_result["pressure_and_target_evicted_tokens_total_delta"]
        ),
        "pressure_phase": pressure_phase,
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
    # run_non_prefix_setting itself, per setting, from a real, live, idle
    # /metrics snapshot taken immediately after that call's own flush --
    # never built once globally here, since a globally-built filler set
    # would not survive past the very first setting's own flush (see
    # flush_exact_radix_cache's docstring) and since the correct filler
    # count depends on each setting's own freshly-measured capacity. ----
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
    # run_non_prefix_setting flushes it away as its own first action
    # below (see that function's docstring) before seeding the head or
    # registering anything -- so the CacheTune reuse requests it issues
    # cannot be silently served by that stale exact-cache entry instead
    # of the real approximate-repair path.

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

    metrics_final = metric_snapshot(args.base_url)
    pool_invariant = idle_pool_invariant(metrics_final)
    health_status = fetch_text(f"{args.base_url}/health")

    known_limitations = [
        "Only one (roofline-derived) ratio configuration received a real "
        f"SM75 server canary in this result: r0={r0:.4f} under mode={mode.value}.",
        "Fresh target-context KV is generated by an explicit dense "
        "preparation request; its cost is included in combined_p50_ms.",
        "seed_head_ms and register_raw_ms are one-time per-setting setup "
        "costs (seeding the exact-match head, and capturing the raw "
        "segment's source-context KV) and are reported but excluded from "
        "combined_ms/combined_p50_ms -- analogous to why the raw "
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
        "runs its own freshly reverse-computed, freshly-materialized "
        "multi-object eviction-pressure phase sized against that "
        "setting's own real, live, idle usable_kv_capacity_tokens "
        "snapshot (see eviction_pressure_filler_count_for_rho / "
        "register_eviction_pressure_objects) -- never a single globally-"
        "shared filler set and never CLI-disableable to a single-object "
        "microbenchmark; whether real device-pool eviction actually "
        "occurred for a given setting must be read from that setting's "
        "own pressure_phase.evicted_tokens_total_delta / "
        "peak_rho_observed / observed_rho_after_target (may legitimately "
        "differ from the nominal target_rho -- reported honestly, never "
        "hidden). The shape sweep's header=0 exact-context control "
        "points are the one deliberate exception: they run no pressure "
        "phase at all and carry no pressure_phase/capacity_tokens/"
        "peak_rho_observed keys (see the next limitation)."
    )
    known_limitations.append(
        "The shape sweep's header=0 points are an exact-context control "
        "point (dense-only, is_exact_context_control=True), never a "
        "genuine CacheTune repair measurement -- see "
        "run_exact_context_control_point's own docstring for why header=0 "
        "cannot build a NonPrefixSegmentWorkload at all."
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
            "exact_radix_flush_at_start_of_every_setting": True,
            "exact_radix_flush_at_start_of_every_setting_rationale": (
                "run_non_prefix_setting flushes the exact-match radix "
                "cache as its own first action, once per setting (the "
                "main setting, every shape-sweep point with header > 0, "
                "and every rho-sweep point) -- otherwise a previous "
                "setting's own already-seeded target_head_ids would "
                "still be sitting in the tree, able to silently produce "
                "a nonzero cached_tokens for an unrelated later setting's "
                "head-seed or raw-segment register request."
            ),
            "cachetune_reuse_flush_between_repeats": False,
            "cachetune_reuse_flush_rationale": (
                "register/reuse requests always set "
                "schedule_batch.Req.skip_radix_cache_insert=True whenever "
                "approx_kv_metadata is present, so they never populate "
                "the exact-match radix tree and cannot exact-hit each "
                "other; only dense baseline requests (no approx_kv "
                "metadata) do, which is why only dense needs inter-repeat "
                "flushing."
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
            "eviction_pressure_phase_runs_once_per_setting": True,
            "eviction_pressure_phase_rationale": (
                "every setting's own flush (see "
                "exact_radix_flush_at_start_of_every_setting) wipes the "
                "entire approx_kv store, so eviction-pressure filler "
                "objects are freshly reverse-computed (from that "
                "setting's own target_rho against a real, live, idle "
                "/metrics snapshot) and re-materialized inside every "
                "run_non_prefix_setting call (main setting, every "
                "shape-sweep point, and every rho-sweep point alike), "
                "never built once globally -- see "
                "eviction_pressure_filler_count_for_rho / "
                "register_eviction_pressure_objects."
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
            "target_rho": args.main_target_rho,
            "observed_rho_after_target": main_result["observed_rho_after_target"],
            "peak_rho_observed": main_result["peak_rho_observed"],
            "pressure_and_target_evicted_tokens_total_delta": (
                main_result["pressure_and_target_evicted_tokens_total_delta"]
            ),
            "pressure_phase": main_pressure_phase,
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
