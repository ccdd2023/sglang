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

* ``source_prompt = source_head_ids + shared_body_ids``
* ``target_prompt = target_head_ids + shared_body_ids + tail_ids``

with ``source_head_ids != target_head_ids`` (different token content --
the whole point) but ``shared_body_ids`` byte-identical between the two.
The "raw" segment is registered from the *source* prompt's body offset;
the "fresh"/reuse segment is registered/matched from the *target*
prompt's body offset. Each of source_head/target_head/shared_body/tail is
tokenized *separately* (``tokenizer.encode(text, add_special_tokens=False)``)
and the resulting integer-id lists are concatenated directly -- never
re-tokenized as a joined string -- so segment offsets are exact by
construction, with no BPE boundary-merging risk across piece boundaries.

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
``source_head_ids + shared_body_ids``) -> a discarded register-fresh +
reuse warmup -> ``--repeats`` formal register-fresh + reuse repeats. The
mandatory ``tail_ids`` (fixed at ``NON_PREFIX_TAIL_TOKENS`` token(s))
ensures every reuse request still has a genuine final forward pass beyond
the restored range, matching ``ApproxKVRequestMetadata``'s own invariant
that a request's last prompt token is never included in any restorable
segment.

Measurement protocol (mandatory, applies to every setting and every
length-sweep point)
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
   function runs once per *setting* (the main setting and every
   length-sweep point), settings are otherwise never isolated from each
   other, and a *previous* setting's own already-seeded
   ``target_head_ids`` would otherwise still be sitting in the tree,
   ready to silently produce a nonzero ``cached_tokens`` for an unrelated
   later setting's head or raw-segment register request.
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
   completes). Every formal reuse response's ``meta_info.cached_tokens``
   is checked against the expected head-only length -- an independent,
   per-request, server-reported signal (generic SGLang exact-prefix
   accounting, unrelated to CacheTune's own Prometheus counters) that the
   live request's own exact-match boundary landed exactly where the
   registered segment expects, not a tautology against the same
   telemetry this script already cross-validates in aggregate.

Register/reuse requests never need inter-repeat flushing:
``schedule_batch.Req.skip_radix_cache_insert`` is forced True whenever
``approx_kv_metadata`` is present, so they can never populate the exact
radix tree themselves; flushing between formal repeats would also invoke
``ApproxKVManager.reset()`` (see ``mem_cache/approx_kv/manager.py``),
deleting the very "raw"/"fresh" segments those repeats depend on.

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
import statistics
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import aiohttp

from benchmark.approx_kv.metrics import idle_pool_invariant, parse_prometheus_text
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

# Fixed, non-CLI construction constants for every NonPrefixSegmentWorkload
# this canary builds (main setting and every length-sweep point alike): a
# stable head length keeps the one-time seed/register-raw setup cost
# comparable across settings, and only body length is meant to vary
# ("target head固定34即可").
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

# Every setting (dense, the main CacheTune point, and each length-sweep
# point) runs exactly this many *discarded* passes before the formal
# repeats begin. This is a fixed measurement-protocol constant, not a CLI
# knob, so every canary result is comparable under the same discipline.
WARMUP_PASSES_PER_SETTING = 1


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
        "--body-tokens",
        type=int,
        default=256,
        help="Shared-body token count for the main setting's "
        "NonPrefixSegmentWorkload (see module docstring). Head length is "
        f"fixed at {NON_PREFIX_HEAD_TOKENS} tokens, tail at "
        f"{NON_PREFIX_TAIL_TOKENS} token(s), for every setting.",
    )
    parser.add_argument(
        "--length-sweep",
        default="128,512",
        help="Comma-separated additional shared-body token counts used "
        "to prove real per-length deterministic re-quantization within "
        "the same running server/controller (no server restart); head "
        "and tail length stay fixed across every sweep point.",
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
    return parser.parse_args()


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

    ``source_prompt_ids = source_head_ids + shared_body_ids`` is the
    prompt CacheTune's "raw" segment is registered from (captures the
    body's KV under *source* context); ``target_prompt_ids =
    target_head_ids + shared_body_ids + tail_ids`` is the prompt the
    "fresh" segment is registered from and the reuse request actually
    targets. ``source_head_ids`` and ``target_head_ids`` are required to
    differ: this is what makes the body's source and target KV genuinely
    distinct (not just an exact-content replay of the same context) --
    see the module docstring's "Why /generate and non-prefix segments"
    section for the full rationale.
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
        return self.source_head_ids + self.shared_body_ids

    @property
    def target_prompt_ids(self) -> tuple[int, ...]:
        return self.target_head_ids + self.shared_body_ids + self.tail_ids

    @property
    def fresh_prompt_ids(self) -> tuple[int, ...]:
        """``target_head_ids + shared_body_ids`` (no tail): the minimal
        prompt needed to capture the body's real KV under *target*
        context for the precomputed fresh-KV adapter (see
        ``cachetune/precomputed.py``)."""
        return self.target_head_ids + self.shared_body_ids

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
) -> NonPrefixSegmentWorkload:
    """Build one ``NonPrefixSegmentWorkload`` from four independently
    tokenized, deterministic pieces (see ``_deterministic_token_ids``).

    ``salt`` must be unique per distinct workload the canary constructs
    (the main setting and each length-sweep point each pass their own
    salt) so that different settings' bodies/tails never share content.
    Cross-setting *head* isolation in the live server's exact radix tree
    is guaranteed separately, by ``run_non_prefix_setting`` flushing that
    tree as its own first action for every setting -- salt uniqueness
    alone would not be enough for that, since ``source_head_ids``/
    ``target_head_ids`` need the stronger, structural
    zero-common-prefix guarantee the literal-prefix markers below provide
    (see the module docstring's "Why /generate and non-prefix segments"
    section).
    """
    return NonPrefixSegmentWorkload(
        source_head_ids=_deterministic_token_ids(
            tokenizer,
            f"{salt}-source-head",
            head_tokens,
            literal_prefix=_SOURCE_HEAD_LITERAL_PREFIX,
        ),
        target_head_ids=_deterministic_token_ids(
            tokenizer,
            f"{salt}-target-head",
            head_tokens,
            literal_prefix=_TARGET_HEAD_LITERAL_PREFIX,
        ),
        shared_body_ids=_deterministic_token_ids(
            tokenizer, f"{salt}-shared-body", body_tokens
        ),
        tail_ids=_deterministic_token_ids(tokenizer, f"{salt}-tail", tail_tokens),
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
    content_hash: str,
    target_start: int,
    length: int,
    model_fingerprint: str,
    cache_dtype: str,
) -> dict:
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
                    "segments": [
                        {
                            "content_hash": content_hash,
                            "target_start": target_start,
                            "length": length,
                        }
                    ],
                }
            },
        },
    }


def reuse_generate_payload(
    *,
    input_ids: Sequence[int],
    raw_content_hash: str,
    target_start: int,
    length: int,
    model_fingerprint: str,
    cache_dtype: str,
) -> dict:
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
                    "segments": [
                        {
                            "content_hash": raw_content_hash,
                            "target_start": target_start,
                            "length": length,
                        }
                    ],
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
    """Assert the server-reported exact-match prefix length
    (``meta_info.cached_tokens``, generic SGLang accounting set from
    ``pre_len - already_computed`` in ``schedule_batch.py`` -- unrelated
    to any CacheTune-specific Prometheus counter) equals ``expected`` for
    this specific request, and return the observed value.

    This is an independent, per-request cross-check that the live
    request's own exact-cache boundary landed exactly where this
    canary's registered segment expects it to -- not a tautology against
    the aggregate Prometheus deltas this script already cross-validates
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
        "body_tokens": args.body_tokens,
        "head_tokens": NON_PREFIX_HEAD_TOKENS,
        "tail_tokens": NON_PREFIX_TAIL_TOKENS,
        "length_sweep": args.length_sweep,
        "repeats_per_setting": args.repeats,
        "warmup_passes_per_setting": WARMUP_PASSES_PER_SETTING,
        "runner_git_sha": args.runner_git_sha,
        "image_digest": args.image_digest,
        "scheduler": "S0 LRU",
        "tier": "GPU-only",
        "prefetch": False,
        "accuracy_metric": False,
    }


def run_non_prefix_setting(
    *,
    base_url: str,
    workload: NonPrefixSegmentWorkload,
    raw_hash: str,
    fresh_hash: str,
    model_fingerprint: str,
    cache_dtype: str,
    repeats: int,
    label: str,
) -> dict[str, Any]:
    """Flush the exact radix cache, seed the exact-cache target head,
    register the raw (source-context) body segment once, run one
    discarded register-fresh + reuse warmup, snapshot Prometheus metrics,
    then run ``repeats`` formal register-fresh + reuse repeats.

    The shared measurement routine used by both the main CacheTune
    setting and every length-sweep point (see the module docstring's
    "Why /generate and non-prefix segments" section for why this
    flush -> seed -> register-raw -> warmup -> repeats ordering is
    mandatory).

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

    Returns a dict with ``seed_head_ms``, ``register_raw_ms``,
    ``fresh_raw_samples``, ``reuse_raw_samples`` (each a list of
    ``{"ttft_ms": float, "cached_tokens": int}`` records, one per formal
    repeat, pairing every repeat's genuine streaming TTFT with the
    server-reported ``meta_info.cached_tokens`` from that exact same
    call), ``fresh_ms_samples``, ``reuse_ms_samples``,
    ``combined_ms_samples`` (the ``ttft_ms``-only projections of the
    above, kept for existing consumers), ``observed_cached_tokens_per_call``
    (the reuse leg's ``cached_tokens`` projection, unchanged), and
    ``metrics_before``/``metrics_after`` -- everything the caller needs
    to build its own output section and telemetry cross-validation.
    """
    flush_exact_radix_cache(base_url)

    seed_response, seed_head_ms = timed_post(
        base_url, dense_generate_payload(workload.target_head_ids)
    )
    require_finished_by_length(seed_response, f"{label} seed target_head (discarded)")
    require_cached_tokens(seed_response, 0, f"{label} seed target_head (discarded)")

    register_raw_response, register_raw_ms = timed_post(
        base_url,
        register_generate_payload(
            input_ids=workload.source_prompt_ids,
            content_hash=raw_hash,
            target_start=workload.body_start_in_source,
            length=workload.body_tokens,
            model_fingerprint=model_fingerprint,
            cache_dtype=cache_dtype,
        ),
    )
    require_finished_by_length(register_raw_response, f"{label} raw register")
    require_cached_tokens(register_raw_response, 0, f"{label} raw register")

    # Discarded warmup (register fresh + reuse): register/reuse requests
    # never write into the exact radix tree (skip_radix_cache_insert is
    # forced True whenever approx_kv_metadata is present), so repeating
    # them -- including this warmup -- needs no flush.
    warmup_fresh_response, _ = timed_post(
        base_url,
        register_generate_payload(
            input_ids=workload.fresh_prompt_ids,
            content_hash=fresh_hash,
            target_start=workload.body_start_in_target,
            length=workload.body_tokens,
            model_fingerprint=model_fingerprint,
            cache_dtype=cache_dtype,
        ),
    )
    require_finished_by_length(
        warmup_fresh_response, f"{label} warmup fresh preparation (discarded)"
    )
    warmup_reuse_response, _ = timed_post(
        base_url,
        reuse_generate_payload(
            input_ids=workload.target_prompt_ids,
            raw_content_hash=raw_hash,
            target_start=workload.body_start_in_target,
            length=workload.body_tokens,
            model_fingerprint=model_fingerprint,
            cache_dtype=cache_dtype,
        ),
    )
    require_finished_by_length(
        warmup_reuse_response, f"{label} warmup reuse (discarded)"
    )

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
                content_hash=fresh_hash,
                target_start=workload.body_start_in_target,
                length=workload.body_tokens,
                model_fingerprint=model_fingerprint,
                cache_dtype=cache_dtype,
            ),
        )
        require_finished_by_length(
            register_fresh_response, f"{label} fresh preparation"
        )

        reuse_response, reuse_ttft_ms = timed_post(
            base_url,
            reuse_generate_payload(
                input_ids=workload.target_prompt_ids,
                raw_content_hash=raw_hash,
                target_start=workload.body_start_in_target,
                length=workload.body_tokens,
                model_fingerprint=model_fingerprint,
                cache_dtype=cache_dtype,
            ),
        )
        require_finished_by_length(reuse_response, f"{label} reuse")
        observed_reuse_cached_tokens = require_cached_tokens(
            reuse_response, workload.body_start_in_target, f"{label} reuse"
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
    # segments" for why source_head != target_head is mandatory). -------
    main_workload = build_non_prefix_segment_workload(
        tokenizer,
        body_tokens=args.body_tokens,
        head_tokens=NON_PREFIX_HEAD_TOKENS,
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
        workload=main_workload,
        raw_hash=raw_hash,
        fresh_hash=fresh_hash,
        model_fingerprint=args.model_fingerprint,
        cache_dtype=args.cache_dtype,
        repeats=args.repeats,
        label="main",
    )
    fresh_raw_samples = main_result["fresh_raw_samples"]
    cachetune_raw_samples = main_result["reuse_raw_samples"]
    fresh_ms_samples = main_result["fresh_ms_samples"]
    cachetune_ms_samples = main_result["reuse_ms_samples"]
    combined_ms_samples = main_result["combined_ms_samples"]
    observed_cached_tokens_per_call = main_result["observed_cached_tokens_per_call"]
    metrics_before_cachetune = main_result["metrics_before"]
    metrics_after_cachetune = main_result["metrics_after"]

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
        "cached_tokens_matches_head_only_every_call": all(
            observed == main_workload.body_start_in_target
            for observed in observed_cached_tokens_per_call
        ),
    }
    if not all(telemetry_checks.values()):
        raise RuntimeError(f"telemetry cross-validation failed: {telemetry_checks}")

    dense_p50_ms = statistics.median(dense_ms_samples)
    cachetune_target_p50_ms = statistics.median(cachetune_ms_samples)
    fresh_preparation_p50_ms = statistics.median(fresh_ms_samples)
    combined_p50_ms = statistics.median(combined_ms_samples)

    # ---- Real per-length re-quantization sweep (no server restart) -----
    length_sweep_targets = [
        int(value) for value in args.length_sweep.split(",") if value.strip()
    ]
    length_sweep_points: list[dict[str, Any]] = []
    for body_tokens in length_sweep_targets:
        sweep_workload = build_non_prefix_segment_workload(
            tokenizer,
            body_tokens=body_tokens,
            head_tokens=NON_PREFIX_HEAD_TOKENS,
            tail_tokens=NON_PREFIX_TAIL_TOKENS,
            salt=f"{CACHE_SALT}-sweep-{body_tokens}",
        )
        quantized = quantize_ratio(
            r0, context_length=sweep_workload.body_tokens, bounds=bounds
        )
        artifact = f"phase4-r5-cachetune-sweep-{body_tokens}"
        sweep_raw_hash = f"cachetune-raw:{artifact}"
        sweep_fresh_hash = f"cachetune-fresh:{artifact}"

        sweep_result = run_non_prefix_setting(
            base_url=args.base_url,
            workload=sweep_workload,
            raw_hash=sweep_raw_hash,
            fresh_hash=sweep_fresh_hash,
            model_fingerprint=args.model_fingerprint,
            cache_dtype=args.cache_dtype,
            repeats=args.repeats,
            label=f"sweep[{body_tokens}]",
        )
        observed_selected_tokens_total = metric_delta(
            sweep_result["metrics_before"],
            sweep_result["metrics_after"],
            "sglang:approx_kv_cachetune_selected_tokens_total",
        )
        observed_dense_fallback = metric_delta(
            sweep_result["metrics_before"],
            sweep_result["metrics_after"],
            "sglang:approx_kv_dense_fallback_total",
        )
        point_expected = expected_repair_totals(
            repair_tokens_per_call=quantized.repair_tokens,
            recomputed_layers_per_call=0,  # not tracked per sweep point
            repeats=args.repeats,
        )
        expected_selected_tokens_for_point = point_expected[
            "expected_selected_tokens_total"
        ]
        cached_tokens_ok = all(
            observed == sweep_workload.body_start_in_target
            for observed in sweep_result["observed_cached_tokens_per_call"]
        )
        length_sweep_points.append(
            {
                "body_tokens": sweep_workload.body_tokens,
                "head_tokens": NON_PREFIX_HEAD_TOKENS,
                "tail_tokens": NON_PREFIX_TAIL_TOKENS,
                "body_source_context_differs_from_target": (
                    sweep_workload.body_source_context_differs_from_target
                ),
                "repeats": args.repeats,
                "seed_head_ms": sweep_result["seed_head_ms"],
                "register_raw_ms": sweep_result["register_raw_ms"],
                "expected_selected_tokens_per_call": quantized.repair_tokens,
                "expected_selected_tokens_total": expected_selected_tokens_for_point,
                "expected_executable_ratio": quantized.executable_ratio,
                "observed_selected_tokens_total": observed_selected_tokens_total,
                "observed_dense_fallback": observed_dense_fallback,
                "observed_cached_tokens_per_call": (
                    sweep_result["observed_cached_tokens_per_call"]
                ),
                "expected_cached_tokens_per_call": sweep_workload.body_start_in_target,
                "fresh_raw_samples": sweep_result["fresh_raw_samples"],
                "reuse_raw_samples": sweep_result["reuse_raw_samples"],
                "fresh_ms_samples": sweep_result["fresh_ms_samples"],
                "reuse_ms_samples": sweep_result["reuse_ms_samples"],
                "combined_ms_samples": sweep_result["combined_ms_samples"],
                "fresh_p50_ms": statistics.median(sweep_result["fresh_ms_samples"]),
                "reuse_p50_ms": statistics.median(sweep_result["reuse_ms_samples"]),
                "combined_p50_ms": statistics.median(
                    sweep_result["combined_ms_samples"]
                ),
                "passed": (
                    observed_selected_tokens_total == expected_selected_tokens_for_point
                    and observed_dense_fallback == 0
                    and cached_tokens_ok
                ),
            }
        )
        time.sleep(0.1)

    if not all(point["passed"] for point in length_sweep_points):
        raise RuntimeError(f"length sweep validation failed: {length_sweep_points}")

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

    payload = {
        "schema_version": 3,
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
                "main setting and every length-sweep point) -- otherwise "
                "a previous setting's own already-seeded target_head_ids "
                "would still be sitting in the tree, able to silently "
                "produce a nonzero cached_tokens for an unrelated later "
                "setting's head-seed or raw-segment register request."
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
        },
        "workload": {
            "kind": "non_prefix_segment",
            "endpoint": "/generate",
            "head_tokens": NON_PREFIX_HEAD_TOKENS,
            "tail_tokens": NON_PREFIX_TAIL_TOKENS,
            "body_tokens": args.body_tokens,
            "description": (
                "source_prompt = source_head_ids + shared_body_ids; "
                "target_prompt = target_head_ids + shared_body_ids + "
                "tail_ids; source_head_ids != target_head_ids by "
                "construction (see NonPrefixSegmentWorkload), so the "
                "registered raw (source-context) and fresh (target-"
                "context) segments capture genuinely different "
                "preceding-context KV for the byte-identical shared body."
            ),
        },
        "hardware_measurement": {
            "t_c_ms": args.t_c_ms,
            "t_i_ms": args.t_i_ms,
            "t_o_ms": args.t_o_ms,
            "roofline_ratio_r0": r0,
        },
        "server_validation": {
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
            "expected_cached_tokens_per_call": main_workload.body_start_in_target,
            "seed_head_ms": main_result["seed_head_ms"],
            "register_raw_ms": main_result["register_raw_ms"],
            "last_prompt_token_real_forward": True,
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
        "length_sweep_points": length_sweep_points,
        "pool_invariant": pool_invariant,
        "health_response": health_status,
        "known_limitations": known_limitations,
        "passed": all(telemetry_checks.values())
        and all(point["passed"] for point in length_sweep_points)
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
                "length_sweep_points": len(payload["length_sweep_points"]),
            },
        },
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
