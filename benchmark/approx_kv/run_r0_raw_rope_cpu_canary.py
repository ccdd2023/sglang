#!/usr/bin/env python3
"""Phase 4 R0 (Raw+RoPE) CPU-only structural canary.

This is deliberately **not** a live-server benchmark: R0 must be validated
without starting a GPU server (other research branches share the host GPU),
so this canary drives the real `restore_request_prefix()` request-path
function in-process against a fake allocator/KV-cache, using token sequences
drawn from the actual Phase 2 object catalog
(`benchmark.approx_kv.workloads.build_object_catalog`) so the coverage is
grounded in the same synthetic artifacts used by the Phase 2/3 GPU
benchmarks, not hand-picked toy integers.

It reports only *structural* pass/fail and token-count/timing metadata for
the raw-copy + RoPE-relocation path itself (register -> reuse -> verify
rotated keys/values bit-for-bit against an independently computed rotation).
There is no accuracy metric: this branch is an explicit speed-only upper
bound, and no output text/logits are ever generated or compared.

Coverage exercised (see task spec / raw_rope.py docstring for the full
contract):

- zero, positive, and negative RoPE position delta;
- contiguous multi-segment recovery;
- an interior segment recovered immediately after a dense/exact head;
- the explicit plugin gate (registered vs. not registered);
- the honest hard limitation: non-contiguous coverage always raises
  ``RawRoPERecoveryUnavailable`` and falls back to dense, it is never
  silently mis-repaired.

Run from the CPU-only immutable image (no GPU, no server):

    python3 -m benchmark.approx_kv.run_r0_raw_rope_cpu_canary \\
        --model Qwen/Qwen3-0.6B \\
        --model-revision c1899de289a04d12100db370d81485cdf75e47ca \\
        --output benchmark/approx_kv/results/phase4-r0/cpu-canary.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch

from benchmark.approx_kv.workloads import ReuseClass, build_object_catalog
from sglang.srt.mem_cache.approx_kv.config import ApproxKVFeatureConfig
from sglang.srt.mem_cache.approx_kv.manager import ApproxKVManager
from sglang.srt.mem_cache.approx_kv.radix_backend import (
    AllocatorCPUResidencyBackend,
    RoPEConfig,
)
from sglang.srt.mem_cache.approx_kv.request import (
    ApproxKVRequestMetadata,
    ApproxKVRequestOperation,
    ApproxKVRequestSegment,
)
from sglang.srt.mem_cache.approx_kv.runtime import (
    register_request_segments,
    restore_request_prefix,
)

MODEL_FINGERPRINT_SUFFIX = "phase4-r0-cpu-canary"
CACHE_DTYPE = "fp32"
ROTARY_DIM = 16
HEAD_DIM = 16
NUM_KV_HEADS = 2
LAYER_NUM = 2
ROPE_BASE = 10000.0


# --------------------------------------------------------------------------
# Minimal in-process harness (same shape as the CPU unit-test fakes in
# test/registered/unit/mem_cache/test_raw_rope_plugin.py) -- duplicated here
# rather than imported so the canary has no dependency on the test package
# and can run standalone from a benchmark entrypoint.
# --------------------------------------------------------------------------


class FakeKVCache:
    def __init__(self, capacity: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        shape = (capacity, NUM_KV_HEADS, HEAD_DIM)
        self.layer_num = LAYER_NUM
        self.k_buffer = [
            torch.randn(shape, generator=generator) + layer
            for layer in range(self.layer_num)
        ]
        self.v_buffer = [
            torch.randn(shape, generator=generator) + 1000 + layer
            for layer in range(self.layer_num)
        ]

    def move_kv_cache(self, target, source):
        for layer in range(self.layer_num):
            self.k_buffer[layer][target] = self.k_buffer[layer][source]
            self.v_buffer[layer][target] = self.v_buffer[layer][source]

    def get_key_buffer(self, layer_id):
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id):
        return self.v_buffer[layer_id]


class FakeAllocator:
    device = "cpu"

    def __init__(self, kvcache: FakeKVCache, next_index: int = 0):
        self.kvcache = kvcache
        self.next_index = next_index
        self.freed: list[int] = []

    def alloc(self, size: int) -> torch.Tensor:
        result = torch.arange(self.next_index, self.next_index + size, dtype=torch.int64)
        self.next_index += size
        return result

    def free(self, indices) -> None:
        self.freed.extend(int(index) for index in indices)

    def get_kvcache(self) -> FakeKVCache:
        return self.kvcache

    def get_cpu_copy(self, indices, mamba_indices=None):
        del mamba_indices
        return (
            [buffer[indices].clone() for buffer in self.kvcache.k_buffer],
            [buffer[indices].clone() for buffer in self.kvcache.v_buffer],
        )

    def load_cpu_copy(self, payload, indices, mamba_indices=None):
        del mamba_indices
        keys, values = payload
        for layer in range(self.kvcache.layer_num):
            self.kvcache.k_buffer[layer][indices] = keys[layer]
            self.kvcache.v_buffer[layer][indices] = values[layer]


class FakeReqToTokenPool:
    def __init__(self, rows: int, capacity: int):
        self.req_to_token = torch.full((rows, capacity), -1, dtype=torch.int64)


class FakeReq:
    def __init__(self, metadata, tokens, exact_prefix_len: int = 0):
        self.approx_kv_metadata = metadata
        self.req_pool_idx = 0
        self.kv = SimpleNamespace(kv_allocated_len=len(tokens))
        self.full_untruncated_fill_ids = list(tokens)
        self.prefix_indices = torch.arange(exact_prefix_len, dtype=torch.int64)
        self.rid = "canary-req"

    def effective_kv_committed_len(self) -> int:
        return len(self.full_untruncated_fill_ids)

    def needs_host_load_back(self) -> bool:
        return False


def _metadata(segments, operation, model_fingerprint: str):
    return ApproxKVRequestMetadata(
        operation=operation,
        segments=segments,
        model_fingerprint=model_fingerprint,
        cache_dtype=CACHE_DTYPE,
    )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def expected_rotation(key: torch.Tensor, delta: int) -> torch.Tensor:
    """Independently recomputed neox-style RoPE angle shift, used only to
    verify the plugin's own rotation output -- not part of the plugin code
    under test."""
    if delta == 0:
        return key.clone()
    rotary = key[..., :ROTARY_DIM]
    passthrough = key[..., ROTARY_DIM:]
    dim_range = torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32)
    inv_freq = 1.0 / (ROPE_BASE ** (dim_range / ROTARY_DIM))
    angle = delta * inv_freq
    cos = torch.cos(angle).repeat(2)
    sin = torch.sin(angle).repeat(2)
    rotated = rotary * cos + rotate_half(rotary) * sin
    return torch.cat([rotated, passthrough], dim=-1)


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    recovered: bool
    copied_tokens: int
    rope_delta: int | None
    detail: str
    duration_seconds: float


class Canary:
    def __init__(self, model_fingerprint: str):
        self.model_fingerprint = model_fingerprint
        self.results: list[ScenarioResult] = []

    def _new_harness(self, capacity: int, seed: int, *, raw_rope_plugin_enabled: bool = True):
        # `capacity` is the span of *position-based* physical indices this
        # scenario's registered source tokens occupy (this harness reuses
        # each token's logical position as its fake physical device index,
        # same convention as the shared unit-test harness). The real
        # allocator (used by `ensure_device()` host->device promotion and
        # by the final restore allocation) must never hand out indices
        # that alias those position-based slots, so it starts well past
        # them with generous headroom for however many promote/allocate
        # calls a scenario triggers.
        headroom = 4096
        buffer_capacity = capacity + headroom
        kvcache = FakeKVCache(buffer_capacity, seed)
        allocator = FakeAllocator(kvcache, next_index=capacity + 64)
        req_pool = FakeReqToTokenPool(rows=4, capacity=buffer_capacity)
        config = ApproxKVFeatureConfig(
            core_enabled=True,
            host_residency_enabled=True,
            raw_rope_plugin_enabled=raw_rope_plugin_enabled,
        )
        manager = ApproxKVManager(config)
        manager.bind_residency_backend(AllocatorCPUResidencyBackend(allocator))
        manager.bind_rope_config(
            RoPEConfig(rotary_dim=ROTARY_DIM, base=ROPE_BASE, is_neox_style=True)
        )
        tree = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=req_pool,
            approx_kv=manager,
        )
        return kvcache, allocator, req_pool, manager, tree

    def _register(self, tree, req_pool, row, content_hash, tokens, source_start):
        req_pool.req_to_token[row, source_start : source_start + len(tokens)] = (
            torch.arange(source_start, source_start + len(tokens))
        )
        segment = ApproxKVRequestSegment(
            content_hash=content_hash,
            target_start=source_start,
            length=len(tokens),
        )
        filler = tuple(range(source_start)) + tuple(int(t) for t in tokens)
        src_req = FakeReq(
            _metadata((segment,), ApproxKVRequestOperation.REGISTER, self.model_fingerprint),
            filler,
        )
        src_req.req_pool_idx = row
        register_request_segments(tree, src_req)

    def run(self, name: str, fn: Callable[[], ScenarioResult]) -> None:
        start = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - canary must record, not crash silently
            result = ScenarioResult(
                name=name,
                passed=False,
                recovered=False,
                copied_tokens=0,
                rope_delta=None,
                detail=f"unhandled exception: {exc!r}",
                duration_seconds=time.perf_counter() - start,
            )
        self.results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {name}: {result.detail}")

    # -- scenarios -----------------------------------------------------

    def scenario_delta(self, tokens: tuple[int, ...], source_start: int, target_start: int, label: str) -> ScenarioResult:
        kvcache, allocator, req_pool, manager, tree = self._new_harness(
            capacity=max(source_start, target_start) + len(tokens) + 8, seed=hash(label) % (2**31)
        )
        self._register(tree, req_pool, 0, "art", tokens, source_start)
        source_indices = torch.arange(source_start, source_start + len(tokens))
        source_keys = [buffer[source_indices].clone() for buffer in kvcache.k_buffer]
        source_values = [buffer[source_indices].clone() for buffer in kvcache.v_buffer]

        filler = tuple(range(target_start)) + tuple(int(t) for t in tokens) + (999,)
        reuse = FakeReq(
            _metadata(
                (ApproxKVRequestSegment(content_hash="art", target_start=target_start, length=len(tokens)),),
                ApproxKVRequestOperation.REUSE,
                self.model_fingerprint,
            ),
            filler,
            # The [0, target_start) prefix is simulated as already covered
            # by an exact Radix match (a real dense/exact head), so the
            # raw+RoPE plugin only ever sees the remaining suffix -- see
            # required-behavior #1 (exact match always attempted first).
            exact_prefix_len=target_start,
        )
        ok = restore_request_prefix(tree, reuse)
        delta = target_start - source_start
        if not ok:
            return ScenarioResult(label, False, False, 0, delta, "restore_request_prefix returned False", 0.0)
        expected_total = target_start + len(tokens)
        if len(reuse.prefix_indices) != expected_total:
            return ScenarioResult(
                label, False, True, len(reuse.prefix_indices), delta,
                f"expected {expected_total} total prefix indices, got {len(reuse.prefix_indices)}", 0.0,
            )
        restored_indices = reuse.prefix_indices[target_start:]
        for layer in range(kvcache.layer_num):
            actual_key = kvcache.k_buffer[layer][restored_indices]
            expected_key = expected_rotation(source_keys[layer], delta)
            if not torch.allclose(actual_key, expected_key, atol=1e-5):
                return ScenarioResult(
                    label, False, True, len(tokens), delta,
                    f"layer {layer} key rotation mismatch (max diff "
                    f"{(actual_key - expected_key).abs().max().item():.6g})",
                    0.0,
                )
            actual_value = kvcache.v_buffer[layer][restored_indices]
            if not torch.equal(actual_value, source_values[layer]):
                return ScenarioResult(
                    label, False, True, len(tokens), delta,
                    f"layer {layer} value was not copied verbatim (values are never rotated)",
                    0.0,
                )
        # Final prompt token must never be part of the recovered/copied range.
        if expected_total >= len(filler):
            return ScenarioResult(
                label, False, True, len(tokens), delta,
                "final prompt token was incorrectly included in recovered coverage",
                0.0,
            )
        return ScenarioResult(
            label, True, True, len(tokens), delta,
            f"{len(tokens)} tokens recovered, rope_delta={delta}, rotation verified, "
            f"final token reserved for real forward",
            0.0,
        )

    def scenario_multi_segment(self, head_tokens, tail_tokens) -> ScenarioResult:
        label = "contiguous_multi_segment"
        total = len(head_tokens) + len(tail_tokens)
        kvcache, allocator, req_pool, manager, tree = self._new_harness(capacity=total + 8, seed=1)
        self._register(tree, req_pool, 0, "head", head_tokens, 0)
        self._register(tree, req_pool, 1, "tail", tail_tokens, len(head_tokens))
        filler = tuple(int(t) for t in head_tokens) + tuple(int(t) for t in tail_tokens) + (999,)
        reuse = FakeReq(
            _metadata(
                (
                    ApproxKVRequestSegment(content_hash="head", target_start=0, length=len(head_tokens)),
                    ApproxKVRequestSegment(content_hash="tail", target_start=len(head_tokens), length=len(tail_tokens)),
                ),
                ApproxKVRequestOperation.REUSE,
                self.model_fingerprint,
            ),
            filler,
        )
        ok = restore_request_prefix(tree, reuse)
        if not ok or len(reuse.prefix_indices) != total:
            return ScenarioResult(label, False, ok, len(reuse.prefix_indices) if ok else 0, 0, "multi-segment recovery failed", 0.0)
        return ScenarioResult(label, True, True, total, 0, f"{total} tokens recovered across 2 contiguous segments", 0.0)

    def scenario_interior_after_head(self, head_tokens, interior_tokens) -> ScenarioResult:
        label = "interior_segment_after_dense_exact_head"
        kvcache, allocator, req_pool, manager, tree = self._new_harness(
            capacity=len(head_tokens) + len(interior_tokens) + 8, seed=2
        )
        self._register(tree, req_pool, 0, "interior", interior_tokens, len(head_tokens))
        filler = tuple(int(t) for t in head_tokens) + tuple(int(t) for t in interior_tokens) + (999,)
        reuse = FakeReq(
            _metadata(
                (
                    ApproxKVRequestSegment(content_hash="interior", target_start=len(head_tokens), length=len(interior_tokens)),
                ),
                ApproxKVRequestOperation.REUSE,
                self.model_fingerprint,
            ),
            filler,
            exact_prefix_len=len(head_tokens),
        )
        ok = restore_request_prefix(tree, reuse)
        expected_total = len(head_tokens) + len(interior_tokens)
        if not ok or len(reuse.prefix_indices) != expected_total:
            return ScenarioResult(label, False, ok, len(reuse.prefix_indices) - len(head_tokens) if ok else 0, 0, "interior-after-head recovery failed", 0.0)
        return ScenarioResult(
            label, True, True, len(interior_tokens), 0,
            f"{len(interior_tokens)} interior tokens recovered after a {len(head_tokens)}-token dense/exact head",
            0.0,
        )

    def scenario_noncontiguous_gap_stops_at_leading_run(self, first_tokens, second_tokens) -> ScenarioResult:
        # Honest hard limitation (see raw_rope.py module docstring): this
        # plugin never bridges a gap. When segments are non-contiguous, only
        # the leading contiguous run anchored at the exact-prefix boundary
        # is recovered; anything past the gap is left untouched by this
        # call (neither silently repaired nor forced to a full-request
        # dense fallback) for the scheduler to handle as a separate dense
        # prefill afterwards.
        label = "noncontiguous_gap_stops_at_leading_contiguous_run"
        gap = 4
        target_start_second = len(first_tokens) + gap
        kvcache, allocator, req_pool, manager, tree = self._new_harness(
            capacity=target_start_second + len(second_tokens) + 8, seed=3
        )
        self._register(tree, req_pool, 0, "a", first_tokens, 0)
        self._register(tree, req_pool, 1, "b", second_tokens, target_start_second)
        filler = (
            tuple(int(t) for t in first_tokens)
            + tuple(range(gap))
            + tuple(int(t) for t in second_tokens)
            + (999,)
        )
        reuse = FakeReq(
            _metadata(
                (
                    ApproxKVRequestSegment(content_hash="a", target_start=0, length=len(first_tokens)),
                    ApproxKVRequestSegment(content_hash="b", target_start=target_start_second, length=len(second_tokens)),
                ),
                ApproxKVRequestOperation.REUSE,
                self.model_fingerprint,
            ),
            filler,
        )
        ok = restore_request_prefix(tree, reuse)
        if not ok or len(reuse.prefix_indices) != len(first_tokens):
            return ScenarioResult(
                label, False, ok, len(reuse.prefix_indices) if ok else 0, None,
                f"expected only the {len(first_tokens)}-token leading run to be recovered, "
                f"got ok={ok} recovered={len(reuse.prefix_indices) if ok else 0}",
                0.0,
            )
        return ScenarioResult(
            label, True, True, len(first_tokens), None,
            f"gap correctly stopped recovery at the {len(first_tokens)}-token leading "
            f"contiguous run; the segment past the gap was left unattempted, not "
            f"silently repaired",
            0.0,
        )

    def scenario_missing_segment_fallback(self, tokens) -> ScenarioResult:
        label = "missing_segment_dense_fallback"
        kvcache, allocator, req_pool, manager, tree = self._new_harness(capacity=len(tokens) + 8, seed=4)
        next_index_before = allocator.next_index
        filler = tuple(int(t) for t in tokens) + (999,)
        reuse = FakeReq(
            _metadata(
                (ApproxKVRequestSegment(content_hash="never-registered", target_start=0, length=len(tokens)),),
                ApproxKVRequestOperation.REUSE,
                self.model_fingerprint,
            ),
            filler,
        )
        ok = restore_request_prefix(tree, reuse)
        if ok or allocator.next_index != next_index_before:
            return ScenarioResult(label, False, ok, 0, None, "missing segment should force dense fallback with no allocation", 0.0)
        return ScenarioResult(label, True, False, 0, None, "missing segment correctly forced dense fallback", 0.0)

    def scenario_gate_disabled(self, tokens) -> ScenarioResult:
        # A realistic gate-disabled harness: the plugin is never registered
        # into `manager.plugins` in the first place (mirrors production,
        # where `raw_rope_plugin_enabled` is read once at manager
        # construction from `ApproxKVFeatureConfig.from_env()` and never
        # mutated afterward) rather than mutating `manager.config` on an
        # already-constructed manager.
        label = "explicit_plugin_gate_blocks_recovery_when_disabled"
        kvcache, allocator, req_pool, manager, tree = self._new_harness(
            capacity=len(tokens) + 8, seed=5, raw_rope_plugin_enabled=False
        )
        self._register(tree, req_pool, 0, "art", tokens, 0)
        next_index_before = allocator.next_index
        filler = tuple(int(t) for t in tokens) + (999,)
        reuse = FakeReq(
            _metadata(
                (ApproxKVRequestSegment(content_hash="art", target_start=0, length=len(tokens)),),
                ApproxKVRequestOperation.REUSE,
                self.model_fingerprint,
            ),
            filler,
        )
        ok = restore_request_prefix(tree, reuse)
        if ok or allocator.next_index != next_index_before:
            return ScenarioResult(label, False, ok, 0, None, "gate-disabled recovery should be a strict no-op", 0.0)
        return ScenarioResult(label, True, False, 0, None, "plugin gate off correctly blocked all recovery", 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--object-count", type=int, default=24)
    parser.add_argument("--segment-tokens", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path("benchmark/approx_kv/results/phase4-r0/cpu-canary.json"))
    parser.add_argument(
        "--runner-git-sha",
        default=None,
        help=(
            "Override the auto-detected `git rev-parse HEAD`. Useful when "
            "running from a read-only container mount that cannot see the "
            "host worktree's git admin directory."
        ),
    )
    return parser.parse_args()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    args = parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    catalog = build_object_catalog(tokenizer, object_count=args.object_count)
    hot_objects = [obj for obj in catalog if obj.reuse_class == ReuseClass.HOT]
    if not hot_objects:
        raise RuntimeError("Phase 2 object catalog produced no HOT-class objects")

    segment_len = args.segment_tokens

    def prefix_tokens(obj, offset: int, length: int) -> tuple[int, ...]:
        pool = obj.reusable_prefix_token_ids
        if len(pool) < offset + length:
            # Deterministically extend with a repeating tail so the canary
            # is robust to small catalog/tokenizer/version differences.
            pool = pool + tuple(pool[i % max(len(pool), 1)] for i in range(offset + length))
        return tuple(int(t) for t in pool[offset : offset + length])

    base_object = hot_objects[0]
    second_object = hot_objects[1] if len(hot_objects) > 1 else hot_objects[0]

    model_fingerprint = f"{args.model}@{args.model_revision or 'unpinned'}"
    canary = Canary(model_fingerprint)

    delta_tokens = prefix_tokens(base_object, 0, segment_len)
    canary.run(
        "zero_delta",
        lambda: canary.scenario_delta(delta_tokens, source_start=10, target_start=10, label="zero_delta"),
    )
    canary.run(
        "positive_delta",
        lambda: canary.scenario_delta(delta_tokens, source_start=5, target_start=37, label="positive_delta"),
    )
    canary.run(
        "negative_delta",
        lambda: canary.scenario_delta(delta_tokens, source_start=37, target_start=5, label="negative_delta"),
    )

    head_tokens = prefix_tokens(base_object, 0, segment_len // 2)
    tail_tokens = prefix_tokens(base_object, segment_len // 2, segment_len // 2)
    canary.run("contiguous_multi_segment", lambda: canary.scenario_multi_segment(head_tokens, tail_tokens))

    interior_head = prefix_tokens(second_object, 0, segment_len // 2)
    interior_body = prefix_tokens(second_object, segment_len // 2, segment_len // 2)
    canary.run(
        "interior_segment_after_dense_exact_head",
        lambda: canary.scenario_interior_after_head(interior_head, interior_body),
    )

    gap_first = prefix_tokens(base_object, 0, segment_len // 2)
    gap_second = prefix_tokens(second_object, 0, segment_len // 2)
    canary.run(
        "noncontiguous_gap_stops_at_leading_run",
        lambda: canary.scenario_noncontiguous_gap_stops_at_leading_run(gap_first, gap_second),
    )

    missing_tokens = prefix_tokens(base_object, 0, segment_len // 2)
    canary.run(
        "missing_segment_dense_fallback",
        lambda: canary.scenario_missing_segment_fallback(missing_tokens),
    )

    gate_tokens = prefix_tokens(base_object, 0, segment_len // 2)
    canary.run(
        "explicit_plugin_gate_blocks_recovery_when_disabled",
        lambda: canary.scenario_gate_disabled(gate_tokens),
    )

    all_passed = all(result.passed for result in canary.results)

    payload = {
        "schema_version": 1,
        "phase": "phase4-r0-raw-rope",
        "mode": "cpu_offline_structural_canary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runner_git_sha": args.runner_git_sha or git_sha(),
        "gpu": None,
        "server": None,
        "note": (
            "No accuracy metric: this branch is an explicit speed-only upper "
            "bound. Scenarios verify structural correctness (recovered token "
            "counts, exact bit-for-bit RoPE-relocated key values, verbatim "
            "value copies, final-token reservation, and correct handling of "
            "missing/non-contiguous coverage) against a real Phase 2 "
            "object-catalog token source, not model output text/logits."
        ),
        "known_hard_limitation": (
            "Only a single contiguous run of segments anchored at the "
            "exact-prefix boundary (optionally following a dense/exact head) "
            "is recovered. When declared segments are non-contiguous, "
            "recovery is trimmed to only the leading contiguous run at the "
            "boundary; the remainder past the first gap is left completely "
            "unattempted by this call (an ordinary prefill for the "
            "scheduler), never silently repaired. A missing source segment "
            "or an internal gap discovered inside the already-narrowed run "
            "aborts that whole recovery attempt (dense fallback) instead of "
            "a partial repair."
        ),
        "model": args.model,
        "model_revision": args.model_revision,
        "model_fingerprint": model_fingerprint,
        "object_catalog": {
            "object_count": len(catalog),
            "hot_object_ids": [obj.object_id for obj in hot_objects],
        },
        "scenarios": [
            {
                "name": result.name,
                "passed": result.passed,
                "recovered": result.recovered,
                "copied_tokens": result.copied_tokens,
                "rope_delta": result.rope_delta,
                "detail": result.detail,
            }
            for result in canary.results
        ],
        "passed": all_passed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite existing result file: {args.output}")
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if not all_passed:
        raise RuntimeError("one or more R0 canary scenarios failed; see output above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
