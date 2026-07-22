#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from sglang.srt.mem_cache.approx_kv.radix_backend import (
    AnchorDeltaRef,
    DeviceKVRef,
    RadixKVTransferBackend,
    RoPEConfig,
)


class FakeKVCache:
    def __init__(
        self,
        *,
        layers: int,
        capacity: int,
        kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.layer_num = layers
        shape = (capacity, kv_heads, head_dim)
        self.keys = [
            torch.randn(shape, dtype=dtype, device=device) for _ in range(layers)
        ]
        self.values = [
            torch.randn(shape, dtype=dtype, device=device) for _ in range(layers)
        ]

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.keys[layer_id]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.values[layer_id]


@dataclass
class FakeAllocator:
    cache: FakeKVCache

    def get_kvcache(self) -> FakeKVCache:
        return self.cache


def measure_ms(
    operation,
    *,
    warmup: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": ordered[round((len(ordered) - 1) * 0.95)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--tokens", type=int, default=3048)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rotary-dim", type=int, default=128)
    parser.add_argument("--rope-delta", type=int, default=256)
    parser.add_argument("--epic-k", type=int, default=16)
    parser.add_argument("--selective-spans", type=int, default=8)
    parser.add_argument("--anchors", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--dense-ttft-ms", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tokens <= args.epic_k:
        raise ValueError("tokens must exceed epic-k")
    device = torch.device("cuda")
    dtype = torch.float16
    capacity = args.tokens * 2 + 32
    cache = FakeKVCache(
        layers=args.layers,
        capacity=capacity,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
    )
    allocator = FakeAllocator(cache)
    source = torch.arange(args.tokens, device=device, dtype=torch.int64)
    target = torch.arange(
        args.tokens,
        args.tokens * 2,
        device=device,
        dtype=torch.int64,
    )
    backend = RadixKVTransferBackend(
        allocator=allocator,
        target_indices=lambda start, length: target[start : start + length],
        dense_prefill=lambda start, length, reason: None,
        rope=RoPEConfig(
            rotary_dim=args.rotary_dim,
            base=10000.0,
            is_neox_style=True,
        ),
    )
    source_ref = DeviceKVRef(source)

    raw = measure_ms(
        lambda: backend.copy_and_rotate(
            source_ref=source_ref,
            source_offset=0,
            target_start=0,
            length=args.tokens,
            rope_delta=args.rope_delta,
        ),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    epic = measure_ms(
        lambda: backend.copy_and_rotate(
            source_ref=source_ref,
            source_offset=args.epic_k,
            target_start=args.epic_k,
            length=args.tokens - args.epic_k,
            rope_delta=args.rope_delta,
        ),
        warmup=args.warmup,
        iterations=args.iterations,
    )

    span_length = args.tokens // args.selective_spans

    def selective_copy() -> None:
        for span_index in range(args.selective_spans):
            start = span_index * span_length
            end = (
                args.tokens
                if span_index == args.selective_spans - 1
                else start + span_length
            )
            backend.copy_and_rotate(
                source_ref=source_ref,
                source_offset=start,
                target_start=start,
                length=end - start,
                rope_delta=args.rope_delta,
            )

    selective = measure_ms(
        selective_copy,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    delta_shape = (
        args.tokens,
        args.kv_heads,
        args.head_dim,
    )
    anchor_refs = tuple(
        AnchorDeltaRef(
            key_deltas=tuple(
                torch.randn(delta_shape, dtype=dtype, device=device) * 0.01
                for _ in range(args.layers)
            ),
            value_deltas=tuple(
                torch.randn(delta_shape, dtype=dtype, device=device) * 0.01
                for _ in range(args.layers)
            ),
        )
        for _ in range(args.anchors)
    )
    weights = tuple(1.0 / args.anchors for _ in range(args.anchors))
    anchor = measure_ms(
        lambda: backend.reconstruct_and_rotate(
            base_ref=source_ref,
            anchor_refs=anchor_refs,
            weights=weights,
            source_offset=0,
            target_start=0,
            length=args.tokens,
            rope_delta=args.rope_delta,
        ),
        warmup=args.warmup,
        iterations=args.iterations,
    )

    results = {
        "config": vars(args) | {"output": str(args.output)},
        "raw_rope": summarize(raw),
        "epic_copy_only": summarize(epic),
        "selective_copy_only": summarize(selective),
        "kvcomm_anchor": summarize(anchor),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    if args.dense_ttft_ms > 0:
        for key in (
            "raw_rope",
            "epic_copy_only",
            "selective_copy_only",
            "kvcomm_anchor",
        ):
            results[key]["speedup_vs_dense"] = (
                args.dense_ttft_ms / results[key]["p50_ms"]
            )
    args.output.write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(results, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
