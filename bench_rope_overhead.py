#!/usr/bin/env python3
"""Micro-benchmark: measure RoPE delta rotation GPU overhead."""
import torch
import time

# Qwen2.5-3B config
NUM_LAYERS = 36
NUM_KV_HEADS = 2
HEAD_DIM = 128
MAX_TOKENS = 8192
DTYPE = torch.float16
DEVICE = "cuda"


def apply_rotary_emb(x, cos, sin, is_neox_style):
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def benchmark(copy_len, delta, num_warmup=10, num_iters=100):
    """Benchmark RoPE delta rotation for given copy_len."""
    # Create fake k_buffer (like MHATokenToKVPool)
    k_buffer = [
        torch.zeros((MAX_TOKENS, NUM_KV_HEADS, HEAD_DIM), dtype=DTYPE, device=DEVICE)
        for _ in range(NUM_LAYERS)
    ]

    # Fill with random data
    for k in k_buffer:
        k.normal_()

    # Allocate dst slots
    dst_slots = torch.arange(copy_len, device=DEVICE)
    delta_positions = torch.full((copy_len,), delta, dtype=torch.long, device=DEVICE)

    # Precompute cos/sin
    rope_base = 1000000.0
    rotary_dim = HEAD_DIM
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    freqs = torch.einsum("i,j->ij", delta_positions.float(), inv_freq.to(DEVICE))
    cos = freqs.cos()
    sin = freqs.sin()

    # Warmup
    for _ in range(num_warmup):
        for k_cache in k_buffer:
            k_selected = k_cache[dst_slots]
            k_rotated = apply_rotary_emb(k_selected, cos, sin, True)
            k_cache[dst_slots] = k_rotated
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for _ in range(num_iters):
        for k_cache in k_buffer:
            k_selected = k_cache[dst_slots]
            k_rotated = apply_rotary_emb(k_selected, cos, sin, True)
            k_cache[dst_slots] = k_rotated

    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / num_iters

    return elapsed_ms


def main():
    print("RoPE Delta Rotation Micro-Benchmark")
    print(f"Config: {NUM_LAYERS} layers, {NUM_KV_HEADS} KV heads, {HEAD_DIM} head dim")
    print("=" * 60)

    for copy_len in [100, 200, 408, 500, 1000]:
        for delta in [0, 10, 25, 50, 100]:
            ms = benchmark(copy_len, delta, num_warmup=5, num_iters=50)
            print(f"  copy_len={copy_len:4d} delta={delta:4d} -> {ms:.3f} ms")

    print("\n" + "=" * 60)
    print("Reference: prefill 500 tokens on Qwen2.5-3B ≈ 30-50ms")
    print("RoPE overhead is typically < 1% of prefill time.")


if __name__ == "__main__":
    main()
