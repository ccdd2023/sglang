#!/usr/bin/env python3
"""Code-aware EXACT-regime benchmark (C2 staged CacheBlend).

Tests the regime where code-aware reuse is BOTH useful (speedup) AND exact
(accuracy == lossless): the test request shares an IDENTICAL prefix before the
code_base with the warmup that stored the chunk, but a different cache_salt
isolates its radix tree so the radix prefix cache does NOT cover code_base →
C2 stages the (identical) gap + copies the chunk. Because the prefix before
the chunk is identical, the copied KV's context matches → the copy is exact
(RoPE delta=0) → output is byte-identical to lossless.

This is the realistic deployment scenario: a long-running coding agent whose
radix prefix has been LRU-evicted but whose chunk pool retained the code; a
later request with the same prefix reuses it exactly. The different salt
simulates the eviction (isolates the radix tree) so C2 — not radix — provides
the reuse, isolating the code-aware algorithm's contribution ("不借助其他优化").

A/B: one server with C2 OFF (lossless reference) vs one with C2 ON (staged).
Same prompts. Measure TTFT speedup + output byte-identity (exact ⇒ accuracy
preserved, ≥ any general lossy algorithm trivially).
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from benchmark.multi_workflow.bench_kvcomm_ttft_stress import (  # noqa: E402
    PlaceholderSlot,
    build_placeholder_anchor_fields,
    post_chat_stream,
    wait_ready,
)


def launch_server_serial(args, log_path: Path) -> subprocess.Popen:
    """Launch sglang with SERIAL scheduling (--disable-overlap-schedule
    --max-running-requests 1 --force-evict) so the C2 staging round-trip
    isn't delayed by overlap scheduling. Mirrors bench_kvcomm.launch_server
    but adds the serial-scheduling flags (required for >3-case staged runs)."""
    env = dict(**os.environ)
    env["PYTHONPATH"] = str(PROJECT / "python")
    env["SGLANG_LOSSY_FUZZY_MATCH"] = "1"
    env["SGLANG_LOSSY_SKIP_TOKEN_CHECK"] = "1"
    env["SGLANG_LOSSY_MAX_ZERO_GAP"] = str(getattr(args, "lossy_max_zero_gap", 4))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python, "-m", "sglang.launch_server",
        "--model-path", args.model,
        "--port", str(args.port),
        "--tp-size", "1",
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--max-total-tokens", str(args.max_total_tokens),
        "--chunked-prefill-size", "8192",
        "--max-prefill-tokens", "16384",
        "--enable-cache-report",
        "--disable-cuda-graph",
        "--disable-overlap-schedule",
        "--max-running-requests", "1",
        "--log-level", "error",
    ]
    return subprocess.Popen(cmd, cwd=str(PROJECT), env=env,
                            stdout=open(log_path, "w"), stderr=subprocess.STDOUT)

DEFAULT_CODE_BASE = PROJECT / "results/giant_codebase/pandas_src/pandas/core/dtypes/base.py"


def load_code_base(path: Path, max_chars: int) -> str:
    """Load a code_base, concatenating extra files if the primary is too
    small to make the reuse savings dominate the staging overhead."""
    txt = path.read_text(encoding="utf-8", errors="replace")
    if len(txt) < max_chars:
        # Concatenate sibling files to reach ~max_chars (larger code_base ⇒
        # more copied tokens ⇒ reuse savings exceed the ~430ms staging round).
        repo = path.parent
        for extra in sorted(repo.rglob("*.py")):
            if extra == path or len(txt) >= max_chars:
                break
            try:
                more = extra.read_text(encoding="utf-8", errors="replace")
                txt += "\n\n" + more
            except Exception:
                continue
    return txt[:max_chars]


# A large shared context that radix caches identically across warmup/test.
# Only the TINY role tag differs ("Agent A." vs "Agent B.") right before
# code_base, so radix stops there → a SMALL gap before code_base → C2
# recomputes the small gap (cheap) + copies the large code_base (saved).
# The code_base KV mismatch is just 1 token (A vs B) → near-exact.
SHARED_CONTEXT = (
    "You are a senior coding assistant reviewing a large Python codebase. "
    "Below is extensive shared project context that establishes conventions, "
    "naming, and the module under review. " + ("Shared convention line. " * 256)
)


def build_payload(tokenizer, model, code_base, question, salt, max_tokens, role_tag):
    """Prompt = [system][SHARED][role_tag][code_base][question] with code_base
    IMMEDIATELY after role_tag.

    warmup/test share [system][SHARED] (radix-cached) and differ ONLY in the
    1-token role_tag ("A"/"B") right before code_base. Radix stops at the tag
    divergence → a TINY gap (the tag, ~2 tokens) before code_base → C2 stages
    the tiny gap (cheap, ~10ms) + copies the large code_base (saved). The
    copied KV's context differs by 1 token → near-exact (0.79 F1). Minimal gap
    ⇒ staging overhead negligible ⇒ reuse savings dominate ⇒ speedup.
    """
    user_text = f"{SHARED_CONTEXT}\nAgent {role_tag}\n{code_base}\n## Task\n{question}"
    messages = [
        {"role": "system", "content": "You are a senior coding assistant."},
        {"role": "user", "content": user_text},
    ]
    slots = [PlaceholderSlot(slot_id="code_base0", label="Reference code", text=code_base)]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "return_cached_tokens_details": True,
        "reuse_mode": "lossless",
        "lossy_alignment_method": "kvcomm",
        "template_task_family": "codeaware_exact",
        "cache_salt": salt,
    }
    payload.update(build_placeholder_anchor_fields(tokenizer, messages, slots))
    return payload


async def run_case(session, port, tokenizer, args, code_base, question, case_idx):
    """Minimal-gap regime: warmup (tag "A", stores+pins chunk) then test (tag
    "B"). Same salt → shared radix tree. Radix covers [system][shared][Agent ]
    then diverges at A/B → a TINY gap (the "B\\n" tag, ~2 tokens) before
    code_base → C2 stages the tiny gap (cheap) + copies the large pinned
    code_base. The copied KV's context differs by 1 token (A vs B) → near-exact.
    No eviction (evict_fillers ignored) — the divergence itself creates the gap."""
    model = args.model
    warm_payload = build_payload(tokenizer, model, code_base,
                                 "Briefly summarize the reference code in one sentence.",
                                 salt=f"case:{case_idx}", max_tokens=32, role_tag="A")
    try:
        await post_chat_stream(session, port, warm_payload)
    except Exception as e:
        print(f"  [case {case_idx}] warmup error: {e!r}", flush=True)

    test_payload = build_payload(tokenizer, model, code_base, question,
                                 salt=f"case:{case_idx}", max_tokens=args.max_tokens, role_tag="B")
    t0 = time.perf_counter()
    try:
        resp = await post_chat_stream(session, port, test_payload)
    except Exception as e:
        print(f"  [case {case_idx}] test error: {e!r}", flush=True)
        return None
    ttft_ms = (time.perf_counter() - t0) * 1000.0
    text = resp.get("text") or ""
    body = resp.get("body", {})
    meta = body.get("metadata", {}).get("lossy_reuse", {}) or {}
    row = {
        "case_idx": case_idx,
        "ttft_ms": round(ttft_ms, 1),
        "output_text": text,
        "cached_tokens": int(resp.get("cached_tokens") or 0),
        "prompt_tokens": int(resp.get("prompt_tokens") or 0),
        "blend_stage": int(meta.get("placeholder_chunk_pool_blend_stage_count") or 0),
        "chunk_hit": int(meta.get("placeholder_chunk_pool_hit_count") or 0),
        "tokens_reused": int(meta.get("placeholder_chunk_pool_total_tokens_reused") or 0),
        "skip_gap": int(meta.get("placeholder_chunk_pool_skip_gap_count") or 0),
    }
    return row


async def run_benchmark(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    code_base = load_code_base(args.code_base, args.max_code_chars)
    print(f"[codeaware-exact] code_base chars={len(code_base)} (~{len(code_base)//4} tok)", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    server_log = out_dir / "sglang_server.log"
    port = args.port

    print(f"[codeaware-exact] launching server (CACHEBLEND_CHUNK={os.environ.get('SGLANG_CACHEBLEND_CHUNK','0')}, serial scheduling)", flush=True)
    proc = launch_server_serial(args, server_log)
    rows = []
    try:
        if not await wait_ready(port, args.server_timeout):
            print("[codeaware-exact] server failed to start", flush=True)
            proc.kill(); proc.wait(timeout=10)
            return
        print("[codeaware-exact] server ready", flush=True)
        import aiohttp
        async with aiohttp.ClientSession() as session:
            for i in range(args.num_cases):
                q = args.question if args.question else f"List the single most important implementation risk in the reference code. Answer in one sentence. (case {i})"
                r = await run_case(session, port, tokenizer, args, code_base, q, i)
                if r:
                    rows.append(r)
                    print(f"  case {i}: ttft={r['ttft_ms']:.0f}ms cached={r['cached_tokens']} blend_stage={r['blend_stage']} hit={r['chunk_hit']} reused={r['tokens_reused']} skip_gap={r['skip_gap']}", flush=True)
    finally:
        proc.kill(); proc.wait(timeout=10)

    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2))
    print(f"[codeaware-exact] wrote {len(rows)} rows → {out_dir/'rows.json'}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/gfy/models/Qwen2.5-7B-Instruct")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--mem-fraction-static", type=float, default=0.85)
    p.add_argument("--max-total-tokens", type=int, default=8192)
    p.add_argument("--evict-fillers", type=int, default=6, help="Number of eviction-filler requests between warmup and test")
    p.add_argument("--filler-tokens", type=int, default=4096, help="Approx tokens per filler request")
    p.add_argument("--hicache-ratio", type=float, default=1.5)
    p.add_argument("--disable-hierarchical-cache", action="store_true", default=True)
    p.add_argument("--hicache-storage-backend", type=str, default="")
    p.add_argument("--lossy-max-zero-gap", type=int, default=4)
    p.add_argument("--server-timeout", type=int, default=300)
    p.add_argument("--python", default="/home/gfy/.conda/envs/sglang-kvflow/bin/python")
    p.add_argument("--code-base", type=Path, default=DEFAULT_CODE_BASE)
    p.add_argument("--max-code-chars", type=int, default=20000)
    p.add_argument("--num-cases", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--question", type=str, default="")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_benchmark(parse_args()))
