#!/usr/bin/env python3
"""Realistic benchmark: code is IDENTICAL across agents, only instructions differ.

This simulates the real multi-agent workflow where:
- All agents process the SAME codebase
- Only their instructions/tasks differ
- Full-Reuse should work perfectly (exact prefix extends through code block)
"""
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent / "benchmark" / "multi_workflow"))
from large_codebase import get_full_codebase

PORT = 30000
MODEL = "/home/gfy/models/Qwen2.5-3B-Instruct"
ROOT = Path(__file__).resolve().parent
PY = "/home/gfy/.conda/envs/sglang-kvflow/bin/python"
OUT = ROOT / "results" / "ma_ttft"
OUT.mkdir(parents=True, exist_ok=True)
TOK = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
SYSTEM = "You are a senior software engineer. Be concise and precise."
CODE = get_full_codebase()


def kill():
    try:
        with open("/proc/net/tcp") as f:
            for r in f.readlines()[1:]:
                p = r.split()
                if p[1].endswith(f":{PORT:04X}") and p[3] == "0A":
                    ino = p[9]
                    break
            else:
                return
    except Exception:
        return
    for pid in sorted(filter(str.isdigit, os.listdir("/proc")), key=int):
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                if os.readlink(f"/proc/{pid}/fd/{fd}") == f"socket:[{ino}]":
                    os.kill(int(pid), signal.SIGTERM)
                    time.sleep(2)
                    return
        except Exception:
            pass


def launch():
    e = os.environ.copy()
    e["PYTHONPATH"] = str(ROOT / "python") + (":" + e.get("PYTHONPATH", "") if e.get("PYTHONPATH") else "")
    return subprocess.Popen(
        [PY, "-m", "sglang.launch_server", "--model-path", MODEL, "--port", str(PORT),
         "--tp-size", "1", "--mem-fraction-static", "0.85", "--max-total-tokens", "65536",
         "--chunked-prefill-size", "8192", "--max-prefill-tokens", "16384",
         "--radix-eviction-policy", "priority", "--enable-hierarchical-cache",
         "--hicache-ratio", "1.5", "--hicache-write-policy", "write_back",
         "--enable-cache-report", "--disable-cuda-graph", "--log-level", "error"],
        env=e, stdout=open(str(OUT / "sglang_realistic.log"), "a"), stderr=subprocess.STDOUT, cwd=str(ROOT),
    )


def wait_ready(t=150):
    import urllib.request
    d = time.monotonic() + t
    while time.monotonic() < d:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health_generate", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(5)
    return False


def build_payload(agent_id, code_text, a1_text="", a2_text=""):
    """Build payload with IDENTICAL code block across agents."""
    instructions = {
        1: "Analyze this codebase. Identify the overall architecture, key data structures, and algorithms used. Summarize in 2-3 sentences.",
        2: f"Propose ONE concrete improvement to this codebase. Analysis: {a1_text}\n\nWrite the improved code snippet.",
        3: f"Review the proposed improvement. Analysis: {a1_text}\n\nFix: {a2_text}\n\nIs it correct and efficient? One-sentence verdict.",
    }
    msg = f"```python\n{code_text}\n```\n\n{instructions[agent_id]}"
    return {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        "max_tokens": 256, "temperature": 0.0,
    }


BASE_URL = f"http://127.0.0.1:{PORT}"


async def req_stream(sess, payload):
    start = time.perf_counter()
    ttft = None
    text = ""
    cached = 0
    async with sess.post(
        f"{BASE_URL}/v1/chat/completions",
        json={**payload, "stream": True, "stream_options": {"include_usage": True}},
        timeout=aiohttp.ClientTimeout(total=300),
    ) as resp:
        async for line in resp.content:
            line = line.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                try:
                    chunk = json.loads(line[6:])
                    if ttft is None:
                        ttft = (time.perf_counter() - start) * 1000
                    choices = chunk.get("choices", [])
                    if choices:
                        text += choices[0].get("delta", {}).get("content", "")
                    usage = chunk.get("usage")
                    if usage and usage.get("prompt_tokens_details"):
                        cached = usage["prompt_tokens_details"].get("cached_tokens", 0)
                except Exception:
                    pass
    total = (time.perf_counter() - start) * 1000
    return {"ttft_ms": round(ttft, 1), "total_ms": round(total, 1), "text": text, "cached": cached}


async def req_nonstream(sess, payload):
    async with sess.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
        body = await resp.json()
    text = body["choices"][0]["message"]["content"] if body.get("choices") else ""
    return {"text": text}


async def req_both(sess, payload):
    stream_res = await req_stream(sess, payload)
    ns_res = await req_nonstream(sess, payload)
    return {**stream_res, "text": ns_res["text"]}


def bleu(ref, hyp):
    try:
        from nltk.translate.bleu_score import sentence_bleu
        return float(sentence_bleu([ref.split()], hyp.split(), weights=(0.25, 0.25, 0.25, 0.25)))
    except Exception:
        rs, hs = set(ref.split()), set(hyp.split())
        return len(rs & hs) / len(rs) if rs else (1.0 if not hyp else 0.0)


async def run_case(sess, code, max_tokens):
    r1 = await req_both(sess, build_payload(1, code))
    r2 = await req_both(sess, build_payload(2, code, a1_text=r1["text"]))
    r3 = await req_both(sess, build_payload(3, code, a1_text=r1["text"], a2_text=r2["text"]))
    return {"a1": r1, "a2": r2, "a3": r3}


async def main():
    print(f"Realistic Benchmark (Identical Code)")
    print(f"Code: {len(CODE.splitlines())} lines, {len(CODE)} chars")
    print("=" * 60)

    max_tokens = 256

    # Phase 1: GT
    print("\n--- Phase 1: Ground Truth ---")
    kill(); time.sleep(2)
    p = launch()
    if not wait_ready():
        print("Server fail"); p.terminate(); return
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
        gt = await run_case(sess, CODE, max_tokens)
    p.terminate(); time.sleep(3); kill()
    for a in ["a1", "a2", "a3"]:
        print(f"  {a.upper()}: {gt[a]['text'][:80]}...")

    # Phase 2: No-Reuse
    print("\n--- Phase 2: No-Reuse ---")
    kill(); time.sleep(2)
    p = launch()
    if not wait_ready():
        print("Server fail"); p.terminate(); return
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
        no_reuse = await run_case(sess, CODE, max_tokens)
    p.terminate(); time.sleep(3); kill()
    for a in ["a2", "a3"]:
        print(f"  {a.upper()}: TTFT={no_reuse[a]['ttft_ms']}ms cache={no_reuse[a]['cached']}")

    # Phase 3: Full-Reuse (same server, warm start)
    print("\n--- Phase 3: Full-Reuse ---")
    kill(); time.sleep(2)
    p = launch()
    if not wait_ready():
        print("Server fail"); p.terminate(); return
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
        # A1 populates cache
        fr1 = await req_both(sess, build_payload(1, CODE))
        # A2/A3 reuse A1's KV (exact prefix through code block)
        fr2 = await req_both(sess, build_payload(2, CODE, a1_text=gt["a1"]["text"]))
        fr3 = await req_both(sess, build_payload(3, CODE, a1_text=gt["a1"]["text"], a2_text=gt["a2"]["text"]))
    p.terminate(); time.sleep(3); kill()
    full_reuse = {"a1": fr1, "a2": fr2, "a3": fr3}
    for a in ["a2", "a3"]:
        print(f"  {a.upper()}: TTFT={full_reuse[a]['ttft_ms']}ms cache={full_reuse[a]['cached']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for agent in ["a2", "a3"]:
        nr = no_reuse[agent]
        fr = full_reuse[agent]
        speedup = (nr["ttft_ms"] / fr["ttft_ms"] - 1) * 100 if fr["ttft_ms"] > 0 else 0
        print(f"\n{agent.upper()}:")
        print(f"  No-Reuse:    TTFT={nr['ttft_ms']}ms  cache={nr['cached']}")
        print(f"  Full-Reuse:  TTFT={fr['ttft_ms']}ms  cache={fr['cached']}  ({speedup:+.1f}%)")

    def b(ref, hyp):
        return bleu(ref["text"], hyp["text"])

    print(f"\nBLEU (vs GT):")
    for agent in ["a2", "a3"]:
        print(f"  {agent.upper()} No={b(gt[agent], no_reuse[agent]):.3f} Full={b(gt[agent], full_reuse[agent]):.3f}")

    all_results = {
        "gt": {k: {"text": v["text"]} for k, v in gt.items()},
        "no_reuse": no_reuse, "full_reuse": full_reuse,
    }
    (OUT / "realistic_results.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nSaved to {OUT}/realistic_results.json")


if __name__ == "__main__":
    asyncio.run(main())
