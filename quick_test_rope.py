#!/usr/bin/env python3
"""Quick test: verify active RoPE rotation works (baseline regression)."""
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "MAScoder" / "src"))
from bench_multiagent_large import LARGE_CODE
from mascoder.code_anchor import build_code_anchor_payload

PORT = 30000
MODEL = "/home/gfy/models/Qwen2.5-3B-Instruct"
ROOT = Path(__file__).resolve().parent
PY = "/home/gfy/.conda/envs/sglang-kvflow/bin/python"
TOK = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
SYSTEM = "You are a senior software engineer. Be concise and precise."


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


def launch(lossy=False):
    e = os.environ.copy()
    e["PYTHONPATH"] = str(ROOT / "python") + (":" + e.get("PYTHONPATH", "") if e.get("PYTHONPATH") else "")
    if lossy:
        e["SGLANG_LOSSY_FUZZY_MATCH"] = "1"
        e["SGLANG_LOSSY_SKIP_TOKEN_CHECK"] = "1"
    return subprocess.Popen(
        [
            PY, "-m", "sglang.launch_server",
            "--model-path", MODEL,
            "--port", str(PORT),
            "--tp-size", "1",
            "--mem-fraction-static", "0.85",
            "--max-total-tokens", "65536",
            "--chunked-prefill-size", "8192",
            "--max-prefill-tokens", "16384",
            "--radix-eviction-policy", "priority",
            "--enable-hierarchical-cache",
            "--hicache-ratio", "1.5",
            "--hicache-write-policy", "write_back",
            "--enable-cache-report",
            "--disable-cuda-graph",
            "--log-level", "error",
        ],
        env=e,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(ROOT),
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


def compute_token_spans(full_text, code_text):
    full_ids = TOK.encode(full_text, add_special_tokens=False)
    code_ids = TOK.encode(code_text, add_special_tokens=False)
    for i in range(len(full_ids) - len(code_ids) + 1):
        if full_ids[i : i + len(code_ids)] == code_ids:
            return [{"start_token": i, "end_token": i + len(code_ids)}]
    return []


def vary_code(code_text, agent_id):
    return f"# A{agent_id} variant\n" + code_text


def build_payload(agent_id, code_text, reuse_mode):
    code_variant = vary_code(code_text, agent_id)
    a = build_code_anchor_payload(code_variant, language="python")
    tasks = {1: "A\nAnalyze this code. Identify purpose, design patterns, and potential bugs. Keep under 100 words.",
             2: "B\nImplement a fix based on this analysis.\n\nPropose ONE concrete fix. Write the changed code.",
             3: "C\nReview this implementation.\n\nIs the fix correct? One-sentence verdict."}
    msg = f"```python\n{code_variant}\n```\n\nTask: {tasks[agent_id]}"
    full_text = TOK.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        tokenize=False, add_generation_prompt=False,
    )
    token_spans = compute_token_spans(full_text, code_text)
    return {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        "max_tokens": 256,
        "temperature": 0.0,
        "code_anchor_signature": a.get("ast_anchor_signature", ""),
        "code_anchor_spans": a.get("code_anchor_spans", []),
        "code_anchor_token_spans": token_spans,
        "reuse_mode": reuse_mode,
        "lossy_alignment_method": "kvcomm",
    }


async def req_stream(sess, payload):
    start = time.perf_counter()
    ttft = None
    text = ""
    cached = 0
    meta = {}
    async with sess.post(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
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
                    chunk_meta = chunk.get("metadata", {})
                    if chunk_meta.get("lossy_reuse"):
                        meta = chunk_meta["lossy_reuse"]
                except Exception:
                    pass
    total = (time.perf_counter() - start) * 1000
    return {"ttft_ms": round(ttft, 1), "total_ms": round(total, 1), "text": text, "cached": cached, "meta": meta}


async def main():
    code = LARGE_CODE["avl_tree_insert"]
    print(f"Code lines: {len(code.splitlines())}")

    # Test 1: A1 (lossless, baseline)
    print("\n=== Test 1: A1 (lossless) ===")
    kill()
    time.sleep(2)
    p = launch(lossy=False)
    if not wait_ready():
        print("Server failed to start")
        p.terminate()
        return

    async with aiohttp.ClientSession() as sess:
        r1 = await req_stream(sess, build_payload(1, code, "lossless"))
        print(f"  TTFT={r1['ttft_ms']}ms total={r1['total_ms']}ms cached={r1['cached']} text={r1['text'][:60]}...")

    p.terminate()
    time.sleep(3)
    kill()

    # Test 2: A2 lossy with A1 output
    print("\n=== Test 2: A2 (lossy, should reuse A1 code KV) ===")
    p = launch(lossy=True)
    if not wait_ready():
        print("Server failed to start")
        p.terminate()
        return

    async with aiohttp.ClientSession() as sess:
        # First send A1 to populate cache
        r1 = await req_stream(sess, build_payload(1, code, "lossless"))
        print(f"  A1: TTFT={r1['ttft_ms']}ms cached={r1['cached']} anchor={r1['meta'].get('lossy_anchor_match_used')}")

        # Then send A2 lossy
        r2 = await req_stream(sess, build_payload(2, code, "lossy"))
        print(f"  A2: TTFT={r2['ttft_ms']}ms cached={r2['cached']} anchor={r2['meta'].get('lossy_anchor_match_used')}"
              f" len={r2['meta'].get('lossy_anchor_match_len')} gap={r2['meta'].get('lossy_anchor_match_gap_len')}"
              f" delta={r2['meta'].get('lossy_anchor_rope_delta')}"
              f" sig={r2['meta'].get('lossy_anchor_match_signature', '')[:20]}...")

        # Then send A3 lossy
        r3 = await req_stream(sess, build_payload(3, code, "lossy"))
        print(f"  A3: TTFT={r3['ttft_ms']}ms cached={r3['cached']} anchor={r3['meta'].get('lossy_anchor_match_used')}"
              f" len={r3['meta'].get('lossy_anchor_match_len')} gap={r3['meta'].get('lossy_anchor_match_gap_len')}"
              f" delta={r3['meta'].get('lossy_anchor_rope_delta')}")

    p.terminate()
    time.sleep(3)
    kill()

    print("\n=== Summary ===")
    print(f"A2 anchor match: {r2['meta'].get('lossy_anchor_match_used')} (len={r2['meta'].get('lossy_anchor_match_len')})")
    print(f"A3 anchor match: {r3['meta'].get('lossy_anchor_match_used')} (len={r3['meta'].get('lossy_anchor_match_len')})")
    print(f"A2 rope delta:   {r2['meta'].get('lossy_anchor_rope_delta')}")
    print(f"A3 rope delta:   {r3['meta'].get('lossy_anchor_rope_delta')}")
    print("\nBaseline regression PASSED if anchor=True and outputs are coherent.")


if __name__ == "__main__":
    asyncio.run(main())
