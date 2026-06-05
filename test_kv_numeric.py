#!/usr/bin/env python3
"""Test: Same prompt with/without KV reuse to measure numeric error."""
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


def build_payload(code_text):
    msg = f"```python\n{code_text}\n```\n\nAnalyze this code."
    a = build_code_anchor_payload(code_text, language="python")
    return {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        "max_tokens": 128,
        "temperature": 0.0,
        "code_anchor_signature": a.get("ast_anchor_signature", ""),
        "code_anchor_spans": a.get("code_anchor_spans", []),
        "reuse_mode": "lossy",
        "lossy_alignment_method": "kvcomm",
    }


async def req_stream(sess, payload):
    text = ""
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
                    choices = chunk.get("choices", [])
                    if choices:
                        text += choices[0].get("delta", {}).get("content", "")
                except Exception:
                    pass
    return text


async def main():
    print("Test: Identical prompt, KV reuse vs no reuse")
    code = LARGE_CODE["avl_tree_insert"]
    payload = build_payload(code)

    # Run 1: Cold start (no cache)
    print("\n--- Run 1: Cold start ---")
    kill()
    time.sleep(2)
    p = launch(lossy=True)
    if not wait_ready():
        print("Server failed")
        p.terminate()
        return

    async with aiohttp.ClientSession() as sess:
        text1 = await req_stream(sess, payload)
    print(f"  Output: {text1[:100]}...")
    p.terminate()
    time.sleep(3)
    kill()

    # Run 2: Warm start (should reuse KV)
    print("\n--- Run 2: Warm start (reuse) ---")
    kill()
    time.sleep(2)
    p = launch(lossy=True)
    if not wait_ready():
        print("Server failed")
        p.terminate()
        return

    async with aiohttp.ClientSession() as sess:
        text2 = await req_stream(sess, payload)
    print(f"  Output: {text2[:100]}...")
    p.terminate()
    time.sleep(3)
    kill()

    print("\n--- Comparison ---")
    print(f"  Match: {text1 == text2}")
    print(f"  Text1: {text1[:120]}")
    print(f"  Text2: {text2[:120]}")

    if text1 == text2:
        print("\n✅ PASSED: Identical output with KV reuse")
    else:
        print("\n⚠️  DIFFERENT: KV reuse introduces output variation")
        # Show first diff
        for i, (c1, c2) in enumerate(zip(text1, text2)):
            if c1 != c2:
                print(f"  First diff at char {i}: '{c1}' vs '{c2}'")
                print(f"  Context: ...{text1[max(0,i-20):i+20]}...")
                break


if __name__ == "__main__":
    asyncio.run(main())
