#!/usr/bin/env python3
"""Test: Short gap (2-3 tokens) zero-fill impact on accuracy.

Uses small variant comment (# A1 variant = 3 tokens) which creates
a minimal gap between exact prefix and code block.
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


def launch(lossy=False):
    e = os.environ.copy()
    e["PYTHONPATH"] = str(ROOT / "python") + (":" + e.get("PYTHONPATH", "") if e.get("PYTHONPATH") else "")
    if lossy:
        e["SGLANG_LOSSY_FUZZY_MATCH"] = "1"
        e["SGLANG_LOSSY_SKIP_TOKEN_CHECK"] = "1"
    return subprocess.Popen(
        [PY, "-m", "sglang.launch_server", "--model-path", MODEL, "--port", str(PORT),
         "--tp-size", "1", "--mem-fraction-static", "0.85", "--max-total-tokens", "65536",
         "--chunked-prefill-size", "8192", "--max-prefill-tokens", "16384",
         "--radix-eviction-policy", "priority", "--enable-hierarchical-cache",
         "--hicache-ratio", "1.5", "--hicache-write-policy", "write_back",
         "--enable-cache-report", "--disable-cuda-graph", "--log-level", "error"],
        env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(ROOT),
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


def vary_code(code_text, agent_id):
    return f"# A{agent_id} variant\n{code_text}"


def compute_token_spans(full_text, code_text):
    full_ids = TOK.encode(full_text, add_special_tokens=False)
    code_ids = TOK.encode(code_text, add_special_tokens=False)
    for i in range(len(full_ids) - len(code_ids) + 1):
        if full_ids[i : i + len(code_ids)] == code_ids:
            return [{"start_token": i, "end_token": i + len(code_ids)}]
    return []


def build_payload(agent_id, code_text, reuse_mode, a1_text="", a2_text=""):
    code_variant = vary_code(code_text, agent_id)
    if agent_id == 1:
        instruction = "Analyze this codebase. Summarize in 2-3 sentences."
        msg = f"```python\n{code_variant}\n```\n\n{instruction}"
    elif agent_id == 2:
        instruction = f"Propose ONE improvement. Analysis: {a1_text}\n\nWrite code."
        msg = f"```python\n{code_variant}\n```\n\n{instruction}"
    else:
        instruction = f"Review. Analysis: {a1_text}\n\nFix: {a2_text}\n\nVerdict?"
        msg = f"```python\n{code_variant}\n```\n\n{instruction}"

    full_text = TOK.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        tokenize=False, add_generation_prompt=False,
    )
    token_spans = compute_token_spans(full_text, code_text)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "benchmark" / "multi_workflow"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "MAScoder" / "src"))
    from mascoder.code_anchor import build_code_anchor_payload
    a = build_code_anchor_payload(code_variant, language="python")

    return {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        "max_tokens": 128, "temperature": 0.0,
        "reuse_mode": reuse_mode,
        "lossy_alignment_method": "kvcomm",
        "code_anchor_signature": a.get("ast_anchor_signature", ""),
        "code_anchor_spans": a.get("code_anchor_spans", []),
        "code_anchor_token_spans": token_spans,
    }


BASE_URL = f"http://127.0.0.1:{PORT}"


async def req_stream(sess, payload):
    start = time.perf_counter()
    ttft = None
    text = ""
    cached = 0
    meta = {}
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
                    chunk_meta = chunk.get("metadata", {})
                    if chunk_meta.get("lossy_reuse"):
                        meta = chunk_meta["lossy_reuse"]
                except Exception:
                    pass
    total = (time.perf_counter() - start) * 1000
    return {"ttft_ms": round(ttft, 1), "total_ms": round(total, 1), "text": text, "cached": cached, "meta": meta}


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


async def main():
    print("Short Gap Test (# A{id} variant = ~3 tokens gap)")
    print(f"Code: {len(CODE.splitlines())} lines")
    print("=" * 60)

    # GT
    print("\n--- Ground Truth (no variant) ---")
    kill(); time.sleep(2)
    p = launch(lossy=False)
    if not wait_ready():
        print("Server fail"); p.terminate(); return

    async with aiohttp.ClientSession() as sess:
        # Use identical code for GT
        gt1 = await req_both(sess, build_payload(1, CODE, "lossless"))
        gt2 = await req_both(sess, build_payload(2, CODE, "lossless", a1_text=gt1["text"]))
        gt3 = await req_both(sess, build_payload(3, CODE, "lossless", a1_text=gt1["text"], a2_text=gt2["text"]))
    p.terminate(); time.sleep(3); kill()

    # Lossy with variant (short gap)
    print("\n--- Lossy (with variant comment, short gap) ---")
    kill(); time.sleep(2)
    p = launch(lossy=True)
    if not wait_ready():
        print("Server fail"); p.terminate(); return

    async with aiohttp.ClientSession() as sess:
        lr1 = await req_both(sess, build_payload(1, CODE, "lossy"))
        lr2 = await req_both(sess, build_payload(2, CODE, "lossy", a1_text=gt1["text"]))
        lr3 = await req_both(sess, build_payload(3, CODE, "lossy", a1_text=gt1["text"], a2_text=gt2["text"]))
    p.terminate(); time.sleep(3); kill()

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON (GT vs Lossy with variant)")
    print("=" * 60)
    for agent, gt_r, lr in [("A2", gt2, lr2), ("A3", gt3, lr3)]:
        print(f"\n{agent}:")
        print(f"  GT:    {gt_r['text'][:80]}...")
        print(f"  Lossy: {lr['text'][:80]}...")
        print(f"  BLEU:  {bleu(gt_r['text'], lr['text']):.3f}")
        print(f"  TTFT:  GT=N/A Lossy={lr['ttft_ms']}ms cache={lr['cached']}")
        m = lr['meta']
        print(f"  Meta:  anchor={m.get('lossy_anchor_match_used')} len={m.get('lossy_anchor_match_len')} gap={m.get('lossy_anchor_match_gap_len')} delta={m.get('lossy_anchor_rope_delta')}")

    # Check if outputs are identical
    a2_match = gt2["text"] == lr2["text"]
    a3_match = gt3["text"] == lr3["text"]
    print(f"\nExact Match: A2={a2_match} A3={a3_match}")
    if a2_match and a3_match:
        print("✅ Short gap (3 tokens) has NO accuracy impact!")
    else:
        print("⚠️  Short gap still affects accuracy - need segmented prefill")


if __name__ == "__main__":
    asyncio.run(main())
