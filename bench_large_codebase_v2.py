#!/usr/bin/env python3
"""Benchmark for LARGE codebase with FIXED context to measure pure KV reuse accuracy.

Key improvement: A2/A3 prompts use the SAME ground-truth context across all modes,
so BLEU differences come ONLY from KV reuse, not from prompt variations.
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "MAScoder" / "src"))
from large_codebase import get_full_codebase
from mascoder.code_anchor import build_code_anchor_payload

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


def launch(lossy=False):
    e = os.environ.copy()
    e["PYTHONPATH"] = str(ROOT / "python") + (
        ":" + e.get("PYTHONPATH", "") if e.get("PYTHONPATH") else ""
    )
    if lossy:
        e["SGLANG_LOSSY_FUZZY_MATCH"] = "1"
        e["SGLANG_LOSSY_SKIP_TOKEN_CHECK"] = "1"
    return subprocess.Popen(
        [
            PY, "-m", "sglang.launch_server", "--model-path", MODEL, "--port", str(PORT),
            "--tp-size", "1", "--mem-fraction-static", "0.85", "--max-total-tokens", "65536",
            "--chunked-prefill-size", "8192", "--max-prefill-tokens", "16384",
            "--radix-eviction-policy", "priority", "--enable-hierarchical-cache",
            "--hicache-ratio", "1.5", "--hicache-write-policy", "write_back",
            "--enable-cache-report", "--disable-cuda-graph", "--log-level", "error",
        ],
        env=e,
        stdout=open(str(OUT / "sglang_large_v2.log"), "a"),
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
    )


def wait_ready(t=150):
    import urllib.request
    d = time.monotonic() + t
    while time.monotonic() < d:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/health_generate", timeout=5
            ) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(5)
    return False


def compute_token_spans(full_text, code_text):
    """Find token spans of raw code_text (without variant comment) inside full_text."""
    code_idx = full_text.find(code_text)
    if code_idx == -1:
        return []
    prefix = full_text[:code_idx]
    prefix_ids = TOK.encode(prefix, add_special_tokens=False)
    code_ids = TOK.encode(code_text, add_special_tokens=False)
    start_token = len(prefix_ids)
    end_token = start_token + len(code_ids)
    return [{"start_token": start_token, "end_token": end_token}]


def vary_code(code_text, agent_id):
    """Add variant comment inside code block to break exact match while preserving AST."""
    return f"# A{agent_id} variant\n{code_text}"


def build_payload(agent_id, code_text, reuse_mode, a1_text="", a2_text=""):
    code_variant = vary_code(code_text, agent_id)
    if agent_id == 1:
        instruction = "Analyze this codebase. Identify the overall architecture, key data structures, and algorithms used. Summarize in 2-3 sentences."
        msg = f"```python\n{code_variant}\n```\n\n{instruction}"
    elif agent_id == 2:
        instruction = f"Propose ONE concrete improvement to this codebase. Analysis: {a1_text}\n\nWrite the improved code snippet."
        msg = f"```python\n{code_variant}\n```\n\n{instruction}"
    else:
        instruction = f"Review the proposed improvement. Analysis: {a1_text}\n\nFix: {a2_text}\n\nIs it correct and efficient? One-sentence verdict."
        msg = f"```python\n{code_variant}\n```\n\n{instruction}"

    full_text = TOK.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        tokenize=False,
        add_generation_prompt=False,
    )
    token_spans = compute_token_spans(full_text, code_text)
    print(
        f"  [Agent {agent_id}] code_pos={token_spans[0]['start_token'] if token_spans else 'N/A'}, "
        f"total_tokens={len(TOK.encode(full_text, add_special_tokens=False))}"
    )

    a = build_code_anchor_payload(code_text, language="python")
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
    return {
        "ttft_ms": round(ttft, 1),
        "total_ms": round(total, 1),
        "text": text,
        "cached": cached,
        "meta": meta,
    }


async def req_nonstream(sess, payload):
    async with sess.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=300),
    ) as resp:
        body = await resp.json()
    text = body["choices"][0]["message"]["content"] if body.get("choices") else ""
    cached = body.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    return {"text": text, "cached": cached}


async def req_both(sess, payload):
    stream_res = await req_stream(sess, payload)
    ns_res = await req_nonstream(sess, payload)
    meta = stream_res.get("meta", {}) or {}
    return {
        "ttft_ms": stream_res["ttft_ms"],
        "total_ms": stream_res["total_ms"],
        "text": ns_res["text"],
        "cached": stream_res["cached"],
        "meta": meta,
    }


def bleu(ref, hyp):
    try:
        from nltk.translate.bleu_score import sentence_bleu
        return float(
            sentence_bleu([ref.split()], hyp.split(), weights=(0.25, 0.25, 0.25, 0.25))
        )
    except Exception:
        rs, hs = set(ref.split()), set(hyp.split())
        return len(rs & hs) / len(rs) if rs else (1.0 if not hyp else 0.0)


async def main():
    print(f"Large Codebase Benchmark v2 (Fixed Context)")
    print(f"Code: {len(CODE.splitlines())} lines, {len(CODE)} chars")
    print("=" * 60)

    max_tokens = 256

    # Phase 1: Collect Ground Truth ONCE
    print("\n--- Phase 1: Collect Ground Truth ---")
    kill()
    time.sleep(2)
    p = launch()
    if not wait_ready():
        print("Server fail")
        p.terminate()
        return

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
        gt1 = await req_both(sess, build_payload(1, CODE, "lossless"))
        gt2 = await req_both(
            sess, build_payload(2, CODE, "lossless", a1_text=gt1["text"])
        )
        gt3 = await req_both(
            sess, build_payload(3, CODE, "lossless", a1_text=gt1["text"], a2_text=gt2["text"])
        )
    p.terminate()
    time.sleep(3)
    kill()

    gt = {"a1": gt1, "a2": gt2, "a3": gt3}
    print(f"  A1: {gt1['text'][:80]}...")
    print(f"  A2: {gt2['text'][:80]}...")
    print(f"  A3: {gt3['text'][:80]}...")

    # Phase 2: No-Reuse Baseline
    print("\n--- Phase 2: No-Reuse Baseline ---")
    kill()
    time.sleep(2)
    p = launch()
    if not wait_ready():
        print("Server fail")
        p.terminate()
        return

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
        nr1 = await req_both(sess, build_payload(1, CODE, "lossless"))
        nr2 = await req_both(
            sess, build_payload(2, CODE, "lossless", a1_text=nr1["text"])
        )
        nr3 = await req_both(
            sess, build_payload(3, CODE, "lossless", a1_text=nr1["text"], a2_text=nr2["text"])
        )
    p.terminate()
    time.sleep(3)
    kill()

    no_reuse = {"a1": nr1, "a2": nr2, "a3": nr3}
    for a in ["a2", "a3"]:
        print(f"  {a.upper()}: TTFT={no_reuse[a]['ttft_ms']}ms cache={no_reuse[a]['cached']}")

    # Phase 3: Full-Reuse (same GT context)
    print("\n--- Phase 3: Full-Reuse (fixed context) ---")
    kill()
    time.sleep(2)
    p = launch()
    if not wait_ready():
        print("Server fail")
        p.terminate()
        return

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
        fr1 = await req_both(sess, build_payload(1, CODE, "lossless"))
        fr2 = await req_both(
            sess, build_payload(2, CODE, "lossless", a1_text=gt1["text"])
        )
        fr3 = await req_both(
            sess, build_payload(3, CODE, "lossless", a1_text=gt1["text"], a2_text=gt2["text"])
        )
    p.terminate()
    time.sleep(3)
    kill()

    full_reuse = {"a1": fr1, "a2": fr2, "a3": fr3}
    for a in ["a2", "a3"]:
        print(f"  {a.upper()}: TTFT={full_reuse[a]['ttft_ms']}ms cache={full_reuse[a]['cached']}")

    # Phase 4: Lossy-Reuse (same GT context)
    print("\n--- Phase 4: Lossy-Reuse (fixed context) ---")
    kill()
    time.sleep(2)
    p = launch(lossy=True)
    if not wait_ready():
        print("Server fail")
        p.terminate()
        return

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
        lr1 = await req_both(sess, build_payload(1, CODE, "lossless"))
        lr2 = await req_both(
            sess, build_payload(2, CODE, "lossy", a1_text=gt1["text"])
        )
        lr3 = await req_both(
            sess, build_payload(3, CODE, "lossy", a1_text=gt1["text"], a2_text=gt2["text"])
        )
    p.terminate()
    time.sleep(3)
    kill()

    lossy_reuse = {"a1": lr1, "a2": lr2, "a3": lr3}
    for a in ["a2", "a3"]:
        m = lossy_reuse[a]["meta"]
        print(
            f"  {a.upper()}: TTFT={lossy_reuse[a]['ttft_ms']}ms cache={lossy_reuse[a]['cached']}"
            f" anchor={m.get('lossy_anchor_match_used')} len={m.get('lossy_anchor_match_len')}"
            f" delta={m.get('lossy_anchor_rope_delta')}"
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for agent in ["a2", "a3"]:
        nr = no_reuse[agent]
        fr = full_reuse[agent]
        lr = lossy_reuse[agent]
        speedup_full = (nr["ttft_ms"] / fr["ttft_ms"] - 1) * 100 if fr["ttft_ms"] > 0 else 0
        speedup_lossy = (nr["ttft_ms"] / lr["ttft_ms"] - 1) * 100 if lr["ttft_ms"] > 0 else 0
        print(f"\n{agent.upper()}:")
        print(f"  No-Reuse:    TTFT={nr['ttft_ms']}ms  cache={nr['cached']}")
        print(f"  Full-Reuse:  TTFT={fr['ttft_ms']}ms  cache={fr['cached']}  ({speedup_full:+.1f}%)")
        print(f"  Lossy-Reuse: TTFT={lr['ttft_ms']}ms  cache={lr['cached']}  ({speedup_lossy:+.1f}%)")
        print(
            f"  Anchor:      used={lr['meta'].get('lossy_anchor_match_used')}"
            f"  len={lr['meta'].get('lossy_anchor_match_len')}"
            f"  delta={lr['meta'].get('lossy_anchor_rope_delta')}"
        )

    def b(ref, hyp):
        return bleu(ref["text"], hyp["text"])

    print(f"\nBLEU (vs GT):")
    for agent in ["a2", "a3"]:
        print(
            f"  {agent.upper()} No={b(gt[agent], no_reuse[agent]):.3f}"
            f" Full={b(gt[agent], full_reuse[agent]):.3f}"
            f" Lossy={b(gt[agent], lossy_reuse[agent]):.3f}"
        )

    all_results = {
        "gt": {k: {"text": v["text"]} for k, v in gt.items()},
        "no_reuse": no_reuse,
        "full_reuse": full_reuse,
        "lossy_reuse": lossy_reuse,
    }
    (OUT / "large_codebase_v2_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False)
    )
    print(f"\nSaved to {OUT}/large_codebase_v2_results.json")


if __name__ == "__main__":
    asyncio.run(main())
