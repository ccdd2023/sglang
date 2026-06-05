#!/usr/bin/env python3
"""Final benchmark: Short gap (variant comment) + RoPE delta rotation.

Tests the realistic multi-agent workflow:
- All agents process the SAME codebase
- Code has a small variant comment to break exact match
- Gap is only 3 tokens → zero-fill impact is negligible
- Code base is 298 lines (~2300 tokens)
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
        env=e, stdout=open(str(OUT / "sglang_final.log"), "a"), stderr=subprocess.STDOUT, cwd=str(ROOT),
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
    """Find token spans using string position (handles tokenizer context differences)."""
    idx = full_text.find(code_text)
    if idx == -1:
        return []
    prefix = full_text[:idx]
    prefix_ids = TOK.encode(prefix, add_special_tokens=False)
    code_ids = TOK.encode(code_text, add_special_tokens=False)
    start_token = len(prefix_ids)
    end_token = start_token + len(code_ids)
    return [{"start_token": start_token, "end_token": end_token}]


def vary_code(code_text, agent_id):
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
        instruction = f"Review this implementation. Analysis: {a1_text}\n\nFix: {a2_text}\n\nIs the fix correct and efficient? One-sentence verdict."
        msg = f"```python\n{code_variant}\n```\n\n{instruction}"

    full_text = TOK.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        tokenize=False, add_generation_prompt=False,
    )
    token_spans = compute_token_spans(full_text, code_variant)

    a = build_code_anchor_payload(code_variant, language="python")
    return {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        "max_tokens": 256, "temperature": 0.0,
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
    return {"ttft_ms": round(ttft, 1), "total_ms": round(total, 1), "text": text, "cached": cached, "meta": meta}


async def req_nonstream(sess, payload):
    async with sess.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
        body = await resp.json()
    text = body["choices"][0]["message"]["content"] if body.get("choices") else ""
    return {"text": text}


async def req_both(sess, payload):
    stream_res = await req_stream(sess, payload)
    ns_res = await req_nonstream(sess, payload)
    meta = stream_res.get("meta", {}) or {}
    return {**stream_res, "text": ns_res["text"]}


def bleu(ref, hyp):
    try:
        from nltk.translate.bleu_score import sentence_bleu
        return float(sentence_bleu([ref.split()], hyp.split(), weights=(0.25, 0.25, 0.25, 0.25)))
    except Exception:
        rs, hs = set(ref.split()), set(hyp.split())
        return len(rs & hs) / len(rs) if rs else (1.0 if not hyp else 0.0)


async def run_case(sess, code, max_tokens, mode_name):
    r1 = await req_both(sess, build_payload(1, code, "lossy" if mode_name == "lossy" else "lossless"))
    r2 = await req_both(sess, build_payload(2, code, "lossy" if mode_name == "lossy" else "lossless", a1_text=r1["text"]))
    r3 = await req_both(sess, build_payload(3, code, "lossy" if mode_name == "lossy" else "lossless", a1_text=r1["text"], a2_text=r2["text"]))
    return {"a1": r1, "a2": r2, "a3": r3}


async def main():
    print("=" * 70)
    print("FINAL BENCHMARK: Short Gap + Large Codebase")
    print(f"Code: {len(CODE.splitlines())} lines, {len(CODE)} chars")
    print("=" * 70)

    max_tokens = 256
    all_results = {}

    # Phase 1: GT
    print("\n[1/4] Ground Truth (lossless, no variant)")
    kill(); time.sleep(2)
    p = launch(lossy=False)
    if not wait_ready():
        print("Server fail"); p.terminate(); return
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
        gt = await run_case(sess, CODE, max_tokens, "gt")
    p.terminate(); time.sleep(3); kill()
    for a in ["a1", "a2", "a3"]:
        print(f"  {a.upper()}: {gt[a]['text'][:70]}...")

    # Phase 2: No-Reuse
    print("\n[2/4] No-Reuse (cold start)")
    kill(); time.sleep(2)
    p = launch(lossy=False)
    if not wait_ready():
        print("Server fail"); p.terminate(); return
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
        no_reuse = await run_case(sess, CODE, max_tokens, "no_reuse")
    p.terminate(); time.sleep(3); kill()
    for a in ["a2", "a3"]:
        print(f"  {a.upper()}: TTFT={no_reuse[a]['ttft_ms']}ms cache={no_reuse[a]['cached']}")

    # Phase 3: Full-Reuse
    print("\n[3/4] Full-Reuse (warm start, same code)")
    kill(); time.sleep(2)
    p = launch(lossy=False)
    if not wait_ready():
        print("Server fail"); p.terminate(); return
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
        # A1 populates cache
        fr1 = await req_both(sess, build_payload(1, CODE, "lossless"))
        # A2/A3 reuse (same code → exact match through code)
        fr2 = await req_both(sess, build_payload(2, CODE, "lossless", a1_text=gt["a1"]["text"]))
        fr3 = await req_both(sess, build_payload(3, CODE, "lossless", a1_text=gt["a1"]["text"], a2_text=gt["a2"]["text"]))
    p.terminate(); time.sleep(3); kill()
    full_reuse = {"a1": fr1, "a2": fr2, "a3": fr3}
    for a in ["a2", "a3"]:
        print(f"  {a.upper()}: TTFT={full_reuse[a]['ttft_ms']}ms cache={full_reuse[a]['cached']}")

    # Phase 4: Lossy-Reuse
    print("\n[4/4] Lossy-Reuse (variant comment, short gap)")
    kill(); time.sleep(2)
    p = launch(lossy=True)
    if not wait_ready():
        print("Server fail"); p.terminate(); return
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
        lossy = await run_case(sess, CODE, max_tokens, "lossy")
    p.terminate(); time.sleep(3); kill()
    for a in ["a2", "a3"]:
        m = lossy[a]['meta']
        print(f"  {a.upper()}: TTFT={lossy[a]['ttft_ms']}ms cache={lossy[a]['cached']}"
              f" anchor={m.get('lossy_anchor_match_used')} len={m.get('lossy_anchor_match_len')}"
              f" gap={m.get('lossy_anchor_match_gap_len')} delta={m.get('lossy_anchor_rope_delta')}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for agent in ["a2", "a3"]:
        nr = no_reuse[agent]
        fr = full_reuse[agent]
        lr = lossy[agent]
        speedup_fr = (nr["ttft_ms"] / fr["ttft_ms"] - 1) * 100 if fr["ttft_ms"] > 0 else 0
        speedup_lr = (nr["ttft_ms"] / lr["ttft_ms"] - 1) * 100 if lr["ttft_ms"] > 0 else 0
        print(f"\n{agent.upper()}:")
        print(f"  No-Reuse:    TTFT={nr['ttft_ms']:6.1f}ms  cache={nr['cached']:4d}  baseline")
        print(f"  Full-Reuse:  TTFT={fr['ttft_ms']:6.1f}ms  cache={fr['cached']:4d}  ({speedup_fr:+6.1f}%)")
        print(f"  Lossy-Reuse: TTFT={lr['ttft_ms']:6.1f}ms  cache={lr['cached']:4d}  ({speedup_lr:+6.1f}%)")

    def b(ref, hyp):
        return bleu(ref["text"], hyp["text"])

    print(f"\nBLEU (vs GT):")
    for agent in ["a1", "a2", "a3"]:
        b_no = b(gt[agent], no_reuse[agent])
        b_fr = b(gt[agent], full_reuse[agent])
        b_lr = b(gt[agent], lossy[agent])
        print(f"  {agent.upper()}: No={b_no:.3f}  Full={b_fr:.3f}  Lossy={b_lr:.3f}")

    # Verify exact match
    print(f"\nExact Match (vs GT):")
    for agent in ["a1", "a2", "a3"]:
        m_no = gt[agent]["text"] == no_reuse[agent]["text"]
        m_fr = gt[agent]["text"] == full_reuse[agent]["text"]
        m_lr = gt[agent]["text"] == lossy[agent]["text"]
        print(f"  {agent.upper()}: No={m_no}  Full={m_fr}  Lossy={m_lr}")

    all_results = {
        "gt": {k: {"text": v["text"]} for k, v in gt.items()},
        "no_reuse": no_reuse, "full_reuse": full_reuse, "lossy_reuse": lossy,
    }
    (OUT / "final_results.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nSaved to {OUT}/final_results.json")


if __name__ == "__main__":
    asyncio.run(main())
