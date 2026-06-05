#!/usr/bin/env python3
"""Experiment 3: Natural Prompt Test — verify BLEU recovers with natural prompts.

Uses natural prompt design where code block comes AFTER the instruction
(rather than before). Different instructions naturally break exact prefix match.
Since positions are no longer aligned, RoPE delta rotation is required.
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
        stdout=open(str(ROOT / "results" / "ma_ttft" / "sglang_natural.log"), "a"),
        stderr=subprocess.STDOUT,
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


def build_payload(agent_id, code_text, reuse_mode):
    """Build payload with NATURAL prompt design.

    Code block comes AFTER the instruction, not before.
    Different instructions naturally break exact prefix match.
    """
    instructions = {
        1: "Analyze this code. Identify purpose, design patterns, and potential bugs. Keep under 100 words.",
        2: "Implement a fix for the following code. Propose ONE concrete fix and write the changed code.",
        3: "Review this implementation. Is the fix correct? Give a one-sentence verdict.",
    }
    msg = f"{instructions[agent_id]}\n\n```python\n{code_text}\n```"

    full_text = TOK.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        tokenize=False, add_generation_prompt=False,
    )
    token_spans = compute_token_spans(full_text, code_text)
    print(f"  [Agent {agent_id}] code_pos={token_spans[0]['start_token'] if token_spans else 'N/A'}")

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


def bleu(ref, hyp):
    try:
        from nltk.translate.bleu_score import sentence_bleu
        return float(sentence_bleu([ref.split()], hyp.split(), weights=(0.25, 0.25, 0.25, 0.25)))
    except Exception:
        rs, hs = set(ref.split()), set(hyp.split())
        return len(rs & hs) / len(rs) if rs else (1.0 if not hyp else 0.0)


async def main():
    print("Experiment 3: Natural Prompt Test")
    print("Model:", MODEL)
    print("=" * 60)

    code = LARGE_CODE["avl_tree_insert"]
    print(f"Code lines: {len(code.splitlines())}")

    # Phase 1: Collect ground truth (no reuse, deterministic)
    print("\n--- Phase 1: Ground Truth (no KV reuse) ---")
    kill()
    time.sleep(2)
    p = launch(lossy=False)
    if not wait_ready():
        print("Server failed")
        p.terminate()
        return

    async with aiohttp.ClientSession() as sess:
        gt1 = await req_stream(sess, build_payload(1, code, "lossless"))
        gt2 = await req_stream(sess, build_payload(2, code, "lossless"))
        gt3 = await req_stream(sess, build_payload(3, code, "lossless"))

    p.terminate()
    time.sleep(3)
    kill()

    print(f"  A1 GT: TTFT={gt1['ttft_ms']}ms text={gt1['text'][:80]}...")
    print(f"  A2 GT: TTFT={gt2['ttft_ms']}ms text={gt2['text'][:80]}...")
    print(f"  A3 GT: TTFT={gt3['ttft_ms']}ms text={gt3['text'][:80]}...")

    # Phase 2: Lossy reuse with natural prompts
    print("\n--- Phase 2: Lossy Reuse (natural prompts) ---")
    kill()
    time.sleep(2)
    p = launch(lossy=True)
    if not wait_ready():
        print("Server failed")
        p.terminate()
        return

    async with aiohttp.ClientSession() as sess:
        r1 = await req_stream(sess, build_payload(1, code, "lossless"))
        print(f"  A1: TTFT={r1['ttft_ms']}ms cached={r1['cached']} text={r1['text'][:80]}...")

        r2 = await req_stream(sess, build_payload(2, code, "lossy"))
        m2 = r2['meta']
        print(f"  A2: TTFT={r2['ttft_ms']}ms cached={r2['cached']}"
              f" anchor={m2.get('lossy_anchor_match_used')} len={m2.get('lossy_anchor_match_len')}"
              f" gap={m2.get('lossy_anchor_match_gap_len')} delta={m2.get('lossy_anchor_rope_delta')}"
              f" text={r2['text'][:80]}...")

        r3 = await req_stream(sess, build_payload(3, code, "lossy"))
        m3 = r3['meta']
        print(f"  A3: TTFT={r3['ttft_ms']}ms cached={r3['cached']}"
              f" anchor={m3.get('lossy_anchor_match_used')} len={m3.get('lossy_anchor_match_len')}"
              f" gap={m3.get('lossy_anchor_match_gap_len')} delta={m3.get('lossy_anchor_rope_delta')}"
              f" text={r3['text'][:80]}...")

    p.terminate()
    time.sleep(3)
    kill()

    # Phase 3: BLEU comparison
    print("\n--- Phase 3: BLEU Comparison ---")
    bleu_a1 = bleu(gt1["text"], r1["text"])
    bleu_a2 = bleu(gt2["text"], r2["text"])
    bleu_a3 = bleu(gt3["text"], r3["text"])
    print(f"  A1 BLEU: {bleu_a1:.3f} (baseline, no reuse)")
    print(f"  A2 BLEU: {bleu_a2:.3f} (lossy reuse)")
    print(f"  A3 BLEU: {bleu_a3:.3f} (lossy reuse)")

    print("\n" + "=" * 60)
    print("Experiment 3 complete.")
    if bleu_a2 >= 0.95 and bleu_a3 >= 0.95:
        print("✅ PASSED: BLEU recovered with natural prompts!")
    else:
        print("⚠️  BLEU below 0.95 — investigate output differences.")


if __name__ == "__main__":
    asyncio.run(main())
