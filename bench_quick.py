#!/usr/bin/env python3
"""Quick validation benchmark: 2 test cases, all 3 modes."""
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
OUT = ROOT / "results" / "ma_ttft"
OUT.mkdir(parents=True, exist_ok=True)
TOK = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
SYSTEM = "You are a senior software engineer. Be concise and precise."

WORKFLOWS = [
    ("same-func (AVL insert)", "avl_tree_insert", "avl_tree_insert"),
    ("same-func (merge sort)", "merge_sort", "merge_sort"),
]


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
        env=e, stdout=open(str(OUT / "sglang_quick.log"), "a"), stderr=subprocess.STDOUT, cwd=str(ROOT),
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


def build_a1_payload(model, code_text, max_tokens):
    code_variant = vary_code(code_text, 1)
    a = build_code_anchor_payload(code_variant, language="python")
    msg = f"```python\n{code_variant}\n```\n\nTask: A\nAnalyze this code. Identify purpose, design patterns, and potential bugs. Keep under 100 words."
    full_text = TOK.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        tokenize=False, add_generation_prompt=False,
    )
    token_spans = compute_token_spans(full_text, code_text)
    return {
        "model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        "max_tokens": max_tokens, "temperature": 0.0,
        "code_anchor_signature": a.get("ast_anchor_signature", ""),
        "code_anchor_spans": a.get("code_anchor_spans", []),
        "code_anchor_token_spans": token_spans,
        "reuse_mode": "lossless", "lossy_alignment_method": "kvcomm",
    }


def build_a2_payload(model, code_text, a1_text, max_tokens, reuse_mode="lossless"):
    code_variant = vary_code(code_text, 2)
    a = build_code_anchor_payload(code_variant, language="python")
    msg = f"```python\n{code_variant}\n```\n\nTask: B\nImplement a fix based on this analysis: {a1_text}\n\nPropose ONE concrete fix. Write the changed code."
    full_text = TOK.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        tokenize=False, add_generation_prompt=False,
    )
    token_spans = compute_token_spans(full_text, code_text)
    return {
        "model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        "max_tokens": max_tokens, "temperature": 0.0,
        "code_anchor_signature": a.get("ast_anchor_signature", ""),
        "code_anchor_spans": a.get("code_anchor_spans", []),
        "code_anchor_token_spans": token_spans,
        "reuse_mode": reuse_mode, "lossy_alignment_method": "kvcomm",
    }


def build_a3_payload(model, code_text, a1_text, a2_text, max_tokens, reuse_mode="lossless"):
    code_variant = vary_code(code_text, 3)
    a = build_code_anchor_payload(code_variant, language="python")
    msg = f"```python\n{code_variant}\n```\n\nTask: C\nReview this implementation. Analysis: {a1_text}\n\nFix: {a2_text}\n\nIs the fix correct? One-sentence verdict."
    full_text = TOK.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        tokenize=False, add_generation_prompt=False,
    )
    token_spans = compute_token_spans(full_text, code_text)
    return {
        "model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        "max_tokens": max_tokens, "temperature": 0.0,
        "code_anchor_signature": a.get("ast_anchor_signature", ""),
        "code_anchor_spans": a.get("code_anchor_spans", []),
        "code_anchor_token_spans": token_spans,
        "reuse_mode": reuse_mode, "lossy_alignment_method": "kvcomm",
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
    cached = body.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    return {"text": text, "cached": cached}


async def req_both(sess, payload):
    stream_res = await req_stream(sess, payload)
    ns_res = await req_nonstream(sess, payload)
    meta = stream_res.get("meta", {}) or {}
    return {
        "ttft_ms": stream_res["ttft_ms"], "total_ms": stream_res["total_ms"],
        "text": ns_res["text"], "cached": stream_res["cached"], "meta": meta,
    }


def bleu(ref, hyp):
    try:
        from nltk.translate.bleu_score import sentence_bleu
        return float(sentence_bleu([ref.split()], hyp.split(), weights=(0.25, 0.25, 0.25, 0.25)))
    except Exception:
        rs, hs = set(ref.split()), set(hyp.split())
        return len(rs & hs) / len(rs) if rs else (1.0 if not hyp else 0.0)


async def run_case(sess, model, code, max_tokens, mode_name):
    """Run A1->A2->A3 for one mode."""
    r1 = await req_both(sess, build_a1_payload(model, code, max_tokens))
    r2 = await req_both(sess, build_a2_payload(model, code, r1["text"], max_tokens,
                         "lossy" if mode_name == "lossy" else "lossless"))
    r3 = await req_both(sess, build_a3_payload(model, code, r1["text"], r2["text"], max_tokens,
                         "lossy" if mode_name == "lossy" else "lossless"))
    return {"a1": r1, "a2": r2, "a3": r3}


async def main():
    all_results = []
    max_tokens = 256

    for desc, k1, k2 in WORKFLOWS:
        code = LARGE_CODE[k1]
        print(f"\n{'='*60}\n{desc} — {len(code.splitlines())}L")

        # GT
        print("  [GT] collecting ground truth...")
        kill(); time.sleep(2)
        p = launch()
        if not wait_ready():
            print("  server fail"); p.terminate(); continue
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
            gt = await run_case(sess, MODEL, code, max_tokens, "gt")
        p.terminate(); time.sleep(3); kill()
        print(f"    A1: {gt['a1']['text'][:60]}...")

        # No-Reuse
        print("  [No-Reuse] cold start...")
        kill(); time.sleep(2)
        p = launch()
        if not wait_ready():
            print("  server fail"); p.terminate(); continue
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
            no_reuse = await run_case(sess, MODEL, code, max_tokens, "no_reuse")
        p.terminate(); time.sleep(3); kill()
        print(f"    A2 TTFT={no_reuse['a2']['ttft_ms']}ms cache={no_reuse['a2']['cached']}")
        print(f"    A3 TTFT={no_reuse['a3']['ttft_ms']}ms cache={no_reuse['a3']['cached']}")

        # Full-Reuse
        print("  [Full-Reuse] warm start...")
        kill(); time.sleep(2)
        p = launch()
        if not wait_ready():
            print("  server fail"); p.terminate(); continue
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
            full_reuse = await run_case(sess, MODEL, code, max_tokens, "full")
        p.terminate(); time.sleep(3); kill()
        print(f"    A2 TTFT={full_reuse['a2']['ttft_ms']}ms cache={full_reuse['a2']['cached']}")
        print(f"    A3 TTFT={full_reuse['a3']['ttft_ms']}ms cache={full_reuse['a3']['cached']}")

        # Lossy-Reuse
        print("  [Lossy-Reuse] lossy match...")
        kill(); time.sleep(2)
        p = launch(lossy=True)
        if not wait_ready():
            print("  server fail"); p.terminate(); continue
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
            lossy_reuse = await run_case(sess, MODEL, code, max_tokens, "lossy")
        p.terminate(); time.sleep(3); kill()
        a2_anchor = lossy_reuse['a2']['meta'].get('lossy_anchor_match_used', False)
        a3_anchor = lossy_reuse['a3']['meta'].get('lossy_anchor_match_used', False)
        print(f"    A2 TTFT={lossy_reuse['a2']['ttft_ms']}ms cache={lossy_reuse['a2']['cached']} anchor={a2_anchor}")
        print(f"    A3 TTFT={lossy_reuse['a3']['ttft_ms']}ms cache={lossy_reuse['a3']['cached']} anchor={a3_anchor}")

        def b(ref, hyp):
            return bleu(ref["text"], hyp["text"])

        all_results.append({
            "name": desc,
            "no_reuse": {"a2": no_reuse['a2'], "a3": no_reuse['a3']},
            "full_reuse": {"a2": full_reuse['a2'], "a3": full_reuse['a3']},
            "lossy_reuse": {"a2": lossy_reuse['a2'], "a3": lossy_reuse['a3']},
            "bleu": {
                "a2_full": b(gt['a2'], full_reuse['a2']), "a2_lossy": b(gt['a2'], lossy_reuse['a2']),
                "a3_full": b(gt['a3'], full_reuse['a3']), "a3_lossy": b(gt['a3'], lossy_reuse['a3']),
            }
        })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in all_results:
        print(f"\n{r['name']}:")
        for agent in ["a2", "a3"]:
            nr = r["no_reuse"][agent]
            fr = r["full_reuse"][agent]
            lr = r["lossy_reuse"][agent]
            speedup_full = (nr['ttft_ms'] / fr['ttft_ms'] - 1) * 100 if fr['ttft_ms'] > 0 else 0
            speedup_lossy = (nr['ttft_ms'] / lr['ttft_ms'] - 1) * 100 if lr['ttft_ms'] > 0 else 0
            print(f"  {agent.upper()}: No={nr['ttft_ms']}ms Full={fr['ttft_ms']}ms({speedup_full:+.1f}%)"
                  f" Lossy={lr['ttft_ms']}ms({speedup_lossy:+.1f}%)"
                  f" cache_no={nr['cached']} cache_full={fr['cached']} cache_lossy={lr['cached']}")
        print(f"  BLEU: A2_full={r['bleu']['a2_full']:.3f} A2_lossy={r['bleu']['a2_lossy']:.3f}"
              f" A3_full={r['bleu']['a3_full']:.3f} A3_lossy={r['bleu']['a3_lossy']:.3f}")

    (OUT / "quick_results.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nDone. Results saved to {OUT}/quick_results.json")


if __name__ == "__main__":
    asyncio.run(main())
