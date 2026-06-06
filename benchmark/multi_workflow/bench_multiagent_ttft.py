#!/usr/bin/env python3
"""Multi-Agent Intermediate-Context KV Reuse — TTFT Acceleration Experiment.

Measures TTFT (Time To First Token) acceleration when reusing intermediate
agent outputs in a chain: Analyzer -> Implementer -> Reviewer.

Compares three configurations:
  1. No-Reuse:    server restarted between agents (cold start every time)
  2. Full-Reuse:  server kept running, lossless prefix match
  3. Lossy-Reuse: server kept running, lossy semantic match with anchor KV store

Key design: Code block is placed FIRST in the user message, before any
agent-specific instruction. This ensures the code block starts at the same
absolute token position across all agents. Lossy mode uses the anchor KV store
to extend prefix matches when exact token matching stops early.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

# Import LARGE_CODE and WORKFLOWS from multi-agent benchmark
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3].parent / "MAScoder" / "src"))
from bench_multiagent_large import LARGE_CODE  # type: ignore
from mascoder.code_anchor import build_code_anchor_payload

# Same-file workflows: A1/A2/A3 all process the SAME code.
WORKFLOWS = [
    ("same-func (AVL insert)", "avl_tree_insert", "avl_tree_insert"),
    ("same-func (RedBlack insert)", "rbtree_insert", "rbtree_insert"),
    ("same-func (merge sort)", "merge_sort", "merge_sort"),
    ("same-func (heap sort)", "heap_sort", "heap_sort"),
    ("same-func (Dijkstra)", "dijkstra", "dijkstra"),
    ("same-func (BFS)", "bfs_shortest", "bfs_shortest"),
]

PORT = 30000
MODEL = "/home/gfy/models/Qwen2.5-3B-Instruct"
ROOT = Path(__file__).resolve().parents[3] / "sglang-kvflow"
PY = "/home/gfy/.conda/envs/sglang-kvflow/bin/python"
OUT = Path(__file__).resolve().parents[3] / "sglang-kvflow" / "results" / "ma_ttft"
OUT.mkdir(parents=True, exist_ok=True)

SYSTEM = "You are a senior software engineer. Be concise and precise."
TOK = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def compute_token_spans(full_text, code_text):
    """Find token start/end positions of code_text within full_text."""
    full_ids = TOK.encode(full_text, add_special_tokens=False)
    code_ids = TOK.encode(code_text, add_special_tokens=False)
    content_signature = hashlib.sha256((code_text or "").encode("utf-8")).hexdigest()[:32]
    for i in range(len(full_ids) - len(code_ids) + 1):
        if full_ids[i : i + len(code_ids)] == code_ids:
            return [
                {
                    "start_token": i,
                    "end_token": i + len(code_ids),
                    "content_signature": content_signature,
                }
            ]
    return []


def vary_code(code_text, agent_id):
    """Return a syntactically identical but textually different code variant.

    All agents get a divergent comment prefix so the exact prefix match stops
    before the shared code block. The anchor span (original code) is then
    reused via the anchor KV store.
    """
    return f"# A{agent_id} variant\n" + code_text


# ---- Server lifecycle (reused from existing benchmarks) ----

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


def launch(lossy=False, model_path=None, port=None):
    e = os.environ.copy()
    e["PYTHONPATH"] = str(ROOT / "python") + (
        ":" + e.get("PYTHONPATH", "") if e.get("PYTHONPATH") else ""
    )
    if lossy:
        e["SGLANG_LOSSY_FUZZY_MATCH"] = "1"
        e["SGLANG_LOSSY_SKIP_TOKEN_CHECK"] = "1"
    return subprocess.Popen(
        [
            PY,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path or MODEL,
            "--port",
            str(port or PORT),
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.85",
            "--max-total-tokens",
            "65536",
            "--chunked-prefill-size",
            "8192",
            "--max-prefill-tokens",
            "16384",
            "--radix-eviction-policy",
            "priority",
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "1.5",
            "--hicache-write-policy",
            "write_back",
            "--enable-cache-report",
            "--disable-cuda-graph",
            "--log-level",
            "error",
        ],
        env=e,
        stdout=open(str(OUT / "sglang.log"), "a"),
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
    )


def wait_ready(t=150, port=PORT):
    import urllib.request

    d = time.monotonic() + t
    while time.monotonic() < d:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health_generate", timeout=5
            ) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(5)
    return False


# ---- BLEU (reused from bench_swe_lite_kv.py) ----

def bleu(ref, hyp):
    try:
        from nltk.translate.bleu_score import sentence_bleu

        return float(
            sentence_bleu([ref.split()], hyp.split(), weights=(0.25, 0.25, 0.25, 0.25))
        )
    except Exception:
        rs, hs = set(ref.split()), set(hyp.split())
        return len(rs & hs) / len(rs) if rs else (1.0 if not hyp else 0.0)


# ---- HTTP requests ----

BASE_URL = f"http://127.0.0.1:{PORT}"


def build_a1_payload(model, code_text, max_tokens):
    """Agent 1: Analyze code."""
    code_variant = vary_code(code_text, 1)
    a = build_code_anchor_payload(code_variant, language="python")
    msg = (
        f"```python\n{code_variant}\n```\n\n"
        "Task: A\nAnalyze this code. Identify purpose, design patterns, and potential bugs. Keep under 100 words."
    )
    full_text = TOK.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        tokenize=False,
        add_generation_prompt=False,
    )
    # Anchor span is the ORIGINAL code (shared across agents), not the variant
    token_spans = compute_token_spans(full_text, code_text)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": msg},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "code_anchor_signature": a.get("ast_anchor_signature", ""),
        "code_content_signature": a.get("code_content_signature", ""),
        "code_anchor_spans": a.get("code_anchor_spans", []),
        "code_anchor_token_spans": token_spans,
        "reuse_mode": "lossless",
        "lossy_alignment_method": "kvcomm",
    }


def build_a2_payload(model, code_text, a1_text, max_tokens, reuse_mode="lossless"):
    """Agent 2: Implement fix."""
    code_variant = vary_code(code_text, 2)
    a = build_code_anchor_payload(code_variant, language="python")
    msg = (
        f"```python\n{code_variant}\n```\n\n"
        f"Task: B\nImplement a fix based on this analysis: {a1_text}\n\n"
        "Propose ONE concrete fix. Write the changed code."
    )
    full_text = TOK.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        tokenize=False,
        add_generation_prompt=False,
    )
    # Anchor span is the ORIGINAL code (shared across agents)
    token_spans = compute_token_spans(full_text, code_text)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": msg},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "code_anchor_signature": a.get("ast_anchor_signature", ""),
        "code_content_signature": a.get("code_content_signature", ""),
        "code_anchor_spans": a.get("code_anchor_spans", []),
        "code_anchor_token_spans": token_spans,
        "reuse_mode": reuse_mode,
        "lossy_alignment_method": "kvcomm",
    }


def build_a3_payload(
    model, code_text, a1_text, a2_text, max_tokens, reuse_mode="lossless"
):
    """Agent 3: Review."""
    code_variant = vary_code(code_text, 3)
    a = build_code_anchor_payload(code_variant, language="python")
    msg = (
        f"```python\n{code_variant}\n```\n\n"
        f"Task: C\nReview this implementation. Analysis: {a1_text}\n\n"
        f"Fix: {a2_text}\n\n"
        "Is the fix correct? One-sentence verdict."
    )
    full_text = TOK.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
        tokenize=False,
        add_generation_prompt=False,
    )
    # Anchor span is the ORIGINAL code (shared across agents)
    token_spans = compute_token_spans(full_text, code_text)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": msg},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "code_anchor_signature": a.get("ast_anchor_signature", ""),
        "code_content_signature": a.get("code_content_signature", ""),
        "code_anchor_spans": a.get("code_anchor_spans", []),
        "code_anchor_token_spans": token_spans,
        "reuse_mode": reuse_mode,
        "lossy_alignment_method": "kvcomm",
    }


async def req_stream(sess, payload):
    """Streaming request: measures TTFT + total latency + text + metadata."""
    start = time.perf_counter()
    ttft = None
    text = ""
    cached_tokens = 0
    meta = {}
    body = None

    async with sess.post(
        f"{BASE_URL}/v1/chat/completions",
        json={**payload, "stream": True, "stream_options": {"include_usage": True}},
        timeout=aiohttp.ClientTimeout(total=300),
    ) as resp:
        async for line in resp.content:
            line = line.decode().strip()
            if not line:
                continue
            if line == "data: [DONE]":
                break
            if line.startswith("data: "):
                try:
                    chunk = json.loads(line[6:])
                    if ttft is None:
                        ttft = (time.perf_counter() - start) * 1000
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        text += delta.get("content", "")
                    usage = chunk.get("usage")
                    if usage and usage.get("prompt_tokens_details"):
                        cached_tokens = usage["prompt_tokens_details"].get("cached_tokens", 0)
                    # Capture lossy metadata from streaming response
                    chunk_meta = chunk.get("metadata", {})
                    if chunk_meta.get("lossy_reuse"):
                        meta = chunk_meta["lossy_reuse"]
                except Exception:
                    pass

    total = (time.perf_counter() - start) * 1000
    return {
        "ttft_ms": round(ttft, 1) if ttft else None,
        "total_ms": round(total, 1),
        "text": text,
        "cached_tokens": cached_tokens,
        "meta": meta,
    }


async def req_nonstream(sess, payload):
    """Non-streaming request: gets cached_tokens from usage."""
    start = time.perf_counter()
    async with sess.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=300),
    ) as resp:
        body = await resp.json()
    total = (time.perf_counter() - start) * 1000

    text = ""
    try:
        text = body["choices"][0]["message"]["content"]
    except Exception:
        pass

    cached = 0
    try:
        cached = body["usage"]["prompt_tokens_details"].get("cached_tokens", 0)
    except Exception:
        pass

    meta = {}
    try:
        meta = body["metadata"]["lossy_reuse"]
    except Exception:
        pass

    return {
        "total_ms": round(total, 1),
        "text": text,
        "cached_tokens": cached,
        "meta": meta,
    }


async def req_both(sess, payload):
    """Get TTFT+cached+metadata from streaming, text quality from non-streaming."""
    stream_res = await req_stream(sess, payload)
    ns_res = await req_nonstream(sess, payload)
    # Prefer streaming metadata (has anchor KV info from actual match path)
    meta = stream_res.get("meta", {}) or ns_res["meta"]
    return {
        "ttft_ms": stream_res["ttft_ms"],
        "total_ms": stream_res["total_ms"],
        "text": ns_res["text"],
        "cached_tokens": stream_res["cached_tokens"],
        "meta": meta,
    }


# ---- Experiment runners ----


async def run_ground_truth(sess, model, code1, code2, max_tokens):
    """Run once to collect deterministic outputs (temperature=0).

    A1 analyzes code1. A2/A3 analyze code2 (cross-file reuse).
    """
    r1 = await req_both(sess, build_a1_payload(model, code1, max_tokens))
    a1_text = r1["text"]

    r2 = await req_both(
        sess, build_a2_payload(model, code2, a1_text, max_tokens, "lossless")
    )
    a2_text = r2["text"]

    r3 = await req_both(
        sess,
        build_a3_payload(model, code2, a1_text, a2_text, max_tokens, "lossless"),
    )
    a3_text = r3["text"]

    return {
        "a1": {"text": a1_text},
        "a2": {"text": a2_text},
        "a3": {"text": a3_text},
    }


async def run_no_reuse(sess, model, code1, code2, gt, max_tokens):
    """No-Reuse: restart server between every agent."""
    global _BENCH_ARGS
    args = _BENCH_ARGS
    results = {}

    # A1 (code1)
    kill()
    time.sleep(2)
    p = launch(model_path=args.model, port=args.port)
    if not wait_ready():
        p.terminate()
        return None
    r1 = await req_both(sess, build_a1_payload(model, code1, max_tokens))
    results["a1"] = r1
    p.terminate()
    time.sleep(3)
    kill()

    # A2 (code2, with A1 output as prefix)
    time.sleep(2)
    p = launch(model_path=args.model, port=args.port)
    if not wait_ready():
        p.terminate()
        return None
    r2 = await req_both(
        sess, build_a2_payload(model, code2, gt["a1"]["text"], max_tokens, "lossless")
    )
    results["a2"] = r2
    p.terminate()
    time.sleep(3)
    kill()

    # A3 (code2, with A1+A2 outputs as prefix)
    time.sleep(2)
    p = launch(model_path=args.model, port=args.port)
    if not wait_ready():
        p.terminate()
        return None
    r3 = await req_both(
        sess,
        build_a3_payload(
            model, code2, gt["a1"]["text"], gt["a2"]["text"], max_tokens, "lossless"
        ),
    )
    results["a3"] = r3
    p.terminate()
    time.sleep(3)
    kill()

    return results


async def run_full_reuse(sess, model, code1, code2, gt, max_tokens):
    """Full-Reuse: single server, lossless mode."""
    global _BENCH_ARGS
    args = _BENCH_ARGS
    results = {}

    r1 = await req_both(sess, build_a1_payload(model, code1, max_tokens))
    results["a1"] = r1

    r2 = await req_both(
        sess, build_a2_payload(model, code2, gt["a1"]["text"], max_tokens, "lossless")
    )
    results["a2"] = r2

    r3 = await req_both(
        sess,
        build_a3_payload(
            model, code2, gt["a1"]["text"], gt["a2"]["text"], max_tokens, "lossless"
        ),
    )
    results["a3"] = r3

    return results


async def run_lossy_reuse(sess, model, code1, code2, gt, max_tokens):
    """Lossy-Reuse: single server, lossy mode."""
    global _BENCH_ARGS
    args = _BENCH_ARGS
    results = {}

    r1 = await req_both(sess, build_a1_payload(model, code1, max_tokens))
    results["a1"] = r1

    r2 = await req_both(
        sess, build_a2_payload(model, code2, gt["a1"]["text"], max_tokens, "lossy")
    )
    results["a2"] = r2

    r3 = await req_both(
        sess,
        build_a3_payload(
            model, code2, gt["a1"]["text"], gt["a2"]["text"], max_tokens, "lossy"
        ),
    )
    results["a3"] = r3

    return results


# ---- Main ----


async def main(args):
    all_results = []

    for desc, k1, k2 in WORKFLOWS:
        code1 = LARGE_CODE[k1]
        code2 = LARGE_CODE[k2]
        print(
            f"\n{'=' * 60}\n{desc} — A1:{k1}({len(code1.splitlines())}L) "
            f"A2/A3:{k2}({len(code2.splitlines())}L)"
        )

        # Phase 1: Collect Ground Truth (single server run)
        print("  [GT] collecting ground truth...")
        kill()
        time.sleep(2)
        p = launch(model_path=args.model, port=args.port)
        if not wait_ready():
            print("  server fail")
            p.terminate()
            continue

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)
        ) as sess:
            gt = await run_ground_truth(sess, args.model, code1, code2, args.max_tokens)
        p.terminate()
        time.sleep(3)
        kill()

        print(f"    A1 GT: {gt['a1']['text'][:80]}...")
        print(f"    A2 GT: {gt['a2']['text'][:80]}...")
        print(f"    A3 GT: {gt['a3']['text'][:80]}...")

        # Phase 2: No-Reuse
        print("  [No-Reuse] cold start every agent...")
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)
        ) as sess:
            no_reuse = await run_no_reuse(
                sess, args.model, code1, code2, gt, args.max_tokens
            )
        if no_reuse:
            print(
                f"    A1 TTFT={no_reuse['a1']['ttft_ms']:.0f}ms "
                f"total={no_reuse['a1']['total_ms']:.0f}ms "
                f"cache={no_reuse['a1']['cached_tokens']}"
            )
            print(
                f"    A2 TTFT={no_reuse['a2']['ttft_ms']:.0f}ms "
                f"total={no_reuse['a2']['total_ms']:.0f}ms "
                f"cache={no_reuse['a2']['cached_tokens']}"
            )
            print(
                f"    A3 TTFT={no_reuse['a3']['ttft_ms']:.0f}ms "
                f"total={no_reuse['a3']['total_ms']:.0f}ms "
                f"cache={no_reuse['a3']['cached_tokens']}"
            )

        # Phase 3: Full-Reuse
        print("  [Full-Reuse] warm start with prefix match...")
        kill()
        time.sleep(2)
        p = launch(model_path=args.model, port=args.port)
        if not wait_ready():
            print("  server fail")
            p.terminate()
            continue

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)
        ) as sess:
            full_reuse = await run_full_reuse(
                sess, args.model, code1, code2, gt, args.max_tokens
            )
        p.terminate()
        time.sleep(3)
        kill()

        print(
            f"    A1 TTFT={full_reuse['a1']['ttft_ms']:.0f}ms "
            f"total={full_reuse['a1']['total_ms']:.0f}ms "
            f"cache={full_reuse['a1']['cached_tokens']}"
        )
        print(
            f"    A2 TTFT={full_reuse['a2']['ttft_ms']:.0f}ms "
            f"total={full_reuse['a2']['total_ms']:.0f}ms "
            f"cache={full_reuse['a2']['cached_tokens']}"
        )
        print(
            f"    A3 TTFT={full_reuse['a3']['ttft_ms']:.0f}ms "
            f"total={full_reuse['a3']['total_ms']:.0f}ms "
            f"cache={full_reuse['a3']['cached_tokens']}"
        )

        # Phase 4: Lossy-Reuse
        print("  [Lossy-Reuse] warm start with lossy match...")
        kill()
        time.sleep(2)
        p = launch(lossy=True, model_path=args.model, port=args.port)
        if not wait_ready():
            print("  server fail")
            p.terminate()
            continue

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)
        ) as sess:
            lossy_reuse = await run_lossy_reuse(
                sess, args.model, code1, code2, gt, args.max_tokens
            )
        p.terminate()
        time.sleep(3)
        kill()

        print(
            f"    A1 TTFT={lossy_reuse['a1']['ttft_ms']:.0f}ms "
            f"total={lossy_reuse['a1']['total_ms']:.0f}ms "
            f"cache={lossy_reuse['a1']['cached_tokens']}"
        )
        a2_anchor = lossy_reuse['a2']['meta'].get('lossy_anchor_match_used', False)
        a3_anchor = lossy_reuse['a3']['meta'].get('lossy_anchor_match_used', False)
        a2_anchor_len = lossy_reuse['a2']['meta'].get('lossy_anchor_match_len', 0)
        a3_anchor_len = lossy_reuse['a3']['meta'].get('lossy_anchor_match_len', 0)
        print(
            f"    A2 TTFT={lossy_reuse['a2']['ttft_ms']:.0f}ms "
            f"total={lossy_reuse['a2']['total_ms']:.0f}ms "
            f"cache={lossy_reuse['a2']['cached_tokens']} "
            f"meta={lossy_reuse['a2']['meta'].get('lossy_first_match_reason', '')}"
            f" anchor={a2_anchor}({a2_anchor_len})"
        )
        print(
            f"    A3 TTFT={lossy_reuse['a3']['ttft_ms']:.0f}ms "
            f"total={lossy_reuse['a3']['total_ms']:.0f}ms "
            f"cache={lossy_reuse['a3']['cached_tokens']} "
            f"meta={lossy_reuse['a3']['meta'].get('lossy_first_match_reason', '')}"
            f" anchor={a3_anchor}({a3_anchor_len})"
        )

        # Compute BLEU accuracy
        def compute_bleus(res, gt):
            return {
                "a1": bleu(gt["a1"]["text"], res["a1"]["text"]),
                "a2": bleu(gt["a2"]["text"], res["a2"]["text"]),
                "a3": bleu(gt["a3"]["text"], res["a3"]["text"]),
            }

        all_results.append(
            {
                "name": desc,
                "code1": k1,
                "code2": k2,
                "code1_lines": len(code1.splitlines()),
                "code2_lines": len(code2.splitlines()),
                "gt": gt,
                "no_reuse": {
                    **no_reuse,
                    "bleu": compute_bleus(no_reuse, gt) if no_reuse else {},
                },
                "full_reuse": {
                    **full_reuse,
                    "bleu": compute_bleus(full_reuse, gt),
                },
                "lossy_reuse": {
                    **lossy_reuse,
                    "bleu": compute_bleus(lossy_reuse, gt),
                },
            }
        )

    report(all_results)
    print(f"\nDone -> {OUT}/")


def report(results):
    (OUT / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    lines = [
        "# Multi-Agent Intermediate-Context KV Reuse — TTFT Acceleration",
        "",
        f"Model: {_BENCH_ARGS.model.split('/')[-1]} | {len(results)} files | 3-Agent workflow",
        "",
        "## Summary",
        "",
    ]

    # Aggregate TTFT
    for agent in ["a1", "a2", "a3"]:
        lines.append(f"### {agent.upper()}")
        lines.append("")
        lines.append(
            "| File | No-Reuse TTFT | Full-Reuse TTFT | Lossy-Reuse TTFT | "
            "No-Reuse Total | Full-Reuse Total | Lossy-Reuse Total | "
            "Full Cache | Lossy Cache | BLEU Full | BLEU Lossy |"
        )
        lines.append(
            "|---|---|---|---|---|---|---|---|---|---|---|"
        )

        for r in results:
            nr = r["no_reuse"].get(agent, {})
            fr = r["full_reuse"].get(agent, {})
            lr = r["lossy_reuse"].get(agent, {})

            def get_val(d, k):
                v = d.get(k)
                return f"{v:.0f}" if v is not None else "N/A"

            bleu_fr = r["full_reuse"].get("bleu", {}).get(agent, 0)
            bleu_lr = r["lossy_reuse"].get("bleu", {}).get(agent, 0)

            lines.append(
                f"| {r['name']} | "
                f"{get_val(nr, 'ttft_ms')} | {get_val(fr, 'ttft_ms')} | {get_val(lr, 'ttft_ms')} | "
                f"{get_val(nr, 'total_ms')} | {get_val(fr, 'total_ms')} | {get_val(lr, 'total_ms')} | "
                f"{fr.get('cached_tokens', 0)} | {lr.get('cached_tokens', 0)} | "
                f"{bleu_fr:.3f} | {bleu_lr:.3f} |"
            )

        # Average row
        def avg_vals(key, subkey):
            vals = [
                r[key].get(agent, {}).get(subkey)
                for r in results
                if r[key].get(agent, {}).get(subkey) is not None
            ]
            return sum(vals) / len(vals) if vals else 0

        lines.append(
            f"| **Avg** | "
            f"{avg_vals('no_reuse', 'ttft_ms'):.0f} | {avg_vals('full_reuse', 'ttft_ms'):.0f} | {avg_vals('lossy_reuse', 'ttft_ms'):.0f} | "
            f"{avg_vals('no_reuse', 'total_ms'):.0f} | {avg_vals('full_reuse', 'total_ms'):.0f} | {avg_vals('lossy_reuse', 'total_ms'):.0f} | "
            f"{avg_vals('full_reuse', 'cached_tokens'):.0f} | {avg_vals('lossy_reuse', 'cached_tokens'):.0f} | "
            f"- | - |"
        )
        lines.append("")

    # TTFT speedup
    lines.append("## TTFT Speedup")
    lines.append("")
    lines.append("| Agent | Full-Reuse Speedup | Lossy-Reuse Speedup |")
    lines.append("|---|---|---|")
    for agent in ["a1", "a2", "a3"]:
        no_ttft = avg_vals("no_reuse", "ttft_ms")
        fr_ttft = avg_vals("full_reuse", "ttft_ms")
        lr_ttft = avg_vals("lossy_reuse", "ttft_ms")
        fr_speedup = (no_ttft / fr_ttft - 1) * 100 if fr_ttft > 0 else 0
        lr_speedup = (no_ttft / lr_ttft - 1) * 100 if lr_ttft > 0 else 0
        lines.append(f"| {agent.upper()} | {fr_speedup:+.1f}% | {lr_speedup:+.1f}% |")
    lines.append("")

    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


_BENCH_ARGS = None  # set by pa() at startup; consumed by run_no_reuse etc.


def pa():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--port", type=int, default=PORT)
    args = p.parse_args()
    # Reload tokenizer + recompute token spans to match the chosen model.
    global TOK, _BENCH_ARGS
    TOK = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    _BENCH_ARGS = args
    return args


if __name__ == "__main__":
    asyncio.run(main(pa()))
