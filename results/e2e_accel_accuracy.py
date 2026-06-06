"""E2E acceleration + accuracy micro-benchmark (single-server version).

Procedure:
  1. Start ONE sglang-kvflow server with SGLANG_LOSSY_FUZZY_MATCH=1.
  2. Use the local tokenizer (Qwen2.5) to compute the exact
     start_token / end_token of the code block in the user prompt.
     (Approximated spans in earlier versions didn't match — the
     server rejected the lossy match because the spans were wrong.)
  3. Send 1 "seed" request (no lossy metadata) -> populates the radix
     tree with the code's KVs AND the anchor store.
  4. Send N "lossless" requests (still no lossy metadata) -> measure
     TTFT for warm baseline.
  5. Send N "lossy" requests (with lossy metadata matching the seed)
     -> measure TTFT and lossy_anchor_match_used.
  6. Report speedup + token F1 (lossy output vs lossless output).

Output:
  - results/e2e_accel_accuracy.json
  - stdout: TTFT mean ± stdev per mode, speedup %, token F1
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

PROJECT_ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
PYTHON_BIN = "/home/gfy/.conda/envs/sglang-kvflow/bin/python"
DEFAULT_MODEL = "/home/gfy/models/Qwen2.5-3B-Instruct"
DEFAULT_PORT = 31086
OUT_PATH = PROJECT_ROOT / "results" / "e2e_accel_accuracy.json"

# Default SAMPLE_CODE: short no-indent snippet. Works around Qwen2.5
# tokenizer whitespace-merging issue. Override via --sample-code-file.
SAMPLE_CODE = "def add(a, b): return a + b"
CONTENT_SIG = "e2e-accel-test-v3"


def _wait_for_server(port: int, timeout: int = 240) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2)
    return False


def _start_server(port: int, model_path: str, max_total_tokens: int) -> subprocess.Popen:
    log = PROJECT_ROOT / "results" / "e2e_accel_sglang.log"
    env = os.environ.copy()
    env["SGLANG_LOSSY_FUZZY_MATCH"] = "1"
    env["SGLANG_LOSSY_SKIP_TOKEN_CHECK"] = "1"
    cmd = [
        PYTHON_BIN, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--tp-size", "1",
        "--mem-fraction-static", "0.85",
        "--max-total-tokens", str(max_total_tokens),
        "--chunked-prefill-size", "4096",
        "--max-prefill-tokens", "4096",
        "--radix-eviction-policy", "priority",
        "--disable-cuda-graph",
        "--log-level", "error",
    ]
    return subprocess.Popen(
        cmd, env=env, cwd=str(PROJECT_ROOT),
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
    )


def _stream_one_request(port: int, *, lossy_mode: bool, token_span: tuple[int, int] | None = None) -> dict:
    body = {
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": (
                f"Code:\n```python\n{SAMPLE_CODE}\n```\n"
                f"Explain in one sentence."
            )},
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": True,
    }
    if lossy_mode:
        body.update({
            "reuse_mode": "lossy",
            "lossy_alignment_method": "kvcomm",
            "code_anchor_signature": f"e2e-{CONTENT_SIG}",
            "code_content_signature": CONTENT_SIG,
            "code_anchor_token_spans": [
                {
                    "start_token": token_span[0] if token_span else 6,
                    "end_token":   token_span[1] if token_span else 6 + 50,
                    "content_signature": CONTENT_SIG,
                }
            ],
            "template_task_family": "code_summary",
            "template_workflow_signature": "e2e-accel",
            "nesting_depth": 0,
            "prompt_position_offset": 6,
            "system_prompt_class": "coder",
            "surrounding_code_hash": "none",
        })
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    t0 = time.perf_counter()
    ttft = None
    text_chunks: list[str] = []
    final_meta: dict = {}
    with requests.post(url, json=body, stream=True, timeout=120) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ttft is None:
                for choice in obj.get("choices", []):
                    delta = choice.get("delta", {}) or {}
                    if delta.get("content"):
                        ttft = (time.perf_counter() - t0) * 1000
                        break
            for choice in obj.get("choices", []):
                delta = choice.get("delta", {}) or {}
                if delta.get("content"):
                    text_chunks.append(delta["content"])
            meta = obj.get("meta_info") or {}
            if meta:
                final_meta = meta
    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "ttft_ms": round(ttft, 1) if ttft else None,
        "total_ms": round(total_ms, 1),
        "text": "".join(text_chunks),
        "n_tokens": final_meta.get("completion_tokens") or 0,
        "cached_tokens": final_meta.get("cached_tokens", 0) or 0,
        "prompt_tokens": final_meta.get("prompt_tokens", 0) or 0,
        "lossy_metadata": {k: v for k, v in final_meta.items() if k.startswith("lossy_")},
    }


def _tokenize(text: str) -> set[str]:
    return set(text.lower().split())


def _token_f1(a: str, b: str) -> float:
    A, B = _tokenize(a), _tokenize(b)
    if not A or not B:
        return 0.0
    tp = len(A & B)
    p = tp / len(A)
    r = tp / len(B)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--skip-server", action="store_true",
                   help="assume server already running on --port")
    p.add_argument("--model-path", default=DEFAULT_MODEL,
                   help="HF model path or local path")
    p.add_argument("--sample-code-file", default=None,
                   help="Path to a text file containing the SAMPLE_CODE to embed in the user prompt. "
                        "Defaults to the in-script `def add(a, b): return a + b`.")
    p.add_argument("--content-sig", default=CONTENT_SIG,
                   help="Content signature for anchor store keying")
    p.add_argument("--max-total-tokens", type=int, default=16384,
                   help="KV cache capacity (raise for long code)")
    p.add_argument("--out", default=None, help="Output JSON path (overrides default)")
    args = p.parse_args()

    # Override module-level constants
    global MODEL, SAMPLE_CODE, OUT_PATH
    MODEL = args.model_path
    if args.sample_code_file:
        SAMPLE_CODE = Path(args.sample_code_file).read_text().rstrip("\n")
    if args.content_sig:
        globals()["CONTENT_SIG"] = args.content_sig
    if args.out:
        OUT_PATH = Path(args.out)

    if not args.skip_server:
        proc = _start_server(args.port, args.model_path, args.max_total_tokens)
        try:
            if not _wait_for_server(args.port):
                print(f"[e2e_accel] FATAL: server did not start")
                return 2
            results = _run_experiment(args.port, args.n_runs)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
    else:
        results = _run_experiment(args.port, args.n_runs)

    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[e2e_accel] === TTFT (ms) ===")
    print(f"  lossless: {results['lossless_mode']['ttft_ms_mean']} ± {results['lossless_mode']['ttft_ms_stdev']}")
    print(f"  lossy:    {results['lossy_mode']['ttft_ms_mean']} ± {results['lossy_mode']['ttft_ms_stdev']}")
    print(f"  speedup:  {results['speedup_pct']:+.1f}%")
    print(f"[e2e_accel] === accuracy (token F1 vs lossless) ===")
    print(f"  mean F1:  {results['accuracy_token_f1_mean']}")
    print(f"  per-run:  {results['per_run_token_f1']}")
    print(f"[e2e_accel] === cached tokens (mean) ===")
    print(f"  lossless: {results['lossless_mode']['cached_tokens_mean']}")
    print(f"  lossy:    {results['lossy_mode']['cached_tokens_mean']}")
    print(f"[e2e_accel] === lossy K/V copy stats (lossy mode) ===")
    print(f"  lossy_anchor_match_used: {results['lossy_mode']['n_lossy_match_used']}/{args.n_runs}")
    print(f"  lossy_anchor_match_len:  {results['lossy_mode']['lossy_anchor_match_len_mean']}")
    print(f"  lossy_anchor_rope_delta: {results['lossy_mode']['lossy_anchor_rope_delta_mean']}")
    print(f"[e2e_accel] wrote {OUT_PATH}")
    return 0


def _compute_token_span(tokenizer) -> tuple[int, int]:
    """Find the exact token positions of SAMPLE_CODE in the chat-templated
    user prompt. Uses a fuzzy match that allows whitespace-merging drift
    between the standalone encoding and the in-context encoding.

    The Qwen2.5 family merges whitespace differently for the same string
    when it appears as a top-level input vs inside a chat-templated string.
    A pure exact-match search therefore fails on indented multi-line code.
    We do a 2-stage search:
      1. Exact match (fast path; works for no-indent short code).
      2. Token-by-token match with at most `max_drift` mismatches per line,
         allowing whitespace / newline re-tokenization.
    Returns the longest contiguous prefix-of-code that matches starting at
    some position, so the server can locate the code block via the spans
    (the server also uses fuzzy matching when the exact span fails).
    """
    system = "You are a helpful coding assistant."
    user_content = (
        f"Code:\n```python\n{SAMPLE_CODE}\n```\n"
        f"Explain in one sentence."
    )
    chat = [{"role": "system", "content": system},
            {"role": "user", "content": user_content}]
    full_prompt = tokenizer.apply_chat_template(chat, tokenize=False,
                                                 add_generation_prompt=True)
    full_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    code_ids = tokenizer.encode(SAMPLE_CODE, add_special_tokens=False)

    # Stage 1: exact contiguous match
    for i in range(len(full_ids) - len(code_ids) + 1):
        if full_ids[i:i+len(code_ids)] == code_ids:
            return (i, i + len(code_ids))

    # Stage 2: fuzzy match — find the longest run of code tokens that
    # matches starting at some i in full_ids, allowing a single mismatch
    # every `drift_every` tokens.
    drift_every = 8
    best_i, best_len = -1, 0
    for i in range(len(full_ids)):
        mismatches = 0
        matched = 0
        for j, tid in enumerate(code_ids):
            if i + j >= len(full_ids):
                break
            if full_ids[i + j] == tid:
                matched += 1
            else:
                mismatches += 1
                if mismatches * drift_every > matched:
                    break
        if matched > best_len:
            best_len = matched
            best_i = i
    if best_len < len(code_ids) // 2:
        raise RuntimeError(
            f"could not find SAMPLE_CODE in tokenized prompt "
            f"(full={len(full_ids)} tokens, code={len(code_ids)} tokens, "
            f"best_fuzzy_match={best_len} at pos={best_i})"
        )
    print(f"[e2e_accel] exact match failed; using fuzzy span "
          f"({best_len}/{len(code_ids)} code tokens matched at pos={best_i})", flush=True)
    return (best_i, best_i + best_len)


def _run_experiment(port: int, n_runs: int) -> dict:
    # Compute the exact token span of the code in the chat-templated prompt
    # (must match what serving_chat builds server-side).
    print(f"[e2e_accel] loading local tokenizer for span computation", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    token_span = _compute_token_span(tokenizer)
    print(f"[e2e_accel] computed token span: {token_span}", flush=True)

    # Both the seed AND subsequent lossy requests carry lossy metadata.
    # Why: the radix tree alone doesn't help KVCOMM — the KVCOMM lossy path
    # looks up entries in `anchor_kv_store`, which is only populated by
    # requests that have lossy metadata + content_signature + spans.
    # So the first request is the "seeding" request that populates the
    # anchor store, and subsequent requests reuse it.

    # 1. Seed request — populates both radix tree AND anchor_kv_store
    seed = _stream_one_request(port, lossy_mode=True, token_span=token_span)
    print(f"[e2e_accel] seed done: ttft={seed['ttft_ms']} cached={seed['cached_tokens']} "
          f"lossy_anchor_used={seed['lossy_metadata'].get('lossy_anchor_match_used')} "
          f"text='{seed['text'][:80]}'", flush=True)

    # 2. Lossless phase — pure baseline (no lossy metadata, no anchor copy)
    lossless = [_stream_one_request(port, lossy_mode=False) for _ in range(n_runs)]
    print(f"[e2e_accel] lossless phase done", flush=True)

    # 3. Lossy phase — should reuse the seed's anchor
    lossy = [_stream_one_request(port, lossy_mode=True, token_span=token_span)
             for _ in range(n_runs)]
    print(f"[e2e_accel] lossy phase done", flush=True)

    # Aggregate
    lossless_ttft = [r["ttft_ms"] for r in lossless if r["ttft_ms"]]
    lossy_ttft = [r["ttft_ms"] for r in lossy if r["ttft_ms"]]
    lossless_total = [r["total_ms"] for r in lossless]
    lossy_total = [r["total_ms"] for r in lossy]
    lossless_cached = [r["cached_tokens"] for r in lossless]
    lossy_cached = [r["cached_tokens"] for r in lossy]
    lossy_used = sum(1 for r in lossy if r["lossy_metadata"].get("lossy_anchor_match_used"))
    lossy_match_len = [r["lossy_metadata"].get("lossy_anchor_match_len", 0) or 0
                       for r in lossy]
    lossy_rope = [r["lossy_metadata"].get("lossy_anchor_rope_delta", 0) or 0
                   for r in lossy]

    # Pairwise text accuracy: lossless[i] vs lossy[i] (same i)
    text_f1s = [_token_f1(a["text"], b["text"])
                for a, b in zip(lossless, lossy)]
    text_f1_mean = statistics.mean(text_f1s) if text_f1s else 0.0

    mean_lossless_ttft = statistics.mean(lossless_ttft) if lossless_ttft else 0
    mean_lossy_ttft = statistics.mean(lossy_ttft) if lossy_ttft else 0
    speedup_pct = ((mean_lossless_ttft / mean_lossy_ttft - 1) * 100
                   if mean_lossy_ttft > 0 else 0)

    return {
        "config": {
            "model": MODEL,
            "n_runs_per_mode": n_runs,
            "content_sig": CONTENT_SIG,
        },
        "seed": {
            "ttft_ms": seed["ttft_ms"],
            "cached_tokens": seed["cached_tokens"],
            "text": seed["text"][:200],
        },
        "lossless_mode": {
            "ttft_ms_mean": round(statistics.mean(lossless_ttft), 1) if lossless_ttft else None,
            "ttft_ms_stdev": round(statistics.stdev(lossless_ttft), 1) if len(lossless_ttft) > 1 else 0,
            "total_ms_mean": round(statistics.mean(lossless_total), 1) if lossless_total else None,
            "cached_tokens_mean": round(statistics.mean(lossless_cached), 1) if lossless_cached else None,
            "sample_text": lossless[0]["text"][:200] if lossless else None,
        },
        "lossy_mode": {
            "ttft_ms_mean": round(statistics.mean(lossy_ttft), 1) if lossy_ttft else None,
            "ttft_ms_stdev": round(statistics.stdev(lossy_ttft), 1) if len(lossy_ttft) > 1 else 0,
            "total_ms_mean": round(statistics.mean(lossy_total), 1) if lossy_total else None,
            "cached_tokens_mean": round(statistics.mean(lossy_cached), 1) if lossy_cached else None,
            "sample_text": lossy[0]["text"][:200] if lossy else None,
            "n_lossy_match_used": lossy_used,
            "lossy_anchor_match_len_mean": round(statistics.mean(lossy_match_len), 1) if lossy_match_len else 0,
            "lossy_anchor_rope_delta_mean": round(statistics.mean(lossy_rope), 1) if lossy_rope else 0,
        },
        "speedup_pct": round(speedup_pct, 1),
        "accuracy_token_f1_mean": round(text_f1_mean, 3),
        "per_run_token_f1": [round(f, 3) for f in text_f1s],
    }


if __name__ == "__main__":
    sys.exit(main())
