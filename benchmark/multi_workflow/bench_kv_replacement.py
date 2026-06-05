#!/usr/bin/env python3
"""Phase 3: KV tensor replacement — measuring acceleration and accuracy.

Concept: prefill code_A's KV → replace code_B's block KV with code_A's →
         measure: generation quality and prefill time saved.

Standalone HuggingFace — no sglang dependency.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL_PATH = "/home/gfy/models/Qwen2.5-3B-Instruct"

CODE_BLOCKS = {
    "bubble_sort": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr""",
    "quick_sort": """def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)""",
    "selection_sort": """def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr""",
    "insertion_sort": """def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr""",
    "binary_search": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
    "linear_search": """def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1""",
    "fibonacci_rec": """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)""",
    "fibonacci_iter": """def fibonacci_iter(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b""",
    "factorial_rec": """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)""",
    "factorial_iter": """def factorial_iter(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result""",
    "reverse_words": """def reverse_words(text):
    return " ".join(reversed(text.split()))""",
    "is_palindrome": """def is_palindrome(s):
    return s == s[::-1]""",
    "count_primes": """def count_primes(n):
    count = 0
    for num in range(2, n):
        is_prime = True
        for div in range(2, int(num**0.5) + 1):
            if num % div == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    return count""",
}

SYSTEM = "You are a Python coding assistant. Write only code, no explanation."
INSTRUCTION = "Explain what this code does in one short sentence."

EXPERIMENTS = [
    # (warmup, eval, desc, expected_behavior)
    ("bubble_sort", "selection_sort", "同功能不同算法 (bubble→selection)", "span_overlap_high"),
    ("bubble_sort", "insertion_sort", "同功能不同算法 (bubble→insertion)", "span_overlap_medium"),
    ("bubble_sort", "quick_sort", "同功能不同算法 (bubble→quick)", "span_overlap_medium"),
    ("factorial_rec", "factorial_iter", "同功能递归→迭代", "span_overlap_high"),
    ("fibonacci_rec", "fibonacci_iter", "同功能递归→迭代(fib)", "span_overlap_medium"),
    ("binary_search", "linear_search", "search: 二分→线性", "span_overlap_medium"),
    ("bubble_sort", "binary_search", "不同功能(sort→search)", "no_anchor_overlap"),
    ("bubble_sort", "fibonacci_rec", "完全不相关", "no_anchor_overlap"),
    ("count_primes", "is_palindrome", "完全不相关(prime→palindrome)", "no_anchor_overlap"),
]


def load_model(path=MODEL_PATH):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    model.cuda().eval()
    return tok, model


def build_prompt(code_text):
    return f"""<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{code_text}\n\n{INSTRUCTION}<|im_end|>\n<|im_start|>assistant\n"""


@torch.no_grad()
def prefetch_kv(model, tokenizer, text, device):
    """Run prefill and return past_key_values (DynamicCache)."""
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
    out = model(input_ids, use_cache=True)
    return out.past_key_values, input_ids.shape[1]


def find_code_boundaries(tokenizer, full_prompt, code_text):
    """Find token start/end positions of code_text within full_prompt."""
    full_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    code_ids = tokenizer.encode(code_text.strip(), add_special_tokens=False)
    for i in range(len(full_ids) - len(code_ids) + 1):
        if full_ids[i : i + len(code_ids)] == code_ids:
            return i, i + len(code_ids)
    return -1, -1


def merge_kv_caches(kv_warmup, kv_eval, replace_start, replace_end, total_len):
    """Replace code block KV in kv_eval with kv_warmup's KV.

    Returns a DynamicCache (same type as HuggingFace expects)."""
    from transformers.cache_utils import DynamicCache
    
    merged = DynamicCache()
    for layer, (w_entry, e_entry) in enumerate(zip(kv_warmup, kv_eval)):
        if isinstance(w_entry, tuple):
            kw, vw = w_entry[0], w_entry[1]
        else:
            kw, vw = w_entry.key_cache, w_entry.value_cache

        if isinstance(e_entry, tuple):
            ke, ve = e_entry[0], e_entry[1]
        else:
            ke, ve = e_entry.key_cache, v_entry.value_cache

        warmup_tokens = kw.shape[2]
        code_block_len = replace_end - replace_start
        min_len = min(warmup_tokens, code_block_len)

        ke_merged = ke.clone()
        ve_merged = ve.clone()
        ke_merged[:, :, replace_start : replace_start + min_len, :] = kw[:, :, :min_len, :]
        ve_merged[:, :, replace_start : replace_start + min_len, :] = vw[:, :, :min_len, :]

        merged.update(ke_merged, ve_merged, layer)

    return merged


@torch.no_grad()
def generate_from_kv(model, tokenizer, kv_cache, device, last_token_id=None, max_new_tokens=80):
    """Manual decode from KV cache."""
    tok_id = last_token_id or tokenizer.bos_token_id or 1
    input_ids = torch.tensor([[tok_id]], device=device)
    if not isinstance(kv_cache, DynamicCache):
        from transformers.cache_utils import DynamicCache as DC
        kv = DC()
        for layer, (k, v) in enumerate(kv_cache):
            kv.update(k, v, layer)
        past = kv
    else:
        past = kv_cache
    generated = []
    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        nxt = torch.argmax(logits, dim=-1).item()
        if nxt == tokenizer.eos_token_id:
            break
        generated.append(nxt)
        past = out.past_key_values
        input_ids = torch.tensor([[nxt]], device=device)
    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def generate_from_kv_with_full_input(model, tokenizer, full_input_ids, kv_cache, device, max_new_tokens=80):
    """Alternative: provide full input + past KV to skip computation."""
    # The model should use past_key_values and skip the input prefill
    # But some HF versions still run the full forward...
    # Just use the direct approach above.
    return generate_from_kv(model, tokenizer, kv_cache, device, max_new_tokens)


def bleu(ref, hyp):
    try:
        from nltk.translate.bleu_score import sentence_bleu
        return float(sentence_bleu([ref.split()], hyp.split(), weights=(.25,.25,.25,.25)))
    except:
        rs, hs = set(ref.split()), set(hyp.split())
        return len(rs & hs) / len(rs) if rs else (1.0 if not hyp else 0.0)


def run_experiment(warmup_key, eval_key, desc, tok, model, device):
    warmup_code = CODE_BLOCKS[warmup_key]
    eval_code = CODE_BLOCKS[eval_key]

    warmup_text = warmup_code.strip()
    eval_full = build_prompt(eval_code)

    # 1. Prefill warmup code → kv_warmup
    t0 = time.perf_counter()
    kv_warmup, n_warmup = prefetch_kv(model, tok, warmup_text, device)
    warmup_ms = (time.perf_counter() - t0) * 1000

    # 2. Prefill full eval prompt → kv_eval
    t0 = time.perf_counter()
    kv_eval, n_eval = prefetch_kv(model, tok, eval_full, device)
    eval_ms = (time.perf_counter() - t0) * 1000

    # Tokenize eval prompt for boundary detection and last_token
    eval_ids = tok.encode(eval_full, return_tensors="pt", add_special_tokens=False).to(device)
    last_tok = eval_ids[0, -1].item()

    # 3. Find code block boundaries
    start, end = find_code_boundaries(tok, eval_full, eval_code)

    # 4. Baseline generation
    t0 = time.perf_counter()
    baseline_out = generate_from_kv(model, tok, kv_eval, device, last_token_id=last_tok)
    baseline_ms = (time.perf_counter() - t0) * 1000

    # 5. Hybrid (merged KV) — only if boundary found
    bleu_val = 0.0
    hybrid_out = ""
    saved_prefill_est = 0.0
    if start >= 0:
        from transformers.cache_utils import DynamicCache as DC
        kv_hybrid = DC()
        for layer, (w_entry, e_entry) in enumerate(zip(kv_warmup, kv_eval)):
            kw, vw = (w_entry[0], w_entry[1]) if isinstance(w_entry, tuple) else (None, None)
            ek, ev = (e_entry[0], e_entry[1]) if isinstance(e_entry, tuple) else (None, None)
            if kw is None: continue

            ekc = ek.clone()
            evc = ev.clone()
            min_len = min(kw.shape[2], (end - start))
            ekc[:, :, start : start + min_len, :] = kw[:, :, :min_len, :]
            evc[:, :, start : start + min_len, :] = vw[:, :, :min_len, :]
            kv_hybrid.update(ekc, evc, layer)
        t0 = time.perf_counter()
        hybrid_out = generate_from_kv(model, tok, kv_hybrid, device, last_token_id=last_tok)
        hybrid_ms = (time.perf_counter() - t0) * 1000
        bleu_val = round(bleu(baseline_out, hybrid_out), 4)
        code_block_tokens = end - start
        token_ms = eval_ms / n_eval if n_eval > 0 else 0
        saved_prefill_est = code_block_tokens * token_ms
    else:
        hybrid_ms = 0
        code_block_tokens = 0

    return {
        "warmup": warmup_key,
        "eval": eval_key,
        "desc": desc,
        "warmup_tokens": n_warmup,
        "eval_tokens": n_eval,
        "code_block_tokens": code_block_tokens,
        "warmup_prefill_ms": round(warmup_ms, 1),
        "eval_prefill_ms": round(eval_ms, 1),
        "saved_prefill_ms": round(saved_prefill_est, 1),
        "baseline_output": baseline_out.strip()[:200],
        "hybrid_output": hybrid_out.strip()[:200],
        "bleu": bleu_val,
        "baseline_gen_ms": round(baseline_ms, 1),
        "hybrid_gen_ms": round(hybrid_ms, 1),
    }


def main():
    args = argparse.Namespace()
    args.model = MODEL_PATH

    print("Loading model...")
    tok, model = load_model(args.model)
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    results = []
    for warmup_key, eval_key, desc, exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"{desc}: {warmup_key} → {eval_key}")
        r = run_experiment(warmup_key, eval_key, desc, tok, model, device)
        results.append(r)

        print(f"  warmup prefill: {r['warmup_prefill_ms']:.0f}ms ({r['warmup_tokens']} tokens)")
        print(f"  eval prefill:   {r['eval_prefill_ms']:.0f}ms ({r['eval_tokens']} tokens)")
        print(f"  saved prefill:  {r['saved_prefill_ms']:.0f}ms ({r['code_block_tokens']} tokens in block)")
        print(f"  BLEU (hybrid vs baseline): {r['bleu']:.4f}")
        print(f"  baseline: {r['baseline_output'][:100]}")
        print(f"  hybrid:   {r['hybrid_output'][:100]}")

    # Summary
    out_dir = Path(__file__).resolve().parents[3] / "sglang-kvflow" / "results" / "kv_replacement"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    lines = ["# KV Tensor Replacement — Acceleration & Accuracy",
             "",
             "| warmup | eval | desc | warmup_ms | eval_ms | saved_ms | block_tok | BLEU |",
             "|---|---|---:|---:|---:|---:|---:|"]

    for r in results:
        lines.append(f"| {r['warmup']} | {r['eval']} | {r['desc']} | "
                     f"{r['warmup_prefill_ms']:.0f} | {r['eval_prefill_ms']:.0f} | "
                     f"{r['saved_prefill_ms']:.0f} | {r['code_block_tokens']} | "
                     f"{r['bleu']:.4f} |")

    same_func = [r for r in results if r['warmup'].split('_')[-1] == r['eval'].split('_')[-1] or 
                 (r['warmup'] in ('bubble_sort','quick_sort','selection_sort','insertion_sort') and
                  r['eval'] in ('bubble_sort','quick_sort','selection_sort','insertion_sort'))]
    diff_func = [r for r in results if not (r['warmup'].split('_')[-1] == r['eval'].split('_')[-1] or 
                 (r['warmup'] in ('bubble_sort','quick_sort','selection_sort','insertion_sort') and
                  r['eval'] in ('bubble_sort','quick_sort','selection_sort','insertion_sort')))]

    if same_func:
        avg_bleu = sum(r['bleu'] for r in same_func) / len(same_func)
        avg_save = sum(r['saved_prefill_ms'] for r in same_func) / len(same_func)
        lines += ["", f"- Same-func avg BLEU: {avg_bleu:.4f}, avg saved: {avg_save:.0f}ms"]

    if diff_func:
        avg_bleu = sum(r['bleu'] for r in diff_func) / len(diff_func)
        avg_save = sum(r['saved_prefill_ms'] for r in diff_func) / len(diff_func)
        lines += [f"- Diff-func avg BLEU: {avg_bleu:.4f}, avg saved: {avg_save:.0f}ms"]

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\n\nReport: {out_dir}/")


if __name__ == "__main__":
    main()
