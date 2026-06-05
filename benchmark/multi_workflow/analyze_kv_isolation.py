#!/usr/bin/env python3
"""Standalone KV isolation analysis for code blocks.

Phases:
  1. Cross-block attention ratio (intra vs inter in multi-block prompts)
  2. KV distribution similarity (cosine / CKA / L2 on same-func vs diff-func block pairs)
  3. KV replacement accuracy (PPL / exact match under aggressive KV swap)

No dependency on sglang serving or MAS.  Uses HuggingFace model directly.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Code block definitions
# ---------------------------------------------------------------------------

CODE_BLOCKS: dict[str, dict[str, str]] = {
    "sort": {
        "v1": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
""",
        "v2": """def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
""",
        "v3": """def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
""",
    },
    "search": {
        "v1": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""",
    },
    "dp": {
        "v1": """def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][capacity]
""",
    },
    "graph": {
        "v1": """def bfs(graph, start):
    visited = set()
    queue = [start]
    visited.add(start)
    while queue:
        node = queue.pop(0)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited
""",
    },
    "string": {
        "v1": """def reverse_words(text: str) -> str:
    return " ".join(reversed(text.split()))
""",
    },
    "ds": {
        "v1": """class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
""",
    },
}


SYSTEM_PROMPT = "You are a coding assistant."
INSTRUCTION = "Analyze the code above and explain what each function does."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_model(model_path: str, eager_attn: bool = False) -> tuple:
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    extra = {}
    if eager_attn:
        extra["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        **extra,
    )
    model.cuda()
    model.eval()
    return tok, model


def build_prompt(blocks: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Build a multi-block prompt.  Returns (text, [(start_tok, end_tok), ...])."""
    parts = [
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n",
    ]
    boundaries: list[tuple[int, int]] = []
    tok = None  # tokenizer assigned later

    for b in blocks:
        parts.append("<|im_start|>user\n")
        parts.append(b.strip())
        parts.append("<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{INSTRUCTION}<|im_end|>\n<|im_start|>assistant\n")
    return "".join(parts), boundaries


def _token_boundaries(
    tokenizer, block_texts: list[str]
) -> list[tuple[int, int]]:
    """Build a multi-block prompt and return token boundaries.

    We use a simpler approach: tokenize each block individually,
    then construct the full prompt deterministically and accumulate offsets.
    """
    # Build the prompt the same way as in run_phase1_attention
    parts = [
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n",
    ]
    for b in block_texts:
        parts.append("<|im_start|>user\n")
        parts.append(b.strip())
        parts.append("<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{INSTRUCTION}<|im_end|>\n<|im_start|>assistant\n")
    prompt = "".join(parts)
    full_ids = tokenizer.encode(prompt, add_special_tokens=False)

    boundaries: list[tuple[int, int]] = []
    offset = 0
    for block in block_texts:
        block_stripped = block.strip()
        block_ids = tokenizer.encode(block_stripped, add_special_tokens=False)
        found = _find_sublist(full_ids, block_ids, offset)
        if found >= 0:
            boundaries.append((found, found + len(block_ids)))
            offset = found + len(block_ids)
        else:
            # Fallback: try forward search from start
            found = _find_sublist(full_ids, block_ids, 0)
            if found >= 0:
                boundaries.append((found, found + len(block_ids)))
            else:
                boundaries.append((-1, -1))
    return boundaries


def _find_sublist(haystack: list[int], needle: list[int], start: int = 0) -> int:
    for i in range(start, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def _get_model_layers(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise RuntimeError(f"Unknown layer structure for model {type(model)}")


def _cosine_sim_tensors(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two KV tensors, aligning to min token count."""
    min_tokens = min(a.shape[0], b.shape[0])
    # reshape: (tokens, ...) → (-1, d_last)
    a_f = a[:min_tokens].float().reshape(-1, a.shape[-1])
    b_f = b[:min_tokens].float().reshape(-1, b.shape[-1])
    if a_f.shape[0] != b_f.shape[0]:
        # fallback: pad or trim further
        min_rows = min(a_f.shape[0], b_f.shape[0])
        a_f = a_f[:min_rows]
        b_f = b_f[:min_rows]
    sims = F.cosine_similarity(a_f, b_f, dim=-1)
    return float(sims.mean().item())


def _l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute L2 distance between two KV tensors, aligning to min token count."""
    min_tokens = min(a.shape[0], b.shape[0])
    a_f = a[:min_tokens].float().reshape(-1, a.shape[-1])
    b_f = b[:min_tokens].float().reshape(-1, b.shape[-1])
    if a_f.shape[0] != b_f.shape[0]:
        min_rows = min(a_f.shape[0], b_f.shape[0])
        a_f = a_f[:min_rows]
        b_f = b_f[:min_rows]
    return float(F.pairwise_distance(a_f, b_f).mean().item())


def _cka_score(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Centered Kernel Alignment between two feature matrices. Robust to different token counts."""
    X = X.float().reshape(-1, X.shape[-1])
    Y = Y.float().reshape(-1, Y.shape[-1])
    # Align to min token count
    min_len = min(X.shape[0], Y.shape[0])
    X = X[:min_len]
    Y = Y[:min_len]
    n = X.shape[0]
    H = torch.eye(n, device=X.device) - 1.0 / n
    K = X @ X.T
    L = Y @ Y.T
    hkh = H @ K @ H
    hlh = H @ L @ H
    num = torch.trace(hkh @ hlh)
    denom = torch.sqrt(torch.trace(hkh @ hkh) * torch.trace(hlh @ hlh) + 1e-10)
    return float((num / denom).item())


# ---------------------------------------------------------------------------
# Phase 1: Cross-block attention analysis
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_phase1_attention(
    model_path: str,
    output_dir: Path,
    block_pairs: list[tuple[str, str, str, str]] | None = None,
) -> dict:
    """Run cross-block attention analysis.

    For each pair of blocks A, B:
      - Build prompt = [A] + [B] + instruction
      - Run prefill with output_attentions=True
      - Compute intra_block (A→A, B→B) and inter_block (A→B, B→A) attention
        per layer and per head.
    """
    tok, model = _load_model(model_path, eager_attn=True)
    device = next(model.parameters()).device
    n_layers = _get_model_layers(model)

    if block_pairs is None:
        block_pairs = [
            ("sort", "v1", "sort", "v2"),       # same func, diff impl
            ("sort", "v1", "search", "v1"),      # diff func
            ("sort", "v1", "string", "v1"),       # very diff
        ]

    results: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (cat_a, var_a, cat_b, var_b) in enumerate(block_pairs):
        text_a = CODE_BLOCKS[cat_a][var_a]
        text_b = CODE_BLOCKS[cat_b][var_b]

        prompt, _ = build_prompt([text_a, text_b])
        boundaries = _token_boundaries(tok, [text_a, text_b])
        a_range, b_range = boundaries
        if a_range[0] < 0 or b_range[0] < 0:
            print(f"  WARN: could not find block boundaries for pair {i}")
            continue

        input_ids = tok.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        seq_len = input_ids.shape[1]

        # Forward with output_attentions=True (works with eager attn_implementation)
        outputs = model(input_ids, output_attentions=True)

        attn_weights = outputs.attentions  # list[tuple] of (batch, n_heads, seq, seq)

        layer_stats = []
        for layer_idx, layer_attn in enumerate(attn_weights):
            # layer_attn: (batch=1, n_heads, seq, seq)
            w = layer_attn[0]  # (n_heads, seq, seq)
            n_heads = w.shape[0]
            head_stats = []
            for h in range(n_heads):
                wh = w[h]  # (seq, seq)
                intra_a = _mean_attention(wh, a_range[0], a_range[1], a_range[0], a_range[1])
                intra_b = _mean_attention(wh, b_range[0], b_range[1], b_range[0], b_range[1])
                inter_ab = _mean_attention(wh, b_range[0], b_range[1], a_range[0], a_range[1])
                inter_ba = _mean_attention(wh, a_range[0], a_range[1], b_range[0], b_range[1])
                inter = (inter_ab + inter_ba) / 2 if inter_ab > 0 or inter_ba > 0 else 0
                intra = (intra_a + intra_b) / 2
                ratio = intra / (inter + 1e-10)
                head_stats.append(
                    {
                        "intra": float(intra),
                        "inter": float(inter),
                        "intra_inter_ratio": float(ratio),
                        "intra_a": float(intra_a),
                        "intra_b": float(intra_b),
                        "inter_ab": float(inter_ab),
                        "inter_ba": float(inter_ba),
                    }
                )
            layer_agg = {
                "intra_mean": float(np.mean([h["intra"] for h in head_stats])),
                "inter_mean": float(np.mean([h["inter"] for h in head_stats])),
                "ratio_mean": float(np.mean([h["intra_inter_ratio"] for h in head_stats])),
            }
            layer_stats.append({"layer": layer_idx, "aggregate": layer_agg, "heads": head_stats})

        pair_result = {
            "pair_id": i,
            "cat_a": cat_a,
            "var_a": var_a,
            "cat_b": cat_b,
            "var_b": var_b,
            "a_tokens": a_range[1] - a_range[0],
            "b_tokens": b_range[1] - b_range[0],
            "total_tokens": seq_len,
            "layers": layer_stats,
        }
        results.append(pair_result)
        print(
            f"  [{i}] {cat_a}/{var_a} vs {cat_b}/{var_b}: "
            f"mean_ratio={layer_stats[n_layers // 2]['aggregate']['ratio_mean']:.1f}x "
            f"(intra={layer_stats[n_layers // 2]['aggregate']['intra_mean']:.4f} "
            f"inter={layer_stats[n_layers // 2]['aggregate']['inter_mean']:.4f})"
        )

    summary = {
        "phase": "attention",
        "model": model_path,
        "n_pairs": len(results),
        "pairs": results,
        "overall": _attention_summary_overall(results),
    }
    (output_dir / "phase1_attention.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _mean_attention(
    w: torch.Tensor,
    q_start: int,
    q_end: int,
    k_start: int,
    k_end: int,
) -> float:
    if q_end <= q_start or k_end <= k_start:
        return 0.0
    block = w[q_start:q_end, k_start:k_end]
    return float(block.mean().item())


def _attention_summary_overall(results: list[dict]) -> dict[str, float]:
    same_func = []
    diff_func = []
    for r in results:
        same = r["cat_a"] == r["cat_b"]
        for layer in r["layers"]:
            agg = layer["aggregate"]
            if same:
                same_func.append(agg["ratio_mean"])
            else:
                diff_func.append(agg["ratio_mean"])
    return {
        "same_func_mean_ratio": float(np.mean(same_func)) if same_func else 0,
        "diff_func_mean_ratio": float(np.mean(diff_func)) if diff_func else 0,
    }


# ---------------------------------------------------------------------------
# Phase 2: KV distribution similarity
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_phase2_similarity(
    model_path: str,
    output_dir: Path,
    comparison_pairs: list[tuple[str, str, str, str]] | None = None,
) -> dict:
    """Compute KV distribution similarity between block pairs.

    For each pair:
      - Run prefill on block A alone, capture K, V per layer
      - Run prefill on block B alone, capture K, V per layer
      - Compute cosine / CKA / L2 between KV pairs
    """
    tok, model = _load_model(model_path)
    device = next(model.parameters()).device
    n_layers = _get_model_layers(model)

    if comparison_pairs is None:
        comparison_pairs = [
            ("sort", "v1", "sort", "v2"),       # same func, diff impl
            ("sort", "v1", "sort", "v3"),       # same func, very diff impl (iter → recursive)
            ("sort", "v1", "search", "v1"),      # diff func
            ("sort", "v1", "string", "v1"),       # very diff
        ]

    # Register hooks to capture KV
    k_caches: dict[int, torch.Tensor] = {}
    v_caches: dict[int, torch.Tensor] = {}
    hooks = []

    def _make_kv_hook(layer_idx: int):
        def hook(module, input, output):
            # Qwen2Attention / LlamaAttention output is a tuple (attn_output, attn_weights, past_key_value)
            # We need to intercept the K, V projection *before* they go into attention
            pass
        return hook

    # Better approach: directly run with output_hidden_states and capture via forward
    # Instead, use a simpler approach:
    # 1. Run the model with the block text
    # 2. Capture K,V through a different mechanism

    def _get_kv_for_text(text: str) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Capture per-layer K, V using past_key_values from a prefill forward."""
        prompt = text.strip()
        input_ids = tok.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        outputs = model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values  # DynamicCache or tuple of (K, V) per layer

        kv_by_layer: dict[int, torch.Tensor] = {}
        v_by_layer: dict[int, torch.Tensor] = {}

        for layer_idx, kv_entry in enumerate(past_kv):
            if isinstance(kv_entry, tuple):
                k, v = kv_entry[0], kv_entry[1]
            elif hasattr(kv_entry, 'key_cache') and hasattr(kv_entry, 'value_cache'):
                k = kv_entry.key_cache
                v = kv_entry.value_cache
            else:
                continue
            k = k[0].detach().cpu()
            v = v[0].detach().cpu()
            k_flat = k.permute(1, 0, 2).reshape(k.shape[1], -1)
            v_flat = v.permute(1, 0, 2).reshape(v.shape[1], -1)
            kv_by_layer[layer_idx] = k_flat
            v_by_layer[layer_idx] = v_flat

        return kv_by_layer, v_by_layer

    results: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute all unique block KVs to avoid redundant recomputation
    unique_blocks: set[tuple[str, str]] = set()
    for a_cat, a_var, b_cat, b_var in comparison_pairs:
        unique_blocks.add((a_cat, a_var))
        unique_blocks.add((b_cat, b_var))

    print("  Precomputing KVs for unique blocks...")
    block_kvs: dict[tuple[str, str], tuple[dict, dict]] = {}
    for cat, var in sorted(unique_blocks):
        text = CODE_BLOCKS[cat][var]
        print(f"    {cat}/{var} ({len(tok.encode(text))} tokens)...")
        kv, vv = _get_kv_for_text(text)
        block_kvs[(cat, var)] = (kv, vv)

    for i, (cat_a, var_a, cat_b, var_b) in enumerate(comparison_pairs):
        kv_a, vv_a = block_kvs[(cat_a, var_a)]
        kv_b, vv_b = block_kvs[(cat_b, var_b)]

        layer_results = []
        for l in range(n_layers):
            k_a = kv_a[l]
            k_b = kv_b[l]
            v_a = vv_a[l]
            v_b = vv_b[l]

            layer_results.append(
                {
                    "layer": l,
                    "k_cosine": round(_cosine_sim_tensors(k_a, k_b), 6),
                    "k_l2": round(_l2_distance(k_a, k_b), 6),
                    "v_cosine": round(_cosine_sim_tensors(v_a, v_b), 6),
                    "v_l2": round(_l2_distance(v_a, v_b), 6),
                    "k_cka": round(_cka_score(k_a, k_b), 6),
                    "v_cka": round(_cka_score(v_a, v_b), 6),
                }
            )

        same_func = cat_a == cat_b
        mid_layer = layer_results[n_layers // 2]
        results.append(
            {
                "pair_id": i,
                "cat_a": cat_a,
                "var_a": var_a,
                "cat_b": cat_b,
                "var_b": var_b,
                "same_function": same_func,
                "layers": layer_results,
            }
        )
        print(
            f"  [{i}] {cat_a}/{var_a} vs {cat_b}/{var_b} "
            f"(same_func={same_func}): "
            f"mid K_cos={mid_layer['k_cosine']:.4f} "
            f"V_cos={mid_layer['v_cosine']:.4f} "
            f"K_CKA={mid_layer['k_cka']:.4f}"
        )

    summary = {
        "phase": "similarity",
        "model": model_path,
        "n_pairs": len(results),
        "pairs": results,
        "overall": _similarity_summary_overall(results, n_layers),
    }
    (output_dir / "phase2_similarity.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _similarity_summary_overall(results: list[dict], n_layers: int) -> dict:
    same_kcos, diff_kcos = [], []
    same_vcos, diff_vcos = [], []
    same_kcka, diff_kcka = [], []
    for r in results:
        for layer in r["layers"]:
            if r["same_function"]:
                same_kcos.append(layer["k_cosine"])
                same_vcos.append(layer["v_cosine"])
                same_kcka.append(layer["k_cka"])
            else:
                diff_kcos.append(layer["k_cosine"])
                diff_vcos.append(layer["v_cosine"])
                diff_kcka.append(layer["k_cka"])
    return {
        "same_func_k_cosine_mean": float(np.mean(same_kcos)) if same_kcos else 0,
        "diff_func_k_cosine_mean": float(np.mean(diff_kcos)) if diff_kcos else 0,
        "same_func_v_cosine_mean": float(np.mean(same_vcos)) if same_vcos else 0,
        "diff_func_v_cosine_mean": float(np.mean(diff_vcos)) if diff_vcos else 0,
        "same_func_k_cka_mean": float(np.mean(same_kcka)) if same_kcka else 0,
        "diff_func_k_cka_mean": float(np.mean(diff_kcka)) if diff_kcka else 0,
        "k_cosine_separation": (
            float(np.mean(same_kcos)) - float(np.mean(diff_kcos))
            if same_kcos and diff_kcos
            else 0
        ),
    }


# ---------------------------------------------------------------------------
# Phase 3: KV replacement accuracy
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------


def build_html_report(phase1: dict | None, phase2: dict | None, output_dir: Path) -> str:
    """Generate an interactive HTML report from Phase 1+2 results."""
    sections = []

    if phase1 and phase1.get("pairs"):
        sections.append(_phase1_html(phase1))
    elif phase1:
        sections.append("<h2>Phase 1: No results (check boundary detection)</h2>")
    if phase2 and phase2.get("pairs"):
        sections.append(_phase2_html(phase2))
    elif phase2:
        sections.append("<h2>Phase 2: No results</h2>")

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Code-Block KV Isolation Analysis</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 24px; background: #fafafa; color: #222; }}
h1 {{ font-size: 22px; }}
h2 {{ font-size: 17px; color: #333; margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
.chart {{ margin: 16px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 12px 0; }}
th {{ background: #f0f0f0; padding: 6px 10px; text-align: left; font-weight: 600; border-bottom: 2px solid #ddd; }}
td {{ padding: 6px 10px; border-bottom: 1px solid #eee; }}
.good {{ color: #2e7d32; font-weight: 600; }}
.bad {{ color: #b71c1c; font-weight: 600; }}
.neutral {{ color: #555; }}
.bar-container {{ display: flex; align-items: center; gap: 8px; }}
.bar {{ height: 20px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>Code-Block KV Isolation Analysis</h1>
<p>模型: {phase1.get('model', '') if phase1 else (phase2.get('model', '') if phase2 else 'N/A')}</p>
{"".join(sections)}
</body>
</html>"""


def _phase1_html(data: dict) -> str:
    rows = []
    for pair in data.get("pairs", []):
        same = pair["cat_a"] == pair["cat_b"]
        mid = pair["layers"][len(pair["layers"]) // 2]["aggregate"]
        label = "✓ 同功能" if same else "✗ 不同功能"
        rows.append(
            f"<tr>"
            f"<td>{pair['cat_a']}/{pair['var_a']} vs {pair['cat_b']}/{pair['var_b']}</td>"
            f"<td>{label}</td>"
            f"<td>{mid['intra_mean']:.4f}</td>"
            f"<td>{mid['inter_mean']:.4f}</td>"
            f"<td class='{'good' if mid['ratio_mean'] > 3 else 'neutral'}'>{mid['ratio_mean']:.1f}x</td>"
            f"</tr>"
        )

    overall = data.get("overall", {})
    return f"""
<h2>Phase 1: Cross-Block Attention Ratio</h2>
<p>度量每层每个 head 的块内 attention / 块间 attention 比值。比值越大=代码块越独立。</p>
<p>汇总: 同功能 mean ratio = <b>{overall.get('same_func_mean_ratio', 0):.1f}x</b> |
   不同功能 mean ratio = <b>{overall.get('diff_func_mean_ratio', 0):.1f}x</b></p>
<table>
<thead><tr><th>Block Pair</th><th>类型</th><th>Intra (块内)</th><th>Inter (块间)</th><th>Ratio</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
"""


def _phase2_html(data: dict) -> str:
    rows = []
    for pair in data.get("pairs", []):
        same = pair["same_function"]
        mid = pair["layers"][len(pair["layers"]) // 2]
        label = "✓ 同功能" if same else "✗ 不同功能"
        kc = mid["k_cosine"]
        vc = mid["v_cosine"]
        rows.append(
            f"<tr>"
            f"<td>{pair['cat_a']}/{pair['var_a']} vs {pair['cat_b']}/{pair['var_b']}</td>"
            f"<td>{label}</td>"
            f"<td class='{'good' if kc > 0.9 else 'neutral'}'>{kc:.4f}</td>"
            f"<td class='{'good' if vc > 0.9 else 'neutral'}'>{vc:.4f}</td>"
            f"<td>{mid['k_cka']:.4f}</td>"
            f"<td>{mid['v_cka']:.4f}</td>"
            f"</tr>"
        )

    overall = data.get("overall", {})
    separation = overall.get("k_cosine_separation", 0)
    return f"""
<h2>Phase 2: KV Distribution Similarity</h2>
<p>计算两个代码块 prefill 后每层 K/V 的 cosine similarity、CKA 和 L2 distance。
   cosine 越接近 1 = KV 分布越相似，越适合复用。</p>
<p>汇总: 同功能 K_cos = <b>{overall.get('same_func_k_cosine_mean', 0):.4f}</b> |
   不同功能 K_cos = <b>{overall.get('diff_func_k_cosine_mean', 0):.4f}</b> |
   分离度 = <b class='{'good' if separation > 0.05 else 'neutral'}'>{separation:.4f}</b></p>
<table>
<thead><tr><th>Block Pair</th><th>类型</th><th>K cosine</th><th>V cosine</th><th>K CKA</th><th>V CKA</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Code-block KV isolation analysis")
    p.add_argument("--mode", default="attention",
                   choices=["attention", "similarity", "replacement", "full"])
    p.add_argument("--model-path", default="/home/gfy/models/Qwen2.5-3B-Instruct")
    p.add_argument("--output-dir", default="/tmp/kv_isolation")
    return p.parse_args()


def main():
    args = parse_args()
    mode = args.mode
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Mode: {mode}, Model: {args.model_path}, Output: {output_dir}")

    phase1 = None
    phase2 = None

    if mode in ("attention", "full"):
        print("\n=== Phase 1: Cross-Block Attention Analysis ===")
        t0 = time.perf_counter()
        phase1 = run_phase1_attention(args.model_path, output_dir)
        print(f"Phase 1 done in {time.perf_counter() - t0:.1f}s")

    if mode in ("similarity", "full"):
        print("\n=== Phase 2: KV Distribution Similarity ===")
        t0 = time.perf_counter()
        phase2 = run_phase2_similarity(args.model_path, output_dir)
        print(f"Phase 2 done in {time.perf_counter() - t0:.1f}s")

    # Build HTML report
    html = build_html_report(phase1, phase2, output_dir)
    report_path = output_dir / "kv_isolation_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
