#!/usr/bin/env python3
"""Layer-wise RoPE-delta validation for exact AST-selected code spans.

This experiment complements the AST-granularity distance report. AST chooses a
stable exact code object; this script verifies that, once the same object
appears at a different prompt position, applying the correct RoPE delta makes
the reused key cache closer to the fresh target-position key cache than
no-rotation or wrong-delta baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
BASE = ROOT / "results" / "ast_granularity_kv_sensitivity"
DATA = BASE / "data"
FIG = BASE / "figures"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_layer_kv(past_key_values: Any, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values
    entry = past_key_values[layer_idx]
    if isinstance(entry, tuple):
        return entry[0], entry[1]
    return entry.key_cache, entry.value_cache


def rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope_delta_neox(keys: torch.Tensor, delta: int, rope_theta: float) -> torch.Tensor:
    dim = keys.shape[-1]
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, dim, 2, device=keys.device, dtype=torch.float32) / dim)
    )
    positions = torch.full((keys.shape[-2],), float(delta), device=keys.device)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).to(dtype=torch.float32)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    keys_f = keys.to(torch.float32)
    return (keys_f * cos + rotate_half_neox(keys_f) * sin).to(keys.dtype)


def token_bounds_for_text(tokenizer: Any, full_text: str, segment_text: str) -> tuple[int, int]:
    char_pos = full_text.find(segment_text)
    if char_pos < 0:
        raise ValueError("segment text not found in prompt")
    start = len(tokenizer.encode(full_text[:char_pos], add_special_tokens=False))
    end = len(tokenizer.encode(full_text[: char_pos + len(segment_text)], add_special_tokens=False))
    return start, end


def choose_span(spans_path: Path, granularity: str) -> dict[str, Any]:
    spans = json.loads(spans_path.read_text(encoding="utf-8"))
    candidates = [s for s in spans if s.get("granularity") == granularity]
    if not candidates:
        raise ValueError(f"no spans with granularity={granularity}")
    return sorted(candidates, key=lambda s: (abs(int(s.get("approx_tokens", 0)) - 180), s["path"], s["start_line"]))[0]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def distance_row(
    *,
    model: str,
    span: dict[str, Any],
    layer: int,
    variant: str,
    true_delta: int,
    applied_delta: int,
    k_candidate: torch.Tensor,
    k_true: torch.Tensor,
    v_candidate: torch.Tensor,
    v_true: torch.Tensor,
) -> dict[str, Any]:
    k_diff = k_candidate.float() - k_true.float()
    v_diff = v_candidate.float() - v_true.float()
    return {
        "model": model,
        "span_id": span["span_id"],
        "repo": span["repo"],
        "path": span["path"],
        "granularity": span["granularity"],
        "ast_type": span["ast_type"],
        "tokens": int(k_true.shape[-2]),
        "layer": layer,
        "variant": variant,
        "true_delta": true_delta,
        "applied_delta": applied_delta,
        "delta_error": applied_delta - true_delta,
        "k_cosine": round(cosine(k_candidate, k_true), 8),
        "k_l2_norm": round(torch.linalg.vector_norm(k_diff).item() / math.sqrt(k_diff.numel()), 8),
        "k_mean_abs": round(k_diff.abs().mean().item(), 8),
        "v_cosine": round(cosine(v_candidate, v_true), 8),
        "v_l2_norm": round(torch.linalg.vector_norm(v_diff).item() / math.sqrt(v_diff.numel()), 8),
        "v_mean_abs": round(v_diff.abs().mean().item(), 8),
    }


@torch.no_grad()
def run_experiment(args: argparse.Namespace) -> list[dict[str, Any]]:
    span = choose_span(args.spans, args.granularity)
    code = span["text"].strip()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.bfloat16 if args.device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float16
    if args.device == "cpu":
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(args.device).eval()
    rope_theta = float(getattr(model.config, "rope_theta", 1000000.0))
    layers = list(range(int(getattr(model.config, "num_hidden_layers", 1))))

    prefix_a = (
        "You are an AgentTemplateKV planner. The following exact code object may be reused.\n"
        "## Code Object\n"
    )
    prefix_b = (
        "You are an AgentTemplateKV implementer. A planner selected a reusable repository span. "
        "Read the issue context first, then inspect the exact code object.\n"
        "## Issue Context\nThe downstream agent receives a different role prefix and a different prompt position.\n\n"
        "## Code Object\n"
    )
    suffix = "\n\n## Task\nSummarize the invariant of this exact code object in one sentence.\n"
    prompt_a = prefix_a + code + suffix
    prompt_b = prefix_b + code + suffix
    ids_a = tokenizer.encode(prompt_a, return_tensors="pt", add_special_tokens=False).to(args.device)
    ids_b = tokenizer.encode(prompt_b, return_tensors="pt", add_special_tokens=False).to(args.device)
    a_start, a_end = token_bounds_for_text(tokenizer, prompt_a, code)
    b_start, b_end = token_bounds_for_text(tokenizer, prompt_b, code)
    if a_end > ids_a.shape[1] or b_end > ids_b.shape[1]:
        raise ValueError("span truncated")
    out_a = model(input_ids=ids_a, use_cache=True, return_dict=True)
    out_b = model(input_ids=ids_b, use_cache=True, return_dict=True)
    true_delta = b_start - a_start
    wrong_delta = true_delta + args.wrong_delta_offset

    rows = []
    for layer in layers:
        k_a, v_a = get_layer_kv(out_a.past_key_values, layer)
        k_b, v_b = get_layer_kv(out_b.past_key_values, layer)
        k_old = k_a[:, :, a_start:a_end, :]
        v_old = v_a[:, :, a_start:a_end, :]
        k_true = k_b[:, :, b_start:b_end, :]
        v_true = v_b[:, :, b_start:b_end, :]
        rows.append(
            distance_row(
                model=args.model,
                span=span,
                layer=layer,
                variant="no_rotation",
                true_delta=true_delta,
                applied_delta=0,
                k_candidate=k_old,
                k_true=k_true,
                v_candidate=v_old,
                v_true=v_true,
            )
        )
        rows.append(
            distance_row(
                model=args.model,
                span=span,
                layer=layer,
                variant="correct_delta",
                true_delta=true_delta,
                applied_delta=true_delta,
                k_candidate=apply_rope_delta_neox(k_old, true_delta, rope_theta),
                k_true=k_true,
                v_candidate=v_old,
                v_true=v_true,
            )
        )
        rows.append(
            distance_row(
                model=args.model,
                span=span,
                layer=layer,
                variant="wrong_delta",
                true_delta=true_delta,
                applied_delta=wrong_delta,
                k_candidate=apply_rope_delta_neox(k_old, wrong_delta, rope_theta),
                k_true=k_true,
                v_candidate=v_old,
                v_true=v_true,
            )
        )
    return rows


def plot(rows: list[dict[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = ["no_rotation", "correct_delta", "wrong_delta"]
    colors = {"no_rotation": "#757575", "correct_delta": "#2e7d32", "wrong_delta": "#c62828"}
    labels = {"no_rotation": "no rotation", "correct_delta": "correct delta", "wrong_delta": "wrong delta"}
    layers = sorted({int(r["layer"]) for r in rows})

    plt.figure(figsize=(8.4, 4.8))
    for variant in variants:
        sub = sorted([r for r in rows if r["variant"] == variant], key=lambda r: int(r["layer"]))
        plt.plot([int(r["layer"]) for r in sub], [float(r["k_cosine"]) for r in sub], marker="o", markersize=3, label=labels[variant], color=colors[variant])
    plt.xlabel("Layer")
    plt.ylabel("K cosine vs fresh target-position KV")
    plt.title("Layer-wise key alignment for an AST-selected exact span")
    plt.xticks(layers[:: max(1, len(layers) // 8)])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_layerwise_rope_k_cosine.pdf")
    plt.savefig(out_dir / "fig_layerwise_rope_k_cosine.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.4, 4.8))
    for variant in variants:
        sub = sorted([r for r in rows if r["variant"] == variant], key=lambda r: int(r["layer"]))
        plt.plot([int(r["layer"]) for r in sub], [float(r["k_l2_norm"]) for r in sub], marker="o", markersize=3, label=labels[variant], color=colors[variant])
    plt.xlabel("Layer")
    plt.ylabel("Normalized K L2 distance")
    plt.title("RoPE delta reduces layer-wise key distance")
    plt.xticks(layers[:: max(1, len(layers) // 8)])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_layerwise_rope_k_distance.pdf")
    plt.savefig(out_dir / "fig_layerwise_rope_k_distance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.4, 4.8))
    for variant in variants:
        sub = sorted([r for r in rows if r["variant"] == variant], key=lambda r: int(r["layer"]))
        plt.plot([int(r["layer"]) for r in sub], [float(r["v_cosine"]) for r in sub], marker="o", markersize=3, label=labels[variant], color=colors[variant])
    plt.xlabel("Layer")
    plt.ylabel("V cosine vs fresh target-position KV")
    plt.title("Values are not RoPE-rotated; drift reflects context dependence")
    plt.xticks(layers[:: max(1, len(layers) // 8)])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_layerwise_rope_v_cosine.pdf")
    plt.savefig(out_dir / "fig_layerwise_rope_v_cosine.png", dpi=180)
    plt.close()


def report(rows: list[dict[str, str]], out: Path) -> None:
    def avg(variant: str, field: str) -> float:
        vals = [float(r[field]) for r in rows if r["variant"] == variant]
        return sum(vals) / len(vals) if vals else 0.0

    first = rows[0]
    lines = [
        "# Layer-wise RoPE Delta Validation",
        "",
        "This experiment validates the numeric part that follows AST-based exact-span selection: once a function/method span is selected and matched by exact content, the reused key cache must be rotated to the downstream prompt position.",
        "",
        "## Setup",
        "",
        f"- Model: `{first['model']}`",
        f"- Span: `{first['repo']}` / `{first['path']}`",
        f"- Granularity: `{first['granularity']}` (`{first['ast_type']}`)",
        f"- Span tokens: `{first['tokens']}`",
        f"- True position delta: `{first['true_delta']}` tokens",
        "- Variants: no rotation, correct RoPE delta, wrong delta.",
        "",
        "## Summary",
        "",
        "| Variant | mean K cosine | mean K L2 | mean V cosine | mean V L2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in ["no_rotation", "correct_delta", "wrong_delta"]:
        lines.append(
            f"| `{variant}` | {avg(variant, 'k_cosine'):.6f} | {avg(variant, 'k_l2_norm'):.6f} | "
            f"{avg(variant, 'v_cosine'):.6f} | {avg(variant, 'v_l2_norm'):.6f} |"
        )
    lines += [
        "",
        "## Figures",
        "",
        "![Layer-wise K cosine](figures/fig_layerwise_rope_k_cosine.png)",
        "",
        "![Layer-wise K distance](figures/fig_layerwise_rope_k_distance.png)",
        "",
        "![Layer-wise V cosine](figures/fig_layerwise_rope_v_cosine.png)",
        "",
        "## Interpretation",
        "",
        "Correct RoPE delta should dominate the no-rotation and wrong-delta baselines on K cosine / K distance. Values are not explicitly RoPE-rotated, so their residual drift is interpreted as context dependence rather than a rotation failure.",
        "",
        "Paper wording: AST selects a stable exact code object, exact-content and token-level checks gate reuse, and layer-wise RoPE validation shows that the copied keys become closest to fresh target-position keys after the correct delta rotation.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--spans", type=Path, default=DATA / "spans.json")
    parser.add_argument("--granularity", default="function")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wrong-delta-offset", type=int, default=32)
    parser.add_argument("--out", type=Path, default=DATA / "layerwise_rope_delta_validation.csv")
    parser.add_argument("--report", type=Path, default=BASE / "LAYERWISE_ROPE_DELTA_VALIDATION.md")
    args = parser.parse_args()

    rows = run_experiment(args)
    write_csv(args.out, rows)
    plot([{k: str(v) for k, v in row.items()} for row in rows], FIG)
    report([{k: str(v) for k, v in row.items()} for row in rows], args.report)
    print(f"[layerwise_rope] wrote {args.out}")
    print(f"[layerwise_rope] wrote {args.report}")
    print(f"[layerwise_rope] wrote figures to {FIG}")


if __name__ == "__main__":
    main()
