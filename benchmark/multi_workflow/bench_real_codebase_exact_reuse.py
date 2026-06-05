#!/usr/bin/env python3
"""Real codebase exact-segment KV reuse experiment.

This benchmark matches the contribution-3 contract:
AST/anchor metadata only locates codebase segments, while reuse is gated by
exact code-content signatures and token equality.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT = Path(__file__).resolve().parents[2]
MAS_SRC = PROJECT.parent / "MAScoder" / "src"
for entry in (str(MAS_SRC), str(PROJECT), str(PROJECT / "python")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mascoder.code_anchor import build_code_anchor_payload, compute_exact_content_signature
from benchmark.multi_workflow.large_codebase import (
    DATA_STRUCTURES,
    GRAPH_ALGORITHMS,
    SORTING,
)


DEFAULT_MODEL = "/home/gfy/models/Qwen2.5-3B-Instruct"
DEFAULT_PY = "/home/gfy/.conda/envs/sglang-kvflow/bin/python"
OUT_DIR = PROJECT / "results" / "real_codebase_exact_reuse"


@dataclass
class CodeSegment:
    name: str
    text: str

    @property
    def signature(self) -> str:
        return compute_exact_content_signature(self.text)


REAL_SEGMENTS = [
    CodeSegment("data_structures", DATA_STRUCTURES.strip()),
    CodeSegment("graph_algorithms", GRAPH_ALGORITHMS.strip()),
    CodeSegment("sorting", SORTING.strip()),
]


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.manifest:
        return [{"case_id": "built_in_large_codebase", "segments": REAL_SEGMENTS}]
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cases = []
    for sample in manifest.get("samples", [])[: args.max_cases]:
        segments = []
        for file_info in sample.get("files", [])[: args.files_per_case]:
            content = Path(file_info["local_path"]).read_text(encoding="utf-8")
            if args.max_segment_chars and len(content) > args.max_segment_chars:
                content = content[: args.max_segment_chars]
            segments.append(CodeSegment(file_info["path"], content.strip()))
        if len(segments) >= 2:
            cases.append({"case_id": sample["instance_id"], "repo_key": sample.get("repo_key", ""), "segments": segments})
    if not cases:
        raise RuntimeError(f"manifest has no usable multi-file samples: {args.manifest}")
    return cases


def sha1_short(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def now_ms() -> float:
    return time.perf_counter() * 1000


def build_messages(role: str, segments: list[CodeSegment], extra_context: str = "") -> list[dict[str, str]]:
    body = [f"## Role\n{role}", ""]
    if extra_context:
        body.extend(["## Context", extra_context, ""])
    for idx, segment in enumerate(segments, 1):
        body.extend(
            [
                f"## code_base{idx}: {segment.name}",
                "```python",
                segment.text,
                "```",
                "",
            ]
        )
    body.append("Return a concise technical answer. Do not restate the full code.")
    return [
        {"role": "system", "content": "You are a senior coding assistant."},
        {"role": "user", "content": "\n".join(body)},
    ]


def find_subsequence(haystack: list[int], needle: list[int], start: int = 0) -> tuple[int, int]:
    if not needle:
        raise ValueError("empty needle")
    stop = len(haystack) - len(needle) + 1
    for i in range(max(0, start), stop):
        if haystack[i : i + len(needle)] == needle:
            return i, i + len(needle)
    raise ValueError("segment token sequence not found in prompt")


def token_bounds_for_text(tokenizer: Any, full_text: str, segment_text: str, char_start: int = 0) -> tuple[int, int, int]:
    char_pos = full_text.find(segment_text, char_start)
    if char_pos < 0:
        raise ValueError("segment text not found in prompt")
    start = len(tokenizer.encode(full_text[:char_pos], add_special_tokens=False))
    end = len(tokenizer.encode(full_text[: char_pos + len(segment_text)], add_special_tokens=False))
    return start, end, char_pos + len(segment_text)


def build_anchor_fields(tokenizer: Any, messages: list[dict[str, str]], segments: list[CodeSegment]) -> dict[str, Any]:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    token_spans = []
    anchor_spans = []
    char_cursor = 0
    for segment in segments:
        start, end, char_cursor = token_bounds_for_text(tokenizer, prompt, segment.text, char_cursor)
        payload = build_code_anchor_payload(segment.text, language="python")
        anchor_spans.append(
            {
                "anchor_type": "code_base",
                "signature": sha1_short(segment.name + ":" + segment.signature),
                "content_signature": segment.signature,
                "start_line": 1,
                "end_line": len(segment.text.splitlines()),
                "segment_name": segment.name,
                "ast_anchor_signature": payload.get("ast_anchor_signature", ""),
            }
        )
        token_spans.append(
            {
                "anchor_type": "code_base",
                "signature": sha1_short(segment.name + ":" + segment.signature),
                "content_signature": segment.signature,
                "start_token": start,
                "end_token": end,
                "segment_name": segment.name,
            }
        )
    joined_sig = sha1_short("|".join(s.signature for s in segments))
    return {
        "prompt_text": prompt,
        "prompt_tokens_local": len(prompt_ids),
        "code_anchor_signature": joined_sig,
        "code_content_signature": sha1_short("joined:" + "|".join(s.signature for s in segments)),
        "code_anchor_spans": anchor_spans,
        "code_anchor_token_spans": token_spans,
    }


def extract_text(body: dict[str, Any]) -> str:
    try:
        return body["choices"][0]["message"]["content"]
    except Exception:
        return ""


def extract_cached_tokens(body: dict[str, Any]) -> int:
    try:
        return int(body["usage"]["prompt_tokens_details"].get("cached_tokens", 0))
    except Exception:
        return 0


def extract_prompt_tokens(body: dict[str, Any]) -> int:
    try:
        return int(body["usage"]["prompt_tokens"])
    except Exception:
        return 0


def extract_lossy_meta(body: dict[str, Any]) -> dict[str, Any]:
    try:
        return dict(body["metadata"]["lossy_reuse"])
    except Exception:
        return {}


def token_f1(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    ref_counts: dict[str, int] = {}
    for tok in ref:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    common = 0
    for tok in hyp:
        if ref_counts.get(tok, 0) > 0:
            common += 1
            ref_counts[tok] -= 1
    precision = common / max(len(hyp), 1)
    recall = common / max(len(ref), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def kill_port(port: int):
    try:
        with open("/proc/net/tcp") as f:
            rows = f.readlines()[1:]
        inode = None
        for row in rows:
            parts = row.split()
            if parts[1].endswith(f":{port:04X}") and parts[3] == "0A":
                inode = parts[9]
                break
        if inode is None:
            return
        for pid in sorted(filter(str.isdigit, os.listdir("/proc")), key=int):
            fd_dir = f"/proc/{pid}/fd"
            try:
                for fd in os.listdir(fd_dir):
                    if os.readlink(f"{fd_dir}/{fd}") == f"socket:[{inode}]":
                        os.kill(int(pid), signal.SIGTERM)
                        time.sleep(2)
                        return
            except Exception:
                continue
    except Exception:
        return


def launch_server(args: argparse.Namespace) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT / "python")
    env["SGLANG_LOSSY_FUZZY_MATCH"] = "1"
    log_path = OUT_DIR / "sglang_server.log"
    cmd = [
        args.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--port",
        str(args.port),
        "--tp-size",
        "1",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "16384",
        "--enable-cache-report",
        "--disable-cuda-graph",
        "--log-level",
        "error",
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT),
        env=env,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )


async def wait_ready(port: int, timeout_s: int = 180) -> bool:
    deadline = time.monotonic() + timeout_s
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(f"http://127.0.0.1:{port}/health_generate") as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                await asyncio.sleep(5)
    return False


async def post_chat(session: aiohttp.ClientSession, port: int, payload: dict[str, Any]) -> dict[str, Any]:
    start = now_ms()
    async with session.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=300),
    ) as resp:
        body = await resp.json()
    return {"elapsed_ms": now_ms() - start, "body": body}


def make_payload(
    model: str,
    tokenizer: Any,
    messages: list[dict[str, str]],
    segments: list[CodeSegment],
    reuse_mode: str,
    max_tokens: int,
    extra_key: str | None = None,
    include_anchor: bool = True,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "return_cached_tokens_details": True,
        "reuse_mode": reuse_mode,
        "lossy_alignment_method": "kvcomm",
        "template_task_family": "coding_mas_exact_codebase",
    }
    if include_anchor:
        anchor = build_anchor_fields(tokenizer, messages, segments)
        payload.update(
            {
                "code_anchor_signature": anchor["code_anchor_signature"],
                "code_content_signature": anchor["code_content_signature"],
                "code_anchor_spans": anchor["code_anchor_spans"],
                "code_anchor_token_spans": anchor["code_anchor_token_spans"],
            }
        )
    if extra_key:
        payload["cache_salt"] = extra_key
    return payload


async def run_sglang_exact_reuse(args: argparse.Namespace, tokenizer: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    kill_port(args.port)
    await asyncio.sleep(1)
    proc = launch_server(args)
    try:
        if not await wait_ready(args.port):
            raise RuntimeError(f"sglang server did not become ready; see {OUT_DIR / 'sglang_server.log'}")

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            case_results = []
            for case in cases:
                segments = case["segments"]
                planner_segments = segments
                implementer_segments = segments[:2]
                debugger_segments = segments[2:3] if len(segments) >= 3 else segments[-1:]

                planner_messages = build_messages(
                    "planner",
                    planner_segments,
                    "Plan which components each downstream agent should inspect.",
                )
                implementer_messages = build_messages(
                    "implementer",
                    implementer_segments,
                    "Use the planner output to propose one concrete implementation change.",
                )
                debugger_messages = build_messages(
                    "debugger",
                    debugger_segments,
                    "Inspect the reported failure and identify the most likely defect.",
                )

                # Cold baselines run before planner warmup and omit anchor
                # metadata so they cannot populate anchor_kv_store.
                impl_lossless = await post_chat(
                    session,
                    args.port,
                    make_payload(
                        args.model,
                        tokenizer,
                        implementer_messages,
                        implementer_segments,
                        "lossless",
                        args.max_tokens,
                        extra_key=f"isolated-baseline:{case['case_id']}:implementer",
                        include_anchor=False,
                    ),
                )
                dbg_lossless = await post_chat(
                    session,
                    args.port,
                    make_payload(
                        args.model,
                        tokenizer,
                        debugger_messages,
                        debugger_segments,
                        "lossless",
                        args.max_tokens,
                        extra_key=f"isolated-baseline:{case['case_id']}:debugger",
                        include_anchor=False,
                    ),
                )

                planner = await post_chat(
                    session,
                    args.port,
                    make_payload(args.model, tokenizer, planner_messages, planner_segments, "lossless", args.max_tokens),
                )

                impl_lossy = await post_chat(
                    session,
                    args.port,
                    make_payload(args.model, tokenizer, implementer_messages, implementer_segments, "lossy", args.max_tokens),
                )
                dbg_lossy = await post_chat(
                    session,
                    args.port,
                    make_payload(args.model, tokenizer, debugger_messages, debugger_segments, "lossy", args.max_tokens),
                )

                pairs = [
                    ("implementer", impl_lossless, impl_lossy),
                    ("debugger", dbg_lossless, dbg_lossy),
                ]
                pair_results = []
                for name, lossless, lossy in pairs:
                    ref = extract_text(lossless["body"])
                    hyp = extract_text(lossy["body"])
                    pair_results.append(
                        {
                            "agent": name,
                            "lossless_elapsed_ms": round(lossless["elapsed_ms"], 2),
                            "lossy_elapsed_ms": round(lossy["elapsed_ms"], 2),
                            "speedup_vs_lossless": round(lossless["elapsed_ms"] / max(lossy["elapsed_ms"], 1e-6), 4),
                            "lossless_cached_tokens": extract_cached_tokens(lossless["body"]),
                            "lossy_cached_tokens": extract_cached_tokens(lossy["body"]),
                            "lossless_prompt_tokens": extract_prompt_tokens(lossless["body"]),
                            "lossy_prompt_tokens": extract_prompt_tokens(lossy["body"]),
                            "exact_output_match": ref == hyp,
                            "token_f1": round(token_f1(ref, hyp), 4),
                            "lossy_meta": extract_lossy_meta(lossy["body"]),
                            "lossless_output": ref,
                            "lossy_output": hyp,
                        }
                    )

                case_results.append(
                    {
                        "case_id": case["case_id"],
                        "repo_key": case.get("repo_key", ""),
                        "segments": [
                            {"name": s.name, "lines": len(s.text.splitlines()), "signature": s.signature}
                            for s in segments
                        ],
                        "planner": {
                            "elapsed_ms": round(planner["elapsed_ms"], 2),
                            "cached_tokens": extract_cached_tokens(planner["body"]),
                            "prompt_tokens": extract_prompt_tokens(planner["body"]),
                            "output": extract_text(planner["body"]),
                        },
                        "pairs": pair_results,
                    }
                )

        return {
            "cases": case_results,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        kill_port(args.port)


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
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device=keys.device, dtype=torch.float32) / dim))
    positions = torch.full((keys.shape[-2],), float(delta), device=keys.device)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).to(dtype=torch.float32)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    keys_f = keys.to(torch.float32)
    return (keys_f * cos + rotate_half_neox(keys_f) * sin).to(keys.dtype)


@torch.no_grad()
def run_hf_kv_delta(args: argparse.Namespace, tokenizer: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the HF KV delta test")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda").eval()

    rope_theta = float(getattr(model.config, "rope_theta", 1000000.0))
    results = []
    for case in cases:
        for segment in case["segments"][: args.files_per_case]:
            prefix_a = "You are reviewing a repository.\n"
            prefix_b = (
                "You are reviewing a repository after the planner produced a detailed "
                "multi-agent workflow summary. Keep the exact shared codebase below in memory.\n"
            )
            suffix = "\n\nQuestion: summarize the key risk in one sentence.\nAnswer:"
            prompt_a = prefix_a + segment.text + suffix
            prompt_b = prefix_b + segment.text + suffix
            ids_a = tokenizer.encode(prompt_a, return_tensors="pt", add_special_tokens=False).to(model.device)
            ids_b = tokenizer.encode(prompt_b, return_tensors="pt", add_special_tokens=False).to(model.device)
            a_start, a_end, _ = token_bounds_for_text(tokenizer, prompt_a, segment.text)
            b_start, b_end, _ = token_bounds_for_text(tokenizer, prompt_b, segment.text)
            if a_end - a_start != b_end - b_start:
                raise RuntimeError("segment token lengths diverged")

            torch.cuda.synchronize()
            t0 = now_ms()
            out_a = model(input_ids=ids_a, use_cache=True)
            torch.cuda.synchronize()
            prefill_a_ms = now_ms() - t0

            torch.cuda.synchronize()
            t0 = now_ms()
            out_b = model(input_ids=ids_b, use_cache=True)
            torch.cuda.synchronize()
            prefill_b_ms = now_ms() - t0

            layer_metrics = []
            for layer_idx in args.layers:
                k_a, v_a = get_layer_kv(out_a.past_key_values, layer_idx)
                k_b, v_b = get_layer_kv(out_b.past_key_values, layer_idx)
                old_k = k_a[:, :, a_start:a_end, :]
                true_k = k_b[:, :, b_start:b_end, :]
                old_v = v_a[:, :, a_start:a_end, :]
                true_v = v_b[:, :, b_start:b_end, :]
                rotated_k = apply_rope_delta_neox(old_k, b_start - a_start, rope_theta)
                k_diff = (rotated_k.float() - true_k.float()).abs()
                v_diff = (old_v.float() - true_v.float()).abs()
                k_cos = torch.nn.functional.cosine_similarity(
                    rotated_k.float().flatten(), true_k.float().flatten(), dim=0
                ).item()
                v_cos = torch.nn.functional.cosine_similarity(
                    old_v.float().flatten(), true_v.float().flatten(), dim=0
                ).item()
                layer_metrics.append(
                    {
                        "layer": layer_idx,
                        "k_mean_abs": round(k_diff.mean().item(), 6),
                        "k_max_abs": round(k_diff.max().item(), 6),
                        "k_cosine": round(k_cos, 6),
                        "v_mean_abs": round(v_diff.mean().item(), 6),
                        "v_max_abs": round(v_diff.max().item(), 6),
                        "v_cosine": round(v_cos, 6),
                    }
                )
            results.append(
                {
                    "case_id": case["case_id"],
                    "repo_key": case.get("repo_key", ""),
                    "codebase": segment.name,
                    "lines": len(segment.text.splitlines()),
                    "tokens": a_end - a_start,
                    "old_start": a_start,
                    "new_start": b_start,
                    "rope_delta": b_start - a_start,
                    "prefill_a_ms": round(prefill_a_ms, 2),
                    "prefill_b_ms": round(prefill_b_ms, 2),
                    "estimated_full_code_prefill_savings_ms": round(
                        prefill_b_ms * ((b_end - b_start) / max(ids_b.shape[1], 1)), 2
                    ),
                    "layers": layer_metrics,
                }
            )
            del out_a, out_b
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return {"rope_theta": rope_theta, "results": results}


async def main(args: argparse.Namespace):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cases = load_cases(args)

    summary: dict[str, Any] = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases": [
            {
                "case_id": case["case_id"],
                "repo_key": case.get("repo_key", ""),
                "segments": [
                    {"name": s.name, "lines": len(s.text.splitlines()), "signature": s.signature}
                    for s in case["segments"]
                ],
            }
            for case in cases
        ],
    }
    if not args.skip_hf:
        summary["hf_kv_delta"] = run_hf_kv_delta(args, tokenizer, cases)
    if not args.skip_sglang:
        summary["sglang_exact_reuse"] = await run_sglang_exact_reuse(args, tokenizer, cases)

    out_json = OUT_DIR / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--python", default=DEFAULT_PY)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument("--mem-fraction-static", type=float, default=0.82)
    parser.add_argument("--num-codebases", type=int, default=3)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--files-per-case", type=int, default=3)
    parser.add_argument("--max-segment-chars", type=int, default=30000)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 12, 24, 35])
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-sglang", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
