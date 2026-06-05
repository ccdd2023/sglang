#!/usr/bin/env python3
"""KVCOMM-style TTFT stress benchmark for long code-base segment reuse.

This benchmark is intentionally prefill/TTFT dominant. It complements the
100-case end-to-end serving run by using long repeated code segments, streaming
requests, and multi-agent reuse sweeps.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

PROJECT = Path(__file__).resolve().parents[2]
MAS_SRC = PROJECT.parent / "MAScoder" / "src"
for entry in (str(MAS_SRC), str(PROJECT), str(PROJECT / "python")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mascoder.code_anchor import build_code_anchor_payload, compute_exact_content_signature  # noqa: E402

DEFAULT_MODEL = "/home/gfy/models/Qwen2.5-7B-Instruct"
DEFAULT_PYTHON = "/home/gfy/.conda/envs/sglang-kvflow/bin/python"
DEFAULT_MANIFEST = PROJECT / "results" / "repo_level_datasets" / "manifest_500.json"
OUT_DIR = PROJECT / "results" / "kvcomm_ttft_stress" / "qwen2_5_7b"

E6_MODES = [
    "no_reuse_fresh_salt",
    "prefix_cache_only",
    "exact_reuse_no_hints",
    "exact_reuse_plus_code_hints",
]
E7_MODES = ["prefix_cache_only", "exact_reuse_plus_code_hints"]
E8_MODES = [
    "ablation_exact_gate_rope",
    "ablation_exact_no_hints",
    "ablation_hints_no_exact",
    "ablation_prefix_only",
]
AGENT_ROLES = ["implementer", "debugger", "reviewer", "verifier", "auditor"]


@dataclass
class CodeSegment:
    name: str
    text: str

    @property
    def signature(self) -> str:
        return compute_exact_content_signature(self.text)


def sha1_short(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def now_ms() -> float:
    return time.perf_counter() * 1000


def token_f1(a: str, b: str) -> float:
    aa = a.split()
    bb = b.split()
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    ca, cb = Counter(aa), Counter(bb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(aa)
    recall = overlap / len(bb)
    return 2 * precision * recall / (precision + recall)


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
        signature = sha1_short(segment.name + ":" + segment.signature)
        anchor_spans.append(
            {
                "anchor_type": "code_base",
                "signature": signature,
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
                "signature": signature,
                "content_signature": segment.signature,
                "start_token": start,
                "end_token": end,
                "segment_name": segment.name,
            }
        )
    return {
        "prompt_text": prompt,
        "prompt_tokens_local": len(prompt_ids),
        "code_anchor_signature": sha1_short("|".join(s.signature for s in segments)),
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


def extract_lossy_meta(body: dict[str, Any]) -> dict[str, Any]:
    try:
        return dict(body["metadata"]["lossy_reuse"])
    except Exception:
        return {}


def kill_port(port: int) -> None:
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


async def wait_ready(port: int, timeout_s: int = 180) -> bool:
    deadline = time.time() + timeout_s
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(2)
    return False


async def post_chat(session: aiohttp.ClientSession, port: int, payload: dict[str, Any]) -> dict[str, Any]:
    start = now_ms()
    async with session.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload) as resp:
        body = await resp.json()
    return {"elapsed_ms": now_ms() - start, "body": body}


def make_codebase_hints(segments: list[CodeSegment], target_agent: str = "implementer") -> list[dict[str, Any]]:
    return [
        {
            "code_base_id": f"code_base{idx}:{segment.name}",
            "content_signature": segment.signature,
            "target_agent": target_agent,
            "steps_to_use": 1,
            "priority": 1,
            "match_required": "exact_code_content_signature",
            "text": segment.text,
        }
        for idx, segment in enumerate(segments, 1)
    ]


def launch_server(args: argparse.Namespace) -> subprocess.Popen:
    env = dict(**os.environ)
    env["PYTHONPATH"] = str(PROJECT / "python")
    env["SGLANG_LOSSY_FUZZY_MATCH"] = "1"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "sglang_server.log"
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
    if not args.disable_hierarchical_cache:
        cmd.extend(
            [
                "--radix-eviction-policy",
                "priority",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                str(args.hicache_ratio),
                "--hicache-write-policy",
                "write_back",
                "--enable-hicache-prefetch",
            ]
        )
    if args.hicache_storage_backend:
        env["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = str(args.out_dir / "hicache_file_storage")
        cmd.extend(
            [
                "--hicache-storage-backend",
                args.hicache_storage_backend,
                "--hicache-storage-prefetch-policy",
                "best_effort",
            ]
        )
    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT),
        env=env,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )


def safe_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs: list[float], pct: float) -> float:
    if not xs:
        return 0.0
    vals = sorted(xs)
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * pct))))
    return vals[idx]


def load_long_cases(args: argparse.Namespace, max_file_chars: int, segment_count: int) -> list[dict[str, Any]]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    candidates = []
    for sample in manifest.get("samples", []):
        files = []
        for info in sample.get("files", []):
            path = Path(info.get("local_path", ""))
            if not path.exists():
                continue
            size = path.stat().st_size
            files.append((size, info, path))
        if len(files) < segment_count:
            continue
        files.sort(reverse=True, key=lambda item: item[0])
        score = sum(size for size, _, _ in files[:segment_count])
        candidates.append((score, sample, files[:segment_count]))

    candidates.sort(reverse=True, key=lambda item: item[0])
    selected = candidates[args.start_index : args.start_index + args.max_cases]
    cases = []
    for _, sample, files in selected:
        segments = []
        for _, info, path in files:
            text = path.read_text(encoding="utf-8", errors="replace").rstrip()
            if max_file_chars and len(text) > max_file_chars:
                text = text[:max_file_chars].rstrip()
            if text:
                segments.append(CodeSegment(info.get("path", path.name), text))
        if len(segments) >= segment_count:
            cases.append(
                {
                    "case_id": sample["instance_id"],
                    "repo": sample.get("repo", sample.get("repo_key", "")),
                    "segments": segments[:segment_count],
                }
            )
    return cases


def build_stress_messages(
    case: dict[str, Any],
    segments: list[CodeSegment],
    role: str,
    agent_idx: int = 0,
    extra_context: str = "",
) -> list[dict[str, str]]:
    body = [
        f"## Agent role\n{role}",
        f"## Case\n{case['case_id']}",
        "## Instruction",
        "Inspect the repeated repository code and answer with one concise implementation risk.",
    ]
    if extra_context:
        body += ["## Upstream context", extra_context]
    for idx, segment in enumerate(segments, 1):
        body += [
            f"## code_base{idx}: {segment.name}",
            "```python",
            segment.text,
            "```",
        ]
    body += [
        "## Output",
        f"Return exactly one short sentence for agent {agent_idx}.",
    ]
    return [
        {"role": "system", "content": "You are a senior software engineering agent."},
        {"role": "user", "content": "\n".join(body)},
    ]


def make_payload(
    args: argparse.Namespace,
    tokenizer: Any,
    case: dict[str, Any],
    segments: list[CodeSegment],
    mode: str,
    max_tokens: int,
    salt: str,
    role: str = "implementer",
    agent_idx: int = 0,
    extra_context: str = "",
) -> dict[str, Any]:
    messages = build_stress_messages(case, segments, role, agent_idx, extra_context)
    include_anchor = mode in {"exact_reuse_no_hints", "exact_reuse_plus_code_hints",
                              "ablation_exact_gate_rope", "ablation_exact_no_hints"}
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "return_cached_tokens_details": True,
        "reuse_mode": "lossy" if include_anchor else "lossless",
        "lossy_alignment_method": "kvcomm",
        "template_task_family": "kvcomm_ttft_stress",
        "cache_salt": salt,
        "priority": 1,
    }
    if mode in {"prefix_cache_only", "exact_reuse_no_hints", "exact_reuse_plus_code_hints"}:
        payload["next_agent_prefix"] = f"You are the {role}. Reuse the planner code context."
    if mode == "exact_reuse_plus_code_hints":
        payload["codebase_prefetch_hints"] = make_codebase_hints(segments, target_agent=role)
    if include_anchor:
        payload.update(build_anchor_fields(tokenizer, messages, segments))
    # E8 ablation: selective field stripping to isolate each subsystem's contribution.
    if mode.startswith("ablation_"):
        if mode == "ablation_exact_gate_rope":
            # Same as exact_reuse_no_hints: anchor spans present, no hints, has agent prefix.
            payload["next_agent_prefix"] = f"You are the {role}. Reuse the planner code context."
        elif mode == "ablation_exact_no_hints":
            # Anchor spans present, but no next_agent_prefix routing — isolates anchor without scheduling.
            pass  # include_anchor already True; deliberately skip next_agent_prefix
        elif mode == "ablation_hints_no_exact":
            # Hints present but reuse_mode=lossless blocks exact-content match server-side.
            payload["reuse_mode"] = "lossless"
            payload["lossy_alignment_method"] = ""
            payload["codebase_prefetch_hints"] = make_codebase_hints(segments, target_agent=role)
            payload["next_agent_prefix"] = f"You are the {role}. Reuse the planner code context."
        elif mode == "ablation_prefix_only":
            # Pure prefix cache: no anchor, no hints, no lossy.
            payload["reuse_mode"] = "lossless"
            payload["lossy_alignment_method"] = ""
            payload["next_agent_prefix"] = f"You are the {role}. Reuse the planner code context."
    return payload


async def post_chat_stream(session: aiohttp.ClientSession, port: int, payload: dict[str, Any]) -> dict[str, Any]:
    start = now_ms()
    ttft_ms = None
    text = ""
    final_body: dict[str, Any] = {}
    cached_tokens = 0
    prompt_tokens = int(payload.get("prompt_tokens_local", 0) or 0)
    meta: dict[str, Any] = {}
    async with session.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={**payload, "stream": True, "stream_options": {"include_usage": True}},
    ) as resp:
        async for raw in resp.content:
            line = raw.decode(errors="ignore").strip()
            if not line or line == "data: [DONE]":
                continue
            if not line.startswith("data: "):
                continue
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            final_body = chunk
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content") or ""
                if content and ttft_ms is None:
                    ttft_ms = now_ms() - start
                text += content
            usage = chunk.get("usage") or {}
            if usage:
                prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens) or 0)
                details = usage.get("prompt_tokens_details") or {}
                cached_tokens = int(details.get("cached_tokens", cached_tokens) or 0)
            chunk_meta = chunk.get("metadata") or {}
            if chunk_meta.get("lossy_reuse"):
                meta = dict(chunk_meta["lossy_reuse"])
    e2e_ms = now_ms() - start
    if ttft_ms is None:
        ttft_ms = e2e_ms
    return {
        "elapsed_ms": round(e2e_ms, 2),
        "e2e_ms": round(e2e_ms, 2),
        "ttft_ms": round(ttft_ms, 2),
        "text": text,
        "cached_tokens": cached_tokens,
        "prompt_tokens": prompt_tokens,
        "body": final_body,
        "metadata": {"lossy_reuse": meta} if meta else {},
    }


async def warm_planner(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    tokenizer: Any,
    case: dict[str, Any],
    segments: list[CodeSegment],
    max_file_chars: int,
    segment_count: int,
) -> None:
    payload = make_payload(
        args,
        tokenizer,
        case,
        segments,
        "exact_reuse_plus_code_hints",
        max_tokens=8,
        salt=f"planner:{case['case_id']}:{max_file_chars}:{segment_count}",
        role="planner",
        agent_idx=0,
    )
    await post_chat(session, args.port, payload)


def row_from_response(
    case: dict[str, Any],
    mode: str,
    response: dict[str, Any],
    max_tokens: int,
    max_file_chars: int,
    segment_count: int,
    experiment: str,
    agent_id: str = "",
    agent_count: int = 0,
    baseline_text: str = "",
) -> dict[str, Any]:
    meta = extract_lossy_meta(response.get("body", {})) or response.get("metadata", {}).get("lossy_reuse", {})
    match_reason = (
        meta.get("lossy_first_match_reason")
        or meta.get("lossy_final_match_reason")
        or meta.get("lossy_anchor_match_used")
        or ""
    )
    matched_sig = (
        meta.get("lossy_first_matched_content_signature")
        or meta.get("lossy_final_matched_content_signature")
        or meta.get("lossy_anchor_match_content_signature")
        or ""
    )
    text = response.get("text") or extract_text(response.get("body", {})) or ""
    cached = int(response.get("cached_tokens") or extract_cached_tokens(response.get("body", {})) or 0)
    prompt_tokens = int(response.get("prompt_tokens") or 0)
    return {
        "experiment": experiment,
        "case_id": case["case_id"],
        "repo": case.get("repo", ""),
        "mode": mode,
        "agent_id": agent_id,
        "agent_count": agent_count,
        "segment_count": segment_count,
        "max_file_chars": max_file_chars,
        "max_tokens": max_tokens,
        "ttft_ms": response.get("ttft_ms", response.get("elapsed_ms")),
        "e2e_ms": response.get("e2e_ms", response.get("elapsed_ms")),
        "elapsed_ms": response.get("elapsed_ms", response.get("e2e_ms")),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "cached_ratio": round(cached / prompt_tokens, 6) if prompt_tokens else 0.0,
        "exact_hit": match_reason == "exact_code_content_signature",
        "match_reason": match_reason,
        "matched_content_signature": matched_sig,
        "output_exact_match_vs_baseline": bool(baseline_text) and text == baseline_text,
        "output_token_f1_vs_baseline": round(token_f1(text, baseline_text), 4) if baseline_text else 1.0,
        "output_chars": len(text),
    }


async def run_e6(args: argparse.Namespace, session: aiohttp.ClientSession, tokenizer: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for max_file_chars in args.length_buckets:
        cases = load_long_cases(args, max_file_chars=max_file_chars, segment_count=args.files_per_case)
        for case in cases:
            segments = case["segments"]
            await warm_planner(session, args, tokenizer, case, segments, max_file_chars, args.files_per_case)
            for max_tokens in args.max_token_settings:
                outputs: dict[str, str] = {}
                mode_responses: dict[str, dict[str, Any]] = {}
                for mode in E6_MODES:
                    salt = (
                        f"fresh:{case['case_id']}:{max_file_chars}:{max_tokens}:{time.time_ns()}"
                        if mode == "no_reuse_fresh_salt"
                        else f"stress:{case['case_id']}:{max_file_chars}:{max_tokens}:{mode}"
                    )
                    payload = make_payload(
                        args,
                        tokenizer,
                        case,
                        segments,
                        mode,
                        max_tokens=max_tokens,
                        salt=salt,
                        role="implementer",
                        agent_idx=1,
                        extra_context=f"Length bucket {max_file_chars}; mode {mode}.",
                    )
                    response = await post_chat_stream(session, args.port, payload)
                    mode_responses[mode] = response
                    outputs[mode] = response.get("text", "")
                baseline_text = outputs.get("prefix_cache_only", outputs.get("no_reuse_fresh_salt", ""))
                for mode in E6_MODES:
                    rows.append(
                        row_from_response(
                            case,
                            mode,
                            mode_responses[mode],
                            max_tokens,
                            max_file_chars,
                            args.files_per_case,
                            "ttft_stress",
                            agent_id="implementer",
                            agent_count=1,
                            baseline_text=baseline_text,
                        )
                    )
            print(f"[E6] {case['case_id']} bucket={max_file_chars} done")
    return rows


async def run_e7(args: argparse.Namespace, session: aiohttp.ClientSession, tokenizer: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for max_file_chars in args.agent_length_buckets:
        for segment_count in args.segment_counts:
            cases = load_long_cases(args, max_file_chars=max_file_chars, segment_count=segment_count)[: args.agent_max_cases]
            for agent_count in args.agent_counts:
                for case in cases:
                    segments = case["segments"][:segment_count]
                    await warm_planner(session, args, tokenizer, case, segments, max_file_chars, segment_count)
                    upstream = "Planner cached exact repository code objects for downstream agents."
                    for mode in E7_MODES:
                        mode_rows = []
                        for idx, role in enumerate(AGENT_ROLES[:agent_count], 1):
                            payload = make_payload(
                                args,
                                tokenizer,
                                case,
                                segments,
                                mode,
                                max_tokens=args.agent_max_tokens,
                                salt=f"agent:{case['case_id']}:{max_file_chars}:{segment_count}:{agent_count}:{mode}:{idx}",
                                role=role,
                                agent_idx=idx,
                                extra_context=upstream + f" Previous agent index: {idx - 1}.",
                            )
                            response = await post_chat_stream(session, args.port, payload)
                            row = row_from_response(
                                case,
                                mode,
                                response,
                                args.agent_max_tokens,
                                max_file_chars,
                                segment_count,
                                "agent_scaling",
                                agent_id=role,
                                agent_count=agent_count,
                            )
                            rows.append(row)
                            mode_rows.append(row)
                            upstream += f" {role} observed {row['cached_tokens']} cached tokens."
                        rows.append(
                            {
                                "experiment": "agent_scaling_workflow",
                                "case_id": case["case_id"],
                                "repo": case.get("repo", ""),
                                "mode": mode,
                                "agent_id": "workflow",
                                "agent_count": agent_count,
                                "segment_count": segment_count,
                                "max_file_chars": max_file_chars,
                                "max_tokens": args.agent_max_tokens,
                                "ttft_ms": round(sum(float(r["ttft_ms"]) for r in mode_rows), 2),
                                "e2e_ms": round(sum(float(r["e2e_ms"]) for r in mode_rows), 2),
                                "elapsed_ms": round(sum(float(r["elapsed_ms"]) for r in mode_rows), 2),
                                "prompt_tokens": sum(int(r["prompt_tokens"]) for r in mode_rows),
                                "cached_tokens": sum(int(r["cached_tokens"]) for r in mode_rows),
                                "cached_ratio": round(
                                    sum(int(r["cached_tokens"]) for r in mode_rows)
                                    / max(sum(int(r["prompt_tokens"]) for r in mode_rows), 1),
                                    6,
                                ),
                                "exact_hit": all(str(r["exact_hit"]).lower() == "true" for r in mode_rows),
                                "match_reason": "workflow_summary",
                                "matched_content_signature": "",
                                "output_exact_match_vs_baseline": "",
                                "output_token_f1_vs_baseline": 1.0,
                                "output_chars": 0,
                            }
                        )
                    print(
                        f"[E7] {case['case_id']} bucket={max_file_chars} "
                        f"segments={segment_count} agents={agent_count} done"
                    )
    return rows


async def run_e8(args: argparse.Namespace, session: aiohttp.ClientSession, tokenizer: Any) -> list[dict[str, Any]]:
    """KVCOMM performance ablation: isolate each subsystem's contribution to TTFT speedup."""
    rows: list[dict[str, Any]] = []
    max_file_chars = args.e8_length
    max_tokens = 32  # quality-check setting
    cases = load_long_cases(args, max_file_chars=max_file_chars, segment_count=args.files_per_case)[: args.e8_cases]
    for case in cases:
        segments = case["segments"]
        await warm_planner(session, args, tokenizer, case, segments, max_file_chars, args.files_per_case)
        outputs: dict[str, str] = {}
        mode_responses: dict[str, dict[str, Any]] = {}
        for mode in E8_MODES:
            salt = f"ablation:{case['case_id']}:{max_file_chars}:{max_tokens}:{mode}"
            payload = make_payload(
                args,
                tokenizer,
                case,
                segments,
                mode,
                max_tokens=max_tokens,
                salt=salt,
                role="implementer",
                agent_idx=1,
                extra_context=f"Ablation; mode {mode}.",
            )
            response = await post_chat_stream(session, args.port, payload)
            mode_responses[mode] = response
            outputs[mode] = response.get("text", "")
        baseline_text = outputs.get("ablation_prefix_only", "")
        for mode in E8_MODES:
            rows.append(
                row_from_response(
                    case,
                    mode,
                    mode_responses[mode],
                    max_tokens,
                    max_file_chars,
                    args.files_per_case,
                    "ablation",
                    agent_id="implementer",
                    agent_count=1,
                    baseline_text=baseline_text,
                )
            )
        print(f"[E8] {case['case_id']} bucket={max_file_chars} done")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["experiment"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("experiment"),
                row.get("mode"),
                row.get("max_file_chars"),
                row.get("max_tokens"),
                row.get("agent_count"),
                row.get("segment_count"),
            )
        ].append(row)
    summary = {}
    for key, rs in grouped.items():
        ttft = [float(r["ttft_ms"]) for r in rs if r.get("agent_id") != "workflow" or r.get("experiment") == "agent_scaling_workflow"]
        cached = [float(r["cached_tokens"]) for r in rs]
        f1 = [float(r["output_token_f1_vs_baseline"]) for r in rs if str(r.get("output_token_f1_vs_baseline", "")) not in {"", "None"}]
        summary["|".join(map(str, key))] = {
            "n": len(rs),
            "avg_ttft_ms": safe_mean(ttft),
            "p50_ttft_ms": percentile(ttft, 0.5),
            "p90_ttft_ms": percentile(ttft, 0.9),
            "avg_cached_tokens": safe_mean(cached),
            "exact_hit_rate": safe_mean([1.0 if str(r.get("exact_hit")).lower() == "true" else 0.0 for r in rs]),
            "avg_token_f1_vs_baseline": safe_mean(f1),
        }
    return summary


def write_report(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# KVCOMM TTFT Stress Report",
        "",
        "This run is prefill/TTFT dominant and is intended to complement the realistic 100-case E2E table.",
        "",
        "## Output Schema",
        "",
        "`ttft_ms`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `cached_ratio`, `exact_hit`, `match_reason`, `matched_content_signature`, `output_exact_match_vs_baseline`, `output_token_f1_vs_baseline`, `mode`, `case_id`, `agent_id`, `segment_count`, `max_file_chars`.",
        "",
        "## Row Counts",
        "",
    ]
    counts = Counter(str(r.get("experiment")) for r in rows)
    for experiment, count in sorted(counts.items()):
        lines.append(f"- {experiment}: {count}")
    lines += ["", "## Summary Groups", ""]
    for key, item in sorted(summary.items()):
        lines.append(
            f"- `{key}`: n={item['n']}, avg TTFT={item['avg_ttft_ms']:.1f} ms, "
            f"p50={item['p50_ttft_ms']:.1f} ms, exact hit={item['exact_hit_rate']:.2f}, "
            f"F1={item['avg_token_f1_vs_baseline']:.4f}"
        )
    (out_dir / "TTFT_STRESS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    args.out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT / args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kill_port(args.port)
    await asyncio.sleep(1)
    proc = launch_server(args)
    rows: list[dict[str, Any]] = []
    try:
        if not await wait_ready(args.port, timeout_s=args.server_timeout):
            raise RuntimeError(f"server did not become ready; see {args.out_dir / 'sglang_server.log'}")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.eval_timeout)) as session:
            if not args.skip_e6:
                rows.extend(await run_e6(args, session, tokenizer))
            if not args.skip_e7:
                rows.extend(await run_e7(args, session, tokenizer))
            if not args.skip_e8:
                rows.extend(await run_e8(args, session, tokenizer))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        kill_port(args.port)

    summary = {
        "model": args.model,
        "manifest": str(args.manifest),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "length_buckets": args.length_buckets,
        "max_token_settings": args.max_token_settings,
        "agent_counts": args.agent_counts,
        "segment_counts": args.segment_counts,
        "agent_length_buckets": args.agent_length_buckets,
        "rows": rows,
    }
    group_summary = summarize_rows(rows)
    summary["group_summary"] = group_summary
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.out_dir / "ttft_stress_table.csv", rows)
    write_report(args.out_dir, rows, group_summary)
    return summary


def parse_csv_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--files-per-case", type=int, default=3)
    parser.add_argument("--length-buckets", type=parse_csv_ints, default=parse_csv_ints("8000,16000,32000,48000"))
    parser.add_argument("--max-token-settings", type=parse_csv_ints, default=parse_csv_ints("1,32"))
    parser.add_argument("--agent-counts", type=parse_csv_ints, default=parse_csv_ints("2,3,5"))
    parser.add_argument("--segment-counts", type=parse_csv_ints, default=parse_csv_ints("1,2,3"))
    parser.add_argument("--agent-length-buckets", type=parse_csv_ints, default=parse_csv_ints("8000,16000,32000"))
    parser.add_argument("--agent-max-cases", type=int, default=10)
    parser.add_argument("--agent-max-tokens", type=int, default=1)
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument("--mem-fraction-static", type=float, default=0.78)
    parser.add_argument("--hicache-ratio", type=float, default=1.5)
    parser.add_argument("--hicache-storage-backend", default="")
    parser.add_argument("--disable-hierarchical-cache", action="store_true")
    parser.add_argument("--server-timeout", type=int, default=180)
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-e6", action="store_true")
    parser.add_argument("--skip-e7", action="store_true")
    parser.add_argument("--skip-e8", action="store_true")
    parser.add_argument("--e8-cases", type=int, default=20)
    parser.add_argument("--e8-length", type=int, default=32000)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_benchmark(parse_args()))
