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
OUT_DIR = PROJECT / "results" / "ttft_agenttemplatekv" / "qwen2_5_7b_micro"

E6_MODES = [
    "no_reuse_fresh_salt",
    "prefix_cache_only",
    "exact_reuse_no_hints",
    "exact_reuse_plus_code_hints",
    "hints_no_exact",
    "placeholder_knn_reuse",
    "placeholder_knn_plus_exact",
]
CORE_TTFT_MODES = [
    # Reordered: run placeholder_knn_reuse FIRST (right after warm_planner)
    # so its per-role writes happen on a fresh cache.  When it ran LAST
    # (original order), the 4 prior modes × 5 agents = 20 prior writes
    # filled the radix tree and LRU-evicted some role paths before the
    # placeholder_knn_reuse agents could read them, causing cold-cache
    # TTFTs at agent_count=5.  Running first keeps the pool of
    # placeholder_knn_reuse writes intact long enough for downstream
    # agents to reuse them.
    "placeholder_knn_reuse",
    "prefix_cache_only",
    "exact_reuse_no_hints",
    "exact_reuse_plus_code_hints",
    "hints_no_exact",
]
E7_MODES = CORE_TTFT_MODES
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


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


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
            for endpoint in ("health_generate", "health"):
                try:
                    async with session.get(f"http://127.0.0.1:{port}/{endpoint}") as resp:
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
    env["SGLANG_LOSSY_SKIP_TOKEN_CHECK"] = "1"
    env["SGLANG_LOSSY_MAX_ZERO_GAP"] = str(args.lossy_max_zero_gap)
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


# ---------------------------------------------------------------------------
# Placeholder slots (Duke 2026 KVCOMM-style).  A `PlaceholderSlot` is one
# chunk of the prompt that semantically represents "this is the upstream
# planner output", "this is the architecture context", etc.  The slot
# taxonomy is keyed by `slot_id`; the server keeps a per-slot embedding
# pool and uses k-NN to look up the nearest historical slot text.
# ---------------------------------------------------------------------------


@dataclass
class PlaceholderSlot:
    """One named block of the prompt.  `slot_id` is the pool key."""

    slot_id: str
    label: str
    text: str


def build_slot_messages(
    case: dict[str, Any],
    segments: list[CodeSegment],
    role: str,
    agent_idx: int = 0,
    extra_context: str = "",
    placeholder_slots: list[PlaceholderSlot] | None = None,
) -> tuple[list[dict[str, str]], list[PlaceholderSlot]]:
    """Build a slot-decomposed user message (Duke 2026 KVCOMM-style).

    Returns (messages, slots_used).  When `placeholder_slots` is None, falls
    back to a default slot taxonomy: a single `extra_context` slot for the
    upstream text plus one `code_base{N}` slot per code segment.  When
    provided, only the supplied slots are rendered (caller controls the
    taxonomy).
    """
    if placeholder_slots is not None:
        slots = list(placeholder_slots)
    else:
        slots = []
        # IMPORTANT: code_base slots come FIRST, extra_context LAST.
        # This ensures extra_context's start_token is LARGE (after the
        # code segments) and therefore > prefix_len.  The k-NN body's
        # `start < prefix_len` guard requires this: when start < prefix_len,
        # the slot overlaps with the prefix's pre-cached region and the
        # flashinfer attention backend can't handle a discontinuous KV
        # layout (it crashes with "q_indptr-35 o_indptr-0 should be non-
        # negative" if we try).  See radix_cache.py `_try_placeholder_knn_
        # lossy_match_body` for the guard.
        for idx, segment in enumerate(segments, 1):
            slots.append(
                PlaceholderSlot(
                    slot_id=f"code_base{idx}", label=f"code_base{idx}: {segment.name}",
                    text=segment.text,
                ),
            )
        if extra_context:
            slots.append(
                PlaceholderSlot(
                    slot_id="extra_context", label="Upstream context",
                    text=extra_context,
                ),
            )

    body = [
        f"## Agent role\n{role}",
        f"## Case\n{case['case_id']}",
        "## Instruction",
        "Inspect the repeated repository code and answer with one concise implementation risk.",
    ]
    for slot in slots:
        body += [f"## {slot.label}", slot.text]
    body += [
        "## Output",
        f"Return exactly one short sentence for agent {agent_idx}.",
    ]
    return (
        [
            {"role": "system", "content": "You are a senior software engineering agent."},
            {"role": "user", "content": "\n".join(body)},
        ],
        slots,
    )


def build_placeholder_anchor_fields(
    tokenizer: Any,
    messages: list[dict[str, str]],
    slots: list[PlaceholderSlot],
) -> dict[str, Any]:
    """Compute per-slot `start_token` / `end_token` for the slot-decomposed
    prompt.  Returns ``placeholder_anchor_token_spans`` ready to attach to
    the OpenAI payload.  Slots whose text is not found in the rendered
    prompt are silently dropped.
    """
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    spans: list[dict[str, Any]] = []
    cursor = 0
    for slot in slots:
        if not slot.text:
            continue
        try:
            start, end, cursor = token_bounds_for_text(
                tokenizer, prompt, slot.text, char_start=cursor,
            )
        except ValueError:
            continue
        spans.append(
            {
                "slot_id": slot.slot_id,
                "label": slot.label,
                "start_token": start,
                "end_token": end,
                "content_signature": sha1_short(slot.slot_id + ":" + slot.text[:64]),
                "text": slot.text,
            },
        )
    return {"placeholder_anchor_token_spans": spans}


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
    if mode in {"placeholder_knn_reuse", "placeholder_knn_plus_exact"}:
        # Build the slot-decomposed prompt and the placeholder anchor spans.
        # The byte-exact path (`+exact`) ALSO receives `code_anchor_token_spans`
        # so both paths can run; the placeholder k-NN runs *after* byte-exact
        # in `match_prefix`.
        messages, slots = build_slot_messages(
            case, segments, role, agent_idx, extra_context,
        )
        include_anchor = mode == "placeholder_knn_plus_exact"
    else:
        messages = build_stress_messages(case, segments, role, agent_idx, extra_context)
        slots = []
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
    if mode in {"prefix_cache_only", "exact_reuse_no_hints", "exact_reuse_plus_code_hints", "hints_no_exact"}:
        payload["next_agent_prefix"] = f"You are the {role}. Reuse the planner code context."
    if mode in {"exact_reuse_plus_code_hints", "hints_no_exact"}:
        payload["codebase_prefetch_hints"] = make_codebase_hints(segments, target_agent=role)
    if mode == "hints_no_exact":
        payload["reuse_mode"] = "lossless"
        payload["lossy_alignment_method"] = ""
    if include_anchor:
        payload.update(build_anchor_fields(tokenizer, messages, segments))
    # Placeholder k-NN: always attach the per-slot spans when in
    # placeholder_knn_reuse or placeholder_knn_plus_exact mode.
    if mode in {"placeholder_knn_reuse", "placeholder_knn_plus_exact"} and slots:
        payload.update(build_placeholder_anchor_fields(tokenizer, messages, slots))
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
    """Warmup the server with one planner request that populates BOTH
    the byte-exact anchor pool (via exact_reuse_plus_code_hints mode) and
    the placeholder k-NN anchor pool (via placeholder_knn_reuse mode,
    issued right after with a salt prefix to avoid cache_salt collisions).

    The placeholder warmup is what gives downstream agents (especially
    agent 1) something to match in their placeholder_anchor_pool, since
    the byte-exact warmup alone doesn't write to that pool.
    """
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
    # O9: warm the placeholder k-NN anchor pool so downstream agents
    # have something to match.  Uses placeholder_knn_reuse mode with
    # the same planner prompt structure so the slot texts are the same.
    placeholder_payload = make_payload(
        args,
        tokenizer,
        case,
        segments,
        "placeholder_knn_reuse",
        max_tokens=8,
        salt=f"placeholder_warmup:{case['case_id']}:{max_file_chars}:{segment_count}",
        role="implementer",
        agent_idx=0,
    )
    await post_chat(session, args.port, placeholder_payload)


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
    hint_count = int(meta.get("codebase_prefetch_hint_count") or 0)
    matched_tokens = int(meta.get("codebase_prefetch_matched_tokens") or 0)
    success_count = int(meta.get("codebase_prefetch_success_count") or 0)
    device_hit_count = int(meta.get("codebase_prefetch_device_hit_count") or 0)
    atkv_hit_count = int(meta.get("agenttemplatekv_prefetch_hit_count") or 0)
    atkv_miss_count = int(meta.get("agenttemplatekv_prefetch_miss_count") or 0)
    protected_tokens = int(meta.get("agenttemplatekv_prefetch_protected_tokens") or 0)
    newly_protected_tokens = int(meta.get("agenttemplatekv_prefetch_newly_protected_tokens") or 0)
    consumed_count = int(meta.get("agenttemplatekv_prefetch_consumed_count") or 0)
    expired_tokens = int(meta.get("agenttemplatekv_prefetch_expired_tokens") or 0)
    large_gap_rejections = int(meta.get("agenttemplatekv_rejected_large_gap_count") or 0)
    anchor_used = bool(meta.get("lossy_anchor_match_used"))
    anchor_gap_len = int(meta.get("lossy_anchor_match_gap_len") or 0)
    anchor_rope_delta = int(meta.get("lossy_anchor_rope_delta") or 0)
    # Placeholder k-NN reuse telemetry (PR 4 of placeholder plan).
    placeholder_hits = int(meta.get("placeholder_anchor_pool_hit_count") or 0)
    placeholder_misses = int(meta.get("placeholder_anchor_pool_miss_count") or 0)
    placeholder_matched_slots = int(meta.get("placeholder_kv_prefill_matched_slots") or 0)
    placeholder_skipped_tokens = int(meta.get("placeholder_kv_prefill_skipped_tokens") or 0)
    placeholder_sim_mean = float(meta.get("placeholder_knn_topk_similarity_mean") or 0.0)
    placeholder_stored = int(meta.get("placeholder_anchor_store_entry_count") or 0)
    placeholder_skipped_low_f1 = int(meta.get("placeholder_anchor_store_skipped_low_f1_count") or 0)
    placeholder_skipped_cost = int(meta.get("placeholder_anchor_pool_skipped_cost_count") or 0)
    placeholder_skipped_high_overlap = int(meta.get("placeholder_knn_skipped_high_overlap_count") or 0)
    placeholder_skipped_short_new = int(meta.get("placeholder_knn_skipped_short_new_tokens_count") or 0)
    placeholder_skipped_high_span = int(meta.get("placeholder_knn_skipped_high_span_overlap_count") or 0)
    placeholder_skipped_high_new = int(meta.get("placeholder_knn_skipped_high_new_token_ratio_count") or 0)
    placeholder_pre_rotated_hit = int(meta.get("placeholder_knn_pre_rotated_hit_count") or 0)
    placeholder_pre_rotated_miss = int(meta.get("placeholder_knn_pre_rotated_miss_count") or 0)
    placeholder_head_rotation_tokens = int(meta.get("placeholder_knn_head_rotation_tokens") or 0)
    placeholder_head_rotation_total_ops = int(meta.get("placeholder_knn_head_rotation_total_ops") or 0)
    placeholder_overlap_tokens = int(meta.get("placeholder_kv_prefill_overlap_tokens") or 0)
    placeholder_copy_method = meta.get("placeholder_knn_copy_method") or "none"
    placeholder_copy_errors = int(meta.get("placeholder_anchor_pool_copy_error_count") or 0)
    if protected_tokens > 0 and consumed_count == 0 and device_hit_count > 0 and anchor_used:
        fast_path_status = "anchor_reuse_device_hit_consumed_counter_gap"
    elif protected_tokens > 0 and consumed_count == 0:
        if expired_tokens > 0:
            fast_path_status = "protected_not_consumed:ttl_or_steps_expired"
        elif large_gap_rejections > 0:
            fast_path_status = "protected_not_consumed:large_gap_rejected"
        elif anchor_gap_len or anchor_rope_delta:
            fast_path_status = "protected_not_consumed:position_mismatch"
        elif not anchor_used:
            fast_path_status = "protected_not_consumed:no_anchor_match"
        elif cached and prompt_tokens and cached / max(prompt_tokens, 1) > 0.98:
            fast_path_status = "protected_not_consumed:prefix_already_satisfied"
        else:
            fast_path_status = "protected_not_consumed:unknown"
    elif hint_count > 0 and protected_tokens == 0 and atkv_miss_count > 0:
        fast_path_status = "hint_text_mismatch_or_missing_entry"
    elif consumed_count > 0:
        fast_path_status = "consumed"
    elif device_hit_count > 0 or atkv_hit_count > 0:
        fast_path_status = "device_hit_without_consumed"
    else:
        fast_path_status = "no_fast_path"
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
        "codebase_prefetch_hint_count": hint_count,
        "codebase_prefetch_matched_tokens": matched_tokens,
        "codebase_prefetch_success_count": success_count,
        "codebase_prefetch_device_hit_count": device_hit_count,
        "agenttemplatekv_prefetch_hit_count": atkv_hit_count,
        "agenttemplatekv_prefetch_miss_count": atkv_miss_count,
        "agenttemplatekv_prefetch_protected_tokens": protected_tokens,
        "agenttemplatekv_prefetch_newly_protected_tokens": newly_protected_tokens,
        "agenttemplatekv_prefetch_consumed_count": consumed_count,
        "agenttemplatekv_prefetch_expired_tokens": expired_tokens,
        "agenttemplatekv_rejected_large_gap_count": large_gap_rejections,
        "lossy_anchor_match_used": anchor_used,
        "lossy_anchor_match_gap_len": anchor_gap_len,
        "lossy_anchor_rope_delta": anchor_rope_delta,
        # Placeholder k-NN reuse telemetry (PR 4).
        "placeholder_anchor_pool_hit_count": placeholder_hits,
        "placeholder_anchor_pool_miss_count": placeholder_misses,
        "placeholder_kv_prefill_matched_slots": placeholder_matched_slots,
        "placeholder_kv_prefill_skipped_tokens": placeholder_skipped_tokens,
        # Phase 2.4: cumulative trim overlap.
        "placeholder_kv_prefill_overlap_tokens": placeholder_overlap_tokens,
        "placeholder_knn_topk_similarity_mean": round(placeholder_sim_mean, 4),
        "placeholder_anchor_store_entry_count": placeholder_stored,
        "placeholder_anchor_store_skipped_low_f1_count": placeholder_skipped_low_f1,
        # Phase 2 cost-aware abort guard telemetry.
        "placeholder_anchor_pool_skipped_cost_count": placeholder_skipped_cost,
        # Phase 2.5: skip-high-overlap telemetry.
        "placeholder_knn_skipped_high_overlap_count": placeholder_skipped_high_overlap,
        # O7: short-new-tokens skip telemetry.
        "placeholder_knn_skipped_short_new_tokens_count": placeholder_skipped_short_new,
        # O8: high-span-overlap skip telemetry.
        "placeholder_knn_skipped_high_span_overlap_count": placeholder_skipped_high_span,
        # O10: high-new-token-ratio skip telemetry (cold prefix).
        "placeholder_knn_skipped_high_new_token_ratio_count": placeholder_skipped_high_new,
        # Phase 2.7 / O5: pre-rotated head K telemetry.
        "placeholder_knn_pre_rotated_hit_count": placeholder_pre_rotated_hit,
        "placeholder_knn_pre_rotated_miss_count": placeholder_pre_rotated_miss,
        # Phase 2.1 head-only RoPE rotation telemetry.
        "placeholder_knn_head_rotation_tokens": placeholder_head_rotation_tokens,
        "placeholder_knn_head_rotation_total_ops": placeholder_head_rotation_total_ops,
        # Phase 2.2 triton-tiled KV copy dispatcher telemetry.
        "placeholder_knn_copy_method": placeholder_copy_method,
        "placeholder_anchor_pool_copy_error_count": placeholder_copy_errors,
        "fast_path_status": fast_path_status,
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
        device_hits = [1.0 if int(r.get("codebase_prefetch_device_hit_count") or 0) > 0 else 0.0 for r in rs]
        consumed = [1.0 if int(r.get("agenttemplatekv_prefetch_consumed_count") or 0) > 0 else 0.0 for r in rs]
        protected = [float(r.get("agenttemplatekv_prefetch_protected_tokens") or 0) for r in rs]
        summary["|".join(map(str, key))] = {
            "n": len(rs),
            "avg_ttft_ms": safe_mean(ttft),
            "p50_ttft_ms": percentile(ttft, 0.5),
            "p90_ttft_ms": percentile(ttft, 0.9),
            "p99_ttft_ms": percentile(ttft, 0.99),
            "avg_cached_tokens": safe_mean(cached),
            "exact_hit_rate": safe_mean([1.0 if str(r.get("exact_hit")).lower() == "true" else 0.0 for r in rs]),
            "device_hit_rate": safe_mean(device_hits),
            "consumed_rate": safe_mean(consumed),
            "avg_protected_tokens": safe_mean(protected),
            "avg_token_f1_vs_baseline": safe_mean(f1),
            "fast_path_status_counts": dict(Counter(str(r.get("fast_path_status", "")) for r in rs)),
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
            f"p50={item['p50_ttft_ms']:.1f} ms, p90={item['p90_ttft_ms']:.1f} ms, "
            f"exact hit={item['exact_hit_rate']:.2f}, device hit={item['device_hit_rate']:.2f}, "
            f"consumed={item['consumed_rate']:.2f}, protected={item['avg_protected_tokens']:.1f}, "
            f"F1={item['avg_token_f1_vs_baseline']:.4f}, status={item['fast_path_status_counts']}"
        )
    lines += ["", "## Exact-Reuse Speedup vs Prefix", ""]
    by_dims: dict[tuple[str, str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for key, item in summary.items():
        experiment, mode, max_file_chars, max_tokens, agent_count, segment_count = key.split("|", 5)
        by_dims[(experiment, max_file_chars, max_tokens, agent_count, segment_count)][mode] = item
    for dims, modes in sorted(by_dims.items()):
        prefix = modes.get("prefix_cache_only")
        exact = modes.get("exact_reuse_plus_code_hints")
        if not prefix or not exact or float(exact["p50_ttft_ms"]) <= 0:
            continue
        p50_speedup = float(prefix["p50_ttft_ms"]) / float(exact["p50_ttft_ms"])
        p90_speedup = (
            float(prefix["p90_ttft_ms"]) / float(exact["p90_ttft_ms"])
            if float(exact["p90_ttft_ms"]) > 0
            else 0.0
        )
        lines.append(
            f"- `{dims}`: p50={p50_speedup:.2f}x, p90={p90_speedup:.2f}x "
            f"(prefix={prefix['p50_ttft_ms']:.1f}/{prefix['p90_ttft_ms']:.1f} ms, "
            f"exact+hints={exact['p50_ttft_ms']:.1f}/{exact['p90_ttft_ms']:.1f} ms)"
        )
    (out_dir / "TTFT_STRESS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "TTFT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    failed_reason = ""
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
    except Exception as exc:
        failed_reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        kill_port(args.port)
        if rows and failed_reason:
            partial_summary = {
                "model": args.model,
                "manifest": str(args.manifest),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "git_commit": git_commit(),
                "command": " ".join(sys.argv),
                "environment": {
                    "python": args.python,
                    "port": args.port,
                    "mem_fraction_static": args.mem_fraction_static,
                    "max_total_tokens": args.max_total_tokens,
                    "lossy_max_zero_gap": args.lossy_max_zero_gap,
                    "hierarchical_cache": not args.disable_hierarchical_cache,
                    "hicache_storage_backend": args.hicache_storage_backend or "disabled",
                    "gpu": "RTX 4090 24GB assumed",
                },
                "length_buckets": args.length_buckets,
                "max_token_settings": args.max_token_settings,
                "agent_counts": args.agent_counts,
                "segment_counts": args.segment_counts,
                "agent_length_buckets": args.agent_length_buckets,
                "rows": rows,
                "failed_reason": failed_reason,
            }
            partial_summary["group_summary"] = summarize_rows(rows)
            (args.out_dir / "summary.json").write_text(json.dumps(partial_summary, indent=2), encoding="utf-8")
            write_csv(args.out_dir / "ttft_stress_table.csv", rows)
            write_csv(args.out_dir / "ttft_table.csv", rows)
            write_report(args.out_dir, rows, partial_summary["group_summary"])

    summary = {
        "model": args.model,
        "manifest": str(args.manifest),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "environment": {
            "python": args.python,
            "port": args.port,
            "mem_fraction_static": args.mem_fraction_static,
            "max_total_tokens": args.max_total_tokens,
            "lossy_max_zero_gap": args.lossy_max_zero_gap,
            "hierarchical_cache": not args.disable_hierarchical_cache,
            "hicache_storage_backend": args.hicache_storage_backend or "disabled",
            "gpu": "RTX 4090 24GB assumed",
        },
        "length_buckets": args.length_buckets,
        "max_token_settings": args.max_token_settings,
        "agent_counts": args.agent_counts,
        "segment_counts": args.segment_counts,
        "agent_length_buckets": args.agent_length_buckets,
        "rows": rows,
        "failed_reason": failed_reason,
    }
    group_summary = summarize_rows(rows)
    summary["group_summary"] = group_summary
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.out_dir / "ttft_stress_table.csv", rows)
    write_csv(args.out_dir / "ttft_table.csv", rows)
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
    parser.add_argument("--max-total-tokens", type=int, default=131072,
                        help="Default 131072 (2x v25 default 65536). The "
                             "larger cache reduces radix-tree LRU eviction "
                             "between the warm_planner's pre-warm writes "
                             "and the placeholder_knn_reuse agent reads "
                             "for agent_count=5, lifting it from 0.44x "
                             "(v25) to a higher floor. Does not affect "
                             "agents=1-3 which were already ≥ 1x.")
    parser.add_argument("--mem-fraction-static", type=float, default=0.78)
    parser.add_argument("--lossy-max-zero-gap", type=int, default=512)
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
