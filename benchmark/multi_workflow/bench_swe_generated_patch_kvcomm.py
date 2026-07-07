#!/usr/bin/env python3
"""Generate and test SWE-bench patches with AgentTemplateKV-style reuse.

KVFlow/KVCOMM remain reference baselines in this harness. The AgentTemplateKV
layout keeps shared codebase blocks stable and early in the prompt so exact
code reuse and device-first prefetch can translate into end-to-end speed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer


PROJECT = Path(__file__).resolve().parents[2]
MAS_SRC = PROJECT.parent / "MAScoder" / "src"
for entry in (str(MAS_SRC), str(PROJECT), str(PROJECT / "python")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mascoder.code_anchor import build_code_anchor_payload, compute_exact_content_signature


DEFAULT_MODEL = "/home/gfy/models/Qwen2.5-3B-Instruct"
DEFAULT_PYTHON = "/home/gfy/.conda/envs/sglang-kvflow/bin/python"
DEFAULT_DATASET = PROJECT / "results" / "repo_level_datasets" / "swe_verified_3_instances.json"
DEFAULT_MANIFEST = PROJECT / "results" / "repo_level_datasets" / "manifest.json"
OUT_DIR = PROJECT / "results" / "swe_generated_patch_kvcomm"


@dataclass
class CodeSegment:
    name: str
    text: str

    @property
    def signature(self) -> str:
        return compute_exact_content_signature(self.text)


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(shlex.quote(x) for x in cmd)}")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout)


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
    # Enable KV allocator defrag so evict_from_tree_cache can reclaim
    # release_pages into free_pages on alloc failure. Without this, the
    # allocator's alloc() only looks at free_pages, while evicted tokens
    # land in release_pages; the alloc_with_defrag fallback is gated by
    # this env var (default off in upstream SGLang). See
    # python/sglang/srt/mem_cache/allocator.py:173.
    if args.kv_allocator_defrag:
        env["SGLANG_KV_ALLOCATOR_DEFRAG"] = "1"
    if args.enable_placeholder_knn:
        # v44 placeholder k-NN body: opt-in via env var, default off.
        # When on, the server-side `_try_placeholder_knn_lossy_match_body`
        # fires for high-similarity anchors (cosine >= SGLANG_PLACEHOLDER_KNN_MIN_COSINE,
        # top-K from SGLANG_PLACEHOLDER_KNN_TOPK) and copies head KV segments
        # instead of forcing dense prefill.
        # See python/sglang/srt/mem_cache/radix_cache.py:2340-2342.
        env["SGLANG_PLACEHOLDER_KNN_MATCH"] = "1"
        env["SGLANG_PLACEHOLDER_KNN_PRE_ROTATE_DELTAS"] = "1"
        env["SGLANG_PLACEHOLDER_KNN_TOPK"] = str(args.placeholder_knn_topk)
        env["SGLANG_PLACEHOLDER_KNN_MIN_COSINE"] = str(args.placeholder_knn_min_cosine)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
        str(args.chunked_prefill_size),
        "--max-prefill-tokens",
        str(args.max_prefill_tokens),
        "--cpu-offload-gb",
        str(args.cpu_offload_gb),
    ]
    if args.disable_overlap_schedule:
        # Serializes prefill batches so the previous request's leaves
        # release their lock_ref=3 before the next starts, making
        # evictable_leaves non-empty for the next prefill. This is the
        # unblock path for the transient lock-pressure OOM documented
        # in results/pass100_attempt/REPORT.md (Step 2.4).
        cmd.append("--disable-overlap-schedule")
    if args.force_evict:
        # SGLANG_RADIX_FORCE_EVICT=1 makes common.py:evict_from_tree_cache
        # retry with force=True when normal evict() freed 0 tokens.
        # RadixCache._force_evict_locked then frees leaves regardless of
        # lock_ref, recovering from transient lock-pressure OOMs. See
        # results/pass100_attempt/REPORT.md Step 2.10 for the empirical
        # finding. Default off (matches upstream SGLang).
        env["SGLANG_RADIX_FORCE_EVICT"] = "1"
    if args.max_running_requests is not None:
        cmd += ["--max-running-requests", str(args.max_running_requests)]
    cmd += [
        "--enable-cache-report",
        "--disable-cuda-graph",
        "--allow-auto-truncate",
        "--log-level",
        "error",
    ]
    return subprocess.Popen(cmd, cwd=str(PROJECT), env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)


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
        timeout=aiohttp.ClientTimeout(total=600),
    ) as resp:
        body = await resp.json()
    return {"elapsed_ms": now_ms() - start, "body": body}


async def post_chat_stream(session: aiohttp.ClientSession, port: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Streaming variant of post_chat that records TTFT (time-to-first-token).

    Returns:
        dict with elapsed_ms (E2E), e2e_ms (alias of elapsed_ms), ttft_ms
        (first content delta in ms; falls back to e2e_ms if no stream delta),
        text, cached_tokens, prompt_tokens, body (last chunk), metadata.
    """
    start = now_ms()
    ttft_ms: float | None = None
    text = ""
    final_body: dict[str, Any] = {}
    cached_tokens = 0
    prompt_tokens = int(payload.get("prompt_tokens_local", 0) or 0)
    meta: dict[str, Any] = {}
    try:
        async with session.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={**payload, "stream": True, "stream_options": {"include_usage": True}},
            timeout=aiohttp.ClientTimeout(total=600),
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
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        # Fall back to non-streaming call so the benchmark can still record
        # a row instead of dropping the case.
        try:
            async with session.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                final_body = await resp.json()
        except Exception:
            final_body = {"error": repr(exc)}
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


async def post_chat_optional_stream(
    session: aiohttp.ClientSession,
    port: int,
    payload: dict[str, Any],
    emit_ttft: bool,
) -> dict[str, Any]:
    """Dispatch to post_chat or post_chat_stream based on the --emit-ttft flag.

    When emit_ttft is False, behavior is identical to the original post_chat
    (elapsed_ms only, body only). When emit_ttft is True, returns the streaming
    payload which also carries ttft_ms and e2e_ms.
    """
    if emit_ttft:
        return await post_chat_stream(session, port, payload)
    return await post_chat(session, port, payload)


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


def token_bounds_for_text(tokenizer: Any, full_text: str, segment_text: str, char_start: int = 0) -> tuple[int, int, int]:
    char_pos = full_text.find(segment_text, char_start)
    if char_pos < 0:
        raise ValueError("segment text not found in prompt")
    start = len(tokenizer.encode(full_text[:char_pos], add_special_tokens=False))
    end = len(tokenizer.encode(full_text[: char_pos + len(segment_text)], add_special_tokens=False))
    return start, end, char_pos + len(segment_text)


def build_anchor_fields(tokenizer: Any, messages: list[dict[str, str]], segments: list[CodeSegment]) -> dict[str, Any]:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    token_spans = []
    anchor_spans = []
    char_cursor = 0
    for segment in segments:
        start, end, char_cursor = token_bounds_for_text(tokenizer, prompt, segment.text, char_cursor)
        payload = build_code_anchor_payload(segment.text, language="python")
        anchor_sig = sha1_short(segment.name + ":" + segment.signature)
        anchor_spans.append(
            {
                "anchor_type": "code_base",
                "signature": anchor_sig,
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
                "signature": anchor_sig,
                "content_signature": segment.signature,
                "start_token": start,
                "end_token": end,
                "segment_name": segment.name,
            }
        )
    return {
        "code_anchor_signature": sha1_short("|".join(s.signature for s in segments)),
        "code_content_signature": sha1_short("joined:" + "|".join(s.signature for s in segments)),
        "code_anchor_spans": anchor_spans,
        "code_anchor_token_spans": token_spans,
    }


def prompt_telemetry(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {
        "target_prompt_sha1": sha1_short(prompt),
        "target_prompt_chars": len(prompt),
    }


def graph_reuse_segments_in_prompt(
    prompt_segments: list[CodeSegment],
    graph_segments: list[CodeSegment],
) -> list[CodeSegment]:
    """Map graph-selected evidence back to text already present in the prompt.

    This keeps the graph-aware mode prompt-fair: the prompt still contains the
    same whole-file code_base as lossless/lossy; graph evidence only selects
    which prompt-resident anchors are exposed to the reuse runtime.
    """
    if not graph_segments:
        return []
    selected: list[CodeSegment] = []
    seen: set[tuple[str, str]] = set()
    for graph_segment in graph_segments:
        graph_path, _, _ = parse_graph_segment_name(graph_segment.name)
        graph_text = graph_segment.text.strip()
        for prompt_segment in prompt_segments:
            key = (prompt_segment.name, prompt_segment.signature)
            if key in seen or prompt_segment.name != graph_path:
                continue
            if graph_text and (graph_text in prompt_segment.text or prompt_segment.text in graph_text):
                selected.append(prompt_segment)
                seen.add(key)
                break
    if selected:
        return selected
    graph_paths = {parse_graph_segment_name(segment.name)[0] for segment in graph_segments}
    for prompt_segment in prompt_segments:
        key = (prompt_segment.name, prompt_segment.signature)
        if prompt_segment.name in graph_paths and key not in seen:
            selected.append(prompt_segment)
            seen.add(key)
    return selected


GRAPH_SEGMENT_MARKER = "::graph::"


def encode_graph_segment_name(path: str, target_symbol: str, bundle_type: str) -> str:
    symbol = str(target_symbol or "").replace("::", ".").strip()
    bundle = str(bundle_type or "").replace("::", "_").strip()
    if not symbol and not bundle:
        return path
    return f"{path}{GRAPH_SEGMENT_MARKER}{symbol}::{bundle}"


def parse_graph_segment_name(name: str) -> tuple[str, str, str]:
    if GRAPH_SEGMENT_MARKER not in name:
        return name, "", ""
    path, rest = name.split(GRAPH_SEGMENT_MARKER, 1)
    parts = rest.rsplit("::", 1)
    if len(parts) == 1:
        return path, parts[0], ""
    return path, parts[0], parts[1]


def build_codebase_prefetch_hints(segments: list[CodeSegment], target_agent: str = "implementer") -> list[dict[str, Any]]:
    hints = []
    for idx, segment in enumerate(segments, 1):
        hints.append(
            {
                "code_base_id": f"code_base{idx}:{segment.name}",
                "content_signature": segment.signature,
                "target_agent": target_agent,
                "steps_to_use": 1,
                "priority": 1,
                "match_required": "exact_code_content_signature",
                "text": segment.text,
            }
        )
    return hints


def patch_paths(patch_text: str) -> list[str]:
    paths = []
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4 and parts[2].startswith("a/"):
                paths.append(parts[2][2:])
    return paths


def load_instance_id_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    if text.startswith("["):
        return {str(item) for item in json.loads(text)}
    return {line.strip() for line in text.splitlines() if line.strip()}


def load_graph_bundle_segments(args: argparse.Namespace) -> dict[str, list[CodeSegment]]:
    """Load exact code graph bundles as optional patch-task code_base segments."""
    if not args.enable_graph_aware_lossy or not args.code_graph_bundle_manifest.exists():
        return {}

    by_case: dict[str, list[CodeSegment]] = {}
    seen: set[tuple[str, str, str]] = set()
    for line in args.code_graph_bundle_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("agent_role") != args.graph_bundle_role:
            continue
        if row.get("bundle_type") != args.graph_bundle_policy:
            continue
        instance_id = str(row.get("instance_id", ""))
        target_file = str(row.get("target_file", "")).strip()
        target_symbol = str(row.get("target_symbol", "")).strip()
        bundle_type = str(row.get("bundle_type", "")).strip()
        bundle_text = str(row.get("bundle_text", "")).rstrip()
        if not instance_id or not target_file or not bundle_text:
            continue
        if args.max_graph_bundle_chars and len(bundle_text) > args.max_graph_bundle_chars:
            bundle_text = bundle_text[: args.max_graph_bundle_chars].rstrip()
        key = (instance_id, target_file, sha1_short(bundle_text))
        if key in seen:
            continue
        seen.add(key)
        segment_name = encode_graph_segment_name(target_file, target_symbol, bundle_type)
        by_case.setdefault(instance_id, []).append(CodeSegment(segment_name, bundle_text))

    for instance_id, segments in by_case.items():
        by_case[instance_id] = segments[: args.graph_bundles_per_case]
    return by_case


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    instance_filter = load_instance_id_filter(args.instance_id_file)
    if instance_filter is not None:
        rows = [row for row in rows if row.get("instance_id") in instance_filter]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_by_id = {row["instance_id"]: row for row in manifest["samples"]}
    graph_segments_by_case = load_graph_bundle_segments(args)
    cases = []
    for row in rows[args.start_index : args.start_index + args.max_cases]:
        target_paths = patch_paths(row.get("patch", ""))
        sample = manifest_by_id.get(row["instance_id"], {})
        segment_paths = list(dict.fromkeys(target_paths + [f["path"] for f in sample.get("files", [])]))
        repo_dir = PROJECT / "results" / "swebench_local_envs" / "repos" / row["instance_id"]
        reset_repo_to_base(row, repo_dir)
        segments = []
        for path in segment_paths[: args.files_per_case]:
            file_path = repo_dir / path
            if not file_path.exists():
                continue
            text = file_path.read_text(encoding="utf-8", errors="replace")
            if args.max_file_chars and len(text) > args.max_file_chars:
                text = text[: args.max_file_chars]
            segments.append(CodeSegment(path, text.rstrip()))
        if segments:
            graph_segments = graph_segments_by_case.get(row["instance_id"], [])
            cases.append(
                {
                    "instance": row,
                    "segments": segments,
                    "graph_segments": graph_segments,
                    "target_paths": target_paths,
                }
            )
    return cases


def reset_repo_to_base(instance: dict[str, Any], repo_dir: Path):
    if not (repo_dir / ".git").exists():
        return
    run(["git", "checkout", "--force", instance["base_commit"]], cwd=repo_dir, timeout=120)
    run(["git", "clean", "-fdx"], cwd=repo_dir, timeout=120)


def build_codebase_block(segments: list[CodeSegment]) -> list[str]:
    body = []
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
    return body


def build_messages(
    instance: dict[str, Any],
    segments: list[CodeSegment],
    mode_label: str,
    output_schema: str,
    prompt_layout: str = "agenttemplatekv",
) -> list[dict[str, str]]:
    if output_schema == "json-edit":
        output_instruction = [
            "Return only compact JSON with this exact schema:",
            '{"edits":[{"path":"repo/relative/path.py","search":"exact original substring","replace":"replacement substring"}]}',
            "The search field must be copied exactly from one provided code_base section.",
            "The replace field must contain complete real code, never placeholders.",
            "Never use ellipsis (`...`) in code unless the original file already contains that exact ellipsis.",
            "Do not edit tests. Only edit implementation files.",
            "Do not include markdown fences, comments, reasoning, analysis, or prose.",
        ]
        system_content = (
            "You are a precise software maintenance agent. "
            "Your entire response must be valid JSON matching the requested edit schema and nothing else."
        )
    else:
        output_instruction = [
            "Return only a unified git diff that starts with 'diff --git'.",
            "Do not include markdown fences, comments, reasoning, analysis, or prose.",
            "Use exact file paths from the provided code_base sections.",
            "Keep the patch minimal and syntactically valid for `git apply`.",
        ]
        system_content = (
            "You are a precise software maintenance agent. "
            "Your entire response must be a valid unified git diff and nothing else."
        )
    shared_task = [
        "## Issue",
        instance.get("problem_statement", "").strip(),
        "",
        "## FAIL_TO_PASS tests",
        instance.get("FAIL_TO_PASS", "").strip(),
        "",
        "## Test patch",
        instance.get("test_patch", "").strip()[:6000],
        "",
        "## Allowed implementation paths",
        "\n".join(f"- {segment.name}" for segment in segments),
        "",
        "## Important constraints",
        "- Only implementation files should be edited.",
        "- Use the Test patch to infer the expected behavior, then edit the implementation path that makes that test pass.",
        "- Do not edit unrelated guards, setup checks, imports, or tests.",
        "- Builtin exceptions such as ValueError do not need imports.",
        "- Search strings must be exact, unique substrings from code_base.",
        "- Replacement strings must preserve the surrounding original code and add only the minimal fix.",
        "- Do not use placeholders, pseudo-code, ellipsis, or abbreviated function signatures.",
        "",
        f"## Agent step: {mode_label}",
        "",
    ]
    if prompt_layout == "legacy":
        body = [
            "You are fixing a real SWE-bench repository issue.",
            "",
            *output_instruction,
            "",
            *shared_task,
            *build_codebase_block(segments),
        ]
    else:
        body = [
            "You are fixing a real SWE-bench repository issue.",
            "",
            *build_codebase_block(segments),
            "## Agent instruction",
            *output_instruction,
            "",
            *shared_task,
        ]
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {"role": "user", "content": "\n".join(body)},
    ]


def build_repair_messages(
    instance: dict[str, Any],
    segments: list[CodeSegment],
    previous_output: str,
    previous_diff: str,
    apply_error: str,
    mode_label: str,
    output_schema: str,
    prompt_layout: str = "agenttemplatekv",
) -> list[dict[str, str]]:
    if output_schema == "json-edit":
        hard_requirements = [
            "- Return only compact valid JSON.",
            '- Use schema: {"edits":[{"path":"repo/relative/path.py","search":"exact original substring","replace":"replacement substring"}]}',
            "- The search field must exactly match the provided code_base content.",
            "- The replacement must be complete real code with no placeholders or ellipsis.",
            "- Do not edit tests.",
            "- Do not include markdown fences, reasoning, analysis, or prose.",
        ]
        previous_label = "Previous raw output"
        system_content = "You repair invalid JSON edit outputs. Return valid JSON only."
    else:
        hard_requirements = [
            "- Return only a unified git diff that starts with 'diff --git'.",
            "- Do not include markdown fences, reasoning, analysis, or prose.",
            "- Use exact file paths from the provided code_base sections.",
            "- Make the hunk line numbers consistent with the provided files.",
        ]
        previous_label = "Previous extracted diff"
        system_content = (
            "You repair invalid patches. "
            "Your entire response must be a valid unified git diff and nothing else."
        )
    repair_tail = [
        "Hard requirements:",
        *hard_requirements,
        "",
        "## Issue",
        instance.get("problem_statement", "").strip(),
        "",
        "## Apply check error",
        apply_error.strip()[-3000:],
        "",
        f"## Agent step: repair {mode_label}",
        "",
        f"## {previous_label}",
        previous_diff.strip()[-6000:],
        "",
        "## Previous raw output",
        previous_output.strip()[-6000:],
        "",
    ]
    if prompt_layout == "legacy":
        body = [
            "Your previous response was not an applyable unified git diff.",
            "Repair it now.",
            "",
            *repair_tail,
            *build_codebase_block(segments),
        ]
    else:
        body = [
            "Your previous response was not an applyable unified git diff.",
            "Repair it now.",
            "",
            *build_codebase_block(segments),
            *repair_tail,
        ]
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {"role": "user", "content": "\n".join(body)},
    ]


def make_payload(
    args: argparse.Namespace,
    tokenizer: Any,
    messages: list[dict[str, str]],
    segments: list[CodeSegment],
    reuse_mode: str,
    include_anchor: bool,
    include_codebase_prefetch: bool,
    salt: str,
) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "return_cached_tokens_details": True,
        "reuse_mode": reuse_mode,
        "lossy_alignment_method": "kvcomm",
        "template_task_family": "coding_mas_swe_patch",
        "cache_salt": salt,
    }
    if include_codebase_prefetch:
        payload["codebase_prefetch_hints"] = build_codebase_prefetch_hints(segments)
    if include_anchor:
        payload.update(build_anchor_fields(tokenizer, messages, segments))
    return payload


def _drop_repetitive_hunks(diff: str, max_hunks_per_file: int = 4,
                           similarity_threshold: float = 0.7) -> str:
    """Drop hunks beyond a per-file cap and any hunk whose body is
    >similarity_threshold identical to an earlier kept hunk in the same
    file. Defends against model repetition failure mode (R33 — Qwen-Coder
    emitted 11 near-identical hunks, last truncated mid-token). Default
    params only affect degenerate patches; well-formed 1–3 hunk patches
    are unchanged."""
    if not diff or max_hunks_per_file <= 0:
        return diff
    sections = re.split(r"(?=^diff --git )", diff, flags=re.M)
    out: list[str] = []
    for section in sections:
        if not section.startswith("diff --git "):
            out.append(section)
            continue
        parts = re.split(r"(?=^@@ )", section, flags=re.M)
        header_block = parts[0]
        hunks = parts[1:]
        if len(hunks) <= max_hunks_per_file:
            out.append(section)
            continue
        kept = list(hunks[:max_hunks_per_file])
        seen_bodies = [hunk.splitlines()[1:] for hunk in kept]
        for hunk in hunks[max_hunks_per_file:]:
            body = hunk.splitlines()[1:]
            if any(
                difflib.SequenceMatcher(None, body, prior).ratio() > similarity_threshold
                for prior in seen_bodies
            ):
                continue  # drop repetitive hunk
            kept.append(hunk)
            seen_bodies.append(body)
        out.append(header_block + "".join(kept))
    return "".join(out)


def extract_unified_diff(text: str, max_hunks_per_file: int = 4,
                         similarity_threshold: float = 0.7) -> str:
    """Extract a unified diff from model output, then drop repetitive hunks.

    Default cap (4 hunks per file) is defensive against the R33 model
    repetition failure mode where Qwen-Coder emitted 11 near-identical
    hunks in one file and was truncated mid-token. Pass
    max_hunks_per_file=0 to disable.
    """
    fenced = re.search(r"```(?:diff|patch)?\s*(diff --git .*?)```", text, re.S)
    if fenced:
        raw = fenced.group(1).strip() + "\n"
    else:
        idx = text.find("diff --git ")
        if idx < 0:
            return ""
        raw = text[idx:].strip() + "\n"
    return _drop_repetitive_hunks(raw, max_hunks_per_file, similarity_threshold)


FIRST_HUNK_RE = re.compile(
    r"^diff --git a/(?P<path>[^\s]+) b/(?P<path2>[^\s]+)\n"
    r"---[^\n]*\n\+\+\+[^\n]*\n"
    r"@@ -(?P<old_start>\d+)[^\n]*\n"
    r"(?:[^\n]*\n){0,5}",
    re.M,
)


def first_hunk_summary(diff_text: str) -> dict:
    """Extract first hunk metadata: target file and old-start line.

    Returns {"extracted": False} when the diff is empty or unparseable.
    """
    if not diff_text:
        return {"extracted": False}
    m = FIRST_HUNK_RE.search(diff_text)
    if not m:
        return {"extracted": False}
    return {
        "extracted": True,
        "target_path": m.group("path"),
        "old_start_line": int(m.group("old_start")),
    }


def first_hunk_vs_gold(model_patch: str, gold_patch: str) -> dict:
    """Weak correctness signal — does model's first hunk target gold's file/line?

    Useful as a coarse signal when apply_check cannot run (env broken).
    A True file_match means the model understood the issue domain even if
    it picked the wrong sub-routine.
    """
    m = first_hunk_summary(model_patch)
    g = first_hunk_summary(gold_patch)
    if not (m["extracted"] and g["extracted"]):
        return {"comparable": False}
    return {
        "comparable": True,
        "model_path": m["target_path"],
        "gold_path": g["target_path"],
        "file_match": m["target_path"] == g["target_path"],
        "model_line": m["old_start_line"],
        "gold_line": g["old_start_line"],
        "line_delta_abs": abs(m["old_start_line"] - g["old_start_line"]),
    }


def extract_json_object(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1].strip()
    return ""


def synthesize_patch_from_json_edits(instance_id: str, text: str) -> tuple[str, dict[str, Any]]:
    repo_dir = PROJECT / "results" / "swebench_local_envs" / "repos" / instance_id
    raw_json = extract_json_object(text)
    if not raw_json:
        return "", {"ok": False, "error": "no json object extracted", "edits": 0}
    try:
        payload = json.loads(raw_json)
    except Exception as exc:
        return "", {"ok": False, "error": f"json parse failed: {exc}", "edits": 0}
    edits = payload.get("edits", [])
    if not isinstance(edits, list) or not edits:
        return "", {"ok": False, "error": "missing edits list", "edits": 0}

    before_after: dict[str, tuple[str, str]] = {}
    seen_edits: set[tuple[str, str, str]] = set()
    for edit in edits:
        if not isinstance(edit, dict):
            return "", {"ok": False, "error": "edit is not an object", "edits": len(edits)}
        path = str(edit.get("path", "")).strip()
        search = edit.get("search", "")
        replace = edit.get("replace", "")
        path = path.lstrip("/")
        if path.startswith("repo/"):
            path = path[len("repo/") :]
        if path.startswith("a/") or path.startswith("b/"):
            path = path[2:]
        if "/tests/" in f"/{path}" or path.startswith("tests/") or path.startswith("test_"):
            continue
        if not path or not isinstance(search, str) or not isinstance(replace, str):
            return "", {"ok": False, "error": f"invalid edit fields for {path}", "edits": len(edits)}
        if "..." in replace and "..." not in search:
            return "", {"ok": False, "error": f"placeholder ellipsis in replacement for {path}", "edits": len(edits)}
        edit_key = (path, search, replace)
        if edit_key in seen_edits:
            continue
        seen_edits.add(edit_key)
        file_path = repo_dir / path
        if not file_path.exists():
            return "", {"ok": False, "error": f"file not found: {path}", "edits": len(edits)}
        original, current = before_after.get(path, (file_path.read_text(encoding="utf-8", errors="replace"), None))
        if current is None:
            current = original
        if search not in current:
            return "", {"ok": False, "error": f"search not found in {path}", "edits": len(edits)}
        current = current.replace(search, replace, 1)
        before_after[path] = (original, current)

    parts: list[str] = []
    for path, (original, current) in before_after.items():
        if original == current:
            continue
        diff_lines = list(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                current.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        if diff_lines:
            parts.append(f"diff --git a/{path} b/{path}\n")
            parts.extend(diff_lines)
            if not parts[-1].endswith("\n"):
                parts[-1] += "\n"
    patch = "".join(parts)
    return patch, {"ok": bool(patch.strip()), "error": "" if patch.strip() else "empty synthesized diff", "edits": len(edits)}


def test_target_args(instance_id: str) -> list[str]:
    if instance_id == "django__django-10097":
        return ["--skip-pre-install", "--test-target", "validators.tests.TestSimpleValidators"]
    if instance_id == "matplotlib__matplotlib-13989":
        return ["--skip-pre-install"]
    return []


def evaluate_candidate(args: argparse.Namespace, instance_id: str, patch_path: Path) -> dict[str, Any]:
    cmd = [
        args.python,
        str(PROJECT / "benchmark" / "multi_workflow" / "setup_swebench_local_env.py"),
        "--dataset",
        str(args.dataset),
        "--instance-id",
        instance_id,
        "--mode",
        "candidate",
        "--candidate-patch",
        str(patch_path.resolve()),
        "--timeout",
        str(args.eval_timeout),
    ] + test_target_args(instance_id)
    if args.recreate_candidate_env:
        cmd.append("--recreate-env")
    result = run(cmd, cwd=PROJECT, timeout=args.eval_timeout + 120)
    return {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-2000:],
    }


def check_patch_apply(instance_id: str, patch_path: Path) -> dict[str, Any]:
    repo_dir = PROJECT / "results" / "swebench_local_envs" / "repos" / instance_id
    if not repo_dir.exists():
        return {
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": f"repo not found: {repo_dir}",
        }
    result = run(["git", "apply", "--check", str(patch_path.resolve())], cwd=repo_dir, timeout=120)
    return {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-4000:],
    }


async def run_benchmark(args: argparse.Namespace):
    global OUT_DIR
    OUT_DIR = args.out_dir if args.out_dir.is_absolute() else PROJECT / args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cases = load_cases(args)
    results = []
    kill_port(args.port)
    await asyncio.sleep(1)
    proc = launch_server(args)
    try:
        if not await wait_ready(args.port, args.server_timeout):
            raise RuntimeError(f"sglang server did not become ready; see {OUT_DIR / 'sglang_server.log'}")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
            for case in cases:
                instance = case["instance"]
                instance_id = instance["instance_id"]
                segments = case["segments"]
                graph_segments = case.get("graph_segments", [])
                case_dir = OUT_DIR / instance_id
                case_dir.mkdir(parents=True, exist_ok=True)
                reset_repo_to_base(instance, PROJECT / "results" / "swebench_local_envs" / "repos" / instance_id)

                warm_messages = build_messages(
                    instance,
                    segments,
                    "planner warmup; identify files and reusable code bases",
                    args.output_schema,
                    args.prompt_layout,
                )
                await post_chat_optional_stream(
                    session,
                    args.port,
                    make_payload(
                        args,
                        tokenizer,
                        warm_messages,
                        segments,
                        "lossless",
                        True,
                        False,
                        f"warm:{instance_id}",
                    ),
                    args.emit_ttft,
                )
                if args.enable_graph_aware_lossy and graph_segments:
                    graph_anchor_segments = graph_reuse_segments_in_prompt(segments, graph_segments)
                    graph_warm_messages = build_messages(
                        instance,
                        segments,
                        "planner warmup; identify files and reusable code bases",
                        args.output_schema,
                        args.prompt_layout,
                    )
                    if graph_anchor_segments:
                        await post_chat_optional_stream(
                            session,
                            args.port,
                            make_payload(
                                args,
                                tokenizer,
                                graph_warm_messages,
                                graph_anchor_segments,
                                "lossless",
                                True,
                                False,
                                f"graphwarm:{instance_id}:{args.graph_bundle_policy}",
                            ),
                            args.emit_ttft,
                        )

                mode_results = []
                mode_specs = [
                    ("lossless", "lossless", False, False, segments, segments, "file_context"),
                    ("lossy", "lossy", True, False, segments, segments, "file_context"),
                    ("lossy_prefetch", "lossy", True, True, segments, segments, "file_context"),
                ]
                if args.enable_placeholder_knn:
                    # v44 placeholder k-NN body: same anchor payload as lossy,
                    # server-side SGLANG_PLACEHOLDER_KNN_MATCH=1 makes the body
                    # fire for high-sim anchors instead of forcing dense prefill.
                    mode_specs.append(("placeholder_knn_lossy", "lossy", True, False, segments, segments, "file_context"))
                if args.enable_graph_aware_lossy and graph_segments:
                    graph_anchor_segments = graph_reuse_segments_in_prompt(segments, graph_segments)
                    mode_specs.append(("graph_aware_lossy", "lossy", True, False, segments, graph_anchor_segments, "code_graph_bundle"))
                for mode, reuse_mode, include_anchor, include_codebase_prefetch, prompt_segments, anchor_segments, reuse_selection_source in mode_specs:
                    messages = build_messages(
                        instance,
                        prompt_segments,
                        "implementation target",
                        args.output_schema,
                        args.prompt_layout,
                    )
                    target_prompt_meta = prompt_telemetry(tokenizer, messages)
                    raw_path = case_dir / f"{mode}_output.txt"
                    patch_path = case_dir / f"{mode}.patch"
                    try:
                        if include_anchor and not anchor_segments:
                            raise ValueError("no prompt-resident anchor segments for mode")
                        response = await post_chat_optional_stream(
                            session,
                            args.port,
                            make_payload(
                                args,
                                tokenizer,
                                messages,
                                anchor_segments,
                                reuse_mode,
                                include_anchor,
                                include_codebase_prefetch,
                                f"gen:{instance_id}:{mode}",
                            ),
                            args.emit_ttft,
                        )
                        output = extract_text(response["body"])
                        if args.output_schema == "json-edit":
                            diff, synthesis = synthesize_patch_from_json_edits(instance_id, output)
                        else:
                            diff = extract_unified_diff(output, args.max_hunks_per_file,
                                                        args.hunk_similarity_threshold)
                            synthesis = {"ok": bool(diff.strip()), "error": "" if diff.strip() else "no diff extracted", "edits": None}
                        raw_path.write_text(output, encoding="utf-8")
                        patch_path.write_text(diff, encoding="utf-8")
                        apply_check = (
                            check_patch_apply(instance_id, patch_path)
                            if diff.strip()
                            else {
                                "returncode": None,
                                "stdout_tail": "",
                                "stderr_tail": "no diff extracted",
                            }
                        )
                        repair_attempted = False
                        repair_elapsed_ms = None
                        repair_ttft_ms = None
                        if (
                            args.repair_attempts > 0
                            and (not diff.strip() or apply_check.get("returncode") not in (0, None))
                        ):
                            repair_attempted = True
                            repair_messages = build_repair_messages(
                                instance,
                                prompt_segments,
                                output,
                                diff,
                                apply_check.get("stderr_tail", ""),
                                "implementation target",
                                args.output_schema,
                                args.prompt_layout,
                            )
                            repair_response = await post_chat_optional_stream(
                                session,
                                args.port,
                                make_payload(
                                    args,
                                    tokenizer,
                                    repair_messages,
                                    anchor_segments,
                                    reuse_mode,
                                    include_anchor,
                                    include_codebase_prefetch,
                                    f"repair:{instance_id}:{mode}",
                                ),
                                args.emit_ttft,
                            )
                            repair_output = extract_text(repair_response["body"])
                            if args.output_schema == "json-edit":
                                repair_diff, repair_synthesis = synthesize_patch_from_json_edits(instance_id, repair_output)
                            else:
                                repair_diff = extract_unified_diff(repair_output, args.max_hunks_per_file,
                                                                   args.hunk_similarity_threshold)
                                repair_synthesis = {
                                    "ok": bool(repair_diff.strip()),
                                    "error": "" if repair_diff.strip() else "no diff extracted",
                                    "edits": None,
                                }
                            repair_raw_path = case_dir / f"{mode}_repair_output.txt"
                            repair_raw_path.write_text(repair_output, encoding="utf-8")
                            if repair_diff.strip():
                                output = repair_output
                                diff = repair_diff
                                synthesis = repair_synthesis
                                raw_path.write_text(output, encoding="utf-8")
                                patch_path.write_text(diff, encoding="utf-8")
                                apply_check = check_patch_apply(instance_id, patch_path)
                            repair_elapsed_ms = round(repair_response["elapsed_ms"], 2)
                            repair_ttft_ms = (
                                round(repair_response["ttft_ms"], 2)
                                if "ttft_ms" in repair_response
                                else None
                            )
                        mode_results.append(
                            {
                                "mode": mode,
                                "elapsed_ms": round(response["elapsed_ms"], 2),
                                "ttft_ms": (
                                    round(response["ttft_ms"], 2)
                                    if "ttft_ms" in response
                                    else None
                                ),
                                "repair_elapsed_ms": repair_elapsed_ms,
                                "repair_ttft_ms": repair_ttft_ms,
                                "cached_tokens": extract_cached_tokens(response["body"]),
                                "lossy_meta": extract_lossy_meta(response["body"]),
                                "segment_source": "file_context",
                                "segment_source_for_prompt": "file_context",
                                "reuse_selection_source": reuse_selection_source,
                                "segments": [{"name": s.name, "lines": len(s.text.splitlines())} for s in prompt_segments],
                                "reuse_segments": [{"name": s.name, "lines": len(s.text.splitlines())} for s in anchor_segments],
                                **target_prompt_meta,
                                "raw_output_path": str(raw_path),
                                "patch_path": str(patch_path),
                                "diff_extracted": bool(diff.strip()),
                                "patch_synthesis": synthesis,
                                "apply_check": apply_check,
                                "repair_attempted": repair_attempted,
                                "generation_error": "",
                                "candidate_test": {
                                    "returncode": None,
                                    "stdout_tail": "",
                                    "stderr_tail": "not evaluated yet",
                                },
                                **(
                                    {"first_hunk_vs_gold": first_hunk_vs_gold(diff, instance.get("patch", ""))}
                                    if args.emit_first_hunk_vs_gold
                                    else {}
                                ),
                            }
                        )
                    except Exception as exc:
                        raw_path.write_text("", encoding="utf-8")
                        patch_path.write_text("", encoding="utf-8")
                        mode_results.append(
                            {
                                "mode": mode,
                                "elapsed_ms": None,
                                "cached_tokens": 0,
                                "lossy_meta": {},
                                "segment_source": "file_context",
                                "segment_source_for_prompt": "file_context",
                                "reuse_selection_source": reuse_selection_source,
                                "segments": [{"name": s.name, "lines": len(s.text.splitlines())} for s in prompt_segments],
                                "reuse_segments": [{"name": s.name, "lines": len(s.text.splitlines())} for s in anchor_segments],
                                **target_prompt_meta,
                                "raw_output_path": str(raw_path),
                                "patch_path": str(patch_path),
                                "diff_extracted": False,
                                "patch_synthesis": {"ok": False, "error": "generation failed", "edits": None},
                                "generation_error": repr(exc),
                                "candidate_test": {
                                    "returncode": None,
                                    "stdout_tail": "",
                                    "stderr_tail": "generation failed",
                                },
                            }
                        )
                target_hashes = {
                    mode_result.get("target_prompt_sha1")
                    for mode_result in mode_results
                    if mode_result.get("target_prompt_sha1")
                }
                prompt_fair_ok = len(target_hashes) <= 1
                for mode_result in mode_results:
                    mode_result["prompt_fair_ok"] = prompt_fair_ok
                results.append(
                    {
                        "instance_id": instance_id,
                        "repo": instance["repo"],
                        "target_paths": case["target_paths"],
                        "segments": [{"name": s.name, "lines": len(s.text.splitlines())} for s in segments],
                        "graph_segments": [{"name": s.name, "lines": len(s.text.splitlines())} for s in graph_segments],
                        "prompt_fair_ok": prompt_fair_ok,
                        "target_prompt_sha1_set": sorted(target_hashes),
                        "modes": mode_results,
                    }
                )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        kill_port(args.port)

    if args.skip_candidate_tests:
        for case in results:
            for mode_result in case["modes"]:
                mode_result["candidate_test"] = {
                    "returncode": None,
                    "stdout_tail": "",
                    "stderr_tail": "skipped by --skip-candidate-tests",
                }
    else:
        for case in results:
            for mode_result in case["modes"]:
                patch_path = Path(mode_result["patch_path"])
                if mode_result["diff_extracted"] and patch_path.read_text(encoding="utf-8").strip():
                    mode_result["candidate_test"] = evaluate_candidate(args, case["instance_id"], patch_path)
                elif not mode_result.get("generation_error"):
                    mode_result["candidate_test"] = {
                        "returncode": None,
                        "stdout_tail": "",
                        "stderr_tail": "no diff extracted",
                    }

    summary = {
        "model": args.model,
        "dataset": str(args.dataset),
        "manifest": str(args.manifest),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "experiment": {
            "repair_attempts": args.repair_attempts,
            "output_schema": args.output_schema,
            "prompt_layout": args.prompt_layout,
            "force_evict": args.force_evict,
            "disable_overlap_schedule": args.disable_overlap_schedule,
            "max_running_requests": args.max_running_requests,
            "max_cases": args.max_cases,
            "start_index": args.start_index,
            "instance_id_file": str(args.instance_id_file) if args.instance_id_file else None,
            "enable_graph_aware_lossy": args.enable_graph_aware_lossy,
            "code_graph_bundle_manifest": str(args.code_graph_bundle_manifest),
            "graph_bundle_policy": args.graph_bundle_policy,
            "graph_bundle_role": args.graph_bundle_role,
            "skip_candidate_tests": args.skip_candidate_tests,
            "prompt_fair_cases": sum(1 for case in results if case.get("prompt_fair_ok") is not False),
            "prompt_unfair_cases": [
                case.get("instance_id")
                for case in results
                if case.get("prompt_fair_ok") is False
            ],
        },
        "results": results,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--instance-id-file", type=Path, default=None,
                        help="Optional JSON list or newline-delimited instance ids used to filter the dataset before start/max slicing.")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--files-per-case", type=int, default=3)
    parser.add_argument("--max-file-chars", type=int, default=22000)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument("--chunked-prefill-size", type=int, default=8192,
                        help="SGLang --chunked-prefill-size. Lower values (e.g. 6000) chunk the prefill under the free_pages headroom and avoid the lock-pressure OOM (see results/pass100_attempt/REPORT.md Step 2.7).")
    parser.add_argument("--max-prefill-tokens", type=int, default=16384,
                        help="SGLang --max-prefill-tokens.")
    parser.add_argument("--mem-fraction-static", type=float, default=0.82)
    parser.add_argument("--cpu-offload-gb", type=int, default=0,
                        help="GB of system RAM reserved for KV-cache CPU offload (SGLang --cpu-offload-gb). 0 = disabled (default).")
    parser.add_argument("--kv-allocator-defrag", action="store_true",
                        help="Set SGLANG_KV_ALLOCATOR_DEFRAG=1 so alloc_with_defrag merges release_pages into free_pages on alloc failure. Default off (matches upstream SGLang).")
    parser.add_argument("--force-evict", action="store_true",
                        help="Set SGLANG_RADIX_FORCE_EVICT=1 so common.py:evict_from_tree_cache retries with force=True when normal evict() freed 0 tokens, bypassing the lock_ref check on leaves. This recovers from transient lock-pressure OOMs (see results/pass100_attempt/REPORT.md Step 2.10). Default off.")
    parser.add_argument("--disable-overlap-schedule", action="store_true",
                        help="Pass --disable-overlap-schedule to sglang.launch_server so prefill batches are serialized. This is the validated unblock path for the transient lock-pressure OOM (see results/pass100_attempt/REPORT.md Step 2.6). Default off (matches upstream SGLang).")
    parser.add_argument("--max-running-requests", type=int, default=None,
                        help="Cap on the number of concurrent in-flight requests (SGLang --max-running-requests). Lower values reduce lock-pressure on radix-tree leaves between requests.")
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument("--skip-candidate-tests", action="store_true",
                        help="Skip setup_swebench_local_env candidate execution and keep generation/apply-check diagnostics only.")
    parser.add_argument("--recreate-candidate-env", action="store_true",
                        help="Pass --recreate-env to setup_swebench_local_env.py for every candidate test run and record fresh env sanity in the candidate report.")
    parser.add_argument("--server-timeout", type=int, default=180)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--output-schema", choices=["diff", "json-edit"], default="diff")
    parser.add_argument("--prompt-layout", choices=["agenttemplatekv", "legacy"], default="agenttemplatekv")
    parser.add_argument("--enable-placeholder-knn", action="store_true",
                        help="Opt-in: enable v44 placeholder_knn_reuse body via SGLANG_PLACEHOLDER_KNN_MATCH=1. "
                             "Default off; use --enable-placeholder-knn to compare against lossless/lossy baselines. "
                             "O5-lite pre-rotated head-K is also enabled when this is on.")
    parser.add_argument("--placeholder-knn-min-cosine", type=float, default=0.70,
                        help="Override SGLANG_PLACEHOLDER_KNN_MIN_COSINE (default 0.70). "
                             "Phase 3 safety-boundary sweep: 0.85/0.90/0.95/0.97/0.99/1.00.")
    parser.add_argument("--placeholder-knn-topk", type=int, default=4,
                        help="Override SGLANG_PLACEHOLDER_KNN_TOPK (default 4). "
                             "Phase 3 K sweep: 1/3/5.")
    parser.add_argument("--enable-graph-aware-lossy", action="store_true",
                        help="Add a prompt-fair graph_aware_lossy mode. Target prompts remain whole-file; graph bundles only select prompt-resident reuse anchors.")
    parser.add_argument("--code-graph-bundle-manifest", type=Path,
                        default=PROJECT / "results" / "code_graph_kv_reuse" / "data" / "code_graph_precision_manifest.jsonl",
                        help="JSONL manifest containing bundle_text records from the code graph precision census.")
    parser.add_argument("--graph-bundle-policy", default="call_neighborhood_1hop",
                        choices=["ast_function_only", "call_neighborhood_1hop", "reverse_callers_1hop", "import_dependency_bundle", "test_target_bundle"],
                        help="Bundle type used for graph_aware_lossy.")
    parser.add_argument("--graph-bundle-role", default="planner", choices=["planner", "coder", "reviewer"],
                        help="Role row to read from the graph bundle manifest. Text is identical across roles; this keeps provenance explicit.")
    parser.add_argument("--graph-bundles-per-case", type=int, default=3,
                        help="Maximum number of graph bundle records loaded per case for internal graph-aware reuse selection.")
    parser.add_argument("--max-graph-bundle-chars", type=int, default=22000,
                        help="Optional per-bundle char cap before mapping graph bundles to prompt-resident anchors. 0 disables truncation.")
    parser.add_argument("--emit-ttft", action="store_true",
                        help="Use streaming post_chat_stream and record per-mode ttft_ms in the result rows.")
    parser.add_argument("--max-hunks-per-file", type=int, default=4,
                        help="Per-file hunk cap passed to extract_unified_diff for defensive "
                             "truncation of model repetition (R33 Qwen-Coder 11-hunk bug). "
                             "Set to 0 to disable.")
    parser.add_argument("--hunk-similarity-threshold", type=float, default=0.7,
                        help="Hunks with body diff ratio > threshold vs an earlier kept hunk "
                             "in the same file are dropped as repetitive.")
    parser.add_argument("--emit-first-hunk-vs-gold", action="store_true",
                        help="Record first_hunk_vs_gold() per mode per instance in summary.json. "
                             "Weak signal — does model's first hunk target the gold file/line?")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_benchmark(parse_args()))
