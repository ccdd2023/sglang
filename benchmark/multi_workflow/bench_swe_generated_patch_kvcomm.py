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


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_by_id = {row["instance_id"]: row for row in manifest["samples"]}
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
            cases.append({"instance": row, "segments": segments, "target_paths": target_paths})
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


def extract_unified_diff(text: str) -> str:
    fenced = re.search(r"```(?:diff|patch)?\s*(diff --git .*?)```", text, re.S)
    if fenced:
        return fenced.group(1).strip() + "\n"
    idx = text.find("diff --git ")
    if idx >= 0:
        return text[idx:].strip() + "\n"
    return ""


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
                await post_chat(
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
                )

                mode_results = []
                for mode, reuse_mode, include_anchor, include_codebase_prefetch in [
                    ("lossless", "lossless", False, False),
                    ("lossy", "lossy", True, False),
                    ("lossy_prefetch", "lossy", True, True),
                ]:
                    messages = build_messages(
                        instance,
                        segments,
                        mode,
                        args.output_schema,
                        args.prompt_layout,
                    )
                    raw_path = case_dir / f"{mode}_output.txt"
                    patch_path = case_dir / f"{mode}.patch"
                    try:
                        response = await post_chat(
                            session,
                            args.port,
                            make_payload(
                                args,
                                tokenizer,
                                messages,
                                segments,
                                reuse_mode,
                                include_anchor,
                                include_codebase_prefetch,
                                f"gen:{instance_id}:{mode}",
                            ),
                        )
                        output = extract_text(response["body"])
                        if args.output_schema == "json-edit":
                            diff, synthesis = synthesize_patch_from_json_edits(instance_id, output)
                        else:
                            diff = extract_unified_diff(output)
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
                        if (
                            args.repair_attempts > 0
                            and (not diff.strip() or apply_check.get("returncode") not in (0, None))
                        ):
                            repair_attempted = True
                            repair_messages = build_repair_messages(
                                instance,
                                segments,
                                output,
                                diff,
                                apply_check.get("stderr_tail", ""),
                                mode,
                                args.output_schema,
                                args.prompt_layout,
                            )
                            repair_response = await post_chat(
                                session,
                                args.port,
                                make_payload(
                                    args,
                                    tokenizer,
                                    repair_messages,
                                    segments,
                                    reuse_mode,
                                    include_anchor,
                                    include_codebase_prefetch,
                                    f"repair:{instance_id}:{mode}",
                                ),
                            )
                            repair_output = extract_text(repair_response["body"])
                            if args.output_schema == "json-edit":
                                repair_diff, repair_synthesis = synthesize_patch_from_json_edits(instance_id, repair_output)
                            else:
                                repair_diff = extract_unified_diff(repair_output)
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
                        mode_results.append(
                            {
                                "mode": mode,
                                "elapsed_ms": round(response["elapsed_ms"], 2),
                                "repair_elapsed_ms": repair_elapsed_ms,
                                "cached_tokens": extract_cached_tokens(response["body"]),
                                "lossy_meta": extract_lossy_meta(response["body"]),
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
                results.append(
                    {
                        "instance_id": instance_id,
                        "repo": instance["repo"],
                        "target_paths": case["target_paths"],
                        "segments": [{"name": s.name, "lines": len(s.text.splitlines())} for s in segments],
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

    summary = {"model": args.model, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
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
    parser.add_argument("--disable-overlap-schedule", action="store_true",
                        help="Pass --disable-overlap-schedule to sglang.launch_server so prefill batches are serialized. This is the validated unblock path for the transient lock-pressure OOM (see results/pass100_attempt/REPORT.md Step 2.6). Default off (matches upstream SGLang).")
    parser.add_argument("--max-running-requests", type=int, default=None,
                        help="Cap on the number of concurrent in-flight requests (SGLang --max-running-requests). Lower values reduce lock-pressure on radix-tree leaves between requests.")
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument("--server-timeout", type=int, default=180)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--output-schema", choices=["diff", "json-edit"], default="diff")
    parser.add_argument("--prompt-layout", choices=["agenttemplatekv", "legacy"], default="agenttemplatekv")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_benchmark(parse_args()))
