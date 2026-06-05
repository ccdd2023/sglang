"""E2E smoke test for sglang-kvflow context_aware_confidence modifier.

Starts a SGLang server (Qwen2.5-3B-Instruct, single GPU) with the
lossy + context_aware paths enabled, sends a 2-request workflow where
the second request carries the 4 new prompt-context fields, and
verifies the response metadata contains:

  - lossy_predicted_distance
  - lossy_context_aware_confidence
  - lossy_context_aware_multiplier

This is the most basic verification that the new plumbing actually
works end-to-end (not just in unit tests).

Usage:
    python results/e2e_smoke.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
TABLE_PATH = PROJECT_ROOT / "results" / "same_code_context_variation" / "data" / "predicted_distance_table.json"
PYTHON_BIN = "/home/gfy/.conda/envs/sglang-kvflow/bin/python"
MODEL = "/home/gfy/models/Qwen2.5-3B-Instruct"
PORT = 31082
STARTUP_TIMEOUT = 240   # seconds
REQUEST_TIMEOUT = 60


def _wait_for_server(port: int, timeout: int) -> bool:
    """Poll the /v1/models endpoint until the server is ready."""
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


def _start_server() -> subprocess.Popen:
    log_path = PROJECT_ROOT / "results" / "e2e_smoke_sglang.log"
    env = os.environ.copy()
    env["SGLANG_LOSSY_FUZZY_MATCH"] = "1"
    env["SGLANG_LOSSY_SKIP_TOKEN_CHECK"] = "1"   # for smoke; tighter checks live elsewhere
    cmd = [
        PYTHON_BIN, "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--port", str(PORT),
        "--tp-size", "1",
        "--mem-fraction-static", "0.85",
        "--max-total-tokens", "32768",
        "--chunked-prefill-size", "4096",
        "--max-prefill-tokens", "8192",
        "--radix-eviction-policy", "priority",
        "--enable-hierarchical-cache",
        "--hicache-ratio", "1.5",
        "--disable-cuda-graph",
        "--log-level", "info",
    ]
    print(f"[e2e] launching server: {' '.join(cmd)}", flush=True)
    print(f"[e2e] log: {log_path}", flush=True)
    return subprocess.Popen(
        cmd, env=env, cwd=str(PROJECT_ROOT),
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
    )


def _send_lossy_request(code: str, *, system_prompt_class: str,
                         prompt_position_offset: int, nesting_depth: int,
                         surrounding_code_hash: str, label: str) -> dict:
    """Send a lossy request with the 4 new prompt-context fields.

    The request is wrapped in a chat template that prefixes the user
    message with `system_prompt_class` worth of padding to push the
    code block to `prompt_position_offset` tokens from the start.
    """
    padding = " ".join(["alpha"] * prompt_position_offset)
    user_content = (
        f"{padding}\n```python\n{code}\n```\n"
        f"Summarise the purpose of this code in one sentence."
    )
    body = {
        "model": "default",
        "messages": [
            {"role": "system", "content": f"You are a {system_prompt_class} agent. Be concise."},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        # The 4 new prompt-context fields (added in feature/context-aware-kv-reuse)
        "reuse_mode": "lossy",
        "lossy_alignment_method": "kvcomm",
        "code_anchor_signature": f"smoke-{label}",
        "code_content_signature": hashlib_of(code),
        "code_anchor_token_spans": [],
        "template_task_family": "code_summary",
        "template_workflow_signature": f"smoke-workflow-{label}",
        "nesting_depth": nesting_depth,
        "prompt_position_offset": prompt_position_offset,
        "system_prompt_class": system_prompt_class,
        "surrounding_code_hash": surrounding_code_hash,
    }
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    r = requests.post(url, json=body, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _send_seed_request(code: str) -> None:
    """First request: just write the code KV to anchor store. No lossy
    fields. Sets up the second request's match."""
    body = {
        "model": "default",
        "messages": [
            {"role": "user", "content": f"```python\n{code}\n```\nSummarise the purpose of this code in one sentence."},
        ],
        "max_tokens": 32,
        "temperature": 0.0,
    }
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    r = requests.post(url, json=body, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()


def hashlib_of(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()[:32]


def _extract_lossy_metadata(chat_response: dict) -> dict:
    """Pull the lossy_* fields from the response. SGLang emits them under
    `response.metadata["lossy_reuse"]` (non-streaming) or alongside the
    first streamed chunk (streaming)."""
    out = {}
    meta = chat_response.get("metadata", {}) or {}
    reuse = meta.get("lossy_reuse", {}) or {}
    out.update(reuse)
    # Also fall back to per-choice meta_info just in case
    for choice in chat_response.get("choices", []):
        ci = choice.get("meta_info", {}) or {}
        for k, v in ci.items():
            if k.startswith("lossy_") and k not in out:
                out[k] = v
        break
    return out


def main() -> int:
    if not TABLE_PATH.exists():
        print(f"[e2e] FATAL: {TABLE_PATH} not found; modifier will no-op")
        return 1
    print(f"[e2e] using table: {TABLE_PATH}")

    proc = _start_server()
    try:
        print(f"[e2e] waiting for server (up to {STARTUP_TIMEOUT}s)...", flush=True)
        if not _wait_for_server(PORT, STARTUP_TIMEOUT):
            print(f"[e2e] FATAL: server did not start in {STARTUP_TIMEOUT}s")
            return 2
        print("[e2e] server is up", flush=True)

        code = (
            "def fibonacci(n):\n"
            "    if n < 2:\n"
            "        return n\n"
            "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
        )

        # Step 1: seed request (no lossy fields) — populates the radix tree
        print("[e2e] step 1/2: seed request (no lossy fields)", flush=True)
        _send_seed_request(code)
        time.sleep(1)

        # Step 2: lossy request with all 4 context fields
        print("[e2e] step 2/2: lossy request with 4 context fields", flush=True)
        resp = _send_lossy_request(
            code,
            system_prompt_class="tester",       # the worst-case bucket
            prompt_position_offset=100,           # 50-100 bin
            nesting_depth=1,
            surrounding_code_hash="imports_wrap",
            label="smoke-1",
        )
        meta = _extract_lossy_metadata(resp)
        print(f"[e2e] lossy metadata captured: {json.dumps(meta, indent=2, default=str)}", flush=True)

        # Verify
        ok = True
        for key in ("lossy_predicted_distance", "lossy_context_aware_confidence",
                    "lossy_context_aware_multiplier", "lossy_anchor_rope_delta"):
            if key not in meta:
                print(f"[e2e] FAIL: missing key {key}")
                ok = False
            else:
                print(f"[e2e]   {key} = {meta[key]}")
        if meta.get("lossy_final_match_reason") not in ("exact_code_content_signature", "exact_code_content_signature_demoted"):
            print(f"[e2e] FAIL: unexpected match_reason {meta.get('lossy_final_match_reason')!r}")
            ok = False
        return 0 if ok else 3
    finally:
        print("[e2e] killing server", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
