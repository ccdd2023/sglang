"""Minimal radix-hit sanity test (streaming mode, matches e2e_accel)."""
import json, os, time, requests

PORT = int(os.environ.get('PORT', '31090'))
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

SYSTEM = "You are a helpful coding assistant."
USER = (
    "Code:\n```python\n"
    "def add(a, b):\n"
    "    return a + b\n\n"
    "def sub(a, b):\n"
    "    return a - b\n"
    "```\n"
    "Explain in one sentence."
)


def one(label, lossy=False, code_offset=None, content_sig=None):
    body = {
        "model": "default",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": True,
    }
    if lossy:
        body.update({
            "reuse_mode": "lossy",
            "lossy_alignment_method": "kvcomm",
            "code_anchor_signature": f"test-{content_sig}",
            "code_content_signature": content_sig,
            "code_anchor_token_spans": [
                {"start_token": 0, "end_token": 0, "content_signature": content_sig}
            ],
            "template_task_family": "code_summary",
            "template_workflow_signature": "min-test",
            "nesting_depth": 0,
            "prompt_position_offset": 6,
            "system_prompt_class": "coder",
            "surrounding_code_hash": "none",
        })
    t0 = time.perf_counter()
    cached = None
    final_meta = {}
    text = ""
    with requests.post(URL, json=body, stream=True, timeout=60) as r:
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
            for c in obj.get("choices", []):
                if c.get("delta", {}).get("content"):
                    text += c["delta"]["content"]
            meta = obj.get("meta_info") or {}
            if meta:
                final_meta = meta
    ms = (time.perf_counter() - t0) * 1000
    print(f"[{label}] ttft+decode={ms:.0f}ms")
    print(f"[{label}] meta_info: {json.dumps(final_meta, indent=2)[:600]}")
    print(f"[{label}] text: {text[:60]!r}")
    return final_meta


print("=== TEST 1: same prompt twice, NO lossy ===")
m1 = one("cold-nolossy")
print()
time.sleep(1)
m2 = one("warm-nolossy")
print()
print(f"=== lossless radix hit? ===")
print(f"  cold cached_tokens: {m1.get('cached_tokens', '?')}")
print(f"  warm cached_tokens: {m2.get('cached_tokens', '?')}")
