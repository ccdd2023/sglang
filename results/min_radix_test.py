"""Minimal radix-hit sanity test.

Sends the SAME prompt twice and checks cached_tokens on the second request.
If cached_tokens=0, the radix tree isn't matching the seed. No lossy metadata.

Run with the e2e server config to reproduce the e2e_accel issue.
"""
import json, os, time, requests

PORT = int(os.environ.get('PORT', '31090'))
MODEL = os.environ.get('MODEL', '/home/gfy/models/Qwen2.5-3B-Instruct')

URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

# Use a small but non-trivial prompt with some system prefix
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

def one(label):
    body = {
        "model": MODEL.split("/")[-1],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": False,
    }
    r = requests.post(URL, json=body, timeout=60).json()
    meta = r.get("meta_info") or r.get("usage") or {}
    print(f"[{label}] keys: {sorted(r.keys())[:6]}")
    print(f"[{label}] full meta: {json.dumps(meta, indent=2)[:500]}")
    print(f"[{label}] text: {r.get('choices', [{}])[0].get('message', {}).get('content', '')[:60]!r}")
    return meta

# 1. cold request
m1 = one("cold")
print()
time.sleep(1)
# 2. warm request (same prompt) — should hit radix prefix
m2 = one("warm")
print()
print(f"=== cached_tokens comparison ===")
print(f"  cold: prompt={m1.get('prompt_tokens', '?')}, cached={m1.get('cached_tokens', '?')}")
print(f"  warm: prompt={m2.get('prompt_tokens', '?')}, cached={m2.get('cached_tokens', '?')}")
