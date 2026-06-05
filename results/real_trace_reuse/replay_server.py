"""Replay SWE-bench 3-agent traces against a running sglang-kvflow server
and log the lossy telemetry to a JSONL file for offline aggregation.

This script does NOT start the server — it expects a server already
running on http://127.0.0.1:31082 with the env vars:
  SGLANG_LOSSY_FUZZY_MATCH=1
  SGLANG_LOSSY_SKIP_TOKEN_CHECK=1   (for fast smoke; remove for prod)

Each record in the output JSONL has:
  - request_id, instance_id, agent, content_signature
  - response.status_code, response.choices[0].meta_info.lossy_*
  - predicted_distance (from the lookup table; we don't have actual d_norm
    without re-running the model, so we use the table's predicted as
    a calibration reference)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
DEFAULT_PORT = 31082
DEFAULT_URL = f"http://127.0.0.1:{DEFAULT_PORT}/v1/chat/completions"

LOSSY_KEYS = (
    "lossy_candidate_count",
    "lossy_first_match_reason",
    "lossy_first_rejected_reason",
    "lossy_first_reuse_allowed",
    "lossy_first_reuse_confidence",
    "lossy_first_matched_anchor_signature",
    "lossy_first_matched_content_signature",
    "lossy_final_match_reason",
    "lossy_final_rejected_reason",
    "lossy_final_reuse_allowed",
    "lossy_final_reuse_confidence",
    "lossy_final_matched_anchor_signature",
    "lossy_final_matched_content_signature",
    "lossy_anchor_match_used",
    "lossy_anchor_match_len",
    "lossy_anchor_match_gap_len",
    "lossy_anchor_match_signature",
    "lossy_anchor_match_content_signature",
    "lossy_anchor_rope_delta",
    "lossy_predicted_distance",
    "lossy_context_aware_confidence",
    "lossy_context_aware_multiplier",
)


def _send(req_payload: dict, url: str, timeout: int = 60) -> dict:
    body = {
        "model": "default",
        "messages": [
            {"role": "system", "content": req_payload["system"]},
            {"role": "user", "content": req_payload["user"]},
        ],
        "max_tokens": 32,
        "temperature": 0.0,
        "reuse_mode": "lossy",
        "lossy_alignment_method": "kvcomm",
        "code_anchor_signature": f"trace-{req_payload['instance_id']}",
        "code_content_signature": req_payload["code_content_signature"],
        "code_anchor_token_spans": [
            {"start_token": req_payload["approx_token_span"][0],
             "end_token":   req_payload["approx_token_span"][1],
             "content_signature": req_payload["code_content_signature"]}
        ],
        "template_task_family": "code_workflow",
        "template_workflow_signature": req_payload["instance_id"],
        # Prompt-context fields (sglang-kvflow context_aware_confidence)
        "nesting_depth": 0,
        "prompt_position_offset": 30,   # rough — small offset
        "system_prompt_class": req_payload["agent"],
        "surrounding_code_hash": "none",
    }
    r = requests.post(url, json=body, timeout=timeout)
    return {"status_code": r.status_code, "body": r.json()}


def _extract_lossy(body: dict) -> dict:
    meta = body.get("metadata", {}) or {}
    reuse = meta.get("lossy_reuse", {}) or {}
    out = dict(reuse)
    for ch in body.get("choices", []):
        ci = ch.get("meta_info", {}) or {}
        for k in LOSSY_KEYS:
            if k in ci and k not in out:
                out[k] = ci[k]
        break
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default=str(PROJECT_ROOT / "results" / "real_trace_reuse" / "data" / "swe_bench_traces.jsonl"))
    p.add_argument("--out", default=str(PROJECT_ROOT / "results" / "real_trace_reuse" / "data" / "replay_log.jsonl"))
    p.add_argument("--url", default=os.environ.get("KVFLOW_REPLAY_URL", DEFAULT_URL))
    p.add_argument("--max-requests", type=int, default=-1, help="-1 = all")
    p.add_argument("--request-timeout", type=int, default=60)
    args = p.parse_args()

    # Load all trace records
    with open(args.traces) as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"[replay] loaded {len(records)} trace records", flush=True)
    if args.max_requests > 0:
        records = records[: args.max_requests]

    n_total = len(records)
    n_ok = 0
    n_lossy_allowed = 0
    by_agent: dict = {}
    with open(args.out, "w") as f:
        for i, rec in enumerate(records):
            t0 = time.time()
            try:
                resp = _send(rec, args.url, timeout=args.request_timeout)
            except Exception as e:
                print(f"[replay] {i+1}/{n_total} {rec['instance_id']}/{rec['agent']} ERROR: {e}", flush=True)
                continue
            elapsed = time.time() - t0
            if resp["status_code"] != 200:
                print(f"[replay] {i+1}/{n_total} {rec['instance_id']}/{rec['agent']} HTTP {resp['status_code']}", flush=True)
                continue
            lossy = _extract_lossy(resp["body"])
            out_rec = {
                "request_index": i,
                "instance_id": rec["instance_id"],
                "agent": rec["agent"],
                "code_content_signature": rec["code_content_signature"],
                "n_tokens": rec["n_tokens"],
                "elapsed_s": round(elapsed, 3),
                **lossy,
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_ok += 1
            by_agent[rec["agent"]] = by_agent.get(rec["agent"], 0) + 1
            if lossy.get("lossy_final_reuse_allowed") is True:
                n_lossy_allowed += 1
            if (i + 1) % 50 == 0:
                print(f"[replay] {i+1}/{n_total} ({n_lossy_allowed} lossy-allowed) elapsed avg {elapsed:.2f}s",
                      flush=True)
    print(f"[replay] DONE: {n_ok}/{n_total} ok, {n_lossy_allowed} lossy-allowed", flush=True)
    print(f"[replay] per-agent counts: {by_agent}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
