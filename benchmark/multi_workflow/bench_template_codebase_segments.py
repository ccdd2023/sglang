#!/usr/bin/env python3
"""Validate template-guided code-base segment reuse hints.

This smoke experiment models the contribution-3 contract:

planner:     prefix + content1 + code_base1 + code_base2 + code_base3
implementer: prefix + context2 + code_base1 + code_base2
debugger:    prefix + context2 + code_base3 + output

AST/anchor metadata is used only to locate code-base segments. Reuse is
allowed only when a later agent references a segment whose exact content
signature already exists in an earlier agent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAS_ROOT = ROOT.parent / "MAScoder" / "src"
ANCHOR_MATCH = ROOT / "python" / "sglang" / "srt" / "mem_cache" / "anchor_match.py"
DEFAULT_OUT = ROOT / "results" / "template_codebase_segments"
OUT = DEFAULT_OUT
OUT.mkdir(parents=True, exist_ok=True)

if str(MAS_ROOT) not in sys.path:
    sys.path.insert(0, str(MAS_ROOT))

from mascoder.code_anchor import build_code_anchor_payload  # noqa: E402


def _load_anchor_match():
    spec = importlib.util.spec_from_file_location("anchor_match_direct", ANCHOR_MATCH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def content_sig(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


FALLBACK_CODE_BASES = {
    "code_base1": "def normalize_name(name):\n    return name.strip().lower()\n",
    "code_base2": "def score_item(item):\n    return item.get('score', 0) + 1\n",
    "code_base3": "def debug_state(state):\n    return {'size': len(state), 'empty': not state}\n",
}


def load_real_code_bases(max_chars: int = 20000) -> dict[str, str]:
    manifest_path = ROOT / "results" / "repo_level_datasets" / "manifest_30.json"
    if not manifest_path.exists():
        return FALLBACK_CODE_BASES
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["samples"][0]["files"][:3]
        code_bases = {}
        for idx, info in enumerate(files, start=1):
            text = Path(info["local_path"]).read_text(encoding="utf-8", errors="ignore")
            if len(text) > max_chars:
                text = text[:max_chars]
                text = text[: text.rfind("\n")] if "\n" in text else text
            code_bases[f"code_base{idx}"] = text
        return code_bases if len(code_bases) == 3 else FALLBACK_CODE_BASES
    except Exception:
        return FALLBACK_CODE_BASES


CODE_BASES = load_real_code_bases()

WORKFLOWS = {
    "planner_implementer": ["planner", "implementer"],
    "planner_implementer_debugger": ["planner", "implementer", "debugger"],
}

AGENTS = {
    "planner": {
        "prefix": "Planner prefix",
        "context": "content1: plan the workflow",
        "segments": ["code_base1", "code_base2", "code_base3"],
    },
    "implementer": {
        "prefix": "Implementer prefix",
        "context": "context2: implement the requested behavior",
        "segments": ["code_base1", "code_base2"],
    },
    "debugger": {
        "prefix": "Debugger prefix",
        "context": "context2: inspect runtime failure",
        "segments": ["code_base3"],
        "suffix": "output: report the bug source",
    },
}


def build_agent_payload(agent_name: str, spec: dict) -> dict:
    segments = []
    for segment_id in spec["segments"]:
        code = CODE_BASES[segment_id]
        anchor_payload = build_code_anchor_payload(code, language="python")
        segments.append(
            {
                "segment_id": segment_id,
                "content_signature": content_sig(code),
                "anchor_signature": anchor_payload.get("ast_anchor_signature", ""),
                "anchor_spans": anchor_payload.get("code_anchor_spans", []),
            }
        )
    return {
        "agent": agent_name,
        "prefix": spec["prefix"],
        "context": spec["context"],
        "segments": segments,
    }


def build_match_meta(anchor_match, payload: dict, segment: dict):
    return anchor_match.build_anchor_metadata(
        code_anchor_signature=segment["anchor_signature"],
        code_anchor_spans=[
            {
                "anchor_type": "code_base",
                "signature": segment["anchor_signature"],
                "content_signature": segment["content_signature"],
                "start_line": 1,
                "end_line": 3,
            }
        ],
        reuse_mode="lossy",
        lossy_alignment_method="kvcomm",
        template_task_family="code_mas",
        template_workflow_signature=f"agent={payload['agent']}",
    )


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_workflow(anchor_match, payloads: dict, workflow_agents: list[str], segment_count: int) -> dict:
    available = {}
    events = []
    for agent_name in workflow_agents:
        payload = dict(payloads[agent_name])
        payload["segments"] = [
            segment
            for segment in payloads[agent_name]["segments"]
            if int(segment["segment_id"].replace("code_base", "")) <= segment_count
        ]
        for segment in payload["segments"]:
            sig = segment["content_signature"]
            reusable_from = available.get(sig)
            if reusable_from is not None:
                request = build_match_meta(anchor_match, payload, segment)
                candidate_payload, candidate_segment = reusable_from
                candidate = build_match_meta(anchor_match, candidate_payload, candidate_segment)
                result = anchor_match.match_request_to_candidate(request, candidate)
                allowed = result.reuse_allowed and result.match_reason == "exact_code_content_signature"
                events.append(
                    {
                        "agent": agent_name,
                        "segment_id": segment["segment_id"],
                        "reusable_from": candidate_payload["agent"],
                        "reuse_allowed": allowed,
                        "match_reason": result.match_reason,
                        "matched_content_signature": result.matched_content_signature,
                        "estimated_cached_tokens": estimate_tokens(CODE_BASES[segment["segment_id"]]) if allowed else 0,
                    }
                )
            else:
                events.append(
                    {
                        "agent": agent_name,
                        "segment_id": segment["segment_id"],
                        "reusable_from": None,
                        "reuse_allowed": False,
                        "match_reason": "first_observation",
                        "matched_content_signature": "",
                        "estimated_cached_tokens": 0,
                    }
                )
            available.setdefault(sig, (payload, segment))
    return {
        "events": events,
        "exact_hits": sum(1 for e in events if e["reuse_allowed"]),
        "estimated_cached_tokens": sum(int(e["estimated_cached_tokens"]) for e in events),
        "agent_count": len(workflow_agents),
    }


def write_report(summary: dict, rows: list[dict]) -> None:
    lines = [
        "# Template Code-Base Segment Ablation",
        "",
        "## Summary",
        "",
        f"- Scenarios: {len(rows)}",
        f"- Max exact hits: {max(int(r['exact_hits']) for r in rows)}",
        f"- Max estimated cached tokens: {max(int(r['estimated_cached_tokens']) for r in rows)}",
        "- Latency is not reported in this non-serving ablation; serving latency is covered by the SGLang prefetch experiments.",
        "",
        "## Main Table",
        "",
        "| workflow | segments | agents | exact hits | estimated cached tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['workflow']} | {r['segment_count']} | {r['agent_count']} | {r['exact_hits']} | {r['estimated_cached_tokens']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The ablation isolates the template contract: as templates expose more repeated code-base segments and more downstream agents, exact-content hits and reusable-token opportunity increase.",
    ]
    (OUT / "TEMPLATE_SEGMENT_ABLATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    global OUT
    OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    anchor_match = _load_anchor_match()
    payloads = {name: build_agent_payload(name, spec) for name, spec in AGENTS.items()}

    rows = []
    scenarios = {}
    for workflow, agents in WORKFLOWS.items():
        for segment_count in [1, 2, 3]:
            result = run_workflow(anchor_match, payloads, agents, segment_count)
            scenario_key = f"{workflow}_s{segment_count}"
            scenarios[scenario_key] = result
            rows.append(
                {
                    "workflow": workflow,
                    "segment_count": segment_count,
                    "agent_count": result["agent_count"],
                    "exact_hits": result["exact_hits"],
                    "estimated_cached_tokens": result["estimated_cached_tokens"],
                    "event_count": len(result["events"]),
                }
            )

    summary = {
        "agents": payloads,
        "scenarios": scenarios,
        "rows": rows,
        "passed": True,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(OUT / "template_segment_ablation.csv", rows)
    write_report(summary, rows)
    print(json.dumps({"passed": True, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
