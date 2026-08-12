#!/usr/bin/env python3
"""Attribute common-agent coding-aware outcomes to actual lossy-copy exposure.

This audit is intentionally downstream of the frozen Fresh24 run.  It never
selects tasks or changes policy.  It separates selector capacity, runtime K/V
events, official task resolution, and exact-token ABBA speed so an accuracy
difference on an unexposed trajectory cannot be credited to lossy reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARM = "coding_dependency_graph_cold_lcb"
NONCE_PATTERN = re.compile(r"(?:call_|\b)(p\d+-m\d+)(?:_|-)" )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def trajectory_nonce_map(run_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*.traj.json")):
        value = read_json(path)
        task = str(value.get("instance_id") or path.stem.removesuffix(".traj"))
        match = NONCE_PATTERN.search(path.read_text(encoding="utf-8"))
        if match is None:
            raise ValueError(f"trajectory has no model nonce: {path}")
        nonce = match.group(1)
        previous = result.setdefault(nonce, task)
        if previous != task:
            raise ValueError(f"nonce {nonce} maps to both {previous} and {task}")
    return result


def official_report(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "OFFICIAL_RESULT.json"
    value = read_json(path)
    if isinstance(value.get("report"), dict):
        return value["report"]
    nested = value.get("result")
    if isinstance(nested, dict) and isinstance(nested.get("report"), dict):
        return nested["report"]
    if "resolved_instances" in value:
        return value
    raise ValueError(f"official report absent: {path}")


def resolved_ids(report: dict[str, Any]) -> set[str]:
    if isinstance(report.get("resolved_ids"), list):
        return {str(value) for value in report["resolved_ids"]}
    return {
        str(row["instance_id"])
        for row in report.get("instances") or []
        if bool(row.get("resolved"))
    }


def outcome_label(dense: bool, coding: bool) -> str:
    if dense and coding:
        return "both_resolved"
    if dense:
        return "coding_damage"
    if coding:
        return "coding_rescue"
    return "both_unresolved"


def counter_values(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted((str(key), int(value)) for key, value in counter.items()))


def analyze(campaign: Path) -> dict[str, Any]:
    dense_dir = campaign / "runs/sglang_formal/dense/full_24"
    coding_dir = campaign / f"runs/sglang_formal/{ARM}/full_24"
    client_path = coding_dir / "CLIENT_LEDGER.jsonl"
    server_path = coding_dir / "SERVER_LEDGER.jsonl"
    runtime_path = coding_dir / "RUNTIME_SUMMARY.json"
    required = (
        dense_dir / "OFFICIAL_RESULT.json",
        coding_dir / "OFFICIAL_RESULT.json",
        client_path,
        server_path,
        runtime_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Fresh24 attribution inputs missing: {missing}")

    nonce_to_task = trajectory_nonce_map(coding_dir)
    requests = [
        row for row in read_jsonl(client_path) if row.get("event") == "request_complete"
    ]
    if not requests:
        raise ValueError("coding-aware client ledger has no completed requests")

    task_rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "agent_requests": 0,
            "source_registered_requests": 0,
            "target_registered_requests": 0,
            "physical_copy_requests": 0,
            "copied_k_tokens": 0,
            "copied_v_tokens": 0,
            "eligible_observations": 0,
            "dependency_hot_observations_protected": 0,
            "dependency_cold_observations": 0,
            "version_invalidated_observations": 0,
        }
    )
    selector = Counter()
    source_skips: Counter[str] = Counter()
    target_guards: Counter[str] = Counter()
    for row in requests:
        nonce = str(row.get("model_instance_nonce") or "")
        if nonce not in nonce_to_task:
            raise ValueError(f"client nonce absent from trajectories: {nonce}")
        task = nonce_to_task[nonce]
        task_row = task_rows[task]
        policy = row.get("reuse_policy_decision") or {}
        metrics = row.get("native_backend_metrics") or {}
        task_row["agent_requests"] += 1
        for key in (
            "eligible_observations",
            "dependency_hot_observations_protected",
            "dependency_cold_observations",
            "version_invalidated_observations",
        ):
            value = int(policy.get(key) or 0)
            selector[key] += value
            task_row[key] += value
        for key in (
            "eligible_observations_before_module_filter",
            "eligible_observations_before_dependency_guard",
            "excluded_repository_searches",
            "excluded_ambiguous_multifile_results",
            "unlocalized_candidate_observations",
        ):
            selector[key] += int(policy.get(key) or 0)
        source_registered = bool(row.get("source_registered"))
        target_registered = bool(row.get("target_registered"))
        physical = bool(metrics.get("physical_reuse")) or (
            int(metrics.get("reused_k_tokens") or 0) > 0
            and int(metrics.get("reused_v_tokens") or 0) > 0
        )
        task_row["source_registered_requests"] += int(source_registered)
        task_row["target_registered_requests"] += int(target_registered)
        task_row["physical_copy_requests"] += int(physical)
        task_row["copied_k_tokens"] += int(metrics.get("reused_k_tokens") or 0)
        task_row["copied_v_tokens"] += int(metrics.get("reused_v_tokens") or 0)
        selector["source_registered_requests"] += int(source_registered)
        selector["target_registered_requests"] += int(target_registered)
        selector["physical_copy_requests"] += int(physical)
        source_skips.update(
            {
                str(key): int(value)
                for key, value in (policy.get("source_skip_reasons") or {}).items()
            }
        )
        for guard in policy.get("target_evidence_guards") or []:
            if isinstance(guard, dict):
                label = guard.get("reason") or guard.get("kind") or "unspecified"
            else:
                label = str(guard)
            target_guards[str(label)] += 1

    server_events = Counter(
        str(row.get("event") or "unknown") for row in read_jsonl(server_path)
    )
    runtime = read_json(runtime_path)
    if int(runtime.get("target_copy_events") or 0) != selector["physical_copy_requests"]:
        raise ValueError(
            "client physical-copy count differs from runtime summary: "
            f"{selector['physical_copy_requests']} vs {runtime.get('target_copy_events')}"
        )

    dense_report = official_report(dense_dir)
    coding_report = official_report(coding_dir)
    dense_resolved = resolved_ids(dense_report)
    coding_resolved = resolved_ids(coding_report)
    registered_tasks = sorted(
        set(task_rows)
        | {str(value) for value in dense_report.get("empty_patch_ids") or []}
        | {str(value) for value in coding_report.get("empty_patch_ids") or []}
        | {
            str(row["instance_id"])
            for row in (dense_report.get("instances") or [])
        }
        | {
            str(row["instance_id"])
            for row in (coding_report.get("instances") or [])
        }
    )
    rows = []
    for task in registered_tasks:
        row = {"task": task, **task_rows[task]}
        row.update(
            copy_exposed=bool(row["physical_copy_requests"]),
            dense_resolved=task in dense_resolved,
            coding_resolved=task in coding_resolved,
            outcome=outcome_label(task in dense_resolved, task in coding_resolved),
        )
        rows.append(row)
    exposed = [row for row in rows if row["copy_exposed"]]
    unexposed = [row for row in rows if not row["copy_exposed"]]

    exact_path = campaign / "exact_prompt_replay/fresh24/sglang_coding/RESULT.json"
    exact: dict[str, Any]
    if exact_path.is_file():
        result = read_json(exact_path)
        exact = {
            "status": result.get("status"),
            "summary": result.get("summary"),
            "targets": result.get("targets"),
            "result": str(exact_path),
        }
    else:
        exact = {"status": "PENDING", "result": str(exact_path)}

    def outcome_counts(values: list[dict[str, Any]]) -> dict[str, int]:
        return counter_values(Counter(str(row["outcome"]) for row in values))

    provenance = {
        str(path): sha256(path)
        for path in required
    }
    if exact_path.is_file():
        provenance[str(exact_path)] = sha256(exact_path)
    return {
        "schema_version": 1,
        "status": "COMPLETE" if exact.get("status") == "PASS" else "ACCURACY_COMPLETE_SPEED_PENDING",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "post-outcome copy-exposure attribution; not a task selector",
        "selector_flow": {
            "requests": len(requests),
            **counter_values(selector),
            "source_skip_reasons": counter_values(source_skips),
            "target_evidence_guards": counter_values(target_guards),
        },
        "runtime_events": counter_values(server_events),
        "official_accuracy": {
            "dense": {
                "resolved": int(dense_report["resolved_instances"]),
                "submitted": int(dense_report["submitted_instances"]),
            },
            "coding_aware": {
                "resolved": int(coding_report["resolved_instances"]),
                "submitted": int(coding_report["submitted_instances"]),
            },
        },
        "copy_exposure": {
            "tasks": len(exposed),
            "requests": selector["physical_copy_requests"],
            "copied_k_tokens": int(runtime.get("copied_tokens") or 0),
            "copied_v_tokens": int(runtime.get("copied_tokens") or 0),
            "exposed_outcomes": outcome_counts(exposed),
            "unexposed_outcomes": outcome_counts(unexposed),
        },
        "task_rows": rows,
        "exact_prompt_speed": exact,
        "interpretation_guardrails": [
            "Only official resolved is accuracy.",
            "Only exact-token ABBA target TTFT is a causal speed comparison.",
            "An outcome difference on a task with zero physical copy exposure is not attributed to lossy reuse.",
            "Selector-flow, K/V deviation, NLL, and free-agent latency are diagnostic rather than final quality or speed metrics.",
        ],
        "provenance_sha256": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    output = args.output or campaign / "summary/CODING_ATTRIBUTION.json"
    value = analyze(campaign)
    write_json(output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
