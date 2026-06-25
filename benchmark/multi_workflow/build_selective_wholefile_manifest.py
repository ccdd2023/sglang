#!/usr/bin/env python3
"""Build a fixed full-dataset manifest for selective whole-file reuse.

The benchmark must not promote ad-hoc partial SWE/codebase pilots. This helper
turns the repo-level 500-case manifest into a deterministic complete subset:
every selected instance has at least one full Python file under the configured
size threshold and at least one function/method span admitted by the selective
policy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from benchmark.multi_workflow.selective_ast_reuse import (  # noqa: E402
    DEFAULT_POLICY_PATH,
    load_selective_policy,
    select_spans,
    split_python_file,
)


DEFAULT_SOURCE_MANIFEST = PROJECT / "results/repo_level_datasets/manifest_500.json"
DEFAULT_SOURCE_DATASET = PROJECT / "results/repo_level_datasets/swe_verified_500_instances.json"
DEFAULT_OUT_MANIFEST = PROJECT / "results/selective_ast_reuse/data/swe_selective_wholefile_80k_manifest.json"
DEFAULT_OUT_DATASET = PROJECT / "results/selective_ast_reuse/data/swe_selective_wholefile_80k_instances.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def task_text(instance: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(instance.get("problem_statement", "")),
            str(instance.get("FAIL_TO_PASS", "")),
            str(instance.get("test_patch", "")),
        ]
    ).lower()


def task_file_score(path: str, text: str) -> int:
    path_lower = path.lower()
    basename = Path(path_lower).name
    stem = Path(path_lower).stem
    parts = [part for part in re.split(r"[^a-z0-9_]+", path_lower) if len(part) >= 3]
    score = 0
    if path_lower and path_lower in text:
        score += 1000
    if basename and basename in text:
        score += 300
    if stem and stem in text:
        score += 120
    for part in set(parts):
        if part in {"src", "test", "tests", "python"}:
            continue
        if part in text:
            score += 25
    return score


def select_files(
    sample: dict[str, Any],
    instance: dict[str, Any],
    policy: dict[str, Any],
    max_file_chars: int,
    files_per_case: int,
    selection_strategy: str,
) -> list[dict[str, Any]]:
    candidates: list[tuple[int, int, int, dict[str, Any]]] = []
    text_for_task = task_text(instance)
    for file_info in sample.get("files", []):
        local_path = Path(file_info.get("local_path", ""))
        if not local_path.exists() or local_path.suffix != ".py":
            continue
        text = local_path.read_text(encoding="utf-8", errors="replace").rstrip()
        if len(text) > max_file_chars:
            continue
        spans = split_python_file(str(file_info.get("path", "")), text, policy)
        selected = select_spans(spans, "selective_function_method_reuse")
        if not selected:
            continue
        enriched = dict(file_info)
        enriched["chars"] = len(text)
        enriched["selective_span_count"] = len(selected)
        enriched["selective_granularities"] = sorted({span.granularity for span in selected})
        enriched["task_file_score"] = task_file_score(str(file_info.get("path", "")), text_for_task)
        if selection_strategy == "task_aware":
            candidates.append((-enriched["task_file_score"], -len(selected), len(text), enriched))
        elif selection_strategy == "span_count":
            candidates.append((0, -len(selected), len(text), enriched))
        else:
            raise ValueError(f"unknown selection strategy: {selection_strategy}")
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3].get("path", "")))
    return [item[3] for item in candidates[:files_per_case]]


def build(args: argparse.Namespace) -> None:
    policy = load_selective_policy(args.policy)
    source_manifest = load_json(args.source_manifest)
    source_rows = load_json(args.source_dataset)
    row_by_id = {row["instance_id"]: row for row in source_rows}

    selected_samples: list[dict[str, Any]] = []
    repo_distribution: dict[str, int] = {}
    file_count = 0
    span_count = 0
    total_chars = 0
    rejected_no_file = 0

    allowed_ids = None
    if args.instance_ids_from_manifest:
        allowed_manifest = load_json(args.instance_ids_from_manifest)
        allowed_ids = {sample["instance_id"] for sample in allowed_manifest.get("samples", [])}

    for sample in source_manifest.get("samples", []):
        if allowed_ids is not None and sample.get("instance_id") not in allowed_ids:
            continue
        instance = row_by_id.get(sample["instance_id"], {})
        selected_files = select_files(
            sample,
            instance,
            policy,
            args.max_file_chars,
            args.files_per_case,
            args.selection_strategy,
        )
        if not selected_files:
            rejected_no_file += 1
            continue
        out_sample = dict(sample)
        out_sample["files"] = selected_files
        out_sample["selective_wholefile_rule"] = {
            "source_manifest": str(args.source_manifest),
            "max_file_chars": args.max_file_chars,
            "files_per_case": args.files_per_case,
            "required_mode": "selective_function_method_reuse",
            "requires_full_file": True,
            "selection": args.selection_strategy,
        }
        selected_samples.append(out_sample)
        repo = str(sample.get("repo", ""))
        repo_distribution[repo] = repo_distribution.get(repo, 0) + 1
        file_count += len(selected_files)
        span_count += sum(int(f.get("selective_span_count", 0)) for f in selected_files)
        total_chars += sum(int(f.get("chars", 0)) for f in selected_files)

    selected_ids = {sample["instance_id"] for sample in selected_samples}
    selected_rows = [row_by_id[iid] for iid in selected_ids if iid in row_by_id]
    selected_rows.sort(key=lambda row: row["instance_id"])
    selected_samples.sort(key=lambda sample: sample["instance_id"])

    out_manifest = {
        "label": args.label,
        "schema_version": "selective_wholefile_manifest_v1",
        "source_manifest": str(args.source_manifest),
        "source_dataset": str(args.source_dataset),
        "policy": str(args.policy),
        "case_count": len(selected_samples),
        "file_count": file_count,
        "selective_span_count": span_count,
        "avg_chars_per_file": round(total_chars / file_count, 2) if file_count else 0.0,
        "repo_distribution": dict(sorted(repo_distribution.items())),
        "rejected_no_eligible_file": rejected_no_file,
        "selection_rule": {
            "max_file_chars": args.max_file_chars,
            "files_per_case": args.files_per_case,
            "requires_python_file": True,
            "requires_complete_file_under_threshold": True,
            "requires_selective_function_or_method_span": True,
            "selection_strategy": args.selection_strategy,
            "instance_ids_from_manifest": str(args.instance_ids_from_manifest) if args.instance_ids_from_manifest else "",
        },
        "samples": selected_samples,
    }

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps(out_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out_dataset.write_text(json.dumps(selected_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out_manifest} ({len(selected_samples)} cases, {file_count} files)")
    print(f"wrote {args.out_dataset} ({len(selected_rows)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--out-manifest", type=Path, default=DEFAULT_OUT_MANIFEST)
    parser.add_argument("--out-dataset", type=Path, default=DEFAULT_OUT_DATASET)
    parser.add_argument("--label", default="swe_selective_wholefile_80k_full")
    parser.add_argument("--max-file-chars", type=int, default=80000)
    parser.add_argument("--files-per-case", type=int, default=2)
    parser.add_argument("--selection-strategy", choices=("span_count", "task_aware"), default="span_count")
    parser.add_argument("--instance-ids-from-manifest", type=Path,
                        help="Restrict output to instance ids already present in another selective manifest.")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
