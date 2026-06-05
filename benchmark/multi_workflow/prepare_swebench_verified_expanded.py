#!/usr/bin/env python3
"""Prepare expanded SWE-bench Verified repo-level samples.

This writes both:
- a codebase-content manifest with large Python files per instance
- the matching SWE-bench Verified metadata rows for local env/test setup
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS


PROJECT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT / "results" / "repo_level_datasets"
CODEBASE_DATASET = "ScalingIntelligence/swe-bench-verified-codebase-content"
VERIFIED_DATASET = "princeton-nlp/SWE-bench_Verified"


def repo_key(instance_id: str) -> str:
    match = re.match(r"(.+__.+)-\d+$", instance_id)
    return match.group(1) if match else instance_id.split("-")[0]


def is_interesting_python_file(path: str) -> bool:
    parts = set(path.split("/"))
    if not path.endswith(".py"):
        return False
    if {"tests", "test", "docs", "doc", "examples"} & parts:
        return False
    name = Path(path).name
    if name.startswith("test_") or name in {"setup.py", "conftest.py"}:
        return False
    return True


def load_verified_rows() -> dict[str, dict[str, Any]]:
    rows = load_dataset(VERIFIED_DATASET, split="test")
    return {row["instance_id"]: dict(row) for row in rows}


def select_problem_rows(max_cases: int, one_per_repo: bool, max_per_repo: int) -> list[dict[str, Any]]:
    verified_by_id = load_verified_rows()
    selected: list[dict[str, Any]] = []
    seen_repos: set[str] = set()
    repo_counts: dict[str, int] = {}
    stream = load_dataset(CODEBASE_DATASET, "problem_files", split="test", streaming=True)
    for row in stream:
        instance_id = row["instance_id"]
        if instance_id not in verified_by_id:
            continue
        verified = verified_by_id[instance_id]
        repo = verified["repo"]
        version = str(verified["version"])
        if repo not in MAP_REPO_VERSION_TO_SPECS or version not in MAP_REPO_VERSION_TO_SPECS[repo]:
            continue
        key = repo_key(instance_id)
        if one_per_repo and key in seen_repos:
            continue
        if max_per_repo > 0 and repo_counts.get(repo, 0) >= max_per_repo:
            continue
        interesting = [f for f in row["files"] if is_interesting_python_file(f["file_path"])]
        if len(interesting) < 2:
            continue
        row = dict(row)
        row["files"] = interesting
        row["verified"] = verified
        selected.append(row)
        seen_repos.add(key)
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
        if len(selected) >= max_cases:
            break
    if len(selected) < max_cases:
        print(f"warning: selected only {len(selected)} rows, requested {max_cases}", file=sys.stderr)
    return selected


def resolve_file_contents(rows: list[dict[str, Any]]) -> dict[str, str]:
    needed = {file_info["content_hash"] for row in rows for file_info in row["files"]}
    resolved: dict[str, str] = {}
    stream = load_dataset(CODEBASE_DATASET, "file_content", split="test", streaming=True)
    for row in stream:
        h = row["hash"]
        if h in needed:
            resolved[h] = row["content"]
            if len(resolved) == len(needed):
                break
    missing = needed - set(resolved)
    if missing:
        print(f"warning: missing {len(missing)} file contents", file=sys.stderr)
    return resolved


def choose_large_files(row: dict[str, Any], contents: dict[str, str], max_files: int) -> list[dict[str, Any]]:
    candidates = []
    for file_info in row["files"]:
        content = contents.get(file_info["content_hash"], "")
        if not content.strip() or len(content.splitlines()) < 40:
            continue
        candidates.append(
            {
                "path": file_info["file_path"],
                "content_hash": file_info["content_hash"],
                "content": content,
                "lines": len(content.splitlines()),
                "chars": len(content),
            }
        )
    candidates.sort(key=lambda item: (item["lines"], item["chars"]), reverse=True)
    return candidates[:max_files]


def write_outputs(rows: list[dict[str, Any]], contents: dict[str, str], max_files: int, out_dir: Path, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    verified_rows = []
    for row in rows:
        files = choose_large_files(row, contents, max_files)
        if len(files) < 2:
            continue
        instance_id = row["instance_id"]
        sample_dir = out_dir / instance_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        written_files = []
        for file_item in files:
            target = sample_dir / file_item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(file_item["content"], encoding="utf-8")
            written_files.append(
                {
                    "path": file_item["path"],
                    "local_path": str(target),
                    "content_hash": file_item["content_hash"],
                    "lines": file_item["lines"],
                    "chars": file_item["chars"],
                }
            )
        verified_rows.append(row["verified"])
        samples.append(
            {
                "source_dataset": CODEBASE_DATASET,
                "metadata_dataset": VERIFIED_DATASET,
                "instance_id": instance_id,
                "repo": row["verified"]["repo"],
                "version": str(row["verified"]["version"]),
                "difficulty": row["verified"].get("difficulty", ""),
                "repo_key": repo_key(instance_id),
                "sample_dir": str(sample_dir),
                "files": written_files,
            }
        )
    repo_distribution: dict[str, int] = {}
    total_files = 0
    total_lines = 0
    total_chars = 0
    for sample in samples:
        repo_distribution[sample["repo"]] = repo_distribution.get(sample["repo"], 0) + 1
        total_files += len(sample["files"])
        total_lines += sum(file_item["lines"] for file_item in sample["files"])
        total_chars += sum(file_item["chars"] for file_item in sample["files"])
    manifest = {
        "source_dataset": CODEBASE_DATASET,
        "metadata_dataset": VERIFIED_DATASET,
        "label": label,
        "case_count": len(samples),
        "repo_distribution": dict(sorted(repo_distribution.items())),
        "file_stats": {
            "total_files": total_files,
            "total_lines": total_lines,
            "total_chars": total_chars,
            "avg_files_per_case": (total_files / len(samples)) if samples else 0,
            "avg_lines_per_case": (total_lines / len(samples)) if samples else 0,
            "avg_chars_per_case": (total_chars / len(samples)) if samples else 0,
        },
        "samples": samples,
    }
    manifest_path = out_dir / f"manifest_{label}.json"
    dataset_path = out_dir / f"swe_verified_{label}_instances.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    dataset_path.write_text(json.dumps(verified_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path, dataset_path, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--max-files", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--label", default="10")
    parser.add_argument("--allow-multiple-per-repo", action="store_true")
    parser.add_argument("--max-per-repo", type=int, default=0, help="0 means no per-repo cap")
    args = parser.parse_args()

    rows = select_problem_rows(
        args.max_cases,
        one_per_repo=not args.allow_multiple_per_repo,
        max_per_repo=args.max_per_repo,
    )
    if not rows:
        raise RuntimeError("no suitable rows found")
    contents = resolve_file_contents(rows)
    manifest_path, dataset_path, manifest = write_outputs(rows, contents, args.max_files, args.out_dir, args.label)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nSaved manifest: {manifest_path}")
    print(f"Saved metadata: {dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
