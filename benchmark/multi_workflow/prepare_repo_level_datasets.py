#!/usr/bin/env python3
"""Prepare real multi-file repo-level benchmark samples.

The default source is the SWE-bench Verified codebase-content dataset, which
contains Python file snapshots for each SWE-bench Verified problem.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset


PROJECT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT / "results" / "repo_level_datasets"
DATASET_NAME = "ScalingIntelligence/swe-bench-verified-codebase-content"


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


def select_problem_rows(max_repos: int) -> list[dict[str, Any]]:
    rows = []
    seen_repos: set[str] = set()
    stream = load_dataset(DATASET_NAME, "problem_files", split="test", streaming=True)
    for row in stream:
        key = repo_key(row["instance_id"])
        if key in seen_repos:
            continue
        interesting = [f for f in row["files"] if is_interesting_python_file(f["file_path"])]
        if len(interesting) < 3:
            continue
        row = dict(row)
        row["files"] = interesting
        rows.append(row)
        seen_repos.add(key)
        if len(rows) >= max_repos:
            break
    return rows


def resolve_file_contents(rows: list[dict[str, Any]]) -> dict[str, str]:
    needed = {
        file_info["content_hash"]
        for row in rows
        for file_info in row["files"]
    }
    resolved: dict[str, str] = {}
    stream = load_dataset(DATASET_NAME, "file_content", split="test", streaming=True)
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
        if not content.strip():
            continue
        if len(content.splitlines()) < 40:
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


def write_samples(rows: list[dict[str, Any]], contents: dict[str, str], max_files: int, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for row in rows:
        files = choose_large_files(row, contents, max_files)
        if len(files) < 2:
            continue
        sample_dir = out_dir / row["instance_id"]
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
        samples.append(
            {
                "source_dataset": DATASET_NAME,
                "instance_id": row["instance_id"],
                "repo_key": repo_key(row["instance_id"]),
                "sample_dir": str(sample_dir),
                "files": written_files,
            }
        )
    manifest = {
        "source_dataset": DATASET_NAME,
        "samples": samples,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-repos", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    rows = select_problem_rows(args.max_repos)
    if not rows:
        raise RuntimeError("no suitable repo-level rows found")
    contents = resolve_file_contents(rows)
    manifest = write_samples(rows, contents, args.max_files, args.out_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nSaved manifest: {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
