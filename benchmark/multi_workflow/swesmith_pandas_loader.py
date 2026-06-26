"""SWE-Smith pandas task loader.

Streams the SWE-bench/SWE-smith HuggingFace dataset, filters for
``pandas-dev/pandas`` tasks, parses each patch to extract affected file paths,
and emits a local ``manifest.jsonl`` that ``bench_giant_codebase_reuse.py``
consumes.

Each emitted record has:
    {
      "instance_id":  str,
      "repo":         str,           # e.g. "pandas-dev__pandas.95280573"
      "image_name":   str,           # Docker image tag (unused for driver)
      "problem_statement": str,
      "files":        [str],         # affected file paths (relative to repo root)
      "patch":        str            # raw unified diff
    }

We cap the cache to ``--max-tasks`` records (default 1000). The first run
streams ~50k HF rows to find enough pandas tasks; subsequent runs load the
local JSONL in <1s.

Usage:
    python -m benchmark.multi_workflow.swesmith_pandas_loader \
        --out results/giant_codebase/tasks/pandas__pandas__1000/ \
        --max-tasks 1000
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

# HuggingFace dataset columns we touch.
_REQUIRED_KEYS = (
    "instance_id",
    "patch",
    "repo",
    "problem_statement",
)

# Match `diff --git a/<path> b/<path>` lines to extract changed file paths.
_GIT_DIFF_RE = re.compile(r"^diff --git a/(?P<path>\S+) b/\S+$", re.MULTILINE)


def parse_patch_files(patch: str) -> list[str]:
    """Return the list of file paths touched by a unified diff.

    Falls back to an empty list if the diff is empty or malformed; callers
    should drop such tasks rather than crash the loader.
    """
    if not patch:
        return []
    paths = _GIT_DIFF_RE.findall(patch)
    # De-duplicate while preserving first-seen order.
    seen: set[str] = set()
    unique: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def stream_pandas_tasks(repo_substr: str = "pandas-dev__pandas", max_tasks: int = 1000) -> list[dict]:
    """Stream SWE-bench/SWE-smith and return up to ``max_tasks`` pandas records.

    Requires `datasets` (HuggingFace). Streams the train split to avoid
    downloading the full 52k-task parquet.
    """
    from datasets import load_dataset

    out: list[dict] = []
    ds = load_dataset("SWE-bench/SWE-smith", split="train", streaming=True)
    for row in ds:
        repo = row.get("repo", "")
        if repo_substr not in repo:
            continue
        # Keep only keys we need to slim the JSONL.
        slim = {k: row.get(k, "") for k in _REQUIRED_KEYS}
        slim["image_name"] = row.get("image_name", "")
        slim["files"] = parse_patch_files(slim["patch"])
        if not slim["files"]:
            # Skip tasks with no parseable file paths; driver cannot serve them.
            continue
        out.append(slim)
        if len(out) >= max_tasks:
            break
    return out


def write_manifest(records: list[dict], out_dir: Path) -> Path:
    """Write records to ``manifest.jsonl`` under ``out_dir``; returns the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load_manifest(path: Path) -> list[dict]:
    """Read ``manifest.jsonl`` previously written by ``write_manifest``."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory to write manifest.jsonl into.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=1000,
        help="Maximum number of pandas tasks to cache (default 1000).",
    )
    parser.add_argument(
        "--repo-substr",
        type=str,
        default="pandas-dev__pandas",
        help="Substring filter on the SWE-Smith repo field (default pandas).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[swesmith_loader] streaming SWE-bench/SWE-smith for {args.repo_substr!r}")
    records = stream_pandas_tasks(args.repo_substr, args.max_tasks)
    print(f"[swesmith_loader] collected {len(records)} tasks")
    if not records:
        print("[swesmith_loader] ERROR: no matching tasks found", flush=True)
        return 1
    path = write_manifest(records, args.out)
    print(f"[swesmith_loader] wrote manifest -> {path}")
    # Print a few example file paths so callers can sanity-check coverage.
    file_counts: dict[str, int] = {}
    for r in records:
        for fp in r["files"]:
            file_counts[fp] = file_counts.get(fp, 0) + 1
    top = sorted(file_counts.items(), key=lambda kv: -kv[1])[:10]
    print("[swesmith_loader] top 10 most-touched files:")
    for fp, count in top:
        print(f"  {count:5d}  {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
