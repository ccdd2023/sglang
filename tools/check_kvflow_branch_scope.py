#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def scope_violations(role: str, paths: list[str]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        normalized = Path(path).as_posix()
        if role == "shared":
            forbidden = (
                "python/sglang/srt/mem_cache/coding_aware/",
                "python/sglang/srt/mem_cache/kvcomm_prefetch/",
                "benchmark/multi_workflow/",
                "results/",
                "paper/",
            )
        elif role == "coding":
            forbidden = (
                "python/sglang/srt/mem_cache/kvcomm_prefetch/",
                "python/sglang/srt/mem_cache/evict_policy.py",
            )
        elif role == "prefetch":
            forbidden = (
                "python/sglang/srt/mem_cache/coding_aware/",
                "python/sglang/srt/mem_cache/ast_chunker.py",
            )
        elif role == "integration":
            forbidden = ("results/", "paper/")
        else:
            raise ValueError(f"unknown branch role: {role}")
        if normalized.startswith(forbidden):
            violations.append(normalized)
    return violations


def changed_paths(base: str) -> list[str]:
    output = subprocess.check_output(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        text=True,
    )
    return [line for line in output.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role",
        required=True,
        choices=("shared", "coding", "prefetch", "integration"),
    )
    parser.add_argument("--base", required=True)
    args = parser.parse_args()

    violations = scope_violations(args.role, changed_paths(args.base))
    if violations:
        print(f"{args.role} branch contains out-of-scope paths:")
        for path in violations:
            print(f"  {path}")
        return 1
    print(f"{args.role} branch scope: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
