#!/usr/bin/env python3
"""Build and validate the Phase 6 result manifest.

Independent review found the hand-built manifest had rotted: two files were
left as ``pending_this_commit`` and one pointed at a commit whose blob no
longer matched, because the manifest was generated before the commit that
contained the files. Generating it is therefore not enough; it has to be
verifiable.

``--check`` re-derives every entry and fails if any file is missing, still
pending, or does not hash-match the blob recorded at ``containing_commit``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

RESULTS = Path("benchmark/approx_kv/results/phase6")
MANIFEST = RESULTS / "RESULT_MANIFEST.json"

ENVIRONMENT = {
    "image_digest": (
        "sha256:0be6e16e2eb288dfd5fa8b0b41015f61731a139fb961d3366ccedf834289d781"
    ),
    "model": "Qwen/Qwen3-0.6B",
    "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "gpu": "NVIDIA GeForce RTX 2080 SUPER, SM75, 8192 MiB",
    "driver": "580.173.02",
    "container_flags": (
        "--runtime=nvidia --gpus all --ipc=host --shm-size=8g --user 1000:1000"
    ),
}

VERIFICATION = {
    "targeted_regression": (
        "python3 -m pytest -q "
        "test/registered/unit/mem_cache/test_approx_kv_core.py "
        "test/registered/unit/mem_cache/test_approx_kv_runtime.py "
        "test/registered/unit/mem_cache/test_approx_kv_integration_source.py "
        "test/registered/unit/mem_cache/test_approx_kv_hicache_backend.py "
        "test/registered/unit/mem_cache/test_approx_kv_cuda.py "
        "test/registered/unit/mem_cache/test_cross_store_substrate.py "
        "test/registered/unit/mem_cache/test_epic_leadingk.py "
        "test/registered/unit/bench/"
    ),
    "manifest_self_check": (
        "python3 -m benchmark.approx_kv.build_result_manifest --check"
    ),
    "known_baseline": (
        "Running the whole test/registered/unit/mem_cache and "
        "test/registered/unit/bench trees reports 935 pre-existing failures "
        "both with and without this branch's changes, verified by stashing. "
        "Judge regressions with the targeted selection above."
    ),
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(("git", *args), capture_output=True, text=True).stdout.strip()


def containing_commit(path: Path) -> str | None:
    return git("log", "-1", "--format=%H", "--", str(path)) or None


def blob_sha256_at(commit: str, path: Path) -> str | None:
    blob = subprocess.run(("git", "show", f"{commit}:{path}"), capture_output=True)
    if blob.returncode != 0:
        return None
    return hashlib.sha256(blob.stdout).hexdigest()


def build_entries() -> list[dict]:
    entries = []
    for path in sorted(RESULTS.iterdir()):
        if not path.is_file() or path.name == MANIFEST.name:
            continue
        entries.append(
            {
                "file": str(path),
                "sha256": sha256_of(path),
                "bytes": path.stat().st_size,
                "containing_commit": containing_commit(path),
            }
        )
    return entries


def check() -> int:
    if not MANIFEST.exists():
        print("FAIL: manifest does not exist")
        return 1
    manifest = json.loads(MANIFEST.read_text())
    problems: list[str] = []
    listed = {entry["file"] for entry in manifest["files"]}

    for path in sorted(RESULTS.iterdir()):
        if path.is_file() and path.name != MANIFEST.name:
            if str(path) not in listed:
                problems.append(f"{path}: present on disk but absent from manifest")

    for entry in manifest["files"]:
        path = Path(entry["file"])
        if not path.exists():
            problems.append(f"{path}: listed but missing on disk")
            continue
        if sha256_of(path) != entry["sha256"]:
            problems.append(f"{path}: working-tree hash differs from manifest")
        commit = entry.get("containing_commit")
        if not commit or commit == "pending_this_commit":
            problems.append(f"{path}: no containing commit recorded")
            continue
        blob = blob_sha256_at(commit, path)
        if blob is None:
            problems.append(f"{path}: not present at {commit[:9]}")
        elif blob != entry["sha256"]:
            problems.append(f"{path}: blob at {commit[:9]} differs from manifest")

    if problems:
        print(f"FAIL: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"OK: {len(manifest['files'])} artifacts verified against their commits")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return check()

    manifest = {
        "schema_version": 2,
        "artifact": "phase6-result-manifest",
        "purpose": (
            "Supply the file-to-commit mapping that the individual result "
            "artifacts cannot: a runner never knows the commit that will "
            "contain its own output, so every artifact carries "
            "result_git_sha=null and result_commit_status="
            "pending_result_commit."
        ),
        "authority": (
            "This manifest, not the result_git_sha field inside an artifact, "
            "is the authoritative mapping. Regenerate and re-check it in the "
            "same commit that adds or changes an artifact, otherwise entries "
            "rot into pending or stale-blob states."
        ),
        "known_gaps": [
            "Server logs are referenced by absolute host path and are not "
            "versioned in this repository."
        ],
        "environment": ENVIRONMENT,
        "verification_commands": VERIFICATION,
        "files": build_entries(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    pending = sum(1 for entry in manifest["files"] if not entry["containing_commit"])
    print(f"wrote {len(manifest['files'])} entries, {pending} not yet committed")
    print("now commit, then re-run with --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
