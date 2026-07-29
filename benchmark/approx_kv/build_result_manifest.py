#!/usr/bin/env python3
"""Build and validate a phase result manifest.

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


def build_entries(results: Path, manifest_path: Path) -> list[dict]:
    entries = []
    for path in sorted(results.rglob("*")):
        if not path.is_file() or path == manifest_path:
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


def check(results: Path, manifest_path: Path) -> int:
    if not manifest_path.exists():
        print("FAIL: manifest does not exist")
        return 1
    manifest = json.loads(manifest_path.read_text())
    problems: list[str] = []
    listed = {entry["file"] for entry in manifest["files"]}

    for path in sorted(results.rglob("*")):
        if path.is_file() and path != manifest_path:
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
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--phase", default="phase6")
    args = parser.parse_args()
    results = args.results_dir or Path(f"benchmark/approx_kv/results/{args.phase}")
    if results.name != args.phase:
        parser.error(
            f"--phase {args.phase!r} does not match results directory {results}"
        )
    manifest_path = results / "RESULT_MANIFEST.json"
    if args.check:
        return check(results, manifest_path)

    manifest = {
        "schema_version": 2,
        "artifact": f"{args.phase}-result-manifest",
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
        "known_gaps": (
            [
                "The primary P6-H, P6-4 and P6-F server logs are versioned and "
                "content-addressed. Some historical failed attempts and "
                "optional diagnostic logs still exist only at absolute host "
                "paths and are not part of the Phase 6 Exit evidence package."
            ]
            if args.phase == "phase6"
            else (
                []
                if (results / "PHASE7_FINAL_DISPOSITION.json").is_file()
                else
                [
                    "Phase7 GPU artifacts and logs are versioned. Final "
                    "dual-model result review and main-session disposition "
                    "remain pending."
                ]
                if (results / "raw").is_dir()
                and any((results / "raw").glob("*.json"))
                else [
                    "The Phase7 plan, runners and implementation are pinned. "
                    "Execution remains blocked on the mandatory final Opus "
                    "review, so GPU result artifacts and logs do not exist yet."
                ]
            )
        ),
        "environment": ENVIRONMENT,
        "verification_commands": {
            **VERIFICATION,
            "manifest_self_check": (
                "python3 -m benchmark.approx_kv.build_result_manifest "
                f"--check --phase {args.phase} --results-dir {results}"
            ),
        },
        "files": build_entries(results, manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    pending = sum(1 for entry in manifest["files"] if not entry["containing_commit"])
    print(f"wrote {len(manifest['files'])} entries, {pending} not yet committed")
    print("now commit, then re-run with --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
