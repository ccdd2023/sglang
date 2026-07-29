#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from benchmark.approx_kv.phase7.correction import (
    BASE_MANIFEST_PATH,
    CAPACITY_CORRECTION_CPU_EVIDENCE_PATH,
    CAPACITY_CORRECTION_MANIFEST_PATH,
    CAPACITY_CORRECTION_REVIEW_PATH,
    CAPACITY_RUNNER_PATH,
    ORIGINAL_RAW_PATH,
    build_authorized_capacity_correction_manifest,
    build_pinned_capacity_correction_manifest,
    file_sha256,
    load_capacity_cpu_evidence,
    validate_capacity_correction_manifest,
    verify_capacity_correction_files,
)
from benchmark.approx_kv.phase7.correction_review import load_correction_review

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(CAPACITY_CORRECTION_MANIFEST_PATH)
DEFAULT_BASE_MANIFEST = Path(BASE_MANIFEST_PATH)
DEFAULT_REVIEW = Path(CAPACITY_CORRECTION_REVIEW_PATH)
DEFAULT_CPU_EVIDENCE = Path(CAPACITY_CORRECTION_CPU_EVIDENCE_PATH)


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument(
        "--status",
        choices=("pinned_blocked", "authorized_correction"),
        default="pinned_blocked",
    )
    parser.add_argument("--correction-manifest-revision", type=int)
    parser.add_argument("--correction-pinned-implementation-sha")
    parser.add_argument(
        "--capacity-cpu-evidence",
        type=Path,
        default=DEFAULT_CPU_EVIDENCE,
    )
    parser.add_argument(
        "--reviewed-correction-manifest",
        type=Path,
        help="required when generating authorized_correction",
    )
    parser.add_argument("--review-artifact", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def _load_base(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(args: argparse.Namespace) -> dict:
    base_manifest = _load_base(args.base_manifest)
    generation_sha = git("rev-parse", "HEAD")
    generation_tree = git("rev-parse", "HEAD^{tree}")
    if args.status == "pinned_blocked":
        if not args.correction_pinned_implementation_sha:
            raise ValueError(
                "--correction-pinned-implementation-sha is required for "
                "pinned_blocked"
            )
        revision = args.correction_manifest_revision or 1
        pin = git(
            "rev-parse",
            f"{args.correction_pinned_implementation_sha}^{{commit}}",
        )
        if pin != args.correction_pinned_implementation_sha:
            raise ValueError("correction implementation pin did not resolve exactly")
        tree = git("rev-parse", f"{pin}^{{tree}}")
        if (
            subprocess.run(
                ("git", "merge-base", "--is-ancestor", pin, "HEAD"),
                cwd=REPO_ROOT,
                capture_output=True,
                check=False,
            ).returncode
            != 0
        ):
            raise ValueError("correction implementation pin is not an ancestor of HEAD")
        runner_sha = file_sha256(REPO_ROOT / CAPACITY_RUNNER_PATH)
        pinned_runner = subprocess.run(
            ("git", "show", f"{pin}:{CAPACITY_RUNNER_PATH}"),
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        ).stdout
        if file_sha256(REPO_ROOT / CAPACITY_RUNNER_PATH) != runner_sha:
            raise ValueError("capacity runner changed during manifest generation")
        if hashlib.sha256(pinned_runner).hexdigest() != runner_sha:
            raise ValueError("capacity runner differs from the correction pin")
        for relative in (BASE_MANIFEST_PATH, ORIGINAL_RAW_PATH):
            pinned_blob = subprocess.run(
                ("git", "show", f"{pin}:{relative}"),
                cwd=REPO_ROOT,
                capture_output=True,
                check=True,
            ).stdout
            if hashlib.sha256(pinned_blob).hexdigest() != file_sha256(
                REPO_ROOT / relative
            ):
                raise ValueError(f"{relative} differs from the correction pin")
        _, evidence_summary = load_capacity_cpu_evidence(
            args.capacity_cpu_evidence,
            runner_sha256=runner_sha,
            image_digest=base_manifest["environment"]["image_digest"],
            repo_root=REPO_ROOT,
        )
        manifest = build_pinned_capacity_correction_manifest(
            base_manifest=base_manifest,
            base_manifest_path=Path(BASE_MANIFEST_PATH),
            base_manifest_file_sha256=file_sha256(args.base_manifest),
            original_raw_file_sha256=file_sha256(REPO_ROOT / ORIGINAL_RAW_PATH),
            correction_manifest_revision=revision,
            correction_pinned_implementation_sha=pin,
            correction_pinned_tree_sha=tree,
            capacity_runner_sha256=runner_sha,
            capacity_cpu_evidence=evidence_summary,
            manifest_generation_sha=generation_sha,
            manifest_generation_tree_sha=generation_tree,
        )
    else:
        if args.reviewed_correction_manifest is None:
            raise ValueError(
                "--reviewed-correction-manifest is required for "
                "authorized_correction"
            )
        reviewed_manifest = json.loads(
            args.reviewed_correction_manifest.read_text(encoding="utf-8")
        )
        validate_capacity_correction_manifest(
            reviewed_manifest,
            base_manifest=base_manifest,
            require_authorized=False,
        )
        if reviewed_manifest.get("status") != "pinned_blocked":
            raise ValueError(
                "authorized_correction must supersede a pinned_blocked manifest"
            )
        verify_capacity_correction_files(
            reviewed_manifest,
            base_manifest=base_manifest,
            manifest_path=args.reviewed_correction_manifest,
            repo_root=REPO_ROOT,
            verify_git=True,
        )
        review = load_correction_review(args.review_artifact)
        revision = args.correction_manifest_revision or (
            int(reviewed_manifest["correction_manifest_revision"]) + 1
        )
        manifest = build_authorized_capacity_correction_manifest(
            reviewed_manifest=reviewed_manifest,
            review=review,
            review_path=args.review_artifact,
            repo_root=REPO_ROOT,
            correction_manifest_revision=revision,
            manifest_generation_sha=generation_sha,
            manifest_generation_tree_sha=generation_tree,
        )
    validate_capacity_correction_manifest(
        manifest,
        base_manifest=base_manifest,
        require_authorized=args.status == "authorized_correction",
        review=(
            load_correction_review(args.review_artifact)
            if args.status == "authorized_correction"
            else None
        ),
    )
    verify_capacity_correction_files(
        manifest,
        base_manifest=base_manifest,
        manifest_path=args.output,
        repo_root=REPO_ROOT,
        verify_git=True,
    )
    return manifest


def main() -> int:
    args = parse_args()
    if args.check:
        manifest = json.loads(args.output.read_text(encoding="utf-8"))
        base_manifest = _load_base(args.base_manifest)
        verify_capacity_correction_files(
            manifest,
            base_manifest=base_manifest,
            manifest_path=args.output,
            repo_root=REPO_ROOT,
            verify_git=True,
        )
        validate_capacity_correction_manifest(
            manifest,
            base_manifest=base_manifest,
            require_authorized=manifest.get("status") == "authorized_correction",
            review=(
                load_correction_review(
                    REPO_ROOT / manifest["review_evidence"]["artifact_path"]
                )
                if manifest.get("status") == "authorized_correction"
                else None
            ),
        )
        print(
            "OK: capacity correction manifest "
            f"{manifest['status']} revision "
            f"{manifest['correction_manifest_revision']}"
        )
        return 0
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {manifest['status']} capacity correction manifest revision "
        f"{manifest['correction_manifest_revision']} to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
