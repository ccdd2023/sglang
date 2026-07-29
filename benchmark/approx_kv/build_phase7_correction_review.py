#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.approx_kv.phase7.correction import (
    BASE_MANIFEST_PATH,
    CAPACITY_CORRECTION_MANIFEST_PATH,
    CAPACITY_CORRECTION_REVIEW_PATH,
    validate_capacity_correction_manifest,
    verify_capacity_correction_files,
)
from benchmark.approx_kv.phase7.correction_review import (
    ACCEPTED_CORRECTION_VERDICTS,
    build_correction_review,
    write_correction_review,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviewed-manifest",
        type=Path,
        default=Path(CAPACITY_CORRECTION_MANIFEST_PATH),
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=Path(BASE_MANIFEST_PATH),
    )
    parser.add_argument(
        "--reviewer",
        default="Claude Opus 5 / Max Thinking / long context",
    )
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument(
        "--verdict",
        choices=ACCEPTED_CORRECTION_VERDICTS,
        required=True,
    )
    parser.add_argument("--open-p0", type=int, default=0)
    parser.add_argument("--open-p1", type=int, default=0)
    parser.add_argument("--finding", action="append", default=[], metavar="JSON")
    parser.add_argument("--disposition", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(CAPACITY_CORRECTION_REVIEW_PATH),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.resolve() != (REPO_ROOT / CAPACITY_CORRECTION_REVIEW_PATH).resolve():
        raise ValueError("correction review output path is frozen")
    manifest = json.loads(args.reviewed_manifest.read_text(encoding="utf-8"))
    base_manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    validate_capacity_correction_manifest(
        manifest,
        base_manifest=base_manifest,
        require_authorized=False,
    )
    if manifest.get("status") != "pinned_blocked":
        raise ValueError("correction review must review a pinned_blocked manifest")
    verify_capacity_correction_files(
        manifest,
        base_manifest=base_manifest,
        manifest_path=args.reviewed_manifest,
        repo_root=REPO_ROOT,
        verify_git=True,
    )
    findings = [json.loads(item) for item in args.finding]
    review = build_correction_review(
        reviewer=args.reviewer,
        model=args.model,
        verdict=args.verdict,
        open_p0=args.open_p0,
        open_p1=args.open_p1,
        reviewed_correction_manifest_revision=manifest["correction_manifest_revision"],
        reviewed_correction_manifest_sha256=manifest["correction_manifest_sha256"],
        base_manifest_revision=manifest["base_manifest_revision"],
        base_manifest_self_sha256=manifest["base_manifest_self_sha256"],
        base_manifest_design_sha256=manifest["base_manifest_design_sha256"],
        base_manifest_path=manifest["base_manifest_path"],
        reviewed_correction_pinned_implementation_sha=manifest[
            "correction_pinned_implementation_sha"
        ],
        reviewed_correction_pinned_tree_sha=manifest["correction_pinned_tree_sha"],
        capacity_runner_sha256=manifest["capacity_runner_sha256"],
        original_raw_sha256=manifest["original_raw_sha256"],
        scope=manifest["scope"],
        allowed_setting=manifest["allowed_setting"],
        restart=manifest["restart"],
        findings=findings,
        disposition=args.disposition,
        timestamp=args.timestamp,
    )
    write_correction_review(args.output, review)
    print(
        "wrote capacity correction review for manifest revision "
        f"{review['reviewed_correction_manifest_revision']} to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
