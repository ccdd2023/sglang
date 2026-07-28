#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.approx_kv.build_phase7_manifest import (
    DEFAULT_OUTPUT,
    RUNNER_SPECS,
    design_payload_sha256,
    payload_sha256,
    sha256_file,
)
from benchmark.approx_kv.phase7.review import (
    ACCEPTED_VERDICTS,
    build_final_review,
    validate_review_binding,
    write_final_review,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed-manifest", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reviewer",
        default="Claude Opus 5 / Max Thinking / long context",
    )
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--verdict", choices=ACCEPTED_VERDICTS, required=True)
    parser.add_argument("--open-p0", type=int, default=0)
    parser.add_argument("--open-p1", type=int, default=0)
    parser.add_argument(
        "--finding",
        action="append",
        default=[],
        metavar="JSON",
        help=(
            "one JSON object per finding with finding_id, severity, summary "
            "and disposition"
        ),
    )
    parser.add_argument("--disposition", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def reviewed_runner_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    hashes = {}
    for name, spec in RUNNER_SPECS.items():
        entry = manifest.get("runners", {}).get(name)
        if not isinstance(entry, dict) or not entry.get("sha256"):
            raise ValueError(f"reviewed manifest lacks a pinned {name} runner hash")
        observed = sha256_file(REPO_ROOT / spec["path"])
        if entry["sha256"] != observed:
            raise ValueError(
                f"{name} runner changed after the reviewed manifest was pinned"
            )
        hashes[name] = entry["sha256"]
    return hashes


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.reviewed_manifest.read_text(encoding="utf-8"))
    if manifest.get("preregistered_manifest_sha256") != payload_sha256(manifest):
        raise ValueError("reviewed manifest self-hash mismatch")
    if manifest.get("design_payload_sha256") != design_payload_sha256(manifest):
        raise ValueError("reviewed manifest design payload hash mismatch")
    implementation = manifest.get("implementation", {})
    pinned_sha = implementation.get("phase7_pinned_implementation_sha")
    pinned_tree = implementation.get("phase7_pinned_tree_sha")
    if not pinned_sha or not pinned_tree:
        raise ValueError("reviewed manifest does not pin an implementation commit")
    findings = [json.loads(item) for item in args.finding]
    payload = build_final_review(
        reviewer=args.reviewer,
        model=args.model,
        verdict=args.verdict,
        open_p0=args.open_p0,
        open_p1=args.open_p1,
        reviewed_manifest_revision=int(manifest["manifest_revision"]),
        reviewed_manifest_sha256=manifest["preregistered_manifest_sha256"],
        design_payload_sha256=manifest["design_payload_sha256"],
        reviewed_pinned_implementation_sha=pinned_sha,
        reviewed_pinned_tree_sha=pinned_tree,
        runner_sha256=reviewed_runner_hashes(manifest),
        findings=findings,
        disposition=args.disposition,
        timestamp=args.timestamp,
    )
    validate_review_binding(
        payload,
        design_payload_sha256=manifest["design_payload_sha256"],
        supersedes_manifest_sha256=manifest["preregistered_manifest_sha256"],
        manifest_revision=int(manifest["manifest_revision"]) + 1,
        pinned_implementation_sha=pinned_sha,
        pinned_tree_sha=pinned_tree,
        runner_sha256={
            name: manifest["runners"][name]["sha256"] for name in RUNNER_SPECS
        },
    )
    write_final_review(args.output, payload)
    print(
        f"wrote final Opus review for manifest revision "
        f"{payload['reviewed_manifest_revision']} to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
