#!/usr/bin/env python3
"""Unpack the in-repo ImpactKV claim pack onto this machine.

No cluster mount. After this:

    export IMPACTKV_ARTIFACTS=$PWD/impactkv-artifacts

Frozen ``RESULT.json`` files are for the paper checker. GPU re-runs must copy
``PLAN.json`` into a *new* directory and must not overwrite job 137185 / 96092.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACK = HERE / "offcluster" / "impactkv-claim-pack.tar.gz"
MANIFEST = HERE / "offcluster" / "MANIFEST.json"

SEVEN_B = "impactkv_swebench_7b_file_modules_prefixkey_20260824"
THIRTY_B = "impactkv_swebench_prerotated_file_modules_20260818"
PREFIX_ON = "impactkv_swebench_7b_prefix_on_20260825"
PREFETCH = "impactkv_swebench_template_prefetch_nextisland_20260821"


def _dest(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    env = os.environ.get("IMPACTKV_ARTIFACTS")
    if env:
        return Path(env).expanduser().resolve()
    return (REPO / "impactkv-artifacts").resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=PACK)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    pack = args.pack.resolve()
    dest = _dest(args.dest)
    if not pack.is_file():
        raise FileNotFoundError(
            f"claim pack missing: {pack}\n"
            "Clone ccdd2023/sglang @ integration/template-prefetch-swebench."
        )
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".impactkv_claim_pack"
    if marker.exists() and not args.force:
        print(f"already unpacked at {dest} (pass --force to replace JSON, not RESULT locks)")
    else:
        print(f"unpacking {pack} -> {dest}")
        with tarfile.open(pack, "r:gz") as tar:
            try:
                tar.extractall(dest, filter="data")
            except TypeError:
                tar.extractall(dest)
        if MANIFEST.is_file():
            meta = json.loads(MANIFEST.read_text(encoding="utf-8"))
            expected = meta.get("pack_sha256")
            if expected and _sha256(pack) != expected:
                raise ValueError("claim pack sha256 mismatch vs offcluster/MANIFEST.json")
            for row in meta.get("files") or []:
                path = dest / row["path"]
                if _sha256(path) != row["sha256"]:
                    raise ValueError(f"sha256 mismatch: {row['path']}")
        marker.write_text("unpacked\n", encoding="utf-8")

    prefix = dest / PREFIX_ON
    prefix.mkdir(parents=True, exist_ok=True)
    src_plan = dest / SEVEN_B / "PLAN.json"
    dst_plan = prefix / "PLAN.json"
    if src_plan.is_file() and not dst_plan.exists():
        shutil.copyfile(src_plan, dst_plan)
        print(f"copied 7B PLAN -> {dst_plan}")

    prefetch = dest / PREFETCH
    prefetch.mkdir(parents=True, exist_ok=True)
    src_30 = dest / THIRTY_B / "PLAN.json"
    dst_30 = prefetch / "PLAN.json"
    if src_30.is_file() and not dst_30.exists():
        shutil.copyfile(src_30, dst_30)
        print(f"copied 30B PLAN -> {dst_30}")

    print()
    print(f"export IMPACTKV_ARTIFACTS={dest}")
    print("Frozen RESULT.json is the paper number. GPU re-runs:")
    print("  bash benchmark/multi_workflow/run_impactkv_headline.sh")
    print("  (writes a new directory under impactkv-artifacts/runs/)")


if __name__ == "__main__":
    main()
