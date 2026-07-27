#!/usr/bin/env python3
"""Materialize the registered SWE-bench Verified rows at the frozen revision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


HERE = Path(__file__).resolve().parent
DEFAULT_REGISTRATION = HERE / "swebench_verified_complex_v1.json"
DEFAULT_OUTPUT = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "swebench_verified_complex_v1_20260724/frozen_subset.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    dataset = registration["dataset"]
    registered_ids = [
        instance["instance_id"] for instance in registration["instances"]
    ]
    rows = load_dataset(
        dataset["name"],
        split=dataset["split"],
        revision=dataset["revision_observed"],
    )
    by_id = {row["instance_id"]: dict(row) for row in rows}
    missing = sorted(set(registered_ids) - set(by_id))
    if missing:
        raise KeyError(f"registered instances absent from frozen revision: {missing}")

    frozen = [by_id[instance_id] for instance_id in registered_ids]
    payload = json.dumps(frozen, indent=2, ensure_ascii=False) + "\n"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    registered_digest = dataset.get("local_snapshot_sha256")
    if registered_digest and digest != registered_digest:
        raise ValueError(
            "frozen subset digest does not match registration: "
            f"{digest} != {registered_digest}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    args.output.with_suffix(".sha256").write_text(
        f"{digest}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(f"wrote {len(frozen)} instances to {args.output}")
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
