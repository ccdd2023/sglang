#!/usr/bin/env python3
"""Offline class template from a coding-agent DYNAMIC_MANIFEST.

Labels are computed on a held training corpus of the *class*, not to
serve those issues again. A source is positive if some later case copies
it at Δ≠0. Output is a small Beta table keyed by task class.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sglang.srt.mem_cache.coding_aware.online_admit import (
    SourceObservation,
    protocol_later_roles,
)
from sglang.srt.mem_cache.coding_aware.online_template import (
    ClassBin,
    OnlineFileTemplate,
    featurize,
    task_class_id,
)


def _observation(source: dict[str, Any]) -> SourceObservation:
    policy = str(source.get("policy_label") or "coding")
    length = max(int(source.get("length") or 1), 1)
    return SourceObservation(
        source_id=str(source["source_id"]),
        source_start=int(source.get("source_start") or 1),
        token_ids=(1,) * length,
        content_hash=str(source.get("content_hash") or source["source_id"]),
        source_prefix_hash=str(
            source.get("source_prefix_token_hash") or source["source_id"]
        ),
        single_file_repository_code=True,
        version_valid=True,
        later_roles_in_protocol=protocol_later_roles(policy),
        policy_label=policy,
    )


def shifted_source_ids(cases: list[dict[str, Any]]) -> set[str]:
    hits: set[str] = set()
    for case in cases:
        if int(case["target_start"]) == int(case["source_start"]):
            continue
        hits.add(str(case["source_id"]))
    return hits


def compile_template(
    sources: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> OnlineFileTemplate:
    positives = shifted_source_ids(cases)
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for source in sources:
        key = featurize(_observation(source))
        counts[key][1] += 1
        if str(source["source_id"]) in positives:
            counts[key][0] += 1
    template = OnlineFileTemplate()
    for key, (pos, n) in counts.items():
        template._bins[key] = ClassBin(
            feature_key=key,
            alpha=float(pos + 1),
            beta=float(n - pos + 1),
            offline_n=n,
        )
    return template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    template = compile_template(manifest["sources"], manifest["cases"])
    payload = template.to_json()
    payload["task_classes"] = sorted(
        {task_class_id(str(row.get("policy_label") or "coding")) for row in manifest["sources"]}
    )
    payload["not_job_137185"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"bins": payload["bins"], "task_classes": payload["task_classes"]}, indent=2))


if __name__ == "__main__":
    main()
