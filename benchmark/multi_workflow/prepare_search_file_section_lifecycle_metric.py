#!/usr/bin/env python3
"""Freeze a source-lifecycle build metric before search exact results exist."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
CAMPAIGN = (
    RuntimePaths.from_project(PROJECT).artifacts
    / "impactkv_common_agent_search_file_section_20260812"
)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_usage(plan: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group in plan.get("groups") or ():
        source_ids = {
            str(case["source_id"]) for case in group.get("cases") or ()
        }
        if len(source_ids) != 1:
            raise ValueError("lifecycle metric requires one source per target group")
        source_id = next(iter(source_ids))
        counts[source_id] = counts.get(source_id, 0) + 1
    return counts


def main() -> None:
    root = CAMPAIGN / "exact_prompt_replay/canary4/sglang_coding"
    plan_path = root / "PLAN.json"
    output = CAMPAIGN / "SEARCH_FILE_SECTION_LIFECYCLE_METRIC_REGISTRATION.json"
    if output.is_file():
        print(output)
        return
    if (root / "RESULT.json").exists():
        raise RuntimeError("cannot register lifecycle metric after exact result")
    plan = read(plan_path)
    usage = source_usage(plan)
    write(
        output,
        {
            "schema_version": 1,
            "status": "FROZEN_BEFORE_EXACT_RESULT",
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "disclosure": (
                "Registered after the AB Dense pass may have completed, without "
                "reading any raw TTFT, and before reuse completion or RESULT.json. "
                "This is a secondary lifecycle metric; the preregistered cache-ready "
                "and accuracy gates remain unchanged."
            ),
            "motivation": (
                "The 57 online target groups consume only three persistent source "
                "handles. Charging one build to every target group overstates the "
                "online lifecycle cost, because a source is built once and remains "
                "resident for later distinct prompts."
            ),
            "topology": {
                "target_groups": sum(usage.values()),
                "distinct_targeted_sources": len(usage),
                "target_uses_per_source": sorted(usage.values()),
            },
            "metric": {
                "source_build_estimator": (
                    "median measured build for each source_id across redundant AB/BA "
                    "materializations; count that estimate once per lifecycle"
                ),
                "target_estimator": (
                    "mean exact-token TTFT per target group pooled across AB/BA"
                ),
                "observed_lifecycle_n1": (
                    "sum dense target TTFT / (sum reuse target TTFT + one build per "
                    "distinct source_id)"
                ),
                "n4_n16": (
                    "repeat the frozen set of target uses 4 or 16 times while keeping "
                    "one source build per persistent lifecycle"
                ),
                "macro_balance": (
                    "median lifecycle speedup over the three targeted source_ids"
                ),
            },
            "inputs": {"plan_sha256": sha256(plan_path)},
            "protected": {
                "primary_gate_changed": False,
                "task_selection_changed": False,
                "prompt_tokens_changed": False,
                "prefetch": False,
            },
        },
    )
    print(output)


if __name__ == "__main__":
    main()
