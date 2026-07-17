#!/usr/bin/env python3
"""Independently validate frozen V11 labels and provenance artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.sessiongraph_v11 import PROFILE, read_jsonl


MODES = ("fileversion", "uniform", "shuffled", "type_only")


def _by_request(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["case_id"])].append(row)
    return dict(output)


def _eligible(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["slot_id"]),
        str(row["chunk_signature"]),
        int(row["chunk_len"]),
        str(row["token_hash"]),
    )


def _budget(rows: Iterable[Mapping[str, Any]]) -> int:
    return sum(int(row["chunk_len"]) - int(row["head_tokens"]) for row in rows)


def _islands(rows: Iterable[Mapping[str, Any]]) -> list[int]:
    output, running = [], 0
    for row in rows:
        copied = int(row["chunk_len"]) - int(row["head_tokens"])
        if copied:
            running += copied
        elif running:
            output.append(running)
            running = 0
    if running:
        output.append(running)
    return sorted(output)


def validate(
    *,
    provenance_gate: Path,
    mutations_path: Path,
    capacity_gate: Path,
    registration_path: Path,
    label_gate: Path,
    prompts_path: Path,
    labels_dir: Path,
) -> dict[str, Any]:
    provenance = json.loads(provenance_gate.read_text(encoding="utf-8"))
    capacity = json.loads(capacity_gate.read_text(encoding="utf-8"))
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    build_gate = json.loads(label_gate.read_text(encoding="utf-8"))
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    mutations = read_jsonl(mutations_path)
    rows = {
        mode: read_jsonl(labels_dir / f"{mode}.jsonl") for mode in MODES
    }
    grouped = {mode: _by_request(values) for mode, values in rows.items()}
    requests = set(grouped["fileversion"])
    violations = []
    for mode in MODES[1:]:
        if set(grouped[mode]) != requests:
            violations.append({"kind": "request_set_mismatch", "mode": mode})
    for case_id in sorted(requests):
        policy = grouped["fileversion"][case_id]
        keys = [_eligible(row) for row in policy]
        budget = _budget(policy)
        islands = _islands(policy)
        for mode in MODES:
            selected = grouped[mode].get(case_id, [])
            if [_eligible(row) for row in selected] != keys:
                violations.append(
                    {"kind": "eligible_set_mismatch", "case_id": case_id, "mode": mode}
                )
            if _budget(selected) != budget:
                violations.append(
                    {"kind": "integer_budget_mismatch", "case_id": case_id, "mode": mode}
                )
            if _islands(selected) != islands:
                violations.append(
                    {"kind": "island_length_mismatch", "case_id": case_id, "mode": mode}
                )
            if any(
                not 0 <= int(row["head_tokens"]) <= int(row["chunk_len"])
                for row in selected
            ):
                violations.append(
                    {"kind": "head_out_of_range", "case_id": case_id, "mode": mode}
                )
    expected_profiles = {
        "fileversion": PROFILE,
        **{
            mode: f"{mode}-matched-fileversion-v11"
            for mode in MODES
            if mode != "fileversion"
        },
    }
    for mode, profile in expected_profiles.items():
        if {str(row["policy_profile"]) for row in rows[mode]} != {profile}:
            violations.append({"kind": "policy_profile_mismatch", "mode": mode})
    prompt_ids = {str(row["impact_case_id"]) for row in prompts}
    if prompt_ids != requests:
        violations.append({"kind": "prompt_request_set_mismatch"})
    if len({str(row["target_prompt_hash"]) for row in prompts}) != len(prompts):
        violations.append({"kind": "target_prompt_hash_collision"})
    canonical_hash = hashlib.sha256(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in mutations
        ).encode("utf-8")
    ).hexdigest()
    if canonical_hash != provenance["canonical_manifest_sha256"]:
        violations.append({"kind": "canonical_manifest_sha_mismatch"})
    if any(row["classification"] == "global_fail_closed" for row in mutations):
        violations.append({"kind": "global_fail_closed_events"})
    return {
        "passed": not violations
        and provenance.get("passed") is True
        and capacity.get("passed") is True
        and build_gate.get("passed") is True
        and registration.get("policy") == PROFILE,
        "turn_requests": len(requests),
        "label_rows": {mode: len(values) for mode, values in rows.items()},
        "exact_eligible_set_per_request": not any(
            row["kind"] == "eligible_set_mismatch" for row in violations
        ),
        "exact_integer_budget_per_request": not any(
            row["kind"] == "integer_budget_mismatch" for row in violations
        ),
        "exact_island_lengths_per_request": not any(
            row["kind"] == "island_length_mismatch" for row in violations
        ),
        "prompt_artifact_shared_across_modes": True,
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "provenance_gate",
        "mutations",
        "capacity_gate",
        "registration",
        "label_gate",
        "prompts",
        "labels_dir",
        "gate_output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        provenance_gate=args.provenance_gate,
        mutations_path=args.mutations,
        capacity_gate=args.capacity_gate,
        registration_path=args.registration,
        label_gate=args.label_gate,
        prompts_path=args.prompts,
        labels_dir=args.labels_dir,
    )
    args.gate_output.parent.mkdir(parents=True, exist_ok=True)
    args.gate_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
