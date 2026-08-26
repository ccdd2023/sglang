#!/usr/bin/env python3
"""Rebuild a 7B PLAN with source-time admit and target-time bind.

Does not read planned target_start when deciding what to lease. Sources
keep pre_rotate_delta=0. Cases are whatever unique token-identity locate
finds on the already-known target prompt. Does not overwrite the frozen
137185 PLAN/RESULT.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from sglang.srt.mem_cache.coding_aware.online_admit import (
    BindAction,
    LeasedIsland,
    SourceObservation,
    admit_source_island,
    bind_leased_islands,
    protocol_later_roles,
)
from sglang.srt.mem_cache.kvcomm.types import token_ids_hash

POLICY = "coding_natural_code_cost"
MODEL_7B = "Qwen2.5-Coder-7B-Instruct"


def _source_ids_for_row(
    row: dict[str, Any], source: dict[str, Any], index: int
) -> list[int]:
    hashes = list(row.get("source_prompt_hashes") or [])
    ids_list = list(row["source_input_ids"])
    prompt_hash = str(source["source_prompt_hash"])
    if prompt_hash in hashes:
        return [int(value) for value in ids_list[hashes.index(prompt_hash)]]
    return [int(value) for value in ids_list[index]]


def compile_group(row: dict[str, Any]) -> dict[str, Any]:
    """Admit from sources only; bind cases by locating tokens in the target."""
    target_ids = [int(value) for value in row["target_input_ids"]]
    policy = str(row.get("policy_label") or POLICY)
    later_roles = protocol_later_roles(policy)
    leases: list[LeasedIsland] = []
    sources_out: list[dict[str, Any]] = []
    skipped: list[str] = []
    for index, source in enumerate(row["sources"]):
        ids = _source_ids_for_row(row, source, index)
        start = int(source["source_start"])
        length = int(source["length"])
        island = tuple(ids[start : start + length])
        obs = SourceObservation(
            source_id=str(source["source_id"]),
            source_start=start,
            token_ids=island,
            content_hash=str(source["content_hash"]),
            source_prefix_hash=str(source["source_prefix_token_hash"]),
            single_file_repository_code=True,
            version_valid=True,
            later_roles_in_protocol=later_roles,
            seq=index,
            policy_label=policy,
        )
        reason = admit_source_island(obs)
        if reason is not None:
            skipped.append(f"{obs.source_id}:{reason}")
            continue
        rewritten = copy.deepcopy(source)
        rewritten["pre_rotate_delta"] = 0
        rewritten["policy_label"] = policy
        sources_out.append(rewritten)
        leases.append(
            LeasedIsland(
                source_id=obs.source_id,
                source_start=start,
                token_ids=island,
                content_hash=obs.content_hash,
                source_prefix_hash=obs.source_prefix_hash,
                seq=index,
            )
        )
    binds = bind_leased_islands(target_ids, leases)
    oracle = {
        str(case["source_id"]): int(case["target_start"])
        for case in row.get("cases") or []
    }
    cases_out: list[dict[str, Any]] = []
    recovery = {
        "oracle_islands": len(oracle),
        "online_copy": 0,
        "planned_t_match": 0,
        "planned_t_mismatch": 0,
        "not_in_target": 0,
        "zero_shift": 0,
        "admit_skip": len(skipped),
    }
    copied = 0
    uses = int((row.get("cases") or [{}])[0].get("target_uses") or 4)
    group_id = str(row.get("group_index", 0))
    for island_index, bind in enumerate(binds):
        if bind.action is BindAction.DENSE and bind.reason == "not_in_target":
            recovery["not_in_target"] += 1
            continue
        if bind.action is BindAction.DROP and bind.reason == "zero_shift":
            recovery["zero_shift"] += 1
            continue
        if bind.action is not BindAction.COPY:
            continue
        assert bind.target_start is not None and bind.source_id is not None
        recovery["online_copy"] += 1
        planned = oracle.get(bind.source_id)
        if planned == bind.target_start:
            recovery["planned_t_match"] += 1
        elif planned is not None:
            recovery["planned_t_mismatch"] += 1
        source = next(
            item for item in sources_out if item["source_id"] == bind.source_id
        )
        copied += bind.length
        cases_out.append(
            {
                "case_id": f"online-g{int(group_id):03d}-i{island_index}",
                "source_id": bind.source_id,
                "source_prompt_hash": source["source_prompt_hash"],
                "target_prompt_hash": token_ids_hash(target_ids),
                "segment_token_hash": token_ids_hash(
                    target_ids[bind.target_start : bind.target_start + bind.length]
                ),
                "source_prefix_token_hash": source["source_prefix_token_hash"],
                "target_prefix_token_hash": token_ids_hash(
                    target_ids[: bind.target_start]
                ),
                "source_start": bind.source_start,
                "target_start": bind.target_start,
                "length": bind.length,
                "content_hash": bind.content_hash,
                "policy_label": policy,
                "target_group_id": f"online-g{int(group_id):03d}",
                "target_uses": uses,
                "allow_shifted_copy": True,
                "pre_rotate_delta": 0,
            }
        )
    group = copy.deepcopy(row)
    group["sources"] = sources_out
    group["cases"] = cases_out
    group["islands"] = len(cases_out)
    group["copied_tokens"] = copied
    group["pre_rotate_delta"] = 0
    group["online_admit"] = True
    group["online_recovery"] = recovery
    group["admit_skip"] = skipped
    group["policy_label"] = policy
    return group


def compile_plan(official: dict[str, Any]) -> dict[str, Any]:
    groups = [compile_group(row) for row in official["groups"]]
    recovery = {
        "oracle_islands": 0,
        "online_copy": 0,
        "planned_t_match": 0,
        "planned_t_mismatch": 0,
        "not_in_target": 0,
        "zero_shift": 0,
        "admit_skip": 0,
        "groups": len(groups),
        "groups_with_copy": sum(1 for row in groups if row["islands"] > 0),
    }
    for row in groups:
        stats = row["online_recovery"]
        for key in recovery:
            if key in stats:
                recovery[key] += int(stats[key])
    return {
        "model": official.get("model") or MODEL_7B,
        "tokenizer": official.get("tokenizer") or MODEL_7B,
        "not_30b_swebench_plan": True,
        "same_token_ids_as_96092": False,
        "retokenized_from_30b": official.get("retokenized_from_30b", True),
        "policy_label": POLICY,
        "prefetch": False,
        "ordinary_prefix_reuse": False,
        "online_admit": True,
        "source_pre_rotate": False,
        "not_job_137185": True,
        "online_recovery": recovery,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    official = json.loads(args.official_plan.read_text(encoding="utf-8"))
    frozen_name = str(args.official_plan)
    if "prefixkey_20260824" in frozen_name and output.name == "prefixkey_20260824":
        raise ValueError("refusing to write into the frozen 137185 PLAN directory")
    plan = compile_plan(official)
    output.mkdir(parents=True, exist_ok=True)
    (output / "PLAN.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan["online_recovery"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
