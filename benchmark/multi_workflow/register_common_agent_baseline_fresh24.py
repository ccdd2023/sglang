#!/usr/bin/env python3
"""Outcome-blind registration for the common-agent native baseline comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.multi_workflow import (
    run_natural_code_cost_expanded_accuracy_campaign as selector,
)
from benchmark.multi_workflow.runtime_paths import RuntimePaths


PROJECT = Path(__file__).resolve().parents[2]
PATHS = RuntimePaths.from_project(PROJECT)
DEFAULT_OUTPUT = PATHS.artifacts / "impactkv_common_agent_baselines_fresh24_20260812"
SELECTION_SALT = "common-agent-native-baselines-qwen25-7b-fresh24-20260812-v1"
TASKS = 24
REPO_CAP = 4
DIFFICULTY_QUOTAS = {
    "<15 min fix": 10,
    "15 min - 1 hour": 10,
    "1-4 hours": 4,
}
CANARY_IDS = (
    "django__django-16631",
    "scikit-learn__scikit-learn-25232",
    "django__django-15268",
    "sphinx-doc__sphinx-7454",
)
ARMS = (
    "sglang_dense",
    "coding_dependency_graph_cold_lcb",
    "cacheblend_dense",
    "cacheblend_reuse",
    "kvcomm_dense",
    "kvcomm_reuse",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def select_fresh(population: list[dict[str, Any]], excluded: set[str]):
    saved = (
        selector.SELECTION_SALT,
        selector.TASKS,
        selector.REPO_CAP,
        selector.DIFFICULTY_QUOTAS,
    )
    try:
        selector.SELECTION_SALT = SELECTION_SALT
        selector.TASKS = TASKS
        selector.REPO_CAP = REPO_CAP
        selector.DIFFICULTY_QUOTAS = DIFFICULTY_QUOTAS
        return selector.select_cohort(population, excluded)
    finally:
        (
            selector.SELECTION_SALT,
            selector.TASKS,
            selector.REPO_CAP,
            selector.DIFFICULTY_QUOTAS,
        ) = saved


def prepare(output: Path) -> dict[str, Any]:
    registration = output / "CAMPAIGN_REGISTRATION.json"
    if registration.exists():
        return json.loads(registration.read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    population = json.loads(PATHS.population.read_text(encoding="utf-8"))
    by_id = {str(row["instance_id"]): row for row in population}
    missing_canary = set(CANARY_IDS).difference(by_id)
    if missing_canary:
        raise ValueError(f"canary tasks absent from population: {sorted(missing_canary)}")
    excluded, exclusion_audit = selector.historical_exclusions()
    formal = select_fresh(population, excluded)
    formal_ids = [str(row["instance_id"]) for row in formal]
    if excluded.intersection(formal_ids):
        raise AssertionError("historical task leaked into formal Fresh24")
    if Counter(str(row["difficulty"]) for row in formal) != Counter(DIFFICULTY_QUOTAS):
        raise AssertionError("formal difficulty quota changed")
    canary = [by_id[value] for value in CANARY_IDS]
    output.mkdir(parents=True)
    write_json(output / "CANARY4.json", canary)
    write_json(output / "FROZEN_FRESH24.json", formal)
    write_jsonl(output / "canary_dataset/test.jsonl", canary)
    write_jsonl(output / "formal_dataset/test.jsonl", formal)
    for name, rows in (("CANARY4", canary), ("FRESH24", formal)):
        write_json(
            output / f"BRIDGE_{name}_REGISTRATION.json",
            {
                "schema_version": 1,
                "registration_id": f"impactkv-common-agent-{name.lower()}-20260812",
                "dataset": {"name": "princeton-nlp/SWE-bench_Verified", "split": "test"},
                "instances": [
                    {"instance_id": str(row["instance_id"])} for row in rows
                ],
            },
        )
    source_paths = (
        PROJECT / "benchmark/multi_workflow/bridge_reuse_litellm_model.py",
        PROJECT / "benchmark/multi_workflow/run_common_native_agent_experiment.py",
        PROJECT / "benchmark/multi_workflow/swebench_common_qwen25_agent.yaml",
        PROJECT / "benchmark/multi_workflow/qwen2_5_coder_tool_chat_template.jinja",
        Path(__file__).resolve(),
    )
    value = {
        "status": "REGISTERED_BEFORE_FRESH24_MODEL_OUTCOMES",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "compare official coding accuracy and causal TTFT for the current "
            "coding-aware method, native CacheBlend, and native KVCOMM under one "
            "rolling-agent prompt contract"
        ),
        "model": {
            "name": "Qwen2.5-Coder-7B-Instruct",
            "dtype": "float16",
            "tensor_parallel": 1,
        },
        "arms": list(ARMS),
        "protocol": {
            "agent": "mini-SWE-agent 2.3.0 rolling6 with one bash tool",
            "step_limit": 32,
            "temperature": 0,
            "prompt_token_limit": 28000,
            "context_length": 32768,
            "max_new_tokens": 2048,
            "accuracy": "official SWE-bench resolved / 24",
            "causal_ttft": "frozen identical input_ids, two warmups, ten ABBA rounds",
            "amortization": [1, 4, 16],
            "prefetch": False,
        },
        "baseline_parameters": {
            "cacheblend": {"recompute_ratio": 0.18},
            "kvcomm": {"threshold": 0.3, "window_size": 5, "max_anchors": 20},
            "tuning_on_formal_tasks": False,
        },
        "canary": {
            "classification": "historical high-opportunity infrastructure-only",
            "instances": list(CANARY_IDS),
            "enters_formal_accuracy": False,
        },
        "formal_selection": {
            "salt": SELECTION_SALT,
            "outcome_used_for_selection": False,
            "historical_excluded_tasks": len(excluded),
            "exclusion_audit": exclusion_audit,
            "repository_cap": REPO_CAP,
            "difficulty_quotas": DIFFICULTY_QUOTAS,
            "capacity_amendment_before_registration": (
                "After excluding all 207 historically exposed tasks, only eight "
                "unseen 1-4 hour tasks remain and all are from django/django. "
                "The repository cap of four therefore makes the intended 10/9/5 "
                "allocation impossible. The closest feasible allocation, chosen "
                "without model outcomes, is 10/10/4."
            ),
            "instances": [
                {
                    "instance_id": str(row["instance_id"]),
                    "repo": str(row["repo"]),
                    "difficulty": str(row["difficulty"]),
                }
                for row in formal
            ],
        },
        "identity_gate": {
            "same_initial_agent_messages": True,
            "same_qwen25_chat_template": True,
            "backend_must_echo_input_ids_sha256": True,
            "future_requests_may_diverge_after_first_output_difference": True,
        },
        "protected": {
            "old_dirty_checkout_modified": False,
            "paper_modified": False,
            "old_preregistration_thresholds_modified": False,
        },
        "inputs": {
            "population": str(PATHS.population),
            "population_sha256": sha256(PATHS.population),
            "canary_sha256": sha256(output / "CANARY4.json"),
            "fresh24_sha256": sha256(output / "FROZEN_FRESH24.json"),
            "source_sha256": {
                str(path.relative_to(PROJECT)): sha256(path) for path in source_paths
            },
            "sglang_commit": git_head(PROJECT),
        },
    }
    write_json(registration, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = prepare(args.output)
    print(json.dumps(value["formal_selection"]["instances"], indent=2))


if __name__ == "__main__":
    main()
