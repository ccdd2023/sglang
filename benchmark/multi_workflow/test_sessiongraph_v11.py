from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmark.multi_workflow.aggregate_sessiongraph_v11 import aggregate
from benchmark.multi_workflow.analyze_sessiongraph_atlas import Observation, analyze
from benchmark.multi_workflow.measure_sessiongraph_atlas import (
    PromptPair,
    _atlas_prompt_hash,
    build_prompt_pair,
    pending_design_rows,
)
from benchmark.multi_workflow.sessiongraph_raw_provenance import (
    parse_openhands,
    parse_sweagent,
)
from benchmark.multi_workflow.sessiongraph_v11 import (
    CostModel,
    assert_online_safe,
    build_label_rows,
    prompt_hash,
    select_fileversion_modules,
)


class ByteTokenizer:
    def encode(self, text: str, **_kwargs):
        return list(text.encode("utf-8"))


def _module(
    module_id: str,
    module_type: str,
    text: str,
    position: int,
    *,
    scope: str = "session",
    first_seen: int = 0,
    workspace: int = 0,
    dependencies=(),
):
    import hashlib

    start = position * 10
    return {
        "module_id": module_id,
        "module_type": module_type,
        "text": text,
        "position": position,
        "cache_scope": scope,
        "first_seen_turn": first_seen,
        "workspace_version": workspace,
        "dependencies": list(dependencies),
        "content_hash": hashlib.sha256(text.encode()).hexdigest(),
        "token_span": [start, start + max(5, len(text))],
    }


def _turn(turn_id: int, modules, *, workspace: int = 0):
    return {
        "session_id": "session-a",
        "turn_id": turn_id,
        "workspace_version": workspace,
        "modules": modules,
    }


def test_raw_provenance_resolves_editor_paths_and_fails_closed():
    events = parse_openhands(
        "s",
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "str_replace_editor",
                            "arguments": json.dumps(
                                {"command": "str_replace", "path": "pkg/a.py"}
                            ),
                        }
                    }
                ],
            }
        ],
    )
    assert events[0].classification == "resolved_python"
    assert events[0].changed_paths == ("pkg/a.py",)

    unresolved = parse_sweagent(
        "s",
        {"history": [{"role": "assistant", "action": "apply_patch"}]},
    )
    assert unresolved[0].classification == "global_fail_closed"


def test_online_path_rejects_gold_and_hidden_test_fields():
    assert_online_safe({"session_id": "s", "modules": [{"text": "public"}]})
    for sealed in (
        {"gold_patch": "answer"},
        {"nested": {"hidden_tests": ["x"]}},
        {"expected_replacement": "answer"},
    ):
        with pytest.raises(ValueError, match="sealed fields"):
            assert_online_safe(sealed)


def test_file_version_source_view_is_reused_only_without_later_write():
    source = _module(
        "event:0002",
        "source_view",
        "pkg/a.py contents",
        0,
        scope="workspace",
        workspace=0,
    )
    old_role = _module("role:0", "role_instruction", "old role", 0, scope="turn")
    new_role = _module("role:1", "role_instruction", "new role", 0, scope="turn")
    source = {**source, "position": 1, "token_span": [10, 27]}
    observation = _module(
        "event:0004", "tool_output", "current observation", 2, scope="turn"
    )
    target = _module("target:1", "target", "target", 3, first_seen=1)
    previous = _turn(0, [old_role, source], workspace=0)
    current = _turn(1, [new_role, source, observation, target], workspace=1)
    cost = CostModel(5.0, 0.4, 0.0, 1.0, 1.0, 1.0)
    selected, _ = select_fileversion_modules(
        previous,
        current,
        runtime_exact_ids={"event:0002"},
        source_view_resources={"event:0002": ("pkg/a.py",)},
        mutations={"event:0003": ()},
        cost_model=cost,
    )
    assert selected == ["event:0002"]
    selected, reasons = select_fileversion_modules(
        previous,
        current,
        runtime_exact_ids={"event:0002"},
        source_view_resources={"event:0002": ("pkg/a.py",)},
        mutations={"event:0003": ("pkg/a.py",)},
        cost_model=cost,
    )
    assert selected == []
    assert reasons["event:0002"] == "viewed_file_edited_later"


def test_label_controls_preserve_eligible_set_budget_and_islands():
    chunks = [
        {
            "module_id": f"m{index}",
            "module_type": "source_view" if index % 2 else "agent_message",
            "slot_id": f"session:m{index}",
            "chunk_signature": f"sig-{index}",
            "chunk_len": 10,
            "token_hash": f"hash-{index}",
            "position": index,
        }
        for index in range(8)
    ]
    modes = build_label_rows(
        case_id="s:t1", chunks=chunks, copied_module_ids={"m0", "m1", "m5"}
    )

    def identity(rows):
        return [
            (
                row["slot_id"],
                row["chunk_signature"],
                row["chunk_len"],
                row["token_hash"],
            )
            for row in rows
        ]

    def budget(rows):
        return sum(row["chunk_len"] - row["head_tokens"] for row in rows)

    def islands(rows):
        output, running = [], 0
        for row in rows:
            copied = row["chunk_len"] - row["head_tokens"]
            if copied:
                running += copied
            elif running:
                output.append(running)
                running = 0
        if running:
            output.append(running)
        return sorted(output)

    policy = modes["fileversion"]
    for rows in modes.values():
        assert identity(rows) == identity(policy)
        assert budget(rows) == budget(policy) == 30
        assert islands(rows) == islands(policy) == [10, 20]


def test_prompt_hash_and_prompt_pair_are_token_based_and_deterministic():
    assert prompt_hash([1, 2, 3]) == prompt_hash((1, 2, 3))
    assert prompt_hash([1, 2, 3]) != prompt_hash([1, 3, 2])
    tokenizer = ByteTokenizer()
    text = "aaaaabbbbbccccc"
    turn = {
        "session_id": "s1",
        "turn_id": 1,
        "rendered_prompt": text,
        "modules": [
            {**_module("m", "source_view", "bbbbb", 0), "token_span": [5, 10]},
            {**_module("target:1", "target", "ccccc", 1), "token_span": [10, 15]},
        ],
    }
    other = {
        "session_id": "s2",
        "turn_id": 1,
        "rendered_prompt": "xxxxxzzzzz",
        "modules": [],
    }
    pair = build_prompt_pair(
        tokenizer=tokenizer,
        turns={("s1", 1): turn, ("s2", 1): other},
        final_turns={"s1": turn, "s2": other},
        session_id="s1",
        module_id="m",
        disturbance="identity",
    )
    assert isinstance(pair, PromptPair)
    pair.validate()
    assert pair.source_ids == pair.target_ids


def test_atlas_prompt_hash_preserves_frozen_int32_encoding():
    tokens = [1, 2, 3, 151_643]
    expected = hashlib.sha256(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest()
    assert _atlas_prompt_hash(tokens) == expected


def test_resume_from_is_read_only_and_pending_plan_is_exact(tmp_path: Path):
    design = tmp_path / "design.jsonl"
    base = tmp_path / "base.jsonl"
    output = tmp_path / "delta.jsonl"
    rows = [
        {
            "cohort": "development",
            "session_id": "s",
            "module_id": "m",
            "disturbance": "position_only",
            "recompute_fraction": dose,
        }
        for dose in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    design.write_text("".join(json.dumps(row) + "\n" for row in rows))
    base.write_text("".join(json.dumps({**row, "status": "ok"}) + "\n" for row in rows[:3]))
    args = argparse.Namespace(
        design=design,
        cohort="development",
        disturbances="position_only",
        session_ids="",
        module_ids="",
        doses="",
        resume_from=[base],
        resume=True,
        output=output,
    )
    pending, summary = pending_design_rows(args)
    assert [row["recompute_fraction"] for row in pending] == [0.75, 1.0]
    assert summary["pending_rows"] == 2
    assert summary["pending_groups"] == 1
    assert not output.exists()


def _formal_artifacts(tmp_path: Path):
    disturbances = [
        "identity",
        "change_after",
        "upstream_edit",
        "semantic_prefix",
        "position_only",
        "module_reorder",
        "same_task",
        "cross_task",
    ]
    rows = []
    for disturbance in disturbances:
        modules = 3 if disturbance == "same_task" else 4
        for session in range(32):
            for module in range(modules):
                for dose in (0.0, 0.25, 0.5, 0.75, 1.0):
                    safe = module % 2 == 0
                    js = (
                        0.0
                        if disturbance in {"identity", "change_after"}
                        else (0.001 if safe else 0.01) * (1.0 - dose)
                    )
                    rows.append(
                        {
                            "cohort": "development",
                            "session_id": f"s{session:02d}",
                            "module_id": f"{disturbance}:m{module}",
                            "module_type": "source_view",
                            "cache_scope": "workspace",
                            "disturbance": disturbance,
                            "recompute_fraction": dose,
                            "token_count": 16 + module,
                            "position_norm": 0.2 + module / 10,
                            "rope_delta": module,
                            "prefix_changed_tokens": module,
                            "graph_distance": 2 if safe else 1,
                            "k_deviation": js,
                            "v_deviation": js,
                            "attention_mass": None,
                            "attention_mass_measured": False,
                            "teacher_logit_js": js,
                            "teacher_top1_changed": False,
                            "causal_splice_logit_js": js,
                            "lookup_ms": 0.1,
                            "source_tokens": 100,
                            "target_tokens": 100,
                            "source_prompt_hash": f"src-{session}-{module}",
                            "target_prompt_hash": f"dst-{session}-{module}",
                            "measurement_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                            "status": "ok",
                        }
                    )
    design = tmp_path / "design.jsonl"
    design.write_text("".join(json.dumps(row) + "\n" for row in rows))
    by_role = {
        "negative_controls": [
            row for row in rows if row["disturbance"] in {"identity", "change_after"}
        ],
        "upstream": [row for row in rows if row["disturbance"] == "upstream_edit"],
        "semantic_prefix": [
            row for row in rows if row["disturbance"] == "semantic_prefix"
        ],
    }
    remaining = [
        row
        for row in rows
        if row["disturbance"]
        in {"position_only", "module_reorder", "same_task", "cross_task"}
    ]
    by_role["remaining_base"] = remaining[:2220]
    by_role["remaining_delta"] = remaining[2220:]
    role_paths = {}
    for role, values in by_role.items():
        path = tmp_path / f"{role}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in values))
        role_paths[role] = path
    summaries = []
    for index in range(4):
        path = tmp_path / f"summary-{index}.json"
        value = {
            "passed": True,
            "errors": [],
            "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        }
        if index == 3:
            value.update(
                {
                    "dtype": "bfloat16",
                    "attention_implementation": "sdpa",
                    "splice_suffix_chunk_size": 512,
                }
            )
        path.write_text(json.dumps(value))
        summaries.append(path)
    amendment = tmp_path / "amendment.json"
    amendment.write_text(
        json.dumps(
            {
                "accepted": True,
                "attention_implementation": "sdpa",
                "dtype": "bfloat16",
                "splice_suffix_chunk_size": 512,
                "thresholds_changed": False,
            }
        )
    )
    return design, role_paths, summaries, amendment


def test_formal_aggregate_requires_exact_4960_design(tmp_path: Path):
    design, roles, summaries, amendment = _formal_artifacts(tmp_path)
    result = aggregate(
        design_path=design,
        role_paths=roles,
        summary_paths=summaries,
        executor_amendment_path=amendment,
        forbidden_paths=[],
        aggregate_output=tmp_path / "formal.jsonl",
        manifest_output=tmp_path / "manifest.json",
        gate_output=tmp_path / "gate.json",
        bootstrap=2,
    )
    assert result["artifact_validation_passed"]
    assert result["manifest"]["formal_rows"] == 4960
    assert result["manifest"]["duplicate_design_keys"] == 0

    forbidden = tmp_path / "invalid-790.jsonl"
    forbidden.write_text(roles["negative_controls"].read_text())
    with pytest.raises(ValueError, match="forbidden unchunked"):
        aggregate(
            design_path=design,
            role_paths={**roles, "negative_controls": forbidden},
            summary_paths=summaries,
            executor_amendment_path=amendment,
            forbidden_paths=[forbidden],
            aggregate_output=tmp_path / "unused.jsonl",
            manifest_output=tmp_path / "unused-manifest.json",
            gate_output=tmp_path / "unused-gate.json",
            bootstrap=1,
        )


def test_statistics_fail_closed_on_incomplete_formal_coverage():
    rows = [
        Observation(
            session_id="s1",
            module_id="m",
            module_type="source_view",
            cache_scope="workspace",
            disturbance="identity",
            recompute_fraction=0.0,
            token_count=8,
            position_norm=0.5,
            rope_delta=0,
            prefix_changed_tokens=0,
            graph_distance=2,
            causal_splice_logit_js=0.0,
            lookup_ms=0.1,
        )
    ]
    result = analyze(rows, iterations=2)
    assert not result["passed"]
    assert result["rows"] == 1
    assert any("formal coverage" in reason for reason in result["reasons"])
