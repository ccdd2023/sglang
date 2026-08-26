from __future__ import annotations

from sglang.srt.mem_cache.kvcomm.types import token_ids_hash

from benchmark.multi_workflow.prepare_online_admit_plan import compile_group, compile_plan


def _group() -> dict:
    source_ids = [1, 2, 3, 4, 5, 8, 9]
    target_ids = [1, 7, 2, 3, 4, 5, 9]
    island = source_ids[2:5]
    return {
        "group_index": 0,
        "target_input_ids": target_ids,
        "target_prompt_hash": token_ids_hash(target_ids),
        "source_input_ids": [source_ids],
        "source_prompt_hashes": [token_ids_hash(source_ids)],
        "policy_label": "coding_natural_code_cost",
        "sources": [
            {
                "source_id": "src0",
                "source_prompt_hash": token_ids_hash(source_ids),
                "segment_token_hash": token_ids_hash(island),
                "source_prefix_token_hash": token_ids_hash(source_ids[:2]),
                "source_start": 2,
                "length": 3,
                "content_hash": token_ids_hash(island),
                "policy_label": "coding_natural_code_cost",
                "pre_rotate_delta": 99,
            }
        ],
        "cases": [
            {
                "source_id": "src0",
                "target_start": 4,
                "target_uses": 4,
            }
        ],
    }


def test_compile_group_ignores_planned_t_and_clears_pre_rotate():
    compiled = compile_group(_group())
    assert compiled["sources"][0]["pre_rotate_delta"] == 0
    assert compiled["pre_rotate_delta"] == 0
    assert compiled["online_admit"] is True
    assert compiled["islands"] == 1
    case = compiled["cases"][0]
    assert case["target_start"] == 3
    assert case["source_start"] == 2
    assert case["pre_rotate_delta"] == 0
    assert compiled["online_recovery"]["planned_t_mismatch"] == 1
    assert compiled["online_recovery"]["planned_t_match"] == 0


def test_zero_shift_is_not_a_copy_case():
    source_ids = [1, 2, 3, 4, 5, 8, 9]
    row = _group()
    row["target_input_ids"] = list(source_ids)
    row["target_prompt_hash"] = token_ids_hash(source_ids)
    compiled = compile_group(row)
    assert compiled["cases"] == []
    assert compiled["online_recovery"]["zero_shift"] == 1


def test_ambiguous_span_stays_dense():
    source_ids = [1, 2, 3, 4, 5, 8, 9]
    island = source_ids[2:5]
    row = _group()
    row["target_input_ids"] = [9, *island, 0, *island, 1]
    compiled = compile_group(row)
    assert compiled["cases"] == []
    assert compiled["online_recovery"]["not_in_target"] == 1


def test_compile_plan_refuses_existing_plan_json(tmp_path):
    import json
    import sys

    import pytest

    dest = tmp_path / "run"
    dest.mkdir()
    (dest / "LAUNCH.txt").write_text("ok\n")
    official = tmp_path / "official.json"
    official.write_text(json.dumps({"groups": [_group()]}))
    argv = sys.argv
    try:
        sys.argv = [
            "prepare_online_admit_plan.py",
            "--official-plan",
            str(official),
            "--output-dir",
            str(dest),
        ]
        from benchmark.multi_workflow import prepare_online_admit_plan as mod

        mod.main()
        assert (dest / "PLAN.json").is_file()
        with pytest.raises(FileExistsError):
            mod.main()
    finally:
        sys.argv = argv


def test_compile_plan_keeps_every_target_group():
    plan = compile_plan({"groups": [_group(), _group()]})
    assert plan["online_admit"] is True
    assert plan["not_job_137185"] is True
    assert plan["prefetch"] is False
    assert len(plan["groups"]) == 2
    assert plan["online_recovery"]["online_copy"] == 2
    assert plan["online_recovery"]["groups"] == 2
