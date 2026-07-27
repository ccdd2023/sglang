from __future__ import annotations

from transformers import AutoTokenizer

from benchmark.multi_workflow.probe_v16_behavioral_contract_kv import (
    MODEL,
    semantic_masks,
)


def test_behavior_mask_prioritizes_task_contract_over_signature() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    text = (
        "Planner stage: inspect the task.\n"
        "# Task: Return -1 if n is greater than m; otherwise return the "
        "rounded average.\n"
        "# Required public interface:\n"
        "# def rounded_avg(n, m):\n"
        "Implementer stage: draft code.\n"
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    masks = semantic_masks(
        case_id="synthetic-contract",
        segment_ids=ids,
        tokenizer=tokenizer,
        budget=8,
    )
    behavior_text = tokenizer.decode(
        [ids[index] for index in masks["behavior32"]]
    )
    signature_text = tokenizer.decode(
        [ids[index] for index in masks["signature32"]]
    )

    assert "Task" in behavior_text or "if" in behavior_text
    assert "def" in signature_text or "interface" in signature_text
    assert len(masks["behavior32"]) == 8
    assert len(set(masks["behavior32"])) == 8
    assert masks["behavior32"] != masks["signature32"]


def test_equal_budget_generic_masks_are_deterministic() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    ids = tokenizer("alpha beta gamma delta " * 20)["input_ids"]

    first = semantic_masks(
        case_id="stable-seed",
        segment_ids=ids,
        tokenizer=tokenizer,
        budget=12,
    )
    second = semantic_masks(
        case_id="stable-seed",
        segment_ids=ids,
        tokenizer=tokenizer,
        budget=12,
    )

    assert first == second
    assert all(len(mask) == 12 for mask in first.values())
