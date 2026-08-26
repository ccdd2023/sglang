from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.multi_workflow.compile_class_template import (
    compile_template,
    replay_finetune,
)
from sglang.srt.mem_cache.coding_aware.online_admit import SourceObservation
from sglang.srt.mem_cache.coding_aware.online_template import OnlineFileTemplate


def test_compile_counts_shifted_cases_per_class_not_per_file():
    sources = [
        {
            "source_id": "a",
            "source_start": 10,
            "length": 8,
            "content_hash": "file-a",
            "source_prefix_token_hash": "p",
            "policy_label": "coding_natural_code_cost",
        },
        {
            "source_id": "b",
            "source_start": 4,
            "length": 5,
            "content_hash": "file-b",
            "source_prefix_token_hash": "p",
            "policy_label": "coding_natural_code_cost",
        },
    ]
    cases = [
        {"source_id": "a", "source_start": 10, "target_start": 40},
        {"source_id": "b", "source_start": 4, "target_start": 4},
    ]
    template = compile_template(sources, cases)
    bin_ = template._bins["coding_agent"]
    assert bin_.offline_n == 2
    assert bin_.alpha == 2
    assert bin_.beta == 2


def test_replay_finetune_on_tiny_group_raises_copied_count():
    from sglang.srt.mem_cache.kvcomm.types import token_ids_hash

    source_ids = [1, 2, 3, 4, 5, 8, 9]
    target_ids = [1, 7, 2, 3, 4, 5, 9]
    island = source_ids[2:5]
    group = {
        "group_index": 0,
        "target_input_ids": target_ids,
        "source_input_ids": [source_ids],
        "source_prompt_hashes": [token_ids_hash(source_ids)],
        "policy_label": "coding_natural_code_cost",
        "sources": [
            {
                "source_id": "src0",
                "source_prompt_hash": token_ids_hash(source_ids),
                "source_prefix_token_hash": token_ids_hash(source_ids[:2]),
                "source_start": 2,
                "length": 3,
                "content_hash": token_ids_hash(island),
                "policy_label": "coding_natural_code_cost",
            }
        ],
        "cases": [{"source_id": "src0", "target_start": 3, "target_uses": 4}],
    }
    template = OnlineFileTemplate()
    replay_finetune(template, [group])
    obs = SourceObservation(
        source_id="src0",
        source_start=2,
        token_ids=(1, 2, 3),
        content_hash="x",
        source_prefix_hash="p",
        single_file_repository_code=True,
        version_valid=True,
        later_roles_in_protocol=3,
        policy_label="coding",
    )
    assert template.bin_for(obs).copied == 1


_PLAN = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_swebench_7b_file_modules_prefixkey_20260824/PLAN.json"
)
_MANIFEST = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_natural_code_cost_agent_expanded24_20260808/online/"
    "coding_natural_code_cost/full_24/DYNAMIC_MANIFEST.json"
)


@pytest.mark.skipif(not _PLAN.is_file() or not _MANIFEST.is_file(), reason="frozen artifacts missing")
def test_damped_replay_on_frozen_coding_plan_stays_a_fine_tune():
    import json

    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    template = compile_template(manifest["sources"], manifest["cases"])
    start = template._bins["coding_agent"].mean
    groups = json.loads(_PLAN.read_text(encoding="utf-8"))["groups"]
    replay_finetune(template, groups)
    end = template._bins["coding_agent"].mean
    obs = SourceObservation(
        source_id="probe",
        source_start=1,
        token_ids=(1, 2, 3),
        content_hash="h",
        source_prefix_hash="p",
        single_file_repository_code=True,
        version_valid=True,
        later_roles_in_protocol=3,
        policy_label="coding",
    )
    assert 0.52 < start < 0.59
    assert start < end < 0.74
    assert template.admit(obs) is None
