from __future__ import annotations

from benchmark.multi_workflow.compile_class_template import compile_template


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
