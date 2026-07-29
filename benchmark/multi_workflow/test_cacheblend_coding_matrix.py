import json

import pytest

from benchmark.multi_workflow.cacheblend_coding_matrix import (
    prepare_case,
    prepare_workload,
    summarize,
)


def _source_row(case_id="case-0"):
    return {
        "_id": case_id,
        "answers": ["return value"],
        "context": ["def f():\n", "    value = 1\n"],
        "input": "    ",
        "language": "python",
        "_qcfuse_coding": {"source_index": 7},
    }


def _record(case_id, mode, text, ttft, build=0.0):
    return {
        "case_id": case_id,
        "mode": mode,
        "ttft_ms": ttft,
        "cache_build_ms": build,
        "reused_k_tokens": 10 if mode == "reuse" else 0,
        "reused_v_tokens": 10 if mode == "reuse" else 0,
        "recomputed_tokens": 4 if mode == "reuse" else 0,
        "error": None,
        "metadata": {"output_text": text, "warmup": False},
    }


def test_prepare_case_marks_only_context_reusable():
    case = prepare_case(_source_row(), "lcc", 0)

    assert case["case_id"] == "case-0"
    assert [segment["reusable"] for segment in case["segments"]] == [
        True,
        True,
        False,
    ]
    assert "".join(segment["text"] for segment in case["segments"]) == (
        "def f():\n    value = 1\n    "
    )
    assert case["metadata"]["answers"] == ["return value"]


def test_prepare_workload_rejects_duplicate_ids(tmp_path):
    source = tmp_path / "lcc.jsonl"
    source.write_text(
        "\n".join(json.dumps(_source_row()) for _ in range(2)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate case IDs"):
        prepare_workload(source, "lcc", 0)


def test_prepare_workload_uses_frozen_case_order(tmp_path):
    source = tmp_path / "repobench-p.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(_source_row(case_id))
            for case_id in ("a", "b", "c")
        )
        + "\n",
        encoding="utf-8",
    )

    workload = prepare_workload(
        source,
        "repobench-p",
        0,
        case_ids=["c", "a"],
    )

    assert [case["case_id"] for case in workload["cases"]] == ["c", "a"]


def test_summary_is_paired_to_native_dense():
    workload = {
        "model": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "dataset": "lcc",
        "protocol": {"claim_scope": "native paired"},
        "cases": [
            {
                "case_id": "a",
                "metadata": {"answers": ["return value"]},
            },
            {
                "case_id": "b",
                "metadata": {"answers": ["x = 1"]},
            },
        ],
    }
    dense = [
        _record("a", "dense", "return value\nextra", 20),
        _record("b", "dense", "wrong", 40),
    ]
    reuse = [
        _record("a", "reuse", "return value", 10, 40),
        _record("b", "reuse", "x = 1", 20, 40),
    ]

    result = summarize(workload, dense, reuse, 0.5)

    assert result["quality"]["dense_exact_line"] == 1
    assert result["quality"]["reuse_exact_line"] == 2
    assert result["quality"]["exact_line_delta_pp"] == 50
    assert result["latency"]["cache_ready_speedup_vs_native_dense"] == 2
    assert (
        result["latency"]["build_amortized"]["4"]["speedup_vs_native_dense"]
        == 1.2
    )
    assert result["physical_reuse"]["mean_reused_k_tokens"] == 10


def test_summary_rejects_unpaired_records():
    workload = {
        "model": "m",
        "dataset": "lcc",
        "protocol": {"claim_scope": "native paired"},
        "cases": [{"case_id": "a", "metadata": {"answers": ["x"]}}],
    }

    with pytest.raises(ValueError, match="do not cover exactly"):
        summarize(workload, [_record("a", "dense", "x", 1)], [], 0.5)
