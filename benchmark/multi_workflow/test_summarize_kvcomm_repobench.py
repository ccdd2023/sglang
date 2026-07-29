from benchmark.multi_workflow.summarize_kvcomm_repobench import (
    summarize_kvcomm,
)


def _record(case_id, mode, output, ttft, build=0):
    return {
        "case_id": case_id,
        "mode": mode,
        "ttft_ms": ttft,
        "cache_build_ms": build,
        "reused_k_tokens": 20 if mode == "reuse" else 0,
        "reused_v_tokens": 20 if mode == "reuse" else 0,
        "recomputed_tokens": 0,
        "error": None,
        "metadata": {
            "output_text": output,
            "warmup": False,
            "source_observation": False,
        },
    }


def test_summary_relabels_native_kvcomm():
    workload = {
        "model": "m",
        "dataset": "repobench-p",
        "protocol": {"claim_scope": "original"},
        "cases": [
            {"case_id": "a", "metadata": {"answers": ["return x"]}}
        ],
    }
    value = summarize_kvcomm(
        workload,
        [_record("a", "dense", "wrong", 20)],
        [_record("a", "reuse", "return x", 10, 4)],
        0.5,
    )

    assert value["method"] == "KVCOMM"
    assert value["quality"]["reuse_exact_line"] == 1
    assert value["config"]["threshold"] == 0.5
    assert "recompute_ratio" not in value
