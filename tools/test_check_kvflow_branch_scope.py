from check_kvflow_branch_scope import scope_violations


def test_shared_rejects_policy_and_results():
    assert scope_violations(
        "shared",
        [
            "python/sglang/srt/mem_cache/coding_aware/policy.py",
            "results/run.json",
        ],
    ) == [
        "python/sglang/srt/mem_cache/coding_aware/policy.py",
        "results/run.json",
    ]


def test_coding_accepts_runtime_adapter_and_rejects_prefetch_policy():
    assert not scope_violations(
        "coding",
        [
            "python/sglang/srt/managers/scheduler.py",
            "python/sglang/srt/managers/schedule_policy.py",
        ],
    )
    assert scope_violations(
        "coding",
        ["python/sglang/srt/mem_cache/kvcomm_prefetch/coordinator.py"],
    )


def test_prefetch_accepts_coordinator_and_rejects_ast():
    assert not scope_violations(
        "prefetch",
        ["python/sglang/srt/mem_cache/kvcomm_prefetch/coordinator.py"],
    )
    assert scope_violations(
        "prefetch", ["python/sglang/srt/mem_cache/ast_chunker.py"]
    )
