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


def test_coding_rejects_scheduler_and_prefetch():
    assert len(
        scope_violations(
            "coding",
            [
                "python/sglang/srt/managers/scheduler.py",
                "python/sglang/srt/mem_cache/kvcomm_prefetch/coordinator.py",
            ],
        )
    ) == 2


def test_prefetch_accepts_coordinator_and_rejects_ast():
    assert not scope_violations(
        "prefetch",
        ["python/sglang/srt/mem_cache/kvcomm_prefetch/coordinator.py"],
    )
    assert scope_violations(
        "prefetch", ["python/sglang/srt/mem_cache/ast_chunker.py"]
    )
