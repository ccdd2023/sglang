from pathlib import Path

from benchmark.multi_workflow.template_prefetch_modes import mode_env, mode_manifest


def test_mode_env_keeps_coding_prefetch_off():
    assert mode_env("coding")["SGLANG_KV_PREFETCH"] == "0"
    assert mode_env("combined")["SGLANG_KV_PREFETCH"] == "1"
    assert mode_env("dense")["SGLANG_KVCOMM_CORE"] == "0"


def test_combined_manifest_spills_and_prefetch_only_disables_copy():
    group = {
        "sources": [{"source_id": "s0"}],
        "cases": [{"case_id": "c0", "reuse_enabled": True}],
    }
    combined = mode_manifest(Path("/tmp/art"), group, "m", "combined")
    assert combined["prefetch_spill_device"] is True
    assert combined["cases"][0]["reuse_enabled"] is True
    prefetch_only = mode_manifest(Path("/tmp/art"), group, "m", "prefetch_only")
    assert prefetch_only["cases"][0]["reuse_enabled"] is False
    coding = mode_manifest(Path("/tmp/art"), group, "m", "coding")
    assert coding["prefetch_spill_device"] is False
