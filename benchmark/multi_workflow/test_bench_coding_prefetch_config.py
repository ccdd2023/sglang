from argparse import Namespace
from pathlib import Path

from benchmark.multi_workflow.bench_coding_kvflow_prefetch import build_server_command


def _base_args(**overrides):
    args = Namespace(
        python="/tmp/python",
        model="/tmp/model",
        port=30000,
        mem_fraction_static=0.78,
        max_total_tokens=65536,
        disable_hierarchical_cache=False,
        hicache_ratio=1.5,
        hicache_storage_backend="",
        baseline_profile="agenttemplatekv",
        server_extra_args="",
        lmcache_config=None,
        out_dir=Path("/tmp/out"),
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_lmcache_extra_arg_suppresses_hicache_flags():
    cmd, env, manifest = build_server_command(
        _base_args(
            server_extra_args="--enable-lmcache",
            lmcache_config=Path("/tmp/lmcache.yaml"),
        )
    )

    assert "--enable-lmcache" in cmd
    assert "--enable-hierarchical-cache" not in cmd
    assert "--enable-hicache-prefetch" not in cmd
    assert manifest["lmcache_requested"] is True
    assert manifest["hierarchical_cache"] is False
    assert manifest["hierarchical_cache_suppressed_for_lmcache"] is True
    assert env["LMCACHE_USE_EXPERIMENTAL"] == "True"
    assert env["LMCACHE_CONFIG_FILE"] == "/tmp/lmcache.yaml"


def test_default_prefetch_runner_keeps_hicache_flags():
    cmd, _, manifest = build_server_command(_base_args())

    assert "--enable-lmcache" not in cmd
    assert "--enable-hierarchical-cache" in cmd
    assert "--enable-hicache-prefetch" in cmd
    assert manifest["hierarchical_cache"] is True
    assert manifest["hierarchical_cache_suppressed_for_lmcache"] is False


def test_lmcache_profile_adds_lmcache_and_default_config():
    cmd, env, manifest = build_server_command(_base_args(baseline_profile="lmcache"))

    assert "--enable-lmcache" in cmd
    assert "--enable-hierarchical-cache" not in cmd
    assert manifest["baseline_profile"] == "lmcache"
    assert manifest["lmcache_requested"] is True
    assert manifest["lmcache_config"].endswith("storage/lmcache/example_config.yaml")
    assert env["LMCACHE_USE_EXPERIMENTAL"] == "True"


def test_lmcache_profile_rejects_explicit_hicache_flags():
    import pytest

    with pytest.raises(ValueError):
        build_server_command(
            _base_args(
                baseline_profile="lmcache",
                server_extra_args="--enable-hierarchical-cache",
            )
        )
