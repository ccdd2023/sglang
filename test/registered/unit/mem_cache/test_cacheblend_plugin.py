from __future__ import annotations

import unittest

from sglang.srt.mem_cache.approx_kv.plugins import RecoveryPluginRegistry
from sglang.srt.mem_cache.approx_kv.types import RecoveryMode
from sglang.srt.mem_cache.cacheblend.hkvd import GradualFilterStage
from sglang.srt.mem_cache.cacheblend.plugin import (
    CACHEBLEND_PLUGIN_NAME,
    CACHEBLEND_RATIOS,
    CacheBlendConfig,
    CacheBlendRecoveryPlugin,
    maybe_register_cacheblend_plugin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def _make_config(ratio: float = 0.05, first_recompute_layer: int = 2) -> CacheBlendConfig:
    return CacheBlendConfig(
        ratio=ratio,
        probe_stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=0.5),),
        first_recompute_layer=first_recompute_layer,
    )


class FakeManager:
    def __init__(self) -> None:
        self.registry = RecoveryPluginRegistry()

    def register_plugin(self, plugin) -> None:
        self.registry.register(plugin)


class TestCacheBlendConfig(unittest.TestCase):
    def test_all_four_sweep_ratios_are_valid(self):
        self.assertEqual(CACHEBLEND_RATIOS, (0.01, 0.05, 0.15, 0.30))
        for ratio in CACHEBLEND_RATIOS:
            _make_config(ratio=ratio)  # must not raise

    def test_invalid_ratio_rejected(self):
        with self.assertRaises(ValueError):
            _make_config(ratio=0.10)

    def test_empty_probe_stages_rejected(self):
        with self.assertRaises(ValueError):
            CacheBlendConfig(ratio=0.05, probe_stages=(), first_recompute_layer=1)

    def test_negative_first_recompute_layer_rejected(self):
        with self.assertRaises(ValueError):
            _make_config(first_recompute_layer=-1)

    def test_probe_layer_must_be_shallower_than_first_recompute_layer(self):
        with self.assertRaises(ValueError):
            CacheBlendConfig(
                ratio=0.05,
                probe_stages=(GradualFilterStage(probe_layer_id=2, keep_ratio=0.5),),
                first_recompute_layer=2,
            )

    def test_from_env_defaults(self):
        config = CacheBlendConfig.from_env(env={})
        self.assertAlmostEqual(config.ratio, 0.05)
        self.assertEqual(len(config.probe_stages), 1)
        self.assertEqual(config.probe_stages[0].probe_layer_id, 0)
        self.assertEqual(config.first_recompute_layer, 1)

    def test_from_env_custom_values(self):
        config = CacheBlendConfig.from_env(
            env={
                "SGLANG_CACHEBLEND_RATIO": "0.15",
                "SGLANG_CACHEBLEND_PROBE_LAYERS": "0,2",
                "SGLANG_CACHEBLEND_FIRST_RECOMPUTE_LAYER": "4",
            }
        )
        self.assertAlmostEqual(config.ratio, 0.15)
        self.assertEqual(
            [s.probe_layer_id for s in config.probe_stages], [0, 2]
        )
        self.assertEqual(config.first_recompute_layer, 4)


class TestCacheBlendRecoveryPlugin(unittest.TestCase):
    def test_name_is_cacheblend(self):
        plugin = CacheBlendRecoveryPlugin(config=_make_config())
        self.assertEqual(plugin.name, CACHEBLEND_PLUGIN_NAME)

    def test_not_capable_without_backends(self):
        plugin = CacheBlendRecoveryPlugin(config=_make_config())
        self.assertFalse(plugin.capable)

    def test_not_capable_with_only_one_backend(self):
        plugin = CacheBlendRecoveryPlugin(
            config=_make_config(), probe_backend=object()
        )
        self.assertFalse(plugin.capable)

    def test_capable_with_both_backends(self):
        plugin = CacheBlendRecoveryPlugin(
            config=_make_config(), probe_backend=object(), recompute_backend=object()
        )
        self.assertTrue(plugin.capable)

    def test_build_plan_is_conservative_dense_only(self):
        from sglang.srt.mem_cache.approx_kv.plugins import RecoveryRequestContext

        plugin = CacheBlendRecoveryPlugin(config=_make_config())
        context = RecoveryRequestContext(
            request_id="r1",
            target_token_ids=tuple(range(10)),
            exact_prefix_length=4,
            custom_metadata={},
        )
        plan = plugin.build_plan(context, store=None)
        self.assertEqual(plan.recovery_mode, RecoveryMode.DENSE)
        self.assertEqual(len(plan.dense_ranges), 1)
        self.assertEqual(plan.dense_ranges[0].length, 6)
        self.assertEqual(
            plan.dense_ranges[0].reason, "cacheblend_requires_online_hkvd_execution"
        )

    def test_build_plan_rejects_fully_exact_prefix(self):
        from sglang.srt.mem_cache.approx_kv.plugins import RecoveryRequestContext

        plugin = CacheBlendRecoveryPlugin(config=_make_config())
        context = RecoveryRequestContext(
            request_id="r1",
            target_token_ids=tuple(range(4)),
            exact_prefix_length=4,
            custom_metadata={},
        )
        with self.assertRaises(ValueError):
            plugin.build_plan(context, store=None)

    def test_scheduler_metadata_is_empty(self):
        plugin = CacheBlendRecoveryPlugin(config=_make_config())
        from sglang.srt.mem_cache.approx_kv.plugins import RecoveryRequestContext

        context = RecoveryRequestContext(
            request_id="r1",
            target_token_ids=(1, 2, 3),
            exact_prefix_length=0,
            custom_metadata={},
        )
        self.assertEqual(plugin.scheduler_metadata(context), ())


class TestMaybeRegisterCacheBlendPlugin(unittest.TestCase):
    def test_disabled_by_default(self):
        manager = FakeManager()
        result = maybe_register_cacheblend_plugin(manager, env={})
        self.assertIsNone(result)
        self.assertEqual(manager.registry.names(), ())

    def test_enabled_registers_plugin(self):
        manager = FakeManager()
        result = maybe_register_cacheblend_plugin(
            manager, env={"SGLANG_APPROX_KV_CACHEBLEND": "1"}
        )
        self.assertIsNotNone(result)
        self.assertEqual(manager.registry.names(), (CACHEBLEND_PLUGIN_NAME,))
        self.assertIs(manager.registry.get(CACHEBLEND_PLUGIN_NAME), result)

    def test_enabled_without_backends_is_not_capable(self):
        manager = FakeManager()
        result = maybe_register_cacheblend_plugin(
            manager, env={"SGLANG_APPROX_KV_CACHEBLEND": "1"}
        )
        self.assertFalse(result.capable)

    def test_enabled_with_backends_is_capable(self):
        manager = FakeManager()
        result = maybe_register_cacheblend_plugin(
            manager,
            env={"SGLANG_APPROX_KV_CACHEBLEND": "1"},
            probe_backend=object(),
            recompute_backend=object(),
        )
        self.assertTrue(result.capable)

    def test_invalid_bool_env_rejected(self):
        manager = FakeManager()
        with self.assertRaises(ValueError):
            maybe_register_cacheblend_plugin(
                manager, env={"SGLANG_APPROX_KV_CACHEBLEND": "maybe"}
            )


if __name__ == "__main__":
    unittest.main()
