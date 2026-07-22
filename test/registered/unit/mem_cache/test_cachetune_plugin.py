from __future__ import annotations

import unittest

from sglang.srt.mem_cache.approx_kv.plugins import RecoveryPluginRegistry
from sglang.srt.mem_cache.approx_kv.types import RecoveryMode
from sglang.srt.mem_cache.cachetune.controller import CacheTuneController
from sglang.srt.mem_cache.cachetune.hardware_profile import (
    CacheTuneMode,
    HardwareMeasurement,
)
from sglang.srt.mem_cache.cachetune.plugin import (
    CACHETUNE_PLUGIN_NAME,
    CacheTuneConfig,
    CacheTuneRecoveryPlugin,
    maybe_register_cachetune_plugin,
)
from sglang.srt.mem_cache.cachetune.token_selection import GradualFilterStage
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def _make_config(
    *,
    mode: CacheTuneMode = CacheTuneMode.SPEED_ONLY,
    hardware_tier: str = "cpu",
    first_recompute_layer: int = 2,
    deployment_measurement=None,
) -> CacheTuneConfig:
    return CacheTuneConfig(
        mode=mode,
        hardware_tier=hardware_tier,
        probe_stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=0.5),),
        first_recompute_layer=first_recompute_layer,
        deployment_measurement=deployment_measurement,
    )


class FakeManager:
    def __init__(self) -> None:
        self.registry = RecoveryPluginRegistry()

    def register_plugin(self, plugin) -> None:
        self.registry.register(plugin)


class TestCacheTuneConfigValidation(unittest.TestCase):
    def test_valid_config_constructs_for_both_modes(self):
        _make_config(mode=CacheTuneMode.PAPER_MECHANISM)
        _make_config(mode=CacheTuneMode.SPEED_ONLY)

    def test_non_enum_mode_rejected(self):
        with self.assertRaises(TypeError):
            CacheTuneConfig(
                mode="speed_only",  # type: ignore[arg-type]
                hardware_tier="cpu",
                probe_stages=(GradualFilterStage(probe_layer_id=0, keep_ratio=0.5),),
                first_recompute_layer=1,
                deployment_measurement=None,
            )

    def test_empty_hardware_tier_rejected(self):
        with self.assertRaises(ValueError):
            _make_config(hardware_tier="  ")

    def test_empty_probe_stages_rejected(self):
        with self.assertRaises(ValueError):
            CacheTuneConfig(
                mode=CacheTuneMode.SPEED_ONLY,
                hardware_tier="cpu",
                probe_stages=(),
                first_recompute_layer=1,
                deployment_measurement=None,
            )

    def test_negative_first_recompute_layer_rejected(self):
        with self.assertRaises(ValueError):
            _make_config(first_recompute_layer=-1)

    def test_probe_layer_must_be_shallower_than_first_recompute_layer(self):
        with self.assertRaises(ValueError):
            CacheTuneConfig(
                mode=CacheTuneMode.SPEED_ONLY,
                hardware_tier="cpu",
                probe_stages=(GradualFilterStage(probe_layer_id=2, keep_ratio=0.5),),
                first_recompute_layer=2,
                deployment_measurement=None,
            )

    def test_deployment_measurement_defaults_to_none(self):
        config = _make_config()
        self.assertIsNone(config.deployment_measurement)

    def test_deployment_measurement_can_be_supplied(self):
        measurement = HardwareMeasurement(t_c_ms=0.1, t_i_ms=0.1, t_o_ms=0.0)
        config = _make_config(deployment_measurement=measurement)
        self.assertIs(config.deployment_measurement, measurement)


class TestCacheTuneConfigFromEnv(unittest.TestCase):
    def test_missing_mode_env_var_rejected(self):
        with self.assertRaises(ValueError):
            CacheTuneConfig.from_env(env={})

    def test_invalid_mode_value_rejected(self):
        with self.assertRaises(ValueError):
            CacheTuneConfig.from_env(env={"SGLANG_CACHETUNE_MODE": "bogus"})

    def test_speed_only_mode_from_env_defaults(self):
        config = CacheTuneConfig.from_env(env={"SGLANG_CACHETUNE_MODE": "speed_only"})
        self.assertEqual(config.mode, CacheTuneMode.SPEED_ONLY)
        self.assertEqual(config.hardware_tier, "cpu")  # no CUDA in this env
        self.assertEqual(len(config.probe_stages), 1)
        self.assertEqual(config.probe_stages[0].probe_layer_id, 0)
        self.assertAlmostEqual(config.probe_stages[0].keep_ratio, 0.5)
        self.assertEqual(config.first_recompute_layer, 1)
        self.assertIsNone(config.deployment_measurement)

    def test_paper_mechanism_mode_from_env(self):
        config = CacheTuneConfig.from_env(
            env={"SGLANG_CACHETUNE_MODE": "paper_mechanism"}
        )
        self.assertEqual(config.mode, CacheTuneMode.PAPER_MECHANISM)

    def test_custom_hardware_tier_override(self):
        config = CacheTuneConfig.from_env(
            env={
                "SGLANG_CACHETUNE_MODE": "speed_only",
                "SGLANG_CACHETUNE_HARDWARE_TIER": "NVIDIA RTX 2080 SUPER",
            }
        )
        self.assertEqual(config.hardware_tier, "NVIDIA RTX 2080 SUPER")

    def test_custom_probe_layers_and_keep_ratio(self):
        config = CacheTuneConfig.from_env(
            env={
                "SGLANG_CACHETUNE_MODE": "speed_only",
                "SGLANG_CACHETUNE_PROBE_LAYERS": "0,2",
                "SGLANG_CACHETUNE_STAGE_KEEP_RATIO": "0.3",
                "SGLANG_CACHETUNE_FIRST_RECOMPUTE_LAYER": "4",
            }
        )
        self.assertEqual([s.probe_layer_id for s in config.probe_stages], [0, 2])
        for stage in config.probe_stages:
            self.assertAlmostEqual(stage.keep_ratio, 0.3)
        self.assertEqual(config.first_recompute_layer, 4)

    def test_all_three_measurement_env_vars_together_are_parsed(self):
        config = CacheTuneConfig.from_env(
            env={
                "SGLANG_CACHETUNE_MODE": "speed_only",
                "SGLANG_CACHETUNE_T_C_MS": "0.12",
                "SGLANG_CACHETUNE_T_I_MS": "0.05",
                "SGLANG_CACHETUNE_T_O_MS": "1.5",
            }
        )
        self.assertIsNotNone(config.deployment_measurement)
        self.assertAlmostEqual(config.deployment_measurement.t_c_ms, 0.12)
        self.assertAlmostEqual(config.deployment_measurement.t_i_ms, 0.05)
        self.assertAlmostEqual(config.deployment_measurement.t_o_ms, 1.5)

    def test_partial_measurement_env_vars_rejected(self):
        with self.assertRaises(ValueError):
            CacheTuneConfig.from_env(
                env={
                    "SGLANG_CACHETUNE_MODE": "speed_only",
                    "SGLANG_CACHETUNE_T_C_MS": "0.12",
                    "SGLANG_CACHETUNE_T_I_MS": "0.05",
                    # T_O_MS deliberately omitted
                }
            )

    def test_non_numeric_measurement_env_var_rejected(self):
        with self.assertRaises(ValueError):
            CacheTuneConfig.from_env(
                env={
                    "SGLANG_CACHETUNE_MODE": "speed_only",
                    "SGLANG_CACHETUNE_T_C_MS": "not-a-number",
                    "SGLANG_CACHETUNE_T_I_MS": "0.05",
                    "SGLANG_CACHETUNE_T_O_MS": "1.5",
                }
            )

    def test_blank_measurement_env_vars_treated_as_absent(self):
        config = CacheTuneConfig.from_env(
            env={
                "SGLANG_CACHETUNE_MODE": "speed_only",
                "SGLANG_CACHETUNE_T_C_MS": "  ",
                "SGLANG_CACHETUNE_T_I_MS": "",
                "SGLANG_CACHETUNE_T_O_MS": "",
            }
        )
        self.assertIsNone(config.deployment_measurement)


class TestCacheTuneRecoveryPlugin(unittest.TestCase):
    def _controller(self) -> CacheTuneController:
        return CacheTuneController(CacheTuneMode.SPEED_ONLY)

    def test_name_is_cachetune(self):
        plugin = CacheTuneRecoveryPlugin(
            config=_make_config(), controller=self._controller()
        )
        self.assertEqual(plugin.name, CACHETUNE_PLUGIN_NAME)

    def test_not_capable_without_backends(self):
        plugin = CacheTuneRecoveryPlugin(
            config=_make_config(), controller=self._controller()
        )
        self.assertFalse(plugin.capable)

    def test_not_capable_with_only_probe_backend(self):
        plugin = CacheTuneRecoveryPlugin(
            config=_make_config(),
            controller=self._controller(),
            probe_backend=object(),
        )
        self.assertFalse(plugin.capable)

    def test_not_capable_with_only_recompute_backend(self):
        plugin = CacheTuneRecoveryPlugin(
            config=_make_config(),
            controller=self._controller(),
            recompute_backend=object(),
        )
        self.assertFalse(plugin.capable)

    def test_capable_with_both_backends(self):
        plugin = CacheTuneRecoveryPlugin(
            config=_make_config(),
            controller=self._controller(),
            probe_backend=object(),
            recompute_backend=object(),
        )
        self.assertTrue(plugin.capable)

    def test_build_plan_is_conservative_dense_only(self):
        from sglang.srt.mem_cache.approx_kv.plugins import RecoveryRequestContext

        plugin = CacheTuneRecoveryPlugin(
            config=_make_config(), controller=self._controller()
        )
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
            plan.dense_ranges[0].reason,
            "cachetune_requires_online_repair_execution",
        )

    def test_build_plan_rejects_fully_exact_prefix(self):
        from sglang.srt.mem_cache.approx_kv.plugins import RecoveryRequestContext

        plugin = CacheTuneRecoveryPlugin(
            config=_make_config(), controller=self._controller()
        )
        context = RecoveryRequestContext(
            request_id="r1",
            target_token_ids=tuple(range(4)),
            exact_prefix_length=4,
            custom_metadata={},
        )
        with self.assertRaises(ValueError):
            plugin.build_plan(context, store=None)

    def test_scheduler_metadata_is_empty(self):
        from sglang.srt.mem_cache.approx_kv.plugins import RecoveryRequestContext

        plugin = CacheTuneRecoveryPlugin(
            config=_make_config(), controller=self._controller()
        )
        context = RecoveryRequestContext(
            request_id="r1",
            target_token_ids=(1, 2, 3),
            exact_prefix_length=0,
            custom_metadata={},
        )
        self.assertEqual(plugin.scheduler_metadata(context), ())

    def test_controller_and_config_are_stored_verbatim(self):
        config = _make_config()
        controller = self._controller()
        plugin = CacheTuneRecoveryPlugin(config=config, controller=controller)
        self.assertIs(plugin.config, config)
        self.assertIs(plugin.controller, controller)


class TestMaybeRegisterCacheTunePlugin(unittest.TestCase):
    def test_disabled_by_default(self):
        manager = FakeManager()
        result = maybe_register_cachetune_plugin(
            manager, env={"SGLANG_CACHETUNE_MODE": "speed_only"}
        )
        self.assertIsNone(result)
        self.assertEqual(manager.registry.names(), ())

    def test_enabled_registers_plugin(self):
        manager = FakeManager()
        result = maybe_register_cachetune_plugin(
            manager,
            env={
                "SGLANG_APPROX_KV_CACHETUNE": "1",
                "SGLANG_CACHETUNE_MODE": "speed_only",
            },
        )
        self.assertIsNotNone(result)
        self.assertEqual(manager.registry.names(), (CACHETUNE_PLUGIN_NAME,))
        self.assertIs(manager.registry.get(CACHETUNE_PLUGIN_NAME), result)

    def test_enabled_without_mode_env_var_raises(self):
        manager = FakeManager()
        with self.assertRaises(ValueError):
            maybe_register_cachetune_plugin(
                manager, env={"SGLANG_APPROX_KV_CACHETUNE": "1"}
            )

    def test_enabled_without_backends_is_not_capable(self):
        manager = FakeManager()
        result = maybe_register_cachetune_plugin(
            manager,
            env={
                "SGLANG_APPROX_KV_CACHETUNE": "1",
                "SGLANG_CACHETUNE_MODE": "speed_only",
            },
        )
        self.assertFalse(result.capable)

    def test_enabled_with_backends_is_capable(self):
        manager = FakeManager()
        result = maybe_register_cachetune_plugin(
            manager,
            env={
                "SGLANG_APPROX_KV_CACHETUNE": "1",
                "SGLANG_CACHETUNE_MODE": "speed_only",
            },
            probe_backend=object(),
            recompute_backend=object(),
        )
        self.assertTrue(result.capable)

    def test_registered_plugin_has_a_real_controller_instance(self):
        manager = FakeManager()
        result = maybe_register_cachetune_plugin(
            manager,
            env={
                "SGLANG_APPROX_KV_CACHETUNE": "1",
                "SGLANG_CACHETUNE_MODE": "paper_mechanism",
            },
        )
        self.assertIsInstance(result.controller, CacheTuneController)
        self.assertEqual(result.controller.mode, CacheTuneMode.PAPER_MECHANISM)

    def test_invalid_bool_env_rejected(self):
        manager = FakeManager()
        with self.assertRaises(ValueError):
            maybe_register_cachetune_plugin(
                manager,
                env={
                    "SGLANG_APPROX_KV_CACHETUNE": "maybe",
                    "SGLANG_CACHETUNE_MODE": "speed_only",
                },
            )

    def test_various_truthy_and_falsy_bool_spellings(self):
        for value, expect_registered in (
            ("1", True),
            ("true", True),
            ("yes", True),
            ("on", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("off", False),
        ):
            with self.subTest(value=value):
                manager = FakeManager()
                result = maybe_register_cachetune_plugin(
                    manager,
                    env={
                        "SGLANG_APPROX_KV_CACHETUNE": value,
                        "SGLANG_CACHETUNE_MODE": "speed_only",
                    },
                )
                self.assertEqual(result is not None, expect_registered)


if __name__ == "__main__":
    unittest.main()
