"""Unit tests for the shared "default RoPE" resolver
(`approx_kv.rope_resolver.resolve_model_rope_config`).

This is the common-core resolver ported verbatim from the R1
EPIC/LegoLink fork's `epic_runtime.py` and the R0 raw+RoPE fork's
`raw_rope.py`, now used by `registry.py::create_tree_cache` to bind a
real production `RoPEConfig` onto the CacheBlend recovery plugin's
`ApproxKVManager`. Only the default (unscaled) Qwen2/Qwen3 RoPE layout
is supported; everything else must resolve to `None` so callers treat
it exactly like "unavailable" and fall back to dense instead of
guessing at a scaled rotation.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.approx_kv.rope_resolver import resolve_model_rope_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def _model_config(**hf_config_fields):
    return SimpleNamespace(hf_config=SimpleNamespace(**hf_config_fields))


class TestResolveModelRopeConfig(unittest.TestCase):
    def test_qwen3_default_layout_resolves(self):
        config = resolve_model_rope_config(
            _model_config(
                model_type="qwen3",
                head_dim=128,
                rope_theta=1000000.0,
                rope_scaling=None,
            )
        )
        self.assertIsNotNone(config)
        self.assertEqual(config.rotary_dim, 128)
        self.assertEqual(config.base, 1000000.0)
        self.assertTrue(config.is_neox_style)

    def test_qwen2_with_explicit_default_rope_scaling_resolves(self):
        config = resolve_model_rope_config(
            _model_config(
                model_type="qwen2",
                head_dim=64,
                rope_theta=None,
                rope_scaling={"rope_type": "default", "rope_theta": 500000.0},
            )
        )
        self.assertIsNotNone(config)
        self.assertEqual(config.rotary_dim, 64)
        self.assertEqual(config.base, 500000.0)

    def test_model_type_is_case_insensitive(self):
        config = resolve_model_rope_config(
            _model_config(
                model_type="Qwen2",
                head_dim=64,
                rope_theta=10000.0,
                rope_scaling=None,
            )
        )
        self.assertIsNotNone(config)

    def test_scaled_rope_scaling_stays_unbound(self):
        for rope_type in ("linear", "yarn", "dynamic"):
            with self.subTest(rope_type=rope_type):
                config = resolve_model_rope_config(
                    _model_config(
                        model_type="qwen3",
                        head_dim=128,
                        rope_theta=None,
                        rope_scaling={"rope_type": rope_type},
                    )
                )
                self.assertIsNone(config)

    def test_non_qwen_model_family_stays_unbound(self):
        config = resolve_model_rope_config(
            _model_config(
                model_type="llama",
                head_dim=128,
                rope_theta=10000.0,
                rope_scaling=None,
            )
        )
        self.assertIsNone(config)

    def test_missing_model_type_stays_unbound(self):
        config = resolve_model_rope_config(
            _model_config(
                head_dim=128,
                rope_theta=10000.0,
                rope_scaling=None,
            )
        )
        self.assertIsNone(config)

    def test_head_dim_derived_from_hidden_size_and_num_heads(self):
        config = resolve_model_rope_config(
            _model_config(
                model_type="qwen2",
                hidden_size=4096,
                num_attention_heads=32,
                rope_theta=10000.0,
                rope_scaling=None,
            )
        )
        self.assertIsNotNone(config)
        # 4096 / 32 == 128.
        self.assertEqual(config.rotary_dim, 128)

    def test_head_dim_derivation_fails_when_not_evenly_divisible(self):
        config = resolve_model_rope_config(
            _model_config(
                model_type="qwen2",
                hidden_size=4097,
                num_attention_heads=32,
                rope_theta=10000.0,
                rope_scaling=None,
            )
        )
        self.assertIsNone(config)

    def test_partial_rotary_factor_scales_rotary_dim(self):
        config = resolve_model_rope_config(
            _model_config(
                model_type="qwen3",
                head_dim=128,
                partial_rotary_factor=0.5,
                rope_theta=10000.0,
                rope_scaling=None,
            )
        )
        self.assertIsNotNone(config)
        self.assertEqual(config.rotary_dim, 64)

    def test_zero_rotary_dim_stays_unbound(self):
        config = resolve_model_rope_config(
            _model_config(
                model_type="qwen3",
                head_dim=128,
                partial_rotary_factor=0.0,
                rope_theta=10000.0,
                rope_scaling=None,
            )
        )
        self.assertIsNone(config)

    def test_odd_rotary_dim_stays_unbound(self):
        config = resolve_model_rope_config(
            _model_config(
                model_type="qwen3",
                head_dim=127,
                rope_theta=10000.0,
                rope_scaling=None,
            )
        )
        self.assertIsNone(config)

    def test_default_base_when_rope_theta_and_scaling_absent(self):
        config = resolve_model_rope_config(
            _model_config(
                model_type="qwen3",
                head_dim=128,
                rope_theta=None,
                rope_scaling=None,
            )
        )
        self.assertIsNotNone(config)
        self.assertEqual(config.base, 10000.0)


if __name__ == "__main__":
    unittest.main()
