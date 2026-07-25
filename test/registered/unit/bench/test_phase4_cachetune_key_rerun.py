from __future__ import annotations

import unittest

from benchmark.approx_kv.run_phase4_cachetune_key_rerun import parse_body_tokens
from sglang.srt.mem_cache.cachetune.hardware_profile import (
    CacheTuneMode,
    HardwareMeasurement,
    RatioBounds,
    quantize_ratio,
    roofline_ratio,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")


class TestCacheTuneKeyRerunHelpers(unittest.TestCase):
    def test_parse_body_tokens(self):
        self.assertEqual(parse_body_tokens("1024,2048"), (1024, 2048))

    def test_parse_body_tokens_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            parse_body_tokens("")

    def test_committed_sm75_calibration_selects_expected_tokens(self):
        measurement = HardwareMeasurement(
            t_c_ms=0.025747446,
            t_i_ms=0.002326677,
            t_o_ms=1.825835613,
        )
        ratio = roofline_ratio(measurement)
        bounds = RatioBounds.for_mode(CacheTuneMode.SPEED_ONLY)
        self.assertEqual(
            quantize_ratio(ratio, context_length=1024, bounds=bounds).repair_tokens,
            85,
        )
        self.assertEqual(
            quantize_ratio(ratio, context_length=2048, bounds=bounds).repair_tokens,
            170,
        )


if __name__ == "__main__":
    unittest.main()
