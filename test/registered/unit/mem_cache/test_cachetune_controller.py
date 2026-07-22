from __future__ import annotations

import math
import unittest

from sglang.srt.mem_cache.cachetune.controller import (
    CacheTuneController,
    CacheTuneDecision,
    CacheTuneProfileError,
    CalibrationResult,
)
from sglang.srt.mem_cache.cachetune.hardware_profile import (
    CacheTuneMode,
    HardwareMeasurement,
    HardwareProfileKey,
    roofline_ratio,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def _key(
    bucket: int = 1024, tier: str = "cpu", model: str = "model"
) -> HardwareProfileKey:
    return HardwareProfileKey(
        hardware_tier=tier, model_fingerprint=model, chunk_length_bucket=bucket
    )


class TestCacheTuneControllerConstruction(unittest.TestCase):
    def test_rejects_non_enum_mode(self):
        with self.assertRaises(TypeError):
            CacheTuneController("speed_only")  # type: ignore[arg-type]

    def test_paper_mechanism_bounds_are_exposed(self):
        controller = CacheTuneController(CacheTuneMode.PAPER_MECHANISM)
        self.assertEqual(controller.mode, CacheTuneMode.PAPER_MECHANISM)
        self.assertAlmostEqual(controller.bounds.r_min, 0.15)

    def test_speed_only_bounds_are_exposed(self):
        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        self.assertAlmostEqual(controller.bounds.r_min, 0.0)


class TestRecordMeasurement(unittest.TestCase):
    def setUp(self):
        self.controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)

    def test_rejects_non_key_type(self):
        with self.assertRaises(TypeError):
            self.controller.record_measurement(
                "not-a-key",  # type: ignore[arg-type]
                HardwareMeasurement(t_c_ms=0.1, t_i_ms=0.1, t_o_ms=0.1),
            )

    def test_rejects_non_measurement_type(self):
        with self.assertRaises(TypeError):
            self.controller.record_measurement(_key(), "not-a-measurement")  # type: ignore[arg-type]

    def test_has_measurement_false_before_recording(self):
        self.assertFalse(self.controller.has_measurement(_key()))

    def test_has_measurement_true_after_recording(self):
        key = _key()
        measurement = HardwareMeasurement(t_c_ms=0.1, t_i_ms=0.1, t_o_ms=0.1)
        self.controller.record_measurement(key, measurement)
        self.assertTrue(self.controller.has_measurement(key))
        self.assertIs(self.controller.measurement(key), measurement)

    def test_re_recording_invalidates_prior_calibration(self):
        key = _key()
        self.controller.record_measurement(
            key, HardwareMeasurement(t_c_ms=0.1, t_i_ms=0.1, t_o_ms=0.0)
        )
        self.controller.calibrate(
            key, context_length=1000, evaluate=lambda ratio: 100.0
        )
        self.assertTrue(self.controller.has_calibration(key))
        self.controller.record_measurement(
            key, HardwareMeasurement(t_c_ms=0.2, t_i_ms=0.2, t_o_ms=0.0)
        )
        self.assertFalse(self.controller.has_calibration(key))
        self.assertIsNone(self.controller.calibration(key))


class TestMeasurementAndRooflineRequireRecording(unittest.TestCase):
    def setUp(self):
        self.controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)

    def test_measurement_raises_for_unrecorded_key(self):
        with self.assertRaises(CacheTuneProfileError):
            self.controller.measurement(_key())

    def test_roofline_raises_for_unrecorded_key(self):
        with self.assertRaises(CacheTuneProfileError):
            self.controller.roofline(_key())

    def test_select_ratio_raises_for_unrecorded_key(self):
        with self.assertRaises(CacheTuneProfileError):
            self.controller.select_ratio(_key(), context_length=100, num_layers=4)

    def test_calibrate_raises_for_unrecorded_key(self):
        with self.assertRaises(CacheTuneProfileError):
            self.controller.calibrate(
                _key(), context_length=100, evaluate=lambda ratio: 1.0
            )


class TestRoofline(unittest.TestCase):
    def test_roofline_matches_module_level_function(self):
        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        key = _key()
        measurement = HardwareMeasurement(t_c_ms=0.3, t_i_ms=0.1, t_o_ms=0.5)
        controller.record_measurement(key, measurement)
        self.assertAlmostEqual(controller.roofline(key), roofline_ratio(measurement))


class TestSelectRatioComputeBound(unittest.TestCase):
    def test_compute_bound_hardware_selects_small_ratio_speed_only(self):
        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        key = _key()
        # t_c >> t_i: recompute much more expensive than transfer.
        controller.record_measurement(
            key, HardwareMeasurement(t_c_ms=2.0, t_i_ms=0.02, t_o_ms=0.0)
        )
        decision = controller.select_ratio(key, context_length=1000, num_layers=8)
        self.assertEqual(decision.source, "roofline")
        self.assertLess(decision.executable_ratio, 0.05)
        self.assertEqual(decision.repair_tokens, decision.quantized.repair_tokens)

    def test_compute_bound_hardware_clamped_to_paper_floor(self):
        controller = CacheTuneController(CacheTuneMode.PAPER_MECHANISM)
        key = _key()
        controller.record_measurement(
            key, HardwareMeasurement(t_c_ms=2.0, t_i_ms=0.02, t_o_ms=0.0)
        )
        decision = controller.select_ratio(key, context_length=1000, num_layers=8)
        # Roofline r0 would be well under 15%, but paper-mechanism mode
        # must clamp the executable ratio up to the 15% quality floor.
        self.assertLess(decision.roofline_ratio, 0.15)
        self.assertGreaterEqual(decision.executable_ratio, 0.15 - 1e-9)


class TestSelectRatioIOBound(unittest.TestCase):
    def test_io_bound_hardware_selects_large_ratio(self):
        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        key = _key()
        # t_i >> t_c: transfer much more expensive than recompute.
        controller.record_measurement(
            key, HardwareMeasurement(t_c_ms=0.02, t_i_ms=2.0, t_o_ms=0.0)
        )
        decision = controller.select_ratio(key, context_length=1000, num_layers=8)
        self.assertGreater(decision.executable_ratio, 0.95)

    def test_io_bound_hardware_capped_at_r_max(self):
        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY, r_max=0.8)
        key = _key()
        controller.record_measurement(
            key, HardwareMeasurement(t_c_ms=0.02, t_i_ms=2.0, t_o_ms=0.0)
        )
        decision = controller.select_ratio(key, context_length=1000, num_layers=8)
        self.assertLessEqual(decision.executable_ratio, 0.8 + 1e-9)


class TestSelectRatioValidation(unittest.TestCase):
    def setUp(self):
        self.controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        self.key = _key()
        self.controller.record_measurement(
            self.key, HardwareMeasurement(t_c_ms=0.1, t_i_ms=0.1, t_o_ms=0.0)
        )

    def test_non_positive_context_length_rejected(self):
        with self.assertRaises(ValueError):
            self.controller.select_ratio(self.key, context_length=0, num_layers=4)

    def test_non_positive_num_layers_rejected(self):
        with self.assertRaises(ValueError):
            self.controller.select_ratio(self.key, context_length=100, num_layers=0)

    def test_decision_predicted_ttft_matches_predict_ttft_ms(self):
        from sglang.srt.mem_cache.cachetune.hardware_profile import predict_ttft_ms

        decision = self.controller.select_ratio(
            self.key, context_length=512, num_layers=6
        )
        expected = predict_ttft_ms(
            self.controller.measurement(self.key),
            num_layers=6,
            context_length=512,
            ratio=decision.executable_ratio,
        )
        self.assertAlmostEqual(decision.predicted_ttft_ms, expected)

    def test_decision_is_a_cachetune_decision_instance(self):
        decision = self.controller.select_ratio(
            self.key, context_length=256, num_layers=4
        )
        self.assertIsInstance(decision, CacheTuneDecision)
        self.assertEqual(decision.repair_tokens, decision.quantized.repair_tokens)
        self.assertEqual(decision.executable_ratio, decision.quantized.executable_ratio)


class TestProfileKeyIsolation(unittest.TestCase):
    def test_distinct_profile_keys_never_share_measurements_or_calibrations(self):
        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        key_a = _key(bucket=1024, tier="gpu-a")
        key_b = _key(bucket=1024, tier="gpu-b")
        controller.record_measurement(
            key_a, HardwareMeasurement(t_c_ms=2.0, t_i_ms=0.02, t_o_ms=0.0)
        )
        controller.record_measurement(
            key_b, HardwareMeasurement(t_c_ms=0.02, t_i_ms=2.0, t_o_ms=0.0)
        )
        decision_a = controller.select_ratio(key_a, context_length=1000, num_layers=4)
        decision_b = controller.select_ratio(key_b, context_length=1000, num_layers=4)
        self.assertLess(decision_a.executable_ratio, 0.05)
        self.assertGreater(decision_b.executable_ratio, 0.95)

        with self.assertRaises(CacheTuneProfileError):
            controller.select_ratio(
                _key(bucket=2048, tier="gpu-a"), context_length=1000, num_layers=4
            )

    def test_distinct_chunk_length_buckets_are_isolated(self):
        controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        small = _key(bucket=512)
        large = _key(bucket=4096)
        controller.record_measurement(
            small, HardwareMeasurement(t_c_ms=0.1, t_i_ms=0.1, t_o_ms=0.0)
        )
        self.assertTrue(controller.has_measurement(small))
        self.assertFalse(controller.has_measurement(large))


class TestCalibrate(unittest.TestCase):
    def setUp(self):
        self.controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
        self.key = _key()
        # Balanced hardware: r0 = 0.5.
        self.controller.record_measurement(
            self.key, HardwareMeasurement(t_c_ms=0.1, t_i_ms=0.1, t_o_ms=0.0)
        )

    def test_calibration_converges_near_the_true_measured_minimum(self):
        # A synthetic measured-TTFT curve whose true minimum sits at
        # ratio=0.30, deliberately offset from the roofline r0=0.5 so the
        # test proves calibration actually moves the selection, not just
        # reports the roofline value back.
        def evaluate(ratio: float) -> float:
            return (ratio - 0.30) ** 2 * 1000.0 + 50.0

        result = self.controller.calibrate(
            self.key, context_length=1000, evaluate=evaluate
        )
        self.assertIsInstance(result, CalibrationResult)
        self.assertAlmostEqual(result.ratio, 0.30, places=1)
        self.assertAlmostEqual(result.warm_start_ratio, 0.5)

    def test_calibration_result_is_memoized_on_controller(self):
        self.controller.calibrate(
            self.key, context_length=1000, evaluate=lambda ratio: 1.0
        )
        self.assertTrue(self.controller.has_calibration(self.key))
        cached = self.controller.calibration(self.key)
        self.assertIsInstance(cached, CalibrationResult)

    def test_select_ratio_uses_calibration_when_available(self):
        def evaluate(ratio: float) -> float:
            return (ratio - 0.30) ** 2 * 1000.0 + 50.0

        self.controller.calibrate(self.key, context_length=1000, evaluate=evaluate)
        decision = self.controller.select_ratio(
            self.key, context_length=1000, num_layers=4
        )
        self.assertEqual(decision.source, "calibrated")
        self.assertAlmostEqual(decision.executable_ratio, 0.30, places=1)

    def test_select_ratio_falls_back_to_roofline_without_calibration(self):
        decision = self.controller.select_ratio(
            self.key, context_length=1000, num_layers=4
        )
        self.assertEqual(decision.source, "roofline")
        self.assertAlmostEqual(decision.executable_ratio, 0.5, places=2)

    def test_evaluate_is_never_called_twice_for_the_same_executable_ratio(self):
        calls: list[float] = []

        def evaluate(ratio: float) -> float:
            calls.append(ratio)
            return (ratio - 0.3) ** 2

        self.controller.calibrate(self.key, context_length=1000, evaluate=evaluate)
        self.assertEqual(len(calls), len(set(calls)))

    def test_evaluate_raising_non_finite_is_rejected(self):
        def evaluate(ratio: float) -> float:
            return math.nan

        with self.assertRaises(ValueError):
            self.controller.calibrate(self.key, context_length=1000, evaluate=evaluate)

    def test_non_positive_context_length_rejected(self):
        with self.assertRaises(ValueError):
            self.controller.calibrate(
                self.key, context_length=0, evaluate=lambda ratio: 1.0
            )

    def test_calibration_tie_break_prefers_smaller_ratio(self):
        # A flat measured-TTFT curve: every executable ratio measures
        # identically, so the deterministic tie-break must select the
        # smallest ratio actually probed, never an arbitrary one.
        def evaluate(ratio: float) -> float:
            return 42.0

        result = self.controller.calibrate(
            self.key, context_length=1000, evaluate=evaluate
        )
        self.assertEqual(result.ratio, min(ratio for ratio, _ in result.probes))

    def test_calibration_is_deterministic_across_repeated_calls(self):
        def evaluate(ratio: float) -> float:
            return (ratio - 0.42) ** 2

        results = set()
        for _ in range(5):
            controller = CacheTuneController(CacheTuneMode.SPEED_ONLY)
            controller.record_measurement(
                self.key, HardwareMeasurement(t_c_ms=0.1, t_i_ms=0.1, t_o_ms=0.0)
            )
            result = controller.calibrate(
                self.key, context_length=1000, evaluate=evaluate
            )
            results.add(result.ratio)
        self.assertEqual(len(results), 1)

    def test_calibration_probes_are_bounded_by_mode(self):
        def evaluate(ratio: float) -> float:
            return (ratio - 0.0) ** 2

        paper_controller = CacheTuneController(CacheTuneMode.PAPER_MECHANISM)
        paper_controller.record_measurement(
            self.key, HardwareMeasurement(t_c_ms=0.1, t_i_ms=0.1, t_o_ms=0.0)
        )
        result = paper_controller.calibrate(
            self.key, context_length=1000, evaluate=evaluate
        )
        for ratio, _ in result.probes:
            self.assertGreaterEqual(ratio, 0.15 - 1e-9)


if __name__ == "__main__":
    unittest.main()
