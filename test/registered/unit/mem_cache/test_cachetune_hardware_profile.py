from __future__ import annotations

import math
import unittest

from sglang.srt.mem_cache.cachetune.hardware_profile import (
    PAPER_MECHANISM_R_MIN,
    CacheTuneMode,
    DenseTimingSample,
    HardwareMeasurement,
    HardwareProfileKey,
    QuantizedRatio,
    RatioBounds,
    TransferTimingSample,
    chunk_length_bucket,
    estimate_measurement_from_samples,
    predict_layer_time_ms,
    predict_ttft_ms,
    quantize_ratio,
    roofline_ratio,
    round_half_up,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class TestHardwareProfileKey(unittest.TestCase):
    def test_valid_key_constructs(self):
        key = HardwareProfileKey(
            hardware_tier="NVIDIA RTX 2080 SUPER",
            model_fingerprint="qwen3-0.6b-fp16",
            chunk_length_bucket=1024,
        )
        self.assertEqual(key.hardware_tier, "NVIDIA RTX 2080 SUPER")
        self.assertEqual(key.chunk_length_bucket, 1024)

    def test_empty_hardware_tier_rejected(self):
        with self.assertRaises(ValueError):
            HardwareProfileKey(
                hardware_tier="  ",
                model_fingerprint="model",
                chunk_length_bucket=1,
            )

    def test_empty_model_fingerprint_rejected(self):
        with self.assertRaises(ValueError):
            HardwareProfileKey(
                hardware_tier="cpu",
                model_fingerprint="",
                chunk_length_bucket=1,
            )

    def test_non_positive_chunk_length_bucket_rejected(self):
        with self.assertRaises(ValueError):
            HardwareProfileKey(
                hardware_tier="cpu",
                model_fingerprint="model",
                chunk_length_bucket=0,
            )
        with self.assertRaises(ValueError):
            HardwareProfileKey(
                hardware_tier="cpu",
                model_fingerprint="model",
                chunk_length_bucket=-4,
            )

    def test_keys_are_hashable_and_isolate_by_every_field(self):
        # Profile-key isolation: distinct hardware tier, model
        # fingerprint, or chunk-length bucket must never collide in a
        # dict/set used to key measurements.
        base = HardwareProfileKey(
            hardware_tier="cpu", model_fingerprint="model-a", chunk_length_bucket=1024
        )
        same = HardwareProfileKey(
            hardware_tier="cpu", model_fingerprint="model-a", chunk_length_bucket=1024
        )
        diff_tier = HardwareProfileKey(
            hardware_tier="gpu", model_fingerprint="model-a", chunk_length_bucket=1024
        )
        diff_model = HardwareProfileKey(
            hardware_tier="cpu", model_fingerprint="model-b", chunk_length_bucket=1024
        )
        diff_bucket = HardwareProfileKey(
            hardware_tier="cpu", model_fingerprint="model-a", chunk_length_bucket=2048
        )
        self.assertEqual(base, same)
        self.assertEqual(hash(base), hash(same))
        registry = {base: "measurement-a"}
        self.assertIn(same, registry)
        self.assertNotIn(diff_tier, registry)
        self.assertNotIn(diff_model, registry)
        self.assertNotIn(diff_bucket, registry)
        distinct_keys = {base, diff_tier, diff_model, diff_bucket}
        self.assertEqual(len(distinct_keys), 4)


class TestChunkLengthBucket(unittest.TestCase):
    def test_exact_power_of_two_maps_to_itself(self):
        self.assertEqual(chunk_length_bucket(1), 1)
        self.assertEqual(chunk_length_bucket(1024), 1024)

    def test_non_power_of_two_rounds_up(self):
        self.assertEqual(chunk_length_bucket(500), 512)
        self.assertEqual(chunk_length_bucket(620), 1024)
        self.assertEqual(chunk_length_bucket(1025), 2048)

    def test_non_positive_context_length_rejected(self):
        with self.assertRaises(ValueError):
            chunk_length_bucket(0)
        with self.assertRaises(ValueError):
            chunk_length_bucket(-1)


class TestHardwareMeasurement(unittest.TestCase):
    def test_valid_measurement_constructs(self):
        measurement = HardwareMeasurement(t_c_ms=0.05, t_i_ms=0.02, t_o_ms=0.1)
        self.assertEqual(measurement.sample_count, 1)

    def test_non_positive_t_c_rejected(self):
        with self.assertRaises(ValueError):
            HardwareMeasurement(t_c_ms=0.0, t_i_ms=0.02, t_o_ms=0.1)
        with self.assertRaises(ValueError):
            HardwareMeasurement(t_c_ms=-1.0, t_i_ms=0.02, t_o_ms=0.1)

    def test_non_positive_t_i_rejected(self):
        with self.assertRaises(ValueError):
            HardwareMeasurement(t_c_ms=0.05, t_i_ms=0.0, t_o_ms=0.1)

    def test_negative_t_o_rejected(self):
        with self.assertRaises(ValueError):
            HardwareMeasurement(t_c_ms=0.05, t_i_ms=0.02, t_o_ms=-0.1)

    def test_zero_t_o_is_allowed(self):
        # Fixed pipeline overhead may genuinely be (measured as) zero;
        # only negative values are invalid.
        HardwareMeasurement(t_c_ms=0.05, t_i_ms=0.02, t_o_ms=0.0)

    def test_non_finite_values_rejected(self):
        with self.assertRaises(ValueError):
            HardwareMeasurement(t_c_ms=math.inf, t_i_ms=0.02, t_o_ms=0.1)
        with self.assertRaises(ValueError):
            HardwareMeasurement(t_c_ms=math.nan, t_i_ms=0.02, t_o_ms=0.1)

    def test_non_positive_sample_count_rejected(self):
        with self.assertRaises(ValueError):
            HardwareMeasurement(t_c_ms=0.05, t_i_ms=0.02, t_o_ms=0.1, sample_count=0)


class TestRooflineRatio(unittest.TestCase):
    def test_compute_bound_hardware_favors_small_ratio(self):
        # t_c (recompute) far more expensive per-token than t_i
        # (transfer): the roofline optimum should sit close to r=0 (do
        # almost everything via transfer, almost nothing via recompute).
        measurement = HardwareMeasurement(t_c_ms=1.0, t_i_ms=0.01, t_o_ms=0.0)
        r0 = roofline_ratio(measurement)
        self.assertLess(r0, 0.05)
        self.assertGreater(r0, 0.0)

    def test_io_bound_hardware_favors_large_ratio(self):
        # t_i (transfer) far more expensive per-token than t_c
        # (recompute): the roofline optimum should sit close to r=1 (do
        # almost everything via recompute, almost nothing via transfer).
        measurement = HardwareMeasurement(t_c_ms=0.01, t_i_ms=1.0, t_o_ms=0.0)
        r0 = roofline_ratio(measurement)
        self.assertGreater(r0, 0.95)
        self.assertLess(r0, 1.0)

    def test_balanced_hardware_gives_r0_one_half(self):
        measurement = HardwareMeasurement(t_c_ms=0.5, t_i_ms=0.5, t_o_ms=0.0)
        self.assertAlmostEqual(roofline_ratio(measurement), 0.5)

    def test_r0_equalizes_the_two_critical_paths(self):
        measurement = HardwareMeasurement(t_c_ms=0.3, t_i_ms=0.07, t_o_ms=1.0)
        r0 = roofline_ratio(measurement)
        context_length = 4096
        recompute_ms = r0 * context_length * measurement.t_c_ms
        transfer_ms = (1.0 - r0) * context_length * measurement.t_i_ms
        self.assertAlmostEqual(recompute_ms, transfer_ms, places=6)

    def test_r0_is_always_strictly_within_unit_interval(self):
        for t_c, t_i in ((1e-6, 1e6), (1e6, 1e-6), (3.3, 3.3), (0.001, 0.002)):
            with self.subTest(t_c=t_c, t_i=t_i):
                r0 = roofline_ratio(
                    HardwareMeasurement(t_c_ms=t_c, t_i_ms=t_i, t_o_ms=0.0)
                )
                self.assertGreater(r0, 0.0)
                self.assertLess(r0, 1.0)


class TestPredictLayerAndTTFT(unittest.TestCase):
    def setUp(self):
        self.measurement = HardwareMeasurement(t_c_ms=0.1, t_i_ms=0.05, t_o_ms=2.0)

    def test_layer_time_is_max_of_the_two_paths_plus_overhead(self):
        context_length = 1000
        ratio = 0.8
        expected = (
            max(
                ratio * context_length * self.measurement.t_c_ms,
                (1.0 - ratio) * context_length * self.measurement.t_i_ms,
            )
            + self.measurement.t_o_ms
        )
        actual = predict_layer_time_ms(
            self.measurement, context_length=context_length, ratio=ratio
        )
        self.assertAlmostEqual(actual, expected)

    def test_layer_time_at_r0_is_the_minimum_over_a_ratio_sweep(self):
        r0 = roofline_ratio(self.measurement)
        context_length = 2048
        at_r0 = predict_layer_time_ms(
            self.measurement, context_length=context_length, ratio=r0
        )
        for ratio in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            with self.subTest(ratio=ratio):
                other = predict_layer_time_ms(
                    self.measurement, context_length=context_length, ratio=ratio
                )
                self.assertLessEqual(at_r0, other + 1e-9)

    def test_ttft_is_num_layers_times_layer_time(self):
        context_length = 512
        ratio = 0.3
        num_layers = 24
        layer_time = predict_layer_time_ms(
            self.measurement, context_length=context_length, ratio=ratio
        )
        ttft = predict_ttft_ms(
            self.measurement,
            num_layers=num_layers,
            context_length=context_length,
            ratio=ratio,
        )
        self.assertAlmostEqual(ttft, num_layers * layer_time)

    def test_non_positive_context_length_rejected(self):
        with self.assertRaises(ValueError):
            predict_layer_time_ms(self.measurement, context_length=0, ratio=0.5)
        with self.assertRaises(ValueError):
            predict_ttft_ms(
                self.measurement, num_layers=1, context_length=-5, ratio=0.5
            )

    def test_ratio_outside_unit_interval_rejected(self):
        with self.assertRaises(ValueError):
            predict_layer_time_ms(self.measurement, context_length=10, ratio=-0.1)
        with self.assertRaises(ValueError):
            predict_layer_time_ms(self.measurement, context_length=10, ratio=1.1)

    def test_non_positive_num_layers_rejected(self):
        with self.assertRaises(ValueError):
            predict_ttft_ms(
                self.measurement, num_layers=0, context_length=10, ratio=0.5
            )


class TestRatioBounds(unittest.TestCase):
    def test_paper_mechanism_uses_fifteen_percent_floor(self):
        bounds = RatioBounds.for_mode(CacheTuneMode.PAPER_MECHANISM)
        self.assertAlmostEqual(bounds.r_min, PAPER_MECHANISM_R_MIN)
        self.assertAlmostEqual(bounds.r_min, 0.15)
        self.assertAlmostEqual(bounds.r_max, 1.0)

    def test_speed_only_allows_zero_floor(self):
        bounds = RatioBounds.for_mode(CacheTuneMode.SPEED_ONLY)
        self.assertAlmostEqual(bounds.r_min, 0.0)
        self.assertAlmostEqual(bounds.r_max, 1.0)

    def test_for_mode_custom_r_max(self):
        bounds = RatioBounds.for_mode(CacheTuneMode.SPEED_ONLY, r_max=0.8)
        self.assertAlmostEqual(bounds.r_max, 0.8)

    def test_for_mode_rejects_non_enum_value(self):
        with self.assertRaises(ValueError):
            RatioBounds.for_mode("speed_only")  # type: ignore[arg-type]

    def test_r_min_greater_than_r_max_rejected(self):
        with self.assertRaises(ValueError):
            RatioBounds(r_min=0.5, r_max=0.3)

    def test_out_of_unit_interval_rejected(self):
        with self.assertRaises(ValueError):
            RatioBounds(r_min=-0.1, r_max=0.5)
        with self.assertRaises(ValueError):
            RatioBounds(r_min=0.1, r_max=1.5)

    def test_clamp_bounds_below_above_and_inside(self):
        bounds = RatioBounds(r_min=0.15, r_max=0.9)
        self.assertAlmostEqual(bounds.clamp(0.0), 0.15)
        self.assertAlmostEqual(bounds.clamp(1.0), 0.9)
        self.assertAlmostEqual(bounds.clamp(0.5), 0.5)

    def test_clamp_rejects_non_finite_ratio(self):
        bounds = RatioBounds(r_min=0.0, r_max=1.0)
        with self.assertRaises(ValueError):
            bounds.clamp(math.nan)
        with self.assertRaises(ValueError):
            bounds.clamp(math.inf)


class TestRoundHalfUp(unittest.TestCase):
    def test_exact_half_boundary_rounds_up_not_to_even(self):
        # Python's builtin round() uses banker's rounding: round(0.5) ==
        # 0, round(1.5) == 2, round(2.5) == 2. round_half_up must always
        # round exact .5 ties up, regardless of even/odd neighbor.
        self.assertEqual(round_half_up(0.5), 1)
        self.assertEqual(round_half_up(1.5), 2)
        self.assertEqual(round_half_up(2.5), 3)
        self.assertEqual(round_half_up(3.5), 4)

    def test_non_boundary_values_round_normally(self):
        self.assertEqual(round_half_up(2.4), 2)
        self.assertEqual(round_half_up(2.6), 3)

    def test_negative_values(self):
        self.assertEqual(round_half_up(-0.5), 0)
        self.assertEqual(round_half_up(-1.5), -1)

    def test_zero(self):
        self.assertEqual(round_half_up(0.0), 0)


class TestQuantizeRatio(unittest.TestCase):
    def test_mid_range_ratio_rounds_to_nearest_token_count(self):
        bounds = RatioBounds(r_min=0.0, r_max=1.0)
        result = quantize_ratio(0.203, context_length=100, bounds=bounds)
        self.assertEqual(result.repair_tokens, 20)
        self.assertAlmostEqual(result.executable_ratio, 0.20)

    def test_exact_fifteen_percent_boundary_at_paper_floor(self):
        # At the paper-mechanism r_min=15% floor, a request whose ratio
        # is clamped exactly to r_min over a context length of 100 must
        # resolve to exactly 15 repair tokens (0.15 * 100), not 14 or 16.
        bounds = RatioBounds.for_mode(CacheTuneMode.PAPER_MECHANISM)
        result = quantize_ratio(0.15, context_length=100, bounds=bounds)
        self.assertEqual(result.repair_tokens, 15)

    def test_boundary_ratio_with_genuine_fp_representation_error_is_included(self):
        # 0.14 * 100 == 14.000000000000002 in binary floating point (a
        # genuine, reproducible representation error at this exact
        # scale -- unlike 0.15 * 100, which happens to be exact). Without
        # the epsilon guard in `quantize_ratio`, `math.ceil(0.14 * 100)`
        # would evaluate to 15, incorrectly excluding the achievable
        # 14-token count from the admissible range and forcing every
        # ratio down at this floor to round up one token too many.
        self.assertEqual(math.ceil(0.14 * 100), 15)
        self.assertNotEqual(0.14 * 100, 14)
        bounds = RatioBounds(r_min=0.14, r_max=1.0)
        result = quantize_ratio(0.0, context_length=100, bounds=bounds)
        self.assertEqual(result.repair_tokens, 14)

    def test_ratio_below_r_min_is_clamped_up_to_the_floor(self):
        bounds = RatioBounds.for_mode(CacheTuneMode.PAPER_MECHANISM)
        result = quantize_ratio(0.0, context_length=200, bounds=bounds)
        self.assertEqual(result.repair_tokens, 30)  # ceil(0.15 * 200)
        self.assertAlmostEqual(result.bounded_ratio, 0.15)

    def test_ratio_above_r_max_is_clamped_down_to_the_ceiling(self):
        bounds = RatioBounds(r_min=0.0, r_max=0.5)
        result = quantize_ratio(0.9, context_length=100, bounds=bounds)
        self.assertEqual(result.repair_tokens, 50)

    def test_speed_only_zero_ratio_yields_zero_repair_tokens(self):
        bounds = RatioBounds.for_mode(CacheTuneMode.SPEED_ONLY)
        result = quantize_ratio(0.0, context_length=777, bounds=bounds)
        self.assertEqual(result.repair_tokens, 0)
        self.assertAlmostEqual(result.executable_ratio, 0.0)

    def test_full_ratio_yields_context_length_tokens(self):
        bounds = RatioBounds.for_mode(CacheTuneMode.SPEED_ONLY)
        result = quantize_ratio(1.0, context_length=777, bounds=bounds)
        self.assertEqual(result.repair_tokens, 777)

    def test_result_preserves_requested_and_bounded_ratio_for_telemetry(self):
        bounds = RatioBounds.for_mode(CacheTuneMode.PAPER_MECHANISM)
        result = quantize_ratio(0.02, context_length=100, bounds=bounds)
        self.assertAlmostEqual(result.requested_ratio, 0.02)
        self.assertAlmostEqual(result.bounded_ratio, 0.15)
        self.assertEqual(result.context_length, 100)

    def test_non_positive_context_length_rejected(self):
        bounds = RatioBounds(r_min=0.0, r_max=1.0)
        with self.assertRaises(ValueError):
            quantize_ratio(0.5, context_length=0, bounds=bounds)

    def test_non_finite_ratio_rejected(self):
        bounds = RatioBounds(r_min=0.0, r_max=1.0)
        with self.assertRaises(ValueError):
            quantize_ratio(math.nan, context_length=100, bounds=bounds)

    def test_degenerate_bounds_with_no_admissible_token_count_rejected(self):
        # r_min=0.004, r_max=0.006 over a context length of 10 admits no
        # integer token count in [ceil(0.04), floor(0.06)] = [1, 0].
        bounds = RatioBounds(r_min=0.004, r_max=0.006)
        with self.assertRaises(ValueError):
            quantize_ratio(0.005, context_length=10, bounds=bounds)

    def test_quantized_ratio_is_deterministic_across_repeated_calls(self):
        bounds = RatioBounds.for_mode(CacheTuneMode.SPEED_ONLY)
        results = {
            quantize_ratio(0.123456, context_length=4096, bounds=bounds).repair_tokens
            for _ in range(20)
        }
        self.assertEqual(len(results), 1)

    def test_quantized_ratio_is_frozen_dataclass(self):
        bounds = RatioBounds.for_mode(CacheTuneMode.SPEED_ONLY)
        result = quantize_ratio(0.5, context_length=100, bounds=bounds)
        self.assertIsInstance(result, QuantizedRatio)
        with self.assertRaises(Exception):
            result.repair_tokens = 999  # type: ignore[misc]


class TestEstimateMeasurementFromSamples(unittest.TestCase):
    def test_derives_measurement_from_genuine_timing_samples(self):
        # Construct samples consistent with a known ground-truth
        # measurement so the derivation can be checked exactly:
        # t_c=0.2ms/token/layer, t_o=1.5ms/layer, num_layers=10.
        num_layers = 10
        t_c_ms = 0.2
        t_o_ms = 1.5
        dense_small = DenseTimingSample(
            context_length=100, ttft_ms=num_layers * (t_c_ms * 100 + t_o_ms)
        )
        dense_large = DenseTimingSample(
            context_length=300, ttft_ms=num_layers * (t_c_ms * 300 + t_o_ms)
        )
        # transfer: t_i=0.05ms/token/layer over 100 tokens, 10 layers.
        transfer = TransferTimingSample(tokens=100, copy_ms=30.0, rope_ms=20.0)
        measurement = estimate_measurement_from_samples(
            dense_small=dense_small,
            dense_large=dense_large,
            transfer=transfer,
            num_layers=num_layers,
        )
        self.assertAlmostEqual(measurement.t_c_ms, t_c_ms, places=9)
        self.assertAlmostEqual(measurement.t_o_ms, t_o_ms, places=9)
        self.assertAlmostEqual(measurement.t_i_ms, 50.0 / (100 * num_layers), places=9)
        self.assertEqual(measurement.sample_count, 3)

    def test_dense_large_must_use_a_larger_context_length(self):
        dense_small = DenseTimingSample(context_length=200, ttft_ms=50.0)
        dense_large = DenseTimingSample(context_length=200, ttft_ms=50.0)
        transfer = TransferTimingSample(tokens=100, copy_ms=10.0, rope_ms=5.0)
        with self.assertRaises(ValueError):
            estimate_measurement_from_samples(
                dense_small=dense_small,
                dense_large=dense_large,
                transfer=transfer,
                num_layers=4,
            )

    def test_non_increasing_ttft_rejected(self):
        dense_small = DenseTimingSample(context_length=100, ttft_ms=50.0)
        dense_large = DenseTimingSample(context_length=300, ttft_ms=50.0)
        transfer = TransferTimingSample(tokens=100, copy_ms=10.0, rope_ms=5.0)
        with self.assertRaises(ValueError):
            estimate_measurement_from_samples(
                dense_small=dense_small,
                dense_large=dense_large,
                transfer=transfer,
                num_layers=4,
            )

    def test_negative_derived_overhead_rejected(self):
        # dense_small's TTFT is smaller than what the derived per-token
        # slope alone would predict at its own context length -- implies
        # a negative fixed overhead, which is inconsistent with the
        # roofline model and must be rejected rather than silently
        # clamped to zero.
        dense_small = DenseTimingSample(context_length=100, ttft_ms=1.0)
        dense_large = DenseTimingSample(context_length=100_000, ttft_ms=100_000.0)
        transfer = TransferTimingSample(tokens=100, copy_ms=10.0, rope_ms=5.0)
        with self.assertRaises(ValueError):
            estimate_measurement_from_samples(
                dense_small=dense_small,
                dense_large=dense_large,
                transfer=transfer,
                num_layers=1,
            )

    def test_non_positive_transfer_timing_rejected(self):
        dense_small = DenseTimingSample(context_length=100, ttft_ms=50.0)
        dense_large = DenseTimingSample(context_length=300, ttft_ms=150.0)
        transfer = TransferTimingSample(tokens=100, copy_ms=0.0, rope_ms=0.0)
        with self.assertRaises(ValueError):
            estimate_measurement_from_samples(
                dense_small=dense_small,
                dense_large=dense_large,
                transfer=transfer,
                num_layers=4,
            )

    def test_non_positive_num_layers_rejected(self):
        dense_small = DenseTimingSample(context_length=100, ttft_ms=50.0)
        dense_large = DenseTimingSample(context_length=300, ttft_ms=150.0)
        transfer = TransferTimingSample(tokens=100, copy_ms=10.0, rope_ms=5.0)
        with self.assertRaises(ValueError):
            estimate_measurement_from_samples(
                dense_small=dense_small,
                dense_large=dense_large,
                transfer=transfer,
                num_layers=0,
            )


class TestDenseAndTransferTimingSampleValidation(unittest.TestCase):
    def test_dense_sample_rejects_non_positive_context_length(self):
        with self.assertRaises(ValueError):
            DenseTimingSample(context_length=0, ttft_ms=10.0)

    def test_dense_sample_rejects_non_positive_ttft(self):
        with self.assertRaises(ValueError):
            DenseTimingSample(context_length=10, ttft_ms=0.0)

    def test_transfer_sample_rejects_non_positive_tokens(self):
        with self.assertRaises(ValueError):
            TransferTimingSample(tokens=0, copy_ms=1.0, rope_ms=1.0)

    def test_transfer_sample_rejects_negative_copy_or_rope(self):
        with self.assertRaises(ValueError):
            TransferTimingSample(tokens=10, copy_ms=-1.0, rope_ms=1.0)
        with self.assertRaises(ValueError):
            TransferTimingSample(tokens=10, copy_ms=1.0, rope_ms=-1.0)

    def test_transfer_sample_allows_zero_copy_or_rope(self):
        TransferTimingSample(tokens=10, copy_ms=0.0, rope_ms=1.0)
        TransferTimingSample(tokens=10, copy_ms=1.0, rope_ms=0.0)


if __name__ == "__main__":
    unittest.main()
