import unittest

from benchmark.approx_kv.metrics import (
    clean_cache_invariant,
    clean_pool_reset_invariant,
    idle_pool_invariant,
    parse_prometheus_text,
    telemetry_delta,
)
from benchmark.approx_kv.workloads import (
    build_object_catalog,
    build_workflow_trace,
    select_objects_for_pressure,
    unique_prefix_token_count,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(len(text.split())))

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(f"token_{token_id}" for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        return_dict,
    ):
        self.assert_options = (tokenize, add_generation_prompt, return_dict)
        text = " ".join(message["content"] for message in messages)
        return self.encode(text)


class TestPhase2Workloads(unittest.TestCase):
    def setUp(self):
        self.catalog = build_object_catalog(
            FakeTokenizer(),
            object_count=8,
            target_sizes=(64, 96, 128, 160),
            tolerance_tokens=16,
        )

    def test_catalog_is_deterministic_and_calibrated(self):
        second = build_object_catalog(
            FakeTokenizer(),
            object_count=8,
            target_sizes=(64, 96, 128, 160),
            tolerance_tokens=16,
        )
        self.assertEqual(
            [item.manifest() for item in self.catalog],
            [item.manifest() for item in second],
        )
        for item in self.catalog:
            self.assertLessEqual(
                abs(item.reusable_prefix_tokens - item.target_prefix_tokens),
                16,
            )

    def test_pressure_selection_uses_unique_trie_tokens(self):
        total = unique_prefix_token_count(self.catalog)
        selection = select_objects_for_pressure(
            self.catalog,
            gpu_kv_capacity_tokens=max(1, total // 2),
            target_ratio=1.0,
            required_object_ids={
                self.catalog[0].object_id,
                self.catalog[1].object_id,
            },
        )
        self.assertEqual(
            selection.active_reusable_tokens,
            unique_prefix_token_count(selection.objects),
        )
        self.assertGreater(len(selection.objects), 0)
        self.assertIn(self.catalog[0], selection.objects)
        self.assertIn(self.catalog[1], selection.objects)

    def test_trace_covers_required_phases_and_reuses_every_object(self):
        trace = build_workflow_trace(self.catalog)
        phases = {item.phase for item in trace}
        self.assertTrue(
            {
                "fill",
                "workflow",
                "cold-filler",
                "retry",
                "branch-fanout",
                "replay",
                "hot-tail",
            }.issubset(phases)
        )
        counts = {}
        for item in trace:
            counts[item.object_id] = counts.get(item.object_id, 0) + 1
            if item.next_use_step is not None:
                self.assertEqual(
                    item.next_use_distance,
                    item.next_use_step - item.step,
                )
        self.assertTrue(all(count >= 2 for count in counts.values()))


class TestPhase2Metrics(unittest.TestCase):
    def test_prometheus_parser_and_delta(self):
        before = parse_prometheus_text(
            """
            sglang:max_total_num_tokens{model_name="m"} 1000
            sglang:cached_tokens_total{cache_source="device"} 100
            sglang:cached_tokens_total{cache_source="host"} 20
            sglang:evicted_tokens_total 5
            sglang:kv_available_tokens 400
            sglang:kv_evictable_tokens 600
            sglang:kv_used_tokens 0
            """
        )
        after = parse_prometheus_text(
            """
            sglang:max_total_num_tokens{model_name="m"} 1000
            sglang:cached_tokens_total{cache_source="device"} 180
            sglang:cached_tokens_total{cache_source="host"} 30
            sglang:evicted_tokens_total 25
            sglang:kv_available_tokens 300
            sglang:kv_evictable_tokens 700
            sglang:kv_used_tokens 0
            """
        )
        delta = telemetry_delta(before, after)
        self.assertEqual(
            delta["counters"]["sglang:cached_tokens_total"],
            90,
        )
        self.assertEqual(
            delta["counters"]["sglang:evicted_tokens_total"],
            20,
        )
        self.assertFalse(delta["fallback_metric_available"])
        self.assertTrue(idle_pool_invariant(after)["passed"])
        self.assertFalse(clean_cache_invariant(after)["passed"])
        clean = {
            **after,
            "sglang:kv_available_tokens": 998,
            "sglang:kv_evictable_tokens": 2,
        }
        self.assertTrue(clean_cache_invariant(clean)["passed"])
        self.assertTrue(clean_pool_reset_invariant(clean, clean)["passed"])

    def test_absent_eviction_counter_means_zero_events(self):
        delta = telemetry_delta(
            {"sglang:prompt_tokens_total": 10},
            {"sglang:prompt_tokens_total": 20},
        )
        self.assertEqual(
            delta["counters"]["sglang:evicted_tokens_total"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
