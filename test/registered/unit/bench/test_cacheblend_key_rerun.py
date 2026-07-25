from __future__ import annotations

import unittest

from benchmark.approx_kv.run_phase4_cacheblend_key_rerun import (
    parse_body_tokens,
    seed_prompt,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")


class TestCacheBlendKeyRerunHelpers(unittest.TestCase):
    def test_parse_body_tokens(self):
        self.assertEqual(parse_body_tokens("1024,2048"), (1024, 2048))

    def test_parse_body_tokens_rejects_non_positive_values(self):
        with self.assertRaises(ValueError):
            parse_body_tokens("1024,0")

    def test_seed_prompt_forces_divergence_before_body(self):
        header = [10, 11, 12]
        prompt = seed_prompt(header, body_first_token=1000)
        self.assertEqual(prompt[:-1], header)
        self.assertNotEqual(prompt[-1], 1000)


if __name__ == "__main__":
    unittest.main()
