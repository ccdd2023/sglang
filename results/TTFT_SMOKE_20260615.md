=== Long-prompt TTFT smoke summary (max_tokens=1, prefill-only) ===

prompt     lossless      lossy    speedup
2K           123.7ms      47.0ms     2.63x
4K           125.4ms      54.6ms     2.30x
8K           225.0ms      63.4ms     3.55x
16K          497.7ms      79.2ms     6.28x
32K         1323.7ms     104.4ms    12.68x

Findings:
1. On long prefill-only prompts, lossy mode saves 60-92% of TTFT
2. Speedup grows with prompt length (2.63x at 2K → 12.68x at 32K)
3. On real SWE-bench (max_tokens=1024), TTFT is only 1.1% of E2E so this is invisible in E2E

Implication for paper:
- The 1.16x headline is a stress-test result, not a real-workload result
- For real users, speedup is in TTFT which is hidden in E2E latency
- Reframe: prefetch-hint saves X seconds of prefill = Y% throughput improvement
