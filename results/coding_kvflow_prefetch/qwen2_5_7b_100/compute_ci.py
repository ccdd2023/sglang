"""Compute 95% CI and paired-bootstrap p-value for the 100-case prefetch data.
Stock SGLang vs KVCOMM and KVCOMM vs KVFlow (prefix only).
"""
import csv, math, random, statistics
import pathlib

src = pathlib.Path("/home/gfy/CodeMAS_Project/sglang-kvflow/results/coding_kvflow_prefetch/qwen2_5_7b_100/prefetch_table.csv")

cases = {}
with src.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        cid = row["instance_id"]
        mode = row["mode"]
        elapsed = float(row["elapsed_ms"])
        cached = int(row["cached_tokens"])
        cases.setdefault(cid, {})[mode] = (elapsed, cached)

# Pair cases
baselines, kvcomm_full, kvflow_only = [], [], []
for cid, modes in cases.items():
    if "baseline_prefix_cache_only" in modes and "kvcomm_lossy_plus_codebase_prefetch" in modes and "kvflow_prefix_only" in modes:
        baselines.append(modes["baseline_prefix_cache_only"])
        kvcomm_full.append(modes["kvcomm_lossy_plus_codebase_prefetch"])
        kvflow_only.append(modes["kvflow_prefix_only"])

# Per-case differences
deltas_lat_full = [b[0] - k[0] for b, k in zip(baselines, kvcomm_full)]
deltas_lat_kvflow = [b[0] - f[0] for b, f in zip(baselines, kvflow_only)]
deltas_cached = [k[1] - b[1] for b, k in zip(baselines, kvcomm_full)]

def ci_mean(d, alpha=0.05):
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    t = 1.96  # rough
    return mean, se, (mean - t * se, mean + t * se)

def paired_bootstrap_pvalue(d, n_resamples=10000):
    """Paired bootstrap: H0 = no speedup, H1 = KVCOMM faster (d > 0)."""
    n = len(d)
    random.seed(0)
    obs_mean = sum(d) / n
    count_le_zero = 0
    for _ in range(n_resamples):
        sample = [d[random.randint(0, n-1)] for _ in range(n)]
        m = sum(sample) / n
        if m <= 0:
            count_le_zero += 1
    # p-value (one-sided) for H1: KVCOMM faster (d > 0)
    return count_le_zero / n_resamples

print(f"n = {len(baselines)}")
print()
print("=== Latency: stock SGLang vs KVCOMM (paired diff = baseline - kvcomm; positive = KVCOMM faster) ===")
m, se, (lo, hi) = ci_mean(deltas_lat_full)
print(f"  mean Δ = {m:+.0f}ms  std = {statistics.stdev(deltas_lat_full):.0f}ms  95% CI = [{lo:+.0f}, {hi:+.0f}]ms")
p = paired_bootstrap_pvalue(deltas_lat_full)
print(f"  paired bootstrap p-value (one-sided, H1: KVCOMM faster) = {p:.4f}")
print()
print("=== Latency: stock SGLang vs KVFlow (no KVCOMM) ===")
m, se, (lo, hi) = ci_mean(deltas_lat_kvflow)
print(f"  mean Δ = {m:+.0f}ms  std = {statistics.stdev(deltas_lat_kvflow):.0f}ms  95% CI = [{lo:+.0f}, {hi:+.0f}]ms")
p = paired_bootstrap_pvalue(deltas_lat_kvflow)
print(f"  paired bootstrap p-value (one-sided, H1: KVFlow faster) = {p:.4f}")
print()
print("=== Cached tokens: stock SGLang vs KVCOMM ===")
m, se, (lo, hi) = ci_mean(deltas_cached)
print(f"  mean Δ = {m:+.0f}  std = {statistics.stdev(deltas_cached):.0f}  95% CI = [{lo:+.0f}, {hi:+.0f}]")
p = paired_bootstrap_pvalue(deltas_cached)
print(f"  paired bootstrap p-value (one-sided, H1: KVCOMM caches more) = {p:.4f}")

# Also compute a few more stats: p50, p90, p99 of latency
def pct(xs, p):
    s = sorted(xs)
    k = int(p/100 * len(s))
    return s[min(k, len(s)-1)]

print()
print("=== Latency percentiles (100 cases each) ===")
print(f"                  p50      p90      p99      max")
for label, data in [("Stock SGLang", [b[0] for b in baselines]),
                     ("KVFlow (no KVCOMM)", [f[0] for f in kvflow_only]),
                     ("KVCOMM (full)", [k[0] for k in kvcomm_full])]:
    print(f"  {label:<20} {pct(data,50):>6.0f}  {pct(data,90):>6.0f}  {pct(data,99):>6.0f}  {max(data):>6.0f}")
