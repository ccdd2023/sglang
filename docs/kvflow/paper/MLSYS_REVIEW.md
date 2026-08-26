# MLSys research-track review: ImpactKV

Target: `paper_swebench_ucm/main.pdf` (ASPLOS-formatted, serving/KV claims).
Field refresh: MLSys 2026 CFP; CacheWise/ForeCache (arXiv:2606.16824);
LMCache (arXiv:2510.09665); SGLang radix; vLLM prefix; Modular KV-cache recap.

## Rapid-review memo (first two pages)

Problem: coding-agent file KV reappears at nonzero RoPE Δ; prefix cache misses.
Method: admit version-valid single-file token-identical islands; source-side K
pre-rotate; mechanical fail-closed; prefetch off.
Headline: 1.375× cache-ready vs own Dense; 0.841× at N=4 with source build;
1684/1684 copies, 0 fallback; 94.8% one-token agreement is not resolved.
Falsify: leftover prefix/prefetch on; mixed token IDs; 1.375 sold as one-use e2e.
MLSys fit: serving/KV, but single-stream exact-prompt replay, not concurrent SLO.

## Summary

The paper studies true-lossy reuse of repository file modules that match in
token IDs but not in position. That is a real hole relative to radix/prefix
and to CacheWise-style prefix-aware eviction. The measurement is honest about
cache-ready vs source-inclusive. It is not yet a serving-systems evaluation
under load: no concurrency, no P99 SLO, no session completion time, no
same-token comparison to CacheBlend/KVCOMM, and quality is a 1-token proxy.

## Strengths

- Correctly separates Δ=0 prefix from Δ≠0 file-module copy; prefetch compiled off.
- Own-Dense paired TTFT on reconstructed SWE-bench traces, not a straw baseline.
- Source-inclusive N-use is reported, including N=1 losing (0.389× after this revision).
- Frozen RESULT.json exists for the headline table.

## Weaknesses (ranked)

1. [fatal for a serving claim] No concurrent load, no P99, no session completion.
   Evidence: eval is 1-token replay, batch implied 1, 235 groups sequential.
   Fix: replay traces with N concurrent sessions; report goodput and P99 TTFT.

2. [major] Cache-ready 1.375× is prefill-dominated by construction (1-token decode).
   Evidence: abstract and Table 1. CacheWise argues session completion is the
   coding-agent SLO, not per-request TTFT.
   Fix: keep cache-ready, add a decode-length or session-level column, or
   narrow the claim to “prefill of shifted file islands.”

3. [major] Closest MLSys 2026 paper (CacheWise/ForeCache) was missing; it attacks
   prefix thrashing, which this paper’s dropped 48 zero-shift islands belong to.
   Evidence: related work named LMCache in one clause without a citation (now fixed).
   Fix: cite and draw the Δ=0 vs Δ≠0 boundary explicitly.

4. [major] No same-token head-to-head vs a general lossy copier (CacheBlend/KVCOMM)
   on this 30B SWE-bench stream. Retention argument from 7B DS-1000 cannot be mixed.
   Fix: one 30B arm with identical target IDs, or drop the “policy vs copier” claim.

5. [minor] One-token agreement 94.8% is not task quality. Live-agent 5/24 vs 3/24
   is underpowered (p=0.625) and correctly not the headline.

6. [minor] Single model (Qwen3-Coder 30B AWQ), one GPU job. Portable-mechanism
   claim is scoped, but Impact is “serving” in the title.

## Questions for authors

1. After source-inclusive N=1 is 0.389×, what production pattern makes N≥8 common?
2. Of 48 dropped zero-shift islands, what fraction of prompt tokens would radix hit?
3. Does the copy kernel still rotate leftover Δ at copy time, or fail-closed?
4. Why not enable ordinary prefix for the Δ=0 remainder while measuring the
   Δ≠0 islands separately (without attributing prefix hits to ImpactKV)?

## Related work the authors missed

- CacheWise / ForeCache (MLSys 2026 / arXiv:2606.16824): coding-agent prefix
  scheduling and eviction. Orthogonal, but the paper must say so.
- LMCache (arXiv:2510.09665): KV as a first-class layer; cite, not a clause.
- SwiftCache (arXiv:2606.16135): P99 TTFT vs vLLM/SGLang/LMCache.
- SGLang HiSparse / HiCache: hierarchical prefix reuse under memory pressure.

## Empirical evaluation checklist

| Category | Verdict |
|---|---|
| Clearly stated claims | pass (scoped, N-use shown) |
| Appropriate baselines | partial (own Dense yes; no production prefix-on arm) |
| Principled workloads | partial (real traces, sequential replay) |
| Sufficient data | pass for paired TTFT (705 pairs) |
| Principled metrics | partial (cache-ready honest; no SLO) |
| Adequate analysis | pass after N-use/length derivation |
| Reproducible presentation | pass (RESULT + ANALYSIS from frozen JSON) |

## Serving kill-shot card

1. Metric laundering — **hit** (cache-ready vs wall-clock; no P99)
2. Baseline confounding — **miss** (prefix/prefetch off; own Dense)
3. Workload — **hit** (no concurrency / session SLO)
4. Quality — **hit** (1-token proxy)
5. Amortization — **miss** after N-use table (N=1 shown)
6. Mechanism honesty — **partial** (paper now matches leftover-rotate kernel)
7. SOTA theater — **miss** (no crown claimed)
8. Overclaim scope — **hit** (one model, one job, “serving” in title)
9. Missing related — **hit** (CacheWise); being patched
10. Artifact — **miss** (RESULT.json + now ANALYSIS.json)

Fatal/major hits on the *serving-systems* reading of the title: 1, 2, 3, 8, 9.
If the claim is narrowed to “prefill copy of shifted file islands,” 1 and 2
downgrade.

## Score

- Novelty: 3 (true-lossy file-module admit is real; not paging/prefix)
- Quality: 3 (honest accounting; incomplete serving eval)
- Interest: 3 (coding agents are hot; this eval is sequential)
- Impact: 2 (N=1 loses; no load; 30B AWQ only)
- Overall: **weak reject** as a serving paper; **borderline** as a
  mechanism paper if the title/claim stay prefill-scoped
- Confidence: medium-high

## Revision list (priority order)

1. Cite CacheWise/LMCache and draw Δ=0 vs Δ≠0 in related work. **(this round)**
2. Publish N-use {1,2,4,8} and length-bucket speedup from frozen 96092 JSON. **(this round)**
3. Narrow title/abstract “serving” to prefill of shifted file islands, or add
   a concurrent replay.
4. Same-token 30B lossy-copier arm, or delete policy-vs-copier language.
5. Optional GPU: concurrent session completion; not required to keep the
   mechanism claim.
