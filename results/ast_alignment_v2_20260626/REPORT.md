# AST-Alignment Partial-Match Hit Rate — Measurement Report

**Date**: 2026-06-26  
**Plan**: `/home/gfy/.claude/plans/whimsical-stirring-thimble.md` (Direction #3 measurement)  
**Workload**: 60-case stratified sweep (manifest_500.json), 5 agents per task, segment_count=3, mode=`placeholder_knn_reuse`, Qwen2.5-3B-Instruct

## Headline

- **Requests sent**: 300 (60 cases × 5 agents)
- **Placeholder pool hits**: 408
- **Placeholder pool misses**: 493
- **Max pool size**: 0
- **Prefix-cache reuse ratio**: 0.4488 (885,383 / 1,972,750 tokens)
- **AST_ALIGN log rows**: 0

## Per-Agent Breakdown

| Agent | Requests | Pool Hits | Pool Misses | Max Pool Stored | Mean Cached Ratio | Mean TTFT (ms) |
|-------|---------:|----------:|------------:|----------------:|-------------------:|---------------:|
| `auditor` | 60 | 77 | 103 | 0 | 0.4216 | 422 |
| `debugger` | 60 | 84 | 96 | 0 | 0.4599 | 395 |
| `implementer` | 60 | 78 | 103 | 0 | 0.4274 | 436 |
| `reviewer` | 60 | 86 | 94 | 0 | 0.4705 | 387 |
| `verifier` | 60 | 83 | 97 | 0 | 0.4539 | 392 |

## Decision

Hit rate measurable: 408 matches across 60 cases × 5 agents.

## Interpretation
