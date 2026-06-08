# Template Code-Base Segment Ablation

## Summary

- Scenarios: 6
- Max exact hits: 3
- Max estimated cached tokens: 14967
- Latency is not reported in this non-serving ablation; serving latency is covered by the SGLang prefetch experiments.

## Main Table

| workflow | segments | agents | exact hits | estimated cached tokens |
|---|---:|---:|---:|---:|
| planner_implementer | 1 | 2 | 1 | 4988 |
| planner_implementer | 2 | 2 | 2 | 9977 |
| planner_implementer | 3 | 2 | 2 | 9977 |
| planner_implementer_debugger | 1 | 3 | 1 | 4988 |
| planner_implementer_debugger | 2 | 3 | 2 | 9977 |
| planner_implementer_debugger | 3 | 3 | 3 | 14967 |

## Interpretation

The ablation isolates the template contract: as templates expose more repeated code-base segments and more downstream agents, exact-content hits and reusable-token opportunity increase.
