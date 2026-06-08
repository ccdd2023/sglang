# Real-Trace Reuse Rate (SWE-bench Verified)

- **Total records**: 1356
- **Distinct instances**: 452

## 1. Hit rate by agent pair

| agent pair | n pairs | lossy allowed | matched content sig | hit rate |
|---|---|---|---|---|
| planner→coder | 452 | 452 | 452 | 100.0% |
| coder→reviewer | 452 | 452 | 452 | 100.0% |
| planner→reviewer | 452 | 452 | 452 | 100.0% |
| planner→planner | 452 | 452 | 1 | 0.2% |
| coder→coder | 452 | 452 | 452 | 100.0% |
| reviewer→reviewer | 452 | 452 | 452 | 100.0% |

## 2. Cache savings (overall)

| instances | records | lossy matches | tokens saved | hit rate overall |
|---|---|---|---|---|
| 452 | 1356 | 523 | 19199 | 38.6% |

## 3. Modifier calibration

| n with predicted_distance | mean predicted_d | n with confidence | mean confidence | n with multiplier | mean multiplier |
|---|---|---|---|---|---|
| 1356 | 2.434 | 1356 | 0.528 | 1356 | 0.556 |

## 4. Plots

- ![](plots/cache_savings_per_task.png)
- ![](plots/hit_rate_per_agent_pair.png)
- ![](plots/modifier_calibration.png)
