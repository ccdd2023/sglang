# Per-layer cross-role K drift (planner-vs-coder, RoPE-aligned)

Source: layerwise_ast_granularity_comparison.csv, correct_delta variant.

Qwen2.5-Coder-7B-Instruct, 28 layers aggregated into 5 layer bins.

Two metrics: mean k_cosine (1.0 = identical direction) and mean k_l2_norm (per-element RMS of K diff).

Hypothesis: cross-role K direction is near-perfect at every layer, but the *contrast* between roles is concentrated in the late layers; the last-4-layers cut is the smallest layer set that preserves the cross-role signal.


| Granularity | bin | n | mean k_cos | min k_cos | mean k_l2 | max k_l2 |
|---|---|---:|---:|---:|---:|---:|
| function | early [0-6] | 35 | 0.9999 | 0.9998 | 0.0277 | 0.0428 |
| function | mid-1 [7-13] | 35 | 0.9989 | 0.9972 | 0.0893 | 0.1673 |
| function | mid-2 [14-20] | 35 | 0.9953 | 0.9924 | 0.2040 | 0.2639 |
| function | late [21-27] | 35 | 0.9954 | 0.9912 | 0.1940 | 0.2349 |
| function | last-4 [24-27] | 20 | 0.9960 | 0.9912 | 0.1834 | 0.2291 |
| method | early [0-6] | 35 | 0.9999 | 0.9996 | 0.0315 | 0.0465 |
| method | mid-1 [7-13] | 35 | 0.9990 | 0.9975 | 0.0844 | 0.1570 |
| method | mid-2 [14-20] | 35 | 0.9959 | 0.9934 | 0.1894 | 0.2399 |
| method | late [21-27] | 35 | 0.9960 | 0.9925 | 0.1714 | 0.2011 |
| method | last-4 [24-27] | 20 | 0.9966 | 0.9925 | 0.1612 | 0.1978 |
| class | early [0-6] | 35 | 0.9999 | 0.9997 | 0.0293 | 0.0427 |
| class | mid-1 [7-13] | 35 | 0.9980 | 0.9923 | 0.1174 | 0.2553 |
| class | mid-2 [14-20] | 35 | 0.9921 | 0.9817 | 0.2611 | 0.3891 |
| class | late [21-27] | 35 | 0.9909 | 0.9763 | 0.2672 | 0.3769 |
| class | last-4 [24-27] | 20 | 0.9915 | 0.9763 | 0.2613 | 0.3680 |
| control_block | early [0-6] | 35 | 0.9999 | 0.9997 | 0.0298 | 0.0477 |
| control_block | mid-1 [7-13] | 35 | 0.9983 | 0.9931 | 0.1063 | 0.2412 |
| control_block | mid-2 [14-20] | 35 | 0.9950 | 0.9896 | 0.2096 | 0.2768 |
| control_block | late [21-27] | 35 | 0.9950 | 0.9906 | 0.1938 | 0.2500 |
| control_block | last-4 [24-27] | 20 | 0.9956 | 0.9906 | 0.1838 | 0.2187 |
| statement_window | early [0-6] | 35 | 0.9999 | 0.9994 | 0.0340 | 0.0641 |
| statement_window | mid-1 [7-13] | 35 | 0.9978 | 0.9910 | 0.1219 | 0.2886 |
| statement_window | mid-2 [14-20] | 35 | 0.9934 | 0.9836 | 0.2339 | 0.3508 |
| statement_window | late [21-27] | 35 | 0.9931 | 0.9810 | 0.2193 | 0.3378 |
| statement_window | last-4 [24-27] | 20 | 0.9937 | 0.9810 | 0.2104 | 0.3172 |
| file_prefix | early [0-6] | 35 | 1.0000 | 0.9998 | 0.0248 | 0.0374 |
| file_prefix | mid-1 [7-13] | 35 | 0.9995 | 0.9985 | 0.0600 | 0.1290 |
| file_prefix | mid-2 [14-20] | 35 | 0.9975 | 0.9938 | 0.1507 | 0.2381 |
| file_prefix | late [21-27] | 35 | 0.9972 | 0.9912 | 0.1580 | 0.2393 |
| file_prefix | last-4 [24-27] | 20 | 0.9975 | 0.9912 | 0.1534 | 0.2393 |
