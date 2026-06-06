# Multi-Agent Intermediate-Context KV Reuse — TTFT Acceleration

Model: Qwen2.5-3B | 6 files | 3-Agent workflow

## Summary

### A1

| File | No-Reuse TTFT | Full-Reuse TTFT | Lossy-Reuse TTFT | No-Reuse Total | Full-Reuse Total | Lossy-Reuse Total | Full Cache | Lossy Cache | BLEU Full | BLEU Lossy |
|---|---|---|---|---|---|---|---|---|---|---|
| same-func (AVL insert) | 79 | 76 | 77 | 1253 | 1258 | 1249 | 0 | 0 | 1.000 | 1.000 |
| same-func (RedBlack insert) | 96 | 102 | 100 | 1396 | 1402 | 1401 | 0 | 0 | 1.000 | 1.000 |
| same-func (merge sort) | 57 | 57 | 54 | 2097 | 2197 | 2095 | 0 | 0 | 1.000 | 1.000 |
| same-func (heap sort) | 55 | 56 | 58 | 2106 | 2098 | 2098 | 0 | 0 | 1.000 | 1.000 |
| same-func (Dijkstra) | 55 | 56 | 56 | 2094 | 2096 | 2096 | 0 | 0 | 1.000 | 1.000 |
| same-func (BFS) | 51 | 54 | 53 | 2090 | 2110 | 2108 | 0 | 0 | 1.000 | 1.000 |
| **Avg** | 66 | 67 | 66 | 1839 | 1860 | 1841 | 0 | 0 | - | - |

### A2

| File | No-Reuse TTFT | Full-Reuse TTFT | Lossy-Reuse TTFT | No-Reuse Total | Full-Reuse Total | Lossy-Reuse Total | Full Cache | Lossy Cache | BLEU Full | BLEU Lossy |
|---|---|---|---|---|---|---|---|---|---|---|
| same-func (AVL insert) | 87 | 74 | 76 | 2130 | 2113 | 2115 | 25 | 25 | 1.000 | 1.000 |
| same-func (RedBlack insert) | 115 | 92 | 94 | 2176 | 2173 | 2135 | 25 | 25 | 1.000 | 1.000 |
| same-func (merge sort) | 69 | 56 | 66 | 2114 | 2096 | 2180 | 25 | 25 | 1.000 | 1.000 |
| same-func (heap sort) | 69 | 56 | 58 | 2110 | 2097 | 2098 | 25 | 25 | 1.000 | 1.000 |
| same-func (Dijkstra) | 67 | 55 | 56 | 2108 | 2097 | 2134 | 25 | 25 | 1.000 | 1.000 |
| same-func (BFS) | 65 | 52 | 53 | 2104 | 2092 | 2094 | 25 | 25 | 1.000 | 1.000 |
| **Avg** | 79 | 64 | 67 | 2124 | 2111 | 2126 | 25 | 25 | - | - |

### A3

| File | No-Reuse TTFT | Full-Reuse TTFT | Lossy-Reuse TTFT | No-Reuse Total | Full-Reuse Total | Lossy-Reuse Total | Full Cache | Lossy Cache | BLEU Full | BLEU Lossy |
|---|---|---|---|---|---|---|---|---|---|---|
| same-func (AVL insert) | 98 | 85 | 89 | 179 | 166 | 170 | 25 | 25 | 1.000 | 1.000 |
| same-func (RedBlack insert) | 122 | 101 | 106 | 235 | 213 | 225 | 25 | 25 | 1.000 | 1.000 |
| same-func (merge sort) | 80 | 66 | 70 | 196 | 179 | 182 | 25 | 25 | 1.000 | 1.000 |
| same-func (heap sort) | 81 | 69 | 69 | 675 | 712 | 664 | 25 | 25 | 1.000 | 1.000 |
| same-func (Dijkstra) | 81 | 63 | 66 | 194 | 175 | 179 | 25 | 25 | 1.000 | 1.000 |
| same-func (BFS) | 86 | 62 | 65 | 168 | 143 | 152 | 25 | 25 | 1.000 | 1.000 |
| **Avg** | 91 | 74 | 78 | 275 | 265 | 262 | 25 | 25 | - | - |

## TTFT Speedup

| Agent | Full-Reuse Speedup | Lossy-Reuse Speedup |
|---|---|---|
| A1 | -2.1% | -1.3% |
| A2 | +22.4% | +17.2% |
| A3 | +23.0% | +17.8% |

