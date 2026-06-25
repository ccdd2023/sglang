# Graph-Aware Strict28 Patch-Harness Report, 2026-06-17

This report adds graph-aware reuse to the same 28 SWE cases used by the
corrected `strictd20` selective-AST presentation. It is a patch-harness run, not
the same token-F1-vs-lossless driver as `selective_ast_reuse`.

## Setup

- Dataset: `results/selective_ast_reuse/data/swe_selective_wholefile_68k_1file_instances.json`
- Cases requested: 28
- Graph policy: `call_neighborhood_1hop`
- Graph manifest: `results/code_graph_kv_reuse/strict28_graph_data_20260617/code_graph_precision_manifest.jsonl`
- Generation/apply run: `results/code_graph_kv_reuse/strict28_graph_aware_skiptest_nostream_20260617`
- TTFT run: `results/code_graph_kv_reuse/strict28_graph_aware_skiptest_20260617`
- Candidate tests: skipped

## Coverage

The graph analyzer produced `call_neighborhood_1hop` bundles for 24/28 cases.
The four skipped cases had patch targets that the current static Python AST
analyzer did not map to a function/class symbol:

- `pytest-dev__pytest-7521`
- `pytest-dev__pytest-7571`
- `pytest-dev__pytest-7982`
- `pytest-dev__pytest-8399`

## Main Results

Generation/apply metrics from the non-streaming run:

| mode | n/28 | elapsed ms | cached | exact sig | synth ok | apply ok | search miss | json parse fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lossless` | 24/28 | 1151.7 | 5638.1 | 0/24 | 12/24 | 12/24 | 9/24 | 3/24 |
| `lossy` | 24/28 | 1268.4 | 6145.0 | 24/24 | 13/24 | 13/24 | 7/24 | 4/24 |
| `lossy_prefetch` | 24/28 | 1423.9 | 6146.0 | 24/24 | 13/24 | 13/24 | 9/24 | 2/24 |
| `graph_aware_lossy` | 24/28 | 1031.0 | 3410.0 | 24/24 | 15/24 | 15/24 | 5/24 | 4/24 |

TTFT metrics from the streaming run:

| mode | n/28 | TTFT ms | elapsed ms | cached | exact sig |
|---|---:|---:|---:|---:|---:|
| `lossless` | 24/28 | 82.1 | 1138.4 | 5832.2 | 0/24 |
| `lossy` | 24/28 | 125.8 | 1342.9 | 5415.4 | 24/24 |
| `lossy_prefetch` | 24/28 | 96.2 | 1488.5 | 5692.3 | 24/24 |
| `graph_aware_lossy` | 24/28 | 94.1 | 1026.7 | 3430.3 | 24/24 |

## Interpretation

- `graph_aware_lossy` is connected to the runtime cache path: exact content
  signature matched on all covered rows (`24/24`).
- It improves patch-harness apply coverage on this strict28 subset:
  `15/24` apply-ok vs `12/24` lossless and `13/24` generic lossy.
- It reuses fewer cached tokens than full-file lossy modes because the graph
  bundle is smaller and relation-selected.
- It is not yet a direct row in the selective-AST TTFT/F1 table because that
  table uses a different driver and reports token F1 vs lossless rather than
  JSON-edit apply checks.

## Case: `psf__requests-5414`

Problem: `http://.example.com` raises a lower-level UnicodeError; the expected
behavior is an `InvalidURL`.

| mode | segment source | segment shape | cached | elapsed | apply |
|---|---|---|---:|---:|---:|
| `lossless` | file context | `requests/models.py`, 973 lines | 8458 | 946.7ms | ok |
| `lossy` | file context | same whole file context | 8459 | 954.8ms | ok |
| `lossy_prefetch` | file context | same whole file context + hints | 8460 | 956.1ms | ok |
| `graph_aware_lossy` | code graph bundle | 2 bundles, 176 + 613 lines | 6945 | 929.0ms | ok |

The graph bundle is not simply "the AST method" and not the whole file. The
main bundle centers on `PreparedRequest.prepare_url` and pulls in direct
callee/import-neighborhood evidence such as:

- `PreparedRequest.prepare_url`
- `PreparedRequest._get_idna_encoded_host`
- `RequestEncodingMixin._encode_params`
- `requote_uri`
- `to_native_string`
- `unicode_is_ascii`
- `InvalidURL`
- `MissingSchema`

This is the key distinction:

| reuse style | what it assumes reusable | why it differs |
|---|---|---|
| whole-file | all of `requests/models.py` | fastest/strongest assumption, brittle to local edits |
| extended AST | selected spans such as file_prefix/method/control_block | syntax-local and safe, but not dependency-aware |
| graph-aware | target symbol plus direct call/import neighborhood | program-relation-aware; smaller than whole file but richer than one AST span |

The current graph-aware result should be presented as a relation-aware
selection result with live runtime evidence, not yet as the final replacement
for the corrected selective-AST TTFT/F1 table.
