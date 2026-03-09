# Issue 10492 Submission Checklist

This file records how to stage and cherry-pick the branch contents for issue `#10492`.

## Commit A: Core Fix and Regression Coverage

These files are the actual product change and should be the default PR payload.

| File | What Changed | Why It Changed | Note |
|---|---|---|---|
| `python/sglang/__init__.py` | Replaced eager public imports with lazy exports. | Prevent `import sglang` from loading frontend APIs and their dependencies at import time. | Largest root-package win. |
| `python/sglang/srt/configs/__init__.py` | Made config exports lazy. | Stop config package import from pulling model config internals too early. | Helps `ModelConfig` import path. |
| `python/sglang/srt/configs/model_config.py` | Deferred heavy helper imports and moved type-only imports behind `TYPE_CHECKING`. | Shorten the `ModelConfig` import chain. | Major measured improvement. |
| `python/sglang/srt/parser/reasoning_parser.py` | Removed the top-level dependency on OpenAI protocol request types. | Keep parser imports lightweight. | Explains the large `ReasoningParser` speedup. |
| `python/sglang/srt/layers/moe/__init__.py` | Switched MoE package exports to lazy resolution. | Avoid `scheduler -> moe -> runner` import-time coupling. | Keeps `utils` path light. |
| `python/sglang/srt/layers/moe/moe_runner/__init__.py` | Switched runner exports to lazy resolution. | Preserve compatibility without eager-loading runner implementations. | Avoids deep-gemm side effects. |
| `python/sglang/srt/layers/quantization/__init__.py` | Replaced eager quantization registry import with a lazy registry. | Prevent quantization package import from loading all backends up front. | Key to the config and scheduler cleanup. |
| `python/sglang/srt/layers/quantization/fp8_config.py` | Added a new lightweight FP8 config module. | Let scheduler code read FP8 config state without importing the full FP8 runtime stack. | New file. |
| `python/sglang/srt/layers/quantization/fp8_utils.py` | Moved config state out and kept compatibility re-exports. | Shorten scheduler-side imports without breaking callers. | Compatibility layer. |
| `python/sglang/srt/layers/quantization/fp8_kernel.py` | Deferred `deep_gemm_wrapper` access until runtime. | Eliminate import-time DeepGEMM side effects. | Keeps runtime behavior intact. |
| `python/sglang/srt/layers/logits_processor.py` | Removed embedded output dataclass and imported it from a lightweight module. | Break a heavy manager-side import chain. | Paired with the new output file. |
| `python/sglang/srt/layers/logits_processor_output.py` | Added a lightweight output dataclass module. | Give manager utilities a small import target. | New file. |
| `python/sglang/srt/managers/utils.py` | Imported the lightweight logits output type instead of the full logits processor. | Avoid pulling logits implementation into scheduler-adjacent utilities. | Reduces hidden coupling. |
| `python/sglang/srt/managers/scheduler.py` | Redirected imports to lightweight MoE and FP8 helpers and localized several heavy imports. | This is the main issue fix for `Scheduler` import time. | Main issue file. |
| `python/sglang/srt/managers/scheduler_dp_attn_mixin.py` | Localized the DP attention preparer import. | Avoid importing extra scheduler subpaths during module load. | Small but targeted cleanup. |
| `python/sglang/srt/managers/scheduler_output_processor_mixin.py` | Moved a heavy type import behind `TYPE_CHECKING`. | Remove unnecessary runtime imports. | Pure import cleanup. |
| `python/sglang/srt/lora/lora_overlap_loader.py` | Moved `LoRAManager` to `TYPE_CHECKING`. | Stop type-only references from importing runtime code. | Pure import cleanup. |
| `python/sglang/srt/server_args.py` | Switched parser and GGUF helpers to lazy resolution. | Prevent `ServerArgs` import from pulling parser and GGUF helpers immediately. | Helps `Engine` and `Scheduler`. |
| `test/unit/test_scheduler_import_regression.py` | Added focused import-regression tests. | Lock in the bug fix and prevent future eager-import regressions. | Must ship with the code fix. |

## Commit B: Investigation and Execution Record

These files explain what was tested, what was blocked by the local GPU, and why the fix is shaped this way. They are useful for reviewer context and for later cherry-picks, but they are not required for the code to work.

| File | What It Contains | Why Keep It | Note |
|---|---|---|---|
| `issue-fix-plan.md` | High-level fix plan, container assumptions, and final scope. | Shows the intended fix boundaries. | Synced to final state. |
| `issue-fix-progress.md` | Atomic execution log for investigation, implementation, validation, and submission prep. | Preserves the step-by-step record. | Includes the submission checklist summary. |
| `issue-test-plan.md` | Layered test strategy, including local-model constraints and performance phase design. | Explains the test matrix and fallback policy. | Useful if tests are rerun later. |
| `issue-test-progress.md` | Actual correctness and integration test outcomes, including blocked cases and fallbacks. | Records what passed locally and why some cases were skipped or blocked. | Important for local reproducibility. |
| `issue-submission-checklist.md` | This staging guide and file-by-file rationale. | Makes cherry-pick decisions explicit. | New file. |

## Commit C: Reproducibility Artifacts

These files preserve the benchmark harness and compact raw timing data so the before/after comparison can be rerun or reanalyzed later. They should usually stay out of the main PR unless reviewers explicitly want the artifacts in-tree.

| File | What It Contains | Why Keep It | Note |
|---|---|---|---|
| `issue-perf-summary.md` | Human-readable import performance summary for `main` vs the fix branch. | Captures the final benchmark conclusions. | References the committed compact JSON data. |
| `test/manual/issue_10492/README.md` | How to rerun the container benchmark and how to interpret the preserved artifacts. | Gives a stable replay path. | Manual-only documentation. |
| `test/manual/issue_10492/bench_imports.py` | Import benchmark harness used for the 10-run measurements. | Reusable for reruns on other hardware or containers. | Promoted out of `.codex-tmp`. |
| `test/manual/issue_10492/perf-results/main-10runs.json` | Compact raw timing data for the `main` baseline. | Preserves the original measured samples. | Contains per-run `real`, `user`, and `sys`. |
| `test/manual/issue_10492/perf-results/fix-10runs.json` | Compact raw timing data for the fix branch. | Preserves the post-fix measured samples. | Contains per-run `real`, `user`, and `sys`. |

## Keep Local Only

These artifacts are worth keeping on the branch or workstation, but they should not be part of a normal PR payload.

| Path | Why Exclude It | Note |
|---|---|---|
| `.codex-tmp/main-baseline-20260308-173917/` | Full repo snapshot used to benchmark `main`; too large and redundant for git history. | Keep locally if more comparisons are needed. |
| `.codex-tmp/perf-results/*-samples/` | Full `importtime` stderr logs for every run; useful for forensic work, but very noisy. | Keep locally for deep analysis. |
| `.codex-tmp/main-dryrun.json` and `main-dryrun-samples/` | Trial run artifacts used to validate the harness. | Superseded by the 10-run data. |
| `.codex-tmp/issue10492_opt125m_shim.py` | Local fallback helper used during manual integration testing. | Investigation-only. |
| `.codex-tmp/issue10492_qwen15b_shim.py` | Local fallback helper used during manual integration testing. | Investigation-only. |
| `.codex-tmp/issue10492_smolllm_shim.py` | Local fallback helper used during manual integration testing. | Investigation-only. |
| `.codex-tmp/issue10492_srt_engine_opt_fallback.py` | Local fallback helper used during manual integration testing. | Investigation-only. |
| `AGENTS.md` | Repository guide created earlier, but unrelated to issue `#10492`. | Keep it in a separate docs-only commit and cherry-pick only if you want the contributor guide in the target branch. |
