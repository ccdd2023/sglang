# Issue #10492 Fix Plan

## Summary

Goal: reduce the import cost of:

```bash
python -X importtime -c "from sglang.srt.managers.scheduler import Scheduler"
```

The fix focuses on removing unnecessary import-time side effects in the `Scheduler` import path, not on redesigning the entire `import sglang` architecture. All development, reproduction, and validation happen inside the existing dev container. The main reproduction path does not require starting the SGLang server.

This plan is synchronized with `issue-fix-progress.md`.

Execution rule for this task:

- Work only inside the existing dev container.
- Execute the plan one atomic step at a time.
- After each completed step, report the observed effect and test result.
- Do not move to the next step until the user confirms the current result.

## Container Workflow

- Use the existing dev container as the only execution environment.
- Mount the current repo into `/sgl-workspace/sglang`.
- Run all investigation, tests, and validation commands inside the container.
- Do not use the host machine as the Python runtime for this issue.
- Do not start `sglang.launch_server` for the main reproduction. This issue is about import behavior, not server startup.

### Current Environment Note

- The official `lmsysorg/sglang:dev` image is currently blocked on this machine by a persistent Docker layer extraction failure (`archive/tar: invalid tar header`) during pull.
- Until that host-level Docker issue is repaired, the active fallback development container is the local `minisgl:dev` image with the repo bind-mounted to `/sgl-workspace/sglang`.
- Validation still remains container-only and repo-mounted.
- The fallback container has working GPU access only when started with explicit `--runtime nvidia --gpus all`.
- The fallback container uses a task-local venv at `/tmp/sglang-issue10492-venv`.
- The venv has been bootstrapped with the repo in editable mode plus the core packages needed for the import-time reproduction path, including `torch`, `numpy`, `psutil`, `pyzmq`, and `setproctitle`.

### Reproduction Boundary

The local machine has a GeForce 2080 Super, not H100 or H200. That means:

- We can reproduce the slow import path and the heavy module import graph.
- We cannot require Hopper-only DeepGEMM symptoms to reproduce locally.
- In particular, the exact H100/H200 logs about DeepGEMM loading or `torch.utils.cpp_extension` may not appear on this machine because DeepGEMM is gated on `SM90+`.

## Execution Stages

### 1. Container Baseline

- Start or attach to the existing dev container.
- Verify the repo bind mount at `/sgl-workspace/sglang`.
- Verify GPU visibility in the container.
- Verify Python resolves imports from the mounted repo.

### 2. Reproduce and Measure the Problem

- Capture the baseline import timing with:
  ```bash
  time python -X importtime -c "from sglang.srt.managers.scheduler import Scheduler" 2> import_sglang.log
  ```
- Extract the highest-cost import hotspots from the importtime log.
- Capture transitive import side effects using subprocess probes and `sys.modules`.
- Current local baseline after bootstrapping the fallback container:
  - `real 7.56`
  - `user 9.39`
  - `sys 1.38`
- Use this baseline to confirm the main unnecessary import chains:
  - `sglang.__init__`
  - `model_config -> hf_transformers_utils / quantization`
  - `reasoning_parser -> openai.protocol`
  - `scheduler -> moe`
  - `scheduler -> fp8`
  - `scheduler -> server_args / tokenizer helpers`
  - `scheduler_pp_mixin -> managers.utils -> logits_processor`

### 3. Apply the Targeted Fixes

- Make `python/sglang/__init__.py` lazy while preserving public names.
- Make `python/sglang/srt/configs/model_config.py` defer `transformers` and HF helper imports until runtime use.
- Make `python/sglang/srt/configs/__init__.py` lazy so config package import does not pull model-specific configs.
- Make `python/sglang/srt/parser/reasoning_parser.py` avoid importing OpenAI protocol types at module import time.
- Change `scheduler.py` to import `initialize_moe_config` from `sglang.srt.layers.moe.utils`.
- Make `python/sglang/srt/layers/moe/__init__.py` lazy while preserving compatibility.
- Make `python/sglang/srt/layers/moe/moe_runner/__init__.py` lazy so config-only paths do not load runner implementations.
- Make `python/sglang/srt/layers/quantization/__init__.py` use a lazy quantization registry so config classes are imported on demand.
- Split FP8 config symbols into a lightweight config module.
- Rewire `scheduler.py` to use the lightweight FP8 config layer.
- Keep `fp8_utils.py` backward compatible via re-export.
- Split `LogitsProcessorOutput` into a lightweight module so `managers.utils` can avoid importing full `logits_processor.py`.
- Defer scheduler helper imports for:
  - multimodal helpers
  - tokenizer helpers
  - encode receiver creation
  - DP attention preparers
  - LoRA type-only helpers
  - server-args parser helpers
- Ensure DeepGEMM loading remains runtime-only, not import-time.
- Add regression tests for the remaining high-impact import paths:
  - `ModelConfig` must not load `transformers`, `compressed_tensors`, or `hf_transformers_utils`
  - `ReasoningParser` must not load `openai` or `sglang.srt.entrypoints.openai.protocol`
  - `ServerArgs` must not load `openai`, `transformers`, or `hf_transformers_utils`
  - `configs` package must not eagerly load model-specific config modules like `deepseekvl2` or `exaone`

### 4. Re-Measure and Validate

- Re-run the baseline import timing command and compare before vs. after.
- Re-run subprocess `sys.modules` probes to confirm side-effect reduction.
- Verify compatibility imports still work:
  - `from sglang.srt.layers.moe import MoeRunner, MoeRunnerConfig, initialize_moe_config`
  - `from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config`
- Run a minimal container-only regression suite as a guard:
  - `PYTHONPATH=python python3 -m unittest discover -s test/unit -p "test_scheduler_import_regression.py" -v`

### Current Validation Result

- `Scheduler` import timing improved from:
  - `real 7.56 / user 9.39 / sys 1.38`
  - to `real 4.51 / user 5.27 / sys 0.84`
- After the fix, plain `from sglang.srt.managers.scheduler import Scheduler` no longer imports:
  - `sglang.lang.api`
  - `transformers`
  - `openai`
  - `compressed_tensors`
  - `flashinfer`
  - `sglang.srt.layers.moe.moe_runner.deep_gemm`
  - `sglang.srt.layers.deep_gemm_wrapper.entrypoint`
  - `torch.utils.cpp_extension`
- Added regression coverage in:
  - `test/unit/test_scheduler_import_regression.py` (8 tests total: 4 existing + 4 new)

### 5. Finalize

- Update progress records in `issue-fix-progress.md`.
- Summarize what was reproduced locally in the dev container.
- Explicitly note what remains Hopper-specific and therefore not strictly reproducible on the local 2080 Super.
- Curate the submission payload before commit:
  - include the import-fix code and regression test as the default PR payload
  - keep the issue notes and execution record in a separate documentation commit for optional cherry-picking
  - promote only the reusable benchmark harness and compact raw timing JSON into a permanent repo location if reproducibility artifacts should be preserved
  - exclude temporary harnesses, baseline snapshots, perf sample directories, and other `.codex-tmp/` artifacts unless they are deliberately promoted into a permanent repo location
  - keep unrelated documentation such as `AGENTS.md` in a separate docs-only commit if the branch is intended to stay focused on issue `#10492`

## Deliverables

- The code change set
- A synchronized atomic execution record in `issue-fix-progress.md`
- Container-only reproduction steps
- Container-only validation commands
- A short explanation of the local GPU limitation versus H100/H200 behavior
- A final verification note describing what was reproduced locally and what remains Hopper-specific
- A curated commit payload that separates:
  - Commit A: issue-relevant source changes and regression coverage
  - Commit B: issue notes, test records, and submission guidance
  - Commit C: optional reproducibility artifacts such as the promoted benchmark harness and compact raw timing data
  - non-committable temporary artifacts that stay local under `.codex-tmp/`

## Assumptions

- The existing dev container is available and usable for iterative development.
- The repo is bind-mounted into the container and editable there.
- The local 2080 Super can validate import-graph improvements but cannot be used as a strict reproduction target for Hopper-only DeepGEMM behavior.
- The fix only aims to remove avoidable import-time side effects, not to eliminate the baseline import cost of `torch` itself.
