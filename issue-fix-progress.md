# Issue #10492 Progress

This file expands `issue-fix-plan.md` into atomic execution steps.

Execution rule for this task:

- Work only inside the existing dev container.
- After each step is completed, report the observed effect and test result.
- Do not continue to the next step until the user confirms the result.

Status values:

- `pending`
- `in_progress`
- `completed`
- `blocked`

## 1. Container Baseline

### 1.1 Start or attach to the existing dev container

- Status: `completed`
- Atomic operation:
  - Check whether the existing dev container is already running.
  - If not, start it using the repository's existing dev-container workflow.
- Expected effect:
  - A running dev container exists and can accept shell commands.
- Verification:
  - Confirm the container is visible in `docker ps`.
  - Confirm `docker exec ... pwd` works.
- User confirmation gate:
  - User confirms that the correct dev container is being used.
- Current blocker:
  - Initial `docker run` failed while pulling `lmsysorg/sglang:dev` with `archive/tar: invalid tar header` during layer extraction.
  - A separate `docker pull lmsysorg/sglang:dev` retry failed again with `invalid tar header`, including on a different layer.
  - To unblock progress, step 1.1 was completed with the existing local `minisgl:dev` image as a fallback dev container.
- Result:
  - Started container: `sglang_dev_issue10492`
  - Image: `minisgl:dev`
  - Verified running state via `docker ps`
  - Verified command execution via `docker exec ... pwd`
  - Verified working directory resolves to `/sgl-workspace/sglang`

### 1.2 Verify repo bind mount inside the container

- Status: `completed`
- Atomic operation:
  - Enter the container and verify that the repo is mounted at `/sgl-workspace/sglang`.
- Expected effect:
  - Container sees the same working tree as the host repo.
- Verification:
  - Run `pwd` and `ls` in the container.
  - Confirm `issue-fix-plan.md` and `issue-fix-progress.md` are visible there.
- User confirmation gate:
  - User confirms the container is using the attached repo rather than a stale copy.
- Result:
  - `pwd` inside the container is `/sgl-workspace/sglang`
  - Repository root contents are visible from inside the container
  - `issue-fix-plan.md` and `issue-fix-progress.md` are both present in the mounted tree

### 1.3 Verify GPU visibility in the container

- Status: `completed`
- Atomic operation:
  - Check that the container can see the local NVIDIA GPU.
- Expected effect:
  - `nvidia-smi` works in the container and reports the 2080 Super.
- Verification:
  - Run `nvidia-smi`.
- User confirmation gate:
  - User confirms GPU access is sufficient for local reproduction and validation.
- Result:
  - Plain `--gpus all` was not sufficient on this host for NVML access inside containers.
  - A minimal CUDA container succeeded only when run with explicit `--runtime nvidia --gpus all`.
  - The active fallback dev container was recreated with explicit `--runtime nvidia`.
  - Verified from inside the container:
    - `NVIDIA GeForce RTX 2080 SUPER, 8192 MiB, 7.5`
- Notes:
  - Earlier Python/Torch probing in the fallback image showed that `torch` is not currently installed there.
  - GPU visibility is now confirmed; Python package readiness will be handled separately from this step.

### 1.4 Verify Python import environment in the container

- Status: `completed`
- Atomic operation:
  - Confirm that Python resolves imports from the mounted repo.
- Expected effect:
  - `sglang` imports from `/sgl-workspace/sglang/python`.
- Verification:
  - Run a small Python command to print `sglang.__file__` or the resolved module path.
- User confirmation gate:
  - User confirms the container is testing the live source tree.
- Result:
  - Python is available in the container: `Python 3.12.3`
  - A dedicated task venv was created at `/tmp/sglang-issue10492-venv`
  - The mounted repo is reachable from Python when using `PYTHONPATH=python`
  - `import sglang` resolves to:
    - `/sgl-workspace/sglang/python/sglang/__init__.py`
  - The key dependencies needed for the reproduction path are now installed in the venv:
    - `torch`
    - `numpy`
    - `psutil`
    - `zmq`
    - `setproctitle`
  - The fallback container is now ready for the import-time reproduction steps.

## 2. Reproduce and Measure the Problem

### 2.1 Capture the baseline `Scheduler` import timing

- Status: `completed`
- Atomic operation:
  - Run:
    ```bash
    time python -X importtime -c "from sglang.srt.managers.scheduler import Scheduler" 2> import_sglang.log
    ```
- Expected effect:
  - A baseline import timing and full import graph log are captured inside the container.
- Verification:
  - Confirm `import_sglang.log` exists.
  - Record total wall-clock time from `time`.
- User confirmation gate:
  - User confirms the baseline run is the correct starting point.
- Result:
  - The first baseline attempt failed because `sgl_kernel` could not load `libnuma.so.1`.
  - The missing system library was installed in the container with `apt-get install -y libnuma1`.
  - The rerun completed successfully inside the container.
  - Captured files:
    - `import_sglang.log`
    - `import_sglang.time`
  - Baseline wall-clock time:
    - `real 7.56`
    - `user 9.39`
    - `sys 1.38`

### 2.2 Identify the top import hotspots from `import_sglang.log`

- Status: `completed`
- Atomic operation:
  - Extract the highest cumulative import-cost modules from the importtime log.
- Expected effect:
  - We have a ranked list of the heaviest import paths for this container environment.
- Verification:
  - Summarize the top modules and their cumulative cost.
- User confirmation gate:
  - User confirms the baseline analysis is sufficient to proceed with fixes.
- Result:
  - The top cumulative hotspots in the container baseline are:
    - `sglang.srt.managers.scheduler`: `6270553`
    - `sglang.srt.configs.model_config`: `4341319`
    - `sglang.srt.configs`: `2332226`
    - `sglang.srt.layers.quantization`: `2008508`
    - `sglang.srt.configs.deepseekvl2`: `1569199`
    - `sglang.srt.layers.quantization.awq`: `1387272`
    - `torch`: `1024021`
    - `sglang.srt.layers.moe`: `897247`
    - `sglang.srt.layers.moe.moe_runner`: `896965`
    - `sglang.srt.layers.moe.moe_runner.deep_gemm`: `887550`
  - The baseline confirms that the import path is not dominated by one issue alone.
  - Besides torch, the heaviest avoidable work is in:
    - `model_config -> hf_transformers_utils / configs / quantization`
    - `moe -> moe_runner -> deep_gemm`

### 2.3 Capture transitive import side effects with `sys.modules`

- Status: `completed`
- Atomic operation:
  - Run subprocess-based probes to check whether heavy modules are imported by the plain `Scheduler` import path.
- Expected effect:
  - We know which heavy modules are brought in transitively on this machine.
- Verification:
  - Check for entries such as:
    - `deep_gemm`
    - `torch.utils.cpp_extension`
    - `transformers`
    - `openai`
    - `compressed_tensors`
- User confirmation gate:
  - User confirms the side-effect baseline before code changes begin.
- Result:
  - `from sglang.srt.managers.scheduler import Scheduler` loads these heavy modules on the local 2080 Super baseline:
    - `sglang.srt.layers.moe.moe_runner.deep_gemm`: `True`
    - `transformers`: `True`
    - `openai`: `True`
    - `compressed_tensors`: `True`
    - `flashinfer`: `True`
    - `torch.utils.cpp_extension`: `False`
    - `deep_gemm`: `False`
  - Isolated per-module probes identified the concrete sources:
    - `import sglang` eagerly loads `sglang.lang.api`
    - `from sglang.srt.configs.model_config import ModelConfig` loads `transformers`, `compressed_tensors`, and `sglang.srt.layers.quantization`
    - `from sglang.srt.layers.moe.utils import initialize_moe_config` still loads `sglang.srt.layers.moe.moe_runner.deep_gemm`
    - `from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config` loads `sglang.srt.layers.deep_gemm_wrapper.entrypoint`
    - `from sglang.srt.parser.reasoning_parser import ReasoningParser` loads `openai` through `sglang.srt.entrypoints.openai.protocol`

## 3. Apply the Targeted Fixes

### 3.1 Make `python/sglang/__init__.py` lazily resolve exports

- Status: `completed`
- Atomic operation:
  - Replace eager frontend/runtime exports with lazy resolution while preserving public names.
- Expected effect:
  - `from sglang.srt...` no longer eagerly loads unrelated frontend modules through the root package.
- Verification:
  - Re-run a subprocess import probe for `import sglang`.
  - Confirm the frontend stack is no longer eagerly present in `sys.modules`.
- User confirmation gate:
  - User confirms the root-package lazy import behavior is correct.
- Result:
  - `python/sglang/__init__.py` now resolves public exports via module-level `__getattr__`.
  - Verified:
    - `import sglang` no longer loads `sglang.lang.api`
    - `import sglang` no longer loads `openai`
    - `import sglang` no longer loads `transformers`
    - `import sglang` no longer loads `compressed_tensors`
  - Compatibility check passed:
    - `from sglang import gen, ServerArgs`

### 3.2 Make `python/sglang/srt/configs/model_config.py` defer heavy HF and quantization helpers

- Status: `completed`
- Atomic operation:
  - Remove the eager top-level dependency on `hf_transformers_utils` and keep type-only `transformers` usage out of module import time.
- Expected effect:
  - Importing `ModelConfig` no longer forces `transformers`, custom configs, and quantization helpers to load immediately.
- Verification:
  - Re-run a subprocess probe for:
    ```python
    from sglang.srt.configs.model_config import ModelConfig
    ```
  - Confirm the import no longer pulls in `transformers` and `compressed_tensors`.
- User confirmation gate:
  - User confirms the `ModelConfig` import is narrowed correctly.
- Result:
  - `model_config.py` now uses `TYPE_CHECKING` for `PretrainedConfig` and `ServerArgs`.
  - HF helpers and CI checks are now resolved through small late-bound helper functions.
  - `python/sglang/srt/configs/__init__.py` was also converted to lazy exports to stop package-root config imports from pulling model-specific config modules.
  - `python/sglang/srt/layers/quantization/__init__.py` now uses a lazy registry so quantization config classes import on demand.
  - Verified:
    - `from sglang.srt.configs.model_config import ModelConfig` no longer loads `transformers`
    - `from sglang.srt.configs.model_config import ModelConfig` no longer loads `compressed_tensors`
    - `from sglang.srt.configs.model_config import ModelConfig` no longer loads `sglang.srt.utils.hf_transformers_utils`
    - `from sglang.srt.configs.model_config import ModelConfig` no longer loads `sglang.srt.configs.deepseekvl2`

### 3.3 Make `python/sglang/srt/parser/reasoning_parser.py` avoid importing OpenAI protocol at module import time

- Status: `completed`
- Atomic operation:
  - Replace the top-level `ChatCompletionRequest` import with a type-only or late-bound path.
- Expected effect:
  - Importing `ReasoningParser` no longer pulls in `openai` and `sglang.srt.entrypoints.openai.protocol`.
- Verification:
  - Re-run a subprocess probe for:
    ```python
    from sglang.srt.parser.reasoning_parser import ReasoningParser
    ```
  - Confirm `openai` is absent from `sys.modules`.
- User confirmation gate:
  - User confirms the reasoning parser import no longer drags in OpenAI protocol types.
- Result:
  - `reasoning_parser.py` now uses postponed annotations and a duck-typed helper instead of runtime `ChatCompletionRequest` imports.
  - Verified:
    - `from sglang.srt.parser.reasoning_parser import ReasoningParser` no longer loads `openai`
    - `from sglang.srt.parser.reasoning_parser import ReasoningParser` no longer loads `sglang.srt.entrypoints.openai.protocol`

### 3.4 Change `scheduler.py` to import `initialize_moe_config` from `moe.utils`

- Status: `completed`
- Atomic operation:
  - Remove the dependency on the MoE package root for config-only initialization.
- Expected effect:
  - Importing `Scheduler` no longer needs to traverse MoE runtime runner exports through the package root.
- Verification:
  - Re-run a subprocess probe for:
    ```python
    from sglang.srt.layers.moe.utils import initialize_moe_config
    ```
  - Confirm it does not import `sglang.srt.layers.moe.moe_runner.deep_gemm`.
- User confirmation gate:
  - User confirms the scheduler-to-MoE dependency is narrowed correctly.
- Result:
  - `scheduler.py` now imports `initialize_moe_config` directly from `sglang.srt.layers.moe.utils`.
  - `python/sglang/srt/layers/moe/__init__.py` and `python/sglang/srt/layers/moe/moe_runner/__init__.py` now lazily resolve exports.
  - Verified:
    - `from sglang.srt.layers.moe.utils import initialize_moe_config` no longer loads `sglang.srt.layers.moe.moe_runner`
    - `from sglang.srt.layers.moe.utils import initialize_moe_config` no longer loads `sglang.srt.layers.moe.moe_runner.deep_gemm`
  - Compatibility check passed:
    - `from sglang.srt.layers.moe import MoeRunner, MoeRunnerConfig, initialize_moe_config`

### 3.5 Make quantization and FP8 config paths lazy

- Status: `completed`
- Atomic operation:
  - Replace eager quantization and FP8 config imports with lightweight late-bound modules.
- Expected effect:
  - Plain `Scheduler` import no longer needs quantization runtime trees or DeepGEMM wrappers just to initialize config.
- Verification:
  - Re-run isolated probes for `ModelConfig`, `fp8_config`, and `fp8_utils`.
- User confirmation gate:
  - User confirms the quantization registry remains compatible while staying lazy.
- Result:
  - `python/sglang/srt/layers/quantization/__init__.py` now uses `LazyQuantizationMethods`.
  - Added lightweight `python/sglang/srt/layers/quantization/fp8_config.py`.
  - `fp8_utils.py` now re-exports config-facing FP8 symbols from `fp8_config.py`.
  - `scheduler.py` now imports `initialize_fp8_gemm_config` from `fp8_config.py`.
  - `fp8_kernel.py` now late-loads `deep_gemm_wrapper`.
  - Verified:
    - `from sglang.srt.layers.quantization.fp8_config import initialize_fp8_gemm_config` does not load `deep_gemm_wrapper`
    - `from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config` still works
    - `get_quantization_config("fp8")` still works without loading unrelated backends such as `awq`

### 3.6 Defer remaining scheduler-path helper imports

- Status: `completed`
- Atomic operation:
  - Move type-only and runtime-only helper imports out of module import time across the remaining `Scheduler` dependency chain.
- Expected effect:
  - Plain `Scheduler` import stops dragging in OpenAI, FlashInfer, DeepGEMM, tokenizer, and model-loading helpers through secondary mixins.
- Verification:
  - Re-run isolated probes for the touched modules.
- User confirmation gate:
  - User confirms the remaining scheduler-path imports are acceptably narrow.
- Result:
  - `server_args.py` now late-loads `FunctionCallParser`, `ReasoningParser`, and `check_gguf_file`.
  - `scheduler.py` now late-loads:
    - multimodal helpers from `mm_utils`
    - tokenizer helpers from `hf_transformers_utils`
    - `create_mm_receiver`
  - `scheduler_dp_attn_mixin.py` now late-loads `TboDPAttentionPreparer`.
  - `scheduler_output_processor_mixin.py` moved `LogitsProcessorOutput` to `TYPE_CHECKING`.
  - `lora_overlap_loader.py` moved `LoRAManager` to `TYPE_CHECKING`.
  - Added lightweight `python/sglang/srt/layers/logits_processor_output.py`.
  - `managers/utils.py` now imports `LogitsProcessorOutput` from the lightweight module instead of importing full `logits_processor.py`.
  - Verified isolated probes:
    - `from sglang.srt.server_args import get_global_server_args` no longer loads `openai`, `transformers`, or `hf_transformers_utils`
    - `from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin` no longer loads `flashinfer`, `deep_gemm_wrapper`, or `moe_runner.deep_gemm`
    - `from sglang.srt.managers.utils import GenerationBatchResult` no longer loads `sglang.srt.layers.logits_processor`

### 3.7 Confirm the full `Scheduler` import path is clean

- Status: `completed`
- Atomic operation:
  - Re-run the full `Scheduler` import probe after all targeted fixes.
- Expected effect:
  - Previously avoidable heavy optional modules disappear from `sys.modules`.
- Verification:
  - Re-run:
    ```python
    from sglang.srt.managers.scheduler import Scheduler
    ```
  - Compare the imported module set against the baseline.
- User confirmation gate:
  - User confirms the import path is now materially lighter.
- Result:
  - The post-fix `Scheduler` probe reports:
    - `sglang.lang.api`: `False`
    - `transformers`: `False`
    - `openai`: `False`
    - `compressed_tensors`: `False`
    - `sglang.srt.layers.moe.moe_runner.deep_gemm`: `False`
    - `sglang.srt.layers.deep_gemm_wrapper.entrypoint`: `False`
    - `torch.utils.cpp_extension`: `False`
    - `flashinfer`: `False`
    - `sglang.srt.utils.hf_transformers_utils`: `False`

## 4. Re-Measure and Validate

### 4.1 Re-run the baseline import timing command

- Status: `completed`
- Atomic operation:
  - Re-run:
    ```bash
    time python -X importtime -c "from sglang.srt.managers.scheduler import Scheduler" 2> import_sglang_after.log
    ```
- Expected effect:
  - Import time and import graph improve relative to the baseline.
- Verification:
  - Compare total time and top import hotspots before vs. after.
- User confirmation gate:
  - User confirms the improvement is meaningful enough.
- Result:
  - Post-fix wall-clock time inside the container:
    - `real 4.51`
    - `user 5.27`
    - `sys 0.84`
  - Baseline before fixes:
    - `real 7.56`
    - `user 9.39`
    - `sys 1.38`
  - Importtime hotspot comparison shows the avoidable `model_config`, `configs`, `quantization`, `deepseekvl2`, `moe_runner`, and `deep_gemm` chains are no longer in the top import costs.
  - The remaining dominant cost is mostly `torch` and the unavoidable scheduler core path.

### 4.2 Re-run the `sys.modules` side-effect probe

- Status: `completed`
- Atomic operation:
  - Repeat the transitive import checks after the code changes.
- Expected effect:
  - Unnecessary modules previously imported by plain `Scheduler` import are no longer present.
- Verification:
  - Compare before/after presence of:
    - `deep_gemm`
    - `torch.utils.cpp_extension`
    - `transformers`
    - `openai`
    - `compressed_tensors`
- User confirmation gate:
  - User confirms the side-effect reduction is correct.
- Result:
  - Before fixes, plain `Scheduler` import loaded:
    - `transformers`
    - `openai`
    - `compressed_tensors`
    - `flashinfer`
    - `sglang.srt.layers.moe.moe_runner.deep_gemm`
  - After fixes, all of the above are absent from `sys.modules`.
  - On this local 2080 Super machine, `torch.utils.cpp_extension` was already absent before the fix, so the local result there is neutral rather than before/after.

### 4.3 Run a minimal runtime smoke test in the container

- Status: `completed`
- Atomic operation:
  - Run a minimal smoke test to ensure the import-related changes do not break basic runtime usage.
- Expected effect:
  - The code still behaves correctly outside the narrow import-only path.
- Verification:
  - Use the smallest practical runtime or import smoke relevant to the touched modules.
- User confirmation gate:
  - User confirms runtime behavior is still acceptable.
- Result:
  - Added regression test file:
    - `test/unit/test_scheduler_import_regression.py`
  - The test file covers:
    - root-package lazy import behavior
    - MoE utils import behavior
    - full `Scheduler` import side effects
    - compatibility imports for public APIs
  - Executed in the container with:
    ```bash
    PYTHONPATH=python python3 -m unittest discover -s test/unit -p "test_scheduler_import_regression.py" -v
    ```
  - Result:
    - `Ran 4 tests in 10.656s`
    - `OK`

## 5. Finalize

### 5.1 Update progress records and summarize findings

- Status: `completed`
- Atomic operation:
  - Mark completed steps in this file.
  - Summarize what was reproduced locally and what remains Hopper-specific.
- Expected effect:
  - The repo contains both the high-level plan and the step-by-step execution record.
- Verification:
  - Review `issue-fix-plan.md` and `issue-fix-progress.md` together.
- User confirmation gate:
  - User confirms the implementation and documentation are complete.
- Result:
  - `issue-fix-plan.md` and `issue-fix-progress.md` are synchronized to the final implementation state.
  - Local container reproduction on the 2080 Super successfully covered:
    - baseline import timing
    - heavy import graph diagnosis
    - import-path cleanup verification
    - regression test execution
  - Hopper-specific DeepGEMM import-time logs remain explicitly out of scope for strict local reproduction because the local GPU is `SM75`, not `SM90+`.

### 5.2 Prepare submission checklist

- Status: `completed`
- Atomic operation:
  - Classify the current branch contents into:
    - core source changes
    - regression tests
    - issue documentation
    - temporary or unrelated artifacts
- Expected effect:
  - The branch has a clear commit boundary before any final commit or PR preparation.
- Verification:
  - Produce an explicit include/exclude checklist tied to the current working tree.
- User confirmation gate:
  - User confirms which categories should be included in the final commit payload.
- Result:
  - Recommended include set for an issue-focused commit:
    - core source changes:
      - `python/sglang/__init__.py`
      - `python/sglang/srt/configs/__init__.py`
      - `python/sglang/srt/configs/model_config.py`
      - `python/sglang/srt/parser/reasoning_parser.py`
      - `python/sglang/srt/layers/moe/__init__.py`
      - `python/sglang/srt/layers/moe/moe_runner/__init__.py`
      - `python/sglang/srt/layers/quantization/__init__.py`
      - `python/sglang/srt/layers/quantization/fp8_config.py`
      - `python/sglang/srt/layers/quantization/fp8_utils.py`
      - `python/sglang/srt/layers/quantization/fp8_kernel.py`
      - `python/sglang/srt/layers/logits_processor.py`
      - `python/sglang/srt/layers/logits_processor_output.py`
      - `python/sglang/srt/managers/utils.py`
      - `python/sglang/srt/managers/scheduler.py`
      - `python/sglang/srt/managers/scheduler_dp_attn_mixin.py`
      - `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
      - `python/sglang/srt/lora/lora_overlap_loader.py`
      - `python/sglang/srt/server_args.py`
    - regression coverage:
      - `test/unit/test_scheduler_import_regression.py`
    - issue documentation and measurement summary:
      - `issue-fix-plan.md`
      - `issue-fix-progress.md`
      - `issue-test-plan.md`
      - `issue-test-progress.md`
      - `issue-perf-summary.md`
  - Recommended exclude set before commit:
    - temporary performance harnesses and local shims:
      - `.codex-tmp/issue10492_import_bench.py`
      - `.codex-tmp/issue10492_opt125m_shim.py`
      - `.codex-tmp/issue10492_qwen15b_shim.py`
      - `.codex-tmp/issue10492_smolllm_shim.py`
      - `.codex-tmp/issue10492_srt_engine_opt_fallback.py`
    - generated perf data and baseline snapshots:
      - `.codex-tmp/perf-results/`
      - `.codex-tmp/main-baseline-20260308-173917/`
      - `.codex-tmp/__pycache__/`
    - unrelated documentation for a separate optional docs-only commit:
      - `AGENTS.md`
  - Rationale:
    - the include set is the issue fix itself plus its reproducibility record
    - the exclude set is either temporary harness material, generated measurement data, or unrelated work not specific to issue `#10492`
  - Final commit structure chosen for this branch:
    - `Commit A`: code fix + regression coverage
    - `Commit B`: plan, progress, test record, and submission checklist
    - `Commit C`: promoted reproducibility artifacts only
