# Repository Guidelines

## Project Structure & Module Organization
`python/sglang/` is the main package. Core serving code lives in `python/sglang/srt/`, the frontend language in `python/sglang/lang/`, CLI entry points in `python/sglang/cli/`, JIT kernels in `python/sglang/jit_kernel/`, and diffusion or multimodal code in `python/sglang/multimodal_gen/`. Shared examples, benchmarks, docs, and assets live in `examples/`, `benchmark/`, `docs/`, and `assets/`.

`test/` contains repository-level Python coverage, including `unit/`, `srt/`, `registered/`, and `manual/`. `sgl-kernel/` is the standalone kernel package, and `sgl-model-gateway/` is the Rust gateway with Python and Go bindings.

## Build, Test, and Development Commands
Install the main package in editable mode with `pip install -e "python"`. Run repository checks with `pre-commit install` and `pre-commit run --all-files`.

Common test entry points:
- `python test/run_suite.py --hw cuda --suite stage-b-test-small-1-gpu` for the standard CUDA per-commit suite.
- `python test/srt/test_srt_endpoint.py` for a single runtime test file.
- `cd sgl-kernel && make build` to build the kernel package.
- `cd sgl-model-gateway && cargo test` or `pytest e2e_test/` to validate the gateway.

## Coding Style & Naming Conventions
Follow `.editorconfig`: 4-space indentation by default, 2 spaces for JSON/YAML/Markdown, and tabs in `Makefile`s. Python formatting is enforced with `isort` and `black`; `ruff` checks critical issues such as unused imports and undefined names. C++ and CUDA code use `clang-format`.

Use `snake_case` for Python modules and functions, `PascalCase` for classes, and descriptive feature-specific filenames such as `allocator_ascend.py`. Keep hot-path runtime code efficient, avoid duplicate logic, and prefer new files over large conditional branches when adding hardware-specific behavior.

## Testing Guidelines
Add a regression test for every bug fix or feature. Name Python test files `test_*.py`. When adding CI tests under `test/srt` or `test/lang`, register them in the relevant `run_suite.py` so CI can discover them. Keep tests fast by reusing launched servers where possible; move long or environment-specific coverage to `test/manual/` or nightly suites.

## Commit & Pull Request Guidelines
Match the recent history: optional subsystem scope in brackets, then a short imperative subject, for example `[diffusion] fix: guard width passthrough`. Do not commit directly to `main`; open a topic branch and submit a PR.

PRs should include a clear problem statement, linked issue when available, and the exact validation commands you ran. If a change affects runtime behavior, kernels, or model output, include benchmark or accuracy notes. Attach screenshots only when the change affects docs, dashboards, or other user-facing visuals.
