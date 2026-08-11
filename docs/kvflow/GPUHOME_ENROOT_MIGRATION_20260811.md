# gpuhome GPU11 migration and Enroot execution

This document records the portable deployment contract for the coding-aware
lossy KV-reuse implementation.  It does not change the algorithm, historical
artifacts, paper, prefetch behavior, or registered experimental thresholds.

## Frozen source

- Migration branch: `migration/gpuhome-enroot-20260811`
- Starting research commit: `45a2de40623ae3e8954f97c2e47f9dc7f68ec312`
- Target: Slurm node `gpu11`, `debug` for validation and `long` only after
  validation passes.
- Persistent and host-temporary storage: `$HOME` only.
- Container backend: Enroot 3.4 using the original SWE-bench Docker image
  references.

## Filesystem contract

Source `benchmark/multi_workflow/slurm/impactkv_home_env.sh` inside every job.
It relocates artifacts, models, Python entry points, Hugging Face state, Enroot
cache/data/runtime/temp paths, XDG runtime data, and Python temporary files
under `$HOME`.  No host project file is placed in `/tmp` or `/run`.

Enroot's container-private `/tmp` remains inside its ephemeral writable
overlay.  It is not the host `/tmp` and disappears with the task namespace.

## Container execution

`EnrootEnvironment` implements mini-SWE-agent's environment protocol.  One
foreground `enroot start --root --rw IMAGE sleep ...` process owns the task
namespace.  Agent actions enter it with `enroot exec PID`, so edits and command
state survive across turns.  Cleanup terminates the namespace and removes its
home-scoped runtime mountpoint.

Images are imported before a campaign with `prepare_enroot_images.py`.  The
resulting `IMAGE_INDEX.json` binds the original Docker reference to registry
digest (when exposed by the registry), local `.sqsh` path, byte size, and
SHA-256.

The repository-owned Enroot evaluator applies the same prediction patch,
executes `TestSpec.eval_script`, and calls SWE-bench 4.1.0's
`get_eval_report`.  Its results remain labelled `container_backend=enroot`;
they are accepted as Docker-equivalent only after the frozen five-task parity
job reports an exact outcome match.

## Validation jobs

Submit in this order:

```bash
cd "$HOME/CodeMAS_Project/sglang"
sbatch benchmark/multi_workflow/slurm/import_enroot_canary.sbatch
sbatch benchmark/multi_workflow/slurm/cuda_sglang_smoke.sbatch
sbatch benchmark/multi_workflow/slurm/enroot_docker_parity.sbatch
sbatch benchmark/multi_workflow/slurm/agent_enroot_smoke.sbatch
```

The import and parity jobs require the frozen dataset and parity inputs to be
present first.  The agent smoke runs Dense and
`coding_dependency_graph_cold_lcb` on the same registered task, with no
prefetch.  Do not submit a `long` campaign until CUDA generation, five-task
Docker/Enroot parity, official task grading, prompt identity, TTFT recording,
and physical KV-copy telemetry all pass.

`nvidia-smi` is diagnostic only.  The runtime gate is a real CUDA matrix
operation followed by a real SGLang generation.
