# gpuhome GPU11 migration and Enroot execution

This document records the portable deployment contract for the coding-aware
lossy KV-reuse implementation.  It does not change the algorithm, historical
artifacts, paper, prefetch behavior, or registered experimental thresholds.

## Frozen source

- Migration branch: `migration/gpuhome-enroot-20260811`
- Starting research commit: `45a2de40623ae3e8954f97c2e47f9dc7f68ec312`
- SSH bootstrap target: `gpuhome_gpu11`.  The home directory is shared across
  the cluster.  Nodes `gpu10` through `gpu13` have known severe interference
  and are forbidden for migration validation.  Because `debug` contains only
  those four nodes, bounded validation jobs use `long` with
  `--exclude=gpu[10-13,23-24]`.  Within `long`, that leaves only
  `gpu14` through `gpu19`; GPU jobs additionally request RTX 4090.  Excluding
  `gpu23` and `gpu24` prevents CPU-only jobs from landing where Enroot is
  unavailable.  The shared environment script also
  fails closed if a Slurm override nevertheless places a job on a forbidden
  node.
- Persistent and host-temporary storage: `$HOME` only.
- Container backend: Enroot 3.4 using the original SWE-bench Docker image
  references.

## Filesystem contract

Source `benchmark/multi_workflow/slurm/impactkv_home_env.sh` inside every job.
It relocates artifacts, models, Python entry points, Hugging Face state, Enroot
cache/data/runtime/temp paths, XDG runtime data, and Python temporary files
under `$HOME`.  It also prepends the SGLang environment's `bin` directory so
runtime JIT compilation resolves the environment-pinned `ninja` executable on
compute nodes.  Slurm jobs use the cluster HTTP proxy instead of node-local
loopback proxies and bypass it for localhost SGLang traffic.  No host project
file is placed in `/tmp` or `/run`.

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

The model is pinned to repository revision
`4bd30395b72ea6045edd04806c4fea448d4467b3`.  The local source snapshot was
originally labelled `2831070b7b8c7aa6b7012333c6c4a2bd257f6cdf`, but that
revision became unreachable after the upstream repository history changed.
The current revision resolves to the same config, tokenizer, index, and four
weight LFS objects; the migration manifest records their content hashes.

The repository-owned Enroot evaluator applies the same prediction patch,
executes `TestSpec.eval_script`, and calls SWE-bench 4.1.0's
`get_eval_report`.  Its results remain labelled `container_backend=enroot`;
they are accepted as Docker-equivalent only after the frozen five-task parity
job reports an exact outcome match.

## Validation jobs

Submit in this order:

```bash
cd "$HOME/CodeMAS_Project/sglang"
import_job=$(sbatch --parsable \
  benchmark/multi_workflow/slurm/import_enroot_canary.sbatch)
cuda_job=$(sbatch --parsable \
  benchmark/multi_workflow/slurm/cuda_sglang_smoke.sbatch)
sbatch --dependency="afterok:$import_job" \
  benchmark/multi_workflow/slurm/enroot_docker_parity.sbatch
sbatch --dependency="afterok:$import_job:$cuda_job" \
  benchmark/multi_workflow/slurm/agent_enroot_smoke.sbatch
```

The import and parity jobs require the frozen dataset and parity inputs to be
present first.  The agent smoke runs Dense and
`coding_dependency_graph_cold_lcb` on the same registered task, with no
prefetch.  The use of the `long` partition here is only a placement workaround;
the scripts retain short canary wall-time limits.  Do not submit a formal
campaign until CUDA generation, five-task Docker/Enroot parity, official task
grading, prompt identity, TTFT recording, and physical KV-copy telemetry all
pass.

The home bootstrap (venv creation, model download, and content hashing) is
CPU/storage work and may be completed on the login host.  Image execution,
CUDA checks, SGLang generation, agent inference, and official grading remain
bounded Slurm jobs.  A pending canary is not a failure: inspect its
reason and estimated start with `squeue --start -j JOB_ID`.

`nvidia-smi` is diagnostic only.  The runtime gate is a real CUDA matrix
operation followed by a real SGLang generation.
