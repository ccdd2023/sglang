#!/usr/bin/env bash
# Sync 7B dual-island prefetch code + PLAN to gpuhome. Never copies 30B PLAN.
set -euo pipefail
REMOTE="${IMPACTKV_REMOTE:-gpuhome_gpu11}"
REPO_REMOTE="${IMPACTKV_REMOTE_REPO:-CodeMAS_Project/worktrees/sglang-common-agent}"
ART_REMOTE="${IMPACTKV_REMOTE_ARTIFACTS:-CodeMAS_Project/kvflow-artifacts}"
LOCAL_REPO="/home/gfy/CodeMAS_Project/sglang-kvflow-worktrees/template-prefetch"
ART_NAME="impactkv_swebench_template_prefetch_7b_dualisland_20260822"
LOCAL_ART="/home/gfy/CodeMAS_Project/kvflow-artifacts/${ART_NAME}"

cd "$LOCAL_REPO"
tar czf - \
  python/sglang/srt/mem_cache/kvcomm_exact.py \
  python/sglang/srt/mem_cache/kvcomm/radix_backend.py \
  python/sglang/srt/mem_cache/radix_cache.py \
  python/sglang/srt/mem_cache/kvcomm_prefetch \
  python/sglang/srt/mem_cache/kvcomm/types.py \
  python/sglang/srt/managers/schedule_policy.py \
  benchmark/multi_workflow/run_swebench_template_prefetch.py \
  benchmark/multi_workflow/template_prefetch_modes.py \
  benchmark/multi_workflow/slurm_export_guard.py \
  benchmark/multi_workflow/run_swebench_prerotated_file_modules.py \
  benchmark/multi_workflow/prepare_7b_dual_island_plan.py \
  benchmark/multi_workflow/slurm/swebench_template_prefetch.sbatch \
  benchmark/multi_workflow/slurm/submit_swebench_template_prefetch.sh \
  benchmark/multi_workflow/slurm/submit_swebench_template_prefetch_parallel.sh \
  | ssh "$REMOTE" "mkdir -p \"\$HOME/$REPO_REMOTE\" && tar xzf - -C \"\$HOME/$REPO_REMOTE\""

if [[ ! -f "$LOCAL_ART/PLAN.json" ]]; then
  echo "missing $LOCAL_ART/PLAN.json; generate the 7B PLAN first" >&2
  exit 1
fi
ssh "$REMOTE" "mkdir -p \"\$HOME/$ART_REMOTE/$ART_NAME\""
ssh "$REMOTE" "cat > \"\$HOME/$ART_REMOTE/$ART_NAME/PLAN.json\"" < "$LOCAL_ART/PLAN.json"
echo "Synced 7B dual-island code and PLAN to $REMOTE"
echo "sbatch: $HOME/$REPO_REMOTE/benchmark/multi_workflow/slurm/submit_swebench_template_prefetch_parallel.sh --exec"
