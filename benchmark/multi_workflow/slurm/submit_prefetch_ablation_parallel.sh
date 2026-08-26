#!/usr/bin/env bash
# One exclusive 4090 job per staircase arm. Never comma-export modes.
# Submit from gpuhome_gpu11. Never writes frozen dualisland RESULT.
set -euo pipefail
EXEC=0
for arg in "$@"; do
  case "$arg" in
    --exec) EXEC=1 ;;
    --help|-h)
      echo "usage: $0 [--exec]" >&2
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done
HERE="$(cd "$(dirname "$0")" && pwd)"
SBATCH="$HERE/swebench_template_prefetch.sbatch"
ART="${IMPACTKV_PREFETCH_ARTIFACT:-$HOME/CodeMAS_Project/kvflow-artifacts/runs/prefetch_ablation_7b_cluster_20260826}"
FROZEN="${IMPACTKV_PREFETCH_PLAN:-$HOME/CodeMAS_Project/kvflow-artifacts/impactkv_swebench_template_prefetch_7b_dualisland_20260822/PLAN.json}"
EXCLUDE="gpu[10-13,15,17,23-24]"
mkdir -p "$ART"
if [[ ! -f "$ART/PLAN.json" ]]; then
  if [[ ! -f "$FROZEN" ]]; then
    echo "missing PLAN: $FROZEN" >&2
    exit 1
  fi
  cp -a "$FROZEN" "$ART/PLAN.json"
fi
if [[ "$ART" == *dualisland_20260822* ]] || [[ "$ART" == *prefixkey_20260824* ]]; then
  echo "refusing to write frozen artifact $ART" >&2
  exit 1
fi
ids=""
for mode in dense lossy_host prefix_prefetch template_prefetch; do
  modes_file="$HERE/ablation_mode_${mode}.txt"
  printf '%s\n' "$mode" > "$modes_file"
  echo "argv: sbatch --exclude=$EXCLUDE --job-name=ikv-stair-${mode} $SBATCH"
  echo "modes_file: $mode artifact: $ART"
  if [[ "$EXEC" -eq 1 ]]; then
    export IMPACTKV_PREFETCH_MODES_FILE="$modes_file"
    export IMPACTKV_PREFETCH_ARTIFACT="$ART"
    export IMPACTKV_MODEL="${IMPACTKV_MODEL:-$HOME/models/Qwen2.5-Coder-7B-Instruct}"
    jid=$(sbatch --parsable --export=ALL \
      --exclude="$EXCLUDE" \
      --job-name="ikv-stair-${mode}" \
      --output="$ART/ikv-stair-${mode}-%j.out" \
      --error="$ART/ikv-stair-${mode}-%j.err" \
      "$SBATCH")
    echo "submitted $mode job $jid"
    ids="${ids:+$ids,}$jid"
  else
    echo "dry-run $mode; pass --exec on gpuhome_gpu11 to submit"
  fi
done
if [[ "$EXEC" -eq 1 ]]; then
  echo "JOB_IDS=$ids"
  echo "ARTIFACT=$ART"
  echo "EXCLUDE=$EXCLUDE"
fi
