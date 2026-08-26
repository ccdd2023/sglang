#!/usr/bin/env bash
# Submit one exclusive Slurm job per isolated 7B dual-island mode.
# Never put comma modes on sbatch --export.
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
ART="${IMPACTKV_PREFETCH_ARTIFACT:-$HOME/CodeMAS_Project/kvflow-artifacts/impactkv_swebench_template_prefetch_7b_dualisland_20260822}"
EXPORT_ARG="--export=ALL"
if [[ "$EXPORT_ARG" == *IMPACTKV_PREFETCH_MODES=* ]]; then
  echo "modes leaked onto --export: $EXPORT_ARG" >&2
  exit 1
fi
mkdir -p "$ART"
ids=""
for mode in dense prefix_only lossy_only dual combined; do
  modes_file="$HERE/prefetch_mode_${mode}.txt"
  printf '%s\n' "$mode" > "$modes_file"
  echo "argv: sbatch $EXPORT_ARG --job-name=impactkv-7b-${mode} $SBATCH"
  echo "modes_file: $(tr -d '\n' < "$modes_file")"
  if [[ "$EXEC" -eq 1 ]]; then
    export IMPACTKV_PREFETCH_MODES_FILE="$modes_file"
    export IMPACTKV_PREFETCH_ARTIFACT="$ART"
    export IMPACTKV_MODEL="${IMPACTKV_MODEL:-$HOME/models/Qwen2.5-Coder-7B-Instruct}"
    jid=$(sbatch --parsable "$EXPORT_ARG" \
      --job-name="impactkv-7b-${mode}" \
      --output="impactkv-7b-${mode}-%j.out" \
      --error="impactkv-7b-${mode}-%j.err" \
      "$SBATCH")
    echo "submitted $mode job $jid"
    ids="${ids:+$ids,}$jid"
  else
    echo "dry-run $mode; pass --exec on gpuhome_gpu11 to submit"
  fi
done
if [[ "$EXEC" -eq 1 ]]; then
  echo "JOB_IDS=$ids"
fi
