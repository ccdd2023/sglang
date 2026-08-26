#!/usr/bin/env bash
# Write prefetch modes to a file. Never put IMPACTKV_PREFETCH_MODES on
# sbatch --export (Slurm comma-splits; job 114807 lost combined that way).
# Login-node python is 3.6; this script is bash-only.
# Usage on gpuhome_gpu11 after sync:
#   bash submit_swebench_template_prefetch.sh combined --exec
set -euo pipefail
MODES="combined"
EXEC=0
for arg in "$@"; do
  case "$arg" in
    --exec) EXEC=1 ;;
    --help|-h)
      echo "usage: $0 [dense|prefix_only|lossy_only|dual|combined] [--exec]" >&2
      exit 0
      ;;
    *) MODES="$arg" ;;
  esac
done
HERE="$(cd "$(dirname "$0")" && pwd)"
SBATCH="$HERE/swebench_template_prefetch.sbatch"
MODES_FILE="$HERE/prefetch_modes.txt"

printf '%s\n' "$MODES" > "$MODES_FILE"
EXPORT_ARG="--export=ALL"
if [[ "$EXPORT_ARG" == *IMPACTKV_PREFETCH_MODES=* ]]; then
  echo "modes leaked onto --export: $EXPORT_ARG" >&2
  exit 1
fi
if [[ "$EXPORT_ARG" == *","* && "$EXPORT_ARG" == *IMPACTKV_PREFETCH_MODES=* ]]; then
  echo "comma modes on --export: $EXPORT_ARG" >&2
  exit 1
fi
echo "argv: sbatch $EXPORT_ARG $SBATCH"
echo "modes_file: $(tr -d '\n' < "$MODES_FILE")"
echo "sbatch $EXPORT_ARG $SBATCH"
if [[ "$EXEC" -eq 1 ]]; then
  sbatch "$EXPORT_ARG" "$SBATCH"
else
  echo "dry-run; pass --exec on gpuhome_gpu11 to submit"
fi

