#!/usr/bin/env bash
set -euo pipefail

exec python3 -m sglang.launch_server \
  --attention-backend torch_native \
  --sampling-backend pytorch \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  "$@"
