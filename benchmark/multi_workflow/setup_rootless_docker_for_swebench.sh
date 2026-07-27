#!/usr/bin/env bash
set -euo pipefail

if ! command -v newuidmap >/dev/null || ! command -v newgidmap >/dev/null; then
  cat >&2 <<'EOF'
Rootless Docker requires the distro's setuid uidmap helpers.
Run this one administrator command, then rerun this script:

  sudo apt-get update && sudo apt-get install -y uidmap
EOF
  exit 2
fi

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
dockerd-rootless-setuptool.sh install

export DOCKER_HOST="unix://${XDG_RUNTIME_DIR}/docker.sock"
docker context use rootless >/dev/null
docker info --format 'rootless Docker server={{.ServerVersion}} root={{.DockerRootDir}}'

cat <<EOF

Rootless Docker is ready.
For non-login shells, export:
  DOCKER_HOST=${DOCKER_HOST}
EOF
