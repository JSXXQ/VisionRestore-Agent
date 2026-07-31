#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for name in api web; do
  if [ -f ".task-pids/$name.pid" ]; then
    kill "$(cat ".task-pids/$name.pid")" || true
    rm -f ".task-pids/$name.pid"
  fi
done
