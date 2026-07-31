#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
mkdir -p .task-pids
python -m uvicorn visionrestore.main:app --app-dir apps/api --host 127.0.0.1 --port 8000 &
echo $! > .task-pids/api.pid
(cd apps/web && npm run dev) &
echo $! > .task-pids/web.pid
echo "网页: http://127.0.0.1:5173"
echo "API 文档: http://127.0.0.1:8000/docs"
