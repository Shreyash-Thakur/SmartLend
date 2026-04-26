#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

cd backend
python -m pip install -r requirements.txt
if [ ! -f artifacts/pipeline_v2.joblib ]; then
  python retrain_pipeline_v2.py
fi
uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!

cd ../frontend
npm install
npm run dev

kill "$BACKEND_PID" >/dev/null 2>&1 || true
