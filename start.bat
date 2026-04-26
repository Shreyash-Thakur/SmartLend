@echo off
setlocal
cd /d "%~dp0"

cd backend
python -m pip install -r requirements.txt
if not exist artifacts\pipeline_v2.joblib (
    python retrain_pipeline_v2.py
)
cd ..
start "SmartLend Backend" cmd /k "set PYTHONPATH=. && uvicorn backend.main:app --reload --port 8000"

cd frontend
npm install
npm run dev
