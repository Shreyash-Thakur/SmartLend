# SmartLend

SmartLend is a full-stack loan decision support system built with a FastAPI backend, a React + Vite frontend, SQLite persistence, and a hybrid ML + CBES scoring flow. The app accepts loan applications, stores them in the database, returns automated decisions, and exposes explainability and dashboard endpoints for review workflows.

## Stack

- Backend: FastAPI, SQLAlchemy, SQLite, pandas, scikit-learn style inference flow
- Frontend: React, TypeScript, Vite, Zustand, Axios
- ML Logic: Hybrid decisioning with ML probability, CBES-based threshold adjustment, and explainability helpers
- Document Support: OCR/parsing pipeline for uploaded application documents

## Project Structure

```text
backend/
  app/
    main.py
    database.py
    models.py
    schemas.py
    routers/
    services/
  requirements-api.txt
  smartlend.db

frontend/
  src/
  package.json
  vite.config.ts
```

## Local Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install backend dependencies

```powershell
pip install -r backend\requirements-api.txt
```

### 3. Install frontend dependencies

```powershell
Set-Location frontend
npm install
Set-Location ..
```

## Run The App

### Backend

From the repository root:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:

- `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

### Frontend

From the repository root:

```powershell
Set-Location frontend
npm run dev
```

The Vite dev server proxies `/api` requests to `http://127.0.0.1:8000` via `frontend/vite.config.ts`.

## Main API Endpoints

- `GET /api/health`
- `GET /api/public-metrics`
- `GET /api/dashboard-metrics`
- `GET /api/trends`
- `GET /api/metrics`
- `GET /api/applications`
- `GET /api/applications/{application_id}`
- `GET /api/applications/{application_id}/explain`
- `POST /api/applications`
- `POST /api/upload-form`
- `POST /api/applications/{application_id}/decision`
- `POST /api/applications/{application_id}/documents`

## Current Flow

1. Submit a loan application from the frontend or directly to the API.
2. Backend validates the payload and computes ML + CBES decision metadata.
3. Application data is stored in SQLite at `backend/smartlend.db`.
4. The frontend can fetch applications, metrics, and explanation data through `/api`.
5. Review workflows can upload documents and apply manual decision overrides.

## Notes

- The backend loads its dataset from `backend/synthetic_indian_loan_dataset.csv`.
- The database file is created automatically when the FastAPI app starts.
- Ignored local artifacts include `.venv/`, `frontend/node_modules/`, `frontend/dist/`, and `*.tsbuildinfo`.
