# SmartLend Backend

## Start API

From the repository root:

```powershell
..\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

## API Endpoints

- `GET /api/health`
- `GET /api/public-metrics`
- `GET /api/dashboard-metrics`
- `GET /api/trends`
- `GET /api/applications?scope=all|customer|org`
- `GET /api/applications/{id}`
- `POST /api/applications`
- `POST /api/applications/{id}/decision`

## Dependency file

Backend API dependencies are in `requirements-api.txt`.
