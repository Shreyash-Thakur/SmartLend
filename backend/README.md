# SmartLend Backend

FastAPI service. Start from the **repository root**:

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Interactive API docs (the authoritative endpoint list — do not maintain one by
hand here): **http://localhost:8000/docs**

Routers live in `backend/app/routers/`:

| Router | Prefix | Purpose |
|---|---|---|
| `public.py` | *(none)* | `/health`, `/predict`, public dashboard metrics |
| `applications.py` | `/api` | applications CRUD, decisions, documents, explain, audit report, reason codes, dashboards, model-analysis |
| `customers.py` | `/api` | sample customer ids, on-file profiles |
| `relearning.py` | `/api` | relearning gate status (read-only, fails closed) |
| `tabpfn.py` | `/api` | optional Colab-hosted TabPFN status + second opinion |
| `agent.py` | `/api` | optional Claude reviewer-briefing agent |
| `voice.py` | `/api` | optional TTS/STT |

Dependencies: `requirements-api.txt` (must keep `from backend.app.main import
app` importable). Setup, layout, working agreements and pitfalls:
**[`../docs/DEV-GUIDE.md`](../docs/DEV-GUIDE.md)**.
