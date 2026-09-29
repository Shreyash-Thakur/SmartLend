# ---------------------------------------------------------------------------
# Stage 1: build the React frontend. Nothing from this stage ships except
# the static dist/ output — Node never reaches the runtime image.
# ---------------------------------------------------------------------------
FROM node:22-alpine AS frontend-build
WORKDIR /build
# Lockfile first: npm ci re-runs only when dependencies change, not on every
# source edit (layer caching).
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: the serving image. python:3.13-slim matches the project venv.
# libgomp1 is the OpenMP runtime the lightgbm/xgboost wheels link against —
# it exists on full images but not on -slim.
# ---------------------------------------------------------------------------
FROM python:3.13-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/smartlend

# Requirements before code: editing a .py file must not invalidate the
# dependency layer. The serving image trims the training-only weight:
#   - xgboost -> xgboost-cpu (same `import xgboost`, official CPU wheel;
#     drops the 469MB nvidia-nccl dependency a t3.micro can never use)
#   - catboost (+ its plotly/graphviz) and matplotlib/seaborn: only
#     train_pipeline() and the never-imported analysis.py use them, and
#     train_pipeline imports its trainers lazily from the full research env.
# The filter runs in the SAME layer as the install on purpose: Docker layers
# are additive, so a later `pip uninstall` hides files without removing them.
COPY backend/requirements-api.txt backend/requirements-api.txt
RUN grep -vE '^(xgboost|catboost|matplotlib|seaborn)==' backend/requirements-api.txt > /tmp/requirements-serve.txt \
    && pip install --no-cache-dir -r /tmp/requirements-serve.txt xgboost-cpu==3.0.0

# The application, with the serving model baked in as a build artifact
# (backend/artifacts/pipeline_v3_real.joblib + thresholds). Nothing is
# fetched at runtime.
COPY backend/ backend/
# The relearning gate: backend/app/services/relearning_service.py imports
# research.relearning.gate so the API's verdict can never drift from the
# CLI's. .dockerignore whitelists exactly this subpackage.
COPY research/ research/
COPY --from=frontend-build /build/dist frontend/dist

EXPOSE 8000
# 0.0.0.0: bind every interface in the container's network namespace, so
# Docker's port mapping can reach the server. 127.0.0.1 here would mean
# "this container only" and the mapped port would connect to nothing.
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
