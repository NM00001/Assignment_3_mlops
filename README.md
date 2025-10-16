# Virtual Diabetes Clinic Triage – MLOps Assignment

## Overview
- Predict short-term progression using scikit-learn Diabetes dataset.
- Serve a regression model via FastAPI.
- CI/CD with GitHub Actions; Docker image published to GHCR.

## Quickstart (Local)
1. Python 3.11
2. Install deps:
```bash
make install
```
3. Train v0.1 (saves artifact to `artifacts/`):
```bash
make train-v0.1
```
4. Run API (uses `MODEL_PATH` env or default `artifacts/model_v0_1.joblib`):
```bash
make api
```
5. Endpoints:
- GET http://localhost:8000/health
- POST http://localhost:8000/predict

### Sample payload
```json
{
  "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
  "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001
}
```

## Docker
Build and run locally:
```bash
docker build -t diabetes-mlops:local .
docker run --rm -p 8000:8000 diabetes-mlops:local
```

## CI/CD
- PR/Push: lint, tests, training smoke, upload artifacts.
- Tag `v*`: build & push GHCR image, container smoke, publish Release with metrics.

## GHCR Setup
- Create repo-level secrets:
  - `GHCR_PAT` (token with `packages:write`, `read:packages`, `repo`)
  - Optionally set Repository variables `GHCR_OWNER` (your GitHub username/org) and `IMAGE_NAME` (defaults to `<owner>/<repo>` if omitted)

## Reproduce
- Deterministic seeds; pinned requirements.
- See `CHANGELOG.md` for v0.1 → v0.2 metrics and rationale.
