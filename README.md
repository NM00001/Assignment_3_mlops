# Virtual Diabetes Clinic Triage – MLOps Assignment

![CI](https://github.com/NM00001/Assignment_3_mlops/actions/workflows/ci.yml/badge.svg)
![Release](https://github.com/NM00001/Assignment_3_mlops/actions/workflows/release.yml/badge.svg)

## Overview
- Predict short-term progression using scikit-learn Diabetes dataset.
- Serve a regression model via FastAPI.
- CI/CD with GitHub Actions; Docker image published to GHCR.

## Quickstart (Local)
Clone and enter the project first:
```bash
git clone https://github.com/NM00001/Assignment_3_mlops.git
cd Assignment_3_mlops
```
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

Optional: run v0.2 locally instead of v0.1
```bash
make train-v0.2
MODEL_PATH=artifacts/model_v0_2.joblib make api
```
Or simply use the GHCR images below.

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

Run published images from GHCR:
```bash
docker pull ghcr.io/nm00001/assignment_3_mlops:v0.1
docker run --rm -p 8000:8000 ghcr.io/nm00001/assignment_3_mlops:v0.1

# or v0.2
docker pull ghcr.io/nm00001/assignment_3_mlops:v0.2
docker run --rm -p 8000:8000 ghcr.io/nm00001/assignment_3_mlops:v0.2
```

## CI/CD
- PR/Push: lint, tests, training smoke, upload artifacts.
- Tag `v*`: build & push GHCR image, container smoke, publish Release with metrics.

## Release/Permissions
- The Release workflow uses the built-in `GITHUB_TOKEN`.
- Ensure repo Settings → Actions → General → Workflow permissions = **Read and write permissions**.
- Make the container package public after the first release (GitHub → Packages → container → Public).

## Reproduce
- Deterministic seeds; pinned requirements.
- See `CHANGELOG.md` for v0.1 → v0.2 metrics and rationale.

### Reproduce locally

v0.1
```bash
make install
make train-v0.1
MODEL_PATH=artifacts/model_v0_1.joblib make api
```

v0.2
```bash
make install
make train-v0.2
MODEL_PATH=artifacts/model_v0_2.joblib make api
```
