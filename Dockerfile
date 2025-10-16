# syntax=docker/dockerfile:1

# --- builder: install deps and train to bake model ---
FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app
COPY pyproject.toml ./

# Train model with fixed seeds; save under /app/artifacts
RUN python -m mlops_diabetes.train_v0_1 --artifacts-dir artifacts

# --- runtime: small image with only what we need ---
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/app ./app
COPY --from=builder /app/src ./src
COPY --from=builder /app/artifacts ./artifacts
COPY --from=builder /app/pyproject.toml ./

ENV MODEL_PATH=artifacts/model_v0_1.joblib
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
