VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
UVICORN=$(VENV)/bin/uvicorn
PYTEST=$(VENV)/bin/pytest
RUFF=$(VENV)/bin/ruff
PKG=mlops_diabetes
ARTIFACTS_DIR=artifacts
MODEL_PATH=$(ARTIFACTS_DIR)/model_v0_1.joblib

.PHONY: install train-v0.1 run api test lint docker-build docker-run

install:
	python3 -m venv $(VENV) && $(PIP) install -U pip && $(PIP) install -r requirements.txt

train-v0.1:
	PYTHONPATH=src $(PYTHON) -m $(PKG).train_v0_1 --artifacts-dir $(ARTIFACTS_DIR)

run:
	PYTHONPATH=src $(UVICORN) app.main:app --host 0.0.0.0 --port 8000 --reload

api:
	PYTHONPATH=src MODEL_PATH=$(MODEL_PATH) $(UVICORN) app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	PYTHONPATH=src $(PYTEST) -q

lint:
	$(RUFF) check .

docker-build:
	docker build -t ghcr.io/$(GITHUB_ORG)/$(GITHUB_REPO):local .

docker-run:
	docker run --rm -p 8000:8000 ghcr.io/$(GITHUB_ORG)/$(GITHUB_REPO):local
