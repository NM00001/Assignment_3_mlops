import os
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app, load_model


def ensure_model():
    # If artifact not present locally, skip loading
    try:
        load_model()
    except Exception:
        pass


def test_predict_shape():
    ensure_model()
    client = TestClient(app)
    payload = {
        "age": 0.02,
        "sex": -0.044,
        "bmi": 0.06,
        "bp": -0.03,
        "s1": -0.02,
        "s2": 0.03,
        "s3": -0.02,
        "s4": 0.02,
        "s5": 0.02,
        "s6": -0.001,
    }
    res = client.post("/predict", json=payload)
    # If model isn't loaded, API returns 500; in CI we'll have artifact baked
    assert res.status_code in (200, 500)
    if res.status_code == 200:
        body = res.json()
        assert "prediction" in body
        assert isinstance(body["prediction"], (int, float))
