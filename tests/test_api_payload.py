from fastapi.testclient import TestClient

from app.main import app, load_model


def test_health_ok(tmp_path, monkeypatch):
    # Ensure model is loaded (assumes artifact exists when running API)
    try:
        load_model()
    except Exception:
        pass
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
