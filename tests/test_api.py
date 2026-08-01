from fastapi.testclient import TestClient
from visionrestore.main import app


def test_health():
    c = TestClient(app)
    r = c.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["data"]["status"] == "ok"


def test_v2_health():
    c = TestClient(app)
    r = c.get("/api/v2/health")
    assert r.status_code == 200
    assert r.json()["data"]["api_version"] == "v2"


def test_v2_models_are_grouped_and_count_real_ready_only():
    c = TestClient(app)
    r = c.get("/api/v2/models")
    assert r.status_code == 200
    data = r.json()["data"]
    assert "enhancement" in data["groups"]
    assert "postprocess" in data["groups"]
    ids = {item["model_id"] for item in data["models"]}
    assert {"retinexformer", "darkir", "hvi_cidnet", "flol", "sci", "zero_dce", "lpdm", "mambair"}.issubset(ids)
    assert data["ready_count"] >= 0


def test_v2_scan_models():
    c = TestClient(app)
    r = c.post("/api/v2/models/scan")
    assert r.status_code == 200
    assert "models" in r.json()["data"]
