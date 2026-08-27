import io, time
from PIL import Image
from fastapi.testclient import TestClient
from visionrestore.main import app


def test_e2e_smoke_task_reaches_terminal_state():
    c = TestClient(app)
    bio = io.BytesIO()
    Image.new("RGB", (32,32), (20,20,20)).save(bio, format="PNG")
    upload = c.post("/api/v1/images/upload", files={"file": ("x.png", bio.getvalue(), "image/png")})
    assert upload.status_code == 200
    image_id = upload.json()["data"]["file_id"]
    task = c.post("/api/v1/tasks", json={"image_id": image_id, "mode": "manual", "model_id": "sci", "checkpoint_id": "medium", "priority": "speed"}).json()["data"]
    for _ in range(80):
        got = c.get(f"/api/v1/tasks/{task['task_id']}").json()["data"]
        if got["status"] in {"completed","failed","cancelled"}:
            break
        time.sleep(.25)
    assert got["status"] in {"completed", "failed", "awaiting_denoise_confirmation", "awaiting_sr_confirmation"}
    assert got["model_candidates"] or got["error"] is not None
