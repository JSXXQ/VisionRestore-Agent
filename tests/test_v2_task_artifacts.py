from pathlib import Path

from fastapi.testclient import TestClient

from visionrestore.main import app
from visionrestore.schemas.task import CandidateResult, TaskRecord
from visionrestore.services.artifact_lineage import ArtifactLineageService
from visionrestore.storage.database import Database


def _save_task(task_id="v2-artifacts-case"):
    task = TaskRecord(
        task_id=task_id,
        status="completed",
        image_id="input-file",
        user_goal="test",
        mode="auto",
        priority="balanced",
        created_at="2026-08-01T00:00:00+00:00",
        candidates=[
            CandidateResult(
                model_id="retinexformer",
                checkpoint_id="lol_v2_real",
                output_file_id="out-1",
                output_url="/api/v1/files/out-1",
                status="completed",
                score=80,
                metrics={
                    "score": 80,
                    "components": {"shadow_recovery": .8, "highlight_protection": .8, "color_stability": .8, "noise_control": .8, "sharpness": .8, "structure": .8},
                    "runtime_ms": 1000,
                    "peak_memory_mb": 2000,
                    "mean_luminance_after": 80,
                    "overexposed_pixel_ratio_after": .01,
                    "color_cast_index_after": .02,
                    "structure_keep_estimate": .8,
                },
            )
        ],
        best_result=CandidateResult(model_id="retinexformer", checkpoint_id="lol_v2_real", output_file_id="out-1", output_url="/api/v1/files/out-1", status="completed", score=80),
    )
    Database().put_task(task.task_id, task.model_dump())
    return task


def test_artifact_lineage_creates_task_layout():
    task = _save_task("artifact-layout-case")
    lineage = ArtifactLineageService().from_task(task)
    root = Path(lineage.task_dir)
    assert (root / "input").exists()
    assert (root / "candidates" / "candidate_01").exists()
    assert lineage.selected.file_id == "out-1"


def test_v2_task_candidates_ranking_and_artifacts():
    task = _save_task("v2-task-query-case")
    c = TestClient(app)
    assert c.get(f"/api/v2/tasks/{task.task_id}/candidates").status_code == 200
    ranking = c.get(f"/api/v2/tasks/{task.task_id}/ranking")
    assert ranking.status_code == 200
    assert ranking.json()["data"]["best"]["model_id"] == "retinexformer"
    artifacts = c.get(f"/api/v2/tasks/{task.task_id}/artifacts")
    assert artifacts.status_code == 200
    assert artifacts.json()["data"]["selected"]["file_id"] == "out-1"


def test_v2_postprocess_decision_records_without_fake_execution():
    task = _save_task("v2-postprocess-decision-case")
    c = TestClient(app)
    response = c.post(f"/api/v2/tasks/{task.task_id}/postprocess/decision", json={"operation": "denoise", "decision": "skip"})
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["executed"] is False
    assert "不伪造" in data["message"]
