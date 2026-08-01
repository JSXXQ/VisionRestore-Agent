from pathlib import Path

from visionrestore.core.config import get_settings
from visionrestore.schemas.artifact import ArtifactLineage, ArtifactRecord


class ArtifactLineageService:
    def __init__(self):
        self.settings = get_settings()

    def task_dir(self, task_id: str) -> Path:
        return self.settings.data_dir / "tasks" / task_id

    def create_task_layout(self, task_id: str) -> Path:
        root = self.task_dir(task_id)
        for name in ["input", "previews", "candidates", "selected", "postprocess", "final", "reports", "logs"]:
            (root / name).mkdir(parents=True, exist_ok=True)
        return root

    def from_task(self, task) -> ArtifactLineage:
        root = self.create_task_layout(task.task_id)
        candidates = []
        for index, item in enumerate(task.candidates, start=1):
            candidate_dir = root / "candidates" / f"candidate_{index:02d}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidates.append(ArtifactRecord(
                role="candidate",
                file_id=item.output_file_id,
                url=item.output_url,
                model_id=item.model_id,
                checkpoint_id=item.checkpoint_id,
                sha256=item.output_sha256,
                metadata={"score": item.score, "status": item.status, "candidate_dir": str(candidate_dir)},
            ))
        selected = None
        if task.best_result:
            selected = ArtifactRecord(
                role="best_enhanced",
                file_id=task.best_result.output_file_id,
                url=task.best_result.output_url,
                model_id=task.best_result.model_id,
                checkpoint_id=task.best_result.checkpoint_id,
                sha256=task.best_result.output_sha256,
                metadata={"score": task.best_result.score},
            )
        reports = []
        if task.report_file_id:
            reports.append(ArtifactRecord(role="report", file_id=task.report_file_id, url=f"/api/v1/tasks/{task.task_id}/report"))
        return ArtifactLineage(
            task_id=task.task_id,
            task_dir=str(root),
            input=ArtifactRecord(role="original", file_id=task.image_id),
            candidates=candidates,
            selected=selected,
            final=selected,
            reports=reports,
            logs_dir=str(root / "logs"),
        )
