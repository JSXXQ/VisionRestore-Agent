from pathlib import Path
from uuid import uuid4

from PIL import Image

from visionrestore.core.config import get_settings
from visionrestore.core.model_config import external_python, file_sha256
from visionrestore.schemas.candidate import CandidateExecutionPlan
from visionrestore.schemas.common import now_iso
from visionrestore.schemas.task import CandidateResult
from visionrestore.services.evaluator import QualityEvaluator
from visionrestore.services.iqa_service import IQAService
from visionrestore.services.region_constraints import RegionConstraintService
from visionrestore.storage.database import Database


class MultiCandidateExecutor:
    def __init__(self, registry=None, evaluator=None, database=None):
        from visionrestore.adapters.registry import ModelRegistry

        self.registry = registry or ModelRegistry()
        self.evaluator = evaluator or QualityEvaluator()
        self.db = database or Database()
        self.settings = get_settings()
        self.iqa_service = IQAService(database=self.db)
        self.region_constraints = RegionConstraintService()

    def execute(self, *, plan: CandidateExecutionPlan, input_path: str, priority: str = "balanced", task_dir: str | Path | None = None, cancel_check=None, on_event=None, region_constraints: list[dict] | None = None) -> list[CandidateResult]:
        source_input = str(input_path)
        output_root = Path(task_dir) if task_dir else self.settings.data_dir / "tasks" / "adhoc" / "candidates"
        output_root.mkdir(parents=True, exist_ok=True)
        input_size = self._image_size(Path(source_input))
        results: list[CandidateResult] = []
        for item in plan.candidates:
            if cancel_check and cancel_check():
                break
            candidate_dir = output_root / item.candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            output_id = str(uuid4())
            output_path = candidate_dir / f"{output_id}.png"
            if on_event:
                on_event("running_candidates", f"调用真实源码和真实权重推理 {item.model_id}:{item.checkpoint_id}。")
            try:
                adapter = self.registry.get(item.model_id)
                result = adapter.enhance(
                    image_path=source_input,
                    output_path=str(output_path),
                    device="cuda",
                    precision=self.settings.default_precision,
                    parameters={"checkpoint_id": item.checkpoint_id, "device": "cuda", "precision": self.settings.default_precision},
                )
                if result.is_mock:
                    raise RuntimeError("Mock result is not allowed in formal multi-candidate execution")
                if on_event:
                    on_event("evaluating_candidates", f"评价候选 {item.candidate_id} 的真实输出。")
                validity = self._validity_metrics(Path(source_input), Path(result.output_path))
                metrics = self.evaluator.evaluate(source_input, result.output_path, priority, result.runtime_ms, result.peak_memory_mb)
                metrics = {
                    **metrics,
                    **validity,
                    "candidate_id": item.candidate_id,
                    "role": item.role,
                    "planning_score": item.planning_score,
                    "model_prior_score": item.model_prior_score,
                    "input_match_score": item.input_match_score,
                    "local_score": item.local_score,
                    "ai_semantic_bonus": item.ai_semantic_bonus,
                    "hardware_adjustment": item.hardware_adjustment,
                    "knowledge_adjustment": item.knowledge_adjustment,
                    "planning_evidence": [evidence.model_dump(mode="json") for evidence in item.planning_evidence],
                    "knowledge_evidence": [evidence.model_dump(mode="json") for evidence in item.knowledge_evidence],
                    "planning_reason": item.reason,
                    "input_policy": item.input_policy,
                }
                if region_constraints:
                    metrics["region_constraints"] = self.region_constraints.evaluate(
                        input_path=source_input,
                        output_path=result.output_path,
                        constraints=region_constraints,
                    )
                iqa_status = self.iqa_service.status()
                if iqa_status.get("available"):
                    iqa_result = self.iqa_service.evaluate_image(Path(result.output_path))
                    metrics["iqa"] = iqa_result.get("raw", {})
                    metrics["iqa_normalized"] = iqa_result.get("normalized", {})
                    metrics["iqa_detail"] = iqa_result
                else:
                    metrics["iqa"] = {"status": "unavailable"}
                    metrics["iqa_normalized"] = {}
                    metrics["iqa_detail"] = {"status": "unavailable", "fallback": "local_technical_score_only", "reason": iqa_status.get("status_message"), "formal_mode_allows_mock": False}

                self._register_output_file(output_id, Path(result.output_path))
                results.append(CandidateResult(
                    candidate_id=item.candidate_id,
                    model_id=item.model_id,
                    checkpoint_id=item.checkpoint_id,
                    role="enhancement",
                    output_file_id=output_id,
                    output_url=f"/api/v1/files/{output_id}",
                    status="completed",
                    score=float(metrics.get("score", 0)),
                    planning_score=item.planning_score,
                    metrics=metrics,
                    parameters={**result.parameters, "role": "enhancement", "candidate_id": item.candidate_id},
                    runtime_ms=result.runtime_ms,
                    peak_memory_mb=result.peak_memory_mb,
                    is_mock=False,
                    adapter_class=result.adapter_class,
                    checkpoint_path=result.checkpoint_path,
                    checkpoint_sha256=result.checkpoint_sha256,
                    worker_python=external_python(),
                    device=result.device,
                    precision=result.precision,
                    input_path=source_input,
                    output_path=str(Path(result.output_path)),
                    input_sha256=result.input_sha256 or file_sha256(source_input),
                    output_sha256=result.output_sha256 or file_sha256(result.output_path),
                    input_size=input_size,
                    output_size=self._image_size(Path(result.output_path)),
                    warnings=list(result.logs or []),
                ))
            except Exception as exc:
                results.append(CandidateResult(
                    candidate_id=item.candidate_id,
                    model_id=item.model_id,
                    checkpoint_id=item.checkpoint_id,
                    role="enhancement",
                    status="failed",
                    planning_score=item.planning_score,
                    error=str(exc),
                    metrics={
                        "candidate_id": item.candidate_id,
                        "role": item.role,
                        "planning_score": item.planning_score,
                        "model_prior_score": item.model_prior_score,
                        "input_match_score": item.input_match_score,
                        "local_score": item.local_score,
                        "ai_semantic_bonus": item.ai_semantic_bonus,
                        "hardware_adjustment": item.hardware_adjustment,
                        "knowledge_adjustment": item.knowledge_adjustment,
                        "planning_evidence": [evidence.model_dump(mode="json") for evidence in item.planning_evidence],
                        "knowledge_evidence": [evidence.model_dump(mode="json") for evidence in item.knowledge_evidence],
                        "planning_reason": item.reason,
                        "input_policy": item.input_policy,
                        "output_exists": False,
                    },
                    input_path=source_input,
                    input_sha256=file_sha256(source_input),
                    input_size=input_size,
                    worker_python=external_python(),
                ))
        return results

    def _register_output_file(self, file_id: str, output_path: Path) -> None:
        with Image.open(output_path) as image:
            width, height = image.size
        try:
            relative_path = output_path.resolve().relative_to(self.settings.data_dir.resolve()).as_posix()
        except ValueError:
            return
        payload = {
            "file_id": file_id,
            "relative_path": relative_path,
            "sha256": file_sha256(output_path),
            "mime_type": "image/png",
            "width": width,
            "height": height,
            "size_bytes": output_path.stat().st_size,
            "uploaded_at": now_iso(),
        }
        self.db.put_file(file_id, payload)

    def _validity_metrics(self, input_path: Path, output_path: Path) -> dict:
        metrics = {"output_exists": output_path.exists(), "decode_ok": False, "size_match": False, "has_nan": False, "has_inf": False}
        if not output_path.exists():
            return metrics
        try:
            with Image.open(input_path) as src, Image.open(output_path) as out:
                src_size = src.size
                out_size = out.size
                out.convert("RGB")
            metrics.update({"decode_ok": True, "size_match": src_size == out_size, "width": out_size[0], "height": out_size[1]})
        except Exception as exc:
            metrics["decode_error"] = str(exc)
        return metrics

    def _image_size(self, path: Path) -> list[int] | None:
        try:
            with Image.open(path) as image:
                width, height = image.size
            return [int(width), int(height)]
        except Exception:
            return None
