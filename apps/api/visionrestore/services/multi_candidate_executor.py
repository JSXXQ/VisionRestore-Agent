from pathlib import Path
from uuid import uuid4

from visionrestore.core.config import get_settings
from visionrestore.core.model_config import file_sha256
from visionrestore.schemas.candidate import CandidateExecutionPlan, CandidatePlanItem
from visionrestore.schemas.task import CandidateResult
from visionrestore.services.evaluator import QualityEvaluator


class MultiCandidateExecutor:
    def __init__(self, registry=None, evaluator=None):
        from visionrestore.adapters.registry import ModelRegistry

        self.registry = registry or ModelRegistry()
        self.evaluator = evaluator or QualityEvaluator()
        self.settings = get_settings()

    def execute(self, *, plan: CandidateExecutionPlan, input_path: str, priority: str = "balanced", task_dir: str | Path | None = None, cancel_check=None) -> list[CandidateResult]:
        source_input = str(input_path)
        output_root = Path(task_dir) if task_dir else self.settings.output_dir
        output_root.mkdir(parents=True, exist_ok=True)
        results: list[CandidateResult] = []
        for item in plan.candidates:
            if cancel_check and cancel_check():
                break
            candidate_dir = output_root / item.candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            output_id = str(uuid4())
            output_path = candidate_dir / f"{output_id}.png"
            try:
                adapter = self.registry.get(item.model_id)
                result = adapter.enhance(
                    image_path=source_input,
                    output_path=str(output_path),
                    device="cuda",
                    precision="fp32",
                    parameters={"checkpoint_id": item.checkpoint_id},
                )
                if result.is_mock:
                    raise RuntimeError("Mock result is not allowed in formal multi-candidate execution")
                metrics = self.evaluator.evaluate(source_input, result.output_path, priority, result.runtime_ms, result.peak_memory_mb)
                results.append(CandidateResult(
                    model_id=item.model_id,
                    checkpoint_id=item.checkpoint_id,
                    output_file_id=output_id,
                    output_url=f"/api/v1/files/{output_id}",
                    status="completed",
                    score=float(metrics.get("score", 0)),
                    metrics={**metrics, "candidate_id": item.candidate_id, "input_policy": item.input_policy},
                    parameters=result.parameters,
                    runtime_ms=result.runtime_ms,
                    peak_memory_mb=result.peak_memory_mb,
                    is_mock=False,
                    adapter_class=result.adapter_class,
                    checkpoint_path=result.checkpoint_path,
                    checkpoint_sha256=result.checkpoint_sha256,
                    device=result.device,
                    precision=result.precision,
                    input_sha256=result.input_sha256 or file_sha256(source_input),
                    output_sha256=result.output_sha256 or file_sha256(result.output_path),
                ))
            except Exception as exc:
                results.append(CandidateResult(
                    model_id=item.model_id,
                    checkpoint_id=item.checkpoint_id,
                    status="failed",
                    error=str(exc),
                    metrics={"candidate_id": item.candidate_id, "input_policy": item.input_policy},
                ))
        return results
