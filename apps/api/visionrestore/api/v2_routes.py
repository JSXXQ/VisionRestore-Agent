from fastapi import APIRouter, File, HTTPException, UploadFile

from visionrestore.adapters.registry import ModelRegistry
from visionrestore.core.config import get_settings
from visionrestore.schemas.common import ok
from visionrestore.schemas.task import TaskCreate
from visionrestore.services.hardware import HardwareInspector
from visionrestore.services.image_analyzer import ImageAnalyzer
from visionrestore.services.artifact_lineage import ArtifactLineageService
from visionrestore.services.candidate_evaluator import CandidateRanker
from visionrestore.services.residual_analyzer import ResidualDegradationAnalyzer
from visionrestore.services.postprocess_controller import PostprocessController
from visionrestore.schemas.postprocess import PostprocessDecision
from visionrestore.services.task_service import task_service
from visionrestore.storage.database import Database
from visionrestore.utils.file_security import UploadValidationError, resolve_registered_path, safe_image_upload

router = APIRouter(prefix="/api/v2")
db = Database()


@router.get("/health")
def health_v2():
    return ok({"status": "ok", "service": "VisionRestore Agent", "api_version": "v2"})


@router.get("/system")
def system_v2():
    settings = get_settings()
    return ok({
        "hardware": HardwareInspector().inspect(),
        "settings": {
            "max_upload_mb": settings.max_upload_mb,
            "max_image_pixels": settings.max_image_pixels,
            "mock_models_enabled": settings.mock_models_enabled,
        },
    })


@router.get("/models")
def models_v2():
    registry = ModelRegistry()
    items = registry.list()
    return ok({
        "models": items,
        "groups": {
            "enhancement": registry.enhancement_model_ids(),
            "postprocess": registry.postprocess_model_ids(),
        },
        "ready_count": len([item for item in items if item.get("available") and item["model_id"] != "mock_model"]),
    })


@router.post("/models/scan")
def scan_models_v2():
    registry = ModelRegistry()
    return ok({"models": registry.list(), "message": "已重新扫描本地配置；未通过健康检查的模型不会显示ready。"})


@router.post("/models/{model_id}/health-check")
def model_health_v2(model_id: str):
    try:
        return ok(ModelRegistry().get(model_id).health_check())
    except KeyError:
        raise HTTPException(status_code=404, detail="模型不存在")


@router.post("/images/upload")
async def upload_image_v2(file: UploadFile = File(...)):
    data = await file.read()
    try:
        record = safe_image_upload(file.filename or "image.png", data, file.content_type)
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.put_file(record.file_id, record.model_dump())
    return ok(record.model_dump())


@router.post("/images/analyze")
async def analyze_image_v2(file: UploadFile = File(...)):
    data = await file.read()
    try:
        record = safe_image_upload(file.filename or "image.png", data, file.content_type)
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.put_file(record.file_id, record.model_dump())
    analysis = ImageAnalyzer().analyze(str(resolve_registered_path(record.relative_path)))
    return ok({"file": record.model_dump(), "analysis": analysis.model_dump()})

@router.post("/tasks")
def create_task_v2(payload: TaskCreate):
    try:
        task = task_service.create(payload)
        return ok(task.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/tasks/{task_id}")
def get_task_v2(task_id: str):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return ok(task.model_dump())


@router.post("/tasks/{task_id}/cancel")
def cancel_task_v2(task_id: str):
    task = task_service.cancel(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return ok(task.model_dump())


@router.get("/tasks/{task_id}/plan")
def task_plan_v2(task_id: str):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return ok({"plan": task.plan.model_dump() if task.plan else None, "model_candidates": task.model_candidates, "checkpoint_candidates": task.checkpoint_candidates})


@router.get("/tasks/{task_id}/candidates")
def task_candidates_v2(task_id: str):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return ok({"candidates": [item.model_dump() for item in task.candidates], "best_result": task.best_result.model_dump() if task.best_result else None})


@router.get("/tasks/{task_id}/ranking")
def task_ranking_v2(task_id: str):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    candidates = [item.model_dump() for item in task.candidates]
    ranking = CandidateRanker().rank(candidates, task.priority)
    return ok({key: (value.__dict__ if hasattr(value, "__dict__") else [item.__dict__ for item in value] if isinstance(value, list) else value) for key, value in ranking.items()})


@router.post("/tasks/{task_id}/select-candidate")
def select_candidate_v2(task_id: str, payload: dict):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    output_file_id = payload.get("output_file_id")
    selected = next((item for item in task.candidates if item.output_file_id == output_file_id), None)
    if not selected:
        raise HTTPException(status_code=404, detail="候选结果不存在")
    task.best_result = selected
    task.final_recommendation = f"用户手动选择 {selected.model_id}:{selected.checkpoint_id} 作为后处理输入。"
    task.logs.append(task.final_recommendation)
    task_service.save(task)
    return ok(task.best_result.model_dump())


@router.get("/tasks/{task_id}/recommendations")
def task_recommendations_v2(task_id: str):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not task.best_result or not task.analysis:
        return ok({"recommendation": None, "message": "任务尚未产生最佳增强结果。"})
    recommendation = ResidualDegradationAnalyzer().analyze(
        original_analysis=task.analysis,
        enhanced_metrics=task.best_result.metrics,
        best_candidate=task.best_result.model_dump(),
        user_intent=task.user_intent,
        hardware=task.hardware_info or {},
    )
    return ok({"recommendation": recommendation.model_dump()})


@router.post("/tasks/{task_id}/postprocess/decision")
def postprocess_decision_v2(task_id: str, payload: dict):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    operation = payload.get("operation")
    decision = payload.get("decision")
    task.logs.append(f"后处理决策已记录：{operation}={decision}。真实执行将在对应worker ready后启用。")
    task_service.save(task)
    return ok({"accepted": True, "operation": operation, "decision": decision, "executed": False, "message": "已记录决策；当前阶段不伪造后处理执行。"})


@router.get("/tasks/{task_id}/artifacts")
def task_artifacts_v2(task_id: str):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return ok(ArtifactLineageService().from_task(task).model_dump())