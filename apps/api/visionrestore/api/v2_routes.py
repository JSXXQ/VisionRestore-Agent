from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse
from uuid import uuid4

from visionrestore.adapters.registry import ModelRegistry
from visionrestore.core.config import get_settings
from visionrestore.schemas.common import ok
from visionrestore.schemas.postprocess import PostprocessDecision
from visionrestore.schemas.task import TaskCreate
from visionrestore.services.artifact_lineage import ArtifactLineageService
from visionrestore.services.hardware import HardwareInspector
from visionrestore.services.image_analyzer import ImageAnalyzer
from visionrestore.services.iqa_service import IQAService
from visionrestore.services.postprocess_controller import PostprocessController
from visionrestore.services.realesrgan_service import RealESRGANStatusService
from visionrestore.services.report import ReportService
from visionrestore.services.residual_analyzer import ResidualDegradationAnalyzer
from visionrestore.services.result_selector import ResultSelector
from visionrestore.services.task_service import task_service
from visionrestore.storage.database import Database
from visionrestore.utils.file_security import UploadValidationError, resolve_registered_path, safe_image_upload

router = APIRouter(prefix="/api/v2")
db = Database()


def _refresh_task_report(task):
    if not task.report_file_id:
        task.report_file_id = str(uuid4())
    ReportService().write_reports(task.model_dump(), get_settings().report_dir / task.report_file_id)


def _apply_postprocess_decision(task, decision: PostprocessDecision):
    if task.workflow_engine == "langgraph":
        try:
            _latest, result = task_service.resume_postprocess(task.task_id, decision)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return result

    result = PostprocessController().decide(task=task, decision=decision)
    if result.next_status:
        task.status = result.next_status
    task.logs.append(result.message)
    if result.executed:
        _refresh_task_report(task)
    task_service.save(task)
    db.put_entity(
        "postprocess_decision",
        f"{task.task_id}:{decision.operation}:{len(task.logs)}",
        {"decision": decision.model_dump(), "result": result.model_dump()},
        task_id=task.task_id,
    )
    if result.executed:
        db.put_entity(
            "postprocess_result",
            f"{task.task_id}:{decision.operation}:latest",
            result.model_dump(),
            task_id=task.task_id,
        )
    return result


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
        result = ModelRegistry().get(model_id).health_check()
        record_id = f"{model_id}:{len(db.list_entities('model_health_record')) + 1}"
        db.put_entity("model_health_record", record_id, result, task_id=None)
        return ok(result)
    except KeyError:
        raise HTTPException(status_code=404, detail="模型不存在")


@router.get("/iqa/status")
def iqa_status_v2():
    return ok(IQAService().status())


@router.get("/models/realesrgan/status")
def realesrgan_status_v2():
    return ok(RealESRGANStatusService().status())


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
    candidates = [item.model_dump() for item in task.candidates]
    selection = ResultSelector().select(candidates, task.priority)
    refreshed_scores = {
        item["candidate_id"]: item
        for item in [*(selection.successful or []), *(selection.failed or [])]
        if item.get("candidate_id")
    }
    refreshed_candidates = []
    for candidate in candidates:
        score = refreshed_scores.get(candidate.get("candidate_id"))
        if score:
            candidate["score"] = score.get("score")
            candidate["final_score"] = score.get("score")
            metrics = candidate.get("metrics") or {}
            metrics["final_score"] = score.get("score")
            metrics["score_layers"] = score.get("layers") or {}
            metrics["score_reasons"] = score.get("reasons") or []
            metrics["score_evidence"] = score.get("evidence") or {}
            metrics["planning_score_is_not_final_score"] = True
            candidate["metrics"] = metrics
        refreshed_candidates.append(candidate)
    selected_id = (selection.selected or {}).get("candidate_id") if selection.selected else None
    best_result = next((item for item in refreshed_candidates if item.get("candidate_id") == selected_id), None)
    return ok({"candidates": refreshed_candidates, "best_result": best_result})


@router.get("/tasks/{task_id}/candidates/{candidate_id}/iqa")
def task_candidate_iqa_v2(task_id: str, candidate_id: str):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return ok(IQAService().evaluate_candidate(task=task, candidate_id=candidate_id))


@router.get("/tasks/{task_id}/ranking")
def task_ranking_v2(task_id: str):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    selection = ResultSelector().select([item.model_dump() for item in task.candidates], task.priority)
    payload = selection.model_dump()
    db.put_entity("candidate_ranking", f"{task_id}:latest", payload, task_id=task_id)
    return ok(payload)


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
    payload = recommendation.model_dump()
    db.put_entity("postprocess_recommendation", f"{task_id}:latest", payload, task_id=task_id)
    return ok({"recommendation": payload})


@router.post("/tasks/{task_id}/postprocess/decision")
def postprocess_decision_v2(task_id: str, payload: dict):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    decision = PostprocessDecision.model_validate(payload)
    result = _apply_postprocess_decision(task, decision)
    return ok(result.model_dump())




@router.post("/tasks/{task_id}/super-resolution/confirm")
def super_resolution_confirm_v2(task_id: str, payload: dict | None = None):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    payload = payload or {}
    parameters = dict(payload.get("parameters") or {})
    for key in ("checkpoint_id", "weight_id"):
        if payload.get(key):
            parameters[key] = payload.get(key)
    if payload.get("allow_x4") is not None:
        parameters["allow_x4"] = bool(payload.get("allow_x4"))
    scale = int(payload.get("scale") or parameters.get("scale") or 2)
    parameters["scale"] = scale
    decision = PostprocessDecision(operation="super_resolution", decision="choose_model" if payload.get("checkpoint_id") else "accept", model_id="realesrgan", scale=scale, parameters=parameters)
    result = _apply_postprocess_decision(task, decision)
    return ok(result.model_dump())


@router.post("/tasks/{task_id}/super-resolution/skip")
def super_resolution_skip_v2(task_id: str):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    decision = PostprocessDecision(operation="super_resolution", decision="skip", model_id="realesrgan", parameters={})
    result = _apply_postprocess_decision(task, decision)
    return ok(result.model_dump())


@router.get("/tasks/{task_id}/workflow")
def task_workflow_v2(task_id: str):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return ok(
        {
            "engine": task.workflow_engine,
            "version": task.workflow_version,
            "thread_id": task.workflow_thread_id,
            "status": task.status,
            "pending_confirmation": task.pending_confirmation,
            "messages": task.messages,
            "events": db.list_entities("workflow_event", task_id=task_id),
            "runs": db.list_entities("graph_run", task_id=task_id),
        }
    )


@router.post('/tasks/{task_id}/postprocess/rollback')
def postprocess_rollback_v2(task_id: str):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail='Task not found')
    result = PostprocessController().rollback(task=task)
    task.logs.append(result.message)
    if result.rollback_performed:
        _refresh_task_report(task)
    task_service.save(task)
    db.put_entity('postprocess_decision', f'{task_id}:rollback:{len(task.logs)}', {'result': result.model_dump()}, task_id=task_id)
    return ok(result.model_dump())

@router.get("/tasks/{task_id}/artifacts")
def task_artifacts_v2(task_id: str):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return ok(ArtifactLineageService().from_task(task).model_dump())


@router.get("/tasks/{task_id}/report")
def task_report_v2(task_id: str):
    task = task_service.get(task_id)
    if not task or not task.report_file_id:
        raise HTTPException(status_code=404, detail="报告不存在")
    path = get_settings().report_dir / f"{task.report_file_id}.md"
    return FileResponse(path, media_type="text/markdown", filename=f"{task_id}.md")


@router.websocket("/tasks/{task_id}/stream")
async def task_stream_v2(websocket: WebSocket, task_id: str):
    import asyncio
    await websocket.accept()
    last = None
    for _ in range(600):
        task = task_service.get(task_id)
        if not task:
            await websocket.send_json({"success": False, "message": "任务不存在"})
            break
        payload = task.model_dump()
        if payload != last:
            await websocket.send_json({"success": True, "data": payload})
            last = payload
        if task.status in {"completed", "failed", "cancelled"}:
            break
        await asyncio.sleep(0.5)
    await websocket.close()
