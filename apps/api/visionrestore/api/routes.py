from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from visionrestore.adapters.registry import ModelRegistry
from visionrestore.core.config import get_settings
from visionrestore.schemas.common import ok
from visionrestore.schemas.task import TaskCreate
from visionrestore.services.hardware import HardwareInspector
from visionrestore.services.image_analyzer import ImageAnalyzer
from visionrestore.services.task_service import task_service
from visionrestore.storage.database import Database
from visionrestore.utils.file_security import UploadValidationError, resolve_registered_path, safe_image_upload

router = APIRouter(prefix="/api/v1")
db = Database()

@router.get("/health")
def health():
    return ok({"status": "ok", "service": "VisionRestore Agent"})

@router.get("/system")
def system():
    settings = get_settings()
    return ok({"hardware": HardwareInspector().inspect(), "settings": {"max_upload_mb": settings.max_upload_mb, "max_image_pixels": settings.max_image_pixels}})

@router.get("/models")
def models():
    return ok(ModelRegistry().list())

@router.get("/models/{model_id}")
def model_detail(model_id: str):
    try:
        return ok(ModelRegistry().get(model_id).get_status().model_dump())
    except KeyError:
        raise HTTPException(status_code=404, detail="模型不存在")

@router.post("/models/{model_id}/load")
def load_model(model_id: str):
    try:
        adapter = ModelRegistry().get(model_id)
        adapter.load()
        return ok(adapter.get_status().model_dump())
    except KeyError:
        raise HTTPException(status_code=404, detail="模型不存在")
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc))

@router.post("/models/{model_id}/unload")
def unload_model(model_id: str):
    try:
        adapter = ModelRegistry().get(model_id)
        adapter.unload()
        return ok(adapter.get_status().model_dump())
    except KeyError:
        raise HTTPException(status_code=404, detail="模型不存在")

@router.post("/models/{model_id}/install")
def install_model(model_id: str):
    try:
        adapter = ModelRegistry().get(model_id)
        return ok({"message": "本地安装由 scripts/download_models.ps1 执行；API 不会自动下载大型权重。", "hint": adapter.install_hint()})
    except KeyError:
        raise HTTPException(status_code=404, detail="模型不存在")

@router.post("/models/{model_id}/health-check")
def model_health(model_id: str):
    try:
        return ok(ModelRegistry().get(model_id).health_check())
    except KeyError:
        raise HTTPException(status_code=404, detail="模型不存在")

@router.post("/images/upload")
async def upload_image(file: UploadFile = File(...)):
    data = await file.read()
    try:
        record = safe_image_upload(file.filename or "image.png", data, file.content_type)
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.put_file(record.file_id, record.model_dump())
    return ok(record.model_dump())

@router.post("/images/analyze")
async def analyze_image(file: UploadFile = File(...)):
    data = await file.read()
    try:
        record = safe_image_upload(file.filename or "image.png", data, file.content_type)
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.put_file(record.file_id, record.model_dump())
    analysis = ImageAnalyzer().analyze(str(resolve_registered_path(record.relative_path)))
    return ok({"file": record.model_dump(), "analysis": analysis.model_dump()})

@router.post("/tasks")
def create_task(payload: TaskCreate):
    try:
        task = task_service.create(payload)
        return ok(task.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

@router.get("/tasks/{task_id}")
def get_task(task_id: str):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return ok(task.model_dump())

@router.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    task = task_service.cancel(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return ok(task.model_dump())

@router.get("/tasks/{task_id}/results")
def task_results(task_id: str):
    task = task_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return ok({"best_result": task.best_result.model_dump() if task.best_result else None, "candidates": [c.model_dump() for c in task.candidates]})

@router.get("/tasks/{task_id}/report")
def task_report(task_id: str):
    task = task_service.get(task_id)
    if not task or not task.report_file_id:
        raise HTTPException(status_code=404, detail="报告不存在")
    path = get_settings().report_dir / f"{task.report_file_id}.md"
    return FileResponse(path, media_type="text/markdown", filename=f"{task_id}.md")

@router.get("/history")
def history():
    return ok([task.model_dump() for task in task_service.list()])

@router.get("/history/{task_id}")
def history_detail(task_id: str):
    return get_task(task_id)

@router.delete("/history/{task_id}")
def delete_history(task_id: str):
    return ok({"deleted": db.delete_task(task_id)})

@router.get("/files/{file_id}")
def get_file(file_id: str):
    rec = db.get_file(file_id)
    if rec:
        path = resolve_registered_path(rec["relative_path"])
        return FileResponse(path)
    out = get_settings().output_dir / f"{file_id}.png"
    report_md = get_settings().report_dir / f"{file_id}.md"
    report_json = get_settings().report_dir / f"{file_id}.json"
    for path in [out, report_md, report_json]:
        if path.exists():
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="文件不存在或未登记")

@router.get("/settings")
def get_settings_api():
    return ok(db.get_settings())

@router.put("/settings")
def put_settings_api(payload: dict):
    db.set_settings(payload)
    return ok(db.get_settings())
