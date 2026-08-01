from fastapi import APIRouter, File, HTTPException, UploadFile

from visionrestore.adapters.registry import ModelRegistry
from visionrestore.core.config import get_settings
from visionrestore.schemas.common import ok
from visionrestore.services.hardware import HardwareInspector
from visionrestore.services.image_analyzer import ImageAnalyzer
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
