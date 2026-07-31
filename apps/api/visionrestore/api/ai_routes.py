from fastapi import APIRouter, HTTPException

from visionrestore.adapters.registry import ModelRegistry
from visionrestore.ai.preview import create_ai_preview
from visionrestore.ai.providers import AIProviderError, DisabledAnalysisProvider
from visionrestore.ai.registry import ProviderRegistry
from visionrestore.core.config import get_settings
from visionrestore.schemas.ai import AIAnalyzeRequest, AISettingsPublic, AISettingsUpdate
from visionrestore.schemas.common import ok
from visionrestore.services.hardware import HardwareInspector
from visionrestore.services.image_analyzer import ImageAnalyzer
from visionrestore.storage.database import Database
from visionrestore.utils.file_security import resolve_registered_path

router = APIRouter(prefix="/api/v1/ai")
db = Database()


def _public_settings() -> AISettingsPublic:
    settings = get_settings()
    registry = ProviderRegistry()
    statuses = {item["provider_id"]: item for item in registry.list_statuses()}
    providers = {
        provider_id: {
            "implemented": item["implemented"],
            "configured": item["configured"],
            "healthy": item["healthy"],
            "supports_image": item["supports_image"],
            "current_model": item["current_model"],
            "last_error": item["last_error"],
            "error_code": item["error_code"],
        }
        for provider_id, item in statuses.items()
    }
    return AISettingsPublic(
        enabled=settings.multimodal_analysis_enabled,
        provider=settings.multimodal_provider,
        send_image=settings.multimodal_send_image,
        send_metrics=settings.multimodal_send_metrics,
        image_max_edge=settings.multimodal_image_max_edge,
        image_quality=settings.multimodal_image_quality,
        timeout_seconds=settings.multimodal_timeout_seconds,
        max_retries=settings.multimodal_max_retries,
        fallback_to_local=settings.multimodal_fallback_to_local,
        remove_metadata=settings.multimodal_remove_metadata,
        providers=providers,
    )


@router.get("/providers")
def providers():
    return ok(ProviderRegistry().list_statuses())


@router.post("/providers/{provider_id}/health-check")
def provider_health(provider_id: str):
    try:
        provider = ProviderRegistry().get_by_id(provider_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="AI provider does not exist")
    return ok(provider.health_check(run_remote=True).model_dump())


@router.get("/settings")
def get_ai_settings():
    return ok(_public_settings().model_dump())


@router.put("/settings")
def put_ai_settings(payload: AISettingsUpdate):
    data = payload.model_dump(exclude_none=True)
    db.set_settings({"ai": data})
    current = _public_settings().model_dump()
    current["saved_overrides"] = data
    return ok(current)


@router.post("/analyze")
def analyze(payload: AIAnalyzeRequest):
    settings = get_settings()
    file_record = db.get_file(payload.image_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="输入图像不存在")
    input_path = resolve_registered_path(file_record["relative_path"])
    image_metrics = ImageAnalyzer().analyze(str(input_path))
    available_models = ModelRegistry().list()
    hardware_summary = HardwareInspector().inspect()
    registry = ProviderRegistry()
    provider = registry.get(payload.provider_id)
    use_external = payload.analysis_mode != "local" and settings.multimodal_analysis_enabled and provider.provider_id != "disabled"
    if not use_external:
        result = DisabledAnalysisProvider(settings).analyze(
            image_preview=None,
            image_metrics=image_metrics,
            user_request=payload.user_request,
            available_models=available_models,
            hardware_summary=hardware_summary,
            analysis_mode=payload.analysis_mode,
        )
        if payload.analysis_mode != "local" and not settings.multimodal_analysis_enabled:
            result.failure_reason = "MULTIMODAL_ANALYSIS_DISABLED"
            result.warnings = ["多模态 AI 未启用，已回退本地规则分析。"]
        return ok({"analysis": result.model_dump(), "preview": None})
    preview = None
    image_preview = None
    if payload.analysis_mode == "multimodal" and settings.multimodal_send_image:
        image_preview, preview = create_ai_preview(input_path)
    try:
        result = provider.analyze(
            image_preview=image_preview,
            image_metrics=image_metrics,
            user_request=payload.user_request,
            available_models=available_models,
            hardware_summary=hardware_summary,
            analysis_mode=payload.analysis_mode,
        )
        return ok({"analysis": result.model_dump(), "preview": preview})
    except AIProviderError as exc:
        if not settings.multimodal_fallback_to_local:
            raise HTTPException(status_code=503, detail={"code": exc.code, "message": exc.message})
        result = DisabledAnalysisProvider(settings).analyze(
            image_preview=None,
            image_metrics=image_metrics,
            user_request=payload.user_request,
            available_models=available_models,
            hardware_summary=hardware_summary,
            analysis_mode=payload.analysis_mode,
        )
        result.failure_reason = exc.code
        result.warnings = [f"多模态 AI 调用失败，已回退本地规则分析：{exc.message}"]
        return ok({"analysis": result.model_dump(), "preview": preview})


@router.post("/test")
def test_current_provider():
    provider = ProviderRegistry().get()
    return ok(provider.health_check(run_remote=True).model_dump())
