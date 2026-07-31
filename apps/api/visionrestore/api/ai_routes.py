from pathlib import Path

from fastapi import APIRouter, HTTPException

from visionrestore.adapters.registry import ModelRegistry
from visionrestore.ai.preview import create_ai_preview
from visionrestore.ai.providers import AIProviderError, DisabledAnalysisProvider, is_image_input_unsupported_error
from visionrestore.ai.registry import ProviderRegistry
from visionrestore.ai.validation import validate_multimodal_result
from visionrestore.core.config import PROJECT_ROOT, get_settings
from visionrestore.schemas.ai import AIAnalyzeRequest, AIProviderConfigUpdate, AISettingsPublic, AISettingsUpdate
from visionrestore.schemas.common import ok
from visionrestore.services.hardware import HardwareInspector
from visionrestore.services.image_analyzer import ImageAnalyzer
from visionrestore.storage.database import Database
from visionrestore.utils.file_security import resolve_registered_path

router = APIRouter(prefix="/api/v1/ai")
db = Database()

PROVIDER_ENV = {
    "openai": {
        "api_key": "OPENAI_API_KEY",
        "model": "OPENAI_MODEL",
        "base_url": "OPENAI_BASE_URL",
    },
    "openai_compatible": {
        "api_key": "OPENAI_COMPATIBLE_API_KEY",
        "model": "OPENAI_COMPATIBLE_MODEL",
        "base_url": "OPENAI_COMPATIBLE_BASE_URL",
    },
    "anthropic": {
        "api_key": "ANTHROPIC_API_KEY",
        "model": "ANTHROPIC_MODEL",
        "base_url": None,
    },
    "gemini": {
        "api_key": "GEMINI_API_KEY",
        "model": "GEMINI_MODEL",
        "base_url": None,
    },
}


def _write_env(updates: dict[str, str]) -> None:
    env_path = PROJECT_ROOT / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if not line or line.lstrip().startswith("#") or "=" not in line:
            out.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")
    env_path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    get_settings.cache_clear()


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


@router.put("/providers/{provider_id}/config")
def configure_provider(provider_id: str, payload: AIProviderConfigUpdate):
    if provider_id not in ProviderRegistry().providers:
        raise HTTPException(status_code=404, detail="AI provider does not exist")
    if provider_id == "disabled":
        updates = {
            "MULTIMODAL_ANALYSIS_ENABLED": "false",
            "MULTIMODAL_PROVIDER": "disabled",
        }
        _write_env(updates)
        return ok(_public_settings().model_dump())
    mapping = PROVIDER_ENV.get(provider_id)
    if not mapping:
        raise HTTPException(status_code=400, detail="AI provider cannot be configured")
    updates: dict[str, str] = {}
    updates["MULTIMODAL_PROVIDER"] = payload.provider or provider_id
    if payload.enabled is not None:
        updates["MULTIMODAL_ANALYSIS_ENABLED"] = str(payload.enabled).lower()
    if payload.send_image is not None:
        updates["MULTIMODAL_SEND_IMAGE"] = str(payload.send_image).lower()
    if payload.send_metrics is not None:
        updates["MULTIMODAL_SEND_METRICS"] = str(payload.send_metrics).lower()
    if payload.fallback_to_local is not None:
        updates["MULTIMODAL_FALLBACK_TO_LOCAL"] = str(payload.fallback_to_local).lower()
    if payload.model is not None and mapping["model"]:
        updates[mapping["model"]] = payload.model.strip()
    if payload.base_url is not None and mapping["base_url"]:
        updates[mapping["base_url"]] = payload.base_url.strip()
    if payload.clear_api_key and mapping["api_key"]:
        updates[mapping["api_key"]] = ""
    elif payload.api_key and mapping["api_key"]:
        updates[mapping["api_key"]] = payload.api_key.strip()
    _write_env(updates)
    return ok(_public_settings().model_dump())


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
        result.validation_passed = True
        result.local_validation = {
            "external_provider": "not_used",
            "final_authority": "local_rules",
        }
        result.adopted = False
        result.rejection_reason = "NO_EXTERNAL_MULTIMODAL_ADVICE"
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
        result = validate_multimodal_result(
            result,
            available_models=available_models,
            hardware_summary=hardware_summary,
            manual_model=payload.manual_model,
            manual_checkpoint=payload.manual_checkpoint,
        )
        return ok({"analysis": result.model_dump(), "preview": preview})
    except AIProviderError as exc:
        if image_preview is not None and is_image_input_unsupported_error(exc):
            try:
                result = provider.analyze(
                    image_preview=None,
                    image_metrics=image_metrics,
                    user_request=payload.user_request,
                    available_models=available_models,
                    hardware_summary=hardware_summary,
                    analysis_mode="text_only",
                )
                result.warnings = [
                    "当前配置的模型不支持图像输入，已自动改用文字/本地指标 AI 分析。",
                    *list(result.warnings or []),
                ]
                result = validate_multimodal_result(
                    result,
                    available_models=available_models,
                    hardware_summary=hardware_summary,
                    manual_model=payload.manual_model,
                    manual_checkpoint=payload.manual_checkpoint,
                )
                result.local_validation = {
                    **result.local_validation,
                    "image_mode_retry": "downgraded_to_text_only",
                    "image_mode_failure": exc.code,
                }
                return ok({"analysis": result.model_dump(), "preview": preview})
            except AIProviderError as retry_exc:
                exc = retry_exc
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
        result.validation_passed = False
        result.validation_errors = [exc.code]
        result.local_validation = {
            "json_parse": "failed" if exc.code == "AI_PROVIDER_INVALID_RESPONSE" else "not_reached",
            "pydantic_schema": "failed" if exc.code == "AI_PROVIDER_INVALID_RESPONSE" else "not_reached",
            "fallback": "local_rules",
        }
        result.adopted = False
        result.rejection_reason = exc.code
        result.warnings = [f"多模态 AI 调用失败，已回退本地规则分析：{exc.message}"]
        return ok({"analysis": result.model_dump(), "preview": preview})


@router.post("/test")
def test_current_provider():
    provider = ProviderRegistry().get()
    return ok(provider.health_check(run_remote=True).model_dump())
