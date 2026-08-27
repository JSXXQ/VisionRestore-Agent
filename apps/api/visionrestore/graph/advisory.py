from pathlib import Path

from visionrestore.ai.preview import create_ai_preview
from visionrestore.ai.providers import AIProviderError, DisabledAnalysisProvider, is_image_input_unsupported_error
from visionrestore.ai.registry import ProviderRegistry
from visionrestore.ai.validation import validate_multimodal_result
from visionrestore.core.config import get_settings
from visionrestore.schemas.ai import MultimodalAnalysisResult


class SemanticAdvisoryService:
    def analyze(
        self,
        *,
        input_path: Path,
        image_metrics,
        user_request: str,
        available_models: list[dict],
        hardware_summary: dict,
        analysis_mode: str,
        manual_model: str | None,
        manual_checkpoint: str | None,
        knowledge_context: list[dict] | None = None,
    ) -> MultimodalAnalysisResult:
        settings = get_settings()
        provider = ProviderRegistry().get()
        use_external = analysis_mode != "local" and settings.multimodal_analysis_enabled and provider.provider_id != "disabled"
        if not use_external:
            result = DisabledAnalysisProvider(settings).analyze(
                image_preview=None,
                image_metrics=image_metrics,
                user_request=user_request,
                available_models=available_models,
                hardware_summary=hardware_summary,
                analysis_mode=analysis_mode,
            )
            result.validation_passed = True
            result.local_validation = {"external_provider": "not_used", "final_authority": "local_rules"}
            result.adopted = False
            result.rejection_reason = "NO_EXTERNAL_MULTIMODAL_ADVICE"
            return result

        image_preview = None
        if analysis_mode == "multimodal" and settings.multimodal_send_image:
            image_preview, _ = create_ai_preview(input_path)
        try:
            return self._validated_call(
                provider=provider,
                image_preview=image_preview,
                image_metrics=image_metrics,
                user_request=user_request,
                available_models=available_models,
                hardware_summary=hardware_summary,
                analysis_mode=analysis_mode,
                manual_model=manual_model,
                manual_checkpoint=manual_checkpoint,
                knowledge_context=knowledge_context,
            )
        except AIProviderError as exc:
            retry_without_image = image_preview is not None and (
                is_image_input_unsupported_error(exc) or exc.code in {"AI_PROVIDER_TIMEOUT", "AI_PROVIDER_UNAVAILABLE"}
            )
            if retry_without_image:
                try:
                    result = self._validated_call(
                        provider=provider,
                        image_preview=None,
                        image_metrics=image_metrics,
                        user_request=user_request,
                        available_models=available_models,
                        hardware_summary=hardware_summary,
                        analysis_mode="text_only",
                        manual_model=manual_model,
                        manual_checkpoint=manual_checkpoint,
                        knowledge_context=knowledge_context,
                    )
                    result.warnings = [
                        "图像多模态调用失败，已自动改用文字/本地指标 AI 分析。",
                        f"原始多模态错误：{exc.message}",
                        *list(result.warnings or []),
                    ]
                    result.local_validation = {
                        **result.local_validation,
                        "image_mode_retry": "downgraded_to_text_only",
                        "image_mode_failure": exc.code,
                    }
                    return result
                except AIProviderError as retry_exc:
                    exc = retry_exc
            result = DisabledAnalysisProvider(settings).analyze(
                image_preview=None,
                image_metrics=image_metrics,
                user_request=user_request,
                available_models=available_models,
                hardware_summary=hardware_summary,
                analysis_mode=analysis_mode,
            )
            result.failure_reason = exc.code
            result.validation_passed = False
            result.validation_errors = [exc.code]
            result.local_validation = {"fallback": "local_rules"}
            result.adopted = False
            result.rejection_reason = exc.code
            result.warnings = [f"多模态 AI 调用失败，已回退本地规则分析：{exc.message}"]
            return result

    @staticmethod
    def _validated_call(**kwargs) -> MultimodalAnalysisResult:
        provider = kwargs.pop("provider")
        manual_model = kwargs.pop("manual_model")
        manual_checkpoint = kwargs.pop("manual_checkpoint")
        available_models = kwargs["available_models"]
        hardware_summary = kwargs["hardware_summary"]
        knowledge_context = kwargs.get("knowledge_context") or []
        result = provider.analyze(**kwargs)
        validated = validate_multimodal_result(
            result,
            available_models=available_models,
            hardware_summary=hardware_summary,
            manual_model=manual_model,
            manual_checkpoint=manual_checkpoint,
        )
        validated.retrieved_context = knowledge_context
        validated.prompt_context = {
            "knowledge_role": "LLM model capability reference only",
            "knowledge_reference_count": sum(
                item.get("source") == "model_roles" for item in knowledge_context
            ),
            "planning_fusion": {"local_weight": 0.5, "llm_weight": 0.5},
        }
        return validated
