from pathlib import Path
from uuid import uuid4
from visionrestore.adapters.registry import ModelRegistry
from visionrestore.agent.intent_parser import IntentParser
from visionrestore.ai.preview import create_ai_preview
from visionrestore.ai.providers import AIProviderError, DisabledAnalysisProvider, is_image_input_unsupported_error
from visionrestore.ai.registry import ProviderRegistry
from visionrestore.ai.validation import validate_multimodal_result
from visionrestore.core.config import get_settings
from visionrestore.core.model_config import get_routing_rules
from visionrestore.routers.hierarchical_router import HierarchicalRouter
from visionrestore.schemas.ai import MultimodalAnalysisResult
from visionrestore.schemas.common import now_iso
from visionrestore.schemas.task import CandidateResult, EnhancementPlan, RetryRecord, TaskRecord
from visionrestore.services.evaluator import QualityEvaluator
from visionrestore.services.hardware import HardwareInspector
from visionrestore.services.image_analyzer import ImageAnalyzer
from visionrestore.services.report import ReportService
from visionrestore.utils.file_security import resolve_registered_path

class EnhancementAgent:
    def __init__(self):
        self.settings = get_settings()
        self.registry = ModelRegistry()
        self.analyzer = ImageAnalyzer()
        self.intent_parser = IntentParser()
        self.hardware = HardwareInspector()
        self.router = HierarchicalRouter()
        self.evaluator = QualityEvaluator()
        self.rules = get_routing_rules()

    def run(self, task: TaskRecord, input_file: dict, save_task):
        logs = task.logs
        def step(status: str, progress: float, message: str):
            task.status = status
            task.progress = progress
            logs.append(message)
            save_task(task)

        try:
            input_path = resolve_registered_path(input_file["relative_path"])
            step("analyzing", 0.10, "分析 RGB 静态图像退化统计。")
            analysis = self.analyzer.analyze(str(input_path))
            task.analysis = analysis

            step("parsing_intent", 0.18, "解析用户自然语言需求为结构化意图。")
            manual_model = task.plan.selected_model if task.plan else None
            manual_weight = task.plan.selected_checkpoint if task.plan else None
            intent = self.intent_parser.parse(task.user_goal, task.priority, task.mode, manual_model, manual_weight)
            task.user_intent = intent
            if intent.priority != task.priority and task.mode == "auto":
                task.priority = intent.priority

            step("inspecting_hardware", 0.26, "检查 CPU、GPU、CUDA、显存和模型状态。")
            hw = self.hardware.inspect()
            task.hardware_info = hw
            model_status = self.registry.list()

            step("routing_model", 0.32, "增强前执行多模态语义分析，并通过本地校验。")
            task.ai_analysis = self._run_ai_analysis(
                input_path=input_path,
                image_metrics=analysis,
                user_request=task.user_goal,
                available_models=model_status,
                hardware_summary=hw,
                analysis_mode=getattr(task, "analysis_mode", "local"),
                manual_model=manual_model,
                manual_checkpoint=manual_weight,
            )
            if task.ai_analysis.adopted:
                logs.append("多模态语义建议已在路由前通过校验，将按配置上限参与模型/权重评分。")
            elif task.ai_analysis.failure_reason or task.ai_analysis.rejection_reason:
                logs.append(f"多模态语义建议未参与评分：{task.ai_analysis.rejection_reason or task.ai_analysis.failure_reason}。")
            save_task(task)

            step("routing_model", 0.36, "执行第一层模型架构路由。")
            route = self.router.route(intent=intent, analysis=analysis, hardware=hw, model_statuses=model_status, mode=task.mode, semantic_analysis=task.ai_analysis)
            task.model_candidates = route.model_candidates
            step("routing_checkpoint", 0.44, "执行第二层权重路由。")
            task.checkpoint_candidates = route.checkpoint_candidates
            task.plan = EnhancementPlan(
                input_id=task.image_id,
                user_goal=task.user_goal,
                mode=task.mode,
                priority=task.priority,
                selected_model=route.selected_model,
                selected_checkpoint=route.selected_checkpoint,
                selection_reason=route.reason,
                parameters={"device": "cuda", "precision": self.settings.default_precision, "checkpoint_id": route.selected_checkpoint},
                preprocessing_steps=["RGB decode", "reflection padding if model size_multiple requires it"],
                postprocessing_steps=["crop to original size", "size assertion", "no-reference quality evaluation"],
                fallback_models=[m for m, _ in route.fallback[:1]],
                fallback_checkpoints=[c for _, c in route.fallback[:1]],
                expected_memory_mb=analysis.estimated_memory_mb,
                expected_runtime_ms=8000 if route.selected_model == "retinexformer" else 1200,
                expected_risks=["无参考指标仅供参考", "自动模式最多一次主推理加一次备用推理"],
            )
            run_list = [(route.selected_model, route.selected_checkpoint)]
            max_candidates = int(self.rules.get("fallback", {}).get("max_auto_candidates", 2))
            if task.mode == "compare":
                run_list = [(route.selected_model, route.selected_checkpoint)] + route.fallback[: max_candidates - 1]
            elif route.fallback:
                run_list.append(route.fallback[0])
            run_list = [x for x in run_list if x[0] and x[1]][:max_candidates]
            first_completed = False
            for index, (model_id, checkpoint_id) in enumerate(run_list):
                if task.status == "cancelled":
                    step("cancelled", task.progress, "任务已取消。")
                    return
                if index == 1 and first_completed and not self._needs_fallback(task.candidates[-1].metrics):
                    logs.append("主结果通过质量检查，跳过备用推理。")
                    break
                status = "fallback_running" if index == 1 else "loading_model"
                step(status, 0.50 + index * 0.18, f"准备模型 {model_id}:{checkpoint_id}。")
                try:
                    adapter = self.registry.get(model_id)
                    output_id = str(uuid4())
                    output_path = self.settings.output_dir / f"{output_id}.png"
                    step("running" if index == 0 else "fallback_running", 0.58 + index * 0.18, f"调用真实源码和真实权重推理 {model_id}:{checkpoint_id}。")
                    result = adapter.enhance(
                        image_path=str(input_path),
                        output_path=str(output_path),
                        device="cuda",
                        precision=self.settings.default_precision,
                        parameters={"checkpoint_id": checkpoint_id, "device": "cuda", "precision": self.settings.default_precision},
                    )
                    step("evaluating", 0.72 + index * 0.12, f"评价 {model_id}:{checkpoint_id} 输出并检查尺寸。")
                    metrics = self.evaluator.evaluate(str(input_path), result.output_path, task.priority, result.runtime_ms, result.peak_memory_mb)
                    cand = CandidateResult(
                        model_id=model_id,
                        checkpoint_id=checkpoint_id,
                        output_file_id=output_id,
                        output_url=f"/api/v1/files/{output_id}",
                        status="completed",
                        score=float(metrics["score"]),
                        metrics=metrics,
                        parameters=result.parameters,
                        runtime_ms=result.runtime_ms,
                        peak_memory_mb=result.peak_memory_mb,
                        is_mock=result.is_mock,
                        adapter_class=result.adapter_class,
                        checkpoint_path=result.checkpoint_path,
                        checkpoint_sha256=result.checkpoint_sha256,
                        device=result.device,
                        precision=result.precision,
                        input_sha256=result.input_sha256,
                        output_sha256=result.output_sha256,
                    )
                    logs.extend(result.logs)
                    task.candidates.append(cand)
                    first_completed = True
                    if index == 0 and len(run_list) > 1 and self._needs_fallback(metrics):
                        fb_model, fb_ck = run_list[1]
                        task.retries.append(RetryRecord(
                            previous_model=model_id,
                            previous_checkpoint=checkpoint_id,
                            previous_parameters=result.parameters,
                            previous_evaluation=metrics,
                            retry_reason="主结果触发备用策略",
                            fallback_model=fb_model,
                            fallback_checkpoint=fb_ck,
                            expected_improvement="降低过曝/偏色/增强不足风险",
                        ))
                        logs.append(f"主结果触发备用策略，准备尝试 {fb_model}:{fb_ck}。")
                except Exception as exc:
                    task.candidates.append(CandidateResult(model_id=model_id, checkpoint_id=checkpoint_id, status="failed", error=str(exc)))
                    logs.append(f"模型 {model_id}:{checkpoint_id} 失败：{exc}")
            step("selecting_result", 0.92, "从真实执行过的候选结果中选择推荐结果。")
            completed = [c for c in task.candidates if c.status == "completed"]
            if completed:
                task.best_result = sorted(completed, key=lambda c: c.score, reverse=True)[0]
                task.final_recommendation = f"推荐 {task.best_result.model_id}:{task.best_result.checkpoint_id}，综合评分 {task.best_result.score}。"
                step("completed", 1.0, task.final_recommendation)
            else:
                task.error = "没有候选模型完成真实推理。"
                step("failed", 0.95, task.error)
            task.completed_at = now_iso()
            report_id = str(uuid4())
            ReportService().write_reports(task.model_dump(), self.settings.report_dir / report_id)
            task.report_file_id = report_id
            save_task(task)
        except Exception as exc:
            task.status = "failed"
            task.error = str(exc)
            task.completed_at = now_iso()
            logs.append(f"任务失败：{exc}")
            save_task(task)

    def _run_ai_analysis(
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
            return self._validated_provider_analysis(
                provider=provider,
                image_preview=image_preview,
                image_metrics=image_metrics,
                user_request=user_request,
                available_models=available_models,
                hardware_summary=hardware_summary,
                analysis_mode=analysis_mode,
                manual_model=manual_model,
                manual_checkpoint=manual_checkpoint,
            )
        except AIProviderError as exc:
            if image_preview is not None and is_image_input_unsupported_error(exc):
                try:
                    result = self._validated_provider_analysis(
                        provider=provider,
                        image_preview=None,
                        image_metrics=image_metrics,
                        user_request=user_request,
                        available_models=available_models,
                        hardware_summary=hardware_summary,
                        analysis_mode="text_only",
                        manual_model=manual_model,
                        manual_checkpoint=manual_checkpoint,
                    )
                    result.warnings = [
                        "当前配置的模型不支持图像输入，已自动改用文字/本地指标 AI 分析。",
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

    def _validated_provider_analysis(
        self,
        *,
        provider,
        image_preview,
        image_metrics,
        user_request: str,
        available_models: list[dict],
        hardware_summary: dict,
        analysis_mode: str,
        manual_model: str | None,
        manual_checkpoint: str | None,
    ) -> MultimodalAnalysisResult:
        result = provider.analyze(
            image_preview=image_preview,
            image_metrics=image_metrics,
            user_request=user_request,
            available_models=available_models,
            hardware_summary=hardware_summary,
            analysis_mode=analysis_mode,
        )
        return validate_multimodal_result(
            result,
            available_models=available_models,
            hardware_summary=hardware_summary,
            manual_model=manual_model,
            manual_checkpoint=manual_checkpoint,
        )

    def _needs_fallback(self, metrics: dict) -> bool:
        fb = self.rules.get("fallback", {})
        if not metrics:
            return True
        if metrics.get("score", 0) < fb.get("min_score_to_accept", 42):
            return True
        if metrics.get("mean_luminance_after", 0) < fb.get("still_dark_mean_luminance", 55) and metrics.get("dark_pixel_ratio_after", 0) > fb.get("still_dark_ratio", 0.55):
            return True
        if metrics.get("overexposed_pixel_ratio_after", 0) - metrics.get("overexposed_pixel_ratio_before", 0) > fb.get("overexposure_increase", 0.035):
            return True
        if metrics.get("color_cast_index_after", 0) - metrics.get("color_cast_index_before", 0) > fb.get("color_cast_increase", 0.12):
            return True
        if metrics.get("noise_estimate_after", 0) - metrics.get("noise_estimate_before", 0) > fb.get("noise_increase", 18):
            return True
        return False
