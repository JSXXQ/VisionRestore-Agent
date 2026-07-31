from pathlib import Path
from uuid import uuid4
from visionrestore.adapters.registry import ModelRegistry
from visionrestore.agent.intent_parser import IntentParser
from visionrestore.core.config import get_settings
from visionrestore.core.model_config import get_routing_rules
from visionrestore.routers.hierarchical_router import HierarchicalRouter
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

            step("routing_model", 0.34, "执行第一层模型架构路由。")
            route = self.router.route(intent=intent, analysis=analysis, hardware=hw, model_statuses=model_status, mode=task.mode)
            task.model_candidates = route.model_candidates
            step("routing_checkpoint", 0.42, "执行第二层权重路由。")
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
