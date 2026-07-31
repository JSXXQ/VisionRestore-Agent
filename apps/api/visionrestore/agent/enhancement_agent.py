from pathlib import Path
from uuid import uuid4
from visionrestore.adapters.registry import ModelRegistry
from visionrestore.agent.planner import DeterministicPlanner
from visionrestore.schemas.common import now_iso
from visionrestore.schemas.task import CandidateResult, TaskRecord
from visionrestore.services.evaluator import QualityEvaluator
from visionrestore.services.hardware import HardwareInspector
from visionrestore.services.image_analyzer import ImageAnalyzer
from visionrestore.services.report import ReportService
from visionrestore.utils.file_security import resolve_registered_path
from visionrestore.core.config import get_settings

class EnhancementAgent:
    def __init__(self):
        self.settings = get_settings()
        self.registry = ModelRegistry()
        self.analyzer = ImageAnalyzer()
        self.hardware = HardwareInspector()
        self.planner = DeterministicPlanner()
        self.evaluator = QualityEvaluator()

    def run(self, task: TaskRecord, input_file: dict, save_task):
        logs = task.logs
        def step(status: str, progress: float, message: str):
            task.status = status
            task.progress = progress
            logs.append(message)
            save_task(task)

        try:
            input_path = resolve_registered_path(input_file["relative_path"])
            step("analyzing", 0.12, "分析 RGB 静态图像退化统计。")
            analysis = self.analyzer.analyze(str(input_path))
            task.analysis = analysis
            step("planning", 0.25, "检查硬件、模型状态并生成确定性增强计划。")
            hw = self.hardware.inspect()
            model_status = self.registry.list()
            task.plan = self.planner.plan(
                image_id=task.image_id,
                user_goal=task.user_goal,
                mode=task.mode,
                priority=task.priority,
                analysis=analysis,
                hardware=hw,
                models=model_status,
                model_id=(task.plan.selected_model if task.plan else None),
                parameters={},
            )
            candidates = [task.plan.selected_model]
            if task.mode == "compare":
                candidates = [m["model_id"] for m in model_status if m["available"]]
            candidates += [m for m in task.plan.fallback_models if m not in candidates]
            if not candidates:
                candidates = [task.plan.selected_model]
            for idx, model_id in enumerate(candidates[:4]):
                if task.status == "cancelled":
                    step("cancelled", task.progress, "任务已取消。")
                    return
                step("loading_model", 0.35 + idx * 0.08, f"准备模型 {model_id}。")
                try:
                    adapter = self.registry.get(model_id)
                    output_id = str(uuid4())
                    output_path = self.settings.output_dir / f"{output_id}.png"
                    step("running", 0.45 + idx * 0.08, f"调用模型适配器 {model_id} 执行真实推理。")
                    result = adapter.enhance(
                        image_path=str(input_path),
                        output_path=str(output_path),
                        device=task.plan.parameters.get("device", self.settings.default_device),
                        precision=self.settings.default_precision,
                        parameters=adapter.validate_parameters(task.plan.parameters),
                    )
                    step("evaluating", 0.72 + idx * 0.05, f"评价模型 {model_id} 输出。")
                    metrics = self.evaluator.evaluate(str(input_path), result.output_path, task.priority)
                    cand = CandidateResult(
                        model_id=model_id,
                        output_file_id=output_id,
                        output_url=f"/api/v1/files/{output_id}",
                        status="completed",
                        score=float(metrics["score"]),
                        metrics=metrics,
                        parameters=result.parameters,
                        runtime_ms=result.runtime_ms,
                        peak_memory_mb=result.peak_memory_mb,
                    )
                    task.candidates.append(cand)
                except Exception as exc:
                    task.candidates.append(CandidateResult(model_id=model_id, status="failed", error=str(exc)))
                    logs.append(f"模型 {model_id} 失败：{exc}")
            completed = [c for c in task.candidates if c.status == "completed"]
            if completed:
                task.best_result = sorted(completed, key=lambda c: c.score, reverse=True)[0]
                step("completed", 0.95, f"选择综合评分最高结果：{task.best_result.model_id}。")
            else:
                task.error = "没有可用模型完成真实推理。请运行 scripts/download_models.ps1 获取官方源码/权重，或在测试环境显式启用 ALLOW_MOCK_MODELS=true。"
                step("failed", 0.95, task.error)
            task.completed_at = now_iso()
            report_id = str(uuid4())
            reports = ReportService().write_reports(task.model_dump(), self.settings.report_dir / report_id)
            task.report_file_id = report_id
            logs.append(f"报告已导出：{reports['markdown']}")
            save_task(task)
        except Exception as exc:
            task.status = "failed"
            task.error = str(exc)
            task.completed_at = now_iso()
            logs.append(f"任务失败：{exc}")
            save_task(task)
