from pathlib import Path
from uuid import uuid4

from visionrestore.agent.candidate_planner import CandidatePlanner
from visionrestore.agent.enhancement_agent import EnhancementAgent
from visionrestore.context import ContextRetrievalService
from visionrestore.schemas.common import now_iso
from visionrestore.schemas.task import EnhancementPlan, TaskRecord
from visionrestore.services.artifact_lineage import ArtifactLineageService
from visionrestore.services.multi_candidate_executor import MultiCandidateExecutor
from visionrestore.services.report import ReportService
from visionrestore.services.residual_analyzer import ResidualDegradationAnalyzer
from visionrestore.services.result_selector import ResultSelector
from visionrestore.services.region_constraints import RegionConstraintService
from visionrestore.storage.database import Database
from visionrestore.utils.file_security import resolve_registered_path


class EnhancementAgentV2(EnhancementAgent):
    def __init__(self):
        super().__init__()
        self.planner = CandidatePlanner()
        self.executor = MultiCandidateExecutor(registry=self.registry, evaluator=self.evaluator)
        self.selector = ResultSelector()
        self.residual = ResidualDegradationAnalyzer()
        self.lineage = ArtifactLineageService()
        self.db = Database()
        self.context_retrieval = ContextRetrievalService()
        self.region_constraints = RegionConstraintService()

    def run(self, task: TaskRecord, input_file: dict, save_task):
        logs = task.logs

        def step(status: str, progress: float, message: str):
            task.status = status
            task.progress = progress
            logs.append(message)
            save_task(task)

        try:
            task.task_mode = "multi_candidate"
            input_path = resolve_registered_path(input_file["relative_path"])

            step("analyzing_input", 0.08, "分析 RGB 静态图像退化统计。")
            analysis = self.analyzer.analyze(str(input_path))
            task.analysis = analysis

            step("parsing_intent", 0.16, "解析用户自然语言需求为结构化意图。")
            manual_model = task.plan.selected_model if task.plan else None
            manual_weight = task.plan.selected_checkpoint if task.plan else None
            intent = self.intent_parser.parse(task.user_goal, task.priority, task.mode, manual_model, manual_weight)
            task.user_intent = intent
            if intent.priority != task.priority and task.mode == "auto":
                task.priority = intent.priority

            step("inspecting_hardware", 0.24, "检查 CPU、GPU、CUDA、显存和模型状态。")
            hw = self.hardware.inspect()
            task.hardware_info = hw
            model_status = self.registry.list()
            retrieved_context = self.context_retrieval.retrieve(
                user_request=task.user_goal,
                image_metrics=analysis,
                available_models=model_status,
                hardware_summary=hw,
            )
            model_reference = self.context_retrieval.model_reference(model_status)
            merged_context = []
            seen_context = set()
            for item in [*model_reference, *retrieved_context]:
                key = item.item_id or f"{item.source}:{item.title}"
                if key in seen_context:
                    continue
                seen_context.add(key)
                merged_context.append(item)
            retrieved_context = merged_context
            task.retrieved_context = [item.model_dump() for item in retrieved_context]
            if task.retrieved_context:
                self.db.put_entity("retrieved_context", f"{task.task_id}:latest", task.retrieved_context, task_id=task.task_id)
                logs.append(
                    f"Loaded {len(task.retrieved_context)} allowlisted context items; model role definitions are sent only as LLM scoring standards when external analysis is enabled."
                )

            step("running_multimodal_analysis", 0.32, "增强前执行多模态语义分析，并通过本地校验。")
            task.ai_analysis = self._run_ai_analysis(
                input_path=Path(input_path),
                image_metrics=analysis,
                user_request=task.user_goal,
                available_models=model_status,
                hardware_summary=hw,
                analysis_mode=getattr(task, "analysis_mode", "local"),
                manual_model=manual_model,
                manual_checkpoint=manual_weight,
                knowledge_context=task.retrieved_context,
            )
            constraints = self.region_constraints.build_constraints(
                user_request=task.user_goal,
                image_path=str(input_path),
                image_size=[analysis.width, analysis.height],
                parameters=task.parameters,
                ai_analysis=task.ai_analysis,
            )
            task.region_constraints = [item.model_dump() for item in constraints]
            if task.region_constraints:
                intent.preferences.protect_highlights = True
                self.db.put_entity("region_constraint", f"{task.task_id}:latest", task.region_constraints, task_id=task.task_id)
                logs.append(f"Region constraint monitor enabled for {len(task.region_constraints)} ROI item(s); hard failures will be rejected after real output evaluation.")
            if task.ai_analysis.adopted:
                logs.append("多模态 LLMScore 与 LocalScore 各占 50% 参与候选规划，不会直接决定最终结果。")
            elif task.ai_analysis.failure_reason or task.ai_analysis.rejection_reason:
                logs.append(f"多模态语义建议未参与评分：{task.ai_analysis.rejection_reason or task.ai_analysis.failure_reason}。")
            save_task(task)

            step("planning_candidates", 0.42, "CandidatePlanner 生成 1-3 个互补增强候选。")
            plan = self.planner.plan(
                image_id=task.image_id,
                intent=intent,
                analysis=analysis,
                hardware=hw,
                model_statuses=model_status,
                mode=task.mode,
                semantic_analysis=task.ai_analysis,
                retrieved_context=task.retrieved_context,
            )
            task.candidate_plan = plan.model_dump()
            task.model_candidates = [item.model_dump() for item in plan.candidates]
            task.checkpoint_candidates = [
                {
                    "candidate_id": item.candidate_id,
                    "model_id": item.model_id,
                    "checkpoint_id": item.checkpoint_id,
                    "checkpoint_score": item.checkpoint_score,
                    "checkpoint_selection_mode": item.checkpoint_selection_mode,
                    "checkpoint_evidence": [
                        evidence.model_dump(mode="json")
                        for evidence in item.checkpoint_evidence
                    ],
                    "checkpoint_candidates": item.checkpoint_candidates,
                    "model_prior_score": item.model_prior_score,
                    "input_match_score": item.input_match_score,
                    "local_score": item.local_score,
                    "llm_score": item.llm_score,
                    "local_weight": item.local_weight,
                    "llm_weight": item.llm_weight,
                    "planning_mode": item.planning_mode,
                    "planning_score": item.planning_score,
                    "planning_evidence": [evidence.model_dump(mode="json") for evidence in item.planning_evidence],
                    "knowledge_adjustment": item.knowledge_adjustment,
                    "knowledge_evidence": [evidence.model_dump(mode="json") for evidence in item.knowledge_evidence],
                    "reason": item.reason,
                }
                for item in plan.candidates
            ]
            first = plan.candidates[0] if plan.candidates else None
            if first:
                task.plan = EnhancementPlan(
                    input_id=task.image_id,
                    user_goal=task.user_goal,
                    mode=task.mode,
                    priority=task.priority,
                    selected_model=first.model_id,
                    selected_checkpoint=first.checkpoint_id,
                    selection_reason="V2 多候选规划入口，仅表示第一个待执行候选，不是最终结果。",
                    parameters={"task_mode": "multi_candidate", "candidate_limit": plan.max_candidates},
                    preprocessing_steps=["all candidates read original input only"],
                    postprocessing_steps=["rank real outputs", "diagnose residual degradation", "interactive denoise/SR if needed"],
                    evaluation_strategy="planning_score before inference; final_score after real output evaluation",
                    expected_risks=["planning_score 只决定是否执行，不代表最终图像质量"],
                )
            self.db.put_entity("candidate_plan", f"{task.task_id}:latest", plan.model_dump(), task_id=task.task_id)
            save_task(task)

            if not plan.candidates:
                task.error = "CandidatePlanner 没有生成可执行候选。"
                step("failed", 0.95, task.error)
                task.completed_at = now_iso()
                save_task(task)
                return

            candidates_root = self.lineage.create_task_layout(task.task_id) / "candidates"
            progress_base = 0.50
            progress_span = 0.22

            def on_event(status: str, message: str):
                completed_count = len(task.candidates)
                progress = min(0.72, progress_base + progress_span * completed_count / max(1, len(plan.candidates)))
                step(status, progress, message)

            results = self.executor.execute(
                plan=plan,
                input_path=str(input_path),
                priority=task.priority,
                task_dir=candidates_root,
                cancel_check=lambda: task.status == "cancelled",
                on_event=on_event,
                region_constraints=task.region_constraints,
            )
            task.candidates.extend(results)
            for result in results:
                self.db.put_entity("candidate_result", f"{task.task_id}:{result.candidate_id or result.model_id}", result.model_dump(), task_id=task.task_id)
            save_task(task)

            if self._region_retry_needed(plan, task.candidates, task.region_constraints):
                logs.append("All first-pass completed candidates violated hard region constraints; expanding to a quality retry plan from the original input.")
                retry_intent = intent.model_copy(deep=True)
                retry_intent.priority = "quality"
                retry_intent.preferences.protect_highlights = True
                retry_plan = self.planner.plan(
                    image_id=task.image_id,
                    intent=retry_intent,
                    analysis=analysis,
                    hardware=hw,
                    model_statuses=model_status,
                    mode="compare",
                    semantic_analysis=task.ai_analysis,
                    retrieved_context=task.retrieved_context,
                )
                already = {(item.model_id, item.checkpoint_id) for item in task.candidates}
                retry_items = [item for item in retry_plan.candidates if (item.model_id, item.checkpoint_id) not in already]
                for idx, item in enumerate(retry_items, start=1):
                    item.candidate_id = f"region_retry_{idx:02d}"
                    item.reason = list(item.reason) + ["区域约束首轮未满足，触发一次从原图补跑的高光保护重试"]
                retry_plan.candidates = retry_items
                retry_plan.max_candidates = len(retry_items)
                if retry_plan.candidates:
                    retry_results = self.executor.execute(
                        plan=retry_plan,
                        input_path=str(input_path),
                        priority="quality",
                        task_dir=candidates_root,
                        cancel_check=lambda: task.status == "cancelled",
                        on_event=on_event,
                        region_constraints=task.region_constraints,
                    )
                    task.candidates.extend(retry_results)
                    task.candidate_plan = {**(task.candidate_plan or {}), "region_retry_plan": retry_plan.model_dump()}
                    for result in retry_results:
                        self.db.put_entity("candidate_result", f"{task.task_id}:{result.candidate_id or result.model_id}", result.model_dump(), task_id=task.task_id)
                    save_task(task)
                else:
                    logs.append("Region retry was requested, but no additional non-duplicate candidate was available.")

            if task.status == "cancelled":
                task.completed_at = now_iso()
                save_task(task)
                return

            step("ranking_candidates", 0.78, "CandidateEvaluator 和 CandidateRanker 对真实候选结果重新评分。")
            selection = self.selector.select([item.model_dump() for item in task.candidates], task.priority)
            task.candidate_ranking = selection.model_dump()
            score_by_candidate = {item.get("candidate_id"): item for item in selection.successful + selection.failed}
            for candidate in task.candidates:
                ranked = score_by_candidate.get(candidate.candidate_id)
                if ranked:
                    candidate.final_score = ranked.get("score")
                    candidate.score = float(ranked.get("score") or candidate.score)
                    candidate.metrics = {
                        **candidate.metrics,
                        "final_score": ranked.get("score"),
                        "score_layers": ranked.get("layers", {}),
                        "score_reasons": ranked.get("reasons", []),
                        "score_evidence": ranked.get("evidence", {}),
                        "planning_score_is_not_final_score": True,
                    }
                    self.db.put_entity("candidate_metric", f"{task.task_id}:{candidate.candidate_id or candidate.model_id}", candidate.metrics, task_id=task.task_id)
            self.db.put_entity("candidate_ranking", f"{task.task_id}:latest", selection.model_dump(), task_id=task.task_id)
            save_task(task)

            step("selecting_best_candidate", 0.84, "根据真实输出最终评分选择最佳增强结果。")
            if not selection.selected:
                task.error = selection.message or "没有候选模型完成有效推理。"
                step("failed", 0.95, task.error)
                task.completed_at = now_iso()
                self._write_report(task)
                save_task(task)
                return
            selected_id = selection.selected.get("candidate_id")
            task.best_result = next((item for item in task.candidates if item.candidate_id == selected_id), None)
            if task.best_result is None:
                task.best_result = max([item for item in task.candidates if item.status == "completed"], key=lambda item: item.score)
            task.final_recommendation = f"最佳增强候选为 {task.best_result.model_id}:{task.best_result.checkpoint_id}，最终评分 {task.best_result.score}。"
            save_task(task)

            step("diagnosing_residual_degradation", 0.90, "ResidualDegradationAnalyzer 诊断最佳增强结果是否需要后处理。")
            recommendation = self.residual.analyze(
                original_analysis=analysis,
                enhanced_metrics=task.best_result.metrics,
                best_candidate=task.best_result.model_dump(),
                user_intent=task.user_intent,
                hardware=task.hardware_info or {},
            )
            task.postprocess_recommendation = recommendation.model_dump()
            self.db.put_entity("postprocess_recommendation", f"{task.task_id}:latest", task.postprocess_recommendation, task_id=task.task_id)
            logs.append("后处理建议已生成；需要用户确认的步骤不会自动覆盖最佳增强结果。")

            if recommendation.denoise_recommended:
                task.status = "awaiting_denoise_confirmation"
                task.progress = 0.94
                task.final_recommendation = f"{task.final_recommendation} 检测到残余噪声，建议先确认是否执行 {recommendation.preferred_denoiser} 去噪。"
            elif recommendation.super_resolution_recommended:
                task.status = "awaiting_sr_confirmation"
                task.progress = 0.95
                task.final_recommendation = f"{task.final_recommendation} 检测到分辨率不足，建议确认是否执行超分。"
            else:
                step("finalizing", 0.98, "无需后处理或未达到建议阈值，准备输出最终结果。")
                task.status = "completed"
                task.progress = 1.0
                task.completed_at = now_iso()
            self._write_report(task)
            save_task(task)
        except Exception as exc:
            task.status = "failed"
            task.error = str(exc)
            task.completed_at = now_iso()
            logs.append(f"任务失败：{exc}")
            self._write_report(task)
            save_task(task)

    def _region_retry_needed(self, plan, candidates, region_constraints: list[dict]) -> bool:
        if not region_constraints or getattr(plan, "max_candidates", 0) >= 3:
            return False
        completed = [item for item in candidates if item.status == "completed"]
        if not completed:
            return False
        return all((item.metrics.get("region_constraints") or {}).get("severe_failure") for item in completed)

    def _write_report(self, task: TaskRecord) -> None:
        if not task.report_file_id:
            task.report_file_id = str(uuid4())
        ReportService().write_reports(task.model_dump(), self.settings.report_dir / task.report_file_id)
