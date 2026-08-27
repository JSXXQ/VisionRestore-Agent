from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from uuid import uuid4

from langgraph.types import Send, interrupt

from visionrestore.adapters.registry import ModelRegistry
from visionrestore.agent.candidate_planner import CandidatePlanner
from visionrestore.agent.intent_parser import IntentParser
from visionrestore.context import ContextRetrievalService
from visionrestore.core.config import get_settings
from visionrestore.graph.advisory import SemanticAdvisoryService
from visionrestore.graph.events import make_event, make_message
from visionrestore.schemas.candidate import CandidateExecutionPlan, CandidatePlanItem
from visionrestore.schemas.common import now_iso
from visionrestore.schemas.image import ImageAnalysisResult
from visionrestore.schemas.postprocess import PostprocessDecision, PostprocessResult
from visionrestore.schemas.task import CandidateResult, EnhancementPlan, TaskRecord
from visionrestore.services.artifact_lineage import ArtifactLineageService
from visionrestore.services.evaluator import QualityEvaluator
from visionrestore.services.hardware import HardwareInspector
from visionrestore.services.image_analyzer import ImageAnalyzer
from visionrestore.services.multi_candidate_executor import MultiCandidateExecutor
from visionrestore.services.postprocess_controller import PostprocessController
from visionrestore.services.report import ReportService
from visionrestore.services.residual_analyzer import ResidualDegradationAnalyzer
from visionrestore.services.result_selector import ResultSelector
from visionrestore.services.region_constraints import RegionConstraintService
from visionrestore.storage.database import Database
from visionrestore.tools import ToolContext, ToolDefinition, ToolRegistry, ToolRouter
from visionrestore.tools.schemas import AnalyzeImageInput, ExecuteCandidateInput, InspectHardwareInput, RetrieveContextInput
from visionrestore.utils.file_security import resolve_registered_path


WORKFLOW_VERSION = "visionrestore-langgraph-v2.2"


@dataclass
class VisionRestoreDependencies:
    settings: object = field(default_factory=get_settings)
    database: Database = field(default_factory=Database)
    registry: ModelRegistry = field(default_factory=ModelRegistry)
    analyzer: ImageAnalyzer = field(default_factory=ImageAnalyzer)
    intent_parser: IntentParser = field(default_factory=IntentParser)
    hardware: HardwareInspector = field(default_factory=HardwareInspector)
    context_retrieval: ContextRetrievalService = field(default_factory=ContextRetrievalService)
    semantic_advisory: SemanticAdvisoryService = field(default_factory=SemanticAdvisoryService)
    planner: CandidatePlanner = field(default_factory=CandidatePlanner)
    evaluator: QualityEvaluator = field(default_factory=QualityEvaluator)
    selector: ResultSelector = field(default_factory=ResultSelector)
    residual: ResidualDegradationAnalyzer = field(default_factory=ResidualDegradationAnalyzer)
    region_constraints: RegionConstraintService = field(default_factory=RegionConstraintService)
    lineage: ArtifactLineageService = field(default_factory=ArtifactLineageService)
    report: ReportService = field(default_factory=ReportService)
    postprocess: PostprocessController = field(default_factory=PostprocessController)
    executor: MultiCandidateExecutor | None = None

    def __post_init__(self):
        if self.executor is None:
            self.executor = MultiCandidateExecutor(registry=self.registry, evaluator=self.evaluator, database=self.database)


class VisionRestoreGraphRuntime:
    def __init__(
        self,
        *,
        save_task: Callable[[TaskRecord], None],
        cancel_check: Callable[[str], bool] | None = None,
        dependencies: VisionRestoreDependencies | None = None,
    ):
        self.save_task = save_task
        self.cancel_check = cancel_check or (lambda _task_id: False)
        self.deps = dependencies or VisionRestoreDependencies()
        self.tools = ToolRegistry()
        self._register_tools()
        self.tool_router = ToolRouter(self.tools)

    def initial_state(self, task: TaskRecord, input_file: dict) -> dict:
        run_id = str(uuid4())
        user_message = make_message(
            role="user",
            content=task.user_goal or "增强这张低照度图像",
            source="task_request",
            step_id="initialize_task",
            metadata={"image_id": task.image_id, "mode": task.mode, "priority": task.priority},
        )
        task.workflow_engine = "langgraph"
        task.workflow_thread_id = task.task_id
        task.workflow_version = WORKFLOW_VERSION
        task.messages = self._append_unique(task.messages, [user_message], "message_id")
        self.save_task(task)
        return {
            "task_id": task.task_id,
            "run_id": run_id,
            "workflow_version": WORKFLOW_VERSION,
            "task_projection": task.model_dump(mode="json"),
            "input_file": input_file,
            "candidate_results": [],
            "retry_count": 0,
            "current_phase": "queued",
            "events": [],
            "messages": [user_message],
            "terminal_status": None,
            "error": None,
        }

    def initialize_task(self, state: dict) -> dict:
        task = self._task(state)
        input_path = resolve_registered_path(state["input_file"]["relative_path"])
        self.deps.database.put_entity(
            "graph_run",
            f"{task.task_id}:{state['run_id']}",
            {
                "task_id": task.task_id,
                "run_id": state["run_id"],
                "workflow_version": WORKFLOW_VERSION,
                "status": "started",
                "started_at": now_iso(),
            },
            task_id=task.task_id,
        )
        update = self._commit(
            state,
            task,
            step_id="initialize_task",
            status="queued",
            progress=0.02,
            message="LangGraph V2 工作流已初始化。",
            metadata={"input_path_registered": True},
        )
        update["input_path"] = str(input_path)
        return update

    def analyze_input(self, state: dict) -> dict:
        task = self._task(state)
        tool = self._tool(
            state,
            "analyze_image",
            {"image_path": state["input_path"]},
            step_id="analyze_input",
        )
        analysis = tool.output["analysis"]
        task.analysis = ImageAnalysisResult.model_validate(analysis)
        update = self._commit(
            state,
            task,
            step_id="analyze_input",
            status="analyzing_input",
            progress=0.08,
            message="分析 RGB 静态图像退化统计。",
            metadata=self._tool_metadata(tool),
        )
        update["image_analysis"] = analysis
        return update

    def parse_intent(self, state: dict) -> dict:
        task = self._task(state)
        manual_model = task.plan.selected_model if task.plan else None
        manual_weight = task.plan.selected_checkpoint if task.plan else None
        intent = self.deps.intent_parser.parse(task.user_goal, task.priority, task.mode, manual_model, manual_weight)
        task.user_intent = intent
        if intent.priority != task.priority and task.mode == "auto":
            task.priority = intent.priority
        update = self._commit(
            state,
            task,
            step_id="parse_intent",
            status="parsing_intent",
            progress=0.16,
            message="解析用户自然语言需求为结构化意图。",
        )
        update["user_intent"] = intent.model_dump(mode="json")
        return update

    def inspect_runtime(self, state: dict) -> dict:
        task = self._task(state)
        tool = self._tool(state, "inspect_runtime", {"include_models": True}, step_id="inspect_runtime")
        hardware = tool.output["hardware"]
        models = tool.output["models"]
        task.hardware_info = hardware
        update = self._commit(
            state,
            task,
            step_id="inspect_runtime",
            status="inspecting_hardware",
            progress=0.24,
            message="检查 CPU、GPU、CUDA、显存和模型状态。",
            metadata=self._tool_metadata(tool),
        )
        update["hardware_snapshot"] = hardware
        update["model_snapshot"] = models
        return update

    def retrieve_context(self, state: dict) -> dict:
        task = self._task(state)
        tool = self._tool(
            state,
            "retrieve_context",
            {
                "user_request": task.user_goal,
                "image_metrics": state.get("image_analysis", {}),
                "available_models": state.get("model_snapshot", []),
                "hardware_summary": state.get("hardware_snapshot", {}),
            },
            step_id="retrieve_context",
        )
        context = tool.output.get("items", [])
        task.retrieved_context = context
        if context:
            self.deps.database.put_entity("retrieved_context", f"{task.task_id}:latest", context, task_id=task.task_id)
        update = self._commit(
            state,
            task,
            step_id="retrieve_context",
            status="inspecting_hardware",
            progress=0.28,
            message=f"检索到 {len(context)} 条受限本地知识上下文。",
            metadata=self._tool_metadata(tool),
        )
        update["retrieved_context"] = context
        return update

    def semantic_analysis(self, state: dict) -> dict:
        task = self._task(state)
        manual_model = task.plan.selected_model if task.plan else None
        manual_weight = task.plan.selected_checkpoint if task.plan else None
        result = self.deps.semantic_advisory.analyze(
            input_path=Path(state["input_path"]),
            image_metrics=task.analysis,
            user_request=task.user_goal,
            available_models=state.get("model_snapshot", []),
            hardware_summary=state.get("hardware_snapshot", {}),
            analysis_mode=task.analysis_mode,
            manual_model=manual_model,
            manual_checkpoint=manual_weight,
            knowledge_context=state.get("retrieved_context", []),
        )
        task.ai_analysis = result
        constraints = self.deps.region_constraints.build_constraints(
            user_request=task.user_goal,
            image_path=state["input_path"],
            image_size=[task.analysis.width, task.analysis.height],
            parameters=task.parameters,
            ai_analysis=result,
        )
        task.region_constraints = [item.model_dump(mode="json") for item in constraints]
        if task.region_constraints and task.user_intent:
            task.user_intent.preferences.protect_highlights = True
            self.deps.database.put_entity("region_constraint", f"{task.task_id}:latest", task.region_constraints, task_id=task.task_id)
        message = (
            "多模态 LLMScore 已通过本地校验，将与 LocalScore 各占 50% 参与候选规划。"
            if result.adopted
            else "外部语义评分不可用，候选规划降级为 LocalScore。"
        )
        update = self._commit(
            state,
            task,
            step_id="semantic_analysis",
            status="running_multimodal_analysis",
            progress=0.34,
            message=message,
            metadata={"provider": result.provider, "adopted": result.adopted, "sent_image": result.sent_image},
        )
        update["ai_advisory"] = result.model_dump(mode="json")
        update["region_constraints"] = task.region_constraints
        update["user_intent"] = task.user_intent.model_dump(mode="json") if task.user_intent else {}
        return update

    def plan_candidates(self, state: dict) -> dict:
        task = self._task(state)
        plan = self.deps.planner.plan(
            image_id=task.image_id,
            intent=task.user_intent,
            analysis=task.analysis,
            hardware=task.hardware_info or {},
            model_statuses=state.get("model_snapshot", []),
            mode=task.mode,
            semantic_analysis=task.ai_analysis,
            retrieved_context=state.get("retrieved_context", []),
        )
        task.candidate_plan = plan.model_dump(mode="json")
        task.model_candidates = [item.model_dump(mode="json") for item in plan.candidates]
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
                selection_reason="LangGraph V2 候选规划入口；首项仅是待执行候选，不是最终结果。",
                parameters={"task_mode": "multi_candidate", "candidate_limit": plan.max_candidates},
                preprocessing_steps=["all candidates read original input only"],
                postprocessing_steps=["rank real outputs", "diagnose residual degradation", "interactive denoise/SR if needed"],
                evaluation_strategy="planning_score before inference; final_score after real output evaluation",
                expected_risks=["planning_score 只决定是否执行，不代表最终图像质量"],
            )
        self.deps.database.put_entity("candidate_plan", f"{task.task_id}:latest", plan.model_dump(mode="json"), task_id=task.task_id)
        llm_scored_count = sum(item.llm_score is not None for item in plan.candidates)
        update = self._commit(
            state,
            task,
            step_id="plan_candidates",
            status="planning_candidates",
            progress=0.42,
            message=(
                f"CandidatePlanner 生成 {len(plan.candidates)} 个互补增强候选；"
                f"{llm_scored_count} 个候选采用 LocalScore/LLMScore 各 50% 融合；"
                "每个模型家族均已完成本地 checkpoint 内部匹配。"
            ),
            metadata={
                "candidate_count": len(plan.candidates),
                "llm_scored_count": llm_scored_count,
                "planning_formula": "0.5 * local_score + 0.5 * llm_score; local fallback when LLM is unavailable",
                "knowledge_role": "LLM reference standard only",
                "planning_scores_are_not_final": True,
            },
        )
        update["candidate_plan"] = plan.model_dump(mode="json")
        update["pending_candidates"] = [item.model_dump(mode="json") for item in plan.candidates]
        return update

    def dispatch_candidates(self, state: dict):
        pending = state.get("pending_candidates", [])
        if not pending:
            return "rank_candidates" if state.get("candidate_results") else "fail_task"
        return [
            Send(
                "candidate_execution",
                {
                    "task_id": state["task_id"],
                    "run_id": state["run_id"],
                    "workflow_version": state["workflow_version"],
                    "candidate_to_execute": candidate,
                    "input_path": state["input_path"],
                    "region_constraints": state.get("region_constraints", []),
                    "task_projection": state["task_projection"],
                    "candidate_results": [],
                    "events": [],
                    "messages": [],
                },
            )
            for candidate in pending
        ]

    def execute_candidate(self, state: dict) -> dict:
        task = self._task(state)
        candidate = state["candidate_to_execute"]
        task_root = self.deps.lineage.create_task_layout(task.task_id) / "candidates"
        tool = self._tool(
            state,
            "execute_candidate",
            {
                "task_id": task.task_id,
                "candidate": candidate,
                "input_path": state["input_path"],
                "priority": task.priority,
                "task_dir": str(task_root),
                "region_constraints": state.get("region_constraints", []),
            },
            step_id=f"execute_{candidate.get('candidate_id', 'candidate')}",
        )
        result = tool.output["candidate_result"]
        event = make_event(
            task_id=task.task_id,
            run_id=state["run_id"],
            step_id=f"execute_{candidate.get('candidate_id', 'candidate')}",
            event_type="tool_completed",
            status=result.get("status"),
            message=f"候选 {candidate.get('candidate_id')} 执行完成：{result.get('status')}。",
            metadata={**self._tool_metadata(tool), "model_id": candidate.get("model_id"), "checkpoint_id": candidate.get("checkpoint_id")},
        )
        message = make_message(
            role="tool",
            content=event["message"],
            source="execute_candidate",
            step_id=event["step_id"],
            metadata=event["metadata"],
        )
        self._persist_event(event)
        return {"candidate_results": [result], "events": [event], "messages": [message]}

    def aggregate_candidates(self, state: dict) -> dict:
        task = self._task(state)
        results = state.get("candidate_results", [])
        task.candidates = [CandidateResult.model_validate(item) for item in results]
        self._merge_trace_into_task(task, state)
        update = self._commit(
            state,
            task,
            step_id="aggregate_candidates",
            status="evaluating_candidates",
            progress=0.72,
            message=f"已汇总 {len(results)} 个候选执行结果。",
            metadata={"completed": len([item for item in results if item.get('status') == 'completed'])},
        )
        update["candidate_results"] = results
        return update

    def route_after_aggregate(self, state: dict) -> str:
        if self.cancel_check(state["task_id"]):
            return "cancel_task"
        if state.get("retry_count", 0) < 1 and self._region_retry_needed(state):
            return "plan_region_retry"
        return "rank_candidates"

    def plan_region_retry(self, state: dict) -> dict:
        task = self._task(state)
        retry_intent = task.user_intent.model_copy(deep=True)
        retry_intent.priority = "quality"
        retry_intent.preferences.protect_highlights = True
        retry_plan = self.deps.planner.plan(
            image_id=task.image_id,
            intent=retry_intent,
            analysis=task.analysis,
            hardware=task.hardware_info or {},
            model_statuses=state.get("model_snapshot", []),
            mode="compare",
            semantic_analysis=task.ai_analysis,
            retrieved_context=state.get("retrieved_context", []),
        )
        already = {(item.get("model_id"), item.get("checkpoint_id")) for item in state.get("candidate_results", [])}
        retry_items = [item for item in retry_plan.candidates if (item.model_id, item.checkpoint_id) not in already]
        for index, item in enumerate(retry_items, start=1):
            item.candidate_id = f"region_retry_{index:02d}"
            item.reason = [*item.reason, "区域硬约束首轮未满足，触发一次从原图执行的质量重试"]
        retry_plan.candidates = retry_items
        retry_plan.max_candidates = len(retry_items)
        original_plan = task.candidate_plan or {}
        task.candidate_plan = {**original_plan, "region_retry_plan": retry_plan.model_dump(mode="json")}
        update = self._commit(
            state,
            task,
            step_id="plan_region_retry",
            status="planning_candidates",
            progress=0.73,
            message=f"区域约束触发一次有界质量重试，新增 {len(retry_items)} 个候选。",
            metadata={"retry_from_original": True, "retry_count": 1},
        )
        update["candidate_plan"] = retry_plan.model_dump(mode="json")
        update["pending_candidates"] = [item.model_dump(mode="json") for item in retry_items]
        update["retry_count"] = 1
        return update

    def rank_candidates(self, state: dict) -> dict:
        task = self._task(state)
        selection = self.deps.selector.select(state.get("candidate_results", []), task.priority)
        task.candidate_ranking = selection.model_dump(mode="json")
        score_by_candidate = {item.get("candidate_id"): item for item in [*selection.successful, *selection.failed]}
        refreshed: list[CandidateResult] = []
        for raw in state.get("candidate_results", []):
            candidate = CandidateResult.model_validate(raw)
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
                self.deps.database.put_entity("candidate_metric", f"{task.task_id}:{candidate.candidate_id}", candidate.metrics, task_id=task.task_id)
            refreshed.append(candidate)
        task.candidates = refreshed
        self.deps.database.put_entity("candidate_ranking", f"{task.task_id}:latest", selection.model_dump(mode="json"), task_id=task.task_id)
        update = self._commit(
            state,
            task,
            step_id="rank_candidates",
            status="ranking_candidates",
            progress=0.78,
            message="CandidateEvaluator 和 CandidateRanker 已对真实候选结果完成最终评分。",
            metadata={"successful": len(selection.successful), "failed": len(selection.failed)},
        )
        update["candidate_results"] = [item.model_dump(mode="json") for item in refreshed]
        update["candidate_ranking"] = selection.model_dump(mode="json")
        return update

    def route_after_ranking(self, state: dict) -> str:
        selected = (state.get("candidate_ranking") or {}).get("selected")
        return "select_best" if selected else "fail_task"

    def select_best(self, state: dict) -> dict:
        task = self._task(state)
        selected = (state.get("candidate_ranking") or {}).get("selected") or {}
        selected_id = selected.get("candidate_id")
        task.best_result = next((item for item in task.candidates if item.candidate_id == selected_id), None)
        if task.best_result is None:
            completed = [item for item in task.candidates if item.status == "completed"]
            task.best_result = max(completed, key=lambda item: item.score) if completed else None
        if task.best_result is None:
            return self.fail_task({**state, "task_projection": task.model_dump(mode="json")})
        task.final_recommendation = f"最佳增强候选为 {task.best_result.model_id}:{task.best_result.checkpoint_id}，最终评分 {task.best_result.score}。"
        update = self._commit(
            state,
            task,
            step_id="select_best",
            status="selecting_best_candidate",
            progress=0.84,
            message=task.final_recommendation,
            metadata={"candidate_id": task.best_result.candidate_id, "final_score": task.best_result.score},
        )
        update["selected_candidate"] = task.best_result.model_dump(mode="json")
        return update

    def diagnose_residual(self, state: dict) -> dict:
        task = self._task(state)
        recommendation = self.deps.residual.analyze(
            original_analysis=task.analysis,
            enhanced_metrics=task.best_result.metrics,
            best_candidate=task.best_result.model_dump(mode="json"),
            user_intent=task.user_intent,
            hardware=task.hardware_info or {},
        )
        task.postprocess_recommendation = recommendation.model_dump(mode="json")
        self.deps.database.put_entity("postprocess_recommendation", f"{task.task_id}:latest", task.postprocess_recommendation, task_id=task.task_id)
        next_operation = "denoise" if recommendation.denoise_recommended else "super_resolution" if recommendation.super_resolution_recommended else "none"
        self._write_report(task)
        update = self._commit(
            state,
            task,
            step_id="diagnose_residual",
            status="diagnosing_residual_degradation",
            progress=0.90,
            message="残余退化诊断完成；需要确认的后处理不会自动覆盖最佳增强结果。",
            metadata={"next_postprocess_operation": next_operation},
        )
        update["residual_diagnosis"] = recommendation.model_dump(mode="json")
        update["next_postprocess_operation"] = next_operation
        return update

    def route_after_diagnosis(self, state: dict) -> str:
        return "postprocess_workflow" if state.get("next_postprocess_operation") in {"denoise", "super_resolution"} else "finalize"

    def route_postprocess_entry(self, state: dict) -> str:
        operation = state.get("next_postprocess_operation", "none")
        if operation == "denoise":
            return "prepare_denoise"
        if operation == "super_resolution":
            return "prepare_sr"
        return "done"

    def prepare_denoise(self, state: dict) -> dict:
        return self._prepare_confirmation(state, "denoise")

    def prepare_sr(self, state: dict) -> dict:
        return self._prepare_confirmation(state, "super_resolution")

    def wait_postprocess_decision(self, state: dict) -> dict:
        decision = interrupt(state.get("pending_confirmation") or {"operation": state.get("next_postprocess_operation")})
        return {"postprocess_decision": decision}

    def apply_postprocess_decision(self, state: dict) -> dict:
        task = self._task(state)
        pending = state.get("pending_confirmation") or {}
        payload = state.get("postprocess_decision") or {}
        decision = PostprocessDecision.model_validate(payload)
        expected_operation = pending.get("operation")
        if expected_operation and decision.operation != expected_operation:
            result = PostprocessResult(
                task_id=task.task_id,
                operation=decision.operation,
                decision=decision.decision,
                accepted=False,
                executed=False,
                next_status=task.status,
                message=f"当前等待 {expected_operation} 决策，拒绝不匹配的 {decision.operation} 决策。",
            )
            next_operation = expected_operation
        else:
            result = self.deps.postprocess.decide(task=task, decision=decision)
            if result.next_status:
                task.status = result.next_status
            next_operation = self._operation_from_status(result.next_status)
        task.pending_confirmation = None
        task.logs.append(result.message)
        self.deps.database.put_entity(
            "postprocess_decision",
            f"{task.task_id}:{decision.operation}:{uuid4()}",
            {"decision": decision.model_dump(mode="json"), "result": result.model_dump(mode="json")},
            task_id=task.task_id,
        )
        if result.executed:
            self.deps.database.put_entity("postprocess_result", f"{task.task_id}:{decision.operation}:latest", result.model_dump(mode="json"), task_id=task.task_id)
        self._write_report(task)
        update = self._commit(
            state,
            task,
            step_id=f"apply_{decision.operation}_decision",
            status=task.status,
            progress=task.progress,
            message=result.message,
            metadata={
                "accepted": result.accepted,
                "executed": result.executed,
                "rollback": result.rollback_performed,
                "next_postprocess_operation": next_operation,
                "before_score": result.metadata.get("before_score"),
                "after_score": result.metadata.get("after_score"),
                "score_layers": result.metadata.get("score_layers", {}),
            },
            append_log=False,
        )
        update["last_postprocess_result"] = result.model_dump(mode="json")
        update["next_postprocess_operation"] = next_operation
        update["pending_confirmation"] = {}
        return update

    def route_after_postprocess(self, state: dict) -> str:
        operation = state.get("next_postprocess_operation", "none")
        if operation == "denoise":
            return "prepare_denoise"
        if operation == "super_resolution":
            return "prepare_sr"
        return "done"

    def finalize(self, state: dict) -> dict:
        task = self._task(state)
        if task.status not in {"failed", "cancelled"}:
            task.status = "completed"
            task.progress = 1.0
            task.completed_at = task.completed_at or now_iso()
            task.pending_confirmation = None
        self._write_report(task)
        self.deps.database.put_entity(
            "graph_run",
            f"{task.task_id}:{state['run_id']}",
            {
                "task_id": task.task_id,
                "run_id": state["run_id"],
                "workflow_version": WORKFLOW_VERSION,
                "status": task.status,
                "completed_at": now_iso(),
            },
            task_id=task.task_id,
        )
        update = self._commit(
            state,
            task,
            step_id="finalize",
            status=task.status,
            progress=task.progress,
            message="LangGraph V2 工作流完成。" if task.status == "completed" else f"工作流以 {task.status} 结束。",
        )
        update["terminal_status"] = task.status
        return update

    def fail_task(self, state: dict) -> dict:
        task = self._task(state)
        ranking = state.get("candidate_ranking") or {}
        task.status = "failed"
        task.error = task.error or ranking.get("message") or "没有候选模型完成有效推理。"
        task.completed_at = now_iso()
        self._write_report(task)
        update = self._commit(
            state,
            task,
            step_id="fail_task",
            status="failed",
            progress=max(task.progress, 0.95),
            message=task.error,
            event_type="workflow_failed",
        )
        update["terminal_status"] = "failed"
        update["error"] = {"code": "V2_WORKFLOW_FAILED", "message": task.error}
        return update

    def cancel_task(self, state: dict) -> dict:
        task = self._task(state)
        task.status = "cancelled"
        task.completed_at = now_iso()
        update = self._commit(
            state,
            task,
            step_id="cancel_task",
            status="cancelled",
            progress=task.progress,
            message="LangGraph 在安全点停止已取消任务。",
            event_type="workflow_cancelled",
        )
        update["terminal_status"] = "cancelled"
        return update

    def _prepare_confirmation(self, state: dict, operation: str) -> dict:
        task = self._task(state)
        recommendation = task.postprocess_recommendation or {}
        if operation == "denoise":
            task.status = "awaiting_denoise_confirmation"
            task.progress = max(task.progress, 0.94)
            payload = {
                "operation": "denoise",
                "recommended_model": recommendation.get("preferred_denoiser", "nafnet"),
                "reason": recommendation.get("denoise_reason", []),
                "risk": recommendation.get("denoise_risk", []),
            }
            message = "工作流已暂停，等待用户确认是否执行去噪。"
        else:
            task.status = "awaiting_sr_confirmation"
            task.progress = max(task.progress, 0.95)
            payload = {
                "operation": "super_resolution",
                "recommended_model": recommendation.get("preferred_sr_model", "realesrgan"),
                "scale": recommendation.get("preferred_scale", 2),
                "reason": recommendation.get("sr_reason", []),
                "risk": recommendation.get("sr_risk", []),
            }
            message = "工作流已暂停，等待用户确认是否执行超分辨率。"
        task.pending_confirmation = payload
        update = self._commit(
            state,
            task,
            step_id=f"prepare_{operation}_confirmation",
            status=task.status,
            progress=task.progress,
            message=message,
            event_type="user_confirmation_required",
            metadata=payload,
        )
        update["pending_confirmation"] = payload
        update["next_postprocess_operation"] = operation
        return update

    def _register_tools(self) -> None:
        self.tools.register(ToolDefinition(
            name="analyze_image",
            description="Analyze a registered RGB image using local deterministic metrics.",
            input_schema=AnalyzeImageInput,
            handler=lambda payload, _context: {"analysis": self.deps.analyzer.analyze(payload.image_path).model_dump(mode="json")},
            risk_level="local_read",
        ))
        self.tools.register(ToolDefinition(
            name="inspect_runtime",
            description="Inspect local hardware and the honest model registry snapshot.",
            input_schema=InspectHardwareInput,
            handler=lambda _payload, _context: {"hardware": self.deps.hardware.inspect(), "models": self.deps.registry.list()},
            risk_level="local_read",
        ))
        self.tools.register(ToolDefinition(
            name="retrieve_context",
            description="Retrieve bounded allowlisted local project knowledge.",
            input_schema=RetrieveContextInput,
            handler=self._handle_retrieve_context,
            risk_level="local_read",
        ))
        self.tools.register(ToolDefinition(
            name="execute_candidate",
            description="Execute one planned enhancement candidate on the original input and evaluate its real output.",
            input_schema=ExecuteCandidateInput,
            handler=self._handle_execute_candidate,
            risk_level="local_inference",
            supports_cancel=True,
        ))

    def _handle_retrieve_context(self, payload: RetrieveContextInput, _context: ToolContext) -> dict:
        items = self.deps.context_retrieval.retrieve(
            user_request=payload.user_request,
            image_metrics=payload.image_metrics,
            available_models=payload.available_models,
            hardware_summary=payload.hardware_summary,
        )
        reference_loader = getattr(self.deps.context_retrieval, "model_reference", None)
        standards = reference_loader(payload.available_models) if callable(reference_loader) else []
        merged = []
        seen = set()
        for item in [*standards, *items]:
            key = item.item_id or f"{item.source}:{item.title}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return {"items": [item.model_dump(mode="json") for item in merged]}

    def _handle_execute_candidate(self, payload: ExecuteCandidateInput, _context: ToolContext) -> dict:
        candidate = CandidatePlanItem.model_validate(payload.candidate)
        record_id = f"{payload.task_id}:{candidate.candidate_id}"
        cached = self.deps.database.get_entity("candidate_result", record_id)
        if cached and cached.get("status") == "completed":
            output_path = cached.get("output_path")
            if output_path and Path(output_path).exists():
                return {"candidate_result": cached, "cache_hit": True}
        plan = CandidateExecutionPlan(
            task_mode="multi_candidate",
            mode="auto",
            priority=payload.priority,
            max_candidates=1,
            input_image_id=payload.task_id,
            candidates=[candidate],
            notes=["LangGraph candidate subgraph executes one candidate from original input only."],
        )
        results = self.deps.executor.execute(
            plan=plan,
            input_path=payload.input_path,
            priority=payload.priority,
            task_dir=payload.task_dir,
            cancel_check=lambda: self.cancel_check(payload.task_id),
            on_event=None,
            region_constraints=payload.region_constraints,
        )
        if not results:
            raise RuntimeError("candidate executor returned no result")
        result = results[0].model_dump(mode="json")
        self.deps.database.put_entity("candidate_result", record_id, result, task_id=payload.task_id)
        return {"candidate_result": result, "cache_hit": False}

    def _tool(self, state: dict, name: str, payload: dict, *, step_id: str):
        context = ToolContext(task_id=state["task_id"], run_id=state["run_id"], step_id=step_id)
        result = self.tool_router.execute(name, payload, context)
        if not result.success:
            raise RuntimeError(f"{result.error_code}: {result.error_message}")
        return result

    def _commit(
        self,
        state: dict,
        task: TaskRecord,
        *,
        step_id: str,
        status: str,
        progress: float,
        message: str,
        event_type: str = "step_completed",
        metadata: dict | None = None,
        append_log: bool = True,
    ) -> dict:
        task.status = status
        task.progress = float(progress)
        event = make_event(
            task_id=task.task_id,
            run_id=state["run_id"],
            step_id=step_id,
            event_type=event_type,
            message=message,
            status=status,
            metadata=metadata,
        )
        graph_message = make_message(
            role="assistant",
            content=message,
            source="visionrestore_graph",
            step_id=step_id,
            metadata={"status": status, **(metadata or {})},
        )
        task.workflow_events = self._append_unique(task.workflow_events, [event], "event_id")[-300:]
        task.messages = self._append_unique(task.messages, [graph_message], "message_id")[-300:]
        if append_log and (not task.logs or task.logs[-1] != message):
            task.logs.append(message)
        self.save_task(task)
        self._persist_event(event)
        return {
            "task_projection": task.model_dump(mode="json"),
            "current_phase": status,
            "events": [event],
            "messages": [graph_message],
        }

    def _persist_event(self, event: dict) -> None:
        self.deps.database.put_entity("workflow_event", event["event_id"], event, task_id=event["task_id"])

    def _merge_trace_into_task(self, task: TaskRecord, state: dict) -> None:
        task.workflow_events = self._append_unique(task.workflow_events, state.get("events", []), "event_id")[-300:]
        task.messages = self._append_unique(task.messages, state.get("messages", []), "message_id")[-300:]

    def _write_report(self, task: TaskRecord) -> None:
        if not task.report_file_id:
            task.report_file_id = str(uuid4())
        self.deps.report.write_reports(task.model_dump(mode="json"), self.deps.settings.report_dir / task.report_file_id)

    def _region_retry_needed(self, state: dict) -> bool:
        constraints = state.get("region_constraints", [])
        plan = state.get("candidate_plan") or {}
        if not constraints or int(plan.get("max_candidates") or 0) >= 3:
            return False
        completed = [item for item in state.get("candidate_results", []) if item.get("status") == "completed"]
        if not completed:
            return False
        return all(((item.get("metrics") or {}).get("region_constraints") or {}).get("severe_failure") for item in completed)

    @staticmethod
    def _operation_from_status(status: str | None) -> str:
        if status == "awaiting_denoise_confirmation":
            return "denoise"
        if status == "awaiting_sr_confirmation":
            return "super_resolution"
        return "none"

    @staticmethod
    def _task(state: dict) -> TaskRecord:
        return TaskRecord.model_validate(state["task_projection"])

    @staticmethod
    def _append_unique(left: list[dict], right: list[dict], key_name: str) -> list[dict]:
        output = list(left or [])
        seen = {item.get(key_name) for item in output if isinstance(item, dict)}
        for item in right or []:
            key = item.get(key_name)
            if key and key in seen:
                continue
            output.append(item)
            if key:
                seen.add(key)
        return output

    @staticmethod
    def _tool_metadata(tool_result) -> dict:
        return {
            "tool_call_id": tool_result.call_id,
            "tool_name": tool_result.tool_name,
            "tool_version": tool_result.tool_version,
            "duration_ms": tool_result.duration_ms,
        }
