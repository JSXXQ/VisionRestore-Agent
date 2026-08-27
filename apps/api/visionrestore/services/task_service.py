from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from threading import Lock
from uuid import uuid4
from visionrestore.agent.enhancement_agent import EnhancementAgent
from visionrestore.agent.enhancement_agent_v2 import EnhancementAgentV2
from visionrestore.graph.agent import LangGraphEnhancementAgentV2
from visionrestore.graph.runtime import WORKFLOW_VERSION
from visionrestore.schemas.postprocess import PostprocessDecision, PostprocessResult
from visionrestore.schemas.common import now_iso
from visionrestore.schemas.task import EnhancementPlan, TaskCreate, TaskRecord
from visionrestore.storage.database import Database
from visionrestore.services.artifact_lineage import ArtifactLineageService

class TaskService:
    def __init__(self):
        self.db = Database()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.cancelled: set[str] = set()
        self._resume_locks: defaultdict[str, Lock] = defaultdict(Lock)

    def create(self, request: TaskCreate) -> TaskRecord:
        file_record = self.db.get_file(request.image_id)
        if not file_record:
            raise ValueError("输入图像不存在")
        task = TaskRecord(
            task_id=str(uuid4()),
            status="queued",
            image_id=request.image_id,
            user_goal=request.user_goal,
            mode=request.mode,
            priority=request.priority,
            analysis_mode=request.analysis_mode,
            parameters=request.parameters,
            created_at=now_iso(),
            logs=["任务已创建，后台队列已接收。"],
        )
        if request.model_id:
            checkpoint = request.checkpoint_id or request.weight_id
            task.plan = EnhancementPlan(
                input_id=request.image_id,
                user_goal=request.user_goal,
                mode=request.mode,
                priority=request.priority,
                selected_model=request.model_id,
                selected_checkpoint=checkpoint,
                selection_reason=f"用户手动指定模型 {request.model_id}。",
                parameters={**request.parameters, "checkpoint_id": checkpoint},
            )
        ArtifactLineageService().create_task_layout(task.task_id)
        task.task_mode = str(request.parameters.get("task_mode") or "multi_candidate")
        requested_engine = str(request.parameters.get("workflow_engine") or "langgraph").lower()
        if task.task_mode == "single_candidate":
            task.workflow_engine = "legacy_single"
        elif requested_engine == "legacy":
            task.workflow_engine = "legacy_v2"
        else:
            task.workflow_engine = "langgraph"
            task.workflow_thread_id = task.task_id
            task.workflow_version = WORKFLOW_VERSION
        self.save(task)
        if task.task_mode == "single_candidate":
            agent = EnhancementAgent()
        elif task.workflow_engine == "legacy_v2":
            agent = EnhancementAgentV2()
        else:
            agent = LangGraphEnhancementAgentV2(cancel_check=lambda task_id: task_id in self.cancelled)
        self.pool.submit(agent.run, task, file_record, self.save)
        return task

    def resume_postprocess(self, task_id: str, decision: PostprocessDecision) -> tuple[TaskRecord, PostprocessResult]:
        with self._resume_locks[task_id]:
            task = self.get(task_id)
            if not task:
                raise ValueError("任务不存在")
            if task.workflow_engine != "langgraph":
                raise ValueError("该任务不是可恢复的 LangGraph 工作流")
            if task.status not in {"awaiting_denoise_confirmation", "awaiting_sr_confirmation"}:
                raise ValueError(f"任务当前状态 {task.status} 不接受后处理确认")
            expected_operation = (task.pending_confirmation or {}).get("operation")
            if expected_operation and decision.operation != expected_operation:
                raise ValueError(f"任务当前等待 {expected_operation}，不能提交 {decision.operation} 决策")
            agent = LangGraphEnhancementAgentV2(cancel_check=lambda current_id: current_id in self.cancelled)
            return agent.resume(task, decision, self.save)

    def save(self, task: TaskRecord):
        if task.task_id in self.cancelled and task.status not in {"completed", "failed"}:
            task.status = "cancelled"
            task.logs.append("收到取消请求。")
        self.db.put_task(task.task_id, task.model_dump())

    def get(self, task_id: str) -> TaskRecord | None:
        data = self.db.get_task(task_id)
        return TaskRecord(**data) if data else None

    def cancel(self, task_id: str) -> TaskRecord | None:
        self.cancelled.add(task_id)
        task = self.get(task_id)
        if task and task.status not in {"completed", "failed"}:
            task.status = "cancelled"
            task.completed_at = now_iso()
            task.logs.append("任务已标记取消；正在运行的模型调用会在安全点停止。")
            self.save(task)
        return task

    def list(self) -> list[TaskRecord]:
        return [TaskRecord(**item) for item in self.db.list_tasks()]

task_service = TaskService()
