from __future__ import annotations

from langgraph.types import Command

from visionrestore.graph.builder import build_visionrestore_graph
from visionrestore.graph.checkpoint import get_graph_checkpointer
from visionrestore.graph.runtime import WORKFLOW_VERSION, VisionRestoreDependencies, VisionRestoreGraphRuntime
from visionrestore.schemas.common import now_iso
from visionrestore.schemas.postprocess import PostprocessDecision, PostprocessResult
from visionrestore.schemas.task import TaskRecord


class LangGraphEnhancementAgentV2:
    workflow_version = WORKFLOW_VERSION

    def __init__(self, *, checkpointer=None, dependencies: VisionRestoreDependencies | None = None, cancel_check=None):
        self.checkpointer = checkpointer or get_graph_checkpointer()
        self.dependencies = dependencies
        self.cancel_check = cancel_check

    def run(self, task: TaskRecord, input_file: dict, save_task):
        runtime = VisionRestoreGraphRuntime(
            save_task=save_task,
            cancel_check=self.cancel_check,
            dependencies=self.dependencies,
        )
        graph = build_visionrestore_graph(runtime, checkpointer=self.checkpointer)
        config = self._config(task.task_id)
        try:
            graph.invoke(runtime.initial_state(task, input_file), config=config)
        except Exception as exc:
            latest = TaskRecord.model_validate(runtime.deps.database.get_task(task.task_id) or task.model_dump())
            latest.status = "failed"
            latest.error = f"LangGraph workflow failed: {exc}"
            latest.completed_at = now_iso()
            latest.logs.append(latest.error)
            save_task(latest)

    def resume(self, task: TaskRecord, decision: PostprocessDecision, save_task) -> tuple[TaskRecord, PostprocessResult]:
        runtime = VisionRestoreGraphRuntime(
            save_task=save_task,
            cancel_check=self.cancel_check,
            dependencies=self.dependencies,
        )
        graph = build_visionrestore_graph(runtime, checkpointer=self.checkpointer)
        result = graph.invoke(Command(resume=decision.model_dump(mode="json")), config=self._config(task.task_id))
        latest_payload = runtime.deps.database.get_task(task.task_id) or result.get("task_projection") or task.model_dump(mode="json")
        latest = TaskRecord.model_validate(latest_payload)
        raw_postprocess = result.get("last_postprocess_result")
        if not raw_postprocess:
            raw_postprocess = {
                "task_id": task.task_id,
                "operation": decision.operation,
                "decision": decision.decision,
                "accepted": False,
                "executed": False,
                "next_status": latest.status,
                "message": "工作流已恢复，但没有产生后处理结果。",
            }
        return latest, PostprocessResult.model_validate(raw_postprocess)

    @staticmethod
    def _config(task_id: str) -> dict:
        return {
            "configurable": {"thread_id": task_id},
            "max_concurrency": 1,
            "recursion_limit": 100,
        }
