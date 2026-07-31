from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from visionrestore.agent.enhancement_agent import EnhancementAgent
from visionrestore.schemas.common import now_iso
from visionrestore.schemas.task import EnhancementPlan, TaskCreate, TaskRecord
from visionrestore.storage.database import Database

class TaskService:
    def __init__(self):
        self.db = Database()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.cancelled: set[str] = set()

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
        self.save(task)
        self.pool.submit(EnhancementAgent().run, task, file_record, self.save)
        return task

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
