from visionrestore.adapters.registry import ModelRegistry
from visionrestore.schemas.postprocess import PostprocessDecision, PostprocessResult


class PostprocessController:
    def __init__(self, registry=None):
        self.registry = registry or ModelRegistry()

    def decide(self, *, task, decision: PostprocessDecision) -> PostprocessResult:
        if decision.operation not in {"denoise", "super_resolution"}:
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=False, message="未知后处理操作")
        if decision.decision == "skip":
            next_status = "awaiting_sr_confirmation" if decision.operation == "denoise" else "finalizing"
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=True, executed=False, next_status=next_status, message="用户选择跳过该后处理步骤。")
        if decision.decision not in {"accept", "choose_model"}:
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=False, message="不支持的后处理决策")
        model_id = decision.model_id or ("lpdm" if decision.operation == "denoise" else "mambair")
        try:
            status = self.registry.get(model_id).get_status()
        except KeyError:
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=False, model_id=model_id, message="后处理模型不存在")
        if not status.available:
            wait_status = "awaiting_denoise_confirmation" if decision.operation == "denoise" else "awaiting_sr_confirmation"
            return PostprocessResult(
                task_id=task.task_id,
                operation=decision.operation,
                decision=decision.decision,
                accepted=True,
                executed=False,
                next_status=wait_status,
                model_id=model_id,
                message="已记录用户确认；模型尚未ready，当前阶段不伪造后处理执行。",
                metadata={"model_status": status.status_message},
            )
        running_status = "denoising" if decision.operation == "denoise" else "super_resolving"
        return PostprocessResult(
            task_id=task.task_id,
            operation=decision.operation,
            decision=decision.decision,
            accepted=True,
            executed=False,
            next_status=running_status,
            model_id=model_id,
            message="模型已ready；真实执行器将在后续阶段接管该步骤。",
            metadata={"model_status": status.status_message},
        )
