from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from PIL import Image

from visionrestore.adapters.registry import ModelRegistry
from visionrestore.agent.checkpoint_selector import CheckpointSelection, CheckpointSelector
from visionrestore.core.config import get_settings
from visionrestore.core.model_config import external_python, file_sha256, get_postprocess_rules
from visionrestore.schemas.common import now_iso
from visionrestore.schemas.postprocess import PostprocessDecision, PostprocessResult
from visionrestore.schemas.task import CandidateResult
from visionrestore.services.candidate_evaluator import CandidateEvaluator
from visionrestore.services.evaluator import QualityEvaluator
from visionrestore.services.image_analyzer import ImageAnalyzer
from visionrestore.services.iqa_service import IQAService
from visionrestore.services.region_constraints import RegionConstraintService
from visionrestore.storage.database import Database
from visionrestore.utils.file_security import resolve_registered_path


class PostprocessController:
    def __init__(self, registry=None, evaluator=None, database=None, candidate_evaluator=None):
        self.registry = registry or ModelRegistry()
        self.evaluator = evaluator or QualityEvaluator()
        self.candidate_evaluator = candidate_evaluator or CandidateEvaluator()
        self.db = database or Database()
        self.settings = get_settings()
        self.rules = get_postprocess_rules().get("residual_degradation", {})
        self.region_constraints = RegionConstraintService()
        self.checkpoint_selector = CheckpointSelector()

    def decide(self, *, task, decision: PostprocessDecision) -> PostprocessResult:
        if decision.operation not in {"denoise", "super_resolution"}:
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=False, message="未知后处理操作。")
        if decision.decision == "skip":
            next_status = self._next_status_after_denoise(task) if decision.operation == "denoise" else "completed"
            task.status = next_status
            if next_status == "completed":
                task.progress = 1.0
                task.completed_at = now_iso()
            else:
                task.progress = max(float(getattr(task, "progress", 0) or 0), 0.95)
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=True, executed=False, next_status=next_status, message="用户选择跳过该后处理步骤。")
        if decision.decision not in {"accept", "choose_model"}:
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=False, message="不支持的后处理决策。")
        if decision.operation == "super_resolution":
            return self._run_super_resolution(task, decision)

        model_id = decision.model_id or "nafnet"
        if model_id == "nafnet" and decision.decision != "choose_model":
            return PostprocessResult(
                task_id=task.task_id,
                operation=decision.operation,
                decision=decision.decision,
                accepted=False,
                executed=False,
                next_status="awaiting_denoise_confirmation",
                model_id=model_id,
                message="NAFNet 去噪需要通过更换模型确认执行，不能被普通 accept 隐式触发。",
            )
        if model_id not in {"lpdm", "nafnet"}:
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=False, model_id=model_id, message="该模型不是允许的去噪后处理模型。")
        if model_id == "lpdm":
            return PostprocessResult(
                task_id=task.task_id,
                operation=decision.operation,
                decision=decision.decision,
                accepted=False,
                executed=False,
                next_status="awaiting_denoise_confirmation",
                model_id=model_id,
                message="LPDM 在当前 Windows/PyTorch 环境下全尺寸后处理不稳定，已暂停执行；请使用 NAFNet 去噪。",
                metadata={"model_status": "lpdm_runtime_unstable", "recommended_model": "nafnet"},
            )
        try:
            adapter = self.registry.get(model_id)
            status = adapter.get_status()
        except KeyError:
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=False, model_id=model_id, message="后处理模型不存在。")
        if not status.available:
            next_status = self._advance_after_unavailable_denoise(task)
            return PostprocessResult(
                task_id=task.task_id,
                operation=decision.operation,
                decision=decision.decision,
                accepted=True,
                executed=False,
                next_status=next_status,
                model_id=model_id,
                message="已记录用户确认；模型尚未 ready，保留原最佳结果并继续后续流程。",
                metadata={"model_status": status.status_message},
            )
        if not getattr(task, "best_result", None) or not task.best_result.output_file_id:
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=False, executed=False, next_status="completed", model_id=model_id, message="任务尚未产生可用于后处理的最佳增强结果。")

        explicit_checkpoint = str(
            decision.parameters.get("checkpoint_id")
            or decision.parameters.get("weight_id")
            or ""
        ) or None
        checkpoint_id = explicit_checkpoint or ""
        checkpoint_selection = CheckpointSelection()
        previous_best = task.best_result
        previous_score = self._candidate_final_score(previous_best)
        try:
            input_path = self._candidate_output_path(previous_best.output_file_id, task_id=task.task_id)
            checkpoint_selection = self._select_postprocess_checkpoint(
                task=task,
                model_id=model_id,
                status=status,
                input_path=input_path,
                operation="denoise",
                explicit_checkpoint=explicit_checkpoint,
            )
            if not checkpoint_selection.checkpoint_id:
                raise RuntimeError(checkpoint_selection.reason)
            checkpoint_id = checkpoint_selection.checkpoint_id
            self._record_postprocess_checkpoint_selection(
                task=task,
                model_id=model_id,
                operation="denoise",
                selection=checkpoint_selection,
            )
            output_id = str(uuid4())
            postprocess_index = self._next_postprocess_index(task)
            output_path = self.settings.data_dir / "tasks" / task.task_id / "postprocess" / f"candidate_{postprocess_index:02d}" / f"{output_id}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            worker_parameters = {**decision.parameters, "checkpoint_id": checkpoint_id}
            if model_id == "lpdm":
                original = self.db.get_file(getattr(task, "image_id", ""))
                if original:
                    worker_parameters["cond_path"] = str(resolve_registered_path(original["relative_path"]))
                worker_parameters.setdefault("max_edge", 512)
            result = adapter.enhance(
                image_path=str(input_path),
                output_path=str(output_path),
                device=str(decision.parameters.get("device") or "cuda"),
                precision=str(decision.parameters.get("precision") or "fp32"),
                parameters=worker_parameters,
            )
            original_path = self._original_input_path(task, fallback=input_path)
            metrics, evaluated = self._evaluate_postprocess_output(
                task=task,
                original_path=original_path,
                output_path=Path(result.output_path),
                candidate_id=f"postprocess_{postprocess_index:02d}",
                model_id=model_id,
                checkpoint_id=checkpoint_id,
                operation="denoise",
                expected_scale=1,
                runtime_ms=result.runtime_ms,
                peak_memory_mb=result.peak_memory_mb,
            )
            self._register_output_file(output_id, output_path)
            candidate = CandidateResult(
                candidate_id=f"postprocess_{postprocess_index:02d}",
                model_id=model_id,
                checkpoint_id=checkpoint_id,
                role="postprocess",
                output_file_id=output_id,
                output_url=f"/api/v1/files/{output_id}",
                status="completed",
                score=float(evaluated.score),
                final_score=float(evaluated.score),
                metrics={**metrics, "postprocess_operation": "denoise", "postprocess_input_file_id": previous_best.output_file_id, "checkpoint_selection": self._selection_payload(checkpoint_selection)},
                parameters={**result.parameters, "role": "postprocess", "operation": "denoise", "input_file_id": previous_best.output_file_id, "checkpoint_selection": self._selection_payload(checkpoint_selection)},
                runtime_ms=result.runtime_ms,
                peak_memory_mb=result.peak_memory_mb,
                is_mock=result.is_mock,
                adapter_class=result.adapter_class,
                checkpoint_path=result.checkpoint_path,
                checkpoint_sha256=result.checkpoint_sha256,
                worker_python=external_python(),
                device=result.device,
                precision=result.precision,
                input_path=str(input_path),
                output_path=str(output_path),
                input_sha256=result.input_sha256,
                output_sha256=result.output_sha256,
                input_size=self._image_size(input_path),
                output_size=self._image_size(output_path),
                warnings=list(result.logs or []),
            )
            after_score = float(candidate.score)
            adopted = not evaluated.eliminated and after_score >= previous_score
            task.candidates.append(candidate)
            if adopted:
                task.best_result = candidate
                message = f"后处理已采用 {model_id}:{checkpoint_id}；重新评分 {after_score:.2f} >= 去噪前 {previous_score:.2f}。"
            else:
                task.best_result = previous_best
                message = f"后处理结果评分下降，已自动回退到去噪前最佳增强结果；去噪前 {previous_score:.2f}，去噪后 {after_score:.2f}。"
            history = getattr(task, "postprocess_history", []) or []
            history.append({
                "operation": "denoise",
                "model_id": model_id,
                "checkpoint_id": checkpoint_id,
                "before_score": previous_score,
                "after_score": after_score,
                "adopted": adopted,
                "rollback_performed": not adopted,
                "candidate_output_file_id": candidate.output_file_id,
                "kept_output_file_id": task.best_result.output_file_id if task.best_result else None,
            })
            task.postprocess_history = history
            task.status = self._next_status_after_denoise(task)
            task.progress = 0.96 if task.status == "awaiting_sr_confirmation" else 1.0
            if task.status == "completed":
                task.completed_at = now_iso()
            task.final_recommendation = message
            return PostprocessResult(
                task_id=task.task_id,
                operation=decision.operation,
                decision=decision.decision,
                accepted=True,
                executed=True,
                next_status=task.status,
                model_id=model_id,
                message=message,
                rollback_performed=not adopted,
                metadata={"candidate": candidate.model_dump(), "model_status": status.status_message, "before_score": previous_score, "after_score": after_score, "adopted": adopted, "score_layers": evaluated.layers, "score_evidence": evaluated.evidence, "checkpoint_selection": self._selection_payload(checkpoint_selection)},
            )
        except Exception as exc:
            history = getattr(task, "postprocess_history", []) or []
            history.append({
                "operation": "denoise",
                "model_id": model_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_score": checkpoint_selection.score,
                "checkpoint_selection_mode": checkpoint_selection.mode,
                "before_score": previous_score,
                "after_score": None,
                "adopted": False,
                "executed": False,
                "rollback_performed": False,
                "candidate_output_file_id": None,
                "kept_output_file_id": previous_best.output_file_id if previous_best else None,
                "error": str(exc),
            })
            task.postprocess_history = history
            next_status = self._advance_after_unavailable_denoise(task)
            task.final_recommendation = f"后处理执行失败，已保留原最佳结果并继续后续流程：{exc}"
            return PostprocessResult(
                task_id=task.task_id,
                operation=decision.operation,
                decision=decision.decision,
                accepted=True,
                executed=False,
                next_status=next_status,
                model_id=model_id,
                message=task.final_recommendation,
                metadata={"model_status": status.status_message, "error": str(exc), "checkpoint_selection": self._selection_payload(checkpoint_selection)},
            )

    def rollback(self, *, task) -> PostprocessResult:
        best = getattr(task, "best_result", None)
        if not best or (best.parameters or {}).get("role") != "postprocess":
            return PostprocessResult(task_id=task.task_id, operation="rollback", decision="rollback", accepted=False, executed=False, next_status=getattr(task, "status", "completed"), message="当前最终结果不是后处理结果，无需撤回。")
        input_file_id = (best.parameters or {}).get("input_file_id") or (best.metrics or {}).get("postprocess_input_file_id")
        target = next((item for item in task.candidates if item.output_file_id == input_file_id), None) if input_file_id else None
        if target is None:
            completed_main = [item for item in task.candidates if (item.parameters or {}).get("role") != "postprocess" and item.status == "completed" and item.output_file_id]
            target = max(completed_main, key=lambda item: item.score) if completed_main else None
        if target is None:
            return PostprocessResult(task_id=task.task_id, operation="rollback", decision="rollback", accepted=False, executed=False, next_status="completed", message="未找到可恢复的主增强结果。", metadata={"postprocess_output_file_id": best.output_file_id})
        task.best_result = target
        task.status = "completed"
        task.progress = 1.0
        task.completed_at = now_iso()
        task.final_recommendation = f"已撤回后处理，恢复为 {target.model_id}:{target.checkpoint_id or '-'} 主增强结果。"
        return PostprocessResult(task_id=task.task_id, operation="rollback", decision="rollback", accepted=True, executed=True, next_status="completed", model_id=target.model_id, message=task.final_recommendation, rollback_performed=True, metadata={"restored_candidate": target.model_dump(), "postprocess_output_file_id": best.output_file_id})

    def _run_super_resolution(self, task, decision: PostprocessDecision) -> PostprocessResult:
        model_id = decision.model_id or "realesrgan"
        if model_id != "realesrgan":
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=False, model_id=model_id, message="当前超分后处理仅支持 Real-ESRGAN。")
        if not getattr(task, "best_result", None) or not task.best_result.output_file_id:
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=False, executed=False, next_status="completed", model_id=model_id, message="任务尚未产生可用于超分的最佳结果。")
        sr_rules = self.rules.get("super_resolution", {})
        scale = int(decision.scale or decision.parameters.get("scale") or sr_rules.get("default_scale", 2))
        if scale not in {2, 4}:
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=False, model_id=model_id, message="Real-ESRGAN 仅支持 x2 或 x4。")
        if scale == 4 and not bool(decision.parameters.get("allow_x4") or decision.decision == "choose_model"):
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=False, model_id=model_id, next_status="awaiting_sr_confirmation", message="x4 超分成本较高，需要用户明确确认。")
        explicit_checkpoint = str(
            decision.parameters.get("checkpoint_id")
            or decision.parameters.get("weight_id")
            or ""
        ) or None
        checkpoint_id = explicit_checkpoint or ""
        checkpoint_selection = CheckpointSelection()
        try:
            adapter = self.registry.get(model_id)
            status = adapter.get_status()
        except KeyError:
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=False, model_id=model_id, message="Real-ESRGAN 模型不存在。")
        if not status.available:
            task.status = "completed"
            task.progress = 1.0
            task.completed_at = now_iso()
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=True, executed=False, next_status="completed", model_id=model_id, message="已跳过超分：Real-ESRGAN 尚未 ready，正式模式不允许伪造结果。", metadata={"model_status": status.status_message, "formal_mode_allows_mock": False})
        previous_best = task.best_result
        previous_score = self._candidate_final_score(previous_best)
        try:
            input_path = self._candidate_output_path(previous_best.output_file_id, task_id=task.task_id)
            checkpoint_selection = self._select_postprocess_checkpoint(
                task=task,
                model_id=model_id,
                status=status,
                input_path=input_path,
                operation="super_resolution",
                explicit_checkpoint=explicit_checkpoint,
                scale=scale,
            )
            if not checkpoint_selection.checkpoint_id:
                raise RuntimeError(checkpoint_selection.reason)
            checkpoint_id = checkpoint_selection.checkpoint_id
            self._record_postprocess_checkpoint_selection(
                task=task,
                model_id=model_id,
                operation="super_resolution",
                selection=checkpoint_selection,
            )
            width, height = self._image_size(input_path) or [0, 0]
            expected_pixels = int(width * height * scale * scale)
            max_pixels = int(sr_rules.get("max_output_pixels", 50_000_000))
            if expected_pixels > max_pixels:
                task.status = "completed"
                task.progress = 1.0
                task.completed_at = now_iso()
                return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=True, executed=False, next_status="completed", model_id=model_id, message=f"已跳过超分：预计输出 {expected_pixels} 像素，超过安全上限 {max_pixels}。", metadata={"input_size": [width, height], "scale": scale, "expected_pixels": expected_pixels, "max_output_pixels": max_pixels})
            output_id = str(uuid4())
            postprocess_index = self._next_postprocess_index(task)
            output_path = self.settings.data_dir / "tasks" / task.task_id / "postprocess" / f"candidate_{postprocess_index:02d}" / f"{output_id}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            params = {**decision.parameters, "checkpoint_id": checkpoint_id, "scale": scale, "outscale": scale, "tile_size": 512, "tile_pad": 10}
            result = adapter.enhance(image_path=str(input_path), output_path=str(output_path), device=str(decision.parameters.get("device") or "cuda"), precision=str(decision.parameters.get("precision") or "fp16"), parameters=params)
            actual_output = Path(result.output_path)
            original_path = self._original_input_path(task, fallback=input_path)
            metrics, evaluated = self._evaluate_postprocess_output(
                task=task,
                original_path=original_path,
                output_path=actual_output,
                candidate_id=f"postprocess_{postprocess_index:02d}",
                model_id=model_id,
                checkpoint_id=checkpoint_id,
                operation="super_resolution",
                expected_scale=scale,
                runtime_ms=result.runtime_ms,
                peak_memory_mb=result.peak_memory_mb,
            )
            metrics = {
                **metrics,
                "pre_sr_score": round(previous_score, 2),
                "post_sr_score": round(evaluated.score, 2),
                "score_delta": round(evaluated.score - previous_score, 2),
                "sr_scale": scale,
                "sr_expected_size": [int(width * scale), int(height * scale)],
                "sr_output_size": self._image_size(actual_output),
                "checkpoint_selection": self._selection_payload(checkpoint_selection),
            }
            self._register_output_file(output_id, actual_output)
            candidate = CandidateResult(
                candidate_id=f"postprocess_{postprocess_index:02d}", model_id=model_id, checkpoint_id=checkpoint_id, role="postprocess", output_file_id=output_id, output_url=f"/api/v1/files/{output_id}", status="completed", score=float(evaluated.score), final_score=float(evaluated.score), metrics=metrics,
                parameters={**result.parameters, "role": "postprocess", "operation": "super_resolution", "input_file_id": previous_best.output_file_id, "checkpoint_selection": self._selection_payload(checkpoint_selection)}, runtime_ms=result.runtime_ms, peak_memory_mb=result.peak_memory_mb, is_mock=result.is_mock, adapter_class=result.adapter_class, checkpoint_path=result.checkpoint_path, checkpoint_sha256=result.checkpoint_sha256, worker_python=external_python(), device=result.device, precision=result.precision, input_path=str(input_path), output_path=str(actual_output), input_sha256=result.input_sha256, output_sha256=result.output_sha256, input_size=self._image_size(input_path), output_size=self._image_size(actual_output), warnings=list(result.logs or [])
            )
            after_score = float(candidate.score)
            adopted = not evaluated.eliminated and after_score >= previous_score
            candidate.metrics = {
                **candidate.metrics,
                "adopted": adopted,
                "rollback_performed": not adopted,
                "rollback_reason": evaluated.reasons if evaluated.eliminated else (["score_decreased"] if not adopted else []),
            }
            task.candidates.append(candidate)
            if adopted:
                task.best_result = candidate
                message = f"Real-ESRGAN x{scale} 已采用；统一重评 {after_score:.2f} >= 超分前 {previous_score:.2f}。"
            else:
                task.best_result = previous_best
                message = f"Real-ESRGAN x{scale} 评分下降，已自动回退到超分前结果；超分前 {previous_score:.2f}，超分后 {after_score:.2f}。"
            history = getattr(task, "postprocess_history", []) or []
            history.append({"operation": "super_resolution", "model_id": model_id, "checkpoint_id": checkpoint_id, "checkpoint_score": checkpoint_selection.score, "checkpoint_selection_mode": checkpoint_selection.mode, "before_score": previous_score, "after_score": after_score, "adopted": adopted, "rollback_performed": not adopted, "candidate_output_file_id": candidate.output_file_id, "kept_output_file_id": task.best_result.output_file_id if task.best_result else None})
            task.postprocess_history = history
            task.status = "completed"
            task.progress = 1.0
            task.completed_at = now_iso()
            task.final_recommendation = message
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=True, executed=True, next_status="completed", model_id=model_id, message=message, rollback_performed=not adopted, metadata={"candidate": candidate.model_dump(), "model_status": status.status_message, "before_score": previous_score, "after_score": after_score, "adopted": adopted, "scale": scale, "score_layers": evaluated.layers, "score_evidence": evaluated.evidence, "checkpoint_selection": self._selection_payload(checkpoint_selection)})
        except Exception as exc:
            task.status = "completed"
            task.progress = 1.0
            task.completed_at = now_iso()
            return PostprocessResult(task_id=task.task_id, operation=decision.operation, decision=decision.decision, accepted=True, executed=False, next_status="completed", model_id=model_id, message=f"Real-ESRGAN 执行失败：{exc}；已保留超分前结果。", metadata={"model_status": status.status_message, "error": str(exc), "formal_mode_allows_mock": False, "checkpoint_selection": self._selection_payload(checkpoint_selection)})

    def _select_postprocess_checkpoint(
        self,
        *,
        task,
        model_id: str,
        status,
        input_path: Path,
        operation: str,
        explicit_checkpoint: str | None,
        scale: int | None = None,
    ) -> CheckpointSelection:
        intent = getattr(task, "user_intent", None) or SimpleNamespace(
            priority=getattr(task, "priority", "balanced"),
            scene="unknown",
            preferences=SimpleNamespace(),
            raw_text=getattr(task, "user_goal", ""),
        )
        status_payload = (
            status.model_dump()
            if hasattr(status, "model_dump")
            else {
                "available": bool(getattr(status, "available", False)),
                "capabilities": getattr(status, "capabilities", {}) or {},
            }
        )
        return self.checkpoint_selector.select(
            model_id=model_id,
            model_status=status_payload,
            intent=intent,
            analysis=ImageAnalyzer().analyze(str(input_path)),
            hardware=getattr(task, "hardware_info", None) or {},
            semantic_analysis=getattr(task, "ai_analysis", None),
            manual_checkpoint=explicit_checkpoint,
            allow_non_auto_route=True,
            context={"operation": operation, "scale": scale},
        )

    def _record_postprocess_checkpoint_selection(
        self,
        *,
        task,
        model_id: str,
        operation: str,
        selection: CheckpointSelection,
    ) -> None:
        record = {
            "role": "postprocess",
            "operation": operation,
            "model_id": model_id,
            "checkpoint_id": selection.checkpoint_id,
            "checkpoint_score": selection.score,
            "checkpoint_selection_mode": selection.mode,
            "checkpoint_evidence": selection.evidence,
            "checkpoint_candidates": selection.ranked_candidates,
        }
        current = list(getattr(task, "checkpoint_candidates", []) or [])
        current = [
            item
            for item in current
            if not (
                item.get("role") == "postprocess"
                and item.get("operation") == operation
            )
        ]
        task.checkpoint_candidates = [*current, record]

    @staticmethod
    def _selection_payload(selection: CheckpointSelection) -> dict:
        return {
            "checkpoint_id": selection.checkpoint_id,
            "checkpoint_score": selection.score,
            "mode": selection.mode,
            "evidence": selection.evidence,
            "ranked_candidates": selection.ranked_candidates,
            "planning_score_is_unchanged": True,
            "final_score_is_unchanged": True,
        }

    def _evaluate_postprocess_output(
        self,
        *,
        task,
        original_path: Path,
        output_path: Path,
        candidate_id: str,
        model_id: str,
        checkpoint_id: str,
        operation: str,
        expected_scale: int,
        runtime_ms: int,
        peak_memory_mb: float,
    ):
        validity = self._postprocess_validity(original_path, output_path, expected_scale)
        try:
            metrics = self.evaluator.evaluate(
                str(original_path),
                str(output_path),
                getattr(task, "priority", "balanced"),
                runtime_ms,
                peak_memory_mb,
                expected_scale=expected_scale,
            )
        except Exception as exc:
            after = ImageAnalyzer().analyze(str(output_path)) if output_path.exists() else None
            metrics = {
                "score": 0.0,
                "components": {},
                "mean_luminance_after": getattr(after, "mean_luminance", 0),
                "overexposed_pixel_ratio_after": getattr(after, "overexposed_pixel_ratio", 0),
                "color_cast_index_after": getattr(after, "color_cast_index", 0),
                "structure_keep_estimate": 0,
                "runtime_ms": runtime_ms,
                "peak_memory_mb": peak_memory_mb,
                "evaluation_error": str(exc),
            }
        metrics = {
            **metrics,
            **validity,
            "postprocess_operation": operation,
            "postprocess_input_file_id": task.best_result.output_file_id if task.best_result else None,
            "evaluation_reference": "original_input",
        }
        if getattr(task, "region_constraints", None):
            metrics["region_constraints"] = self.region_constraints.evaluate(
                input_path=str(original_path),
                output_path=str(output_path),
                constraints=task.region_constraints,
            )
        iqa_service = IQAService(database=self.db)
        iqa_status = iqa_service.status()
        if iqa_status.get("available"):
            iqa_result = iqa_service.evaluate_image(output_path)
            metrics["iqa"] = iqa_result.get("raw", {})
            metrics["iqa_normalized"] = iqa_result.get("normalized", {})
            metrics["iqa_detail"] = iqa_result
        else:
            metrics["iqa"] = {"status": "unavailable"}
            metrics["iqa_normalized"] = {}
            metrics["iqa_detail"] = {
                "status": "unavailable",
                "fallback": "local_perceptual_score",
                "reason": iqa_status.get("status_message"),
            }
        evaluated = self.candidate_evaluator.score(
            {
                "candidate_id": candidate_id,
                "model_id": model_id,
                "checkpoint_id": checkpoint_id,
                "status": "completed",
                "metrics": metrics,
                "runtime_ms": runtime_ms,
                "peak_memory_mb": peak_memory_mb,
            },
            getattr(task, "priority", "balanced"),
        )
        metrics = {
            **metrics,
            "score": evaluated.score,
            "final_score": evaluated.score,
            "score_layers": evaluated.layers,
            "score_reasons": evaluated.reasons,
            "score_evidence": evaluated.evidence,
            "planning_score_is_not_final_score": True,
        }
        return metrics, evaluated

    @staticmethod
    def _postprocess_validity(input_path: Path, output_path: Path, expected_scale: int) -> dict:
        metrics = {
            "output_exists": output_path.exists(),
            "decode_ok": False,
            "size_match": False,
            "has_nan": False,
            "has_inf": False,
        }
        if not output_path.exists():
            return metrics
        try:
            with Image.open(input_path) as source, Image.open(output_path) as output:
                output.convert("RGB")
                expected_size = (source.width * expected_scale, source.height * expected_scale)
                metrics.update(
                    {
                        "decode_ok": True,
                        "size_match": output.size == expected_size,
                        "width": output.width,
                        "height": output.height,
                        "expected_output_size": list(expected_size),
                    }
                )
        except Exception as exc:
            metrics["decode_error"] = str(exc)
        return metrics

    def _original_input_path(self, task, *, fallback: Path) -> Path:
        record = self.db.get_file(getattr(task, "image_id", ""))
        if record:
            path = resolve_registered_path(record["relative_path"])
            if path.exists():
                return path
        return Path(fallback)

    @staticmethod
    def _candidate_final_score(candidate) -> float:
        if candidate is None:
            return 0.0
        value = candidate.final_score if candidate.final_score is not None else candidate.score
        return float(value or 0)

    @staticmethod
    def _next_status_after_denoise(task) -> str:
        recommendation = getattr(task, "postprocess_recommendation", None) or {}
        return (
            "awaiting_sr_confirmation"
            if bool(recommendation.get("super_resolution_recommended"))
            else "completed"
        )

    def _advance_after_unavailable_denoise(self, task) -> str:
        next_status = self._next_status_after_denoise(task)
        task.status = next_status
        if next_status == "completed":
            task.progress = 1.0
            task.completed_at = now_iso()
        else:
            task.progress = max(float(getattr(task, "progress", 0) or 0), 0.95)
        return next_status

    def _register_output_file(self, file_id: str, output_path: Path) -> None:
        with Image.open(output_path) as image:
            width, height = image.size
        relative_path = output_path.resolve().relative_to(self.settings.data_dir.resolve()).as_posix()
        payload = {
            "file_id": file_id,
            "relative_path": relative_path,
            "sha256": file_sha256(output_path),
            "mime_type": "image/png",
            "width": width,
            "height": height,
            "size_bytes": output_path.stat().st_size,
            "uploaded_at": now_iso(),
        }
        self.db.put_file(file_id, payload)

    def _candidate_output_path(self, file_id: str, *, task_id: str) -> Path:
        registered = self.db.get_file(file_id)
        if registered:
            return resolve_registered_path(registered["relative_path"])
        direct = self.settings.output_dir / f"{file_id}.png"
        if direct.exists():
            return direct
        task_root = self.settings.data_dir / "tasks" / task_id
        matches = list(task_root.rglob(f"{file_id}.png")) if task_root.exists() else []
        if matches:
            return matches[0]
        raise FileNotFoundError(f"候选输出文件不存在: {file_id}")

    def _next_postprocess_index(self, task) -> int:
        postprocess_root = self.settings.data_dir / "tasks" / task.task_id / "postprocess"
        used = [
            int(path.name.split("_")[-1])
            for path in postprocess_root.glob("candidate_*")
            if path.is_dir() and path.name.split("_")[-1].isdigit()
        ] if postprocess_root.exists() else []
        candidate_count = len([item for item in task.candidates if (item.parameters or {}).get("role") == "postprocess"])
        history_count = len(getattr(task, "postprocess_history", []) or [])
        return max([0, candidate_count, history_count, *used]) + 1

    def _image_size(self, path: Path) -> list[int] | None:
        try:
            with Image.open(path) as image:
                width, height = image.size
            return [int(width), int(height)]
        except Exception:
            return None
