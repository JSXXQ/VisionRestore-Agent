from __future__ import annotations

from visionrestore.schemas.candidate import CandidateExecutionPlan, CandidatePlanItem


class CandidatePlanner:
    def plan(self, *, image_id: str, intent, analysis, hardware: dict, model_statuses: list[dict], mode: str, semantic_analysis=None) -> CandidateExecutionPlan:
        priority = getattr(intent, "priority", "balanced") or "balanced"
        max_candidates = self._budget(priority, mode)
        status = {m["model_id"]: m for m in model_statuses}
        available = {m["model_id"] for m in model_statuses if m.get("available")}
        rejected: list[dict] = []
        items: list[CandidatePlanItem] = []

        manual_model = getattr(intent, "manual_model", None)
        manual_weight = getattr(intent, "manual_weight", None)
        if manual_model:
            self._try_add(items, rejected, status, available, manual_model, manual_weight, "用户明确手动选择模型/权重", max_candidates)
            return self._finish(image_id, mode, priority, max_candidates, items, rejected, ["手动选择优先，大模型建议不会覆盖。"])

        ordered_models = self._ordered_models(priority, intent, analysis, hardware, available)
        for model_id, reason in ordered_models:
            if len(items) >= max_candidates:
                break
            self._try_add(items, rejected, status, available, model_id, None, reason, max_candidates)

        if not items:
            for model_id in ["retinexformer", "flol", "sci", "zero_dce"]:
                if len(items) >= max_candidates:
                    break
                self._try_add(items, rejected, status, available, model_id, None, "兜底候选", max_candidates)

        return self._finish(image_id, mode, priority, max_candidates, items, rejected, ["所有增强候选均独立读取同一张原始输入图像。"])

    def _budget(self, priority: str, mode: str) -> int:
        if priority == "speed":
            return 1
        if priority in {"quality", "extreme_quality"} or mode == "compare":
            return 3
        return 2

    def _ordered_models(self, priority: str, intent, analysis, hardware: dict, available: set[str]) -> list[tuple[str, str]]:
        prefs = getattr(intent, "preferences", None)
        reduce_noise = bool(getattr(prefs, "reduce_noise", False))
        preserve_color = bool(getattr(prefs, "preserve_color", False) or getattr(prefs, "natural_result", False))
        noisy = getattr(analysis, "noise_estimate", 0) >= 12
        blurry = getattr(analysis, "laplacian_sharpness", 9999) < 80
        color_cast = getattr(analysis, "color_cast_index", 0) >= 0.12
        high_res = max(getattr(analysis, "width", 0), getattr(analysis, "height", 0)) >= 2500
        low_memory = bool(getattr(prefs, "low_memory", False)) or (hardware.get("gpu_memory_mb") or 0) < 4096

        if priority == "speed":
            return [("flol", "速度优先，优先快速真实低照度专家"), ("sci", "FLOL不可用时使用轻量兜底"), ("zero_dce", "最后轻量基线")]

        ordered = [("retinexformer", "普通真实低照度主候选")]
        if noisy or blurry or reduce_noise:
            ordered.append(("darkir", "噪声/模糊风险较高，加入联合恢复专家"))
        if color_cast or preserve_color:
            ordered.append(("hvi_cidnet", "颜色自然度或色偏风险较高，加入颜色恢复专家"))
        if high_res or low_memory:
            ordered.append(("flol", "高分辨率或显存压力下加入快速候选"))
        if priority in {"quality", "extreme_quality"}:
            for item in [("darkir", "质量模式保证架构多样性"), ("hvi_cidnet", "质量模式补充颜色专家")]:
                if item[0] not in {m for m, _ in ordered}:
                    ordered.append(item)
        ordered.append(("sci", "轻量兜底候选"))
        ordered.append(("zero_dce", "最后基线候选"))
        return ordered

    def _try_add(self, items: list[CandidatePlanItem], rejected: list[dict], status: dict, available: set[str], model_id: str, checkpoint_id: str | None, reason: str, max_candidates: int) -> None:
        if len(items) >= max_candidates:
            return
        if model_id not in status:
            rejected.append({"model_id": model_id, "reason": "模型未注册"})
            return
        if model_id not in available:
            rejected.append({"model_id": model_id, "reason": status[model_id].get("status_message", "模型不可用")})
            return
        if any(item.model_id == model_id and item.checkpoint_id == (checkpoint_id or item.checkpoint_id) for item in items):
            return
        checkpoint = checkpoint_id or self._select_checkpoint(model_id, status[model_id])
        if not checkpoint:
            rejected.append({"model_id": model_id, "reason": "没有可用权重"})
            return
        items.append(CandidatePlanItem(
            candidate_id=f"candidate_{len(items) + 1:02d}",
            model_id=model_id,
            checkpoint_id=checkpoint,
            reason=reason,
            estimated_cost={"max_candidates": max_candidates},
        ))

    def _select_checkpoint(self, model_id: str, model_status: dict) -> str:
        weights = [w for w in model_status.get("capabilities", {}).get("weights", []) if w.get("status") == "found" or w.get("exists")]
        if not weights:
            return ""
        defaults = [w for w in weights if w.get("default")]
        if defaults:
            return defaults[0]["checkpoint_id"]
        if model_id == "retinexformer":
            for checkpoint in ["lol_v2_real", "sdsd_outdoor", "sdsd_indoor", "ntire"]:
                if any(w.get("checkpoint_id") == checkpoint for w in weights):
                    return checkpoint
        if model_id == "sci":
            for checkpoint in ["medium", "easy", "difficult"]:
                if any(w.get("checkpoint_id") == checkpoint for w in weights):
                    return checkpoint
        return weights[0]["checkpoint_id"]

    def _finish(self, image_id: str, mode: str, priority: str, max_candidates: int, items: list[CandidatePlanItem], rejected: list[dict], notes: list[str]) -> CandidateExecutionPlan:
        return CandidateExecutionPlan(mode=mode, priority=priority, max_candidates=max_candidates, input_image_id=image_id, candidates=items, rejected=rejected, notes=notes)
