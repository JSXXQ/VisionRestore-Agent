from visionrestore.schemas.task import EnhancementPlan

class DeterministicPlanner:
    def plan(self, *, image_id: str, user_goal: str, mode: str, priority: str, analysis, hardware: dict, models: list[dict], model_id: str | None = None, requested_models: list[str] | None = None, parameters: dict | None = None) -> EnhancementPlan:
        text = (user_goal or "").lower()
        installed = {m["model_id"]: m for m in models if m["available"]}
        all_ids = [m["model_id"] for m in models]
        preferred = "zero_dce"
        reason = "默认使用轻量模型作为首选。"
        if model_id:
            preferred = model_id
            reason = f"用户手动指定模型 {model_id}。"
        elif "retinexformer" in text or priority == "quality":
            preferred = "retinexformer"
            reason = "质量优先或用户提到 Retinexformer。"
        elif "snr" in text or (analysis.dark_pixel_ratio > 0.55 and analysis.noise_estimate > 14):
            preferred = "snr_aware"
            reason = "图像极暗且噪声估计较高，优先考虑 SNR-Aware。"
        elif priority == "speed" or "快" in text or "速度" in text or "显存有限" in text:
            preferred = "sci" if "sci" in all_ids else "zero_dce"
            reason = "速度优先或显存有限，优先轻量模型。"
        elif "zero" in text:
            preferred = "zero_dce"
            reason = "用户提到 Zero-DCE。"
        if preferred not in installed:
            fallback = [m for m in ["zero_dce", "sci", "retinexformer", "snr_aware"] if m in installed and m != preferred]
            reason += f" 但 {preferred} 当前不可用，将尝试可用备用模型。" if fallback else f" 但 {preferred} 当前不可用，且没有可用真实模型。"
        else:
            fallback = [m for m in ["zero_dce", "sci", "retinexformer", "snr_aware"] if m in installed and m != preferred]
        params = {"tile_size": 512, "tile_overlap": 32} | (parameters or {})
        if analysis.suggest_tile_inference:
            params["tile_enabled"] = True
        if hardware.get("cuda_available") is False:
            params["device"] = "cpu"
        return EnhancementPlan(
            input_id=image_id,
            user_goal=user_goal or "",
            mode=mode,
            priority=priority,
            selected_model=preferred,
            selection_reason=reason,
            parameters=params,
            preprocessing_steps=["RGB decode", "statistical degradation analysis"],
            postprocessing_steps=["quality evaluation", "report export"],
            fallback_models=fallback,
            expected_memory_mb=analysis.estimated_memory_mb,
            expected_runtime_ms=1200 if priority == "speed" else 4500,
            expected_risks=["无参考指标仅供参考", "未安装权重时模型会明确失败并触发回退"],
        )
