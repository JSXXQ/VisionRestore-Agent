from visionrestore.core.model_config import get_postprocess_rules
from visionrestore.schemas.postprocess import PostprocessRecommendation


class ResidualDegradationAnalyzer:
    def __init__(self):
        self.rules = get_postprocess_rules().get("residual_degradation", {})

    def analyze(self, *, original_analysis, enhanced_metrics: dict, best_candidate: dict, user_intent, hardware: dict) -> PostprocessRecommendation:
        denoise_reason: list[str] = []
        denoise_risk: list[str] = []
        sr_reason: list[str] = []
        sr_risk: list[str] = []

        denoise_rules = self.rules.get("denoise", {})
        model_id = (best_candidate or {}).get("model_id", "")
        noise_before = float(enhanced_metrics.get("noise_estimate_before", getattr(original_analysis, "noise_estimate", 0)))
        noise_after = float(enhanced_metrics.get("noise_estimate_after", noise_before))
        color_cast_after = float(enhanced_metrics.get("color_cast_index_after", 0))
        noise_increase = noise_after - noise_before
        threshold_scale = float(denoise_rules.get("darkir_extra_threshold", 1.35)) if model_id == "darkir" else 1.0
        noise_after_min = float(denoise_rules.get("noise_after_min", 12)) * threshold_scale
        noise_increase_min = float(denoise_rules.get("noise_increase_min", 4)) * threshold_scale

        if noise_after >= noise_after_min:
            denoise_reason.append("增强后噪声估计仍较高")
        if noise_increase >= noise_increase_min:
            denoise_reason.append("增强后噪声相对原图上升")
        if color_cast_after >= float(denoise_rules.get("color_noise_cast_min", 0.18)):
            denoise_reason.append("暗部彩色噪声或色偏风险较高")
        if model_id == "darkir":
            denoise_risk.append("DarkIR本身包含联合恢复，额外去噪阈值已提高")
        if denoise_reason:
            denoise_risk.append("可能轻微降低细纹理")

        denoise_confidence = min(1.0, 0.28 * len(denoise_reason) + max(0.0, noise_after - noise_after_min) / 40.0)

        sr_rules = self.rules.get("super_resolution", {})
        width = int(enhanced_metrics.get("width", getattr(original_analysis, "width", 0)))
        height = int(enhanced_metrics.get("height", getattr(original_analysis, "height", 0)))
        short_edge = min(width, height) if width and height else min(getattr(original_analysis, "width", 0), getattr(original_analysis, "height", 0))
        wants_sr = any(word in (getattr(user_intent, "raw_text", "") or "").lower() for word in ["sr", "super", "放大", "超分", "高清"])
        if short_edge and short_edge < int(sr_rules.get("short_edge_threshold", 900)):
            sr_reason.append("图像短边较低，可能缺乏可用像素尺寸")
        restoration_evidence = (
            ((enhanced_metrics.get("score_evidence") or {}).get("restoration") or {}).get("components")
            or {}
        )
        detail_recovery = restoration_evidence.get("detail_recovery")
        if (
            detail_recovery is not None
            and float(detail_recovery) < float(sr_rules.get("detail_recovery_max", 0.55))
            and short_edge
            and short_edge < int(sr_rules.get("detail_short_edge_max", 1400))
        ):
            sr_reason.append("统一重评显示细节恢复仍不足，且当前像素尺寸适合受控超分")
        if wants_sr:
            sr_reason.append("用户明确表达放大或超分需求")
        if short_edge >= int(sr_rules.get("high_resolution_short_edge", 1600)) and not wants_sr:
            sr_risk.append("输入已经是高分辨率，默认不建议超分")
            sr_reason = []
        if sr_reason:
            sr_risk.append("超分可能生成不可靠细节或边缘伪影")

        sr_confidence = min(1.0, 0.35 * len(sr_reason))
        return PostprocessRecommendation(
            denoise_recommended=denoise_confidence >= 0.45,
            denoise_confidence=round(denoise_confidence, 2),
            denoise_reason=denoise_reason,
            denoise_risk=denoise_risk,
            preferred_denoiser="nafnet",
            super_resolution_recommended=sr_confidence >= 0.35,
            sr_confidence=round(sr_confidence, 2),
            sr_reason=sr_reason,
            sr_risk=sr_risk,
            preferred_sr_model=str(sr_rules.get("preferred_model", "realesrgan")) if sr_reason else "none",
            preferred_scale=int(sr_rules.get("default_scale", 2)),
        )

