import numpy as np
from PIL import Image
from visionrestore.services.image_analyzer import ImageAnalyzer

class QualityEvaluator:
    def evaluate(self, input_path: str, output_path: str, priority: str = "balanced") -> dict:
        before = ImageAnalyzer().analyze(input_path)
        after = ImageAnalyzer().analyze(output_path)
        exposure_gain = max(0.0, min(1.0, (after.mean_luminance - before.mean_luminance) / 90.0))
        overexposure_penalty = min(1.0, after.overexposed_pixel_ratio * 4)
        noise_penalty = min(1.0, max(0.0, (after.noise_estimate - before.noise_estimate) / 35.0))
        color_penalty = min(1.0, after.color_cast_index * 2.5)
        contrast_bonus = max(0.0, min(1.0, after.rms_contrast / 90.0))
        weights = {
            "quality": (0.35, 0.25, 0.20, 0.15, 0.05),
            "balanced": (0.34, 0.22, 0.18, 0.16, 0.10),
            "speed": (0.28, 0.18, 0.16, 0.12, 0.26),
        }.get(priority, (0.34, 0.22, 0.18, 0.16, 0.10))
        base = (
            weights[0] * exposure_gain
            + weights[1] * contrast_bonus
            + weights[2] * (1 - overexposure_penalty)
            + weights[3] * (1 - noise_penalty)
            + weights[4] * (1 - color_penalty)
        )
        return {
            "score": round(float(base * 100), 2),
            "mean_luminance_before": round(before.mean_luminance, 3),
            "mean_luminance_after": round(after.mean_luminance, 3),
            "overexposed_pixel_ratio": after.overexposed_pixel_ratio,
            "noise_estimate_before": before.noise_estimate,
            "noise_estimate_after": after.noise_estimate,
            "color_cast_index": after.color_cast_index,
            "note": "无参考指标不能完全替代人工主观判断。",
        }
