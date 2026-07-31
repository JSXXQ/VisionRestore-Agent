from pathlib import Path
import numpy as np
from PIL import Image
from visionrestore.core.model_config import get_routing_rules
from visionrestore.services.image_analyzer import ImageAnalyzer

class QualityEvaluator:
    def evaluate(self, input_path: str, output_path: str, priority: str = "balanced", runtime_ms: int = 0, peak_memory_mb: float = 0) -> dict:
        before = ImageAnalyzer().analyze(input_path)
        after = ImageAnalyzer().analyze(output_path)
        with Image.open(input_path) as a, Image.open(output_path) as b:
            arr_a = np.asarray(a.convert("RGB")).astype(np.float32)
            arr_b = np.asarray(b.convert("RGB")).astype(np.float32)
        if arr_a.shape != arr_b.shape:
            raise RuntimeError(f"输出尺寸异常: {arr_b.shape} != {arr_a.shape}")
        diff = np.mean(np.abs(arr_a - arr_b)) / 255.0
        structure_keep = float(max(0.0, min(1.0, 1.0 - diff * 0.75)))
        shadow_recovery = max(0.0, min(1.0, (before.dark_pixel_ratio - after.dark_pixel_ratio) * 1.6 + (after.mean_luminance - before.mean_luminance) / 120.0))
        highlight_protection = max(0.0, min(1.0, 1.0 - max(0.0, after.overexposed_pixel_ratio - before.overexposed_pixel_ratio) * 10.0))
        noise_control = max(0.0, min(1.0, 1.0 - max(0.0, after.noise_estimate - before.noise_estimate) / 35.0))
        color_stability = max(0.0, min(1.0, 1.0 - max(0.0, after.color_cast_index - before.color_cast_index) * 3.0))
        sharpness_keep = max(0.0, min(1.0, after.laplacian_sharpness / (before.laplacian_sharpness + 1e-6)))
        runtime_score = max(0.0, min(1.0, 1.0 - runtime_ms / 30000.0))
        memory_score = max(0.0, min(1.0, 1.0 - peak_memory_mb / 12000.0))
        weights = get_routing_rules().get("quality_weights", {}).get(priority, get_routing_rules().get("quality_weights", {}).get("balanced", {}))
        components = {
            "shadow_recovery": shadow_recovery,
            "highlight_protection": highlight_protection,
            "color_stability": color_stability,
            "noise_control": noise_control,
            "sharpness": sharpness_keep,
            "structure": structure_keep,
            "runtime": runtime_score,
            "memory": memory_score,
        }
        score = sum(components[k] * float(weights.get(k, 0)) for k in components) * 100.0
        return {
            "score": round(float(score), 2),
            "components": {k: round(float(v), 4) for k, v in components.items()},
            "mean_luminance_before": before.mean_luminance,
            "mean_luminance_after": after.mean_luminance,
            "dark_pixel_ratio_before": before.dark_pixel_ratio,
            "dark_pixel_ratio_after": after.dark_pixel_ratio,
            "overexposed_pixel_ratio_before": before.overexposed_pixel_ratio,
            "overexposed_pixel_ratio_after": after.overexposed_pixel_ratio,
            "rms_contrast_before": before.rms_contrast,
            "rms_contrast_after": after.rms_contrast,
            "dynamic_range_before": before.dynamic_range,
            "dynamic_range_after": after.dynamic_range,
            "laplacian_sharpness_before": before.laplacian_sharpness,
            "laplacian_sharpness_after": after.laplacian_sharpness,
            "noise_estimate_before": before.noise_estimate,
            "noise_estimate_after": after.noise_estimate,
            "color_cast_index_before": before.color_cast_index,
            "color_cast_index_after": after.color_cast_index,
            "image_entropy_before": before.image_entropy,
            "image_entropy_after": after.image_entropy,
            "structure_keep_estimate": structure_keep,
            "runtime_ms": runtime_ms,
            "peak_memory_mb": peak_memory_mb,
            "output_file_size_bytes": Path(output_path).stat().st_size if Path(output_path).exists() else 0,
            "note": "无参考指标只能作为辅助判断，不能完全替代人工主观评价；未提供GT时不计算PSNR、SSIM或LPIPS。",
        }
