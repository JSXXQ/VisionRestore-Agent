from pathlib import Path
import numpy as np
from PIL import Image
from visionrestore.core.model_config import get_routing_rules, get_scoring_rules
from visionrestore.services.image_analyzer import ImageAnalyzer

class QualityEvaluator:
    def evaluate(
        self,
        input_path: str,
        output_path: str,
        priority: str = "balanced",
        runtime_ms: int = 0,
        peak_memory_mb: float = 0,
        expected_scale: int = 1,
    ) -> dict:
        before = ImageAnalyzer().analyze(input_path)
        after = ImageAnalyzer().analyze(output_path)
        with Image.open(input_path) as a, Image.open(output_path) as b:
            arr_a = np.asarray(a.convert("RGB")).astype(np.float32)
            arr_b = np.asarray(b.convert("RGB")).astype(np.float32)
            expected_size = (a.width * expected_scale, a.height * expected_scale)
            if b.size != expected_size:
                raise RuntimeError(f"输出尺寸异常: {b.size} != {expected_size}")
            comparison = b.convert("RGB")
            if expected_scale != 1:
                comparison = comparison.resize(a.size, Image.Resampling.LANCZOS)
            arr_b_compare = np.asarray(comparison).astype(np.float32)
        diff = np.mean(np.abs(arr_a - arr_b_compare)) / 255.0
        structure_keep = float(max(0.0, min(1.0, 1.0 - diff * 0.75)))
        restoration_rules = get_scoring_rules().get("restoration", {}) or {}
        target = restoration_rules.get("brightness_target", {}) or {}
        brightness_fit = self._target_brightness_fit(
            after.mean_luminance,
            minimum=float(target.get("minimum", 55)),
            ideal=float(target.get("ideal", 95)),
            maximum=float(target.get("maximum", 135)),
            ceiling=float(target.get("overbright_ceiling", 185)),
        )
        brightness_gain = max(
            0.0,
            min(
                1.0,
                (before.dark_pixel_ratio - after.dark_pixel_ratio) * 1.4
                + max(0.0, after.mean_luminance - before.mean_luminance) / 140.0,
            ),
        )
        fit_weight = max(0.0, float(restoration_rules.get("brightness_fit_weight", 0.60)))
        gain_weight = max(0.0, float(restoration_rules.get("brightness_gain_weight", 0.40)))
        brightness_weight = fit_weight + gain_weight
        shadow_recovery = (
            (brightness_fit * fit_weight + brightness_gain * gain_weight) / brightness_weight
            if brightness_weight
            else brightness_fit
        )
        highlight_protection = max(0.0, min(1.0, 1.0 - max(0.0, after.overexposed_pixel_ratio - before.overexposed_pixel_ratio) * 10.0))
        noise_increase_score = 1.0 - max(0.0, after.noise_estimate - before.noise_estimate) / 24.0
        residual_noise_score = 1.0 - max(0.0, after.noise_estimate - 3.0) / 12.0
        raw_noise_control = max(0.0, min(1.0, noise_increase_score * 0.45 + residual_noise_score * 0.55))
        color_stability = max(0.0, min(1.0, 1.0 - max(0.0, after.color_cast_index - before.color_cast_index) * 3.0))
        sharpness_ratio = after.laplacian_sharpness / (before.laplacian_sharpness + 1e-6)
        sharpness_keep = self._sharpness_fit(sharpness_ratio)
        noise_control = raw_noise_control * (0.70 + 0.30 * sharpness_keep)
        high_frequency_penalty = max(0.0, sharpness_ratio - 10.0) / 20.0
        entropy_penalty = max(0.0, after.image_entropy - before.image_entropy - 1.8) / 9.0
        residual_noise_penalty = max(0.0, after.noise_estimate - 3.0) / 12.0
        artifact_control = max(0.0, min(1.0, 1.0 - residual_noise_penalty * 0.55 - high_frequency_penalty * 0.25 - entropy_penalty * 0.20))
        runtime_score = max(0.0, min(1.0, 1.0 - runtime_ms / 30000.0))
        memory_score = max(0.0, min(1.0, 1.0 - peak_memory_mb / 12000.0))
        weights = get_routing_rules().get("quality_weights", {}).get(priority, get_routing_rules().get("quality_weights", {}).get("balanced", {}))
        components = {
            "shadow_recovery": shadow_recovery,
            "highlight_protection": highlight_protection,
            "color_stability": color_stability,
            "noise_control": noise_control,
            "artifact_control": artifact_control,
            "sharpness": sharpness_keep,
            "structure": structure_keep,
            "brightness_target_fit": brightness_fit,
            "brightness_recovery_gain": brightness_gain,
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
            "sharpness_ratio": sharpness_ratio,
            "color_cast_index_before": before.color_cast_index,
            "color_cast_index_after": after.color_cast_index,
            "image_entropy_before": before.image_entropy,
            "image_entropy_after": after.image_entropy,
            "structure_keep_estimate": structure_keep,
            "runtime_ms": runtime_ms,
            "peak_memory_mb": peak_memory_mb,
            "output_file_size_bytes": Path(output_path).stat().st_size if Path(output_path).exists() else 0,
            "evaluation_expected_scale": int(expected_scale),
            "note": "无参考指标只能作为辅助判断，不能完全替代人工主观评价；未提供GT时不计算PSNR、SSIM或LPIPS。",
        }

    @staticmethod
    def _target_brightness_fit(
        value: float,
        *,
        minimum: float,
        ideal: float,
        maximum: float,
        ceiling: float,
    ) -> float:
        if value < minimum:
            return max(0.0, min(0.85, value / max(minimum, 1e-6) * 0.85))
        if value <= ideal:
            return 0.85 + 0.15 * (value - minimum) / max(ideal - minimum, 1e-6)
        if value <= maximum:
            return 1.0 - 0.15 * (value - ideal) / max(maximum - ideal, 1e-6)
        return max(0.0, 0.85 * (1.0 - (value - maximum) / max(ceiling - maximum, 1e-6)))

    @staticmethod
    def _sharpness_fit(ratio: float) -> float:
        if ratio <= 1.0:
            return max(0.0, ratio * 0.70)
        if ratio <= 4.0:
            return 0.70 + (ratio - 1.0) * 0.10
        if ratio <= 10.0:
            return 1.0
        return max(0.0, 1.0 - (ratio - 10.0) / 20.0)
