from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from visionrestore.core.model_config import get_scoring_rules


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    model_id: str
    checkpoint_id: str
    valid: bool
    eliminated: bool
    score: float
    layers: dict[str, float]
    reasons: list[str]
    evidence: dict[str, Any]
    raw_metrics: dict[str, Any]


class CandidateEvaluator:
    """Evaluate real outputs in four auditable layers.

    Planning scores, retrieved knowledge, and hardware planning adjustments are
    deliberately excluded from this class.
    """

    def __init__(self):
        self.rules = get_scoring_rules()

    def score(self, candidate: dict, priority: str = "balanced") -> CandidateScore:
        metrics = candidate.get("metrics") or {}
        identity = {
            "candidate_id": candidate.get("candidate_id") or candidate.get("output_file_id") or "",
            "model_id": candidate.get("model_id", ""),
            "checkpoint_id": candidate.get("checkpoint_id", ""),
        }
        validity_reasons = self._validity_failures(candidate, metrics)
        if validity_reasons:
            return CandidateScore(
                **identity,
                valid=False,
                eliminated=True,
                score=0.0,
                layers={
                    "image_quality": 0.0,
                    "restoration": 0.0,
                    "constraint": 0.0,
                    "stability": 0.0,
                },
                reasons=validity_reasons,
                evidence={
                    "hard_validity": {"passed": False, "failures": validity_reasons},
                    "excluded_from_final_score": self._excluded_inputs(candidate, metrics),
                },
                raw_metrics=metrics,
            )

        image_quality, image_evidence = self._image_quality_score(metrics)
        restoration, restoration_evidence = self._restoration_score(metrics)
        constraint, constraint_evidence = self._constraint_score(metrics)
        stability, stability_evidence = self._stability_score(metrics)
        layers = {
            "image_quality": image_quality,
            "restoration": restoration,
            "constraint": constraint,
            "stability": stability,
        }
        priority_key = "quality" if priority == "extreme_quality" else priority
        weights = self.rules.get("weights", {}).get(
            priority_key,
            self.rules.get("weights", {}).get("balanced", {}),
        )
        score = sum(layers[key] * float(weights.get(key, 0)) for key in layers) * 100.0
        rounded_layers = {key: round(float(value), 4) for key, value in layers.items()}
        return CandidateScore(
            **identity,
            valid=True,
            eliminated=False,
            score=round(float(score), 2),
            layers=rounded_layers,
            reasons=[
                "通过硬性有效性检查",
                "final_score 仅由真实输出的图像质量、恢复效果、约束满足和稳定性组成",
            ],
            evidence={
                "formula": "image_quality + restoration + constraint + stability (weighted)",
                "weights": {key: float(weights.get(key, 0)) for key in layers},
                "image_quality": image_evidence,
                "restoration": restoration_evidence,
                "constraint": constraint_evidence,
                "stability": stability_evidence,
                "hard_validity": {"passed": True, "failures": []},
                "excluded_from_final_score": self._excluded_inputs(candidate, metrics),
            },
            raw_metrics=metrics,
        )

    def _validity_failures(self, candidate: dict, metrics: dict) -> list[str]:
        if candidate.get("status") not in {None, "completed"}:
            return [candidate.get("error") or "候选未成功完成"]
        rules = self.rules.get("hard_validity", {})
        failures: list[str] = []
        if metrics.get("output_exists") is False:
            failures.append("输出不存在")
        if metrics.get("decode_ok") is False:
            failures.append("输出无法解码")
        if metrics.get("size_match") is False:
            failures.append("输出尺寸错误")
        if metrics.get("has_nan") is True or metrics.get("has_inf") is True:
            failures.append("输出包含NaN或Inf")
        if float(metrics.get("mean_luminance_after", 1) or 0) < float(
            rules.get("min_mean_luminance", 1)
        ):
            failures.append("输出接近纯黑")
        if float(metrics.get("overexposed_pixel_ratio_after", 0) or 0) > float(
            rules.get("max_overexposed_ratio", 0.45)
        ):
            failures.append("大面积过曝")
        if float(metrics.get("color_cast_index_after", 0) or 0) > float(
            rules.get("max_color_cast_index", 0.75)
        ):
            failures.append("严重色偏")
        if float(metrics.get("structure_keep_estimate", 1) or 0) < float(
            rules.get("min_structure_keep", 0.35)
        ):
            failures.append("结构保持不足")
        region = metrics.get("region_constraints") or {}
        if region.get("hard_failed") and self._severe_region_failure(region):
            failures.extend(region.get("violations") or ["局部区域硬性约束未满足"])
        return failures

    def _image_quality_score(self, metrics: dict) -> tuple[float, dict]:
        config = self.rules.get("image_quality") or {}
        components = metrics.get("components") or {}
        color_naturalness, color_evidence = self._scene_color_naturalness(metrics)
        fallback_weights = config.get("fallback_component_weights") or {}
        derived_components = dict(components)
        derived_components.update(
            {
                "brightness_restoration": self._brightness_recovery(components),
                "scene_color_naturalness": color_naturalness,
                "effective_sharpness": self._effective_sharpness(metrics, components),
            }
        )
        fallback_values = {
            name: self._clamp(float(derived_components.get(name, 0.5)))
            for name in fallback_weights
        }
        fallback_score = self._weighted_average(fallback_values, fallback_weights)
        normalized = metrics.get("iqa_normalized") or {}
        iqa_values = {
            str(name): self._clamp(float(value))
            for name, value in normalized.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        if iqa_values:
            weights = config.get("iqa_metric_weights") or {}
            iqa_score = self._weighted_average(iqa_values, weights)
            expected = {"musiq", "clipiqa"}
            complete = expected <= set(iqa_values)
            blend_weight = float(
                config.get(
                    "iqa_blend_weight_complete" if complete else "iqa_blend_weight_partial",
                    0.70 if complete else 0.50,
                )
            )
            blend_weight = self._clamp(blend_weight)
            score = iqa_score * blend_weight + fallback_score * (1.0 - blend_weight)
            return score, {
                "score": round(score, 4),
                "source": "hybrid_no_reference_iqa",
                "backend": (metrics.get("iqa_detail") or {}).get("backend", "configured_iqa"),
                "components": iqa_values,
                "iqa_score": round(iqa_score, 4),
                "local_perceptual_score": round(fallback_score, 4),
                "iqa_blend_weight": round(blend_weight, 4),
                "metric_coverage": "complete" if complete else "partial",
                "configured_metrics": list(weights),
                "color_naturalness": color_evidence,
            }

        score = fallback_score
        return score, {
            "score": round(score, 4),
            "source": "local_perceptual_fallback",
            "components": fallback_values,
            "fallback_reason": (metrics.get("iqa_detail") or {}).get(
                "reason", "no normalized IQA metric available"
            ),
            "extensible_metrics": ["MUSIQ", "CLIP-IQA", "TOPIQ-NR", "NIQE", "BRISQUE"],
            "color_naturalness": color_evidence,
        }

    def _restoration_score(self, metrics: dict) -> tuple[float, dict]:
        components = metrics.get("components") or {}
        brightness = self._brightness_recovery(components)
        noise = self._clamp(float(components.get("noise_control", 0.5)))
        sharpness = self._effective_sharpness(metrics, components)
        structure = self._clamp(
            float(components.get("structure", metrics.get("structure_keep_estimate", 0.5)))
        )
        artifact = self._clamp(float(components.get("artifact_control", noise)))
        values = {
            "brightness_recovery": brightness,
            "residual_noise_control": noise,
            "edge_preservation": self._clamp(sharpness * 0.80 + structure * 0.20),
            "detail_recovery": self._clamp(sharpness * 0.50 + structure * 0.15 + artifact * 0.35),
        }
        weights = (self.rules.get("restoration") or {}).get("component_weights") or {}
        score = self._weighted_average(values, weights)
        return score, {
            "score": round(score, 4),
            "source": "real_output_restoration_metrics",
            "components": {key: round(value, 4) for key, value in values.items()},
            "observed": {
                "mean_luminance_before": metrics.get("mean_luminance_before"),
                "mean_luminance_after": metrics.get("mean_luminance_after"),
                "noise_estimate_before": metrics.get("noise_estimate_before"),
                "noise_estimate_after": metrics.get("noise_estimate_after"),
                "laplacian_sharpness_before": metrics.get("laplacian_sharpness_before"),
                "laplacian_sharpness_after": metrics.get("laplacian_sharpness_after"),
            },
        }

    def _brightness_recovery(self, components: dict) -> float:
        config = self.rules.get("restoration") or {}
        fit = self._clamp(
            float(components.get("brightness_target_fit", components.get("shadow_recovery", 0.5)))
        )
        gain = self._clamp(
            float(components.get("brightness_recovery_gain", components.get("shadow_recovery", 0.5)))
        )
        fit_weight = max(0.0, float(config.get("brightness_fit_weight", 0.70)))
        gain_weight = max(0.0, float(config.get("brightness_gain_weight", 0.30)))
        total = fit_weight + gain_weight
        return self._clamp(
            (fit * fit_weight + gain * gain_weight) / total if total else fit
        )

    def _scene_color_naturalness(self, metrics: dict) -> tuple[float, dict]:
        """Distinguish natural dominant scene colors from artificial color drift."""
        config = (self.rules.get("stability") or {}).get("color_naturalness") or {}
        after = max(0.0, float(metrics.get("color_cast_index_after", 0) or 0))
        before_value = metrics.get("color_cast_index_before")
        before = max(0.0, float(before_value or 0)) if before_value is not None else None
        components = metrics.get("components") or {}

        change_tolerance = max(0.0, float(config.get("change_tolerance", 0.08)))
        change_ceiling = max(
            change_tolerance + 1e-6,
            float(config.get("change_ceiling", 0.35)),
        )
        if before is None:
            change_score = self._clamp(float(components.get("color_stability", 0.5)))
            increase = None
        else:
            increase = max(0.0, after - before)
            change_score = 1.0 if increase <= change_tolerance else self._clamp(
                1.0 - (increase - change_tolerance) / (change_ceiling - change_tolerance)
            )

        absolute_tolerance = max(0.0, float(config.get("absolute_tolerance", 0.20)))
        absolute_ceiling = max(
            absolute_tolerance + 1e-6,
            float(config.get("absolute_ceiling", 0.60)),
        )
        absolute_score = 1.0 if after <= absolute_tolerance else self._clamp(
            1.0 - (after - absolute_tolerance) / (absolute_ceiling - absolute_tolerance)
        )
        change_weight = max(0.0, float(config.get("change_weight", 0.70)))
        absolute_weight = max(0.0, float(config.get("absolute_weight", 0.30)))
        total = change_weight + absolute_weight
        adaptive_score = self._clamp(
            (change_score * change_weight + absolute_score * absolute_weight) / total
            if total
            else change_score
        )
        normalized_iqa = metrics.get("iqa_normalized") or {}
        available_iqa = {
            str(name)
            for name, value in normalized_iqa.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        required_iqa = {
            str(name) for name in config.get("required_iqa_metrics", ["musiq", "clipiqa"])
        }
        minimum_iqa_value = self._clamp(float(config.get("minimum_iqa_value", 0.60)))
        iqa_supported = (
            bool(required_iqa)
            and required_iqa <= available_iqa
            and all(float(normalized_iqa[name]) >= minimum_iqa_value for name in required_iqa)
        )
        fallback_limit = max(
            1e-6,
            float(config.get("fallback_absolute_soft_limit", 0.25)),
        )
        conservative_fallback = self._clamp(1.0 - after / fallback_limit)
        score = adaptive_score if iqa_supported else conservative_fallback
        return score, {
            "score": round(score, 4),
            "source": "scene_adaptive_color_consistency"
            if iqa_supported
            else "conservative_absolute_color_fallback",
            "iqa_supported": iqa_supported,
            "required_iqa_metrics": sorted(required_iqa),
            "available_iqa_metrics": sorted(available_iqa),
            "minimum_iqa_value": minimum_iqa_value,
            "adaptive_score": round(adaptive_score, 4),
            "conservative_fallback_score": round(conservative_fallback, 4),
            "change_fidelity": round(change_score, 4),
            "absolute_cast_guard": round(absolute_score, 4),
            "observed_before": before,
            "observed_after": after,
            "observed_increase": increase,
            "change_tolerance": change_tolerance,
            "absolute_tolerance": absolute_tolerance,
        }

    def _effective_sharpness(self, metrics: dict, components: dict) -> float:
        ratio_value = metrics.get("sharpness_ratio")
        if ratio_value is None:
            return self._clamp(float(components.get("sharpness", 0.5)))
        ratio = max(0.0, float(ratio_value))
        config = (self.rules.get("restoration") or {}).get("detail_quality") or {}
        low_ratio = max(1e-6, float(config.get("low_ratio", 1.0)))
        full_start = max(low_ratio, float(config.get("full_score_start_ratio", 8.0)))
        full_end = max(full_start, float(config.get("full_score_end_ratio", 18.0)))
        ceiling = max(full_end + 1e-6, float(config.get("high_ratio_ceiling", 40.0)))
        score_at_one = self._clamp(float(config.get("low_ratio_score_at_one", 0.60)))
        if ratio <= low_ratio:
            return self._clamp(ratio / low_ratio * score_at_one)
        if ratio <= full_start:
            return self._clamp(
                score_at_one
                + (ratio - low_ratio)
                * (1.0 - score_at_one)
                / max(full_start - low_ratio, 1e-6)
            )
        if ratio <= full_end:
            return 1.0
        return self._clamp(1.0 - (ratio - full_end) / (ceiling - full_end))

    def _constraint_score(self, metrics: dict) -> tuple[float, dict]:
        region = metrics.get("region_constraints") or {}
        if not region.get("enabled"):
            score = self._clamp(float((self.rules.get("constraint") or {}).get("no_roi_score", 1.0)))
            return score, {
                "score": round(score, 4),
                "source": "global_validity_only",
                "roi_enabled": False,
                "hard_failed": False,
            }
        score = self._clamp(float(region.get("score", 0)))
        if region.get("hard_failed"):
            score = 0.0
        return score, {
            "score": round(score, 4),
            "source": "region_constraint_evaluator",
            "roi_enabled": True,
            "hard_failed": bool(region.get("hard_failed")),
            "violations": list(region.get("violations") or []),
            "items": list(region.get("items") or []),
        }

    def _stability_score(self, metrics: dict) -> tuple[float, dict]:
        config = self.rules.get("stability") or {}
        mean_luminance = float(metrics.get("mean_luminance_after", 0) or 0)
        overexposed = float(metrics.get("overexposed_pixel_ratio_after", 0) or 0)
        color_cast = float(metrics.get("color_cast_index_after", 0) or 0)
        color_naturalness, color_evidence = self._scene_color_naturalness(metrics)
        components = metrics.get("components") or {}
        black_safe = float(config.get("black_safe_luminance", 20))
        soft_over = float(config.get("soft_overexposed_ratio", 0.20))
        soft_brightness = float(config.get("soft_brightness_max", 135))
        brightness_ceiling = float(config.get("brightness_ceiling", 185))
        pixel_overexposure_safety = self._clamp(1.0 - overexposed / max(soft_over, 1e-6))
        brightness_safety = 1.0
        if mean_luminance > soft_brightness:
            brightness_safety = self._clamp(
                1.0
                - (mean_luminance - soft_brightness)
                / max(brightness_ceiling - soft_brightness, 1e-6)
            )
        values = {
            "black_output_safety": self._clamp(mean_luminance / max(black_safe, 1e-6)),
            "overexposure_safety": min(pixel_overexposure_safety, brightness_safety),
            "scene_color_naturalness": color_naturalness,
            "artifact_stability": self._clamp(float(components.get("artifact_control", 0.5))),
            "numeric_integrity": 0.0
            if metrics.get("has_nan") is True or metrics.get("has_inf") is True
            else 1.0,
        }
        weights = config.get("component_weights") or {}
        score = self._weighted_average(values, weights)
        anomalies = [name for name, value in values.items() if value < 0.35]
        return score, {
            "score": round(score, 4),
            "source": "output_anomaly_checks",
            "components": {key: round(value, 4) for key, value in values.items()},
            "detected_anomalies": anomalies,
            "color_naturalness": color_evidence,
            "observed": {
                "mean_luminance_after": mean_luminance,
                "overexposed_pixel_ratio_after": overexposed,
                "brightness_safety": round(brightness_safety, 4),
                "color_cast_index_after": color_cast,
                "has_nan": bool(metrics.get("has_nan")),
                "has_inf": bool(metrics.get("has_inf")),
            },
        }

    @staticmethod
    def _excluded_inputs(candidate: dict, metrics: dict) -> dict:
        return {
            "planning_score": candidate.get("planning_score", metrics.get("planning_score")),
            "knowledge_adjustment": metrics.get("knowledge_adjustment"),
            "hardware_adjustment": metrics.get("hardware_adjustment"),
            "runtime_ms": metrics.get("runtime_ms", candidate.get("runtime_ms")),
            "peak_memory_mb": metrics.get("peak_memory_mb", candidate.get("peak_memory_mb")),
            "reason": "这些字段保留用于规划、资源观测或报告，不参与 final_score。",
        }

    @staticmethod
    def _severe_region_failure(region: dict) -> bool:
        if bool(region.get("severe_failure")):
            return True
        for item in region.get("items") or []:
            if not isinstance(item, dict) or item.get("priority") != "hard":
                continue
            after_over = float(item.get("overexposed_ratio_after", 0) or 0)
            before_over = float(item.get("overexposed_ratio_before", 0) or 0)
            after_p95 = float(item.get("luminance_p95_after", 0) or 0)
            before_p95 = float(item.get("luminance_p95_before", 0) or 0)
            if after_over >= 0.35 and after_over > before_over + 0.12:
                return True
            if after_p95 >= 254.0 and after_p95 > before_p95 + 12.0:
                return True
        return False

    @classmethod
    def _weighted_average(cls, values: dict[str, float], weights: dict) -> float:
        if not values:
            return 0.0
        active_weights = {key: max(0.0, float(weights.get(key, 0))) for key in values}
        total_weight = sum(active_weights.values())
        if total_weight <= 0:
            return cls._clamp(sum(values.values()) / len(values))
        return cls._clamp(
            sum(cls._clamp(values[key]) * active_weights[key] for key in values) / total_weight
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))


class CandidateRanker:
    def rank(self, candidates: list[dict], priority: str = "balanced") -> dict:
        evaluator = CandidateEvaluator()
        scores = [evaluator.score(candidate, priority) for candidate in candidates]
        successful = sorted(
            [score for score in scores if not score.eliminated],
            key=lambda item: item.score,
            reverse=True,
        )
        eliminated = [score for score in scores if score.eliminated]
        close = len(successful) >= 2 and abs(successful[0].score - successful[1].score) < 3.0
        return {
            "best": successful[0] if successful else None,
            "second_best": successful[1] if len(successful) > 1 else None,
            "ranked": successful,
            "eliminated": eliminated,
            "close_competition": close,
            "note": "两个候选结果接近，建议人工对比。" if close else "",
        }
