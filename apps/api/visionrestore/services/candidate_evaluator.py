from __future__ import annotations

from dataclasses import dataclass
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
    raw_metrics: dict[str, Any]


class CandidateEvaluator:
    def __init__(self):
        self.rules = get_scoring_rules()

    def score(self, candidate: dict, priority: str = "balanced") -> CandidateScore:
        metrics = candidate.get("metrics") or {}
        validity_reasons = self._validity_failures(candidate, metrics)
        if validity_reasons:
            return CandidateScore(
                candidate_id=candidate.get("candidate_id") or candidate.get("output_file_id") or "",
                model_id=candidate.get("model_id", ""),
                checkpoint_id=candidate.get("checkpoint_id", ""),
                valid=False,
                eliminated=True,
                score=0.0,
                layers={"hard_validity": 0.0},
                reasons=validity_reasons,
                raw_metrics=metrics,
            )
        layers = {
            "technical_quality": self._technical_quality(metrics),
            "no_reference_iqa": self._iqa_score(metrics),
            "user_match": self._user_match(metrics),
            "runtime_cost": self._runtime_cost(metrics),
        }
        weights = self.rules.get("weights", {}).get(priority, self.rules.get("weights", {}).get("balanced", {}))
        score = sum(layers[key] * float(weights.get(key, 0)) for key in layers) * 100.0
        return CandidateScore(
            candidate_id=candidate.get("candidate_id") or candidate.get("output_file_id") or "",
            model_id=candidate.get("model_id", ""),
            checkpoint_id=candidate.get("checkpoint_id", ""),
            valid=True,
            eliminated=False,
            score=round(float(score), 2),
            layers={key: round(float(value), 4) for key, value in layers.items()},
            reasons=["通过硬性有效性检查", "分数为候选集合内的推荐依据，不代表绝对图像质量"],
            raw_metrics=metrics,
        )

    def _validity_failures(self, candidate: dict, metrics: dict) -> list[str]:
        if candidate.get("status") not in {None, "completed"}:
            return [candidate.get("error") or "候选未成功完成"]
        rules = self.rules.get("hard_validity", {})
        failures = []
        if metrics.get("output_exists") is False:
            failures.append("输出不存在")
        if metrics.get("decode_ok") is False:
            failures.append("输出无法解码")
        if metrics.get("size_match") is False:
            failures.append("输出尺寸错误")
        if metrics.get("has_nan") is True or metrics.get("has_inf") is True:
            failures.append("输出包含NaN或Inf")
        if metrics.get("mean_luminance_after", 1) < rules.get("min_mean_luminance", 1):
            failures.append("输出接近纯黑")
        if metrics.get("overexposed_pixel_ratio_after", 0) > rules.get("max_overexposed_ratio", 0.45):
            failures.append("大面积过曝")
        if metrics.get("color_cast_index_after", 0) > rules.get("max_color_cast_index", 0.75):
            failures.append("严重色偏")
        if metrics.get("structure_keep_estimate", 1) < rules.get("min_structure_keep", 0.35):
            failures.append("结构保持不足")
        return failures

    def _technical_quality(self, metrics: dict) -> float:
        components = metrics.get("components") or {}
        if components:
            keys = ["shadow_recovery", "highlight_protection", "color_stability", "noise_control", "sharpness", "structure"]
            values = [float(components.get(key, 0)) for key in keys]
            return self._clamp(sum(values) / len(values))
        return self._clamp(float(metrics.get("score", 0)) / 100.0)

    def _iqa_score(self, metrics: dict) -> float:
        iqa = metrics.get("iqa") or {}
        normalized = [float(v) for v in iqa.values() if isinstance(v, int | float)]
        if not normalized:
            return 0.5
        return self._clamp(sum(normalized) / len(normalized))

    def _user_match(self, metrics: dict) -> float:
        if "user_match" in metrics:
            return self._clamp(float(metrics["user_match"]))
        components = metrics.get("components") or {}
        natural = min(float(components.get("color_stability", 0.5)), float(components.get("highlight_protection", 0.5)))
        recovery = float(components.get("shadow_recovery", 0.5))
        return self._clamp(natural * 0.45 + recovery * 0.55)

    def _runtime_cost(self, metrics: dict) -> float:
        rules = self.rules.get("runtime_cost", {})
        runtime = float(metrics.get("runtime_ms", 0))
        memory = float(metrics.get("peak_memory_mb", 0))
        runtime_score = 1.0 - runtime / float(rules.get("max_runtime_ms", 30000))
        memory_score = 1.0 - memory / float(rules.get("max_peak_memory_mb", 12000))
        return self._clamp(runtime_score * 0.6 + memory_score * 0.4)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))


class CandidateRanker:
    def rank(self, candidates: list[dict], priority: str = "balanced") -> dict:
        evaluator = CandidateEvaluator()
        scores = [evaluator.score(candidate, priority) for candidate in candidates]
        successful = sorted([score for score in scores if not score.eliminated], key=lambda item: item.score, reverse=True)
        eliminated = [score for score in scores if score.eliminated]
        close = False
        if len(successful) >= 2:
            close = abs(successful[0].score - successful[1].score) < 3.0
        return {
            "best": successful[0] if successful else None,
            "second_best": successful[1] if len(successful) > 1 else None,
            "ranked": successful,
            "eliminated": eliminated,
            "close_competition": close,
            "note": "两个候选结果接近，建议人工对比。" if close else "",
        }
