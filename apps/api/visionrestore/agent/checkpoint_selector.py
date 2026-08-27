from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from visionrestore.core.model_config import get_planning_rules


@dataclass
class CheckpointSelection:
    checkpoint_id: str = ""
    score: float = 0
    mode: str = "unavailable"
    evidence: list[dict[str, Any]] = field(default_factory=list)
    ranked_candidates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if not self.checkpoint_id:
            if self.evidence and self.evidence[0].get("reason"):
                return str(self.evidence[0]["reason"])
            return "没有满足本地可用性与硬件约束的 checkpoint。"
        if self.mode == "manual_override":
            return f"用户明确指定 checkpoint {self.checkpoint_id}。"
        return f"家族内部权重匹配选择 {self.checkpoint_id}，checkpoint_score={self.score:.2f}。"


class CheckpointSelector:
    """Select one healthy checkpoint inside an already selected model family.

    The checkpoint score is intentionally family-local. It never changes the
    model-family PlanningScore and never participates in FinalScore.
    """

    def __init__(self, rules: dict | None = None):
        self.rules = rules or get_planning_rules()
        self.config = self.rules.get("checkpoint_selection") or {}
        self.signal_definitions = (self.rules.get("input_match") or {}).get("signals") or {}

    def select(
        self,
        *,
        model_id: str,
        model_status: dict,
        intent,
        analysis,
        hardware: dict,
        semantic_analysis=None,
        manual_checkpoint: str | None = None,
        allow_non_auto_route: bool = False,
        context: dict[str, Any] | None = None,
    ) -> CheckpointSelection:
        context = context or {}
        discovered = [
            item
            for item in model_status.get("capabilities", {}).get("weights", [])
            if item.get("status") == "found" or item.get("exists")
        ]
        if not discovered:
            return CheckpointSelection()

        if manual_checkpoint:
            selected = next(
                (item for item in discovered if item.get("checkpoint_id") == manual_checkpoint),
                None,
            )
            if selected is None:
                return CheckpointSelection(
                    mode="manual_invalid",
                    evidence=[{
                        "source": "manual_selection",
                        "signal": "checkpoint_not_available",
                        "observed": {"checkpoint_id": manual_checkpoint},
                        "contribution": 0,
                        "reason": f"用户指定的 checkpoint {manual_checkpoint} 不存在或不可用。",
                    }],
                )
            context_reason = self._context_gate_reason(
                (((self.config.get("models") or {}).get(model_id) or {}).get("checkpoints") or {}).get(manual_checkpoint) or {},
                context,
            )
            if context_reason:
                return CheckpointSelection(
                    mode="manual_invalid",
                    evidence=[{
                        "source": "checkpoint_context_gate",
                        "signal": "context_requirement",
                        "observed": context,
                        "contribution": 0,
                        "reason": context_reason,
                    }],
                )
            evidence = [{
                "source": "manual_selection",
                "signal": "explicit_checkpoint_choice",
                "observed": {"checkpoint_id": manual_checkpoint},
                "contribution": 100,
                "reason": "用户明确指定模型家族内部权重；本地白名单和文件可用性校验已通过。",
            }]
            candidate = self._candidate_payload(selected, 100, evidence, selected=True)
            return CheckpointSelection(
                checkpoint_id=manual_checkpoint,
                score=100,
                mode="manual_override",
                evidence=evidence,
                ranked_candidates=[candidate],
            )

        available = [
            item
            for item in discovered
            if allow_non_auto_route or item.get("auto_route", True)
        ]
        if not available:
            return CheckpointSelection()

        model_rules = ((self.config.get("models") or {}).get(model_id) or {})
        checkpoint_rules = model_rules.get("checkpoints") or {}
        effective_scene, scene_source, scene_confidence = self._effective_scene(
            intent, semantic_analysis
        )
        ranked: list[dict[str, Any]] = []
        for weight in available:
            checkpoint_id = str(weight.get("checkpoint_id") or "")
            rule = checkpoint_rules.get(checkpoint_id) or {}
            context_reason = self._context_gate_reason(rule, context)
            if context_reason:
                ranked.append({
                    "checkpoint_id": checkpoint_id,
                    "display_name": weight.get("display_name") or checkpoint_id,
                    "score": 0,
                    "selected": False,
                    "eligible": False,
                    "reason": context_reason,
                    "evidence": [{
                        "source": "checkpoint_context_gate",
                        "signal": "context_requirement",
                        "observed": context,
                        "contribution": 0,
                        "reason": context_reason,
                    }],
                })
                continue
            hardware_reason = self._hardware_gate_reason(rule, hardware)
            if hardware_reason:
                ranked.append({
                    "checkpoint_id": checkpoint_id,
                    "display_name": weight.get("display_name") or checkpoint_id,
                    "score": 0,
                    "selected": False,
                    "eligible": False,
                    "reason": hardware_reason,
                    "evidence": [{
                        "source": "hardware_gate",
                        "signal": "gpu_memory_mb",
                        "observed": {"gpu_memory_mb": hardware.get("gpu_memory_mb")},
                        "contribution": 0,
                        "reason": hardware_reason,
                    }],
                })
                continue

            score, evidence = self._score_checkpoint(
                checkpoint_id=checkpoint_id,
                weight=weight,
                rule=rule,
                intent=intent,
                analysis=analysis,
                hardware=hardware,
                effective_scene=effective_scene,
                scene_source=scene_source,
                scene_confidence=scene_confidence,
            )
            ranked.append(self._candidate_payload(weight, score, evidence))

        eligible = [item for item in ranked if item.get("eligible", True)]
        if not eligible:
            return CheckpointSelection(mode="hardware_blocked", ranked_candidates=ranked)

        default_id = str(
            model_rules.get("default")
            or ((self.config.get("defaults") or {}).get(model_id))
            or ""
        )
        eligible.sort(
            key=lambda item: (
                float(item.get("score", 0)),
                item.get("checkpoint_id") == default_id,
                bool(item.get("default")),
            ),
            reverse=True,
        )
        selected = eligible[0]
        selected["selected"] = True
        selected_id = str(selected["checkpoint_id"])

        ordered = eligible + [item for item in ranked if not item.get("eligible", True)]
        return CheckpointSelection(
            checkpoint_id=selected_id,
            score=float(selected.get("score", 0)),
            mode="local_checkpoint_match",
            evidence=list(selected.get("evidence") or []),
            ranked_candidates=ordered,
        )

    def _score_checkpoint(
        self,
        *,
        checkpoint_id: str,
        weight: dict,
        rule: dict,
        intent,
        analysis,
        hardware: dict,
        effective_scene: str,
        scene_source: str,
        scene_confidence: float,
    ) -> tuple[float, list[dict[str, Any]]]:
        evidence: list[dict[str, Any]] = []
        base = float(rule.get("base_score", 1 if weight.get("default") else 0))
        evidence.append({
            "source": "checkpoint_config",
            "signal": "checkpoint_prior",
            "observed": {"checkpoint_id": checkpoint_id, "domain": weight.get("domain", "")},
            "contribution": base,
            "reason": str(rule.get("evidence") or f"{checkpoint_id} 配置化权重先验。"),
        })

        priority = str(getattr(intent, "priority", "balanced") or "balanced")
        self._append_adjustment(
            evidence,
            source="user_intent",
            signal=f"priority:{priority}",
            observed={"priority": priority},
            contribution=float((rule.get("priority_adjustments") or {}).get(priority, 0)),
            reason=f"任务优先级 {priority} 对 {checkpoint_id} 的内部匹配调整。",
        )

        scene_adjustment = float(
            (rule.get("scene_adjustments") or {}).get(effective_scene, 0)
        )
        self._append_adjustment(
            evidence,
            source=scene_source,
            signal="scene_match",
            observed={"scene": effective_scene, "confidence": scene_confidence},
            contribution=scene_adjustment,
            reason=f"场景 {effective_scene} 与 {checkpoint_id} 的适用域匹配。",
        )

        preferences = getattr(intent, "preferences", None)
        for preference, contribution in (rule.get("preference_adjustments") or {}).items():
            if bool(getattr(preferences, preference, False)):
                self._append_adjustment(
                    evidence,
                    source="user_intent",
                    signal=f"preference:{preference}",
                    observed={"enabled": True},
                    contribution=float(contribution),
                    reason=f"用户偏好 {preference} 与 {checkpoint_id} 匹配。",
                )

        raw_text = str(getattr(intent, "raw_text", "") or "").lower()
        keywords = [str(item).lower() for item in rule.get("keywords") or []]
        matched_keywords = [item for item in keywords if item and item in raw_text]
        if matched_keywords:
            self._append_adjustment(
                evidence,
                source="user_intent",
                signal="keyword_match",
                observed={"matched_keywords": matched_keywords},
                contribution=float(rule.get("keyword_bonus", 0)),
                reason=f"用户文本关键词与 {checkpoint_id} 的适用域匹配。",
            )

        for signal, points in (rule.get("signal_points") or {}).items():
            severity, observed = self._signal_severity(analysis, signal)
            contribution = round(severity * float(points), 2)
            self._append_adjustment(
                evidence,
                source="image_analyzer",
                signal=signal,
                observed={**observed, "severity": round(severity, 4)},
                contribution=contribution,
                reason=f"图像退化信号 {signal} 与 {checkpoint_id} 的内部能力匹配。",
            )

        gpu_memory = float(hardware.get("gpu_memory_mb") or 0)
        unknown_gpu_penalty = float(rule.get("unknown_gpu_penalty", 0))
        if gpu_memory <= 0 and unknown_gpu_penalty:
            self._append_adjustment(
                evidence,
                source="hardware_snapshot",
                signal="gpu_memory_unknown",
                observed={"gpu_memory_mb": hardware.get("gpu_memory_mb")},
                contribution=unknown_gpu_penalty,
                reason=f"显存未知，保守降低 {checkpoint_id} 的内部优先级。",
            )

        minimum = float(self.config.get("score_min", 0))
        maximum = float(self.config.get("score_max", 100))
        total = sum(float(item.get("contribution", 0)) for item in evidence)
        return round(max(minimum, min(maximum, total)), 2), evidence

    def _signal_severity(self, analysis, signal: str) -> tuple[float, dict[str, float]]:
        definition = self.signal_definitions.get(signal) or {}
        weighted = 0.0
        total_weight = 0.0
        observed: dict[str, float] = {}
        for metric in definition.get("metrics") or []:
            name = str(metric.get("name") or "")
            value = self._analysis_value(analysis, name)
            if value is None:
                continue
            observed[name] = round(value, 4)
            healthy = float(metric.get("healthy", 0))
            severe = float(metric.get("severe", 1))
            direction = str(metric.get("direction") or "higher")
            denominator = (healthy - severe) if direction == "lower" else (severe - healthy)
            normalized = (
                (healthy - value) / denominator
                if direction == "lower" and denominator
                else (value - healthy) / denominator
                if denominator
                else 0
            )
            weight = max(0.0, float(metric.get("weight", 1)))
            weighted += max(0.0, min(1.0, normalized)) * weight
            total_weight += weight
        return (
            max(0.0, min(1.0, weighted / total_weight)) if total_weight else 0.0,
            observed,
        )

    @staticmethod
    def _analysis_value(analysis, name: str) -> float | None:
        value = getattr(analysis, name, None)
        if value is None and name == "total_pixels":
            value = (getattr(analysis, "width", 0) or 0) * (getattr(analysis, "height", 0) or 0)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _effective_scene(self, intent, semantic_analysis) -> tuple[str, str, float]:
        scene = str(getattr(intent, "scene", "unknown") or "unknown")
        if scene != "unknown":
            return scene, "user_intent", 1.0
        minimum_confidence = float(self.config.get("semantic_scene_min_confidence", 0.65))
        if semantic_analysis and getattr(semantic_analysis, "adopted", False):
            semantic_scene = str(getattr(semantic_analysis, "scene", "unknown") or "unknown")
            confidence = float(getattr(semantic_analysis, "scene_confidence", 0) or 0)
            if semantic_scene != "unknown" and confidence >= minimum_confidence:
                return semantic_scene, "semantic_scene_advisory", confidence
        return "unknown", "local_scene_fallback", 0.0

    @staticmethod
    def _context_gate_reason(rule: dict, context: dict[str, Any]) -> str:
        for key, expected in (rule.get("context_requirements") or {}).items():
            allowed = expected if isinstance(expected, list) else [expected]
            observed = context.get(key)
            if observed not in allowed:
                return f"checkpoint 要求 {key} 属于 {allowed}，当前为 {observed!r}。"
        return ""

    @staticmethod
    def _hardware_gate_reason(rule: dict, hardware: dict) -> str:
        minimum = float(rule.get("min_gpu_memory_mb") or 0)
        available = float(hardware.get("gpu_memory_mb") or 0)
        if minimum and available and available < minimum:
            return f"checkpoint 需要至少 {minimum:.0f} MB 显存，当前约 {available:.0f} MB。"
        if rule.get("requires_cuda") and hardware.get("cuda_available") is False:
            return "checkpoint 配置要求 CUDA，但当前 CUDA 不可用。"
        return ""

    @staticmethod
    def _append_adjustment(
        evidence: list[dict[str, Any]],
        *,
        source: str,
        signal: str,
        observed: dict,
        contribution: float,
        reason: str,
    ) -> None:
        if contribution == 0:
            return
        evidence.append({
            "source": source,
            "signal": signal,
            "observed": observed,
            "contribution": round(contribution, 2),
            "reason": reason,
        })

    @staticmethod
    def _candidate_payload(
        weight: dict,
        score: float,
        evidence: list[dict[str, Any]],
        *,
        selected: bool = False,
    ) -> dict[str, Any]:
        reasons = [str(item.get("reason") or "") for item in evidence if item.get("reason")]
        return {
            "checkpoint_id": str(weight.get("checkpoint_id") or ""),
            "display_name": weight.get("display_name") or weight.get("checkpoint_id"),
            "domain": weight.get("domain", ""),
            "score": round(float(score), 2),
            "selected": selected,
            "eligible": True,
            "default": bool(weight.get("default")),
            "reason": "；".join(reasons),
            "evidence": evidence,
        }
