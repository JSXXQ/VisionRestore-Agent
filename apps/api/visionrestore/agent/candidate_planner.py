from __future__ import annotations

from visionrestore.core.model_config import get_model_config, get_planning_rules
from visionrestore.agent.checkpoint_selector import CheckpointSelection, CheckpointSelector
from visionrestore.schemas.candidate import CandidateExecutionPlan, CandidatePlanItem, PlanningScoreEvidence


class CandidatePlanner:
    """Build an explainable execution plan without judging real output quality."""

    def __init__(self):
        self.rules = get_planning_rules()
        self.eligible_models = tuple(self.rules.get("eligible_models") or ())
        self.model_priors = self.rules.get("model_priors") or {}
        self.fusion = self.rules.get("score_fusion") or {}
        self.checkpoint_selector = CheckpointSelector(self.rules)

    def plan(
        self,
        *,
        image_id: str,
        intent,
        analysis,
        hardware: dict,
        model_statuses: list[dict],
        mode: str,
        semantic_analysis=None,
        retrieved_context: list[dict] | None = None,
    ) -> CandidateExecutionPlan:
        priority = getattr(intent, "priority", "balanced") or "balanced"
        max_candidates = self._budget(priority, mode)
        status = {item["model_id"]: item for item in model_statuses}
        available = {
            item["model_id"]
            for item in model_statuses
            if item.get("available") and item.get("model_id") in self.eligible_models
        }
        hardware_blocked = {
            item.get("model_id"): reason
            for item in model_statuses
            if (reason := self._hardware_gate_reason(item, hardware))
        }
        available -= set(hardware_blocked)
        rejected: list[dict] = []
        scored: list[CandidatePlanItem] = []

        manual_model = getattr(intent, "manual_model", None)
        manual_weight = getattr(intent, "manual_weight", None)
        if manual_model:
            checkpoint_selection = self.checkpoint_selector.select(
                model_id=manual_model,
                model_status=status.get(manual_model, {}),
                intent=intent,
                analysis=analysis,
                hardware=hardware,
                semantic_analysis=semantic_analysis,
                manual_checkpoint=manual_weight,
            )
            evidence = [
                PlanningScoreEvidence(
                    source="manual_selection",
                    signal="explicit_model_choice",
                    contribution=100,
                    reason="用户明确指定模型；该分数只用于建立单候选执行计划。",
                )
            ]
            self._try_add(
                scored,
                rejected,
                status,
                available | {manual_model},
                manual_model,
                checkpoint_selection,
                ["用户明确手动选择模型/权重"],
                model_prior_score=100,
                input_match_score=0,
                planning_evidence=evidence,
                llm_score=None,
            )
            return self._finish(
                image_id,
                mode,
                priority,
                1,
                scored[:1],
                rejected,
                ["手动选择优先，知识与多模态建议不会覆盖。"],
                task_mode="single_candidate" if mode == "manual" else "multi_candidate",
            )

        llm_scores = self._llm_scores(semantic_analysis)
        for model_id in self.eligible_models:
            if model_id in hardware_blocked:
                rejected.append({"model_id": model_id, "reason": hardware_blocked[model_id]})
                continue
            prior_score, prior_evidence, reasons = self._model_prior_score(model_id, priority)
            input_score, input_evidence = self._input_match_score(
                model_id=model_id,
                intent=intent,
                analysis=analysis,
            )
            checkpoint_selection = self.checkpoint_selector.select(
                model_id=model_id,
                model_status=status.get(model_id, {}),
                intent=intent,
                analysis=analysis,
                hardware=hardware,
                semantic_analysis=semantic_analysis,
            )
            self._try_add(
                scored,
                rejected,
                status,
                available,
                model_id,
                checkpoint_selection,
                reasons + [item.reason for item in input_evidence],
                model_prior_score=prior_score,
                input_match_score=input_score,
                planning_evidence=prior_evidence + input_evidence,
                llm_score=llm_scores.get(model_id),
            )

        scored.sort(key=lambda item: item.planning_score, reverse=True)
        items = scored[:max_candidates]
        for idx, item in enumerate(items, start=1):
            item.candidate_id = f"candidate_{idx:02d}"
        return self._finish(
            image_id,
            mode,
            priority,
            max_candidates,
            items,
            rejected,
            [
                "所有增强候选均独立读取同一张原始输入图像。",
                "外部语义建议有效时，planning_score = local_score × 50% + llm_score × 50%。",
                "外部语义建议不可用时，planning_score = local_score；Knowledge 只作为 LLM 参照标准。",
                "planning_score 仅用于执行前候选规划，不参与 final_score。",
                "checkpoint_score 只在已选模型家族内部选择权重，不改变 planning_score 或 final_score。",
            ],
        )

    def _budget(self, priority: str, mode: str) -> int:
        if priority == "speed":
            return 1
        if priority in {"quality", "extreme_quality"} or mode == "compare":
            return 3
        return 2

    def _model_prior_score(
        self, model_id: str, priority: str
    ) -> tuple[float, list[PlanningScoreEvidence], list[str]]:
        config = self.model_priors.get(model_id) or {}
        base = float(config.get("score", 0))
        priority_adjustment = float((config.get("priority_adjustments") or {}).get(priority, 0))
        score = round(base + priority_adjustment, 2)
        description = str(config.get("evidence") or f"{model_id} 配置化模型先验")
        reason = f"模型先验 {score:.2f}：{description}"
        evidence = [
            PlanningScoreEvidence(
                source="model_prior_config",
                signal="model_capability_prior",
                observed={"base": base, "priority": priority, "priority_adjustment": priority_adjustment},
                contribution=score,
                reason=reason,
            )
        ]
        return score, evidence, [reason]

    def _input_match_score(
        self,
        *,
        model_id: str,
        intent,
        analysis,
    ) -> tuple[float, list[PlanningScoreEvidence]]:
        config = self.rules.get("input_match") or {}
        evidence: list[PlanningScoreEvidence] = []
        for signal, signal_config in (config.get("signals") or {}).items():
            max_points = float((signal_config.get("model_points") or {}).get(model_id, 0))
            if max_points == 0:
                continue
            severity, observed = self._signal_severity(analysis, signal_config)
            contribution = round(severity * max_points, 2)
            if contribution <= 0:
                continue
            label = str(signal_config.get("label") or signal)
            evidence.append(
                PlanningScoreEvidence(
                    source="image_analyzer",
                    signal=signal,
                    observed=observed,
                    severity=round(severity, 4),
                    contribution=contribution,
                    reason=f"{label}={severity:.2f}，与 {model_id} 的配置化能力匹配 +{contribution:.2f}。",
                )
            )

        preferences = getattr(intent, "preferences", None)
        for preference, adjustments in (config.get("intent_preferences") or {}).items():
            if not bool(getattr(preferences, preference, False)):
                continue
            contribution = float((adjustments or {}).get(model_id, 0))
            if contribution:
                evidence.append(
                    PlanningScoreEvidence(
                        source="user_intent",
                        signal=preference,
                        observed={"enabled": True},
                        contribution=contribution,
                        reason=f"用户约束 {preference} 与 {model_id} 能力匹配 {contribution:+.2f}。",
                    )
                )

        bounded = self._bound_positive_evidence(evidence, float(config.get("max_score", 48)))
        score = round(sum(item.contribution for item in bounded), 2)
        return score, bounded

    def _signal_severity(self, analysis, signal_config: dict) -> tuple[float, dict]:
        weighted = 0.0
        total_weight = 0.0
        observed: dict[str, float] = {}
        for metric in signal_config.get("metrics") or []:
            name = str(metric.get("name") or "")
            value = self._analysis_value(analysis, name)
            if value is None:
                continue
            observed[name] = round(value, 4)
            healthy = float(metric.get("healthy", 0))
            severe = float(metric.get("severe", 1))
            direction = str(metric.get("direction") or "higher")
            if direction == "lower":
                denominator = healthy - severe
                normalized = (healthy - value) / denominator if denominator else 0
            else:
                denominator = severe - healthy
                normalized = (value - healthy) / denominator if denominator else 0
            weight = max(0.0, float(metric.get("weight", 1)))
            weighted += self._clamp(normalized) * weight
            total_weight += weight
        if total_weight <= 0:
            return 0.0, observed
        return self._clamp(weighted / total_weight), observed

    def _analysis_value(self, analysis, name: str) -> float | None:
        value = getattr(analysis, name, None)
        if value is None and name == "total_pixels":
            width = getattr(analysis, "width", 0) or 0
            height = getattr(analysis, "height", 0) or 0
            value = width * height
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _llm_scores(self, semantic_analysis) -> dict[str, float]:
        if not semantic_analysis or not getattr(semantic_analysis, "adopted", False):
            return {}
        scores: dict[str, float] = {}
        for item in getattr(semantic_analysis, "model_candidates", []) or []:
            model_id = getattr(item, "model_id", "")
            if model_id not in self.eligible_models:
                continue
            score = self._clamp_score(float(getattr(item, "score", 0) or 0), 0.0, 100.0)
            scores[model_id] = max(scores.get(model_id, 0), round(score, 2))
        return scores

    @staticmethod
    def _bound_positive_evidence(
        evidence: list[PlanningScoreEvidence], maximum: float
    ) -> list[PlanningScoreEvidence]:
        bounded: list[PlanningScoreEvidence] = []
        running = 0.0
        for item in evidence:
            accepted = min(maximum - running, max(0.0, float(item.contribution)))
            if accepted <= 0:
                continue
            bounded.append(item.model_copy(update={"contribution": round(accepted, 2)}))
            running += accepted
            if running >= maximum:
                break
        return bounded

    def _try_add(
        self,
        items: list[CandidatePlanItem],
        rejected: list[dict],
        status: dict,
        available: set[str],
        model_id: str,
        checkpoint_selection: CheckpointSelection,
        reasons: list[str],
        *,
        model_prior_score: float,
        input_match_score: float,
        planning_evidence: list[PlanningScoreEvidence],
        llm_score: float | None,
    ) -> None:
        if model_id not in status:
            rejected.append({"model_id": model_id, "reason": "模型未注册"})
            return
        if model_id not in available:
            rejected.append(
                {"model_id": model_id, "reason": status[model_id].get("status_message", "模型不可用")}
            )
            return
        if not self._model_auto_route_enabled(model_id):
            rejected.append(
                {"model_id": model_id, "reason": "模型标记为 manual_only/experimental，不参与 V2 自动增强候选"}
            )
            return
        checkpoint_id = checkpoint_selection.checkpoint_id
        if not checkpoint_id:
            rejected.append({"model_id": model_id, "reason": checkpoint_selection.reason})
            return
        if any(item.model_id == model_id for item in items):
            return
        local_score = round(
            self._clamp_score(
                float(model_prior_score + input_match_score),
                float(self.fusion.get("local_min", 0)),
                float(self.fusion.get("local_max", 100)),
            ),
            2,
        )
        if llm_score is None:
            local_weight = 1.0
            llm_weight = 0.0
            planning_score = local_score
            planning_mode = "local_fallback"
        else:
            configured_local = max(0.0, float(self.fusion.get("local_weight", 0.5)))
            configured_llm = max(0.0, float(self.fusion.get("llm_weight", 0.5)))
            total_weight = configured_local + configured_llm
            local_weight = configured_local / total_weight if total_weight else 0.5
            llm_weight = configured_llm / total_weight if total_weight else 0.5
            planning_score = round(local_score * local_weight + llm_score * llm_weight, 2)
            planning_mode = "local_llm_fusion"
            planning_evidence = [
                *planning_evidence,
                PlanningScoreEvidence(
                    source="semantic_advisory",
                    signal="llm_model_fit_score",
                    observed={"llm_score": round(llm_score, 2), "weight": round(llm_weight, 2)},
                    contribution=round(llm_score * llm_weight, 2),
                    reason=f"大模型参照本地 Knowledge 给出 {llm_score:.2f}/100，按 {llm_weight:.0%} 融合。",
                ),
            ]
            reasons = [
                *reasons,
                f"LLMScore {llm_score:.2f}/100 与 LocalScore {local_score:.2f}/100 各占 50%。",
            ]
        model_config = self.model_priors.get(model_id) or {}
        checkpoint_evidence = [
            PlanningScoreEvidence.model_validate(item)
            for item in checkpoint_selection.evidence
        ]
        items.append(
            CandidatePlanItem(
                candidate_id="candidate_pending",
                model_id=model_id,
                checkpoint_id=checkpoint_id,
                checkpoint_score=round(checkpoint_selection.score, 2),
                checkpoint_selection_mode=checkpoint_selection.mode,
                checkpoint_evidence=checkpoint_evidence,
                checkpoint_candidates=checkpoint_selection.ranked_candidates,
                role=str(model_config.get("role") or "enhancement"),
                model_prior_score=round(float(model_prior_score), 2),
                input_match_score=round(float(input_match_score), 2),
                planning_evidence=planning_evidence,
                local_score=local_score,
                llm_score=round(float(llm_score), 2) if llm_score is not None else None,
                local_weight=round(local_weight, 2),
                llm_weight=round(llm_weight, 2),
                planning_mode=planning_mode,
                ai_semantic_bonus=round(float(llm_score or 0), 2),
                knowledge_adjustment=0,
                hardware_adjustment=0,
                planning_score=planning_score,
                reason=list(dict.fromkeys(
                    reason
                    for reason in [*reasons, checkpoint_selection.reason]
                    if reason
                )),
                estimated_cost={"planning_score_only": True},
            )
        )

    def _model_auto_route_enabled(self, model_id: str) -> bool:
        config = (get_model_config().get("models", {}) or {}).get(model_id, {})
        return bool(config.get("auto_route", True))

    @staticmethod
    def _hardware_gate_reason(model_status: dict, hardware: dict) -> str:
        if hardware.get("cuda_available") is not False:
            return ""
        supported = set(model_status.get("supported_devices") or [])
        if supported and "cpu" not in supported:
            return "当前未检测到 CUDA，且模型不支持 CPU 自动执行"
        return ""

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _clamp_score(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    @staticmethod
    def _finish(
        image_id: str,
        mode: str,
        priority: str,
        max_candidates: int,
        items: list[CandidatePlanItem],
        rejected: list[dict],
        notes: list[str],
        task_mode: str = "multi_candidate",
    ) -> CandidateExecutionPlan:
        return CandidateExecutionPlan(
            task_mode=task_mode,
            mode=mode,
            priority=priority,
            max_candidates=max_candidates,
            input_image_id=image_id,
            candidates=items,
            rejected=rejected,
            notes=notes,
        )
