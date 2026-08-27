from dataclasses import dataclass
from typing import Any

from visionrestore.core.model_config import get_routing_rules
from visionrestore.schemas.ai import MultimodalAnalysisResult


@dataclass
class RouteResult:
    selected_model: str
    selected_checkpoint: str
    model_candidates: list[dict[str, Any]]
    checkpoint_candidates: list[dict[str, Any]]
    fallback: list[tuple[str, str]]
    reason: str


class HierarchicalRouter:
    ENHANCEMENT_MODELS = ["retinexformer", "darkir", "hvi_cidnet", "flol", "sci", "zero_dce"]

    def __init__(self):
        self.rules = get_routing_rules()

    def route(
        self,
        *,
        intent,
        analysis,
        hardware: dict,
        model_statuses: list[dict],
        mode: str,
        requested_models: list[str] | None = None,
        semantic_analysis: MultimodalAnalysisResult | None = None,
    ) -> RouteResult:
        status = {m["model_id"]: m for m in model_statuses}
        available = {m["model_id"] for m in model_statuses if m.get("available")}
        mem = hardware.get("gpu_memory_mb") or 0
        cuda = bool(hardware.get("cuda_available"))
        routing = self.rules.get("model_routing", {})
        scores = {model_id: 0.0 for model_id in self.ENHANCEMENT_MODELS}
        reasons = {model_id: [] for model_id in self.ENHANCEMENT_MODELS}

        manual_model = getattr(intent, "manual_model", None)
        manual_weight = getattr(intent, "manual_weight", None)
        if manual_model:
            scores.setdefault(manual_model, 0.0)
            reasons.setdefault(manual_model, [])
            scores[manual_model] += 100
            reasons[manual_model].append("用户明确指定模型")

        noisy = getattr(analysis, "noise_estimate", 0) >= routing.get("darkir_noise_threshold", 12)
        blurry = getattr(analysis, "laplacian_sharpness", 9999) < routing.get("darkir_blur_sharpness_threshold", 80)
        color_cast = getattr(analysis, "color_cast_index", 0) >= routing.get("hvi_color_cast_threshold", 0.12)
        high_res = max(getattr(analysis, "width", 0), getattr(analysis, "height", 0)) >= routing.get("flol_high_resolution_edge", 2500)
        prefs = getattr(intent, "preferences", None)
        reduce_noise = bool(getattr(prefs, "reduce_noise", False))
        preserve_color = bool(getattr(prefs, "preserve_color", False) or getattr(prefs, "natural_result", False))
        low_memory = bool(getattr(prefs, "low_memory", False)) or (cuda and mem and mem < routing.get("sufficient_gpu_memory_mb", 6000))

        if intent.priority == "quality" and cuda and mem >= routing.get("sufficient_gpu_memory_mb", 6000):
            scores["retinexformer"] += routing.get("retinexformer_quality_bonus", 45)
            reasons["retinexformer"].append("质量优先且 CUDA 显存充足")
            scores["darkir"] += routing.get("darkir_quality_bonus", 24)
            reasons["darkir"].append("质量模式加入联合恢复专家")
            scores["hvi_cidnet"] += routing.get("hvi_quality_bonus", 22)
            reasons["hvi_cidnet"].append("质量模式加入颜色/亮度专家")
        if intent.priority == "balanced" and cuda and mem >= routing.get("sufficient_gpu_memory_mb", 6000):
            scores["retinexformer"] += routing.get("retinexformer_balanced_gpu_bonus", 28)
            reasons["retinexformer"].append("均衡模式且 GPU 资源充足")
            scores["flol"] += routing.get("flol_balanced_bonus", 18)
            reasons["flol"].append("均衡模式加入快速真实低照度候选")
        if intent.priority == "speed":
            scores["flol"] += routing.get("flol_speed_bonus", 55)
            reasons["flol"].append("速度优先，选择快速真实低照度模型")
            scores["sci"] += routing.get("sci_speed_bonus", 45)
            reasons["sci"].append("速度优先轻量候选")
        if not cuda:
            scores["sci"] += routing.get("sci_cpu_bonus", 40)
            reasons["sci"].append("CUDA 不可用，选择轻量模型")
        if low_memory:
            scores["flol"] += routing.get("flol_low_memory_bonus", 35)
            reasons["flol"].append("低显存或高显存压力，加入快速低成本模型")
            scores["sci"] += routing.get("sci_low_memory_bonus", 35)
            reasons["sci"].append("低显存偏好或显存不足")
        if noisy or blurry or reduce_noise:
            scores["darkir"] += routing.get("darkir_degradation_bonus", 42)
            reasons["darkir"].append("噪声/模糊风险较高，加入 DarkIR 联合恢复")
        if color_cast or preserve_color:
            scores["hvi_cidnet"] += routing.get("hvi_color_bonus", 40)
            reasons["hvi_cidnet"].append("色偏或自然色彩需求较高，加入 HVI-CIDNet")
        if high_res:
            scores["flol"] += routing.get("flol_high_resolution_bonus", 30)
            reasons["flol"].append("高分辨率输入，加入 FLOL 快速候选")

        # Zero-DCE is kept as a baseline/fallback, not the first auto candidate.
        if mode != "auto":
            scores["zero_dce"] += 5
        self._apply_semantic_model_bonus(scores, reasons, semantic_analysis, manual_locked=bool(manual_model or manual_weight or mode == "manual"))

        if requested_models:
            requested = set(requested_models)
            for model_id in list(scores):
                if model_id not in requested:
                    scores[model_id] -= 500
                    reasons[model_id].append("不在本次用户请求候选列表中")

        candidates = []
        for model_id, score in scores.items():
            st = status.get(model_id, {})
            if model_id not in available:
                score -= 1000
                reasons[model_id].append(st.get("status_message", "模型不可用"))
            candidates.append({
                "model_id": model_id,
                "score": score,
                "status": "found" if model_id in available else "unavailable",
                "selected": False,
                "reasons": reasons[model_id] or ["默认候选"],
            })
        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected_model = candidates[0]["model_id"] if candidates else "sci"
        if candidates:
            candidates[0]["selected"] = True
        checkpoints = self._route_checkpoints(selected_model, intent, analysis, status.get(selected_model, {}), semantic_analysis)
        selected_checkpoint = checkpoints[0]["checkpoint_id"] if checkpoints else ""
        if checkpoints:
            checkpoints[0]["selected"] = True
        fallback = []
        for cand in candidates[1:]:
            if cand["status"] != "unavailable":
                ck = self._route_checkpoints(cand["model_id"], intent, analysis, status.get(cand["model_id"], {}), semantic_analysis)
                if ck:
                    fallback.append((cand["model_id"], ck[0]["checkpoint_id"]))
        for ck in checkpoints[1:]:
            fallback.insert(0, (selected_model, ck["checkpoint_id"]))
        return RouteResult(selected_model, selected_checkpoint, candidates, checkpoints, fallback, "; ".join(candidates[0]["reasons"]))

    def _route_checkpoints(self, model_id: str, intent, analysis, model_status: dict, semantic_analysis: MultimodalAnalysisResult | None = None) -> list[dict[str, Any]]:
        weights = model_status.get("capabilities", {}).get("weights", [])
        existing = {w["checkpoint_id"]: w for w in weights if w.get("exists") or w.get("status") == "found"}
        if not existing:
            return []

        if getattr(intent, "manual_weight", None) and intent.manual_weight in existing:
            ordered = [intent.manual_weight] + [ck for ck in self._default_checkpoint_order(model_id, intent, analysis) if ck != intent.manual_weight]
        else:
            ordered = self._default_checkpoint_order(model_id, intent, analysis)

        ranked = []
        semantic_bonus = self._semantic_checkpoint_bonus(model_id, semantic_analysis, manual_locked=bool(getattr(intent, "manual_weight", None)))
        for ck in ordered:
            if ck in existing and ck not in {item["checkpoint_id"] for item in ranked}:
                idx = len(ranked)
                bonus = semantic_bonus.get(ck, 0)
                ranked.append({
                    "checkpoint_id": ck,
                    "display_name": existing[ck].get("display_name", ck),
                    "score": 100 - idx * 10 + bonus,
                    "status": "found",
                    "selected": False,
                    "reason": self._checkpoint_reason(model_id, ck, intent, analysis) + (f"；多模态语义建议 +{bonus:.1f} 分" if bonus else ""),
                    "metadata": existing[ck],
                })
        for ck, meta in existing.items():
            if ck not in {item["checkpoint_id"] for item in ranked}:
                idx = len(ranked)
                bonus = semantic_bonus.get(ck, 0)
                ranked.append({
                    "checkpoint_id": ck,
                    "display_name": meta.get("display_name", ck),
                    "score": 100 - idx * 10 + bonus,
                    "status": "found",
                    "selected": False,
                    "reason": self._checkpoint_reason(model_id, ck, intent, analysis) + (f"；多模态语义建议 +{bonus:.1f} 分" if bonus else ""),
                    "metadata": meta,
                })
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    def _default_checkpoint_order(self, model_id: str, intent, analysis) -> list[str]:
        if model_id == "retinexformer":
            ordered = list(self.rules.get("retinexformer_weights", {}).get(intent.scene, self.rules.get("retinexformer_weights", {}).get("unknown", ["lol_v2_real"])))
            if intent.priority == "speed":
                ordered = [x for x in ordered if x != "ntire"] + (["ntire"] if "ntire" in ordered else [])
            return ordered
        if model_id == "darkir":
            blurry = getattr(analysis, "laplacian_sharpness", 9999) < 80
            if blurry or getattr(intent.preferences, "reduce_noise", False):
                if intent.priority == "quality":
                    return ["lol_blur_w64", "real_lsrw", "lol_blur", "all_lol"]
                return ["real_lsrw", "lol_blur", "lol_blur_w64", "all_lol"]
            return ["real_lsrw", "all_lol", "lol_blur", "lol_blur_w64"]
        if model_id == "hvi_cidnet":
            if getattr(analysis, "color_cast_index", 0) >= 0.12 or getattr(intent.preferences, "preserve_color", False):
                return ["sice", "fivek", "lol_blur", "sid"]
            if getattr(analysis, "mean_luminance", 255) <= 25:
                return ["sid", "sice", "lol_blur", "fivek"]
            return ["sice", "fivek", "lol_blur", "sid"]
        if model_id == "flol":
            high_res = max(getattr(analysis, "width", 0), getattr(analysis, "height", 0)) >= 2500
            return ["uhd_ll", "lol_v2_real"] if high_res else ["lol_v2_real", "uhd_ll"]
        if model_id == "sci":
            first = self._select_sci_weight(analysis)
            return [first] + [x for x in ["medium", "easy", "difficult"] if x != first]
        return ["epoch99"]

    def _apply_semantic_model_bonus(self, scores: dict[str, float], reasons: dict[str, list[str]], semantic_analysis: MultimodalAnalysisResult | None, manual_locked: bool) -> None:
        if manual_locked or not semantic_analysis or not semantic_analysis.adopted:
            return
        rules = self.rules.get("multimodal_routing", {})
        cap = float(rules.get("semantic_bonus_max", 20))
        confidence = max(0.0, min(float(semantic_analysis.confidence or semantic_analysis.scene_confidence or 0), 1.0))
        for item in semantic_analysis.model_candidates:
            if item.model_id not in scores:
                continue
            bonus = cap * max(0.0, min(float(item.score), 100.0)) / 100.0 * confidence
            if bonus <= 0:
                continue
            scores[item.model_id] += bonus
            reasons[item.model_id].append(f"多模态语义建议 +{bonus:.1f} 分（上限 {cap:.0f}）")

    def _semantic_checkpoint_bonus(self, model_id: str, semantic_analysis: MultimodalAnalysisResult | None, manual_locked: bool) -> dict[str, float]:
        if manual_locked or not semantic_analysis or not semantic_analysis.adopted:
            return {}
        rules = self.rules.get("multimodal_routing", {})
        cap = float(rules.get("semantic_bonus_max", 20))
        confidence = max(0.0, min(float(semantic_analysis.confidence or semantic_analysis.scene_confidence or 0), 1.0))
        bonuses: dict[str, float] = {}
        for item in semantic_analysis.checkpoint_candidates:
            if item.model_id != model_id:
                continue
            bonus = cap * max(0.0, min(float(item.score), 100.0)) / 100.0 * confidence
            if bonus > bonuses.get(item.checkpoint_id, 0):
                bonuses[item.checkpoint_id] = bonus
        return bonuses

    def _select_sci_weight(self, analysis) -> str:
        th = self.rules.get("sci_thresholds", {})
        diff = th.get("difficult", {})
        easy = th.get("easy", {})
        difficult_votes = 0
        difficult_votes += analysis.mean_luminance <= diff.get("mean_luminance_max", 32)
        difficult_votes += analysis.median_luminance <= diff.get("median_luminance_max", 30)
        difficult_votes += analysis.luminance_p25 <= diff.get("p25_max", 24)
        difficult_votes += analysis.dark_pixel_ratio >= diff.get("dark_ratio_min", 0.72)
        difficult_votes += analysis.dynamic_range <= diff.get("dynamic_range_max", 70)
        if difficult_votes >= 3:
            return "difficult"
        easy_votes = 0
        easy_votes += analysis.mean_luminance >= easy.get("mean_luminance_min", 70)
        easy_votes += analysis.median_luminance >= easy.get("median_luminance_min", 62)
        easy_votes += analysis.dark_pixel_ratio <= easy.get("dark_ratio_max", 0.35)
        if easy_votes >= 2:
            return "easy"
        return "medium"

    def _checkpoint_reason(self, model_id: str, checkpoint_id: str, intent, analysis) -> str:
        if getattr(intent, "manual_weight", None) == checkpoint_id:
            return "用户明确指定权重"
        if model_id == "retinexformer":
            if checkpoint_id == "lol_v2_real":
                return "场景未知或通用真实低照度默认权重"
            if checkpoint_id == "sdsd_indoor":
                return "用户描述包含室内场景"
            if checkpoint_id == "sdsd_outdoor":
                return "用户描述包含室外/城市夜景场景"
            if checkpoint_id == "ntire":
                return "质量优先或高质量备用候选"
        if model_id == "darkir":
            if checkpoint_id == "real_lsrw":
                return "真实低照度联合恢复默认权重"
            if checkpoint_id == "lol_blur_w64":
                return "质量优先的低照度去模糊高容量权重"
            if checkpoint_id == "lol_blur":
                return "低照度模糊风险权重"
            return "通用 LOL 低照度恢复权重"
        if model_id == "hvi_cidnet":
            if checkpoint_id == "sice":
                return "低照度曝光与色彩恢复默认权重"
            if checkpoint_id == "fivek":
                return "自然摄影色彩增强权重"
            if checkpoint_id == "lol_blur":
                return "低照度伴随模糊风险权重"
            return "极暗场景候选权重"
        if model_id == "flol":
            if checkpoint_id == "uhd_ll":
                return "高分辨率快速低照度权重"
            return "快速真实低照度默认权重"
        if model_id == "sci":
            return "根据亮度分位数、暗像素比例和动态范围综合选择"
        return "最后兜底轻量基线"