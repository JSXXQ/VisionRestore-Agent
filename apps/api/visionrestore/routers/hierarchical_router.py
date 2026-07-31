from dataclasses import dataclass
from typing import Any
from visionrestore.core.model_config import get_routing_rules

@dataclass
class RouteResult:
    selected_model: str
    selected_checkpoint: str
    model_candidates: list[dict[str, Any]]
    checkpoint_candidates: list[dict[str, Any]]
    fallback: list[tuple[str, str]]
    reason: str

class HierarchicalRouter:
    def __init__(self):
        self.rules = get_routing_rules()

    def route(self, *, intent, analysis, hardware: dict, model_statuses: list[dict], mode: str, requested_models: list[str] | None = None) -> RouteResult:
        status = {m["model_id"]: m for m in model_statuses}
        available = {m["model_id"] for m in model_statuses if m.get("available")}
        mem = hardware.get("gpu_memory_mb") or 0
        cuda = bool(hardware.get("cuda_available"))
        routing = self.rules.get("model_routing", {})
        scores = {"retinexformer": 0, "sci": 0, "zero_dce": 0}
        reasons = {"retinexformer": [], "sci": [], "zero_dce": []}
        if intent.manual_model:
            scores[intent.manual_model] += 100; reasons[intent.manual_model].append("用户明确指定模型")
        if intent.priority == "quality" and cuda and mem >= routing.get("sufficient_gpu_memory_mb", 6000):
            scores["retinexformer"] += routing.get("retinexformer_quality_bonus", 45); reasons["retinexformer"].append("质量优先且 CUDA 显存充足")
        if intent.priority == "balanced" and cuda and mem >= routing.get("sufficient_gpu_memory_mb", 6000):
            scores["retinexformer"] += routing.get("retinexformer_balanced_gpu_bonus", 28); reasons["retinexformer"].append("均衡模式且 GPU 资源充足")
        if intent.priority == "speed":
            scores["sci"] += routing.get("sci_speed_bonus", 45); reasons["sci"].append("速度优先")
        if not cuda:
            scores["sci"] += routing.get("sci_cpu_bonus", 40); reasons["sci"].append("CUDA 不可用，选择轻量模型")
        if intent.preferences.low_memory or (cuda and mem and mem < routing.get("sufficient_gpu_memory_mb", 6000)):
            scores["sci"] += routing.get("sci_low_memory_bonus", 35); reasons["sci"].append("低显存偏好或显存不足")
        if mode == "manual" and intent.manual_model == "zero_dce":
            scores["zero_dce"] += routing.get("zero_dce_manual_bonus", 30); reasons["zero_dce"].append("手动基线模型")
        # Zero-DCE is not first priority in auto mode.
        if mode != "auto":
            scores["zero_dce"] += 5
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
        candidates[0]["selected"] = True
        checkpoints = self._route_checkpoints(selected_model, intent, analysis, status.get(selected_model, {}))
        selected_checkpoint = checkpoints[0]["checkpoint_id"] if checkpoints else ""
        if checkpoints:
            checkpoints[0]["selected"] = True
        fallback = []
        for cand in candidates[1:]:
            if cand["status"] != "unavailable":
                ck = self._route_checkpoints(cand["model_id"], intent, analysis, status.get(cand["model_id"], {}))
                if ck:
                    fallback.append((cand["model_id"], ck[0]["checkpoint_id"]))
        for ck in checkpoints[1:]:
            fallback.insert(0, (selected_model, ck["checkpoint_id"]))
        return RouteResult(selected_model, selected_checkpoint, candidates, checkpoints, fallback, "; ".join(candidates[0]["reasons"]))

    def _route_checkpoints(self, model_id: str, intent, analysis, model_status: dict) -> list[dict[str, Any]]:
        weights = model_status.get("capabilities", {}).get("weights", [])
        existing = {w["checkpoint_id"]: w for w in weights if w.get("exists")}
        if not existing:
            return []
        ordered: list[str]
        if intent.manual_weight and intent.manual_weight in existing:
            ordered = [intent.manual_weight]
            if "lol_v2_real" in existing and intent.manual_weight != "lol_v2_real":
                ordered.append("lol_v2_real")
        elif model_id == "retinexformer":
            ordered = list(self.rules.get("retinexformer_weights", {}).get(intent.scene, self.rules.get("retinexformer_weights", {}).get("unknown", ["lol_v2_real"])))
            if intent.priority == "speed":
                ordered = [x for x in ordered if x != "ntire"] + (["ntire"] if "ntire" in ordered else [])
        elif model_id == "sci":
            ordered = [self._select_sci_weight(analysis)]
            ordered += [x for x in ["medium", "easy", "difficult"] if x not in ordered]
        else:
            ordered = ["epoch99"]
        ranked = []
        for idx, ck in enumerate(ordered):
            if ck in existing:
                ranked.append({
                    "checkpoint_id": ck,
                    "display_name": existing[ck].get("display_name", ck),
                    "score": 100 - idx * 10,
                    "status": "found",
                    "selected": False,
                    "reason": self._checkpoint_reason(model_id, ck, intent),
                    "metadata": existing[ck],
                })
        return ranked

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

    def _checkpoint_reason(self, model_id: str, checkpoint_id: str, intent) -> str:
        if intent.manual_weight == checkpoint_id:
            return "用户明确指定权重"
        if model_id == "retinexformer":
            if checkpoint_id == "lol_v2_real": return "场景未知或通用真实低照度默认权重"
            if checkpoint_id == "sdsd_indoor": return "用户描述包含室内场景"
            if checkpoint_id == "sdsd_outdoor": return "用户描述包含室外/城市夜景场景"
            if checkpoint_id == "ntire": return "质量优先或高质量备用候选"
        if model_id == "sci":
            return "根据亮度分位数、暗像素比例和动态范围综合选择"
        return "最后兜底轻量基线"
