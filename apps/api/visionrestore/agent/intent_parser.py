from visionrestore.schemas.intent import UserIntent, UserPreferences

class IntentParser:
    def parse(self, text: str, priority: str = "balanced", mode: str = "auto", model_id: str | None = None, weight_id: str | None = None) -> UserIntent:
        raw = text or ""
        t = raw.lower()
        evidence: list[str] = []
        parsed_priority = priority or "balanced"
        if any(k in t for k in ["质量", "quality", "细节", "更长", "longer"]):
            parsed_priority = "quality"; evidence.append("quality keyword")
        if any(k in t for k in ["速度", "快速", "快", "speed", "显存有限", "low memory"]):
            parsed_priority = "speed"; evidence.append("speed/low-memory keyword")
        scene = "unknown"
        if any(k in t for k in ["室内", "房间", "实验室", "走廊", "商场", "indoor", "room"]):
            scene = "indoor"; evidence.append("indoor keyword")
        if any(k in t for k in ["室外", "户外", "街道", "道路", "城市", "夜景", "建筑", "outdoor", "street", "city"]):
            scene = "outdoor"; evidence.append("outdoor keyword")
        manual_model = model_id
        for key, value in [("retinexformer", "retinexformer"), ("sci", "sci"), ("zero-dce", "zero_dce"), ("zero_dce", "zero_dce")]:
            if key in t:
                manual_model = value; evidence.append(f"manual model {value}")
        manual_weight = weight_id
        aliases = {
            "lol-v2-real": "lol_v2_real", "lol_v2_real": "lol_v2_real", "lol v2 real": "lol_v2_real",
            "sdsd-indoor": "sdsd_indoor", "sdsd_indoor": "sdsd_indoor", "indoor weight": "sdsd_indoor",
            "sdsd-outdoor": "sdsd_outdoor", "sdsd_outdoor": "sdsd_outdoor", "outdoor weight": "sdsd_outdoor",
            "ntire": "ntire", "easy": "easy", "medium": "medium", "difficult": "difficult", "epoch99": "epoch99",
        }
        for key, value in aliases.items():
            if key in t:
                manual_weight = value; evidence.append(f"manual weight {value}")
        prefs = UserPreferences(
            preserve_color=any(k in t for k in ["保持原始颜色", "保持颜色", "preserve color", "不要偏色"]),
            protect_highlights=any(k in t for k in ["不要过曝", "保护高光", "路灯", "窗户", "highlight"]),
            recover_shadows=any(k in t for k in ["暗部", "阴影", "recover shadow", "shadow"]),
            reduce_noise=any(k in t for k in ["降噪", "噪声", "noise"]),
            natural_result=any(k in t for k in ["自然", "natural"]),
            strong_enhancement=any(k in t for k in ["强增强", "更亮", "strong"]),
            low_memory=any(k in t for k in ["显存有限", "省显存", "low memory"]),
            allow_long_runtime=any(k in t for k in ["更长", "可以接受", "long runtime"]),
        )
        return UserIntent(
            priority=parsed_priority,
            scene=scene,
            preferences=prefs,
            manual_model=manual_model,
            manual_weight=manual_weight,
            comparison_mode=(mode == "compare" or any(k in t for k in ["对比", "compare"])),
            raw_text=raw,
            evidence=evidence,
        )
