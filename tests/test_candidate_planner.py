from types import SimpleNamespace

from visionrestore.agent.candidate_planner import CandidatePlanner
from visionrestore.schemas.intent import UserIntent, UserPreferences


def _model(model_id, available=True, weights=None, message="ok"):
    return {
        "model_id": model_id,
        "available": available,
        "status_message": message,
        "capabilities": {"weights": weights or [{"checkpoint_id": "default", "exists": True, "status": "found", "default": True}]},
    }


def _analysis(**kwargs):
    defaults = dict(width=1280, height=720, noise_estimate=4, laplacian_sharpness=200, color_cast_index=0.02)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_speed_mode_plans_only_one_fast_candidate():
    statuses = [
        _model("flol", True, [{"checkpoint_id": "lol_v2_real", "exists": True, "status": "found"}]),
        _model("sci", True, [{"checkpoint_id": "medium", "exists": True, "status": "found"}]),
    ]
    plan = CandidatePlanner().plan(image_id="img", intent=UserIntent(priority="speed"), analysis=_analysis(), hardware={"gpu_memory_mb": 8192}, model_statuses=statuses, mode="auto")
    assert plan.max_candidates == 1
    assert len(plan.candidates) == 1
    assert plan.candidates[0].model_id == "flol"
    assert plan.candidates[0].input_policy == "original_input_only"


def test_quality_mode_keeps_unready_new_models_out_of_candidates():
    statuses = [
        _model("retinexformer", True, [{"checkpoint_id": "lol_v2_real", "exists": True, "status": "found", "default": True}]),
        _model("darkir", False, message="尚未通过真实小图健康检查"),
        _model("hvi_cidnet", False, message="worker脚本不存在"),
        _model("sci", True, [{"checkpoint_id": "medium", "exists": True, "status": "found"}]),
    ]
    intent = UserIntent(priority="quality", preferences=UserPreferences(reduce_noise=True, preserve_color=True))
    plan = CandidatePlanner().plan(image_id="img", intent=intent, analysis=_analysis(noise_estimate=18, color_cast_index=0.2), hardware={"gpu_memory_mb": 8192}, model_statuses=statuses, mode="auto")
    assert plan.max_candidates == 3
    assert [item.model_id for item in plan.candidates] == ["retinexformer", "sci"]
    rejected = {item["model_id"]: item["reason"] for item in plan.rejected}
    assert "darkir" in rejected
    assert "hvi_cidnet" in rejected


def test_manual_choice_is_not_overridden_by_planner():
    statuses = [
        _model("sci", True, [{"checkpoint_id": "difficult", "exists": True, "status": "found"}]),
        _model("retinexformer", True, [{"checkpoint_id": "lol_v2_real", "exists": True, "status": "found"}]),
    ]
    intent = UserIntent(priority="quality", manual_model="sci", manual_weight="difficult")
    plan = CandidatePlanner().plan(image_id="img", intent=intent, analysis=_analysis(noise_estimate=20), hardware={"gpu_memory_mb": 8192}, model_statuses=statuses, mode="manual")
    assert len(plan.candidates) == 1
    assert plan.candidates[0].model_id == "sci"
    assert plan.candidates[0].checkpoint_id == "difficult"
    assert "手动" in plan.notes[0]
