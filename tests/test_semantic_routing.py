from types import SimpleNamespace

from visionrestore.routers.hierarchical_router import HierarchicalRouter
from visionrestore.schemas.ai import MultimodalAnalysisResult
from visionrestore.schemas.intent import UserIntent, UserPreferences


def _models():
    return [
        {
            "model_id": "retinexformer",
            "available": True,
            "supported_devices": ["cuda"],
            "capabilities": {"weights": [
                {"checkpoint_id": "lol_v2_real", "exists": True, "status": "found", "display_name": "LOL-v2-real"},
                {"checkpoint_id": "ntire", "exists": True, "status": "found", "display_name": "NTIRE"},
            ]},
        },
        {
            "model_id": "sci",
            "available": True,
            "supported_devices": ["cuda"],
            "capabilities": {"weights": [
                {"checkpoint_id": "medium", "exists": True, "status": "found", "display_name": "medium"},
            ]},
        },
        {
            "model_id": "zero_dce",
            "available": True,
            "supported_devices": ["cuda"],
            "capabilities": {"weights": [
                {"checkpoint_id": "epoch99", "exists": True, "status": "found", "display_name": "Epoch99"},
            ]},
        },
    ]


def test_adopted_multimodal_analysis_adds_limited_route_score():
    intent = UserIntent(priority="balanced", scene="unknown", preferences=UserPreferences())
    analysis = SimpleNamespace(
        mean_luminance=30,
        median_luminance=28,
        luminance_p25=20,
        dark_pixel_ratio=0.7,
        dynamic_range=60,
    )
    semantic = MultimodalAnalysisResult(
        provider="test",
        model="vision-model",
        adopted=True,
        validation_passed=True,
        confidence=1,
        scene="unknown",
        model_candidates=[{"model_id": "sci", "score": 100, "reason": "semantic speed hint"}],
        checkpoint_candidates=[{"model_id": "retinexformer", "checkpoint_id": "ntire", "score": 100, "reason": "quality detail hint"}],
    )

    route = HierarchicalRouter().route(
        intent=intent,
        analysis=analysis,
        hardware={"cuda_available": True, "gpu_memory_mb": 8192},
        model_statuses=_models(),
        mode="auto",
        semantic_analysis=semantic,
    )

    sci = next(item for item in route.model_candidates if item["model_id"] == "sci")
    assert sci["score"] == 20
    assert any("多模态语义建议" in reason for reason in sci["reasons"])
    assert route.checkpoint_candidates[0]["checkpoint_id"] == "ntire"
    assert route.checkpoint_candidates[0]["score"] == 110


def test_new_installed_models_participate_in_main_routing():
    intent = UserIntent(priority="balanced", scene="unknown", preferences=UserPreferences(reduce_noise=True, preserve_color=True))
    analysis = SimpleNamespace(
        mean_luminance=28,
        median_luminance=24,
        luminance_p25=18,
        dark_pixel_ratio=0.72,
        dynamic_range=58,
        noise_estimate=18,
        laplacian_sharpness=45,
        color_cast_index=0.18,
        width=1920,
        height=1080,
    )
    models = _models() + [
        {
            "model_id": "darkir",
            "available": True,
            "supported_devices": ["cuda"],
            "capabilities": {"weights": [
                {"checkpoint_id": "real_lsrw", "exists": True, "status": "found", "display_name": "DarkIR real-LSRW"},
                {"checkpoint_id": "lol_blur", "exists": True, "status": "found", "display_name": "DarkIR LOLBlur"},
            ]},
        },
        {
            "model_id": "hvi_cidnet",
            "available": True,
            "supported_devices": ["cuda"],
            "capabilities": {"weights": [
                {"checkpoint_id": "sice", "exists": True, "status": "found", "display_name": "HVI-CIDNet SICE"},
                {"checkpoint_id": "fivek", "exists": True, "status": "found", "display_name": "HVI-CIDNet FiveK"},
            ]},
        },
        {
            "model_id": "flol",
            "available": True,
            "supported_devices": ["cuda"],
            "capabilities": {"weights": [
                {"checkpoint_id": "lol_v2_real", "exists": True, "status": "found", "display_name": "FLOL LOLv2-Real"},
            ]},
        },
    ]

    route = HierarchicalRouter().route(
        intent=intent,
        analysis=analysis,
        hardware={"cuda_available": True, "gpu_memory_mb": 8192},
        model_statuses=models,
        mode="auto",
    )

    ids = [item["model_id"] for item in route.model_candidates[:4]]
    assert "darkir" in ids
    assert "hvi_cidnet" in ids
    assert route.selected_model in {"darkir", "hvi_cidnet"}
