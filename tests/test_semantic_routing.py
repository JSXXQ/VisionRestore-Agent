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
