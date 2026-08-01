from types import SimpleNamespace

from visionrestore.services.residual_analyzer import ResidualDegradationAnalyzer


def _analysis(width=1280, height=720, noise=8):
    return SimpleNamespace(width=width, height=height, noise_estimate=noise)


def _intent(text=""):
    return SimpleNamespace(raw_text=text)


def test_residual_analyzer_recommends_denoise_for_noisy_result():
    result = ResidualDegradationAnalyzer().analyze(
        original_analysis=_analysis(noise=7),
        enhanced_metrics={"noise_estimate_before": 7, "noise_estimate_after": 22, "color_cast_index_after": 0.2, "width": 1200, "height": 800},
        best_candidate={"model_id": "retinexformer"},
        user_intent=_intent(),
        hardware={"gpu_memory_mb": 8192},
    )
    assert result.denoise_recommended is True
    assert result.preferred_denoiser == "lpdm"
    assert result.denoise_reason
    assert result.super_resolution_recommended is True


def test_residual_analyzer_raises_darkir_denoise_threshold():
    result = ResidualDegradationAnalyzer().analyze(
        original_analysis=_analysis(noise=7),
        enhanced_metrics={"noise_estimate_before": 7, "noise_estimate_after": 14, "color_cast_index_after": 0.05, "width": 1600, "height": 1200},
        best_candidate={"model_id": "darkir"},
        user_intent=_intent(),
        hardware={},
    )
    assert result.denoise_recommended is False
    assert any("DarkIR" in item for item in result.denoise_risk) or result.denoise_risk == []


def test_residual_analyzer_does_not_suggest_sr_for_high_resolution_without_user_request():
    result = ResidualDegradationAnalyzer().analyze(
        original_analysis=_analysis(width=3000, height=2000, noise=4),
        enhanced_metrics={"noise_estimate_before": 4, "noise_estimate_after": 5, "width": 3000, "height": 2000},
        best_candidate={"model_id": "retinexformer"},
        user_intent=_intent(),
        hardware={},
    )
    assert result.super_resolution_recommended is False
    assert any("高分辨率" in item for item in result.sr_risk)


def test_residual_analyzer_honors_user_sr_request():
    result = ResidualDegradationAnalyzer().analyze(
        original_analysis=_analysis(width=1400, height=1000, noise=4),
        enhanced_metrics={"noise_estimate_before": 4, "noise_estimate_after": 5, "width": 1400, "height": 1000},
        best_candidate={"model_id": "retinexformer"},
        user_intent=_intent("请超分放大"),
        hardware={},
    )
    assert result.super_resolution_recommended is False or result.sr_confidence > 0
    assert "用户明确表达放大或超分需求" in result.sr_reason
    assert result.preferred_scale == 2
