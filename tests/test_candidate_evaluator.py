from visionrestore.services.candidate_evaluator import CandidateEvaluator, CandidateRanker


def _candidate(candidate_id, score=80, runtime_ms=1000, status="completed", **extra):
    metrics = {
        "score": score,
        "components": {
            "shadow_recovery": score / 100,
            "highlight_protection": 0.8,
            "color_stability": 0.8,
            "noise_control": 0.75,
            "sharpness": 0.8,
            "structure": 0.85,
        },
        "runtime_ms": runtime_ms,
        "peak_memory_mb": 2000,
        "mean_luminance_after": 80,
        "overexposed_pixel_ratio_after": 0.02,
        "color_cast_index_after": 0.05,
        "structure_keep_estimate": 0.8,
    }
    metrics.update(extra)
    return {"candidate_id": candidate_id, "model_id": "retinexformer", "checkpoint_id": "lol_v2_real", "status": status, "metrics": metrics}


def test_candidate_evaluator_eliminates_invalid_output():
    score = CandidateEvaluator().score(_candidate("bad", mean_luminance_after=0), priority="quality")
    assert score.eliminated is True
    assert score.score == 0
    assert any("纯黑" in reason for reason in score.reasons)


def test_candidate_ranker_selects_best_successful_candidate():
    ranking = CandidateRanker().rank([_candidate("a", 60), _candidate("b", 90)], priority="balanced")
    assert ranking["best"].candidate_id == "b"
    assert ranking["second_best"].candidate_id == "a"
    assert ranking["eliminated"] == []


def test_candidate_ranker_flags_close_competition():
    ranking = CandidateRanker().rank([_candidate("a", 80), _candidate("b", 81)], priority="quality")
    assert ranking["close_competition"] is True
    assert "人工对比" in ranking["note"]


def test_missing_iqa_falls_back_to_local_technical_quality():
    score = CandidateEvaluator().score(_candidate("good", 90), priority="quality")
    assert set(score.layers) == {"image_quality", "restoration", "constraint", "stability"}
    assert score.evidence["image_quality"]["source"] == "local_perceptual_fallback"


def test_iqa_is_blended_with_local_perceptual_quality():
    score = CandidateEvaluator().score(
        _candidate("hybrid", iqa_normalized={"musiq": 0.8, "clipiqa": 0.6}),
        priority="quality",
    )

    evidence = score.evidence["image_quality"]
    assert evidence["source"] == "hybrid_no_reference_iqa"
    assert evidence["iqa_blend_weight"] == 0.7
    assert evidence["metric_coverage"] == "complete"


def test_cleaner_dark_candidate_beats_brighter_noisy_candidate():
    noisy_bright = _candidate(
        "retinex",
        90,
        components={
            "shadow_recovery": 1.0,
            "highlight_protection": 0.992,
            "color_stability": 1.0,
            "noise_control": 0.75,
            "artifact_control": 0.61,
            "sharpness": 1.0,
            "structure": 0.8088,
        },
        iqa_normalized={"musiq": 0.5573, "clipiqa": 0.5182},
        runtime_ms=1279,
        peak_memory_mb=571,
    )
    cleaner_dark = _candidate(
        "darkir",
        90,
        components={
            "shadow_recovery": 0.941,
            "highlight_protection": 0.9987,
            "color_stability": 1.0,
            "noise_control": 0.84,
            "artifact_control": 0.746,
            "sharpness": 1.0,
            "structure": 0.8649,
        },
        iqa_normalized={"musiq": 0.6699, "clipiqa": 0.4208},
        runtime_ms=777,
        peak_memory_mb=498,
    )

    ranking = CandidateRanker().rank([noisy_bright, cleaner_dark], priority="balanced")

    assert ranking["best"].candidate_id == "darkir"


def test_reference_like_colorful_output_beats_brighter_desaturated_output():
    brighter = _candidate(
        "brighter_desaturated",
        components={
            "shadow_recovery": 0.991,
            "highlight_protection": 0.7291,
            "color_stability": 1.0,
            "noise_control": 0.5737,
            "artifact_control": 0.6735,
            "sharpness": 1.0,
            "structure": 0.7606,
            "brightness_target_fit": 0.9871,
            "brightness_recovery_gain": 1.0,
        },
        iqa_normalized={"musiq": 0.5538, "clipiqa": 0.6552},
        mean_luminance_after=98.43,
        overexposed_pixel_ratio_after=0.0271,
        color_cast_index_before=0.0843,
        color_cast_index_after=0.0245,
        sharpness_ratio=4.6054,
        structure_keep_estimate=0.7606,
    )
    reference_like = _candidate(
        "reference_like_colorful",
        components={
            "shadow_recovery": 0.944,
            "highlight_protection": 1.0,
            "color_stability": 0.7938,
            "noise_control": 0.6088,
            "artifact_control": 0.7073,
            "sharpness": 1.0,
            "structure": 0.8508,
            "brightness_target_fit": 0.92,
            "brightness_recovery_gain": 1.0,
        },
        iqa_normalized={"musiq": 0.6557, "clipiqa": 0.6789},
        mean_luminance_after=73.68,
        overexposed_pixel_ratio_after=0.0,
        color_cast_index_before=0.0843,
        color_cast_index_after=0.1530,
        sharpness_ratio=5.1058,
        structure_keep_estimate=0.8508,
    )

    ranking = CandidateRanker().rank([brighter, reference_like], priority="balanced")

    assert ranking["best"].candidate_id == "reference_like_colorful"


def test_final_score_excludes_planning_knowledge_and_runtime_cost():
    evaluator = CandidateEvaluator()
    first = _candidate("first", 85, runtime_ms=500)
    first["planning_score"] = 100
    first["metrics"]["knowledge_adjustment"] = 5
    second = _candidate("second", 85, runtime_ms=25000)
    second["planning_score"] = 10
    second["metrics"]["knowledge_adjustment"] = -5

    first_score = evaluator.score(first, priority="balanced")
    second_score = evaluator.score(second, priority="balanced")

    assert first_score.score == second_score.score
    assert first_score.evidence["excluded_from_final_score"]["planning_score"] == 100
    assert second_score.evidence["excluded_from_final_score"]["runtime_ms"] == 25000


def test_stability_layer_detects_soft_overexposure_without_hard_elimination():
    stable = CandidateEvaluator().score(
        _candidate("stable", overexposed_pixel_ratio_after=0.02), priority="balanced"
    )
    unstable = CandidateEvaluator().score(
        _candidate("unstable", overexposed_pixel_ratio_after=0.30), priority="balanced"
    )

    assert unstable.eliminated is False
    assert unstable.layers["stability"] < stable.layers["stability"]
    assert "overexposure_safety" in unstable.evidence["stability"]["detected_anomalies"]


def test_constraint_layer_uses_roi_result_and_keeps_hard_validity_boundary():
    score = CandidateEvaluator().score(
        _candidate(
            "roi",
            region_constraints={
                "enabled": True,
                "score": 0.72,
                "hard_failed": False,
                "items": [{"constraint_id": "lamp", "passed": True}],
            },
        ),
        priority="quality",
    )

    assert score.layers["constraint"] == 0.72
    assert score.evidence["constraint"]["source"] == "region_constraint_evaluator"


def test_numeric_instability_is_a_hard_failure():
    score = CandidateEvaluator().score(_candidate("nan", has_nan=True), priority="quality")

    assert score.eliminated is True
    assert "输出包含NaN或Inf" in score.reasons


def test_severe_output_color_shift_is_penalized_without_model_specific_rules():
    neutral = _candidate(
        "neutral",
        color_cast_index_before=0.04,
        color_cast_index_after=0.06,
    )
    cast = _candidate(
        "cast",
        color_cast_index_before=0.04,
        color_cast_index_after=0.58,
    )

    neutral_score = CandidateEvaluator().score(neutral, priority="quality")
    cast_score = CandidateEvaluator().score(cast, priority="quality")

    assert neutral_score.score > cast_score.score
    assert (
        neutral_score.evidence["image_quality"]["components"]["scene_color_naturalness"]
        > cast_score.evidence["image_quality"]["components"]["scene_color_naturalness"]
    )


def test_natural_dominant_scene_color_is_not_misclassified_as_a_cast():
    score = CandidateEvaluator().score(
        _candidate(
            "naturally_colorful",
            color_cast_index_before=0.09,
            color_cast_index_after=0.16,
            iqa_normalized={"musiq": 0.66, "clipiqa": 0.68},
            components={
                "shadow_recovery": 0.94,
                "highlight_protection": 1.0,
                "color_stability": 0.79,
                "noise_control": 0.61,
                "artifact_control": 0.71,
                "sharpness": 1.0,
                "structure": 0.85,
            },
        ),
        priority="quality",
    )

    evidence = score.evidence["stability"]["color_naturalness"]
    assert evidence["score"] == 1.0
    assert evidence["source"] == "scene_adaptive_color_consistency"
    assert evidence["iqa_supported"] is True


def test_effective_detail_curve_penalizes_overly_smooth_output():
    detailed = _candidate("detailed", sharpness_ratio=12.0)
    smooth = _candidate("smooth", sharpness_ratio=3.0)

    detailed_score = CandidateEvaluator().score(detailed, priority="quality")
    smooth_score = CandidateEvaluator().score(smooth, priority="quality")

    assert detailed_score.score > smooth_score.score
    assert (
        detailed_score.evidence["image_quality"]["components"]["effective_sharpness"]
        > smooth_score.evidence["image_quality"]["components"]["effective_sharpness"]
    )
