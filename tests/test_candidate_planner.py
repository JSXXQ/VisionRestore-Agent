from types import SimpleNamespace

from visionrestore.agent.candidate_planner import CandidatePlanner
from visionrestore.schemas.ai import MultimodalAnalysisResult
from visionrestore.schemas.intent import UserIntent, UserPreferences


def _model(model_id, available=True, weights=None, message="ok"):
    return {
        "model_id": model_id,
        "available": available,
        "status_message": message,
        "capabilities": {"weights": weights or [{"checkpoint_id": "default", "exists": True, "status": "found", "default": True}]},
    }


def _analysis(**kwargs):
    defaults = dict(
        width=1280,
        height=720,
        total_pixels=1280 * 720,
        mean_luminance=85,
        dark_pixel_ratio=0.35,
        noise_estimate=4,
        laplacian_sharpness=200,
        color_cast_index=0.02,
        image_entropy=6.5,
    )
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
    assert plan.candidates[0].knowledge_adjustment == 0
    assert "手动" in plan.notes[0]


def test_retrieved_context_is_an_llm_reference_not_an_independent_score():
    statuses = [
        _model("retinexformer", True, [{"checkpoint_id": "lol_v2_real", "exists": True, "status": "found"}]),
        _model("darkir", True, [{"checkpoint_id": "real_lsrw", "exists": True, "status": "found"}]),
        _model("sci", True, [{"checkpoint_id": "medium", "exists": True, "status": "found"}]),
    ]
    context = [{
        "item_id": "model-role-darkir",
        "source": "model_roles",
        "title": "DarkIR role",
        "content": "trusted local role description",
        "tags": ["darkir", "noise", "blur"],
        "matched_terms": ["darkir", "noise", "blur"],
        "score": 4.25,
    }]

    plan = CandidatePlanner().plan(
        image_id="img",
        intent=UserIntent(priority="quality", preferences=UserPreferences(reduce_noise=True)),
        analysis=_analysis(noise_estimate=18, laplacian_sharpness=60),
        hardware={"gpu_memory_mb": 8192},
        model_statuses=statuses,
        mode="auto",
        retrieved_context=context,
    )

    darkir = next(item for item in plan.candidates if item.model_id == "darkir")
    assert darkir.knowledge_adjustment == 0
    assert darkir.knowledge_evidence == []
    assert darkir.planning_score == darkir.local_score
    assert darkir.planning_mode == "local_fallback"
    assert {item.model_id for item in plan.candidates} <= {item["model_id"] for item in statuses}


def test_planning_score_has_explainable_layered_evidence():
    statuses = [
        _model("retinexformer", True, [{"checkpoint_id": "lol_v2_real", "exists": True, "status": "found"}]),
        _model("darkir", True, [{"checkpoint_id": "real_lsrw", "exists": True, "status": "found"}]),
        _model("hvi_cidnet", True, [{"checkpoint_id": "sice", "exists": True, "status": "found"}]),
    ]
    plan = CandidatePlanner().plan(
        image_id="img",
        intent=UserIntent(priority="quality", preferences=UserPreferences(reduce_noise=True)),
        analysis=_analysis(mean_luminance=28, dark_pixel_ratio=0.72, noise_estimate=19, laplacian_sharpness=55),
        hardware={"gpu_memory_mb": 8192},
        model_statuses=statuses,
        mode="auto",
    )

    darkir = next(item for item in plan.candidates if item.model_id == "darkir")
    assert darkir.planning_score == darkir.local_score
    assert darkir.local_score == darkir.model_prior_score + darkir.input_match_score
    assert {item.source for item in darkir.planning_evidence} >= {"model_prior_config", "image_analyzer"}
    assert {item.signal for item in darkir.planning_evidence} >= {"noise_level", "blur_level"}


def test_input_degradation_changes_expert_match_scores():
    statuses = [
        _model("retinexformer", True, [{"checkpoint_id": "lol_v2_real", "exists": True, "status": "found"}]),
        _model("darkir", True, [{"checkpoint_id": "real_lsrw", "exists": True, "status": "found"}]),
        _model("hvi_cidnet", True, [{"checkpoint_id": "sice", "exists": True, "status": "found"}]),
    ]
    plan = CandidatePlanner().plan(
        image_id="img",
        intent=UserIntent(priority="quality"),
        analysis=_analysis(noise_estimate=20, laplacian_sharpness=35, color_cast_index=0.30),
        hardware={"gpu_memory_mb": 8192},
        model_statuses=statuses,
        mode="auto",
    )
    by_model = {item.model_id: item for item in plan.candidates}

    assert by_model["darkir"].input_match_score > by_model["retinexformer"].input_match_score
    hvi_color = next(
        evidence for evidence in by_model["hvi_cidnet"].planning_evidence if evidence.signal == "color_shift"
    )
    assert hvi_color.contribution == 19


def test_flol_automatic_planning_selects_uhd_checkpoint_for_high_resolution_input():
    statuses = [
        _model(
            "flol",
            True,
            [
                {"checkpoint_id": "lol_v2_real", "exists": True, "status": "found"},
                {"checkpoint_id": "uhd_ll", "exists": True, "status": "found"},
            ],
        )
    ]
    plan = CandidatePlanner().plan(
        image_id="img",
        intent=UserIntent(priority="speed"),
        analysis=_analysis(width=3200, height=1800, total_pixels=3200 * 1800),
        hardware={"gpu_memory_mb": 8192},
        model_statuses=statuses,
        mode="auto",
    )

    candidate = plan.candidates[0]
    assert candidate.checkpoint_id == "uhd_ll"
    assert candidate.checkpoint_selection_mode == "local_checkpoint_match"
    assert candidate.checkpoint_score > 0
    assert any(item.signal == "high_resolution" for item in candidate.checkpoint_evidence)


def test_darkir_selects_high_capacity_blur_checkpoint_when_quality_and_hardware_allow():
    statuses = [
        _model(
            "darkir",
            True,
            [
                {"checkpoint_id": "real_lsrw", "exists": True, "status": "found", "default": True},
                {"checkpoint_id": "lol_blur", "exists": True, "status": "found"},
                {"checkpoint_id": "lol_blur_w64", "exists": True, "status": "found"},
                {"checkpoint_id": "all_lol", "exists": True, "status": "found"},
            ],
        )
    ]
    intent = UserIntent(
        priority="quality",
        raw_text="严重模糊，可以接受更长运行时间",
        preferences=UserPreferences(reduce_noise=True, allow_long_runtime=True),
    )
    plan = CandidatePlanner().plan(
        image_id="img",
        intent=intent,
        analysis=_analysis(noise_estimate=14, laplacian_sharpness=35, image_entropy=5.0),
        hardware={"cuda_available": True, "gpu_memory_mb": 8192},
        model_statuses=statuses,
        mode="auto",
    )

    candidate = plan.candidates[0]
    assert candidate.checkpoint_id == "lol_blur_w64"
    assert candidate.planning_score == candidate.local_score
    assert any(item.signal == "blur_level" for item in candidate.checkpoint_evidence)


def test_darkir_falls_back_inside_family_when_w64_is_hardware_blocked():
    statuses = [
        _model(
            "darkir",
            True,
            [
                {"checkpoint_id": "real_lsrw", "exists": True, "status": "found", "default": True},
                {"checkpoint_id": "lol_blur", "exists": True, "status": "found"},
                {"checkpoint_id": "lol_blur_w64", "exists": True, "status": "found"},
            ],
        )
    ]
    plan = CandidatePlanner().plan(
        image_id="img",
        intent=UserIntent(priority="quality", raw_text="严重模糊"),
        analysis=_analysis(noise_estimate=8, laplacian_sharpness=35, image_entropy=5.0),
        hardware={"cuda_available": True, "gpu_memory_mb": 4096},
        model_statuses=statuses,
        mode="auto",
    )

    candidate = plan.candidates[0]
    assert candidate.checkpoint_id == "lol_blur"
    blocked = next(item for item in candidate.checkpoint_candidates if item["checkpoint_id"] == "lol_blur_w64")
    assert blocked["eligible"] is False
    assert "显存" in blocked["reason"]


def test_retinexformer_selects_scene_specific_checkpoint():
    statuses = [
        _model(
            "retinexformer",
            True,
            [
                {"checkpoint_id": "lol_v2_real", "exists": True, "status": "found", "default": True},
                {"checkpoint_id": "sdsd_indoor", "exists": True, "status": "found"},
                {"checkpoint_id": "sdsd_outdoor", "exists": True, "status": "found"},
                {"checkpoint_id": "ntire", "exists": True, "status": "found"},
            ],
        )
    ]
    plan = CandidatePlanner().plan(
        image_id="img",
        intent=UserIntent(priority="quality", scene="indoor", raw_text="室内走廊低照度"),
        analysis=_analysis(mean_luminance=45, dark_pixel_ratio=0.60),
        hardware={"cuda_available": True, "gpu_memory_mb": 8192},
        model_statuses=statuses,
        mode="auto",
    )

    assert plan.candidates[0].checkpoint_id == "sdsd_indoor"


def test_validated_semantic_scene_may_inform_checkpoint_but_direct_checkpoint_advice_is_ignored():
    statuses = [
        _model(
            "retinexformer",
            True,
            [
                {"checkpoint_id": "lol_v2_real", "exists": True, "status": "found", "default": True},
                {"checkpoint_id": "sdsd_indoor", "exists": True, "status": "found"},
                {"checkpoint_id": "sdsd_outdoor", "exists": True, "status": "found"},
                {"checkpoint_id": "ntire", "exists": True, "status": "found"},
            ],
        )
    ]
    advisory = MultimodalAnalysisResult(
        provider="openai_compatible",
        model="vlm",
        scene="outdoor",
        scene_confidence=0.92,
        model_candidates=[{"model_id": "retinexformer", "score": 80, "reason": "outdoor night"}],
        checkpoint_candidates=[{
            "model_id": "retinexformer",
            "checkpoint_id": "ntire",
            "score": 100,
            "reason": "direct checkpoint advice must not control local routing",
        }],
        confidence=0.9,
        validation_passed=True,
        adopted=True,
    )
    plan = CandidatePlanner().plan(
        image_id="img",
        intent=UserIntent(priority="balanced", scene="unknown"),
        analysis=_analysis(mean_luminance=45, dark_pixel_ratio=0.60),
        hardware={"cuda_available": True, "gpu_memory_mb": 8192},
        model_statuses=statuses,
        mode="auto",
        semantic_analysis=advisory,
    )

    candidate = plan.candidates[0]
    assert candidate.checkpoint_id == "sdsd_outdoor"
    assert candidate.checkpoint_id != "ntire"
    assert any(item.source == "semantic_scene_advisory" for item in candidate.checkpoint_evidence)


def test_hvi_selects_sid_for_extreme_dark_input():
    statuses = [
        _model(
            "hvi_cidnet",
            True,
            [
                {"checkpoint_id": "sice", "exists": True, "status": "found", "default": True},
                {"checkpoint_id": "fivek", "exists": True, "status": "found"},
                {"checkpoint_id": "lol_blur", "exists": True, "status": "found"},
                {"checkpoint_id": "sid", "exists": True, "status": "found"},
            ],
        )
    ]
    plan = CandidatePlanner().plan(
        image_id="img",
        intent=UserIntent(
            priority="quality",
            raw_text="极暗画面，需要强增强和降噪",
            preferences=UserPreferences(strong_enhancement=True, reduce_noise=True),
        ),
        analysis=_analysis(mean_luminance=20, dark_pixel_ratio=0.82, noise_estimate=18),
        hardware={"cuda_available": True, "gpu_memory_mb": 8192},
        model_statuses=statuses,
        mode="auto",
    )

    assert plan.candidates[0].checkpoint_id == "sid"


def test_sci_selects_easy_medium_and_difficult_by_degradation():
    statuses = [
        _model(
            "sci",
            True,
            [
                {"checkpoint_id": "easy", "exists": True, "status": "found"},
                {"checkpoint_id": "medium", "exists": True, "status": "found", "default": True},
                {"checkpoint_id": "difficult", "exists": True, "status": "found"},
            ],
        )
    ]

    def selected(analysis):
        return CandidatePlanner().plan(
            image_id="img",
            intent=UserIntent(priority="speed"),
            analysis=analysis,
            hardware={"gpu_memory_mb": 2048},
            model_statuses=statuses,
            mode="auto",
        ).candidates[0].checkpoint_id

    assert selected(_analysis(mean_luminance=105, dark_pixel_ratio=0.10, noise_estimate=3)) == "easy"
    assert selected(_analysis(mean_luminance=58, dark_pixel_ratio=0.48, noise_estimate=7)) == "medium"
    assert selected(_analysis(mean_luminance=20, dark_pixel_ratio=0.82, noise_estimate=18, laplacian_sharpness=45, image_entropy=4.5)) == "difficult"


def test_valid_llm_scores_are_fused_with_local_scores_fifty_fifty():
    statuses = [
        _model("retinexformer", True, [{"checkpoint_id": "lol_v2_real", "exists": True, "status": "found"}]),
        _model("darkir", True, [{"checkpoint_id": "real_lsrw", "exists": True, "status": "found"}]),
    ]
    advisory = MultimodalAnalysisResult(
        provider="openai_compatible",
        model="vlm",
        model_candidates=[
            {"model_id": "retinexformer", "score": 40, "reason": "general illumination"},
            {"model_id": "darkir", "score": 90, "reason": "visible real noise and blur"},
        ],
        confidence=0.9,
        validation_passed=True,
        adopted=True,
    )

    plan = CandidatePlanner().plan(
        image_id="img",
        intent=UserIntent(priority="quality"),
        analysis=_analysis(noise_estimate=18, laplacian_sharpness=60),
        hardware={"gpu_memory_mb": 8192},
        model_statuses=statuses,
        mode="auto",
        semantic_analysis=advisory,
    )
    by_model = {item.model_id: item for item in plan.candidates}

    for item in by_model.values():
        assert item.local_weight == 0.5
        assert item.llm_weight == 0.5
        assert item.planning_mode == "local_llm_fusion"
        assert item.planning_score == round(item.local_score * 0.5 + item.llm_score * 0.5, 2)
    assert by_model["darkir"].llm_score == 90
