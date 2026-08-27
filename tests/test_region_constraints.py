from types import SimpleNamespace

import numpy as np
from PIL import Image

from visionrestore.schemas.region import RegionConstraint
from visionrestore.services.candidate_evaluator import CandidateEvaluator
from visionrestore.services.region_constraints import RegionConstraintService


def _write_image(path, value=30, bright_box=None):
    arr = np.full((64, 64, 3), value, dtype=np.uint8)
    if bright_box:
        x1, y1, x2, y2 = bright_box
        arr[y1:y2, x1:x2] = 255
    Image.fromarray(arr).save(path)


def test_region_constraint_service_uses_explicit_bbox(tmp_path):
    image = tmp_path / "input.png"
    _write_image(image)
    constraints = RegionConstraintService().build_constraints(
        user_request="路灯不要过曝",
        image_path=str(image),
        image_size=[64, 64],
        parameters={"region_constraints": [{"target": "streetlight", "bbox": [10, 10, 20, 20]}]},
    )

    assert len(constraints) == 1
    assert constraints[0].target == "streetlight"
    assert constraints[0].bbox == [10, 10, 20, 20]
    assert constraints[0].source == "api"


def test_region_constraint_service_detects_local_light_source_from_prompt(tmp_path):
    image = tmp_path / "input.png"
    _write_image(image, bright_box=[42, 5, 48, 12])
    constraints = RegionConstraintService().build_constraints(
        user_request="请增强暗部，但路灯不要过曝",
        image_path=str(image),
        image_size=[64, 64],
        parameters={},
    )

    assert constraints
    assert constraints[0].constraint_type == "avoid_overexposure"
    assert constraints[0].source == "local_highlight_detector"


def test_region_constraint_evaluation_flags_hard_overexposure(tmp_path):
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    _write_image(before, value=40)
    _write_image(after, value=40, bright_box=[10, 10, 24, 24])
    constraint = RegionConstraint(target="streetlight", bbox=[10, 10, 24, 24], max_overexposed_ratio=0.05)

    result = RegionConstraintService().evaluate(
        input_path=str(before),
        output_path=str(after),
        constraints=[constraint],
    )

    assert result["enabled"] is True
    assert result["hard_failed"] is True
    assert result["violations"]


def test_candidate_evaluator_eliminates_region_hard_failure():
    candidate = {
        "candidate_id": "c1",
        "model_id": "retinexformer",
        "checkpoint_id": "lol_v2_real",
        "status": "completed",
        "metrics": {
            "score": 82,
            "components": {
                "shadow_recovery": 0.8,
                "highlight_protection": 0.8,
                "color_stability": 0.8,
                "noise_control": 0.8,
                "sharpness": 0.8,
                "structure": 0.8,
            },
            "mean_luminance_after": 80,
            "overexposed_pixel_ratio_after": 0.02,
            "color_cast_index_after": 0.05,
            "structure_keep_estimate": 0.9,
            "region_constraints": {
                "enabled": True,
                "score": 0.1,
                "hard_failed": True,
                "severe_failure": True,
                "violations": ["streetlight区域过曝比例超过阈值"],
            },
        },
    }

    score = CandidateEvaluator().score(candidate)

    assert score.eliminated is True
    assert any("streetlight" in reason for reason in score.reasons)


def test_candidate_evaluator_keeps_mild_region_failure_as_soft_penalty():
    candidate = {
        "candidate_id": "c1",
        "model_id": "retinexformer",
        "checkpoint_id": "lol_v2_real",
        "status": "completed",
        "metrics": {
            "score": 82,
            "components": {
                "shadow_recovery": 0.82,
                "highlight_protection": 0.82,
                "color_stability": 0.82,
                "noise_control": 0.82,
                "sharpness": 0.82,
                "structure": 0.82,
            },
            "mean_luminance_after": 80,
            "overexposed_pixel_ratio_after": 0.02,
            "color_cast_index_after": 0.05,
            "structure_keep_estimate": 0.9,
            "region_constraints": {
                "enabled": True,
                "score": 0.72,
                "hard_failed": True,
                "severe_failure": False,
                "items": [
                    {
                        "priority": "hard",
                        "overexposed_ratio_before": 0.02,
                        "overexposed_ratio_after": 0.11,
                        "luminance_p95_before": 244,
                        "luminance_p95_after": 249,
                    }
                ],
                "violations": ["streetlight区域轻微超过阈值"],
            },
        },
    }

    score = CandidateEvaluator().score(candidate)

    assert score.eliminated is False
    assert score.score > 0


def test_multimodal_region_constraint_is_accepted(tmp_path):
    image = tmp_path / "input.png"
    _write_image(image)
    ai = SimpleNamespace(region_constraints=[
        RegionConstraint(target="streetlight", bbox=[1, 2, 12, 14], source="multimodal", confidence=0.8)
    ])

    constraints = RegionConstraintService().build_constraints(
        user_request="protect the streetlight",
        image_path=str(image),
        image_size=[64, 64],
        parameters={},
        ai_analysis=ai,
    )

    assert constraints[0].source == "multimodal"
    assert constraints[0].bbox == [1, 2, 12, 14]
