import json
from pathlib import Path

from visionrestore.context import ContextRetrievalService, PlanningKnowledgeAdapter
from visionrestore.core.config import PROJECT_ROOT
from visionrestore.schemas.ai import CHECKPOINTS_BY_MODEL
from visionrestore.schemas.candidate import CandidatePlanItem


KNOWLEDGE_PATH = PROJECT_ROOT / "apps" / "api" / "knowledge" / "model_capability.json"


def _load_payload() -> dict:
    return json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))


def test_model_capability_knowledge_covers_registered_enhancement_checkpoints():
    payload = _load_payload()
    items = payload["items"]
    expected = {
        (model_id, checkpoint_id)
        for model_id, checkpoints in CHECKPOINTS_BY_MODEL.items()
        for checkpoint_id in checkpoints
    }
    actual = {(item["model_family"], item["checkpoint"]) for item in items}

    assert payload["knowledge_type"] == "model_capability"
    assert payload["policy"]["cannot_add_candidates"] is True
    assert payload["policy"]["cannot_affect_final_score"] is True
    assert len(items) == 18
    assert len(actual) == len(items)
    assert actual == expected
    assert all(item["source"] == "model_capability" for item in items)
    assert all(item["keywords"] and item["good_for"] for item in items)
    assert all("select_when" in item["checkpoint_boundary"] for item in items)
    assert all(all(isinstance(value, (int, float)) for value in item["score_bias"].values()) for item in items)


def test_context_retrieval_loads_structured_model_capability_items():
    ContextRetrievalService._load_items.cache_clear()
    service = ContextRetrievalService()
    results = service.retrieve(
        user_request="indoor video illumination retinexformer",
        image_metrics={"noise_estimate": 3, "laplacian_sharpness": 180, "color_cast_index": 0.02},
        available_models=[{"model_id": "retinexformer", "available": True}],
        hardware_summary={"gpu_memory_mb": 8192},
        max_items=20,
    )

    matching = [item for item in results if item.item_id == "model-capability-retinexformer-sdsd-indoor"]
    assert matching
    assert matching[0].source == "model_capability"
    assert {"indoor", "video", "illumination"} & set(matching[0].matched_terms)


def test_checkpoint_capability_knowledge_does_not_adjust_planning_score():
    candidates = [
        CandidatePlanItem(
            candidate_id="candidate_01",
            model_id="hvi_cidnet",
            checkpoint_id="sice",
            model_prior_score=35,
            input_match_score=30,
            planning_score=65,
        ),
        CandidatePlanItem(
            candidate_id="candidate_02",
            model_id="hvi_cidnet",
            checkpoint_id="fivek",
            model_prior_score=35,
            input_match_score=30,
            planning_score=65,
        ),
    ]
    contexts = [
        {
            "item_id": "model-capability-hvi-cidnet-sice",
            "source": "model_capability",
            "title": "HVI-CIDNet / SICE capability",
            "content": "trusted local capability",
            "tags": ["hvi_cidnet", "sice", "exposure", "color", "illumination"],
            "matched_terms": ["hvi_cidnet", "sice", "exposure", "color", "illumination"],
            "score": 5.0,
        }
    ]

    PlanningKnowledgeAdapter().apply(candidates, contexts)

    assert candidates[0].knowledge_adjustment == 0
    assert candidates[0].planning_score == 65
    assert candidates[0].knowledge_evidence == []
    assert candidates[1].knowledge_adjustment == 0
    assert candidates[1].planning_score == 65
